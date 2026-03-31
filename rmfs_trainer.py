"""
RMFS RL Trainer.

RMFSBatchEnv 환경과 GATActorCritic 모델을 사용한다.
REINFORCE와 PPO 알고리즘을 모두 지원한다.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam as Optimizer
from torch.distributions import Categorical

import pickle
import time
from datetime import datetime
import random
import numpy as np
import copy

from rmfs_env_batch import RMFSBatchEnv, RMFSState
from rmfs_model import GATActorCritic
from rmfs_data_generator import generate_rmfs_data_batch, generate_rmfs_validation_batch

import logging
import warnings
import wandb

warnings.filterwarnings('ignore')

WANDB_AVAILABLE = True


def set_random_seed(seed):
    """모든 랜덤 시드를 고정하여 재현 가능한 결과를 보장"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


class RMFS_Trainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # WandB 설정
        self.use_wandb = trainer_params.get('use_wandb', False) and WANDB_AVAILABLE

        self.seed = trainer_params.get('seed', None)

        mode = trainer_params.get('mode', 'train')

        # 체크포인트 폴더 생성
        if mode == 'train':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alg_type = trainer_params.get('algorithm_type', 'reinforce')
            alg_label = {'ppo': 'PPO', 'reinforce': 'REINFORCE'}.get(alg_type, 'REINFORCE')
            self.checkpoint_dir = f"./checkpoints/{timestamp}_RMFS_GAT_{alg_label}"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"Checkpoint dir: {self.checkpoint_dir}")
        else:
            self.checkpoint_dir = None

        # Device 설정
        device_str = trainer_params.get('device', 'cpu')
        self.device = torch.device(device_str)
        print(f"Device: {self.device}")

        # 환경 크기 파라미터
        block_rows = env_params['block_rows']
        block_cols = env_params['block_cols']
        block_h = env_params.get('block_h', 4)
        block_w = env_params.get('block_w', 2)
        self.N_S = block_rows * block_cols * block_h * block_w
        self.N_W = env_params['N_W']
        self.V = self.N_S + self.N_W

        # 모델 초기화
        from types import SimpleNamespace
        model_config = SimpleNamespace(
            N_S=self.N_S,
            N_W=self.N_W,
            storage_feat_dim=model_params.get('storage_feat_dim', 4),
            ws_feat_dim=model_params.get('ws_feat_dim', 4),
            d_edge=model_params.get('d_edge', 9),
            d_hidden=model_params['d_hidden'],
            n_gat_layers=model_params['n_gat_layers'],
            n_heads=model_params['n_heads'],
            dropout_prob=model_params.get('dropout_prob', 0.0),
            num_mlp_layers_actor=model_params['num_mlp_layers_actor'],
            hidden_dim_actor=model_params['hidden_dim_actor'],
            num_mlp_layers_critic=model_params['num_mlp_layers_critic'],
            hidden_dim_critic=model_params['hidden_dim_critic'],
        )
        self.model = GATActorCritic(model_config).to(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])

        print(f"GATActorCritic initialized (N_S={self.N_S}, N_W={self.N_W}, V={self.V})")
        print(f"  d_hidden={model_params['d_hidden']}, n_gat_layers={model_params['n_gat_layers']}, "
              f"n_heads={model_params['n_heads']}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Static adjacency matrix (same for all instances)
        self.adj = self._build_adjacency().to(self.device)

        # Reward 방식 설정
        self.reward_type = trainer_params.get('reward_type', 'stepwise')

        # 알고리즘 타입 설정
        self.algorithm_type = trainer_params.get('algorithm_type', 'reinforce')

        # PPO 전용 초기화
        if self.algorithm_type == 'ppo':
            from copy import deepcopy
            from rmfs_ppo_utils import RMFSPPOMemory
            self.model_old = deepcopy(self.model)
            self.model_old.load_state_dict(self.model.state_dict())
            self.eps_clip = trainer_params.get('eps_clip', 0.2)
            self.k_epochs = trainer_params.get('k_epochs', 4)
            self.gae_lambda = trainer_params.get('gae_lambda', 0.98)
            self.gamma = trainer_params.get('gamma', 1.0)
            self.vloss_coef = trainer_params.get('vloss_coef', 0.5)
            self.ploss_coef = trainer_params.get('ploss_coef', 1.0)
            self.tau = trainer_params.get('tau', 0.0)
            self.ppo_minibatch_size = trainer_params.get('ppo_minibatch_size', 4096)
            self.n_resample = trainer_params.get('n_resample', 20)
            self.ppo_adv_norm_type = trainer_params.get('ppo_adv_norm_type', 'batch')
            self.ppo_memory = RMFSPPOMemory(self.gamma, self.gae_lambda)
            print(f"PPO initialized (eps_clip={self.eps_clip}, k_epochs={self.k_epochs}, "
                  f"gae_lambda={self.gae_lambda}, n_resample={self.n_resample})")

        # Validation 데이터셋 준비
        self.validation_problem = None
        if mode == 'train' and trainer_params.get('use_validation', False):
            self._prepare_validation_dataset(
                trainer_params.get('validation_batch_size', 10)
            )

        # 체크포인트에서 재개
        self.start_epoch = 0
        self.initial_val_score_from_checkpoint = None
        resume_checkpoint = trainer_params.get('resume_from_checkpoint', None)
        resume_training = trainer_params.get('resume_training', False)

        if resume_checkpoint is not None and mode == 'train':
            self._load_checkpoint(resume_checkpoint, resume_training)

    # =================================================================
    # Utility Methods
    # =================================================================

    def _build_adjacency(self):
        """Static adjacency: WS↔Storage (all pairs), WS↔WS, self-loops."""
        V = self.V
        adj = torch.zeros(V, V, dtype=torch.bool)

        # WS→Storage and Storage→WS (all pairs)
        for w in range(self.N_W):
            wi = self.N_S + w
            adj[wi, :self.N_S] = True
            adj[:self.N_S, wi] = True

        # WS↔WS (all pairs, w1≠w2)
        for w1 in range(self.N_W):
            for w2 in range(self.N_W):
                if w1 != w2:
                    adj[self.N_S + w1, self.N_S + w2] = True

        # Self-loops
        idx = torch.arange(V)
        adj[idx, idx] = True

        return adj.unsqueeze(0)  # (1, V, V)

    def _state_to_device(self, state):
        """RMFSState의 모든 텐서를 device로 이동."""
        return RMFSState(
            storage_features=state.storage_features.to(self.device),
            ws_features=state.ws_features.to(self.device),
            edge_feat=state.edge_feat.to(self.device),
            curws_idx=state.curws_idx.to(self.device),
            action_mask=state.action_mask.to(self.device),
        )

    # =================================================================
    # Main Training Loop
    # =================================================================

    def run(self):
        if self.seed is not None:
            set_random_seed(self.seed)

        if self.use_wandb:
            wandb.login(key='b933182253a486cf22871819201b22e9b4b1d581')
            wandb.require('core')
            project_name = self.trainer_params.get('wandb_project', 'RMFS')

            timestamp = datetime.now().strftime("%m%d_%H%M")
            alg_upper = self.algorithm_type.upper()
            batch_size = self.env_params.get('batch_size', 0)
            lr = self.optimizer_params['optimizer']['lr']
            run_name = self.trainer_params.get('wandb_run_name')
            if run_name is None:
                run_name = f"RMFS_GAT_{alg_upper}_bs{batch_size}_lr{lr}_{timestamp}"

            config = {
                "algorithm": alg_upper,
                "model_type": "GATv2_ActorCritic",
                "batch_size": batch_size,
                "N_P": self.env_params.get('N_P'),
                "N_R": self.env_params.get('N_R'),
                "N_W": self.env_params.get('N_W'),
                "N_S": self.N_S,
                "block_rows": self.env_params.get('block_rows'),
                "block_cols": self.env_params.get('block_cols'),
                "block_h": self.env_params.get('block_h'),
                "block_w": self.env_params.get('block_w'),
                "Total_PodTask": self.env_params.get('Total_PodTask'),
                "epochs": self.trainer_params.get('epochs'),
                "learning_rate": lr,
                "reward_type": self.reward_type,
                "d_hidden": self.model_params.get('d_hidden'),
                "n_gat_layers": self.model_params.get('n_gat_layers'),
                "n_heads": self.model_params.get('n_heads'),
            }

            wandb_init_kwargs = {
                "project": project_name,
                "name": run_name,
                "config": config
            }

            wandb_run_id = self.trainer_params.get('wandb_run_id', None)
            wandb_resume = self.trainer_params.get('wandb_resume', 'allow')
            if wandb_run_id is not None:
                wandb_init_kwargs["id"] = wandb_run_id
                wandb_init_kwargs["resume"] = wandb_resume

            wandb.init(**wandb_init_kwargs)

        best_train_score = 1e99
        best_val_score = 1e99
        best_model_epoch = 0
        initial_val_score = self.initial_val_score_from_checkpoint

        # Validation 설정
        use_validation = self.trainer_params.get('use_validation', False)
        validation_interval = self.trainer_params.get('validation_interval', 5)
        validation_batch_size = self.trainer_params.get('validation_batch_size', 10)

        start_epoch = max(0, self.start_epoch)
        end_epoch = self.trainer_params['epochs']

        # 초기 validation (start_epoch == 0인 경우에만)
        if start_epoch == 0:
            print(f"epoch: 0 (initial state)")
            if use_validation:
                val_start_time = time.time()
                val_obj_avg = self._eval_validation(validation_batch_size)
                val_elapsed_time = time.time() - val_start_time
                print(f"  Initial Validation Makespan: {val_obj_avg:.4f}, time: {val_elapsed_time:.2f}s")

                initial_val_score = val_obj_avg
                best_val_score = val_obj_avg
                best_model_epoch = 0

                if self.use_wandb:
                    wandb.log({
                        "val/makespan": val_obj_avg,
                        "val/best_makespan": best_val_score,
                        "val/improvement_pct": 0.0,
                    }, step=0)

                # Epoch 0 체크포인트 저장
                if self.checkpoint_dir is not None:
                    ckpt_path = os.path.join(self.checkpoint_dir, "epoch0.pt")
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'env_params': self.env_params,
                        'model_params': self.model_params,
                        'trainer_params': self.trainer_params,
                        'epoch': 0,
                        'train_score': None,
                        'val_score': val_obj_avg,
                        'initial_val_score': initial_val_score,
                        'algorithm_type': self.algorithm_type,
                        'model_old_state_dict': self.model_old.state_dict() if self.algorithm_type == 'ppo' else None,
                    }, ckpt_path)
                    print(f"  Checkpoint saved: {ckpt_path}")

            start_epoch = 1

        # PPO: N_r epoch마다 새 학습 데이터 생성
        ppo_problem = None
        self._epoch_counter = start_epoch

        try:
            for epoch in range(start_epoch, end_epoch + 1):
                epoch_start_time = time.time()
                self._epoch_counter = epoch

                print(f"epoch: {epoch}")
                if self.algorithm_type == 'ppo':
                    if ppo_problem is None or (epoch - start_epoch) % self.n_resample == 0:
                        ppo_problem = generate_rmfs_data_batch(self.env_params, epoch=epoch)
                    train_loss_avg, train_reward_avg = self._train_ppo_one_batch(ppo_problem)
                else:
                    train_loss_avg, train_reward_avg = self._train_one_batch()

                train_obj_avg = -train_reward_avg
                epoch_elapsed_time = time.time() - epoch_start_time
                print(f"epoch: {epoch}, loss: {train_loss_avg:.4f}, makespan: {train_obj_avg:.4f}, "
                      f"time: {epoch_elapsed_time:.2f}s")

                if train_obj_avg < best_train_score:
                    best_train_score = train_obj_avg

                wandb_log_dict = {
                    "train/loss": train_loss_avg,
                    "train/makespan": train_obj_avg,
                    "train/best_makespan": best_train_score,
                    "epoch_time": epoch_elapsed_time
                }

                # Validation
                val_obj_avg = None
                if use_validation and epoch % validation_interval == 0:
                    val_start_time = time.time()
                    val_obj_avg = self._eval_validation(validation_batch_size)
                    val_elapsed_time = time.time() - val_start_time

                    if initial_val_score is not None and initial_val_score > 0:
                        val_improvement_pct = ((initial_val_score - val_obj_avg) / initial_val_score) * 100
                    else:
                        val_improvement_pct = 0.0

                    print(f"  Validation Makespan: {val_obj_avg:.4f} ({val_improvement_pct:+.2f}%), "
                          f"time: {val_elapsed_time:.2f}s")

                    if val_obj_avg < best_val_score:
                        best_val_score = val_obj_avg
                        best_model_epoch = epoch
                        # Best model 저장
                        if self.checkpoint_dir is not None:
                            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                            torch.save({
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'env_params': self.env_params,
                                'model_params': self.model_params,
                                'trainer_params': self.trainer_params,
                                'epoch': epoch,
                                'val_score': val_obj_avg,
                                'train_score': train_obj_avg,
                                'initial_val_score': initial_val_score,
                                'algorithm_type': self.algorithm_type,
                                'model_old_state_dict': self.model_old.state_dict() if self.algorithm_type == 'ppo' else None,
                            }, best_model_path)
                            print(f"  Best Model saved: {best_model_path} (Makespan: {val_obj_avg:.4f})")

                    wandb_log_dict["val/makespan"] = val_obj_avg
                    wandb_log_dict["val/best_makespan"] = best_val_score
                    wandb_log_dict["val/improvement_pct"] = val_improvement_pct

                if self.use_wandb:
                    wandb.log(wandb_log_dict, step=epoch)

                # 주기적 체크포인트 저장
                if epoch % 5 == 0 and self.checkpoint_dir is not None:
                    ckpt_path = os.path.join(self.checkpoint_dir, f"epoch{epoch}.pt")
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'env_params': self.env_params,
                        'model_params': self.model_params,
                        'trainer_params': self.trainer_params,
                        'epoch': epoch,
                        'train_score': train_obj_avg,
                        'val_score': val_obj_avg,
                        'initial_val_score': initial_val_score,
                        'algorithm_type': self.algorithm_type,
                        'model_old_state_dict': self.model_old.state_dict() if self.algorithm_type == 'ppo' else None,
                    }, ckpt_path)
                    print(f"  Checkpoint saved: {ckpt_path}")

        except KeyboardInterrupt:
            print(f"\nCtrl+C detected at epoch {epoch}. Stopping training.")

        if use_validation and best_model_epoch > 0:
            print(f"\nBest Model: Epoch {best_model_epoch}, Val Makespan: {best_val_score:.4f}")
        else:
            print(f"\nBest Train Makespan: {best_train_score:.4f}")

        if self.use_wandb:
            wandb.finish()

    # =================================================================
    # REINFORCE Training
    # =================================================================

    def _train_one_batch(self):
        """REINFORCE with batch baseline."""
        self.model.train()
        self.model.zero_grad()
        total_loss = 0
        total_reward = 0

        accumulation_steps = self.trainer_params.get('accumulation_steps', 1)

        for i in range(accumulation_steps):
            loss, obj_value = self._train_one_minibatch()
            loss.backward()
            total_loss += loss.item()
            total_reward += obj_value

        grad_clip_norm = self.trainer_params.get('grad_clip_norm', None)
        if grad_clip_norm is not None:
            clip_grad_norm_(self.model.parameters(), grad_clip_norm)

        self.optimizer.step()
        self.model.zero_grad()

        avg_loss = total_loss / accumulation_steps
        avg_reward = total_reward / accumulation_steps

        return avg_loss, avg_reward

    def _train_one_minibatch(self):
        """REINFORCE 1 minibatch: rollout + loss 계산."""
        batch_size = self.env_params['batch_size']
        entropy_coef = self.trainer_params.get('entropy_coef', 0.0)
        use_entropy = entropy_coef > 0

        # 환경 초기화
        env = RMFSBatchEnv(self.env_params, device='cpu')
        problem = generate_rmfs_data_batch(self.env_params, epoch=self._epoch_counter)
        state = env.reset(problem)

        log_prob_sum = torch.zeros(batch_size, device=self.device)
        cumulative_reward = torch.zeros(batch_size, device=self.device)
        if use_entropy:
            cumulative_entropy = torch.zeros(batch_size, device=self.device)

        all_done = False
        step_count = 0

        while not all_done:
            step_count += 1

            state_dev = self._state_to_device(state)
            pi, v = self.model(state_dev, self.adj)

            dist = Categorical(pi)
            action = dist.sample()         # (B,)
            log_prob = dist.log_prob(action)  # (B,)

            # 활성 인스턴스만 누적
            active_mask = env.get_active_mask().float().to(self.device)
            log_prob_sum += log_prob * active_mask

            if use_entropy:
                cumulative_entropy += dist.entropy() * active_mask

            state, rewards, all_done = env.step(action.cpu())
            cumulative_reward += rewards.to(self.device) * active_mask

        # Reward: stepwise이면 누적된 step reward, sparse이면 -makespan
        if self.reward_type == 'stepwise':
            reward = cumulative_reward
        else:
            reward = -env.get_makespan().to(self.device)

        # Advantage 계산 (batch baseline)
        baseline_type = self.trainer_params.get('baseline_type', 'batch')
        normalize_advantage = self.trainer_params.get('normalize_advantage', True)

        if baseline_type == 'batch':
            advantage = reward - reward.mean()
            if normalize_advantage:
                advantage = advantage / (advantage.std() + 1e-8)
        else:
            advantage = reward

        # Policy loss
        policy_loss = -advantage.detach() * log_prob_sum

        if use_entropy:
            entropy_bonus = entropy_coef * cumulative_entropy
            loss = policy_loss.mean() - entropy_bonus.mean()
        else:
            loss = policy_loss.mean()

        # 목적함수값 (makespan 평균)
        obj_value = env.get_makespan().mean().item()

        return loss, -obj_value

    # =================================================================
    # PPO Training
    # =================================================================

    def _train_ppo_one_batch(self, problem):
        """PPO-Clip + GAE for RMFS with graph state."""
        from rmfs_ppo_utils import eval_actions
        import math

        batch_size = self.env_params['batch_size']
        entropy_coef = self.trainer_params.get('entropy_coef', 0.0)
        grad_clip_norm = self.trainer_params.get('grad_clip_norm', None)

        # 1. Hard copy: policy_old ← policy
        self.model_old.load_state_dict(self.model.state_dict())

        # 2. Rollout (no_grad)
        memory = self.ppo_memory
        memory.clear_memory()

        env = RMFSBatchEnv(self.env_params, device='cpu')
        state = env.reset(problem)
        all_done = False
        step_count = 0

        self.model.eval()
        with torch.no_grad():
            while not all_done:
                step_count += 1

                state_dev = self._state_to_device(state)
                pi, v = self.model(state_dev, self.adj)

                dist = Categorical(pi)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # 활성 마스크 (step 전)
                active_mask = env.get_active_mask().to(self.device)

                # State 저장 (step 전, CPU state)
                memory.push_state(state)

                # Step
                state, rewards, all_done = env.step(action.cpu())

                # Reward 처리
                if self.reward_type == 'stepwise':
                    reward = rewards.to(self.device)
                else:
                    if all_done:
                        reward = -env.get_makespan().to(self.device)
                    else:
                        reward = torch.zeros(batch_size, device=self.device)

                done_tensor = env.done_mask.to(self.device)

                memory.push_transition(
                    action.to(self.device),
                    log_prob.to(self.device),
                    v.squeeze(-1).to(self.device),
                    reward,
                    done_tensor,
                    active_mask,
                )

        # Final objective
        final_obj = env.get_makespan().mean().item()

        # 3. GAE Advantages
        t_data = memory.transpose_data()
        # t_data indices:
        #   0: storage_features (B*T, N_S, 4)
        #   1: ws_features      (B*T, N_W, 4)
        #   2: edge_feat        (B*T, V, V, 9)
        #   3: curws_idx        (B*T,)
        #   4: action_mask      (B*T, N_S+1)
        #   5: actions          (B*T,)
        #   6: rewards          (B*T,)
        #   7: values           (B*T,)
        #   8: dones            (B*T,)
        #   9: log_probs        (B*T,)
        #  10: masks            (B*T,)
        t_data = tuple(t.to(self.device) for t in t_data)
        t_advantage, v_target = memory.get_gae_advantages()
        t_advantage = t_advantage.to(self.device)
        v_target = v_target.to(self.device)

        # Advantage normalization
        if self.ppo_adv_norm_type != 'per_instance':
            t_advantage = (t_advantage - t_advantage.mean()) / (t_advantage.std() + 1e-8)

        full_batch_size = t_data[5].shape[0]  # actions: (B*T,)
        num_mini = math.ceil(full_batch_size / self.ppo_minibatch_size)

        # 4. K-epoch PPO Updates
        self.model.train()
        total_loss = 0.0
        update_count = 0

        for _ in range(self.k_epochs):
            perm = torch.randperm(full_batch_size, device=self.device)

            for i in range(num_mini):
                start = i * self.ppo_minibatch_size
                end = min(start + self.ppo_minibatch_size, full_batch_size)
                idx = perm[start:end]

                # Construct RMFSState mini-batch
                mini_state = RMFSState(
                    storage_features=t_data[0][idx],
                    ws_features=t_data[1][idx],
                    edge_feat=t_data[2][idx],
                    curws_idx=t_data[3][idx],
                    action_mask=t_data[4][idx],
                )

                # Forward through CURRENT policy
                pi_new, v_new = self.model(mini_state, self.adj)

                actions_batch = t_data[5][idx]
                old_logprobs = t_data[9][idx]
                adv_batch = t_advantage[idx]
                v_target_batch = v_target[idx]

                # Evaluate actions with new policy
                new_logprobs, ent = eval_actions(pi_new, actions_batch)

                # PPO ratio and clipped surrogate loss
                ratios = torch.exp(new_logprobs - old_logprobs.detach())
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                p_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                v_loss = F.mse_loss(v_new.squeeze(-1), v_target_batch)

                # Entropy bonus
                ent_loss = -ent

                loss = self.ploss_coef * p_loss + self.vloss_coef * v_loss + entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()

                total_loss += loss.item()
                update_count += 1

        # 5. Soft update policy_old (if tau > 0)
        if self.tau > 0:
            for old_p, new_p in zip(self.model_old.parameters(), self.model.parameters()):
                old_p.data.copy_(self.tau * old_p.data + (1.0 - self.tau) * new_p.data)

        avg_loss = total_loss / max(update_count, 1)
        avg_reward = -final_obj

        return avg_loss, avg_reward

    # =================================================================
    # Validation
    # =================================================================

    def _eval_validation(self, validation_batch_size=10):
        """Greedy rollout on validation set."""
        self.model.eval()

        # Validation 데이터 로드
        if self.validation_problem is not None:
            problem = self.validation_problem
        else:
            val_file_path = os.path.join('./data/rmfs_val', 'val_batch.pickle')
            with open(val_file_path, 'rb') as f:
                problem = pickle.load(f)
            self.validation_problem = problem

        # 환경 초기화 (validation batch size)
        val_env_params = copy.deepcopy(self.env_params)
        val_env_params['batch_size'] = validation_batch_size

        val_env = RMFSBatchEnv(val_env_params, device='cpu')
        state = val_env.reset(problem)
        all_done = False

        with torch.no_grad():
            while not all_done:
                state_dev = self._state_to_device(state)
                pi, v = self.model(state_dev, self.adj)
                action = torch.argmax(pi, dim=-1)  # Greedy
                state, rewards, all_done = val_env.step(action.cpu())

        makespans = val_env.get_makespan()
        avg_makespan = makespans.mean().item()

        self.model.train()
        return avg_makespan

    # =================================================================
    # Validation Dataset Preparation
    # =================================================================

    def _prepare_validation_dataset(self, validation_batch_size=10):
        """고정 시드로 validation 데이터셋 생성 또는 로드."""
        val_data_folder = './data/rmfs_val'
        os.makedirs(val_data_folder, exist_ok=True)
        batch_file_path = os.path.join(val_data_folder, 'val_batch.pickle')

        if os.path.exists(batch_file_path):
            with open(batch_file_path, 'rb') as f:
                self.validation_problem = pickle.load(f)
            print(f"Validation data loaded: {batch_file_path}")
            return

        print(f"Generating validation dataset (instances: {validation_batch_size})...")

        problem = generate_rmfs_validation_batch(
            self.env_params, validation_batch_size, seed=2025
        )

        with open(batch_file_path, 'wb') as f:
            pickle.dump(problem, f)

        self.validation_problem = problem
        print(f"Validation data saved: {batch_file_path}")

    # =================================================================
    # Checkpoint Load
    # =================================================================

    def _load_checkpoint(self, checkpoint_path, resume_training=False):
        """체크포인트에서 모델 가중치 및 학습 상태 로드."""
        print(f"Loading checkpoint: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model weights loaded")

            if self.algorithm_type == 'ppo' and 'model_old_state_dict' in checkpoint and checkpoint['model_old_state_dict'] is not None:
                self.model_old.load_state_dict(checkpoint['model_old_state_dict'])
                print(f"PPO policy_old weights loaded")

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Optimizer state loaded")

            if resume_training and 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {self.start_epoch}")
            else:
                self.start_epoch = 0
                print(f"Weights loaded, starting from epoch 0 (fine-tuning)")

            if 'train_score' in checkpoint and checkpoint['train_score'] is not None:
                print(f"  Saved Train Score: {checkpoint['train_score']:.4f}")
            if 'val_score' in checkpoint and checkpoint['val_score'] is not None:
                print(f"  Saved Val Score: {checkpoint['val_score']:.4f}")
            if 'initial_val_score' in checkpoint:
                self.initial_val_score_from_checkpoint = checkpoint['initial_val_score']

        except FileNotFoundError:
            print(f"Checkpoint not found: {checkpoint_path}")
            self.start_epoch = 0
        except Exception as e:
            print(f"Checkpoint load failed: {e}")
            self.start_epoch = 0
