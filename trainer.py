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
import matplotlib.pyplot as plt 
import random
import numpy as np 
import copy

# 환경 import (완료)
from scheduling_env import SchedulingEnv
from data_generator import generate_scheduling_data_batch

# GNN 모델 import
from gnn_model import SchedulingModel

# DANIEL 모델 import
from model.main_model import DANIEL

import logging
import warnings
import wandb

warnings.filterwarnings('ignore')

WANDB_AVAILABLE = True  # wandb 사용 가능 여부

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

# =================================================================

class Scheduling_Trainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments 
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # WandB 설정
        self.use_wandb = trainer_params.get('use_wandb', False) and WANDB_AVAILABLE
        if trainer_params.get('use_wandb', False) and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Disabling wandb logging.")
            
        self.seed = trainer_params.get('seed', None)

        # Main Components
        mode = trainer_params.get('mode', 'train')
        
        # 날짜/시간 기반 체크포인트 폴더 생성 (train 모드일 때만)
        if mode == 'train':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            objective = env_params.get('objective', 'tardiness')
            model_type = trainer_params.get('model_type', 'gat').upper()
            alg_label = 'PPO' if trainer_params.get('algorithm_type', 'reinforce') == 'ppo' else 'REINFORCE'
            self.checkpoint_dir = f"./checkpoints/{timestamp}_{objective}_{model_type}_{alg_label}"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"✅ 체크포인트 저장 폴더 생성: {self.checkpoint_dir}")
        else:
            # test 모드에서는 폴더 생성하지 않음
            self.checkpoint_dir = None
        
        self.debug_env = trainer_params.get('debug_env', False)
        self.debug_model = trainer_params.get('debug_model', False)
        self.debug_instance_info = trainer_params.get('debug_instance_info', False)  # 배치 내 인스턴스별 정보 출력
        self.verbose_logging = trainer_params.get('verbose_logging', False)  # 각 batch/POMO별 상세 출력
        
        # 목적함수 설정
        self.objective = env_params.get('objective', 'tardiness')

        # Reward 방식 설정
        self.reward_type = trainer_params.get('reward_type', 'sparse')

        # Device 설정
        device_str = trainer_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        self.device_mode = trainer_params.get('device_mode', 'cpu')
        
        # Hybrid 모드: 모델은 GPU, 환경은 CPU
        # GPU 모드: 모델과 환경 모두 GPU
        # CPU 모드: 모델과 환경 모두 CPU
        if self.device_mode == 'hybrid':
            print(f"Device: {self.device} (Model on GPU, Environment on CPU)")
        else:
            print(f"Device: {self.device}")
        print(f"Debug ENV: {self.debug_env}, Debug Model: {self.debug_model}")
        
        # 모델 타입 설정
        self.model_type = trainer_params.get('model_type', 'gat')
        
        # 모델 초기화
        if self.model_type == 'daniel':
            # DANIEL 모델: RCMPSP 파라미터 이름 → DANIEL 내부 이름 매핑
            from types import SimpleNamespace
            daniel_config = SimpleNamespace(
                device=str(self.device),
                # RCMPSP 이름(train.py) → DANIEL 원본 이름(model/main_model.py)
                fea_j_input_dim=self.model_params['fea_act_input_dim'],   # activity feature dim
                fea_m_input_dim=self.model_params['fea_team_input_dim'],  # team feature dim
                num_heads_OAB=self.model_params['num_heads_AAB'],         # Activity Attention Block
                num_heads_MAB=self.model_params['num_heads_TAB'],         # Team Attention Block
                layer_fea_output_dim=self.model_params['layer_fea_output_dim'],
                dropout_prob=self.model_params['dropout_prob'],
                num_mlp_layers_actor=self.model_params['num_mlp_layers_actor'],
                hidden_dim_actor=self.model_params['hidden_dim_actor'],
                num_mlp_layers_critic=self.model_params['num_mlp_layers_critic'],
                hidden_dim_critic=self.model_params['hidden_dim_critic'],
            )
            self.model = DANIEL(daniel_config).to(self.device)
            self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
            
            print("✅ DANIEL 모델 초기화 완료")
            print(f"   모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        else:
            # GNN (GAT) 모델
            model_params_with_env = copy.deepcopy(self.model_params)
            model_params_with_env['N_T'] = env_params['N_T']
            model_params_with_env['N_P'] = env_params['N_P']
            
            self.model = SchedulingModel(model_params_with_env, debug_model=self.debug_model).to(self.device)
            self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
            
            print("✅ GNN 모델 초기화 완료")
            print(f"   모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")

        self.result_train_loss_log = []

        # 알고리즘 타입 설정
        self.algorithm_type = trainer_params.get('algorithm_type', 'reinforce')

        # PPO 전용 초기화 (DANIEL 모델만 지원)
        if self.algorithm_type == 'ppo':
            if self.model_type != 'daniel':
                raise ValueError("PPO는 DANIEL 모델에서만 지원됩니다. MODEL_TYPE='daniel'로 설정하세요.")
            from copy import deepcopy
            from ppo_utils import PPOMemory
            self.model_old = deepcopy(self.model)
            self.model_old.load_state_dict(self.model.state_dict())
            self.eps_clip           = trainer_params.get('eps_clip', 0.2)
            self.k_epochs           = trainer_params.get('k_epochs', 4)
            self.gae_lambda         = trainer_params.get('gae_lambda', 0.98)
            self.gamma              = trainer_params.get('gamma', 1.0)
            self.vloss_coef         = trainer_params.get('vloss_coef', 0.5)
            self.ploss_coef         = trainer_params.get('ploss_coef', 1.0)
            self.tau                = trainer_params.get('tau', 0.0)
            self.ppo_minibatch_size = trainer_params.get('ppo_minibatch_size', 32)
            self.n_resample         = trainer_params.get('n_resample', 20)
            self.ppo_memory         = PPOMemory(self.gamma, self.gae_lambda)
            print(f"✅ PPO 초기화 완료 (eps_clip={self.eps_clip}, k_epochs={self.k_epochs}, "
                  f"gae_lambda={self.gae_lambda}, n_resample={self.n_resample})")

        # Validation 데이터셋 미리 생성 (고정된 validation set)
        self.validation_problem = None
        if mode == 'train' and trainer_params.get('use_validation', False):
            self._prepare_validation_dataset(
                trainer_params.get('validation_batch_size', 10),
                trainer_params.get('validation_pomo_size', 1)
            )
        
        # 체크포인트에서 재개 (옵션)
        self.start_epoch = 0
        self.initial_val_score_from_checkpoint = None
        resume_checkpoint = trainer_params.get('resume_from_checkpoint', None)
        resume_training = trainer_params.get('resume_training', False)
        
        if resume_checkpoint is not None and mode == 'train':
            self._load_checkpoint(resume_checkpoint, resume_training)



    def run(self):
        # 시드 고정 (시드가 지정된 경우)
        if self.seed is not None:
            set_random_seed(self.seed)
            
        if self.use_wandb:
            wandb.login(key='b933182253a486cf22871819201b22e9b4b1d581')
            wandb.require('core')
            project_name = self.trainer_params.get('wandb_project', 'RCMPSP')
            
            # Run 이름 생성
            objective = self.env_params.get('objective', 'tardiness')
            n_p = self.env_params.get('N_P', 0)
            n_t = self.env_params.get('N_T', 0)
            batch_size = self.env_params.get('batch_size', 0)
            pomo_size = self.env_params.get('pomo_size', 0)
            learning_rate = self.optimizer_params['optimizer']['lr']
            normalize_advantage = self.trainer_params.get('normalize_advantage', False)
            adv_norm_str = 'advnorm' if normalize_advantage else 'no_advnorm'
            timestamp = datetime.now().strftime("%m%d_%H%M")
            alg_upper = self.algorithm_type.upper()
            run_name = self.trainer_params.get('wandb_run_name')
            if run_name is None:
                run_name = f"{objective}_{alg_upper}_P{n_p}_T{n_t}_bs{batch_size}_p{pomo_size}_lr{learning_rate}_{adv_norm_str}_{timestamp}"

            # Config 설정
            config = {
                "objective": objective,
                "algorithm": alg_upper,
                "model_type": self.model_type,
                "N_P": self.env_params.get('N_P'),
                "N_A_min": self.env_params.get('N_A_min'),
                "N_A_max": self.env_params.get('N_A_max'),
                "N_T": self.env_params.get('N_T'),
                "batch_size": batch_size,
                "pomo_size": pomo_size,
                "epochs": self.trainer_params.get('epochs'),
                "learning_rate": learning_rate,
                "normalize_advantage": normalize_advantage,
                "entropy_coef": self.trainer_params.get('entropy_coef', 0.0),
                "reward_type": self.reward_type,
            }
            # 모델별 파라미터 추가
            if self.model_type == 'daniel':
                config.update({
                    "fea_act_input_dim": self.model_params.get('fea_act_input_dim'),
                    "fea_team_input_dim": self.model_params.get('fea_team_input_dim'),
                    "layer_fea_output_dim": self.model_params.get('layer_fea_output_dim'),
                })
            else:
                config.update({
                    "embedding_dim": self.model_params.get('embedding_dim'),
                    "num_head": self.model_params.get('num_head'),
                    "num_encoder_layer": self.model_params.get('num_encoder_layer'),
                })
            
            # Wandb init 파라미터 구성
            wandb_init_kwargs = {
                "project": project_name,
                "name": run_name,
                "config": config
            }
            
            # Resume 설정 (기존 run에 이어붙이기)
            wandb_run_id = self.trainer_params.get('wandb_run_id', None)
            wandb_resume = self.trainer_params.get('wandb_resume', 'allow')
            
            if wandb_run_id is not None:
                wandb_init_kwargs["id"] = wandb_run_id
                wandb_init_kwargs["resume"] = wandb_resume
                print(f"📌 WandB Resume: run_id={wandb_run_id}, resume={wandb_resume}")
            
            wandb.init(**wandb_init_kwargs)
            
        best_train_score = 1e99
        best_val_score = 1e99
        best_model_epoch = 0
        # 초기 validation score (체크포인트에서 로드했으면 그 값 사용)
        initial_val_score = self.initial_val_score_from_checkpoint if hasattr(self, 'initial_val_score_from_checkpoint') else None
        objective_name = self.env_params.get('objective', 'makespan')
        
        # Validation 설정
        use_validation = self.trainer_params.get('use_validation', False)
        validation_interval = self.trainer_params.get('validation_interval', 5)
        validation_batch_size = self.trainer_params.get('validation_batch_size', 10)
        validation_pomo_size = self.trainer_params.get('validation_pomo_size', 1)
        
        # 시작 epoch 설정 (체크포인트에서 재개하는 경우 start_epoch > 0)
        start_epoch = max(0, self.start_epoch)
        end_epoch = self.trainer_params['epochs']
        
        # 초기 validation (start_epoch == 0인 경우에만)
        if start_epoch == 0:
            print(f"epoch: 0 (초기 상태 - 학습 전)")
            if use_validation:
                print(f"  🔍 초기 Validation 실행 중... (학습 전)")
                val_start_time = time.time()
                val_obj_avg = self._eval_validation(validation_batch_size, validation_pomo_size)
                val_elapsed_time = time.time() - val_start_time
                print(f"  ✅ Initial Validation {objective_name}: {val_obj_avg:.4f}, time: {val_elapsed_time:.2f}s")
                
                # 초기 validation score 저장 (개선율 계산용)
                initial_val_score = val_obj_avg
                
                # 초기 validation score를 best로 설정
                best_val_score = val_obj_avg
                best_model_epoch = 0
                
                # WandB 로깅 (Initial Validation)
                if self.use_wandb:
                    wandb.log({
                        "val/" + objective_name: val_obj_avg,
                        "val/best_" + objective_name: best_val_score,
                        "val/improvement_pct": 0.0,  # 초기 상태는 0% 개선
                        "val/best_improvement_pct": 0.0
                    }, step=0)
                
                # Epoch 0 체크포인트 저장
                if self.checkpoint_dir is not None:
                    ckpt_path = os.path.join(self.checkpoint_dir, f"epoch0.pt")
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
                    print(f"  💾 체크포인트 저장: {ckpt_path}")
                    
                    if self.use_wandb:
                        artifact = wandb.Artifact(
                            name=f"model_epoch_0",
                            type="model"
                        )
                        artifact.add_file(ckpt_path)
                        wandb.log_artifact(artifact)
            start_epoch = 1  # 다음부터는 epoch 1부터 시작
        
        # PPO: N_r 에피소드마다 새 학습 데이터 생성
        ppo_problem = None

        for epoch in range(start_epoch, end_epoch+1):
            epoch_start_time = time.time()

            # 학습 수행
            print(f"epoch: {epoch}")
            if self.algorithm_type == 'ppo':
                # N_r 에피소드마다 새 문제 데이터 생성
                if ppo_problem is None or (epoch - start_epoch) % self.n_resample == 0:
                    ppo_problem = generate_scheduling_data_batch(self.env_params)
                train_loss_avg, train_reward_avg = self._train_ppo_one_batch(ppo_problem)
            else:
                train_loss_avg, train_reward_avg = self._train_one_batch()
            train_obj_avg = -train_reward_avg  # Reward는 -objective이므로 부호 반전
            epoch_elapsed_time = time.time() - epoch_start_time
            print(f"epoch: {epoch}, loss: {train_loss_avg:.4f}, reward: {train_reward_avg:.4f}, {objective_name}: {train_obj_avg:.4f}, time: {epoch_elapsed_time:.2f}s")
            
            # Train score 업데이트
            if train_obj_avg < best_train_score:
                best_train_score = train_obj_avg
            
            # WandB 로깅 (Train)
            wandb_log_dict = {
                "train/loss": train_loss_avg, 
                "train/" + objective_name: train_obj_avg,
                "train/best_" + objective_name: best_train_score,
                "epoch_time": epoch_elapsed_time
            }
            
            # Validation 실행 (validation_interval 마다)
            val_obj_avg = None
            if use_validation and epoch % validation_interval == 0:
                print(f"  🔍 Validation 실행 중...")
                val_start_time = time.time()
                val_obj_avg = self._eval_validation(validation_batch_size, validation_pomo_size)
                val_elapsed_time = time.time() - val_start_time
                
                # 개선율 계산 (초기값 대비)
                if initial_val_score is not None and initial_val_score > 0:
                    val_improvement_pct = ((initial_val_score - val_obj_avg) / initial_val_score) * 100
                    best_improvement_pct = ((initial_val_score - best_val_score) / initial_val_score) * 100
                else:
                    val_improvement_pct = 0.0
                    best_improvement_pct = 0.0
                
                print(f"  ✅ Validation {objective_name}: {val_obj_avg:.4f} ({val_improvement_pct:+.2f}%), time: {val_elapsed_time:.2f}s")
                
                # Validation score 업데이트
                if val_obj_avg < best_val_score:
                    best_val_score = val_obj_avg
                    best_model_epoch = epoch
                    best_improvement_pct = ((initial_val_score - best_val_score) / initial_val_score) * 100 if initial_val_score is not None and initial_val_score > 0 else 0.0
                    # Best model 저장
                    if self.checkpoint_dir is not None:
                        best_model_path = os.path.join(self.checkpoint_dir, f"best_model.pt")
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'env_params': self.env_params,
                            'model_params': self.model_params,
                            'trainer_params': self.trainer_params,
                            'epoch': epoch,
                            'val_score': val_obj_avg,
                            'train_score': train_obj_avg,
                            'initial_val_score': initial_val_score,  # 개선율 계산용
                            'algorithm_type': self.algorithm_type,
                            'model_old_state_dict': self.model_old.state_dict() if self.algorithm_type == 'ppo' else None,
                        }, best_model_path)
                        print(f"  🏆 Best Model 저장: {best_model_path} (Val {objective_name}: {val_obj_avg:.4f})")
                
                # WandB 로깅 (Validation)
                wandb_log_dict["val/" + objective_name] = val_obj_avg
                wandb_log_dict["val/best_" + objective_name] = best_val_score
                wandb_log_dict["val/improvement_pct"] = val_improvement_pct
                wandb_log_dict["val/best_improvement_pct"] = best_improvement_pct
            
            # WandB 로깅
            if self.use_wandb:
                wandb.log(wandb_log_dict, step=epoch)
                
            # 체크포인트 저장 (주기적으로)
            if epoch % 5 == 0 and self.checkpoint_dir is not None:
                ckpt_path = os.path.join(self.checkpoint_dir, f"epoch{epoch}.pt")
                # 모델 가중치와 함께 env_params, model_params, trainer_params, optimizer 상태도 저장
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'env_params': self.env_params,
                    'model_params': self.model_params,
                    'trainer_params': self.trainer_params,
                    'epoch': epoch,
                    'train_score': train_obj_avg,
                    'val_score': val_obj_avg if val_obj_avg is not None else None,
                    'initial_val_score': initial_val_score,  # 개선율 계산용
                    'algorithm_type': self.algorithm_type,
                    'model_old_state_dict': self.model_old.state_dict() if self.algorithm_type == 'ppo' else None,
                }, ckpt_path)
                print(f"  💾 체크포인트 저장: {ckpt_path}")
                
                if self.use_wandb:
                    artifact = wandb.Artifact(
                        name=f"model_epoch_{epoch}",
                        type="model"
                    )
                    artifact.add_file(ckpt_path)
                    wandb.log_artifact(artifact)
        
        # 최종 Best Model 정보 출력
        if use_validation and best_model_epoch > 0:
            print(f"\n🏆 Best Model: Epoch {best_model_epoch}, Val {objective_name}: {best_val_score:.4f}")
        else:
            print(f"\n🏆 Best Train Score: {best_train_score:.4f}")

        # wandb 종료
        if self.use_wandb:
            wandb.finish()
        
        # # plot score
        # self.plot_score(self.result_valid_score_log, self.result_valid_loss_log)

    def _train_one_batch(self):
        self.model.train()
        self.model.zero_grad()  # 그래디언트 초기화
        total_loss = 0  # 전체 손실 초기화
        total_reward = 0  # 전체 보상 초기화

        for i in range(self.trainer_params['accumulation_steps']):
            if self.verbose_logging:
                print(f"accumulation_steps: {i}")
            
            loss, obj_value, step_count, batch_info = self.train_one_minibatch()
            
            if self.verbose_logging:
                print(f"  평균 목적함수값: {obj_value:.4f}, Loss: {loss.item():.4f}")
            
            loss.backward()  # REINFORCE는 여기서 backward
            total_loss += loss.item()
            
            # 배치 내 각 인스턴스별 정보 출력 (옵션)
            if self.debug_instance_info:
                rewards = batch_info['rewards']
                num_activities_list = batch_info['num_activities']
                done_status = batch_info['done_status']
                for b in range(self.env_params['batch_size']):
                    for p in range(self.env_params['pomo_size']):
                        instance_idx = b * self.env_params['pomo_size'] + p
                        reward_val = rewards[b, p].item()
                        n_activities = num_activities_list[instance_idx].item()
                        is_done = done_status[instance_idx].item()
                        done_mark = "✅" if is_done else "❌"
                        print(f"    Instance{b}_POMO{p}: Reward={reward_val:.4f}, Activities={n_activities}, Steps={step_count}, Done={done_mark}")
            
            total_reward += obj_value  # 보상 누적
        
        # Gradient Clipping (옵션)
        grad_clip_norm = self.trainer_params.get('grad_clip_norm', None)
        if grad_clip_norm is not None:
            clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            
        self.optimizer.step()  # 옵티마이저 업데이트
        self.model.zero_grad()  # 그래디언트 초기화

        avg_loss = total_loss / self.trainer_params['accumulation_steps']
        avg_reward = total_reward / self.trainer_params['accumulation_steps']
        
        return avg_loss, avg_reward

    # =================================================================
    # PPO Training
    # =================================================================

    def _train_ppo_one_batch(self, problem):
        """
        PPO-Clip 알고리즘을 사용한 학습 (1 에피소드 롤아웃 + K-epoch 업데이트).

        1. policy_old ← 현재 policy (hard copy)
        2. No-grad 롤아웃: 상태/행동/가치/보상 수집 → PPOMemory
        3. GAE advantage 계산
        4. K epoch × mini-batch PPO 업데이트 (clip + value loss + entropy)
        5. tau > 0이면 policy_old soft update

        Args:
            problem: generate_scheduling_data_batch()로 생성된 문제 배치

        Returns:
            (avg_loss, avg_reward)  — avg_reward = -avg_obj_value
        """
        from ppo_utils import eval_actions
        import math

        batch_total = self.env_params['batch_size'] * self.env_params['pomo_size']
        entropy_coef = self.trainer_params.get('entropy_coef', 0.0)
        grad_clip_norm = self.trainer_params.get('grad_clip_norm', None)

        # 환경 device 결정
        env_device = self.device if self.device_mode == 'gpu' else 'cpu'

        # --------------------------------------------------
        # 1. Hard-copy: policy_old ← policy (before rollout)
        # --------------------------------------------------
        self.model_old.load_state_dict(self.model.state_dict())

        # --------------------------------------------------
        # 2. Rollout (no_grad)
        # --------------------------------------------------
        memory = self.ppo_memory
        memory.clear_memory()

        env = SchedulingEnv(self.env_params, debug_env=self.debug_env, device=env_device)
        env._reset(problem)

        s = env._get_state()
        done = False
        step_count = 0

        self.model.eval()
        with torch.no_grad():
            while not done:
                step_count += 1

                # State tensors → model device
                fea_act  = s.fea_act_tensor.to(self.device)
                act_mask = s.act_mask_tensor.to(self.device)
                candidate = s.candidate_tensor.to(self.device)
                fea_team  = s.fea_team_tensor.to(self.device)
                team_mask = s.team_mask_tensor.to(self.device)
                comp_idx  = s.comp_idx_tensor.to(self.device)
                dyn_pmask = s.dynamic_pair_mask_tensor.to(self.device)
                fea_pairs = s.fea_pairs_tensor.to(self.device)
                pred_idx  = s.pred_idx_tensor.to(self.device)
                succ_idx  = s.succ_idx_tensor.to(self.device)

                # Forward pass → (pi, v)
                pi, v = self.model(
                    fea_act, act_mask, candidate, fea_team,
                    team_mask, comp_idx, dyn_pmask, fea_pairs,
                    pred_idx, succ_idx
                )

                # Sample action
                dist = Categorical(pi)
                action_flat = dist.sample()          # [B*P]
                log_prob    = dist.log_prob(action_flat)  # [B*P]

                # Store state before stepping
                memory.push_state(s)

                # Step environment: flat action → (activity, team)
                N_T      = env.N_T
                act_idx  = action_flat // N_T   # (B,) — 직접 activity index
                team_idx = action_flat % N_T    # (B,) — team index

                s, obj_value, done = env.step_pair(
                    act_idx.to(env.device),
                    team_idx.to(env.device)
                )

                # Reward 계산
                if self.reward_type == 'stepwise':
                    # Dense reward: r_t = est_tardiness(s_t) - est_tardiness(s_{t+1})
                    reward = env.step_reward.to(self.device)
                else:
                    # Sparse reward: 0 at intermediate steps, -objective at terminal
                    if done:
                        reward = -obj_value.to(self.device)
                    else:
                        reward = torch.zeros(batch_total, device=self.device)

                done_tensor = torch.full((batch_total,), done, dtype=torch.bool, device=self.device)

                memory.push_transition(
                    action_flat.to(self.device),
                    log_prob.to(self.device),
                    v.squeeze(-1).to(self.device),
                    reward,
                    done_tensor
                )

        # Final objective for logging
        final_obj = env._get_obj().mean().item()

        # --------------------------------------------------
        # 3. GAE Advantages
        # --------------------------------------------------
        t_data = memory.transpose_data()
        # Move all to model device
        t_data = tuple(t.to(self.device) for t in t_data)
        t_advantage, v_target = memory.get_gae_advantages()
        t_advantage = t_advantage.to(self.device)
        v_target    = v_target.to(self.device)

        # 논문: Advantage batch normalization (학습 안정화)
        t_advantage = (t_advantage - t_advantage.mean()) / (t_advantage.std() + 1e-8)

        full_batch_size = t_data[-1].shape[0]  # B*P*T
        num_mini = math.ceil(full_batch_size / self.ppo_minibatch_size)

        # --------------------------------------------------
        # 4. K-epoch PPO Updates
        # --------------------------------------------------
        self.model.train()
        total_loss   = 0.0
        total_v_loss = 0.0
        update_count = 0

        for _ in range(self.k_epochs):
            for i in range(num_mini):
                start = i * self.ppo_minibatch_size
                end   = min(start + self.ppo_minibatch_size, full_batch_size)

                # Forward through CURRENT policy
                pi_new, v_new = self.model(
                    t_data[0][start:end],   # fea_act
                    t_data[1][start:end],   # act_mask
                    t_data[6][start:end],   # candidate
                    t_data[2][start:end],   # fea_team
                    t_data[3][start:end],   # team_mask
                    t_data[5][start:end],   # comp_idx
                    t_data[4][start:end],   # dynamic_pair_mask
                    t_data[7][start:end],   # fea_pairs
                    t_data[8][start:end],   # pred_idx
                    t_data[9][start:end],   # succ_idx
                )

                actions_batch   = t_data[10][start:end]  # action (shifted by 2)
                old_logprobs    = t_data[14][start:end]  # stored log_probs (shifted by 2)
                adv_batch       = t_advantage[start:end]
                v_target_batch  = v_target[start:end]

                # Evaluate actions with new policy
                new_logprobs, ent = eval_actions(pi_new, actions_batch)

                # PPO ratio and clipped surrogate loss
                ratios = torch.exp(new_logprobs - old_logprobs.detach())
                surr1  = ratios * adv_batch
                surr2  = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                p_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                v_loss = F.mse_loss(v_new.squeeze(-1), v_target_batch)

                # Entropy bonus (negative entropy = discourage exploration, so subtract)
                ent_loss = -ent

                loss = self.ploss_coef * p_loss + self.vloss_coef * v_loss + entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()

                total_loss   += loss.item()
                total_v_loss += v_loss.item()
                update_count += 1

        # --------------------------------------------------
        # 5. Soft update policy_old (if tau > 0)
        # --------------------------------------------------
        if self.tau > 0:
            for old_p, new_p in zip(self.model_old.parameters(), self.model.parameters()):
                old_p.data.copy_(self.tau * old_p.data + (1.0 - self.tau) * new_p.data)

        avg_loss   = total_loss / max(update_count, 1)
        avg_reward = -final_obj  # run()에서 다시 부호 반전함

        return avg_loss, avg_reward

    def train_one_minibatch(self):
        """REINFORCE 알고리즘을 사용한 학습"""
        
        # 학습 데이터 생성 seed 고정 옵션 (디버깅용)
        training_seed_fix = self.trainer_params.get('training_seed_fix', False)
        training_seed = self.trainer_params.get('training_seed', None)
        
        if training_seed_fix and training_seed is not None:
            # 매 batch마다 동일한 seed로 고정 (디버깅용)
            import random
            import numpy as np
            random.seed(training_seed)
            np.random.seed(training_seed)
            torch.manual_seed(training_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(training_seed)
        
        # 환경 초기화 (device_mode에 따라 환경 device 결정)
        if self.device_mode == 'gpu':
            env_device = self.device  # GPU 모드: 환경도 GPU
        else:
            env_device = 'cpu'  # Hybrid/CPU 모드: 환경은 CPU
        
        env = SchedulingEnv(self.env_params, debug_env=self.debug_env, device=env_device)
        self.env = env
        problem = generate_scheduling_data_batch(self.env_params)
        env._reset(problem)
        
        batch_total = self.env_params['batch_size'] * self.env_params['pomo_size']
        
        # GAT 모드: action_to_pair 전달
        if self.model_type == 'gat':
            action_to_pair, max_action_space = env.action_to_pair, env.max_action_space
            self.model.set_action_space(action_to_pair.to(self.device), max_action_space)
        
        done = False
        s = env._get_state()
        log_prob_tmp = torch.zeros(batch_total).to(self.device)
        
        step_count = 0
        MAX_STEPS = 1000
        
        # Entropy regularization 계수 확인
        entropy_coef = self.trainer_params.get('entropy_coef', 0.0)
        use_entropy = entropy_coef > 0
        
        # Shaped reward 및 entropy 누적용
        cumulative_reward = torch.zeros(batch_total)
        cumulative_step_reward = torch.zeros(batch_total, device=self.device)
        if use_entropy:
            cumulative_entropy = torch.zeros(batch_total).to(self.device)

        while not done:
            step_count += 1
            
            if self.model_type == 'daniel':
                # ============================================
                # DANIEL 모델 경로
                # ============================================
                # EnvState의 텐서들을 모델 device로 이동
                fea_act = s.fea_act_tensor.to(self.device)
                act_mask = s.act_mask_tensor.to(self.device)
                candidate = s.candidate_tensor.to(self.device)
                fea_team = s.fea_team_tensor.to(self.device)
                team_mask = s.team_mask_tensor.to(self.device)
                comp_idx = s.comp_idx_tensor.to(self.device)
                dynamic_pair_mask = s.dynamic_pair_mask_tensor.to(self.device)
                fea_pairs = s.fea_pairs_tensor.to(self.device)
                pred_idx = s.pred_idx_tensor.to(self.device)
                succ_idx = s.succ_idx_tensor.to(self.device)

                # DANIEL forward: (pi, v)
                pi, v = self.model(
                    fea_act, act_mask, candidate, fea_team,
                    team_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                    pred_idx, succ_idx
                )
                
                # Action 샘플링
                dist = Categorical(pi)
                action_flat = dist.sample()  # (B,) — index in [0, P*T)
                log_prob = dist.log_prob(action_flat)
                entropy = dist.entropy()
                
                log_prob_tmp += log_prob
                if use_entropy:
                    cumulative_entropy += entropy
                
                # Action 변환: flat index → (activity, team)
                N_T = env.N_T
                act_idx  = action_flat // N_T  # (B,) — 직접 activity index
                team_idx = action_flat % N_T   # (B,)

                # 환경 step (activity, team 쌍으로 직접)
                s, obj_value, done = env.step_pair(
                    act_idx.to(env.device),
                    team_idx.to(env.device)
                )
            else:
                # ============================================
                # GAT (GNN) 모델 경로 (기존 코드)
                # ============================================
                action, log_prob, entropy = self.model.get_action(s)
                log_prob_tmp += log_prob
                if use_entropy:
                    cumulative_entropy += entropy
                
                # device_mode에 따라 action을 환경 device로 이동
                if self.device_mode == 'gpu':
                    s, obj_value, done = env.step(action)
                else:
                    s, obj_value, done = env.step(action.to('cpu'))

            # Step-wise reward 누적 (REINFORCE: 에피소드 전체 합을 최종 reward로 사용)
            if self.reward_type == 'stepwise':
                cumulative_step_reward += env.step_reward.to(self.device)

        # 목적함수값을 reward로 변환
        if self.reward_type == 'stepwise':
            # Dense reward 합산: init_est - final_tardiness (positive = better)
            reward = cumulative_step_reward
        else:
            reward = -obj_value  # reward = -목적함수 (최소화 문제를 최대화 문제로)
        
        # POMO 체크: -1 또는 1이면 POMO 사용 안함
        use_pomo = self.env_params['pomo_size'] > 1
        
        reward_reshape = reward.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
        log_prob_tmp_valid = log_prob_tmp
        
        # Baseline 타입에 따른 Advantage 계산
        baseline_type = self.trainer_params.get('baseline_type', 'pomo')
        normalize_advantage = self.trainer_params.get('normalize_advantage', True)
        
        if baseline_type == 'batch':
            # 배치 전체 baseline + 표준화 (B×K)
            if self.verbose_logging:
                print("   ℹ️ Batch Baseline: 배치 전체에서 baseline 계산 (B×K)")
            reward_flat = reward_reshape.reshape(-1)  # (B*K,)
            
            # Baseline = 배치 전체 평균
            advantage = reward_flat - reward_flat.mean()
            
            # 표준편차로 정규화 (옵션)
            if normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                if self.verbose_logging:
                    print(f"     └─ 표준화 완료: mean=0, std=1")
        
        elif baseline_type == 'pomo' and use_pomo:
            # POMO baseline (인스턴스별)
            if self.verbose_logging:
                print("   ℹ️ POMO Baseline: 인스턴스별 POMO 평균 사용")
            advantage = reward_reshape - torch.mean(reward_reshape, -1, keepdim=True)
            
            # 표준편차로 정규화 (옵션)
            if normalize_advantage:
                advantage_std = torch.std(reward_reshape, -1, keepdim=True, unbiased=False)
                advantage = advantage / (advantage_std + 1e-8)
            
            advantage = advantage.reshape(-1)
        
        else:
            # Baseline 없음
            if self.verbose_logging:
                print("   ℹ️ No Baseline: raw reward 사용")
            advantage = reward_reshape.reshape(-1)  # Raw reward 그대로 사용
        
        advantage = advantage.to(self.device)
        
        # Policy loss 계산
        policy_loss = -advantage * log_prob_tmp_valid
        
        # Entropy regularization 추가 (사용하는 경우에만)
        if use_entropy:
            entropy_bonus = entropy_coef * cumulative_entropy  # Entropy를 보너스로 추가 (최대화)
            loss = policy_loss.mean() - entropy_bonus.mean()  # Entropy bonus 추가 (음수 = 최대화)
        else:
            loss = policy_loss.mean()
        
        # 목적함수값 계산
        obj_values = env._get_obj()  # (batch_size * pomo_size,) 목적함수값
        
        # 에피소드 완료 시 각 배치별 목적함수 출력 (verbose_logging이 True일 때만)
        if self.verbose_logging:
            print(f"\n📊 에피소드 완료 - 배치별 목적함수 ({self.objective}):")
            for b_idx in range(self.env_params['batch_size']):
                batch_offset = b_idx * self.env_params['pomo_size']
                batch_obj_values = obj_values[batch_offset:batch_offset + self.env_params['pomo_size']]
                for p_idx in range(self.env_params['pomo_size']):
                    instance_idx = batch_offset + p_idx
                    
                    # Activity 상태 개수 계산
                    num_total_activities = env.num_activities[instance_idx].item()
                    num_started = env.activity_started[instance_idx, :num_total_activities].sum().item()
                    num_ended = env.activity_ended[instance_idx, :num_total_activities].sum().item()
                    
                    print(f"   Batch {b_idx} - POMO {p_idx}: {batch_obj_values[p_idx].item():.2f} "
                          f"(started: {num_started}/{num_total_activities}, ended: {num_ended}/{num_total_activities})")
        
        # 목적함수값 계산 (실제 목적함수의 평균)
        obj_value = obj_values.mean().item()
        
        # 각 인스턴스별 simulation done 여부 확인
        # Scheduling 환경: 모든 activity가 완료되었는지
        done_status = env.activity_ended.all(dim=1)  # (batch_size * pomo_size,) - 각 인스턴스별 done 여부
        
        # 배치 내 각 인스턴스별 정보 저장
        batch_info = {
            'rewards': reward_reshape,  # (batch_size, pomo_size)
            'step_count': step_count,
            'num_activities': env.num_activities,  # 각 배치별 activity 개수 (batch_size * pomo_size,)
            'done_status': done_status  # 각 인스턴스별 simulation done 여부
        }

        return loss, obj_value, step_count, batch_info
    
    def _eval_one_problem(self, mode='test', test_file_start=0, test_file_end=4, data_folder=None):
        """
        테스트 전용 함수: 저장된 테스트 데이터로 모델 평가
        학습 중에는 호출되지 않음
        """
        
        if mode == 'test':
            self.model.eval()
        
        test_score_list = []
        
        # 테스트 파일 범위 설정
        test_file_indices = list(range(test_file_start, test_file_end + 1))
        episodes_to_run = len(test_file_indices)
        
        # 데이터 폴더 설정
        if data_folder is None:
            data_folder = './generated_datasets/data_test'

        for idx, i in enumerate(test_file_indices):
            # silent_mode 확인
            silent_mode = self.trainer_params.get('silent_mode', False)
            
            if not silent_mode:
                print(f"\n📋 테스트 파일 {i}.pickle ({idx + 1}/{episodes_to_run})")
            
            # 테스트 데이터 먼저 로드
            from pathlib import Path
            problem_path = str(Path(data_folder) / f"{i}.pickle")
            try:
                with open(problem_path, 'rb') as fr:
                    problem = pickle.load(fr)
                if not silent_mode:
                    print(f"   테스트 데이터 로드: {problem_path}")
                
                # pickle 파일의 env_params를 사용하여 환경 초기화
                loaded_env_params = problem['env_params'].copy()
                
                # 배치 범위 설정
                batch_start = self.env_params.get('batch_start', 0)
                batch_end = self.env_params.get('batch_end', None)
                original_batch_size = problem['order_sku_requirements'].shape[0]
                
                # batch_end가 None이면 전체 배치 사용
                if batch_end is None:
                    batch_end = original_batch_size - 1
                
                # 범위 검증
                if batch_start < 0 or batch_start >= original_batch_size:
                    if not silent_mode:
                        print(f"   ⚠️ batch_start({batch_start})가 유효하지 않습니다. 0으로 설정합니다.")
                    batch_start = 0
                if batch_end >= original_batch_size:
                    if not silent_mode:
                        print(f"   ⚠️ batch_end({batch_end})가 배치 크기({original_batch_size})를 초과합니다. {original_batch_size-1}로 조정합니다.")
                    batch_end = original_batch_size - 1
                if batch_start > batch_end:
                    if not silent_mode:
                        print(f"   ⚠️ batch_start가 batch_end보다 큽니다. 전체 배치를 사용합니다.")
                    batch_start = 0
                    batch_end = original_batch_size - 1
                
                # 테스트할 배치 수
                test_batch_size = batch_end - batch_start + 1
                
                if not silent_mode:
                    print(f"   배치 범위: {batch_start}~{batch_end} (총 {test_batch_size}개 인스턴스)")
                
                # problem 데이터를 슬라이싱
                sliced_problem = {}
                for key, value in problem.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] == original_batch_size:
                        sliced_problem[key] = value[batch_start:batch_end+1]
                    else:
                        sliced_problem[key] = value
                
                # 테스트용 설정 덮어쓰기
                loaded_env_params['batch_size'] = test_batch_size
                loaded_env_params['pomo_size'] = self.env_params.get('pomo_size', 1)
                loaded_env_params['debug_env'] = self.env_params.get('debug_env', False)
                loaded_env_params['debug_env_verbose'] = self.env_params.get('debug_env_verbose', False)
                loaded_env_params['use_fcfs_sort'] = self.env_params.get('use_fcfs_sort', False)
                loaded_env_params['objective'] = self.env_params.get('objective', 'makespan')  # 목적함수 덮어쓰기
                
                # Action space 덮어쓰기 (test.py에서 설정한 action space 사용)
                if 'action_space' in self.env_params:
                    loaded_env_params['action_space'] = self.env_params['action_space']
                if 'item_selection_rule' in self.env_params:
                    loaded_env_params['item_selection_rule'] = self.env_params['item_selection_rule']
                
                # Order-CP composite rule 설정 덮어쓰기
                if 'order_cp_order_selection_rule' in self.env_params:
                    loaded_env_params['order_cp_order_selection_rule'] = self.env_params['order_cp_order_selection_rule']
                if 'order_cp_cp_selection_rule' in self.env_params:
                    loaded_env_params['order_cp_cp_selection_rule'] = self.env_params['order_cp_cp_selection_rule']
                if 'order_cp_priority' in self.env_params:
                    loaded_env_params['order_cp_priority'] = self.env_params['order_cp_priority']
                
                if not silent_mode:
                    print(f"   환경 파라미터: N_O={loaded_env_params['N_O']}, N_S={loaded_env_params['N_S']}, "
                          f"N_C={loaded_env_params['N_C']}, N_R={loaded_env_params['N_R']}, N_L={loaded_env_params['N_L']}")
                    print(f"   목적함수: {loaded_env_params['objective']}")
                    print(f"   Action Space: {loaded_env_params.get('action_space', 'N/A')}")
                    # Rule 모드에서만 룰 설정 출력
                    if mode == 'rule' and loaded_env_params.get('action_space') == 'order-cp':
                        print(f"   Order-CP 룰 설정 (Rule 모드):")
                        print(f"     Order Rule: {loaded_env_params.get('order_cp_order_selection_rule', 'N/A')}")
                        print(f"     CP Rule: {loaded_env_params.get('order_cp_cp_selection_rule', 'N/A')}")
                        print(f"     Priority: {loaded_env_params.get('order_cp_priority', 'N/A')}")
                    elif mode == 'test':
                        print(f"   Item Selection Rule (환경 내부): {loaded_env_params.get('item_selection_rule', 'N/A')}")
                
                sim = RSSEnv(loaded_env_params, debug_env=self.debug_env, sequence_encoder=self.sequence_encoder)
                sim.problem = sliced_problem
                sim._reset(-1, -1)
            except FileNotFoundError:
                if not silent_mode:
                    print(f"   ❌ 테스트 데이터 없음: {problem_path}")
                    print(f"   스킵하고 다음 문제로 진행")
                continue
            
            s = sim._get_state()
            done = False
            
            # 파일 내 인스턴스 수 확인 (슬라이싱된 배치 수)
            num_instances = sim.problem['N_I_for_batch'].shape[0]
            if not silent_mode:
                print(f"   초기화 완료 (테스트 인스턴스 수: {num_instances})")

            if mode == 'test':
                # 모델 기반 Greedy rollout
                while not done:
                    action = self.model.get_max_action(s)
                    s, r, done = sim.step(action.to('cpu'))
            
            elif mode == 'rule':
                # 룰 기반 rollout
                rule_adapter.set_env(sim)
                while not done:
                    action = rule_adapter.get_action()  # state 불필요
                    s, r, done = sim.step(action.to('cpu'))

            score = sim._get_obj()
            
            # silent_mode 확인
            silent_mode = self.trainer_params.get('silent_mode', False)
            
            # 목적함수 이름 가져오기
            objective_name = loaded_env_params.get('objective', 'makespan')
            
            # 각 인스턴스별 점수 저장 및 출력
            if isinstance(score, torch.Tensor):
                instance_scores = score.cpu().numpy()
                test_score_list.append(instance_scores)
                avg_score = instance_scores.mean()
                if not silent_mode:
                    print(f"\n   📊 인스턴스별 {objective_name}:")
                    for inst_idx, inst_score in enumerate(instance_scores):
                        print(f"      Instance {inst_idx}: {inst_score:.4f}")
                    print(f"   ✅ 평균: {avg_score:.4f}")
            else:
                test_score_list.append([score])
                if not silent_mode:
                    print(f"   ✅ 완료 - {objective_name}: {score:.4f}")
            
            # 간트차트 생성 (옵션에 따라)
            if show_gantt:
                # policy_name_override가 있으면 사용, 없으면 기본 rule_name 사용
                if policy_name_override:
                    policy_name = f"Rule-{policy_name_override} Policy"
                else:
                    policy_name = f"{'RL' if mode == 'test' else 'Rule-' + rule_name.upper()} Policy"
                
                if not silent_mode:
                    print(f"\n{'='*60}")
                    print(f"   🎨 간트차트 생성 요청됨 (show_gantt={show_gantt})")
                    print(f"   정책 이름: {policy_name}")
                    print(f"{'='*60}")
                try:
                    fig = sim.create_gantt_chart(batch_idx=0, show_plot=True, policy_name=policy_name)
                    if not silent_mode:
                        print(f"   ✅ 간트차트 생성 완료")
                except Exception as e:
                    if not silent_mode:
                        print(f"   ❌ 간트차트 생성 실패: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                if not silent_mode:
                    print(f"   ⚠️  show_gantt={show_gantt}이므로 간트차트를 생성하지 않습니다.")

        # 전체 평균 및 파일별 결과 반환
        all_scores_flat = []
        for scores in test_score_list:
            if isinstance(scores, np.ndarray):
                all_scores_flat.extend(scores)
            else:
                all_scores_flat.extend(scores)
        
        overall_mean = np.mean(all_scores_flat) if all_scores_flat else 0.0
        return overall_mean, test_score_list

    def _load_checkpoint(self, checkpoint_path, resume_training=False):
        """
        체크포인트에서 모델 가중치 및 학습 상태 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            resume_training: True면 epoch 번호도 이어받기, False면 가중치만 로드
        """
        print(f"\n{'='*60}")
        print(f"📂 체크포인트 로드 중: {checkpoint_path}")
        print(f"{'='*60}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 모델 가중치 로드 완료")

            # PPO: model_old 복원
            if self.algorithm_type == 'ppo' and 'model_old_state_dict' in checkpoint and checkpoint['model_old_state_dict'] is not None:
                self.model_old.load_state_dict(checkpoint['model_old_state_dict'])
                print(f"✅ PPO policy_old 가중치 로드 완료")

            # Optimizer 상태 로드 (있는 경우)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✅ Optimizer 상태 로드 완료")
            
            # Epoch 번호 이어받기 (resume_training=True인 경우)
            if resume_training and 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"✅ 학습 재개: Epoch {self.start_epoch}부터 시작")
            else:
                self.start_epoch = 0
                print(f"✅ 가중치만 로드: Epoch 0부터 시작 (Fine-tuning 모드)")
            
            # 저장된 성능 정보 출력
            if 'train_score' in checkpoint:
                print(f"   📊 저장된 Train Score: {checkpoint['train_score']:.4f}")
            if 'val_score' in checkpoint and checkpoint['val_score'] is not None:
                print(f"   📊 저장된 Val Score: {checkpoint['val_score']:.4f}")
            
            # 초기 validation score 로드 (개선율 계산용)
            if 'initial_val_score' in checkpoint:
                self.initial_val_score_from_checkpoint = checkpoint['initial_val_score']
                print(f"   📊 초기 Val Score: {self.initial_val_score_from_checkpoint:.4f}")
            
            print(f"{'='*60}\n")
            
        except FileNotFoundError:
            print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
            print(f"   처음부터 학습을 시작합니다.")
            self.start_epoch = 0
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {e}")
            print(f"   처음부터 학습을 시작합니다.")
            self.start_epoch = 0
    
    def _prepare_validation_dataset(self, validation_batch_size=10, validation_pomo_size=1):
        """
        고정된 validation 데이터셋 준비 (단일 배치 파일: val_batch.pickle)
        모든 epoch에서 동일한 데이터로 평가하기 위함

        Args:
            validation_batch_size: Validation 인스턴스 수
            validation_pomo_size: Validation POMO 크기 (보통 1)
        """
        # 시드 고정하여 재현 가능한 validation set 생성
        validation_seed = 2025

        # Validation 데이터 폴더 경로
        val_data_folder = './data/val'
        os.makedirs(val_data_folder, exist_ok=True)

        # 단일 배치 파일 경로
        batch_file_path = os.path.join(val_data_folder, 'val_batch.pickle')

        if os.path.exists(batch_file_path):
            # 기존 배치 파일 로드하여 validation_problem에 저장
            with open(batch_file_path, 'rb') as f:
                self.validation_problem = pickle.load(f)
            print(f"📂 기존 Validation 배치 데이터셋 로드: {batch_file_path}")
            print(f"   인스턴스 수: {validation_batch_size}")
            print(f"✅ Validation 데이터셋 로드 완료 (배치 형식)")
            return

        # 파일이 없으면 새로 생성
        print(f"📦 새로운 Validation 데이터셋 생성 중... (인스턴스 수: {validation_batch_size})")

        # 모든 랜덤 시드 저장 (복원용)
        original_random_state = random.getstate()
        original_np_state = np.random.get_state()
        original_torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            original_cuda_rng_state = torch.cuda.get_rng_state()

        # 모든 랜덤 시드 고정
        random.seed(validation_seed)
        np.random.seed(validation_seed)
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(validation_seed)

        # 배치 전체를 한 번에 생성
        val_env_params = copy.deepcopy(self.env_params)
        val_env_params['batch_size'] = validation_batch_size
        val_env_params['pomo_size'] = validation_pomo_size

        batch_problem = generate_scheduling_data_batch(val_env_params)

        # 파일로 저장
        with open(batch_file_path, 'wb') as f:
            pickle.dump(batch_problem, f)

        # RNG 상태 복원
        random.setstate(original_random_state)
        np.random.set_state(original_np_state)
        torch.set_rng_state(original_torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(original_cuda_rng_state)

        # 메모리에도 보관
        self.validation_problem = batch_problem

        print(f"✅ Validation 데이터셋 생성 완료 (고정 시드: {validation_seed})")
        print(f"💾 저장 위치: {batch_file_path}")
        print(f"   인스턴스 수: {validation_batch_size}, POMO: {validation_pomo_size}")
        print(f"   ℹ️  다음 학습부터는 이 파일을 재사용합니다.")
    
    def _eval_validation(self, validation_batch_size=10, validation_pomo_size=1):
        """
        Validation 평가 함수: 배치 데이터로 모델 성능 평가 (Greedy policy)
        학습과 동일하게 배치 단위로 한 번에 처리

        Args:
            validation_batch_size: Validation 인스턴스 수
            validation_pomo_size: Validation POMO 크기 (보통 1)

        Returns:
            avg_score: 평균 validation score (목적함수값)
        """
        self.model.eval()

        # Validation 데이터 로드 (메모리 캐시 또는 파일)
        if self.validation_problem is not None:
            problem = self.validation_problem
        else:
            batch_file_path = os.path.join('./data/val', 'val_batch.pickle')
            with open(batch_file_path, 'rb') as f:
                problem = pickle.load(f)
            self.validation_problem = problem  # 캐시

        # 배치 환경 파라미터
        val_env_params = copy.deepcopy(self.env_params)
        val_env_params['batch_size'] = validation_batch_size
        val_env_params['pomo_size'] = validation_pomo_size

        # device_mode에 따라 환경 device 결정
        env_device = self.device if self.device_mode == 'gpu' else 'cpu'

        batch_total = validation_batch_size * validation_pomo_size

        # 환경 초기화 (배치 전체를 한 번에)
        val_env = SchedulingEnv(val_env_params, debug_env=False, device=env_device)
        val_env._reset(problem)

        if self.model_type == 'gat':
            action_to_pair, max_action_space = val_env.action_to_pair, val_env.max_action_space
            self.model.set_action_space(action_to_pair.to(self.device), max_action_space)

        done = False
        s = val_env._get_state()

        with torch.no_grad():
            while not done:
                if self.model_type == 'daniel':
                    fea_act = s.fea_act_tensor.to(self.device)
                    act_mask = s.act_mask_tensor.to(self.device)
                    candidate = s.candidate_tensor.to(self.device)
                    fea_team = s.fea_team_tensor.to(self.device)
                    team_mask = s.team_mask_tensor.to(self.device)
                    comp_idx = s.comp_idx_tensor.to(self.device)
                    dynamic_pair_mask = s.dynamic_pair_mask_tensor.to(self.device)
                    fea_pairs = s.fea_pairs_tensor.to(self.device)
                    pred_idx = s.pred_idx_tensor.to(self.device)
                    succ_idx = s.succ_idx_tensor.to(self.device)

                    pi, v = self.model(
                        fea_act, act_mask, candidate, fea_team,
                        team_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                        pred_idx, succ_idx
                    )

                    # Greedy: argmax
                    action_flat = torch.argmax(pi, dim=1)

                    N_T = val_env.N_T
                    act_idx  = action_flat // N_T
                    team_idx = action_flat % N_T

                    s, obj_value, done = val_env.step_pair(
                        act_idx.to(val_env.device),
                        team_idx.to(val_env.device)
                    )
                else:
                    action = self.model.get_max_action(s)
                    if self.device_mode == 'gpu':
                        s, obj_value, done = val_env.step(action)
                    else:
                        s, obj_value, done = val_env.step(action.to('cpu'))

        # 배치 전체 목적함수값 계산 → 평균
        obj_values = val_env._get_obj()  # (batch_total,)
        avg_obj_value = obj_values.mean().item()

        self.model.train()
        return avg_obj_value

    def plot_score(self, valid_score, valid_loss):
        plt.figure()
        plt.title("result_valid_score_log")
        plt.plot(valid_score)
        plt.show()
        plt.figure()
        plt.title("valid_loss")
        plt.plot(valid_loss)
        plt.show()

