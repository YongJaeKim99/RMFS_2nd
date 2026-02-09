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

# TODO: GNN 모델을 추가해주세요
# from scheduling_model import SchedulingModel

import logging
import warnings
import wandb

warnings.filterwarnings('ignore')

WANDB_AVAILABLE = True  # wandb 사용 가능 여부

class RolloutBuffer:
    """PPO를 위한 롤아웃 버퍼"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        return {
            'states': self.states,
            'actions': self.actions,
            'log_probs': torch.stack(self.log_probs) if self.log_probs else None,
            'rewards': torch.stack(self.rewards) if self.rewards else None,
            'values': torch.stack(self.values) if self.values else None,
            'dones': torch.tensor(self.dones) if self.dones else None
        }

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
            algorithm = trainer_params.get('algorithm', 'REINFORCE')
            self.checkpoint_dir = f"./checkpoints/{timestamp}_{objective}_{algorithm}"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"✅ 체크포인트 저장 폴더 생성: {self.checkpoint_dir}")
        else:
            # test 모드에서는 폴더 생성하지 않음
            self.checkpoint_dir = None
        
        self.debug_env = trainer_params.get('debug_env', False)
        self.debug_model = trainer_params.get('debug_model', False)
        
        # Device 설정
        device_mode = trainer_params.get('device_mode', 'hybrid')
        model_device = trainer_params.get('model_device', 'cuda' if torch.cuda.is_available() else 'cpu')
        env_device = trainer_params.get('env_device', 'cpu')
        
        self.model_device = torch.device(model_device)
        self.env_device = torch.device(env_device)
        
        print(f"Model Device: {self.model_device}, Environment Device: {self.env_device}")
        print(f"Debug ENV: {self.debug_env}, Debug Model: {self.debug_model}")
        
        # TODO: 모델 초기화 - GNN 모델 추가 후 주석 해제
        # self.model = SchedulingModel(self.model_params, debug_model=self.debug_model).to(self.model_device)
        # self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        
        # 임시: 모델이 없으므로 None으로 설정
        self.model = None
        self.optimizer = None
        print("⚠️  경고: GNN 모델이 아직 추가되지 않았습니다. 모델 초기화를 건너뜁니다.")

        self.result_train_loss_log = []
        
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
            algorithm = self.trainer_params.get('algorithm', 'REINFORCE')
            n_p = self.env_params.get('N_P', 0)
            n_t = self.env_params.get('N_T', 0)
            batch_size = self.env_params.get('batch_size', 0)
            pomo_size = self.env_params.get('pomo_size', 0)
            learning_rate = self.optimizer_params['optimizer']['lr']
            normalize_advantage = self.trainer_params.get('normalize_advantage', False)
            adv_norm_str = 'advnorm' if normalize_advantage else 'no_advnorm'
            timestamp = datetime.now().strftime("%m%d_%H%M")
            run_name = self.trainer_params.get('wandb_run_name')
            if run_name is None:
                run_name = f"{objective}_{algorithm}_P{n_p}_T{n_t}_bs{batch_size}_p{pomo_size}_lr{learning_rate}_{adv_norm_str}_{timestamp}"
            
            # Config 설정
            config = {
                "objective": objective,
                "algorithm": algorithm,
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
                "embedding_dim": self.model_params.get('embedding_dim'),
                "num_head": self.model_params.get('num_head'),
                "num_encoder_layer": self.model_params.get('num_encoder_layer'),
            }
            
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
            start_epoch = 1  # 다음부터는 epoch 1부터 시작
        
        for epoch in range(start_epoch, end_epoch+1):
            epoch_start_time = time.time()
            
            # 학습 수행
            print(f"epoch: {epoch}")
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
                            'initial_val_score': initial_val_score  # 개선율 계산용
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
                ckpt_path = os.path.join(self.checkpoint_dir, f"pomo_rss_epoch{epoch}.pt")
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
                    'initial_val_score': initial_val_score  # 개선율 계산용
                }, ckpt_path)
                print(f"  💾 체크포인트 저장: {ckpt_path}")
                
                if self.use_wandb:
                    artifact = wandb.Artifact(
                        name=f"pomo_rss_model_epoch_{epoch}",   # Artifact 이름 (wandb 상에서 보이는 이름)
                        type="model"                        # 타입은 자유롭게 지정 가능 (ex. model, checkpoint 등)
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
        
        # 알고리즘 선택
        algorithm = self.trainer_params.get('algorithm', 'REINFORCE')

        for i in range(self.trainer_params['accumulation_steps']):
            print(f"accumulation_steps: {i}")
            
            # 알고리즘에 따라 적절한 학습 함수 호출
            if algorithm == 'PPO':
                loss, obj_value, step_count, batch_info = self.train_one_micro_batch_ppo()
                print(f"  평균 목적함수값: {obj_value:.4f}, Loss: {loss:.4f}")
                total_loss += loss
            else:  # REINFORCE
                # REINFORCE loss 타입 확인
                rl_loss_type = self.trainer_params.get('rl_loss_type', 'standard')
                if rl_loss_type == 'sil':
                    loss, obj_value, step_count, batch_info = self.train_one_micro_batch_sil()
                    print(f"  평균 목적함수값: {obj_value:.4f}, Loss: {loss.item():.4f} (SIL)")
                else:
                    loss, obj_value, step_count, batch_info = self.train_one_micro_batch()
                    print(f"  평균 목적함수값: {obj_value:.4f}, Loss: {loss.item():.4f}")
                loss.backward()  # REINFORCE는 여기서 backward
                total_loss += loss.item()
            
            # 배치 내 각 인스턴스별 정보 출력 (옵션)
            if self.debug_instance_info:
                rewards = batch_info['rewards']
                N_I_list = batch_info['N_I']
                done_status = batch_info['done_status']
                for b in range(self.env_params['batch_size']):
                    for p in range(self.env_params['pomo_size']):
                        instance_idx = b * self.env_params['pomo_size'] + p
                        reward_val = rewards[b, p].item()
                        n_items = N_I_list[instance_idx].item()
                        is_done = done_status[instance_idx].item()
                        done_mark = "✅" if is_done else "❌"
                        print(f"    Instance{b}_POMO{p}: Reward={reward_val:.4f}, Items={n_items}, Steps={step_count}, Done={done_mark}")
            
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

    def train_one_micro_batch_ppo(self):
        """PPO 알고리즘을 사용한 학습"""
        # TODO: GNN 모델 추가 후 구현
        raise NotImplementedError("GNN 모델(SchedulingModel)을 추가한 후 PPO 학습을 구현해주세요.")
        
        # Anomaly detection for debugging in-place operation issues
        torch.autograd.set_detect_anomaly(True)
        
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
        
        # 환경 초기화
        env = SchedulingEnv(self.env_params, debug_env=self.debug_env)
        self.env = env
        problem = generate_scheduling_data_batch(self.env_params)
        env._reset(problem)
        done = False
        s = env._get_state()
        
        # Rollout buffer 초기화
        buffer = RolloutBuffer()
        
        step_count = 0
        MAX_STEPS = 1000
        
        # Rollout phase: 환경과 상호작용하며 데이터 수집
        while not done:
            step_count += 1
            
            # Action 샘플링
            action, log_prob, entropy = self.model.get_action(s)
            
            # Value 계산
            value = self.model.get_value(s)
            
            # 환경과 상호작용 (use_step_reward=True로 매 step reward 받기)
            next_s, reward, done = env.step(action.to('cpu'), use_step_reward=True)
            
            # Buffer에 저장 (log_prob과 value는 detach하여 computational graph에서 분리)
            buffer.add(s, action, log_prob.detach(), reward, value.detach(), done)
            
            if not done:
                s = next_s
            
            if step_count >= MAX_STEPS:
                break
        
        # Buffer에서 데이터 가져오기
        data = buffer.get()
        
        # GAE (Generalized Advantage Estimation) 계산
        rewards = data['rewards']  # (num_steps, batch_size)
        values = data['values']    # (num_steps, batch_size)
        old_log_probs = data['log_probs']  # (num_steps, batch_size)
        
        # Deadlock 발생 체크 및 제외 (옵션)
        exclude_deadlock = self.trainer_params.get('exclude_deadlock_instances', False)
        batch_size = self.env_params['batch_size'] * self.env_params['pomo_size']
        valid_indices = torch.arange(batch_size)
        
        if exclude_deadlock:
            # deadlock_occurred를 (batch_size, pomo_size)로 reshape
            deadlock_reshape = env.deadlock_occurred.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
            # 각 배치별로 하나라도 deadlock이 발생하면 해당 배치의 모든 POMO 제외
            deadlock_per_batch = deadlock_reshape.any(dim=1)  # (batch_size,)
            
            num_deadlock = deadlock_per_batch.sum().item()
            if num_deadlock > 0:
                print(f"   ⚠️ Deadlock 발생: {num_deadlock}개 배치 제외 (총 {self.env_params['batch_size']}개 중)")
            
            # valid한 인스턴스 인덱스 구하기
            valid_batch_mask = ~deadlock_per_batch  # (batch_size,)
            valid_indices = []
            for batch_idx in range(self.env_params['batch_size']):
                if valid_batch_mask[batch_idx]:
                    # 해당 배치의 모든 POMO 인덱스 추가
                    start_idx = batch_idx * self.env_params['pomo_size']
                    end_idx = start_idx + self.env_params['pomo_size']
                    valid_indices.extend(range(start_idx, end_idx))
            
            valid_indices = torch.tensor(valid_indices, dtype=torch.long)
            
            # valid한 인스턴스가 없으면 학습 스킵
            if len(valid_indices) == 0:
                print(f"   ❌ 모든 배치에서 deadlock 발생! 이번 micro-batch는 학습하지 않습니다.")
                batch_info = {
                    'rewards': rewards.sum(dim=0).reshape(self.env_params['batch_size'], self.env_params['pomo_size']),
                    'step_count': step_count,
                    'N_I': env.N_I,
                    'done_status': env.order_cleared.all(dim=1)
                }
                return 0.0, 0.0, step_count, batch_info
            
            # valid한 인스턴스만 선택
            rewards = rewards[:, valid_indices]  # (num_steps, valid_size)
            values = values[:, valid_indices]    # (num_steps, valid_size)
            old_log_probs = old_log_probs[:, valid_indices]  # (num_steps, valid_size)
            
            # states와 actions도 필터링
            filtered_states = []
            for t in range(len(data['states'])):
                state_list = data['states'][t]
                filtered_state_list = [state_list[i] for i in valid_indices]
                filtered_states.append(filtered_state_list)
            data['states'] = filtered_states
            
            filtered_actions = []
            for t in range(len(data['actions'])):
                action_tensor = data['actions'][t]
                filtered_action_tensor = action_tensor[valid_indices]
                filtered_actions.append(filtered_action_tensor)
            data['actions'] = filtered_actions
        
        # Returns 및 Advantages 계산
        returns = []
        advantages = []
        
        # 각 배치 인스턴스별로 계산 (valid한 인스턴스만)
        num_valid = rewards.shape[1]
        
        for b in range(num_valid):
            # 해당 배치의 rewards와 values 추출
            rewards_b = rewards[:, b]  # (num_steps,)
            values_b = values[:, b]    # (num_steps,)
            
            # Returns 계산 (Monte Carlo return)
            returns_b = torch.zeros_like(rewards_b)
            running_return = 0.0
            for t in reversed(range(len(rewards_b))):
                running_return = rewards_b[t] + 0.99 * running_return  # gamma=0.99
                returns_b[t] = running_return
            
            # Advantage 계산 (A = R - V)
            advantages_b = returns_b - values_b
            
            returns.append(returns_b)
            advantages.append(advantages_b)
        
        returns = torch.stack(returns, dim=1)  # (num_steps, valid_size)
        advantages = torch.stack(advantages, dim=1)  # (num_steps, valid_size)
        
        # POMO 체크
        use_pomo = self.env_params['pomo_size'] > 1
        
        # Advantage normalization (옵션)
        # PPO에서는 value function이 이미 baseline 역할, 여기서는 정규화만 적용
        baseline_type = self.trainer_params.get('baseline_type', 'pomo')
        normalize_advantage = self.trainer_params.get('normalize_advantage', True)
        
        if normalize_advantage:
            if baseline_type == 'batch':
                # 배치 전체 표준화 (B×K 전체)
                print("   ℹ️ PPO Advantage 정규화: 배치 전체 (B×K)")
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif baseline_type == 'pomo' and use_pomo:
                # POMO별 표준화 (인스턴스별)
                print("   ℹ️ PPO Advantage 정규화: 인스턴스별 (POMO)")
                # 시간축 평균/표준편차 계산 후 정규화
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                print("   ℹ️ PPO Advantage 정규화: 전체 평균/표준편차")
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # old_log_probs와 values를 device로 이동 (detached 상태)
        device = next(self.model.parameters()).device
        old_log_probs = old_log_probs.to(device)
        returns = returns.to(device)
        advantages = advantages.to(device)
        
        # PPO update epochs
        ppo_epochs = self.trainer_params.get('ppo_epochs', 4)
        ppo_clip = self.trainer_params.get('ppo_clip', 0.2)
        value_coef = self.trainer_params.get('ppo_value_coef', 0.5)
        entropy_coef = self.trainer_params.get('ppo_entropy_coef', 0.01)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(ppo_epochs):
            # 모든 step에 대해 update
            for t in range(len(data['states'])):
                s_t = data['states'][t]
                action_t = data['actions'][t]
                old_log_prob_t = old_log_probs[t]  # 이미 detached 상태
                return_t = returns[t]
                advantage_t = advantages[t]
                
                # 실제 취했던 액션의 log_prob을 현재 정책으로 재계산
                # forward를 사용하여 logit을 얻고, Categorical로 log_prob 계산
                s_t_loader = DataLoader(s_t, batch_size=len(s_t))
                for s_t_batch in s_t_loader:
                    break
                s_t_batch = s_t_batch.to(device)
                action_logits = self.model.forward(s_t_batch)
                action_logits = action_logits.reshape(len(s_t), -1)
                
                new_log_probs_list = []
                entropies_list = []
                for b in range(len(s_t)):
                    mask_b = s_t[b].mask
                    logits_b = action_logits[b, :len(mask_b)]
                    
                    # 마스크 적용
                    mask_b = mask_b.to(logits_b.device)
                    if logits_b.device.type == 'cuda':
                        logits_b = torch.where(mask_b, logits_b, torch.tensor(-1e9, device=logits_b.device))
                    else:
                        logits_b = logits_b.clone()
                        logits_b[~mask_b] = -1e9
                    
                    prob_b = F.softmax(logits_b, dim=0)
                    policy_b = Categorical(prob_b)
                    
                    # 기존 액션의 log_prob
                    new_log_prob_b = policy_b.log_prob(action_t[b])
                    entropy_b = policy_b.entropy()
                    
                    new_log_probs_list.append(new_log_prob_b)
                    entropies_list.append(entropy_b)
                
                new_log_prob_t = torch.stack(new_log_probs_list)
                entropy_t = torch.stack(entropies_list)
                
                # Value 재계산
                new_value_t = self.model.get_value(s_t)
                
                # PPO clipped objective
                ratio = torch.exp(new_log_prob_t - old_log_prob_t)
                surr1 = ratio * advantage_t
                surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantage_t
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = F.mse_loss(new_value_t, return_t)
                
                # Total loss
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_t.mean()
                
                # Backward (gradient 누적)
                loss.backward()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_t.mean().item()
        
        # 평균 계산
        num_updates = ppo_epochs * len(data['states'])
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        # 최종 reward 계산 (makespan)
        final_reward = rewards.sum(dim=0).mean().item()  # 모든 step reward의 합
        
        # 배치 정보
        batch_info = {
            'rewards': rewards.sum(dim=0).reshape(self.env_params['batch_size'], self.env_params['pomo_size']),
            'step_count': step_count,
            'N_I': env.N_I,
            'done_status': env.order_cleared.all(dim=1)
        }
        
        return avg_policy_loss, final_reward, step_count, batch_info
    
    def train_one_micro_batch(self):
        """REINFORCE 알고리즘을 사용한 학습"""
        # TODO: GNN 모델 추가 후 구현
        raise NotImplementedError("GNN 모델(SchedulingModel)을 추가한 후 REINFORCE 학습을 구현해주세요.")
        
        # Anomaly detection for debugging in-place operation issues
        torch.autograd.set_detect_anomaly(True)
        
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
        
        # 환경 초기화
        env = SchedulingEnv(self.env_params, debug_env=self.debug_env)
        self.env = env
        problem = generate_scheduling_data_batch(self.env_params)
        env._reset(problem)
        done = False
        s = env._get_state()
        log_prob_tmp = torch.zeros(self.env_params['batch_size'] * self.env_params['pomo_size']).to(self.model_device)
        
        # action_type을 모두 0으로 초기화 (Action Type 0: Item-Order->CP)
        #batch_size = self.env_params['batch_size'] * self.env_params['pomo_size']
        #_, action_type, action_mask = env.move_next_state(torch.arange(batch_size, dtype=torch.long))
        
        # 환경에서 받은 텐서들을 GPU로 이동
        '''
        if action_type is not None:
            action_type = action_type.to(self.device)
        if action_mask is not None:
            action_mask = action_mask.to(self.device)
        '''
 
        step_count = 0
        MAX_STEPS = 1000  # 타임아웃: 1000 steps 이상 소요 시 조기 종료
        
        # Entropy regularization 계수 확인
        entropy_coef = self.trainer_params.get('entropy_coef', 0.0)
        use_entropy = entropy_coef > 0
        
        # Shaped reward 및 entropy 누적용
        cumulative_reward = torch.zeros(self.env_params['batch_size'] * self.env_params['pomo_size'])
        if use_entropy:
            cumulative_entropy = torch.zeros(self.env_params['batch_size'] * self.env_params['pomo_size']).to(self.device)

        while not done:
            step_count += 1
            action, log_prob, entropy = self.model.get_action(s)
            log_prob_tmp += log_prob
            if use_entropy:
                cumulative_entropy += entropy  # Entropy 누적
            s, reward, done = env.step(action.to('cpu'))  # s 업데이트!

        # POMO 체크: -1 또는 1이면 POMO 사용 안함
        use_pomo = self.env_params['pomo_size'] > 1
        
        reward_reshape = reward.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
        
        # Deadlock 발생 체크 및 제외 (옵션)
        exclude_deadlock = self.trainer_params.get('exclude_deadlock_instances', False)
        valid_mask = None
        
        if exclude_deadlock:
            # deadlock_occurred를 (batch_size, pomo_size)로 reshape
            deadlock_reshape = env.deadlock_occurred.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
            # 각 배치별로 하나라도 deadlock이 발생하면 해당 배치의 모든 POMO 제외
            deadlock_per_batch = deadlock_reshape.any(dim=1)  # (batch_size,)
            valid_batches = ~deadlock_per_batch  # deadlock이 없는 배치들
            
            num_deadlock = deadlock_per_batch.sum().item()
            if num_deadlock > 0:
                print(f"   ⚠️ Deadlock 발생: {num_deadlock}개 배치 제외 (총 {self.env_params['batch_size']}개 중)")
                
            # valid한 배치만 선택
            reward_reshape = reward_reshape[valid_batches]  # (valid_batch_size, pomo_size)
            log_prob_tmp_reshape = log_prob_tmp.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
            log_prob_tmp_valid = log_prob_tmp_reshape[valid_batches].reshape(-1).to(self.device)  # (valid_batch_size * pomo_size,)
            
            if use_entropy:
                cumulative_entropy_reshape = cumulative_entropy.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
                cumulative_entropy = cumulative_entropy_reshape[valid_batches].reshape(-1)  # (valid_batch_size * pomo_size,)
            
            # valid한 배치가 없으면 학습 스킵
            if reward_reshape.shape[0] == 0:
                print(f"   ❌ 모든 배치에서 deadlock 발생! 이번 micro-batch는 학습하지 않습니다.")
                return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, step_count, batch_info
        else:
            log_prob_tmp_valid = log_prob_tmp
        
        # Baseline 타입에 따른 Advantage 계산
        baseline_type = self.trainer_params.get('baseline_type', 'pomo')
        normalize_advantage = self.trainer_params.get('normalize_advantage', True)
        
        if baseline_type == 'batch':
            # 배치 전체 baseline + 표준화 (B×K)
            print("   ℹ️ Batch Baseline: 배치 전체에서 baseline 계산 (B×K)")
            reward_flat = reward_reshape.reshape(-1)  # (B*K,)
            
            # Baseline = 배치 전체 평균
            advantage = reward_flat - reward_flat.mean()
            
            # 표준편차로 정규화 (옵션)
            if normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                print(f"     └─ 표준화 완료: mean=0, std=1")
        
        elif baseline_type == 'pomo' and use_pomo:
            # POMO baseline (인스턴스별)
            print("   ℹ️ POMO Baseline: 인스턴스별 POMO 평균 사용")
            advantage = reward_reshape - torch.mean(reward_reshape, -1, keepdim=True)
            
            # 표준편차로 정규화 (옵션)
            if normalize_advantage:
                advantage_std = torch.std(reward_reshape, -1, keepdim=True, unbiased=False)
                advantage = advantage / (advantage_std + 1e-8)
            
            advantage = advantage.reshape(-1)
        
        else:
            # Baseline 없음
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
        
        # 목적함수값 계산 (reward의 평균, valid한 배치만)
        obj_value = reward_reshape.mean().item()
        
        # 각 인스턴스별 simulation done 여부 확인 (모든 order가 cleared 되었는지)
        done_status = env.order_cleared.all(dim=1)  # (batch_size * pomo_size,) - 각 인스턴스별 done 여부
        
        # 배치 내 각 인스턴스별 정보 저장
        batch_info = {
            'rewards': reward_reshape,  # (batch_size, pomo_size)
            'step_count': step_count,
            'N_I': env.N_I,  # 각 배치별 아이템 개수 (batch_size * pomo_size,)
            'done_status': done_status  # 각 인스턴스별 simulation done 여부
        }

        return loss, obj_value, step_count, batch_info
    
    def train_one_micro_batch_sil(self):
        """
        Self-Imitation Learning (SIL) 알고리즘을 사용한 학습
        각 배치에서 최고 reward를 받은 경로만 선택하여 학습
        (Self-improvement for neural combinatorial optimization: sample without replacement, but improvement - Pirnay et al., 2024)
        """
        # TODO: GNN 모델 추가 후 구현
        raise NotImplementedError("GNN 모델(SchedulingModel)을 추가한 후 SIL 학습을 구현해주세요.")
        
        # Anomaly detection for debugging in-place operation issues
        torch.autograd.set_detect_anomaly(True)
        
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
        
        # 환경 초기화
        env = SchedulingEnv(self.env_params, debug_env=self.debug_env)
        self.env = env
        problem = generate_scheduling_data_batch(self.env_params)
        env._reset(problem)
        done = False
        s = env._get_state()
        log_prob_tmp = torch.zeros(self.env_params['batch_size'] * self.env_params['pomo_size']).to(self.model_device)
        
        step_count = 0
        MAX_STEPS = 1000  # 타임아웃: 1000 steps 이상 소요 시 조기 종료
        
        # Entropy regularization 계수 확인
        entropy_coef = self.trainer_params.get('entropy_coef', 0.0)
        use_entropy = entropy_coef > 0
        
        # Shaped reward 및 entropy 누적용
        if use_entropy:
            cumulative_entropy = torch.zeros(self.env_params['batch_size'] * self.env_params['pomo_size']).to(self.device)

        while not done:
            step_count += 1
            action, log_prob, entropy = self.model.get_action(s)
            log_prob_tmp += log_prob
            if use_entropy:
                cumulative_entropy += entropy  # Entropy 누적
            s, reward, done = env.step(action.to('cpu'))  # s 업데이트!

        # POMO 체크: -1 또는 1이면 POMO 사용 안함
        use_pomo = self.env_params['pomo_size'] > 1
        
        if not use_pomo:
            raise ValueError("SIL은 POMO_SIZE > 1일 때만 사용 가능합니다. POMO를 활성화하거나 다른 알고리즘을 선택하세요.")
        
        reward_reshape = reward.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
        log_prob_tmp_reshape = log_prob_tmp.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
        
        # Deadlock 발생 체크 및 제외 (옵션)
        exclude_deadlock = self.trainer_params.get('exclude_deadlock_instances', False)
        
        if exclude_deadlock:
            # deadlock_occurred를 (batch_size, pomo_size)로 reshape
            deadlock_reshape = env.deadlock_occurred.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
            # 각 배치별로 하나라도 deadlock이 발생하면 해당 배치의 모든 POMO 제외
            deadlock_per_batch = deadlock_reshape.any(dim=1)  # (batch_size,)
            valid_batches = ~deadlock_per_batch  # deadlock이 없는 배치들
            
            num_deadlock = deadlock_per_batch.sum().item()
            if num_deadlock > 0:
                print(f"   ⚠️ Deadlock 발생: {num_deadlock}개 배치 제외 (총 {self.env_params['batch_size']}개 중)")
                
            # valid한 배치만 선택
            reward_reshape = reward_reshape[valid_batches]  # (valid_batch_size, pomo_size)
            log_prob_tmp_reshape = log_prob_tmp_reshape[valid_batches]  # (valid_batch_size, pomo_size)
            
            if use_entropy:
                cumulative_entropy_reshape = cumulative_entropy.reshape(self.env_params['batch_size'], self.env_params['pomo_size'])
                cumulative_entropy_valid = cumulative_entropy_reshape[valid_batches]  # (valid_batch_size, pomo_size)
            
            # valid한 배치가 없으면 학습 스킵
            if reward_reshape.shape[0] == 0:
                print(f"   ❌ 모든 배치에서 deadlock 발생! 이번 micro-batch는 학습하지 않습니다.")
                batch_info = {
                    'rewards': torch.zeros(self.env_params['batch_size'], self.env_params['pomo_size']),
                    'step_count': step_count,
                    'N_I': env.N_I,
                    'done_status': env.order_cleared.all(dim=1)
                }
                return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, step_count, batch_info
        
        print("   ℹ️ SIL: 각 배치에서 최고 reward만 선택하여 학습")
        
        # Self-Imitation Learning: 각 배치에서 최고 reward를 받은 경로만 선택
        # reward는 -objective이므로, max가 가장 좋은 것
        best_reward, best_idx = reward_reshape.max(dim=1)  # (valid_batch_size,)
        
        # 각 배치의 best index에 해당하는 log_prob 선택
        batch_indices = torch.arange(reward_reshape.shape[0], device=self.device)
        best_log_probs = log_prob_tmp_reshape[batch_indices, best_idx]  # (valid_batch_size,)
        
        # Loss 계산: 최고 reward를 받은 경로의 log_prob만 사용
        loss = -best_log_probs.mean()
        
        # Entropy regularization 추가 (사용하는 경우에만)
        if use_entropy:
            best_entropy = cumulative_entropy_valid[batch_indices, best_idx]  # (valid_batch_size,)
            entropy_bonus = entropy_coef * best_entropy.mean()
            loss = loss - entropy_bonus  # Entropy bonus 추가 (음수 = 최대화)
        
        # 목적함수값 계산 (best reward의 평균)
        obj_value = best_reward.mean().item()
        
        # 각 인스턴스별 simulation done 여부 확인 (모든 order가 cleared 되었는지)
        done_status = env.order_cleared.all(dim=1)  # (batch_size * pomo_size,) - 각 인스턴스별 done 여부
        
        # 배치 내 각 인스턴스별 정보 저장
        batch_info = {
            'rewards': reward_reshape,  # (batch_size, pomo_size)
            'step_count': step_count,
            'N_I': env.N_I,  # 각 배치별 아이템 개수 (batch_size * pomo_size,)
            'done_status': done_status  # 각 인스턴스별 simulation done 여부
        }

        return loss, obj_value, step_count, batch_info
    
    def _eval_one_problem(self, mode='test', test_file_start=0, test_file_end=4, data_folder=None):
        """
        테스트 전용 함수: 저장된 테스트 데이터로 모델 평가
        학습 중에는 호출되지 않음
        
        Args:
            mode: 'test' (모델 평가)
            test_file_start: 테스트 시작 파일 번호
            test_file_end: 테스트 끝 파일 번호 (포함)
            data_folder: 데이터 폴더 경로 (기본값: './generated_datasets/data_test')
        """
        # TODO: GNN 모델 추가 후 구현
        raise NotImplementedError("GNN 모델(SchedulingModel)을 추가한 후 테스트를 구현해주세요.")
        
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
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 모델 가중치 로드 완료")
            
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
        고정된 validation 데이터셋 준비 (파일이 있으면 로드, 없으면 생성)
        모든 epoch에서 동일한 데이터로 평가하기 위함
        
        Args:
            validation_batch_size: Validation 배치 수
            validation_pomo_size: Validation POMO 크기 (보통 1)
        """
        # 시드 고정하여 재현 가능한 validation set 생성
        validation_seed = 2025
        
        # Validation 데이터 파일 경로
        val_data_folder = './generated_datasets/data_val'
        os.makedirs(val_data_folder, exist_ok=True)
        val_data_path = os.path.join(val_data_folder, f"validation_data_P{self.env_params['N_P']}_T{self.env_params['N_T']}_seed{validation_seed}.pickle")
        
        # 파일이 이미 존재하는지 확인
        if os.path.exists(val_data_path):
            print(f"📂 기존 Validation 데이터셋 로드 중...")
            print(f"   파일 경로: {val_data_path}")
            
            try:
                with open(val_data_path, 'rb') as f:
                    validation_data = pickle.load(f)
                
                # validation_problem 추출 (env_params와 validation_seed 제외)
                self.validation_problem = {k: v for k, v in validation_data.items() 
                                          if k not in ['env_params', 'validation_seed']}
                
                # 배치 크기 확인
                loaded_batch_size = validation_data.get('env_params', {}).get('batch_size', validation_batch_size)
                loaded_seed = validation_data.get('validation_seed', 'unknown')
                
                print(f"✅ Validation 데이터셋 로드 완료")
                print(f"   배치 크기: {loaded_batch_size}, 시드: {loaded_seed}")
                
                # 배치 크기가 다르면 경고
                if loaded_batch_size != validation_batch_size:
                    print(f"   ⚠️  요청한 배치 크기({validation_batch_size})와 파일의 배치 크기({loaded_batch_size})가 다릅니다.")
                    print(f"   파일의 배치 크기를 사용합니다.")
                
                return
                
            except Exception as e:
                print(f"   ❌ 파일 로드 실패: {e}")
                print(f"   새로운 Validation 데이터셋을 생성합니다.")
        
        # 파일이 없거나 로드 실패 시 새로 생성
        print(f"📦 새로운 Validation 데이터셋 생성 중... (배치 크기: {validation_batch_size})")
        
        # Validation 환경 파라미터 설정
        val_env_params = copy.deepcopy(self.env_params)
        val_env_params['batch_size'] = validation_batch_size
        val_env_params['pomo_size'] = validation_pomo_size
        
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
        
        # Validation 데이터 생성
        self.validation_problem = generate_scheduling_data_batch(val_env_params)
        
        # RNG 상태 복원
        random.setstate(original_random_state)
        np.random.set_state(original_np_state)
        torch.set_rng_state(original_torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(original_cuda_rng_state)
        
        # 저장할 데이터 구성
        validation_data = {
            'env_params': val_env_params,
            'validation_seed': validation_seed,  # 시드 정보도 저장
            **self.validation_problem
        }
        
        # 파일로 저장
        with open(val_data_path, 'wb') as f:
            pickle.dump(validation_data, f)
        
        print(f"✅ Validation 데이터셋 생성 완료 (고정 시드: {validation_seed})")
        print(f"💾 Validation 데이터셋 저장: {val_data_path}")
        print(f"   ℹ️  다음 학습부터는 이 파일을 재사용합니다.")
    
    def _eval_validation(self, validation_batch_size=10, validation_pomo_size=1):
        """
        Validation 평가 함수: 고정된 데이터로 모델 성능 평가 (Greedy policy)
        
        Args:
            validation_batch_size: Validation 배치 수
            validation_pomo_size: Validation POMO 크기 (보통 1)
        
        Returns:
            avg_score: 평균 validation score (목적함수값)
        """
        # TODO: GNN 모델 추가 후 구현
        raise NotImplementedError("GNN 모델(SchedulingModel)을 추가한 후 validation을 구현해주세요.")
        
        if self.validation_problem is None:
            raise RuntimeError("Validation 데이터셋이 준비되지 않았습니다. _prepare_validation_dataset()를 먼저 호출하세요.")
        
        self.model.eval()
        
        # Validation 환경 파라미터 설정
        val_env_params = copy.deepcopy(self.env_params)
        val_env_params['batch_size'] = validation_batch_size
        val_env_params['pomo_size'] = validation_pomo_size
        
        # 고정된 validation 데이터로 환경 초기화
        val_env = SchedulingEnv(val_env_params, debug_env=False)
        val_env._reset(self.validation_problem)
        
        done = False
        s = val_env._get_state()
        
        with torch.no_grad():
            while not done.all():
                # Greedy action 선택
                action = self.model.get_max_action(s)
                s, reward, done = val_env.step(action.to('cpu'))
        
        # 목적함수값 계산
        obj_value = val_env.get_objective()
        if isinstance(obj_value, torch.Tensor):
            obj_value = obj_value.mean().item()
        
        self.model.train()
        return obj_value

    def plot_score(self, valid_score, valid_loss):
        plt.figure()
        plt.title("result_valid_score_log")
        plt.plot(valid_score)
        plt.show()
        plt.figure()
        plt.title("valid_loss")
        plt.plot(valid_loss)
        plt.show()

