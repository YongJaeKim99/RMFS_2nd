import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

from trainer import Scheduling_Trainer

if __name__ == "__main__":
    # =================================================================
    # 🎯 학습 파라미터 설정
    # =================================================================

    # ------------------------------------------------------------------
    # 알고리즘 선택 (가장 중요한 설정)
    # ------------------------------------------------------------------
    ALGORITHM_TYPE = 'reinforce'   # 'reinforce' or 'ppo'
    # 'reinforce': REINFORCE + POMO baseline (GAT / DANIEL 모두 가능)
    # 'ppo':       PPO-Clip + GAE (DANIEL 전용, 논문 알고리즘)

    # 모델 선택 옵션
    MODEL_TYPE = 'daniel'  # 'gat': 기존 GNN(GAT) 모델, 'daniel': DANIEL 모델
    # ※ PPO는 반드시 MODEL_TYPE='daniel' 이어야 합니다.

    # 목적함수 선택
    OBJECTIVE = 'tardiness'  # 'tardiness' or 'makespan'

    # PPO + 잘못된 모델 타입 조합 사전 차단
    if ALGORITHM_TYPE == 'ppo' and MODEL_TYPE != 'daniel':
        raise ValueError("PPO는 DANIEL 모델에서만 지원됩니다. MODEL_TYPE='daniel'로 설정하세요.")

    # ------------------------------------------------------------------
    # 알고리즘별 기본 하이퍼파라미터 (논문 값 기준)
    # ------------------------------------------------------------------
    if ALGORITHM_TYPE == 'ppo':
        EPOCHS                = 200       # 논문: max_updates = 1,000
        BATCH_SIZE            = 64       # 논문: num_envs = 20 → GPU 활용 위해 확장
        POMO_SIZE             = 1          # PPO 학습 시 기본 1
        VALIDATION_INTERVAL   = 5         # 논문: validate_timestep = 10
        VALIDATION_BATCH_SIZE = 50
        VALIDATION_POMO_SIZE  = 1
        optimizer_params = {'optimizer': {'lr': 3e-4, 'weight_decay': 1e-6}}  # 논문: 3×10⁻⁴
        USE_ENTROPY_REG     = True
        ENTROPY_COEF        = 0.01         # 논문: entloss_coef = 0.01
        BASELINE_TYPE       = 'none'       # PPO는 value function이 baseline
        NORMALIZE_ADVANTAGE = False        # GAE 내부에서 정규화
    else:  # 'reinforce'
        EPOCHS                = 1000       # ← 동작 확인용 (원래: 200)
        BATCH_SIZE            = 8
        POMO_SIZE             = 8
        VALIDATION_INTERVAL   = 5          # ← 매 epoch 확인 (원래: 5)
        VALIDATION_BATCH_SIZE = 50          # ← 동작 확인용 (원래: 50)
        VALIDATION_POMO_SIZE  = 1
        optimizer_params = {'optimizer': {'lr': 3e-5, 'weight_decay': 1e-6}}
        USE_ENTROPY_REG     = False
        ENTROPY_COEF        = 0.01
        BASELINE_TYPE       = 'pomo'
        NORMALIZE_ADVANTAGE = False

    # ------------------------------------------------------------------
    # PPO 전용 파라미터 (논문 값)
    # ------------------------------------------------------------------
    PPO_EPS_CLIP       = 0.2        # 논문: clipping parameter ε
    PPO_K_EPOCHS       = 4          # 논문: 에피소드당 업데이트 epoch 수 K
    PPO_GAE_LAMBDA     = 0.98       # 논문: GAE parameter λ
    PPO_GAMMA          = 1.0        # 논문: discount factor γ
    PPO_VLOSS_COEF     = 0.5        # 논문: value loss 계수
    PPO_PLOSS_COEF     = 1.0        # 논문: policy loss 계수
    PPO_TAU            = 0.0        # 0.0 = hard copy (standard PPO), >0 = soft update
    PPO_MINIBATCH_SIZE = 4096       # 논문: 1024 → GPU 활용 위해 확장
    N_RESAMPLE         = 20         # 논문: 학습 데이터 리샘플링 주기 N_r

    # 체크포인트 재개 옵션
    RESUME_FROM_CHECKPOINT = None  # None: 처음부터 학습, "path/to/checkpoint.pt": 체크포인트에서 이어서 학습
    RESUME_TRAINING = False  # True: epoch 번호도 이어받기, False: epoch는 0부터 시작 (가중치만 로드)

    # Device 옵션
    #DEVICE_MODE = 'cpu'  # 'cpu', 'hybrid', 'gpu'
    DEVICE_MODE = 'gpu'  # 'cpu', 'hybrid', 'gpu'
    #DEVICE_MODE = 'hybrid'  # 'cpu', 'hybrid', 'gpu'
    # 'cpu': 전부 CPU, 'hybrid': 모델/학습은 GPU + 환경은 CPU, 'gpu': 전부 GPU

    # Wandb 옵션
    USE_WANDB = True    # ← 동작 확인용 (원래: True)
    WANDB_PROJECT = "RCMPSP"
    WANDB_RUN_NAME = None
    WANDB_RUN_ID = None
    WANDB_RESUME = None

    # Validation 사용 여부
    USE_VALIDATION = True

    # Wait / Dominance 옵션 (DANIEL action space)
    ALLOW_WAIT_RELEASE = False   # True: release time 미도래 activity도 대기 후 스케줄 허용
    ALLOW_WAIT_MUTEX = False     # True: mutex 파트너 실행 중인 activity도 대기 후 스케줄 허용
    DOMINANCE_RULE = False       # True: 대기 pair의 dominance 필터링

    # Reward 방식 선택
    #REWARD_TYPE = 'stepwise'  # 'sparse': episode 끝에만 reward (기본)
                            # 'stepwise': 매 step마다 dense reward (DANIEL 논문 방식, tardiness 전용)
    REWARD_TYPE = 'sparse'  # 'sparse': episode 끝에만 reward (기본)

    # 디버그 옵션
    DEBUG_ENV = False
    DEBUG_MODEL = False
    STEP_PROGRESS_LOG = False  # True: 매 step마다 "Ended: X / Y" 출력

    # 로깅 상세도 옵션
    VERBOSE_LOGGING = False  # True: 각 batch/POMO별 목적함수와 started/ended 출력
                              # False: accumulation마다 평균값만, epoch 끝날 때만 출력

    # =================================================================
    # GPU/CPU 설정 및 검증
    # =================================================================
    cuda_available = torch.cuda.is_available()

    if not cuda_available and DEVICE_MODE != 'cpu':
        print(f"⚠️ GPU가 사용 불가능합니다. DEVICE_MODE를 'cpu'로 변경합니다.")
        DEVICE_MODE = 'cpu'

    # Device mode 설정
    if DEVICE_MODE == 'cpu':
        device = 'cpu'
        device_desc = "CPU"
    elif DEVICE_MODE == 'hybrid':
        device = 'cuda'
        device_desc = "GPU (Hybrid Mode)"
    elif DEVICE_MODE == 'gpu':
        device = 'cuda'
        device_desc = "GPU"
    else:
        raise ValueError(f"Invalid DEVICE_MODE: {DEVICE_MODE}")

    print(f"🖥️  Device Mode: {DEVICE_MODE}")
    if cuda_available:
        print(f"   ├─ CUDA Available: Yes (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"   ├─ CUDA Available: No")
    print(f"   └─ Using: {device_desc}")
    print()

    # 시드 고정 옵션
    USE_SEED_FIX = False
    SEED = 42

    # 학습 데이터 생성 시드 고정 옵션
    TRAINING_SEED_FIX = False
    TRAINING_SEED = 1234

    # =================================================================
    # 환경 파라미터 설정 (Multi-Project Scheduling)
    # =================================================================

    # 논문 수준 사이즈 (FJSP 10x5 대응: 10 Projects, 5 Teams, ~50 Activities)
    env_params = {
         'batch_size': BATCH_SIZE,
         'pomo_size': POMO_SIZE,
         'N_P': 10,  # 프로젝트 수 (논문: n_j=10)
         'N_A_min': 4,  # 프로젝트당 최소 activity 수 (총 ~50, 논문: n_op=50)
         'N_A_max': 6,  # 프로젝트당 최대 activity 수
         'N_T': 5,  # 팀 수 (논문: n_m=5)
         'duration_min': 1,  # 최소 작업 시간 (논문: low=1)
         'duration_max': 99,  # 최대 작업 시간 (논문: high=99)
         'precedence_prob': 0.3,  # 선행 관계 생성 확률
         'mutex_prob': 0.03,  # 동시 불가 생성 확률
         'max_preds': 5,   # activity당 최대 선행 작업 수 (tensor 패딩 크기)
         'max_succs': 5,   # activity당 최대 후행 작업 수 (tensor 패딩 크기)
         'max_mutex': 2,  # activity당 최대 동시 불가 작업 수 (tensor 패딩 크기)
         'eligible_teams_ratio': 0.6,  # 평균 eligible 팀 비율
         'due_date_tightness': 1.3,  # Due date 여유도 (1.0 = tight, 1.5 = loose)
         'objective': OBJECTIVE,
         'debug_env': DEBUG_ENV,
         'state_mode': 'daniel' if MODEL_TYPE == 'daniel' else 'pyg',
         'step_log': STEP_PROGRESS_LOG,
         'allow_wait_release': ALLOW_WAIT_RELEASE,
         'allow_wait_mutex': ALLOW_WAIT_MUTEX,
         'dominance_rule': DOMINANCE_RULE,
         'reward_type': REWARD_TYPE,
    }
    '''
    # 큰 사이즈
    env_params = {
        'batch_size': BATCH_SIZE,
        'pomo_size': POMO_SIZE,
        'N_P': 10,  # 프로젝트 수 (5 → 10)
        'N_A_min': 10,  # 프로젝트당 최소 activity 수 (4 → 10)
        'N_A_max': 20,  # 프로젝트당 최대 activity 수 (6 → 20)
        'N_T': 8,  # 팀 수 (4 → 8)
        'duration_min': 1,  # 최소 작업 시간 (2 → 1)
        'duration_max': 10,  # 최대 작업 시간 (6 → 10)
        'precedence_prob': 0.3,  # 선행 관계 생성 확률
        'mutex_prob': 0.1,  # 동시 불가 생성 확률
        'max_preds': 5,   # activity당 최대 선행 작업 수
        'max_succs': 5,   # activity당 최대 후행 작업 수
        'max_mutex': 10,  # activity당 최대 동시 불가 작업 수
        'eligible_teams_ratio': 0.6,  # 평균 eligible 팀 비율
        'due_date_tightness': 1.3,  # Due date 여유도 (1.0 = tight, 1.5 = loose)
        'objective': OBJECTIVE,
        'debug_env': DEBUG_ENV,
    }
    '''
    # 모델 파라미터 설정
    if MODEL_TYPE == 'gat':
        model_params = {
            'embedding_dim': 128,
            'num_head': 8,
            'num_encoder_layer': 3,
            'input_dim': 8,  # 패딩 방식: Activity(4) + Team(1) + Project(3) = 8
        }
    elif MODEL_TYPE == 'daniel':
         # 논문 원본 파라미터 (DANIEL, Tesla T4 기준, ~28K params)
        model_params = {
             'fea_act_input_dim': 10,
             'fea_team_input_dim': 8,
             'num_heads_AAB': [4, 4],
             'num_heads_TAB': [4, 4],
             'layer_fea_output_dim': [32, 8],
             'dropout_prob': 0.0,
             'num_mlp_layers_actor': 3,
             'hidden_dim_actor': 64,
             'num_mlp_layers_critic': 3,
             'hidden_dim_critic': 64,
        }
        # 큰 사이즈 파라미터 (~300K params)
        '''
        model_params = {
            # DAN (Dual Attention Network) 파라미터
            'fea_act_input_dim': 10,    # Activity 피처 차원 (env 출력과 일치)
            'fea_team_input_dim': 8,    # Team 피처 차원 (env 출력과 일치)
            'num_heads_AAB': [8, 8, 8],    # Activity Attention Block 헤드 수 (3층)
            'num_heads_TAB': [8, 8, 8],    # Team Attention Block 헤드 수 (3층)
            'layer_fea_output_dim': [128, 64, 32],  # DAN 레이어 출력 차원 (3층)
            'dropout_prob': 0.0,
            # Actor-Critic MLP 파라미터
            'num_mlp_layers_actor': 3,
            'hidden_dim_actor': 256,
            'num_mlp_layers_critic': 3,
            'hidden_dim_critic': 256,
        }
        '''
    else:
        raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Use 'gat' or 'daniel'.")

    # 트레이너 파라미터 설정
    trainer_params = {
        'epochs': EPOCHS,
        'accumulation_steps': 8,
        'grad_clip_norm': 1.0,
        'entropy_coef': ENTROPY_COEF if USE_ENTROPY_REG else 0.0,
        'baseline_type': BASELINE_TYPE,
        'normalize_advantage': NORMALIZE_ADVANTAGE,
        'use_wandb': USE_WANDB,
        'wandb_project': WANDB_PROJECT if USE_WANDB else None,
        'wandb_run_name': WANDB_RUN_NAME,
        'wandb_run_id': WANDB_RUN_ID,
        'wandb_resume': WANDB_RESUME,
        'seed': SEED if USE_SEED_FIX else None,
        'training_seed_fix': TRAINING_SEED_FIX,
        'training_seed': TRAINING_SEED if TRAINING_SEED_FIX else None,
        'device': device,
        'device_mode': DEVICE_MODE,
        'mode': 'train',
        'debug_env': DEBUG_ENV,
        'debug_model': DEBUG_MODEL,
        'verbose_logging': VERBOSE_LOGGING,
        'use_validation': USE_VALIDATION,
        'validation_interval': VALIDATION_INTERVAL,
        'validation_batch_size': VALIDATION_BATCH_SIZE,
        'validation_pomo_size': VALIDATION_POMO_SIZE,
        'resume_from_checkpoint': RESUME_FROM_CHECKPOINT,
        'resume_training': RESUME_TRAINING,
        'model_type': MODEL_TYPE,
        # 알고리즘 타입
        'algorithm_type': ALGORITHM_TYPE,
        # PPO 전용 파라미터 (REINFORCE 시에도 전달되지만 무시됨)
        'eps_clip': PPO_EPS_CLIP,
        'k_epochs': PPO_K_EPOCHS,
        'gae_lambda': PPO_GAE_LAMBDA,
        'gamma': PPO_GAMMA,
        'vloss_coef': PPO_VLOSS_COEF,
        'ploss_coef': PPO_PLOSS_COEF,
        'tau': PPO_TAU,
        'ppo_minibatch_size': PPO_MINIBATCH_SIZE,
        'n_resample': N_RESAMPLE,
        'reward_type': REWARD_TYPE,
    }

    # 트레이너 생성
    trainer = Scheduling_Trainer(env_params, model_params, optimizer_params, trainer_params)

    print("="*60)
    print("🚀 RCMPSP RL 학습 시작")
    print("="*60)
    if RESUME_FROM_CHECKPOINT:
        print(f"📂 체크포인트 재개: {RESUME_FROM_CHECKPOINT}")
        print(f"   └─ Epoch 이어받기: {'ON' if RESUME_TRAINING else 'OFF'}")

    print(f"Algorithm: {ALGORITHM_TYPE.upper()}")
    if ALGORITHM_TYPE == 'ppo':
        print(f"  └─ eps_clip={PPO_EPS_CLIP}, k_epochs={PPO_K_EPOCHS}, gae_lambda={PPO_GAE_LAMBDA}")
        print(f"  └─ gamma={PPO_GAMMA}, vloss_coef={PPO_VLOSS_COEF}, ploss_coef={PPO_PLOSS_COEF}")
        print(f"  └─ n_resample={N_RESAMPLE}, ppo_minibatch_size={PPO_MINIBATCH_SIZE}, tau={PPO_TAU}")
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Objective: {OBJECTIVE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")

    if POMO_SIZE <= 1:
        print(f"POMO Size: {POMO_SIZE} (POMO 미사용)")
    else:
        print(f"POMO Size: {POMO_SIZE}")

    if ALGORITHM_TYPE == 'reinforce':
        print(f"Accumulation Steps: {trainer_params['accumulation_steps']}")

    print(f"\n문제 설정:")
    print(f"  └─ 프로젝트 수: {env_params['N_P']}")
    print(f"  └─ 프로젝트당 Activity 수: {env_params['N_A_min']}-{env_params['N_A_max']}")
    print(f"  └─ 팀 수: {env_params['N_T']}")
    print(f"  └─ 작업 시간 범위: {env_params['duration_min']}-{env_params['duration_max']}")
    print(f"  └─ 선행 관계 확률: {env_params['precedence_prob']}")
    print(f"  └─ 동시 불가 확률: {env_params['mutex_prob']}")
    print(f"  └─ Due Date Tightness: {env_params['due_date_tightness']}")

    print(f"\nReward Type: {REWARD_TYPE}")
    if REWARD_TYPE == 'stepwise':
        print(f"  └─ Dense reward: r_t = est_tardiness(s_t) - est_tardiness(s_t+1)")

    print(f"Entropy Regularization: {'ON' if USE_ENTROPY_REG else 'OFF'}")
    if USE_ENTROPY_REG:
        print(f"  └─ Entropy Coef: {ENTROPY_COEF}")

    if ALGORITHM_TYPE == 'reinforce':
        print(f"Baseline Type: {BASELINE_TYPE}")
        print(f"Advantage Normalization: {'ON' if NORMALIZE_ADVANTAGE else 'OFF'}")
    print(f"Device Mode: {DEVICE_MODE}")
    print(f"  └─ Device: {device_desc}")

    print(f"Global Seed Fix: {USE_SEED_FIX}")
    if USE_SEED_FIX:
        print(f"  └─ Global Seed: {SEED}")

    print(f"Training Data Seed Fix: {'ON' if TRAINING_SEED_FIX else 'OFF'}")
    if TRAINING_SEED_FIX:
        print(f"  └─ Training Seed: {TRAINING_SEED}")

    print(f"Wandb: {USE_WANDB}")
    if USE_WANDB:
        print(f"  └─ Wandb Project: {WANDB_PROJECT}")

    print(f"Validation: {'ON' if USE_VALIDATION else 'OFF'}")
    if USE_VALIDATION:
        print(f"  └─ Validation Interval: {VALIDATION_INTERVAL} epochs")
        print(f"  └─ Validation Batch Size: {VALIDATION_BATCH_SIZE}")

    print(f"Debug ENV: {DEBUG_ENV}, Debug Model: {DEBUG_MODEL}")
    print(f"Verbose Logging: {'ON' if VERBOSE_LOGGING else 'OFF'}")
    if not VERBOSE_LOGGING:
        print(f"  └─ 간략 모드: accumulation마다 평균값만, epoch 끝날 때만 출력")
    print("="*60)

    # 학습 실행
    trainer.run()

    print("\n✅ 학습 완료!")
