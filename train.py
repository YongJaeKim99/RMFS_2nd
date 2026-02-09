import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

from trainer import Scheduling_Trainer

if __name__ == "__main__":
    # =================================================================
    # 🎯 학습 파라미터 설정
    # =================================================================

    # Baseline 및 Advantage Normalization 옵션
    BASELINE_TYPE = 'pomo'  # 'pomo': POMO 내 baseline, 'batch': 배치 전체 baseline, 'none': baseline 없음
    NORMALIZE_ADVANTAGE = False  # True: Advantage 정규화, False: 정규화 안 함

    # 체크포인트 재개 옵션
    RESUME_FROM_CHECKPOINT = None  # None: 처음부터 학습, "path/to/checkpoint.pt": 체크포인트에서 이어서 학습
    RESUME_TRAINING = False  # True: epoch 번호도 이어받기, False: epoch는 0부터 시작 (가중치만 로드)
    
    # Device 옵션
    DEVICE_MODE = 'hybrid'  # 'cpu', 'hybrid', 'gpu'
    # 'cpu': 전부 CPU, 'hybrid': 모델/학습은 GPU + 환경은 CPU, 'gpu': 전부 GPU
            
    # 기본 학습 파라미터
    EPOCHS = 500
    BATCH_SIZE = 16
    POMO_SIZE = 8  # -1 또는 1로 설정하면 POMO 미사용

    # Wandb 옵션
    USE_WANDB = False
    WANDB_PROJECT = "RCMPSP"  # Resource-Constrained Multi-Project Scheduling Problem
    WANDB_RUN_NAME = None
    WANDB_RUN_ID = None
    WANDB_RESUME = None
    
    # Validation 옵션
    USE_VALIDATION = False
    VALIDATION_INTERVAL = 10  # Validation 수행 주기 (epoch 단위)
    VALIDATION_BATCH_SIZE = 50
    VALIDATION_POMO_SIZE = 1
    
    # 옵티마이저 파라미터
    optimizer_params = {
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 1e-6,
        }
    }    
    
    # Entropy Regularization 옵션
    USE_ENTROPY_REG = False
    ENTROPY_COEF = 0.01
    
    # 학습 알고리즘 선택
    ALGORITHM = 'REINFORCE'  # 'REINFORCE' or 'PPO'
    
    # REINFORCE Loss 타입
    RL_LOSS_TYPE = 'standard'  # 'standard' or 'sil'
    
    # PPO 하이퍼파라미터
    PPO_EPOCHS = 4
    PPO_CLIP = 0.2
    PPO_VALUE_COEF = 0.5
    PPO_ENTROPY_COEF = 0.01
    
    # 목적함수 선택
    OBJECTIVE = 'tardiness'  # 'tardiness' or 'makespan'
    
    # 디버그 옵션
    DEBUG_ENV = False
    DEBUG_MODEL = False
    
    # =================================================================
    # GPU/CPU 설정 및 검증
    # =================================================================
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available and DEVICE_MODE != 'cpu':
        print(f"⚠️ GPU가 사용 불가능합니다. DEVICE_MODE를 'cpu'로 변경합니다.")
        DEVICE_MODE = 'cpu'
    
    # Device mode 설정
    if DEVICE_MODE == 'cpu':
        model_device = 'cpu'
        env_device = 'cpu'
        device_desc = "전부 CPU"
    elif DEVICE_MODE == 'hybrid':
        model_device = 'cuda'
        env_device = 'cpu'
        device_desc = "모델/학습=GPU, 환경=CPU"
    elif DEVICE_MODE == 'gpu':
        model_device = 'cuda'
        env_device = 'cuda'
        device_desc = "전부 GPU"
    else:
        raise ValueError(f"Invalid DEVICE_MODE: {DEVICE_MODE}")
    
    print(f"🖥️  Device Mode: {DEVICE_MODE} ({device_desc})")
    if cuda_available:
        print(f"   ├─ CUDA Available: Yes (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"   ├─ CUDA Available: No")
    print(f"   ├─ Model/Training Device: {model_device}")
    print(f"   └─ Environment Device: {env_device}")
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
    
    env_params = {
        'batch_size': BATCH_SIZE,
        'pomo_size': POMO_SIZE,
        'N_P': 5,  # 프로젝트 수
        'N_A_min': 4,  # 프로젝트당 최소 activity 수
        'N_A_max': 6,  # 프로젝트당 최대 activity 수
        'N_T': 4,  # 팀 수
        'duration_min': 2,  # 최소 작업 시간
        'duration_max': 6,  # 최대 작업 시간
        'precedence_prob': 0.3,  # 선행 관계 생성 확률
        'mutex_prob': 0.1,  # 동시 불가 생성 확률
        'eligible_teams_ratio': 0.6,  # 평균 eligible 팀 비율
        'due_date_tightness': 1.3,  # Due date 여유도 (1.0 = tight, 1.5 = loose)
        'objective': OBJECTIVE,
        'debug_env': DEBUG_ENV,
    }
    
    # 모델 파라미터 설정
    model_params = {
        'embedding_dim': 128,
        'num_head': 8,
        'num_encoder_layer': 3,
        'input_dim': 10,
    }
    
    # 트레이너 파라미터 설정
    trainer_params = {
        'epochs': EPOCHS,
        'accumulation_steps': 1,
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
        'device_mode': DEVICE_MODE,
        'model_device': model_device,
        'env_device': env_device,
        'mode': 'train',
        'debug_env': DEBUG_ENV,
        'debug_model': DEBUG_MODEL,
        'algorithm': ALGORITHM,
        'rl_loss_type': RL_LOSS_TYPE,
        'ppo_epochs': PPO_EPOCHS,
        'ppo_clip': PPO_CLIP,
        'ppo_value_coef': PPO_VALUE_COEF,
        'ppo_entropy_coef': PPO_ENTROPY_COEF,
        'use_validation': USE_VALIDATION,
        'validation_interval': VALIDATION_INTERVAL,
        'validation_batch_size': VALIDATION_BATCH_SIZE,
        'validation_pomo_size': VALIDATION_POMO_SIZE,
        'resume_from_checkpoint': RESUME_FROM_CHECKPOINT,
        'resume_training': RESUME_TRAINING,
    }
    
    # 트레이너 생성
    trainer = Scheduling_Trainer(env_params, model_params, optimizer_params, trainer_params)
    
    print("="*60)
    print("🚀 RCMPSP RL 학습 시작")
    print("="*60)
    if RESUME_FROM_CHECKPOINT:
        print(f"📂 체크포인트 재개: {RESUME_FROM_CHECKPOINT}")
        print(f"   └─ Epoch 이어받기: {'ON' if RESUME_TRAINING else 'OFF'}")
    
    print(f"Algorithm: {ALGORITHM}")
    if ALGORITHM == 'REINFORCE':
        print(f"  └─ RL Loss Type: {RL_LOSS_TYPE}")
    print(f"Objective: {OBJECTIVE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    if POMO_SIZE <= 1:
        print(f"POMO Size: {POMO_SIZE} (POMO 미사용)")
    else:
        print(f"POMO Size: {POMO_SIZE}")
    
    print(f"Accumulation Steps: {trainer_params['accumulation_steps']}")
    
    if ALGORITHM == 'PPO':
        print(f"PPO Epochs: {PPO_EPOCHS}")
        print(f"PPO Clip: {PPO_CLIP}")
        print(f"PPO Value Coef: {PPO_VALUE_COEF}")
        print(f"PPO Entropy Coef: {PPO_ENTROPY_COEF}")
    
    print(f"\n문제 설정:")
    print(f"  └─ 프로젝트 수: {env_params['N_P']}")
    print(f"  └─ 프로젝트당 Activity 수: {env_params['N_A_min']}-{env_params['N_A_max']}")
    print(f"  └─ 팀 수: {env_params['N_T']}")
    print(f"  └─ 작업 시간 범위: {env_params['duration_min']}-{env_params['duration_max']}")
    print(f"  └─ 선행 관계 확률: {env_params['precedence_prob']}")
    print(f"  └─ 동시 불가 확률: {env_params['mutex_prob']}")
    print(f"  └─ Due Date Tightness: {env_params['due_date_tightness']}")
    
    print(f"\nEntropy Regularization: {'ON' if USE_ENTROPY_REG else 'OFF'}")
    if USE_ENTROPY_REG:
        print(f"  └─ Entropy Coef: {ENTROPY_COEF}")
    
    print(f"Baseline Type: {BASELINE_TYPE}")
    print(f"Advantage Normalization: {'ON' if NORMALIZE_ADVANTAGE else 'OFF'}")
    print(f"Device Mode: {DEVICE_MODE} ({device_desc})")
    print(f"  └─ Model/Training: {model_device}, Environment: {env_device}")
    
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
    print("="*60)
    
    # 학습 실행
    trainer.run()
    
    print("\n✅ 학습 완료!")
