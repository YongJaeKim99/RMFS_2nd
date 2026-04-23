"""
RMFS RL Training Entry Point.
REINFORCE 및 PPO 알고리즘을 지원한다.
GATv2 Actor-Critic with graph state.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

from rmfs_trainer import RMFS_Trainer

if __name__ == "__main__":
    # =================================================================
    # 알고리즘 선택
    # =================================================================
    ALGORITHM_TYPE = 'ppo'  # 'reinforce' or 'ppo'

    # =================================================================
    # Action space 선택
    # =================================================================
    ACTION_TYPE = 'continuous_gaussian'  # 'discrete', 'continuous_beta', 'continuous_gaussian'

    # ------------------------------------------------------------------
    # 알고리즘별 기본 하이퍼파라미터
    # ------------------------------------------------------------------
    if ALGORITHM_TYPE == 'ppo':
        EPOCHS = 500
        BATCH_SIZE = 32
        VALIDATION_INTERVAL = 5
        VALIDATION_BATCH_SIZE = 20
        optimizer_params = {'optimizer': {'lr': 3e-4, 'weight_decay': 0}}
        USE_ENTROPY_REG = True
        ENTROPY_COEF = 0.01
        BASELINE_TYPE = 'none'
        NORMALIZE_ADVANTAGE = False
    else:  # 'reinforce'
        EPOCHS = 500
        BATCH_SIZE = 64
        VALIDATION_INTERVAL = 5
        VALIDATION_BATCH_SIZE = 20
        optimizer_params = {'optimizer': {'lr': 3e-4, 'weight_decay': 0}}
        USE_ENTROPY_REG = True
        ENTROPY_COEF = 0.01
        BASELINE_TYPE = 'batch'
        NORMALIZE_ADVANTAGE = True

    # ------------------------------------------------------------------
    # PPO 전용 파라미터
    # ------------------------------------------------------------------
    PPO_EPS_CLIP = 0.2
    PPO_K_EPOCHS = 4
    PPO_GAE_LAMBDA = 0.98
    PPO_GAMMA = 1.0
    PPO_VLOSS_COEF = 0.5
    PPO_PLOSS_COEF = 1.0
    PPO_TAU = 0.0
    PPO_MINIBATCH_SIZE = 4096
    N_RESAMPLE = 20
    PPO_ADV_NORM_TYPE = 'per_instance'

    # ------------------------------------------------------------------
    # Reward 방식
    # ------------------------------------------------------------------
    REWARD_TYPE = 'stepwise'  # 'stepwise': 매 step Pre_Makespan - Makespan (기본)
                               # 'sparse':   episode 끝에만 -Makespan

    # ------------------------------------------------------------------
    # Deadlock 방지: Stay(action 0) 강제 마스킹
    # ------------------------------------------------------------------
    FORCE_MASK_STAY = True  # True: action 0(WS 유지)을 항상 마스킹 → 반드시 Storage 반납
                            # False: 기존 동작 (Stay 항상 유효)

    # ------------------------------------------------------------------
    # 디버그: 매 step 로그 출력
    # ------------------------------------------------------------------
    DEBUG_LOG_STEPS = True  # True: 매 step마다 decisions/returned 카운트 출력

    # ------------------------------------------------------------------
    # 체크포인트 재개
    # ------------------------------------------------------------------
    RESUME_FROM_CHECKPOINT = None  # None or "path/to/checkpoint.pt"
    RESUME_TRAINING = False

    # Device
    DEVICE_MODE = 'gpu'  # RMFS env는 numpy 기반이라 CPU 권장

    # Wandb
    USE_WANDB = True
    WANDB_PROJECT = "RMFS"
    WANDB_RUN_NAME = None
    WANDB_RUN_ID = None
    WANDB_RESUME = None

    # Validation
    USE_VALIDATION = True

    # Seed
    USE_SEED_FIX = False
    SEED = 42

    # =================================================================
    # Device 설정
    # =================================================================
    cuda_available = torch.cuda.is_available()

    if DEVICE_MODE == 'cpu' or not cuda_available:
        device = 'cpu'
        device_desc = "CPU"
    else:
        device = 'cuda'
        device_desc = f"GPU ({torch.cuda.get_device_name(0)})"

    print(f"Device: {device_desc}")

    # =================================================================
    # RMFS 환경 파라미터
    # =================================================================
    env_params = {
        'batch_size': BATCH_SIZE,
        'block_rows': 3,       # Storage 블록 그리드 행 수
        'block_cols': 3,       # Storage 블록 그리드 열 수
        'block_h': 4,          # 블록 내부 행 수
        'block_w': 2,          # 블록 내부 열 수
        # N_S = block_rows * block_cols * block_h * block_w = 72 (자동 계산)
        'N_P': 40,             # Number of pods
        'N_R': 10,             # Number of robots
        'N_W': 4,              # Number of workstations
        'Total_PodTask': 60,   # Episode length (total pod tasks)
        'Unit_PT': 15,         # Unit processing time
        'ST': 5,               # Setup time
        'UT': 1,               # Unit travel time
        'Large': True,         # Large layout flag
        'seed_base': 0,        # Base seed for instance generation
        'force_mask_stay': FORCE_MASK_STAY,  # True: Stay 마스킹으로 deadlock 방지
        'action_type': ACTION_TYPE,
    }

    # =================================================================
    # 모델 파라미터 (GATv2 Actor-Critic)
    # =================================================================
    model_params = {
        'storage_feat_dim': 4,   # Storage node features (x, y, occupied, idle_time)
        'ws_feat_dim': 4,        # WS node features (x, y, occupied, idle_time)
        'd_edge': 9,             # Edge feature dimension
        'd_hidden': 64,          # GATv2 hidden dimension
        'n_gat_layers': 3,       # Number of GATv2 layers
        'n_heads': 4,            # Number of attention heads
        'dropout_prob': 0.0,     # Attention dropout
        'num_mlp_layers_actor': 2,
        'hidden_dim_actor': 64,
        'num_mlp_layers_critic': 2,
        'hidden_dim_critic': 64,
        'action_type': ACTION_TYPE,
    }

    # =================================================================
    # 트레이너 파라미터
    # =================================================================
    trainer_params = {
        'epochs': EPOCHS,
        'accumulation_steps': 1,   # REINFORCE gradient accumulation steps
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
        'device': device,
        'mode': 'train',
        'use_validation': USE_VALIDATION,
        'validation_interval': VALIDATION_INTERVAL,
        'validation_batch_size': VALIDATION_BATCH_SIZE,
        'resume_from_checkpoint': RESUME_FROM_CHECKPOINT,
        'resume_training': RESUME_TRAINING,
        # Algorithm type
        'algorithm_type': ALGORITHM_TYPE,
        # PPO params
        'eps_clip': PPO_EPS_CLIP,
        'k_epochs': PPO_K_EPOCHS,
        'gae_lambda': PPO_GAE_LAMBDA,
        'gamma': PPO_GAMMA,
        'vloss_coef': PPO_VLOSS_COEF,
        'ploss_coef': PPO_PLOSS_COEF,
        'tau': PPO_TAU,
        'ppo_minibatch_size': PPO_MINIBATCH_SIZE,
        'n_resample': N_RESAMPLE,
        'ppo_adv_norm_type': PPO_ADV_NORM_TYPE,
        'reward_type': REWARD_TYPE,
        'debug_log_steps': DEBUG_LOG_STEPS,
        'action_type': ACTION_TYPE,
    }

    # =================================================================
    # Trainer 생성 및 실행
    # =================================================================
    trainer = RMFS_Trainer(env_params, model_params, optimizer_params, trainer_params)

    N_S = env_params['block_rows'] * env_params['block_cols'] * env_params['block_h'] * env_params['block_w']
    N_W = env_params['N_W']

    print("=" * 60)
    print("RMFS RL Training")
    print("=" * 60)
    print(f"Algorithm: {ALGORITHM_TYPE.upper()}")
    if ALGORITHM_TYPE == 'ppo':
        print(f"  eps_clip={PPO_EPS_CLIP}, k_epochs={PPO_K_EPOCHS}, "
              f"gae_lambda={PPO_GAE_LAMBDA}")
        print(f"  gamma={PPO_GAMMA}, vloss_coef={PPO_VLOSS_COEF}, "
              f"ploss_coef={PPO_PLOSS_COEF}")
        print(f"  n_resample={N_RESAMPLE}, minibatch_size={PPO_MINIBATCH_SIZE}")
        print(f"  adv_norm_type={PPO_ADV_NORM_TYPE}")
    print(f"Action Type: {ACTION_TYPE}")
    print(f"Model: GATv2 Actor-Critic "
          f"(d_hidden={model_params['d_hidden']}, "
          f"n_layers={model_params['n_gat_layers']}, "
          f"n_heads={model_params['n_heads']})")
    if ACTION_TYPE == 'discrete':
        print(f"  Action space: {N_S + 1} (Stay + {N_S} storages)")
    else:
        print(f"  Action space: continuous (x, y) -> nearest available storage")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Reward Type: {REWARD_TYPE}")
    print(f"Entropy: {'ON' if USE_ENTROPY_REG else 'OFF'} "
          f"(coef={ENTROPY_COEF})")
    print(f"Device: {device_desc}")
    print(f"Validation: {'ON' if USE_VALIDATION else 'OFF'} "
          f"(interval={VALIDATION_INTERVAL})")
    print()
    print(f"RMFS Environment:")
    print(f"  Block={env_params['block_rows']}x{env_params['block_cols']} "
          f"({env_params['block_h']}x{env_params['block_w']}/block), "
          f"N_S={N_S}, N_P={env_params['N_P']}, "
          f"N_R={env_params['N_R']}, N_W={N_W}")
    print(f"  Total_PodTask={env_params['Total_PodTask']}")
    print(f"  Unit_PT={env_params['Unit_PT']}, ST={env_params['ST']}, "
          f"UT={env_params['UT']}")
    print(f"  Large={env_params['Large']}")
    print(f"  Force Mask Stay: {'ON' if FORCE_MASK_STAY else 'OFF'}")
    print("=" * 60)

    trainer.run()

    print("\nTraining complete!")
