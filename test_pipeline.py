"""
RCMPSP 전체 파이프라인 테스트
환경, 모델, 학습이 정상적으로 작동하는지 확인
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

print("="*60)
print("RCMPSP 파이프라인 테스트 시작")
print("="*60)

# 1. 데이터 생성 테스트
print("\n[1/5] 데이터 생성 테스트...")
from data_generator import generate_scheduling_data_batch

env_params = {
    'batch_size': 2,
    'pomo_size': 1,
    'N_P': 3,  # 프로젝트 수
    'N_A_min': 3,  # 프로젝트당 최소 activity 수
    'N_A_max': 5,  # 프로젝트당 최대 activity 수
    'N_T': 3,  # 팀 수
    'duration_min': 2,
    'duration_max': 5,
    'precedence_prob': 0.3,
    'mutex_prob': 0.1,
    'eligible_teams_ratio': 0.6,
    'due_date_tightness': 1.3,
    'objective': 'tardiness',
}

problem = generate_scheduling_data_batch(env_params)
print(f"✅ 데이터 생성 성공")
print(f"   - 배치 크기: {env_params['batch_size']}")
print(f"   - 프로젝트 수: {env_params['N_P']}")
print(f"   - 최대 Activity 수: {problem['env_params']['max_N_A']}")

# 2. 환경 초기화 테스트
print("\n[2/5] 환경 초기화 테스트...")
from scheduling_env import SchedulingEnv

env = SchedulingEnv(env_params, debug_env=False)
env._reset(problem)
print(f"✅ 환경 초기화 성공")

# 3. State 생성 테스트
print("\n[3/5] State 생성 테스트...")
state = env._get_state()
print(f"✅ State 생성 성공")
print(f"   - State 개수: {len(state)}")
print(f"   - Batch 0 노드 수: {state[0].x.shape[0]}")
print(f"   - Batch 0 엣지 수: {state[0].edge_index.shape[1]}")
print(f"   - Batch 0 Action mask 크기: {state[0].mask.shape[0]}")

# 4. 모델 생성 테스트
print("\n[4/5] 모델 생성 테스트...")
from gnn_model import SchedulingModel

model_params = {
    'embedding_dim': 128,
    'num_head': 8,
    'num_encoder_layer': 3,
    'input_dim': 10,
    'gat_version': 'v2',
    'use_hetero_conv': False,  # 일단 기본 모델로 테스트
    'N_T': env_params['N_T'],
    'N_P': env_params['N_P'],
}

model = SchedulingModel(model_params, debug_model=False)
print(f"✅ 모델 생성 성공")
print(f"   - 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# 5. Forward pass 테스트
print("\n[5/5] Forward pass 테스트...")
try:
    # Action 샘플링 테스트
    action, log_prob, entropy = model.get_action(state)
    print(f"✅ Action 샘플링 성공")
    print(f"   - Action: {action}")
    print(f"   - Log Prob: {log_prob}")
    print(f"   - Entropy: {entropy}")
    
    # Greedy action 테스트
    greedy_action = model.get_max_action(state)
    print(f"✅ Greedy Action 선택 성공")
    print(f"   - Greedy Action: {greedy_action}")
    
    # Value 계산 테스트
    value = model.get_value(state)
    print(f"✅ Value 계산 성공")
    print(f"   - Value: {value}")
    
except Exception as e:
    print(f"❌ Forward pass 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. 환경 Step 테스트
print("\n[6/6] 환경 Step 테스트...")
try:
    next_state, reward, done = env.step(action.to('cpu'))
    print(f"✅ Step 성공")
    print(f"   - Reward: {reward}")
    print(f"   - Done: {done}")
    print(f"   - Sim Time: {env.sim_time}")
except Exception as e:
    print(f"❌ Step 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✅ 전체 파이프라인 테스트 완료!")
print("="*60)
print("\n모든 컴포넌트가 정상적으로 작동합니다.")
print("이제 train.py를 실행하여 학습을 시작하세요!")
