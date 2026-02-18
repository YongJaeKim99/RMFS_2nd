import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
from data_generator import generate_scheduling_data_batch

env_params = {
    'N_P': 10, 'N_A_min': 4, 'N_A_max': 6, 'N_T': 5,
    'duration_min': 1, 'duration_max': 99,
    'precedence_prob': 0.3, 'mutex_prob': 0.03,
    'max_preds': 50, 'max_succs': 50,
    'max_mutex': 50,
    'eligible_teams_ratio': 0.6,
    'due_date_tightness': 1.3,
    'objective': 'tardiness',
}

torch.manual_seed(42)

all_n_preds = []
all_n_succs = []

batch_size = 64
n_batches = 80  # 64*80 = 5120 instances
print("Generating data...", flush=True)
for i in range(n_batches):
    prob, _ = generate_scheduling_data_batch(batch_size, env_params)
    pred_idx = prob['activity_predecessors']
    succ_idx = prob['activity_successors']
    n_preds = (pred_idx >= 0).sum(dim=2).flatten().tolist()
    n_succs = (succ_idx >= 0).sum(dim=2).flatten().tolist()
    all_n_preds.extend(n_preds)
    all_n_succs.extend(n_succs)

p = np.array(all_n_preds)
s = np.array(all_n_succs)

print(f'\n=== {batch_size*n_batches} instances (N_P=10, N_A=4~6, precedence_prob=0.3) ===')
print(f'총 activity 샘플 수: {len(p):,}')
print()
print('선행자(pred) 개수 분포 (activity 단위):')
for v in range(int(p.max())+1):
    cnt = (p == v).sum()
    pct = cnt/len(p)*100
    bar = '#' * int(pct)
    print(f'  {v:2d}개: {cnt:7d} ({pct:5.1f}%) {bar}')
print(f'  → max={int(p.max())}, mean={p.mean():.2f}, p95={np.percentile(p,95):.0f}, p99={np.percentile(p,99):.0f}, p100={np.percentile(p,100):.0f}')

print()
print('후행자(succ) 개수 분포 (activity 단위):')
for v in range(int(s.max())+1):
    cnt = (s == v).sum()
    pct = cnt/len(s)*100
    bar = '#' * int(pct)
    print(f'  {v:2d}개: {cnt:7d} ({pct:5.1f}%) {bar}')
print(f'  → max={int(s.max())}, mean={s.mean():.2f}, p95={np.percentile(s,95):.0f}, p99={np.percentile(s,99):.0f}, p100={np.percentile(s,100):.0f}')

print()
print('현재 max_preds=5 설정 시 truncation 비율:')
print(f'  pred > 5: {(p > 5).sum()} / {len(p)} = {(p > 5).mean()*100:.3f}%')
print(f'  succ > 5: {(s > 5).sum()} / {len(s)} = {(s > 5).mean()*100:.3f}%')
