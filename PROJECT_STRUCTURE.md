# RMFS 프로젝트 구조 설계 문서

## 개요

RMFS(Robotic Mobile Fulfillment System) 창고에서 로봇이 Pod를 Workstation(WS)으로 운반하여 처리한 뒤, Pod를 Storage에 반납합니다.
RL 에이전트는 **Pod를 어떤 Storage 위치에 반납할지** 매 step마다 결정합니다.

| 요소 | 설명 |
|------|------|
| **Pod** | 상품이 담긴 선반. WS에서 처리 후 Storage에 반납 |
| **Robot** | Pod를 운반하는 이동 로봇 (N_R대) |
| **Workstation (WS)** | Pod를 처리하는 작업장 (N_W개) |
| **Storage** | Pod 보관 위치 (N_S개, 기본: 3×3×4×2 = 72) |

**목적함수**: Makespan 최소화 (모든 Pod Task 완료 시점)

---

## 1. 파일 구조

### 학습/추론 파이프라인

| 파일 | 역할 |
|------|------|
| `rmfs_train.py` | 학습 진입점 + 모든 하이퍼파라미터 설정 |
| `rmfs_trainer.py` | REINFORCE/PPO 학습 루프, validation, checkpoint 관리 |
| `rmfs_env_batch.py` | B개의 `RMFS_Environment`를 감싸는 배치 래퍼 (`RMFSState` 그래프 상태 반환) |
| `RMFS_ENV.py` | 원본 RMFS 환경 (단일 인스턴스, numpy/list 기반 DES) |
| `rmfs_model.py` | `GATActorCritic`: GATv2 기반 Actor-Critic (graph → π, v) |
| `model/gat_layers.py` | `GATv2Layer`: edge feature 포함 dense adjacency GATv2 |
| `model/sub_layers.py` | `Actor`, `Critic`, `MLP` 클래스 (tanh activation) |
| `rmfs_ppo_utils.py` | `RMFSPPOMemory` (graph state 저장) + per-instance GAE |
| `rmfs_data_generator.py` | seed 기반 배치 문제 생성 |

### 테스트/벤치마크

| 파일 | 역할 |
|------|------|
| `rmfs_test.py` | 체크포인트 로드 → greedy/sampling rollout + 선택적 Gurobi MIP 비교 → Excel |
| `MILP.py` | `solve_rmfs_milp()`: Gurobi MILP 정식화 (31개 제약 그룹) |
| `rmfs_milp_utils.py` | `convert_env_to_milp_data()`: RMFS 환경 → 1-indexed MILP 입력 변환 |
| `rmfs_mip_replay.py` | Gurobi MIP 솔루션을 RMFS 환경에서 step-by-step 재현 및 정합성 검증 |
| `rmfs_data_generation.py` | 테스트용 pickle 파일 생성 스크립트 (`data/rmfs_test/`) |
| `gantt_chart.py` | Gantt chart 시각화 유틸리티 |

### 참조/미사용

| 파일 | 비고 |
|------|------|
| `model_refer/` | 다른 스케줄링 프로젝트(RCMPSP/FJSP)의 참조 모델. RMFS에 사용되지 않음 |
| `cp_replay.py`, `test.py`, `data_generation.py` | 다른 프로젝트 파일. 존재하지 않는 모듈 import → 실행 불가 |

---

## 2. MDP 구조

### State (상태) — Heterogeneous Graph

`RMFSState` 데이터클래스 (총 V = N_S + N_W = 76 노드):

| 필드 | Shape | 설명 |
|------|-------|------|
| `storage_features` | (B, N_S, 4) | Storage 노드 특성 |
| `ws_features` | (B, N_W, 4) | WS 노드 특성 |
| `edge_feat` | (B, V, V, 9) | Edge 특성 |
| `curws_idx` | (B,) | 현재 결정 대상 WS 인덱스 |
| `action_mask` | (B, N_S+1) | 유효 action bool 마스크 |

**Storage 노드 특성 (4차원)**:
- normalized x, normalized y, occupied flag, normalized idle time

**WS 노드 특성 (4차원)**:
- normalized x, normalized y, busy flag, normalized idle time

**Edge 특성 (9차원)**:
- `[0-2]` type flags: ws_to_s, s_to_ws, ws_to_ws
- `[3]` 정규화 맨하탄 거리
- `[4]` pod_stored_needs_ws (해당 storage의 pod가 해당 WS에 작업이 있는지)
- `[5]` 정규화 방문 순서 차이
- `[6]` 정규화 WS 간 공유 pod 수
- `[7]` 정규화 근접도 합
- `[8]` curpod_needs_other_ws (현재 pod가 다른 WS에 작업이 있는지)

### Action (행동) — N_S+1 discrete

| Action | 설명 |
|--------|------|
| 0 | Stay: 현재 WS에 Pod를 유지 (반납하지 않음) |
| 1 ~ N_S | 해당 인덱스의 Storage 위치에 Pod를 반납 |

**Infeasible action**: 선택한 storage가 `Pod_departure_time + travel_time` 시점에 이미 점유되면 infeasible.
reward = -100000, done = True.

**Action masking**: `action_mask (B, N_S+1)` bool 텐서로 유효 action만 표시.
`force_mask_stay=True`이면 action 0은 항상 masked → deadlock 방지.

### Reward (보상)

```python
# Stepwise (기본): 매 step마다 이전 Makespan과의 차이
reward = Pre_Makespan - Makespan
# 양수 = Makespan 감소 (좋음), 음수 = Makespan 증가 (나쁨)

# Sparse: episode 끝에만
reward = -Makespan
```

### State Transition (상태 전이)

```
1. pod_assign()  [RMFS_ENV.py 내부]
   - 다음 처리할 Pod-WS-Robot을 greedy 할당 (가장 빠른 WS 우선)
   - Pod가 WS에 있으면 robot 불필요, Storage에 있으면 robot 할당
   - graph state 계산: 노드 특성 + edge 특성 + action mask
   ↓
2. step(action)  [agent가 storage 위치 선택]
   - 선택된 storage로 Pod 반납, Robot/Storage 상태 업데이트
   - Makespan 갱신
   ↓
3. 종료 조건: Total_PodTask steps 완료 → done
```

### Static Adjacency (인접행렬)

한 번 생성 후 모든 배치/step에서 재사용 (1, V, V) bool:
- WS → Storage (모든 쌍), Storage → WS (모든 쌍)
- WS ↔ WS (자기 제외)
- 모든 노드 self-loop
- Storage ↔ Storage 연결은 **없음**

---

## 3. 배치 환경 구조 (`rmfs_env_batch.py`)

```python
@dataclass
class RMFSState:
    storage_features: Tensor   # (B, N_S, 4)
    ws_features: Tensor        # (B, N_W, 4)
    edge_feat: Tensor          # (B, V, V, 9)
    curws_idx: Tensor          # (B,) int
    action_mask: Tensor        # (B, N_S+1) bool

class RMFSBatchEnv:
    """
    B개의 독립적인 RMFS_Environment를 병렬 관리.

    배치 처리 방식:
    - pod_assign() 내부 로직이 조건 분기/가변 탐색으로 완전 벡터화 비실용적
    - per-instance for-loop으로 simulation 실행
    - 결과 numpy를 torch.tensor로 stack하여 배치 인터페이스 제공
    """
    def reset(self, problem) -> RMFSState
    def step(self, actions) -> (RMFSState, rewards, all_done)
    def get_makespan(self) -> Tensor         # (B,)
    def get_active_mask(self) -> Tensor      # (B,) True=active
```

Done 인스턴스는 `_zero_graph_state()`(전부 0인 dummy state)를 반환합니다.

---

## 4. GATv2 Actor-Critic 모델 (`rmfs_model.py`)

```python
class GATActorCritic(nn.Module):
    """
    Input:  RMFSState (graph) + adj (1, V, V)
    Output: pi (B, N_S+1), v (B, 1)
    """
```

### Forward 흐름

```
1. Node Projection
   storage_proj: Linear(4 → d) + node_type_embed[0]  → h_s (B, N_S, d)
   ws_proj:      Linear(4 → d) + node_type_embed[1]  → h_w (B, N_W, d)
   concat → h (B, V, d)

2. GATv2 Layers × n_gat_layers (residual + LayerNorm)
   각 레이어: h (B,V,d) + adj (B,V,V) + edge_feat (B,V,V,9) → h_new (B,V,d)
   h = LayerNorm(h + h_new)

3. Split & Pool
   h_storage (B, N_S, d),  h_ws (B, N_W, d)
   h_global_s = mean(h_storage)  → (B, d)
   h_global_w = mean(h_ws)       → (B, d)
   h_curws    = h_ws[curws_idx]  → (B, d)

4. Actor: per-storage scores
   입력 = [h_storage, h_curws, h_global_s, h_global_w, edge_curws_to_s]
        = (B, N_S, 4d+9)
   → Actor MLP(4d+9 → d → 1) → storage_scores (B, N_S)

   Stay head:
   입력 = [h_curws, h_global_s, h_global_w] = (B, 3d)
   → Actor MLP(3d → d → 1) → stay_score (B, 1)

   logits = cat([stay_score, storage_scores]) → (B, N_S+1)
   masked_fill(~action_mask, -inf) → softmax → π (B, N_S+1)

5. Critic
   입력 = [h_global_s, h_global_w] = (B, 2d)
   → Critic MLP(2d → d → 1) → v (B, 1)
```

### GATv2Layer (`model/gat_layers.py`)

```
W_src(h)[i] + W_dst(h)[j] + W_edge(edge_feat)[i,j] + bias
→ LeakyReLU(combined)
→ e[i,j,head] = a_head · combined  (adjacency로 masking)
→ alpha = softmax(e, dim=j)
→ h'[i] = Σ_j alpha[i,j] · W_val(h)[j]
→ 멀티헤드 concat → output projection
```

### 기본 파라미터

```python
model_params = {
    'storage_feat_dim': 4,   # Storage 노드 feature 차원
    'ws_feat_dim': 4,        # WS 노드 feature 차원
    'd_edge': 9,             # Edge feature 차원
    'd_hidden': 64,          # GATv2 hidden 차원 (= d)
    'n_gat_layers': 3,       # GATv2 레이어 수
    'n_heads': 4,            # 어텐션 헤드 수
    'dropout_prob': 0.0,
    'num_mlp_layers_actor': 2,
    'hidden_dim_actor': 64,
    'num_mlp_layers_critic': 2,
    'hidden_dim_critic': 64,
}
```

---

## 5. PPO Memory (`rmfs_ppo_utils.py`)

Graph state 필드를 각각 별도 리스트로 저장하는 PPO Memory.

```
RMFSPPOMemory 저장 구조:

  상태 시퀀스 (5개 필드, 각각 [T, tensor]):
    storage_features_seq   [T, (B, N_S, 4)]
    ws_features_seq        [T, (B, N_W, 4)]
    edge_feat_seq          [T, (B, V, V, 9)]
    curws_idx_seq          [T, (B,)]
    action_mask_seq        [T, (B, N_S+1)]

  전이 시퀀스 (6개):
    action_seq, log_probs, val_seq, reward_seq, done_seq, mask_seq

  transpose_data():
    [T, B, ...] → [B*T, ...] 11개 텐서 반환

  get_gae_advantages():
    per-instance 정규화된 GAE → (advantages [B*T], v_targets [B*T])
```

### GAE 계산

```
역순 순회 (i = T-1 → 0):
  delta = r[i] + γ * V[i+1] * mask[i] - V[i]
  advantage = delta + γ * λ * advantage * mask[i]

Per-instance 정규화:
  adv = (adv - mean(adv, dim=T)) / (std(adv, dim=T) + 1e-8)

v_target = unnormalized_advantage + V
```

---

## 6. 학습 흐름 (`rmfs_trainer.py`)

### REINFORCE

```
epoch마다:
  1. generate_rmfs_data_batch() → seeds 리스트
  2. RMFSBatchEnv 생성 & reset
  3. Rollout (model → sample → step) until all_done
     - log_prob_sum, cumulative_reward 누적 (active_mask 적용)
  4. Advantage = reward - mean(reward)  [batch baseline]
  5. Loss = -advantage * log_prob_sum - entropy_coef * entropy
  6. backward → grad_clip(1.0) → optimizer.step
```

### PPO

```
epoch마다 (N_RESAMPLE=20 epoch마다 새 problem 생성):
  1. model_old ← model (hard copy)
  2. Rollout (model.eval, no_grad):
     - memory.push_state(), memory.push_transition()
  3. memory.transpose_data() → [B*T, ...] 텐서
  4. memory.get_gae_advantages() → advantages, v_targets
  5. K_EPOCHS=4 반복:
     for minibatch (size 4096):
       - RMFSState 미니배치 구성
       - model(mini_state, adj) → π_new, v_new
       - ratio = exp(log_π_new - log_π_old)
       - surr_loss = -min(ratio*adv, clip(ratio)*adv)
       - v_loss = MSE(v_new, v_target)
       - loss = 1.0*p_loss + 0.5*v_loss + entropy_coef*(-entropy)
       - optimizer.step()
```

### Validation

5 epoch마다 greedy rollout (argmax(π)):
- 고정 20개 인스턴스 (`data/rmfs_val/val_batch.pickle`, seed 2025)
- best_model.pt 갱신 기준

---

## 7. 환경 파라미터 (`rmfs_train.py`)

```python
env_params = {
    'batch_size': 3,         # PPO 배치 크기 (REINFORCE: 64)
    'block_rows': 3,         # Storage 블록 그리드 행
    'block_cols': 3,         # Storage 블록 그리드 열
    'block_h': 4,            # 블록 내부 행
    'block_w': 2,            # 블록 내부 열
    # N_S = 3*3*4*2 = 72
    'N_P': 40,               # Pods
    'N_R': 10,               # Robots
    'N_W': 4,                # Workstations
    'Total_PodTask': 60,     # Episode length
    'Unit_PT': 15,           # Unit processing time
    'ST': 5,                 # Setup time
    'UT': 1,                 # Unit travel time
    'Large': True,           # Large layout flag
    'seed_base': 0,          # Base seed
    'force_mask_stay': True, # Stay 마스킹 (deadlock 방지)
}
```

---

## 8. 체크포인트 구조

**경로**: `checkpoints/{timestamp}_RMFS_GAT_{PPO|REINFORCE}/`

| 파일 | 조건 |
|------|------|
| `epoch0.pt` | 학습 전 초기 상태 |
| `epoch{N}.pt` | 5 epoch마다 |
| `best_model.pt` | validation 개선 시 |

**저장 dict keys**:
```python
{
    'model_state_dict',          # GATActorCritic 가중치
    'optimizer_state_dict',
    'env_params',
    'model_params',
    'trainer_params',
    'epoch',
    'train_score',
    'val_score',
    'initial_val_score',
    'algorithm_type',
    'model_old_state_dict',      # PPO only (None for REINFORCE)
}
```

---

## 9. 테스트/벤치마크 파이프라인

### RL 테스트 (`rmfs_test.py`)

1. `best_model.pt` 로드 → `GATActorCritic` 재구성 (저장된 `model_params` 사용)
2. 인접행렬 재구성
3. Greedy rollout: argmax(π) 로 전체 테스트 인스턴스 실행
4. Stochastic sampling: K=64 rollouts per instance → 최소 makespan 선택
5. (선택) Gurobi MIP 비교
6. `results/{timestamp}/` 에 Excel 저장

### MIP 벤치마크

- `rmfs_milp_utils.py`: `RMFS_Environment`를 reset 후 `convert_env_to_milp_data()`로 MILP 입력 dict 생성
- `MILP.py`: Gurobi로 MILP 최적해 계산 (31개 제약 그룹, pod-ws 할당/로봇/시간/순서 등)
- `rmfs_mip_replay.py`: MIP 솔루션을 RMFS 환경에서 step-by-step 실행하여 정합성 검증

---

## 10. 확장 시 유의사항

1. **노드/edge feature 변경**: `RMFS_ENV.py`의 `pod_assign()`에서 생성하는 graph state dict와
   `rmfs_model.py`의 `storage_feat_dim`, `ws_feat_dim`, `d_edge`를 일치시켜야 합니다.

2. **Action space 변경**: `N_S`가 변경되면 (블록 설정 변경) action space(`N_S+1`)도 자동 변경.
   `env_params`의 block 파라미터만 수정하면 됩니다.

3. **Infeasible action 처리**: action masking이 이미 구현되어 있으며,
   `force_mask_stay=True`로 Stay action도 마스킹 가능.
   NaN safety로 모든 action masked 시 uniform 분포 fallback.

4. **배치 성능**: per-instance for-loop이 병목.
   `batch_size`를 크게 늘리려면 `multiprocessing` 또는 환경 벡터화 고려 필요.

5. **PPO 문제 재사용**: `N_RESAMPLE=20`으로 같은 problem을 20 epoch 동안 재사용.
   다양성이 필요하면 이 값을 줄일 수 있음.
