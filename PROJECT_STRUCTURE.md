# RMFS 프로젝트 구조 설계 문서

## 개요

RMFS(Robotic Mobile Fulfillment System) 창고에서 로봇이 Pod를 Workstation(WS)으로 운반하여 처리한 뒤, Pod를 Storage에 반납합니다.
RL 에이전트는 **Pod를 어떤 Storage 위치에 반납할지** 매 step마다 결정합니다.

| 요소 | 설명 |
|------|------|
| **Pod** | 상품이 담긴 선반. WS에서 처리 후 Storage에 반납 |
| **Robot** | Pod를 운반하는 이동 로봇 (N_R대) |
| **Workstation (WS)** | Pod를 처리하는 작업장 (N_W개) |
| **Storage** | Pod 보관 위치 (N_S개, Large layout: 480개) |

**목적함수**: Makespan 최소화 (모든 Pod Task 완료 시점)

---

## 1. 파일 구조

| 파일 | 역할 |
|------|------|
| `RMFS_ENV.py` | 원본 RMFS 환경 (단일 인스턴스, numpy/list 기반 DES) |
| `rmfs_env_batch.py` | B개의 `RMFS_Environment`를 감싸는 배치 래퍼 |
| `rmfs_model.py` | MLP Actor-Critic (12 features → 4 actions) |
| `rmfs_trainer.py` | REINFORCE/PPO 학습 루프 |
| `rmfs_ppo_utils.py` | 단순화된 PPO Memory + GAE (상태 텐서 1개) |
| `rmfs_data_generator.py` | seed 기반 배치 문제 생성 |
| `rmfs_train.py` | 학습 진입점 + 하이퍼파라미터 설정 |
| `model/sub_layers.py` | Actor, Critic, MLP 클래스 (공용 신경망 레이어) |

---

## 2. MDP 구조

### State (상태) — 12차원 feature vector

```
인덱스  이름                    설명
  0    target_ws_dist          현재 Pod의 Target WS까지 거리 / Max_TT_WW
  1    robot_avail_ratio       가용 로봇 비율 = sum(Robot_S) / N_R
  2    min_diff                WS간 시간 차이 / Max_min_diff
  3    nearest_storage_time    Nearest rule Storage 시간 / Max_TT_WS
  4    nearest_next_time       Nearest rule → 다음 WS 시간 / Max_TT_WS
  5    nearest_makespan_gain   Nearest rule Makespan 개선 비율
  6    target_storage_time     Target rule Storage 시간 / Max_TT_WS
  7    target_next_time        Target rule → 다음 WS 시간 / Max_TT_WS
  8    target_makespan_gain    Target rule Makespan 개선 비율
  9    sum_storage_time        Sum rule Storage 시간 / Max_TT_WS
 10    sum_next_time           Sum rule → 다음 WS 시간 / Max_TT_WS
 11    sum_makespan_gain       Sum rule Makespan 개선 비율

* 모든 feature는 [0, 1] 범위로 정규화됨
```

### Action (행동) — 4 discrete

| Action | 이름 | 설명 |
|--------|------|------|
| 0 | Stay | 현재 WS에 Pod를 유지 (반납하지 않음) |
| 1 | Nearest | 가장 가까운 Storage에 반납 |
| 2 | Target | 다음 필요한 WS에 가까운 Storage에 반납 |
| 3 | Sum | 모든 WS까지 거리 합이 최소인 Storage에 반납 |

**Infeasible action**: action 2, 3은 Target_Storage = -1일 때 infeasible.
이 경우 reward = -100000, done = True (에이전트가 학습으로 회피해야 함).

### Reward (보상) — Dense Stepwise

```python
reward = Pre_Makespan - Makespan
# 매 step마다 이전 Makespan과의 차이를 보상으로 제공
# 양수 = Makespan 감소 (좋음), 음수 = Makespan 증가 (나쁨)
```

### State Transition (상태 전이)

```
1. pod_assign()
   - 다음 처리할 Pod-WS-Robot 할당 (greedy: 가장 빠른 WS 우선)
   - 각 Storage rule별 후보 위치 계산 (Nearest, Target, Sum)
   - 12개 feature 계산 → state
   ↓
2. step(action)
   - 선택된 rule에 따라 Storage 위치 결정
   - Pod 이동, Robot 상태 업데이트
   - Makespan 갱신
   ↓
3. 종료 조건: Total_PodTask steps 완료 → done
```

---

## 3. 배치 환경 구조 (`rmfs_env_batch.py`)

```python
class RMFSBatchEnv:
    """
    B개의 독립적인 RMFS_Environment를 병렬 관리.

    배치 처리 방식:
    - pod_assign() 내부 로직이 조건 분기/가변 탐색으로 완전 벡터화 비실용적
    - per-instance for-loop으로 simulation 실행
    - 결과를 torch.tensor로 묶어 배치 인터페이스 제공
    """
    def reset(self, problem) -> RMFSState:       # (B, 12)
    def step(self, actions) -> (RMFSState, rewards, all_done)
    def get_makespan(self) -> Tensor              # (B,)
    def get_active_mask(self) -> Tensor            # (B,) True=active
```

---

## 4. MLP Actor-Critic 모델 (`rmfs_model.py`)

```python
class MLPActorCritic(nn.Module):
    """
    model/sub_layers.py의 Actor(tanh MLP), Critic(tanh MLP) 재사용.

    Forward:
        state_features (B, 12) → Actor → logits (B, 4) → softmax → pi (B, 4)
                                → Critic → v (B, 1)
    """
```

**파라미터 구조 (rmfs_train.py)**:
```python
model_params = {
    'state_dim': 12,              # RMFS_ENV.py feature 수와 일치 필요
    'action_dim': 4,              # Storage assignment rules 수
    'num_mlp_layers_actor': 3,    # Actor MLP 레이어 수
    'hidden_dim_actor': 128,      # Actor hidden 차원
    'num_mlp_layers_critic': 3,   # Critic MLP 레이어 수
    'hidden_dim_critic': 128,     # Critic hidden 차원
}
```

---

## 5. PPO Memory (`rmfs_ppo_utils.py`)

(B, 12) feature 텐서 1개를 상태로 저장하는 단순화된 PPO Memory.
Variable-length episode 지원을 위한 `mask_seq` 추가.

```
RMFSPPOMemory 저장 구조:
  상태 시퀀스 (1개):
    state_seq      [T, (B, 12)]

  전이 시퀀스 (6개):
    action_seq, log_probs, val_seq, reward_seq, done_seq, mask_seq

  transpose_data(): [T, B, ...] → [B*T, ...]
  get_gae_advantages(): GAE with active mask → (advantages, v_targets)
```

---

## 6. 환경 파라미터 (`rmfs_train.py`)

```python
env_params = {
    'batch_size': 64,
    'N_S': 480,            # Storage locations (Large: 40*12=480)
    'N_P': 100,            # Pods
    'N_R': 20,             # Robots
    'N_W': 4,              # Workstations
    'Total_PodTask': 1300, # Episode length
    'Unit_PT': 15,         # Unit processing time
    'ST': 5,               # Setup time
    'UT': 1,               # Unit travel time
    'Large': True,         # Large layout flag
    'seed_base': 0,        # Base seed for instance generation
}
```

---

## 7. 확장 시 유의사항

1. **`state_dim=12`**: `RMFS_ENV.py`의 `pod_assign()`에서 생성하는 feature 수와
   `rmfs_train.py`의 `model_params['state_dim']`을 항상 일치시켜야 합니다.
   Feature를 추가/제거하면 두 곳 모두 수정 필요.

2. **`action_dim=4`**: 새로운 Storage rule을 추가하면
   `RMFS_ENV.py`의 `step()`, `rmfs_train.py`의 `model_params['action_dim']` 모두 수정 필요.

3. **Infeasible action 처리**: 현재 Target/Sum rule이 -1을 반환하면 큰 벌점(-100000).
   향후 action masking을 도입하면 `rmfs_model.py`에 mask 파라미터를 추가해야 합니다.

4. **배치 성능**: per-instance for-loop이 병목. `batch_size`를 크게 늘리면
   `multiprocessing` 또는 환경 벡터화 고려 필요.
