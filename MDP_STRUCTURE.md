# RCMPSP MDP 구조 설계 문서

## 문제 정의
**Resource-Constrained Multi-Project Scheduling Problem (RCMPSP)**
- 여러 프로젝트의 Activity들을 제한된 Team 리소스에 할당하여 스케줄링
- 목적: Tardiness 최소화 또는 Makespan 최소화

---

## 1. State (상태)

### 1.1 그래프 구조

State는 **이종 그래프(Heterogeneous Graph)** 구조로 표현됩니다.

#### **노드 타입**
1. **Activity 노드**: 스케줄링할 작업 (최대 `max_N_A`개)
2. **Team 노드**: 리소스 (고정 `N_T`개)
3. **Project 노드**: 프로젝트 (고정 `N_P`개)

#### **엣지 타입 및 엣지 피처**
1. **Activity → Activity (Precedence)**
   - 의미: 선행 관계 (A → B: A가 완료되어야 B 시작 가능)
   - 방향: 단방향 (선행 → 후행)
   - 소스: `activity_predecessors` (batch_size, max_N_A, max_preds)
   - **엣지 피처**: 없음 (구조 정보만)

2. **Activity ↔ Activity (Mutex)**
   - 의미: 동시 실행 불가 (자원 충돌)
   - 방향: 양방향
   - 소스: `activity_mutex` (batch_size, max_N_A, max_mutex)
   - **엣지 피처**: `is_ordered` (1차원)
     - 평소: 0
     - 둘 중 하나가 시작하면 선후관계 발생 → 1
     - 동적으로 업데이트됨

3. **Activity ↔ Team (Eligible)**
   - 의미: Activity가 할당 가능한 Team
   - 방향: **양방향** (Activity ↔ Team)
   - 소스: `activity_eligible_teams` (batch_size, max_N_A, N_T)
   - **엣지 피처**: `is_assigned` (1차원)
     - 평소: 0
     - Activity가 해당 Team에 할당되면 → 1
     - **양방향 모두 업데이트됨**

4. **Activity ↔ Project (Belongs-to)**
   - 의미: Activity가 속한 Project
   - 방향: **양방향** (Activity ↔ Project)
   - 소스: `activity_project` (batch_size, max_N_A)
   - **엣지 피처**: 없음 (구조 정보만)

### 1.2 노드 피처

모든 노드는 **동일한 차원 (8차원)**을 사용하며, 각 노드 타입은 자신의 피처 위치에만 값을 채우고 나머지는 0으로 패딩합니다.

#### **전체 노드 피처 구조** (차원: 8)
```python
# 인덱스:     0          1        2       3             4              5            6        7
# 피처:   [duration, started, ended, remain_time, avail_time, release_time, due_date, completed]
#         |<-- Activity (4) -->||<- Team ->||<------ Project (3) ------>|
```

#### **Activity 노드 피처**
```python
# 위치: 인덱스 0~3 사용, 4~7은 0으로 패딩
[duration, started, ended, remaining_time, 0, 0, 0, 0]

# Static 속성
- [0] duration: float              # Activity 수행 시간 (정규화: /duration_max)

# Dynamic 속성
- [1] started: bool                # 시작 여부 (0 또는 1)
- [2] ended: bool                  # 완료 여부 (0 또는 1)
- [3] remaining_time: float        # 남은 수행 시간 (정규화: /duration_max)
                                   # - 완료된 경우: 0
                                   # - 진행 중: duration - elapsed_time
                                   # - 시작 안 한 경우: duration
```

**참고**: 선행 작업, Mutex, 할당 가능한 팀 등의 정보는 엣지 구조로 표현됩니다.

#### **Team 노드 피처**
```python
# 위치: 인덱스 0~3은 0으로 패딩, 4 사용, 5~7은 0으로 패딩
[0, 0, 0, 0, available_time, 0, 0, 0]

# Dynamic 속성
- [4] available_time: float        # 팀이 다시 사용 가능한 시간 (정규화: /max_time)
                                   # - 사용 가능: 0
                                   # - 사용 중: 남은 시간 (양수)
```

#### **Project 노드 피처**
```python
# 위치: 인덱스 0~4는 0으로 패딩, 5~7 사용
[0, 0, 0, 0, 0, release_time, due_date, completed]

# Static 속성
- [5] release_time: float          # 프로젝트 시작 가능 시간 (정규화: /max_time)
- [6] due_date: float              # 납기 (정규화: /max_time)

# Dynamic 속성
- [7] completed: bool              # 완료 여부 (0 또는 1)
```

### 1.3 State 텐서 구조

```python
# 배치별로 PyG Data 객체 생성
Data(
    x: Tensor,                  # (num_nodes, 8) - 노드 피처 (모든 노드 동일 차원)
                                # Activity: [0:4] 사용, [4:] 패딩 (인덱스 0,1,2,3)
                                # Team: [0:4] 패딩, [4] 사용, [5:] 패딩 (인덱스 4)
                                # Project: [0:5] 패딩, [5:8] 사용 (인덱스 5,6,7)
    edge_index: Tensor,         # (2, num_edges) - 엣지 연결 정보
    edge_attr: Tensor,          # (num_edges, edge_feature_dim) - 엣지 피처
                                # Precedence: 없음 (구조만)
                                # Mutex: 1차원 (is_ordered)
                                # Eligible: 1차원 (is_assigned, 양방향)
                                # Belongs-to: 없음 (구조만)
    mask: Tensor,               # (max_action_space,) - 가능한 action 마스크
    batch_idx: int,             # 배치 인덱스
    num_activities: int,        # 실제 activity 수
    sim_time: float             # 현재 시뮬레이션 시간
)
```

### 1.4 시뮬레이션 상태

```python
- sim_time: Tensor              # (batch_size,) - 현재 시뮬레이션 시간
- step_count: Tensor            # (batch_size,) - 스텝 카운터
- done: Tensor                  # (batch_size,) - 종료 여부
```

---

## 2. Action (행동)

### 2.1 Action Space

**Action**: `(Activity, Team)` 페어 선택
- 형식: 1D 인덱스 (0 ~ max_action_space-1)
- 매핑: `action_to_pair[batch_idx, action_idx] → (activity_id, team_id)`

```python
# Action space 구성
action_to_pair: Tensor          # (batch_size, max_action_space, 2)
                                # [:, :, 0] = activity_id
                                # [:, :, 1] = team_id
```

### 2.2 Action Feasibility 조건

Action이 실행 가능하려면 다음 조건을 **모두** 만족해야 합니다:

1. **Eligibility**: `activity_eligible_teams[b, act, team] == True`
   - 해당 Team이 해당 Activity를 수행 가능

2. **Not Started**: `activity_started[b, act] == False`
   - 아직 시작되지 않은 Activity

3. **Project Released**: `project_release_time[b, project] <= sim_time[b]`
   - Activity가 속한 Project가 release됨

4. **Predecessors Completed**: 모든 선행 작업 완료
   ```python
   for pred in activity_predecessors[b, act]:
       if pred >= 0:
           assert activity_ended[b, pred] == True
   ```

5. **Mutex Not Running**: Mutex 관계인 Activity가 실행 중이지 않음
   ```python
   for mutex in activity_mutex[b, act]:
       if mutex >= 0:
           assert not (activity_started[b, mutex] and not activity_ended[b, mutex])
   ```

6. **Team Available**: `team_available_time[b, team] <= sim_time[b]`
   - 해당 Team이 현재 사용 가능

### 2.3 Action Mask

```python
available_actions: Tensor       # (batch_size, max_action_space)
                                # True: 실행 가능한 action
                                # False: 실행 불가능한 action
```

---

## 3. Reward (보상)

### 3.1 Reward 설계

**Sparse Reward**: 에피소드 종료 시점에만 보상 제공

```python
# 에피소드 진행 중
reward = None

# 에피소드 종료 시 (모든 Activity 완료)
if objective == 'tardiness':
    reward = -total_tardiness    # (음수: 작을수록 좋음)
    # total_tardiness = sum(max(0, completion_time - due_date))
    
elif objective == 'makespan':
    reward = -makespan           # (음수: 작을수록 좋음)
    # makespan = max(completion_time)
```

### 3.2 목적함수

#### **Tardiness (지연 시간)**
```python
obj = sum_{p=1}^{N_P} max(0, completion_time[p] - due_date[p])
```
- 각 프로젝트의 지연 시간 합계
- 납기를 초과한 시간만 페널티

#### **Makespan (총 완료 시간)**
```python
obj = max_{p=1}^{N_P} completion_time[p]
```
- 모든 프로젝트 중 가장 늦게 완료된 시간

---

## 4. State Transition (상태 전이)

### 4.1 DES (Discrete Event Simulation) 방식

**Simulation Time**: 연속 시간 (float)
**Decision Points**: Activity 시작 시점

#### **전이 프로세스**

```
1. Action 선택: (activity_id, team_id)
   ↓
2. Activity 스케줄링
   - activity_started[b, activity_id] = True
   - activity_start_time[b, activity_id] = sim_time[b]  (내부 추적용)
   - activity_end_time[b, activity_id] = sim_time[b] + duration  (내부 추적용)
   - activity_remaining_time[b, activity_id] = duration
   - activity_assigned_team[b, activity_id] = team_id
   - team_available_time[b, team_id] = sim_time[b] + duration
   
   **엣지 피처 업데이트:**
   - Activity→Team 엣지: is_assigned[activity_id, team_id] = 1
   - Mutex 엣지: activity_id가 시작되면 mutex 관계의 엣지 is_ordered = 1
   ↓
3. 시간 진행 (move_next_state)
   LOOP:
     - 가능한 action이 있으면 → 의사결정 (STOP)
     - 가능한 action이 없으면 → 다음 이벤트까지 시간 진행
   ↓
4. Activity 완료 처리
   - sim_time[b] += time_delta
   - activity_remaining_time[b, :] -= time_delta
   - activity_ended[b, completed_acts] = True
   - team_current_activity[b, completed_teams] = -1
   - project_completion_time 업데이트 (내부 추적용)
   ↓
5. 종료 조건 확인
   - 모든 Activity 완료 → done = True
   - 아니면 → 3번으로 돌아가기
```

### 4.2 시간 진행 로직

```python
def get_next_move_t(batch_idxs):
    """다음 이벤트까지의 시간 계산"""
    # 진행 중인 activity들의 남은 시간
    remaining_times = activity_remaining_time[batch_idxs]  # (n_batch, max_N_A)
    
    # 0보다 큰 시간만 고려 (진행 중인 것만)
    masked_times = where(remaining_times > 0, remaining_times, inf)
    
    # 각 배치별 최소값 (가장 먼저 완료될 activity의 남은 시간)
    time_delta = min(masked_times, dim=1)
    
    return time_delta
```

### 4.3 State Transition Diagram

```
┌─────────────────────────────────────────────────────┐
│  State s_t                                          │
│  - 그래프 (Activity, Team, Project 노드)            │
│  - Dynamic 속성 (started, ended, available_time 등) │
│  - sim_time = t                                     │
└─────────────────────────────────────────────────────┘
                    │
                    │ Action a_t: (activity, team)
                    ↓
┌─────────────────────────────────────────────────────┐
│  Scheduling                                         │
│  - activity 시작                                    │
│  - team 할당                                        │
│  - end_time 계산                                    │
└─────────────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────┐
│  Time Advance (DES)                                 │
│  - 다음 decision point까지 시간 진행                │
│  - activity 완료 처리                               │
│  - project 완료 확인                                │
└─────────────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────┐
│  State s_{t+1}                                      │
│  - 그래프 구조 유지 (노드/엣지 불변)                 │
│  - Dynamic 노드 피처 업데이트                        │
│    (started, ended, remaining_time, available_time) │
│  - Dynamic 엣지 피처 업데이트                        │
│    (is_ordered, is_assigned)                        │
│  - sim_time = t + Δt                                │
└─────────────────────────────────────────────────────┘
                    │
                    │ if all activities ended
                    ↓
                 Terminal State → Reward
```

---

## 5. GNN 기반 정책 네트워크

### 5.1 아키텍처

```
Input: Heterogeneous Graph
  ↓
Node Feature (8차원 패딩 방식):
  - Activity: [0:4] 사용, [4:8] 패딩
  - Team: [4] 사용, 나머지 패딩
  - Project: [5:8] 사용, 나머지 패딩
  ↓
Node Embedding: Linear(input_dim=8, embedding_dim=128) + ReLU
  ↓
GAT Layers × L (default: 3)
  - Multi-head attention (heads=8)
  - Edge features 활용 (Mutex: is_ordered, Eligible: is_assigned)
  - Residual connection: Concat[GAT_out, input] → Linear(256, 128)
  ↓
Node Embeddings: (total_nodes, 128)
  ↓
Action Decoder:
  For each eligible (activity, team) pair:
    - Concatenate: [act_emb, team_emb]  # (256,)
    - MLP: Linear(256, 128) → ReLU → Linear(128, 1)
    - Output: action_logit
  ↓
Action Logits: (max_action_space,)
  ↓
Masking: logits += log(mask + eps)  # invalid action → -inf
  ↓
Policy: π(a|s) = Softmax(masked_logits)
```

### 5.2 주요 특징

1. **그래프 구조 학습**
   - GAT를 통해 노드 간 관계 학습
   - Precedence, Mutex, Eligible 관계 반영
   - **엣지 피처 활용**: Mutex 선후관계, Team 할당 정보

2. **이종 노드 처리**
   - Activity (4차원), Team (1차원), Project (3차원) 피처를 각각 처리
   - 통일된 embedding 차원으로 변환 후 GNN 처리

3. **Eligible Action만 디코딩**
   - 모든 (activity, team) 조합이 아닌
   - Eligible한 페어만 계산 (효율성)

4. **Action Masking**
   - Feasibility 조건을 만족하지 않는 action → -inf
   - 정책이 항상 유효한 action만 선택

---

## 6. MDP 요약

| 구성 요소 | 설명 | 차원/형태 |
|----------|------|----------|
| **State** | 이종 그래프 (Activity, Team, Project 노드) | PyG Data 객체 |
| **Action** | (Activity, Team) 페어 선택 | 1D 인덱스 (0 ~ max_action_space-1) |
| **Reward** | -Tardiness 또는 -Makespan (에피소드 끝) | Scalar |
| **Transition** | DES 기반 시뮬레이션 | Deterministic |
| **Horizon** | Variable (모든 Activity 완료 시 종료) | ~50-200 steps |

### 6.1 핵심 특징

1. **Deterministic MDP**: Action에 대한 전이가 결정적
2. **Sparse Reward**: 에피소드 끝에만 보상
3. **Variable Episode Length**: 문제 크기에 따라 가변적
4. **Combinatorial Action Space**: Eligible 기반으로 축소
5. **Graph-structured State**: 이종 그래프로 관계 표현

---

## 7. 학습 알고리즘

### 7.1 REINFORCE

**Vanilla Policy Gradient** 방식으로 학습합니다.

```python
# Episode 수집 (Rollout)
log_probs = []
for step in range(max_steps):
    action, log_prob, entropy = policy.get_action(state)
    log_probs.append(log_prob)
    next_state, reward, done = env.step(action)

# Episode 종료 시 Reward 할당
if done:
    cumulative_log_prob = sum(log_probs)  # 전체 궤적의 로그 확률
    reward = -objective_value  # Sparse reward (음수)
    
    # Advantage 계산 (Baseline 사용)
    if baseline_type == 'pomo':
        # POMO baseline: 인스턴스별 평균
        advantage = reward - reward.mean(dim=pomo)
    elif baseline_type == 'batch':
        # Batch baseline: 배치 전체 평균
        advantage = reward - reward.mean()
    else:
        # No baseline
        advantage = reward
    
    # Advantage Normalization (옵션)
    if normalize_advantage:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    
    # Policy Loss
    loss = -advantage * cumulative_log_prob
    
    # Entropy Regularization (옵션)
    if use_entropy_reg:
        loss = loss - entropy_coef * entropy
    
    # Gradient 업데이트
    loss.backward()
    optimizer.step()
```

**핵심 특징:**
- **On-policy**: 현재 정책으로 수집한 경험으로만 학습
- **Sparse reward**: 에피소드 끝에만 보상 제공
- **Baseline**: POMO 평균 또는 배치 평균으로 분산 감소
- **High variance**: PPO보다 분산이 크지만 구현이 단순

### 7.2 POMO (Policy Optimization with Multiple Optima)

- 동일 인스턴스에 대해 여러 롤아웃 수행
- 최적 솔루션에 가까운 궤적으로 학습
- Exploration 향상

---

## 8. 코드 구조 매핑

### 8.1 환경 (scheduling_env.py)

| 메서드 | 역할 |
|--------|------|
| `_reset()` | 환경 초기화, State 생성 |
| `step(action)` | Action 실행, State Transition |
| `_schedule_activity()` | Activity 스케줄링 |
| `move_next_state()` | DES 시뮬레이션 (시간 진행) |
| `_get_state()` | 그래프 State 생성 |
| `_get_obj()` | 목적함수값 계산 |
| `_update_available_actions()` | Action Mask 업데이트 |

### 8.2 모델 (gnn_model.py)

| 클래스/메서드 | 역할 |
|--------------|------|
| `GNNModel` | GAT 기반 그래프 인코더 |
| `forward()` | State → Action Logits |
| `get_action()` | 정책에서 Action 샘플링 |
| `get_max_action()` | Greedy Action 선택 |

---

## 부록: 벡터화 구현

모든 연산은 배치 처리되어 효율적으로 실행됩니다:

```python
# 배치 전체에 대한 벡터 연산
activity_remaining_time[batch_idxs] -= time_deltas.unsqueeze(1)
just_completed_mask = (activity_remaining_time <= 0) & activity_started & ~activity_ended
```

이를 통해 수백 개의 인스턴스를 동시에 시뮬레이션할 수 있습니다.
