# RCMPSP MDP 구조 설계 문서

## 문제 정의
**Resource-Constrained Multi-Project Scheduling Problem (RCMPSP)**
- 여러 프로젝트의 Activity들을 제한된 Team 리소스에 할당하여 스케줄링
- 목적: Tardiness 최소화 또는 Makespan 최소화

---

## 1. State (상태)

### 1.1 상태 표현

State는 **DANIEL 모델**의 입력 형태인 `EnvState` 데이터클래스로 표현됩니다.
Activity, Team, Pair 정보를 분리된 텐서로 구성하며, 관계 정보는 마스크와 인덱스 텐서로 표현합니다.

#### **구성 요소**
1. **Activity 텐서**: 각 activity의 상태 및 속성 (12차원 피처)
2. **Team 텐서**: 각 team의 상태 및 속성 (8차원 피처)
3. **Pair 텐서**: (activity, team) 조합별 피처 (8차원)
4. **마스크/인덱스**: attention mask, 경쟁 인덱스, 실행 불가 pair mask

#### **제약 조건의 반영 방식**
1. **Precedence** (선행 관계): `dynamic_pair_mask`에서 선행자 미완료 시 마스킹 + `pred_idx`/`succ_idx`로 DAG 구조 전달
2. **Mutex** (동시 실행 불가): `dynamic_pair_mask`에서 mutex 파트너 실행 중 시 마스킹
3. **Eligible** (팀 수행 가능): `dynamic_pair_mask`에서 비eligible pair 마스킹
4. **Release time / Due date**: `dynamic_pair_mask` + pair feature에 반영

### 1.2 EnvState 텐서 구조

```python
@dataclass
class EnvState:
    fea_act_tensor:           Tensor  # (B, N, 12)  — activity features
    act_mask_tensor:          Tensor  # (B, N, 3)   — activity attention mask
    fea_team_tensor:          Tensor  # (B, T, 8)   — team features
    team_mask_tensor:         Tensor  # (B, T, T)   — team attention mask
    dynamic_pair_mask_tensor: Tensor  # (B, N, T)   — 불가능 pair 마스크
    comp_idx_tensor:          Tensor  # (B, T, T, N) — team 경쟁 인덱스
    candidate_tensor:         Tensor  # (B, N)       — activity identity (0..N-1)
    fea_pairs_tensor:         Tensor  # (B, N, T, 8) — pair feature
    pred_idx_tensor:          Tensor  # (B, N, max_preds) — predecessor 인덱스
    succ_idx_tensor:          Tensor  # (B, N, max_succs) — successor 인덱스
```

#### **Activity Feature (`fea_act`) — 12차원**
```
인덱스  이름                    설명
  0    started                 시작 여부 (0 or 1)
  1    ended                   완료 여부 (0 or 1)
  2    norm_duration           duration / max_duration
  3    min_pt                  eligible 팀 중 최소 processing time / max_duration
  4    pt_span                 eligible 팀 PT 범위 (max - min) / max_duration
  5    norm_remaining          remaining_time / max_duration
  6    pred_completion_ratio   선행자 완료 비율 (pred_done / total_pred)
  7    ready                   현재 실행 가능 여부 (0 or 1)
  8    completion_lb           예상 완료 시간 하한 / max_time (DAG relaxation)
  9    proj_rem_act_ratio      프로젝트 내 남은 activity 비율
 10    proj_rem_work_ratio     프로젝트 내 남은 작업량 비율
 11    eligible_ratio          eligible 팀 수 / 전체 팀 수
```

#### **Team Feature (`fea_team`) — 8차원**
```
인덱스  이름                    설명
  0    avail_cand_ratio        현재 가용한 activity pair 수 / N
  1    compat_unstarted_ratio  eligible한 미시작 activity 수 / N
  2    min_elig_dur            eligible 미시작 activity 중 최소 duration / max_dur
  3    mean_elig_dur           eligible 미시작 activity 중 평균 duration / max_dur
  4    idle_time               max(0, current - avail_time) / max_time
  5    wait_time               max(0, avail_time - current) / max_dur
  6    norm_avail_time         avail_time / max_time
  7    is_busy                 avail_time > current_time (0 or 1)
```

#### **Pair Feature (`fea_pairs`) — 8차원**
```
인덱스  이름                 설명
  0    norm_pt_global       processing_time / max_duration
  1    norm_pt_per_act      processing_time / 해당 activity 최대 PT
  2    norm_pt_per_team     processing_time / 해당 team 최대 PT
  3    norm_pt_global_max   processing_time / 전체 최대 PT
  4    team_workload_ratio  processing_time / (team_remaining + PT)
  5    due_date_slack       (due_date - est_completion) / max_time
  6    proj_work_ratio      processing_time / 프로젝트 남은 작업량
  7    total_wait_time      (team_wait + release_wait [+ mutex_wait]) / max_time
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
- 형식: 1D flat 인덱스 (0 ~ N*T-1)
- 매핑: `act_idx = action // N_T`, `team_idx = action % N_T`

```python
# Action space 구조 (DANIEL)
# N: 최대 activity 수, T: team 수
# flat action index → (activity, team) 직접 분해
action_flat: Tensor             # (batch_size,) — 0 ~ N*T-1
act_idx  = action_flat // N_T   # 어느 activity인지
team_idx = action_flat % N_T    # 어느 팀인지
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
dynamic_pair_mask: Tensor       # (batch_size, N, T)
                                # True: 실행 불가능한 pair (마스킹)
                                # False: 실행 가능한 pair
# Actor 출력에서 masked pair에 -inf 부여 → softmax 후 확률 0
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
│  State s_t (EnvState)                               │
│  - fea_act, fea_team, fea_pairs (피처 텐서)         │
│  - dynamic_pair_mask, comp_idx (마스크/인덱스)      │
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
│  State s_{t+1} (EnvState)                           │
│  - fea_act 업데이트 (started, ended, remaining 등)  │
│  - fea_team 업데이트 (available_time 등)            │
│  - dynamic_pair_mask 재계산                         │
│  - fea_pairs 재계산                                 │
│  - sim_time = t + Δt                                │
└─────────────────────────────────────────────────────┘
                    │
                    │ if all activities ended
                    ↓
                 Terminal State → Reward
```

---

## 5. DANIEL 정책 네트워크

### 5.1 아키텍처

```
Input: EnvState 텐서
  ├─ fea_act    [B, N, 12]   Activity feature
  ├─ act_mask   [B, N, 3]    Activity attention mask
  ├─ fea_team   [B, T, 8]    Team feature
  ├─ team_mask  [B, T, T]    Team attention mask
  ├─ comp_idx   [B, T, T, N] Team 간 경쟁 인덱스
  ├─ pair_mask  [B, N, T]    실행 불가 pair 마스크
  └─ fea_pairs  [B, N, T, 8] Pair feature
  ↓
DualAttentionNetwork (L 레이어 반복):
  ① Activity Attention Block (activity 간 순서/경쟁 학습)
  ② Team Attention Block (team 간 경쟁 학습, comp_idx 활용)
  ↓
Encoded Embeddings:
  fea_act_enc   [B, N, d]   activity 임베딩
  fea_team_enc  [B, T, d]   team 임베딩
  fea_act_g     [B, d]      global activity 임베딩
  fea_team_g    [B, d]      global team 임베딩
  ↓
Action Feature 조립:
  concat(activity_emb, team_emb, global_act, global_team, pair_feature)
  → candidate_feature [B, N*T, 4d+8]
  ↓
Actor MLP → [B, N*T] (raw logits)
  ↓
Pair Mask → masked logits (-inf)
  ↓
Softmax → π [B, N*T] (행동 확률)

Critic MLP → [B, 1] (상태 가치)
```

### 5.2 주요 특징

1. **이중 어텐션 (Dual Attention)**
   - Activity Attention Block: activity 간 순서/경쟁 관계 학습
   - Team Attention Block: team 간 경쟁 관계 학습 (comp_idx 기반)
   - 두 블록이 레이어마다 번갈아 상호 영향

2. **Pair Feature 활용**
   - (activity, team) 조합별 8차원 피처 (processing time, due date slack 등)
   - Actor 입력에 직접 concat하여 의사결정에 활용

3. **N×T Action Space**
   - 전체 (activity, team) 조합을 flat index로 관리
   - `dynamic_pair_mask`로 실행 불가 pair를 -inf 마스킹

4. **내장 Critic**
   - global activity + global team 임베딩으로 상태 가치 추정
   - PPO에서 GAE advantage 계산에 활용 (REINFORCE에서는 미사용)

---

## 6. MDP 요약

| 구성 요소 | 설명 | 차원/형태 |
|----------|------|----------|
| **State** | EnvState (Activity, Team, Pair 텐서 + 마스크) | EnvState 데이터클래스 (10개 텐서) |
| **Action** | (Activity, Team) 페어 선택 | 1D flat 인덱스 (0 ~ N*T-1) |
| **Reward** | -Tardiness 또는 -Makespan (에피소드 끝) | Scalar |
| **Transition** | DES 기반 시뮬레이션 | Deterministic |
| **Horizon** | Variable (모든 Activity 완료 시 종료) | ~50-200 steps |

### 6.1 핵심 특징

1. **Deterministic MDP**: Action에 대한 전이가 결정적
2. **Sparse Reward**: 에피소드 끝에만 보상
3. **Variable Episode Length**: 문제 크기에 따라 가변적
4. **Combinatorial Action Space**: Eligible 기반으로 축소
5. **Tensor-structured State**: 분리된 텐서 (Activity, Team, Pair) + 마스크로 관계 표현

---

## 7. 학습 알고리즘

### 7.1 REINFORCE

**Vanilla Policy Gradient** 방식으로 학습합니다.

```python
# Episode 수집 (Rollout)
log_probs = []
for step in range(max_steps):
    pi, v = model(state)
    action_flat = Categorical(pi).sample()
    log_probs.append(Categorical(pi).log_prob(action_flat))
    act_idx = action_flat // N_T
    team_idx = action_flat % N_T
    next_state, reward, done = env.step_pair(act_idx, team_idx)

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
| `step_pair(act_idx, team_idx)` | Action 실행, State Transition |
| `_schedule_activity()` | Activity 스케줄링 |
| `move_next_state()` | DES 시뮬레이션 (시간 진행) |
| `_get_state()` | EnvState 생성 (DANIEL 입력 텐서) |
| `_get_obj()` | 목적함수값 계산 |
| `_update_available_actions()` | Action Mask 업데이트 |

### 8.2 모델 (model/main_model.py)

| 클래스/메서드 | 역할 |
|--------------|------|
| `DANIEL` | Dual Attention Network 기반 정책+가치 모델 |
| `DualAttentionNetwork` | Activity/Team 이중 어텐션 인코더 |
| `forward()` | EnvState → π (행동 확률), v (상태 가치) |

---

## 부록: 벡터화 구현

모든 연산은 배치 처리되어 효율적으로 실행됩니다:

```python
# 배치 전체에 대한 벡터 연산
activity_remaining_time[batch_idxs] -= time_deltas.unsqueeze(1)
just_completed_mask = (activity_remaining_time <= 0) & activity_started & ~activity_ended
```

이를 통해 수백 개의 인스턴스를 동시에 시뮬레이션할 수 있습니다.
