# RCMPSP 프로젝트 구조 설계 문서

## 개요

**RCMPSP (Resource-Constrained Multi-Project Scheduling Problem)**을 딥러닝 기반 강화학습(RL)으로 해결하는 프로젝트입니다.

여러 프로젝트의 Activity를 제한된 Team(리소스)에 할당하여 **Tardiness** 또는 **Makespan**을 최소화합니다.

### 핵심 제약 조건
| 제약 | 설명 |
|------|------|
| **Precedence** | Activity 간 선행 관계 (임의 DAG) |
| **Mutex** | 동시에 실행 불가한 Activity 쌍 |
| **Eligible** | Activity마다 수행 가능한 Team 목록 |
| **Release time** | 프로젝트별 시작 가능 시간 |
| **Due date** | 프로젝트별 납기 |

### 정책 네트워크
**DANIEL** (Dual Attention Network for Integrated Scheduling) — FJSP용으로 설계된 모델을 RCMPSP에 맞게 이식하여 사용합니다.

### 비교 알고리즘
- **GA** (Genetic Algorithm): Random Key 방식의 유전 알고리즘
- **MIP/CP-SAT**: Gurobi MIP + OR-Tools CP-SAT 정확해 솔버

---

## 1. FJSP → RCMPSP 주요 차이점

원본 DANIEL은 FJSP(Flexible Job-Shop Scheduling Problem)를 위해 설계되었습니다.
RCMPSP는 FJSP와 구조적으로 유사하지만 다음과 같은 핵심 차이가 있으며,
각각 상태 표현과 환경 로직에 반영되었습니다.

### 1.1 Precedence: 선형 체인 → 임의 DAG

| | FJSP (원본) | RCMPSP (이 프로젝트) |
|---|---|---|
| **구조** | Job 내 Operation이 **선형 체인** (O₁→O₂→...→Oₙ) | Project 내 Activity가 **임의 DAG** (다수 선행자 가능) |
| **선행 조건** | 항상 직전 1개 operation만 확인 | `activity_predecessors` 리스트의 **모든** 선행자 완료 확인 |
| **후행자** | 항상 직후 1개 operation | 임의 개수의 후행자 가능 |

**환경 반영 (`scheduling_env.py`)**:
- `_update_available_actions()`: `activity_predecessors[b, a, :]` 텐서를 gather하여 모든 선행자가 ended인지 벡터화 확인
- FJSP는 `job_first_op_id`, `job_last_op_id`로 체인을 추적하지만, RCMPSP는 DAG이므로 매 step `ready` 마스크를 재계산

**Activity Feature 반영**:
- `f6` (predecessor completion ratio): `pred_done_count / total_pred_count` — 다수 선행자의 완료 비율을 피처로 제공

**act_mask 반영** (Activity Attention Block용):
```
FJSP:   op_mask[op, 0] = "체인의 첫 번째 op" (이전 op 없음)
        op_mask[op, 2] = "체인의 마지막 op" (다음 op 없음)
        → 체인 내 (이전, 자신, 다음) 3-이웃 어텐션에 정확히 대응

RCMPSP: act_mask[act, 0] = DAG source (선행자 없음) → pred_agg 스킵
        act_mask[act, 2] = DAG sink (후행자 없음) → succ_agg 스킵
        → 실제 DAG 구조 기반: activity_predecessors/successors 존재 여부로 결정
        → 패딩 activity는 전부 mask (1.0)
```

### 1.2 Mutex 제약 (RCMPSP 고유)

FJSP에는 **mutex 제약이 없습니다**. RCMPSP에서 추가된 제약입니다.

> Mutex: 특정 activity 쌍은 **동시에 실행할 수 없음** (e.g., 같은 장소를 사용하는 작업)

**환경 반영**:
```python
# _update_available_actions() — 가용성 제약 중 하나
mutex_data = self.activity_mutex[batch_idxs]        # (B, N, max_mutex)
mutex_partner_started = gather(started, mutex_ids)
mutex_partner_ended = gather(ended, mutex_ids)
mutex_running = started & ~ended & valid_mask       # 실행 중인 mutex 파트너
mutex_ok = ~mutex_running.any(dim=2)                # 하나라도 실행 중이면 불가
```

**상태 반영**:
- `dynamic_pair_mask`에 `~mutex_ok` 포함 (기본) → mutex 파트너 실행 중이면 해당 pair 마스킹
- 모델은 마스킹된 action에 `-inf` 부여 → **mutex 위반 행동을 원천 차단**
- **`allow_wait_mutex=True`** 옵션: mutex 조건을 pair mask에서 제거하여, 팀이 mutex 파트너 종료까지 대기 후 스케줄 가능

### 1.3 Release Time & Due Date (RCMPSP 고유)

FJSP에는 **시간 제약이 없습니다**. 모든 job이 시간 0부터 실행 가능하고 납기가 없습니다.

> RCMPSP: 각 프로젝트에 `release_time` (시작 가능 시점)과 `due_date` (납기)가 존재

**환경 반영**:
```python
# 가용성 제약: release_time 이전에는 프로젝트의 activity를 시작할 수 없음
act_release = gather(project_release_time, activity_project)
released = act_release <= current_time

# 목적함수: Tardiness = max(0, completion_time - due_date)
tardiness = clamp(project_completion_time - project_due_date, min=0).sum()
```

**상태 반영**:
- `dynamic_pair_mask`: `~act_released` 포함 (기본) → release 전 activity의 pair 마스킹
- **`allow_wait_release=True`** 옵션: release 조건을 pair mask에서 제거하여, 팀이 release time까지 대기 후 스케줄 가능
- Pair Feature `p5`: **(due_date - 예상 완료시간) / max_time** — 납기 슬랙
- Pair Feature `p7`: **(team_wait + release_wait [+ mutex_wait]) / max_time** — 총 대기 시간

### 1.4 Project 개념 (FJSP의 Job과의 차이)

| | FJSP Job | RCMPSP Project |
|---|---|---|
| **내부 구조** | 선형 체인 | 임의 DAG |
| **시간 제약** | 없음 | release_time, due_date |
| **candidate 의미** | "체인에서 다음 operation" | Identity mapping: 전체 activity (마스킹으로 제어) |
| **목적함수** | Makespan only | Tardiness (프로젝트별 납기 초과) 또는 Makespan |

**상태에 Project 정보를 반영하는 방법**:

1. **Candidate = Activity Identity** (`_get_state_daniel()`):
   ```python
   # Candidate = 전체 activity index (0..N-1). 마스킹은 dynamic_pair_mask에서 처리.
   candidate = torch.arange(N).unsqueeze(0).expand(B, -1)  # (B, N)
   ```
   RCMPSP는 DAG 기반이므로 프로젝트 내에 동시 schedulable activity가 여러 개 존재할 수 있음.
   따라서 프로젝트별 1개가 아닌 **전체 activity**를 candidate로 사용하고, `dynamic_pair_mask`로 제어.

2. **Activity Feature에 프로젝트 통계 포함**:
   - `f9`: 해당 activity가 속한 프로젝트의 **남은 activity 비율** (rem_count / tot_count)
   - `f10`: 해당 activity가 속한 프로젝트의 **남은 작업량 비율** (rem_work / tot_work)

3. **Pair Feature에 프로젝트 납기 정보 포함**:
   - `p5`: 해당 프로젝트의 **납기 슬랙** (due_date - est_completion)
   - `p6`: 해당 activity의 작업량이 **프로젝트 남은 작업 대비 비율**
   - `p7`: **release time 대기 [+ mutex 대기]** 포함 총 대기 시간

4. **Wait 옵션** (`allow_wait_release`, `allow_wait_mutex`):
   - False: release time 미도래 또는 mutex 파트너 실행 중인 activity는 pair mask에서 제외
   - True (현재 기본값): 해당 제약을 pair mask에서 제거하여 "기다려서 스케줄" 허용. `_schedule_activity()`에서 delayed start 처리
   - Wait 옵션 활성화 시 team 비가용 마스킹도 제거됨 (바쁜 팀에도 미래 시작 시간으로 assign 가능)

5. **Dominance Rule** (`dominance_rule`):
   - Wait 옵션 활성화 시, 대기하는 pair 중 idle time 동안 다른 즉시 실행 가능한 activity를 할 수 있는 경우 해당 대기 pair를 제외

---

## 2. MDP 구조

### 2.1 State (상태)

State는 DANIEL 모델의 입력 형태인 `EnvState` 데이터클래스로 표현됩니다.
Activity, Team, Pair 정보를 분리된 텐서로 구성하며, 관계 정보는 마스크와 인덱스 텐서로 표현합니다.

#### 제약 조건의 반영 방식
1. **Precedence** (선행 관계): `dynamic_pair_mask`에서 선행자 미완료 시 마스킹 + `pred_idx`/`succ_idx`로 DAG 구조 전달
2. **Mutex** (동시 실행 불가): `dynamic_pair_mask`에서 mutex 파트너 실행 중 시 마스킹
3. **Eligible** (팀 수행 가능): `dynamic_pair_mask`에서 비eligible pair 마스킹
4. **Release time / Due date**: `dynamic_pair_mask` + pair feature에 반영

#### EnvState 텐서 구조

```python
@dataclass
class EnvState:
    fea_act_tensor:           Tensor  # (B, N, 14)   — activity features
    act_mask_tensor:          Tensor  # (B, N, 3)    — activity attention mask
    fea_team_tensor:          Tensor  # (B, T, 8)    — team features
    team_mask_tensor:         Tensor  # (B, T, T)    — team attention mask
    dynamic_pair_mask_tensor: Tensor  # (B, N, T)    — 불가능 pair 마스크
    comp_idx_tensor:          Tensor  # (B, T, T, N) — team 경쟁 인덱스
    candidate_tensor:         Tensor  # (B, N)       — activity identity (0..N-1)
    fea_pairs_tensor:         Tensor  # (B, N, T, 8) — pair feature
    pred_idx_tensor:          Tensor  # (B, N, max_preds) — predecessor 인덱스
    succ_idx_tensor:          Tensor  # (B, N, max_succs) — successor 인덱스
```

#### Activity Feature (`fea_act`) — 14차원

```
인덱스  이름                    설명
  0    started                 시작 여부 (0 or 1)
  1    ended                   완료 여부 (0 or 1)
  2    norm_duration           duration / max_duration
  3    min_pt                  eligible 팀 중 최소 processing time / max_duration
  4    pt_span                 eligible 팀 PT 범위 (max - min) / max_duration
  5    norm_remaining          remaining_time / max_duration
  6    pred_completion_ratio   선행자 완료 비율 (pred_done / total_pred) — DAG 반영
  7    ready                   현재 실행 가능 여부 (0 or 1)
  8    completion_lb           예상 완료 시간 하한 / max_time — Tardiness-aware ★
  9    proj_rem_act_ratio      프로젝트 내 남은 activity 비율
 10    proj_rem_work_ratio     프로젝트 내 남은 작업량 비율
 11    eligible_ratio          eligible 팀 수 / 전체 팀 수
 12    local_slack             (local_due - best_comp) / max_time — 양수=여유, 음수=지각 예상 ★
 13    remaining_after         backward critical path 길이 / max_time ★

★ Tardiness-aware features (f8, f12, f13):
  - stepwise reward 사용 시: _estimate_tardiness()에서 계산된 경합 반영 값 재사용 (캐시)
  - sparse reward 사용 시: forward DAG relaxation 인라인 계산
    - simple 모드: min_pt 기반 (DANIEL-style lower bound)
    - sbh 모드: avg_pt 기반 (Shifting Bottleneck Heuristic)
  - best_comp: 예상 완료 시간 (forward DAG propagation)
  - local_due: 프로젝트 due_date - remaining_after (activity별 마감)
  - remaining_after: backward critical path (후행 작업 체인 길이)

* 정규화: 유효 activity 기준 mean/std 정규화 (패딩 노드 제외)
* 패딩 노드는 모두 0으로 설정
```

#### Team Feature (`fea_team`) — 8차원

```
인덱스  이름                    설명
  0    avail_cand_ratio        현재 가용한 activity pair 수 / N
  1    compat_unstarted_ratio  eligible한 미시작 activity 수 / N
  2    min_elig_dur            eligible 미시작 activity 중 최소 duration / max_dur
  3    mean_elig_dur           eligible 미시작 activity 중 평균 duration / max_dur
  4    idle_time               max(0, current - avail_time) / max_time (유휴 시간)
  5    wait_time               max(0, avail_time - current) / max_dur (대기 시간)
  6    norm_avail_time         avail_time / max_time
  7    is_busy                 avail_time > current_time (0 or 1)

* 정규화: 팀 축 기준 mean/std 정규화
```

#### Pair Feature (`fea_pairs`) — 8차원

```
(activity, team) 조합별 피처

인덱스  이름                 설명
  0    norm_pt_global       processing_time / max_duration (전역 정규화)
  1    norm_pt_per_act      processing_time / 해당 activity 최대 PT
  2    norm_pt_per_team     processing_time / 해당 team 최대 PT
  3    norm_pt_global_max   processing_time / 전체 최대 PT
  4    team_workload_ratio  processing_time / (team_remaining + PT) (팀 부하 비율)
  5    due_date_slack       (due_date - est_completion) / max_time — RCMPSP 핵심
  6    proj_work_ratio      processing_time / 프로젝트 남은 작업량
  7    total_wait_time      (team_wait + release_wait [+ mutex_wait]) / max_time — Wait 반영

* 마스킹된 pair는 0으로 설정
* allow_wait_mutex=True일 때 p7에 mutex_wait 추가
```

#### `dynamic_pair_mask` — 실행 불가 pair 마스크 (B, N, T)

`True` = 마스킹 (실행 불가). 기본 조건 + wait 옵션에 따라 조건 추가/제거:

```
기본 마스킹 (항상 적용):
1. ~valid_act:          패딩 activity
2. activity_started:    이미 시작된 activity
3. ~all_preds_done:     선행자 미완료
4. ~eligible:           팀이 해당 activity를 수행할 수 없음

조건부 마스킹:
5. ~act_released:       release time 미도래  (allow_wait_release=False일 때만)
6. ~mutex_ok:           mutex 파트너 실행 중  (allow_wait_mutex=False일 때만)

Dominance rule (allow_wait + dominance_rule=True일 때):
7. dominated:           대기 idle time 동안 다른 즉시 실행 가능 activity를 할 수 있는 pair
```

#### `candidate` 텐서 — Activity Identity Mapping

```python
candidate: (B, N)
# candidate = torch.arange(N).unsqueeze(0).expand(B, -1)
# 전체 activity를 candidate로 사용. 필터링은 dynamic_pair_mask에서 처리.
# 모델의 J 차원이 N으로 자동 적응 (sz_b, M, _, J = comp_idx.size())

env.daniel_candidate  # (B, N) — identity mapping이므로 action 역변환 시 직접 사용
# action_flat → act_idx = action_flat // N_T, team_idx = action_flat % N_T
```

#### `comp_idx` — Team 간 경쟁 인덱스

```python
comp_idx: (B, T, T, N)
# comp_idx[b, k, q, n] = 1.0  ←  team k와 team q 모두
#                              activity n을 수행할 수 있을 때 (두 팀이 경쟁 관계)
# comp_idx[b, k, q, n] = 0.0  ←  경쟁 없음

# 계산:
avail_t  = avail_pair.permute(0, 2, 1).float()  # (B, T, N)
comp_idx = avail_t.unsqueeze(2) * avail_t.unsqueeze(1)  # (B, T, T, N)
```

#### 시뮬레이션 상태

```python
sim_time: Tensor      # (batch_size,) - 현재 시뮬레이션 시간
step_count: Tensor    # (batch_size,) - 스텝 카운터
done: Tensor          # (batch_size,) - 종료 여부
```

### 2.2 Action (행동)

#### Action Space

**Action**: `(Activity, Team)` 페어 선택
- 형식: 1D flat 인덱스 (0 ~ N*T-1)
- 매핑: `act_idx = action // N_T`, `team_idx = action % N_T`

```python
# Action space 구조
# N: 최대 activity 수, T: team 수
# flat action index → (activity, team) 직접 분해
action_flat: Tensor             # (batch_size,) — 0 ~ N*T-1
act_idx  = action_flat // N_T   # 어느 activity인지
team_idx = action_flat % N_T    # 어느 팀인지
```

#### Action Feasibility 조건

Action이 실행 가능하려면 다음 조건을 **모두** 만족해야 합니다:

1. **Eligibility**: `activity_eligible_teams[b, act, team] == True` — 해당 Team이 해당 Activity를 수행 가능
2. **Not Started**: `activity_started[b, act] == False` — 아직 시작되지 않은 Activity
3. **Project Released**: `project_release_time[b, project] <= sim_time[b]` — Activity가 속한 Project가 release됨
4. **Predecessors Completed**: 모든 선행 작업 완료
5. **Mutex Not Running**: Mutex 관계인 Activity가 실행 중이지 않음
6. **Team Available**: `team_available_time[b, team] <= sim_time[b]` — 해당 Team이 현재 사용 가능

#### Action Mask

```python
dynamic_pair_mask: Tensor       # (batch_size, N, T)
                                # True: 실행 불가능한 pair (마스킹)
                                # False: 실행 가능한 pair
# Actor 출력에서 masked pair에 -inf 부여 → softmax 후 확률 0
```

### 2.3 Reward (보상)

#### Sparse Reward (기본)

에피소드 종료 시점에만 보상 제공:

```python
# 에피소드 진행 중
reward = None

# 에피소드 종료 시 (모든 Activity 완료)
if objective == 'tardiness':
    reward = -total_tardiness    # total_tardiness = sum(max(0, completion_time - due_date))

elif objective == 'makespan':
    reward = -makespan           # makespan = max(completion_time)
```

#### Stepwise Reward (dense, tardiness 전용)

매 step마다 estimated tardiness 변화량을 보상으로 제공:
```python
# r_t = est_tardiness(s_t) - est_tardiness(s_{t+1})
# tardiness 추정이 줄어들면 양의 보상
# 에피소드 마지막 step은 실제 tardiness를 사용
```

Tardiness 추정 방식 (`tardiness_est_type`):
- **`simple`** (기본): Forward DAG relaxation + min_pt. 리소스 경합 미반영 (lower bound)
- **`sbh`**: Shifting Bottleneck Heuristic 7-Phase. ATC 우선순위 + team contention delay 반영. 더 정확하지만 계산 비용 높음

> `stepwise` reward 사용 시 추정 결과는 캐시되어 Activity Feature (f8, f12, f13) 계산에 재사용됩니다.

#### 목적함수

**Tardiness (지연 시간)**:
```python
obj = sum_{p=1}^{N_P} max(0, completion_time[p] - due_date[p])
```

**Makespan (총 완료 시간)**:
```python
obj = max_{p=1}^{N_P} completion_time[p]
```

### 2.4 State Transition (상태 전이)

#### DES (Discrete Event Simulation) 방식

**Simulation Time**: 연속 시간 (float)
**Decision Points**: Activity 시작 시점

```
1. Action 선택: (activity_id, team_id)
   ↓
2. Activity 스케줄링
   - activity_started[b, activity_id] = True
   - activity_start_time[b, activity_id] = sim_time[b]
   - activity_end_time[b, activity_id] = sim_time[b] + duration
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
   - project_completion_time 업데이트
   ↓
5. 종료 조건 확인
   - 모든 Activity 완료 → done = True
   - 아니면 → 3번으로 돌아가기
```

#### 시간 진행 로직

```python
def get_next_move_t(batch_idxs):
    """다음 이벤트까지의 시간 계산"""
    remaining_times = activity_remaining_time[batch_idxs]   # (n_batch, max_N_A)
    masked_times = where(remaining_times > 0, remaining_times, inf)
    time_delta = min(masked_times, dim=1)  # 가장 먼저 완료될 activity
    return time_delta
```

#### State Transition Diagram

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
│  - activity 시작, team 할당, end_time 계산          │
└─────────────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────┐
│  Time Advance (DES)                                 │
│  - 다음 decision point까지 시간 진행                │
│  - activity 완료 처리, project 완료 확인            │
└─────────────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────┐
│  State s_{t+1} (EnvState)                           │
│  - fea_act/fea_team/fea_pairs 업데이트              │
│  - dynamic_pair_mask 재계산                         │
│  - sim_time = t + Δt                                │
└─────────────────────────────────────────────────────┘
                    │
                    │ if all activities ended
                    ↓
                 Terminal State → Reward
```

### 2.5 MDP 요약

| 구성 요소 | 설명 | 차원/형태 |
|----------|------|------------|
| **State** | EnvState (Activity, Team, Pair 텐서 + 마스크) | 10개 텐서 |
| **Action** | (Activity, Team) 페어 선택 | flat 인덱스 (0 ~ N*T-1) |
| **Reward** | -Tardiness 또는 -Makespan (에피소드 끝) | Scalar |
| **Transition** | DES 기반 시뮬레이션 | Deterministic |
| **Horizon** | Variable (모든 Activity 완료 시 종료) | ~50-200 steps |

핵심 특징:
1. **Deterministic MDP**: Action에 대한 전이가 결정적
2. **Sparse Reward** (기본): 에피소드 끝에만 보상 / Stepwise도 지원
3. **Variable Episode Length**: 문제 크기에 따라 가변적
4. **Combinatorial Action Space**: Eligible 기반으로 축소
5. **Tensor-structured State**: 분리된 텐서 (Activity, Team, Pair) + 마스크로 관계 표현

---

## 3. DANIEL 모델 아키텍처

### 3.1 파일 구조

```
model/
├── __init__.py          # 패키지 초기화
├── main_model.py        # DANIEL 최상위 모델 + DualAttentionNetwork
├── attention_layer.py   # Activity/Team Attention Block (핵심 레이어)
└── sub_layers.py        # Actor, Critic, MLP 구현

ppo_utils.py             # PPO 전용 유틸리티 (PPOMemory, eval_actions)
```

### 3.2 전체 아키텍처

```
입력 (환경 → EnvState 텐서)
│
├─ fea_act    [B, N, 14]   Activity feature (14차원)
├─ act_mask   [B, N, 3]    Activity attention mask
├─ candidate  [B, N]       activity identity (0..N-1)
├─ fea_team   [B, T, 8]    Team feature (8차원)
├─ team_mask  [B, T, T]    Team attention mask
├─ comp_idx   [B, T, T, N] Team 간 경쟁 인덱스
├─ pair_mask  [B, N, T]    실행 불가 pair 마스크
└─ fea_pairs  [B, N, T, 8] Pair feature (8차원)
│
▼
┌──────────────────────────────────────────────────────┐
│  DualAttentionNetwork (레이어 × L회 반복)            │
│                                                      │
│  Layer i:                                            │
│    ① candidate activity 피처 수집                    │
│       fea_j_jc = gather(fea_act, candidate)          │
│                                                      │
│    ② comp_val 계산 (team 경쟁 강도)                  │
│       comp_val = comp_idx @ fea_j_jc                 │
│       shape: [B, T, T, dim_i]                        │
│                                                      │
│    ③ Activity Attention Block                        │
│       fea_act' = MultiHeadOpAttnBlock(fea_act, mask) │
│                                                      │
│    ④ Team Attention Block                            │
│       fea_team' = MultiHeadMchAttnBlock(             │
│                     fea_team, team_mask, comp_val)   │
│                                                      │
│  출력:                                               │
│    fea_act_enc   [B, N, d]   activity 임베딩         │
│    fea_team_enc  [B, T, d]   team 임베딩             │
│    fea_act_g     [B, d]      global activity 임베딩  │
│    fea_team_g    [B, d]      global team 임베딩      │
└──────────────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│  Action Feature 조립 (Actor 입력 구성)               │
│                                                      │
│  후보 activity 임베딩 수집:                          │
│    Fea_j_JC = gather(fea_act_enc, candidate)         │
│    shape: [B, N, d]                                  │
│                                                      │
│  N×T 직렬화 (모든 조합 나열):                        │
│    Fea_j_JC_serial  [B, N*T, d]                      │
│    Fea_m_serial     [B, N*T, d]                      │
│    Fea_Gj_expand    [B, N*T, d]                      │
│    Fea_Gm_expand    [B, N*T, d]                      │
│    fea_pairs_flat   [B, N*T, 8]                      │
│                                                      │
│  concat → candidate_feature [B, N*T, 4d+8]          │
└──────────────────────────────────────────────────────┘
│                         │
▼                         ▼
┌─────────────┐     ┌─────────────┐
│  Actor MLP  │     │ Critic MLP  │
│ [B,N*T,4d+8]│     │  [B, 2d]    │
│      ↓      │     │      ↓      │
│  [B, N*T]   │     │   [B, 1]    │
│  (raw logit)│     │  (가치 v)   │
└─────────────┘     └─────────────┘
│
▼
pair_mask로 마스킹 (-inf)
↓
Softmax → π  [B, N*T]

출력:
  π  [B, N*T]  — 행동 확률 (activity×team 조합별)
  v  [B, 1]    — 상태 가치
```

### 3.3 각 파일별 상세

#### `model/main_model.py` — DANIEL 최상위 모델

**`nonzero_averaging(x)`** — 유틸리티 함수
```
목적: 비영(non-zero) 벡터만 골라 평균을 냄 (패딩된 노드 제외)
입력: x [B, node_num, d]
출력: [B, d]  ← 유효 노드들의 평균 임베딩
```

**`DualAttentionNetwork`** — 이중 어텐션 인코더
```
목적: Activity 피처와 Team 피처를 교차 어텐션으로 인코딩
구성:
  - N개의 Activity Attention Block (op_attention_blocks)
  - N개의 Team Attention Block (mch_attention_blocks)
  두 블록이 레이어마다 번갈아 상호 영향을 주며 업데이트
```

**`DANIEL`** — 최상위 모델 (정책 + 가치 함수)
```
구성 요소:
  - feature_exact: DualAttentionNetwork (인코더)
  - actor:         MLP (행동 확률 계산)
  - critic:        MLP (상태 가치 추정)

forward() 입력:
  fea_j             (B, N, 14) — activity feature
  op_mask           (B, N, 3)  — activity attention mask
  candidate         (B, N)     — activity identity (0..N-1)
  fea_m             (B, T, 8)  — team feature
  mch_mask          (B, T, T)  — team attention mask
  comp_idx          (B, T, T, N) — 경쟁 인덱스
  dynamic_pair_mask (B, N, T)  — 실행 불가 pair 마스크 (True=masked)
  fea_pairs         (B, N, T, 8) — (activity, team) pair feature

forward() 출력:
  pi  (B, N*T) — 각 (activity, team) 조합의 행동 확률 (softmax 후)
  v   (B, 1)   — 현재 상태의 가치 추정
```

#### `model/attention_layer.py` — 이중 어텐션 블록

**Activity Attention Block (Op Attention)**:
```
목적: Activity 간의 순서/경쟁 관계를 학습

SingleOpAttnBlock:
  - 각 activity가 자신의 이전(roll −1), 자신, 다음(roll +1) 노드에 어텐션
  - op_mask: 존재하지 않는 선행/후행 노드를 -inf로 마스킹
  - 구현 핵심:
      Wh = h @ W                             # [B, N, out]
      e  = LeakyReLU(Wh1 + [Wh2_prev, Wh2, Wh2_next])
      attention = softmax(where(mask, -∞, e))
      h_new = attention @ Wh_concat

MultiHeadOpAttnBlock:
  - num_heads개의 SingleOpAttnBlock을 병렬 실행
  - concat=True  → 헤드별 출력 이어 붙이기 (중간 레이어)
  - concat=False → 헤드별 출력 평균 (마지막 레이어)
```

**Team Attention Block (Mch Attention)**:
```
목적: Team 간 경쟁 관계를 학습 (어떤 team이 같은 activity를 두고 경쟁하는가)

SingleMchAttnBlock:
  - node feature (fea_m) + edge feature (comp_val) 동시 활용
  - comp_val: team k와 team q가 동일한 candidate activity를 두고 경쟁하는 정도
  - 구현 핵심:
      Wh     = h     @ W        # [B, M, out]  (team node)
      W_edge = comp  @ W_edge   # [B, M, M, out] (edge)
      e = LeakyReLU(Wh1 + Wh2^T + edge_feas)
      attention = softmax(where(mask, e, -∞))
      h' = attention @ Wh

MultiHeadMchAttnBlock:
  - MultiHeadOpAttnBlock과 동일한 구조 (헤드 병렬화)
```

#### `model/sub_layers.py` — 기반 레이어

```
MLP(num_layers, input_dim, hidden_dim, output_dim)
  - ReLU 활성화 함수
  - 일반적인 특징 변환에 사용

Actor(num_layers, input_dim, hidden_dim, output_dim)
  - tanh 활성화 함수
  - (activity, team) pair 스코어 계산에 사용
  - input:  [B, N*T, 4*d + 8]  (candidate + team + global + pair feature)
  - output: [B, N*T, 1]        (행동 로짓)

Critic(num_layers, input_dim, hidden_dim, output_dim)
  - tanh 활성화 함수
  - 상태 가치 추정에 사용
  - input:  [B, 2*d]  (global activity + global team embedding)
  - output: [B, 1]    (가치 스칼라)
```

### 3.4 model_params 구조

```python
# 논문 원본 파라미터 (DANIEL, ~28K params)
model_params = {
    'fea_act_input_dim':  14,    # Activity feature 차원 (환경 고정값)
    'fea_team_input_dim': 8,     # Team feature 차원 (환경 고정값)
    'num_heads_AAB':  [4, 4],    # Activity Attention Block: 레이어별 헤드 수
    'num_heads_TAB':  [4, 4],    # Team Attention Block: 레이어별 헤드 수
    'layer_fea_output_dim': [32, 8],  # 레이어별 출력 차원
    'dropout_prob':    0.0,
    'num_mlp_layers_actor':  3,
    'hidden_dim_actor':     64,
    'num_mlp_layers_critic': 3,
    'hidden_dim_critic':    64,
}

# RTX 5090 확장 파라미터 (~300K params)
model_params = {
    'fea_act_input_dim': 14,
    'fea_team_input_dim': 8,
    'num_heads_AAB': [8, 8, 8],       # 3층
    'num_heads_TAB': [8, 8, 8],       # 3층
    'layer_fea_output_dim': [128, 64, 32],  # 3층
    'dropout_prob': 0.0,
    'num_mlp_layers_actor': 3,
    'hidden_dim_actor': 256,
    'num_mlp_layers_critic': 3,
    'hidden_dim_critic': 256,
}
```

> **파라미터 이름 매핑**: `train.py`/`test.py`의 RCMPSP 이름과
> DANIEL 내부 코드의 원본 FJSP 이름이 다릅니다.
> `trainer.py`에서 `SimpleNamespace`로 변환합니다:
>
> | train.py (RCMPSP 이름) | model/main_model.py (DANIEL 원본) |
> |------------------------|-----------------------------------|
> | `fea_act_input_dim`    | `fea_j_input_dim`                 |
> | `fea_team_input_dim`   | `fea_m_input_dim`                 |
> | `num_heads_AAB`        | `num_heads_OAB`                   |
> | `num_heads_TAB`        | `num_heads_MAB`                   |

---

## 4. 학습 알고리즘

### 4.1 알고리즘 선택

```python
# train.py 상단에서 설정
ALGORITHM_TYPE = 'reinforce'  # REINFORCE + POMO baseline (현재 기본값)
ALGORITHM_TYPE = 'ppo'        # PPO-Clip + GAE

# Reward 방식
REWARD_TYPE = 'sparse'        # 'sparse' (에피소드 끝에만) 또는 'stepwise' (매 step dense)

# Wait / Dominance 옵션 (현재 기본값: 모두 True)
ALLOW_WAIT_RELEASE = True     # release time 미래 activity도 대기 후 스케줄 허용
ALLOW_WAIT_MUTEX = True       # mutex 파트너 실행 중 activity도 대기 후 스케줄 허용
DOMINANCE_RULE = True         # 대기 pair의 dominance 필터링

# Tardiness 추정 방식 (stepwise reward + feature 계산에 사용)
TARDINESS_EST_TYPE = 'simple' # 'simple': forward DAG + min_pt (lower bound)
                              # 'sbh': Shifting Bottleneck Heuristic 7-Phase (ATC + contention)
```

### 4.2 환경-모델 데이터 흐름

```
train.py (파라미터 설정)
    └─ Scheduling_Trainer.__init__()
         ├─ SchedulingEnv(env_params)          # 환경 생성
         ├─ DANIEL(config)                     # 모델 생성
         └─ run()
              └─ REINFORCE: _train_one_batch() → train_one_minibatch()
                 PPO:        _train_ppo_one_batch()
                   ├─ env._reset(problem)      # 새 문제 로드
                   ├─ env._get_state()         # 상태 추출 → EnvState
                   ├─ model.forward(state)     # 행동 확률 계산
                   ├─ Categorical(π).sample()  # 행동 샘플링
                   ├─ env.step_pair(a, t)      # (activity, team) 직접
                   └─ REINFORCE: loss = -advantage * log_prob
                      PPO: PPOMemory → GAE → K-epoch clip update
```

### 4.3 REINFORCE (POMO Baseline)

DANIEL의 Critic(`v`)을 **사용하지 않는** 단순 정책 경사 알고리즘입니다.

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

# Episode 종료 시
reward = -objective_value         # 최소화 → 최대화 변환

# Baseline 옵션 (trainer_params['baseline_type']):
#   'pomo': 동일 인스턴스의 POMO 롤아웃 평균 (기본값)
#   'batch': 미니배치 전체 평균
#   'none': baseline 없음

advantage = reward - baseline
loss = -advantage * sum(log_probs)
```

### 4.4 PPO-Clip (DANIEL 전용)

DANIEL 원본 논문의 학습 알고리즘입니다.
Critic(`v`)을 **활용하여** value function 학습과 GAE advantage를 사용합니다.

#### PPO 하이퍼파라미터

```python
PPO_EPS_CLIP       = 0.2        # clipping parameter ε
PPO_K_EPOCHS       = 4          # 에피소드당 업데이트 epoch 수 K
PPO_GAE_LAMBDA     = 0.98       # GAE parameter λ
PPO_GAMMA          = 1.0        # discount factor γ (스케줄링: 에피소드 보상이므로 1.0)
PPO_VLOSS_COEF     = 0.5        # value loss 계수
PPO_PLOSS_COEF     = 1.0        # policy loss 계수
PPO_TAU            = 0.0        # 0.0 = hard copy (standard), >0 = soft update
PPO_MINIBATCH_SIZE = 4096       # mini-batch 크기
N_RESAMPLE         = 20         # 학습 데이터 리샘플링 주기 N_r
ENTROPY_COEF       = 0.01       # entropy regularization 계수
```

#### PPO 학습 흐름 (`_train_ppo_one_batch`)

```
1. Hard Copy: policy_old ← policy (현재 파라미터 복사)

2. Rollout (no_grad):
   while not done:
     π, v = model(state)           # 현재 정책으로 행동 확률 + 가치 계산
     action ~ Categorical(π)       # 확률적 행동 선택
     memory.push_state(state)      # 상태 저장 (8개 텐서)
     state, obj, done = env.step_pair(activity, team)
     reward = -obj if done else 0  # 희소 보상 (에피소드 끝에만)
     memory.push_transition(action, log_prob, v, reward, done)

3. GAE Advantage 계산:
   t_data = memory.transpose_data()     # [T, B] → [B*T, ...] 평탄화
   advantage, v_target = memory.get_gae_advantages()
   advantage = normalize(advantage)     # batch 정규화

4. K-Epoch Mini-Batch Update:
   for k in range(K_EPOCHS):
     for mini_batch in split(t_data, MINIBATCH_SIZE):
       π_new, v_new = model(state_batch)   # 현재(갱신 중) 정책
       new_logprobs = log π_new(action)

       # PPO Clipped Surrogate Loss
       ratio = exp(new_logprobs - old_logprobs)
       surr1 = ratio * advantage
       surr2 = clamp(ratio, 1-ε, 1+ε) * advantage
       policy_loss = -min(surr1, surr2).mean()

       # Value Loss (MSE)
       value_loss = MSE(v_new, v_target)

       # Entropy Bonus
       entropy_loss = -entropy

       # Total Loss
       loss = ploss_coef * policy_loss
            + vloss_coef * value_loss
            + entropy_coef * entropy_loss

5. (선택) Soft Update: policy_old ← τ * old + (1-τ) * new
```

#### PPO Memory 구조 (`ppo_utils.py`)

```
PPOMemory 저장 구조:
  상태 시퀀스 (8개):
    fea_act_seq, act_mask_seq, fea_team_seq, team_mask_seq,
    dynamic_pair_mask_seq, comp_idx_seq, candidate_seq, fea_pairs_seq

  전이 시퀀스 (5개):
    action_seq, log_probs, val_seq, reward_seq, done_seq

  transpose_data() 변환:
    [T steps, B envs, ...] → [B*T, ...] (mini-batch 학습용)

  get_gae_advantages():
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)           # TD error
    A_t = δ_t + (γλ) * A_{t+1}                      # GAE 역순 누적
    v_target = A_t + V_old(s_t)                      # value regression target
    A_t = normalize(A_t, dim=time)                   # 시간 축 정규화

eval_actions(pi, actions):
  - 주어진 행동에 대한 log probability와 entropy 계산
  - 입력: pi [B, N*T] (softmax된 정책), actions [B] (행동 인덱스)
  - 출력: (log_probs [B], entropy scalar)
```

#### REINFORCE vs PPO 비교

| 항목 | REINFORCE | PPO-Clip |
|------|-----------|----------|
| **Critic 사용** | 미사용 (forward에서 v 계산하지만 loss에 불포함) | 사용 (value loss + GAE advantage) |
| **Baseline** | POMO rollout 평균 | Value function V(s) |
| **정책 업데이트** | 1회 (에피소드당) | K회 mini-batch (에피소드당) |
| **보상 구조** | 에피소드 종료 시 -obj | 희소 보상 (step별 0, 종료 시 -obj) |
| **데이터 효율** | 낮음 (on-policy, 1회 사용 후 폐기) | 높음 (K epoch 재사용) |
| **POMO 지원** | O (pomo_size > 1) | X (pomo_size = 1 권장) |
| **학습 데이터** | 매 epoch 새로 생성 | N_r epoch마다 리샘플링 |

### 4.5 Validation 평가

학습 루프와 동일하나 `argmax`로 greedy 선택:

```python
action_flat = torch.argmax(pi, dim=1)   # greedy (sampling 대신)
```

### 4.6 MDP 관점에서의 DANIEL 역할

```
State s_t (EnvState)
     │
     ▼
DANIEL.forward(s_t)
  ├─ π(a | s_t)  [B, N*T]   ← Actor: 행동 선택에 사용
  └─ v(s_t)      [B, 1]     ← Critic: PPO에서 advantage 계산에 사용
     │
     ▼
Action 샘플링 (학습) 또는 argmax (테스트)
  action_flat → act_idx = action // N_T, team_idx = action % N_T
     │
     ▼
env.step_pair(activity_idx, team_idx)
     │
     ▼
State s_{t+1} (다음 EnvState 반환)
```

---

## 5. GA (Genetic Algorithm) — 비교 알고리즘

### 5.1 Solution Structure (Random Key 방식)

Random Key 방식은 **간접 표현(indirect representation)**을 사용하는 진화 알고리즘 인코딩 방법입니다.

```python
class Solution:
    def __init__(self, pairs: List[Pair]):
        self.pairs = pairs                    # 모든 (activity, team) 페어
        self.random_keys = [random.random()   # 각 페어에 대응하는 0~1 사이 실수
                           for _ in pairs]
        self.objective: float = float('inf')  # 목적함수 값 (작을수록 좋음)
        self.fitness: float = 0.0             # 적합도 (클수록 좋음)
        self.schedule: Dict = {}              # activity_id -> (start, end, team_id)
```

**Pair 구조**: 각 Activity마다 eligible_teams에 있는 팀과만 페어 생성.
Random Key가 작을수록 우선순위가 높음 (먼저 스케줄링 시도).

### 5.2 디코딩 절차

Solution의 Random Key를 실제 Schedule로 변환하는 과정입니다.

#### Batch Mode (기본)
1. 모든 페어를 random_key 순서로 정렬
2. 정렬된 순서대로 스케줄링 시도
3. 한 번 순회 완료 후, 다시 처음부터 순회
4. 더 이상 스케줄할 수 없을 때까지 반복

#### Immediate Mode (적극적)
1. 모든 페어를 random_key 순서로 정렬
2. 정렬된 순서대로 스케줄링 시도
3. Activity 하나라도 스케줄되면 즉시 처음부터 다시 시작
4. 더 이상 스케줄할 수 없을 때까지 반복

#### 디코딩 시 제약조건 체크 순서

1. **Precedence**: 모든 선행 작업이 스케줄되었는지 확인. 미완료 시 skip.
2. **Mutex**: 동시 수행 불가 activity가 스케줄되었으면 그 종료 후 시작.
3. **Resource**: 해당 팀의 가용 시간 이후에만 시작 가능.
4. **Release Time**: 프로젝트 시작 가능 시간 이후에만 시작 가능.

### 5.3 유전 연산자

**Crossover (Uniform Crossover)**:
- 각 페어의 random key를 부모 중 하나에서 50% 확률로 선택

**Mutation (Random Replacement)**:
- 전체 페어의 10%를 랜덤하게 선택하여 새로운 random key 할당

**Selection (Roulette Wheel)**:
```python
# Objective → Fitness 변환 (클수록 좋게)
solution.fitness = max_obj - solution.objective + epsilon
# fitness에 비례하는 확률로 선택
```

### 5.4 진화 알고리즘 흐름

```
1. 초기화: population_size개의 랜덤 solution 생성
2. 평가: 모든 solution 디코딩 + 목적함수 계산
3. 최적해 추적
4. 진화 루프 (generations회 반복):
   a) Elitism: 최고 solution 1개 보존
   b) 새로운 population 생성:
      ├─ Selection: Roulette Wheel로 parent1, parent2 선택
      ├─ Crossover: 확률 crossover_rate로 교차 수행
      └─ Mutation: 확률 mutation_rate로 변이 수행
   c) 평가: 새로운 population의 objective 계산
   d) 최적해 업데이트
5. 종료: 최종 best solution 반환
```

### 5.5 파라미터 설정

```python
GeneticAlgorithm(
    projects,
    num_teams,
    population_size=100,     # Population 크기
    generations=500,         # 진화 세대 수
    crossover_rate=0.8,      # 교차 확률
    mutation_rate=0.2,       # 변이 확률
    decode_mode="batch",     # "batch" 또는 "immediate"
)
```

### 5.6 목적함수

```python
# Tardiness 최소화
objective = sum(max(0, completion_time[p] - due_date[p]) for p in projects)

# 스케줄되지 않은 activity가 있으면 큰 벌점
if unscheduled:
    objective = len(unscheduled) * 100000
```

---

## 6. 코드 구조

### 6.1 핵심 파일 역할

| 파일 | 역할 |
|------|------|
| `train.py` | 학습 파라미터 설정 후 `Scheduling_Trainer.run()` 호출 |
| `test.py` | 체크포인트로 RL(greedy) 및 GA 성능 비교, Excel 저장 |
| `scheduling_env.py` | RCMPSP 환경 (DES 기반). `_reset()`, `step_pair()`, `_get_state()` |
| `model/main_model.py` | DANIEL 정책+가치 네트워크 |
| `model/attention_layer.py` | Activity/Team Attention Block |
| `model/sub_layers.py` | Actor, Critic, MLP |
| `trainer.py` | REINFORCE/PPO 학습 루프, validation, 체크포인트, WandB 로깅 |
| `ppo_utils.py` | PPO 롤아웃 버퍼(`PPOMemory`) 및 GAE advantage 계산 |
| `data_generator.py` | 문제 인스턴스 배치 생성 (`generate_scheduling_data_batch`) |
| `data_generation.py` | `data/test/` 폴더에 테스트 pickle 파일 생성 |
| `GA.py` | 비교용 유전 알고리즘 (Random Key 방식) |
| `samsung_MIP.py` | Gurobi MIP + OR-Tools CP-SAT 정확해 솔버 |
| `gantt_chart.py` | 결과 시각화 |
| `_validate_gen.py` | `data_generator` 및 `SchedulingEnv` sanity check |

### 6.2 환경 메서드 매핑

| 메서드 | 역할 |
|--------|------|
| `_reset()` | 환경 초기화, State 생성 |
| `step_pair(act_idx, team_idx)` | Action 실행, State Transition |
| `_schedule_activity()` | Activity 스케줄링 |
| `move_next_state()` | DES 시뮬레이션 (시간 진행) |
| `_get_state()` | EnvState 생성 (DANIEL 입력 텐서) |
| `_get_obj()` | 목적함수값 계산 |
| `_update_available_actions()` | Action Mask 업데이트 |

### 6.3 체크포인트 구조

학습 시 `checkpoints/{timestamp}_{objective}_{MODEL}_{ALG}/` 폴더 자동 생성.

- `epoch{N}.pt`: 5 epoch마다 저장
- `best_model.pt`: validation 기준 최적 모델
- `epoch0.pt`: 학습 전 초기 상태

체크포인트 dict: `model_state_dict`, `optimizer_state_dict`, `env_params`, `model_params`, `trainer_params`, `epoch`, `train_score`, `val_score`, `algorithm_type`

PPO 체크포인트는 `model_old_state_dict`도 포함합니다.

> **test.py에서 로드 시 `ALGORITHM_TYPE`을 학습 때와 반드시 동일하게 설정해야 합니다.**

---

## 7. 확장 시 유의사항

1. **`fea_act_input_dim`**: `_get_state_daniel()`에서 생성하는 feature 차원(14)과
   `train.py`의 `model_params['fea_act_input_dim']`(14)을 항상 일치시켜야 합니다.

2. **`fea_team_input_dim`**: 마찬가지로 환경 출력(8)과 `model_params`(8) 일치 필요.

3. **`layer_fea_output_dim`**: 마지막 레이어의 출력 차원이
   Actor 입력 차원(`4 * output_dim[-1] + 8`)과 Critic 입력 차원(`2 * output_dim[-1]`)을
   결정합니다. 변경 시 `pair_input_dim=8`은 고정임을 유의하세요.

4. **PPO 활용 시**: `ALGORITHM_TYPE='ppo'`로 설정하면 Critic의 value loss가 자동으로 포함됩니다.
   REINFORCE에서는 `v`를 손실에 포함하지 않습니다.

5. **체크포인트 호환성**: `ALGORITHM_TYPE`을 학습 때와 테스트 때 반드시 동일하게 설정해야 합니다
   (PPO 체크포인트에는 `model_old` 포함).

6. **Mutex/Release time 피처 추가**: 현재 mutex와 release time은 마스킹과 피처(pair feature p5, p7)로
   반영됩니다. Activity feature에는 local_slack(f12)과 remaining_after(f13)로 tardiness 관련 정보가 포함됩니다.
   추가적인 명시적 피처가 필요하면 `fea_act_input_dim`을 늘려야 합니다.

7. **`FJSP-DRL-main/`**: 원본 FJSP 프로젝트 참고용 폴더. 이 프로젝트에서는 `model/` 폴더에 이식된 버전만 사용합니다.

8. **벡터화 구현**: 모든 연산은 배치 처리되어 수백 개의 인스턴스를 동시에 시뮬레이션할 수 있습니다.
   ```python
   activity_remaining_time[batch_idxs] -= time_deltas.unsqueeze(1)
   just_completed_mask = (activity_remaining_time <= 0) & activity_started & ~activity_ended
   ```
