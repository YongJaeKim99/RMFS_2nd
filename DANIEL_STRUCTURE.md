# DANIEL 모델 구조 설계 문서

## 개요

**DANIEL** (Dual Attention Network for Integrated Scheduling)은
FJSP (Flexible Job-Shop Scheduling Problem)를 위해 설계된 딥러닝 기반 스케줄링 모델로,
이 프로젝트에서는 **RCMPSP (Resource-Constrained Multi-Project Scheduling Problem)** 에 맞게 이식되었습니다.

원본 출처: [FJSP-DRL-main](./FJSP-DRL-main/) (Liu et al., 2023)

---

## 1. FJSP → RCMPSP 주요 차이점과 반영 방법

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
- `f4` (predecessor completion ratio): `pred_done_count / total_pred_count` — 다수 선행자의 완료 비율을 피처로 제공하여 모델이 "얼마나 실행 가능에 가까운지" 학습 가능

**act_mask 반영** (Activity Attention Block용):
```
FJSP:   op_mask[op, 0] = "체인의 첫 번째 op" (이전 op 없음)
        op_mask[op, 2] = "체인의 마지막 op" (다음 op 없음)
        → 체인 내 (이전, 자신, 다음) 3-이웃 어텐션에 정확히 대응

RCMPSP: act_mask[act, 0] = "프로젝트 내 첫 번째 activity" (index 기준)
        act_mask[act, 2] = "프로젝트 내 마지막 activity" (index 기준)
        → DAG에는 "이전/다음"이 1:1이 아니므로, 프로젝트 내 순서 경계를 근사적으로 사용
        → roll(-1)/roll(+1)이 프로젝트 경계를 넘을 때 마스킹
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
   - `f7`: 해당 activity가 속한 프로젝트의 **남은 activity 비율** (rem_count / tot_count)
   - `f8`: 해당 activity가 속한 프로젝트의 **남은 작업량 비율** (rem_work / tot_work)
   → 모델이 "이 activity를 처리하면 프로젝트가 얼마나 진척되는지" 학습 가능

3. **Pair Feature에 프로젝트 납기 정보 포함**:
   - `p5`: 해당 프로젝트의 **납기 슬랙** (due_date - est_completion)
   - `p6`: 해당 activity의 작업량이 **프로젝트 남은 작업 대비 비율**
   - `p7`: **release time 대기 [+ mutex 대기]** 포함 총 대기 시간

4. **Action Space 구조** (N × T):
   - DANIEL의 action은 `(activity, team)` 조합
   - flat index에서 `activity_idx = action // N_T`, `team_idx = action % N_T`로 직접 분해
   - FJSP의 `(job, machine)` 구조를 계승하되, 프로젝트 단위가 아닌 **activity 단위 의사결정**

5. **Wait 옵션** (`allow_wait_release`, `allow_wait_mutex`):
   - 기본(False): release time 미도래 또는 mutex 파트너 실행 중인 activity는 pair mask에서 제외
   - True: 해당 제약을 pair mask에서 제거하여 "기다려서 스케줄" 허용. `_schedule_activity()`에서 delayed start 처리

6. **Dominance Rule** (`dominance_rule`):
   - Wait 옵션 활성화 시, 대기하는 pair 중 idle time 동안 다른 즉시 실행 가능한 activity를 할 수 있는 경우 해당 대기 pair를 제외

---

## 2. 파일 구조

```
model/
├── __init__.py          # 패키지 초기화 (비어 있음)
├── main_model.py        # DANIEL 최상위 모델 + DualAttentionNetwork
├── attention_layer.py   # Activity/Team Attention Block (핵심 레이어)
└── sub_layers.py        # Actor, Critic, MLP 구현

ppo_utils.py             # PPO 전용 유틸리티 (PPOMemory, eval_actions)
```

---

## 3. 각 파일 역할

### 3.1 `model/main_model.py` — DANIEL 최상위 모델

**담당**: 모델 전체 구조의 조립 및 forward pass 실행

#### 포함된 클래스

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
  fea_j             (B, N, 10) — activity feature
  op_mask           (B, N, 3)  — activity attention mask
  candidate         (B, N)     — activity identity (0..N-1)
  fea_m             (B, T, 8)  — team feature
  mch_mask          (B, T, T)  — team attention mask
  comp_idx          (B, T, T, N) — 경쟁 인덱스 (team 간 경쟁 강도)
  dynamic_pair_mask (B, N, T)  — 실행 불가 pair 마스크 (True=masked)
  fea_pairs         (B, N, T, 8) — (activity, team) pair feature

forward() 출력:
  pi  (B, N*T) — 각 (activity, team) 조합의 행동 확률 (softmax 후)
  v   (B, 1)   — 현재 상태의 가치 추정
```

---

### 3.2 `model/attention_layer.py` — 이중 어텐션 블록

**담당**: DualAttentionNetwork에서 사용하는 두 종류의 멀티헤드 어텐션 레이어

#### Activity Attention Block (Op Attention)

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

#### Team Attention Block (Mch Attention)

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

---

### 3.3 `model/sub_layers.py` — 기반 레이어

**담당**: Actor/Critic에서 사용하는 MLP 구현

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

---

### 3.4 `ppo_utils.py` — PPO 학습 유틸리티

**담당**: PPO-Clip 알고리즘에 필요한 메모리 관리와 GAE 계산

```
eval_actions(pi, actions)
  - 주어진 행동에 대한 log probability와 entropy 계산
  - 입력: pi [B, N*T] (softmax된 정책), actions [B] (행동 인덱스)
  - 출력: (log_probs [B], entropy scalar)

PPOMemory(gamma, gae_lambda)
  - 에피소드 롤아웃 동안의 상태/행동/보상을 저장
  - push_state(EnvState): 매 step 상태 텐서 8개 저장
  - push_transition(action, log_prob, val, reward, done): 전이 데이터 저장
  - transpose_data(): [T, B] → [B*T, ...] 으로 mini-batch용 평탄화
  - get_gae_advantages(): GAE(γ, λ) 기반 advantage + value target 계산
  - clear_memory(): 에피소드 종료 후 메모리 초기화
```

---

## 4. DANIEL 모델 전체 아키텍처

```
입력 (환경 → EnvState 텐서)
│
├─ fea_act    [B, N, 10]   Activity feature (10차원)
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
│ [B,N*T,4d+8]│     │  [B, 2d]   │
│      ↓      │     │      ↓     │
│  [B, N*T]   │     │   [B, 1]   │
│  (raw logit)│     │  (가치 v)  │
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

---

## 5. RCMPSP 환경과의 상호작용

### 5.1 DANIEL 전용 상태: `EnvState` (`scheduling_env.py`)

환경이 `state_mode='daniel'`로 초기화되면 `_get_state()` 호출 시
`_get_state_daniel()`을 실행하여 `EnvState` 데이터클래스를 반환합니다.

```python
@dataclass
class EnvState:
    fea_act_tensor:         Tensor  # (B, N, 10)  — activity features
    act_mask_tensor:        Tensor  # (B, N, 3)   — activity attention mask
    fea_team_tensor:        Tensor  # (B, T, 8)   — team features
    team_mask_tensor:       Tensor  # (B, T, T)   — team attention mask
    dynamic_pair_mask_tensor: Tensor  # (B, N, T) — 불가능 pair 마스크
    comp_idx_tensor:        Tensor  # (B, T, T, N) — team 경쟁 인덱스
    candidate_tensor:       Tensor  # (B, N)       — activity identity (0..N-1)
    fea_pairs_tensor:       Tensor  # (B, N, T, 8) — pair feature
```

### 5.2 Activity Feature (`fea_act`) — 10차원

```
인덱스  이름                    설명
  0    started                 시작 여부 (0 or 1)
  1    ended                   완료 여부 (0 or 1)
  2    norm_duration           duration / max_duration
  3    norm_remaining          remaining_time / max_duration
  4    pred_completion_ratio   선행자 완료 비율 (DAG 반영: pred_done / total_pred)  ★ RCMPSP 핵심
  5    ready                   현재 실행 가능 여부 (0 or 1)
  6    norm_start_time         시작 시간 / max_time (미시작 시 0)
  7    proj_rem_act_ratio      프로젝트 내 남은 activity 비율  ★ Project 정보
  8    proj_rem_work_ratio     프로젝트 내 남은 작업량 비율  ★ Project 정보
  9    eligible_ratio          eligible 팀 수 / 전체 팀 수

* 정규화: 유효 activity 기준 mean/std 정규화 (패딩 노드 제외)
* 패딩 노드는 모두 0으로 설정
```

### 5.3 Team Feature (`fea_team`) — 8차원

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

### 5.4 Pair Feature (`fea_pairs`) — 8차원

```
(activity, team) 조합별 피처

인덱스  이름                 설명
  0    norm_pt_global       processing_time / max_duration (전역 정규화)
  1    norm_pt_per_act      processing_time / 해당 activity 최대 PT
  2    norm_pt_per_team     processing_time / 해당 team 최대 PT
  3    norm_pt_global_max   processing_time / 전체 최대 PT
  4    team_workload_ratio  processing_time / (team_remaining + PT) (팀 부하 비율)
  5    due_date_slack       (due_date - est_completion) / max_time  ★ RCMPSP 핵심
  6    proj_work_ratio      processing_time / 프로젝트 남은 작업량
  7    total_wait_time      (team_wait + release_wait [+ mutex_wait]) / max_time  ★ Wait 반영

* 마스킹된 pair는 0으로 설정
* allow_wait_mutex=True일 때 p7에 mutex_wait 추가
```

### 5.5 `dynamic_pair_mask` — 실행 불가 pair 마스크 (B, N, T)

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

### 5.6 `candidate` 텐서 — Activity Identity Mapping

```python
candidate: (B, N)
# candidate = torch.arange(N).unsqueeze(0).expand(B, -1)
# 전체 activity를 candidate로 사용. 필터링은 dynamic_pair_mask에서 처리.
# 모델의 J 차원이 N으로 자동 적응 (sz_b, M, _, J = comp_idx.size())

env.daniel_candidate  # (B, N) — identity mapping이므로 action 역변환 시 직접 사용
# action_flat → act_idx = action_flat // N_T, team_idx = action_flat % N_T
```

### 5.7 `comp_idx` — Team 간 경쟁 인덱스

```python
comp_idx: (B, T, T, N)
# comp_idx[b, k, q, n] = 1.0  ←  team k와 team q 모두
#                              activity n을 수행할 수 있을 때 (두 팀이 경쟁 관계)
# comp_idx[b, k, q, n] = 0.0  ←  경쟁 없음

# 계산:
avail_t  = avail_pair.permute(0, 2, 1).float()  # (B, T, N)
comp_idx = avail_t.unsqueeze(2) * avail_t.unsqueeze(1)  # (B, T, T, N)
```

### 5.8 DANIEL 전용 step: `step_pair()`

```python
def step_pair(self, activity_ids, team_ids):
    """
    DANIEL 모델 전용 — (activity_id, team_id) 직접 입력
    GAT의 step()은 action_to_pair 인덱스를 요구하지만,
    DANIEL은 project-level action → activity/team으로 직접 변환하므로
    별도의 step_pair()를 사용

    activity_ids: (B,)  tensor
    team_ids:     (B,)  tensor
    """
```

---

## 6. train.py / test.py 와의 상호작용

### 6.1 모델 및 알고리즘 선택

```python
# train.py 상단에서 설정
ALGORITHM_TYPE = 'ppo'     # 'reinforce' or 'ppo'
MODEL_TYPE = 'daniel'      # 'gat' or 'daniel'
# ※ PPO는 반드시 MODEL_TYPE='daniel' 이어야 합니다 (Critic 필요)
```

모델 타입에 따라 아래가 자동으로 분기됩니다:

| 항목 | GAT | DANIEL |
|------|-----|--------|
| `env_params['state_mode']` | `'pyg'` | `'daniel'` |
| `model_params` | embedding_dim, num_head, ... | fea_act_input_dim, num_heads_AAB, ... |
| `_get_state()` 반환 | PyG `Data` 객체 리스트 | `EnvState` 데이터클래스 |
| `env.step(action)` | 사용 | 미사용 |
| `env.step_pair(act, team)` | 미사용 | 사용 |
| Action 공간 설정 | `set_action_space()` 필요 | 불필요 |
| 지원 알고리즘 | REINFORCE only | REINFORCE 또는 PPO |

### 6.2 model_params 구조

```python
# 논문 원본 파라미터 (DANIEL, ~28K params)
model_params = {
    'fea_act_input_dim':  10,    # Activity feature 차원 (환경 고정값)
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
    'fea_act_input_dim': 10,
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

## 7. trainer.py 와의 상호작용

### 7.1 모델 초기화

```python
# trainer.py __init__()
if self.model_type == 'daniel':
    daniel_config = SimpleNamespace(
        fea_j_input_dim  = model_params['fea_act_input_dim'],
        fea_m_input_dim  = model_params['fea_team_input_dim'],
        num_heads_OAB    = model_params['num_heads_AAB'],
        num_heads_MAB    = model_params['num_heads_TAB'],
        layer_fea_output_dim = model_params['layer_fea_output_dim'],
        ...
    )
    self.model = DANIEL(daniel_config).to(device)
```

### 7.2 알고리즘 분기

```python
# trainer.py run()
for epoch in range(start_epoch, end_epoch+1):
    if self.algorithm_type == 'ppo':
        # N_r epoch마다 학습 데이터 리샘플링
        if ppo_problem is None or (epoch - start_epoch) % self.n_resample == 0:
            ppo_problem = generate_scheduling_data_batch(self.env_params)
        train_loss, train_reward = self._train_ppo_one_batch(ppo_problem)
    else:
        train_loss, train_reward = self._train_one_batch()  # REINFORCE
```

### 7.3 REINFORCE 학습 루프

```python
while not done:
    # 1. EnvState 텐서를 모델 device로 이동
    fea_act   = s.fea_act_tensor.to(device)
    fea_team  = s.fea_team_tensor.to(device)
    ...

    # 2. DANIEL forward: π (확률), v (가치)
    pi, v = model(fea_act, act_mask, candidate, fea_team, ...)

    # 3. π에서 action 샘플링
    dist = Categorical(pi)
    action_flat = dist.sample()   # [B] — flat index (0 ~ N*T-1)
    log_prob    = dist.log_prob(action_flat)

    # 4. flat index → (activity, team) 직접 분해
    act_idx  = action_flat // N_T  # 어느 activity인지
    team_idx = action_flat % N_T   # 어느 팀인지

    # 5. 환경 step
    s, obj_value, done = env.step_pair(act_idx, team_idx)
```

> **핵심**: DANIEL의 action space는 `N × T` (activity 수 × 팀 수) 구조입니다.
> 전체 activity 중 `dynamic_pair_mask`로 실행 가능한 `(activity, team)` 조합만 선택 가능합니다.
> candidate가 identity mapping이므로 flat index에서 activity를 직접 분해합니다.

### 7.4 Validation 평가 (`_eval_validation`)

학습 루프와 동일하나 `argmax`로 greedy 선택:

```python
action_flat = torch.argmax(pi, dim=1)   # greedy (sampling 대신)
```

---

## 8. 학습 알고리즘

### 8.1 REINFORCE (POMO Baseline)

DANIEL의 Critic(`v`)을 **사용하지 않는** 단순 정책 경사 알고리즘입니다.

```
Episode 종료 시:
  reward = -objective_value         (최소화 → 최대화 변환)

Baseline 옵션 (trainer_params['baseline_type']):
  'pomo': 동일 인스턴스의 POMO 롤아웃 평균 (기본값)
  'batch': 미니배치 전체 평균
  'none': baseline 없음

Loss:
  advantage = reward - baseline
  loss      = -advantage * Σ log π(a_t | s_t)
```

### 8.2 PPO-Clip (DANIEL 전용)

DANIEL 원본 논문의 학습 알고리즘입니다.
Critic(`v`)을 **활용하여** value function 학습과 GAE advantage를 사용합니다.

#### PPO 하이퍼파라미터

```python
# train.py에서 설정
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
| **GAT 지원** | O | X (DANIEL 전용) |
| **학습 데이터** | 매 epoch 새로 생성 | N_r epoch마다 리샘플링 |

---

## 9. DANIEL vs GAT 비교

| 항목 | GAT (gnn_model.py) | DANIEL (model/main_model.py) |
|------|-------------------|------------------------------|
| **상태 표현** | 이종 그래프 (PyG) | 분리된 텐서 (Activity, Team) |
| **인코더** | Graph Attention Network | Dual Attention Network |
| **Action Space** | `(activity, team)` 직접 열거 | `N × T` activity 단위 |
| **가치 함수** | 없음 (REINFORCE only) | 내장 Critic (PPO 가능) |
| **pair feature** | 없음 | (B, N, T, 8) pair feature 활용 |
| **step 메서드** | `env.step(action)` | `env.step_pair(act, team)` |
| **state 생성** | `_get_state_pyg()` | `_get_state_daniel()` |
| **스케일** | Activity 수 증가 시 action space 가변 | N×T 고정 (스케일에 유리) |
| **Wait 옵션** | 없음 | allow_wait_release, allow_wait_mutex |
| **학습 알고리즘** | REINFORCE only | REINFORCE 또는 PPO |

---

## 10. MDP 관점에서의 DANIEL 역할

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

## 11. 확장 시 유의사항

1. **`fea_act_input_dim`**: `_get_state_daniel()`에서 생성하는 feature 차원(10)과
   `train.py`의 `model_params['fea_act_input_dim']`(10)을 항상 일치시켜야 합니다.

2. **`fea_team_input_dim`**: 마찬가지로 환경 출력(8)과 `model_params`(8) 일치 필요.

3. **`layer_fea_output_dim`**: 마지막 레이어의 출력 차원이
   Actor 입력 차원(`4 * output_dim[-1] + 8`)과 Critic 입력 차원(`2 * output_dim[-1]`)을
   결정합니다. 변경 시 `pair_input_dim=8`은 고정임을 유의하세요.

4. **PPO 활용 시**: `ALGORITHM_TYPE='ppo'`로 설정하면 Critic의 value loss가 자동으로 포함됩니다.
   REINFORCE에서는 `v`를 손실에 포함하지 않습니다.

5. **체크포인트 호환성**: GAT와 DANIEL의 모델 구조가 완전히 다르므로,
   `MODEL_TYPE`을 학습 때와 테스트 때 반드시 동일하게 설정해야 합니다.
   마찬가지로 `ALGORITHM_TYPE`도 체크포인트와 일치해야 합니다 (PPO 체크포인트에는 `model_old` 포함).

6. **Mutex/Release time 피처 추가**: 현재 mutex와 release time은 마스킹과 피처(pair feature p5, p7)로
   반영됩니다. 추가적인 명시적 피처가 필요하면 `fea_act_input_dim`을 늘려야 합니다.
