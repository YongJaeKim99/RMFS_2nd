# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

**Python 인터프리터**: 반드시 conda 환경을 사용해야 합니다.
```bash
# 올바른 Python 경로
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe"

# 학습 실행
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" train.py

# 테스트 실행
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" test.py

# 테스트 데이터 생성 (data/test/ 폴더)
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" data_generation.py

# 빠른 sanity check (data_generator + SchedulingEnv 검증)
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" _validate_gen.py

# 빠른 임포트 확인
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" -c "from scheduling_env import SchedulingEnv; print('OK')"
```

> 주의: `python` 또는 `python.exe` 단독 명령은 Windows Store 더미 Python을 가리켜 exit code 49로 실패합니다.

**주요 패키지**: torch 2.5.1, wandb, openpyxl, numpy, pandas

---

## 프로젝트 개요

**RCMPSP (Resource-Constrained Multi-Project Scheduling Problem)**을 RL로 푸는 연구 코드입니다.
여러 프로젝트의 Activity를 제한된 Team(리소스)에 할당하여 Tardiness 또는 Makespan을 최소화합니다.

핵심 제약:
- **Precedence**: Activity 간 선행 관계
- **Mutex**: 동시에 실행 불가한 Activity 쌍
- **Eligible**: Activity마다 수행 가능한 Team 목록
- **Release time / Due date**: 프로젝트별 시작 가능 시간과 납기

---

## 코드 아키텍처

### 실행 진입점

- **`train.py`**: 학습 파라미터 설정 후 `Scheduling_Trainer.run()` 호출. 모든 하이퍼파라미터는 이 파일 상단의 변수로 제어합니다.
- **`test.py`**: 저장된 체크포인트로 RL(greedy) 및 GA 성능 비교. 결과를 `results/` 폴더에 Excel로 저장합니다.
- **`data_generation.py`**: `data/test/` 폴더에 테스트용 pickle 파일을 생성합니다.

### 알고리즘 선택 (train.py)

```python
ALGORITHM_TYPE = 'reinforce'  # REINFORCE + POMO baseline
ALGORITHM_TYPE = 'ppo'        # PPO-Clip + GAE
```

### 모델

DANIEL (Dual Attention Network) 모델만 사용합니다 (`model/main_model.py`).
환경의 `state_mode`는 항상 `'daniel'`입니다.

### Reward 방식

```python
REWARD_TYPE = 'sparse'    # 에피소드 끝에만 -objective 반환 (기본값)
REWARD_TYPE = 'stepwise'  # 매 step마다 dense reward (tardiness 전용)
                          # r_t = est_tardiness(s_t) - est_tardiness(s_{t+1})
```

### 핵심 파일 역할

| 파일 | 역할 |
|------|------|
| `scheduling_env.py` | RCMPSP 환경 (DES 기반). `_reset()`, `step()`, `step_pair()`, `_get_state()` |
| `model/main_model.py` | DANIEL 정책+가치 네트워크. 텐서 묶음 입력 → π, v 출력 |
| `trainer.py` | REINFORCE/PPO 학습 루프, validation, 체크포인트 저장/로드, WandB 로깅 |
| `ppo_utils.py` | PPO 롤아웃 버퍼(`PPOMemory`) 및 GAE advantage 계산. `transpose_data()` → `get_gae_advantages()` |
| `data_generator.py` | 문제 인스턴스 배치 생성 (`generate_scheduling_data_batch`) |
| `data_generation.py` | `data/test/` 폴더에 테스트 pickle 파일을 생성하는 스크립트 |
| `GA.py` | 비교용 유전 알고리즘 (Random Key 방식) |
| `samsung_MIP.py` | Gurobi MIP + OR-Tools CP-SAT 기반 정확해 솔버 (비교 베이스라인) |
| `gantt_chart.py` | 결과 시각화 |
| `_validate_gen.py` | `data_generator` 및 `SchedulingEnv` 빠른 sanity check 스크립트 |

### 환경-모델 데이터 흐름

```
train.py (파라미터 설정)
    └─ Scheduling_Trainer.__init__()
         ├─ SchedulingEnv(env_params)          # 환경 생성
         ├─ DANIEL(config)                     # 모델 생성
         └─ run()
              └─ REINFORCE: _train_one_batch() → train_one_minibatch()
                 PPO:        _train_ppo_one_batch()
                   ├─ env._reset(problem)      # 새 문제 로드
                   ├─ env._get_state()         # 상태 추출 → EnvState 데이터클래스
                   ├─ model.forward(state)     # 행동 확률 계산
                   ├─ Categorical(π).sample()  # 행동 샘플링
                   ├─ env.step_pair(a, t)      # (activity, team) 직접
                   └─ REINFORCE: loss = -advantage * log_prob
                      PPO: PPOMemory → GAE → K-epoch clip update
```

### 상태 표현

- `EnvState` 데이터클래스: 10개 텐서 (`fea_act [B,N,12]`, `fea_team [B,T,8]`, `candidate [B,N]`, `comp_idx [B,T,T,N]`, `dynamic_pair_mask [B,N,T]`, `fea_pairs [B,N,T,8]`, `pred_idx`, `succ_idx`, ...)
- Action space: `N × T` (현재 후보 activity × 팀)
- flat action index → `act_idx = action // N_T`, `team_idx = action % N_T`

### 체크포인트 구조

학습 시 `checkpoints/{timestamp}_{objective}_{MODEL}_{ALG}/` 폴더 자동 생성.
(`ALG`는 `REINFORCE` 또는 `PPO`)
- `epoch{N}.pt`: 5 epoch마다 저장
- `best_model.pt`: validation 기준 최적 모델
- `epoch0.pt`: 학습 전 초기 상태

체크포인트 dict: `model_state_dict`, `optimizer_state_dict`, `env_params`, `model_params`, `trainer_params`, `epoch`, `train_score`, `val_score`, `algorithm_type`

PPO 체크포인트는 `model_old_state_dict`도 포함합니다.

**test.py에서 로드 시 `ALGORITHM_TYPE`을 학습 때와 반드시 동일하게 설정해야 합니다.**

### Validation 데이터셋

`USE_VALIDATION=True`로 첫 학습 실행 시 `data/val/val_batch.pickle` 단일 배치 파일을 자동 생성 (고정 시드 2025).
이후 학습에서는 이 파일을 재사용합니다.

---

## 문서

| 파일 | 내용 |
|------|------|
| `MDP_STRUCTURE.md` | MDP 전체 구조 (State/Action/Reward/Transition), DANIEL 아키텍처, 학습 알고리즘 |
| `DANIEL_STRUCTURE.md` | DANIEL 모델 상세 (파일별 역할, 피처 정의, 환경 상호작용) |
| `GA_STRUCTURE.md` | 유전 알고리즘 구조 |

---

## 중요 주의사항

- **`FJSP-DRL-main/`**: 원본 FJSP 프로젝트 참고용 폴더. 이 프로젝트에서는 `model/` 폴더에 이식된 버전만 사용합니다.
- **`model/main_model.py`의 파라미터 이름**: 원본 FJSP의 `fea_j_input_dim`, `num_heads_OAB` 등을 사용. `train.py`에서는 RCMPSP 이름(`fea_act_input_dim`, `num_heads_AAB`)을 사용하며, `trainer.py`가 `SimpleNamespace`로 매핑합니다.
- **`fea_act_input_dim=12`**: `scheduling_env.py`의 `_get_state_daniel()`이 생성하는 차원과 항상 일치해야 합니다. 피처 수를 바꾸면 두 곳 모두 수정해야 합니다.
- **DANIEL의 Critic**: 현재 REINFORCE 손실에는 포함되지 않습니다. PPO에서는 `vloss_coef * F.mse_loss(v, v_target)`로 학습됩니다.
- **`samsung_MIP.py`**: Gurobi 라이센스가 필요합니다. 라이센스가 없으면 OR-Tools CP-SAT 경로를 사용합니다.
- **Wait / Dominance 옵션** (`ALLOW_WAIT_RELEASE`, `ALLOW_WAIT_MUTEX`, `DOMINANCE_RULE`): action space를 확장/축소하는 실험적 옵션입니다. 기본값은 모두 `False`입니다.
