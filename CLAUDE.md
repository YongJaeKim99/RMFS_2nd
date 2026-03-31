# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

**Python 인터프리터**: 반드시 conda 환경을 사용해야 합니다.
```bash
# 올바른 Python 경로
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe"

# 학습 실행
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" rmfs_train.py

# 빠른 임포트 확인
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" -c "from rmfs_env_batch import RMFSBatchEnv; print('OK')"
```

> 주의: `python` 또는 `python.exe` 단독 명령은 Windows Store 더미 Python을 가리켜 exit code 49로 실패합니다.

**주요 패키지**: torch 2.5.1, wandb, numpy

---

## 프로젝트 개요

### RMFS (Robotic Mobile Fulfillment System)

창고 내 로봇이 Pod를 Workstation에 운반한 뒤, **반납할 Storage 위치를 선택**하는 문제입니다. Makespan을 최소화합니다.

핵심 요소:
- **Pod**: 작업장(WS)에서 처리 후 Storage에 반납
- **Robot**: Pod를 운반하는 로봇 (N_R대)
- **Workstation**: Pod를 처리하는 작업장 (N_W개)
- **Storage**: Pod 보관 위치 (N_S개)
- **Action**: 4가지 Storage 선택 규칙 (Stay, Nearest, Target, Sum)

---

## 코드 아키텍처

### 실행 진입점

| 진입점 | 설명 |
|--------|------|
| `rmfs_train.py` | MLP Actor-Critic + RMFSBatchEnv 학습 |

### 알고리즘 선택

```python
ALGORITHM_TYPE = 'reinforce'  # REINFORCE + batch baseline
ALGORITHM_TYPE = 'ppo'        # PPO-Clip + GAE
```

### 핵심 파일

| 파일 | 역할 |
|------|------|
| `RMFS_ENV.py` | 원본 RMFS 환경 (단일 인스턴스, numpy 기반) |
| `rmfs_env_batch.py` | B개의 `RMFS_Environment`를 감싸는 배치 래퍼. `reset()`, `step()`, `get_makespan()` |
| `rmfs_model.py` | MLP Actor-Critic (12 features → 4 actions). `model/sub_layers.py`의 Actor/Critic 재사용 |
| `rmfs_trainer.py` | REINFORCE/PPO 학습 루프 |
| `rmfs_ppo_utils.py` | 단순화된 PPO Memory + GAE (상태 텐서 1개) |
| `rmfs_data_generator.py` | seed 기반 배치 문제 생성 |
| `rmfs_train.py` | RMFS 학습 진입점 + 하이퍼파라미터 설정 |
| `model/sub_layers.py` | Actor, Critic, MLP 클래스 |

### 환경-모델 데이터 흐름

```
rmfs_train.py (파라미터 설정)
    └─ RMFS_Trainer.__init__()
         ├─ RMFSBatchEnv(env_params)           # 배치 환경 생성 (B개 RMFS_Environment)
         ├─ MLPActorCritic(config)             # MLP 모델 생성
         └─ run()
              └─ REINFORCE: _train_one_batch() → _train_one_minibatch()
                 PPO:        _train_ppo_one_batch()
                   ├─ env.reset(problem)        # seed로 B개 인스턴스 초기화
                   ├─ state.features            # (B, 12) 텐서
                   ├─ model(features)           # π (B, 4), v (B, 1)
                   ├─ Categorical(π).sample()   # 행동 샘플링
                   ├─ env.step(actions)         # (state, rewards, all_done)
                   └─ REINFORCE: loss = -advantage * log_prob
                      PPO: RMFSPPOMemory → GAE → K-epoch clip update
```

### 상태 표현

- `RMFSState` 데이터클래스: `features (B, 12)` 단일 텐서
- 12개 features:
  - `[0]` Target WS까지 거리 (정규화)
  - `[1]` 가용 로봇 비율
  - `[2]` 최소 차이 (정규화)
  - `[3-5]` Nearest rule: storage 시간, 다음 WS 시간, makespan 개선 비율
  - `[6-8]` Target rule: storage 시간, 다음 WS 시간, makespan 개선 비율
  - `[9-11]` Sum rule: storage 시간, 다음 WS 시간, makespan 개선 비율
- Action space: 4개 discrete (0=WS 유지, 1=Nearest, 2=Target, 3=Sum)

### 체크포인트 구조

**경로**: `checkpoints/{timestamp}_RMFS_MLP_{ALG}/`

- `epoch{N}.pt`: 5 epoch마다 저장
- `best_model.pt`: validation 기준 최적 모델
- `epoch0.pt`: 학습 전 초기 상태

체크포인트 dict: `model_state_dict`, `optimizer_state_dict`, `env_params`, `model_params`, `trainer_params`, `epoch`, `train_score`, `val_score`, `algorithm_type`

PPO 체크포인트는 `model_old_state_dict`도 포함합니다.

### Validation 데이터셋

- `data/rmfs_val/val_batch.pickle` (고정 시드 2025, seed 목록 저장)

---

## 문서

| 파일 | 내용 |
|------|------|
| `PROJECT_STRUCTURE.md` | RMFS 프로젝트 구조 상세 |

---

## 중요 주의사항

- **`RMFS_ENV.py`**: 원본 단일 인스턴스 환경. `rmfs_env_batch.py`가 B개를 감싸서 배치 인터페이스 제공.
- **배치 처리 방식**: `pod_assign()` 내부 로직이 조건 분기/가변 탐색으로 완전 벡터화가 비실용적. per-instance for-loop + 결과 텐서 묶기 방식 사용.
- **random seed 격리**: `RMFS_ENV.reset()`이 `random.seed()`를 전역 호출하므로 배치 환경에서 순서 의존성 있음.
- **Infeasible action**: action 2(Target), 3(Sum)은 Target_Storage가 -1일 때 infeasible. reward=-100000, done=True. RL이 학습으로 회피해야 함.
- **`state_dim=12`**: `RMFS_ENV.py`의 `pod_assign()`이 생성하는 feature 수와 `rmfs_train.py`의 `model_params['state_dim']`을 일치시켜야 합니다.
- **`force_mask_stay`**: `env_params['force_mask_stay']=True`이면 action 0(Stay/WS 유지)을 항상 마스킹하여 로봇이 반드시 Storage에 Pod를 반납하도록 강제. Deadlock 방지용 옵션.
