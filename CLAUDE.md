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

# 테스트 실행 (체크포인트 경로를 rmfs_test.py 내부에서 지정)
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" rmfs_test.py

# MIP 솔루션 재현
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" rmfs_mip_replay.py

# 빠른 임포트 확인
KMP_DUPLICATE_LIB_OK=TRUE "C:\Users\YongJae\anaconda3\envs\RSS_1st\python.exe" -c "from rmfs_env_batch import RMFSBatchEnv; print('OK')"
```

> 주의: `python` 또는 `python.exe` 단독 명령은 Windows Store 더미 Python을 가리켜 exit code 49로 실패합니다.

**주요 패키지**: torch 2.5.1, wandb, numpy, gurobipy (MILP solver)

---

## 프로젝트 개요

### RMFS (Robotic Mobile Fulfillment System)

창고 내 로봇이 Pod를 Workstation에 운반하여 처리한 뒤, **반납할 Storage 위치를 선택**하는 문제입니다. Makespan을 최소화합니다.

- **Pod**: 작업장(WS)에서 처리 후 Storage에 반납
- **Robot**: Pod를 운반하는 이동 로봇 (N_R대)
- **Workstation (WS)**: Pod를 처리하는 작업장 (N_W개)
- **Storage**: Pod 보관 위치 (N_S개, 기본: 3×3×4×2 = 72개)
- **Action**: N_S+1개 (0=Stay, 1~N_S=specific storage location)

---

## 코드 아키텍처

### 실행 진입점

| 진입점 | 설명 |
|--------|------|
| `rmfs_train.py` | GATv2 Actor-Critic 학습 (REINFORCE / PPO) |
| `rmfs_test.py` | 체크포인트 로드 → greedy/sampling rollout + 선택적 Gurobi MIP 비교 → Excel 저장 |
| `rmfs_mip_replay.py` | Gurobi MIP 솔루션을 RMFS 환경에서 step-by-step 재현 및 검증 |

### 알고리즘 선택

`rmfs_train.py` 내에서 설정:
```python
ALGORITHM_TYPE = 'reinforce'  # REINFORCE + batch baseline
ALGORITHM_TYPE = 'ppo'        # PPO-Clip + GAE (기본)
```

### 핵심 파일

| 파일 | 역할 |
|------|------|
| `RMFS_ENV.py` | 원본 RMFS 환경 (단일 인스턴스, numpy/list 기반 DES). `reset()` → `pod_assign()` → graph state dict 반환 |
| `rmfs_env_batch.py` | B개의 `RMFS_Environment`를 감싸는 배치 래퍼. `RMFSState` 데이터클래스로 graph state 제공 |
| `rmfs_model.py` | `GATActorCritic`: GATv2 기반 Actor-Critic. graph → `pi (B, N_S+1)`, `v (B, 1)` |
| `model/gat_layers.py` | `GATv2Layer`: dense adjacency 기반 GATv2 (edge features 포함, Brody et al. ICLR 2022) |
| `model/sub_layers.py` | `Actor`, `Critic`, `MLP` 클래스 (tanh activation) |
| `rmfs_trainer.py` | REINFORCE/PPO 학습 루프 + validation + checkpoint 관리 |
| `rmfs_ppo_utils.py` | `RMFSPPOMemory` (graph state 저장) + per-instance GAE 계산 |
| `rmfs_data_generator.py` | seed 기반 배치 문제 생성 |
| `MILP.py` | `solve_rmfs_milp()`: Gurobi MILP 정식화 (31개 제약 그룹) |
| `rmfs_milp_utils.py` | `convert_env_to_milp_data()`: RMFS 환경 → 1-indexed MILP 입력 변환 |

### 환경-모델 데이터 흐름

```
rmfs_train.py (파라미터 설정)
    └─ RMFS_Trainer.__init__()
         ├─ GATActorCritic(config)              # GATv2 모델 생성
         ├─ RMFSBatchEnv(env_params)            # 배치 환경 (B개 인스턴스)
         ├─ _build_adjacency()                  # (1, V, V) static bool 인접행렬
         └─ run()
              └─ PPO: _train_ppo_one_batch()
                   ├─ env.reset(problem)         # seed로 초기화 → RMFSState
                   ├─ model(state, adj)          # π (B, N_S+1), v (B, 1)
                   ├─ Categorical(π).sample()    # 행동 샘플링
                   ├─ env.step(actions)          # → (state, rewards, all_done)
                   └─ RMFSPPOMemory → GAE → K-epoch clip update
                 REINFORCE: _train_one_batch()
                   └─ loss = -advantage * log_prob_sum
```

### MDP 구조

**State (Graph State)**: `RMFSState` 데이터클래스
- `storage_features` (B, N_S, 4): normalized x, y, occupied flag, idle time
- `ws_features` (B, N_W, 4): normalized x, y, busy flag, idle time
- `edge_feat` (B, V, V, 9): type flags (3), distance, pod-ws relation, visit diff, shared pod count, proximity sum, curpod needs
- `curws_idx` (B,): 현재 WS 인덱스
- `action_mask` (B, N_S+1): 유효 action 마스크 (bool)

**Action**: 0 = Stay at WS, 1~N_S = specific storage location index

**Reward**: `Pre_Makespan - Makespan` (stepwise, dense) 또는 `-Makespan` (sparse)

**Done**: `Total_PodTask` steps 완료 또는 infeasible action (reward=-100000)

### GATv2 Actor-Critic 모델 구조

```
Input: RMFSState + adj (1, V, V)
  ├─ storage_proj: Linear(4 → d) + type_embed[0]  →  h_s (B, N_S, d)
  ├─ ws_proj:      Linear(4 → d) + type_embed[1]  →  h_w (B, N_W, d)
  ├─ concat → h (B, V, d)  [V = N_S + N_W]
  │
  ├─ GATv2Layer × n_gat_layers (residual + LayerNorm)
  │     └─ h (B,V,d) + adj (B,V,V) + edge_feat (B,V,V,9) → h_new (B,V,d)
  │
  ├─ Split → h_storage (B, N_S, d), h_ws (B, N_W, d)
  ├─ Pool  → h_global_s (B, d), h_global_w (B, d), h_curws (B, d)
  │
  ├─ Actor: [h_storage, h_curws, h_global_s, h_global_w, edge_curws_to_s]
  │         → MLP(4d+9 → 1) → storage_scores (B, N_S)
  │         Stay head: [h_curws, h_global_s, h_global_w] → MLP(3d → 1)
  │         → concat → logits (B, N_S+1) → mask → softmax → π
  │
  └─ Critic: [h_global_s, h_global_w] → MLP(2d → 1) → v (B, 1)
```

### Static Adjacency

`_build_adjacency()`가 한 번 생성, 모든 배치/step에 재사용:
- WS ↔ Storage (모든 쌍, 양방향)
- WS ↔ WS (자기 제외, 양방향)
- 모든 노드 self-loop

### 체크포인트

**경로**: `checkpoints/{timestamp}_RMFS_GAT_{ALG}/`

- `epoch0.pt`: 학습 전 초기 상태
- `epoch{N}.pt`: 5 epoch마다 저장
- `best_model.pt`: validation 기준 최적 모델

체크포인트 dict keys: `model_state_dict`, `optimizer_state_dict`, `env_params`, `model_params`, `trainer_params`, `epoch`, `train_score`, `val_score`, `algorithm_type`. PPO는 `model_old_state_dict` 추가.

### 테스트/평가 파이프라인

`rmfs_test.py`:
1. `checkpoints/{FOLDER}/best_model.pt` 로드 → `GATActorCritic` 재구성
2. Greedy rollout (argmax) + stochastic sampling (K=64, best 선택)
3. 선택적 Gurobi MIP 비교
4. `results/{timestamp}/` 에 Excel 저장

`rmfs_mip_replay.py`: MIP 솔루션을 RMFS 환경에서 step-by-step 실행하여 정합성 검증

### Validation 데이터

- `data/rmfs_val/val_batch.pickle` (고정 시드 2025, 20개 인스턴스)
- `data/rmfs_test/test_batch.pickle` (테스트용, `rmfs_data_generation.py`로 생성)

---

## 중요 주의사항

- **배치 처리 방식**: `pod_assign()` 내부 로직이 조건 분기/가변 탐색으로 완전 벡터화 비실용적. per-instance for-loop + 결과 텐서 묶기 방식 사용.
- **random seed 격리**: `RMFS_ENV.reset()`이 `random.seed()`를 전역 호출하므로 배치 환경에서 순서 의존성 있음.
- **Infeasible action**: 선택한 storage가 `Pod_departure_time + TT_WS[ws][storage]` 시점에 사용 불가하면 infeasible. reward=-100000, done=True.
- **`force_mask_stay`**: `True`이면 action 0(Stay)을 항상 마스킹 → 로봇이 반드시 Storage에 반납. Deadlock 방지용.
- **NaN safety**: 모든 action이 masked되면 softmax → NaN 발생 가능. `GATActorCritic.forward()`에서 uniform 분포로 fallback.
- **Device**: RMFS 환경이 numpy 기반이므로 CPU 권장 (`DEVICE_MODE = 'cpu'`).
- **Wandb**: `rmfs_trainer.py`에 wandb 로그인 키가 하드코딩되어 있음. 학습 시 train/val makespan, loss 등 기록.
- **`model_refer/`**: 다른 스케줄링 프로젝트(RCMPSP/FJSP)의 참조 모델. RMFS 학습에는 사용되지 않음.
- **`cp_replay.py`, `test.py`**: 다른 프로젝트 파일. 이 레포에 없는 모듈을 import하므로 실행 불가.

---

## 문서

| 파일 | 내용 |
|------|------|
| `PROJECT_STRUCTURE.md` | 프로젝트 구조 상세 (주의: MLP 기반 구버전 설명이 일부 남아있음) |