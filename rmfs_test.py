"""
RMFS 테스트 스크립트.

MIP (Gurobi) 및 RL (Greedy + Sampling) 테스트를 수행하고 결과를 Excel로 저장한다.
"""
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import io
import torch
from torch.distributions import Categorical
from rmfs_ppo_utils import sample_continuous_action
import numpy as np
import pandas as pd
from datetime import datetime
import time
import pickle
import random
import copy
import math

if __name__ == "__main__":
    # =================================================================
    # 프로젝트 루트 경로
    # =================================================================
    project_root = Path(__file__).parent

    # =================================================================
    # 테스트 파라미터 설정
    # =================================================================

    # -----------------------------
    # 1) 테스트할 알고리즘
    # -----------------------------
    test_algorithms = ["RL"]  # ["MIP"], ["RL"], ["MIP", "RL"]

    # -----------------------------
    # 2) RL 체크포인트 설정
    # -----------------------------
    CHECKPOINT_FOLDER = "YYYYMMDD_RMFS_GAT_PPO"
    CHECKPOINT_FILE = "best_model.pt"

    # -----------------------------
    # 3) MIP (Gurobi) 설정
    # -----------------------------
    MIP_TIME_LIMIT = 300      # 인스턴스당 시간 제한 (초)
    MIP_GAP = 0.05            # MIP Gap 허용치

    # -----------------------------
    # 4) RL Sampling 설정
    # -----------------------------
    RL_SAMPLING_K = 64        # Sampling rollout 총 횟수 (0이면 greedy만)
    RL_SAMPLING_BATCH = 4     # 한 번에 병렬 처리할 sampling 수

    # -----------------------------
    # 5) 인스턴스 범위
    # -----------------------------
    INSTANCE_START = 0
    INSTANCE_END = 3          # 미포함 (None이면 끝까지)

    # -----------------------------
    # 6) 데이터 경로
    # -----------------------------
    TEST_DATA_DIR = "data/rmfs_test"

    # -----------------------------
    # 7) 기타
    # -----------------------------
    DEVICE_MODE = 'cpu'       # 'cpu' or 'gpu'
    SEED = 0
    DEBUG_ENV = False

    # =================================================================
    # 디바이스 설정
    # =================================================================
    cuda_available = torch.cuda.is_available()

    if DEVICE_MODE != 'cpu' and not cuda_available:
        print(f"GPU가 사용 불가능합니다. CPU 모드로 전환합니다.")
        DEVICE_MODE = 'cpu'

    if DEVICE_MODE == 'cpu':
        device = torch.device('cpu')
        device_desc = "CPU"
    else:
        device = torch.device('cuda')
        device_desc = f"GPU ({torch.cuda.get_device_name(0)})"

    print(f"Device: {device_desc}")

    # 시드 고정
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if cuda_available:
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed fixed to: {SEED}")

    # =================================================================
    # 테스트 데이터 로드
    # =================================================================
    data_base_dir = project_root / TEST_DATA_DIR

    if not data_base_dir.exists():
        print(f"테스트 데이터 폴더가 존재하지 않습니다: {data_base_dir}")
        print(f"rmfs_data_generation.py를 먼저 실행해주세요.")
        sys.exit(1)

    pickle_files = sorted(data_base_dir.glob("*.pickle"))
    if not pickle_files:
        print(f"{data_base_dir}에 .pickle 파일이 없습니다.")
        sys.exit(1)

    print(f"\n테스트 데이터 폴더: {data_base_dir}")
    print(f"  발견된 pickle 파일: {len(pickle_files)}개")
    for pf in pickle_files:
        print(f"    - {pf.name}")

    # =================================================================
    # 설정 출력
    # =================================================================
    print(f"\n{'=' * 60}")
    print("RMFS 테스트 설정")
    print(f"{'=' * 60}")
    print(f"  DEVICE: {device_desc}")
    print(f"  ALGORITHMS: {', '.join(test_algorithms)}")
    _inst_range = (f"[{INSTANCE_START}, "
                   f"{INSTANCE_END if INSTANCE_END is not None else 'end'})")
    print(f"  INSTANCE RANGE: {_inst_range}")
    if "MIP" in test_algorithms:
        print(f"  MIP Settings:")
        print(f"    - Time Limit: {MIP_TIME_LIMIT}s")
        print(f"    - MIP Gap: {MIP_GAP}")
    if "RL" in test_algorithms:
        print(f"  RL Checkpoint: {CHECKPOINT_FOLDER}/{CHECKPOINT_FILE}")
        if RL_SAMPLING_K > 0:
            print(f"    - Sampling: K={RL_SAMPLING_K}, "
                  f"mini_batch={RL_SAMPLING_BATCH}")
    print(f"{'=' * 60}\n")

    # =================================================================
    # 결과 저장 폴더
    # =================================================================
    results_dir = project_root / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_results_dir = results_dir / timestamp

    if not session_results_dir.exists():
        session_results_dir.mkdir(parents=True)
        print(f"결과 저장 폴더 생성: {session_results_dir}")

    # =================================================================
    # 유틸 함수: Excel 저장
    # =================================================================
    def save_rl_results_to_excel(rl_results, checkpoint_name, pickle_name):
        filename = f'RMFS_RL_{checkpoint_name}_{pickle_name}.xlsx'
        output_path = session_results_dir / filename

        exp_rows = [
            {'Item': 'Algorithm', 'Value': 'RL'},
            {'Item': 'Checkpoint', 'Value': checkpoint_name},
            {'Item': 'Data File', 'Value': pickle_name},
            {'Item': 'Device', 'Value': device_desc},
            {'Item': 'Num Instances', 'Value': len(rl_results)},
        ]
        if SEED is not None:
            exp_rows.append({'Item': 'Random Seed', 'Value': SEED})
        experiment_info_df = pd.DataFrame(exp_rows, columns=['Item', 'Value'])

        if not rl_results:
            print(f"결과 데이터가 없습니다: RL")
            return None

        detailed_data = []
        for r in rl_results:
            detailed_data.append({
                'Algorithm': r['algorithm'],
                'Instance': r['instance'],
                'Makespan': r['makespan'],
                'Runtime': r.get('runtime', 0),
            })

        valid_ms = [r['makespan'] for r in rl_results
                    if r['makespan'] is not None]
        overall_row = {
            'Algorithm': rl_results[0]['algorithm'] if rl_results else 'RL',
            'Instance': 'Average',
            'Makespan': np.mean(valid_ms) if valid_ms else None,
            'Runtime': np.mean([r.get('runtime', 0) for r in rl_results]),
        }

        detailed_df = pd.DataFrame(detailed_data)
        overall_avg_df = pd.DataFrame([overall_row])

        for df in [detailed_df, overall_avg_df]:
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].round(4)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            experiment_info_df.to_excel(writer, sheet_name='Experiment_Info',
                                        index=False)
            detailed_df.to_excel(writer, sheet_name='Detailed_Results',
                                 index=False)
            overall_avg_df.to_excel(writer, sheet_name='Overall_Average',
                                    index=False)

        print(f"결과 저장: {output_path}")
        return output_path

    def save_mip_results_to_excel(mip_results, pickle_name):
        output_path = session_results_dir / f'RMFS_MIP_{pickle_name}.xlsx'

        exp_rows = [
            {'Item': 'Algorithm', 'Value': 'MIP (Gurobi)'},
            {'Item': 'Data File', 'Value': pickle_name},
            {'Item': 'Time Limit', 'Value': MIP_TIME_LIMIT},
            {'Item': 'MIP Gap', 'Value': MIP_GAP},
            {'Item': 'Device', 'Value': 'CPU'},
            {'Item': 'Num Instances', 'Value': len(mip_results)},
        ]
        if SEED is not None:
            exp_rows.append({'Item': 'Random Seed', 'Value': SEED})
        experiment_info_df = pd.DataFrame(exp_rows, columns=['Item', 'Value'])

        if not mip_results:
            print(f"결과 데이터가 없습니다: MIP")
            return None

        detailed_data = []
        for r in mip_results:
            detailed_data.append({
                'Instance': r['instance'],
                'Makespan': r['makespan'],
                'Status': r['status'],
                'Runtime': r['runtime'],
            })
        detailed_df = pd.DataFrame(detailed_data)

        valid_ms = [r['makespan'] for r in mip_results
                    if r['makespan'] is not None]
        overall_row = {
            'Algorithm': 'MIP (Gurobi)',
            'Avg Makespan': np.mean(valid_ms) if valid_ms else None,
            'Solved': len(valid_ms),
            'Total': len(mip_results),
            'Avg Runtime': np.mean([r['runtime'] for r in mip_results]),
        }
        overall_avg_df = pd.DataFrame([overall_row])

        for df in [detailed_df, overall_avg_df]:
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].round(4)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            experiment_info_df.to_excel(writer, sheet_name='Experiment_Info',
                                        index=False)
            detailed_df.to_excel(writer, sheet_name='Detailed_Results',
                                 index=False)
            overall_avg_df.to_excel(writer, sheet_name='Overall_Average',
                                    index=False)

        print(f"결과 저장: {output_path}")
        return output_path

    # =================================================================
    # RL 모델 준비
    # =================================================================
    model = None
    adj = None
    model_params = None
    _rl_env_params = None

    if "RL" in test_algorithms:
        from rmfs_model import GATActorCritic
        from rmfs_env_batch import RMFSBatchEnv, RMFSState
        from types import SimpleNamespace

        ckpt_dir = project_root / "checkpoints" / CHECKPOINT_FOLDER
        ckpt_path = ckpt_dir / CHECKPOINT_FILE

        if not ckpt_path.exists():
            print(f"체크포인트 파일이 존재하지 않습니다: {ckpt_path}")
            test_algorithms = [a for a in test_algorithms if a != "RL"]
        else:
            checkpoint = torch.load(ckpt_path, map_location=device,
                                    weights_only=False)

            if isinstance(checkpoint, dict):
                model_params = checkpoint.get('model_params')
                _ckpt_env_params = checkpoint.get('env_params', {})
            else:
                print("checkpoint 형식을 인식할 수 없습니다.")
                test_algorithms = [a for a in test_algorithms if a != "RL"]

            if model_params is not None:
                # 환경 크기 파라미터 추출
                block_rows = _ckpt_env_params.get('block_rows', 3)
                block_cols = _ckpt_env_params.get('block_cols', 3)
                block_h = _ckpt_env_params.get('block_h', 4)
                block_w = _ckpt_env_params.get('block_w', 2)
                _N_S = block_rows * block_cols * block_h * block_w
                _N_W = _ckpt_env_params.get('N_W', 4)
                _V = _N_S + _N_W

                # action_type: checkpoint에서 로드 (없으면 discrete)
                _action_type = model_params.get(
                    'action_type',
                    checkpoint.get('trainer_params', {}).get('action_type', 'discrete')
                )

                model_config = SimpleNamespace(
                    N_S=_N_S,
                    N_W=_N_W,
                    storage_feat_dim=model_params.get('storage_feat_dim', 4),
                    ws_feat_dim=model_params.get('ws_feat_dim', 4),
                    d_edge=model_params.get('d_edge', 9),
                    d_hidden=model_params['d_hidden'],
                    n_gat_layers=model_params['n_gat_layers'],
                    n_heads=model_params['n_heads'],
                    dropout_prob=model_params.get('dropout_prob', 0.0),
                    num_mlp_layers_actor=model_params['num_mlp_layers_actor'],
                    hidden_dim_actor=model_params['hidden_dim_actor'],
                    num_mlp_layers_critic=model_params['num_mlp_layers_critic'],
                    hidden_dim_critic=model_params['hidden_dim_critic'],
                    action_type=_action_type,
                )
                model = GATActorCritic(model_config).to(device)

                if 'model_state_dict' in checkpoint:
                    load_result = model.load_state_dict(
                        checkpoint['model_state_dict'], strict=False)
                    if load_result.unexpected_keys:
                        print(f"  무시된 키: {load_result.unexpected_keys}")
                    if load_result.missing_keys:
                        print(f"  누락된 키: {load_result.missing_keys}")
                else:
                    model.load_state_dict(checkpoint, strict=False)

                model.eval()
                print(f"RL 모델 로드 완료: {ckpt_path.name}")
                print(f"  N_S={_N_S}, N_W={_N_W}, "
                      f"d_hidden={model_params['d_hidden']}, "
                      f"action_type={_action_type}")

                # Adjacency matrix
                adj_mat = torch.zeros(_V, _V, dtype=torch.bool)
                for w in range(_N_W):
                    wi = _N_S + w
                    adj_mat[wi, :_N_S] = True
                    adj_mat[:_N_S, wi] = True
                for w1 in range(_N_W):
                    for w2 in range(_N_W):
                        if w1 != w2:
                            adj_mat[_N_S + w1, _N_S + w2] = True
                idx = torch.arange(_V)
                adj_mat[idx, idx] = True
                adj = adj_mat.unsqueeze(0).to(device)

                _rl_env_params = _ckpt_env_params

    # =================================================================
    # 유틸 함수: state를 device로 이동
    # =================================================================
    def _state_to_device(state):
        from rmfs_env_batch import RMFSState
        return RMFSState(
            storage_features=state.storage_features.to(device),
            ws_features=state.ws_features.to(device),
            edge_feat=state.edge_feat.to(device),
            curws_idx=state.curws_idx.to(device),
            action_mask=state.action_mask.to(device),
        )

    # =================================================================
    # pickle 파일별 실험 실행
    # =================================================================
    algorithm_summaries = {}
    saved_files = []

    for pickle_path in pickle_files:
        pickle_name = pickle_path.stem
        print(f"\n{'=' * 70}")
        print(f"Processing: {pickle_path.name}")
        print(f"{'=' * 70}")

        with open(pickle_path, 'rb') as f:
            problem = pickle.load(f)

        seeds = problem['seeds']
        N_P = problem['N_P']
        N_R = problem['N_R']
        N_W = problem['N_W']
        Total_PodTask = problem['Total_PodTask']
        saved_env_params = problem.get('env_params', {})

        total_instances = len(seeds)
        inst_start = INSTANCE_START if INSTANCE_START is not None else 0
        inst_end = INSTANCE_END if INSTANCE_END is not None else total_instances
        inst_start = max(0, min(inst_start, total_instances))
        inst_end = max(inst_start, min(inst_end, total_instances))
        num_instances = inst_end - inst_start
        print(f"  전체 인스턴스: {total_instances}, "
              f"실행 범위: [{inst_start}, {inst_end}) ({num_instances}개)")

        # 환경 파라미터 구성
        block_rows = saved_env_params.get('block_rows', 3)
        block_cols = saved_env_params.get('block_cols', 3)
        block_h = saved_env_params.get('block_h', 4)
        block_w = saved_env_params.get('block_w', 2)
        Unit_PT = saved_env_params.get('Unit_PT', 15)
        ST = saved_env_params.get('ST', 5)
        UT = saved_env_params.get('UT', 1)
        Large = saved_env_params.get('Large', True)
        force_mask_stay = saved_env_params.get('force_mask_stay', True)

        # =============================================
        # MIP (Gurobi) 실행
        # =============================================
        if "MIP" in test_algorithms:
            from RMFS_ENV import RMFS_Environment
            from rmfs_milp_utils import convert_env_to_milp_data
            from MILP import solve_rmfs_milp, HAS_GUROBI

            if not HAS_GUROBI:
                print("\n  Gurobi가 설치되지 않았습니다. MIP를 건너뜁니다.")
            else:
                print(f"\n  --- MIP Gurobi ({pickle_name}) ---")
                mip_results = []

                # RMFS 환경 생성 (print 억제)
                _stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    mip_env = RMFS_Environment(
                        block_rows, block_cols, block_h, block_w,
                        Unit_PT, ST, UT, Large,
                        force_mask_stay=force_mask_stay)
                finally:
                    sys.stdout = _stdout

                for i in range(inst_start, inst_end):
                    print(f"\n   Instance {i}/{inst_end - 1}")

                    try:
                        # 환경 reset (print 억제)
                        _stdout = sys.stdout
                        sys.stdout = io.StringIO()
                        try:
                            mip_env.reset(seeds[i], N_P, N_R,
                                          Total_PodTask, N_W)
                        finally:
                            sys.stdout = _stdout

                        # MILP 데이터 변환
                        milp_data = convert_env_to_milp_data(mip_env)
                        print(f"     MILP 데이터: P={len(milp_data['P'])}, "
                              f"K={len(milp_data['K'])}, "
                              f"S={len(milp_data['S'])}, "
                              f"R={len(milp_data['R'])}, "
                              f"N_T={milp_data['N_T']}")

                        # MILP 풀기
                        start_time = time.time()
                        result = solve_rmfs_milp(
                            milp_data,
                            time_limit=MIP_TIME_LIMIT,
                            mip_gap=MIP_GAP,
                            verbose=False)
                        runtime = time.time() - start_time

                        if result is not None:
                            makespan, status, gurobi_runtime = result
                            print(f"     [MIP][Instance {i}] "
                                  f"Makespan: {makespan:.4f}, "
                                  f"Status: {status}, "
                                  f"Runtime: {runtime:.4f}s")

                            mip_results.append({
                                'instance': i,
                                'makespan': makespan,
                                'status': status,
                                'runtime': runtime,
                            })
                        else:
                            print(f"     [MIP][Instance {i}] "
                                  f"No solution found, Runtime: {runtime:.4f}s")
                            mip_results.append({
                                'instance': i,
                                'makespan': None,
                                'status': 'no_solution',
                                'runtime': runtime,
                            })

                    except Exception as e:
                        print(f"     [MIP][Instance {i}] 오류: {e}")
                        import traceback
                        traceback.print_exc()
                        mip_results.append({
                            'instance': i,
                            'makespan': None,
                            'status': 'error',
                            'runtime': 0,
                        })

                if mip_results:
                    valid_ms = [r['makespan'] for r in mip_results
                                if r['makespan'] is not None]
                    if valid_ms:
                        avg_ms = np.mean(valid_ms)
                        solved = len(valid_ms)
                        algorithm_summaries[f'MIP ({pickle_name})'] = avg_ms
                        print(f"  MIP 평균 Makespan: {avg_ms:.4f} "
                              f"({solved}/{num_instances} solved)")

                    output_path = save_mip_results_to_excel(
                        mip_results, pickle_name)
                    if output_path:
                        saved_files.append(output_path)

        # =============================================
        # RL 실행 (배치 추론)
        # =============================================
        if "RL" in test_algorithms and model is not None:
            from rmfs_env_batch import RMFSBatchEnv, RMFSState

            print(f"\n  --- RL ({pickle_name}) ---")

            # RL용 env_params 구성
            rl_env_params = {
                'batch_size': total_instances,
                'block_rows': block_rows,
                'block_cols': block_cols,
                'block_h': block_h,
                'block_w': block_w,
                'Unit_PT': Unit_PT,
                'ST': ST,
                'UT': UT,
                'Large': Large,
                'N_W': N_W,
                'force_mask_stay': force_mask_stay,
            }

            # checkpoint env_params로 보충
            if _rl_env_params:
                for k, v in _rl_env_params.items():
                    if k not in rl_env_params:
                        rl_env_params[k] = v

            checkpoint_name = Path(CHECKPOINT_FILE).stem

            try:
                with torch.no_grad():
                    # ----- Greedy Rollout -----
                    print(f"  Greedy rollout 시작 "
                          f"(전체 {total_instances}개)")
                    start_time = time.time()

                    test_env = RMFSBatchEnv(rl_env_params, device='cpu')
                    state = test_env.reset(problem)
                    all_done = False

                    while not all_done:
                        state_dev = _state_to_device(state)
                        output, v = model(state_dev, adj)
                        if _action_type == 'discrete':
                            action = torch.argmax(output, dim=-1)
                        else:
                            action, _, _ = sample_continuous_action(
                                output, _action_type, greedy=True)
                        state, rewards, all_done = test_env.step(action.cpu())

                    greedy_scores = test_env.get_makespan()
                    greedy_end_time = time.time()
                    greedy_runtime = greedy_end_time - start_time
                    greedy_avg_runtime = greedy_runtime / total_instances

                    # Greedy 결과 저장
                    rl_greedy_label = ('RL_greedy' if RL_SAMPLING_K > 0
                                       else 'RL')
                    rl_results = []
                    for i in range(inst_start, inst_end):
                        score_i = greedy_scores[i].item()
                        rl_results.append({
                            'algorithm': rl_greedy_label,
                            'instance': i,
                            'makespan': score_i,
                            'runtime': greedy_avg_runtime,
                        })
                        print(f"     [{rl_greedy_label}][Instance {i}] "
                              f"Makespan: {score_i:.4f}")

                    print(f"\n     Greedy Total: {greedy_runtime:.4f}s "
                          f"(avg {greedy_avg_runtime:.4f}s/instance)")

                    # ----- Sampling Rollout -----
                    rl_sampling_results = []
                    best_sampling_scores = None

                    if RL_SAMPLING_K > 0:
                        K = RL_SAMPLING_K
                        MB = RL_SAMPLING_BATCH
                        n_mini = math.ceil(K / MB)
                        print(f"\n  Sampling rollout 시작 "
                              f"(K={K}, mini_batch={MB}, {n_mini}회)")

                        sample_idx = 0

                        for mb_idx in range(n_mini):
                            mb_k = min(MB, K - mb_idx * MB)
                            mb_total = total_instances * mb_k
                            print(f"     [Mini batch {mb_idx + 1}/{n_mini}] "
                                  f"{mb_k} samples (batch={mb_total})")

                            # seeds를 mb_k번 반복
                            sampling_seeds = []
                            for s in seeds:
                                for _ in range(mb_k):
                                    sampling_seeds.append(s)

                            sampling_problem = {
                                'seeds': sampling_seeds,
                                'N_P': N_P,
                                'N_R': N_R,
                                'N_W': N_W,
                                'Total_PodTask': Total_PodTask,
                            }

                            sampling_env_params = dict(rl_env_params)
                            sampling_env_params['batch_size'] = mb_total

                            sample_env = RMFSBatchEnv(
                                sampling_env_params, device='cpu')
                            state = sample_env.reset(sampling_problem)
                            all_done = False

                            while not all_done:
                                state_dev = _state_to_device(state)
                                output, v = model(state_dev, adj)
                                if _action_type == 'discrete':
                                    dist = Categorical(output)
                                    action = dist.sample()
                                else:
                                    action, _, _ = sample_continuous_action(
                                        output, _action_type, greedy=False)
                                state, rewards, all_done = sample_env.step(
                                    action.cpu())

                            # (total*mb_k,) → (total, mb_k)
                            mb_scores = sample_env.get_makespan()
                            mb_scores = mb_scores.reshape(
                                total_instances, mb_k)

                            for k_idx in range(mb_k):
                                sample_idx += 1
                                avg_k = mb_scores[
                                    inst_start:inst_end, k_idx
                                ].mean().item()
                                print(f"       [Sample {sample_idx}/{K}] "
                                      f"avg Makespan: {avg_k:.4f}")

                            mb_best = mb_scores.min(dim=1).values
                            if best_sampling_scores is None:
                                best_sampling_scores = mb_best
                            else:
                                best_sampling_scores = torch.min(
                                    best_sampling_scores, mb_best)

                        # Sampling 결과
                        sampling_end_time = time.time()
                        sampling_runtime = sampling_end_time - greedy_end_time
                        avg_sampling_runtime = (sampling_runtime
                                                / total_instances)

                        sampling_avg = best_sampling_scores[
                            inst_start:inst_end].mean().item()
                        greedy_avg = greedy_scores[
                            inst_start:inst_end].mean().item()
                        print(f"\n     Sampling best avg: {sampling_avg:.4f} "
                              f"| Greedy avg: {greedy_avg:.4f}")

                        for i in range(inst_start, inst_end):
                            score_i = best_sampling_scores[i].item()
                            rl_sampling_results.append({
                                'algorithm': f'RL_sampling(K={K})',
                                'instance': i,
                                'makespan': score_i,
                                'runtime': avg_sampling_runtime,
                            })
                            print(f"     [RL_sampling(K={K})][Instance {i}] "
                                  f"Makespan: {score_i:.4f}")

                        print(f"\n     Sampling Total: "
                              f"{sampling_runtime:.4f}s "
                              f"(avg {avg_sampling_runtime:.4f}s/instance)")

            except Exception as e:
                print(f"  RL 추론 중 오류: {e}")
                import traceback
                traceback.print_exc()
                rl_results = []
                rl_sampling_results = []

            # Greedy 결과 저장
            if rl_results:
                valid_ms = [r['makespan'] for r in rl_results
                            if r['makespan'] is not None]
                if valid_ms:
                    avg_ms = np.mean(valid_ms)
                    algorithm_summaries[
                        f'{rl_greedy_label} ({pickle_name})'] = avg_ms
                    print(f"  {rl_greedy_label} 평균 Makespan: {avg_ms:.4f}")

                output_path = save_rl_results_to_excel(
                    rl_results, checkpoint_name, pickle_name)
                if output_path:
                    saved_files.append(output_path)

            # Sampling 결과 저장
            if rl_sampling_results:
                valid_ms = [r['makespan'] for r in rl_sampling_results
                            if r['makespan'] is not None]
                if valid_ms:
                    avg_ms = np.mean(valid_ms)
                    algorithm_summaries[
                        f'RL_sampling(K={RL_SAMPLING_K}) '
                        f'({pickle_name})'] = avg_ms
                    print(f"  RL_sampling(K={RL_SAMPLING_K}) "
                          f"평균 Makespan: {avg_ms:.4f}")

                output_path = save_rl_results_to_excel(
                    rl_sampling_results,
                    f"{checkpoint_name}_sampling_K{RL_SAMPLING_K}",
                    pickle_name)
                if output_path:
                    saved_files.append(output_path)

    # =================================================================
    # 전체 결과 요약
    # =================================================================
    if algorithm_summaries:
        print(f"\n{'=' * 60}")
        print("전체 결과 요약")
        print(f"{'=' * 60}")

        for algo, avg_ms in algorithm_summaries.items():
            print(f"  {algo}: {avg_ms:.4f}")

        best_algo = min(algorithm_summaries, key=algorithm_summaries.get)
        best_ms = algorithm_summaries[best_algo]
        print(f"\n  Best: {best_algo} (Makespan: {best_ms:.4f})")

    print(f"\n{'=' * 60}")
    print(f"저장된 파일 목록 (폴더: {session_results_dir})")
    print(f"{'=' * 60}")
    for fp in saved_files:
        print(f"  - {fp.name}")

    print(f"\n모든 결과가 {session_results_dir}에 저장되었습니다.")
    print("프로그램 종료")
