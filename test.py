from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import time
import pickle
import random

from trainer import Scheduling_Trainer
from scheduling_env import SchedulingEnv
from GA import GeneticAlgorithm, Activity, Project
from gantt_chart import create_gantt_chart_from_env, create_gantt_chart_from_ga_solution, create_gantt_chart_from_schedule, create_precedence_graph

# -----------------------------
# 0) CPU 디바이스로 고정 (추론 시에는 CPU로)
# -----------------------------
device = torch.device("cpu")
print("Using device:", device)

if __name__ == "__main__":
    # =================================================================
    # 🎯 프로젝트 루트 경로 설정
    # =================================================================
    project_root = Path(__file__).parent

    # =================================================================
    # 🎯 테스트 파라미터 설정
    # =================================================================

    # -----------------------------
    # 1) 체크포인트 설정
    # -----------------------------
    CHECKPOINT_FOLDER = "20260224_105414_tardiness_DANIEL_REINFORCE"
    CHECKPOINT_FILE = "best_model.pt"
    TEST_ALL_CHECKPOINTS = False

    # -----------------------------
    # 2-1) 학습 알고리즘 설정 (체크포인트와 일치해야 함)
    # -----------------------------
    ALGORITHM_TYPE = 'reinforce'  # 'reinforce' or 'ppo'

    # -----------------------------
    # 3) 테스트할 알고리즘 설정
    # -----------------------------
    test_algorithms = ["RL", "GA", "CP"]  # ["RL"], ["GA"], ["MIP"](Gurobi), ["CP"](CP-SAT), 또는 조합

    # GA 설정
    GA_POPULATION_SIZE = 50
    GA_GENERATIONS = 1000
    GA_DECODE_MODE = "immediate"
    GA_CROSSOVER_RATE = 0.8
    GA_MUTATION_RATE = 0.2
    GA_RELEASE_MODE = "wait"
    GA_MUTEX_MODE = "wait"
    GA_VERBOSE = False
    GA_DOMINANCE_RULE = True
    GA_REPEATS = 1

    # MIP (Gurobi) 설정
    MIP_TIME_LIMIT = 30       # 인스턴스당 시간 제한 (초)

    # CP (CP-SAT) 설정
    CP_TIME_LIMIT = 30         # 인스턴스당 시간 제한 (초)

    # IL (Imitation Learning) 레이블 데이터 수집 설정
    COLLECT_IL = False                              # True: CP 기반 IL 레이블 데이터 수집 모드
    IL_CP_TIME_LIMIT = 60                           # 레이블 수집용 CP 시간 제한 (넉넉하게)
    IL_SAVE_PATH = 'data/il/il_labels.pickle'       # 저장 경로

    # 간트차트 설정
    SHOW_GANTT_CHART = True
    SHOW_PRECEDENCE_GRAPH = True

    # =================================================================
    # 🎯 목적함수 선택
    # =================================================================
    OBJECTIVE = 'tardiness'  # 'tardiness' or 'makespan'

    # =================================================================
    # 📊 테스트 데이터 설정
    # =================================================================
    # data/test/ 폴더 내 모든 .pickle 파일을 대상으로 테스트
    # 각 pickle 파일은 서로 다른 인스턴스 사이즈/데이터를 의미하며,
    # 파일 내부에 여러 인스턴스가 배치(batch)로 들어있음
    TEST_DATA_DIR = "data/test"

    # -----------------------------
    # 인스턴스 범위 설정
    # -----------------------------
    INSTANCE_START = 0  # 시작 인스턴스 번호 (None이면 0부터)
    INSTANCE_END = 1    # 끝 인스턴스 번호, 미포함 (None이면 끝까지)

    # -----------------------------
    # 기타 설정
    # -----------------------------
    ALLOW_WAIT_RELEASE = True
    ALLOW_WAIT_MUTEX = True
    DOMINANCE_RULE = True
    USE_MUTEX_ATTENTION = True

    DEBUG_ENV = False
    DEBUG_MODEL = False

    SEED = 0

    # 시드 고정
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✅ Random seed fixed to: {SEED}")

    # =================================================================
    # 📂 테스트 데이터 파일 목록 로드
    # =================================================================
    data_base_dir = project_root / TEST_DATA_DIR

    if not data_base_dir.exists():
        print(f"⚠️  경고: 테스트 데이터 폴더가 존재하지 않습니다: {data_base_dir}")
        print(f"   data_generation.py를 먼저 실행해주세요.")
        exit(1)

    # .pickle 파일 목록 (정렬)
    pickle_files = sorted(data_base_dir.glob("*.pickle"))
    if not pickle_files:
        print(f"⚠️  경고: {data_base_dir}에 .pickle 파일이 없습니다.")
        exit(1)

    print(f"📂 테스트 데이터 폴더: {data_base_dir}")
    print(f"   발견된 pickle 파일: {len(pickle_files)}개")
    for pf in pickle_files:
        print(f"     - {pf.name}")

    print("\n" + "="*50)
    print("🎯 테스트 설정")
    print("="*50)
    print(f"  📌 OBJECTIVE: {OBJECTIVE}")
    print(f"  📌 PICKLE FILES: {len(pickle_files)}개")
    print(f"  📌 ALGORITHMS: {', '.join(test_algorithms)}")
    _inst_range = f"[{INSTANCE_START if INSTANCE_START is not None else 0}, {INSTANCE_END if INSTANCE_END is not None else 'end'})"
    print(f"  📌 INSTANCE RANGE: {_inst_range}")
    if "GA" in test_algorithms:
        print(f"  📌 GA Settings:")
        print(f"     - Population: {GA_POPULATION_SIZE}")
        print(f"     - Generations: {GA_GENERATIONS}")
        print(f"     - Decode Mode: {GA_DECODE_MODE}")
        print(f"     - Release Mode: {GA_RELEASE_MODE}")
        print(f"     - Mutex Mode: {GA_MUTEX_MODE}")
        print(f"     - Dominance Rule: {GA_DOMINANCE_RULE}")
        print(f"     - Repeats/Instance: {GA_REPEATS}")
    if "MIP" in test_algorithms:
        print(f"  📌 MIP (Gurobi) Settings:")
        print(f"     - Time Limit: {MIP_TIME_LIMIT}s")
    if "CP" in test_algorithms:
        print(f"  📌 CP (CP-SAT) Settings:")
        print(f"     - Time Limit: {CP_TIME_LIMIT}s")
    print(f"  📌 SHOW_GANTT_CHART: {SHOW_GANTT_CHART}")
    print(f"  📌 SHOW_PRECEDENCE_GRAPH: {SHOW_PRECEDENCE_GRAPH}")
    print("="*50 + "\n")

    # =================================================================
    # 파라미터 설정
    # =================================================================
    model_params = None

    optimizer_params = {
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 1e-6,
        }
    }

    trainer_params = {
        'epochs': 0,
        'accumulation_steps': 1,
        'use_wandb': False,
        'wandb_project': None,
        'seed': SEED,
        'mode': 'test',
        'debug_env': DEBUG_ENV,
        'debug_model': DEBUG_MODEL,
        'device_mode': 'cpu',
        'model_device': 'cpu',
        'env_device': 'cpu',
        'model_type': 'daniel',
        'algorithm_type': ALGORITHM_TYPE,
    }

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    # =================================================================
    # 결과 저장 폴더 설정
    # =================================================================
    current_dir = Path(__file__).parent
    results_dir = current_dir / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_results_dir = results_dir / timestamp

    if not session_results_dir.exists():
        session_results_dir.mkdir(parents=True)
        print(f"✅ 결과 저장 폴더 생성: {session_results_dir}")

    gantt_dir = session_results_dir / "gantt_charts"
    if SHOW_GANTT_CHART or SHOW_PRECEDENCE_GRAPH:
        gantt_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # 유틸 함수: 결과를 엑셀로 저장
    # =================================================================
    def save_rl_results_to_excel(rl_results, checkpoint_name, pickle_name):
        """RL 결과를 엑셀 파일로 저장"""
        filename = f'RCMPSP_RL_{checkpoint_name}_{pickle_name}.xlsx'
        output_path = session_results_dir / filename

        exp_rows = [
            {'Item': 'Algorithm', 'Value': 'RL'},
            {'Item': 'Checkpoint Used', 'Value': checkpoint_name},
            {'Item': 'Data File', 'Value': pickle_name},
            {'Item': 'Device', 'Value': 'CPU'},
            {'Item': 'Num Instances', 'Value': len(rl_results)},
        ]
        if SEED is not None:
            exp_rows.append({'Item': 'Random Seed', 'Value': SEED})
        experiment_info_df = pd.DataFrame(exp_rows, columns=['Item', 'Value'])

        if not rl_results:
            print(f"결과 데이터가 없습니다: RL")
            return None

        detailed_data = []
        for result in rl_results:
            detailed_data.append({
                'Algorithm': result['algorithm'],
                'Instance': result['instance'],
                OBJECTIVE.capitalize(): result['objective_value'],
                'Runtime': result.get('runtime', 0)
            })

        valid_objectives = [r['objective_value'] for r in rl_results if r['objective_value'] is not None]
        overall_row = {
            'Algorithm': 'RL',
            'Instance': 'Average',
            OBJECTIVE.capitalize(): np.mean(valid_objectives) if valid_objectives else None,
            'Runtime': np.mean([r.get('runtime', 0) for r in rl_results])
        }

        detailed_df = pd.DataFrame(detailed_data)
        overall_avg_df = pd.DataFrame([overall_row])

        for df in [detailed_df, overall_avg_df]:
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].round(4)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            experiment_info_df.to_excel(writer, sheet_name='Experiment_Info', index=False)
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            overall_avg_df.to_excel(writer, sheet_name='Overall_Average', index=False)

        print(f"결과가 {output_path}에 저장되었습니다.")
        return output_path

    def save_ga_results_to_excel(ga_results, ga_all_repeats, pickle_name):
        """GA 결과를 엑셀 파일로 저장"""
        output_path = session_results_dir / f'RCMPSP_GA_{pickle_name}.xlsx'

        exp_rows = [
            {'Item': 'Algorithm',        'Value': 'GA'},
            {'Item': 'Data File',        'Value': pickle_name},
            {'Item': 'Population Size',  'Value': GA_POPULATION_SIZE},
            {'Item': 'Generations',      'Value': GA_GENERATIONS},
            {'Item': 'Decode Mode',      'Value': GA_DECODE_MODE},
            {'Item': 'Release Mode',     'Value': GA_RELEASE_MODE},
            {'Item': 'Mutex Mode',       'Value': GA_MUTEX_MODE},
            {'Item': 'Dominance Rule',   'Value': GA_DOMINANCE_RULE},
            {'Item': 'Repeats/Instance', 'Value': GA_REPEATS},
            {'Item': 'Device',           'Value': 'CPU'},
            {'Item': 'Num Instances',    'Value': len(ga_results)},
        ]
        if SEED is not None:
            exp_rows.append({'Item': 'Random Seed', 'Value': SEED})
        experiment_info_df = pd.DataFrame(exp_rows, columns=['Item', 'Value'])

        if not ga_results:
            print("결과 데이터가 없습니다: GA")
            return None

        detailed_data = []
        for r in ga_all_repeats:
            detailed_data.append({
                'Instance': r['instance'],
                'Repeat':   r['repeat'],
                OBJECTIVE.capitalize(): r['objective_value'],
                'Runtime':  r['runtime'],
            })
        detailed_df = pd.DataFrame(detailed_data)

        instance_summary_data = []
        for r in ga_results:
            instance_summary_data.append({
                'Instance':                       r['instance'],
                f'Best {OBJECTIVE.capitalize()}': r['best_objective'],
                f'Avg {OBJECTIVE.capitalize()}':  r['avg_objective'],
                'Total Runtime':                  r['runtime'],
            })
        instance_summary_df = pd.DataFrame(instance_summary_data)

        best_objs = [r['best_objective'] for r in ga_results if r['best_objective'] is not None]
        avg_objs  = [r['avg_objective']  for r in ga_results if r['avg_objective']  is not None]
        overall_avg_df = pd.DataFrame([{
            'Algorithm':                             'GA',
            f'Avg of Best {OBJECTIVE.capitalize()}': np.mean(best_objs) if best_objs else None,
            f'Avg of Avg {OBJECTIVE.capitalize()}':  np.mean(avg_objs)  if avg_objs  else None,
            'Avg Total Runtime':                     np.mean([r['runtime'] for r in ga_results]),
        }])

        for df in [detailed_df, instance_summary_df, overall_avg_df]:
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].round(4)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            experiment_info_df.to_excel(writer,  sheet_name='Experiment_Info',  index=False)
            detailed_df.to_excel(writer,         sheet_name='Detailed_Results', index=False)
            instance_summary_df.to_excel(writer, sheet_name='Instance_Summary', index=False)
            overall_avg_df.to_excel(writer,      sheet_name='Overall_Average',  index=False)

        print(f"결과가 {output_path}에 저장되었습니다.")
        return output_path

    def save_mip_results_to_excel(mip_results, pickle_name, solver_name):
        """MIP/CP 결과를 엑셀 파일로 저장"""
        output_path = session_results_dir / f'RCMPSP_{solver_name}_{pickle_name}.xlsx'

        exp_rows = [
            {'Item': 'Algorithm',     'Value': solver_name},
            {'Item': 'Data File',     'Value': pickle_name},
            {'Item': 'Time Limit',    'Value': MIP_TIME_LIMIT},
            {'Item': 'Device',        'Value': 'CPU'},
            {'Item': 'Num Instances', 'Value': len(mip_results)},
        ]
        if SEED is not None:
            exp_rows.append({'Item': 'Random Seed', 'Value': SEED})
        experiment_info_df = pd.DataFrame(exp_rows, columns=['Item', 'Value'])

        if not mip_results:
            print(f"결과 데이터가 없습니다: {solver_name}")
            return None

        detailed_data = []
        for r in mip_results:
            detailed_data.append({
                'Instance': r['instance'],
                OBJECTIVE.capitalize(): r['objective_value'],
                'Status': r['status'],
                'Runtime': r['runtime'],
            })
        detailed_df = pd.DataFrame(detailed_data)

        valid_objs = [r['objective_value'] for r in mip_results
                      if r['objective_value'] is not None]
        overall_row = {
            'Algorithm': solver_name,
            f'Avg {OBJECTIVE.capitalize()}': np.mean(valid_objs) if valid_objs else None,
            'Solved': len(valid_objs),
            'Total': len(mip_results),
            'Avg Runtime': np.mean([r['runtime'] for r in mip_results]),
        }
        overall_avg_df = pd.DataFrame([overall_row])

        for df in [detailed_df, overall_avg_df]:
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].round(4)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            experiment_info_df.to_excel(writer, sheet_name='Experiment_Info', index=False)
            detailed_df.to_excel(writer,        sheet_name='Detailed_Results', index=False)
            overall_avg_df.to_excel(writer,     sheet_name='Overall_Average', index=False)

        print(f"결과가 {output_path}에 저장되었습니다.")
        return output_path

    # =================================================================
    # 체크포인트 로드 준비 (RL)
    # =================================================================
    ckpt_dir = project_root / "checkpoints" / CHECKPOINT_FOLDER
    all_ckpts = []

    if "RL" in test_algorithms:
        # 첫 번째 pickle에서 checkpoint env_params 보충용 기본 env_params 구성
        first_problem = pickle.load(open(pickle_files[0], 'rb'))
        _base_env = {
            'batch_size': 1, 'pomo_size': 1,
            'objective': OBJECTIVE, 'debug_env': DEBUG_ENV,
            'state_mode': 'daniel',
            'allow_wait_release': ALLOW_WAIT_RELEASE,
            'allow_wait_mutex': ALLOW_WAIT_MUTEX,
            'dominance_rule': DOMINANCE_RULE,
            'use_mutex_attention': USE_MUTEX_ATTENTION,
        }
        for k, v in first_problem['env_params'].items():
            if k not in ('batch_size', 'pomo_size', 'state_mode', 'debug_env'):
                _base_env[k] = v

        # checkpoint에서 model_params 및 누락 env_params 보충
        if ckpt_dir.exists():
            _ckpt_for_params = ckpt_dir / (CHECKPOINT_FILE or "epoch0.pt")
            if _ckpt_for_params.exists():
                _ckpt_data = torch.load(_ckpt_for_params, map_location='cpu', weights_only=False)
                if isinstance(_ckpt_data, dict) and 'env_params' in _ckpt_data:
                    _ckpt_env = _ckpt_data['env_params']
                    _filled = []
                    for k, v in _ckpt_env.items():
                        if k not in ('batch_size', 'pomo_size', 'state_mode', 'debug_env'):
                            if k not in _base_env:
                                _base_env[k] = v
                                _filled.append(f"{k}={v}")
                            elif _base_env[k] != v and k in ('duration_min', 'duration_max'):
                                print(f"⚠️  {k} 불일치! pickle={_base_env[k]}, checkpoint={v} → checkpoint 값 사용")
                                _base_env[k] = v
                    if _filled:
                        print(f"📌 checkpoint env_params에서 누락값 보충: {', '.join(_filled)}")
                if isinstance(_ckpt_data, dict) and 'model_params' in _ckpt_data:
                    model_params = _ckpt_data['model_params']
                    print(f"📌 checkpoint에서 model_params 로드 완료")
                del _ckpt_data

        # 필수값 검증
        _required_keys = ['N_P', 'N_T', 'N_A_min', 'N_A_max', 'duration_min', 'duration_max']
        _missing = [k for k in _required_keys if k not in _base_env]
        if _missing:
            print(f"❌ env_params 필수값 누락: {_missing}")
            exit(1)

        print(f"\n📋 기본 env_params (pickle + checkpoint 기반):")
        print(f"   N_P={_base_env.get('N_P')}, N_T={_base_env.get('N_T')}, "
              f"N_A=[{_base_env.get('N_A_min')},{_base_env.get('N_A_max')}], "
              f"duration=[{_base_env.get('duration_min')},{_base_env.get('duration_max')}]")

        # 체크포인트 파일 목록
        if not ckpt_dir.exists():
            print(f"[WARNING] Checkpoint directory does not exist: {ckpt_dir}")
        elif CHECKPOINT_FILE is not None:
            ckpt_path = ckpt_dir / CHECKPOINT_FILE
            if ckpt_path.exists():
                all_ckpts = [ckpt_path]
                print(f"[INFO] Using specified checkpoint: {CHECKPOINT_FILE}")
            else:
                print(f"[WARNING] Specified checkpoint file not found: {ckpt_path}")
        elif TEST_ALL_CHECKPOINTS:
            all_ckpts = sorted(
                ckpt_dir.glob("scheduling_epoch*.pt"),
                key=lambda p: int(p.stem.replace('scheduling_epoch', ''))
            )
            if all_ckpts:
                print(f"[INFO] Found {len(all_ckpts)} checkpoints")
        else:
            ckpts = sorted(
                ckpt_dir.glob("scheduling_epoch*.pt"),
                key=lambda p: int(p.stem.replace('scheduling_epoch', ''))
            )
            if ckpts:
                all_ckpts = [ckpts[-1]]
                print(f"[INFO] Using latest checkpoint: {all_ckpts[0].name}")

    # =================================================================
    # 📦 pickle 파일별 실험 실행
    # =================================================================
    algorithm_summaries = {}
    saved_files = []

    # IL 레이블 데이터 누적용
    il_all_pairs = []

    for pickle_path in pickle_files:
        pickle_name = pickle_path.stem  # e.g., "test_batch"
        print(f"\n{'='*70}")
        print(f"📂 Processing: {pickle_path.name}")
        print(f"{'='*70}")

        with open(pickle_path, 'rb') as f:
            problem = pickle.load(f)

        total_instances = problem['num_activities'].shape[0]
        inst_start = INSTANCE_START if INSTANCE_START is not None else 0
        inst_end = INSTANCE_END if INSTANCE_END is not None else total_instances
        inst_start = max(0, min(inst_start, total_instances))
        inst_end = max(inst_start, min(inst_end, total_instances))
        num_instances = inst_end - inst_start
        print(f"   전체 인스턴스: {total_instances}, 실행 범위: [{inst_start}, {inst_end}) ({num_instances}개)")

        # 이 pickle에 맞는 env_params 구성 (RL은 전체 배치로 추론)
        env_params = {
            'batch_size': total_instances,
            'pomo_size': 1,
            'objective': OBJECTIVE,
            'debug_env': DEBUG_ENV,
            'state_mode': 'daniel',
            'allow_wait_release': ALLOW_WAIT_RELEASE,
            'allow_wait_mutex': ALLOW_WAIT_MUTEX,
            'dominance_rule': DOMINANCE_RULE,
            'use_mutex_attention': USE_MUTEX_ATTENTION,
        }
        saved_env = problem['env_params']
        for k, v in saved_env.items():
            if k not in ('batch_size', 'pomo_size', 'state_mode', 'debug_env'):
                env_params[k] = v

        # checkpoint에서 보충 (RL 테스트 시)
        if "RL" in test_algorithms and '_base_env' in dir():
            for k in ('duration_min', 'duration_max'):
                if k in _base_env and (k not in env_params or env_params[k] != _base_env[k]):
                    env_params[k] = _base_env[k]

        print(f"   N_P={env_params.get('N_P')}, N_T={env_params.get('N_T')}, "
              f"N_A=[{env_params.get('N_A_min')},{env_params.get('N_A_max')}], "
              f"duration=[{env_params.get('duration_min')},{env_params.get('duration_max')}]")

        # 선후관계 그래프 생성 추적
        precedence_graphs_created = set()

        # =============================================
        # GA 실행 (인스턴스별 for문)
        # =============================================
        if "GA" in test_algorithms:
            print(f"\n  --- GA ({pickle_name}) ---")

            ga_results = []
            ga_all_repeats = []

            from data_generator import convert_problem_to_ga_format

            for i in range(inst_start, inst_end):
                print(f"\n   📋 Instance {i}/{inst_end-1}")

                start_time = time.time()

                try:
                    projects = convert_problem_to_ga_format(problem, i, env_params['N_T'])

                    best_solution = None
                    best_ga = None
                    best_objective = float('inf')
                    rep_objectives = []

                    for rep in range(GA_REPEATS):
                        rep_start = time.time()
                        ga = GeneticAlgorithm(
                            projects=projects,
                            num_teams=env_params['N_T'],
                            population_size=GA_POPULATION_SIZE,
                            generations=GA_GENERATIONS,
                            crossover_rate=GA_CROSSOVER_RATE,
                            mutation_rate=GA_MUTATION_RATE,
                            decode_mode=GA_DECODE_MODE,
                            release_mode=GA_RELEASE_MODE,
                            mutex_mode=GA_MUTEX_MODE,
                            dominance_rule=GA_DOMINANCE_RULE,
                            verbose=GA_VERBOSE
                        )
                        sol = ga.evolve()
                        rep_runtime = time.time() - rep_start

                        rep_obj = sol.objective
                        rep_objectives.append(rep_obj)
                        ga_all_repeats.append({
                            'instance':        i,
                            'repeat':          rep + 1,
                            'objective_value': rep_obj,
                            'runtime':         rep_runtime,
                        })

                        if rep_obj < best_objective:
                            best_objective = rep_obj
                            best_solution = sol
                            best_ga = ga

                    ga = best_ga

                    end_time = time.time()
                    runtime = end_time - start_time

                    avg_objective = np.mean(rep_objectives)
                    ga_results.append({
                        'algorithm':      'GA',
                        'instance':       i,
                        'best_objective': best_objective,
                        'avg_objective':  avg_objective,
                        'runtime':        runtime
                    })

                    rep_str = f" (best of {GA_REPEATS}, avg: {avg_objective:.4f})" if GA_REPEATS > 1 else ""
                    print(f"     [GA][Instance {i}] {OBJECTIVE}: best={best_objective:.4f}, Runtime: {runtime:.4f}s{rep_str}")

                    if SHOW_GANTT_CHART:
                        try:
                            project_due_dates = {proj.id: proj.due_date for proj in ga.projects}
                            create_gantt_chart_from_ga_solution(
                                solution=best_solution,
                                activity_to_project=ga.activity_to_project,
                                num_teams=env_params['N_T'],
                                instance_name=f"{pickle_name}_instance_{i}",
                                objective_value=best_objective,
                                project_due_dates=project_due_dates,
                                save_dir=gantt_dir,
                                show=True
                            )
                        except Exception as e:
                            print(f"     ⚠️ 간트차트 생성 실패: {e}")
                    if SHOW_PRECEDENCE_GRAPH and i not in precedence_graphs_created:
                        try:
                            activity_predecessors = {}
                            activity_mutex = {}
                            act_eligible = {}
                            act_proc_times = {}
                            for act in ga.activities:
                                activity_predecessors[act.id] = act.predecessors
                                activity_mutex[act.id] = act.mutually_exclusive
                                act_eligible[act.id] = act.eligible_teams
                                if act.duration_by_team:
                                    act_proc_times[act.id] = act.duration_by_team
                            create_precedence_graph(
                                activity_predecessors=activity_predecessors,
                                activity_mutex=activity_mutex,
                                activity_to_project=ga.activity_to_project,
                                instance_name=f"{pickle_name}_instance_{i}",
                                activity_eligible_teams=act_eligible,
                                activity_processing_times=act_proc_times if act_proc_times else None,
                                save_dir=gantt_dir,
                                show=True
                            )
                            precedence_graphs_created.add(i)
                        except Exception as e:
                            print(f"     ⚠️ 선후관계 그래프 생성 실패: {e}")

                except Exception as e:
                    print(f"     ❌ [GA][Instance {i}] 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if ga_results:
                best_objs = [r['best_objective'] for r in ga_results if r['best_objective'] is not None]
                avg_objs  = [r['avg_objective']  for r in ga_results if r['avg_objective']  is not None]
                if best_objs:
                    avg_of_best = np.mean(best_objs)
                    avg_of_avg  = np.mean(avg_objs) if avg_objs else None
                    algorithm_summaries[f'GA_best ({pickle_name})'] = avg_of_best
                    if avg_of_avg is not None:
                        algorithm_summaries[f'GA_avg ({pickle_name})'] = avg_of_avg
                    print(f"  GA 평균 {OBJECTIVE}: best={avg_of_best:.4f}" +
                          (f", avg={avg_of_avg:.4f}" if avg_of_avg is not None else ""))

                output_path = save_ga_results_to_excel(ga_results, ga_all_repeats, pickle_name)
                if output_path:
                    saved_files.append(output_path)

        # =============================================
        # MIP (Gurobi) 실행
        # =============================================
        if "MIP" in test_algorithms:
            if OBJECTIVE != 'tardiness':
                print(f"\n  ⚠️ MIP (Gurobi) 솔버는 현재 tardiness만 지원합니다. (현재: {OBJECTIVE}) → 스킵")
            else:
                from data_generator import convert_problem_to_mip_format
                from samsung_MIP import HAS_GUROBI
                if not HAS_GUROBI:
                    print("\n  ⚠️ Gurobi가 설치되지 않았습니다. MIP를 건너뜁니다.")
                else:
                    from samsung_MIP import solve_rcmpsp_gurobi
                    print(f"\n  --- MIP Gurobi ({pickle_name}) ---")
                    mip_results = []

                    for i in range(inst_start, inst_end):
                        print(f"\n   📋 Instance {i}/{inst_end-1}")
                        try:
                            inst = convert_problem_to_mip_format(problem, i)
                            start_time = time.time()
                            result = solve_rcmpsp_gurobi(inst, MIP_TIME_LIMIT)
                            runtime = time.time() - start_time

                            if result is not None:
                                obj_val, start_times_sol, assigned_teams_sol, status = result
                                print(f"     [Gurobi][Instance {i}] "
                                      f"{OBJECTIVE}: {obj_val:.4f}, "
                                      f"Status: {status}, "
                                      f"Runtime: {runtime:.4f}s")

                                if SHOW_GANTT_CHART:
                                    try:
                                        schedule = {}
                                        for act_id in range(inst.num_activities):
                                            t = assigned_teams_sol.get(act_id)
                                            if t is not None:
                                                s = start_times_sol[act_id]
                                                dur = inst.activity_team_durations[act_id][t]
                                                schedule[act_id] = (s, s + dur, t)

                                        project_due_dates = {p: inst.due_dates[p] for p in range(inst.num_projects)}
                                        create_gantt_chart_from_schedule(
                                            schedule=schedule,
                                            activity_to_project=inst.activity_to_project,
                                            num_teams=inst.num_teams,
                                            instance_name=f"{pickle_name}_instance_{i}",
                                            algorithm="Gurobi",
                                            objective_value=obj_val,
                                            project_due_dates=project_due_dates,
                                            save_dir=gantt_dir,
                                            show=True
                                        )
                                    except Exception as e:
                                        print(f"     ⚠️ 간트차트 생성 실패: {e}")
                                if SHOW_PRECEDENCE_GRAPH and i not in precedence_graphs_created:
                                    try:
                                        inst_proc_times = {}
                                        has_team_dur = hasattr(inst, 'activity_team_durations')
                                        for a_id in range(inst.num_activities):
                                            inst_proc_times[a_id] = {}
                                            for k in inst.activity_to_teams.get(a_id, []):
                                                if has_team_dur:
                                                    inst_proc_times[a_id][k] = inst.activity_team_durations[a_id][k]
                                                else:
                                                    inst_proc_times[a_id][k] = inst.durations[a_id]
                                        create_precedence_graph(
                                            activity_predecessors=inst.precedences,
                                            activity_mutex=inst.nooverlaps,
                                            activity_to_project=inst.activity_to_project,
                                            instance_name=f"{pickle_name}_instance_{i}",
                                            activity_eligible_teams=inst.activity_to_teams,
                                            activity_processing_times=inst_proc_times,
                                            save_dir=gantt_dir,
                                            show=True
                                        )
                                        precedence_graphs_created.add(i)
                                    except Exception as e:
                                        print(f"     ⚠️ 선후관계 그래프 생성 실패: {e}")
                            else:
                                obj_val = None
                                status = "no_solution"
                                print(f"     [Gurobi][Instance {i}] "
                                      f"No solution found, Runtime: {runtime:.4f}s")

                            mip_results.append({
                                'instance': i,
                                'objective_value': obj_val,
                                'status': status,
                                'runtime': runtime,
                            })
                        except Exception as e:
                            print(f"     ❌ [Gurobi][Instance {i}] 오류: {e}")
                            import traceback
                            traceback.print_exc()
                            mip_results.append({
                                'instance': i,
                                'objective_value': None,
                                'status': 'error',
                                'runtime': 0,
                            })

                    if mip_results:
                        valid_objs = [r['objective_value'] for r in mip_results
                                      if r['objective_value'] is not None]
                        if valid_objs:
                            avg_obj = np.mean(valid_objs)
                            solved = len(valid_objs)
                            algorithm_summaries[f'Gurobi ({pickle_name})'] = avg_obj
                            print(f"  Gurobi 평균 {OBJECTIVE}: {avg_obj:.4f} "
                                  f"({solved}/{num_instances} solved)")

                        output_path = save_mip_results_to_excel(
                            mip_results, pickle_name, "Gurobi")
                        if output_path:
                            saved_files.append(output_path)

        # =============================================
        # CP (CP-SAT) 실행
        # =============================================
        if "CP" in test_algorithms:
            if OBJECTIVE != 'tardiness':
                print(f"\n  ⚠️ CP (CP-SAT) 솔버는 현재 tardiness만 지원합니다. (현재: {OBJECTIVE}) → 스킵")
            else:
                from data_generator import convert_problem_to_mip_format
                from samsung_MIP import solve_rcmpsp_cp
                print(f"\n  --- CP-SAT ({pickle_name}) ---")
                cp_results = []

                for i in range(inst_start, inst_end):
                    print(f"\n   📋 Instance {i}/{inst_end-1}")
                    try:
                        inst = convert_problem_to_mip_format(problem, i)
                        start_time = time.time()
                        result = solve_rcmpsp_cp(inst, CP_TIME_LIMIT)
                        runtime = time.time() - start_time

                        if result is not None:
                            obj_val, start_times_sol, assigned_teams_sol, status = result
                            print(f"     [CP-SAT][Instance {i}] "
                                  f"{OBJECTIVE}: {obj_val:.4f}, "
                                  f"Status: {status}, "
                                  f"Runtime: {runtime:.4f}s")

                            if SHOW_GANTT_CHART:
                                try:
                                    schedule = {}
                                    for act_id in range(inst.num_activities):
                                        t = assigned_teams_sol.get(act_id)
                                        if t is not None:
                                            s = start_times_sol[act_id]
                                            dur = inst.activity_team_durations[act_id][t]
                                            schedule[act_id] = (s, s + dur, t)

                                    project_due_dates = {p: inst.due_dates[p] for p in range(inst.num_projects)}
                                    create_gantt_chart_from_schedule(
                                        schedule=schedule,
                                        activity_to_project=inst.activity_to_project,
                                        num_teams=inst.num_teams,
                                        instance_name=f"{pickle_name}_instance_{i}",
                                        algorithm="CP-SAT",
                                        objective_value=obj_val,
                                        project_due_dates=project_due_dates,
                                        save_dir=gantt_dir,
                                        show=True
                                    )
                                except Exception as e:
                                    print(f"     ⚠️ 간트차트 생성 실패: {e}")
                            if SHOW_PRECEDENCE_GRAPH and i not in precedence_graphs_created:
                                try:
                                    inst_proc_times_cp = {}
                                    has_team_dur_cp = hasattr(inst, 'activity_team_durations')
                                    for a_id in range(inst.num_activities):
                                        inst_proc_times_cp[a_id] = {}
                                        for k in inst.activity_to_teams.get(a_id, []):
                                            if has_team_dur_cp:
                                                inst_proc_times_cp[a_id][k] = inst.activity_team_durations[a_id][k]
                                            else:
                                                inst_proc_times_cp[a_id][k] = inst.durations[a_id]
                                    create_precedence_graph(
                                        activity_predecessors=inst.precedences,
                                        activity_mutex=inst.nooverlaps,
                                        activity_to_project=inst.activity_to_project,
                                        instance_name=f"{pickle_name}_instance_{i}",
                                        activity_eligible_teams=inst.activity_to_teams,
                                        activity_processing_times=inst_proc_times_cp,
                                        save_dir=gantt_dir,
                                        show=True
                                    )
                                    precedence_graphs_created.add(i)
                                except Exception as e:
                                    print(f"     ⚠️ 선후관계 그래프 생성 실패: {e}")
                        else:
                            obj_val = None
                            status = "no_solution"
                            print(f"     [CP-SAT][Instance {i}] "
                                  f"No solution found, Runtime: {runtime:.4f}s")

                        cp_results.append({
                            'instance': i,
                            'objective_value': obj_val,
                            'status': status,
                            'runtime': runtime,
                        })
                    except Exception as e:
                        print(f"     ❌ [CP-SAT][Instance {i}] 오류: {e}")
                        import traceback
                        traceback.print_exc()
                        cp_results.append({
                            'instance': i,
                            'objective_value': None,
                            'status': 'error',
                            'runtime': 0,
                        })

                if cp_results:
                    valid_objs = [r['objective_value'] for r in cp_results
                                  if r['objective_value'] is not None]
                    if valid_objs:
                        avg_obj = np.mean(valid_objs)
                        solved = len(valid_objs)
                        algorithm_summaries[f'CP-SAT ({pickle_name})'] = avg_obj
                        print(f"  CP-SAT 평균 {OBJECTIVE}: {avg_obj:.4f} "
                              f"({solved}/{num_instances} solved)")

                    output_path = save_mip_results_to_excel(
                        cp_results, pickle_name, "CP-SAT")
                    if output_path:
                        saved_files.append(output_path)

        # =============================================
        # IL 레이블 데이터 수집 (CP-SAT → replay → (state, action) 쌍)
        # =============================================
        if COLLECT_IL:
            if OBJECTIVE != 'tardiness':
                print(f"\n  ⚠️ IL 레이블 수집은 현재 tardiness만 지원합니다. (현재: {OBJECTIVE}) → 스킵")
            else:
                from il_utils import extract_single_instance, replay_cp_solution
                from data_generator import convert_problem_to_mip_format
                from samsung_MIP import solve_rcmpsp_cp

                print(f"\n  --- IL Label Collection ({pickle_name}) ---")
                il_success_count = 0

                for i in range(inst_start, inst_end):
                    print(f"\n   📋 IL Instance {i}/{inst_end-1}")
                    try:
                        inst = convert_problem_to_mip_format(problem, i)
                        il_start_time = time.time()
                        result = solve_rcmpsp_cp(inst, IL_CP_TIME_LIMIT)
                        il_runtime = time.time() - il_start_time

                        if result is None:
                            print(f"     ⚠️ [Instance {i}] CP 해 없음 → 스킵 (Runtime: {il_runtime:.2f}s)")
                            continue

                        obj_val, start_times_sol, assigned_teams_sol, status = result
                        print(f"     [Instance {i}] CP obj: {obj_val:.4f}, status: {status}, runtime: {il_runtime:.2f}s")

                        # 단일 인스턴스 추출
                        problem_single, env_params_single = extract_single_instance(problem, env_params, i)

                        # 환경에서 replay → (state, action) 수집
                        pairs = replay_cp_solution(
                            problem_single, env_params_single,
                            start_times_sol, assigned_teams_sol
                        )

                        il_all_pairs.extend(pairs)
                        il_success_count += 1
                        print(f"     ✅ [Instance {i}] {len(pairs)} pairs collected")

                    except Exception as e:
                        print(f"     ❌ [Instance {i}] 오류: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                print(f"\n  IL 수집 결과 ({pickle_name}): "
                      f"{il_success_count}/{inst_end-inst_start} instances, "
                      f"{len(il_all_pairs)} total pairs (누적)")

        # =============================================
        # RL 실행 (배치 추론)
        # =============================================
        for ckpt_path in all_ckpts:
            checkpoint_name = ckpt_path.stem
            print(f"\n  --- RL: {checkpoint_name} ({pickle_name}) ---")

            try:
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

                if model_params is None:
                    if isinstance(checkpoint, dict) and 'model_params' in checkpoint:
                        model_params = checkpoint['model_params']
                        print(f"  📌 checkpoint에서 model_params 로드 완료")
                    else:
                        print(f"  ❌ model_params를 찾을 수 없습니다")
                        continue

                trainer = Scheduling_Trainer(env_params, model_params, optimizer_params, trainer_params)

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    trainer.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"  ✅ 모델 로드: {checkpoint_name}")
                    saved_alg = checkpoint.get('algorithm_type',
                                checkpoint.get('trainer_params', {}).get('algorithm_type', 'reinforce'))
                    print(f"     └─ Algorithm: {str(saved_alg).upper()}")
                else:
                    trainer.model.load_state_dict(checkpoint)
                    print(f"  ✅ 모델 로드 (이전 형식): {checkpoint_name}")

                trainer.model.to(device)
                trainer.model.eval()

            except Exception as e:
                print(f"  ❌ 모델 로드 실패: {e}")
                import traceback
                traceback.print_exc()
                continue

            print(f"  📋 배치 추론 시작 (전체 {total_instances}개, 결과 범위 [{inst_start},{inst_end}))")
            start_time = time.time()

            try:
                with torch.no_grad():
                    test_env = SchedulingEnv(env_params, debug_env=False, device='cpu')
                    test_env._reset(problem)

                    done = False
                    s = test_env._get_state()

                    while not done:
                        fea_act = s.fea_act_tensor.to(device)
                        act_mask = s.act_mask_tensor.to(device)
                        candidate = s.candidate_tensor.to(device)
                        fea_team = s.fea_team_tensor.to(device)
                        team_mask = s.team_mask_tensor.to(device)
                        comp_idx = s.comp_idx_tensor.to(device)
                        dynamic_pair_mask = s.dynamic_pair_mask_tensor.to(device)
                        fea_pairs = s.fea_pairs_tensor.to(device)
                        pred_idx = s.pred_idx_tensor.to(device)
                        succ_idx = s.succ_idx_tensor.to(device)
                        mutex_idx = s.mutex_idx_tensor.to(device) if s.mutex_idx_tensor is not None else None

                        pi, v = trainer.model(
                            fea_act, act_mask, candidate, fea_team,
                            team_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                            pred_idx, succ_idx, mutex_idx
                        )
                        action_flat = torch.argmax(pi, dim=1)
                        N_T = test_env.N_T
                        act_idx  = action_flat // N_T
                        team_idx = action_flat % N_T
                        s, obj_value, done = test_env.step_pair(
                            act_idx.to(test_env.device),
                            team_idx.to(test_env.device)
                        )

                    scores = test_env._get_obj()  # (total_instances,)

                end_time = time.time()
                total_runtime = end_time - start_time
                avg_runtime = total_runtime / total_instances

                rl_results = []
                for i in range(inst_start, inst_end):
                    score_i = scores[i].item()
                    rl_results.append({
                        'algorithm': 'RL',
                        'instance': i,
                        'objective_value': score_i,
                        'runtime': avg_runtime,
                    })
                    print(f"     [RL][Instance {i}] {OBJECTIVE}: {score_i:.4f}")

                print(f"\n     Total: {total_runtime:.4f}s (avg {avg_runtime:.4f}s/instance)")

                if SHOW_GANTT_CHART or SHOW_PRECEDENCE_GRAPH:
                    for i in range(inst_start, inst_end):
                        try:
                            num_act = test_env.num_activities[i].item()

                            # env에서 schedule 추출
                            schedule = {}
                            act_to_proj = {}
                            for act_id in range(num_act):
                                if test_env.activity_started[i, act_id].item():
                                    s = test_env.activity_start_time[i, act_id].item()
                                    e = test_env.activity_end_time[i, act_id].item()
                                    t = test_env.activity_assigned_team[i, act_id].item()
                                    schedule[act_id] = (s, e, t)
                                act_to_proj[act_id] = test_env.activity_project[i, act_id].item()

                            if SHOW_GANTT_CHART:
                                proj_due = {}
                                for p in range(test_env.N_P):
                                    proj_due[p] = test_env.project_due_date[i, p].item()

                                create_gantt_chart_from_schedule(
                                    schedule=schedule,
                                    activity_to_project=act_to_proj,
                                    num_teams=test_env.N_T,
                                    instance_name=f"{pickle_name}_instance_{i}",
                                    algorithm="RL",
                                    objective_value=scores[i].item(),
                                    project_due_dates=proj_due,
                                    save_dir=gantt_dir,
                                    show=True
                                )

                            if SHOW_PRECEDENCE_GRAPH and i not in precedence_graphs_created:
                                activity_predecessors = {}
                                activity_mutex = {}
                                rl_eligible = {}
                                rl_proc_times = {}
                                for act_id in range(num_act):
                                    preds = test_env.activity_predecessors[i, act_id].cpu().numpy().tolist()
                                    activity_predecessors[act_id] = [p for p in preds if p >= 0]
                                    mutex = test_env.activity_mutex[i, act_id].cpu().numpy().tolist()
                                    activity_mutex[act_id] = [m for m in mutex if m >= 0]
                                    elig_mask = test_env.activity_eligible_teams[i, act_id].cpu()
                                    teams = [t for t in range(test_env.N_T) if elig_mask[t].item()]
                                    rl_eligible[act_id] = teams
                                    durations = test_env.activity_team_duration[i, act_id].cpu()
                                    rl_proc_times[act_id] = {t: durations[t].item() for t in teams}
                                create_precedence_graph(
                                    activity_predecessors=activity_predecessors,
                                    activity_mutex=activity_mutex,
                                    activity_to_project=act_to_proj,
                                    instance_name=f"{pickle_name}_instance_{i}",
                                    activity_eligible_teams=rl_eligible,
                                    activity_processing_times=rl_proc_times,
                                    save_dir=gantt_dir,
                                    show=True
                                )
                                precedence_graphs_created.add(i)
                        except Exception as e:
                            print(f"     ⚠️ 간트차트 생성 실패 (instance {i}): {e}")

            except Exception as e:
                print(f"  ❌ 배치 추론 중 오류: {e}")
                import traceback
                traceback.print_exc()
                rl_results = []

            if rl_results:
                valid_objectives = [r['objective_value'] for r in rl_results if r['objective_value'] is not None]
                if valid_objectives:
                    avg_objective = np.mean(valid_objectives)
                    algorithm_summaries[f'RL_{checkpoint_name} ({pickle_name})'] = avg_objective
                    print(f"  RL 평균 {OBJECTIVE}: {avg_objective:.4f}")

                output_path = save_rl_results_to_excel(rl_results, checkpoint_name, pickle_name)
                if output_path:
                    saved_files.append(output_path)

    # =================================================================
    # IL 레이블 데이터 저장 (모든 pickle 파일에서 누적된 pairs)
    # =================================================================
    if COLLECT_IL and il_all_pairs:
        il_save_dir = os.path.dirname(IL_SAVE_PATH)
        if il_save_dir:
            os.makedirs(il_save_dir, exist_ok=True)
        with open(IL_SAVE_PATH, 'wb') as f:
            pickle.dump({
                'pairs': il_all_pairs,
                'env_params': env_params,
            }, f)
        print(f"\n{'='*60}")
        print(f"IL Label Data Saved")
        print(f"{'='*60}")
        print(f"  총 (state, action) 쌍: {len(il_all_pairs)}")
        print(f"  저장 경로: {IL_SAVE_PATH}")
    elif COLLECT_IL:
        print(f"\n⚠️ IL 레이블 데이터가 수집되지 않았습니다. (CP 해를 찾은 인스턴스 없음)")

    # =================================================================
    # 전체 결과 요약
    # =================================================================
    if algorithm_summaries:
        print(f"\n{'='*60}")
        print("전체 결과 요약")
        print(f"{'='*60}")

        for algo, avg_obj in algorithm_summaries.items():
            print(f"  {algo}: {avg_obj:.4f}")

        best_algo = min(algorithm_summaries, key=algorithm_summaries.get)
        best_obj = algorithm_summaries[best_algo]
        print(f"\n🏆 Best: {best_algo} ({OBJECTIVE}: {best_obj:.4f})")

    print(f"\n{'='*60}")
    print(f"저장된 파일 목록 (폴더: {session_results_dir})")
    print(f"{'='*60}")
    for file_path in saved_files:
        print(f"- {file_path.name}")

    print(f"\n✅ 모든 결과가 {session_results_dir}에 저장되었습니다.")
    print("✅ 프로그램 종료")
