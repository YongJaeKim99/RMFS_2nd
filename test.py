from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.distributions import Categorical
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
# 0) 디바이스 설정은 __main__ 블록에서 DEVICE_MODE로 제어
# -----------------------------

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
    CHECKPOINT_FOLDER = "TOPOFORMER_PPO"
    CHECKPOINT_FILE = "best_model.pt"
    TEST_ALL_CHECKPOINTS = False

    # --- 폴더 이름에서 model_type, algorithm_type 자동 감지 ---
    # 폴더 이름 형식: {timestamp}_{objective}_{MODEL}_{ALG} 또는 {MODEL}_{ALG}
    _MODEL_LABEL_MAP = {
        'DANIEL': 'daniel', 'GATV1': 'gat_v1', 'GATV2': 'gat_v2',
        'RGAT': 'rgat', 'TOPOFORMER': 'topoformer',
    }
    _ALG_LABEL_MAP = {
        'PPO': 'ppo', 'IL': 'il', 'REINFORCE': 'reinforce',
    }

    def parse_checkpoint_folder(folder_name):
        """폴더 이름에서 model_type, algorithm_type 추출 (감지 실패 시 None)"""
        parts = folder_name.upper().split('_')
        detected_model, detected_alg = None, None
        for part in reversed(parts):
            if detected_alg is None and part in _ALG_LABEL_MAP:
                detected_alg = _ALG_LABEL_MAP[part]
            elif detected_model is None and part in _MODEL_LABEL_MAP:
                detected_model = _MODEL_LABEL_MAP[part]
            if detected_model and detected_alg:
                break
        return detected_model, detected_alg

    _folder_model_type, _folder_alg_type = parse_checkpoint_folder(CHECKPOINT_FOLDER)

    # 최종 ALGORITHM_TYPE / MODEL_TYPE (폴더 이름 우선, checkpoint 파일에서 fallback)
    ALGORITHM_TYPE = _folder_alg_type  # None이면 checkpoint 파일에서 갱신
    MODEL_TYPE = _folder_model_type    # None이면 checkpoint 파일에서 갱신

    # -----------------------------
    # 3) 테스트할 알고리즘 설정
    # -----------------------------
    test_algorithms = ["CP"]  # ["RL"], ["GA"], ["MIP"](Gurobi), ["CP"](CP-SAT), 또는 조합

    # RL Sampling 설정
    RL_SAMPLING_K = 64   # Sampling rollout 총 횟수 (0이면 greedy만 수행)
    RL_SAMPLING_BATCH = 4  # 한 번에 병렬 처리할 sampling 수 (메모리 제어)

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
    GA_REPEATS = 3

    # MIP (Gurobi) 설정
    MIP_TIME_LIMIT = 30       # 인스턴스당 시간 제한 (초)

    # CP (CP-SAT) 설정
    CP_TIME_LIMIT = 30         # 인스턴스당 시간 제한 (초)

    # CP 실행 시 결정 시퀀스를 csv로 저장 (인간 확인용)
    SAVE_CP_DECISIONS = True
    #SAVE_CP_DECISIONS = False

    # CP Solution 저장 설정 (IL 학습 및 cp_replay.py 공용)
    SAVE_CP_SOLUTIONS = True                         # True: CP 실행 후 solution pickle 저장
    CP_SOLUTION_DIR = "data/il/cp_solutions"          # CP solution 저장 폴더

    # 간트차트 설정
    #SHOW_GANTT_CHART = True
    #SHOW_PRECEDENCE_GRAPH = True
    SHOW_GANTT_CHART = False
    SHOW_PRECEDENCE_GRAPH = False
    # =================================================================
    # 🎯 목적함수 선택
    # =================================================================
    #OBJECTIVE = 'makespan'  # 'tardiness' or 'makespan'
    OBJECTIVE = 'tardiness'


    # =================================================================
    # 📊 테스트 데이터 설정
    # =================================================================
    # SAVE_CP_SOLUTIONS=False: data/test/ 폴더에서 테스트
    # SAVE_CP_SOLUTIONS=True:  data/il/il_label_instance/ 폴더에서 CP solution 수집
    TEST_DATA_DIR = "data/test"
    IL_DATA_DIR = "data/il/il_label_instance"

    # -----------------------------
    # 인스턴스 범위 설정
    # -----------------------------
    INSTANCE_START = 0  # 시작 인스턴스 번호 (None이면 0부터)
    INSTANCE_END = 3    # 끝 인스턴스 번호, 미포함 (None이면 끝까지)

    # -----------------------------
    # 기타 설정
    # -----------------------------
    DEVICE_MODE = 'gpu'  # 'cpu', 'hybrid', 'gpu'
    # 'cpu': 전부 CPU, 'hybrid': 모델은 GPU + 환경은 CPU, 'gpu': 전부 GPU

    ALLOW_WAIT_RELEASE = True
    ALLOW_WAIT_MUTEX = True
    ALLOW_WAIT_TEAM = True
    ALLOW_WAIT_PRED = True
    USE_MUTEX_ATTENTION = True

    DEBUG_ENV = False
    DEBUG_MODEL = False

    SEED = 0

    # =================================================================
    # 🖥️ 디바이스 설정
    # =================================================================
    cuda_available = torch.cuda.is_available()

    if not cuda_available and DEVICE_MODE != 'cpu':
        print(f"⚠️ GPU가 사용 불가능합니다. DEVICE_MODE를 'cpu'로 변경합니다.")
        DEVICE_MODE = 'cpu'

    if DEVICE_MODE == 'cpu':
        device = torch.device('cpu')
        device_desc = "CPU"
    elif DEVICE_MODE == 'hybrid':
        device = torch.device('cuda')
        device_desc = "GPU (Hybrid Mode)"
    elif DEVICE_MODE == 'gpu':
        device = torch.device('cuda')
        device_desc = "GPU"
    else:
        raise ValueError(f"Invalid DEVICE_MODE: {DEVICE_MODE}")

    print(f"🖥️  Device Mode: {DEVICE_MODE}")
    if cuda_available:
        print(f"   ├─ CUDA Available: Yes (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"   ├─ CUDA Available: No")
    print(f"   └─ Using: {device_desc}")

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
    data_base_dir = project_root / (IL_DATA_DIR if SAVE_CP_SOLUTIONS else TEST_DATA_DIR)

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
    print(f"  📌 DEVICE MODE: {DEVICE_MODE} ({device_desc})")
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
        print(f"     - Save Decisions: {SAVE_CP_DECISIONS}")
    if "RL" in test_algorithms:
        print(f"  📌 RL Checkpoint: {CHECKPOINT_FOLDER}")
        if RL_SAMPLING_K > 0:
            print(f"     - Sampling: K={RL_SAMPLING_K}, mini_batch={RL_SAMPLING_BATCH}")
    if SAVE_CP_SOLUTIONS:
        print(f"  📌 SAVE_CP_SOLUTIONS Settings:")
        print(f"     - IL Data Dir: {IL_DATA_DIR}")
        print(f"     - CP Solution Dir: {CP_SOLUTION_DIR}")
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
        'device': 'cuda' if DEVICE_MODE in ('hybrid', 'gpu') else 'cpu',
        'device_mode': DEVICE_MODE,
        'model_type': MODEL_TYPE or 'daniel',     # 폴더 이름 → checkpoint fallback
        'algorithm_type': ALGORITHM_TYPE or 'reinforce',  # 폴더 이름 → checkpoint fallback
    }

    # model_type 추적 (폴더 이름 → checkpoint 파일 순으로 갱신)
    _model_type_from_ckpt = _folder_model_type

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
        has_lb = any('lower_bound' in r for r in mip_results)
        for r in mip_results:
            row = {
                'Instance': r['instance'],
                OBJECTIVE.capitalize(): r['objective_value'],
            }
            if has_lb:
                row['Lower Bound'] = r.get('lower_bound')
                row['Gap (%)'] = r.get('gap')
            row['Status'] = r['status']
            row['Runtime'] = r['runtime']
            detailed_data.append(row)
        detailed_df = pd.DataFrame(detailed_data)

        valid_objs = [r['objective_value'] for r in mip_results
                      if r['objective_value'] is not None]
        overall_row = {
            'Algorithm': solver_name,
            f'Avg {OBJECTIVE.capitalize()}': np.mean(valid_objs) if valid_objs else None,
        }
        if has_lb:
            valid_lbs = [r['lower_bound'] for r in mip_results if r.get('lower_bound') is not None]
            valid_gaps = [r['gap'] for r in mip_results if r.get('gap') is not None]
            overall_row['Avg Lower Bound'] = np.mean(valid_lbs) if valid_lbs else None
            overall_row['Avg Gap (%)'] = np.mean(valid_gaps) if valid_gaps else None
        overall_row.update({
            'Solved': len(valid_objs),
            'Total': len(mip_results),
            'Avg Runtime': np.mean([r['runtime'] for r in mip_results]),
        })
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

    def save_cp_decisions_to_csv(instance_idx, pickle_name, inst,
                                  start_times_sol, assigned_teams_sol,
                                  obj_val, status, save_dir):
        """CP 결정 시퀀스를 csv 파일로 저장 (step 순서 = start_time 오름차순)
        간트차트의 A{act_id}({step}) 표기와 동일한 step 순서"""
        import csv
        decisions_dir = save_dir / "cp_decisions"
        decisions_dir.mkdir(parents=True, exist_ok=True)
        filepath = decisions_dir / f"{pickle_name}_instance_{instance_idx}.csv"

        sorted_acts = sorted(
            [a for a in range(inst.num_activities) if assigned_teams_sol.get(a) is not None],
            key=lambda a: (start_times_sol.get(a, float('inf')), a)
        )

        with open(filepath, 'w', newline='') as f:
            # 메타데이터를 첫 줄에 주석으로
            f.write(f"# pickle={pickle_name} instance={instance_idx} objective_value={obj_val} status={status} num_activities={inst.num_activities} num_teams={inst.num_teams} num_projects={inst.num_projects}\n")
            writer = csv.writer(f)
            writer.writerow(['step', 'activity', 'project', 'team', 'start_time', 'end_time', 'duration'])
            for step, act_id in enumerate(sorted_acts):
                team_id = assigned_teams_sol[act_id]
                s_time = start_times_sol[act_id]
                dur = inst.activity_team_durations[act_id][team_id]
                e_time = s_time + dur
                proj_id = inst.activity_to_project[act_id]
                writer.writerow([step, act_id, proj_id, team_id, s_time, e_time, dur])

        print(f"     📝 CP decisions saved: {filepath.name}")
        return filepath

    # =================================================================
    # 유틸 함수: problem dict 텐서를 CPU로 이동
    # =================================================================
    def ensure_cpu_problem(problem):
        """problem dict의 모든 텐서를 CPU로 이동 (GA/MIP/CP용)"""
        cpu_problem = {}
        for k, v in problem.items():
            if isinstance(v, torch.Tensor):
                cpu_problem[k] = v.cpu()
            else:
                cpu_problem[k] = v
        return cpu_problem

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
            'state_mode': 'gat' if _model_type_from_ckpt in ('gat_v1', 'gat_v2', 'rgat') else ('topoformer' if _model_type_from_ckpt == 'topoformer' else 'daniel'),
            'allow_wait_release': ALLOW_WAIT_RELEASE,
            'allow_wait_mutex': ALLOW_WAIT_MUTEX,
            'allow_wait_team': ALLOW_WAIT_TEAM,
            'allow_wait_pred': ALLOW_WAIT_PRED,
            'use_mutex_attention': USE_MUTEX_ATTENTION,
        }
        _no_overwrite = ('batch_size', 'pomo_size', 'state_mode', 'debug_env',
                         'objective',
                         'allow_wait_release', 'allow_wait_mutex', 'allow_wait_team',
                         'allow_wait_pred',
                         'use_mutex_attention')
        for k, v in first_problem['env_params'].items():
            if k not in _no_overwrite:
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
                if isinstance(_ckpt_data, dict):
                    # model_type: 폴더 이름에서 감지 실패 시 checkpoint fallback
                    if _model_type_from_ckpt is None:
                        _model_type_from_ckpt = _ckpt_data.get('model_type')
                        if _model_type_from_ckpt is None:
                            _model_type_from_ckpt = _ckpt_data.get('trainer_params', {}).get('model_type')
                        if _model_type_from_ckpt is None and model_params is not None:
                            _model_type_from_ckpt = 'rgat' if 'd_hidden' in model_params else 'daniel'
                        if _model_type_from_ckpt is not None:
                            print(f"📌 checkpoint 파일에서 model_type fallback: {_model_type_from_ckpt}")
                    if _model_type_from_ckpt is not None:
                        trainer_params['model_type'] = _model_type_from_ckpt

                    # algorithm_type: 폴더 이름에서 감지 실패 시 checkpoint fallback
                    if ALGORITHM_TYPE is None:
                        _ckpt_alg = _ckpt_data.get('algorithm_type')
                        if _ckpt_alg is None:
                            _ckpt_alg = _ckpt_data.get('trainer_params', {}).get('algorithm_type')
                        if _ckpt_alg is not None:
                            ALGORITHM_TYPE = _ckpt_alg
                            trainer_params['algorithm_type'] = _ckpt_alg
                            print(f"📌 checkpoint 파일에서 algorithm_type fallback: {_ckpt_alg}")
                        else:
                            ALGORITHM_TYPE = 'reinforce'
                            trainer_params['algorithm_type'] = 'reinforce'
                            print(f"⚠️ algorithm_type 감지 실패, 기본값 사용: reinforce")
                del _ckpt_data

        # 필수값 검증
        _required_keys = ['N_P', 'N_T', 'N_A_min', 'N_A_max', 'duration_min', 'duration_max']
        _missing = [k for k in _required_keys if k not in _base_env]
        if _missing:
            print(f"❌ env_params 필수값 누락: {_missing}")
            exit(1)

        # 감지 결과 출력
        _src_model = "폴더" if _folder_model_type else "checkpoint"
        _src_alg = "폴더" if _folder_alg_type else ("checkpoint" if ALGORITHM_TYPE else "기본값")
        print(f"\n📋 RL 설정 감지 결과:")
        print(f"   MODEL_TYPE: {trainer_params['model_type']} ({_src_model}에서 감지)")
        print(f"   ALGORITHM_TYPE: {trainer_params['algorithm_type']} ({_src_alg}에서 감지)")

        print(f"\n📋 기본 env_params (pickle + checkpoint 기반):")
        print(f"   N_P={_base_env.get('N_P')}, N_T={_base_env.get('N_T')}, "
              f"N_A=[{_base_env.get('N_A_min')},{_base_env.get('N_A_max')}], "
              f"duration=[{_base_env.get('duration_min')},{_base_env.get('duration_max')}]")

        # 체크포인트 파일 목록
        if not ckpt_dir.exists():
            print(f"[WARNING] Checkpoint directory does not exist: {ckpt_dir}")
        elif TEST_ALL_CHECKPOINTS:
            # TEST_ALL_CHECKPOINTS가 True이면 폴더 내 모든 .pt 파일 로드
            all_pt_files = list(ckpt_dir.glob("*.pt"))
            # epoch*.pt 파일과 그 외 파일(best_model.pt 등)을 분리하여 정렬
            epoch_ckpts = []
            other_ckpts = []
            for p in all_pt_files:
                # epoch 번호 추출 시도
                stem = p.stem
                if stem.startswith('epoch') and stem[5:].isdigit():
                    epoch_ckpts.append((int(stem[5:]), p))
                else:
                    other_ckpts.append(p)
            epoch_ckpts.sort(key=lambda x: x[0])
            all_ckpts = [p for _, p in epoch_ckpts] + sorted(other_ckpts)
            if all_ckpts:
                print(f"[INFO] Found {len(all_ckpts)} checkpoints: {[p.name for p in all_ckpts]}")
            else:
                print(f"[WARNING] No .pt files found in {ckpt_dir}")
        elif CHECKPOINT_FILE is not None:
            ckpt_path = ckpt_dir / CHECKPOINT_FILE
            if ckpt_path.exists():
                all_ckpts = [ckpt_path]
                print(f"[INFO] Using specified checkpoint: {CHECKPOINT_FILE}")
            else:
                print(f"[WARNING] Specified checkpoint file not found: {ckpt_path}")
        else:
            ckpts = sorted(
                [p for p in ckpt_dir.glob("epoch*.pt") if p.stem[5:].isdigit()],
                key=lambda p: int(p.stem[5:])
            )
            if ckpts:
                all_ckpts = [ckpts[-1]]
                print(f"[INFO] Using latest checkpoint: {all_ckpts[0].name}")

    # =================================================================
    # 📦 pickle 파일별 실험 실행
    # =================================================================
    algorithm_summaries = {}
    saved_files = []

    # IL CP label 저장 경로 추적
    il_saved_labels = []

    for pickle_path in pickle_files:
        pickle_name = pickle_path.stem  # e.g., "test_batch"
        print(f"\n{'='*70}")
        print(f"📂 Processing: {pickle_path.name}")
        print(f"{'='*70}")

        with open(pickle_path, 'rb') as f:
            problem = pickle.load(f)

        # IL CP label 누적용 (pickle 파일별)
        il_cp_results = []

        total_instances = problem['num_activities'].shape[0]
        inst_start = INSTANCE_START if INSTANCE_START is not None else 0
        inst_end = INSTANCE_END if INSTANCE_END is not None else total_instances
        inst_start = max(0, min(inst_start, total_instances))
        inst_end = max(inst_start, min(inst_end, total_instances))
        num_instances = inst_end - inst_start
        print(f"   전체 인스턴스: {total_instances}, 실행 범위: [{inst_start}, {inst_end}) ({num_instances}개)")

        # 이 pickle에 맞는 env_params 구성 (RL은 전체 배치로 추론)
        _state_mode = 'gat' if _model_type_from_ckpt in ('gat_v1', 'gat_v2', 'rgat') else ('topoformer' if _model_type_from_ckpt == 'topoformer' else 'daniel')
        env_params = {
            'batch_size': total_instances,
            'pomo_size': 1,
            'objective': OBJECTIVE,
            'debug_env': DEBUG_ENV,
            'state_mode': _state_mode,
            'allow_wait_release': ALLOW_WAIT_RELEASE,
            'allow_wait_mutex': ALLOW_WAIT_MUTEX,
            'allow_wait_team': ALLOW_WAIT_TEAM,
            'allow_wait_pred': ALLOW_WAIT_PRED,
            'use_mutex_attention': USE_MUTEX_ATTENTION,
        }
        _no_overwrite = ('batch_size', 'pomo_size', 'state_mode', 'debug_env',
                         'objective',
                         'allow_wait_release', 'allow_wait_mutex', 'allow_wait_team',
                         'allow_wait_pred',
                         'use_mutex_attention')
        saved_env = problem['env_params']
        for k, v in saved_env.items():
            if k not in _no_overwrite:
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
            cpu_problem = ensure_cpu_problem(problem)

            for i in range(inst_start, inst_end):
                print(f"\n   📋 Instance {i}/{inst_end-1}")

                start_time = time.time()

                try:
                    projects = convert_problem_to_ga_format(cpu_problem, i, env_params['N_T'])

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
                                objective_type=OBJECTIVE,
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
                            ga_release_times = {proj.id: proj.release_time for proj in ga.projects}
                            ga_due_dates_map = {proj.id: proj.due_date for proj in ga.projects}
                            create_precedence_graph(
                                activity_predecessors=activity_predecessors,
                                activity_mutex=activity_mutex,
                                activity_to_project=ga.activity_to_project,
                                instance_name=f"{pickle_name}_instance_{i}",
                                activity_eligible_teams=act_eligible,
                                activity_processing_times=act_proc_times if act_proc_times else None,
                                project_release_times=ga_release_times,
                                project_due_dates=ga_due_dates_map,
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
                    cpu_problem = ensure_cpu_problem(problem)

                    for i in range(inst_start, inst_end):
                        print(f"\n   📋 Instance {i}/{inst_end-1}")
                        try:
                            inst = convert_problem_to_mip_format(cpu_problem, i)
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

                                        # Gurobi step 순서: start_time 오름차순 (IL replay와 동일 로직)
                                        sorted_acts_g = sorted(
                                            [a for a in range(inst.num_activities) if assigned_teams_sol.get(a) is not None],
                                            key=lambda a: (start_times_sol.get(a, float('inf')), a)
                                        )
                                        gurobi_step_order = {a: step for step, a in enumerate(sorted_acts_g)}

                                        project_due_dates = {p: inst.due_dates[p] for p in range(inst.num_projects)}
                                        create_gantt_chart_from_schedule(
                                            schedule=schedule,
                                            activity_to_project=inst.activity_to_project,
                                            num_teams=inst.num_teams,
                                            instance_name=f"{pickle_name}_instance_{i}",
                                            algorithm="Gurobi",
                                            objective_value=obj_val,
                                            project_due_dates=project_due_dates,
                                            activity_step_order=gurobi_step_order,
                                            objective_type=OBJECTIVE,
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
                                            project_release_times=inst.release_times,
                                            project_due_dates=inst.due_dates,
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
            from data_generator import convert_problem_to_mip_format
            from samsung_MIP import solve_rcmpsp_cp
            print(f"\n  --- CP-SAT ({pickle_name}) ---")
            cp_results = []
            il_success_count = 0
            cpu_problem = ensure_cpu_problem(problem)

            for i in range(inst_start, inst_end):
                print(f"\n   📋 Instance {i}/{inst_end-1}")
                try:
                    inst = convert_problem_to_mip_format(cpu_problem, i)
                    start_time = time.time()
                    result = solve_rcmpsp_cp(inst, CP_TIME_LIMIT, objective_type=OBJECTIVE)
                    runtime = time.time() - start_time

                    if result is not None:
                        obj_val, start_times_sol, assigned_teams_sol, status, lower_bound = result
                        gap = (obj_val - lower_bound) / max(obj_val, 1e-9) * 100 if obj_val > 0 else 0.0
                        print(f"     [CP-SAT][Instance {i}] "
                              f"{OBJECTIVE}: {obj_val:.4f}, "
                              f"LB: {lower_bound:.4f}, Gap: {gap:.2f}%, "
                              f"Status: {status}, "
                              f"Runtime: {runtime:.4f}s")

                        # CP 결정 시퀀스 저장
                        if SAVE_CP_DECISIONS:
                            try:
                                save_cp_decisions_to_csv(
                                    i, pickle_name, inst,
                                    start_times_sol, assigned_teams_sol,
                                    obj_val, status, session_results_dir
                                )
                            except Exception as e:
                                print(f"     ⚠️ CP 결정 저장 실패: {e}")

                        if SHOW_GANTT_CHART:
                            try:
                                schedule = {}
                                for act_id in range(inst.num_activities):
                                    t = assigned_teams_sol.get(act_id)
                                    if t is not None:
                                        s = start_times_sol[act_id]
                                        dur = inst.activity_team_durations[act_id][t]
                                        schedule[act_id] = (s, s + dur, t)

                                # CP step 순서: start_time 오름차순 (IL replay와 동일 로직)
                                sorted_acts = sorted(
                                    [a for a in range(inst.num_activities) if assigned_teams_sol.get(a) is not None],
                                    key=lambda a: (start_times_sol.get(a, float('inf')), a)
                                )
                                cp_step_order = {a: step for step, a in enumerate(sorted_acts)}

                                project_due_dates = {p: inst.due_dates[p] for p in range(inst.num_projects)}
                                create_gantt_chart_from_schedule(
                                    schedule=schedule,
                                    activity_to_project=inst.activity_to_project,
                                    num_teams=inst.num_teams,
                                    instance_name=f"{pickle_name}_instance_{i}",
                                    algorithm="CP-SAT",
                                    objective_value=obj_val,
                                    project_due_dates=project_due_dates,
                                    activity_step_order=cp_step_order,
                                    objective_type=OBJECTIVE,
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
                                    project_release_times=inst.release_times,
                                    project_due_dates=inst.due_dates,
                                    save_dir=gantt_dir,
                                    show=True
                                )
                                precedence_graphs_created.add(i)
                            except Exception as e:
                                print(f"     ⚠️ 선후관계 그래프 생성 실패: {e}")

                        # CP Solution 저장용 결과 누적
                        if SAVE_CP_SOLUTIONS:
                            il_cp_results.append({
                                'instance_idx': i,
                                'start_times': start_times_sol,
                                'assigned_teams': assigned_teams_sol,
                                'obj_val': obj_val,
                                'status': status,
                                'runtime': runtime,
                            })
                            il_success_count += 1
                            print(f"     [IL] CP schedule recorded")
                    else:
                        obj_val = None
                        lower_bound = None
                        status = "no_solution"
                        print(f"     [CP-SAT][Instance {i}] "
                              f"No solution found, Runtime: {runtime:.4f}s")

                    cp_results.append({
                        'instance': i,
                        'objective_value': obj_val,
                        'lower_bound': lower_bound if obj_val is not None else None,
                        'gap': gap if obj_val is not None and obj_val > 0 else None,
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
                        'lower_bound': None,
                        'gap': None,
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
                    valid_lbs = [r['lower_bound'] for r in cp_results
                                 if r['lower_bound'] is not None]
                    valid_gaps = [r['gap'] for r in cp_results
                                  if r['gap'] is not None]
                    lb_str = f", 평균 LB: {np.mean(valid_lbs):.4f}" if valid_lbs else ""
                    gap_str = f", 평균 Gap: {np.mean(valid_gaps):.2f}%" if valid_gaps else ""
                    print(f"  CP-SAT 평균 {OBJECTIVE}: {avg_obj:.4f}{lb_str}{gap_str} "
                          f"({solved}/{num_instances} solved)")

                output_path = save_mip_results_to_excel(
                    cp_results, pickle_name, "CP-SAT")
                if output_path:
                    saved_files.append(output_path)

            if SAVE_CP_SOLUTIONS:
                print(f"\n  CP solution 수집 ({pickle_name}): "
                      f"{il_success_count}/{inst_end-inst_start} instances")

        # =============================================
        # CP Solution 저장 (pickle 파일별)
        # =============================================
        if SAVE_CP_SOLUTIONS and il_cp_results:
            from il_utils import save_cp_solutions
            source_rel = str(pickle_path.relative_to(project_root))
            cpu_problem = ensure_cpu_problem(problem)
            solution_path = save_cp_solutions(
                cp_results=il_cp_results,
                problem=cpu_problem,
                env_params=env_params,
                source_pickle_path=source_rel,
                objective=OBJECTIVE,
                cp_time_limit=CP_TIME_LIMIT,
                save_dir=CP_SOLUTION_DIR,
            )
            il_saved_labels.append(solution_path)

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

                # checkpoint에서 model_type 갱신 (top-level → trainer_params → model_params 키로 추론)
                if isinstance(checkpoint, dict):
                    ckpt_model_type = checkpoint.get('model_type')
                    if ckpt_model_type is None:
                        ckpt_model_type = checkpoint.get('trainer_params', {}).get('model_type')
                    if ckpt_model_type is None and model_params is not None:
                        # model_params 키로 추론: GAT 계열은 'd_hidden' 키를 가짐
                        ckpt_model_type = 'rgat' if 'd_hidden' in model_params else 'daniel'
                    if ckpt_model_type is not None:
                        trainer_params['model_type'] = ckpt_model_type
                        _ckpt_state_mode = 'gat' if ckpt_model_type in ('gat_v1', 'gat_v2', 'rgat') else ('topoformer' if ckpt_model_type == 'topoformer' else 'daniel')
                        env_params['state_mode'] = _ckpt_state_mode

                # model_params의 edge_feat_dim을 env_params에 전달 (checkpoint 호환)
                if model_params and 'edge_feat_dim' in model_params:
                    env_params['edge_feat_dim'] = model_params['edge_feat_dim']

                trainer = Scheduling_Trainer(env_params, model_params, optimizer_params, trainer_params)

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    load_result = trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    if load_result.unexpected_keys:
                        print(f"  ⚠️ 무시된 키: {load_result.unexpected_keys}")
                    if load_result.missing_keys:
                        print(f"  ⚠️ 누락된 키: {load_result.missing_keys}")
                    print(f"  ✅ 모델 로드: {checkpoint_name}")
                    saved_alg = checkpoint.get('algorithm_type',
                                checkpoint.get('trainer_params', {}).get('algorithm_type', 'reinforce'))
                    saved_model_type = checkpoint.get('model_type') or checkpoint.get('trainer_params', {}).get('model_type', 'daniel')
                    print(f"     └─ Algorithm: {str(saved_alg).upper()}, Model: {saved_model_type.upper()}")
                else:
                    load_result = trainer.model.load_state_dict(checkpoint, strict=False)
                    if load_result.unexpected_keys:
                        print(f"  ⚠️ 무시된 키: {load_result.unexpected_keys}")
                    if load_result.missing_keys:
                        print(f"  ⚠️ 누락된 키: {load_result.missing_keys}")
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
                    env_device = device if DEVICE_MODE == 'gpu' else 'cpu'

                    # ----- Greedy Rollout -----
                    test_env = SchedulingEnv(env_params, debug_env=False, device=env_device)
                    test_env._reset(problem)

                    done = False
                    s = test_env._get_state()
                    _step_count = 0

                    while not done:
                        _step_count += 1
                        pi, v = trainer._model_forward(s)

                        action_flat = torch.argmax(pi, dim=1)
                        N_T = test_env.N_T
                        act_idx  = action_flat // N_T
                        team_idx = action_flat % N_T

                        s, obj_value, done = test_env.step_pair(
                            act_idx.to(test_env.device),
                            team_idx.to(test_env.device)
                        )

                    greedy_scores = test_env._get_obj_original_scale()  # (total_instances,) 원본 스케일
                    greedy_end_time = time.time()

                    # ----- Sampling Rollout (배치 병렬화 + mini batch) -----
                    if RL_SAMPLING_K > 0:
                        import copy, math
                        K = RL_SAMPLING_K
                        MB = RL_SAMPLING_BATCH
                        n_mini = math.ceil(K / MB)
                        print(f"  📋 Sampling rollout 시작 (K={K}, mini_batch={MB}, {n_mini}회)")

                        best_sampling_scores = None  # (total_instances,) 인스턴스별 best
                        sample_idx = 0  # 전체 sample 번호 카운터

                        for mb_idx in range(n_mini):
                            mb_k = min(MB, K - mb_idx * MB)  # 이번 mini batch의 sample 수
                            mb_total = total_instances * mb_k
                            print(f"     [Mini batch {mb_idx+1}/{n_mini}] {mb_k} samples (batch={mb_total})")

                            # problem 텐서를 mb_k배 repeat_interleave
                            sampling_problem = {}
                            for pk, pv in problem.items():
                                if isinstance(pv, torch.Tensor) and pv.size(0) == total_instances:
                                    sampling_problem[pk] = pv.repeat_interleave(mb_k, dim=0)
                                elif pk == 'batch_projects':
                                    sampling_problem[pk] = [proj for proj in pv for _ in range(mb_k)]
                                elif pk == 'batch_activities':
                                    sampling_problem[pk] = [acts for acts in pv for _ in range(mb_k)]
                                elif pk == 'env_params':
                                    sampling_problem[pk] = copy.deepcopy(pv)
                                else:
                                    sampling_problem[pk] = pv

                            sampling_env_params = dict(env_params)
                            sampling_env_params['batch_size'] = mb_total
                            sampling_problem['env_params']['batch_size'] = mb_total

                            sample_env = SchedulingEnv(sampling_env_params, debug_env=False, device=env_device)
                            sample_env._reset(sampling_problem)

                            done = False
                            s = sample_env._get_state()

                            while not done:
                                pi, v = trainer._model_forward(s)
                                dist = Categorical(pi)
                                action_flat = dist.sample()

                                N_T = sample_env.N_T
                                act_idx  = action_flat // N_T
                                team_idx = action_flat % N_T

                                s, obj_value, done = sample_env.step_pair(
                                    act_idx.to(sample_env.device),
                                    team_idx.to(sample_env.device)
                                )

                            # (total_instances * mb_k,) → (total_instances, mb_k)
                            mb_scores = sample_env._get_obj_original_scale()
                            mb_scores = mb_scores.reshape(total_instances, mb_k)

                            for k_idx in range(mb_k):
                                sample_idx += 1
                                avg_k = mb_scores[inst_start:inst_end, k_idx].mean().item()
                                print(f"       [Sample {sample_idx}/{K}] avg {OBJECTIVE}: {avg_k:.4f}")

                            # 이번 mini batch의 best
                            mb_best = mb_scores.min(dim=1).values  # (total_instances,)
                            if best_sampling_scores is None:
                                best_sampling_scores = mb_best
                            else:
                                best_sampling_scores = torch.min(best_sampling_scores, mb_best)

                        # Sampling best vs Greedy 비교
                        sampling_avg = best_sampling_scores[inst_start:inst_end].mean().item()
                        greedy_avg = greedy_scores[inst_start:inst_end].mean().item()
                        print(f"     Sampling best avg: {sampling_avg:.4f} | Greedy avg: {greedy_avg:.4f}")

                end_time = time.time()
                greedy_runtime = greedy_end_time - start_time
                greedy_avg_runtime = greedy_runtime / total_instances

                # ----- Greedy 결과 저장 -----
                rl_greedy_label = 'RL_greedy' if RL_SAMPLING_K > 0 else 'RL'
                rl_results = []
                for i in range(inst_start, inst_end):
                    score_i = greedy_scores[i].item()
                    rl_results.append({
                        'algorithm': rl_greedy_label,
                        'instance': i,
                        'objective_value': score_i,
                        'runtime': greedy_avg_runtime,
                    })
                    print(f"     [{rl_greedy_label}][Instance {i}] {OBJECTIVE}: {score_i:.4f}")

                print(f"\n     Greedy Total: {greedy_runtime:.4f}s (avg {greedy_avg_runtime:.4f}s/instance)")

                # ----- Sampling 결과 저장 -----
                rl_sampling_results = []
                if RL_SAMPLING_K > 0:
                    sampling_runtime = end_time - greedy_end_time
                    avg_sampling_runtime = sampling_runtime / total_instances
                    for i in range(inst_start, inst_end):
                        score_i = best_sampling_scores[i].item()
                        rl_sampling_results.append({
                            'algorithm': f'RL_sampling(K={RL_SAMPLING_K})',
                            'instance': i,
                            'objective_value': score_i,
                            'runtime': avg_sampling_runtime,
                        })
                        print(f"     [RL_sampling(K={RL_SAMPLING_K})][Instance {i}] {OBJECTIVE}: {score_i:.4f}")

                    print(f"\n     Sampling Total: {sampling_runtime:.4f}s (avg {avg_sampling_runtime:.4f}s/instance)")

                if SHOW_GANTT_CHART or SHOW_PRECEDENCE_GRAPH:
                    for i in range(inst_start, inst_end):
                        try:
                            num_act = test_env.num_activities[i].item()

                            # env에서 schedule 추출 (원본 스케일로 복원)
                            ts = test_env.time_scale[i, 0].item()  # time_scale 복원 계수
                            schedule = {}
                            act_to_proj = {}
                            for act_id in range(num_act):
                                if test_env.activity_reserved[i, act_id].item():
                                    s = test_env.activity_start_time[i, act_id].item() * ts
                                    e = test_env.activity_end_time[i, act_id].item() * ts
                                    t = test_env.activity_assigned_team[i, act_id].item()
                                    schedule[act_id] = (s, e, t)
                                act_to_proj[act_id] = test_env.activity_project[i, act_id].item()

                            if SHOW_GANTT_CHART:
                                proj_due = {}
                                for p in range(test_env.N_P):
                                    proj_due[p] = test_env.project_due_date[i, p].item() * ts

                                # RL step 순서 추출
                                step_order = {}
                                for act_id in range(num_act):
                                    s = test_env.activity_scheduled_step[i, act_id].item()
                                    if s >= 0:
                                        step_order[act_id] = s

                                create_gantt_chart_from_schedule(
                                    schedule=schedule,
                                    activity_to_project=act_to_proj,
                                    num_teams=test_env.N_T,
                                    instance_name=f"{pickle_name}_instance_{i}",
                                    algorithm="RL",
                                    objective_value=greedy_scores[i].item(),
                                    project_due_dates=proj_due,
                                    activity_step_order=step_order,
                                    objective_type=OBJECTIVE,
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
                                rl_release_times = {}
                                rl_due_dates = {}
                                for p in range(test_env.N_P):
                                    rl_release_times[p] = test_env.project_release_time[i, p].item() * ts
                                    rl_due_dates[p] = test_env.project_due_date[i, p].item() * ts
                                create_precedence_graph(
                                    activity_predecessors=activity_predecessors,
                                    activity_mutex=activity_mutex,
                                    activity_to_project=act_to_proj,
                                    instance_name=f"{pickle_name}_instance_{i}",
                                    activity_eligible_teams=rl_eligible,
                                    activity_processing_times=rl_proc_times,
                                    project_release_times=rl_release_times,
                                    project_due_dates=rl_due_dates,
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
                rl_sampling_results = []

            if rl_results:
                valid_objectives = [r['objective_value'] for r in rl_results if r['objective_value'] is not None]
                if valid_objectives:
                    avg_objective = np.mean(valid_objectives)
                    algorithm_summaries[f'{rl_greedy_label}_{checkpoint_name} ({pickle_name})'] = avg_objective
                    print(f"  {rl_greedy_label} 평균 {OBJECTIVE}: {avg_objective:.4f}")

                output_path = save_rl_results_to_excel(rl_results, checkpoint_name, pickle_name)
                if output_path:
                    saved_files.append(output_path)

            if rl_sampling_results:
                valid_objectives = [r['objective_value'] for r in rl_sampling_results if r['objective_value'] is not None]
                if valid_objectives:
                    avg_objective = np.mean(valid_objectives)
                    algorithm_summaries[f'RL_sampling(K={RL_SAMPLING_K})_{checkpoint_name} ({pickle_name})'] = avg_objective
                    print(f"  RL_sampling(K={RL_SAMPLING_K}) 평균 {OBJECTIVE}: {avg_objective:.4f}")

                output_path = save_rl_results_to_excel(rl_sampling_results, f"{checkpoint_name}_sampling_K{RL_SAMPLING_K}", pickle_name)
                if output_path:
                    saved_files.append(output_path)

    # =================================================================
    # CP Solution 저장 요약
    # =================================================================
    if SAVE_CP_SOLUTIONS and il_saved_labels:
        print(f"\n{'='*60}")
        print(f"CP Solutions Saved")
        print(f"{'='*60}")
        for lp in il_saved_labels:
            print(f"  - {lp}")
        print(f"  (각 파일에 원본 start_times + active_start_times 둘 다 포함)")
        print(f"\n  다음 단계:")
        print(f"    - cp_replay.py를 실행하여 CP solution 검증")
        print(f"    - il_trajectory_generation.py를 실행하여 MDP trajectory 생성")
    elif SAVE_CP_SOLUTIONS:
        print(f"\nCP solution이 수집되지 않았습니다. (CP 해를 찾은 인스턴스 없음)")

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
