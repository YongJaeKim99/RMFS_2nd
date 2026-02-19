from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import time
import pickle
import random

from trainer import Scheduling_Trainer
from scheduling_env import SchedulingEnv
from GA import GeneticAlgorithm, Activity, Project
from gantt_chart import create_gantt_chart_from_env, create_gantt_chart_from_ga_solution, create_precedence_graph

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
    # 체크포인트 폴더 이름 (checkpoints/ 아래의 폴더명)
    # 예: "20260210_130751_tardiness_REINFORCE"
    #CHECKPOINT_FOLDER = "20260210_175204_tardiness_REINFORCE"
    CHECKPOINT_FOLDER = "20260219_151324_tardiness_DANIEL_PPO"
    
    # 특정 체크포인트 파일 이름 (None이면 자동 선택)
    # 예: "scheduling_epoch100.pt" 또는 None
    CHECKPOINT_FILE = "epoch0.pt"
    
    # 모든 체크포인트에 대해 실험할지 여부
    # True: 폴더 내 모든 체크포인트 테스트
    # False: 가장 최신 체크포인트만 테스트 (또는 CHECKPOINT_FILE 지정 시 해당 파일만)
    TEST_ALL_CHECKPOINTS = False
    
    # -----------------------------
    # 2-1) 학습 알고리즘 설정 (체크포인트와 일치해야 함)
    # -----------------------------
    ALGORITHM_TYPE = 'ppo'  # 'reinforce' or 'ppo'
    # ※ 학습 시 사용한 ALGORITHM_TYPE과 동일하게 설정하세요.
    #   추론(inference) 자체는 두 알고리즘 모두 동일하게 동작합니다.

    # -----------------------------
    # 3) 테스트할 알고리즘 설정
    # -----------------------------
    # RL: 학습된 GNN 모델 사용
    # GA: 유전 알고리즘 (GA.py)
    test_algorithms = ["RL"]  # ["RL"], ["GA"], 또는 ["RL", "GA"]
    
    # GA 설정
    GA_POPULATION_SIZE = 50
    GA_GENERATIONS = 1000
    GA_DECODE_MODE = "immediate"  # "batch" or "immediate"
    GA_CROSSOVER_RATE = 0.8  # Uniform Crossover 적용 확률
    GA_MUTATION_RATE = 0.2   # Mutation 적용 확률 (페어의 10%에 새 random key 할당)
    #GA_RELEASE_MODE = "wait"   # "wait": release time까지 기다림, "skip": 아직이면 건너뜀
    GA_RELEASE_MODE = "wait"
    #GA_MUTEX_MODE = "wait"     # "wait": mutex 끝날 때까지 기다림, "skip": 진행 중이면 건너뜀
    GA_MUTEX_MODE = "wait"
    GA_VERBOSE = False  # True: 세대별 진행상황 출력, False: 최종 결과만
    GA_DOMINANCE_RULE = True  # True: wait으로 인한 idle time에 즉시 수행 가능 activity가 있으면 해당 pair skip
    #   ※ release_mode="wait" 또는 mutex_mode="wait" 일 때만 유효 (RL의 DOMINANCE_RULE 대응)
    
    # -----------------------------
    # 간트차트 생성 설정
    # -----------------------------
    #SAVE_GANTT_CHART = False  # True: 간트차트 생성, False: 생성 안 함
    #SHOW_GANTT_CHART = False  # True: 브라우저에서 표시, False: 저장만
    SAVE_GANTT_CHART = False
    SHOW_GANTT_CHART = False
    
    # =================================================================
    # 🎯 목적함수 선택
    # =================================================================
    OBJECTIVE = 'tardiness'  # 'tardiness' or 'makespan'
    
    # =================================================================
    # 📊 테스트 데이터 설정
    # =================================================================
    # data/test/ 폴더의 0.pickle, 1.pickle, ... 파일 사용
    TEST_FILE_START = 0
    TEST_FILE_END = 49
    GA_REPEATS = 3  # GA를 인스턴스당 반복 실행 횟수 (best objective 채택)
    
    # -----------------------------
    # 3) 기타 설정
    # -----------------------------
    # Wait / Dominance 옵션
    ALLOW_WAIT_RELEASE = True
    ALLOW_WAIT_MUTEX = True
    DOMINANCE_RULE = True

    # 디버그 모드 설정
    DEBUG_ENV = False
    DEBUG_MODEL = False
    
    # 시드 고정
    SEED = 0
    
    # 시드 고정 (재현성 보장)
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✅ Random seed fixed to: {SEED}")
    
    print("\n" + "="*50)
    print("🎯 테스트 설정")
    print("="*50)
    print(f"  📌 OBJECTIVE: {OBJECTIVE}")
    print(f"  📌 TEST_FILE_RANGE: {TEST_FILE_START}~{TEST_FILE_END}")
    print(f"  📌 ALGORITHMS: {', '.join(test_algorithms)}")
    if "GA" in test_algorithms:
        print(f"  📌 GA Settings:")
        print(f"     - Population: {GA_POPULATION_SIZE}")
        print(f"     - Generations: {GA_GENERATIONS}")
        print(f"     - Decode Mode: {GA_DECODE_MODE}")
        print(f"     - Release Mode: {GA_RELEASE_MODE}")
        print(f"     - Mutex Mode: {GA_MUTEX_MODE}")
        print(f"     - Dominance Rule: {GA_DOMINANCE_RULE}")
        print(f"     - Repeats/Instance: {GA_REPEATS}")
    print(f"  📌 SAVE_GANTT_CHART: {SAVE_GANTT_CHART}")
    if SAVE_GANTT_CHART:
        print(f"     - Show in Browser: {SHOW_GANTT_CHART}")
    print("="*50 + "\n")
    
    # =================================================================
    
    # -----------------------------
    # 4) 파라미터 설정
    # -----------------------------
    # env_params: 옵션만 설정. 구체적인 문제 파라미터(N_P, N_T, duration_max 등)는
    # 섹션 6에서 pickle/checkpoint로부터 자동 로드됩니다.
    env_params = {
        'batch_size': 1,
        'pomo_size': 1,
        'objective': OBJECTIVE,
        'debug_env': DEBUG_ENV,
        'state_mode': 'daniel',
        'allow_wait_release': ALLOW_WAIT_RELEASE,
        'allow_wait_mutex': ALLOW_WAIT_MUTEX,
        'dominance_rule': DOMINANCE_RULE,
    }

    # model_params: RL 테스트 시 checkpoint에서 자동 로드 (섹션 6.5)
    # GA 전용 테스트 시에는 불필요
    model_params = None

    # 옵티마이저 파라미터 설정 (테스트 시 학습은 안 하지만 Trainer 초기화에 필요)
    optimizer_params = {
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 1e-6,
        }
    }
    
    # 트레이너 파라미터 설정
    trainer_params = {
        'epochs': 0,  # 테스트 시 사용 안 함
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
    
    # 체크포인트 폴더 생성
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    
    # -----------------------------
    # 5) 유틸 함수: 결과를 엑셀로 저장
    # -----------------------------
    current_dir = Path(__file__).parent
    results_dir = current_dir / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_results_dir = results_dir / timestamp
    
    if not session_results_dir.exists():
        session_results_dir.mkdir(parents=True)
        print(f"✅ 결과 저장 폴더 생성: {session_results_dir}")
    
    # 간트차트 저장 폴더 생성
    gantt_dir = None
    if SAVE_GANTT_CHART:
        gantt_dir = session_results_dir / "gantt_charts"
        gantt_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 간트차트 저장 폴더 생성: {gantt_dir}")
    
    def save_results_to_excel(all_results, algorithm, checkpoint_name):
        """결과를 엑셀 파일로 저장"""
        if algorithm == "RL":
            filename = f'RCMPSP_{algorithm}_{checkpoint_name}.xlsx'
        else:
            filename = f'RCMPSP_{algorithm}.xlsx'
        
        output_path = session_results_dir / filename
        
        # Experiment info sheet
        exp_rows = []
        exp_rows.append({'Item': 'Algorithm', 'Value': algorithm})
        if algorithm == "RL":
            exp_rows.append({'Item': 'Checkpoint Used', 'Value': checkpoint_name})
        elif algorithm == "GA":
            exp_rows.append({'Item': 'Population Size', 'Value': GA_POPULATION_SIZE})
            exp_rows.append({'Item': 'Generations', 'Value': GA_GENERATIONS})
            exp_rows.append({'Item': 'Decode Mode', 'Value': GA_DECODE_MODE})
            exp_rows.append({'Item': 'Release Mode', 'Value': GA_RELEASE_MODE})
            exp_rows.append({'Item': 'Mutex Mode', 'Value': GA_MUTEX_MODE})
            exp_rows.append({'Item': 'Repeats/Instance', 'Value': GA_REPEATS})
        exp_rows.append({'Item': 'Device', 'Value': 'CPU'})
        exp_rows.append({'Item': 'Test Files', 'Value': f'{TEST_FILE_START} to {TEST_FILE_END}'})
        if SEED is not None:
            exp_rows.append({'Item': 'Random Seed', 'Value': SEED})
        experiment_info_df = pd.DataFrame(exp_rows, columns=['Item', 'Value'])
        
        if not all_results:
            print(f"결과 데이터가 없습니다: {algorithm}")
            return None
            
        # 상세 결과 데이터
        detailed_data = []
        for result in all_results:
            row = {
                'Algorithm': result['algorithm'],
                'Instance': result['instance'],
                OBJECTIVE.capitalize(): result['objective_value'],
                'Runtime': result.get('runtime', 0)
            }
            detailed_data.append(row)
        
        # 전체 평균 성능
        valid_objectives = [r['objective_value'] for r in all_results if r['objective_value'] is not None]
        overall_row = {
            'Algorithm': all_results[0]['algorithm'] if all_results else algorithm,
            'Instance': 'Average',
            OBJECTIVE.capitalize(): np.mean(valid_objectives) if valid_objectives else None,
            'Runtime': np.mean([r.get('runtime', 0) for r in all_results])
        }
        overall_average_data = [overall_row]
        
        # 데이터프레임 생성 및 저장
        detailed_df = pd.DataFrame(detailed_data)
        overall_avg_df = pd.DataFrame(overall_average_data)

        for df in [detailed_df, overall_avg_df]:
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].round(4)
        
        # 엑셀 파일 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            experiment_info_df.to_excel(writer, sheet_name='Experiment_Info', index=False)
            detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            overall_avg_df.to_excel(writer, sheet_name='Overall_Average', index=False)
        
        print(f"결과가 {output_path}에 저장되었습니다.")
        return output_path

    def save_ga_results_to_excel(ga_results, ga_all_repeats):
        """GA 결과를 엑셀 파일로 저장 (반복 실행 전체 포함)"""
        output_path = session_results_dir / 'RCMPSP_GA.xlsx'

        # Experiment info
        exp_rows = [
            {'Item': 'Algorithm',        'Value': 'GA'},
            {'Item': 'Population Size',  'Value': GA_POPULATION_SIZE},
            {'Item': 'Generations',      'Value': GA_GENERATIONS},
            {'Item': 'Decode Mode',      'Value': GA_DECODE_MODE},
            {'Item': 'Release Mode',     'Value': GA_RELEASE_MODE},
            {'Item': 'Mutex Mode',       'Value': GA_MUTEX_MODE},
            {'Item': 'Dominance Rule',   'Value': GA_DOMINANCE_RULE},
            {'Item': 'Repeats/Instance', 'Value': GA_REPEATS},
            {'Item': 'Device',           'Value': 'CPU'},
            {'Item': 'Test Files',       'Value': f'{TEST_FILE_START} to {TEST_FILE_END}'},
        ]
        if SEED is not None:
            exp_rows.append({'Item': 'Random Seed', 'Value': SEED})
        experiment_info_df = pd.DataFrame(exp_rows, columns=['Item', 'Value'])

        if not ga_results:
            print("결과 데이터가 없습니다: GA")
            return None

        # Detailed_Results: 반복 실행 전체 (instance × repeat)
        detailed_data = []
        for r in ga_all_repeats:
            detailed_data.append({
                'Instance': r['instance'],
                'Repeat':   r['repeat'],
                OBJECTIVE.capitalize(): r['objective_value'],
                'Runtime':  r['runtime'],
            })
        detailed_df = pd.DataFrame(detailed_data)

        # Instance_Summary: 인스턴스별 best / avg
        instance_summary_data = []
        for r in ga_results:
            instance_summary_data.append({
                'Instance':                       r['instance'],
                f'Best {OBJECTIVE.capitalize()}': r['best_objective'],
                f'Avg {OBJECTIVE.capitalize()}':  r['avg_objective'],
                'Total Runtime':                  r['runtime'],
            })
        instance_summary_df = pd.DataFrame(instance_summary_data)

        # Overall_Average: 전체 평균
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

    # -----------------------------
    # 6) 데이터 경로 설정
    # -----------------------------
    data_base_dir = project_root / "data" / "test"

    print(f"📂 데이터 경로: {data_base_dir}")

    if not data_base_dir.exists():
        print(f"⚠️  경고: 데이터 폴더가 존재하지 않습니다: {data_base_dir}")
        print(f"   테스트 데이터를 먼저 생성해주세요.")
        exit(1)
    # 첫 번째 파일에서 env_params 로드
    first_file = data_base_dir / f"{TEST_FILE_START}.pickle"
    if first_file.exists():
        with open(first_file, 'rb') as f:
            first_problem = pickle.load(f)
        saved_env = first_problem['env_params']
        for k, v in saved_env.items():
            if k not in ('batch_size', 'pomo_size', 'state_mode', 'debug_env'):
                env_params[k] = v
        print(f"📄 Test 데이터 env_params 적용 (from {TEST_FILE_START}.pickle)")
    print(f"📄 데이터 파일 범위: {TEST_FILE_START}~{TEST_FILE_END}")
    
    # -----------------------------
    # 6.5) 체크포인트 env_params로 누락값 보충
    # -----------------------------
    # 기존 pickle에는 duration_max 등 feature 정규화에 필요한 값이 누락될 수 있음.
    # 체크포인트에 저장된 학습 시 env_params로 보충.
    ckpt_dir = project_root / "checkpoints" / CHECKPOINT_FOLDER
    if "RL" in test_algorithms and ckpt_dir.exists():
        _ckpt_for_params = ckpt_dir / (CHECKPOINT_FILE or "epoch0.pt")
        if _ckpt_for_params.exists():
            _ckpt_data = torch.load(_ckpt_for_params, map_location='cpu', weights_only=False)
            if isinstance(_ckpt_data, dict) and 'env_params' in _ckpt_data:
                _ckpt_env = _ckpt_data['env_params']
                _filled = []
                for k, v in _ckpt_env.items():
                    if k not in ('batch_size', 'pomo_size', 'state_mode', 'debug_env'):
                        if k not in env_params:
                            env_params[k] = v
                            _filled.append(f"{k}={v}")
                        elif env_params[k] != v and k in ('duration_min', 'duration_max'):
                            # 피처 정규화에 직접 영향을 주는 값은 checkpoint 기준으로 보정
                            print(f"⚠️  {k} 불일치! pickle={env_params[k]}, checkpoint={v} → checkpoint 값 사용")
                            env_params[k] = v
                if _filled:
                    print(f"📌 checkpoint env_params에서 누락값 보충: {', '.join(_filled)}")
                # model_params도 checkpoint에서 로드
                if 'model_params' in _ckpt_data:
                    model_params = _ckpt_data['model_params']
                    print(f"📌 checkpoint에서 model_params 로드 완료")
            del _ckpt_data

    # env_params 필수값 검증
    _required_keys = ['N_P', 'N_T', 'N_A_min', 'N_A_max', 'duration_min', 'duration_max']
    _missing = [k for k in _required_keys if k not in env_params]
    if _missing and "RL" in test_algorithms:
        print(f"❌ env_params 필수값 누락: {_missing}")
        print(f"   pickle 또는 checkpoint에 해당 값이 없습니다. 학습을 다시 실행하거나 data를 재생성하세요.")
        exit(1)

    # 최종 env_params 출력
    print(f"\n📋 최종 env_params (pickle + checkpoint 기반):")
    print(f"   N_P={env_params.get('N_P')}, N_T={env_params.get('N_T')}, "
          f"N_A=[{env_params.get('N_A_min')},{env_params.get('N_A_max')}], "
          f"duration=[{env_params.get('duration_min')},{env_params.get('duration_max')}]")
    print(f"   allow_wait_release={env_params.get('allow_wait_release')}, "
          f"allow_wait_mutex={env_params.get('allow_wait_mutex')}, "
          f"dominance_rule={env_params.get('dominance_rule')}")

    # -----------------------------
    # 7) 체크포인트 로드 (RL 테스트가 있을 때만)
    # -----------------------------
    all_ckpts = []
    if "RL" in test_algorithms:
        if not ckpt_dir.exists():
            print(f"[WARNING] Checkpoint directory does not exist: {ckpt_dir}")
            print(f"[INFO] RL 테스트를 건너뜁니다.")
        elif CHECKPOINT_FILE is not None:
            # 특정 체크포인트 파일 지정
            ckpt_path = ckpt_dir / CHECKPOINT_FILE
            if ckpt_path.exists():
                all_ckpts = [ckpt_path]
                print(f"[INFO] Using specified checkpoint: {CHECKPOINT_FILE}")
            else:
                print(f"[WARNING] Specified checkpoint file not found: {ckpt_path}")
                print(f"[INFO] RL 테스트를 건너뜁니다.")
        elif TEST_ALL_CHECKPOINTS:
            # 모든 체크포인트 테스트
            all_ckpts = sorted(
                ckpt_dir.glob("scheduling_epoch*.pt"),
                key=lambda p: int(p.stem.replace('scheduling_epoch', ''))
            )
            if not all_ckpts:
                print(f"[WARNING] No checkpoints found in {ckpt_dir}")
                print(f"[INFO] RL 테스트를 건너뜁니다.")
            else:
                print(f"[INFO] Found {len(all_ckpts)} checkpoints: {[ckpt.name for ckpt in all_ckpts]}")
        else:
            # 가장 최신 체크포인트만 테스트
            ckpts = sorted(
                ckpt_dir.glob("scheduling_epoch*.pt"),
                key=lambda p: int(p.stem.replace('scheduling_epoch', ''))
            )
            if ckpts:
                all_ckpts = [ckpts[-1]]  # 가장 최신 체크포인트만
                print(f"[INFO] Using latest checkpoint: {all_ckpts[0].name}")
            else:
                print(f"[WARNING] No checkpoints found in {ckpt_dir}")
                print(f"[INFO] RL 테스트를 건너뜁니다.")
    
    # -----------------------------
    # 8) 알고리즘별 실험 실행
    # -----------------------------
    algorithm_summaries = {}
    saved_files = []
    
    # 선후관계 그래프 생성 추적 (인스턴스당 한 번만)
    precedence_graphs_created = set()
    
    # GA 실행 (RL과 독립적)
    if "GA" in test_algorithms:
        print(f"\n{'='*60}")
        print(f"Testing Algorithm: GA")
        print(f"{'='*60}")
        
        ga_results = []
        ga_all_repeats = []

        from data_generator import convert_problem_to_ga_format

        for i in range(TEST_FILE_START, TEST_FILE_END + 1):
                data_path = data_base_dir / f"{i}.pickle"
                if not data_path.exists():
                    print(f"⚠️  Warning: Data file not found: {data_path}")
                    continue
                print(f"\n📋 테스트 파일 {i}.pickle")
                with open(data_path, 'rb') as fr:
                    problem = pickle.load(fr)
                batch_idx = 0

                start_time = time.time()

                try:
                    # pickle 데이터를 GA 형식으로 변환
                    projects = convert_problem_to_ga_format(problem, batch_idx, env_params['N_T'])

                    # GA 반복 실행 (전체 결과 수집, best 채택)
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

                    ga = best_ga  # 간트차트 등에서 사용

                    end_time = time.time()
                    runtime = end_time - start_time

                    avg_objective = np.mean(rep_objectives)
                    result = {
                        'algorithm':      'GA',
                        'instance':       i,
                        'best_objective': best_objective,
                        'avg_objective':  avg_objective,
                        'runtime':        runtime
                    }
                    ga_results.append(result)

                    if GA_REPEATS > 1:
                        rep_str = f" (best of {GA_REPEATS} runs, avg: {avg_objective:.4f})"
                    else:
                        rep_str = ""
                    print(f"   [GA][Instance {i}] {OBJECTIVE}: best={best_objective:.4f}, Runtime: {runtime:.4f}s{rep_str}")
                    
                    # 간트차트 및 선후관계 그래프 생성
                    if SAVE_GANTT_CHART:
                        try:
                            # project_due_dates 추출
                            project_due_dates = {proj.id: proj.due_date for proj in ga.projects}
                            
                            # 간트차트 생성 (알고리즘별)
                            create_gantt_chart_from_ga_solution(
                                solution=best_solution,
                                activity_to_project=ga.activity_to_project,
                                num_teams=env_params['N_T'],
                                instance_name=f"instance_{i}",
                                objective_value=best_objective,
                                project_due_dates=project_due_dates,
                                save_dir=gantt_dir,
                                show=SHOW_GANTT_CHART
                            )
                            
                            # 선후관계 그래프 생성 (인스턴스당 한 번만)
                            if i not in precedence_graphs_created:
                                activity_predecessors = {}
                                activity_mutex = {}
                                for act in ga.activities:
                                    activity_predecessors[act.id] = act.predecessors
                                    activity_mutex[act.id] = act.mutually_exclusive
                                
                                create_precedence_graph(
                                    activity_predecessors=activity_predecessors,
                                    activity_mutex=activity_mutex,
                                    activity_to_project=ga.activity_to_project,
                                    instance_name=f"instance_{i}",
                                    save_dir=gantt_dir,
                                    show=SHOW_GANTT_CHART
                                )
                                precedence_graphs_created.add(i)
                        except Exception as e:
                            print(f"   ⚠️ 간트차트/그래프 생성 실패: {e}")
                
                except Exception as e:
                    print(f"   ❌ [GA][Instance {i}] 실행 중 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # GA 결과 저장
        if ga_results:
            best_objs = [r['best_objective'] for r in ga_results if r['best_objective'] is not None]
            avg_objs  = [r['avg_objective']  for r in ga_results if r['avg_objective']  is not None]
            if best_objs:
                avg_of_best = np.mean(best_objs)
                avg_of_avg  = np.mean(avg_objs) if avg_objs else None
                algorithm_summaries['GA (best)'] = avg_of_best
                if avg_of_avg is not None:
                    algorithm_summaries['GA (avg)'] = avg_of_avg
                print(f"  GA 평균 {OBJECTIVE}: best={avg_of_best:.4f}" +
                      (f", avg={avg_of_avg:.4f}" if avg_of_avg is not None else ""))

            output_path = save_ga_results_to_excel(ga_results, ga_all_repeats)
            if output_path:
                saved_files.append(output_path)
    
    # RL 실행 (체크포인트별)
    for ckpt_path in all_ckpts:
        checkpoint_name = ckpt_path.stem
        print(f"\n{'='*60}")
        print(f"Testing Checkpoint: {checkpoint_name}")
        print(f"{'='*60}")

        # 체크포인트 로드 및 Trainer 초기화
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

            # model_params를 checkpoint에서 로드 (아직 없으면)
            if model_params is None:
                if isinstance(checkpoint, dict) and 'model_params' in checkpoint:
                    model_params = checkpoint['model_params']
                    print(f"📌 checkpoint에서 model_params 로드 완료")
                else:
                    print(f"❌ model_params를 찾을 수 없습니다 (checkpoint에 없음)")
                    continue

            # Trainer 초기화
            trainer = Scheduling_Trainer(env_params, model_params, optimizer_params, trainer_params)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ RL 모델 로드 완료: {checkpoint_name}")
                # 학습 알고리즘 정보 표시
                saved_alg = checkpoint.get('algorithm_type',
                            checkpoint.get('trainer_params', {}).get('algorithm_type', 'reinforce'))
                print(f"   └─ Saved algorithm: {str(saved_alg).upper()}")
            else:
                trainer.model.load_state_dict(checkpoint)
                print(f"✅ RL 모델 로드 완료 (이전 형식): {checkpoint_name}")

            trainer.model.to(device)
            trainer.model.eval()

        except Exception as e:
            print(f"❌ RL 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        rl_results = []

        for i in range(TEST_FILE_START, TEST_FILE_END + 1):
            data_path = data_base_dir / f"{i}.pickle"
            if not data_path.exists():
                print(f"⚠️  Warning: Data file not found: {data_path}")
                continue
            print(f"\n📋 테스트 파일 {i}.pickle")
            with open(data_path, 'rb') as fr:
                problem = pickle.load(fr)

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

                        pi, v = trainer.model(
                            fea_act, act_mask, candidate, fea_team,
                            team_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                            pred_idx, succ_idx
                        )
                        action_flat = torch.argmax(pi, dim=1)
                        N_T = test_env.N_T
                        act_idx  = action_flat // N_T
                        team_idx = action_flat % N_T
                        s, obj_value, done = test_env.step_pair(
                            act_idx.to(test_env.device),
                            team_idx.to(test_env.device)
                        )

                    test_score = test_env._get_obj()
                    if isinstance(test_score, torch.Tensor):
                        test_score = test_score.mean().item()

                end_time = time.time()
                runtime = end_time - start_time

                rl_results.append({
                    'algorithm': 'RL',
                    'instance': i,
                    'objective_value': test_score,
                    'runtime': runtime,
                })
                print(f"   [RL][Instance {i}] {OBJECTIVE}: {test_score:.4f}, Runtime: {runtime:.4f}s")

                # 간트차트 및 선후관계 그래프 생성
                if SAVE_GANTT_CHART:
                    try:
                        create_gantt_chart_from_env(
                            env=test_env,
                            instance_name=f"instance_{i}",
                            algorithm="RL",
                            objective_value=test_score,
                            save_dir=gantt_dir,
                            show=SHOW_GANTT_CHART
                        )
                        if i not in precedence_graphs_created:
                            num_act = test_env.num_activities[0].item()
                            activity_predecessors = {}
                            activity_mutex = {}
                            activity_to_project = {}
                            for act_id in range(num_act):
                                preds = test_env.activity_predecessors[0, act_id].cpu().numpy().tolist()
                                activity_predecessors[act_id] = [p for p in preds if p >= 0]
                                mutex = test_env.activity_mutex[0, act_id].cpu().numpy().tolist()
                                activity_mutex[act_id] = [m for m in mutex if m >= 0]
                                activity_to_project[act_id] = test_env.activity_project[0, act_id].item()
                            create_precedence_graph(
                                activity_predecessors=activity_predecessors,
                                activity_mutex=activity_mutex,
                                activity_to_project=activity_to_project,
                                instance_name=f"instance_{i}",
                                save_dir=gantt_dir,
                                show=SHOW_GANTT_CHART
                            )
                            precedence_graphs_created.add(i)
                    except Exception as e:
                        print(f"   ⚠️ 간트차트/그래프 생성 실패: {e}")

            except Exception as e:
                print(f"   ❌ [RL][Instance {i}] 실행 중 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # RL 결과 저장
        if rl_results:
            valid_objectives = [r['objective_value'] for r in rl_results if r['objective_value'] is not None]
            if valid_objectives:
                avg_objective = np.mean(valid_objectives)
                algorithm_summaries[f'RL_{checkpoint_name}'] = avg_objective
                print(f"  RL 평균 {OBJECTIVE}: {avg_objective:.4f}")
            
            output_path = save_results_to_excel(rl_results, 'RL', checkpoint_name)
            if output_path:
                saved_files.append(output_path)
    
    # -----------------------------
    # 9) 전체 결과 요약
    # -----------------------------
    if algorithm_summaries:
        print(f"\n{'='*60}")
        print("전체 결과 요약")
        print(f"{'='*60}")
        
        for algo, avg_obj in algorithm_summaries.items():
            print(f"  {algo}: {avg_obj:.4f}")
        
        # Best 알고리즘 찾기
        best_algo = min(algorithm_summaries, key=algorithm_summaries.get)
        best_obj = algorithm_summaries[best_algo]
        print(f"\n🏆 Best Algorithm: {best_algo} ({OBJECTIVE}: {best_obj:.4f})")
    
    print(f"\n{'='*60}")
    print(f"저장된 파일 목록 (폴더: {session_results_dir})")
    print(f"{'='*60}")
    for file_path in saved_files:
        print(f"- {file_path.name}")
    
    print(f"\n✅ 모든 결과가 {session_results_dir}에 저장되었습니다.")
    print("✅ 프로그램 종료")
