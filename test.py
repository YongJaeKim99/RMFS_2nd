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

# -----------------------------
# 0) CPU 디바이스로 고정 (추론 시에는 CPU로)
# -----------------------------
device = torch.device("cpu")
print("Using device:", device)

if __name__ == "__main__":
    # =================================================================
    # 🎯 테스트 파라미터 설정
    # =================================================================
    
    # -----------------------------
    # 1) 경로 설정 및 체크포인트 선택 옵션
    # -----------------------------
    project_root = Path(__file__).parent 
    ckpt_dir = project_root / "checkpoints/20260209_120000_tardiness_REINFORCE"  # 예시
    
    # 모든 체크포인트에 대해 실험할지 여부 (True: 모든 체크포인트, False: 특정 체크포인트만)
    TEST_ALL_CHECKPOINTS = True
    
    # 디버그 모드 설정
    DEBUG_ENV = False
    DEBUG_MODEL = False
    
    # 시드 고정
    SEED = 42
    
    # =================================================================
    # 🎯 목적함수 선택
    # =================================================================
    OBJECTIVE = 'tardiness'  # 'tardiness' or 'makespan'
    
    # =================================================================
    # 📊 테스트 데이터 설정
    # =================================================================
    TEST_DATA_TYPE = 'test'  # 'test' or 'val'
    
    # 테스트 데이터 파일 범위 설정
    TEST_FILE_START = 0
    TEST_FILE_END = 10
    
    # -----------------------------
    # 2) 테스트할 알고리즘 설정
    # -----------------------------
    # RL: 학습된 GNN 모델 사용
    # GA: 유전 알고리즘 (GA.py)
    test_algorithms = ["GA", "RL"]
    
    # GA 설정
    GA_POPULATION_SIZE = 100
    GA_GENERATIONS = 300
    GA_DECODE_MODE = "immediate"  # "batch" or "immediate"
    
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
    print(f"  📌 TEST_DATA_TYPE: {TEST_DATA_TYPE}")
    print(f"  📌 TEST_FILE_RANGE: {TEST_FILE_START}~{TEST_FILE_END}")
    print(f"  📌 ALGORITHMS: {', '.join(test_algorithms)}")
    if "GA" in test_algorithms:
        print(f"  📌 GA Settings:")
        print(f"     - Population: {GA_POPULATION_SIZE}")
        print(f"     - Generations: {GA_GENERATIONS}")
        print(f"     - Decode Mode: {GA_DECODE_MODE}")
    print("="*50 + "\n")
    
    # =================================================================
    
    # -----------------------------
    # 3) 파라미터 설정
    # -----------------------------
    env_params = {
        'batch_size': 1,  # 테스트 시 배치 크기
        'pomo_size': 1,   # 테스트 시 POMO 크기
        'N_P': 5,  # 프로젝트 수
        'N_A_min': 4,  # 프로젝트당 최소 activity 수
        'N_A_max': 6,  # 프로젝트당 최대 activity 수
        'N_T': 4,  # 팀 수
        'duration_min': 2,
        'duration_max': 6,
        'precedence_prob': 0.3,
        'mutex_prob': 0.1,
        'eligible_teams_ratio': 0.6,
        'due_date_tightness': 1.3,
        'objective': OBJECTIVE,
        'debug_env': DEBUG_ENV,
    }

    # 모델 파라미터 설정
    model_params = {
        'embedding_dim': 128, 
        'num_head': 8,
        'num_encoder_layer': 3,
        'input_dim': 10,
    }
    
    # 옵티마이저 파라미터 설정
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
    }
    
    # 체크포인트 폴더 생성
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    
    # -----------------------------
    # 4) 유틸 함수: 결과를 엑셀로 저장
    # -----------------------------
    current_dir = Path(__file__).parent
    results_dir = current_dir / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_results_dir = results_dir / timestamp
    
    if not session_results_dir.exists():
        session_results_dir.mkdir(parents=True)
        print(f"✅ 결과 저장 폴더 생성: {session_results_dir}")
    
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
    
    # -----------------------------
    # 5) 데이터 경로 설정
    # -----------------------------
    if TEST_DATA_TYPE == 'val':
        data_base_dir = project_root / "generated_datasets" / "data_val"
    else:
        data_base_dir = project_root / "generated_datasets" / "data_test"
    
    print(f"데이터 경로: {data_base_dir}")
    
    # -----------------------------
    # 6) 체크포인트 로드 (RL 테스트가 있을 때만)
    # -----------------------------
    all_ckpts = []
    if "RL" in test_algorithms:
        if TEST_ALL_CHECKPOINTS and ckpt_dir.exists():
            all_ckpts = sorted(
                ckpt_dir.glob("scheduling_epoch*.pt"),
                key=lambda p: int(p.stem.replace('scheduling_epoch', ''))
            )
            if not all_ckpts:
                print(f"[WARNING] No checkpoints found in {ckpt_dir}")
                print(f"[INFO] RL 테스트를 건너뜁니다.")
            else:
                print(f"[INFO] Found {len(all_ckpts)} checkpoints: {[ckpt.name for ckpt in all_ckpts]}")
        elif ckpt_dir.exists():
            ckpts = sorted(
                ckpt_dir.glob("scheduling_epoch*.pt"),
                key=lambda p: int(p.stem.replace('scheduling_epoch', ''))
            )
            if ckpts:
                all_ckpts = [ckpts[-1]]  # 가장 최신 체크포인트만
                print(f"[INFO] Using single checkpoint: {all_ckpts[0].name}")
            else:
                print(f"[WARNING] No checkpoints found in {ckpt_dir}")
        else:
            print(f"[WARNING] Checkpoint directory does not exist: {ckpt_dir}")
    
    # -----------------------------
    # 7) 알고리즘별 실험 실행
    # -----------------------------
    algorithm_summaries = {}
    saved_files = []
    
    # GA 실행 (RL과 독립적)
    if "GA" in test_algorithms:
        print(f"\n{'='*60}")
        print(f"Testing Algorithm: GA")
        print(f"{'='*60}")
        
        ga_results = []
        
        for i in range(TEST_FILE_START, TEST_FILE_END + 1):
            data_path = data_base_dir / f"{i}.pickle"
            if not data_path.exists():
                print(f"Warning: Data file not found: {data_path}")
                continue
            
            print(f"\n📋 테스트 파일 {i}.pickle")
            
            start_time = time.time()
            
            try:
                # 테스트 데이터 로드
                with open(data_path, 'rb') as fr:
                    problem = pickle.load(fr)
                
                print(f"   테스트 데이터 로드: {data_path}")
                
                # pickle 데이터에서 프로젝트 정보 추출
                # batch_projects와 batch_activities 사용
                if 'batch_projects' in problem and 'batch_activities' in problem:
                    # 첫 번째 배치의 프로젝트 사용 (테스트는 배치 크기 1)
                    projects = problem['batch_projects'][0] if isinstance(problem['batch_projects'], list) else problem['batch_projects']
                    
                    # GA 실행
                    ga = GeneticAlgorithm(
                        projects=projects,
                        num_teams=env_params['N_T'],
                        population_size=GA_POPULATION_SIZE,
                        generations=GA_GENERATIONS,
                        decode_mode=GA_DECODE_MODE
                    )
                    best_solution = ga.evolve()
                    objective_value = best_solution.objective
                else:
                    print("   ⚠️  경고: pickle 데이터에 'batch_projects'가 없습니다.")
                    objective_value = 0.0
                
                end_time = time.time()
                runtime = end_time - start_time
                
                result = {
                    'algorithm': 'GA',
                    'instance': i,
                    'objective_value': objective_value,
                    'runtime': runtime
                }
                ga_results.append(result)
                
                print(f"[GA][Instance {i}] {OBJECTIVE}: {objective_value:.4f}, Runtime: {runtime:.4f}s")
            
            except Exception as e:
                print(f"❌ [GA][Instance {i}] 실행 중 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # GA 결과 저장
        if ga_results:
            valid_objectives = [r['objective_value'] for r in ga_results if r['objective_value'] is not None]
            if valid_objectives:
                avg_objective = np.mean(valid_objectives)
                algorithm_summaries['GA'] = avg_objective
                print(f"  GA 평균 {OBJECTIVE}: {avg_objective:.4f}")
            
            output_path = save_results_to_excel(ga_results, 'GA', 'N/A')
            if output_path:
                saved_files.append(output_path)
    
    # RL 실행 (체크포인트별)
    for ckpt_path in all_ckpts:
        checkpoint_name = ckpt_path.stem
        print(f"\n{'='*60}")
        print(f"Testing Checkpoint: {checkpoint_name}")
        print(f"{'='*60}")
        
        # Trainer 초기화
        trainer = Scheduling_Trainer(env_params, model_params, optimizer_params, trainer_params)
        
        # 체크포인트 로드
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # TODO: 모델 로드 (모델 추가 후 주석 해제)
            # if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            #     trainer.model.load_state_dict(checkpoint['model_state_dict'])
            #     print(f"✅ RL 모델 로드 완료: {checkpoint_name}")
            # else:
            #     trainer.model.load_state_dict(checkpoint)
            #     print(f"✅ RL 모델 로드 완료 (이전 형식): {checkpoint_name}")
            
            # trainer.model.to(device)
            # trainer.model.eval()
            
            print("⚠️  경고: GNN 모델이 없어 체크포인트 로드를 건너뜁니다.")
        except Exception as e:
            print(f"❌ RL 모델 로드 실패: {e}")
            continue
        
        rl_results = []
        
        for i in range(TEST_FILE_START, TEST_FILE_END + 1):
            data_path = data_base_dir / f"{i}.pickle"
            if not data_path.exists():
                print(f"Warning: Data file not found: {data_path}")
                continue
            
            print(f"\n📋 테스트 파일 {i}.pickle")
            
            start_time = time.time()
            
            try:
                # TODO: GNN 모델 추가 후 RL 테스트 구현
                print("   ⚠️  경고: GNN 모델이 없어 RL 테스트를 건너뜁니다.")
                print("   GNN 모델(SchedulingModel) 추가 후 RL 테스트를 구현하세요.")
                
                # 예시: RL 테스트 (모델 추가 후 주석 해제)
                # with torch.no_grad():
                #     # 테스트 데이터 로드
                #     with open(data_path, 'rb') as fr:
                #         problem = pickle.load(fr)
                #     
                #     # 환경 초기화
                #     test_env = SchedulingEnv(env_params, debug_env=False)
                #     test_env._reset(problem)
                #     
                #     done = False
                #     s = test_env._get_state()
                #     
                #     while not done.all():
                #         action = trainer.model.get_max_action(s)
                #         s, reward, done = test_env.step(action.to('cpu'))
                #     
                #     test_score = test_env.get_objective()
                #     if isinstance(test_score, torch.Tensor):
                #         test_score = test_score.mean().item()
                
                # 임시: 더미 결과
                test_score = 0.0
                
                end_time = time.time()
                runtime = end_time - start_time
                
                result = {
                    'algorithm': 'RL',
                    'instance': i,
                    'objective_value': test_score,
                    'runtime': runtime
                }
                rl_results.append(result)
                
                print(f"[RL][Instance {i}] {OBJECTIVE}: {test_score:.4f}, Runtime: {runtime:.4f}s")
            
            except Exception as e:
                print(f"❌ [RL][Instance {i}] 실행 중 오류: {e}")
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
    # 8) 전체 결과 요약
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
