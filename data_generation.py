"""
RCMPSP (Resource-Constrained Multi-Project Scheduling Problem) 테스트 데이터 생성 스크립트
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OpenMP 중복 로드 경고 무시

from data_generator import generate_scheduling_data_batch
import torch
import random
import numpy as np
from pathlib import Path
import pickle

if __name__ == "__main__":
    # =================================================================
    # 🎯 데이터 생성 설정
    # =================================================================
    
    # Seed 설정
    SEED = 0
    
    # 배치 크기 (각 파일당 인스턴스 수)
    BATCH_SIZE = 1  # 테스트 데이터는 파일당 1개 인스턴스
    POMO_SIZE = 1   # 테스트는 POMO 사용 안 함
    
    # 생성할 파일 수
    NUM_FILES = 30  # 0.pickle ~ 99.pickle
    
    # =================================================================
    # 🎯 목적함수 선택
    # =================================================================
    OBJECTIVE = 'tardiness'  # 'tardiness' or 'makespan'
    
    # =================================================================
    # 📏 문제 파라미터 설정
    # =================================================================
    env_params = {
        'batch_size': BATCH_SIZE,
        'pomo_size': POMO_SIZE,
        'N_P': 5,  # 프로젝트 수
        'N_A_min': 4,  # 프로젝트당 최소 activity 수
        'N_A_max': 6,  # 프로젝트당 최대 activity 수
        'N_T': 4,  # 팀 수
        'duration_min': 2,  # 최소 작업 시간
        'duration_max': 6,  # 최대 작업 시간
        'precedence_prob': 0.3,  # 선행 관계 생성 확률
        'mutex_prob': 0.1,  # 동시 불가 생성 확률
        'eligible_teams_ratio': 0.6,  # 평균 eligible 팀 비율
        'due_date_tightness': 1.3,  # Due date 여유도 (1.0 = tight, 1.5 = loose)
        'objective': OBJECTIVE,
        'debug_env': False,
    }
    
    # =================================================================
    # 📂 출력 폴더 설정
    # =================================================================
    current_path = Path(os.getcwd())
    output_folder = "data/test/"
    
    print("\n" + "="*80)
    print("🎲 RCMPSP 테스트 데이터 생성")
    print("="*80)
    print(f"  목적함수: {OBJECTIVE}")
    print(f"  생성할 파일 수: {NUM_FILES}")
    print(f"  Seed: {SEED}")
    print(f"  프로젝트 수 (N_P): {env_params['N_P']}")
    print(f"  Activity 수 범위: {env_params['N_A_min']}-{env_params['N_A_max']}")
    print(f"  팀 수 (N_T): {env_params['N_T']}")
    print(f"  작업 시간 범위: {env_params['duration_min']}-{env_params['duration_max']}")
    print(f"  선행 관계 확률: {env_params['precedence_prob']}")
    print(f"  Mutex 확률: {env_params['mutex_prob']}")
    print(f"  Eligible 팀 비율: {env_params['eligible_teams_ratio']}")
    print(f"  Due Date Tightness: {env_params['due_date_tightness']}")
    print("="*80 + "\n")
    
    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 폴더 생성: {output_folder}")
    
    # =================================================================
    # 📦 데이터 생성
    # =================================================================
    for i in range(NUM_FILES):
        # 각 파일마다 시드 변경 (재현성 보장)
        current_seed = SEED + i
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)
        
        file_name = f"{i}"
        print(f"📦 생성 중... [{i+1}/{NUM_FILES}] {file_name}.pickle (Seed: {current_seed})")
        
        # 문제 생성
        problem = generate_scheduling_data_batch(env_params)
        
        # Pickle 파일 저장
        pickle_path = str(current_path) + "/" + output_folder + file_name + '.pickle'
        with open(pickle_path, 'wb') as f:
            pickle.dump(problem, f, pickle.HIGHEST_PROTOCOL)
        
        # 진행률 표시 (10개마다)
        if (i + 1) % 10 == 0 or i == NUM_FILES - 1:
            print(f"   ✅ {i+1}/{NUM_FILES} 완료 ({(i+1)/NUM_FILES*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("🎉 테스트 데이터 생성 완료!")
    print(f"{'='*80}")
    print(f"  생성된 파일 수: {NUM_FILES}")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  POMO 크기: {POMO_SIZE}")
    print(f"  저장 위치: {output_folder}")
    print(f"  파일 범위: 0.pickle ~ {NUM_FILES-1}.pickle")
    
    # 요약 파일 생성
    summary_path = str(current_path) + "/" + output_folder + "_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"RCMPSP Test Dataset Generation Summary\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"[전체 설정]\n")
        f.write(f"  Seed 시작: {SEED}\n")
        f.write(f"  배치 크기: {BATCH_SIZE}\n")
        f.write(f"  POMO 크기: {POMO_SIZE}\n")
        f.write(f"  생성된 파일 수: {NUM_FILES}\n")
        f.write(f"  저장 위치: {output_folder}\n")
        f.write(f"  파일 범위: 0.pickle ~ {NUM_FILES-1}.pickle\n\n")
        
        f.write(f"[문제 파라미터]\n")
        f.write(f"  목적함수: {OBJECTIVE}\n")
        f.write(f"  프로젝트 수 (N_P): {env_params['N_P']}\n")
        f.write(f"  프로젝트당 Activity 수: {env_params['N_A_min']}-{env_params['N_A_max']}\n")
        f.write(f"  팀 수 (N_T): {env_params['N_T']}\n")
        f.write(f"  작업 시간 범위: {env_params['duration_min']}-{env_params['duration_max']}\n")
        f.write(f"  선행 관계 확률: {env_params['precedence_prob']}\n")
        f.write(f"  Mutex 확률: {env_params['mutex_prob']}\n")
        f.write(f"  Eligible 팀 비율: {env_params['eligible_teams_ratio']}\n")
        f.write(f"  Due Date Tightness: {env_params['due_date_tightness']}\n\n")
        
        f.write(f"[생성 정보]\n")
        f.write(f"  각 파일의 Seed: {SEED} + 파일번호\n")
        f.write(f"    예) 0.pickle: Seed {SEED}\n")
        f.write(f"        1.pickle: Seed {SEED+1}\n")
        f.write(f"        ...\n")
        f.write(f"        {NUM_FILES-1}.pickle: Seed {SEED+NUM_FILES-1}\n\n")
        
        f.write(f"{'='*80}\n")
    
    print(f"📄 요약 저장 완료: {output_folder}_summary.txt")
    
    # 샘플 파일 검증
    print(f"\n{'='*80}")
    print("📋 샘플 파일 검증 (0.pickle)")
    print(f"{'='*80}")
    
    try:
        pickle_file_path = str(current_path) + "/" + output_folder + "0.pickle"
        with open(pickle_file_path, 'rb') as f:
            loaded_problem = pickle.load(f)
        
        print(f"  ✅ 파일 로드 성공: 0.pickle")
        print(f"  배치 크기: {loaded_problem['num_activities'].shape[0]}")
        print(f"  실제 Activity 수: {loaded_problem['num_activities'][0].item()}")
        print(f"  최대 Activity 수: {loaded_problem['env_params']['max_N_A']}")
        print(f"  프로젝트 수: {loaded_problem['project_release_time'].shape[1]}")
        print(f"  팀 수: {loaded_problem['activity_eligible_teams'].shape[2]}")
        print(f"  목적함수: {loaded_problem['env_params']['objective']}")
        
    except Exception as e:
        print(f"  ❌ 파일 검증 중 오류: {e}")
    
    print(f"\n{'='*80}")
    print("✨ 데이터 생성 및 검증 완료!")
    print(f"{'='*80}")
