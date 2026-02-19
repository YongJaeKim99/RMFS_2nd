"""
RCMPSP (Resource-Constrained Multi-Project Scheduling Problem) 테스트 데이터 생성 스크립트
단일 배치 pickle 파일로 생성 (test.py에서 RL은 배치 추론, GA는 인스턴스별 루프)
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

    # 인스턴스 수 (하나의 pickle 파일에 담길 인스턴스 수)
    NUM_INSTANCES = 50
    POMO_SIZE = 1   # 테스트는 POMO 사용 안 함

    # =================================================================
    # 🎯 목적함수 선택
    # =================================================================
    OBJECTIVE = 'tardiness'  # 'tardiness' or 'makespan'

    # =================================================================
    # 📏 문제 파라미터 설정
    # =================================================================
    env_params = {
        'batch_size': NUM_INSTANCES,
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
    output_file = "test_batch.pickle"

    print("\n" + "="*80)
    print("🎲 RCMPSP 테스트 데이터 생성")
    print("="*80)
    print(f"  목적함수: {OBJECTIVE}")
    print(f"  인스턴스 수: {NUM_INSTANCES}")
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
    # 📦 데이터 생성 (단일 배치)
    # =================================================================
    # 시드 고정
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    print(f"📦 {NUM_INSTANCES}개 인스턴스 배치 생성 중...")
    problem = generate_scheduling_data_batch(env_params)

    # Pickle 파일 저장
    pickle_path = str(current_path / output_folder / output_file)
    with open(pickle_path, 'wb') as f:
        pickle.dump(problem, f, pickle.HIGHEST_PROTOCOL)
    print(f"✅ 저장 완료: {pickle_path}")

    print(f"\n{'='*80}")
    print("🎉 테스트 데이터 생성 완료!")
    print(f"{'='*80}")
    print(f"  인스턴스 수: {NUM_INSTANCES}")
    print(f"  POMO 크기: {POMO_SIZE}")
    print(f"  저장 위치: {output_folder}{output_file}")

    # 요약 파일 생성
    summary_path = str(current_path / output_folder / "_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"RCMPSP Test Dataset Generation Summary\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"[전체 설정]\n")
        f.write(f"  Seed: {SEED}\n")
        f.write(f"  인스턴스 수: {NUM_INSTANCES}\n")
        f.write(f"  POMO 크기: {POMO_SIZE}\n")
        f.write(f"  저장 위치: {output_folder}{output_file}\n\n")

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

        f.write(f"{'='*80}\n")

    print(f"📄 요약 저장 완료: {output_folder}_summary.txt")

    # 파일 검증
    print(f"\n{'='*80}")
    print(f"📋 파일 검증 ({output_file})")
    print(f"{'='*80}")

    try:
        with open(pickle_path, 'rb') as f:
            loaded_problem = pickle.load(f)

        num_loaded = loaded_problem['num_activities'].shape[0]
        print(f"  ✅ 파일 로드 성공: {output_file}")
        print(f"  인스턴스 수: {num_loaded}")
        print(f"  인스턴스별 Activity 수: {loaded_problem['num_activities'].tolist()}")
        print(f"  최대 Activity 수 (max_N_A): {loaded_problem['env_params']['max_N_A']}")
        print(f"  프로젝트 수: {loaded_problem['project_release_time'].shape[1]}")
        print(f"  팀 수: {loaded_problem['activity_eligible_teams'].shape[2]}")
        print(f"  목적함수: {loaded_problem['env_params']['objective']}")

    except Exception as e:
        print(f"  ❌ 파일 검증 중 오류: {e}")

    print(f"\n{'='*80}")
    print("✨ 데이터 생성 및 검증 완료!")
    print(f"{'='*80}")
