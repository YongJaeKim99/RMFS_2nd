"""
RMFS 테스트 데이터 생성 스크립트.

rmfs_test.py에서 사용할 테스트 인스턴스 배치를 pickle 파일로 생성한다.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import numpy as np
import pickle
from pathlib import Path

from rmfs_data_generator import generate_rmfs_data_batch


if __name__ == "__main__":
    # =================================================================
    # 데이터 생성 설정
    # =================================================================
    SEED = 2027
    NUM_INSTANCES = 50

    # =================================================================
    # RMFS 환경 파라미터 (rmfs_train.py와 동일)
    # =================================================================
    env_params = {
        'batch_size': NUM_INSTANCES,
        'block_rows': 1,
        'block_cols': 1,
        'block_h': 4,
        'block_w': 2,
        # N_S = 3*3*4*2 = 72 storage locations
        'N_P': 3,
        'N_R': 2,
        'N_W': 2,
        'Total_PodTask': 4,
        'Unit_PT': 15,
        'ST': 5,
        'UT': 5,
        'Large': True,
        'force_mask_stay': True,
        'seed_base': SEED,
    }

    # =================================================================
    # 출력 설정
    # =================================================================
    output_folder = "data/rmfs_test"
    output_file = "test_batch.pickle"

    N_S = env_params['block_rows'] * env_params['block_cols'] * env_params['block_h'] * env_params['block_w']

    print("\n" + "=" * 80)
    print("RMFS 테스트 데이터 생성")
    print("=" * 80)
    print(f"  인스턴스 수: {NUM_INSTANCES}")
    print(f"  Seed: {SEED}")
    print(f"  Block: {env_params['block_rows']}x{env_params['block_cols']} "
          f"({env_params['block_h']}x{env_params['block_w']}/block)")
    print(f"  N_S: {N_S}")
    print(f"  N_P: {env_params['N_P']}")
    print(f"  N_R: {env_params['N_R']}")
    print(f"  N_W: {env_params['N_W']}")
    print(f"  Total_PodTask: {env_params['Total_PodTask']}")
    print(f"  Unit_PT: {env_params['Unit_PT']}")
    print(f"  ST: {env_params['ST']}")
    print(f"  UT: {env_params['UT']}")
    print(f"  Large: {env_params['Large']}")
    print(f"  Force Mask Stay: {env_params['force_mask_stay']}")
    print("=" * 80 + "\n")

    # 출력 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"폴더 생성: {output_folder}")

    # =================================================================
    # 데이터 생성
    # =================================================================
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"{NUM_INSTANCES}개 인스턴스 배치 생성 중...")

    problem = generate_rmfs_data_batch(env_params, epoch=0)

    # 환경 파라미터도 함께 저장 (테스트 시 환경 재생성에 필요)
    problem['env_params'] = {
        'block_rows': env_params['block_rows'],
        'block_cols': env_params['block_cols'],
        'block_h': env_params['block_h'],
        'block_w': env_params['block_w'],
        'Unit_PT': env_params['Unit_PT'],
        'ST': env_params['ST'],
        'UT': env_params['UT'],
        'Large': env_params['Large'],
        'force_mask_stay': env_params['force_mask_stay'],
    }

    # Pickle 파일 저장
    current_path = Path(os.getcwd())
    pickle_path = str(current_path / output_folder / output_file)

    with open(pickle_path, 'wb') as f:
        pickle.dump(problem, f, pickle.HIGHEST_PROTOCOL)
    print(f"저장 완료: {pickle_path}")

    # =================================================================
    # 검증
    # =================================================================
    print(f"\n{'=' * 80}")
    print(f"파일 검증")
    print(f"{'=' * 80}")

    try:
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)

        print(f"  파일 로드 성공: {output_file}")
        print(f"  seeds 수: {len(loaded['seeds'])}")
        print(f"  seeds 범위: [{loaded['seeds'][0]}, {loaded['seeds'][-1]}]")
        print(f"  N_P: {loaded['N_P']}")
        print(f"  N_R: {loaded['N_R']}")
        print(f"  N_W: {loaded['N_W']}")
        print(f"  Total_PodTask: {loaded['Total_PodTask']}")
        print(f"  env_params keys: {list(loaded['env_params'].keys())}")
    except Exception as e:
        print(f"  파일 검증 중 오류: {e}")

    print(f"\n{'=' * 80}")
    print("데이터 생성 완료!")
    print(f"{'=' * 80}")
    print(f"  저장 위치: {output_folder}/{output_file}")
    print(f"  rmfs_test.py에서 TEST_DATA_DIR = \"{output_folder}\" 로 설정하여 사용")
