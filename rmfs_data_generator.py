"""
RMFS 문제 인스턴스 배치 생성기.

RMFS_Environment.reset()은 seed를 받아 내부적으로 모든 랜덤 데이터를 생성하므로,
여기서는 고유 seed 목록과 공유 파라미터만 제공한다.
"""
import random


def generate_rmfs_data_batch(env_params, epoch=0, pomo_size=1):
    """
    RMFS 문제 인스턴스 배치를 생성한다.

    Args:
        env_params: dict with keys:
            batch_size, N_P, N_R, N_W, Total_PodTask, seed_base
        epoch: 현재 epoch 번호 (seed 계산에 사용)
        pomo_size: POMO 크기 (같은 인스턴스를 K번 반복). 기본 1.

    Returns:
        problem: dict {
            'seeds': list[int],    # 인스턴스별 고유 시드 (POMO 시 반복 포함)
            'N_P': int,
            'N_R': int,
            'N_W': int,
            'Total_PodTask': int,
        }
    """
    batch_size = env_params['batch_size']
    seed_base = env_params.get('seed_base', 0)

    # 매 epoch 고유 시드 생성
    base_seeds = [seed_base + epoch * batch_size + i for i in range(batch_size)]

    # POMO: 각 시드를 pomo_size번 반복 → 총 batch_size * pomo_size개
    if pomo_size > 1:
        seeds = []
        for s in base_seeds:
            seeds.extend([s] * pomo_size)
    else:
        seeds = base_seeds

    return {
        'seeds': seeds,
        'N_P': env_params['N_P'],
        'N_R': env_params['N_R'],
        'N_W': env_params['N_W'],
        'Total_PodTask': env_params['Total_PodTask'],
    }


def generate_rmfs_validation_batch(env_params, validation_batch_size, seed=2025):
    """
    고정 시드로 validation 배치를 생성한다.

    Args:
        env_params: 환경 파라미터
        validation_batch_size: validation 인스턴스 수
        seed: 고정 시드

    Returns:
        problem: dict (generate_rmfs_data_batch와 동일 형식)
    """
    seeds = [seed + i for i in range(validation_batch_size)]

    return {
        'seeds': seeds,
        'N_P': env_params['N_P'],
        'N_R': env_params['N_R'],
        'N_W': env_params['N_W'],
        'Total_PodTask': env_params['Total_PodTask'],
    }
