"""
RMFS 환경 데이터를 MILP 솔버 입력 형식으로 변환하는 유틸리티.

RMFS_Environment (0-indexed) → MILP data dict (1-indexed)
"""
import numpy as np


def convert_env_to_milp_data(env, n_t_multiplier=1.5):
    """
    reset() 직후의 RMFS_Environment에서 MILP 입력 데이터를 추출한다.

    Args:
        env: RMFS_Environment instance (reset 완료 상태)
        n_t_multiplier: N_T 추정 시 여유 계수 (default 1.5)

    Returns:
        dict with keys (1-indexed):
            N_T, P, K, S, R, TT, UT, n_Seq_P, Seq_P, PT, ST, S_P, IS
    """
    N_S = env.N_S
    N_W = env.N_W
    N_P = env.N_P
    N_R = env.N_R

    # --- Sets (1-indexed) ---
    # S: storage locations [1, ..., N_S]
    S = list(range(1, N_S + 1))

    # K: workstations [N_S+1, ..., N_S+N_W]
    K = list(range(N_S + 1, N_S + N_W + 1))

    # L: all locations = S + K
    L = S + K

    # P: pods that appear in workstation sequences (1-indexed)
    used_pods = set()
    for w in range(N_W):
        for pod_idx in env.Pod_Sequence_in_WS[w]:
            used_pods.add(pod_idx)
    P = sorted([p + 1 for p in used_pods])  # 0-indexed → 1-indexed

    # R: robots [1, ..., N_R]
    R = list(range(1, N_R + 1))

    # --- Travel Time matrix (1-indexed dict of dicts) ---
    TT = {}
    for loc1 in L:
        TT[loc1] = {}
        for loc2 in L:
            if loc1 <= N_S and loc2 <= N_S:
                # storage → storage
                TT[loc1][loc2] = int(env.TT_SS[loc1 - 1][loc2 - 1])
            elif loc1 > N_S and loc2 > N_S:
                # workstation → workstation
                w1 = loc1 - N_S - 1
                w2 = loc2 - N_S - 1
                TT[loc1][loc2] = int(env.TT_WW[w1][w2])
            elif loc1 > N_S and loc2 <= N_S:
                # workstation → storage
                w = loc1 - N_S - 1
                s = loc2 - 1
                TT[loc1][loc2] = int(env.TT_WS[w][s])
            else:
                # storage → workstation (맨하탄 거리는 대칭)
                w = loc2 - N_S - 1
                s = loc1 - 1
                TT[loc1][loc2] = int(env.TT_WS[w][s])

    # --- Pod Sequence per Workstation (1-indexed) ---
    n_Seq_P = max(len(env.Pod_Sequence_in_WS[w]) for w in range(N_W))

    Seq_P = {}
    PT_dict = {}
    for w in range(N_W):
        k = N_S + w + 1  # workstation 1-indexed ID
        Seq_P[k] = {}
        PT_dict[k] = {}
        for sp in range(1, n_Seq_P + 1):
            seq_idx = sp - 1  # 0-indexed
            if seq_idx < len(env.Pod_Sequence_in_WS[w]):
                pod_0idx = env.Pod_Sequence_in_WS[w][seq_idx]
                Seq_P[k][sp] = pod_0idx + 1  # 1-indexed pod ID
                PT_dict[k][sp] = int(env.PT[w][seq_idx])
            else:
                Seq_P[k][sp] = 0  # no pod
                PT_dict[k][sp] = 0

    # --- Pod initial storage positions (1-indexed) ---
    S_P = {}
    for p_1idx in P:
        p_0idx = p_1idx - 1
        S_P[p_1idx] = int(env.Pod_Init[p_0idx]) + 1  # storage 1-indexed

    # --- Robot initial positions (1-indexed) ---
    IS = {}
    for r in range(N_R):
        IS[r + 1] = int(env.Robot_Init[r]) + 1  # storage 1-indexed

    # --- Estimate N_T ---
    total_tasks = sum(len(env.Pod_Sequence_in_WS[w]) for w in range(N_W))
    max_tasks_per_ws = max(len(env.Pod_Sequence_in_WS[w]) for w in range(N_W))
    avg_pt = np.mean([env.PT[w][j]
                       for w in range(N_W)
                       for j in range(len(env.PT[w]))
                       if env.PT[w][j] > 0]) if total_tasks > 0 else env.Unit_PT
    max_travel = int(np.max(env.TT_WS))
    estimated_time = max_tasks_per_ws * (avg_pt + env.ST + max_travel)
    N_T = int(estimated_time / env.UT * n_t_multiplier)
    N_T = max(N_T, total_tasks * 5)  # 최소 보장

    return {
        'N_T': N_T,
        'P': P,
        'K': K,
        'S': S,
        'R': R,
        'TT': TT,
        'UT': env.UT,
        'n_Seq_P': n_Seq_P,
        'Seq_P': Seq_P,
        'PT': PT_dict,
        'ST': env.ST,
        'S_P': S_P,
        'IS': IS,
    }
