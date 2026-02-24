"""
Imitation Learning (Behavioral Cloning) 유틸리티
- CP-SAT 최적 스케줄을 RL 환경에서 replay하여 (state, action) 쌍 수집
- ILDataset: 수집된 데이터를 mini-batch로 로딩
"""

import os
import copy
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from scheduling_env import SchedulingEnv
from data_generator import convert_problem_to_mip_format
from samsung_MIP import solve_rcmpsp_cp


def extract_single_instance(problem, env_params, batch_idx):
    """
    batch problem dict에서 특정 인스턴스를 추출하여 batch_size=1인 problem dict 반환

    Args:
        problem: generate_scheduling_data_batch()로 생성된 batch problem dict
        env_params: 원본 env_params dict
        batch_idx: 추출할 인스턴스의 배치 인덱스

    Returns:
        problem_single: batch_size=1인 problem dict
        env_params_single: batch_size=1, pomo_size=1인 env_params dict
    """
    problem_single = {}

    # 텐서 키들을 슬라이싱
    tensor_keys = [
        'activity_duration', 'activity_team_duration', 'activity_project',
        'activity_eligible_teams', 'activity_predecessors', 'activity_successors',
        'activity_mutex', 'project_release_time', 'project_due_date',
        'num_activities'
    ]

    for key in tensor_keys:
        if key in problem and isinstance(problem[key], torch.Tensor):
            problem_single[key] = problem[key][batch_idx:batch_idx + 1]

    # env_params 복사 (max_N_A 등 유지)
    if 'env_params' in problem:
        problem_single['env_params'] = copy.deepcopy(problem['env_params'])
        problem_single['env_params']['batch_size'] = 1
        problem_single['env_params']['pomo_size'] = 1

    # 디버그용 raw 데이터 (있으면 복사)
    if 'batch_projects' in problem:
        problem_single['batch_projects'] = [problem['batch_projects'][batch_idx]]
    if 'batch_activities' in problem:
        problem_single['batch_activities'] = [problem['batch_activities'][batch_idx]]

    # env_params도 batch_size=1로 복사
    env_params_single = copy.deepcopy(env_params)
    env_params_single['batch_size'] = 1
    env_params_single['pomo_size'] = 1

    return problem_single, env_params_single


def replay_cp_solution(problem_single, env_params_single, start_times, assigned_teams):
    """
    CP 최적 스케줄을 RL 환경에서 단계별로 재현하여 (state, action) 쌍을 수집한다.

    논문의 학습 과정:
    (1) CP 최적 스케줄 획득
    (2) Active schedule로 변환 (환경이 자동으로 수행)
    (3) 환경에서 replay하면서 (state, action) 쌍 수집

    Args:
        problem_single: batch_size=1인 problem dict
        env_params_single: batch_size=1, pomo_size=1인 env_params
        start_times: Dict[int, float] — CP solver의 activity별 시작 시간
        assigned_teams: Dict[int, int] — CP solver의 activity별 팀 할당

    Returns:
        pairs: List[Dict] — 각 항목: {'state': {텐서dict}, 'action': int}
    """
    num_activities = problem_single['num_activities'][0].item()
    N_T = env_params_single['N_T']

    # 환경 생성 및 리셋
    env = SchedulingEnv(env_params_single, debug_env=False, device='cpu')
    env._reset(problem_single)

    pairs = []
    done = False
    s = env._get_state()
    step_count = 0
    MAX_STEPS = num_activities + 10  # safety margin

    while not done and step_count < MAX_STEPS:
        step_count += 1

        # dynamic_pair_mask: (1, N, T) — True = 불가능한 pair
        mask = s.dynamic_pair_mask_tensor[0]  # (N, T)

        # 사용 가능한 activity 찾기 (mask가 False인 (a, t) 쌍이 존재하는 activity)
        available_mask = ~mask  # True = 사용 가능
        available_acts = available_mask.any(dim=1).nonzero(as_tuple=False).squeeze(-1).tolist()

        # 패딩 activity 제거 (num_activities 이상 인덱스 제외)
        if isinstance(available_acts, int):
            available_acts = [available_acts]
        available_acts = [a for a in available_acts if a < num_activities]

        if len(available_acts) == 0:
            # 사용 가능한 activity가 없으면 종료 (이론상 발생 안 함)
            break

        # CP start_time 기준으로 가장 빠른 activity 선택
        best_act = None
        best_start = float('inf')
        for a in available_acts:
            cp_start = start_times.get(a, float('inf'))
            if cp_start < best_start:
                best_start = cp_start
                best_act = a

        if best_act is None:
            best_act = available_acts[0]  # fallback

        # CP에서 할당된 팀
        cp_team = assigned_teams.get(best_act, -1)

        # 검증: 해당 (activity, team) pair가 환경에서 사용 가능한지 확인
        if cp_team >= 0 and cp_team < N_T and available_mask[best_act, cp_team]:
            team_id = cp_team
        else:
            # Fallback: 해당 activity에서 사용 가능한 팀 중 아무거나 선택
            available_teams = available_mask[best_act].nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(available_teams, int):
                available_teams = [available_teams]
            if len(available_teams) > 0:
                team_id = available_teams[0]
            else:
                # 이론상 발생 안 함 (available_acts에서 이미 확인)
                break

        # State 텐서 저장 (batch 차원 squeeze, CPU에 clone)
        state_dict = {
            'fea_act': s.fea_act_tensor[0].clone(),
            'act_mask': s.act_mask_tensor[0].clone(),
            'fea_team': s.fea_team_tensor[0].clone(),
            'team_mask': s.team_mask_tensor[0].clone(),
            'dynamic_pair_mask': s.dynamic_pair_mask_tensor[0].clone(),
            'comp_idx': s.comp_idx_tensor[0].clone(),
            'candidate': s.candidate_tensor[0].clone(),
            'fea_pairs': s.fea_pairs_tensor[0].clone(),
            'pred_idx': s.pred_idx_tensor[0].clone(),
            'succ_idx': s.succ_idx_tensor[0].clone(),
        }
        if s.mutex_idx_tensor is not None:
            state_dict['mutex_idx'] = s.mutex_idx_tensor[0].clone()
        else:
            state_dict['mutex_idx'] = None

        # Flat action index: act_id * N_T + team_id
        flat_action = best_act * N_T + team_id

        pairs.append({
            'state': state_dict,
            'action': flat_action,
        })

        # 환경 step 실행
        act_tensor = torch.tensor([best_act], dtype=torch.long)
        team_tensor = torch.tensor([team_id], dtype=torch.long)
        s, obj_value, done = env.step_pair(act_tensor, team_tensor)

    return pairs


def collect_il_data(problem, env_params, cp_time_limit, inst_start=0, inst_end=None):
    """
    batch problem의 모든 인스턴스에 대해 CP solver → replay → (state, action) 쌍 수집

    Args:
        problem: batch problem dict
        env_params: env_params dict
        cp_time_limit: CP solver 시간 제한 (초)
        inst_start: 시작 인스턴스 인덱스
        inst_end: 끝 인스턴스 인덱스 (미포함, None이면 전체)

    Returns:
        all_pairs: List[Dict] — 모든 (state, action) 쌍
        success_count: int — 성공한 인스턴스 수
    """
    batch_size = problem['num_activities'].shape[0]
    if inst_end is None:
        inst_end = batch_size

    all_pairs = []
    success_count = 0

    for i in range(inst_start, inst_end):
        print(f"\n   📋 IL Instance {i}/{inst_end - 1}")
        try:
            # 1. MIP 형식으로 변환
            inst = convert_problem_to_mip_format(problem, i)

            # 2. CP-SAT solver로 풀기
            result = solve_rcmpsp_cp(inst, cp_time_limit)

            if result is None:
                print(f"     ⚠️ [Instance {i}] CP 해 없음 → 스킵")
                continue

            obj_val, start_times_sol, assigned_teams_sol, status = result
            print(f"     [Instance {i}] CP objective: {obj_val:.4f}, status: {status}")

            # 3. 단일 인스턴스 추출
            problem_single, env_params_single = extract_single_instance(problem, env_params, i)

            # 4. 환경에서 replay → (state, action) 수집
            pairs = replay_cp_solution(
                problem_single, env_params_single,
                start_times_sol, assigned_teams_sol
            )

            all_pairs.extend(pairs)
            success_count += 1
            print(f"     ✅ [Instance {i}] {len(pairs)} pairs collected")

        except Exception as e:
            print(f"     ❌ [Instance {i}] 오류: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_pairs, success_count


class ILDataset(Dataset):
    """
    Imitation Learning 데이터셋
    수집된 (state, optimal_action) 쌍을 mini-batch로 제공
    """

    def __init__(self, data_path):
        """
        Args:
            data_path: pickle 파일 경로 (collect_il_data로 저장된 파일)
        """
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.samples = data['pairs']
        self.env_params = data.get('env_params', None)
        print(f"✅ IL Dataset 로드 완료: {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['state'], sample['action']

    @staticmethod
    def collate_fn(batch):
        """
        DataLoader용 collate function: list of (state_dict, action) → batched tensors

        Returns:
            batched_state: dict of batched tensors (각 키마다 (B, ...) 형태)
            actions: (B,) LongTensor
        """
        states, actions = zip(*batch)

        # State dict의 각 키를 stack
        batched_state = {}
        keys = states[0].keys()
        for key in keys:
            tensors = [s[key] for s in states]
            if tensors[0] is None:
                batched_state[key] = None
            else:
                batched_state[key] = torch.stack(tensors, dim=0)

        actions_tensor = torch.tensor(actions, dtype=torch.long)

        return batched_state, actions_tensor

    def get_dataloader(self, batch_size, shuffle=True):
        """DataLoader 생성"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=False,
        )
