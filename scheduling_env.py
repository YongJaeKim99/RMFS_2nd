"""
RCMPSP (Resource-Constrained Multi-Project Scheduling Problem) 환경
DES (Discrete Event Simulation) 기반
MDP Action: 현재 시작 가능한 (Activity, Team) 페어 선택
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
import copy
from typing import List, Dict, Tuple, Optional

from data_generator import generate_scheduling_data_batch


class SchedulingEnv:
    """
    RCMPSP 환경 클래스
    
    State:
        - Activities: duration, project_id, eligible_teams, predecessors, mutex
        - Projects: release_time, due_date, completion_time
        - Teams: available_time
    
    Action:
        - (activity_id, team_id) 페어 선택
        - 현재 시작 가능한 페어만 선택 가능
    
    DES Events:
        - Activity 시작: 팀이 activity를 시작
        - Activity 종료: activity 완료 및 후속 activity 활성화
    
    Objective:
        - Tardiness: sum of max(0, completion_time - due_date) for all projects
        - Makespan: max completion_time across all projects
    """
    
    def __init__(self, env_params, debug_env=False, device='cpu'):
        """
        환경 초기화
        
        Args:
            env_params: 환경 파라미터 딕셔너리
            debug_env: 디버그 모드 활성화
            device: 텐서를 생성할 device ('cpu' 또는 'cuda')
        """
        self.env_params = env_params
        self.debug_env = debug_env
        self.device = torch.device(device)
        
        # 문제 파라미터
        self.batch_size = env_params['batch_size'] * env_params['pomo_size']
        self.batch = env_params['batch_size']
        self.pomo = env_params['pomo_size']
        
        self.N_P = env_params['N_P']  # 프로젝트 수
        self.N_A_min = env_params['N_A_min']  # 프로젝트당 최소 activity 수
        self.N_A_max = env_params['N_A_max']  # 프로젝트당 최대 activity 수
        self.N_T = env_params['N_T']  # 팀 수
        self.duration_min = env_params['duration_min']
        self.duration_max = env_params['duration_max']
        
        # 목적함수
        self.objective = env_params.get('objective', 'tardiness')  # 'tardiness' or 'makespan'
        
        # 상태 출력 모드: 'pyg' (GNN 모델) 또는 'daniel' (DANIEL 모델)
        self.state_mode = env_params.get('state_mode', 'pyg')
        
        # 문제 데이터 (reset에서 초기화)
        self.problem = None
        self.max_N_A = None  # 최대 activity 수
        self.num_activities = None  # 각 배치의 실제 activity 수
    
    def _reset(self, problem=None):
        """
        환경 리셋 및 문제 데이터 로드
        
        Args:
            problem: 외부에서 생성된 문제 데이터 (None이면 내부에서 생성)
        """
        if problem is None:
            # 새로운 문제 생성
            self.problem = generate_scheduling_data_batch(self.env_params)
        else:
            # 외부에서 전달된 문제 사용 (validation 등)
            self.problem = problem
        
        # 문제 파라미터 추출 (device로 이동)
        self.max_N_A = self.problem['env_params']['max_N_A']
        self.num_activities = self.problem['num_activities'].to(self.device)  # (batch_size,)
        
        # 배치 인덱스
        self.BATCH_IDX = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        
        # ========================================
        # Activity 상태 (Static + Dynamic)
        # ========================================
        # Static (device로 이동)
        self.activity_duration = self.problem['activity_duration'].to(self.device)  # (batch_size, max_N_A)
        self.activity_project = self.problem['activity_project'].to(self.device)  # (batch_size, max_N_A)
        self.activity_eligible_teams = self.problem['activity_eligible_teams'].to(self.device)  # (batch_size, max_N_A, N_T)
        self.activity_predecessors = self.problem['activity_predecessors'].to(self.device)  # (batch_size, max_N_A, max_preds)
        self.activity_mutex = self.problem['activity_mutex'].to(self.device)  # (batch_size, max_N_A, max_mutex)
        
        # Dynamic (device에서 생성)
        self.activity_started = torch.zeros(self.batch_size, self.max_N_A, dtype=torch.bool, device=self.device)  # 시작 여부
        self.activity_ended = torch.zeros(self.batch_size, self.max_N_A, dtype=torch.bool, device=self.device)  # 종료 여부
        self.activity_start_time = torch.full((self.batch_size, self.max_N_A), -1.0, device=self.device)  # 시작 시간 (절대 시간)
        self.activity_end_time = torch.full((self.batch_size, self.max_N_A), -1.0, device=self.device)  # 종료 시간 (절대 시간)
        self.activity_remaining_time = self.activity_duration.clone()  # 남은 시간: 초기값 = duration
        self.activity_assigned_team = torch.full((self.batch_size, self.max_N_A), -1, dtype=torch.long, device=self.device)  # 할당된 팀
        
        # 패딩된 activity들은 started와 ended를 True로 설정, remaining_time은 0으로 (벡터 연산)
        activity_indices = torch.arange(self.max_N_A, device=self.device).unsqueeze(0)  # (1, max_N_A)
        num_activities_expanded = self.num_activities.unsqueeze(1)  # (batch_size, 1)
        padding_mask = activity_indices >= num_activities_expanded  # (batch_size, max_N_A)
        
        self.activity_started[padding_mask] = True
        self.activity_ended[padding_mask] = True
        self.activity_remaining_time[padding_mask] = 0.0
        
        # ========================================
        # Project 상태 (Static + Dynamic)
        # ========================================
        # Static (device로 이동)
        self.project_release_time = self.problem['project_release_time'].to(self.device)  # (batch_size, N_P)
        self.project_due_date = self.problem['project_due_date'].to(self.device)  # (batch_size, N_P)
        
        # Dynamic (device에서 생성)
        # project_completion_time은 _get_obj()에서 필요할 때 계산 (중복 저장 안 함)
        self.project_completed = torch.zeros(self.batch_size, self.N_P, dtype=torch.bool, device=self.device)  # 완료 여부
        
        # ========================================
        # Team 상태 (Dynamic)
        # ========================================
        self.team_available_time = torch.zeros(self.batch_size, self.N_T, device=self.device)  # 각 팀이 사용 가능한 시간
        self.team_current_activity = torch.full((self.batch_size, self.N_T), -1, dtype=torch.long, device=self.device)  # 현재 수행 중인 activity
        
        # ========================================
        # Simulation 상태
        # ========================================
        self.sim_time = torch.zeros(self.batch_size, device=self.device)  # 현재 시뮬레이션 시간
        self.step_count = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)  # 스텝 카운터
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)  # 종료 여부
        
        # ========================================
        # 가능한 액션 (Action Mask) - Eligible 기반
        # ========================================
        # Action space 초기화: eligible한 (activity, team) 조합만 고려
        self._initialize_action_space()
        
        # Static edge 사전 구축 (PyG 모드에서 사용)
        self._precompute_static_edges()
        
        # (batch_size, max_action_space) - 가능한 action은 True
        self.available_actions = torch.zeros(self.batch_size, self.max_action_space, dtype=torch.bool, device=self.device)        
        
        # ========================================
        # 초기 가능한 액션 업데이트
        # ========================================
        self._update_available_actions(self.BATCH_IDX)
    
    def _initialize_action_space(self):
        """
        각 배치별로 eligible한 (activity, team) 조합만 추출하여 action space 구성
        배치 내 최대 action space 크기에 맞춰 패딩 적용 (완전 벡터화 -- for문 0개)
        """
        B = self.batch_size
        N_T = self.N_T
        
        # Valid activity mask (패딩 제외)
        act_indices = torch.arange(self.max_N_A, device=self.device).unsqueeze(0)  # (1, max_N_A)
        valid_act = act_indices < self.num_activities.unsqueeze(1)  # (B, max_N_A)
        
        # Eligible mask: valid activity AND eligible team
        eligible = self.activity_eligible_teams & valid_act.unsqueeze(2)  # (B, max_N_A, N_T)
        
        # Flatten to (B, max_N_A * N_T) -- flat_idx = act_id * N_T + team_id
        eligible_flat = eligible.reshape(B, -1)  # (B, max_N_A * N_T)
        
        # Max action space = max eligible pairs across batches
        action_counts = eligible_flat.sum(dim=1)  # (B,)
        self.max_action_space = max(action_counts.max().item(), 1)  # 최소 1
        
        # Initialize action_to_pair with -1 padding
        self.action_to_pair = torch.full(
            (B, self.max_action_space, 2), -1, dtype=torch.long, device=self.device
        )
        
        # Get all (batch, flat_index) positions where eligible
        batch_idx, flat_idx = eligible_flat.nonzero(as_tuple=True)
        
        if len(batch_idx) == 0:
            return
        
        # Convert flat index -> (activity_id, team_id)
        act_ids = flat_idx // N_T
        team_ids = flat_idx % N_T
        
        # Compute per-batch sequential index (vectorized cumcount)
        # batch_idx is sorted since nonzero preserves row-major order
        batch_counts = torch.bincount(batch_idx, minlength=B)  # (B,)
        batch_starts = torch.zeros(B, dtype=torch.long, device=self.device)
        if B > 1:
            batch_starts[1:] = batch_counts[:-1].cumsum(0)
        
        global_idx = torch.arange(len(batch_idx), dtype=torch.long, device=self.device)
        seq_idx = global_idx - batch_starts[batch_idx]
        
        # Scatter into action_to_pair
        self.action_to_pair[batch_idx, seq_idx, 0] = act_ids
        self.action_to_pair[batch_idx, seq_idx, 1] = team_ids
    
    def _precompute_static_edges(self):
        """
        _reset() 시점에 static 그래프 구조(edge_index, edge_type)를 사전 구축.
        에피소드 중 변하지 않는 엣지 구조를 미리 계산하여 _get_state_pyg()에서 재사용.
        
        각 배치별로 다음 엣지 타입을 사전 구축:
            0 = Precedence (pred -> act, 단방향)
            1 = Mutex (act <-> act, 양방향)
            2 = Eligible (act <-> team, 양방향)
            3 = Belongs-to (act <-> project, 양방향)
        """
        B = self.batch_size
        self.static_edges = []
        
        for b in range(B):
            num_act = self.num_activities[b].item()
            edges_src = []
            edges_dst = []
            edge_types = []  # 0=precedence, 1=mutex, 2=eligible, 3=belongs_to
            
            # --- 동적 edge_attr 계산을 위한 메타데이터 ---
            mutex_act1_list = []
            mutex_act2_list = []
            eligible_act_list = []
            eligible_team_list = []
            
            # 1. Precedence edges: pred_id -> act_id (단방향)
            preds = self.activity_predecessors[b, :num_act]  # (num_act, max_preds)
            valid_pred = preds >= 0
            act_coords, slot_coords = valid_pred.nonzero(as_tuple=True)
            if len(act_coords) > 0:
                pred_ids = preds[act_coords, slot_coords]
                in_range = pred_ids < num_act
                if in_range.any():
                    src = pred_ids[in_range]
                    dst = act_coords[in_range]
                    edges_src.append(src)
                    edges_dst.append(dst)
                    edge_types.append(torch.zeros(len(src), dtype=torch.long, device=self.device))
            
            # 2. Mutex edges: 양방향 (deduplicated by act1 < act2)
            mutex_data = self.activity_mutex[b, :num_act]  # (num_act, max_mutex)
            valid_mutex = mutex_data >= 0
            act_coords_m, slot_coords_m = valid_mutex.nonzero(as_tuple=True)
            if len(act_coords_m) > 0:
                mutex_ids = mutex_data[act_coords_m, slot_coords_m]
                dedup = (mutex_ids > act_coords_m) & (mutex_ids < num_act)
                if dedup.any():
                    a1 = act_coords_m[dedup]
                    a2 = mutex_ids[dedup]
                    edges_src.append(torch.cat([a1, a2]))
                    edges_dst.append(torch.cat([a2, a1]))
                    n_pairs = dedup.sum().item()
                    edge_types.append(torch.ones(n_pairs * 2, dtype=torch.long, device=self.device))
                    mutex_act1_list.append(torch.cat([a1, a2]))
                    mutex_act2_list.append(torch.cat([a2, a1]))
            
            # 3. Eligible edges: 양방향 (act <-> team_node)
            elig = self.activity_eligible_teams[b, :num_act]  # (num_act, N_T)
            act_coords_e, team_coords_e = elig.nonzero(as_tuple=True)
            if len(act_coords_e) > 0:
                team_node_ids = (num_act + team_coords_e).long()
                edges_src.append(torch.cat([act_coords_e, team_node_ids]))
                edges_dst.append(torch.cat([team_node_ids, act_coords_e]))
                n_elig = len(act_coords_e) * 2
                edge_types.append(torch.full((n_elig,), 2, dtype=torch.long, device=self.device))
                eligible_act_list.append(torch.cat([act_coords_e, act_coords_e]))
                eligible_team_list.append(torch.cat([team_coords_e, team_coords_e]))
            
            # 4. Belongs-to edges: 양방향 (act <-> proj_node)
            proj_ids = self.activity_project[b, :num_act]  # (num_act,)
            act_ids_bt = torch.arange(num_act, device=self.device)
            proj_node_ids = (num_act + self.N_T + proj_ids).long()
            edges_src.append(torch.cat([act_ids_bt, proj_node_ids]))
            edges_dst.append(torch.cat([proj_node_ids, act_ids_bt]))
            edge_types.append(torch.full((num_act * 2,), 3, dtype=torch.long, device=self.device))
            
            # Concatenate all edges
            if edges_src:
                all_src = torch.cat(edges_src)
                all_dst = torch.cat(edges_dst)
                edge_index = torch.stack([all_src, all_dst], dim=0).long()  # (2, E)
                edge_type = torch.cat(edge_types)  # (E,)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_type = torch.empty(0, dtype=torch.long, device=self.device)
            
            # Mutex metadata
            mutex_act1 = torch.cat(mutex_act1_list) if mutex_act1_list else torch.empty(0, dtype=torch.long, device=self.device)
            mutex_act2 = torch.cat(mutex_act2_list) if mutex_act2_list else torch.empty(0, dtype=torch.long, device=self.device)
            
            # Eligible metadata
            eligible_act = torch.cat(eligible_act_list) if eligible_act_list else torch.empty(0, dtype=torch.long, device=self.device)
            eligible_team = torch.cat(eligible_team_list) if eligible_team_list else torch.empty(0, dtype=torch.long, device=self.device)
            
            self.static_edges.append({
                'edge_index': edge_index,
                'edge_type': edge_type,
                'num_nodes': num_act + self.N_T + self.N_P,
                'num_act': num_act,
                'mutex_act1': mutex_act1,
                'mutex_act2': mutex_act2,
                'eligible_act': eligible_act,
                'eligible_team': eligible_team,
            })
    
    def _update_available_actions(self, batch_idxs):
        """
        가능한 모든 액션의 feasibility를 계산하여 self.available_actions에 저장 (완전 벡터화 -- for문 0개)
        gather + clamp(min=0) 패턴으로 -1 패딩을 안전하게 처리
        
        Args:
            batch_idxs: (n_active,) - 업데이트할 배치 인덱스들
        """
        B = len(batch_idxs)
        A = self.max_action_space
        
        # 지정된 배치들의 모든 액션 마스크를 False로 초기화
        self.available_actions[batch_idxs] = False
        
        # action_to_pair에서 act_id와 team_id 추출
        atp = self.action_to_pair[batch_idxs]  # (B, A, 2)
        act_ids = atp[:, :, 0]  # (B, A)
        team_ids = atp[:, :, 1]  # (B, A)
        
        # Valid action mask (non-padding, act_id >= 0)
        valid = act_ids >= 0  # (B, A)
        
        # Safe indices for gather (clamp -1 to 0, will be masked out by 'valid')
        safe_act = act_ids.clamp(min=0)  # (B, A)
        safe_team = team_ids.clamp(min=0)  # (B, A)
        
        # Current time
        current_times = self.sim_time[batch_idxs].unsqueeze(1)  # (B, 1)
        
        # 1. Unstarted check -- gather activity_started at action's activity
        started = self.activity_started[batch_idxs]  # (B, max_N_A)
        unstarted = ~torch.gather(started, 1, safe_act)  # (B, A)
        
        # 2. Project release time check
        proj_ids = torch.gather(self.activity_project[batch_idxs], 1, safe_act)  # (B, A)
        release_times = torch.gather(self.project_release_time[batch_idxs], 1, proj_ids)  # (B, A)
        released = release_times <= current_times  # (B, A)
        
        # 3. Team availability check
        team_avail_times = torch.gather(self.team_available_time[batch_idxs], 1, safe_team)  # (B, A)
        team_ok = team_avail_times <= current_times  # (B, A)
        
        # Basic feasibility
        basic = valid & unstarted & released & team_ok  # (B, A)
        
        # 4. Predecessor check (vectorized with gather)
        preds = self.activity_predecessors[batch_idxs]  # (B, max_N_A, max_preds)
        max_preds = preds.shape[2]
        
        # Gather predecessors for each action's activity: preds[b, safe_act[b,a], :]
        act_exp_p = safe_act.unsqueeze(2).expand(-1, -1, max_preds)  # (B, A, max_preds)
        action_preds = torch.gather(preds, 1, act_exp_p)  # (B, A, max_preds)
        
        # Check if all valid predecessors are ended
        valid_pred_mask = action_preds >= 0  # (B, A, max_preds)
        safe_preds = action_preds.clamp(min=0)  # (B, A, max_preds)
        
        ended = self.activity_ended[batch_idxs]  # (B, max_N_A)
        pred_ended = torch.gather(
            ended, 1, safe_preds.reshape(B, -1)
        ).reshape(B, A, max_preds)  # (B, A, max_preds)
        
        # Invalid preds (padding=-1) count as ended
        pred_ok = (pred_ended | ~valid_pred_mask).all(dim=2)  # (B, A)
        
        # 5. Mutex check (vectorized with gather)
        mutex = self.activity_mutex[batch_idxs]  # (B, max_N_A, max_mutex)
        max_mutex = mutex.shape[2]
        
        act_exp_m = safe_act.unsqueeze(2).expand(-1, -1, max_mutex)  # (B, A, max_mutex)
        action_mutex = torch.gather(mutex, 1, act_exp_m)  # (B, A, max_mutex)
        
        valid_mutex_mask = action_mutex >= 0  # (B, A, max_mutex)
        safe_mutex = action_mutex.clamp(min=0)  # (B, A, max_mutex)
        
        mutex_started = torch.gather(
            started, 1, safe_mutex.reshape(B, -1)
        ).reshape(B, A, max_mutex)  # (B, A, max_mutex)
        mutex_ended = torch.gather(
            ended, 1, safe_mutex.reshape(B, -1)
        ).reshape(B, A, max_mutex)  # (B, A, max_mutex)
        
        # Running = started AND not ended AND valid mutex entry
        mutex_running = mutex_started & ~mutex_ended & valid_mutex_mask  # (B, A, max_mutex)
        mutex_ok = ~mutex_running.any(dim=2)  # (B, A)
        
        # Combine all conditions
        self.available_actions[batch_idxs] = basic & pred_ok & mutex_ok
    
    def step(self, action):
        """
        Action 수행 및 시뮬레이션 진행 (벡터화 버전)
        
        Args:
            action: (batch_size,) - 각 배치의 선택된 action 인덱스
                    action_to_pair[batch, action] → (activity_id, team_id)
        
        Returns:
            state: 다음 상태 (Graph 형태)
            obj_value: (batch_size,) - 목적함수값 (에피소드 끝날 때만, 아니면 None)
            done: (batch_size,) - 종료 여부
        """
        # Action index를 (activity_id, team_id)로 변환
        batch_indices = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        activity_id = self.action_to_pair[batch_indices, action, 0]
        team_id = self.action_to_pair[batch_indices, action, 1]        
        
        # Activity 스케줄링
        active_batch_idxs = torch.arange(self.batch_size, device=self.device)[~self.done]
        if len(active_batch_idxs) > 0:
            self._schedule_activity(active_batch_idxs, activity_id, team_id)
        
        all_done = self.move_next_state(self.BATCH_IDX)

        if all_done:
            obj_value = self._get_obj()  # 목적함수값 반환 (양수)
            return None, obj_value, True
        else:
            self.state = self._get_state()
            return self.state, None, False
    
    def _schedule_activity(self, batch_idxs, activity_ids, team_ids):
        """
        Activity를 팀에 할당하고 시작
        
        Args:
            batch_idxs: (n_active,) - 처리할 배치 인덱스들
            activity_ids: (batch_size,) - 각 배치의 activity ID
            team_ids: (batch_size,) - 각 배치의 team ID
        """
        if len(batch_idxs) == 0:
            return
        
        # 시작 시간 = 현재 시뮬레이션 시간
        # Feasible action에서 이미 모든 제약(선행작업, mutex, 팀 가용, 프로젝트 release)을 체크했으므로
        # 선택된 activity는 즉시 시작 가능함이 보장됨
        start_times = self.sim_time[batch_idxs]  # (n_active,)
        
        # 종료 시간 계산
        end_times = start_times + self.activity_duration[batch_idxs, activity_ids[batch_idxs]]  # (n_active,)
        
        # Duration 추출
        durations = self.activity_duration[batch_idxs, activity_ids[batch_idxs]]  # (n_active,)
        
        # Activity 상태 업데이트
        self.activity_started[batch_idxs, activity_ids[batch_idxs]] = True
        self.activity_start_time[batch_idxs, activity_ids[batch_idxs]] = start_times
        self.activity_end_time[batch_idxs, activity_ids[batch_idxs]] = end_times
        self.activity_remaining_time[batch_idxs, activity_ids[batch_idxs]] = durations  # 남은 시간 설정
        self.activity_assigned_team[batch_idxs, activity_ids[batch_idxs]] = team_ids[batch_idxs]
        
        # 팀 상태 업데이트
        self.team_available_time[batch_idxs, team_ids[batch_idxs]] = end_times
        self.team_current_activity[batch_idxs, team_ids[batch_idxs]] = activity_ids[batch_idxs] 

    def move_next_state(self, batch_idxs):
        """
        시뮬레이터 로직에 따라 다음 의사결정 이벤트까지 시간 진행 (벡터화)
        배치 상태 분류를 all/any 배치 연산으로 수행 (for문 0개)
        
        Args:
            batch_idxs: 처리할 배치 인덱스들
            
        Returns:
            bool: all_done - 모든 배치의 activity가 완료되었는지 여부
        """
        max_iterations = 1000  # 무한 루프 방지
        iterations = 0
        
        # 루프 시작 전 초기 액션 업데이트
        self._update_available_actions(batch_idxs)
        
        while iterations < max_iterations:
            iterations += 1
            
            # 벡터화된 배치 상태 분류
            all_ended = self.activity_ended[batch_idxs].all(dim=1)     # (n,)
            all_started = self.activity_started[batch_idxs].all(dim=1)  # (n,)
            has_actions = self.available_actions[batch_idxs].any(dim=1)  # (n,)
            
            # 모든 배치 완료 체크
            if all_ended.all():
                break
            
            # 시간 진행이 필요한 배치 결정 (벡터 연산)
            # 1. 모든 activity가 시작됨 & 아직 다 끝나지 않음 -> 시간 진행
            # 2. 아직 끝나지 않음 & 가능한 액션 없음 -> 시간 진행
            needs_advance = (all_started & ~all_ended) | (~all_ended & ~has_actions)
            
            if not needs_advance.any():
                # 모든 활성 배치가 의사결정 가능 상태 -> 중단
                break
            
            advance_idxs = batch_idxs[needs_advance]
            self._advance_to_next_decision_event_for_batches(advance_idxs)
            self._update_available_actions(advance_idxs)
        
        # 최대 반복 도달 시 디버그 출력
        if iterations >= max_iterations:
            print(f"\n❌ [move_next_state] 최대 반복 횟수({max_iterations}) 도달!")
        
        # 최종 완료 체크 (벡터화)
        final_all_done = self.activity_ended[batch_idxs].all().item()
        
        return final_all_done

    def _advance_to_next_decision_event_for_batches(self, batch_idxs):
        """
        지정된 배치들에서 시간을 다음 이벤트까지 진행하고 완료된 activity 처리 (벡터 연산)
        
        Args:
            batch_idxs: (n_batches,) - 처리할 배치 인덱스들
        """
        # 다음 이벤트까지의 시간 계산
        time_deltas = self.get_next_move_t(batch_idxs)  # (n_batches,)
        
        # 시뮬레이션 시간 진행
        self.sim_time[batch_idxs] += time_deltas
        
        # Activity 남은 시간 감소 (벡터 연산)
        self.activity_remaining_time[batch_idxs] = torch.clamp(
            self.activity_remaining_time[batch_idxs] - time_deltas.unsqueeze(1), min=0)
        
        # 완료된 activity 처리 (남은 시간이 0이고 시작했지만 아직 완료 안 된 것)
        just_completed_mask = (
            (self.activity_remaining_time[batch_idxs] <= 0) & 
            self.activity_started[batch_idxs] & 
            ~self.activity_ended[batch_idxs]
        )  # (n_batches, max_N_A)
        
        # _complete_activity 함수 호출 (벡터 연산으로 완료 처리)
        self._complete_activity(batch_idxs, just_completed_mask)
                

    def get_next_move_t(self, batch_idxs):
        """
        지정된 배치들에서 다음 이벤트까지의 시간을 배치별로 계산 (벡터 연산)
        
        Args:
            batch_idxs: (n_batches,) - 처리할 배치 인덱스들
        
        Returns:
            time_deltas: (n_batches,) - 각 배치의 다음 이벤트까지의 시간 차이
        """
        # 진행 중인 activity들의 남은 시간 가져오기 (batch_idxs, max_N_A)
        remaining_times = self.activity_remaining_time[batch_idxs]  # (n_batches, max_N_A)
        
        # 0보다 큰 시간들만 고려 (진행 중인 activity만)
        # 0 이하인 값들을 매우 큰 값으로 대체
        masked_times = torch.where(remaining_times > 0, remaining_times, torch.tensor(float('inf'), device=self.device))
        
        # 각 배치별 최소값 계산 (가장 먼저 완료될 activity의 남은 시간)
        batch_mins, min_indices = masked_times.min(dim=1)  # (n_batches,)
        
        # inf인 경우 (진행 중인 activity가 없음) 0.0으로 대체
        batch_mins = torch.where(batch_mins == float('inf'), torch.tensor(0.0, device=self.device), batch_mins)        
        
        return batch_mins

    def _complete_activity(self, batch_idxs, just_completed_mask):
        """
        완료된 activity들을 벡터 연산으로 처리
        
        Args:
            batch_idxs: (n_batches,) - 처리할 배치 인덱스들
            just_completed_mask: (n_batches, max_N_A) - 방금 완료된 activity 마스크
        """
        # 1. Activity completed 상태 업데이트 (완전 벡터)
        self.activity_ended[batch_idxs] |= just_completed_mask
        
        # 2. Team current_activity 업데이트 (완전 벡터 - Advanced Indexing)
        # 완료된 activity들의 (batch, activity) 좌표 찾기
        relative_batch_coords, act_coords = just_completed_mask.nonzero(as_tuple=True)
        
        if len(relative_batch_coords) == 0:
            return  # 완료된 activity가 없으면 종료
        
        # 상대 배치 인덱스 → 절대 배치 인덱스 변환
        absolute_batch_coords = batch_idxs[relative_batch_coords]
        
        # 완료된 activity들의 팀 ID 가져오기 (2D advanced indexing)
        completed_teams = self.activity_assigned_team[absolute_batch_coords, act_coords]
        
        # 팀 상태 업데이트 (2D advanced indexing)
        self.team_current_activity[absolute_batch_coords, completed_teams] = -1
        
        # 3. 프로젝트 완료 여부 확인 (완전 벡터화)
        # 각 배치의 각 프로젝트에 속한 모든 activity가 ended인지 확인
        
        activity_indices = torch.arange(self.max_N_A, device=self.device)
        project_indices = torch.arange(self.N_P, device=self.device)
        
        # 선택된 배치들만 추출
        selected_activity_project = self.activity_project[batch_idxs]  # (n_batch, max_N_A)
        selected_activity_ended = self.activity_ended[batch_idxs]  # (n_batch, max_N_A)
        selected_num_activities = self.num_activities[batch_idxs]  # (n_batch,)
        
        # 프로젝트별 activity 마스크 (n_batch, N_P, max_N_A)
        proj_mask = (selected_activity_project.unsqueeze(1) == project_indices.view(1, -1, 1))  # (n_batch, N_P, max_N_A)
        
        # 유효한 activity 마스크 (패딩 제외) (n_batch, max_N_A)
        valid_mask = activity_indices.unsqueeze(0) < selected_num_activities.unsqueeze(1)  # (n_batch, max_N_A)
        
        # 프로젝트 마스크에 유효성 적용 (n_batch, N_P, max_N_A)
        proj_mask = proj_mask & valid_mask.unsqueeze(1)
        
        # 프로젝트에 속한 activity가 모두 ended인지 확인
        # (ended | ~proj_mask)가 모두 True이면 프로젝트 완료
        proj_completed_check = selected_activity_ended.unsqueeze(1) | ~proj_mask  # (n_batch, N_P, max_N_A)
        proj_all_completed = proj_completed_check.all(dim=2)  # (n_batch, N_P)
        
        # 프로젝트 완료 상태 업데이트 (벡터 연산)
        self.project_completed[batch_idxs] = proj_all_completed
        
    def _get_obj(self):
        """
        목적함수값 계산 (에피소드 끝에 한번만 호출)
        
        Returns:
            obj: (batch_size,) - 목적함수값 (작을수록 좋음)
        """
        # 프로젝트별 완료 시간 계산 (완전 벡터화)
        # 각 프로젝트에 속한 activity의 최대 end_time
        
        # 프로젝트별 activity 마스크 (batch_size, N_P, max_N_A)
        project_indices = torch.arange(self.N_P, device=self.device)
        proj_mask = (self.activity_project.unsqueeze(1) == project_indices.view(1, -1, 1))  # (batch_size, N_P, max_N_A)
        
        # 유효한 activity 마스크 (batch_size, max_N_A)
        activity_indices = torch.arange(self.max_N_A, device=self.device)
        valid_mask = activity_indices.unsqueeze(0) < self.num_activities.unsqueeze(1)  # (batch_size, max_N_A)
        
        # 프로젝트에 속하고 유효한 activity만 선택 (batch_size, N_P, max_N_A)
        proj_valid_mask = proj_mask & valid_mask.unsqueeze(1)
        
        # end_time을 3D로 확장하고, 유효하지 않은 곳은 -inf로 처리
        end_time_3d = self.activity_end_time.unsqueeze(1).expand(-1, self.N_P, -1)  # (batch_size, N_P, max_N_A)
        masked_end_time = torch.where(
            proj_valid_mask,
            end_time_3d,
            torch.tensor(-float('inf'), device=self.device)
        )
        
        # 각 프로젝트별 최대값 (batch_size, N_P)
        project_completion_time = masked_end_time.max(dim=2)[0]
        
        # -inf는 0으로 처리 (프로젝트에 activity가 없는 경우)
        project_completion_time = torch.where(
            torch.isinf(project_completion_time),
            torch.tensor(0.0, device=self.device),
            project_completion_time
        )
        
        if self.objective == 'tardiness':
            # Total tardiness
            obj = torch.clamp(
                project_completion_time - self.project_due_date,
                min=0.0
            ).sum(dim=1)  # (batch_size,)
        
        elif self.objective == 'makespan':
            # Makespan
            obj = project_completion_time.max(dim=1)[0]  # (batch_size,)
        
        else:
            obj = torch.zeros(self.batch_size, device=self.device)
        
        return obj
    
    def _get_state(self):
        """
        현재 상태를 모델 입력 형태로 반환
        self.state_mode에 따라 PyG 또는 DANIEL 형태로 출력
        
        Returns:
            state: 모델 입력 형태의 상태 (mode에 따라 다름)
        """
        if self.state_mode == 'daniel':
            return self._get_state_daniel()
        else:
            return self._get_state_pyg()
    
    def _get_state_pyg(self):
        """
        현재 상태를 PyG (Graph) 입력 형태로 반환 (GNN 모델용)
        노드 피처는 텐서 슬라이싱으로 구성 (for문 없음), 엣지는 사전 계산된 static edge 재사용
        
        Returns:
            state: List of PyG Data objects (배치별로 하나씩)
        """
        state_list = []
        
        # 정규화용 상수 (배치 전체 한번에)
        max_duration = self.duration_max
        max_times = torch.clamp(self.sim_time + 1.0, min=1.0)  # (B,)
        max_release = self.project_release_time.max(dim=1)[0].clamp(min=1.0)  # (B,)
        max_due = self.project_due_date.max(dim=1)[0].clamp(min=1.0)  # (B,)
        
        for b in range(self.batch_size):
            info = self.static_edges[b]
            num_act = info['num_act']
            num_nodes = info['num_nodes']
            edge_index = info['edge_index']
            edge_type = info['edge_type']
            
            # ========================================
            # 노드 피처 구성 (텐서 슬라이싱 -- for문 없음)
            # ========================================
            # 전체 노드 피처: 8차원
            # Activity [0:4]: duration, started, ended, remaining_time
            # Team [4]: available_time
            # Project [5:8]: remaining_release, remaining_due, completed
            node_features = torch.zeros(num_nodes, 8, device=self.device)
            
            # Activity 노드 [0:num_act]
            node_features[:num_act, 0] = self.activity_duration[b, :num_act] / max_duration
            node_features[:num_act, 1] = self.activity_started[b, :num_act].float()
            node_features[:num_act, 2] = self.activity_ended[b, :num_act].float()
            node_features[:num_act, 3] = self.activity_remaining_time[b, :num_act] / max_duration
            
            # Team 노드 [num_act:num_act+N_T]
            node_features[num_act:num_act + self.N_T, 4] = (
                self.team_available_time[b] / max_times[b]
            )
            
            # Project 노드 [num_act+N_T:num_act+N_T+N_P]
            current_time = self.sim_time[b]
            proj_start = num_act + self.N_T
            remaining_release = torch.clamp(self.project_release_time[b] - current_time, min=0.0)
            remaining_due = torch.clamp(self.project_due_date[b] - current_time, min=0.0)
            node_features[proj_start:proj_start + self.N_P, 5] = remaining_release / max_release[b]
            node_features[proj_start:proj_start + self.N_P, 6] = remaining_due / max_due[b]
            node_features[proj_start:proj_start + self.N_P, 7] = self.project_completed[b].float()
            
            # ========================================
            # 엣지 속성 (dynamic) 계산 -- static edge 재사용
            # ========================================
            num_edges = edge_index.shape[1]
            edge_attr = torch.zeros(num_edges, 1, device=self.device)
            
            # Mutex edges: is_ordered (둘 중 하나가 started이면 1)
            mutex_mask = (edge_type == 1)
            if mutex_mask.any():
                ma1 = info['mutex_act1']
                ma2 = info['mutex_act2']
                is_ordered = (
                    self.activity_started[b, ma1] | self.activity_started[b, ma2]
                ).float()
                edge_attr[mutex_mask, 0] = is_ordered
            
            # Eligible edges: is_assigned (activity가 해당 team에 할당되었으면 1)
            elig_mask = (edge_type == 2)
            if elig_mask.any():
                ea = info['eligible_act']
                et = info['eligible_team']
                is_assigned = (self.activity_assigned_team[b, ea] == et).float()
                edge_attr[elig_mask, 0] = is_assigned
            
            # ========================================
            # PyG Data 객체 생성
            # ========================================
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mask=self.available_actions[b],
                batch_idx=b,
                num_activities=num_act,
                sim_time=self.sim_time[b].item()
            )
            state_list.append(data)
        
        return state_list
    
    def _get_state_daniel(self):
        """
        현재 상태를 DANIEL 모델 입력 형태로 반환
        
        DANIEL이 기대하는 출력:
            fea_act: (B, N, 10) - activity feature vectors
            act_mask: (B, N, 3) - predecessor/successor mask
            fea_team: (B, T, 8) - team feature vectors
            team_mask: (B, T, T) - team attention mask
            comp_idx: (B, T, T, P) - competition index
            dynamic_pair_mask: (B, P, T) - incompatible pair mask
            candidate: (B, P) - candidate activity indices
            fea_pairs: (B, P, T, 8) - pair features
        
        Note: RCMPSP는 FJSP와 문제 구조가 다르므로 (DAG 선행관계, mutex 제약 등)
              피처 설계가 별도로 필요합니다.
        
        Returns:
            EnvState 객체
        """
        raise NotImplementedError(
            "DANIEL 모델용 상태 출력은 아직 구현되지 않았습니다. "
            "RCMPSP에 맞는 피처 설계(fea_act, comp_idx 등)가 필요합니다. "
            "fjsp_env_same_op_nums.py의 construct_*_features()를 참고하여 구현하세요."
        )