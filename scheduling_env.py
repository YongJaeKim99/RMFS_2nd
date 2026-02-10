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
        
        # (batch_size, max_action_space) - 가능한 action은 True
        self.available_actions = torch.zeros(self.batch_size, self.max_action_space, dtype=torch.bool, device=self.device)        
        
        # ========================================
        # 초기 가능한 액션 업데이트
        # ========================================
        self._update_available_actions(self.BATCH_IDX)
    
    def _initialize_action_space(self):
        """
        각 배치별로 eligible한 (activity, team) 조합만 추출하여 action space 구성
        배치 내 최대 action space 크기에 맞춰 패딩 적용
        """
        # 각 배치별로 eligible한 action 목록 생성
        action_space_sizes = []
        action_mappings = []  # 각 배치의 (action_idx → (activity, team)) 매핑
        
        for b in range(self.batch_size):
            eligible_actions = []  # [(activity_id, team_id), ...]
            num_act = self.num_activities[b].item()
            
            for act_id in range(num_act):
                # 해당 activity의 eligible teams 찾기
                eligible_teams = self.activity_eligible_teams[b, act_id]  # (N_T,)
                eligible_team_ids = eligible_teams.nonzero(as_tuple=False).squeeze(-1)  # eligible한 team ID들
                
                for team_id in eligible_team_ids:
                    eligible_actions.append((act_id, team_id.item()))
            
            action_space_sizes.append(len(eligible_actions))
            action_mappings.append(eligible_actions)
        
        # 배치 내 최대 action space 크기
        self.max_action_space = max(action_space_sizes)
        
        # Action index → (activity, team) 매핑 텐서 생성 (패딩 포함)
        # shape: (batch_size, max_action_space, 2) where [:, :, 0] = activity_id, [:, :, 1] = team_id
        self.action_to_pair = torch.full((self.batch_size, self.max_action_space, 2), -1, dtype=torch.long, device=self.device)
        
        for b in range(self.batch_size):
            for action_idx, (act_id, team_id) in enumerate(action_mappings[b]):
                self.action_to_pair[b, action_idx, 0] = act_id
                self.action_to_pair[b, action_idx, 1] = team_id
    
    def _update_available_actions(self, batch_idxs):
        """
        가능한 모든 액션을 self 변수에 저장 (벡터화)
        현재 시점에서 시작 가능한 (activity, team) 페어를 계산하여 self.available_actions에 저장
        
        Args:
            batch_idxs: (n_active,) - 업데이트할 배치 인덱스들
        """
        # 지정된 배치들의 모든 액션 마스크를 False로 초기화
        self.available_actions[batch_idxs] = False
        
        for b in batch_idxs:
            b_item = b.item() if torch.is_tensor(b) else b
            
            current_time = self.sim_time[b_item].item()
            num_act = self.num_activities[b_item].item()
            
            # 벡터화: action_to_pair에서 act_id와 team_id 추출
            action_pairs = self.action_to_pair[b_item]  # (max_action_space, 2)
            act_ids = action_pairs[:, 0]  # (max_action_space,)
            team_ids = action_pairs[:, 1]  # (max_action_space,)
            
            # 패딩된 action 마스크 (act_id >= 0인 것만 유효)
            valid_action_mask = act_ids >= 0  # (max_action_space,)
            
            # 초기 feasibility: False로 시작
            feasible = torch.zeros(self.max_action_space, dtype=torch.bool, device=self.device)
            
            # 유효한 action만 처리
            valid_act_ids = act_ids[valid_action_mask]
            valid_team_ids = team_ids[valid_action_mask]
            
            if len(valid_act_ids) == 0:
                continue
            
            # 1. 시작 안 된 activity 체크 (벡터화)
            unstarted = ~self.activity_started[b_item, valid_act_ids]  # (num_valid_actions,)
            
            # 2. 프로젝트 release 시간 체크 (벡터화)
            proj_ids = self.activity_project[b_item, valid_act_ids]
            project_released = self.project_release_time[b_item, proj_ids] <= current_time
            
            # 3. 팀 가용성 체크 (벡터화)
            team_available = self.team_available_time[b_item, valid_team_ids] <= current_time
            
            # 기본 조건 결합
            basic_feasible = unstarted & project_released & team_available  # (num_valid_actions,)
            
            # 4. Predecessor 체크 (loop 필요)
            pred_feasible = torch.ones(len(valid_act_ids), dtype=torch.bool, device=self.device)
            for i, act_id in enumerate(valid_act_ids):
                if not basic_feasible[i]:
                    pred_feasible[i] = False
                    continue
                predecessors = self.activity_predecessors[b_item, act_id]
                valid_preds = predecessors[predecessors >= 0]
                if len(valid_preds) > 0:
                    if not self.activity_ended[b_item, valid_preds].all():
                        pred_feasible[i] = False
            
            # 5. Mutex 체크 (loop 필요)
            mutex_feasible = torch.ones(len(valid_act_ids), dtype=torch.bool, device=self.device)
            for i, act_id in enumerate(valid_act_ids):
                if not (basic_feasible[i] and pred_feasible[i]):
                    mutex_feasible[i] = False
                    continue
                mutex_activities = self.activity_mutex[b_item, act_id]
                valid_mutex = mutex_activities[mutex_activities >= 0]
                if len(valid_mutex) > 0:
                    mutex_running = (self.activity_started[b_item, valid_mutex] & 
                                   ~self.activity_ended[b_item, valid_mutex]).any()
                    if mutex_running:
                        mutex_feasible[i] = False
            
            # 모든 조건 결합
            final_feasible = basic_feasible & pred_feasible & mutex_feasible
            
            # 결과를 available_actions에 저장
            feasible[valid_action_mask] = final_feasible
            self.available_actions[b_item] = feasible
    
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
        시뮬레이터 로직에 따라 다음 의사결정 이벤트까지 시간 진행
        배치별로 처리하며, 의사결정이 필요한 시점에서 멈춤
        
        Args:
            batch_idxs: 처리할 배치 인덱스들
            
        Returns:
            bool: all_done - 모든 배치의 activity가 완료되었는지 여부
        """        
        max_iterations = 1000  # 무한 루프 방지
        iterations = 0

        # 처리할 배치들 마스킹
        batch_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        batch_mask[batch_idxs] = True

        # 루프 시작 전 초기 액션 업데이트
        self._update_available_actions(batch_idxs)

        while iterations < max_iterations:
            iterations += 1
            
            # 각 배치별로 activity 완료 상태 확인
            batches_started = []  # 모든 activity가 started된 배치들
            batches_not_started = []  # 모든 activity가 아직 started 안된 배치들
            batches_ended = []  # 모든 activity가 ended된 배치들
            
            for b in batch_idxs:
                b_item = b.item() if torch.is_tensor(b) else b
                activity_started = self.activity_started[b_item].all().item()
                activity_ended = self.activity_ended[b_item].all().item()
                
                if activity_ended:
                    batches_ended.append(b_item)
                elif activity_started:
                    batches_started.append(b_item)
                else:
                    batches_not_started.append(b_item)
            
            all_done = (len(batches_ended) == len(batch_idxs))

            if all_done:
                break
            
            # 시간을 진행해야 할 배치들 결정
            batches_to_advance = []
            
            # 1. started된 배치들 (ended될 때 까지 시간 진행)
            batches_to_advance.extend(batches_started)
            
            # 2. started 안된 배치들 중 가능한 액션이 없는 배치들 (다음 의사결정까지 시간 진행)
            for b_idx in batches_not_started:
                if not self.available_actions[b_idx].any().item():
                    batches_to_advance.append(b_idx)
            
            if len(batches_to_advance) > 0:
                batches_to_advance_tensor = torch.tensor(batches_to_advance, dtype=torch.long, device=self.device)
                self._advance_to_next_decision_event_for_batches(batches_to_advance_tensor)
                self._update_available_actions(batches_to_advance_tensor)
            else:
                # 시간 진행할 배치가 없으면 중단 (모든 배치가 의사결정 가능 상태)
                break

        # 루프 종료 후 정보 출력
        if iterations >= max_iterations:
            print(f"\n❌ [move_next_state] 최대 반복 횟수({max_iterations}) 도달!")
            for b in batch_idxs:
                b_item = b.item() if torch.is_tensor(b) else b
                num_started = self.activity_started[b_item].sum().item()
                num_completed = self.activity_ended[b_item].sum().item()
                num_total = self.num_activities[b_item].item()
                num_available = self.available_actions[b_item].sum().item()
                print(f"   Batch {b_item}: Started={num_started}/{num_total}, "
                      f"Completed={num_completed}/{num_total}, "
                      f"Available Actions={num_available}")
        
        final_all_done = True
        for b in batch_idxs:
            b_item = b.item() if torch.is_tensor(b) else b
            if not self.activity_ended[b_item].all().item():
                final_all_done = False
                break
        
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
        현재 상태를 GNN 입력 형태로 반환
        
        Returns:
            state: List of PyG Data objects (배치별로 하나씩)
        """
        state_list = []
        
        for b in range(self.batch_size):
            # ========================================
            # 노드 피처 구성 (패딩 방식)
            # ========================================
            # 전체 노드 피처: 8차원
            # Activity: [0:4] = [duration, started, ended, remaining_time], [4:8] = 0
            # Team: [0:4] = 0, [4] = available_time, [5:8] = 0
            # Project: [0:5] = 0, [5:8] = [release_time, due_date, completed]
            
            num_act = self.num_activities[b].item()
            num_nodes = num_act + self.N_T + self.N_P
            
            # 노드 피처 초기화 (8차원, 모두 0으로 시작)
            node_features = torch.zeros(num_nodes, 8, device=self.device)
            
            # 정규화를 위한 시간 스케일
            max_time = max(1.0, self.sim_time[b].item() + 1.0)
            max_duration = self.duration_max
            max_release_time = self.project_release_time[b].max().item()  # 배치 내 최대 release time
            max_due_date = self.project_due_date[b].max().item()  # 배치 내 최대 due date
            current_time = self.sim_time[b].item()
            
            # Activity 노드 (0 ~ num_act-1)
            for act_id in range(num_act):
                idx = act_id
                # [0:4] 사용
                node_features[idx, 0] = self.activity_duration[b, act_id] / max_duration  # duration
                node_features[idx, 1] = float(self.activity_started[b, act_id])  # started
                node_features[idx, 2] = float(self.activity_ended[b, act_id])  # ended
                node_features[idx, 3] = self.activity_remaining_time[b, act_id] / max_duration  # remaining_time
                # [4:8]은 이미 0으로 패딩됨
            
            # Team 노드 (num_act ~ num_act+N_T-1)
            for team_id in range(self.N_T):
                idx = num_act + team_id
                # [0:4]는 이미 0으로 패딩됨
                # [4] 사용
                node_features[idx, 4] = self.team_available_time[b, team_id] / max_time
                # [5:8]은 이미 0으로 패딩됨
            
            # Project 노드 (num_act+N_T ~ num_act+N_T+N_P-1)
            for proj_id in range(self.N_P):
                idx = num_act + self.N_T + proj_id
                # [0:5]는 이미 0으로 패딩됨
                # [5:8] 사용
                # release까지 남은 시간 (음수면 0)
                remaining_release = max(0.0, self.project_release_time[b, proj_id].item() - current_time)
                node_features[idx, 5] = remaining_release / max(1.0, max_release_time)
                
                # due date까지 남은 시간 (음수면 0)
                remaining_due = max(0.0, self.project_due_date[b, proj_id].item() - current_time)
                node_features[idx, 6] = remaining_due / max(1.0, max_due_date)
                
                node_features[idx, 7] = float(self.project_completed[b, proj_id])
            
            # ========================================
            # 엣지 구성 (edge_index + edge_attr)
            # ========================================
            edge_index_list = []
            edge_attr_list = []
            
            # 1. Activity → Activity (Precedence) - 엣지 피처 없음
            num_precedence = 0
            for act_id in range(num_act):
                predecessors = self.activity_predecessors[b, act_id]
                for pred_id in predecessors:
                    if pred_id >= 0 and pred_id < num_act:
                        edge_index_list.append([pred_id, act_id])
                        edge_attr_list.append([0.0])  # 구조 정보만, 더미 피처
                        num_precedence += 1
            
            # 2. Activity ↔ Activity (Mutex - 양방향) - 엣지 피처: is_ordered
            num_mutex_start = len(edge_index_list)
            for act_id in range(num_act):
                mutex_activities = self.activity_mutex[b, act_id]
                for mutex_id in mutex_activities:
                    if mutex_id >= 0 and mutex_id < num_act and mutex_id > act_id:
                        # 양방향 엣지 추가
                        # is_ordered 계산: 둘 중 하나가 시작했으면 1, 아니면 0
                        is_ordered = float(self.activity_started[b, act_id] or self.activity_started[b, mutex_id])
                        
                        edge_index_list.append([act_id, mutex_id])
                        edge_attr_list.append([is_ordered])
                        
                        edge_index_list.append([mutex_id, act_id])
                        edge_attr_list.append([is_ordered])
            
            # 3. Activity ↔ Team (Eligible - 양방향) - 엣지 피처: is_assigned
            num_eligible_start = len(edge_index_list)
            for act_id in range(num_act):
                eligible_teams = self.activity_eligible_teams[b, act_id]
                for team_id in range(self.N_T):
                    if eligible_teams[team_id]:
                        # is_assigned 계산: 이 activity가 이 team에 할당되었으면 1
                        is_assigned = float(self.activity_assigned_team[b, act_id] == team_id)
                        
                        # 양방향 엣지 추가 (둘 다 동일한 is_assigned 값)
                        edge_index_list.append([act_id, num_act + team_id])
                        edge_attr_list.append([is_assigned])
                        
                        edge_index_list.append([num_act + team_id, act_id])
                        edge_attr_list.append([is_assigned])
            
            # 4. Activity ↔ Project (Belongs-to - 양방향) - 엣지 피처 없음
            num_belongs_start = len(edge_index_list)
            for act_id in range(num_act):
                proj_id = self.activity_project[b, act_id].item()
                # 양방향 엣지 추가
                edge_index_list.append([act_id, num_act + self.N_T + proj_id])
                edge_attr_list.append([0.0])  # 구조 정보만, 더미 피처
                
                edge_index_list.append([num_act + self.N_T + proj_id, act_id])
                edge_attr_list.append([0.0])  # 구조 정보만, 더미 피처
            
            if edge_index_list:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long, device=self.device).t().contiguous()
                edge_attr = torch.tensor(edge_attr_list, dtype=torch.float, device=self.device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                edge_attr = torch.empty((0, 1), dtype=torch.float, device=self.device)
            
            # ========================================
            # Action 마스크 (가능한 action만 True)
            # ========================================
            # Eligible 기반 action space 사용 (이미 1D)
            action_mask = self.available_actions[b]  # (max_action_space,)
            
            # PyG Data 객체 생성
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,  # 엣지 피처 추가
                mask=action_mask,  # Action 마스크
                batch_idx=b,
                num_activities=num_act,
                sim_time=self.sim_time[b].item()
            )
            
            state_list.append(data)
        
        return state_list