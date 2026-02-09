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
    
    def __init__(self, env_params, debug_env=False):
        """
        환경 초기화
        
        Args:
            env_params: 환경 파라미터 딕셔너리
            debug_env: 디버그 모드 활성화
        """
        self.env_params = env_params
        self.debug_env = debug_env
        
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
        
        if self.debug_env:
            print(f"\n✅ SchedulingEnv 초기화 완료")
            print(f"   Batch Size: {self.batch_size}")
            print(f"   Projects: {self.N_P}")
            print(f"   Activities per Project: {self.N_A_min}-{self.N_A_max}")
            print(f"   Teams: {self.N_T}")
            print(f"   Objective: {self.objective}")
    
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
        
        # 문제 파라미터 추출
        self.max_N_A = self.problem['env_params']['max_N_A']
        self.num_activities = self.problem['num_activities']  # (batch_size,)
        
        # 배치 인덱스
        self.BATCH_IDX = torch.arange(self.batch_size, dtype=torch.long)
        
        # ========================================
        # Activity 상태 (Static + Dynamic)
        # ========================================
        # Static
        self.activity_duration = self.problem['activity_duration']  # (batch_size, max_N_A)
        self.activity_project = self.problem['activity_project']  # (batch_size, max_N_A)
        self.activity_eligible_teams = self.problem['activity_eligible_teams']  # (batch_size, max_N_A, N_T)
        self.activity_predecessors = self.problem['activity_predecessors']  # (batch_size, max_N_A, max_preds)
        self.activity_mutex = self.problem['activity_mutex']  # (batch_size, max_N_A, max_mutex)
        
        # Dynamic
        self.activity_started = torch.zeros(self.batch_size, self.max_N_A, dtype=torch.bool)  # 시작 여부
        self.activity_completed = torch.zeros(self.batch_size, self.max_N_A, dtype=torch.bool)  # 완료 여부
        self.activity_start_time = torch.full((self.batch_size, self.max_N_A), -1.0)  # 시작 시간 (절대 시간)
        self.activity_end_time = torch.full((self.batch_size, self.max_N_A), -1.0)  # 종료 시간 (절대 시간)
        self.activity_remaining_time = torch.zeros(self.batch_size, self.max_N_A)  # 남은 시간 (상대 시간)
        self.activity_assigned_team = torch.full((self.batch_size, self.max_N_A), -1, dtype=torch.long)  # 할당된 팀
        
        # ========================================
        # Project 상태 (Static + Dynamic)
        # ========================================
        # Static
        self.project_release_time = self.problem['project_release_time']  # (batch_size, N_P)
        self.project_due_date = self.problem['project_due_date']  # (batch_size, N_P)
        
        # Dynamic
        self.project_completion_time = torch.zeros(self.batch_size, self.N_P)  # 완료 시간
        self.project_completed = torch.zeros(self.batch_size, self.N_P, dtype=torch.bool)  # 완료 여부
        
        # ========================================
        # Team 상태 (Dynamic)
        # ========================================
        self.team_available_time = torch.zeros(self.batch_size, self.N_T)  # 각 팀이 사용 가능한 시간
        self.team_current_activity = torch.full((self.batch_size, self.N_T), -1, dtype=torch.long)  # 현재 수행 중인 activity
        
        # ========================================
        # Simulation 상태
        # ========================================
        self.sim_time = torch.zeros(self.batch_size)  # 현재 시뮬레이션 시간
        self.step_count = torch.zeros(self.batch_size, dtype=torch.long)  # 스텝 카운터
        self.done = torch.zeros(self.batch_size, dtype=torch.bool)  # 종료 여부
        
        # ========================================
        # 가능한 액션 (Action Mask)
        # ========================================
        # (batch_size, max_N_A, N_T) - 가능한 (activity, team) 페어는 True
        self.available_actions = torch.zeros(self.batch_size, self.max_N_A, self.N_T, dtype=torch.bool)
        
        # ========================================
        # 디버깅 정보
        # ========================================
        if self.debug_env:
            print(f"\n🔄 환경 리셋 완료")
            print(f"   Max Activities: {self.max_N_A}")
            print(f"   Batch 0 Activities: {self.num_activities[0].item()}")
            
            # 첫 번째 배치의 프로젝트 정보 출력
            if self.batch_size > 0:
                print(f"\n   📊 Batch 0 Projects:")
                for p in range(self.N_P):
                    release = self.project_release_time[0, p].item()
                    due = self.project_due_date[0, p].item()
                    # 프로젝트별 activity 수 계산
                    proj_activities = (self.activity_project[0] == p).sum().item()
                    print(f"      Project {p}: {proj_activities} activities, Release={release}, Due={due}")
        
        # ========================================
        # 초기 가능한 액션 업데이트
        # ========================================
        self._update_available_actions(self.BATCH_IDX)
    
    def _update_available_actions(self, batch_idxs):
        """
        가능한 모든 액션을 self 변수에 저장 (부분 벡터화)
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
            
            # 벡터화: 시작되지 않은 activity 필터링
            unstarted_mask = ~self.activity_started[b_item, :num_act]  # (num_act,)
            
            # 벡터화: 프로젝트 release 시간 체크
            proj_ids = self.activity_project[b_item, :num_act]  # (num_act,)
            project_released = self.project_release_time[b_item, proj_ids] <= current_time  # (num_act,)
            
            # 팀 가용성 체크 (벡터화)
            available_teams = self.team_available_time[b_item] <= current_time  # (N_T,)
            
            for act_id in range(num_act):
                # 기본 필터: 시작 안 됨 & 프로젝트 released
                if not (unstarted_mask[act_id] and project_released[act_id]):
                    continue
                
                # 선행 작업 완료 확인
                predecessors = self.activity_predecessors[b_item, act_id]
                valid_preds = predecessors[predecessors >= 0]
                if len(valid_preds) > 0:
                    if not self.activity_completed[b_item, valid_preds].all():
                        continue
                
                # Mutex 제약 확인
                mutex_activities = self.activity_mutex[b_item, act_id]
                valid_mutex = mutex_activities[mutex_activities >= 0]
                if len(valid_mutex) > 0:
                    # 진행 중인 mutex activity가 있는지 확인
                    mutex_running = (self.activity_started[b_item, valid_mutex] & ~self.activity_completed[b_item, valid_mutex]).any()
                    if mutex_running:
                        continue
                
                # Eligible teams 확인 (벡터화)
                eligible_teams = self.activity_eligible_teams[b_item, act_id]  # (N_T,)
                possible_teams = eligible_teams & available_teams
                self.available_actions[b_item, act_id] = possible_teams
    
    def step(self, action):
        """
        Action 수행 및 시뮬레이션 진행 (벡터화 버전)
        
        Args:
            action: (batch_size,) - 각 배치의 선택된 action 인덱스
                    action = activity_id * N_T + team_id (flattened index)
        
        Returns:
            state: 다음 상태 (Graph 형태)
            reward: (batch_size,) - 보상
            done: (batch_size,) - 종료 여부
        """
        # Action을 (activity_id, team_id)로 분해
        activity_id = action // self.N_T
        team_id = action % self.N_T
        
        # 디버깅
        if self.debug_env:
            for b in range(min(1, self.batch_size)):
                if not self.done[b]:
                    act = activity_id[b].item()
                    team = team_id[b].item()
                    print(f"\n🎬 [Batch {b}] Step {self.step_count[b].item()}: Activity {act} → Team {team}")
        
        # Activity 스케줄링
        active_batch_idxs = torch.arange(self.batch_size, device=self.done.device)[~self.done]
        if len(active_batch_idxs) > 0:
            self._schedule_activity(active_batch_idxs, activity_id, team_id)
        
        all_done = self.move_next_state(self.BATCH_IDX)

        if all_done:
            reward = self._get_reward()
            return None, reward, True
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
        
        # 디버깅
        if self.debug_env:
            for idx, b in enumerate(batch_idxs):
                if b.item() == 0:
                    act_id = activity_ids[b].item()
                    duration = durations[idx].item()
                    start_time = start_times[idx].item()
                    end_time = end_times[idx].item()
                    print(f"   ✅ Activity {act_id} scheduled: Start={start_time:.1f}, End={end_time:.1f}, Duration={duration}")
                    break

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
        batch_mask = torch.zeros(self.batch_size, dtype=torch.bool)
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
                activity_completed = self.activity_completed[b_item].all().item()
                
                if activity_completed:
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
            
            # 1. ended된 배치들 (다음 의사결정까지 시간 진행)
            batches_to_advance.extend(batches_ended)
            
            # 2. started 안된 배치들 중 가능한 액션이 없는 배치들 (다음 의사결정까지 시간 진행)
            for b_idx in batches_not_started:
                if not self.available_actions[b_idx].any().item():
                    batches_to_advance.append(b_idx)
            
            if len(batches_to_advance) > 0:
                batches_to_advance_tensor = torch.tensor(batches_to_advance, dtype=torch.long)
                self._advance_to_next_decision_event_for_batches(batches_to_advance_tensor)
                self._update_available_actions(batches_to_advance_tensor)
            else:
                break

        final_all_done = True
        for b in batch_idxs:
            b_item = b.item() if torch.is_tensor(b) else b
            if not self.activity_completed[b_item].all().item():
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
            ~self.activity_completed[batch_idxs]
        )  # (n_batches, max_N_A)
        
        # 완전 벡터 연산으로 상태 업데이트 (for문 완전 제거)
        
        # 1. Activity completed 상태 업데이트 (완전 벡터)
        self.activity_completed[batch_idxs] |= just_completed_mask
        
        # 2. Team current_activity 업데이트 (완전 벡터 - Advanced Indexing)
        # 완료된 activity들의 (batch, activity) 좌표 찾기
        relative_batch_coords, act_coords = just_completed_mask.nonzero(as_tuple=True)
        
        if len(relative_batch_coords) > 0:
            # 상대 배치 인덱스 → 절대 배치 인덱스 변환
            absolute_batch_coords = batch_idxs[relative_batch_coords]
            
            # 완료된 activity들의 팀 ID 가져오기 (2D advanced indexing)
            completed_teams = self.activity_assigned_team[absolute_batch_coords, act_coords]
            
            # 팀 상태 업데이트 (2D advanced indexing)
            self.team_current_activity[absolute_batch_coords, completed_teams] = -1
            
            # 3. 프로젝트 완료 시간 업데이트 (완전 벡터 - scatter_reduce)
            # 완료된 activity들의 프로젝트 ID와 종료 시간
            completed_proj_ids = self.activity_project[absolute_batch_coords, act_coords]
            completed_end_times = self.activity_end_time[absolute_batch_coords, act_coords]
            
            # 배치별로 scatter_reduce 적용 (배치 차원 처리)
            for i, b in enumerate(batch_idxs):
                b_item = b.item() if torch.is_tensor(b) else b
                # 이 배치에 속한 완료된 activity들
                batch_mask = (absolute_batch_coords == b_item)
                if batch_mask.any():
                    batch_proj_ids = completed_proj_ids[batch_mask]
                    batch_end_times = completed_end_times[batch_mask]
                    
                    # scatter_reduce로 프로젝트별 최대 종료 시간 계산
                    self.project_completion_time[b_item].scatter_reduce_(
                        0,
                        batch_proj_ids,
                        batch_end_times,
                        reduce='amax',
                        include_self=True
                    )
            
            # 4. 프로젝트 완료 여부 확인 (완전 벡터)
            for i, b in enumerate(batch_idxs):
                b_item = b.item() if torch.is_tensor(b) else b
                
                # 유효한 activity 마스크
                num_activities = self.num_activities[b_item].item()
                valid_activities = torch.arange(self.max_N_A, device=self.activity_project.device) < num_activities
                
                # 각 프로젝트에 속한 유효한 activity 마스크 (N_P, max_N_A)
                proj_activity_mask = (self.activity_project[b_item].unsqueeze(0) == torch.arange(self.N_P, device=self.activity_project.device).unsqueeze(1))
                proj_activity_mask = proj_activity_mask & valid_activities.unsqueeze(0)
                
                # 각 프로젝트에 activity가 있는지
                proj_has_activities = proj_activity_mask.any(dim=1)  # (N_P,)
                
                # 완료된 activity 마스크
                completed_mask_expanded = self.activity_completed[b_item].unsqueeze(0)  # (1, max_N_A)
                proj_completed_activities = proj_activity_mask & completed_mask_expanded  # (N_P, max_N_A)
                
                # 각 프로젝트의 완료된 activity 개수 == 전체 activity 개수
                proj_all_completed = (proj_completed_activities.sum(dim=1) == proj_activity_mask.sum(dim=1)) & proj_has_activities
                
                # 프로젝트 완료 상태 업데이트
                newly_completed = proj_all_completed & ~self.project_completed[b_item]
                self.project_completed[b_item] |= newly_completed
                
                # 디버깅
                if self.debug_env and b_item == 0:
                    newly_completed_projs = newly_completed.nonzero(as_tuple=False).squeeze(-1)
                    for p_idx in newly_completed_projs:
                        p_idx_item = p_idx.item()
                        completion_time = self.project_completion_time[b_item, p_idx_item].item()
                        due_date = self.project_due_date[b_item, p_idx_item].item()
                        tardiness = max(0, completion_time - due_date)
                        print(f"   🎉 Project {p_idx_item} completed! Time={completion_time:.1f}, Due={due_date}, Tardiness={tardiness:.1f}")
                    
                    if len(completed_acts) > 0:
                        act_id = completed_acts[0].item()
                        end_time = self.activity_end_time[b_item, act_id].item()
                        print(f"   ✅ Activity {act_id} completed at t={end_time:.1f}")

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
        masked_times = torch.where(remaining_times > 0, remaining_times, torch.tensor(float('inf'), device=remaining_times.device))
        
        # 각 배치별 최소값 계산 (가장 먼저 완료될 activity의 남은 시간)
        batch_mins, min_indices = masked_times.min(dim=1)  # (n_batches,)
        
        # inf인 경우 (진행 중인 activity가 없음) 0.0으로 대체
        batch_mins = torch.where(batch_mins == float('inf'), torch.tensor(0.0, device=batch_mins.device), batch_mins)
        
        # 디버그 출력
        if self.debug_env and len(batch_idxs) > 0:
            for i, b_idx in enumerate(batch_idxs[:1]):  # 첫 번째 배치만
                min_time = batch_mins[i].item()
                act_idx = min_indices[i].item()
                b_item = b_idx.item() if torch.is_tensor(b_idx) else b_idx
                print(f"    ⏱️ [Next Event] Batch {b_item}, dt={min_time:.2f}, Activity {act_idx} completion")
        
        return batch_mins

    def _complete_activity(self, batch_idx, activity_id, team_id):
        """
        Activity 완료 처리
        
        Args:
            batch_idx: 배치 인덱스
            activity_id: Activity ID
            team_id: 팀 ID
        """
        # Activity 상태 업데이트
        self.activity_completed[batch_idx, activity_id] = True
        
        # 팀 상태 업데이트
        self.team_current_activity[batch_idx, team_id] = -1
        
        # 프로젝트 완료 시간 업데이트
        proj_id = self.activity_project[batch_idx, activity_id].item()
        end_time = self.activity_end_time[batch_idx, activity_id].item()
        self.project_completion_time[batch_idx, proj_id] = max(
            self.project_completion_time[batch_idx, proj_id].item(),
            end_time
        )
        
        # 프로젝트의 모든 activity가 완료되었는지 확인
        proj_activities = (self.activity_project[batch_idx] == proj_id)
        proj_completed = self.activity_completed[batch_idx] | ~proj_activities  # 해당 프로젝트가 아닌 것은 True로
        num_activities = self.num_activities[batch_idx].item()
        
        # 유효한 activity만 확인
        valid_activities = torch.arange(self.max_N_A) < num_activities
        proj_activities_valid = proj_activities & valid_activities
        
        if (proj_completed | ~proj_activities_valid).all():
            self.project_completed[batch_idx, proj_id] = True
            
            if self.debug_env and batch_idx == 0:
                completion_time = self.project_completion_time[batch_idx, proj_id].item()
                due_date = self.project_due_date[batch_idx, proj_id].item()
                tardiness = max(0, completion_time - due_date)
                print(f"   🎉 Project {proj_id} completed! Time={completion_time:.1f}, Due={due_date}, Tardiness={tardiness:.1f}")
        
        if self.debug_env and batch_idx == 0:
            end_time = self.activity_end_time[batch_idx, activity_id].item()
            print(f"   ✅ Activity {activity_id} completed at t={end_time:.1f}")
    
    
    def _get_reward(self):
        """
        보상 계산 (벡터화 버전)
        에피소드가 끝났을 때만 실제 보상 반환
        
        Returns:
            reward: (batch_size,) - 보상 (음의 목적함수값)
        """
        reward = torch.zeros(self.batch_size, device=self.done.device)
        
        if self.done.any():
            if self.objective == 'tardiness':
                # Total tardiness (작을수록 좋음)
                tardiness = torch.clamp(
                    self.project_completion_time - self.project_due_date,
                    min=0.0
                ).sum(dim=1)  # (batch_size,)
                reward = torch.where(self.done, -tardiness, reward)
            
            elif self.objective == 'makespan':
                # Makespan (작을수록 좋음)
                makespan = self.project_completion_time.max(dim=1)[0]  # (batch_size,)
                reward = torch.where(self.done, -makespan, reward)
        
        return reward
    
    def _get_obj(self):
        """
        목적함수값 계산 (벡터화 버전, 테스트용)
        
        Returns:
            obj: (batch_size,) - 목적함수값 (작을수록 좋음)
        """
        if self.objective == 'tardiness':
            # Total tardiness
            obj = torch.clamp(
                self.project_completion_time - self.project_due_date,
                min=0.0
            ).sum(dim=1)  # (batch_size,)
        
        elif self.objective == 'makespan':
            # Makespan
            obj = self.project_completion_time.max(dim=1)[0]  # (batch_size,)
        
        else:
            obj = torch.zeros(self.batch_size, device=self.project_completion_time.device)
        
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
            # 노드 피처 구성
            # ========================================
            # Activity 노드: duration, project_id, started, completed, start_time (정규화됨)
            # Team 노드: available_time (정규화됨)
            # Project 노드: release_time, due_date, completion_time (정규화됨)
            
            num_act = self.num_activities[b].item()
            num_nodes = num_act + self.N_T + self.N_P
            
            # 노드 피처 초기화
            node_features = torch.zeros(num_nodes, 10)  # 10차원 피처
            
            # Activity 노드 (0 ~ num_act-1)
            # 정규화를 위한 시간 스케일
            max_time = max(1.0, self.sim_time[b].item() + 1.0)
            
            for act_id in range(num_act):
                idx = act_id
                node_features[idx, 0] = self.activity_duration[b, act_id] / 10.0  # 정규화
                node_features[idx, 1] = self.activity_project[b, act_id] / self.N_P
                node_features[idx, 2] = float(self.activity_started[b, act_id])
                node_features[idx, 3] = float(self.activity_completed[b, act_id])
                if self.activity_started[b, act_id]:
                    node_features[idx, 4] = self.activity_start_time[b, act_id] / max_time
            
            # Team 노드 (num_act ~ num_act+N_T-1)
            for team_id in range(self.N_T):
                idx = num_act + team_id
                node_features[idx, 5] = self.team_available_time[b, team_id] / max_time
                node_features[idx, 6] = float(self.team_current_activity[b, team_id] >= 0)
            
            # Project 노드 (num_act+N_T ~ num_act+N_T+N_P-1)
            for proj_id in range(self.N_P):
                idx = num_act + self.N_T + proj_id
                node_features[idx, 7] = self.project_release_time[b, proj_id] / max_time
                node_features[idx, 8] = self.project_due_date[b, proj_id] / max_time
                node_features[idx, 9] = float(self.project_completed[b, proj_id])
            
            # ========================================
            # 엣지 구성
            # ========================================
            edge_index_list = []
            
            # 1. Activity → Activity (Precedence)
            for act_id in range(num_act):
                predecessors = self.activity_predecessors[b, act_id]
                for pred_id in predecessors:
                    if pred_id >= 0 and pred_id < num_act:
                        edge_index_list.append([pred_id, act_id])
            
            # 2. Activity ↔ Activity (Mutex - 양방향)
            for act_id in range(num_act):
                mutex_activities = self.activity_mutex[b, act_id]
                for mutex_id in mutex_activities:
                    if mutex_id >= 0 and mutex_id < num_act and mutex_id > act_id:
                        edge_index_list.append([act_id, mutex_id])
                        edge_index_list.append([mutex_id, act_id])
            
            # 3. Activity → Team (Eligible)
            for act_id in range(num_act):
                eligible_teams = self.activity_eligible_teams[b, act_id]
                for team_id in range(self.N_T):
                    if eligible_teams[team_id]:
                        edge_index_list.append([act_id, num_act + team_id])
            
            # 4. Activity → Project (Belongs to)
            for act_id in range(num_act):
                proj_id = self.activity_project[b, act_id].item()
                edge_index_list.append([act_id, num_act + self.N_T + proj_id])
            
            if edge_index_list:
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # ========================================
            # Action 마스크 (가능한 action만 True)
            # ========================================
            # 실제 activity 수에 맞춰 mask 생성
            action_mask = self.available_actions[b, :num_act, :].reshape(-1)  # (num_act * N_T,)
            
            # PyG Data 객체 생성
            data = Data(
                x=node_features,
                edge_index=edge_index,
                mask=action_mask,  # Action 마스크
                batch_idx=b,
                num_activities=num_act,
                sim_time=self.sim_time[b].item()
            )
            
            state_list.append(data)
        
        return state_list
    
    def get_objective(self):
        """
        현재 목적함수값 반환 (테스트용)
        
        Returns:
            obj: (batch_size,) or float - 목적함수값
        """
        obj = self._get_obj()
        
        if self.batch_size == 1:
            return obj[0].item()
        else:
            return obj


# ========================================
# 테스트 코드
# ========================================
if __name__ == "__main__":
    print("="*60)
    print("SchedulingEnv 테스트")
    print("="*60)
    
    # 환경 파라미터
    env_params = {
        'batch_size': 2,
        'pomo_size': 1,
        'N_P': 3,
        'N_A_min': 3,
        'N_A_max': 5,
        'N_T': 3,
        'duration_min': 2,
        'duration_max': 5,
        'precedence_prob': 0.3,
        'mutex_prob': 0.1,
        'eligible_teams_ratio': 0.6,
        'due_date_tightness': 1.3,
        'objective': 'tardiness',
        'debug_env': True,
    }
    
    # 환경 생성
    env = SchedulingEnv(env_params, debug_env=True)
    
    # 리셋
    env._reset()
    
    # 초기 상태 확인
    state = env._get_state()
    print(f"\n초기 상태: {len(state)}개 배치")
    print(f"Batch 0 노드 수: {state[0].x.shape[0]}")
    print(f"Batch 0 엣지 수: {state[0].edge_index.shape[1]}")
    
    # 가능한 액션 확인
    env._update_available_actions(env.BATCH_IDX)
    num_feasible = env.available_actions.view(env.batch_size, -1).sum(dim=1)
    print(f"\n가능한 action 수: {num_feasible[0].item()}")
    
    # 랜덤 액션 수행 (에피소드 끝까지)
    import random
    step = 0
    max_steps = 100
    
    while not env.done.all() and step < max_steps:
        # 가능한 액션 선택
        active_batch_idxs = torch.arange(env.batch_size)[~env.done]
        if len(active_batch_idxs) > 0:
            env._update_available_actions(active_batch_idxs)
        
        actions = []
        for b in range(env.batch_size):
            if env.done[b]:
                actions.append(0)  # 더미
            else:
                # 가능한 액션 중 랜덤 선택
                feasible_actions = env.available_actions[b].nonzero(as_tuple=False)
                if len(feasible_actions) > 0:
                    chosen = random.choice(feasible_actions)
                    act_id = chosen[0].item()
                    team_id = chosen[1].item()
                    action_idx = act_id * env.N_T + team_id
                    actions.append(action_idx)
                else:
                    actions.append(0)  # 가능한 액션이 없으면 더미
        
        actions = torch.tensor(actions, dtype=torch.long)
        
        # Step
        next_state, reward, done = env.step(actions)
        
        step += 1
    
    # 최종 결과
    print(f"\n{'='*60}")
    print("시뮬레이션 완료!")
    print(f"{'='*60}")
    print(f"총 스텝: {step}")
    print(f"최종 시뮬레이션 시간: {env.sim_time[0].item():.1f}")
    print(f"목적함수 ({env.objective}): {env.get_objective()}")
    
    print("\n✅ SchedulingEnv 테스트 완료!")
