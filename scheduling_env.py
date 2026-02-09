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
        self.activity_scheduled = torch.zeros(self.batch_size, self.max_N_A, dtype=torch.bool)  # 스케줄 완료 여부
        self.activity_started = torch.zeros(self.batch_size, self.max_N_A, dtype=torch.bool)  # 시작 여부
        self.activity_completed = torch.zeros(self.batch_size, self.max_N_A, dtype=torch.bool)  # 완료 여부
        self.activity_start_time = torch.full((self.batch_size, self.max_N_A), -1.0)  # 시작 시간
        self.activity_end_time = torch.full((self.batch_size, self.max_N_A), -1.0)  # 종료 시간
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
        # 이벤트 큐 (각 배치별로 관리)
        # ========================================
        # 이벤트: (time, event_type, activity_id, team_id)
        # event_type: 0=activity_end
        self.event_queue = [[] for _ in range(self.batch_size)]  # List of lists
        
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
    
    def _get_feasible_actions(self):
        """
        현재 시점에서 시작 가능한 (activity, team) 페어 반환
        
        Returns:
            feasible_mask: (batch_size, max_N_A, N_T) - 가능한 페어는 True
            num_feasible: (batch_size,) - 각 배치의 가능한 페어 수
        """
        feasible_mask = torch.zeros(self.batch_size, self.max_N_A, self.N_T, dtype=torch.bool)
        
        for b in range(self.batch_size):
            # 이미 완료된 배치는 스킵
            if self.done[b]:
                continue
            
            current_time = self.sim_time[b].item()
            num_act = self.num_activities[b].item()
            
            for act_id in range(num_act):
                # 이미 스케줄되었으면 스킵
                if self.activity_scheduled[b, act_id]:
                    continue
                
                # 프로젝트가 release되지 않았으면 스킵
                proj_id = self.activity_project[b, act_id].item()
                if current_time < self.project_release_time[b, proj_id].item():
                    continue
                
                # 선행 작업이 완료되었는지 확인
                predecessors = self.activity_predecessors[b, act_id]
                all_preds_done = True
                for pred_id in predecessors:
                    if pred_id >= 0 and not self.activity_completed[b, pred_id]:
                        all_preds_done = False
                        break
                
                if not all_preds_done:
                    continue
                
                # Mutually exclusive 제약 확인
                # 동시 수행 불가인 activity가 현재 진행 중이면 스킵
                mutex_activities = self.activity_mutex[b, act_id]
                mutex_running = False
                for mutex_id in mutex_activities:
                    if mutex_id >= 0 and self.activity_started[b, mutex_id] and not self.activity_completed[b, mutex_id]:
                        mutex_running = True
                        break
                
                if mutex_running:
                    continue
                
                # Eligible teams 확인 및 가능한 팀 마킹
                eligible_teams = self.activity_eligible_teams[b, act_id]  # (N_T,)
                available_teams = self.team_available_time[b] <= current_time  # 현재 시간에 사용 가능한 팀
                
                # Eligible하고 available한 팀만 가능
                possible_teams = eligible_teams & available_teams
                feasible_mask[b, act_id] = possible_teams
        
        # 각 배치의 가능한 페어 수 계산
        num_feasible = feasible_mask.view(self.batch_size, -1).sum(dim=1)
        
        return feasible_mask, num_feasible
    
    def step(self, action):
        """
        Action 수행 및 시뮬레이션 진행
        
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
        
        # 각 배치별로 action 수행
        for b in range(self.batch_size):
            if self.done[b]:
                continue
            
            act_id = activity_id[b].item()
            tm_id = team_id[b].item()
            
            # Activity 스케줄링
            self._schedule_activity(b, act_id, tm_id)
        
        # 이벤트 처리 및 시간 진행
        self._process_events()
        
        # 종료 조건 확인
        self._check_done()
        
        # 다음 상태 생성
        next_state = self._get_state()
        
        # 보상 계산 (에피소드 끝에만)
        reward = self._get_reward()
        
        # 스텝 카운터 증가
        self.step_count += 1
        
        return next_state, reward, self.done
    
    def _schedule_activity(self, batch_idx, activity_id, team_id):
        """
        Activity를 팀에 할당하고 시작
        
        Args:
            batch_idx: 배치 인덱스
            activity_id: Activity ID
            team_id: 팀 ID
        """
        # 현재 시간
        current_time = self.sim_time[batch_idx].item()
        
        # 시작 시간 결정
        # 1) 현재 시간
        # 2) 프로젝트 release time
        # 3) 팀 available time
        # 4) 선행 작업 완료 시간
        proj_id = self.activity_project[batch_idx, activity_id].item()
        start_time = max(
            current_time,
            self.project_release_time[batch_idx, proj_id].item(),
            self.team_available_time[batch_idx, team_id].item()
        )
        
        # 선행 작업 완료 시간 고려
        predecessors = self.activity_predecessors[batch_idx, activity_id]
        for pred_id in predecessors:
            if pred_id >= 0:
                pred_end = self.activity_end_time[batch_idx, pred_id].item()
                start_time = max(start_time, pred_end)
        
        # Mutually exclusive 제약 고려
        mutex_activities = self.activity_mutex[batch_idx, activity_id]
        for mutex_id in mutex_activities:
            if mutex_id >= 0 and self.activity_scheduled[batch_idx, mutex_id]:
                mutex_end = self.activity_end_time[batch_idx, mutex_id].item()
                start_time = max(start_time, mutex_end)
        
        # 종료 시간 계산
        duration = self.activity_duration[batch_idx, activity_id].item()
        end_time = start_time + duration
        
        # Activity 상태 업데이트
        self.activity_scheduled[batch_idx, activity_id] = True
        self.activity_started[batch_idx, activity_id] = True
        self.activity_start_time[batch_idx, activity_id] = start_time
        self.activity_end_time[batch_idx, activity_id] = end_time
        self.activity_assigned_team[batch_idx, activity_id] = team_id
        
        # 팀 상태 업데이트
        self.team_available_time[batch_idx, team_id] = end_time
        self.team_current_activity[batch_idx, team_id] = activity_id
        
        # 이벤트 큐에 종료 이벤트 추가
        self.event_queue[batch_idx].append((end_time, 0, activity_id, team_id))
        
        # 디버깅
        if self.debug_env and batch_idx == 0:
            print(f"   ✅ Activity {activity_id} scheduled: Start={start_time:.1f}, End={end_time:.1f}, Duration={duration}")
    
    def _process_events(self):
        """
        이벤트 큐를 처리하고 시뮬레이션 시간을 진행
        """
        for b in range(self.batch_size):
            if self.done[b]:
                continue
            
            # 이벤트 큐가 비어있으면 다음 action까지 대기
            if not self.event_queue[b]:
                continue
            
            # 이벤트를 시간순으로 정렬
            self.event_queue[b].sort(key=lambda x: x[0])
            
            # 가장 빠른 이벤트 처리 (여러 개 동시 발생 가능)
            earliest_time = self.event_queue[b][0][0]
            
            # 해당 시간까지 진행
            self.sim_time[b] = earliest_time
            
            # 해당 시간의 모든 이벤트 처리
            processed_events = []
            for event in self.event_queue[b]:
                event_time, event_type, act_id, tm_id = event
                
                if event_time == earliest_time:
                    # Activity 종료 이벤트
                    if event_type == 0:
                        self._complete_activity(b, act_id, tm_id)
                    processed_events.append(event)
            
            # 처리된 이벤트 제거
            for event in processed_events:
                self.event_queue[b].remove(event)
    
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
    
    def _check_done(self):
        """
        에피소드 종료 조건 확인
        모든 activity가 스케줄되면 종료
        """
        for b in range(self.batch_size):
            if self.done[b]:
                continue
            
            num_act = self.num_activities[b].item()
            scheduled_count = self.activity_scheduled[b, :num_act].sum().item()
            
            if scheduled_count == num_act:
                self.done[b] = True
                
                if self.debug_env and b == 0:
                    print(f"\n✅ Batch {b} 완료! 모든 {num_act}개 activity 스케줄 완료")
                    print(f"   최종 시뮬레이션 시간: {self.sim_time[b].item():.1f}")
    
    def _get_reward(self):
        """
        보상 계산
        에피소드가 끝났을 때만 실제 보상 반환
        
        Returns:
            reward: (batch_size,) - 보상 (음의 목적함수값)
        """
        reward = torch.zeros(self.batch_size)
        
        for b in range(self.batch_size):
            if self.done[b]:
                # 목적함수 계산
                if self.objective == 'tardiness':
                    # Total tardiness (작을수록 좋음)
                    tardiness = torch.clamp(
                        self.project_completion_time[b] - self.project_due_date[b],
                        min=0.0
                    ).sum().item()
                    reward[b] = -tardiness  # 보상은 음수 (최대화)
                
                elif self.objective == 'makespan':
                    # Makespan (작을수록 좋음)
                    makespan = self.project_completion_time[b].max().item()
                    reward[b] = -makespan  # 보상은 음수 (최대화)
        
        return reward
    
    def _get_obj(self):
        """
        목적함수값 계산 (테스트용)
        
        Returns:
            obj: (batch_size,) - 목적함수값 (작을수록 좋음)
        """
        obj = torch.zeros(self.batch_size)
        
        for b in range(self.batch_size):
            if self.objective == 'tardiness':
                # Total tardiness
                tardiness = torch.clamp(
                    self.project_completion_time[b] - self.project_due_date[b],
                    min=0.0
                ).sum().item()
                obj[b] = tardiness
            
            elif self.objective == 'makespan':
                # Makespan
                makespan = self.project_completion_time[b].max().item()
                obj[b] = makespan
        
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
            action_mask = torch.zeros(self.max_N_A * self.N_T, dtype=torch.bool)
            feasible_mask, _ = self._get_feasible_actions()
            action_mask_2d = feasible_mask[b].view(-1)  # (max_N_A * N_T,)
            action_mask[:len(action_mask_2d)] = action_mask_2d
            
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
    feasible_mask, num_feasible = env._get_feasible_actions()
    print(f"\n가능한 action 수: {num_feasible[0].item()}")
    
    # 랜덤 액션 수행 (에피소드 끝까지)
    import random
    step = 0
    max_steps = 100
    
    while not env.done.all() and step < max_steps:
        # 가능한 액션 선택
        feasible_mask, num_feasible = env._get_feasible_actions()
        
        actions = []
        for b in range(env.batch_size):
            if env.done[b]:
                actions.append(0)  # 더미
            else:
                # 가능한 액션 중 랜덤 선택
                feasible_actions = feasible_mask[b].nonzero(as_tuple=False)
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
