"""
RCMPSP (Resource-Constrained Multi-Project Scheduling Problem) 환경
DES (Discrete Event Simulation) 기반
MDP Action: 현재 시작 가능한 (Activity, Team) 페어 선택
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from data_generator import generate_scheduling_data_batch


@dataclass
class EnvState:
    """DANIEL 모델용 상태 정의 (RCMPSP 환경). Action space: (activity, team) 쌍."""
    fea_act_tensor: torch.Tensor = None       # (B, N, 12) activity features
    act_mask_tensor: torch.Tensor = None       # (B, N, 3)  attention mask
    fea_team_tensor: torch.Tensor = None       # (B, T, 8)  team features
    team_mask_tensor: torch.Tensor = None      # (B, T, T)  team attention mask
    dynamic_pair_mask_tensor: torch.Tensor = None  # (B, N, T)  incompatible pair mask
    comp_idx_tensor: torch.Tensor = None       # (B, T, T, N) competition index
    candidate_tensor: torch.Tensor = None      # (B, N) activity identity (0..N-1)
    fea_pairs_tensor: torch.Tensor = None      # (B, N, T, 8) pair features
    pred_idx_tensor: torch.Tensor = None       # (B, N, max_preds) predecessor indices (-1 padded)
    succ_idx_tensor: torch.Tensor = None       # (B, N, max_succs) successor indices (-1 padded)


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
        
        # 상태 출력 모드 (DANIEL 모델)
        self.state_mode = 'daniel'

        # Wait 옵션: release time / mutex로 즉시 실행 불가해도 기다려서 스케줄 허용
        self.allow_wait_release = env_params.get('allow_wait_release', False)
        self.allow_wait_mutex = env_params.get('allow_wait_mutex', False)
        # Dominance rule: 대기 중 idle time에 다른 activity를 할 수 있으면 해당 대기 pair 제외
        self.dominance_rule = env_params.get('dominance_rule', False)

        # Step 진행 로그: 매 step마다 완료 activity 수 출력
        self.step_log = env_params.get('step_log', False)

        # Reward 방식: 'sparse' (에피소드 끝에만) 또는 'stepwise' (매 step마다 dense reward)
        self.reward_type = env_params.get('reward_type', 'sparse')
        if self.reward_type == 'stepwise' and self.objective != 'tardiness':
            raise NotImplementedError(
                "Step-wise reward is currently implemented for 'tardiness' objective only. "
                "Makespan의 경우 FJSP 논문의 op_ct_lb 방식을 참고하세요."
            )

        # Stepwise reward: ATC scaling parameter (Pinedo Section 7.3)
        self._atc_scaling = env_params.get('atc_scaling', 2.0)

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
        self.activity_duration = self.problem['activity_duration'].to(self.device)              # (batch_size, max_N_A) 평균 처리 시간
        self.activity_team_duration = self.problem['activity_team_duration'].to(self.device)   # (batch_size, max_N_A, N_T) 팀별 처리 시간
        self.activity_project = self.problem['activity_project'].to(self.device)  # (batch_size, max_N_A)
        self.activity_eligible_teams = self.problem['activity_eligible_teams'].to(self.device)  # (batch_size, max_N_A, N_T)
        self.activity_predecessors = self.problem['activity_predecessors'].to(self.device)  # (batch_size, max_N_A, max_preds)
        self.activity_successors = self.problem['activity_successors'].to(self.device)   # (batch_size, max_N_A, max_succs)
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
        
        # Step 로그용 카운터
        self._env_step_count = 0

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

        # ========================================
        # Step-wise reward 초기화
        # ========================================
        if self.reward_type == 'stepwise':
            self.estimated_tardiness = self._compute_estimated_tardiness()
            self.step_reward = torch.zeros(self.batch_size, device=self.device)

    def _initialize_action_space(self):
        """
        각 배치별로 eligible한 (activity, team) 조합만 추출하여 action space 구성
        배치 내 최대 action space 크기에 맞춰 패딩 적용
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
        feasible = basic & pred_ok & mutex_ok

        # Wait 옵션: release/mutex 제약을 완화하여 기다려서 스케줄 가능한 pair도 허용
        if self.allow_wait_release or self.allow_wait_mutex:
            # wait 가능한 pair: basic 조건에서 released와 team_ok 제거 + pred_ok
            wait_base = valid & unstarted & pred_ok
            if not self.allow_wait_release:
                wait_base = wait_base & released
            if not self.allow_wait_mutex:
                wait_base = wait_base & mutex_ok
            feasible = feasible | wait_base

        self.available_actions[batch_idxs] = feasible
    
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

        if self.step_log:
            self._env_step_count += 1
            # 패딩 제외 실제 활동 기준으로 집계
            padding_cnt = (self.max_N_A - self.num_activities)          # (B,)
            real_started = self.activity_started.sum(dim=1) - padding_cnt  # (B,)
            real_ended   = self.activity_ended.sum(dim=1)   - padding_cnt  # (B,)
            avg_n = self.num_activities.float().mean().item()
            avg_e = real_ended.float().mean().item()
            s_min, s_max = real_started.min().item(), real_started.max().item()
            # 진짜 불일치 검사: expected = min(step_count, num_activities[b])
            # s_min < s_max 이더라도 인스턴스별 num_activities 차이(완료 인스턴스) 때문일 수 있으므로
            # 실제로 step 수보다 적게 스케줄된 인스턴스가 있는지만 체크
            expected = torch.minimum(
                torch.full_like(self.num_activities, self._env_step_count),
                self.num_activities)
            behind = (expected - real_started).clamp(min=0)  # (B,) 양수 = 해당 step수만큼 안 스케줄됨
            n_behind = (behind > 0).sum().item()
            if n_behind > 0:
                print(f"  ⚠️  [Step {self._env_step_count}] Started 진짜 불일치! "
                      f"{n_behind}개 인스턴스 미달 (max_behind={int(behind.max().item())}, "
                      f"range={int(s_min)}~{int(s_max)}) / {avg_n:.0f} avg, Ended: {avg_e:.1f} (B={self.batch_size})")
            else:
                # 범위 차이는 인스턴스별 num_activities 차이(완료 인스턴스) — 정상
                print(f"  [Step {self._env_step_count}] Started: {int(s_min)}~{int(s_max)} / {avg_n:.0f} avg, "
                      f"Ended: {avg_e:.1f} (B={self.batch_size})")

        # Step-wise reward 계산
        if self.reward_type == 'stepwise':
            prev_est = self.estimated_tardiness
            if all_done:
                new_est = self._get_obj()  # Terminal: 실제 tardiness
            else:
                new_est = self._compute_estimated_tardiness()
            self.step_reward = prev_est - new_est  # (B,) positive = improvement
            self.estimated_tardiness = new_est

        if all_done:
            obj_value = self._get_obj()  # 목적함수값 반환 (양수)
            return None, obj_value, True
        else:
            self.state = self._get_state()
            return self.state, None, False

    def step_pair(self, activity_ids, team_ids):
        """
        DANIEL 모델용 step: (activity_id, team_id) 쌍을 직접 받아 처리
        action_to_pair 매핑을 거치지 않음
        
        Args:
            activity_ids: (batch_size,) - 각 배치의 activity ID
            team_ids: (batch_size,) - 각 배치의 team ID
        
        Returns:
            state, obj_value, done (step과 동일)
        """
        active_batch_idxs = torch.arange(self.batch_size, device=self.device)[~self.done]
        if len(active_batch_idxs) > 0:
            self._schedule_activity(active_batch_idxs, activity_ids, team_ids)
        
        all_done = self.move_next_state(self.BATCH_IDX)

        if self.step_log:
            self._env_step_count += 1
            # 패딩 제외 실제 활동 기준으로 집계
            padding_cnt = (self.max_N_A - self.num_activities)          # (B,)
            real_started = self.activity_started.sum(dim=1) - padding_cnt  # (B,)
            real_ended   = self.activity_ended.sum(dim=1)   - padding_cnt  # (B,)
            avg_n = self.num_activities.float().mean().item()
            avg_e = real_ended.float().mean().item()
            s_min, s_max = real_started.min().item(), real_started.max().item()
            # 진짜 불일치 검사: expected = min(step_count, num_activities[b])
            # s_min < s_max 이더라도 인스턴스별 num_activities 차이(완료 인스턴스) 때문일 수 있으므로
            # 실제로 step 수보다 적게 스케줄된 인스턴스가 있는지만 체크
            expected = torch.minimum(
                torch.full_like(self.num_activities, self._env_step_count),
                self.num_activities)
            behind = (expected - real_started).clamp(min=0)  # (B,) 양수 = 해당 step수만큼 안 스케줄됨
            n_behind = (behind > 0).sum().item()
            if n_behind > 0:
                print(f"  ⚠️  [Step {self._env_step_count}] Started 진짜 불일치! "
                      f"{n_behind}개 인스턴스 미달 (max_behind={int(behind.max().item())}, "
                      f"range={int(s_min)}~{int(s_max)}) / {avg_n:.0f} avg, Ended: {avg_e:.1f} (B={self.batch_size})")
            else:
                # 범위 차이는 인스턴스별 num_activities 차이(완료 인스턴스) — 정상
                print(f"  [Step {self._env_step_count}] Started: {int(s_min)}~{int(s_max)} / {avg_n:.0f} avg, "
                      f"Ended: {avg_e:.1f} (B={self.batch_size})")

        # Step-wise reward 계산
        if self.reward_type == 'stepwise':
            prev_est = self.estimated_tardiness
            if all_done:
                new_est = self._get_obj()  # Terminal: 실제 tardiness
            else:
                new_est = self._compute_estimated_tardiness()
            self.step_reward = prev_est - new_est  # (B,) positive = improvement
            self.estimated_tardiness = new_est

        if all_done:
            obj_value = self._get_obj()
            return None, obj_value, True
        else:
            self.state = self._get_state()
            return self.state, None, False

    def _get_mutex_clear_times(self):
        """각 activity의 mutex 파트너 중 실행 중인 것의 최대 end_time 반환. (B, N)"""
        B = self.batch_size
        mutex_data = self.activity_mutex  # (B, N, max_mutex)
        valid_mp = mutex_data >= 0
        safe_mp = mutex_data.clamp(min=0)
        mp_started = torch.gather(
            self.activity_started, 1, safe_mp.reshape(B, -1)
        ).reshape_as(safe_mp)
        mp_ended = torch.gather(
            self.activity_ended, 1, safe_mp.reshape(B, -1)
        ).reshape_as(safe_mp)
        mp_end_time = torch.gather(
            self.activity_end_time, 1, safe_mp.reshape(B, -1)
        ).reshape_as(safe_mp)
        mp_running = mp_started & ~mp_ended & valid_mp
        clear_times = torch.where(mp_running, mp_end_time, torch.zeros_like(mp_end_time))
        return clear_times.max(dim=2)[0]  # (B, N)

    def _schedule_activity(self, batch_idxs, activity_ids, team_ids):
        """
        Activity를 팀에 할당하고 시작 (wait 옵션 시 지연 시작 처리)

        Args:
            batch_idxs: (n_active,) - 처리할 배치 인덱스들
            activity_ids: (batch_size,) - 각 배치의 activity ID
            team_ids: (batch_size,) - 각 배치의 team ID
        """
        if len(batch_idxs) == 0:
            return

        acts = activity_ids[batch_idxs]
        teams = team_ids[batch_idxs]

        # 시작 시간 = 현재 시뮬레이션 시간 (기본)
        start_times = self.sim_time[batch_idxs].clone()  # (n_active,)

        # 팀 가용 시간까지 대기 (항상 적용)
        team_avail = self.team_available_time[batch_idxs, teams]
        start_times = torch.max(start_times, team_avail)

        # Release time 대기 (allow_wait_release=True일 때만 실제로 지연 발생 가능)
        if self.allow_wait_release:
            proj_ids = self.activity_project[batch_idxs, acts]
            release = self.project_release_time[batch_idxs, proj_ids]
            start_times = torch.max(start_times, release)

        # Mutex 파트너 종료 대기 (allow_wait_mutex=True일 때만 실제로 지연 발생 가능)
        if self.allow_wait_mutex:
            mutex_partners = self.activity_mutex[batch_idxs, acts]  # (n_active, max_mutex)
            valid_mp = mutex_partners >= 0
            safe_mp = mutex_partners.clamp(min=0)
            mp_started = torch.gather(self.activity_started[batch_idxs], 1, safe_mp)
            mp_ended = torch.gather(self.activity_ended[batch_idxs], 1, safe_mp)
            mp_end_time = torch.gather(self.activity_end_time[batch_idxs], 1, safe_mp)
            mp_running = mp_started & ~mp_ended & valid_mp
            mutex_clear = torch.where(mp_running, mp_end_time, torch.zeros_like(mp_end_time)).max(dim=1)[0]
            start_times = torch.max(start_times, mutex_clear)

        # 종료 시간 계산 (팀별 처리 시간 사용)
        end_times = start_times + self.activity_team_duration[batch_idxs, acts, teams]  # (n_active,)

        # Activity 상태 업데이트
        self.activity_started[batch_idxs, acts] = True
        self.activity_start_time[batch_idxs, acts] = start_times
        self.activity_end_time[batch_idxs, acts] = end_times
        # remaining_time = end_time - sim_time (지연 시작 시 duration보다 클 수 있음)
        self.activity_remaining_time[batch_idxs, acts] = end_times - self.sim_time[batch_idxs]
        self.activity_assigned_team[batch_idxs, acts] = teams

        # 팀 상태 업데이트
        self.team_available_time[batch_idxs, teams] = end_times
        self.team_current_activity[batch_idxs, teams] = acts 

    def move_next_state(self, batch_idxs):
        """
        시뮬레이터 로직에 따라 다음 의사결정 이벤트까지 시간 진행 (벡터화)
        배치 상태 분류를 all/any 배치 연산으로 수행 (for문 0개)
        
        Args:
            batch_idxs: 처리할 배치 인덱스들
            
        Returns:
            bool: all_done - 모든 배치의 activity가 완료되었는지 여부
        """
        max_iterations = 10000  # 무한 루프 방지
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
            self._advance_to_next_decision_event(advance_idxs)
            self._update_available_actions(advance_idxs)
        
        # 최대 반복 도달 시 디버그 출력
        if iterations >= max_iterations:
            print(f"\n❌ [move_next_state] 최대 반복 횟수({max_iterations}) 도달!")
        
        # 최종 완료 체크 (벡터화)
        final_all_done = self.activity_ended[batch_idxs].all().item()
        
        return final_all_done

    def _advance_to_next_decision_event(self, batch_idxs):
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
        
    def _compute_estimated_tardiness(self):
        """
        Shifting Bottleneck Heuristic (Pinedo 7.3) 기반 Tardiness 추정 (step-wise reward용)

        기존 Forward DAG relaxation (리소스 경합 무시)을 확장하여
        팀(리소스) 경합에 의한 지연을 반영한 tighter 추정치 계산.

        7 Phases:
          1. Forward DAG relaxation → ECT 하한 (리소스 무시)
          2. Backward DAG → Tail length → Local due date (activity별 마감시간)
          3. ATC (Apparent Tardiness Cost) 우선순위 per (activity, team)
          4. 팀 경합 지연: ATC 순서 기반 누적 대기시간
          5. Contention-adjusted ECT per (activity, team) → best team 선택
          6. DAG 재전파 (경합 지연을 후속 activity로 전파)
          7. Project completion → Total Tardiness

        Returns:
            estimated_tardiness: (batch_size,) - 추정 총 tardiness
        """
        B, N, T = self.batch_size, self.max_N_A, self.N_T
        device = self.device

        # ================================================================
        # 공통 마스크 및 데이터 추출
        # ================================================================
        valid = torch.arange(N, device=device).unsqueeze(0) < self.num_activities.unsqueeze(1)  # (B, N)
        scheduled = self.activity_started  # (B, N) -- 패딩도 True
        unscheduled = ~scheduled & valid   # (B, N) -- 실제 미스케줄 activity만

        proj_ids = self.activity_project.clamp(min=0)  # (B, N)
        release = torch.gather(self.project_release_time, 1, proj_ids)  # (B, N)
        due_date = torch.gather(self.project_due_date, 1, proj_ids)    # (B, N) activity별 프로젝트 due date

        preds = self.activity_predecessors   # (B, N, max_preds)
        succs = self.activity_successors     # (B, N, max_succs)
        valid_pred = preds >= 0
        valid_succ = succs >= 0
        safe_preds = preds.clamp(min=0)
        safe_succs = succs.clamp(min=0)

        elig = self.activity_eligible_teams  # (B, N, T) bool
        pt = self.activity_team_duration     # (B, N, T) 팀별 처리시간

        # Eligible team 평균 처리시간 (DAG relaxation 및 tail 계산에 사용)
        elig_count = elig.sum(dim=2).float().clamp(min=1)  # (B, N)
        avg_pt = (pt * elig.float()).sum(dim=2) / elig_count  # (B, N)

        # ================================================================
        # Phase 1: Forward DAG relaxation → ECT 하한 (리소스 무시)
        # ================================================================
        completion_lb = torch.where(
            scheduled,
            self.activity_end_time,
            release + avg_pt
        )
        completion_lb = torch.where(valid, completion_lb, torch.zeros_like(completion_lb))

        max_preds_dim = preds.shape[2]
        for _ in range(self.N_A_max):
            pred_comp = torch.gather(
                completion_lb, 1, safe_preds.reshape(B, -1)
            ).reshape(B, N, max_preds_dim)
            pred_comp = torch.where(valid_pred, pred_comp, torch.zeros_like(pred_comp))
            max_pred_comp = pred_comp.max(dim=2)[0]  # (B, N)

            new_start = torch.max(release, max_pred_comp)
            new_comp = torch.where(
                scheduled,
                self.activity_end_time,
                new_start + avg_pt
            )
            new_comp = torch.where(valid, new_comp, torch.zeros_like(new_comp))

            if torch.equal(new_comp, completion_lb):
                break
            completion_lb = new_comp

        # EST (earliest start time)
        est = completion_lb - avg_pt  # (B, N)
        est = torch.where(valid, est, torch.zeros_like(est))

        # ================================================================
        # Phase 2: Backward DAG → Tail length → Local due date
        # tail[a] = avg_pt[a] + max(tail[successors])
        # local_due[a] = due_project - (tail[a] - avg_pt[a])
        # ================================================================
        tail = torch.where(valid, avg_pt.clone(), torch.zeros(B, N, device=device))

        max_succs_dim = succs.shape[2]
        for _ in range(self.N_A_max):
            succ_tail = torch.gather(
                tail, 1, safe_succs.reshape(B, -1)
            ).reshape(B, N, max_succs_dim)
            succ_tail = torch.where(valid_succ, succ_tail, torch.zeros_like(succ_tail))
            max_succ_tail = succ_tail.max(dim=2)[0]  # (B, N)

            new_tail = avg_pt + max_succ_tail
            new_tail = torch.where(valid, new_tail, torch.zeros_like(new_tail))

            if torch.equal(new_tail, tail):
                break
            tail = new_tail

        # Local due date: activity가 프로젝트 납기를 지키기 위한 최소 완료시간
        remaining_after = tail - avg_pt  # (B, N) activity 이후 남은 critical path 길이
        local_due = due_date - remaining_after  # (B, N)

        # ================================================================
        # Phase 3: ATC priority index per (activity, team)
        # I_at = (1/p_at) * exp(-slack_at⁺ / (K * p̄_t))
        # ================================================================
        K = self._atc_scaling

        # Team ready time: (B, T) → (B, N, T)
        team_ready = self.team_available_time.unsqueeze(1).expand(-1, N, -1)  # (B, N, T)

        # Effective earliest start per (activity, team)
        eff_start = torch.max(
            est.unsqueeze(2).expand(-1, -1, T),
            team_ready
        )  # (B, N, T)

        # Slack: local_due - p_at - eff_start
        slack_pos = torch.clamp(
            local_due.unsqueeze(2).expand(-1, -1, T) - pt - eff_start,
            min=0.0
        )  # (B, N, T)

        # p̄_t: 팀별 미스케줄 eligible activity 평균 처리시간
        unsched_elig = unscheduled.unsqueeze(2) & elig  # (B, N, T)
        p_bar = (pt * unsched_elig.float()).sum(dim=1) / unsched_elig.sum(dim=1).float().clamp(min=1)  # (B, T)
        p_bar = p_bar.clamp(min=1e-6)
        p_bar_exp = p_bar.unsqueeze(1).expand(-1, N, -1)  # (B, N, T)

        # ATC index 계산
        safe_pt_denom = torch.where(elig, pt, torch.ones_like(pt)).clamp(min=1e-6)
        inv_pt = 1.0 / safe_pt_denom  # (B, N, T)

        exp_arg = (-slack_pos / (K * p_bar_exp + 1e-8)).clamp(min=-20.0, max=0.0)
        atc_index = inv_pt * torch.exp(exp_arg)  # (B, N, T)
        atc_index = torch.where(unsched_elig, atc_index, torch.zeros_like(atc_index))

        # ================================================================
        # Phase 4: Team contention delay (ATC 순서 기반 누적 대기시간)
        # ================================================================
        # (B, T, N) 공간에서 팀별 정렬
        atc_for_sort = atc_index.permute(0, 2, 1)  # (B, T, N)
        pt_for_sort = pt.permute(0, 2, 1)           # (B, T, N)
        mask_for_sort = unsched_elig.permute(0, 2, 1)  # (B, T, N)

        # Masked-out은 -inf → 정렬 시 맨 뒤로
        sort_key = torch.where(
            mask_for_sort, atc_for_sort,
            torch.full_like(atc_for_sort, -float('inf'))
        )
        sorted_indices = sort_key.argsort(dim=2, descending=True)  # (B, T, N)

        # 정렬 순서대로 PT 수집
        sorted_pt = torch.gather(pt_for_sort, 2, sorted_indices)  # (B, T, N)
        sorted_mask = torch.gather(mask_for_sort, 2, sorted_indices)  # (B, T, N)
        sorted_pt_masked = sorted_pt * sorted_mask.float()

        # 누적합 → delay = cumsum - own_pt (자기보다 긴급한 activity들의 PT 합)
        sorted_delay = torch.cumsum(sorted_pt_masked, dim=2) - sorted_pt_masked  # (B, T, N)

        # 원래 activity 순서로 복원
        unsort_indices = sorted_indices.argsort(dim=2)
        delay = torch.gather(sorted_delay, 2, unsort_indices).permute(0, 2, 1)  # (B, N, T)
        delay = torch.where(unsched_elig, delay, torch.zeros_like(delay))

        # ================================================================
        # Phase 5: Contention-adjusted ECT → best team 선택
        # adj_ect = max(est, team_ready) + delay + p_at
        # ================================================================
        adj_ect = eff_start + delay + pt  # (B, N, T)
        adj_ect = torch.where(
            unsched_elig,
            adj_ect,
            torch.full_like(adj_ect, float('inf'))
        )

        # Activity별 best team (최소 adj_ect)
        best_ect, _ = adj_ect.min(dim=2)  # (B, N)
        best_ect = torch.where(best_ect.isinf(), completion_lb, best_ect)

        # Final ECT: max(DAG LB, contention-adjusted) — 두 제약 모두 반영
        adj_completion = torch.where(
            scheduled,
            self.activity_end_time,
            torch.max(completion_lb, best_ect)
        )
        adj_completion = torch.where(valid, adj_completion, torch.zeros_like(adj_completion))

        # ================================================================
        # Phase 6: DAG 재전파 (경합 지연을 후속 activity로 전파)
        # ================================================================
        for _ in range(self.N_A_max):
            pred_comp = torch.gather(
                adj_completion, 1, safe_preds.reshape(B, -1)
            ).reshape(B, N, max_preds_dim)
            pred_comp = torch.where(valid_pred, pred_comp, torch.zeros_like(pred_comp))
            max_pred_comp = pred_comp.max(dim=2)[0]

            new_start = torch.max(release, max_pred_comp)
            new_comp = torch.where(
                scheduled,
                self.activity_end_time,
                torch.max(new_start + avg_pt, best_ect)  # DAG 경로 vs 경합 중 tighter
            )
            new_comp = torch.where(valid, new_comp, torch.zeros_like(new_comp))

            if torch.equal(new_comp, adj_completion):
                break
            adj_completion = new_comp

        # ================================================================
        # Phase 7: Project completion → Total Tardiness
        # ================================================================
        proj_range = torch.arange(self.N_P, device=device)
        proj_mask = (self.activity_project.unsqueeze(1) == proj_range.view(1, -1, 1)) & valid.unsqueeze(1)  # (B, N_P, N)

        comp_3d = adj_completion.unsqueeze(1).expand(-1, self.N_P, -1)  # (B, N_P, N)
        masked = torch.where(proj_mask, comp_3d, torch.tensor(-float('inf'), device=device))
        proj_completion = masked.max(dim=2)[0]  # (B, N_P)
        proj_completion = torch.where(
            torch.isinf(proj_completion),
            torch.zeros_like(proj_completion),
            proj_completion
        )

        return torch.clamp(proj_completion - self.project_due_date, min=0.0).sum(dim=1)  # (B,)

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
        현재 상태를 DANIEL 모델 입력 형태로 반환

        Returns:
            state: EnvState 데이터클래스
        """
        return self._get_state_daniel()

    def _get_state_daniel(self):
        """
        현재 상태를 DANIEL 모델 입력 형태로 반환 (완전 벡터화)

        Action space: (activity, team) 쌍. Candidate = 전체 N개 activity (마스킹으로 제어)

        RCMPSP에 맞게 적응된 피처:
            fea_act: (B, N, 12) - activity feature vectors
            act_mask: (B, N, 3) - predecessor/successor attention mask
            fea_team: (B, T, 8) - team feature vectors
            team_mask: (B, T, T) - team attention mask
            comp_idx: (B, T, T, N) - competition index
            dynamic_pair_mask: (B, N, T) - incompatible pair mask
            candidate: (B, N) - activity identity (0..N-1)
            fea_pairs: (B, N, T, 8) - pair features

        Returns:
            EnvState 객체
        """
        B = self.batch_size
        N = self.max_N_A
        P = self.N_P
        T = self.N_T
        device = self.device
        
        # ========================================
        # 공통 마스크 및 상수
        # ========================================
        act_indices = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        valid_act = act_indices < self.num_activities.unsqueeze(1)  # (B, N)
        current_time = self.sim_time  # (B,)
        max_dur = float(self.duration_max)
        max_time_val = max(1.0, current_time.max().item() + 1.0)
        
        # ========================================
        # 1. Ready mask & Candidate 계산
        # ========================================
        # Ready = not started AND all predecessors ended AND valid
        preds = self.activity_predecessors  # (B, N, max_preds)
        valid_pred_mask = preds >= 0
        safe_preds = preds.clamp(min=0)
        pred_ended = torch.gather(
            self.activity_ended, 1, safe_preds.reshape(B, -1)
        ).reshape(B, N, preds.shape[2])
        all_preds_done = (pred_ended | ~valid_pred_mask).all(dim=2)  # (B, N)
        ready = ~self.activity_started & all_preds_done & valid_act  # (B, N)

        # ---- Schedulable mask (available_actions와 동일 5개 제약) ----
        # Mutex check: 실행 중인 mutex 파트너 없음
        mutex_data = self.activity_mutex  # (B, N, max_mutex)
        max_mutex_dim = mutex_data.shape[2]
        valid_mutex_mask = mutex_data >= 0
        safe_mutex_ids = mutex_data.clamp(min=0)
        mutex_partner_started = torch.gather(
            self.activity_started, 1, safe_mutex_ids.reshape(B, -1)
        ).reshape(B, N, max_mutex_dim)
        mutex_partner_ended = torch.gather(
            self.activity_ended, 1, safe_mutex_ids.reshape(B, -1)
        ).reshape(B, N, max_mutex_dim)
        mutex_running = mutex_partner_started & ~mutex_partner_ended & valid_mutex_mask
        mutex_ok = ~mutex_running.any(dim=2)  # (B, N)

        # Team availability: eligible 팀 중 최소 1개 available
        team_avail_now = self.team_available_time <= current_time.unsqueeze(1)  # (B, T)
        has_avail_team = (self.activity_eligible_teams & team_avail_now.unsqueeze(1)).any(dim=2)  # (B, N)

        # Release time check per activity
        act_proj_safe = self.activity_project.clamp(min=0)  # (B, N)
        act_release = torch.gather(self.project_release_time, 1, act_proj_safe)  # (B, N)
        act_released = act_release <= current_time.unsqueeze(1)  # (B, N)

        # Schedulable = ready + mutex_ok + 팀 가용 + released
        schedulable = ready & mutex_ok & has_avail_team & act_released  # (B, N)

        # 프로젝트 소속 마스크: (B, P, N)
        proj_range = torch.arange(P, device=device).view(1, -1, 1)
        proj_membership = (self.activity_project.unsqueeze(1) == proj_range)  # (B, P, N)

        # Candidate = activity identity (0..N-1). 마스킹은 dynamic_pair_mask에서 처리.
        candidate = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        self.daniel_candidate = candidate
        
        # ========================================
        # 2. 프로젝트 통계 (피처 구성에 사용)
        # ========================================
        proj_valid = proj_membership & valid_act.unsqueeze(1)  # (B, P, N)
        proj_not_started = (~self.activity_started).unsqueeze(1) & proj_valid  # (B, P, N)
        proj_rem_count = proj_not_started.sum(dim=2).float()  # (B, P)
        proj_tot_count = proj_valid.sum(dim=2).float().clamp(min=1)  # (B, P)
        proj_rem_work = (self.activity_duration.unsqueeze(1) * proj_not_started.float()).sum(dim=2)  # (B, P)
        proj_tot_work = (self.activity_duration.unsqueeze(1) * proj_valid.float()).sum(dim=2).clamp(min=1)  # (B, P)
        
        # ========================================
        # 3. fea_act (B, N, 12)
        # ========================================
        # act_proj_safe: 위 schedulable 계산에서 이미 정의됨

        f0 = self.activity_started.float()
        f1 = self.activity_ended.float()
        f2 = self.activity_duration / max_dur

        # Min PT / Span over eligible teams (논문 피처 1, 3 복구)
        elig_mask = self.activity_eligible_teams  # (B, N, T) bool
        min_fill = torch.where(elig_mask, self.activity_team_duration,
                               torch.full_like(self.activity_team_duration, float('inf')))
        min_pt_raw = min_fill.min(dim=2)[0]  # (B, N)
        min_pt = torch.where(min_pt_raw.isinf(), torch.zeros_like(min_pt_raw), min_pt_raw)
        max_fill = torch.where(elig_mask, self.activity_team_duration,
                               torch.full_like(self.activity_team_duration, float('-inf')))
        max_pt_raw = max_fill.max(dim=2)[0]  # (B, N)
        max_pt = torch.where(max_pt_raw.isinf(), torch.zeros_like(max_pt_raw), max_pt_raw)
        f_minpt = min_pt / max_dur          # (B, N) 논문 피처 1
        f_span  = (max_pt - min_pt) / max_dur  # (B, N) 논문 피처 3

        f3 = self.activity_remaining_time / max_dur

        # Predecessor completion ratio
        pred_done_cnt = (pred_ended & valid_pred_mask).sum(dim=2).float()
        total_pred_cnt = valid_pred_mask.sum(dim=2).float().clamp(min=1)
        f4 = pred_done_cnt / total_pred_cnt

        f5 = ready.float()  # currently ready (simplified waiting indicator)

        # f6: C(activity) LB — Estimated completion time lower bound (논문 피처 6)
        # Scheduled: actual end_time (확정값)
        # Unscheduled: release + max(pred LB) + duration (iterative DAG relaxation)
        # preds / valid_pred_mask / safe_preds 는 위에서 이미 계산됨 (재사용)
        completion_lb = torch.where(
            self.activity_started,
            self.activity_end_time,
            act_release + self.activity_duration
        )
        completion_lb = torch.where(valid_act, completion_lb, torch.zeros_like(completion_lb))
        max_preds_dim = preds.shape[2]
        for _ in range(self.N_A_max):
            pred_comp = torch.gather(
                completion_lb, 1, safe_preds.reshape(B, -1)
            ).reshape(B, N, max_preds_dim)
            pred_comp = torch.where(valid_pred_mask, pred_comp, torch.zeros_like(pred_comp))
            max_pred_comp = pred_comp.max(dim=2)[0]  # (B, N)
            new_comp = torch.where(
                self.activity_started,
                self.activity_end_time,
                torch.max(act_release, max_pred_comp) + self.activity_duration
            )
            new_comp = torch.where(valid_act, new_comp, torch.zeros_like(new_comp))
            if torch.equal(new_comp, completion_lb):
                break
            completion_lb = new_comp
        f6 = completion_lb / max(1.0, max_time_val)

        # Project remaining act/work ratios (mapped per activity)
        f7 = torch.gather(proj_rem_count, 1, act_proj_safe) / torch.gather(proj_tot_count, 1, act_proj_safe)
        f8 = torch.gather(proj_rem_work, 1, act_proj_safe) / torch.gather(proj_tot_work, 1, act_proj_safe)
        f9 = self.activity_eligible_teams.sum(dim=2).float() / T

        fea_act = torch.stack([f0, f1, f2, f_minpt, f_span, f3, f4, f5, f6, f7, f8, f9], dim=2)
        fea_act[~valid_act] = 0
        
        # Normalization (FJSP 방식: activity 축 기준 mean/std)
        valid_count = valid_act.sum(dim=1, keepdim=True).float().clamp(min=1)
        fea_sum = (fea_act * valid_act.unsqueeze(2).float()).sum(dim=1, keepdim=True)
        mean_fea = fea_sum / valid_count.unsqueeze(2)
        temp = torch.where(valid_act.unsqueeze(2), fea_act, mean_fea)
        var_fea = ((temp - mean_fea) ** 2).sum(dim=1, keepdim=True) / valid_count.unsqueeze(2)
        std_fea = torch.sqrt(var_fea + 1e-8)
        fea_act = (temp - mean_fea) / std_fea
        fea_act[~valid_act] = 0
        
        # ========================================
        # 4. act_mask (B, N, 3) — attention neighbor mask
        # ========================================
        # slot 0: mask=1 → skip pred_agg  (DAG source: no predecessors)
        # slot 1: self (항상 attend)
        # slot 2: mask=1 → skip succ_agg  (DAG sink: no successors)
        # mask=1 means DON'T attend (FJSP convention)

        # 실제 DAG 구조 기반: 선행자/후행자 존재 여부
        has_pred = (self.activity_predecessors >= 0).any(dim=2)  # (B, N)
        has_succ = (self.activity_successors >= 0).any(dim=2)    # (B, N)

        act_mask = torch.zeros(B, N, 3, device=device)
        # DAG source (선행자 없음): pred_agg는 0-벡터이므로 outer attention에서 제외
        act_mask[:, :, 0] = (~has_pred & valid_act).float()
        # DAG sink (후행자 없음): succ_agg는 0-벡터이므로 outer attention에서 제외
        act_mask[:, :, 2] = (~has_succ & valid_act).float()
        act_mask[~valid_act] = 1.0  # 패딩 activity는 전부 mask
        
        # ========================================
        # 5. dynamic_pair_mask (B, N, T) — True=masked
        # ========================================
        # 기본 마스킹: 패딩, 이미 시작됨, 선행자 미완료, eligibility 없음
        dynamic_pair_mask = (
            ~valid_act.unsqueeze(2)                    # 패딩 activity
            | self.activity_started.unsqueeze(2)       # 이미 시작됨
            | (~all_preds_done).unsqueeze(2)           # 선행자 미완료
            | ~self.activity_eligible_teams            # 팀 eligibility 없음
        )  # (B, N, T)

        # Wait 옵션이 없을 때만 팀 비가용 마스킹
        # Wait 옵션 사용 시 _update_available_actions의 wait_base는 team_ok를 요구하지 않으므로
        # (바쁜 팀에도 미래 시작 시간으로 assign 가능), dynamic_pair_mask도 이에 맞춰 팀 마스크 제거.
        # 이를 제거하지 않으면: 모든 ready activity의 eligible 팀이 전부 바쁠 때
        # 모델이 유효 action이 없어 이미 시작된 활동을 재선택하는 무한루프 발생.
        if not (self.allow_wait_release or self.allow_wait_mutex):
            dynamic_pair_mask = dynamic_pair_mask | ~team_avail_now.unsqueeze(1)

        # Release/mutex 조건 on/off
        if not self.allow_wait_release:
            dynamic_pair_mask = dynamic_pair_mask | ~act_released.unsqueeze(2)
        if not self.allow_wait_mutex:
            dynamic_pair_mask = dynamic_pair_mask | (~mutex_ok).unsqueeze(2)

        # Dominance rule: 대기 pair 중 idle time에 다른 즉시 실행 가능 activity가 있으면 제외
        if (self.allow_wait_release or self.allow_wait_mutex) and self.dominance_rule:
            imm_schedulable = ready & mutex_ok & act_released  # (B, N)
            imm_elig = imm_schedulable.unsqueeze(2) & self.activity_eligible_teams  # (B, N, T)
            imm_dur = torch.where(imm_elig, self.activity_duration.unsqueeze(2).expand(-1, -1, T),
                                  torch.full((B, N, T), float('inf'), device=device))
            min_imm_dur = imm_dur.min(dim=1)[0]  # (B, T)

            # 각 (activity, team) pair의 예상 시작 시간
            est_start = torch.max(
                current_time.unsqueeze(1).unsqueeze(2).expand(-1, N, T),
                self.team_available_time.unsqueeze(1).expand(-1, N, -1)
            )
            if self.allow_wait_release:
                act_release_t = torch.gather(self.project_release_time, 1, act_proj_safe)  # (B, N)
                est_start = torch.max(est_start, act_release_t.unsqueeze(2).expand(-1, -1, T))
            if self.allow_wait_mutex:
                mutex_end_all = self._get_mutex_clear_times()  # (B, N)
                est_start = torch.max(est_start, mutex_end_all.unsqueeze(2).expand(-1, -1, T))

            effective_avail = torch.max(
                current_time.unsqueeze(1).unsqueeze(2).expand(-1, N, T),
                self.team_available_time.unsqueeze(1).expand(-1, N, -1)
            )
            idle_time = (est_start - effective_avail).clamp(min=0)  # (B, N, T)

            needs_wait = idle_time > 0
            dominated = needs_wait & (min_imm_dur.unsqueeze(1).expand(-1, N, -1) <= idle_time)
            dynamic_pair_mask = dynamic_pair_mask | dominated

        # available pairs (inverse of mask)
        avail_pair = ~dynamic_pair_mask  # (B, N, T)

        # ========================================
        # 6. comp_idx (B, T, T, N) — competition index
        # ========================================
        avail_t = avail_pair.permute(0, 2, 1).float()  # (B, T, N)
        comp_idx = (avail_t.unsqueeze(2) * avail_t.unsqueeze(1))  # (B, T, T, N)

        # ========================================
        # 7. team_mask (B, T, T) — team attention mask
        # ========================================
        team_mask = (comp_idx.sum(dim=3) > 0).float()  # (B, T, T)
        diag = torch.arange(T, device=device)
        team_mask[:, diag, diag] = 1.0
        
        # ========================================
        # 9. fea_team (B, T, 8)
        # ========================================
        t0 = avail_pair.sum(dim=1).float() / max(1, N)  # available activity count per team

        unstarted_elig = self.activity_eligible_teams & (~self.activity_started & valid_act).unsqueeze(2)
        t1 = unstarted_elig.sum(dim=1).float() / max(1, N)  # compatible unstarted count

        dur_exp = self.activity_team_duration  # (B, N, T) 팀별 처리 시간 (비적합 팀은 0)
        dur_masked = torch.where(unstarted_elig, dur_exp, torch.full_like(dur_exp, float('inf')))
        t2_raw = dur_masked.min(dim=1)[0]
        t2 = torch.where(t2_raw.isinf(), torch.zeros_like(t2_raw), t2_raw / max_dur)
        
        dur_sum = (dur_exp * unstarted_elig.float()).sum(dim=1)
        dur_cnt = unstarted_elig.sum(dim=1).float().clamp(min=1)
        t3 = (dur_sum / dur_cnt) / max_dur
        
        t4 = torch.clamp(current_time.unsqueeze(1) - self.team_available_time, min=0) / max_time_val
        t5 = torch.clamp(self.team_available_time - current_time.unsqueeze(1), min=0) / max_dur
        t6 = self.team_available_time / max_time_val
        t7 = (self.team_available_time > current_time.unsqueeze(1)).float()
        
        fea_team = torch.stack([t0, t1, t2, t3, t4, t5, t6, t7], dim=2)
        
        # Team feature normalization
        mean_team = fea_team.mean(dim=1, keepdim=True)
        std_team = fea_team.std(dim=1, keepdim=True).clamp(min=1e-8)
        fea_team = (fea_team - mean_team) / std_team
        
        # ========================================
        # 9. fea_pairs (B, N, T, 8) — pair features (activity-team)
        # ========================================
        # act_pt: (B, N, T) — 팀별 처리 시간 (비적합 팀은 이미 0)
        eligible = self.activity_eligible_teams  # (B, N, T)
        act_pt = self.activity_team_duration * eligible.float()  # (B, N, T)

        max_act_pt = act_pt.max(dim=2, keepdim=True)[0].clamp(min=1e-8)
        team_max_pt = act_pt.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        global_max_pt = act_pt.max().clamp(min=1e-8)

        p0 = act_pt / max_dur
        p1 = act_pt / max_act_pt
        p2 = act_pt / team_max_pt
        p3 = act_pt / global_max_pt

        # team remaining work ratio
        team_rem = torch.clamp(self.team_available_time - current_time.unsqueeze(1), min=0)  # (B, T)
        p4 = act_pt / (team_rem.unsqueeze(1) + act_pt + 1e-8)

        # estimated completion time ratio
        est_start_p = torch.max(
            current_time.unsqueeze(1).unsqueeze(2).expand(-1, N, T),
            self.team_available_time.unsqueeze(1).expand(-1, N, -1)
        )
        est_comp = est_start_p + act_pt
        # 프로젝트 due date per activity
        act_proj_id = act_proj_safe.clamp(min=0, max=self.N_P - 1)  # (B, N)
        act_due = torch.gather(self.project_due_date, 1, act_proj_id)  # (B, N)
        p5 = (act_due.unsqueeze(2) - est_comp) / max(1.0, max_time_val)  # slack

        # project remaining work ratio per activity
        act_proj_rem = torch.gather(proj_rem_work, 1, act_proj_id).unsqueeze(2).expand(-1, -1, T).clamp(min=1e-8)
        p6 = act_pt / act_proj_rem

        # pair wait time
        team_wait = torch.clamp(
            self.team_available_time.unsqueeze(1) - current_time.unsqueeze(1).unsqueeze(2), min=0
        )
        act_release_time = torch.gather(self.project_release_time, 1, act_proj_id)  # (B, N)
        release_wait = torch.clamp(
            act_release_time.unsqueeze(2) - current_time.unsqueeze(1).unsqueeze(2), min=0
        )
        p7 = (team_wait + release_wait) / max(1.0, max_time_val)

        # mutex wait time (allow_wait_mutex=True일 때 추가)
        if self.allow_wait_mutex:
            mutex_clear = self._get_mutex_clear_times()  # (B, N)
            mutex_wait = torch.clamp(mutex_clear.unsqueeze(2) - current_time.unsqueeze(1).unsqueeze(2), min=0)
            p7 = (team_wait + release_wait + mutex_wait) / max(1.0, max_time_val)

        fea_pairs = torch.stack([p0, p1, p2, p3, p4, p5, p6, p7], dim=3)
        fea_pairs[dynamic_pair_mask] = 0
        
        # ========================================
        # EnvState 생성
        # ========================================
        state = EnvState(
            fea_act_tensor=fea_act,
            act_mask_tensor=act_mask.float(),
            fea_team_tensor=fea_team,
            team_mask_tensor=team_mask,
            dynamic_pair_mask_tensor=dynamic_pair_mask,
            comp_idx_tensor=comp_idx,
            candidate_tensor=candidate,
            fea_pairs_tensor=fea_pairs,
            pred_idx_tensor=self.activity_predecessors,  # (B, N, max_preds)
            succ_idx_tensor=self.activity_successors,    # (B, N, max_succs)
        )
        return state