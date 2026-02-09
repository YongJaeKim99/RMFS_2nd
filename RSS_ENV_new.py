import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data

import itertools
import math
import pandas as pd
import os
import copy
from collections import defaultdict
from itertools import combinations

from data_generator import *

class RSSEnv():
    def __init__(self, env_params, debug_env=None, sequence_encoder=None):
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_params = env_params
        # debug_env가 명시적으로 전달되지 않으면 env_params에서 가져옴
        if debug_env is not None:
            self.debug_env = debug_env
        else:
            self.debug_env = env_params.get('debug_env', False)
        
        # debug_env_verbose: 가능한 액션 리스트 등 상세 정보 출력
        self.debug_env_verbose = env_params.get('debug_env_verbose', False)
        
        # debug_action_num: 매 스텝마다 가능한 액션 수 / 전체 액션 수 출력
        self.debug_action_num = env_params.get('debug_action_num', False)
        
        # use_fcfs_sort: LS 도착 시 로봇을 FCFS 순서로 정렬
        self.use_fcfs_sort = env_params.get('use_fcfs_sort', False)
        
        # action_space: 에이전트가 선택하는 액션 종류
        self.action_space = env_params.get('action_space', 'item-cp')
        
        # order_selection_rule: CP에 order가 할당되지 않은 경우 order 선택 룰 (action_space=='item-cp'일 때 사용)
        self.order_selection_rule = env_params.get('order_selection_rule', 'smallest')
        
        # cp_selection_rule: CP 자동 선택 룰 (action_space=='item-order'일 때 사용)
        self.cp_selection_rule = env_params.get('cp_selection_rule', 'nearest')
        
        # item_selection_rule: Item 자동 선택 룰 (action_space=='order-cp'일 때 사용)
        self.item_selection_rule = env_params.get('item_selection_rule', 'nearest')
        
        # order-cp 전용 룰 설정 (룰 기반 정책에서 사용)
        self.order_cp_order_selection_rule = env_params.get('order_cp_order_selection_rule', 'fewest_remaining_items')
        self.order_cp_cp_selection_rule = env_params.get('order_cp_cp_selection_rule', 'nearest')
        self.order_cp_priority = env_params.get('order_cp_priority', 'order_first')
        
        # backward_trip_rule: 로봇이 CP에서 LS로 돌아갈 때 LS 선택 룰
        # 'nearest': 가장 가까운 LS 선택 (기존 방식)
        # 'balance': 처리해야 할 아이템이 가장 많은 LS 우선, tie는 nearest
        self.backward_trip_rule = env_params.get('backward_trip_rule', 'nearest')
        
        # objective: 목적함수 설정
        # 'makespan': 모든 주문 완료까지 걸린 시간 (sim_t) - 최소화
        # 'throughput': 모든 아이템의 throughput time 합 - 최소화
        # 'order_throughput': 모든 주문의 clear time 합 - 최소화
        self.objective = env_params.get('objective', 'makespan')
        
        # 시퀀스 인코더 (LSTM/GRU) - 모델에서 전달됨
        self.sequence_encoder = sequence_encoder
        
        # train params
        self.batch = self.env_params['batch_size']
        self.pomo = self.env_params['pomo_size']
        self.batch_size = self.batch * self.pomo
        
        # RSS 시스템 파라미터
        self.N_O = self.env_params['N_O']  # 주문 수
        self.N_S = self.env_params['N_S']  # SKU 타입 수
        self.N_C = self.env_params['N_C']  # Collection Point 수
        self.N_R = self.env_params['N_R']  # 로봇 수
        self.N_L = self.env_params['N_L']  # Loading Station 수
        self.W = self.env_params['W']      # Aisle 개수
        self.L = self.env_params['L']      # Cross-aisle 개수
        self.n_east_ls = self.env_params['n_east_ls']    # 동쪽 LS 개수
        self.n_north_ls = self.env_params['n_north_ls']  # 북쪽 LS 개수
        self.path_topology = self.env_params['path_topology']
        self.LT = self.env_params['LT']  # 로딩 시간
        self.UT = self.env_params['UT']  # 언로딩 시간
        self.look_ahead = self.env_params['look_ahead']
        
        # 물리적 파라미터
        self.vm = self.env_params['vm']  # 로봇 속도 (m/s)
        self.tt = self.env_params['tt']  # 90도 회전 시간 (s)

    def _reset(self, batch, pomo):
        if pomo == -1:
            #self.problem = batch
            # RSS 데이터 구조에 맞게 수정 필요
            self.batch_size = self.problem['order_sku_requirements'].shape[0]
        else:
            self.problem = generate_rss_data_batch(self.env_params)

        self.BATCH_IDX = torch.arange(self.batch_size, dtype=torch.long)
        self.LS_IDX = torch.arange(self.N_L, dtype=torch.long)
        self.CP_IDX = torch.arange(self.N_C, dtype=torch.long)
        
        # env_params 정보
        #self.env_params = self.problem['env_params']

        self.N_I_max = self.problem['max_N_I']

        self.N_I = self.problem['N_I_for_batch']
        
        # Step counter for each batch
        self.step_count = torch.zeros(self.batch_size, dtype=torch.long)
        
        # 로봇 정보       
        #static info
        self.robot_initial_position = torch.tensor(self.problem['robot_initial_position']).long()  # (batch_size, N_R)
        #dynamic info
        self.robot_state = torch.full((self.batch_size, self.N_R), 3, dtype=torch.long)  # 0=READY_TO_PICK, 1=MOVE_CP, 2=MOVE_LS, 3=QUEUE_WAIT, 4=CP_WAIT, 5 = ITEM_WAIT
        self.robot_now_remain_t = torch.zeros(self.batch_size, self.N_R)  # (batch_size, N_R)
        self.robot_current_ls = self.robot_initial_position.clone()
        self.robot_current_cp = -torch.ones(self.batch_size, self.N_R, dtype=torch.long)
        self.robot_carrying_item = -torch.ones(self.batch_size, self.N_R, dtype=torch.long)
        self.robot_target_item = -torch.ones(self.batch_size, self.N_R, dtype=torch.long)  # 로봇이 픽업하러 가기로 할당받은 아이템 (CP->LS 이동 시)
        
        # 각 LS에 로봇이 할당된 누적 횟수 (Rule-based backward trip용)
        # Forward trip (LS->CP)은 에이전트가 결정하지만, 
        # Backward trip (CP->LS)은 가장 적게 할당된 LS로 보내는 룰 사용
        # - (batch_size, N_L)
        self.ls_assignment_count = torch.zeros(self.batch_size, self.N_L, dtype=torch.long)
        
        # 초기 로봇 위치를 카운트에 반영
        for b in range(self.batch_size):
            for r in range(self.N_R):
                initial_ls = self.robot_initial_position[b, r].item()
                self.ls_assignment_count[b, initial_ls] += 1
        
        # item 정보
        #static info
        self.item_sku_type = torch.tensor(self.problem['item_sku_type']).long()  # (batch_size, N_I)
        self.item_loading_station = torch.tensor(self.problem['item_loading_station']).long()  # (batch_size, N_I)
        self.item_initial_position = torch.tensor(self.problem['item_position']).long()  # (batch_size, N_I) - 각 아이템의 LS 내 위치
        self.item_successor = torch.tensor(self.problem['item_successor']).long()  # (batch_size, N_I) - 각 아이템의 다음 아이템
        
        # 아이템 쌍 간의 SKU 타입 일치 여부 (batch_size, N_I, N_I)
        # 벡터화 연산: item_sku_type을 확장하여 비교
        sku_expanded_i = self.item_sku_type.unsqueeze(2).expand(-1, -1, self.N_I_max)  # (batch_size, N_I, N_I)
        sku_expanded_j = self.item_sku_type.unsqueeze(1).expand(-1, self.N_I_max, -1)  # (batch_size, N_I, N_I)
        self.item_has_same_sku = (sku_expanded_i == sku_expanded_j)  # (batch_size, N_I, N_I)

        #dynamic info
        self.item_current_position = self.item_initial_position.clone()  # item_initial_position으로 초기화 (batch_size, N_I)
        self.item_completed = torch.zeros(self.batch_size, self.N_I_max, dtype=torch.bool)
        self.item_current_cp = -torch.ones(self.batch_size, self.N_I_max, dtype=torch.long)  # 아이템이 현재 위치한 CP
        self.item_start_time = torch.zeros(self.batch_size, self.N_I_max)
        self.item_end_time = torch.zeros(self.batch_size, self.N_I_max)
        self.item_assigned_robot = -torch.ones(self.batch_size, self.N_I_max, dtype=torch.long) # 아이템에 할당된 로봇 (픽업 예정/진행 중)
        self.item_assigned_order = -torch.ones(self.batch_size, self.N_I_max, dtype=torch.long) # 아이템에 할당된 주문 (CP 전달 시 결정)
        self.item_cp_start_time = torch.zeros(self.batch_size, self.N_I_max)  # CP에서 언로드 시작 시간
        self.item_visible = (self.item_current_position < self.look_ahead)

        # LS 정보
        #static info
        self.ls_item_queue = torch.tensor(self.problem['ls_item_queue']).long()
       
        #dynamic info
        self.ls_available_time = torch.zeros(self.batch_size, self.N_L)
        self.ls_item_queue_lookahead = self.ls_item_queue[:, :, :self.look_ahead].clone()
        # 각 LS별로 현재 lookahead에서 다음에 가져올 아이템의 인덱스 추적
        self.ls_item_queue_index = torch.full((self.batch_size, self.N_L), self.look_ahead, dtype=torch.long)
        # 각 배치의 각 LS별 대기 로봇 큐: [batch][ls][robots]
        self.ls_robot_queue = torch.tensor(self.problem['ls_robot_queue']).long()        
        
        self.ls_remaining_time = torch.zeros(self.batch_size, self.N_L)  # (batch_size, N_L)
        
        # CP 정보
        self.cp_available_time = torch.zeros(self.batch_size, self.N_C)  # (batch_size, N_C) - CP가 다음 작업을 시작할 수 있는 절대 시간
        
        # 디버그: 초기 로봇 배치 및 큐 상태 출력
        if self.debug_env and self.batch_size > 0:
            print(f"\n🤖 [Initial Robot Distribution]")
            for b in range(min(1, self.batch_size)):  # 첫 번째 배치만 출력
                print(f"  Batch {b}:")
                total_robots = 0
                for ls_idx in range(self.N_L):
                    robots_at_ls = self.ls_robot_queue[b, ls_idx]
                    robot_list = [r.item() for r in robots_at_ls if r >= 0]
                    total_robots += len(robot_list)
                    print(f"    LS{ls_idx}: {len(robot_list)} robots → {robot_list}")
                print(f"  Total: {total_robots} robots across {self.N_L} loading stations")
                print(f"  Expected items: {self.N_I[b].item()}\n")

        # Order 정보

        #static info
        self.order_sku_requirements = torch.tensor(self.problem['order_sku_requirements']).long()  # (batch_size, N_O, N_S)

        #dynamic info
        self.order_sku_arrived = torch.zeros(self.batch_size, self.N_O, self.N_S, dtype=torch.long)  # 주문별 SKU 도착 개수
        self.order_sku_reserved = torch.zeros(self.batch_size, self.N_O, self.N_S, dtype=torch.long)  # 주문별 SKU 예약 개수
        self.order_completed = torch.zeros(self.batch_size, self.N_O, dtype=torch.bool)  # 주문 완료 (모든 SKU가 예약됨) 여부
        self.order_completion_time = torch.zeros(self.batch_size, self.N_O)  # 주문 완료 시간
        self.order_cleared = torch.zeros(self.batch_size, self.N_O, dtype=torch.bool)  # 주문 처리 완료 (모든 SKU가 CP에 실제로 도착) 여부
        self.order_cleared_time = torch.zeros(self.batch_size, self.N_O)  # 주문 처리 완료 시간
        #self.order_activate_start_time = torch.zeros(self.batch_size, self.N_O)  # 주문 활성화 시작 시간
        #self.order_activate_end_time = torch.zeros(self.batch_size, self.N_O)  # 주문 활성화 종료 시간
        self.order_confirmed = torch.zeros(self.batch_size, self.N_O, dtype=torch.bool)  # 주문 CP 확정 여부
        self.order_confirmed_cp = -torch.ones(self.batch_size, self.N_O, dtype=torch.long)  # 주문이 확정된 CP 인덱스

        #주문 관리
        self.cp_confirmed_order_idx = -torch.ones(self.batch_size, self.N_C, dtype=torch.long)  # (batch_size, N_C)

        #SKU 정보

        #static info
        self.sku_total_count = self.order_sku_requirements.sum(dim=1)  # (batch_size, N_S), 각 SKU 별 개수
        
        # 각 배치별 SKU 노드 수 계산 (order_sku_requirements에서 0보다 큰 값들의 개수)
        self.N_SKU = torch.zeros(self.batch_size, dtype=torch.long)
        for b in range(self.batch_size):
            self.N_SKU[b] = (self.order_sku_requirements[b] > 0).sum().item()
        
        # 모든 배치에서 최대 SKU 노드 수 계산
        self.N_SKU_max = self.N_SKU.max().item()

        #dynamic info
        self.sku_remaining_count = self.sku_total_count.clone()  # 남아 있는 개수 (예약되지 않은)

        # SKU-Order 매핑 정보 초기화 (Order 순서로 구성)
        self.sku_order_order_idx = torch.full((self.batch_size, self.N_SKU_max), -1, dtype=torch.long)
        self.sku_order_sku_type = torch.full((self.batch_size, self.N_SKU_max), -1, dtype=torch.long)
        self.sku_order_qty = torch.full((self.batch_size, self.N_SKU_max), 0, dtype=torch.long)
        
        # Order 순서로 SKU 정보 구성: Order 0 (SKU 0,1,2,...), Order 1 (SKU 0,1,2,...), ...
        for b in range(self.batch_size):
            sku_order_idx = 0
            for order_idx in range(self.N_O):
                for sku_type in range(self.N_S):
                    required_qty = self.order_sku_requirements[b, order_idx, sku_type].item()
                    if required_qty > 0 and sku_order_idx < self.N_SKU_max:
                        self.sku_order_order_idx[b, sku_order_idx] = order_idx
                        self.sku_order_sku_type[b, sku_order_idx] = sku_type
                        self.sku_order_qty[b, sku_order_idx] = required_qty
                        sku_order_idx += 1

        # travel time 정보
        self.travel_time_ls_to_cp = torch.tensor(self.problem['travel_time_ls_to_cp']).float()  # (batch_size, N_L, N_C)
        self.travel_time_cp_to_ls = torch.tensor(self.problem['travel_time_cp_to_ls']).float()  # (batch_size, N_C, N_L)
        self.travel_time_ls_to_ls = torch.tensor(self.problem['travel_time_ls_to_ls']).float()  # (batch_size, N_L, N_L)        

        # 시뮬레이션 시간 (dynamic info)
        self.sim_t = torch.zeros(self.batch_size)
        self.prev_obj = torch.zeros(self.batch_size)  # PPO를 위한 이전 objective value
        
        # ═══ 정규화를 위한 최대값 계산 ═══
        # 1. item_current_position 최대값: look_ahead를 기준으로 설정
        self.max_item_position = float(self.look_ahead)
        
        # 2. travel_time 최대값: forward와 backward 구분
        self.max_forward_travel_time = self.travel_time_ls_to_cp.max().item()  # LS → CP
        self.max_backward_travel_time = self.travel_time_cp_to_ls.max().item()  # CP → LS
        
        # 3. robot_arrival_upper_bound 최대값: 최대 forward travel time + backward travel time + loading/unloading time
        self.max_robot_time = self.max_forward_travel_time + self.max_backward_travel_time + self.LT + self.UT
        
        # 4. ls_remaining_time 최대값: loading time (LT)
        self.max_ls_time = self.LT
        
        # 5. order_sku_remaining_items 최대값: 모든 Order-SKU의 요구량 중 최대값
        self.max_order_sku_remaining = self.problem['order_sku_requirements'].max().item()
        
        # 6. order_remaining_items 최대값: 모든 Order의 총 요구량 중 최대값
        order_total_requirements = self.problem['order_sku_requirements'].sum(dim=2)  # (batch_size, N_O)
        self.max_order_remaining = order_total_requirements.max().item()
        
        # 참고: SKU와 CP feature는 이제 충족률(0~1)을 사용하므로 정규화 불필요
        # max_sku_quantity, max_order_items는 더 이상 사용 안 함
        
        # Action availability masks (동적으로 업데이트됨)
        # 초기화는 _get_init_state 이후에 수행 (action_index_decoder가 필요)
        
        # 배치 내 각 인스턴스별 action 개수 계산
        self._calculate_max_action_sizes()
        
        self._get_init_state()
        
        # Action availability masks 초기화 (텐서로 변경)
        self.decision_available = torch.zeros(self.batch_size, self.max_num_actions, dtype=torch.bool)  # (batch_size, max_num_actions)
        
        # order-cp action space를 위한 selected item 저장 (중복 계산 방지)
        self.order_cp_selected_item = torch.full((self.batch_size,), -1, dtype=torch.long)  # (batch_size,)
        
        # 로봇-아이템 직접 할당 제거: LS에 로봇이 있으면 해당 LS의 아이템 운반 가능
        # (item_assigned_robot 사용 안 함)

        # 초기 상태의 가능한 액션 업데이트 (첫 액션을 위해)
        self._update_available_actions(self.BATCH_IDX)

    def _calculate_max_action_sizes(self):
        """
        배치 내 각 인스턴스별 액션 개수를 계산하고 배치 전체에서 최대값을 구함
        액션 = action_space에 따라 달라짐
        """
        action_sizes = []
        
        for b in range(self.batch_size):
            if self.action_space == 'item-order-cp':
                # 액션: (Item, SKU-Order, CP) 조합
                num_actions = 0
                for item_idx in range(self.N_I[b]):
                    item_sku_type = self.item_sku_type[b, item_idx].item()
                    # 해당 SKU 타입을 가진 Order-SKU 조합 찾기
                    for sku_order_idx in range(self.N_SKU_max):
                        if self.sku_order_order_idx[b, sku_order_idx] == -1:
                            break
                        if self.sku_order_sku_type[b, sku_order_idx] == item_sku_type:
                            num_actions += self.N_C  # CP 개수만큼 액션 생성
                action_sizes.append(num_actions)
            
            elif self.action_space == 'item-cp':
                # 액션: (Item, CP) 조합
                num_actions = self.N_I[b].item() * self.N_C
                action_sizes.append(num_actions)
            
            elif self.action_space == 'item-order':
                # 액션: (Item, SKU-Order) 조합
                num_actions = 0
                for item_idx in range(self.N_I[b]):
                    item_sku_type = self.item_sku_type[b, item_idx].item()
                    # 해당 SKU 타입을 가진 Order-SKU 조합 찾기
                    for sku_order_idx in range(self.N_SKU_max):
                        if self.sku_order_order_idx[b, sku_order_idx] == -1:
                            break
                        if self.sku_order_sku_type[b, sku_order_idx] == item_sku_type:
                            num_actions += 1
                action_sizes.append(num_actions)
            
            elif self.action_space == 'order-cp':
                # 액션: (SKU-Order, CP) 조합
                # 모든 SKU-Order와 모든 CP의 조합
                num_sku_orders = 0
                for sku_order_idx in range(self.N_SKU_max):
                    if self.sku_order_order_idx[b, sku_order_idx] == -1:
                        break
                    num_sku_orders += 1
                num_actions = num_sku_orders * self.N_C
                action_sizes.append(num_actions)
        
        # 배치 전체에서 최대값
        self.max_num_actions = max(action_sizes)

    def _get_obj(self):
        """
        RSS 환경의 objective function
        
        Returns:
            objective value (batch_size,)
            - makespan: 시뮬레이션 시간 (모든 주문 완료까지 걸린 시간)
            - throughput: 모든 아이템의 throughput time 합 (sum of item_end_time - item_start_time)
            - order_throughput: 모든 주문의 clear time 합 (total order completion time)
        """
        objective = self.env_params.get('objective', 'makespan')
        
        if objective == 'makespan':
            return self.sim_t  # shape: (batch_size,)
        
        elif objective == 'throughput':
            # 각 배치별로 아이템의 throughput time 합 계산
            throughput_times = []
            for b in range(self.batch_size):
                N_I = self.N_I[b].item()
                # 해당 배치의 실제 아이템들만 선택
                item_throughput = self.item_end_time[b, :N_I] - self.item_start_time[b, :N_I]
                total_throughput = item_throughput.sum()
                throughput_times.append(total_throughput)
            
            return torch.stack(throughput_times)  # shape: (batch_size,)
        
        elif objective == 'order_throughput':
            # 각 배치별로 주문의 clear time 합 계산
            order_throughput_times = []
            for b in range(self.batch_size):
                # 해당 배치의 모든 주문의 cleared time 합
                total_order_throughput = self.order_cleared_time[b, :self.N_O].sum()
                order_throughput_times.append(total_order_throughput)
            
            return torch.stack(order_throughput_times)  # shape: (batch_size,)
        
        else:
            raise ValueError(f"Unknown objective: {objective}. Choose 'makespan', 'throughput', or 'order_throughput'.")

    def step(self, action: torch.Tensor, use_step_reward=False):
        """
        Args:
            action: 선택된 액션
            use_step_reward: True이면 매 step마다 목적함수 변화량을 reward로 반환 (PPO용)
                            False이면 에피소드 끝에만 최종 reward 반환 (REINFORCE용)
        """
        # synchronize to catch indexing errors on CUDA
        #torch.cuda.synchronize()
        
        # Step counter 증가
        self.step_count[self.BATCH_IDX] += 1

        # (Item, SKU-Order, CP) 조합 액션 실행
        self._step_ls_to_cp(action, self.BATCH_IDX)

        if self.debug_env:
            print(f"Step {self.step_count[self.BATCH_IDX]}")
        
        # 각 배치별로 모든 주문이 완료되었는지 확인
        batches_not_completed = []
        for b in self.BATCH_IDX:
            # 모든 주문이 완료되었는지 확인
            orders_done = self.order_completed[b].all().item()
            
            if not orders_done:
                batches_not_completed.append(b.item())
        
        # 완료되지 않은 주문이 있는 배치들만 액션 마스크 업데이트
        # (주석 처리: move_next_state()에서 처리하므로 중복)
        # if len(batches_not_completed) > 0:
        #     batches_not_completed_tensor = torch.tensor(batches_not_completed, dtype=torch.long)
        #     self._update_available_actions(batches_not_completed_tensor)

        # 모든 배치에 대해 다음 상태로 이동
        all_done = self.move_next_state(self.BATCH_IDX)
        
        # PPO용 step reward 계산
        if use_step_reward:
            # 현재 목적함수 값 계산
            current_obj = self._get_obj()
            # 목적함수 변화량 = -(현재 obj - 이전 obj)
            # 목적함수가 증가하면 negative reward, 감소하면 positive reward
            step_reward = -(current_obj - self.prev_obj)
            self.prev_obj = current_obj.clone()
        
        # check done - move_next_state에서 반환된 all_done 사용
        if all_done:
            if use_step_reward:
                # PPO: 매 step reward를 반환
                return None, step_reward, True
            else:
                # REINFORCE: 최종 reward만 반환
                reward = -self._get_obj()
                return None, reward, True
        else:
            self.state = self._get_state()
            if use_step_reward:
                # PPO: 매 step reward를 반환
                return self.state, step_reward, False
            else:
                # REINFORCE: 중간 step에서는 reward 없음
                return self.state, None, False

    def move_next_state(self, batch_idxs: torch.Tensor):
        """
        시뮬레이터 로직에 따라 다음 의사결정 이벤트까지 시간 진행
        배치별로 처리하며, 의사결정이 필요한 시점에서 멈춤
        
        Args:
            batch_idxs: 처리할 배치 인덱스들
            
        Returns:
            bool: all_done - 모든 배치의 order가 cleared 되었는지 여부
        """
        max_iterations = 1000  # 무한 루프 방지
        iterations = 0
        
        # # 교착 상태 감지를 위한 변수들
        # prev_sim_times = self.sim_t[batch_idxs].clone()
        # no_progress_count = 0
        # max_no_progress = 10  # 시간 진행 없이 10번 반복되면 교착 상태로 판단
        
        # 처리할 배치들 마스킹
        batch_mask = torch.zeros(self.batch_size, dtype=torch.bool)
        batch_mask[batch_idxs] = True
        
        # 루프 시작 전 초기 액션 업데이트
        self._update_available_actions(batch_idxs)
        
        while iterations < max_iterations:
            iterations += 1
            
            # 각 배치별로 order_completed와 order_cleared 상태 확인
            batches_completed = []  # order가 completed된 배치들
            batches_not_completed = []  # order가 아직 completed 안된 배치들
            batches_cleared = []  # order가 cleared된 배치들
            
            for b in batch_idxs:
                b_idx = b.item()
                orders_completed = self.order_completed[b].all().item()
                orders_cleared = self.order_cleared[b].all().item()
                
                if orders_cleared:
                    batches_cleared.append(b_idx)
                elif orders_completed:
                    batches_completed.append(b_idx)
                else:
                    batches_not_completed.append(b_idx)
            
            # 모든 배치가 cleared 되었는지 확인
            all_done = (len(batches_cleared) == len(batch_idxs))
            
            if all_done:
                # 모든 주문 cleared → 완료
                break            
            
            # 시간을 진행해야 할 배치들 결정
            batches_to_advance = []
            
            # 1. completed된 배치들 (clear될 때까지 시간 진행)
            batches_to_advance.extend(batches_completed)
            
            # 2. completed 안된 배치들 중 액션이 없는 배치들 (다음 의사결정까지 시간 진행)
            for b_idx in batches_not_completed:
                if not self.decision_available[b_idx].any().item():
                    batches_to_advance.append(b_idx)
            
            if len(batches_to_advance) > 0:
                batches_to_advance_tensor = torch.tensor(batches_to_advance, dtype=torch.long)
                                
                self._advance_to_next_decision_event_for_batches(batches_to_advance_tensor)
                
                # 시간 진행 후 액션 업데이트 (상태가 변경된 배치들만)
                self._update_available_actions(batches_to_advance_tensor)
                
            else:
                # 시간 진행할 배치가 없으면 중단 (모든 배치가 의사결정 가능 상태)
                break                
        
        # 최종 all_done 상태 확인 및 반환
        final_all_done = True
        for b in batch_idxs:
            if not self.order_cleared[b].all().item():
                final_all_done = False
                break
        
        return final_all_done
        
    def get_next_move_t(self, batch_idxs):
        """
        반환값: 최소 이동 시간 (로봇, LS, CP 모두 고려)
        지정된 배치들에서 다음 이벤트까지의 최소 시간을 배치별로 torch.Tensor로 반환
        """
        # 배치 인덱스에 해당하는 모든 시간 값들을 배치별로 가져오기
        robot_times = self.robot_now_remain_t[batch_idxs, :]  # (len(batch_idxs), N_R)
        ls_times = self.ls_remaining_time[batch_idxs, :]      # (len(batch_idxs), N_L)
        # CP available_time은 절대 시간이므로 다음 이벤트 계산에서 제외
        # (CP는 로봇 도착 시 처리되므로 별도 이벤트 불필요)
        
        # 각 배치별로 모든 시간 값들을 결합
        all_times = torch.cat([robot_times, ls_times], dim=1)  # (len(batch_idxs), N_R + N_L)
        
        # 0보다 큰 시간들만 고려하여 각 배치별 최소값 계산
        # 0 이하인 값들을 매우 큰 값으로 대체하여 최소값 계산에서 제외
        masked_times = torch.where(all_times > 0, all_times, torch.tensor(float('inf')))
        
        # 각 배치별 최소값 계산
        batch_mins, min_indices = masked_times.min(dim=1)  # (len(batch_idxs),)
        
        # inf인 경우 (모든 시간이 0 이하) 0.0으로 대체
        batch_mins = torch.where(batch_mins == float('inf'), torch.tensor(0.0), batch_mins)
        
        # 디버그: 다음 이벤트 정보 출력
        if self.debug_env and len(batch_idxs) > 0:
            for i, b_idx in enumerate(batch_idxs[:1]):  # 첫 번째 배치만
                min_time = batch_mins[i].item()
                min_idx = min_indices[i].item()
                
                if min_idx < self.N_R:
                    event_type = f"Robot{min_idx}"
                else:
                    ls_idx = min_idx - self.N_R
                    event_type = f"LS{ls_idx}"
                
                print(f"    ⏱️ [Next Event] Batch {b_idx.item()}, dt={min_time:.2f}, event={event_type}")
        
        return batch_mins

    def _select_cp_for_item_and_order(self, batch_idx: int, item_idx: int, order_idx: int) -> int:
        """
        주어진 item과 order에 대해 적절한 CP를 선택
        
        Args:
            batch_idx: 배치 인덱스
            item_idx: 아이템 인덱스
            order_idx: 주문 인덱스
            
        Returns:
            cp_idx: 선택된 CP 인덱스 (-1이면 선택 불가)
        """
        # 1. Order가 이미 CP에 할당된 경우
        order_confirmed_cp = self.order_confirmed_cp[batch_idx, order_idx].item()
        if order_confirmed_cp >= 0:
            return order_confirmed_cp
        
        # 2. Order가 아직 CP에 할당되지 않은 경우 → 룰 기반 선택
        ls_idx = self.item_loading_station[batch_idx, item_idx].item()
        
        candidate_cps = []
        for cp_idx in range(self.N_C):
            # CP에 이미 다른 order가 할당되었는지 확인
            cp_confirmed_order = self.cp_confirmed_order_idx[batch_idx, cp_idx].item()
            if cp_confirmed_order == -1:  # 빈 CP만 후보로
                candidate_cps.append(cp_idx)
        
        if len(candidate_cps) == 0:
            return -1  # 사용 가능한 CP 없음
        
        # CP 선택 룰 적용
        if self.cp_selection_rule == 'nearest':
            # 가장 가까운 CP 선택
            min_distance = float('inf')
            selected_cp = candidate_cps[0]
            for cp_idx in candidate_cps:
                distance = self.travel_time_ls_to_cp[batch_idx, ls_idx, cp_idx].item()
                if distance < min_distance:
                    min_distance = distance
                    selected_cp = cp_idx
            return selected_cp
        
        elif self.cp_selection_rule == 'least_loaded':
            # 가장 적게 로드된 CP 선택
            min_load = float('inf')
            selected_cp = candidate_cps[0]
            for cp_idx in candidate_cps:
                load = 0
                for i in range(self.N_I[batch_idx].item()):
                    if self.item_current_cp[batch_idx, i].item() == cp_idx:
                        load += 1
                if load < min_load:
                    min_load = load
                    selected_cp = cp_idx
            return selected_cp
        
        elif self.cp_selection_rule == 'random':
            # 랜덤 CP 선택
            import random
            return random.choice(candidate_cps)
        
        else:
            # 기본값: 첫 번째 후보
            return candidate_cps[0]
    
    def _select_order_for_item_and_cp(self, batch_idx: int, item_idx: int, cp_idx: int) -> int:
        """
        주어진 item과 cp에 대해 적절한 order를 선택
        
        Args:
            batch_idx: 배치 인덱스
            item_idx: 아이템 인덱스
            cp_idx: CP 인덱스
            
        Returns:
            order_idx: 선택된 order 인덱스 (-1이면 선택 불가)
        """
        item_sku_type = self.item_sku_type[batch_idx, item_idx].item()
        
        # 1. CP에 이미 order가 할당된 경우
        cp_confirmed_order = self.cp_confirmed_order_idx[batch_idx, cp_idx].item()
        if cp_confirmed_order >= 0:
            # 해당 order가 이 item의 SKU를 필요로 하는지 확인
            if self.order_is_sku_needed(batch_idx, cp_confirmed_order, item_sku_type):
                return cp_confirmed_order
            else:
                return -1  # 해당 order가 이 SKU를 필요로 하지 않음
        
        # 2. CP에 order가 할당되지 않은 경우
        # 아직 어떤 CP에도 할당되지 않은 order 중에서 이 item을 필요로 하는 order 찾기
        candidate_orders = []
        for order_idx in range(self.N_O):
            # Order가 아직 CP에 할당되지 않았는지 확인
            if self.order_confirmed_cp[batch_idx, order_idx].item() == -1:
                # 이 order가 해당 SKU를 필요로 하는지 확인
                if self.order_is_sku_needed(batch_idx, order_idx, item_sku_type):
                    # 해당 order의 총 item 수 계산
                    total_items = self.order_sku_requirements[batch_idx, order_idx, :].sum().item()
                    candidate_orders.append((order_idx, total_items))
        
        # 후보 order가 없으면 -1 반환
        if len(candidate_orders) == 0:
            return -1
        
        # Order 선택 룰 적용
        if self.order_selection_rule == 'smallest':
            # 총 item 수가 가장 적은 order 선택
            selected_order = min(candidate_orders, key=lambda x: x[1])[0]
        elif self.order_selection_rule == 'largest':
            # 총 item 수가 가장 많은 order 선택
            selected_order = max(candidate_orders, key=lambda x: x[1])[0]
        else:
            # 기본값: 첫 번째 후보 선택
            selected_order = candidate_orders[0][0]
        
        return selected_order    
    
    def _step_ls_to_cp(self, action: torch.Tensor, batch_idxs: torch.Tensor):
        """
        (Item, CP) 조합을 단일 액션으로 선택
        Order는 자동으로 결정됨
        
        두 가지 형태의 액션을 지원:
        1. action.dim() == 1: 액션 인덱스 (기존 방식, GNN 모델용)
        2. action.dim() == 2: 직접 (item, order, cp, ls, robot) 5개 값 전달 (룰 기반용)
        """        

        # 텐서로 변환
        item_idxs = -torch.ones(batch_idxs.shape[0], dtype=torch.long)
        order_idxs = -torch.ones(batch_idxs.shape[0], dtype=torch.long)
        cp_idxs = -torch.ones(batch_idxs.shape[0], dtype=torch.long)
        ls_idxs = -torch.ones(batch_idxs.shape[0], dtype=torch.long)
        robot_idxs = -torch.ones(batch_idxs.shape[0], dtype=torch.long)

        # 액션 형태 체크: 인덱스 vs 직접 값
        if action.dim() == 2 and action.shape[1] == 5:
            # 직접 값 전달 방식 (룰 기반)
            for i, b in enumerate(batch_idxs):
                item_idxs[i] = action[b, 0].item()
                order_idxs[i] = action[b, 1].item()
                cp_idxs[i] = action[b, 2].item()
                ls_idxs[i] = action[b, 3].item()
                robot_idxs[i] = action[b, 4].item()
                
                # 해당 LS의 모든 아이템들의 position을 1씩 감소
                ls_items_mask = (self.item_loading_station[b] == ls_idxs[i])
                self.item_current_position[b, ls_items_mask] = torch.clamp(
                    self.item_current_position[b, ls_items_mask] - 1, min=0)
        
        else:
            # 액션 인덱스 방식 (GNN 모델)
            # 액션 실행 직전 체크 (디버깅)
            if self.debug_env:
                for i, b in enumerate(batch_idxs):
                    action_idx = action[b].item()
                    if action_idx < self.decision_available.shape[1]:
                        is_available = self.decision_available[b, action_idx].item()
                        if not is_available:
                            # print(f"  ⚠️ [ERROR] Batch {b}: Executing unavailable action {action_idx}!")
                            pass

            # action_space에 따라 다르게 처리
            if self.action_space == 'item-order-cp':
                # Action: (Item, SKU-Order, CP) - 모두 선택됨
                for i, b in enumerate(batch_idxs):
                    action_idx = action[b].item()
                    
                    item_node_idx = self.action_index_decoder[b, 0, action_idx].item()
                    sku_node_idx = self.action_index_decoder[b, 1, action_idx].item()
                    cp_node_idx = self.action_index_decoder[b, 2, action_idx].item()
                    
                    item_idx = item_node_idx
                    cp_idx = cp_node_idx - self.N_I_max - self.N_SKU_max
                    sku_order_idx = sku_node_idx - self.N_I_max
                    
                    order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    
                    robot_queue = self.ls_robot_queue[b, ls_idx]
                    robot_idx = -1
                    for r_idx in robot_queue:
                        if r_idx >= 0:
                            robot_idx = r_idx.item()
                            break
                    
                    item_idxs[i] = item_idx
                    order_idxs[i] = order_idx
                    cp_idxs[i] = cp_idx
                    ls_idxs[i] = ls_idx
                    robot_idxs[i] = robot_idx
                    
                    ls_items_mask = (self.item_loading_station[b] == ls_idx)
                    self.item_current_position[b, ls_items_mask] = torch.clamp(
                        self.item_current_position[b, ls_items_mask] - 1, min=0)
            
            elif self.action_space == 'item-cp':
                # Action: (Item, CP) - Order는 룰로 선택
                for i, b in enumerate(batch_idxs):
                    action_idx = action[b].item()

                    item_node_idx = self.action_index_decoder[b, 0, action_idx].item()
                    cp_node_idx = self.action_index_decoder[b, 1, action_idx].item()

                    item_idx = item_node_idx
                    cp_idx = cp_node_idx - self.N_I_max - self.N_SKU_max
                    
                    if action_idx >= self.action_index_decoder.shape[2]:
                        print(f"[ERROR] Batch {b}: action_idx ({action_idx}) >= decoder size ({self.action_index_decoder.shape[2]})")
                    
                    order_idx = self._select_order_for_item_and_cp(b.item(), item_idx, cp_idx)
                    
                    if order_idx == -1:
                        raise ValueError(f"Batch {b}: 유효한 order를 찾을 수 없습니다. item={item_idx}, cp={cp_idx}")
                    
                    ls_idx = self.item_loading_station[b, item_idx].item()

                    robot_queue = self.ls_robot_queue[b, ls_idx]
                    robot_idx = -1
                    for r_idx in robot_queue:
                        if r_idx >= 0:
                            robot_idx = r_idx.item()
                            break
                    
                    item_idxs[i] = item_idx
                    order_idxs[i] = order_idx
                    cp_idxs[i] = cp_idx
                    ls_idxs[i] = ls_idx
                    robot_idxs[i] = robot_idx            
                    
                    ls_items_mask = (self.item_loading_station[b] == ls_idx)
                    self.item_current_position[b, ls_items_mask] = torch.clamp(
                        self.item_current_position[b, ls_items_mask] - 1, min=0)
            
            elif self.action_space == 'item-order':
                # Action: (Item, SKU-Order) - CP는 룰로 선택
                for i, b in enumerate(batch_idxs):
                    action_idx = action[b].item()
                    
                    item_node_idx = self.action_index_decoder[b, 0, action_idx].item()
                    sku_node_idx = self.action_index_decoder[b, 1, action_idx].item()
                    
                    item_idx = item_node_idx
                    sku_order_idx = sku_node_idx - self.N_I_max
                    
                    order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                    
                    # CP를 자동으로 선택
                    cp_idx = self._select_cp_for_item_and_order(b.item(), item_idx, order_idx)
                    
                    if cp_idx == -1:
                        raise ValueError(f"Batch {b}: 유효한 CP를 찾을 수 없습니다. item={item_idx}, order={order_idx}")
                    
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    
                    robot_queue = self.ls_robot_queue[b, ls_idx]
                    robot_idx = -1
                    for r_idx in robot_queue:
                        if r_idx >= 0:
                            robot_idx = r_idx.item()
                            break
                    
                    item_idxs[i] = item_idx
                    order_idxs[i] = order_idx
                    cp_idxs[i] = cp_idx
                    ls_idxs[i] = ls_idx
                    robot_idxs[i] = robot_idx
                    
                    ls_items_mask = (self.item_loading_station[b] == ls_idx)
                    self.item_current_position[b, ls_items_mask] = torch.clamp(
                        self.item_current_position[b, ls_items_mask] - 1, min=0)
            
            elif self.action_space == 'order-cp':
                # Action: (SKU-Order, CP) - Item은 _update_available_actions에서 이미 특정됨
                for i, b in enumerate(batch_idxs):
                    action_idx = action[b].item()
                    
                    sku_node_idx = self.action_index_decoder[b, 0, action_idx].item()
                    cp_node_idx = self.action_index_decoder[b, 1, action_idx].item()
                    
                    sku_order_idx = sku_node_idx - self.N_I_max
                    order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                    cp_idx = cp_node_idx - self.N_I_max - self.N_SKU_max
                    
                    # Item은 _update_available_actions에서 이미 특정되어 저장됨 (중복 계산 방지)
                    item_idx = self.order_cp_selected_item[b].item()
                    
                    if item_idx == -1:
                        raise ValueError(f"Batch {b}: 유효한 Item이 선택되지 않았습니다. order={order_idx}, cp={cp_idx}")
                    
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    
                    robot_queue = self.ls_robot_queue[b, ls_idx]
                    robot_idx = -1
                    for r_idx in robot_queue:
                        if r_idx >= 0:
                            robot_idx = r_idx.item()
                            break
                    
                    item_idxs[i] = item_idx
                    order_idxs[i] = order_idx
                    cp_idxs[i] = cp_idx
                    ls_idxs[i] = ls_idx
                    robot_idxs[i] = robot_idx
                    
                    ls_items_mask = (self.item_loading_station[b] == ls_idx)
                    self.item_current_position[b, ls_items_mask] = torch.clamp(
                        self.item_current_position[b, ls_items_mask] - 1, min=0)
            

        #sanity checks
        assert torch.all(item_idxs >= 0) and torch.all(item_idxs < self.N_I_max), \
            f"item_idx out of bounds: {item_idxs}"
        assert torch.all(order_idxs >= 0) and torch.all(order_idxs < self.N_O), \
            f"order_idx out of bounds: {order_idxs}"
        assert torch.all(cp_idxs >= 0) and torch.all(cp_idxs < self.N_C), \
            f"cp_idx out of bounds: {cp_idxs}"
        assert torch.all(ls_idxs >= 0) and torch.all(ls_idxs < self.N_L), \
            f"ls_idx out of bounds: {ls_idxs}"


        # 시간 계산: 로딩 시작 → 로딩 완료 → CP 도착
        loading_end_time = self.sim_t[batch_idxs] + self.LT
        travel_time = self.travel_time_ls_to_cp[batch_idxs, ls_idxs, cp_idxs]
        arrival_time = loading_end_time + travel_time

        # 로봇 상태 업데이트
        self.robot_state[batch_idxs, robot_idxs] = 1 # 1=MOVE_CP
        #self.robot_now_remain_t[batch_idxs, robot_idxs] = arrival_time - self.sim_t[batch_idxs]
        self.robot_now_remain_t[batch_idxs, robot_idxs] = arrival_time + self.UT - self.sim_t[batch_idxs]
        self.robot_current_ls[batch_idxs, robot_idxs] = -1
        self.robot_current_cp[batch_idxs, robot_idxs] = cp_idxs
        self.robot_carrying_item[batch_idxs, robot_idxs] = item_idxs
        
        # 아이템에 로봇 할당 기록 (간트차트 표시용)
        self.item_assigned_robot[batch_idxs, item_idxs] = robot_idxs
       
        # LS에서 아이템 제거 및 lookahead 큐 업데이트 (벡터 연산)
        # 1. 맨 앞 아이템 제거하고 왼쪽으로 shift (벡터화)
        self.ls_item_queue_lookahead[batch_idxs, ls_idxs, :-1] = self.ls_item_queue_lookahead[batch_idxs, ls_idxs, 1:]
        
        # 2. 마지막 슬롯을 -1로 초기화 (벡터화)
        self.ls_item_queue_lookahead[batch_idxs, ls_idxs, -1] = -1
        
        # 3. 원본 큐에서 다음 아이템 가져오기 (벡터화)
        current_indices = self.ls_item_queue_index[batch_idxs, ls_idxs]  # (B,)
        
        # current_indices가 범위를 넘어가는지 체크
        max_queue_length = self.ls_item_queue.shape[2]  # ls_item_queue의 최대 길이
        valid_indices_mask = current_indices < max_queue_length  # (B,)
        
        # 유효한 인덱스만 사용하여 아이템 가져오기
        next_items = torch.full_like(current_indices, -1)  # 기본값 -1
        if valid_indices_mask.any():
            valid_batch_idxs = batch_idxs[valid_indices_mask]
            valid_ls_idxs = ls_idxs[valid_indices_mask]
            valid_current_indices = current_indices[valid_indices_mask]
            
            valid_next_items = self.ls_item_queue[valid_batch_idxs, valid_ls_idxs, valid_current_indices]
            next_items[valid_indices_mask] = valid_next_items
            
            # visible 설정은 유효한 아이템에만 적용
            self.item_visible[valid_batch_idxs, valid_next_items] = True
        
        self.ls_item_queue_lookahead[batch_idxs, ls_idxs, -1] = next_items

        # 4. 인덱스 카운터 증가 (벡터화) 인덱스 안넘어가게 수정
        self.ls_item_queue_index[batch_idxs, ls_idxs] += 1
        
        # 5. 로봇 큐에서 픽업한 로봇 제거
        for i in range(len(batch_idxs)):
            b = batch_idxs[i].item()
            ls = ls_idxs[i].item()
            robot = robot_idxs[i].item()
            
            # 해당 로봇을 큐에서 찾아서 제거
            robot_queue = self.ls_robot_queue[b, ls]
            # 로봇을 찾아서 제거하고 왼쪽으로 shift
            robot_found = False
            for j in range(len(robot_queue) - 1):
                if robot_queue[j].item() == robot:
                    robot_found = True
                if robot_found:
                    robot_queue[j] = robot_queue[j + 1]
            # 마지막 슬롯을 -1로
            if robot_found or robot_queue[-1].item() == robot:
                robot_queue[-1] = -1
                
            # 디버그: 로봇 큐에서 제거
            if self.debug_env and i < 5:  # 처음 5개만 출력
                sim_t = self.sim_t[b].item()
                remaining_robots = [r.item() for r in robot_queue if r >= 0]
                print(f"       🔹 LS{ls} Queue after pickup: {remaining_robots}")

        # 아이템 시작 시간 기록 (LS에서 픽업 시작)
        self.item_start_time[batch_idxs, item_idxs] = self.sim_t[batch_idxs]
        
        # 디버그: LS 로딩 시작
        if self.debug_env and len(batch_idxs) > 0:
            for i in range(min(5, len(batch_idxs))):  # 처음 5개만 출력
                b = batch_idxs[i].item()
                ls = ls_idxs[i].item()
                item = item_idxs[i].item()
                robot = robot_idxs[i].item()
                sim_t = self.sim_t[b].item()
                prev_ls_time = self.ls_remaining_time[b, ls].item()
                # 로봇 큐 상태 표시
                robot_queue = self.ls_robot_queue[b, ls]
                queue_list = [r.item() for r in robot_queue if r >= 0]
                print(f"    📦 [LS Loading] Batch {b}, LS{ls}, Item{item}, Robot{robot}, t={sim_t:.2f}")
                print(f"       🔹 LS{ls} Queue before pickup: {queue_list}")
                
        # CP에 아이템 예약 처리 (도착 예정으로 등록)
        sku_types = self.item_sku_type[batch_idxs, item_idxs]
        self.reserve_item_and_order(batch_idxs, cp_idxs, item_idxs, sku_types, order_idxs)
        
        
        # 아이템 예약 직후 주문 완성 체크
        self._check_and_complete_orders_at_cp(batch_idxs, cp_idxs, item_idxs, sku_types, order_idxs)

        #done에서 어떻게 처리되는지 (검증 필요)
        self.ls_remaining_time[batch_idxs, ls_idxs] += self.LT
        # 아이템의 현재 CP 저장
        self.item_current_cp[batch_idxs, item_idxs] = cp_idxs
        
        # 로봇-아이템 직접 할당 제거: LS에 로봇이 있으면 해당 LS의 아이템 운반 가능



    def arrive_item(self, batch_idxs, cp_idx, item_idx, sku_types):
        """CP에 아이템 실제 도착 (예약에서 실제로 이전) - 벡터화 버전"""
        # 파라미터로 받은 sku_types 사용 (중복 제거)
        
        # 해당 CP들의 confirmed order 찾기 (벡터화)
        confirmed_orders = self.cp_confirmed_order_idx[batch_idxs, cp_idx]
        
        # 중복된 (batch_idx, order_idx, sku_type) 조합에 대해 올바르게 누적하기 위해 scatter_add_ 사용
        # 인덱스 준비
        indices = torch.stack([batch_idxs, confirmed_orders, sku_types], dim=1)  # (N, 3)
        
        # 각 조합별로 카운트 계산
        unique_indices, counts = torch.unique(indices, dim=0, return_counts=True)
        
        # unique_indices에서 각 차원 분리
        unique_batch_idxs = unique_indices[:, 0]
        unique_confirmed_orders = unique_indices[:, 1]
        unique_sku_types = unique_indices[:, 2]
        
        # 예약에서 도착으로 이전 (중복 고려하여 올바르게 처리)
        self.order_sku_reserved[unique_batch_idxs, unique_confirmed_orders, unique_sku_types] -= counts
        self.order_sku_arrived[unique_batch_idxs, unique_confirmed_orders, unique_sku_types] += counts
        
        # 아이템 완료 시간 기록 (CP에서 언로딩 완료)
        current_time = self.sim_t[batch_idxs] + self.UT
        self.item_end_time[batch_idxs, item_idx] = current_time
        self.item_completed[batch_idxs, item_idx] = True                

    def _clear_completed_orders_at_cp(self, batch_idxs, cp_idx):
        """
        CP에서 완성된 주문들의 아이템을 실제로 제거 (벡터화 버전)
        아이템이 실제 도착할 때 호출됨
        완성 조건: 예약된 SKU가 없고, 도착한 SKU가 주문과 정확히 일치
        
        Args:
            batch_idxs: (B,) 배치 인덱스들
            cp_idx: (B,) CP 인덱스들
        """
        # 각 (batch, cp)에 대해 confirmed order가 있는지 확인
        confirmed_orders = self.cp_confirmed_order_idx[batch_idxs, cp_idx]  # (B,)
        has_confirmed = (confirmed_orders >= 0)  # (B,) bool mask
        
        if not has_confirmed.any():
            return  # 확정된 주문이 없으면 종료
        
        # 확정된 주문이 있는 배치들만 필터링
        valid_batch_idxs = batch_idxs[has_confirmed]
        valid_cp_idx = cp_idx[has_confirmed]
        valid_confirmed_orders = confirmed_orders[has_confirmed]
        
        B_valid = len(valid_batch_idxs)
        
        # 완성 조건 체크 (벡터화)
        can_fulfill_mask = torch.ones(B_valid, dtype=torch.bool)
        
        # 1. 예약된 SKU가 없는지 확인 (order_sku_reserved 기반으로 계산)
        reserved_qty = self.order_sku_reserved[valid_batch_idxs, valid_confirmed_orders, :]  # (B_valid, N_S)
        has_reserved = (reserved_qty.sum(dim=1) > 0)  # (B_valid,)
        can_fulfill_mask &= ~has_reserved  # 예약된 아이템이 있으면 fulfill 불가
        
        # 2. 도착한 SKU가 주문과 정확히 일치하는지 확인 (order_sku_arrived 기반으로 계산)
        required_qty = self.order_sku_requirements[valid_batch_idxs, valid_confirmed_orders, :]  # (B_valid, N_S)
        arrived_qty = self.order_sku_arrived[valid_batch_idxs, valid_confirmed_orders, :]  # (B_valid, N_S)
        
        # 요구량과 도착량이 정확히 일치하는지 확인 (모든 SKU가 일치해야 함)
        exact_match = (required_qty == arrived_qty)  # (B_valid, N_S)
        all_sku_match = exact_match.all(dim=1)  # (B_valid,)
        can_fulfill_mask &= all_sku_match
        
        # 3. 주문에 없는 다른 SKU가 있으면 안됨 (이미 위에서 체크됨)
        # order_sku_requirements가 0인 SKU는 actual_qty도 0이어야 함
        
        # 완성 조건을 만족하는 배치들만 필터링
        fulfillable_mask = can_fulfill_mask
        if not fulfillable_mask.any():
            return  # 완성 가능한 주문이 없으면 종료
        
        final_batch_idxs = valid_batch_idxs[fulfillable_mask]
        final_cp_idx = valid_cp_idx[fulfillable_mask]
        final_confirmed_orders = valid_confirmed_orders[fulfillable_mask]
        
        # 주문 완료 처리 (벡터화)
        current_time = self.sim_t[final_batch_idxs] + self.UT
        
        # 주문 cleared 표시
        self.order_cleared[final_batch_idxs, final_confirmed_orders] = True
        self.order_cleared_time[final_batch_idxs, final_confirmed_orders] = current_time
        
        # CP 상태 초기화 (벡터화) - 실제 아이템은 item 상태로 관리됨
        self.cp_confirmed_order_idx[final_batch_idxs, final_cp_idx] = -1
        
    def reserve_item_and_order(self, batch_idxs, cp_idxs, item_idxs, sku_types, order_idxs):
        
        # 과다 예약 체크: 예약 전에 이미 충분한지 확인
        for i in range(len(batch_idxs)):
            b = batch_idxs[i].item()
            order = order_idxs[i].item()
            sku = sku_types[i].item()
            req = self.order_sku_requirements[b, order, sku].item()
            rsv = self.order_sku_reserved[b, order, sku].item()
            arr = self.order_sku_arrived[b, order, sku].item()
            
        
        # SKU 예약
        self.sku_remaining_count[batch_idxs, sku_types] -= 1
     
        # CP에 예약 아이템 추가
        self.order_sku_reserved[batch_idxs, order_idxs, sku_types] += 1

        # CP의 현재 order 저장
        self.cp_confirmed_order_idx[batch_idxs, cp_idxs] = order_idxs
        self.order_confirmed[batch_idxs, order_idxs] = True
        self.order_confirmed_cp[batch_idxs, order_idxs] = cp_idxs
        
        # 아이템이 어떤 order에 할당되었는지 추적
        self.item_assigned_order[batch_idxs, item_idxs] = order_idxs
        
        # 과다 예약 체크: 예약 후 검증 (이중 체크)
        for i in range(len(batch_idxs)):
            b = batch_idxs[i].item()
            order = order_idxs[i].item()
            sku = sku_types[i].item()
            req = self.order_sku_requirements[b, order, sku].item()
            rsv = self.order_sku_reserved[b, order, sku].item()
            arr = self.order_sku_arrived[b, order, sku].item()
            

    def order_is_sku_needed(self, batch_idxs, order_idx, sku_type) -> bool:
        """특정 배치의 특정 주문이 해당 SKU를 아직 필요로 하는지 확인 (True/False)"""
        # order_sku_requirements: (batch_size, N_O, N_S)
        # order_sku_arrived: (batch_size, N_O, N_S)
        required = self.order_sku_requirements[batch_idxs, order_idx, sku_type]
        arrived = self.order_sku_arrived[batch_idxs, order_idx, sku_type]
        reserved = self.order_sku_reserved[batch_idxs, order_idx, sku_type]
        
        
        # 텐서 연산을 위해 & 연산자 사용, 그리고 .item()으로 스칼라 변환
        result = (required > 0) & (arrived + reserved < required)
        # 텐서일 경우 .item()으로 변환, 아니면 그대로 반환
        return result.item() if torch.is_tensor(result) else result

    def _check_and_complete_orders_at_cp(self, batch_idxs, cp_idxs, item_idxs, sku_types, order_idxs):
        """
        특정 CP에서 confirmed된 주문이 완성 가능한지 체크하고 완성 처리
        batch_idxs는 step_ls_to_cp에서 처리된 batch idx만 input됨
        """
        
        # 각 주문이 완성 가능한지 체크
        # 주문 요구사항 (B, N_S)
        required_qty = self.order_sku_requirements[batch_idxs, order_idxs, :]
        
        # 주문별 예약+도착 수량 (B, N_S)
        reserved_qty = self.order_sku_reserved[batch_idxs, order_idxs, :]
        arrived_qty = self.order_sku_arrived[batch_idxs, order_idxs, :]
        total_qty = reserved_qty + arrived_qty
        
        # 요구량과 정확히 일치하는지 체크 (B, N_S)
        exact_match = (required_qty == total_qty)
        
        # 각 주문별로 모든 SKU가 일치하는지 체크 (B,)
        completable_mask = exact_match.all(dim=1)
        
        # 완성 가능한 주문들만 완성 처리
        if completable_mask.any():
            completable_batch_idxs = batch_idxs[completable_mask]
            completable_order_idxs = order_idxs[completable_mask]
            
            # 완성 처리
            # 완성 시간 기록
            self.order_completed[completable_batch_idxs, completable_order_idxs] = True  
            self.order_completion_time[completable_batch_idxs, completable_order_idxs] = self.sim_t[completable_batch_idxs]          

    def _update_available_actions(self, batch_idxs):
        """
        가능한 모든 액션을 self 변수에 저장
        self.decision_available: action_space에 따라 다른 조합의 가능한 액션 마스크
        """
        # 지정된 배치들의 모든 액션 마스크를 False로 초기화
        self.decision_available[batch_idxs] = False
        
        # 액션 체크 - decoder를 직접 순회
        for b in batch_idxs:
            decoder_size = self.action_index_decoder.shape[2]  # max_num_actions
            action_valid_mask = torch.zeros(decoder_size, dtype=torch.bool)
            
            if self.action_space == 'item-order-cp':
                # Action: (Item, SKU-Order, CP) - 모두 선택
                for action_idx in range(decoder_size):
                    item_node_idx = self.action_index_decoder[b, 0, action_idx].item()
                    sku_node_idx = self.action_index_decoder[b, 1, action_idx].item()
                    cp_node_idx = self.action_index_decoder[b, 2, action_idx].item()

                    if item_node_idx == -1:
                        continue
                    
                    item_idx = item_node_idx
                    cp_idx = cp_node_idx - self.N_I_max - self.N_SKU_max
                    sku_order_idx = sku_node_idx - self.N_I_max
                    
                    if not (0 <= sku_order_idx < self.N_SKU_max):
                        continue
                    
                    order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                    sku_type = self.sku_order_sku_type[b, sku_order_idx].item()
                    
                    if order_idx == -1 or sku_type == -1:
                        continue
                    
                    # Item이 해당 SKU 타입을 가지는지 확인
                    if self.item_sku_type[b, item_idx].item() != sku_type:
                        continue
                    
                    # LS 조건 체크
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    time_ok = self.ls_remaining_time[b, ls_idx] <= 0
                    robot_queue = self.ls_robot_queue[b, ls_idx]
                    robot_ok = (robot_queue >= 0).any().item()
                    
                    if not (time_ok and robot_ok):
                        continue
                    
                    front_item = self.ls_item_queue_lookahead[b, ls_idx, 0].item()
                    if front_item != item_idx:
                        continue
                    
                    # Order-SKU 필요 여부 확인
                    if not self.order_is_sku_needed(b, order_idx, sku_type):
                        continue
                    
                    # CP 가능성 체크
                    cp_available = True
                    assigned_cp = self.order_confirmed_cp[b, order_idx].item()
                    if assigned_cp >= 0 and assigned_cp != cp_idx:
                        cp_available = False
                    
                    cp_confirmed_order = self.cp_confirmed_order_idx[b, cp_idx].item()
                    if cp_confirmed_order >= 0 and cp_confirmed_order != order_idx:
                        cp_available = False
                    
                    if cp_available:
                        action_valid_mask[action_idx] = True
            
            elif self.action_space == 'item-cp':
                # Action: (Item, CP) - Order는 룰로 선택
                for action_idx in range(decoder_size):
                    item_node_idx = self.action_index_decoder[b, 0, action_idx].item()
                    cp_node_idx = self.action_index_decoder[b, 1, action_idx].item()

                    if item_node_idx == -1:
                        continue
                    
                    item_idx = item_node_idx
                    cp_idx = cp_node_idx - self.N_I_max - self.N_SKU_max
                    item_sku_type = self.item_sku_type[b, item_idx].item()
                    
                    # LS 조건 체크
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    time_ok = self.ls_remaining_time[b, ls_idx] <= 0
                    robot_queue = self.ls_robot_queue[b, ls_idx]
                    robot_ok = (robot_queue >= 0).any().item()
                    
                    if not (time_ok and robot_ok):
                        continue
                    
                    front_item = self.ls_item_queue_lookahead[b, ls_idx, 0].item()
                    if front_item != item_idx:
                        continue
                    
                    # CP 조건 체크
                    cp_confirmed_order = self.cp_confirmed_order_idx[b, cp_idx].item()
                    is_valid = False
                    
                    if cp_confirmed_order >= 0:
                        # CP에 이미 order가 할당된 경우
                        if self.order_is_sku_needed(b, cp_confirmed_order, item_sku_type):
                            is_valid = True
                    else:
                        # CP에 order가 할당되지 않은 경우
                        for order_idx in range(self.N_O):
                            if self.order_confirmed_cp[b, order_idx].item() == -1:
                                if self.order_is_sku_needed(b, order_idx, item_sku_type):
                                    is_valid = True
                                    break
                    
                    if is_valid:
                        action_valid_mask[action_idx] = True
            
            elif self.action_space == 'item-order':
                # Action: (Item, SKU-Order) - CP는 룰로 선택
                for action_idx in range(decoder_size):
                    item_node_idx = self.action_index_decoder[b, 0, action_idx].item()
                    sku_node_idx = self.action_index_decoder[b, 1, action_idx].item()

                    if item_node_idx == -1:
                        continue
                    
                    item_idx = item_node_idx
                    sku_order_idx = sku_node_idx - self.N_I_max
                    
                    if not (0 <= sku_order_idx < self.N_SKU_max):
                        continue
                    
                    order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                    sku_type = self.sku_order_sku_type[b, sku_order_idx].item()
                    
                    if order_idx == -1 or sku_type == -1:
                        continue
                    
                    # Item이 해당 SKU 타입을 가지는지 확인
                    if self.item_sku_type[b, item_idx].item() != sku_type:
                        continue
                    
                    # LS 조건 체크
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    time_ok = self.ls_remaining_time[b, ls_idx] <= 0
                    robot_queue = self.ls_robot_queue[b, ls_idx]
                    robot_ok = (robot_queue >= 0).any().item()
                    
                    if not (time_ok and robot_ok):
                        continue
                    
                    front_item = self.ls_item_queue_lookahead[b, ls_idx, 0].item()
                    if front_item != item_idx:
                        continue
                    
                    # Order-SKU 필요 여부 확인
                    if not self.order_is_sku_needed(b, order_idx, sku_type):
                        continue
                    
                    # CP를 룰로 선택 가능한지 확인 (최소 1개라도 사용 가능한 CP가 있는지)
                    order_confirmed_cp = self.order_confirmed_cp[b, order_idx].item()
                    if order_confirmed_cp >= 0:
                        # 이미 할당된 CP가 있으면 OK
                        action_valid_mask[action_idx] = True
                    else:
                        # 빈 CP가 있는지 확인
                        has_available_cp = False
                        for cp_idx in range(self.N_C):
                            if self.cp_confirmed_order_idx[b, cp_idx].item() == -1:
                                has_available_cp = True
                                break
                        if has_available_cp:
                            action_valid_mask[action_idx] = True
            
            elif self.action_space == 'order-cp':
                # Action: (SKU-Order, CP) - Item 먼저 1개 특정 후 가능한 (Order, CP) 조합 찾기
                
                # 1단계: 픽업 가능한 Item을 1개만 찾기 (LS index가 작은 것 우선)
                selected_item = None
                selected_item_sku_type = None
                candidate_items = []  # (item_idx, item_sku_type, ls_idx)
                
                for item_idx in range(self.N_I[b].item()):
                    # 아이템이 visible하고 아직 완료되지 않았는지 확인
                    if not self.item_visible[b, item_idx].item():
                        continue
                    if self.item_completed[b, item_idx].item():
                        continue
                    
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    
                    # LS 조건 체크: 처리 가능하고 로봇이 있어야 함
                    time_ok = self.ls_remaining_time[b, ls_idx] <= 0
                    robot_queue = self.ls_robot_queue[b, ls_idx]
                    robot_ok = (robot_queue >= 0).any().item()
                    
                    if not (time_ok and robot_ok):
                        continue
                    
                    # LS의 맨 앞 아이템인지 확인
                    front_item = self.ls_item_queue_lookahead[b, ls_idx, 0].item()
                    if front_item != item_idx:
                        continue
                    
                    # 이 item의 SKU 타입
                    item_sku_type = self.item_sku_type[b, item_idx].item()
                    
                    candidate_items.append((item_idx, item_sku_type, ls_idx))
                
                # Item 선택 룰에 따라 Tie-breaking
                if len(candidate_items) > 0:
                    if self.item_selection_rule == 'ls_index':
                        # LS index 기준 정렬 후 첫 번째 선택
                        candidate_items.sort(key=lambda x: x[2])  # x[2] = ls_idx
                    elif self.item_selection_rule == 'item_index':
                        # Item index 기준 정렬 후 첫 번째 선택
                        candidate_items.sort(key=lambda x: x[0])  # x[0] = item_idx
                    elif self.item_selection_rule == 'nearest':
                        # 가장 가까운 LS의 item 선택 (모든 CP까지의 평균 거리가 가장 짧은 LS)
                        item_scores = []
                        for item_idx, item_sku_type, ls_idx in candidate_items:
                            avg_dist = self.travel_time_ls_to_cp[b, ls_idx, :].mean().item()
                            item_scores.append((item_idx, item_sku_type, ls_idx, avg_dist))
                        item_scores.sort(key=lambda x: x[3])  # x[3] = avg_dist
                        candidate_items = [(x[0], x[1], x[2]) for x in item_scores]
                    elif self.item_selection_rule == 'earliest':
                        # 가장 먼저 픽업 가능한 item (LS 대기 시간 고려)
                        item_scores = []
                        for item_idx, item_sku_type, ls_idx in candidate_items:
                            ls_wait_time = self.ls_remaining_time[b, ls_idx].item()
                            item_scores.append((item_idx, item_sku_type, ls_idx, ls_wait_time))
                        item_scores.sort(key=lambda x: x[3])  # x[3] = ls_wait_time
                        candidate_items = [(x[0], x[1], x[2]) for x in item_scores]
                    elif self.item_selection_rule == 'smallest_valid_action':
                        # 가능한 (order, cp) 조합이 적은 아이템 우선
                        item_scores = []
                        for item_idx, item_sku_type, ls_idx in candidate_items:
                            # 이 아이템으로 가능한 (order, cp) 조합 개수 계산
                            valid_action_count = 0
                            for action_idx in range(decoder_size):
                                sku_node_idx = self.action_index_decoder[b, 0, action_idx].item()
                                cp_node_idx = self.action_index_decoder[b, 1, action_idx].item()
                                
                                if sku_node_idx == -1:
                                    continue
                                
                                sku_order_idx = sku_node_idx - self.N_I_max
                                if not (0 <= sku_order_idx < self.N_SKU_max):
                                    continue
                                
                                order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                                sku_type = self.sku_order_sku_type[b, sku_order_idx].item()
                                cp_idx = cp_node_idx - self.N_I_max - self.N_SKU_max
                                
                                if order_idx == -1 or sku_type == -1:
                                    continue
                                
                                # Item의 SKU 타입이 Order가 필요로 하는 타입과 일치하는지 확인
                                if item_sku_type != sku_type:
                                    continue
                                
                                # Order-SKU 필요 여부 확인
                                if not self.order_is_sku_needed(b, order_idx, sku_type):
                                    continue
                                
                                # CP 가능성 체크
                                order_confirmed_cp = self.order_confirmed_cp[b, order_idx].item()
                                if order_confirmed_cp >= 0 and order_confirmed_cp != cp_idx:
                                    continue
                                
                                cp_confirmed_order = self.cp_confirmed_order_idx[b, cp_idx].item()
                                if cp_confirmed_order >= 0 and cp_confirmed_order != order_idx:
                                    continue
                                
                                valid_action_count += 1
                            
                            item_scores.append((item_idx, item_sku_type, ls_idx, valid_action_count))
                        item_scores.sort(key=lambda x: x[3])  # x[3] = valid_action_count
                        candidate_items = [(x[0], x[1], x[2]) for x in item_scores]
                    else:
                        # 기본값: LS index 기준
                        candidate_items.sort(key=lambda x: x[2])
                    
                    selected_item = candidate_items[0][0]
                    selected_item_sku_type = candidate_items[0][1]
                    
                    # 선택된 item 저장 (나중에 _step_ls_to_cp에서 재사용)
                    self.order_cp_selected_item[b] = selected_item
                else:
                    # 가능한 item이 없으면 -1로 초기화
                    self.order_cp_selected_item[b] = -1
                
                # 2단계: 선택된 Item이 있으면, 그 Item으로 가능한 (Order, CP) 조합 찾기
                if selected_item is not None:
                    # 이 Item으로 가능한 모든 action 확인
                    for action_idx in range(decoder_size):
                        sku_node_idx = self.action_index_decoder[b, 0, action_idx].item()
                        cp_node_idx = self.action_index_decoder[b, 1, action_idx].item()

                        if sku_node_idx == -1:
                            continue
                        
                        sku_order_idx = sku_node_idx - self.N_I_max
                        
                        if not (0 <= sku_order_idx < self.N_SKU_max):
                            continue
                        
                        order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                        sku_type = self.sku_order_sku_type[b, sku_order_idx].item()
                        cp_idx = cp_node_idx - self.N_I_max - self.N_SKU_max
                        
                        if order_idx == -1 or sku_type == -1:
                            continue
                        
                        # Item의 SKU 타입이 Order가 필요로 하는 타입과 일치하는지 확인
                        if selected_item_sku_type != sku_type:
                            continue
                        
                        # Order-SKU 필요 여부 확인
                        if not self.order_is_sku_needed(b, order_idx, sku_type):
                            continue
                        
                        # CP 가능성 체크: Order와 CP가 호환되는지 확인
                        order_confirmed_cp = self.order_confirmed_cp[b, order_idx].item()
                        if order_confirmed_cp >= 0 and order_confirmed_cp != cp_idx:
                            # 이미 다른 CP에 할당된 order
                            continue
                        
                        cp_confirmed_order = self.cp_confirmed_order_idx[b, cp_idx].item()
                        if cp_confirmed_order >= 0 and cp_confirmed_order != order_idx:
                            # 이미 다른 order가 할당된 CP
                            continue
                        
                        # 모든 조건 만족 → 이 (Order, CP) 조합은 valid!
                        action_valid_mask[action_idx] = True
                # else: 가능한 item이 없으면 가능한 action도 없음 (action_valid_mask는 모두 False)
            
            # 계산된 마스크를 전역 마스크에 할당
            self.decision_available[b] = action_valid_mask

    def _advance_to_next_decision_event_for_batches(self, batch_idxs):
        """지정된 배치들에서 시간을 다음 이벤트까지 진행"""
        
        # 다음 이벤트까지의 시간 계산 (배치별)
        time_delta = self.get_next_move_t(batch_idxs)  # shape: (len(batch_idxs),)
        
        # 시뮬레이션 시간 진행
        self.sim_t[batch_idxs] += time_delta

        # 로봇 대기 시간 감소
        self.robot_now_remain_t[batch_idxs] = torch.clamp(
            self.robot_now_remain_t[batch_idxs] - time_delta.unsqueeze(1), min=0)

        # 로봇 상태 업데이트
        self._update_robot_state(batch_idxs)

        # CP available_time은 절대 시간이므로 감소 불필요
        
        # LS 처리 시간 감소
        self.ls_remaining_time[batch_idxs] = torch.clamp(
            self.ls_remaining_time[batch_idxs] - time_delta.unsqueeze(1), min=0)        

    def _update_robot_state(self, batch_idxs):
        """지정된 배치들에서 로봇 상태 업데이트"""
        # 이동 중인 로봇 마스크 (state 1 or 2)
        moving_mask = (self.robot_state[batch_idxs] == 1) | (self.robot_state[batch_idxs] == 2)
        
        # 이동이 완료된 로봇 마스크
        completed_mask = (self.robot_now_remain_t[batch_idxs] <= 0) & moving_mask
        
        # MOVE_CP (state=1)에서 완료된 로봇들을 CP_WAIT (state=4)로 변경
        cp_completed_mask = completed_mask & (self.robot_state[batch_idxs] == 1)
        
        # MOVE_LS (state=2)에서 완료된 로봇들을 READY_TO_PICK (state=0)로 변경
        ls_completed_mask = completed_mask & (self.robot_state[batch_idxs] == 2)
        
        # LS에 도착한 로봇을 해당 LS의 로봇 큐에 추가
        if ls_completed_mask.any():
            # LS에 도착한 로봇들의 배치와 로봇 인덱스 찾기
            batch_robot_pairs = torch.nonzero(ls_completed_mask, as_tuple=False)
            
            if len(batch_robot_pairs) > 0:
                # 배치 인덱스들과 로봇 인덱스들 분리
                rel_batch_idxs = batch_robot_pairs[:, 0]
                robot_idxs = batch_robot_pairs[:, 1]
                
                # 실제 배치 인덱스들 계산
                actual_batch_idxs = batch_idxs[rel_batch_idxs]
                
                # 로봇 상태를 0으로 변경
                self.robot_state[actual_batch_idxs, robot_idxs] = 0
                
                # 각 로봇의 현재 LS 인덱스 가져오기
                ls_idxs = self.robot_current_ls[actual_batch_idxs, robot_idxs]
                
                # 유효한 LS 인덱스만 필터링 (ls_idx >= 0)
                valid_mask = ls_idxs >= 0
                
                if valid_mask.any():
                    valid_actual_batch_idxs = actual_batch_idxs[valid_mask]
                    valid_robot_idxs = robot_idxs[valid_mask]
                    valid_ls_idxs = ls_idxs[valid_mask]
                    
                    # 각 유효한 로봇에 대해 빈 슬롯 찾기 및 할당
                    for i in range(len(valid_actual_batch_idxs)):
                        b_idx = valid_actual_batch_idxs[i].item()
                        robot_idx = valid_robot_idxs[i].item()
                        ls_idx = valid_ls_idxs[i].item()
                        
                        # 빈 자리 찾아서 로봇 추가
                        empty_slots = (self.ls_robot_queue[b_idx, ls_idx] == -1)
                        if empty_slots.any():
                            first_empty = torch.argmax(empty_slots.int())
                            self.ls_robot_queue[b_idx, ls_idx, first_empty] = robot_idx
                            
                            # 디버그: 로봇 큐에 추가
                            if self.debug_env and i < 5:  # 처음 5개만 출력
                                sim_t = self.sim_t[b_idx].item()
                                queue_robots = [r.item() for r in self.ls_robot_queue[b_idx, ls_idx] if r >= 0]
                                print(f"    🟢 [Robot Return] Robot{robot_idx} → LS{ls_idx}, t={sim_t:.2f}, Queue: {queue_robots}")
                    
                    # 로봇 추가 후, 각 LS에서 FCFS로 로봇과 아이템 할당
                    # 각 LS별로 unique하게 처리
                    unique_batch_ls = {}
                    for i in range(len(valid_actual_batch_idxs)):
                        b_idx = valid_actual_batch_idxs[i].item()
                        ls_idx = valid_ls_idxs[i].item()
                        key = (b_idx, ls_idx)
                        if key not in unique_batch_ls:
                            unique_batch_ls[key] = True
                            
                            # 이 LS의 로봇 큐에서 할당되지 않은 로봇들 확인
                            robot_queue = self.ls_robot_queue[b_idx, ls_idx]
                            available_robots = [r.item() for r in robot_queue if r >= 0]
                            
                            if self.debug_env and len(available_robots) > 0:
                                print(f"      🔄 [FCFS Matching] Batch {b_idx}, LS{ls_idx}, Available robots: {available_robots}")

        # CP에 도착한 로봇 처리
        if cp_completed_mask.any():
            # CP 완료된 로봇들의 배치와 로봇 인덱스 찾기
            batch_robot_pairs = torch.nonzero(cp_completed_mask, as_tuple=False)
            
            if len(batch_robot_pairs) > 0:
                # 배치 인덱스들과 로봇 인덱스들 분리
                rel_batch_idxs = batch_robot_pairs[:, 0]
                robot_idxs = batch_robot_pairs[:, 1]
                
                # 실제 배치 인덱스들 계산

                actual_batch_idxs = batch_idxs[rel_batch_idxs]
                
                # 로봇 상태를 4로 변경
                self.robot_state[actual_batch_idxs, robot_idxs] = 4
                
                # 각 로봇의 현재 CP 인덱스 가져오기
                cp_idxs = self.robot_current_cp[actual_batch_idxs, robot_idxs]

                valid_mask = cp_idxs >= 0

                if valid_mask.any():
                    valid_actual_batch_idxs = actual_batch_idxs[valid_mask]
                    valid_robot_idxs = robot_idxs[valid_mask]
                    valid_cp_idxs = cp_idxs[valid_mask]

                    # CP에서 언로드 시작 시간 계산: max(로봇 도착 시간, CP 가용 시간)
                    robot_arrival_time = self.sim_t[valid_actual_batch_idxs]  # 로봇이 CP에 도착한 시간
                    unloading_start_time = torch.zeros_like(robot_arrival_time)
                    unloading_end_time = torch.zeros_like(robot_arrival_time)
                    
                    # 각 로봇을 순차적으로 처리 (같은 CP에 동시 도착 시 순서대로 처리)
                    for i in range(len(valid_actual_batch_idxs)):
                        b_idx = valid_actual_batch_idxs[i]
                        cp_idx = valid_cp_idxs[i]
                        arrival_t = robot_arrival_time[i].item()
                        
                        # CP가 사용 가능한 시간 (절대 시간)
                        cp_free_t = self.cp_available_time[b_idx, cp_idx].item()
                        
                        # 언로드 시작 시간 = max(로봇 도착 시간, CP 가용 시간)
                        start_t = max(arrival_t, cp_free_t)
                        end_t = start_t + self.UT
                        
                        # 결과 저장
                        unloading_start_time[i] = start_t
                        unloading_end_time[i] = end_t
                        
                        # 즉시 CP 가용 시간 업데이트 (다음 로봇이 업데이트된 값을 보도록)
                        self.cp_available_time[b_idx, cp_idx] = end_t
                    
                    # 예약에서 실제 도착으로 이전 - 로봇이 들고 있던 아이템 사용

                    carrying_item_idxs = self.robot_carrying_item[valid_actual_batch_idxs, valid_robot_idxs]

                    carrying_sku_types = self.item_sku_type[valid_actual_batch_idxs, carrying_item_idxs]
                    
                    # 디버그: arrive_item 호출 전후 상태 출력
                    if self.debug_env:
                        for i in range(len(valid_actual_batch_idxs)):
                            b = valid_actual_batch_idxs[i].item()
                            r = valid_robot_idxs[i].item()
                            cp = valid_cp_idxs[i].item()
                            item = carrying_item_idxs[i].item()
                            sku = carrying_sku_types[i].item()
                            sku_letter = chr(ord('A') + sku)
                            order = self.cp_confirmed_order_idx[b, cp].item()
                    
                    self.arrive_item(valid_actual_batch_idxs, valid_cp_idxs, carrying_item_idxs, carrying_sku_types)
                    
                    # 아이템의 CP 언로드 시작/완료 시간 기록 (대기 시간 반영)
                    self.item_cp_start_time[valid_actual_batch_idxs, carrying_item_idxs] = unloading_start_time
                    self.item_end_time[valid_actual_batch_idxs, carrying_item_idxs] = unloading_end_time
                    
                    # 로봇이 언로드 완료까지 기다려야 하는 시간 설정
                    self.robot_now_remain_t[valid_actual_batch_idxs, valid_robot_idxs] = unloading_end_time - robot_arrival_time
                    
                    # 디버그: 언로드 시간 정보 출력 (같은 CP 그룹화)
                    if self.debug_env:
                        # 같은 CP에 도착한 로봇들을 그룹화
                        cp_groups = {}
                        for i in range(len(valid_actual_batch_idxs)):
                            b = valid_actual_batch_idxs[i].item()
                            cp = valid_cp_idxs[i].item()
                            key = (b, cp)
                            if key not in cp_groups:
                                cp_groups[key] = []
                            cp_groups[key].append(i)
                        
                        # 각 CP 그룹별로 출력
                        printed = 0
                        for (b, cp), indices in cp_groups.items():
                            if printed >= 3:  # 최대 3개 CP 그룹만 출력
                                break
                            
                            if len(indices) > 1:
                                print(f"    🔄 [CP{cp} Multiple Arrivals] {len(indices)} robots at t={robot_arrival_time[indices[0]].item():.2f}")
                            
                            for i in indices[:5]:  # 각 CP 그룹에서 최대 5개
                                r = valid_robot_idxs[i].item()
                                item = carrying_item_idxs[i].item()
                                arrival_t = robot_arrival_time[i].item()
                                start_t = unloading_start_time[i].item()
                                end_t = unloading_end_time[i].item()
                                wait_time = start_t - arrival_t
                                actual_ut = end_t - start_t
                                
                                if wait_time > 0.01:
                                    print(f"      ⏳ Robot{r}, Item{item}: wait={wait_time:.2f}, unload={start_t:.2f}~{end_t:.2f}")
                                else:
                                    print(f"      📦 Robot{r}, Item{item}: unload={start_t:.2f}~{end_t:.2f}")
                            
                            printed += 1
                    
                    # 완료된 주문 정리
                    self._clear_completed_orders_at_cp(valid_actual_batch_idxs, valid_cp_idxs)
        
        # CP_WAIT 상태에서 언로딩이 완료된 로봇들을 아이템에 할당하여 LS로 보냄 (Rule-based backward trip)
        cp_wait_mask = self.robot_state[batch_idxs] == 4  # CP_WAIT 상태
        unloading_done_mask = self.robot_now_remain_t[batch_idxs] <= 0  # 언로딩 완료
        
        # 언로딩이 완료된 CP_WAIT 로봇들
        ready_to_return_mask = cp_wait_mask & unloading_done_mask
        
        if ready_to_return_mask.any():
            batch_robot_pairs = torch.nonzero(ready_to_return_mask, as_tuple=False)
            
            if len(batch_robot_pairs) > 0:
                rel_batch_idxs = batch_robot_pairs[:, 0]
                robot_idxs = batch_robot_pairs[:, 1]
                actual_batch_idxs = batch_idxs[rel_batch_idxs]
                
                # 각 로봇마다 픽업할 아이템을 직접 할당 (룰 기반)
                target_ls_list = []
                target_item_list = []
                
                for i in range(len(actual_batch_idxs)):
                    b_idx = actual_batch_idxs[i].item()
                    r_idx = robot_idxs[i].item()
                    current_cp = self.robot_current_cp[b_idx, r_idx].item()
                    
                    # 1. LS별로 할당 가능한 아이템들 찾기 + 로봇 필요 여부 계산
                    ls_unassigned_items = {}  # {ls_idx: [(item_idx, position), ...]}
                    ls_robot_needs = {}  # {ls_idx: 필요한 로봇 수}
                    
                    for item_idx in range(self.N_I_max):
                        # 아이템이 완료되지 않았고, 가시 범위에 있는 아이템
                        if (not self.item_completed[b_idx, item_idx].item() and
                            self.item_visible[b_idx, item_idx].item()):
                            ls_idx = self.item_loading_station[b_idx, item_idx].item()
                            position = self.item_current_position[b_idx, item_idx].item()
                            
                            if ls_idx not in ls_unassigned_items:
                                ls_unassigned_items[ls_idx] = []
                            ls_unassigned_items[ls_idx].append((item_idx, position))
                    
                    # 각 LS별 필요한 로봇 수 계산
                    for ls_idx, items in ls_unassigned_items.items():
                        total_items = len(items)
                        
                        # LS에 현재 있는 로봇 수
                        robots_at_ls = (self.ls_robot_queue[b_idx, ls_idx] >= 0).sum().item()
                        
                        # LS로 이동 중인 로봇 수 (state=2 and current_ls=ls_idx)
                        moving_to_ls = 0
                        for r in range(self.N_R):
                            if (self.robot_state[b_idx, r].item() == 2 and 
                                self.robot_current_ls[b_idx, r].item() == ls_idx):
                                moving_to_ls += 1
                        
                        # 필요한 로봇 수 = 총 아이템 수 - (현재 로봇 + 이동 중인 로봇)
                        needed_robots = total_items - (robots_at_ls + moving_to_ls)
                        ls_robot_needs[ls_idx] = needed_robots
                    
                    
                    ls_unassigned_items = {ls: items for ls, items in ls_unassigned_items.items() 
                                           if ls_robot_needs[ls] > 0}
                    
                    # 2. Backward trip rule에 따라 LS 선택
                    if len(ls_unassigned_items) > 0:
                        if self.backward_trip_rule == 'balance':
                            # Balance rule: 아이템이 가장 많이 남은 LS 우선, tie는 nearest
                            ls_scores = []
                            for ls_idx in ls_unassigned_items.keys():
                                num_items = len(ls_unassigned_items[ls_idx])
                                distance = self.travel_time_cp_to_ls[b_idx, current_cp, ls_idx].item()
                                # Score: (-num_items, distance) → 아이템 많은 순, 같으면 가까운 순
                                ls_scores.append((ls_idx, -num_items, distance))
                            
                            # 정렬: 아이템 많은 순 (음수이므로 오름차순), tie는 거리 짧은 순
                            ls_scores.sort(key=lambda x: (x[1], x[2]))
                            selected_ls = ls_scores[0][0]
                        else:
                            # Nearest rule (기본): 가장 가까운 LS 선택
                            ls_distances = []
                            for ls_idx in ls_unassigned_items.keys():
                                distance = self.travel_time_cp_to_ls[b_idx, current_cp, ls_idx].item()
                                ls_distances.append((ls_idx, distance))
                            
                            # 거리순으로 정렬하여 가장 가까운 LS 선택
                            ls_distances.sort(key=lambda x: x[1])
                            selected_ls = ls_distances[0][0]
                        
                        # 3. 선택된 LS 내에서 position이 가장 앞인 아이템 선택
                        ls_items = ls_unassigned_items[selected_ls]
                        ls_items.sort(key=lambda x: x[1])  # position 순 정렬
                        selected_item = ls_items[0][0]  # position이 가장 작은 아이템
                        
                        # 4. 로봇-아이템 직접 할당 제거: LS에 도착하면 자동으로 사용 가능
                        # self.item_assigned_robot[b_idx, selected_item] = r_idx
                        
                        target_ls_list.append(selected_ls)
                        target_item_list.append(selected_item)
                    else:
                        # 할당 가능한 아이템이 없으면 초기 위치 LS로 이동
                        initial_ls = self.robot_initial_position[b_idx, r_idx].item()
                        target_ls_list.append(initial_ls)
                        target_item_list.append(-1)  # 할당된 아이템 없음
                
                # 텐서로 변환
                target_ls_tensor = torch.tensor(target_ls_list, dtype=torch.long)
                target_item_tensor = torch.tensor(target_item_list, dtype=torch.long)
                
                # 이동 시간 계산
                current_cp = self.robot_current_cp[actual_batch_idxs, robot_idxs]
                travel_time = self.travel_time_cp_to_ls[actual_batch_idxs, current_cp, target_ls_tensor]
                
                # 로봇 상태 업데이트: LS로 이동
                self.robot_state[actual_batch_idxs, robot_idxs] = 2  # MOVE_LS
                self.robot_now_remain_t[actual_batch_idxs, robot_idxs] = travel_time
                self.robot_current_ls[actual_batch_idxs, robot_idxs] = target_ls_tensor
                self.robot_current_cp[actual_batch_idxs, robot_idxs] = -1
                
                
                self.robot_carrying_item[actual_batch_idxs, robot_idxs] = -1
                self.robot_target_item[actual_batch_idxs, robot_idxs] = target_item_tensor


    def _is_simulation_done(self, batch_idxs):
        """지정된 배치들에서 시뮬레이션 완료 여부 확인 (torch tensor 반환)"""
        # 각 배치에서 모든 주문이 처리 완료되었는지 확인
        done_mask = torch.zeros(len(batch_idxs), dtype=torch.bool)
        
        for i, b in enumerate(batch_idxs):
            b_idx = b.item()
            # 시뮬레이터 로직: 모든 주문이 cleared되면 완료
            all_orders_cleared = self.order_cleared[b_idx].all().item()
            done_mask[i] = all_orders_cleared
            
        return done_mask

    def _get_state(self): #노드 feature 업데이트 
        # 새로운 Item-SKU-CP 피처 업데이트
        self.item_feature = self.item_feature_init.clone()
        self.sku_feature = self.sku_feature_init.clone()
        self.cp_feature = self.cp_feature_init.clone()


        # Item 노드 피처 업데이트 (인덱스 0~2 사용) - 벡터 연산 + 정규화
        # 0: item_current_position (정규화: 0 ~ max_item_position)
        if self.max_item_position > 0:
            self.item_feature[:, :, 0] = self.item_current_position.float() / self.max_item_position
        else:
            self.item_feature[:, :, 0] = self.item_current_position.float()
        
        # 1: robot_arrival_upper_bound (upper bound 기반 로봇 도착 남은 시간, 정규화)
        # 먼저 모든 값을 max(1.0)로 초기화 (도착 예정 없음)
        self.item_feature[:, :, 1] = 1.0
        
        # 각 배치별로 처리
        for b in range(self.batch_size):
            for item_idx in range(self.N_I[b]):
                # 1) 이미 로봇이 할당된 경우: 해당 로봇의 남은 시간
                if self.item_assigned_robot[b, item_idx] >= 0:
                    robot_idx = self.item_assigned_robot[b, item_idx].item()
                    robot_time = self.robot_now_remain_t[b, robot_idx].item()
                    # 정규화: 0 ~ max_robot_time
                    if self.max_robot_time > 0:
                        self.item_feature[b, item_idx, 1] = robot_time / self.max_robot_time
                    else:
                        self.item_feature[b, item_idx, 1] = robot_time
                
                # 2) 로봇이 할당되지 않은 경우: upper bound 계산
                else:
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    item_position = self.item_current_position[b, item_idx].item()
                    
                    # LS의 로봇 큐에 있는 로봇 수 계산
                    robot_queue = self.ls_robot_queue[b, ls_idx]
                    num_robots_in_queue = (robot_queue >= 0).sum().item()
                    
                    # 2a) 아이템이 로봇 큐 범위 내에 있는 경우 → 도착 시간 0
                    if item_position < num_robots_in_queue:
                        self.item_feature[b, item_idx, 1] = 0.0
                    
                    # 2b) 아이템이 로봇 큐 범위를 벗어난 경우 → CP에서 돌아오는 로봇 중 가장 빨리 도착하는 시간
                    else:
                        min_arrival_time = float('inf')
                        
                        # 모든 로봇을 순회하며 해당 LS로 돌아오는 로봇 찾기
                        for robot_idx in range(self.N_R):
                            robot_state = self.robot_state[b, robot_idx].item()
                            robot_target_ls = self.robot_current_ls[b, robot_idx].item()
                            
                            # 상태가 1(MOVE_CP), 2(MOVE_LS), 4(CP_WAIT)이고 해당 LS로 향하는 로봇
                            if robot_state in [1, 2, 4] and robot_target_ls == ls_idx:
                                robot_time = self.robot_now_remain_t[b, robot_idx].item()
                                min_arrival_time = min(min_arrival_time, robot_time)
                        
                        # 돌아오는 로봇이 있는 경우
                        if min_arrival_time != float('inf'):
                            if self.max_robot_time > 0:
                                self.item_feature[b, item_idx, 1] = min_arrival_time / self.max_robot_time
                            else:
                                self.item_feature[b, item_idx, 1] = min_arrival_time
                        # 돌아오는 로봇이 없는 경우 → 이미 1.0으로 초기화됨
        
        # 2: ls_remaining_time - LS의 현재 작업 남은 시간 (정규화: 0 ~ LT)
        for b in range(self.batch_size):
            for item_idx in range(self.N_I[b]):
                ls_idx = self.item_loading_station[b, item_idx].item()
                ls_remaining = self.ls_remaining_time[b, ls_idx].item()
                # 정규화: 0 ~ max_ls_time (LT)
                if self.max_ls_time > 0:
                    self.item_feature[b, item_idx, 2] = ls_remaining / self.max_ls_time
                else:
                    self.item_feature[b, item_idx, 2] = ls_remaining

        # SKU 노드 피처 업데이트 (인덱스 3~8 사용) - sku_order 변수들 사용 + 정규화
        for b in range(self.batch_size):
            for sku_order_idx in range(self.N_SKU_max):
                # 유효한 SKU Order인지 확인
                if self.sku_order_order_idx[b, sku_order_idx] == -1:
                    break
                
                order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                sku_type = self.sku_order_sku_type[b, sku_order_idx].item()
                
                # 3: sku_fulfillment_rate: 이 SKU의 충족률 (0~1)
                # (arrived + reserved) / required
                required_qty = self.order_sku_requirements[b, order_idx, sku_type].item()
                arrived_qty = self.order_sku_arrived[b, order_idx, sku_type].item()
                reserved_qty = self.order_sku_reserved[b, order_idx, sku_type].item()
                
                if required_qty > 0:
                    sku_fulfillment_rate = (arrived_qty + reserved_qty) / required_qty
                else:
                    sku_fulfillment_rate = 1.0  # 요구사항이 없으면 완료로 간주
                
                self.sku_feature[b, sku_order_idx, 3] = sku_fulfillment_rate
                
                # 4: order_has_cp_assigned: 해당 Order가 CP에 할당되었는지 여부 (0 or 1)
                has_cp_assigned = (self.order_confirmed_cp[b, order_idx] >= 0).float()
                self.sku_feature[b, sku_order_idx, 4] = has_cp_assigned
                
                # 5: order_progress: 해당 Order의 전체 충족률 (0~1)
                # (arrived + reserved) / total_required
                total_required = self.order_sku_requirements[b, order_idx, :].sum().item()
                total_arrived = self.order_sku_arrived[b, order_idx, :].sum().item()
                total_reserved = self.order_sku_reserved[b, order_idx, :].sum().item()
                if total_required > 0:
                    order_progress = (total_arrived + total_reserved) / total_required
                else:
                    order_progress = 1.0  # 요구사항이 없으면 완료로 간주
                self.sku_feature[b, sku_order_idx, 5] = order_progress
                
                # 6: order_arrival_rate: 해당 Order의 도착률 (0~1)
                # arrived / total_required (실제로 CP에 도착한 비율)
                if total_required > 0:
                    order_arrival_rate = total_arrived / total_required
                else:
                    order_arrival_rate = 1.0  # 요구사항이 없으면 완료로 간주
                self.sku_feature[b, sku_order_idx, 6] = order_arrival_rate
                
                # 7: order_sku_remaining_items: 이 Order-SKU의 남은 item 개수 (정규화)
                # required - (arrived + reserved)
                remaining_sku = required_qty - (arrived_qty + reserved_qty)
                if self.max_order_sku_remaining > 0:
                    self.sku_feature[b, sku_order_idx, 7] = remaining_sku / self.max_order_sku_remaining
                else:
                    self.sku_feature[b, sku_order_idx, 7] = remaining_sku
                
                # 8: order_remaining_items: 이 Order의 전체 남은 item 개수 (정규화)
                # total_required - (total_arrived + total_reserved)
                order_remaining = total_required - (total_arrived + total_reserved)
                if self.max_order_remaining > 0:
                    self.sku_feature[b, sku_order_idx, 8] = order_remaining / self.max_order_remaining
                else:
                    self.sku_feature[b, sku_order_idx, 8] = order_remaining

        # CP 노드 피처 업데이트 (인덱스 9 사용)
        for b in range(self.batch_size):
            for cp_idx in range(self.N_C):
                # 9: has_assigned_order: 현재 order 할당 여부
                has_order = (self.cp_confirmed_order_idx[b, cp_idx] >= 0).float()
                self.cp_feature[b, cp_idx, 9] = has_order                
        
        # Edge 8 feature 동적 업데이트 (Order-CP 할당 상태)
        for b in range(self.batch_size):
            edge8_edges = self.edge8_sku_cp_init[b]  # (2, num_edges)
            num_edges = edge8_edges.shape[1]
            edge8_feat = torch.zeros((num_edges, 1), dtype=torch.float)
            
            for edge_idx in range(num_edges):
                src = edge8_edges[0, edge_idx].item()
                dst = edge8_edges[1, edge_idx].item()
                
                # SKU 노드인지 CP 노드인지 판별
                if src >= self.N_I_max and src < self.N_I_max + self.N_SKU_max:
                    # src가 SKU 노드
                    sku_node_idx = src
                    cp_node_idx = dst
                else:
                    # dst가 SKU 노드
                    sku_node_idx = dst
                    cp_node_idx = src
                
                sku_order_idx = sku_node_idx - self.N_I_max
                cp_idx = cp_node_idx - self.N_I_max - self.N_SKU_max
                
                # 해당 Order가 이 CP에 할당되었는지 확인
                if self.sku_order_order_idx[b, sku_order_idx] >= 0:
                    order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                    confirmed_cp = self.order_confirmed_cp[b, order_idx].item()
                    
                    if confirmed_cp == cp_idx:
                        edge8_feat[edge_idx, 0] = 1.0
            
            # 동적으로 생성한 feature 저장
            self.edge8_feature_init[b] = edge8_feat
        
        # 시퀀스 인코더를 사용하는 경우 item feature에 context vector 추가
        if self.sequence_encoder is not None:
            # item_features: (batch_size, N_I_max, feature_dim)
            # item_to_ls: (batch_size, N_I_max) - 각 item이 속한 LS 인덱스
            # ls_item_sequences: (batch_size, N_L, look_ahead) - 각 LS의 관측 가능한 item 시퀀스
            
            # 시퀀스 인코더의 device 가져오기
            encoder_device = next(self.sequence_encoder.parameters()).device
            
            # 데이터를 시퀀스 인코더의 device로 이동
            item_feature_device = self.item_feature.to(encoder_device)
            item_to_ls_device = self.item_loading_station.to(encoder_device)
            ls_item_sequences_device = self.ls_item_queue_lookahead.to(encoder_device)
            
            # context vector 생성
            with torch.no_grad():  # 환경에서는 gradient 계산 안 함
                ls_context_vectors = self.sequence_encoder(
                    item_features=item_feature_device,
                    item_to_ls=item_to_ls_device,
                    ls_item_sequences=ls_item_sequences_device,
                    num_ls=self.N_L
                )  # (batch_size, N_I_max, sequence_hidden_dim)
            
            # context vector를 원래 device로 이동
            ls_context_vectors = ls_context_vectors.to(self.item_feature.device)
            
            # item feature에 context vector concat
            self.item_feature = torch.cat([self.item_feature, ls_context_vectors], dim=2)
        
        # 시퀀스 인코더를 사용하지 않는 경우 SKU와 CP feature의 차원을 맞춰줌
        # (모든 노드가 동일한 feature 차원을 가져야 하므로)
        if self.sequence_encoder is not None:
            # item feature: (batch_size, N_I_max, feature_dim + sequence_hidden_dim)
            # sku/cp feature: (batch_size, N_SKU_max/N_C, feature_dim)
            # -> sku/cp feature에 zero padding 추가
            sequence_hidden_dim = self.item_feature.shape[2] - self.sku_feature.shape[2]
            sku_padding = torch.zeros(self.batch_size, self.N_SKU_max, sequence_hidden_dim, device=self.sku_feature.device)
            cp_padding = torch.zeros(self.batch_size, self.N_C, sequence_hidden_dim, device=self.cp_feature.device)
            self.sku_feature = torch.cat([self.sku_feature, sku_padding], dim=2)
            self.cp_feature = torch.cat([self.cp_feature, cp_padding], dim=2)
        
        # 노드 피처 결합 (Item + SKU + CP)
        self.node_feature = torch.cat([self.item_feature, self.sku_feature, self.cp_feature], dim=1)
        
        # build graph state 직전에 마스크 업데이트 (최신 상태 반영)
        # (주석 처리: move_next_state()에서 이미 업데이트하므로 중복)
        # self._update_available_actions(self.BATCH_IDX)
        
        # 액션 수 디버그 출력 (실제 agent가 액션을 선택하기 직전)
        if self.debug_action_num:
            for b in self.BATCH_IDX:
                # order_completed가 아닌 배치만 출력
                if not self.order_completed[b].all().item():
                    # batch와 pomo 인덱스 계산
                    instance_idx = b.item() // self.pomo
                    pomo_idx = b.item() % self.pomo
                    step = self.step_count[b].item()
                    
                    available_count = self.decision_available[b].sum().item()
                    total_count = self.max_num_actions
                    print(f"  📊 [Step {step}] Instance {instance_idx}, POMO {pomo_idx}: {available_count}/{total_count} actions ({available_count/total_count*100:.1f}%)")
        
        # build graph state
        self.state = self._get_graph_state()
        return self.state
    
    #batch graph state 생성 (새로운 Item-SKU-CP 구조)
    def _get_graph_state(self):
        
        graph_list = []
        # 초기 그래프 총 노드 수: Item + SKU + CP
        N = self.N_I_max + self.N_SKU_max + self.N_C
        
        for b in range(self.batch_size):
            # 노드 마스크 생성
            # Item: LS lookahead 큐에 실제로 있는 아이템만 포함 (운반 중이거나 도착한 아이템 제외)
            item_mask = torch.zeros(self.N_I_max, dtype=torch.bool)  # (N_I_max,)로 초기화
            
            # 각 LS의 lookahead 큐를 순회하며 실제로 있는 아이템만 True로 설정
            for ls_idx in range(self.N_L):
                for pos in range(self.look_ahead):
                    item_idx = self.ls_item_queue_lookahead[b, ls_idx, pos].item()
                    if item_idx >= 0 and item_idx < self.N_I[b]:  # 유효한 아이템인 경우
                        item_mask[item_idx] = True
            
            # 기존 방식 (주석 처리):
            # Item: visible하고 아직 CP에 도착하지 않은 아이템들 (운반 중 포함)
            # if self.N_I[b] > 0:
            #     actual_item_condition = ((self.item_current_cp[b, :self.N_I[b]] == -1) & 
            #                            self.item_visible[b, :self.N_I[b]])
            #     item_mask[:self.N_I[b]] = actual_item_condition
            
            # SKU: Order가 완료되지 않았으면 해당 Order의 모든 SKU 노드 유지
            # (같은 Order 내 SKU 간 연결 유지를 위해 Order 단위로 처리)
            sku_mask = torch.zeros(self.N_SKU_max, dtype=torch.bool)
            for sku_order_idx in range(self.N_SKU_max):
                if self.sku_order_order_idx[b, sku_order_idx] == -1:
                    break
                order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                # Order가 완료되지 않았으면 해당 Order의 모든 SKU 노드 유지
                if not self.order_completed[b, order_idx]:
                    sku_mask[sku_order_idx] = True
            
            # CP: 모든 CP 활성화
            cp_mask = torch.ones(self.N_C, dtype=torch.bool)
            
            # 전체 노드 마스크 결합
            full_node_mask = torch.cat([item_mask, sku_mask, cp_mask])

            # 8가지 엣지 타입별 subgraph 생성
            edge1, edge1_feat = subgraph(
                full_node_mask,
                self.edge1_item_item_successor_init[b],
                self.edge1_feature_init[b] if self.edge1_feature_init[b].numel() > 0 else None,
                relabel_nodes=False,
                num_nodes=N
            )

            edge2, edge2_feat = subgraph(
                full_node_mask,
                self.edge2_item_item_same_sku_init[b],
                self.edge2_feature_init[b] if self.edge2_feature_init[b].numel() > 0 else None,
                relabel_nodes=False,
                num_nodes=N
            )

            edge3, edge3_feat = subgraph(
                full_node_mask,
                self.edge3_item_sku_init[b],
                self.edge3_feature_init[b] if self.edge3_feature_init[b].numel() > 0 else None,
                relabel_nodes=False,
                num_nodes=N
            )

            edge4, edge4_feat = subgraph(
                full_node_mask,
                self.edge4_item_cp_init[b],
                self.edge4_feature_init[b] if self.edge4_feature_init[b].numel() > 0 else None,
                relabel_nodes=False,
                num_nodes=N
            )

            edge5, edge5_feat = subgraph(
                full_node_mask,
                self.edge5_cp_item_init[b],
                self.edge5_feature_init[b] if self.edge5_feature_init[b].numel() > 0 else None,
                relabel_nodes=False,
                num_nodes=N
            )

            edge6, edge6_feat = subgraph(
                full_node_mask,
                self.edge6_sku_sku_same_order_init[b],
                self.edge6_feature_init[b] if self.edge6_feature_init[b].numel() > 0 else None,
                relabel_nodes=False,
                num_nodes=N
            )

            edge7, edge7_feat = subgraph(
                full_node_mask,
                self.edge7_sku_sku_same_type_init[b],
                self.edge7_feature_init[b] if self.edge7_feature_init[b].numel() > 0 else None,
                relabel_nodes=False,
                num_nodes=N
            )

            edge8, edge8_feat = subgraph(
                full_node_mask,
                self.edge8_sku_cp_init[b],
                self.edge8_feature_init[b] if self.edge8_feature_init[b].numel() > 0 else None,
                relabel_nodes=False,
                num_nodes=N
            )

            x_b = self.node_feature[b]  # shape: (N, F)

            # Action decoder 사용 (텐서에서 직접 접근)
            # Action: (Item, CP) 조합
            action_decoder = self.action_index_decoder[b]  # 텐서에서 배치별 decoder
            action_mask = self.decision_available[b]  # 텐서에서 배치별 마스크

            # 가능한 액션 마스크 생성 (self.decision_available 기반)
            mask_available_actions = self.decision_available[b].clone()
            
            # PyG Batch가 올바르게 concat하도록 각 row를 개별 속성으로 저장
            # action_space에 따라 다른 개수의 decoder rows 사용
            decoder_row0 = action_decoder[0, :]  # Item 노드 인덱스 (num_actions,)
            
            if self.action_space == 'item-order-cp':
                decoder_row1 = action_decoder[1, :]  # SKU-Order 노드 인덱스
                decoder_row2 = action_decoder[2, :]  # CP 노드 인덱스
                
                data_b = Data(
                    x = x_b,
                    edge_index1 = edge1,
                    edge_index2 = edge2,
                    edge_index3 = edge3,
                    edge_index4 = edge4,
                    edge_attr4 = edge4_feat,
                    edge_index5 = edge5,
                    edge_attr5 = edge5_feat,
                    edge_index6 = edge6,
                    edge_index7 = edge7,
                    edge_index8 = edge8,
                    edge_attr8 = edge8_feat,
                    decoder_row0 = decoder_row0,
                    decoder_row1 = decoder_row1,
                    decoder_row2 = decoder_row2,
                    mask = mask_available_actions,
                    N_I_max = self.N_I_max,
                    N_SKU_max = self.N_SKU_max,
                    N_C = self.N_C,
                    max_num_actions = self.max_num_actions,
                    all_orders_completed = self.order_completed[b].all().item(),
                )
            
            elif self.action_space == 'item-cp':
                decoder_row1 = action_decoder[1, :]  # CP 노드 인덱스
                
                data_b = Data(
                    x = x_b,
                    edge_index1 = edge1,
                    edge_index2 = edge2,
                    edge_index3 = edge3,
                    edge_index4 = edge4,
                    edge_attr4 = edge4_feat,
                    edge_index5 = edge5,
                    edge_attr5 = edge5_feat,
                    edge_index6 = edge6,
                    edge_index7 = edge7,
                    edge_index8 = edge8,
                    edge_attr8 = edge8_feat,
                    decoder_row0 = decoder_row0,
                    decoder_row1 = decoder_row1,
                    mask = mask_available_actions,
                    N_I_max = self.N_I_max,
                    N_SKU_max = self.N_SKU_max,
                    N_C = self.N_C,
                    max_num_actions = self.max_num_actions,
                    all_orders_completed = self.order_completed[b].all().item(),
                )
            
            elif self.action_space == 'item-order':
                decoder_row1 = action_decoder[1, :]  # SKU-Order 노드 인덱스
                
                data_b = Data(
                    x = x_b,
                    edge_index1 = edge1,
                    edge_index2 = edge2,
                    edge_index3 = edge3,
                    edge_index4 = edge4,
                    edge_attr4 = edge4_feat,
                    edge_index5 = edge5,
                    edge_attr5 = edge5_feat,
                    edge_index6 = edge6,
                    edge_index7 = edge7,
                    edge_index8 = edge8,
                    edge_attr8 = edge8_feat,
                    decoder_row0 = decoder_row0,
                    decoder_row1 = decoder_row1,
                    mask = mask_available_actions,
                    N_I_max = self.N_I_max,
                    N_SKU_max = self.N_SKU_max,
                    N_C = self.N_C,
                    max_num_actions = self.max_num_actions,
                    all_orders_completed = self.order_completed[b].all().item(),
                )
            
            elif self.action_space == 'order-cp':
                decoder_row1 = action_decoder[1, :]  # CP 노드 인덱스
                
                data_b = Data(
                    x = x_b,
                    edge_index1 = edge1,
                    edge_index2 = edge2,
                    edge_index3 = edge3,
                    edge_index4 = edge4,
                    edge_attr4 = edge4_feat,
                    edge_index5 = edge5,
                    edge_attr5 = edge5_feat,
                    edge_index6 = edge6,
                    edge_index7 = edge7,
                    edge_index8 = edge8,
                    edge_attr8 = edge8_feat,
                    decoder_row0 = decoder_row0,
                    decoder_row1 = decoder_row1,
                    mask = mask_available_actions,
                    N_I_max = self.N_I_max,
                    N_SKU_max = self.N_SKU_max,
                    N_C = self.N_C,
                    max_num_actions = self.max_num_actions,
                    all_orders_completed = self.order_completed[b].all().item(),
                )
            
            graph_list.append(data_b)

        #graph_batch = Batch.from_data_list(graph_list)
        return graph_list

    def _get_init_state(self):
        # 새로운 Item-SKU-CP 기반 노드 정의
        # sku_order_* 변수들이 이미 reset에서 초기화됨
        
        # 모든 노드 타입이 동일한 feature_dim 공유, 각 타입은 특정 인덱스만 사용
        # Item: [0~2] - item_current_position, robot_arrival_upper_bound, ls_remaining_time
        # SKU:  [3~8] - sku_fulfillment_rate, order_has_cp_assigned, order_progress, order_arrival_rate, order_sku_remaining_items, order_remaining_items
        # CP:   [9]   - has_assigned_order
        feature_dim = 10  # Item(0-2), SKU(3-8), CP(9)
        
        self.item_feature_init = torch.zeros(self.batch_size, self.N_I_max, feature_dim)# LS에서 FCFS로 나오는 아이템
        self.sku_feature_init = torch.zeros(self.batch_size, self.N_SKU_max, feature_dim)# Order에서 요구하는 SKU
        self.cp_feature_init = torch.zeros(self.batch_size, self.N_C, feature_dim)# Collection Point

        # Item 피처 초기화 (인덱스 0~2 사용) - 벡터 연산 + 정규화
        # 0: item_current_position (정규화: 0 ~ max_item_position)
        if self.max_item_position > 0:
            self.item_feature_init[:, :, 0] = self.item_current_position.float() / self.max_item_position
        else:
            self.item_feature_init[:, :, 0] = self.item_current_position.float()
        
        # 1: robot_arrival_upper_bound - 초기에는 모든 로봇이 LS 큐에 있으므로 upper bound 계산
        # 초기값은 1.0 (max)
        self.item_feature_init[:, :, 1] = 1.0
        
        # 각 배치별로 처리
        for b in range(self.batch_size):
            for item_idx in range(self.N_I[b]):
                ls_idx = self.item_loading_station[b, item_idx].item()
                item_position = self.item_current_position[b, item_idx].item()
                
                # LS의 로봇 큐에 있는 로봇 수 계산
                robot_queue = self.ls_robot_queue[b, ls_idx]
                num_robots_in_queue = (robot_queue >= 0).sum().item()
                
                # 아이템이 로봇 큐 범위 내에 있는 경우 → 도착 시간 0
                if item_position < num_robots_in_queue:
                    self.item_feature_init[b, item_idx, 1] = 0.0
                # 그 외에는 1.0 유지 (초기에는 모든 로봇이 LS에 있으므로)
        
        # 2: ls_remaining_time - LS의 현재 작업 남은 시간 (정규화: 0 ~ LT)
        # 초기값은 0.0 (모든 LS가 idle 상태)
        self.item_feature_init[:, :, 2] = 0.0
        
        # 각 배치별로 처리
        for b in range(self.batch_size):
            for item_idx in range(self.N_I[b]):
                ls_idx = self.item_loading_station[b, item_idx].item()
                ls_remaining = self.ls_remaining_time[b, ls_idx].item()
                # 정규화: 0 ~ max_ls_time (LT)
                if self.max_ls_time > 0:
                    self.item_feature_init[b, item_idx, 2] = ls_remaining / self.max_ls_time
                else:
                    self.item_feature_init[b, item_idx, 2] = ls_remaining

        # SKU 피처 초기화 (인덱스 3~8 사용) - sku_order 변수들 사용 + 정규화
        for b in range(self.batch_size):
            for sku_order_idx in range(self.N_SKU_max):
                # 유효한 SKU Order인지 확인
                if self.sku_order_order_idx[b, sku_order_idx] == -1:
                    break  # 더 이상 유효한 SKU Order가 없음
                
                order_idx = self.sku_order_order_idx[b, sku_order_idx].item()
                sku_type = self.sku_order_sku_type[b, sku_order_idx].item()
                sku_node_idx = self.N_I_max + sku_order_idx
                required_qty = self.sku_order_qty[b, sku_order_idx].item()
                
                # 3: sku_fulfillment_rate (초기값 = 0.0, 아직 충족 안됨)
                self.sku_feature_init[b, sku_order_idx, 3] = 0.0
                
                # 4: order_has_cp_assigned (초기값 = 0, 아직 할당 안됨)
                self.sku_feature_init[b, sku_order_idx, 4] = 0.0
                
                # 5: order_progress (초기값 = 0, 아직 진행 안됨)
                self.sku_feature_init[b, sku_order_idx, 5] = 0.0
                
                # 6: order_arrival_rate (초기값 = 0, 아직 도착 안됨)
                self.sku_feature_init[b, sku_order_idx, 6] = 0.0
                
                # 7: order_sku_remaining_items (초기값 = required_qty, 정규화)
                if self.max_order_sku_remaining > 0:
                    self.sku_feature_init[b, sku_order_idx, 7] = required_qty / self.max_order_sku_remaining
                else:
                    self.sku_feature_init[b, sku_order_idx, 7] = required_qty
                
                # 8: order_remaining_items (초기값 = order의 전체 required, 정규화)
                order_total_required = self.order_sku_requirements[b, order_idx, :].sum().item()
                if self.max_order_remaining > 0:
                    self.sku_feature_init[b, sku_order_idx, 8] = order_total_required / self.max_order_remaining
                else:
                    self.sku_feature_init[b, sku_order_idx, 8] = order_total_required

        # CP 피처 초기화 (인덱스 9 사용)
        # 모든 값이 이미 0으로 초기화됨
        # 9: has_assigned_order (초기값: 0)

        self._get_init_edge()
    
    def _get_init_edge(self):
        # 새로운 Item-SKU-CP 기반 8가지 엣지 정의

        # 8가지 엣지 타입 초기화
        self.edge1_item_item_successor_init = []    # 1. Item-Item 방향성 (LS 내 선후관계)
        self.edge1_feature_init = []
        
        self.edge2_item_item_same_sku_init = []     # 2. Item-Item 무방향 (같은 SKU)  
        self.edge2_feature_init = []
        
        self.edge3_item_sku_init = []               # 3. Item-SKU 무방향 (같은 SKU)
        self.edge3_feature_init = []
        
        self.edge4_item_cp_init = []                # 4. Item-CP 방향성 (아이템 → CP)
        self.edge4_feature_init = []
        
        self.edge5_cp_item_init = []                # 5. CP-Item 방향성 (CP → 아이템)
        self.edge5_feature_init = []
        
        self.edge6_sku_sku_same_order_init = []     # 6. SKU-SKU 무방향 (같은 Order)
        self.edge6_feature_init = []
        
        self.edge7_sku_sku_same_type_init = []      # 7. SKU-SKU 무방향 (같은 SKU 타입)
        self.edge7_feature_init = []
        
        self.edge8_sku_cp_init = []                 # 8. SKU-CP 무방향 (Order-CP 할당)
        self.edge8_feature_init = []
        
        # Action decoders - 텐서로 초기화 (action_space에 따라 크기 결정)
        if self.action_space == 'item-order-cp':
            decoder_rows = 3  # Item, SKU-Order, CP
        elif self.action_space == 'item-cp':
            decoder_rows = 2  # Item, CP
        elif self.action_space == 'item-order':
            decoder_rows = 2  # Item, SKU-Order
        elif self.action_space == 'order-cp':
            decoder_rows = 2  # SKU-Order, CP
        else:
            decoder_rows = 2  # 기본값
        
        self.action_index_decoder = torch.full((self.batch_size, decoder_rows, self.max_num_actions), -1, dtype=torch.long)

        for b in range(self.batch_size):
            # 노드 인덱스 계산
            # Item: 0 ~ N_I_max-1
            # SKU: N_I_max ~ N_I_max+N_SKU_max-1  
            # CP: N_I_max+N_SKU_max ~ N_I_max+N_SKU_max+N_C-1
            
            item_start = 0
            sku_start = self.N_I_max
            cp_start = self.N_I_max + self.N_SKU_max
            
            # ═══ 1. Item-Item 방향성 엣지 (LS 내 선후관계) ═══
            edge1_tmp = [[], []]
            edge1_feat_tmp = []
            
            for item_idx in range(self.N_I[b]):
                successor_idx = self.item_successor[b, item_idx].item()
                if successor_idx != -1:  # 후속 아이템이 있는 경우
                    edge1_tmp[0].append(item_start + item_idx)
                    edge1_tmp[1].append(item_start + successor_idx)
                    edge1_feat_tmp.append([])  # feature 없음
            
            # ═══ 2. Item-Item 무방향 엣지 (같은 SKU) ═══
            edge2_tmp = [[], []]
            edge2_feat_tmp = []
            
            for i in range(self.N_I[b]):
                for j in range(i + 1, self.N_I[b]):
                    if self.item_has_same_sku[b, i, j]:
                        # 양방향 연결
                        edge2_tmp[0].extend([item_start + i, item_start + j])
                        edge2_tmp[1].extend([item_start + j, item_start + i])
                        edge2_feat_tmp.extend([[], []])  # feature 없음
            
            # ═══ 3. Item-SKU 무방향 엣지 (같은 SKU) ═══
            edge3_tmp = [[], []]
            edge3_feat_tmp = []
            
            for item_idx in range(self.N_I[b]):
                item_sku_type = self.item_sku_type[b, item_idx].item()
                # sku_order 변수들을 사용하여 같은 SKU 타입 찾기
                for sku_order_idx in range(self.N_SKU_max):
                    # 유효한 SKU Order인지 확인
                    if self.sku_order_order_idx[b, sku_order_idx] == -1:
                        break
                    
                    if self.sku_order_sku_type[b, sku_order_idx] == item_sku_type:
                        sku_node_idx = sku_start + sku_order_idx  # SKU 노드 인덱스
                        # 양방향 연결
                        edge3_tmp[0].extend([item_start + item_idx, sku_node_idx])
                        edge3_tmp[1].extend([sku_node_idx, item_start + item_idx])
                        edge3_feat_tmp.extend([[], []])  # feature 없음
            
            # ═══ 4. Item-CP 방향성 엣지 (아이템 → CP) ═══
            edge4_tmp = [[], []]
            edge4_feat_tmp = []
            
            for item_idx in range(self.N_I[b]):
                for cp_idx in range(self.N_C):
                    edge4_tmp[0].append(item_start + item_idx)
                    edge4_tmp[1].append(cp_start + cp_idx)
                    # forward travel_time feature (정규화: 0 ~ max_forward_travel_time)
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    forward_travel_time = self.travel_time_ls_to_cp[b, ls_idx, cp_idx].item()
                    if self.max_forward_travel_time > 0:
                        normalized_time = forward_travel_time / self.max_forward_travel_time
                    else:
                        normalized_time = forward_travel_time
                    edge4_feat_tmp.append([normalized_time])
            
            # ═══ 5. CP-Item 방향성 엣지 (CP → 아이템) ═══
            edge5_tmp = [[], []]
            edge5_feat_tmp = []
            
            for cp_idx in range(self.N_C):
                for item_idx in range(self.N_I[b]):
                    edge5_tmp[0].append(cp_start + cp_idx)
                    edge5_tmp[1].append(item_start + item_idx)
                    # backward travel_time feature (정규화: 0 ~ max_backward_travel_time)
                    ls_idx = self.item_loading_station[b, item_idx].item()
                    backward_travel_time = self.travel_time_cp_to_ls[b, cp_idx, ls_idx].item()
                    if self.max_backward_travel_time > 0:
                        normalized_time = backward_travel_time / self.max_backward_travel_time
                    else:
                        normalized_time = backward_travel_time
                    edge5_feat_tmp.append([normalized_time])
            
            # ═══ 6. SKU-SKU 무방향 엣지 (같은 Order) ═══
            edge6_tmp = [[], []]
            edge6_feat_tmp = []
            
            # 같은 주문에 속하는 SKU 노드들끼리 연결
            for order_idx in range(self.N_O):
                order_sku_nodes = []
                # sku_order 변수들을 사용하여 같은 주문의 SKU 찾기
                for sku_order_idx in range(self.N_SKU_max):
                    # 유효한 SKU Order인지 확인
                    if self.sku_order_order_idx[b, sku_order_idx] == -1:
                        break
                    
                    if self.sku_order_order_idx[b, sku_order_idx] == order_idx:
                        sku_node_idx = sku_start + sku_order_idx
                        order_sku_nodes.append(sku_node_idx)
                
                # 같은 주문의 SKU 노드들끼리 양방향 연결
                for i in range(len(order_sku_nodes)):
                    for j in range(i + 1, len(order_sku_nodes)):
                        sku_i = order_sku_nodes[i]
                        sku_j = order_sku_nodes[j]
                        edge6_tmp[0].extend([sku_i, sku_j])
                        edge6_tmp[1].extend([sku_j, sku_i])
                        edge6_feat_tmp.extend([[], []])  # feature 없음
            
            # ═══ 7. SKU-SKU 무방향 엣지 (같은 SKU 타입) ═══
            edge7_tmp = [[], []]
            edge7_feat_tmp = []
            
            # 같은 SKU 타입을 가진 SKU 노드들끼리 연결
            for sku_type in range(self.N_S):
                same_type_sku_nodes = []
                # sku_order 변수들을 사용하여 같은 SKU 타입 찾기
                for sku_order_idx in range(self.N_SKU_max):
                    # 유효한 SKU Order인지 확인
                    if self.sku_order_order_idx[b, sku_order_idx] == -1:
                        break
                    
                    if self.sku_order_sku_type[b, sku_order_idx] == sku_type:
                        sku_node_idx = sku_start + sku_order_idx
                        same_type_sku_nodes.append(sku_node_idx)
                
                # 같은 SKU 타입의 노드들끼리 양방향 연결
                for i in range(len(same_type_sku_nodes)):
                    for j in range(i + 1, len(same_type_sku_nodes)):
                        sku_i = same_type_sku_nodes[i]
                        sku_j = same_type_sku_nodes[j]
                        edge7_tmp[0].extend([sku_i, sku_j])
                        edge7_tmp[1].extend([sku_j, sku_i])
                        edge7_feat_tmp.extend([[], []])  # feature 없음
            
            # ═══ 8. SKU-CP 무방향 엣지 (Order-CP 할당) ═══
            edge8_tmp = [[], []]
            edge8_feat_tmp = []
            
            # sku_order 변수들을 사용하여 SKU-CP 엣지 생성
            for sku_order_idx in range(self.N_SKU_max):
                # 유효한 SKU Order인지 확인
                if self.sku_order_order_idx[b, sku_order_idx] == -1:
                    break
                
                sku_node_idx = sku_start + sku_order_idx
                for cp_idx in range(self.N_C):
                    # 양방향 연결
                    edge8_tmp[0].extend([sku_node_idx, cp_start + cp_idx])
                    edge8_tmp[1].extend([cp_start + cp_idx, sku_node_idx])
                    # is_assigned feature (초기값: False)
                    edge8_feat_tmp.extend([[0.0], [0.0]])
            
            # 텐서로 변환하여 저장
            self.edge1_item_item_successor_init.append(
                torch.tensor(edge1_tmp, dtype=torch.long) if edge1_tmp[0] 
                else torch.zeros((2, 0), dtype=torch.long)
            )
            self.edge1_feature_init.append(
                torch.tensor(edge1_feat_tmp, dtype=torch.float) if edge1_feat_tmp
                else torch.zeros((0, 0), dtype=torch.float)
            )
            
            self.edge2_item_item_same_sku_init.append(
                torch.tensor(edge2_tmp, dtype=torch.long) if edge2_tmp[0]
                else torch.zeros((2, 0), dtype=torch.long)
            )
            self.edge2_feature_init.append(
                torch.tensor(edge2_feat_tmp, dtype=torch.float) if edge2_feat_tmp
                else torch.zeros((0, 0), dtype=torch.float)
            )
            
            self.edge3_item_sku_init.append(
                torch.tensor(edge3_tmp, dtype=torch.long) if edge3_tmp[0]
                else torch.zeros((2, 0), dtype=torch.long)
            )
            self.edge3_feature_init.append(
                torch.tensor(edge3_feat_tmp, dtype=torch.float) if edge3_feat_tmp
                else torch.zeros((0, 0), dtype=torch.float)
            )
            
            self.edge4_item_cp_init.append(
                torch.tensor(edge4_tmp, dtype=torch.long) if edge4_tmp[0]
                else torch.zeros((2, 0), dtype=torch.long)
            )
            self.edge4_feature_init.append(
                torch.tensor(edge4_feat_tmp, dtype=torch.float) if edge4_feat_tmp
                else torch.zeros((0, 1), dtype=torch.float)
            )
            
            self.edge5_cp_item_init.append(
                torch.tensor(edge5_tmp, dtype=torch.long) if edge5_tmp[0]
                else torch.zeros((2, 0), dtype=torch.long)
            )
            self.edge5_feature_init.append(
                torch.tensor(edge5_feat_tmp, dtype=torch.float) if edge5_feat_tmp
                else torch.zeros((0, 1), dtype=torch.float)
            )
            
            self.edge6_sku_sku_same_order_init.append(
                torch.tensor(edge6_tmp, dtype=torch.long) if edge6_tmp[0]
                else torch.zeros((2, 0), dtype=torch.long)
            )
            self.edge6_feature_init.append(
                torch.tensor(edge6_feat_tmp, dtype=torch.float) if edge6_feat_tmp
                else torch.zeros((0, 0), dtype=torch.float)
            )
            
            self.edge7_sku_sku_same_type_init.append(
                torch.tensor(edge7_tmp, dtype=torch.long) if edge7_tmp[0]
                else torch.zeros((2, 0), dtype=torch.long)
            )
            self.edge7_feature_init.append(
                torch.tensor(edge7_feat_tmp, dtype=torch.float) if edge7_feat_tmp
                else torch.zeros((0, 0), dtype=torch.float)
            )
            
            self.edge8_sku_cp_init.append(
                torch.tensor(edge8_tmp, dtype=torch.long) if edge8_tmp[0]
                else torch.zeros((2, 0), dtype=torch.long)
            )
            self.edge8_feature_init.append(
                torch.tensor(edge8_feat_tmp, dtype=torch.float) if edge8_feat_tmp
                else torch.zeros((0, 1), dtype=torch.float)
            )
            
            # ═══ Action Decoders ═══
            action_decoder_tmp = []
            
            if self.action_space == 'item-order-cp':
                # Action: (Item, SKU-Order, CP) 조합
                for item_idx in range(self.N_I[b]):
                    item_sku_type = self.item_sku_type[b, item_idx].item()
                    for sku_order_idx in range(self.N_SKU_max):
                        if self.sku_order_order_idx[b, sku_order_idx] == -1:
                            break
                        if self.sku_order_sku_type[b, sku_order_idx] == item_sku_type:
                            sku_node_idx = sku_start + sku_order_idx
                            for cp_idx in range(self.N_C):
                                cp_node_idx = cp_start + cp_idx
                                action_decoder_tmp.append([
                                    item_start + item_idx,  # Item 노드 인덱스
                                    sku_node_idx,            # SKU 노드 인덱스
                                    cp_node_idx              # CP 노드 인덱스
                                ])
            
            elif self.action_space == 'item-cp':
                # Action: (Item, CP) 조합
                for item_idx in range(self.N_I[b]):
                    for cp_idx in range(self.N_C):
                        cp_node_idx = cp_start + cp_idx
                        action_decoder_tmp.append([
                            item_start + item_idx,  # Item 노드 인덱스
                            cp_node_idx              # CP 노드 인덱스
                        ])
            
            elif self.action_space == 'item-order':
                # Action: (Item, SKU-Order) 조합
                for item_idx in range(self.N_I[b]):
                    item_sku_type = self.item_sku_type[b, item_idx].item()
                    for sku_order_idx in range(self.N_SKU_max):
                        if self.sku_order_order_idx[b, sku_order_idx] == -1:
                            break
                        if self.sku_order_sku_type[b, sku_order_idx] == item_sku_type:
                            sku_node_idx = sku_start + sku_order_idx
                            action_decoder_tmp.append([
                                item_start + item_idx,  # Item 노드 인덱스
                                sku_node_idx             # SKU 노드 인덱스
                            ])
            
            elif self.action_space == 'order-cp':
                # Action: (SKU-Order, CP) 조합
                for sku_order_idx in range(self.N_SKU_max):
                    if self.sku_order_order_idx[b, sku_order_idx] == -1:
                        break
                    sku_node_idx = sku_start + sku_order_idx
                    for cp_idx in range(self.N_C):
                        cp_node_idx = cp_start + cp_idx
                        action_decoder_tmp.append([
                            sku_node_idx,   # SKU 노드 인덱스
                            cp_node_idx     # CP 노드 인덱스
                        ])
            
            # 텐서에 직접 할당 (부족한 부분은 이미 -1로 패딩됨)
            actual_size = len(action_decoder_tmp)
            if actual_size > 0:
                action_data = torch.tensor(action_decoder_tmp, dtype=torch.long).T
                self.action_index_decoder[b, :, :actual_size] = action_data
    

    def get_robot_idx_from_cp(self, batch_idx, cp_idx):
        """
        CP에서 출발 가능한 로봇의 인덱스를 반환합니다.
        
        Args:
            batch_idx: 배치 인덱스
            cp_idx: CP 인덱스
            
        Returns:
            int: 로봇 인덱스 (출발 가능한 로봇이 없으면 -1)
        """
        for robot_idx in range(self.N_R):
            if (self.robot_state[batch_idx, robot_idx] == 4 and  # CP_WAIT
                self.robot_current_cp[batch_idx, robot_idx] == cp_idx and
                self.robot_now_remain_t[batch_idx, robot_idx] <= 0):  # 남은 시간이 0
                return robot_idx
        
        return -1  # 출발 가능한 로봇이 없음
    
    def create_gantt_chart(self, batch_idx: int = 0, show_plot: bool = True, policy_name: str = "RL Policy"):
        """
        간트차트 생성 및 표시
        
        Args:
            batch_idx: 시각화할 배치 인덱스 (기본값: 0)
            show_plot: 브라우저에 표시할지 여부
            policy_name: 정책 이름
            
        Returns:
            plotly.graph_objects.Figure: 생성된 간트차트 figure
        """
        from rl_gantt_visualizer import RLGanttVisualizer
        
        # 시각화기 생성
        visualizer = RLGanttVisualizer(
            num_ls=self.N_L,
            num_cp=self.N_C,
            num_robots=self.N_R,
            policy_name=policy_name
        )
        
        # 간트차트 생성
        fig = visualizer.create_gantt_from_env(self, batch_idx, show_plot)
        
        return fig