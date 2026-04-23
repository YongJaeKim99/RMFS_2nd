import random
import numpy as np

class RMFS_Environment(object):

    def __init__(self, block_rows, block_cols, block_h, block_w, Unit_PT, ST, UT, Large,
                 force_mask_stay=False):

        self.N_P = None  # 포드 수
        self.N_R = None
        self.Total_PodTask = 100

        self.block_rows = block_rows  # 블록 그리드 행 수
        self.block_cols = block_cols  # 블록 그리드 열 수
        self.block_h = block_h        # 블록 내부 행 수 (default 4)
        self.block_w = block_w        # 블록 내부 열 수 (default 2)
        self.N_S = block_rows * block_cols * block_h * block_w  # 총 Storage 수

        self.Unit_PT = Unit_PT
        self.ST = ST
        self.UT = UT
        self.Large = Large
        self.force_mask_stay = force_mask_stay  # True: action 0(Stay) 강제 마스킹

        self._max_episode_steps = None
        self.Pod_Sequence_in_WS = None
        self.Number_Task_Pod_in_WS = None
        self.Number_Task_Pod = None
        self.PT = None
        self.Pod_Init = None
        self.Robot_Init = None

        self.graph_state = None
        self.action = 0

        self.reward = 0
        self.current_time = 0

        #RMFS 환경에 사용되는 변수들
        self.WS_curseq = None
        self.WS_AT = None

        self.Pod_curseq = None
        self.Pod_AT = None
        self.Pod_loc = None
        self.Pod_S = None

        self.Robot_AT = None
        self.Robot_loc = None
        self.Robot_S = None

        self.Storage_AT = None

        self.Makespan = None
        self.Pre_Makespan = None
        self.Travel_distance = None
        self.Pre_Travel_distance = None

        self.curpod = None
        self.curws = None
        self.currobot = None
        self.curws_seq = None
        self.curpodtaskseq = None

        self.Remain_Number_Task_Pod_in_WS = None

        self.Robot_Pod = None

        self.Visit_Sequence = None

        self.n = None

        self.check = False
        self.done = False

        # Debug tracking
        self.decided_count = 0
        self.returned_count = 0
        self._pending_arrivals = []

        print('Environment created...')

    def seed(self, n):
        np.random.seed(n)

    def pod_assign(self):

        # first avaliable WS 찾기
        Min_WS_AT = 1000000
        Max_Robot_S = max(self.Robot_S)

        self.curws = -1
        self.currobot = - 1

        for w in range(self.N_W):
            if self.WS_curseq[w] < len(self.Pod_Sequence_in_WS[w]):  # WS의 Pod task가 끝나지 않은 경우

                curws_seq = self.WS_curseq[w]
                curpod = self.Pod_Sequence_in_WS[w][curws_seq]

                if self.Pod_S[curpod] == 0:  # Pod가 WS에 존재하는 경우
                    WS_AT = max(self.WS_AT[w], self.Pod_AT[curpod] + self.TT_WW[w][self.Pod_loc[curpod]])  # WS의 작업 시작 시간

                    if Min_WS_AT > WS_AT:
                        Min_WS_AT = WS_AT
                        self.curws = w
                        self.currobot = self.Robot_Pod[curpod]

                elif Max_Robot_S == 1:  # Pod가 Storage location에 존재하고 모든 로봇이 점유되지 않은 경우

                    Min_Robot_AT = 1000000
                    currobot = -1

                    # 가장 먼저 사용 가능한 로봇 할당
                    for r in range(self.N_R):
                        if self.Robot_S[r] == 1 and Min_Robot_AT > self.Robot_AT[r] + \
                                self.TT_SS[self.Robot_loc[r]][
                                    self.Pod_loc[curpod]]:
                            Min_Robot_AT = self.Robot_AT[r] + self.TT_SS[self.Robot_loc[r]][
                                self.Pod_loc[curpod]]
                            currobot = r

                    # 포드 도착시간: WS 사용 가능 시간, Pod 사용 가능 시간, 로봇 사용 가능 시간
                    WS_AT = max(self.WS_AT[w], self.Pod_AT[curpod] + self.TT_WS[w][self.Pod_loc[curpod]],
                                Min_Robot_AT + self.TT_WS[w][self.Pod_loc[curpod]])

                    if Min_WS_AT > WS_AT:
                        Min_WS_AT = WS_AT
                        self.curws = w
                        self.currobot = currobot

        if self.curws == -1:
            return True

        # pod 호출
        self.curws_seq = self.WS_curseq[self.curws]
        self.curpod = self.Pod_Sequence_in_WS[self.curws][self.curws_seq]
        self.curpodtaskseq = self.Pod_curseq[self.curpod]

        #Travel distance update
        if self.Pod_S[self.curpod] == 0:
            self.Travel_distance += self.TT_WW[self.curws][self.Pod_loc[self.curpod]]
        else:
            self.Travel_distance += self.TT_SS[self.Robot_loc[self.currobot]][self.Pod_loc[self.curpod]]
            self.Travel_distance += self.TT_WS[self.curws][self.Pod_loc[self.curpod]]

        self.Pod_arrival_time = Min_WS_AT

        if self.Pod_S[self.curpod] == 1:
            self.Storage_AT[self.Pod_loc[self.curpod]] = self.Pod_arrival_time - self.TT_WS[self.curws][self.Pod_loc[self.curpod]]

        self.Pod_departure_time = self.Pod_arrival_time + self.PT[self.curws][self.curws_seq]

        self.WS_AT[self.curws] = self.Pod_departure_time + self.ST
        self.WS_curseq[self.curws] += 1
        self.Pod_curseq[self.curpod] += 1
        self.Pod_loc[self.curpod] = self.curws
        self.Pod_S[self.curpod] = 0

        self.Robot_Pod[self.curpod] = self.currobot
        self.Robot_AT[self.currobot] = 1000000

        self.Robot_S[self.currobot] = 0
        self.Robot_loc[self.currobot] = self.curws

        self.Remain_Number_Task_Pod_in_WS[self.curpod][self.curws] -= 1
        self.Visit_Sequence[self.curpod][self.curws].pop(0)

        # === Graph state computation ===
        V = self.N_S + self.N_W

        # Storage-to-pod mapping
        storage_to_pod = np.full(self.N_S, -1, dtype=np.int32)
        for p in range(self.N_P):
            if self.Pod_S[p] == 1:
                storage_to_pod[self.Pod_loc[p]] = p

        # -- Storage node features (N_S, 4) --
        storage_features = np.zeros((self.N_S, 4), dtype=np.float32)
        storage_features[:, 0] = self.storage_coords[:, 0] / self.max_x
        storage_features[:, 1] = self.storage_coords[:, 1] / self.max_y
        storage_features[:, 2] = (self.Storage_AT >= 999999).astype(np.float32)  # occupied
        idle_s = np.maximum(0, self.Storage_AT - self.Pod_departure_time)
        idle_s[self.Storage_AT >= 999999] = 0  # occupied → 0
        storage_features[:, 3] = idle_s / max(self.Max_TT_WS, 1)

        # -- WS node features (N_W, 4) --
        ws_features = np.zeros((self.N_W, 4), dtype=np.float32)
        ws_features[:, 0] = self.ws_coords[:, 0] / self.max_x
        ws_features[:, 1] = self.ws_coords[:, 1] / self.max_y
        ws_features[:, 2] = (self.WS_AT > self.Pod_departure_time).astype(np.float32)  # busy
        ws_idle = np.maximum(0, self.WS_AT - self.Pod_departure_time)
        ws_features[:, 3] = ws_idle / max(self.Max_TT_WS, 1)

        # -- Edge features (V, V, 9) --
        edge_feat = np.zeros((V, V, 9), dtype=np.float32)

        # WS→Storage edges
        for w in range(self.N_W):
            wi = self.N_S + w
            edge_feat[wi, :self.N_S, 0] = 1.0  # type_ws_to_s
            edge_feat[wi, :self.N_S, 3] = self.TT_WS[w] / self.Max_TT_WS  # manhattan_dist

            for s in range(self.N_S):
                p = storage_to_pod[s]
                if p >= 0 and self.Visit_Sequence[p][w]:
                    edge_feat[wi, s, 4] = 1.0  # pod_stored_needs_ws
                    diff = self.Visit_Sequence[p][w][0] - self.WS_curseq[w]
                    edge_feat[wi, s, 5] = diff / max(self.Max_min_diff, 1)

        # Storage→WS edges
        for w in range(self.N_W):
            wi = self.N_S + w
            edge_feat[:self.N_S, wi, 1] = 1.0  # type_s_to_ws
            edge_feat[:self.N_S, wi, 3] = self.TT_WS[w] / self.Max_TT_WS

        # WS↔WS edges
        for w1 in range(self.N_W):
            for w2 in range(self.N_W):
                if w1 == w2:
                    continue
                wi1 = self.N_S + w1
                wi2 = self.N_S + w2
                edge_feat[wi1, wi2, 2] = 1.0  # type_ws_to_ws
                edge_feat[wi1, wi2, 3] = self.TT_WW[w1][w2] / max(self.Max_TT_WW, 1)

                shared_count = 0
                proximity_sum = 0.0
                for p in range(self.N_P):
                    if self.Remain_Number_Task_Pod_in_WS[p][w1] > 0 and self.Remain_Number_Task_Pod_in_WS[p][w2] > 0:
                        shared_count += 1
                        d1 = (self.Visit_Sequence[p][w1][0] - self.WS_curseq[w1]) if self.Visit_Sequence[p][w1] else self.Max_min_diff
                        d2 = (self.Visit_Sequence[p][w2][0] - self.WS_curseq[w2]) if self.Visit_Sequence[p][w2] else self.Max_min_diff
                        proximity_sum += min(d1, d2)

                edge_feat[wi1, wi2, 6] = shared_count / max(self.N_P, 1)
                if shared_count > 0:
                    edge_feat[wi1, wi2, 7] = (proximity_sum / shared_count) / max(self.Max_min_diff, 1)

                if self.Visit_Sequence[self.curpod][w2]:
                    edge_feat[wi1, wi2, 8] = 1.0  # curpod_needs_other_ws

        # -- Action mask (N_S+1) --
        action_mask = np.zeros(self.N_S + 1, dtype=np.bool_)
        available = self.Storage_AT <= self.Pod_departure_time + self.TT_WS[self.curws]
        action_mask[1:] = available
        if self.force_mask_stay:
            action_mask[0] = False  # Stay 강제 마스킹 (deadlock 방지)
        else:
            action_mask[0] = True   # Stay always valid

        self.graph_state = {
            'storage_features': storage_features,
            'ws_features': ws_features,
            'edge_feat': edge_feat,
            'curws_idx': self.curws,
            'action_mask': action_mask,
        }

        return False

    def reset(self, seed, N_P, N_R, Total_PodTask, N_W):

        self.N_W = N_W

        # Storage 좌표: block_h × block_w 블록을 block_rows × block_cols 그리드로 배치 (블록 간 1칸 gap)
        storage_coords = []
        for br in range(self.block_rows):
            for bc in range(self.block_cols):
                x_base = bc * (self.block_w + 1)
                y_base = br * (self.block_h + 1)
                for row in range(self.block_h):
                    for col in range(self.block_w):
                        storage_coords.append((x_base + col, y_base + row))
        storage_location_coordinate = np.array(storage_coords)
        self.storage_coords = storage_location_coordinate

        # Workstation 좌표: storage 영역 오른쪽 배치
        storage_max_x = (self.block_cols - 1) * (self.block_w + 1) + (self.block_w - 1)
        storage_max_y = (self.block_rows - 1) * (self.block_h + 1) + (self.block_h - 1)
        ws_x = storage_max_x + 3
        if self.N_W == 1:
            ws_ys = [storage_max_y // 2]
        else:
            ws_spacing = storage_max_y / (self.N_W - 1)
            ws_ys = [round(i * ws_spacing) for i in range(self.N_W)]
        workstation_coordinate = np.array([(ws_x, y) for y in ws_ys])
        self.ws_coords = workstation_coordinate
        self.max_x = max(float(ws_x), 1.0)
        self.max_y = max(float(storage_max_y), 1.0)

        # TT_SS 계산
        x_diff_ss = np.abs(
            storage_location_coordinate[:, np.newaxis, 0] - storage_location_coordinate[np.newaxis, :, 0])
        y_diff_ss = np.abs(
            storage_location_coordinate[:, np.newaxis, 1] - storage_location_coordinate[np.newaxis, :, 1])
        self.TT_SS = (x_diff_ss + y_diff_ss) * self.UT

        # TT_WW 계산
        x_diff_ww = np.abs(workstation_coordinate[:, np.newaxis, 0] - workstation_coordinate[np.newaxis, :, 0])
        y_diff_ww = np.abs(workstation_coordinate[:, np.newaxis, 1] - workstation_coordinate[np.newaxis, :, 1])
        self.TT_WW = (x_diff_ww + y_diff_ww) * self.UT

        # TT_WS 계산
        x_diff_ws = np.abs(workstation_coordinate[:, np.newaxis, 0] - storage_location_coordinate[np.newaxis, :, 0])
        y_diff_ws = np.abs(workstation_coordinate[:, np.newaxis, 1] - storage_location_coordinate[np.newaxis, :, 1])
        self.TT_WS = (x_diff_ws + y_diff_ws) * self.UT

        self.Max_TT_WW = np.max(self.TT_WW)
        self.Max_TT_WS = np.max(self.TT_WS)

        if self.Large:
            self.Max_min_diff = 35
        else:
            self.Max_min_diff = 3

        self.location_search_sequence = np.argsort(self.TT_WS, axis=1)

        self.N_P = N_P
        self.N_R = N_R
        self.Total_PodTask = Total_PodTask
        self._max_episode_steps = Total_PodTask

        if seed >= 0:
            random.seed(seed)

        #랜덤 파라미터 초기화

        if self.Large:

            min_value = 1
            self.n = [min_value] * self.N_W
            remaining = self.Total_PodTask - self.N_W * min_value

            average = self.Total_PodTask / self.N_W
            max_value = int(average * 1.1)

            while remaining > 0:
                index = random.choice(range(self.N_W))
                possible_increase = min(max_value - self.n[index], remaining)
                if possible_increase > 0:
                    self.n[index] += 1
                    remaining -= 1

            self.Visit_Sequence = [[[] for _ in range(self.N_W)] for _ in
                                   range(self.N_P)]
            self.Number_Task_Pod_in_WS = [[0 for _ in range(self.N_W)] for _ in range(self.N_P)]
            self.Number_Task_Pod = [0 for _ in range(self.N_P)]

            self.Pod_Sequence_in_WS = [[] for _ in range(self.N_W)]

            n_count = [0] * self.N_W

            while sum(n_count) < sum(self.n):
                random_n = min(random.randint(25, 35), N_P)
                random_population = random.sample(range(0, N_P), random_n)
                for i in range(self.N_W):
                    random_numbers = random_population.copy()
                    random.shuffle(random_numbers)
                    while random_numbers and n_count[i] < self.n[i]:
                        curpod = random_numbers.pop(0)
                        if random.random() < 0.9:
                            self.Pod_Sequence_in_WS[i].append(curpod)
                            self.Visit_Sequence[curpod][i].append(n_count[i])
                            self.Number_Task_Pod_in_WS[curpod][i] += 1
                            self.Number_Task_Pod[curpod] += 1
                            n_count[i] += 1

            self.PT = [[0 for _ in self.Pod_Sequence_in_WS[i]] for i in range(self.N_W)]

            for i in range(self.N_W):
                for j in range(len(self.Pod_Sequence_in_WS[i])):
                    self.PT[i][j] = self.Unit_PT * random.randint(1, 5)
        else:
            self.n = []

            if self.Total_PodTask % 2 == 0:
                self.n.append(self.Total_PodTask // 2)
                self.n.append(self.Total_PodTask // 2)
            else:
                if random.random() < 0.5:
                    self.n.append(self.Total_PodTask // 2)
                    self.n.append(self.Total_PodTask // 2 + 1)
                else:
                    self.n.append(self.Total_PodTask // 2 + 1)
                    self.n.append(self.Total_PodTask // 2)

            self.Number_Task_Pod = [0 for _ in range(self.N_P)]

            self.Pod_Sequence_in_WS = [[] for _ in range(self.N_W)]
            self.Visit_Sequence = [[[] for _ in range(self.N_W)] for _ in range(self.N_P)]
            self.Number_Task_Pod_in_WS = [[0 for _ in range(self.N_W)] for _ in range(self.N_P)]
            self.Number_Task_Pod = [0 for _ in range(self.N_P)]
            for i in range(self.N_W):
                curpod = random.randint(0, self.N_P - 1)
                self.Pod_Sequence_in_WS[i].append(curpod)
                self.Visit_Sequence[curpod][i].append(0)
                self.Number_Task_Pod_in_WS[curpod][0] += 1
                self.Number_Task_Pod[curpod] += 1
                for j in range(self.n[i] - 1):
                    while True:
                        next_pod = random.randint(0, self.N_P - 1)
                        if next_pod not in self.Pod_Sequence_in_WS[i]:
                            self.Pod_Sequence_in_WS[i].append(next_pod)
                            self.Visit_Sequence[next_pod][i].append(j + 1)
                            self.Number_Task_Pod_in_WS[next_pod][i] += 1
                            self.Number_Task_Pod[next_pod] += 1

                            break


            self.PT = [[0 for _ in self.Pod_Sequence_in_WS[i]] for i in range(self.N_W)]

            for i in range(self.N_W):
                for j in range(len(self.Pod_Sequence_in_WS[i])):
                    self.PT[i][j] = self.Unit_PT

        self.Pod_Init = random.sample(range(self.N_S), self.N_P)
        self.Robot_Init = random.sample(range(self.N_S), self.N_R)

        self.Number_Task_Pod_in_WS = np.array(self.Number_Task_Pod_in_WS)
        self.Number_Task_Pod = np.array(self.Number_Task_Pod)
        self.Pod_Init = np.array(self.Pod_Init)
        self.Robot_Init = np.array(self.Robot_Init)

        #RMFS 환경에 사용되는 변수들 초기화
        self.WS_curseq = np.zeros(self.N_W, dtype=np.int16)
        self.WS_AT = np.zeros(self.N_W)

        self.Pod_curseq = np.zeros(self.N_P, dtype=np.int16)
        self.Pod_AT = np.zeros(self.N_P)
        self.Pod_loc = np.copy(self.Pod_Init)
        self.Pod_S = np.ones(self.N_P, dtype=np.int16)

        self.Robot_AT = np.zeros(self.N_R)
        self.Robot_loc = np.copy(self.Robot_Init)
        self.Robot_S = np.ones(self.N_R, dtype=np.int16)

        self.Storage_AT = np.zeros(self.N_S)

        self.Remain_Number_Task_Pod_in_WS = np.copy(self.Number_Task_Pod_in_WS)

        for i in self.Pod_loc:
            self.Storage_AT[i] = 1000000

        self.Makespan = 0
        self.Pre_Makespan = 0
        self.LB_Makespan = 0
        self.Pre_LB_Makespan = 0
        self.Travel_distance = 0
        self.Pre_Travel_distance = 0

        self.Robot_Pod = np.full(self.N_P, -1)

        infeasible = self.pod_assign()

        # 초기 LB 계산 (pod_assign 이후)
        self.LB_Makespan = self.compute_lb_makespan()
        self.Pre_LB_Makespan = self.LB_Makespan

        self.current_time = 0

        self.done = False

        # Debug tracking reset
        self.decided_count = 0
        self.returned_count = 0
        self._pending_arrivals = []

        return self.graph_state

    # 시뮬레이션 한 단계 진행
    def step(self, action):

        self.action = action

        infeasible2 = False

        if self.action == 0:
            # Stay: pod remains at WS
            self.Pod_AT[self.curpod] = self.Pod_departure_time
            self.Pod_loc[self.curpod] = self.curws
            self.Robot_loc[self.currobot] = self.curws
            self.Pod_S[self.curpod] = 0
            self.Robot_S[self.currobot] = 0
        else:
            # Direct storage assignment: action 1~N_S → storage index (action-1)
            curstorage = self.action - 1

            if curstorage < 0 or curstorage >= self.N_S:
                infeasible2 = True
            elif self.Storage_AT[curstorage] > self.Pod_departure_time + self.TT_WS[self.curws][curstorage]:
                infeasible2 = True

            if not infeasible2:
                self.Storage_AT[curstorage] = 1000000
                self.Robot_AT[self.currobot] = self.Pod_departure_time + self.TT_WS[self.curws][curstorage]
                self.Pod_AT[self.curpod] = self.Robot_AT[self.currobot]
                self.Pod_loc[self.curpod] = curstorage
                self.Robot_loc[self.currobot] = curstorage
                self.Pod_S[self.curpod] = 1
                self.Robot_S[self.currobot] = 1
                self.Robot_Pod[self.curpod] = -1
                self.Travel_distance += self.TT_WS[self.curws][curstorage]

        # --- Debug tracking ---
        self.decided_count += 1
        if self.action == 0:
            self.returned_count += 1  # Stay = 즉시 반납 간주
        elif not infeasible2:
            self._pending_arrivals.append(self.Pod_AT[self.curpod])

        infeasible = False

        if self.current_time < self._max_episode_steps - 1:

            infeasible = self.pod_assign()

            # pending pods: Pod_arrival_time(다음 task 시뮬레이션 시간) 이전에 도착한 pod 카운트
            if not infeasible:
                new_pending = []
                for at in self._pending_arrivals:
                    if at <= self.Pod_arrival_time:
                        self.returned_count += 1
                    else:
                        new_pending.append(at)
                self._pending_arrivals = new_pending
            else:
                self.returned_count += len(self._pending_arrivals)
                self._pending_arrivals = []
        else:
            # 마지막 step: 모든 pending pod는 결국 도착
            self.returned_count += len(self._pending_arrivals)
            self._pending_arrivals = []

        if infeasible or infeasible2:
            self.reward = -100000
            self.lb_reward = -100000
            self.done = True
            self.Makespan = 100000
            self.LB_Makespan = 100000
        else:
            self.Makespan = max(self.Pod_AT)
            self.LB_Makespan = self.compute_lb_makespan()
            self.reward = self.Pre_Makespan - self.Makespan
            self.lb_reward = self.Pre_LB_Makespan - self.LB_Makespan

        self.current_time += 1

        self.Pre_Makespan = self.Makespan
        self.Pre_LB_Makespan = self.LB_Makespan

        self.Pre_Travel_distance = self.Travel_distance

        return self.graph_state, self.reward, self.isFinished(self.current_time), infeasible, self.Makespan

    def compute_lb_makespan(self):
        """
        현재 상태에서 최종 makespan의 lower bound를 계산한다.

        LB = max(
            max(Pod_AT),                # 이미 확정된 Pod 완료 시각
            max_w(ws_workload_lb[w])    # 각 WS의 최소 작업완료 시각
        )

        각 WS의 workload LB:
            WS_AT[w]  (WS 가용 시각)
            + sum of (min_travel + PT + ST) for each remaining task
            + min_travel (마지막 pod storage 반납)
        """
        lb = float(max(self.Pod_AT))

        for w in range(self.N_W):
            remaining = len(self.Pod_Sequence_in_WS[w]) - self.WS_curseq[w]
            if remaining <= 0:
                continue

            min_tt = float(np.min(self.TT_WS[w]))

            # WS가 다시 가용해지는 시각부터 시작
            ws_lb = float(self.WS_AT[w])

            # 남은 각 task: 최소 pod 도착 이동시간 + 처리시간 + 셋업시간
            for k in range(remaining):
                task_idx = self.WS_curseq[w] + k
                ws_lb += min_tt + self.PT[w][task_idx] + self.ST

            # 마지막 pod의 storage 반납 최소 이동시간
            ws_lb += min_tt

            lb = max(lb, ws_lb)

        return lb

    def find_nearest_available_storage(self, x, y):
        """
        연속 (x, y) 좌표에서 가장 가까운 가용 storage를 찾는다.

        Args:
            x: float, warehouse x-coordinate
            y: float, warehouse y-coordinate

        Returns:
            int: 가용 storage 중 맨하탄 거리가 최소인 storage index (0-indexed)
        """
        available = self.Storage_AT <= self.Pod_departure_time + self.TT_WS[self.curws]
        distances = np.abs(self.storage_coords[:, 0] - x) + np.abs(self.storage_coords[:, 1] - y)
        distances[~available] = np.inf
        return int(np.argmin(distances))

    def step_continuous(self, x, y):
        """
        연속 (x, y) 좌표로 step. nearest available storage를 찾아 기존 step()에 위임.

        Args:
            x: float, warehouse x-coordinate
            y: float, warehouse y-coordinate

        Returns:
            step()과 동일: (graph_state, reward, done, infeasible, makespan)
        """
        storage_idx = self.find_nearest_available_storage(x, y)
        return self.step(storage_idx + 1)  # +1: action 0=Stay offset

    def isFinished(self, current_time):
        return current_time == self._max_episode_steps or self.done

    def random_action(self):
        return random.randint(0, self.N_S)
