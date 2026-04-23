"""
Batched RMFS Environment with Graph State.

B개의 RMFS_Environment 인스턴스를 감싸서 배치 인터페이스를 제공한다.
Graph state (storage features, ws features, edge features, action mask)를 반환.
"""
import io
import sys
from dataclasses import dataclass

import torch
import numpy as np

from RMFS_ENV import RMFS_Environment


@dataclass
class RMFSState:
    """Batched RMFS graph state."""
    storage_features: torch.Tensor   # (B, N_S, 4)
    ws_features: torch.Tensor        # (B, N_W, 4)
    edge_feat: torch.Tensor          # (B, V, V, 9)
    curws_idx: torch.Tensor          # (B,) int
    action_mask: torch.Tensor        # (B, N_S+1) bool


class RMFSBatchEnv:
    """
    B개의 독립적인 RMFS_Environment를 병렬로 관리하는 배치 환경.

    Interface:
        reset(problem) -> RMFSState
        step(actions)  -> (RMFSState, rewards, all_done)
        get_makespan() -> Tensor(B,)
    """

    def __init__(self, env_params, device='cpu', reward_type='stepwise'):
        self.batch_size = env_params['batch_size']
        self.device = torch.device(device)
        self.env_params = env_params
        self.reward_type = reward_type

        block_rows = env_params['block_rows']
        block_cols = env_params['block_cols']
        block_h = env_params.get('block_h', 4)
        block_w = env_params.get('block_w', 2)
        Unit_PT = env_params['Unit_PT']
        ST = env_params['ST']
        UT = env_params['UT']
        Large = env_params['Large']
        force_mask_stay = env_params.get('force_mask_stay', False)

        self.N_S = block_rows * block_cols * block_h * block_w
        self.N_W = env_params['N_W']
        self.V = self.N_S + self.N_W
        self.d_edge = 9  # edge feature dimension

        # 연속 action 좌표 스케일링용 (storage 영역 범위)
        self.storage_max_x = float((block_cols - 1) * (block_w + 1) + (block_w - 1))
        self.storage_max_y = float((block_rows - 1) * (block_h + 1) + (block_h - 1))

        # B개의 독립 RMFS 환경 생성 (print 억제)
        self.envs = []
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            for _ in range(self.batch_size):
                env = RMFS_Environment(block_rows, block_cols, block_h, block_w, Unit_PT, ST, UT, Large,
                                       force_mask_stay=force_mask_stay)
                self.envs.append(env)
        finally:
            sys.stdout = _stdout

        # Static adjacency matrix (same for all instances in batch)
        self.adj = self._build_adjacency()

        # 상태 추적 변수
        self.done_mask = None
        self.makespans = None
        self.step_count = 0

    def _build_adjacency(self):
        """Static adjacency: WS↔Storage (all pairs), WS↔WS, self-loops."""
        V = self.V
        adj = torch.zeros(V, V, dtype=torch.bool)

        # WS→Storage and Storage→WS (all pairs)
        for w in range(self.N_W):
            wi = self.N_S + w
            adj[wi, :self.N_S] = True  # WS→Storage
            adj[:self.N_S, wi] = True  # Storage→WS

        # WS↔WS (all pairs, w1≠w2)
        for w1 in range(self.N_W):
            for w2 in range(self.N_W):
                if w1 != w2:
                    adj[self.N_S + w1, self.N_S + w2] = True

        # Self-loops
        idx = torch.arange(V)
        adj[idx, idx] = True

        # Expand to (1, V, V) for broadcasting with batch
        return adj.unsqueeze(0)

    def _zero_graph_state(self):
        """Done 인스턴스용 zero state dict."""
        return {
            'storage_features': np.zeros((self.N_S, 4), dtype=np.float32),
            'ws_features': np.zeros((self.N_W, 4), dtype=np.float32),
            'edge_feat': np.zeros((self.V, self.V, self.d_edge), dtype=np.float32),
            'curws_idx': 0,
            'action_mask': np.zeros(self.N_S + 1, dtype=np.bool_),
        }

    def _stack_graph_states(self, graph_states):
        """per-instance graph_state dict 리스트 → batched RMFSState."""
        storage_features = torch.tensor(
            np.stack([gs['storage_features'] for gs in graph_states]),
            dtype=torch.float32, device=self.device
        )
        ws_features = torch.tensor(
            np.stack([gs['ws_features'] for gs in graph_states]),
            dtype=torch.float32, device=self.device
        )
        edge_feat = torch.tensor(
            np.stack([gs['edge_feat'] for gs in graph_states]),
            dtype=torch.float32, device=self.device
        )
        curws_idx = torch.tensor(
            [gs['curws_idx'] for gs in graph_states],
            dtype=torch.long, device=self.device
        )
        action_mask = torch.tensor(
            np.stack([gs['action_mask'] for gs in graph_states]),
            dtype=torch.bool, device=self.device
        )
        return RMFSState(
            storage_features=storage_features,
            ws_features=ws_features,
            edge_feat=edge_feat,
            curws_idx=curws_idx,
            action_mask=action_mask,
        )

    def reset(self, problem):
        """
        모든 환경을 problem의 시드로 초기화한다.

        Args:
            problem: dict from generate_rmfs_data_batch()

        Returns:
            state: RMFSState
        """
        seeds = problem['seeds']
        N_P = problem['N_P']
        N_R = problem['N_R']
        Total_PodTask = problem['Total_PodTask']
        N_W = problem['N_W']

        graph_states = []
        for i in range(self.batch_size):
            gs = self.envs[i].reset(seeds[i], N_P, N_R, Total_PodTask, N_W)
            graph_states.append(gs)

        self.done_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.makespans = torch.zeros(self.batch_size, device=self.device)
        self.step_count = 0

        return self._stack_graph_states(graph_states)

    def step(self, actions):
        """
        모든 활성 환경에 action을 적용한다.

        Args:
            actions: (B,) int tensor (discrete) 또는 (B, 2) float tensor (continuous x,y in [0,1])

        Returns:
            state: RMFSState
            rewards: (B,) float tensor
            all_done: bool
        """
        is_continuous = (actions.dim() == 2 and actions.shape[1] == 2)
        actions_np = actions.cpu().numpy()

        graph_states = []
        rewards = torch.zeros(self.batch_size, device=self.device)

        for i in range(self.batch_size):
            if self.done_mask[i]:
                graph_states.append(self._zero_graph_state())
                continue

            if is_continuous:
                x = float(actions_np[i, 0]) * self.storage_max_x
                y = float(actions_np[i, 1]) * self.storage_max_y
                gs, r, done, infeasible, makespan = self.envs[i].step_continuous(x, y)
            else:
                gs, r, done, infeasible, makespan = self.envs[i].step(int(actions_np[i]))

            if self.reward_type == 'lb_stepwise':
                rewards[i] = self.envs[i].lb_reward
            else:
                rewards[i] = r
            self.makespans[i] = makespan

            if done:
                self.done_mask[i] = True
                graph_states.append(self._zero_graph_state())
            else:
                graph_states.append(gs)

        self.step_count += 1
        state = self._stack_graph_states(graph_states)
        all_done = self.done_mask.all().item()

        return state, rewards, all_done

    def get_makespan(self):
        """최종 makespan 반환. (B,)"""
        return self.makespans.clone()

    def get_active_mask(self):
        """활성(아직 완료되지 않은) 인스턴스 마스크 반환. (B,) True=active"""
        return ~self.done_mask

    def get_debug_counts(self, instance=0):
        """디버그용: 단일 인스턴스 (decided, returned, total, sim_time, makespan) 반환."""
        env = self.envs[instance]
        sim_time = getattr(env, 'Pod_departure_time', 0.0)
        makespan = getattr(env, 'Makespan', 0.0)
        return env.decided_count, env.returned_count, env._max_episode_steps, sim_time, makespan

    def get_debug_counts_all(self):
        """디버그용: 전체 배치 통계 반환. dict with avg/min/max for each metric."""
        decisions = [e.decided_count for e in self.envs]
        returned = [e.returned_count for e in self.envs]
        total = self.envs[0]._max_episode_steps
        makespans = [getattr(e, 'Makespan', 0.0) for e in self.envs]
        sim_times = [getattr(e, 'current_time', 0) for e in self.envs]
        return {
            'total': total,
            'decisions': {'avg': sum(decisions)/len(decisions), 'min': min(decisions), 'max': max(decisions)},
            'returned': {'avg': sum(returned)/len(returned), 'min': min(returned), 'max': max(returned)},
            'sim_time': {'avg': sum(sim_times)/len(sim_times), 'min': min(sim_times), 'max': max(sim_times)},
            'makespan': {'avg': sum(makespans)/len(makespans), 'min': min(makespans), 'max': max(makespans)},
        }
