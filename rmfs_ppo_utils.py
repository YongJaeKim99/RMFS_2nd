"""
PPO utilities for RMFS training with graph state.

Graph state (RMFSState 필드별)를 저장하는 PPO Memory.
Variable-length episode를 위한 mask_seq도 추가한다.
"""
import torch
from torch.distributions import Categorical


def eval_actions(pi, actions):
    """
    주어진 action에 대한 log probability와 entropy를 계산한다.

    Args:
        pi: (B, action_dim) softmaxed action probabilities
        actions: (B,) action indices

    Returns:
        log_probs: (B,) log pi(a|s)
        entropy: scalar mean entropy
    """
    dist = Categorical(pi.squeeze())
    log_probs = dist.log_prob(actions).reshape(-1)
    entropy = dist.entropy().mean()
    return log_probs, entropy


class RMFSPPOMemory:
    """
    PPO trajectory memory for RMFS with graph state.

    State fields stored separately:
    - storage_features (B, N_S, 4)
    - ws_features (B, N_W, 4)
    - edge_feat (B, V, V, 9)
    - curws_idx (B,)
    - action_mask (B, N_S+1)
    """

    def __init__(self, gamma, gae_lambda):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # State: per-field sequences
        self.storage_features_seq = []   # [T, tensor (B, N_S, 4)]
        self.ws_features_seq = []        # [T, tensor (B, N_W, 4)]
        self.edge_feat_seq = []          # [T, tensor (B, V, V, 9)]
        self.curws_idx_seq = []          # [T, tensor (B,)]
        self.action_mask_seq = []        # [T, tensor (B, N_S+1)]

        # Transition data
        self.action_seq = []     # [T, tensor (B,)]
        self.reward_seq = []     # [T, tensor (B,)]
        self.val_seq = []        # [T, tensor (B,)]
        self.done_seq = []       # [T, tensor (B,) bool]
        self.log_probs = []      # [T, tensor (B,)]
        self.mask_seq = []       # [T, tensor (B,) bool] True=active

        # Filled by transpose_data()
        self.t_old_val_seq = None  # [B, T]

    def push_state(self, state):
        """현재 step의 graph state를 저장."""
        self.storage_features_seq.append(state.storage_features.detach())
        self.ws_features_seq.append(state.ws_features.detach())
        self.edge_feat_seq.append(state.edge_feat.detach())
        self.curws_idx_seq.append(state.curws_idx.detach())
        self.action_mask_seq.append(state.action_mask.detach())

    def push_transition(self, action, log_prob, val, reward, done, mask):
        """현재 step의 transition 데이터를 저장."""
        self.action_seq.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.val_seq.append(val.detach())
        self.reward_seq.append(reward.detach())
        self.done_seq.append(done.detach())
        self.mask_seq.append(mask.detach())

    def transpose_data(self):
        """
        시퀀스를 stack & flatten: [T steps, B instances] → [B*T, ...]

        Returns tuple (11 elements):
            0: storage_features [B*T, N_S, 4]
            1: ws_features      [B*T, N_W, 4]
            2: edge_feat        [B*T, V, V, 9]
            3: curws_idx        [B*T]
            4: action_mask      [B*T, N_S+1]
            5: actions          [B*T]
            6: rewards          [B*T]
            7: values           [B*T]
            8: dones            [B*T]
            9: log_probs        [B*T]
           10: masks            [B*T]
        """
        def _stack_flat(seq):
            return torch.stack(seq, dim=0).transpose(0, 1).flatten(0, 1)

        t_storage_features = _stack_flat(self.storage_features_seq)
        t_ws_features = _stack_flat(self.ws_features_seq)
        t_edge_feat = _stack_flat(self.edge_feat_seq)
        t_curws_idx = _stack_flat(self.curws_idx_seq)
        t_action_mask = _stack_flat(self.action_mask_seq)

        t_actions = _stack_flat(self.action_seq)
        t_rewards = _stack_flat(self.reward_seq)

        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_vals = self.t_old_val_seq.flatten(0, 1)

        t_dones = _stack_flat(self.done_seq)
        t_logprobs = _stack_flat(self.log_probs)
        t_masks = _stack_flat(self.mask_seq)

        return (t_storage_features, t_ws_features, t_edge_feat, t_curws_idx, t_action_mask,
                t_actions, t_rewards, t_vals, t_dones, t_logprobs, t_masks)

    def get_gae_advantages(self):
        """
        GAE (Generalized Advantage Estimation) 계산.

        Must be called AFTER transpose_data().

        Returns:
            advantages: [B*T] normalized per instance
            v_targets:  [B*T] value regression targets
        """
        reward_arr = torch.stack(self.reward_seq, dim=0)
        values = self.t_old_val_seq.transpose(0, 1)
        masks = torch.stack(self.mask_seq, dim=0).float()

        len_trajectory, len_envs = reward_arr.shape
        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []

        for i in reversed(range(len_trajectory)):
            if i == len_trajectory - 1:
                delta_t = reward_arr[i] - values[i]
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            advantage = delta_t + self.gamma * self.gae_lambda * advantage * masks[i]
            advantage_seq.insert(0, advantage)

        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)

        v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)

        t_advantage_seq = (
            (t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True))
            / (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)
        )

        return t_advantage_seq.flatten(0, 1), v_target_seq

    def clear_memory(self):
        """모든 저장된 시퀀스를 초기화한다."""
        del self.storage_features_seq[:]
        del self.ws_features_seq[:]
        del self.edge_feat_seq[:]
        del self.curws_idx_seq[:]
        del self.action_mask_seq[:]
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]
        del self.mask_seq[:]
        self.t_old_val_seq = None
