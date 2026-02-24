"""
PPO utilities for RCMPSP training.
Adapted from FJSP-DRL-main/model/PPO.py and FJSP-DRL-main/common_utils.py
with RCMPSP field names.
"""
import torch
from torch.distributions import Categorical
import numpy as np


def eval_actions(pi, actions):
    """
    Compute log probability and entropy for given actions.
    Equivalent to FJSP common_utils.eval_actions().

    :param pi: policy probability distribution [sz_b, N*T] (already softmaxed)
    :param actions: action indices [sz_b]
    :return: (log_probs [sz_b], entropy scalar)
    """
    dist = Categorical(pi.squeeze())
    log_probs = dist.log_prob(actions).reshape(-1)
    entropy = dist.entropy().mean()
    return log_probs, entropy


class PPOMemory:
    """
    Trajectory memory for PPO training on RCMPSP.

    Stores per-step state (EnvState fields) and transition data
    (action, log_prob, value, reward, done) for a full episode.

    After the rollout, transpose_data() flattens everything to
    [B*N*T, ...] for mini-batch PPO updates, and get_gae_advantages()
    computes GAE advantages + value targets.
    """

    def __init__(self, gamma: float, gae_lambda: float):
        """
        :param gamma: discount factor (논문: γ=1)
        :param gae_lambda: GAE λ parameter (논문: λ=0.98)
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # State tensors (one per step, shape [B, ...])
        self.fea_act_seq = []           # [T_steps, tensor[B, N, 14]]
        self.act_mask_seq = []          # [T_steps, tensor[B, N, 3 or 4]]
        self.fea_team_seq = []          # [T_steps, tensor[B, T_teams, 8]]
        self.team_mask_seq = []         # [T_steps, tensor[B, T_teams, T_teams]]
        self.dynamic_pair_mask_seq = [] # [T_steps, tensor[B, N, T_teams]]
        self.comp_idx_seq = []          # [T_steps, tensor[B, T_teams, T_teams, N]]
        self.candidate_seq = []         # [T_steps, tensor[B, N]]
        self.fea_pairs_seq = []         # [T_steps, tensor[B, N, T_teams, 8]]
        self.pred_idx_seq = []          # [T_steps, tensor[B, N, max_preds]]
        self.succ_idx_seq = []          # [T_steps, tensor[B, N, max_succs]]
        self.mutex_idx_seq = []         # [T_steps, tensor[B, N, max_mutex] or None]

        # Transition data (one per step)
        self.action_seq = []   # [T, tensor[B*P]]
        self.reward_seq = []   # [T, tensor[B*P]]
        self.val_seq = []      # [T, tensor[B*P]]
        self.done_seq = []     # [T, tensor[B*P]]
        self.log_probs = []    # [T, tensor[B*P]]

        # Filled by transpose_data(), used in get_gae_advantages()
        self.t_old_val_seq = None  # [B*P, T]

    # ------------------------------------------------------------------
    # Push methods
    # ------------------------------------------------------------------

    def push_state(self, state):
        """Append all EnvState tensors for the current step."""
        self.fea_act_seq.append(state.fea_act_tensor)
        self.act_mask_seq.append(state.act_mask_tensor)
        self.fea_team_seq.append(state.fea_team_tensor)
        self.team_mask_seq.append(state.team_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)
        self.comp_idx_seq.append(state.comp_idx_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)
        self.pred_idx_seq.append(state.pred_idx_tensor)
        self.succ_idx_seq.append(state.succ_idx_tensor)
        self.mutex_idx_seq.append(state.mutex_idx_tensor)  # None if disabled

    def push_transition(self, action, log_prob, val, reward, done):
        """
        Append transition data for the current step.

        :param action:   sampled action index  [B]
        :param log_prob: log π(a|s)            [B]
        :param val:      critic value V(s)     [B]
        :param reward:   step reward           [B]
        :param done:     done flag             [B] bool tensor
        """
        self.action_seq.append(action)
        self.log_probs.append(log_prob)
        self.val_seq.append(val)
        self.reward_seq.append(reward)
        self.done_seq.append(done)

    # ------------------------------------------------------------------
    # Flatten for mini-batch updates
    # ------------------------------------------------------------------

    def transpose_data(self):
        """
        Stack and flatten trajectory to [B*T_steps, ...] format.

        Layout of returned tuple (index → content):
          0  fea_act      [B*T_steps, N, 14]
          1  act_mask     [B*T_steps, N, 3 or 4]
          2  fea_team     [B*T_steps, T_teams, 8]
          3  team_mask    [B*T_steps, T_teams, T_teams]
          4  dyn_pair_msk [B*T_steps, N, T_teams]
          5  comp_idx     [B*T_steps, T_teams, T_teams, N]
          6  candidate    [B*T_steps, N]
          7  fea_pairs    [B*T_steps, N, T_teams, 8]
          8  pred_idx     [B*T_steps, N, max_preds]
          9  succ_idx     [B*T_steps, N, max_succs]
         10  mutex_idx    [B*T_steps, N, max_mutex] or None
         11  action       [B*T_steps]
         12  reward       [B*T_steps]
         13  val          [B*T_steps]
         14  done         [B*T_steps]
         15  log_probs    [B*T_steps]
        """
        def _stack_flat(seq):
            # [T, B*P, ...] → [B*P, T, ...] → [B*P*T, ...]
            return torch.stack(seq, dim=0).transpose(0, 1).flatten(0, 1)

        t_fea_act      = _stack_flat(self.fea_act_seq)
        t_act_mask     = _stack_flat(self.act_mask_seq)
        t_fea_team     = _stack_flat(self.fea_team_seq)
        t_team_mask    = _stack_flat(self.team_mask_seq)
        t_dyn_pmask    = _stack_flat(self.dynamic_pair_mask_seq)
        t_comp_idx     = _stack_flat(self.comp_idx_seq)
        t_candidate    = _stack_flat(self.candidate_seq)
        t_fea_pairs    = _stack_flat(self.fea_pairs_seq)
        t_pred_idx     = _stack_flat(self.pred_idx_seq)
        t_succ_idx     = _stack_flat(self.succ_idx_seq)

        # mutex_idx: None if disabled, tensor if enabled
        if self.mutex_idx_seq[0] is not None:
            t_mutex_idx = _stack_flat(self.mutex_idx_seq)
        else:
            t_mutex_idx = None

        t_action       = _stack_flat(self.action_seq)
        t_reward       = _stack_flat(self.reward_seq)

        # val_seq: store [B*P, T] for GAE, then flatten
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)  # [B*P, T]
        t_val = self.t_old_val_seq.flatten(0, 1)   # [B*P*T]

        t_done     = _stack_flat(self.done_seq)
        t_logprobs = _stack_flat(self.log_probs)

        return (t_fea_act, t_act_mask, t_fea_team, t_team_mask,
                t_dyn_pmask, t_comp_idx, t_candidate, t_fea_pairs,
                t_pred_idx, t_succ_idx, t_mutex_idx,
                t_action, t_reward, t_val, t_done, t_logprobs)

    # ------------------------------------------------------------------
    # GAE Advantage Estimation
    # ------------------------------------------------------------------

    def get_gae_advantages(self):
        """
        Compute Generalized Advantage Estimation (GAE).

        Must be called AFTER transpose_data() (which sets t_old_val_seq).

        Returns:
            advantages: [B*T_steps]  — normalized per env-instance
            v_targets:  [B*T_steps]  — value regression targets
        """
        # reward_arr: [T, B*P],  values: [T, B*P]
        reward_arr = torch.stack(self.reward_seq, dim=0)          # [T, B*P]
        values     = self.t_old_val_seq.transpose(0, 1)           # [T, B*P]
        len_trajectory, len_envs = reward_arr.shape

        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []

        for i in reversed(range(len_trajectory)):
            if i == len_trajectory - 1:
                # Terminal step: no next state
                delta_t = reward_arr[i] - values[i]
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1] - values[i]
            advantage = delta_t + self.gamma * self.gae_lambda * advantage
            advantage_seq.insert(0, advantage)

        # t_advantage_seq: [B*P, T]
        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)

        # Value targets = advantage + old_value
        v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)  # [B*P*T]

        # Normalize per env-instance (dim=1 = time axis)
        t_advantage_seq = (
            (t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True))
            / (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)
        )

        return t_advantage_seq.flatten(0, 1), v_target_seq  # both [B*P*T]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_memory(self):
        """Clear all stored sequences."""
        del self.fea_act_seq[:]
        del self.act_mask_seq[:]
        del self.fea_team_seq[:]
        del self.team_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.comp_idx_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]
        del self.pred_idx_seq[:]
        del self.succ_idx_seq[:]
        del self.mutex_idx_seq[:]
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]
        self.t_old_val_seq = None
