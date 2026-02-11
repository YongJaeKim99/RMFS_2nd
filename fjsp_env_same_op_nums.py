from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
import copy
from params import configs
import sys
import torch


@dataclass
class EnvState:
    """
        state definition
    """
    fea_act_tensor: torch.Tensor = None
    act_mask_tensor: torch.Tensor = None
    fea_team_tensor: torch.Tensor = None
    team_mask_tensor: torch.Tensor = None
    dynamic_pair_mask_tensor: torch.Tensor = None
    comp_idx_tensor: torch.Tensor = None
    candidate_tensor: torch.Tensor = None
    fea_pairs_tensor: torch.Tensor = None

    device = torch.device(configs.device)

    def update(self, fea_act, act_mask, fea_team, team_mask, dynamic_pair_mask,
               comp_idx, candidate, fea_pairs):
        """
            update the state information
        :param fea_act: input activity feature vectors with shape [sz_b, N, 10]
        :param act_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_team: input team feature vectors with shape [sz_b, T, 8]
        :param team_mask: used for masking attention coefficients (with shape [sz_b, T, T])
        :param comp_idx: a tensor with shape [sz_b, T, T, P] used for computing T_E
                    the value of comp_idx[i, k, q, p] (any i) means whether
                    team $T_k$ and $T_q$ are competing for candidate[i,p]
        :param dynamic_pair_mask: a tensor with shape [sz_b, P, T], used for masking
                            incompatible act-team pairs
        :param candidate: the index of candidate activities with shape [sz_b, P]
        :param fea_pairs: pair features with shape [sz_b, P, T, 8]
        :return:
        """
        device = self.device
        self.fea_act_tensor = torch.from_numpy(np.copy(fea_act)).float().to(device)
        self.fea_team_tensor = torch.from_numpy(np.copy(fea_team)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)

        self.act_mask_tensor = torch.from_numpy(np.copy(act_mask)).to(device)
        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.team_mask_tensor = torch.from_numpy(np.copy(team_mask)).float().to(device)
        self.comp_idx_tensor = torch.from_numpy(np.copy(comp_idx)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(device)

    def print_shape(self):
        print(self.fea_act_tensor.shape)
        print(self.act_mask_tensor.shape)
        print(self.candidate_tensor.shape)
        print(self.fea_team_tensor.shape)
        print(self.team_mask_tensor.shape)
        print(self.comp_idx_tensor.shape)
        print(self.dynamic_pair_mask_tensor.shape)
        print(self.fea_pairs_tensor.shape)


class FJSPEnvForSameActNums:
    """
        a batch of scheduling environments that have the same number of activities

        let E/N/P/T denote the number of envs/activities/projects/teams
        Remark: The index of activities has been rearranged in natural order
        eg. {A_{11}, A_{12}, A_{13}, A_{21}, A_{22}}  <--> {0,1,2,3,4}

        Attributes:

        proj_length: the number of activities in each project (shape [P])
        act_pt: the processing time matrix with shape [N, T],
                where act_pt[i,j] is the processing time of the ith activity
                on the jth team or 0 if $A_i$ can not be processed by $T_j$

        candidate: the index of candidates  [sz_b, P]
        fea_act: input activity feature vectors with shape [sz_b, N, 8]
        act_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        fea_team: input team feature vectors with shape [sz_b, T, 6]
        team_mask: used for masking attention coefficients (with shape [sz_b, T, T])
        comp_idx: a tensor with shape [sz_b, T, T, P] used for computing T_E
                    the value of comp_idx[i, k, q, p] (any i) means whether
                    team $T_k$ and $T_q$ are competing for candidate[i,p]
        dynamic_pair_mask: a tensor with shape [sz_b, P, T], used for masking incompatible act-team pairs
        fea_pairs: pair features with shape [sz_b, P, T, 8]
    """

    def __init__(self, n_p, n_t):
        """
        :param n_p: the number of projects
        :param n_t: the number of teams
        """
        self.number_of_projects = n_p
        self.number_of_teams = n_t
        self.old_state = EnvState()

        # the dimension of activity raw features
        self.act_fea_dim = 10
        # the dimension of team raw features
        self.team_fea_dim = 8

    def set_static_properties(self):
        """
            define static properties
        """
        self.multi_env_team_diag = np.tile(np.expand_dims(np.eye(self.number_of_teams, dtype=bool), axis=0),
                                          (self.number_of_envs, 1, 1))

        self.env_idxs = np.arange(self.number_of_envs)
        self.env_proj_idx = self.env_idxs.repeat(self.number_of_projects).reshape(self.number_of_envs, self.number_of_projects)
        self.act_idx = np.arange(self.number_of_acts)[np.newaxis, :]

    def set_initial_data(self, proj_length_list, act_pt_list):
        """
            initialize the data of the instances

        :param proj_length_list: the list of 'proj_length'
        :param act_pt_list: the list of 'act_pt'
        """

        self.number_of_envs = len(proj_length_list)
        self.proj_length = np.array(proj_length_list)
        self.act_pt = np.array(act_pt_list)
        self.number_of_acts = self.act_pt.shape[1]
        self.number_of_teams = act_pt_list[0].shape[1]
        self.number_of_projects = proj_length_list[0].shape[0]

        self.set_static_properties()

        # [E, N, T]
        self.pt_lower_bound = np.min(self.act_pt)
        self.pt_upper_bound = np.max(self.act_pt)
        self.true_act_pt = np.copy(self.act_pt)

        # normalize the processing time
        self.act_pt = (self.act_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)

        # bool 3-d array formulating the compatible relation with shape [E,N,T]
        self.process_relation = (self.act_pt != 0)
        self.reverse_process_relation = ~self.process_relation

        # number of compatible teams of each activity ([E,N])
        self.compatible_act = np.sum(self.process_relation, 2)
        # number of activities that each team can process ([E,T])
        self.compatible_team = np.sum(self.process_relation, 1)

        self.unmasked_act_pt = np.copy(self.act_pt)

        head_act_id = np.zeros((self.number_of_envs, 1))

        # the index of first activity of each project ([E,P])
        self.proj_first_act_id = np.concatenate([head_act_id, np.cumsum(self.proj_length, axis=1)[:, :-1]], axis=1).astype(
            'int')
        # the index of last activity of each project ([E,P])
        self.proj_last_act_id = self.proj_first_act_id + self.proj_length - 1

        self.initial_vars()

        self.init_act_mask()

        self.act_pt = ma.array(self.act_pt, mask=self.reverse_process_relation)

        """
            compute activity raw features
        """
        self.act_mean_pt = np.mean(self.act_pt, axis=2).data

        self.act_min_pt = np.min(self.act_pt, axis=-1).data
        self.act_max_pt = np.max(self.act_pt, axis=-1).data
        self.pt_span = self.act_max_pt - self.act_min_pt
        # [E, T]
        self.team_min_pt = np.max(self.act_pt, axis=1).data
        self.team_max_pt = np.max(self.act_pt, axis=1)

        # the estimated lower bound of complete time of activities
        self.act_ct_lb = copy.deepcopy(self.act_min_pt)
        for k in range(self.number_of_envs):
            for i in range(self.number_of_projects):
                self.act_ct_lb[k][self.proj_first_act_id[k][i]:self.proj_last_act_id[k][i] + 1] = np.cumsum(
                    self.act_ct_lb[k][self.proj_first_act_id[k][i]:self.proj_last_act_id[k][i] + 1])

        # project remaining number of activities
        self.act_match_proj_left_act_nums = np.array([np.repeat(self.proj_length[k],
                                                             repeats=self.proj_length[k])
                                                   for k in range(self.number_of_envs)])
        self.proj_remain_work = []
        for k in range(self.number_of_envs):
            self.proj_remain_work.append(
                [np.sum(self.act_mean_pt[k][self.proj_first_act_id[k][i]:self.proj_last_act_id[k][i] + 1])
                 for i in range(self.number_of_projects)])

        self.act_match_proj_remain_work = np.array([np.repeat(self.proj_remain_work[k], repeats=self.proj_length[k])
                                                  for k in range(self.number_of_envs)])

        self.construct_act_features()

        # shape reward
        self.init_quality = np.max(self.act_ct_lb, axis=1)

        self.max_endTime = self.init_quality
        """
            compute team raw features
        """
        self.team_available_act_nums = np.copy(self.compatible_team)
        self.team_current_available_act_nums = np.copy(self.compatible_team)
        # [E, P, T]
        self.candidate_pt = np.array([self.unmasked_act_pt[k][self.candidate[k]] for k in range(self.number_of_envs)])

        # construct dynamic pair mask : [E, P, T]
        self.dynamic_pair_mask = (self.candidate_pt == 0)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.team_current_available_pc_nums = np.sum(~self.candidate_process_relation, axis=1)

        self.team_mean_pt = np.mean(self.act_pt, axis=1).filled(0)
        # construct team features [E, T, 6]

        # construct 'comp_idx' : [E, T, T, P]
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.init_team_mask()
        self.construct_team_features()

        self.construct_pair_features()

        self.old_state.update(self.fea_act, self.act_mask,
                              self.fea_team, self.team_mask,
                              self.dynamic_pair_mask, self.comp_idx, self.candidate,
                              self.fea_pairs)

        # old record
        self.old_act_mask = np.copy(self.act_mask)
        self.old_team_mask = np.copy(self.team_mask)
        self.old_act_ct_lb = np.copy(self.act_ct_lb)
        self.old_act_match_proj_left_act_nums = np.copy(self.act_match_proj_left_act_nums)
        self.old_act_match_proj_remain_work = np.copy(self.act_match_proj_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_team_current_available_act_nums = np.copy(self.team_current_available_act_nums)
        self.old_team_current_available_pc_nums = np.copy(self.team_current_available_pc_nums)
        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def reset(self):
        """
           reset the environments
        :return: the state
        """
        self.initial_vars()

        # copy the old data
        self.act_mask = np.copy(self.old_act_mask)
        self.team_mask = np.copy(self.old_team_mask)
        self.act_ct_lb = np.copy(self.old_act_ct_lb)
        self.act_match_proj_left_act_nums = np.copy(self.old_act_match_proj_left_act_nums)
        self.act_match_proj_remain_work = np.copy(self.old_act_match_proj_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.team_current_available_act_nums = np.copy(self.old_team_current_available_act_nums)
        self.team_current_available_pc_nums = np.copy(self.old_team_current_available_pc_nums)
        # copy the old state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def initial_vars(self):
        """
            initialize variables for further use
        """
        self.step_count = 0
        # the array that records the makespan of all environments
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        # the complete time of activities ([E,N])
        self.act_ct = np.zeros((self.number_of_envs, self.number_of_acts))
        self.team_free_time = np.zeros((self.number_of_envs, self.number_of_teams))
        self.team_remain_work = np.zeros((self.number_of_envs, self.number_of_teams))

        self.team_waiting_time = np.zeros((self.number_of_envs, self.number_of_teams))
        self.team_working_flag = np.zeros((self.number_of_envs, self.number_of_teams))

        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_projects))

        self.true_act_ct = np.zeros((self.number_of_envs, self.number_of_acts))
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_projects))
        self.true_team_free_time = np.zeros((self.number_of_envs, self.number_of_teams))

        self.candidate = np.copy(self.proj_first_act_id)

        # mask[i,p] : whether the pth project of ith env is scheduled (have no unscheduled activities)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_projects), fill_value=0, dtype=bool)

        self.act_scheduled_flag = np.zeros((self.number_of_envs, self.number_of_acts))
        self.act_waiting_time = np.zeros((self.number_of_envs, self.number_of_acts))
        self.act_remain_work = np.zeros((self.number_of_envs, self.number_of_acts))

        self.act_available_team_nums = np.copy(self.compatible_act) / self.number_of_teams
        self.pair_free_time = np.zeros((self.number_of_envs, self.number_of_projects,
                                        self.number_of_teams))
        self.remain_process_relation = np.copy(self.process_relation)

        self.delete_mask_fea_act = np.full(shape=(self.number_of_envs, self.number_of_acts, self.act_fea_dim),
                                         fill_value=0, dtype=bool)
        # mask[i,j] : whether the jth activity of ith env is deleted (from the set $A_u$)
        self.deleted_act_nodes = np.full(shape=(self.number_of_envs, self.number_of_acts),
                                        fill_value=0, dtype=bool)

    def step(self, actions):
        """
            perform the state transition & return the next state and reward
        :param actions: the action list with shape [E]
        :return: the next state, reward and the done flag
        """
        chosen_proj = actions // self.number_of_teams
        chosen_team = actions % self.number_of_teams
        chosen_act = self.candidate[self.env_idxs, chosen_proj]

        if (self.reverse_process_relation[self.env_idxs, chosen_act, chosen_team]).any():
            print(
                f'Env Error from choosing action: Act {chosen_act} can\'t be processed by Team {chosen_team}')
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (chosen_act != self.proj_last_act_id[self.env_idxs, chosen_proj])
        self.candidate[self.env_idxs, chosen_proj] += candidate_add_flag
        self.mask[self.env_idxs, chosen_proj] = (1 - candidate_add_flag)

        # the start processing time of chosen activities
        chosen_act_st = np.maximum(self.candidate_free_time[self.env_idxs, chosen_proj],
                                  self.team_free_time[self.env_idxs, chosen_team])

        self.act_ct[self.env_idxs, chosen_act] = chosen_act_st + self.act_pt[
            self.env_idxs, chosen_act, chosen_team]
        self.candidate_free_time[self.env_idxs, chosen_proj] = self.act_ct[self.env_idxs, chosen_act]
        self.team_free_time[self.env_idxs, chosen_team] = self.act_ct[self.env_idxs, chosen_act]

        true_chosen_act_st = np.maximum(self.true_candidate_free_time[self.env_idxs, chosen_proj],
                                       self.true_team_free_time[self.env_idxs, chosen_team])
        self.true_act_ct[self.env_idxs, chosen_act] = true_chosen_act_st + self.true_act_pt[
            self.env_idxs, chosen_act, chosen_team]
        self.true_candidate_free_time[self.env_idxs, chosen_proj] = self.true_act_ct[
            self.env_idxs, chosen_act]
        self.true_team_free_time[self.env_idxs, chosen_team] = self.true_act_ct[
            self.env_idxs, chosen_act]

        self.current_makespan = np.maximum(self.current_makespan, self.true_act_ct[
            self.env_idxs, chosen_act])

        # update the candidate message
        mask_temp = candidate_add_flag
        self.candidate_pt[mask_temp, chosen_proj[mask_temp]] = self.unmasked_act_pt[mask_temp, chosen_act[mask_temp] + 1]
        self.candidate_process_relation[mask_temp, chosen_proj[mask_temp]] = \
            self.reverse_process_relation[mask_temp, chosen_act[mask_temp] + 1]
        self.candidate_process_relation[~mask_temp, chosen_proj[~mask_temp]] = 1

        # compute the next schedule time

        # [E, P, T]
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        teamFT_for_compare = np.expand_dims(self.team_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, teamFT_for_compare)

        schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)

        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data

        self.remain_process_relation[self.env_idxs, chosen_act] = 0
        self.act_scheduled_flag[self.env_idxs, chosen_act] = 1

        """
            update the mask for deleting nodes
        """
        self.deleted_act_nodes = \
            np.logical_and((self.act_ct <= self.next_schedule_time[:, np.newaxis]),
                           self.act_scheduled_flag)
        self.delete_mask_fea_act = np.tile(self.deleted_act_nodes[:, :, np.newaxis],
                                         (1, 1, self.act_fea_dim))

        """
            update the state
        """
        self.update_act_mask()

        # update activity raw features
        diff = self.act_ct[self.env_idxs, chosen_act] - self.act_ct_lb[self.env_idxs, chosen_act]

        mask1 = (self.act_idx >= chosen_act[:, np.newaxis]) & \
                (self.act_idx < (self.proj_last_act_id[self.env_idxs, chosen_proj] + 1)[:,
                               np.newaxis])
        self.act_ct_lb[mask1] += np.tile(diff[:, np.newaxis], (1, self.number_of_acts))[mask1]

        mask2 = (self.act_idx >= (self.proj_first_act_id[self.env_idxs, chosen_proj])[:,
                                np.newaxis]) & \
                (self.act_idx < (self.proj_last_act_id[self.env_idxs, chosen_proj] + 1)[:,
                               np.newaxis])
        self.act_match_proj_left_act_nums[mask2] -= 1
        self.act_match_proj_remain_work[mask2] -= \
            np.tile(self.act_mean_pt[self.env_idxs, chosen_act][:, np.newaxis], (1, self.number_of_acts))[mask2]

        self.act_waiting_time = np.zeros((self.number_of_envs, self.number_of_acts))
        self.act_waiting_time[self.env_proj_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                         - self.candidate_free_time, 0) + self.mask * self.act_waiting_time[
                self.env_proj_idx, self.candidate]

        self.act_remain_work = np.maximum(self.act_ct -
                                         np.expand_dims(self.next_schedule_time, axis=1), 0)

        self.construct_act_features()

        # update dynamic pair mask
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]

        self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

        # update comp_idx
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

        self.update_team_mask()

        # update team raw features
        self.team_current_available_pc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.team_current_available_act_nums -= self.process_relation[
            self.env_idxs, chosen_act]

        team_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.team_free_time
        team_free_flag = team_free_duration < 0
        self.team_working_flag = team_free_flag + 0
        self.team_waiting_time = (1 - team_free_flag) * team_free_duration

        self.team_remain_work = np.maximum(-team_free_duration, 0)

        self.construct_team_features()

        self.construct_pair_features()

        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        reward = self.max_endTime - np.max(self.act_ct_lb, axis=1)
        self.max_endTime = np.max(self.act_ct_lb, axis=1)

        # update the state
        self.state.update(self.fea_act, self.act_mask, self.fea_team, self.team_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs)

        return self.state, np.array(reward), self.done()

    def done(self):
        """
            compute the done flag
        """
        return np.ones(self.number_of_envs) * (self.step_count >= self.number_of_acts)

    def construct_act_features(self):
        """
            construct activity raw features
        """
        self.fea_act = np.stack((self.act_scheduled_flag,
                               self.act_ct_lb,
                               self.act_min_pt,
                               self.pt_span,
                               self.act_mean_pt,
                               self.act_waiting_time,
                               self.act_remain_work,
                               self.act_match_proj_left_act_nums,
                               self.act_match_proj_remain_work,
                               self.act_available_team_nums), axis=2)

        if self.step_count != self.number_of_acts:
            self.norm_act_features()

    def norm_act_features(self):
        """
            normalize activity raw features (across the second dimension)
        """
        self.fea_act[self.delete_mask_fea_act] = 0
        num_delete_nodes = np.count_nonzero(self.deleted_act_nodes, axis=1)
        num_delete_nodes = num_delete_nodes[:, np.newaxis]
        num_left_nodes = self.number_of_acts - num_delete_nodes
        mean_fea_act = np.sum(self.fea_act, axis=1) / num_left_nodes
        temp = np.where(self.delete_mask_fea_act, mean_fea_act[:, np.newaxis, :], self.fea_act)
        var_fea_act = np.var(temp, axis=1)

        std_fea_act = np.sqrt(var_fea_act * self.number_of_acts / num_left_nodes)

        self.fea_act = (temp - mean_fea_act[:, np.newaxis, :]) / \
                     (std_fea_act[:, np.newaxis, :] + 1e-8)

    def construct_team_features(self):
        """
            construct team raw features
        """
        self.fea_team = np.stack((self.team_current_available_pc_nums,
                               self.team_current_available_act_nums,
                               self.team_min_pt,
                               self.team_mean_pt,
                               self.team_waiting_time,
                               self.team_remain_work,
                               self.team_free_time,
                               self.team_working_flag), axis=2)

        if self.step_count != self.number_of_acts:
            self.norm_team_features()

    def norm_team_features(self):
        """
            normalize team raw features (across the second dimension)
        """
        self.fea_team[self.delete_mask_fea_team] = 0
        num_delete_teams = np.count_nonzero(self.delete_mask_fea_team[:, :, 0], axis=1)
        num_delete_teams = num_delete_teams[:, np.newaxis]
        num_left_teams = self.number_of_teams - num_delete_teams
        mean_fea_team = np.sum(self.fea_team, axis=1) / num_left_teams
        temp = np.where(self.delete_mask_fea_team,
                        mean_fea_team[:, np.newaxis, :], self.fea_team)
        var_fea_team = np.var(temp, axis=1)
        std_fea_team = np.sqrt(var_fea_team * self.number_of_teams / num_left_teams)

        self.fea_team = (temp - mean_fea_team[:, np.newaxis, :]) / \
                     (std_fea_team[:, np.newaxis, :] + 1e-8)

    def construct_pair_features(self):
        """
            construct pair features
        """
        remain_act_pt = ma.array(self.act_pt, mask=~self.remain_process_relation)

        chosen_act_max_pt = np.expand_dims(self.act_max_pt[self.env_proj_idx, self.candidate], axis=-1)

        max_remain_act_pt = np.max(np.max(remain_act_pt, axis=1, keepdims=True), axis=2, keepdims=True) \
            .filled(0 + 1e-8)

        team_max_remain_act_pt = np.max(remain_act_pt, axis=1, keepdims=True). \
            filled(0 + 1e-8)

        pair_max_pt = np.max(np.max(self.candidate_pt, axis=1, keepdims=True),
                             axis=2, keepdims=True) + 1e-8

        team_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8

        pair_wait_time = self.act_waiting_time[self.env_proj_idx, self.candidate][:, :,
                         np.newaxis] + self.team_waiting_time[:, np.newaxis, :]

        chosen_proj_remain_work = np.expand_dims(self.act_match_proj_remain_work
                                                [self.env_proj_idx, self.candidate],
                                                axis=-1) + 1e-8

        self.fea_pairs = np.stack((self.candidate_pt,
                                   self.candidate_pt / chosen_act_max_pt,
                                   self.candidate_pt / team_max_candidate_pt,
                                   self.candidate_pt / max_remain_act_pt,
                                   self.candidate_pt / team_max_remain_act_pt,
                                   self.candidate_pt / pair_max_pt,
                                   self.candidate_pt / chosen_proj_remain_work,
                                   pair_wait_time), axis=-1)

    def update_team_mask(self):
        """
            update 'team_mask'
        """
        self.team_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_team = np.tile(~(np.sum(self.team_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.team_fea_dim))
        self.team_mask[self.multi_env_team_diag] = 1

    def init_team_mask(self):
        """
            initialize 'team_mask'
        """
        self.team_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_team = np.tile(~(np.sum(self.team_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.team_fea_dim))
        self.team_mask[self.multi_env_team_diag] = 1

    def init_act_mask(self):
        """
            initialize 'act_mask'
        """
        self.act_mask = np.full(shape=(self.number_of_envs, self.number_of_acts, 3),
                               fill_value=0, dtype=np.float32)
        self.act_mask[self.env_proj_idx, self.proj_first_act_id, 0] = 1
        self.act_mask[self.env_proj_idx, self.proj_last_act_id, 2] = 1

    def update_act_mask(self):
        """
            update 'act_mask'
        """
        object_mask = np.zeros_like(self.act_mask)
        object_mask[:, :, 2] = self.deleted_act_nodes
        object_mask[:, 1:, 0] = self.deleted_act_nodes[:, :-1]
        self.act_mask = np.logical_or(object_mask, self.act_mask).astype(np.float32)

    def logic_operator(self, x, flagT=True):
        """
            a customized operator for computing some masks
        :param x: a 3-d array with shape [s,a,b]
        :param flagT: whether transpose x in the last two dimensions
        :return:  a 4-d array c, where c[i,j,k,l] = x[i,j,l] & x[i,k,l] for each i,j,k,l
        """
        if flagT:
            x = x.transpose(0, 2, 1)
        d1 = np.expand_dims(x, 2)
        d2 = np.expand_dims(x, 1)

        return np.logical_and(d1, d2).astype(np.float32)
