import json
import random

from torch.distributions.categorical import Categorical
import sys
import numpy as np
import torch
import copy

"""
    agent utils
"""


def sample_action(p):
    """
        sample an action by the distribution p
    :param p: this distribution with the probability of choosing each action
    :return: an action sampled by p
    """
    dist = Categorical(p)
    s = dist.sample()  # index
    return s, dist.log_prob(s)


def eval_actions(p, actions):
    """
    :param p: the policy
    :param actions: action sequences
    :return: the log probability of actions and the entropy of p
    """
    softmax_dist = Categorical(p.squeeze())
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p):
    _, index = torch.max(p, dim=1)
    return index


def min_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the minimum element of the array
    """
    min_element = np.min(array)
    candidate = np.where(array == min_element)
    return candidate


def max_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the maximum element of the array
    """
    max_element = np.max(array)
    candidate = np.where(array == max_element)
    return candidate


def available_team_list_for_proj(chosen_proj, env):
    """
    :param chosen_proj: the selected project
    :param env: the scheduling environment
    :return: the teams which can immediately process the chosen project
    """
    team_state = ~env.candidate_process_relation[0, chosen_proj]
    available_team_list = np.where(team_state == True)[0]
    team_free_time = env.team_free_time[0][available_team_list]
    proj_free_time = env.candidate_free_time[0][chosen_proj]
    # case1 eg:
    # ProjF: 50
    # TeamF: 55 60 65 70
    if (proj_free_time < team_free_time).all():
        chosen_team_list = available_team_list[min_element_index(team_free_time)]
    # case2 eg:
    # ProjF: 50
    # TeamF: 35 40 55 60
    else:
        chosen_team_list = available_team_list[np.where(team_free_time <= proj_free_time)]

    return chosen_team_list


def heuristic_select_action(method, env):
    """
    :param method: the name of heuristic method
    :param env: the environment
    :return: the action selected by the heuristic method

    here are heuristic methods selected for comparison:

    FIFO: First in first out
    MAR(or MOANR): Most activities remaining
    SPT: Shortest processing time
    MWKR: Most work remaining
    """
    chosen_proj = -1
    chosen_team = -1

    proj_state = (env.mask[0] == 0)

    process_proj_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
    proj_state = process_proj_state & proj_state

    available_projs = np.where(proj_state == True)[0]
    available_acts = env.candidate[0][available_projs]

    if method == 'FIFO':
        # selecting the earliest ready candidate activity
        candidate_free_time = env.candidate_free_time[0][available_projs]
        chosen_proj_list = available_projs[min_element_index(candidate_free_time)]
        chosen_proj = np.random.choice(chosen_proj_list)

        # select the earliest ready team
        team_state = ~env.candidate_process_relation[0, chosen_proj]
        available_teams = np.where(team_state == True)[0]
        team_free_time = env.team_free_time[0][available_teams]
        chosen_team_list = available_teams[min_element_index(team_free_time)]
        chosen_team = np.random.choice(chosen_team_list)

    elif method == 'MAR':
        remain_acts = env.act_match_proj_left_act_nums[0][available_acts]
        chosen_proj_list = available_projs[max_element_index(remain_acts)]
        chosen_proj = np.random.choice(chosen_proj_list)

        # select a team which can immediately process the chosen project
        chosen_team_list = available_team_list_for_proj(chosen_proj, env)
        chosen_team = np.random.choice(chosen_team_list)

    elif method == 'SPT':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")
        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.min(pt_list))[0]

        action = np.random.choice(action_list)
        return action

    elif method == 'MWKR':
        proj_remain_work_list = env.act_match_proj_remain_work[0][available_acts]

        chosen_proj = available_projs[np.random.choice(max_element_index(proj_remain_work_list)[0])]

        # select a team which can immediately process the chosen project
        chosen_team_list = available_team_list_for_proj(chosen_proj, env)
        chosen_team = np.random.choice(chosen_team_list)

    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()

    if chosen_proj == -1 or chosen_team == -1:
        print(f'Error From choosing action: choose project {chosen_proj}, team {chosen_team}')
        sys.exit()

    action = chosen_proj * env.number_of_teams + chosen_team
    return action


"""
    common utils
"""


def save_default_params(config):
    """
        save parameters in the config
    :param config: a package of parameters
    :return:
    """
    with open('./config_default.json', 'wt') as f:
        json.dump(vars(config), f, indent=4)
    print("successfully save default params")


def nonzero_averaging(x):
    """
        remove zero vectors and then compute the mean of x
        (The deleted nodes are represented by zero vectors)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)


def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('123')
