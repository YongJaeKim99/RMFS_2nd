from common_utils import nonzero_averaging
from model.attention_layer import *
from model.sub_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttentionNetwork(nn.Module):
    def __init__(self, config):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_act_input_dim = config.fea_act_input_dim
        self.fea_team_input_dim = config.fea_team_input_dim
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_AAB = config.num_heads_AAB
        self.num_heads_TAB = config.num_heads_TAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_AAB)
        assert len(config.num_heads_TAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_AAB_per_layer = [1] + self.num_heads_AAB
        num_heads_TAB_per_layer = [1] + self.num_heads_TAB

        mid_dim = self.output_dim_per_layer[:-1]

        act_input_dim_per_layer = [self.fea_act_input_dim] + mid_dim

        team_input_dim_per_layer = [self.fea_team_input_dim] + mid_dim

        self.act_attention_blocks = torch.nn.ModuleList()
        self.team_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.act_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_AAB_per_layer[i] * act_input_dim_per_layer[i],
                    num_heads=self.num_heads_AAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        for i in range(self.num_dan_layers):
            self.team_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_TAB_per_layer[i] * team_input_dim_per_layer[i],
                    edge_input_dim=num_heads_AAB_per_layer[i] * act_input_dim_per_layer[i],
                    num_heads=self.num_heads_TAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

    def forward(self, fea_act, act_mask, candidate, fea_team, team_mask, comp_idx):
        """
        :param candidate: the index of candidates  [sz_b, P]
        :param fea_act: input activity feature vectors with shape [sz_b, N, 8]
        :param act_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_team: input team feature vectors with shape [sz_b, T, 6]
        :param team_mask: used for masking attention coefficients (with shape [sz_b, T, T])
        :param comp_idx: a tensor with shape [sz_b, T, T, P] used for computing T_E
                    the value of comp_idx[i, k, q, p] (any i) means whether
                    team $T_k$ and $T_q$ are competing for candidate[i,p]
        :return:
            fea_act.shape = [sz_b, N, output_dim]
            fea_team.shape = [sz_b, T, output_dim]
            fea_act_global.shape = [sz_b, output_dim]
            fea_team_global.shape = [sz_b, output_dim]
        """
        sz_b, T, _, P = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, P)

        for layer in range(self.num_dan_layers):
            candidate_idx = candidate.unsqueeze(-1). \
                repeat(1, 1, fea_act.shape[-1]).type(torch.int64)

            # fea_act_pc: candidate features with shape [sz_b, N, P]
            fea_act_pc = torch.gather(fea_act, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul,
                                     fea_act_pc).reshape(sz_b, T, T, -1)
            fea_act = self.act_attention_blocks[layer](fea_act, act_mask)
            fea_team = self.team_attention_blocks[layer](fea_team, team_mask, comp_val_layer)

        fea_act_global = nonzero_averaging(fea_act)
        fea_team_global = nonzero_averaging(fea_team)

        return fea_act, fea_team, fea_act_global, fea_team_global


class DANIEL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for scheduling
        :param config: a package of parameters
        """
        super(DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(
            device)
        self.actor = Actor(config.num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
                           config.hidden_dim_actor, 1).to(device)
        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)

    def forward(self, fea_act, act_mask, candidate, fea_team, team_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        :param candidate: the index of candidate activities with shape [sz_b, P]
        :param fea_act: input activity feature vectors with shape [sz_b, N, 8]
        :param act_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_team: input team feature vectors with shape [sz_b, T, 6]
        :param team_mask: used for masking attention coefficients (with shape [sz_b, T, T])
        :param comp_idx: a tensor with shape [sz_b, T, T, P] used for computing T_E
                    the value of comp_idx[i, k, q, p] (any i) means whether
                    team $T_k$ and $T_q$ are competing for candidate[i,p]
        :param dynamic_pair_mask: a tensor with shape [sz_b, P, T], used for masking
                            incompatible act-team pairs
        :param fea_pairs: pair features with shape [sz_b, P, T, 8]
        :return:
            pi: scheduling policy with shape [sz_b, P*T]
            v: the value of state with shape [sz_b, 1]
        """

        fea_act, fea_team, fea_act_global, fea_team_global = self.feature_exact(fea_act, act_mask, candidate, fea_team,
                                                                                team_mask, comp_idx)
        sz_b, T, _, P = comp_idx.size()
        d = fea_act.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_act_PC = torch.gather(fea_act, 1, candidate_idx)

        Fea_act_PC_serialized = Fea_act_PC.unsqueeze(2).repeat(1, 1, T, 1).reshape(sz_b, T * P, d)
        Fea_team_serialized = fea_team.unsqueeze(1).repeat(1, P, 1, 1).reshape(sz_b, T * P, d)

        Fea_Gact_input = fea_act_global.unsqueeze(1).expand_as(Fea_act_PC_serialized)
        Fea_Gteam_input = fea_team_global.unsqueeze(1).expand_as(Fea_act_PC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, P*T, 4*output_dim + 8]
        candidate_feature = torch.cat((Fea_act_PC_serialized, Fea_team_serialized, Fea_Gact_input,
                                       Fea_Gteam_input, fea_pairs), dim=-1)

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible act-team pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_act_global, fea_team_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v
