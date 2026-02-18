"""
DANIEL (Dual Attention Network for Integrated Scheduling) 모델
원본: FJSP-DRL-main/model/main_model.py
RCMPSP 프로젝트에 맞게 이식 (common_utils 의존성 제거)
"""

from model.attention_layer import *
from model.sub_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonzero_averaging(x):
    """
    비영(non-zero) 벡터들의 평균을 계산
    (삭제된 노드는 zero 벡터로 표현됨)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)


class DualAttentionNetwork(nn.Module):
    def __init__(self, config):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
            config.fea_j_input_dim  : activity feature input dim
            config.fea_m_input_dim  : team feature input dim
            config.num_heads_OAB    : list of head counts for Activity Attention Blocks
            config.num_heads_MAB    : list of head counts for Team Attention Blocks
            config.layer_fea_output_dim : list of output dims per DAN layer
            config.dropout_prob     : dropout probability
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim
        self.fea_m_input_dim = config.fea_m_input_dim
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        mid_dim = self.output_dim_per_layer[:-1]

        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim
        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i] * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, pred_idx, succ_idx):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: input activity feature vectors with shape [sz_b, N, fea_j_input_dim]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input team feature vectors with shape [sz_b, M, fea_m_input_dim]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
        :param pred_idx: predecessor indices [sz_b, N, max_preds], -1 padded
        :param succ_idx: successor indices   [sz_b, N, max_succs], -1 padded
        :return:
            fea_j.shape = [sz_b, N, output_dim]
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
        """
        sz_b, M, _, J = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, J)

        for layer in range(self.num_dan_layers):
            candidate_idx = candidate.unsqueeze(-1). \
                repeat(1, 1, fea_j.shape[-1]).type(torch.int64)

            # fea_j_jc: candidate features with shape [sz_b, N, J]
            fea_j_jc = torch.gather(fea_j, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul,
                                     fea_j_jc).reshape(sz_b, M, M, -1)
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask, pred_idx, succ_idx)
            fea_m = self.mch_attention_blocks[layer](fea_m, mch_mask, comp_val_layer)

        fea_j_global = nonzero_averaging(fea_j)
        fea_m_global = nonzero_averaging(fea_m)

        return fea_j, fea_m, fea_j_global, fea_m_global


class DANIEL(nn.Module):
    def __init__(self, config):
        """
            Dual Attention Network for Integrated Scheduling (RCMPSP 버전)
        :param config: a package of parameters (SimpleNamespace)
            config.fea_j_input_dim      : activity feature input dim
            config.fea_m_input_dim      : team feature input dim
            config.num_heads_OAB        : Activity Attention Block head counts (list)
            config.num_heads_MAB        : Team Attention Block head counts (list)
            config.layer_fea_output_dim : output dims per DAN layer (list)
            config.dropout_prob         : dropout probability
            config.num_mlp_layers_actor : Actor MLP layer count
            config.hidden_dim_actor     : Actor hidden dim
            config.num_mlp_layers_critic: Critic MLP layer count
            config.hidden_dim_critic    : Critic hidden dim
            config.device               : torch device string
        """
        super(DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim (fixed: 8-dim pair features from env)
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(device)
        self.actor = Actor(
            config.num_mlp_layers_actor,
            4 * self.embedding_output_dim + self.pair_input_dim,
            config.hidden_dim_actor,
            1
        ).to(device)
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * self.embedding_output_dim,
            config.hidden_dim_critic,
            1
        ).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                dynamic_pair_mask, fea_pairs, pred_idx, succ_idx):
        """
        :param fea_j: activity feature vectors [sz_b, N, fea_j_input_dim]
        :param op_mask: activity attention mask [sz_b, N, 3]
        :param candidate: candidate activity index per project [sz_b, J]
        :param fea_m: team feature vectors [sz_b, M, fea_m_input_dim]
        :param mch_mask: team attention mask [sz_b, M, M]
        :param comp_idx: competition index [sz_b, M, M, J]
        :param dynamic_pair_mask: incompatible pair mask [sz_b, J, M]  (True = masked)
        :param fea_pairs: pair feature vectors [sz_b, J, M, 8]
        :param pred_idx: predecessor indices [sz_b, N, max_preds], -1 padded
        :param succ_idx: successor indices   [sz_b, N, max_succs], -1 padded
        :return:
            pi: scheduling policy [sz_b, J*M]
            v:  state value [sz_b, 1]
        """
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(
            fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, pred_idx, succ_idx
        )
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # Gather candidate activity embeddings
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d).type(torch.int64)
        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        # Serialize: [sz_b, J*M, d]
        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)

        # candidate_feature: [sz_b, J*M, 4*output_dim + 8]
        candidate_feature = torch.cat(
            (Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input, Fea_Gm_input, fea_pairs),
            dim=-1
        )

        candidate_scores = self.actor(candidate_feature).squeeze(-1)  # [sz_b, J*M]

        # Mask incompatible (activity, team) pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)

        # Safety: all-masked → NaN → uniform fallback (non-inplace for autograd)
        if torch.isnan(pi).any():
            uniform = torch.full_like(pi, 1.0 / pi.shape[1])
            pi = torch.where(torch.isnan(pi), uniform, pi)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)

        return pi, v
