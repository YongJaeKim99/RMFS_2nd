"""
GATv2 Layer for RMFS.
Dense adjacency-based implementation.
Reference: Brody et al., "How Attentive are Graph Attention Networks?" (ICLR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """
    GATv2 Multi-Head Attention Layer with Edge Features.
    LeakyReLU를 먼저 적용하여 dynamic attention 문제를 해결.
    """

    def __init__(self, d_hidden, d_edge, n_heads, dropout=0.0):
        """
        :param d_hidden: 노드 feature 차원 (n_heads * d_head)
        :param d_edge:   edge feature 차원
        :param n_heads:  attention head 수
        :param dropout:  attention dropout 확률
        """
        super().__init__()
        assert d_hidden % n_heads == 0, f"d_hidden({d_hidden}) must be divisible by n_heads({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        self.d_hidden = d_hidden
        self.dropout_p = dropout

        self.W_src = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_dst = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_edge = nn.Linear(d_edge, d_hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_hidden))

        self.W_val = nn.Linear(d_hidden, d_hidden, bias=False)

        self.a = nn.Parameter(torch.empty(n_heads, self.d_head))

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.W_src.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W_dst.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W_edge.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W_val.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, h, adj, edge_feat):
        """
        :param h:         (B, V, d_hidden) 노드 features
        :param adj:       (B, V, V) boolean adjacency matrix
        :param edge_feat: (B, V, V, d_edge) edge features
        :return:          (B, V, d_hidden) 업데이트된 노드 features
        """
        B, V, _ = h.shape
        H, d = self.n_heads, self.d_head

        src = self.W_src(h).view(B, V, H, d)
        dst = self.W_dst(h).view(B, V, H, d)
        edge = self.W_edge(edge_feat).view(B, V, V, H, d)
        bias = self.bias.view(1, 1, 1, H, d)

        # GATv2: LeakyReLU FIRST, then attention projection
        combined = src.unsqueeze(2) + dst.unsqueeze(1) + edge + bias
        combined = F.leaky_relu(combined, 0.2)

        e = (combined * self.a).sum(dim=-1)          # (B, V, V, H)
        e = e.masked_fill(~adj.unsqueeze(-1), float('-inf'))

        alpha = F.softmax(e, dim=2)                  # (B, V, V, H)
        alpha = alpha.nan_to_num(0.0)
        alpha = F.dropout(alpha, self.dropout_p, training=self.training)

        val = self.W_val(h).view(B, V, H, d)
        h_prime = torch.einsum('bijh,bjhd->bihd', alpha, val)
        return h_prime.reshape(B, V, -1)             # (B, V, d_hidden)
