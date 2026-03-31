"""
GAT (Graph Attention Network) v1/v2/Relational Layers for RCMPSP Scheduling.
Dense adjacency 기반 구현 — 배치 내 다른 그래프 구조를 for문 없이 처리.

GATv1: e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j + a_edge·We_ij)
GATv2: e_ij = a^T · LeakyReLU(W_src·h_i + W_dst·h_j + W_edge·ef_ij + bias)
RelationalGATv2: edge type별 독립 GATv2 + concat → projection (논문 Eq.4-7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv1Layer(nn.Module):
    """
    GATv1 Multi-Head Attention Layer with Edge Features.

    논문: Velickovic et al., "Graph Attention Networks" (ICLR 2018)
    확장: edge feature를 attention score에 반영.
    """

    def __init__(self, d_hidden, d_edge, n_heads, dropout=0.0):
        """
        :param d_hidden: 노드 feature 차원 (n_heads * d_head)
        :param d_edge:   edge feature 차원 (raw, e.g. 10)
        :param n_heads:  attention head 수
        :param dropout:  attention dropout 확률
        """
        super().__init__()
        assert d_hidden % n_heads == 0, f"d_hidden({d_hidden}) must be divisible by n_heads({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        self.d_hidden = d_hidden
        self.dropout_p = dropout

        # Node feature transformation: (d_hidden → d_hidden)
        self.W = nn.Linear(d_hidden, d_hidden, bias=False)

        # Edge feature transformation: (d_edge → d_hidden)
        self.W_edge = nn.Linear(d_edge, d_hidden, bias=False)

        # Attention parameters per head: a_src, a_dst, a_edge ∈ R^{d_head}
        self.a_src = nn.Parameter(torch.empty(n_heads, self.d_head))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.d_head))
        self.a_edge = nn.Parameter(torch.empty(n_heads, self.d_head))

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.W_edge.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        nn.init.xavier_uniform_(self.a_edge, gain=1.414)

    def forward(self, h, adj, edge_feat):
        """
        :param h:         (B, V, d_hidden) 노드 features
        :param adj:       (B, V, V) boolean adjacency matrix
        :param edge_feat: (B, V, V, d_edge) edge features
        :return:          (B, V, d_hidden) 업데이트된 노드 features
        """
        B, V, _ = h.shape
        H, d = self.n_heads, self.d_head

        # Transform node features → (B, V, H, d)
        Wh = self.W(h).view(B, V, H, d)

        # Transform edge features → (B, V, V, H, d)
        We = self.W_edge(edge_feat).view(B, V, V, H, d)

        # GATv1 attention: e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j + a_edge·We_ij)
        score_src = (Wh * self.a_src).sum(dim=-1)           # (B, V, H)
        score_dst = (Wh * self.a_dst).sum(dim=-1)           # (B, V, H)
        score_edge = (We * self.a_edge).sum(dim=-1)         # (B, V, V, H)

        # Broadcast: (B,V,1,H) + (B,1,V,H) + (B,V,V,H)
        e = score_src.unsqueeze(2) + score_dst.unsqueeze(1) + score_edge
        e = F.leaky_relu(e, 0.2)

        # Mask non-edges → -inf
        e = e.masked_fill(~adj.unsqueeze(-1), float('-inf'))

        # Softmax over neighbors (dim=2)
        alpha = F.softmax(e, dim=2)                         # (B, V, V, H)
        alpha = alpha.nan_to_num(0.0)                       # all-masked rows → 0
        alpha = F.dropout(alpha, self.dropout_p, training=self.training)

        # Aggregate: h'_i = Σ_j α_ij · Wh_j
        h_prime = torch.einsum('bijh,bjhd->bihd', alpha, Wh)
        return h_prime.reshape(B, V, -1)                    # (B, V, d_hidden)


class GATv2Layer(nn.Module):
    """
    GATv2 Multi-Head Attention Layer with Edge Features.

    논문: Brody et al., "How Attentive are Graph Attention Networks?" (ICLR 2022)
    GATv2는 LeakyReLU를 먼저 적용하여 dynamic attention 문제를 해결.
    """

    def __init__(self, d_hidden, d_edge, n_heads, dropout=0.0):
        """
        :param d_hidden: 노드 feature 차원 (n_heads * d_head)
        :param d_edge:   edge feature 차원 (raw, e.g. 10)
        :param n_heads:  attention head 수
        :param dropout:  attention dropout 확률
        """
        super().__init__()
        assert d_hidden % n_heads == 0, f"d_hidden({d_hidden}) must be divisible by n_heads({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        self.d_hidden = d_hidden
        self.dropout_p = dropout

        # Separate transformations for source, destination, edge
        self.W_src = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_dst = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_edge = nn.Linear(d_edge, d_hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_hidden))

        # Value transformation (separate from attention key)
        self.W_val = nn.Linear(d_hidden, d_hidden, bias=False)

        # Attention vector per head
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

        # Separate transformations → (B, V, H, d)
        src = self.W_src(h).view(B, V, H, d)
        dst = self.W_dst(h).view(B, V, H, d)
        edge = self.W_edge(edge_feat).view(B, V, V, H, d)
        bias = self.bias.view(1, 1, 1, H, d)

        # GATv2: LeakyReLU FIRST, then attention projection
        # combined_ij = W_src·h_i + W_dst·h_j + W_edge·ef_ij + bias
        combined = src.unsqueeze(2) + dst.unsqueeze(1) + edge + bias  # (B, V, V, H, d)
        combined = F.leaky_relu(combined, 0.2)

        # Attention score: e_ij = a^T · combined_ij
        e = (combined * self.a).sum(dim=-1)                 # (B, V, V, H)

        # Mask non-edges → -inf
        e = e.masked_fill(~adj.unsqueeze(-1), float('-inf'))

        # Softmax over neighbors (dim=2)
        alpha = F.softmax(e, dim=2)                         # (B, V, V, H)
        alpha = alpha.nan_to_num(0.0)
        alpha = F.dropout(alpha, self.dropout_p, training=self.training)

        # Value transformation → (B, V, H, d)
        val = self.W_val(h).view(B, V, H, d)

        # Aggregate: h'_i = Σ_j α_ij · val_j
        h_prime = torch.einsum('bijh,bjhd->bihd', alpha, val)
        return h_prime.reshape(B, V, -1)                    # (B, V, d_hidden)


class RelationalGATv2Layer(nn.Module):
    """
    Relational GATv2 Layer: edge type별 독립 attention + concat → projection.

    논문 Eq.4-7:
      각 edge type k에 대해 독립적인 GATv2 attention 수행:
        h^k_i = GATv2_k(h, adj_k, edge_feat)
      결과를 concat 후 projection:
        h_new = W_combine · [h^1_i, h^2_i, ..., h^K_i]
    """

    def __init__(self, d_hidden, d_edge, n_heads, n_edge_types=4, dropout=0.0):
        """
        :param d_hidden:      노드 feature 차원
        :param d_edge:        edge feature 차원 (전체, e.g. 10)
        :param n_heads:       attention head 수 (각 type layer에 동일 적용)
        :param n_edge_types:  edge type 수 (default 4: prec_fwd, prec_bwd, mutex, eligible)
        :param dropout:       attention dropout 확률
        """
        super().__init__()
        self.n_edge_types = n_edge_types

        # K개의 독립 GATv2 sub-layer
        self.type_layers = nn.ModuleList([
            GATv2Layer(d_hidden, d_edge, n_heads, dropout)
            for _ in range(n_edge_types)
        ])

        # Concat(K * d_hidden) → d_hidden projection
        self.W_combine = nn.Linear(n_edge_types * d_hidden, d_hidden)
        nn.init.xavier_uniform_(self.W_combine.weight, gain=1.414)

    def forward(self, h, adj_per_type, edge_feat):
        """
        :param h:             (B, V, d_hidden) 노드 features
        :param adj_per_type:  list of K tensors, each (B, V, V) boolean adjacency
        :param edge_feat:     (B, V, V, d_edge) edge features (전체)
        :return:              (B, V, d_hidden) 업데이트된 노드 features
        """
        type_outputs = []
        for k, layer in enumerate(self.type_layers):
            h_k = layer(h, adj_per_type[k], edge_feat)        # (B, V, d_hidden)
            type_outputs.append(h_k)

        h_concat = torch.cat(type_outputs, dim=-1)             # (B, V, K*d_hidden)
        return self.W_combine(h_concat)                        # (B, V, d_hidden)
