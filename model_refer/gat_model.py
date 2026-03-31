"""
GAT Scheduling Model for RCMPSP.
Heterogeneous graph (Activity + Team nodes) with edge features.
GATv1/v2/Relational 선택 가능, DANIEL과 동일한 Actor-Critic 인터페이스 제공.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gat_layers import GATv1Layer, GATv2Layer, RelationalGATv2Layer
from model.sub_layers import Actor, Critic



class GATSchedulingModel(nn.Module):
    """
    Graph Attention Network for RCMPSP Scheduling.

    Heterogeneous graph:
      - Activity nodes (0..N-1): 16-dim features
      - Team nodes (N..N+T-1): 5-dim features
      - Edges: precedence, mutex, eligible (edge type encoded in edge features)

    Output: (pi, v) — DANIEL과 동일한 인터페이스.
    """

    def __init__(self, config):
        """
        :param config: SimpleNamespace with:
            fea_act_input_dim   : activity feature dim (16)
            fea_team_input_dim  : team feature dim (5)
            edge_feat_dim       : edge feature dim (7)
            d_hidden            : hidden dimension
            n_gat_layers        : number of GAT layers
            n_heads             : number of attention heads
            gat_version         : 'v1', 'v2', or 'relational'
            dropout_prob        : dropout probability
            n_edge_types        : (relational only) edge type 수 (default 4)
            num_mlp_layers_actor  : actor MLP layers
            hidden_dim_actor      : actor hidden dim
            num_mlp_layers_critic : critic MLP layers
            hidden_dim_critic     : critic hidden dim
        """
        super().__init__()

        d = config.d_hidden
        self.d_hidden = d
        self.edge_feat_dim = config.edge_feat_dim
        self.gat_version = config.gat_version
        self.n_edge_types = getattr(config, 'n_edge_types', 4)

        # ── Node Projection ──────────────────────────────────────
        # Activity (18-dim) → d_hidden,  Team (5-dim) → d_hidden
        self.act_proj = nn.Linear(config.fea_act_input_dim, d)
        self.team_proj = nn.Linear(config.fea_team_input_dim, d)
        # Node type embedding: 0=activity, 1=team (v1/v2 전용, relational은 불필요)
        if config.gat_version != 'relational':
            self.node_type_embed = nn.Embedding(2, d)

        # ── GAT Layers ───────────────────────────────────────────
        if config.gat_version == 'relational':
            self.gat_layers = nn.ModuleList([
                RelationalGATv2Layer(d, config.edge_feat_dim, config.n_heads,
                                     self.n_edge_types, config.dropout_prob)
                for _ in range(config.n_gat_layers)
            ])
        else:
            GATLayer = GATv1Layer if config.gat_version == 'v1' else GATv2Layer
            self.gat_layers = nn.ModuleList([
                GATLayer(d, config.edge_feat_dim, config.n_heads, config.dropout_prob)
                for _ in range(config.n_gat_layers)
            ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d) for _ in range(config.n_gat_layers)
        ])

        # ── Actor Head ───────────────────────────────────────────
        # Input: [h_act, h_team, h_global_act, h_global_team, pair_edge_feat]
        #  = 4*d + edge_feat_dim
        self.actor = Actor(
            config.num_mlp_layers_actor,
            4 * d + config.edge_feat_dim,
            config.hidden_dim_actor,
            1
        )

        # ── Critic Head ──────────────────────────────────────────
        # Input: [h_global_act, h_global_team] = 2*d
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * d,
            config.hidden_dim_critic,
            1
        )

    def forward(self, fea_act, fea_team, adj, edge_feat,
                dynamic_pair_mask, node_valid_mask,
                return_logits=False):
        """
        :param fea_act:           (B, N, 16)      activity features
        :param fea_team:          (B, T, 5)       team features
        :param adj:               (B, V, V)       boolean adjacency (V = N + T)
        :param edge_feat:         (B, V, V, F_e)  edge features
        :param dynamic_pair_mask: (B, N, T)       True = masked (invalid pair)
        :param node_valid_mask:   (B, V)          True = valid node
        :return:
            pi: scheduling policy  (B, N*T)
            v:  state value        (B, 1)
        """
        B, N, _ = fea_act.shape
        T = fea_team.shape[1]
        V = N + T
        device = fea_act.device

        # ── 1. Node Projection ───────────────────────────────────
        act_h = self.act_proj(fea_act)                              # (B, N, d)
        team_h = self.team_proj(fea_team)                           # (B, T, d)

        # Add type embedding (v1/v2만, relational은 edge type별 W로 구분)
        if self.gat_version != 'relational':
            act_type = torch.zeros(B, N, dtype=torch.long, device=device)
            team_type = torch.ones(B, T, dtype=torch.long, device=device)
            act_h = act_h + self.node_type_embed(act_type)
            team_h = team_h + self.node_type_embed(team_type)

        h = torch.cat([act_h, team_h], dim=1)                      # (B, V, d)

        # Zero out padded nodes
        valid_mask_f = node_valid_mask.unsqueeze(-1).float()        # (B, V, 1)
        h = h * valid_mask_f

        # ── 2. GAT Layers (residual + LayerNorm) ─────────────────
        if self.gat_version == 'relational':
            adj_per_type = self._decompose_adj_by_type(adj, edge_feat)
            for gat_layer, ln in zip(self.gat_layers, self.layer_norms):
                h_new = gat_layer(h, adj_per_type, edge_feat)
                h = ln(F.relu(h + h_new))                             # Eq.7: ReLU(residual) + LN
                h = h * valid_mask_f
        else:
            for gat_layer, ln in zip(self.gat_layers, self.layer_norms):
                h_new = gat_layer(h, adj, edge_feat)
                h = ln(h + h_new)                                      # residual + LN
                h = h * valid_mask_f                                    # re-zero padded

        # ── 3. Split back ────────────────────────────────────────
        h_act = h[:, :N, :]                                         # (B, N, d)
        h_team = h[:, N:, :]                                        # (B, T, d)

        # Global pooling (node_valid_mask 기반, 패딩 노드 제외)
        act_valid_mask = node_valid_mask[:, :N].unsqueeze(-1).float()   # (B, N, 1)
        h_global_act = (h_act * act_valid_mask).sum(dim=1) / act_valid_mask.sum(dim=1).clamp(min=1)  # (B, d)
        h_global_team = h_team.mean(dim=1)                              # (B, d) team은 전부 valid

        # ── 4. Actor ─────────────────────────────────────────────
        # Pair features: act→team 블록의 raw edge features
        pair_edge = edge_feat[:, :N, N:, :]                         # (B, N, T, F_e)

        # Expand node embeddings for all (N, T) pairs
        h_act_exp = h_act.unsqueeze(2).expand(-1, -1, T, -1)       # (B, N, T, d)
        h_team_exp = h_team.unsqueeze(1).expand(-1, N, -1, -1)     # (B, N, T, d)
        h_ga = h_global_act.unsqueeze(1).unsqueeze(2).expand(-1, N, T, -1)
        h_gt = h_global_team.unsqueeze(1).unsqueeze(2).expand(-1, N, T, -1)

        # Concatenate: [h_act, h_team, h_global_act, h_global_team, pair_edge]
        score_input = torch.cat([h_act_exp, h_team_exp, h_ga, h_gt, pair_edge], dim=-1)
        # shape: (B, N, T, 4*d + F_e)

        scores = self.actor(score_input).squeeze(-1)                # (B, N, T)
        scores[dynamic_pair_mask] = float('-inf')
        pi = F.softmax(scores.reshape(B, -1), dim=1)               # (B, N*T)

        # NaN safety: all-masked → uniform fallback
        if torch.isnan(pi).any():
            uniform = torch.full_like(pi, 1.0 / pi.shape[1])
            pi = torch.where(torch.isnan(pi), uniform, pi)

        # ── 5. Critic ────────────────────────────────────────────
        global_feat = torch.cat([h_global_act, h_global_team], dim=-1)  # (B, 2*d)
        v = self.critic(global_feat)                                # (B, 1)

        if return_logits:
            return pi, v, scores.reshape(B, -1)
        return pi, v

    def _decompose_adj_by_type(self, adj, edge_feat):
        """
        통합 adjacency를 edge type별로 분리.

        edge_feat[:,:,:,0:n_edge_types]의 binary flag 사용:
          [0] prec_fwd, [1] prec_bwd, [2] mutex, [3] eligible, [4] same_project

        Self-loop(대각원소)는 모든 type adjacency에 추가.

        :param adj:       (B, V, V) boolean
        :param edge_feat: (B, V, V, F_e)
        :return: list of n_edge_types tensors, each (B, V, V) boolean
        """
        type_flags = edge_feat[:, :, :, :self.n_edge_types] > 0.5  # (B, V, V, K)

        # Self-loop mask: adj 대각원소
        B, V, _ = adj.shape
        idx = torch.arange(V, device=adj.device)
        diag_mask = torch.zeros(B, V, V, dtype=torch.bool, device=adj.device)
        diag_mask[:, idx, idx] = adj[:, idx, idx]

        adj_per_type = []
        for k in range(self.n_edge_types):
            adj_k = (adj & type_flags[:, :, :, k]) | diag_mask
            adj_per_type.append(adj_k)

        return adj_per_type
