"""
RMFS GATv2 Actor-Critic Model.

Heterogeneous graph (Storage + WS nodes) with edge features.
GATv2 layers로 node embedding을 학습하고,
Actor는 per-storage scoring + Stay scoring,
Critic는 global pooling으로 state value를 추정.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gat_layers import GATv2Layer
from model.sub_layers import Actor, Critic


class GATActorCritic(nn.Module):
    """
    GATv2 Actor-Critic for RMFS.

    Input:  RMFSState (graph) + adj
    Output: pi (B, N_S+1), v (B, 1)
    """

    def __init__(self, config):
        """
        Args:
            config: SimpleNamespace with:
                N_S, N_W: int
                storage_feat_dim, ws_feat_dim: int (default 4)
                d_edge: int (default 9)
                d_hidden: int
                n_gat_layers: int
                n_heads: int
                dropout_prob: float
                num_mlp_layers_actor, hidden_dim_actor: int
                num_mlp_layers_critic, hidden_dim_critic: int
        """
        super().__init__()

        d = config.d_hidden
        self.d_hidden = d
        self.N_S = config.N_S
        self.N_W = config.N_W
        self.V = self.N_S + self.N_W

        # Node projections
        self.storage_proj = nn.Linear(config.storage_feat_dim, d)
        self.ws_proj = nn.Linear(config.ws_feat_dim, d)
        self.node_type_embed = nn.Embedding(2, d)  # 0=storage, 1=ws

        # GATv2 layers (residual + LayerNorm)
        self.gat_layers = nn.ModuleList([
            GATv2Layer(d, config.d_edge, config.n_heads, config.dropout_prob)
            for _ in range(config.n_gat_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d) for _ in range(config.n_gat_layers)
        ])

        # Actor: per-storage scores
        # Input: [h_s, h_curws, h_global_s, h_global_w, edge_curws_to_s]
        # = 4*d + d_edge
        self.actor = Actor(
            config.num_mlp_layers_actor,
            4 * d + config.d_edge,
            config.hidden_dim_actor,
            1
        )

        # Stay head: [h_curws, h_global_s, h_global_w] = 3*d
        self.stay_head = Actor(
            config.num_mlp_layers_actor,
            3 * d,
            config.hidden_dim_actor,
            1
        )

        # Critic: [h_global_s, h_global_w] = 2*d
        self.critic = Critic(
            config.num_mlp_layers_critic,
            2 * d,
            config.hidden_dim_critic,
            1
        )

    def forward(self, state, adj):
        """
        Args:
            state: RMFSState with storage_features, ws_features, edge_feat, curws_idx, action_mask
            adj: (B, V, V) or (1, V, V) bool adjacency

        Returns:
            pi: (B, N_S+1) action probabilities (softmaxed)
            v:  (B, 1) state value estimate
        """
        B = state.storage_features.shape[0]
        device = state.storage_features.device

        # 1. Node projection
        h_s = self.storage_proj(state.storage_features)    # (B, N_S, d)
        h_w = self.ws_proj(state.ws_features)               # (B, N_W, d)

        # Add type embeddings
        s_type = torch.zeros(B, self.N_S, dtype=torch.long, device=device)
        w_type = torch.ones(B, self.N_W, dtype=torch.long, device=device)
        h_s = h_s + self.node_type_embed(s_type)
        h_w = h_w + self.node_type_embed(w_type)

        h = torch.cat([h_s, h_w], dim=1)  # (B, V, d)

        # 2. GATv2 layers (residual + LayerNorm)
        adj_exp = adj.expand(B, -1, -1) if adj.shape[0] == 1 else adj

        for gat_layer, ln in zip(self.gat_layers, self.layer_norms):
            h_new = gat_layer(h, adj_exp, state.edge_feat)
            h = ln(h + h_new)

        # 3. Split back
        h_storage = h[:, :self.N_S, :]     # (B, N_S, d)
        h_ws = h[:, self.N_S:, :]           # (B, N_W, d)

        # Global pooling
        h_global_s = h_storage.mean(dim=1)  # (B, d)
        h_global_w = h_ws.mean(dim=1)       # (B, d)

        # Current WS embedding
        curws_gather = state.curws_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.d_hidden)
        h_curws = h_ws.gather(1, curws_gather).squeeze(1)  # (B, d)

        # 4. Actor: per-storage scores
        # Edge features from curws to each storage
        batch_idx = torch.arange(B, device=device)
        curws_v_idx = self.N_S + state.curws_idx  # (B,)
        edge_from_curws = state.edge_feat[batch_idx, curws_v_idx]       # (B, V, d_edge)
        edge_curws_to_s = edge_from_curws[:, :self.N_S, :]              # (B, N_S, d_edge)

        # Expand for concatenation
        h_curws_exp = h_curws.unsqueeze(1).expand(-1, self.N_S, -1)         # (B, N_S, d)
        h_global_s_exp = h_global_s.unsqueeze(1).expand(-1, self.N_S, -1)   # (B, N_S, d)
        h_global_w_exp = h_global_w.unsqueeze(1).expand(-1, self.N_S, -1)   # (B, N_S, d)

        score_input = torch.cat([
            h_storage, h_curws_exp, h_global_s_exp, h_global_w_exp, edge_curws_to_s
        ], dim=-1)  # (B, N_S, 4*d + d_edge)

        storage_scores = self.actor(score_input).squeeze(-1)  # (B, N_S)

        # Stay score
        stay_input = torch.cat([h_curws, h_global_s, h_global_w], dim=-1)  # (B, 3*d)
        stay_score = self.stay_head(stay_input)  # (B, 1)

        # Combine: [stay_score, storage_scores]
        logits = torch.cat([stay_score, storage_scores], dim=-1)  # (B, N_S+1)

        # Apply action mask
        logits = logits.masked_fill(~state.action_mask, float('-inf'))

        pi = F.softmax(logits, dim=-1)

        # NaN safety
        if torch.isnan(pi).any():
            uniform = torch.full_like(pi, 1.0 / pi.shape[1])
            pi = torch.where(torch.isnan(pi), uniform, pi)

        # 5. Critic
        global_feat = torch.cat([h_global_s, h_global_w], dim=-1)  # (B, 2*d)
        v = self.critic(global_feat)  # (B, 1)

        return pi, v

    def get_max_action(self, state, adj):
        """Greedy action selection (for validation/test)."""
        with torch.no_grad():
            pi, _ = self.forward(state, adj)
            return torch.argmax(pi, dim=-1)
