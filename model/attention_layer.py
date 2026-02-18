import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleOpAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        """
            The implementation of Operation Message Attention Block
        :param input_dim: the dimension of input feature vectors
        :param output_dim: the dimension of output feature vectors
        :param dropout_prob: the parameter p for nn.Dropout()

        """
        super(SingleOpAttnBlock, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.alpha = 0.2

        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # inner attention용 파라미터: e_{i,k} = LeakyReLU(a_inner^T [Wh_i || Wh_k])
        self.a_inner = nn.Parameter(torch.empty(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a_inner.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, op_mask, pred_idx, succ_idx):
        """
        DAG-based 2-level GAT attention.

        :param h: operation feature vectors with shape [sz_b, N, input_dim]
        :param op_mask: used for masking zero pred/succ aggregations
                        with shape [sz_b, N, 3]
        :param pred_idx: predecessor indices [sz_b, N, max_preds], -1 padded
        :param succ_idx: successor indices   [sz_b, N, max_succs], -1 padded
        :return: output feature vectors with shape [sz_b, N, output_dim]
        """
        Wh = torch.matmul(h, self.W)  # (B, N, D)
        B, N, D = Wh.shape

        def inner_attend(neighbor_idx):
            """선행자/후행자 집합에 대한 inner attention → (B, N, D)"""
            max_k = neighbor_idx.shape[2]
            valid = neighbor_idx >= 0                          # (B, N, max_k)
            safe  = neighbor_idx.clamp(min=0)                  # (B, N, max_k)

            # gather neighbor embeddings: (B, N*max_k, D) → (B, N, max_k, D)
            nbr_emb = torch.gather(
                Wh, 1,
                safe.reshape(B, -1).unsqueeze(-1).expand(-1, -1, D)
            ).reshape(B, N, max_k, D)

            # inner attention: e_{i,k} = LeakyReLU(a_inner^T [Wh_i || Wh_k])
            q = torch.matmul(Wh,     self.a_inner[:D, :])   # (B, N, 1)
            k = torch.matmul(nbr_emb, self.a_inner[D:, :])  # (B, N, max_k, 1)
            e = self.leaky_relu(q.unsqueeze(2) + k)          # (B, N, max_k, 1)
            e = e.masked_fill(~valid.unsqueeze(-1), -9e15)

            alpha = F.softmax(e, dim=2)                       # (B, N, max_k, 1)
            agg   = (alpha * nbr_emb).sum(dim=2)             # (B, N, D)

            # 유효 이웃이 없으면 zero vector (op_mask로 outer에서 마스킹)
            agg = agg * valid.any(dim=2, keepdim=True).float()
            return agg

        pred_agg = inner_attend(pred_idx)   # (B, N, D)
        succ_agg = inner_attend(succ_idx)   # (B, N, D)

        # outer attention: {pred_agg, Wh_self, succ_agg} 3-slot
        Wh_concat = torch.stack([pred_agg, Wh, succ_agg], dim=-2)    # (B, N, 3, D)

        Wh1      = torch.matmul(Wh,       self.a[:D, :])              # (B, N, 1)
        Wh2_pred = torch.matmul(pred_agg, self.a[D:, :])              # (B, N, 1)
        Wh2_self = torch.matmul(Wh,       self.a[D:, :])              # (B, N, 1)
        Wh2_succ = torch.matmul(succ_agg, self.a[D:, :])              # (B, N, 1)
        Wh2_concat = torch.stack([Wh2_pred, Wh2_self, Wh2_succ], dim=-1)  # (B, N, 1, 3)

        e_outer  = self.leaky_relu(Wh1.unsqueeze(-1) + Wh2_concat)   # (B, N, 1, 3)
        zero_vec = -9e15 * torch.ones_like(e_outer)
        attention = torch.where(op_mask.unsqueeze(-2) > 0, zero_vec, e_outer)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_new = torch.matmul(attention, Wh_concat).squeeze(-2)        # (B, N, D)

        return h_new


class MultiHeadOpAttnBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        """
            The implementation of Operation Message Attention Block with multi-head attention
        :param input_dim: the dimension of input feature vectors
        :param output_dim: the dimension of each head's output
        :param dropout_prob: the parameter p for nn.Dropout()
        :param num_heads: the number of attention heads
        :param activation: the activation function used before output
        :param concat: the aggregation operator, true/false means concat/averaging
        """
        super(MultiHeadOpAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.num_heads = num_heads
        self.concat = concat
        self.activation = activation
        self.attentions = [
            SingleOpAttnBlock(input_dim, output_dim, dropout_prob) for
            _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, op_mask, pred_idx, succ_idx):
        """
        :param h: operation feature vectors with shape [sz_b, N, input_dim]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param pred_idx: predecessor indices [sz_b, N, max_preds], -1 padded
        :param succ_idx: successor indices   [sz_b, N, max_succs], -1 padded
        :return: output feature vectors with shape
                [sz_b, N, num_heads * output_dim] (if concat == true)
                or [sz_b, N, output_dim]
        """
        h = self.dropout(h)

        # shape: [ [sz_b, N, output_dim], ... [sz_b, N, output_dim]]
        h_heads = [att(h, op_mask, pred_idx, succ_idx) for att in self.attentions]

        if self.concat:
            h = torch.cat(h_heads, dim=-1)
        else:
            # h.shape : [sz_b, N, output_dim, num_heads]
            h = torch.stack(h_heads, dim=-1)
            # h.shape : [sz_b, N, output_dim]
            h = h.mean(dim=-1)

        return h if self.activation is None else self.activation(h)


class SingleMchAttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob):
        """
            The implementation of Machine Message Attention Block
        :param node_input_dim: the dimension of input node feature vectors
        :param edge_input_dim: the dimension of input edge feature vectors
        :param output_dim: the dimension of output feature vectors
        :param dropout_prob: the parameter p for nn.Dropout()
        """
        super(SingleMchAttnBlock, self).__init__()
        self.node_in_features = node_input_dim
        self.edge_in_features = edge_input_dim
        self.out_features = output_dim
        self.alpha = 0.2
        self.W = nn.Parameter(torch.empty(size=(node_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W_edge = nn.Parameter(torch.empty(size=(edge_input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(3 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, h, mch_mask, comp_val):
        """
        :param h: operation feature vectors with shape [sz_b, M, node_input_dim]
        :param mch_mask:  used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_val: a tensor with shape [sz_b, M, M, edge_in_features]
                    comp_val[i, k, q] corresponds to $c_{kq}$ in the paper,
                    which serves as a measure of the intensity of competition
                    between machine $M_k$ and $M_q$
        :return: output feature vectors with shape [sz_b, N, output_dim]
        """
        # Wh.shape: [sz_b, M, out_features]
        # W_edge.shape: [sz_b, M, M, out_features]

        Wh = torch.matmul(h, self.W)
        W_edge = torch.matmul(comp_val, self.W_edge)

        # compute attention matrix

        e = self.get_attention_coef(Wh, W_edge)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(mch_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)

        return h_prime

    def get_attention_coef(self, Wh, W_edge):
        """
            compute attention coefficients using node and edge features
        :param Wh: transformed node features
        :param W_edge: transformed edge features
        :return:
        """

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # [sz_b, M, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:2 * self.out_features, :])  # [sz_b, M, 1]
        edge_feas = torch.matmul(W_edge, self.a[2 * self.out_features:, :])  # [sz_b, M, M, 1]

        # broadcast add
        e = Wh1 + Wh2.transpose(-1, -2) + edge_feas.squeeze(-1)

        return self.leaky_relu(e)


class MultiHeadMchAttnBlock(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, dropout_prob, num_heads, activation, concat=True):
        """
            The implementation of Machine Message Attention Block with multi-head attention
        :param node_input_dim: the dimension of input node feature vectors
        :param edge_input_dim: the dimension of input edge feature vectors
        :param output_dim: the dimension of each head's output
        :param dropout_prob: the parameter p for nn.Dropout()
        :param num_heads: the number of attention heads
        :param activation: the activation function used before output
        :param concat: the aggregation operator, true/false means concat/averaging
        """
        super(MultiHeadMchAttnBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.concat = concat
        self.activation = activation
        self.num_heads = num_heads

        self.attentions = [SingleMchAttnBlock
                           (node_input_dim, edge_input_dim, output_dim, dropout_prob) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, h, mch_mask, comp_val):
        """
        :param h: operation feature vectors with shape [sz_b, M, node_input_dim]
        :param mch_mask:  used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_val: a tensor with shape [sz_b, M, M, edge_in_features]
                    comp_val[i, k, q] (any i) corresponds to $c_{kq}$ in the paper,
                    which serves as a measure of the intensity of competition
                    between machine $M_k$ and $M_q$
        :return: output feature vectors with shape
                [sz_b, M, num_heads * output_dim] (if concat == true)
                or [sz_b, M, output_dim]
        """
        h = self.dropout(h)

        h_heads = [att(h, mch_mask, comp_val) for att in self.attentions]

        if self.concat:
            # h.shape : [sz_b, M, output_dim*num_heads]
            h = torch.cat(h_heads, dim=-1)
        else:
            # h.shape : [sz_b, M, output_dim, num_heads]
            h = torch.stack(h_heads, dim=-1)
            h = h.mean(dim=-1)

        return h if self.activation is None else self.activation(h)
