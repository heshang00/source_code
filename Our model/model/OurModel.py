import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from lib.utils import norm_Adj

flag = False


def clones(module, N):
    """Produce N identical layers.

    Args:
        module: nn.Module
        N: int

    Returns:
        torch.nn.ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """Mask out subsequent positions.

    Args:
        size: int

    Returns:
        (1, size, size)
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0  # 1 means reachable; 0 means unreachable


class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        """Spatial graph convolution operation.

        Args:
            x: (batch_size, N, T, F_in)

        Returns:
            (batch_size, N, T, F_out)
        """
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape(
            (batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        """Spatial graph convolution operation.

        Args:
            x: (batch_size, N, F_in)

        Returns:
            (batch_size, N, F_out)
        """
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)


class Spatial_Attention_layer(nn.Module):
    """Compute spatial attention scores."""

    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, T, F_in)

        Returns:
            (batch_size, T, N, N)
        """
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)
        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))


class spatialAttentionScaledGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        """Spatial graph convolution operation.

        Args:
            x: (batch_size, N, T, F_in)

        Returns:
            (batch_size, N, T, F_out)
        """
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        spatial_attention = self.SAt(x) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)
        x = x.permute(0, 2, 1, 3).reshape(
            (-1, num_of_vertices, in_channels))  # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)
        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape(
            (batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout, gcn=None, smooth_layer_num=0):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        self.gcn_smooth_layers = None
        if (gcn is not None) and (smooth_layer_num > 0):
            self.gcn_smooth_layers = nn.ModuleList([gcn for _ in range(smooth_layer_num)])

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, T, F_in)

        Returns:
            (batch_size, N, T, F_out)
        """
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        if self.gcn_smooth_layers is not None:
            for _, l in enumerate(self.gcn_smooth_layers):
                embed = l(embed)  # (1,N,d_model) -> (1,N,d_model)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, T, F_in)

        Returns:
            (batch_size, N, T, F_out)
        """
        if self.lookup_index is not None:
            x = x + self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = x + self.pe[:, :, :x.size(2), :]
        return self.dropout(x.detach())


class HybridPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(HybridPositionalEncoding, self).__init__()
        self.learned_pe = nn.Embedding(max_len, d_model)
        fixed_pe = self.create_fixed_pe(d_model, max_len)
        self.register_buffer('fixed_pe', fixed_pe)
        self.dropout = nn.Dropout(dropout)

    def create_fixed_pe(self, d_model, max_len):
        """Create fixed sinusoidal position encoding.

        Args:
            d_model: model dimension
            max_len: max sequence length

        Returns:
            fixed sinusoidal position encoding tensor
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # (d_model // 2)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions use sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions use cosine
        return pe.unsqueeze(0).unsqueeze(0)  # shape (1, 1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: input tensor with shape (batch_size, N, T, F_in)

        Returns:
            tensor after position encoding
        """
        batch_size, N, T, F_in = x.size()
        device = x.device
        learned_pe = self.learned_pe(torch.arange(T, device=device))  # (T, d_model)
        learned_pe = learned_pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_model)
        fixed_pe = self.fixed_pe[:, :, :T]  # (1, 1, T, d_model)
        x = x + learned_pe + fixed_pe  # (batch_size, N, T, d_model)
        return self.dropout(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return self.dropout(sublayer(x))


class PositionWiseGCNFeedForward(nn.Module):
    def __init__(self, gcn, dropout=.0):
        super(PositionWiseGCNFeedForward, self).__init__()
        self.gcn = gcn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, N_nodes, T, F_in)

        Returns:
            (B, N, T, F_out)
        """
        return self.dropout(F.relu(self.gcn(x)))  # output shape (B, N_nodes, T, F_in)


def attention(query, key, value, mask=None, dropout=None):
    """
    Args:
        query: (batch, N, h, T1, d_k)
        key: (batch, N, h, T2, d_k)
        value: (batch, N, h, T2, d_k)
        mask: (batch, 1, 1, T2, T2)
        dropout:

    Returns:
        (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)


class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        """
        Args:
            query: (batch, N, T, d_model)
            key: (batch, N, T, d_model)
            value: (batch, N, T, d_model)
            mask: (batch, T, T)

        Returns:
            x: (batch, N, T, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        N = query.size(1)
        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
                             zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous()
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        avg_pool = avg_pool.view(b, c)  # Flatten to (b, c)
        excitation = F.relu(self.fc1(avg_pool))
        excitation = self.sigmoid(self.fc2(excitation))
        excitation = excitation.view(b, c, 1, 1)  # Reshape to (b, c, 1, 1)
        return x * excitation  # Apply the channel-wise attention


class SEConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, reduction=16):
        super(SEConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.se_block = SEBlock(out_channels, reduction)

    def forward(self, x):
        x = self.conv(x)  # Perform convolution
        x = self.se_block(x)  # Apply SE attention
        return x


class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module):
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3,
                 dropout=.0, dilation=2):
        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.padding = (kernel_size - 1) * dilation
        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding), dilation=(1, dilation)),
            2
        )
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        N = query.size(1)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [
                    l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                        -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context,
                                    (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))
                ]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [
                    l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                        -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context, (
                    query[:, :, self.w_length:self.w_length + self.d_length, :],
                    key[:, :, self.w_length:self.w_length + self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [
                    l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                        -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context, (
                    query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :],
                    key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):
            query, key = [
                l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                    -1).permute(0, 3, 1, 4, 2) for
                l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):
            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2))[:, :, :,
                    :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](
                    key[:, :, :self.w_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](
                    key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2))[:, :, :,
                        :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](
                    key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      2))[
                        :, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1,
                                                                                                              4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous()
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3,
                 dropout=.0, dilation=2):
        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.padding = (kernel_size - 1) // 2
        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        N = query.size(1)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w, key_w = [
                    l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context,
                                    (query[:, :, :self.w_length, :], key[:, :, :self.w_length, :]))]
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d, key_d = [
                    l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context, (
                        query[:, :, self.w_length:self.w_length + self.d_length, :],
                        key[:, :, self.w_length:self.w_length + self.d_length, :]))]
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h, key_h = [
                    l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                    for l, x in zip(self.conv1Ds_aware_temporal_context, (
                        query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :],
                        key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):
            query, key = [
                l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for
                l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):
            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous().view(nbatches,
                                                                                                        self.h,
                                                                                                        self.d_k, N,
                                                                                                        -1).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    4,
                                                                                                                    2)
            key_list = []

            if self.w_length > 0:
                key_w = self.conv1Ds_aware_temporal_context[1](
                    key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                        -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.conv1Ds_aware_temporal_context[1](
                    key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](
                    key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous()
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_qc_k1d(nn.Module):
    def __init__(self, nb_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3,
                 dropout=.0, dilation=2):
        super(MultiHeadAttentionAwareTemporalContex_qc_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.causal_padding = (kernel_size - 1) * dilation
        self.padding_1D = (kernel_size + 1) // 2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                              padding=(0, self.causal_padding), dilation=(1, dilation))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                            padding=(0, self.padding_1D), dilation=(1, dilation))
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        N = query.size(1)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                query_w = self.query_conv1Ds_aware_temporal_context(query[:, :, :self.w_length, :].permute(0, 3, 1, 2))[
                          :, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(
                    0, 3, 1, 4, 2)
                key_w = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                        -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_w)
                key_list.append(key_w)

            if self.d_length > 0:
                query_d = self.query_conv1Ds_aware_temporal_context(
                    query[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2))[:, :, :,
                          :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1,
                                                                                                              4, 2)
                key_d = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_d)
                key_list.append(key_d)

            if self.h_length > 0:
                query_h = self.query_conv1Ds_aware_temporal_context(
                    query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(
                        0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                       -1).permute(0, 3, 1, 4, 2)
                key_h = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):
            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :,
                    :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            key = self.key_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h,
                                                                                                       self.d_k, N,
                                                                                                       -1).permute(0, 3,
                                                                                                                   1, 4,
                                                                                                                   2)

        elif (not query_multi_segment) and (key_multi_segment):
            query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :,
                    :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
            key_list = []

            if self.w_length > 0:
                key_w = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, :self.w_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                        -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_w)

            if self.d_length > 0:
                key_d = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, self.w_length:self.w_length + self.d_length, :].permute(0, 3, 1, 2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_d)

            if self.h_length > 0:
                key_h = self.key_conv1Ds_aware_temporal_context(
                    key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      2)).contiguous().view(
                    nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(2, 3).contiguous()
        x = x.view(nbatches, N, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_dense, trg_dense, generator, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.trg_embed = trg_dense
        self.prediction_generator = generator
        self.to(DEVICE)

    def forward(self, src, trg):
        """
        Args:
            src: (batch_size, N, T_in, F_in)
            trg: (batch, N, T_out, F_out)
        """
        encoder_output = self.encode(src)
        return self.decode(trg, encoder_output)

    def encode(self, src):
        """
        Args:
            src: (batch_size, N, T_in, F_in)
        """
        h = self.src_embed(src)
        return self.encoder(h)

    def decode(self, trg, encoder_output):
        return self.prediction_generator(self.decoder(self.trg_embed(trg), encoder_output))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gcn, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward_gcn = gcn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, query_multi_segment=True, key_multi_segment=True))
        return self.sublayer[1](x, self.feed_forward_gcn)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_gcn = gcn
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory):
        m = memory
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, query_multi_segment=False,
                                                         key_multi_segment=False))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, query_multi_segment=False, key_multi_segment=True))
        return self.sublayer[2](x, self.feed_forward_gcn)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory):
        """
        Args:
            x: (batch, N, T', d_model)
            memory: (batch, N, T, d_model)

        Returns:
            (batch, N, T', d_model)
        """
        for layer in self.layers:
            x = layer(x, memory)
        return self.norm(x)


def search_index(max_len, num_of_depend, num_for_predict, points_per_hour, units):
    """
    Args:
        max_len: int, length of all encoder input
        num_of_depend: int,
        num_for_predict: int, the number of points will be predicted for each sample
        units: int, week: 7 * 24, day: 24, recent(hour): 1
        points_per_hour: int, number of points per hour, depends on data

    Returns:
        list[(start_idx, end_idx)]
    """
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hour * units * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx


def make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, nb_head, num_of_weeks,
               num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=.0, aware_temporal_context=True,
               ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True,
               use_LayerNorm=True):
    c = copy.deepcopy
    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)
    num_of_vertices = norm_Adj_matrix.shape[0]
    src_dense = nn.Linear(encoder_input_size, d_model)

    if ScaledSAt:
        position_wise_gcn = PositionWiseGCNFeedForward(spatialAttentionScaledGCN(norm_Adj_matrix, d_model, d_model),
                                                       dropout=dropout)
    else:
        position_wise_gcn = PositionWiseGCNFeedForward(spatialGCN(norm_Adj_matrix, d_model, d_model), dropout=dropout)

    trg_dense = nn.Linear(decoder_output_size, d_model)

    max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict,
                  num_of_hours * num_for_predict)

    w_index = search_index(max_len, num_of_weeks, num_for_predict, points_per_hour, 7 * 24)
    d_index = search_index(max_len, num_of_days, num_for_predict, points_per_hour, 24)
    h_index = search_index(max_len, num_of_hours, num_for_predict, points_per_hour, 1)
    en_lookup_index = w_index + d_index + h_index

    if aware_temporal_context:
        attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head, d_model, num_of_weeks, num_of_days,
                                                                num_of_hours, num_for_predict, kernel_size,
                                                                dropout=dropout)
        attn_st = MultiHeadAttentionAwareTemporalContex_qc_k1d(nb_head, d_model, num_of_weeks, num_of_days,
                                                               num_of_hours, num_for_predict, kernel_size,
                                                               dropout=dropout)
        att_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head, d_model, num_of_weeks, num_of_days, num_of_hours,
                                                             num_for_predict, kernel_size, dropout=dropout)
    else:
        attn_ss = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        attn_st = MultiHeadAttention(nb_head, d_model, dropout=dropout)
        att_tt = MultiHeadAttention(nb_head, d_model, dropout=dropout)

    if SE and TE:
        encode_temporal_position = HybridPositionalEncoding(d_model, max_len)
        decode_temporal_position = HybridPositionalEncoding(d_model, max_len)
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout,
                                                     GCN(norm_Adj_matrix, d_model, d_model),
                                                     smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position), c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position), c(spatial_position))
    elif SE and (not TE):
        spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout,
                                                     GCN(norm_Adj_matrix, d_model, d_model),
                                                     smooth_layer_num=smooth_layer_num)
        encoder_embedding = nn.Sequential(src_dense, c(spatial_position))
        decoder_embedding = nn.Sequential(trg_dense, c(spatial_position))
    elif (not SE) and (TE):
        encode_temporal_position = HybridPositionalEncoding(d_model, max_len)
        decode_temporal_position = HybridPositionalEncoding(d_model, max_len)
        encoder_embedding = nn.Sequential(src_dense, c(encode_temporal_position))
        decoder_embedding = nn.Sequential(trg_dense, c(decode_temporal_position))
    else:
        encoder_embedding = nn.Sequential(src_dense)
        decoder_embedding = nn.Sequential(trg_dense)

    encoderLayer = EncoderLayer(d_model, attn_ss, c(position_wise_gcn), dropout)
    encoder = Encoder(encoderLayer, num_layers)
    decoderLayer = DecoderLayer(d_model, att_tt, attn_st, c(position_wise_gcn), dropout)
    decoder = Decoder(decoderLayer, num_layers)
    generator = nn.Linear(d_model, decoder_output_size)
    model = EncoderDecoder(encoder, decoder, encoder_embedding, decoder_embedding, generator, DEVICE)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model