##########################################################
# pytorch-kaldi v.0.1
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from distutils.util import strtobool
import math
import json

# uncomment below if you want to use SRU
# and you need to install SRU: pip install sru[cuda].
# or you can install it from source code: https://github.com/taolei87/sru.
# import sru

#**********************새로 추가하는 부분 (첫번째 시도 > 실패)*******************************************************
# def _pre_hook(state_dict, prefix, local_metadata, strict,
#               missing_keys, unexpected_keys, error_msgs):
#     """Perform pre-hook in load_state_dict for backward compatibility.
#     Note:
#         We saved self.pe until v.0.5.2 but we have omitted it later.
#         Therefore, we remove the item "pe" from `state_dict` for backward compatibility.
#     """
#     k = prefix + "pe"
#     if k in state_dict:
#         state_dict.pop(k)
#
# # class PositionalEncoding(torch.nn.Module):
# #     """Positional encoding.
# #     :param int d_model: embedding dim
# #     :param float dropout_rate: dropout rate
# #     :param int max_len: maximum input length
# #     """
# #
# #     def __init__(self, d_model, dropout_rate, max_len=5000):
# #         """Construct an PositionalEncoding object."""
# #         super(PositionalEncoding, self).__init__()
# #         self.d_model = d_model
# #         self.xscale = math.sqrt(self.d_model)
# #         self.dropout = torch.nn.Dropout(p=dropout_rate)
# #         #self.pe = None
# #         self.extend_pe(torch.tensor(0.0).expand(1, max_len))
# #         self._register_load_state_dict_pre_hook(_pre_hook)
# #
# #     def extend_pe(self, x):
# #         """Reset the positional encodings."""
# #         if self.pe is not None:
# #             if self.pe.size(1) >= x.size(1):
# #                 if self.pe.dtype != x.dtype or self.pe.device != x.device:
# #                     self.pe = self.pe.to(dtype=x.dtype, device=x.device)
# #                 return
# #         pe = torch.zeros(x.size(1), self.d_model)
# #         position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
# #         div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) *
# #                              -(math.log(10000.0) / self.d_model))
# #         pe[:, 0::2] = torch.sin(position * div_term)
# #         print('pe[:, 0::2]', pe[:, 0::2].size())
# #         pe[:, 1::2] = torch.cos(position * div_term)
# #         print('pe[:, 1::2]', pe[:, 1::2].size())
# #         pe = pe.unsqueeze(0)
# #         print('pe',pe.size())
# #         self.pe = pe.to(device=x.device, dtype=x.dtype)
# #
# #     def forward(self, x: torch.Tensor):
# #         """Add positional encoding.
# #         Args:
# #             x (torch.Tensor): Input. Its shape is (batch, time, ...)
# #         Returns:
# #             torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
# #         """
# #         print('x', x.size(), x.type())
# #         self.extend_pe(x)
# #         print('pe(x)', x.size(), x.type())
# #         print('self.pe[:, :x.size(1)]', self.pe[:, :x.size(1)].size())
# #         print('self.xscale', self.xscale.size())
# #         x = x * self.xscale + self.pe[:, :x.size(1)]
# #         return self.dropout(x)
# # class PositionalEncoding2(torch.nn.Module):
# #     """Positional encoding module until v.0.5.2."""
# #
# #     def __init__(self, d_model, dropout_rate, max_len=5000):
# #         import math
# #         super().__init__()
# #         self.dropout = torch.nn.Dropout(p=dropout_rate)
# #         # Compute the positional encodings once in log space.
# #         pe = torch.zeros(max_len, d_model)
# #         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
# #         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
# #                              -(math.log(10000.0) / d_model))
# #         pe[:, 0::2] = torch.sin(position * div_term)
# #         pe[:, 1::2] = torch.cos(position * div_term)
# #         pe = pe.unsqueeze(0)
# #         self.max_len = max_len
# #         self.xscale = math.sqrt(d_model)
# #         self.register_buffer('pe', pe)
# #
# #     def forward(self, x):
# #         print('임베딩후에 position임베딩의 입력값 x', x.size(), x.type())
# #         print('self.pe[:, :x.size(1)]', self.pe[:, :x.size(1)].size(), self.pe[:, :x.size(1)].type())
# #         x = x * self.xscale + self.pe[:, :x.size(1)]
# #         print('position 임베딩 후의 입력값 x', x.size(), x.type())
# #         return self.dropout(x)
# #
# # class MultiHeadedAttention(nn.Module):
# #     """Multi-Head Attention layer.
# #     :param int n_head: the number of head s
# #     :param int n_feat: the number of features
# #     :param float dropout_rate: dropout rate
# #     """
# #
# #     def __init__(self, n_head, n_feat, dropout_rate):
# #         """Construct an MultiHeadedAttention object."""
# #         super(MultiHeadedAttention, self).__init__()
# #         assert n_feat % n_head == 0
# #         # We assume d_v always equals d_k
# #         self.d_k = n_feat // n_head
# #         self.h = n_head
# #         self.linear_q = nn.Linear(n_feat, n_feat)
# #         self.linear_k = nn.Linear(n_feat, n_feat)
# #         self.linear_v = nn.Linear(n_feat, n_feat)
# #         self.linear_out = nn.Linear(n_feat, n_feat)
# #         self.attn = None
# #         self.dropout = nn.Dropout(p=dropout_rate)
# #
# #     def forward(self, query, key, value, mask):
# #         """Compute 'Scaled Dot Product Attention'.
# #         :param torch.Tensor query: (batch, time1, size)
# #         :param torch.Tensor key: (batch, time2, size)
# #         :param torch.Tensor value: (batch, time2, size)
# #         :param torch.Tensor mask: (batch, time1, time2)
# #         :param torch.nn.Dropout dropout:
# #         :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
# #              weighted by the query dot key attention (batch, head, time1, time2)
# #         """
# #         n_batch = query.size(0)
# #         q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
# #         k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
# #         v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
# #         q = q.transpose(1, 2)  # (batch, head, time1, d_k)
# #         k = k.transpose(1, 2)  # (batch, head, time2, d_k)
# #         v = v.transpose(1, 2)  # (batch, head, time2, d_k)
# # ###############
# #         print('q', q.size(), q.type())
# #         print('mask', mask.size(), mask.type())
# #         ###########
# #         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
# #         if mask is not None:
# #             mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
# #             min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
# #             scores = scores.masked_fill(mask, min_value)
# #             self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
# #         else:
# #             self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
# #
# #         p_attn = self.dropout(self.attn)
# #         x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
# #         x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
# #         return self.linear_out(x)  # (batch, time1, d_model)
# #
# # class FullyconnectedFeedForward(torch.nn.Module):
# #     """Positionwise feed forward layer.를 transformer am 논문에서 원하는 형태로 약간 수정
# #     linear>gelu>dropout>linear
# #     :param int idim: input dimenstion
# #     :param int hidden_units: number of hidden units
# #     :param float dropout_rate: dropout rate
# #     """
# #
# #     def __init__(self, idim, hidden_units, dropout_rate):
# #         """Construct an PositionwiseFeedForward object."""
# #         super(FullyconnectedFeedForward, self).__init__()
# #         self.w_1 = torch.nn.Linear(idim, hidden_units)
# #         self.w_2 = torch.nn.Linear(hidden_units, idim)
# #         self.dropout = torch.nn.Dropout(dropout_rate)
# #
# #     def forward(self, x):
# #         """Forward funciton."""
# #         return self.w_2(self.dropout(nn.GELU(self.w_1(x)))) #activation 함수 이렇게 불러오면 되나
# #
# #
# # class Transformer(nn.Module):
# #     def __init__(self, options, inp_dim):
# #         super(Transformer, self).__init__()
# #         self.input_dim = inp_dim
# #         print('이게 진짜 inputdim', self.input_dim)
# #
# #         #pos_enc_class = PositionalEncoding()
# #         #MHA = MultiHeadedAttention()
# #         #FFN = FullyconnectedFeedForward()
# #         self.attention_dim = 512
# #         self.attention_heads = 8
# #         self.linear_units = 2048
# #         self.attention_dropout_rate = 0.0
# #         self.positional_dropout_rate = 0.1
# #         self.dropout_rate = 0.1
# #         self.padding_idx = -1
# #
# #         self.dropout = torch.nn.Dropout(p=0.1)
# #
# #         self.xs_mask = None
# #
# #         self.MHA = MultiHeadedAttention(self.attention_heads, self.attention_dim, self.attention_dropout_rate)
# #         #self.embed = torch.nn.Sequential(torch.nn.Embedding(self.input_dim, self.attention_dim, padding_idx=self.padding_idx),
# #         #                                 PositionalEncoding(self.attention_dim, self.positional_dropout_rate)).double()
# #         self.pre_embed = torch.nn.Embedding(self.input_dim, self.attention_dim, padding_idx=self.padding_idx)
# #         self.embed = PositionalEncoding2(self.attention_dim, self.positional_dropout_rate)
# #         self.FFN = FullyconnectedFeedForward(self.attention_dim, self.linear_units, self.dropout_rate)
# #         self.transformer_lay = list(map(int, options["transformer_lay"].split(",")))
# #         self.N_transformer_lay = len(self.transformer_lay)
# #
# #         self.ln = LayerNorm(self.attention_dim) #이거 뭘까
# #         self.out_dim = self.attention_dim
# #
# #     def _generate_square_subsequent_mask(self, sz):
# #         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
# #         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
# #         return mask
# #
# #     def forward(self, xs): #forward(self, xs, masks):
# #         #masks=self.make_fmllr_mask(xs)
# #         print('fmllr 입력값', xs.size(), xs.type())
# #         if self.xs_mask is None or self.xs_mask.size(0) != len(xs):
# #             device = xs.device
# #             masks = self._generate_square_subsequent_mask(len(xs)).to(device)
# #             self.xs_mask = masks
# #         #######
# #         # emb = torch.nn.Embedding(self.input_dim, self.attention_dim, padding_idx=self.padding_idx)
# #         # emb = emb.double()
# #         # emb_xs = emb(xs)
# #         # print('emb_xs', emb_xs.size(), emb_xs.type())
# #         # xs = self.embed(emb_xs)
# #         # print('position emb 후', xs.size(), xs.type())
# #         xs = xs.long()
# #         xs = self.pre_embed(xs)
# #         print('그냥embeeding후', xs.size())
# #         #xs = xs.float()
# #         xs = self.embed(xs) #xs, masks=self.embed(xs, masks) masks는 어떨때 embedding시키는거지 conv로 임베딩할때만이야?
# #         print('최종 임베딩 후', xs.size(), xs.type())
# #         for i in range(self.N_transformer_lay):
# #             residual1 = xs
# #             xs = self.ln(xs)
# #             sub_x = residual1 + self.dropout(self.MHA(xs, xs, xs, self.xs_mask))
# #             residual2 = sub_x
# #             sub_x = self.ln(sub_x)
# #             out_x = residual2 + self.dropout(self.FFN(sub_x))
# #             xs = self.ln(out_x)
# #             print('최종 output', xs.size(), xs.type())
# #         return xs
#
# #****************************************************************************************
# #*******************************새로짜는 부분 두번째 시도*************************************
# def make_pad_mask(lengths, xs=None, length_dim=-1):
#     if length_dim == 0:
#         raise ValueError('length_dim cannot be 0: {}'.format(length_dim))
#
#     if not isinstance(lengths, list):
#         lengths = lengths.tolist()
#     bs = int(len(lengths))
#     if xs is None:
#         maxlen = int(max(lengths))
#     else:
#         maxlen = xs.size(length_dim)
#
#     seq_range = torch.arange(0, maxlen, dtype=torch.int64)
#     seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
#     seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
#     mask = seq_range_expand >= seq_length_expand
#
#     if xs is not None:
#         assert xs.size(0) == bs, (xs.size(0), bs)
#
#         if length_dim < 0:
#             length_dim = xs.dim() + length_dim
#         # ind = (:, None, ..., None, :, , None, ..., None)
#         ind = tuple(slice(None) if i in (0, length_dim) else None
#                     for i in range(xs.dim()))
#         mask = mask[ind].expand_as(xs).to(xs.device)
#     return mask
#
# def make_non_pad_mask(lengths, xs=None, length_dim=-1):
#     return ~make_pad_mask(lengths, xs, length_dim)
#
# class PositionalEncoding2(torch.nn.Module):
#     """Positional encoding module until v.0.5.2."""
#
#     def __init__(self, d_model, max_len=5000):
#         import math
#         super().__init__()
#         self.dropout = torch.nn.Dropout(p=0.1)
#         device = torch.device("cuda")
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).to(device)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1).to(device)
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
#                              -(math.log(10000.0) / d_model)).to(device)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.max_len = max_len
#         self.xscale = math.sqrt(d_model)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         print('임베딩후에 position임베딩의 입력값 x', x.size(), x.type())
#         print('self.pe[:, :x.size(1)]', self.pe[:, :x.size(1)].size(), self.pe[:, :x.size(1)].type())
#         x = x * self.xscale + self.pe[:, :x.size(1)]
#         print('position 임베딩 후의 입력값 x', x.size(), x.type())
#         return self.dropout(x)
#
# class MultiHeadedAttention(nn.Module):
#     """Multi-Head Attention layer.
#     :param int n_head: the number of head s
#     :param int n_feat: the number of features
#     :param float dropout_rate: dropout rate
#     """
#
#     def __init__(self, n_head, n_feat):
#         """Construct an MultiHeadedAttention object."""
#         super(MultiHeadedAttention, self).__init__()
#         assert n_feat % n_head == 0
#         # We assume d_v always equals d_k
#         self.d_k = n_feat // n_head
#         self.h = n_head
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         self.linear_k = nn.Linear(n_feat, n_feat)
#         self.linear_v = nn.Linear(n_feat, n_feat)
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         self.attn = None
#         self.dropout = nn.Dropout(p= 0.1)
#
#     def forward(self, query, key, value, mask):
#         """Compute 'Scaled Dot Product Attention'.
#         :param torch.Tensor query: (batch, time1, size)
#         :param torch.Tensor key: (batch, time2, size)
#         :param torch.Tensor value: (batch, time2, size)
#         :param torch.Tensor mask: (batch, time1, time2)
#         :param torch.nn.Dropout dropout:
#         :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
#              weighted by the query dot key attention (batch, head, time1, time2)
#         """
#         print('qq',query.size())
#         print('mm',mask.size())
#         n_batch = query.size(0)
#         q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
#         k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
#         v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
#         q = q.transpose(1, 2)  # (batch, head, time1, d_k)
#         k = k.transpose(1, 2)  # (batch, head, time2, d_k)
#         v = v.transpose(1, 2)  # (batch, head, time2, d_k)
# ###############
#         print('q', q.size(), q.type())
#         print('mask', mask.size(), mask.type())
#         ###########
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
#         if mask is not None:
#             mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
#             min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).np().dtype).min)
#             scores = scores.masked_fill(mask, min_value)
#             self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
#         else:
#             self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
#
#         p_attn = self.dropout(self.attn)
#         x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
#         x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
#         return self.linear_out(x)  # (batch, time1, d_model)
#
# class FullyconnectedFeedForward(torch.nn.Module):
#     """ linear>gelu>dropout>linear"""
#
#     def __init__(self, idim, hidden_units):
#         """Construct an PositionwiseFeedForward object."""
#         super(FullyconnectedFeedForward, self).__init__()
#         self.w_1 = torch.nn.Linear(idim, hidden_units)
#         self.w_2 = torch.nn.Linear(hidden_units, idim)
#         self.dropout = torch.nn.Dropout(p=0.1)
#
#     def forward(self, x):
#         """Forward funciton."""
#         return self.w_2(self.dropout(nn.GELU(self.w_1(x)))) #activation 함수 이렇게 불러오면 되나
#
#
# class Transformer(nn.Module):
#     def __init__(self, options, inp_dim):
#         super(Transformer, self).__init__()
#         self.input_dim = inp_dim
#         print('이게 진짜 inputdim', self.input_dim)
#         self.transf_lay = list(map(int, options["transf_lay"].split(",")))
#         self.transf_drop = list(map(float, options["transf_drop"].split(",")))
#         self.transf_use_laynorm = list(map(strtobool, options["transf_use_laynorm"].split(",")))
#         self.transf_use_laynorm_inp = strtobool(options["transf_use_laynorm_inp"])
#
#         self.wx = nn.ModuleList([])
#         self.ln = nn.ModuleList([])
#         self.drop = nn.ModuleList([])
#
#         self.hid_dim = 512
#         self.padding_idx = -1
#         self.pre_embed = torch.nn.Embedding(self.input_dim, self.hid_dim, padding_idx=self.padding_idx)
#         self.embed = PositionalEncoding2(self.hid_dim)
#         #self.transformer = TransformerLayer(self.transf_lay)
#
#         # input layer normalization
#         if self.transf_use_laynorm_inp:
#             self.ln0 = LayerNorm(self.input_dim)
#
#         self.N_transf_lay = len(self.transf_lay)
#         current_input = self.input_dim
#
#         # Initialization of hidden layers
#         for i in range(self.N_transf_lay):
#             # dropout
#             self.drop.append(nn.Dropout(p=self.transf_drop[i]))
#
#             # layer norm initialization
#             self.ln.append(LayerNorm(self.transf_lay[i]))
#
#             # Linear operations
#             self.wx.append(nn.Linear(current_input, self.transf_lay[i]))
#
#             # weight initialization
#             self.wx[i].weight = torch.nn.Parameter(
#                 torch.Tensor(self.transf_lay[i], current_input).uniform_(
#                     -np.sqrt(0.01 / (current_input + self.transf_lay[i])),
#                     np.sqrt(0.01 / (current_input + self.transf_lay[i])),
#                 )
#             )
#             current_input = self.transf_lay[i]
#         self.out_dim = current_input
#     def forward(self, xs):
#         print('입력값 xs', xs.size())
#         device = torch.device("cuda")
#         if bool(self.transf_use_laynorm_inp):
#             xs = self.ln0((xs))
#         ##xs = xs.long()
#         ##xs = self.pre_embed(xs)
#         ##print('그냥embeeding후', xs.size())
#         ##xs = self.embed(xs)
#         for i in range(self.N_transf_lay):
#             if self.transf_use_laynorm[i]:
#                 #xs = self.ln[i](self.wx[i](xs))
#                 transformer = TransformerLayer(self.transf_lay[i])
#                 xs = transformer(xs)
#             print('출력값 output', xs.size())
#         return xs
#
#
#
#
# class TransformerLayer(nn.Module):
#     def __init__(self, attention_dim):
#         super(TransformerLayer, self).__init__()
#         self.attention_dim = attention_dim
#         self.attention_heads = 16
#         self.linear_units = 2048
#
#         self.dropout = torch.nn.Dropout(p=0.1)
#         self.xs_mask = None
#         self.ln = LayerNorm(self.attention_dim)
#
#         self.MHA = MultiHeadedAttention(self.attention_heads, attention_dim)
#         self.FFN = FullyconnectedFeedForward(attention_dim, self.linear_units)
#
#     def forward(self, xs):
#         print('레이어별  input', xs.size(), xs.type())
#         device = torch.device("cuda")
#         xs_len=torch.tensor([100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100])
#         xs = xs[:, :max(xs_len)]
#         if self.xs_mask is None or self.xs_mask.size(0) != xs_len:
#             xs_mask = make_non_pad_mask(xs_len.tolist()).to(xs.device).unsqueeze(-2)
#             #self.xs_mask = masks
#
#         residual1 = xs
#         xs = self.ln(xs)
#         sub_x = residual1 + self.dropout(self.MHA(xs, xs, xs, xs_mask)).to(device)
#         residual2 = sub_x
#         sub_x = self.ln(sub_x)
#         out_x = residual2 + self.dropout(self.FFN(sub_x)).to(device)
#         xs = self.ln(out_x)
#         print('레이어별 output', xs.size(), xs.type())
#         return xs
#
# class Transform_LayerNorm(torch.nn.LayerNorm):
#     """Layer normalization module.
#
#     :param int nout: output dim size
#     :param int dim: dimension to be normalized
#     """
#
#     def __init__(self, nout, dim=-1):
#         """Construct an LayerNorm object."""
#         super(LayerNorm, self).__init__(nout, eps=1e-12)
#         self.dim = dim
#
#     def forward(self, x):
#         """Apply layer normalization.
#
#         :param torch.Tensor x: input tensor
#         :return: layer normalized tensor
#         :rtype torch.Tensor
#         """
#         if self.dim == -1:
#             return super(LayerNorm, self).forward(x)
#         return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)
#*******************************세번째 시도*************************
##########################################################
# pytorch-kaldi v.0.1
# transformer AM
# 2020/3.17 (껍데기는 돌아감. 임베딩빼고, mask빼고, layer6에 batch는 4/2까지 줄였는데 loss 안떨어짐)
##########################################################
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import numpy
# from distutils.util import strtobool
# import math
# import json
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# def act_fun(act_type):
#     if act_type == "gelu":
#         return nn.GELU()  # 표현 이거 맞나
#
# def make_pad_mask(lengths, xs=None, length_dim=-1):
#     if length_dim == 0:
#         raise ValueError('length_dim cannot be 0: {}'.format(length_dim))
#
#     if not isinstance(lengths, list):
#         lengths = lengths.tolist()
#     bs = int(len(lengths))
#     if xs is None:
#         maxlen = int(max(lengths))
#     else:
#         maxlen = xs.size(length_dim)
#
#     seq_range = torch.arange(0, maxlen, dtype=torch.int64)
#     seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
#     seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
#     mask = seq_range_expand >= seq_length_expand
#
#     if xs is not None:
#         assert xs.size(0) == bs, (xs.size(0), bs)
#
#         if length_dim < 0:
#             length_dim = xs.dim() + length_dim
#         # ind = (:, None, ..., None, :, , None, ..., None)
#         ind = tuple(slice(None) if i in (0, length_dim) else None
#                     for i in range(xs.dim()))
#         mask = mask[ind].expand_as(xs).to(xs.device)
#     return mask
#
# def make_non_pad_mask(lengths, xs=None, length_dim= -1):
#     return ~make_pad_mask(lengths, xs, length_dim)
#
#
# def _pre_hook(state_dict, prefix, local_metadata, strict,
#               missing_keys, unexpected_keys, error_msgs):
#     k = prefix + "pe"
#     if k in state_dict:
#         state_dict.pop(k)
#
# class GELU(nn.Module):
#     """
#     Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
#     """
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#
# class PositionwiseFeedForward(torch.nn.Module):
#     def __init__(self, idim, hidden_units, dropout_rate):
#         """Construct an PositionwiseFeedForward object."""
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = torch.nn.Linear(idim, hidden_units)
#         self.w_2 = torch.nn.Linear(hidden_units, idim)
#         self.dropout = torch.nn.Dropout(dropout_rate)
#         self.activation = GELU()
#
#     def forward(self, x):
#         """Forward funciton."""
#         #######print('FFN input x', x.size())
#         x=self.w_1(x)
#         #######print('ffn에서 linear 하나 통과', x.size())
#         x = self.activation(x)
#         #######print('gelu통과한',x.size())
#         x = self.w_2(self.dropout(x))
#         #######print('ffn최종 통과한 ', x.size())
#         return x
#
# class TLayerNorm(torch.nn.LayerNorm):
#
#     def __init__(self, nout, dim=-1):
#         """Construct an TLayerNorm object."""
#         super(TLayerNorm, self).__init__(nout, eps=1e-12)
#         self.dim = dim
#     def forward(self, x):
#
#         #######print('TLayerNorm input x', x.size())
#         if self.dim == -1:
#             return super(TLayerNorm, self).forward(x)
#         return super(TLayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)
#
#
# class MultiHeadedAttention(nn.Module):
#
#     def __init__(self, n_head, n_feat, dropout_rate):
#         """Construct an MultiHeadedAttention object."""
#         super(MultiHeadedAttention, self).__init__()
#         assert n_feat % n_head == 0
#         # We assume d_v always equals d_k
#         self.d_k = n_feat // n_head
#         self.h = n_head
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         self.linear_k = nn.Linear(n_feat, n_feat)
#         self.linear_v = nn.Linear(n_feat, n_feat)
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout_rate)
#
#     def forward(self, query, key, value): #forward(self, query, key, value, mask):
#         #######print('query size', query.size())
#         #######print('key size', key.size())
#         n_batch = query.size(0)
#         q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
#         k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
#         v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
#         #######print('q size', q.size())
#         #######print('k size', k.size())
#
#         q = q.transpose(1, 2)  # (batch, head, time1, d_k)
#         k = k.transpose(1, 2)  # (batch, head, time2, d_k)
#         v = v.transpose(1, 2)  # (batch, head, time2, d_k)
#         #######print('transpose q size', q.size())
#         #######print('transpose k size', k.size())
#
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
#         self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
#
#         p_attn = self.dropout(self.attn)
#         x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
#         x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
#         return self.linear_out(x)  # (batch, time1, d_model)
#
# class Transformer(nn.Module):
#
#     def __init__(self, options, idim):
#         """Construct an Encoder object."""
#         super(Transformer, self).__init__()
#         self.attention_dim = 512
#         self.attention_heads = 8
#         self.linear_units = 2048
#         self.num_blocks = 6
#         self.dropout_rate = 0.1
#         self.positional_dropout_rate = 0.1
#         self.attention_dropout_rate = 0.0
#         self.input_layer = "embed"
#         #pos_enc_class = PositionalEncoding()
#         self.normalize_before = True
#         self.concat_after = False
#         self.positionwise_layer_type = "linear"
#         self.positionwise_conv_kernel_size = 1
#         self.padding_idx = -1
#         self.linear_xs = nn.Linear(idim, self.attention_dim)
#         print('idim', idim)
#         # if self.input_layer == "linear":
#         #     self.embed = torch.nn.Sequential(
#         #         torch.nn.Linear(idim, self.attention_dim),
#         #         torch.nn.LayerNorm(self.attention_dim),
#         #         torch.nn.Dropout(self.dropout_rate),
#         #         torch.nn.ReLU(),
#         #         PositionalEncoding(self.attention_dim, self.positional_dropout_rate)
#         #     )
#         # elif self.input_layer == "conv2d":
#         #     self.embed = Conv2dSubsampling(idim, self.attention_dim, self.dropout_rate)
#         # elif self.input_layer == 'vgg2l':
#         #     self.embed = VGG2L(idim, self.attention_dim)
#         # elif self.input_layer == "embed":
#         #     self.embed = torch.nn.Sequential(
#         #         torch.nn.Embedding(idim, self.attention_dim, padding_idx=self.padding_idx),
#         #         PositionalEncoding(self.attention_dim, self.positional_dropout_rate, device)
#         #     )
#         # elif isinstance(self.input_layer, torch.nn.Module):
#         #     self.embed = torch.nn.Sequential(
#         #         self.input_layer,
#         #         PositionalEncoding(self.attention_dim, self.positional_dropout_rate),
#         #     )
#         # elif self.input_layer is None:
#         #     self.embed = torch.nn.Sequential(
#         #         PositionalEncoding(self.attention_dim, self.positional_dropout_rate)
#         #     )
# #        else:
#  #           raise ValueError("unknown input_layer: " + self.input_layer)
#         self.normalize_before = self.normalize_before
#         if self.positionwise_layer_type == "linear":
#             positionwise_layer = PositionwiseFeedForward
#             positionwise_layer_args = (self.attention_dim, self.linear_units, self.dropout_rate)
#         elif self.positionwise_layer_type == "conv1d":
#             positionwise_layer = MultiLayeredConv1d
#             positionwise_layer_args = (self.attention_dim, self.linear_units, self.positionwise_conv_kernel_size, self.dropout_rate)
#         elif self.positionwise_layer_type == "conv1d-linear":
#             positionwise_layer = Conv1dLinear
#             positionwise_layer_args = (self.attention_dim, self.linear_units, self.positionwise_conv_kernel_size, self.dropout_rate)
#         else:
#             raise NotImplementedError("Support only linear or conv1d.")
#
#         self.transformers = TransformerLayer(
#             self.attention_dim,
#             MultiHeadedAttention(self.attention_heads, self.attention_dim, self.attention_dropout_rate),
#             positionwise_layer(*positionwise_layer_args),
#             self.dropout_rate,
#             self.normalize_before,
#             self.concat_after
#         )
#         if self.normalize_before:
#             self.after_norm = TLayerNorm(self.attention_dim)
#         self.out_dim = self.attention_dim
#
#
#     def forward(self, xs):
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         xs = xs.transpose(0, 1)
#         #######print('fmllr input', xs.size())
#         xs = self.linear_xs(xs)
#         #######print('transformer 들어가기 전 xs', xs.size())
#
#         for i in range(self.num_blocks):
#             xs = self.transformers(xs)
#         #######print('transformer 후0 xs', xs.size())
#         if self.normalize_before:
#             xs = self.after_norm(xs)
#         #######print('transformer 후 xs', xs.size())
#         return xs
#
# class TransformerLayer(nn.Module):
#
#     def __init__(self, size, self_attn, feed_forward, dropout_rate,
#                  normalize_before=True, concat_after=False):
#
#         super(TransformerLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.norm1 = TLayerNorm(size)
#         self.norm2 = TLayerNorm(size)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.size = size
#         self.normalize_before = normalize_before
#         self.concat_after = concat_after
#         if self.concat_after:
#             self.concat_linear = nn.Linear(size + size, size)
#
#     def forward(self, x, cache=None):
#
#         residual = x
#         #######print('transformer layer에 들어온 x',x.size())
#         if self.normalize_before:
#             x = self.norm1(x)
#
#         if cache is None:
#             x_q = x
#         else:
#             assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
#             x_q = x[:, -1:, :]
#             residual = residual[:, -1:, :]
#             mask = None if mask is None else mask[:, -1:, :]
#
#         if self.concat_after:
#             x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
#             x = residual + self.concat_linear(x_concat)
#         else:
#             x = residual + self.dropout(self.self_attn(x_q, x, x))
#         if not self.normalize_before:
#             x = self.norm1(x)
#
#         #######print('selfattention 통과한 layer x',x.size())
#         residual = x
#         if self.normalize_before:
#             x = self.norm2(x)
#         x = residual + self.dropout(self.feed_forward(x))
#         #######print('ffn layer까지 모두 통과한 x',x.size())
#         if not self.normalize_before:
#             x = self.norm2(x)
#         #######print('transformer layer에서 return 할 x', x.size())
#         if cache is not None:
#             x = torch.cat([cache, x], dim=1)
#
#         return x

#*********************************************************************************************************8

# #*******************************네번째 시도*************************
# ##########################################################
# # pytorch-kaldi v.0.1
# # transformer AM
# # 2020/3.19 (임베딩 넣고 cuda memory 때문에 layer 하나에 batch는 2/1로 함 WER 98.72. loss 안떨어짐. libri_Transformer_fmllr )
# ##########################################################
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# #import numpy as np
# import numpy
# from distutils.util import strtobool
# import math
# import json
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# def act_fun(act_type):
#     if act_type == "gelu":
#         return nn.GELU()  # 표현 이거 맞나
#
# def make_pad_mask(lengths, xs=None, length_dim=-1):
#     if length_dim == 0:
#         raise ValueError('length_dim cannot be 0: {}'.format(length_dim))
#
#     if not isinstance(lengths, list):
#         lengths = lengths.tolist()
#     bs = int(len(lengths))
#     if xs is None:
#         maxlen = int(max(lengths))
#     else:
#         maxlen = xs.size(length_dim)
#
#     seq_range = torch.arange(0, maxlen, dtype=torch.int64)
#     seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
#     seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
#     mask = seq_range_expand >= seq_length_expand
#
#     if xs is not None:
#         assert xs.size(0) == bs, (xs.size(0), bs)
#
#         if length_dim < 0:
#             length_dim = xs.dim() + length_dim
#         # ind = (:, None, ..., None, :, , None, ..., None)
#         ind = tuple(slice(None) if i in (0, length_dim) else None
#                     for i in range(xs.dim()))
#         mask = mask[ind].expand_as(xs).to(xs.device)
#     return mask
#
# def make_non_pad_mask(lengths, xs=None, length_dim= -1):
#     return ~make_pad_mask(lengths, xs, length_dim)
#
# def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
#     """padding position is set to 0, either use input_lengths or pad_idx
#     """
#     assert input_lengths is not None or pad_idx is not None
#     if input_lengths is not None:
#         # padded_input: N x T x ..
#         N = padded_input.size(0)
#         non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
#         for i in range(N):
#             non_pad_mask[i, input_lengths[i]:] = 0
#     if pad_idx is not None:
#         # padded_input: N x T
#         assert padded_input.dim() == 2
#         non_pad_mask = padded_input.ne(pad_idx).float()
#     # unsqueeze(-1) for broadcast
#     return non_pad_mask.unsqueeze(-1)
#
# def get_attn_pad_mask(padded_input, input_lengths, expand_length):
#     """mask position is set to 1"""
#     # N x Ti x 1
#     non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
#     # N x Ti, lt(1) like not operation
#     pad_mask = non_pad_mask.squeeze(-1).lt(1)
#     attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
#     return attn_mask
#
# def _pre_hook(state_dict, prefix, local_metadata, strict,
#               missing_keys, unexpected_keys, error_msgs):
#     """Perform pre-hook in load_state_dict for backward compatibility.
#     Note:
#         We saved self.pe until v.0.5.2 but we have omitted it later.
#         Therefore, we remove the item "pe" from `state_dict` for backward compatibility.
#     """
#     k = prefix + "pe"
#     if k in state_dict:
#         state_dict.pop(k)
#
# class GELU(nn.Module):
#     """
#     Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
#     """
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#
# class PositionalEncoding(torch.nn.Module):
#     """Positional encoding module until v.0.5.2."""
#
#     def __init__(self, d_model, dropout_rate, max_len=5000):
#         import math
#         super().__init__()
#         self.dropout = torch.nn.Dropout(p=0.1)
#         device = torch.device("cuda")
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).to(device)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1).to(device)
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
#                              -(math.log(10000.0) / d_model)).to(device)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.max_len = max_len
#         self.xscale = math.sqrt(d_model)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         #######print('임베딩후에 position임베딩 전의 입력값 x', x.size(), x.type())
#         #######print('self.pe[:, :x.size(1)]', self.pe[:, :x.size(1)].size(), self.pe[:, :x.size(1)].type())
#         x = x * self.xscale + self.pe[:, :x.size(1)]
#         #######print('position 임베딩 후의 입력값 x', x.size(), x.type())
#         return self.dropout(x)
#
# # class PositionalEncoding0(nn.Module):
# #     """Positional encoding.
# #     :param int d_model: embedding dim
# #     :param float dropout_rate: dropout rate
# #     :param int max_len: maximum input length
# #     """
# #     def __init__(self, d_model, dropout_rate, max_len=5000):
# #         """Construct an PositionalEncoding object."""
# #         super(PositionalEncoding, self).__init__()
# #         self.device = device
# #         self.d_model = d_model
# #         self.xscale = math.sqrt(self.d_model)
# #         self.dropout = torch.nn.Dropout(p=dropout_rate)
# #         self.pe = None
# #         self.extend_pe(torch.tensor(0.0).expand(1, max_len))
# #         self._register_load_state_dict_pre_hook(_pre_hook)
# #
# #     def extend_pe(self, x):
# #         """Reset the positional encodings."""
# #
# #         # if self.pe is not None:
# #         #     if self.pe.size(1) >= x.size(1):
# #         #         if self.pe.dtype != x.dtype or self.pe.device != x.device:
# #         #             self.pe = self.pe.to(dtype=x.dtype, device=device)
# #         #         return
# #         pe = torch.zeros(x.size(1), self.d_model)
# #         position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
# #         div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) *
# #                              -(math.log(10000.0) / self.d_model))
# #         pe[:, 0::2] = torch.sin(position * div_term)
# #         pe[:, 1::2] = torch.cos(position * div_term)
# #         pe = pe.unsqueeze(0)
# #         self.pe = pe.to(device=x.device, dtype=x.dtype)
# #
# #     def forward(self, x: torch.Tensor):
# #         """Add positional encoding.
# #         Args:
# #             x (torch.Tensor): Input. Its shape is (batch, time, ...)
# #         Returns:
# #             torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
# #         """
# #
# #         self.extend_pe(x)
# #         x = x * self.xscale + self.pe[:, :x.size(1)]
# #         return self.dropout(x)
#
# class PositionwiseFeedForward(torch.nn.Module):
#     """Positionwise feed forward layer.
#
#     :param int idim: input dimenstion
#     :param int hidden_units: number of hidden units
#     :param float dropout_rate: dropout rate
#
#     """
#
#     def __init__(self, idim, hidden_units, dropout_rate):
#         """Construct an PositionwiseFeedForward object."""
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = torch.nn.Linear(idim, hidden_units)
#         self.w_2 = torch.nn.Linear(hidden_units, idim)
#         self.dropout = torch.nn.Dropout(dropout_rate)
#         self.activation = GELU()
#
#     def forward(self, x):
#         """Forward funciton."""
#         #######print('FFN input x', x.size())
#         x=self.w_1(x)
#         #######print('ffn에서 linear 하나 통과', x.size())
#         x = self.activation(x)
#         #######print('gelu통과한',x.size())
#         x = self.w_2(self.dropout(x))
#         #######print('ffn최종 통과한 ', x.size())
#         return x #self.w_2(self.dropout(self.activation(self.w_1(x))))
#
# class TLayerNorm(torch.nn.LayerNorm):
#     """Layer normalization module.
#     :param int nout: output dim size
#     :param int dim: dimension to be normalized
#     """
#     def __init__(self, nout, dim=-1):
#         """Construct an TLayerNorm object."""
#         super(TLayerNorm, self).__init__(nout, eps=1e-12)
#         self.dim = dim
#     def forward(self, x):
#         """Apply layer normalization.
#         :param torch.Tensor x: input tensor
#         :return: layer normalized tensor
#         :rtype torch.Tensor
#         """
#         #######print('TLayerNorm input x', x.size())
#         if self.dim == -1:
#             return super(TLayerNorm, self).forward(x)
#         return super(TLayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)
#
# class MultiSequential(torch.nn.Sequential):
#     """Multi-input multi-output torch.nn.Sequential."""
#     def forward(self, *args):
#         """Repeat."""
#         for m in self:
#             args = m(*args)
#         return args
# def repeat(N, fn):
#     """Repeat module N times.
#     :param int N: repeat time
#     :param function fn: function to generate module
#     :return: repeated modules
#     :rtype: MultiSequential
#     """
#     return MultiSequential(*[fn() for _ in range(N)])
#
# class MultiHeadedAttention(nn.Module):
#     """Multi-Head Attention layer.
#
#     :param int n_head: the number of head s
#     :param int n_feat: the number of features
#     :param float dropout_rate: dropout rate
#
#     """
#     def __init__(self, n_head, n_feat, dropout_rate):
#         """Construct an MultiHeadedAttention object."""
#         super(MultiHeadedAttention, self).__init__()
#         assert n_feat % n_head == 0
#         # We assume d_v always equals d_k
#         self.d_k = n_feat // n_head
#         self.h = n_head
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         self.linear_k = nn.Linear(n_feat, n_feat)
#         self.linear_v = nn.Linear(n_feat, n_feat)
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout_rate)
#
#     def forward(self, query, key, value): #forward(self, query, key, value, mask):
#         print('query size', query.size())
#         print('key size', key.size())
#         #print('mask size', mask.size())
#         """Compute 'Scaled Dot Product Attention'.
#
#         :param torch.Tensor query: (batch, time1, size)
#         :param torch.Tensor key: (batch, time2, size)
#         :param torch.Tensor value: (batch, time2, size)
#         :param torch.Tensor mask: (batch, time1, time2)
#         :param torch.nn.Dropout dropout:
#         :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
#              weighted by the query dot key attention (batch, head, time1, time2)
#         """
#         n_batch = query.size(0)
#         q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
#         k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
#         v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
#         print('q size', q.size())
#         print('k size', k.size())
#
#         q = q.transpose(1, 2)  # (batch, head, time1, d_k)
#         k = k.transpose(1, 2)  # (batch, head, time2, d_k)
#         v = v.transpose(1, 2)  # (batch, head, time2, d_k)
#         print('transpose q size', q.size())
#         print('transpose k size', k.size())
#
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
#         # if mask is not None:
#         #     mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
#         #     min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
#         #     scores = scores.masked_fill(mask, min_value)
#         #     self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
#         # else:
#         self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
#
#         p_attn = self.dropout(self.attn)
#         x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
#         x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
#         return self.linear_out(x)  # (batch, time1, d_model)
#
# class Transformer(nn.Module):
#     """Transformer encoder module.
#
#     :param int idim: input dim
#     :param int attention_dim: dimention of attention
#     :param int attention_heads: the number of heads of multi head attention
#     :param int linear_units: the number of units of position-wise feed forward
#     :param int num_blocks: the number of decoder blocks
#     :param float dropout_rate: dropout rate
#     :param float attention_dropout_rate: dropout rate in attention
#     :param float positional_dropout_rate: dropout rate after adding positional encoding
#     :param str or torch.nn.Module input_layer: input layer type
#     :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
#     :param bool normalize_before: whether to use layer_norm before the first block
#     :param bool concat_after: whether to concat attention layer's input and output
#         if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
#         if False, no additional linear will be applied. i.e. x -> x + att(x)
#     :param str positionwise_layer_type: linear of conv1d
#     :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
#     :param int padding_idx: padding_idx for input_layer=embed
#     """
#
#     def __init__(self, options, idim):
#         """Construct an Encoder object."""
#         super(Transformer, self).__init__()
#         self.attention_dim = 256
#         self.attention_heads = 4
#         self.linear_units = 2048
#         self.num_blocks = 1
#         self.dropout_rate = 0.1
#         self.positional_dropout_rate = 0.1
#         self.attention_dropout_rate = 0.2
#         self.input_layer = None
#         self.normalize_before = True
#         self.concat_after = False
#         self.positionwise_layer_type = "linear"
#         self.positionwise_conv_kernel_size = 1
#         self.padding_idx = -1
#         self.linear_xs = nn.Linear(idim, self.attention_dim)
#         print('idim', idim)
#         if self.input_layer == "linear":
#             self.embed = torch.nn.Sequential(
#                 torch.nn.Linear(idim, self.attention_dim),
#                 torch.nn.LayerNorm(self.attention_dim),
#                 torch.nn.Dropout(self.dropout_rate),
#                 torch.nn.ReLU(),
#                 PositionalEncoding(self.attention_dim, self.positional_dropout_rate)
#             )
#         elif self.input_layer == "conv2d":
#             self.embed = Conv2dSubsampling(idim, self.attention_dim, self.dropout_rate)
#         elif self.input_layer == 'vgg2l':
#             self.embed = VGG2L(idim, self.attention_dim)
#         elif self.input_layer == "embed":
#             self.embed = torch.nn.Sequential(
#                 torch.nn.Embedding(idim, self.attention_dim, padding_idx=self.padding_idx),
#                 PositionalEncoding(self.attention_dim, self.positional_dropout_rate, device)
#             )
#         elif isinstance(self.input_layer, torch.nn.Module):
#             self.embed = torch.nn.Sequential(
#                 self.input_layer,
#                 PositionalEncoding(self.attention_dim, self.positional_dropout_rate),
#             )
#         elif self.input_layer is None:
#             self.embed = torch.nn.Sequential(
#                 PositionalEncoding(self.attention_dim, self.positional_dropout_rate)
#             )
#        # else:
#        #     raise ValueError("unknown input_layer: " + self.input_layer)
#         self.normalize_before = self.normalize_before
#         if self.positionwise_layer_type == "linear":
#             positionwise_layer = PositionwiseFeedForward
#             positionwise_layer_args = (self.attention_dim, self.linear_units, self.dropout_rate)
#         elif self.positionwise_layer_type == "conv1d":
#             positionwise_layer = MultiLayeredConv1d
#             positionwise_layer_args = (self.attention_dim, self.linear_units, self.positionwise_conv_kernel_size, self.dropout_rate)
#         elif self.positionwise_layer_type == "conv1d-linear":
#             positionwise_layer = Conv1dLinear
#             positionwise_layer_args = (self.attention_dim, self.linear_units, self.positionwise_conv_kernel_size, self.dropout_rate)
#         else:
#             raise NotImplementedError("Support only linear or conv1d.")
#         # self.transformers = repeat(
#         #     self.num_blocks,
#         #     lambda: TransformerLayer(
#         #         self.attention_dim,
#         #         MultiHeadedAttention(self.attention_heads, self.attention_dim, self.attention_dropout_rate),
#         #         positionwise_layer(*positionwise_layer_args),
#         #         self.dropout_rate,
#         #         self.normalize_before,
#         #         self.concat_after
#         #     )
#         # )
#         self.transformers = TransformerLayer(
#             self.attention_dim,
#             MultiHeadedAttention(self.attention_heads, self.attention_dim, self.attention_dropout_rate),
#             positionwise_layer(*positionwise_layer_args),
#             self.dropout_rate,
#             self.normalize_before,
#             self.concat_after
#         )
#         if self.normalize_before:
#             self.after_norm = TLayerNorm(self.attention_dim)
#         self.out_dim = self.attention_dim
#
#
#     def forward(self, xs):
#         """Encode input sequence.
#         :param torch.Tensor xs: input tensor
#         :param torch.Tensor masks: input mask
#         :return: position embedded tensor and mask
#         :rtype Tuple[torch.Tensor, torch.Tensor]:
#         """
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         xs = xs.transpose(0, 1)
#         print('fmllr input', xs.size())
#
#         xs = self.linear_xs(xs) # 40을 512로 바꾼거
#         #print('linear하고 임베딩 전', xs.size())
#         ########non_pad_mask = get_non_pad_mask(xs, input_lengths=None)
#         ########length = padded_input.size(1)
#         ########slf_attn_mask = get_attn_pad_mask(xs, None, length)
#
#         xs = xs.long()
#         xs = self.embed(xs).to(device)
#         print('임베딩 후 transformer 들어가기 전 xs', xs.size())
#         #print('transformer 들어가기 전 masks', masks.size())
#         #xs = self.transformers(xs) #xs, masks = self.transformers(xs, mask=None)
#         for i in range(self.num_blocks):
#             xs = self.transformers(xs)
#         print('transformer 후0 xs', xs.size())
#         if self.normalize_before:
#             xs = self.after_norm(xs)
#         print('transformer 후 xs', xs.size())
#         return xs
#
# class TransformerLayer(nn.Module):
#     """Encoder layer module.
#     :param int size: input dim
#     :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
#     :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
#         feed forward module
#     :param float dropout_rate: dropout rate
#     :param bool normalize_before: whether to use layer_norm before the first block
#     :param bool concat_after: whether to concat attention layer's input and output
#         if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
#         if False, no additional linear will be applied. i.e. x -> x + att(x)
#     """
#     def __init__(self, size, self_attn, feed_forward, dropout_rate,
#                  normalize_before=True, concat_after=False):
#         """Construct an EncoderLayer object."""
#         super(TransformerLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.norm1 = TLayerNorm(size)
#         self.norm2 = TLayerNorm(size)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.size = size
#         self.normalize_before = normalize_before
#         self.concat_after = concat_after
#         if self.concat_after:
#             self.concat_linear = nn.Linear(size + size, size)
#
#     def forward(self, x, cache=None):
#         """Compute encoded features.
#
#         :param torch.Tensor x: encoded source features (batch, max_time_in, size)
#         :param torch.Tensor mask: mask for x (batch, max_time_in)
#         :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
#         :rtype: Tuple[torch.Tensor, torch.Tensor]
#         """
#         residual = x
#         #######print('transformer layer에 들어온 x',x.size())
#         if self.normalize_before:
#             x = self.norm1(x)
#
#         if cache is None:
#             x_q = x
#         else:
#             assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
#             x_q = x[:, -1:, :]
#             residual = residual[:, -1:, :]
#             mask = None if mask is None else mask[:, -1:, :]
#
#         if self.concat_after:
#             x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
#             x = residual + self.concat_linear(x_concat)
#         else:
#             x = residual + self.dropout(self.self_attn(x_q, x, x))
#         if not self.normalize_before:
#             x = self.norm1(x)
#
#         #######print('selfattention 통과한 layer x',x.size())
#         residual = x
#         if self.normalize_before:
#             x = self.norm2(x)
#         x = residual + self.dropout(self.feed_forward(x))
#         #######print('ffn layer까지 모두 통과한 x',x.size())
#         if not self.normalize_before:
#             x = self.norm2(x)
#         #######print('transformer layer에서 return 할 x', x.size())
#         if cache is not None:
#             x = torch.cat([cache, x], dim=1)
#
#         return x
#
# #********************************************************

# #*******************************다섯번째 시도*************************
# ##########################################################
# # pytorch-kaldi v.0.1
# # transformer AM
# # 2020/3.23 (임베딩 넣고 mask넣을거임 libri_Transformer5_fmllr ) >코드는 다 돌아 학습이 안될뿐
# ##########################################################
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# #import numpy as np
# import numpy
# from distutils.util import strtobool
# import math
# import json
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
#     """padding position is set to 0, either use input_lengths or pad_idx
#     """
#     assert input_lengths is not None or pad_idx is not None
#     if input_lengths is not None:
#         # padded_input: N x T x ..
#         N = padded_input.size(0)
#         non_pad_mask = padded_input.new_zeros(padded_input.size()[:-1])  # N x T ones>zeros
#         for i in range(N):
#             non_pad_mask[i, input_lengths[i]:] = 1 #0>1
#     if pad_idx is not None:
#         # padded_input: N x T
#         assert padded_input.dim() == 2
#         non_pad_mask = padded_input.ne(pad_idx).float()
#     # unsqueeze(-1) for broadcast
#     return non_pad_mask.unsqueeze(-1)
#
# def get_attn_pad_mask(padded_input, input_lengths, expand_length):
#     """mask position is set to 1"""
#     # N x Ti x 1
#     non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
#     # N x Ti, lt(1) like not operation
#     pad_mask = non_pad_mask.squeeze(-1).lt(1)
#     attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
#     return attn_mask
#
# def _pre_hook(state_dict, prefix, local_metadata, strict,
#               missing_keys, unexpected_keys, error_msgs):
#     """Perform pre-hook in load_state_dict for backward compatibility.
#     Note:
#         We saved self.pe until v.0.5.2 but we have omitted it later.
#         Therefore, we remove the item "pe" from `state_dict` for backward compatibility.
#     """
#     k = prefix + "pe"
#     if k in state_dict:
#         state_dict.pop(k)
#
# class GELU(nn.Module):
#     """
#     Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
#     """
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
#
# class Conv2dSubsampling(torch.nn.Module):
#     """Convolutional 2D subsampling (to 1/4 length).
#
#     :param int idim: input dim
#     :param int odim: output dim
#     :param flaot dropout_rate: dropout rate
#
#     """
#
#     def __init__(self, idim, odim, dropout_rate):
#         """Construct an Conv2dSubsampling object."""
#         super(Conv2dSubsampling, self).__init__()
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(1, odim, 3, 2),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(odim, odim, 3, 2),
#             torch.nn.ReLU()
#         )
#         self.out = torch.nn.Sequential(
#             torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
#             PositionalEncoding(odim, dropout_rate)
#         )
#
#     def forward(self, x, x_mask):
#         print('conv 전 x', x.size())
#         print('conv 전 x_mask', x_mask.size())
#         x = x.unsqueeze(1)  # (b, c, t, f)
#         print('convvvv x', x.size())
#         x = self.conv(x)
#         b, c, t, f = x.size()
#         x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
#         if x_mask is None:
#             return x, None
#         print('이거뭐지', x_mask[:, :-2:2, :-2:2][:, :-2:2, :-2:2], x_mask[:, :-2:2, :-2:2][:, :-2:2, :-2:2].size())
#         return x, x_mask[:, :-2:2, :-2:2][:, :-2:2, :-2:2] # x_mask[:, :, :-2:2][:, :, :-2:2]
#
# class PositionalEncoding(torch.nn.Module):
#     """Positional encoding module until v.0.5.2."""
#
#     def __init__(self, d_model, dropout_rate, max_len=5000):
#         import math
#         super().__init__()
#         self.dropout = torch.nn.Dropout(p=0.1)
#         device = torch.device("cuda")
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).to(device)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1).to(device)
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
#                              -(math.log(10000.0) / d_model)).to(device)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.max_len = max_len
#         self.xscale = math.sqrt(d_model)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         print('임베딩후에 position임베딩 전의 입력값 x', x.size(), x.type())
#         print('self.pe[:, :x.size(1)]', self.pe[:, :x.size(1)].size(), self.pe[:, :x.size(1)].type())
#         x = x * self.xscale + self.pe[:, :x.size(1)]
#         print('position 임베딩 후의 입력값 x', x.size(), x.type())
#         return self.dropout(x)
#
# class PositionwiseFeedForward(torch.nn.Module):
#
#     def __init__(self, idim, hidden_units, dropout_rate):
#         """Construct an PositionwiseFeedForward object."""
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = torch.nn.Linear(idim, hidden_units)
#         self.w_2 = torch.nn.Linear(hidden_units, idim)
#         self.dropout = torch.nn.Dropout(dropout_rate)
#         self.activation = GELU()
#         #self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
# #self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1) 이걸로 해보자 버트에서 쓰는거래
#
#     def forward(self, x):
#         """Forward funciton."""
#         #######print('FFN input x', x.size())
#         x=self.w_1(x)
#         #######print('ffn에서 linear 하나 통과', x.size())
#         x = self.activation(x)
#         #######print('gelu통과한',x.size())
#         x = self.w_2(self.dropout(x))
#         #######print('ffn최종 통과한 ', x.size())
#         return x #self.w_2(self.dropout(self.activation(self.w_1(x))))
#
# class TLayerNorm(torch.nn.LayerNorm):
#     def __init__(self, nout, dim=-1):
#         """Construct an TLayerNorm object."""
#         super(TLayerNorm, self).__init__(nout, eps=1e-12)
#         self.dim = dim
#     def forward(self, x):
#         #######print('TLayerNorm input x', x.size())
#         if self.dim == -1:
#             return super(TLayerNorm, self).forward(x)
#         return super(TLayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)
#
# class MultiSequential(torch.nn.Sequential):
#     """Multi-input multi-output torch.nn.Sequential."""
#     def forward(self, *args):
#         """Repeat."""
#         for m in self:
#             args = m(*args)
#         return args
# def repeat(N, fn):
#     return MultiSequential(*[fn() for _ in range(N)])
#
# class MultiHeadedAttention(nn.Module):
#     def __init__(self, n_head, n_feat, dropout_rate):
#         """Construct an MultiHeadedAttention object."""
#         super(MultiHeadedAttention, self).__init__()
#         assert n_feat % n_head == 0
#         # We assume d_v always equals d_k
#         self.d_k = n_feat // n_head
#         self.h = n_head
#         self.linear_q = nn.Linear(n_feat, n_feat)
#         self.linear_k = nn.Linear(n_feat, n_feat)
#         self.linear_v = nn.Linear(n_feat, n_feat)
#         self.linear_out = nn.Linear(n_feat, n_feat)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout_rate)
#
#     def forward(self, query, key, value, mask): #forward(self, query, key, value, mask):
#         print('query size', query.size())
#         print('key size', key.size())
#         print('mask size', mask.size())
#         """Compute 'Scaled Dot Product Attention'.
#
#         :param torch.Tensor query: (batch, time1, size)
#         :param torch.Tensor key: (batch, time2, size)
#         :param torch.Tensor value: (batch, time2, size)
#         :param torch.Tensor mask: (batch, time1, time2)
#         :param torch.nn.Dropout dropout:
#         :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
#              weighted by the query dot key attention (batch, head, time1, time2)
#         """
#         n_batch = query.size(0)
#         q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
#         k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
#         v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
#         print('q size', q.size())
#         print('k size', k.size())
#
#         q = q.transpose(1, 2)  # (batch, head, time1, d_k)
#         k = k.transpose(1, 2)  # (batch, head, time2, d_k)
#         v = v.transpose(1, 2)  # (batch, head, time2, d_k)
#         print('transpose q size', q.size())
#         print('transpose k size', k.size())
#
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
#         if mask is not None:
#             mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
#             min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
#             scores = scores.masked_fill(mask, min_value)
#             self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
#         else:
#             self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
#
#         p_attn = self.dropout(self.attn)
#         x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
#         x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
#         return self.linear_out(x)  # (batch, time1, d_model)
#
# class Transformer5(nn.Module):
#
#     def __init__(self, options, idim):
#         """Construct an Encoder object."""
#         super(Transformer5, self).__init__()
#         self.attention_dim = 512 #512
#         self.attention_heads = 8
#         self.linear_units = 2048 #2048
#         self.num_blocks = 8
#         self.dropout_rate = 0.1
#         self.positional_dropout_rate = 0.1
#         self.attention_dropout_rate = 0.2
#         self.input_layer = None
#         self.normalize_before = True
#         self.concat_after = False
#         self.positionwise_layer_type = "linear"
#         self.positionwise_conv_kernel_size = 1
#         self.padding_idx = -1
#         self.linear_xs = nn.Linear(idim, self.attention_dim)
#         self.linear_xss = nn.Linear(self.attention_dim, self.attention_dim*2)
#         print('idim', idim)
#         if self.input_layer == "linear":
#             self.embed = torch.nn.Sequential(
#                 torch.nn.Linear(idim, self.attention_dim),
#                 torch.nn.LayerNorm(self.attention_dim),
#                 torch.nn.Dropout(self.dropout_rate),
#                 torch.nn.ReLU(),
#                 PositionalEncoding(self.attention_dim, self.positional_dropout_rate)
#             )
#         elif self.input_layer == "conv2d":
#             self.embed = Conv2dSubsampling(idim, self.attention_dim, self.dropout_rate)
#         elif self.input_layer == 'vgg2l':
#             self.embed = VGG2L(idim, self.attention_dim)
#         elif self.input_layer == "embed":
#             self.embed = torch.nn.Sequential(
#                 torch.nn.Embedding(idim, self.attention_dim, padding_idx=self.padding_idx),
#                 PositionalEncoding(self.attention_dim, self.positional_dropout_rate, device)
#             )
#         elif isinstance(self.input_layer, torch.nn.Module):
#             self.embed = torch.nn.Sequential(
#                 self.input_layer,
#                 PositionalEncoding(self.attention_dim, self.positional_dropout_rate),
#             )
#         elif self.input_layer is None:
#             self.embed = torch.nn.Sequential(
#                 PositionalEncoding(self.attention_dim, self.positional_dropout_rate)
#             )
#        # else:
#        #     raise ValueError("unknown input_layer: " + self.input_layer)
#         self.normalize_before = self.normalize_before
#         if self.positionwise_layer_type == "linear":
#             positionwise_layer = PositionwiseFeedForward
#             positionwise_layer_args = (self.attention_dim, self.linear_units, self.dropout_rate)
#         elif self.positionwise_layer_type == "conv1d":
#             positionwise_layer = MultiLayeredConv1d
#             positionwise_layer_args = (self.attention_dim, self.linear_units, self.positionwise_conv_kernel_size, self.dropout_rate)
#         elif self.positionwise_layer_type == "conv1d-linear":
#             positionwise_layer = Conv1dLinear
#             positionwise_layer_args = (self.attention_dim, self.linear_units, self.positionwise_conv_kernel_size, self.dropout_rate)
#         else:
#             raise NotImplementedError("Support only linear or conv1d.")
#         # self.transformers = repeat(
#         #     self.num_blocks,
#         #     lambda: TransformerLayer(
#         #         self.attention_dim,
#         #         MultiHeadedAttention(self.attention_heads, self.attention_dim, self.attention_dropout_rate),
#         #         positionwise_layer(*positionwise_layer_args),
#         #         self.dropout_rate,
#         #         self.normalize_before,
#         #         self.concat_after
#         #     )
#         # )
#         self.transformers = TransformerLayer(
#             self.attention_dim,
#             MultiHeadedAttention(self.attention_heads, self.attention_dim, self.attention_dropout_rate),
#             positionwise_layer(*positionwise_layer_args),
#             self.dropout_rate,
#             self.normalize_before,
#             self.concat_after
#         )
#         if self.normalize_before:
#             self.after_norm = TLayerNorm(self.attention_dim)
#         #self.out_dim = self.attention_dim
#         self.dnn_lay = list(map(int, options["dnn_lay"].split(",")))
#         self.dnn_drop = list(map(float, options["dnn_drop"].split(",")))
#         self.dnn_use_batchnorm = list(map(strtobool, options["dnn_use_batchnorm"].split(",")))
#         self.dnn_use_laynorm = list(map(strtobool, options["dnn_use_laynorm"].split(",")))
#         self.dnn_use_laynorm_inp = strtobool(options["dnn_use_laynorm_inp"])
#         self.dnn_use_batchnorm_inp = strtobool(options["dnn_use_batchnorm_inp"])
#         self.dnn_act = options["dnn_act"].split(",")
#
#         self.wx = nn.ModuleList([])
#         self.bn = nn.ModuleList([])
#         self.ln = nn.ModuleList([])
#         self.act = nn.ModuleList([])
#         self.drop = nn.ModuleList([])
#
#         # input layer normalization
#         if self.dnn_use_laynorm_inp:
#             self.ln0 = LayerNorm(self.input_dim)
#
#         # input batch normalization
#         if self.dnn_use_batchnorm_inp:
#             self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)
#
#         self.N_dnn_lay = len(self.dnn_lay)
#
#         current_input = self.attention_dim*2
#
#         # Initialization of hidden layers
#
#         for i in range(self.N_dnn_lay):
#
#             # dropout
#             self.drop.append(nn.Dropout(p=self.dnn_drop[i]))
#
#             # activation
#             self.act.append(act_fun(self.dnn_act[i]))
#
#             add_bias = True
#
#             # layer norm initialization
#             self.ln.append(LayerNorm(self.dnn_lay[i]))
#             self.bn.append(nn.BatchNorm1d(self.dnn_lay[i], momentum=0.05))
#
#             if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
#                 add_bias = False
#
#             # Linear operations
#             self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=add_bias))
#
#             # weight initialization
#             self.wx[i].weight = torch.nn.Parameter(
#                 torch.Tensor(self.dnn_lay[i], current_input).uniform_(
#                     -np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
#                     np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
#                 )
#             )
#             self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))
#
#             current_input = self.dnn_lay[i]
#
#         self.out_dim = current_input
#
#
#     def forward(self, xs):
#         """Encode input sequence.
#         :param torch.Tensor xs: input tensor
#         :param torch.Tensor masks: input mask
#         :return: position embedded tensor and mask
#         :rtype Tuple[torch.Tensor, torch.Tensor]:
#         """
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         xs = xs.transpose(0, 1)
#         print('fmllr input', xs.size())
#
#         xs = self.linear_xs(xs) # 40을 512로 바꾼거
#         print('linear하고 임베딩 전', xs.size())
#         xs_len = int(xs.size(1))
#         input_lengths = torch.tensor([xs_len,xs_len]) # batch 2
#         #input_lengths = torch.tensor([xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len, xs_len])
#         non_pad_mask = get_non_pad_mask(xs, input_lengths)
#         length = xs.size(1)
#         slf_attn_mask = get_attn_pad_mask(xs, input_lengths, length)
#         print('생서ㅇ된 mask',slf_attn_mask,slf_attn_mask.size())
#         xs = xs.long() #pos emb만 할때
#         xs = self.embed(xs).to(device) #pos emb만 할때
#         #xs, mask = self.embed(xs, slf_attn_mask) #conv embedding 할때
#         print('임베딩 후 transformer 들어가기 전 xs', xs.size() )
#         #print('transformer 들어가기 전 masks', masks.size())
#         #xs  = self.transformers(xs, slf_attn_mask) #xs, masks = self.transformers(xs, mask=None)
#         for i in range(self.num_blocks):
#             xs = self.transformers(xs, slf_attn_mask)
#         print('transformer 후0 xs', xs.size())
#         if self.normalize_before:
#             xs = self.after_norm(xs)
#         print('transformer 후 xs', xs.size())
#         xs = self.linear_xss(xs)
#
#         print('xss 512>1024', xs.size())
#         xs = xs.reshape(-1,self.attention_dim*2)
#         print('MLP입력값 x', xs.size())
#         # Applying Layer/Batch Norm
#         if bool(self.dnn_use_laynorm_inp):
#             xs = self.ln0((xs))
#
#         if bool(self.dnn_use_batchnorm_inp):
#
#             xs = self.bn0((xs))
#
#         for i in range(self.N_dnn_lay):
#
#             if self.dnn_use_laynorm[i] and not (self.dnn_use_batchnorm[i]):
#                 xs = self.drop[i](self.act[i](self.ln[i](self.wx[i](xs))))
#
#             if self.dnn_use_batchnorm[i] and not (self.dnn_use_laynorm[i]):
#                 xs = self.drop[i](self.act[i](self.bn[i](self.wx[i](xs))))
#
#             if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i] == True:
#                 xs = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](xs)))))
#
#             if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i] == False:
#                 xs = self.drop[i](self.act[i](self.wx[i](xs)))
#             print('MLP출력값 x', xs.size())
#         return xs
#
# class TransformerLayer(nn.Module):
#     """Encoder layer module.
#     :param int size: input dim
#     :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
#     :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
#         feed forward module
#     :param float dropout_rate: dropout rate
#     :param bool normalize_before: whether to use layer_norm before the first block
#     :param bool concat_after: whether to concat attention layer's input and output
#         if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
#         if False, no additional linear will be applied. i.e. x -> x + att(x)
#     """
#     def __init__(self, size, self_attn, feed_forward, dropout_rate,
#                  normalize_before=True, concat_after=False):
#         """Construct an EncoderLayer object."""
#         super(TransformerLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.norm1 = TLayerNorm(size)
#         self.norm2 = TLayerNorm(size)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.size = size
#         self.normalize_before = normalize_before
#         self.concat_after = concat_after
#         if self.concat_after:
#             self.concat_linear = nn.Linear(size + size, size)
#
#     def forward(self, x, mask, cache=None):
#         """Compute encoded features.
#
#         :param torch.Tensor x: encoded source features (batch, max_time_in, size)
#         :param torch.Tensor mask: mask for x (batch, max_time_in)
#         :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
#         :rtype: Tuple[torch.Tensor, torch.Tensor]
#         """
#         residual = x
#         print('transformer layer에 들어온 x',x.size())
#         if self.normalize_before:
#             x = self.norm1(x)
#
#         if cache is None:
#             x_q = x
#         else:
#             assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
#             x_q = x[:, -1:, :]
#             residual = residual[:, -1:, :]
#             mask = None if mask is None else mask[:, -1:, :]
#
#         if self.concat_after:
#             x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
#             x = residual + self.concat_linear(x_concat)
#         else:
#             x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
#         if not self.normalize_before:
#             x = self.norm1(x)
#
#         print('selfattention 통과한 layer x',x.size())
#         residual = x
#         if self.normalize_before:
#             x = self.norm2(x)
#         x = residual + self.dropout(self.feed_forward(x))
#         #######print('ffn layer까지 모두 통과한 x',x.size())
#         if not self.normalize_before:
#             x = self.norm2(x)
#         #######print('transformer layer에서 return 할 x', x.size())
#         if cache is not None:
#             x = torch.cat([cache, x], dim=1)
#
#         return x
#
# #********************************************************
#*******************************libri960h*************************
##########################################################
# pytorch-kaldi v.0.1
# transformer AM
# 2020/4.1 (리브리960시간으로 테스트 transformer1로 이름 붙였고 모델은 5랑 같아 )
##########################################################
import torch
import torch.nn.functional as F
import torch.nn as nn
#import numpy as np
import numpy
from distutils.util import strtobool
import math
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_zeros(padded_input.size()[:-1])  # N x T ones>zeros
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 1 #0>1
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask

def _pre_hook(state_dict, prefix, local_metadata, strict,
              missing_keys, unexpected_keys, error_msgs):

    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Conv2dSubsampling(torch.nn.Module):

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):

        x = x.unsqueeze(1)  # (b, c, t, f)

        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :-2:2, :-2:2][:, :-2:2, :-2:2] # x_mask[:, :, :-2:2][:, :, :-2:2]

class PositionalEncoding(torch.nn.Module):
    """Positional encoding module until v.0.5.2."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
        import math
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1)
        device = torch.device("cuda")
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.max_len = max_len
        self.xscale = math.sqrt(d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x * self.xscale + self.pe[:, :x.size(1)]

        return self.dropout(x)

class PositionwiseFeedForward(torch.nn.Module):

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = GELU()
        #self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1) 이걸로 해보자 버트에서 쓰는거래

    def forward(self, x):
        """Forward funciton."""

        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class TLayerNorm(torch.nn.LayerNorm):
    def __init__(self, nout, dim=-1):
        """Construct an TLayerNorm object."""
        super(TLayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim
    def forward(self, x):

        if self.dim == -1:
            return super(TLayerNorm, self).forward(x)
        return super(TLayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)

class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential."""
    def forward(self, *args):
        """Repeat."""
        for m in self:
            args = m(*args)
        return args
def repeat(N, fn):
    return MultiSequential(*[fn() for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask): #forward(self, query, key, value, mask):

        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

class Transformer1(nn.Module):

    def __init__(self, options, idim):
        """Construct an Encoder object."""
        super(Transformer1, self).__init__()
        self.attention_dim = 512 #512
        self.attention_heads = 8
        self.linear_units = 2048 #2048
        self.num_blocks = 6
        self.dropout_rate = 0.1
        self.positional_dropout_rate = 0.1
        self.attention_dropout_rate = 0.2
        self.input_layer = None
        self.normalize_before = True
        self.concat_after = False
        self.positionwise_layer_type = "linear"
        self.positionwise_conv_kernel_size = 1
        self.padding_idx = -1
        self.linear_xs = nn.Linear(idim, self.attention_dim)
        self.linear_xss = nn.Linear(self.attention_dim, self.attention_dim*2)
        print('idim', idim)
        if self.input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, self.attention_dim),
                torch.nn.LayerNorm(self.attention_dim),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.ReLU(),
                PositionalEncoding(self.attention_dim, self.positional_dropout_rate)
            )
        elif self.input_layer == "conv2d":
            self.embed = Conv2dSubsampling(idim, self.attention_dim, self.dropout_rate)
        elif self.input_layer == 'vgg2l':
            self.embed = VGG2L(idim, self.attention_dim)
        elif self.input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, self.attention_dim, padding_idx=self.padding_idx),
                PositionalEncoding(self.attention_dim, self.positional_dropout_rate, device)
            )
        elif isinstance(self.input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                self.input_layer,
                PositionalEncoding(self.attention_dim, self.positional_dropout_rate),
            )
        elif self.input_layer is None:
            self.embed = torch.nn.Sequential(
                PositionalEncoding(self.attention_dim, self.positional_dropout_rate)
            )
       # else:
       #     raise ValueError("unknown input_layer: " + self.input_layer)
        self.normalize_before = self.normalize_before
        if self.positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (self.attention_dim, self.linear_units, self.dropout_rate)
        elif self.positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (self.attention_dim, self.linear_units, self.positionwise_conv_kernel_size, self.dropout_rate)
        elif self.positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (self.attention_dim, self.linear_units, self.positionwise_conv_kernel_size, self.dropout_rate)
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        self.transformers = TransformerLayer(
            self.attention_dim,
            MultiHeadedAttention(self.attention_heads, self.attention_dim, self.attention_dropout_rate),
            positionwise_layer(*positionwise_layer_args),
            self.dropout_rate,
            self.normalize_before,
            self.concat_after
        )
        if self.normalize_before:
            self.after_norm = TLayerNorm(self.attention_dim)
        #self.out_dim = self.attention_dim
        self.dnn_lay = list(map(int, options["dnn_lay"].split(",")))
        self.dnn_drop = list(map(float, options["dnn_drop"].split(",")))
        self.dnn_use_batchnorm = list(map(strtobool, options["dnn_use_batchnorm"].split(",")))
        self.dnn_use_laynorm = list(map(strtobool, options["dnn_use_laynorm"].split(",")))
        self.dnn_use_laynorm_inp = strtobool(options["dnn_use_laynorm_inp"])
        self.dnn_use_batchnorm_inp = strtobool(options["dnn_use_batchnorm_inp"])
        self.dnn_act = options["dnn_act"].split(",")

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.dnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.dnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_dnn_lay = len(self.dnn_lay)

        current_input = self.attention_dim*2

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.dnn_drop[i]))

            # activation
            self.act.append(act_fun(self.dnn_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.dnn_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.dnn_lay[i], momentum=0.05))

            if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.dnn_lay[i], current_input).uniform_(
                    -np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                    np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                )
            )
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))

            current_input = self.dnn_lay[i]

        self.out_dim = current_input


    def forward(self, xs):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xs = xs.transpose(0, 1)
        print('fmllr input', xs.size())

        xs = self.linear_xs(xs) # 40을 512로 바꾼거

        xs_len = int(xs.size(1))
        input_lengths = torch.tensor([xs_len,xs_len]) # batch 2

        non_pad_mask = get_non_pad_mask(xs, input_lengths)
        length = xs.size(1)
        slf_attn_mask = get_attn_pad_mask(xs, input_lengths, length)

        xs = xs.long() #pos emb만 할때
        xs = self.embed(xs).to(device) #pos emb만 할때

        for i in range(self.num_blocks):
            xs = self.transformers(xs, slf_attn_mask)

        if self.normalize_before:
            xs = self.after_norm(xs)

        xs = self.linear_xss(xs)


        xs = xs.reshape(-1,self.attention_dim*2)

        # Applying Layer/Batch Norm
        if bool(self.dnn_use_laynorm_inp):
            xs = self.ln0((xs))

        if bool(self.dnn_use_batchnorm_inp):

            xs = self.bn0((xs))

        for i in range(self.N_dnn_lay):

            if self.dnn_use_laynorm[i] and not (self.dnn_use_batchnorm[i]):
                xs = self.drop[i](self.act[i](self.ln[i](self.wx[i](xs))))

            if self.dnn_use_batchnorm[i] and not (self.dnn_use_laynorm[i]):
                xs = self.drop[i](self.act[i](self.bn[i](self.wx[i](xs))))

            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i] == True:
                xs = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](xs)))))

            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i] == False:
                xs = self.drop[i](self.act[i](self.wx[i](xs)))

        return xs

class TransformerLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout_rate,
                 normalize_before=True, concat_after=False):
        """Construct an EncoderLayer object."""
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = TLayerNorm(size)
        self.norm2 = TLayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x, mask, cache=None):

        residual = x

        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)


        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x

#********************************************************

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        print('self.gamma', self.gamma.size())
        print('self.beta', self.beta.size())
        #print('self.eps', self.eps.size())

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def act_fun(act_type):

    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!

    if act_type == "gelu":
        return nn.GELU()  # 표현 이거 맞나

class MLP(nn.Module):
    def __init__(self, options, inp_dim):
        super(MLP, self).__init__()

        self.input_dim = inp_dim
        print('MLP에서의 self.input_dim', self.input_dim)
        self.dnn_lay = list(map(int, options["dnn_lay"].split(",")))
        self.dnn_drop = list(map(float, options["dnn_drop"].split(",")))
        self.dnn_use_batchnorm = list(map(strtobool, options["dnn_use_batchnorm"].split(",")))
        self.dnn_use_laynorm = list(map(strtobool, options["dnn_use_laynorm"].split(",")))
        self.dnn_use_laynorm_inp = strtobool(options["dnn_use_laynorm_inp"])
        self.dnn_use_batchnorm_inp = strtobool(options["dnn_use_batchnorm_inp"])
        self.dnn_act = options["dnn_act"].split(",")

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.dnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.dnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_dnn_lay = len(self.dnn_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.dnn_drop[i]))

            # activation
            self.act.append(act_fun(self.dnn_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.dnn_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.dnn_lay[i], momentum=0.05))

            if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.dnn_lay[i], current_input).uniform_(
                    -np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                    np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                )
            )
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))

            current_input = self.dnn_lay[i]

        self.out_dim = current_input

    def forward(self, x):
        print('MLP입력값 x', x.size())
        # Applying Layer/Batch Norm
        if bool(self.dnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.dnn_use_batchnorm_inp):

            x = self.bn0((x))

        for i in range(self.N_dnn_lay):

            if self.dnn_use_laynorm[i] and not (self.dnn_use_batchnorm[i]):
                x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] and not (self.dnn_use_laynorm[i]):
                x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](x)))))

            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](self.wx[i](x)))
            print('MLP출력값 x', x.size())
        return x


class LSTM_cudnn(nn.Module):
    def __init__(self, options, inp_dim):
        super(LSTM_cudnn, self).__init__()

        self.input_dim = inp_dim
        self.hidden_size = int(options["hidden_size"])
        self.num_layers = int(options["num_layers"])
        self.bias = bool(strtobool(options["bias"]))
        self.batch_first = bool(strtobool(options["batch_first"]))
        self.dropout = float(options["dropout"])
        self.bidirectional = bool(strtobool(options["bidirectional"]))

        self.lstm = nn.ModuleList(
            [
                nn.LSTM(
                    self.input_dim,
                    self.hidden_size,
                    self.num_layers,
                    bias=self.bias,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            ]
        )

        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
            c0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)

        if x.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        output, (hn, cn) = self.lstm[0](x, (h0, c0))

        return output


class GRU_cudnn(nn.Module):
    def __init__(self, options, inp_dim):
        super(GRU_cudnn, self).__init__()

        self.input_dim = inp_dim
        self.hidden_size = int(options["hidden_size"])
        self.num_layers = int(options["num_layers"])
        self.bias = bool(strtobool(options["bias"]))
        self.batch_first = bool(strtobool(options["batch_first"]))
        self.dropout = float(options["dropout"])
        self.bidirectional = bool(strtobool(options["bidirectional"]))

        self.gru = nn.ModuleList(
            [
                nn.GRU(
                    self.input_dim,
                    self.hidden_size,
                    self.num_layers,
                    bias=self.bias,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            ]
        )

        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)

        if x.is_cuda:
            h0 = h0.cuda()

        output, hn = self.gru[0](x, h0)

        return output


class RNN_cudnn(nn.Module):
    def __init__(self, options, inp_dim):
        super(RNN_cudnn, self).__init__()

        self.input_dim = inp_dim
        self.hidden_size = int(options["hidden_size"])
        self.num_layers = int(options["num_layers"])
        self.nonlinearity = options["nonlinearity"]
        self.bias = bool(strtobool(options["bias"]))
        self.batch_first = bool(strtobool(options["batch_first"]))
        self.dropout = float(options["dropout"])
        self.bidirectional = bool(strtobool(options["bidirectional"]))

        self.rnn = nn.ModuleList(
            [
                nn.RNN(
                    self.input_dim,
                    self.hidden_size,
                    self.num_layers,
                    nonlinearity=self.nonlinearity,
                    bias=self.bias,
                    dropout=self.dropout,
                    bidirectional=self.bidirectional,
                )
            ]
        )

        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)

        if x.is_cuda:
            h0 = h0.cuda()

        output, hn = self.rnn[0](x, h0)

        return output


class LSTM(nn.Module):
    def __init__(self, options, inp_dim):
        super(LSTM, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        print('self.input_dim', self.input_dim)
        self.lstm_lay = list(map(int, options["lstm_lay"].split(",")))
        self.lstm_drop = list(map(float, options["lstm_drop"].split(",")))
        self.lstm_use_batchnorm = list(map(strtobool, options["lstm_use_batchnorm"].split(",")))
        self.lstm_use_laynorm = list(map(strtobool, options["lstm_use_laynorm"].split(",")))
        self.lstm_use_laynorm_inp = strtobool(options["lstm_use_laynorm_inp"])
        self.lstm_use_batchnorm_inp = strtobool(options["lstm_use_batchnorm_inp"])
        self.lstm_act = options["lstm_act"].split(",")
        self.lstm_orthinit = strtobool(options["lstm_orthinit"])

        self.bidir = strtobool(options["lstm_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wfx = nn.ModuleList([])  # Forget
        self.ufh = nn.ModuleList([])  # Forget

        self.wix = nn.ModuleList([])  # Input
        self.uih = nn.ModuleList([])  # Input

        self.wox = nn.ModuleList([])  # Output
        self.uoh = nn.ModuleList([])  # Output

        self.wcx = nn.ModuleList([])  # Cell state
        self.uch = nn.ModuleList([])  # Cell state

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wfx = nn.ModuleList([])  # Batch Norm
        self.bn_wix = nn.ModuleList([])  # Batch Norm
        self.bn_wox = nn.ModuleList([])  # Batch Norm
        self.bn_wcx = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.lstm_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.lstm_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_lstm_lay = len(self.lstm_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_lstm_lay):

            # Activations
            self.act.append(act_fun(self.lstm_act[i]))

            add_bias = True

            if self.lstm_use_laynorm[i] or self.lstm_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wfx.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wix.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wox.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wcx.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))

            # Recurrent connections
            self.ufh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            self.uih.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            self.uoh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            self.uch.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))

            if self.lstm_orthinit:
                nn.init.orthogonal_(self.ufh[i].weight)
                nn.init.orthogonal_(self.uih[i].weight)
                nn.init.orthogonal_(self.uoh[i].weight)
                nn.init.orthogonal_(self.uch[i].weight)

            # batch norm initialization
            self.bn_wfx.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wix.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wox.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wcx.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.lstm_lay[i]))

            if self.bidir:
                current_input = 2 * self.lstm_lay[i]
            else:
                current_input = self.lstm_lay[i]

        self.out_dim = self.lstm_lay[i] + self.bidir * self.lstm_lay[i]

    def forward(self, x):
        print('입력x', x.size())
        # Applying Layer/Batch Norm
        if bool(self.lstm_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.lstm_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_lstm_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.lstm_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
                print('flip x',x.size())
            else:
                h_init = torch.zeros(x.shape[1], self.lstm_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.lstm_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.lstm_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wfx_out = self.wfx[i](x)
            wix_out = self.wix[i](x)
            wox_out = self.wox[i](x)
            wcx_out = self.wcx[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.lstm_use_batchnorm[i]:

                wfx_out_bn = self.bn_wfx[i](wfx_out.view(wfx_out.shape[0] * wfx_out.shape[1], wfx_out.shape[2]))
                wfx_out = wfx_out_bn.view(wfx_out.shape[0], wfx_out.shape[1], wfx_out.shape[2])

                wix_out_bn = self.bn_wix[i](wix_out.view(wix_out.shape[0] * wix_out.shape[1], wix_out.shape[2]))
                wix_out = wix_out_bn.view(wix_out.shape[0], wix_out.shape[1], wix_out.shape[2])

                wox_out_bn = self.bn_wox[i](wox_out.view(wox_out.shape[0] * wox_out.shape[1], wox_out.shape[2]))
                wox_out = wox_out_bn.view(wox_out.shape[0], wox_out.shape[1], wox_out.shape[2])

                wcx_out_bn = self.bn_wcx[i](wcx_out.view(wcx_out.shape[0] * wcx_out.shape[1], wcx_out.shape[2]))
                wcx_out = wcx_out_bn.view(wcx_out.shape[0], wcx_out.shape[1], wcx_out.shape[2])

            # Processing time steps
            hiddens = []
            ct = h_init
            ht = h_init

            for k in range(x.shape[0]):

                # LSTM equations
                ft = torch.sigmoid(wfx_out[k] + self.ufh[i](ht))
                it = torch.sigmoid(wix_out[k] + self.uih[i](ht))
                ot = torch.sigmoid(wox_out[k] + self.uoh[i](ht))
                ct = it * self.act[i](wcx_out[k] + self.uch[i](ht)) * drop_mask + ft * ct
                ht = ot * self.act[i](ct)

                if self.lstm_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h
            print('중간중간 output x', x.size())
        print('최종 output x', x.size())
        return x


class GRU(nn.Module):
    def __init__(self, options, inp_dim):
        super(GRU, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.gru_lay = list(map(int, options["gru_lay"].split(",")))
        self.gru_drop = list(map(float, options["gru_drop"].split(",")))
        self.gru_use_batchnorm = list(map(strtobool, options["gru_use_batchnorm"].split(",")))
        self.gru_use_laynorm = list(map(strtobool, options["gru_use_laynorm"].split(",")))
        self.gru_use_laynorm_inp = strtobool(options["gru_use_laynorm_inp"])
        self.gru_use_batchnorm_inp = strtobool(options["gru_use_batchnorm_inp"])
        self.gru_orthinit = strtobool(options["gru_orthinit"])
        self.gru_act = options["gru_act"].split(",")
        self.bidir = strtobool(options["gru_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.wr = nn.ModuleList([])  # Reset Gate
        self.ur = nn.ModuleList([])  # Reset Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm
        self.bn_wr = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.gru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.gru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_gru_lay = len(self.gru_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_gru_lay):

            # Activations
            self.act.append(act_fun(self.gru_act[i]))

            add_bias = True

            if self.gru_use_laynorm[i] or self.gru_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))
            self.wr.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))
            self.ur.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))

            if self.gru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
                nn.init.orthogonal_(self.ur[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))
            self.bn_wr.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.gru_lay[i]))

            if self.bidir:
                current_input = 2 * self.gru_lay[i]
            else:
                current_input = self.gru_lay[i]

        self.out_dim = self.gru_lay[i] + self.bidir * self.gru_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.gru_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.gru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_gru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.gru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.gru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.gru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.gru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            wr_out = self.wr[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.gru_use_batchnorm[i]:

                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

                wr_out_bn = self.bn_wr[i](wr_out.view(wr_out.shape[0] * wr_out.shape[1], wr_out.shape[2]))
                wr_out = wr_out_bn.view(wr_out.shape[0], wr_out.shape[1], wr_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # gru equation
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                rt = torch.sigmoid(wr_out[k] + self.ur[i](ht))
                at = wh_out[k] + self.uh[i](rt * ht)
                hcand = self.act[i](at) * drop_mask   # hcand=gt
                ht = zt * ht + (1 - zt) * hcand

                if self.gru_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x


class logMelFb(nn.Module):
    def __init__(self, options, inp_dim):
        super(logMelFb, self).__init__()
        import torchaudio

        self._sample_rate = int(options["logmelfb_nr_sample_rate"])
        self._nr_of_filters = int(options["logmelfb_nr_filt"])
        self._stft_window_size = int(options["logmelfb_stft_window_size"])
        self._stft_window_shift = int(options["logmelfb_stft_window_shift"])
        self._use_cuda = strtobool(options["use_cuda"])
        self.out_dim = self._nr_of_filters
        self._mspec = torchaudio.transforms.MelSpectrogram(
            sr=self._sample_rate,
            n_fft=self._stft_window_size,
            ws=self._stft_window_size,
            hop=self._stft_window_shift,
            n_mels=self._nr_of_filters,
        )

    def forward(self, x):
        def _safe_log(inp, epsilon=1e-20):
            eps = torch.FloatTensor([epsilon])
            if self._use_cuda:
                eps = eps.cuda()
            log_inp = torch.log10(torch.max(inp, eps.expand_as(inp)))
            return log_inp

        assert x.shape[-1] == 1, "Multi channel time signal processing not suppored yet"
        x_reshape_for_stft = torch.squeeze(x, -1).transpose(0, 1)
        if self._use_cuda:
            window = self._mspec.window(self._stft_window_size).cuda()
        else:
            window = self._mspec.window(self._stft_window_size)
        x_stft = torch.stft(
            x_reshape_for_stft, self._stft_window_size, hop_length=self._stft_window_shift, center=False, window=window
        )
        x_power_stft = x_stft.pow(2).sum(-1)
        x_power_stft_reshape_for_filterbank_mult = x_power_stft.transpose(1, 2)
        mel_spec = self._mspec.fm(x_power_stft_reshape_for_filterbank_mult).transpose(0, 1)
        log_mel_spec = _safe_log(mel_spec)
        out = log_mel_spec
        return out


class channel_averaging(nn.Module):
    def __init__(self, options, inp_dim):
        super(channel_averaging, self).__init__()
        self._use_cuda = strtobool(options["use_cuda"])
        channel_weights = [float(e) for e in options["chAvg_channelWeights"].split(",")]
        self._nr_of_channels = len(channel_weights)
        np_weights = np.asarray(channel_weights, dtype=np.float32) * 1.0 / np.sum(channel_weights)
        self._weights = torch.from_np(np_weights)
        if self._use_cuda:
            self._weights = self._weights.cuda()
        self.out_dim = 1

    def forward(self, x):
        assert self._nr_of_channels == x.shape[-1]
        out = torch.einsum("tbc,c->tb", x, self._weights).unsqueeze(-1)
        return out


class liGRU(nn.Module):
    def __init__(self, options, inp_dim):
        super(liGRU, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.ligru_lay = list(map(int, options["ligru_lay"].split(",")))
        self.ligru_drop = list(map(float, options["ligru_drop"].split(",")))
        self.ligru_use_batchnorm = list(map(strtobool, options["ligru_use_batchnorm"].split(",")))
        self.ligru_use_laynorm = list(map(strtobool, options["ligru_use_laynorm"].split(",")))
        self.ligru_use_laynorm_inp = strtobool(options["ligru_use_laynorm_inp"])
        self.ligru_use_batchnorm_inp = strtobool(options["ligru_use_batchnorm_inp"])
        self.ligru_orthinit = strtobool(options["ligru_orthinit"])
        self.ligru_act = options["ligru_act"].split(",")
        self.bidir = strtobool(options["ligru_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.ligru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.ligru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_ligru_lay = len(self.ligru_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_ligru_lay):

            # Activations
            self.act.append(act_fun(self.ligru_act[i]))

            add_bias = True

            if self.ligru_use_laynorm[i] or self.ligru_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))

            if self.ligru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.ligru_lay[i]))

            if self.bidir:
                current_input = 2 * self.ligru_lay[i]
            else:
                current_input = self.ligru_lay[i]

        self.out_dim = self.ligru_lay[i] + self.bidir * self.ligru_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.ligru_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.ligru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_ligru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.ligru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.ligru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(
                    torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.ligru_drop[i])
                )
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.ligru_use_batchnorm[i]:

                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # ligru equation
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                at = wh_out[k] + self.uh[i](ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand

                if self.ligru_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x


class minimalGRU(nn.Module):
    def __init__(self, options, inp_dim):
        super(minimalGRU, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.minimalgru_lay = list(map(int, options["minimalgru_lay"].split(",")))
        self.minimalgru_drop = list(map(float, options["minimalgru_drop"].split(",")))
        self.minimalgru_use_batchnorm = list(map(strtobool, options["minimalgru_use_batchnorm"].split(",")))
        self.minimalgru_use_laynorm = list(map(strtobool, options["minimalgru_use_laynorm"].split(",")))
        self.minimalgru_use_laynorm_inp = strtobool(options["minimalgru_use_laynorm_inp"])
        self.minimalgru_use_batchnorm_inp = strtobool(options["minimalgru_use_batchnorm_inp"])
        self.minimalgru_orthinit = strtobool(options["minimalgru_orthinit"])
        self.minimalgru_act = options["minimalgru_act"].split(",")
        self.bidir = strtobool(options["minimalgru_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.wz = nn.ModuleList([])  # Update Gate
        self.uz = nn.ModuleList([])  # Update Gate

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm
        self.bn_wz = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.minimalgru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.minimalgru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_minimalgru_lay = len(self.minimalgru_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_minimalgru_lay):

            # Activations
            self.act.append(act_fun(self.minimalgru_act[i]))

            add_bias = True

            if self.minimalgru_use_laynorm[i] or self.minimalgru_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.minimalgru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.minimalgru_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.minimalgru_lay[i], self.minimalgru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.minimalgru_lay[i], self.minimalgru_lay[i], bias=False))

            if self.minimalgru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.minimalgru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.minimalgru_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.minimalgru_lay[i]))

            if self.bidir:
                current_input = 2 * self.minimalgru_lay[i]
            else:
                current_input = self.minimalgru_lay[i]

        self.out_dim = self.minimalgru_lay[i] + self.bidir * self.minimalgru_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.minimalgru_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.minimalgru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])

        for i in range(self.N_minimalgru_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.minimalgru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.minimalgru_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(
                    torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.minimalgru_drop[i])
                )
            else:
                drop_mask = torch.FloatTensor([1 - self.minimalgru_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.minimalgru_use_batchnorm[i]:

                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # minimalgru equation
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                at = wh_out[k] + self.uh[i](zt * ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand

                if self.minimalgru_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h

        return x


class RNN(nn.Module):
    def __init__(self, options, inp_dim):
        super(RNN, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        print('self.input_dim', self.input_dim)
        self.rnn_lay = list(map(int, options["rnn_lay"].split(",")))
        self.rnn_drop = list(map(float, options["rnn_drop"].split(",")))
        self.rnn_use_batchnorm = list(map(strtobool, options["rnn_use_batchnorm"].split(",")))
        self.rnn_use_laynorm = list(map(strtobool, options["rnn_use_laynorm"].split(",")))
        self.rnn_use_laynorm_inp = strtobool(options["rnn_use_laynorm_inp"])
        self.rnn_use_batchnorm_inp = strtobool(options["rnn_use_batchnorm_inp"])
        self.rnn_orthinit = strtobool(options["rnn_orthinit"])
        self.rnn_act = options["rnn_act"].split(",")
        self.bidir = strtobool(options["rnn_bidir"])
        self.use_cuda = strtobool(options["use_cuda"])
        self.to_do = options["to_do"]

        if self.to_do == "train":
            self.test_flag = False
        else:
            self.test_flag = True

        # List initialization
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])

        self.ln = nn.ModuleList([])  # Layer Norm
        self.bn_wh = nn.ModuleList([])  # Batch Norm

        self.act = nn.ModuleList([])  # Activations

        # Input layer normalization
        if self.rnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # Input batch normalization
        if self.rnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_rnn_lay = len(self.rnn_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_rnn_lay):

            # Activations
            self.act.append(act_fun(self.rnn_act[i]))

            add_bias = True

            if self.rnn_use_laynorm[i] or self.rnn_use_batchnorm[i]:
                add_bias = False

            # Feed-forward connections
            self.wh.append(nn.Linear(current_input, self.rnn_lay[i], bias=add_bias))

            # Recurrent connections
            self.uh.append(nn.Linear(self.rnn_lay[i], self.rnn_lay[i], bias=False))

            if self.rnn_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)

            # batch norm initialization
            self.bn_wh.append(nn.BatchNorm1d(self.rnn_lay[i], momentum=0.05))

            self.ln.append(LayerNorm(self.rnn_lay[i]))

            if self.bidir:
                current_input = 2 * self.rnn_lay[i]
            else:
                current_input = self.rnn_lay[i]

        self.out_dim = self.rnn_lay[i] + self.bidir * self.rnn_lay[i]

    def forward(self, x):
        print('입력값 x', x.size())
        # Applying Layer/Batch Norm
        if bool(self.rnn_use_laynorm_inp):
            x = self.ln0((x))
            print('self.ln0((x))', x.size())

        if bool(self.rnn_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])
            print('self.bn.view((x))', x.size())

        for i in range(self.N_rnn_lay):

            # Initial state and concatenation
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.rnn_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.rnn_lay[i])

            # Drop mask initilization (same mask for all time steps)
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.rnn_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.rnn_drop[i]])

            if self.use_cuda:
                h_init = h_init.cuda()
                drop_mask = drop_mask.cuda()

            # Feed-forward affine transformations (all steps in parallel)
            wh_out = self.wh[i](x)

            # Apply batch norm if needed (all steos in parallel)
            if self.rnn_use_batchnorm[i]:

                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])

            # Processing time steps
            hiddens = []
            ht = h_init

            for k in range(x.shape[0]):

                # rnn equation
                at = wh_out[k] + self.uh[i](ht)
                ht = self.act[i](at) * drop_mask

                if self.rnn_use_laynorm[i]:
                    ht = self.ln[i](ht)

                hiddens.append(ht)

            # Stacking hidden states
            h = torch.stack(hiddens)

            # Bidirectional concatenations
            if self.bidir:
                h_f = h[:, 0 : int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2) : x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)

            # Setup x for the next hidden layer
            x = h
            print('최종 output', x.size())

        return x


class CNN(nn.Module):
    def __init__(self, options, inp_dim):
        super(CNN, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.cnn_N_filt = list(map(int, options["cnn_N_filt"].split(",")))

        self.cnn_len_filt = list(map(int, options["cnn_len_filt"].split(",")))
        self.cnn_max_pool_len = list(map(int, options["cnn_max_pool_len"].split(",")))

        self.cnn_act = options["cnn_act"].split(",")
        self.cnn_drop = list(map(float, options["cnn_drop"].split(",")))

        self.cnn_use_laynorm = list(map(strtobool, options["cnn_use_laynorm"].split(",")))
        self.cnn_use_batchnorm = list(map(strtobool, options["cnn_use_batchnorm"].split(",")))
        self.cnn_use_laynorm_inp = strtobool(options["cnn_use_laynorm_inp"])
        self.cnn_use_batchnorm_inp = strtobool(options["cnn_use_batchnorm_inp"])

        self.N_cnn_lay = len(self.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):

            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm([N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])])
            )

            self.bn.append(
                nn.BatchNorm1d(
                    N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]), momentum=0.05
                )
            )

            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filt, len_filt))

            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])

        self.out_dim = current_input * N_filt

    def forward(self, x):

        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):

            if self.cnn_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

        x = x.view(batch, -1)

        return x


class SincNet(nn.Module):
    def __init__(self, options, inp_dim):
        super(SincNet, self).__init__()

        # Reading parameters
        self.input_dim = inp_dim
        self.sinc_N_filt = list(map(int, options["sinc_N_filt"].split(",")))

        self.sinc_len_filt = list(map(int, options["sinc_len_filt"].split(",")))
        self.sinc_max_pool_len = list(map(int, options["sinc_max_pool_len"].split(",")))

        self.sinc_act = options["sinc_act"].split(",")
        self.sinc_drop = list(map(float, options["sinc_drop"].split(",")))

        self.sinc_use_laynorm = list(map(strtobool, options["sinc_use_laynorm"].split(",")))
        self.sinc_use_batchnorm = list(map(strtobool, options["sinc_use_batchnorm"].split(",")))
        self.sinc_use_laynorm_inp = strtobool(options["sinc_use_laynorm_inp"])
        self.sinc_use_batchnorm_inp = strtobool(options["sinc_use_batchnorm_inp"])

        self.N_sinc_lay = len(self.sinc_N_filt)

        self.sinc_sample_rate = int(options["sinc_sample_rate"])
        self.sinc_min_low_hz = int(options["sinc_min_low_hz"])
        self.sinc_min_band_hz = int(options["sinc_min_band_hz"])

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.sinc_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.sinc_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_sinc_lay):

            N_filt = int(self.sinc_N_filt[i])
            len_filt = int(self.sinc_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.sinc_drop[i]))

            # activation
            self.act.append(act_fun(self.sinc_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm([N_filt, int((current_input - self.sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])])
            )

            self.bn.append(
                nn.BatchNorm1d(
                    N_filt, int((current_input - self.sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i]), momentum=0.05
                )
            )

            if i == 0:
                self.conv.append(
                    SincConv(
                        1,
                        N_filt,
                        len_filt,
                        sample_rate=self.sinc_sample_rate,
                        min_low_hz=self.sinc_min_low_hz,
                        min_band_hz=self.sinc_min_band_hz,
                    )
                )

            else:
                self.conv.append(nn.Conv1d(self.sinc_N_filt[i - 1], self.sinc_N_filt[i], self.sinc_len_filt[i]))

            current_input = int((current_input - self.sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])

        self.out_dim = current_input * N_filt

    def forward(self, x):

        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.sinc_use_laynorm_inp):
            x = self.ln0(x)

        if bool(self.sinc_use_batchnorm_inp):
            x = self.bn0(x)

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_sinc_lay):

            if self.sinc_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i]))))

            if self.sinc_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i]))))

            if self.sinc_use_batchnorm[i] == False and self.sinc_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i])))

        x = x.view(batch, -1)

        return x


class SincConv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):

        super(SincConv, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n + 1).view(1, -1) / self.sample_rate

    def sinc(self, x):
        # Numerically stable definition
        x_left = x[:, 0 : int((x.shape[1] - 1) / 2)]
        y_left = torch.sin(x_left) / x_left
        y_right = torch.flip(y_left, dims=[1])

        sinc = torch.cat([y_left, torch.ones([x.shape[0], 1]).to(x.device), y_right], dim=1)

        return sinc

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz / self.sample_rate + torch.abs(self.band_hz_)

        f_times_t = torch.matmul(low, self.n_)

        low_pass1 = 2 * low * self.sinc(2 * math.pi * f_times_t * self.sample_rate)

        f_times_t = torch.matmul(high, self.n_)
        low_pass2 = 2 * high * self.sinc(2 * math.pi * f_times_t * self.sample_rate)

        band_pass = low_pass2 - low_pass1
        max_, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_

        self.filters = (band_pass * self.window_).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :, getattr(torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda])().long(), :
    ]
    return x.view(xsize)


class SRU(nn.Module):
    def __init__(self, options, inp_dim):
        super(SRU, self).__init__()
        self.input_dim = inp_dim
        self.hidden_size = int(options["sru_hidden_size"])
        self.num_layers = int(options["sru_num_layers"])
        self.dropout = float(options["sru_dropout"])
        self.rnn_dropout = float(options["sru_rnn_dropout"])
        self.use_tanh = bool(strtobool(options["sru_use_tanh"]))
        self.use_relu = bool(strtobool(options["sru_use_relu"]))
        self.use_selu = bool(strtobool(options["sru_use_selu"]))
        self.weight_norm = bool(strtobool(options["sru_weight_norm"]))
        self.layer_norm = bool(strtobool(options["sru_layer_norm"]))
        self.bidirectional = bool(strtobool(options["sru_bidirectional"]))
        self.is_input_normalized = bool(strtobool(options["sru_is_input_normalized"]))
        self.has_skip_term = bool(strtobool(options["sru_has_skip_term"]))
        self.rescale = bool(strtobool(options["sru_rescale"]))
        self.highway_bias = float(options["sru_highway_bias"])
        self.n_proj = int(options["sru_n_proj"])
        self.sru = sru.SRU(
            self.input_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            rnn_dropout=self.rnn_dropout,
            bidirectional=self.bidirectional,
            n_proj=self.n_proj,
            use_tanh=self.use_tanh,
            use_selu=self.use_selu,
            use_relu=self.use_relu,
            weight_norm=self.weight_norm,
            layer_norm=self.layer_norm,
            has_skip_term=self.has_skip_term,
            is_input_normalized=self.is_input_normalized,
            highway_bias=self.highway_bias,
            rescale=self.rescale,
        )
        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size * 2)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
        if x.is_cuda:
            h0 = h0.cuda()
        output, hn = self.sru(x, c0=h0)
        return output


class PASE(nn.Module):
    def __init__(self, options, inp_dim):
        super(PASE, self).__init__()

        # To use PASE within PyTorch-Kaldi, please clone the current PASE repository: https://github.com/santi-pdp/pase
        # Note that you have to clone the dev branch.
        # Take a look into the requirements (requirements.txt) and install in your environment what is missing. An important requirement is QRNN (https://github.com/salesforce/pytorch-qrnn).
        # Before starting working with PASE, it could make sense to a quick test  with QRNN independently (see “usage” section in the QRNN repository).
        # Remember to install pase. This way it can be used outside the pase folder directory.  To do it, go into the pase folder and type:
        # "python setup.py install"

        from pase.models.frontend import wf_builder

        self.input_dim = inp_dim
        self.pase_cfg = options["pase_cfg"]
        self.pase_model = options["pase_model"]

        self.pase = wf_builder(self.pase_cfg)

        self.pase.load_pretrained(self.pase_model, load_last=True, verbose=True)

        # Reading the out_dim from the config file:
        with open(self.pase_cfg) as json_file:
            config = json.load(json_file)

        self.out_dim = int(config["emb_dim"])

    def forward(self, x):

        x = x.unsqueeze(0).unsqueeze(0)
        output = self.pase(x)

        return output
