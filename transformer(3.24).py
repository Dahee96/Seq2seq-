#*******************************다섯번째 시도*************************
##########################################################
# pytorch-kaldi v.0.1
# transformer AM
# 2020/3.23 
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

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


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
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
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

class Transformer5(nn.Module):

    def __init__(self, options, idim):
        """Construct an Encoder object."""
        super(Transformer5, self).__init__()
        self.attention_dim = 512
        self.attention_heads = 8
        self.linear_units = 2048
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
            
        # transformer 후에 MLP 바로 이어 붙인 구조
        
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

        current_input = self.attention_dim

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
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xs = xs.transpose(0, 1) #[Batch, Length, Dimension]

        xs = self.linear_xs(xs) # Dimension 40을 512로 바꾼거
       
        # mask 생성 [Batch]
        xs_len = int(xs.size(1))
        input_lengths = torch.tensor([xs_len,xs_len])
        non_pad_mask = get_non_pad_mask(xs, input_lengths)
        length = xs.size(1)
        slf_attn_mask = get_attn_pad_mask(xs, input_lengths, length)
        
        xs = xs.long() 
        xs = self.embed(xs).to(device) #positional encoding 해줌
        
        # transformer layer (MHA > FFN) 구문 돌리는 부분
        for i in range(self.num_blocks):
            xs = self.transformers(xs, slf_attn_mask)
       
        if self.normalize_before:
            xs = self.after_norm(xs)
        
        # MLP layer
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
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
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


