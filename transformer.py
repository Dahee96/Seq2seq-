#*******************************새로짜는 부분 두번째 시도*************************************
class PositionalEncoding2(torch.nn.Module):
    """Positional encoding module until v.0.5.2."""

    def __init__(self, d_model, dropout_rate, max_len=5000):
        import math
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.max_len = max_len
        self.xscale = math.sqrt(d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.
    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

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

    def forward(self, query, key, value, mask):
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

class FullyconnectedFeedForward(torch.nn.Module):
    """ linear>gelu>dropout>linear"""

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(FullyconnectedFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(nn.GELU(self.w_1(x)))) # gelu activation 함수 이렇게 불러오면 되나


class Transformer(nn.Module):
    def __init__(self, options, inp_dim):
        super(Transformer, self).__init__()
        self.input_dim = inp_dim
        print('inputdim', self.input_dim)  #40

        self.attention_dim = 512
        self.positional_dropout_rate = 0.1
        self.dropout_rate = 0.1
        self.padding_idx = -1

        self.dropout = torch.nn.Dropout(p=0.1)


        self.pre_embed = torch.nn.Embedding(self.input_dim, self.attention_dim, padding_idx=self.padding_idx)
        self.embed = PositionalEncoding2(self.attention_dim, self.positional_dropout_rate)

        self.transformer_lay = list(map(int, options["transformer_lay"].split(",")))  #transformer_lay = 550,550,550,550,550,550,550,550,550,550,550,550,550,550,550,550
        self.N_transformer_lay = len(self.transformer_lay)
        self.layers = nn.ModuleList([TransformerLayer()
                                     for _ in range(self.N_transformer_lay)])




        self.ln = LayerNorm(self.input_dim)
        self.out_dim = self.attention_dim

    def forward(self, xs): #forward(self, xs, masks):

        print('fmllr 입력값', xs.size(), xs.type()) #[16,40] # [batch, input_dim]

        xs = xs.long()
        xs = self.pre_embed(xs)
        xs = self.embed(xs)
        #xs=[batch size, input dim, emb_dim], [16,40,512]
        for layer in self.layers:
            xs = layer(xs)
        return xs

class TransformerLayer(nn.Module):
    def __init__(self):
        super(TransformerLayer, self).__init__()

        self.attention_dim = 512
        self.attention_heads = 8
        self.linear_units = 2048
        self.attention_dropout_rate = 0.0
        self.dropout_rate = 0.1

        self.dropout = torch.nn.Dropout(p=0.1)
        self.xs_mask = None
        self.ln = LayerNorm(self.attention_dim)
        self.out_dim = self.attention_dim

        self.MHA = MultiHeadedAttention(self.attention_heads, self.attention_dim, self.attention_dropout_rate)
        self.FFN = FullyconnectedFeedForward(self.attention_dim, self.linear_units, self.dropout_rate)


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, xs):
        if self.xs_mask is None or self.xs_mask.size(0) != len(xs):
            device = xs.device
            masks = self._generate_square_subsequent_mask(len(xs)).to(device)
            self.xs_mask = masks

        residual1 = xs
        xs = self.ln(xs)
        sub_x = residual1 + self.dropout(self.MHA(xs, xs, xs, self.xs_mask))
        residual2 = sub_x
        sub_x = self.ln(sub_x)
        out_x = residual2 + self.dropout(self.FFN(sub_x))
        xs = self.ln(out_x)
        return xs
        
        
     ################error#####################
     RuntimeError: CUDA error: device-side assert triggered
