import torch
from torch import nn
from modules.feature_map import elu_feature_map
from fastNLP.modules.decoder.seq2seq_state import TransformerState

class RecurrentCrossLinearAttention(nn.Module):
    """Implement autoregressive linear cross attention as a recurrent
    module.
    See fast_transformers.attention.linear_attention.LinearAttention .
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, d_query, feature_map=None, eps=1e-6):
        super(RecurrentCrossLinearAttention, self).__init__()
        self.feature_map = (
            feature_map(d_query) if feature_map else
            elu_feature_map(d_query)
        )
        self.eps = eps

    def forward(self, query, key, value, key_mask=None, info=None):
        if info is None:
            self.feature_map.new_feature_map()

        Q = self.feature_map.forward_queries(query)

        if info is None:
            K = self.feature_map.forward_keys(key)
            if key_mask is not None:
                _key_mask = key_mask[:, :, None, None].bool()
                K = K.masked_fill(_key_mask, 0.0)
            S = torch.einsum("bsnd,bsnm->bnmd", K, value)
            Z = K.sum(dim=1)
        else:
            S, Z = info

        QZ = 1 / (torch.einsum("bnd,bnd->bn", Q, Z) + self.eps)
        V = torch.einsum("bnd,bnmd,bn->bnm", Q, S, QZ)

        return V.contiguous(), [S, Z]


class RecurrentCrossAttentionLayer(nn.Module):
    """See fast_transformers.attention.attention_layer.AttentionLayer .
    The differences with the aforementioned module as well as the
    RecurrentAttentionLayer are that this module projects the query every time
    and the keys and values only the first time they are provided.
    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
    """
    def __init__(self, d_model, n_head, layer_idx):
        super(RecurrentCrossAttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.layer_idx = layer_idx
        assert d_model % n_head == 0, "d_model should be divisible by n_head"
        self.head_dim = d_model // n_head
        self.scaling = self.head_dim ** -0.5

        self.cross_attention = RecurrentCrossLinearAttention(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def reset_params(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, querys, keys, values, key_mask=None, info=None, state=None):
        assert keys.size() == values.size()
        if state is not None:
            assert self.layer_idx is not None

        q = self.q_proj(querys)
        q *= self.scaling
        k = v = None
        # k = self.k_proj(keys)
        # v = self.v_proj(values)

        if isinstance(state, TransformerState):
            k = state.encoder_key[self.layer_idx]
            v = state.encoder_value[self.layer_idx]

        if k is None:
            k = self.k_proj(keys)
            v = self.v_proj(values)

        if isinstance(state, TransformerState):
            state.encoder_key[self.layer_idx] = k
            state.encoder_value[self.layer_idx] = v

        batch_size, length_q, d_model = querys.size()
        length_k, length_v = keys.size(1), values.size(1)

        Q = q.reshape(batch_size, length_q, self.n_head, self.head_dim)
        K = k.reshape(batch_size, length_k, self.n_head, self.head_dim)
        V = v.reshape(batch_size, length_v, self.n_head, self.head_dim)

        output = []
        for i in range(length_q):
            Qi = Q[:, i, :, :]
            Vi, info = self.cross_attention(Qi, K, V, key_mask, info)
            Vi = Vi.reshape(batch_size, 1, self.n_head, self.head_dim)
            output.append(Vi)

        output = torch.cat(output, dim=1)
        output = output.reshape(batch_size, length_q, -1)
        output = self.o_proj(output)

        return output

