import torch
from torch import nn
from utils import check_state
from .feature_map import elu_feature_map
from fastNLP.modules.decoder.seq2seq_state import TransformerState

class RecurrentLinearAttention(nn.Module):
    """Implement fast_transformers.attention.causal_linear_attention as a
    fixed-dimensional state recurrent model.
    See fast_transformers.attention.linear_attention and
    fast_transformers.attention.causal_linear_attention for the general concept
    of replacing the softmax with feature maps.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, d_query, feature_map=None, eps=1e-6):
        super(RecurrentLinearAttention, self).__init__()
        self.feature_map = (
            feature_map(d_query) if feature_map else
            elu_feature_map(d_query)
        )
        self.eps = eps

    def forward(self, query, key, value, info=None):
        # state = check_state(state, memroy)

        if info is None:
            self.feature_map.new_feature_map()

        Q = self.feature_map.forward_queries(query)
        K = self.feature_map.forward_keys(key)

        B, N, D = Q.shape
        _, _, M = value.shape

        if info is None:
            Si = query.new_zeros((B, N, D, M))
            Zi = query.new_zeros((B, N, D))
        else:
            Si, Zi = info

        if len(Si) != B:
            raise ValueError("The batch size changed during iteration")

        if K.grad_fn is not None or value.grad_fn is not None:
            Zi = Zi + K
            Si = Si + torch.einsum("bnd,bnm->bndm", K, value)
        else:
            Zi += K
            Si += torch.einsum("bnd,bnm->bndm", K, value)

        Z = 1. / (torch.einsum("bnd,bnd->bn", Q, Zi) + self.eps)
        V = torch.einsum("bnd,bndm,bn->bnm", Q, Si, Z)

        return V, [Si, Zi]


class RecurrentSelfAttentionLayer(nn.Module):
    """See fast_transformers.attention.attention_layer.AttentionLayer.

    The only difference with the corresponding module is that this projects
    only one input and then calls the inner attention with the provided
    previous state.
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
    def __init__(self, d_model, n_heads, layer_idx):
        super(RecurrentSelfAttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_heads
        self.layer_idx = layer_idx
        assert d_model % n_heads == 0, "d_model should be divisible by n_head"
        self.head_dim = d_model // n_heads
        self.scaling = self.head_dim ** -0.5

        self.self_attention = RecurrentLinearAttention(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def reset_params(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, querys, keys, values, info=None, state=None):
        assert keys.size() == values.size()
        if state is not None:
            assert self.layer_idx is not None

        # info = check_state(info, memory)

        q = self.q_proj(querys)
        k = self.k_proj(keys)
        v = self.v_proj(values)
        q *= self.scaling
        prev_k = prev_v = None

        if isinstance(state, TransformerState):
            prev_k = state.decoder_prev_key[self.layer_idx]
            prev_v = state.decoder_prev_value[self.layer_idx]

        if prev_k is not None:
            k = torch.cat((prev_k, k), dim=1)
            v = torch.cat((prev_v, v), dim=1)

        if isinstance(state, TransformerState):
            state.decoder_prev_key[self.layer_idx] = k
            state.decoder_prev_value[self.layer_idx] = v

        batch_size, length_q, d_model = querys.size()
        length_k, length_v = keys.size(1), values.size(1)

        assert length_q == length_k == length_v, "The length of Q, K and V are not the same."

        Q = q.reshape(batch_size, length_q, self.n_head, self.head_dim)
        K = k.reshape(batch_size, length_k, self.n_head, self.head_dim)
        V = v.reshape(batch_size, length_v, self.n_head, self.head_dim)

        output = []
        for i in range(length_q):
            Qi = Q[:, i, :, :]
            Ki = K[:, i, :, :]
            Vi = V[:, i, :, :]
            Vi, info = self.self_attention(Qi, Ki, Vi, info)
            Vi = Vi.reshape(batch_size, 1, self.n_head, self.head_dim)
            output.append(Vi)

        output = torch.cat(output, dim=1)
        output = output.reshape(batch_size, length_q, -1)
        output = self.o_proj(output)

        return output
