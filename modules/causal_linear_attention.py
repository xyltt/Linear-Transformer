import torch
from torch import nn
from modules.feature_map import elu_feature_map
from LinearKernel.causal_product import causal_dot_product
from fastNLP.modules.decoder.seq2seq_state import TransformerState
from modules.state import LinearTransformerState

def permute_for_matrix(matrix: torch.Tensor):
    assert matrix.dim() == 4
    return matrix.contiguous().permute(0, 2, 1, 3)

def causal_linear(Q, K, V):
    Q = permute_for_matrix(Q)
    K = permute_for_matrix(K)
    V = permute_for_matrix(V)
    V_new = causal_dot_product(Q, K, V)
    return permute_for_matrix(V_new)


class CausalLinearAttention(nn.Module):
    """Implement causally masked attention using dot product of feature maps in
    O(N D^2) complexity.
    See fast_transformers.attention.linear_attention.LinearAttention for the
    general concept of replacing the softmax with feature maps. In addition to
    that, we also make use of the fact that causal masking is a triangular mask
    which allows us to apply the masking and still compute the attention in O(N
    D^2) complexity.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super(CausalLinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        # self.cnt = 0

    def _make_sizes_compatible(self, Q, K):
        """Either slice or pad K in case that the sizes do not match between Q
        and K."""
        N, L, H, E = Q.shape
        _, S, _, _ = K.shape
        if L == S:
            return Q, K

        if L < S:
            return Q, K[:, :L, :, :]

        if L > S:
            temp = K.new_zeros(N, L-S, H, E)
            K = torch.cat([K, temp], dim=1)
            return Q, K

    def forward(self, queries, keys, values, key_mask):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map()
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # if key_mask is not None:
        #     _key_mask = ~key_mask[:, :, None, None].bool()
        #     K = K.masked_fill(_key_mask, 0.0)

        # Ensure that Q and K have compatible sizes for the following
        # computations, namely L == S
        # Q, K = self._make_sizes_compatible(Q, K)

        # Compute the normalizers
        Z = 1/(torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)

        # Compute the unnormalized result
        V = causal_linear(
            Q,
            K,
            values
        )
        V = V * Z[:, :, :, None]
        return V

class CausalLinearAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head, layer_idx, feature_map=None, eps=1e-6):
        super(CausalLinearAttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.layer_idx = layer_idx
        assert d_model % n_head == 0, "d_model should be divisible by n_head"
        self.head_dim = d_model // n_head
        self.scaling = self.head_dim ** -0.5
        self.eps = eps

        self.attention = CausalLinearAttention(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.feature_map = (
            feature_map(self.head_dim) if feature_map else
            elu_feature_map(self.head_dim)
        )

    def reset_params(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, querys, keys, values, key_mask=None, state=None):
        assert keys.size() == values.size()
        if state is not None:
            assert self.layer_idx is not None
        # qkv_same = querys.data_ptr() == keys.data_ptr() == values.data_ptr()

        q = self.q_proj(querys)
        q *= self.scaling
        k = self.k_proj(keys)
        v = self.v_proj(values)
        prev_kv_sum = prev_k_sum = None

        batch_size, length_q, d_model = querys.size()
        length_k, length_v = keys.size(1), values.size(1)

        if isinstance(state, LinearTransformerState) and self.training is False:
            prev_k_sum = state.decoder_k_sum[self.layer_idx]
            prev_kv_sum = state.decoder_kv_sum[self.layer_idx]

            Q = q.reshape(batch_size, length_q, self.n_head, self.head_dim)
            K = k.reshape(batch_size, length_k, self.n_head, self.head_dim)
            V = v.reshape(batch_size, length_v, self.n_head, self.head_dim)

            self.feature_map.new_feature_map()
            Q = self.feature_map.forward_queries(Q)
            K = self.feature_map.forward_keys(K)

            Q = permute_for_matrix(Q)
            K = permute_for_matrix(K)
            V = permute_for_matrix(V)

            k_sum = prev_k_sum + K.contiguous().reshape(batch_size, self.n_head, -1)
            kv = torch.einsum('...sd,...se->...de', K, V)
            kv_sum = prev_kv_sum + kv

            state.decoder_k_sum[self.layer_idx] = k_sum
            state.decoder_kv_sum[self.layer_idx] = kv_sum

            Z = 1.0 / torch.einsum('...sd,...d->...s', Q, k_sum + self.eps)
            V_new = torch.einsum('...de,...sd,...s->...se', kv_sum, Q, Z)

            output = permute_for_matrix(V_new)
            output = output.reshape(batch_size, length_q, -1)
            output = self.o_proj(output)
        else:
            Q = q.contiguous().reshape(batch_size, length_q, self.n_head, self.head_dim)
            K = k.contiguous().reshape(batch_size, length_k, self.n_head, self.head_dim)
            V = v.contiguous().reshape(batch_size, length_v, self.n_head, self.head_dim)

            output = self.attention(Q, K, V, key_mask)
            output = output.contiguous().reshape(batch_size, length_q, -1)
            output = self.o_proj(output)

        return output