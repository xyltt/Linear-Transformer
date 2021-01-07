import math
import torch
from torch import nn
from .feature_map import elu_feature_map
from modules.state import LinearTransformerState
from fastNLP.modules.decoder.seq2seq_state import TransformerState

def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps):
    Q = q.contiguous().permute(0, 2, 1, 3)
    K = k.contiguous().permute(0, 2, 1, 3)
    V = v.contiguous().permute(0, 2, 1, 3)
    KV = torch.einsum('...sd,...se->...de', K, V)
    Z = 1.0 / torch.einsum('...sd,...d->...s', Q, K.sum(dim=-2)+eps)
    V_new = torch.einsum('...de,...sd,...s->...se', KV, Q, Z)
    return V_new.contiguous().permute(0, 2, 1, 3)

class LinearMultiHeadAttention(nn.Module):
    """
    Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.
    Given the queries, keys and values as Q, K, V instead of computing
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    we make use of a feature map function Φ(.) and perform the following
    computation
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, d_model: int, n_head: int, dropout: float,
                 layer_idx=None, feature_map=None, eps=1e-6):
        super(LinearMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.eps = eps
        self.layer_idx = layer_idx
        assert d_model % n_head == 0, "d_model should be divisible by n_head"
        self.head_dim = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.feature_map = (
            feature_map(d_model) if feature_map else
            elu_feature_map(d_model)
        )

    def reset_params(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, query, key, value, key_mask, state=None):
        """

        :param query: [batch_size, seq_len, d_model]
        :param key: [batch_size, seq_len, d_model]
        :param value: [batch_size, seq_len, d_model]
        :param key_mask: [batch_size, seq_len] 用于指示哪些key不要被attend到；注意到mask为1的地方是要attend的
        :param attn_mask: [seq_len, seq_len] 用于mask掉attention map。主要是用在训练时decoder端的self-attention，下三角为1
        """
        q = self.q_proj(query)
        k = v = None

        if self.layer_idx is not None:
            if isinstance(state, LinearTransformerState):
                k = state.encoder_key[self.layer_idx]
                v = state.encoder_value[self.layer_idx]

        if k is None:
            k = self.k_proj(key)
            v = self.v_proj(value)

        if self.layer_idx is not None:
            if isinstance(state, LinearTransformerState):
                state.encoder_key[self.layer_idx] = k
                state.encoder_value[self.layer_idx] = v

        batch_size, q_len, d_model = q.size()
        k_len, v_len = k.size(1), v.size(1)

        q = q.reshape(batch_size, q_len, self.n_head, self.head_dim)
        k = k.reshape(batch_size, k_len, self.n_head, self.head_dim)
        v = v.reshape(batch_size, v_len, self.n_head, self.head_dim)

        self.feature_map.new_feature_map()
        q = self.feature_map.forward_queries(q)
        k = self.feature_map.forward_keys(k)

        if key_mask is not None:
            _key_mask = ~key_mask[:, :, None, None].bool()
            k = k.masked_fill(_key_mask, 0.0)

        # KV = torch.einsum("bsnd,bsnm->bnmd", K, V)
        #
        # Z = 1 / (torch.einsum("bsnd,bnd->bsn", Q, K.sum(dim=1))+self.eps)
        #
        # V_new = torch.einsum("bsnd,bnmd,bsn->bsnm", Q, KV, Z)
        V_new = linear_attention(q, k, v, self.eps)

        output = V_new.contiguous().reshape(batch_size, q_len, -1)
        output = self.o_proj(output)

        return output









