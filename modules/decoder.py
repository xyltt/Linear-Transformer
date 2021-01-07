import math
import torch
from torch import nn
from typing import Union, Tuple
import torch.nn.functional as F
from fastNLP.embeddings import StaticEmbedding
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.embeddings.utils import get_embeddings
from modules.state import LinearTransformerState
from modules.self_attention import RecurrentSelfAttentionLayer
from modules.cross_attention import RecurrentCrossAttentionLayer
from modules.linear_attention import LinearMultiHeadAttention
from modules.causal_linear_attention import CausalLinearAttentionLayer
from fastNLP.modules.decoder.seq2seq_decoder import Seq2SeqDecoder
from fastNLP.modules.decoder.seq2seq_state import TransformerState

class LinearTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, dim_ff=2048, dropout=0.1, layer_idx=None):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.layer_idx = layer_idx

        # self.self_attention = RecurrentSelfAttentionLayer(d_model, n_head, layer_idx)
        self.self_attention = CausalLinearAttentionLayer(d_model, n_head, layer_idx)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.cross_attention = LinearMultiHeadAttention(d_model, n_head, dropout, layer_idx)
        self.cross_attn_layer_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(dim_ff, d_model),
                                 nn.Dropout(p=dropout))

        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, encoder_mask=None, state=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attention(querys=x,
                                   keys=x,
                                   values=x,
                                   state=state)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attention(query=x,
                                 key=encoder_output,
                                 value=encoder_output,
                                 key_mask=encoder_mask,
                                 state=state)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x

class TiedEmbedding(nn.Module):
    """
    用于将weight和原始weight绑定

    """
    def __init__(self, weight):
        super().__init__()
        self.weight = weight  # vocab_size x embed_size

    def forward(self, x):
        """

        :param torch.FloatTensor x: bsz x * x embed_size
        :return: torch.FloatTensor bsz x * x vocab_size
        """
        return torch.matmul(x, self.weight.t())

def get_binded_decoder_output_embed(embed):
    """
    给定一个embedding，输出对应的绑定的embedding，输出对象为TiedEmbedding

    :param embed:
    :return:
    """
    if isinstance(embed, StaticEmbedding):
        for idx, map2idx in enumerate(embed.words_to_words):
            assert idx == map2idx, "Invalid StaticEmbedding for Decoder, please check:(1) whether the vocabulary " \
                                   "include `no_create_entry=True` word; (2) StaticEmbedding should  not initialize with " \
                                   "`lower=True` or `min_freq!=1`."
    elif not isinstance(embed, nn.Embedding):
        raise TypeError("Only nn.Embedding or StaticEmbedding is allowed for binding.")

    return TiedEmbedding(embed.weight)


class LinearTransformerDecoder(Seq2SeqDecoder):
    def __init__(self, embed: Union[nn.Module, StaticEmbedding, Tuple[int, int]],
                 pos_embed: nn.Module=None, d_model=512, num_layers=6, n_head=8,
                 dim_ff=2048, dropout=0.1, bind_decoder_input_output_embed=True):
        super().__init__()
        self.embed = get_embeddings(embed)
        self.pos_embed = pos_embed

        if bind_decoder_input_output_embed:
            self.output_layer = get_binded_decoder_output_embed(self.embed)
        else:
            self.output_embed = get_embeddings((self.embed.num_embeddings, self.embed.embedding_dim))
            self.output_layer = TiedEmbedding(self.output_embed.weight)

        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_fc = nn.Linear(self.embed.embedding_dim, d_model)
        self.layer_stacks = nn.ModuleList([LinearTransformerDecoderLayer(d_model, n_head, dim_ff, dropout, layer_idx)
                                           for layer_idx in range(num_layers)])

        self.embed_scale = math.sqrt(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_fc = nn.Linear(d_model, self.embed.embedding_dim)

    def forward(self, tokens, state, return_attention=False):
        encoder_output = state.encoder_output
        encoder_mask = state.encoder_mask

        assert state.decode_length < tokens.size(1), "The decoded tokens in State should be less than tokens."
        tokens = tokens[:, state.decode_length:]
        device = tokens.device

        # x = []
        # if self.training:
        #     tgt_seq_len = tokens.size(1) - 1
        # else:
        #     tgt_seq_len = tokens.size(1)
        # for i in range(tgt_seq_len):
        #     # 这里需要设计一下
        #     embed = self.embed(tokens[:, i])
        #     x.append(embed)
        #
        # x = torch.stack(x, dim=1)
        # x = self.embed_scale * x

        x = self.embed_scale * self.embed(tokens)
        if self.pos_embed is not None:
            position = torch.arange(state.decode_length, state.decode_length+tokens.size(1)).long().to(device)[None]
            x += self.pos_embed(position)
        x = self.input_fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layer_stacks:
            x = layer(x=x,
                      encoder_output=encoder_output,
                      encoder_mask=encoder_mask,
                      state=state
                      )

        state.decode_length += 1

        x = self.layer_norm(x)
        x = self.output_fc(x)
        feats = self.output_layer(x)

        return feats

    def init_state(self, encoder_output, encoder_mask):
        if isinstance(encoder_output, torch.Tensor):
            encoder_output = encoder_output
        elif isinstance(encoder_output, (list, tuple)):
            encoder_output = encoder_output[0]  # 防止是LSTMEncoder的输出结果
        else:
            raise TypeError("Unsupported `encoder_output` for TransformerSeq2SeqDecoder")
        state = LinearTransformerState(encoder_output, encoder_mask, num_decoder_layer=self.num_layers)
        return state


