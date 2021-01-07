import torch
from torch import nn
from typing import Union, Tuple
import torch.nn.functional as F
from fastNLP.embeddings import StaticEmbedding
from fastNLP.core.utils import seq_len_to_mask
from modules.encoder import LinearTransformerEncoder
from fastNLP.modules.encoder.seq2seq_encoder import TransformerSeq2SeqEncoder


class TextClassification(nn.Module):
    def __init__(self, embed: Union[nn.Module, StaticEmbedding, Tuple[int, int]], pos_embed=None,
                 num_layers=6, d_model=512, n_head=8, dim_ff=2048, dropout=0.1, class_num=2):
        super(TextClassification, self).__init__()
        self.encoder = TransformerSeq2SeqEncoder(embed=embed,
                                          pos_embed=pos_embed,
                                          num_layers=num_layers,
                                          d_model=d_model,
                                          n_head=n_head,
                                          dim_ff=dim_ff,
                                          dropout=dropout)

        self.linear = nn.Linear(d_model, class_num)

    def forward(self, words, seq_len):
        x, _ = self.encoder(words, seq_len)
        feats, _ = torch.max(x, dim=1)
        logits = self.linear(feats)
        return {'pred': logits}