import torch
from torch import nn

from fastNLP.embeddings import get_embeddings
from fastNLP.models.seq2seq_model import Seq2SeqModel
from fastNLP.embeddings.utils import get_sinusoid_encoding_table
from modules.encoder import LinearTransformerEncoder
from modules.decoder import LinearTransformerDecoder


class LinearTransformerSeq2SeqModel(Seq2SeqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, src_embed, tgt_embed=None, pos_embed='sin',
                    max_position=1024, num_layers=6, d_model=512,
                    n_head=8, dim_ff=2048, dropout=0.1,
                    bind_encoder_decoder_embed=False,
                    bind_decoder_input_output_embed=True):

        if bind_encoder_decoder_embed and tgt_embed is not None:
            raise RuntimeError("If you set `bind_encoder_decoder_embed=True`, please do not provide `tgt_embed`.")

        src_embed = get_embeddings(src_embed)

        if bind_encoder_decoder_embed:
            tgt_embed = src_embed
        else:
            assert tgt_embed is not None, "You need to pass `tgt_embed` when `bind_encoder_decoder_embed=False`"
            tgt_embed = get_embeddings(tgt_embed)

        if pos_embed == 'sin':
            encoder_pos_embed = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(max_position + 1, src_embed.embedding_dim, padding_idx=0),
                freeze=True)  # 这里规定0是padding
            deocder_pos_embed = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(max_position + 1, tgt_embed.embedding_dim, padding_idx=0),
                freeze=True)  # 这里规定0是padding
        elif pos_embed == 'learned':
            encoder_pos_embed = get_embeddings((max_position + 1, src_embed.embedding_dim), padding_idx=0)
            deocder_pos_embed = get_embeddings((max_position + 1, src_embed.embedding_dim), padding_idx=1)
        else:
            raise ValueError("pos_embed only supports sin or learned.")

        encoder = LinearTransformerEncoder(embed=src_embed, pos_embed=encoder_pos_embed,
                                            num_layers=num_layers, d_model=d_model, n_head=n_head, dim_ff=dim_ff,
                                            dropout=dropout)
        decoder = LinearTransformerDecoder(embed=tgt_embed, pos_embed=deocder_pos_embed,
                                            d_model=d_model, num_layers=num_layers, n_head=n_head, dim_ff=dim_ff,
                                            dropout=dropout,
                                            bind_decoder_input_output_embed=bind_decoder_input_output_embed)

        return cls(encoder, decoder)


