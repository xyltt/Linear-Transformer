import time
from tqdm import tqdm
from data.iwslt17_loader import IWSLT2017Pipe
from fastNLP.models.seq2seq_model import TransformerSeq2SeqModel
from model.seq2seq_model import LinearTransformerSeq2SeqModel
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from fastNLP.models import SequenceGeneratorModel
import torch
from torch import optim
import torch.nn.functional as F
from fastNLP import seq_len_to_mask

def prepare_env(seq_len, batch_size):
    vocab = Vocabulary().add_word_lst("This is a test .".split())
    vocab.add_word_lst("Another test !".split())
    embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5)

    src_words_idx = torch.arange(1, seq_len + 1)
    src_words_idx = src_words_idx.expand([batch_size, seq_len])

    mask = []
    lens_src = []
    for i in range(batch_size):
        length = seq_len - i * 15
        lens_src.append(length)
        maski = torch.randint(1, 2, (1, length)).view(1, length)
        zero_extend = torch.zeros((1, seq_len-length), dtype=torch.long).view(1, seq_len - length)
        maski = torch.cat((maski, zero_extend), dim=1)
        mask.append(maski)
    mask = torch.cat(mask, dim=0).bool()
    src_words_idx = src_words_idx.masked_fill_(mask, 0).long()
    batch_src_words_idx = src_words_idx.expand((100, batch_size, seq_len))

    tgt_words_idx = torch.arange(1, seq_len + 1)
    tgt_words_idx = tgt_words_idx.expand((batch_size, seq_len))

    mask = []
    lens_tgt = []
    for i in range(batch_size):
        length = seq_len - i * 20
        lens_tgt.append(length)
        maski = torch.randint(1, 2, (1, length)).view(1, length)
        zero_extend = torch.zeros((1, seq_len - length), dtype=torch.long).view(1, seq_len - length)
        maski = torch.cat((maski, zero_extend), dim=1)
        mask.append(maski)

    mask = torch.cat(mask, dim=0).bool()
    tgt_words_idx = tgt_words_idx.masked_fill_(mask, 0).long()
    batch_tgt_words_idx = tgt_words_idx.expand((100, batch_size, seq_len))

    src_seq_len = torch.tensor(lens_src, dtype=torch.long)
    batch_src_seq_len = src_seq_len.expand([100, batch_size])
    tgt_seq_len = torch.tensor(lens_tgt, dtype=torch.long)
    batch_tgt_seq_len = tgt_seq_len.expand([100, batch_size])

    return embed, batch_src_words_idx, batch_tgt_words_idx, batch_src_seq_len, batch_tgt_seq_len


def train_model(model, batch_src_words_idx, batch_tgt_words_idx, batch_tgt_seq_len, batch_src_seq_len, device):
    print("===开始训练===")
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for i in tqdm(range(20)):
        src_words_idx = batch_src_words_idx[i].to(device)
        tgt_words_idx = batch_tgt_words_idx[i].to(device)
        src_seq_len = batch_src_seq_len[i].to(device)
        tgt_seq_len = batch_tgt_seq_len[i].to(device)
        mask = seq_len_to_mask(tgt_seq_len).eq(0).to(device)
        target = tgt_words_idx.masked_fill(mask, 1e-5).to(device)

        optimizer.zero_grad()
        pred = model(src_words_idx, tgt_words_idx, src_seq_len)['pred']  # bsz x max_len x vocab_size
        loss = F.cross_entropy(pred.transpose(1, 2), target)
        loss.backward()
        optimizer.step()

    print("===训练结束===")


# 测试能否train到overfit
batch_size = 1
seq_len = 65536
device = torch.device('cuda')
embed, batch_src_words_idx, batch_tgt_words_idx, batch_src_seq_len, \
                        batch_tgt_seq_len = prepare_env(seq_len, batch_size)

model = LinearTransformerSeq2SeqModel.build_model(src_embed=embed, tgt_embed=None,
            pos_embed='sin', max_position=80000, num_layers=3, d_model=256, n_head=4,
            dim_ff=512, dropout=0.1, bind_encoder_decoder_embed=True,
            bind_decoder_input_output_embed=True)

model = model.to(device)
start = time.clock()
train_model(model, batch_src_words_idx, batch_tgt_words_idx, batch_tgt_seq_len, batch_src_seq_len, device)
end = time.clock()
print("训练时间为: ", end - start)

