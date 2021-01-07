import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random

from data.wmt16_loader import MTPipe
from losses.mtloss import MTLoss
from metrics.bleu import BLUEMetric
from model.seq2seq_model import LinearTransformerSeq2SeqModel
from fastNLP import Trainer
from fastNLP.embeddings import StaticEmbedding
from fastNLP.models import SequenceGeneratorModel
from fastNLP import BucketSampler, GradientClipCallback, cache_results
from fastNLP import SortedSampler, WarmupCallback, FitlogCallback


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.set_device(0)

#######hyper
lr = 5e-4
n_epochs = 20
batch_size = 64
n_heads = 8
d_model = 256
dim_ff = 512
num_layers = 3
bind_decoder_input_output_embed = False
#######hyper

# @cache_results('caches/data.pkl')
def get_data():
    data_bundle = MTPipe().process_from_file(paths='./wmt16')
    return data_bundle

data_bundle = get_data()
max_len_train = max([seq_len for seq_len in data_bundle.get_dataset('train')['tgt_seq_len']])
max_len_dev = max([seq_len for seq_len in data_bundle.get_dataset('dev')['tgt_seq_len']])
max_len_test = max([seq_len for seq_len in data_bundle.get_dataset('test')['tgt_seq_len']])
max_len = max([max_len_train, max_len_dev, max_len_test])
print(data_bundle)
print("The maximal length of target is ", max_len)
src_vocab = data_bundle.get_vocab('src_tokens')
tgt_vocab = data_bundle.get_vocab('tgt_tokens')

src_embed = StaticEmbedding(data_bundle.get_vocab('src_tokens'), embedding_dim=d_model, model_dir_or_name=None)
tgt_embed = StaticEmbedding(data_bundle.get_vocab('tgt_tokens'), embedding_dim=d_model, model_dir_or_name=None)

model = LinearTransformerSeq2SeqModel.build_model(src_embed=src_embed, tgt_embed=tgt_embed,
            pos_embed='sin', max_position=1024, num_layers=6, d_model=256, n_head=8, dim_ff=512, dropout=0.1,
            bind_encoder_decoder_embed=False, bind_decoder_input_output_embed=bind_decoder_input_output_embed)

model = SequenceGeneratorModel(model, bos_token_id=tgt_vocab.to_index('<s>'),
                               eos_token_id=tgt_vocab.to_index('</s>'), max_length=max_len,
                               num_beams=4, do_sample=False, temperature=1.0, top_k=20, top_p=1.0,
                               repetition_penalty=1, length_penalty=1.0, pad_token_id=0)

optimizer = optim.AdamW(model.parameters(), lr=lr)

callbacks = []
callbacks.append(GradientClipCallback(clip_value=1, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
# callbacks.append(FitlogCallback(data_bundle.get_dataset('test')))
sampler = BucketSampler(seq_len_field_name='src_seq_len')
trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                    loss=MTLoss(), batch_size=batch_size, sampler=sampler, drop_last=False,
                    update_every=1, num_workers=2, n_epochs=n_epochs, print_every=1, device=0,
                    use_tqdm=True, dev_data=data_bundle.get_dataset('dev'),
                    metrics=BLUEMetric(tgt_vocab), metric_key=None,
                    validate_every=-1, save_path=None, callbacks=callbacks,
                    # check_code_level=0, test_use_tqdm=False,
                    test_sampler=SortedSampler('src_seq_len')
                  )

trainer.train(load_best_model=False)



# model = model.to(device)
#
# LEARNING_RATE = 0.0005
#
# optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
#
# criterion = nn.CrossEntropyLoss(ignore_index = tgt_PAD_IDX)
#
# def train(model, iterator, optimizer, criterion, clip):
#     model.train()
#     epoch_loss = 0.0
#     for i, batch in tqdm(enumerate(train_iterator)):
#         src_info = batch.src
#         tgt_info = batch.trg
#
#         src = src_info[0]
#         src_len = src_info[1].cpu()
#         src_len = src_len.view(1, len(src_len))
#
#         tgt = tgt_info[0]
#         tgt_len = tgt_info[1].cpu()
#         tgt_len = tgt_len.view(1, len(tgt_len))
#
#         optimizer.zero_grad()
#
#         output = model(src, tgt, src_len[0], tgt_len[0])
#         pred = output['pred']
#         output_dim = pred.shape[-1]
#
#         pred = pred.contiguous().view(-1, output_dim)
#         tgt = tgt.view(-1)
#
#         loss = criterion(pred, tgt)
#
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#
#         optimizer.step()
#
#         epoch_loss += loss.item()
#     return epoch_loss / len(iterator)
#
# def evaluate(model, iterator, criterion):
#     model.eval()
#     epoch_loss = 0.0
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):
#             src_info = batch.src
#             tgt_info = batch.trg
#
#             src = src_info[0]
#             src_len = src_info[1].cpu()
#             src_len = src_len.view(1, len(src_len))
#
#             tgt = tgt_info[0]
#             tgt_len = tgt_info[1].cpu()
#             tgt_len = tgt_len.view(1, len(tgt_len))
#
#             output = model(src, tgt, src_len[0], tgt_len[0])
#             pred = output['pred']
#             output_dim = pred.shape[-1]
#
#             pred = pred.contiguous().view(-1, output_dim)
#             tgt = tgt.view(-1)
#
#             loss = criterion(pred, tgt)
#
#             epoch_loss += loss.item()
#         return epoch_loss / len(iterator)
#
# EPOCHS = 10
# CLIP = 1

