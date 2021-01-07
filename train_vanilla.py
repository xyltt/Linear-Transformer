import time
from torch import nn
from torch import optim
from data.data_loader import load_data
from fastNLP import Trainer
from fastNLP import AccuracyMetric
from fastNLP import CrossEntropyLoss
from fastNLP.embeddings import BertEmbedding, StaticEmbedding
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.embeddings.utils import get_sinusoid_encoding_table
from fastNLP import BucketSampler, GradientClipCallback, WarmupCallback
from model.classfication_vanilla import TextClassification

lr = 1e-5
n_epochs = 10
batch_size = 2
n_head = 4
d_model = 256
dim_ff = 512
dropout = 0.3
num_layers = 6
pos_embed = 'sin'
data_path = './data/imdb_data_bundle.pt'
bind_decoder_input_output_embed = False

data_bundle = load_data(data_path)
print(data_bundle)

src_embed = StaticEmbedding(vocab=data_bundle.get_vocab('words'), model_dir_or_name='en-glove-840B-300d')

max_length_train = max([seq_len for seq_len in data_bundle.get_dataset('train')['seq_len']])
max_length_test = max([seq_len for seq_len in data_bundle.get_dataset('test')['seq_len']])
max_length = max(max_length_train, max_length_test)
print("数据集样本最大长度max_length =", max_length)

if pos_embed == 'sin':
    encoder_pos_embed = nn.Embedding.from_pretrained(
        get_sinusoid_encoding_table(max_length + 1, src_embed.embedding_dim, padding_idx=0), freeze=True
    )
elif pos_embed == 'learned':
    encoder_pos_embed = get_embeddings((max_length + 1, src_embed.embedding_dim), padding_idx=0)

model = TextClassification(embed=src_embed,
                           pos_embed=encoder_pos_embed,
                           num_layers=num_layers,
                           d_model=d_model,
                           n_head=n_head,
                           dim_ff=dim_ff,
                           dropout=dropout,
                           class_num=2)

parametrs = []
params = {'lr': lr}
params['params'] = [param for param in model.parameters() if param.requires_grad]
parametrs.append(params)

optimizer = optim.Adam(parametrs)

callbacks = []
callbacks.append(GradientClipCallback(clip_value=1, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))

sampler = BucketSampler(seq_len_field_name='seq_len')
trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss=CrossEntropyLoss(), batch_size=batch_size, sampler=sampler,
                  drop_last=False, update_every=1, num_workers=2, n_epochs=n_epochs,
                  print_every=1, dev_data=data_bundle.get_dataset('test'),
                  metrics=AccuracyMetric(), metric_key=None, validate_every=-1,
                  save_path=None, use_tqdm=True, device=1)


start = time.time()
trainer.train(load_best_model=False)
end = time.time()
print(end - start)