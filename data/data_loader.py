import os
import torch
from numpy import *
import fastNLP
from fastNLP import DataSet
from nltk.corpus import stopwords
from fastNLP import Vocabulary
from fastNLP.io import IMDBLoader, IMDBPipe, DataBundle
from fastNLP.io import SST2Loader, SST2Pipe

stop_words = stopwords.words('english')

def load_data(data_path):
    if os.path.exists(data_path):
        print("===已经存在处理好的数据，正在加载===")
        data_bundle = torch.load(data_path)
    else:
        print("===正在处理数据并保存到指定文件===")
        loader = SST2Loader()
        data_dir = loader.download()
        data_bundle = loader.load(data_dir)
        data_bundle = SST2Pipe(lower=True).process(data_bundle)
        # loader = IMDBLoader()
        # data_dir = loader.download()
        # data_bundle = loader.load(data_dir)
        # data_bundle = IMDBPipe(lower=True).process(data_bundle)
        torch.save(data_bundle, data_path)
        # data_bundle.get_dataset('train').drop(lambda ins: len(ins['words']) > 1000)
        # data_bundle.get_dataset('test').drop(lambda ins: len(ins['words']) > 1000)
    return data_bundle

if __name__ == '__main__':
    data_path = './imdb_data_bundle.pt'
    data_bundle = load_data(data_path)
    print(data_bundle)
    max_length_train = max([seq_len for seq_len in data_bundle.get_dataset('train')['seq_len']])
    # max_length_dev = max([seq_len for seq_len in data_bundle.get_dataset('dev')['seq_len']])
    max_length_test = max([seq_len for seq_len in data_bundle.get_dataset('test')['seq_len']])
    max_length = max(max_length_train, max_length_test)
    print("数据集样本最大长度max_length =", max_length)

    len_train = [seq_len for seq_len in data_bundle.get_dataset('train')['seq_len']]
    mean_len_train = mean(len_train)
    # len_dev = [seq_len for seq_len in data_bundle.get_dataset('dev')['seq_len']]
    # mean_len_dev = mean(len_dev)
    len_test = [seq_len for seq_len in data_bundle.get_dataset('test')['seq_len']]
    mean_len_test = mean(len_test)
    print("The average length of train data is", int(mean_len_train))
    # print("The average length of dev data is", int(mean_len_dev))
    print("The average length of test data is", int(mean_len_test))
    print(mean([mean_len_train, mean_len_test]))