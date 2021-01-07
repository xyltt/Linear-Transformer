import os
import torch
import spacy
from numpy import *
from fastNLP.io.loader import Loader
from fastNLP.io.pipe import Pipe
from fastNLP.io import DataBundle
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

class IWSLT2017Loader(Loader):
    def __init__(self):
        super().__init__()

    def load_train_data(self, path_en, path_de) -> DataSet:
        dataset = DataSet()
        with open(path_en, encoding='utf-8', mode='r') as en_corpus:
            with open(path_de, encoding='utf-8', mode='r') as de_corpus:
                cnt = 0
                en_sentences, de_sentences = '', ''
                for line_en, line_de in zip(en_corpus, de_corpus):
                    line_en = line_en.strip()
                    line_de = line_de.strip()
                    if line_en == '' or line_de == '':
                        continue
                    if line_en.startswith('<') or line_de.startswith('<'):
                        if line_en.startswith('<reviewer href') and en_sentences != '':
                            cnt = 0
                            en_sentences = '<s> ' + en_sentences + ' </s>'
                            dataset.append(instance=Instance(src_raw=de_sentences, tgt_raw=en_sentences))
                            en_sentences = ''
                            de_sentences = ''
                        continue
                    cnt += 1
                    en_sentences += (' ' + line_en)
                    de_sentences += (' ' + line_de)
                    if cnt == 60:
                        cnt = 0
                        en_sentences = '<s> ' + en_sentences + ' </s>'
                        dataset.append(instance=Instance(src_raw=de_sentences, tgt_raw=en_sentences))
                        en_sentences = ''
                        de_sentences = ''
        return dataset

    def load_dev_test(self, path_en, path_de) -> DataSet:
        dataset = DataSet()
        with open(path_en, encoding='utf-8', mode='r') as en_corpus:
            with open(path_de, encoding='utf-8', mode='r') as de_corpus:
                seg_id, cnt = 0, 0
                en_sentences, de_sentences = '', ''
                for line_en, line_de in zip(en_corpus, de_corpus):
                    line_en = line_en.strip()
                    line_de = line_de.strip()
                    # if line_en == '' or line_de == '':
                    #     continue
                    if line_en.startswith('</doc>') and line_de.startswith('</doc>'):
                        if en_sentences != '':
                            seg_id, cnt = 0, 0
                            en_sentences = '<s> ' + en_sentences + ' </s>'
                            dataset.append(instance=Instance(src_raw=de_sentences, tgt_raw=en_sentences))
                            en_sentences = ''
                            de_sentences = ''

                    if line_en.startswith('<seg id=') and line_de.startswith('<seg id='):
                        seg_id += 1
                        cnt += 1
                        strip_str = '<seg id="' + str(seg_id) + '">'
                        line_en = line_en.lstrip(strip_str)
                        line_en = line_en.rstrip('</seg>')
                        line_en = line_en.strip()

                        line_de = line_de.lstrip(strip_str)
                        line_de = line_de.rstrip('</seg>')
                        line_de = line_de.strip()

                        en_sentences += (' ' + line_en)
                        de_sentences += (' ' + line_de)

                        if cnt == 60:
                            cnt = 0
                            en_sentences = '<s> ' + en_sentences + ' </s>'
                            dataset.append(instance=Instance(src_raw=de_sentences, tgt_raw=en_sentences))
                            en_sentences = ''
                            de_sentences = ''
        return dataset

    def load(self, paths=None) -> DataBundle:
        datasets = {}
        path_en = '../IWSLT2017/train.tags.de-en.en'
        path_de = '../IWSLT2017/train.tags.de-en.de'
        train_dataset = self.load_train_data(path_en, path_de)
        datasets['train'] = train_dataset

        path_en = '../IWSLT2017/IWSLT17.TED.dev2010.de-en.en.xml'
        path_de = '../IWSLT2017/IWSLT17.TED.dev2010.de-en.de.xml'
        dev_dataset = self.load_dev_test(path_en, path_de)
        datasets['dev'] = dev_dataset

        path_en = '../IWSLT2017/IWSLT17.TED.tst2010.de-en.en.xml'
        path_de = '../IWSLT2017/IWSLT17.TED.tst2010.de-en.de.xml'
        test_dataset = self.load_dev_test(path_en, path_de)
        datasets['test'] = test_dataset

        data_bundle = DataBundle(datasets=datasets)

        return data_bundle


class IWSLT2017Pipe(Pipe):
    def __init__(self):
        super().__init__()

    def _tokenize(self, data_bundle):
        for name, dataset in data_bundle.datasets.items():

            def tokenize_de(raw_text):
                output = [(token.text).lower() for token in spacy_de.tokenizer(raw_text)]
                return output
            dataset.apply_field(tokenize_de, field_name='src_raw',
                                new_field_name='src_tokens')
            def tokenize_en(raw_text):
                output = [(token.text).lower() for token in spacy_en.tokenizer(raw_text)]
                return output
            dataset.apply_field(tokenize_en, field_name='tgt_raw',
                                new_field_name='tgt_tokens')

            dataset.apply_field(lambda x: len(x), field_name='src_tokens',
                                new_field_name='src_seq_len')
            dataset.apply_field(lambda x: len(x), field_name='tgt_tokens',
                                new_field_name='tgt_seq_len')

        return data_bundle


    def process(self, data_bundle: DataBundle) -> DataBundle:
        self._tokenize(data_bundle)

        fields = ['src_tokens', 'tgt_tokens']
        for field in fields:
            if field == 'src_tokens':
                vocab = Vocabulary()
            else:
                vocab = Vocabulary(unknown='<unk>')
            vocab.from_dataset(data_bundle.get_dataset('train'),
                               field_name=field,
                               no_create_entry_dataset=[data_bundle.get_dataset('dev'),
                                                        data_bundle.get_dataset('test')]
                               )
            vocab.index_dataset(*data_bundle.datasets.values(), field_name=field)
            data_bundle.set_vocab(vocab, field)

        # def padding(seq):
        #     length = len(seq)
        #     seq = seq + [0] * (2000 - length)
        #     return seq
        #
        # data_bundle.apply_field(padding, field_name='src_tokens', new_field_name='src_tokens')

        data_bundle.set_input('src_tokens', 'tgt_tokens', 'src_seq_len', 'tgt_seq_len')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len')

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        if os.path.exists(paths):
            data_bundle = torch.load(paths)
        else:
            data_bundle = IWSLT2017Loader().load(paths)
            data_bundle = self.process(data_bundle)
            torch.save(data_bundle, paths)
        return data_bundle

if __name__ == '__main__':
    save_path = '../IWSLT2017/processed_data_bundle_large.pt'
    data_bundle = IWSLT2017Pipe().process_from_file(save_path)
    print(data_bundle)
    len_train = [seq_len for seq_len in data_bundle.get_dataset('train')['tgt_seq_len']]
    mean_len_train = mean(len_train)
    len_dev = [seq_len for seq_len in data_bundle.get_dataset('dev')['tgt_seq_len']]
    mean_len_dev = mean(len_dev)
    len_test = [seq_len for seq_len in data_bundle.get_dataset('test')['tgt_seq_len']]
    mean_len_test = mean(len_test)
    print("The average length of train data is", int(mean_len_train))
    print("The average length of dev data is", int(mean_len_dev))
    print("The average length of test data is", int(mean_len_test))
    print(mean([mean_len_train, mean_len_dev, mean_len_test]))