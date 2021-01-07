# import torch
# from fastNLP import MetricBase
# from fastNLP import Vocabulary
# from torchtext.data.metrics import bleu_score
#
# class BLUEMetric(MetricBase):
#     def __init__(self, tgt_vocab: Vocabulary):
#         super().__init__()
#         self.tgt_vocab = tgt_vocab
#         self.pred_tgt = []
#         self.gold_tgt = []
#
#     def evaluate(self, tgt_tokens: torch.Tensor, pred: torch.Tensor, tgt_seq_len: torch.Tensor):
#         if pred.dim == 3:
#             pred = pred.argmax(dim=-1)
#
#         batch_size, _ = pred.size()

#         assert batch_size == tgt_tokens.size(0)
#         for b in range(batch_size):
#             pred_tgt = [self.tgt_vocab.idx2word[index.item()] for index in pred[b]]
#             gold_tgt = [self.tgt_vocab.idx2word[index.item()] for index in tgt_tokens[b]]
#             gold_tgt = gold_tgt[1:tgt_seq_len[b].item()]
#             self.pred_tgt.append(pred_tgt)
#             self.gold_tgt.append([gold_tgt])
#
#     def get_metric(self, reset=True):
#         res = {"BLEU Score": bleu_score(self.pred_tgt, self.gold_tgt)}
#         if reset:
#             self.pred_tgt = []
#             self.gold_tgt = []
#         return res

from fastNLP import MetricBase
import sacrebleu


class BLEUMetric(MetricBase):
    def __init__(self, vocab, eos_index, bpe_indicator='@@'):
        super().__init__()
        self.vocab = vocab
        self.eos_index = eos_index
        self.bpe_indicator = bpe_indicator
        self.goldens = []
        self.preds = []
        self.get_golden = True

    def evaluate(self, tgt_tokens, tgt_seq_len, pred):
        """

        :param tgt_tokens: bsz x max_len (构成为[<SOS>] + [tokens] + [<EOS>])
        :param tgt_seq_len: bsz
        :param pred: bsz x max_len' (构成为[<SOS>] + [tokens] + [<EOS>])
        :return:
        """
        for i in range(tgt_tokens.size(0)):
            self.goldens.append(' '.join(map(self.vocab.to_word, tgt_tokens[i, 1:tgt_seq_len[i]-1].tolist())).replace(f'{self.bpe_indicator} ', ''))

        for i in range(pred.size(0)):
            words = []
            for idx in pred[i, 1:].tolist():
                if idx==self.eos_index:
                    break
                words.append(self.vocab.to_word(idx))
            self.preds.append(' '.join(words).replace(f'{self.bpe_indicator} ', ''))

    def get_metric(self, reset=True):
        bleu = sacrebleu.corpus_bleu(self.preds, [self.goldens], force=True)
        if reset:
            self.preds = []
            self.goldens = []
        return {'bleu': bleu.score}


