import torch
import torch.nn.functional as F
from fastNLP import LossBase
from fastNLP import seq_len_to_mask

class MTLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_loss(self, pred, tgt_tokens, tgt_seq_len):
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
        loss = F.cross_entropy(target=tgt_tokens, input=pred[:, :-1].transpose(1, 2))
        return loss
