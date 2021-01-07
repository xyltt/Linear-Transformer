import torch
from fastNLP.modules.decoder.seq2seq_state import TransformerState, State

class LinearTransformerState(State):
    def __init__(self, encoder_output, encoder_mask, num_decoder_layer):
        """
        与TransformerSeq2SeqDecoder对应的State，

        :param torch.FloatTensor encoder_output: bsz x encode_max_len x encoder_output_size, encoder的输出
        :param torch.ByteTensor encoder_mask: bsz x encode_max_len 为1的地方需要attend
        :param int num_decoder_layer: decode有多少层
        """
        super().__init__(encoder_output, encoder_mask)
        self.encoder_key = [None] * num_decoder_layer
        self.encoder_value = [None] * num_decoder_layer
        self.decoder_k_sum = [0] * num_decoder_layer
        self.decoder_kv_sum = [0] * num_decoder_layer
        self.decode_length = 0

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.encoder_key = self._reorder_state(self.encoder_key, indices)
        self.encoder_value = self._reorder_state(self.encoder_value, indices)
        self.decoder_k_sum = self._reorder_state(self.decoder_k_sum, indices)
        self.decoder_kv_sum = self._reorder_state(self.decoder_kv_sum, indices)
