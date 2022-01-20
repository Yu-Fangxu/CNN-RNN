import torch
import torch.nn as nn

from models.SentAttention import SentLevelRNN
from models.WordAttention import WordLevelCNN

import time
class Conv_GRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.word_cnn = WordLevelCNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)

    def forward(self, x,  **kwargs):
        x = x.permute(1, 2, 0) # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_cnn(x[i, :, :])
            # word_attn: 1 * batch_size * n_filters
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        #word_attentions: num_sentences * batch_size * n_filters
        return self.sentence_attention_rnn(word_attentions)

        