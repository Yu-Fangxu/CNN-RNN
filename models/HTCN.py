import torch
import torch.nn as nn

from models.SentAttention import SentLevelTCN
from models.WordAttention import WordLevelTCN

class HTCN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.word_attention_tcn = WordLevelTCN(config)
        self.sentence_attention_tcn = SentLevelTCN(config)

    def forward(self, x,  **kwargs):
        x = x.permute(1, 2, 0) # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        # start_time = time.time()
        for i in range(num_sentences):
            
            word_attn = self.word_attention_tcn(x[i, :, :])

            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        # end_time = time.time()
        # print(end_time - start_time)
        # word_attentions: seq_len, batch_size, embedding_size
        word_attentions = word_attentions.permute(1, 2, 0)
        return self.sentence_attention_tcn(word_attentions)