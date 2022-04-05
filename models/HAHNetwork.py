import torch
import torch.nn as nn

from models.SentAttention import SentLevelRNN
from models.WordAttention import WordLevelRCNN
from models.WordAttention import WordLevelDilatedRCNN, WordLevelRTCN

class HAHNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.word_attention_rnn = WordLevelRCNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)

    def forward(self, x,  **kwargs):
        x = x.permute(1, 2, 0) # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        return self.sentence_attention_rnn(word_attentions)
        
class HAHNN_Dilation(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.word_attention_rnn = WordLevelDilatedRCNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)

    def forward(self, x,  **kwargs):
        x = x.permute(1, 2, 0) # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        return self.sentence_attention_rnn(word_attentions)
        
class HAHNN_TCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.word_attention_rnn = WordLevelRTCN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)

    def forward(self, x,  **kwargs):
        x = x.permute(1, 2, 0) # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        return self.sentence_attention_rnn(word_attentions)