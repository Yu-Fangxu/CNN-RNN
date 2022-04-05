import torch
import torch.nn as nn
from models.TCN import *

class SentLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        sentence_num_hidden = config.sentence_num_hidden
        word_num_hidden = config.word_num_hidden
        target_class = config.target_class
        self.sentence_context_weights = nn.Parameter(torch.rand(2 * sentence_num_hidden, 1))
        self.sentence_context_weights.data.uniform_(-0.1, 0.1)
        self.sentence_gru = nn.GRU(2 * word_num_hidden, sentence_num_hidden, bidirectional=True, dropout=config.dropout, num_layers=2)
        self.sentence_linear = nn.Linear(2 * sentence_num_hidden, 2 * sentence_num_hidden, bias=True)
        self.fc = nn.Linear(2 * sentence_num_hidden , target_class)
        self.soft_sent = nn.Softmax()
        self.dropout = nn.Dropout(p=config.dropout)
    def forward(self,x):
        # x = self.dropout(x)
        sentence_h,_ = self.sentence_gru(x)
        x = torch.tanh(self.sentence_linear(sentence_h))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_sent(x.transpose(1,0))
        x = torch.mul(sentence_h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        x = self.fc(x.squeeze(0))
        return x

class SentLevelTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        sentence_num_hidden = config.sentence_num_hidden
        word_num_hidden = config.word_num_hidden
        target_class = config.target_class
        self.sentence_context_weights = nn.Parameter(torch.rand(2 * sentence_num_hidden, 1))
        self.sentence_context_weights.data.uniform_(-0.1, 0.1)
        self.tcn = TemporalConvNet(num_inputs=2 * sentence_num_hidden, num_channels=[100, 75, 50])
        self.tcn2 = TemporalConvNet(num_inputs=2 * sentence_num_hidden, num_channels=[100, 75, 50])
        self.sentence_linear = nn.Linear(2 * sentence_num_hidden, 2 * sentence_num_hidden, bias=True)
        self.fc = nn.Linear(2 * sentence_num_hidden , target_class)
        self.soft_sent = nn.Softmax()
        self.dropout = nn.Dropout(p=config.dropout)
    def forward(self,x):
        # x = self.dropout(x)
        # sentence_h,_ = self.sentence_gru(x)
        sentence_h1 = self.tcn(x)
        sentence_h2 = self.tcn(torch.flip(x, dims=[-1]))
        sentence_h = torch.cat([sentence_h1, sentence_h2], dim=1)
        sentence_h = sentence_h.permute(2, 0, 1)
        x = torch.tanh(self.sentence_linear(sentence_h))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_sent(x.transpose(1,0))
        x = torch.mul(sentence_h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        x = self.fc(x.squeeze(0))
        return x