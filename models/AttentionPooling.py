import torch
import torch.nn as nn
import torch.nn.functional as F

class AttPooling(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        # self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.fs = 3
        self.in_channels = n_filters
        self.out_channels = n_filters // 2
        self.conv = nn.Conv1d(in_channels = n_filters, 
                              out_channels = self.out_channels, 
                              kernel_size = self.fs,
                              padding=1)
        # self.word_context_weights = nn.Parameter(torch.rand(1, self.out_channels, 1))#n_filters // 2, 1
        self.word_context_weights = nn.Linear(self.out_channels, 1)
        # self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_max = nn.Softmax()
    
    def forward(self, x):
        #x: batch_size, n_filters, seq_len
        h = self.conv(x)
        #x: batch_size, n_filters // 2, seq_len
        h = h.permute(0, 2, 1)
        # print(h.shape)
        #x: batch_size, seq_len, n_filters // 2
        h = self.word_context_weights(h)
        #x: batch_size, seq_len, 1
        h = self.soft_max(h, dim=1)
        #x: batch_size, seq_len, 1
        h = torch.matmul(x, h)
        #x: batch_size, n_filters, 1
        return h