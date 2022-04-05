import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TCN import *
from models.AttentionPooling import AttPooling
class WordLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True, dropout=config.dropout, num_layers=2)
        # self.lstm = nn.LSTM(words_dim, word_num_hidden, dropout=config.dropout, num_layers=1,
        #                    bidirectional=self.is_bidirectional, batch_first=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()
        self.dropout = nn.Dropout(p=config.dropout)
    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else :
            print("Unsupported mode")
            exit()
        x = self.dropout(x)
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        # x: 1 * batch_size * (word_num_hidden * 2)
        return x

class WordLevelCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.filter_sizes = [3, 4, 5]
        self.n_filters = 100
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.word_level_convs = nn.ModuleList([
                                nn.Conv2d(in_channels = 1, 
                                          out_channels = self.n_filters, 
                                          kernel_size = (fs, words_dim)) 
                                    for fs in self.filter_sizes
                                    ])
        self.attn_pooling = AttPooling(self.n_filters)
        self.dropout = nn.Dropout(p=config.dropout)
    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else :
            print("Unsupported mode")
            exit()
        x = self.dropout(x)
        # x : num_words, batch_size, embedding_size
        x = x.permute(1, 0, 2)
        
        # x: batch_size, num_words, embedding_size
        
        x = x.unsqueeze(dim=1)
        
        # x: batch_size, 1, num_words, words_dim
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.word_level_convs]
        
        # x: batch_size, n_filters, num_words - filter_sizes[n] + 1
        
        # x = [F.max_pool1d(conv, conv.shape[2]).squeeze(dim=2) for conv in x]
        x = [F.max_pool1d(conv, conv.shape[2]) for conv in x]           #max_pooling
        # x = [self.attn_pooling(conv) for conv in x]
        # x[i]: batch_size, n_filters, 1
        x = torch.cat(x, dim=-1)

        x = torch.mean(x, dim=-1)
        # opt: average / concat
        
        x = x.unsqueeze(dim=0)

        return x

class WordLevelRCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.filter_sizes = [3, 4, 5]
        self.n_filters = words_dim
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.word_level_convs = nn.ModuleList([
                                nn.Conv2d(in_channels = 1, 
                                          out_channels = self.n_filters, 
                                          kernel_size = (fs, words_dim)) 
                                    for fs in self.filter_sizes
                                    ])
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.linear2= nn.Linear(words_dim, words_dim)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()
        self.attn_pooling = AttPooling(self.n_filters)
        # self.resblock = ResBlock(config)
        self.dropout = nn.Dropout(p=config.dropout)
    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else :
            print("Unsupported mode")
            exit()
        # print(x.shape)
        # x : num_words, batch_size, embedding_size
        
        x = self.dropout(x)
        
        x = x.permute(1, 0, 2)
        # x: batch_size, num_words, embedding_size
        
        x = x.unsqueeze(dim=1)
        
        # x: batch_size, 1, num_words, words_dim
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.word_level_convs]
        
        # x: batch_size, n_filters, num_words
        # x = self.resblock(x)
        x = [F.max_pool1d(x[i], kernel_size=self.filter_sizes[i]) for i in range(len(self.filter_sizes))]   #max_pooling
        
        # x: batch_size, embedding_size, seq_len
        x = torch.cat(x, dim=-1)
        
        x = x.permute(2, 0, 1)
        
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        # x: 1 * batch_size * (word_num_hidden * 2)
        return x

class WordLevelDilatedCNN(WordLevelCNN):
    def __init__(self, config):
        super().__init__(config)
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.filter_sizes = [3, 4, 5]
        self.n_filters = words_dim
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.dilation = 3
        self.word_level_convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = self.n_filters, 
                                              kernel_size = (fs, words_dim),
                                              dilation = (self.dilation, 1)) 
                                        for fs in self.filter_sizes
                                        ])
        self.dropout = nn.Dropout(p=config.dropout)
class WordLevelDilatedRCNN(WordLevelRCNN):
    def __init__(self, config):
        super().__init__(config)
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.filter_sizes = [3, 4, 5]
        self.n_filters = words_dim
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.word_level_convs = nn.ModuleList([
                                nn.Conv2d(in_channels = 1, 
                                          out_channels = self.n_filters, 
                                          kernel_size = (fs, words_dim)) 
                                    for fs in self.filter_sizes
                                    ])
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.linear2= nn.Linear(words_dim, words_dim)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()
        self.dilation = 3
        self.word_level_convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = self.n_filters, 
                                              kernel_size = (fs, words_dim),
                                              dilation = (self.dilation, 1)) 
                                        for fs in self.filter_sizes
                                        ])
        self.dropout = nn.Dropout(p=config.dropout)
        
class WordLevelRTCN(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.filter_sizes = [3, 4, 5]
        self.n_filters = words_dim
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.tcn = TemporalConvNet(num_inputs=words_dim, num_channels=[100, 100, 100], kernel_size=3)
        # self.tcn2 = TemporalConvNet(num_inputs=words_dim, num_channels=[100, 300])
        self.GRU = nn.GRU(100, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.linear2= nn.Linear(words_dim, words_dim)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()
        self.dropout = nn.Dropout(p=config.dropout)
    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else :
            print("Unsupported mode")
            exit()
        # print(x.shape)
        # x : num_words, batch_size, embedding_size
        
        x = self.dropout(x)
        
        x = x.permute(1, 2, 0)
        # x: batch_size, embedding_size, num_words
        x = self.tcn(x)
        # x: batch_size, embedding_size, num_words
        # x2 = self.tcn2(torch.flip(x, dims=[-1]))
        # x2 = torch.flip(x2, dims=[-1])
        # x = torch.cat([x1, x2], dim=1)
        
        x = x.permute(2, 0, 1)
        
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        # x: 1 * batch_size * (word_num_hidden * 2)
        return x

class WordLevelTCN(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.n_filters = words_dim
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.tcn = TemporalConvNet(num_inputs=words_dim, num_channels=[100, 75, 50])
        self.tcn2 = TemporalConvNet(num_inputs=words_dim, num_channels=[100, 75, 50])
        # self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.linear2= nn.Linear(words_dim, words_dim)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()
        self.dropout = nn.Dropout(p=config.dropout)
    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else :
            print("Unsupported mode")
            exit()
        # print(x.shape)
        # x : num_words, batch_size, embedding_size
        
        x = self.dropout(x)
        
        x = x.permute(1, 2, 0)
        # x: batch_size, embedding_size, num_words
        x1 = self.tcn(x)
        # x: batch_size, embedding_size, num_words
        x2 = self.tcn2(torch.flip(x, dims=[-1]))
        x = torch.cat([x1, x2], dim=1)
        h = x.permute(2, 0, 1)
        # h: num_words, batch_size, hidden_size
        # h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        # x: 1 * batch_size * (word_num_hidden * 2)
        return x

class ResBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        words_dim = config.words_dim
        self.mode = "non-static"
        self.n_filters = config.words_dim
        
        self.conv_region1 = nn.Conv2d(1, self.n_filters, (3, words_dim))
        self.conv1 = nn.Conv2d(self.n_filters, self.n_filters, (3, 1), stride=1)
        self.conv_region2 = nn.Conv2d(1, self.n_filters, (4, words_dim))
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters, (4, 1), stride=1)
        self.conv_region3 = nn.Conv2d(1, self.n_filters, (5, words_dim))
        self.conv3 = nn.Conv2d(self.n_filters, self.n_filters, (5, 1), stride=1)
        # self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 1, 1))  # bottom
        self.padding3 = nn.ZeroPad2d((0, 0, 1, 2))
        self.padding4 = nn.ZeroPad2d((0, 0, 1, 2))
        self.padding5 = nn.ZeroPad2d((0, 0, 2, 2))
        self.padding6 = nn.ZeroPad2d((0, 0, 2, 2))
        self.relu = nn.ReLU()
        # self.fc = nn.Linear(self.n_filters, config.target_class)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):  
        # x = x.unsqueeze(dim=1)  # [batch_size, 1, seq_len, embed]

        # x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        # x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        # x = self.relu(x)
        # x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        # x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        # x = self.relu(x)
        # x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        # while x.size()[2] > 1:
        # x = self._block(x)
        # x = x.squeeze(dim=-1)
        # x = x.squeeze(dim=-1)
        # print(x.shape)
        # x = x.view(x.size()[0], 2 * self.n_filters)
        # x = torch.cat((x[:,:,0], x[:,:,1]), dim=0)
        # x = self.fc(self.dropout(x))
        
        x1 = self.conv_region1(x)
        x1 = self.relu(self._block1(x1))
        x2 = self.conv_region2(x)
        x2 = self.relu(self._block2(x2))
        x3 = self.conv_region3(x)
        x3 = self.relu(self._block3(x3))
        return [x1.squeeze(3), x2.squeeze(3), x3.squeeze(3)]
        
    def _block1(self, x):
        x = self.padding2(x)
        px = x
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = x + px  # short cut
        return x
    def _block2(self, x):
        x = self.padding4(x)
        px = x
        x = self.padding3(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x + px  # short cut
        return x
    def _block3(self, x):
        x = self.padding6(x)
        px = x
        x = self.padding5(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = x + px  # short cut
        return x