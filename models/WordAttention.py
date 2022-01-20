import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()

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
        
        # x : num_words, batch_size, embedding_size
        
        x = x.permute(1, 0, 2)
        
        # x: batch_size, num_words, embedding_size
        
        x = x.unsqueeze(dim=1)
        
        # x: batch_size, 1, num_words, words_dim
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.word_level_convs]
        
        # x: batch_size, n_filters, num_words - filter_sizes[n] + 1
        
        # x = [F.max_pool1d(conv, conv.shape[2]).squeeze(dim=2) for conv in x]
        x = [F.max_pool1d(conv, conv.shape[2]) for conv in x]
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
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()

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
        
        # x : num_words, batch_size, embedding_size
        
        x = x.permute(1, 0, 2)
        
        # x: batch_size, num_words, embedding_size
        
        x = x.unsqueeze(dim=1)
        
        # x: batch_size, 1, num_words, words_dim
        
        x = [F.relu(conv(x)).squeeze(3).unsqueeze(0) for conv in self.word_level_convs]
        
        # x: batch_size, n_filters, num_words - filter_sizes[n] + 1

        
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        # x: 1 * batch_size * (word_num_hidden * 2)
        return x
