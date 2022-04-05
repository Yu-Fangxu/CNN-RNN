import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):

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
        self.dropout = nn.Dropout(config.dropout)
        ks = len(self.filter_sizes)
        self.fc1 = nn.Linear(ks * self.n_filters, config.target_class)
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
            
        x = x.unsqueeze(dim=1)

        # x: batch_size, 1, num_words, words_dim
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.word_level_convs]
        
        # x: batch_size, n_filters, num_words - filter_sizes[n] + 1
        
        # x = [F.max_pool1d(conv, conv.shape[2]).squeeze(dim=2) for conv in x]
        x = [F.max_pool1d(conv, conv.shape[2]) for conv in x]
        # x[i]: batch_size, n_filters, 1
        x = torch.cat(x, dim=1).squeeze(dim=-1)
        # x: batch_size, 3 * n_filters, 1
        x = self.dropout(x)
        
        logit = self.fc1(x)
        # opt: average / concat

        return logit