import torch
import torch.nn as nn
import torch.nn.functional as F

    
class DPCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.n_filters = 250
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
        
        self.conv_region = nn.Conv2d(1, self.n_filters, (3, words_dim), stride=1)
        self.conv = nn.Conv2d(self.n_filters, self.n_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.n_filters, config.target_class)
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.dropout(self.embed(x))
        elif self.mode == 'static':
            x = self.dropout(self.static_embed(x))
        elif self.mode == 'non-static':
            x = self.dropout(self.non_static_embed(x))
        else :
            print("Unsupported mode")
            exit()
            
        x = x.unsqueeze(dim=1)  # [batch_size, 1, seq_len, embed]

        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 1:
            x = self._block(x)
        x = x.squeeze(dim=-1)
        x = x.squeeze(dim=-1)
        # print(x.shape)
        # x = x.view(x.size()[0], 2 * self.n_filters)
        # x = torch.cat((x[:,:,0], x[:,:,1]), dim=0)
        x = self.fc(self.dropout(x))
        return x
        
    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px  # short cut
        return x
