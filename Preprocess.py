from Utils import *
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# load data
word_freq = {}
train_doc = []
f = open('./data/IMDB/imdb-train.txt.tok', encoding='utf-8')
for line in f.readlines():
    temp = line.strip().lower()
    train_doc.append(temp)
    words = temp.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
f.close()

test_doc = []
f = open('./data/IMDB/imdb-test.txt.tok', encoding='utf-8')
for line in f.readlines():
    temp = line.strip().lower()
    test_doc.append(temp)
f.close()

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_doc), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(vocab(['here', 'is', 'an', 'example']))