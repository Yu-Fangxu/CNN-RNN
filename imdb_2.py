import logging
import os
import random
from copy import deepcopy
import glob
import io
import numpy as np
import torch
from torchtext.legacy.data import NestedField, Field, TabularDataset, Dataset, Example
from torchtext.legacy.data import BucketIterator
from torchtext.vocab import Vectors, GloVe, FastText
import csv
from Utils import clean_string, split_sents, process_labels, generate_ngrams, Clean_Str

from torch.utils.data import Dataset, DataLoader
from args import get_args
from torchtext.datasets import IMDB
import pandas as pd
from models.HAN import HAN
from models.Conv_GRNN import Conv_GRNN
from models.HAHNetwork import HAHNN
def split_train_valid(train_data, train_size, valid_part=0.1):
    random.shuffle(train_data)
    valid_size = round(valid_part * train_size)
    valid_data = train_data[:valid_size]
    train_data = train_data[valid_size:]
    return train_data, valid_data

class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]

class IMDB_2(TabularDataset):
    NAME = 'IMDB_2'
    NUM_CLASSES = 2
    IS_MULTILABEL = False

    TEXT_FIELD = Field(lower=True, batch_first=True, tokenize=Clean_Str, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train=os.path.join('IMDB_2', 'train.tsv'),
               validation=os.path.join('IMDB_2', 'dev.tsv'),
               test=os.path.join('IMDB_2', 'test.tsv'), **kwargs):
        return super(IMDB_2, cls).splits(
            path, train=train, validation=validation, test=test,
            format='tsv', fields=[('label', cls.LABEL_FIELD), ('text', cls.TEXT_FIELD)]
        )

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)

        train, val, test = cls.splits(path)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=True, device=device)

class IMDBHierarchical_2(IMDB_2):
    NESTING_FIELD = Field(batch_first=True, tokenize=Clean_Str, lower=True)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents)
if __name__ == "__main__":
    train_iter = IMDB(split='train',root="/home/fpy5052/CNN-RNN/datasets/")

    def tokenize(label, line):
        return line.split()

    tokens = []
    for label, line in train_iter:
        tokens += tokenize(label, line)
        
    dic = {"pos":"01", "neg":"10"}
    dirname = 'aclImdb'
    args = get_args()
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    path = os.path.join(args.data_dir, "IMDB", dirname, "train")
    examples = []
    for label in ['pos', 'neg']:
        for fname in glob.iglob(os.path.join(path, label, '*.txt')):
            with io.open(fname, 'r', encoding="utf-8") as f:
                text = f.readline()
            examples.append([dic[label], text])
    train_set, val_set = split_train_valid(examples, len(examples), valid_part=0.1)
    train_set = pd.DataFrame(train_set)
    val_set = pd.DataFrame(val_set)
    train_set.to_csv(r'/home/fpy5052/CNN-RNN/datasets/IMDB_2/train.tsv', index=False, header=None, sep="\t")
    val_set.to_csv(r'/home/fpy5052/CNN-RNN/datasets/IMDB_2/dev.tsv', index=False, header=None, sep="\t")
    # with open(r'/home/fpy5052/CNN-RNN/datasets/IMDB_2/train.tsv','w',newline='') as tsv_file:
        # writer=csv.writer(tsv_file,delimiter='\t')
        # for row in train_set:
            # writer.writerow(row)
    # with open(r'/home/fpy5052/CNN-RNN/datasets/IMDB_2/dev.tsv','w',newline='') as tsv_file:
        # writer=csv.writer(tsv_file,delimiter='\t')
        # for row in val_set:
            # writer.writerow(row)
    
    path = os.path.join(args.data_dir, "IMDB", dirname, "test")
    examples = []
    for label in ['pos', 'neg']:
        for fname in glob.iglob(os.path.join(path, label, '*.txt')):
            with io.open(fname, 'r', encoding="utf-8") as f:
                text = f.readline()
            examples.append([dic[label], text])
    examples = pd.DataFrame(examples)
    examples.to_csv(r'/home/fpy5052/CNN-RNN/datasets/IMDB_2/test.tsv', index=False, header=None, sep="\t")
    # with open(r'/home/fpy5052/CNN-RNN/datasets/IMDB_2/test.tsv','w',newline='') as tsv_file:
        # writer=csv.writer(tsv_file,delimiter='\t')
        # for row in val_set:
            # writer.writerow(row)
    vector = GloVe(name='6B', dim=300)

    dataset_class = IMDBHierarchical_2
    train_iter, dev_iter, test_iter = dataset_class.iters(args.data_dir,
                                                              args.word_vectors_file,
                                                              args.word_vectors_dir,
                                                              batch_size=args.batch_size,
                                                              device=args.gpu,
                                                              unk_init=UnknownWordVecCache.unk,
                                                              vectors=vector)
                                                              
    config = args
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)
    # model = HAN(config).to(args.gpu)
    model = Conv_GRNN(config).to(args.gpu)
    # model = HAHNN(config).to(args.gpu)
    # for batch_idx, batch in enumerate(test_iter):
    #     print(model(batch.text).shape)
    #     break
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')