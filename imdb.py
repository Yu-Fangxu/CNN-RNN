import logging
import os
import random
from copy import deepcopy

import numpy as np
import torch
from torchtext.legacy.data import NestedField, Field, TabularDataset
from torchtext.legacy.data import BucketIterator
from torchtext.vocab import Vectors, GloVe, FastText

from Utils import clean_string, split_sents, process_labels, generate_ngrams

from torch.utils.data import Dataset, DataLoader
from args import get_args
from models.HAN import HAN
from models.Conv_GRNN import Conv_GRNN
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

class IMDB(TabularDataset):
    NAME = 'IMDB'
    NUM_CLASSES = 10
    IS_MULTILABEL = False

    TEXT_FIELD = Field(lower=True, batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train=os.path.join('IMDB', 'train.tsv'),
               validation=os.path.join('IMDB', 'dev.tsv'),
               test=os.path.join('IMDB', 'test.tsv'), **kwargs):
        return super(IMDB, cls).splits(
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


class IMDBHierarchical(IMDB):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string)
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=split_sents)
    
if __name__ == "__main__":
    args = get_args()
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    vector = GloVe(name='6B', dim=300)
    # vector = FastText()
    dataset_class = IMDBHierarchical
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
    x = torch.randn(64, 5, 50)
    # model = HAN(config).to(args.gpu)
    model = Conv_GRNN(config).to(args.gpu)
    for batch_idx, batch in enumerate(train_iter):
        #print(model(batch.text).shape)
        print(batch.text.shape)
        
        
    print(len(train_iter.dataset))
    print(len(test_iter.dataset))