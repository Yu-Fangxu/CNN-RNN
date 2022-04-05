import logging
import os
import random
import time
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
import torch.onnx
import torch.nn.functional as F
from torchtext.vocab import Vectors, GloVe, FastText

from imdb import IMDB# IMDBHierarchical as IMDB
from imdb_2 import IMDBHierarchical_2 as IMDB_2
from elec import ELECHierarchical as ELEC
from args import get_args
from models.HAN import HAN, HAN_TCN
from models.Conv_GRNN import Conv_GRNN, Conv_GRNN_Dilation
from models.HAHNetwork import HAHNN, HAHNN_Dilation, HAHNN_TCN
from models.TextCNN import TextCNN
from models.DPCNN import DPCNN
from models.HTCN import HTCN
from Utils import *
from tensorboardX import SummaryWriter
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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



if __name__ == "__main__":
    args = get_args()
    torch.backends.cudnn.deterministic = False
    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    eps = 1e-7
    
    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('Note: You are using GPU for training')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print('Warning: Using CPU for training')
        
    vector = GloVe(name='6B', dim=300)
    # vector = FastText()
    
    dataset_map = {
        'IMDB': IMDB,
        'IMDB_2': IMDB_2,
        'ELEC': ELEC
    }
    model_map = {
        'HAN': HAN,
        'Conv_GRNN': Conv_GRNN,
        'HAHNN': HAHNN,
        'TextCNN': TextCNN,
        'DPCNN': DPCNN,
        'HAHNN_TCN': HAHNN_TCN,
        'HAN_TCN': HAN_TCN,
        'HTCN':HTCN
    }
    dataset_class = dataset_map[args.dataset]
    train_iter, dev_iter, test_iter = dataset_class.iters(args.data_dir,
                                                          args.word_vectors_file,
                                                          args.word_vectors_dir,
                                                          batch_size=args.batch_size,
                                                          device=args.gpu,
                                                          unk_init=UnknownWordVecCache.unk,
                                                          vectors=vector)
                                                              
    config = deepcopy(args)
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)
    print("Model_name:", args.model)
    
    # set model
    model = model_map[args.model](config).to(args.gpu)
    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    
    def save_checkpoint(filename):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename)
    
    def eval_model():
        model.eval()
        with torch.no_grad():
            val_loss = 0
            n_correct, n_total = 0, 0
            for batch_idx, batch in enumerate(dev_iter):
                # batch.text = batch.text.cuda()
                
                scores = model(batch.text)
                
                if 'is_multilabel' in config and config['is_multilabel']:
                    predictions = F.sigmoid(scores).round().long()
                    for tensor1, tensor2 in zip(predictions, batch.label):
                        if np.array_equal(tensor1, tensor2):
                            n_correct += 1
                    loss = F.binary_cross_entropy_with_logits(scores, batch.label.float())
                else:
                    preds = torch.max(scores, 1)[1]  # get label from max score per example
                    n_correct += (preds == torch.max(batch.label.data, 1)[1]).sum()
                    loss = F.cross_entropy(scores, torch.argmax(batch.label.data, dim=1))
                    
                val_loss += loss
                n_total += batch.batch_size
            val_acc = 100. * n_correct / n_total
            val_loss /= n_total
            
            writer.add_scalar("validation_loss", val_loss, epoch)
            writer.add_scalar("validation_accuracy", val_acc, epoch)    
        
            return val_loss.item()
            
            
    os.makedirs('ckpt/%s' % args.dataset, exist_ok=True)
    if args.load_ckpt:
        ckpt_path = args.load_ckpt
    else:
        ckpt_path = os.path.join('ckpt', args.dataset, 'model.ckpt')
    
    if os.path.exists(ckpt_path) and not args.retrain:
        # print("Load checkpoint from {}".format(ckpt_path))
        print("Load trained model")
        checkpoint = torch.load(f'ckpt/{args.dataset}/{args.model}.ckpt')
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        save_checkpoint(f'ckpt/{args.dataset}/{args.model}.ckpt')
        args.retrain = True
        start_epoch = 1
    
    # def train_model():    
    writer = SummaryWriter(comment=args.output)
    if args.retrain:
        print("start training")
        start_time = time.time()
        best_loss = 100
        patient = args.patient
        for epoch in tqdm(range(args.epochs)):
            n_correct, n_total = 0, 0
            train_loss = 0
            model.train()
            for batch_idx, batch in enumerate(train_iter):
                # batch.text = batch.text.cuda()
                optimizer.zero_grad()
                # b_time = time.time()
                scores = model(batch.text)

                if 'is_multilabel' in config and config['is_multilabel']:
                    predictions = F.sigmoid(scores).round().long()
                    for tensor1, tensor2 in zip(predictions, batch.label):
                        if np.array_equal(tensor1, tensor2):
                            n_correct += 1
                    loss = F.binary_cross_entropy_with_logits(scores, batch.label.float())
                else:
                    # for tensor1, tensor2 in zip(torch.argmax(scores, dim=1), torch.argmax(batch.label.data, dim=1)):
                        # if np.array_equal(tensor1, tensor2):
                            # n_correct += 1
                    preds = torch.max(scores, 1)[1]  # get label from max score per example

                    n_correct += (preds == torch.max(batch.label.data, 1)[1]).sum()
                    loss = F.cross_entropy(scores, torch.argmax(batch.label.data, dim=1))
                
                train_loss += loss.item()
                n_total += batch.batch_size
                
                
                loss.backward()
                optimizer.step()
                # e_time = time.time()
                # print(e_time - b_time)
            

            StepLR.step()
            train_acc = 100. * n_correct / n_total
            train_loss = train_loss / n_total
            patient -= 1
            
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_accuracy", train_acc, epoch)
            val_loss = eval_model()
            if val_loss < best_loss:
                best_loss = val_loss
                patient = args.patient
                save_checkpoint(f'ckpt/{args.dataset}/{args.model}.ckpt')
            elif patient <= 0:
                print("Early stop.")
                break
   
        print("Train time:", time.time() - start_time)
    else:
        print("continue training")
        start_time = time.time()
        best_loss = 100
        patient = args.patient
        for epoch in tqdm(range(args.epochs)):
            n_correct, n_total = 0, 0
            train_loss = 0
            model.train()
            for batch_idx, batch in enumerate(train_iter):
                # batch.text = batch.text.cuda()
                
                optimizer.zero_grad()
                # b_time = time.time()
                scores = model(batch.text[0])

                if 'is_multilabel' in config and config['is_multilabel']:
                    predictions = F.sigmoid(scores).round().long()
                    for tensor1, tensor2 in zip(predictions, batch.label):
                        if np.array_equal(tensor1, tensor2):
                            n_correct += 1
                    loss = F.binary_cross_entropy_with_logits(scores, batch.label.float())
                else:
                    # for tensor1, tensor2 in zip(torch.argmax(scores, dim=1), torch.argmax(batch.label.data, dim=1)):
                        # if np.array_equal(tensor1, tensor2):
                            # n_correct += 1
                    preds = torch.max(scores, 1)[1]  # get label from max score per example

                    n_correct += (preds == torch.max(batch.label.data, 1)[1]).sum()
                    loss = F.cross_entropy(scores, torch.argmax(batch.label.data, dim=1))
                
                train_loss += loss.item()
                n_total += batch.batch_size
                
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                # e_time = time.time()
                # print(e_time - b_time)
            

            StepLR.step()
            train_acc = 100. * n_correct / n_total
            train_loss = train_loss / n_total
            patient -= 1
            
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_accuracy", train_acc, epoch)
            val_loss = eval_model()
            if val_loss < best_loss:
                best_loss = val_loss
                patient = args.patient
                save_checkpoint(f'ckpt/{args.dataset}/{args.model}.ckpt')
            elif patient <= 0:
                print("Early stop.")
                break
    writer.close()
    # test
    checkpoint = torch.load(f'ckpt/{args.dataset}/{args.model}.ckpt')
    # Load checkpoint with lowest val loss
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        print("start testing")
        test_loss = 0
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(test_iter):
            # batch.text = batch.text.cuda()
            scores = model(batch.text)
            if 'is_multilabel' in config and config['is_multilabel']:
                predictions = F.sigmoid(scores).round().long()
                for tensor1, tensor2 in zip(predictions, batch.label):
                    if np.array_equal(tensor1, tensor2):
                        n_correct += 1
            else:
                preds = torch.max(scores, 1)[1]  # get label from max score per example
                n_correct += (preds == torch.max(batch.label.data, 1)[1]).sum()

            n_total += batch.batch_size
        test_acc = 100. * n_correct / n_total
    
    print("test_acc", test_acc.item(),  "%")