import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="PyTorch deep learning models for document classification")

    parser.add_argument('--mode', type=str, default='non-static', choices=['rand', 'static', 'non-static'])
    parser.add_argument('--dataset', type=str, default='IMDB_2', choices=['Reuters', 'ELEC', 'IMDB', 'Yelp2014', 'IMDB_2'])
    parser.add_argument('--output_channel', type=int, default=100)
    parser.add_argument('--words_dim', type=int, default=300)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch_decay', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--word_num_hidden', type=int, default=50)
    parser.add_argument('--sentence_num_hidden', type=int, default=50)

    parser.add_argument('--word_vectors_dir', default=os.path.join(os.pardir, 'hedwig-data', 'embeddings', 'word2vec'))
    parser.add_argument('--word_vectors_file', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--save_path', type=str, default=os.path.join('model_checkpoints', 'han'))
    parser.add_argument('--resume_snapshot', type=str)
    parser.add_argument('--trained_model', type=str)
    parser.add_argument('--retrain', type=bool, default=True)
    parser.add_argument('--load_ckpt', default=None, help='Load parameters from checkpoint.')
    
    parser.add_argument('--no_cuda', action='store_false', dest='cuda')
    parser.add_argument('--gpu', type=str, default="cuda:0")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--data_dir', default='datasets')
    parser.add_argument('--output', default='experiment')
    parser.add_argument('--patient', type=int, default=10)
    args = parser.parse_args()
    return args