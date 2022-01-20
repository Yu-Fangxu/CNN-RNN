import re
import numpy as np
import nltk
import spacy
# import TextBlob
def Seq_Split(seq, split_list):
    seq_array = np.array(seq)
    split_indexes = []
    for split_symbol in split_list:
        tmp = np.where(seq_array == split_symbol)
        split_indexes.extend(tmp[0].tolist())
    split_indexes.sort()

    result = []
    result_len = []
    if len(split_indexes)==0:
        result.append(seq)
        result_len.append(len(seq))
    else:
        split_start = 0
        for split_index in split_indexes:
            tmp = seq[split_start:split_index+1]
            if len(tmp) <= 1:
                continue
            result.append(tmp)
            result_len.append(len(tmp))
            split_start = split_index+1

        if split_start != len(seq):
            result.append(seq[split_start:])
            result_len.append(len(seq[split_start:]))
    return [result_len, result]

def Seq_Max_Len(seqs):
    max_len = 0
    for i in range(len(seqs)):
        if len(seqs[i]) > max_len:
            max_len = len(seqs[i])
    return max_len

def Seq_Min_Len(seqs):
    min_len = 999
    for i in range(len(seqs)):
        if len(seqs[i]) < min_len:
            min_len = len(seqs[i])
    return min_len

def Get_Minibatches_Index(n, minibatch_size, shuffle=False):
    index_list = np.arange(n, dtype='int32')

    if shuffle:
        np.random.shuffle(index_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n//minibatch_size):
        minibatches.append(index_list[minibatch_start:minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return minibatches

def Prepare_Data_For_CNN(seqs, max_len, num_words, filter_h):
    max_len_pad = max_len + 2*(filter_h-1)

    new_seqs = []
    for seq in seqs:
        if len(seq)>max_len:
            seq = seq[:max_len]

        new_seq = [num_words-1]*(filter_h -1) + seq
        new_seq = new_seq + [num_words-1]*(max_len_pad-len(new_seq))
        new_seqs.append(new_seq)

    return new_seqs

def Clean_Str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.replace('<br />', ' ').replace('<br>', ' ').replace('\\n', ' ').replace('&#xd;', ' ')
    return string.strip().lower().split()
    
def clean_string(string):
    """
    Performs tokenization and string cleaning for the Reuters dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`.]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip().split()

def split_sents(string):
    # return nltk.sent_tokenize(string.strip())
    string = re.sub(r"[!?]"," ", string)
    return string.strip().split('.')

def generate_ngrams(tokens, n=2):
    n_grams = zip(*[tokens[i:] for i in range(n)])
    tokens.extend(['-'.join(x) for x in n_grams])
    return tokens


def load_json(string):
    split_val = json.loads(string)
    return np.asarray(split_val, dtype=np.float32)


def char_quantize(string, max_length=1000):
    identity = np.identity(len(ReutersCharQuantized.ALPHABET))
    quantized_string = np.array([identity[ReutersCharQuantized.ALPHABET[char]] for char in list(string.lower()) if char in ReutersCharQuantized.ALPHABET], dtype=np.float32)
    if len(quantized_string) > max_length:
        return quantized_string[:max_length]
    else:
        return np.concatenate((quantized_string, np.zeros((max_length - len(quantized_string), len(ReutersCharQuantized.ALPHABET)), dtype=np.float32)))


def process_labels(string):
    """
    Returns the label string as a list of integers
    """
    return [float(x) for x in string]

if __name__ == "__main__":
    a = "Hello World! what about it. how come"
    a = clean_string(a)
    nlp = spacy.load('en_core_web_sm')
    print(a)
    print(split_sents(a))
