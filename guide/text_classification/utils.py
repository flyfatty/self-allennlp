import json
import jieba
import pickle
import numpy as np
import os
import torch
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def metrics(y_truth_lst, y_hat_lst, y_proba_lst=None):
    assert len(y_truth_lst) == len(y_hat_lst)
    acc = round(accuracy_score(y_truth_lst, y_hat_lst), 4)
    prec = round(precision_score(y_truth_lst, y_hat_lst), 4)
    rec = round(recall_score(y_truth_lst, y_hat_lst), 4)
    f1 = round(2*prec*rec / (prec+rec+1e-10), 4)
    f_05 = round((1+0.5*0.5) *(prec*rec) / (0.5*0.5*prec+rec+1e-10), 4)
    if(y_proba_lst):
        auc = round(roc_auc_score(y_truth_lst, y_proba_lst), 4)
        return acc, prec, rec, f1, auc, f_05
    else:
        return acc, prec, rec, f1, f_05

def load_txt(path):
    with open(path, 'r') as f:
        data = [x.replace('\n','') for x in f.readlines() if len(x)>0]
    return data

def save_txt(data, path):
    with open(path,'w') as f:
        for item in data:
            line = item.replace('\n', '') + '\n'
            f.write(line)

def load_json(path):
    with open(path, 'r') as handle:
        data = json.load(handle)
    return data

def save_json(data, path):
    with open(path, 'w') as handle:
        json.dump(data, handle)


# def tokenizer(text, w2i, max_seq_len):
#     words = list(jieba.lcut(text))
#     if(len(words)<max_seq_len):
#         sequence = []
#         for w in words:
#             if(w in w2i):
#                 sequence.append(w2i[w])
#             else:
#                 sequence.append(w2i['unk'])
#         sequence += [0 for _ in range(max_seq_len-len(sequence))]
#         return sequence
#     else:
#         return []

def tokenizer_f(text, w2i, max_seq_len):
    words = list(jieba.lcut(text))
    if len(words) > max_seq_len:
        words = words[:max_seq_len]
    sequence = []
    for w in words:
        if(w in w2i):
            sequence.append(w2i[w])
        else:
            sequence.append(w2i['unk'])
    if len(words) < max_seq_len:
        sequence += [0 for _ in range(max_seq_len-len(sequence))]
    return sequence


def preprocess(data_path, w2i, max_seq_len, model_name='other', tokenizer=None):
    result = []
    if tokenizer is not None:
        print('Enable model tokenizer')
        model_tokenizer = tokenizer
    data = load_json(data_path)
    if 'data' in data:
        data = data.get('data')
    for item in data:
        text = item['text']
        if tokenizer is not None:
            if model_name == 'bert':
                sequence = model_tokenizer.encode(''.join(['[CLS]', text, '[SEP]']))
                # sequence = model_tokenizer.encode(text, add_special_tokens=True)
            elif model_name == 'xlnet':
                # sequence = model_tokenizer.encode(text, add_special_tokens=True)
                sequence = model_tokenizer.encode(''.join(['<cls>', text, '<sep>']))
            elif model_name == 'roberta':
                # sequence = model_tokenizer.encode(''.join(['[CLS]', text, '[SEP]']))
                sequence = model_tokenizer.encode(text, add_special_tokens=True)
            if(len(sequence)<max_seq_len):
                sequence += [0 for _ in range(max_seq_len-len(sequence))]
            else:
                sequence = []
        else:
            sequence = tokenizer_f(text, w2i, max_seq_len)
        if(sequence!=[]):
            label = item['label']
            result.append({
                'sequence' : sequence,
                'label' : label
            })
    return result


def get_batch_valid(data, batch_size, device, step):
    N = len(data)
    inp, label = [], []

    st_idx = int(batch_size*step)
    ed_idx = min((st_idx + batch_size), N)

    inp = [x['sequence'] for x in data[st_idx : ed_idx]]
    label = [x['label'] for x in data[st_idx : ed_idx]]
        
    inp_tensor = torch.tensor(inp).to(device)
    label_tensor = torch.tensor(label).to(device).view(-1)

    return inp_tensor, label_tensor

def get_batch(data, batch_size, device, sampling=None):
    assert sampling==None or (sampling>0 and sampling<1)
    N = len(data)
    if(sampling==None):
        selected_idx = np.random.randint(low=0, high=N, size=batch_size)
        inp = [data[x]['sequence'] for x in selected_idx]
        label = [data[x]['label'] for x in selected_idx]
    else:
        n = max(int(batch_size*sampling), 1)
        positive = np.random.choice([x for x in data if x['label']==1], size=n)
        negative = np.random.choice([x for x in data if x['label']==0], size=batch_size-n)
        batch = shuffle(list(positive) + list(negative))
        inp = [x['sequence'] for x in batch]
        label = [x['label'] for x in batch]

    inp_tensor = torch.tensor(inp).to(device)
    label_tensor = torch.tensor(label).to(device).view(-1)

    return inp_tensor, label_tensor


def load_all_data(data_path, embd_path, max_seq_len, model_name='other', tokenizer=None):
    
    train_file_name = [x for x in os.listdir(data_path) if 'train' in x]
    valid_file_name = [x for x in os.listdir(data_path) if 'valid' in x]

    if(len(train_file_name)!=1 or len(valid_file_name)!=1):
        raise ValueError('Too many train and validation data. Found {} train files and {} validation files'.format(len(train_file_name), len(valid_file_name)))
    else:
        train_file_name = train_file_name[0]
        valid_file_name = valid_file_name[0]
    train_data_path = os.path.join(data_path, train_file_name)
    valid_data_path = os.path.join(data_path, valid_file_name)
    test_data_path = os.path.join(data_path, 'test.json')

    print('Loading training data:{}'.format(train_data_path))
    print('Loading validation data:{}'.format(valid_data_path))
    print('Loading test data:{}'.format(test_data_path))

    if tokenizer is not None:
        train_data = preprocess(train_data_path, None, max_seq_len, model_name, tokenizer)
        valid_data = preprocess(valid_data_path, None, max_seq_len, model_name, tokenizer)
        test_data = preprocess(test_data_path, None, max_seq_len, model_name, tokenizer)

        print('Training data size {}, validation data size {}, test data size {}'.format(len(train_data), len(valid_data), len(test_data)))
        N = len(load_json(train_data_path).get('data'))
        ratio = 100*round(len(train_data)/N, 4)
        print('Preserve {}%  training data at max_seq_len={}'.format(ratio, max_seq_len))
        return None, None, train_data, valid_data, test_data
    else:
        w2i = pickle.load(open(os.path.join(embd_path, 'w2i.pkl'),'rb'))
        embd = np.load(os.path.join(embd_path, 'matrix.npy'))

        train_data = preprocess(train_data_path, w2i, max_seq_len, model_name='other')
        valid_data = preprocess(valid_data_path, w2i, max_seq_len, model_name='other')
        test_data = preprocess(test_data_path, w2i, max_seq_len, model_name='other')
        print('Vocab size {}'.format(embd.shape[0]))
        print('Training data size {}, validation data size {}, test data size {}'.format(len(train_data), len(valid_data), len(test_data)))
        N = len(load_json(train_data_path).get('data'))
        ratio = 100*round(len(train_data)/N, 4)
        print('Preserve {}%  training data at max_seq_len={}'.format(ratio, max_seq_len))

        return w2i, embd, train_data, valid_data, test_data


def sentence_split(text):
    seperators = ['。', '！', '？', '?', '!']
    for sep in seperators:
        text = text.replace(sep, '{}@@@@'.format(sep))
        
    sent_lst = [x.replace('\n', '') for x in text.split('@@@@') if x!=[]]
    return sent_lst


def select_data_bank(target_grades=None):
    data_bank = load_json('/share/作文批改/data/spider/essay/data.json')
    valid_grades = set([x['grade'] for x in data_bank])
    for grade in target_grades:
        if(grade not in valid_grades):
            raise ValueError('Found invalid grade={}'.format(grade))

    if(target_grades is not None):
        print('Select data from {}'.format(target_grades))
        data_bank = [x['content'] for x in data_bank if x['grade'] in target_grades]
    else:
        print('Use full data bank')
        data_bank = [x['content'] for x in data_bank]
    return data_bank

import re
def merge_symmetry(sentences, symmetry=('“','”')):
    '''合并对称符号，如双引号'''
    effective_ = []
    merged = True
    for index in range(len(sentences)):       
        if symmetry[0] in sentences[index] and symmetry[1] not in sentences[index]:
            merged = False
            effective_.append(sentences[index])
        elif symmetry[1] in sentences[index] and not merged:
            merged = True
            effective_[-1] += sentences[index]
        elif symmetry[0] not in sentences[index] and symmetry[1] not in sentences[index] and not merged :
            effective_[-1] += sentences[index]
        else:
            effective_.append(sentences[index])
    return [i.strip() for i in effective_ if len(i.strip()) > 0]
    
def split_content(paragraph):
    """由段落切分成句子"""
    sentences = re.split(r"(。|！|\…\…)", paragraph)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    sentences = [i.strip() for i in sentences if len(i.strip()) > 0]
    for j in range(1, len(sentences)):
        if sentences[j][0] == '”':
            sentences[j-1] = sentences[j-1] + '”'
            sentences[j] = sentences[j][1:]  

    return merge_symmetry(sentences)