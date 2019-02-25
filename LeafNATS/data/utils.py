'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import re
import glob
import shutil
import random
import numpy as np

import torch
from torch.autograd import Variable
'''
Construct vocabulary
'''
def construct_vocab(file_, max_size=200000, mincount=5):
    vocab2id = {'<s>': 2, '</s>': 3, '<pad>': 1, '<unk>': 0, '<stop>': 4}
    id2vocab = {2: '<s>', 3: '</s>', 1: '<pad>', 0: '<unk>', 4: '<stop>'}
    word_pad = {'<s>': 2, '</s>': 3, '<pad>': 1, '<unk>': 0, '<stop>': 4}
    
    cnt = len(vocab2id)
    with open(file_, 'r') as fp:
        for line in fp:
            arr = re.split(' ', line[:-1])
            if len(arr) == 1:
                arr = re.split('<sec>', line[:-1])
            if arr[0] == ' ':
                continue
            if arr[0] in word_pad:
                continue
            if int(arr[1]) >= mincount:
                vocab2id[arr[0]] = cnt
                id2vocab[cnt] = arr[0]
                cnt += 1
            if len(vocab2id) == max_size:
                break
    
    return vocab2id, id2vocab
'''
Users cannot rewrite this function, unless they want to rewrite the engine.

Split the corpus into batches.
advantage: Don't worry about the memeory.
disadvantage: Takes some time to split the batches.
'''
def create_batch_file(path_data, path_work, is_shuffle, fkey_, file_, batch_size):
    file_name = os.path.join(path_data, file_)
    folder = os.path.join(path_work, 'batch_'+fkey_+'_'+str(batch_size))
    
    try:
        shutil.rmtree(folder)
        os.mkdir(folder)
    except:
        os.mkdir(folder)
    
    corpus_arr = []
    fp = open(file_name, 'r')
    for line in fp:
        corpus_arr.append(line.lower())
    fp.close()
    if is_shuffle:
        random.shuffle(corpus_arr)
        
    cnt = 0
    for itm in corpus_arr:
        try:
            arr.append(itm)
        except:
            arr = [itm]
        if len(arr) == batch_size:
            fout = open(os.path.join(folder, str(cnt)), 'w')
            for sen in arr:
                fout.write(sen)
            fout.close()
            arr = []
            cnt += 1
        
    if len(arr) > 0:
        fout = open(os.path.join(folder, str(cnt)), 'w')
        for sen in arr:
            fout.write(sen)
        fout.close()
        arr = []
        cnt += 1
    
    return cnt
'''
Users cannot rewrite this function, unless they want to rewrite the engine.

This will store data in memeory.
'''
def create_batch_memory(path_, file_, is_shuffle, batch_size):
    
    file_name = os.path.join(path_, file_)
    
    corpus_arr = []
    fp = open(file_name, 'r')
    for line in fp:
        corpus_arr.append(line.lower())
    fp.close()
    if is_shuffle:
        random.shuffle(corpus_arr)
        
    data_split = []
    for itm in corpus_arr:
        try:
            arr.append(itm)
        except:
            arr = [itm]
        if len(arr) == batch_size:
            data_split.append(arr)
            arr = []
        
    if len(arr) > 0:
        data_split.append(arr)
        arr = []
    
    return data_split