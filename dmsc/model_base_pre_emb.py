'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.data.utils import load_vocab_pretrain

from .model_base import modelDMSCBase


class modelDMSCBasePreEmb(modelDMSCBase):
    '''
    Document Level Multi-Aspect Sentiment Classification (DMSC)
    Load pre-trained word embeddings.
    Rewrite vocabulary and base model parameters modules.
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_vocabulary(self):
        '''
        vocabulary
        '''
        vocab2id, id2vocab, pretrain_vec = load_vocab_pretrain(
            os.path.join(self.args.data_dir, self.args.file_pretrain_vocab),
            os.path.join(self.args.data_dir, self.args.file_pretrain_vec))
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['pretrain_emb'] = pretrain_vec
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

    def init_base_model_params(self):
        '''
        Initialize Base Model Parameters.
        '''
        emb_para = torch.FloatTensor(
            self.batch_data['pretrain_emb']).to(self.args.device)
        self.base_models['embedding'].weight = torch.nn.Parameter(emb_para)

        for model_name in self.base_models:
            if model_name == 'embedding':
                continue
            fl_ = os.path.join(self.args.base_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))
