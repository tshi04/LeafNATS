'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time
import numpy as np

import torch
from torch.autograd import Variable

from .model_base import modelNLIBase

from LeafNATS.data.utils import load_vocab_pretrain
from LeafNATS.data.utils import construct_pos_vocab
from LeafNATS.data.utils import construct_char_vocab
from LeafNATS.data.nli.process_minibatch_v2 import process_minibatch

class modelNLIBasePreEmb(modelNLIBase):
    '''
    Natural Language Inference.
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

        vocab2id_pos, id2vocab_pos = construct_pos_vocab(
            file_=os.path.join(self.args.data_dir, self.args.file_vocab_pos))
        vocab_size_pos = len(vocab2id_pos)
        self.batch_data['pos_vocab2id'] = vocab2id_pos
        self.batch_data['pos_id2vocab'] = id2vocab_pos
        self.batch_data['pos_vocab_size'] = vocab_size_pos
        print('The vocabulary (pos) size: {}'.format(vocab_size_pos))
        
        vocab2id_char, id2vocab_char = construct_char_vocab(
            file_=os.path.join(self.args.data_dir, self.args.file_vocab_char))
        vocab_size_char = len(vocab2id_char)
        self.batch_data['char_vocab2id'] = vocab2id_char
        self.batch_data['char_id2vocab'] = id2vocab_char
        self.batch_data['char_vocab_size'] = vocab_size_char
        print('The vocabulary (char) size: {}'.format(vocab_size_char))
        
    def build_batch(self, batch_):
        '''
        get batch data
        '''
        prem_var, hypo_var, prem_char_var, hypo_char_var, \
        prem_pos_var, hypo_pos_var, prem_mask, hypo_mask, label_var = process_minibatch(
            input_=batch_,
            vocab2id=self.batch_data['vocab2id'],
            vocab2id_char=self.batch_data['char_vocab2id'],
            vocab2id_pos=self.batch_data['pos_vocab2id'],
            premise_max_lens=self.args.premise_max_lens,
            hypothesis_max_lens=self.args.hypothesis_max_lens
        )
        self.batch_data['prem'] = prem_var.to(self.args.device)
        self.batch_data['hypo'] = hypo_var.to(self.args.device)    
        self.batch_data['prem_char'] = prem_char_var.to(self.args.device)
        self.batch_data['hypo_char'] = hypo_char_var.to(self.args.device) 
        self.batch_data['prem_pos'] = prem_pos_var.to(self.args.device)
        self.batch_data['hypo_pos'] = hypo_pos_var.to(self.args.device) 
        self.batch_data['prem_mask'] = prem_mask.to(self.args.device)
        self.batch_data['hypo_mask'] = hypo_mask.to(self.args.device)
        self.batch_data['label'] = label_var.to(self.args.device)
        
    def init_base_model_params(self):
        '''
        Initialize Base Model Parameters.
        '''
        emb_para = torch.FloatTensor(self.batch_data['pretrain_emb']).to(self.args.device)
        self.base_models['embedding'].weight = torch.nn.Parameter(emb_para)
        
        for model_name in self.base_models:
            if model_name == 'embedding':
                continue
            fl_ = os.path.join(self.args.base_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))
            
    def build_pipe(self):
        '''
        Pipes shared by training/validation/testing
        '''
        raise NotImplementedError
        
    def build_pipelines(self):
        '''
        Data flow from input to output.
        '''
        logits = self.build_pipe()        
        loss = self.loss_criterion(logits, self.batch_data['label'].view(-1))

        return loss