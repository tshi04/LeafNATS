'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torch.autograd import Variable

from LeafNATS.engines.end2end_mtclass import End2EndBase
from LeafNATS.data.utils import construct_vocab
from LeafNATS.data.utils import load_vocab_pretrain
from LeafNATS.utils.utils import *
'''
pointer generator network
''' 
class modelBase(End2EndBase):
    
    def __init__(self, args):
        super(modelBase, self).__init__(args=args)
        
    def build_vocabulary(self):
        '''
        vocabulary
        '''
        if self.args.emb_source == 'pretrain':
            vocab2id, id2vocab, pretrain_vec = load_vocab_pretrain(
                os.path.join(self.args.data_dir, self.args.file_pretrain_vocab),
                os.path.join(self.args.data_dir, self.args.file_pretrain_vec))
            vocab_size = len(vocab2id)
            self.batch_data['vocab2id'] = vocab2id
            self.batch_data['id2vocab'] = id2vocab
            self.batch_data['pretrain_emb'] = pretrain_vec
            self.batch_data['vocab_size'] = vocab_size
            print('The vocabulary size: {}'.format(vocab_size))
        elif self.args.emb_source == 'scrach':
            vocab2id, id2vocab = construct_vocab(
                file_=os.path.join(self.args.data_dir, self.args.file_vocab),
                max_size=self.args.max_vocab_size,
                mincount=self.args.word_minfreq)
            vocab_size = len(vocab2id)
            self.batch_data['vocab2id'] = vocab2id
            self.batch_data['id2vocab'] = id2vocab
            self.batch_data['vocab_size'] = vocab_size
            print('The vocabulary size: {}'.format(vocab_size))
                    
    def build_optimizer(self, params):
        '''
        init model optimizer
        '''
        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)
                
        return optimizer
    
    def init_base_model_params(self):
        '''
        Initialize Model Parameters
        '''
        for model_name in self.base_models:
            fl_ = os.path.join(self.args.base_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))
    
    def init_train_model_params(self):
        '''
        Initialize Model Parameters
        '''
        for model_name in self.train_models:
            fl_ = os.path.join(
                self.args.train_model_dir, 
                model_name+'_'+str(self.args.best_model)+'.model')
            self.train_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))
        
    def build_pipe(self):
        '''
        Shared pipe
        '''
        raise NotImplementedError

    def build_pipelines(self):
        '''
        here we have all data flow from the input to output
        '''
        logits = self.build_pipe()        
        loss = self.loss_criterion(logits, self.batch_data['label'].view(-1))

        return loss
            
    def test_worker(self):
        '''
        For the testing.
        '''
        raise NotImplementedError
        
    def run_evaluation(self):
        '''
        For the evaluation.
        '''
        self.pred_data = np.array(self.pred_data)
        self.true_data = np.array(self.true_data)

        accu = accuracy_score(self.true_data, self.pred_data)
        print('Accuracy={}'.format(np.round(accu, 4)))
        