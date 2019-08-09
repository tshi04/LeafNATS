'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

from LeafNATS.engines.end2end_small import End2EndBase


class modelNLIBase(End2EndBase):
    '''
    Natural Language Inference.
    '''

    def __init__(self, args):
        super().__init__(args=args)

    def build_vocabulary(self):
        '''
        vocabulary
        '''
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
        Initialize Base Model Parameters.
        '''
        for model_name in self.base_models:
            fl_ = os.path.join(self.args.base_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))

    def init_train_model_params(self):
        '''
        Initialize Train Model Parameters.
        For testing and visulization.
        '''
        for model_name in self.train_models:
            fl_ = os.path.join(
                self.args.train_model_dir,
                model_name+'_'+str(self.args.best_model)+'.model')
            self.train_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage))

    def test_worker(self):
        '''
        For the testing.
        '''
        logits = self.build_pipe()
        logits = torch.softmax(logits, dim=1)

        ratePred = logits.topk(1, dim=1)[1].squeeze(1).data.cpu().numpy()
        ratePred -= 1
        ratePred = ratePred.tolist()

        rateTrue = self.batch_data['label'].data.cpu().numpy()
        rateTrue -= 1
        rateTrue = rateTrue.tolist()

        return ratePred, rateTrue

    def run_evaluation(self):
        '''
        For the evaluation.
        '''
        self.pred_data = np.array(self.pred_data)
        self.true_data = np.array(self.true_data)

        accu = accuracy_score(self.true_data, self.pred_data)
        print('Accuracy={}'.format(np.round(accu, 4)))

        return accu
