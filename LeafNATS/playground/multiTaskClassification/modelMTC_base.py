'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import torch
from torch.autograd import Variable

from LeafNATS.engines.end2end_mtclass import End2EndBase
from LeafNATS.data.utils import construct_vocab
from LeafNATS.utils.utils import *
'''
pointer generator network
''' 
class modelMTCBase(End2EndBase):
    
    def __init__(self, args):
        super(modelMTCBase, self).__init__(args=args)
        
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
        logits = logits.contiguous().view(-1, self.args.n_class)
                
        loss = self.loss_criterion(logits, self.batch_data['rating'].view(-1))

        return loss
            
    def test_worker(self):
        '''
        For the testing.
        '''
        logits = self.build_pipe()
        logits = torch.softmax(logits, dim=2)
        
        ratePred = logits.topk(1, dim=2)[1].squeeze(2).data.cpu().numpy()
        ratePred += 1
        ratePred = ratePred.tolist()
        
        rateTrue = self.batch_data['rating'].data.cpu().numpy()
        rateTrue += 1
        rateTrue = rateTrue.tolist()
        
        return ratePred, rateTrue
        
    def run_evaluation(self):
        '''
        For the evaluation.
        '''
        self.pred_data = np.array(self.pred_data)
        self.true_data = np.array(self.true_data)

        avgf1 = []
        avgaccu = []
        avgmse = []
        for k in range(self.args.n_tasks):
            (p1, r1, f1, _) = precision_recall_fscore_support(self.true_data[:, k], self.pred_data[:, k], average='macro')
            accu = accuracy_score(self.true_data[:, k], self.pred_data[:, k])
            mse = mean_squared_error(self.true_data[:, k], self.pred_data[:, k])
            avgf1.append(f1)
            avgaccu.append(accu)
            avgmse.append(mse)
            print('f_score={}, Accuracy={}, MSE={}'.format(
                np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
        avgf1 = np.average(np.array(avgf1))
        avgaccu = np.average(np.array(avgaccu))
        avgmse = np.average(np.array(avgmse))
        print('Average f_score={}, accuracy={}, MSE={}'.format(
            np.round(avgf1, 4), np.round(avgaccu, 4), np.round(avgmse, 4)
        ))
        