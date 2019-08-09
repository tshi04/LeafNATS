'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.data.dmsc.process_minibatch_v2 import process_minibatch
from LeafNATS.data.utils import construct_vocab
from LeafNATS.engines.end2end_small import End2EndBase
from LeafNATS.eval_scripts.utils import eval_dmsc


class modelDMSCBase(End2EndBase):
    '''
    Document Level Multi-Aspect Sentiment Classification (DMSC)
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

    def build_batch(self, batch_):
        '''
        get batch data
        '''
        review, weight_mask, rating = process_minibatch(
            input_=batch_,
            vocab2id=self.batch_data['vocab2id'],
            max_lens=self.args.review_max_lens
        )
        self.batch_data['review'] = review.to(self.args.device)
        self.batch_data['weight_mask'] = weight_mask.to(self.args.device)
        self.batch_data['rating'] = rating.to(self.args.device)

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
        logits = logits.contiguous().view(-1, self.args.n_class)

        loss = self.loss_criterion(logits, self.batch_data['rating'].view(-1))

        return loss

    def test_worker(self):
        '''
        Testing.
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
        For evaluation.
        '''
        self.pred_data = np.array(self.pred_data)
        self.true_data = np.array(self.true_data)

        label_pred = []
        label_true = []
        for k in range(self.args.n_tasks):
            predlb = [rt for idx, rt in enumerate(
                self.pred_data[:, k].tolist()) if self.true_data[idx, k] != 0]
            truelb = [rt for idx, rt in enumerate(
                self.true_data[:, k].tolist()) if self.true_data[idx, k] != 0]
            label_pred += predlb
            label_true += truelb

        accu, mse = eval_dmsc(label_pred, label_true)
        accu = np.round(accu, 4)
        mse = np.round(mse, 4)

        print('Accuracy={}, MSE={}'.format(accu, mse))
        return accu
