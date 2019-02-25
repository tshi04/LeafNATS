'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable

from LeafNATS.playground.multiTaskClassification.modelMTC_base import modelMTCBase
from LeafNATS.data.MultiTaskClassification.process_minibatch_v1 import process_minibatch
from LeafNATS.utils.utils import *
'''
pointer generator network
''' 
class modelMTCBaselines(modelMTCBase):
    
    def __init__(self, args):
        super(modelMTCBaselines, self).__init__(args=args)
            
    def build_models(self):
        '''
        build all models.
        in this model source and target share embeddings
        '''
        if self.args.mtModel == 'mtCNN':
            from LeafNATS.modules.MultiTaskClassification.mt_cnn import modelCNN
            
            self.train_models['model'] = modelCNN(
                n_tasks = self.args.n_tasks,
                n_class = self.args.n_class,
                vocab_size = self.batch_data['vocab_size'],
                emb_dim = self.args.emb_dim,
                kernel_size = self.args.cnn_kernel_size,
                kernel_nums = self.args.cnn_kernel_nums,
                device = self.args.device
            ).to(self.args.device)
            
        if self.args.mtModel == 'mtRNN':
            from LeafNATS.modules.MultiTaskClassification.mt_rnn import modelRNN
            
            self.train_models['model'] = modelRNN(
                n_tasks = self.args.n_tasks,
                n_class = self.args.n_class,
                vocab_size = self.batch_data['vocab_size'],
                emb_dim = self.args.emb_dim,
                hidden_dim = self.args.rnn_hidden_dim,
                rnn_network = self.args.rnn_network,
                nLayers = self.args.rnn_nLayers,
                device = self.args.device
            ).to(self.args.device)
            
        if self.args.mtModel == 'mtRNNATTN':
            from LeafNATS.modules.MultiTaskClassification.mt_rnn_attn import modelRNNATTN
            
            self.train_models['model'] = modelRNNATTN(
                n_tasks = self.args.n_tasks,
                n_class = self.args.n_class,
                vocab_size = self.batch_data['vocab_size'],
                emb_dim = self.args.emb_dim,
                hidden_dim = self.args.rnn_hidden_dim,
                rnn_network = self.args.rnn_network,
                nLayers = self.args.rnn_nLayers,
                device = self.args.device
            ).to(self.args.device)
        
        self.loss_criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
                        
    def build_batch(self, batch_):
        '''
        get batch data
        '''
        review, weight_mask, rating, features = process_minibatch(
            input_=batch_,
            vocab2id=self.batch_data['vocab2id'],
            max_lens=self.args.review_max_lens
        )
        self.batch_data['review'] = review.to(self.args.device)
        self.batch_data['weight_mask'] = weight_mask.to(self.args.device)
        self.batch_data['rating'] = rating.to(self.args.device)
        
    def build_pipe(self):
        '''
        Shared pipe
        '''
        return self.train_models['model'](self.batch_data['review'])
    