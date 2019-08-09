'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from LeafNATS.playground.nli.model_base_pre_emb import modelNLIBasePreEmb

from LeafNATS.modules.encoder.encoder_rnn import EncoderRNN
from LeafNATS.modules.attention.attention_cross import CrossAttention
from LeafNATS.modules.utils.highway_v1 import HighwayFeedForward

from LeafNATS.data.utils import create_batch_memory
'''
pointer generator network
''' 
class modelNLI(modelNLIBasePreEmb):
    
    def __init__(self, args):
        super().__init__(args=args)
        
    def build_scheduler(self, optimizer):
        '''
        Schedule Learning Rate
        '''
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=self.args.step_size, 
            gamma=self.args.step_decay)
        
        return scheduler
        
    def build_models(self):
        '''
        build all models.
        in this model source and target share embeddings
        '''
        self.base_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'], self.args.emb_dim, 
            padding_idx=0
        ).to(self.args.device)
        
        self.train_models['encoderLI'] = EncoderRNN(
            emb_dim = self.args.emb_dim,
            hidden_size = self.args.rnn_hidden_dim,
            nLayers = self.args.rnn_nLayers,
            rnn_network = self.args.rnn_network,
            device = self.args.device
        ).to(self.args.device)
                
        self.train_models['crossAttn'] = CrossAttention(
        ).to(self.args.device)
        
        self.train_models['wrap'] = torch.nn.Linear(
            self.args.rnn_hidden_dim*8, self.args.emb_dim
        ).to(self.args.device)
        
        self.train_models['encoderIC'] = EncoderRNN(
            emb_dim = self.args.emb_dim,
            hidden_size = self.args.rnn_hidden_dim,
            nLayers = self.args.rnn_nLayers,
            rnn_network = self.args.rnn_network,
            device = self.args.device
        ).to(self.args.device)
                                   
        self.train_models['ff1'] = torch.nn.Linear(
            self.args.rnn_hidden_dim*8, self.args.rnn_hidden_dim*4
        ).to(self.args.device)
        self.train_models['ff2'] = torch.nn.Linear(
            self.args.rnn_hidden_dim*4, self.args.rnn_hidden_dim*4
        ).to(self.args.device)
        self.train_models['classifier'] = torch.nn.Linear(
            self.args.rnn_hidden_dim*4, self.args.n_class
        ).to(self.args.device)
        
        self.train_models['drop'] = torch.nn.Dropout(self.args.drop_rate)
        
        self.loss_criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        
    def build_pipe(self):
        '''
        Shared pipe
        '''
        batch_size = self.batch_data['prem'].size(0)
        
        prem_emb = self.base_models['embedding'](self.batch_data['prem'])
        hypo_emb = self.base_models['embedding'](self.batch_data['hypo'])
        
        prem_enc, _ = self.train_models['encoderLI'](prem_emb)
        hypo_enc, _ = self.train_models['encoderLI'](hypo_emb)

        hypo_cs, prem_cs = self.train_models['crossAttn'](
            prem_enc, hypo_enc, self.batch_data['prem_mask'], self.batch_data['hypo_mask'])
        
        prem_feat = torch.relu(self.train_models['drop'](self.train_models['wrap'](
            torch.cat((prem_enc, prem_enc+prem_cs, prem_enc-prem_cs, prem_enc*prem_cs), 2))))
        hypo_feat = torch.relu(self.train_models['drop'](self.train_models['wrap'](
            torch.cat((hypo_enc, hypo_enc+hypo_cs, hypo_enc-hypo_cs, hypo_enc*hypo_cs), 2))))
        
        prem_enc, _ = self.train_models['encoderIC'](prem_feat)
        hypo_enc, _ = self.train_models['encoderIC'](hypo_feat)
                
        prem_maxp = F.max_pool1d(prem_enc.transpose(1, 2), prem_enc.size(1)).squeeze(2)
        hypo_maxp = F.max_pool1d(hypo_enc.transpose(1, 2), hypo_enc.size(1)).squeeze(2)
        
        prem_avgp = F.avg_pool1d(prem_enc.transpose(1, 2), prem_enc.size(1)).squeeze(2)
        hypo_avgp = F.avg_pool1d(hypo_enc.transpose(1, 2), hypo_enc.size(1)).squeeze(2)
        
        enc_cat = torch.cat([prem_maxp, hypo_maxp, prem_avgp, hypo_avgp], 1)
                        
        fc = torch.relu(self.train_models['drop'](self.train_models['ff1'](enc_cat)))
        fc = torch.relu(self.train_models['drop'](self.train_models['ff2'](fc)))

        logits = self.train_models['classifier'](fc)
                
        return logits