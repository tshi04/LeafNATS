'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable

from LeafNATS.playground.multiTaskClassification.modelMTC_base import modelMTCBase
from LeafNATS.data.MultiTaskClassification.process_minibatch_v1 import process_minibatch
from LeafNATS.utils.utils import *

from LeafNATS.modules.encoder.encoder_rnn import EncoderRNN
'''
Application: multi-aspect sentiment classification.
''' 
class modelSSA(modelMTCBase):
    
    def __init__(self, args):
        super(modelSSA, self).__init__(args=args)
            
    def build_models(self):
        '''
        Build models.
        '''
        self.train_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'], self.args.emb_dim
        ).to(self.args.device)
        torch.nn.init.uniform_(self.train_models['embedding'].weight, -1.0, 1.0)
        
        self.train_models['encoder'] = EncoderRNN(
            emb_dim = self.args.emb_dim,
            hidden_size = self.args.rnn_hidden_dim,
            nLayers = self.args.rnn_nLayers,
            rnn_network = self.args.rnn_network,
            device = self.args.device
        ).to(self.args.device)
        
        self.train_models['attn_forward'] = torch.nn.ModuleList(
            [torch.nn.Linear(self.args.rnn_hidden_dim*2, self.args.rnn_hidden_dim, bias=False) 
             for k in range(self.args.n_tasks)]).to(self.args.device)
        self.train_models['attn_wrap'] = torch.nn.ModuleList(
            [torch.nn.Linear(self.args.rnn_hidden_dim, 1, bias=False) 
             for k in range(self.args.n_tasks)]).to(self.args.device)
        
        self.train_models['classifier'] = torch.nn.ModuleList(
            [torch.nn.Linear(2*self.args.rnn_hidden_dim, self.args.n_class)
             for k in range(self.args.n_tasks)]).to(self.args.device)
        
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
        Pipe: Input to logits and attention.
        '''
        review_emb = self.train_models['embedding'](self.batch_data['review'])
        batch_size = review_emb.size(0)
        encoder_hy, hidden_ = self.train_models['encoder'](review_emb)
        logits = []
        out_attn = []
        for k in range(self.args.n_tasks):
            attn = torch.tanh(self.train_models['attn_forward'][k](encoder_hy))
            attn = self.train_models['attn_wrap'][k](attn).squeeze(2)
            attn = torch.softmax(attn, 1)
        
            cv_hidden = torch.tanh(torch.bmm(attn.unsqueeze(1), encoder_hy).squeeze(1))
            logits.append(self.train_models['classifier'][k](cv_hidden))
            out_attn.append(attn)

        out_attn = torch.cat(out_attn, 0)
        out_attn = out_attn.view(self.args.n_tasks, batch_size, -1)
        out_attn = out_attn.transpose(0, 1)
        
        logits = torch.cat(logits, 0)
        logits = logits.view(self.args.n_tasks, batch_size, self.args.n_class)
        logits = logits.transpose(0, 1)
                
        return logits, out_attn
        
    def build_pipelines(self):
        '''
        From pipe to loss.
        '''
        logits, out_attn = self.build_pipe()
        logits = logits.contiguous().view(-1, self.args.n_class)
        
        out_attn = torch.bmm(out_attn, out_attn.transpose(1, 2))
        mask = Variable(torch.eye(self.args.n_tasks)).to(self.args.device)
        mask = mask.unsqueeze(0).repeat(out_attn.size(0), 1, 1)
        out_attn = out_attn*(1-mask)
        loss_cv = torch.mean(out_attn)
                
        loss = self.loss_criterion(logits, self.batch_data['rating'].view(-1))

        return loss + loss_cv
