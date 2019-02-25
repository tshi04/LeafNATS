'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from LeafNATS.modules.encoder.encoder_rnn import EncoderRNN

class modelRNN(torch.nn.Module):
    def __init__(
        self,
        n_tasks,
        n_class,
        vocab_size,
        emb_dim,
        hidden_dim, # RNN hidden
        rnn_network,
        nLayers,
        device
    ):
        super(modelRNN, self).__init__()        
        self.vocab_size = vocab_size
        self.nLayers = nLayers
        self.hidden_dim = hidden_dim
        self.rnn_network = rnn_network
        self.n_tasks = n_tasks
        self.n_class = n_class
        
        self.embedding = torch.nn.Embedding(
            vocab_size, emb_dim).to(device)
        torch.nn.init.uniform_(self.embedding.weight, -1.0, 1.0)
        
        self.rnn_ = EncoderRNN(
            emb_dim = emb_dim,
            hidden_size = hidden_dim,
            nLayers = nLayers,
            rnn_network = rnn_network,
            device = device
        ).to(device)
        
        self.fc = torch.nn.ModuleList(
            [torch.nn.Linear(2*hidden_dim, n_class)
             for k in range(n_tasks)]).to(device)
        
    def forward(self, review):
        
        emb = self.embedding(review)
        batch_size = emb.size(0)
        
        _, hidden_ = self.rnn_(emb)
        h_t = hidden_
        if self.rnn_network == 'lstm':
            h_t = hidden_[0]
       
        output_ = torch.tanh(torch.cat((h_t[-1], h_t[-2]), 1))
        output_ = [self.fc[k](output_) for k in range(self.n_tasks)]
        output_ = torch.cat(output_, 0).view(self.n_tasks, batch_size, self.n_class)
        output_ = output_.transpose(0, 1)
        
        return output_