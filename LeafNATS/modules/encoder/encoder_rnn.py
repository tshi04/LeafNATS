'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable

class EncoderRNN(torch.nn.Module):
    
    def __init__(
        self,
        emb_dim, # input_dim
        hidden_size,
        nLayers,
        rnn_network,
        device = torch.device("cpu")
    ):
        '''
        RNN encoder
        '''
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_network = rnn_network
        self.nLayers = nLayers
        self.device = device
        
        if rnn_network == 'lstm':
            self.encoder = torch.nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=nLayers,
                batch_first=True,
                bidirectional=True).to(device)
        elif rnn_network == 'gru':
            self.encoder = torch.nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_size,
                num_layers=nLayers,
                batch_first=True,
                bidirectional=True).to(device)

    def forward(self, input_):
        '''
        get encoding
        '''
        batch_size = input_.size(0)
        
        h0_encoder = Variable(torch.zeros(2*self.nLayers, batch_size, self.hidden_size)).to(self.device)
        if self.rnn_network == 'lstm':
            c0_encoder = Variable(torch.zeros(2*self.nLayers, batch_size, self.hidden_size)).to(self.device)
            # encoding
            hy_encoder, (ht_encoder, ct_encoder) = self.encoder(input_, (h0_encoder, c0_encoder))

            return hy_encoder, (ht_encoder, ct_encoder)
            
        elif self.rnn_network == 'gru':
            # encoding
            hy_encoder, ht_encoder = self.encoder(input_, h0_encoder)
                        
            return hy_encoder, ht_encoder
