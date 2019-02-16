'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable

class AttentionSelf(torch.nn.Module):
    
    def __init__(
        self,
        input_size
        hidden_size # hidden dimension
    ):
        '''
        implementation of self-attention.
        '''
        super(AttentionSelf, self).__init__()
        
        self.attn_in = torch.nn.Linear(
            input_size, hidden_size)
        self.attn_warp = torch.nn.Linear(
            hidden_size, 1, bias=False)

    def forward(self, input_):
        '''
        input vector: input_
        output:
            attn_: attention weights
            cv: context vector
        '''
        attn_ = F.tanh(self.attn_in(input_))
        attn_ = self.attn_warp(attn_).squeeze(2)
        attn_ = torch.softmax(attn_, dim=1)
        cv = torch.bmm(attn_.unsqueeze(1), input_).squeeze(1)
        
        return attn_, cv
