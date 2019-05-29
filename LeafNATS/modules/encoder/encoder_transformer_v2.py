'''
@author Tian Shi
Please contact tshi@vt.edu

https://github.com/codertimo/BERT-pytorch.git
https://github.com/namisan/mt-dnn
https://github.com/dhlee347/pytorchic-bert.git
'''
import math
import torch

from LeafNATS.modules.attention.attention_multi_head_v2 import MultiHeadedAttention
from LeafNATS.modules.utils.LayerNormalization import LayerNormalization
from LeafNATS.modules.utils.PositionwiseFeedForward import PositionwiseFeedForward

class TransformerBlock(torch.nn.Module):
    '''
    Implementation of Transformer
    '''
    def __init__(
        self, 
        input_size,
        hidden_size,
        n_heads, 
        drop_rate
    ):
        super(TransformerBlock, self).__init__()
        # multi-head attention
        self.attentionMH = MultiHeadedAttention(n_heads, input_size, hidden_size, drop_rate)
        self.input_proj = torch.nn.Linear(input_size, hidden_size, bias=False)
        # layer normalization
        self.norm1 = LayerNormalization(hidden_size)
        self.norm2 = LayerNormalization(hidden_size)
        # layer feed-forward
        self.layer_ff = PositionwiseFeedForward(hidden_size, hidden_size*4, hidden_size, drop_rate)
        
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, input_, mask=None):
        '''
        Transformer
        '''
        hd = self.attentionMH(input_, mask)
        hd = self.norm1(self.input_proj(input_) + self.drop(hd))
        hd = self.norm2(hd + self.layer_ff(hd))

        return self.drop(hd)

    
    
    