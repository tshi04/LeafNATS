'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math
import torch
import torch.nn.functional as F

class CrossAttention(torch.nn.Module):
    '''
    Implement of multi-head attention.
    '''
    def __init__(
        self,
        input_size,
        hidden_size
    ):
        super().__init__()
        
        self.proj = torch.nn.Linear(input_size, hidden_size)
        
    def forward(self, inputA, inputB, maskA=None, maskB=None):
        '''
        Input: embedding.
        '''
        batch_size = inputA.size(0)
        
        inputA = torch.relu(self.proj(inputA))
        inputB = torch.relu(self.proj(inputB))
        
        scores = torch.bmm(inputA, inputB.transpose(1, 2))
        if maskA is not None:
            maskA = maskA[:, :, None]
            maskB = maskB[:, None, :]
            mask = torch.bmm(maskA, maskB)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attnA = torch.softmax(scores, 1)
        attnB = torch.softmax(scores, 2)
        
        cvA = torch.bmm(attnA.transpose(1, 2), inputA)
        cvB = torch.bmm(attnB, inputB)

        return cvA, cvB
