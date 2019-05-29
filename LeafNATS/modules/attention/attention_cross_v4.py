'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import math
import torch
import torch.nn.functional as F

class CrossAttention(torch.nn.Module):
    '''
    Implement of Co-attention.
    '''
    def __init__(
        self,
        inputA_size,
        inputB_size
    ):
        super().__init__()
        
        self.projA = torch.nn.Linear(inputA_size, inputB_size)
        
    def forward(self, inputA, inputB, maskA=None, maskB=None):
        '''
        Input: embedding.
        '''
        batch_size = inputA.size(0)
        
        projA = self.projA(inputA)
        
        scores = torch.bmm(projA, inputB.transpose(1, 2))
        if maskA is not None and maskB is not None:
            maskA = maskA[:, :, None]
            maskB = maskB[:, None, :]
            mask = torch.bmm(maskA, maskB)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attnA = torch.softmax(scores, 1)
        attnB = torch.softmax(scores, 2)
        
        cvA = torch.bmm(attnA.transpose(1, 2), inputA)
        cvB = torch.bmm(attnB, inputB)

        return cvA, cvB
