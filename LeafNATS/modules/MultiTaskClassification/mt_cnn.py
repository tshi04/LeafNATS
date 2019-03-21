'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re
import torch
from torch.autograd import Variable

from LeafNATS.modules.encoder.encoder_cnn import EncoderCNN

class modelCNN(torch.nn.Module):
    def __init__(
        self, 
        n_tasks,
        n_class,
        vocab_size,
        emb_dim,
        kernel_size, # 3,4,5
        kernel_nums,  # 100,200,100
        device
    ):
        '''
        This is an implementation of 
        CNN based multi-aspect sentiment classification baseline.
        '''
        super(modelCNN, self).__init__()        
        self.vocab_size = vocab_size
        self.device = device
        self.n_tasks = n_tasks
        self.n_class = n_class
        
        self.embedding = torch.nn.Embedding(
            vocab_size, emb_dim).to(device)
        torch.nn.init.uniform_(self.embedding.weight, -1.0, 1.0)
        
        self.mt_conv = []
        for k in range(n_tasks):
            conv = EncoderCNN(
                emb_dim = emb_dim,
                kernel_size = kernel_size,
                kernel_nums = kernel_nums)
            self.mt_conv.append(conv)
        self.mt_conv = torch.nn.ModuleList(self.mt_conv).to(device)
        
        kNums = re.split(',', kernel_nums)
        kNums = [int(itm) for itm in kNums]
        ksum = sum(kNums)
        self.fc = torch.nn.ModuleList(
            [torch.nn.Linear(ksum, n_class)
             for k in range(n_tasks)]).to(device)
        
    def forward(self, review):
        '''
        input:
            review
        output:
            features
        '''
        emb = self.embedding(review)
        batch_size = emb.shape[0]
        
        input_ = emb.unsqueeze(1)
        output_ = []
        for j in range(self.n_tasks):
            zz = self.mt_conv[j](input_)
            zz = self.fc[j](zz)
            output_.append(zz)
        output_ = torch.cat(output_, 0).view(self.n_tasks, batch_size, self.n_class)
        output_ = output_.transpose(0, 1)
        
        return output_