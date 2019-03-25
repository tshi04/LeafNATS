'''
@author Tian Shi
Please contact tshi@vt.edu

https://github.com/codertimo/BERT-pytorch.git
https://github.com/namisan/mt-dnn
https://github.com/dhlee347/pytorchic-bert.git
'''
import torch

class WordEmbedding(torch.nn.Module):
    
    def __init__(
        self,
        vocab_size, # vocab size
        emb_size, # embedding dimension
        share_emb_weight
    ):
        '''
        embedding and decoding.
        '''
        super(natsEmbedding, self).__init__()
        
        # in LeafNATS padding index is 1.
        self.embedding = torch.nn.Embedding(
            vocab_size, emb_dim, padding_idx=1)
        
    def forward(self, input_):
        
        return self.embedding(input_)
    
    