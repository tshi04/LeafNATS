'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
from torch.autograd import Variable


class AttentionTopic(torch.nn.Module):

    def __init__(self, input_size, n_topics,
                 device=torch.device("cpu")):
        '''
        implementation of self-attention.
        '''
        super().__init__()
        self.n_topics = n_topics

        self.ff = torch.nn.ModuleList([
            torch.nn.Linear(input_size, 1, bias=False)
            for k in range(n_topics)
        ]).to(device)

    def forward(self, input_, mask=None):
        '''
        input vector: input_
        output:
            attn_weights: attention weights
            attn_ctx_vec: context vector
        '''
        batch_size = input_.size(0)

        attn_weight = []
        attn_ctx_vec = []
        for k in range(self.n_topics):
            attn_ = self.ff[k](input_).squeeze(2)
            if mask is not None:
                attn_ = attn_.masked_fill(mask == 0, -1e9)
            attn_ = torch.softmax(attn_, dim=1)
            ctx_vec = torch.bmm(attn_.unsqueeze(1), input_).squeeze(1)
            attn_weight.append(attn_)
            attn_ctx_vec.append(ctx_vec)

        attn_weight = torch.cat(attn_weight, 0).view(
            self.n_topics, batch_size, -1)
        attn_weight = attn_weight.transpose(0, 1)
        attn_ctx_vec = torch.cat(attn_ctx_vec, 0).view(
            self.n_topics, batch_size, -1)
        attn_ctx_vec = attn_ctx_vec.transpose(0, 1)

        return attn_weight, attn_ctx_vec
