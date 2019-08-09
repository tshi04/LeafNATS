'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from LeafNATS.modules.encoder.encoder_cnn import EncoderCNN
from LeafNATS.playground.dmsc.model_base import modelDMSCBase


class modelDMSC(modelDMSCBase):

    def __init__(self, args):
        super().__init__(args=args)

    def build_scheduler(self, optimizer):
        '''
        Schedule Learning Rate
        '''
        if self.args.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=self.args.step_size, 
                gamma=self.args.step_decay)

        return scheduler

    def build_models(self):
        '''
        Build all models.
        '''
        self.train_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'], self.args.emb_dim
        ).to(self.args.device)

        kNums = self.args.cnn_kernel_nums.split(',')
        kNums = [int(itm) for itm in kNums]
        ksum = sum(kNums)
        self.train_models['encoder'] = torch.nn.ModuleList(
            [EncoderCNN(self.args.emb_dim, self.args.cnn_kernel_size, self.args.cnn_kernel_nums)
             for k in range(self.args.n_tasks)]).to(self.args.device)

        self.train_models['ff'] = torch.nn.ModuleList(
            [torch.nn.Linear(ksum, ksum)
             for k in range(self.args.n_tasks)]).to(self.args.device)
        self.train_models['classifier'] = torch.nn.ModuleList(
            [torch.nn.Linear(ksum, self.args.n_class)
             for k in range(self.args.n_tasks)]).to(self.args.device)

        self.train_models['drop'] = torch.nn.Dropout(self.args.drop_rate)

        self.loss_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-1).to(self.args.device)

    def build_pipe(self):
        '''
        Shared pipe
        '''
        review_emb = self.train_models['embedding'](self.batch_data['review'])
        batch_size = review_emb.size(0)

        logits = []
        for k in range(self.args.n_tasks):
            review_enc = self.train_models['encoder'][k](review_emb)
            fc = torch.relu(self.train_models['drop'](
                self.train_models['ff'][k](review_enc)))
            logits.append(self.train_models['drop'](
                self.train_models['classifier'][k](fc)))

        logits = torch.cat(logits, 0)
        logits = logits.view(self.args.n_tasks, batch_size, self.args.n_class)
        logits = logits.transpose(0, 1)

        return logits
