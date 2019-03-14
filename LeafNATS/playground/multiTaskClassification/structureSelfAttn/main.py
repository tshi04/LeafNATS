'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
import argparse

from LeafNATS.utils.utils import str2bool
from LeafNATS.playground.multiTaskClassification.structureSelfAttn.model import modelSSA
from LeafNATS.eval_scripts.eval_MultiTaskClassification import evaluation

parser = argparse.ArgumentParser()
'''
Use in the framework and cannot remove.
'''
parser.add_argument('--debug', type=str2bool, default=False, help='Debug?')
parser.add_argument('--task', default='train', help='train | evaluate')

parser.add_argument('--data_dir', default='../data', help='directory that store the data.')
parser.add_argument('--file_vocab', default='vocab.csv', help='file store training vocabulary.')
parser.add_argument('--file_train', default='train.csv', help='file store training documents.')
parser.add_argument('--file_val', default='valid.csv', help='validation (dev) data')
parser.add_argument('--file_test', default='test.csv', help='test data')

parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=20, help='batch size.')
parser.add_argument('--checkpoint', type=int, default=500, help='How often you want to save model?')

parser.add_argument('--continue_training', type=str2bool, default=True, help='Do you want to continue?')
parser.add_argument('--train_base_model', type=str2bool, default=False, help='True: Use Pretrained Param | False: Transfer Learning')
'''
User specified parameters.
'''
parser.add_argument('--device', default=torch.device("cuda:0"), help='device')
# optimization
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0, help='clip the gradient norm.')
# vocabulary
parser.add_argument('--max_vocab_size', type=int, default=50000, help='max number of words in the vocabulary.')
parser.add_argument('--word_minfreq', type=int, default=5, help='min word frequency')
# shared
parser.add_argument('--n_tasks', type=int, default=4, help='---')
parser.add_argument('--n_class', type=int, default=5, help='---')
parser.add_argument('--review_max_lens', type=int, default=250, help='length of documents.')
parser.add_argument('--emb_dim', type=int, default=128, help='source embedding dimension')
# RNN coefficient
parser.add_argument('--rnn_network', default='gru', help='RNN: gru | lstm')
parser.add_argument('--rnn_nLayers', type=int, default=1, help='number of layers')
parser.add_argument('--rnn_hidden_dim', type=int, default=128, help='encoder hidden dimension')

args = parser.parse_args()

'''
run model
'''
if args.task == 'evaluate':
    evaluation(args)
else:
    model = modelSSA(args)
    if args.task == 'train':
        model.train()
