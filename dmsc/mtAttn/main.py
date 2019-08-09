'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import argparse

import torch

from LeafNATS.utils.utils import str2bool

parser = argparse.ArgumentParser()
'''
Use in the framework and cannot remove.
'''
parser.add_argument('--debug', type=str2bool, default=False, help='Debug?')
parser.add_argument('--task', default='train', help='train | evaluate')

parser.add_argument(
    '--data_dir', default='../data/trip_binary', help='data dir')
parser.add_argument('--file_vocab', default='vocab',
                    help='file store vocabulary.')
parser.add_argument('--file_train', default='train', help='train data.')
parser.add_argument('--file_val', default='dev', help='dev data')
parser.add_argument('--file_test', default='test', help='test data')

parser.add_argument('--n_epoch', type=int, default=10,
                    help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=20, help='batch size.')
parser.add_argument('--checkpoint', type=int, default=500,
                    help='How often you want to save model?')

parser.add_argument('--continue_training', type=str2bool,
                    default=True, help='Do you want to continue?')
parser.add_argument('--train_base_model', type=str2bool, default=False,
                    help='True: Use Pretrained Param | False: Transfer Learning')
parser.add_argument('--is_lower', type=str2bool,
                    default=True, help='lower case for all words?')
'''
User specified parameters.
'''
parser.add_argument('--device', default=torch.device("cuda:0"), help='device')
# optimization
parser.add_argument('--learning_rate', type=float,
                    default=0.0005, help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0,
                    help='clip the gradient norm.')
# vocabulary
parser.add_argument('--max_vocab_size', type=int,
                    default=50000, help='vocabulary size clip')
parser.add_argument('--word_minfreq', type=int, default=5,
                    help='tokens min frequency')
# shared
parser.add_argument('--n_tasks', type=int, default=7, help='number of aspects')
parser.add_argument('--n_class', type=int, default=2, help='number of clsses')
parser.add_argument('--review_max_lens', type=int,
                    default=400, help='max length documents.')
parser.add_argument('--emb_dim', type=int, default=300,
                    help='embedding dimension')
# RNN coefficient
parser.add_argument('--rnn_network', default='lstm', help='RNN: gru | lstm')
parser.add_argument('--rnn_nLayers', type=int,
                    default=4, help='number of layers')
parser.add_argument('--rnn_hidden_dim', type=int,
                    default=600, help='encoder hidden dimension')
# dropout
parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout.')
# scheduler
parser.add_argument('--lr_schedule', type=str2bool,
                    default=True, help='Schedule learning rate.')
parser.add_argument('--step_size', type=int, default=2, help='---')
parser.add_argument('--step_decay', type=float, default=0.8, help='---')

args = parser.parse_args()

'''
run model
'''
if args.task == 'evaluate':
    from LeafNATS.eval_scripts.eval_dmsc import evaluation
    evaluation(args)
else:
    from .model import modelDMSC
    model = modelDMSC(args)
    if args.task == 'train':
        model.train()
