'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
import argparse

from LeafNATS.utils.utils import str2bool

parser = argparse.ArgumentParser()
'''
Use in the framework and cannot remove.
'''
parser.add_argument('--debug', type=str2bool, default=False, help='Debug?')
parser.add_argument('--task', default='train', help='train | evaluate | app')

parser.add_argument('--data_dir', default='../snli_data', help='directory that store the data.')
parser.add_argument('--file_train', default='train.json', help='file store training documents.')
parser.add_argument('--file_val', default='dev.json', help='validation (dev) data')
parser.add_argument('--file_test', default='test.json', help='test data')
parser.add_argument('--file_app', default='test.json', help='app data')

parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=100, help='batch size.')
parser.add_argument('--checkpoint', type=int, default=500, help='How often you want to save model?')

parser.add_argument('--continue_training', type=str2bool, default=True, help='Do you want to continue?')
parser.add_argument('--train_base_model', type=str2bool, default=False, help='True: Use Pretrained Param | False: Transfer Learning')
parser.add_argument('--is_lower', type=str2bool, default=False, help='convert all tokens to lower case?')
'''
User specified parameters.
'''
parser.add_argument('--device', default=torch.device("cuda:0"), help='device')
# pretrained word embedding. GloVe
parser.add_argument('--emb_source', default='pretrain', help='scrach | pretrain')
parser.add_argument('--emb_dim', type=int, default=300, help='source embedding dimension')
parser.add_argument('--file_vocab', default='vocab', help='file store vocabulary.')
parser.add_argument('--file_vocab_pos', default='vocab_pos', help='file store vocabulary.')
parser.add_argument('--file_vocab_char', default='vocab_char', help='file store vocabulary.')
# optimization
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0, help='clip the gradient norm.')
# vocabulary
parser.add_argument('--max_vocab_size', type=int, default=30000, 
                    help='max number of words in the vocabulary.')
parser.add_argument('--word_minfreq', type=int, default=5, help='min word frequency')
parser.add_argument('--file_pretrain_vocab', default='vocab_glove_840B_300d', 
                    help='file store pretrain vocabulary.')
parser.add_argument('--file_pretrain_vec', default='glove_840B_300d.npy', 
                    help='file store pretrain vec.')
# RNN coefficient
parser.add_argument('--rnn_network', default='lstm', help='RNN: gru | lstm')
parser.add_argument('--rnn_nLayers', type=int, default=1, help='number of layers')
parser.add_argument('--rnn_hidden_dim', type=int, default=600, help='encoder hidden dimension')
# shared
parser.add_argument('--n_class', type=int, default=3, help='---')
parser.add_argument('--premise_max_lens', type=int, default=50, help='length of premise.')
parser.add_argument('--hypothesis_max_lens', type=int, default=50, help='length of hypothesis.')
# dropout
parser.add_argument('--drop_rate', type=float, default=0.1, help='dropout.')
# scheduler
parser.add_argument('--step_size', type=int, default=2, help='---')
parser.add_argument('--step_decay', type=float, default=0.8, help='---')

args = parser.parse_args()

'''
run model
'''
if args.task == 'train':
    from .model import modelNLI

    model = modelNLI(args)
    model.train()
if args.task == 'evaluate':
    from LeafNATS.eval_scripts.eval_nli import evaluation

    evaluation(args)
