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
parser.add_argument('--task', default='train',
                    help='train | validate | rouge | beam | app')

parser.add_argument('--data_dir', default='../mt_data_spacy/',
                    help='directory that store the data.')
parser.add_argument('--file_corpus', default='train.txt',
                    help='file store training documents.')
parser.add_argument('--file_val', default='val.txt', help='val data')

parser.add_argument('--n_epoch', type=int, default=35,
                    help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=16, help='batch size.')
parser.add_argument('--checkpoint', type=int, default=100,
                    help='How often you want to save model?')
parser.add_argument('--val_num_batch', type=int,
                    default=30, help='how many batches')
parser.add_argument('--nbestmodel', type=int, default=10,
                    help='How many models you want to keep?')

parser.add_argument('--continue_training', type=str2bool,
                    default=True, help='Do you want to continue?')
parser.add_argument('--train_base_model', type=str2bool, default=False,
                    help='True: Use Pretrained Param | False: Transfer Learning')
parser.add_argument('--use_move_avg', type=str2bool,
                    default=False, help='move average')
parser.add_argument('--use_optimal_model', type=str2bool,
                    default=True, help='Do you want to use the best model?')
parser.add_argument('--model_optimal_key', default='0,0', help='epoch,batch')
parser.add_argument('--is_lower', type=str2bool, default=True,
                    help='convert all tokens to lower case?')
'''
User specified parameters.
'''
parser.add_argument('--device', default=torch.device("cuda:0"), help='device')
parser.add_argument('--file_vocab', default='vocab',
                    help='file store training vocabulary.')

parser.add_argument('--max_vocab_size', type=int, default=50000,
                    help='max number of words in the vocabulary.')
parser.add_argument('--word_minfreq', type=int,
                    default=5, help='min word frequency')

parser.add_argument('--emb_dim', type=int, default=128,
                    help='source embedding dimension')
parser.add_argument('--src_hidden_dim', type=int,
                    default=256, help='encoder hidden dimension')
parser.add_argument('--trg_hidden_dim', type=int,
                    default=256, help='decoder hidden dimension')
parser.add_argument('--src_seq_lens', type=int, default=400,
                    help='length of source documents.')
parser.add_argument('--sum_seq_lens', type=int, default=50,
                    help='length of target documents.')
parser.add_argument('--ttl_seq_lens', type=int, default=20,
                    help='length of target documents.')

parser.add_argument('--rnn_network', default='lstm', help='gru | lstm')
parser.add_argument('--attn_method', default='luong_concat',
                    help='luong_concat | luong_general')
parser.add_argument('--repetition', default='vanilla',
                    help='vanilla | temporal. Repetition Handling')
parser.add_argument('--pointer_net', type=str2bool,
                    default=True, help='Use pointer network?')
parser.add_argument('--oov_explicit', type=str2bool,
                    default=True, help='explicit OOV?')
parser.add_argument('--attn_decoder', type=str2bool,
                    default=True, help='attention decoder?')
parser.add_argument('--share_emb_weight', type=str2bool,
                    default=True, help='share_emb_weight')

parser.add_argument('--learning_rate', type=float,
                    default=0.0001, help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0,
                    help='clip the gradient norm.')
# for beam search
parser.add_argument('--file_test', default='test.txt', help='test data')
parser.add_argument('--beam_size', type=int, default=5, help='beam size.')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='batch size for beam search.')
parser.add_argument('--copy_words', type=str2bool,
                    default=True, help='Do you want to copy words?')
parser.add_argument('--task_key', default='summary', help='summary | title')
# for evaluation
parser.add_argument('--file_output', default='summary.txt',
                    help='test output file')
# for app
parser.add_argument('--app_model_dir', default='../model/',
                    help='directory that stores models.')
parser.add_argument('--app_data_dir', default='/var/www/html/leafNATS/tmp',
                    help='directory that stores data.')

args = parser.parse_args()

if args.repetition == 'asee' and args.task == 'train':
    args.repetition = 'asee_train'
if not args.pointer_net:
    args.repetition = 'vanilla'
    args.oov_explicit = False

if args.task == 'train' or args.task == 'validate' or args.task == 'beam':
    from .model import modelMultiTask

    model = modelMultiTask(args)
    if args.task == "train":
        model.train()
    if args.task == "validate":
        model.validate()
    if args.task == "beam":
        model.test()

if args.task == "rouge":
    from LeafNATS.eval_scripts.eval_pyrouge import run_pyrouge

    run_pyrouge(args)

if args.task == "app":
    from .model_app import modelMultiTaskApp

    model = modelMultiTaskApp(args)
    model.app2Go()
