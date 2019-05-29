'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import sys
import numpy as np
import torch
from torch.autograd import Variable
'''
Display progress
'''
def show_progress(curr_, total_, time=""):
    prog_ = int(round(100.0*float(curr_)/float(total_)))
    dstr = '[' + '>'*int(round(prog_/4)) + ' '*(25-int(round(prog_/4))) + ']'
    sys.stdout.write(dstr + str(prog_) + '%' + time +'\r')
    sys.stdout.flush()
'''
Argparser
'''
def str2bool(input_):
    if input_.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input_.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
'''
This function is used in beam search
'''
def tensor_transformer(seq0, batch_size, beam_size):
    seq = seq0.unsqueeze(2)
    seq = seq.repeat(1, 1, beam_size, 1)
    seq = seq.contiguous().view(batch_size, beam_size*beam_size, seq.size(3))
    return seq
'''
evaluate accuracy
'''
def eval_accuracy(preds, golds):
    
    nm = len(preds)
    
    preds = np.array(preds)
    golds = np.array(golds)
    
    diff = preds - golds
    diff = diff * diff
    
    accu = preds - golds
    accu[accu != 0] = 1.0
    accu = 1.0 - accu
    
    return np.sum(accu)/nm, np.sum(diff)/nm
