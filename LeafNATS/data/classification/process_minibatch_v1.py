'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re

import torch
from torch.autograd import Variable


def process_minibatch(input_, vocab2id, max_lens):
    '''
    Process the minibatch for beeradvocate and tripadvisor datasets
    The data format
    [rating]\t\t\treview
    '''
    len_review = []
    review_arr = []
    rating_arr = []
    for line in input_:
        arr = re.split('\t\t\t', line[:-1])
        rating_arr.append(int(arr[0]))

        review = re.split(r'\s', arr[-1])
        review = list(filter(None, review))
        len_review.append(len(review))

        review2id = [vocab2id[wd] if wd in vocab2id else vocab2id['<unk>']
                     for wd in review]
        review_arr.append(review2id)

    review_lens = min(max_lens, max(len_review))

    review_arr = [itm[:review_lens] for itm in review_arr]
    review_arr = [itm + [vocab2id['<pad>']] *
                  (review_lens-len(itm)) for itm in review_arr]

    review_var = Variable(torch.LongTensor(review_arr))
    rating_var = Variable(torch.LongTensor(rating_arr))

    weight_mask = Variable(torch.FloatTensor(review_arr))
    weight_mask[weight_mask != float(vocab2id['<pad>'])] = -1.0
    weight_mask[weight_mask == float(vocab2id['<pad>'])] = 0.0
    weight_mask = -weight_mask

    return review_var, weight_mask, rating_var
