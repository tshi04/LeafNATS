'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import glob
import json
import os
import time

import spacy
import torch
from torch.autograd import Variable

from LeafNATS.data.summarization.load_multitask import *
from LeafNATS.data.utils import construct_vocab
from LeafNATS.modules.decoding.word_copy import word_copy
from LeafNATS.utils.utils import *

from .beam_search import fast_beam_search
from .model import modelPointerGenerator

nlp = spacy.load('en', disable=['logging', 'ner'])


class modelPGApp(modelPointerGenerator):
    '''
    pointer generator network
    '''

    def __init__(self, args):
        super(modelPGApp, self).__init__(args=args)

        self.args.data_dir = self.args.app_model_dir

    def init_base_model_params(self):
        '''
        Initialize Model Parameters
        '''
        for model_name in self.base_models:
            fl_ = os.path.join(self.args.app_model_dir, model_name+'.model')
            self.base_models[model_name].load_state_dict(torch.load(
                fl_, map_location=lambda storage, loc: storage))

    def attnWeight2rgbPercent(self, input_):

        maxV = np.max(input_)
        minV = np.min(input_)
        output_ = (input_ - minV) / (maxV - minV)

        return output_

    def app_worker(self):
        '''
        For the beam search in application.
        '''
        files_ = glob.glob(os.path.join(self.args.app_data_dir, '*_in.json'))
        for curr_file in files_:
            print("Read {}.".format(curr_file))
            fTmp = re.split('\_', curr_file)[0]
            fp = open(curr_file, 'r')
            data_input = json.load(fp)
            fp.close()
            article = nlp(data_input['content'].lower())
            article = ' '.join([wd.text for wd in article])
            article = re.split('\s', article)
            article = list(filter(None, article))
            data_input['content_token'] = article

            self.args.src_seq_lens = len(article)
            ext_id2oov, src_var, src_var_ex, src_arr, src_msk = \
                process_data_app(
                    data_input, self.batch_data['vocab2id'], self.args.src_seq_lens)
            self.batch_data['ext_id2oov'] = ext_id2oov
            src_msk = src_msk.to(self.args.device)

            curr_batch_size = src_var.size(0)
            src_text_rep = src_var.unsqueeze(1).clone().repeat(
                1, self.args.beam_size, 1).view(-1, src_var.size(1)).to(self.args.device)
            if self.args.oov_explicit:
                src_text_rep_ex = src_var_ex.unsqueeze(1).clone().repeat(
                    1, self.args.beam_size, 1).view(-1, src_var_ex.size(1)).to(self.args.device)
            else:
                src_text_rep_ex = src_text_rep.clone()

            beam_seq, beam_prb, beam_attn_ = fast_beam_search(
                self.args, self.base_models, self.batch_data, src_text_rep, src_text_rep_ex, curr_batch_size)
            beam_out = beam_attn_[:, :, 0].squeeze(
            )[:, :self.args.src_seq_lens].data.cpu().numpy()
            beam_out = self.attnWeight2rgbPercent(beam_out)
            trg_arr = word_copy(
                self.args, beam_seq, beam_attn_, src_msk, src_arr, curr_batch_size,
                self.batch_data['id2vocab'], self.batch_data['ext_id2oov'])
            trg_arr = re.split('\s', trg_arr[0])
            out_arr = []
            for idx, wd in enumerate(trg_arr):
                if wd == '<stop>':
                    break
                if wd != '<s>' and wd != '</s>':
                    out_arr.append(
                        {"key": wd, "attention": beam_out[idx].tolist()})
            data_input['summary'] = out_arr

            print('Write {}.'.format(fTmp+'_out.json'))
            fout = open(fTmp+'_out.json', 'w')
            json.dump(data_input, fout)
            fout.close()

#             os.unlink(curr_file)
