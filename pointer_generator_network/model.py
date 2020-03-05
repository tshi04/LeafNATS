'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable

from LeafNATS.data.summarization.load_single import *
from LeafNATS.data.utils import construct_vocab
from LeafNATS.engines.end2end import natsEnd2EndBase
from LeafNATS.modules.decoder.nats_decoder_pointer_generator import \
    PointerGeneratorDecoder
from LeafNATS.modules.decoding.word_copy import word_copy
from LeafNATS.modules.embedding.nats_embedding import natsEmbedding
from LeafNATS.modules.encoder2decoder.nats_encoder2decoder import \
    natsEncoder2Decoder
from LeafNATS.modules.encoder.nats_encoder_rnn import natsEncoder
from LeafNATS.utils.utils import *

from .beam_search import fast_beam_search


class modelPointerGenerator(natsEnd2EndBase):
    '''
    pointer generator network
    '''

    def __init__(self, args):
        super(modelPointerGenerator, self).__init__(args=args)

    def build_vocabulary(self):
        '''
        vocabulary
        '''
        vocab2id, id2vocab = construct_vocab(
            file_=os.path.join(self.args.data_dir, self.args.file_vocab),
            max_size=self.args.max_vocab_size,
            mincount=self.args.word_minfreq)
        vocab_size = len(vocab2id)
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['vocab_size'] = vocab_size
        print('The vocabulary size: {}'.format(vocab_size))

    def build_models(self):
        '''
        build all models.
        in this model source and target share embeddings
        '''
        self.train_models['embedding'] = natsEmbedding(
            vocab_size=self.batch_data['vocab_size'],
            emb_dim=self.args.emb_dim,
            share_emb_weight=self.args.share_emb_weight
        ).to(self.args.device)

        self.train_models['encoder'] = natsEncoder(
            emb_dim=self.args.emb_dim,
            hidden_size=self.args.src_hidden_dim,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        self.train_models['encoder2decoder'] = natsEncoder2Decoder(
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            rnn_network=self.args.rnn_network
        ).to(self.args.device)

        self.train_models['pgdecoder'] = PointerGeneratorDecoder(
            input_size=self.args.emb_dim,
            src_hidden_size=self.args.src_hidden_dim,
            trg_hidden_size=self.args.trg_hidden_dim,
            attn_method=self.args.attn_method,
            repetition=self.args.repetition,
            pointer_net=self.args.pointer_net,
            attn_decoder=self.args.attn_decoder,
            rnn_network=self.args.rnn_network,
            device=self.args.device
        ).to(self.args.device)

        # decoder to vocab
        if self.args.share_emb_weight:
            self.train_models['decoder2proj'] = torch.nn.Linear(
                self.args.trg_hidden_dim, self.args.emb_dim, bias=False).to(self.args.device)
        else:
            self.train_models['decoder2vocab'] = torch.nn.Linear(
                self.args.trg_hidden_dim, self.batch_data['vocab_size']).to(self.args.device)

    def build_optimizer(self, params):
        '''
        init model optimizer
        '''
        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)

        return optimizer

    def build_batch(self, batch_id):
        '''
        get batch data
        '''
        if self.args.oov_explicit:
            ext_id2oov, src_var, trg_input_var, \
                src_var_ex, trg_output_var = process_minibatch_explicit(
                    batch_id=batch_id,
                    path_=os.path.join('..', 'nats_results'),
                    fkey_=self.args.task,
                    batch_size=self.args.batch_size,
                    vocab2id=self.batch_data['vocab2id'],
                    max_lens=[self.args.src_seq_lens, self.args.trg_seq_lens])
            src_var = src_var.to(self.args.device)
            src_var_ex = src_var_ex.to(self.args.device)
            trg_input_var = trg_input_var.to(self.args.device)
            trg_output_var = trg_output_var.to(self.args.device)
        else:
            src_var, trg_input_var, trg_output_var = process_minibatch(
                batch_id=batch_id,
                path_=os.path.join('..', 'nats_results'),
                fkey_=self.args.task,
                batch_size=self.args.batch_size,
                vocab2id=self.batch_data['vocab2id'],
                max_lens=[self.args.src_seq_lens, self.args.trg_seq_lens])
            ext_id2oov = {}
            src_var = src_var.to(self.args.device)
            src_var_ex = src_var.clone()
            trg_input_var = trg_input_var.to(self.args.device)
            trg_output_var = trg_output_var.to(self.args.device)

        self.batch_data['ext_id2oov'] = ext_id2oov
        self.batch_data['src_var'] = src_var
        self.batch_data['src_var_ex'] = src_var_ex
        self.batch_data['trg_input_var'] = trg_input_var
        self.batch_data['trg_output_var'] = trg_output_var

    def build_pipelines(self):
        '''
        here we have all data flow from the input to output
        '''
        src_emb = self.train_models['embedding'].get_embedding(
            self.batch_data['src_var'])
        encoder_hy, hidden_encoder = self.train_models['encoder'](src_emb)
        hidden_decoder = self.train_models['encoder2decoder'](hidden_encoder)

        trg_emb = self.train_models['embedding'].get_embedding(
            self.batch_data['trg_input_var'])

        batch_size = self.batch_data['src_var'].size(0)
        src_seq_len = self.batch_data['src_var'].size(1)
        trg_seq_len = trg_emb.size(1)
        if self.args.repetition == 'temporal':
            past_attn = Variable(torch.ones(
                batch_size, src_seq_len)).to(self.args.device)
        else:
            past_attn = Variable(torch.zeros(
                batch_size, src_seq_len)).to(self.args.device)
        h_attn = Variable(torch.zeros(batch_size, self.args.trg_hidden_dim)).to(
            self.args.device)
        p_gen = Variable(torch.zeros(batch_size, trg_seq_len)
                         ).to(self.args.device)
        past_dehy = Variable(torch.zeros(1, 1)).to(self.args.device)

        trg_h, _, _, attn_, _, p_gen, _, loss_cv = self.train_models['pgdecoder'](
            0, trg_emb, hidden_decoder, h_attn, encoder_hy, past_attn, p_gen, past_dehy)

        # prepare output
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0)*trg_h.size(1), trg_h.size(2))
        # consume a lot of memory.
        if self.args.share_emb_weight:
            decoder_proj = self.train_models['decoder2proj'](trg_h_reshape)
            logits_ = self.train_models['embedding'].get_decode2vocab(
                decoder_proj)
        else:
            logits_ = self.train_models['decoder2vocab'](trg_h_reshape)
        logits_ = logits_.view(trg_h.size(0), trg_h.size(1), logits_.size(1))
        logits_ = torch.softmax(logits_, dim=2)

        ex_vocab_size = len(self.batch_data['ext_id2oov'])
        vocab_size = len(self.batch_data['vocab2id']) + ex_vocab_size
        if self.args.pointer_net:
            if self.args.oov_explicit:
                logits_ex = Variable(torch.zeros(1, 1, 1)).to(self.args.device)
                logits_ex = logits_ex.repeat(
                    batch_size, trg_seq_len, ex_vocab_size)
                if ex_vocab_size > 0:
                    logits_ = torch.cat((logits_, logits_ex), -1)
                # pointer
                attn_ = attn_.transpose(0, 1)
                # calculate index matrix
                pt_idx = Variable(torch.FloatTensor(
                    torch.zeros(1, 1, 1))).to(self.args.device)
                pt_idx = pt_idx.repeat(batch_size, src_seq_len, vocab_size)
                pt_idx.scatter_(
                    2, self.batch_data['src_var_ex'].unsqueeze(2), 1.0)
                logits_ = p_gen.unsqueeze(
                    2)*logits_ + (1.0-p_gen.unsqueeze(2))*torch.bmm(attn_, pt_idx)
                logits_ = logits_ + 1e-20
            else:
                attn_ = attn_.transpose(0, 1)
                pt_idx = Variable(torch.FloatTensor(
                    torch.zeros(1, 1, 1))).to(self.args.device)
                pt_idx = pt_idx.repeat(batch_size, src_seq_len, vocab_size)
                pt_idx.scatter_(
                    2, self.batch_data['src_var'].unsqueeze(2), 1.0)
                logits_ = p_gen.unsqueeze(
                    2)*logits_ + (1.0-p_gen.unsqueeze(2))*torch.bmm(attn_, pt_idx)

        weight_mask = torch.ones(vocab_size).to(self.args.device)
        weight_mask[self.batch_data['vocab2id']['<pad>']] = 0
        loss_criterion = torch.nn.NLLLoss(
            weight=weight_mask).to(self.args.device)

        logits_ = torch.log(logits_)
        loss = loss_criterion(
            logits_.contiguous().view(-1, vocab_size),
            self.batch_data['trg_output_var'].view(-1))

        if self.args.repetition == 'asee_train':
            loss = loss + loss_cv[0]

        return loss

    def test_worker(self, _nbatch):
        '''
        For the beam search in testing.
        '''
        start_time = time.time()
        fout = open(os.path.join('..', 'nats_results',
                                 self.args.file_output), 'w')
        for batch_id in range(_nbatch):
            ext_id2oov, src_var, src_var_ex, src_arr, src_msk, trg_arr \
                = process_minibatch_explicit_test(
                    batch_id=batch_id,
                    path_=os.path.join('..', 'nats_results'),
                    fkey_=self.args.task,
                    batch_size=self.args.test_batch_size,
                    vocab2id=self.batch_data['vocab2id'],
                    src_lens=self.args.src_seq_lens
                )
            src_msk = src_msk.to(self.args.device)
            src_var = src_var.to(self.args.device)
            src_var_ex = src_var_ex.to(self.args.device)
            self.batch_data['ext_id2oov'] = ext_id2oov

            curr_batch_size = src_var.size(0)
            src_text_rep = src_var.unsqueeze(1).clone().repeat(
                1, self.args.beam_size, 1).view(-1, src_var.size(1)).to(self.args.device)
            if self.args.oov_explicit:
                src_text_rep_ex = src_var_ex.unsqueeze(1).clone().repeat(
                    1, self.args.beam_size, 1).view(-1, src_var_ex.size(1)).to(self.args.device)
            else:
                src_text_rep_ex = src_text_rep.clone()

            models = {}
            for model_name in self.base_models:
                models[model_name] = self.base_models[model_name]
            for model_name in self.train_models:
                models[model_name] = self.train_models[model_name]
            beam_seq, beam_prb, beam_attn_ = fast_beam_search(
                self.args, models, self.batch_data,
                src_text_rep, src_text_rep_ex, curr_batch_size)
            # copy unknown words
            out_arr = word_copy(
                self.args, beam_seq, beam_attn_, src_msk, src_arr, curr_batch_size,
                self.batch_data['id2vocab'], self.batch_data['ext_id2oov'])
            for k in range(curr_batch_size):
                fout.write('<sec>'.join([out_arr[k], trg_arr[k]])+'\n')

            end_time = time.time()
            show_progress(batch_id, _nbatch, str(
                (end_time-start_time)/3600)[:8]+"h")
        fout.close()
