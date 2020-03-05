'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.utils.utils import tensor_transformer


def fast_beam_search(args, models, batch_data,
                     src_text_rep, src_text_rep_ex,
                     batch_size, task_key):
    '''
    fast beam search
    '''

    src_emb = models['embedding_base'].get_embedding(src_text_rep)
    encoder_hy0, _ = models['encoder_base'](src_emb)
    encoder_hy, hidden_encoder = models['encoder_'+task_key](encoder_hy0)
    hidden_decoder = models['encoder2decoder_'+task_key](hidden_encoder)
    if args.rnn_network == 'lstm':
        (h0_new, c0_new) = hidden_decoder
    else:
        h0_new = hidden_decoder

    beam_size = args.beam_size
    src_seq_len = src_emb.size(1)
    if args.repetition == 'temporal':
        past_attn_new = Variable(torch.ones(
            batch_size*beam_size, src_seq_len)).to(args.device)
    else:
        past_attn_new = Variable(torch.zeros(
            batch_size*beam_size, src_seq_len)).to(args.device)
    h_attn_new = Variable(torch.zeros(
        batch_size*beam_size, args.trg_hidden_dim)).to(args.device)
    past_dehy_new = Variable(torch.zeros(1, 1)).to(args.device)

    if args.task_key[-7:] == 'summary':
        max_len = args.sum_seq_lens
    if args.task_key[-5:] == 'title':
        max_len = args.ttl_seq_lens
    beam_seq = Variable(torch.LongTensor(
        batch_size, beam_size, max_len+1).fill_(batch_data['vocab2id']['<pad>'])).to(args.device)
    beam_seq[:, :, 0] = batch_data['vocab2id']['<s>']
    beam_prb = torch.FloatTensor(batch_size, beam_size).fill_(1.0)
    last_wd = Variable(torch.LongTensor(batch_size, beam_size, 1).fill_(
        batch_data['vocab2id']['<s>'])).to(args.device)
    beam_attn_ = Variable(torch.FloatTensor(
        max_len, batch_size, beam_size, src_seq_len).fill_(0.0)).to(args.device)

    for j in range(max_len):
        if args.oov_explicit:
            last_wd[last_wd >= len(batch_data['vocab2id'])
                    ] = batch_data['vocab2id']['<unk>']
        trg_emb = models['embedding_base'].get_embedding(last_wd.view(-1, 1))
        p_gen = Variable(torch.zeros(batch_size*beam_size, 1)).to(args.device)
        if args.rnn_network == 'lstm':
            trg_h, (h0, c0), h_attn, attn_, past_attn, p_gen, past_dehy, _ =\
                models['pgdecoder_'+task_key](
                j, trg_emb, (h0_new, c0_new), h_attn_new, encoder_hy, past_attn_new, p_gen, past_dehy_new)
        elif network == 'gru':
            trg_h, h0, h_attn, attn_, past_attn, p_gen, past_dehy, _ =\
                models['pgdecoder_'+task_key](
                    j, trg_emb, h0_new, h_attn_new, encoder_hy, past_attn_new, p_gen, past_dehy_new)
        # prepare output
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0)*trg_h.size(1), trg_h.size(2))
        if args.share_emb_weight:
            decoder_proj = models['decoder2proj_'+task_key](trg_h_reshape)
            logits = models['embedding_base'].get_decode2vocab(decoder_proj)
        else:
            logits = models['decoder2vocab_'+task_key](trg_h_reshape)
        logits = logits.view(trg_h.size(0), trg_h.size(1), logits.size(1))
        logits = torch.softmax(logits, dim=2)

        ex_vocab_size = len(batch_data['ext_id2oov'])
        vocab_size = len(batch_data['vocab2id']) + ex_vocab_size
        if args.pointer_net:
            if args.oov_explicit and ex_vocab_size > 0:
                logitsex = Variable(torch.zeros(1, 1, 1)).to(args.device)
                logitsex = logitsex.repeat(
                    batch_size*beam_size, 1, ex_vocab_size)
                if ex_vocab_size > 0:
                    logits = torch.cat((logits, logitsex), -1)
                # pointer
                attn_ = attn_.transpose(0, 1)
                # calculate index matrix
                pt_idx = Variable(torch.FloatTensor(
                    torch.zeros(1, 1, 1))).to(args.device)
                pt_idx = pt_idx.repeat(
                    batch_size*beam_size, src_seq_len, vocab_size)
                pt_idx.scatter_(2, src_text_rep_ex.unsqueeze(2), 1.0)
                logits = p_gen.unsqueeze(
                    2)*logits + (1.0-p_gen.unsqueeze(2))*torch.bmm(attn_, pt_idx)
                logits = logits + 1e-20
            else:
                attn_ = attn_.transpose(0, 1)
                pt_idx = Variable(torch.FloatTensor(
                    torch.zeros(1, 1, 1))).to(args.device)
                pt_idx = pt_idx.repeat(
                    batch_size*beam_size, src_seq_len, vocab_size)
                pt_idx.scatter_(2, src_text_rep.unsqueeze(2), 1.0)
                logits = p_gen.unsqueeze(
                    2)*logits + (1.0-p_gen.unsqueeze(2))*torch.bmm(attn_, pt_idx)

        prob, wds = logits.data.topk(k=beam_size)
        prob = prob.view(batch_size, beam_size, prob.size(1), prob.size(2))
        wds = wds.view(batch_size, beam_size, wds.size(1), wds.size(2))
        if j == 0:
            beam_prb = prob[:, 0, 0]
            beam_seq[:, :, 1] = wds[:, 0, 0]
            last_wd = Variable(wds[:, 0, 0].unsqueeze(
                2).clone()).to(args.device)

            if args.rnn_network == 'lstm':
                h0_new = h0
                c0_new = c0
            else:
                hidden_decoder_new = hidden_decoder
            h_attn_new = h_attn
            attn_new = attn_
            past_attn_new = past_attn
            past_dehy_new = past_dehy
            beam_attn_[j] = attn_new.view(
                batch_size, beam_size, attn_new.size(-1))
            continue

        cand_seq = tensor_transformer(beam_seq, batch_size, beam_size)
        cand_seq[:, :, j+1] = wds.squeeze(2).view(batch_size, -1)
        cand_last_wd = wds.squeeze(2).view(batch_size, -1)

        cand_prob = beam_prb.unsqueeze(1).repeat(
            1, beam_size, 1).transpose(1, 2)
        cand_prob *= prob[:, :, 0]
        cand_prob = cand_prob.contiguous().view(batch_size, beam_size*beam_size)
        if args.rnn_network == 'lstm':
            h0_new = Variable(torch.zeros(
                batch_size, beam_size, h0.size(-1))).to(args.device)
            c0_new = Variable(torch.zeros(
                batch_size, beam_size, c0.size(-1))).to(args.device)
        else:
            hidden_decoder_new = Variable(torch.zeros(
                batch_size, beam_size, hidden_decoder.size(-1))).to(args.device)
        h_attn_new = Variable(torch.zeros(
            batch_size, beam_size, h_attn.size(-1))).to(args.device)
        attn_new = Variable(torch.zeros(
            batch_size, beam_size, attn_.size(-1))).to(args.device)
        past_attn_new = Variable(torch.zeros(
            batch_size, beam_size, past_attn.size(-1))).to(args.device)
        if args.attn_decoder:
            pdn_size1, pdn_size2 = past_dehy.size(-2), past_dehy.size(-1)
            past_dehy_new = Variable(torch.zeros(
                batch_size, beam_size, pdn_size1*pdn_size2)).to(args.device)
        if args.rnn_network == 'lstm':
            h0 = h0.view(batch_size, beam_size, h0.size(-1))
            h0 = tensor_transformer(h0, batch_size, beam_size)
            c0 = c0.view(batch_size, beam_size, c0.size(-1))
            c0 = tensor_transformer(c0, batch_size, beam_size)
        else:
            hidden_decoder = hidden_decoder.view(
                batch_size, beam_size, hidden_decoder.size(-1))
            hidden_decoder = tensor_transformer(
                hidden_decoder, batch_size, beam_size)
        h_attn = h_attn.view(batch_size, beam_size, h_attn.size(-1))
        h_attn = tensor_transformer(h_attn, batch_size, beam_size)
        attn_ = attn_.view(batch_size, beam_size, attn_.size(-1))
        attn_ = tensor_transformer(attn_, batch_size, beam_size)
        past_attn = past_attn.view(batch_size, beam_size, past_attn.size(-1))
        past_attn = tensor_transformer(past_attn, batch_size, beam_size)
        if args.attn_decoder:
            past_dehy = past_dehy.contiguous().view(
                batch_size, beam_size, past_dehy.size(-2)*past_dehy.size(-1))
            past_dehy = tensor_transformer(past_dehy, batch_size, beam_size)
        tmp_prb, tmp_idx = cand_prob.topk(k=beam_size, dim=1)
        for x in range(batch_size):
            for b in range(beam_size):
                last_wd[x, b] = cand_last_wd[x, tmp_idx[x, b]]
                beam_seq[x, b] = cand_seq[x, tmp_idx[x, b]]
                beam_prb[x, b] = tmp_prb[x, b]

                if args.rnn_network == 'lstm':
                    h0_new[x, b] = h0[x, tmp_idx[x, b]]
                    c0_new[x, b] = c0[x, tmp_idx[x, b]]
                else:
                    hidden_decoder_new[x, b] = hidden_decoder[x, tmp_idx[x, b]]
                h_attn_new[x, b] = h_attn[x, tmp_idx[x, b]]
                attn_new[x, b] = attn_[x, tmp_idx[x, b]]
                past_attn_new[x, b] = past_attn[x, tmp_idx[x, b]]
                if args.attn_decoder:
                    past_dehy_new[x, b] = past_dehy[x, tmp_idx[x, b]]

        beam_attn_[j] = attn_new
        if args.rnn_network == 'lstm':
            h0_new = h0_new.view(-1, h0_new.size(-1))
            c0_new = c0_new.view(-1, c0_new.size(-1))
        else:
            hidden_decoder_new = hidden_decoder_new.view(
                -1, hidden_decoder_new.size(-1))
        h_attn_new = h_attn_new.view(-1, h_attn_new.size(-1))
        attn_new = attn_new.view(-1, attn_new.size(-1))
        past_attn_new = past_attn_new.view(-1, past_attn_new.size(-1))
        if args.attn_decoder:
            past_dehy_new = past_dehy_new.view(-1, pdn_size1, pdn_size2)

#         torch.cuda.empty_cache()
    return beam_seq, beam_prb, beam_attn_
