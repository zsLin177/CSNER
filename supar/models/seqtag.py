
import pdb, math
from typing import no_type_check_decorator
from torch import autograd
from torch._C import dtype
import torch.nn as nn
from torch.nn.init import normal_
from supar.models.model import Model
from supar.modules import MLP
from supar.utils import Config
from supar.utils.common import MIN
# from supar.structs import CRFLinearChain
from supar.structs import LinearChainCRF
from supar.modules.scalar_mix import ScalarMix
import torch
from torch.nn.utils.rnn import pad_sequence
# from parser.JParser import JParser

class HmmModel(nn.Module):
    r"""
    TODO: introduction
    """
    def __init__(self, n_words, n_labels, pad_index, unk_index, alpha=0.01, **kwargs):
        super().__init__()
        self.args = Config().update(locals())
        self.n_words = n_words
        self.n_labels = n_labels
        self.unk_index = unk_index
        self.pad_index = pad_index
        self.alpha = alpha


        self.trans_count = nn.Parameter(torch.zeros((n_labels+1, n_labels+1)), requires_grad=False)
        self.emit_count = nn.Parameter(torch.zeros((n_words, n_labels)), requires_grad=False)
        self.trans = nn.Parameter(torch.empty((n_labels, n_labels)), requires_grad=False)
        self.strans = nn.Parameter(torch.empty((n_labels,)), requires_grad=False)
        self.emit = nn.Parameter(torch.empty((n_words, n_labels)), requires_grad=False)

    def forward(self, words, labels):
        '''
        just use during training, to count
        words: [batch_size, seq_len]
        labels: [batch_size, seq_len]
        no bos and eos
        '''
        batch_size = words.shape[0]
        # [batch_size]
        lens = words.ne(self.pad_index).sum(-1)
        ext_mask = words.ge(self.n_words)
        ext_words = words.masked_fill(ext_mask, self.unk_index)
        for i in range(batch_size):
            pre = -1
            for j in range(lens[i]):
                w = ext_words[i, j]
                t = labels[i, j]
                self.trans_count[t, pre] += 1
                self.emit_count[w, t] += 1
                pre = t

    
    def smooth_logp(self):
        '''
        get the smoothed log probability
        '''
        alpha = self.alpha
        sum_t_c = torch.sum(self.trans_count, 0)
        smoothed_t_p = (self.trans_count + alpha) / (sum_t_c + alpha*self.trans_count.shape[0])
        sum_e_c = torch.sum(self.emit_count, 0)
        smoothed_e_p = (self.emit_count + alpha) / (sum_e_c + alpha*self.emit_count.shape[0])
        # x->y : post<-pre
        self.trans.data = torch.log(smoothed_t_p[:-1, :-1])
        self.strans.data = torch.log(smoothed_t_p[:-1, -1])
        self.emit.data = torch.log(smoothed_e_p)
    
    def decode(self, words):
        '''
        use viterbi decode
        words: [batch_size, seq_len]
        '''
        batch_size, seq_len = words.shape
        ext_mask = words.ge(self.n_words)
        ext_words = words.masked_fill(ext_mask, self.unk_index)
        # [batch_size, seq_len, n_labels]
        emit = self.emit[ext_words.view(batch_size*seq_len)].reshape((batch_size, seq_len, self.n_labels))
        # [seq_len, batch_size, n_labels]
        emit = emit.transpose(0, 1)
        n_tags = emit.shape[2]
        # x->y: pre->post
        trans = self.trans.transpose(0, 1)
        delta = emit.new_zeros(seq_len, batch_size, n_tags)
        paths = emit.new_zeros(seq_len, batch_size, n_tags, dtype=torch.long)
        # [batch_size, n_tags]
        delta[0] = self.strans + emit[0]

        for i in range(1, seq_len):
            scores = trans + delta[i - 1].unsqueeze(-1)
            scores, paths[i] = scores.max(1)
            delta[i] = scores + emit[i]

        preds = []
        lens = words.ne(self.pad_index).sum(-1)
        for i, length in enumerate(lens.tolist()):
            prev = torch.argmax(delta[length-1, i])
            pred = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                pred.append(prev)
            preds.append(paths.new_tensor(pred).flip(0))
        
        preds = pad_sequence(preds, True, -1)
        return preds

class SimpleSeqTagModel(Model):
    r"""
    TODO: introduction
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 syntax=False,
                 synatax_path='',
                 n_syntax=800,
                 mix=False,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.2,
                 n_lstm_hidden=300,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp=400,
                 n_mlp_argument=300,
                 n_mlp_relation=300,
                 repr_mlp_dropout=.3,
                 scorer_mlp_dropout=.2,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 self_uncertain=False,
                 policy_grad=False,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        self.n_labels = n_labels
        self.n_mlp = n_mlp
        self.syntax = syntax
        self.n_synatax = n_syntax
        self.mix = mix
        self.self_uncertain = self_uncertain
        self.policy_grad = policy_grad
        
        self.repr_mlp = MLP(n_in=self.args.n_hidden, n_out=n_mlp, dropout=repr_mlp_dropout)
        self.scorer = MLP(n_in=n_mlp, n_out=n_labels, activation=False)
        if not self_uncertain:
            self.ce_criterion = nn.CrossEntropyLoss()
            self.pg_ce_criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.uncer_mlp = nn.Sequential(MLP(n_in=self.args.n_hidden, n_out=self.args.n_hidden//2, dropout=repr_mlp_dropout),
                                            MLP(n_in=self.args.n_hidden//2, n_out=n_mlp, dropout=repr_mlp_dropout))
            self.uncer_scorer = MLP(n_in=n_mlp, n_out=1, activation=False)

    def forward(self, words, sens_lst=None, feats=None):
        # [batch_size, seq_len, n_hidden]
        x = self.encode(words, feats)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(self.repr_mlp(x))
        if not self.self_uncertain:
            return score, None
        else:
            # [batch_size, seq_len]
            uncer = self.uncer_scorer(self.uncer_mlp(x)).squeeze(-1)
            return score, uncer
    
    def su_metric(self, uncer, threshold=0.5):
        ssu = self.sig_scale(uncer.pow(2))
        ssu_mask = ssu.gt(threshold)
        return ssu, ssu_mask

    def multi_forward(self, words, sens_lst=None, feats=None, times=3, if_T=False, req_grad=False):
        '''
        To produce the set of predicted distributions Q={q1, q2, q3, ...}
        '''
        if not req_grad:
            with torch.no_grad():
                q_set = []
                score_T = words.new_tensor([self.n_mlp], dtype=torch.float).sqrt()
                for i in range(times):
                    tmp_feats = feats[:]
                    # [batch_size, seq_len, n_hidden]
                    x = self.encode(words, tmp_feats)
                    # [batch_size, seq_len, n_labels]
                    score = self.scorer(self.repr_mlp(x))
                    if if_T:
                        score /= score_T
                    q_set.append(score.softmax(-1))
                # [batch_size, seq_len, times, n_labels]
                q = torch.stack(q_set).permute(1, 2, 0, 3)
                return q
        else:
            q_set = []
            score_T = words.new_tensor([self.n_mlp], dtype=torch.float).sqrt()
            for i in range(times):
                tmp_feats = feats[:]
                # [batch_size, seq_len, n_hidden]
                x = self.encode(words, tmp_feats)
                # [batch_size, seq_len, n_labels]
                score = self.scorer(self.repr_mlp(x))
                if if_T:
                    score /= score_T
                q_set.append(score.softmax(-1))
            # [batch_size, seq_len, times, n_labels]
            q = torch.stack(q_set).permute(1, 2, 0, 3)
            return q

    def skl_div(self, q, p=None):
        '''
        q: predicted distributions
            [batch_size, seq_len, times, n_labels]
        p: the "gold" label
            [batch_size, seq_len]
        '''
        batch_size, seq_len, times, _ = q.shape
        if p != None:
            # return the kl with "gold"
            # return : [batch_size, seq_len, times]
            # [batch_size, seq_len, n_labels, times]
            q = -q.permute(0, 1, 3, 2).log()
            index = p.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, times)
            # [batch_size, seq_len, times]
            kl = torch.gather(q, 2, index).squeeze(2)
            return self.sig_scale(kl)
        else:
            # first compute an average distribution
            # then return skl with this avg distribution
            # return : [batch_size, seq_len, times]
            # [batch_size, seq_len, n_labels]
            avg_dis = q.mean(2)
            # [batch_size, seq_len]
            neg_avg_e = (avg_dis * avg_dis.log()).sum(-1)
            expanded_avg = avg_dis.unsqueeze(2).expand(-1, -1, times, -1)
            # [batch_size, seq_len, times]
            cross_e = -(expanded_avg * q.log()).sum(-1)
            kl = neg_avg_e.unsqueeze(-1) + cross_e
            return self.sig_scale(kl)

    def avg_metric(self, q, p=None, threshold=0.5, req_grad=False):
        '''
        return : [batch_size, seq_len]
        '''
        if not req_grad:
            with torch.no_grad():
                skl = self.skl_div(q, p)
                # skl:[batch_size, seq_len, times]
                res = skl.mean(-1)
                res_mask = res.gt(threshold)
                return res, res_mask
        else:
            skl = self.skl_div(q, p)
            # skl:[batch_size, seq_len, times]
            res = skl.mean(-1)
            res_mask = res.gt(threshold)
            return res, res_mask
    
    def vote_metric(self, q, p=None, vote_low_rate=0.3, vote_up_rate=0.9):
        '''
        q: predicted distributions
            [batch_size, seq_len, times, n_labels]
        p: the "gold" label
            [batch_size, seq_len]
        '''
        times = q.shape[2]
        preds = q.argmax(-1)
        if p is not None:
            # [batch_size, seq_len]
            vote = times - p.unsqueeze(-1).expand(-1, -1, times).eq(preds).sum(-1)
            # if the 'gold' label gets votes more than vote_rate, then we think this may be a mistake
            mask = (vote/times).ge(vote_low_rate) & (vote/times).le(vote_up_rate)
            return vote, mask
        else:
            # [batch_size, seq_len]
            mode = torch.mode(preds, -1)[0]
            uncer = 1 - (mode.unsqueeze(-1).expand(-1, -1, times).eq(preds).sum(-1)/times)
            mask = uncer.ge(vote_low_rate)
            return uncer, mask



    def var_metric(self, q, p=None, varhold=0.1):
        skl = self.skl_div(q, p)
        # skl:[batch_size, seq_len, times]
        res = skl.std(-1)
        res_mask = res.gt(varhold)
        return res, res_mask
        
    def sig_scale(self, kl):
        '''
        scale the kl based on sigmoid:
            skl = 2 * (sigmoid(kl) - 0.5)
            0<skl<1
        '''
        return 2*(kl.sigmoid()-0.5)

    def decode(self, score, uncer=None):
        """
        TODO:introduce
        """
        if not self.self_uncertain:
            return score.argmax(-1)
        else:
            # [batch_size, seq_len, n_labels]
            sample = torch.empty_like(score).normal_(mean=0, std=1)
            f_score = score + uncer.unsqueeze(-1) * sample
            return f_score.argmax(-1)


    def loss(self, score, gold_labels, mask, reward=None):
        """
        TODO:
        uncer: [batch_size, seq_len]
        """
        if not self.policy_grad:
            return self.ce_criterion(score[mask], gold_labels[mask[:, 1:]])
        else:
            if reward is not None:
                # [k]
                each_loss = self.pg_ce_criterion(score[mask], gold_labels[mask[:, 1:]])
                loss = (reward[mask] * each_loss).mean()
                return loss
            else:
                '''
                need policy gradient but without aux model and epoch <= start epoch
                '''
                return self.ce_criterion(score[mask], gold_labels[mask[:, 1:]])

class CrfSeqTagModel(Model):
    r"""
    TODO: introduction
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 syntax=False,
                 synatax_path='',
                 n_syntax=800,
                 mix=False,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 bert_req_grad=False,
                 embed_dropout=.2,
                 n_lstm_hidden=300,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp=400,
                 n_mlp_argument=300,
                 n_mlp_relation=300,
                 repr_mlp_dropout=.2,
                 scorer_mlp_dropout=.2,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 split=True,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        self.n_labels = n_labels
        self.n_mlp = n_mlp
        self.syntax = syntax
        self.n_synatax = n_syntax
        self.mix = mix
        
        self.repr_mlp = MLP(n_in=self.args.n_hidden, n_out=n_mlp, dropout=repr_mlp_dropout)
        self.scorer = MLP(n_in=n_mlp, n_out=n_labels, activation=False)
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, words, sens_lst=None, feats=None):
        # batch_size, seq_len, _ = words.shape
        # [batch_size, seq_len, n_hidden]
        x = self.encode(words, feats)

        # [batch_size, seq_len, n_labels]
        score = self.scorer(self.repr_mlp(x))
        return score

    def decode(self, score, mask):
        """
        TODO: return the best label seq, and the prob
        """
        # dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        # return dist.argmax.argmax(-1), (dist.max - dist.log_partition).exp()
        dist = LinearChainCRF(score[:, 1:], self.trans, lens=mask[:, 1:].sum(-1))
        return dist.argmax, (dist.max - dist.log_partition).exp()


    def loss(self, score, gold_labels, mask):
        """
        TODO:introduce
        """
        batch_size, seq_len = mask.shape
        # loss = -CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans).log_prob(gold_labels).sum() / seq_len
        loss = -LinearChainCRF(score[:, 1:], self.trans, lens=mask[:, 1:].sum(-1)).log_prob(gold_labels).sum() / seq_len
        return loss
    
    def dist(self, score, mask):
        # return CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return LinearChainCRF(score[:, 1:], self.trans, lens=mask[:, 1:].sum(-1))
    
    def kl_loss(self, score, mask, other_dist):
        this_dist = self.dist(score, mask)
        return this_dist.kl(other_dist).sum() / score.shape[0]

class SentLabelModel(Model):
    def __init__(self,
                 n_words,
                 n_labels,
                 syntax=False,
                 synatax_path='',
                 n_syntax=800,
                 mix=False,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 bert_req_grad=False,
                 embed_dropout=.2,
                 n_lstm_hidden=300,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp=400,
                 n_mlp_argument=300,
                 n_mlp_relation=300,
                 repr_mlp_dropout=.2,
                 scorer_mlp_dropout=.2,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 split=True,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        self.n_labels = n_labels
        self.n_mlp = n_mlp
        self.repr_mlp = MLP(n_in=self.args.n_hidden, n_out=n_mlp, dropout=repr_mlp_dropout)
        self.scorer = MLP(n_in=n_mlp, n_out=n_labels, activation=False)
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, words, sens_lst=None, feats=None):
        # batch_size, seq_len, _ = words.shape
        # [batch_size, seq_len, n_hidden]
        x = self.encode(words, feats)

        # take the cls as the representation
        # [batch_size, n_hidden]
        x = x[:, 0]

        # [batch_size, n_labels]
        score = self.scorer(self.repr_mlp(x))
        return score
    
    def decode(self, score):
        """
        score: [batch_size, n_labels]
        """
        return score.argmax(-1)
    
    def loss(self, score, gold_labels):
        loss = self.ce_criterion(score, gold_labels)
        return loss



        