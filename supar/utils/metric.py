# -*- coding: utf-8 -*-

from collections import Counter

class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class ArgumentMetric(Metric):
    def __init__(self, eps=1e-12):
        super(Metric, self).__init__()

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0

        self.predicate_t = 0.0
        self.predicate_pred = 0.0
        self.predicate_gold = 0.0

        self.span_t = 0.0
        self.span_pred = 0.0
        self.span_gold = 0.0

        self.eps = eps

    def __call__(self, preds, golds, pred_p, gold_p, pred_span, gold_span):
        """
        preds, golds: [batch_size, seq_len, seq_len, seq_len]
        pred_p, gold_p: [batch_size, seq_len]
        pred_span, gold_span: [batch_size, seq_len, seq_len]
        """
        # TODO: add predicate and span metric
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()

        pred_p_mask = pred_p.gt(0)
        gold_p_mask = gold_p.gt(0)
        p_mask = pred_p_mask & gold_p_mask
        self.predicate_pred += pred_p_mask.sum().item()
        self.predicate_gold += gold_p_mask.sum().item()
        self.predicate_t += p_mask.sum().item()

        pred_s_mask = pred_span.gt(0)
        gold_s_mask = gold_span.gt(0)
        s_mask = pred_s_mask & gold_s_mask
        self.span_pred += pred_s_mask.sum().item()
        self.span_gold += gold_s_mask.sum().item()
        self.span_t += s_mask.sum().item()

        return self

    def __repr__(self):
        return f"P_P: {self.p_p:6.2%} P_R: {self.p_r:6.2%} P_F: {self.p_f:6.2%}  S_P: {self.s_p:6.2%} S_R:{self.s_r:6.2%} S_F: {self.s_f:6.2%} UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def p_p(self):
        return self.predicate_t / (self.predicate_pred + self.eps)
    
    @property
    def p_r(self):
        return self.predicate_t / (self.predicate_gold + self.eps)

    @property
    def p_f(self):
        return 2 * self.predicate_t / (self.predicate_pred + self.predicate_gold + self.eps)

    @property
    def s_p(self):
        return self.span_t / (self.span_pred + self.eps)

    @property
    def s_r(self):
        return self.span_t / (self.span_gold + self.eps)

    @property
    def s_f(self):
        return 2 * self.span_t / (self.span_pred + self.span_gold + self.eps)

    @property
    def score(self):
        return self.f

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)


class AttachmentMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.eps = eps

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
        return s

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        lens = mask.sum(1)
        arc_mask = arc_preds.eq(arc_golds) & mask
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()
        return self

    @property
    def score(self):
        return self.las

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)


class SpanMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            upred = Counter([(i, j) for i, j, label in pred])
            ugold = Counter([(i, j) for i, j, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)
        return self

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"

        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)


class ChartMetric(Metric):

    def __init__(self, eps=1e-12):
        super(ChartMetric, self).__init__()

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()
        return self

    def __repr__(self):
        return f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)


class SeqTagMetric(Metric):
    def __init__(self, label_vocab, eps=1e-12):
        super().__init__()

        self.label_tp_lst = [0.0]*len(label_vocab)
        self.label_pd_lst = [0.0]*len(label_vocab)
        self.label_gd_lst = [0.0]*len(label_vocab)

        self.label_vocab = label_vocab
        self.tp = 0.0
        self.sum_num = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        span_mask = golds.ge(0)
        self.sum_num += span_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        for i in range(len(self.label_vocab)):
            gold_mask = golds.eq(i) 
            pred_mask = preds.eq(i)
            self.label_gd_lst[i] += gold_mask.sum().item()
            self.label_pd_lst[i] += pred_mask.sum().item()
            self.label_tp_lst[i] += (pred_mask & gold_mask & span_mask).sum().item()

        return self

    def __repr__(self):
        s = ''
        for i in range(len(self.label_vocab)):
            label = self.label_vocab[i]
            p = self.label_tp_lst[i] / (self.label_pd_lst[i] + self.eps)
            r = self.label_tp_lst[i] / (self.label_gd_lst[i] + self.eps)
            f = 2*self.label_tp_lst[i] / (self.label_pd_lst[i] + self.eps + self.label_gd_lst[i])
            s += f"{label}: P:{p:6.2%} R:{r:6.2%} F:{f:6.2%} "
        return f"Accuracy: {self.accuracy:6.2%}"

    @property
    def score(self):
        return self.accuracy

    @property
    def accuracy(self):
        return self.tp / (self.sum_num + self.eps)
    
class NERMetric(Metric):
    # with the position 
    def __init__(self, label_vocab_itos, eps=1e-12):
        super().__init__()

        self.label_vocab_itos = label_vocab_itos
        self.utp = 0.0
        self.tp = 0.0
        self.gold = 0.0
        self.pred = 0.0
        self.eps = eps

    def get_nes(self, ne_tensor, ins_labels=None):
        ne_lst = ne_tensor.tolist()
        str_ne_sets = []
        for nes in ne_lst:
            this_ne_set = set()
            this_str_ne_lst = []
            for ne in nes:
                this_str_ne_lst.append(self.label_vocab_itos[ne])
            
            start = 0
            ne_label = ''
            flag = False
            for i, ne_string in enumerate(this_str_ne_lst):
                if ne_string.startswith('B'):
                    if flag:
                        this_ne_set.add((start, i, ne_label))
                        flag = False
                    start = i
                    ne_label = ne_string[2:]
                    flag = True
                elif ne_string.startswith('I'):
                    continue
                else:
                    if flag:
                        this_ne_set.add((start, i, ne_label))
                        flag = False
            if flag:
                this_ne_set.add((start, i+1, ne_label))
                flag = False

            str_ne_sets.append(this_ne_set)
        
        if ins_labels is not None:
            res = []
            for ne_set, ins_label in zip(str_ne_sets, ins_labels):
                if ins_label != "导航":
                    res.append(ne_set)
                else:
                    this_set = set()
                    for ne in ne_set:
                        if ne[2] == "ORG":
                            this_set.add((ne[0], ne[1], "LOC"))
                        else:
                            this_set.add(ne)
                    res.append(this_set)
            return res
        return str_ne_sets
            
    def __call__(self, preds, golds, ins_labels=None):
        pred_nes = self.get_nes(preds, ins_labels)
        gold_nes = self.get_nes(golds)
        for pred_ne, gold_ne in zip(pred_nes, gold_nes):
            self.gold += len(gold_ne)
            self.pred += len(pred_ne)
            self.tp += len(pred_ne & gold_ne)
            u_pre_set = set([(start, end) for start, end, label in pred_ne])
            u_gold_set = set([(start, end) for start, end, label in gold_ne])
            self.utp += len(u_pre_set & u_gold_set)
        return self

    def __repr__(self):
        return f"P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%} UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%}"

    @property
    def score(self):
        return self.f
    
    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)
    
    @property
    def up(self):
        return self.utp / (self.pred + self.eps)
    
    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)
    
    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)
    
class NoPosNERMetric(Metric):
    # without the position 
    def __init__(self, label_vocab_itos, eps=1e-12):
        super().__init__()

        self.label_vocab_itos = label_vocab_itos
        self.tp = 0.0
        self.gold = 0.0
        self.pred = 0.0
        self.eps = eps

    def get_nes(self, ne_tensor, text_lst):
        ne_lst = ne_tensor.tolist()
        str_ne_sets = []
        for nes, text in zip(ne_lst, text_lst):
            this_ne_set = set()
            this_str_ne_lst = []
            for ne in nes:
                this_str_ne_lst.append(self.label_vocab_itos[ne])
            
            start = 0
            ne_label = ''
            flag = False
            for i, ne_string in enumerate(this_str_ne_lst):
                if ne_string.startswith('B'):
                    if flag:
                        this_ne_set.add((text[start:i], ne_label))
                        flag = False
                    start = i
                    ne_label = ne_string[2:]
                    flag = True
                elif ne_string.startswith('I'):
                    continue
                else:
                    if flag:
                        this_ne_set.add((text[start:i], ne_label))
                        flag = False
            if flag:
                this_ne_set.add((text[start:i+1], ne_label))
                flag = False

            str_ne_sets.append(this_ne_set)

        return str_ne_sets
            
    def __call__(self, preds, golds, pred_txt_lst, gold_txt_lst):
        pred_nes = self.get_nes(preds, pred_txt_lst)
        gold_nes = self.get_nes(golds, gold_txt_lst)
        for pred_ne, gold_ne in zip(pred_nes, gold_nes):
            self.gold += len(gold_ne)
            self.pred += len(pred_ne)
            self.tp += len(pred_ne & gold_ne)
        return self

    def __repr__(self):
        return f"P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f
    
    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)

class LevenNERMetric(Metric):
    # position is geted by levenstein
    def __init__(self, label_vocab_itos, eps=1e-12):
        super().__init__()

        self.label_vocab_itos = label_vocab_itos
        self.utp = 0.0
        self.tp = 0.0
        self.gold = 0.0
        self.pred = 0.0
        self.action = 0.0
        self.sum_gold = 0.0
        self.recalled_ne_str = 0.0
        self.eps = eps
        self.goldNE_asrNonNE = []
        self.asrNE_goldNonNE = []

    def get_nes(self, ne_tensor, ins_labels=None):
        ne_lst = ne_tensor.tolist()
        str_ne_sets = []
        for nes in ne_lst:
            this_ne_set = set()
            this_str_ne_lst = []
            for ne in nes:
                this_str_ne_lst.append(self.label_vocab_itos[ne])
            
            start = 0
            ne_label = ''
            flag = False
            for i, ne_string in enumerate(this_str_ne_lst):
                if ne_string.startswith('B'):
                    if flag:
                        this_ne_set.add((start, i, ne_label))
                        flag = False
                    start = i
                    ne_label = ne_string[2:]
                    flag = True
                elif ne_string.startswith('I'):
                    continue
                else:
                    if flag:
                        this_ne_set.add((start, i, ne_label))
                        flag = False
            if flag:
                this_ne_set.add((start, i+1, ne_label))
                flag = False

            str_ne_sets.append(this_ne_set)
        
        if ins_labels is not None:
            res = []
            for ne_set, ins_label in zip(str_ne_sets, ins_labels):
                if ins_label != "导航":
                    res.append(ne_set)
                else:
                    this_set = set()
                    for ne in ne_set:
                        if ne[2] == "ORG":
                            this_set.add((ne[0], ne[1], "LOC"))
                        else:
                            this_set.add(ne)
                    res.append(this_set)
            return res
        return str_ne_sets
    
    def compute_ne_string_recall(self, pred_aligned_str, gold_ne_set, gold_position_map, gold_str, pad_char, gold_aligned_str):
        goldNE_asrNonNE = "" + pred_aligned_str
        asrNE_goldNonNE = "" + gold_aligned_str
        for gold_ne in gold_ne_set:
            start, end, label = gold_ne
            gold_ne_s = gold_str[start:end]
            n_start = gold_position_map[start]
            n_end = gold_position_map[end-1] + 1
            pred_ne_s = pred_aligned_str[n_start:n_end]
            pred_ne_s = pred_ne_s.replace(pad_char, '')
            if pred_ne_s == gold_ne_s:
                self.recalled_ne_str += 1

            goldNE_asrNonNE = goldNE_asrNonNE[:n_start] + gold_aligned_str[n_start:n_end] + goldNE_asrNonNE[n_end:]
            asrNE_goldNonNE = asrNE_goldNonNE[:n_start] + pred_aligned_str[n_start:n_end] + asrNE_goldNonNE[n_end:]
        self.goldNE_asrNonNE.append(goldNE_asrNonNE.replace(pad_char, ''))
        self.asrNE_goldNonNE.append(asrNE_goldNonNE.replace(pad_char, ''))
            
    def __call__(self, preds, golds, pred_txt_lst, gold_txt_lst, ins_labels=None):
        pred_nes = self.get_nes(preds, ins_labels)
        gold_nes = self.get_nes(golds)
        for pred_s, gold_s, pred_ne_set, gold_ne_set in zip(pred_txt_lst, gold_txt_lst, pred_nes, gold_nes):
            aligned_pred, aligned_gold, position_map_pred, position_map_gold, action, sum_gold = self.levenshtein_alignment(pred_s, gold_s, pad_char='淦')
            self.compute_ne_string_recall(aligned_pred, gold_ne_set, position_map_gold, gold_s, '淦', aligned_gold)
            self.action += action
            self.sum_gold += sum_gold
            self.pred += len(pred_ne_set)
            self.gold += len(gold_ne_set)
            pred_lst = [(pred_s[ne_t[0]: ne_t[1]], ne_t[2]) for ne_t in pred_ne_set]
            gold_lst = [(gold_s[ne_t[0]: ne_t[1]], ne_t[2]) for ne_t in gold_ne_set]
            
            pred_wolabel_lst = [pred_s[ne_t[0]: ne_t[1]] for ne_t in pred_ne_set]
            gold_wolabel_lst = [gold_s[ne_t[0]: ne_t[1]] for ne_t in gold_ne_set]
            for pred_ne in pred_wolabel_lst:
                if pred_ne in gold_wolabel_lst:
                    self.utp += 1
                    gold_wolabel_lst.remove(pred_ne)

            for pred_ne in pred_lst:
                if pred_ne in gold_lst:
                    self.tp += 1
                    gold_lst.remove(pred_ne)
        return self

    # have to think more about this
    # def __call__(self, preds, golds, pred_txt_lst, gold_txt_lst):
    #     pred_nes = self.get_nes(preds)
    #     gold_nes = self.get_nes(golds)
    #     for pred_s, gold_s, pred_ne_set, gold_ne_set in zip(pred_txt_lst, gold_txt_lst, pred_nes, gold_nes):
    #         aligned_pred, aligned_gold, position_map_pred, position_map_gold, action, sum_gold = self.levenshtein_alignment(pred_s, gold_s, pad_char='淦')
    #         self.action += action
    #         self.sum_gold += sum_gold
    #         leven_pred_ne_set = self.update_ne_set(pred_ne_set, position_map_pred, aligned_pred)
    #         leven_gold_ne_set = self.update_ne_set(gold_ne_set, position_map_gold, aligned_gold)
    #         if len(leven_pred_ne_set) > 0:
    #             import pdb; pdb.set_trace()
    #         self.pred += len(leven_pred_ne_set)
    #         self.gold += len(leven_gold_ne_set)
    #         self.tp += len(leven_pred_ne_set & leven_gold_ne_set)
    #     return self

    def update_ne_set(self, ne_set, position_map, aligned_txt):
        new_ne_set = set()
        for ne in ne_set:
            start, end, label = ne
            new_start = position_map[start]
            new_end = position_map[end-1]
            new_ne_set.add((new_start, new_end+1, aligned_txt[new_start:new_end+1], label))
        return new_ne_set

    def levenshtein_alignment(self, s1, s2, pad_char='淦'):
        # Step 1: Compute Levenshtein distance matrix
        rows = len(s1) + 1
        cols = len(s2) + 1
        dp = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(1, rows):
            dp[i][0] = i
        for j in range(1, cols):
            dp[0][j] = j

        for i in range(1, rows):
            for j in range(1, cols):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
        
        # # Step 2: Compute the CER
        # cer = dp[-1][-1] / len(s2)
        # # s2 should be the ground truth
        
        # Step 3: Trace back through the matrix to find the alignment
        aligned_s1 = ""
        aligned_s2 = ""
        i, j = len(s1), len(s2)
        position_map_1 = [0] * len(s1)
        position_map_2 = [0] * len(s2)
        s1_index = len(s1) - 1
        s2_index = len(s2) - 1

        while i > 0 and j > 0:
            if s1[i-1] == s2[j-1]:
                aligned_s1 = s1[i-1] + aligned_s1
                aligned_s2 = s2[j-1] + aligned_s2
                position_map_1[s1_index] = len(aligned_s1) - 1
                position_map_2[s2_index] = len(aligned_s2) - 1
                i, j = i-1, j-1
                s1_index -= 1
                s2_index -= 1
            elif dp[i][j] == dp[i-1][j-1] + 1:
                aligned_s1 = s1[i-1] + aligned_s1
                aligned_s2 = s2[j-1] + aligned_s2
                position_map_1[s1_index] = len(aligned_s1) - 1
                position_map_2[s2_index] = len(aligned_s2) - 1
                i, j = i-1, j-1
                s1_index -= 1
                s2_index -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                aligned_s1 = s1[i-1] + aligned_s1
                aligned_s2 = pad_char + aligned_s2
                position_map_1[s1_index] = len(aligned_s1) - 1
                i -= 1
                s1_index -= 1
            else:
                aligned_s1 = pad_char + aligned_s1
                aligned_s2 = s2[j-1] + aligned_s2
                position_map_2[s2_index] = len(aligned_s2) - 1
                j -= 1
                s2_index -= 1
        
        while i > 0:
            aligned_s1 = s1[i-1] + aligned_s1
            aligned_s2 = pad_char + aligned_s2
            position_map_1[s1_index] = len(aligned_s1) - 1
            i -= 1
            s1_index -= 1
        
        while j > 0:
            aligned_s1 = pad_char + aligned_s1
            aligned_s2 = s2[j-1] + aligned_s2
            position_map_2[s2_index] = len(aligned_s2) - 1
            j -= 1
            s2_index -= 1
    
        position_map_1 = [len(aligned_s1)-1-index for index in position_map_1]
        position_map_2 = [len(aligned_s2)-1-index for index in position_map_2]

        return aligned_s1, aligned_s2, position_map_1, position_map_2, dp[-1][-1], len(s2)

    def __repr__(self):
        with open('metric_out.txt', 'w', encoding="utf8") as f:
            for goldNE_asrNonNE, asrNE_goldNonNE in zip(self.goldNE_asrNonNE, self.asrNE_goldNonNE):
                f.write(f"{goldNE_asrNonNE}\t{asrNE_goldNonNE}\n")
        return f"CER: {self.cer:6.2%} NE-Str-R: {self.ne_str_recall:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%} UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%}"
        

    @property
    def score(self):
        return self.f
    
    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)
    
    @property
    def up(self):
        return self.utp / (self.pred + self.eps)
    
    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)
    
    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)
    
    @property
    def cer(self):
        return self.action / (self.sum_gold + self.eps)
    
    @property
    def ne_str_recall(self):
        return self.recalled_ne_str / (self.gold + self.eps)
    
class LevenMetric(Metric):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.action = 0.0
        self.sum_gold = 0.0
        self.eps = eps

    def levenshtein_alignment(self, s1, s2, pad_char='淦'):
        # Step 1: Compute Levenshtein distance matrix
        rows = len(s1) + 1
        cols = len(s2) + 1
        dp = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(1, rows):
            dp[i][0] = i
        for j in range(1, cols):
            dp[0][j] = j

        for i in range(1, rows):
            for j in range(1, cols):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j]) + 1
        
        # # Step 2: Compute the CER
        # cer = dp[-1][-1] / len(s2)
        # # s2 should be the ground truth
        
        # Step 3: Trace back through the matrix to find the alignment
        aligned_s1 = ""
        aligned_s2 = ""
        i, j = len(s1), len(s2)
        position_map_1 = [0] * len(s1)
        position_map_2 = [0] * len(s2)
        s1_index = len(s1) - 1
        s2_index = len(s2) - 1

        while i > 0 and j > 0:
            if s1[i-1] == s2[j-1]:
                aligned_s1 = s1[i-1] + aligned_s1
                aligned_s2 = s2[j-1] + aligned_s2
                position_map_1[s1_index] = len(aligned_s1) - 1
                position_map_2[s2_index] = len(aligned_s2) - 1
                i, j = i-1, j-1
                s1_index -= 1
                s2_index -= 1
            elif dp[i][j] == dp[i-1][j-1] + 1:
                aligned_s1 = s1[i-1] + aligned_s1
                aligned_s2 = s2[j-1] + aligned_s2
                position_map_1[s1_index] = len(aligned_s1) - 1
                position_map_2[s2_index] = len(aligned_s2) - 1
                i, j = i-1, j-1
                s1_index -= 1
                s2_index -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                aligned_s1 = s1[i-1] + aligned_s1
                aligned_s2 = pad_char + aligned_s2
                position_map_1[s1_index] = len(aligned_s1) - 1
                i -= 1
                s1_index -= 1
            else:
                aligned_s1 = pad_char + aligned_s1
                aligned_s2 = s2[j-1] + aligned_s2
                position_map_2[s2_index] = len(aligned_s2) - 1
                j -= 1
                s2_index -= 1
        
        while i > 0:
            aligned_s1 = s1[i-1] + aligned_s1
            aligned_s2 = pad_char + aligned_s2
            position_map_1[s1_index] = len(aligned_s1) - 1
            i -= 1
            s1_index -= 1
        
        while j > 0:
            aligned_s1 = pad_char + aligned_s1
            aligned_s2 = s2[j-1] + aligned_s2
            position_map_2[s2_index] = len(aligned_s2) - 1
            j -= 1
            s2_index -= 1
    
        position_map_1 = [len(aligned_s1)-1-index for index in position_map_1]
        position_map_2 = [len(aligned_s2)-1-index for index in position_map_2]

        return aligned_s1, aligned_s2, position_map_1, position_map_2, dp[-1][-1], len(s2)

    def __call__(self, pred_txt_lst, gold_txt_lst):
        for pred_s, gold_s in zip(pred_txt_lst, gold_txt_lst):
            aligned_pred, aligned_gold, position_map_pred, position_map_gold, action, sum_gold = self.levenshtein_alignment(pred_s, gold_s, pad_char='淦')
            self.action += action
            self.sum_gold += sum_gold
        return self
    
    @property
    def cer(self):
        return self.action / (self.sum_gold + self.eps)
    
    def __repr__(self):
        return f"CER: {self.cer:6.2%}"