# -*- coding: utf-8 -*-

import os
from collections.abc import Iterable

import nltk
from supar.utils.logging import get_logger, progress_bar
from supar.utils.tokenizer import Tokenizer

logger = get_logger(__name__)


# class Transform(object):
#     r"""
#     A Transform object corresponds to a specific data format.
#     It holds several instances of data fields that provide instructions for preprocessing and numericalizing, etc.

#     Attributes:
#         training (bool):
#             Sets the object in training mode.
#             If ``False``, some data fields not required for predictions won't be returned.
#             Default: ``True``.
#     """

#     fields = []

#     def __init__(self):
#         self.training = True

#     def __repr__(self):
#         s = '\n'
#         for i, field in enumerate(self):
#             if not isinstance(field, Iterable):
#                 field = [field]
#             for f in field:
#                 if f is not None:
#                     s += f"  {f}\n"
#         return f"{self.__class__.__name__}({s})"

#     def __call__(self, sentences):
#         pairs = dict()
#         for field in self:
#             if field not in self.src and field not in self.tgt:
#                 continue
#             if not self.training and field in self.tgt:
#                 continue
#             if not isinstance(field, Iterable):
#                 field = [field]
#             for f in field:
#                 if f is not None:
#                     pairs[f] = f.transform([getattr(i, f.name) for i in sentences])

#         return pairs

#     def __getitem__(self, index):
#         return getattr(self, self.fields[index])

#     def train(self, training=True):
#         self.training = training

#     def eval(self):
#         self.train(False)

#     def append(self, field):
#         self.fields.append(field.name)
#         setattr(self, field.name, field)

#     @property
#     def src(self):
#         raise AttributeError

#     @property
#     def tgt(self):
#         raise AttributeError

#     def save(self, path, sentences):
#         with open(path, 'w') as f:
#             f.write('\n'.join([str(i) for i in sentences]) + '\n')


class Transform(object):
    r"""
    A Transform object corresponds to a specific data format.
    It holds several instances of data fields that provide instructions for preprocessing and numericalizing, etc.
    Attributes:
        training (bool):
            Sets the object in training mode.
            If ``False``, some data fields not required for predictions won't be returned.
            Default: ``True``.
    """

    fields = []

    def __init__(self):
        self.training = True

    def __len__(self):
        return len(self.fields)

    def __repr__(self):
        s = '\n' + '\n'.join([f" {f}" for f in self.flattened_fields]) + '\n'
        return f"{self.__class__.__name__}({s})"

    def __call__(self, sentences):
        # numericalize the fields of each sentence
        for sentence in progress_bar(sentences):
            for f in self.flattened_fields:
                sentence.transformed[f.name] = f.transform([getattr(sentence, f.name)])[0]
        return self.flattened_fields

    def __getitem__(self, index):
        return getattr(self, self.fields[index])

    @property
    def flattened_fields(self):
        flattened = []
        for field in self:
            if field not in self.src and field not in self.tgt:
                continue
            if not self.training and field in self.tgt:
                continue
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None:
                    flattened.append(f)
        return flattened

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(False)

    def append(self, field):
        self.fields.append(field.name)
        setattr(self, field.name, field)

    @property
    def src(self):
        raise AttributeError

    @property
    def tgt(self):
        raise AttributeError

    def save(self, path, sentences):
        with open(path, 'w') as f:
            f.write('\n'.join([str(i) for i in sentences]) + '\n')


class Batch(object):

    def __init__(self, sentences):
        self.sentences = sentences
        self.transformed = {f.name: f.compose([s.transformed[f.name] for s in sentences])
                            for f in sentences[0].transform.flattened_fields}
        self.fields = list(self.transformed.keys())

    def __repr__(self):
        s = ', '.join([f"{name}" for name in self.fields])
        return f"{self.__class__.__name__}({s})"

    def __getitem__(self, index):
        return self.transformed[self.fields[index]]

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if name in self.transformed:
            return self.transformed[name]
        if hasattr(self.sentences[0], name):
            return [getattr(s, name) for s in self.sentences]
        raise AttributeError


# class Sentence(object):
#     r"""
#     A Sentence object holds a sentence with regard to specific data format.
#     """

#     def __init__(self, transform):
#         self.transform = transform

#         # mapping from each nested field to their proper position
#         self.maps = dict()
#         # names of each field
#         self.keys = set()
#         # values of each position
#         self.values = []
#         for i, field in enumerate(self.transform):
#             if not isinstance(field, Iterable):
#                 field = [field]
#             for f in field:
#                 if f is not None:
#                     self.maps[f.name] = i
#                     self.keys.add(f.name)

#     def __len__(self):
#         return len(self.values[0])

#     def __contains__(self, key):
#         return key in self.keys

#     def __getattr__(self, name):
#         if name in self.__dict__:
#             return self.__dict__[name]
#         elif name in self.maps:
#             return self.values[self.maps[name]]
#         else:
#             raise AttributeError

#     def __setattr__(self, name, value):
#         if 'keys' in self.__dict__ and name in self:
#             index = self.maps[name]
#             if index >= len(self.values):
#                 self.__dict__[name] = value
#             else:
#                 self.values[index] = value
#         else:
#             self.__dict__[name] = value

#     def __getstate__(self):
#         return vars(self)

#     def __setstate__(self, state):
#         self.__dict__.update(state)


class CoNLL(Transform):
    r"""
    The CoNLL object holds ten fields required for CoNLL-X data format (:cite:`buchholz-marsi-2006-conll`).
    Each field can be binded with one or more :class:`Field` objects. For example,
    ``FORM`` can contain both :class:`Field` and :class:`SubwordField` to produce tensors for words and subwords.

    Attributes:
        ID:
            Token counter, starting at 1.
        FORM:
            Words in the sentence.
        LEMMA:
            Lemmas or stems (depending on the particular treebank) of words, or underscores if not available.
        CPOS:
            Coarse-grained part-of-speech tags, where the tagset depends on the treebank.
        POS:
            Fine-grained part-of-speech tags, where the tagset depends on the treebank.
        FEATS:
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or underscores if not available.
        HEAD:
            Heads of the tokens, which are either values of ID or zeros.
        DEPREL:
            Dependency relations to the HEAD.
        PHEAD:
            Projective heads of tokens, which are either values of ID or zeros, or underscores if not available.
        PDEPREL:
            Dependency relations to the PHEAD, or underscores if not available.
    """

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self,
                 ID=None, FORM=None, LEMMA=None, CPOS=None, POS=None,
                 FEATS=None, HEAD=None, DEPREL=None, PHEAD=None, PDEPREL=None):
        super().__init__()

        self.ID = ID
        self.FORM = FORM
        self.LEMMA = LEMMA
        self.CPOS = CPOS
        self.POS = POS
        self.FEATS = FEATS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.PHEAD = PHEAD
        self.PDEPREL = PDEPREL

    @property
    def src(self):
        return self.FORM, self.LEMMA, self.CPOS, self.POS, self.FEATS

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL, self.PHEAD, self.PDEPREL

    @classmethod
    def get_arcs(cls, sequence, placeholder='_'):
        return [-1 if i == placeholder else int(i) for i in sequence]

    @classmethod
    def get_sibs(cls, sequence, placeholder='_'):
        sibs = [[0] * (len(sequence) + 1) for _ in range(len(sequence) + 1)]
        heads = [0] + [-1 if i == placeholder else int(i) for i in sequence]

        for i, hi in enumerate(heads[1:], 1):
            for j, hj in enumerate(heads[i+1:], i + 1):
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i][hi] = j
                    else:
                        sibs[j][hj] = i
                    break
        return sibs[1:]

    @classmethod
    def get_edges(cls, sequence):
        edges = [[0]*(len(sequence)+1) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edges[i][int(pair.split(':')[0])] = 1
        return edges

    @classmethod
    def get_labels(cls, sequence):
        labels = [[None]*(len(sequence)+1) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edge, label = pair.split(':')
                    labels[i][int(edge)] = label
        return labels

    @classmethod
    def get_span_labels(cls, sequence):
        # add bos and eos
        labels = [[None]*(len(sequence)+2) for _ in range(len(sequence)+2)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edge, label = pair.split(':')
                    if(edge != '0'):
                        labels[i][int(edge)] = label[2:]
        return labels

    @classmethod
    def get_spans(cls, sequence):
        #return [seq_len, seq_len, seq_len]
        # seq_len contain root and eos
        # results = [[[None]*(len(sequence)+2) for i in range(len(sequence)+2)] for j in range(len(sequence)+2)]
        results = [None] * (len(sequence)+2)

        prd_map = {}  # 1:33, 2:44
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edge, label = pair.split(':')
                    if(edge == '0' and label == '[prd]'):
                        prd_map[len(prd_map)+1] = i
                        break
        
        re_prd_map = {}  # 33:1, 44:2
        for key, value in prd_map.items():
            re_prd_map[value] = key
        
        arc_values = []
        for i, s in enumerate(sequence, 1):
            if(s == '_'):
                arc_value = [[] for j in range(len(prd_map))]
                arc_values.append(arc_value)
            else:
                relas = s.split('|')
                arc_value = [[] for j in range(len(prd_map))]
                for rela in relas:
                    head, rel = rela.split(':')
                    head_idx = int(head)
                    if(head_idx in re_prd_map):
                        arc_value[re_prd_map[head_idx]-1].append(rel)
                arc_values.append(arc_value)
        
        for key, value in prd_map.items():
            this_prd_arc = [
                word_arc_lsts[key - 1] for word_arc_lsts in arc_values
            ]
            prd_idx = value
            results[value] = cls.get_span_one_sentence(this_prd_arc, prd_idx)
        for i in range(len(results)):
            if(results[i] == None):
                results[i] = [[None] * (len(sequence)+2) for _ in range(len(sequence)+2)]
        return results

    @classmethod
    def get_span_one_sentence(cls, relas, prd_idx):
        span = [[None] * (len(relas)+2) for _ in range(len(relas)+2)] 
        i = 0
        while (i < len(relas)):
            rel = relas[i]
            # print(i)
            # print(relas)
            if ((i + 1) == prd_idx):
                # 其实谓词不影响
                i += 1
            elif(rel == ['[prd]']):
                i += 1
            elif(rel == ['Other']):
                i += 1
            elif (len(rel) == 0):
                i += 1
            else:
                s_rel = rel[0]
                position_tag = s_rel[0]
                label = s_rel[2:]  # label直接按第一个边界的label
                if (position_tag in ('B', 'I')):
                    span_start = i
                    span_end = -1
                    i += 1
                    while (i < len(relas)):
                        if (len(relas[i]) == 0 or relas[i] == ['Other']):
                            i += 1
                            continue
                        else:
                            if (relas[i][0][0] == 'B'):
                                break
                            else:
                                span_end = i
                                # label2 = relas[i][0][2:]  # 以后面那个作为label
                                i += 1
                                break
                    if (span_end != -1):
                        span[span_start+1][span_end+1] = label
                    else:
                        span[span_start+1][span_start+1] = label
        return span

    @classmethod
    def build_relations(cls, chart):
        sequence = ['_'] * len(chart)
        for i, row in enumerate(chart):
            pairs = [(j, label) for j, label in enumerate(row) if label is not None]
            if len(pairs) > 0:
                sequence[i] = '|'.join(f"{label}" for head, label in pairs)
        return sequence

    @classmethod
    def build_relas_from_spans(cls, chart):
        seq_len = len(chart)  # the real len of seq, not contain bos, pad, eos
        sequence = ['_'] * seq_len
        for idx, p_chart in enumerate(chart, 1):
            flag = 0  # if it is a predicate
            for head in range(1, seq_len+1):
                for tail in range(1, seq_len+1):
                    if(p_chart[head-1][tail-1] is not None):
                        label = p_chart[head-1][tail-1]
                        if(sequence[head-1] == '_'):
                            sequence[head-1] = str(idx)+':'+'B-'+label
                        else:
                            sequence[head-1] += '|'+str(idx)+':'+'B-'+label
                        if(head != tail):
                            if(sequence[tail-1] == '_'):
                                sequence[tail-1] = str(idx)+':'+'I-'+label
                            else:
                                sequence[tail-1] += '|'+str(idx)+':'+'I-'+label
                        flag = 1
            if(flag == 1):
                if(sequence[idx-1] == '_'):
                    sequence[idx-1] = str(0)+':'+'[prd]'
                else:
                    sequence[idx-1] += '|'+str(0)+':'+'[prd]'
        return sequence

    @classmethod
    def toconll(cls, tokens):
        r"""
        Converts a list of tokens to a string in CoNLL-X format.
        Missing fields are filled with underscores.

        Args:
            tokens (list[str] or list[tuple]):
                This can be either a list of words, word/pos pairs or word/lemma/pos triples.

        Returns:
            A string in CoNLL-X format.

        Examples:
            >>> print(CoNLL.toconll(['She', 'enjoys', 'playing', 'tennis', '.']))
            1       She     _       _       _       _       _       _       _       _
            2       enjoys  _       _       _       _       _       _       _       _
            3       playing _       _       _       _       _       _       _       _
            4       tennis  _       _       _       _       _       _       _       _
            5       .       _       _       _       _       _       _       _       _

            >>> print(CoNLL.toconll([('She', 'PRP'), ('enjoys', 'VBZ'), ('playing', 'VBG'), ('tennis', 'NN'), ('.', '.')]))
            1       She     _       PRP     _       _       _       _       _       _
            2       enjoys  _       VBZ     _       _       _       _       _       _
            3       playing _       VBG     _       _       _       _       _       _
            4       tennis  _       NN      _       _       _       _       _       _
            5       .       _       .       _       _       _       _       _       _

        """

        if isinstance(tokens[0], str):
            s = '\n'.join([f"{i}\t{word}\t" + '\t'.join(['_']*8)
                           for i, word in enumerate(tokens, 1)])
        elif len(tokens[0]) == 2:
            s = '\n'.join([f"{i}\t{word}\t_\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, tag) in enumerate(tokens, 1)])
        elif len(tokens[0]) == 3:
            s = '\n'.join([f"{i}\t{word}\t{lemma}\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, lemma, tag) in enumerate(tokens, 1)])
        else:
            raise RuntimeError(f"Invalid sequence {tokens}. Only list of str or list of word/pos/lemma tuples are support.")
        return s + '\n'

    @classmethod
    def isprojective(cls, sequence):
        r"""
        Checks if a dependency tree is projective.
        This also works for partial annotation.

        Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
        which are hard to detect in the scenario of partial annotation.

        Args:
            sequence (list[int]):
                A list of head indices.

        Returns:
            ``True`` if the tree is projective, ``False`` otherwise.

        Examples:
            >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
            False
            >>> CoNLL.isprojective([3, -1, 2])
            False
        """

        pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
        for i, (hi, di) in enumerate(pairs):
            for hj, dj in pairs[i+1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    return False
                if lj <= hi <= rj and hj == di:
                    return False
                if (li < lj < ri or li < rj < ri) and (li - lj)*(ri - rj) > 0:
                    return False
        return True

    @classmethod
    def istree(cls, sequence, proj=False, multiroot=False):
        r"""
        Checks if the arcs form an valid dependency tree.

        Args:
            sequence (list[int]):
                A list of head indices.
            proj (bool):
                If ``True``, requires the tree to be projective. Default: ``False``.
            multiroot (bool):
                If ``False``, requires the tree to contain only a single root. Default: ``True``.

        Returns:
            ``True`` if the arcs form an valid tree, ``False`` otherwise.

        Examples:
            >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
            True
            >>> CoNLL.istree([3, 0, 0, 3], proj=True)
            False
        """

        from supar.utils.alg import tarjan
        if proj and not cls.isprojective(sequence):
            return False
        n_roots = sum(head == 0 for head in sequence)
        if n_roots == 0:
            return False
        if not multiroot and n_roots > 1:
            return False
        if any(i == head for i, head in enumerate(sequence, 1)):
            return False
        return next(tarjan(sequence), None) is None

    def load(self, data, lang=None, proj=False, max_len=None, **kwargs):
        r"""
        Loads the data in CoNLL-X format.
        Also supports for loading data from CoNLL-U file with comments and non-integer IDs.

        Args:
            data (list[list] or str):
                A list of instances or a filename.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: None.
            proj (bool):
                If ``True``, discards all non-projective sentences. Default: ``False``.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`CoNLLSentence` instances.
        """

        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]
        else:
            if lang is not None:
                tokenizer = Tokenizer(lang)
                data = [tokenizer(i) for i in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = '\n'.join([self.toconll(i) for i in data]).split('\n')

        i, start, sentences = 0, 0, []
        for line in progress_bar(lines):
            if not line:
                sentences.append(CoNLLSentence(self, lines[start:i]))
                start = i + 1
            i += 1
        if proj:
            sentences = [i for i in sentences if self.isprojective(list(map(int, i.arcs)))]
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences


# class CoNLLSentence(Sentence):
#     r"""
#     Sencence in CoNLL-X format.

#     Args:
#         transform (CoNLL):
#             A :class:`CoNLL` object.
#         lines (list[str]):
#             A list of strings composing a sentence in CoNLL-X format.
#             Comments and non-integer IDs are permitted.

#     Examples:
#         >>> lines = ['# text = But I found the location wonderful and the neighbors very kind.',
#                      '1\tBut\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '2\tI\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '3\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '4\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '7\tand\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '8\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '10\tvery\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '11\tkind\t_\t_\t_\t_\t_\t_\t_\t_',
#                      '12\t.\t_\t_\t_\t_\t_\t_\t_\t_']
#         >>> sentence = CoNLLSentence(transform, lines)  # fields in transform are built from ptb.
#         >>> sentence.arcs = [3, 3, 0, 5, 6, 3, 6, 9, 11, 11, 6, 3]
#         >>> sentence.rels = ['cc', 'nsubj', 'root', 'det', 'nsubj', 'xcomp',
#                              'cc', 'det', 'dep', 'advmod', 'conj', 'punct']
#         >>> sentence
#         # text = But I found the location wonderful and the neighbors very kind.
#         1       But     _       _       _       _       3       cc      _       _
#         2       I       _       _       _       _       3       nsubj   _       _
#         3       found   _       _       _       _       0       root    _       _
#         4       the     _       _       _       _       5       det     _       _
#         5       location        _       _       _       _       6       nsubj   _       _
#         6       wonderful       _       _       _       _       3       xcomp   _       _
#         7       and     _       _       _       _       6       cc      _       _
#         7.1     found   _       _       _       _       _       _       _       _
#         8       the     _       _       _       _       9       det     _       _
#         9       neighbors       _       _       _       _       11      dep     _       _
#         10      very    _       _       _       _       11      advmod  _       _
#         11      kind    _       _       _       _       6       conj    _       _
#         12      .       _       _       _       _       3       punct   _       _
#     """

#     def __init__(self, transform, lines):
#         super().__init__(transform)

#         self.values = []
#         # record annotations for post-recovery
#         self.annotations = dict()

#         for i, line in enumerate(lines):
#             value = line.split('\t')
#             if value[0].startswith('#') or not value[0].isdigit():
#                 self.annotations[-i-1] = line
#             else:
#                 self.annotations[len(self.values)] = line
#                 self.values.append(value)
#         self.values = list(zip(*self.values))

#     def __repr__(self):
#         # cover the raw lines
#         merged = {**self.annotations,
#                   **{i: '\t'.join(map(str, line))
#                      for i, line in enumerate(zip(*self.values))}}
#         return '\n'.join(merged.values()) + '\n'


class Tree(Transform):
    r"""
    The Tree object factorize a constituency tree into four fields, each associated with one or more :class:`Field` objects.

    Attributes:
        WORD:
            Words in the sentence.
        POS:
            Part-of-speech tags, or underscores if not available.
        TREE:
            The raw constituency tree in :class:`nltk.tree.Tree` format.
        CHART:
            The factorized sequence of binarized tree traversed in pre-order.
    """

    root = ''
    fields = ['WORD', 'POS', 'TREE', 'CHART']

    def __init__(self, WORD=None, POS=None, TREE=None, CHART=None):
        super().__init__()

        self.WORD = WORD
        self.POS = POS
        self.TREE = TREE
        self.CHART = CHART

    @property
    def src(self):
        return self.WORD, self.POS, self.TREE

    @property
    def tgt(self):
        return self.CHART,

    @classmethod
    def totree(cls, tokens, root='', special_tokens={'(': '-LRB-', ')': '-RRB-'}):
        r"""
        Converts a list of tokens to a :class:`nltk.tree.Tree`.
        Missing fields are filled with underscores.

        Args:
            tokens (list[str] or list[tuple]):
                This can be either a list of words or word/pos pairs.
            root (str):
                The root label of the tree. Default: ''.
            special_tokens (dict):
                A dict for normalizing some special tokens to avoid tree construction crash.
                Default: {'(': '-LRB-', ')': '-RRB-'}.

        Returns:
            A :class:`nltk.tree.Tree` object.

        Examples:
            >>> print(Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], 'TOP'))
            (TOP ( (_ She)) ( (_ enjoys)) ( (_ playing)) ( (_ tennis)) ( (_ .)))
        """

        if isinstance(tokens[0], str):
            tokens = [(token, '_') for token in tokens]
        mapped = []
        for i, (word, pos) in enumerate(tokens):
            if word in special_tokens:
                tokens[i] = (special_tokens[word], pos)
                mapped.append((i, word))
        tree = nltk.Tree.fromstring(f"({root} {' '.join([f'( ({pos} {word}))' for word, pos in tokens])})")
        for i, word in mapped:
            tree[i][0][0] = word
        return tree

    @classmethod
    def binarize(cls, tree):
        r"""
        Conducts binarization over the tree.

        First, the tree is transformed to satisfy `Chomsky Normal Form (CNF)`_.
        Here we call :meth:`~nltk.tree.Tree.chomsky_normal_form` to conduct left-binarization.
        Second, all unary productions in the tree are collapsed.

        Args:
            tree (nltk.tree.Tree):
                The tree to be binarized.

        Returns:
            The binarized tree.

        Examples:
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ She))
                                                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                                (_ .)))
                                            ''')
            >>> print(Tree.binarize(tree))
            (TOP
              (S
                (S|<>
                  (NP (_ She))
                  (VP
                    (VP|<> (_ enjoys))
                    (S::VP (VP|<> (_ playing)) (NP (_ tennis)))))
                (S|<> (_ .))))

        .. _Chomsky Normal Form (CNF):
            https://en.wikipedia.org/wiki/Chomsky_normal_form
        """

        tree = tree.copy(True)
        if len(tree) == 1 and not isinstance(tree[0][0], nltk.Tree):
            tree[0] = nltk.Tree(f"{tree.label()}|<>", [tree[0]])
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, nltk.Tree):
                nodes.extend([child for child in node])
                if len(node) > 1:
                    for i, child in enumerate(node):
                        if not isinstance(child[0], nltk.Tree):
                            node[i] = nltk.Tree(f"{node.label()}|<>", [child])
        tree.chomsky_normal_form('left', 0, 0)
        tree.collapse_unary(joinChar='::')

        return tree

    @classmethod
    def factorize(cls, tree, delete_labels=None, equal_labels=None):
        r"""
        Factorizes the tree into a sequence.
        The tree is traversed in pre-order.

        Args:
            tree (nltk.tree.Tree):
                The tree to be factorized.
            delete_labels (set[str]):
                A set of labels to be ignored. This is used for evaluation.
                If it is a pre-terminal label, delete the word along with the brackets.
                If it is a non-terminal label, just delete the brackets (don't delete childrens).
                In `EVALB`_, the default set is:
                {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
                Default: ``None``.
            equal_labels (dict[str, str]):
                The key-val pairs in the dict are considered equivalent (non-directional). This is used for evaluation.
                The default dict defined in `EVALB`_ is: {'ADVP': 'PRT'}
                Default: ``None``.

        Returns:
            The sequence of the factorized tree.

        Examples:
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ She))
                                                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                                                (_ .)))
                                            ''')
            >>> Tree.factorize(tree)
            [(0, 5, 'TOP'), (0, 5, 'S'), (0, 1, 'NP'), (1, 4, 'VP'), (2, 4, 'S'), (2, 4, 'VP'), (3, 4, 'NP')]
            >>> Tree.factorize(tree, delete_labels={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''})
            [(0, 5, 'S'), (0, 1, 'NP'), (1, 4, 'VP'), (2, 4, 'S'), (2, 4, 'VP'), (3, 4, 'NP')]

        .. _EVALB:
            https://nlp.cs.nyu.edu/evalb/
        """

        def track(tree, i):
            label = tree.label()
            if delete_labels is not None and label in delete_labels:
                label = None
            if equal_labels is not None:
                label = equal_labels.get(label, label)
            if len(tree) == 1 and not isinstance(tree[0], nltk.Tree):
                return (i+1 if label is not None else i), []
            j, spans = i, []
            for child in tree:
                j, s = track(child, j)
                spans += s
            if label is not None and j > i:
                spans = [(i, j, label)] + spans
            return j, spans
        return track(tree, 0)[1]

    @classmethod
    def build(cls, tree, sequence):
        r"""
        Builds a constituency tree from the sequence. The sequence is generated in pre-order.
        During building the tree, the sequence is de-binarized to the original format (i.e.,
        the suffixes ``|<>`` are ignored, the collapsed labels are recovered).

        Args:
            tree (nltk.tree.Tree):
                An empty tree that provides a base for building a result tree.
            sequence (list[tuple]):
                A list of tuples used for generating a tree.
                Each tuple consits of the indices of left/right boundaries and label of the constituent.

        Returns:
            A result constituency tree.

        Examples:
            >>> tree = Tree.totree(['She', 'enjoys', 'playing', 'tennis', '.'], 'TOP')
            >>> sequence = [(0, 5, 'S'), (0, 4, 'S|<>'), (0, 1, 'NP'), (1, 4, 'VP'), (1, 2, 'VP|<>'),
                            (2, 4, 'S::VP'), (2, 3, 'VP|<>'), (3, 4, 'NP'), (4, 5, 'S|<>')]
            >>> print(Tree.build(tree, sequence))
            (TOP
              (S
                (NP (_ She))
                (VP (_ enjoys) (S (VP (_ playing) (NP (_ tennis)))))
                (_ .)))
        """

        root = tree.label()
        leaves = [subtree for subtree in tree.subtrees()
                  if not isinstance(subtree[0], nltk.Tree)]

        def track(node):
            i, j, label = next(node)
            if j == i+1:
                children = [leaves[i]]
            else:
                children = track(node) + track(node)
            if label is None or label.endswith('|<>'):
                return children
            labels = label.split('::')
            tree = nltk.Tree(labels[-1], children)
            for label in reversed(labels[:-1]):
                tree = nltk.Tree(label, [tree])
            return [tree]
        return nltk.Tree(root, track(iter(sequence)))

    def load(self, data, lang=None, max_len=None, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                A list of instances or a filename.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: None.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`TreeSentence` instances.
        """
        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                trees = [nltk.Tree.fromstring(s) for s in f]
            self.root = trees[0].label()
        else:
            if lang is not None:
                tokenizer = Tokenizer(lang)
                data = [tokenizer(i) for i in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            trees = [self.totree(i, self.root) for i in data]

        i, sentences = 0, []
        for tree in progress_bar(trees):
            sentences.append(TreeSentence(self, tree))
            i += 1
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences

class Sentence(object):

    def __init__(self, transform):
        self.transform = transform

        # mapping from each nested field to their proper position
        self.maps = dict()
        # names of each field
        self.keys = set()
        for i, field in enumerate(self.transform):
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None:
                    self.maps[f.name] = i
                    self.keys.add(f.name)
        # original values and numericalized values of each position
        self.values = []
        self.transformed = {key: None for key in self.keys}

    def __contains__(self, key):
        return key in self.keys

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.maps:
            return self.values[self.maps[name]]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if 'keys' in self.__dict__ and name in self:
            index = self.maps[name]
            if index >= len(self.values):
                self.__dict__[name] = value
            else:
                self.values[index] = value
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)


class CoNLLSentence(Sentence):
    r"""
    Sencence in CoNLL-X format.
    Args:
        transform (CoNLL):
            A :class:`~supar.utils.transform.CoNLL` object.
        lines (list[str]):
            A list of strings composing a sentence in CoNLL-X format.
            Comments and non-integer IDs are permitted.
    Examples:
        >>> lines = ['# text = But I found the location wonderful and the neighbors very kind.',
                     '1\tBut\t_\t_\t_\t_\t_\t_\t_\t_',
                     '2\tI\t_\t_\t_\t_\t_\t_\t_\t_',
                     '3\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '4\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_',
                     '6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7\tand\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '8\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_',
                     '10\tvery\t_\t_\t_\t_\t_\t_\t_\t_',
                     '11\tkind\t_\t_\t_\t_\t_\t_\t_\t_',
                     '12\t.\t_\t_\t_\t_\t_\t_\t_\t_']
        >>> sentence = CoNLLSentence(transform, lines)  # fields in transform are built from ptb.
        >>> sentence.arcs = [3, 3, 0, 5, 6, 3, 6, 9, 11, 11, 6, 3]
        >>> sentence.rels = ['cc', 'nsubj', 'root', 'det', 'nsubj', 'xcomp',
                             'cc', 'det', 'dep', 'advmod', 'conj', 'punct']
        >>> sentence
        # text = But I found the location wonderful and the neighbors very kind.
        1       But     _       _       _       _       3       cc      _       _
        2       I       _       _       _       _       3       nsubj   _       _
        3       found   _       _       _       _       0       root    _       _
        4       the     _       _       _       _       5       det     _       _
        5       location        _       _       _       _       6       nsubj   _       _
        6       wonderful       _       _       _       _       3       xcomp   _       _
        7       and     _       _       _       _       6       cc      _       _
        7.1     found   _       _       _       _       _       _       _       _
        8       the     _       _       _       _       9       det     _       _
        9       neighbors       _       _       _       _       11      dep     _       _
        10      very    _       _       _       _       11      advmod  _       _
        11      kind    _       _       _       _       6       conj    _       _
        12      .       _       _       _       _       3       punct   _       _
    """

    def __init__(self, transform, lines):
        super().__init__(transform)

        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i-1] = line
            else:
                self.annotations[len(self.values)] = line
                self.values.append(value)
        self.values = list(zip(*self.values))

    def __repr__(self):
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values))}}
        return '\n'.join(merged.values()) + '\n'


class TreeSentence(Sentence):
    r"""
    Args:
        transform (Tree):
            A :class:`Tree` object.
        tree (nltk.tree.Tree):
            A :class:`nltk.tree.Tree` object.
    """

    def __init__(self, transform, tree):
        super().__init__(transform)

        words, tags = zip(*tree.pos())
        chart = [[None]*(len(words)+1) for _ in range(len(words)+1)]
        for i, j, label in Tree.factorize(Tree.binarize(tree)[0]):
            chart[i][j] = label
        self.values = [words, tags, tree, chart]

    def __repr__(self):
        return self.values[-2].pformat(1000000)


