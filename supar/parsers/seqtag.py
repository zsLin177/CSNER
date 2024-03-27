# -*- coding: utf-8 -*-

import os, json, string

import pdb
from re import X

import torch
import torch.nn as nn
from supar.models import (SimpleSeqTagModel, CrfSeqTagModel, HmmModel)
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk, eos
from supar.utils.field import Field, SubwordField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.logging import init_logger, logger
from supar.utils.metric import SeqTagMetric, NERMetric
from supar.utils.transform import CoNLL
from datetime import datetime

logger = get_logger(__name__)


class HmmSeqTagParser(Parser):
    r"""
    TODO:introduction
    """
    NAME = 'HmmSeqTag'
    MODEL = HmmModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.LABEL = self.transform.POS

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              update_steps=1,
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
        """

        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        
        logger.info("Loading the data")
        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev)
        test = Dataset(self.transform, args.test)
        train.build(args.batch_size//args.update_steps, args.buckets)
        dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        start = datetime.now()
        self._train(train.loader)
        dev_metric = self._evaluate(dev.loader)
        logger.info(f"dev: {dev_metric}")
        test_metric = self._evaluate(test.loader)
        logger.info(f"test: {test_metric}")
        t = datetime.now() - start
        logger.info(f"{t}s elapsed \n")
        self.save(args.path)
        
    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), SeqTagMetric(self.LABEL.vocab)

        for i, batch in enumerate(bar, 1):
            sentences_lst = [s.words for s in batch.sentences]
            words, labels = batch
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            self.model(words, labels)
        self.model.smooth_logp()

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SeqTagMetric(self.LABEL.vocab)

        for batch in loader:
            sentences_lst = [s.words for s in batch.sentences]
            words, labels = batch
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            preds = self.model.decode(words)
            metric(preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
        # total_loss /= len(loader)

        return metric

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        total_loss, metric = 0, SeqTagMetric(self.LABEL.vocab)
        preds = {'labels': [], 'probs': [] if self.args.prob else None}
        # for words, *feats, labels in loader:
        for batch in loader:
            # pdb.set_trace()
            sentences_lst = [s.words for s in batch.sentences]
            words, labels = batch
            word_mask = words.ne(self.args.pad_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            output = self.model.decode(words)
            metric(output.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
            lens = mask.sum(-1).tolist()
            preds['labels'].extend(out[0:i].tolist() for i, out in zip(lens, output))
        preds['labels'] = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None] for i in out]) for out in preds['labels']]
        # total_loss /= len(loader)
        print(metric)
        return preds

    @classmethod
    def build(cls, path, min_freq=3, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        WORD = Field('words', pad=pad, unk=unk, lower=True)
        LABEL = Field('labels')
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=LABEL)

        train = Dataset(transform, args.train)

        WORD.build(
            train, args.min_freq)
        LABEL.build(train)
        args.update({
            'n_words':
            WORD.vocab.n_init,
            'n_labels':
            len(LABEL.vocab),
            'pad_index':
            WORD.pad_index,
            'unk_index':
            WORD.unk_index,
        })

        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)

class SimpleSeqTagParser(Parser):
    r"""
    TODO:introduction
    """
    NAME = 'SimpleSeqTag'
    MODEL = SimpleSeqTagModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.LABEL = self.transform.POS
        

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              update_steps=1,
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def filter(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().filter(**Config().update(locals()))

    def _train(self, loader, epoch=None):
        self.model.train()
        if self.aux_model is not None:
            self.aux_model.train()

        bar, metric = progress_bar(loader), SeqTagMetric(self.LABEL.vocab)
        masked_word_num = 0
        sum_word = 0

        for i, batch in enumerate(bar, 1):
            sentences_lst = [s.words for s in batch.sentences]
            labels_lst = [s.labels for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            sum_word += mask.sum().item()
            if self.args.policy_grad:
                score, self_uncer = self.model(words, sentences_lst, feats[:])
                if self.aux_model is not None:
                    q = self.aux_model.multi_forward(words, sentences_lst, feats, times=self.args.times, if_T=False, req_grad=False)
                    skl, if_false_mask = self.aux_model.avg_metric(q, p=torch.cat((torch.zeros_like(labels[:,0]).unsqueeze(-1), labels), -1), threshold=0.3)
                    loss = self.model.loss(score, labels, mask, 1-skl)
                elif epoch <= self.args.pg_start_epoch:
                    loss = self.model.loss(score, labels, mask)
                else:
                    q = self.model.multi_forward(words, sentences_lst, feats, times=self.args.times, if_T=False, req_grad=True)
                    skl, if_false_mask = self.model.avg_metric(q, p=torch.cat((torch.zeros_like(labels[:,0]).unsqueeze(-1), labels), -1), threshold=0.3, req_grad=False)
                    loss = self.model.loss(score, labels, mask, 1-skl)
            else:
                score, self_uncer = self.model(words, sentences_lst, feats)
                loss = self.model.loss(score, labels, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            preds = self.model.decode(score, self_uncer)[:, 1:]
            mask = mask[:, 1:]
            metric(preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )
        logger.info(f"{bar.postfix}")


    def constrain(self, sentences_lst, labels_lst, if_false_mask):
        min_freq = 3
        ignore_label_set = {'NR'}
        mask = torch.zeros_like(if_false_mask).bool()
        for i in range(len(sentences_lst)):
            word_lst = sentences_lst[i]
            label_lst = labels_lst[i]
            for j in range(len(word_lst)):
                if self.WORD.counter[word_lst[j]] >= min_freq:
                    mask[i, j+1] = True
                elif label_lst[j] not in ignore_label_set:
                    mask[i, j+1] = True
        return mask

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SeqTagMetric(self.LABEL.vocab)

        for batch in loader:
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score, self_uncer = self.model(words, sentences_lst, feats)
            preds = self.model.decode(score, self_uncer)[:, 1:]
            mask = mask[:, 1:]
            metric(preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
        # total_loss /= len(loader)

        return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        total_loss, metric = 0, SeqTagMetric(self.LABEL.vocab)
        preds = {'labels': [], 'probs': [] if self.args.prob else None}
        # for words, *feats, labels in loader:
        for batch in loader:
            # pdb.set_trace()
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score, self_uncer = self.model(words, sentences_lst, feats)
            output = self.model.decode(score, self_uncer)[:, 1:]
            mask = mask[:, 1:]
            metric(output.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
            lens = mask.sum(-1).tolist()
            preds['labels'].extend(out[0:i].tolist() for i, out in zip(lens, output))
        preds['labels'] = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None] for i in out]) for out in preds['labels']]
        # total_loss /= len(loader)
        print(metric)
        return preds

    @torch.no_grad()
    def _filter(self, loader, out_file='filter.jsonlines'):
        # dropout open
        self.model.train()
        # self.model.eval()
        
        sum_word = .0
        sum_picked_word = .0
        picked_data = []

        bar = progress_bar(loader)
        total_loss, metric = 0, SeqTagMetric(self.LABEL.vocab)
        for batch in bar:
            sentences_lst = [s.words for s in batch.sentences]
            label_str_lst = [s.labels for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score, self_uncer = self.model(words, sentences_lst, feats[:])
            preds = self.model.decode(score, self_uncer)[:, 1:]
            n_mask = mask[:, 1:]
            metric(preds.masked_fill(~n_mask, -1), labels.masked_fill(~n_mask, -1))

            q = self.model.multi_forward(words, sentences_lst, feats, times=10, if_T=False)

            # [batch_size, seq_len]
            # use std metric
            # with "gold"
            # skl, if_false_mask = self.model.var_metric(q, p=torch.cat((torch.zeros_like(labels[:,0]).unsqueeze(-1), labels), -1), varhold=0.375)
            # without "gold"
            # skl, if_false_mask = self.model.var_metric(q, varhold=0.317)

            # use mean metric
            # with "gold"
            skl, if_false_mask = self.model.avg_metric(q, p=torch.cat((torch.zeros_like(labels[:,0]).unsqueeze(-1), labels), -1), threshold=0.25)
            # without "gold"
            # skl, if_false_mask = self.model.avg_metric(q, threshold=0.65)

            # use vote metric
            # use "gold"
            # skl, if_false_mask = self.model.vote_metric(q, p=torch.cat((torch.zeros_like(labels[:,0]).unsqueeze(-1), labels), -1), vote_low_rate=0.3, vote_up_rate=0.5)
            # without "gold"
            # skl, if_false_mask = self.model.vote_metric(q, vote_low_rate=0.3)

            # use self uncer
            # skl, if_false_mask = self.model.su_metric(self_uncer, threshold=0.71)


            if_false_mask = if_false_mask & mask

            # sum_picked_word += if_false_mask.sum().item()
            sum_word += mask.sum().item()
            # [k]
            b_idx, s_idx = if_false_mask.nonzero()[:, 0].tolist(), if_false_mask.nonzero()[:, 1].tolist()
            skl_value = skl[if_false_mask].tolist()
            prd_seq = preds[if_false_mask[:, 1:]].tolist()

            min_freq = 3
            ignore_label_set = {'NR'}
            for i in range(len(b_idx)):
                this_dict = {'sentence': sentences_lst[b_idx[i]], 'label_seq': label_str_lst[b_idx[i]],
                            'wrong_idx': s_idx[i]-1, 'wrong_word': sentences_lst[b_idx[i]][s_idx[i]-1],
                            'annotated_label': label_str_lst[b_idx[i]][s_idx[i]-1], 'predicted_label': self.LABEL.vocab.itos[prd_seq[i]],
                            'sort_key_value': skl_value[i]}
                wrong_word = this_dict['wrong_word']
                annotated_label = this_dict['annotated_label']
                if(self.WORD.counter[wrong_word] >= min_freq):
                    picked_data.append(this_dict)
                elif(annotated_label not in ignore_label_set):
                    picked_data.append(this_dict)
        
        sum_picked_word = len(picked_data)
        picked_ratio = sum_picked_word / sum_word
        print(f'picked words: {sum_picked_word}, sum words: {sum_word}, ratio: {picked_ratio:6.2%}')
        print(metric)

        picked_data.sort(key=lambda x:x['sort_key_value'], reverse=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            for this_dict in picked_data:
                f.write(json.dumps(this_dict, ensure_ascii=False)+'\n')
        return metric


    @classmethod
    def build(cls, path, min_freq=3, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        if args.aux:
            aux_parser = cls.load(args.aux_path)

        logger.info("Building the fields")
        
        if args.aux:
            TAG, CHAR, LEMMA, BERT = None, None, None, None
            WORD, CHAR, BERT = aux_parser.transform.FORM
            LEMMA = aux_parser.transform.LEMMA
            LABEL = aux_parser.transform.POS
            transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=LABEL)
        else:
            TAG, CHAR, LEMMA, BERT = None, None, None, None
            if args.encoder == 'bert':
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                        GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                WORD = SubwordField(
                    'words',
                    pad=t.pad_token,
                    unk=t.unk_token,
                    bos=t.bos_token or t.cls_token,
                    tokenize=t.tokenize,
                    fn=None if not isinstance(t,
                                            (GPT2Tokenizer, GPT2TokenizerFast))
                    else lambda x: ' ' + x)
                WORD.vocab = t.get_vocab()
            else:
                WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
                if 'tag' in args.feat:
                    TAG = Field('tags', bos=bos)
                if 'char' in args.feat:
                    CHAR = SubwordField('chars',
                                        pad=pad,
                                        unk=unk,
                                        bos=bos,
                                        fix_len=args.fix_len)
                if 'lemma' in args.feat:
                    LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
                if 'bert' in args.feat:
                    from transformers import (AutoTokenizer, GPT2Tokenizer,
                                            GPT2TokenizerFast)
                    t = AutoTokenizer.from_pretrained(args.bert)
                    BERT = SubwordField(
                        'bert',
                        pad=t.pad_token,
                        unk=t.unk_token,
                        bos=t.bos_token or t.cls_token,
                        fix_len=args.fix_len,
                        tokenize=t.tokenize,
                        fn=None
                        if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast))
                        else lambda x: ' ' + x)
                    BERT.vocab = t.get_vocab()
            # LABEL = ChartField('labels', fn=CoNLL.get_labels)
            # word:[bos, seq, eos] spans:[bos, seq, eos]
            LABEL = Field('labels')
            transform = CoNLL(FORM=(WORD, CHAR, BERT),
                            LEMMA=LEMMA,
                            POS=LABEL)

            train = Dataset(transform, args.train)
            if args.encoder != 'bert':
                WORD.build(
                    train, args.min_freq,
                    (Embedding.load(args.embed, args.unk) if args.embed else None))
                if TAG is not None:
                    TAG.build(train)
                if CHAR is not None:
                    CHAR.build(train)
                if LEMMA is not None:
                    LEMMA.build(train)
            # LABEL.build(train)
            LABEL.build(train)
        
        args.update({
            'n_words':
            len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_labels':
            len(LABEL.vocab),
            'n_tags':
            len(TAG.vocab) if TAG is not None else None,
            'n_chars':
            len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index':
            CHAR.pad_index if CHAR is not None else None,
            'n_lemmas':
            len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index':
            BERT.pad_index if BERT is not None else None,
            'pad_index':
            WORD.pad_index,
            'bos_index':
            WORD.bos_index,
            'unk_index':
            WORD.unk_index,
            'interpolation': args.itp,
            'self_uncertain': args.self_uncer,
            'policy_grad': args.policy_grad
        })

        if(args.encoder == 'bert'):
            args.update({
            'lr':
            5e-5,
            'epochs': 20, 
            'warmup':
            0.1
        })
        elif args.encoder == 'lstm':
            args.update({
            'lr':
            1e-3,
            'epochs': 5000, 
            'mu': .0,
            'nu': .95,
            'eps': 1e-12,
            'weight_decay': 3e-9,
            'decay': .75,
            'decay_steps': 5000
        })
        elif args.encoder == 'transformer':
            args.update({
            'lr':
            0.04,
            'epochs': 5000, 
            'warmsteps':2000
        })

        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(
            WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        aux_model = None
        if args.aux:
            logger.info("Show the auxiliary model")
            aux_model = aux_parser.model
            logger.info(f"{aux_model}\n")

        return cls(args, model, transform, aux_model)


class CrfSeqTagParser(Parser):
    r"""
    TODO:introduction
    """
    NAME = 'CrfSeqTag'
    MODEL = CrfSeqTagModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.LABEL = self.transform.POS

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              update_steps=1,
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader, epoch):
        self.model.train()

        bar, metric = progress_bar(loader), NERMetric(self.LABEL.vocab.itos)

        for i, batch in enumerate(bar, 1):
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score = self.model(words, sentences_lst, feats)
            loss = self.model.loss(score, labels, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            preds = self.model.decode(score, mask)
            mask = mask[:, 1:]
            metric(preds.masked_fill(~mask, self.LABEL.vocab.stoi['O']), labels.masked_fill(~mask, self.LABEL.vocab.stoi['O']))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )
        logger.info(f"{bar.postfix}")


    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, NERMetric(self.LABEL.vocab)

        # for words, *feats, labels in loader:
        for batch in loader:
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score = self.model(words, sentences_lst, feats)
            preds = self.model.decode(score, mask)
            mask = mask[:, 1:]
            metric(preds.masked_fill(~mask, self.LABEL.vocab.stoi['O']), labels.masked_fill(~mask, self.LABEL.vocab.stoi['O']))
        # total_loss /= len(loader)

        return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        total_loss, metric = 0, NERMetric(self.LABEL.vocab)
        preds = {'labels': [], 'probs': [] if self.args.prob else None}
        bar = progress_bar(loader)
        for i, batch in enumerate(bar, 1):
            # pdb.set_trace()
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score = self.model(words, sentences_lst, feats)
            output = self.model.decode(score, mask)
            mask = mask[:, 1:]
            metric(output.masked_fill(~mask, self.LABEL.vocab.stoi['O']), labels.masked_fill(~mask, self.LABEL.vocab.stoi['O']))
            lens = mask.sum(-1).tolist()
            preds['labels'].extend(out[0:i].tolist() for i, out in zip(lens, output))
        preds['labels'] = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None] for i in out]) for out in preds['labels']]
        # total_loss /= len(loader)
        print(metric)
        return preds

    @classmethod
    def build(cls, path, min_freq=3, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        if args.encoder == 'bert':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField(
                'words',
                pad=t.pad_token,
                unk=t.unk_token,
                bos=t.bos_token or t.cls_token,
                tokenize=t.tokenize,
                fn=None if not isinstance(t,
                                          (GPT2Tokenizer, GPT2TokenizerFast))
                else lambda x: ' ' + x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=bos)
            if 'char' in args.feat:
                CHAR = SubwordField('chars',
                                    pad=pad,
                                    unk=unk,
                                    bos=bos,
                                    fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField(
                    'bert',
                    pad=t.pad_token,
                    unk=t.unk_token,
                    bos=t.bos_token or t.cls_token,
                    fix_len=args.fix_len,
                    tokenize=t.tokenize,
                    fn=None
                    if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast))
                    else lambda x: ' ' + x)
                BERT.vocab = t.get_vocab()
        # LABEL = ChartField('labels', fn=CoNLL.get_labels)
        # word:[bos, seq, eos] spans:[bos, seq, eos]
        LABEL = Field('labels', pad='O')
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=LABEL)

        train = Dataset(transform, args.train)
        if args.encoder != 'bert':
            WORD.build(
                train, args.min_freq,
                (Embedding.load(args.embed, args.unk) if args.embed else None))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
            if LEMMA is not None:
                LEMMA.build(train)
        LABEL.build(train)
        args.update({
            'n_words':
            len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_labels':
            len(LABEL.vocab),
            'n_tags':
            len(TAG.vocab) if TAG is not None else None,
            'n_chars':
            len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index':
            CHAR.pad_index if CHAR is not None else None,
            'n_lemmas':
            len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index':
            BERT.pad_index if BERT is not None else None,
            'pad_index':
            WORD.pad_index,
            'bos_index':
            WORD.bos_index,
            'unk_index':
            WORD.unk_index,
            'interpolation': args.itp
        })

        if(args.encoder == 'bert'):
            args.update({
            'lr':
            5e-5,
            'epochs': 20, 
            'warmup':
            0.1
        })
        elif args.encoder == 'lstm':
            args.update({
            'lr':
            1e-3,
            'epochs': 5000, 
            'mu': .0,
            'nu': .95,
            'eps': 1e-12,
            'weight_decay': 3e-9,
            'decay': .75,
            'decay_steps': 5000
        })
        elif args.encoder == 'transformer':
            args.update({
            'lr':
            0.04,
            'epochs': 5000, 
            'warmsteps':2000
        })

        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(
            WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)