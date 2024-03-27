# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import (CharLSTM, IndependentDropout, SharedDropout,
                           TransformerEmbedding, VariationalLSTM, SelfAttentionEncoder)
from supar.utils import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb

class Model(nn.Module):

    def __init__(self,
                 n_words,
                 encoder='bert',
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat=['char'],
                 n_embed=100,
                 n_embed_proj=100,
                 n_pretrained_embed=300,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=100,
                 char_pad_index=0,
                 char_dropout=0,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 bert_req_grad=False,
                 embed_dropout=.33,
                 n_lstm_hidden=300,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())

        if self.args.encoder != 'bert':
            self.word_embed = nn.Embedding(num_embeddings=self.args.n_words,
                                           embedding_dim=self.args.n_embed)
            self.embed_proj = nn.Linear(n_pretrained_embed, n_pretrained_embed+12)
            self.args.n_input = self.args.n_embed + n_pretrained_embed + 12
            if 'tag' in self.args.feat:
                self.tag_embed = nn.Embedding(num_embeddings=self.args.n_tags,
                                              embedding_dim=self.args.n_feat_embed)
                self.args.n_input += self.args.n_feat_embed
            if 'char' in self.args.feat:
                self.char_embed = CharLSTM(n_chars=self.args.n_chars,
                                           n_embed=self.args.n_char_embed,
                                           n_hidden=self.args.n_char_hidden,
                                           n_out=self.args.n_feat_embed,
                                           pad_index=self.args.char_pad_index,
                                           dropout=self.args.char_dropout)
                self.args.n_input += self.args.n_feat_embed
            if 'lemma' in self.args.feat:
                self.lemma_embed = nn.Embedding(num_embeddings=self.args.n_lemmas,
                                                embedding_dim=self.args.n_feat_embed)
                self.args.n_input += self.args.n_feat_embed
            if 'bert' in self.args.feat:
                self.bert_embed = TransformerEmbedding(model=self.args.bert,
                                                       n_layers=self.args.n_bert_layers,
                                                       n_out=self.args.n_feat_embed,
                                                       pooling=bert_pooling,
                                                       pad_index=self.args.bert_pad_index,
                                                       dropout=self.args.mix_dropout,
                                                       requires_grad=(not self.args.freeze))
                self.args.n_input += self.bert_embed.n_out
            self.embed_dropout = IndependentDropout(p=self.args.embed_dropout)
        if self.args.encoder == 'lstm':
            self.encoder = VariationalLSTM(input_size=self.args.n_input,
                                           hidden_size=self.args.n_lstm_hidden,
                                           num_layers=self.args.n_lstm_layers,
                                           bidirectional=True,
                                           dropout=self.args.encoder_dropout)
            self.encoder_dropout = SharedDropout(p=self.args.encoder_dropout)
            self.args.n_hidden = self.args.n_lstm_hidden * 2
        elif self.args.encoder == 'transformer':
            self.encoder = SelfAttentionEncoder(num_encoder_layers=self.args.transformer_layers, emb_size=self.args.n_input, dim_feedforward=800, num_heads=8, position_embed=True, dropout=self.args.encoder_dropout)
            self.encoder_dropout = SharedDropout(p=self.args.encoder_dropout)
            self.args.n_hidden = self.args.n_input
        elif self.args.encoder == 'bert':
            self.encoder = TransformerEmbedding(model=self.args.bert,
                                                n_layers=self.args.n_bert_layers,
                                                pooling=bert_pooling,
                                                pad_index=self.args.pad_index,
                                                dropout=self.args.mix_dropout,
                                                requires_grad=self.args.bert_req_grad)
            self.encoder_dropout = nn.Dropout(p=self.args.encoder_dropout)
            self.args.n_hidden = self.encoder.n_out

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed.to(self.args.device))
        return self

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def embed(self, words, feats):
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed = torch.cat(
                (word_embed, self.embed_proj(self.pretrained(words))), -1)

        feat_embeds = []
        if 'tag' in self.args.feat:
            feat_embeds.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embeds.append(self.char_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embeds.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embeds.append(self.lemma_embed(feats.pop(0)))
        word_embed, feat_embed = self.embed_dropout(word_embed, torch.cat(feat_embeds, -1))
        # concatenate the word and feat representations
        embed = torch.cat((word_embed, feat_embed), -1)

        return embed

    def encode(self, words, feats=None):
        if self.args.encoder == 'lstm':
            x = pack_padded_sequence(self.embed(words, feats), words.ne(self.args.pad_index).sum(1).tolist(), True, False)
            x, _ = self.encoder(x)
            x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        elif self.args.encoder == 'transformer':
            embed = self.embed(words, feats)
            mask = words.ne(self.args.pad_index)
            x = self.encoder(embed, ~mask)
        else:
            x = self.encoder(words)
        return self.encoder_dropout(x)

    def decode(self):
        raise NotImplementedError
