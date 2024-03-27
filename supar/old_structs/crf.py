# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.structs.distribution import StructuredDistribution
from supar.structs.semiring import LogSemiring
from supar.utils.alg import mst
from supar.utils.fn import stripe
from torch.distributions.utils import lazy_property


class CRFLinearChain(StructuredDistribution):

    def __init__(self, scores, mask, trans):
        super().__init__(scores, mask, trans=trans)

        self.mask = mask
        self.lens = mask.sum(-1)
        self.trans = trans

    def __repr__(self):
        return f"{self.__class__.__name__}(n_tags={self.scores.shape[-1]})"

    def score(self, value):
        scores, mask, value = self.scores.transpose(0, 1), self.mask.t(), value.t()
        prev, succ = torch.cat((torch.full_like(value[:1], -1), value[:-1]), 0), value

        # [seq_len, batch_size]
        alpha = scores.gather(-1, value.unsqueeze(-1)).squeeze(-1)
        # [batch_size]
        alpha = LogSemiring.prod(LogSemiring.one_mask(LogSemiring.mul(alpha, self.trans[prev, succ]), ~mask), 0)
        alpha = alpha + self.trans[value.gather(0, self.lens.unsqueeze(0) - 1).squeeze(0), torch.full_like(value[0], -1)]

        return alpha

    def forward(self, semiring):
        scores, mask = self.scores.transpose(0, 1), self.mask.t()
        seq_len, _ = mask.shape

        # [batch_size, n_tags]
        alpha = semiring.mul(self.trans[-1, :-1], scores[0])
        for i in range(1, seq_len):
            # [batch_size, n_tags]
            alpha[mask[i]] = semiring.sum(semiring.times(self.trans[:-1, :-1].unsqueeze(0),
                                                         scores[i].unsqueeze(1),
                                                         alpha.unsqueeze(2)), 1)[mask[i]]
        alpha = semiring.mul(alpha, self.trans[:-1, -1])
        alpha = semiring.sum(alpha)

        return alpha
