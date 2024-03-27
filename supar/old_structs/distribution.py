# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
from supar.structs.semiring import (CrossEntropySemiring, EntropySemiring,
                                    KLDivergenceSemiring, KMaxSemiring,
                                    LogSemiring, MaxSemiring, Semiring,
                                    VarianceSemiring)
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property


class StructuredDistribution(Distribution):

    def __init__(self, scores, mask, **kwargs):
        self.mask = mask
        self.kwargs = kwargs

        self.scores = scores.requires_grad_() if isinstance(scores, torch.Tensor) else [s.requires_grad_() for s in scores]

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @lazy_property
    def log_partition(self):
        return self.forward(LogSemiring)

    @lazy_property
    def marginals(self):
        return self.backward(self.log_partition.sum())

    @lazy_property
    def max(self):
        return self.forward(MaxSemiring)

    @lazy_property
    def argmax(self):
        return self.backward(self.max.sum())

    @lazy_property
    def mode(self):
        return self.argmax

    def kmax(self, k):
        return self.forward(KMaxSemiring(k))

    def topk(self, k):
        raise NotImplementedError

    @lazy_property
    def entropy(self):
        return self.forward(EntropySemiring)

    @lazy_property
    def variance(self):
        return self.forward(VarianceSemiring)

    @lazy_property
    def count(self):
        if isinstance(self.scores, torch.Tensor):
            scores = torch.ones_like(self.scores)
        else:
            scores = [torch.ones_like(i) for i in self.scores]
        return self.__class__(scores, self.mask, **self.kwargs).forward(Semiring).long()

    def cross_entropy(self, other):
        if isinstance(self.scores, torch.Tensor):
            scores = torch.stack((self.scores, other.scores))
        else:
            scores = [torch.stack((i, j)) for i, j in zip(self.scores, other.scores)]
        return self.__class__(scores, self.mask, **self.kwargs).forward(CrossEntropySemiring)

    def kl(self, other):
        if isinstance(self.scores, torch.Tensor):
            scores = torch.stack((self.scores, other.scores))
        else:
            scores = [torch.stack((i, j)) for i, j in zip(self.scores, other.scores)]
        return self.__class__(scores, self.mask, **self.kwargs).forward(KLDivergenceSemiring)

    def log_prob(self, value):
        return self.score(value) - self.log_partition

    def score(self, value):
        raise NotImplementedError

    @torch.enable_grad()
    def forward(self, semiring):
        raise NotImplementedError

    def backward(self, log_partition):
        return autograd.grad(log_partition,
                             self.scores if isinstance(self.scores, torch.Tensor) else self.scores[0],
                             retain_graph=True)[0]