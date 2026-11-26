"""Prediction heads: CORAL ordinal head for severity and a 2-layer MLP (DESIGN sec. 7)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Linear -> ReLU -> Dropout -> Linear."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))


class CoralHead(nn.Module):
    """CORAL (Cao et al. 2020): one shared linear score + K-1 biases -> logits (n, K-1), logit_k = log-odds P(y > k).
    Biases start descending (rank-consistent), as in the reference implementation."""

    def __init__(self, in_dim: int, n_levels: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.arange(n_levels - 1, 0, -1).float() / (n_levels - 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x) + self.bias


def _exceed(logits: Tensor) -> Tensor:
    # P(y > k), forced non-increasing in k so class probabilities are never negative
    return torch.sigmoid(logits.float()).cummin(dim=-1).values


def coral_probs(logits: Tensor) -> Tensor:
    """(n, K-1) CORAL logits -> (n, K) class probabilities (non-negative, rows sum to 1)."""
    s = _exceed(logits)
    one, zero = s.new_ones(s.shape[:-1] + (1,)), s.new_zeros(s.shape[:-1] + (1,))
    return torch.cat([one, s], -1) - torch.cat([s, zero], -1)


def coral_expected(logits: Tensor) -> Tensor:
    """(n, K-1) CORAL logits -> (n,) expected ordinal level E[y] = sum_k P(y > k)."""
    return _exceed(logits).sum(-1)
