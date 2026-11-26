"""Permutation-invariant bag pooling of particle embeddings into oil-sample embeddings (DESIGN sec. 7).

Every aggregator: forward(x (n, D), index (n,) bag id, n_bags) -> (pooled (n_bags, D), weights (n,)), where weights
are each item's share of its bag (sum to 1 per non-empty bag). Empty bags pool to zeros.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.nn.aggr import SetTransformerAggregation
from torch_geometric.utils import scatter, softmax


def _uniform_weights(x: Tensor, index: Tensor, n_bags: int) -> Tensor:
    return x.new_ones(index.numel()) / torch.bincount(index, minlength=n_bags)[index]


class GatedAttentionPool(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018): a_i = softmax_bag(w^T (tanh(V x_i) * sigmoid(U x_i))),
    pooled_b = sum_{i in b} a_i x_i; weights = a."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.V, self.U, self.w = nn.Linear(in_dim, hidden), nn.Linear(in_dim, hidden), nn.Linear(hidden, 1)

    def forward(self, x: Tensor, index: Tensor, n_bags: int) -> tuple[Tensor, Tensor]:
        score = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
        a = softmax(score, index, num_nodes=n_bags)
        return scatter(a[:, None] * x, index, 0, n_bags, "sum"), a


class MeanPool(nn.Module):
    """Bag mean; weights = 1 / bag size."""

    def forward(self, x: Tensor, index: Tensor, n_bags: int) -> tuple[Tensor, Tensor]:
        return scatter(x, index, 0, n_bags, "mean"), _uniform_weights(x, index, n_bags)


class SetTransformerPool(nn.Module):
    """PyG SetTransformerAggregation (1 seed point, gcd(D, 4) heads); attention is internal, so weights are uniform."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.st = SetTransformerAggregation(in_dim, num_seed_points=1, heads=math.gcd(in_dim, 4))

    def forward(self, x: Tensor, index: Tensor, n_bags: int) -> tuple[Tensor, Tensor]:
        perm = torch.argsort(index, stable=True)  # PyG to_dense_batch requires a sorted index
        return self.st(x[perm], index[perm], dim_size=n_bags), _uniform_weights(x, index, n_bags)


def make_aggregator(name: str, in_dim: int, hidden: int) -> nn.Module:
    """name: gated_attention | mean | set_transformer (`hidden` = gate width of gated_attention)."""
    if name == "gated_attention":
        return GatedAttentionPool(in_dim, hidden)
    if name == "mean":
        return MeanPool()
    if name == "set_transformer":
        return SetTransformerPool(in_dim)
    raise ValueError(f"unknown aggregator {name!r} (gated_attention | mean | set_transformer)")
