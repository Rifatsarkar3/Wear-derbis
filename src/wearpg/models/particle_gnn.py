"""Message-passing encoder for (multi-view) particle contour/surface graphs (DESIGN sec. 7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, GINEConv, aggr
from torch_geometric.utils import scatter


class ParticleGNN(nn.Module):
    """Input projection -> `layers` x [GINEConv | GATv2Conv (edge features) + residual + LayerNorm + dropout] ->
    concatenated node readouts (`pooling` subset of mean, max, attention).

    Node-permutation invariant per graph; works with E = 0 (edge_attr must then have shape (0, edge_dim)).
    `out_dim = hidden * len(pooling)`. conv="gatv2" uses 4 heads of hidden // 4 channels (needs train.batch_runs <= 2
    for memory).
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 128, layers: int = 4, conv: str = "gine",
                 dropout: float = 0.1, pooling=("mean", "max", "attention")):
        super().__init__()
        if conv not in ("gine", "gatv2"):
            raise ValueError(f"conv must be gine|gatv2, got {conv!r}")
        if conv == "gatv2" and hidden % 4:
            raise ValueError(f"conv=gatv2 needs hidden divisible by 4 heads, got {hidden}")

        def make_conv():
            if conv == "gine":
                return GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                                train_eps=True, edge_dim=edge_dim)
            return GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=edge_dim)

        readouts = {"mean": aggr.MeanAggregation, "max": aggr.MaxAggregation,
                    "attention": lambda: aggr.AttentionalAggregation(nn.Linear(hidden, 1))}
        self.inp = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList(make_conv() for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.pools = nn.ModuleList(readouts[p]() for p in pooling)
        self.out_dim = hidden * len(self.pools)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor,
                num_graphs: int | None = None) -> Tensor:
        """x (N, node_dim), edge_index (2, E), edge_attr (E, edge_dim), batch (N,) graph index -> (n_graphs, out_dim)."""
        h = self.inp(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + self.dropout(F.relu(conv(h, edge_index, edge_attr))))
        n = int(batch.max()) + 1 if num_graphs is None else int(num_graphs)
        return torch.cat([pool(h, batch, dim_size=n) for pool in self.pools], dim=-1)


def pool_views(view_emb: Tensor, view_to_particle: Tensor, n_particles: int) -> Tensor:
    """Multi-view fusion: (n_v, D) view embeddings -> (n_p, 2D) = mean || max over each particle's views
    (a particle without views gets zeros)."""
    return torch.cat([scatter(view_emb, view_to_particle, 0, n_particles, "mean"),
                      scatter(view_emb, view_to_particle, 0, n_particles, "max")], dim=-1)
