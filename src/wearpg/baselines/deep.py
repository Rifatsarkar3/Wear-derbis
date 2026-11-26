"""Deep particle-classifier baselines (DESIGN sec. 9).

Every model maps a ParticleBatch to mechanism logits (n_p, C) and concatenates batch.particle_feats (the standardized
baseline globals, which carry absolute size: DESIGN sec. 0 rule 6) to its pooled embedding before the classifier.
FECNN-style (ref. [6]) and the SV-ERnet-style soft-vote ensemble "vit_ensemble" (ref. [7]) are re-implementations in
the spirit of the planning review's references. Verify them against the original articles before claiming
equivalence.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

from wearpg.batch import ModelDims, ParticleBatch
from wearpg.models.heads import mlp
from wearpg.models.image_branch import SmallCNN
from wearpg.models.losses import mechanism_ce
from wearpg.models.particle_gnn import ParticleGNN, pool_views

DEEP_BASELINES = ("particle_gnn", "contour_cnn", "pointnet", "fecnn", "resnet18", "convnext_tiny", "vit_ensemble")
VIT_ENSEMBLE = ("vit_tiny_patch16_224", "resnet18", "efficientnet_b0")


def _conv1d(c_in: int, c_out: int, k: int = 1, dilation: int = 1) -> nn.Sequential:
    """Conv1d (circular padding, stride 1: equivariant to cyclic shifts) -> GroupNorm -> ReLU."""
    return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=dilation * (k // 2), dilation=dilation,
                                   padding_mode="circular"), nn.GroupNorm(8, c_out), nn.ReLU())


class ContourCNN(nn.Module):
    """1-D contour CNN: 4 dilated (1, 2, 4, 8) circular Conv1d layers (k = 5, 64 ch) on contour_signal (n_p, S, N) ->
    global average || max over N -> || globals -> MLP. Stride-1 circular convolutions are equivariant to cyclic shifts
    and the global pools are invariant, so the logits do not depend on the contour start point. Needs N >= 16."""

    def __init__(self, dims: ModelDims, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.Sequential(_conv1d(dims.signal_channels, 64, 5, 1), *(_conv1d(64, 64, 5, d) for d in (2, 4, 8)))
        self.head = mlp(2 * 64 + dims.particle_feat_dim, hidden, dims.n_classes, dropout)

    def forward(self, batch: ParticleBatch) -> Tensor:
        h = self.convs(batch.contour_signal)
        return self.head(torch.cat([h.mean(-1), h.amax(-1), batch.particle_feats], -1))


class PointNetClassifier(nn.Module):
    """PointNet (Qi et al. 2017) without input transforms: shared per-point MLP (1x1 conv 2-64-128-256) on
    contour_points (n_p, N, 2) -> max pool over points (order invariant) -> || globals -> MLP. In training mode with
    `rotate` (data.augment.point_rotation) each particle's points are rotated by a uniform random angle about the
    canonical centroid (origin)."""

    def __init__(self, dims: ModelDims, hidden: int = 128, dropout: float = 0.1, rotate: bool = True):
        super().__init__()
        self.rotate = rotate
        self.point_mlp = nn.Sequential(_conv1d(2, 64), _conv1d(64, 128), _conv1d(128, 256))
        self.head = mlp(256 + dims.particle_feat_dim, hidden, dims.n_classes, dropout)

    def forward(self, batch: ParticleBatch) -> Tensor:
        p = batch.contour_points
        if self.training and self.rotate:
            a = torch.rand(p.shape[0], 1, device=p.device, dtype=p.dtype) * (2 * math.pi)
            c, s = a.cos(), a.sin()
            p = torch.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]], -1)
        h = self.point_mlp(p.transpose(1, 2)).amax(-1)
        return self.head(torch.cat([h, batch.particle_feats], -1))


class ParticleGNNClassifier(nn.Module):
    """models.ParticleGNN on batch.graphs (PyG Batch; the dataset supplies view-0 GEOMETRIC graphs: no texture columns,
    no interior grey) -> pool_views (mean || max over each particle's views) -> || globals -> MLP. The graph-value
    gate's proposed model ("particle_gnn"); dims.node_dim / edge_dim must be those of that geometric selection."""

    def __init__(self, dims: ModelDims, hidden: int = 128, layers: int = 4, conv: str = "gine", dropout: float = 0.1,
                 pooling=("mean", "max", "attention")):
        super().__init__()
        self.gnn = ParticleGNN(dims.node_dim, dims.edge_dim, hidden, layers, conv, dropout, tuple(pooling))
        self.head = mlp(2 * self.gnn.out_dim + dims.particle_feat_dim, hidden, dims.n_classes, dropout)

    def forward(self, batch: ParticleBatch) -> Tensor:
        g = batch.graphs
        views = self.gnn(g.x, g.edge_index, g.edge_attr, g.batch, g.num_graphs)
        return self.head(torch.cat([pool_views(views, batch.view_to_particle, batch.n_particles),
                                    batch.particle_feats], -1))


class FECNN(nn.Module):
    """FECNN-style (ref. [6]): CNN features of the crop (models SmallCNN on (n_p, 4, H, W)) fused with handcrafted
    globals (globals_kind "fecnn": shape_*, fractal_*, tex_*) -> MLP."""

    def __init__(self, dims: ModelDims, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cnn = SmallCNN(dims.crop_channels, hidden)
        self.head = mlp(hidden + dims.particle_feat_dim, hidden, dims.n_classes, dropout)

    def forward(self, batch: ParticleBatch) -> Tensor:
        return self.head(torch.cat([self.cnn(batch.crops), batch.particle_feats], -1))


class TimmImageClassifier(nn.Module):
    """timm.create_model(name, pretrained, in_chans=crop_channels, num_classes=0 [, img_size=crop_size for vit*])
    pooled features -> || globals -> Linear. timm is imported lazily."""

    def __init__(self, name: str, dims: ModelDims, pretrained: bool = False):
        super().__init__()
        import timm

        kw = {"img_size": dims.crop_size} if name.startswith("vit") else {}
        self.backbone = timm.create_model(name, pretrained=pretrained, in_chans=dims.crop_channels, num_classes=0,
                                          **kw)
        self.head = nn.Linear(self.backbone.num_features + dims.particle_feat_dim, dims.n_classes)

    def forward(self, batch: ParticleBatch) -> Tensor:
        return self.head(torch.cat([self.backbone(batch.crops), batch.particle_feats], -1))


class SoftVoteEnsemble(nn.Module):
    """Soft-vote ensemble: logits = ln mean_k softmax(member_k(batch)), so softmax(logits) is the mean member
    probability. train_particle_classifier trains the members independently, as in soft-voting ensembles."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, batch: ParticleBatch) -> Tensor:
        logp = torch.stack([m(batch).float().log_softmax(-1) for m in self.members])
        return torch.logsumexp(logp, 0) - math.log(len(self.members))


def make_deep_baseline(name: str, dims: ModelDims, cfg: dict) -> nn.Module:
    """Deep baseline by config name (DEEP_BASELINES): heads use model.hidden / model.dropout, particle_gnn the proposed
    model's GNN settings, timm backbones baselines.pretrained, pointnet data.augment.point_rotation. dims must
    describe the batches the model will see (globals of its globals_kind, geometric graph dims for particle_gnn)."""
    m = cfg["model"]
    h, p = int(m["hidden"]), float(m["dropout"])
    pretrained = bool(cfg["baselines"]["pretrained"])
    if name == "particle_gnn":
        return ParticleGNNClassifier(dims, h, int(m["gnn_layers"]), m["conv"], p, tuple(m["pooling"]))
    if name == "contour_cnn":
        return ContourCNN(dims, h, p)
    if name == "pointnet":
        return PointNetClassifier(dims, h, p, bool(cfg["data"]["augment"]["point_rotation"]))
    if name == "fecnn":
        return FECNN(dims, h, p)
    if name in ("resnet18", "convnext_tiny"):
        return TimmImageClassifier(name, dims, pretrained)
    if name == "vit_ensemble":
        return SoftVoteEnsemble([TimmImageClassifier(n, dims, pretrained) for n in VIT_ENSEMBLE])
    raise ValueError(f"unknown deep baseline {name!r}; expected one of {DEEP_BASELINES}")


@torch.no_grad()
def _mean_loss(model: nn.Module, loader, device, class_weights, smoothing: float) -> float:
    """Loss over the labelled particles of `loader` (batch losses weighted by labelled count); NaN if none."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader if loader is not None else ():
        batch = batch.to(device)
        k = int((batch.mech_target >= 0).sum())
        if k:
            loss = mechanism_ce(model(batch), batch.mech_target, batch.mech_soft, class_weights, smoothing)
            total, n = total + k * loss.item(), n + k
    return total / n if n else math.nan


def train_particle_classifier(model: nn.Module, train_loader, val_loader, cfg: dict, device,
                              class_weights=None) -> dict:
    """Train a deep particle classifier in place.

    AdamW (train.lr, train.weight_decay) for <= train.particle_epochs epochs with the proposed model's mechanism loss
    (class-weighted CE with train.label_smoothing; soft targets when batch.mech_soft is set; unlabelled ignored), grad
    clipping at train.grad_clip, early stopping on the validation loss (train.early_stopping_patience) and the best
    state restored. Without labelled validation particles (val_loader None or unlabelled) every epoch runs and the
    last state is kept. A SoftVoteEnsemble trains each member independently.
    Returns {"history": [{"epoch", "train_loss", "val_loss"}], "best_epoch", "best_val_loss"} (+ "members").
    """
    if isinstance(model, SoftVoteEnsemble):
        members = [train_particle_classifier(m, train_loader, val_loader, cfg, device, class_weights)
                   for m in model.members]
        val = _mean_loss(model.to(device), val_loader, device, class_weights, float(cfg["train"]["label_smoothing"]))
        return {"history": [], "best_epoch": -1, "best_val_loss": val, "members": members}
    t = cfg["train"]
    smoothing, patience = float(t["label_smoothing"]), int(t["early_stopping_patience"])
    has_val = val_loader is not None and any(bool((b.mech_target >= 0).any()) for b in val_loader)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    best_val, best_epoch, best_state, history = (math.inf if has_val else math.nan), -1, None, []
    for epoch in range(int(t["particle_epochs"])):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            loss = mechanism_ce(model(batch), batch.mech_target, batch.mech_soft, class_weights, smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip"]))
            opt.step()
            k = int((batch.mech_target >= 0).sum())
            total, n = total + k * loss.item(), n + k
        val = _mean_loss(model, val_loader, device, class_weights, smoothing) if has_val else math.nan
        history.append({"epoch": epoch, "train_loss": total / n if n else math.nan, "val_loss": val})
        if not has_val or val < best_val:  # a diverged (NaN) validation loss never counts as an improvement
            best_val, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val}


@torch.no_grad()
def predict_particle_proba(model: nn.Module, loader, device) -> tuple[list[str], np.ndarray]:
    """(particle ids in loader order, (n, C) softmax probabilities)."""
    model.to(device).eval()
    ids, probs = [], []
    for batch in loader:
        ids += list(batch.particle_ids)
        probs.append(model(batch.to(device)).float().softmax(-1).cpu().numpy())
    return ids, (np.concatenate(probs) if probs else np.zeros((0, 0)))
