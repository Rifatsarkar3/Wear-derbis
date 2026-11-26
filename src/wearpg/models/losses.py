"""Multi-task training objective of the proposed model (DESIGN sec. 7).

Every term masks non-finite predictions AND targets and is 0 (never NaN) when nothing is left.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter

from wearpg.batch import RunBatch
from wearpg.models.heads import coral_expected
from wearpg.models.physics import consistency_loss, monotone_severity_loss


def ordinal_coral_loss(logits: Tensor, target: Tensor) -> Tensor:
    """CORAL loss: BCE-with-logits summed over the K-1 tasks [y > k], mean over items with target >= 0 and finite
    logits. logits (n, K-1), target (n,) long (-1 ignored)."""
    logits = logits.float()
    m = (target >= 0) & torch.isfinite(logits).all(-1)
    levels = (target[m, None] > torch.arange(logits.shape[-1], device=logits.device)).float()
    return F.binary_cross_entropy_with_logits(logits[m], levels, reduction="sum") / m.sum().clamp_min(1)


def class_weights_from_targets(targets, n_classes: int) -> Tensor:
    """Balanced class weights w_c = n / (C_present n_c) from labelled targets (>= 0); 0 for absent classes.
    targets: 1-D ints (tensor, array or list). Returns (n_classes,) float32."""
    t = torch.as_tensor(targets).long().flatten()
    counts = torch.bincount(t[t >= 0], minlength=n_classes).float()
    present = counts > 0
    return torch.where(present, counts.sum() / (present.sum() * counts), 0.0)


def _huber(pred: Tensor, target: Tensor) -> Tensor:
    m = torch.isfinite(pred) & torch.isfinite(target)
    return F.huber_loss(pred[m], target[m].to(pred.dtype), reduction="sum") / m.sum().clamp_min(1)


def _mechanism_ce(logits: Tensor, target: Tensor, soft: Tensor | None, weight, smoothing: float) -> Tensor:
    """Class-weighted (label-smoothed) CE on labelled particles, normalised by the summed item weights like
    F.cross_entropy(reduction="mean"); soft rater-vote targets when `soft` is given (rows without votes skipped)."""
    logits = logits.float()
    w = logits.new_ones(logits.shape[1]) if weight is None else torch.as_tensor(weight).to(logits)
    m = (target >= 0) & torch.isfinite(logits).all(1)
    if soft is not None:
        m &= torch.isfinite(soft).all(1) & (soft.sum(1) > 0)
    t = target[m] if soft is None else soft[m].to(logits)
    loss = F.cross_entropy(logits[m], t, weight=w, label_smoothing=smoothing, reduction="none").sum()
    denom = w[t].sum() if soft is None else (t * w).sum()
    return torch.where(denom > 0, loss / denom.clamp_min(1e-12), 0.0)  # val batch of only unweighted classes -> 0


def compute_losses(out: dict[str, Tensor], batch: RunBatch, cfg: dict, class_weights=None
                   ) -> tuple[Tensor, dict[str, float]]:
    """total = sum_k cfg.loss[k] * term_k over mechanism CE, severity CORAL, rate_run / rate_sample / volume Huber,
    consistency, monotone and eta_offset_l2. Run-level terms (rate_run, consistency) only use runs with
    coverage >= physics.min_run_coverage. Returns (total, {term: value, ..., "total": value})."""
    lw, min_cov = cfg["loss"], float(cfg["physics"]["min_run_coverage"])
    s2r, n_r = batch.sample_to_run, batch.n_runs
    cov_ok = out["coverage_run"] >= min_cov
    n_obs = torch.bincount(s2r, minlength=n_r)
    sev_sum = scatter(coral_expected(out["severity_logits"]), s2r, 0, n_r, "sum")
    sev_run = torch.where(n_obs > 0, sev_sum / n_obs.clamp_min(1), float("nan"))
    terms = {
        "mechanism": _mechanism_ce(out["mech_logits"], batch.mech_target, batch.mech_soft, class_weights,
                                   float(cfg["train"]["label_smoothing"])),
        "severity": ordinal_coral_loss(out["severity_logits"], batch.severity_target),
        "rate_run": _huber(out["log10_rate_run"], torch.where(cov_ok, batch.log10_rate_run_target, float("nan"))),
        "rate_sample": _huber(out["log10_rate_sample"], batch.log10_rate_sample_target),
        "volume": _huber(out["log_volume"], batch.log_volume_target),
        "consistency": consistency_loss(out["log_debris_volume_run_mm3"], batch.log_grav_volume_mm3,
                                        out["coverage_run"], min_cov),
        "monotone": monotone_severity_loss(sev_run, batch.load_N, batch.mono_group, batch.condition_idx,
                                           float(lw["monotone_margin"])),
        "eta_offset_l2": out["eta_offset_l2"],
    }
    total = sum(float(lw[k]) * v for k, v in terms.items())
    values = torch.stack([v.detach().float() for v in (*terms.values(), total)]).tolist()  # one device sync
    return total, dict(zip([*terms, "total"], values))


mechanism_ce = _mechanism_ce  # public name: every particle classifier uses this exact CE (fairness)
