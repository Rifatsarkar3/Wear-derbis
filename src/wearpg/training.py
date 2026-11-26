"""[I1] Training loop and run-bag prediction of the proposed model (DESIGN sec. 11).

fit_model: AdamW + per-epoch cosine schedule, gradient clipping, optional bf16 autocast for the forward pass (the losses
are computed outside autocast in float32), early stopping on the validation total loss with the best weights restored.
predict_runs: value columns of the particle / sample / run prediction tables (schema.*_PRED_COLUMNS); keys, ids and
targets are attached by experiments.attach_hierarchy.
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from wearpg.models.heads import coral_probs
from wearpg.models.losses import compute_losses
from wearpg.schema import SEVERITY_PROB_COLUMNS, prob_columns
from wearpg.utils import get_logger, save_json

PARTICLE_VALUE_COLUMNS = ["particle_id", "y_pred", *prob_columns(), "log_volume_pred"]
SAMPLE_VALUE_COLUMNS = ["sample_id", "severity_pred", *SEVERITY_PROB_COLUMNS, "log10_rate_pred", "log_debris_volume_mm3"]
RUN_VALUE_COLUMNS = ["run_id", "log10_rate_pred", "log_debris_volume_mm3", "coverage"]


def _autocast(device: torch.device, amp: str):
    """bf16 autocast context for train.amp == "bf16", a disabled one for "none"."""
    if amp not in ("none", "bf16"):
        raise ValueError(f"train.amp must be none | bf16, got {amp!r}")
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp == "bf16")


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)  # Path.rename fails on Windows when the target exists


@torch.no_grad()
def _val_loss(model: nn.Module, loader, cfg: dict, device: torch.device, class_weights, amp: str) -> float:
    """Total loss over the validation runs (batch totals weighted by their run count); NaN without runs."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        with _autocast(device, amp):
            out = model(batch)
        loss, _ = compute_losses(out, batch, cfg, class_weights)
        total, n = total + float(loss) * batch.n_runs, n + batch.n_runs
    return total / n if n else math.nan


def fit_model(model: nn.Module, train_loader, val_loader, cfg: dict, device, out_dir=None,
              class_weights=None) -> dict:
    """Train `model` in place on RunBatches with models.losses.compute_losses.

    AdamW (train.lr, train.weight_decay), cosine annealing over train.epochs (stepped per epoch), gradient norm clipped
    at train.grad_clip, train.amp "bf16" autocasts the forward pass. Batches with a non-finite total loss are skipped
    (counted), so one bad batch cannot corrupt the weights. Early stopping on the validation total loss
    (train.early_stopping_patience epochs without improvement) and the best weights are restored; without validation
    runs (val_loader None or empty) every epoch runs and the last weights are kept.
    out_dir: history.json and best.pt (state_dict) written atomically.
    Returns {"history": [{"epoch", "lr", "train_loss", "train_<term>", "val_loss", "n_skipped"}], "best_epoch",
    "best_val_loss" (NaN without validation), "epochs_run", "seconds"}.
    """
    t0, t = time.perf_counter(), cfg["train"]
    device, amp = torch.device(device), str(t.get("amp", "none"))
    _autocast(device, amp)  # validate train.amp before training
    epochs, patience = int(t["epochs"]), int(t["early_stopping_patience"])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    has_val = val_loader is not None and len(val_loader) > 0
    best_val, best_epoch, best_state, history = (math.inf if has_val else math.nan), -1, None, []
    for epoch in range(epochs):
        model.train()
        sums, n_batches, n_skipped = defaultdict(float), 0, 0
        lr = opt.param_groups[0]["lr"]
        for batch in train_loader:
            batch = batch.to(device)
            with _autocast(device, amp):
                out = model(batch)
            loss, terms = compute_losses(out, batch, cfg, class_weights)
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip"]))
            opt.step()
            for k, v in terms.items():
                sums[k] += v
            n_batches += 1
        sched.step()
        val = _val_loss(model, val_loader, cfg, device, class_weights, amp) if has_val else math.nan
        mean = {f"train_{k}": v / n_batches for k, v in sums.items()} if n_batches else {}
        history.append({"epoch": epoch, "lr": lr, "train_loss": mean.pop("train_total", math.nan), **mean,
                        "val_loss": val, "n_skipped": n_skipped})
        if n_skipped:
            get_logger().warning("fit_model: epoch %d skipped %d batches with a non-finite loss", epoch, n_skipped)
        if not has_val or val < best_val:  # a NaN validation loss never counts as an improvement
            best_val, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    result = {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": len(history),
              "seconds": round(time.perf_counter() - t0, 2)}
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _atomic(out / "best.pt", lambda p: torch.save(model.state_dict(), p))
        _atomic(out / "history.json", lambda p: save_json(result, p))
    return result


@torch.no_grad()
def predict_runs(model: nn.Module, loader, device) -> dict[str, pd.DataFrame]:
    """Predictions of every particle, sample and run in `loader` (an evaluation run loader: all surviving particles).

    particles: particle_id, y_pred (argmax), p_<class> (softmax), log_volume_pred [ln um^3];
    samples: sample_id, severity_pred (argmax sev_p_*), sev_p_mild/transition/severe (CORAL), log10_rate_pred (interval
    k [mm^3/(N m)], NaN for cumulative runs), log_debris_volume_mm3 [ln mm^3];
    runs: run_id, log10_rate_pred, log_debris_volume_mm3 [ln mm^3], coverage.
    """
    model.to(device).eval()
    parts, samples, runs = [], [], []
    for batch in loader:
        out = model(batch.to(device))
        p = torch.softmax(out["mech_logits"].float(), -1).cpu().numpy()
        sev = coral_probs(out["severity_logits"]).cpu().numpy()
        parts.append(pd.DataFrame({"particle_id": batch.particle_ids, "y_pred": p.argmax(1),
                                   **dict(zip(prob_columns(), p.T)),
                                   "log_volume_pred": out["log_volume"].float().cpu().numpy()}))
        samples.append(pd.DataFrame({"sample_id": batch.sample_ids, "severity_pred": sev.argmax(1),
                                     **dict(zip(SEVERITY_PROB_COLUMNS, sev.T)),
                                     "log10_rate_pred": out["log10_rate_sample"].float().cpu().numpy(),
                                     "log_debris_volume_mm3": out["log_debris_volume_sample_mm3"].float().cpu().numpy()}))
        runs.append(pd.DataFrame({"run_id": batch.run_ids, "log10_rate_pred": out["log10_rate_run"].float().cpu().numpy(),
                                  "log_debris_volume_mm3": out["log_debris_volume_run_mm3"].float().cpu().numpy(),
                                  "coverage": out["coverage_run"].float().cpu().numpy()}))

    def cat(frames, columns):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

    return {"particles": cat(parts, PARTICLE_VALUE_COLUMNS), "samples": cat(samples, SAMPLE_VALUE_COLUMNS),
            "runs": cat(runs, RUN_VALUE_COLUMNS)}
