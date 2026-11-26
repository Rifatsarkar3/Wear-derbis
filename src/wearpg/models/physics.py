"""Aggregate debris-volume physics (DESIGN sec. 3), computed in float32.

Particle volumes only enter through sample/run sums (Archard is never applied per particle). Empty groups yield
-inf / NaN outputs that callers mask; no gradient path reaches parameters through an empty group.
Units: particle volumes ln um^3, debris volumes ln mm^3, distances m, loads N.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch_geometric.utils import scatter

from wearpg.schema import UM3_PER_MM3


def scatter_logsumexp(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Group-wise log-sum-exp of 1-D `src` (n,) by `index` (n,) -> (dim_size,); empty group -> -inf.

    The group max is a detached shift, so the gradient is exactly the within-group softmax.
    """
    m = scatter(src.detach(), index, 0, dim_size, "max")
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))  # all -inf group: avoid -inf - -inf
    return scatter((src - m[index]).exp(), index, 0, dim_size, "sum").log() + m


def metal_fraction_log(mech_probs_detached: Tensor, c_by_class: Tensor) -> Tensor:
    """ln c_i = ln sum_k p_ik c_k for (n_p, C) mechanism probabilities and (C,) metal-equivalent factors -> (n_p,).

    Detached again here: the mechanism classifier is never trained through the volume path.
    """
    return torch.log(mech_probs_detached.detach().float() @ c_by_class.float())


def debris_log_volume_sample(log_vol_um3: Tensor, log_c: Tensor, particle_to_sample: Tensor, n_samples: int,
                             log_subsample_factor: Tensor, log_obs_fraction: Tensor, log_eta_sample: Tensor) -> Tensor:
    """ln D_s [mm^3] = LSE_{i in bag}(ln v_i + ln c_i) + ln HT_s - ln f_s - ln eta_r - ln 1e9 -> (n_s,).

    Particle inputs (n_p,), sample inputs (n_s,); a sample without particles gives -inf.
    Interval runs: D_s = interval wear volume V_s; cumulative runs: D_s = oil inventory M_s.
    """
    lse = scatter_logsumexp(log_vol_um3.float() + log_c.float(), particle_to_sample, n_samples)
    return (lse + log_subsample_factor.float() - log_obs_fraction.float() - log_eta_sample.float()
            - math.log(UM3_PER_MM3))


def debris_log_volume_run(log_d_sample: Tensor, sample_to_run: Tensor, n_runs: int, sample_distance_m: Tensor,
                          sliding_distance_m: Tensor, sample_time_s: Tensor, sample_log_q: Tensor,
                          run_cumulative: Tensor, sample_distance_end_m: Tensor | None = None
                          ) -> tuple[Tensor, Tensor]:
    """Run debris volume with coverage correction over the OBSERVED samples (those present in the batch).

    interval:   ln V_r = LSE_s ln V_s + ln(L / sum_s ds_s),               coverage = sum_s ds_s / L
    cumulative: V_r = (M_last + sum_{s<last} q_s M_s) * L / s_end(last),   coverage = s_end(last) / L
    (last = latest observed sample by `sample_time_s`; the removal sum only sees observed earlier samples.)
    Returns (ln V_r [mm^3], coverage_r), both (n_runs,); runs without observed samples -> (NaN, 0).

    `sample_distance_end_m` (n_s,) = cumulative sliding distance at each sample's end [m]; it gives the exact
    s_end(last). shortcut: when it is None (RunBatch does not carry it yet), s_end(last) = sum of observed ds, exact
    only if no earlier sample of a cumulative run is absent; pass it once RunBatch has the field.
    """
    cum_s = run_cumulative[sample_to_run]
    t_last = scatter(sample_time_s.float(), sample_to_run, 0, n_runs, "max")
    is_last = sample_time_s.float() >= t_last[sample_to_run]
    term = log_d_sample.float() + torch.where(cum_s & ~is_last, sample_log_q.float(), 0.0)
    lse = scatter_logsumexp(term, sample_to_run, n_runs)
    covered = scatter(sample_distance_m.float(), sample_to_run, 0, n_runs, "sum")
    if sample_distance_end_m is not None:
        s_end = scatter(torch.where(is_last, sample_distance_end_m.float(), 0.0), sample_to_run, 0, n_runs, "sum")
        covered = torch.where(run_cumulative, s_end, covered)
    coverage = covered / sliding_distance_m.float()
    return torch.where(coverage > 0, lse - torch.log(coverage), float("nan")), coverage


def consistency_loss(log_v_run: Tensor, log_grav: Tensor, coverage: Tensor, min_coverage: float = 0.5) -> Tensor:
    """Mean squared ln(V_debris / V_grav) over runs with finite prediction AND target and coverage >= min_coverage;
    0 when there is none."""
    m = torch.isfinite(log_v_run) & torch.isfinite(log_grav) & (coverage >= min_coverage)
    d = log_v_run[m] - log_grav[m].to(log_v_run.dtype)
    return d.pow(2).sum() / m.sum().clamp_min(1)


def monotone_severity_loss(expected_sev_run: Tensor, load_N: Tensor, mono_group: Tensor, condition_key: Tensor,
                           margin: float = 0.0) -> Tensor:
    """Condition-mean severity must not decrease with load inside a mono group.

    E_c = mean expected severity of the finite runs of condition c (`condition_key`); over condition pairs of the same
    `mono_group` with load_c < load_c': mean relu(E_c - E_c' + margin); 0 if there is no pair.
    """
    ok = torch.isfinite(expected_sev_run) & torch.isfinite(load_N)
    _, inv = torch.unique(condition_key[ok], return_inverse=True)
    n = int(inv.max()) + 1 if inv.numel() else 0
    e = scatter(expected_sev_run[ok], inv, 0, n, "mean")
    load = scatter(load_N[ok].to(e.dtype), inv, 0, n, "mean")
    group = scatter(mono_group[ok], inv, 0, n, "max")
    pair = (group[:, None] == group[None, :]) & (load[:, None] < load[None, :])
    return torch.relu(e[:, None] - e[None, :] + margin)[pair].sum() / pair.sum().clamp_min(1)
