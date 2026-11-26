import math

import numpy as np
import pytest
import torch

from wearpg.models.physics import (consistency_loss, debris_log_volume_run, debris_log_volume_sample,
                                   metal_fraction_log, monotone_severity_loss, scatter_logsumexp)
from wearpg.schema import MECHANISMS

C_BY_CLASS = torch.tensor([0.47 if c == "oxide" else 1.0 for c in MECHANISMS])
OXIDE = MECHANISMS.index("oxide")


def t(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


def run_volume(log_d, s2r, n_runs, ds, L, time, log_q, cumulative, s_end=None):
    return debris_log_volume_run(log_d, t(s2r, torch.long), n_runs, t(ds), t(L), t(time), log_q, t(cumulative, torch.bool),
                                 None if s_end is None else t(s_end))


def test_scatter_logsumexp_matches_torch_empty_group_and_softmax_grad():
    src = torch.randn(7, requires_grad=True)
    index = t([0, 0, 2, 2, 2, 3, 0], torch.long)
    out = scatter_logsumexp(src, index, 5)
    for g in (0, 2, 3):
        assert torch.allclose(out[g], torch.logsumexp(src[index == g], 0), atol=1e-6)
    assert torch.isneginf(out[1]) and torch.isneginf(out[4])
    (out[0] + out[2] + out[3]).backward()
    for g in (0, 2, 3):  # gradient = within-group softmax
        assert torch.allclose(src.grad[index == g], torch.softmax(src[index == g].detach(), 0), atol=1e-6)
    all_neg_inf = scatter_logsumexp(t([-math.inf, -math.inf, 1.0]), t([0, 0, 1], torch.long), 2)
    assert torch.isneginf(all_neg_inf[0]) and torch.isclose(all_neg_inf[1], t(1.0))


def test_metal_fraction_log_is_detached_mixture():
    p = torch.zeros(3, len(MECHANISMS), requires_grad=False)
    p[0, OXIDE] = 1.0
    p[1, 0] = 1.0
    p[2, OXIDE] = p[2, 1] = 0.5
    p.requires_grad_(True)
    out = metal_fraction_log(p, C_BY_CLASS)
    assert torch.allclose(out, torch.log(t([0.47, 1.0, 0.735])))
    assert not out.requires_grad


def test_debris_volume_sample_hand_example():
    # sample 0: v = 100 + 300 um^3 (metal), HT 2, f 0.01, eta 0.5 -> 400 * 2 / 0.005 / 1e9 = 1.6e-4 mm^3
    # sample 1: v = 1000 um^3 oxide (c 0.47), HT 1, f 0.02, eta 0.25 -> 470 / 0.005 / 1e9 = 9.4e-5 mm^3
    out = debris_log_volume_sample(torch.log(t([100.0, 1000.0, 300.0])), torch.log(t([1.0, 0.47, 1.0])),
                                   t([0, 1, 0], torch.long), 2, torch.log(t([2.0, 1.0])), torch.log(t([0.01, 0.02])),
                                   torch.log(t([0.5, 0.25])))
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.log(t([1.6e-4, 9.4e-5])), atol=1e-5)


def test_consistency_near_zero_when_debris_follows_generative_relation():
    """Particles generated so that metal-equivalent volume ~ A^1.5 and each sample's table holds the full
    eta * f * V_s budget; the bag is a random half (area-ratio HT factor); interval runs fully observed."""
    rng = np.random.default_rng(0)
    load, L, k, eta = np.array([20.0, 120.0]), 720.0, np.array([2e-6, 1.5e-4]), np.array([0.6, 0.2])
    ds = np.array([[100.0, 200.0, 420.0], [240.0, 240.0, 240.0]])
    log_v, log_c, p2s, log_ht, log_f, log_eta, s2r = [], [], [], [], [], [], []
    for r in range(2):
        for s in range(3):
            f = rng.uniform(1e-4, 1e-2)
            budget_um3 = eta[r] * f * k[r] * load[r] * ds[r, s] * 1e9
            area = rng.uniform(5, 500, 12)
            c = np.where(rng.random(12) < 0.3, 0.47, 1.0)
            v = budget_um3 * area**1.5 / (area**1.5).sum() / c   # sum c v = budget, c v ~ A^1.5
            bag = rng.choice(12, 6, replace=False)
            log_v += np.log(v[bag]).tolist()
            log_c += np.log(c[bag]).tolist()
            p2s += [3 * r + s] * 6
            log_ht.append(np.log((area**1.5).sum() / (area[bag] ** 1.5).sum()))
            log_f.append(np.log(f))
            log_eta.append(np.log(eta[r]))
            s2r.append(r)
    log_d = debris_log_volume_sample(t(log_v), t(log_c), t(p2s, torch.long), 6, t(log_ht), t(log_f), t(log_eta))
    assert torch.allclose(log_d, torch.log(t((k[:, None] * load[:, None] * ds).ravel())), atol=1e-4)
    log_vr, cov = run_volume(log_d, s2r, 2, ds.ravel(), [L, L], [600, 1200, 1800] * 2, torch.zeros(6), [False, False])
    log_grav = torch.log(t(k * load * L))
    assert torch.allclose(cov, torch.ones(2))
    assert consistency_loss(log_vr, log_grav, cov) < 1e-7


def test_dropping_half_the_samples_of_constant_rate_run_keeps_run_volume():
    k, load = 1e-5, 60.0
    ds = np.array([100.0, 300.0, 250.0, 150.0])
    log_d = torch.log(t(k * load * ds))
    log_all, cov_all = run_volume(log_d, [0] * 4, 1, ds, [800.0], [1, 2, 3, 4], torch.zeros(4), [False])
    keep = [0, 2]
    log_half, cov_half = run_volume(log_d[keep], [0, 0], 1, ds[keep], [800.0], [1, 3], torch.zeros(2), [False])
    assert torch.allclose(log_all, torch.log(t([k * load * 800.0])), atol=1e-5)
    assert torch.allclose(log_half, log_all, atol=1e-5)
    assert torch.isclose(cov_all, t([1.0])).all() and torch.isclose(cov_half, t([350.0 / 800.0])).all()


def test_cumulative_run_volume_hand_example_mixed_with_interval_run():
    # cumulative run 0: M = 1, 2.5, 4 mm^3, q = 0.1, three 25 m intervals, L = 100 m
    #   V = (4 + 0.1 * 1 + 0.1 * 2.5) * 100 / 75 = 5.8, coverage 0.75
    # interval run 1: V_s = 1, 2 over 30 m each, L = 100 -> V = 3 * 100 / 60 = 5, coverage 0.6
    # samples deliberately out of time order: "last" is chosen by time, not position
    log_d = torch.log(t([2.5, 1.0, 2.0, 4.0, 1.0]))
    s2r, time = [0, 1, 1, 0, 0], [50.0, 60.0, 120.0, 75.0, 25.0]
    ds = [25.0, 30.0, 30.0, 25.0, 25.0]
    log_q = torch.log(torch.full((5,), 0.1))
    log_v, cov = run_volume(log_d, s2r, 2, ds, [100.0, 100.0], time, log_q, [True, False])
    assert torch.allclose(log_v.exp(), t([5.8, 5.0]), rtol=1e-5)
    assert torch.allclose(cov, t([0.75, 0.6]))
    # middle cumulative sample absent: exact with the end distance of the last sample ...
    keep = [1, 2, 3, 4]
    log_v, cov = run_volume(log_d[keep], [s2r[i] for i in keep], 2, [ds[i] for i in keep], [100.0, 100.0],
                            [time[i] for i in keep], log_q[keep], [True, False], s_end=[60.0, 120.0, 75.0, 25.0])
    assert torch.allclose(log_v.exp(), t([(4.0 + 0.1) * 100 / 75, 5.0]), rtol=1e-5)
    assert torch.allclose(cov, t([0.75, 0.6]))
    # ... and the documented fallback (sum of observed ds) without it
    log_v, cov = run_volume(log_d[keep], [s2r[i] for i in keep], 2, [ds[i] for i in keep], [100.0, 100.0],
                            [time[i] for i in keep], log_q[keep], [True, False])
    assert torch.allclose(log_v.exp(), t([4.1 * 100 / 50, 5.0]), rtol=1e-5) and torch.allclose(cov, t([0.5, 0.6]))


def test_run_without_observed_samples_is_nan_and_gradient_safe():
    log_d = torch.log(t([1e-3, 2e-3])).requires_grad_(True)
    log_v, cov = run_volume(log_d, [0, 0], 3, [10.0, 10.0], [20.0, 50.0, 40.0], [1.0, 2.0], torch.zeros(2),
                            [False, False, True])
    assert torch.isfinite(log_v[0]) and torch.isnan(log_v[1:]).all() and (cov[1:] == 0).all()
    loss = consistency_loss(log_v, torch.log(t([3e-3, 1.0, 1.0])), cov) + (log_v * 0).nansum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(log_d.grad).all()


def test_consistency_loss_masking_and_empty():
    pred = t([0.0, 1.0, math.nan, 2.0, 5.0]).requires_grad_(True)
    target = t([0.5, math.nan, 0.0, 1.0, 0.0])
    cov = t([1.0, 1.0, 1.0, 0.8, 0.3])  # last run below min coverage
    loss = consistency_loss(pred, target, cov, min_coverage=0.5)
    assert torch.isclose(loss, t((0.25 + 1.0) / 2))
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[[1, 2, 4]].eq(0).all()
    empty = consistency_loss(pred, torch.full((5,), math.nan), cov)
    assert empty.item() == 0.0 and empty.requires_grad


def test_monotone_loss_only_for_violating_condition_pairs():
    load = t([20.0, 20.0, 60.0, 120.0, 20.0, 120.0])
    group = t([0, 0, 0, 0, 1, 1], torch.long)
    cond = t([0, 0, 1, 2, 3, 4], torch.long)
    ok = t([0.1, 0.3, 0.5, 1.5, 0.0, 1.0])  # monotone within both groups (cond 0 mean 0.2)
    assert monotone_severity_loss(ok, load, group, cond).item() == 0.0
    # group 1 high severity at low load must not be compared with group 0 (only same-group pairs)
    cross = t([0.1, 0.3, 0.5, 1.5, 1.9, 2.0])
    assert monotone_severity_loss(cross, load, group, cond).item() == 0.0
    bad = t([1.0, 1.0, 0.5, 1.5, 0.0, 1.0]).requires_grad_(True)  # cond 0 (20 N) above cond 1 (60 N)
    loss = monotone_severity_loss(bad, load, group, cond)
    assert torch.isclose(loss, t(0.5 / 4))  # pairs: (0,1) (0,2) (1,2) in group 0, (3,4) in group 1
    loss.backward()
    assert bad.grad[0] > 0 and bad.grad[2] < 0 and bad.grad[3] == 0
    with_nan = t([1.0, math.nan, 0.5, 1.5, 0.0, 1.0])  # NaN runs are ignored in the condition means
    assert torch.isclose(monotone_severity_loss(with_nan, load, group, cond), t(0.5 / 4))
    assert torch.isclose(monotone_severity_loss(ok, load, group, cond, margin=0.25), t(0.0))
    assert monotone_severity_loss(t([0.3]), t([20.0]), t([0], torch.long), t([0], torch.long)).item() == 0.0
    assert monotone_severity_loss(t([math.nan]), t([20.0]), t([0], torch.long), t([0], torch.long)).item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_physics_outputs_float32(dtype):
    log_d = debris_log_volume_sample(torch.log(t([10.0, 20.0])).to(dtype), torch.zeros(2, dtype=dtype),
                                     t([0, 0], torch.long), 1, torch.zeros(1, dtype=dtype),
                                     torch.full((1,), -5.0, dtype=dtype), torch.zeros(1, dtype=dtype))
    log_v, cov = run_volume(log_d, [0], 1, [10.0], [10.0], [1.0], torch.zeros(1), [False])
    assert log_d.dtype == log_v.dtype == cov.dtype == torch.float32
