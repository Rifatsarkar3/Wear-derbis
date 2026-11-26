"""Tests for scripts/open_oil_posthoc.py (post-hoc NASA analyses): exact tests against brute force, LOO against refits and
the frozen predictions, transcription identities, interval construction. No network."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import open_oil_posthoc as ph  # noqa: E402


def test_signflip_matches_brute_force():
    rng = np.random.default_rng(1)
    for n in (1, 5, 9):
        d = np.round(rng.normal(0.4, 1, n), 2)
        s = np.array([abs(np.dot(sg, d)) for sg in itertools.product([1, -1], repeat=n)])
        assert ph.signflip_p(d) == pytest.approx((s >= abs(d.sum()) - 1e-9).mean())
    assert ph.signflip_p([1, 2, 3, 4, 5]) == pytest.approx(2 / 32) and ph.signflip_p([1, -1]) == 1.0
    assert ph.sign_test_p([1, 1, 1, 1]) == pytest.approx(2 / 16)


def test_loo_linear_equals_refitting():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(9, 2)), rng.normal(size=9)
    for Z in (X, X[:, :1], np.empty((9, 0))):
        want = []
        for i in range(9):
            tr = np.arange(9) != i
            A = np.column_stack([np.ones(8), Z[tr]])
            want.append(np.concatenate([[1.0], Z[i]]) @ np.linalg.lstsq(A, y[tr], rcond=None)[0])
        np.testing.assert_allclose(ph.loo_linear(Z, y), want, atol=1e-12)
    two = ph.loo_linear(X, np.column_stack([y, 2 * y]))
    np.testing.assert_allclose(two[:, 1], 2 * two[:, 0], atol=1e-12)


def test_frozen_loo_predictions_are_reproduced():
    csv = ROOT / "outputs/open_data/oil/nasa_loo_regression_predictions.csv"
    if not csv.exists():
        pytest.skip("frozen outputs absent")
    frozen = pd.read_csv(csv).pivot(index="exp", columns="model", values="n_hat")
    mine = ph.loo_predictions(ph.experiments()).set_index("exp")
    for m in ("time_only", "debris_only", "debris_plus_time"):
        np.testing.assert_allclose(mine[m].to_numpy(), frozen.loc[mine.index, m].to_numpy(), atol=1e-9)


def test_sets_and_split_test_identity():
    sets = ph.experiment_sets()
    assert [len(s) for s in sets.values()] == [24, 23, 22, 18, 16]
    s1 = sets["S1_16_17_merged"].set_index("exp")
    assert 17 not in s1.index and s1.loc[16, "run_min"] == 21446 and s1.loc[16, "mass_mg"] == pytest.approx(0.163)
    assert 380 + 21066 == 21446 and 0.099 + 0.064 == pytest.approx(0.163)  # TM-2003 exps 16 + 17 = TM-2001 exp 16
    assert (sets["S0_all_24"].n_de > 0).sum() == 14 and 11 not in set(sets["S2_S1_without_11"].exp)
    broken = ph.experiments()
    broken.loc[broken.exp == 17, "mass_mg"] = 1.0
    with pytest.raises(ValueError, match="16"):
        ph.merge_16_17(broken)
    rep = ph.repeatability()  # 350 + 357 + 497 h = 1204 h (TM-2000-210371 p. 7)
    assert int(sum(rep[rep.exp.isin([9, 15, 16])].run_h.astype(int))) == 1204


def test_intervals_origin_rules_and_damage_increments():
    iv = ph.intervals()
    assert len(iv) == 58 and len(ph.intervals(origin="f6")) == 56
    assert iv[iv.exp == 4].dn.tolist() == [0, 0, 0, 3, 0, 1, 2, 4] and iv[iv.exp == 8].dn.tolist() == [0, 12]
    e2 = iv[iv.exp == 2].iloc[0]  # 'Run-in Wear' row with mass > 0 is an inspection
    assert (e2.t0, e2.t1, e2.m1) == (0.0, 1573.0, 3.285)
    assert ph.intervals(origin="f6").query("exp == 2").iloc[0].m0 == 0
    assert iv[iv.exp == 5].dn.tolist() == [0, 0, 21] and ph.intervals("table_3_14").query("exp == 5").dn.tolist() == [
        0, 2, 19]
    assert iv.groupby("exp").dm.sum().round(6).loc[1] == 36.108 and (iv.groupby("exp").completion.sum() == 1).all()


def test_exact_conditional_logistic_against_enumeration():
    rng = np.random.default_rng(3)
    rows = []
    for e, (n, k) in {1: (5, 2), 2: (4, 1), 3: (3, 3), 4: (4, 3)}.items():  # stratum 3 carries no contrast
        lab = rng.permutation([True] * k + [False] * (n - k))
        rows += [{"exp": e, "rate": float(np.exp(rng.normal(0.5 * v, 1))), "progress": bool(v)} for v in lab]
    iv = pd.DataFrame(rows)
    r = ph.exact_conditional_logistic(iv)
    assert r["n_experiments"] == 3 and r["n_arrangements"] == 10 * 4 * 4
    per, obs = [], 0.0
    for e, g in iv.groupby("exp"):
        x, lab = np.log(g.rate.to_numpy()), g.progress.to_numpy()
        if lab.all():
            continue
        per.append([x[list(c)].sum() for c in itertools.combinations(range(len(x)), int(lab.sum()))])
        obs += x[lab].sum()
    tot = np.array([sum(c) for c in itertools.product(*per)])
    assert r["p_exact"] == pytest.approx((np.abs(tot - tot.mean()) >= abs(obs - tot.mean()) - 1e-9).mean())
    beta = r["beta_ln"]  # conditional MLE: observed statistic equals its expectation
    expect = sum(np.dot(np.exp(beta * np.array(s)), s) / np.exp(beta * np.array(s)).sum() for s in per)
    assert expect == pytest.approx(obs, abs=1e-3)
    assert ph.exact_conditional_logistic(iv[iv.exp == 3])["n_experiments"] == 0


def test_post_vs_before_and_within_contrast():
    iv = pd.DataFrame({"exp": [1, 1, 1, 2, 2], "dm": [1.0, 2.0, 8.0, 1.0, 1.0], "dt": [10.0, 10.0, 10.0, 10.0, 10.0],
                       "t1": [10.0, 20.0, 30.0, 10.0, 20.0], "destructive": [False, True, True, False, True],
                       "progress": [False, True, False, False, True]})
    iv["rate"] = iv.dm / iv.dt
    h2 = ph.post_vs_before(iv)  # exp 1: before 3 / 20, after 8 / 10; exp 2 stopped at first destructive pitting -> left out
    assert h2["n_experiments"] == 1 and h2["per_experiment"][0]["ratio"] == pytest.approx(0.8 / 0.15)
    w = ph.within_contrast(iv, "progress")
    assert w["per_experiment_ln_ratio"][2] == pytest.approx(0.0) and w["n_experiments"] == 2


def test_stage_respecting_null_keeps_the_stage_contrast():
    r = ph.stage_respecting(ph.experiments())
    assert r["observed_rho"] == pytest.approx(0.8612858594555258, abs=1e-12)
    assert r["null_mean_rho"] < r["observed_rho"] and r["n_permuted"] == 14 and 0 < r["p_two_sided"] <= 1


def test_ordinal_pieces():
    assert ph.ordinal_levels([0, 1, 4, 5, 21]).tolist() == [0, 1, 1, 2, 2]
    z = np.array([0, 2])
    np.testing.assert_allclose(ph.rps(np.array([[1.0, 1.0], [1.0, 1.0]]), z), [0.0, 1.0])
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 1))
    u = 1.5 * (x[:, 0] - x.mean()) / x.std() + rng.logistic(size=1500)  # latent PO model with cuts -0.5, 1.0
    zs = np.digitize(u, [-0.5, 1.0])
    predict = ph.fit_proportional_odds(x, zs)
    xq = (np.array([[-1.0], [0.0], [1.0]]) * x.std() + x.mean())
    want = 1 / (1 + np.exp(-(np.array([-0.5, 1.0])[None, :] - 1.5 * np.array([[-1.0], [0.0], [1.0]]))))
    np.testing.assert_allclose(predict(xq), want, atol=0.06)


def test_stop_rules_flag_inconsistent_stops():
    iv = ph.intervals()
    summ, t = ph.stop_rules(iv)
    assert set(t.rule) == {"mass>=5.453", "mass>=8.69", "mass>=15.475", "rate>=0.005"}
    s = summ["mass>=8.69"]
    assert s["n_tests"] == 14 and s["exps_end_below_rule"] == [20, 22, 23] and 1 in s["exps_run_on_past_rule"]
    assert s["n_consistent"] == 14 - len(set(s["exps_end_below_rule"]) | set(s["exps_run_on_past_rule"]))


def test_lognormal_summary_and_tm2004_table():
    r = ph.lognormal_summary([1.0, 10.0, 100.0], [1, 2, 3])
    assert r["geometric_mean"] == pytest.approx(10) and r["max_over_min"] == pytest.approx(100) and r["n_clusters"] == 3
    t = ph.oo._table(ph.TM2004_END).astype(float)
    ex = ph.experiments().set_index("exp")
    d = (t.set_index("exp").mass_mg - ex.mass_mg).dropna().abs()
    assert len(t) == 17 and set(d.index[d > 0.06]) == {4, 5, 6}  # re-processed masses that differ beyond rounding


def test_committed_results_follow_the_prewritten_rules():
    f = ROOT / "outputs/open_data/oil/posthoc/results.json"
    if not f.exists():
        pytest.skip("run scripts/open_oil_posthoc.py first")
    r = ph.load_json(f)
    s0 = r["A_baselines"]["S0_all_24"]
    for m, v in ph.FROZEN_MAE.items():
        assert s0["mae"][m] == pytest.approx(v, abs=1e-9)
    pa = s0["paired"]["debris_only_minus_best_intercept"]
    assert (r["verdicts"]["V-A"] == "STRENGTHENED") == (pa["hi"] < 0 and pa["p_signflip"] < 0.05)
    assert r["verdicts"]["V-F"] == "NOT SUPPORTED" and s0["best_intercept_only"] in ph.INTERCEPT_ONLY
