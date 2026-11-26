import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss

from wearpg.metrics import (classification_metrics, expected_calibration_error, metrics_by_group, regression_metrics,
                            reliability_curve, severity_metrics, transition_timing, volume_consistency)


def _random_case(n=300, c=7, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c - 1, n)  # class c-1 absent from y_true
    y[:10] = -1  # unlabelled rows must be ignored
    logits = rng.normal(size=(n, c)) + 2.0 * np.eye(c)[np.clip(y, 0, None)]
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y, p


def test_classification_metrics_match_sklearn():
    y, p = _random_case()
    m = classification_metrics(y, p)
    yl, pl = y[y >= 0], p[y >= 0]
    pred = pl.argmax(axis=1)
    present = np.unique(yl)
    assert m["n"] == len(yl)
    assert m["macro_f1"] == pytest.approx(f1_score(yl, pred, labels=present, average="macro", zero_division=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns about predicted classes absent from y_true
        assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(yl, pred))
    assert m["accuracy"] == pytest.approx((pred == yl).mean())
    assert m["log_loss"] == pytest.approx(log_loss(yl, pl, labels=np.arange(7)))
    assert m["brier"] == pytest.approx(((pl - np.eye(7)[yl]) ** 2).sum(axis=1).mean())
    assert np.array_equal(m["confusion"], confusion_matrix(yl, pred, labels=np.arange(7)))  # rows = true
    pc = m["per_class"]
    assert np.mean([pc[c]["f1"] for c in pc if pc[c]["support"] > 0]) == pytest.approx(m["macro_f1"])
    assert np.isnan(pc["oxide"]["sensitivity"]) and pc["oxide"]["support"] == 0  # absent class
    assert 0 <= pc["oxide"]["specificity"] <= 1
    assert sum(v["support"] for v in pc.values()) == m["n"]


def test_classification_metrics_perfect_and_empty():
    y = np.array([0, 1, 2, 2, 5])
    m = classification_metrics(y, np.eye(7)[y])
    assert m["macro_f1"] == 1.0 and m["balanced_accuracy"] == 1.0 and m["accuracy"] == 1.0
    assert m["ece"] == 0.0 and m["brier"] == 0.0 and m["log_loss"] == pytest.approx(0.0, abs=1e-12)
    e = classification_metrics(np.array([-1, -1]), np.full((2, 7), 1 / 7))
    assert e["n"] == 0 and np.isnan(e["macro_f1"]) and e["confusion"].sum() == 0


def test_ece_and_reliability_hand_computed():
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])  # confidence 0.8, accuracy 0.5
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.3)
    rc = reliability_curve(y, p, n_bins=10)
    assert rc["count"][7] == 4 and rc["count"].sum() == 4  # 0.8 lies in bin (0.7, 0.8]
    assert rc["confidence"][7] == pytest.approx(0.8) and rc["accuracy"][7] == pytest.approx(0.5)
    # bin (0, .5]: conf .40 (right), .45 (wrong); bin (.5, 1]: conf .9, .7 (both right)
    p3 = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25], [0.45, 0.3, 0.25], [0.2, 0.7, 0.1]])
    want = 0.5 * abs(0.5 - 0.425) + 0.5 * abs(1.0 - 0.8)
    assert expected_calibration_error(y, p3, n_bins=2) == pytest.approx(want)


def test_metrics_by_group():
    y, p = _random_case(seed=1)
    g = np.where(np.arange(len(y)) % 3 == 0, 500.0, 200.0)
    df = metrics_by_group(y, p, g)
    assert list(df.columns) == ["group", "metric", "value", "n"]
    assert df.group.unique().tolist() == [200.0, 500.0]
    one = classification_metrics(y[g == 500.0], p[g == 500.0])
    row = df[(df.group == 500.0) & (df.metric == "macro_f1")].iloc[0]
    assert row.value == pytest.approx(one["macro_f1"]) and row.n == one["n"]


def test_severity_metrics_qwk_hand_computed():
    y = np.array([0, 0, 1, 1, 2, 2, -1])
    pred = np.array([0, 1, 1, 2, 2, 2, 0])
    m = severity_metrics(y, np.eye(3)[pred])
    # confusion [[1,1,0],[0,1,1],[0,0,2]], quadratic weights: observed 2, expected 8 -> 1 - 2/8
    assert m["qwk"] == pytest.approx(0.75)
    assert m["ordinal_mae"] == pytest.approx(2 / 6) and m["n"] == 6
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(severity_metrics(np.zeros(4, int), np.tile([0.9, 0.05, 0.05], (4, 1)))["qwk"])
    # chance-level agreement with the middle level unused -> 0 (labels fixed to [0, 1, 2])
    m2 = severity_metrics(np.array([0, 0, 2, 2]), np.eye(3)[[0, 2, 2, 0]])
    assert m2["qwk"] == pytest.approx(0.0)


def test_regression_metrics_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    yh = np.array([1.5, 2.0, 2.5, 5.0, 1.0])
    m = regression_metrics(y, yh)
    assert m["n"] == 4 and m["mae"] == pytest.approx(0.5) and m["bias"] == pytest.approx(0.25)
    assert m["rmse"] == pytest.approx(np.sqrt((0.25 + 0 + 0.25 + 1) / 4))
    assert m["r2"] == pytest.approx(1 - 1.5 / 5.0)
    assert m["spearman"] == pytest.approx(stats.spearmanr(y[:4], yh[:4])[0])
    assert m["pearson"] == pytest.approx(stats.pearsonr(y[:4], yh[:4])[0])
    assert np.isnan(regression_metrics([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])["r2"])


def test_within_condition_spearman():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(6), 4)
    level = 2.0 * g  # strong between-condition trend shared by y and yhat
    noise = rng.normal(size=g.size)
    y = level + noise
    yh = level - noise  # within-condition order reversed
    m = regression_metrics(y, yh, groups=g)
    assert m["spearman"] > 0.8 and m["within_condition_spearman"] == pytest.approx(-1.0)
    # conditions-only style predictions (constant within condition) carry no within-condition signal
    const = regression_metrics(y, np.repeat(rng.normal(size=6), 4) + 0.1, groups=g)
    assert np.isnan(const["within_condition_spearman"])
    single = regression_metrics(y[::4], yh[::4], groups=g[::4])  # every group has 1 member
    assert np.isnan(single["within_condition_spearman"])
    assert "within_condition_spearman" not in regression_metrics(y, yh)


def test_transition_timing_hand_computed():
    t0 = np.arange(4) * 100.0
    runs, samples = [], []

    def add(run, t_true, p, sev):
        runs.append({"run_id": run, "duration_s": 400.0, "transition_time_true": t_true})
        samples.extend({"run_id": run, "t_start_s": a, "time_s": a + 100.0, "p_severe": q, "severity_true": s}
                       for a, q, s in zip(t0, p, sev))

    add("R1", 120.0, [0.1, 0.2, 0.7, 0.9], [0, 1, 2, 2])     # detected at 250 -> +130; oracle 150 -> 30
    add("R2", 350.0, [0.1, 0.2, 0.3, 0.4], [0, 0, 0, 1])     # missed -> scored at 400 -> +50; oracle 350
    add("R3", 250.0, [0.6, 0.1, 0.9, 0.9], [0, 0, 1, 2])     # early alarm at 50 -> -200; oracle 250
    add("S1", np.nan, [0.1, 0.8, 0.1, 0.1], [0, 0, 0, 0])    # false alarm
    add("S2", np.nan, [0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0])
    runs.append({"run_id": "R_unobserved", "duration_s": 400.0, "transition_time_true": 10.0})  # no samples
    s = pd.DataFrame(samples).sample(frac=1.0, random_state=0)  # order must not matter
    m = transition_timing(s, pd.DataFrame(runs), threshold=0.5)
    assert m["n_transition_runs"] == 3 and m["n_stable_runs"] == 2
    assert m["mae_s"] == pytest.approx((130 + 50 + 200) / 3)
    assert m["bias_s"] == pytest.approx((130 + 50 - 200) / 3)
    assert m["interval_s"] == 100.0 and m["mae_intervals"] == pytest.approx(m["mae_s"] / 100.0)
    assert m["detection_rate"] == pytest.approx(2 / 3) and m["early_alarm_rate"] == pytest.approx(1 / 3)
    assert m["false_alarm_rate"] == pytest.approx(0.5)
    assert m["oracle_floor_s"] == pytest.approx(30 / 3)
    pr = m["per_run"].set_index("run_id")
    assert pr.loc["R2", "t_det"] == 400.0 and not pr.loc["R2", "detected"] and pr.loc["R3", "early"]


def test_volume_consistency():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    same = volume_consistency(x, x)
    assert same["ccc"] == pytest.approx(1.0) and same["slope"] == pytest.approx(1.0)
    assert same["intercept"] == pytest.approx(0.0, abs=1e-12) and same["r2"] == pytest.approx(1.0)
    assert same["mean_log_ratio"] == 0.0 and same["sd_log_ratio"] == 0.0 and same["n"] == 50
    c = 0.7
    shifted = volume_consistency(np.append(x + c, np.nan), np.append(x, 1.0))
    assert shifted["n"] == 50 and shifted["mean_log_ratio"] == pytest.approx(c)
    assert shifted["ccc"] == pytest.approx(2 * x.var() / (2 * x.var() + c**2))
    assert np.isnan(volume_consistency([1.0], [1.0])["ccc"])


def test_fast_paths_equal_the_library_definitions():
    """The bootstrap hot paths (QWK from the confusion matrix, Spearman via ranks, group counting) reproduce sklearn /
    scipy / the pandas definition (evaluate_results speed-up must not change results)."""
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 3, 40)
        p = rng.dirichlet(np.ones(3), 40)
        ref = cohen_kappa_score(y, p.argmax(1), labels=[0, 1, 2], weights="quadratic")
        assert severity_metrics(y, p)["qwk"] == pytest.approx(ref, abs=1e-12)
        x, z = rng.normal(size=30).round(1), rng.normal(size=30).round(1)  # rounded: ties
        g = rng.choice(["a", "b", "c", "d"], 30)
        r = regression_metrics(x, z, g)
        assert r["spearman"] == pytest.approx(stats.spearmanr(x, z)[0], abs=1e-12)
        assert r["pearson"] == pytest.approx(stats.pearsonr(x, z)[0], abs=1e-12)
        gs = pd.Series(g)
        keep = (gs.map(gs.value_counts()) >= 2).to_numpy()
        cx = x[keep] - pd.Series(x[keep]).groupby(g[keep]).transform("mean").to_numpy()
        cz = z[keep] - pd.Series(z[keep]).groupby(g[keep]).transform("mean").to_numpy()
        assert r["within_condition_spearman"] == pytest.approx(stats.spearmanr(cx, cz)[0], abs=1e-12)
