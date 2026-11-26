import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss

from wearpg.stats import (adjust, benjamini_hochberg, cluster_bootstrap, holm, icc_oneway, paired_cluster_bootstrap,
                          repeatability_cv, repeatability_sd, seed_summary, spearman_permutation)


def test_holm_and_bh_hand_computed():
    p = [0.01, 0.04, 0.03, 0.005]
    # Holm: sorted .005 .01 .03 .04 x (4, 3, 2, 1) = .02 .03 .06 .04 -> running max .02 .03 .06 .06
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    # BH: sorted p * 4 / rank = .02 .02 .04 .04 -> running min from the top keeps them
    assert np.allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])
    assert np.allclose(holm([0.01, 0.02, 0.03, 0.04, 0.05]), [0.05, 0.08, 0.09, 0.09, 0.09])
    assert np.allclose(benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05]), 0.05)
    assert np.allclose(holm([0.6, 0.9]), [1.0, 1.0])  # capped at 1
    out = holm([0.01, np.nan, 0.02])  # NaN passes through, m counts finite p only
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [0.02, 0.02])
    assert np.allclose(adjust(p, "holm"), holm(p)) and np.allclose(adjust(p, "bh"), benjamini_hochberg(p))
    with pytest.raises(ValueError):
        adjust(p, "bonferroni")


def test_seed_summary():
    s = seed_summary([1.0, 2.0, 3.0, np.nan])
    half = ss.t.ppf(0.975, 2) / np.sqrt(3)
    assert s["n"] == 3 and s["mean"] == 2.0 and s["sd"] == 1.0
    assert s["lo"] == pytest.approx(2 - half) and s["hi"] == pytest.approx(2 + half)
    one = seed_summary([0.7])
    assert one["mean"] == 0.7 and np.isnan(one["sd"]) and np.isnan(one["lo"])


def test_icc_oneway_shrout_fleiss():
    # Shrout & Fleiss (1979) 6 targets x 4 judges; one-way BMS 11.24, WMS 6.26 -> ICC(1,1) = 0.17
    data = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    r = icc_oneway(data.ravel(), np.repeat(np.arange(6), 4))
    assert r["icc"] == pytest.approx(0.166, abs=1e-3) and r["F"] == pytest.approx(11.24 / 6.26, rel=1e-3)
    assert r["k"] == 4 and r["n_groups"] == 6 and r["n"] == 24 and r["lo"] < r["icc"] < r["hi"]
    with_singleton = icc_oneway(np.append(data.ravel(), 99.0), np.append(np.repeat(np.arange(6), 4), 7))
    assert with_singleton["icc"] == pytest.approx(r["icc"])  # singleton group ignored
    assert np.isnan(icc_oneway([1.0, 2.0], ["a", "a"])["icc"])  # one group


def test_repeatability_sd_and_cv():
    v = [1.0, 2.0, 3.0, 10.0, 10.0, 5.0, np.nan]
    g = ["a", "a", "a", "b", "b", "c", "c"]  # c has one finite value -> dropped
    sd = repeatability_sd(v, g)
    assert sd["per_group"] == {"a": 1.0, "b": 0.0} and sd["median"] == 0.5 and sd["n_groups"] == 2
    cv = repeatability_cv(v, g)
    assert cv["per_group"] == {"a": 0.5, "b": 0.0} and cv["median"] == 0.25


def test_spearman_permutation_plain_and_stratified():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = x + rng.normal(size=40)
    r = spearman_permutation(x, y, n_perm=4999, seed=1)
    ref = ss.spearmanr(x, y)
    assert r["rho"] == pytest.approx(ref[0]) and r["p_value"] == pytest.approx(ref[1], abs=0.01)  # vs t-approximation
    assert r == spearman_permutation(x, y, n_perm=4999, seed=1)
    null = spearman_permutation(x, rng.normal(size=40), n_perm=999)
    assert 0.01 < null["p_value"] <= 1.0
    # confounding: between-stratum offsets make the pooled rho positive, within strata it is -1
    xs = np.r_[np.arange(10.0), np.arange(10.0) + 100]
    ys = np.r_[-np.arange(10.0), -np.arange(10.0) + 100]
    strata = np.repeat(["A", "B"], 10)
    assert spearman_permutation(xs, ys, n_perm=200)["rho"] > 0.4
    st = spearman_permutation(xs, ys, n_perm=999, strata=strata)
    assert st["rho"] == pytest.approx(-1.0) and st["p_value"] == pytest.approx(1 / 1000)
    nan_case = spearman_permutation(np.append(xs, 1.0), np.append(ys, np.nan), n_perm=10, strata=np.append(strata, "A"))
    assert nan_case["n"] == 20


def test_spearman_permutation_permutes_within_strata():
    # strata of sizes 2 and 5 with perfect within-stratum order: exact within-strata p = 2 / (2! 5!) = 1/120
    # (a pooled permutation would give 2 / 7! ~ 0.0004)
    x = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    r = spearman_permutation(x, x, n_perm=20000, seed=0, strata=list("AABBBBB"))
    assert r["rho"] == pytest.approx(1.0) and 0.005 < r["p_value"] < 0.012


def test_cluster_bootstrap_resamples_whole_clusters():
    sizes = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    df = pd.DataFrame({"c": np.repeat(list(sizes), list(sizes.values()))})
    df["row"] = np.arange(len(df))
    df = df.sample(frac=1.0, random_state=0)  # unsorted input

    def metric(d):  # every drawn cluster contributes all of its rows, each equally often
        ok = all(g.row.nunique() == sizes[c] and g.row.value_counts().nunique() == 1 for c, g in d.groupby("c"))
        return {"complete": float(ok), "rows": float(len(d))}

    r = cluster_bootstrap(metric, df, "c", n_boot=200, seed=5)
    assert r["n_clusters"] == 5 and r["complete"]["lo"] == r["complete"]["hi"] == 1.0
    assert r["rows"]["point"] == 15.0 and r["rows"]["lo"] < 15.0 < r["rows"]["hi"]
    assert r == cluster_bootstrap(metric, df, "c", n_boot=200, seed=5)  # deterministic
    scalar = cluster_bootstrap(lambda d: float(len(d)), df, "c", n_boot=50)
    assert set(scalar) == {"value", "n_clusters"} and set(scalar["value"]) == {"point", "lo", "hi", "se"}
    with pytest.raises(ValueError):
        cluster_bootstrap(metric, df.iloc[:0], "c")


def test_cluster_bootstrap_seed_resampling():
    df = pd.DataFrame({"c": np.tile(np.arange(10), 2), "seed": np.repeat([0, 1], 10), "y": np.repeat([0.0, 1.0], 10)})
    r = cluster_bootstrap(lambda d: {"m": d.y.mean(), "n_seeds": d.seed.nunique()}, df, "c", seed_col="seed",
                          n_boot=400, seed=0)
    assert r["m"]["point"] == 0.5  # mean over seeds
    assert r["n_seeds"]["lo"] == r["n_seeds"]["hi"] == 1.0  # the metric sees one seed at a time
    assert r["m"]["lo"] == 0.0 and r["m"]["hi"] == 1.0  # seed draws {0,0}, {0,1}, {1,1} -> 0, .5, 1


def test_cluster_bootstrap_ci_coverage():
    # 40 clusters x 5 rows, ICC 0.5: the cluster bootstrap CI of the mean should cover ~95 %, a row-level
    # bootstrap (each row its own cluster) should under-cover (design effect 3)
    rng = np.random.default_rng(123)
    n_sim, cover, cover_naive, se = 120, 0, 0, []
    for i in range(n_sim):
        df = pd.DataFrame({"c": np.repeat(np.arange(40), 5), "row": np.arange(200)})
        df["y"] = np.repeat(rng.normal(size=40), 5) + rng.normal(size=200)
        r = cluster_bootstrap(lambda d: d.y.mean(), df, "c", n_boot=300, seed=i)["value"]
        rn = cluster_bootstrap(lambda d: d.y.mean(), df, "row", n_boot=300, seed=i)["value"]
        cover += r["lo"] <= 0.0 <= r["hi"]
        cover_naive += rn["lo"] <= 0.0 <= rn["hi"]
        se.append(r["se"])
    assert 0.86 <= cover / n_sim <= 0.99
    assert cover_naive / n_sim < 0.85
    assert np.mean(se) == pytest.approx(np.sqrt(1.2 / 40), rel=0.15)  # true SE of the mean


def _base(n_c=20, m=5, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"c": np.repeat(np.arange(n_c), m), "key": np.arange(n_c * m)})
    df["y"] = rng.normal(size=len(df)) + 3.0
    return df


def test_paired_cluster_bootstrap_identical_and_shifted():
    a = _base()
    metric = lambda d: {"mean": d.y.mean()}  # noqa: E731
    same = paired_cluster_bootstrap(metric, a, a.copy(), "c", "key", n_boot=500)["mean"]
    assert same["diff"] == 0.0 and same["lo"] == same["hi"] == 0.0 and same["p_value"] == 1.0
    b = a.assign(y=a.y - 1.0)
    r = paired_cluster_bootstrap(metric, a, b, "c", "key", n_boot=500, seed=1)
    m = r["mean"]
    assert m["diff"] == pytest.approx(1.0) and m["lo"] > 0
    assert m["p_value"] == pytest.approx(2 / 501) and m["p_value"] > 0  # never 0
    assert m["rel_diff"] == pytest.approx(1.0 / b.y.mean()) and r["n_clusters"] == 20
    extra = pd.concat([b, pd.DataFrame({"c": [99], "key": [10**6], "y": [1e6]})], ignore_index=True)
    assert paired_cluster_bootstrap(metric, a, extra, "c", "key", n_boot=500, seed=1) == r  # unpaired keys ignored
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(metric, a, b.assign(key=b.key + 10**6), "c", "key")


def test_paired_cluster_bootstrap_seeds():
    base = _base()
    a = pd.concat([base.assign(seed=0), base.assign(seed=1, y=base.y + 2.0)], ignore_index=True)
    b = base.assign(seed=7)  # one seed: reused in every replicate
    r = paired_cluster_bootstrap(lambda d: d.y.mean(), a, b, "c", "key", seed_col="seed", n_boot=400)["value"]
    assert r["diff"] == pytest.approx(1.0)
    assert r["lo"] == pytest.approx(0.0, abs=1e-9) and r["hi"] == pytest.approx(2.0)  # a's seed draws: 0, 1, 2
    assert 0.2 < r["p_value"] <= 1.0


def test_cluster_bootstrap_seed_average_equals_np_average():
    """_evaluate's vectorised seed average gives exactly np.average over the drawn seeds, metric by metric."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"c": np.repeat(np.arange(8), 5).tolist() * 3, "seed": np.repeat([0, 1, 2], 40),
                       "v": rng.normal(size=120)})
    fn = lambda d: {"mean": d.v.mean(), "sd": d.v.std()}  # noqa: E731
    res = cluster_bootstrap(fn, df, "c", "seed", n_boot=50, seed=3)
    draws = np.random.default_rng(3)
    reps = []
    for _ in range(50):
        s, c = draws.integers(0, 3, 3), draws.integers(0, 8, 8)
        u, w = np.unique(s, return_counts=True)
        vals = [fn(pd.concat([df[(df.seed == k) & (df.c == ci)] for ci in c])) for k in u]
        reps.append({m: np.average([v[m] for v in vals], weights=w) for m in ("mean", "sd")})
    for m in ("mean", "sd"):
        lo, hi = np.quantile([r[m] for r in reps], [0.025, 0.975])
        assert res[m]["lo"] == pytest.approx(lo, abs=1e-12) and res[m]["hi"] == pytest.approx(hi, abs=1e-12)
