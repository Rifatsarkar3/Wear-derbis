"""POST-HOC analyses of the NASA spur-gear oil-debris result (NASAPLUS). The plan and every verdict rule were written to
outputs/agent_logs/NASAPLUS.md before any statistic here was computed; section letters A-K below are those of the plan.
Frozen pre-specified results (outputs/open_data/oil/oil_results.json) are only read, for a reproduction assertion.

Data: transcribed tables in wearpg.open_oil (Dempsey, NASA/TM-2003-211307 and TM-2001-210936) plus the end masses and
particle counts of NASA/TM-2004-212883 Tables 4-5 (below). Unit of inference = experiment.
Writes outputs/open_data/oil/posthoc/*.csv + results.json (deterministic: same files on every run).
Usage: .venv/Scripts/python.exe -I scripts/open_oil_posthoc.py   (about 70 s, most of it the ordinal permutations)
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # -I drops the cwd

from wearpg import open_oil as oo  # noqa: E402
from wearpg.stats import cluster_bootstrap, holm, icc_oneway, spearman_permutation  # noqa: E402
from wearpg.utils import load_json, save_json  # noqa: E402

OIL = ROOT / "outputs" / "open_data" / "oil"
OUT = OIL / "posthoc"
B, N_PERM, N_PERM_FIT, N_PERM_ORDINAL = 10_000, 199_999, 19_999, 999
CONFIRMED = (1, 2, 3, 4, 5, 6, 8, 21, 24)  # pitting recorded at an inspection before completion (Table 3.9, p. 49)
INTERCEPT_ONLY = ("mean", "mean_log", "median")
FROZEN_MAE = {"time_only": 3.610833081265054, "debris_only": 2.3170952389737502,
              "debris_plus_time": 2.2450319944482193}  # oil_results.json nasa.loo_regression_n_destructive_teeth
LIMITS_MG = (5.453, 8.69, 15.475)  # membership-function limits, TM-2003 p. 38 and Table 4.3 p. 84 (spur column)
RATE_LIMIT = 0.005  # mg/min, TM-2000-210371 p. 6 eq. (5) and p. 7

# Dempsey, Lewicki, Decker, NASA/TM-2004-212883: Table 4 p. 5 (final reading with damage) and Table 5 p. 6 (no damage);
# born-digital, read from the rendered pages. Re-processed masses: exps 5 and 6 differ from TM-2003.
TM2004_END = """exp|mass_mg|counts
1|36.1|211
2|26.3|129
3|14.1|117
4|26.3|221
5|11.5|160
6|23.1|271
7|3.4|30
8|19.1|132
9|2.4|18
10|5.5|46
11|0.4|4
12|2.3|26
13|3.2|47
14|0.00|0
15|0.13|4
16|0.10|3
17|0.06|1
"""


# ------------------------------------------------------------------------------------------------ exact tests, CIs
def signflip_p(d) -> float:
    """Exact two-sided sign-flip p of the mean of paired differences (all 2^n sign vectors, meet in the middle)."""
    d = np.asarray(d, float)

    def sums(v):
        s = np.zeros(1)
        for x in v:
            s = np.concatenate([s + x, s - x])
        return s

    a, b = sums(d[: len(d) // 2]), np.sort(sums(d[len(d) // 2:]))
    t = abs(d.sum()) - 1e-9 * max(1.0, np.abs(d).sum())
    if t <= 0:
        return 1.0
    hits = (len(b) - np.searchsorted(b, t - a, "left")).sum() + np.searchsorted(b, -t - a, "right").sum()
    return float(hits / 2 ** len(d))


def sign_test_p(d) -> float:
    d = np.asarray(d, float)
    return float(sps.binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue)


def boot_index(n: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, n, (B, n))


def pct(reps) -> dict:
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def _pearson_rows(a, b):
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def spearman(x, y) -> dict:
    """Spearman rho, experiment-bootstrap percentile CI and two-sided permutation p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = boot_index(len(x))
    reps = _pearson_rows(sps.rankdata(x[i], axis=1), sps.rankdata(y[i], axis=1))
    p = spearman_permutation(x, y, n_perm=N_PERM, seed=0)
    return {"rho": p["rho"], **pct(reps), "p_perm": p["p_value"], "n": p["n"]}


def auc_boot(score, pos) -> dict:
    """AUC of score for pos (bool), experiment-bootstrap CI, exact two-sided Mann-Whitney p."""
    s, pos = np.asarray(score, float), np.asarray(pos, bool)
    i = boot_index(len(s))
    r, pb = sps.rankdata(s[i], axis=1), pos[i]
    n1 = pb.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = ((r * pb).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (pos.size - n1))
    p = sps.mannwhitneyu(s[pos], s[~pos], method="exact").pvalue
    return {"auc": oo.auc(s[~pos], s[pos]), **pct(reps), "p_exact": float(p), "n_pos": int(pos.sum()),
            "n_neg": int((~pos).sum())}


# ------------------------------------------------------------------------------------------------ experiment sets
def experiments() -> pd.DataFrame:
    ex = oo.nasa_experiments()
    return pd.DataFrame({"exp": ex.exp, "run_min": ex.rdg_end, "mass_mg": ex.mass_end_mg, "n_de": ex.n_de_end,
                         "n_pitted": ex.n_pitted_end, "stage": ex.stage_end})


def merge_16_17(df: pd.DataFrame) -> pd.DataFrame:
    """TM-2001 Table 4 p. 7 lists one test (exp 16: 21446 readings, 0.163 mg) that TM-2003 Tables 3.7 / 3.9 split into
    exp 16 (380, 0.099 mg) and exp 17 (21066, 0.064 mg): the sums agree exactly."""
    a, b = (df[df.exp == e].iloc[0] for e in (16, 17))
    if int(a.run_min + b.run_min) != 21446 or abs(a.mass_mg + b.mass_mg - 0.163) > 1e-9:
        raise ValueError("experiments 16 + 17 do not add up to TM-2001 experiment 16")
    out = df[df.exp != 17].copy()
    out.loc[out.exp == 16, ["run_min", "mass_mg"]] = [21446.0, 0.163]
    return out


def experiment_sets() -> dict[str, pd.DataFrame]:
    s0 = experiments()
    s1 = merge_16_17(s0)
    s2 = s1[s1.exp != 11]
    confirmed = lambda d: d[d.exp.isin(CONFIRMED) | d.exp.between(9, 17)]
    sets = {"S0_all_24": s0, "S1_16_17_merged": s1, "S2_S1_without_11": s2, "S3_confirmed_plus_no_damage": confirmed(s0),
            "S3m_confirmed_merged": confirmed(s2)}
    return {k: v.reset_index(drop=True) for k, v in sets.items()}


# ------------------------------------------------------------------------------------------------ A / F: LOO baselines
def loo_linear(X, y) -> np.ndarray:
    """Leave-one-out OLS predictions (intercept added) through the hat matrix; X may have no columns, y may be (n, P)."""
    A = np.column_stack([np.ones(len(y)), X])
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return y - resid / (1 - h if resid.ndim == 1 else (1 - h)[:, None])


def loo_predictions(df: pd.DataFrame, fwd=np.log1p, inv=np.expm1, oracle: bool = True) -> pd.DataFrame:
    """LOO predictions [teeth] of the intercept-only, time, debris and debris + time models (OLS on fwd(n_de) against
    log1p features, back-transformed and clipped at 0 = the frozen models) and of the stage oracle (not feasible)."""
    y, n = df.n_de.to_numpy(float), len(df)
    f = np.log1p(df[["run_min", "mass_mg"]].to_numpy(float))
    teeth = lambda X: np.clip(inv(loo_linear(X, fwd(y))), 0, None)
    loo = lambda fn, keep: np.array([fn(y[keep & (np.arange(n) != i)]) for i in range(n)])
    out = pd.DataFrame({"exp": df.exp, "n_de": y, "mean": loo(np.mean, y >= 0), "mean_log": teeth(np.empty((n, 0))),
                        "median": loo(np.median, y >= 0), "time_only": teeth(f[:, :1]), "debris_only": teeth(f[:, 1:]),
                        "debris_plus_time": teeth(f[:, [1, 0]])})
    if oracle:
        out["oracle_stage"] = np.where(y > 0, loo(np.median, y > 0), 0.0)
        out["oracle_in_sample"] = np.where(y > 0, np.median(y[y > 0]), 0.0)
    out["constant_in_sample"] = np.median(y)  # optimistic (uses the held-out value); the LOO median jumps in the gap 1-3
    return out


def mae_table(pred: pd.DataFrame) -> dict[str, float]:
    return {m: float((pred[m] - pred.n_de).abs().mean()) for m in pred.columns.drop(["exp", "n_de"])}


def best_intercept(mae: dict[str, float]) -> str:
    return min(INTERCEPT_ONLY, key=mae.get)


def paired(pred: pd.DataFrame, a: str, b: str) -> dict:
    """Paired difference of LOO absolute errors a - b: mean, experiment-bootstrap CI, exact sign-flip p."""
    d = ((pred[a] - pred.n_de).abs() - (pred[b] - pred.n_de).abs()).to_numpy()
    return {"a": a, "b": b, "diff": float(d.mean()), **pct(d[boot_index(len(d))].mean(1)), "p_signflip": signflip_p(d)}


def baseline_block(df: pd.DataFrame, **kw) -> dict:
    pred = loo_predictions(df, **kw)
    mae = mae_table(pred)
    base = best_intercept(mae)
    pairs = [("debris_only", base), ("debris_only", "time_only"), ("debris_plus_time", base),
             ("debris_plus_time", "debris_only"), ("time_only", base)]
    pairs += [("debris_only", "constant_in_sample")]
    pairs += [("debris_only", o) for o in ("oracle_stage", "oracle_in_sample") if o in pred]
    return {"n": len(df), "mae": mae, "best_intercept_only": base,
            "skill_vs_best_intercept": {m: 1 - v / mae[base] for m, v in mae.items()},
            "paired": {f"{a}_minus_{b}" if b != base else f"{a}_minus_best_intercept": paired(pred, a, b)
                       for a, b in pairs}, "_pred": pred}


# ------------------------------------------------------------------------------------------------ B: permutation tests
def permutation_loo_mae(df: pd.DataFrame) -> dict:
    """One-sided permutation p of the debris-only LOO MAE (mass permuted over the experiments)."""
    y, lm = df.n_de.to_numpy(float), np.log1p(df.mass_mg.to_numpy(float))
    mae = lambda x: np.abs(np.clip(np.expm1(loo_linear(x[:, None], np.log1p(y))), 0, None) - y).mean()
    rng, obs = np.random.default_rng(0), mae(lm)
    perm = np.array([mae(rng.permutation(lm)) for _ in range(N_PERM_FIT)])
    return {"observed_mae": float(obs), "null_mean_mae": float(perm.mean()), "null_q025": float(np.quantile(perm, 0.025)),
            "null_q975": float(np.quantile(perm, 0.975)), "p_one_sided": float((1 + (perm <= obs).sum()) / (N_PERM_FIT + 1))}


def freedman_lane(df: pd.DataFrame) -> dict:
    """Debris beyond run time: residuals of log1p(n_de) on log1p(run time) permuted; statistic = LOO MAE(debris + time) -
    LOO MAE(time) [teeth], one-sided (smaller = debris helps)."""
    ly = np.log1p(df.n_de.to_numpy(float))
    lt, lm = np.log1p(df.run_min.to_numpy(float)), np.log1p(df.mass_mg.to_numpy(float))
    Xt, Xf = lt[:, None], np.column_stack([lm, lt])

    def stat(Y):
        c = np.expm1(Y)
        err = lambda X: np.abs(np.clip(np.expm1(loo_linear(X, Y)), 0, None) - c).mean(0)
        return err(Xf) - err(Xt)

    A = np.column_stack([np.ones(len(ly)), Xt])
    fit = A @ np.linalg.lstsq(A, ly, rcond=None)[0]
    rng = np.random.default_rng(0)
    perm = np.array([rng.permutation(ly - fit) for _ in range(N_PERM_FIT)]).T + fit[:, None]
    obs, null = float(stat(ly[:, None])[0]), stat(perm)
    return {"observed_diff": obs, "null_mean": float(null.mean()),
            "p_one_sided": float((1 + (null <= obs + 1e-12).sum()) / (N_PERM_FIT + 1))}


def stage_respecting(df: pd.DataFrame) -> dict:
    """G1: permute mass only among destructive tests (n_de is constant 0 elsewhere): null distribution of the overall
    Spearman given the stage contrast; two-sided p around the null mean."""
    de = (df.n_de > 0).to_numpy()
    rm, ry = sps.rankdata(df.mass_mg), sps.rankdata(df.n_de)
    ry = ry - ry.mean()
    den = np.sqrt(((rm - rm.mean()) ** 2).sum() * (ry**2).sum())
    obs = float((rm - rm.mean()) @ ry / den)
    rng = np.random.default_rng(0)
    fixed = float((rm - rm.mean())[~de] @ ry[~de])
    vals, w = (rm - rm.mean())[de], ry[de]
    null = []
    for start in range(0, N_PERM, 20_000):
        m = min(20_000, N_PERM - start)
        null.append((fixed + vals[np.argsort(rng.random((m, de.sum())), axis=1)] @ w) / den)
    null = np.concatenate(null)
    dev = abs(obs - null.mean())
    return {"observed_rho": obs, "null_mean_rho": float(null.mean()), "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "p_two_sided": float((1 + (np.abs(null - null.mean()) >= dev - 1e-12).sum()) / (N_PERM + 1)),
            "n_permuted": int(de.sum())}


# ------------------------------------------------------------------------------------------------ C: hurdle, ordinal
FEATURES = {"none": [], "time_only": ["run_min"], "debris_only": ["mass_mg"], "debris_plus_time": ["mass_mg", "run_min"]}


def hurdle_loo(df: pd.DataFrame, feats) -> pd.DataFrame:
    """LOO hurdle: L2 logistic P(destructive) x OLS of ln(n_de) among destructive training tests (log1p features)."""
    y, X = df.n_de.to_numpy(float), np.log1p(df[list(feats)].to_numpy(float))
    rows = []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        if len(feats):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], y[tr] > 0)
            p = float(m.predict_proba(X[i:i + 1])[0, 1])
        else:
            p = float((y[tr] > 0).mean())
        tp = tr & (y > 0)
        A = np.column_stack([np.ones(tp.sum()), X[tp]])
        mu = float(np.exp(np.concatenate([[1.0], X[i]]) @ np.linalg.lstsq(A, np.log(y[tp]), rcond=None)[0]))
        rows.append((df.exp.iloc[i], y[i], p, mu if p >= 0.5 else 0.0, p * mu))
    return pd.DataFrame(rows, columns=["exp", "n_de", "p_de", "pred_median_type", "pred_expectation"])


def fit_proportional_odds(X, z, lam: float = 1.0):
    """Proportional-odds logit, P(z <= k) = expit(theta_k - x beta), ridge 0.5 lam |beta|^2 (standardised x).
    Returns predict(Xnew) -> cumulative probabilities P(z <= 0), P(z <= 1) for 3 ordered levels."""
    mu, sd = X.mean(0), X.std(0)
    Xs, k = (X - mu) / sd, X.shape[1]

    def cdf(par, Xq):
        th = np.array([par[0], par[0] + np.exp(par[1])])
        return expit(th[None, :] - (Xq @ par[2:])[:, None])

    def nll(par):
        c = np.column_stack([np.zeros(len(z)), cdf(par, Xs), np.ones(len(z))])
        p = np.clip(c[np.arange(len(z)), z + 1] - c[np.arange(len(z)), z], 1e-12, None)
        return -np.log(p).sum() + 0.5 * lam * (par[2:] ** 2).sum()

    f0 = np.clip(np.array([(z <= 0).mean(), (z <= 1).mean()]), 0.02, 0.98)
    th0 = np.log(f0 / (1 - f0))
    par = minimize(nll, np.concatenate([[th0[0], np.log(max(th0[1] - th0[0], 0.1))], np.zeros(k)]), method="BFGS").x
    return lambda Xq: cdf(par, (Xq - mu) / sd)


def ordinal_levels(n_de) -> np.ndarray:
    n = np.asarray(n_de)
    return np.where(n == 0, 0, np.where(n <= 4, 1, 2))


def rps(cum, z) -> np.ndarray:
    """Ranked probability score per observation (3 levels, normalised by 2)."""
    obs = np.column_stack([z <= 0, z <= 1]).astype(float)
    return ((cum - obs) ** 2).sum(1) / 2


def ordinal_loo(X, z) -> np.ndarray:
    """LOO cumulative probabilities (n, 2); X with no columns = class frequencies of the training tests."""
    out = np.empty((len(z), 2))
    for i in range(len(z)):
        tr = np.arange(len(z)) != i
        out[i] = (fit_proportional_odds(X[tr], z[tr])(X[i:i + 1])[0] if X.shape[1]
                  else [(z[tr] <= 0).mean(), (z[tr] <= 1).mean()])
    return out


def alternative_models(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """C: hurdle and ordinal alternatives, LOO, with paired comparisons against their intercept-only versions."""
    z, y = ordinal_levels(df.n_de), df.n_de.to_numpy(float)
    X = {k: np.log1p(df[f].to_numpy(float)) for k, f in FEATURES.items()}
    hur = {k: hurdle_loo(df, f) for k, f in FEATURES.items()}
    cum = {k: ordinal_loo(X[k], z) for k in FEATURES}
    score = {k: rps(c, z) for k, c in cum.items()}
    ae = {k: (h.pred_median_type - y).abs().to_numpy() for k, h in hur.items()}
    rows = [{"model": "hurdle", "features": k, "loo_mae_teeth": float(ae[k].mean()),
             "loo_mae_expectation": float((h.pred_expectation - y).abs().mean()),
             "brier_destructive": float(((h.p_de - (y > 0)) ** 2).mean())} for k, h in hur.items()]
    rows += [{"model": "ordinal", "features": k, "loo_rps": float(s.mean()),
              "loo_accuracy": float((np.column_stack([c[:, 0], c[:, 1] - c[:, 0], 1 - c[:, 1]]).argmax(1) == z).mean())}
             for (k, s), c in zip(score.items(), cum.values())]
    cmp_ = {}
    for name, per in (("hurdle_mae", ae), ("ordinal_rps", score)):
        for a, b in (("debris_only", "none"), ("debris_only", "time_only"), ("debris_plus_time", "debris_only"),
                     ("time_only", "none")):
            d = per[a] - per[b]
            cmp_[f"{name}|{a}_minus_{b}"] = {"diff": float(d.mean()), **pct(d[boot_index(len(d))].mean(1)),
                                             "p_signflip": signflip_p(d)}
    rng, obs = np.random.default_rng(0), float(score["debris_only"].mean())
    perm = np.array([rps(ordinal_loo(rng.permutation(X["debris_only"]), z), z).mean() for _ in range(N_PERM_ORDINAL)])
    res = {"levels": {"0": int((z == 0).sum()), "1-4": int((z == 1).sum()), ">=5": int((z == 2).sum())},
           "table": rows, "paired": cmp_,
           "ordinal_permutation_debris": {"observed_rps": obs, "null_mean_rps": float(perm.mean()),
                                          "p_one_sided": float((1 + (perm <= obs).sum()) / (N_PERM_ORDINAL + 1))}}
    loo = pd.concat([h.assign(model="hurdle", features=k) for k, h in hur.items()])
    return res, loo


# ------------------------------------------------------------------------------------------------ H: longitudinal
def run_in_rows() -> dict[int, tuple[float, float]]:
    t = oo._table(oo.NASA_TM2003_READINGS)
    r = t[t.description == "Run-in Wear"]
    return {int(e): (float(a), float(m)) for e, a, m in zip(r.exp, r.reading, r.mass_mg)}


def intervals(exp5: str = "table_3_9", origin: str = "primary") -> pd.DataFrame:
    """Intervals between consecutive loaded inspections of one experiment. origin 'primary': a run-in row with mass 0
    is the origin, one with mass > 0 (exps 1, 2) is itself an inspection after origin (0, 0); no tabulated row -> (60 min,
    0 mg). 'f6': run-in rows dropped, origin mass 0 at the run-in reading (scripts/diag_nasa_onset.py)."""
    ri = run_in_rows()
    rows = []
    for e, g in oo.nasa_readings(exp5_rdg2566=exp5).groupby("exp"):
        t0, m0 = ri.get(e, (60.0, 0.0))
        pts = list(g[["reading", "mass_mg", "n_de"]].itertuples(index=False, name=None))
        if origin == "primary" and m0 > 0:
            pts, t0 = [(t0, m0, 0), *pts], 0.0
        prev = (t0, 0.0, 0)
        for k, p in enumerate(pts):
            rows.append({"exp": int(e), "t0": prev[0], "t1": p[0], "m0": prev[1], "m1": p[1], "n0": int(prev[2]),
                         "n1": int(p[2]), "completion": k == len(pts) - 1})
            prev = p
    iv = pd.DataFrame(rows)
    iv["dt"], iv["dm"], iv["dn"] = iv.t1 - iv.t0, iv.m1 - iv.m0, iv.n1 - iv.n0
    if (iv.dt <= 0).any() or (iv.dm < 0).any() or (iv.dn < 0).any():
        raise ValueError("inspections not monotone")
    iv["rate"], iv["progress"], iv["destructive"] = iv.dm / iv.dt, iv.dn > 0, iv.n1 > 0
    return iv


def _strata(iv: pd.DataFrame, label: str) -> list[tuple[int, pd.DataFrame]]:
    """Experiments with both kinds of interval (the only ones a within-experiment contrast can use)."""
    return [(e, g) for e, g in iv.groupby("exp") if g[label].nunique() == 2]


def exact_conditional_logistic(iv: pd.DataFrame, label: str = "progress", col: str = "rate") -> dict:
    """Conditional logistic regression of label on ln(col) with experiment strata: conditional MLE, profile-likelihood CI
    and the exact two-sided p of the sufficient statistic over all label arrangements within strata."""
    st = _strata(iv, label)
    if not st:
        return {"n_experiments": 0}
    sums, obs = [], 0.0
    for _, g in st:
        x, lab = np.log(g[col].to_numpy(float)), g[label].to_numpy(bool)
        sums.append(np.array([x[list(c)].sum() for c in itertools.combinations(range(len(x)), int(lab.sum()))]))
        obs += x[lab].sum()
    half = np.argsort([len(s) for s in sums])[::-1]
    group = [np.zeros(1), np.zeros(1)]
    for j in half:  # meet in the middle over the strata
        k = int(len(group[0]) > len(group[1]))
        group[k] = (group[k][:, None] + sums[j][None, :]).ravel()
    a, b = group[0], np.sort(group[1])
    centre = sum(s.mean() for s in sums)
    dev = abs(obs - centre) - 1e-9
    total = len(a) * len(b)
    hits = (len(b) - np.searchsorted(b, centre + dev - a, "left")).sum() + np.searchsorted(b, centre - dev - a,
                                                                                         "right").sum()
    ll = lambda beta: beta * obs - sum(logsumexp(beta * s) for s in sums)
    opt = minimize_scalar(lambda v: -ll(v), bounds=(-30, 30), method="bounded")
    beta, top = float(opt.x), float(-opt.fun)
    crit = lambda v: ll(v) - (top - sps.chi2.ppf(0.95, 1) / 2)
    lo = brentq(crit, -60, beta) if crit(-60) < 0 else float("-inf")
    hi = brentq(crit, beta, 60) if crit(60) < 0 else float("inf")
    return {"n_experiments": len(st), "experiments": [int(e) for e, _ in st], "n_intervals": int(sum(len(g) for _, g in st)),
            "n_label_intervals": int(sum(g[label].sum() for _, g in st)), "beta_ln": beta,
            "or_per_doubling": 2.0**beta, "or_lo": 2.0**lo, "or_hi": 2.0**hi,
            "n_arrangements": int(total), "p_exact": float(min(1.0, hits / total))}


def conditional_logistic_lr(iv: pd.DataFrame, label: str = "progress") -> dict:
    """Conditional logistic with ln(dm) and ln(dt): LR test of ln(dm) beyond elapsed time (asymptotic)."""
    comb, obs = [], np.zeros(2)
    for _, g in _strata(iv, label):
        x = np.log(g[["dm", "dt"]].to_numpy(float))
        lab = g[label].to_numpy(bool)
        comb.append(np.array([x[list(c)].sum(0) for c in itertools.combinations(range(len(x)), int(lab.sum()))]))
        obs += x[lab].sum(0)
    ll = lambda b: b @ obs - sum(logsumexp(c @ b) for c in comb)
    full = minimize(lambda b: -ll(b), np.zeros(2), method="BFGS")
    time = minimize(lambda b: -ll(np.array([0.0, b[0]])), np.zeros(1), method="BFGS")
    lr = max(0.0, 2 * (time.fun - full.fun))
    return {"beta_ln_dm": float(full.x[0]), "beta_ln_dt": float(full.x[1]), "or_dm_per_doubling": 2.0 ** float(full.x[0]),
            "lr_dm_given_dt": float(lr), "p_chi2": float(sps.chi2.sf(lr, 1))}


def within_contrast(iv: pd.DataFrame, label: str) -> dict:
    """Per experiment: mean ln rate (label) - mean ln rate (not label); exact sign-flip and sign test across experiments."""
    d = {int(e): float(np.log(g.rate[g[label]]).mean() - np.log(g.rate[~g[label]]).mean()) for e, g in _strata(iv, label)}
    v = np.array(list(d.values()))
    if not len(v):
        return {"n_experiments": 0}
    return {"n_experiments": len(v), "per_experiment_ln_ratio": d, "n_positive": int((v > 0).sum()),
            "geometric_mean_ratio": float(np.exp(v.mean())), **{f"ratio_{k}": float(np.exp(x)) for k, x in
                                                               pct(v[boot_index(len(v))].mean(1)).items()},
            "p_signflip": signflip_p(v), "p_sign_test": sign_test_p(v)}


def post_vs_before(iv: pd.DataFrame) -> dict:
    """H2: rate after the first inspection with destructive pitting over the rate from the origin to it, tests run on."""
    d, rows = {}, []
    for e, g in iv.groupby("exp"):
        k = np.flatnonzero(g.destructive.to_numpy())
        if not len(k) or k[0] == len(g) - 1:
            continue
        g = g.reset_index(drop=True)
        pre, post = g.iloc[: k[0] + 1], g.iloc[k[0] + 1:]
        rb, ra = pre.dm.sum() / pre.dt.sum(), post.dm.sum() / post.dt.sum()
        d[int(e)] = float(np.log(ra / rb))
        rows.append({"exp": int(e), "first_destructive_reading": float(pre.t1.iloc[-1]), "rate_before_mg_min": float(rb),
                     "rate_after_mg_min": float(ra), "ratio": float(ra / rb)})
    v = np.array(list(d.values()))
    return {"n_experiments": len(v), "per_experiment": rows, "n_positive": int((v > 0).sum()),
            "geometric_mean_ratio": float(np.exp(v.mean())),
            **{f"ratio_{k}": float(np.exp(x)) for k, x in pct(v[boot_index(len(v))].mean(1)).items()},
            "p_signflip": signflip_p(v), "p_sign_test": sign_test_p(v)}


def longitudinal(iv: pd.DataFrame) -> dict:
    """H1, H2 (primary) and the secondary H2b, H3, two-covariate model and all-interval AUC for one interval table."""
    if set(iv.exp[iv.rate <= 0]) - {14}:  # exp 14 completed at 0 mg; it has one interval, so no within-experiment role
        raise ValueError("zero debris increment outside experiment 14")
    pos = iv[iv.rate > 0]
    multi = iv[iv.groupby("exp").exp.transform("size") >= 2]
    h3 = {c: spearman_permutation(multi[c], multi.dn, n_perm=N_PERM, seed=0, strata=multi.exp) for c in ("rate", "dm")}
    au = cluster_bootstrap(lambda d: oo.auc(d.loc[~d.progress, "rate"], d.loc[d.progress, "rate"])
                           if d.progress.nunique() == 2 else float("nan"), iv, "exp", n_boot=2000, seed=0)["value"]
    return {"n_intervals": len(iv), "n_progress": int(iv.progress.sum()), "n_experiments": int(iv.exp.nunique()),
            "H1_exact_conditional_logistic_rate": exact_conditional_logistic(pos),
            "H1_within_contrast_progress": within_contrast(pos, "progress"),
            "H2_after_vs_before_first_destructive": post_vs_before(iv),
            "H2b_within_contrast_destructive_state": within_contrast(pos, "destructive"),
            "H3_stratified_spearman_vs_new_teeth": h3, "conditional_logistic_dm_and_dt": conditional_logistic_lr(pos),
            "auc_rate_progress_intervals": au,
            "exploratory_added_after_first_run": {c: exact_conditional_logistic(pos, col=c) for c in ("dm", "dt")}}


# ------------------------------------------------------------------------------------------------ G3 / G4, I, J, K
def first_observation(exp5: str = "table_3_9") -> pd.DataFrame:
    """G3: state at the first inspection with destructive pitting (destructive tests), else at completion."""
    rd, ex = oo.nasa_readings(exp5_rdg2566=exp5), experiments()
    f = rd[rd.n_de > 0].groupby("exp").first()
    out = ex.copy()
    for e, r in f.iterrows():
        out.loc[out.exp == e, ["run_min", "mass_mg", "n_de"]] = [r.reading, r.mass_mg, r.n_de]
    return out.astype({"n_de": int})


def stop_rules(iv: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """G4: is one fixed debris rule consistent with the recorded stops of the destructive tests (end satisfies it, no
    earlier loaded inspection of that test did)? Interval mean rates stand in for the report's step-change rate."""
    rules = {**{f"mass>={m:g}": (lambda d, m=m: d.m1 >= m) for m in LIMITS_MG},
             f"rate>={RATE_LIMIT:g}": lambda d: d.rate >= RATE_LIMIT}
    rows = []
    for e, g in iv.groupby("exp"):
        for name, rule in rules.items():
            hit = rule(g).to_numpy()
            rows.append({"exp": int(e), "destructive_at_end": bool(g.destructive.iloc[-1]), "rule": name,
                         "end_mass_mg": float(g.m1.iloc[-1]), "last_interval_rate": float(g.rate.iloc[-1]),
                         "end_satisfies": bool(hit[-1]), "n_earlier_inspections": len(g) - 1,
                         "n_earlier_satisfying": int(hit[:-1].sum()),
                         "consistent": bool(hit[-1] and not hit[:-1].any())})
    t = pd.DataFrame(rows)
    d = t[t.destructive_at_end]
    summ = {r: {"n_tests": len(g), "n_consistent": int(g.consistent.sum()),
                "n_end_below_rule": int((~g.end_satisfies).sum()),
                "n_run_on_past_rule": int((g.n_earlier_satisfying > 0).sum()),
                "exps_end_below_rule": g.exp[~g.end_satisfies].tolist(),
                "exps_run_on_past_rule": g.exp[g.n_earlier_satisfying > 0].tolist()} for r, g in d.groupby("rule", sort=False)}
    return summ, t


def lognormal_summary(v, clusters) -> dict:
    """Geometric mean / SD, range, CV, max/min of positive values; bootstrap CIs over `clusters` (experiments)."""
    df = pd.DataFrame({"v": np.asarray(v, float), "c": np.asarray(clusters)})

    def fn(d):
        lv = np.log(d.v)
        return {"geometric_mean": float(np.exp(lv.mean())), "geometric_sd": float(np.exp(lv.std(ddof=1))),
                "cv": float(d.v.std(ddof=1) / d.v.mean())}

    bs = cluster_bootstrap(fn, df, "c", n_boot=B, seed=0)
    return {"n": len(df), "n_clusters": bs["n_clusters"], "min": float(df.v.min()), "median": float(df.v.median()),
            "max": float(df.v.max()), "max_over_min": float(df.v.max() / df.v.min()),
            **{k: bs[k]["point"] for k in ("geometric_mean", "geometric_sd", "cv")},
            **{f"{k}_{e}": bs[k][e] for k in ("geometric_mean", "geometric_sd", "cv") for e in ("lo", "hi")}}


def yields(iv: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """I: debris mass per destructively pitted tooth at completion (D14) and per newly pitted tooth between inspections."""
    d = experiments()
    d = d[d.n_de > 0].assign(mg_per_destructive_tooth=lambda t: t.mass_mg / t.n_de,
                             mg_per_pitted_tooth=lambda t: t.mass_mg / t.n_pitted)
    lx, ly = np.log(d.n_de.to_numpy(float)), np.log(d.mass_mg.to_numpy(float))
    slope = lambda a, b: np.polyfit(a, b, 1)[0] if np.ptp(a) > 0 else np.nan
    reps = np.array([slope(lx[j], ly[j]) for j in boot_index(len(lx))])
    p = iv[iv.progress].assign(mg_per_new_tooth=lambda t: t.dm / t.dn)
    fit = sps.linregress(lx, ly)
    res = {"end_per_destructive_tooth": lognormal_summary(d.mg_per_destructive_tooth, d.exp),
           "end_per_any_pitted_tooth": lognormal_summary(d.mg_per_pitted_tooth, d.exp),
           "interval_per_new_destructive_tooth": lognormal_summary(p.mg_per_new_tooth, p.exp),
           "loglog_slope_mass_on_teeth": {"slope": float(fit.slope), **pct(reps), "r2": float(fit.rvalue**2),
                                          "n": len(lx), "note": "proportional yield = slope 1"},
           "icc_ln_interval_yield_by_experiment": icc_oneway(np.log(p.mg_per_new_tooth), p.exp),
           "rule_of_thumb_9mg_per_tooth": {"share_at_or_above": float((d.mg_per_destructive_tooth >= 9).mean()),
                                           "ratio_min": float(d.mg_per_destructive_tooth.min() / 9),
                                           "ratio_max": float(d.mg_per_destructive_tooth.max() / 9)}}
    for c in ("run_min", "mass_mg", "n_de"):
        res[f"spearman_yield_vs_{c}"] = spearman_permutation(d.mg_per_destructive_tooth, d[c], n_perm=N_PERM, seed=0)
    return res, d, p


def repeatability() -> pd.DataFrame:
    """J: the only group at one nominal torque (96 N m; TM-2000-210371 p. 2 and 7): gear sets 1-2 (exps 7, 8; TM-2003 p.
    25) and the three undamaged tests of 350 / 357 / 497 h (1204 h in total = exps 9, 15 and 16 + 17)."""
    d = merge_16_17(experiments()).set_index("exp").loc[[7, 8, 9, 15, 16]].reset_index()
    d["test"] = ["gear set 1", "gear set 2", "no damage", "no damage", "no damage (16 + 17)"]
    return d.assign(run_h=d.run_min / 60)[["exp", "test", "run_min", "run_h", "mass_mg", "n_de", "n_pitted"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"label": "post_hoc (plan and rules: outputs/agent_logs/NASAPLUS.md); frozen verdicts unchanged",
                 "bootstrap": B, "permutations": N_PERM, "permutations_refit": N_PERM_FIT}
    sets = experiment_sets()
    s0 = sets["S0_all_24"]

    # --- A baselines, D associations
    blocks = {k: baseline_block(d) for k, d in sets.items()}
    frozen = load_json(OIL / "oil_results.json")["nasa"]["loo_regression_n_destructive_teeth"]
    for m, v in FROZEN_MAE.items():  # reproduction of the frozen models by the hat-matrix LOO
        if abs(blocks["S0_all_24"]["mae"][m] - v) > 1e-9 or abs(frozen[m]["loo_mae_teeth"] - v) > 1e-12:
            raise AssertionError(f"frozen LOO MAE of {m} not reproduced: {blocks['S0_all_24']['mae'][m]} vs {v}")
    pd.concat([b["_pred"].assign(set=k) for k, b in blocks.items()]).to_csv(OUT / "baselines_loo_predictions.csv", index=False)
    pd.DataFrame([{"set": k, "n": b["n"], "model": m, "loo_mae_teeth": v,
                   "skill_vs_best_intercept": b["skill_vs_best_intercept"][m],
                   "best_intercept_only": b["best_intercept_only"]} for k, b in blocks.items()
                  for m, v in b["mae"].items()]).to_csv(OUT / "baselines_summary.csv", index=False)
    pd.DataFrame([{"set": k, "contrast": c, **p} for k, b in blocks.items() for c, p in b["paired"].items()]).to_csv(
        OUT / "baselines_paired.csv", index=False)
    res["A_baselines"] = {k: {kk: v for kk, v in b.items() if kk != "_pred"} for k, b in blocks.items()}

    assoc = {k: {"spearman": spearman(d.mass_mg, d.n_de), "auc_destructive": auc_boot(d.mass_mg, d.n_de > 0),
                 "spearman_run_time": spearman(d.run_min, d.n_de)} for k, d in sets.items()}
    d14 = s0[s0.n_de > 0].reset_index(drop=True)
    assoc["D14_destructive"] = {"spearman": spearman(d14.mass_mg, d14.n_de),
                                "spearman_run_time": spearman(d14.run_min, d14.n_de)}
    res["D_associations"] = assoc
    pd.DataFrame([{"set": k, "statistic": s, **v} for k, a in assoc.items() for s, v in a.items()]).to_csv(
        OUT / "associations.csv", index=False)

    # --- B permutation tests, G1
    res["B_permutation_loo_mae_debris_only"] = permutation_loo_mae(s0)
    res["B_freedman_lane_debris_beyond_time"] = freedman_lane(s0)
    res["G1_stage_respecting_permutation"] = stage_respecting(s0)

    # --- C alternative models
    res["C_alternative_models"], loo = alternative_models(s0)
    loo.to_csv(OUT / "hurdle_loo_predictions.csv", index=False)
    pd.DataFrame(res["C_alternative_models"]["table"]).to_csv(OUT / "alternative_models.csv", index=False)

    # --- E influence
    rows = []
    for e in s0.exp:
        d = s0[s0.exp != e].reset_index(drop=True)
        b = mae_table(loo_predictions(d, oracle=False))
        sub = d[d.n_de > 0]
        rows.append({"dropped_exp": int(e), "spearman": float(sps.spearmanr(d.mass_mg, d.n_de)[0]),
                     "auc": oo.auc(d.mass_mg[d.n_de == 0], d.mass_mg[d.n_de > 0]),
                     "mae_debris_only": b["debris_only"], "mae_best_intercept": b[best_intercept(b)],
                     "mae_diff": b["debris_only"] - b[best_intercept(b)],
                     "spearman_destructive_only": float(sps.spearmanr(sub.mass_mg, sub.n_de)[0])})
    inf = pd.DataFrame(rows)
    inf.to_csv(OUT / "influence_leave_one_out.csv", index=False)
    res["E_influence"] = {c: {"min": float(inf[c].min()), "min_exp": int(inf.dropped_exp[inf[c].idxmin()]),
                              "max": float(inf[c].max()), "max_exp": int(inf.dropped_exp[inf[c].idxmax()])}
                          for c in ("spearman", "auc", "mae_diff", "spearman_destructive_only")}

    # --- F destructive tests alone
    fb = baseline_block(d14, fwd=np.log, inv=np.exp, oracle=False)
    fb.pop("_pred").to_csv(OUT / "destructive_only_loo_predictions.csv", index=False)
    res["F_destructive_only"] = fb

    # --- G stopping rule
    res["G3_first_observation"] = {}
    for v in ("table_3_9", "table_3_14"):
        g3 = first_observation(v)
        b = baseline_block(g3)
        b.pop("_pred")
        res["G3_first_observation"][v] = {"spearman": spearman(g3.mass_mg, g3.n_de),
                                          "auc_destructive": auc_boot(g3.mass_mg, g3.n_de > 0), **b}
        g3.to_csv(OUT / f"first_observation_{v}.csv", index=False)
    iv = intervals()
    iv.to_csv(OUT / "intervals.csv", index=False)
    res["G4_stop_rule_consistency"], t = stop_rules(iv)
    t.to_csv(OUT / "stop_rule_consistency.csv", index=False)

    # --- H longitudinal
    variants = {"primary": iv, "exp5_table_3_14": intervals(exp5="table_3_14"), "origin_f6": intervals(origin="f6"),
                "completion_excluded": iv[~iv.completion]}
    res["H_longitudinal"] = {k: longitudinal(v) for k, v in variants.items()}
    pd.DataFrame([{"variant": k, "n_intervals": h["n_intervals"], "n_progress": h["n_progress"],
                   "h1_or_per_doubling": h["H1_exact_conditional_logistic_rate"].get("or_per_doubling"),
                   "h1_p_exact": h["H1_exact_conditional_logistic_rate"].get("p_exact"),
                   "h1_n_experiments": h["H1_exact_conditional_logistic_rate"]["n_experiments"],
                   "h1_contrast_ratio": h["H1_within_contrast_progress"].get("geometric_mean_ratio"),
                   "h1_contrast_p_signflip": h["H1_within_contrast_progress"].get("p_signflip"),
                   "h2_ratio": h["H2_after_vs_before_first_destructive"]["geometric_mean_ratio"],
                   "h2_p_signflip": h["H2_after_vs_before_first_destructive"]["p_signflip"],
                   "h2_n_positive": h["H2_after_vs_before_first_destructive"]["n_positive"],
                   "h2_n_experiments": h["H2_after_vs_before_first_destructive"]["n_experiments"]}
                  for k, h in res["H_longitudinal"].items()]).to_csv(OUT / "longitudinal_summary.csv", index=False)

    # --- I, J, K
    res["I_yield"], d, p = yields(iv)
    d.to_csv(OUT / "yield_per_tooth_end.csv", index=False)
    p.to_csv(OUT / "yield_per_new_tooth_intervals.csv", index=False)
    rep = repeatability()
    rep.to_csv(OUT / "repeatability_96Nm.csv", index=False)
    res["J_repeatability_96Nm"] = rep.to_dict("records")
    new = oo._table(TM2004_END).astype(float)
    k = s0.merge(new.rename(columns={"mass_mg": "mass_tm2004", "counts": "counts_tm2004"}), on="exp", how="left")
    k["mass_hybrid"] = k.mass_tm2004.fillna(k.mass_mg)
    first17 = k[k.exp <= 17]
    res["K_measurement"] = {
        "tm2004_mass_exps_1_17_rest_tm2003": {"spearman": spearman(k.mass_hybrid, k.n_de),
                                             "auc_destructive": auc_boot(k.mass_hybrid, k.n_de > 0)},
        "particle_counts_exps_1_17": {"spearman": spearman(first17.counts_tm2004, first17.n_de),
                                      "auc_destructive": auc_boot(first17.counts_tm2004, first17.n_de > 0)},
        "mass_tm2003_exps_1_17": {"spearman": spearman(first17.mass_mg, first17.n_de),
                                  "auc_destructive": auc_boot(first17.mass_mg, first17.n_de > 0)}}
    k.to_csv(OUT / "measurement_robustness.csv", index=False)

    # --- verdicts (rules of the plan) and Holm over the six primary post-hoc p-values
    a, h = res["A_baselines"]["S0_all_24"], res["H_longitudinal"]["primary"]
    pa = a["paired"]["debris_only_minus_best_intercept"]
    h1, h2 = h["H1_exact_conditional_logistic_rate"], h["H2_after_vs_before_first_destructive"]
    prim = {"A_debris_vs_best_intercept": pa["p_signflip"], "F_D14_spearman": assoc["D14_destructive"]["spearman"]["p_perm"],
            "G2_S3_spearman": assoc["S3_confirmed_plus_no_damage"]["spearman"]["p_perm"],
            "G3_first_observation_spearman": res["G3_first_observation"]["table_3_9"]["spearman"]["p_perm"],
            "H1_exact_conditional_logistic": h1["p_exact"], "H2_after_vs_before": h2["p_signflip"]}
    res["primary_p"] = {k: {"p": v, "p_holm": float(q)} for (k, v), q in zip(prim.items(), holm(list(prim.values())))}
    pd.DataFrame([{"test": k, **v} for k, v in res["primary_p"].items()]).to_csv(OUT / "primary_p_holm.csv", index=False)
    va = ("STRENGTHENED" if pa["diff"] < 0 and pa["hi"] < 0 and pa["p_signflip"] < 0.05
          else "WEAKENED" if pa["diff"] >= 0 else "INCONCLUSIVE")
    f = fb["paired"]["debris_only_minus_best_intercept"]
    c = res["C_alternative_models"]["paired"]
    subsets = [(assoc[k], res["A_baselines"][k]) for k in ("S3_confirmed_plus_no_damage", "S3m_confirmed_merged")]
    subsets.append((res["G3_first_observation"]["table_3_9"], res["G3_first_observation"]["table_3_9"]))
    ok_g = [s["spearman"]["lo"] > 0 and s["auc_destructive"]["auc"] >= 0.9
            and b["paired"]["debris_only_minus_best_intercept"]["diff"] < 0 for s, b in subsets]
    vg = "ROBUST" if all(ok_g) else "NOT ROBUST" if any(s["spearman"]["lo"] <= 0 for s, _ in subsets) else "PARTLY"
    s1, s2 = h1["p_exact"] < 0.05 and h1["or_per_doubling"] > 1, h2["p_signflip"] < 0.05 and h2["geometric_mean_ratio"] > 1
    vh = "SUPPORT" if s1 and s2 else "PARTIAL" if s1 or s2 else "NONE"
    res["verdicts"] = {
        "V-A": va, "V-A2_time_only_is_null": bool(a["mae"]["time_only"] >= a["mae"][a["best_intercept_only"]]),
        "V-C": "debris beats intercept-only in both" if c["hurdle_mae|debris_only_minus_none"]["hi"] < 0
        and c["ordinal_rps|debris_only_minus_none"]["hi"] < 0 else "depends on the model",
        "V-E": "robust" if res["E_influence"]["spearman"]["min"] >= 0.75 and res["E_influence"]["mae_diff"]["max"] < 0
        else "not robust",
        "V-F": "SUPPORTED" if assoc["D14_destructive"]["spearman"]["p_perm"] < 0.05 and f["diff"] < 0 and f["hi"] < 0
        else "NOT SUPPORTED", "V-G": vg, "V-H": vh,
        "overall": "STRENGTHENED" if va == "STRENGTHENED" and vg == "ROBUST" and vh != "NONE"
        else "WEAKENED" if va == "WEAKENED" or vg == "NOT ROBUST" else "MIXED"}
    save_json(res, OUT / "results.json")
    print(res["verdicts"])


if __name__ == "__main__":
    main()
