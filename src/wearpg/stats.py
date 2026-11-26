"""Inference statistics: cluster (+ seed) bootstrap, paired comparisons, multiplicity control, seed
summaries, repeatability (ICC, SD, CV) and permutation Spearman. Clusters = independent experimental units
(DESIGN sec. 0 rule 3); seeds = training seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

NAN = float("nan")


def _as_dict(v) -> dict[str, float]:
    return {k: float(x) for k, x in v.items()} if isinstance(v, dict) else {"value": float(v)}


def _indexer(df: pd.DataFrame, cluster_col: str, clusters: pd.Index, seed_col: str | None):
    """(n_seeds, rows) with rows(s, c) = row positions of seed index s (sorted seeds) in the drawn cluster
    indices c, duplicates kept. Per-(seed, cluster) rows are precomputed in a CSR layout so a replicate is
    one vectorized gather."""
    if df[cluster_col].isna().any():
        raise ValueError(f"{cluster_col!r} has missing values")
    k = len(clusters)
    cc = clusters.get_indexer(df[cluster_col])
    sc, seeds = (np.zeros(len(df), np.int64), [None]) if seed_col is None else pd.factorize(df[seed_col], sort=True)
    code = sc * k + cc
    order = np.argsort(code, kind="stable")
    counts = np.bincount(code, minlength=len(seeds) * k)
    starts = np.cumsum(counts) - counts

    def rows(s: int, c: np.ndarray) -> np.ndarray:
        cnt = counts[s * k + c]
        return order[np.repeat(starts[s * k + c] - (np.cumsum(cnt) - cnt), cnt) + np.arange(cnt.sum())]

    return len(seeds), rows


def _evaluate(metric_fn, df, rows, seed_draw, cluster_draw) -> dict[str, float]:
    """Metric of one replicate: per drawn seed on the drawn clusters, averaged over the seed draw (a seed drawn
    twice counts twice but is evaluated once)."""
    u, w = np.unique(seed_draw, return_counts=True)
    vals = [_as_dict(metric_fn(df.iloc[rows(s, cluster_draw)])) for s in u]
    keys = list(vals[0])
    a = np.array([[v.get(k, NAN) for k in keys] for v in vals], dtype=float)
    # = np.average(column, weights=w) for every key at once (same operations, one call instead of one per key)
    return dict(zip(keys, ((a * w[:, None]).sum(axis=0) / float(w.sum())).tolist()))


def _interval(reps: list[float], ci: float) -> tuple[float, float, float]:
    """Percentile CI (lo, hi) and SE of the finite replicates."""
    r = np.asarray(reps, dtype=float)
    r = r[np.isfinite(r)]
    if not r.size:
        return NAN, NAN, NAN
    lo, hi = np.quantile(r, [(1 - ci) / 2, (1 + ci) / 2])
    return float(lo), float(hi), float(r.std(ddof=1)) if r.size > 1 else NAN


def cluster_bootstrap(metric_fn, df: pd.DataFrame, cluster_col: str, seed_col: str | None = None,
                      n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Cluster bootstrap with optional seed resampling; one resampling loop for every metric of metric_fn.

    metric_fn(df) -> float | dict[str, float] (a float is reported under key "value"). Each replicate
    resamples clusters with replacement; with seed_col it also resamples seeds with replacement, computes the
    metric per drawn seed on the same drawn clusters and averages. point = metric on all data (mean over
    seeds); lo/hi = percentile interval; se = SD of replicates.
    Returns {metric: {"point", "lo", "hi", "se"}, "n_clusters": int}.
    """
    if df.empty:
        raise ValueError("cluster_bootstrap needs a non-empty df")
    clusters = pd.Index(pd.unique(df[cluster_col])).sort_values()
    k = len(clusters)
    n_seeds, rows = _indexer(df, cluster_col, clusters, seed_col)
    point = _evaluate(metric_fn, df, rows, np.arange(n_seeds), np.arange(k))
    rng = np.random.default_rng(seed)
    reps = [_evaluate(metric_fn, df, rows, rng.integers(0, n_seeds, n_seeds), rng.integers(0, k, k))
            for _ in range(n_boot)]
    out = {}
    for m, p in point.items():
        lo, hi, se = _interval([r.get(m, NAN) for r in reps], ci)
        out[m] = {"point": p, "lo": lo, "hi": hi, "se": se}
    out["n_clusters"] = k
    return out


def paired_cluster_bootstrap(metric_fn, df_a: pd.DataFrame, df_b: pd.DataFrame, cluster_col: str, key_col: str,
                             seed_col: str | None = None, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Paired cluster bootstrap of metric(a) - metric(b) on the items (key_col) present in BOTH frames.

    Each replicate draws ONE cluster resample used for a and b; seeds are resampled independently per model
    (a model with one seed reuses it). diff = point(a) - point(b); lo/hi percentile; rel_diff = diff / |point(b)|;
    p_value = min(1, 2 (1 + min(#{d* <= 0}, #{d* >= 0})) / (B + 1)) over the B finite replicates (never 0).
    Returns {metric: {"diff", "lo", "hi", "p_value", "rel_diff"}, "n_clusters": int}.
    """
    common = np.intersect1d(df_a[key_col].unique(), df_b[key_col].unique())
    if not common.size:
        raise ValueError("df_a and df_b share no keys")
    a, b = df_a[df_a[key_col].isin(common)], df_b[df_b[key_col].isin(common)]
    clusters = pd.Index(pd.unique(pd.concat([a[cluster_col], b[cluster_col]]))).sort_values()
    k = len(clusters)
    sa, rows_a = _indexer(a, cluster_col, clusters, seed_col)
    sb, rows_b = _indexer(b, cluster_col, clusters, seed_col)
    pa = _evaluate(metric_fn, a, rows_a, np.arange(sa), np.arange(k))
    pb = _evaluate(metric_fn, b, rows_b, np.arange(sb), np.arange(k))
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        c = rng.integers(0, k, k)
        ma = _evaluate(metric_fn, a, rows_a, rng.integers(0, sa, sa), c)
        mb = _evaluate(metric_fn, b, rows_b, rng.integers(0, sb, sb), c)
        reps.append({m: ma[m] - mb.get(m, NAN) for m in ma})
    out = {}
    for m in pa:
        diff = pa[m] - pb.get(m, NAN)
        d = np.asarray([r.get(m, NAN) for r in reps], dtype=float)
        d = d[np.isfinite(d)]
        lo, hi, _ = _interval(d, ci)
        p = min(1.0, 2 * (1 + min((d <= 0).sum(), (d >= 0).sum())) / (d.size + 1)) if d.size else NAN
        pbm = pb.get(m, NAN)
        rel = diff / abs(pbm) if np.isfinite(pbm) and pbm != 0 else NAN
        out[m] = {"diff": diff, "lo": lo, "hi": hi, "p_value": p, "rel_diff": rel}
    out["n_clusters"] = k
    return out


def _on_finite(pvals, fn) -> np.ndarray:
    """Apply an adjustment to the finite p-values only (NaN stays NaN, m = number of finite p)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, NAN)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = fn(p[ok])
    return out


def _holm(q: np.ndarray) -> np.ndarray:
    m = q.size
    o = np.argsort(q, kind="stable")
    adj = np.empty(m)
    adj[o] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * q[o]))
    return adj


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (FWER), original order."""
    return _on_finite(pvals, _holm)


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR), original order (scipy false_discovery_control)."""
    return _on_finite(pvals, lambda q: stats.false_discovery_control(q, method="bh"))


def adjust(pvals, method: str) -> np.ndarray:
    """Multiplicity adjustment; method "holm" | "bh" (config eval.multiplicity)."""
    fns = {"holm": holm, "bh": benjamini_hochberg}
    if method not in fns:
        raise ValueError(f"method must be one of {tuple(fns)}, got {method!r}")
    return fns[method](pvals)


def seed_summary(values, ci: float = 0.95) -> dict:
    """Mean, sd (ddof 1), t-interval lo/hi and n of the finite values (e.g. one metric over seeds)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    mean = float(v.mean()) if n else NAN
    sd = float(v.std(ddof=1)) if n > 1 else NAN
    half = float(stats.t.ppf((1 + ci) / 2, n - 1) * sd / np.sqrt(n)) if n > 1 else NAN
    return {"mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half, "n": int(n)}


def _repeats(values, groups) -> pd.Series:
    """Finite values indexed by group, restricted to groups with >= 2 values."""
    s = pd.Series(np.asarray(values, dtype=float), index=pd.Index(np.asarray(groups, dtype=object)))
    s = s[np.isfinite(s.to_numpy())]
    return s[s.groupby(level=0).transform("size").to_numpy() >= 2]


def icc_oneway(values, groups, ci: float = 0.95) -> dict:
    """One-way random-effects ICC(1) (Shrout & Fleiss 1979, ICC(1,1)) over groups with >= 2 values.

    k = (N - sum n_i^2 / N) / (a - 1) (mean group size for unbalanced data); F = MSB / MSW; F-based CI
    (McGraw & Wong 1996). Returns icc, lo, hi, k, F, p_value, n_groups, n.
    """
    v = _repeats(values, groups)
    g = v.groupby(level=0)
    n_i = g.size().to_numpy(float)
    a, n = len(n_i), n_i.sum()
    out = {"icc": NAN, "lo": NAN, "hi": NAN, "k": NAN, "F": NAN, "p_value": NAN, "n_groups": int(a), "n": int(n)}
    if a < 2 or n - a < 1:
        return out
    msb = (n_i * (g.mean().to_numpy() - v.mean()) ** 2).sum() / (a - 1)
    msw = ((v - g.transform("mean")) ** 2).sum() / (n - a)
    k = (n - (n_i**2).sum() / n) / (a - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = msb / msw
        fl = f / stats.f.ppf((1 + ci) / 2, a - 1, n - a)
        fu = f * stats.f.ppf((1 + ci) / 2, n - a, a - 1)
        out.update(icc=float((msb - msw) / (msb + (k - 1) * msw)), lo=float((fl - 1) / (fl + k - 1)),
                   hi=float((fu - 1) / (fu + k - 1)), k=float(k), F=float(f), p_value=float(stats.f.sf(f, a - 1, n - a)))
    return out


def repeatability_sd(values, groups) -> dict:
    """Per-group SD (ddof 1) over repeats (groups with >= 2 finite values) and their median.
    Returns {"per_group": {group: sd}, "median", "n_groups"}."""
    sd = _repeats(values, groups).groupby(level=0).std(ddof=1)
    return {"per_group": sd.to_dict(), "median": float(sd.median()) if len(sd) else NAN, "n_groups": int(len(sd))}


def repeatability_cv(values, groups) -> dict:
    """Per-group coefficient of variation SD / mean (ratio-scale values, e.g. k, not log10 k) and the median.
    Returns {"per_group": {group: cv}, "median", "n_groups"}."""
    g = _repeats(values, groups).groupby(level=0)
    cv = g.std(ddof=1) / g.mean()
    return {"per_group": cv.to_dict(), "median": float(cv.median()) if len(cv) else NAN, "n_groups": int(len(cv))}


def spearman_permutation(x, y, n_perm: int = 5000, seed: int = 0, strata=None) -> dict:
    """Spearman rho with a two-sided permutation p-value (1 + #{|rho*| >= |rho|}) / (n_perm + 1).

    With strata: ranks are computed WITHIN each stratum and centred on the stratum mean rank, rho = Pearson
    of these centred ranks, and y is permuted within strata (removes between-stratum confounding, e.g.
    material). Without strata this is the ordinary Spearman rho. Non-finite pairs are dropped.
    Returns {"rho", "p_value", "n"} (NaN for n < 3 or no within-stratum variation).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    s = np.zeros(len(x), np.int64) if strata is None else pd.factorize(pd.Series(np.asarray(strata, dtype=object)))[0]
    ok = np.isfinite(x) & np.isfinite(y) & (s >= 0)
    x, y, s = x[ok], y[ok], s[ok]
    n = len(x)

    def centred_ranks(v):
        r = pd.Series(v).groupby(s).rank()
        return (r - r.groupby(s).transform("mean")).to_numpy()

    xc, yc = centred_ranks(x), centred_ranks(y)
    den = np.sqrt((xc**2).sum() * (yc**2).sum())
    if n < 3 or den == 0:
        return {"rho": NAN, "p_value": NAN, "n": int(n)}
    rho = float(xc @ yc / den)
    rng = np.random.default_rng(seed)
    x_grouped = xc[np.argsort(s, kind="stable")]  # positions grouped by stratum, original order inside
    hits, chunk = 0, max(1, 2_000_000 // n)  # bound memory to ~2e6 floats per chunk
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        perm = np.argsort(s[None, :] + rng.random((m, n)), axis=1)  # stratum-grouped, shuffled inside strata
        hits += int((np.abs(yc[perm] @ x_grouped / den) >= abs(rho) - 1e-12).sum())
    return {"rho": rho, "p_value": (1 + hits) / (n_perm + 1), "n": int(n)}
