"""Published experiments that pair a debris quantity with measured wear: schema, loader and the small-sample statistics used to
test the aggregate conversion of DESIGN sec. 3 (wear = debris / (observed fraction x capture efficiency)), i.e. in log10
    log10 W = 1 * log10 D + b        one rig, one instrument, ONE calibrated constant b.

Data: data/real/paired/<id>/paired.csv (one row per paired observation) + study.json (pre-specified conditions, resolved
predictor, sensitivities). Plan and verdict rules: outputs/agent_logs/PAIRED.md; results: docs/paired_data_results.md.
All of it is post hoc / exploratory. Unit of inference = independent test (column `unit`).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REQUIRED = ("study", "set", "unit", "row", "order", "mode", "rig", "material_pair", "lubricant", "debris", "debris_metric",
            "debris_unit", "debris_type", "wear", "wear_metric", "wear_unit", "resolved", "debris_2", "wear_2", "debris_err",
            "wear_err", "page", "item", "extraction", "note")
MODES = ("interval", "cumulative", "series", "rate")
DEBRIS_TYPES = ("mass", "volume", "count", "index", "concentration", "share", "class")
MIN_UNITS = 5  # fewer independent paired units -> descriptive only (plan, part 1)
B_BOOT, N_PERM, MAX_EXACT, SEED = 10_000, 199_999, 9, 0


# -------------------------------------------------------------------------------------------------------- schema, loader
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check a paired.csv frame; optional columns are cond_* (operating conditions), class_* (particle classes) and alt_* (alternative
    debris or wear measures). Raises ValueError on any violation."""
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"missing columns {missing}")
    extra = [c for c in df if c not in REQUIRED and not c.startswith(("cond_", "class_", "alt_"))]
    if extra:
        raise ValueError(f"unknown columns {extra} (use cond_, class_ or alt_ prefixes)")
    if df.duplicated(["set", "row"]).any():
        raise ValueError("duplicated (set, row)")
    bad = sorted(set(df["mode"]) - set(MODES)) + sorted(set(df["debris_type"]) - set(DEBRIS_TYPES))
    if bad:
        raise ValueError(f"unknown mode or debris_type {bad}")
    if (df["wear"] < 0).any() or (df["debris"] < 0).any() or df["wear"].isna().any():
        raise ValueError("negative or missing wear, or negative debris")
    if df[["page", "item", "unit"]].isna().any().any():
        raise ValueError("provenance (page, item) and unit are mandatory")
    return df


def load_studies(root) -> list[dict]:
    """Every data/real/paired/<id>/ with paired.csv + study.json, sorted by id: study.json plus `data` (validated frame)."""
    out = []
    for f in sorted(Path(root).glob("*/paired.csv")):
        spec = json.loads((f.parent / "study.json").read_text(encoding="utf-8"))
        df = validate(pd.read_csv(f, dtype={"page": str, "item": str, "unit": str, "row": str, "note": str}))
        unknown = set(df["set"]) ^ set(spec["sets"])
        if unknown:
            raise ValueError(f"{f.parent.name}: sets of paired.csv and study.json differ: {sorted(unknown)}")
        out.append({**spec, "data": df})
    return out


def select(df: pd.DataFrame, debris: str = "debris", wear: str = "wear", unit: str = "unit", mode: str | None = None, drop=(), include=(),
           exclude=(), fallback: bool = False) -> pd.DataFrame:
    """Rows usable on a log scale with columns D, W, u added. `exclude` = primary exclusions, `include` re-admits some of them."""
    d = df[~df["unit"].isin(set(drop) | (set(exclude) - set(include)))]
    if mode:
        d = d[d["mode"] == mode]
    D, W = d[debris].astype(float), d[wear].astype(float)
    if fallback:
        D, W = D.fillna(d["debris"]), W.fillna(d["wear"])
    d = d.assign(D=D, W=W, u=d[unit])
    return d[(d["D"] > 0) & (d["W"] > 0)].reset_index(drop=True)


def unit_means(d: pd.DataFrame, cols=("D", "W")) -> pd.DataFrame:
    """One row per unit: mean log10 of each column (named as the column)."""
    return np.log10(d.set_index("u")[list(cols)]).groupby(level=0, sort=False).mean()


# ------------------------------------------------------------------------------------------------------ exact inference
def permutation_index(n: int) -> tuple[np.ndarray, bool]:
    """All n! relabellings (identity first) if n <= MAX_EXACT, else the identity plus N_PERM random ones. -> (index, exact)."""
    if n <= MAX_EXACT:
        return np.array(list(itertools.permutations(range(n)))), True
    rng = np.random.default_rng(SEED)
    return np.vstack([np.arange(n), np.argsort(rng.random((N_PERM, n)), axis=1)]), False


def boot_index(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, (B_BOOT, n))


def _pct(reps) -> tuple[float, float]:
    reps = np.asarray(reps, float)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5]) if np.isfinite(reps).any() else (math.nan, math.nan)
    return float(lo), float(hi)


def slope(x, y, alpha: float = 0.05, reference: float = 1.0) -> dict:
    """OLS slope of y on x with (a) the exact permutation CI (test inversion: e(b) = y - b x is exchangeable relative to x; statistic
    |sum (x - mean) e|), (b) its p at `reference` and at 0, (c) t interval, (d) unit bootstrap, (e) errors-in-variables bracket
    (inverse regression, reduced major axis). n <= MAX_EXACT is enumerated."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    xc, yc = x - x.mean(), y - y.mean()
    sxx, sxy, syy = float(xc @ xc), float(xc @ yc), float(yc @ yc)
    b = sxy / sxx
    perm, exact = permutation_index(n)
    A, B = yc[perm] @ xc, xc[perm] @ xc
    B[np.abs(B - sxx) < 1e-12 * sxx] = sxx
    B[np.abs(B + sxx) < 1e-12 * sxx] = -sxx
    A[np.abs(A - sxy) < 1e-12 * math.sqrt(sxx * syy + 1e-300)] = sxy
    with np.errstate(divide="ignore", invalid="ignore"):
        r1, r2 = (sxy - A) / (sxx - B), (sxy + A) / (sxx + B)
    same = np.isnan(r1) & (A == sxy)  # relabellings that leave x and y as they are can never be rejected
    lo, hi = np.fmin(r1, r2), np.fmax(r1, r2)
    lo[same], hi[same] = -np.inf, np.inf
    k = int(math.floor(alpha * len(perm) + 1e-9))
    tol = 1e-9 * (abs(sxy) + sxx)
    p_at = lambda b0: float((np.abs(A - b0 * B) >= abs(sxy - b0 * sxx) - tol).mean())
    resid = yc - b * xc
    df = n - 2
    se = math.sqrt(max(resid @ resid, 0.0) / df / sxx) if df > 0 else math.nan
    tq = sps.t.ppf(1 - alpha / 2, df) if df > 0 else math.nan
    i = boot_index(n)
    xb, yb = x[i] - x[i].mean(1, keepdims=True), y[i] - y[i].mean(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        reps = (xb * yb).sum(1) / (xb * xb).sum(1)
    reps[(xb * xb).sum(1) < 1e-12 * sxx] = np.nan
    blo, bhi = _pct(reps)
    return {"n": n, "slope": b, "perm_lo": float(np.sort(lo)[k]), "perm_hi": float(np.sort(hi)[::-1][k]), "perm_exact": exact,
            "n_perm": len(perm), "p_slope_eq_ref": p_at(reference), "p_slope_eq_0": p_at(0.0), "t_lo": b - tq * se, "t_hi": b + tq * se,
            "boot_lo": blo, "boot_hi": bhi, "inverse_slope": syy / sxy if sxy != 0 else math.nan,
            "rma_slope": math.copysign(math.sqrt(syy / sxx), sxy), "r": sxy / math.sqrt(sxx * syy) if syy > 0 else math.nan,
            "intercept": float(y.mean() - b * x.mean()), "resid_sd": math.sqrt(max(resid @ resid, 0.0) / df) if df > 0 else math.nan}


def spearman(x, y) -> dict:
    """Spearman rho, exact (n <= MAX_EXACT) or Monte-Carlo permutation p (two-sided), unit-bootstrap percentile CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rx, ry = sps.rankdata(x), sps.rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt((rx @ rx) * (ry @ ry))
    if den == 0:
        return {"n": n, "rho": math.nan, "p": math.nan, "exact": False, "boot_lo": math.nan, "boot_hi": math.nan}
    rho = float(rx @ ry / den)
    perm, exact = permutation_index(n)
    p = float((np.abs(ry[perm] @ rx) >= abs(rho) * den - 1e-9).mean())
    i = boot_index(n)
    a, b = sps.rankdata(x[i], axis=1), sps.rankdata(y[i], axis=1)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        reps = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    lo, hi = _pct(reps)
    return {"n": n, "rho": rho, "p": p, "exact": exact, "boot_lo": lo, "boot_hi": hi}


def signflip_p(d) -> float:
    """Two-sided sign-flip p of the mean of d: all 2^n sign vectors up to n = 16, else N_PERM random ones plus the identity."""
    d = np.asarray(d, float)
    if len(d) <= 16:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    else:
        signs = np.vstack([np.ones(len(d)), np.random.default_rng(SEED).choice((1.0, -1.0), (N_PERM, len(d)))])
    s = np.abs(signs @ d)
    return float((s >= abs(d.sum()) - 1e-12 * max(1.0, np.abs(d).sum())).mean())


def mean_diff(per_unit) -> dict:
    """Mean of per-unit differences: bootstrap CI over units and sign-flip p."""
    d = np.asarray(per_unit, float)
    lo, hi = _pct(d[boot_index(len(d))].mean(1))
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "p": signflip_p(d), "n_units": len(d)}


# ---------------------------------------------------------------------------------------------------------- prediction
def conditions_matrix(d: pd.DataFrame, conditions, n_units: int) -> tuple[np.ndarray, list[str]]:
    """Design columns of the pre-specified conditions [[column, 'log10' | 'id'], ...]; constant columns are dropped and the list is cut to
    (n_units - 2) // 2 entries (plan, part 2)."""
    cols, names = [], []
    for name, tr in conditions:
        v = np.log10(d[name].astype(float)) if tr == "log10" else d[name].astype(float)
        if np.ptp(v.to_numpy()) > 0:
            cols.append(v.to_numpy()), names.append(name)
    keep = max((n_units - 2) // 2, 0)
    return (np.column_stack(cols[:keep]) if cols[:keep] else np.empty((len(d), 0))), names[:keep]


def loo_predictions(x, y, Z, unit) -> pd.DataFrame:
    """Leave-one-unit-out predictions of y = log10 W from x = log10 D and condition columns Z.
    M0 mean; M1 conditions; M2 = x + one constant; M3 = x + conditions acting on y - x; M2f / M3f free slope of x; R-models are run by
    passing the resolved predictor as x. NaN where a training fold has fewer rows than parameters."""
    x, y, Z, unit = np.asarray(x, float), np.asarray(y, float), np.asarray(Z, float).reshape(len(y), -1), np.asarray(unit)
    one = np.ones((len(y), 1))
    designs = {"M0": (one, 0 * x), "M1": (np.hstack([one, Z]), 0 * x), "M2": (one, x), "M3": (np.hstack([one, Z]), x),
               "M2f": (np.column_stack([one, x]), 0 * x), "M3f": (np.column_stack([one, x, Z]), 0 * x)}
    out = {m: np.full(len(y), np.nan) for m in designs}
    for u in pd.unique(unit):
        te = unit == u
        for m, (X, off) in designs.items():
            if (~te).sum() >= X.shape[1]:
                beta = np.linalg.lstsq(X[~te], (y - off)[~te], rcond=None)[0]
                out[m][te] = X[te] @ beta + off[te]
    return pd.DataFrame(out)


def unit_errors(pred: pd.DataFrame, y, unit) -> pd.DataFrame:
    """Mean absolute log10 error per unit and model."""
    return (pred.sub(np.asarray(y, float), axis=0).abs()).groupby(np.asarray(unit), sort=False).mean()


# ------------------------------------------------------------------------------------------------------------ verdicts
def verdict_slope(lo: float, hi: float) -> str:
    """V1 (plan): contains 1 and excludes 0 -> for_us; excludes 1 -> against_us; else inconclusive."""
    if not lo <= 1 <= hi:
        return "against_us"
    if lo > 0:
        return "for_us strong" if lo >= 0.5 and hi <= 2 else "for_us"
    return "inconclusive"


def verdict_spearman(rho: float, lo: float, p: float) -> str:
    return "for_us" if lo > 0 and p < 0.05 else "against_us" if rho <= 0 else "inconclusive"


def verdict_diff(diff: float, hi: float) -> str:
    """V2-V5 (plan): error(a) - error(b): < 0 with upper CI bound < 0 -> for_us; point estimate > 0 -> against_us; else inconclusive."""
    if not np.isfinite(diff):
        return "not testable"
    return "for_us" if diff < 0 and hi < 0 else "against_us" if diff > 0 else "inconclusive"


def verdict_gsd(gsd: float, n_sets: int) -> str:
    """V6: between-set geometric SD of the conversion factor: <= 1.25 for_us, >= 2 against_us."""
    return "not testable" if n_sets < 3 else "for_us" if gsd <= 1.25 else "against_us" if gsd >= 2 else "inconclusive"


def conversion(ratio_log10) -> dict:
    """Geometric mean, geometric SD, range of D / W from its log10 values."""
    r = np.asarray(ratio_log10, float)
    return {"n": len(r), "gm": float(10 ** r.mean()), "gsd": float(10 ** r.std(ddof=1)) if len(r) > 1 else math.nan,
            "min": float(10 ** r.min()), "max": float(10 ** r.max()), "max_over_min": float(10 ** np.ptp(r))}
