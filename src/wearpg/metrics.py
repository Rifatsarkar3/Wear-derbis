"""Evaluation metrics: classification, calibration, ordinal severity, regression, transition timing, volume
consistency. Macro averages run over the classes present in y_true; y_true < 0 (unlabelled) is ignored."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from wearpg.schema import MECHANISMS, SEVERITY_LEVELS

NAN = float("nan")


def _labelled(y_true, proba) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=float)
    return y[y >= 0], p[y >= 0]


def reliability_curve(y_true, proba, n_bins: int = 15) -> dict:
    """Top-label reliability over equal-width confidence bins (lo, hi] (confidence 0 -> first bin).

    Returns arrays (n_bins,): bin_lo, bin_hi, confidence (mean max-prob), accuracy, count (NaN if empty).
    """
    y, p = _labelled(y_true, proba)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    b = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    count = np.bincount(b, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.bincount(b, conf, minlength=n_bins) / count
        acc = np.bincount(b, correct, minlength=n_bins) / count
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return {"bin_lo": edges[:-1], "bin_hi": edges[1:], "confidence": mean_conf, "accuracy": acc, "count": count}


def expected_calibration_error(y_true, proba, n_bins: int = 15) -> float:
    """Top-label ECE = sum_b (n_b / n) |acc_b - conf_b| (Guo et al. 2017). NaN if no labelled rows."""
    rc = reliability_curve(y_true, proba, n_bins)
    n = rc["count"].sum()
    ok = rc["count"] > 0
    return float((rc["count"][ok] * np.abs(rc["accuracy"][ok] - rc["confidence"][ok])).sum() / n) if n else NAN


def classification_metrics(y_true, proba, classes=MECHANISMS, ece_bins: int = 15) -> dict:
    """Mechanism (or any) classification metrics from probabilities (n, C); prediction = argmax.

    macro_f1 / balanced_accuracy average over classes present in y_true (== sklearn f1_score(labels=present,
    average="macro", zero_division=0) / balanced_accuracy_score); brier = mean_i sum_c (p_ic - 1[y_i=c])^2
    (multi-class, in [0, 2]); log_loss = mean -ln max(p_iy, 1e-15); per_class {cls: sensitivity, specificity,
    f1, support} (NaN where undefined); confusion (C, C) int array, rows = true, cols = predicted.
    """
    y, p = _labelled(y_true, proba)
    n, k = len(y), len(classes)
    cm = np.bincount(y * k + p.argmax(axis=1), minlength=k * k).reshape(k, k)  # fast in bootstrap loops
    tp = np.diag(cm).astype(float)
    fn, fp = cm.sum(axis=1) - tp, cm.sum(axis=0) - tp
    tn = n - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        sens, spec, f1 = tp / (tp + fn), tn / (tn + fp), 2 * tp / (2 * tp + fp + fn)
    present = (tp + fn) > 0
    return {
        "macro_f1": float(f1[present].mean()) if n else NAN,
        "balanced_accuracy": float(sens[present].mean()) if n else NAN,
        "accuracy": float(tp.sum() / n) if n else NAN,
        "ece": expected_calibration_error(y, p, ece_bins),
        "brier": float(((p - np.eye(k)[y]) ** 2).sum(axis=1).mean()) if n else NAN,
        "log_loss": float(-np.log(np.clip(p[np.arange(n), y], 1e-15, 1.0)).mean()) if n else NAN,
        "n": int(n),
        "per_class": {c: {"sensitivity": float(sens[i]), "specificity": float(spec[i]), "f1": float(f1[i]),
                          "support": int(tp[i] + fn[i])} for i, c in enumerate(classes)},
        "confusion": cm,
    }


def metrics_by_group(y_true, proba, groups, classes=MECHANISMS) -> pd.DataFrame:
    """Scalar classification metrics per group, long format: columns group, metric, value, n
    (metrics macro_f1, balanced_accuracy, accuracy, ece, brier, log_loss; groups sorted, NaN groups dropped)."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=float)
    rows = []
    for g, idx in pd.Series(np.arange(len(y))).groupby(np.asarray(groups, dtype=object)).groups.items():
        m = classification_metrics(y[idx], p[idx], classes)
        rows += [{"group": g, "metric": k, "value": m[k], "n": m["n"]}
                 for k in ("macro_f1", "balanced_accuracy", "accuracy", "ece", "brier", "log_loss")]
    return pd.DataFrame(rows, columns=["group", "metric", "value", "n"])


def severity_metrics(y_true, proba) -> dict:
    """Ordinal severity (0 mild, 1 transition, 2 severe) from (n, 3) probabilities; prediction = argmax.
    macro_f1, balanced_accuracy, ordinal_mae (mean |pred - true| in levels), qwk (quadratic-weighted
    Cohen kappa with fixed labels [0, 1, 2]; NaN if undefined), n."""
    y, p = _labelled(y_true, proba)
    m = classification_metrics(y, p, SEVERITY_LEVELS)
    pred = p.argmax(axis=1)
    defined = len(set(y.tolist()) | set(pred.tolist())) >= 2  # else expected disagreement = 0 -> 0/0
    return {
        "macro_f1": m["macro_f1"],
        "balanced_accuracy": m["balanced_accuracy"],
        "ordinal_mae": float(np.abs(pred - y).mean()) if len(y) else NAN,
        "qwk": _qwk(m["confusion"]) if defined else NAN,
        "n": m["n"],
    }


def _qwk(cm: np.ndarray) -> float:
    """Quadratic-weighted Cohen kappa from a confusion matrix (rows true, cols predicted): the exact operations of
    sklearn.metrics.cohen_kappa_score(weights="quadratic") without its input validation (bootstrap hot path)."""
    sum0, sum1 = np.sum(cm, axis=0), np.sum(cm, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)
    w = np.zeros(cm.shape, dtype=int) + np.arange(cm.shape[0])
    w = (w - w.T) ** 2
    return float(1 - np.sum(w * cm) / np.sum(w * expected))


def _corr(x: np.ndarray, y: np.ndarray, spearman: bool = False) -> float:
    """Pearson / Spearman correlation; NaN for n < 3 or constant input."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return NAN
    if spearman:  # = scipy.stats.spearmanr(x, y)[0] (Pearson of average ranks) without its p-value
        x, y = stats.rankdata(x), stats.rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(y_true, y_pred, groups=None) -> dict:
    """Regression metrics on finite pairs (log10 k in this project): mae, rmse, r2 (1 - SS_res / SS_tot),
    bias (mean pred - true), spearman, pearson, n. With `groups` (e.g. condition ids) also
    within_condition_spearman = Spearman of (y - group mean y) vs (yhat - group mean yhat) over groups with
    >= 2 members (NaN if predictions or targets are constant within every group)."""
    y, yh = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(yh)
    y, yh = y[ok], yh[ok]
    e = yh - y
    n = len(y)
    ss_tot = ((y - y.mean()) ** 2).sum() if n else 0.0
    out = {
        "mae": float(np.abs(e).mean()) if n else NAN,
        "rmse": float(np.sqrt((e**2).mean())) if n else NAN,
        "r2": float(1 - (e**2).sum() / ss_tot) if n >= 2 and ss_tot > 0 else NAN,
        "bias": float(e.mean()) if n else NAN,
        "spearman": _corr(y, yh, spearman=True),
        "pearson": _corr(y, yh),
        "n": int(n),
    }
    if groups is not None:
        g = np.asarray(groups, dtype=object)[ok]
        known = ~pd.isna(g)
        codes = pd.factorize(g[known])[0]  # equality grouping, as value_counts
        keep = np.zeros(len(g), bool)
        keep[np.flatnonzero(known)] = np.bincount(codes)[codes] >= 2
        gk = g[keep]

        def centred(v):
            c = v - pd.Series(v).groupby(gk).transform("mean").to_numpy()
            c[np.abs(c) <= 1e-12 * np.abs(v)] = 0.0  # drop float noise of the mean so constant groups tie exactly
            return c

        out["within_condition_spearman"] = _corr(centred(y[keep]), centred(yh[keep]), spearman=True)
    return out


def transition_timing(samples: pd.DataFrame, runs: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Mild->severe transition timing per run [s].

    samples: run_id, t_start_s, time_s (= interval end), p_severe (= P(severity >= 1)), severity_true.
    runs: run_id, duration_s, transition_time_true (NaN = stable run). Only runs with >= 1 sample are scored.
    t_det = midpoint of the first interval (by t_start_s) with p_severe >= threshold; no alarm -> duration_s.
    Transition runs: mae_s / bias_s (t_det - t_true, misses included), detection_rate (any alarm),
    early_alarm_rate (alarm interval ends <= t_true), oracle_floor_s (MAE of the same rule applied to
    severity_true >= 1: the discretization floor of a perfect classifier). Stable runs: false_alarm_rate.
    interval_s = median interval length; mae_intervals = mae_s / interval_s. per_run: DataFrame.
    """
    s = samples.sort_values(["run_id", "t_start_s"], kind="stable").assign(
        mid=lambda d: (d.t_start_s + d.time_s) / 2)
    alarm = s[s.p_severe >= threshold].drop_duplicates("run_id").set_index("run_id")
    oracle = s[s.severity_true >= 1].drop_duplicates("run_id").set_index("run_id")
    r = runs.drop_duplicates("run_id")
    r = r.loc[r.run_id.isin(s.run_id), ["run_id", "duration_s", "transition_time_true"]].reset_index(drop=True)
    r["detected"] = r.run_id.isin(alarm.index)
    r["t_det"] = r.run_id.map(alarm.mid).fillna(r.duration_s)
    r["t_oracle"] = r.run_id.map(oracle.mid).fillna(r.duration_s)
    r["error_s"] = r.t_det - r.transition_time_true
    r["early"] = r.detected & (r.run_id.map(alarm.time_s) <= r.transition_time_true)
    tr = r.transition_time_true.notna()
    t, st = r[tr], r[~tr]
    interval_s = float(np.median(s.time_s - s.t_start_s)) if len(s) else NAN
    mae = float(t.error_s.abs().mean()) if len(t) else NAN
    return {
        "mae_s": mae,
        "bias_s": float(t.error_s.mean()) if len(t) else NAN,
        "mae_intervals": mae / interval_s if interval_s > 0 else NAN,
        "detection_rate": float(t.detected.mean()) if len(t) else NAN,
        "early_alarm_rate": float(t.early.mean()) if len(t) else NAN,
        "false_alarm_rate": float(st.detected.mean()) if len(st) else NAN,
        "oracle_floor_s": float((t.t_oracle - t.transition_time_true).abs().mean()) if len(t) else NAN,
        "interval_s": interval_s,
        "n_transition_runs": int(len(t)),
        "n_stable_runs": int(len(st)),
        "per_run": r,
    }


def volume_consistency(log_v_debris, log_v_grav) -> dict:
    """Debris-implied vs gravimetric run volumes (natural logs of mm^3), finite pairs only.

    mean_log_ratio / sd_log_ratio of (debris - grav); slope, intercept, r2 of the OLS fit debris ~ grav
    (ideal 1, 0, 1; NaN for n < 3 or constant grav); ccc = Lin's concordance correlation (population moments).
    """
    x, y = np.asarray(log_v_grav, dtype=float), np.asarray(log_v_debris, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    d = y - x
    fit = stats.linregress(x, y) if n >= 3 and np.ptp(x) > 0 else None
    den = x.var() + y.var() + (x.mean() - y.mean()) ** 2 if n >= 2 else 0.0
    return {
        "mean_log_ratio": float(d.mean()) if n else NAN,
        "sd_log_ratio": float(d.std(ddof=1)) if n >= 2 else NAN,
        "slope": float(fit.slope) if fit else NAN,
        "intercept": float(fit.intercept) if fit else NAN,
        "r2": float(fit.rvalue**2) if fit else NAN,
        "ccc": float(2 * ((x - x.mean()) * (y - y.mean())).mean() / den) if den > 0 else NAN,
        "n": int(n),
    }
