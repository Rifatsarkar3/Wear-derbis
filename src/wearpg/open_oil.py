"""Real open oil-debris data (no laboratory): loaders, audits, the DESIGN sec. 3 cumulative observation model,
unit-level statistics and NASA spur-gear oil-debris tables transcribed from public-domain reports.

Sources, licences, citations and known biases: docs/open_data_oil.md. Raw downloads live in
data/real/<name>/raw/ (gitignored, never redistributed); scripts/open_oil_fetch.py re-downloads them.
Units of the Zenodo element columns are not stated in the file; ppm (mg/kg oil, the unit of oil spectroscopy) is
assumed everywhere ("ppm/h" = mg per kg oil per working hour).
"""

from __future__ import annotations

import io
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------ Zenodo
ZENODO_ELEMENTS = ("Fe", "Cr", "Al", "Cu", "Pb", "Sn", "Ni", "Ti", "Ag", "Mo", "Si", "Na", "B", "Zn", "P", "Ca",
                   "Ba", "Mg", "Mn", "Cd")
ZENODO_HEADER = ("Tractor age", "Working Hours", *ZENODO_ELEMENTS, "Water content", "Fuel Dilution", "PQ", "Vis40")
_RENAME = {"Tractor age": "tractor_age", "Working Hours": "working_hours", "Water content": "water_content",
           "Fuel Dilution": "fuel_dilution", "PQ": "pq", "Vis40": "vis40"}
WEAR_METALS = ("Fe", "Cr", "Al", "Cu", "Pb", "Sn", "Ni", "Ti", "Ag", "Mn")  # additives/contaminants excluded
COMPOSITION_NUMERATORS = ("Cr", "Al", "Cu", "Pb", "Mn")  # wear metals that are > 0 in every Zenodo sample
PHASES = ("infant_mortality", "useful_life", "wear_out")
SAMPLE_HOURS = (20, 40, 60, 80, 100)
_PHASE_NAMES = {"infant mortality life": "infant_mortality", "useful life": "useful_life", "wear-out life": "wear_out"}
PHASE_SOURCE = "inferred: tractor age 1-4 / 5-8 / >=9 (= block order); the file has no per-row phase column"


def phase_from_age(age: float) -> str:
    """Life phase inferred from tractor age [years]: <= 4 infant_mortality, 5-8 useful_life, >= 9 wear_out."""
    return PHASES[0] if age <= 4 else PHASES[1] if age <= 8 else PHASES[2]


def _sheet_rows(path, sheet: str) -> list[tuple]:
    import openpyxl  # xlsx is read as data only; nothing in the workbook is executed

    wb = openpyxl.load_workbook(Path(path), read_only=True, data_only=True)
    try:
        return [tuple(r) for r in wb[sheet].iter_rows(values_only=True)]
    finally:
        wb.close()


def load_zenodo_wear_debris(path) -> pd.DataFrame:
    """Sheet1 of Zenodo 17287193 'wear derbis.xlsx' as one row per oil sample.

    Sheet1 holds blocks of 5 rows (working hours 20..100) separated by blank rows; block k is unit T{k:02d}
    (one tractor). Returns unit_id, block, tractor_age, working_hours, the 20 element columns (ppm, as given),
    water_content (str, 'negetive' spelling normalised), fuel_dilution, pq, vis40, phase, phase_source.
    The phase is inferred from age (phase_from_age); ValueError unless phases form contiguous groups in block order.
    """
    rows = _sheet_rows(path, "Sheet1")
    n = len(ZENODO_HEADER)
    if not rows or tuple(rows[0][:n]) != ZENODO_HEADER:
        raise ValueError(f"unexpected Sheet1 header: {rows[0][:n] if rows else None}")
    recs, block, blank = [], 0, True
    for r in rows[1:]:
        vals = tuple(r[:n]) + (None,) * max(0, n - len(r))
        if all(v is None for v in vals):
            blank = True
            continue
        block += blank
        blank = False
        recs.append((block, *vals))
    df = pd.DataFrame(recs, columns=["block", *ZENODO_HEADER]).rename(columns=_RENAME)
    num = ["tractor_age", "working_hours", *ZENODO_ELEMENTS, "fuel_dilution", "pq", "vis40"]
    bad = [c for c in num if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in df[c])]
    if bad:
        raise ValueError(f"non-numeric cells in {bad}")
    df[num] = df[num].astype(float)
    if (df[list(ZENODO_ELEMENTS)] < 0).any().any():
        raise ValueError("negative concentration")
    for b, g in df.groupby("block"):
        if tuple(g["working_hours"]) != SAMPLE_HOURS or g["tractor_age"].nunique() != 1:
            raise ValueError(f"block {b}: expected hours {SAMPLE_HOURS} at one tractor age")
    df["water_content"] = df["water_content"].astype(str).str.strip().str.lower().replace({"negetive": "negative"})
    df.insert(0, "unit_id", [f"T{b:02d}" for b in df["block"]])
    df["phase"] = [phase_from_age(a) for a in df["tractor_age"]]
    order = df.groupby("block")["phase"].first().map(PHASES.index).to_numpy()
    if (np.diff(order) < 0).any():
        raise ValueError("age-inferred phases are not contiguous in block order")
    df["phase_source"] = PHASE_SOURCE
    return df


def load_zenodo_phase_summary(path) -> pd.DataFrame:
    """Sheet2 (phase summary tables) as long rows (phase, variable, value); header rows have an empty first cell."""
    out, header = [], None
    for r in _sheet_rows(path, "Sheet2"):
        if not any(v is not None for v in r):
            continue
        if r[0] is None:
            header = r
        elif header is not None and str(r[0]).strip().lower() in _PHASE_NAMES:
            ph = _PHASE_NAMES[str(r[0]).strip().lower()]
            out += [(ph, str(h).strip(), float(v)) for h, v in zip(header[1:], r[1:]) if h is not None and v is not None]
    return pd.DataFrame(out, columns=["phase", "variable", "value"])


def summary_reproduction(samples: pd.DataFrame, summary: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """For each numeric Sheet2 cell: units whose 5-sample mean equals it (within tol), the nearest unit, and the
    mean over all units of the age-inferred phase (what a phase summary would normally report)."""
    rename = {**_RENAME, **{e: e for e in ZENODO_ELEMENTS}}
    unit_mean = samples.groupby("unit_id")[[c for c in rename.values() if c in samples and c != "water_content"
                                            and samples[c].dtype.kind == "f"]].mean()
    phase_of = samples.groupby("unit_id")["phase"].first()
    rows = []
    for ph, var, val in summary.itertuples(index=False):
        col = rename.get(var)
        if col not in unit_mean:
            continue
        d = (unit_mean[col] - val).abs()
        hits = d.index[d <= tol].tolist()
        rows.append({"phase": ph, "variable": var, "sheet2_value": val, "matching_units": ";".join(hits),
                     "n_match": len(hits), "nearest_unit": d.idxmin(), "nearest_abs_diff": float(d.min()),
                     "nearest_unit_phase": phase_of[d.idxmin()],
                     "phase_mean_all_units": float(unit_mean.loc[phase_of == ph, col].mean())})
    return pd.DataFrame(rows)


def pairwise_identity_audit(samples: pd.DataFrame, cols) -> pd.DataFrame:
    """Every pair of units at the same working hours: number of `cols` with exactly identical values.

    Exact multi-column identity of independent spectrometric readings of different engines is implausible;
    a high share flags templated or copied rows. Columns: working_hours, unit_a, unit_b, phase_a, phase_b,
    same_phase, n_identical, n_compared, identical_cols."""
    cols = list(cols)
    rows = []
    for h, g in samples.groupby("working_hours"):
        g = g.set_index("unit_id")
        for a, b in itertools.combinations(g.index, 2):
            eq = [c for c in cols if g.at[a, c] == g.at[b, c]]
            rows.append({"working_hours": h, "unit_a": a, "unit_b": b, "phase_a": g.at[a, "phase"],
                         "phase_b": g.at[b, "phase"], "same_phase": g.at[a, "phase"] == g.at[b, "phase"],
                         "n_identical": len(eq), "n_compared": len(cols), "identical_cols": ";".join(eq)})
    return pd.DataFrame(rows)


def outlier_audit(samples: pd.DataFrame, cols=WEAR_METALS, ratio: float = 5.0, drop: float = 0.3) -> pd.DataFrame:
    """Flag (a) values > ratio x or < 1/ratio x the median of the same phase and working hours, and (b) drops
    > `drop` (fraction) between consecutive samples of one unit (a sump inventory should not fall that fast).
    Flags only; nothing is removed or imputed."""
    rows = []
    for c in cols:
        med = samples.groupby(["phase", "working_hours"])[c].transform("median")
        for i in np.flatnonzero((med > 0) & ((samples[c] > ratio * med) | (samples[c] < med / ratio))):
            r = samples.iloc[i]
            rows.append({"rule": "ratio_to_phase_hour_median", "unit_id": r.unit_id, "working_hours": r.working_hours,
                         "column": c, "value": r[c], "reference": float(med.iloc[i])})
        prev = samples.groupby("unit_id")[c].shift()
        for i in np.flatnonzero((prev > 0) & (samples[c] < (1 - drop) * prev)):
            r = samples.iloc[i]
            rows.append({"rule": "within_unit_drop", "unit_id": r.unit_id, "working_hours": r.working_hours,
                         "column": c, "value": r[c], "reference": float(prev.iloc[i])})
    return pd.DataFrame(rows, columns=["rule", "unit_id", "working_hours", "column", "value", "reference"])


# ------------------------------------------------------------------------------- cumulative observation model
def debris_rates(samples: pd.DataFrame, cols, q: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DESIGN sec. 3 cumulative (sump) mode on oil samples, per kg of oil.

    C_s = sum of `cols` [ppm] at working hours t_s; generated debris G_s = C_s + q * sum_{j<s} C_j adds back what
    earlier samplings removed (q = removed oil fraction per sample); C(0) = 0 (fresh oil, wear metals).
    Returns (per-sample frame with unit_id, phase, working_hours, conc, generated, running_rate = G_s / t_s,
    per-unit frame with rate_cum = G_last / t_last [ppm/h, primary] and rate_ols = OLS slope of G on t [ppm/h,
    no C(0) assumption]). Mass rate [mg/h] = rate [ppm/h] x oil mass [kg] (not given in the dataset)."""
    s = samples.sort_values(["unit_id", "working_hours"])
    conc = s[list(cols)].sum(axis=1)
    removed_before = conc.groupby(s["unit_id"]).transform(lambda v: v.cumsum().shift(fill_value=0.0))
    per = pd.DataFrame({"unit_id": s["unit_id"], "phase": s["phase"], "working_hours": s["working_hours"],
                        "conc": conc, "generated": conc + q * removed_before})
    per["running_rate"] = per["generated"] / per["working_hours"]
    unit = per.groupby("unit_id").agg(phase=("phase", "first"), t_last=("working_hours", "max"),
                                      g_last=("generated", "last"))
    unit["rate_cum"] = unit["g_last"] / unit["t_last"]
    unit["rate_ols"] = per.groupby("unit_id").apply(lambda g: np.polyfit(g["working_hours"], g["generated"], 1)[0],
                                                     include_groups=False)
    return per.reset_index(drop=True), unit


# ------------------------------------------------------------------------------------------ unit statistics
def cliffs_delta(a, b) -> float:
    """P(A > B) - P(A < B) over all pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sign(a[:, None] - b[None, :]).mean())


def hedges_g(a, b) -> float:
    """Bias-corrected standardized mean difference (a - b) with pooled SD."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))) if sp > 0 else float("nan")


def permutation_p(a, b, max_exact: int = 200_000, n_perm: int = 20_000, seed: int = 0) -> tuple[float, str]:
    """Two-sided permutation p of the difference in means: exact enumeration of all splits when there are at most
    max_exact, else Monte Carlo (1 + hits) / (n_perm + 1). Returns (p, 'exact' | 'monte_carlo')."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    x, na = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean()) - 1e-12
    total, n = x.sum(), len(x)
    if math.comb(n, na) <= max_exact:
        idx = np.array(list(itertools.combinations(range(n), na)))
        sa = x[idx].sum(axis=1)
        d = np.abs(sa / na - (total - sa) / (n - na))
        return float((d >= obs).mean()), "exact"
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_perm, n)), axis=1)[:, :na]
    sa = x[perm].sum(axis=1)
    d = np.abs(sa / na - (total - sa) / (n - na))
    return float((1 + (d >= obs).sum()) / (n_perm + 1)), "monte_carlo"


def group_contrast(a, b, log10: bool = True, n_boot: int = 2000, seed: int = 0) -> dict:
    """Effect sizes of group a vs b (units are the observations): medians, Cliff's delta and Hedges g (on log10
    values when log10) with percentile CIs from a bootstrap that resamples units within each group, and the
    permutation p of the difference in (log10) means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ta, tb = (np.log10(a), np.log10(b)) if log10 else (a, b)
    rng = np.random.default_rng(seed)
    reps = np.array([(cliffs_delta(ta[ia], tb[ib]), hedges_g(ta[ia], tb[ib]))
                     for ia, ib in ((rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b)))
                                    for _ in range(n_boot))])
    p, kind = permutation_p(ta, tb, seed=seed)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5], axis=0)
    return {"n_a": len(a), "n_b": len(b), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "ratio_of_medians": float(np.median(a) / np.median(b)) if np.median(b) != 0 else float("nan"),
            "cliffs_delta": cliffs_delta(ta, tb), "cliffs_delta_lo": float(lo[0]), "cliffs_delta_hi": float(hi[0]),
            "hedges_g": hedges_g(ta, tb), "hedges_g_lo": float(lo[1]), "hedges_g_hi": float(hi[1]),
            "perm_p": p, "perm_kind": kind, "scale": "log10" if log10 else "linear"}


def youden_threshold(neg, pos) -> float:
    """Threshold t of the rule 'x >= t' maximizing TPR - FPR. Candidates: midpoints between consecutive sorted
    unique pooled values plus one value below and one above all; ties -> the lower median of the tied
    candidates (for separated classes: the midpoint of the gap)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    u = np.unique(np.concatenate([neg, pos]))
    cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    j = (pos[None, :] >= cand[:, None]).mean(axis=1) - (neg[None, :] >= cand[:, None]).mean(axis=1)
    best = np.flatnonzero(j >= j.max() - 1e-12)
    return float(cand[best[(len(best) - 1) // 2]])


def loo_threshold_alarms(unit_feature: pd.Series, positive: pd.Series, running: pd.DataFrame,
                         time_col: str, value_col: str, extra_negatives=()) -> pd.DataFrame:
    """Leave-one-unit-out transition alarms: for each labelled unit, tau = youden_threshold fitted on the OTHER
    labelled units' unit_feature; the alarm fires at the first `time_col` where the unit's running `value_col`
    >= tau. extra_negatives (never in any training set) use tau fitted on all labelled units.
    Returns unit_id, positive, tau, alarmed, alarm_time (NaN if no alarm), role ('loo' | 'extra_negative')."""
    lab = positive.astype(bool)
    rows = []
    run = running.sort_values(["unit_id", time_col])

    def alarm(u, tau):
        r = run[run["unit_id"] == u]
        hit = r.loc[r[value_col] >= tau, time_col]
        return (len(hit) > 0, float(hit.iloc[0]) if len(hit) else float("nan"))

    for u in lab.index:
        tr = lab.index != u
        tau = youden_threshold(unit_feature[lab.index[tr & ~lab.to_numpy()]], unit_feature[lab.index[tr & lab.to_numpy()]])
        rows.append({"unit_id": u, "positive": bool(lab[u]), "tau": tau, **dict(zip(("alarmed", "alarm_time"),
                                                                                  alarm(u, tau))), "role": "loo"})
    tau_all = youden_threshold(unit_feature[lab.index[~lab.to_numpy()]], unit_feature[lab.index[lab.to_numpy()]])
    for u in extra_negatives:
        rows.append({"unit_id": u, "positive": False, "tau": tau_all,
                     **dict(zip(("alarmed", "alarm_time"), alarm(u, tau_all))), "role": "extra_negative"})
    return pd.DataFrame(rows)


def alarm_summary(alarms: pd.DataFrame) -> dict:
    """Detection rate (positives alarmed), false-alarm rate (negatives alarmed), balanced accuracy, median alarm time
    of detected positives; LOO units only (extra negatives reported separately)."""
    loo = alarms[alarms["role"] == "loo"]
    pos, neg = loo[loo["positive"]], loo[~loo["positive"]]
    det, fa = float(pos["alarmed"].mean()), float(neg["alarmed"].mean())
    ext = alarms[alarms["role"] == "extra_negative"]
    return {"n_pos": len(pos), "n_neg": len(neg), "detection_rate": det, "false_alarm_rate": fa,
            "balanced_accuracy": (det + 1 - fa) / 2, "median_alarm_time_pos": float(pos["alarm_time"].median()),
            "n_extra_negative": len(ext),
            "extra_negative_false_alarm_rate": float(ext["alarmed"].mean()) if len(ext) else float("nan")}


def composition_features(samples: pd.DataFrame, numerators=COMPOSITION_NUMERATORS, denominator: str = "Fe"
                         ) -> pd.DataFrame:
    """Per unit: median over its samples of log(X / Fe) (additive log-ratio; the median is robust to single
    outlier cells). ValueError on non-positive values (zeros would need an imputation we do not make)."""
    cols = [*numerators, denominator]
    if (samples[cols] <= 0).any().any():
        raise ValueError("non-positive concentration in composition columns")
    lr = np.log(samples[list(numerators)].div(samples[denominator], axis=0)).add_prefix("alr_")
    return lr.groupby(samples["unit_id"]).median()


def loo_proba(X: np.ndarray, y: np.ndarray, n_classes: int, C: float = 1.0) -> np.ndarray:
    """Leave-one-out probabilities (n, n_classes) of StandardScaler + LogisticRegression(class_weight balanced);
    rows are units, so LOO is grouped by unit. Classes absent from a training fold get probability 0."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = np.asarray(X, float).reshape(len(y), -1), np.asarray(y)
    out = np.zeros((len(y), n_classes))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
        m.fit(X[tr], y[tr])
        out[i, m.classes_] = m.predict_proba(X[i:i + 1])[0]
    return out


def balanced_accuracy(y_true, proba) -> float:
    """Mean recall over classes present in y_true (prediction = argmax)."""
    y, pred = np.asarray(y_true), np.asarray(proba).argmax(axis=1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def permutation_balanced_accuracy(X, y, n_classes: int, n_perm: int = 999, seed: int = 0) -> tuple[float, float]:
    """(observed LOO balanced accuracy, permutation p = (1 + #{perm >= obs}) / (n_perm + 1)), labels permuted across
    units."""
    obs = balanced_accuracy(y, loo_proba(X, y, n_classes))
    rng = np.random.default_rng(seed)
    hits = sum(balanced_accuracy(yp, loo_proba(X, yp, n_classes)) >= obs - 1e-12
               for yp in (rng.permutation(y) for _ in range(n_perm)))
    return obs, (1 + hits) / (n_perm + 1)


def auc(neg, pos) -> float:
    """Mann-Whitney AUC = P(pos > neg) + 0.5 P(tie)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    return float((np.sign(pos[:, None] - neg[None, :]).mean() + 1) / 2)


# ------------------------------------------------------------------------------------------ NASA tables
NASA_GEAR_TEETH = 28  # TM-2001-210936 p. 3 / TM-2003-211307 sec. 2: standard spur gears with 28 teeth

# Dempsey, NASA/TM-2003-211307, Table 3.9 "Summary of Experiments", report p. 49 (PDF p. 55); born-digital text,
# checked cell by cell against the rendered page. Damage text verbatim (driver / driven lines joined by " / ").
NASA_TM2003_TABLE_3_9 = """exp|rdg_first|desc_first|mass_first_mg|rdg_end|desc_end|mass_end_mg
1|14369|Dr: de 1t / Dn: de 1t|15.475|15136|Dr: de 3t, ip allt / Dn: de3t|36.108
2|2199|Dr: / Dn: de 1t|8.934|2444|Dr: de 2t, ip allt / Dn de 3t, ip 3t|26.268
3|2669|Dr: de 2t / Dn: de 2t|8.690|3029|Dr: de 3t, ip allt / Dn: de 3t, ip3t|14.148
4|2065|Dr: de 3t / Dn:|12.132|4863|Dr: de 7t, ip allt / Dn de 3t, ip allt|26.227
5|2566|Dr: ip 2t / Dn:|7.413|4425|Dr: de 11t, ip allt / Dn de 10t, ip allt|10.811
6|12061|Dr: / Dn: de 1t|14.365|12368|Dr: de 1t, ip allt / Dn de 2t, ip allt|22.851
7||||13716|Dr: ip 1t / Dn ip 1t|3.381
8|5181|Dr: ip 2t / Dn: ip 3t|6.012|5314|Dr: de 6t, ip8t / Dn de6t, ip7t|19.101
9||||29866|No damage|2.359
10||||20452|No damage|5.453
11||||204|No damage|.418
12||||15654|No damage|2.276
13||||25259|No damage|3.159
14||||5322|No damage|0
15||||21016|No damage|.125
16||||380|No damage|.099
17||||21066|No damage|.064
18||||888|Dr: de 6t, ip allt / Dn de 4t, ip allt|22.541
19||||199|Dr: de 3t, ip allt / Dn de 1t, ip allt|11.230
20||||1593|Dr: de 1t, ip allt / Dn  ip allt|5.346
21|317|Dr: de 1t / Dn: de 1t|4.04|514|Dr: de 2t, ip allt / Dn: de 2t, ip allt|17.912
22||||838|Dr: ip 5t / Dn: de 3t, ip allt|7.224
23||||10688|Dr: de 2t,  ip allt / Dn: de 1t, ip allt|6.399
24|7170|Dr: de 1t / Dn:|6.186|7224|Dr: de 1t, ip allt / Dn: ip allt|9.681
"""

# Same report, Tables 3.10-3.24 (report pp. 50-61): every inspection reading of the experiments with damage.
# Teeth lists are tooth NUMBERS ("All" = all 28 teeth); empty = none listed. Transcribed from the rendered pages.
NASA_TM2003_READINGS = """exp|table|page|reading|mass_mg|description|de_driver|de_driven|ip_driver|ip_driven
1|3.10|50|60|1.003|Run-in Wear||||
1|3.10|50|120|1.418|Wear||||
1|3.10|50|1581|5.113|Wear||||
1|3.10|50|10622|12.533|Wear||||
1|3.10|50|14369|15.475|Destructive Pitting|6|6||
1|3.10|50|14430|22.468|Destructive Pitting|6|6||
1|3.10|50|14512|24.586|Destructive Pitting|6,7|6,7||
1|3.10|50|14688|28.451|Destructive Pitting|6,7|6,7||
1|3.10|50|14846|30.686|Destructive Pitting|6,7|6,7||
1|3.10|50|15136|36.108|Initial Pitting; Destructive Pitting|6,7,8|6,7,10|All|
2|3.11|50|1573|3.285|Run-in Wear||||
2|3.11|50|2199|8.934|Destructive Pitting||11||
2|3.11|50|2296|16.267|Destructive Pitting||10,11||
2|3.11|50|2444|26.268|Initial Pitting; Destructive Pitting|10,11|10,11,14|All|10,11,14
3|3.12|50|58|0|Run-in Wear||||
3|3.12|50|2669|8.69|Destructive Pitting|1,28|1,28||
3|3.12|50|2857|11.889|Destructive Pitting|1,6,28|1,6,28||
3|3.12|50|3029|14.148|Initial Pitting; Destructive Pitting|1,6,28|1,6,28|All|1,6,28
4|3.13|51|64|0|Run-in Wear||||
4|3.13|51|150|2.233|Wear||||
4|3.13|51|378|8.297|Wear||||
4|3.13|51|518|9.462|Wear||||
4|3.13|51|2065|12.132|Destructive Pitting|3,6,7|||
4|3.13|51|2366|13.977|Destructive Pitting|3,6,7|||
4|3.13|51|3671|17.361|Destructive Pitting|3,6,7|5||
4|3.13|51|4655|23.12|Destructive Pitting|3,6,7|3,5,7||
4|3.13|51|4863|26.227|Initial Pitting; Destructive Pitting|3,6,7,11,14,16,28|3,5,7|All|All
5|3.14|52|62|0|Run-in Wear||||
5|3.14|52|1405|4.214|Wear||||
5|3.14|52|2566|7.413|Destructive Pitting|17,25|||
5|3.14|52|4425|10.811|Initial Pitting; Destructive Pitting|1,3,17,18,19,20,21,24,25,26,28|1,17,18,19,20,21,22,24,25,26|All|All
6|3.15|53|60|0|Run-in Wear||||
6|3.15|53|2810|3.192|Wear||||
6|3.15|53|2885|6.396|Wear||||
6|3.15|53|2957|8.704|Wear||||
6|3.15|53|9328|11.692|Wear||||
6|3.15|53|12061|14.365|Destructive Pitting||22||
6|3.15|53|12368|22.851|Initial Pitting; Destructive Pitting|25|22,25|All|All
7|3.16|54|13716|3.381|Initial Pitting|||12|12
8|3.17|54|5181|6.012|Initial Pitting; Destructive Pitting|||15,16|15,16,17
8|3.17|54|5314|19.101|Initial Pitting; Destructive Pitting|9,15,16,17,18,24|9,15,16,17,18,24|19,24,27|14
18|3.18|55|60|0|Run-in Wear||||
18|3.18|55|888|22.541|Initial Pitting; Destructive Pitting|5,6,7,11,25,27|5,6,7,11|All|All
19|3.19|56|62|0|Run-in Wear||||
19|3.19|56|199|11.230|Initial Pitting; Destructive Pitting|3,4,13|3|All|All
20|3.20|57|66|0|Run-in Wear||||
20|3.20|57|1593|5.346|Initial Pitting; Destructive Pitting|26||All|All
21|3.21|58|62|0|Run-in Wear||||
21|3.21|58|317|4.04|Destructive Pitting|22|22||
21|3.21|58|373|9.808|Destructive Pitting|22,24|22,24||
21|3.21|58|455|10.936|Destructive Pitting|22,24|22,24||
21|3.21|58|514|17.912|Initial Pitting; Destructive Pitting|22,24|22,24|All|All
22|3.22|59|129|0|Run-in Wear||||
22|3.22|59|838|7.224|Initial Pitting; Destructive Pitting||18,19,26|12,14,18,19,26|All
23|3.23|60|62|0|Run-in Wear||||
23|3.23|60|10688|6.399|Initial Pitting; Destructive Pitting|10,11|10|All|All
24|3.24|61|0|0|Run-in Wear||||
24|3.24|61|4077|2.344|Wear||||
24|3.24|61|7170|6.186|Destructive Pitting|26|||
24|3.24|61|7224|9.681|Initial Pitting; Destructive Pitting|26||All|All
"""

# Dempsey, NASA/TM-2001-210936 (scanned): Table 2 p. 5, Table 3 p. 6, Table 4 p. 7, transcribed from the page images
# (zoomed). first_destructive = highlighted cell (Table 2) / 'Destructive' row (Table 3). Cross-check only.
NASA_TM2001_TABLES = """exp|table|page|reading|mass_mg|first_destructive|note
1|2|5|60|1.003|0|
1|2|5|120|1.418|0|
1|2|5|1581|5.113|0|
1|2|5|10622|12.533|0|
1|2|5|14369|15.475|1|
1|2|5|14430|22.468|0|
1|2|5|14512|24.586|0|
1|2|5|14688|28.451|0|
1|2|5|14846|30.686|0|
1|2|5|15136|36.108|0|
2|2|5|1573|3.285|0|
2|2|5|2199|8.934|1|
2|2|5|2296|16.267|0|
2|2|5|2444|26.268|0|
3|2|5|58|0|0|
3|2|5|2669|8.69|1|
3|2|5|2857|11.889|0|
3|2|5|3029|14.148|0|
4|2|5|64|0|0|
4|2|5|150|2.233|0|
4|2|5|378|8.297|0|
4|2|5|518|9.462|0|
4|2|5|2065|12.132|1|
4|2|5|2366|13.977|0|
4|2|5|3671|17.361|0|
4|2|5|4655|23.12|0|
4|2|5|4863|26.227|0|
5|2|5|62|0|0|
5|2|5|1405|4.214|0|
5|2|5|2566|7.413|0|
5|2|5|4425|10.811|1|
6|2|5|60|0|0|
6|2|5|2810|3.192|0|
6|2|5|2885|6.396|0|
6|2|5|2957|8.704|0|
6|2|5|9328|11.692|0|
6|2|5|12061|14.365|1|
6|2|5|12368|22.851|0|
7|3|6|13716|3.381|0|initial pitting
8|3|6|5181|6.012|0|initial pitting
8|3|6|5314|19.101|1|destructive pitting
9|4|7|29866|2.359|0|no damage, completion
10|4|7|20452|5.453|0|no damage, completion
11|4|7|204|0.418|0|no damage, completion
12|4|7|15654|2.276|0|no damage, completion
13|4|7|25259|3.159|0|no damage, completion
14|4|7|5322|0|0|no damage, completion
15|4|7|21016|0.125|0|no damage, completion
16|4|7|21446|0.163|0|no damage, completion
"""


def _table(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)


def n_teeth(cell: str) -> int:
    """Number of teeth in a tooth list cell: '' -> 0, 'All' / 'All teeth' -> 28, '6, 7,8' -> 3."""
    cell = cell.strip()
    if not cell:
        return 0
    if cell.lower().startswith("all"):
        return NASA_GEAR_TEETH
    return len({int(t) for t in cell.split(",")})


def parse_damage(desc: str) -> dict:
    """Counts from a Table 3.9 damage text such as 'Dr: de 3t, ip allt / Dn: de3t' -> de/ip teeth per gear.
    'ip' in Table 3.9 counts every pitted tooth, destructive ones included (verified on experiment 8)."""
    out = {"de_driver": 0, "de_driven": 0, "ip_driver": 0, "ip_driven": 0}
    if desc.strip().lower() in ("", "no damage"):
        return out
    parts = desc.split("/")
    if len(parts) != 2:
        raise ValueError(f"expected 'Dr ... / Dn ...': {desc!r}")
    for gear, part in zip(("driver", "driven"), parts):
        for kind in ("de", "ip"):
            m = re.search(kind + r"\s*(\d+|all)t", part)
            if m:
                out[f"{kind}_{gear}"] = NASA_GEAR_TEETH if m.group(1) == "all" else int(m.group(1))
    return out


def nasa_experiments() -> pd.DataFrame:
    """One row per spur-gear experiment (TM-2003-211307 Table 3.9): rdg_end [min], mass_end_mg, n_de_end
    (destructively pitted teeth, driver + driven), n_pitted_end (any pitting), stage_end (0 none, 1 initial pitting
    only, 2 destructive pitting), first-observation columns (NaN when pitting was first seen at completion)."""
    t = _table(NASA_TM2003_TABLE_3_9)
    rows = []
    for r in t.itertuples(index=False):
        d = parse_damage(r.desc_end)
        n_de = d["de_driver"] + d["de_driven"]
        n_pit = sum(max(d[f"ip_{g}"], d[f"de_{g}"]) for g in ("driver", "driven"))
        rows.append({"exp": int(r.exp), "rdg_end": float(r.rdg_end), "mass_end_mg": float(r.mass_end_mg),
                     "n_de_end": n_de, "n_pitted_end": n_pit, "stage_end": 2 if n_de else 1 if n_pit else 0,
                     "desc_end": r.desc_end, "rdg_first": float(r.rdg_first) if r.rdg_first else float("nan"),
                     "mass_first_mg": float(r.mass_first_mg) if r.mass_first_mg else float("nan"),
                     "desc_first": r.desc_first, "source": "TM-2003-211307 Table 3.9 p. 49"})
    return pd.DataFrame(rows)


def nasa_readings(exp5_rdg2566: str = "table_3_9", include_run_in: bool = False) -> pd.DataFrame:
    """Inspection readings with a known damage state: Tables 3.10-3.24 (experiments with damage) plus the
    completion readings of experiments 9-17 (no damage, Table 3.9). stage: 0 no pitting, 1 initial only,
    2 destructive (n_de > 0). Experiment 5 at reading 2566 is printed 'Destructive Pitting 17,25' in Table 3.14
    but 'Dr: ip 2t' in Table 3.9 and not highlighted in TM-2001 Table 2: exp5_rdg2566 = 'table_3_9' (primary,
    initial pitting) or 'table_3_14' (as printed). Run-in rows (before the loaded test) are dropped unless
    include_run_in."""
    if exp5_rdg2566 not in ("table_3_9", "table_3_14"):
        raise ValueError(exp5_rdg2566)
    t = _table(NASA_TM2003_READINGS)
    rows = []
    for r in t.itertuples(index=False):
        if r.description == "Run-in Wear" and not include_run_in:
            continue
        n_de = n_teeth(r.de_driver) + n_teeth(r.de_driven)
        n_ip = n_teeth(r.ip_driver) + n_teeth(r.ip_driven)
        if (int(r.exp), int(r.reading)) == (5, 2566) and exp5_rdg2566 == "table_3_9":
            n_de, n_ip = 0, 2  # Table 3.9: 'Dr: ip 2t' (the two teeth 17, 25 of Table 3.14 as initial pits)
        rows.append({"exp": int(r.exp), "reading": float(r.reading), "mass_mg": float(r.mass_mg), "n_de": n_de,
                     "stage": 2 if n_de else 1 if n_ip else 0, "source": f"TM-2003-211307 Table {r.table} p. {r.page}"})
    for r in nasa_experiments().itertuples(index=False):
        if 9 <= r.exp <= 17:
            rows.append({"exp": r.exp, "reading": r.rdg_end, "mass_mg": r.mass_end_mg, "n_de": 0, "stage": 0,
                         "source": "TM-2003-211307 Table 3.9 p. 49 (completion)"})
    return pd.DataFrame(rows).sort_values(["exp", "reading"]).reset_index(drop=True)


def nasa_cross_check() -> pd.DataFrame:
    """TM-2001-210936 Tables 2-4 vs TM-2003-211307 (same experiments): mass at the same (exp, reading) and the first
    destructive reading. Returns one row per TM-2001 entry with the TM-2003 mass (NaN if absent) and agree flag."""
    old = _table(NASA_TM2001_TABLES)
    new = _table(NASA_TM2003_READINGS)
    new_mass = {(int(r.exp), int(r.reading)): float(r.mass_mg) for r in new.itertuples(index=False)}
    new_mass.update({(r.exp, int(r.rdg_end)): r.mass_end_mg for r in nasa_experiments().itertuples(index=False)})
    rows = []
    for r in old.itertuples(index=False):
        key = (int(r.exp), int(r.reading))
        m = new_mass.get(key, float("nan"))
        rows.append({"exp": key[0], "reading": key[1], "mass_tm2001": float(r.mass_mg), "mass_tm2003": m,
                     "agree": bool(abs(m - float(r.mass_mg)) < 5e-4) if np.isfinite(m) else False,
                     "first_destructive_tm2001": int(r.first_destructive), "tm2001_table": r.table})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ Dataverse
DATAVERSE_COLUMNS = ("id", "icm_isocode", "icm_rh", "icm_flow", "icm_temp", "icm_iso4", "icm_iso6", "icm_iso14",
                     "icm_pc4", "icm_pc6", "icm_pc14", "fps_vcst", "fps_v", "fps_density", "fps_dielectric", "fps_temp",
                     "oh_temp", "oh_parama", "oh_paramb", "oh_paramc", "oh_od")


def load_dataverse_oil(path) -> pd.DataFrame:
    """Harvard Dataverse SO9SXL 'Lubricating Oil Parameters_1.xlsx': row 1 = column names, row 2 = descriptions,
    then one row per online measurement. icm_isocode is kept as text; cells Excel turned into dates (e.g.
    '15/10/5' -> 2005-10-15) become '' with isocode_excel_date = True (the disaggregated icm_iso* columns are
    intact). Other columns are coerced to float (empty -> NaN)."""
    import datetime as dt

    import openpyxl

    wb = openpyxl.load_workbook(Path(path), read_only=True, data_only=True)
    try:
        it = wb.worksheets[0].iter_rows(values_only=True)
        header = tuple(next(it))[:len(DATAVERSE_COLUMNS)]
        if header != DATAVERSE_COLUMNS:
            raise ValueError(f"unexpected header {header}")
        next(it)  # description row
        rows = [tuple(r[:len(DATAVERSE_COLUMNS)]) for r in it]
    finally:
        wb.close()
    df = pd.DataFrame(rows, columns=list(DATAVERSE_COLUMNS))
    iso = df["icm_isocode"]
    df["isocode_excel_date"] = [isinstance(v, (dt.datetime, dt.date)) for v in iso]
    df["icm_isocode"] = [("" if isinstance(v, (dt.datetime, dt.date)) or v is None else str(v)) for v in iso]
    num = [c for c in DATAVERSE_COLUMNS if c != "icm_isocode"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df
