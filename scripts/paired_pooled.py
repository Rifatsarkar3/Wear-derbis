"""Post-hoc analyses after the frozen hold-out run (plan part 6 of outputs/agent_logs/PAIRED.md, logged before this program ran): what the planned
verdict depends on, the 23 FZG gear pairs together, and cumulative against interval correlation. The frozen run stays the primary result.

    .venv/Scripts/python.exe -I scripts/paired_pooled.py     # -> outputs/open_data/paired/pooled/{results.json, pairs.csv, rows.csv}
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]  # -I drops the cwd

import paired_holdout as ph  # noqa: E402

from wearpg import paired as pp  # noqa: E402
from wearpg.utils import save_json  # noqa: E402

HOLD = ROOT / "outputs" / "open_data" / "paired" / "holdout"
OUT = ROOT / "outputs" / "open_data" / "paired" / "pooled"
CONDS = list(ph.CONDS)
EXTREMES = ("feup_martins2008:CM_pl", "feup_martins2008:E1_pI", "feup_martins2008:E2_pI")  # two scuffed pairs and the 7-mg pair
SEED = 0


# ------------------------------------------------------------------------------------------------------------- data
def readings() -> tuple[dict, dict]:
    r1 = json.loads((HOLD / "reading1_part5.json").read_text(encoding="utf-8"))["cumulative"]
    with open(HOLD / "reading2_part5.csv", encoding="utf-8") as f:
        r2 = {(r["source"], r["series"], r["point"]): float(r["value"]) for r in csv.DictReader(f) if r["quantity"] == "cpuc"}
    return r1, r2


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Thesis rows and new rows with the debris value as run (D0) and with the K6 sample in the first micropitting rows (D1)."""
    t = ph.thesis_rows().assign(study="feup_martins2008")
    t["unit"] = "feup_martins2008:" + t.unit
    n = ph.load_new()
    r1, r2 = readings()
    t["D0"], t["D1"], n["D0"], n["D1"] = t.debris, t.debris, n.debris, n.debris
    for src in ("jet495", "tribint2009"):
        for oil, c in r1[src].items():
            k3, k6 = (float(np.sqrt(v * r2[src, oil, p])) for v, p in zip(c[:2], ("K3", "K6")))  # geometric mean of the two readings, as in part 5
            i = n.index[(n.unit == f"{src}:{oil}") & n.row.str.endswith("K3+K6")]
            assert len(i) == 1 and abs(n.debris[i[0]] / k6 - 1) < 0.002, (src, oil)             # the value as run is the curve at K6
            n.loc[i, "D1"] = k6 - k3
    return t, n


def with_p3(n: pd.DataFrame, w: float) -> pd.DataFrame:
    """The pair P3 of pij2011 (printed mass loss 0): D by the rule of the plan (last CPUC + 40/1500 x earlier ones), W = w mg."""
    r1, r2 = readings()
    v = [float(np.sqrt(a * r2["pij2011", "P3", p])) for a, p in zip(r1["pij2011"]["P3"], ("t12", "t24", "t36", "t48"))]
    d = v[3] + 40 / 1500 * sum(v[:3])
    p1 = n[n.unit == "pij2011:P1"].iloc[0].to_dict()
    return pd.concat([n, pd.DataFrame([{**p1, "unit": "pij2011:P3", "row": "pij2011:P3:end", "debris": d, "D0": d, "D1": d, "wear": w}])], ignore_index=True)


def totals(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per pair: D and W summed over the stages, cycles summed, cycle-weighted mean square of the pressure."""
    r = []
    for u, g in d.groupby("unit", sort=False):
        cyc = g.cond_cycles_M.sum()
        r.append({"unit": u, "row": u + ":total", "debris": g[col].sum(), "wear": g.wear.sum(), "cond_cycles_M": cyc, "mode": g["mode"].iloc[0], "cond_ester": g.cond_ester.iloc[0], "cond_adi": g.cond_adi.iloc[0],
                  "cond_hertz_MPa": float(np.sqrt((g.cond_cycles_M * g.cond_hertz_MPa**2).sum() / cyc))})
    return pd.DataFrame(r)


# ------------------------------------------------------------------------------------------------------------- A, B, C
def frame(d: pd.DataFrame, col: str) -> pd.DataFrame:
    return d.assign(debris=d[col])


def tests(new: pd.DataFrame, thesis: pd.DataFrame, c: dict) -> dict:
    """The frozen program's evaluation, shortened."""
    r, _ = ph.evaluate(new, thesis, c)
    keep = ("n_units", "n_rows", "fold", "within_2", "H1a_verdict", "H1b_verdict", "mean_log10_error_debris_ci", "H1b_debris_vs_constant", "H1c_debris_vs_refitted_constant",
            "H2_debris_vs_archard", "H3_debris_vs_conditions", "H4_slope", "H5_spearman", "H6_pooled_refitted_loo", "decision")
    r = {k: r[k] for k in keep if k in r}
    r["rows_within_2"] = int((np.abs(np.log10(new.wear) - (c["debris_b"] + np.log10(new.debris))) <= np.log10(2)).sum())
    return r


def signed(new: pd.DataFrame, c: dict) -> dict:
    """Signed errors (predicted over measured, log10 unit means) of the frozen predictors: mean factor, bootstrap interval, scatter (GSD)."""
    e = ph.predict(new, c).sub(np.log10(new.wear), axis=0).groupby(new.unit, sort=False).mean()
    out = {}
    for k in e:
        v = e[k].to_numpy()
        out[k] = {"factor": float(10 ** v.mean()), "ci": [float(10 ** q) for q in ph.boot_ci(v)], "p_signflip": pp.signflip_p(v), "gsd": float(10 ** v.std(ddof=1))}
    src = e.index.str.split(":").str[0]
    out["by_source_debris"] = {s: float(10 ** e.debris[src == s].mean()) for s in sorted(set(src))}
    return out


def coverage(new: pd.DataFrame, thesis: pd.DataFrame, c: dict) -> dict:
    """Share of new pairs (unit means) and rows inside the 95 % range of one prediction computed on the 14 thesis pairs (t13, GSD of D/W)."""
    um = np.log10(thesis.set_index("unit")[["debris", "wear"]]).groupby(level=0, sort=False).mean()
    sd, n = (um.debris - um.wear).std(ddof=1), len(um)
    f = 10 ** (stats.t.ppf(0.975, n - 1) * sd * np.sqrt(1 + 1 / n))
    e = (c["debris_b"] + np.log10(new.debris)) - np.log10(new.wear)
    ue = e.groupby(new.unit, sort=False).mean()
    return {"range_factor": float(f), "pairs_inside": int((ue.abs() <= np.log10(f)).sum()), "n_pairs": len(ue), "worst_pair": float(10 ** ue.abs().max()), "rows_inside": int((e.abs() <= np.log10(f)).sum()), "n_rows": len(e),
            "rows_below_calibration_debris_min": int((new.debris < thesis.debris.min()).sum()), "new_debris_range": [float(new.debris.min()), float(new.debris.max())], "new_wear_range": [float(new.wear.min()), float(new.wear.max())],
            "calibration_debris_range": [float(thesis.debris.min()), float(thesis.debris.max())], "calibration_wear_range": [float(thesis.wear.min()), float(thesis.wear.max())]}


def refit(new: pd.DataFrame, model: str) -> float:
    pr = pp.loo_predictions(np.log10(new.debris), np.log10(new.wear), np.empty((len(new), 0)), new.unit.to_numpy())[[model]]
    return float(10 ** pp.unit_errors(pr, np.log10(new.wear), new.unit.to_numpy())[model].mean())


def constants_table(new: pd.DataFrame, thesis: pd.DataFrame, c: dict) -> dict:
    """Mean error (factor) of one constant for all new pairs, for constants an analyst could have used, and the frozen debris factor beside it."""
    y = np.log10(new.wear)
    deb = ((c["debris_b"] + np.log10(new.debris)) - y).abs().groupby(new.unit, sort=False).mean()

    def err(k):
        return (np.log10(k) - y).abs().groupby(new.unit, sort=False).mean()

    grid = {str(k): float(10 ** err(k).mean()) for k in (5, 8, 10, 12, 14, 16, 18, 20, 24, 28, 47, 71)}
    mp = thesis[thesis["mode"] == "interval"]
    first = mp[mp.row.str.contains(r"K3\+")]
    alt = {"frozen (all thesis rows)": 10 ** np.log10(thesis.wear).mean(), "thesis micropitting rows": 10 ** np.log10(mp.wear).mean(), "thesis micropitting first stages": 10 ** np.log10(first.wear).mean(),
           "thesis rows without the two scuffed pairs": 10 ** np.log10(thesis[~thesis.unit.isin(EXTREMES[:2])].wear).mean()}
    out = {"fold_debris": float(10 ** deb.mean()), "grid_mg": grid, "as_good_as_debris_mg": [k for k in grid if grid[k] <= 10 ** deb.mean()], "alternatives": {}}
    for name, k in alt.items():
        d = deb - err(k)
        out["alternatives"][name] = {"constant_mg": float(k), "fold": float(10 ** err(k).mean()), "diff_debris_minus_constant": float(d.mean()), "ci": ph.boot_ci(d), "p_signflip": pp.signflip_p(d)}
    st = new.unit.str.split(":").str[0]
    src = pd.Series(index=new.index, dtype=float)
    for s in st.unique():
        src[st == s] = y[st != s].mean()
    out["refitted"] = {"constant_leave_one_pair_out": refit(new, "M0"), "constant_leave_one_source_out": float(10 ** (src - y).abs().groupby(new.unit, sort=False).mean().mean()), "debris_leave_one_pair_out": refit(new, "M2")}
    return out


def sensitivity(new: pd.DataFrame, thesis: pd.DataFrame, c: dict) -> dict:
    """The decisive numbers of the plan in every subset (H1c was missing in sensitivity.json)."""
    st = new.unit.str.split(":").str[0]
    s = {"all": new, "without K3 rows": new[~new.row.str.contains("K3")], "without the K3 row of ilt2011": new[~((new.unit == "ilt2011:E2") & new.row.str.endswith(":K3"))], "interval rows only": new[new["mode"] == "interval"],
         "rows as units": new.assign(unit=new.unit + ":" + new.row)}
    first, last = new.sort_values("order").groupby("unit", sort=False).head(1), new.sort_values("order").groupby("unit", sort=False).tail(1)
    s.update({"first stage only": first, "last stage only": last})
    s.update({f"without source {k}": new[st != k] for k in sorted(st.unique())})
    s.update({f"without pair {u}": new[new.unit != u] for u in new.unit.unique()})
    out = {}
    for k, v in s.items():
        r = tests(v.reset_index(drop=True), thesis, c)
        b, h = r["H1b_debris_vs_constant"], r["H1c_debris_vs_refitted_constant"]
        h2, h3 = r.get("H2_debris_vs_archard", {}), r.get("H3_debris_vs_conditions", {})
        out[k] = {"rows_key": "|".join(sorted(v.unit + "/" + v.row)), "n_units": r["n_units"], "fold_debris": r["fold"]["debris"], "fold_archard": r["fold"].get("archard"), "fold_conditions": r["fold"].get("conditions"), "H2_diff": h2.get("diff"), "H3_diff": h3.get("diff"), "H2_holm": h2.get("p_holm"), "H1a_ci": r["mean_log10_error_debris_ci"], "H1a": r["H1a_verdict"], "H1b": r["H1b_verdict"], "H1b_diff": b["diff"], "H1b_p": b["p_signflip"], "H1c_diff": h["diff"], "H1c_ci": h["ci"], "H1c_p": h["p_signflip"],
                  "H2_p": r.get("H2_debris_vs_archard", {}).get("p_signflip"), "H3_p": r.get("H3_debris_vs_conditions", {}).get("p_signflip"), "decision": r["decision"]}
    return out


def unit_mean_scoring(new: pd.DataFrame, c: dict) -> dict:
    """Plan sec. 6 (missing in sensitivity.json): scored on the unit means of log10 D and log10 W."""
    m = np.log10(new.set_index("unit")[["debris", "wear"]]).groupby(level=0, sort=False).mean()
    deb, con = (c["debris_b"] + m.debris - m.wear).abs(), (c["constant"] - m.wear).abs()
    own = pd.Series({u: abs(m.wear.drop(u).mean() - m.wear[u]) for u in m.index})
    return {"fold_debris": float(10 ** deb.mean()), "ci": [float(10 ** q) for q in ph.boot_ci(deb)], "fold_constant": float(10 ** con.mean()), "fold_refitted_constant": float(10 ** own.mean()), "within_2": int((deb <= np.log10(2)).sum()),
            "H1c_diff": float((deb - own).mean()), "H1c_ci": ph.boot_ci(deb - own)}


# ------------------------------------------------------------------------------------------------------------- D
def gsd_ci(sd: float, n: int) -> list[float]:
    return [float(v) for v in 10 ** (sd * np.sqrt((n - 1) / stats.chi2.ppf([0.975, 0.025], n - 1)))]


def mean_diff(d: pd.Series) -> dict:
    return {"diff": float(d.mean()), "ci": ph.boot_ci(d), "p_signflip": pp.signflip_p(d), "better_in": int((d < 0).sum()), "n": len(d)}


def pooled(d: pd.DataFrame, col: str) -> dict:
    """Slope, rank correlation, scatter, factor and leave-one-pair-out errors of one set of rows (unit = gear pair)."""
    x, y, u = np.log10(d[col].to_numpy(float)), np.log10(d.wear.to_numpy(float)), d.unit.to_numpy()
    um = pd.DataFrame({"x": x, "y": y}).groupby(u, sort=False).mean()
    s, sp, n = pp.slope(um.x, um.y), pp.spearman(um.x, um.y), len(um)
    r = um.y - um.x
    sd = float(r.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    Z = np.column_stack([np.log10(d.cond_cycles_M), np.log10(d.cond_hertz_MPa), d.cond_ester, d.cond_adi])
    Z = Z[:, Z.std(0) > 0]
    pr = pp.loo_predictions(x, y, Z, u)
    pr["archard"] = pp.loo_predictions(ph.archard(d), y, np.empty((len(d), 0)), u).M2
    err = pp.unit_errors(pr, y, u)
    rng = float(10 ** (stats.t.ppf(0.975, n - 1) * sd * np.sqrt(1 + 1 / n)))
    return {"n_pairs": n, "n_rows": len(d), "n_sources": int(d.unit.str.split(":").str[0].nunique()), "wear_mg_units": [float(10 ** um.y.min()), float(10 ** um.y.max())], "wear_mg_rows": [float(10 ** y.min()), float(10 ** y.max())],
            "decades_wear_units": float(np.ptp(um.y)), "slope": s["slope"], "perm_ci": [s["perm_lo"], s["perm_hi"]], "t_ci": [s["t_lo"], s["t_hi"]], "p_slope_1": s["p_slope_eq_ref"], "p_slope_0": s["p_slope_eq_0"],
            "inverse_slope": s["inverse_slope"], "rma_slope": s["rma_slope"], "rho": sp["rho"], "rho_p": sp["p"], "rho_ci": [sp["boot_lo"], sp["boot_hi"]], "pearson_r": s["r"],
            "gsd_D_over_W": float(10 ** sd), "gsd_ci": gsd_ci(sd, n), "gsd_W_units": float(10 ** um.y.std(ddof=1)), "range95_factor": rng, "pairs_inside_range": int((np.abs(r - r.mean()) <= np.log10(rng)).sum()),
            "factor_W_over_D": float(10 ** r.mean()), "factor_ci": [float(10 ** (r.mean() - h)), float(10 ** (r.mean() + h))],
            "fold_loo": {k: float(10 ** err[v].mean()) for k, v in (("debris", "M2"), ("constant", "M0"), ("conditions", "M1"), ("archard", "archard"), ("fused", "M3"), ("free_slope", "M2f"))},
            "within_2_pairs_loo": int((err.M2 <= np.log10(2)).sum()), "worst_pair_loo": float(10 ** err.M2.max()),
            "debris_minus": {k: mean_diff(err.M2 - err[v]) for k, v in (("constant", "M0"), ("conditions", "M1"), ("archard", "archard"))}, "fused_minus_debris": mean_diff(err.M3 - err.M2)}


def source_transfer(d: pd.DataFrame, col: str) -> dict:
    """Leave-one-SOURCE-out calibration of the factor (unit means of the other sources) and the between-source scatter of log W/D."""
    x, y, src = np.log10(d[col]), np.log10(d.wear), d.unit.str.split(":").str[0]
    r = (y - x).groupby(d.unit, sort=False).mean()
    rs = r.index.str.split(":").str[0]
    out = {"loso": {}}
    for s in src.unique():
        b = r[rs != s].mean()
        e = ((x + b) - y)[src == s].abs().groupby(d.unit[src == s], sort=False).mean()
        out["loso"][s] = {"n_pairs": len(e), "fold": float(10 ** e.mean()), "signed": float(10 ** (b - r[rs == s].mean()))}
    g = [r[rs == s].to_numpy() for s in sorted(set(rs))]                      # one-way random effects, pooled variance inside the sources
    k, n, N = len(g), np.array([len(q) for q in g]), sum(len(q) for q in g)
    msb = sum(len(q) * (q.mean() - r.mean()) ** 2 for q in g) / (k - 1)
    msw = sum(((q - q.mean()) ** 2).sum() for q in g) / (N - k)
    sb2 = max(0.0, (msb - msw) / ((N - (n**2).sum() / N) / (k - 1)))
    out["between_sources"] = {"k": k, "gm_by_source": {s: float(10 ** r[rs == s].mean()) for s in sorted(set(rs))}, "gsd_within": float(10 ** np.sqrt(msw)), "gsd_between": float(10 ** np.sqrt(sb2)), "F": float(msb / msw), "df": [k - 1, N - k],
                              "p": float(stats.f.sf(msb / msw, k - 1, N - k))}
    return out


# ------------------------------------------------------------------------------------------------------------- E
def cumulative_vs_interval(d: pd.DataFrame, col: str, n_perm: int = 10_000) -> dict:
    """Squared correlation of cumulative debris and cumulative wear at the stage ends (as the sources plot them) against the interval correlation; nulls by shuffling."""
    g = [(q[col].to_numpy(float), q.wear.to_numpy(float), q.unit.iloc[0]) for _, q in d.groupby("unit", sort=False)]
    cum = [(np.cumsum(a), np.cumsum(b)) for a, b, _ in g]

    def r2(a, b):
        return float(np.corrcoef(a, b)[0, 1] ** 2)

    def pair_mean(parts):
        return float(np.mean([r2(a, b) for a, b in parts]))

    def pool(parts):
        return r2(np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]))

    rng = np.random.default_rng(SEED)
    all_d, sizes = np.concatenate([a for a, _, _ in g]), [len(a) for a, _, _ in g]
    nul = {"within_pair": ([], []), "all_rows": ([], [])}
    for _ in range(n_perm):
        p = [(np.cumsum(rng.permutation(a)), np.cumsum(b)) for a, b, _ in g]
        q, o, parts = rng.permutation(all_d), 0, []
        for (_, b, _), m in zip(g, sizes):
            parts.append((np.cumsum(q[o:o + m]), np.cumsum(b)))
            o += m
        for k, z in (("within_pair", p), ("all_rows", parts)):
            nul[k][0].append(pair_mean(z)), nul[k][1].append(pool(z))
    x, y = np.log10(all_d), np.log10(np.concatenate([b for _, b, _ in g]))
    unit = np.concatenate([[u] * len(a) for a, _, u in g])
    xw = x - pd.Series(x).groupby(unit).transform("mean").to_numpy()
    yw = y - pd.Series(y).groupby(unit).transform("mean").to_numpy()
    obs_w = float(np.corrcoef(xw, yw)[0, 1])
    nw = [float(np.corrcoef(np.concatenate([rng.permutation(xw[unit == u]) for u in pd.unique(unit)]), yw)[0, 1]) for _ in range(n_perm)]
    sp = stats.spearmanr(x, y)
    per = {u: {"r2_cumulative": r2(*c), "n_points": len(c[0])} for (_, _, u), c in zip(g, cum)}
    obs, opool = pair_mean(cum), pool(cum)
    out = {"n_rows": len(x), "n_pairs": len(g), "mean_r2_cumulative_per_pair": obs, "r2_per_pair": per, "pooled_r2_cumulative": opool,
           "mean_r2_by_source": {s: float(np.mean([v["r2_cumulative"] for u, v in per.items() if u.split(":")[0] == s])) for s in sorted({u.split(":")[0] for u in per})},
           "pooled_r2_by_source": {s: pool([c for (_, _, u), c in zip(g, cum) if u.split(":")[0] == s]) for s in sorted({u.split(":")[0] for u in per})},
           "interval_spearman": float(sp.statistic), "interval_spearman_p": float(sp.pvalue), "interval_pearson_log_r2": float(np.corrcoef(x, y)[0, 1] ** 2), "within_pair_pearson_log": obs_w, "within_pair_p_perm": float((np.abs(nw) >= abs(obs_w)).mean())}
    for k, (a, b) in nul.items():
        out["null_" + k] = {"mean_r2": float(np.mean(a)), "q05_q95": [float(v) for v in np.percentile(a, [5, 95])], "share_at_least_observed": float(np.mean(np.array(a) >= obs)), "pooled_r2_median": float(np.median(b)), "share_pooled_at_least_observed": float(np.mean(np.array(b) >= opool))}
        out["null_" + k]["draws"] = [round(float(v), 5) for v in a[:1000]]
    return out


# ------------------------------------------------------------------------------------------------------------- F
def stage_ratios(d: pd.DataFrame, col: str) -> dict:
    out = {}
    for u, g in d[d["mode"] == "interval"].groupby("unit", sort=False):
        g = g.sort_values("order")
        if len(g) >= 2:
            out[u] = float((g[col].iloc[0] / g.wear.iloc[0]) / (g[col].iloc[-1] / g.wear.iloc[-1]))
    v = np.array(list(out.values()))
    return {"first_over_last": out, "range": [float(v.min()), float(v.max())], "median": float(np.median(v)), "above_1": int((v > 1).sum()), "n": len(v)}


def group_effects(rows: pd.DataFrame, col: str) -> dict:
    """Ratio W/D of groups of pairs (unit means of log10 W/D) with a permutation p; exploratory."""
    r = (np.log10(rows.wear) - np.log10(rows[col])).groupby(rows.unit, sort=False).mean()
    one = rows.groupby("unit", sort=False).first()
    g = {"micropitting vs power loss": one["mode"] == "interval", "ester vs mineral": one.cond_ester == 1, "ADI vs steel": one.cond_adi == 1, "nitrided vs other": one.material_pair.str.contains("nitrided"),
         "new vs thesis": ~one.index.str.startswith("feup")}
    rng, out, v = np.random.default_rng(SEED), {}, r.to_numpy()
    for k, m in g.items():
        m = np.asarray(m, bool)
        obs = abs(v[m].mean() - v[~m].mean())
        cnt = 0
        for _ in range(99_999):
            p = rng.permutation(v)
            cnt += abs(p[m].mean() - p[~m].mean()) >= obs - 1e-12
        a, b = v[m], v[~m]
        s2 = (a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(v) - 2)
        h = float(stats.t.ppf(0.975, len(v) - 2) * np.sqrt(s2 * (1 / len(a) + 1 / len(b))))
        d = float(a.mean() - b.mean())
        out[k] = {"n_1": int(m.sum()), "n_0": int((~m).sum()), "ratio_W_over_D": float(10 ** d), "ci": [float(10 ** (d - h)), float(10 ** (d + h))], "p_perm": (1 + cnt) / 100_000, "gsd_1": float(10 ** a.std(ddof=1)), "gsd_0": float(10 ** b.std(ddof=1))}
    return out


def two_groups(d: pd.DataFrame, col: str) -> dict:
    """One constant for each of the two groups of tests (thesis / later papers), left one pair out, beside one debris factor; slope on the unit means with and without a group indicator (t intervals)."""
    x, y, u = np.log10(d[col].to_numpy(float)), np.log10(d.wear.to_numpy(float)), d.unit.to_numpy()
    grp = (~d.unit.str.startswith("feup")).to_numpy(float)
    err = pp.unit_errors(pp.loo_predictions(x, y, grp.reshape(-1, 1), u), y, u)     # M1 = [1, group] constants, M2 = debris factor
    um = pd.DataFrame({"x": x, "y": y, "g": grp}).groupby(u, sort=False).mean()
    n = len(um)
    A = np.column_stack([np.ones(n), um.x, um.g])
    beta, *_ = np.linalg.lstsq(A, um.y, rcond=None)
    s2 = float(((um.y - A @ beta) ** 2).sum() / (n - 3))
    se = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1]))
    tq = float(stats.t.ppf(0.975, n - 3))
    return {"n_pairs": n, "fold_constant_per_group": float(10 ** err.M1.mean()), "fold_debris": float(10 ** err.M2.mean()), "debris_minus_constant_per_group": mean_diff(err.M2 - err.M1),
            "slope_with_group_indicator": float(beta[1]), "slope_t_ci": [float(beta[1] - tq * se), float(beta[1] + tq * se)]}


def dilution(d: pd.DataFrame, col: str) -> dict:
    """GSD of wear and of debris per unit wear inside a set of pairs and the slope that an exactly proportional relation with that scatter would give (regression dilution)."""
    um = np.log10(d.set_index("unit")[[col, "wear"]]).groupby(level=0, sort=False).mean()
    vw, ve = float(um.wear.var(ddof=1)), float((um[col] - um.wear).var(ddof=1))
    fwd, rev = stats.linregress(um[col], um.wear), stats.linregress(um.wear, um[col])
    h = float(stats.t.ppf(0.975, len(um) - 2)) * rev.stderr
    return {"n": len(um), "gsd_wear": float(10 ** np.sqrt(vw)), "gsd_D_over_W": float(10 ** np.sqrt(ve)), "expected_slope_if_proportional": vw / (vw + ve), "slope": float(fwd.slope), "gsd_debris": float(10 ** um[col].std(ddof=1)),
            "gsd_debris_needed": float(10 ** np.sqrt(vw + ve)), "reverse_slope": float(rev.slope), "reverse_t_ci": [float(rev.slope - h), float(rev.slope + h)]}


def pair_mean_folds(d: pd.DataFrame, col: str) -> dict:
    """Leave-one-pair-out errors scored on the geometric means of a pair (not on its rows): one debris factor, one constant, one constant per group."""
    um = np.log10(d.set_index("unit")[[col, "wear"]]).groupby(level=0, sort=False).mean()
    grp = (~um.index.str.startswith("feup")).astype(float).reshape(-1, 1)
    x, y, u = um[col].to_numpy(), um.wear.to_numpy(), um.index.to_numpy()
    err = pp.unit_errors(pp.loo_predictions(x, y, grp, u), y, u)
    return {k: float(10 ** err[m].mean()) for k, m in (("debris", "M2"), ("constant", "M0"), ("constant_per_group", "M1"))}


def two_stage(new: pd.DataFrame, thesis: pd.DataFrame, b: int = 100_000) -> dict:
    """Two-stage bootstrap: the thesis pairs that set the factor and the constant are resampled (all rows of a drawn pair), then the new pairs. Also the t interval of the mean error over the new pairs with the frozen factor."""
    rng = np.random.default_rng(SEED)
    r, y = np.log10(thesis.wear) - np.log10(thesis.debris), np.log10(thesis.wear)
    g = pd.DataFrame({"r": r, "y": y, "m": 1.0}).groupby(thesis.unit, sort=False).sum().to_numpy()
    s = g[rng.integers(0, len(g), (b, len(g)))].sum(1)                      # sums over the drawn pairs
    f, k = s[:, 0] / s[:, 2], s[:, 1] / s[:, 2]
    X, Y = np.log10(new.debris.to_numpy(float)), np.log10(new.wear.to_numpy(float))
    names = list(pd.unique(new.unit))
    A = np.array([(new.unit == n).to_numpy(float) / (new.unit == n).sum() for n in names])   # pairs x rows
    ed, ec, es = (np.abs(X + f[:, None] - Y) @ A.T, np.abs(k[:, None] - Y) @ A.T, (X + f[:, None] - Y) @ A.T)
    j = rng.integers(0, len(names), (b, len(names)))
    d, c, sg = (np.take_along_axis(e, j, 1).mean(1) for e in (ed, ec, es))
    e0 = (np.abs(X + r.mean() - Y) @ A.T)
    h = float(stats.t.ppf(0.975, len(names) - 1)) * e0.std(ddof=1) / np.sqrt(len(names))

    def q(v):
        return [float(a) for a in np.percentile(v, [2.5, 97.5])]
    return {"fold": float(10 ** e0.mean()), "fold_ci_two_stage": [10 ** a for a in q(d)], "fold_ci_t": [float(10 ** (e0.mean() - h)), float(10 ** (e0.mean() + h))], "share_fold_above_2": float((d > np.log10(2)).mean()),
            "share_fold_above_2.5": float((d > np.log10(2.5)).mean()), "H1b_diff_ci_two_stage": q(d - c), "share_H1b_diff_above_0": float((d - c > 0).mean()), "signed_ci_two_stage": [10 ** a for a in q(sg)], "replicates": b}


def distinct_subsets(s: dict) -> dict:
    """The list of subsets holds the full set and duplicates: counted by the rows and units they hold."""
    seen = {}
    for k, v in s.items():
        seen.setdefault(v["rows_key"], (k, v["H1c_diff"]))
    full = s["all"]["rows_key"]
    sub = [d for key, (_, d) in seen.items() if key != full]
    return {"names": len(s), "different": len(seen), "different_without_the_full_set": len(sub), "H1c_negative": int(sum(d < 0 for d in sub))}


def iron_check(factor_mg_per_unit: float) -> dict:
    """Iron (ppm, X-ray fluorescence) of the used oils of two power-loss tests of the thesis (Table 6.6) against the final CPUC of Fig. 6.13 (PDF p. 89, vector end points; axis 0 at y = 612.2, 800 at 456.9).
    No mass loss is printed for these tests. The weighed wear per unit of CPUC is the calibrated factor over 1.5 l of oil (density 0.9 kg/l assumed)."""
    y0, y8 = 612.2, 456.9
    cpuc = {k: (y0 - y) / (y0 - y8) * 800 for k, y in (("M1", 480.2), ("E1", 567.2))}
    fe = {"M1": 18.0, "E1": 5.0}
    wear = factor_mg_per_unit / (1.5 * 0.9)                                    # mg per kg of oil and unit
    return {"cpuc_final": cpuc, "fe_ppm": fe, "fe_ppm_per_unit": {k: fe[k] / cpuc[k] for k in fe}, "weighed_wear_ppm_per_unit": wear, "fraction_found_as_iron": {k: fe[k] / cpuc[k] / wear for k in fe}}


def stage_number_r2(d: pd.DataFrame) -> float:
    """Mean squared correlation of the cumulative wear with the stage number, micropitting pairs of the later papers."""
    r = []
    for _, g in d[d["mode"] == "interval"].groupby("unit", sort=False):
        w = np.cumsum(g.sort_values("order").wear.to_numpy(float))
        r.append(np.corrcoef(np.arange(1, len(w) + 1), w)[0, 1] ** 2)
    return float(np.mean(r))


def per_litre() -> dict:
    """Wear per litre of oil and unit of CPUC, gears (23 pairs, 1.5 l) and the four-ball set of the thesis ("about 8 ml" plus a rinse of unknown volume), density 7.8 mg/mm3."""
    rev = json.loads((ROOT / "outputs" / "open_data" / "paired" / "extra" / "review.json").read_text(encoding="utf-8"))["factors"]["per_set"]["feup_martins2008/fourball"]
    s = pd.read_csv(ROOT / "outputs" / "open_data" / "paired" / "summary.csv")
    raw = float(s[(s.study == "feup_martins2008") & (s.set == "fourball") & (s.analysis == "no_elastic_correction")].DoverW_gm.iloc[0])
    return {"fourball_mg_per_litre_unit_at_8ml_no_elastic_correction": 7.8 / raw / 0.008, "fourball_mg_per_unit": rev["factor_wear_per_debris"] * 7.8, "fourball_mg_per_litre_unit_at_8ml": rev["factor_wear_per_debris"] * 7.8 / 0.008, "fourball_ci_at_8ml": [v * 7.8 / 0.008 for v in rev["ci95_mean"]],
            "n_fourball": rev["n"], "volume_ratio": 1.5 / 0.008}


def large_particles(thesis_csv: Path) -> dict:
    """Percentage of large particles (DL - DS)/(DL + DS) of the thesis rows against log10 D/W (row and pair level)."""
    t = pd.read_csv(thesis_csv)
    t = t[(t.set == "fzg") & t.class_DL.notna() & t.class_DS.notna()].copy()
    t["plp"] = (t.class_DL - t.class_DS) / (t.class_DL + t.class_DS)
    t["r"] = np.log10(t.debris) - np.log10(t.wear)
    a, um = stats.spearmanr(t.plp, t.r), t.groupby("unit", sort=False)[["plp", "r"]].mean()
    b = stats.spearmanr(um.plp, um.r)
    return {"rows": len(t), "row_rho": float(a.statistic), "row_p": float(a.pvalue), "units": len(um), "unit_rho": float(b.statistic), "unit_p": float(b.pvalue)}


# ------------------------------------------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c = json.loads((HOLD / "frozen.json").read_text())["constants"]
    thesis, new = load()
    th = thesis.assign(unit=thesis.unit.str.split(":").str[1])      # the frozen program prefixes the thesis units itself
    res = {"status": "POST HOC (plan part 6 of outputs/agent_logs/PAIRED.md, logged before the run); the frozen run of part 5 is the primary result and is not touched"}
    # A decision table
    cases = {"D0 as run": frame(new, "D0"), "D1 K6 sample": frame(new, "D1")}
    for w in (0.5, 1.0, 2.0):
        cases[f"D2 as run, P3 at {w:g} mg"] = frame(with_p3(new, w), "D0")
        cases[f"D3 K6 sample, P3 at {w:g} mg"] = frame(with_p3(new, w), "D1")
    res["decision_table"] = {k: tests(v, th, c) for k, v in cases.items()}
    tt = totals(th.assign(D0=th.debris), "D0")
    for tag, col in (("D4a run totals, as run", "D0"), ("D4b run totals, K6 sample", "D1")):
        nt = totals(new, col)
        res["decision_table"][tag + ", frozen row constants"] = tests(nt, tt, c)
        res["decision_table"][tag + ", constants of the thesis run totals"] = tests(nt, tt, ph.constants(tt))
    f, d0 = json.loads((HOLD / "results.json").read_text()), res["decision_table"]["D0 as run"]
    assert abs(d0["fold"]["debris"] - f["fold"]["debris"]) < 1e-9 and d0["decision"] == f["decision"] and abs(d0["H1c_debris_vs_refitted_constant"]["diff"] - f["H1c_debris_vs_refitted_constant"]["diff"]) < 1e-9
    # B opponents, H1c in every subset, unit-mean scoring; C level
    both = pd.concat([thesis, new], ignore_index=True)
    res["opponents"], res["sensitivity_H1"], res["unit_mean_scoring"], res["signed_errors"], res["coverage"] = {}, {}, {}, {}, {}
    for tag, col in (("as run", "D0"), ("K6 sample", "D1")):
        n = frame(new, col)
        res["opponents"][tag], res["sensitivity_H1"][tag] = constants_table(n, thesis, c), sensitivity(n, th, c)
        res["unit_mean_scoring"][tag], res["signed_errors"][tag], res["coverage"][tag] = unit_mean_scoring(n, c), signed(n, c), coverage(n, thesis, c)
    # D pooled
    P = res["pooled"] = {}
    for tag, col, d in (("23 pairs, as run", "D0", both), ("23 pairs, K6 sample", "D1", both), ("14 thesis pairs", "D0", thesis), ("9 new pairs, as run", "D0", new), ("9 new pairs, K6 sample", "D1", new)):
        P[tag] = pooled(d, col)
        if tag.startswith("23"):
            P[tag + ", without the three extreme pairs"] = pooled(d[~d.unit.isin(EXTREMES)].reset_index(drop=True), col)
    p3 = pd.concat([thesis, with_p3(new, 1.0)], ignore_index=True)
    P["24 pairs with P3 at 1 mg, as run"], P["24 pairs with P3 at 1 mg, K6 sample"] = pooled(p3, "D0"), pooled(p3, "D1")
    for tag, col in (("as run", "D0"), ("K6 sample", "D1")):
        tot = pd.concat([totals(thesis, "D0"), totals(new, col)], ignore_index=True)
        P[f"23 pairs, run totals ({tag})"] = pooled(tot.assign(D0=tot.debris), "D0")
    th_tot = totals(thesis, "D0").rename(columns={"debris": "D0"})                       # M7: both groups in both units
    P["14 thesis pairs, run totals"] = pooled(th_tot, "D0")
    P["11 ordinary thesis pairs"] = pooled(thesis[~thesis.unit.isin(EXTREMES)].reset_index(drop=True), "D0")
    P["11 ordinary thesis pairs, run totals"] = pooled(th_tot[~th_tot.unit.isin(EXTREMES)].reset_index(drop=True), "D0")
    for tag, col in (("as run", "D0"), ("K6 sample", "D1")):
        P[f"9 new pairs, run totals ({tag})"] = pooled(totals(new, col).rename(columns={"debris": "D0"}), "D0")
    P["23 pairs, as run, without the K3 row of ilt2011"] = pooled(both[~((both.unit == "ilt2011:E2") & both.row.str.endswith(":K3"))].reset_index(drop=True), "D0")
    pitch = both.copy()
    m = (pitch.study == "feup_martins2008") & (pitch["mode"] == "interval")
    for old, pit in ((1180, 1094), (1505, 1395)):                    # pitch-point values of the thesis (Paper B, Table 5) instead of the maximum
        pitch.loc[m & (pitch.cond_hertz_MPa == old), "cond_hertz_MPa"] = pit
    P["23 pairs, as run, pitch-point pressures of the thesis micropitting rows"] = pooled(pitch, "D0")
    g = np.log10(both.set_index("unit")[["D0", "wear"]]).groupby(level=0, sort=False).mean()
    th_, nw_ = g.index.str.startswith("feup"), ~g.index.str.startswith("feup")
    res["level_shift_thesis_over_new"] = {"debris": float(10 ** (g.D0[th_].mean() - g.D0[nw_].mean())), "wear": float(10 ** (g.wear[th_].mean() - g.wear[nw_].mean()))}
    res["source_transfer"] = {"as run": source_transfer(both, "D0"), "K6 sample": source_transfer(both, "D1")}
    # G two groups, H run totals without extremes, K oil volume
    for tag, col in (("as run", "D0"), ("K6 sample", "D1")):
        tot = pd.concat([totals(thesis, "D0"), totals(new, col)], ignore_index=True).assign(**{"D0": lambda q: q.debris})
        P[f"20 pairs, run totals ({tag}), without the three extreme pairs"] = pooled(tot[~tot.unit.isin(EXTREMES)].reset_index(drop=True), "D0")
        res.setdefault("two_groups", {})[f"23 pairs, stage means ({tag})"] = two_groups(both, col)
        res["two_groups"][f"20 pairs, stage means ({tag})"] = two_groups(both[~both.unit.isin(EXTREMES)].reset_index(drop=True), col)
        res["two_groups"][f"23 pairs, run totals ({tag})"] = two_groups(tot, "D0")
        res["two_groups"][f"20 pairs, run totals ({tag})"] = two_groups(tot[~tot.unit.isin(EXTREMES)].reset_index(drop=True), "D0")
    tot0 = pd.concat([totals(thesis, "D0"), totals(new, "D0")], ignore_index=True)
    res["level_shift_run_totals"] = {"debris": float(10 ** (np.log10(tot0.debris[tot0.unit.str.startswith("feup")]).mean() - np.log10(tot0.debris[~tot0.unit.str.startswith("feup")]).mean())),
                                     "wear": float(10 ** (np.log10(tot0.wear[tot0.unit.str.startswith("feup")]).mean() - np.log10(tot0.wear[~tot0.unit.str.startswith("feup")]).mean())),
                                     "new_total_loss_mg": [float(tot0.wear[~tot0.unit.str.startswith("feup")].min()), float(tot0.wear[~tot0.unit.str.startswith("feup")].max())], "all_total_loss_mg": [float(tot0.wear.min()), float(tot0.wear.max())]}
    res["stage_number_r2"] = {"7 new pairs": stage_number_r2(new)}
    ord_ = both[~both.unit.isin(EXTREMES)].reset_index(drop=True)
    res["dilution"] = {k: dilution(v, "D0") for k, v in (("9 later pairs", new), ("11 ordinary thesis pairs", ord_[ord_.unit.str.startswith("feup")]), ("20 ordinary pairs", ord_))}
    res["dilution"]["9 later pairs, K6 sample"] = dilution(new, "D1")
    for tag, col in (("", "D0"), (", K6 sample", "D1")):
        res["dilution"]["9 later pairs, run totals" + tag] = dilution(totals(new, col).rename(columns={"debris": "D0"}), "D0")
    # M (addendum 2): errors on pair means, share of the gain of one constant per group, uncertainty of the factor, different subsets
    res["pair_mean_folds"] = {f"{n} pairs ({tag})": pair_mean_folds(v, col) for n, v in ((23, both), (20, ord_)) for tag, col in (("as run", "D0"), ("K6 sample", "D1"))}
    keys = {"23 pairs, stage means": "23 pairs, {}", "20 pairs, stage means": "23 pairs, {}, without the three extreme pairs", "23 pairs, run totals": "23 pairs, run totals ({})", "20 pairs, run totals": "20 pairs, run totals ({}), without the three extreme pairs"}
    res["gain_share_of_group_constant"] = {}
    for tag in ("as run", "K6 sample"):
        for g_, p_ in keys.items():
            a_, b_, e_ = (float(np.log10(v)) for v in (P[p_.format(tag)]["fold_loo"]["constant"], res["two_groups"][f"{g_} ({tag})"]["fold_constant_per_group"], P[p_.format(tag)]["fold_loo"]["debris"]))
            res["gain_share_of_group_constant"][f"{g_} ({tag})"] = (a_ - b_) / (a_ - e_)
    res["two_stage_bootstrap"] = {tag: two_stage(frame(new, col), thesis) for tag, col in (("as run", "D0"), ("K6 sample", "D1"))}
    res["sensitivity_H1_different_subsets"] = {tag: distinct_subsets(v) for tag, v in res["sensitivity_H1"].items()}
    for v in res["sensitivity_H1"].values():
        for q in v.values():
            q.pop("rows_key")
    res["iron_check"] = iron_check(P["23 pairs, run totals (as run)"]["factor_W_over_D"])
    res["per_litre"] = per_litre()
    # E cumulative against interval
    mp_new, mp_th = new[new["mode"] == "interval"].reset_index(drop=True), thesis[thesis["mode"] == "interval"].reset_index(drop=True)
    mp_all = pd.concat([mp_th, mp_new], ignore_index=True)
    res["cumulative_vs_interval"] = {f"{k}, {t}": cumulative_vs_interval(v, col) for k, v in (("7 new pairs", mp_new), ("11 pairs", mp_all)) for t, col in (("as run", "D0"), ("K6 sample", "D1"))}
    # F scatter
    res["stage_ratios"] = {"new, as run": stage_ratios(new, "D0"), "new, K6 sample": stage_ratios(new, "D1"), "thesis": stage_ratios(thesis, "D0"), "all 11, as run": stage_ratios(both, "D0"), "all 11, K6 sample": stage_ratios(both, "D1")}
    res["group_effects"] = {"as run": group_effects(both, "D0"), "K6 sample": group_effects(both, "D1")}
    res["large_particles_thesis"] = large_particles(ph.THESIS)
    save_json(res, OUT / "results.json")
    u = np.log10(both.set_index("unit")[["D0", "D1", "wear"]]).groupby(level=0, sort=False).mean().reset_index()
    u.insert(1, "study", u.unit.str.split(":").str[0])
    u.to_csv(OUT / "pairs.csv", index=False)
    both[["study", "unit", "row", "order", "mode", "material_pair", "lubricant", *CONDS, "D0", "D1", "wear"]].to_csv(OUT / "rows.csv", index=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
