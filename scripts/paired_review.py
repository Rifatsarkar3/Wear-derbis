"""Calculations asked for by the re-review of 2026-11-25 (post hoc, after every result was known; nothing here replaces
scripts/paired_analysis.py or scripts/paired_extra.py). Each block answers one finding of outputs/agent_logs/review_2026-11-25/R1_numbers.md.

  .venv/Scripts/python.exe -I scripts/paired_review.py     # -> outputs/open_data/paired/extra/review.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]  # -I drops the cwd

import paired_extra as px  # noqa: E402

from wearpg import paired as pp  # noqa: E402
from wearpg.utils import save_json  # noqa: E402

PAIRED = ROOT / "outputs" / "open_data" / "paired"
DATA = ROOT / "data" / "real" / "paired"
SETS = px.SETS


def gsd_ci(sd, n):
    return [float(v) for v in 10 ** (sd * np.sqrt((n - 1) / stats.chi2.ppf([0.975, 0.025], n - 1)))]


def identity_table(u):
    """F02/F03: spread of wear, scatter of D/W and leave-one-out errors on unit means, with the mean absolute deviations the loss uses."""
    out = {}
    for (st, se), g in u.groupby(["study", "set"], sort=False):
        n = len(g)
        if n < 5:
            continue
        y, r = g.log10_W.to_numpy(), (g.log10_D - g.log10_W).to_numpy()
        mad = lambda v: np.abs(v - v.mean()).mean()
        sw, sr = y.std(ddof=1), r.std(ddof=1)
        slope = stats.linregress(g.log10_D, g.log10_W).slope
        out[f"{st}/{se}"] = {"n": n, "sd_log_wear": float(sw), "sd_log_ratio": float(sr), "sd_ratio": float(sw / sr), "mad_ratio": float(mad(y) / mad(r)), "slope": float(slope),
                             "debris_better_than_constant_on_unit_means": bool(mad(r) < mad(y)), "slope_above_0.5": bool(slope > 0.5),
                             "gsd_wear": float(10 ** sw), "gsd_factor": float(10 ** sr), "gsd_factor_ci": gsd_ci(sr, n), "max_over_min_factor": float(10 ** np.ptp(r))}
    med = float(np.median([v["sd_log_ratio"] for v in out.values()]))
    for v in out.values():
        v["ratio_if_common_scatter_median"] = float(v["sd_log_wear"] / med)
    return {"per_set": out, "median_sd_log_ratio": med, "gsd_median": float(10 ** med),
            "gsd_of_fzg_scatter": out["feup_martins2008/fzg"]["gsd_factor"]}


def trend(u):
    """F06: dependence of D/W on duration (dry four-ball) and on distance (plate); GSD about a straight line in that variable."""
    out = {}
    for study, se, pat in (("phm_raadnui2015", "dry_fourball", "min"), ("jsme_higashi2003", "plate_intervals", None)):
        g = u[(u.study == study) & (u.set == se)].copy()
        if pat:
            x = np.log10(g["unit"].str.replace("min", "").astype(float).to_numpy())
        else:
            x = g["unit"].str.extract(r"(\d+)-")[0].astype(float).to_numpy()
        r = (g.log10_D - g.log10_W).to_numpy()
        lin = stats.linregress(x, r)
        resid = r - lin.intercept - lin.slope * x
        s = resid.std(ddof=2) if len(r) > 2 else np.nan
        rho, p = stats.spearmanr(x, r)
        perm = pp.spearman(x, r)
        out[f"{study}/{se}"] = {"n": len(r), "ratio_min_median_max": [float(v) for v in 10 ** np.array([r.min(), np.median(r), r.max()])], "spearman": float(rho), "p_perm": perm["p"],
                               "gsd": float(10 ** r.std(ddof=1)), "gsd_about_trend": float(10 ** s), "order_variable": "log10 duration" if pat else "distance (start of interval)"}
    return out


def zero_shift(u):
    """F07: bearing rig (relative debris axis): slope, GSD and wear/scatter ratio if s divisions are added to every debris reading."""
    d = pd.read_csv(DATA / "phm_bolander2009" / "paired.csv")
    d = d[(d.debris > 0) & (d.wear > 0)]
    out = {}
    for s in (-0.30, -0.15, -0.10, 0.0, 0.10, 0.15, 0.30):
        x = d.debris.to_numpy() + s
        ok = x > 0
        lx, ly = np.log10(x[ok]), np.log10(d.wear.to_numpy()[ok])
        r = lx - ly
        out[f"{s:+.2f}"] = {"n": int(ok.sum()), "slope": float(stats.linregress(lx, ly).slope), "gsd": float(10 ** r.std(ddof=1)), "sd_ratio_wear_over_factor": float(ly.std(ddof=1) / r.std(ddof=1))}
    g = [v["gsd"] for v in out.values()]
    return {"unit": "axis divisions added to the debris reading; 0 = the axis crossing assumed in the paper", "shifts": out, "gsd_range": [min(g), max(g)], "slope_range": [min(v["slope"] for v in out.values()), max(v["slope"] for v in out.values())]}


def repeatability(u):
    """F08: three pairs of duplicate four-ball tests (new balls, new oil, whole oil collected): debris, wear and their ratio."""
    f = u[u.set == "fourball"].assign(oil=lambda d: d.unit.str[:2])
    out = {}
    for name, v in (("debris", f.log10_D), ("wear", f.log10_W), ("ratio", f.log10_D - f.log10_W)):
        d = f.assign(v=v).groupby("oil").v.agg(lambda q: q.iloc[0] - q.iloc[1])
        s = float(np.sqrt((d ** 2).sum() / (2 * len(d))))
        ci = [float(10 ** (s * np.sqrt(len(d) / q))) for q in stats.chi2.ppf([0.975, 0.025], len(d))]
        out[name] = {"pair_ratios": {k: float(10 ** abs(x)) for k, x in d.items()}, "replicate_gsd": float(10 ** s), "df": len(d), "gsd_ci": ci}
    return out


def nasa(u_path):
    """F13: the NASA yield (mass per destructively pitted tooth) in the units of section 5.2: GSD and the same identity on 14 tests."""
    d = pd.read_csv(u_path)
    g = d[d.n_de_end > 0]
    x, y = np.log10(g.mass_end_mg.to_numpy()), np.log10(g.n_de_end.to_numpy())
    r = x - y
    sw, sr = y.std(ddof=1), r.std(ddof=1)
    return {"n": len(g), "gsd_mass_per_tooth": float(10 ** sr), "gsd_ci": gsd_ci(sr, len(g)), "sd_log_teeth": float(sw), "sd_log_mass_per_tooth": float(sr), "sd_ratio": float(sw / sr),
            "max_over_min": float(10 ** np.ptp(r)), "slope_log_teeth_on_log_mass": float(stats.linregress(x, y).slope)}


def verdicts(prim):
    """F01/F15: verdict counts of V1 (slope) and V2 (debris against constant) over the five inferential sets, with and without the dry set,
    pre-specified (B = 10,000) and with the converged interval of the FZG set."""
    conv = {("feup_martins2008", "fzg"): "inconclusive"}
    rows = {}
    for tag, drop in (("five_sets", []), ("without_dry", [("phm_raadnui2015", "dry_fourball")]), ("lubricated_only", [("phm_raadnui2015", "dry_fourball")])):
        for kind in ("pre_specified", "converged"):
            cnt = {"V1": {}, "V2": {}}
            for k in SETS:
                if k in drop:
                    continue
                r = prim.loc[k]
                for v in cnt:
                    val = r[v].split()[0]
                    if kind == "converged" and v == "V2" and k in conv:
                        val = conv[k]
                    cnt[v][val] = cnt[v].get(val, 0) + 1
            rows[f"{tag}:{kind}"] = cnt
    return rows


def pooled(prim):
    """F26: pooled slope over the five inferential sets and over the four lubricated ones; DerSimonian-Laird with Hartung-Knapp interval (t)."""
    out = {}
    for tag, keys in (("five_sets", SETS), ("four_lubricated", [k for k in SETS if k[1] != "dry_fourball"])):
        b, se = [], []
        for k in keys:
            r = prim.loc[k]
            t = stats.t.ppf(0.975, r.n_units - 2)
            b.append(float(r.slope))
            se.append(float((r.slope_t_hi - r.slope_t_lo) / (2 * t)))
        b, se = np.array(b), np.array(se)
        w = 1 / se ** 2
        fe = (w * b).sum() / w.sum()
        q, k = float((w * (b - fe) ** 2).sum()), len(b)
        tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        wr = 1 / (se ** 2 + tau2)
        mu = (wr * b).sum() / wr.sum()
        s2 = (wr * (b - mu) ** 2).sum() / ((k - 1) * wr.sum())
        half = stats.t.ppf(0.975, k - 1) * np.sqrt(s2)
        out[tag] = {"k": k, "n_units": int(sum(prim.loc[x].n_units for x in keys)), "slope_random_effects": float(mu), "normal_ci": [float(mu - 1.96 / np.sqrt(wr.sum())), float(mu + 1.96 / np.sqrt(wr.sum()))],
                    "hartung_knapp_ci": [float(mu - half), float(mu + half)], "Q": q, "df": k - 1, "p_Q": float(stats.chi2.sf(q, k - 1)), "tau2": float(tau2)}
    return out


def fzg_ordinary(prim_dir):
    """F27: the 11 ordinary pairs (without the two scuffed pairs and the 7-mg pair): permutation interval of the slope and p of slope = 1."""
    u = pd.read_csv(PAIRED / "units.csv")
    f = u[(u.set == "fzg") & ~u.unit.isin(["CM_pl", "E1_pI", "E2_pI"])]
    s = pp.slope(f.log10_D, f.log10_W)
    return {"n": s["n"], "slope": s["slope"], "perm_ci": [s["perm_lo"], s["perm_hi"]], "t_ci": [s["t_lo"], s["t_hi"]], "p_slope_1": s["p_slope_eq_ref"], "p_slope_0": s["p_slope_eq_0"],
            "spearman": pp.spearman(f.log10_D, f.log10_W)["rho"], "spearman_p": pp.spearman(f.log10_D, f.log10_W)["p"], "wear_range_mg": [float(10 ** f.log10_W.min()), float(10 ** f.log10_W.max())]}


def fold_table(loo):
    """F29: median, mean, share within a factor of 2 and worst factor of the debris prediction (leave-one-unit-out) per set with a score."""
    out = {}
    for (st, se), e in loo.groupby(["study", "set"], sort=False):
        e = e[e.M2.notna()]
        if len(e) and e.unit.nunique() >= 5:
            m = px.unit_errors(e, "M2").to_numpy()
            out[f"{st}/{se}"] = {"n": len(m), "median": float(10 ** np.median(m)), "mean": float(10 ** m.mean()), "worst": float(10 ** m.max()), "within_2": int((m <= np.log10(2)).sum())}
    return out


def gain_shares():
    """F18/F41: share of the summed leave-one-out difference (debris minus opponent) that the three extreme pairs carry."""
    f = pd.read_csv(DATA / "feup_martins2008" / "paired.csv")
    f = f[f.set == "fzg"].copy()
    f["y"], f["x"] = np.log10(f.wear), np.log10(f.debris)
    f["archard"] = np.log10(f.cond_cycles_M * f.cond_hertz_MPa**2)
    deb, ext = px.loo_fit(f, "x"), ["CM_pl", "E1_pI", "E2_pI"]
    return {k: float((deb - o)[ext].sum() / (deb - o).sum()) for k, o in (("constant", px.loo_fit(f)), ("conditions", px.conditions_loo(f)), ("archard", px.loo_fit(f, "archard")))}


def fzg_group_intervals(u):
    """F34: pooled-variance t intervals of the group ratios W/D inside the FZG set (unit means of the log ratio, two groups)."""
    f = u[u.set == "fzg"].set_index("unit")
    ld = f.log10_W - f.log10_D
    groups = {"ADI": ["AM_mp", "AE_mp", "AM_pl", "AE_pl"], "micropitting": ["CM_mp", "CE_mp", "AM_mp", "AE_mp"],
              "ester": ["CE_mp", "AE_mp", "CE_pl", "ME_pl", "AE_pl", "E1_pI", "E2_pI", "E3_pI"], "coated": ["MM_pl", "ME_pl"]}
    out = {}
    for k, m in groups.items():
        a, b = ld[m].to_numpy(), ld.drop(m).to_numpy()
        df = len(a) + len(b) - 2
        d = a.mean() - b.mean()
        s2 = (a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / df
        h = stats.t.ppf(0.975, df) * np.sqrt(s2 * (1 / len(a) + 1 / len(b)))
        out[k] = {"n": [len(a), len(b)], "ratio": float(10 ** d), "pooled_ci": [float(10 ** (d - h)), float(10 ** (d + h))], "df": df}
    return out


def later_checks():
    """Re-review 2: C07 printed Holzhauer slopes; C09 Krasmik per pairing and sensor steps; C10 Nield with material and humidity baselines."""
    out = {}
    sl = np.log10([1.63, 1.41, 1.12, 2.45])                      # Table 1 of the source (concentration per mass loss)
    out["holzhauer_printed_slopes"] = {"slopes": [1.63, 1.41, 1.12, 2.45], "gsd": float(10 ** sl.std(ddof=1)), "gsd_ci": gsd_ci(sl.std(ddof=1), 4), "max_over_min": float(10 ** np.ptp(sl))}
    h = pd.read_csv(DATA / "holzhauer1983" / "later.csv").sort_values(["unit", "order"]).groupby("unit")
    end, beg = h.last(), h.first()
    r = np.log10(end.conc_ppm / end.wear_mg)
    r2 = np.log10((end.conc_ppm - beg.conc_ppm) / end.wear_mg)
    lots = r[["A", "D"]].mean() - r[["B", "C"]].mean()
    out["holzhauer"] = {"end_ratio_start_subtracted": (10 ** r2).round(3).to_dict(), "lot_ratio_AD_over_BC": float(10 ** lots)}
    k = pd.read_csv(DATA / "krasmik2018" / "later.csv")
    k = k[k.hour >= 1]
    per = {}
    for u, g in k.sort_values("hour").groupby("unit"):
        g = g.sort_values("hour")
        dn, dv = np.diff(g.n_per_image), np.diff(g.wear_volume_mm3)
        per[u] = {"increments": int(len(dv)), "spearman": float(stats.spearmanr(dn, dv)[0]), "wear_volume_end_mm3": float(g.wear_volume_mm3.iloc[-1]), "n_tests": 1}
    out["krasmik_per_pairing"] = per
    d = pd.read_csv(DATA / "nield1961" / "later.csv")
    y, x, hum = np.log10(d.wear_rate_mg_h.to_numpy()), np.log10(d.ratio_X.to_numpy()), d.cond_rh_pct.to_numpy(float)
    mat = pd.get_dummies(d.material).to_numpy(float)[:, 1:]
    hl = pd.get_dummies(d.cond_rh_pct.astype(int).astype(str)).to_numpy(float)[:, 1:]
    one = np.ones((len(d), 1))
    X = {"constant": one, "humidity": np.column_stack([one, hum]), "material": np.column_stack([one, mat]), "material_plus_humidity": np.column_stack([one, hum, mat]),
         "humidity_levels": np.column_stack([one, hl]), "humidity_levels_plus_material": np.column_stack([one, hl, mat]), "ratio": np.column_stack([one, x])}
    err = {}
    for kk, A in X.items():
        lev = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
        if (lev > 1 - 1e-6).any():
            continue
        err[kk] = np.abs((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) / (1 - lev))
    out["nield"] = {"mae_log10": {kk: float(v.mean()) for kk, v in err.items()}, "n": len(d), "ratio_min_max": [float(d.ratio_X.min()), float(d.ratio_X.max())]}
    for kk in err:
        if kk != "ratio":
            out["nield"][f"ratio_minus_{kk}"] = pp.mean_diff(err["ratio"] - err[kk])
    return out


def factors():
    """C26: the calibrated factor itself: geometric mean of wear / debris per set (unit means) with a t interval, GSD of D/W, 95 % range, coverage."""
    u = pd.read_csv(PAIRED / "units.csv")
    loo = pd.read_csv(PAIRED / "loo_predictions.csv")
    src = {}
    for (st, se), g in u.groupby(["study", "set"], sort=False):
        if len(g) < 5:
            continue
        lw = (g.log10_W - g.log10_D).to_numpy()
        n, m, s = len(lw), lw.mean(), lw.std(ddof=1)
        tq = stats.t.ppf(0.975, n - 1)
        info = pd.read_csv(DATA / st / "paired.csv")
        info = info[info.set == se].iloc[0]
        src[f"{st}/{se}"] = {"n": n, "factor_wear_per_debris": float(10 ** m), "ci95_mean": [float(10 ** (m - tq * s / np.sqrt(n))), float(10 ** (m + tq * s / np.sqrt(n)))], "debris_unit": str(info.debris_unit), "wear_unit": str(info.wear_unit),
                             "gsd_factor": float(10 ** s), "gsd_ci": gsd_ci(s, n), "pi95_factor": float(10 ** (tq * s * np.sqrt(1 + 1 / n)))}
    cover = {}
    for (st, se), e in loo.groupby(["study", "set"], sort=False):
        e = e[e.M2.notna()]
        k = f"{st}/{se}"
        if k in src:
            m2 = px.unit_errors(e, "M2").to_numpy()
            cover[k] = {"n": int(len(m2)), "outside_95_range": int((m2 > np.log10(src[k]["pi95_factor"])).sum()), "within_2": int((m2 <= np.log10(2)).sum())}
    return {"per_set": src, "coverage_of_95_range": cover}


def main():
    u = pd.read_csv(PAIRED / "units.csv")
    summ = pd.read_csv(PAIRED / "summary.csv")
    prim = summ[summ.analysis == "primary"].set_index(["study", "set"])
    loo = pd.read_csv(PAIRED / "loo_predictions.csv")
    sens = summ[summ.DoverW_gsd.notna()][["study", "set", "analysis", "DoverW_gsd", "n_units"]]
    res = {"status": "post hoc, added after the re-review of 2026-11-25", "identity": identity_table(u), "trend": trend(u), "bearing_rig_zero_shift": zero_shift(u), "repeatability": repeatability(u),
           "nasa_yield": nasa(ROOT / "outputs" / "open_data" / "oil" / "nasa_experiments_tm2003_table3_9.csv"), "verdict_counts": verdicts(prim), "pooled_slope": pooled(prim),
           "fzg_ordinary_pairs": fzg_ordinary(prim), "fold_table": fold_table(loo), "gain_share_of_three_extremes": gain_shares(), "later_checks": later_checks(), "factors": factors(), "fzg_group_ratios": fzg_group_intervals(u),
           "gsd_sensitivity": {f"{r.study}/{r.set}/{r.analysis}": float(r.DoverW_gsd) for r in sens.itertuples()}}
    save_json(res, PAIRED / "extra" / "review.json")
    print(json.dumps({k: res[k] for k in res if k != "gsd_sensitivity"}, indent=1)[:12000])


if __name__ == "__main__":
    main()
