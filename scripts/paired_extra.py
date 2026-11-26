"""Additional analyses of the published paired data, ADDED AFTER an audit of the first run (status: post hoc, after the
results were known; nothing here replaces scripts/paired_analysis.py). Answers the audit's points:
  1 the bootstrap interval of the FZG gain with B = 10,000 depended on the seed -> converged intervals (B = 2,000,000);
  2 "a factor of 1.6" is a mean -> median, share within a factor of 2, worst case;
  3 leverage of the extreme gear tests (two scuffed, one of 7 mg); scoring on unit means;
  4 fairer opponents: one-constant Archard (load x cycles, load ~ Hertz pressure squared for a line contact) and an oracle
    that knows whether the gears scuffed;
  5 the five inferential slopes pooled (fixed effect, DerSimonian-Laird, prediction interval for a new rig) and the
    errors-in-variables bracket of the FZG slope.

    .venv/Scripts/python.exe -I scripts/paired_extra.py     # -> outputs/open_data/paired/extra/{results.json, fzg_table.csv}
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # -I drops the cwd
from wearpg import paired as pp  # noqa: E402
from wearpg.stats import holm  # noqa: E402

PAIRED = ROOT / "outputs" / "open_data" / "paired"
OUT = PAIRED / "extra"
B, SEED = 2_000_000, 0
SETS = [("feup_martins2008", "fzg"), ("feup_martins2008", "fourball"), ("phm_raadnui2015", "dry_fourball"),
        ("soton_craig2010", "bearings"), ("uct_jordan1995", "grease")]


def boot_ci(d, b=B, seed=SEED, chunk=200_000):
    """Percentile interval of the mean of d over units, with b resamples of the units."""
    d, rng, m = np.asarray(d, float), np.random.default_rng(seed), []
    for _ in range(0, b, chunk):
        m.append(d[rng.integers(0, len(d), size=(chunk, len(d)))].mean(1))
    return [float(x) for x in np.percentile(np.concatenate(m), [2.5, 97.5])]


def signflip(d):
    """Exact two-sided sign-flip p-value of the mean of d."""
    d = np.asarray(d, float)
    s = np.array(list(itertools.product((1, -1), repeat=len(d))))
    v = np.abs((s * d).mean(1))
    return float((v >= abs(d.mean()) - 1e-12).mean())


def unit_errors(loo, model):
    return (np.log10(loo.W) - loo[model]).abs().groupby(loo.unit).mean()


def loo_fit(df, x=None):
    """Leave-one-unit-out predictions of y from one constant (x None) or from x with slope 1 and one constant."""
    pred = pd.Series(index=df.index, dtype=float)
    for u in df.unit.unique():
        tr, te = df[df.unit != u], df[df.unit == u]
        if x is None:
            pred[te.index] = tr.y.mean()
        else:
            pred[te.index] = te[x] + (tr.y - tr[x]).mean()
    return (df.y - pred).abs().groupby(df.unit).mean()


def paired_summary(a, b):
    d = (a - b).to_numpy()
    return {"diff": float(d.mean()), "ci": boot_ci(d), "p_signflip": signflip(d), "n_units": len(d)}


def slope_perm(x, y, n_perm=199_999, seed=0):
    """OLS slope and a permutation p-value of slope = 1 (random orderings)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = stats.linregress(x, y)
    e, rng = y - x, np.random.default_rng(seed)
    xc = x - x.mean()
    obs = abs((xc * e).sum())
    cnt = sum(abs((xc * rng.permutation(e)).sum()) >= obs - 1e-12 for _ in range(n_perm // 10))
    t = stats.t.ppf(0.975, len(x) - 2)
    return {"slope": float(f.slope), "t_lo": float(f.slope - t * f.stderr), "t_hi": float(f.slope + t * f.stderr),
            "p_slope_1": (1 + cnt) / (1 + n_perm // 10)}


def loo_means(y, x=None):
    """Leave-one-out mean absolute error on unit means: constant (x None) or slope 1 with one constant."""
    v = y if x is None else y - x
    n = len(v)
    return float(np.abs(v - v.mean()).mean() * n / (n - 1))  # identity: leaving one out scales every deviation by n / (n - 1)


def spread(name, x, y):
    """Range of wear against scatter of debris per unit wear; debris beats a constant on unit means iff mad_R < mad_W (identity)."""
    r, n = x - y, len(x)
    sw, sr = y.std(ddof=1), r.std(ddof=1)
    e0, e2 = loo_means(y), loo_means(y, x)
    ci = sr * np.sqrt((n - 1) / stats.chi2.ppf([0.975, 0.025], n - 1))
    return {"set": name, "n": n, "decades_wear": float(np.ptp(y)), "sd_log_wear": float(sw), "sd_log_D_over_W": float(sr), "ratio": float(sw / sr),
            "gsd": float(10 ** sr), "gsd_ci": [float(v) for v in 10 ** ci], "constant_factor": 10 ** e0, "debris_factor": 10 ** e2,
            "gain_log10": e2 - e0, "pi95_factor": float(10 ** (stats.t.ppf(0.975, n - 1) * sr * np.sqrt(1 + 1 / n)))}


def criterion(u):
    out = []
    for (st, se), g in u.groupby(["study", "set"], sort=False):
        if len(g) >= 5:
            out.append(spread(f"{st}/{se}", g.log10_D.to_numpy(), g.log10_W.to_numpy()))
    f = u[u.set == "fzg"]
    g = u[u.set == "grease"]
    for name, q in (("fzg without scuffed", f[~f.unit.isin(["CM_pl", "E1_pI"])]), ("fzg without scuffed and 7 mg", f[~f.unit.isin(["CM_pl", "E1_pI", "E2_pI"])]),
                    ("grease without floor readings", g[g.log10_D > 0])):
        out.append(spread(name, q.log10_D.to_numpy(), q.log10_W.to_numpy()))
    for o in out:                                          # the rule is an identity on unit means
        assert (o["gain_log10"] < 0) == (o["debris_factor"] < o["constant_factor"])
    return out


def groups(u):
    """FZG: factor W/D of one group against the other (exact permutation) and leave-group-out prediction (ADI planned, others exploratory)."""
    f = u[u.set == "fzg"].set_index("unit")
    ld = f.log10_W - f.log10_D
    sets = {"ADI": ["AM_mp", "AE_mp", "AM_pl", "AE_pl"], "micropitting": ["CM_mp", "CE_mp", "AM_mp", "AE_mp"],
            "ester": ["CE_mp", "AE_mp", "CE_pl", "ME_pl", "AE_pl", "E1_pI", "E2_pI", "E3_pI"], "coated": ["MM_pl", "ME_pl"]}
    out = {}
    for name, members in sets.items():
        a, b = ld[members].to_numpy(), ld.drop(members).to_numpy()
        allv, k, d = np.concatenate([a, b]), len(a), abs(a.mean() - b.mean())
        c = [abs(allv[list(i)].mean() - np.delete(allv, list(i)).mean()) for i in itertools.combinations(range(len(allv)), k)]
        out[name] = {"n_1": k, "n_0": len(b), "ratio_W_over_D": float(10 ** (a.mean() - b.mean())), "p_exact": float(np.mean(np.array(c) >= d - 1e-12)),
                     "error_group_from_rest": float(10 ** np.abs(a - b.mean()).mean()), "error_rest_from_group": float(10 ** np.abs(b - a.mean()).mean())}
    return out


def conditions_loo(f):
    Z, _ = pp.conditions_matrix(f, [["cond_cycles_M", "log10"], ["cond_hertz_MPa", "log10"], ["cond_ester", "id"], ["cond_adi", "id"]], f.unit.nunique())
    pr = pp.loo_predictions(f.x.to_numpy(), f.y.to_numpy(), Z, f.unit.to_numpy())
    return (f.y.to_numpy() - pr.M1).abs().groupby(f.unit.to_numpy()).mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"status": "post hoc, added after an audit of the first run (2026-11-24); B = 2,000,000 for the intervals"}
    loo = pd.read_csv(PAIRED / "loo_predictions.csv")
    summ = pd.read_csv(PAIRED / "summary.csv")
    prim = summ[summ.analysis == "primary"].set_index(["study", "set"])

    res["converged"] = {}
    for st, se in SETS:
        e = loo[(loo.study == st) & (loo.set == se)]
        r = {}
        err = {m: unit_errors(e, m) for m in ("M0", "M1", "M2", "M3", "R2") if m in e and e[m].notna().all()}
        assert abs(err["M2"].mean() - prim.loc[(st, se), "mae_M2"]) < 1e-4
        r["debris_minus_constant"] = paired_summary(err["M2"], err["M0"])
        if "M1" in err:
            r["debris_minus_conditions"] = paired_summary(err["M2"], err["M1"])
            best = err["M1"] if err["M1"].mean() < err["M2"].mean() else err["M2"]
            r["both_minus_best_single"] = paired_summary(err["M3"], best)
        if "R2" in err:
            r["resolved_minus_total"] = paired_summary(err["R2"], err["M2"])
        m2 = err["M2"].to_numpy()
        r["debris_error"] = {"mean_factor": float(10 ** m2.mean()), "median_factor": float(10 ** np.median(m2)),
                             "worst_factor": float(10 ** m2.max()), "within_2": int((m2 <= np.log10(2)).sum()), "n": len(m2)}
        res["converged"][f"{st}/{se}"] = r

    f = pd.read_csv(ROOT / "data" / "real" / "paired" / "feup_martins2008" / "paired.csv")
    f = f[f.set == "fzg"].copy()
    f["y"], f["x"] = np.log10(f.wear), np.log10(f.debris)
    f["archard"] = np.log10(f.cond_cycles_M * f.cond_hertz_MPa ** 2)
    um = f.groupby("unit", as_index=False)[["x", "y", "archard"]].mean()
    res["fzg_archard"] = {"archard_factor": float(10 ** loo_fit(f, "archard").mean()),
                          "constant_factor": float(10 ** loo_fit(f).mean()), "debris_factor": float(10 ** loo_fit(f, "x").mean()),
                          "debris_minus_archard": paired_summary(loo_fit(f, "x"), loo_fit(f, "archard"))}
    sc = f.assign(x2=f.class_scuffed if f.class_scuffed.dtype != object else f.class_scuffed.astype(float))
    scuffed = sorted(sc.loc[sc.class_scuffed == 1, "unit"].unique()) or ["CM_pl", "E1_pI"]
    pred = {}
    for u in f.unit.unique():                                  # oracle: constant among scuffed / among not scuffed tests
        grp = f.unit.isin(scuffed)
        te = f.unit == u
        tr = f[(grp == grp[te].iloc[0]) & ~te]
        pred[u] = tr.y.mean() if len(tr) else f[~te].y.mean()
    err_or = (f.y - f.unit.map(pred)).abs().groupby(f.unit).mean()
    res["fzg_scuffing_oracle"] = {"factor": float(10 ** err_or.mean()), "debris_minus_oracle": paired_summary(loo_fit(f, "x"), err_or),
                                  "scuffed_units": scuffed}
    res["fzg_archard_leverage"] = {
        name: {"archard_factor": float(10 ** loo_fit(f[~f.unit.isin(drop)], "archard").mean()),
               "debris_factor": float(10 ** loo_fit(f[~f.unit.isin(drop)], "x").mean()),
               "debris_minus_archard": paired_summary(loo_fit(f[~f.unit.isin(drop)], "x"), loo_fit(f[~f.unit.isin(drop)], "archard"))}
        for name, drop in (("without_scuffed", scuffed), ("without_scuffed_and_7mg", scuffed + ["E2_pI"]))}
    res["fzg_leverage"] = {}
    for name, drop in (("all", []), ("without_scuffed", scuffed), ("without_scuffed_and_7mg", scuffed + ["E2_pI"])):
        s = f[~f.unit.isin(drop)]
        u = um[~um.unit.isin(drop)]
        res["fzg_leverage"][name] = {
            "n_units": len(u), "wear_range_mg_unit_means": [float(10 ** u.y.min()), float(10 ** u.y.max())],
            **slope_perm(u.x, u.y), "spearman": float(stats.spearmanr(u.x, u.y)[0]),
            "constant_factor": float(10 ** loo_fit(s).mean()), "debris_factor": float(10 ** loo_fit(s, "x").mean()),
            "debris_minus_constant": paired_summary(loo_fit(s, "x"), loo_fit(s)),
            "unit_means": {"constant_factor": float(10 ** loo_fit(u).mean()), "debris_factor": float(10 ** loo_fit(u, "x").mean()),
                           "debris_minus_constant": paired_summary(loo_fit(u, "x"), loo_fit(u))}}
    res["fzg_unit_mean_scoring"] = {
        "constant_factor": float(10 ** loo_fit(um).mean()), "debris_factor": float(10 ** loo_fit(um, "x").mean()),
        "debris_minus_constant": paired_summary(loo_fit(um, "x"), loo_fit(um))}
    # ---- added after the reviews of 2026-11-24 (post hoc, exploratory)
    u = pd.read_csv(PAIRED / "units.csv")
    res["criterion"], res["fzg_groups"] = criterion(u), groups(u)
    fb = u[u.set == "fourball"].assign(oil=lambda d: d.unit.str[:2], r=lambda d: d.log10_D - d.log10_W)
    dup = fb.groupby("oil").r.agg(lambda v: v.iloc[0] - v.iloc[1])
    k9 = f[f.row.str.contains("K9")].assign(r=lambda d: d.x - d.y).groupby("unit").r.agg(lambda v: v.iloc[0] - v.iloc[1])
    res["repeatability"] = {"fourball_duplicates_ratio_D_over_W": {k: float(10 ** abs(v)) for k, v in dup.items()},
                            "fourball_replicate_gsd": float(10 ** np.sqrt((dup ** 2).sum() / (2 * len(dup)))), "fourball_replicate_df": len(dup),
                            "fzg_repeated_K9_ratio_D_over_W": {k: float(10 ** abs(v)) for k, v in k9.items()}}
    tot = f.groupby("unit")[["debris", "wear"]].sum()
    lt = np.log10(tot)
    td = np.abs(lt.wear - lt.debris - (lt.wear - lt.debris).mean()) * 14 / 13
    tc = np.abs(lt.wear - lt.wear.mean()) * 14 / 13
    res["fzg_run_totals"] = {"slope": float(stats.linregress(lt.debris, lt.wear).slope), "spearman": float(stats.spearmanr(lt.debris, lt.wear)[0]),
                             "debris_factor": float(10 ** td.mean()), "constant_factor": float(10 ** tc.mean()), "debris_minus_constant": paired_summary(td, tc),
                             "micropitting_totals_mg": {k: float(v) for k, v in tot.wear.items() if k.endswith("_mp")}}
    lev = {}
    for name, drop in (("all", []), ("without_scuffed", scuffed), ("without_scuffed_and_7mg", scuffed + ["E2_pI"])):
        s = f[~f.unit.isin(drop)].reset_index(drop=True)
        ec, ed = conditions_loo(s), loo_fit(s, "x")
        lev[name] = {"conditions_factor": float(10 ** ec.mean()), "debris_factor": float(10 ** ed.mean()), "debris_minus_conditions": paired_summary(ed, ec)}
    res["fzg_conditions_leverage"] = lev
    pv = {"constant": res["converged"]["feup_martins2008/fzg"]["debris_minus_constant"]["p_signflip"],
          "conditions": res["converged"]["feup_martins2008/fzg"]["debris_minus_conditions"]["p_signflip"],
          "archard": res["fzg_archard"]["debris_minus_archard"]["p_signflip"]}
    res["fzg_holm_three_opponents"] = {k: {"p": pv[k], "p_holm": float(h)} for k, h in zip(pv, holm(list(pv.values())))}
    q = prim.loc[("feup_martins2008", "fzg")]
    res["fzg_errors_in_variables"] = {"ols": float(q.slope), "reduced_major_axis": float(q.slope_rma), "inverse": float(q.slope_inverse)}

    rows = []
    for st, se in SETS:
        r = prim.loc[(st, se)]
        t = stats.t.ppf(0.975, r.n_units - 2)
        rows.append((st + "/" + se, float(r.slope), float((r.slope_t_hi - r.slope_t_lo) / (2 * t)), int(r.n_units)))
    b, se_ = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    w = 1 / se_ ** 2
    fe = (w * b).sum() / w.sum()
    Q = float((w * (b - fe) ** 2).sum())
    k = len(b)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (se_ ** 2 + tau2)
    re_ = (wr * b).sum() / wr.sum()
    ser = np.sqrt(1 / wr.sum())
    t = stats.t.ppf(0.975, k - 2)
    res["pooled_slope"] = {"sets": rows, "n_tests": int(sum(r[3] for r in rows)), "fixed": float(fe),
                           "fixed_ci": [float(fe - 1.96 / np.sqrt(w.sum())), float(fe + 1.96 / np.sqrt(w.sum()))],
                           "Q": Q, "df": k - 1, "p_Q": float(stats.chi2.sf(Q, k - 1)), "tau2": float(tau2),
                           "random": float(re_), "random_ci": [float(re_ - 1.96 * ser), float(re_ + 1.96 * ser)],
                           "prediction_interval": [float(re_ - t * np.sqrt(tau2 + ser ** 2)), float(re_ + t * np.sqrt(tau2 + ser ** 2))],
                           "weight_fzg_fixed": float(w[0] / w.sum())}
    tab = f[["unit", "row", "mode", "material_pair", "lubricant", "cond_cycles_M", "cond_hertz_MPa", "debris", "wear"]].copy()
    tab["pred_factor_error"] = 10 ** (np.log10(tab.wear) - loo[(loo.study == "feup_martins2008") & (loo.set == "fzg")].set_index("row").M2.reindex(tab.row).to_numpy())
    tab.to_csv(OUT / "fzg_table.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:9000])


if __name__ == "__main__":
    main()
