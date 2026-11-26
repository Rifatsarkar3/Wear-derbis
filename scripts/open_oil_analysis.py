"""ODA analyses on real open oil-debris data (pre-specified plan: outputs/agent_logs/ODA.md).

A. Zenodo 10.5281/zenodo.17287193 tractor engine oil (27 tractors x 5 samples): data audits, DESIGN sec. 3
   cumulative-mode debris wear-rate proxy, phase contrasts, leave-one-tractor-out transition alarms, wear-metal
   composition vs Fe-only classification.
B. NASA spur-gear oil debris (TM-2003-211307 tables, transcribed): debris mass vs destructive pitting, run-time
   baseline, onset detection at inspections, mass per pitted tooth.
C. Harvard Dataverse 10.7910/DVN/SO9SXL online lubricant sensors: descriptive only (no wear label exists).

Writes CSV/JSON to outputs/open_data/oil/ (derived results; raw data stay in data/real/, never redistributed).
Usage: .venv/Scripts/python.exe -I scripts/open_oil_analysis.py [--root "<project root>"] [--n-perm 999]
       [--skip-dataverse]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # -I drops the cwd; the editable install also provides wearpg

from wearpg import open_oil as oo  # noqa: E402
from wearpg.metrics import classification_metrics  # noqa: E402
from wearpg.stats import cluster_bootstrap, paired_cluster_bootstrap, spearman_permutation  # noqa: E402
from wearpg.utils import save_json  # noqa: E402

ADDITIVE_COLS = ["Mo", "Si", "Na", "B", "Zn", "P", "Ca", "Mg", "Mn", "vis40"]  # identity audit (4-digit/3-digit values)
CONTRASTS = [("wear_out", "useful_life"), ("useful_life", "infant_mortality"), ("wear_out", "infant_mortality")]
N_BOOT = 2000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ A. Zenodo
def zenodo(root: Path, out: Path, n_perm: int) -> dict:
    raw = root / "data/real/zenodo_wear_debris/raw/wear_derbis.xlsx"
    s = oo.load_zenodo_wear_debris(raw)
    ext = root / "data/real/zenodo_wear_debris/extracted"
    ext.mkdir(parents=True, exist_ok=True)
    s.to_csv(ext / "zenodo_samples_tidy.csv", index=False)
    summ = oo.load_zenodo_phase_summary(raw)
    res: dict = {"n_samples": len(s), "n_units": s.unit_id.nunique(),
                 "units_per_phase": s.groupby("phase").unit_id.nunique().to_dict(), "phase_source": oo.PHASE_SOURCE}
    units = s.groupby("unit_id").agg(block=("block", "first"), tractor_age=("tractor_age", "first"),
                                     phase=("phase", "first"))

    # --- audits
    rep = oo.summary_reproduction(s, summ)
    rep.to_csv(out / "zenodo_sheet2_reproduction.csv", index=False)
    informative = rep[rep.sheet2_value != 0]
    res["sheet2"] = {ph: {"n_cells_nonzero": int(len(g)), "n_reproduced_by_one_unit_mean": int((g.n_match > 0).sum()),
                          "units_matched": sorted(set(";".join(g.matching_units).split(";")) - {""})}
                     for ph, g in informative.groupby("phase")}
    ident = oo.pairwise_identity_audit(s, ADDITIVE_COLS)
    ident.to_csv(out / "zenodo_identity_pairs.csv", index=False)
    ident_sum = ident.assign(ge5=ident.n_identical >= 5).groupby("same_phase").agg(
        n_pairs=("n_identical", "size"), mean_identical=("n_identical", "mean"), max_identical=("n_identical", "max"),
        share_ge5_of_10=("ge5", "mean")).reset_index()
    ident_sum.to_csv(out / "zenodo_identity_summary.csv", index=False)
    res["identity_audit"] = ident_sum.to_dict("records")
    flags = oo.outlier_audit(s)
    flags.to_csv(out / "zenodo_outlier_flags.csv", index=False)
    gross = sorted(set(flags.loc[(flags.column == "Fe") | (flags.rule == "ratio_to_phase_hour_median"), "unit_id"]))
    res["outlier_flags"] = flags.to_dict("records")
    res["sensitivity_excluded_units"] = gross

    # --- cumulative observation model (primary q = 0; sensitivity q = 0.02)
    sets = {"Fe": ["Fe"], "wear_metals": list(oo.WEAR_METALS)}
    rate_rows, running = [], {}
    for name, cols in sets.items():
        for q in (0.0, 0.02):
            per, unit = oo.debris_rates(s, cols, q=q)
            running[(name, q)] = per
            rate_rows.append(unit[["rate_cum", "rate_ols"]].add_suffix(f"_{name}_q{q:g}"))
    rates = units.join(rate_rows)
    rates.to_csv(out / "zenodo_unit_rates.csv")
    running[("Fe", 0.0)].to_csv(out / "zenodo_running_rates_Fe.csv", index=False)

    # --- phase contrasts (debris proxies + age/hours baselines in the same table)
    feats = {f"rate_cum_{n}_q{q:g}": True for n in sets for q in (0.0, 0.02)}
    feats.update({f"rate_ols_{n}_q0": True for n in sets})
    rows = []
    for subset, keep in (("all_units", rates.index), ("excl_flagged", rates.index.difference(gross))):
        r = rates.loc[keep]
        for f, lg in [*feats.items(), ("tractor_age", False)]:
            lg = lg and bool((rates[f] > 0).all())  # OLS slopes can be <= 0 (T13 Cr cell) -> linear scale
            for a, b in CONTRASTS:
                va, vb = r.loc[r.phase == a, f].to_numpy(), r.loc[r.phase == b, f].to_numpy()
                rows.append({"subset": subset, "feature": f, "a": a, "b": b,
                             **oo.group_contrast(va, vb, log10=lg, n_boot=N_BOOT, seed=0)})
        # working hours: every unit is sampled at 20..100 h -> identical unit feature (t_last = 100 h)
        for a, b in CONTRASTS:
            rows.append({"subset": subset, "feature": "working_hours_t_last", "a": a, "b": b, "n_a": 9, "n_b": 9,
                         "median_a": 100.0, "median_b": 100.0, "cliffs_delta": 0.0, "cliffs_delta_lo": 0.0,
                         "cliffs_delta_hi": 0.0, "perm_p": 1.0, "perm_kind": "by construction",
                         "scale": "linear"})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "zenodo_phase_contrasts.csv", index=False)
    log("zenodo contrasts done")

    # --- transition alarms (useful_life -> wear_out), leave-one-tractor-out
    lab_units = rates.index[rates.phase.isin(["useful_life", "wear_out"])]
    positive = rates.loc[lab_units, "phase"].eq("wear_out")
    infant = rates.index[rates.phase == "infant_mortality"].tolist()
    feat_specs = {  # name -> (unit feature, running frame, time col, value col)
        "rate_cum_Fe_q0": (rates["rate_cum_Fe_q0"], running[("Fe", 0.0)], "working_hours", "running_rate"),
        "rate_cum_wear_metals_q0": (rates["rate_cum_wear_metals_q0"], running[("wear_metals", 0.0)],
                                    "working_hours", "running_rate"),
        "rate_cum_Fe_q0.02": (rates["rate_cum_Fe_q0.02"], running[("Fe", 0.02)], "working_hours", "running_rate"),
        "age_only": (rates["tractor_age"], s.assign(v=s.tractor_age)[["unit_id", "working_hours", "v"]],
                     "working_hours", "v"),
        "working_hours_only": (pd.Series(100.0, index=rates.index), s.assign(v=s.working_hours)[
            ["unit_id", "working_hours", "v"]], "working_hours", "v"),
    }
    al_all, summ_rows = [], []
    for subset, drop in (("all_units", []), ("excl_flagged", gross)):
        lu = lab_units.difference(drop)
        for name, (uf, run, tcol, vcol) in feat_specs.items():
            al = oo.loo_threshold_alarms(uf, positive.loc[lu], run, tcol, vcol,
                                         extra_negatives=[u for u in infant if u not in drop])
            al.insert(0, "feature", name)
            al.insert(0, "subset", subset)
            al_all.append(al)
            summ_rows.append({"subset": subset, "feature": name, **oo.alarm_summary(al),
                              "tau_min": float(al.tau.min()), "tau_max": float(al.tau.max())})
    alarms = pd.concat(al_all, ignore_index=True)
    alarms.to_csv(out / "zenodo_transition_alarms.csv", index=False)
    tsum = pd.DataFrame(summ_rows)
    tsum.to_csv(out / "zenodo_transition_summary.csv", index=False)
    res["transition"] = tsum.to_dict("records")
    log("zenodo transition done")

    # --- composition (C2 analogue): LOO multinomial logistic regression, composition vs Fe-only
    comp = oo.composition_features(s)
    comp.join(units).to_csv(out / "zenodo_composition_features.csv")
    fe_feat = np.log10(rates["rate_cum_Fe_q0"]).rename("log10_rate_cum_Fe")
    variants = {"composition": comp, "fe_only": fe_feat.to_frame(), "composition_plus_fe": comp.join(fe_feat)}
    tasks = {"3class": list(oo.PHASES), "infant_vs_useful": ["infant_mortality", "useful_life"]}
    comp_rows, pred_frames = [], {}
    for subset, drop in (("all_units", []), ("excl_flagged", gross)):
        for task, classes in tasks.items():
            idx = units.index[units.phase.isin(classes)].difference(drop)
            y = units.loc[idx, "phase"].map(classes.index).to_numpy()
            for vname, X in variants.items():
                Xv = X.loc[idx].to_numpy()
                proba = oo.loo_proba(Xv, y, len(classes))
                df = pd.DataFrame(proba, columns=[f"p{k}" for k in range(len(classes))])
                df.insert(0, "y_true", y)
                df.insert(0, "unit_id", idx)
                pred_frames[(subset, task, vname)] = df
                pcols = [f"p{k}" for k in range(len(classes))]

                def mfn(d, pcols=pcols, classes=classes):
                    m = classification_metrics(d.y_true.to_numpy(), d[pcols].to_numpy(), classes=tuple(classes))
                    return {"balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"]}

                bs = cluster_bootstrap(mfn, df, "unit_id", n_boot=N_BOOT, seed=0)
                pre = subset == "all_units" and vname != "composition_plus_fe"
                p_perm = oo.permutation_balanced_accuracy(Xv, y, len(classes), n_perm=n_perm, seed=0)[1] if pre \
                    else float("nan")
                m = classification_metrics(y, proba, classes=tuple(classes))
                comp_rows.append({"subset": subset, "task": task, "features": vname, "n_units": len(idx),
                                  "pre_specified": pre, "balanced_accuracy": m["balanced_accuracy"],
                                  "ba_lo": bs["balanced_accuracy"]["lo"], "ba_hi": bs["balanced_accuracy"]["hi"],
                                  "macro_f1": m["macro_f1"], "f1_lo": bs["macro_f1"]["lo"], "f1_hi": bs["macro_f1"]["hi"],
                                  "perm_p_balanced_accuracy": p_perm, "n_perm": n_perm if pre else 0,
                                  "confusion": m["confusion"].tolist()})
                log(f"composition {subset} {task} {vname}: BA {m['balanced_accuracy']:.3f} p {p_perm}")
            a, b = pred_frames[(subset, task, "composition")], pred_frames[(subset, task, "fe_only")]
            pcols = [c for c in a.columns if c.startswith("p")]
            pb = paired_cluster_bootstrap(
                lambda d, pcols=pcols, classes=classes: classification_metrics(
                    d.y_true.to_numpy(), d[pcols].to_numpy(), classes=tuple(classes))["balanced_accuracy"],
                a, b, "unit_id", "unit_id", n_boot=N_BOOT, seed=0)["value"]
            comp_rows.append({"subset": subset, "task": task, "features": "composition_minus_fe_only",
                              "n_units": len(a), "pre_specified": subset == "all_units",
                              "balanced_accuracy": pb["diff"], "ba_lo": pb["lo"], "ba_hi": pb["hi"],
                              "perm_p_balanced_accuracy": pb["p_value"], "note": "paired unit bootstrap; p two-sided"})
    compres = pd.DataFrame(comp_rows)
    compres.to_csv(out / "zenodo_composition_results.csv", index=False)
    pd.concat([d.assign(subset=k[0], task=k[1], features=k[2]) for k, d in pred_frames.items()]).to_csv(
        out / "zenodo_composition_loo_predictions.csv", index=False)
    res["composition"] = compres.drop(columns=["confusion"], errors="ignore").to_dict("records")

    # composition shift per log-ratio: wear_out vs useful and useful vs infant (unit-level, linear scale of ALR)
    shift = [{"alr": c, "a": a, "b": b, **oo.group_contrast(comp.loc[units.phase == a, c], comp.loc[units.phase == b, c],
                                                           log10=False, n_boot=N_BOOT, seed=0)}
             for c in comp.columns for a, b in CONTRASTS]
    pd.DataFrame(shift).to_csv(out / "zenodo_composition_shift.csv", index=False)
    return res


# ------------------------------------------------------------------------------------------------ B. NASA
def _loo_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[i] = np.concatenate([[1.0], X[i]]) @ coef
    return pred


def nasa(out: Path) -> dict:
    ex = oo.nasa_experiments()
    ex.to_csv(out / "nasa_experiments_tm2003_table3_9.csv", index=False)
    oo.nasa_cross_check().to_csv(out / "nasa_cross_check_tm2001_vs_tm2003.csv", index=False)
    res: dict = {"n_experiments": len(ex), "stage_counts": ex.stage_end.value_counts().sort_index().to_dict()}

    # --- experiment level: debris mass vs destructive pitting; run time = time-only baseline
    corr = {}
    for f in ("mass_end_mg", "rdg_end"):
        sp = spearman_permutation(ex[f], ex.n_de_end, n_perm=20000, seed=0)
        bs = cluster_bootstrap(lambda d, f=f: sps.spearmanr(d[f], d.n_de_end)[0], ex, "exp", n_boot=N_BOOT, seed=0)
        corr[f] = {"spearman": sp["rho"], "perm_p": sp["p_value"], "lo": bs["value"]["lo"], "hi": bs["value"]["hi"],
                   "n": sp["n"]}
        sp2 = spearman_permutation(ex[f], ex.n_pitted_end, n_perm=20000, seed=0)
        corr[f]["spearman_vs_n_pitted"] = sp2["rho"]
        corr[f]["perm_p_vs_n_pitted"] = sp2["p_value"]
    res["spearman_vs_n_destructive_teeth"] = corr
    # within the destructively pitted experiments only (does mass rank the extent of damage?)
    d2 = ex[ex.stage_end == 2]
    res["spearman_within_destructive"] = {f: spearman_permutation(d2[f], d2.n_de_end, n_perm=20000, seed=0)
                                          for f in ("mass_end_mg", "rdg_end")}
    # partial association: rank of mass vs rank of damage after removing (linear) rank of run time
    rk = ex[["mass_end_mg", "rdg_end", "n_de_end"]].rank()
    resid = {c: rk[c] - np.polyval(np.polyfit(rk.rdg_end, rk[c], 1), rk.rdg_end) for c in ("mass_end_mg", "n_de_end")}
    res["partial_spearman_mass_damage_given_time"] = float(np.corrcoef(resid["mass_end_mg"], resid["n_de_end"])[0, 1])

    lab = ex.assign(y=(ex.stage_end == 2).astype(int))
    aucs = {}
    for f in ("mass_end_mg", "rdg_end"):
        fn = (lambda d, f=f: oo.auc(d.loc[d.y == 0, f], d.loc[d.y == 1, f]))
        bs = cluster_bootstrap(fn, lab, "exp", n_boot=N_BOOT, seed=0)
        aucs[f] = {"auc": bs["value"]["point"], "lo": bs["value"]["lo"], "hi": bs["value"]["hi"]}
    pa = paired_cluster_bootstrap(lambda d: oo.auc(d.loc[d.y == 0, "score"], d.loc[d.y == 1, "score"]),
                                  lab.assign(score=lab.mass_end_mg), lab.assign(score=lab.rdg_end), "exp", "exp",
                                  n_boot=N_BOOT, seed=0)["value"]
    aucs["mass_minus_time"] = {"diff": pa["diff"], "lo": pa["lo"], "hi": pa["hi"], "p_value": pa["p_value"]}
    res["auc_destructive_vs_not"] = aucs

    y = np.log1p(ex.n_de_end.to_numpy(float))
    feats = {"time_only": np.log1p(ex[["rdg_end"]].to_numpy(float)),
             "debris_only": np.log1p(ex[["mass_end_mg"]].to_numpy(float)),
             "debris_plus_time": np.log1p(ex[["mass_end_mg", "rdg_end"]].to_numpy(float))}
    preds = {}
    reg = {}
    for k, X in feats.items():
        n_hat = np.clip(np.expm1(_loo_linear(X, y)), 0, None)
        preds[k] = pd.DataFrame({"exp": ex.exp, "abs_err": np.abs(n_hat - ex.n_de_end), "n_hat": n_hat})
        reg[k] = {"loo_mae_teeth": float(preds[k].abs_err.mean()),
                  "loo_spearman": float(sps.spearmanr(n_hat, ex.n_de_end)[0])}
    for a, b in (("debris_only", "time_only"), ("debris_plus_time", "time_only")):
        pb = paired_cluster_bootstrap(lambda d: d.abs_err.mean(), preds[a], preds[b], "exp", "exp", n_boot=N_BOOT,
                                      seed=0)["value"]
        reg[f"{a}_minus_{b}_mae"] = {"diff": pb["diff"], "lo": pb["lo"], "hi": pb["hi"], "p_value": pb["p_value"]}
    res["loo_regression_n_destructive_teeth"] = reg
    pd.concat([d.assign(model=k) for k, d in preds.items()]).merge(ex[["exp", "n_de_end"]]).to_csv(
        out / "nasa_loo_regression_predictions.csv", index=False)

    # --- onset detection at inspections (C4 analogue), leave-one-experiment-out thresholds
    onset = {}
    reading_rows = []
    for variant in ("table_3_9", "table_3_14"):
        rd = oo.nasa_readings(exp5_rdg2566=variant)
        rd["y"] = (rd.stage == 2).astype(int)
        for feat in ("mass_mg", "reading"):
            alarm = np.zeros(len(rd), bool)
            tau = np.zeros(len(rd))
            for e in rd.exp.unique():
                tr, te = rd.exp != e, rd.exp == e
                t = oo.youden_threshold(rd.loc[tr & (rd.y == 0), feat], rd.loc[tr & (rd.y == 1), feat])
                alarm[te.to_numpy()] = rd.loc[te, feat].to_numpy() >= t
                tau[te.to_numpy()] = t
            r = rd.assign(feature=feat, variant=variant, alarm=alarm, tau=tau)
            reading_rows.append(r)
            per_exp = []
            for e, g in r.groupby("exp"):
                g = g.sort_values("reading").reset_index(drop=True)
                has = g.y.any()
                first_d = int(np.flatnonzero(g.y)[0]) if has else -1
                first_a = int(np.flatnonzero(g.alarm)[0]) if g.alarm.any() else -1
                per_exp.append({"exp": e, "destructive": bool(has), "alarmed": bool(g.alarm.any()),
                                "detected": bool((g.alarm & (g.y == 1)).any()),
                                "early_alarm": bool(has and first_a >= 0 and first_a < first_d),
                                "alarm_minus_first_destructive_inspections": (first_a - first_d)
                                if has and first_a >= 0 else np.nan, "n_inspections": len(g)})
            pe = pd.DataFrame(per_exp)
            pos, neg = pe[pe.destructive], pe[~pe.destructive]
            sens = float(r.loc[r.y == 1, "alarm"].mean())
            spec = float(1 - r.loc[r.y == 0, "alarm"].mean())
            fn = (lambda d, feat=feat: oo.auc(d.loc[d.y == 0, feat], d.loc[d.y == 1, feat]))
            bs = cluster_bootstrap(fn, rd, "exp", n_boot=N_BOOT, seed=0)
            onset[f"{variant}|{feat}"] = {
                "n_readings": len(r), "n_pos_readings": int(r.y.sum()), "reading_sensitivity": sens,
                "reading_specificity": spec, "reading_balanced_accuracy": (sens + spec) / 2,
                "auc_raw": bs["value"]["point"], "auc_lo": bs["value"]["lo"], "auc_hi": bs["value"]["hi"],
                "n_exp_destructive": len(pos), "n_exp_not": len(neg),
                "exp_detection_rate": float(pos.detected.mean()), "exp_false_alarm_rate": float(neg.alarmed.mean()),
                "exp_early_alarm_rate": float(pos.early_alarm.mean()),
                "median_alarm_minus_first_destructive_inspections": float(
                    pos.alarm_minus_first_destructive_inspections.median()),
                "tau_min": float(r.tau.min()), "tau_max": float(r.tau.max())}
            pe.assign(variant=variant, feature=feat).to_csv(out / f"nasa_onset_per_experiment_{variant}_{feat}.csv",
                                                             index=False)
        pa = paired_cluster_bootstrap(lambda d: oo.auc(d.loc[d.y == 0, "score"], d.loc[d.y == 1, "score"]),
                                      rd.assign(score=rd.mass_mg, key=np.arange(len(rd))),
                                      rd.assign(score=rd.reading, key=np.arange(len(rd))), "exp", "key",
                                      n_boot=N_BOOT, seed=0)["value"]
        onset[f"{variant}|auc_mass_minus_reading"] = {"diff": pa["diff"], "lo": pa["lo"], "hi": pa["hi"],
                                                      "p_value": pa["p_value"]}
    pd.concat(reading_rows).to_csv(out / "nasa_onset_readings.csv", index=False)
    res["onset_detection"] = onset

    # --- calibration: debris mass per destructively pitted tooth (TM-2003 p. 37 rule of thumb: ~9 mg if 0.0397 cm
    # pits densely covered 25 % of one tooth; the sensor counts only particles >= 125 um)
    cal = d2.assign(mg_per_destructive_tooth=d2.mass_end_mg / d2.n_de_end)
    v = cal.mg_per_destructive_tooth
    res["mass_per_destructive_tooth_mg"] = {"n": len(v), "min": float(v.min()), "median": float(v.median()),
                                            "max": float(v.max()), "cv": float(v.std(ddof=1) / v.mean()),
                                            "max_over_min": float(v.max() / v.min()),
                                            "share_above_9mg_rule_of_thumb": float((v > 9).mean())}
    cal.to_csv(out / "nasa_mass_per_destructive_tooth.csv", index=False)
    return res


# ------------------------------------------------------------------------------------------------ C. Dataverse
def dataverse(root: Path, out: Path) -> dict:
    ext = root / "data/real/dataverse_lubricating_oil/extracted"
    cache = ext / "lubricating_oil_parameters.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = oo.load_dataverse_oil(root / "data/real/dataverse_lubricating_oil/raw/Lubricating_Oil_Parameters_1.xlsx")
        ext.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    num = [c for c in oo.DATAVERSE_COLUMNS if c != "icm_isocode"]
    desc = pd.DataFrame({c: {"n": int(df[c].notna().sum()), "n_missing": int(df[c].isna().sum()),
                             "min": df[c].min(), "p01": df[c].quantile(0.01), "median": df[c].median(),
                             "p99": df[c].quantile(0.99), "max": df[c].max(), "n_unique": int(df[c].nunique())}
                         for c in num}).T
    desc.to_csv(out / "dataverse_column_summary.csv")
    ok = (df.icm_pc4 > 0) & (df.icm_pc6 > 0) & (df.icm_pc14 > 0)
    b414 = np.log(df.icm_pc4[ok] / df.icm_pc14[ok]) / np.log(14 / 4)
    b46 = np.log(df.icm_pc4[ok] / df.icm_pc6[ok]) / np.log(6 / 4)
    b614 = np.log(df.icm_pc6[ok] / df.icm_pc14[ok]) / np.log(14 / 6)
    q = lambda v: {"median": float(v.median()), "p25": float(v.quantile(0.25)), "p75": float(v.quantile(0.75))}
    blocks = df.assign(block=np.arange(len(df)) // 1000).groupby("block")[["icm_pc4", "icm_pc14"]].median()
    return {"n_rows": len(df), "n_columns": len(oo.DATAVERSE_COLUMNS),
            "isocode_excel_date_share": float(df.isocode_excel_date.mean()),
            "share_rows_pc14_zero": float((df.icm_pc14 == 0).mean()), "n_rows_all_counts_positive": int(ok.sum()),
            "beta_cumulative_4_14": q(b414), "beta_4_6": q(b46), "beta_6_14": q(b614),
            "share_beta_4_14_below_3": float((b414 < 3).mean()),
            "spearman_pc4_pc14_rows": float(sps.spearmanr(df.icm_pc4, df.icm_pc14, nan_policy="omit")[0]),
            "spearman_pc4_pc14_1000row_block_medians": float(sps.spearmanr(blocks.icm_pc4, blocks.icm_pc14)[0]),
            "note": "no machine id, timestamp, wear-metal spectroscopy, ferrography or wear measurement in the file"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--skip-dataverse", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "outputs/open_data/oil"
    out.mkdir(parents=True, exist_ok=True)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": "outputs/agent_logs/ODA.md (pre-specified)"}
    log("NASA")
    res["nasa"] = nasa(out)
    save_json(res, out / "oil_results.json")
    log("Zenodo")
    res["zenodo"] = zenodo(root, out, a.n_perm)
    save_json(res, out / "oil_results.json")
    if not a.skip_dataverse:
        log("Dataverse (slow xlsx parse on first run)")
        res["dataverse"] = dataverse(root, out)
    save_json(res, out / "oil_results.json")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
