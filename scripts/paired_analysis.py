"""Test of the aggregate debris -> wear conversion on published paired data (PAIRED; post hoc / exploratory). Plan and verdict
rules were written to outputs/agent_logs/PAIRED.md (parts 1 and 2) before any statistic was computed.

Reads every data/real/paired/<id>/{paired.csv, study.json} (made by scripts/paired_digitise.py, or added later by hand with the
same schema; docs/paired_data_results.md sec. 6). Writes outputs/open_data/paired/{results.json, summary.csv, units.csv,
loo_predictions.csv, paired_all.csv, nasa_cups.csv, volume_prior.csv}. Deterministic and idempotent.
Usage: .venv/Scripts/python.exe -I scripts/paired_analysis.py   (about 1 min)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # -I drops the cwd

from wearpg import paired as pp  # noqa: E402
from wearpg.open_oil import permutation_p  # noqa: E402
from wearpg.stats import holm, spearman_permutation  # noqa: E402
from wearpg.utils import save_json  # noqa: E402

DATA, OUT = ROOT / "data" / "real" / "paired", ROOT / "outputs" / "open_data" / "paired"
MODELS = ("M0", "M1", "M2", "M3", "M2f", "M3f")
FAMILIES = {"slope_eq_1": "p_slope_eq_1", "spearman": "p_spearman", "V2_debris_vs_intercept": "V2_p", "V3_debris_vs_conditions": "V3_p",
            "V4_fusion_vs_best_single": "V4_p", "V5_resolved_vs_total": "V5_p"}


# ------------------------------------------------------------------------------------------------------ one analysis
def analyse(d: pd.DataFrame, spec: dict, resolved: bool = True) -> tuple[dict, pd.DataFrame]:
    """All statistics of one set (plan items 1-5) for rows with columns D, W, u. -> (flat dict, row-level predictions)."""
    u = pp.unit_means(d)
    n = len(u)
    out = {"n_rows": len(d), "n_units": n, "decades_D": float(np.ptp(u.D)), "decades_W": float(np.ptp(u.W))}
    if n < 3 or np.ptp(u.D) == 0 or np.ptp(u.W) == 0:
        return out, pd.DataFrame()
    s, r = pp.slope(u.D, u.W), pp.spearman(u.D, u.W)
    out |= {"slope": s["slope"], "slope_lo": s["perm_lo"], "slope_hi": s["perm_hi"], "perm_exact": s["perm_exact"],
            "p_slope_eq_1": s["p_slope_eq_ref"], "p_slope_eq_0": s["p_slope_eq_0"], "slope_t_lo": s["t_lo"], "slope_t_hi": s["t_hi"],
            "slope_boot_lo": s["boot_lo"], "slope_boot_hi": s["boot_hi"], "slope_inverse": s["inverse_slope"], "slope_rma": s["rma_slope"],
            "pearson_r": s["r"], "resid_sd_log10": s["resid_sd"], "spearman": r["rho"], "p_spearman": r["p"], "spearman_lo": r["boot_lo"],
            "spearman_hi": r["boot_hi"]}
    out |= {"V1": pp.verdict_slope(s["perm_lo"], s["perm_hi"]), "V1b": pp.verdict_spearman(r["rho"], r["boot_lo"], r["p"])}
    c = pp.conversion(u.D - u.W)
    out |= {f"DoverW_{k}": v for k, v in c.items() if k != "n"}
    x, y, unit = np.log10(d.D.to_numpy()), np.log10(d.W.to_numpy()), d.u.to_numpy()
    Z, names = pp.conditions_matrix(d, spec.get("conditions", []), n)
    pred = pp.loo_predictions(x, y, Z, unit)
    if not names:
        pred[["M1", "M3", "M3f"]] = np.nan
    err = pp.unit_errors(pred, y, unit)
    mae = err.mean()
    out |= {"conditions": "+".join(names) or "none", **{f"mae_{m}": float(mae[m]) for m in MODELS},
            "fold_M0": float(10 ** mae.M0), "fold_M2": float(10 ** mae.M2)}

    def compare(tag, a, b):
        out.update({f"{tag}_diff": math.nan, f"{tag}_lo": math.nan, f"{tag}_hi": math.nan, f"{tag}_p": math.nan})
        if not (a.isna().any() or b.isna().any()):
            m = pp.mean_diff(a - b)
            out.update({f"{tag}_diff": m["diff"], f"{tag}_lo": m["lo"], f"{tag}_hi": m["hi"], f"{tag}_p": m["p"]})
        out[tag] = pp.verdict_diff(out[f"{tag}_diff"], out[f"{tag}_hi"])

    best = "M1" if mae.M1 <= mae.M2 else "M2"
    compare("V2", err.M2, err.M0)
    compare("V3", err.M2, err.M1)
    compare("V4", err.M3, err[best])
    out["V4_best_single"] = best if names else "none"
    compare("V2_free_slope", err.M2f, err.M0)
    res = d.resolved.astype(float)
    if resolved and spec.get("resolved_def") and (res > 0).all():
        rp = pp.loo_predictions(np.log10(res.to_numpy()), y, Z, unit)
        re = pp.unit_errors(rp, y, unit)
        out |= {"mae_resolved": float(re.M2.mean()), "fold_resolved": float(10 ** re.M2.mean())}
        compare("V5", re.M2, err.M2)
        ru = pp.unit_means(d.assign(R=res), ("R", "W"))
        rs, rr = pp.slope(ru.R, ru.W), pp.spearman(ru.R, ru.W)
        out |= {"resolved_slope": rs["slope"], "resolved_slope_lo": rs["perm_lo"], "resolved_slope_hi": rs["perm_hi"],
                "resolved_spearman": rr["rho"]}
        pred["R2"] = rp.M2
    else:
        out |= {"mae_resolved": math.nan, "V5": "not testable", "V5_diff": math.nan, "V5_p": math.nan}
    return out, pd.concat([d[["study", "set", "unit", "row", "D", "W"]].reset_index(drop=True), pred], axis=1)


def role_of(spec: dict, d: pd.DataFrame) -> str:
    if spec.get("analysis") or spec.get("units_unknown") or (d["mode"] == "series").any():
        return "descriptive"
    return "inferential" if d.u.nunique() >= pp.MIN_UNITS else "descriptive"


# ------------------------------------------------------------------------------------------------- set-specific parts
def fractions_exploratory(d: pd.DataFrame, cols: list[str]) -> dict:
    """Raadnui: each size fraction alone as predictor (one constant, LOO) and leave-one-out non-negative weights (exploratory)."""
    y, unit = np.log10(d.W.to_numpy()), d.u.to_numpy()
    out = {}
    for c in [*cols, "D"]:
        e = pp.unit_errors(pp.loo_predictions(np.log10(d[c].to_numpy(float)), y, np.empty((len(d), 0)), unit), y, unit)
        out[f"mae_{c}"] = float(e.M2.mean())
    X, W, pred = d[cols].to_numpy(float), d.W.to_numpy(), np.full(len(d), np.nan)
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        w = 1 / W[tr]  # relative least squares
        b, _ = nnls(X[tr] * w[:, None], W[tr] * w)
        pred[i] = X[i] @ b
    with np.errstate(divide="ignore"):
        out["mae_nnls_weights"] = float(np.abs(np.log10(pred) - y).mean())
    out["nnls_weights_all_rows"] = nnls(X / d.W.to_numpy()[:, None], np.ones(len(d)))[0].tolist()
    return out


def cups(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """NASA bearing spalls (plan D2): per-cup conversion factors, cross-cup prediction against time, increments."""
    rows, parts = [], {}
    for cup, g in d.groupby("unit", sort=True):
        g = g.sort_values("order")
        M, L, t = g.debris.to_numpy(), g.wear.to_numpy(), g.cond_time_h.to_numpy()
        kap, rate = np.polyfit(M, L, 1)[0], np.polyfit(t, L, 1)[0]
        k = (M > M[0]) & (L > L[0] + 0.05)
        ll = np.polyfit(np.log10(M[k] - M[0]), np.log10(L[k] - L[0]), 1)[0]
        two = [(L[j] - L[0]) / (M[j] - M[0]) for j in (len(L) - 1,)]
        rows.append({"cup": cup, "n": len(g), "load_pct": float(g.cond_load_pct.iloc[0]), "M0": M[0], "L0": L[0], "M_end": M[-1], "L_end": L[-1],
                     "kappa_mm_per_mg": kap, "kappa_end_mm_per_mg": two[0], "loglog_slope": ll, "rate_mm_per_h": rate})
        parts[cup] = (M, L, t)
    T = pd.DataFrame(rows)
    pred = []
    for i, r in T.iterrows():
        oth = T.drop(index=i)
        M, L, t = parts[r.cup]
        pred.append({"cup": r.cup, "mae_debris_mm": float(np.abs(L[0] + oth.kappa_mm_per_mg.mean() * (M - M[0]) - L)[1:].mean()),
                     "mae_time_mm": float(np.abs(L[0] + oth.rate_mm_per_h.mean() * (t - t[0]) - L)[1:].mean())})
    T = T.merge(pd.DataFrame(pred), on="cup")
    inc = []
    for cup, (M, L, t) in parts.items():
        inc.append(pd.DataFrame({"cup": cup, "dM": np.diff(M), "dL": np.diff(L), "dt": np.diff(t)}))
    inc = pd.concat(inc)
    strat = spearman_permutation(inc.dM, inc.dL, n_perm=pp.N_PERM, seed=0, strata=inc.cup)
    strat_t = spearman_permutation(inc.dt, inc.dL, n_perm=pp.N_PERM, seed=0, strata=inc.cup)
    lg = np.log10(T.kappa_mm_per_mg)
    diff = (T.mae_debris_mm - T.mae_time_mm).to_numpy()
    return {"n_cups": len(T), "n_inspections": int(T.n.sum()), "kappa_gm": float(10 ** lg.mean()), "kappa_gsd": float(10 ** lg.std(ddof=1)),
            "kappa_max_over_min": float(10 ** np.ptp(lg)), "kappa_end_max_over_min": float(T.kappa_end_mm_per_mg.max() / T.kappa_end_mm_per_mg.min()),
            "loglog_slope_mean": float(T.loglog_slope.mean()), "loglog_slopes": T.loglog_slope.round(4).tolist(),
            "crosscup_mae_debris_mm": float(T.mae_debris_mm.mean()), "crosscup_mae_time_mm": float(T.mae_time_mm.mean()),
            "crosscup_diff_mm": float(diff.mean()), "crosscup_signflip_p": pp.signflip_p(diff), "cups_debris_better": int((diff < 0).sum()),
            "increments_n": len(inc), "increments_rho_dM_dL": strat["rho"], "increments_p_dM_dL": strat["p_value"],
            "increments_rho_dt_dL": strat_t["rho"], "increments_p_dt_dL": strat_t["p_value"]}, T


def regime(d: pd.DataFrame) -> dict:
    """Quinn: does the dominant oxide of the debris explain wear rate / load beyond load and speed? (mechanism -> rate, not conversion)."""
    y = np.log10(d.wear / d.cond_load_N).to_numpy()
    lo, sp = np.log10(d.cond_load_N).to_numpy(), np.log10(d.cond_speed_ms).to_numpy()
    reg = np.column_stack([(d.class_regime == k).astype(float) for k in ("Fe3O4", "FeO")])
    one = np.ones((len(d), 1))
    X = {"constant": one, "conditions": np.column_stack([one, lo, sp]), "regime": np.hstack([one, reg]),
         "conditions_plus_regime": np.column_stack([one, lo, sp, reg])}
    err = {}
    for k, A in X.items():
        h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
        err[k] = np.abs((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) / (1 - h))
    out = {"n": len(d), "regime_counts": d.class_regime.value_counts().to_dict(), **{f"mae_{k}": float(v.mean()) for k, v in err.items()}}
    for a, b in (("regime", "conditions"), ("conditions_plus_regime", "conditions"), ("regime", "constant")):
        out[f"{a}_minus_{b}"] = pp.mean_diff(err[a] - err[b])
    g = {k: np.log10(v.wear / v.cond_load_N) for k, v in d.groupby("class_regime")}
    out["gm_wear_rate_per_load"] = {k: float(10 ** v.mean()) for k, v in g.items()}
    return out


def volume_prior(f: Path) -> tuple[dict, pd.DataFrame]:
    """Plan item 10: the paper's volume prior v = 0.3 A^1.5 against measured particle thickness (Higashi 2003, Fig. 5)."""
    t = pd.read_csv(f)
    s = pp.slope(np.log10(t.size_um), np.log10(t.thickness_um))
    area = math.pi * t.size_um**2 / 4
    t["prior_over_measured"] = 0.3 * area**1.5 / (area * t.thickness_um)
    r = t.prior_over_measured
    ok = s["perm_lo"] <= 1 <= s["perm_hi"] and r.between(0.5, 2).all()
    return {"n": len(t), "slope_log_thickness_on_log_size": s["slope"], "slope_lo": s["perm_lo"], "slope_hi": s["perm_hi"],
            "slope_t_lo": s["t_lo"], "slope_t_hi": s["t_hi"], "p_slope_eq_1": s["p_slope_eq_ref"],
            "prior_over_measured_min": float(r.min()), "prior_over_measured_median": float(r.median()), "prior_over_measured_max": float(r.max()),
            "thickness_over_size_min": float((t.thickness_um / t.size_um).min()), "thickness_over_size_max": float((t.thickness_um / t.size_um).max()),
            "drift_10_to_80um": float(r.iloc[-1] / r.iloc[1]), "verdict": "for_us" if ok else "against_us"}, t


def craig_iron(d: pd.DataFrame) -> dict:
    """Sensitivity of V6: spectrometric iron in the oil stream x flow (4 l/min, 0.87 kg/l assumed) against the weighed loss."""
    g = d[d["class_fe_ppm_h"] > 0]
    fe = g["class_fe_ppm_h"] * 4 * 60 * 0.87  # ppm h x kg/h = mg
    return {u: {"iron_mg": float(a), "wear_mg": float(w), "rho": float(a / w)} for u, a, w in zip(g["unit"], fe, g["wear"])}


# ---------------------------------------------------------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    studies = pp.load_studies(DATA)
    summary, preds, units, results = [], [], [], {"label": "post_hoc / exploratory", "plan": "outputs/agent_logs/PAIRED.md parts 1-2",
                                                 "sets": {}, "studies": {s["id"]: {k: s[k] for k in ("citation", "doi", "url", "rights")} for s in studies}}
    extra: dict[str, object] = {}
    for st in studies:
        for name, spec in st["sets"].items():
            full = st["data"][st["data"]["set"] == name].reset_index(drop=True)
            if full["debris"].notna().any():
                d = pp.select(full, exclude=spec.get("exclude_primary", ()))
            else:
                d = full.assign(D=np.nan, W=full.wear, u=full.unit)
            role = role_of(spec, d)
            key = f"{st['id']}/{name}"
            entry = {"study": st["id"], "set": name, "role": role, "added_after_first_run": bool(st.get("added_after_first_run")),
                     "debris_type": full.debris_type.iloc[0], "wear_unit": full.wear_unit.iloc[0], "extraction": full.extraction.iloc[0]}
            if spec.get("analysis") == "series_cups":
                res, table = cups(full)
                table.to_csv(OUT / "nasa_cups.csv", index=False, lineterminator="\n", float_format="%.6g")
                extra["nasa_bearing"] = res
            elif spec.get("analysis") == "regime":
                extra["quinn_regime"] = regime(full)
            elif spec.get("analysis") == "table_only":
                pass
            else:
                if role == "descriptive" and d.u.nunique() < pp.MIN_UNITS <= len(d):
                    d = d.assign(u=d["row"])  # one run followed in intervals: rows as pseudo-units, never inferential
                    entry["rows_as_units"] = True
                # cumulative curves of one run rise by construction: no statistic, only the recovery ratio below
                prim, pr = ({"n_rows": len(d), "n_units": d["unit"].nunique()}, pd.DataFrame()) if (d["mode"] == "series").any() else analyse(d, spec)
                results["sets"][key] = {**entry, "primary": prim, "conditions_spec": spec.get("conditions", []), "resolved_def": spec.get("resolved_def")}
                summary.append({**entry, "analysis": "primary", **prim})
                preds.append(pr)
                if (d["mode"] != "series").all():
                    units.append(pp.unit_means(d).reset_index().assign(study=st["id"], set=name))
                sens = {}
                for sname, o in spec.get("sensitivity", {}).items():
                    ds = pp.select(full, **{k: v for k, v in o.items() if k in ("debris", "wear", "unit", "mode", "fallback")}, drop=o.get("drop", ()),
                                   include=o.get("include", ()), exclude=spec.get("exclude_primary", ()))
                    if ds.u.nunique() >= 3:
                        sens[sname], _ = analyse(ds, spec, resolved="debris" not in o and "wear" not in o)
                        summary.append({**entry, "analysis": sname, **sens[sname]})
                results["sets"][key]["sensitivity"] = sens
                if spec.get("group"):
                    lr = np.log10(d.W / d.D).groupby(d.u, sort=False).mean()
                    grp = d.groupby("u", sort=False)[spec["group"]].first().reindex(lr.index)
                    a, b = lr[grp == 1], lr[grp == 0]
                    p, kind = permutation_p(a, b)
                    results["sets"][key]["group_contrast"] = {"group": spec["group"], "n_1": len(a), "n_0": len(b), "p": p, "p_kind": kind,
                                                              "ratio_of_W_over_D_1_to_0": float(10 ** (a.mean() - b.mean()))}
                if spec.get("absolute"):
                    f = spec["absolute"]["factor"]
                    rho = np.log10(d.D / d.W * f).groupby(d.u, sort=False).mean() if role == "inferential" else np.log10(d.D / d.W * f)
                    results["sets"][key]["recovery"] = {**pp.conversion(rho), "note": spec["absolute"]["note"], "per_row": (d.D / d.W * f).round(4).tolist()}
                if spec.get("fractions"):
                    results["sets"][key]["fractions_exploratory"] = fractions_exploratory(d, spec["fractions"])
    # ---- families, verdict counts
    inf = [k for k, v in results["sets"].items() if v["role"] == "inferential" and not v["added_after_first_run"]]
    later = [k for k, v in results["sets"].items() if v["role"] == "inferential" and v["added_after_first_run"]]
    results["inferential_sets"], results["inferential_sets_added_later"] = inf, later
    results["families"] = {}
    for tag, sets in (("first_run", inf), ("with_later_studies", inf + later)):
        if tag == "with_later_studies" and not later:
            continue
        fam = {}
        for f, col in FAMILIES.items():
            p = [results["sets"][k]["primary"].get(col, math.nan) for k in sets]
            fam[f] = {k: {"p": a, "p_holm": b} for k, a, b in zip(sets, p, holm(p).tolist())}
        count = {v: pd.Series([results["sets"][k]["primary"].get(v, "not testable").split()[0] for k in sets]).value_counts().to_dict()
                 for v in ("V1", "V1b", "V2", "V3", "V4", "V5")}
        n = len(sets)
        overall = "SUPPORTED" if count["V1"].get("for_us", 0) > n / 2 and count["V2"].get("for_us", 0) > n / 2 else \
            "NOT SUPPORTED" if max(count["V1"].get("against_us", 0), count["V2"].get("against_us", 0)) > n / 2 else "MIXED"
        results["families"][tag] = {"holm": fam, "verdict_counts": count, "n_inferential": n, "overall": overall}
    # ---- across studies: recovery (absolute units)
    gm = {k: v["recovery"]["gm"] for k, v in results["sets"].items() if "recovery" in v and k.split("/")[1] != "pin_weight_series"}
    lg = np.log10(list(gm.values()))
    gsd = float(10 ** lg.std(ddof=1)) if len(lg) > 1 else math.nan
    results["across_studies"] = {"recovery_gm_per_set": gm, "between_set_gsd": gsd, "max_over_min": float(10 ** np.ptp(lg)) if len(lg) else math.nan,
                                 "V6": pp.verdict_gsd(gsd, len(lg)), "within_set_gsd_D_over_W": {
                                     k: v["primary"].get("DoverW_gsd") for k, v in results["sets"].items() if v["role"] == "inferential"},
                                 "fold_error_one_constant": {k: v["primary"].get("fold_M2") for k, v in results["sets"].items()
                                                             if v["role"] == "inferential"},
                                 "craig_iron_stream_sensitivity": craig_iron(next(s for s in studies if s["id"] == "soton_craig2010")["data"])
                                 if any(s["id"] == "soton_craig2010" for s in studies) else {}}
    if (f := DATA / "jsme_higashi2003" / "thickness.csv").exists():
        results["volume_prior_check"], vt = volume_prior(f)
        vt.to_csv(OUT / "volume_prior.csv", index=False, lineterminator="\n", float_format="%.5g")
    results["descriptive"] = extra
    S = pd.DataFrame(summary)
    for f, col in FAMILIES.items():
        ph = {k: v["p_holm"] for k, v in results["families"]["first_run"]["holm"][f].items()}
        S[f"holm_{f}"] = [ph.get(f"{a}/{b}", math.nan) if c == "primary" else math.nan for a, b, c in zip(S.study, S["set"], S.analysis)]
    S.to_csv(OUT / "summary.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.concat(preds).to_csv(OUT / "loo_predictions.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.concat(units).rename(columns={"u": "unit", "D": "log10_D", "W": "log10_W"}).to_csv(OUT / "units.csv", index=False, lineterminator="\n",
                                                                                         float_format="%.6g")
    cols = ["study", "set", "unit", "row", "mode", "rig", "material_pair", "lubricant", "debris", "debris_metric", "debris_unit", "wear", "wear_metric",
            "wear_unit", "resolved", "debris_2", "wear_2", "debris_err", "wear_err", "page", "item", "extraction", "note"]
    pd.concat([s["data"][cols] for s in studies]).to_csv(OUT / "paired_all.csv", index=False, lineterminator="\n", float_format="%.6g")
    save_json(results, OUT / "results.json")
    show = ["study", "set", "analysis", "role", "n_rows", "n_units", "slope", "slope_lo", "slope_hi", "p_slope_eq_1", "spearman", "p_spearman", "mae_M0",
            "mae_M1", "mae_M2", "mae_M3", "fold_M2", "DoverW_gsd", "V1", "V1b", "V2", "V3", "V4", "V5"]
    print(S[S.analysis == "primary"][show].round(3).to_string())
    print(results["families"]["first_run"]["verdict_counts"], results["families"]["first_run"]["overall"])


if __name__ == "__main__":
    main()
