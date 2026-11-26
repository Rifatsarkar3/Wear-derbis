"""Four sources supplied on 2026-11-25, after the first run (plan: outputs/agent_logs/PAIRED.md part 4 and its addendum). Descriptive or
exploratory; nothing here changes the frozen first-run verdict or any file of scripts/paired_analysis.py.

  A  holzhauer1983  ball bearings with an on-line Ferrograph: end ratios, within-bearing slopes, increments, cross-bearing prediction
  B  krasmik2018    three metal pairings with optical particle counts and wear volume against time
  C  quinn1980      oxide of the debris against wear rate (EN31; EN8 at 2 m/s is the digitisation check of CR-3686)
  D  nield1961      iron/haematite ratio of the debris against equilibrium wear rate and humidity

    .venv/Scripts/python.exe -I scripts/paired_later.py     # -> outputs/open_data/paired/later/results.json (run once)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]  # -I drops the cwd

import paired_analysis as pa  # noqa: E402
import paired_extra as px  # noqa: E402

from wearpg import paired as pp  # noqa: E402
from wearpg.stats import spearman_permutation  # noqa: E402
from wearpg.utils import save_json  # noqa: E402

DATA, OUT = ROOT / "data" / "real" / "paired", ROOT / "outputs" / "open_data" / "paired" / "later"
OIL_KG = 0.350 * 0.87  # density 0.87 kg/l assumed (plan)


def holzhauer(d: pd.DataFrame, conc: str = "conc_ppm") -> dict:
    """A1 end of test, A2 weighings within bearings (as the NASA cups: factor = slope of wear on debris, one per bearing)."""
    d = d.assign(D=d[conc].fillna(d.conc_ppm)).sort_values(["unit", "order"])
    end = d.groupby("unit").last()
    first = d[d.order == 1].set_index("unit")
    r = np.log10(end.D / end.wear_mg)
    r_run_in = np.log10((end.D - first.D) / (end.wear_mg - first.wear_mg))
    slopes = {u: float(np.polyfit(g.wear_mg, g.D, 1)[0]) for u, g in d.groupby("unit")}
    cup = d.rename(columns={"wear_mg": "wear"}).assign(debris=d.D, cond_time_h=d.t_min / 60, cond_load_pct=np.nan)
    a2, table = pa.cups(cup)
    rec = end.D * OIL_KG / end.wear_mg
    lo = pp.conversion(np.log10(list(slopes.values())))
    end_spread = px.spread("holzhauer1983/ball_bearings", np.log10(end.D.to_numpy()), np.log10(end.wear_mg.to_numpy()))
    return {"n_bearings": len(end), "role": "descriptive", "end_D_over_W": (end.D / end.wear_mg).round(4).to_dict(), "end_conversion": pp.conversion(r),
            "end_spread": end_spread, "without_run_in_weighing": pp.conversion(r_run_in), "slope_conc_per_mg": slopes, "slopes_conversion": lo,
            "recovery_per_bearing": rec.round(4).to_dict(), "recovery_gm": float(10 ** np.log10(rec).mean()), "recovery_gsd": float(10 ** np.log10(rec).std(ddof=1)),
            "lot_ratio_AD_over_BC": float(10 ** (r[["A", "D"]].mean() - r[["B", "C"]].mean())), "cross_bearing": {k: v for k, v in a2.items() if k != "kappa_end_max_over_min"},
            "cross_bearing_table": table.round(4).to_dict("records")}


def krasmik(h: pd.DataFrame, t3: pd.DataFrame, sz: pd.DataFrame) -> dict:
    h = h[h.hour >= 1].sort_values(["unit", "hour"])
    end = h.groupby("unit").last()
    t3 = t3.set_index("unit")
    out = {"n_pairings": len(end), "role": "descriptive"}
    for name, w in (("displacement_volume_fig7", end.wear_volume_mm3), ("displacement_volume_table3", t3.wear_volume_linear_mm3), ("wear_mark", t3.wear_mark_volume_mm3),
                    ("weight_loss", t3.weight_loss_volume_mm3)):
        out[f"B1_N_over_{name}"] = {**pp.conversion(np.log10(end.n_per_image / w)), "per_pairing": (end.n_per_image / w).round(2).to_dict()}
    out["B4_wear_mark_over_weight_loss"] = (t3.wear_mark_volume_mm3 / t3.weight_loss_volume_mm3).round(4).to_dict()
    out["B4_displacement_over_weight_loss"] = (t3.wear_volume_linear_mm3 / t3.weight_loss_volume_mm3).round(4).to_dict()
    inc = pd.concat([pd.DataFrame({"unit": u, "dN": np.diff(g.n_per_image), "dV": np.diff(g.wear_volume_mm3)}) for u, g in h.groupby("unit")])
    s = spearman_permutation(inc.dN, inc.dV, n_perm=pp.N_PERM, seed=0, strata=inc.unit)
    g = h.groupby("unit")
    k = (g.wear_volume_mm3.last() - g.wear_volume_mm3.first()) / (g.n_per_image.last() - g.n_per_image.first())      # mm3 per particle per image
    rate = (g.wear_volume_mm3.last() - g.wear_volume_mm3.first()) / 14                                                   # mm3 per h
    mae = {}
    for u, q in inc.groupby("unit"):
        mae[u] = {"debris_mm3": float((q.dV - k.drop(u).mean() * q.dN).abs().mean()), "constant_rate_mm3": float((q.dV - rate.drop(u).mean()).abs().mean())}
    out["B2_increments"] = {"n": len(inc), "stratified_spearman": s["rho"], "p": s["p_value"], "cross_pairing_mae": mae, "factor_mm3_per_count": k.round(5).to_dict()}
    mid = {"1-5": 5**.5, "6-10": 60**.5, "11-15": 165**.5, "16-20": 320**.5, "21-25": 525**.5, "26-30": 780**.5}
    b3 = {}
    for u, q in sz[sz.size_class_um != "all"].groupby("unit"):
        lim = 2.5 if u.startswith("CuZn37") else 0.25                                                                    # reading limit, particles per image (addendum)
        n, d3 = q.n_per_image_end.to_numpy(), np.array([mid[c] ** 3 for c in q.size_class_um])
        s0, hi, lo = (n * d3).sum(), ((n + lim) * d3).sum(), (np.clip(n - lim, 0, None) * d3).sum()
        b3[u] = {"sum_n_d3": float(s0), "hi_over_lo": float(hi / lo), "share_above_15um": float((n * d3)[3:].sum() / s0)}
    out["B3_size_weighted"] = {**b3, "interpretable": bool(all(v["hi_over_lo"] <= 2 for k, v in b3.items()))}
    return out


def quinn(d: pd.DataFrame) -> dict:
    r = pa.regime(d[d.material == "EN31"])
    q = pd.read_csv(DATA / "nasa_cr3686_quinn1983" / "paired.csv")
    q = q[q["unit"].str.startswith("2m/s")]
    e = d[d.material == "EN8"]
    near = [float(min(abs(q.wear / w - 1))) for _, w in zip(e.cond_load_N, e.wear)]
    return {"role": "exploratory", "EN31": r, "EN8_2ms_check": {"printed_rows": len(e), "digitised_rows": len(q), "median_rel_diff_of_nearest_wear": float(np.median(near)), "max": float(max(near))}}


def nield(d: pd.DataFrame) -> dict:
    """LOO error of log10 wear rate: constant, humidity, ratio, both (linear in the predictor)."""
    y, x, h, one = np.log10(d.wear_rate_mg_h.to_numpy()), np.log10(d.ratio_X.to_numpy()), d.cond_rh_pct.to_numpy(float), np.ones((len(d), 1))
    X = {"constant": one, "humidity": np.column_stack([one, h]), "ratio": np.column_stack([one, x]), "both": np.column_stack([one, h, x])}
    err = {}
    for k, A in X.items():
        lev = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
        err[k] = np.abs((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) / (1 - lev))
    out = {"role": "exploratory", "n": len(d), **{f"mae_log10_{k}": float(v.mean()) for k, v in err.items()}}
    for a, b in (("ratio", "constant"), ("ratio", "humidity"), ("both", "humidity"), ("both", "ratio")):
        out[f"{a}_minus_{b}"] = pp.mean_diff(err[a] - err[b])
    fit = np.polyfit(d.wear_rate_mg_h, d.ratio_X, 1)
    out["check_printed_regression"] = {"slope": float(fit[0]), "intercept": float(fit[1]), "r": float(np.corrcoef(d.ratio_X, d.wear_rate_mg_h)[0, 1]), "printed": [0.242, -0.157, 0.973]}
    out["spearman_rate_ratio"] = pp.spearman(d.ratio_X, d.wear_rate_mg_h)
    out["verdict"] = "supported" if out["ratio_minus_humidity"]["diff"] < 0 and out["ratio_minus_humidity"]["hi"] < 0 else "not supported"
    return out


def main() -> None:
    res = OUT / "results.json"
    if res.exists():
        raise SystemExit(f"{res} exists; the plan allows one run")
    OUT.mkdir(parents=True, exist_ok=True)
    rd = lambda p: pd.read_csv(DATA / p)
    ho = rd("holzhauer1983/later.csv")
    out = {"label": "added after the first run; descriptive / exploratory (PAIRED.md part 4)",
           "A_holzhauer1983": {"primary": holzhauer(ho), "low_scale": holzhauer(ho, "conc_low_scale_ppm")},
           "B_krasmik2018": krasmik(rd("krasmik2018/later.csv"), rd("krasmik2018/table3.csv"), rd("krasmik2018/sizes.csv")),
           "C_quinn1980": quinn(rd("quinn1980/later.csv")), "D_nield1961": nield(rd("nield1961/later.csv"))}
    # V6 with the later study: the recovery factor of Holzhauer's bearings next to the three first-run ones
    first = {"bu_torres2011/pin_totals": 1.396984372050445, "jsme_higashi2003/plate_intervals": 0.6657720010799566, "phm_raadnui2015/dry_fourball": 1.3626361432755627}
    g = np.log10([*first.values(), out["A_holzhauer1983"]["primary"]["recovery_gm"]])
    out["V6_with_later_study"] = {"recovery_gm_per_set": {**first, "holzhauer1983/ball_bearings": out["A_holzhauer1983"]["primary"]["recovery_gm"]},
                                  "between_set_gsd": float(10 ** g.std(ddof=1)), "max_over_min": float(10 ** np.ptp(g)), "verdict": pp.verdict_gsd(float(10 ** g.std(ddof=1)), len(g)),
                                  "first_run": {"between_set_gsd": 1.5233101470225237, "verdict": "inconclusive"}}
    save_json(out, res)
    print("written", res)


if __name__ == "__main__":
    main()
