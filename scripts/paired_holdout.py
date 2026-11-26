"""Hold-out test of the ONE calibrated factor of the FZG set on FZG-family papers that were not seen when it was frozen.
Plan, rules and go/no-go decision: docs/preregistration_paired2.md (frozen 2026-11-24, before any of the papers was available).

The constants are computed from the 20 thesis rows (14 gear pairs, Martins 2008) by `freeze` and never refitted. New sources are added
as data/real/paired/<id>/{paired.csv, study.json} (schema of src/wearpg/paired.py) with "holdout": "fzg" in study.json; the rows of the
sets named in "holdout_sets" are scored.

    .venv/Scripts/python.exe -I scripts/paired_holdout.py freeze    # once; refuses to overwrite
    .venv/Scripts/python.exe -I scripts/paired_holdout.py run       # -> outputs/open_data/paired/holdout/results.json, predictions.csv
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # -I drops the cwd

from wearpg import paired as pp  # noqa: E402
from wearpg.stats import holm  # noqa: E402

THESIS = ROOT / "data" / "real" / "paired" / "feup_martins2008" / "paired.csv"
OUT = ROOT / "outputs" / "open_data" / "paired" / "holdout"
B, MIN_UNITS, MAX_FOLD_OK, MAX_FOLD_BAD, ALPHA = 2_000_000, 8, 2.0, 2.5, 0.025  # 0.025 = two confirmatory tests (Bonferroni)
CONDS = ("cond_cycles_M", "cond_hertz_MPa", "cond_ester", "cond_adi")


def design(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(d)), np.log10(d.cond_cycles_M), np.log10(d.cond_hertz_MPa), d.cond_ester, d.cond_adi])


def archard(d: pd.DataFrame) -> np.ndarray:
    return np.log10(d.cond_cycles_M * d.cond_hertz_MPa**2).to_numpy()


def constants(d: pd.DataFrame) -> dict:
    """The five fitted quantities of the thesis rows: constant, factor of debris, factor of Archard, conditions, conditions + debris."""
    x, y, Z = np.log10(d.debris.to_numpy(float)), np.log10(d.wear.to_numpy(float)), design(d)
    return {"constant": float(y.mean()), "debris_b": float((y - x).mean()), "archard_b": float((y - archard(d)).mean()),
            "conditions_beta": np.linalg.lstsq(Z, y, rcond=None)[0].tolist(),
            "fused_beta": np.linalg.lstsq(Z, y - x, rcond=None)[0].tolist()}


def predict(d: pd.DataFrame, c: dict) -> pd.DataFrame:
    x = np.log10(d.debris.to_numpy(float))
    has = d[list(CONDS)].notna().all(axis=1).to_numpy() if set(CONDS) <= set(d) else np.zeros(len(d), bool)
    p = pd.DataFrame({"constant": c["constant"], "debris": c["debris_b"] + x}, index=d.index)
    for name in ("archard", "conditions", "fused"):
        p[name] = np.nan
    if has.any():
        s = d[has]
        Z = design(s)
        p.loc[has, "archard"] = c["archard_b"] + archard(s)
        p.loc[has, "conditions"] = Z @ np.array(c["conditions_beta"])
        p.loc[has, "fused"] = Z @ np.array(c["fused_beta"]) + x[has]
    return p


def boot_ci(d, b: int = B, seed: int = 0, chunk: int = 200_000) -> list[float]:
    d, rng, out = np.asarray(d, float), np.random.default_rng(seed), []
    for _ in range(0, b, chunk):
        out.append(d[rng.integers(0, len(d), (chunk, len(d)))].mean(1))
    return [float(v) for v in np.percentile(np.concatenate(out), [2.5, 97.5])]


def diff(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    return {"n": len(d), "diff": float(d.mean()), "ci": boot_ci(d), "p_signflip": pp.signflip_p(d)}


def verdict_accuracy(fold: float, hi_log10: float, n: int) -> str:
    """H1a: mean error of the frozen factor at most 2.0 (95 % upper bound at most 2.5)."""
    if n < MIN_UNITS:
        return "not testable (fewer than 8 new gear pairs)"
    if fold > MAX_FOLD_BAD:
        return "against"
    return "supported" if fold <= MAX_FOLD_OK and hi_log10 <= np.log10(MAX_FOLD_BAD) else "inconclusive"


def verdict_h1(d: dict, n: int) -> str:
    """H1b: frozen debris factor better than the frozen constant (docs/preregistration_paired2.md sec. 4)."""
    if n < MIN_UNITS:
        return "not testable (fewer than 8 new gear pairs)"
    return "against" if d["diff"] >= 0 else "supported" if d["ci"][1] < 0 and d["p_signflip"] < ALPHA else "inconclusive"


def decision(r: dict) -> str:
    """Go / no-go of the plan (sec. 6)."""
    if r["H1a_verdict"] == r["H1b_verdict"] == "supported" and r["H1c_debris_vs_refitted_constant"]["diff"] < 0:
        return "GO, strong claim (accurate and better than a constant on unseen gear pairs)"
    h6 = r["H6_pooled_refitted_loo"]
    if r["H1a_verdict"] == "supported" and r.get("H4_slope", {}).get("verdict", "").startswith("for_us") and h6["ci"][1] < 0 and h6["p_signflip"] < 0.05:
        return "GO, limited claim (accurate to a factor 2 and proportional on unseen pairs; better than a constant when pooled)"
    return "NO-GO for the conversion as the headline"


def evaluate(new: pd.DataFrame, thesis: pd.DataFrame, c: dict) -> tuple[dict, pd.DataFrame]:
    """Rows of `new` (columns unit, debris, wear, cond_*): error per gear pair of every frozen predictor, H1-H6."""
    new = new[(new.debris > 0) & (new.wear > 0)].reset_index(drop=True)
    pred = predict(new, c)
    err = (pred.sub(np.log10(new.wear), axis=0).abs()).groupby(new.unit, sort=False).mean()
    fold = {k: float(10 ** v) for k, v in err.mean().items() if np.isfinite(v)}
    n = len(err)
    r = {"n_units": n, "n_rows": len(new), "fold": fold, "within_2": int((err.debris <= np.log10(2)).sum())}
    r["H1b_debris_vs_constant"] = diff(err.debris, err.constant)
    own = pp.loo_predictions(np.log10(new.debris), np.log10(new.wear), np.empty((len(new), 0)), new.unit.to_numpy()).M0
    own_err = pp.unit_errors(own.to_frame("M0"), np.log10(new.wear.to_numpy()), new.unit.to_numpy()).M0
    r["H1c_debris_vs_refitted_constant"] = diff(err.debris, own_err)
    r["mean_log10_error_debris_ci"] = boot_ci(err.debris)
    r["H1a_verdict"] = verdict_accuracy(fold["debris"], r["mean_log10_error_debris_ci"][1], n)
    r["H1b_verdict"] = verdict_h1(r["H1b_debris_vs_constant"], n)
    for tag, other in (("H2_debris_vs_archard", "archard"), ("H3_debris_vs_conditions", "conditions")):
        if err[other].notna().all():
            r[tag] = diff(err.debris, err[other])
    um = np.log10(new.set_index("unit")[["debris", "wear"]]).groupby(level=0, sort=False).mean()
    if n >= 3 and np.ptp(um.debris) > 0:
        s, sp = pp.slope(um.debris, um.wear), pp.spearman(um.debris, um.wear)
        r["H4_slope"] = {"slope": s["slope"], "ci": [s["perm_lo"], s["perm_hi"]], "verdict": pp.verdict_slope(s["perm_lo"], s["perm_hi"])}
        r["H5_spearman"] = {"rho": sp["rho"], "p": sp["p"], "ci": [sp["boot_lo"], sp["boot_hi"]]}
    both = pd.concat([thesis.assign(unit="T:" + thesis.unit), new.assign(unit="N:" + new.unit)])
    e = pp.unit_errors(pp.loo_predictions(np.log10(both.debris), np.log10(both.wear), np.empty((len(both), 0)), both.unit.to_numpy()),
                       np.log10(both.wear.to_numpy()), both.unit.to_numpy())
    r["H6_pooled_refitted_loo"] = {"fold_debris": float(10 ** e.M2.mean()), "fold_constant": float(10 ** e.M0.mean()), **diff(e.M2, e.M0)}
    sec = [k for k in ("H2_debris_vs_archard", "H3_debris_vs_conditions") if k in r]
    for k, p in zip(sec, np.asarray(holm([r[k]["p_signflip"] for k in sec])).tolist()):
        r[k]["p_holm"] = p
    r["decision"] = decision(r)
    out = pd.concat([new[["unit", "debris", "wear"]].reset_index(drop=True), pred.add_prefix("pred_log10_").reset_index(drop=True)], axis=1)
    return r, out


def thesis_rows() -> pd.DataFrame:
    d = pd.read_csv(THESIS)
    return d[d.set == "fzg"].reset_index(drop=True)


def load_new() -> pd.DataFrame:
    """Rows of every added source that declares holdout 'fzg'; duplicates of thesis rows stop the run."""
    parts = []
    for st in pp.load_studies(ROOT / "data" / "real" / "paired"):
        if st.get("holdout") == "fzg":
            d = st["data"]
            parts.append(d[d.set.isin(st["holdout_sets"])].assign(study=st["id"], unit=st["id"] + ":" + d.unit))
    if not parts:
        raise SystemExit("no hold-out source yet (study.json with \"holdout\": \"fzg\")")
    new = pd.concat(parts).reset_index(drop=True)
    t = thesis_rows()
    for r in new.itertuples():
        dup = t[(np.abs(t.wear - r.wear) <= 0.011 * t.wear) & (np.abs(t.debris - r.debris) <= 0.011 * t.debris)]
        if len(dup):
            raise SystemExit(f"possible duplicate of a thesis row: {r.unit} {r.row} = {dup.row.tolist()} (remove the new row, log it)")
    return new


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "frozen.json"
    if sys.argv[1:] == ["freeze"]:
        if f.exists():
            raise SystemExit(f"{f} exists; the constants stay frozen")
        t = thesis_rows()
        f.write_text(json.dumps({"source": "feup_martins2008/fzg, 20 rows, 14 gear pairs, rows as fitting cases", "n_rows": len(t),
                                 "n_units": int(t.unit.nunique()), "constants": constants(t),
                                 "thesis_csv_sha256": hashlib.sha256(THESIS.read_bytes()).hexdigest()}, indent=1), encoding="utf-8")
        print(f.read_text())
        return
    frozen = json.loads(f.read_text())
    r, pred = evaluate(load_new(), thesis_rows(), frozen["constants"])
    pred.to_csv(OUT / "predictions.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
