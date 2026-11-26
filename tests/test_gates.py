"""Gates on tiny_dataset with constructed descriptor / prediction tables: pass, fail and not-evaluable (None) cases of
every gate, and the synthetic banner + decision_valid false in written reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wearpg.config import load_config
from wearpg.gates import (SYNTHETIC_BANNER, VERIFICATION_NOTICE, GateResult, evaluate_gates, graph_value_gate,
                          mechanism_gate, pilot_gate, portfolio_check, wear_gate, wear_rate_gate, write_gate_report)
from wearpg.schema import MECHANISMS, coerce_table, prob_columns

ROOT = Path(__file__).resolve().parents[1]


def _cfg(*overrides):
    return load_config(ROOT / "configs" / "default.yaml", ["eval.n_bootstrap=200", *overrides])


def _log10k(ds) -> pd.Series:
    return np.log10(ds.runs.set_index("run_id").wear_rate_mm3_Nm)


def _descriptors(ds, log_ecd_of_run) -> pd.DataFrame:
    """Module-B-like descriptors: shape_log_ecd_um per particle from its run; other aggregates constant (untestable)."""
    run = ds.particles.sample_id.map(ds.samples.set_index("sample_id").run_id)
    log_ecd = run.map(log_ecd_of_run).to_numpy(float)
    return pd.DataFrame({"shape_log_ecd_um": log_ecd, "shape_ecd_um": np.exp(log_ecd), "fractal_divider_D": 1.1,
                         "fractal_valid": 1.0, "shape_aspect_ratio": 1.5},
                        index=pd.Index(ds.particles.particle_id.to_numpy(), name="particle_id"))


def _particle_preds(ds, quality: dict, seeds=(0, 1), protocol="cv", split="test") -> pd.DataFrame:
    """One row per particle x (model, variant) x seed; probability 0.9 on the true class with probability `quality`,
    else on a wrong class."""
    rng = np.random.default_rng(0)
    h = ds.particle_hierarchy()
    y = ds.label_indices().reindex(h.particle_id).to_numpy()
    parts = []
    for (model, variant), q in quality.items():
        for seed in seeds:
            pred = np.where(rng.random(len(h)) < q, y, (y + rng.integers(1, len(MECHANISMS), len(h))) % len(MECHANISMS))
            proba = np.full((len(h), len(MECHANISMS)), 0.1 / (len(MECHANISMS) - 1))
            proba[np.arange(len(h)), pred] = 0.9
            df = pd.DataFrame(proba, columns=prob_columns())
            df = df.assign(protocol=protocol, model=model, variant=variant, fold="f0", seed=seed, split=split,
                           particle_id=h.particle_id, specimen_id=h.specimen_id, run_id=h.run_id,
                           condition_id=h.condition_id, y_true=y, y_pred=pred)
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _run_preds(ds, err: dict, protocol="external", seeds=(0,)) -> pd.DataFrame:
    """Run rows; log10_rate_pred = true + err[(model, variant)][material] (NaN -> note 'eta not transferable').
    External folds are leave-one-material-out groups."""
    rt = ds.run_table()
    y = np.log10(rt.wear_rate_mm3_Nm.to_numpy())
    parts = []
    for (model, variant), e in err.items():
        off = rt.material_id.map(e).to_numpy(float)
        for seed in seeds:
            parts.append(pd.DataFrame({
                "protocol": protocol, "model": model, "variant": variant, "seed": seed, "split": "test",
                "fold": ("logo_material_id_" + rt.material_id) if protocol == "external" else "f0",
                "run_id": rt.run_id, "specimen_id": rt.specimen_id, "material_id": rt.material_id,
                "condition_id": rt.condition_id, "log10_rate_true": y, "log10_rate_pred": y + off,
                "note": np.where(np.isnan(off), "eta not transferable", "")}))
    return pd.concat(parts, ignore_index=True)


MATS = ("AISI52100", "CuSn12")


def _both(v) -> dict:
    return {m: v for m in MATS}


# ----------------------------------------------------------------------------- pilot


def test_pilot_gate_pass_when_size_tracks_wear_rate(tiny_dataset):
    desc = _descriptors(tiny_dataset, 4.0 + 0.3 * _log10k(tiny_dataset))  # ECD 10-17 um, above the floor
    r = pilot_gate(tiny_dataset, desc, _cfg())
    assert r.name == "pilot" and r.passed is True
    f = r.evidence["features"]["median_log_ecd"]
    assert f["rho"] == pytest.approx(1.0) and f["p_adj"] < 0.05 and f["passes"]
    assert f["pooled_rho"] == pytest.approx(1.0)
    assert math.isnan(r.evidence["features"]["mean_aspect"]["p_value"])  # constant aggregate: not testable
    assert r.evidence["ecd_floor_um"] == pytest.approx(2 * np.sqrt(30 / np.pi) * 0.5)  # coarsest px = 0.5 um
    assert r.evidence["n_particles_used"] == len(tiny_dataset.particles)


def test_pilot_gate_gates_on_the_within_material_trend_only(tiny_dataset):
    ds = tiny_dataset
    cu = ds.runs.specimen_id.str.contains("CuSn12").to_numpy()
    ds.runs.loc[cu, "wear_rate_mm3_Nm"] *= 1e3  # material confounding: CuSn12 wears 1000x faster
    lk = _log10k(ds)
    is_cu = pd.Series(cu, index=ds.runs.run_id)
    within = lk - lk.groupby(is_cu).transform("mean")
    # material offset (a pooled trend) + opposite within-material trends that cancel in the stratified rho
    agg = 4.0 + 0.5 * is_cu + 0.1 * np.where(is_cu, -1.0, 1.0) * within
    r = pilot_gate(ds, _descriptors(ds, agg), _cfg())
    f = r.evidence["features"]["median_log_ecd"]
    assert abs(f["rho"]) < 1e-9 and f["p_value"] == pytest.approx(1.0) and not f["passes"]
    assert f["pooled_rho"] == pytest.approx(1 - 6 * 70 / (12 * 143))  # reported, not gated
    assert r.passed is False


def test_pilot_gate_design_criterion(tiny_dataset):
    desc = _descriptors(tiny_dataset, 4.0 + 0.3 * _log10k(tiny_dataset))
    r = pilot_gate(tiny_dataset, desc, _cfg("gates.min_runs_per_condition=4"))  # tiny has 3 runs per condition
    assert r.passed is False and len(r.evidence["conditions_below_min"]) == 4
    assert r.evidence["features"]["median_log_ecd"]["passes"]  # (b) alone cannot pass the gate


def test_pilot_gate_not_evaluable(tiny_dataset):
    desc = _descriptors(tiny_dataset, 4.0 + 0.3 * _log10k(tiny_dataset))
    small = _descriptors(tiny_dataset, pd.Series(1.0, index=tiny_dataset.runs.run_id))  # ECD 2.7 um < floor
    r = pilot_gate(tiny_dataset, small, _cfg())
    assert r.passed is None and r.evidence["n_particles_used"] == 0 and "not evaluable" in r.message
    tiny_dataset.runs["wear_rate_mm3_Nm"] = np.nan  # no measured wear rate
    assert pilot_gate(tiny_dataset, desc, _cfg()).passed is None


# ----------------------------------------------------------------------------- graph value

GOOD = {("particle_gnn", "default"): 0.95, ("contour_cnn", "default"): 0.3, ("fourier_fractal_svm", "default"): 0.3}


def test_graph_value_gate_pass_and_point_mode(tiny_dataset):
    preds = _particle_preds(tiny_dataset, GOOD)
    noise = _particle_preds(tiny_dataset, GOOD, protocol="external")  # other protocols are ignored
    r = graph_value_gate(pd.concat([preds, noise]), _cfg())
    assert r.passed is True
    for b in ("contour_cnn", "fourier_fractal_svm"):
        e = r.evidence[b]
        assert e["lo"] > 0 and e["passes"] and e["n_clusters"] == 12 and e["n_particles"] == 96
        assert e["diff"] == pytest.approx(e["proposed"] - e["baseline"])
    assert graph_value_gate(preds, _cfg("gates.graph_gate_mode=point")).passed is True


def test_graph_value_gate_fails_when_a_baseline_is_as_good(tiny_dataset):
    q = {**GOOD, ("contour_cnn", "default"): 0.95}
    r = graph_value_gate(_particle_preds(tiny_dataset, q), _cfg())
    assert r.passed is False
    assert not r.evidence["contour_cnn"]["passes"] and r.evidence["fourier_fractal_svm"]["passes"]


def test_graph_value_gate_mode_ci_needs_lower_bound(tiny_dataset):
    preds = _particle_preds(tiny_dataset, {**GOOD, ("contour_cnn", "default"): 0.85})
    ci = graph_value_gate(preds, _cfg())
    e = ci.evidence["contour_cnn"]
    assert e["diff"] > 0 and e["lo"] <= 0  # better point estimate, CI includes 0
    assert ci.passed is False
    assert graph_value_gate(preds, _cfg("gates.graph_gate_mode=point")).passed is True


def test_graph_value_gate_not_evaluable(tiny_dataset):
    no_svm = {k: v for k, v in GOOD.items() if k[0] != "fourier_fractal_svm"}
    r = graph_value_gate(_particle_preds(tiny_dataset, no_svm), _cfg())
    assert r.passed is None and r.evidence["fourier_fractal_svm"] == {"available": False}
    assert graph_value_gate(None, _cfg()).passed is None
    no_gnn = {k: v for k, v in GOOD.items() if k[0] != "particle_gnn"}
    assert graph_value_gate(_particle_preds(tiny_dataset, no_gnn), _cfg()).passed is None
    val_only = _particle_preds(tiny_dataset, GOOD, split="val")
    assert graph_value_gate(val_only, _cfg()).passed is None  # only test-split rows count
    with pytest.raises(ValueError):
        graph_value_gate(val_only, _cfg("gates.graph_gate_mode=always"))


# ----------------------------------------------------------------------------- mechanism


def _agreeing_labels(ds, reference=True):
    """Rater B copies rater A; optional SEMREF labels (= adjudicated) on particles with SEM or EDS."""
    lab = ds.labels.copy()
    a = lab[lab.rater_id == "A"].set_index("particle_id").label
    lab.loc[lab.rater_id == "B", "label"] = lab.loc[lab.rater_id == "B", "particle_id"].map(a).to_numpy()
    if reference:
        p = ds.particles[ds.particles.has_sem | ds.particles.has_eds]
        ref = pd.DataFrame({"particle_id": p.particle_id, "rater_id": "SEMREF",
                            "label": p.particle_id.map(ds.labels_adjudicated.set_index("particle_id").label),
                            "confidence": 5, "evidence": "sem_eds"})
        lab = pd.concat([lab, ref], ignore_index=True)
    ds.labels = coerce_table(lab, "labels")
    return ds


def test_mechanism_gate_pass(tiny_dataset):
    ds = _agreeing_labels(tiny_dataset)
    r = mechanism_gate(ds, _cfg("gates.min_double_rated=50"))
    assert r.passed is True
    assert r.evidence["agreement"]["kappa"] == pytest.approx(1.0)
    assert r.evidence["agreement"]["raters"] == ["A", "B"]  # SEMREF excluded from agreement
    assert r.evidence["validity"]["n"] == 48 and r.evidence["validity"]["kappa"] == pytest.approx(1.0)
    assert r.evidence["sem_eds_coverage"] == pytest.approx(0.5)  # particles 0 and 2 of every 4 have SEM / EDS


def test_mechanism_gate_failures(tiny_dataset):
    cfg = _cfg("gates.min_double_rated=50")
    # tiny_dataset: rater B disagrees on 2 of 4 particles -> low kappa
    r = mechanism_gate(tiny_dataset, cfg)
    assert r.passed is False and not r.evidence["agreement_passes"] and r.evidence["agreement"]["kappa"] < 0.6
    ds = _agreeing_labels(tiny_dataset)
    assert mechanism_gate(ds, _cfg()).passed is False  # 96 double-rated items < default min_double_rated 100
    assert mechanism_gate(ds, _cfg("gates.min_double_rated=50", "gates.min_sem_eds_fraction=0.9")).passed is False
    ds.labels = ds.labels[ds.labels.rater_id != "B"]
    r = mechanism_gate(ds, cfg)
    assert r.passed is False and r.evidence["agreement"]["raters"] == ["A"]  # fewer than 2 raters


def test_mechanism_gate_validity_not_evaluable_without_reference(tiny_dataset):
    ds = _agreeing_labels(tiny_dataset, reference=False)
    r = mechanism_gate(ds, _cfg("gates.min_double_rated=50"))
    assert r.evidence["agreement_passes"] and r.evidence["coverage_passes"]
    assert r.evidence["validity_passes"] is None and r.passed is None


# ----------------------------------------------------------------------------- wear

GOOD_EXT = {("wearpg", "full"): {"AISI52100": 0.05, "CuSn12": 0.5},
            ("physics_only", "default"): _both(0.2), ("conditions_only", "default"): _both(-0.3)}


def test_wear_gate_repeatability_and_external_pass(tiny_dataset):
    r = wear_gate(tiny_dataset, _cfg(), _run_preds(tiny_dataset, GOOD_EXT))
    lk = _log10k(tiny_dataset)
    sd = lk.groupby(tiny_dataset.runs.set_index("run_id").condition_id).std(ddof=1)
    rep = r.evidence["repeatability"]
    assert rep["median_sd_log10"] == pytest.approx(sd.median()) and rep["n_conditions"] == 4 and rep["passes"]
    assert rep["p_transition"] == {"AISI52100_F120": 1.0, "AISI52100_F20": 0.0, "CuSn12_F120": 1.0, "CuSn12_F20": 0.0}
    assert np.isfinite(rep["icc"]["icc"]) and np.isfinite(rep["median_cv"])
    ext = r.evidence["external"]
    assert ext["groups"]["logo_material_id_AISI52100"]["passes"]  # 0.05 <= best baseline 0.2
    assert not ext["groups"]["logo_material_id_CuSn12"]["passes"]  # 0.5 > 0.2
    assert ext["groups"]["logo_material_id_CuSn12"]["best_baseline"] == "physics_only"
    assert ext["passes"] is True and r.passed is True


def test_wear_gate_failures(tiny_dataset):
    worse = {**GOOD_EXT, ("wearpg", "full"): _both(0.5)}
    r = wear_gate(tiny_dataset, _cfg(), _run_preds(tiny_dataset, worse))
    assert r.evidence["repeatability"]["passes"] and r.evidence["external"]["passes"] is False and r.passed is False
    r = wear_gate(tiny_dataset, _cfg("gates.max_repeat_sd_log10=0.001"), _run_preds(tiny_dataset, GOOD_EXT))
    assert r.evidence["repeatability"]["passes"] is False and r.passed is False


def test_wear_gate_regime_mixed_conditions_excluded(tiny_dataset):
    rid = tiny_dataset.runs.index[tiny_dataset.runs.run_id == "R_CuSn12_120_0"]
    tiny_dataset.runs.loc[rid, "regime"] = "mild"
    rep = wear_gate(tiny_dataset, _cfg("gates.require_external=false")).evidence["repeatability"]
    assert rep["regime_mixed_conditions"] == ["CuSn12_F120"] and rep["n_conditions"] == 3
    assert "CuSn12_F120" not in rep["sd_log10"]


def test_wear_gate_not_evaluable_cases(tiny_dataset):
    assert wear_gate(tiny_dataset, _cfg()).passed is None  # external required but missing
    assert wear_gate(tiny_dataset, _cfg("gates.require_external=false")).passed is True
    # eta not transferable for CuSn12 (proposed NaN) and the AISI fold worse -> no group passes
    eta = {**GOOD_EXT, ("wearpg", "full"): {"AISI52100": 0.5, "CuSn12": np.nan}}
    r = wear_gate(tiny_dataset, _cfg(), _run_preds(tiny_dataset, eta))
    assert r.evidence["external"]["n_eta_not_transferable"] == 6
    assert list(r.evidence["external"]["groups"]) == ["logo_material_id_AISI52100"] and r.passed is False
    all_eta = {**GOOD_EXT, ("wearpg", "full"): _both(np.nan)}
    assert wear_gate(tiny_dataset, _cfg(), _run_preds(tiny_dataset, all_eta)).evidence["external"]["passes"] is None
    tiny_dataset.runs["wear_rate_mm3_Nm"] = np.nan
    r = wear_gate(tiny_dataset, _cfg("gates.require_external=false"))
    assert r.evidence["repeatability"]["passes"] is None and r.passed is None


# ----------------------------------------------------------------------------- wear rate


def _with_volumes(df, ds, slope=1.0, noise=0.05):
    """Adds log_grav_volume_mm3 (ln grav_volume_mm3) and log_debris_volume_mm3 = slope * grav + N(0, noise)."""
    lg = df.run_id.map(np.log(ds.grav_volume_mm3())).to_numpy(float)
    return df.assign(log_grav_volume_mm3=lg,
                     log_debris_volume_mm3=slope * lg + np.random.default_rng(1).normal(0, noise, len(df)))


def test_wear_rate_gate_pass(tiny_dataset):
    ds = tiny_dataset
    r = wear_rate_gate(_with_volumes(_run_preds(ds, R_OK, protocol="cv", seeds=(0, 1)), ds), _cfg())
    assert r.name == "wear_rate" and r.passed is True
    mae = r.evidence["mae"]
    assert set(mae) == {"conditions_only", "physics_only"} and all(m["passes"] for m in mae.values())
    assert mae["physics_only"]["diff"] == pytest.approx(0.1 - 0.2) and mae["physics_only"]["hi"] < 0
    vol = r.evidence["volume"]
    assert vol["passes"] and vol["slope"]["lo"] <= 1 <= vol["slope"]["hi"] and vol["ccc"]["point"] >= 0.8
    assert r.evidence["within_condition_spearman"]["lo"] > 0  # pred = true + constant within every condition


def test_wear_rate_gate_failures_and_not_evaluable(tiny_dataset):
    ds, cfg = tiny_dataset, _cfg()
    tie = _with_volumes(_run_preds(ds, {**R_OK, ("physics_only", "default"): _both(0.1)}, protocol="cv"), ds)
    r = wear_rate_gate(tie, cfg)  # as good as a baseline is not better
    assert r.evidence["mae"]["physics_only"]["passes"] is False and r.passed is False
    r = wear_rate_gate(_with_volumes(_run_preds(ds, R_OK, protocol="cv"), ds, slope=0.5), cfg)
    assert r.evidence["volume"]["passes"] is False and r.passed is False
    runs = _with_volumes(_run_preds(ds, R_OK, protocol="cv"), ds)
    w = runs.model == "wearpg"  # anti-correlated within conditions: condition mean right, run deviations reversed
    mean = runs[w].groupby("condition_id").log10_rate_true.transform("mean")
    runs.loc[w, "log10_rate_pred"] = 2 * mean - runs.loc[w, "log10_rate_true"]
    r = wear_rate_gate(runs, cfg)
    assert r.evidence["within_condition_spearman"]["passes"] is False and r.passed is False
    no_vol = _run_preds(ds, R_OK, protocol="cv").assign(log_grav_volume_mm3=np.nan, log_debris_volume_mm3=np.nan)
    r = wear_rate_gate(no_vol, cfg)
    assert r.evidence["volume"]["passes"] is None and r.passed is None
    assert wear_rate_gate(None, cfg).passed is None
    assert wear_rate_gate(_run_preds(ds, R_OK), cfg).passed is None  # external protocol only


# ----------------------------------------------------------------------------- portfolio

P_OK = {("wearpg", "full"): 0.9, ("contour_cnn", "default"): 0.5, ("fourier_fractal_svm", "default"): 0.6,
        ("wearpg", "no_graph"): 1.0}  # ablation variants are not baselines
R_OK = {("wearpg", "full"): _both(0.1), ("conditions_only", "default"): _both(0.3),
        ("physics_only", "default"): _both(-0.2)}


def test_portfolio_check_pass_and_fail(tiny_dataset):
    parts, runs = _particle_preds(tiny_dataset, P_OK), _run_preds(tiny_dataset, R_OK, protocol="cv")
    r = portfolio_check(parts, runs, _cfg())
    assert r.passed is True
    assert set(r.evidence["particles"]["comparisons"]) == {"contour_cnn", "fourier_fractal_svm"}
    assert r.evidence["runs"]["best_baseline"] == "physics_only"
    assert r.evidence["runs"]["comparisons"]["physics_only"]["baseline"] == pytest.approx(0.2)
    worse = _run_preds(tiny_dataset, {**R_OK, ("wearpg", "full"): _both(0.25)}, protocol="cv")
    r = portfolio_check(parts, worse, _cfg())
    assert r.passed is False and r.evidence["particles"]["passes"] and not r.evidence["runs"]["passes"]


def test_portfolio_check_not_evaluable(tiny_dataset):
    parts = _particle_preds(tiny_dataset, P_OK)
    no_wearpg = _run_preds(tiny_dataset, {k: v for k, v in R_OK.items() if k[0] != "wearpg"}, protocol="cv")
    r = portfolio_check(parts, no_wearpg, _cfg())
    assert r.passed is None and r.evidence["runs"] == {"available": False}
    external_only = _run_preds(tiny_dataset, R_OK)  # portfolio uses protocol cv only
    assert portfolio_check(parts, external_only, _cfg()).passed is None
    assert portfolio_check(None, None, _cfg()).passed is None


# ----------------------------------------------------------------------------- evaluate + report


def test_evaluate_gates_and_synthetic_report(tiny_dataset, tmp_path):
    ds = _agreeing_labels(tiny_dataset)
    cfg = _cfg("gates.min_double_rated=50")
    quality = {**GOOD, ("particle_gnn", "default"): 0.8, ("wearpg", "full"): 0.97}  # particle_gnn is a baseline too
    results = {"particles": _particle_preds(ds, quality),
               "runs": pd.concat([_with_volumes(_run_preds(ds, R_OK, protocol="cv"), ds), _run_preds(ds, GOOD_EXT)])}
    gates = evaluate_gates(ds, _descriptors(ds, 4.0 + 0.3 * _log10k(ds)), results, cfg)
    assert [g.name for g in gates] == ["pilot", "graph_value", "mechanism", "wear", "wear_rate", "portfolio"]
    assert all(isinstance(g, GateResult) and g.passed is True for g in gates)

    jp, mp = write_gate_report(gates, tmp_path / "gates", synthetic=ds.is_synthetic)
    doc = json.loads(jp.read_text(encoding="utf-8"))
    assert doc["synthetic"] is True and doc["decision_valid"] is False and doc["passed"] is True
    assert doc["banner"] == SYNTHETIC_BANNER and doc["notice"] == VERIFICATION_NOTICE
    assert [g["name"] for g in doc["gates"]] == [g.name for g in gates]
    assert doc["gates"][0]["evidence"]["features"]["median_log_ecd"]["passes"] is True  # bools stay bools
    md = mp.read_text(encoding="utf-8")
    assert SYNTHETIC_BANNER in md and VERIFICATION_NOTICE in md and "decision_valid: false" in md
    assert md.isascii() and md.count("| PASS |") == 6


def test_real_data_report_has_no_banner(tmp_path):
    gates = [GateResult("pilot", None, "not evaluable", {"min": 3}, {"rho": float("nan")}),
             GateResult("wear", False, "repeatability FAIL", {}, {"n": np.int64(4)})]
    jp, mp = write_gate_report(gates, tmp_path, synthetic=False)
    doc = json.loads(jp.read_text(encoding="utf-8"))
    assert doc["decision_valid"] is True and doc["passed"] is False and doc["banner"] == ""
    assert doc["gates"][1]["evidence"]["n"] == 4
    md = mp.read_text(encoding="utf-8")
    assert SYNTHETIC_BANNER not in md and VERIFICATION_NOTICE not in md
    assert "| pilot | NOT EVALUABLE |" in md and "| wear | FAIL |" in md
