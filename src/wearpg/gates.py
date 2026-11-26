"""[H] Go/no-go gates of the planning review (s12) and the portfolio-discipline check (DESIGN sec. 10).

Every gate returns a GateResult whose `passed` is True, False or None. None = not evaluable with the available data:
it is reported, never silently passed, and a gate is True only when all its criteria are True. On synthetic data the
values are kept, but the written report carries SYNTHETIC_BANNER and VERIFICATION_NOTICE and sets
decision_valid: false (DESIGN sec. 0 rule 1).

Prediction tables are the concatenated parts of schema.PARTICLE_PRED_COLUMNS / RUN_PRED_COLUMNS. The gates use only
split == "test" rows. The proposed model is ("wearpg", "full"); baselines are the other models with variant "default".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wearpg.baselines.run_level import NOTE_ETA
from wearpg.labels import agreement_report, validity_report
from wearpg.metrics import classification_metrics, regression_metrics, volume_consistency
from wearpg.schema import prob_columns
from wearpg.stats import (cluster_bootstrap, holm, icc_oneway, paired_cluster_bootstrap, repeatability_cv,
                          repeatability_sd, spearman_permutation)
from wearpg.utils import get_logger, load_json, save_json

from wearpg.schema import SYNTHETIC_BANNER, VERIFICATION_NOTICE  # single source
PROPOSED = ("wearpg", "full")
PILOT_FEATURES = ("median_log_ecd", "mean_fractal_D", "mean_aspect", "frac_large")


@dataclass
class GateResult:
    """One gate. passed: True | False | None (None = not evaluable with the available data). criteria = thresholds
    and settings used; evidence = computed values (JSON-safe)."""

    name: str
    passed: bool | None
    message: str
    criteria: dict
    evidence: dict


def _combine(*flags) -> bool | None:
    """Criteria of one gate: False if any is False, else None if any is not evaluable, else True."""
    if any(f is False for f in flags):
        return False
    return None if any(f is None for f in flags) else True


def _status(passed: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "NOT EVALUABLE"}[passed]


def _test(df: pd.DataFrame | None, protocol: str) -> pd.DataFrame | None:
    """Test-split rows of `protocol`; None if there are none."""
    if df is None or df.empty:
        return None
    out = df[(df.protocol == protocol) & (df.split == "test")]
    return out if len(out) else None


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.y_true >= 0]


def _scored_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Runs with a finite target and prediction (coverage < min and 'eta not transferable' rows are NaN)."""
    return df[np.isfinite(df.log10_rate_true.to_numpy(float)) & np.isfinite(df.log10_rate_pred.to_numpy(float))]


def _particle_metric(name: str):
    cols = prob_columns()
    return lambda d: classification_metrics(d.y_true.to_numpy(), d[cols].to_numpy(float))[name]


def _run_metric(name: str):
    return lambda d: regression_metrics(d.log10_rate_true.to_numpy(float), d.log10_rate_pred.to_numpy(float))[name]


def _seed_mean(df: pd.DataFrame, fn) -> float:
    """The metric per training seed, averaged over seeds (the headline point estimate, DESIGN sec. 11)."""
    return float(np.mean([fn(g) for _, g in df.groupby("seed")]))


def _versus(df: pd.DataFrame, key: str, fn) -> dict[str, dict]:
    """{baseline: {"proposed", "baseline", "n"}}: the seed-mean metric of PROPOSED and of each baseline (variant
    "default") on the `key` items both predicted. Empty if PROPOSED has no rows."""
    p = df[(df.model == PROPOSED[0]) & (df.variant == PROPOSED[1])]
    out = {}
    if p.empty:
        return out
    for b, bd in df[(df.model != PROPOSED[0]) & (df.variant == "default")].groupby("model"):
        common = np.intersect1d(p[key].unique(), bd[key].unique())
        if common.size:
            out[b] = {"proposed": _seed_mean(p[p[key].isin(common)], fn),
                      "baseline": _seed_mean(bd[bd[key].isin(common)], fn), "n": int(common.size)}
    return out


def pilot_gate(ds, descriptors: pd.DataFrame, cfg: dict) -> GateResult:
    """Pilot gate (s12.1): repeated runs per condition, and morphology that varies with the measured wear rate.

    (a) every condition has >= gates.min_runs_per_condition runs. (b) Per-run aggregates over the particles with
    shape_ecd_um >= 2 sqrt(detection_min_area_px / pi) * max(um_per_px). This floor is twice the detection-limit ECD
    at the coarsest primary-image pixel size [um], so magnification does not truncate the runs' size distributions
    differently. Aggregates: median shape_log_ecd_um, mean fractal_divider_D (fractal_valid particles only), mean
    shape_aspect_ratio, frac_large (share with ECD >= frac_large_um). Each is tested against the run's log10
    wear_rate_mm3_Nm with spearman_permutation (strata = material_id), Holm over the 4.
    passed iff (a) and >= 1 aggregate has p_adj < feature_regime_alpha and |rho| >= min_abs_spearman. False if (a)
    fails; None if no aggregate is testable. The pooled (unstratified) rho is reported, not gated.
    descriptors: particle descriptors indexed by particle_id (datasets.FeatureCache.descriptors).
    """
    g = cfg["gates"]
    alpha, min_rho = float(g["feature_regime_alpha"]), float(g["min_abs_spearman"])
    criteria = {k: g[k] for k in ("min_runs_per_condition", "feature_regime_alpha", "min_abs_spearman",
                                  "detection_min_area_px", "frac_large_um")}
    criteria.update(strata="material_id", multiplicity="holm", target="log10 wear_rate_mm3_Nm")
    runs = ds.runs
    per_cond = runs.groupby("condition_id").size()
    below = per_cond.index[per_cond < int(g["min_runs_per_condition"])].tolist()
    design_ok = len(per_cond) > 0 and not below

    um_max = ds.images.loc[ds.images.image_id.isin(ds.particles.image_id), "um_per_px"].max()
    floor = float(2 * np.sqrt(float(g["detection_min_area_px"]) / np.pi) * um_max)
    d = descriptors.reindex(ds.particles.particle_id.to_numpy())
    d["run_id"] = ds.particles.sample_id.map(ds.samples.set_index("sample_id").run_id).to_numpy()
    d = d[d.shape_ecd_um >= floor]
    by = d.groupby("run_id")
    agg = pd.DataFrame({
        "median_log_ecd": by.shape_log_ecd_um.median(),
        "mean_fractal_D": d[d.fractal_valid > 0.5].groupby("run_id").fractal_divider_D.mean(),
        "mean_aspect": by.shape_aspect_ratio.mean(),
        "frac_large": (d.shape_ecd_um >= float(g["frac_large_um"])).groupby(d.run_id).mean(),
    }).reindex(runs.run_id)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(runs.wear_rate_mm3_Nm.to_numpy(float))
    material = runs.specimen_id.map(ds.specimens.set_index("specimen_id").material_id).to_numpy()
    tests = {f: spearman_permutation(agg[f].to_numpy(float), y, seed=int(cfg.get("seed", 0)), strata=material)
             for f in PILOT_FEATURES}
    p_adj = holm([t["p_value"] for t in tests.values()])
    feats = {}
    for (f, t), pa in zip(tests.items(), p_adj):
        feats[f] = {**t, "p_adj": float(pa), "passes": bool(pa < alpha and abs(t["rho"]) >= min_rho),
                    "pooled_rho": spearman_permutation(agg[f].to_numpy(float), y, n_perm=0)["rho"]}  # rho only
    testable = [f for f in PILOT_FEATURES if np.isfinite(feats[f]["p_adj"])]
    n_pass = sum(feats[f]["passes"] for f in PILOT_FEATURES)
    passed = False if not design_ok else (None if not testable else n_pass > 0)

    msg = (f"(a) {len(per_cond) - len(below)}/{len(per_cond)} conditions have >= {g['min_runs_per_condition']} runs; "
           f"(b) {n_pass}/{len(PILOT_FEATURES)} run aggregates with p_adj < {alpha:g} and abs(rho) >= {min_rho:g}")
    if testable:
        best = min(testable, key=lambda f: feats[f]["p_adj"])
        msg += f" (best {best}: rho={feats[best]['rho']:.2f}, p_adj={feats[best]['p_adj']:.3g})"
    else:
        msg += " - not evaluable: no run has both a finite log10 wear rate and particle aggregates"
    evidence = {"runs_per_condition": {str(k): int(v) for k, v in per_cond.items()}, "conditions_below_min": below,
                "ecd_floor_um": floor, "n_particles_used": int(len(d)), "features": feats}
    return GateResult("pilot", passed, msg, criteria, evidence)


def graph_value_gate(particle_preds: pd.DataFrame | None, cfg: dict) -> GateResult:
    """Graph-value gate (s12.2): the graph model must beat the contour CNN / Fourier descriptors on held-out units.

    cv test rows of gates.graph_gate_proposed (model, variant) vs each gates.graph_gate_baselines model (variant
    "default"): paired_cluster_bootstrap of macro-F1 on the labelled particles both predicted (clusters =
    eval.split_level units, training seeds resampled, eval.n_bootstrap replicates, eval.ci level). gates.graph_gate_mode
    "ci": the lower CI bound of the difference > 0 for every baseline; "point": the difference > 0. False if any
    comparison fails; None if the proposed model or a baseline has no predictions (and nothing failed).
    """
    g, e = cfg["gates"], cfg["eval"]
    pm, pv = g["graph_gate_proposed"]
    mode = g["graph_gate_mode"]
    if mode not in ("ci", "point"):
        raise ValueError(f"gates.graph_gate_mode must be ci|point, got {mode!r}")
    criteria = {"proposed": [pm, pv], "baselines": list(g["graph_gate_baselines"]), "mode": mode,
                "metric": "macro_f1", "protocol": "cv", "cluster": e["split_level"],
                "n_bootstrap": int(e["n_bootstrap"]), "ci": float(e["ci"])}
    df = _test(particle_preds, "cv")
    a = None if df is None else _labelled(df[(df.model == pm) & (df.variant == pv)])
    if a is None or a.empty:
        return GateResult("graph_value", None, f"not evaluable: no labelled cv test predictions of {pm}/{pv}",
                          criteria, {})
    fn = _particle_metric("macro_f1")
    evidence, flags = {}, []
    for b in g["graph_gate_baselines"]:
        bd = _labelled(df[(df.model == b) & (df.variant == "default")])
        common = np.intersect1d(a.particle_id.unique(), bd.particle_id.unique())
        if not common.size:
            evidence[b] = {"available": False}
            flags.append(None)
            continue
        ac, bc = a[a.particle_id.isin(common)], bd[bd.particle_id.isin(common)]
        r = paired_cluster_bootstrap(fn, ac, bc, f"{e['split_level']}_id", "particle_id", seed_col="seed",
                                     n_boot=int(e["n_bootstrap"]), ci=float(e["ci"]), seed=int(cfg.get("seed", 0)))
        ok = bool(r["value"]["lo"] > 0 if mode == "ci" else r["value"]["diff"] > 0)
        evidence[b] = {"available": True, "proposed": _seed_mean(ac, fn), "baseline": _seed_mean(bc, fn),
                       **r["value"], "n_clusters": int(r["n_clusters"]), "n_particles": int(common.size), "passes": ok}
        flags.append(ok)
    beaten = [b for b, f in zip(g["graph_gate_baselines"], flags) if f]
    missing = [b for b, f in zip(g["graph_gate_baselines"], flags) if f is None]
    msg = (f"{pm} macro-F1 beats {len(beaten)}/{len(flags)} baselines "
           f"({'lower CI bound' if mode == 'ci' else 'point difference'} > 0)")
    if missing:
        msg += f"; no common predictions for {', '.join(missing)}"
    return GateResult("graph_value", _combine(*flags), msg, criteria, evidence)


def mechanism_gate(ds, cfg: dict) -> GateResult:
    """Mechanism gate (s12.3): independent label evidence and inter-rater agreement.

    agreement: labels.agreement_report kappa (non-reference raters) >= gates.min_kappa with >= 2 raters and
    >= min_double_rated items rated by >= 2 raters, else False. validity: labels.validity_report kappa of the
    adjudicated vs gates.reference_raters labels >= min_validity_kappa (None without common reference labels or with an
    undefined kappa). coverage: share of adjudicated (labelled) particles with SEM or EDS (has_sem | has_eds)
    >= min_sem_eds_fraction (None without labelled particles).
    """
    g = cfg["gates"]
    refs = tuple(g["reference_raters"])
    criteria = {k: g[k] for k in ("min_kappa", "min_double_rated", "min_validity_kappa", "min_sem_eds_fraction")}
    criteria["reference_raters"] = list(refs)
    ag = agreement_report(ds.labels, reference_raters=refs)
    agree = bool(len(ag["raters"]) >= 2 and ag["n_multi_rated"] >= int(g["min_double_rated"])
                 and ag["kappa"] >= float(g["min_kappa"]))
    va = validity_report(ds.labels, ds.labels_adjudicated, reference_raters=refs)
    valid = (None if va["n"] == 0 or not np.isfinite(va["kappa"])
             else bool(va["kappa"] >= float(g["min_validity_kappa"])))
    lab = ds.particles[ds.particles.particle_id.isin(ds.labels_adjudicated.particle_id)]
    coverage = float((lab.has_sem | lab.has_eds).mean()) if len(lab) else float("nan")
    cov_ok = None if not len(lab) else bool(coverage >= float(g["min_sem_eds_fraction"]))
    msg = (f"agreement {_status(agree)} (kappa={ag['kappa']:.3f} [{ag['kappa_method']}], {len(ag['raters'])} raters, "
           f"{ag['n_multi_rated']} multi-rated items); validity {_status(valid)} (kappa={va['kappa']:.3f}, "
           f"n={va['n']}); SEM/EDS coverage {_status(cov_ok)} ({coverage:.3f} of {len(lab)} labelled particles)")
    evidence = {"agreement": ag, "agreement_passes": agree, "validity": va, "validity_passes": valid,
                "sem_eds_coverage": coverage, "n_labelled": int(len(lab)), "coverage_passes": cov_ok}
    return GateResult("mechanism", _combine(agree, valid, cov_ok), msg, criteria, evidence)


def wear_gate(ds, cfg: dict, external_run_preds: pd.DataFrame | None = None) -> GateResult:
    """Wear gate (s12.4): repeatability statistics and >= 1 external material or operating condition.

    repeatability: median per-condition SD (ddof 1) of log10 wear_rate_mm3_Nm over the repeats of regime-homogeneous
    conditions (a single runs.regime value) <= gates.max_repeat_sd_log10; None without such a condition with >= 2
    finite repeats. Reported, not gated: median CV of k, ICC(1) of log10 k (same conditions), the regime-mixed
    conditions, per-condition P(transition) (share of runs with a transition time or final regime severe).
    external (protocol "external", split test; held-out group = fold): passes iff for >= 1 group the proposed model's
    run log10-k MAE <= the best run baseline's MAE, each pair compared on the runs both scored (finite target and
    prediction: coverage-excluded and 'eta not transferable' runs drop out and the latter are counted). Without
    external predictions: None if gates.require_external, else this criterion is skipped.
    """
    g = cfg["gates"]
    criteria = {"max_repeat_sd_log10": g["max_repeat_sd_log10"], "require_external": bool(g["require_external"]),
                "external_metric": "mae of log10 k", "external_rule": "proposed <= best run baseline in >= 1 group"}
    r = ds.runs
    k = r.wear_rate_mm3_Nm.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lk = np.log10(k)
    cond = r.condition_id.to_numpy()
    mixed = (r.groupby("condition_id").regime.transform("nunique") > 1).to_numpy()
    sd = repeatability_sd(lk[~mixed], cond[~mixed])
    rep = None if not sd["n_groups"] else bool(sd["median"] <= float(g["max_repeat_sd_log10"]))
    p_tr = (r.transition_time_s.notna() | r.regime.eq("severe")).groupby(r.condition_id).mean()
    evidence = {"repeatability": {
        "median_sd_log10": sd["median"], "sd_log10": sd["per_group"], "n_conditions": sd["n_groups"],
        "median_cv": repeatability_cv(k[~mixed], cond[~mixed])["median"], "icc": icc_oneway(lk[~mixed], cond[~mixed]),
        "regime_mixed_conditions": sorted(set(cond[mixed].tolist())), "p_transition": p_tr.to_dict(),
        "passes": rep}}
    msg = (f"repeatability {_status(rep)} (median SD log10 k = {sd['median']:.3f} over {sd['n_groups']} "
           f"regime-homogeneous conditions, {int(mixed.sum())} runs in regime-mixed conditions)")

    flags = [rep]
    ext = _test(external_run_preds, "external")
    if ext is None:
        if g["require_external"]:
            flags.append(None)
            msg += "; external NOT EVALUABLE (no external run predictions)"
        else:
            msg += "; external not required and no external run predictions"
    else:
        groups = {}
        for fold, d in _scored_runs(ext).groupby("fold"):
            v = _versus(d, "run_id", _run_metric("mae"))
            if v:
                groups[str(fold)] = {"mae": v, "best_baseline": min(v, key=lambda b: v[b]["baseline"]),
                                     "passes": bool(all(x["proposed"] <= x["baseline"] for x in v.values()))}
        n_ok = sum(x["passes"] for x in groups.values())
        ext_ok = None if not groups else n_ok > 0
        flags.append(ext_ok)
        eta = ext[(ext.model == PROPOSED[0]) & (ext.variant == PROPOSED[1]) & (ext.note == NOTE_ETA)]
        evidence["external"] = {"groups": groups, "passes": ext_ok,
                                "n_eta_not_transferable": int(len(eta[["fold", "run_id"]].drop_duplicates()))}
        msg += f"; external {_status(ext_ok)} ({n_ok}/{len(groups)} held-out groups with MAE <= best run baseline)"
    return GateResult("wear", _combine(*flags), msg, criteria, evidence)


def wear_rate_gate(run_preds: pd.DataFrame | None, cfg: dict) -> GateResult:
    """Wear-rate gate (main claim C1 / C1b, FRAMING sec. 10): protocol cv, split test, scored runs only.

    (a) accuracy: gates.wear_rate_proposed (model, variant) vs each gates.wear_rate_baselines model (variant "default"):
    paired_cluster_bootstrap of the run MAE of log10 k on the runs both scored; passes iff the upper CI bound of
    (proposed - baseline) < 0 for every baseline. (b) volume consistency of the proposed model: cluster bootstrap of
    metrics.volume_consistency (debris-implied vs gravimetric ln run volume); passes iff the slope CI contains 1 and the
    CCC point estimate >= gates.min_volume_ccc. (c) beyond the operating conditions: cluster bootstrap of the
    within-condition Spearman of log10 k; passes iff its lower CI bound > 0. Clusters = eval.split_level units, training
    seeds resampled (eval.n_bootstrap, eval.ci); points are seed means. A criterion without data is None.
    """
    g, e = cfg["gates"], cfg["eval"]
    pm, pv = g["wear_rate_proposed"]
    min_ccc, cluster = float(g["min_volume_ccc"]), f"{e['split_level']}_id"
    boot = {"n_boot": int(e["n_bootstrap"]), "ci": float(e["ci"]), "seed": int(cfg.get("seed", 0))}
    criteria = {"proposed": [pm, pv], "baselines": list(g["wear_rate_baselines"]), "metric": "run mae of log10 k",
                "mae_rule": "upper CI bound of (proposed - baseline) < 0 for every baseline",
                "volume_rule": "slope CI contains 1 and CCC >= min_volume_ccc", "min_volume_ccc": min_ccc,
                "within_rule": "within-condition Spearman lower CI bound > 0", "protocol": "cv",
                "cluster": e["split_level"], "n_bootstrap": boot["n_boot"], "ci": boot["ci"]}
    df = _test(run_preds, "cv")
    p = None if df is None else _scored_runs(df[(df.model == pm) & (df.variant == pv)])
    if p is None or p.empty:
        return GateResult("wear_rate", None, f"not evaluable: no scored cv test runs of {pm}/{pv}", criteria, {})

    fn, evidence, flags = _run_metric("mae"), {"mae": {}}, []
    for b in g["wear_rate_baselines"]:
        bd = _scored_runs(df[(df.model == b) & (df.variant == "default")])
        common = np.intersect1d(p.run_id.unique(), bd.run_id.unique())
        if not common.size:
            evidence["mae"][b] = {"available": False}
            flags.append(None)
            continue
        pc, bc = p[p.run_id.isin(common)], bd[bd.run_id.isin(common)]
        r = paired_cluster_bootstrap(fn, pc, bc, cluster, "run_id", seed_col="seed", **boot)["value"]
        evidence["mae"][b] = {"available": True, "proposed": _seed_mean(pc, fn), "baseline": _seed_mean(bc, fn), **r,
                              "n_runs": int(common.size), "passes": bool(r["hi"] < 0)}
        flags.append(bool(r["hi"] < 0))
    mae_ok = _combine(*flags)

    vol = p[np.isfinite(p.log_debris_volume_mm3.to_numpy(float)) & np.isfinite(p.log_grav_volume_mm3.to_numpy(float))]
    vol_ok, vc = None, {}
    if vol.run_id.nunique() >= 3:
        vc = cluster_bootstrap(lambda d: volume_consistency(d.log_debris_volume_mm3, d.log_grav_volume_mm3), vol,
                               cluster, "seed", **boot)
        s, c = vc["slope"], vc["ccc"]
        if np.isfinite(s["point"]) and np.isfinite(c["point"]):
            vol_ok = bool(s["lo"] <= 1.0 <= s["hi"] and c["point"] >= min_ccc)
    evidence["volume"] = {k: vc.get(k) for k in ("slope", "intercept", "ccc", "mean_log_ratio")}
    evidence["volume"].update(n_runs=int(vol.run_id.nunique()), passes=vol_ok)

    w = cluster_bootstrap(lambda d: regression_metrics(d.log10_rate_true, d.log10_rate_pred, d.condition_id)
                          ["within_condition_spearman"], p, cluster, "seed", **boot)["value"]
    within_ok = bool(w["lo"] > 0) if np.isfinite(w["lo"]) else None
    evidence["within_condition_spearman"] = {**w, "passes": within_ok}

    def ci(x) -> str:
        return "n/a" if not x else f"{x['point']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"

    beaten = sum(bool(f) for f in flags)
    msg = (f"{pm}/{pv}: run MAE beats {beaten}/{len(flags)} baselines (upper CI bound < 0) {_status(mae_ok)}; "
           f"volume slope {ci(vc.get('slope'))}, CCC {ci(vc.get('ccc'))} {_status(vol_ok)}; "
           f"within-condition Spearman {ci(w)} {_status(within_ok)}")
    return GateResult("wear_rate", _combine(mae_ok, vol_ok, within_ok), msg, criteria, evidence)


def portfolio_check(particle_preds: pd.DataFrame | None, run_preds: pd.DataFrame | None, cfg: dict) -> GateResult:
    """Portfolio discipline (s12): no full campaign before the smoke test beats the strongest relevant baseline.

    Protocol cv, split test: the proposed model (wearpg/full) must be strictly better than every baseline (variant
    "default") at EVERY level of eval.claim_metrics ({level: {metric, better: max|min}}), on point estimates
    (seed-mean metric on the items both models scored). Supported levels: particles (metrics.classification_metrics on
    labelled particles) and runs (metrics.regression_metrics of log10 k on scored runs). A level without proposed or
    baseline predictions (or another level) is None.
    """
    claims = cfg["eval"]["claim_metrics"]
    levels = {"particles": (particle_preds, "particle_id", _particle_metric, _labelled),
              "runs": (run_preds, "run_id", _run_metric, _scored_runs)}
    criteria = {"protocol": "cv", "claim_metrics": claims, "proposed": list(PROPOSED), "estimate": "point (seed mean)"}
    evidence, flags, parts = {}, [], []
    for level, spec in claims.items():
        preds, key, metric, valid = levels.get(level, (None, None, None, None))
        df = _test(preds, "cv")
        v = {} if df is None else _versus(valid(df), key, metric(spec["metric"]))
        if not v:
            evidence[level] = {"available": False}
            flags.append(None)
            parts.append(f"{level} NOT EVALUABLE")
            continue
        sign = {"max": 1.0, "min": -1.0}[spec["better"]]
        best = max(v, key=lambda b: sign * v[b]["baseline"])
        ok = bool(all(sign * (x["proposed"] - x["baseline"]) > 0 for x in v.values()))
        evidence[level] = {"available": True, "metric": spec["metric"], "comparisons": v, "best_baseline": best,
                           "passes": ok}
        flags.append(ok)
        parts.append(f"{level} {spec['metric']} {_status(ok)} (wearpg {v[best]['proposed']:.4g} vs best baseline "
                     f"{best} {v[best]['baseline']:.4g})")
    return GateResult("portfolio", _combine(*flags), "; ".join(parts), criteria, evidence)


def evaluate_gates(ds, descriptors: pd.DataFrame, results: dict, cfg: dict) -> list[GateResult]:
    """All gates in order pilot, graph_value, mechanism, wear, wear_rate, portfolio. results = {"particles", "samples",
    "runs"}: concatenated prediction parts filtered to split == "test" (a missing level = no predictions)."""
    particles, runs = results.get("particles"), results.get("runs")
    return [pilot_gate(ds, descriptors, cfg), graph_value_gate(particles, cfg), mechanism_gate(ds, cfg),
            wear_gate(ds, cfg, runs), wear_rate_gate(runs, cfg), portfolio_check(particles, runs, cfg)]


def write_gate_report(results: list[GateResult], out_dir, synthetic: bool) -> tuple[Path, Path]:
    """Write out_dir/gates.json and out_dir/gates.md; returns their paths.

    gates.json = {"synthetic", "decision_valid" (= not synthetic), "passed" (all gates combined), "banner", "notice",
    "gates": [GateResult as dict]}. On synthetic data both files start with SYNTHETIC_BANNER and VERIFICATION_NOTICE and
    decision_valid is false; the gate values are kept (DESIGN sec. 0 rule 1).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    passed = _combine(*(r.passed for r in results))
    doc = {"synthetic": bool(synthetic), "decision_valid": not synthetic, "passed": passed,
           "banner": SYNTHETIC_BANNER if synthetic else "", "notice": VERIFICATION_NOTICE if synthetic else "",
           "gates": [asdict(r) for r in results]}
    json_path = save_json(doc, out / "gates.json")
    gates = load_json(json_path)["gates"]  # render the markdown from exactly what the JSON holds (plain types)
    lines = ["# Go/no-go gates", ""]
    if synthetic:
        lines += [f"**{SYNTHETIC_BANNER}**", "", f"**{VERIFICATION_NOTICE}** (decision_valid: false)", ""]
    lines += [f"Overall: {_status(passed)}", "", "| gate | status | message |", "|---|---|---|"]
    lines += [f"| {r['name']} | {_status(r['passed'])} | {r['message']} |" for r in gates]
    for r in gates:
        lines += ["", f"## {r['name']}: {_status(r['passed'])}", "", r["message"], "", "Criteria:", "", "```json",
                  json.dumps(r["criteria"], indent=2), "```", "", "Evidence:", "", "```json",
                  json.dumps(r["evidence"], indent=2), "```"]
    md_path = out / "gates.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if synthetic:
        get_logger().warning("gates: %s - %s", VERIFICATION_NOTICE, SYNTHETIC_BANNER)
    return json_path, md_path
