"""[I1] Experiment orchestration (DESIGN sec. 11): data + feature cache, grouped splits, per split x seed the proposed
variants and the baselines, resumable prediction parts, evaluation (cluster + seed bootstrap), gates and the smoke run.

Results layout (paths.output_dir): config.yaml (with config_hash), dataset_meta.json, splits.json,
predictions/parts/{level}__{protocol}__{model}__{variant}__{fold}__seed{seed}.parquet (every name component safe_name'd;
written atomically; a model whose parts all exist is skipped), training/{model}/{variant}/{fold}_seed{seed}/,
metrics/*.csv + summary.json + claims.csv, gates/gates.{json,md}, <cfg name>_report.md, report/.

run_proposed / run_particle_baseline / run_run_baseline return FULL schema prediction tables (keys, ids and targets
attached by attach_hierarchy, split "test"), so callers (cli train) can write them as parts directly.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from wearpg.baselines.classical import CLASSICAL_BASELINES, make_particle_baseline
from wearpg.baselines.deep import (DEEP_BASELINES, make_deep_baseline, predict_particle_proba,
                                   train_particle_classifier)
from wearpg.baselines.run_level import NOTE_ETA, RUN_BASELINES, make_run_baseline
from wearpg.config import apply_overrides, load_config
from wearpg.datasets import (PARTICLE_GRAPH_SELECTION, FeatureCache, _mech_targets, baseline_global_columns,
                             build_cache, class_weights, fit_scalers, magnetic_index, make_particle_loader,
                             make_run_loader, material_index, model_dims)
from wearpg.gates import PROPOSED, evaluate_gates, write_gate_report
from wearpg.graphs import selected_dims
from wearpg.metrics import (classification_metrics, metrics_by_group, regression_metrics, severity_metrics,
                            transition_timing, volume_consistency)
from wearpg.models import MechanismLinkedModel
from wearpg.schema import (MECHANISMS, PARTICLE_PRED_COLUMNS, RUN_PRED_COLUMNS, SAMPLE_PRED_COLUMNS,
                           SEVERITY_PROB_COLUMNS, SYNTHETIC_BANNER, VERIFICATION_NOTICE, WearDataset, prob_columns)
from wearpg.splits import (Split, audit_leakage, grouped_kfold, leave_one_group_out, load_splits, particles_for_runs,
                           safe_name, save_splits, transfer_split)
from wearpg.stats import adjust, cluster_bootstrap, paired_cluster_bootstrap, seed_summary
from wearpg.training import fit_model, predict_runs
from wearpg.utils import get_device, get_logger, save_json, seed_everything, stable_hash

VARIANTS: dict[str, list[str]] = {
    "full": [], "no_graph": ["model.use_graph=false"], "edges_ring": ["model.edges=ring"],
    "edges_none": ["model.edges=none"], "no_interior": ["model.use_interior_nodes=false"],
    "no_fractal": ["model.use_fractal_features=false"], "no_texture": ["model.use_texture_branch=false"],
    "no_multiview": ["model.use_multiview=false"], "no_composition": ["model.use_composition=false"],
    "no_run_aggregation": ["model.use_run_aggregation=false"],
    "no_physics": ["model.physics_head=false", "loss.consistency=0", "loss.monotone=0"],
    "no_consistency": ["loss.consistency=0"], "with_conditions": ["model.use_operating_conditions=true"],
    "no_mechanism_link": ["model.mechanism_link=false"],
}
LEVEL_COLUMNS = {"particles": PARTICLE_PRED_COLUMNS, "samples": SAMPLE_PRED_COLUMNS, "runs": RUN_PRED_COLUMNS}
LEVEL_NAME = {"particles": "particle", "samples": "sample", "runs": "run"}  # level column of regression/comparisons
NOTE_EMPTY = "no observed sample"  # run without surviving particles: absent from the loaders
METRIC_COLUMNS = {
    "classification": ["protocol", "model", "variant", "metric", "point", "lo", "hi", "seed_mean", "seed_sd", "n"],
    "per_class": ["protocol", "model", "variant", "class", "metric", "point", "lo", "hi", "n"],
    "severity": ["protocol", "model", "variant", "metric", "point", "lo", "hi", "seed_mean", "seed_sd", "n"],
    "regression": ["protocol", "model", "variant", "level", "metric", "point", "lo", "hi", "seed_mean", "seed_sd", "n"],
    "comparisons": ["protocol", "level", "family", "metric", "model_a", "variant_a", "model_b", "variant_b", "diff",
                    "lo", "hi", "rel_diff", "p_value", "p_adj", "method"],
    "per_group": ["protocol", "model", "variant", "group_col", "group", "metric", "value", "n"],
    "confusion_by_group": ["protocol", "model", "variant", "group_col", "group", "true", "pred", "count"],
    "transition": ["protocol", "model", "variant", "mae_s", "bias_s", "mae_intervals", "detection_rate",
                   "early_alarm_rate", "false_alarm_rate", "oracle_floor_s", "interval_s", "n_transition_runs",
                   "n_stable_runs"],
    "volume_consistency": ["protocol", "model", "variant", "mean_log_ratio", "sd_log_ratio", "slope", "intercept",
                           "r2", "ccc", "n"],
    "exclusions": ["protocol", "model", "variant", "reason", "n_runs"],  # extra: excluded runs per reason, counted
}
_CLS = ("macro_f1", "balanced_accuracy", "accuracy", "ece", "brier", "log_loss")
_PER_CLASS = ("sensitivity", "specificity", "f1")
_SEV = ("macro_f1", "balanced_accuracy", "ordinal_mae", "qwk")
_REG = ("mae", "rmse", "r2", "bias", "spearman", "within_condition_spearman")
_KEY = {"particles": "particle_id", "samples": "sample_id", "runs": "run_id"}
_GROUP = ["protocol", "model", "variant"]
_TARGETS = ["log10_rate_pred", "log10_rate_true", "log10_rate_true_gt", "condition_id"]
_METRIC_INPUTS = {"particles": [*prob_columns(), "y_true", "y_true_gt"],  # the only columns the metric fns read
                  "samples": [*SEVERITY_PROB_COLUMNS, "severity_true", *_TARGETS], "runs": _TARGETS}
_FILL = {"y_pred": -1, "severity_pred": -1, "note": ""}
_STATUS = {True: "PASS", False: "FAIL", None: "NOT EVALUABLE"}
LN10 = math.log(10.0)


# ============================================================================ helpers


def _ln(x) -> np.ndarray:
    """Natural log of positive values, NaN elsewhere (no warnings)."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    np.log(x, out=out, where=x > 0)
    return out


def _log10(x) -> np.ndarray:
    return _ln(x) / LN10


def _min_cov(cfg: dict) -> float:
    """Run-level scoring threshold for every protocol. Transfer too: magnification / focus vary per sample, so a
    transfer test run is never fully covered; its run volume is coverage-corrected (DESIGN sec. 3)."""
    return float(cfg["physics"]["min_run_coverage"])


def _device(cfg: dict, device) -> torch.device:
    return torch.device(device) if device is not None else get_device(cfg.get("device", "auto"))


def _role_pids(ds: WearDataset, cache: FeatureCache, runs, image_filter, role: str) -> list[str]:
    """Surviving particles of `runs` for `role` (splits.particles_for_runs) that are in the feature cache."""
    cached = set(cache.particle_ids)
    return [p for p in particles_for_runs(ds, runs, image_filter, role) if p in cached]


def variant_config(cfg: dict, variant: str) -> dict:
    """cfg with the VARIANTS overrides of `variant` applied."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    return apply_overrides(cfg, VARIANTS[variant])


# ============================================================================ data, splits, prediction tables


def prepare(cfg: dict, force: bool = False) -> tuple[WearDataset, FeatureCache]:
    """Load paths.data_root (generated first from paths.synthetic_config when missing), assert_valid, feature cache
    (force: rebuild it even when current)."""
    root = Path(cfg["paths"]["data_root"])
    if not (root / "dataset.json").exists():
        syn = cfg["paths"].get("synthetic_config")
        if not syn:
            raise FileNotFoundError(f"no dataset at {root} and no paths.synthetic_config to generate one")
        from wearpg.synthetic import generate_dataset, load_synthetic_config

        get_logger().info("prepare: generating the synthetic dataset %s from %s", root, syn)
        generate_dataset(load_synthetic_config(syn), root)
    ds = WearDataset.load(root)
    ds.assert_valid()
    return ds, build_cache(ds, cfg, force=force)


def make_splits(ds: WearDataset, cfg: dict) -> list[Split]:
    """Grouped k-fold over eval.split_level units (protocol cv), stratified by eval.stratify."""
    e = cfg["eval"]
    return grouped_kfold(ds, e["split_level"], int(e["n_splits"]), int(cfg["seed"]), float(e["val_fraction"]),
                         e["stratify"])


def _alarm_times(samples: pd.DataFrame, ds: WearDataset, threshold: float) -> pd.Series:
    """run_id -> midpoint [s] of the run's first sample interval with P(severity >= 1) >= threshold."""
    s = ds.samples.set_index("sample_id").loc[samples.sample_id, ["run_id", "t_start_s", "time_s"]]
    p = (samples.sev_p_transition + samples.sev_p_severe).to_numpy(float)
    first = s[p >= threshold].sort_values(["run_id", "t_start_s"]).drop_duplicates("run_id")
    return pd.Series(((first.t_start_s + first.time_s) / 2).to_numpy(), index=first.run_id.to_numpy())


def attach_hierarchy(df: pd.DataFrame, ds: WearDataset, level: str, cfg: dict,
                     samples: pd.DataFrame | None = None) -> pd.DataFrame:
    """Prediction values + PRED_KEYS -> the schema prediction table of `level` (particles | samples | runs).

    Adds ids / attributes and targets from ds. particles: y_true (adjudicated class index, -1 unlabelled), y_true_gt
    (synthetic truth, -1 unknown), log_volume_true (ln measured volume_um3). samples: severity_true, log10_rate_true
    (log10 measured interval k = interval_wear_volume_mm3 / (F ds) [mm^3/(N m)]). runs: log10_rate_true,
    log_grav_volume_mm3 (ln grav_volume_mm3()), transition_time_true and transition_time_pred [s] = midpoint of the
    run's first interval whose P(severity >= 1) = sev_p_transition + sev_p_severe in `samples` (the same model's sample
    values) reaches eval.transition_threshold (NaN without an alarm or without `samples`). Run targets are NaN where
    coverage < physics.min_run_coverage (every protocol), so no metric or gate scores them. Absent value columns
    become NaN (-1 for y_pred / severity_pred, "" for note).
    """
    if level not in LEVEL_COLUMNS:
        raise ValueError(f"level must be one of {tuple(LEVEL_COLUMNS)}, got {level!r}")
    out = df.copy()
    rt = ds.run_table().set_index("run_id")
    if level == "particles":
        pid = out.particle_id
        h = ds.particle_hierarchy().set_index("particle_id")
        for c in ("image_id", "sample_id", "run_id", "specimen_id", "material_id", "condition_id", "magnification"):
            out[c] = pid.map(h[c])
        out["y_true"] = pid.map(ds.label_indices("adjudicated")).fillna(-1).astype(np.int64)
        out["y_true_gt"] = pid.map(ds.label_indices("true")).fillna(-1).astype(np.int64)
        out["log_volume_true"] = _ln(pid.map(ds.particles.set_index("particle_id").volume_um3))
    elif level == "samples":
        s, sid = ds.samples.set_index("sample_id"), out.sample_id
        out["run_id"] = sid.map(s.run_id)
        out["t_start_s"], out["time_s"] = sid.map(s.t_start_s), sid.map(s.time_s)
        out["severity_true"] = sid.map(s.severity).fillna(-1).astype(np.int64)
        k = s.interval_wear_volume_mm3 / (s.run_id.map(rt.load_N) * (s.sliding_distance_end_m - s.sliding_distance_start_m))
        out["log10_rate_true"] = _log10(sid.map(k))
    else:
        rid = out.run_id
        out["log10_rate_true"] = _log10(rid.map(rt.wear_rate_mm3_Nm))
        out["log_grav_volume_mm3"] = _ln(rid.map(ds.grav_volume_mm3()))
        out["transition_time_true"] = rid.map(rt.transition_time_s)
        out["transition_time_pred"] = (np.nan if samples is None else
                                       rid.map(_alarm_times(samples, ds, float(cfg["eval"]["transition_threshold"]))))
        low = ~(out.coverage.to_numpy(float) >= _min_cov(cfg))
        out.loc[low, ["log10_rate_true", "log_grav_volume_mm3"]] = np.nan
    if level != "particles":
        for c in ("specimen_id", "material_id", "condition_id"):
            out[c] = out.run_id.map(rt[c])
    for c in LEVEL_COLUMNS[level]:
        if c not in out.columns:
            out[c] = _FILL.get(c, np.nan)
    return out[LEVEL_COLUMNS[level]].reset_index(drop=True)


def _finish(values: dict[str, pd.DataFrame], ds: WearDataset, cfg: dict, split: Split, seed: int, model: str,
            variant: str) -> dict[str, pd.DataFrame]:
    """Value-column frames -> full prediction tables (keys + attach_hierarchy) of the split's test runs."""
    keys = {"protocol": split.protocol, "model": model, "variant": variant, "fold": split.name, "seed": int(seed),
            "split": "test"}
    return {lvl: attach_hierarchy(df.assign(**keys), ds, lvl, cfg, samples=values.get("samples"))
            for lvl, df in values.items()}


def _part_path(out_dir, level: str, protocol: str, model: str, variant: str, fold: str, seed: int) -> Path:
    name = "__".join(safe_name(x) for x in (level, protocol, model, variant, fold)) + f"__seed{int(seed)}.parquet"
    return Path(out_dir) / "predictions" / "parts" / name


def _write_part(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)  # Path.rename fails on Windows when the target exists


def load_predictions(results_dir, level: str) -> pd.DataFrame:
    """All prediction parts of `level` concatenated (every protocol / model / fold / seed / split)."""
    files = sorted((Path(results_dir) / "predictions" / "parts").glob(f"{level}__*.parquet"))
    if not files:
        return pd.DataFrame(columns=LEVEL_COLUMNS[level])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ============================================================================ models


def run_proposed(ds: WearDataset, cache: FeatureCache, cfg: dict, split: Split, seed: int, variant: str = "full",
                 device=None, out_dir=None) -> dict[str, pd.DataFrame]:
    """Train the proposed model (VARIANTS[variant]) on split.train_runs (early stopping on split.val_runs) and predict
    the test runs. Class weights from the TRAIN labelled particles. Materials unseen in training get delta = 0
    (fill_unseen_eta); when no training material shares their magnetic class, their debris volumes (and physics-head
    rates) are NaN with note 'eta not transferable'. Test samples / runs without surviving particles (absent from the
    loaders) get NaN / -1 predictions (runs: coverage 0, note 'no observed sample'). out_dir: history.json, best.pt.
    Returns full prediction tables {"particles", "samples", "runs"}."""
    c = variant_config(cfg, variant)
    dev, filt, log = _device(cfg, device), split.meta.get("image_filter"), get_logger()
    seed_everything(int(seed))
    scalers = fit_scalers(ds, cache, split.train_runs, c, filt)
    model = MechanismLinkedModel(c, model_dims(ds, cache, c, scalers))
    train = make_run_loader(ds, cache, split.train_runs, c, scalers, True, seed, filt, "train")
    val = (make_run_loader(ds, cache, split.val_runs, c, scalers, False, seed, filt, "train")
           if _role_pids(ds, cache, split.val_runs, filt, "train") else None)
    weights = class_weights(ds, _role_pids(ds, cache, split.train_runs, filt, "train"), c)
    fit = fit_model(model, train, val, c, dev, out_dir, weights)
    log.info("wearpg/%s %s seed %d: %d epochs (best %d) in %.0f s", variant, split.name, seed, fit["epochs_run"],
             fit["best_epoch"], fit["seconds"])

    mat, mag, rt = material_index(ds), magnetic_index(ds), ds.run_table().set_index("run_id")
    seen = sorted({mat[m] for m in rt.loc[split.train_runs, "material_id"]})
    model.fill_unseen_eta(seen)
    test = (make_run_loader(ds, cache, split.test_runs, c, scalers, False, seed, filt, "test")
            if _role_pids(ds, cache, split.test_runs, filt, "test") else [])
    pred = predict_runs(model, test, dev)

    test_samples = ds.samples.loc[ds.samples.run_id.isin(set(split.test_runs)), "sample_id"]
    smp = pred["samples"].set_index("sample_id").reindex(test_samples).reset_index()
    smp["severity_pred"] = smp.severity_pred.fillna(-1).astype(np.int64)
    runs = pred["runs"].set_index("run_id").reindex(list(split.test_runs)).reset_index()
    runs["note"] = np.where(runs.coverage.isna(), NOTE_EMPTY, "")
    runs["coverage"] = runs.coverage.fillna(0.0)
    if (runs.note == NOTE_EMPTY).any():
        log.warning("wearpg/%s %s: %d test runs have no surviving particle (NaN predictions)", variant, split.name,
                    int((runs.note == NOTE_EMPTY).sum()))
    by_idx = {i: mag[m] for m, i in mat.items()}
    blocked = [r for r in split.test_runs if not model.eta_transferable(mat[rt.at[r, "material_id"]], seen, by_idx)]
    if blocked:
        b_s = smp.sample_id.map(ds.samples.set_index("sample_id").run_id).isin(blocked).to_numpy()
        b_r = runs.run_id.isin(blocked).to_numpy()
        cols = ["log_debris_volume_mm3"] + (["log10_rate_pred"] if c["model"]["physics_head"] else [])
        smp.loc[b_s, cols] = np.nan
        runs.loc[b_r, cols] = np.nan
        runs.loc[b_r, "note"] = NOTE_ETA
    return _finish({"particles": pred["particles"], "samples": smp, "runs": runs}, ds, cfg, split, seed, "wearpg",
                   variant)


def run_particle_baseline(ds: WearDataset, cache: FeatureCache, cfg: dict, split: Split, seed: int, name: str,
                          device=None) -> pd.DataFrame:
    """Particle classifier baseline `name` (classical.CLASSICAL_BASELINES or deep.DEEP_BASELINES) trained on the train
    runs' surviving particles, tuned / early-stopped on the val runs (role "train"), predicting the test-role particles.
    Classical: descriptor rows (multiview_svm: per-view rows) with label indices; deep: ParticleBatch loaders with the
    model's globals_kind (fecnn -> "fecnn", else "geometry"), the geometric view-0 graph dims and the TRAIN class
    weights. Returns the full particle prediction table (variant "default")."""
    filt = split.meta.get("image_filter")
    tr, va = (_role_pids(ds, cache, r, filt, "train") for r in (split.train_runs, split.val_runs))
    te = _role_pids(ds, cache, split.test_runs, filt, "test")
    seed_everything(int(seed))
    if name in CLASSICAL_BASELINES:
        y = pd.Series(_mech_targets(ds, cache.particle_ids, cfg), index=cache.particle_ids)
        clf = make_particle_baseline(name, len(MECHANISMS), int(seed), cfg)
        ids, proba = te, np.zeros((0, len(MECHANISMS)))  # sklearn rejects 0-row inputs
        if name == "multiview_svm":
            vd = cache.view_descriptors
            views = lambda pids: vd[vd.particle_id.isin(set(pids))]  # noqa: E731
            clf.fit(views(tr), y[tr], views(va) if va else None, y[va] if va else None)
            proba = clf.predict_proba(views(te), te) if te else proba
        else:
            d = cache.descriptors
            clf.fit(d.loc[tr], y[tr].to_numpy(), d.loc[va] if va else None, y[va].to_numpy() if va else None)
            proba = clf.predict_proba(d.loc[te]) if te else proba
    else:
        if name not in DEEP_BASELINES:
            raise ValueError(f"unknown particle baseline {name!r}")
        kind, dev = ("fecnn" if name == "fecnn" else "geometry"), _device(cfg, device)
        scalers = fit_scalers(ds, cache, split.train_runs, cfg, filt)
        dims = replace(model_dims(ds, cache, cfg, scalers),
                       node_dim=selected_dims(cache.graph_cfg, PARTICLE_GRAPH_SELECTION)[0],
                       particle_feat_dim=len(baseline_global_columns(cache.descriptors.columns, kind)))
        model = make_deep_baseline(name, dims, cfg)
        loader = lambda pids, train: make_particle_loader(  # noqa: E731
            ds, cache, pids, cfg, scalers, train, int(seed), globals_kind=kind)
        fit = train_particle_classifier(model, loader(tr, True), loader(va, False) if va else None, cfg, dev,
                                        class_weights(ds, tr, cfg))
        get_logger().info("%s %s seed %d: %d epochs, best %d (val loss %.4g)", name, split.name, seed,
                          len(fit["history"]), fit["best_epoch"], fit["best_val_loss"])
        ids, proba = predict_particle_proba(model, loader(te, False), dev)
    proba = np.asarray(proba, dtype=float).reshape(len(ids), len(MECHANISMS))
    values = pd.DataFrame({"particle_id": list(ids), "y_pred": proba.argmax(1),
                           **dict(zip(prob_columns(), proba.T)), "log_volume_pred": np.nan})
    return _finish({"particles": values}, ds, cfg, split, seed, name, "default")["particles"]


def run_run_baseline(ds: WearDataset, cache: FeatureCache, cfg: dict, split: Split, seed: int,
                     name: str) -> dict[str, pd.DataFrame]:
    """Run-level baseline `name` (run_level.RUN_BASELINES): fit on the train runs with the descriptors of the train +
    val runs' surviving particles (val runs for tuning), predict the test runs from the test-role surviving particles.
    Returns full prediction tables {"samples", "runs"} (variant "default")."""
    filt, d = split.meta.get("image_filter"), cache.descriptors
    model = make_run_baseline(name, cfg, int(seed))
    model.fit(ds.subset(split.train_runs), d.loc[_role_pids(ds, cache, split.train_runs + split.val_runs, filt, "train")],
              ds.subset(split.val_runs) if split.val_runs else None)
    samples, runs = model.predict(ds.subset(split.test_runs), d.loc[_role_pids(ds, cache, split.test_runs, filt, "test")])
    return _finish({"samples": samples, "runs": runs}, ds, cfg, split, seed, name, "default")


# ============================================================================ protocols


def _run_split(ds: WearDataset, cache: FeatureCache, cfg: dict, split: Split, seed: int, models, variants, device,
               out_dir) -> None:
    """Every model (wearpg: every variant) on one split x seed; a model whose parts all exist is skipped (resume).
    wearpg ablation variants and particle baselines run only for the seeds in eval.secondary_seeds (null = every seed);
    wearpg/full and the run baselines run for every seed."""
    out_dir = Path(out_dir)
    sec = cfg["eval"].get("secondary_seeds")
    primary_only = sec is not None and int(seed) not in [int(x) for x in sec]

    def todo(model, variant, levels) -> bool:
        return not all(_part_path(out_dir, lvl, split.protocol, model, variant, split.name, seed).exists()
                       for lvl in levels)

    def write(model, variant, tables) -> None:
        for lvl, df in tables.items():  # runs last: a crash before it leaves the set incomplete -> recomputed
            _write_part(df, _part_path(out_dir, lvl, split.protocol, model, variant, split.name, seed))

    for name in models:
        if name == "wearpg":
            for v in variants:
                if v != PROPOSED[1] and primary_only:
                    continue
                if todo(name, v, LEVEL_COLUMNS):
                    train_dir = out_dir / "training" / name / safe_name(v) / f"{safe_name(split.name)}_seed{seed}"
                    write(name, v, run_proposed(ds, cache, cfg, split, seed, v, device, train_dir))
        elif name in RUN_BASELINES:
            if todo(name, "default", ("samples", "runs")):
                write(name, "default", run_run_baseline(ds, cache, cfg, split, seed, name))
        elif not primary_only and todo(name, "default", ("particles",)):
            write(name, "default", {"particles": run_particle_baseline(ds, cache, cfg, split, seed, name, device)})


def _check_resume(out: Path, cfg: dict) -> None:
    """Parts are resumed by name only: refuse an output dir whose parts were made under another config."""
    old = out / "config.yaml"
    if old.exists() and any((out / "predictions" / "parts").glob("*.parquet")):
        prev, new = yaml.safe_load(old.read_text(encoding="utf-8")).get("config_hash"), stable_hash(cfg)
        if prev != new:
            raise ValueError(f"{out} holds predictions of another config (config_hash {prev} != {new}); use a new "
                             "paths.output_dir or delete the old results")


def _start(cfg: dict) -> tuple[Path, WearDataset, FeatureCache]:
    """Output dir with config.yaml (+ config_hash) and dataset_meta.json; the prepared dataset and cache."""
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    _check_resume(out, cfg)
    ds, cache = prepare(cfg)
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "config_hash": stable_hash(cfg)}, sort_keys=False),
                                     encoding="utf-8")
    save_json({**ds.meta, "describe": ds.describe(), "feature_cache": cache.meta}, out / "dataset_meta.json")
    return out, ds, cache


def _execute(ds, cache, cfg, out: Path, splits: list[Split], models, variants, device) -> Path:
    """Validate names, merge the splits into splits.json, audit leakage, then every split x eval.seeds."""
    known = {"wearpg", *CLASSICAL_BASELINES, *DEEP_BASELINES, *RUN_BASELINES}
    bad = [m for m in models if m not in known] + [v for v in variants if v not in VARIANTS]
    if bad:
        raise ValueError(f"unknown models / variants: {bad}")
    path = out / "splits.json"
    old = {s.name: s for s in load_splits(path)} if path.exists() else {}
    save_splits(list({**old, **{s.name: s for s in splits}}.values()), path)
    for split in splits:
        leaks = audit_leakage(ds, split)
        if leaks:
            raise ValueError(f"split {split.name} leaks: {leaks[:3]}")
        for seed in cfg["eval"]["seeds"]:
            _run_split(ds, cache, cfg, split, int(seed), models, variants, device, out)
    return out


def run_cv(cfg: dict, models=None, variants=None, device=None) -> Path:
    """Grouped k-fold (make_splits) x eval.seeds; default models = wearpg + baselines.particle + baselines.run,
    variants = full + ablations. Returns paths.output_dir."""
    out, ds, cache = _start(cfg)
    models = list(models or ["wearpg", *cfg["baselines"]["particle"], *cfg["baselines"]["run"]])
    variants = list(variants or ["full", *cfg.get("ablations", [])])
    return _execute(ds, cache, cfg, out, make_splits(ds, cfg), models, variants, device)


def run_external(cfg: dict, by=None, models=None, variants=None, device=None) -> Path:
    """Leave-one-group-out (protocol external) by `by` (default eval.external_by; material_id falls back to load_N on a
    single-material dataset); default models = wearpg + eval.shift_baselines + baselines.run, variants =
    eval.shift_variants. Per-held-out-group metrics: evaluate_results (per_group, group_col "fold")."""
    out, ds, cache = _start(cfg)
    e = cfg["eval"]
    by = by or e["external_by"]
    if by == "material_id" and ds.specimens.material_id.nunique() < 2:
        by = "load_N"
    splits = leave_one_group_out(ds, by, int(cfg["seed"]), float(e["val_fraction"]))
    models = list(models or ["wearpg", *e["shift_baselines"], *cfg["baselines"]["run"]])
    return _execute(ds, cache, cfg, out, splits, models, list(variants or e["shift_variants"]), device)


def run_transfer(cfg: dict, by=None, train_values=None, test_values=None, models=None, variants=None,
                 device=None) -> Path:
    """Domain transfer (protocol transfer): the given (by, train_values, test_values) or every eval.transfer entry;
    an entry whose domains the dataset lacks is skipped (logged). Defaults as run_external."""
    out, ds, cache = _start(cfg)
    e = cfg["eval"]
    entries = [{"by": by, "train": train_values, "test": test_values}] if by else (e.get("transfer") or [])
    splits = []
    for t in entries:
        try:
            splits.append(transfer_split(ds, t["by"], t["train"], t["test"], e["split_level"], int(cfg["seed"]),
                                         val_fraction=float(e["val_fraction"])))
        except ValueError as err:
            get_logger().warning("run_transfer: entry %s skipped: %s", t, err)
    models = list(models or ["wearpg", *e["shift_baselines"], *cfg["baselines"]["run"]])
    return _execute(ds, cache, cfg, out, splits, models, list(variants or e["shift_variants"]), device)


# ============================================================================ evaluation


def _particle_metrics(gt: bool, ece_bins: int):
    """Flat metric dict of particle rows (one bootstrap loop for all): _CLS vs y_true, '<class>|<sensitivity|
    specificity|f1>', and _CLS + '_gt' vs y_true_gt on synthetic data."""
    cols = prob_columns()

    def fn(d: pd.DataFrame) -> dict:
        p = d[cols].to_numpy(float)
        m = classification_metrics(d.y_true.to_numpy(), p, ece_bins=ece_bins)
        out = {k: m[k] for k in _CLS}
        out.update({f"{c}|{k}": v[k] for c, v in m["per_class"].items() for k in _PER_CLASS})
        if gt:
            g = classification_metrics(d.y_true_gt.to_numpy(), p, ece_bins=ece_bins)
            out.update({f"{k}_gt": g[k] for k in _CLS})
        return out
    return fn


def _regression(d: pd.DataFrame, gt: bool) -> dict:
    """_REG of log10 k vs the measured target (+ '_gt' vs the synthetic-truth target); conditions = groups."""
    pred, groups = d.log10_rate_pred.to_numpy(float), d.condition_id.to_numpy()
    out = {k: v for k, v in regression_metrics(d.log10_rate_true.to_numpy(float), pred, groups).items() if k in _REG}
    if gt:
        g = regression_metrics(d.log10_rate_true_gt.to_numpy(float), pred, groups)
        out.update({f"{k}_gt": v for k, v in g.items() if k in _REG})
    return out


def _sample_metrics(gt: bool):
    def fn(d: pd.DataFrame) -> dict:
        p = d[list(SEVERITY_PROB_COLUMNS)].to_numpy(float)
        ok = np.isfinite(p).all(1)  # samples a model did not predict
        m = severity_metrics(d.severity_true.to_numpy()[ok], p[ok])
        return {**{k: m[k] for k in _SEV}, **_regression(d, gt)}
    return fn


def _run_metrics(gt: bool):
    return lambda d: _regression(d, gt)


def _with_gt(df: pd.DataFrame, ds: WearDataset, level: str, cfg: dict) -> pd.DataFrame:
    """Test rows; samples / runs get log10_rate_true_gt from the SYNTHETIC true interval volumes (runs: their sum over
    the run, under the same coverage rule as the measured target; NaN on real data)."""
    df = df[df.split == "test"].reset_index(drop=True)
    if level == "particles":
        return df
    rt, s = ds.runs.set_index("run_id"), ds.samples.set_index("sample_id")
    if level == "samples":
        k = s.true_interval_wear_volume_mm3 / (s.run_id.map(rt.load_N) * (s.sliding_distance_end_m - s.sliding_distance_start_m))
        return df.assign(log10_rate_true_gt=_log10(df.sample_id.map(k)))
    v = s.groupby("run_id").true_interval_wear_volume_mm3.sum(min_count=1)
    k = _log10(df.run_id.map(v / (rt.load_N * rt.sliding_distance_m)))
    return df.assign(log10_rate_true_gt=np.where(df.coverage.to_numpy(float) >= _min_cov(cfg), k, np.nan))


def _counts(d: pd.DataFrame, level: str) -> dict[str, int]:
    """Number of distinct scored items behind each metric family ("" measured, "_gt", "sev", "<class>|")."""
    def n(mask) -> int:
        return int(d.loc[np.asarray(mask, dtype=bool), _KEY[level]].nunique())

    if level == "particles":
        return {"": n(d.y_true >= 0), "_gt": n(d.y_true_gt >= 0),
                **{f"{c}|": n(d.y_true == i) for i, c in enumerate(MECHANISMS)}}
    pred = np.isfinite(d.log10_rate_pred.to_numpy(float))
    out = {"": n(pred & np.isfinite(d.log10_rate_true.to_numpy(float))),
           "_gt": n(pred & np.isfinite(d.log10_rate_true_gt.to_numpy(float)))}
    if level == "samples":
        out["sev"] = n((d.severity_true >= 0) & np.isfinite(d[list(SEVERITY_PROB_COLUMNS)].to_numpy(float)).all(1))
    return out


def _group_metrics(d: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """metrics_by_group per training seed, averaged over seeds (n = max per-seed count)."""
    cols = prob_columns()
    frames = [metrics_by_group(g.y_true.to_numpy(), g[cols].to_numpy(float), groups.loc[g.index].to_numpy())
              for _, g in d.groupby("seed")]
    return (pd.concat(frames).groupby(["group", "metric"], sort=True)
            .agg(value=("value", "mean"), n=("n", "max")).reset_index())


def _both_scored(a: pd.DataFrame, b: pd.DataFrame, level: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of the items both models predicted (samples: severity probabilities, runs: log10 k), so a paired
    comparison never sets one model's excluded runs (e.g. 'eta not transferable') against the other's."""
    col = {"samples": SEVERITY_PROB_COLUMNS[0], "runs": "log10_rate_pred"}.get(level)
    if col is None:
        return a, b
    k = _KEY[level]
    keys = set(a.loc[np.isfinite(a[col].to_numpy(float)), k]) & set(b.loc[np.isfinite(b[col].to_numpy(float)), k])
    return a[a[k].isin(keys)], b[b[k].isin(keys)]


def _seed_mean(vals: list[dict], keys, ints=()) -> dict:
    """Per-seed result dicts -> mean over seeds per key (ints rounded)."""
    out = {k: float(np.mean([v[k] for v in vals])) for k in keys}
    return {k: int(round(v)) if k in ints else v for k, v in out.items()}


def evaluate_results(results_dir, cfg: dict) -> dict:
    """Metrics of every test prediction part -> results_dir/metrics/*.csv (DESIGN sec. 11 columns) + summary.json.

    Per (protocol, model, variant, level) ONE cluster bootstrap (clusters = eval.split_level units, training seeds
    resampled, eval.n_bootstrap, eval.ci) shared by all its metrics; seed_mean / seed_sd over the per-seed metrics
    (predictions are never seed-averaged). Run rows were already restricted by attach_hierarchy (coverage rule; 'eta
    not transferable' rows have NaN predictions): excluded runs are counted in metrics/exclusions.csv. Comparisons
    (paired cluster bootstrap on the items both models predicted): families baselines (wearpg/full vs every baseline at the same level), ablations
    (wearpg/full vs each variant), transfer (full minus no_fractal, every protocol); p-values only for the claim metric of a
    level (particle macro_f1, sample qwk, run mae; eval.claim_metrics overrides), adjusted with eval.multiplicity within
    (protocol, level, family, metric). per_group: eval.group_metrics_by columns, fold (external / transfer; runs as
    run_<metric>) and source_subset {multiview (n_views > 1), eds (has_eds)}. '*_gt' rows (synthetic data only) use the
    synthetic truth. summary.json: {"config_hash", protocol: {"model|variant": {"<level>.<metric>": {point, lo, hi,
    seed_mean, seed_sd}}}} with level particle | sample | run. Returns the summary.
    """
    out, e, log = Path(results_dir), cfg["eval"], get_logger()
    ds = WearDataset.load(cfg["paths"]["data_root"])
    gt, cluster = ds.is_synthetic, f"{e['split_level']}_id"
    boot = {"n_boot": int(e["n_bootstrap"]), "ci": float(e["ci"]), "seed": int(cfg["seed"])}
    preds = {lvl: _with_gt(load_predictions(out, lvl), ds, lvl, cfg) for lvl in LEVEL_COLUMNS}
    fns = {"particles": _particle_metrics(gt, int(e["ece_bins"])), "samples": _sample_metrics(gt),
           "runs": _run_metrics(gt)}
    tab = {k: [] for k in METRIC_COLUMNS}
    summary: dict = {"config_hash": stable_hash(cfg)}

    cols = {lvl: [*_METRIC_INPUTS[lvl], cluster, "seed", _KEY[lvl]] for lvl in LEVEL_COLUMNS}
    for lvl, df in preds.items():  # point estimates and CIs (bootstrap frames trimmed to the metric inputs: speed)
        for (proto, model, variant), d in df.groupby(_GROUP, sort=True):
            res = cluster_bootstrap(fns[lvl], d[cols[lvl]], cluster, "seed", **boot)
            per_seed = [fns[lvl](g) for _, g in d.groupby("seed")]
            counts, key = _counts(d, lvl), {"protocol": proto, "model": model, "variant": variant}
            for metric, r in res.items():
                if metric == "n_clusters":
                    continue
                ss = seed_summary([p[metric] for p in per_seed])
                vals = {"point": r["point"], "lo": r["lo"], "hi": r["hi"], "seed_mean": ss["mean"], "seed_sd": ss["sd"]}
                summary.setdefault(proto, {}).setdefault(f"{model}|{variant}", {})[f"{LEVEL_NAME[lvl]}.{metric}"] = vals
                if "|" in metric:
                    cls, m = metric.split("|")
                    tab["per_class"].append({**key, "class": cls, "metric": m, "point": r["point"], "lo": r["lo"],
                                             "hi": r["hi"], "n": counts[f"{cls}|"]})
                elif lvl == "particles":
                    tab["classification"].append({**key, "metric": metric, **vals,
                                                  "n": counts["_gt" if metric.endswith("_gt") else ""]})
                elif lvl == "samples" and metric in _SEV:
                    tab["severity"].append({**key, "metric": metric, **vals, "n": counts["sev"]})
                else:
                    tab["regression"].append({**key, "level": LEVEL_NAME[lvl], "metric": metric, **vals,
                                              "n": counts["_gt" if metric.endswith("_gt") else ""]})

    claim = {"particles": "macro_f1", "samples": "qwk", "runs": "mae",
             **{lvl: v["metric"] for lvl, v in (e.get("claim_metrics") or {}).items()}}
    method = f"paired cluster bootstrap, {e['multiplicity']} within (protocol, level, family, metric)"
    for lvl, df in preds.items():  # comparisons
        for proto, dp in df.groupby("protocol", sort=True):
            groups = dict(tuple(dp.groupby(["model", "variant"], sort=True)))
            if PROPOSED not in groups:
                continue
            for (mb, vb), b in groups.items():
                fam = ["baselines"] if mb != PROPOSED[0] else ["ablations"] if vb != PROPOSED[1] else []
                fam += ["transfer"] if (mb, vb) == (PROPOSED[0], "no_fractal") else []
                if not fam:
                    continue
                try:
                    a2, b2 = _both_scored(groups[PROPOSED][cols[lvl]], b[cols[lvl]], lvl)
                    r = paired_cluster_bootstrap(fns[lvl], a2, b2, cluster, _KEY[lvl], "seed", **boot)
                except ValueError as err:  # no common items
                    log.warning("evaluate_results: %s %s %s/%s not compared: %s", proto, lvl, mb, vb, err)
                    continue
                for metric, v in r.items():
                    if metric == "n_clusters" or "|" in metric:
                        continue
                    for f in fam:
                        tab["comparisons"].append({
                            "protocol": proto, "level": LEVEL_NAME[lvl], "family": f, "metric": metric,
                            "model_a": PROPOSED[0], "variant_a": PROPOSED[1], "model_b": mb, "variant_b": vb,
                            "diff": v["diff"], "lo": v["lo"], "hi": v["hi"], "rel_diff": v["rel_diff"],
                            "p_value": v["p_value"] if metric == claim.get(lvl) else np.nan, "p_adj": np.nan,
                            "method": method})
    comp = pd.DataFrame(tab["comparisons"], columns=METRIC_COLUMNS["comparisons"])
    if len(comp):
        comp["p_adj"] = comp.groupby(["protocol", "level", "family", "metric"]).p_value.transform(
            lambda p: adjust(p.to_numpy(float), e["multiplicity"]))

    part = preds["particles"]
    flags = ds.particles.set_index("particle_id")
    part = part.assign(n_views=part.particle_id.map(flags.n_views), has_eds=part.particle_id.map(flags.has_eds))
    for (proto, model, variant), d in part[part.y_true >= 0].groupby(_GROUP, sort=True):  # per group, confusion
        key = {"protocol": proto, "model": model, "variant": variant}
        specs = [(c, d[c]) for c in e.get("group_metrics_by") or [] if c in d.columns]
        specs += [("fold", d.fold)] if proto != "cv" else []
        subsets = [(n, m) for n, m in (("multiview", d.n_views > 1), ("eds", d.has_eds.astype(bool))) if m.any()]
        for col, grp in specs + [("source_subset", pd.Series(n, index=d.index[m])) for n, m in subsets]:
            for r in _group_metrics(d.loc[grp.index], grp).itertuples():
                tab["per_group"].append({**key, "group_col": col, "group": r.group, "metric": r.metric,
                                         "value": r.value, "n": int(r.n)})
        for col, grp in [("all", pd.Series("all", index=d.index))] + specs:
            cnt = d.groupby([grp.astype(str).to_numpy(), d.y_true.to_numpy(), d.y_pred.to_numpy()]).size()
            tab["confusion_by_group"] += [{**key, "group_col": col, "group": g, "true": MECHANISMS[t],
                                           "pred": MECHANISMS[p], "count": int(n)} for (g, t, p), n in cnt.items()]

    runs = preds["runs"]
    for (proto, model, variant), d in runs[runs.protocol != "cv"].groupby(_GROUP, sort=True):  # per held-out group
        r = pd.DataFrame([{"group": f, **regression_metrics(g.log10_rate_true, g.log10_rate_pred)}
                          for (f, _), g in d.groupby(["fold", "seed"])]).groupby("group").mean()
        tab["per_group"] += [{"protocol": proto, "model": model, "variant": variant, "group_col": "fold", "group": f,
                              "metric": f"run_{m}", "value": row[m], "n": int(round(row["n"]))}
                             for f, row in r.iterrows() for m in ("mae", "rmse", "bias", "spearman")]

    rmeta = ds.runs[["run_id", "duration_s"]].assign(transition_time_true=ds.runs.transition_time_s)
    t_keys = METRIC_COLUMNS["transition"][3:]
    for (proto, model, variant), d in preds["samples"].groupby(_GROUP, sort=True):
        d = d[np.isfinite(d[list(SEVERITY_PROB_COLUMNS)].to_numpy(float)).all(1)]
        if d.empty:
            continue
        vals = [transition_timing(g.assign(p_severe=g.sev_p_transition + g.sev_p_severe), rmeta,
                                  float(e["transition_threshold"])) for _, g in d.groupby("seed")]
        tab["transition"].append({"protocol": proto, "model": model, "variant": variant,
                                  **_seed_mean(vals, t_keys, ("n_transition_runs", "n_stable_runs"))})

    v_keys = METRIC_COLUMNS["volume_consistency"][3:]
    for (proto, model, variant), d in runs.groupby(_GROUP, sort=True):
        key = {"protocol": proto, "model": model, "variant": variant}
        cov = d.coverage.to_numpy(float)
        for reason, m in ((NOTE_ETA, d.note == NOTE_ETA), (NOTE_EMPTY, ~(cov > 0)),
                          ("coverage below min", (cov > 0) & ~(cov >= _min_cov(cfg)))):
            tab["exclusions"].append({**key, "reason": reason, "n_runs": int(d.run_id[np.asarray(m)].nunique())})
        if np.isfinite(d.log_debris_volume_mm3.to_numpy(float)).any():
            vals = [volume_consistency(g.log_debris_volume_mm3, g.log_grav_volume_mm3) for _, g in d.groupby("seed")]
            tab["volume_consistency"].append({**key, **_seed_mean(vals, v_keys, ("n",))})

    mdir = out / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    for name, cols in METRIC_COLUMNS.items():
        frame = comp if name == "comparisons" else pd.DataFrame(tab[name], columns=cols)
        frame.to_csv(mdir / f"{name}.csv", index=False, lineterminator="\n")
    save_json(summary, mdir / "summary.json")
    log.info("evaluate_results: metrics written to %s", mdir)
    return summary


# ============================================================================ smoke


def _gates(ds: WearDataset, cache: FeatureCache, out: Path, cfg: dict) -> list:
    """gates.evaluate_gates on the test predictions of every protocol -> out/gates/gates.{json,md}."""
    res = {lvl: load_predictions(out, lvl) for lvl in LEVEL_COLUMNS}
    gates = evaluate_gates(ds, cache.descriptors, {k: v[v.split == "test"] for k, v in res.items()}, cfg)
    write_gate_report(gates, out / "gates", ds.is_synthetic)
    return gates


def _fmt(m: dict | None) -> str:
    return "-" if m is None else f"{m['point']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def _verdict(passed: bool, wrong_side: bool) -> str:
    """DESIGN sec. 15: criterion met -> for_us; point estimate on the wrong side -> against_us; else inconclusive."""
    return "for_us" if passed else "against_us" if wrong_side else "inconclusive"


def claim_verdicts(results_dir, cfg: dict, gates: list) -> pd.DataFrame:
    """Pre-specified verdicts of the paper claims (FRAMING sec. 3, criteria in DESIGN sec. 15) from the cv test results:
    C1 (wear_rate gate a + b), C1b (gate c), C2 (wearpg/full vs physics_only and no_mechanism_link, run MAE), C3
    (graph_value gate), C4 (transition timing of wearpg/full) and H-F (physics_conditions vs physics_only and
    conditions_only + within-condition Spearman; with_conditions likewise vs wearpg/full and conditions_only).
    Writes metrics/claims.csv (claim, criterion, evidence, verdict); a claim without results is "not evaluable"."""
    out, e = Path(results_dir), cfg["eval"]
    g = {r.name: r for r in gates}
    reg = pd.read_csv(out / "metrics" / "regression.csv")
    tr = pd.read_csv(out / "metrics" / "transition.csv")
    runs = load_predictions(out, "runs")
    runs = runs[(runs.protocol == "cv") & (runs.split == "test")]
    boot = {"n_boot": int(e["n_bootstrap"]), "ci": float(e["ci"]), "seed": int(cfg["seed"])}
    rows = []

    def add(claim, criterion, evidence, verdict):
        rows.append({"claim": claim, "criterion": criterion, "evidence": evidence, "verdict": verdict})

    def mae_diff(a, b):
        """Paired run-MAE difference a - b ((model, variant) pairs) on the runs both scored, or None."""
        da = runs[(runs.model == a[0]) & (runs.variant == a[1])]
        db = runs[(runs.model == b[0]) & (runs.variant == b[1])]
        da, db = (d[np.isfinite(d.log10_rate_true.to_numpy(float))] for d in _both_scored(da, db, "runs"))
        if da.empty or db.empty:
            return None
        fn = lambda d: regression_metrics(d.log10_rate_true, d.log10_rate_pred)["mae"]  # noqa: E731
        return paired_cluster_bootstrap(fn, da, db, f"{e['split_level']}_id", "run_id", "seed", **boot)["value"]

    def within(model, variant):
        r = reg[(reg.protocol == "cv") & (reg.level == "run") & (reg.model == model) & (reg.variant == variant)
                & (reg.metric == "within_condition_spearman")]
        return None if r.empty else r.iloc[0]

    def fmt(d):
        return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"

    def ci(d):
        return f"{d.get('point', np.nan):.3f} [{d.get('lo', np.nan):.3f}, {d.get('hi', np.nan):.3f}]"

    w = g.get("wear_rate")
    if w is None or not w.evidence:
        add("C1", "wear_rate gate (a) + (b)", "no results", "not evaluable")
        add("C1b", "wear_rate gate (c)", "no results", "not evaluable")
    else:
        mae, vol = w.evidence.get("mae", {}), w.evidence.get("volume", {})
        slope, ccc = vol.get("slope") or {}, vol.get("ccc") or {}
        ok = [m.get("passes") for m in mae.values()] + [vol.get("passes")]
        wrong = (any(m["diff"] >= 0 for m in mae.values() if m.get("available"))
                 or (bool(slope) and not slope["lo"] <= 1 <= slope["hi"])
                 or ccc.get("point", 1.0) < float(cfg["gates"]["min_volume_ccc"]))
        ev = "; ".join(f"MAE diff vs {b} {fmt(m)}" for b, m in mae.items() if m.get("available"))
        add("C1", "run MAE CI < 0 vs conditions_only and physics_only; slope CI contains 1; CCC >= min_volume_ccc",
            f"{ev}; slope {ci(slope)}; CCC {ci(ccc)}", _verdict(all(x is True for x in ok), wrong))
        wc = w.evidence.get("within_condition_spearman", {})
        add("C1b", "within-condition Spearman lower CI bound > 0", ci(wc),
            _verdict(wc.get("passes") is True, not wc.get("point", np.nan) > 0))
    d = {b: mae_diff(PROPOSED, b) for b in (("physics_only", "default"), (PROPOSED[0], "no_mechanism_link"))}
    if all(v is not None for v in d.values()):
        add("C2", "wearpg/full run MAE CI < 0 vs physics_only and vs no_mechanism_link",
            "; ".join(f"vs {'/'.join(b)} {fmt(v)}" for b, v in d.items()),
            _verdict(all(v["hi"] < 0 for v in d.values()), any(v["diff"] >= 0 for v in d.values())))
    else:
        add("C2", "wearpg/full run MAE CI < 0 vs physics_only and vs no_mechanism_link", "missing predictions",
            "not evaluable")
    gv = g.get("graph_value")
    ev = {} if gv is None else {b: x for b, x in gv.evidence.items() if x.get("available")}
    if ev:
        add("C3", "particle_gnn macro-F1 lower CI bound > 0 vs every graph-gate baseline",
            "; ".join(f"vs {b} {fmt(x)}" for b, x in ev.items()),
            _verdict(gv.passed is True, any(x["diff"] <= 0 for x in ev.values())))
    else:
        add("C3", "graph_value gate", "no results", "not evaluable")
    t = tr[(tr.protocol == "cv") & (tr.model == PROPOSED[0]) & (tr.variant == PROPOSED[1])]
    if len(t) and t.n_transition_runs.iloc[0] > 0:
        t = t.iloc[0]
        ok = bool(t.detection_rate >= 0.8 and t.false_alarm_rate <= 0.2 and t.mae_intervals <= 1.0)
        add("C4", "detection >= 0.8, false alarm <= 0.2, timing MAE <= 1 interval (point estimates)",
            f"detection {t.detection_rate:.2f}, false alarm {t.false_alarm_rate:.2f}, MAE {t.mae_intervals:.2f} "
            f"intervals (oracle floor {t.oracle_floor_s / t.interval_s:.2f}); {int(t.n_transition_runs)} transition / "
            f"{int(t.n_stable_runs)} stable runs", _verdict(ok, not ok))
    else:
        add("C4", "transition timing of wearpg/full", "no transition runs scored", "not evaluable")
    fusions = ((("physics_conditions", "default"), (("physics_only", "default"), ("conditions_only", "default")),
                "H-F (C1 support)"),
               ((PROPOSED[0], "with_conditions"), (PROPOSED, ("conditions_only", "default")),
                "H-F with_conditions (secondary)"))
    for fused, comps, name in fusions:
        d = {c: mae_diff(fused, c) for c in comps}
        wc = within(*fused)
        if wc is None or any(v is None for v in d.values()):
            add(name, "fusion beats both components; within-condition Spearman CI > 0", "missing predictions",
                "not evaluable")
            continue
        add(name, f"{'/'.join(fused)} run MAE CI < 0 vs both components; within-condition Spearman lower CI > 0",
            "; ".join(f"vs {'/'.join(c)} {fmt(v)}" for c, v in d.items())
            + f"; within-condition Spearman {wc.point:.3f} [{wc.lo:.3f}, {wc.hi:.3f}]",
            _verdict(all(v["hi"] < 0 for v in d.values()) and wc.lo > 0,
                     any(v["diff"] >= 0 for v in d.values()) or not wc.point > 0))
    df = pd.DataFrame(rows, columns=["claim", "criterion", "evidence", "verdict"])
    df.to_csv(out / "metrics" / "claims.csv", index=False, lineterminator="\n")
    return df


def _smoke_report(out: Path, cfg_path, cfg: dict, ds: WearDataset, summary: dict, gates: list, report: str,
                  seconds: float, claims: pd.DataFrame | None = None) -> Path:
    """<output_dir>/<cfg name>_report.md: headline table (primary endpoint eval.primary_metric first), gates, claims."""
    name = cfg.get("name", "smoke")
    lines = [f"# wearpg {name} report", ""]
    if ds.is_synthetic:
        lines += [f"**{SYNTHETIC_BANNER}**", "", f"**{VERIFICATION_NOTICE}** (SIMULATION results)", ""]
    lines += [f"- config: {Path(cfg_path).as_posix()} (config_hash {summary['config_hash']})",
              f"- dataset: {ds.meta.get('name', '')} {json.dumps(ds.describe())}",
              f"- runtime: {seconds:.0f} s (cv: {int(cfg['eval']['n_splits'])} folds x seeds {cfg['eval']['seeds']})",
              f"- results: {out.as_posix()} (metrics/, gates/, predictions/parts/)", f"- report: {report}",
              f"- primary endpoint (eval.primary_metric): {cfg['eval'].get('primary_metric')}", "",
              f"## Headline metrics, protocol cv (point [{100 * float(cfg['eval']['ci']):.0f}% CI])", "",
              "| model | variant | run mae log10 k (primary) | run within-cond. Spearman | particle macro_f1 | "
              "sample qwk |", "|---|---|---|---|---|---|"]
    for mv, m in sorted(summary.get("cv", {}).items()):
        model, variant = mv.split("|")
        lines.append(f"| {model} | {variant} | {_fmt(m.get('run.mae'))} | {_fmt(m.get('run.within_condition_spearman'))}"
                     f" | {_fmt(m.get('particle.macro_f1'))} | {_fmt(m.get('sample.qwk'))} |")
    lines += ["", "## Gates", "", "| gate | status | message |", "|---|---|---|"]
    lines += [f"| {g.name} | {_STATUS[g.passed]} | {g.message} |" for g in gates]
    if claims is not None and len(claims):
        lines += ["", "## Claim verdicts (paper/FRAMING.md sec. 3; criteria DESIGN sec. 15)"
                  + (" - SIMULATION, not evidence" if ds.is_synthetic else ""), "",
                  "| claim | criterion | evidence | verdict |", "|---|---|---|---|"]
        lines += [f"| {r.claim} | {r.criterion} | {r.evidence} | {r.verdict} |" for r in claims.itertuples()]
    path = out / f"{name}_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_smoke(cfg_path="configs/smoke.yaml", device=None) -> Path:
    """End to end: run_cv + evaluate_results + gates + claim_verdicts + <cfg name>_report.md (banner and VERIFICATION
    notice on synthetic data), then report.build_report when wearpg.report is importable (a build failure is written
    into the run report and re-raised). Returns the report path (smoke_report.md for configs/smoke.yaml)."""
    t0, log = time.perf_counter(), get_logger()
    cfg = load_config(cfg_path)
    out = run_cv(cfg, device=device)
    summary = evaluate_results(out, cfg)
    ds, cache = prepare(cfg)
    gates = _gates(ds, cache, out, cfg)
    failure, report = None, "not built: wearpg.report is not importable"
    try:
        from wearpg.report import build_report
    except ImportError as err:
        log.warning("run_smoke: report skipped (%s)", err)
    else:
        try:
            report = Path(build_report(out, cfg)).as_posix()
        except Exception as err:  # keep the smoke summary; the failure is re-raised below
            failure, report = err, f"FAILED: {err!r}"
    claims = claim_verdicts(out, cfg, gates)
    path = _smoke_report(out, cfg_path, cfg, ds, summary, gates, report, time.perf_counter() - t0, claims)
    log.info("run_smoke: %s (%.0f s)", path, time.perf_counter() - t0)
    if failure is not None:
        raise failure
    return path
