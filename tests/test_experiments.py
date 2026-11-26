import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wearpg import experiments as ex
from wearpg.baselines.run_level import NOTE_ETA
from wearpg.config import load_config
from wearpg.metrics import classification_metrics
from wearpg.schema import (PARTICLE_PRED_COLUMNS, RUN_PRED_COLUMNS, SAMPLE_PRED_COLUMNS, SEVERITY_PROB_COLUMNS,
                           prob_columns)
from wearpg.splits import Split

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(*overrides):
    return load_config(ROOT / "configs" / "default.yaml", list(overrides))


def split_of(test_runs, name="specimen_k2_f0", protocol="cv"):
    return Split(name, protocol, train_runs=[], val_runs=[], test_runs=sorted(test_runs), meta={})


def fake_tables(ds, cfg, split, seed, model, variant, levels, rng):
    """Random but schema-valid prediction tables of the split's test runs (through experiments._finish)."""
    test = set(split.test_runs)
    sids = ds.samples.loc[ds.samples.run_id.isin(test), "sample_id"].to_numpy()
    pids = ds.particles.loc[ds.particles.sample_id.isin(set(sids)), "particle_id"].to_numpy()
    vals = {}
    if "particles" in levels:
        p = rng.dirichlet(np.ones(7), len(pids))
        vals["particles"] = pd.DataFrame({"particle_id": pids, "y_pred": p.argmax(1), **dict(zip(prob_columns(), p.T)),
                                          "log_volume_pred": 0.0})
    if "samples" in levels:
        q = rng.dirichlet(np.ones(3), len(sids))
        vals["samples"] = pd.DataFrame({"sample_id": sids, "severity_pred": q.argmax(1),
                                        **dict(zip(SEVERITY_PROB_COLUMNS, q.T)),
                                        "log10_rate_pred": rng.normal(-5, 1, len(sids)),
                                        "log_debris_volume_mm3": rng.normal(0, 1, len(sids))})
    if "runs" in levels:
        r = split.test_runs
        vals["runs"] = pd.DataFrame({"run_id": r, "log10_rate_pred": rng.normal(-5, 1, len(r)),
                                     "log_debris_volume_mm3": rng.normal(-3, 1, len(r)), "coverage": 1.0, "note": ""})
    return ex._finish(vals, ds, cfg, split, seed, model, variant)


def test_variants_override_existing_config_keys():
    cfg = make_cfg()
    for name, overrides in ex.VARIANTS.items():
        for item in overrides:
            *path, last = item.split("=")[0].split(".")
            node = cfg
            for p in path:
                node = node[p]
            assert last in node, f"{name}: {item} does not hit an existing key"
    assert set(cfg["ablations"]) <= set(ex.VARIANTS)
    v = ex.variant_config(cfg, "no_physics")
    assert v["model"]["physics_head"] is False and v["loss"]["consistency"] == 0 and v["loss"]["monotone"] == 0
    assert ex.variant_config(cfg, "edges_ring")["model"]["edges"] == "ring"
    assert cfg["model"]["physics_head"] is True  # the base config is not mutated
    with pytest.raises(ValueError):
        ex.variant_config(cfg, "no_such_variant")


def test_attach_hierarchy_particles_and_samples(tiny_dataset):
    ds, cfg = tiny_dataset, make_cfg()
    split = split_of(sorted(ds.runs.run_id)[:2])
    t = fake_tables(ds, cfg, split, 3, "wearpg", "full", ("particles", "samples"), np.random.default_rng(0))
    part, smp = t["particles"], t["samples"]
    assert list(part.columns) == PARTICLE_PRED_COLUMNS and list(smp.columns) == SAMPLE_PRED_COLUMNS
    assert (part.seed == 3).all() and (part.split == "test").all() and (part.fold == split.name).all()
    h = ds.particle_hierarchy().set_index("particle_id").loc[part.particle_id]
    for c in ("sample_id", "run_id", "specimen_id", "material_id", "condition_id", "magnification"):
        assert (part[c].to_numpy() == h[c].to_numpy()).all(), c
    assert (part.y_true.to_numpy() == ds.label_indices("adjudicated").loc[part.particle_id].to_numpy()).all()
    assert (part.y_true_gt.to_numpy() == ds.label_indices("true").loc[part.particle_id].to_numpy()).all()
    assert part.log_volume_true.isna().all()  # no measured volumes in the tiny dataset

    s = ds.samples.set_index("sample_id").loc[smp.sample_id]
    load = ds.runs.set_index("run_id").load_N.loc[s.run_id].to_numpy()
    k = s.interval_wear_volume_mm3.to_numpy() / (load * (s.sliding_distance_end_m - s.sliding_distance_start_m).to_numpy())
    np.testing.assert_allclose(smp.log10_rate_true, np.log10(k))
    assert (smp.severity_true.to_numpy() == s.severity.to_numpy()).all()
    assert (smp.specimen_id.to_numpy() == ds.runs.set_index("run_id").specimen_id.loc[s.run_id].to_numpy()).all()

    bare = ex.attach_hierarchy(pd.DataFrame({"sample_id": smp.sample_id}).assign(protocol="cv", model="m",
                               variant="default", fold="f", seed=0, split="test"), ds, "samples", cfg)
    assert (bare.severity_pred == -1).all() and bare.log_debris_volume_mm3.isna().all()  # absent value columns
    with pytest.raises(ValueError):
        ex.attach_hierarchy(smp, ds, "images", cfg)


def test_attach_hierarchy_runs_transition_midpoint_and_coverage_rule(tiny_dataset):
    ds, cfg = tiny_dataset, make_cfg()
    runs = sorted(ds.runs.run_id)[:3]  # each run: two samples [0, 900) and [900, 1800) s
    p = {f"{runs[0]}_s0": 0.2, f"{runs[0]}_s1": 0.7,   # alarm in the 2nd interval -> 1350 s
         f"{runs[1]}_s0": 0.1, f"{runs[1]}_s1": 0.4,   # no alarm -> NaN
         f"{runs[2]}_s0": 0.9, f"{runs[2]}_s1": 0.9}   # alarm in the 1st interval -> 450 s
    v = np.array(list(p.values()))
    samples = pd.DataFrame({"sample_id": list(p), "sev_p_mild": 1 - v, "sev_p_transition": v / 2,
                            "sev_p_severe": v / 2})
    rdf = pd.DataFrame({"run_id": runs, "log10_rate_pred": -5.0, "log_debris_volume_mm3": 0.0,
                        "coverage": [1.0, 0.3, 0.9], "note": ""}).assign(protocol="cv", model="m", variant="default",
                                                                          fold="f", seed=0, split="test")
    out = ex.attach_hierarchy(rdf, ds, "runs", cfg, samples=samples)
    assert list(out.columns) == RUN_PRED_COLUMNS
    np.testing.assert_allclose(out.transition_time_pred.to_numpy(), [1350.0, np.nan, 450.0])
    rt = ds.run_table().set_index("run_id").loc[runs]
    np.testing.assert_allclose(out.transition_time_true.to_numpy(), rt.transition_time_s.to_numpy())
    # coverage 0.3 < physics.min_run_coverage (0.5): run targets NaN, everything else kept
    np.testing.assert_allclose(out.log10_rate_true[[0, 2]], np.log10(rt.wear_rate_mm3_Nm.to_numpy()[[0, 2]]))
    np.testing.assert_allclose(out.log_grav_volume_mm3[[0, 2]], np.log(rt.wear_volume_mm3.to_numpy()[[0, 2]]))
    assert np.isnan(out.log10_rate_true[1]) and np.isnan(out.log_grav_volume_mm3[1])
    assert (out.log10_rate_pred == -5.0).all()
    # transfer protocol: the same coverage rule (partial runs are coverage-corrected, DESIGN sec. 3)
    tr = ex.attach_hierarchy(rdf.assign(protocol="transfer"), ds, "runs", cfg, samples=samples)
    assert np.isfinite(tr.log10_rate_true[[0, 2]]).all() and np.isnan(tr.log10_rate_true[1])
    # NaN coverage (run absent from the loaders) is never scored; no sample predictions -> no transition prediction
    nan_cov = ex.attach_hierarchy(rdf.assign(coverage=np.nan), ds, "runs", cfg)
    assert nan_cov.log10_rate_true.isna().all() and nan_cov.transition_time_pred.isna().all()


def test_part_paths_are_safe_and_load_predictions_round_trips(tiny_dataset, tmp_path):
    ds, cfg = tiny_dataset, make_cfg()
    unsafe = "transfer_focus level/0.5:inf*?"
    p = ex._part_path(tmp_path, "particles", "transfer", "wearpg", "full", unsafe, 2)
    assert p.parent == tmp_path / "predictions" / "parts"
    assert re.fullmatch(r"[A-Za-z0-9._-]+", p.name)
    assert p.name.startswith("particles__transfer__wearpg__full__") and p.name.endswith("__seed2.parquet")
    assert ex.load_predictions(tmp_path, "particles").columns.tolist() == PARTICLE_PRED_COLUMNS  # nothing yet

    rng = np.random.default_rng(0)
    a = fake_tables(ds, cfg, split_of(sorted(ds.runs.run_id)[:2], unsafe, "transfer"), 2, "wearpg", "full",
                    ("particles",), rng)["particles"]
    b = fake_tables(ds, cfg, split_of(sorted(ds.runs.run_id)[2:3], "f1"), 0, "contour_cnn", "default",
                    ("particles",), rng)["particles"]
    ex._write_part(a, p)
    ex._write_part(b, ex._part_path(tmp_path, "particles", "cv", "contour_cnn", "default", "f1", 0))
    got = ex.load_predictions(tmp_path, "particles")
    assert len(got) == len(a) + len(b) and set(got.fold) == {unsafe, "f1"}  # the fold key keeps its real name
    assert got.columns.tolist() == PARTICLE_PRED_COLUMNS and not list(p.parent.glob("*.tmp"))
    assert ex.load_predictions(tmp_path, "runs").empty


def test_run_split_writes_every_part_and_resumes(tiny_dataset, tmp_path, monkeypatch):
    ds, cfg = tiny_dataset, make_cfg()
    split = split_of(sorted(ds.runs.run_id)[:4])
    calls = []

    def proposed(ds_, cache, cfg_, split_, seed, variant="full", device=None, out_dir=None):
        calls.append(variant)
        return fake_tables(ds_, cfg_, split_, seed, "wearpg", variant, ex.LEVEL_COLUMNS, np.random.default_rng(1))

    def particle(ds_, cache, cfg_, split_, seed, name, device=None):
        calls.append(name)
        return fake_tables(ds_, cfg_, split_, seed, name, "default", ("particles",), np.random.default_rng(2))["particles"]

    def run_level(ds_, cache, cfg_, split_, seed, name):
        calls.append(name)
        return fake_tables(ds_, cfg_, split_, seed, name, "default", ("samples", "runs"), np.random.default_rng(3))

    monkeypatch.setattr(ex, "run_proposed", proposed)
    monkeypatch.setattr(ex, "run_particle_baseline", particle)
    monkeypatch.setattr(ex, "run_run_baseline", run_level)
    args = (ds, None, cfg, split, 0, ["wearpg", "contour_cnn", "physics_only"], ["full", "no_fractal"], "cpu", tmp_path)
    ex._run_split(*args)
    assert calls == ["full", "no_fractal", "contour_cnn", "physics_only"]
    assert len(list((tmp_path / "predictions" / "parts").glob("*.parquet"))) == 3 + 3 + 1 + 2
    ex._run_split(*args)  # everything exists: nothing is recomputed
    assert len(calls) == 4
    ex._part_path(tmp_path, "runs", "cv", "wearpg", "no_fractal", split.name, 0).unlink()
    ex._run_split(*args)  # an incomplete set is recomputed
    assert calls[4:] == ["no_fractal"]
    cfg["eval"]["secondary_seeds"] = [0]  # seed 1: wearpg/full and run baselines only
    ex._run_split(ds, None, cfg, split, 1, ["wearpg", "contour_cnn", "physics_only"], ["full", "no_fractal"], "cpu",
                  tmp_path)
    assert calls[5:] == ["full", "physics_only"]


def test_resume_refuses_parts_of_another_config(tmp_path):
    cfg = make_cfg()
    ex._check_resume(tmp_path, cfg)  # empty dir
    (tmp_path / "config.yaml").write_text("config_hash: other\n", encoding="utf-8")
    ex._check_resume(tmp_path, cfg)  # no parts yet: nothing to mix up
    ex._write_part(pd.DataFrame({"x": [1]}), tmp_path / "predictions" / "parts" / "runs__cv__m__v__f__seed0.parquet")
    with pytest.raises(ValueError, match="another config"):
        ex._check_resume(tmp_path, cfg)
    (tmp_path / "config.yaml").write_text(f"config_hash: {ex.stable_hash(cfg)}\n", encoding="utf-8")
    ex._check_resume(tmp_path, cfg)  # same config: resume


def test_both_scored_pairs_only_items_both_models_predicted():
    a = pd.DataFrame({"run_id": ["r0", "r1", "r2"], "log10_rate_pred": [1.0, np.nan, 2.0]})
    b = pd.DataFrame({"run_id": ["r0", "r1", "r2", "r3"], "log10_rate_pred": [np.nan, 1.0, 2.0, 3.0]})
    a2, b2 = ex._both_scored(a, b, "runs")
    assert a2.run_id.tolist() == ["r2"] and b2.run_id.tolist() == ["r2"]
    p = pd.DataFrame({"particle_id": ["p0"]})
    assert ex._both_scored(p, p, "particles")[0] is p


def test_evaluate_results_writes_every_metrics_file(tiny_dataset_on_disk, tmp_path):
    ds = tiny_dataset_on_disk
    out = tmp_path / "results"
    cfg = make_cfg("eval.n_bootstrap=20", "eval.seeds=[0, 1]")
    cfg["paths"].update(data_root=str(ds.root), output_dir=str(out))
    runs = sorted(ds.runs.run_id)
    splits = [split_of(runs[0::2], "specimen_k2_f0"), split_of(runs[1::2], "specimen_k2_f1"),
              split_of([r for r in runs if "AISI" in r], "logo_material_id_AISI52100", "external")]
    rng = np.random.default_rng(0)
    for split in splits:
        for seed in (0, 1):
            tables = {("wearpg", "full"): ex.LEVEL_COLUMNS, ("wearpg", "no_fractal"): ex.LEVEL_COLUMNS,
                      ("contour_cnn", "default"): ("particles",), ("physics_only", "default"): ("samples", "runs")}
            for (model, variant), levels in tables.items():
                for lvl, df in fake_tables(ds, cfg, split, seed, model, variant, levels, rng).items():
                    if model == "physics_only" and lvl == "runs" and split.protocol == "external":
                        df.loc[0, ["log10_rate_pred", "log_debris_volume_mm3", "note"]] = [np.nan, np.nan, NOTE_ETA]
                    ex._write_part(df, ex._part_path(out, lvl, split.protocol, model, variant, split.name, seed))

    summary = ex.evaluate_results(out, cfg)
    m = {name: pd.read_csv(out / "metrics" / f"{name}.csv") for name in ex.METRIC_COLUMNS}
    for name, cols in ex.METRIC_COLUMNS.items():
        assert m[name].columns.tolist() == cols, name
        assert len(m[name]), name

    cls = m["classification"]
    assert {"macro_f1", "ece", "log_loss", "macro_f1_gt"} <= set(cls.metric)  # *_gt rows: the tiny dataset is synthetic
    assert set(cls.model) == {"wearpg", "contour_cnn"} and set(cls.protocol) == {"cv", "external"}
    # point = metric on all data averaged over the training seeds, computed by hand
    part = ex.load_predictions(out, "particles")
    cc = part[(part.model == "contour_cnn") & (part.protocol == "cv")]
    hand = np.mean([classification_metrics(g.y_true, g[prob_columns()].to_numpy())["macro_f1"]
                    for _, g in cc.groupby("seed")])
    row = cls[(cls.model == "contour_cnn") & (cls.protocol == "cv") & (cls.metric == "macro_f1")].iloc[0]
    assert row.point == pytest.approx(hand) and row.seed_mean == pytest.approx(hand) and row.lo <= row.hi
    assert row.n == cc.particle_id[cc.y_true >= 0].nunique()
    assert set(m["per_class"].metric) == {"sensitivity", "specificity", "f1"}
    assert set(m["severity"].metric) == {"macro_f1", "balanced_accuracy", "ordinal_mae", "qwk"}
    assert set(m["regression"].level) == {"sample", "run"} and "within_condition_spearman" in set(m["regression"].metric)

    comp = m["comparisons"]
    assert set(comp.family) == {"baselines", "ablations", "transfer"}
    assert set(comp[comp.family == "transfer"].protocol) == {"cv", "external"}
    claim = {("particle", "macro_f1"), ("sample", "qwk"), ("run", "mae")}
    has_p = comp.p_value.notna()
    assert {tuple(x) for x in comp.loc[has_p, ["level", "metric"]].to_numpy()} == claim
    assert (comp.p_adj.notna() == has_p).all() and (comp.p_adj[has_p] >= comp.p_value[has_p] - 1e-12).all()
    assert set(comp[comp.family == "baselines"].model_b) == {"contour_cnn", "physics_only"}

    pg = m["per_group"]
    assert {"material_id", "magnification", "source_subset", "fold"} <= set(pg.group_col)
    assert set(pg[pg.group_col == "source_subset"].group) == {"eds"}  # tiny particles are single-view
    assert {"run_mae", "macro_f1"} <= set(pg[(pg.group_col == "fold") & (pg.protocol == "external")].metric)
    assert set(m["transition"].model) == {"wearpg", "physics_only"}
    assert set(m["volume_consistency"].model) == {"wearpg", "physics_only"}
    exc = m["exclusions"]
    eta = exc[(exc.reason == NOTE_ETA) & (exc.protocol == "external")].set_index(["model", "variant"]).n_runs
    assert eta["physics_only", "default"] == 1 and eta["wearpg", "full"] == 0
    conf = m["confusion_by_group"]
    assert conf[(conf.group_col == "all") & (conf.model == "contour_cnn") & (conf.protocol == "cv")]["count"].sum() \
        == (cc.y_true >= 0).sum()

    cl = ex.claim_verdicts(out, cfg, [])  # no gates, no no_mechanism_link / fusion parts: not evaluable, no crash
    assert cl.claim.tolist() == ["C1", "C1b", "C2", "C3", "C4", "H-F (C1 support)", "H-F with_conditions (secondary)"]
    assert set(cl.verdict) <= {"for_us", "against_us", "inconclusive", "not evaluable"}
    assert (out / "metrics" / "claims.csv").exists() and cl.set_index("claim").verdict["C2"] == "not evaluable"

    js = json.loads((out / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert js["config_hash"] == summary["config_hash"]
    assert set(js["cv"]["wearpg|full"]["particle.macro_f1"]) == {"point", "lo", "hi", "seed_mean", "seed_sd"}
    assert "run.mae" in js["external"]["physics_only|default"]


@pytest.mark.slow  # generator fixture + cache build + training on generated images (~1-2 min on CPU)
def test_mini_end_to_end_on_synthetic_small(synthetic_small, tmp_path):
    out = tmp_path / "results"
    cfg = make_cfg("model.hidden=32", "model.gnn_layers=2", "train.epochs=2", "train.particle_epochs=2",
                   "eval.n_splits=3", "eval.seeds=[0]", "eval.n_bootstrap=30")
    cfg["paths"].update(data_root=str(synthetic_small.root), cache_dir=str(tmp_path / "cache"), output_dir=str(out))
    ds, cache = ex.prepare(cfg)
    split = ex.make_splits(ds, cfg)[0]
    ex._run_split(ds, cache, cfg, split, 0, ["wearpg", "fourier_fractal_svm", "physics_only"], ["full"], "cpu", out)
    assert len(list((out / "predictions" / "parts").glob("*.parquet"))) == 3 + 1 + 2
    assert (out / "training" / "wearpg" / "full" / f"{split.name}_seed0" / "best.pt").exists()

    runs = ex.load_predictions(out, "runs")
    for model in ("wearpg", "physics_only"):
        assert sorted(runs.run_id[runs.model == model]) == sorted(split.test_runs)
    part = ex.load_predictions(out, "particles")
    assert set(part.model) == {"wearpg", "fourier_fractal_svm"}
    np.testing.assert_allclose(part[prob_columns()].sum(axis=1), 1.0, atol=1e-5)

    cv = ex.evaluate_results(out, cfg)["cv"]
    for model in ("wearpg|full", "fourier_fractal_svm|default"):
        assert np.isfinite(cv[model]["particle.macro_f1"]["point"])
    for model in ("wearpg|full", "physics_only|default"):
        assert np.isfinite(cv[model]["run.mae"]["point"])

    gates = ex._gates(ds, cache, out, cfg)
    assert [g.name for g in gates] == ["pilot", "graph_value", "mechanism", "wear", "wear_rate", "portfolio"]
    assert json.loads((out / "gates" / "gates.json").read_text(encoding="utf-8"))["decision_valid"] is False
