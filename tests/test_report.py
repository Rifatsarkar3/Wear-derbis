"""report.py on fabricated results directories (DESIGN sec. 11 layout and metrics columns)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wearpg.gates import GateResult, write_gate_report
from wearpg.report import FIGURES, TABLE_SPECS, build_report, latex_table
from wearpg.schema import (MECHANISMS, PARTICLE_PRED_COLUMNS, RUN_PRED_COLUMNS, SAMPLE_PRED_COLUMNS, SEVERITY_PROB_COLUMNS,
                           SYNTHETIC_BANNER, VERIFICATION_NOTICE, prob_columns)
from wearpg.utils import save_json

MODELS = [("wearpg", "full"), ("wearpg", "no_physics"), ("contour_cnn", "default"), ("fourier_fractal_svm", "default")]
RUN_MODELS = [("wearpg", "full"), ("physics_only", "default"), ("conditions_only", "default")]


def _ci_rows(models, metrics, extra=None, gt=()):
    rows = []
    for i, (m, v) in enumerate(models):
        for metric in [*metrics, *gt]:
            p = 0.9 - 0.1 * i
            rows.append(dict(protocol="cv", model=m, variant=v, **(extra or {}), metric=metric, point=p, lo=p - 0.05,
                             hi=p + 0.05, seed_mean=p, seed_sd=0.01, n=40))
    return rows


def _keys(model, variant, seed=0):
    return dict(protocol="cv", model=model, variant=variant, fold="specimen_k3_f0", seed=seed, split="test")


def make_results(root, synthetic: bool = True, rng_seed: int = 0):
    """Results dir with every metrics file, gates, profiling and prediction parts for MODELS / RUN_MODELS."""
    rng = np.random.default_rng(rng_seed)
    save_json({"name": "fab", "is_synthetic": synthetic, "seed": 0, "stats": {"x": 1}, "describe": {"n_runs": 3}},
              root / "dataset_meta.json")
    met = root / "metrics"
    met.mkdir(parents=True)
    pd.DataFrame([dict(protocol="external", model="wearpg", variant="full", reason="eta not transferable",
                       n_runs=2)]).to_csv(met / "exclusions.csv", index=False)
    gt = ("macro_f1_gt",) if synthetic else ()
    pd.DataFrame(_ci_rows(MODELS, ["macro_f1", "balanced_accuracy", "accuracy", "ece", "brier", "log_loss"], gt=gt)
                 ).to_csv(met / "classification.csv", index=False)
    pd.DataFrame(_ci_rows(MODELS, ["macro_f1", "balanced_accuracy", "ordinal_mae", "qwk"])).to_csv(
        met / "severity.csv", index=False)
    reg = [r for lvl in ("sample", "run") for r in _ci_rows(RUN_MODELS, ["mae", "rmse", "r2", "bias", "spearman",
                                                                         "within_condition_spearman"], {"level": lvl})]
    pd.DataFrame(reg).to_csv(met / "regression.csv", index=False)
    pc = [dict(protocol="cv", model=m, variant=v, **{"class": c}, metric=k, point=0.5, lo=0.4, hi=0.6, n=10)
          for m, v in MODELS for c in MECHANISMS for k in ("sensitivity", "specificity", "f1")]
    pd.DataFrame(pc).to_csv(met / "per_class.csv", index=False)
    pd.DataFrame([
        dict(protocol="cv", level="particles", family="baselines", metric="macro_f1", model_a="wearpg",
             variant_a="full", model_b="contour_cnn", variant_b="default", diff=0.1, lo=0.02, hi=0.18, rel_diff=0.12,
             p_value=0.0004, p_adj=0.0008, method="holm"),
        dict(protocol="cv", level="particles", family="baselines", metric="ece", model_a="wearpg", variant_a="full",
             model_b="contour_cnn", variant_b="default", diff=-0.01, lo=-0.03, hi=0.01, rel_diff=-0.1,
             p_value=np.nan, p_adj=np.nan, method="holm"),
    ]).to_csv(met / "comparisons.csv", index=False)
    pg = [dict(protocol="cv", model=m, variant=v, group_col=gc, group=g, metric=k, value=0.7, n=12)
          for m, v in MODELS for gc, g in (("material_id", "AISI52100"), ("magnification", 200.0),
                                           ("source_subset", "multiview")) for k in ("macro_f1", "accuracy")]
    pd.DataFrame(pg).to_csv(met / "per_group.csv", index=False)
    pd.DataFrame([dict(protocol="cv", model=m, variant=v, mae_s=120.0, bias_s=-10.0, mae_intervals=0.4,
                       detection_rate=0.8, early_alarm_rate=0.1, false_alarm_rate=0.2, oracle_floor_s=60.0,
                       interval_s=300.0, n_transition_runs=5, n_stable_runs=7) for m, v in RUN_MODELS]
                 ).to_csv(met / "transition.csv", index=False)
    pd.DataFrame([dict(protocol="cv", model=m, variant=v, mean_log_ratio=0.1, sd_log_ratio=0.3, slope=0.9,
                       intercept=0.2, r2=0.8, ccc=0.85, n=12) for m, v in RUN_MODELS[:2]]
                 ).to_csv(met / "volume_consistency.csv", index=False)
    write_gate_report([GateResult("pilot", False, "rho below threshold & 50% power", {"alpha": 0.05}, {}),
                       GateResult("wear", None, "no external predictions", {}, {})], root / "gates", synthetic)
    prof = root / "profiling"
    prof.mkdir()
    grid = [(n, b) for n in (64, 128) for b in (64, 256)]
    pd.DataFrame([dict(n_nodes=n, graphs_per_batch=b, peak_mem_mb=n * b / 50, fwd_bwd_ms=5.0, graphs_per_s=b / 0.005,
                       status="ok") for n, b in grid]).to_csv(prof / "gnn.csv", index=False)
    pd.DataFrame([dict(n_nodes=n, batch_size=b, peak_mem_mb=n * b / 500, fwd_bwd_ms=1.0, particles_per_s=b / 0.001,
                       status="ok") for n, b in grid]).to_csv(prof / "contour_cnn.csv", index=False)
    save_json({"device_name": "fake GPU", "vram_budget_mb": 9000.0, "recommended_max_nodes": {"64": 128}},
              prof / "summary.json")

    parts = root / "predictions" / "parts"
    parts.mkdir(parents=True)
    runs = [f"R{i}" for i in range(3)]
    for m, v in MODELS:
        proba = rng.dirichlet(np.ones(len(MECHANISMS)), size=40)
        df = pd.DataFrame(proba, columns=prob_columns())
        df = df.assign(**_keys(m, v), particle_id=[f"p{i}" for i in range(40)], y_true=rng.integers(-1, 7, 40),
                       y_true_gt=-1, y_pred=proba.argmax(1), log_volume_pred=np.nan, log_volume_true=np.nan,
                       run_id=np.repeat(runs, [14, 13, 13]))
        df.reindex(columns=PARTICLE_PRED_COLUMNS).to_parquet(parts / f"particles__cv__{m}__{v}__specimen_k3_f0__seed0.parquet")
    for m, v in RUN_MODELS:
        sp = rng.dirichlet(np.ones(3), size=9)
        s = pd.DataFrame(sp, columns=list(SEVERITY_PROB_COLUMNS)).assign(
            **_keys(m, v), sample_id=[f"{r}_s{j}" for r in runs for j in range(3)], run_id=np.repeat(runs, 3),
            t_start_s=np.tile([0.0, 600, 1200], 3), time_s=np.tile([600.0, 1200, 1800], 3), severity_pred=sp.argmax(1))
        s.reindex(columns=SAMPLE_PRED_COLUMNS).to_parquet(parts / f"samples__cv__{m}__{v}__specimen_k3_f0__seed0.parquet")
        r = pd.DataFrame(dict(**_keys(m, v), run_id=runs, log10_rate_true=[-6.0, -5.0, -4.0],
                              log10_rate_pred=[-5.8, -5.1, -4.3], log_grav_volume_mm3=[-3.0, -1.0, 0.5],
                              log_debris_volume_mm3=[-2.8, -1.2, 0.9] if m != "conditions_only" else np.nan,
                              transition_time_true=[np.nan, 900.0, 300.0], transition_time_pred=[np.nan, 1500.0, 900.0],
                              coverage=1.0, note=""))
        r.reindex(columns=RUN_PRED_COLUMNS).to_parquet(parts / f"runs__cv__{m}__{v}__specimen_k3_f0__seed0.parquet")
    return root


@pytest.fixture
def cfg_with_data(tiny_dataset, tmp_path):
    ds_root = tiny_dataset.save(tmp_path / "data")
    return {"paths": {"data_root": ds_root.as_posix()}, "eval": {"ci": 0.95, "ece_bins": 10, "transition_threshold": 0.5}}


def test_latex_table_structure_and_escaping():
    df = pd.DataFrame({"model": ["fourier_fractal_svm", "a & b"], "score": [0.8123, np.nan], "n": [12, 3],
                       "ci": ["0.812 [0.750, 0.861]", "-"]})
    tex = latex_table(df, "Accuracy at 95 % CI", "tab:x")
    assert tex.count(r"\begin{table}") == 1 and r"\end{table}" in tex
    for rule in (r"\toprule", r"\midrule", r"\bottomrule", r"\label{tab:x}", r"\begin{tabular}{lrrr}"):
        assert rule in tex
    assert r"fourier\_fractal\_svm" in tex and r"a \& b" in tex and r"95 \% CI" in tex
    assert "0.812 & 12" in tex and "- & 3" in tex  # compact numbers, NaN -> '-'
    assert SYNTHETIC_BANNER not in tex
    syn = latex_table(df, "Accuracy", "tab:x", synthetic=True)
    assert rf"\caption{{\textbf{{{SYNTHETIC_BANNER}.}} Accuracy}}" in syn
    assert latex_table(pd.DataFrame({"msg": ["x" * 60]}), "c", "tab:y").count(r"{p{0.5\linewidth}}") == 1


def test_build_report_synthetic_banner_everywhere_and_no_paper_copy(tmp_path, cfg_with_data):
    res = make_results(tmp_path / "out", synthetic=True)
    paper = tmp_path / "paper"
    md_path = build_report(res, cfg_with_data, paper_dir=paper)
    md = md_path.read_text(encoding="utf-8")
    assert md_path == res / "report" / "report.md"
    assert SYNTHETIC_BANNER in md and VERIFICATION_NOTICE in md
    for stem in TABLE_SPECS:  # every table written, complete float, banner in the caption
        tex = (res / "report" / "tables" / f"{stem}.tex").read_text(encoding="utf-8")
        assert rf"\label{{tab:{stem}}}" in tex and rf"\caption{{\textbf{{{SYNTHETIC_BANNER}" in tex, stem
    for stem in FIGURES:
        assert (res / "report" / "figures" / f"{stem}.png").read_bytes()[:4] == b"\x89PNG", stem
    assert not paper.exists()  # synthetic results never enter the manuscript
    assert "Not copied to" in md and "Skipped" not in md
    assert "| n_runs | 3 |" in md and "eta not transferable" in md  # describe counts, exclusions listed
    cls = (res / "report" / "tables" / "classification.tex").read_text(encoding="utf-8")
    assert "Macro-F1" in cls and "\\_gt" not in cls  # ground-truth diagnostics stay out of the tables ...
    assert "macro_f1_gt" in md  # ... but are listed in the report
    # strongest baseline by cv macro-F1 = contour_cnn (0.7 > 0.6): per-class/per-group show it, not the weaker one
    pcl = (res / "report" / "tables" / "per_class.tex").read_text(encoding="utf-8")
    assert r"contour\_cnn" in pcl and "fourier" not in pcl
    cmp_tex = (res / "report" / "tables" / "comparisons.tex").read_text(encoding="utf-8")
    assert "<0.001" in cmp_tex and " ece " not in cmp_tex  # claim rows (with p) only
    assert "Method: holm." in cmp_tex and "& holm" not in cmp_tex  # constant method text moved to the caption
    ds_tex = (res / "report" / "tables" / "dataset_summary.tex").read_text(encoding="utf-8")
    assert "all &  & - & - & 12 & 6 & 24 & 96 & 96 & 96" in ds_tex  # tiny dataset totals


def test_build_report_real_data_copies_to_paper(tmp_path):
    res = make_results(tmp_path / "out", synthetic=False)
    (res / "report" / "tables").mkdir(parents=True)
    (res / "report" / "tables" / "transition.tex").write_text("stale", encoding="utf-8")
    paper = tmp_path / "paper"
    md = build_report(res, {}, paper_dir=paper).read_text(encoding="utf-8")
    assert SYNTHETIC_BANNER not in md and VERIFICATION_NOTICE not in md
    written = sorted(p.name for p in (res / "report" / "tables").glob("*.tex"))
    assert written == sorted(f"{s}.tex" for s in TABLE_SPECS if s != "dataset_summary")  # no dataset without cfg
    assert sorted(p.name for p in (paper / "tables").glob("*.tex")) == written
    assert sorted(p.name for p in (paper / "figures").glob("*.png")) == sorted(f"{s}.png" for s in FIGURES)
    for p in (paper / "tables").glob("*.tex"):
        assert SYNTHETIC_BANNER not in p.read_text(encoding="utf-8")
    assert (paper / "tables" / "transition.tex").read_text(encoding="utf-8") != "stale"


def test_build_report_missing_inputs_and_unknown_status(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path / "nope")
    res = tmp_path / "empty"
    res.mkdir()
    md = build_report(res, paper_dir=tmp_path / "paper").read_text(encoding="utf-8")
    assert SYNTHETIC_BANNER in md  # nothing says real data -> treated as synthetic
    assert not (tmp_path / "paper").exists()
    assert "## Skipped" in md and all(f"table {s}" in md for s in TABLE_SPECS) and all(f"figure {s}" in md
                                                                                    for s in FIGURES)
    assert not list((res / "report" / "tables").glob("*.tex"))
