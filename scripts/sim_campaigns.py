"""SIMULATION stress tests and power analysis of the run-level debris estimators (DESIGN sec. 15 [CORE]).

Each campaign = one parameter-level synthetic campaign (synthetic.simulate_tables, no rendering) evaluated exactly like
the pipeline: grouped 5-fold cv by specimen (experiments.make_splits) and experiments.run_run_baseline for
physics_only, conditions_only and physics_conditions (attach_hierarchy coverage rule included). Results are SIMULATION
(software and design verification), never evidence about real wear.

  stress: one factor at a time around configs/synthetic_pilot.yaml, --reps campaigns per setting; point metrics per
          campaign -> outputs/simulation/stress/stress_campaigns.csv (appended per campaign, resumable)
  power:  runs per condition x number of conditions x imaged fraction, --reps campaigns per cell; per campaign the
          paired cluster-bootstrap CI of the run-MAE difference vs conditions_only and the cluster-bootstrap CI of the
          within-condition Spearman -> outputs/simulation/power/power_campaigns.csv (appended, resumable)
  summary: aggregates both CSVs into *_summary.csv + PNG figures + README.md

Usage: python scripts/sim_campaigns.py stress|power|summary [--reps N] [--workers N] [--n-boot N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "simulation"
MODELS = ("physics_only", "conditions_only", "physics_conditions")
DEBRIS = ("physics_only", "physics_conditions")
BANNER = ("SIMULATION - parameter-level synthetic campaigns (no rendering); design and software verification only, "
          "not evidence about real wear")
STRESS = {  # factor -> (generator config path, levels); the pilot value of each factor is marked in the summary
    "budget_sigma": ("debris.budget_sigma", [0.1, 0.3, 0.6]),
    "imaged_fraction_range": ("debris.imaged_fraction_range",
                              [[0.05, 0.5], [0.02, 0.2], [0.01, 0.1], [0.005, 0.05]]),
    "material_sigma": ("capture.material_sigma", [0.0, 0.2, 0.5]),
    "min_area_px": ("imaging.min_area_px", [30, 80, 150]),
    "samples_per_run": ("samples_per_run", [3, 5, 8]),
}
PILOT = {"budget_sigma": 0.3, "imaged_fraction_range": [0.05, 0.5], "material_sigma": 0.2, "min_area_px": 30,
         "samples_per_run": 5}
CONDITIONS = {6: ([20, 120], [0.6]), 12: ([20, 120], [0.2, 0.6]), 18: ([20, 60, 120], [0.2, 0.6])}  # x 3 materials
POWER = {"repeats": [3, 5, 8], "n_conditions": [6, 12, 18],
         "imaged_fraction_range": [[0.05, 0.5], [0.015, 0.15], [0.005, 0.05]]}


def _set(cfg: dict, dotted: str, value) -> dict:
    from wearpg.config import apply_overrides
    import json
    return apply_overrides(cfg, [f"{dotted}={json.dumps(value)}"])


def run_campaign(gen_cfg: dict, seed: int, n_boot: int) -> dict:
    """Pipeline-identical cv of the three run baselines on one simulated campaign -> flat metric dict."""
    import numpy as np
    from wearpg import experiments as ex
    from wearpg.config import load_config
    from wearpg.metrics import regression_metrics, volume_consistency
    from wearpg.stats import cluster_bootstrap, paired_cluster_bootstrap
    from wearpg.synthetic import simulate_tables

    cfg = load_config(ROOT / "configs" / "default.yaml", ["eval.seeds=[0]"])
    ds, desc = simulate_tables(gen_cfg, seed)
    cache = SimpleNamespace(descriptors=desc, particle_ids=list(desc.index))
    parts = {m: [] for m in MODELS}
    for split in ex.make_splits(ds, cfg):
        for m in MODELS:
            parts[m].append(ex.run_run_baseline(ds, cache, cfg, split, 0, m)["runs"])
    runs = {m: __import__("pandas").concat(v, ignore_index=True) for m, v in parts.items()}
    out = {"n_runs": int(len(ds.runs)), "n_severe": int((ds.runs.regime == "severe").sum()),
           "median_n_total": float(ds.samples.n_particles_total.median()),
           "empty_sample_fraction": float((ds.samples.n_particles_total == 0).mean())}
    for m, d in runs.items():
        d = d[np.isfinite(d.log10_rate_true.to_numpy(float)) & np.isfinite(d.log10_rate_pred.to_numpy(float))]
        r = regression_metrics(d.log10_rate_true, d.log10_rate_pred, d.condition_id)
        out.update({f"{m}.mae": r["mae"], f"{m}.within_spearman": r["within_condition_spearman"], f"{m}.n": r["n"]})
        if m in DEBRIS:
            v = volume_consistency(d.log_debris_volume_mm3, d.log_grav_volume_mm3)
            out.update({f"{m}.slope": v["slope"], f"{m}.ccc": v["ccc"]})
        runs[m] = d
    if n_boot:
        cond = runs["conditions_only"]
        mae = lambda d: regression_metrics(d.log10_rate_true, d.log10_rate_pred)["mae"]  # noqa: E731
        wsp = lambda d: regression_metrics(d.log10_rate_true, d.log10_rate_pred,  # noqa: E731
                                           d.condition_id)["within_condition_spearman"]
        for m in DEBRIS:
            a, b = ex._both_scored(runs[m], cond, "runs")
            p = paired_cluster_bootstrap(mae, a, b, "specimen_id", "run_id", n_boot=n_boot, seed=seed)["value"]
            w = cluster_bootstrap(wsp, runs[m], "specimen_id", n_boot=n_boot, seed=seed)["value"]
            out.update({f"{m}.mae_diff": p["diff"], f"{m}.mae_diff_lo": p["lo"], f"{m}.mae_diff_hi": p["hi"],
                        f"{m}.within_lo": w["lo"], f"{m}.within_hi": w["hi"]})
    return out


def _worker(job: tuple) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    key, gen_cfg, seed, n_boot = job
    t = time.perf_counter()
    res = run_campaign(gen_cfg, seed, n_boot)
    return {**key, "seed": seed, **res, "seconds": round(time.perf_counter() - t, 1)}


def _jobs(kind: str, reps: int, n_boot: int) -> list[tuple]:
    import json
    from wearpg.synthetic import load_synthetic_config
    base = load_synthetic_config(ROOT / "configs" / "synthetic_pilot.yaml")
    jobs = []
    if kind == "stress":
        for factor, (path, levels) in STRESS.items():
            for lv in levels:
                key = {"factor": factor, "level": json.dumps(lv), "pilot_level": lv == PILOT[factor]}
                jobs += [(key, _set(base, path, lv), 1000 + s, 0) for s in range(reps)]
    else:
        for rep in POWER["repeats"]:
            for nc in POWER["n_conditions"]:
                for imf in POWER["imaged_fraction_range"]:
                    loads, speeds = CONDITIONS[nc]
                    c = _set(_set(_set(_set(base, "repeats", rep), "loads_N", loads), "speeds_m_s", speeds),
                             "debris.imaged_fraction_range", imf)
                    key = {"repeats": rep, "n_conditions": nc, "imaged_fraction_range": json.dumps(imf)}
                    jobs += [(key, c, 2000 + s, n_boot) for s in range(reps)]
    return sorted(jobs, key=lambda j: j[2])  # replicate-major: a partial run still covers every setting


def run(kind: str, reps: int, workers: int, n_boot: int) -> Path:
    import pandas as pd
    out_dir = OUT / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{kind}_campaigns.csv"
    jobs = _jobs(kind, reps, n_boot)
    key_cols = list(jobs[0][0]) + ["seed"]
    done = set()
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        done = {tuple(r) for r in old[key_cols].itertuples(index=False)}
    todo = [j for j in jobs if tuple(str(v) for v in [*j[0].values(), j[2]]) not in done]
    print(f"{kind}: {len(jobs)} campaigns, {len(todo)} to run ({BANNER})", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker, j) for j in todo]
        for i, f in enumerate(as_completed(futs), 1):
            row = pd.DataFrame([f.result()])
            row.to_csv(path, mode="a", header=not path.exists(), index=False, lineterminator="\n")
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.perf_counter() - t0:.0f} s", flush=True)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["stress", "power", "summary"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    if a.kind == "summary":
        sys.path.insert(0, str(Path(__file__).parent))
        from sim_summary import overview, summarize
        summarize(OUT)
        overview(OUT)
        return 0
    run(a.kind, a.reps or (20 if a.kind == "stress" else 50), a.workers, a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
