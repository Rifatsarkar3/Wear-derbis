"""Simulation-study scripts (scripts/sim_campaigns.py, scripts/sim_summary.py): one pipeline-identical campaign on a
tiny parameter-level config, and the summary statistics (Wilson interval, power shares, stress shares)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sim_campaigns as sc  # noqa: E402
import sim_summary as ss  # noqa: E402
from wearpg.config import deep_merge  # noqa: E402
from wearpg.synthetic import load_synthetic_config  # noqa: E402


def test_wilson_interval():
    lo, hi = ss.wilson(np.array([8]), np.array([10]))
    assert lo[0] == pytest.approx(0.4902, abs=1e-3) and hi[0] == pytest.approx(0.9433, abs=1e-3)
    lo, hi = ss.wilson(np.array([0]), np.array([50]))
    assert lo[0] == pytest.approx(0.0, abs=1e-12) and 0.05 < hi[0] < 0.08


def test_campaign_runs_the_pipeline_baselines():
    cfg = deep_merge(load_synthetic_config("smoke"), {"samples_per_run": 2, "debris": {"max_stored_per_sample": 8}})
    r = sc.run_campaign(cfg, seed=5, n_boot=20)
    assert r["n_runs"] == 12
    for m in sc.MODELS:
        assert r[f"{m}.n"] > 0 and np.isfinite(r[f"{m}.mae"])
    for m in sc.DEBRIS:
        assert r[f"{m}.mae_diff_lo"] <= r[f"{m}.mae_diff"] <= r[f"{m}.mae_diff_hi"]
        assert r[f"{m}.mae_diff"] == pytest.approx(r[f"{m}.mae"] - r["conditions_only.mae"], abs=1e-9)


def test_power_and_stress_summaries():
    rows = []
    for seed in range(4):
        rows.append({"repeats": 3, "n_conditions": 6, "imaged_fraction_range": "[0.05, 0.5]", "seed": seed,
                     "n_runs": 18, "conditions_only.mae": 0.4, "physics_only.mae": 0.2, "physics_conditions.mae": 0.25,
                     "physics_only.mae_diff_hi": -0.1 if seed < 3 else 0.1, "physics_only.within_lo": 0.2,
                     "physics_conditions.mae_diff_hi": -0.1, "physics_conditions.within_lo": -0.1 if seed else 0.1,
                     "physics_only.within_spearman": 0.8, "physics_conditions.within_spearman": 0.7})
    p = ss.power_summary(pd.DataFrame(rows)).set_index("model")
    assert p.at["physics_only", "p_mae_ci_below_0"] == 0.75 and p.at["physics_only", "p_both"] == 0.75
    assert p.at["physics_conditions", "p_within_ci_above_0"] == 0.25 and p.at["physics_conditions", "n_campaigns"] == 4
    st = pd.DataFrame([{"factor": "f", "level": "1", "pilot_level": True, "seed": s, **{f"{m}.mae": v + s / 10
                        for m, v in zip(sc.MODELS, (0.2, 0.4, 0.21))}} for s in range(3)])
    s = ss.stress_summary(st)
    mae = s[(s.metric == "mae") & (s.model == "physics_only")].iloc[0]
    assert mae["mean"] == pytest.approx(0.3) and mae.n_campaigns == 3
    share = s[s.metric == "share_mae_below_conditions_only"].iloc[0]
    assert share["mean"] == 1.0
