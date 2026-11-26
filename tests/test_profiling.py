"""Profiling on CPU with tiny node counts; the budget-based 'oom' / 'skipped' paths via a monkeypatched total VRAM
(on CPU the peak is the deterministic saved-for-backward estimate)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from wearpg import profiling
from wearpg.config import load_config
from wearpg.profiling import profile_contour_cnn, profile_gnn, recommend_max_nodes, run_profiling

ROOT = Path(__file__).resolve().parents[1]
GNN_COLS = ["n_nodes", "graphs_per_batch", "peak_mem_mb", "fwd_bwd_ms", "graphs_per_s", "status"]


def _cfg(*overrides):
    return load_config(ROOT / "configs" / "default.yaml",
                       ["model.hidden=16", "model.gnn_layers=2", "device=cpu", *overrides])


def _peaks(df: pd.DataFrame, batch_col: str = "graphs_per_batch") -> pd.Series:
    return df.set_index(["n_nodes", batch_col]).peak_mem_mb


def test_profile_gnn_cpu():
    df = profile_gnn([48, 32], [4, 2], _cfg(), device=torch.device("cpu"), repeats=1)
    assert list(df.columns) == GNN_COLS
    assert df[["n_nodes", "graphs_per_batch"]].values.tolist() == [[32, 2], [32, 4], [48, 2], [48, 4]]  # ascending
    assert (df.status == "ok").all() and (df.peak_mem_mb > 0).all() and (df.fwd_bwd_ms > 0).all()
    assert df.graphs_per_s.to_numpy() == pytest.approx((df.graphs_per_batch / df.fwd_bwd_ms * 1e3).to_numpy())
    m = _peaks(df)
    assert m[32, 4] / m[32, 2] == pytest.approx(2.0, rel=0.1)  # ~linear in graphs per batch
    assert m[48, 2] > m[32, 2]
    again = profile_gnn([32, 48], [2, 4], _cfg(), device="cpu", repeats=1)
    assert _peaks(again).equals(m)  # the CPU estimate is deterministic


def test_profile_gnn_budget_oom_path(monkeypatch):
    monkeypatch.setattr(profiling, "_total_mb", lambda device: 1e-6)  # tiny "VRAM": any measured peak is over budget
    df = profile_gnn([32, 48], [2, 4], _cfg(), device="cpu", repeats=1)
    assert df.status.tolist() == ["oom", "skipped", "skipped", "skipped"]
    assert df.peak_mem_mb.iloc[0] > 0 and math.isfinite(df.fwd_bwd_ms.iloc[0])
    assert df.peak_mem_mb.iloc[1:].isna().all() and df.fwd_bwd_ms.iloc[1:].isna().all()  # skipped = not run


def test_profile_gnn_budget_skips_predicted_overruns(monkeypatch):
    cfg = _cfg()
    free = _peaks(profile_gnn([32, 48], [2, 4], cfg, device="cpu", repeats=1))
    budget = 1.5 * free[32, 2]  # (32, 4) is predicted at ~2x the (32, 2) peak, (48, 2) at 64/48 = 1.33x
    monkeypatch.setattr(profiling, "_total_mb", lambda device: budget / cfg["profiling"]["vram_budget_fraction"])
    df = profile_gnn([32, 48], [2, 4], cfg, device="cpu", repeats=1)
    st = df.set_index(["n_nodes", "graphs_per_batch"]).status
    assert st.to_dict() == {(32, 2): "ok", (32, 4): "skipped", (48, 2): "ok", (48, 4): "skipped"}
    assert recommend_max_nodes(df, budget, 2) == 48 and recommend_max_nodes(df, budget, 4) is None


def test_recommend_max_nodes():
    nan = float("nan")
    df = pd.DataFrame({"n_nodes": [64, 128, 256, 64, 128, 256], "graphs_per_batch": [64] * 3 + [256] * 3,
                       "peak_mem_mb": [100.0, 200.0, 400.0, 400.0, 800.0, nan], "fwd_bwd_ms": 1.0,
                       "graphs_per_s": 1.0, "status": ["ok", "ok", "oom", "ok", "oom", "skipped"]})
    assert recommend_max_nodes(df, 1000.0, 64) == 128  # 256 is oom
    assert recommend_max_nodes(df, 150.0, 64) == 64  # a stricter budget than the profiling one
    assert recommend_max_nodes(df, 1000.0, 256) == 64
    assert recommend_max_nodes(df, 50.0, 64) is None
    assert recommend_max_nodes(df, 1000.0, 1024) is None  # batch size not profiled


def test_profile_contour_cnn_cpu(monkeypatch):
    df = profile_contour_cnn([32, 64], [2, 4], device="cpu", repeats=1)
    assert list(df.columns) == ["n_nodes", "batch_size", "peak_mem_mb", "fwd_bwd_ms", "particles_per_s", "status"]
    assert (df.status == "ok").all() and (df.fwd_bwd_ms > 0).all()
    m = _peaks(df, "batch_size")
    assert m[32, 4] > m[32, 2] and m[64, 2] > m[32, 2]
    monkeypatch.setattr(profiling, "_total_mb", lambda device: 1e-6)
    assert profile_contour_cnn([32], [2, 4], device="cpu", repeats=1).status.tolist() == ["oom", "skipped"]


def test_run_profiling_writes_outputs(tmp_path):
    cfg = _cfg("profiling.node_counts=[32]", "profiling.graphs_per_batch=[2, 4]")
    summary = run_profiling(cfg, tmp_path)
    out = tmp_path / "profiling"
    gnn, cnn = pd.read_csv(out / "gnn.csv"), pd.read_csv(out / "contour_cnn.csv")
    assert list(gnn.columns) == GNN_COLS and len(gnn) == 2 and (gnn.status == "ok").all()
    assert list(cnn.batch_size) == [2, 4]
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    assert summary["device"] == "cpu" and summary["total_vram_mb"] is None and summary["vram_budget_mb"] is None
    assert summary["recommended_max_nodes"] == {"2": 32, "4": 32}  # no VRAM budget on CPU
    assert summary["graph_n_nodes"] == 128 and summary["model"] == {"hidden": 16, "gnn_layers": 2, "conv": "gine"}
