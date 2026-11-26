"""[H] Memory / speed profiling of the particle GNN on real contour graphs and of the simpler 1-D contour CNN
(DESIGN sec. 10; planning review s10: define the maximum node count for 12 GB VRAM).

Each (n_nodes, batch) cell runs one warm-up and `repeats` timed training steps (fp32 forward + backward, train mode).
WDDM spills to shared memory instead of raising CUDA OOM, so memory is budgeted explicitly:
budget = profiling.vram_budget_fraction * total VRAM. A cell whose measured peak exceeds the budget is "oom". A cell
predicted to exceed it (peak linear in the batch's node count, extrapolated from the largest completed cell; edge
density is constant at ~12 directed edges per node) is "skipped" without running, so no cell spills.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from wearpg.baselines.deep import ContourCNN
from wearpg.batch import ModelDims
from wearpg.graphs import GraphConfig, GraphSelection, contour_graph, select_graph, selected_dims
from wearpg.models import ParticleGNN
from wearpg.schema import MECHANISMS
from wearpg.utils import get_device, get_logger, save_json

N_SHAPES = 8  # distinct synthetic contours per node count, tiled to fill a batch
N_GLOBALS = 25  # ContourCNN geometry globals (16 shape_ + 9 fractal_ columns); negligible for its cost


def _total_mb(device: torch.device) -> float | None:
    """Total memory of a CUDA device [MiB]; None on CPU (no VRAM budget)."""
    return torch.cuda.get_device_properties(device).total_memory / 2**20 if device.type == "cuda" else None


def _contours(k: int, seed: int) -> list[np.ndarray]:
    """k rough star-shaped closed contours [px], 512 points each: aspect 1-3, harmonics 2..24 with ~h^-1.2 amplitudes
    summing to 0.6 (radius in [0.4, 1.6] r0, so the curve is simple)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
    h = np.arange(2, 25)
    out = []
    for _ in range(k):
        amp = rng.uniform(0.5, 1.0, h.size) * h**-1.2
        amp *= 0.6 / amp.sum()
        r = 1 + (amp[:, None] * np.cos(h[:, None] * t + rng.uniform(0, 2 * np.pi, h.size)[:, None])).sum(axis=0)
        a = rng.uniform(1.0, 3.0)
        out.append(np.column_stack([100 * np.sqrt(a) * r * np.cos(t), 100 / np.sqrt(a) * r * np.sin(t)]) + 300.0)
    return out


def _measure(step, device: torch.device, repeats: int) -> tuple[float, float]:
    """(peak MiB, median ms) of `repeats` timed calls of step() after one warm-up call.

    CUDA: torch.cuda.max_memory_reserved, the device memory the step actually took (what can spill on WDDM).
    CPU (no allocator statistics): bytes of the tensors saved for backward, a lower-bound estimate.
    """
    cuda = device.type == "cuda"
    saved: dict[int, int] = {}

    def pack(t):
        s = t.untyped_storage()
        saved[s.data_ptr()] = s.nbytes()
        return t

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    hooks = contextlib.nullcontext() if cuda else torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
    times, cpu_peak = [], 0
    with hooks:
        for _ in range(repeats + 1):
            saved.clear()
            if cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            step()
            if cuda:
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
            cpu_peak = max(cpu_peak, sum(saved.values()))
    peak = torch.cuda.max_memory_reserved(device) if cuda else cpu_peak
    return peak / 2**20, 1e3 * float(np.median(times[1:]))


def _sweep(node_counts, batch_sizes, setup, extra_nodes: int, device: torch.device, repeats: int,
           budget_mb: float | None, batch_col: str, rate_col: str) -> pd.DataFrame:
    """Profile every (n_nodes, batch) cell in ascending order. setup(n) -> make(b) -> step() (one forward + backward);
    setup runs lazily, once per n_nodes. A cell's peak is predicted as peak_ref * size / size_ref from the largest
    completed cell, with size = batch * (n_nodes + extra_nodes) nodes."""
    rows, ref = [], None  # ref = (size, peak) of the largest completed cell
    cuda = device.type == "cuda"
    for n in sorted(set(int(v) for v in node_counts)):
        make = None
        for b in sorted(set(int(v) for v in batch_sizes)):
            size = b * (n + extra_nodes)
            row = {"n_nodes": n, batch_col: b, "peak_mem_mb": math.nan, "fwd_bwd_ms": math.nan, rate_col: math.nan,
                   "status": "skipped"}
            rows.append(row)
            if budget_mb is not None and ref is not None and ref[1] * size / ref[0] > budget_mb:
                continue
            make = make or setup(n)
            try:
                row["peak_mem_mb"], row["fwd_bwd_ms"] = _measure(make(b), device, repeats)
                row[rate_col] = b / row["fwd_bwd_ms"] * 1e3
                row["status"] = "oom" if budget_mb is not None and row["peak_mem_mb"] > budget_mb else "ok"
            except torch.cuda.OutOfMemoryError:
                row["peak_mem_mb"], row["status"] = torch.cuda.max_memory_reserved(device) / 2**20, "oom"
            if cuda:
                torch.cuda.empty_cache()
            if ref is None or size > ref[0]:
                ref = (size, row["peak_mem_mb"])
            get_logger().info("profile %s n_nodes=%d %s=%d: %s peak %.1f MiB, %.1f ms", rate_col, n, batch_col, b,
                              row["status"], row["peak_mem_mb"], row["fwd_bwd_ms"])
    return pd.DataFrame(rows, columns=["n_nodes", batch_col, "peak_mem_mb", "fwd_bwd_ms", rate_col, "status"])


def _budget(device: torch.device, fraction: float) -> float | None:
    total = _total_mb(device)
    return None if total is None else float(fraction) * total


def profile_gnn(node_counts, graphs_per_batch, cfg: dict, device=None, repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """Training-step memory / speed of models.ParticleGNN (cfg model.hidden, gnn_layers, conv, dropout, pooling) on
    batches of real contour graphs.

    N_SHAPES synthetic rough contours -> graphs.contour_graph at each n_nodes (the other graph.* build parameters from
    cfg, so ring/chord/interior edge density is real) -> select_graph with the cfg's GraphSelection (real node_dim),
    tiled to graphs_per_batch graphs. device: None -> cfg["device"] ("auto" = CUDA if available).
    Columns: n_nodes, graphs_per_batch, peak_mem_mb [MiB], fwd_bwd_ms (median of `repeats`), graphs_per_s,
    status ok | oom | skipped ("oom" also when peak_mem_mb > profiling.vram_budget_fraction * total VRAM, since WDDM
    never raises; "skipped" = predicted over budget, not run).
    """
    device = get_device(device or cfg.get("device", "auto"))
    gcfg, sel, m = GraphConfig.from_cfg(cfg), GraphSelection.from_cfg(cfg), cfg["model"]
    torch.manual_seed(seed)
    model = ParticleGNN(*selected_dims(gcfg, sel), hidden=int(m["hidden"]), layers=int(m["gnn_layers"]),
                        conv=m["conv"], dropout=float(m["dropout"]), pooling=tuple(m["pooling"])).to(device).train()
    contours = _contours(N_SHAPES, seed)

    def setup(n: int):
        c = replace(gcfg, n_nodes=n)
        data = [select_graph(contour_graph(xy, 1.0, c), c, sel) for xy in contours]

        def make(b: int):
            batch = Batch.from_data_list([data[i % len(data)] for i in range(b)]).to(device)
            return lambda: model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                                 batch.num_graphs).sum().backward()
        return make

    return _sweep(node_counts, graphs_per_batch, setup, gcfg.interior_nodes if sel.interior else 0, device, repeats,
                  _budget(device, cfg["profiling"]["vram_budget_fraction"]), "graphs_per_batch", "graphs_per_s")


def profile_contour_cnn(node_counts, batch_sizes, device=None, repeats: int = 3,
                        vram_budget_fraction: float = 0.8) -> pd.DataFrame:
    """Training-step memory / speed of baselines.deep.ContourCNN (default hidden 128) on contour_signal batches
    (batch, 4 + len(GraphConfig().curvature_scales) = 8 channels, n_nodes) of random values (its cost does not depend
    on them). device: None -> CUDA if available. Columns: n_nodes, batch_size, peak_mem_mb, fwd_bwd_ms,
    particles_per_s, status (budget rule of profile_gnn with vram_budget_fraction)."""
    device = get_device(device)
    s = 4 + len(GraphConfig().curvature_scales)
    model = ContourCNN(ModelDims(0, 0, N_GLOBALS, 0, 0, len(MECHANISMS), signal_channels=s)).to(device).train()
    gen = torch.Generator().manual_seed(0)

    def setup(n: int):
        def make(b: int):
            # ContourCNN.forward reads only these two ParticleBatch fields
            batch = SimpleNamespace(contour_signal=torch.randn(b, s, n, generator=gen).to(device),
                                    particle_feats=torch.randn(b, N_GLOBALS, generator=gen).to(device))
            return lambda: model(batch).sum().backward()
        return make

    return _sweep(node_counts, batch_sizes, setup, 0, device, repeats, _budget(device, vram_budget_fraction),
                  "batch_size", "particles_per_s")


def recommend_max_nodes(df: pd.DataFrame, vram_budget_mb: float, graphs_per_batch: int) -> int | None:
    """Largest profiled n_nodes whose (n_nodes, graphs_per_batch) cell of a profile_gnn table is "ok" with
    peak_mem_mb <= vram_budget_mb [MiB]; None if there is no such cell."""
    ok = df[(df.graphs_per_batch == graphs_per_batch) & (df.status == "ok") & (df.peak_mem_mb <= vram_budget_mb)]
    return int(ok.n_nodes.max()) if len(ok) else None


def run_profiling(cfg: dict, out_dir) -> dict:
    """Profile the GNN and the contour CNN over profiling.node_counts x profiling.graphs_per_batch (also the CNN batch
    sizes, for a like-for-like comparison) on cfg["device"]; write out_dir/profiling/{gnn.csv, contour_cnn.csv,
    summary.json} and return the summary: device, device_name, total_vram_mb, vram_budget_fraction, vram_budget_mb
    (None on CPU), graph_n_nodes (configured graph.n_nodes), model {hidden, gnn_layers, conv},
    recommended_max_nodes {str(graphs_per_batch): n_nodes | None}, files {gnn, contour_cnn, summary}."""
    p = cfg["profiling"]
    device = get_device(cfg.get("device", "auto"))
    out = Path(out_dir) / "profiling"
    out.mkdir(parents=True, exist_ok=True)
    gnn = profile_gnn(p["node_counts"], p["graphs_per_batch"], cfg, device)
    cnn = profile_contour_cnn(p["node_counts"], p["graphs_per_batch"], device,
                              vram_budget_fraction=p["vram_budget_fraction"])
    budget = _budget(device, p["vram_budget_fraction"])
    files = {"gnn": out / "gnn.csv", "contour_cnn": out / "contour_cnn.csv", "summary": out / "summary.json"}
    gnn.to_csv(files["gnn"], index=False)
    cnn.to_csv(files["contour_cnn"], index=False)
    summary = {
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "total_vram_mb": _total_mb(device), "vram_budget_fraction": float(p["vram_budget_fraction"]),
        "vram_budget_mb": budget, "graph_n_nodes": int(cfg["graph"]["n_nodes"]),
        "model": {k: cfg["model"][k] for k in ("hidden", "gnn_layers", "conv")},
        "recommended_max_nodes": {str(g): recommend_max_nodes(gnn, math.inf if budget is None else budget, int(g))
                                  for g in p["graphs_per_batch"]},
        "files": {k: v.as_posix() for k, v in files.items()},
    }
    save_json(summary, files["summary"])
    return summary
