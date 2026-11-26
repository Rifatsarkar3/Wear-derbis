"""[F] Feature cache, train-only scalers and the run-bag / particle datasets (DESIGN sec. 8).

Particles of a role come from `splits.particles_for_runs(ds, runs, image_filter, role)` (and must be in the cache).
Everything a bag sees - its particles, sample_feats and its HT factor - is computed from these SURVIVING particles
only, so excluded-domain particles never enter a transfer evaluation. With an image filter, "table" in the DESIGN
sec. 3 HT factor therefore means the sample's surviving rows; without one they are the sample's particle rows.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import QhullError
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from wearpg.batch import ModelDims, ParticleBatch, RunBatch
from wearpg.features.contour import canonicalize, clean_closed, signed_area
from wearpg.features.descriptors import particle_descriptors, view_descriptors
from wearpg.graphs import (FEATURE_VERSION, GraphConfig, GraphSelection, boundary_signals, contour_graph, pack_graphs,
                           select_graph, selected_dims, unpack_graph)
from wearpg.labels import soft_labels
from wearpg.schema import EDS_COLUMNS, ELEMENTS, MECHANISMS, SEVERITY_LEVELS, WearDataset
from wearpg.segmentation import extract_crop, load_image, native_patch, to_gray
from wearpg.splits import particles_for_runs, safe_name
from wearpg.utils import get_logger, load_json, save_json, stable_hash

SAMPLE_FEATURES = ["log_n_total", "log_obs_fraction", "log_ecd_q10", "log_ecd_q50", "log_ecd_q90",
                   "log_total_area_um2"]
COND_FEATURES = ["log_load_N", "log_speed_m_s", "log_hardness_MPa", "log_sliding_distance_m"]
PARTICLE_GRAPH_SELECTION = GraphSelection(boundary_texture=False)  # ParticleDataset: geometric view-0 graphs
_KEY_FILES = ("dataset.json", "particles.csv", "images.csv", "views.parquet")
_GLOBAL_PREFIXES = {"geometry": ("shape_", "fractal_"), "fecnn": ("shape_", "fractal_", "tex_")}
_VIEW_PREFIXES = ("shape_", "efd_", "fractal_")
_LUMA = np.array([0.2125, 0.7154, 0.0721], np.float32)  # skimage rgb2gray weights


# ============================================================================ feature cache


@dataclass
class FeatureCache:
    """Per-particle features of one dataset; array rows follow `particle_ids` (particles-table order).

    descriptors: particle_descriptors (primary view + native-patch texture/colour + multi-view summaries), index
    particle_id. view_descriptors: view_descriptors of every used view, index view_id, + particle_id / view_idx
    columns. crops: (n_p, H, W, 4) uint8 RGB + mask of view 0. contour_points: (n_p, n_nodes, 2) canonical view-0
    contour. contour_signal: (n_p, S, n_nodes) graphs.boundary_signals of it. graphs: pack_graphs of the FULL graph
    of every used view. meta: dataset/base/graph keys, their inputs, paths and what was rebuilt.
    """

    particle_ids: list[str]
    descriptors: pd.DataFrame
    view_descriptors: pd.DataFrame
    crops: np.ndarray
    contour_points: np.ndarray
    contour_signal: np.ndarray
    graphs: dict | None
    graph_cfg: GraphConfig | None
    meta: dict


def dataset_key(ds: WearDataset) -> str:
    """f"{safe(meta.name)}_{stable_hash(sha1 of dataset.json, particles.csv, images.csv, views.parquet)}"."""
    if ds.root is None:
        raise ValueError("dataset_key needs a saved dataset (ds.root is None)")
    digests = [hashlib.sha1((ds.root / f).read_bytes()).hexdigest() if (ds.root / f).exists() else ""
               for f in _KEY_FILES]
    return f"{safe_name(ds.meta.get('name', 'dataset'))}_{stable_hash(digests)}"


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as f:  # np.save(str) would append ".npy" to the temporary name
        np.save(f, arr)


def _used_views(ds: WearDataset, max_views: int, min_points: int) -> tuple[pd.DataFrame, dict]:
    """(views the cache describes, {view_id: cleaned contour (N, 2) px}): per particle the first `max_views` valid
    views (>= min_points distinct points, non-zero area) by view_idx; particles without a valid primary view are
    dropped (logged)."""
    v = ds.views.sort_values(["particle_id", "view_idx"], kind="stable")
    xy = {r.view_id: clean_closed(WearDataset.contour(r)) for r in v.itertuples()}
    ok = np.array([len(c) >= min_points and signed_area(c) != 0 for c in xy.values()], dtype=bool)
    v = v[ok].groupby("particle_id", sort=False).head(max_views)
    v = v[v.particle_id.isin(set(v.particle_id[v.view_idx == 0]))]
    n_drop = len(ds.particles) - v.particle_id.nunique()
    if n_drop:
        get_logger().warning("feature cache: %d particles without a valid primary contour dropped", n_drop)
    return v, xy


def _compute(ds: WearDataset, views: pd.DataFrame, xy: dict, d: dict, gcfg: GraphConfig, need_base: bool,
             need_graphs: bool) -> tuple[dict | None, dict | None]:
    """One pass over the images of `views` (each loaded once): FULL graphs of every view and, for primary views,
    descriptors + crop (native patch / crop of the view's own image). Particles whose features fail on a degenerate
    contour are dropped (logged). Returns (base dict or None, {view_id: graph} or None)."""
    efd, size, pad = int(d["efd_order"]), int(d["crop_size"]), float(d["crop_pad"])
    by_pid = {p: [(xy[v], u) for v, u in zip(g.view_id, g.um_per_px)] for p, g in views.groupby("particle_id")}
    paths = ds.images.set_index("image_id")["path"]
    rows, crops, graphs, failed = {}, {}, {}, set()
    for image_id, grp in tqdm(views.groupby("image_id", sort=True), desc="features", disable=None, ascii=True):
        img = load_image(ds.root / paths[image_id])
        gray = to_gray(img) if need_graphs else None
        for v in grp.itertuples():
            c = xy[v.view_id]
            try:
                if need_graphs:
                    graphs[v.view_id] = contour_graph(c, v.um_per_px, gcfg, gray)
                if need_base and v.view_idx == 0:
                    patch, mask = native_patch(img, c)
                    rows[v.particle_id] = particle_descriptors(by_pid[v.particle_id], patch, mask, efd)
                    crops[v.particle_id] = extract_crop(img, c, size, pad)
            except (ValueError, QhullError) as e:
                failed.add(v.particle_id)
                get_logger().warning("feature cache: particle %s dropped (%s)", v.particle_id, e)
    keep = set(views.particle_id) - failed
    if need_graphs:
        pid_of = dict(zip(views.view_id, views.particle_id))
        graphs = {v: g for v, g in graphs.items() if pid_of[v] in keep}
    if not need_base:
        return None, graphs
    pids = [p for p in ds.particles.particle_id if p in keep]
    vrows, vids = [], []
    for v in views[views.particle_id.isin(keep)].itertuples():
        vd = ({k: x for k, x in rows[v.particle_id].items() if k.startswith(_VIEW_PREFIXES)} if v.view_idx == 0
              else view_descriptors(xy[v.view_id], v.um_per_px, efd))
        vrows.append({"particle_id": v.particle_id, "view_idx": int(v.view_idx), **vd})
        vids.append(v.view_id)
    pos = [canonicalize(by_pid[p][0][0], gcfg.n_nodes)[0] for p in pids]
    base = {"descriptors": pd.DataFrame([rows[p] for p in pids], index=pd.Index(pids, name="particle_id")),
            "view_descriptors": pd.DataFrame(vrows, index=pd.Index(vids, name="view_id")),
            "crops": np.stack([crops[p] for p in pids]),
            "contour_points": np.stack(pos).astype(np.float32),
            "contour_signal": np.stack([boundary_signals(x, gcfg).T for x in pos]).astype(np.float32)}
    return base, graphs


_BASE_TABLES = ("descriptors", "view_descriptors")
_BASE_ARRAYS = ("crops", "contour_points", "contour_signal")


def _load_base(base_dir: Path, inputs: dict) -> dict | None:
    meta = base_dir / "meta.json"  # written last: present + matching inputs = complete
    if not meta.exists() or load_json(meta).get("key_inputs") != inputs:
        return None
    return {**{k: pd.read_parquet(base_dir / f"{k}.parquet") for k in _BASE_TABLES},
            **{k: np.load(base_dir / f"{k}.npy") for k in _BASE_ARRAYS}}


def build_cache(ds: WearDataset, cfg: dict, cache_dir=None, force: bool = False) -> FeatureCache:
    """Feature cache of `ds`, rebuilt only when missing, stale or forced.

    cache_dir (default cfg.paths.cache_dir)/<dataset_key>/base_<base_key>/ holds descriptors, view_descriptors,
    crops, contour points / signal and meta.json (written last; its key inputs must match, else rebuild);
    base_key = stable_hash(crop_size, crop_pad, efd_order, max_views, min_contour_points, graph.n_nodes,
    curvature_scales, roughness_window, FEATURE_VERSION). The ONE full graph cache of all ablations is
    cache_dir/<dataset_key>/graphs_<GraphConfig.key()>.pt ({"key_inputs", "packed"}; reused when its inputs match
    and it covers every used view). Each image is loaded once per build; files are written atomically.
    """
    t0 = time.perf_counter()
    d, gcfg = cfg["data"], GraphConfig.from_cfg(cfg)
    base_inputs = {"crop_size": int(d["crop_size"]), "crop_pad": float(d["crop_pad"]), "efd_order": int(d["efd_order"]),
                   "max_views": int(d["max_views"]), "min_contour_points": int(d["min_contour_points"]),
                   "n_nodes": int(gcfg.n_nodes), "curvature_scales": list(gcfg.curvature_scales),
                   "roughness_window": int(gcfg.roughness_window), "feature_version": FEATURE_VERSION}
    graph_inputs = {**asdict(gcfg), "feature_version": FEATURE_VERSION}
    dkey = dataset_key(ds)
    root = Path(cache_dir if cache_dir is not None else cfg["paths"]["cache_dir"]) / dkey
    base_dir, graph_path = root / f"base_{stable_hash(base_inputs)}", root / f"graphs_{gcfg.key()}.pt"

    base = None if force else _load_base(base_dir, base_inputs)
    packed = None
    if not force and graph_path.exists():
        blob = torch.load(graph_path, weights_only=True)
        packed = blob["packed"] if blob.get("key_inputs") == graph_inputs else None
    views, xy = _used_views(ds, base_inputs["max_views"], base_inputs["min_contour_points"])
    if base is not None:
        views = views[views.view_id.isin(base["view_descriptors"].index)]
    if views.empty:
        raise ValueError("build_cache: the dataset has no particle with a valid primary contour")
    if packed is not None and not set(views.view_id) <= set(packed["index"]):
        packed = None
    rebuilt = {"base": base is None, "graphs": packed is None}
    if base is None or packed is None:
        new_base, graphs = _compute(ds, views, xy, d, gcfg, base is None, packed is None)
        root.mkdir(parents=True, exist_ok=True)
        if base is None:
            base = new_base
            base_dir.mkdir(parents=True, exist_ok=True)
            for k in _BASE_TABLES:
                _atomic(base_dir / f"{k}.parquet", base[k].to_parquet)
            for k in _BASE_ARRAYS:
                _atomic(base_dir / f"{k}.npy", lambda p, a=base[k]: _save_npy(p, a))
            _atomic(base_dir / "meta.json", lambda p: save_json(
                {"key_inputs": base_inputs, "dataset_key": dkey, "n_particles": len(base["descriptors"]),
                 "n_views": len(base["view_descriptors"])}, p))
        if packed is None:
            used = set(base["view_descriptors"].index)
            packed = pack_graphs({v: g for v, g in graphs.items() if v in used})
            _atomic(graph_path, lambda p: torch.save({"key_inputs": graph_inputs, "packed": packed}, p))
    desc = base["descriptors"]
    meta = {"dataset_key": dkey, "base_dir": base_dir.as_posix(), "graph_path": graph_path.as_posix(),
            "base_inputs": base_inputs, "graph_inputs": graph_inputs, "rebuilt": rebuilt,
            "n_particles": len(desc), "n_views": len(base["view_descriptors"]),
            "seconds": round(time.perf_counter() - t0, 2)}
    get_logger().info("feature cache %s: %d particles, %d views, rebuilt %s in %.1f s", dkey, meta["n_particles"],
                      meta["n_views"], rebuilt, meta["seconds"])
    return FeatureCache(particle_ids=desc.index.tolist(), descriptors=desc, view_descriptors=base["view_descriptors"],
                        crops=base["crops"], contour_points=base["contour_points"],
                        contour_signal=base["contour_signal"], graphs=packed, graph_cfg=gcfg, meta=meta)


# ============================================================================ columns, scalers, indices


def particle_feature_columns(columns, cfg: dict) -> list[str]:
    """Proposed-model particle globals, in `columns` order: shape_* always; fractal_* if use_fractal_features;
    tex_* if use_texture_branch; color_* if use_composition; mv_* if use_multiview. Never efd_*."""
    m = cfg["model"]
    flags = (("shape_", True), ("fractal_", m["use_fractal_features"]), ("tex_", m["use_texture_branch"]),
             ("color_", m["use_composition"]), ("mv_", m["use_multiview"]))
    prefixes = tuple(p for p, on in flags if on)
    return [c for c in columns if str(c).startswith(prefixes)]


def baseline_global_columns(columns, kind: str = "geometry") -> list[str]:
    """Baseline globals, in `columns` order: "geometry" = shape_* + fractal_*; "fecnn" = + tex_*."""
    if kind not in _GLOBAL_PREFIXES:
        raise ValueError(f"kind must be one of {tuple(_GLOBAL_PREFIXES)}, got {kind!r}")
    return [c for c in columns if str(c).startswith(_GLOBAL_PREFIXES[kind])]


class Standardizer:
    """Column-wise z-score fitted on TRAIN rows of a DataFrame. NaN / inf -> the train median before scaling;
    zero-variance or all-NaN columns map to 0. transform(df) uses the statistics of df's columns (any subset of
    the fitted ones) and returns float32 (n, n_columns)."""

    def fit(self, df: pd.DataFrame) -> "Standardizer":
        x = df.astype(float).replace([np.inf, -np.inf], np.nan)
        self.median = x.median().fillna(0.0)
        x = x.fillna(self.median)
        self.mean = x.mean().fillna(0.0)
        sd = x.std(ddof=0)
        self.std = sd.where(sd > 1e-12, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = list(df.columns)
        x = df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(self.median[cols])
        return ((x - self.mean[cols]) / self.std[cols]).to_numpy(np.float32)


@dataclass
class Scalers:
    """particle: fitted on EVERY descriptor column (so any globals subset can be standardized); sample: on
    SAMPLE_FEATURES; cond: on COND_FEATURES; feature_columns = particle_feature_columns of the config."""

    particle: Standardizer
    sample: Standardizer
    cond: Standardizer
    feature_columns: list[str]


def _log(x) -> pd.Series:
    """ln of positive values, NaN otherwise (no warnings)."""
    x = pd.Series(x, dtype=float)
    return np.log(x.where(x > 0))


def _surviving(ds: WearDataset, cache: FeatureCache, run_ids, image_filter, role: str) -> pd.DataFrame:
    """Surviving particles of the runs in particles-table order (pass the role's image filter, present in the cache):
    index particle_id; columns row (cache row), sample_id, area [um^2] and log_ecd [ln um] of view 0."""
    row = pd.Series(np.arange(len(cache.particle_ids)), index=cache.particle_ids)
    pids = [p for p in particles_for_runs(ds, run_ids, image_filter, role) if p in row.index]
    desc = cache.descriptors.loc[pids]
    return pd.DataFrame({"row": row[pids].to_numpy(),
                         "sample_id": ds.particles.set_index("particle_id").loc[pids, "sample_id"].to_numpy(),
                         "area": desc.shape_area_um2.to_numpy(float), "log_ecd": desc.shape_log_ecd_um.to_numpy(float)},
                        index=pd.Index(pids, name="particle_id"))


def _sample_frame(ds: WearDataset, surv: pd.DataFrame) -> pd.DataFrame:
    """One row per OBSERVED sample (>= 1 surviving particle), sorted by (run_id, t_start_s), index sample_id:
    raw SAMPLE_FEATURES from the surviving particles, n_used, sum_a15 (sum A^1.5), log_ht (ln n_total / n_used, the
    HT factor of a full bag) and the physics inputs / targets."""
    g = surv.assign(a15=surv.area ** 1.5).groupby("sample_id")
    q = g.log_ecd.quantile([0.1, 0.5, 0.9]).unstack().reindex(columns=[0.1, 0.5, 0.9])  # columns even when empty
    s = ds.samples.set_index("sample_id").loc[q.index]
    res = s.run_id.map(ds.runs.set_index("run_id")["reservoir_volume_ml"])
    load = s.run_id.map(ds.runs.set_index("run_id")["load_N"])
    n_used = g.size().loc[q.index]
    n_total = np.maximum(s.n_particles_total.to_numpy(float), n_used.to_numpy(float))
    dist = s.sliding_distance_end_m - s.sliding_distance_start_m
    out = pd.DataFrame({
        "run_id": s.run_id, "t_start_s": s.t_start_s, "time_s": s.time_s, "distance_m": dist,
        "distance_end_m": s.sliding_distance_end_m,
        "log_n_total": np.log(n_total), "log_obs_fraction": _log(s.observation_fraction).to_numpy(),
        "log_ecd_q10": q[0.1], "log_ecd_q50": q[0.5], "log_ecd_q90": q[0.9],
        "log_total_area_um2": np.log(g.area.sum().loc[q.index]),
        "n_used": n_used, "sum_a15": g.a15.sum().loc[q.index], "log_ht": np.log(n_total / n_used.to_numpy(float)),
        "severity": s.severity, "log_q": _log(s.sampled_oil_ml / res).to_numpy(),
        "log10_rate": (_log(s.interval_wear_volume_mm3 / (load * dist)) / np.log(10)).to_numpy(),
    }, index=q.index)
    return out.sort_values(["run_id", "t_start_s"], kind="stable")


def _run_frame(ds: WearDataset, run_ids) -> pd.DataFrame:
    """index run_id (given order): run targets, physics inputs, indices and raw COND_FEATURES."""
    rt = ds.run_table().set_index("run_id").loc[list(run_ids)]
    mono, cond, mat, mag = mono_groups(ds), _condition_index(ds), material_index(ds), magnetic_index(ds)
    return pd.DataFrame({
        "log_grav_volume_mm3": _log(ds.grav_volume_mm3().reindex(rt.index)).to_numpy(),
        "log10_rate_run": (_log(rt.wear_rate_mm3_Nm) / np.log(10)).to_numpy(),
        "load_N": rt.load_N, "sliding_distance_m": rt.sliding_distance_m,
        "cumulative": rt.debris_sampling == "cumulative",
        "material_idx": rt.material_id.map(mat), "magnetic_idx": rt.material_id.map(mag),
        "mono_group": rt.index.map(mono), "condition_idx": rt.condition_id.map(cond),
        "log_load_N": _log(rt.load_N).to_numpy(), "log_speed_m_s": _log(rt.speed_m_s).to_numpy(),
        "log_hardness_MPa": _log(rt.hardness_MPa).to_numpy(),
        "log_sliding_distance_m": _log(rt.sliding_distance_m).to_numpy(),
    }, index=rt.index)


def fit_scalers(ds: WearDataset, cache: FeatureCache, train_runs, cfg: dict, image_filter=None) -> Scalers:
    """Standardizers fitted on TRAIN RUNS ONLY: particle = surviving train particles (role "train" of the filter),
    sample = their observed samples' SAMPLE_FEATURES, cond = the train runs' COND_FEATURES."""
    surv = _surviving(ds, cache, train_runs, image_filter, "train")
    return Scalers(particle=Standardizer().fit(cache.descriptors.loc[surv.index]),
                   sample=Standardizer().fit(_sample_frame(ds, surv)[SAMPLE_FEATURES]),
                   cond=Standardizer().fit(_run_frame(ds, train_runs)[COND_FEATURES]),
                   feature_columns=particle_feature_columns(cache.descriptors.columns, cfg))


def _mech_targets(ds: WearDataset, pids, cfg: dict) -> np.ndarray:
    """(n,) int64 class index per data.label_source (adjudicated | soft -> adjudicated, true -> synthetic truth)."""
    src = cfg["data"]["label_source"]
    src = "true" if src is True else src  # YAML reads an unquoted `true` as a boolean
    if src not in ("adjudicated", "soft", "true"):
        raise ValueError(f"data.label_source must be adjudicated | soft | true, got {src!r}")
    lab = ds.label_indices("true" if src == "true" else "adjudicated")
    return lab.reindex(list(pids)).fillna(-1).to_numpy(np.int64)


def class_weights(ds: WearDataset, particle_ids, cfg: dict) -> np.ndarray:
    """(C,) float32 from the labelled particles among `particle_ids` (pass TRAIN particles): train.class_weighting
    "balanced" -> w_c = n / (C_present n_c), 0 for absent classes; "none" -> ones."""
    mode, n_cls = cfg["train"]["class_weighting"], len(MECHANISMS)
    if mode == "none":
        return np.ones(n_cls, np.float32)
    if mode != "balanced":
        raise ValueError(f"train.class_weighting must be balanced | none, got {mode!r}")
    y = _mech_targets(ds, particle_ids, cfg)
    counts = np.bincount(y[y >= 0], minlength=n_cls).astype(float)
    w = np.zeros(n_cls)
    w[counts > 0] = counts.sum() / ((counts > 0).sum() * counts[counts > 0])
    return w.astype(np.float32)


def model_dims(ds: WearDataset, cache: FeatureCache, cfg: dict, scalers: Scalers) -> ModelDims:
    """Input dimensions of the proposed model for this config (graph selection, feature groups)."""
    node_dim, edge_dim = selected_dims(cache.graph_cfg, GraphSelection.from_cfg(cfg))
    return ModelDims(node_dim=node_dim, edge_dim=edge_dim, particle_feat_dim=len(scalers.feature_columns),
                     sample_feat_dim=len(SAMPLE_FEATURES), cond_dim=len(COND_FEATURES), n_classes=len(MECHANISMS),
                     n_severity=len(SEVERITY_LEVELS), n_elements=len(ELEMENTS), n_materials=len(material_index(ds)),
                     n_magnetic=2, crop_channels=int(cache.crops.shape[-1]), crop_size=int(cache.crops.shape[1]),
                     signal_channels=int(cache.contour_signal.shape[1]),
                     contour_points=int(cache.contour_points.shape[1]))


def material_index(ds: WearDataset) -> dict[str, int]:
    """material_id -> index into the sorted materials of the FULL dataset."""
    return {m: i for i, m in enumerate(sorted(ds.specimens.material_id.unique()))}


def magnetic_index(ds: WearDataset) -> dict[str, int]:
    """material_id -> magnetic-capture class (0 non-ferrous, 1 ferrous; any ferromagnetic specimen counts)."""
    return {m: int(f) for m, f in ds.specimens.groupby("material_id").ferromagnetic.max().items()}


def mono_groups(ds: WearDataset) -> dict[str, int]:
    """run_id -> group of runs comparable by load: key (material_pair, rig, lubricant_id, speed_m_s, duration_s,
    round(temperature_C, -1)) over the FULL dataset (NaN temperature is its own key value)."""
    rt = ds.run_table()
    key = rt[["material_pair", "rig", "lubricant_id", "speed_m_s", "duration_s"]].assign(temp=rt.temperature_C.round(-1))
    return dict(zip(rt.run_id, key.groupby(list(key.columns), dropna=False).ngroup()))


def _condition_index(ds: WearDataset) -> dict[str, int]:
    return {c: i for i, c in enumerate(sorted(ds.runs.condition_id.unique()))}


# ============================================================================ run bags


def _crop_batch(crops_u8: np.ndarray, rgb: bool, rng: np.random.Generator | None) -> np.ndarray:
    """(k, 4, H, W) float32 [0, 1] from (k, H, W, 4) uint8: luminance x3 + mask unless rgb; random dihedral
    transform per crop when rng is given (training)."""
    c = crops_u8.astype(np.float32) / 255.0
    if not rgb:
        c[..., :3] = (c[..., :3] @ _LUMA)[..., None]
    if rng is not None:
        out = []
        for x in c:
            x = np.rot90(x, int(rng.integers(4)))
            out.append(x[:, ::-1] if rng.random() < 0.5 else x)
        c = np.stack(out)
    return np.ascontiguousarray(c.transpose(0, 3, 1, 2))


class RunBagDataset(Dataset):
    """Item = one run with all its OBSERVED samples (samples with 0 surviving particles are dropped; runs without
    any observed sample are left out and logged, since a bag needs particles).

    Training bags subsample <= data.max_particles_per_sample particles per sample without replacement (seeded
    generator, new draws every epoch) with log_subsample_factor = ln[(sum_surv A^1.5 / sum_bag A^1.5)
    (n_total / n_surv)] (DESIGN sec. 3); evaluation uses every surviving particle. Views: every cached view
    (<= data.max_views) if model.use_multiview else view 0; graphs are GraphSelection.from_cfg(cfg) views of the
    cached full graphs. Attributes `run_ids`, `groups` (mono group) and `loads` follow the item order.
    """

    def __init__(self, ds: WearDataset, cache: FeatureCache, run_ids, cfg: dict, scalers: Scalers, train: bool,
                 seed: int = 0, image_filter=None, role: str = "train"):
        d, m = cfg["data"], cfg["model"]
        self.cache, self.train = cache, bool(train)
        self.rng = np.random.default_rng(seed)
        self.max_pp = int(d["max_particles_per_sample"])
        self.sel, self.gcfg = GraphSelection.from_cfg(cfg), cache.graph_cfg
        self.rgb, self.texture = bool(m["use_composition"]), bool(m["use_texture_branch"])
        self.augment = self.train and bool(d["augment"]["crop_rot90_flip"])

        surv = _surviving(ds, cache, run_ids, image_filter, role)
        smp = _sample_frame(ds, surv)
        observed = set(smp.run_id)
        runs = _run_frame(ds, [r for r in dict.fromkeys(run_ids) if r in observed])
        if len(runs) < len(set(run_ids)):
            get_logger().warning("RunBagDataset: %d of %d runs have no observed sample and are left out",
                                 len(set(run_ids)) - len(runs), len(set(run_ids)))

        self.pids = surv.index.tolist()
        p = ds.particles.set_index("particle_id").loc[self.pids]
        self.rows = surv.row.to_numpy()
        self.feats = scalers.particle.transform(cache.descriptors.loc[self.pids, scalers.feature_columns])
        self.comp = np.nan_to_num(p[list(EDS_COLUMNS)].to_numpy(float)).astype(np.float32)
        self.comp_mask = p.has_eds.to_numpy(bool)
        self.log_area = np.log(surv.area.to_numpy(float))
        self.a15 = surv.area.to_numpy(float) ** 1.5
        self.target = _mech_targets(ds, self.pids, cfg)
        self.soft = soft_labels(ds.labels, self.pids) if d["label_source"] == "soft" else None
        self.log_vol = _log(p.volume_um3).to_numpy()
        vlists = cache.view_descriptors.reset_index().groupby("particle_id", sort=False).view_id.agg(list)
        self.views = [vlists[q] if m["use_multiview"] else vlists[q][:1] for q in self.pids]

        self.smp = smp
        self.smp_feats = scalers.sample.transform(smp[SAMPLE_FEATURES])
        members = surv.groupby("sample_id").indices  # sample_id -> positions in surv (ascending)
        self.members = [members[s] for s in smp.index]
        self.run_samples = [np.flatnonzero(smp.run_id.to_numpy() == r) for r in runs.index]
        self.runs = runs
        self.cond = scalers.cond.transform(runs[COND_FEATURES])
        self.run_ids = runs.index.tolist()
        self.groups = runs.mono_group.to_numpy(int)
        self.loads = runs.load_N.to_numpy(float)

    def __len__(self) -> int:
        return len(self.run_ids)

    def __getitem__(self, i: int) -> dict:
        bags, p2s, log_ht = [], [], []
        for k, s in enumerate(self.run_samples[i]):
            mem = self.members[s]
            ht = float(self.smp.log_ht.iat[s])
            if self.train and len(mem) > self.max_pp:
                bag = np.sort(self.rng.choice(mem, self.max_pp, replace=False))
                ht += float(np.log(self.a15[mem].sum() / self.a15[bag].sum()))
            else:
                bag = mem
            bags.append(bag)
            p2s.append(np.full(len(bag), k))
            log_ht.append(ht)
        pp = np.concatenate(bags)
        graphs, v2p = [], []
        for j, q in enumerate(pp):
            for vid in self.views[q]:
                graphs.append(select_graph(unpack_graph(self.cache.graphs, vid), self.gcfg, self.sel))
                v2p.append(j)
        s, r = self.smp.iloc[self.run_samples[i]], self.runs.iloc[i]
        return {
            "graphs": graphs, "view_to_particle": np.array(v2p),
            "crops": (_crop_batch(self.cache.crops[self.rows[pp]], self.rgb, self.rng if self.augment else None)
                      if self.texture else None),
            "particle_feats": self.feats[pp], "composition": self.comp[pp], "composition_mask": self.comp_mask[pp],
            "log_area_um2": self.log_area[pp], "mech_target": self.target[pp],
            "mech_soft": None if self.soft is None else self.soft[pp], "log_volume_target": self.log_vol[pp],
            "particle_to_sample": np.concatenate(p2s),
            "log_obs_fraction": s.log_obs_fraction.to_numpy(), "log_subsample_factor": np.array(log_ht),
            "severity_target": s.severity.to_numpy(), "log10_rate_sample_target": s.log10_rate.to_numpy(),
            "sample_time_s": s.time_s.to_numpy(), "sample_distance_m": s.distance_m.to_numpy(),
            "sample_distance_end_m": s.distance_end_m.to_numpy(),
            "sample_log_q": s.log_q.to_numpy(), "sample_feats": self.smp_feats[self.run_samples[i]],
            "log_grav_volume_mm3": r.log_grav_volume_mm3, "log10_rate_run_target": r.log10_rate_run,
            "load_N": r.load_N, "sliding_distance_m": r.sliding_distance_m, "run_cond_feats": self.cond[i],
            "material_idx": r.material_idx, "magnetic_idx": r.magnetic_idx, "run_cumulative": r.cumulative,
            "mono_group": r.mono_group, "condition_idx": r.condition_idx,
            "particle_ids": [self.pids[q] for q in pp], "sample_ids": s.index.tolist(), "run_id": self.run_ids[i],
        }


def collate_runs(items: list[dict]) -> RunBatch:
    """Concatenate RunBagDataset items into a RunBatch; local view/particle/sample indices are offset so that they
    index the batch."""
    def cat(key, dtype=torch.float32):
        return torch.from_numpy(np.concatenate([it[key] for it in items])).to(dtype)

    def offset(key, counts):
        o = np.cumsum([0, *counts[:-1]])
        return torch.from_numpy(np.concatenate([it[key] + k for it, k in zip(items, o)])).long()

    def per_run(key, dtype=torch.float32):
        return torch.tensor(np.array([it[key] for it in items]), dtype=dtype)

    n_p = [len(it["particle_ids"]) for it in items]
    n_s = [len(it["sample_ids"]) for it in items]
    return RunBatch(
        graphs=Batch.from_data_list([g for it in items for g in it["graphs"]]),
        view_to_particle=offset("view_to_particle", n_p),
        crops=None if items[0]["crops"] is None else cat("crops"),
        particle_feats=cat("particle_feats"), composition=cat("composition"),
        composition_mask=cat("composition_mask", torch.bool), log_area_um2=cat("log_area_um2"),
        mech_target=cat("mech_target", torch.long),
        mech_soft=None if items[0]["mech_soft"] is None else cat("mech_soft"),
        log_volume_target=cat("log_volume_target"), particle_to_sample=offset("particle_to_sample", n_s),
        sample_to_run=torch.from_numpy(np.repeat(np.arange(len(items)), n_s)).long(),
        log_obs_fraction=cat("log_obs_fraction"), log_subsample_factor=cat("log_subsample_factor"),
        severity_target=cat("severity_target", torch.long), log10_rate_sample_target=cat("log10_rate_sample_target"),
        sample_time_s=cat("sample_time_s"), sample_distance_m=cat("sample_distance_m"),
        sample_distance_end_m=cat("sample_distance_end_m"),
        sample_log_q=cat("sample_log_q"), sample_feats=cat("sample_feats"),
        log_grav_volume_mm3=per_run("log_grav_volume_mm3"), log10_rate_run_target=per_run("log10_rate_run_target"),
        load_N=per_run("load_N"), sliding_distance_m=per_run("sliding_distance_m"),
        run_cond_feats=torch.from_numpy(np.stack([it["run_cond_feats"] for it in items])),
        material_idx=per_run("material_idx", torch.long), magnetic_idx=per_run("magnetic_idx", torch.long),
        run_cumulative=per_run("run_cumulative", torch.bool), mono_group=per_run("mono_group", torch.long),
        condition_idx=per_run("condition_idx", torch.long),
        particle_ids=[p for it in items for p in it["particle_ids"]],
        sample_ids=[s for it in items for s in it["sample_ids"]], run_ids=[it["run_id"] for it in items])


class GroupedRunSampler(Sampler):
    """Training batch sampler: every batch holds <= batch_runs runs of ONE mono group, taken round-robin over the
    group's loads (so a batch spans >= 2 loads whenever the group still has them); runs, loads and batch order are
    reshuffled every epoch from a seeded generator. Yields lists of dataset indices."""

    def __init__(self, groups, loads, batch_runs: int, seed: int = 0):
        self.groups, self.loads = np.asarray(groups), np.asarray(loads, dtype=float)
        self.batch_runs = int(batch_runs)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        batches = []
        for g in np.unique(self.groups):
            idx = np.flatnonzero(self.groups == g)
            by_load = [self.rng.permutation(idx[self.loads[idx] == ld]).tolist()
                       for ld in self.rng.permutation(np.unique(self.loads[idx]))]
            order = [i for tup in zip_longest(*by_load) for i in tup if i is not None]
            batches += [order[k:k + self.batch_runs] for k in range(0, len(order), self.batch_runs)]
        return iter([batches[k] for k in self.rng.permutation(len(batches))])

    def __len__(self) -> int:
        return int(sum(-(-np.sum(self.groups == g) // self.batch_runs) for g in np.unique(self.groups)))


def make_run_loader(ds: WearDataset, cache: FeatureCache, run_ids, cfg: dict, scalers: Scalers, train: bool,
                    seed: int = 0, image_filter=None, role: str = "train") -> DataLoader:
    """train: GroupedRunSampler with train.batch_runs; eval: train.eval_batch_runs runs per batch in order."""
    dset = RunBagDataset(ds, cache, run_ids, cfg, scalers, train, seed, image_filter, role)
    t = cfg["train"]
    if train:
        return DataLoader(dset, batch_sampler=GroupedRunSampler(dset.groups, dset.loads, t["batch_runs"], seed),
                          collate_fn=collate_runs, num_workers=0)
    return DataLoader(dset, batch_size=int(t["eval_batch_runs"]), shuffle=False, collate_fn=collate_runs,
                      num_workers=0)


# ============================================================================ particles (baselines)


class ParticleDataset(Dataset):
    """Labelled particles (data.label_source; unlabelled and uncached ids are skipped) for particle classifiers:
    view-0 graph in PARTICLE_GRAPH_SELECTION (geometric: no tex_* columns, so no interior grey), canonical contour
    points (point rotation augmentation is the PointNet model's job), contour signal, crop (dihedral augmentation
    when training and data.augment.crop_rot90_flip; RGB if model.use_composition else luminance) and baseline
    globals baseline_global_columns(kind=globals_kind) standardized with the TRAIN statistics of scalers.particle."""

    def __init__(self, ds: WearDataset, cache: FeatureCache, particle_ids, cfg: dict, scalers: Scalers, train: bool,
                 seed: int = 0, globals_kind: str = "geometry"):
        d = cfg["data"]
        self.cache, self.rng = cache, np.random.default_rng(seed)
        self.augment = bool(train) and bool(d["augment"]["crop_rot90_flip"])
        self.rgb = bool(cfg["model"]["use_composition"])
        row = pd.Series(np.arange(len(cache.particle_ids)), index=cache.particle_ids)
        pids = [q for q in particle_ids if q in row.index]
        target = _mech_targets(ds, pids, cfg)
        self.pids = [q for q, t in zip(pids, target) if t >= 0]
        self.target = target[target >= 0]
        self.rows = row[self.pids].to_numpy()
        cols = baseline_global_columns(cache.descriptors.columns, globals_kind)
        self.feats = scalers.particle.transform(cache.descriptors.loc[self.pids, cols])
        p = ds.particles.set_index("particle_id").loc[self.pids]
        self.comp = np.nan_to_num(p[list(EDS_COLUMNS)].to_numpy(float)).astype(np.float32)
        self.comp_mask = p.has_eds.to_numpy(bool)
        self.soft = soft_labels(ds.labels, self.pids).astype(np.float32) if d["label_source"] == "soft" else None

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, i: int) -> dict:
        r, pid = self.rows[i], self.pids[i]
        graph = select_graph(unpack_graph(self.cache.graphs, f"{pid}_v0"), self.cache.graph_cfg,
                             PARTICLE_GRAPH_SELECTION)
        return {"graph": graph, "points": self.cache.contour_points[r], "signal": self.cache.contour_signal[r],
                "crop": _crop_batch(self.cache.crops[r:r + 1], self.rgb, self.rng if self.augment else None)[0],
                "feats": self.feats[i], "composition": self.comp[i], "composition_mask": self.comp_mask[i],
                "target": self.target[i], "soft": None if self.soft is None else self.soft[i], "particle_id": pid}


def collate_particles(items: list[dict]) -> ParticleBatch:
    def stack(key, dtype=torch.float32):
        return torch.from_numpy(np.stack([it[key] for it in items])).to(dtype)

    return ParticleBatch(
        graphs=Batch.from_data_list([it["graph"] for it in items]), view_to_particle=torch.arange(len(items)),
        contour_points=stack("points"), contour_signal=stack("signal"), crops=stack("crop"),
        particle_feats=stack("feats"), composition=stack("composition"),
        composition_mask=stack("composition_mask", torch.bool), mech_target=stack("target", torch.long),
        mech_soft=None if items[0]["soft"] is None else stack("soft"),
        particle_ids=[it["particle_id"] for it in items])


def make_particle_loader(ds: WearDataset, cache: FeatureCache, particle_ids, cfg: dict, scalers: Scalers, train: bool,
                         seed: int = 0, batch_size: int | None = None, globals_kind: str = "geometry") -> DataLoader:
    """Shuffled (seeded) when training; batch size train.particle_batch_size unless given."""
    dset = ParticleDataset(ds, cache, particle_ids, cfg, scalers, train, seed, globals_kind)
    return DataLoader(dset, batch_size=int(batch_size or cfg["train"]["particle_batch_size"]), shuffle=bool(train),
                      generator=torch.Generator().manual_seed(seed) if train else None,
                      collate_fn=collate_particles, num_workers=0)
