"""Batch containers exchanged between datasets.py (producer) and models/training (consumers).

Index tensors are LOCAL to the batch (0..n-1). Shapes use:
    n_v = number of view graphs, n_p = particles, n_s = samples (oil samples), n_r = runs,
    C = mechanism classes, E = len(schema.ELEMENTS).
Targets use NaN (float) or -1 (long) for "unknown"; losses must mask them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class ModelDims:
    """Input dimensions the model needs; produced by datasets.model_dims()."""

    node_dim: int
    edge_dim: int
    particle_feat_dim: int
    sample_feat_dim: int
    cond_dim: int
    n_classes: int
    n_severity: int = 3
    n_elements: int = 11
    n_materials: int = 1
    n_magnetic: int = 2       # magnetic-capture classes (0 non-ferrous, 1 ferrous)
    crop_channels: int = 4  # RGB + mask
    crop_size: int = 64
    signal_channels: int = 0  # ParticleBatch.contour_signal channels (baselines)
    contour_points: int = 0   # ParticleBatch.contour_points length (baselines)


def _move(obj, device, non_blocking: bool):
    if obj is None:
        return None
    if isinstance(obj, Tensor) or hasattr(obj, "to"):
        return obj.to(device, non_blocking=non_blocking)
    return obj


@dataclass
class RunBatch:
    """A batch of whole runs: every run carries all its samples; each sample a particle bag."""

    # ---- view graphs (PyG Batch: x, edge_index, edge_attr, batch, pos, node_type)
    graphs: object                      # torch_geometric.data.Batch with n_v graphs
    view_to_particle: Tensor            # (n_v,) long
    # ---- particle level
    crops: Tensor | None                # (n_p, 4, H, W) float [0,1]: RGB + mask
    particle_feats: Tensor              # (n_p, F_p) float, standardized handcrafted globals
    composition: Tensor                 # (n_p, E) float EDS fractions (0 where unknown)
    composition_mask: Tensor            # (n_p,) bool, True where EDS known
    log_area_um2: Tensor                # (n_p,) ln area of primary view [um^2]
    mech_target: Tensor                 # (n_p,) long, -1 unlabeled
    mech_soft: Tensor | None            # (n_p, C) rater vote distribution (rows sum 1) or None
    log_volume_target: Tensor           # (n_p,) ln measured volume [um^3], NaN unknown
    particle_to_sample: Tensor          # (n_p,) long
    # ---- sample level
    sample_to_run: Tensor               # (n_s,) long
    log_obs_fraction: Tensor            # (n_s,) ln(observation_fraction)
    log_subsample_factor: Tensor        # (n_s,) ln HT_s, area-ratio estimator (DESIGN sec. 3), n_used >= 1
    severity_target: Tensor             # (n_s,) long, -1 unknown
    log10_rate_sample_target: Tensor    # (n_s,) log10 interval specific wear rate, NaN unknown
    sample_time_s: Tensor               # (n_s,) float
    sample_distance_m: Tensor           # (n_s,) interval sliding distance [m] (end - start)
    sample_distance_end_m: Tensor       # (n_s,) cumulative sliding distance at t_end [m] (cumulative-mode s_end)
    sample_log_q: Tensor                # (n_s,) ln(sampled_oil_ml / reservoir_volume_ml)
    sample_feats: Tensor                # (n_s, F_s) float standardized bag context
    # ---- run level
    log_grav_volume_mm3: Tensor         # (n_r,) ln measured run wear volume [mm^3], NaN unknown
    log10_rate_run_target: Tensor       # (n_r,) log10 specific wear rate [mm^3/(N m)], NaN unknown
    load_N: Tensor                      # (n_r,) float
    sliding_distance_m: Tensor          # (n_r,) float
    run_cond_feats: Tensor              # (n_r, F_c) standardized operating conditions
    material_idx: Tensor                # (n_r,) long index into the dataset's sorted material list
    magnetic_idx: Tensor                # (n_r,) long magnetic-capture class of the run's material
    run_cumulative: Tensor              # (n_r,) bool, True for debris_sampling == 'cumulative'
    mono_group: Tensor                  # (n_r,) long; runs comparable by load share a group (load excluded)
    condition_idx: Tensor               # (n_r,) long index of condition_id (monotone loss uses condition means)
    # ---- ids (not moved to device)
    particle_ids: list[str]
    sample_ids: list[str]
    run_ids: list[str]

    @property
    def n_particles(self) -> int:
        return int(self.particle_to_sample.numel())

    @property
    def n_samples(self) -> int:
        return int(self.sample_to_run.numel())

    @property
    def n_runs(self) -> int:
        return int(self.load_N.numel())

    def to(self, device, non_blocking: bool = False) -> "RunBatch":
        return replace(self, **{f.name: _move(getattr(self, f.name), device, non_blocking)
                                for f in fields(self) if not f.name.endswith("_ids")})


@dataclass
class ParticleBatch:
    """Particle-level batch for particle classifiers (deep baselines)."""

    graphs: object | None               # PyG Batch of view graphs or None
    view_to_particle: Tensor | None     # (n_v,) long or None
    contour_points: Tensor              # (n_p, N, 2) canonical normalized primary-view contour
    contour_signal: Tensor              # (n_p, S, N) invariant 1D signals along the contour
    crops: Tensor                       # (n_p, 4, H, W) float [0,1]
    particle_feats: Tensor              # (n_p, F_p) standardized handcrafted globals
    composition: Tensor                 # (n_p, E)
    composition_mask: Tensor            # (n_p,) bool
    mech_target: Tensor                 # (n_p,) long, -1 unlabeled
    mech_soft: Tensor | None            # (n_p, C) or None
    particle_ids: list[str]

    @property
    def n_particles(self) -> int:
        return int(self.mech_target.numel())

    def to(self, device, non_blocking: bool = False) -> "ParticleBatch":
        return replace(self, **{f.name: _move(getattr(self, f.name), device, non_blocking)
                                for f in fields(self) if not f.name.endswith("_ids")})
