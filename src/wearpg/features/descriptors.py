"""Per-view and per-particle handcrafted descriptor dictionaries (all values float, never NaN)."""

from __future__ import annotations

import numpy as np

from wearpg.features.fourier import efd_feature_vector
from wearpg.features.fractal import fractal_descriptors
from wearpg.features.shape import shape_descriptors
from wearpg.features.texture import color_descriptors, texture_descriptors

DESCRIPTOR_PREFIXES = ("shape_", "efd_", "fractal_", "tex_", "color_", "mv_")


def view_descriptors(xy_px, um_per_px: float, efd_order: int = 10) -> dict:
    """shape_* + efd_0..efd_{4 order - 4} + fractal_* of one view (contour in px of its own image)."""
    out = shape_descriptors(xy_px, um_per_px)
    out.update({f"efd_{i}": float(v) for i, v in enumerate(efd_feature_vector(xy_px, efd_order))})
    out.update(fractal_descriptors(xy_px))
    return out


def particle_descriptors(views: list[tuple[np.ndarray, float]], patch_rgb, patch_mask, efd_order: int = 10) -> dict:
    """Descriptors of one particle.

    views: [(contour_px, um_per_px), ...] with the primary view first. Returns view_descriptors of view 0,
    texture/colour of the native view-0 patch, and multi-view summaries over all views (areas compared in um^2
    since views can have different pixel sizes): mv_n_views, mv_aspect_min/max/std (population std),
    mv_area_ratio_min = min area / max area, mv_roundness_mean/min. A single view gives the view-0 values, std 0 and
    area ratio 1.
    """
    out = view_descriptors(views[0][0], views[0][1], efd_order)
    out.update(texture_descriptors(patch_rgb, patch_mask))
    out.update(color_descriptors(patch_rgb, patch_mask))
    shapes = [out] + [shape_descriptors(xy, u) for xy, u in views[1:]]
    asp = np.array([s["shape_aspect_ratio"] for s in shapes])
    area = np.array([s["shape_area_um2"] for s in shapes])
    rnd = np.array([s["shape_roundness"] for s in shapes])
    out.update({
        "mv_n_views": float(len(views)),
        "mv_aspect_min": float(asp.min()),
        "mv_aspect_max": float(asp.max()),
        "mv_aspect_std": float(asp.std()),
        "mv_area_ratio_min": float(area.min() / area.max()),
        "mv_roundness_mean": float(rnd.mean()),
        "mv_roundness_min": float(rnd.min()),
    })
    return out
