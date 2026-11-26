"""Size and shape descriptors of one particle view."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from wearpg.features.contour import area_moments, clean_closed, ensure_ccw, perimeter


def shape_descriptors(xy_px, um_per_px: float) -> dict:
    """Shape descriptors of a closed contour given in px; lengths in um, areas in um^2, ratios dimensionless.

    major/minor = full axes of the ellipse with the same area second moments; aspect_ratio = major/minor;
    roundness = 4 pi A / P^2; solidity = A / A_hull; convexity = P_hull / P; eccentricity of that ellipse;
    Feret max / min = caliper diameters; elongation = Feret max / Feret min; curl = Feret max / fibre length
    (rectangle-equivalent length (P + sqrt(P^2 - 16 A)) / 4, ISO 9276-6 style), clipped to <= 1 (1 = straight or
    compact, < 1 = curled ribbon).
    """
    xy = ensure_ccw(clean_closed(xy_px)) * float(um_per_px)
    area, _, cov = area_moments(xy)
    if not area > 0:
        raise ValueError("degenerate contour (zero area)")
    per = perimeter(xy)
    hull = ConvexHull(xy)  # 2-D: .volume = area, .area = perimeter; Qhull is the slow part, call it once
    h = xy[hull.vertices]  # counter-clockwise
    e = np.roll(h, -1, axis=0) - h
    proj = h @ (np.column_stack([e[:, 1], -e[:, 0]]) / np.linalg.norm(e, axis=1, keepdims=True)).T
    fmax, fmin = float(pdist(h).max()), float((proj.max(0) - proj.min(0)).min())  # min width is edge-flush
    lmin, lmax = np.linalg.eigvalsh(cov)
    major, minor = 4 * np.sqrt(lmax), 4 * np.sqrt(max(lmin, 0.0))
    ecd = np.sqrt(4 * area / np.pi)
    fibre = (per + np.sqrt(max(per * per - 16 * area, 0.0))) / 4
    return {
        "shape_area_um2": float(area),
        "shape_log_area_um2": float(np.log(area)),
        "shape_perimeter_um": per,
        "shape_ecd_um": float(ecd),
        "shape_log_ecd_um": float(np.log(ecd)),
        "shape_major_um": float(major),
        "shape_minor_um": float(minor),
        "shape_aspect_ratio": float(major / minor),
        "shape_roundness": float(4 * np.pi * area / per ** 2),
        "shape_solidity": float(area / hull.volume),
        "shape_convexity": float(hull.area / per),
        "shape_eccentricity": float(np.sqrt(max(1.0 - lmin / lmax, 0.0))),
        "shape_feret_max_um": fmax,
        "shape_feret_min_um": fmin,
        "shape_elongation": fmax / fmin,
        "shape_curl": float(min(1.0, fmax / fibre)),
    }
