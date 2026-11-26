"""Fractal descriptors of closed contours given in PIXELS.

Scale rule shared by both estimators: L = max Feret diameter [px] (max projection extent over 64 directions,
relative error <= 3e-4, avoiding a Qhull call); scales log-spaced in [max(2 px, min_frac L), max_frac L] (the 2-px
floor keeps the pixel staircase out of the fit). An estimate needs >= 4 scales spanning >= 2x (i.e. L >= 40 px with
the defaults); otherwise {"D": 1.0, "r2": nan, "valid": False}. Both estimators start at the max-radius vertex at
positive orientation: exactly invariant to translation and start point, and to scale once 0.01 L >= 2 px; rotation
changes the divider D only through the L approximation (~1e-3) and the box grid by more (box counting is not
rotation invariant).
"""

from __future__ import annotations

import math
from bisect import bisect_left

import numpy as np

from wearpg.features.contour import (_max_radius_index, clean_closed, ensure_ccw, perimeter, resample_closed,
                                     smooth_closed)

_INVALID = {"D": 1.0, "r2": float("nan"), "valid": False}
_MS_POINTS = 256  # resampling for the scale-free multiscale perimeter (sigma unit = perimeter / 256)
_ANG = np.linspace(0.0, np.pi, 64, endpoint=False)
_DIRS = np.stack([np.cos(_ANG), np.sin(_ANG)])  # (2, 64) unit directions for the max Feret


def _prepare(xy_px, n: int, min_frac: float, max_frac: float):
    """(contour starting at the max-radius vertex, scales [px]) or (contour, None) if the scale rule fails."""
    xy = ensure_ccw(clean_closed(xy_px))
    xy = np.roll(xy, -_max_radius_index(xy), axis=0)
    proj = xy @ _DIRS
    big = float((proj.max(axis=0) - proj.min(axis=0)).max())
    lo, hi = max(2.0, min_frac * big), max_frac * big
    return xy, (np.geomspace(lo, hi, n) if n >= 4 and hi >= 2 * lo else None)


def _loglog_fit(x, y) -> tuple[float, float]:
    """(slope, r2) of the least-squares line through (x, y) (r2 = 1 for a perfectly flat y)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    dx, dy = x - x.mean(), y - y.mean()
    slope = float(dx @ dy / (dx @ dx))
    ss_tot = float(dy @ dy)
    return slope, (1.0 - float(np.sum((dy - slope * dx) ** 2)) / ss_tot) if ss_tot > 0 else 1.0


def _divider_length(xs: list, ys: list, arc: list, step: float) -> float:
    """Walking-divider length [px] of the closed polyline from vertex 0 (xs/ys/arc repeat vertex 0 at the end;
    arc = cumulative length): full steps * step + the final partial chord back to the start. Each new point is the
    first crossing of the circle of radius `step` FORWARD along the contour (never across necks)."""
    n = len(xs) - 1
    s2 = step * step
    cx, cy, cu = xs[0], ys[0], 0.0  # current divider point and its arc position
    px, py, pu = cx, cy, cu         # start of the sub-segment being searched (inside the circle)
    k, count = 1, 0
    while True:
        j = bisect_left(arc, cu + step, k)  # vertices k..j-1 are < step away along the arc, hence also by chord
        if j > k:
            k = j
            px, py, pu = xs[k - 1], ys[k - 1], arc[k - 1]
        while k <= n:
            vx, vy = xs[k], ys[k]
            if (vx - cx) ** 2 + (vy - cy) ** 2 >= s2:
                break
            px, py, pu = vx, vy, arc[k]
            k += 1
        else:
            return count * step + math.hypot(xs[0] - cx, ys[0] - cy)
        # positive root of |p + t d - c|^2 = step^2 on the segment p -> v
        dx, dy, fx, fy = vx - px, vy - py, px - cx, py - cy
        a, b, c = dx * dx + dy * dy, fx * dx + fy * dy, fx * fx + fy * fy - s2
        t = (-b + math.sqrt(max(b * b - a * c, 0.0))) / a
        cx, cy, cu = px + t * dx, py + t * dy, pu + t * math.sqrt(a)
        px, py, pu = cx, cy, cu
        count += 1


def divider_dimension(xy_px, n_steps: int = 10, min_frac: float = 0.01, max_frac: float = 0.10) -> dict:
    """Richardson walking-divider dimension D = 1 - slope(ln length vs ln step); {"D", "r2", "valid"}."""
    xy, steps = _prepare(xy_px, n_steps, min_frac, max_frac)
    if steps is None:
        return dict(_INVALID)
    closed = np.vstack([xy, xy[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]).tolist()
    xs, ys = closed[:, 0].tolist(), closed[:, 1].tolist()
    lengths = np.array([_divider_length(xs, ys, arc, s) for s in steps])
    slope, r2 = _loglog_fit(np.log(steps), np.log(lengths))
    return {"D": 1.0 - slope, "r2": r2, "valid": True}


def box_counting_dimension(xy_px, n_scales: int = 10) -> dict:
    """Box-counting dimension of the boundary curve, D = -slope(ln N(eps) vs ln eps), boxes anchored at the contour's
    bounding-box corner; same scale rule as the divider. {"D", "r2", "valid"}."""
    xy, eps = _prepare(xy_px, n_scales, 0.01, 0.10)
    if eps is None:
        return dict(_INVALID)
    pts = resample_closed(xy, int(math.ceil(4 * perimeter(xy) / eps[0])) + 8, start="first")  # spacing <= eps/4
    pts -= pts.min(axis=0)
    counts = []
    for e in eps:
        ij = np.floor(pts / e).astype(np.int64)
        counts.append(np.unique(ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]).size)
    slope, r2 = _loglog_fit(np.log(eps), np.log(counts))
    return {"D": -slope, "r2": r2, "valid": True}


def multiscale_perimeter_slopes(xy, sigmas=(1, 2, 4, 8, 16)) -> np.ndarray:
    """(len(sigmas) - 1,) slopes d ln P / d ln sigma of the Gaussian-smoothed perimeter. The contour is first
    resampled to 256 points, so sigma is a fraction of the perimeter: similarity invariant (units irrelevant)."""
    r = resample_closed(xy, _MS_POINTS)
    p = np.array([perimeter(smooth_closed(r, s)) for s in sigmas])
    return np.diff(np.log(p)) / np.diff(np.log(np.asarray(sigmas, dtype=np.float64)))


def fractal_descriptors(xy_px) -> dict:
    """fractal_divider_D/_r2, fractal_box_D/_r2, fractal_valid (1.0 iff both valid), fractal_ms_0..3. No NaN
    (an invalid estimate gives D = 1, r2 = 0)."""
    d, b = divider_dimension(xy_px), box_counting_dimension(xy_px)
    out = {
        "fractal_divider_D": float(d["D"]),
        "fractal_divider_r2": float(np.nan_to_num(d["r2"])),
        "fractal_box_D": float(b["D"]),
        "fractal_box_r2": float(np.nan_to_num(b["r2"])),
        "fractal_valid": float(d["valid"] and b["valid"]),
    }
    out.update({f"fractal_ms_{i}": float(v) for i, v in enumerate(multiscale_perimeter_slopes(xy_px))})
    return out
