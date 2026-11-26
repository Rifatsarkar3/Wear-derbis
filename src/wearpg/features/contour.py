"""Closed-contour geometry primitives.

A contour is an (N, 2) float array of (x, y) points, closed implicitly (no repeated end point). "Positive"
orientation = positive shoelace area in (x, y) as given (counter-clockwise in a y-up frame, i.e. clockwise on
screen for image coordinates). Functions promising a sign (curvature, turning angle, normals) correct for the input
orientation themselves. Lengths are in input units (px or um), angles in radians.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from skimage.draw import polygon2mask


def clean_closed(xy) -> np.ndarray:
    """(N, 2) float64 copy without the repeated end point and without consecutive duplicate points."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    keep = np.any(xy != np.roll(xy, -1, axis=0), axis=1)  # point differs from its successor
    return xy[keep] if keep.any() else xy[:1].copy()


def signed_area(xy) -> float:
    """Shoelace area [units^2]; > 0 for counter-clockwise order in (x, y) as given."""
    x, y = np.asarray(xy, dtype=np.float64).T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def ensure_ccw(xy) -> np.ndarray:
    """Copy with positive shoelace area (reversed if needed, first point kept)."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.roll(xy[::-1], 1, axis=0) if signed_area(xy) < 0 else xy.copy()


def _orientation(xy) -> float:
    return 1.0 if signed_area(xy) >= 0 else -1.0


def perimeter(xy) -> float:
    """Length of the closed polygon [units]."""
    xy = np.asarray(xy, dtype=np.float64)
    return float(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1).sum())


def area_moments(xy) -> tuple[float, np.ndarray, np.ndarray]:
    """(signed area [units^2], area centroid (2,), central second-moment covariance (2, 2) [units^2]).

    Exact polygon integrals (Green's theorem); centroid and covariance do not depend on orientation or sampling.
    Zero area gives (0, vertex mean, zeros).
    """
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = cr.sum() / 2
    if a == 0:  # degenerate (collinear): callers raise on area <= 0
        return 0.0, xy.mean(axis=0), np.zeros((2, 2))
    c = np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (6 * a)
    x, y, xn, yn = x - c[0], y - c[1], xn - c[0], yn - c[1]
    cr = x * yn - xn * y  # cross terms are not translation invariant: recompute about the centroid
    sxx = ((x * x + x * xn + xn * xn) * cr).sum() / 12
    syy = ((y * y + y * yn + yn * yn) * cr).sum() / 12
    sxy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cr).sum() / 24
    return float(a), c, np.array([[sxx, sxy], [sxy, syy]]) / a


def _max_radius_index(xy) -> int:
    return int(np.argmax(np.sum((xy - area_moments(xy)[1]) ** 2, axis=1)))


def _rms_radius(rel) -> float:
    """Arc-length weighted RMS distance of the polygon boundary from the origin (exact per straight segment)."""
    nxt = np.roll(rel, -1, axis=0)
    seg = np.linalg.norm(nxt - rel, axis=1)
    m2 = (np.sum(rel * rel, 1) + np.sum(rel * nxt, 1) + np.sum(nxt * nxt, 1)) / 3
    return float(np.sqrt(np.dot(seg, m2) / seg.sum()))


def resample_closed(xy, n: int, start: str = "canonical") -> np.ndarray:
    """(n, 2) points at uniform arc length, positive area.

    start "canonical": the vertex farthest from the area centroid (rotation / start-point invariant);
    "first": the first input point.
    """
    xy = ensure_ccw(clean_closed(xy))
    if start == "canonical":
        xy = np.roll(xy, -_max_radius_index(xy), axis=0)
    elif start != "first":
        raise ValueError(f"start must be 'canonical' or 'first', got {start!r}")
    closed = np.vstack([xy, xy[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    t = np.arange(n) * (s[-1] / n)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def canonicalize(xy, n: int | None = None) -> tuple[np.ndarray, dict]:
    """Similarity-normalised contour and the removed transform.

    Centroid (area) -> 0, RMS boundary radius -> 1, major principal axis of the area moments -> +x (sign: the
    max-radius vertex has x >= 0), start at the max-radius vertex; resampled to n points if n is given.
    info: centroid (2,), scale (RMS radius, input units), angle (rad, of the major axis), perimeter, area.
    """
    xy = ensure_ccw(clean_closed(xy))
    area, c, cov = area_moments(xy)
    if not area > 0:
        raise ValueError("degenerate contour (zero area)")
    rel = xy - c
    scale = _rms_radius(rel)
    v = np.linalg.eigh(cov)[1][:, -1]
    k = int(np.argmax(np.sum(rel * rel, axis=1)))
    if rel[k] @ v < 0:
        v = -v
    angle = float(np.arctan2(v[1], v[0]))
    ca, sa = np.cos(angle), np.sin(angle)
    out = rel @ np.array([[ca, -sa], [sa, ca]]) / scale  # rotate by -angle
    out = resample_closed(out, n) if n else np.roll(out, -k, axis=0)
    return out, {"centroid": c, "scale": scale, "angle": angle, "perimeter": perimeter(xy), "area": area}


def _gauss_spectrum(xy, sigma):
    """(FFT of xy (N, 2), angular frequencies (N, 1) [rad/sample], periodic Gaussian transfer function)."""
    w = 2 * np.pi * np.fft.fftfreq(len(xy))[:, None]
    return np.fft.fft(xy, axis=0), w, np.exp(-0.5 * (w * sigma) ** 2)


def smooth_closed(xy, sigma) -> np.ndarray:
    """Exact periodic Gaussian smoothing (FFT); sigma in samples (resample to uniform spacing first)."""
    xy = np.asarray(xy, dtype=np.float64)
    if sigma <= 0:
        return xy.copy()
    spec, _, g = _gauss_spectrum(xy, sigma)
    return np.fft.ifft(spec * g, axis=0).real


def curvature(xy, scales=(1, 2, 4, 8)) -> np.ndarray:
    """(N, S) curvature [1/units] of the contour Gaussian-smoothed at each scale (sigma > 0 in samples; assumes
    roughly uniform spacing). Convex > 0 for either input orientation; a circle of radius r gives ~1/r.

    Derivatives are spectral: scipy's truncated derivative-of-Gaussian kernels do not sum to exactly 0, and on
    finely sampled contours that DC leak is as large as the true second derivative.
    """
    xy = np.asarray(xy, dtype=np.float64)
    out = np.empty((len(xy), len(scales)))
    for j, s in enumerate(scales):
        spec, w, g = _gauss_spectrum(xy, s)
        d1 = np.fft.ifft(spec * g * (1j * w), axis=0).real
        d2 = np.fft.ifft(spec * g * -(w * w), axis=0).real
        speed2 = np.maximum(np.sum(d1 * d1, axis=1), 1e-300)
        out[:, j] = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed2 ** 1.5
    return _orientation(xy) * out


def turning_angle(xy) -> np.ndarray:
    """(N,) exterior turning angle at each vertex [rad, (-pi, pi]]; convex > 0; sums to 2 pi for a simple contour."""
    xy = np.asarray(xy, dtype=np.float64)
    e = np.roll(xy, -1, axis=0) - xy  # outgoing edge
    ep = np.roll(e, 1, axis=0)        # incoming edge
    cr = ep[:, 0] * e[:, 1] - ep[:, 1] * e[:, 0]
    return _orientation(xy) * np.arctan2(cr, np.sum(ep * e, axis=1))


def local_roughness(xy, window: int = 9) -> np.ndarray:
    """(N,) RMS deviation [units] of the contour from its local quadratic (Savitzky-Golay) fit over `window` samples,
    RMS-averaged over the same window: ~0 for smooth curves of any curvature, grows with sub-window irregularity."""
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    w = min(window | 1, n if n % 2 else n - 1)  # odd and <= N
    res = xy - savgol_filter(xy, w, 2, axis=0, mode="wrap")
    return np.sqrt(np.maximum(uniform_filter1d(np.sum(res * res, axis=1), w, mode="wrap"), 0.0))  # fp noise < 0


def convexity_defect(xy) -> np.ndarray:
    """(N,) distance [units] from each vertex to the convex-hull boundary (0 on the hull, > 0 in concavities)."""
    xy = np.asarray(xy, dtype=np.float64)
    eq = ConvexHull(xy).equations  # rows (nx, ny, off): n.x + off <= 0 inside, |n| = 1
    return np.clip(-(xy @ eq[:, :2].T + eq[:, 2]).max(axis=1), 0.0, None)


def radial_signature(xy) -> np.ndarray:
    """(N,) distance [units] of each vertex from the area centroid."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.linalg.norm(xy - area_moments(xy)[1], axis=1)


def tangents(xy) -> np.ndarray:
    """(N, 2) unit tangents (central differences) in traversal order."""
    xy = np.asarray(xy, dtype=np.float64)
    t = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)


def inward_normals(xy) -> np.ndarray:
    """(N, 2) unit normals pointing into the enclosed region, for either input orientation."""
    t = tangents(xy)
    return _orientation(xy) * np.column_stack([-t[:, 1], t[:, 0]])


def polygon_mask(xy, shape) -> np.ndarray:
    """bool (H, W) mask of the pixels whose centres (x = col, y = row) lie inside the polygon."""
    return polygon2mask(tuple(shape[:2]), np.asarray(xy, dtype=np.float64)[:, ::-1])


def distance_to_boundary(xy, points) -> np.ndarray:
    """(M,) Euclidean distance [units] from points (M, 2) to the nearest contour segment."""
    a = np.asarray(xy, dtype=np.float64)
    ab = np.roll(a, -1, axis=0) - a
    l2 = np.maximum(np.sum(ab * ab, axis=1), 1e-300)
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    out = np.empty(len(pts))
    for i in range(0, len(pts), 1024):  # bounds the (chunk, N, 2) temporary
        p = pts[i:i + 1024, None, :]
        t = np.clip(np.sum((p - a) * ab, axis=-1) / l2, 0.0, 1.0)
        out[i:i + 1024] = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1).min(axis=1)
    return out
