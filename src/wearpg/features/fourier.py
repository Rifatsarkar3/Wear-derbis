"""Elliptic Fourier descriptors (Kuhl & Giardina 1982, Comput. Graph. Image Process. 18:236-258)."""

from __future__ import annotations

import numpy as np

from wearpg.features.contour import clean_closed, ensure_ccw


def elliptic_fourier_descriptors(xy, order: int = 10, normalize: bool = True) -> np.ndarray:
    """(order, 4) coefficients [a_n, b_n, c_n, d_n], n = 1..order, of the closed contour at positive orientation.

    Raw coefficients are in input units. normalize=True removes start point, rotation and size: a_1 = 1,
    b_1 = c_1 = 0, d_1 = minor/major of the first-harmonic ellipse (> 0). Kuhl-Giardina leave a 180-degree
    ambiguity (the two ends of the major axis; the alternative negates every even harmonic); it is fixed by starting
    at the end farther from the centre, i.e. sum of the even-harmonic a_n >= 0.
    """
    xy = ensure_ccw(clean_closed(xy))
    d = np.diff(np.vstack([xy, xy[:1]]), axis=0)
    dt = np.linalg.norm(d, axis=1)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    n = np.arange(1, order + 1, dtype=np.float64)
    phi = 2 * np.pi * n[:, None] * t[None, :] / t[-1]
    dcos, dsin = np.diff(np.cos(phi), axis=1), np.diff(np.sin(phi), axis=1)
    k = t[-1] / (2 * np.pi ** 2 * n ** 2)
    ux, uy = d[:, 0] / dt, d[:, 1] / dt
    a, b, c, dd = k * (dcos @ ux), k * (dsin @ ux), k * (dcos @ uy), k * (dsin @ uy)
    if not normalize:
        return np.column_stack([a, b, c, dd])
    th = 0.5 * np.arctan2(2 * (a[0] * b[0] + c[0] * dd[0]), a[0] ** 2 - b[0] ** 2 + c[0] ** 2 - dd[0] ** 2)
    cs, sn = np.cos(n * th), np.sin(n * th)  # start point -> major-axis end of harmonic 1
    a, b, c, dd = a * cs + b * sn, -a * sn + b * cs, c * cs + dd * sn, -c * sn + dd * cs
    psi = np.arctan2(c[0], a[0])
    cp, sp = np.cos(psi), np.sin(psi)  # rotate the major axis onto +x
    out = np.column_stack([cp * a + sp * c, cp * b + sp * dd, -sp * a + cp * c, -sp * b + cp * dd])
    out /= abs(out[0, 0])
    if out[1::2, 0].sum() < 0:
        out[1::2] *= -1
    return out


def efd_feature_vector(xy, order: int = 10) -> np.ndarray:
    """(4 order - 3,) normalised EFD without the constants a_1 = 1, b_1 = c_1 = 0."""
    return elliptic_fourier_descriptors(xy, order, normalize=True).ravel()[3:]
