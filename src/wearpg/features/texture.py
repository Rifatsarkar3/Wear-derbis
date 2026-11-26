"""Texture and colour descriptors of a NATIVE-resolution masked patch (never the rescaled crop: resampling would
change the pixel-scale statistics with particle size)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates
from skimage.color import rgb2gray
from skimage.feature import canny, graycomatrix, graycoprops, local_binary_pattern, structure_tensor
from skimage.util import img_as_float

GLCM_LEVELS = 32
_GLCM_DIST = (1, 2, 4)
_GLCM_ANG = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
_GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation", "dissimilarity")


def _mask(mask) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    return m if m.any() else np.ones_like(m)  # degenerate mask: describe the whole patch rather than fail


def texture_descriptors(patch_rgb, mask) -> dict:
    """Grey-level texture inside the mask (grey = luminance in [0, 1]).

    tex_mean/std/skew/p10/p90: grey statistics. tex_glcm_*: masked GLCM (grey p1..p99 inside the mask mapped to
    levels 1..32, outside = 0 and dropped; symmetric; distances 1, 2, 4 px x 4 angles; mean over the non-empty
    pairs). tex_lbp_0..9: uniform LBP (P=8, R=1) histogram over the mask (sums to 1). On the core (mask eroded by
    2 px, so the particle outline does not count as texture): tex_edge_density = Canny edge fraction,
    tex_coherence = (l1 - l2)/(l1 + l2) of the mean structure tensor (1 = one dominant orientation, e.g.
    striations). tex_dark_fraction = fraction of mask pixels below 0.75 x the mask median (pits, dark tint).
    tex_rim_contrast = Michelson contrast core vs 2-px inner rim (> 0: darker rim).
    """
    g = rgb2gray(patch_rgb) if np.ndim(patch_rgb) == 3 else img_as_float(patch_rgb)
    m = _mask(mask)
    v = g[m]
    sd = float(v.std())
    p1, p10, p50, p90, p99 = np.percentile(v, [1, 10, 50, 90, 99])
    skew = float(np.mean((v - v.mean()) ** 3) / sd ** 3) if sd > 1e-9 else 0.0  # population skewness
    out = {"tex_mean": float(v.mean()), "tex_std": sd, "tex_skew": skew,
           "tex_p10": float(p10), "tex_p90": float(p90)}

    # graycomatrix has no mask: quantise inside to 1..L, outside 0, then drop row/col 0
    q = 1 + np.clip(((g - p1) / max(p99 - p1, 1e-6) * GLCM_LEVELS).astype(int), 0, GLCM_LEVELS - 1)
    q[~m] = 0
    glcm = graycomatrix(q.astype(np.uint8), _GLCM_DIST, _GLCM_ANG, levels=GLCM_LEVELS + 1, symmetric=True)[1:, 1:]
    ok = glcm.sum(axis=(0, 1)) > 0
    for prop in _GLCM_PROPS:
        out[f"tex_glcm_{prop}"] = float(graycoprops(glcm, prop)[ok].mean()) if ok.any() else 0.0

    lbp = local_binary_pattern(np.round(g * 255).astype(np.uint8), 8, 1, method="uniform")
    hist = np.bincount(lbp[m].astype(int), minlength=10)[:10] / m.sum()
    out.update({f"tex_lbp_{i}": float(h) for i, h in enumerate(hist)})

    core = binary_erosion(m, iterations=2)
    region = core if core.any() else m
    out["tex_edge_density"] = float(canny(g, sigma=1.0, mode="nearest")[core].mean()) if core.any() else 0.0
    arr, arc, acc = (t[region].mean() for t in structure_tensor(g, sigma=1.0, mode="nearest", order="rc"))
    tr = arr + acc
    out["tex_coherence"] = float(np.sqrt((arr - acc) ** 2 + 4 * arc ** 2) / tr) if tr > 1e-12 else 0.0
    out["tex_dark_fraction"] = float(np.mean(v < 0.75 * p50))
    rim = m & ~core
    if core.any() and rim.any():
        mc, mr = float(g[core].mean()), float(g[rim].mean())
        out["tex_rim_contrast"] = (mc - mr) / (mc + mr + 1e-6)
    else:
        out["tex_rim_contrast"] = 0.0
    return out


def color_descriptors(patch_rgb, mask) -> dict:
    """Mean colour inside the mask: color_r/g/b in [0, 1] and color_sat = mean HSV saturation (COMPOSITION group)."""
    rgb = img_as_float(np.asarray(patch_rgb))[..., :3][_mask(mask)]  # (n, 3)
    mx = rgb.max(axis=1)
    sat = np.where(mx > 0, (mx - rgb.min(axis=1)) / np.maximum(mx, 1e-12), 0.0)  # HSV saturation
    mean = rgb.mean(axis=0)
    return {"color_r": float(mean[0]), "color_g": float(mean[1]), "color_b": float(mean[2]),
            "color_sat": float(sat.mean())}


def boundary_band_profile(gray, xy, normals, depths_px) -> np.ndarray:
    """(N, D) bilinear grey samples at xy + depth * normal (xy, normals (N, 2) in image px (x = col, y = row);
    depths_px (D,) shared or (N, D) per point, e.g. clipped per node; depth > 0 along `normals`, e.g. inward).
    Points outside the image take the nearest edge value."""
    d = np.asarray(depths_px, dtype=np.float64)
    d = d[None, :, None] if d.ndim == 1 else d[:, :, None]
    pts = np.asarray(xy, dtype=np.float64)[:, None, :] + d * np.asarray(normals, dtype=np.float64)[:, None, :]
    vals = map_coordinates(np.asarray(gray), [pts[..., 1].ravel(), pts[..., 0].ravel()], order=1, mode="nearest")
    return vals.reshape(pts.shape[:2]).astype(np.float64)
