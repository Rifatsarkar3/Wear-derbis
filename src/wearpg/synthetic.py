"""[A] Synthetic wear tests + ferrograph images for PIPELINE VERIFICATION ONLY (DESIGN sec. 4).

Chain: per-run wear model (hardness-scaled Archard, stochastic mild->severe transition, noisy rounded gravimetry)
-> per-sample debris budget (capture x observation fraction x interval or inventory volume) -> vectorized
particle-parameter draws until the metal-equivalent budget is reached -> detection (area, contrast, field-of-view
fit) -> uniform stored subset -> contours (base outline + calibrated fractal roughness) -> field-of-view images and
multi-view silhouette frames -> EDS / SEM / confocal / rater annotation.

Units: um, um^2, um^3 (particles), mm^3 (wear volumes), px with x = column, y = row and pixel centres at integer
coordinates. Same seed -> identical tables and images. Synthetic-only extra particle columns:
``true_fractal_D_target`` (D_target of the roughness) and ``true_fractal_D`` (divider D of the stored view-0 GT
contour measured by this module's private estimator; NaN when the estimator is not valid).
``imaging.renderer: v2`` (default v1) re-renders the same particles under bichromatic illumination with ferrographic
chain deposition and correlated particle shading: rendering only (docs/renderer_realism.md).
"""

from __future__ import annotations

import functools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from scipy.stats.contingency import association
from skimage.measure import find_contours

from .config import load_config
from .schema import EDS_COLUMNS, ELEMENTS, MECHANISMS, SEVERITY_LEVELS, TABLES, UM3_PER_MM3, WearDataset
from .utils import get_logger

from wearpg.schema import SYNTHETIC_BANNER  # single source
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# size = (median, lo, hi) [um] of the major axis (cutting: curved ribbon length; spherical: diameter); aspect, D
# (D_target), thick (thickness / minor axis) and refl (reflectance r_class) are uniform ranges; spherical particles
# are exact spheres (circle planform: aspect unused, thickness = diameter); striation / pits / tint = texture
# probabilities (DESIGN sec. 4 table).
MECHANISM_PRIORS: dict[str, dict] = {
    "rubbing": dict(size=(6, 2, 15), aspect=(1.0, 2.0), D=(1.02, 1.05), thick=(0.10, 0.10), refl=(0.85, 0.95),
                    shape="ellipse", striation=0.1, pits=0.05, tint=0.05),
    "cutting": dict(size=(25, 10, 60), aspect=(5.0, 20.0), D=(1.03, 1.06), thick=(0.8, 1.2), refl=(0.75, 0.90),
                    shape="ribbon", striation=0.5, pits=0.05, tint=0.05),
    "spherical": dict(size=(3, 1, 8), aspect=(1.0, 1.05), D=(1.00, 1.02), thick=(1.0, 1.0), refl=(0.45, 0.60),
                      shape="circle", striation=0.1, pits=0.05, tint=0.05),
    "fatigue_chunk": dict(size=(25, 10, 60), aspect=(1.0, 2.0), D=(1.10, 1.20), thick=(0.4, 0.6), refl=(0.55, 0.75),
                          shape="ellipse", striation=0.1, pits=0.6, tint=0.05),
    "laminar": dict(size=(30, 15, 50), aspect=(1.2, 2.0), D=(1.05, 1.10), thick=(0.05, 0.10), refl=(0.85, 0.95),
                    shape="ellipse", striation=0.1, pits=0.2, tint=0.05),
    "severe_sliding": dict(size=(35, 15, 80), aspect=(1.5, 4.0), D=(1.05, 1.12), thick=(0.15, 0.25),
                           refl=(0.65, 0.85), shape="polygon", striation=0.8, pits=0.05, tint=0.05),
    "oxide": dict(size=(15, 5, 35), aspect=(1.0, 2.0), D=(1.15, 1.25), thick=(0.3, 0.5), refl=(0.45, 0.60),
                  shape="agglomerate", striation=0.1, pits=0.05, tint=0.9),
}
SEVERITY_MIXTURES: dict[int, dict[str, float]] = {
    0: {"rubbing": .60, "cutting": .05, "spherical": .05, "fatigue_chunk": .03, "laminar": .10,
        "severe_sliding": .02, "oxide": .15},
    2: {"rubbing": .08, "cutting": .22, "spherical": .05, "fatigue_chunk": .20, "laminar": .10,
        "severe_sliding": .30, "oxide": .05},
}
# documentation only: transition intervals draw each particle from mild or severe by phase volume / E[v_phase]
SEVERITY_MIXTURES[1] = {m: (SEVERITY_MIXTURES[0][m] + SEVERITY_MIXTURES[2][m]) / 2 for m in MECHANISMS}
CONFUSABLE: dict[str, tuple[str, ...]] = {
    "rubbing": ("laminar", "oxide"), "cutting": ("severe_sliding", "laminar"), "spherical": ("oxide", "rubbing"),
    "fatigue_chunk": ("severe_sliding", "oxide", "laminar"), "laminar": ("rubbing", "severe_sliding", "fatigue_chunk"),
    "severe_sliding": ("fatigue_chunk", "cutting", "laminar"), "oxide": ("fatigue_chunk", "spherical", "rubbing"),
}

_PR = {k: np.array([MECHANISM_PRIORS[m][k] for m in MECHANISMS]) for k in MECHANISM_PRIORS["rubbing"]}
_CUM = {s: np.cumsum([SEVERITY_MIXTURES[s][m] for m in MECHANISMS]) for s in (0, 2)}
_CONF_IDX = np.array([[MECHANISMS.index(c) for c in CONFUSABLE[m]] + [-1] * (3 - len(CONFUSABLE[m]))
                      for m in MECHANISMS])
_CONF_N = np.array([len(CONFUSABLE[m]) for m in MECHANISMS])
_OXIDE = MECHANISMS.index("oxide")
_LUMA = np.array([0.2125, 0.7154, 0.0721])  # skimage rgb2gray weights (segmentation sees the same luminance)
_SS = 4            # supersampling factor of the anti-aliased coverage
_K_FLAT = 6        # roughness spectrum flattened below this harmonic (keeps high-D outlines compact, see _roughness)
_CUTOFF_UM = 0.5   # physical roughness cutoff wavelength [um]
_GAP_PX, _MARGIN_PX = 3, 2   # min gap between non-touching particles; FOV border margin


def load_synthetic_config(path_or_name: str | Path) -> dict:
    """"smoke" | "pilot" -> configs/synthetic_<name>.yaml; anything else is a YAML path (``base:`` supported)."""
    p = Path(path_or_name)
    if not p.suffix and not p.exists():
        p = _CONFIG_DIR / f"synthetic_{path_or_name}.yaml"
    return load_config(p)


# ============================================================================ geometry helpers


def _shoelace(xy: np.ndarray) -> float:
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


def _perimeter(xy: np.ndarray) -> float:
    return float(np.hypot(*(np.roll(xy, -1, axis=0) - xy).T).sum())


def _resample(xy: np.ndarray, n: int) -> np.ndarray:
    """n points uniformly spaced in arc length along the closed polyline xy (no repeated end point)."""
    closed = np.vstack([xy, xy[:1]])
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    t = np.linspace(0.0, d[-1], n, endpoint=False)
    return np.c_[np.interp(t, d, closed[:, 0]), np.interp(t, d, closed[:, 1])]


def _feret_max(xy: np.ndarray) -> float:
    try:
        xy = xy[ConvexHull(xy).vertices]
    except QhullError:
        pass
    return float(pdist(xy).max())


def _walk(xs: list, ys: list, step: float) -> float:
    """Richardson walking-divider length of a closed polyline: hop to the first point along the curve at Euclidean
    distance `step`; the final partial hop back to the start is added as its length."""
    n = len(xs)
    px, py, s2, i, count = xs[0], ys[0], step * step, 0, 0
    while True:
        j = i + 1
        while j <= n and (xs[j % n] - px) ** 2 + (ys[j % n] - py) ** 2 < s2:
            j += 1
        if j > n:
            break
        a, b = (j - 1) % n, j % n
        ax, ay, ex, ey = xs[a] - px, ys[a] - py, xs[b] - xs[a], ys[b] - ys[a]
        qa, qb, qc = ex * ex + ey * ey, 2 * (ax * ex + ay * ey), ax * ax + ay * ay - s2
        t = (-qb + math.sqrt(max(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa) if qa > 0 else 1.0
        px, py, i, count = xs[a] + t * ex, ys[a] + t * ey, j - 1, count + 1
        if i >= n:
            break
    return count * step + math.hypot(px - xs[0], py - ys[0])


def _divider_dimension(xy_px: np.ndarray, n_steps: int = 10, min_frac: float = 0.01, max_frac: float = 0.10) -> dict:
    """Private copy of the features.fractal.divider_dimension definition: L = max Feret [px]; n_steps log-spaced
    steps in [max(2 px, min_frac L), max_frac L]; D = 1 - slope(ln length vs ln step); invalid if the steps span < 2x."""
    xy = np.asarray(xy_px, float)
    big_l = _feret_max(xy)
    lo, hi = max(2.0, min_frac * big_l), max_frac * big_l
    if hi < 2 * lo:
        return {"D": 1.0, "r2": float("nan"), "valid": False}
    steps = np.geomspace(lo, hi, n_steps)
    xs, ys = xy[:, 0].tolist(), xy[:, 1].tolist()
    ln_len = np.log([_walk(xs, ys, s) for s in steps])
    slope, icept = np.polyfit(np.log(steps), ln_len, 1)
    resid = ln_len - (slope * np.log(steps) + icept)
    r2 = 1 - resid.var() / ln_len.var() if ln_len.var() > 0 else float("nan")
    return {"D": float(1 - slope), "r2": float(r2), "valid": True}


def _roughness(rng: np.random.Generator, m: int, k_max: int, h: float, c: float, k_min: int = 2) -> np.ndarray:
    """(m,) periodic sequence sum_{k=k_min..k_max} c * max(k, _K_FLAT)^-(h+0.5) cos(2 pi k j/m + phi_k).

    DESIGN uses k^-(H+0.5) from k = 2; flattening below k = 6 keeps high-D particles compact instead of
    propeller-shaped while the band the divider measures (k >~ 30) follows the power law exactly."""
    k_max = min(k_max, m // 2 - 1)
    if c <= 0 or k_max < k_min:
        return np.zeros(m)
    k = np.arange(k_min, k_max + 1)
    spec = np.zeros(m // 2 + 1, complex)
    spec[k] = c * np.maximum(k, _K_FLAT) ** -(h + 0.5) * np.exp(2j * np.pi * rng.random(k.size)) * (m / 2)
    return np.fft.irfft(spec, m)


def _ray_polygon(th: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Radius along rays th of a polygon that is star-shaped about the origin (vertices sorted by angle)."""
    ang = np.arctan2(vy, vx) % (2 * np.pi)
    o = np.argsort(ang)
    vx, vy, ang = vx[o], vy[o], ang[o]
    j = (np.searchsorted(ang, th) - 1) % len(ang)
    px, py, qx, qy = vx[j], vy[j], np.roll(vx, -1)[j], np.roll(vy, -1)[j]
    return (px * qy - py * qx) / (np.cos(th) * (qy - py) - np.sin(th) * (qx - px))


def _ray_ellipses(th: np.ndarray, cx, cy, ea, eb) -> np.ndarray:
    """Max exit distance along rays th over axis-aligned ellipses that all contain the origin (star-shaped union)."""
    ux, uy = np.cos(th)[:, None], np.sin(th)[:, None]
    qa = (ux / ea) ** 2 + (uy / eb) ** 2
    qb = -2 * (ux * cx / ea ** 2 + uy * cy / eb ** 2)
    qc = (cx / ea) ** 2 + (cy / eb) ** 2 - 1
    return ((-qb + np.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)).max(axis=1)


def _planform(rng: np.random.Generator, shape: str, size: float, minor: float, d_target: float,
              c: float) -> np.ndarray:
    """Dense rough outline (M, 2) [um], major axis along x, before area normalisation.

    Star-shaped bases (ellipse, circle, straight-edged polygon, union of ellipses containing the origin) get the
    radial roughness r(theta) *= max(1 + rough, 0.3) with K = perimeter / 0.5 um harmonics, which can never
    self-intersect. The curved ribbon (not star-shaped) is a rough stadium (normal displacement, harmonics from the
    ribbon-width scale, amplitude 0.7 P / 2 pi, which matches its class-mean divider D to D_target despite the
    divider cutting across the ends) bent along a circular arc (a diffeomorphism, so still simple)."""
    a, b = size / 2, minor / 2
    h = 2 - d_target
    if shape == "ribbon":
        big_l, w = size, minor
        per = 2 * (big_l - w) + np.pi * w
        k_max = int(per / _CUTOFF_UM)
        m = int(np.clip(4 * k_max, 256, 8192))
        u = np.linspace(0, per, m, endpoint=False)   # arc length: bottom edge, right cap, top edge, left cap (CCW)
        s1, half = big_l - w, (big_l - w) / 2
        s2, s3 = s1 + np.pi * b, 2 * s1 + np.pi * b
        cap_r = (u >= s1) & (u < s2)
        cap_l = u >= s3
        ph = np.where(cap_r, -np.pi / 2 + (u - s1) / b, np.pi / 2 + (u - s3) / b)
        nx = np.where(cap_r | cap_l, np.cos(ph), 0.0)                       # outward normal
        ny = np.where(cap_r | cap_l, np.sin(ph), np.where(u < s1, -1.0, 1.0))
        x = np.select([u < s1, cap_r, u < s3], [u - half, half + b * nx, half - (u - s2)], -half + b * nx)
        y = b * ny                                                           # edges: ny = -1 / +1
        disp = np.clip(0.7 * per / (2 * np.pi) * _roughness(rng, m, k_max, h, c, int(np.ceil(per / (np.pi * w)))),
                       -0.7 * b, b)
        x, y = x + disp * nx, y + disp * ny
        rc = big_l / rng.uniform(0.5, 3.0)   # bend radius: arc angle 0.5-3 rad
        return np.c_[(rc - y) * np.sin(x / rc), rc - (rc - y) * np.cos(x / rc)]
    per = np.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    k_max = int(per / _CUTOFF_UM)
    m = int(np.clip(4 * k_max, 256, 8192))
    th = np.linspace(0, 2 * np.pi, m, endpoint=False)
    if shape == "polygon":
        n = int(rng.integers(5, 10))
        tv = (np.arange(n) + rng.uniform(-0.3, 0.3, n)) * 2 * np.pi / n + rng.uniform(0, 2 * np.pi)
        rv = rng.uniform(0.8, 1.0, n)
        r = _ray_polygon(th, a * rv * np.cos(tv), b * rv * np.sin(tv))
    elif shape == "agglomerate":
        n = int(rng.integers(3, 8))
        al, rho = rng.uniform(0, 2 * np.pi, n), rng.uniform(0.35, 0.6, n)
        d = rng.uniform(0.5, 0.95, n) * rho     # lobe centre inside its own lobe -> union is star-shaped
        r = _ray_ellipses(th, a * d * np.cos(al), b * d * np.sin(al), 1.6 * a * rho, 1.6 * b * rho)
    else:  # ellipse | circle
        r = a * b / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)
    r = r * np.maximum(1 + _roughness(rng, m, k_max, h, c), 0.3)
    return np.c_[r * np.cos(th), r * np.sin(th)]


def _area_um2(shape, size, minor):
    """Planform area [um^2] by base shape (vectorized); contours are rescaled to exactly this area."""
    return np.select([np.isin(shape, ("ellipse", "circle")), shape == "ribbon"],
                     [np.pi / 4 * size * minor, (size - minor) * minor + np.pi / 4 * minor ** 2],
                     0.65 * size * minor)


def _contour_px(rng, shape, size, minor, d_target, c, upp, theta, area_um2) -> np.ndarray:
    """GT contour [px] centred at its vertex mean: rotated by theta, ~1.5 px spacing (32-512 points), positive
    shoelace area exactly area_um2 / upp^2."""
    xy = _planform(rng, shape, size, minor, d_target, c)
    ct, st = math.cos(theta), math.sin(theta)
    xy = xy @ np.array([[ct, st], [-st, ct]]) / upp
    xy = _resample(xy, int(np.clip(round(_perimeter(xy) / 1.5), 32, 512)))
    area = _shoelace(xy)
    if area < 0:
        xy, area = xy[::-1], -area
    xy = xy * math.sqrt(area_um2 / upp ** 2 / area)
    return xy - xy.mean(axis=0)


@functools.cache
def _amplitude_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean divider D of rough circles (40 um at 0.5 um/px, 3 replicates) on a (D_target x c) grid; ~1 s, once per
    process. Rows are made non-decreasing in c so they can be inverted."""
    rng = np.random.default_rng(20240607)
    d_grid = np.arange(1.0, 1.301, 0.05)
    c_grid = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0])
    area = float(_area_um2(np.array("circle"), 40.0, 40.0))
    meas = np.array([[np.mean([_divider_dimension(_contour_px(rng, "circle", 40.0, 40.0, d, c, 0.5, 0.0, area))["D"]
                               for _ in range(3)]) for c in c_grid] for d in d_grid])
    return d_grid, c_grid, np.maximum.accumulate(meas, axis=1) + 1e-9 * np.arange(len(c_grid))


def _amplitude(d_target: float) -> float:
    """Roughness amplitude c whose measured divider D on the reference circle equals d_target (clamped to grid)."""
    d_grid, c_grid, meas = _amplitude_table()
    row = [np.interp(d_target, d_grid, meas[:, j]) for j in range(len(c_grid))]
    return float(np.interp(d_target, row, c_grid))


# ============================================================================ parameter path (no rendering)


def _round_to(x, res: float):
    return np.round(np.asarray(x, float) / res) * res


def _draw_particles(rng: np.random.Generator, n: int, p_severe: float, load_n: float, oxide_c: float) -> dict:
    """n particle parameter sets (vectorized). Severe-phase particles come from the severe mixture with sizes x1.3;
    all sizes x (F/60)^0.15. volume = planform area x thickness (sphere pi d^3/6); metal = volume x c_class."""
    sev = rng.random(n) < p_severe
    u = rng.random(n)
    mech = np.minimum(np.where(sev, np.searchsorted(_CUM[2], u, side="right"),
                               np.searchsorted(_CUM[0], u, side="right")), len(MECHANISMS) - 1)
    med, lo, hi = _PR["size"][mech].T
    size = np.clip(np.exp(rng.normal(np.log(med), np.log(hi / lo) / 4)), lo, hi)
    size = size * np.where(sev, 1.3, 1.0) * (load_n / 60.0) ** 0.15
    shape = _PR["shape"][mech]
    sphere = shape == "circle"
    minor = np.where(sphere, size, size / rng.uniform(*_PR["aspect"][mech].T))
    d_target = rng.uniform(*_PR["D"][mech].T)
    thick = np.where(sphere, size, rng.uniform(*_PR["thick"][mech].T) * minor)
    area = _area_um2(shape, size, minor)
    vol = np.where(sphere, np.pi / 6 * size ** 3, area * thick)
    refl = np.minimum(np.exp(rng.normal(np.log(rng.uniform(*_PR["refl"][mech].T)), 0.25)), 0.97)
    return dict(mech=mech, severe=sev, size=size, minor=minor, d_target=d_target, thick=thick, area=area, vol=vol,
                refl=refl, metal=vol * np.where(mech == _OXIDE, oxide_c, 1.0))


@functools.cache
def _mean_metal(severe: bool, load_n: float, oxide_c: float) -> float:
    """E[metal-equivalent particle volume] [um^3] of one phase at load F (fixed-seed Monte Carlo, cached)."""
    return float(_draw_particles(np.random.default_rng(1234), 200_000, float(severe), load_n, oxide_c)["metal"].mean())


def _draw_budget(rng, budget_um3: float, p_severe: float, load_n: float, oxide_c: float) -> dict:
    """Draw particles until their cumulative metal-equivalent volume first reaches budget_um3 (the last particle is
    the one that crosses it). Zero budget -> no particles."""
    mean = p_severe * _mean_metal(True, load_n, oxide_c) + (1 - p_severe) * _mean_metal(False, load_n, oxide_c)
    parts, total = [], 0.0
    while total < budget_um3:
        n = int(min(1.2 * (budget_um3 - total) / mean + 32, 2_000_000))
        p = _draw_particles(rng, n, p_severe, load_n, oxide_c)
        cum = total + np.cumsum(p["metal"])
        k = int(np.searchsorted(cum, budget_um3))   # first index with cum >= budget
        if k < n:
            p = {key: val[:k + 1] for key, val in p.items()}
        parts.append(p)
        total = float(cum[min(k, n - 1)])
    if not parts:
        return _draw_particles(rng, 0, p_severe, load_n, oxide_c)
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def _luma(rgb) -> np.ndarray:
    return np.asarray(rgb, float) @ _LUMA


def _fov_limit_px(size: int) -> int:
    """Largest contour extent [px] that still fits a size x size field (border margin + patch border)."""
    return size - 2 * _MARGIN_PX - 6


def _detected(area_um2, size_um, contrast, upp: float, img: dict) -> np.ndarray:
    """Detection rule: primary-view area >= min_area_px at the sample's px size, contrast >= min_contrast, and the
    nominal size fits a field of view (larger particles would be cut by the image border)."""
    fov_um = _fov_limit_px(int(img["image_size_px"])) * upp
    return (area_um2 / upp ** 2 >= img["min_area_px"]) & (contrast >= img["min_contrast"]) & (size_um <= fov_um)


def _simulate(cfg: dict, seed: int) -> dict:
    """Parameter-only path: specimens, runs, samples (with um_per_px / magnification), stored-particle parameters
    (no contours or images), per-sample budget bookkeeping and the parameter-level meta stats."""
    w, deb, img = cfg["wear"], cfg["debris"], cfg["imaging"]
    mats = {m["material_id"]: m for m in cfg["materials"]}
    cap = cfg["capture"]
    rng_eta = np.random.default_rng([seed, 4])
    eta_mat = {k: cap["ferrous" if m["ferromagnetic"] else "nonferrous"] * math.exp(rng_eta.normal(0, cap["material_sigma"]))
               for k, m in mats.items()}
    r_cf = float(deb.get("counterface_wear_ratio", 0.0))
    oxide_c = float(deb["metal_fraction"].get("oxide", 1.0))
    cumulative = deb["sampling"] == "cumulative"
    big_t, n_s = float(cfg["duration_s"]), int(cfg["samples_per_run"])
    t0, t1 = np.linspace(0.0, big_t, n_s + 1)[:-1], np.linspace(0.0, big_t, n_s + 1)[1:]
    ref, res, noise = w["severity_ref"], w["mass_resolution_mg"], w["mass_noise_rel"]
    specimens, runs, samples, src = [], [], [], []
    exp_severe = {}
    for mat in cfg["materials"]:
        mid, hard, rho = mat["material_id"], mat["hardness_MPa"], mat["density_g_cm3"]
        cf = mats.get(mat["counterface_id"])
        if r_cf > 0 and cf is None:
            raise ValueError(f"counterface {mat['counterface_id']!r} must be listed in materials when "
                             "debris.counterface_wear_ratio > 0")
        k_mild = w["k_mild_ref"] * w["hardness_ref_MPa"] / hard
        k_sev = w["k_severe_ref"] * w["hardness_ref_MPa"] / hard
        p_tr = []
        for load in cfg["loads_N"]:
            for speed in cfg["speeds_m_s"]:
                sev_idx = (load / ref["load_N"] * math.sqrt(speed / ref["speed_m_s"])
                           * (ref["hardness_MPa"] / hard) ** w["hardness_exponent"])
                p_tr.append(1 - math.exp(-(sev_idx / w["transition_time_scale"]) ** 2))
                cond = f"{mid}_F{load:g}_v{speed:g}"
                for rep in range(int(cfg["repeats"])):
                    r = len(runs)
                    rng = np.random.default_rng([seed, 0, r])
                    t_tr = w["transition_time_scale"] * big_t * math.sqrt(rng.exponential()) / sev_idx
                    w_sev = np.clip((t1 - np.maximum(t0, t_tr)) / (t1 - t0), 0.0, 1.0)
                    eff = math.exp(rng.normal(0, w["run_sigma"])) * np.exp(rng.normal(0, w["interval_sigma"], n_s))
                    eff[0] *= w["run_in_factor"]
                    ds = speed * (t1 - t0)
                    dv_mild, dv_sev = (1 - w_sev) * k_mild * eff * load * ds, w_sev * k_sev * eff * load * ds  # mm^3
                    dv = dv_mild + dv_sev
                    mass = float(_round_to(rho * dv.sum() * (1 + rng.normal(0, noise)), res))
                    vol = mass / rho if mass > 0 else np.nan
                    m_int = _round_to(rho * dv * (1 + rng.normal(0, noise, n_s)), res)
                    cf_mass = (float(_round_to(cf["density_g_cm3"] * r_cf * dv.sum() * (1 + rng.normal(0, noise)), res))
                               if r_cf > 0 else np.nan)
                    oil = rng.uniform(*cfg["sampled_oil_ml"], n_s)
                    imf = np.exp(rng.uniform(*np.log(deb["imaged_fraction_range"]), n_s))
                    q = oil / cfg["reservoir_volume_ml"]
                    run_id, spec_id = f"R{r:03d}_{cond}_r{rep}", f"SP{r:03d}_{mid}"
                    specimens.append(dict(specimen_id=spec_id, material_id=mid, counterface_id=mat["counterface_id"],
                                          material_pair=f"{mid}|{mat['counterface_id']}", hardness_MPa=hard,
                                          density_g_cm3=rho, ferromagnetic=bool(mat["ferromagnetic"]),
                                          counterface_density_g_cm3=cf["density_g_cm3"] if cf else np.nan))
                    runs.append(dict(run_id=run_id, specimen_id=spec_id, condition_id=cond, rig=cfg["rig"],
                                     load_N=float(load), speed_m_s=float(speed), lubricant_id=cfg["lubricant_id"],
                                     temperature_C=cfg.get("temperature_C", np.nan), duration_s=big_t,
                                     sliding_distance_m=speed * big_t, repeat_idx=rep,
                                     reservoir_volume_ml=float(cfg["reservoir_volume_ml"]), mass_loss_mg=mass,
                                     wear_volume_mm3=vol, wear_volume_method="gravimetric",
                                     wear_rate_mm3_Nm=vol / (load * speed * big_t),
                                     regime="severe" if t_tr < big_t else "mild",
                                     transition_time_s=t_tr if t_tr < big_t else np.nan,
                                     debris_sampling=deb["sampling"], counterface_mass_loss_mg=cf_mass))
                    inv_m = inv_s = 0.0
                    for s in range(n_s):
                        if cumulative:  # oil inventory: earlier samplings removed q of what was there
                            inv_m = inv_m * (1 - q[s - 1]) + dv_mild[s] if s else dv_mild[s]
                            inv_s = inv_s * (1 - q[s - 1]) + dv_sev[s] if s else dv_sev[s]
                        else:
                            inv_m, inv_s = dv_mild[s], dv_sev[s]
                        src.append((inv_m, inv_s, load, mid, mat["counterface_id"]))
                        samples.append(dict(
                            sample_id=f"{run_id}_s{s}", run_id=run_id, t_start_s=t0[s], t_end_s=t1[s], time_s=t1[s],
                            sliding_distance_start_m=speed * t0[s], sliding_distance_end_m=speed * t1[s],
                            sampled_oil_ml=oil[s], imaged_fraction=imf[s], observation_fraction=q[s] * imf[s],
                            n_particles_total=0,
                            interval_wear_volume_mm3=(m_int[s] / rho if m_int[s] > 0 else np.nan)
                            if deb["expose_interval_volume"] else np.nan,
                            severity=0 if w_sev[s] <= 0 else (2 if w_sev[s] >= 1 else 1),
                            true_interval_wear_volume_mm3=dv[s]))
        exp_severe[mid] = float(np.mean(p_tr))
    samples, runs, specimens = pd.DataFrame(samples), pd.DataFrame(runs), pd.DataFrame(specimens)
    run_mat = runs.specimen_id.map(specimens.set_index("specimen_id").material_id)
    material = samples.run_id.map(pd.Series(run_mat.to_numpy(), index=runs.run_id))

    # magnification: stratified randomization (balanced within material x severity), uniform marginally
    mags = cfg["imaging"]["magnifications"]
    rng_mag = np.random.default_rng([seed, 5])
    mag_idx = np.zeros(len(samples), int)
    for _, idx in samples.groupby([material, samples.severity]).indices.items():
        lab = np.roll(np.resize(np.arange(len(mags)), len(idx)), int(rng_mag.integers(len(mags))))
        mag_idx[idx] = rng_mag.permutation(lab)
    samples["magnification"] = [float(mags[i]["magnification"]) for i in mag_idx]
    samples["um_per_px"] = [float(mags[i]["um_per_px"]) for i in mag_idx]

    bg_luma = _luma(img["background_rgb"]) / 255
    stored, budget = [], np.zeros((len(samples), 4))  # budget, drawn metal, last particle metal, n drawn
    n_total = np.zeros(len(samples), int)
    for i, (inv_m, inv_s, load, mid, cf_id) in enumerate(src):
        rng = np.random.default_rng([seed, 1, i])
        f_s, upp = samples.observation_fraction[i], samples.um_per_px[i]
        lognorm = math.exp(rng.normal(0, deb["budget_sigma"]))
        e_s, e_m = _mean_metal(True, load, oxide_c), _mean_metal(False, load, oxide_c)
        p_sev = (inv_s / e_s) / (inv_s / e_s + inv_m / e_m)
        sources = [(mid, eta_mat[mid] * lognorm * f_s * (inv_m + inv_s) * UM3_PER_MM3)]
        if r_cf > 0:  # counterface debris joins the budget with its own (ferrous) capture
            sources.append((cf_id, eta_mat[cf_id] * lognorm * f_s * r_cf * (inv_m + inv_s) * UM3_PER_MM3))
        drawn = []
        for k, (src_id, b) in enumerate(sources):
            p = _draw_budget(rng, b, p_sev, load, oxide_c)
            p["src"] = np.full(len(p["mech"]), k)
            p["contrast"] = bg_luma - np.minimum(p["refl"] * _luma(mats[src_id]["color_rgb"]) / 255,
                                                 img["max_luminance_ratio"] * bg_luma)
            drawn.append(p)
            if k == 0:
                budget[i] = b, p["metal"].sum(), p["metal"][-1] if len(p["metal"]) else 0.0, len(p["metal"])
        p = {key: np.concatenate([d[key] for d in drawn]) for key in drawn[0]}
        det = np.flatnonzero(_detected(p["area"], p["size"], p["contrast"], upp, img))
        n_total[i] = len(det)
        keep = np.sort(rng.choice(det, min(len(det), int(deb["max_stored_per_sample"])), replace=False))
        stored.append(pd.DataFrame({key: val[keep] for key, val in p.items()}).assign(sample_idx=i))
    samples["n_particles_total"] = n_total
    parts = pd.concat(stored, ignore_index=True)

    n_stored = parts.groupby("sample_idx").size().reindex(range(len(samples)), fill_value=0).to_numpy()
    sev_name = samples.severity.map(dict(enumerate(SEVERITY_LEVELS)))
    stats = {
        "expected_severe_fraction": exp_severe,
        "severe_fraction": (runs.regime == "severe").groupby(run_mat).mean().to_dict(),
        "median_n_total": {f"{m}|{s}": float(v) for (m, s), v in
                           samples.n_particles_total.groupby([material, sev_name]).median().items()},
        "empty_sample_fraction": float(np.mean(n_total == 0)),
        "mean_ht_factor": float(np.mean(n_total[n_stored > 0] / n_stored[n_stored > 0])) if n_stored.any() else np.nan,
        "cramers_v_magnification_severity": _cramers_v(samples.magnification, samples.severity),
        "cramers_v_magnification_material": _cramers_v(samples.magnification, material),
        "eta_material": eta_mat,
        "n_drawn": int(budget[:, 3].sum()),
        "n_detected": int(n_total.sum()),
        "n_stored": int(len(parts)),
    }
    return dict(specimens=specimens, runs=runs, samples=samples, parts=parts, budget=budget, stats=stats,
                material=material.to_numpy())


def simulate_tables(cfg: dict, seed: int | None = None) -> tuple[WearDataset, pd.DataFrame]:
    """Parameter-level campaign without rendering, for simulation stress tests and power analysis:
    (dataset, descriptors). Specimens, runs, samples (n_particles_total = detected count) and stored particles are the
    ones generate_dataset draws before rendering (same cfg and seed). Each stored particle gets a square view-0 contour
    with its exact planform area at the sample's px size (generate_dataset rescales its GT contours to that same area);
    descriptors (index particle_id) hold shape_area_um2 and shape_log_ecd_um.
    shortcut: no images, labels, annotation or render-time field-of-view drops; enough for the run-level physics /
    conditions baselines only (image models need generate_dataset)."""
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    sim = _simulate(cfg, seed)
    smp, parts = sim["samples"], sim["parts"]
    i = parts.sample_idx.to_numpy(int)
    sid = smp.sample_id.to_numpy(object)[i]
    pid = sid + "_p" + parts.groupby("sample_idx").cumcount().map("{:03d}".format).to_numpy(object)
    upp = smp.um_per_px.to_numpy(float)[i]
    area = parts.area.to_numpy(float)
    side = np.sqrt(area) / upp
    zero = np.zeros_like(side)
    views = pd.DataFrame({"view_id": pid + "_v0", "particle_id": pid, "view_idx": 0, "image_id": sid + "_f00",
                          "contour_x": list(np.stack([zero, side, side, zero], 1)),
                          "contour_y": list(np.stack([zero, zero, side, side], 1)), "um_per_px": upp, "source": "gt"})
    particles = pd.DataFrame({"particle_id": pid, "sample_id": sid, "image_id": sid + "_f00", "n_views": 1,
                              "true_mechanism": np.array(MECHANISMS)[parts.mech.to_numpy(int)],
                              "true_volume_um3": parts.vol.to_numpy(float)})
    ds = WearDataset(specimens=sim["specimens"], runs=sim["runs"],
                     samples=smp.drop(columns=["magnification", "um_per_px"]), particles=particles, views=views,
                     meta=dict(name=cfg.get("name", "synthetic"), is_synthetic=True, generator="wearpg.synthetic",
                               seed=seed, stats=_plain(sim["stats"]), warning=SYNTHETIC_BANNER))
    desc = pd.DataFrame({"shape_area_um2": area, "shape_log_ecd_um": np.log(np.sqrt(4 * area / np.pi))},
                        index=pd.Index(pid, name="particle_id"))
    return ds, desc


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b).to_numpy()
    return float(association(table, method="cramer")) if min(table.shape) > 1 else 0.0


# ============================================================================ rendering


def _fill(xy: np.ndarray, h: int, w: int) -> np.ndarray:
    """(h, w) bool mask of pixel centres (integer coords, x = column) inside polygon xy: vectorized even-odd scanline
    fill (skimage.draw.polygon tests every bbox pixel against every edge, ~100x slower on dense contours)."""
    x0, y0 = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    lo = np.clip(np.ceil(np.minimum(y0, y1)), 0, h).astype(int)   # edge crosses rows min(y) <= r < max(y)
    cnt = np.clip(np.ceil(np.maximum(y0, y1)), 0, h).astype(int) - lo
    e = np.repeat(np.arange(len(x0)), cnt)
    r = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + lo[e]
    xc = x0[e] + (r - y0[e]) / (y1[e] - y0[e]) * (x1[e] - x0[e])
    o = np.lexsort((xc, r))
    r, xc = r[o][0::2], xc[o]
    cols = np.clip(np.ceil(xc), 0, w).astype(int)   # centre c inside a span iff x_in <= c < x_out
    n = h * (w + 1)
    diff = np.bincount(r * (w + 1) + cols[0::2], minlength=n) - np.bincount(r * (w + 1) + cols[1::2], minlength=n)
    return np.cumsum(diff.reshape(h, w + 1), axis=1)[:, :w] > 0


def _coverage(xy: np.ndarray, h: int, w: int, sweep_rows: int = 0) -> np.ndarray:
    """Anti-aliased (_SS x _SS supersampled) coverage (h, w) float32 of polygon xy [px] on an (h, w) pixel grid,
    optionally swept by sweep_rows supersampled rows towards +y (Minkowski sum with a segment: a running sum)."""
    m = _fill((xy + 0.5) * _SS - 0.5, h * _SS, w * _SS)
    if sweep_rows > 0:
        c = np.cumsum(m, axis=0, dtype=np.int32)
        c[sweep_rows + 1:] -= c[:-sweep_rows - 1].copy()
        m = c > 0
    return m.reshape(h, _SS, w, _SS).mean(axis=(1, 3), dtype=np.float32)


def _background(rng: np.random.Generator, size: int, bg_rgb) -> tuple[np.ndarray, np.ndarray]:
    """Background canvas (size, size, 3) float32 [0-255] with a low-frequency illumination field (returned too)."""
    yy, xx = (np.mgrid[0:size, 0:size] / max(size - 1, 1) - 0.5).astype(np.float32)
    psi = rng.uniform(0, 2 * np.pi)
    illum = 1 + rng.uniform(0, 0.12) * (xx * math.cos(psi) + yy * math.sin(psi)) - rng.uniform(0, 0.1) * (xx ** 2 + yy ** 2)
    return np.asarray(bg_rgb, np.float32) * illum[..., None], illum


def _appearance(rng: np.random.Generator, a: dict, alpha: np.ndarray, centre: tuple[float, float], theta: float,
                upp: float, cap_luma: float) -> np.ndarray:
    """RGB patch (h, w, 3) [0-255] of one particle view: body colour x textures (sizes in um), luminance <= cap."""
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - centre[0], yy - centre[1]
    v = (-dx * math.sin(theta) + dy * math.cos(theta)) * upp      # um across the major axis
    mult = np.ones((h, w))
    if a["str_amp"] > 0:   # striations along the major axis
        mult *= 1 - 0.3 * a["str_amp"] * (0.5 + 0.5 * np.cos(2 * np.pi * v / a["str_period"] + a["str_phase"]))
    inside = alpha >= 0.5
    if a["pit_amp"] > 0 and inside.any():
        iy, ix = np.nonzero(inside)
        for j in rng.choice(len(iy), int(np.clip(rng.poisson(a["area"] / 15), 1, 30))):
            rad = rng.uniform(0.3, 1.0) / upp                         # px
            y0, y1 = max(int(iy[j] - 3 * rad), 0), min(int(iy[j] + 3 * rad) + 2, h)
            x0, x1 = max(int(ix[j] - 3 * rad), 0), min(int(ix[j] + 3 * rad) + 2, w)
            d2 = (yy[y0:y1, x0:x1] - iy[j]) ** 2 + (xx[y0:y1, x0:x1] - ix[j]) ** 2
            mult[y0:y1, x0:x1] *= 1 - 0.5 * a["pit_amp"] * np.exp(-d2 / (2 * rad * rad))
    mult *= 1 - 0.35 * np.exp(-ndimage.distance_transform_edt(inside) * upp / 0.6)   # rim darkening
    rgb = a["rgb"] * mult[..., None]
    if a["spec_amp"] > 0:  # specular highlight of a sphere, towards the luminance cap
        r_px = a["size"] / 2 / upp
        s = a["spec_amp"] * np.exp(-((dx + 0.3 * r_px) ** 2 + (dy + 0.3 * r_px) ** 2) / (2 * (0.25 * r_px) ** 2))
        rgb += (a["rgb"] * cap_luma / max(_luma(a["rgb"]), 1e-6) - rgb) * s[..., None]
    luma = rgb @ _LUMA
    return rgb * np.minimum(1.0, cap_luma / np.maximum(luma, 1e-6))[..., None]


def _finish(rng: np.random.Generator, canvas: np.ndarray, sigma: float, noise: float) -> np.ndarray:
    if sigma > 0:
        canvas = ndimage.gaussian_filter(canvas, (sigma, sigma, 0))
    canvas = canvas + rng.standard_normal(canvas.shape, dtype=np.float32) * np.float32(noise * 255)
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def _save_png(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, compress_level=1)


def _place(rng: np.random.Generator, masks: list[np.ndarray], size: int, touching_fraction: float) -> list:
    """Rejection-sampled placement (largest first) of particle masks into as many size x size images as needed.
    Free placements keep >= _GAP_PX px from other particles; with probability touching_fraction a particle is slid
    out of a placed neighbour until it no longer overlaps (contact). Returns (image, row, col) per mask."""
    g, out, occ, centres = _GAP_PX, [None] * len(masks), [], []
    st = ndimage.generate_binary_structure(2, 1)
    for i in np.argsort([-m.sum() for m in masks], kind="stable"):
        m = masks[i]
        h, w = m.shape
        hi_y, hi_x = size - _MARGIN_PX - h, size - _MARGIN_PX - w
        grown = ndimage.binary_dilation(np.pad(m, g), st, iterations=g)
        pos = None
        if occ and centres[-1] and rng.random() < touching_fraction:
            cy, cx = centres[-1][rng.integers(len(centres[-1]))]
            psi = rng.uniform(0, 2 * np.pi)
            for t in range(2 * size):
                y, x = int(round(cy + t * math.sin(psi) - h / 2)), int(round(cx + t * math.cos(psi) - w / 2))
                if _MARGIN_PX <= y <= hi_y and _MARGIN_PX <= x <= hi_x and not (occ[-1][y + g:y + g + h, x + g:x + g + w] & m).any():
                    pos = (len(occ) - 1, y, x)
                    break
        for k in range(len(occ) - 1, -1, -1) if pos is None else ():
            for _ in range(30):
                y, x = int(rng.integers(_MARGIN_PX, hi_y + 1)), int(rng.integers(_MARGIN_PX, hi_x + 1))
                if not (occ[k][y:y + h + 2 * g, x:x + w + 2 * g] & grown).any():
                    pos = (k, y, x)
                    break
            if pos is not None:
                break
        if pos is None:
            occ.append(np.zeros((size + 2 * g, size + 2 * g), bool))
            centres.append([])
            pos = (len(occ) - 1, int(rng.integers(_MARGIN_PX, hi_y + 1)), int(rng.integers(_MARGIN_PX, hi_x + 1)))
        k, y, x = pos
        occ[k][y + g:y + g + h, x + g:x + g + w] |= m
        centres[k].append((y + h / 2, x + w / 2))
        out[i] = pos
    return out


# ---------------------------------------------------------------------------- renderer v2 (imaging.renderer: v2)
# Rendering only: same particles, contours (up to a rigid rotation), physics, budgets and labels as v1. Constants are
# calibrated on Zenodo trainO ferrograms only (outputs/renderer/calibrate_trainO.py; docs/renderer_realism.md).
_BG_PALETTE = np.array([[151, 161, 142], [39, 252, 24], [136, 229, 156], [110, 129, 27], [85, 198, 92], [23, 14, 8]],
                       float)   # k-means centres of trainO background RGB (bichromatic: mostly green transmitted light)
_BG_W = np.array([0.327, 0.19, 0.178, 0.134, 0.128, 0.044])
_BG_SD = np.array([[20, 13, 18], [15, 10, 9], [30, 22, 27], [44, 24, 25], [29, 27, 25], [11, 19, 11]], float)
_HL_RGB = np.array([1.0, 0.86, 0.95])   # reflected-light (red/white) specular colour
_CHAIN_LAM = 6.0      # chain length = 1 + Poisson(_CHAIN_LAM) ferromagnetic particles
_ALIGN_SD = 0.2       # rad, scatter of the major axis about the field direction
_LAT_SD = 0.05        # lateral chain offset SD / particle width


def _field_angle(rng: np.random.Generator) -> float:
    """Magnetic-field direction in the image [rad, x = column, y = row]: trainO chains are vertical in most fields."""
    u = rng.random()
    base = np.pi / 2 if u < 0.85 else (0.0 if u < 0.95 else rng.uniform(0, np.pi))
    return float(base + rng.normal(0, 0.07))


def _align(xy: np.ndarray, ang: float, lim: int) -> np.ndarray:
    """Rotate a contour so its principal axis points along ang; the nearest image axis if that would not fit."""
    vec = np.linalg.eigh(np.cov(xy.T))[1][:, -1]
    for a in (ang, round(ang / (np.pi / 2)) * (np.pi / 2)):
        r = a - math.atan2(vec[1], vec[0])
        out = xy @ np.array([[math.cos(r), math.sin(r)], [-math.sin(r), math.cos(r)]])
        if max(np.ptp(out, axis=0)) <= lim:
            return out
    return xy


def _chains(rng: np.random.Generator, alphas: list[np.ndarray], order: list[int], u: np.ndarray, lim: int) -> list:
    """String particles (coverage patches, in order) into chains along unit vector u (x, y): each particle starts
    beyond the chain end with a lateral offset N(0, _LAT_SD x width), slides along -u until its coverage >= 0.5 core
    overlaps the chain's (so the chain is one component of the GT raster too), then is pushed in by U(0.1, 0.35) of
    its length w.p. 0.8 (agglomerate overlap). A chain is closed at 1 + Poisson(_CHAIN_LAM) particles or before it
    would exceed lim px. Returns [(composite coverage > 0 mask, [(i, dy, dx), ...]), ...]."""
    v = np.array([-u[1], u[0]])
    units, k = [], 0
    while k < len(order):
        n_max = 1 + int(rng.poisson(_CHAIN_LAM))
        big = 3 * lim + 2 * max(max(alphas[i].shape) for i in order[k:k + n_max])
        core, full = np.zeros((big, big), bool), np.zeros((big, big), bool)
        mem = []
        while len(mem) < n_max and k < len(order):
            i = order[k]
            m = alphas[i] >= 0.5
            h, w = m.shape
            if not mem:
                pos = (big // 2 - h // 2, big // 2 - w // 2)
            else:
                py, px = np.nonzero(core)
                proj = px * u[0] + py * u[1]
                last = mem[-1]
                c_last = np.array([last[2] + alphas[last[0]].shape[1] / 2, last[1] + alphas[last[0]].shape[0] / 2])
                half = 0.5 * (abs(w * u[0]) + abs(h * u[1]))       # half extent of the new bbox along u
                pos = None
                for lat in (float(c_last @ v) + rng.normal(0, _LAT_SD * min(h, w)), float(c_last @ v)):
                    c = (proj.max() + half + 2) * u + lat * v
                    for t in range(int(np.ptp(proj)) + 2 * max(h, w) + 4):
                        oy, ox = int(round(c[1] - t * u[1] - h / 2)), int(round(c[0] - t * u[0] - w / 2))
                        if min(oy, ox) >= 0 and core[oy:oy + h, ox:ox + w][m].any():
                            push = int(round(rng.uniform(0.1, 0.35) * max(h, w))) if rng.random() < 0.8 else 0
                            pos = (int(round(oy - push * u[1])), int(round(ox - push * u[0])))
                            break
                    if pos is not None:
                        break
                if pos is None or min(pos) < 0 or pos[0] + h > big or pos[1] + w > big:
                    break
                ys, xs = np.nonzero(full)
                ext = max(max(ys.max(), pos[0] + h - 1) - min(ys.min(), pos[0]),
                          max(xs.max(), pos[1] + w - 1) - min(xs.min(), pos[1])) + 1
                if ext > lim:
                    break
            core[pos[0]:pos[0] + h, pos[1]:pos[1] + w] |= m
            full[pos[0]:pos[0] + h, pos[1]:pos[1] + w] |= alphas[i] > 0
            mem.append((i, pos[0], pos[1]))
            k += 1
        y0, x0 = min(oy for _, oy, _ in mem), min(ox for _, _, ox in mem)   # bbox of the member patches
        y1 = max(oy + alphas[i].shape[0] for i, oy, _ in mem)
        x1 = max(ox + alphas[i].shape[1] for i, _, ox in mem)
        units.append((full[y0:y1, x0:x1].copy(), [(i, oy - y0, ox - x0) for i, oy, ox in mem]))
    return units


def _deposit(rng: np.random.Generator, alphas: list[np.ndarray], ferro: np.ndarray, ang: float, size: int,
             touching_fraction: float) -> list:
    """Ferrographic deposition -> (image, row, col) per coverage patch. Ferromagnetic debris (already field-aligned)
    forms chains along the field (size-ordered, _chains); non-ferrous debris deposits randomly as single units. Units
    are placed by _place (gap rule, as many fields as needed)."""
    order = [int(i) for i in np.argsort([-(a > 0).sum() for a in alphas], kind="stable")]
    units = _chains(rng, alphas, [i for i in order if ferro[i]], np.array([math.cos(ang), math.sin(ang)]),
                    _fov_limit_px(size))
    units += [(alphas[i] > 0, [(i, 0, 0)]) for i in order if not ferro[i]]
    out = [None] * len(alphas)
    for (_, mem), (k, y, x) in zip(units, _place(rng, [m for m, _ in units], size, touching_fraction)):
        for i, dy, dx in mem:
            out[i] = (k, y + dy, x + dx)
    return out


def _background_v2(rng: np.random.Generator, size: int) -> np.ndarray:
    """Bichromatic ferrogram background (size, size, 3) float32 [0-255]: trainO palette colour + jitter, smooth
    illumination field and oil-film blotches, dark speckle (dirt) and out-of-focus droplet rings."""
    k = int(rng.choice(len(_BG_W), p=_BG_W / _BG_W.sum()))
    bg = np.clip(_BG_PALETTE[k] + rng.normal(0, 0.7 * _BG_SD[k]), 4, 252)
    canvas, _ = _background(rng, size, bg)
    film = ndimage.gaussian_filter(rng.standard_normal((size, size)).astype(np.float32), size / 12)
    dark = 1 - rng.uniform(0.0, 0.05) * film / (film.std() + 1e-9)
    imp = np.zeros((size, size), np.float32)                     # speckle: dark Gaussian dots, sd 0.6-1.6 px
    n = int(rng.uniform(0.002, 0.04) * size * size / 6)
    sd = rng.uniform(0.6, 1.6)
    imp[rng.integers(0, size, n), rng.integers(0, size, n)] = rng.uniform(0.05, 0.3, n) * 2 * np.pi * sd * sd
    dark = dark - ndimage.gaussian_filter(imp, sd)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(int(rng.poisson(4))):                         # droplets: bright centre, dark rim
        cy, cx, r = rng.uniform(0, size, 2).tolist() + [rng.uniform(2, 10)]
        d = np.hypot(yy - cy, xx - cx)
        dark = dark * (1 - rng.uniform(0.05, 0.2) * np.exp(-((d - r) / 0.8) ** 2) + 0.04 * (d < r))
    return canvas * dark[..., None]


def _shade_v2(rng: np.random.Generator, a: dict, alpha: np.ndarray, centre: tuple[float, float], theta: float,
              upp: float, under: np.ndarray) -> np.ndarray:
    """RGB patch (h, w, 3) [0-255] of one particle under bichromatic light over the background patch `under`: body =
    leaked transmitted light (background hue x tau) + diffuse reflection (material hue x reflectance), both with
    spatially correlated relief shading; specular facets mix towards the white/red reflected light; dark rim;
    striations, pits, oxide tint (in a["rgb"]) and the sphere highlight as in v1."""
    h, w = alpha.shape
    inside = alpha >= 0.5
    if not inside.any():
        inside = alpha > 0
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - centre[0], yy - centre[1]
    width_px = 2 * math.sqrt(inside.sum() / np.pi)
    s = float(np.clip(0.04 * width_px, 0.8, 2.0))                # facet / relief correlation length [px]
    # relief elongated along the major axis; shortcut: anisotropy snapped to the nearest image axis (most particles
    # are field-aligned, i.e. near an axis); use a rotated kernel if diagonal fields matter
    n1 = ndimage.gaussian_filter(rng.standard_normal((h, w)), (4 * s, 1.5 * s) if abs(math.sin(theta)) > 0.7071
                                 else (1.5 * s, 4 * s))
    n2 = ndimage.gaussian_filter(rng.standard_normal((h, w)), s)
    n1 /= n1[inside].std() + 1e-9
    n2 = (n2 - n2[inside].mean()) / (n2[inside].std() + 1e-9)
    level = float(_luma(a["rgb"])) / 255                         # material luminance x reflectance (x oxide tint)
    tint = a["rgb"] / max(float(_luma(a["rgb"])), 1e-6)
    t_dark = np.quantile(n2[inside], rng.uniform(0.1, 0.35))      # dark facets / cores (sharp-edged)
    mult = (1 + 0.2 * n1 + 0.03 * rng.standard_normal((h, w))) * (1 - 0.5 * np.clip((t_dark - n2) / 0.2, 0, 1))
    if a["str_amp"] > 0:
        v = (-dx * math.sin(theta) + dy * math.cos(theta)) * upp
        mult *= 1 - 0.3 * a["str_amp"] * (0.5 + 0.5 * np.cos(2 * np.pi * v / a["str_period"] + a["str_phase"]))
    if a["pit_amp"] > 0:
        iy, ix = np.nonzero(inside)
        for j in rng.choice(len(iy), int(np.clip(rng.poisson(a["area"] / 15), 1, 30))):
            rad = rng.uniform(0.3, 1.0) / upp
            mult *= 1 - 0.5 * a["pit_amp"] * np.exp(-((yy - iy[j]) ** 2 + (xx - ix[j]) ** 2) / (2 * rad * rad))
    body = (rng.uniform(0.45, 0.85) * under + 255 * level * rng.uniform(0.1, 0.4) * tint) * mult[..., None]
    t = np.quantile(n2[inside], 1 - rng.uniform(0.03, 0.35) * min(1.0, level / 0.35))
    spec = np.clip((n2 - t) / 0.2, 0, 1)
    if a["spec_amp"] > 0:   # sphere: one bright reflection off-centre
        r_px = a["size"] / 2 / upp
        spec = np.maximum(spec, a["spec_amp"] * np.exp(-((dx + 0.3 * r_px) ** 2 + (dy + 0.3 * r_px) ** 2)
                                                       / (2 * (0.25 * r_px) ** 2)))
    hl = under + (255 * _HL_RGB * tint ** 0.6 - under) * rng.uniform(0.35, 0.9)
    rgb = body * (1 - spec[..., None]) + hl * spec[..., None]
    rgb *= (1 - rng.uniform(0.0, 0.3) * np.exp(-ndimage.distance_transform_edt(inside) / 0.8))[..., None]
    return np.clip(rgb, 0, 255)


def _fov_v2(rng: np.random.Generator, items: list, size: int, upp: float, focus: float, noise: float) -> np.ndarray:
    """One v2 ferrograph field: background, particles (alpha compositing in list order), dark halo around the
    particles, optical blur sqrt(psf^2 + focus^2) and white noise. items: (look, alpha, centre, theta, y, x)."""
    canvas = _background_v2(rng, size)
    cov = np.zeros((size, size), np.float32)
    for look, alpha, centre, theta, y, x in items:
        h, w = alpha.shape
        under = canvas[y:y + h, x:x + w]
        a = alpha[..., None]
        canvas[y:y + h, x:x + w] = under * (1 - a) + _shade_v2(rng, look, alpha, centre, theta, upp, under) * a
        cov[y:y + h, x:x + w] = np.maximum(cov[y:y + h, x:x + w], alpha)
    if items:
        d = ndimage.distance_transform_edt(cov < 0.5)
        canvas *= (1 - rng.uniform(0.05, 0.3) * np.exp(-d / 1.5) * (1 - cov))[..., None]
    sigma = math.hypot(rng.uniform(0.5, 0.9), focus)
    canvas = ndimage.gaussian_filter(canvas, (sigma, sigma, 0))
    white = rng.standard_normal(canvas.shape).astype(np.float32)   # drawn before its SD (frozen draw order)
    canvas = canvas + white * np.float32(255 * rng.uniform(0.25, 1) * noise)
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def _silhouette(rng, planform_px: np.ndarray, thick_um: float, sphere: bool, upp_s: float, frame: int):
    """Silhouette of the planform extruded by thick_um and tilted by 30-80 deg about a random in-plane axis (spheres:
    the planform itself): projected planform (compressed by cos tilt) swept by thick*sin tilt along +y. Frame px
    size = max(upp_s, 1.3 x max(bbox w, h) / frame), so the silhouette never touches the border; the GT contour is
    the 0.5 iso-line of the anti-aliased coverage. Returns (alpha (frame, frame), contour px (n, 2), um_per_px)."""
    xy_um = planform_px * upp_s
    ang = rng.uniform(0, np.pi)
    tau = 0.0 if sphere else rng.uniform(np.radians(30), np.radians(80))
    u = xy_um @ np.array([math.cos(ang), math.sin(ang)])
    v = xy_um @ np.array([-math.sin(ang), math.cos(ang)]) * math.cos(tau)
    sweep = thick_um * math.sin(tau)
    upp = max(upp_s, 1.3 * max(np.ptp(u), np.ptp(v) + sweep) / frame)
    x = (u - (u.max() + u.min()) / 2) / upp + (frame - 1) / 2
    y = (v - (v.max() + v.min() + sweep) / 2) / upp + (frame - 1) / 2
    x0, y0 = int(x.min()) - 1, int(y.min()) - 1
    x1, y1 = int(np.ceil(x.max())) + 2, int(np.ceil(y.max() + sweep / upp)) + 2
    alpha = np.zeros((frame, frame), np.float32)
    alpha[y0:y1, x0:x1] = _coverage(np.c_[x - x0, y - y0], y1 - y0, x1 - x0, int(round(sweep / upp * _SS)))
    level = 0.5 * min(1.0, float(alpha.max()))   # sub-pixel-thin silhouettes never reach coverage 0.5
    xy = max(find_contours(np.pad(alpha, 1), level), key=len)[:-1, ::-1] - 1.0
    xy = _resample(xy, int(np.clip(round(_perimeter(xy) / 1.5), 32, 512)))
    return alpha, (xy if _shoelace(xy) > 0 else xy[::-1]), upp


def _bbox(xy: np.ndarray, w: int, h: int) -> dict:
    return dict(bbox_x0=max(int(np.floor(xy[:, 0].min())), 0), bbox_y0=max(int(np.floor(xy[:, 1].min())), 0),
                bbox_x1=min(int(np.ceil(xy[:, 0].max())) + 1, w), bbox_y1=min(int(np.ceil(xy[:, 1].max())) + 1, h))


def _render(sim: dict, cfg: dict, seed: int, root: Path, flags: dict, progress: bool) -> dict:
    """Contours, FOV images and multi-view frames for the stored particles; writes PNGs under root/images."""
    img, mv = cfg["imaging"], cfg["multiview"]
    mats = {m["material_id"]: m for m in cfg["materials"]}
    samples, parts = sim["samples"], sim["parts"]   # samples.n_particles_total is updated in place
    size, bg_rgb = int(img["image_size_px"]), img["background_rgb"]
    bg_luma = _luma(bg_rgb)
    v2 = img.get("renderer", "v1") == "v2"
    images, views, prow = [], [], []
    d_meas = np.full(len(parts), np.nan)
    n_dropped = 0
    by_sample = parts.groupby("sample_idx").indices
    it = range(len(samples))
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="synthetic samples")
    for i in it:
        s = samples.iloc[i]
        rng = np.random.default_rng([seed, 2, i])
        rng2 = np.random.default_rng([seed, 9, i])   # v2 draws only, so v1 contours and v1 images stay identical
        field = _field_angle(rng2) if v2 else 0.0
        upp, mag = float(s.um_per_px), float(s.magnification)
        mid = sim["material"][i]
        cf_id = mats[mid]["counterface_id"]
        idx = by_sample.get(i, np.empty(0, int))
        geo = []
        for j in idx:
            p = parts.iloc[j]
            shape = MECHANISM_PRIORS[MECHANISMS[p.mech]]["shape"]
            xy = _contour_px(rng, shape, p["size"], p.minor, p.d_target, _amplitude(p.d_target), upp,
                             rng.uniform(0, 2 * np.pi), p.area)
            lim = _fov_limit_px(size)
            if max(np.ptp(xy, axis=0)) > lim:  # too big for this orientation: align the principal axis with x
                vec = np.linalg.eigh(np.cov(xy.T))[1][:, -1]
                ang = -math.atan2(vec[1], vec[0])
                xy = xy @ np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
            if max(np.ptp(xy, axis=0)) > lim:   # cannot fit any field: border-excluded, not counted as detected
                n_dropped += 1
                continue
            src_mat = mats[mid if p.src == 0 else cf_id]
            if v2 and src_mat["ferromagnetic"]:   # magnetic alignment of the major axis with the field (rigid rotation)
                xy = _align(xy, field + rng2.normal(0, _ALIGN_SD), lim)
            loc = xy + rng.random(2) - np.floor(xy.min(axis=0)) + 1.0   # sub-pixel position, 1 px patch border
            hw = (int(np.ceil(loc[:, 1].max())) + 2, int(np.ceil(loc[:, 0].max())) + 2)
            alpha = _coverage(loc, *hw)
            ru = rng.random(10)
            mech = MECHANISMS[p.mech]
            pr = MECHANISM_PRIORS[mech]
            rgb = np.asarray(src_mat["color_rgb"], float) * p.refl
            if ru[2] < pr["tint"]:   # oxide tint: dark or reddish
                rgb = rgb * ((1 - 0.4 * (0.3 + 0.7 * ru[3])) if ru[4] < 0.5
                             else np.array([1.0, 1 - 0.25 * (0.3 + 0.7 * ru[3]), 1 - 0.45 * (0.3 + 0.7 * ru[3])]))
            look = dict(rgb=rgb, area=p.area, size=p["size"],
                        str_amp=(0.3 + 0.7 * ru[5]) if ru[0] < pr["striation"] else 0.0,
                        str_period=1.5 + 2.5 * ru[6], str_phase=2 * np.pi * ru[7],
                        pit_amp=(0.3 + 0.7 * ru[8]) if ru[1] < pr["pits"] else 0.0,
                        spec_amp=(0.3 + 0.7 * ru[9]) if mech == "spherical" else 0.0)
            vec = np.linalg.eigh(np.cov(loc.T))[1][:, -1]
            geo.append(dict(j=j, loc=loc, alpha=alpha, look=look, theta=math.atan2(vec[1], vec[0]),
                            ferro=bool(src_mat["ferromagnetic"])))
        n_used = len(geo)
        samples.loc[samples.index[i], "n_particles_total"] = int(s.n_particles_total) - (len(idx) - n_used)
        place = (_deposit(rng2, [gq["alpha"] for gq in geo], np.array([gq["ferro"] for gq in geo], bool), field,
                          size, float(img["touching_fraction"])) if v2
                 else _place(rng, [gq["alpha"] > 0 for gq in geo], size, float(img["touching_fraction"])))
        n_img = max([pos[0] for pos in place], default=0) + 1   # empty samples still get one (empty) field
        for k in range(n_img):
            image_id = f"{s.sample_id}_f{k:02d}"
            rel = f"images/{s.run_id}/{image_id}.png"
            if not v2:
                canvas, illum = _background(rng, size, bg_rgb)
            items = []
            for gq, (kk, y, x) in zip(geo, place):
                if kk != k:
                    continue
                h, w = gq["alpha"].shape
                cy, cx = gq["loc"].mean(axis=0)[::-1]
                if v2:
                    items.append((gq["look"], gq["alpha"], (cx, cy), gq["theta"], y, x))
                else:
                    cap = img["max_luminance_ratio"] * bg_luma * illum[min(int(y + cy), size - 1),
                                                                       min(int(x + cx), size - 1)]
                    rgb = _appearance(rng, gq["look"], gq["alpha"], (cx, cy), gq["theta"], upp, cap)
                    a = gq["alpha"][..., None]
                    canvas[y:y + h, x:x + w] = canvas[y:y + h, x:x + w] * (1 - a) + rgb * a
                xy = gq["loc"] + (x, y)
                pid = f"{s.sample_id}_p{gq['j'] - idx[0]:03d}"
                gq.update(pid=pid, image_id=image_id)
                views.append(dict(view_id=f"{pid}_v0", particle_id=pid, view_idx=0, image_id=image_id,
                                  **_bbox(xy, size, size), contour_x=xy[:, 0], contour_y=xy[:, 1], um_per_px=upp,
                                  source="gt"))
                d = _divider_dimension(xy)
                d_meas[gq["j"]] = d["D"] if d["valid"] else np.nan
            if v2:
                sigma = float(rng2.choice(img["focus_sigmas_px"]))
                _save_png(_fov_v2(rng2, items, size, upp, sigma, img["gaussian_noise"]), root / rel)
            else:
                sigma = float(rng.choice(img["focus_sigmas_px"]))
                _save_png(_finish(rng, canvas, sigma, img["gaussian_noise"]), root / rel)
            images.append(dict(image_id=image_id, sample_id=s.sample_id, path=rel, modality="ferrograph_optical",
                               magnification=mag, um_per_px=upp, focus_level=sigma,
                               illumination="bichromatic" if v2 else "reflected", width_px=size, height_px=size))
        frame = int(mv["frame_size_px"])
        for gq in geo:
            n_views = int(mv["n_views"]) if flags["multiview"][gq["j"]] else 1
            p = parts.iloc[gq["j"]]
            for vi in range(1, n_views):
                alpha, xy, upp_f = _silhouette(rng, gq["loc"] - gq["loc"].mean(axis=0), p.thick,
                                               MECHANISMS[p.mech] == "spherical", upp, frame)
                image_id = f"{gq['pid']}_m{vi}"
                rel = f"images/{s.run_id}/{image_id}.png"
                canvas, illum = _background(rng, frame, bg_rgb)
                cap = img["max_luminance_ratio"] * bg_luma * illum[frame // 2, frame // 2]
                rgb = _appearance(rng, gq["look"], alpha, tuple(xy.mean(axis=0)), 0.0, upp_f, cap)
                canvas = canvas * (1 - alpha[..., None]) + rgb * alpha[..., None]
                sigma = float(rng.choice(img["focus_sigmas_px"]))
                _save_png(_finish(rng, canvas, sigma, img["gaussian_noise"]), root / rel)
                images.append(dict(image_id=image_id, sample_id=s.sample_id, path=rel, modality="multiview_frame",
                                   magnification=mag * upp / upp_f, um_per_px=upp_f, focus_level=sigma,
                                   illumination="transmitted", width_px=frame, height_px=frame))
                views.append(dict(view_id=f"{gq['pid']}_v{vi}", particle_id=gq["pid"], view_idx=vi, image_id=image_id,
                                  **_bbox(xy, frame, frame), contour_x=xy[:, 0], contour_y=xy[:, 1], um_per_px=upp_f,
                                  source="gt"))
            prow.append(dict(j=gq["j"], particle_id=gq["pid"], sample_id=s.sample_id, image_id=gq["image_id"],
                             n_views=n_views))
    prow = pd.DataFrame(prow, columns=["j", "particle_id", "sample_id", "image_id", "n_views"])
    # explicit columns: schema.coerce_table types missing columns of a 0-row frame as float64 (validate then raises)
    views = pd.DataFrame(views, columns=list(TABLES["views"]))
    return dict(images=pd.DataFrame(images), views=views, prow=prow, d_meas=d_meas, n_dropped=n_dropped)


# ============================================================================ annotation


def _rate(rng: np.random.Generator, true_idx: np.ndarray, acc: float) -> tuple[np.ndarray, np.ndarray]:
    """Rater labels: correct w.p. acc, else uniform among the confusable neighbours; confidence U{3,4,5} if correct
    else U{1,2,3}."""
    n = len(true_idx)
    ok = rng.random(n) < acc
    wrong = _CONF_IDX[true_idx, (rng.random(n) * _CONF_N[true_idx]).astype(int)]
    return np.where(ok, true_idx, wrong), np.where(ok, rng.integers(3, 6, n), rng.integers(1, 4, n))


def _annotate_labels(rng: np.random.Generator, pids: np.ndarray, true_idx: np.ndarray, has_sem: np.ndarray,
                     has_eds: np.ndarray, ann: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raters A, B (+ reference SEMREF on has_sem | has_eds) and A/B adjudication -> (labels, labels_adjudicated)."""
    n = len(pids)
    labelled = rng.random(n) < ann["label_fraction"]
    double = labelled & (rng.random(n) < ann["double_label_fraction"])
    evidence = np.select([has_sem & has_eds, has_sem, has_eds], ["sem_eds", "sem", "eds"], "optical")
    acc_a, acc_b = ann["rater_accuracy"]
    la, ca = _rate(rng, true_idx, acc_a)
    lb, cb = _rate(rng, true_idx, acc_b)
    lr, cr = _rate(rng, true_idx, ann["reference_accuracy"])
    ref = has_sem | has_eds
    names = np.array(MECHANISMS)
    labels = pd.concat([pd.DataFrame(dict(particle_id=pids[m], rater_id=r, label=names[lab[m]], confidence=conf[m],
                                          evidence=evidence[m]))
                        for r, m, lab, conf in (("A", labelled, la, ca), ("B", double, lb, cb),
                                                ("SEMREF", ref, lr, cr))], ignore_index=True)
    # adjudicator: correct w.p. adjudicator_accuracy, otherwise one of the raters' (wrong) labels
    adj_ok = rng.random(n) < ann["adjudicator_accuracy"]
    pick_a = rng.random(n) < 0.5
    cand = np.where(pick_a, la, lb)
    final = np.where(adj_ok, true_idx, np.where(cand == true_idx, np.where(pick_a, lb, la), cand))
    final = np.where(double & (la != lb), final, la)
    method = np.where(~double, "single", np.where(la == lb, "unanimous", "adjudicated"))
    agreement = np.where(double, ((la == final).astype(float) + (lb == final)) / 2, 1.0)
    adj = pd.DataFrame(dict(particle_id=pids, label=names[final], method=method, n_raters=np.where(double, 2, 1),
                            agreement=agreement))[labelled]
    return labels, adj


def _eds(rng: np.random.Generator, material: dict, oxide: bool, fe_transfer: bool) -> np.ndarray:
    """EDS mass fractions over ELEMENTS (sum 1): material composition, + O 0.2-0.35 for oxide, + up to 3 % Fe
    transfer from a steel counterface, Dirichlet noise."""
    base = np.array([material["composition"].get(e, 0.0) for e in ELEMENTS], float)
    base /= base.sum()
    if fe_transfer:
        base[ELEMENTS.index("Fe")] += rng.uniform(0, 0.03)
        base /= base.sum()
    if oxide:
        o = rng.uniform(0.2, 0.35)
        base = base * (1 - o)
        base[ELEMENTS.index("O")] += o
    return rng.dirichlet(200 * base + 0.05)


# ============================================================================ public API


def generate_dataset(cfg: dict, out_dir: str | Path, seed: int | None = None, progress: bool = False) -> WearDataset:
    """Generate a synthetic WearDataset (tables + PNG images) in out_dir. Synthetic: never evidence (meta banner).

    cfg keys: configs/synthetic_pilot.yaml. seed None -> cfg["seed"]. Returns the saved, validated dataset whose
    meta carries name, is_synthetic, generator, seed, generator_config, stats and the warning banner."""
    t_start = time.perf_counter()
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    root = Path(out_dir)
    sim = _simulate(cfg, seed)
    parts = sim["parts"]
    ann, mv = cfg["annotation"], cfg["multiview"]
    rng_f = np.random.default_rng([seed, 6])
    n = len(parts)
    flags = dict(has_eds=rng_f.random(n) < ann["eds_fraction"], has_sem=rng_f.random(n) < ann["sem_fraction"],
                 confocal=rng_f.random(n) < ann["confocal_fraction"], multiview=rng_f.random(n) < mv["fraction"])
    ren = _render(sim, cfg, seed, root, flags, progress)

    pr = ren["prow"]
    j = pr.j.to_numpy(int)
    sub = parts.iloc[j].reset_index(drop=True)
    mech = sub.mech.to_numpy(int)
    has_eds, has_sem, conf = flags["has_eds"][j], flags["has_sem"][j], flags["confocal"][j]
    mats = {m["material_id"]: m for m in cfg["materials"]}
    mat_of = sim["material"][sub.sample_idx.to_numpy(int)]
    rng_e = np.random.default_rng([seed, 7])
    eds = np.full((len(sub), len(ELEMENTS)), np.nan)
    for row in np.flatnonzero(has_eds):
        spec = mats[mat_of[row]]
        cf = mats.get(spec["counterface_id"])
        from_cf = sub.src[row] == 1
        eds[row] = _eds(rng_e, cf if from_cf else spec, mech[row] == _OXIDE,
                        not from_cf and cf is not None and bool(cf["ferromagnetic"]))
    rng_c = np.random.default_rng([seed, 8])
    ln_t, ln_v = rng_c.normal(0, 0.1, len(sub)), rng_c.normal(0, 0.15, len(sub))
    particles = pd.DataFrame(dict(
        particle_id=pr.particle_id.to_numpy(object), sample_id=pr.sample_id.to_numpy(object),
        image_id=pr.image_id.to_numpy(object), n_views=pr.n_views.to_numpy(int),
        has_eds=has_eds, **{c: eds[:, k] for k, c in enumerate(EDS_COLUMNS)}, has_sem=has_sem,
        thickness_um=np.where(conf, sub.thick.to_numpy(float) * np.exp(ln_t), np.nan),
        volume_um3=np.where(conf, sub.vol.to_numpy(float) * np.exp(ln_v), np.nan),
        true_mechanism=np.array(MECHANISMS)[mech], true_volume_um3=sub.vol.to_numpy(float),
        true_fractal_D_target=sub.d_target.to_numpy(float), true_fractal_D=ren["d_meas"][j]))
    labels, adj = _annotate_labels(np.random.default_rng([seed, 3]), particles.particle_id.to_numpy(object), mech,
                                   has_sem, has_eds, ann)

    stats = dict(sim["stats"])
    stats.update(n_stored=int(len(particles)), n_fov_dropped=int(ren["n_dropped"]),
                 n_fov_images=int((ren["images"].modality == "ferrograph_optical").sum()),
                 n_multiview_frames=int((ren["images"].modality == "multiview_frame").sum()),
                 fractal=_fractal_stats(particles, ren["views"], sim["samples"]))
    samples = sim["samples"].drop(columns=["magnification", "um_per_px"])
    ds = WearDataset(specimens=sim["specimens"], runs=sim["runs"], samples=samples, images=ren["images"],
                     particles=particles, views=ren["views"], labels=labels, labels_adjudicated=adj,
                     meta=dict(name=cfg.get("name", "synthetic"), is_synthetic=True, generator="wearpg.synthetic",
                               seed=seed, generator_config=cfg, stats=_plain(stats), warning=SYNTHETIC_BANNER))
    ds.assert_valid()
    ds.save(root)
    get_logger().info("synthetic dataset %r: %d runs, %d samples, %d particles, %d images in %.1f s (%s)",
                      ds.meta["name"], len(ds.runs), len(ds.samples), len(ds.particles), len(ds.images),
                      time.perf_counter() - t_start, SYNTHETIC_BANNER)
    return ds


def _fractal_stats(particles: pd.DataFrame, views: pd.DataFrame, samples: pd.DataFrame) -> dict:
    """Achieved divider D vs D_target on view-0 GT contours with perimeter >= 100 px (valid estimates only)."""
    if particles.empty:
        return {}
    v0 = views[views.view_idx == 0].set_index("particle_id")
    per = np.array([_perimeter(WearDataset.contour(v0.loc[p])) for p in particles.particle_id])
    mag = particles.sample_id.map(samples.set_index("sample_id").magnification).to_numpy(float)
    d_t, d_m = particles.true_fractal_D_target.to_numpy(float), particles.true_fractal_D.to_numpy(float)
    ok = (per >= 100) & np.isfinite(d_m)
    out = {"achieved_D_mean_by_class": particles.groupby("true_mechanism").true_fractal_D.mean().to_dict(),
           "target_D_mean_by_class": particles.groupby("true_mechanism").true_fractal_D_target.mean().to_dict()}
    for m in np.unique(mag):
        sel = ok & (mag == m)
        out[f"spearman_mag{m:g}"] = float(spearmanr(d_t[sel], d_m[sel])[0]) if sel.sum() > 2 else float("nan")
        out[f"n_mag{m:g}"] = int(sel.sum())
    return out


def _plain(obj):
    """JSON-safe plain Python types (numpy scalars -> float/int)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj
