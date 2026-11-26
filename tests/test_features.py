import time

import numpy as np
import pytest

from wearpg.features import contour as C
from wearpg.features.descriptors import DESCRIPTOR_PREFIXES, particle_descriptors, view_descriptors
from wearpg.features.fourier import efd_feature_vector, elliptic_fourier_descriptors
from wearpg.features.fractal import (box_counting_dimension, divider_dimension, fractal_descriptors,
                                     multiscale_perimeter_slopes)
from wearpg.features.shape import shape_descriptors
from wearpg.features.texture import boundary_band_profile, color_descriptors, texture_descriptors


# ----------------------------------------------------------------------------- shapes


def circle(r, n=256, c=(0.0, 0.0)):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def ellipse(a, b, n=400):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[a * np.cos(t), b * np.sin(t)]


def blob(n=400, r0=60.0):
    """Asymmetric star-shaped contour with odd and even EFD harmonics and a unique max radius."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r0 * (1 + 0.25 * np.cos(2 * t) + 0.12 * np.sin(3 * t + 0.4) + 0.08 * np.cos(5 * t + 1.0))
    return np.c_[r * np.cos(t), r * np.sin(t)]


def rough(r0=150.0, dim=1.25, amp=0.4, n=3000, seed=0):
    """Radial fractal roughness r *= 1 + amp sum_k k^-(H + 0.5) cos(k t + phi), H = 2 - D."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = np.arange(2, 400)[:, None]
    r = r0 * (1 + amp * np.sum(k ** -(2 - dim + 0.5) * np.cos(k * t + rng.uniform(0, 2 * np.pi, k.shape)), axis=0))
    return np.c_[r * np.cos(t), r * np.sin(t)]


def koch(level=5, side=400.0):
    pts = np.array([[0, 0], [side, 0], [side / 2, side * np.sqrt(3) / 2]], float)
    rot = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
    for _ in range(level):
        new = []
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            d = (b - a) / 3
            new += [a, a + d, a + d + rot @ d, a + 2 * d]
        pts = np.array(new)
    return pts


def similar(xy, angle=0.0, scale=1.0, shift=(0.0, 0.0), roll=0):
    c, s = np.cos(angle), np.sin(angle)
    return np.roll(xy @ np.array([[c, s], [-s, c]]) * scale + np.asarray(shift), roll, axis=0)


# ----------------------------------------------------------------------------- contour


def test_clean_orientation_perimeter():
    sq = np.array([[0, 0], [2, 0], [2, 0], [2, 2], [0, 2], [0, 0]], float)
    c = C.clean_closed(sq)
    assert c.shape == (4, 2)
    assert C.signed_area(c) == pytest.approx(4.0) and C.perimeter(c) == pytest.approx(8.0)
    cw = c[::-1]
    assert C.signed_area(cw) == pytest.approx(-4.0)
    fixed = C.ensure_ccw(cw)
    assert C.signed_area(fixed) == pytest.approx(4.0) and np.allclose(fixed[0], cw[0])
    assert C.signed_area(circle(10, 512)) == pytest.approx(np.pi * 100, rel=1e-3)


def test_area_moments_rectangle():
    w, h = 6.0, 2.0
    rect = similar(np.array([[0, 0], [w, 0], [w, h], [0, h]], float), angle=0.7, shift=(5, -3))
    a, cen, cov = C.area_moments(rect)
    assert a == pytest.approx(w * h)
    assert np.allclose(cen, similar(np.array([[w / 2, h / 2]]), angle=0.7, shift=(5, -3))[0])
    assert np.allclose(np.linalg.eigvalsh(cov), [h * h / 12, w * w / 12])
    a2, _, cov2 = C.area_moments(rect[::-1])  # orientation independent except the area sign
    assert a2 == pytest.approx(-w * h) and np.allclose(cov2, cov)


def test_resample_closed():
    xy = blob()
    r = C.resample_closed(xy, 128)
    seg = np.linalg.norm(np.diff(np.vstack([r, r[:1]]), axis=0), axis=1)
    assert r.shape == (128, 2) and seg.std() / seg.mean() < 0.02
    assert C.signed_area(C.resample_closed(xy[::-1], 64)) > 0
    assert np.allclose(C.resample_closed(np.roll(xy, 37, axis=0), 128), r)  # canonical start
    assert np.allclose(C.resample_closed(xy, 50, start="first")[0], xy[0])
    with pytest.raises(ValueError):
        C.resample_closed(xy, 10, start="middle")


def test_canonicalize_properties():
    out, info = C.canonicalize(similar(blob(), angle=0.4, scale=2.0, shift=(30, 40)))
    a, cen, cov = C.area_moments(out)
    assert np.allclose(cen, 0, atol=1e-9)
    assert C._rms_radius(out) == pytest.approx(1.0)
    assert abs(cov[0, 1]) < 1e-9 and cov[0, 0] > cov[1, 1]
    rad = C.radial_signature(out)
    assert rad.argmax() == 0 and out[0, 0] > 0
    assert info["area"] == pytest.approx(C.signed_area(blob()) * 4) and info["centroid"].shape == (2,)


def test_canonicalize_invariance():
    xy = blob()
    ref, info0 = C.canonicalize(xy, n=128)
    out, info = C.canonicalize(similar(xy, angle=1.1, scale=3.7, shift=(40, -7), roll=53), n=128)
    assert np.allclose(out, ref, atol=1e-6)
    assert info["scale"] / info0["scale"] == pytest.approx(3.7)
    assert np.cos(info["angle"] - info0["angle"] - 1.1) == pytest.approx(1.0)
    assert info["perimeter"] / info0["perimeter"] == pytest.approx(3.7)


def test_curvature_circle_and_sign():
    for xy in (circle(10, 512), circle(10, 512)[::-1]):
        k = C.curvature(xy)
        assert k.shape == (512, 4) and np.allclose(k, 0.1, rtol=1e-2)
    notch = C.resample_closed(np.array([[0, 0], [10, 0], [10, 10], [5, 4], [0, 10]], float), 200)
    k = C.curvature(notch, scales=(2,))[:, 0]
    assert k.min() < 0 < k.max()  # concave notch negative, convex corners positive


def test_smooth_closed():
    xy = circle(10, 64, c=(3, 4))
    assert np.allclose(C.smooth_closed(xy, 0), xy)
    sm = C.smooth_closed(xy, 4)
    assert np.allclose(sm.mean(axis=0), [3, 4]) and C.radial_signature(sm).max() < 10


def test_turning_normals_tangents():
    for xy in (blob(), blob()[::-1]):
        assert C.turning_angle(xy).sum() == pytest.approx(2 * np.pi)
    for xy in (circle(5, 100, c=(2, 1)), circle(5, 100, c=(2, 1))[::-1]):
        nrm = C.inward_normals(xy)
        assert np.allclose(nrm, -(xy - [2, 1]) / 5, atol=1e-3)
        assert np.allclose(np.sum(nrm * C.tangents(xy), axis=1), 0, atol=1e-9)


def test_convexity_defect_radial_roughness():
    assert C.convexity_defect(circle(10, 100)).max() < 1e-9
    notch = np.array([[0, 0], [10, 0], [10, 10], [5, 6], [0, 10]], float)
    assert np.allclose(C.convexity_defect(notch), [0, 0, 0, 4, 0])
    assert np.allclose(C.radial_signature(circle(7, 64, c=(1, 2))), 7)
    smooth = circle(50, 400)
    assert C.local_roughness(smooth).max() < 1e-3
    noisy = smooth + np.random.default_rng(0).normal(0, 0.5, smooth.shape)
    assert C.local_roughness(noisy).mean() > 0.2


def test_polygon_mask_and_distance():
    m = C.polygon_mask(circle(20, 200, c=(50, 40)), (100, 120))
    assert m.shape == (100, 120) and m[40, 50] and not m[40, 80]  # row = y, col = x
    assert m.sum() == pytest.approx(np.pi * 400, rel=0.02)
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert np.allclose(C.distance_to_boundary(sq, [[5, 5], [2, 5], [5, 9]]), [5, 2, 1])
    pts = np.random.default_rng(0).uniform(1, 9, (2500, 2))  # > one chunk
    expect = np.minimum.reduce([pts[:, 0], pts[:, 1], 10 - pts[:, 0], 10 - pts[:, 1]])
    assert np.allclose(C.distance_to_boundary(sq, pts), expect)


# ----------------------------------------------------------------------------- fractal


def test_divider_circle_and_tiny():
    d = divider_dimension(circle(100, 2000))
    assert d["valid"] and 1.0 <= d["D"] < 1.02
    from skimage.draw import disk
    from skimage.measure import find_contours
    m = np.zeros((160, 160), bool)
    m[disk((80, 80), 60)] = True
    px = find_contours(m.astype(np.uint8), 0.5)[0][:, ::-1]
    d = divider_dimension(px)
    assert d["valid"] and d["D"] < 1.02
    tiny = divider_dimension(circle(10, 200))  # L = 20 px < 40 px: fewer than a 2x step span above 2 px
    assert tiny["valid"] is False and tiny["D"] == 1.0 and np.isnan(tiny["r2"])


def test_divider_rough_larger():
    smooth = divider_dimension(circle(150, 3000))["D"]
    k = divider_dimension(koch())
    assert k["valid"] and k["D"] > 1.15 and k["D"] > smooth + 0.1
    d = [divider_dimension(rough(dim=dim))["D"] for dim in (1.15, 1.25, 1.35)]
    assert smooth + 0.03 < d[0] < d[1] < d[2]  # tracks the synthesis roughness exponent


def test_box_counting():
    c = box_counting_dimension(circle(150, 3000))
    k = box_counting_dimension(koch())
    assert c["valid"] and k["valid"] and c["D"] < 1.15 and k["D"] > c["D"] + 0.15
    assert box_counting_dimension(circle(10, 200)) == {"D": 1.0, "r2": pytest.approx(np.nan, nan_ok=True),
                                                        "valid": False}


def test_fractal_invariance():
    xy = rough(r0=120.0)
    d0, b0 = divider_dimension(xy)["D"], box_counting_dimension(xy)["D"]
    assert divider_dimension(similar(xy, shift=(13.3, -4), roll=611))["D"] == pytest.approx(d0, abs=1e-9)
    assert divider_dimension(similar(xy, scale=2.0))["D"] == pytest.approx(d0, abs=1e-9)  # L >= 200: steps scale
    assert divider_dimension(similar(xy, angle=0.9))["D"] == pytest.approx(d0, abs=2e-3)
    assert box_counting_dimension(similar(xy, shift=(13.3, -4), roll=611))["D"] == pytest.approx(b0, abs=1e-9)
    assert box_counting_dimension(similar(xy, angle=0.9))["D"] == pytest.approx(b0, abs=0.05)


def test_multiscale_slopes():
    xy = rough(r0=60.0)
    s = multiscale_perimeter_slopes(xy)
    assert s.shape == (4,)
    assert np.allclose(multiscale_perimeter_slopes(similar(xy, angle=2.0, scale=0.3, shift=(5, 5), roll=99)), s)
    assert s[0] < multiscale_perimeter_slopes(circle(60, 3000))[0]  # roughness is lost faster at fine scales


def test_fractal_descriptors_keys():
    keys = {"fractal_divider_D", "fractal_divider_r2", "fractal_box_D", "fractal_box_r2", "fractal_valid",
            "fractal_ms_0", "fractal_ms_1", "fractal_ms_2", "fractal_ms_3"}
    small = fractal_descriptors(circle(6, 30))
    assert set(small) == keys and small["fractal_valid"] == 0.0 and small["fractal_divider_r2"] == 0.0
    assert all(np.isfinite(v) for v in small.values())
    assert fractal_descriptors(koch())["fractal_valid"] == 1.0


# ----------------------------------------------------------------------------- fourier


def test_efd_circle():
    e = elliptic_fourier_descriptors(circle(3, 512, c=(10, 10)), order=6)
    assert np.allclose(e[0], [1, 0, 0, 1], atol=1e-9) and np.allclose(e[1:], 0, atol=1e-9)


def test_efd_raw_matches_direct_fourier_integral():
    xy = blob()
    raw = elliptic_fourier_descriptors(xy, order=8, normalize=False)
    closed = np.vstack([xy, xy[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    t = np.linspace(0, s[-1], 200000, endpoint=False)
    x, y = np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])
    for n in range(1, 9):
        cs, sn = np.cos(2 * np.pi * n * t / s[-1]), np.sin(2 * np.pi * n * t / s[-1])
        direct = 2 * np.array([np.mean(x * cs), np.mean(x * sn), np.mean(y * cs), np.mean(y * sn)])
        assert np.allclose(raw[n - 1], direct, atol=1e-4 * 60)


def test_efd_ellipse():
    e = elliptic_fourier_descriptors(ellipse(5, 2), order=6)
    assert np.allclose(e[0, :3], [1, 0, 0], atol=1e-9) and 0.3 < e[0, 3] < 0.6  # d1 ~ minor/major, > 0
    assert np.allclose(e[1::2], 0, atol=1e-9)  # centrally symmetric: no even harmonics
    moved = similar(ellipse(5, 2), angle=0.8, scale=7.0, shift=(100, 50), roll=77)
    assert np.allclose(elliptic_fourier_descriptors(moved, order=6), e, atol=1e-8)
    raw = elliptic_fourier_descriptors(ellipse(5, 2), order=6, normalize=False)
    assert np.allclose(raw[1::2], 0, atol=1e-9) and raw[0, 0] == pytest.approx(5, rel=0.15)


def test_efd_invariance_asymmetric():
    xy = blob()
    ref = efd_feature_vector(xy, order=10)
    assert ref.shape == (37,) and np.abs(elliptic_fourier_descriptors(xy)[1::2]).max() > 0.01
    for t in (similar(xy, angle=2.5, scale=0.2, shift=(-9, 3), roll=123), similar(xy, angle=-1.0, roll=200),
              xy[::-1]):
        assert np.allclose(efd_feature_vector(t, order=10), ref, atol=1e-8)


# ----------------------------------------------------------------------------- shape


def test_shape_circle():
    s = shape_descriptors(circle(50, 1000, c=(200, 100)), um_per_px=0.5)
    assert set(s) == {f"shape_{k}" for k in (
        "area_um2", "log_area_um2", "perimeter_um", "ecd_um", "log_ecd_um", "major_um", "minor_um", "aspect_ratio",
        "roundness", "solidity", "convexity", "eccentricity", "feret_max_um", "feret_min_um", "elongation", "curl")}
    assert s["shape_area_um2"] == pytest.approx(np.pi * 625, rel=1e-3)
    assert s["shape_ecd_um"] == pytest.approx(50, rel=1e-3)
    assert s["shape_log_ecd_um"] == pytest.approx(np.log(50), abs=1e-3)
    assert s["shape_roundness"] > 0.999 and s["shape_aspect_ratio"] == pytest.approx(1, abs=1e-3)
    assert s["shape_solidity"] == pytest.approx(1, abs=1e-4)
    assert s["shape_convexity"] == pytest.approx(1, abs=1e-4)
    assert s["shape_feret_max_um"] == pytest.approx(50, rel=1e-3)
    assert s["shape_feret_min_um"] == pytest.approx(50, rel=1e-3)
    assert s["shape_curl"] == 1.0 and s["shape_eccentricity"] < 0.05


def test_shape_rectangle_and_ribbon():
    s = shape_descriptors(np.array([[0, 0], [100, 0], [100, 20], [0, 20]], float), um_per_px=2.0)
    assert s["shape_aspect_ratio"] == pytest.approx(5.0)
    assert s["shape_feret_min_um"] == pytest.approx(40)
    assert s["shape_feret_max_um"] == pytest.approx(2 * np.hypot(100, 20))
    assert s["shape_elongation"] == pytest.approx(np.hypot(100, 20) / 20)
    assert s["shape_roundness"] == pytest.approx(4 * np.pi * 2000 / 240 ** 2) and s["shape_curl"] == 1.0
    assert s["shape_major_um"] == pytest.approx(2 * 4 * np.sqrt(100 ** 2 / 12))
    t = np.linspace(0, 1.5 * np.pi, 200)  # 3/4 annulus, width 4: a curled ribbon
    ribbon = np.vstack([np.c_[42 * np.cos(t), 42 * np.sin(t)], np.c_[38 * np.cos(t), 38 * np.sin(t)][::-1]])
    rs = shape_descriptors(ribbon, 1.0)
    assert rs["shape_curl"] < 0.6 and rs["shape_solidity"] < 0.3


def test_shape_invariance():
    xy = blob()
    ref = shape_descriptors(xy, 0.4)
    moved = shape_descriptors(similar(xy, angle=1.3, scale=2.5, shift=(300, 20), roll=91), 0.4 / 2.5)
    assert ref.keys() == moved.keys()
    for k, v in ref.items():
        assert moved[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k
    ratio = shape_descriptors(xy, 0.8)
    for k in ("shape_aspect_ratio", "shape_roundness", "shape_solidity", "shape_convexity", "shape_eccentricity",
              "shape_elongation", "shape_curl"):
        assert ratio[k] == pytest.approx(ref[k])
    assert ratio["shape_area_um2"] == pytest.approx(4 * ref["shape_area_um2"])


def test_shape_degenerate_raises():
    with pytest.raises(ValueError):
        shape_descriptors(np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float), 1.0)


# ----------------------------------------------------------------------------- texture


def disc_patch(n=48, r=20):
    yy, xx = np.mgrid[0:n, 0:n]
    return (xx - n / 2) ** 2 + (yy - n / 2) ** 2 < r * r, yy, xx


TEX_KEYS = ({f"tex_{k}" for k in ("mean", "std", "skew", "p10", "p90", "edge_density", "coherence", "dark_fraction",
                                  "rim_contrast")}
            | {f"tex_glcm_{k}" for k in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity")}
            | {f"tex_lbp_{i}" for i in range(10)})


def test_texture_keys_and_coherence():
    mask, yy, xx = disc_patch()
    stripes = 0.5 + 0.3 * np.sin(2 * np.pi * (xx + 0.5 * yy) / 6)
    noise = 0.5 + 0.1 * np.random.default_rng(0).standard_normal(mask.shape)
    ts = texture_descriptors(np.repeat(stripes[..., None], 3, axis=2), mask)
    tn = texture_descriptors(np.repeat(noise[..., None], 3, axis=2), mask)
    assert set(ts) == TEX_KEYS and all(np.isfinite(v) for v in ts.values())
    assert sum(ts[f"tex_lbp_{i}"] for i in range(10)) == pytest.approx(1.0)
    assert ts["tex_coherence"] > 0.7 > 0.3 > tn["tex_coherence"]
    flat = texture_descriptors(np.full(mask.shape + (3,), 0.5), mask)
    assert flat["tex_std"] == 0 and flat["tex_skew"] == 0 and flat["tex_edge_density"] == 0
    assert all(np.isfinite(v) for v in flat.values())


def test_texture_masked_glcm_ignores_outside():
    mask, _, _ = disc_patch()
    rng = np.random.default_rng(1)
    a = (rng.uniform(0.2, 0.6, mask.shape + (1,)).repeat(3, axis=2) * 255).astype(np.uint8)
    b = a.copy()
    b[~mask] = rng.integers(0, 256, (int((~mask).sum()), 3))
    ta, tb = texture_descriptors(a, mask), texture_descriptors(b, mask)
    for k in ("tex_mean", "tex_std", "tex_p10", "tex_p90", "tex_dark_fraction", *[f"tex_glcm_{p}" for p in
              ("contrast", "homogeneity", "energy", "correlation", "dissimilarity")]):
        assert ta[k] == pytest.approx(tb[k]), k


def test_texture_rim_dark_edges():
    mask, yy, xx = disc_patch()
    img = np.full(mask.shape, 0.6)
    img[mask & ((xx - 24) ** 2 + (yy - 24) ** 2 > 17 ** 2)] = 0.3  # dark rim
    t = texture_descriptors(np.repeat(img[..., None], 3, axis=2), mask)
    assert t["tex_rim_contrast"] > 0.1
    spots = np.full(mask.shape, 0.6)
    spots[20:24, 20:24] = 0.2
    t = texture_descriptors(np.repeat(spots[..., None], 3, axis=2), mask)
    assert t["tex_dark_fraction"] == pytest.approx(16 / mask.sum())
    checker = 0.3 + 0.4 * (((xx // 4) + (yy // 4)) % 2)
    assert texture_descriptors(np.repeat(checker[..., None], 3, axis=2), mask)["tex_edge_density"] > 0.1


def test_tiny_and_empty_masks_are_finite():
    patch = (np.random.default_rng(2).uniform(0, 1, (5, 6, 3)) * 255).astype(np.uint8)
    for m in (np.zeros((5, 6), bool), np.pad(np.ones((1, 2), bool), ((2, 2), (2, 2)))):
        t = {**texture_descriptors(patch, m), **color_descriptors(patch, m)}
        assert all(np.isfinite(v) for v in t.values())


def test_color_descriptors():
    mask, _, _ = disc_patch()
    red = np.zeros(mask.shape + (3,), np.uint8)
    red[...] = (170, 120, 60)
    grey = np.full(mask.shape + (3,), 120, np.uint8)
    cr, cg = color_descriptors(red, mask), color_descriptors(grey, mask)
    assert set(cr) == {"color_r", "color_g", "color_b", "color_sat"}
    assert cr["color_r"] == pytest.approx(170 / 255) and cr["color_r"] > cr["color_b"]
    assert cr["color_sat"] > 0.5 > cg["color_sat"] == pytest.approx(0)


def test_boundary_band_profile():
    gray = np.tile(np.arange(100, dtype=np.float32) / 100, (50, 1))  # grey = x / 100
    xy = np.array([[20.0, 10.0], [40.5, 30.0]])
    nrm = np.array([[1.0, 0.0], [-1.0, 0.0]])
    prof = boundary_band_profile(gray, xy, nrm, [0, 1.5, 3])
    assert prof.shape == (2, 3)
    assert np.allclose(prof, [[0.2, 0.215, 0.23], [0.405, 0.39, 0.375]], atol=1e-6)
    per_point = boundary_band_profile(gray, xy, nrm, [[1, 2], [0.5, 10]])  # (N, D) depths, e.g. clipped per node
    assert np.allclose(per_point, [[0.21, 0.22], [0.40, 0.305]], atol=1e-6)


# ----------------------------------------------------------------------------- descriptors


def test_view_descriptors_keys():
    d = view_descriptors(blob(), 0.5, efd_order=10)
    assert all(k.startswith(("shape_", "efd_", "fractal_")) for k in d)
    assert sum(k.startswith("efd_") for k in d) == 37 and "efd_36" in d
    assert all(isinstance(v, float) and np.isfinite(v) for v in d.values())


def test_particle_descriptors_single_and_multi():
    mask, _, _ = disc_patch()
    patch = np.full(mask.shape + (3,), 120, np.uint8)
    single = particle_descriptors([(circle(20, 200, c=(24, 24)), 0.5)], patch, mask)
    assert single["mv_n_views"] == 1 and single["mv_aspect_std"] == 0 and single["mv_area_ratio_min"] == 1
    assert single["mv_aspect_min"] == single["mv_aspect_max"] == single["shape_aspect_ratio"]
    assert single["mv_roundness_min"] == single["shape_roundness"]
    views = [(circle(20, 200), 0.5), (ellipse(40, 10, 200), 0.25)]  # areas 314 vs 78.5 um^2
    multi = particle_descriptors(views, patch, mask)
    assert multi["mv_n_views"] == 2 and multi["mv_area_ratio_min"] == pytest.approx(0.25, rel=1e-2)
    assert multi["mv_aspect_max"] == pytest.approx(4, rel=1e-2) and multi["mv_aspect_std"] > 1
    assert multi["shape_area_um2"] == pytest.approx(single["shape_area_um2"])
    assert all(k.startswith(DESCRIPTOR_PREFIXES) for k in multi)
    assert all(np.isfinite(v) for v in multi.values())


def test_no_nan_on_random_and_tiny_contours():
    rng = np.random.default_rng(3)
    for i in range(25):
        n = int(rng.integers(8, 300))
        t = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = rng.uniform(2, 120) * (1 + 0.3 * rng.uniform(0, 1) * np.sin(rng.integers(2, 7) * t + rng.uniform(0, 6)))
        d = view_descriptors(np.c_[r * np.cos(t), r * np.sin(t)], float(rng.uniform(0.1, 1.0)))
        assert all(np.isfinite(v) for v in d.values()), i


@pytest.mark.slow
def test_geometry_descriptor_speed():
    from skimage.draw import ellipse as draw_ellipse
    from skimage.measure import find_contours
    m = np.zeros((200, 200), bool)
    m[draw_ellipse(100, 100, 40, 80, shape=m.shape)] = True
    xy = find_contours(m.astype(np.uint8), 0.5)[0][:, ::-1]  # ~480 points, L = 160 px
    view_descriptors(xy, 0.5)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        view_descriptors(xy, 0.5)
        times.append(time.perf_counter() - t0)
    assert np.median(times) < 0.010  # target ~5 ms; loose bound for shared CI machines
