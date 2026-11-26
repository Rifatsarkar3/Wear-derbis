"""Tests for wearpg.open_data (toy masks and a fake Zenodo archive layout; no network, no real data)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from wearpg.open_data import (contours_mask, load_zenodo_ferro, mask_components, pixel_scores, threshold_baseline,
                              to_wear_dataset)
from wearpg.features.contour import signed_area


def _disc(shape, cy, cx, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _toy_mask(shape=(64, 64)):
    m = _disc(shape, 20, 20, 8)  # interior disc
    m |= _disc(shape, 44, 44, 9) & ~_disc(shape, 44, 44, 3)  # ring: hole ignored by the outer contour
    m[0:6, 50:60] = True  # touches the top border (60 px)
    m[55:59, 5:9] = True  # 16 px: below min_area_px
    return m


def test_mask_components_contours_and_flags():
    m = _toy_mask()
    comps = mask_components(m, min_area_px=30)
    assert len(comps) == 3
    assert [c["touches_border"] for c in comps] == [True, False, False]  # row-major label order
    for c in comps:
        assert signed_area(c["contour"]) > 0 and len(c["contour"]) >= 8
        x0, y0, x1, y1 = c["bbox"]
        assert x0 - 0.5 == c["contour"][:, 0].min() and c["contour"][:, 0].max() == x1 - 0.5  # 0.5 iso-line
    assert comps[1]["area_px"] == float(_disc(m.shape, 20, 20, 8).sum())
    # rasterising the outer contours gives back the hole-filled components exactly
    from scipy import ndimage as ndi
    kept = m.copy()
    kept[55:59, 5:9] = False
    assert np.array_equal(contours_mask([c["contour"] for c in comps], m.shape), ndi.binary_fill_holes(kept))


def test_pixel_scores_known_values():
    gt = np.zeros((10, 8), bool)
    gt[:, :6] = True
    pred = np.zeros_like(gt)
    pred[:, :4] = True
    s = pixel_scores(pred, gt)
    assert np.isclose(s["iou"], 4 / 6) and np.isclose(s["dice"], 2 * 40 / 100)
    assert np.isclose(s["area_rel_err"], -1 / 3)
    e = pixel_scores(np.zeros_like(gt), np.zeros_like(gt))
    assert e["iou"] == 1.0 and e["dice"] == 1.0 and np.isnan(e["area_rel_err"])


def test_threshold_baselines_find_dark_and_bright_discs():
    shape = (96, 96)
    ramp = np.linspace(0.45, 0.65, shape[1])[None, :].repeat(shape[0], 0)
    disc = _disc(shape, 48, 40, 12)
    for val in (0.1, 0.95):  # dark and bright particle: Otsu takes the minority side
        img = (np.dstack([np.where(disc, val, ramp)] * 3) * 255).astype(np.uint8)
        assert pixel_scores(threshold_baseline(img, "otsu"), disc)["iou"] > 0.9
    dark = np.where(disc, 0.1, ramp)
    img = (np.dstack([dark] * 3) * 255).astype(np.uint8)
    assert pixel_scores(threshold_baseline(img, "local", block_size=51, offset=0.05), disc)["iou"] > 0.8


def _fake_archive(root):
    """dataset/{trainO,testO}/{images,labels}: 2 train pairs, 1 test pair and 1 orphan test mask."""
    rng = np.random.default_rng(0)
    for split, stems in (("trainO", ["1", "10"]), ("testO", ["2"])):
        for d in ("images", "labels"):
            (root / "dataset" / split / d).mkdir(parents=True, exist_ok=True)
        for s in stems:
            m = _toy_mask()
            img = np.where(m[..., None], 40, 200).astype(np.uint8).repeat(3, 2) + rng.integers(0, 5, (64, 64, 3),
                                                                                              dtype=np.uint8)
            Image.fromarray(img).save(root / "dataset" / split / "images" / f"{s}.png")
            Image.fromarray(m.astype(np.uint8) * 255).save(root / "dataset" / split / "labels" / f"{s}.png")
    Image.fromarray(np.zeros((64, 64), np.uint8)).save(root / "dataset" / "testO" / "labels" / "99.png")


def test_loader_and_converter_give_a_valid_dataset(tmp_path):
    _fake_archive(tmp_path)
    idx = load_zenodo_ferro(tmp_path)
    assert idx.image_id.tolist() == ["trainO_1", "trainO_10", "testO_2"]  # orphan mask 99 skipped
    ds = to_wear_dataset(idx, tmp_path)
    assert ds.validate() == []
    assert len(ds.images) == 3 and len(ds.particles) == 9 and len(ds.labels) == 0
    assert ds.meta["is_synthetic"] is False and ds.meta["units"] == "px, uncalibrated"
    assert ds.meta["source"]["license"] == "CC-BY-4.0" and "zenodo.22997504" in ds.meta["source"]["citation"]
    assert set(ds.views.source) == {"manual"} and (ds.views.um_per_px == 1.0).all()
    assert ds.runs.wear_volume_mm3.isna().all() and ds.particles.touches_border.sum() == 3
    assert ds.image_path("trainO_10").exists()


# ----------------------------------------------------------------------------- script logic (scripts/open_ferro_*.py)


def _script(name: str):
    """Import scripts/<name>.py (env changes made at import, e.g. BLAS thread pinning, are rolled back)."""
    import importlib.util
    import os
    from pathlib import Path
    from unittest import mock

    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ):
        spec.loader.exec_module(mod)
    return mod


def test_robust_transforms_keep_normalised_features():
    ro = _script("open_ferro_robust")
    yy, xx = np.mgrid[:120, :120]
    ell = ((xx - 60) / 30.0) ** 2 + ((yy - 60) / 15.0) ** 2 <= 1
    img = np.dstack([np.where(ell, 0.2, 0.7)] * 3)
    ref = ro.features(ro._largest(ell), img, 1.0)
    assert ref["chord_median_hop"] > ro.GCFG.n_nodes / 8  # width chords cross the ellipse
    rng = np.random.default_rng(0)
    for t in ("rot45", "rot90", "scale0.5", "blur1"):
        im_t, (m_t,), s = ro.transform(img, [ell], t, 33.0, rng)
        f = ro.features(ro._largest(m_t), im_t, s)
        assert abs(f["shape_area_um2"] / ref["shape_area_um2"] - 1) < 0.05, t  # area / s^2
        assert abs(f["shape_ecd_um"] / ref["shape_ecd_um"] - 1) < 0.03, t  # length / s
        assert abs(f["shape_aspect_ratio"] - ref["shape_aspect_ratio"]) < 0.1, t
    im_n, (m_n,), _ = ro.transform(img, [ell], "noise0.05", 0.0, rng)
    assert m_n is ell and not np.array_equal(im_n, img)  # noise perturbs the image only


def test_robust_stability_table():
    import pandas as pd

    ro = _script("open_ferro_robust")
    base = {"shape_roundness": [0.2, 0.4, 0.6, 0.8], "fractal_divider_D": [1.0, 1.05, 1.1, 1.2],
            "fractal_valid": [0.0, 1.0, 1.0, 1.0]}
    rows = []
    for t, lost in (("identity", [False] * 4), ("rot45", [False] * 4), ("scale0.25", [False, False, False, True])):
        for i in range(4):
            r = {"mode": "mask", "transform": t, "particle": f"p{i}", "lost": lost[i]}
            if not lost[i]:
                r.update({k: v[i] * (1.01 if t != "identity" else 1.0) for k, v in base.items()})
                r["fractal_valid"] = base["fractal_valid"][i]
            rows.append(r)
    st = ro.stability(pd.DataFrame(rows)).set_index(["transform", "feature"])
    assert st.loc[("rot45", "shape_roundness"), "spearman"] == 1.0
    assert st.loc[("rot45", "fractal_divider_D"), "n"] == 3  # floored D only over particles valid in both
    assert st.loc[("rot45", "shape_roundness"), "median_rel_change"] < 0.011
    assert st.loc[("scale0.25", "shape_roundness"), "n_lost"] == 1
    assert ("scale0.25", "fractal_divider_D") not in st.index  # < 3 particles valid in both


def test_eval_summary_metrics_and_methods():
    import pandas as pd

    ev = _script("open_ferro_eval")
    df = pd.DataFrame({"iou": [0.5, 0.7, 0.6], "dice": [0.6, 0.8, 0.7], "precision": [1.0, 0.5, 0.5],
                       "recall": [0.5, 0.5, 1.0], "f1": [0.6, 0.5, 0.6], "area_rel_err": [0.1, -0.2, 0.0],
                       "count_rel_err": [0.0, 0.5, -0.5], "area_pred": [110, 96, 300], "area_gt": [100, 120, 300],
                       "n_pred": [1, 3, 2], "n_gt": [1, 2, 4]})
    s = ev.summary_metrics(df)
    assert np.isclose(s["iou"], 0.6) and np.isclose(s["abs_area_rel_err"], 0.1)
    assert np.isclose(s["abs_count_rel_err"], 1 / 3) and np.isclose(s["spearman_area"], 0.5)
    tuned = {"best": {"ours": {"polarity": "both", "smooth_sigma": 2.0, "exclude_border": False, "train_iou": 0.5},
                      "local": {"block_size": 201.0, "offset": 0.05, "train_iou": 0.4}}}
    ms = ev.methods(tuned)
    assert ms["ours_tuned"][1]["exclude_border"] is False and "train_iou" not in ms["ours_tuned"][1]
    assert ms["local_tuned"] == ("local", {"block_size": 201, "offset": 0.05})
    assert ms["ours_default"][1] == ev.SEG_DEFAULTS


def test_eval_leak_check_finds_rotated_copy(tmp_path):
    from scipy import ndimage as ndi

    ev = _script("open_ferro_eval")
    rng = np.random.default_rng(1)
    pics = {}
    for split, stems in (("trainO", ["1", "10"]), ("testO", ["2", "3"])):
        for d in ("images", "labels"):
            (tmp_path / "dataset" / split / d).mkdir(parents=True, exist_ok=True)
        for s in stems:
            g = ndi.gaussian_filter(rng.random((64, 64)), 3)
            pics[(split, s)] = (g, g > np.median(g))
    pics[("testO", "2")] = tuple(np.rot90(a) for a in pics[("trainO", "10")])  # rotated copy of a training image
    for (split, s), (g, m) in pics.items():
        img = (np.dstack([g] * 3) / g.max() * 255).astype(np.uint8)
        Image.fromarray(img).save(tmp_path / "dataset" / split / "images" / f"{s}.png")
        Image.fromarray(m.astype(np.uint8) * 255).save(tmp_path / "dataset" / split / "labels" / f"{s}.png")
    ev.OUT = tmp_path
    ev.stage_leak(load_zenodo_ferro(tmp_path))
    import pandas as pd

    df = pd.read_csv(tmp_path / "seg_leak_check.csv").set_index("test_image")
    assert df.loc["testO_2", "near_dup"] and df.loc["testO_2", "best_train_image"] == "trainO_10"
    assert df.loc["testO_2", "mask_iou"] > 0.99 and not df.loc["testO_2", "exact_dup"]
    assert not df.loc["testO_3", "near_dup"]
