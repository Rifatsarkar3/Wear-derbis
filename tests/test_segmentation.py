import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi
from skimage.draw import disk, ellipse
from skimage.io import imread, imsave

from wearpg.features.contour import polygon_mask, signed_area
from wearpg.schema import WearDataset
from wearpg.segmentation import (export_rating_package, extract_crop, load_image, native_patch, segment_dataset,
                                 segment_image, segmentation_quality, to_gray)


def dmask(center_rc, r, size=200):
    m = np.zeros((size, size), bool)
    m[disk(center_rc, r, shape=m.shape)] = True
    return m


def toy(shapes, size=200, blur=1.0, noise=0.01, seed=0):
    """RGB uint8 image: grey 0.5 background with an illumination gradient; shapes = [(mask, grey value)]."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = 0.5 + 0.08 * (xx / size - 0.5) + 0.04 * (yy / size - 0.5)
    for m, val in shapes:
        img[m] = val
    img = ndi.gaussian_filter(img, blur) + np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(np.repeat(img[..., None], 3, axis=2) * 255, 0, 255).astype(np.uint8)


def gt_contour(m):
    from skimage.measure import find_contours
    return find_contours(np.pad(m, 1).astype(np.uint8), 0.5)[0][:, ::-1] - 1


def test_load_image_and_to_gray(tmp_path):
    g = (np.arange(20 * 30).reshape(20, 30) % 256).astype(np.uint8)
    imsave(tmp_path / "g.png", g, check_contrast=False)
    rgba = np.dstack([g, g, g, np.full_like(g, 255)])
    imsave(tmp_path / "a.png", rgba, check_contrast=False)
    imsave(tmp_path / "w.png", (g.astype(np.uint16) * 257), check_contrast=False)
    for name in ("g.png", "a.png", "w.png"):
        img = load_image(tmp_path / name)
        assert img.shape == (20, 30, 3) and img.dtype == np.uint8 and np.array_equal(img[..., 0], g), name
    gray = to_gray(load_image(tmp_path / "g.png"))
    assert gray.dtype == np.float32 and np.allclose(gray, g / 255, atol=1e-3)
    assert to_gray(g).dtype == np.float32


def test_polarity_free_dark_and_bright_discs():
    dark, bright = dmask((60, 60), 20), dmask((130, 140), 25)
    img = toy([(dark, 0.2), (bright, 0.8)])
    segs = segment_image(img)
    assert len(segs) == 2
    q = segmentation_quality([s["contour"] for s in segs], [gt_contour(dark), gt_contour(bright)], img.shape)
    assert q["f1"] == 1.0 and q["mean_iou"] > 0.9
    only_dark = segment_image(img, {"polarity": "dark"})
    only_bright = segment_image(img, {"polarity": "bright"})
    assert len(only_dark) == 1 and only_dark[0]["bbox"][0] < 100
    assert len(only_bright) == 1 and only_bright[0]["bbox"][0] > 100
    with pytest.raises(ValueError):
        segment_image(img, {"polarity": "sideways"})


def test_segment_contract_conventions():
    m = dmask((80, 100), 18)
    s = segment_image(toy([(m, 0.25)]))[0]
    xy = s["contour"]
    assert xy.ndim == 2 and xy.shape[1] == 2 and signed_area(xy) > 0 and not np.allclose(xy[0], xy[-1])
    x0, y0, x1, y1 = s["bbox"]
    assert all(isinstance(v, int) for v in s["bbox"])
    assert x0 - 0.5 <= xy[:, 0].min() and xy[:, 0].max() <= x1 - 0.5 and y0 - 0.5 <= xy[:, 1].min()
    assert s["area_px"] == pytest.approx(m.sum(), rel=0.05)
    assert polygon_mask(xy, m.shape).sum() == pytest.approx(s["area_px"], rel=0.01)


@pytest.mark.parametrize("blur", [0.5, 1.0, 1.6])
def test_watershed_splits_touching_discs(blur):
    a, b = dmask((100, 70), 15), dmask((100, 100), 15)  # centres 2r apart: in contact
    segs = segment_image(toy([(a | b, 0.25)], blur=blur, noise=0.02))
    assert len(segs) == 2
    assert all(abs(s["area_px"] - a.sum()) / a.sum() < 0.15 for s in segs)
    q = segmentation_quality([s["contour"] for s in segs], [gt_contour(a), gt_contour(b)], (200, 200))
    assert q["f1"] == 1.0


def test_watershed_keeps_agglomerates_and_elongated_particles_whole():
    lobes = dmask((100, 80), 10) | dmask((100, 90), 10) | dmask((108, 85), 10)
    thin = np.zeros((200, 200), bool)
    thin[ellipse(100, 100, 6, 50, shape=thin.shape, rotation=0.3)] = True
    for m in (lobes, thin):
        for blur in (0.5, 1.6):
            assert len(segment_image(toy([(m, 0.25)], blur=blur, noise=0.02))) == 1


def test_large_coverage_with_illumination_falloff():
    """Particles covering ~1/3 of the frame must not bias the background fit or inflate the noise floor."""
    big, smalls = dmask((110, 110), 62), [dmask((25, 30), 9), dmask((30, 170), 8), dmask((175, 25), 10)]
    img = toy([(big, 0.2)] + [(m, 0.25) for m in smalls], noise=0.02).astype(float)
    yy, xx = np.mgrid[0:200, 0:200] / 100 - 1
    img = np.clip(img * (1 - 0.1 * (xx ** 2 + yy ** 2))[..., None], 0, 255).astype(np.uint8)  # radial falloff
    segs = segment_image(img)
    q = segmentation_quality([s["contour"] for s in segs], [gt_contour(m) for m in [big, *smalls]], (200, 200))
    assert q["f1"] == 1.0 and q["mean_iou"] > 0.85


def test_min_area_border_and_empty():
    small, edge, big = dmask((50, 50), 2), dmask((100, 2), 12), dmask((140, 140), 12)
    img = toy([(small, 0.2), (edge, 0.2), (big, 0.2)])
    assert len(segment_image(img)) == 1
    assert len(segment_image(img, {"exclude_border": False})) == 2
    assert segment_image(toy([], noise=0.03)) == []


def test_segmentation_quality_cases():
    a, b = gt_contour(dmask((50, 50), 10)), gt_contour(dmask((120, 120), 15))
    q = segmentation_quality([b, a], [a, b], (200, 200))
    assert (q["precision"], q["recall"], q["f1"], q["n_matched"]) == (1.0, 1.0, 1.0, 2)
    assert q["mean_iou"] == pytest.approx(1.0)
    q = segmentation_quality([a], [a, b], (200, 200))
    assert q["precision"] == 1.0 and q["recall"] == 0.5 and q["f1"] == pytest.approx(2 / 3)
    shifted = a + [12, 0]  # IoU below 0.5 -> no match
    q = segmentation_quality([shifted], [a], (200, 200))
    assert q["n_matched"] == 0 and q["f1"] == 0.0 and np.isnan(q["mean_iou"])
    assert segmentation_quality([], [], (10, 10))["f1"] == 1.0
    assert segmentation_quality([], [a], (200, 200))["recall"] == 0.0


def test_segment_dataset_new_ids(tiny_dataset_on_disk):
    ds = tiny_dataset_on_disk
    parts, views, q = segment_dataset(ds)
    assert len(parts) == len(ds.particles) and (views.source == "segmented").all()
    assert parts.particle_id.str.contains("_seg").all() and (views.view_idx == 0).all()
    assert q["n_images"] == len(ds.images) and q["recall"] >= 0.95 and q["mean_iou"] > 0.85
    assert (parts.n_views == 1).all() and not parts.has_eds.any()


def test_segment_dataset_match_gt_keeps_ids_and_validates(tiny_dataset_on_disk):
    ds = tiny_dataset_on_disk
    parts, views, q = segment_dataset(ds, match_gt=True)
    assert set(parts.particle_id) == set(ds.particles.particle_id)
    assert q["precision"] == 1.0 and q["recall"] == 1.0 and q["n_images_with_gt"] == len(ds.images)
    gt = ds.particles.set_index("particle_id")
    assert (parts.set_index("particle_id").has_eds == gt.has_eds.loc[parts.particle_id]).all()
    seg = WearDataset(specimens=ds.specimens, runs=ds.runs, samples=ds.samples, images=ds.images,
                      particles=parts, views=views, labels=ds.labels, labels_adjudicated=ds.labels_adjudicated,
                      meta=ds.meta)
    assert seg.validate() == []


def test_segment_dataset_multiview_frames(tiny_dataset_on_disk):
    ds = tiny_dataset_on_disk
    ds.images.loc[0, "modality"] = "multiview_frame"  # frame segments are only kept when matched
    parts, views, q = segment_dataset(ds, modalities=("ferrograph_optical", "multiview_frame"))
    assert q["n_unassigned_frame_segments"] == 4
    assert not views.image_id.eq(ds.images.image_id[0]).any()
    parts, views, _ = segment_dataset(ds, modalities=("ferrograph_optical", "multiview_frame"), match_gt=True)
    assert views.image_id.eq(ds.images.image_id[0]).sum() == 4  # matched GT views keep their ids


def test_extract_crop():
    img = toy([(dmask((100, 100), 20), 0.2)])
    xy = gt_contour(dmask((100, 100), 20))
    crop = extract_crop(img, xy, size=64, pad=1.25)
    assert crop.shape == (64, 64, 4) and crop.dtype == np.uint8
    assert set(np.unique(crop[..., 3])) <= {0, 255}
    side = int(np.ceil(1.25 * 40))
    assert (crop[..., 3] > 0).mean() == pytest.approx(np.pi * 20 ** 2 / side ** 2, rel=0.05)
    assert crop[32, 32, 0] < 100 < crop[0, 0, 0]  # particle centred, background in the corner
    img[0, :] = img[:, 0] = (255, 0, 0)  # red image border
    corner = gt_contour(dmask((0, 0), 10))  # quarter disc in the image corner
    c2 = extract_crop(img, corner, size=32)
    assert c2[0, 0, 3] == 0 and (c2[..., 3] > 0).any()  # out-of-image mask is 0
    assert tuple(c2[0, 0, :3]) == (255, 0, 0)  # out-of-image RGB replicates the edge


def test_native_patch():
    img = toy([(dmask((80, 60), 10), 0.2)])
    xy = gt_contour(dmask((80, 60), 10))
    patch, mask = native_patch(img, xy, margin=2)
    assert patch.shape[:2] == mask.shape and patch.shape[2] == 3
    assert patch.shape[0] == int(np.ceil(xy[:, 1].max())) - int(np.floor(xy[:, 1].min())) + 5
    assert mask.sum() == dmask((80, 60), 10).sum()
    edge = np.array([[0.5, 0.5], [6, 0.5], [6, 6], [0.5, 6]])
    p2, m2 = native_patch(img, edge, margin=3)
    assert p2.shape[:2] == m2.shape and p2.shape[0] == 10  # clipped at the image border


def test_export_rating_package(tiny_dataset_on_disk, tmp_path):
    ds = tiny_dataset_on_disk
    out = export_rating_package(ds, tmp_path / "pkg", crop_size=48, raters=("A", "B"))
    crops = sorted((out / "crops").glob("*.png"))
    assert len(crops) == len(ds.particles) and imread(crops[0]).shape == (48, 48, 3)
    ta, tb = (pd.read_csv(out / f"labels_template_{r}.csv", keep_default_na=False) for r in ("A", "B"))
    for t, r in ((ta, "A"), (tb, "B")):
        assert set(t.particle_id) == set(ds.particles.particle_id) and (t.rater_id == r).all()
        assert {"label", "confidence", "evidence", "crop_file", "crop_width_um"} <= set(t.columns)
        assert (t.label == "").all() and (t.crop_width_um > 0).all()
    assert list(ta.particle_id) != list(tb.particle_id)  # independently shuffled order
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert readme.startswith("SYNTHETIC DATA - PIPELINE VERIFICATION ONLY")
    assert all(c in readme for c in ("rubbing", "cutting", "spherical", "fatigue_chunk", "laminar",
                                     "severe_sliding", "oxide"))


@pytest.mark.slow
def test_segment_synthetic_small(synthetic_small):
    parts, views, q = segment_dataset(synthetic_small, match_gt=True)
    assert q["n_images_with_gt"] > 0 and q["recall"] > 0.85 and q["precision"] > 0.9 and q["mean_iou"] > 0.85
