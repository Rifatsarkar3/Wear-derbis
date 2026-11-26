"""[ODI] Open real-image data (the author has no laboratory): loader and WearDataset converter for the Zenodo
"Ferrographic Image Dataset" (ferrograms + foreground masks; no labels, no wear data, no pixel calibration), plus the
mask helpers the open-data scripts share (components -> contours, pixel scores, simple threshold baselines).

Unit of inference = source image. Each image becomes its own specimen / run / sample with PLACEHOLDER values for the
required physical fields (listed in meta["placeholders"]; never use them as data) and um_per_px = 1.0
(meta["units"] = "px, uncalibrated").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.filters import threshold_local, threshold_otsu
from skimage.measure import find_contours

from wearpg.features.contour import clean_closed, ensure_ccw, signed_area
from wearpg.schema import WearDataset
from wearpg.segmentation import load_image, to_gray

ZENODO_FERRO = {
    "name": "zenodo_ferro",
    "title": "Ferrographic Image Dataset",
    "creator": "Li, Zhenlin (ORCID 0009-0007-9916-5399)",
    "published": "2026-09-27",
    "doi": "10.5281/zenodo.22997504",
    "url": "https://zenodo.org/records/22997504",
    "license": "CC-BY-4.0",
    "citation": "Li, Z. (2026). Ferrographic Image Dataset [Data set]. Zenodo. "
                "https://doi.org/10.5281/zenodo.22997504",
}
SPLITS = ("trainO", "testO")
PLACEHOLDER = 1.0  # value of every unknown required numeric field (schema requires finite values)
_PLACEHOLDER_FIELDS = ["specimens.hardness_MPa", "specimens.density_g_cm3", "runs.load_N", "runs.speed_m_s",
                       "runs.duration_s", "runs.sliding_distance_m", "runs.reservoir_volume_ml",
                       "samples.t_end_s", "samples.time_s", "samples.sliding_distance_end_m",
                       "samples.sampled_oil_ml", "samples.imaged_fraction", "samples.observation_fraction",
                       "images.magnification", "images.um_per_px", "views.um_per_px"]


def load_zenodo_ferro(root) -> pd.DataFrame:
    """Index of the extracted archive (root = the folder holding dataset/{trainO,testO}/{images,labels}): one row per
    image that has a mask, columns split, src_id (file stem), image_id ("{split}_{stem}"), image_path, mask_path.
    Masks without an image are skipped (in v. 2026-09-27: 23 testO masks, byte-identical copies of trainO masks)."""
    base = Path(root) / "dataset"
    rows = []
    for split in SPLITS:
        for p in sorted((base / split / "images").glob("*.png"), key=lambda q: (len(q.stem), q.stem)):
            m = base / split / "labels" / p.name
            if m.exists():
                rows.append({"split": split, "src_id": p.stem, "image_id": f"{split}_{p.stem}", "image_path": p,
                             "mask_path": m})
    if not rows:
        raise FileNotFoundError(f"no image/mask pairs under {base}")
    return pd.DataFrame(rows)


def read_mask(path) -> np.ndarray:
    """bool (H, W) foreground of a mask image (grey > 0.5)."""
    return to_gray(load_image(path)) > 0.5


def mask_components(mask, min_area_px: int = 30) -> list[dict]:
    """Connected components (4-connectivity, as segment_image) with >= min_area_px pixels -> [{"contour": (N, 2) float
    px (x, y), positive area, outer boundary from find_contours (holes ignored), "bbox": (x0, y0, x1, y1) exclusive,
    "area_px": pixel count, "touches_border": bool}] in label order (contours with < 8 points dropped)."""
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    lab, _ = ndi.label(mask)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        area = int(sub.sum())
        if area < min_area_px:
            continue
        rc = max(find_contours(np.pad(sub, 1).astype(np.uint8), 0.5), key=lambda a: abs(signed_area(a)))
        xy = ensure_ccw(clean_closed(np.column_stack([rc[:, 1] + sl[1].start - 1, rc[:, 0] + sl[0].start - 1])))
        if len(xy) >= 8:
            out.append({"contour": xy, "bbox": (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop),
                        "area_px": float(area),
                        "touches_border": bool(min(sl[0].start, sl[1].start) == 0 or sl[0].stop == h
                                               or sl[1].stop == w)})
    return out


def to_wear_dataset(index: pd.DataFrame, root, min_area_px: int = 30) -> WearDataset:
    """Schema-valid WearDataset of the indexed images (root = dataset root for images.path): one placeholder
    specimen / run / sample per image, particles and views (source "manual", view_idx 0) from mask_components of the
    GT mask. No labels, no wear fields, is_synthetic False. Extension columns: images.split,
    particles.area_px / touches_border."""
    root = Path(root)
    sp, ru, sa, im, pa, vi = [], [], [], [], [], []
    for r in index.itertuples(index=False):
        mask = read_mask(r.mask_path)
        with Image.open(r.image_path) as f:
            w, h = f.size
        if mask.shape != (h, w):
            raise ValueError(f"{r.image_id}: mask {mask.shape} != image {(h, w)}")
        uid, sid = f"ZF_{r.image_id}", f"ZF_{r.image_id}_s0"
        comps = mask_components(mask, min_area_px)
        sp.append({"specimen_id": uid, "material_id": "unknown", "counterface_id": "unknown",
                   "material_pair": "unknown|unknown", "hardness_MPa": PLACEHOLDER, "density_g_cm3": PLACEHOLDER})
        ru.append({"run_id": uid, "specimen_id": uid, "condition_id": uid, "rig": "other", "load_N": PLACEHOLDER,
                   "speed_m_s": PLACEHOLDER, "lubricant_id": "unknown", "duration_s": PLACEHOLDER,
                   "sliding_distance_m": PLACEHOLDER, "repeat_idx": 0, "reservoir_volume_ml": PLACEHOLDER})
        sa.append({"sample_id": sid, "run_id": uid, "t_start_s": 0.0, "t_end_s": PLACEHOLDER, "time_s": PLACEHOLDER,
                   "sliding_distance_start_m": 0.0, "sliding_distance_end_m": PLACEHOLDER,
                   "sampled_oil_ml": PLACEHOLDER, "imaged_fraction": PLACEHOLDER,
                   "observation_fraction": PLACEHOLDER, "n_particles_total": len(comps), "severity": -1})
        im.append({"image_id": r.image_id, "sample_id": sid, "path": Path(r.image_path).relative_to(root).as_posix(),
                   "modality": "ferrograph_optical", "magnification": PLACEHOLDER, "um_per_px": 1.0,
                   "width_px": w, "height_px": h, "split": r.split})
        for k, c in enumerate(comps):
            pid = f"{r.image_id}_p{k:03d}"
            pa.append({"particle_id": pid, "sample_id": sid, "image_id": r.image_id, "n_views": 1,
                       "has_eds": False, "has_sem": False, "area_px": c["area_px"],
                       "touches_border": c["touches_border"]})
            x0, y0, x1, y1 = c["bbox"]
            vi.append({"view_id": f"{pid}_v0", "particle_id": pid, "view_idx": 0, "image_id": r.image_id,
                       "bbox_x0": x0, "bbox_y0": y0, "bbox_x1": x1, "bbox_y1": y1, "contour_x": c["contour"][:, 0],
                       "contour_y": c["contour"][:, 1], "um_per_px": 1.0, "source": "manual"})
    meta = {"name": ZENODO_FERRO["name"], "is_synthetic": False, "source": dict(ZENODO_FERRO),
            "units": "px, uncalibrated", "unit_of_inference": "source image",
            "placeholders": {f: PLACEHOLDER for f in _PLACEHOLDER_FIELDS},
            "particles_from": f"connected components (4-conn) of the GT foreground masks, >= {min_area_px} px; a "
                              "component can be a chain or agglomerate of several particles",
            "labels": "none", "wear_fields": "none"}
    return WearDataset(specimens=pd.DataFrame(sp), runs=pd.DataFrame(ru), samples=pd.DataFrame(sa),
                       images=pd.DataFrame(im), particles=pd.DataFrame(pa), views=pd.DataFrame(vi), meta=meta,
                       root=root)


# ----------------------------------------------------------------------------- benchmark helpers


def contours_mask(contours, shape) -> np.ndarray:
    """bool (H, W) union of the filled contours (pixel centres inside, as segmentation_quality rasterises them)."""
    m = np.zeros(tuple(shape[:2]), bool)
    for xy in contours:
        xy = np.asarray(xy, float)
        rr, cc = polygon(xy[:, 1], xy[:, 0], shape=m.shape)
        m[rr, cc] = True
    return m


def pixel_scores(pred, gt) -> dict:
    """Pixel IoU / Dice of bool masks, foreground areas [px] and the relative area error (pred - gt) / gt
    (NaN if gt is empty; empty vs empty scores 1)."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    inter, union = int((pred & gt).sum()), int((pred | gt).sum())
    a_p, a_g = int(pred.sum()), int(gt.sum())
    return {"iou": inter / union if union else 1.0, "dice": 2 * inter / (a_p + a_g) if a_p + a_g else 1.0,
            "area_pred": a_p, "area_gt": a_g, "area_rel_err": (a_p - a_g) / a_g if a_g else float("nan")}


def threshold_baseline(img, method: str = "otsu", block_size: int = 51, offset: float = 0.02,
                       min_area_px: int = 30) -> np.ndarray:
    """bool foreground of a simple baseline on the grey image, with segment_image's cleanup (hole filling, drop
    components < min_area_px; no border exclusion). "otsu": global Otsu, foreground = the minority side
    (polarity free). "local": textbook adaptive threshold, foreground = grey < Gaussian local mean over block_size px
    (skimage threshold_local) - offset (dark particles; an |grey - mean| variant marks halos on both edge sides)."""
    g = to_gray(img)
    if method == "otsu":
        fg = g < threshold_otsu(g)
        fg = fg if fg.mean() <= 0.5 else ~fg
    elif method == "local":
        fg = g < threshold_local(g, block_size, method="gaussian", offset=offset)
    else:
        raise ValueError(f"unknown method {method!r}")
    lab, n = ndi.label(ndi.binary_fill_holes(fg))
    keep = np.bincount(lab.ravel(), minlength=n + 1) >= min_area_px
    keep[0] = False
    return keep[lab]
