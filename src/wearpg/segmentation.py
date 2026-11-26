"""Particle segmentation of ferrograph images, crops / native patches, and the blinded rating package.

Image coordinates: x = column, y = row, pixel centres at integers; contours are (N, 2) float (x, y) with positive
shoelace area and no repeated end point (DESIGN sec. 5); bboxes (x0, y0, x1, y1) with exclusive x1, y1.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import io
from skimage.color import rgb2gray
from skimage.draw import polygon
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label
from skimage.morphology import local_maxima, reconstruction
from skimage.segmentation import watershed
from skimage.transform import resize
from skimage.util import img_as_float32, img_as_ubyte

from wearpg.features.contour import clean_closed, ensure_ccw, polygon_mask, signed_area
from wearpg.schema import MECHANISMS, WearDataset, coerce_table
from wearpg.utils import get_logger

from wearpg.schema import SYNTHETIC_BANNER as _BANNER  # single source

# segment_image parameters (flat dict; cfg overrides individual keys)
SEG_DEFAULTS = {
    "polarity": "both",     # both (|gray - background|) | dark | bright
    "smooth_sigma": 1.0,    # px, Gaussian smoothing of (gray - background) before thresholding
    "noise_k": 3.0,         # threshold floor = noise_k * robust noise sd (guards Otsu on particle-free images)
    "min_area_px": 30,      # drop segments smaller than this [px]
    "split_ratio": 0.5,     # watershed splits only where the neck half-width < split_ratio * smaller lobe radius
    "exclude_border": True,  # drop particles touching the image border (truncated contours)
}

CLASS_DEFINITIONS = {
    "rubbing": "Normal rubbing wear: thin, flat platelets with smooth surfaces and rounded outlines; small "
               "compared with the other classes.",
    "cutting": "Cutting wear: elongated curved or spiral ribbons / curls (like machining swarf); length much "
               "greater than width.",
    "spherical": "Spheres: near-perfect spheres, usually very small, often with a central specular highlight "
                 "(rolling-contact fatigue cracks, melting or welding).",
    "fatigue_chunk": "Fatigue chunks: thick, blocky particles with irregular, jagged outlines and low aspect "
                     "ratio (spalling / pitting).",
    "laminar": "Laminar particles: large, very thin, flat platelets with smooth surfaces, sometimes with holes "
               "(material passed through a rolling contact).",
    "severe_sliding": "Severe sliding wear: large particles with straight edges and parallel surface "
                      "striations along the sliding direction.",
    "oxide": "Oxides: dark (black oxide) or reddish (red oxide) particles of low reflectance, often multi-lobed "
             "agglomerates of small grains.",
}


# ----------------------------------------------------------------------------- images


def load_image(path) -> np.ndarray:
    """(H, W, 3) uint8 RGB; grey images are repeated to 3 channels, alpha is dropped, other dtypes rescaled."""
    img = io.imread(Path(path))
    if img.ndim == 3 and img.shape[2] in (2, 4):  # grey+alpha or RGBA
        img = img[..., : img.shape[2] - 1]
    if img.ndim == 2 or img.shape[2] == 1:
        img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, axis=2)
    return np.ascontiguousarray(img if img.dtype == np.uint8 else img_as_ubyte(img))


def to_gray(img) -> np.ndarray:
    """float32 luminance in [0, 1] (Rec. 709 weights for RGB)."""
    img = np.asarray(img)
    return (rgb2gray(img) if img.ndim == 3 else img_as_float32(img)).astype(np.float32)


def _background(gray: np.ndarray, step: int = 4, n_iter: int = 4) -> tuple[np.ndarray, float]:
    """(illumination surface, raw noise sd). Quadratic surface least-squares fitted on a pixel grid, starting from the
    half of the grid closest to the median and refitted on pixels within 3 noise sd of the surface: survives
    particles of any size, number and polarity as long as background is the majority of the image (a median window
    does not). The noise sd comes from neighbour differences, so particle coverage cannot inflate it (floor: half
    an 8-bit step)."""
    sd = max(1.4826 / np.sqrt(2) * float(np.median(np.abs(np.diff(gray, axis=1)))), 0.5 / 255)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    x, y, v = xx.ravel() / w - 0.5, yy.ravel() / h - 0.5, gray[::step, ::step].ravel().astype(np.float64)
    design = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    r = np.abs(v - np.median(v))
    keep = r <= np.median(r)  # least-trimmed start
    for _ in range(n_iter):
        coef = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
        new = np.abs(v - design @ coef) <= 3 * sd
        if new.sum() < design.shape[1]:
            break
        keep = new
    x, y = np.arange(w) / w - 0.5, (np.arange(h) / h - 0.5)[:, None]
    bg = coef[0] + coef[1] * x + coef[2] * y + coef[3] * x * x + coef[4] * x * y + coef[5] * y * y
    return bg.astype(np.float32), sd


def segment_image(img, cfg: dict | None = None) -> list[dict]:
    """Segment particles; returns [{"contour": (N, 2) float px (x, y), "bbox": (x0, y0, x1, y1), "area_px": float
    (pixel count)}] in label order.

    Pipeline: robust quadratic background; score = |gray - background| smoothed (or the one-sided contrast for
    cfg["polarity"] dark | bright); threshold = max(Otsu(score), noise_k * robust sd of the score on background
    pixels); hole filling; drop components < min_area_px or touching the border; watershed on the distance transform
    with markers = h-maxima of ln(distance) (h = ln(1 / split_ratio): a relative neck criterion, so touching
    particles split while multi-lobed agglomerates and ribbons of varying width do not); outer find_contours
    contour per region. cfg keys and defaults: SEG_DEFAULTS.
    """
    c = {**SEG_DEFAULTS, **(cfg or {})}
    if c["polarity"] not in ("both", "dark", "bright"):
        raise ValueError(f"polarity must be both | dark | bright, got {c['polarity']!r}")
    gray = to_gray(img)
    bg, sd = _background(gray)
    diff = ndi.gaussian_filter(gray - bg, c["smooth_sigma"])
    score = {"both": np.abs(diff), "dark": np.maximum(-diff, 0), "bright": np.maximum(diff, 0)}[c["polarity"]]
    d_bg = diff[np.abs(gray - bg) <= 3 * sd]  # background pixels only: particle coverage must not inflate the floor
    noise = 1.4826 * float(np.median(np.abs(d_bg - np.median(d_bg)))) if d_bg.size else 0.0
    thr = max(float(threshold_otsu(score)), c["noise_k"] * noise)
    mask = ndi.binary_fill_holes(score > thr)

    lab, n = ndi.label(mask)
    drop = np.bincount(lab.ravel(), minlength=n + 1) < c["min_area_px"]
    if c["exclude_border"]:
        drop[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True
    drop[0] = True
    mask = ~drop[lab]

    dist = ndi.distance_transform_edt(mask)
    h = int(round(100 * np.log(1.0 / c["split_ratio"])))
    q = np.zeros(mask.shape, np.int32)  # ln(distance) in 1/100 units, lifted so every component keeps a maximum
    q[mask] = np.round(100 * np.log(dist[mask])).astype(np.int32) + h + 1
    # h-maxima = regional maxima of the reconstruction of q - h under q (skimage's h_maxima marks equal ridge peaks
    # joined by shallow saddles separately, which shreds elongated particles)
    markers = label(local_maxima(reconstruction(q - h, q)) & mask)
    regions = watershed(-dist, markers, mask=mask)

    out = []
    for i, sl in enumerate(ndi.find_objects(regions), start=1):
        if sl is None:
            continue
        sub = regions[sl] == i
        area = int(sub.sum())
        if area < c["min_area_px"]:
            continue
        cs = find_contours(np.pad(sub, 1).astype(np.uint8), 0.5)
        rc = max(cs, key=lambda a: abs(signed_area(a)))  # outer boundary
        xy = ensure_ccw(clean_closed(np.column_stack([rc[:, 1] + sl[1].start - 1, rc[:, 0] + sl[0].start - 1])))
        if len(xy) >= 8:
            out.append({"contour": xy, "bbox": (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop),
                        "area_px": float(area)})
    return out


# ----------------------------------------------------------------------------- quality


def _label_image(contours, shape) -> np.ndarray:
    """int32 label image (k + 1 inside contour k; later contours overwrite earlier ones where they overlap)."""
    lab = np.zeros(tuple(shape[:2]), np.int32)
    for k, xy in enumerate(contours, start=1):
        xy = np.asarray(xy, dtype=np.float64)
        rr, cc = polygon(xy[:, 1], xy[:, 0], shape=lab.shape)
        lab[rr, cc] = k
    return lab


def _iou_matrix(pred, gt, shape) -> np.ndarray:
    """(n_pred, n_gt) pixel-mask IoU from one contingency table of the two label images."""
    n_p, n_g = len(pred), len(gt)
    key = _label_image(pred, shape).astype(np.int64) * (n_g + 1) + _label_image(gt, shape)
    cont = np.bincount(key.ravel(), minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
    inter = cont[1:, 1:]
    union = cont[1:].sum(axis=1)[:, None] + cont[:, 1:].sum(axis=0)[None, :] - inter
    return inter / np.maximum(union, 1)


def _match(iou: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian (max total IoU) one-to-one matching, keeping pairs with IoU >= thr."""
    if iou.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    r, c = linear_sum_assignment(-iou)
    keep = iou[r, c] >= thr
    return r[keep], c[keep]


def _prf(tp: int, n_pred: int, n_gt: int, sum_iou: float) -> dict:
    p = tp / n_pred if n_pred else float(n_gt == 0)
    r = tp / n_gt if n_gt else float(n_pred == 0)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r > 0 else 0.0,
            "mean_iou": sum_iou / tp if tp else float("nan"), "n_pred": n_pred, "n_gt": n_gt, "n_matched": tp}


def segmentation_quality(pred: list[np.ndarray], gt: list[np.ndarray], shape, iou_thr: float = 0.5) -> dict:
    """Object-level quality of predicted vs ground-truth contours on one image of `shape`: Hungarian matching on
    pixel-mask IoU; a match counts if IoU >= iou_thr. {precision, recall, f1, mean_iou (over matches, NaN if none),
    n_pred, n_gt, n_matched}; an empty prediction of an empty image scores 1."""
    iou = _iou_matrix(pred, gt, shape)
    r, c = _match(iou, iou_thr)
    return _prf(len(r), len(pred), len(gt), float(iou[r, c].sum()))


# ----------------------------------------------------------------------------- dataset


def segment_dataset(ds: WearDataset, cfg: dict | None = None, modalities=("ferrograph_optical",),
                    match_gt: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Segment every image of the given modalities -> (particles_df, views_df with source "segmented", quality).

    Each segment is a new particle "{image_id}_seg{k:04d}" with one view (view_idx 0); with match_gt a segment
    matched (IoU >= 0.5, Hungarian) to an existing view keeps that view's particle_id / view_idx, and the particle
    row (EDS, labels via id, ...) is copied from ds.particles. Segments in multiview_frame images are kept only when
    matched (otherwise they cannot be assigned to a particle), and views of particles without a segmented primary
    view are dropped. quality = segmentation_quality pooled over the images that have views in ds (NaN scores and
    zero counts if none) + n_images, n_images_with_gt, n_segments, n_unassigned_frame_segments.
    Callers replacing ds tables must also filter labels to the kept particles and raise samples.n_particles_total
    to at least the segmented count per sample.
    """
    if ds.root is None:
        raise ValueError("dataset has no root; cannot load images")
    images = ds.images[ds.images.modality.isin(list(modalities))]
    gt_views = {k: g for k, g in ds.views.groupby("image_id")}
    rows, tp, n_pred_gt, n_gt, sum_iou, n_with_gt, n_seg, n_unassigned = [], 0, 0, 0, 0.0, 0, 0, 0
    for im in images.itertuples(index=False):
        img = load_image(ds.root / im.path)
        segs = segment_image(img, cfg)
        n_seg += len(segs)
        assign: dict[int, tuple[str, int]] = {}
        g = gt_views.get(im.image_id)
        if g is not None and len(g):
            iou = _iou_matrix([s["contour"] for s in segs], [WearDataset.contour(v) for v in g.itertuples()], img.shape)
            r, cidx = _match(iou, 0.5)
            tp, n_gt, n_pred_gt, n_with_gt = tp + len(r), n_gt + len(g), n_pred_gt + len(segs), n_with_gt + 1
            sum_iou += float(iou[r, cidx].sum())
            if match_gt:
                assign = {int(i): (g.particle_id.iat[j], int(g.view_idx.iat[j])) for i, j in zip(r, cidx)}
        for k, s in enumerate(segs):
            if k in assign:
                pid, vidx = assign[k]
            elif im.modality == "multiview_frame":
                n_unassigned += 1
                continue
            else:
                pid, vidx = f"{im.image_id}_seg{k:04d}", 0
            x0, y0, x1, y1 = s["bbox"]
            rows.append(dict(view_id=f"{pid}_v{vidx}", particle_id=pid, view_idx=vidx, image_id=im.image_id,
                             bbox_x0=x0, bbox_y0=y0, bbox_x1=x1, bbox_y1=y1, contour_x=s["contour"][:, 0],
                             contour_y=s["contour"][:, 1], um_per_px=im.um_per_px, source="segmented",
                             sample_id=im.sample_id))
    views = pd.DataFrame(rows, columns=["view_id", "particle_id", "view_idx", "image_id", "bbox_x0", "bbox_y0",
                                        "bbox_x1", "bbox_y1", "contour_x", "contour_y", "um_per_px", "source",
                                        "sample_id"])
    views = views[views.particle_id.isin(set(views.particle_id[views.view_idx == 0]))]
    n_views = views.groupby("particle_id").size()
    known = ds.particles.set_index("particle_id")
    parts = []
    for v in views[views.view_idx == 0].itertuples(index=False):
        row = known.loc[v.particle_id].to_dict() if v.particle_id in known.index else {"sample_id": v.sample_id}
        row.update(particle_id=v.particle_id, image_id=v.image_id, n_views=int(n_views[v.particle_id]))
        parts.append(row)
    quality = _prf(tp, n_pred_gt, n_gt, sum_iou)
    if not n_with_gt:
        quality.update({k: float("nan") for k in ("precision", "recall", "f1", "mean_iou")})
    quality.update(n_images=len(images), n_images_with_gt=n_with_gt, n_segments=n_seg,
                   n_unassigned_frame_segments=n_unassigned)
    get_logger().info("segmented %d images: %d segments, %d particles kept", len(images), n_seg, len(parts))
    return (coerce_table(pd.DataFrame(parts), "particles"),
            coerce_table(views.drop(columns="sample_id"), "views"), quality)


# ----------------------------------------------------------------------------- crops


def _crop_window(xy: np.ndarray, pad: float) -> tuple[int, int, int]:
    """(x0, y0, s): top-left pixel and side [px] of the square crop around the contour's bbox centre."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    s = int(np.ceil(max(8.0, pad * float((hi - lo).max()))))
    c = (lo + hi) / 2
    return int(round(c[0] - (s - 1) / 2)), int(round(c[1] - (s - 1) / 2)), s


def extract_crop(img, contour_xy, size: int = 64, pad: float = 1.25) -> np.ndarray:
    """(size, size, 4) uint8: RGB + mask * 255 of the square (side pad * max(bbox w, h), >= 8 px) centred on the
    contour's bbox, resampled to size x size (anti-aliased when shrinking). Outside the image the RGB is
    edge-padded and the mask is 0. The mask is rasterised at crop resolution."""
    xy = np.asarray(contour_xy, dtype=np.float64)
    h, w = img.shape[:2]
    x0, y0, s = _crop_window(xy, pad)
    region = img[np.ix_(np.clip(np.arange(y0, y0 + s), 0, h - 1), np.clip(np.arange(x0, x0 + s), 0, w - 1))]
    rgb = resize(region, (size, size), order=1, anti_aliasing=s > size, preserve_range=True)
    f = size / s  # crop px per image px; output pixel j centre <- image x = x0 - 0.5 + (j + 0.5) / f
    mask = polygon_mask((xy - [x0 - 0.5, y0 - 0.5]) * f - 0.5, (size, size))
    centers = (np.arange(size) + 0.5) / f - 0.5
    mask &= ((y0 + centers >= -0.5) & (y0 + centers < h - 0.5))[:, None]
    mask &= ((x0 + centers >= -0.5) & (x0 + centers < w - 0.5))[None, :]
    return np.dstack([np.clip(np.round(rgb), 0, 255).astype(np.uint8), mask.astype(np.uint8) * 255])


def native_patch(img, contour_xy, margin: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(patch (h, w, C) copy at native resolution, bool mask (h, w)): the contour's pixel bbox grown by `margin` px
    and clipped to the image."""
    xy = np.asarray(contour_xy, dtype=np.float64)
    h, w = img.shape[:2]
    x0, y0 = max(int(np.floor(xy[:, 0].min())) - margin, 0), max(int(np.floor(xy[:, 1].min())) - margin, 0)
    x1 = min(int(np.ceil(xy[:, 0].max())) + margin + 1, w)
    y1 = min(int(np.ceil(xy[:, 1].max())) + margin + 1, h)
    patch = np.array(img[y0:y1, x0:x1])
    return patch, polygon_mask(xy - [x0, y0], patch.shape)


# ----------------------------------------------------------------------------- rating package


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", str(s))


def export_rating_package(ds: WearDataset, out_dir, crop_size: int = 128, raters=("A", "B")) -> Path:
    """Blinded labeling package: crops/<particle_id>.png (RGB view-0 crop), labels_template_<rater>.csv (labels
    schema columns + crop_file + crop_width_um, rows shuffled per rater with a fixed seed) and README.txt with
    instructions and class definitions (synthetic banner first when ds.is_synthetic). Returns out_dir."""
    if ds.root is None:
        raise ValueError("dataset has no root; cannot load images")
    out = Path(out_dir)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    v0 = ds.views[ds.views.view_idx == 0].set_index("particle_id")
    paths = ds.images.set_index("image_id")["path"]
    rows = []
    for image_id, g in ds.particles.groupby("image_id", sort=True):
        img = load_image(ds.root / paths[image_id])
        for pid in g.particle_id:
            v = v0.loc[pid]
            xy = WearDataset.contour(v)
            name = f"crops/{_safe(pid)}.png"
            io.imsave(out / name, extract_crop(img, xy, size=crop_size)[..., :3], check_contrast=False)
            rows.append({"particle_id": pid, "crop_file": name,
                         "crop_width_um": round(_crop_window(xy, 1.25)[2] * float(v.um_per_px), 2)})
    base = pd.DataFrame(rows, columns=["particle_id", "crop_file", "crop_width_um"])
    for i, rater in enumerate(raters):
        t = base.sample(frac=1.0, random_state=i).assign(rater_id=rater, label="", confidence="", evidence="")
        t[["particle_id", "rater_id", "label", "confidence", "evidence", "crop_file", "crop_width_um"]].to_csv(
            out / f"labels_template_{_safe(rater)}.csv", index=False, encoding="utf-8", lineterminator="\n")
    lines = ([_BANNER, ""] if ds.is_synthetic else []) + [
        "WEAR-DEBRIS RATING PACKAGE",
        "",
        f"{len(base)} particles. crops/ holds one RGB image per particle (the particle is centred; the crop is",
        f"rescaled to {crop_size} px, so read its physical width from the crop_width_um column).",
        "",
        "Instructions:",
        "- Label every particle independently; do not discuss labels with other raters.",
        "- Use only the image (and SEM/EDS evidence if you were given it); ignore file names and row order.",
        f"- Fill `label` with exactly one class name: {', '.join(MECHANISMS)}. Leave it empty to skip.",
        "- `confidence`: 1 (guess) to 5 (certain).",
        "- `evidence`: optical | sem | eds | sem_eds (what you used).",
        "",
        "Class definitions (see docs/labeling_protocol.md for the authoritative criteria):",
        *[f"- {k}: {v}" for k, v in CLASS_DEFINITIONS.items()],
        "",
    ]
    (out / "README.txt").write_text("\n".join(lines), encoding="utf-8")
    return out
