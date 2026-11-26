"""Open real data: (c) real-particle morphology vs the synthetic smoke dataset and (e) the real-vs-synthetic domain gap.

Particle sets (border-touching particles dropped everywhere; um_per_px = 1.0 so every "um" feature is in px):
  zenodo_gt        GT-mask components of all 429 Zenodo images (doi 10.5281/zenodo.22997504)
  zenodo_seg       segment_image with the trainO-tuned cfg (seg_tuned_params.json) on the same images
  tcbgy_seg        TCBGY particle crops (He et al. 2024; images <= 1500 px; 4 full ferrograms skipped): largest
                   segment_image segment (tuned cfg, exclude_border False) - NO ground truth
  synthetic_gt     data/synthetic_smoke view-0 GT contours in ferrograph FOV images (SIMULATION)
  synthetic_raster the same contours rasterised and re-traced like the Zenodo GT masks (same discretisation; SIMULATION)
Outputs (outputs/open_data/ferro/): particle_descriptors.csv.gz, morph_summary.csv, morph_ks.csv,
domain_gap_auc.csv, domain_gap_features.csv.
Run: .venv/Scripts/python.exe -I scripts/open_ferro_morph.py [--workers 8]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process (the pool provides the parallelism)

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from wearpg.features.descriptors import particle_descriptors
from wearpg.open_data import contours_mask, load_zenodo_ferro, mask_components, read_mask
from wearpg.schema import WearDataset
from wearpg.segmentation import load_image, native_patch, segment_image
from wearpg.stats import cluster_bootstrap
from wearpg.utils import load_json

ROOT = Path(__file__).resolve().parents[1]
ZEN = ROOT / "data" / "real" / "zenodo_ferro" / "extracted"
TCBGY = ROOT / "data" / "real" / "tcbgy" / "images"
SYN = ROOT / "data" / "synthetic_smoke"
OUT = ROOT / "outputs" / "open_data" / "ferro"
MIN_AREA = 30
TCBGY_MAX_SIDE = 1500  # larger TCBGY images are full ferrograms, not particle crops
MORPH = {"size_area_px": "shape_area_um2", "size_ecd_px": "shape_ecd_um", "aspect": "shape_aspect_ratio",
         "roundness": "shape_roundness", "solidity": "shape_solidity", "fractal_D": "fractal_divider_D"}
SHAPE_FREE = ["shape_aspect_ratio", "shape_roundness", "shape_solidity", "shape_convexity", "shape_eccentricity",
              "shape_elongation", "shape_curl"]
PAIRS = [("zenodo_gt", "synthetic_raster"), ("zenodo_seg", "synthetic_raster"), ("tcbgy_seg", "synthetic_raster"),
         ("zenodo_gt", "synthetic_gt"), ("zenodo_gt", "tcbgy_seg"), ("zenodo_gt", "zenodo_seg")]


def _inside(xy, w, h) -> bool:
    """Contour fully inside the image (does not touch the border; 0.5 iso-lines of edge pixels lie at -0.5)."""
    return bool(xy.min() >= 0 and xy[:, 0].max() <= w - 1 and xy[:, 1].max() <= h - 1)


def describe(job: tuple) -> list[dict]:
    """Descriptor rows of the non-border particles of one image."""
    set_name, image_id, image_path, kind, payload = job
    img = load_image(image_path)
    h, w = img.shape[:2]
    if kind == "gt":
        items = [(f"{image_id}_p{k:03d}", c["contour"]) for k, c in enumerate(mask_components(read_mask(payload),
                                                                                              MIN_AREA))]
    elif kind in ("seg", "seg_largest"):
        segs = segment_image(img, payload)
        if kind == "seg_largest":
            segs = sorted(segs, key=lambda s: -s["area_px"])[:1]
        items = [(f"{image_id}_seg{k:04d}", s["contour"]) for k, s in enumerate(segs)]
    elif kind == "raster":  # re-trace rasterised contours like the GT masks
        items = []
        for pid, xy in payload:
            comps = mask_components(contours_mask([xy], img.shape), MIN_AREA)
            if comps:
                items.append((pid, max(comps, key=lambda c: c["area_px"])["contour"]))
    else:
        items = payload
    rows = []
    for pid, xy in items:
        if not _inside(xy, w, h):
            continue
        patch, pm = native_patch(img, xy)
        rows.append({"set": set_name, "image_id": image_id, "particle_id": pid, "n_contour_pts": len(xy),
                     **particle_descriptors([(xy, 1.0)], patch, pm)})
    return rows


def jobs(seg_cfg: dict) -> list[tuple]:
    idx = load_zenodo_ferro(ZEN)
    out = [("zenodo_gt", r.image_id, r.image_path, "gt", r.mask_path) for r in idx.itertuples()]
    out += [("zenodo_seg", r.image_id, r.image_path, "seg", seg_cfg) for r in idx.itertuples()]
    from PIL import Image
    for p in sorted(TCBGY.glob("*.png")):
        with Image.open(p) as f:
            if max(f.size) <= TCBGY_MAX_SIDE:
                out.append(("tcbgy_seg", p.stem, p, "seg_largest", {**seg_cfg, "exclude_border": False}))
    ds = WearDataset.load(SYN)
    fov = set(ds.images.image_id[ds.images.modality == "ferrograph_optical"])
    v = ds.views[(ds.views.view_idx == 0) & ds.views.image_id.isin(fov)]
    for image_id, g in v.groupby("image_id"):
        items = [(r.particle_id, WearDataset.contour(r)) for r in g.itertuples()]
        out.append(("synthetic_gt", image_id, ds.image_path(image_id), "contours", items))
        out.append(("synthetic_raster", image_id, ds.image_path(image_id), "raster", items))
    return out


def morphology(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-set quantiles of the MORPH features (fractal D over fractal_valid particles) and pairwise KS tests."""
    def vals(s: str, col: str) -> np.ndarray:
        d = df[df.set == s]
        return d.loc[d.fractal_valid > 0.5, col].to_numpy() if col == "fractal_divider_D" else d[col].to_numpy()

    summ = []
    for s, d in df.groupby("set"):
        row = {"set": s, "n_particles": len(d), "n_images": d.image_id.nunique(),
               "fractal_valid_fraction": float(d.fractal_valid.mean())}
        for name, col in MORPH.items():
            x = vals(s, col)
            row.update({f"{name}_n": len(x), **{f"{name}_q{int(q * 100):02d}": float(np.quantile(x, q)) if len(x)
                                                 else np.nan for q in (0.1, 0.25, 0.5, 0.75, 0.9)}})
        summ.append(row)
    ks = []
    for a, b in PAIRS:
        for name, col in MORPH.items():
            x, y = vals(a, col), vals(b, col)
            r = sps.ks_2samp(x, y) if len(x) and len(y) else None
            ks.append({"a": a, "b": b, "feature": name, "ks_D": r.statistic if r else np.nan,
                       "p_value": r.pvalue if r else np.nan, "median_a": np.median(x) if len(x) else np.nan,
                       "median_b": np.median(y) if len(y) else np.nan, "n_a": len(x), "n_b": len(y)})
        fa, fb = df[df.set == a].fractal_valid.mean(), df[df.set == b].fractal_valid.mean()
        ks.append({"a": a, "b": b, "feature": "fractal_valid_fraction", "median_a": fa, "median_b": fb,
                   "n_a": int((df.set == a).sum()), "n_b": int((df.set == b).sum())})
    return pd.DataFrame(summ), pd.DataFrame(ks)


def groups(cols) -> dict[str, list[str]]:
    return {"shape_scale_free": SHAPE_FREE, "size_px": ["shape_log_area_um2", "shape_log_ecd_um"],
            "efd": [c for c in cols if c.startswith("efd_")], "fractal": [c for c in cols if c.startswith("fractal_")],
            "texture": [c for c in cols if c.startswith("tex_")], "colour": [c for c in cols if c.startswith("color_")]}


def domain_gap(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classifier two-sample test per descriptor group: out-of-fold AUC (HistGradientBoosting defaults,
    5-fold StratifiedGroupKFold by source image) with a 95% image-cluster bootstrap CI; plus univariate AUCs."""
    rows, feats = [], []
    for a, b in PAIRS[:5]:
        d = df[df.set.isin([a, b])].reset_index(drop=True)
        y = (d.set == a).to_numpy(int)
        for gname, cols in groups(df.columns).items():
            oof = np.zeros(len(d))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(d, y, d.image_id):
                clf = HistGradientBoostingClassifier(random_state=0).fit(d.loc[tr, cols], y[tr])
                oof[te] = clf.predict_proba(d.loc[te, cols])[:, 1]
            e = pd.DataFrame({"y": y, "p": oof, "image_id": d.image_id})
            ci = cluster_bootstrap(lambda t: roc_auc_score(t.y, t.p) if t.y.nunique() == 2 else np.nan, e,
                                   "image_id", n_boot=1000, seed=0)["value"]
            rows.append({"a": a, "b": b, "group": gname, "n_features": len(cols), "auc": ci["point"],
                         "lo": ci["lo"], "hi": ci["hi"], "n_a": int(y.sum()), "n_b": int((1 - y).sum())})
            print(f"gap {a} vs {b} {gname}: auc {ci['point']:.3f}", flush=True)
            for c in cols:
                feats.append({"a": a, "b": b, "group": gname, "feature": c,
                              "auc_univariate": roc_auc_score(y, d[c]), "median_a": d.loc[y == 1, c].median(),
                              "median_b": d.loc[y == 0, c].median()})
    f = pd.DataFrame(feats)
    f["effect"] = (f.auc_univariate - 0.5).abs()
    return pd.DataFrame(rows), f.sort_values(["a", "b", "effect"], ascending=[True, True, False])


def main() -> None:
    ap = argparse.ArgumentParser(description="real vs synthetic morphology and domain gap")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cache = OUT / "particle_descriptors.csv.gz"
    if not cache.exists():
        tuned = load_json(OUT / "seg_tuned_params.json")["best"]["ours"]
        seg_cfg = {k: v for k, v in tuned.items() if not k.startswith("train_")}
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for part in ex.map(describe, jobs(seg_cfg), chunksize=4) for r in part]
        pd.DataFrame(rows).to_csv(cache, index=False)
    df = pd.read_csv(cache)
    print(df.groupby("set").size().to_string())
    summ, ks = morphology(df)
    summ.to_csv(OUT / "morph_summary.csv", index=False)
    ks.to_csv(OUT / "morph_ks.csv", index=False)
    auc, feats = domain_gap(df)
    auc.to_csv(OUT / "domain_gap_auc.csv", index=False)
    feats.to_csv(OUT / "domain_gap_features.csv", index=False)


if __name__ == "__main__":
    main()
