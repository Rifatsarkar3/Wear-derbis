"""Open real data: (d) invariance / robustness of descriptors and contour-graph features on REAL particles (claim C5 and
the "a contour graph is a cycle graph" objection). Zenodo Ferrographic Image Dataset (doi 10.5281/zenodo.22997504).

300 random non-border GT components with >= 120 px (seed 0, trainO + testO; no tuning happens here) are transformed:
rotation (45, 90 deg, one uniform random angle per image), rescale 0.5x / 0.25x (magnification shift, anti-aliased),
Gaussian blur sigma 1 / 2 px, additive Gaussian noise sd 0.02 / 0.05 (grey in [0, 1]). Two contour sources:
  mask      image and the particle's GT mask transformed together; contour re-traced from the transformed mask
            (blurred / rescaled masks thresholded at 0.5; noise leaves the mask unchanged, so noise is pipeline only)
  pipeline  segment_image (trainO-tuned cfg) re-run on the transformed full image; the segment with IoU >= 0.5 to the
            transformed GT mask is used; reference = the same on the untransformed image
Features: view_descriptors (shape_*, efd_*, fractal_*; size features rescaled by 1/s, 1/s^2), boundary-node means /
SDs of graphs.contour_graph node features (with the transformed image for tex_*), median width-chord hop distance.
Outputs (outputs/open_data/ferro/): robust_particles.csv.gz, robust_stability.csv, robust_chord_hops.csv,
robust_summary.json.   Run: .venv/Scripts/python.exe -I scripts/open_ferro_robust.py [--workers 8]
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
from scipy import ndimage as ndi
from scipy import stats as sps
from skimage.transform import rescale, rotate

from wearpg.features.descriptors import view_descriptors
from wearpg.graphs import EDGE_CHORD, GraphConfig, contour_graph
from wearpg.open_data import contours_mask, load_zenodo_ferro, mask_components, read_mask
from wearpg.segmentation import load_image, segment_image
from wearpg.utils import load_json, save_json

ROOT = Path(__file__).resolve().parents[1]
ZEN = ROOT / "data" / "real" / "zenodo_ferro" / "extracted"
OUT = ROOT / "outputs" / "open_data" / "ferro"
MIN_AREA, N_PARTICLES, MIN_SEL_AREA, SEED = 30, 300, 120, 0
GCFG = GraphConfig()
TRANSFORMS = ["identity", "rot45", "rot90", "rot_random", "scale0.5", "scale0.25", "blur1", "blur2", "noise0.02",
              "noise0.05"]
FLOORED = ("fractal_divider_D", "fractal_divider_r2", "fractal_box_D", "fractal_box_r2")  # px-floor validity rule
LEN_FEATS = ["shape_perimeter_um", "shape_ecd_um", "shape_major_um", "shape_minor_um", "shape_feret_max_um",
             "shape_feret_min_um"]


def transform(img: np.ndarray, masks: list[np.ndarray], name: str, angle: float, rng) -> tuple:
    """(transformed float image, transformed bool masks, scale factor)."""
    m = [x.astype(float) for x in masks]
    if name.startswith("rot"):
        a = {"rot45": 45.0, "rot90": 90.0}.get(name, angle)
        return (rotate(img, a, resize=True, order=1, mode="reflect"),
                [rotate(x, a, resize=True, order=1) > 0.5 for x in m], 1.0)
    if name.startswith("scale"):
        s = float(name[5:])
        return (rescale(img, s, order=1, anti_aliasing=True, channel_axis=-1),
                [rescale(x, s, order=1, anti_aliasing=True) > 0.5 for x in m], s)
    if name.startswith("blur"):
        s = float(name[4:])
        return ndi.gaussian_filter(img, (s, s, 0)), [ndi.gaussian_filter(x, s) > 0.5 for x in m], 1.0
    if name.startswith("noise"):
        return np.clip(img + rng.normal(0, float(name[5:]), img.shape), 0, 1), masks, 1.0
    return img, masks, 1.0


def features(xy: np.ndarray, image: np.ndarray, s: float) -> dict:
    """view_descriptors (size features divided by s, s^2), boundary-node mean / SD of the contour-graph node features
    and the median width-chord hop distance."""
    d = view_descriptors(xy, 1.0)
    d["shape_area_um2"] /= s * s
    d["shape_log_area_um2"] -= 2 * np.log(s)
    d["shape_log_ecd_um"] -= np.log(s)
    for c in LEN_FEATS:
        d[c] /= s
    g = contour_graph(xy, 1.0, GCFG, image)
    x = g["x"][g["node_type"] == 0].numpy()
    for j, name in enumerate(GCFG.node_feature_names()[:-1]):  # is_interior is constant on boundary nodes
        d[f"graph_mean_{name}"], d[f"graph_std_{name}"] = float(x[:, j].mean()), float(x[:, j].std())
    n = GCFG.n_nodes
    i, k = g["edge_index"]
    ch = ((g["edge_kind"] & EDGE_CHORD) > 0) & (i < k) & (k < n)
    hop = np.minimum((k - i)[ch].numpy(), n - (k - i)[ch].numpy())
    d["chord_median_hop"] = float(np.median(hop)) if hop.size else np.nan
    d["chord_n"] = int(hop.size)
    return d


def _largest(mask: np.ndarray) -> np.ndarray | None:
    comps = mask_components(mask, MIN_AREA)
    return max(comps, key=lambda c: c["area_px"])["contour"] if comps else None


def _match(segs: list[dict], target: np.ndarray) -> np.ndarray | None:
    """Contour of the segment with the highest IoU (>= 0.5) to the target mask."""
    best, best_iou = None, 0.5
    for s in segs:
        x0, y0, x1, y1 = s["bbox"]
        if not target[y0:y1, x0:x1].any():
            continue
        m = contours_mask([s["contour"]], target.shape)
        iou = (m & target).sum() / (m | target).sum()
        if iou >= best_iou:
            best, best_iou = s["contour"], iou
    return best


def run_image(job: tuple) -> list[dict]:
    image_id, image_path, mask_path, labels, seg_cfg, seed = job
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0, 360))
    img = load_image(image_path).astype(float) / 255
    comps = mask_components(read_mask(mask_path), MIN_AREA)
    masks = [contours_mask([comps[k]["contour"]], img.shape) for k in labels]
    rows = []
    for t in TRANSFORMS:
        im_t, m_t, s = transform(img, masks, t, angle, rng)
        segs = segment_image(im_t, seg_cfg)
        for k, mt in zip(labels, m_t):
            for mode, xy in (("mask", _largest(mt) if not t.startswith("noise") else None), ("pipeline", _match(segs, mt))):
                base = {"image_id": image_id, "particle": f"{image_id}_p{k:03d}", "area_px": comps[k]["area_px"],
                        "mode": mode, "transform": t, "angle": angle if t == "rot_random" else np.nan}
                if xy is None or len(xy) < 8:
                    rows.append({**base, "lost": True})
                    continue
                rows.append({**base, "lost": False, **features(xy, im_t, s)})
    return rows


def stability(df: pd.DataFrame) -> pd.DataFrame:
    """Per (mode, transform, feature): Spearman of transformed vs reference values over particles present in both,
    median |delta| / SD(reference), median relative change; FLOORED fractal values only over particles valid in both."""
    feats = [c for c in df.columns if c.startswith(("shape_", "efd_", "fractal_", "graph_"))]
    rows = []
    for mode, g in df.groupby("mode"):
        ref = g[(g["transform"] == "identity") & ~g.lost].set_index("particle")
        for t in TRANSFORMS[1:]:
            tr = g[g["transform"] == t].set_index("particle")
            if tr.lost.all():
                continue
            ok = tr.index[~tr.lost.astype(bool)].intersection(ref.index)
            both_valid = ok[(ref.loc[ok, "fractal_valid"] > 0.5).to_numpy() & (tr.loc[ok, "fractal_valid"] > 0.5).to_numpy()]
            ref_valid = ok[(ref.loc[ok, "fractal_valid"] > 0.5).to_numpy()]
            for f in feats:
                ids = both_valid if f in FLOORED else ok
                a, b = ref.loc[ids, f].to_numpy(float), tr.loc[ids, f].to_numpy(float)
                if len(a) < 3:
                    continue
                dlt = np.abs(b - a)
                sd = a.std()
                rows.append({"mode": mode, "transform": t, "feature": f,
                             "group": f.split("_")[0] if not f.startswith("graph_") else
                             ("graph_tex" if "_tex_" in f else "graph_geom"),
                             "spearman": sps.spearmanr(a, b)[0] if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan,
                             "median_abs_delta_over_sd": float(np.median(dlt) / sd) if sd > 0 else np.nan,
                             "median_rel_change": float(np.median(dlt / np.maximum(np.abs(a), 1e-12))),
                             "n": len(a), "n_ref": len(ref), "n_lost": int(tr.lost.sum()),
                             "fractal_valid_ref": len(ref_valid) / max(len(ok), 1),
                             "fractal_valid_flip": 1 - len(both_valid) / max(len(ref_valid), 1)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="real-particle invariance / robustness")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cache = OUT / "robust_particles.csv.gz"
    if not cache.exists():
        tuned = load_json(OUT / "seg_tuned_params.json")["best"]["ours"]
        seg_cfg = {k: v for k, v in tuned.items() if not k.startswith("train_")}
        idx = load_zenodo_ferro(ZEN)
        cand = []
        for r in idx.itertuples():
            m = read_mask(r.mask_path)
            cand += [(r.image_id, k) for k, c in enumerate(mask_components(m, MIN_AREA))
                     if not c["touches_border"] and c["area_px"] >= MIN_SEL_AREA]
        rng = np.random.default_rng(SEED)
        pick = pd.DataFrame([cand[i] for i in rng.choice(len(cand), N_PARTICLES, replace=False)],
                            columns=["image_id", "k"])
        paths = idx.set_index("image_id")
        jobs = [(i, paths.at[i, "image_path"], paths.at[i, "mask_path"], sorted(g.k.tolist()), seg_cfg, SEED + n)
                for n, (i, g) in enumerate(pick.groupby("image_id"))]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = [r for part in ex.map(run_image, jobs) for r in part]
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        save_json({"n_candidates": len(cand), "n_selected": N_PARTICLES, "n_images": len(jobs),
                   "min_selection_area_px": MIN_SEL_AREA, "seed": SEED, "seg_cfg": seg_cfg,
                   "graph_cfg_n_nodes": GCFG.n_nodes}, OUT / "robust_summary.json")
    df = pd.read_csv(cache)
    st = stability(df)
    st.to_csv(OUT / "robust_stability.csv", index=False)
    ref = df[(df["transform"] == "identity") & ~df.lost]
    hops = ref[["mode", "particle", "area_px", "chord_median_hop", "chord_n"]]
    hops.to_csv(OUT / "robust_chord_hops.csv", index=False)
    summ = load_json(OUT / "robust_summary.json")
    summ["chord_hop"] = {m: {"n_particles": len(h), "median_of_median_hop": float(h.chord_median_hop.median()),
                             "fraction_median_hop_gt_n_over_8": float((h.chord_median_hop > GCFG.n_nodes / 8).mean()),
                             "n_over_8": GCFG.n_nodes / 8} for m, h in hops.groupby("mode")}
    summ["group_median_spearman"] = (st.groupby(["mode", "transform", "group"]).spearman.median().unstack("group")
                                     .round(4).reset_index().to_dict(orient="records"))
    save_json(summ, OUT / "robust_summary.json")
    print(st.groupby(["mode", "transform", "group"]).spearman.median().unstack("group").round(3).to_string())


if __name__ == "__main__":
    main()
