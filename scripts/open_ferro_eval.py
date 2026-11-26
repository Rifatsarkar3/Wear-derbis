"""Open real data, Zenodo "Ferrographic Image Dataset" (doi 10.5281/zenodo.22997504, CC-BY-4.0):
(a) WearDataset conversion check and (b) segmentation benchmark against the GT foreground masks.

Stages (outputs in outputs/open_data/ferro/):
  convert  index + open_data.to_wear_dataset + validate            -> dataset_summary.json
  tune     small grids on a fixed random subset of trainO ONLY (TUNE_IMAGES images, seed 0; segment_image cfg dict;
           local adaptive baseline), pixel metrics only, selection by mean pixel IoU
                                                                    -> seg_tuning_grid.csv, seg_tuned_params.json
  test     testO scored ONCE (re-scoring needs --force; the summary is recomputed from the saved per-image CSV)
                                                                    -> seg_per_image_{train,test}.csv,
                                                                       seg_summary_test.json, seg_paired_test.csv,
                                                                       seg_test_provenance.json
  leak     exact / near-duplicate testO images in trainO (8 rotations / flips) -> seg_leak_check.csv,
                                                                       seg_leak_summary.json
Run from anywhere:  .venv/Scripts/python.exe -I scripts/open_ferro_eval.py --stage all
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process (the pool provides the parallelism)

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from wearpg.metrics import volume_consistency
from wearpg.open_data import (ZENODO_FERRO, contours_mask, load_zenodo_ferro, mask_components, pixel_scores,
                              read_mask, threshold_baseline, to_wear_dataset)
from wearpg.segmentation import SEG_DEFAULTS, load_image, segment_image, segmentation_quality, to_gray
from wearpg.stats import cluster_bootstrap, holm, paired_cluster_bootstrap
from wearpg.utils import load_json, save_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real" / "zenodo_ferro" / "extracted"
OUT = ROOT / "outputs" / "open_data" / "ferro"
MIN_AREA = 30
# grids fixed before any test image was scored; ours: keys of segmentation.SEG_DEFAULTS (split_ratio stays at its
# default: it cannot change pixel IoU, the selection criterion). Round 2 was added (trainO only, before any test image
# was scored) because round 1 put both methods' optima on the grid edge (smooth_sigma 2; block 201 / offset 0.05).
OURS_GRID = [{"polarity": ["both", "dark"], "smooth_sigma": [0.5, 1.0, 2.0], "noise_k": [2.0, 3.0, 6.0],
              "exclude_border": [True, False]},
             {"polarity": ["both"], "smooth_sigma": [3.0, 4.0], "noise_k": [2.0, 3.0, 6.0], "exclude_border": [False]}]
LOCAL_GRID = [{"block_size": [25, 51, 101, 201], "offset": [0.0, 0.02, 0.05]},
              {"block_size": [401], "offset": [0.0, 0.02, 0.05, 0.08]},
              {"block_size": [25, 51, 101, 201], "offset": [0.08]}]
LOCAL_DEFAULT = {"block_size": 51, "offset": 0.02}
SELECT_BY = "iou"
TUNE_IMAGES = 120  # shortcut: grid on a random trainO subset (shared CPU); all trainO + testO for the chosen configs


def _grid(rounds: list[dict]) -> list[dict]:
    return [dict(zip(g, v)) for g in rounds for v in itertools.product(*g.values())]


def predict(img, method: str, params: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """(bool foreground mask, object contours) of one method on one image."""
    if method == "ours":
        cs = [s["contour"] for s in segment_image(img, params)]
        return contours_mask(cs, img.shape), cs
    m = threshold_baseline(img, method, min_area_px=MIN_AREA, **params)
    return m, [c["contour"] for c in mask_components(m, MIN_AREA)]


def score(job: tuple) -> dict:
    """Pixel + object metrics of one (image, method, params) job against the GT mask (tag "tune_*": pixel only)."""
    image_id, image_path, mask_path, method, params, tag = job
    img, gt = load_image(image_path), read_mask(mask_path)
    pred, pc = predict(img, method, params)
    if tag.startswith("tune_"):
        return {"image_id": image_id, "method": tag, **pixel_scores(pred, gt)}
    gcomp = mask_components(gt, MIN_AREA)
    q = segmentation_quality(pc, [c["contour"] for c in gcomp], gt.shape)
    n_gt = q["n_gt"]
    return {"image_id": image_id, "method": tag, **pixel_scores(pred, gt), "precision": q["precision"],
            "recall": q["recall"], "f1": q["f1"], "mean_iou_matched": q["mean_iou"], "n_pred": q["n_pred"],
            "n_gt": n_gt, "n_matched": q["n_matched"], "n_gt_border": sum(c["touches_border"] for c in gcomp),
            "count_rel_err": (q["n_pred"] - n_gt) / n_gt if n_gt else np.nan}


def run_jobs(jobs: list[tuple], pool: ProcessPoolExecutor) -> pd.DataFrame:
    return pd.DataFrame(list(pool.map(score, jobs, chunksize=4)))


def jobs_for(idx: pd.DataFrame, method: str, params: dict, tag: str) -> list[tuple]:
    return [(r.image_id, r.image_path, r.mask_path, method, params, tag) for r in idx.itertuples(index=False)]


# ----------------------------------------------------------------------------- stages


def stage_convert(idx: pd.DataFrame) -> None:
    ds = to_wear_dataset(idx, DATA, MIN_AREA)
    errs = ds.validate()
    p = ds.particles.merge(ds.images[["image_id", "split"]], on="image_id")
    per_img = p.groupby("image_id").size().reindex(ds.images.image_id, fill_value=0)
    orphans = sorted({m.stem for m in (DATA / "dataset" / "testO" / "labels").glob("*.png")}
                     - set(idx.loc[idx.split == "testO", "src_id"]))
    save_json({"source": ZENODO_FERRO, "validate_errors": errs, "describe": ds.describe(),
               "n_images": idx.split.value_counts().to_dict(), "n_particles": p.split.value_counts().to_dict(),
               "n_particles_touching_border": int(p.touches_border.sum()),
               "particles_per_image_quantiles": per_img.quantile([0, .25, .5, .75, 1]).to_dict(),
               "particle_area_px_quantiles": p.area_px.quantile([0, .1, .25, .5, .75, .9, 1]).to_dict(),
               "min_area_px": MIN_AREA, "orphan_testO_masks_skipped": orphans,
               "meta": {k: v for k, v in ds.meta.items() if k != "source"}}, OUT / "dataset_summary.json")
    print(f"convert: {len(ds.images)} images, {len(ds.particles)} particles, validate errors: {len(errs)}")
    if errs:
        raise SystemExit("\n".join(errs))


def stage_tune(idx: pd.DataFrame, pool: ProcessPoolExecutor) -> None:
    train = idx[idx.split == "trainO"].sample(TUNE_IMAGES, random_state=0)
    configs = [(m, k, p) for m, g in (("ours", OURS_GRID), ("local", LOCAL_GRID)) for k, p in enumerate(_grid(g))]
    jobs = [j for m, k, p in configs
            for j in jobs_for(train, m, {**SEG_DEFAULTS, **p} if m == "ours" else p, f"tune_{m}_{k}")]
    res = run_jobs(jobs, pool)
    rows = []
    for m, k, p in configs:
        df = res[res.method == f"tune_{m}_{k}"]
        rows.append({"method": m, "config": k, **{f"p_{a}": b for a, b in p.items()},
                     **df[["iou", "dice"]].mean().to_dict(), "mean_abs_area_rel_err": df.area_rel_err.abs().mean(),
                     "mean_area_rel_err": df.area_rel_err.mean(), "n_images": len(df)})
        print(f"tune {m} {k} {p} iou={rows[-1]['iou']:.4f}", flush=True)
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(OUT / "seg_tuning_grid.csv", index=False)
    best = {}
    for method, g in grid_df.groupby("method", sort=False):
        b = g.loc[g[SELECT_BY].idxmax()]  # first maximum in grid order
        best[method] = {c[2:]: (b[c].item() if hasattr(b[c], "item") else b[c])
                        for c in g.columns if c.startswith("p_") and pd.notna(b[c])}
        best[method]["train_" + SELECT_BY] = float(b[SELECT_BY])
    save_json({"selected_by": f"mean per-image pixel {SELECT_BY} on {len(train)} random trainO images (seed 0)",
               "tune_image_ids": train.image_id.tolist(),
               "grids": {"ours": OURS_GRID, "local": LOCAL_GRID}, "ours_defaults": SEG_DEFAULTS,
               "local_defaults": LOCAL_DEFAULT, "best": best}, OUT / "seg_tuned_params.json")


def methods(tuned: dict) -> dict[str, tuple[str, dict]]:
    """tag -> (method, params) of the five compared methods."""
    ours_t = {k: v for k, v in tuned["best"]["ours"].items() if not k.startswith("train_")}
    loc_t = {k: v for k, v in tuned["best"]["local"].items() if not k.startswith("train_")}
    loc_t["block_size"] = int(loc_t["block_size"])
    return {"ours_default": ("ours", dict(SEG_DEFAULTS)), "ours_tuned": ("ours", {**SEG_DEFAULTS, **ours_t}),
            "otsu": ("otsu", {}), "local_default": ("local", dict(LOCAL_DEFAULT)), "local_tuned": ("local", loc_t)}


def summary_metrics(df: pd.DataFrame) -> dict:
    """Image-level means + across-image agreement of the debris amount (foreground area, object count)."""
    lc = volume_consistency(np.log(df.area_pred.clip(lower=1)), np.log(df.area_gt))
    out = {m: float(df[m].mean()) for m in ("iou", "dice", "precision", "recall", "f1", "area_rel_err",
                                             "count_rel_err")}
    out.update({"abs_area_rel_err": float(df.area_rel_err.abs().mean()),
                "median_abs_area_rel_err": float(df.area_rel_err.abs().median()),
                "abs_count_rel_err": float(df.count_rel_err.abs().mean()),
                "spearman_area": float(sps.spearmanr(df.area_pred, df.area_gt)[0]),
                "spearman_count": float(sps.spearmanr(df.n_pred, df.n_gt)[0]),
                "ccc_log_area": lc["ccc"], "slope_log_area": lc["slope"]})
    return out


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_test(idx: pd.DataFrame, pool: ProcessPoolExecutor, force: bool) -> None:
    """Scores trainO + testO once; the summary is always (re)computed from the frozen per-image CSV, so --force is
    needed only to re-score the images (reproduction, never tuning)."""
    out_csv = OUT / "seg_per_image_test.csv"
    tuned = load_json(OUT / "seg_tuned_params.json")
    ms = methods(tuned)
    if force or not out_csv.exists():
        for split, path in (("trainO", OUT / "seg_per_image_train.csv"), ("testO", out_csv)):
            sub = idx[idx.split == split]
            df = run_jobs([j for tag, (m, p) in ms.items() for j in jobs_for(sub, m, p, tag)], pool)
            df.insert(1, "split", split)
            df.to_csv(path, index=False)
        save_json({f: _sha256(ROOT / f) for f in ("src/wearpg/segmentation.py", "src/wearpg/open_data.py",
                                                  "scripts/open_ferro_eval.py")}, OUT / "seg_test_provenance.json")
    else:
        print(f"{out_csv} exists: testO already scored once; recomputing the summary from it")
    test = pd.read_csv(out_csv)
    summ = {"note": "testO evaluated once; image = bootstrap cluster; 95% percentile CIs, 2000 replicates",
            "n_images": int(test.image_id.nunique()), "methods": {t: [m, p] for t, (m, p) in ms.items()},
            "provenance_sha256": load_json(OUT / "seg_test_provenance.json")}
    for tag, g in test.groupby("method", sort=False):
        summ[tag] = cluster_bootstrap(summary_metrics, g, "image_id", n_boot=2000, seed=0)
        print(f"test {tag}: iou {summ[tag]['iou']['point']:.3f} f1 {summ[tag]['f1']['point']:.3f}", flush=True)
    pairs = [("ours_tuned", "ours_default"), ("ours_tuned", "otsu"), ("ours_tuned", "local_tuned"),
             ("ours_tuned", "local_default"), ("ours_default", "otsu"), ("ours_default", "local_default")]
    rows = []
    for a, b in pairs:
        r = paired_cluster_bootstrap(
            lambda d: {"iou": d.iou.mean(), "f1": d.f1.mean(), "abs_area_rel_err": d.area_rel_err.abs().mean(),
                       "abs_count_rel_err": d.count_rel_err.abs().mean()},
            test[test.method == a], test[test.method == b], "image_id", "image_id", n_boot=2000, seed=0)
        rows += [{"a": a, "b": b, "metric": m, **r[m]} for m in r if m != "n_clusters"]
    pr = pd.DataFrame(rows)
    pr["p_holm"] = pr.groupby("metric").p_value.transform(lambda p: holm(p.to_numpy()))
    pr.to_csv(OUT / "seg_paired_test.csv", index=False)
    save_json(summ, OUT / "seg_summary_test.json")


NEAR_DUP = {"thumb_corr": 0.95, "mask_iou": 0.5}  # fixed before the check was run


def _thumb(path, n: int = 32) -> np.ndarray:
    """z-scored n x n block-mean grey thumbnail."""
    from skimage.transform import resize
    t = resize(to_gray(load_image(path)), (n, n), order=1, anti_aliasing=True)
    return (t - t.mean()) / (t.std() + 1e-9)


def _d8(a: np.ndarray) -> list[np.ndarray]:
    """The 8 rotations / flips of a square array."""
    return [np.rot90(a, k) for k in range(4)] + [np.fliplr(np.rot90(a, k)) for k in range(4)]


def stage_leak(idx: pd.DataFrame) -> None:
    """Train/test overlap: exact byte duplicates and, per testO image, the most similar trainO image over the 8
    rotations / flips (Pearson r of 32 x 32 grey thumbnails) with the GT-mask IoU under the same transform.
    Near duplicate = r >= 0.95 and mask IoU >= 0.5. If testO was scored, metrics without the near duplicates."""
    import hashlib
    tr, te = idx[idx.split == "trainO"].reset_index(drop=True), idx[idx.split == "testO"]
    sha = {r.image_id: hashlib.sha1(Path(r.image_path).read_bytes()).hexdigest() for r in idx.itertuples()}
    tr_sha = {sha[i] for i in tr.image_id}
    th = {r.image_id: _thumb(r.image_path) for r in idx.itertuples()}
    bank = np.stack([[d.ravel() for d in _d8(th[i])] for i in tr.image_id])  # (n_train, 8, n*n)
    rows = []
    for r in te.itertuples():
        c = bank @ th[r.image_id].ravel() / th[r.image_id].size
        j, k = np.unravel_index(np.argmax(c), c.shape)
        m_te, m_tr = read_mask(r.mask_path), _d8(read_mask(tr.mask_path[j]))[k]
        iou = (m_te & m_tr).sum() / max((m_te | m_tr).sum(), 1) if m_te.shape == m_tr.shape else np.nan
        rows.append({"test_image": r.image_id, "best_train_image": tr.image_id[j], "dihedral": int(k),
                     "thumb_corr": float(c[j, k]), "mask_iou": float(iou), "exact_dup": sha[r.image_id] in tr_sha})
    df = pd.DataFrame(rows)
    df["near_dup"] = df.exact_dup | ((df.thumb_corr >= NEAR_DUP["thumb_corr"]) & (df.mask_iou >= NEAR_DUP["mask_iou"]))
    df.to_csv(OUT / "seg_leak_check.csv", index=False)
    out = {"rule": NEAR_DUP, "n_test": len(df), "n_exact_dup": int(df.exact_dup.sum()),
           "n_near_dup": int(df.near_dup.sum()), "n_corr_ge_0.9": int((df.thumb_corr >= 0.9).sum()),
           "thumb_corr_quantiles": df.thumb_corr.quantile([0, .5, .9, 1]).to_dict(),
           "n_train_exact_dup_pairs": int(len(tr) - len({sha[i] for i in tr.image_id}))}
    per = OUT / "seg_per_image_test.csv"
    if per.exists():
        t = pd.read_csv(per).merge(df[["test_image", "near_dup"]], left_on="image_id", right_on="test_image")
        out["test_metrics_by_near_dup"] = (t.assign(abs_area_rel_err=t.area_rel_err.abs())
                                           .groupby(["method", "near_dup"])
                                           .agg(n=("iou", "size"), iou=("iou", "mean"), f1=("f1", "mean"),
                                                abs_area_rel_err=("abs_area_rel_err", "mean"))
                                           .round(4).reset_index().to_dict(orient="records"))
    save_json(out, OUT / "seg_leak_summary.json")
    print(f"leak: {out['n_exact_dup']} exact, {out['n_near_dup']} near-duplicate testO images of {len(df)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Zenodo ferrography: conversion + segmentation benchmark")
    ap.add_argument("--stage", choices=["convert", "tune", "test", "leak", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_zenodo_ferro(DATA)
    if a.stage in ("convert", "all"):
        stage_convert(idx)
    with ProcessPoolExecutor(a.workers) as pool:
        if a.stage in ("tune", "all") and (a.force or not (OUT / "seg_tuned_params.json").exists()):
            stage_tune(idx, pool)
        if a.stage in ("test", "all"):
            stage_test(idx, pool, a.force)
    if a.stage in ("leak", "all"):
        stage_leak(idx)


if __name__ == "__main__":
    main()
