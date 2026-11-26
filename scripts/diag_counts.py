"""F3 POST-HOC DIAGNOSTIC (DIAG; rules fixed beforehand in outputs/agent_logs/DIAG.md): is the object-count /
object-F1 deficit of every segmenter on the Zenodo ferrograms (doi 10.5281/zenodo.22997504, CC-BY-4.0)
definitional? The GT often marks a whole chain or agglomerate as one object. Predicted fragments separated by a
small gap are merged (merge radius rho chosen on trainO ONLY so that object counts match the GT), and the object
metrics are recomputed on testO, stratified by GT object type (chain-like vs single-like).

SECOND LOOK AT testO: the frozen benchmark scored testO once; this diagnostic scores it again with the frozen
objects (level "none" must reproduce the frozen per-image numbers) and with merged objects. Merging leaves every
pixel metric unchanged (IoU, debris area, CCC 0.890 of the observation model's area input).

Stages: score (per image x method x merge level; cached) -> stats.  Outputs: outputs/diagnostics/f3_*.
Run: .venv/Scripts/python.exe -I scripts/diag_counts.py [--workers 4]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats as sps

from diag_common import FERRO, NOTE, OUT, ZEN, load_script
from wearpg.features.shape import shape_descriptors
from wearpg.metrics import volume_consistency
from wearpg.open_data import load_zenodo_ferro, mask_components, read_mask
from wearpg.segmentation import _label_image, _match, _prf, load_image
from wearpg.stats import cluster_bootstrap, holm, paired_cluster_bootstrap
from wearpg.utils import load_json, save_json

ev = load_script("open_ferro_eval")  # frozen predict() / methods()
LEVELS = ("none", 0, 1, 2, 3, 4, 6, 8, 12, 16)  # merge radius rho [px]; "none" = frozen objects
METHODS = ("ours_tuned", "otsu", "local_tuned")
MIN_AREA = 30
B = 2000


def contingency_iou(plab: np.ndarray, n_p: int, glab: np.ndarray, n_g: int) -> np.ndarray:
    """(n_p, n_g) pixel IoU of two label images (labels 1..n), as segmentation._iou_matrix."""
    key = plab.astype(np.int64) * (n_g + 1) + glab
    cont = np.bincount(key.ravel(), minlength=(n_p + 1) * (n_g + 1)).reshape(n_p + 1, n_g + 1)
    inter = cont[1:, 1:]
    union = cont[1:].sum(axis=1)[:, None] + cont[:, 1:].sum(axis=0)[None, :] - inter
    return inter / np.maximum(union, 1)


def merge_labels(plab: np.ndarray, edt: np.ndarray, rho: int) -> tuple[np.ndarray, int]:
    """Merged objects: fragments whose pixels fall in one 4-connected component of the predicted foreground dilated
    by a Euclidean disk of radius rho (edt = distance of each pixel to the nearest foreground pixel). Object pixels
    are the fragments' own pixels. Returns (consecutive label image, number of objects)."""
    fg = plab > 0
    comp, _ = ndi.label(edt <= rho if rho > 0 else fg)
    lab = np.where(fg, comp, 0)
    u, inv = np.unique(lab, return_inverse=True)
    if u[0] != 0:  # no background pixel at all
        inv = inv + 1
    return inv.reshape(lab.shape).astype(np.int32), int((u > 0).sum())


def score_image(job: tuple) -> tuple[list[dict], list[dict], list[dict]]:
    """Per-image rows (method x level), per-GT-object descriptor rows and per-GT match rows."""
    image_id, split, image_path, mask_path, ms = job
    img, gt = load_image(image_path), read_mask(mask_path)
    gcomp = mask_components(gt, MIN_AREA)
    gc = [c["contour"] for c in gcomp]
    glab, n_g = _label_image(gc, gt.shape), len(gc)
    gt_rows = []
    for k, c in enumerate(gcomp):
        d = shape_descriptors(c["contour"], 1.0)
        gt_rows.append({"image_id": image_id, "split": split, "gt_k": k, "area_px": c["area_px"],
                        "touches_border": c["touches_border"], "aspect_ratio": d["shape_aspect_ratio"],
                        "solidity": d["shape_solidity"]})
    rows, match_rows = [], []
    for tag, (method, params) in ms.items():
        _, pc = ev.predict(img, method, params)
        plab0 = _label_image(pc, gt.shape)
        edt = ndi.distance_transform_edt(plab0 == 0)
        for lev in LEVELS:
            plab, n_p = (plab0, len(pc)) if lev == "none" else merge_labels(plab0, edt, lev)
            iou = contingency_iou(plab, n_p, glab, n_g)
            r, c = _match(iou, 0.5)
            q = _prf(len(r), n_p, n_g, float(iou[r, c].sum()))
            rows.append({"image_id": image_id, "split": split, "method": tag, "level": str(lev), **q})
            matched = np.zeros(n_g, bool)
            matched[c] = True
            match_rows += [{"image_id": image_id, "method": tag, "level": str(lev), "gt_k": k, "matched": bool(m)}
                           for k, m in enumerate(matched)]
    return rows, gt_rows, match_rows


def stage_score(workers: int) -> None:
    idx = load_zenodo_ferro(ZEN)
    ms = {k: v for k, v in ev.methods(load_json(FERRO / "seg_tuned_params.json")).items() if k in METHODS}
    jobs = [(r.image_id, r.split, r.image_path, r.mask_path, ms) for r in idx.itertuples(index=False)]
    rows, gts, matches = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for n, (a, b, c) in enumerate(ex.map(score_image, jobs, chunksize=2), start=1):
            rows += a
            gts += b
            matches += c
            if n % 50 == 0:
                print(f"scored {n}/{len(jobs)} images", flush=True)
    pd.DataFrame(gts).to_csv(OUT / "f3_gt_objects.csv.gz", index=False)
    pd.DataFrame(matches).to_csv(OUT / "f3_gt_matches.csv.gz", index=False)
    pd.DataFrame(rows).to_csv(OUT / "f3_per_image_levels.csv.gz", index=False)  # written last = completion flag


# ----------------------------------------------------------------------------- statistics


def count_summary(d: pd.DataFrame) -> dict:
    ratio = d.n_pred / d.n_gt.where(d.n_gt > 0)
    return {"pooled_count_ratio": d.n_pred.sum() / d.n_gt.sum(), "median_count_ratio": float(ratio.median()),
            "mean_abs_count_rel_err": float((ratio - 1).abs().mean()),
            "spearman_count": float(sps.spearmanr(d.n_pred, d.n_gt)[0]),
            "ccc_log1p_count": volume_consistency(np.log1p(d.n_pred), np.log1p(d.n_gt))["ccc"],
            "f1": d.f1.mean(), "precision": d.precision.mean(), "recall": d.recall.mean()}


def gt_type(g: pd.DataFrame) -> pd.Series:
    """single-like aspect < 2, intermediate 2-3, chain-like >= 3 (rule fixed in DIAG.md)."""
    return pd.cut(g.aspect_ratio, [0, 2, 3, np.inf], right=False, labels=["single_like", "intermediate",
                                                                           "chain_like"]).astype(str)


def recall_by(col: str):
    def fn(d: pd.DataFrame) -> dict:
        return {f"recall_{k}": float(v) for k, v in d.groupby(col).matched.mean().items()}
    return fn


def single_minus_chain(d: pd.DataFrame) -> float:
    r = d.groupby("type").matched.mean()
    return float(r.get("single_like", np.nan) - r.get("chain_like", np.nan))


def stage_stats() -> None:
    per = pd.read_csv(OUT / "f3_per_image_levels.csv.gz")
    gto = pd.read_csv(OUT / "f3_gt_objects.csv.gz")
    gto["type"] = gt_type(gto)
    gto["compact"] = np.where(gto.solidity >= 0.9, "compact", "non_compact")
    mt = pd.read_csv(OUT / "f3_gt_matches.csv.gz").merge(gto, on=["image_id", "gt_k"])
    res: dict = {"note": NOTE, "status": "post_hoc_diagnostic", "testO": "SECOND LOOK at testO (frozen benchmark "
                 "scored it once); level 'none' reproduces the frozen objects", "levels": list(map(str, LEVELS))}

    # reproduction of the frozen per-image object metrics at level "none"
    frozen = pd.concat([pd.read_csv(FERRO / f) for f in ("seg_per_image_train.csv", "seg_per_image_test.csv")])
    chk = per[per.level == "none"].merge(frozen, on=["image_id", "method"], suffixes=("", "_frozen"))
    res["reproduction_level_none"] = {
        "n_rows": len(chk), "n_pred_equal": int((chk.n_pred == chk.n_pred_frozen).sum()),
        "n_matched_equal": int((chk.n_matched == chk.n_matched_frozen).sum()),
        "max_abs_f1_diff": float((chk.f1 - chk.f1_frozen).abs().max())}

    # level selection on trainO only: pooled count ratio closest to 1 (ties -> less merging, i.e. LEVELS order)
    tr = per[per.split == "trainO"]
    sel_rows, chosen = [], {}
    for m in METHODS:
        for lev in map(str, LEVELS):
            g = tr[(tr.method == m) & (tr.level == lev)]
            sel_rows.append({"method": m, "level": lev, "train_pooled_count_ratio": g.n_pred.sum() / g.n_gt.sum(),
                             "train_median_count_ratio": float((g.n_pred / g.n_gt).median()),
                             "train_f1": g.f1.mean()})
        s = pd.DataFrame([r for r in sel_rows if r["method"] == m])
        chosen[m] = s.level.iloc[int(np.argmin(np.abs(np.log(s.train_pooled_count_ratio.to_numpy()))))]
    pd.DataFrame(sel_rows).to_csv(OUT / "f3_trainO_level_selection.csv", index=False)
    res["chosen_level_trainO"] = chosen

    # testO summaries at none and at the chosen level, with paired merged - none differences
    te = per[per.split == "testO"]
    summ_rows = []
    for m in METHODS:
        for lev in ("none", chosen[m]):
            bs = cluster_bootstrap(count_summary, te[(te.method == m) & (te.level == lev)], "image_id", n_boot=B,
                                   seed=0)
            summ_rows += [{"method": m, "level": lev, "chosen": lev == chosen[m], "metric": k, **v}
                          for k, v in bs.items() if k != "n_clusters"]
        pb = paired_cluster_bootstrap(count_summary, te[(te.method == m) & (te.level == chosen[m])],
                                      te[(te.method == m) & (te.level == "none")], "image_id", "image_id",
                                      n_boot=B, seed=0)
        summ_rows += [{"method": m, "level": f"{chosen[m]}_minus_none", "metric": k, **v}
                      for k, v in pb.items() if k != "n_clusters"]
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(OUT / "f3_testO_counts.csv", index=False)

    # after-merge object F1: ours vs each merged baseline (paired image bootstrap, Holm over 2)
    cmp_rows = []
    a = te[(te.method == "ours_tuned") & (te.level == chosen["ours_tuned"])]
    for bname in ("otsu", "local_tuned"):
        b = te[(te.method == bname) & (te.level == chosen[bname])]
        r = paired_cluster_bootstrap(lambda d: d.f1.mean(), a, b, "image_id", "image_id", n_boot=B, seed=0)["value"]
        cmp_rows.append({"a": f"ours_tuned@{chosen['ours_tuned']}", "b": f"{bname}@{chosen[bname]}", "metric": "f1",
                         **r})
    cmp = pd.DataFrame(cmp_rows)
    cmp["p_holm"] = holm(cmp.p_value.to_numpy())
    cmp.to_csv(OUT / "f3_testO_merged_vs_baselines.csv", index=False)

    # recall by GT type (testO), level none and chosen level, every method
    rec_rows = []
    mte = mt[mt.split == "testO"]
    for m in METHODS:
        for lev in ("none", chosen[m]):
            d = mte[(mte.method == m) & (mte.level == lev)]
            for col in ("type", "compact"):
                bs = cluster_bootstrap(recall_by(col), d, "image_id", n_boot=B, seed=0)
                n = d.groupby(col).size()
                rec_rows += [{"method": m, "level": lev, "stratify": col, "stratum": k[7:], "n_gt": int(n[k[7:]]),
                              **v} for k, v in bs.items() if k != "n_clusters"]
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(OUT / "f3_testO_recall_by_gt_type.csv", index=False)
    d_none = mte[(mte.method == "ours_tuned") & (mte.level == "none")]
    dt = cluster_bootstrap(single_minus_chain, d_none, "image_id", n_boot=B, seed=0)["value"]
    d_ch = mte[(mte.method == "ours_tuned") & (mte.level == chosen["ours_tuned"])]
    dt_merged = cluster_bootstrap(single_minus_chain, d_ch, "image_id", n_boot=B, seed=0)["value"]
    res["type_counts_testO"] = gto[gto.split == "testO"].type.value_counts().to_dict()
    res["recall_single_minus_chain_ours_none"] = dt
    res["recall_single_minus_chain_ours_merged"] = dt_merged

    # decision rules (DIAG.md)
    o = summ[(summ.method == "ours_tuned")]
    ratio_m = o[(o.level == chosen["ours_tuned"]) & (o.metric == "pooled_count_ratio")].iloc[0]
    df1 = o[(o.level == f"{chosen['ours_tuned']}_minus_none") & (o.metric == "f1")].iloc[0]
    rules = {"D_count": bool(0.8 <= ratio_m["point"] <= 1.25), "D_F1": bool(df1["lo"] > 0), "D_type": bool(dt["lo"] > 0)}
    res["rules"] = rules
    res["diagnosis"] = ("deficit definitional" if all(rules.values()) else
                        "partly definitional" if rules["D_type"] else "not shown")
    still_lower = bool((cmp["diff"] < 0).any())
    res["after_merge_vs_baselines"] = cmp.to_dict("records")
    res["object_level_verdict"] = ("against_us (frozen, unchanged); after merging ours is still below a baseline"
                                   if still_lower else
                                   "against_us (frozen, unchanged); after merging ours is not below the baselines")
    save_json(res, OUT / "f3_summary.json")
    pd.set_option("display.width", 200)
    print(summ[summ.metric.isin(["pooled_count_ratio", "f1"])].round(3).to_string())
    print(rec[rec.method == "ours_tuned"].round(3).to_string())
    print({k: res[k] for k in ("reproduction_level_none", "chosen_level_trainO", "rules", "diagnosis")})


def main() -> None:
    ap = argparse.ArgumentParser(description="F3 chain-merge object-count diagnostic (post hoc)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", choices=["score", "stats", "all"], default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("score", "all") and not (OUT / "f3_per_image_levels.csv.gz").exists():
        stage_score(a.workers)
    if a.stage in ("stats", "all"):
        stage_stats()


if __name__ == "__main__":
    main()
