"""F4 POST-HOC DIAGNOSTIC (DIAG; rules fixed beforehand in outputs/agent_logs/DIAG.md): end-to-end magnification
robustness of contour-graph geometry on REAL ferrogram particles (Zenodo doi 10.5281/zenodo.22997504), stratified
by particle size at the reduced scale, plus a scale-adapted segmentation (smooth_sigma = 2.0 * s, every other
setting frozen; rule fixed before running) and a minimum-resolution rule (Feret px) DERIVED ON trainO particles and
VALIDATED on testO particles.

Particles: every non-border GT component >= 120 px (the frozen selection rule of open_ferro_robust.py, all images).
The frozen transform(), _match(), _largest(), features() and stability() of scripts/open_ferro_robust.py are reused
unchanged. Modes: mask (contour from the rescaled GT mask: the resolution floor without segmentation error),
pipeline_frozen (trainO-tuned cfg at every scale), pipeline_adapted (sigma 2.0 * s). Identity reference = frozen cfg.

Stages: run (per-image cache in outputs/diagnostics/f4_cache/, resumable) -> stats.  Outputs: outputs/diagnostics/f4_*.
Run: .venv/Scripts/python.exe -I scripts/diag_magnification.py [--workers 4]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir

import numpy as np
import pandas as pd

from diag_common import FERRO, NOTE, OUT, ZEN, group_features, group_spearman, load_script, wide_pairs
from wearpg.features.shape import shape_descriptors
from wearpg.open_data import contours_mask, load_zenodo_ferro, mask_components, read_mask
from wearpg.segmentation import load_image, segment_image
from wearpg.stats import cluster_bootstrap, paired_cluster_bootstrap
from wearpg.utils import load_json, save_json

rob = load_script("open_ferro_robust")
SCALES = (0.5, 0.25)
MODES = ("mask", "pipeline_frozen", "pipeline_adapted")
GRID = (8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128)  # Feret_reduced thresholds [px]
BINS = (0, 16, 32, 64, np.inf)
MIN_N_RULE, MIN_N_CI, TARGET, FRACTAL_FLOOR_PX = 20, 10, 0.8, 40
CACHE = OUT / "f4_cache"
B = 2000


def adapted_cfg(cfg: dict, s: float) -> dict:
    """Scale-adapted segmentation (rule fixed before running): smoothing sigma proportional to scale."""
    return {**cfg, "smooth_sigma": float(cfg["smooth_sigma"]) * s}


def run_image(job: tuple) -> pd.DataFrame:
    image_id, image_path, mask_path, labels, cfg = job
    img = load_image(image_path).astype(float) / 255
    comps = mask_components(read_mask(mask_path), rob.MIN_AREA)
    masks = [contours_mask([comps[k]["contour"]], img.shape) for k in labels]
    feret = {k: shape_descriptors(comps[k]["contour"], 1.0)["shape_feret_max_um"] for k in labels}  # native px
    rows = []

    def add(mode: str, t: str, k: int, xy, im, s: float) -> dict:
        base = {"image_id": image_id, "particle": f"{image_id}_p{k:03d}", "area_px": comps[k]["area_px"],
                "feret_native_px": feret[k], "mode": mode, "transform": t}
        row = {**base, "lost": True} if xy is None or len(xy) < 8 else {**base, "lost": False, **rob.features(xy, im, s)}
        rows.append(row)
        return row

    segs = segment_image(img, cfg)
    for k, m in zip(labels, masks):
        add("mask", "identity", k, rob._largest(m), img, 1.0)
        r = add("pipeline_frozen", "identity", k, rob._match(segs, m), img, 1.0)
        rows.append({**r, "mode": "pipeline_adapted"})  # sigma 2.0 * 1 = frozen cfg: identical reference
    for s in SCALES:
        t = f"scale{s:g}"
        im_t, m_t, _ = rob.transform(img, masks, t, 0.0, None)
        seg_f, seg_a = segment_image(im_t, cfg), segment_image(im_t, adapted_cfg(cfg, s))
        for k, mt in zip(labels, m_t):
            add("mask", t, k, rob._largest(mt), im_t, s)
            add("pipeline_frozen", t, k, rob._match(seg_f, mt), im_t, s)
            add("pipeline_adapted", t, k, rob._match(seg_a, mt), im_t, s)
    return pd.DataFrame(rows)


def stage_run(workers: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tuned = load_json(FERRO / "seg_tuned_params.json")["best"]["ours"]
    cfg = {k: v for k, v in tuned.items() if not k.startswith("train_")}
    idx = load_zenodo_ferro(ZEN)
    jobs = []
    for r in idx.itertuples(index=False):
        if (CACHE / f"{r.image_id}.csv.gz").exists():
            continue
        comps = mask_components(read_mask(r.mask_path), rob.MIN_AREA)
        labels = [k for k, c in enumerate(comps) if not c["touches_border"] and c["area_px"] >= rob.MIN_SEL_AREA]
        if labels:
            jobs.append((r.image_id, r.image_path, r.mask_path, labels, cfg))
    print(f"{len(jobs)} images to run ({sum(len(j[3]) for j in jobs)} particles)", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run_image, j): j[0] for j in jobs}
        for n, f in enumerate(as_completed(futs), start=1):
            tmp = CACHE / f"{futs[f]}.tmp.csv.gz"
            f.result().to_csv(tmp, index=False)
            os.replace(tmp, CACHE / f"{futs[f]}.csv.gz")  # atomic: a killed run leaves no partial file
            if n % 25 == 0:
                print(f"done {n}/{len(jobs)}", flush=True)


# ----------------------------------------------------------------------------- statistics


def spearman_ci(wide: pd.DataFrame) -> dict:
    if len(wide) < MIN_N_CI:
        return {"point": group_spearman(wide) if len(wide) >= 3 else np.nan, "lo": np.nan, "hi": np.nan}
    v = cluster_bootstrap(group_spearman, wide, "image_id", n_boot=B, seed=0)["value"]
    return {"point": v["point"], "lo": v["lo"], "hi": v["hi"]}


def f_min(sweep: pd.DataFrame) -> float:
    """Smallest F with S(F') >= TARGET for every F' >= F that has n >= MIN_N_RULE (NaN if none)."""
    ok = sweep[sweep.n >= MIN_N_RULE].sort_values("F")
    for i, F in enumerate(ok.F):
        if (ok.S.iloc[i:] >= TARGET).all():
            return float(F)
    return float("nan")


def stage_stats() -> None:
    files = sorted(CACHE.glob("*.csv.gz"))
    df = pd.concat([pd.read_csv(f) for f in files if ".tmp." not in f.name], ignore_index=True)
    df["split"] = df.image_id.str.split("_").str[0]
    geom = group_features(df, "graph_geom")
    res: dict = {"note": NOTE, "status": "post_hoc_diagnostic", "n_images": int(df.image_id.nunique()),
                 "n_particles": int(df.particle.nunique()),
                 "n_particles_by_split": df.drop_duplicates("particle").split.value_counts().to_dict(),
                 "graph_geom_features": geom, "adapted_rule": "smooth_sigma = 2.0 * s (frozen 2.0 at s = 1)"}

    # same-code check: frozen robust_particles (pipeline mode) vs our pipeline_frozen rows for shared particles
    old = pd.read_csv(FERRO / "robust_particles.csv.gz")
    old = old[(old["mode"] == "pipeline") & old["transform"].isin(["identity", "scale0.5", "scale0.25"]) & ~old.lost]
    new = df[(df["mode"] == "pipeline_frozen") & ~df.lost.astype(bool)]
    j = old.merge(new, on=["particle", "transform"], suffixes=("_old", "_new"))
    res["same_code_check"] = {"n_shared_rows": len(j), "max_abs_diff_graph_geom": float(np.nanmax(np.abs(
        j[[f"{c}_old" for c in geom]].to_numpy(float) - j[[f"{c}_new" for c in geom]].to_numpy(float))))
        if len(j) else None}

    # statistic identity with the frozen stability() on the full set
    full = {}
    for mode in MODES:
        st = rob.stability(df[df["mode"] == mode].assign(mode="x"))
        for s in SCALES:
            t = f"scale{s:g}"
            frozen_val = float(st[(st["transform"] == t) & (st.group == "graph_geom")].spearman.median())
            ours = group_spearman(wide_pairs(df, mode, t, geom))
            full[f"{mode}|{t}"] = {"stability_graph_geom": frozen_val, "group_spearman": ours}
            assert abs(frozen_val - ours) < 1e-12, (mode, t, frozen_val, ours)
    res["full_set_graph_geom_spearman"] = full

    # size-stratified table (all splits), loss rate by bin
    feret = df.drop_duplicates("particle").set_index("particle").feret_native_px
    rows = []
    for mode in MODES:
        g = df[df["mode"] == mode]
        ref_ok = set(g.particle[(g["transform"] == "identity") & ~g.lost.astype(bool)])
        for s in SCALES:
            t = f"scale{s:g}"
            fr = feret * s
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                in_bin = set(fr.index[(fr >= lo) & (fr < hi)]) & ref_ok
                w = wide_pairs(df, mode, t, geom, particles=sorted(in_bin))
                rows.append({"mode": mode, "scale": s, "feret_reduced_lo": lo, "feret_reduced_hi": hi,
                             "n_ref": len(in_bin), "n_matched": len(w),
                             "loss_rate": 1 - len(w) / len(in_bin) if in_bin else np.nan, **spearman_ci(w)})
    pd.DataFrame(rows).to_csv(OUT / "f4_size_strata.csv", index=False)

    # minimum-size rule: derived on trainO particles, validated on testO particles
    sweeps, fmins = [], {}
    for mode in ("pipeline_frozen", "pipeline_adapted", "mask"):
        for split in ("trainO", "testO"):
            for s in SCALES:
                t = f"scale{s:g}"
                fr = feret * s
                for F in GRID:
                    ids = sorted(fr.index[(fr >= F) & fr.index.str.startswith(split)])
                    w = wide_pairs(df, mode, t, geom, particles=ids)
                    sweeps.append({"mode": mode, "split": split, "scale": s, "F": F, "n": len(w),
                                   "S": group_spearman(w) if len(w) >= 3 else np.nan})
    sw = pd.DataFrame(sweeps)
    sw.to_csv(OUT / "f4_min_size_sweep.csv", index=False)
    for mode in ("pipeline_frozen", "pipeline_adapted", "mask"):
        per_s = {s: f_min(sw[(sw["mode"] == mode) & (sw.split == "trainO") & (sw.scale == s)]) for s in SCALES}
        fmins[mode] = {"F_min_by_scale_trainO": per_s, "F_min_trainO": float(max(per_s.values()))
                       if all(np.isfinite(v) for v in per_s.values()) else float("nan")}
    a, f = fmins["pipeline_adapted"]["F_min_trainO"], fmins["pipeline_frozen"]["F_min_trainO"]
    rec = "pipeline_adapted" if np.isfinite(a) and (not np.isfinite(f) or a < f) else "pipeline_frozen"
    res["F_min"] = fmins
    res["recommended_variant"] = rec
    val = {}
    for mode in ("pipeline_frozen", "pipeline_adapted"):
        F = fmins[mode]["F_min_trainO"]
        if not np.isfinite(F):
            val[mode] = {"F_min": None, "validated": False, "reason": "no grid size reaches the target on trainO"}
            continue
        v = {}
        for s in SCALES:
            ids = sorted(feret.index[(feret * s >= F) & feret.index.str.startswith("testO")])
            w = wide_pairs(df, mode, f"scale{s:g}", geom, particles=ids)
            v[str(s)] = {"n": len(w), **spearman_ci(w)}
        val[mode] = {"F_min": F, "testO": v,
                     "validated": bool(all(np.isfinite(x["point"]) and x["point"] >= TARGET for x in v.values()))}
    res["validation_testO"] = val
    res["fractal_floor_feret_px"] = FRACTAL_FLOOR_PX

    # adapted vs frozen on particles matched by both at scale s (paired image bootstrap)
    cmp = {}
    for s in SCALES:
        t = f"scale{s:g}"
        wa, wf = wide_pairs(df, "pipeline_adapted", t, geom), wide_pairs(df, "pipeline_frozen", t, geom)
        common = sorted(set(wa.particle) & set(wf.particle))
        pb = paired_cluster_bootstrap(group_spearman, wa[wa.particle.isin(common)], wf[wf.particle.isin(common)],
                                      "image_id", "particle", n_boot=B, seed=0)["value"]
        cmp[t] = {"n_matched_adapted": len(wa), "n_matched_frozen": len(wf), "n_common": len(common),
                  "S_adapted_own_set": group_spearman(wa), "S_frozen_own_set": group_spearman(wf), **pb,
                  "adaptation_helps": bool(pb["lo"] > 0)}
    res["adapted_vs_frozen"] = cmp
    save_json(res, OUT / "f4_summary.json")
    print(pd.DataFrame(rows).round(3).to_string())
    print({k: res[k] for k in ("F_min", "recommended_variant", "validation_testO", "adapted_vs_frozen")})


def main() -> None:
    ap = argparse.ArgumentParser(description="F4 magnification diagnostic (post hoc)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", choices=["run", "stats", "all"], default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("run", "all"):
        stage_run(a.workers)
    if a.stage in ("stats", "all"):
        stage_stats()


if __name__ == "__main__":
    main()
