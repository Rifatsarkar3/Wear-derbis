"""Score a synthetic dataset against real ferrograms with the FROZEN scripts/open_ferro_morph.py (renderer realism).

The frozen module is imported unchanged; its own describe / morphology / domain_gap functions do all the scoring.
Only the synthetic dataset path and the output folder differ. Real rows (zenodo_gt, zenodo_seg, tcbgy_seg) come from
the frozen cache outputs/open_data/ferro/particle_descriptors.csv.gz (they do not depend on the synthetic data);
synthetic rows are recomputed with describe() in the frozen job order, so the frozen pairs equal a fresh run of the
frozen script with SYN pointed at --syn.

Added sets (pre-specified in outputs/agent_logs/RENDER.md, scoring functions unchanged):
  synthetic_chain     union of the view-0 GT contours of each FOV image (open_data.contours_mask) -> components with
                      open_data.mask_components (the zenodo_gt derivation), via describe(kind="gt")
  synthetic_chain_fe  the same, images of ferromagnetic-material samples only
  zenodo_gt_trainO / zenodo_gt_testO   split breakdown of zenodo_gt (trainO = tuning data, testO = held out)
SIMULATION: every synthetic number is simulation.

Run: python -I outputs/renderer/score_renderer.py --syn data/synthetic_smoke_v2 --tag smoke_v2
     [--tune] (trainO rows + primary pairs only)  [--check-frozen] (v1: must reproduce the frozen CSVs exactly)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "scripts" / "open_ferro_morph.py"
FROZEN_SHA = "1be182bb63b572d4c4e661d1feb5056ca611d5424640fbb4b8a455257a602630"
_spec = importlib.util.spec_from_file_location("open_ferro_morph", FROZEN)
M = importlib.util.module_from_spec(_spec)
sys.modules["open_ferro_morph"] = M  # worker processes unpickle M.describe by module name
_spec.loader.exec_module(M)

from wearpg.open_data import contours_mask  # noqa: E402
from wearpg.schema import WearDataset  # noqa: E402
from wearpg.utils import load_json  # noqa: E402

REAL = ("zenodo_gt", "zenodo_seg", "tcbgy_seg")
EXTRA_PAIRS = [("zenodo_gt", "synthetic_chain_fe"), ("zenodo_gt", "synthetic_chain"), ("tcbgy_seg", "synthetic_raster")]
SPLIT_PAIRS = [("zenodo_gt_trainO", "synthetic_raster"), ("zenodo_gt_testO", "synthetic_raster"),
               ("zenodo_gt_trainO", "synthetic_chain_fe"), ("zenodo_gt_testO", "synthetic_chain_fe")]
TUNE_PAIRS = [("zenodo_gt_trainO", "synthetic_raster"), ("zenodo_gt_trainO", "synthetic_chain_fe")]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def chain_jobs(syn: Path, mask_dir: Path) -> list[tuple]:
    """describe() jobs of the chain-level sets: one GT-style union mask PNG per FOV image (kind "gt")."""
    ds = WearDataset.load(syn)
    im = ds.images[ds.images.modality == "ferrograph_optical"]
    ferro = im.sample_id.map(ds.samples.set_index("sample_id").run_id).map(
        ds.runs.set_index("run_id").specimen_id).map(ds.specimens.set_index("specimen_id").ferromagnetic)
    v = ds.views[ds.views.view_idx == 0]
    by_img = v.groupby("image_id")
    mask_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for r, fe in zip(im.itertuples(), ferro):
        if r.image_id not in by_img.groups:
            continue
        g = by_img.get_group(r.image_id)
        m = contours_mask([WearDataset.contour(x) for x in g.itertuples()], (int(r.height_px), int(r.width_px)))
        mp = mask_dir / f"{r.image_id}.png"
        Image.fromarray((m * 255).astype(np.uint8)).save(mp)
        out.append(("synthetic_chain", r.image_id, ds.image_path(r.image_id), "gt", mp))
        if bool(fe):
            out.append(("synthetic_chain_fe", r.image_id, ds.image_path(r.image_id), "gt", mp))
    return out


def split_rows(df: pd.DataFrame) -> pd.DataFrame:
    z = df[df.set == "zenodo_gt"].copy()
    z["set"] = np.where(z.image_id.str.startswith("trainO_"), "zenodo_gt_trainO", "zenodo_gt_testO")
    return z


def run_pairs(df: pd.DataFrame, pairs: list[tuple]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The frozen morphology / domain_gap functions on another list of pairs (they read the module-level PAIRS)."""
    saved = M.PAIRS
    try:
        M.PAIRS = list(pairs)
        _, ks = M.morphology(df)
        auc, feats = M.domain_gap(df) if len(pairs) <= 5 else (None, None)
    finally:
        M.PAIRS = saved
    return ks, auc, feats


def main() -> None:
    ap = argparse.ArgumentParser(description="renderer realism score with the frozen open_ferro_morph functions")
    ap.add_argument("--syn", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tune", action="store_true", help="trainO rows + primary pairs only (tuning loop)")
    ap.add_argument("--check-frozen", action="store_true", help="v1: frozen CSVs must be reproduced exactly")
    a = ap.parse_args()
    assert _sha(FROZEN) == FROZEN_SHA, "frozen scoring script changed"
    syn = (ROOT / a.syn).resolve()
    out = ROOT / "outputs" / "renderer" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    cache = M.OUT / "particle_descriptors.csv.gz"
    real = pd.read_csv(cache)
    frozen_syn = real[~real.set.isin(REAL)]
    real = real[real.set.isin(REAL)]

    M.SYN = syn
    tuned = load_json(M.OUT / "seg_tuned_params.json")["best"]["ours"]
    seg_cfg = {k: v for k, v in tuned.items() if not k.startswith("train_")}
    jobs = [j for j in M.jobs(seg_cfg) if j[0].startswith("synthetic")] + chain_jobs(syn, out / "chain_masks")
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for part in ex.map(M.describe, jobs, chunksize=2) for r in part]
    pd.DataFrame(rows).to_csv(out / "synthetic_descriptors.csv.gz", index=False)
    sdf = pd.read_csv(out / "synthetic_descriptors.csv.gz")   # CSV round trip, as the frozen script reads its cache
    frozen_sets = sdf[sdf.set.isin(["synthetic_gt", "synthetic_raster"])]
    df = pd.concat([real, frozen_sets], ignore_index=True)       # exactly the frozen script's frame
    chains = sdf[sdf.set.str.startswith("synthetic_chain")]
    print(sdf.groupby("set").size().to_string(), flush=True)

    if a.check_frozen:   # v1 only: recomputed synthetic rows and the frozen outputs must match exactly
        old = frozen_syn.reset_index(drop=True)
        new = frozen_sets.reset_index(drop=True)[old.columns]
        pd.testing.assert_frame_equal(old, new, check_dtype=False, rtol=1e-9)
        summ, ks = M.morphology(df)
        auc, _ = M.domain_gap(df)
        pd.testing.assert_frame_equal(auc.reset_index(drop=True), pd.read_csv(M.OUT / "domain_gap_auc.csv"),
                                      check_dtype=False, rtol=1e-9)
        pd.testing.assert_frame_equal(ks.reset_index(drop=True), pd.read_csv(M.OUT / "morph_ks.csv"),
                                      check_dtype=False, rtol=1e-9)
        print("CHECK OK: frozen domain_gap_auc.csv and morph_ks.csv reproduced exactly", flush=True)

    full = pd.concat([df, chains, split_rows(df)], ignore_index=True)
    if a.tune:
        ks, auc, feats = run_pairs(full, TUNE_PAIRS + [("zenodo_gt_trainO", "synthetic_chain")])
        ks.to_csv(out / "tune_ks.csv", index=False)
        auc.to_csv(out / "tune_auc.csv", index=False)
        feats.to_csv(out / "tune_features.csv", index=False)
    else:
        if not a.check_frozen:
            summ, ks = M.morphology(df)
            auc, feats = M.domain_gap(df)
            feats.to_csv(out / "domain_gap_features.csv", index=False)
        summ.to_csv(out / "morph_summary.csv", index=False)
        ks.to_csv(out / "morph_ks.csv", index=False)
        auc.to_csv(out / "domain_gap_auc.csv", index=False)
        ks2, auc2, f2 = run_pairs(full, EXTRA_PAIRS)
        ks2.to_csv(out / "chain_ks.csv", index=False)
        auc2.to_csv(out / "chain_domain_gap_auc.csv", index=False)
        f2.to_csv(out / "chain_domain_gap_features.csv", index=False)
        ks3, auc3, _ = run_pairs(full, SPLIT_PAIRS)
        ks3.to_csv(out / "split_ks.csv", index=False)
        auc3.to_csv(out / "split_domain_gap_auc.csv", index=False)
        M.morphology(full)[0].to_csv(out / "morph_summary_all_sets.csv", index=False)
    json.dump({"syn": a.syn, "frozen_sha256": FROZEN_SHA, "tune": a.tune, "check_frozen": a.check_frozen,
               "n_rows": sdf.groupby("set").size().to_dict()}, open(out / "provenance.json", "w", encoding="utf-8"),
              indent=1)


if __name__ == "__main__":
    main()
