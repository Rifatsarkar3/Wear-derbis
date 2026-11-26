"""SIMULATION stress test of wearpg/full under confocal volume-measurement noise, the one stress factor the run-level
baselines cannot see (only the wearpg volume head uses confocal volumes).

Variants of the v1 pilot dataset (data/synthetic_pilot; renderer not calibrated to real ferrograms):
  base   the dataset as generated (confocal volume = true x LogNormal(0, 0.15) on 20 % of the particles)
  noisy  the same confocal particles with volume_um3 = true_volume_um3 x LogNormal(0, 0.6) (seed 0)
  none   no confocal volumes at all (the volume head is constrained only by the physics losses)
Variant tables go to data/sim_stress/confocal_<variant>/ (images shared through a directory junction). Each variant:
run_cv with the frozen configs/pilot.yaml training settings for wearpg/full, physics_only, conditions_only and
physics_conditions (5 folds, seed 0), evaluate_results, then outputs/simulation/stress_wearpg/{summary.csv, README.md}.
Resumable: prediction parts are skipped when present. Usage: python scripts/sim_wearpg_stress.py [variant ...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wearpg import experiments as ex
from wearpg.config import load_config
from wearpg.schema import WearDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "synthetic_pilot"
OUT = ROOT / "outputs" / "simulation" / "stress_wearpg"
VARIANTS = ("base", "noisy", "none")
MODELS = ["wearpg", "physics_only", "conditions_only", "physics_conditions"]
BANNER = ("SIMULATION - synthetic pilot data (renderer not calibrated to real ferrograms); verification of the "
          "physics chain only, not evidence about real wear")


def make_variant(name: str) -> Path:
    """Data root of the variant (base = the v1 dataset itself)."""
    if name == "base":
        return SRC
    dst = ROOT / "data" / "sim_stress" / f"confocal_{name}"
    if (dst / "dataset.json").exists():
        return dst
    ds = WearDataset.load(SRC)
    p = ds.particles.copy()
    conf = p.volume_um3.notna().to_numpy()
    if name == "noisy":
        p.loc[conf, "volume_um3"] = p.true_volume_um3[conf] * np.exp(np.random.default_rng(0).normal(0, 0.6, conf.sum()))
    else:
        p["volume_um3"] = np.nan
        p["thickness_um"] = np.nan
    ds.particles = p
    ds.meta = {**ds.meta, "name": f"synthetic_pilot_confocal_{name}",
               "stress_variant": f"confocal volumes {name} (scripts/sim_wearpg_stress.py)"}
    ds.save(dst)  # tables only; images via a junction to the v1 images
    if not (dst / "images").exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst / "images"), str(SRC / "images")], check=True,
                       capture_output=True)
    WearDataset.load(dst).assert_valid()
    return dst


def config(name: str) -> dict:
    cfg = load_config(ROOT / "configs" / "pilot.yaml", ["eval.seeds=[0]", "eval.secondary_seeds=null",
                                                         "eval.n_bootstrap=1000", "ablations=[]"])
    cfg["paths"].update(data_root=str(make_variant(name)), output_dir=str(OUT / name))
    cfg["baselines"]["particle"] = []
    return cfg


def summarize() -> None:
    rows = []
    for v in VARIANTS:
        m = OUT / v / "metrics"
        if not (m / "regression.csv").exists():
            continue
        reg, vc = pd.read_csv(m / "regression.csv"), pd.read_csv(m / "volume_consistency.csv")
        for (model, variant), g in reg[(reg.level == "run") & (reg.protocol == "cv")].groupby(["model", "variant"]):
            g = g.set_index("metric")
            row = {"variant": v, "model": model, "model_variant": variant}
            for k in ("mae", "within_condition_spearman"):
                row.update({k: g.at[k, "point"], f"{k}_lo": g.at[k, "lo"], f"{k}_hi": g.at[k, "hi"]})
            c = vc[(vc.model == model) & (vc.variant == variant) & (vc.protocol == "cv")]
            row.update({"slope": c.slope.iloc[0] if len(c) else np.nan, "ccc": c.ccc.iloc[0] if len(c) else np.nan,
                        "n_runs": int(g.at["mae", "n"])})
            rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False, lineterminator="\n")
    fmt = lambda r, k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"  # noqa: E731
    lines = ["# wearpg/full under confocal volume noise", "", f"**{BANNER}**", "",
             "Variants of data/synthetic_pilot (v1): base = as generated (confocal volume sd 0.15 ln units on 20 % of "
             "the particles), noisy = sd 0.6 on the same particles, none = no confocal volumes. Frozen configs/pilot.yaml "
             "training settings, protocol cv (5 folds), seed 0, cluster bootstrap 1000 replicates (run CIs).", "",
             "| variant | model | run MAE log10 k | within-cond. Spearman | volume slope | volume CCC | n runs |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['variant']} | {r['model']}/{r['model_variant']} | {fmt(r, 'mae')} | "
              f"{fmt(r, 'within_condition_spearman')} | {r['slope']:.3f} | {r['ccc']:.3f} | {r['n_runs']} |"
              for _, r in df.iterrows()]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(VARIANTS)
    for name in names:
        cfg = config(name)
        out = ex.run_cv(cfg, models=MODELS, variants=["full"])
        ex.evaluate_results(out, cfg)
        summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
