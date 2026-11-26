"""Before (v1 smoke) / after (v2 smoke) table of the pre-specified renderer-realism criteria (outputs/agent_logs/RENDER.md
sec. 1) from the frozen-code outputs in outputs/renderer/smoke_v1 and smoke_v2. SIMULATION vs real ferrograms.
Writes outputs/renderer/before_after.csv. Run: python -I outputs/renderer/before_after.py
"""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
AUC_FILES = ("domain_gap_auc.csv", "chain_domain_gap_auc.csv", "split_domain_gap_auc.csv")
KS_FILES = ("morph_ks.csv", "chain_ks.csv", "split_ks.csv")
PAIRS = {"P1": ("zenodo_gt", "synthetic_raster"), "P2": ("zenodo_seg", "synthetic_raster"),
         "P3": ("tcbgy_seg", "synthetic_raster"), "P4": ("zenodo_gt", "synthetic_chain_fe"),
         "P5": ("zenodo_gt", "synthetic_chain"), "P1_testO": ("zenodo_gt_testO", "synthetic_raster"),
         "P4_testO": ("zenodo_gt_testO", "synthetic_chain_fe")}
PRIMARY_AUC = {("P1", "texture"), ("P1", "colour"), ("P4", "efd"), ("P4", "shape_scale_free")}


def load(tag: str, files, key: str, val: str) -> pd.Series:
    d = pd.concat([pd.read_csv(OUT / tag / f) for f in files]).drop_duplicates(["a", "b", key])
    return d.set_index(["a", "b", key])[val]


def main() -> None:
    rows = []
    auc0, auc1 = (load(t, AUC_FILES, "group", "auc") for t in ("smoke_v1", "smoke_v2"))
    for p, (a, b) in PAIRS.items():
        for g in ("texture", "colour", "efd", "shape_scale_free", "fractal", "size_px"):
            x0, x1 = auc0[(a, b, g)], auc1[(a, b, g)]
            met = bool(x1 <= x0 - 0.15 or x1 < 0.85) if g in ("texture", "colour", "efd", "shape_scale_free") else None
            rows.append(dict(target="a", pair=p, a=a, b=b, measure=f"AUC {g}", before=x0, after=x1, change=x1 - x0,
                             met=met, role="primary" if (p, g) in PRIMARY_AUC else
                             ("held-out breakdown" if "testO" in p else "secondary")))
    ks0, ks1 = (load(t, KS_FILES, "feature", "ks_D") for t in ("smoke_v1", "smoke_v2"))
    for p in ("P4", "P5", "P4_testO", "P3"):
        a, b = PAIRS[p]
        for f in ("aspect", "roundness", "solidity", "fractal_D", "size_area_px"):
            x0, x1 = ks0[(a, b, f)], ks1[(a, b, f)]
            if p == "P3":
                tgt, met = "c", (bool(x1 <= x0 + 0.05) if f in ("aspect", "roundness", "solidity") else None)
            else:
                tgt, met = "b", (bool(x1 <= 0.25) if f in ("aspect", "roundness", "solidity") else None)
            role = ("primary" if p in ("P4", "P3") and met is not None else
                    "held-out breakdown" if "testO" in p else "secondary")
            rows.append(dict(target=tgt, pair=p, a=a, b=b, measure=f"KS D {f}", before=x0, after=x1, change=x1 - x0,
                             met=met, role=role))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "before_after.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[df.role == "primary"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
