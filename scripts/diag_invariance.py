"""F5 SECONDARY ANALYSIS (DIAG; rules fixed beforehand in outputs/agent_logs/DIAG.md): the frozen C3 invariance rule
failed only through the elliptic Fourier descriptor (EFD 0.691 < 0.7 at 0.5x). Per transform, the difference in
stability Delta = group-median Spearman(graph geometry) - group-median Spearman(EFD) on the SAME real particles
(outputs/open_data/ferro/robust_particles.csv.gz; frozen stability() statistic, point values reproduce
robust_stability.csv), with a paired bootstrap: image-cluster (primary; image = unit of inference) and plain
particle (secondary). Holm across the 7 mask-mode transforms (primary family); pipeline mode = secondary family.
The frozen C3 verdict (against_us) is unchanged; this is a secondary analysis.
Outputs: outputs/diagnostics/f5_*.   Run: .venv/Scripts/python.exe -I scripts/diag_invariance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir

import pandas as pd

from diag_common import FERRO, NOTE, OUT, group_features, group_spearman, wide_pairs
from wearpg.stats import holm, paired_cluster_bootstrap
from wearpg.utils import save_json

FAMILIES = {"mask": ("rot45", "rot90", "rot_random", "scale0.5", "scale0.25", "blur1", "blur2"),
            "pipeline": ("rot45", "rot90", "rot_random", "scale0.5", "scale0.25", "blur1", "blur2", "noise0.02",
                         "noise0.05")}
KEY_TRANSFORMS = ("scale0.5", "rot45", "rot_random")  # where EFD failed the frozen rule
B = 2000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FERRO / "robust_particles.csv.gz")
    st = pd.read_csv(FERRO / "robust_stability.csv")
    geom, efd = group_features(df, "graph_geom"), group_features(df, "efd")
    rows = []
    for mode, ts in FAMILIES.items():
        for t in ts:
            wg, we = wide_pairs(df, mode, t, geom), wide_pairs(df, mode, t, efd)
            frozen = st[(st["mode"] == mode) & (st["transform"] == t)].groupby("group").spearman.median()
            sg, se = group_spearman(wg), group_spearman(we)
            assert abs(sg - frozen["graph_geom"]) < 1e-12 and abs(se - frozen["efd"]) < 1e-12, (mode, t, sg, se)
            for unit in ("image_id", "particle"):
                pb = paired_cluster_bootstrap(group_spearman, wg, we, unit, "particle", n_boot=B, seed=0)
                rows.append({"mode": mode, "transform": t, "bootstrap_unit": unit, "n_particles": len(wg),
                             "n_clusters": pb["n_clusters"], "S_graph_geom": sg, "S_efd": se, **pb["value"]})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby(["mode", "bootstrap_unit"]).p_value.transform(lambda p: holm(p.to_numpy()))
    out["graph_more_invariant"] = (out.p_holm < 0.05) & (out.lo > 0)
    out.to_csv(OUT / "f5_graph_minus_efd.csv", index=False)
    prim = out[(out["mode"] == "mask") & (out.bootstrap_unit == "image_id")].set_index("transform")
    res = {"note": NOTE, "status": "secondary analysis (frozen C3 verdict against_us unchanged)",
           "n_graph_geom_features": len(geom), "n_efd_features": len(efd),
           "primary_family": prim[["S_graph_geom", "S_efd", "diff", "lo", "hi", "p_holm",
                                   "graph_more_invariant"]].reset_index().to_dict("records"),
           "statement_graph_more_invariant_than_fourier": bool(prim.loc[list(KEY_TRANSFORMS),
                                                                        "graph_more_invariant"].all())}
    save_json(res, OUT / "f5_summary.json")
    pd.set_option("display.width", 200)
    print(out.drop(columns=["rel_diff"]).round(4).to_string())
    print("statement:", res["statement_graph_more_invariant_than_fourier"])


if __name__ == "__main__":
    main()
