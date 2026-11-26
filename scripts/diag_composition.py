"""F7 SENSITIVITY (DIAG; rules fixed beforehand in outputs/agent_logs/DIAG.md): is the composition-vs-Fe-only result
(Zenodo 10.5281/zenodo.17287193, 27 tractors, 3 inferred life phases) robust to the suspected copied / templated
rows? Tractor pairs flagged by the frozen identity audit (outputs/open_data/oil/zenodo_identity_pairs.csv: >= 5 of
10 additive/contaminant columns exactly identical at the same working hours; secondary >= 4) are removed:
  literal  drop every tractor in any flagged pair ("not estimable" if a phase keeps < 3 tractors);
  MIS      every MAXIMUM duplicate-free subset (maximum independent set of the flagged-pair graph).
Same features (open_oil.composition_features), classifier and leave-one-unit-out (open_oil.loo_proba), metrics
(wearpg.metrics.classification_metrics) and paired unit bootstrap as scripts/open_oil_analysis.py. The frozen C2
result stays primary.
Outputs: outputs/diagnostics/f7_*.   Run: .venv/Scripts/python.exe -I scripts/diag_composition.py [--n-perm 999]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir

import numpy as np
import pandas as pd

from diag_common import NOTE, OIL, OUT, ROOT
from wearpg import open_oil as oo
from wearpg.metrics import classification_metrics
from wearpg.stats import paired_cluster_bootstrap
from wearpg.utils import save_json

THRESHOLDS = (5, 4)  # primary = the audit's own ">= 5 of 10" flag; secondary = above the cross-phase max of 3
COMP_COLS = ("Fe", "Cr", "Al", "Cu", "Pb", "Mn")
B = 2000


def flagged_edges(ident: pd.DataFrame, thr: int) -> set[tuple[str, str]]:
    e = ident[ident.n_identical >= thr]
    return {tuple(sorted(p)) for p in zip(e.unit_a, e.unit_b)}


def max_independent_sets(nodes: list[str], edges: set[tuple[str, str]]) -> list[tuple[str, ...]]:
    """Every maximum-size subset of nodes without an edge inside (exact enumeration; nodes <= ~15)."""
    for k in range(len(nodes), 0, -1):
        sets = [c for c in itertools.combinations(sorted(nodes), k)
                if not any(tuple(sorted(p)) in edges for p in itertools.combinations(c, 2))]
        if sets:
            return sets
    return [()]


def ba(y, proba, classes) -> float:
    return classification_metrics(y, proba, classes=tuple(classes))["balanced_accuracy"]


def evaluate(units: list[str], phase: pd.Series, variants: dict, n_perm: int) -> list[dict]:
    """3-class LOO of each feature variant on `units` + paired unit bootstrap composition - fe_only."""
    classes = list(oo.PHASES)
    y = phase.loc[units].map(classes.index).to_numpy()
    rows, frames = [], {}
    for name, X in variants.items():
        Xv = X.loc[units].to_numpy()
        proba = oo.loo_proba(Xv, y, len(classes))
        frames[name] = pd.DataFrame(proba, columns=["p0", "p1", "p2"]).assign(y_true=y, unit_id=units)
        p = oo.permutation_balanced_accuracy(Xv, y, len(classes), n_perm=n_perm, seed=0)[1] \
            if n_perm and name in ("composition", "fe_only") else float("nan")
        rows.append({"features": name, "balanced_accuracy": ba(y, proba, classes), "perm_p": p,
                     "n_perm": n_perm if np.isfinite(p) else 0})
    for name in [v for v in variants if v != "fe_only"]:
        pb = paired_cluster_bootstrap(lambda d: ba(d.y_true.to_numpy(), d[["p0", "p1", "p2"]].to_numpy(), classes),
                                      frames[name], frames["fe_only"], "unit_id", "unit_id", n_boot=B, seed=0)["value"]
        rows.append({"features": f"{name}_minus_fe_only", "balanced_accuracy": pb["diff"], "lo": pb["lo"],
                     "hi": pb["hi"], "p_boot": pb["p_value"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="F7 composition sensitivity without copied-row tractors")
    ap.add_argument("--n-perm", type=int, default=999)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    s = oo.load_zenodo_wear_debris(ROOT / "data/real/zenodo_wear_debris/raw/wear_derbis.xlsx")
    phase = s.groupby("unit_id").phase.first()
    comp = oo.composition_features(s)
    _, unit = oo.debris_rates(s, ["Fe"], q=0.0)
    fe = np.log10(unit["rate_cum"]).rename("log10_rate_cum_Fe").to_frame()
    variants = {"composition": comp, "fe_only": fe, "composition_no_Mn": comp.drop(columns="alr_Mn")}
    ident = pd.read_csv(OIL / "zenodo_identity_pairs.csv")
    res: dict = {"note": NOTE, "status": "sensitivity", "primary": "frozen C2: composition 1.00 vs Fe-only 0.815, "
                 "diff 0.185 [0.067, 0.296] (all 27 tractors) - unchanged"}

    # reproduction of the frozen all-unit balanced accuracies (no permutations needed)
    rep = {r["features"]: r["balanced_accuracy"] for r in evaluate(sorted(phase.index), phase, variants, 0)}
    res["reproduction_all_27"] = rep
    assert abs(rep["composition"] - 1.0) < 1e-12 and abs(rep["fe_only"] - 0.8148148148148149) < 1e-12, rep

    rows = []
    for thr in THRESHOLDS:
        edges = flagged_edges(ident, thr)
        assert all(phase[u] == phase[v] for u, v in edges), "flagged pair across phases"
        involved = sorted({u for e in edges for u in e})
        keep = phase.drop(involved).value_counts().reindex(list(oo.PHASES), fill_value=0)
        lit = {"threshold": thr, "n_flagged_pairs": len(edges), "n_involved": len(involved), "involved": involved,
               "remaining_per_phase": keep.to_dict()}
        if (keep < 3).any():
            lit["result"] = "not estimable (a phase keeps < 3 tractors)"
        else:
            lit["rows"] = evaluate(sorted(phase.drop(involved).index), phase, variants, 0)
        res[f"literal_ge{thr}"] = lit
        per_phase = {ph: max_independent_sets(sorted(phase.index[phase == ph]), edges) for ph in oo.PHASES}
        subsets = [sorted(sum(c, ())) for c in itertools.product(*per_phase.values())]
        res[f"mis_ge{thr}"] = {"n_subsets": len(subsets), "subset_size": len(subsets[0]),
                               "per_phase_size": {ph: len(v[0]) for ph, v in per_phase.items()},
                               "subsets": subsets}
        for i, units in enumerate(subsets):
            for r in evaluate(units, phase, variants, a.n_perm if thr == THRESHOLDS[0] else 0):
                rows.append({"threshold": thr, "subset": i, "n_units": len(units), **r})
            print(f"thr {thr} subset {i}: {rows[-4:]}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "f7_composition_sensitivity.csv", index=False)

    # rules (DIAG.md), primary threshold, composition vs fe_only
    d = out[(out.threshold == THRESHOLDS[0]) & (out.features == "composition_minus_fe_only")]
    res["rule"] = ("C2 robust to the copied-row concern" if (d.lo > 0).all() else
                   "C2 weakened" if (d.balanced_accuracy <= 0).any() else "inconclusive")
    res["diff_range_primary"] = {"min": float(d.balanced_accuracy.min()), "max": float(d.balanced_accuracy.max()),
                                 "lo_min": float(d.lo.min()), "hi_max": float(d.hi.max())}

    # descriptive identity audit on the composition columns themselves
    ic = oo.pairwise_identity_audit(s, COMP_COLS)
    res["identity_composition_columns"] = ic.groupby("same_phase").n_identical.agg(
        n_pairs="size", mean="mean", max="max", share_ge2=lambda v: float((v >= 2).mean())).reset_index().to_dict(
        "records")
    save_json(res, OUT / "f7_summary.json")
    print(out.round(3).to_string())
    print(res["rule"], res["diff_range_primary"], res["identity_composition_columns"])


if __name__ == "__main__":
    main()
