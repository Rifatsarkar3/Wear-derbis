"""Sensitivity analyses of the hold-out test of the frozen FZG factor (docs/preregistration_paired2.md sec. 6: "reported, never decisive"). Run AFTER
scripts/paired_holdout.py run; reads the same sources and the frozen constants and changes nothing of the main result.

    .venv/Scripts/python.exe -I scripts/paired_holdout_extra.py     # -> outputs/open_data/paired/holdout/sensitivity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]  # -I drops the cwd

import paired_holdout as ph  # noqa: E402

from wearpg.utils import save_json  # noqa: E402

OUT = ROOT / "outputs" / "open_data" / "paired" / "holdout"


def short(r: dict) -> dict:
    """The parts of the main result that a reader needs for a subset."""
    keys = ("n_units", "n_rows", "fold", "within_2", "H1a_verdict", "H1b_verdict", "H1b_debris_vs_constant", "H4_slope", "H5_spearman")
    return {k: r[k] for k in keys if k in r}


def run(new: pd.DataFrame, thesis: pd.DataFrame, c: dict) -> dict:
    r, _ = ph.evaluate(new, thesis, c)
    return short(r)


def main() -> None:
    c = json.loads((OUT / "frozen.json").read_text())["constants"]
    thesis, new = ph.thesis_rows(), ph.load_new()
    new["study_id"] = new.unit.str.split(":").str[0]
    out = {"status": "sensitivity, reported, never decisive; no refit of the frozen constants"}
    out["primary"] = run(new, thesis, c)
    out["without_each_source"] = {s: run(new[new.study_id != s].reset_index(drop=True), thesis, c) for s in sorted(new.study_id.unique())}
    out["single_material"] = {m: run(new[new.material_pair == m].reset_index(drop=True), thesis, c) for m in sorted(new.material_pair.unique())}
    out["rows_as_units"] = run(new.assign(unit=new.unit + ":" + new.row), thesis, c)
    k3 = new["row"].str.contains("K3")
    out["without_running_in_rows_(K3_stages)"] = run(new[~k3].reset_index(drop=True), thesis, c)
    out["interval_rows_only"] = run(new[new["mode"] == "interval"].reset_index(drop=True), thesis, c)
    second = new.assign(debris=new.debris_2.fillna(new.debris))
    out["second_reading_of_debris"] = run(second, thesis, c)
    # the gear pair whose mass loss is the pinion only does not exist in the primary; the one-gear sensitivity of the plan has no row (pij2011 prints no CPUC per stage)
    out["one_gear_only_rows"] = "none (the pinion-only micropitting tests of pij2011 print no CPUC)"
    ester1 = new.copy()
    ester1.loc[ester1.study_id == "pij2011", "cond_ester"] = 1
    out["pij2011_counted_as_ester"] = run(ester1, thesis, c)
    # ---- post hoc (after the result was known): what the new pairs look like
    um = np.log10(new.set_index("unit")[["debris", "wear"]]).groupby(level=0, sort=False).mean()
    rat = um.debris - um.wear
    first_last = {}
    for u, g in new[new["mode"] == "interval"].groupby("unit", sort=False):
        g = g.sort_values("order")
        first_last[u] = float((g.debris.iloc[0] / g.wear.iloc[0]) / (g.debris.iloc[-1] / g.wear.iloc[-1]))
    out["posthoc"] = {"unit_mean_wear_mg": [float(10 ** um.wear.min()), float(10 ** um.wear.max())], "gsd_wear_units": float(10 ** um.wear.std(ddof=1)), "gsd_debris_per_wear_units": float(10 ** rat.std(ddof=1)),
                      "row_wear_mg": [float(new.wear.min()), float(new.wear.max())], "debris_per_wear_first_over_last_stage": first_last, "first_over_last_range": [min(first_last.values()), max(first_last.values())], "rows_within_2": int((np.abs(np.log10(new.wear) - (c["debris_b"] + np.log10(new.debris))) <= np.log10(2)).sum())}
    save_json(out, OUT / "sensitivity.json")
    print(json.dumps({k: (v if isinstance(v, str) else {kk: vv for kk, vv in (v.items() if isinstance(v, dict) and "n_units" in v else [])}) for k, v in out.items()}, indent=1)[:5000])


if __name__ == "__main__":
    main()
