"""How many independent gear pairs show that debris beats a constant? Resampling of the per-pair error differences of the FZG set
(leave-one-pair-out, debris minus constant; outputs/open_data/paired/loo_predictions.csv). Planning calculation, approximate.

    .venv/Scripts/python.exe -I scripts/paired_power.py    # -> outputs/open_data/paired/holdout/power.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "open_data" / "paired" / "holdout" / "power.json"


def main():
    e = pd.read_csv(ROOT / "outputs" / "open_data" / "paired" / "loo_predictions.csv")
    e = e[(e.study == "feup_martins2008") & (e.set == "fzg")]
    y = np.log10(e.W)
    err = {m: (y - e[m]).abs().groupby(e.unit).mean() for m in ("M0", "M2")}
    d = (err["M2"] - err["M0"]).to_numpy()
    rng = np.random.default_rng(0)
    res = {"n_pairs_observed": len(d), "mean_difference": float(d.mean()), "sd_difference": float(d.std(ddof=1)), "test": "two-sided one-sample t, alpha 0.05, 20000 resamples", "power": {}}
    for n in (8, 12, 16, 20, 24, 28, 32):
        s = d[rng.integers(0, len(d), (20000, n))]
        p = stats.ttest_1samp(s, 0, axis=1).pvalue
        res["power"][n] = float(((s.mean(1) < 0) & (p < 0.05)).mean())
    ext = np.isin(err["M0"].index, ["CM_pl", "E1_pI", "E2_pI"])          # the two scuffed pairs and the 7 mg pair
    res["share_of_gain_from_three_extreme_pairs"] = float(d[ext].sum() / d.sum())
    res["power_without_extremes"] = {}
    for n in (20, 32):
        s = d[~ext][rng.integers(0, int((~ext).sum()), (20000, n))]
        res["power_without_extremes"][n] = float(((s.mean(1) < 0) & (stats.ttest_1samp(s, 0, axis=1).pvalue < 0.05)).mean())
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
