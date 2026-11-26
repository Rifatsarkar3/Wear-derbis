"""Post-hoc analyses after the frozen run (scripts/paired_pooled.py): the K6-sample definition touches six rows only and cumulative correlations ignore a constant shift."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("paired_pooled", ROOT / "scripts" / "paired_pooled.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)


def test_k6_sample_changes_the_six_first_rows_only():
    thesis, new = pl.load()
    d = new[new.D0 != new.D1]
    assert len(d) == 6 and set(d.unit.str.split(":").str[0]) == {"jet495", "tribint2009"} and (d.D1 < d.D0).all()
    assert (thesis.D0 == thesis.D1).all() and (new.D0 == new.debris).all()


def test_p3_row_is_added_with_the_given_wear():
    _, new = pl.load()
    p3 = pl.with_p3(new, 1.0)
    assert len(p3) == len(new) + 1 and p3.wear.iloc[-1] == 1.0 and 10 < p3.debris.iloc[-1] < 13


def test_cumulative_correlation_does_not_see_the_k3_offset():
    _, new = pl.load()
    mp = new[new["mode"] == "interval"].reset_index(drop=True)
    a = pl.cumulative_vs_interval(mp, "D0", n_perm=20)
    b = pl.cumulative_vs_interval(mp, "D1", n_perm=20)
    assert np.isclose(a["mean_r2_cumulative_per_pair"], b["mean_r2_cumulative_per_pair"])      # shift of a whole curve: the same correlation
    assert a["interval_spearman"] < b["interval_spearman"]


def test_dilution_two_directions_and_pair_mean_errors():
    thesis, new = pl.load()
    q = pl.dilution(new, "D0")
    um = np.log10(new.set_index("unit")[["D0", "wear"]]).groupby(level=0, sort=False).mean()
    assert np.isclose(q["slope"] * q["reverse_slope"], np.corrcoef(um.D0, um.wear)[0, 1] ** 2)
    both = pl.pd.concat([thesis, new], ignore_index=True)
    um = np.log10(both.set_index("unit")[["D0", "wear"]]).groupby(level=0, sort=False).mean()
    y, n = um.wear.to_numpy(), len(um)
    own = np.abs(y - (y.sum() - y) / (n - 1)).mean()                        # leave-one-out constant by hand
    assert np.isclose(np.log10(pl.pair_mean_folds(both, "D0")["constant"]), own)


def test_two_stage_bootstrap_reproduces_the_frozen_error_and_is_wider():
    thesis, new = pl.load()
    q = pl.two_stage(pl.frame(new, "D0"), thesis, b=4000)
    assert 1.78 < q["fold"] < 1.79 and q["fold_ci_two_stage"][0] < q["fold"] < q["fold_ci_two_stage"][1]
    assert q["fold_ci_two_stage"][1] > 2.08 and q["signed_ci_two_stage"][0] < 0.83     # wider than the frozen-factor intervals
