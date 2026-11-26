"""Later sources (scripts/paired_later.py) and the re-review calculations (scripts/paired_review.py): stored results against the digitised data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "real" / "paired", ROOT / "outputs" / "open_data" / "paired"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def later():
    return json.loads((OUT / "later" / "results.json").read_text(encoding="utf-8"))


def test_holzhauer_end_ratio_and_slopes_follow_from_the_digitised_table(later):
    d = pd.read_csv(DATA / "holzhauer1983" / "later.csv").sort_values(["unit", "order"])
    end = d.groupby("unit").last()
    a = later["A_holzhauer1983"]["primary"]
    for u, v in a["end_D_over_W"].items():
        assert v == pytest.approx(end.loc[u, "conc_ppm"] / end.loc[u, "wear_mg"], abs=1e-3)
    assert a["end_conversion"]["gsd"] == pytest.approx(10 ** np.log10(end.conc_ppm / end.wear_mg).std(ddof=1))
    for u, s in a["slope_conc_per_mg"].items():
        q = d[d["unit"] == u]
        assert s == pytest.approx(np.polyfit(q.wear_mg, q.conc_ppm, 1)[0])


def test_nield_readings_reproduce_the_printed_regression(later):
    c = later["D_nield1961"]["check_printed_regression"]
    assert abs(c["slope"] - c["printed"][0]) < 0.01 and abs(c["r"] - c["printed"][2]) < 0.005


def test_identity_holds_on_made_up_sets():
    """Debris has the smaller leave-one-out error on unit means exactly when the mean absolute deviation of log D/W is the smaller one."""
    px = load("paired_extra")
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(5, 15))
        x, y = rng.normal(0, rng.uniform(0.1, 1), n), rng.normal(0, rng.uniform(0.1, 1), n)
        mad = lambda v: np.abs(v - v.mean()).mean()
        assert (px.loo_means(y, x) < px.loo_means(y)) == (mad(x - y) < mad(y))


def test_review_json_matches_a_recalculation():
    r = json.loads((OUT / "extra" / "review.json").read_text(encoding="utf-8"))
    assert r["bearing_rig_zero_shift"]["shifts"]["+0.00"]["gsd"] == pytest.approx(r["identity"]["per_set"]["phm_bolander2009/rig_tests"]["gsd_factor"])
    u = pd.read_csv(OUT / "units.csv")
    g = u[u.set == "fzg"]
    s = (g.log10_D - g.log10_W).std(ddof=1)
    assert r["identity"]["per_set"]["feup_martins2008/fzg"]["gsd_factor"] == pytest.approx(10 ** s)
    assert sum(v for v in r["verdict_counts"]["five_sets:pre_specified"]["V2"].values()) == 5
