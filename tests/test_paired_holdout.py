"""Frozen-factor hold-out (scripts/paired_holdout.py): verdict rules and no-refit behaviour on made-up gear pairs."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("paired_holdout", Path(__file__).resolve().parents[1] / "scripts" / "paired_holdout.py")
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)
ph.B = 4000  # enough for the rules; the plan uses 2,000,000
ph.boot_ci.__defaults__ = (4000, 0, 4000)


def made_up(n, b, seed, proportional=True):
    rng = np.random.default_rng(seed)
    d = 10 ** rng.uniform(2, 4, n)
    w = 10 ** (b + np.log10(d) + rng.normal(0, 0.15, n)) if proportional else 10 ** rng.uniform(0, 3, n)
    return pd.DataFrame({"unit": [f"u{i}" for i in range(n)], "debris": d, "wear": w, "cond_cycles_M": 4.32, "cond_hertz_MPa": 1500.0,
                         "cond_ester": rng.integers(0, 2, n), "cond_adi": 0})


@pytest.fixture
def thesis():
    t = made_up(14, -1.0, 1).assign(cond_cycles_M=lambda d: np.linspace(1.4, 5.4, 14), cond_hertz_MPa=lambda d: np.linspace(760, 1800, 14))
    return t


def test_proportional_new_pairs_support_h1(thesis):
    r, _ = ph.evaluate(made_up(12, -1.0, 2), thesis, ph.constants(thesis))
    assert r["H1a_verdict"] == r["H1b_verdict"] == "supported" and r["decision"].startswith("GO, strong")
    assert r["fold"]["debris"] < 2 and r["H4_slope"]["verdict"].startswith("for_us")


def test_unrelated_new_pairs_and_small_samples_do_not(thesis):
    c = ph.constants(thesis)
    bad = ph.evaluate(made_up(12, -1.0, 3, proportional=False), thesis, c)[0]
    assert bad["H1a_verdict"] != "supported" and bad["H1b_verdict"] != "supported" and bad["decision"].startswith("NO-GO")
    few = ph.evaluate(made_up(7, -1.0, 4), thesis, c)[0]
    assert few["H1a_verdict"].startswith("not testable") and few["decision"].startswith("NO-GO")


def test_constants_are_frozen_and_one_factor_is_wrong_when_it_drifts(thesis):
    c = ph.constants(thesis)
    drift = made_up(12, 0.0, 5)  # ten times more wear per debris than the calibration rig: the frozen factor must be marked as failing
    r, p = ph.evaluate(drift, thesis, c)
    assert r["H1a_verdict"] == "against" and r["H1b_verdict"] == "against" and np.allclose(p.pred_log10_debris.iloc[0], c["debris_b"] + np.log10(drift.debris.iloc[0]))
