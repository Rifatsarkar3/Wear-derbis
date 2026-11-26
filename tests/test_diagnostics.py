"""Tests for the DIAG diagnostic helpers (scripts/diag_*.py): synthetic inputs only, no network, no real data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import diag_common as dc  # noqa: E402


def test_group_spearman_matches_scipy_with_ties_nan_and_constants():
    rng = np.random.default_rng(0)
    n = 40
    w = pd.DataFrame({"image_id": rng.integers(0, 8, n)})
    w["ref__a"], w["tr__a"] = rng.normal(size=n), rng.normal(size=n)
    w["ref__b"] = np.round(rng.normal(size=n), 1)  # ties
    w["tr__b"] = w["ref__b"] + rng.normal(0, 0.1, n)
    w["ref__c"], w["tr__c"] = 1.0, rng.normal(size=n)  # constant -> NaN, skipped
    w["ref__d"], w["tr__d"] = rng.normal(size=n), rng.normal(size=n)
    w.loc[3, "tr__d"] = np.nan  # NaN -> skipped (scipy propagates NaN)
    expect = np.median([sps.spearmanr(w["ref__a"], w["tr__a"])[0], sps.spearmanr(w["ref__b"], w["tr__b"])[0]])
    assert dc.group_spearman(w) == pytest.approx(expect, abs=1e-12)
    assert np.isnan(dc.group_spearman(w.iloc[:2]))


def test_wide_pairs_pairs_identity_with_transform_and_drops_lost():
    df = pd.DataFrame({"image_id": ["i1"] * 6, "particle": ["p1", "p2", "p3"] * 2, "mode": "mask",
                       "transform": ["identity"] * 3 + ["scale0.5"] * 3, "lost": [False, False, True, False, True,
                                                                                False],
                       "graph_mean_r_norm": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]})
    w = dc.wide_pairs(df, "mask", "scale0.5", ["graph_mean_r_norm"])
    assert w.particle.tolist() == ["p1"]  # p2 lost at scale, p3 lost at identity
    assert (w.ref__graph_mean_r_norm.iloc[0], w.tr__graph_mean_r_norm.iloc[0]) == (1.0, 10.0)
    assert dc.feature_group("graph_mean_tex_mean") == "graph_tex" and dc.feature_group("efd_3") == "efd"


def test_merge_labels_groups_fragments_by_gap_and_iou_matches_frozen():
    import diag_counts as f3
    from scipy import ndimage as ndi
    from wearpg.segmentation import _iou_matrix, _label_image

    lab = np.zeros((40, 60), np.int32)
    lab[10:20, 5:15] = 1
    lab[10:20, 19:29] = 2  # gap of 4 px to fragment 1
    lab[10:20, 45:55] = 3  # gap of 16 px to fragment 2
    edt = ndi.distance_transform_edt(lab == 0)
    assert f3.merge_labels(lab, edt, 1)[1] == 3
    merged, n = f3.merge_labels(lab, edt, 2)  # disks of radius 2 bridge the 4-px gap
    assert n == 2 and merged[15, 7] == merged[15, 25] != merged[15, 50]
    assert (merged > 0).sum() == (lab > 0).sum()  # object pixels = fragment pixels (dilation only groups)
    assert f3.merge_labels(lab, edt, 8)[1] == 1
    sq = [np.array([[5, 5], [15, 5], [15, 15], [5, 15]], float), np.array([[30, 30], [40, 30], [40, 40], [30, 40]],
                                                                          float)]
    gt = [np.array([[6, 6], [16, 6], [16, 16], [6, 16]], float)]
    shape = (50, 50)
    ours = f3.contingency_iou(_label_image(sq, shape), 2, _label_image(gt, shape), 1)
    np.testing.assert_allclose(ours, _iou_matrix(sq, gt, shape))


def test_f_min_requires_all_larger_sizes_to_pass():
    import diag_magnification as f4

    sw = pd.DataFrame({"F": [8, 12, 16, 20, 24, 32], "S": [0.85, 0.7, 0.82, 0.81, 0.9, 0.5],
                       "n": [100, 90, 80, 70, 60, 10]})  # F = 32 has n < 20 -> ignored
    assert f4.f_min(sw) == 16.0
    assert np.isnan(f4.f_min(sw.assign(S=0.5)))
    assert f4.adapted_cfg({"smooth_sigma": 2.0, "noise_k": 2.0}, 0.25) == {"smooth_sigma": 0.5, "noise_k": 2.0}


def test_add_rate_uses_run_in_origin_and_loo_alarm_or_rule():
    import diag_nasa_onset as f6

    rd = pd.DataFrame({"exp": [1, 1, 2, 3, 4], "reading": [160.0, 260.0, 1060.0, 560.0, 300.0],
                       "mass_mg": [2.0, 7.0, 3.0, 1.0, 9.0], "y": [0, 1, 0, 0, 1]})
    r = f6.add_rate(rd, {1: 60.0})
    np.testing.assert_allclose(r.rate, [2 / 100, 5 / 100, 3 / 1000, 1 / 500, 9 / 240])  # exp 2-4: default t0 60
    assert f6.add_rate(rd, None).rate.iloc[0] == pytest.approx(2 / 160)
    a_mass, a_or = f6.loo_alarms(r, ["mass_mg"]), f6.loo_alarms(r, ["mass_mg", "rate"])
    assert (a_or >= a_mass).all()  # the OR rule alarms wherever the mass rule does
    pe = f6.per_experiment(r.assign(alarm=a_or))
    assert pe.set_index("exp").loc[1, "destructive"] and not pe.set_index("exp").loc[2, "destructive"]
    with pytest.raises(ValueError):
        f6.add_rate(rd.assign(reading=[50.0, 260.0, 1060.0, 560.0, 300.0]), {1: 60.0})


def test_max_independent_sets_enumerates_every_maximum_set():
    import diag_composition as f7

    edges = {("A", "B"), ("B", "C"), ("C", "D")}  # path A-B-C-D: maximum sets {A,C}, {A,D}, {B,D}
    assert sorted(f7.max_independent_sets(["A", "B", "C", "D"], edges)) == [("A", "C"), ("A", "D"), ("B", "D")]
    assert f7.max_independent_sets(["A", "E"], edges) == [("A", "E")]
    ident = pd.DataFrame({"unit_a": ["T1", "T1", "T2"], "unit_b": ["T2", "T3", "T3"], "n_identical": [5, 4, 6]})
    assert f7.flagged_edges(ident, 5) == {("T1", "T2"), ("T2", "T3")}
