"""Tests for wearpg.paired and scripts/paired_analysis.py: exact statistics against brute force, schema, verdict rules, acceptance of
additional studies, and (if the local data exist) the transcribed numbers. No network."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wearpg import paired as pp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real" / "paired"
sys.path.insert(0, str(ROOT / "scripts"))

import paired_analysis as pa  # noqa: E402


def frame(D, W, set_="s", units=None, study="toy", **kw):
    n = len(D)
    base = {"study": study, "set": set_, "unit": units or [f"u{i}" for i in range(n)], "row": [f"r{i}" for i in range(n)], "order": 1,
            "mode": "interval", "rig": "r", "material_pair": "m", "lubricant": "l", "debris": D, "debris_metric": "d", "debris_unit": "mg",
            "debris_type": "mass", "wear": W, "wear_metric": "w", "wear_unit": "mg", "resolved": np.nan, "debris_2": np.nan, "wear_2": np.nan,
            "debris_err": np.nan, "wear_err": np.nan, "page": "1", "item": "t", "extraction": "text", "note": ""}
    return pd.DataFrame({**base, **kw})


def test_schema_rejects_bad_frames():
    ok = frame([1.0, 2.0], [1.0, 2.0])
    assert pp.validate(ok) is ok
    for broken in (ok.drop(columns="page"), ok.assign(foo=1), ok.assign(row="x"), ok.assign(mode="other"), ok.assign(wear=[-1.0, 1.0]),
                   ok.assign(wear=[np.nan, 1.0]), ok.assign(debris_type="unknown"), ok.assign(page=[None, "2"])):
        with pytest.raises(ValueError):
            pp.validate(broken)
    assert pp.validate(ok.assign(cond_load=1.0, class_x=2.0, alt_debris=3.0)) is not None


def test_select_exclusions_and_fallback():
    d = frame([1.0, 2.0, 0.0, 4.0], [1.0, 2.0, 3.0, 4.0], units=["a", "b", "c", "d"], wear_2=[np.nan, 5.0, np.nan, np.nan])
    assert pp.select(d).u.tolist() == ["a", "b", "d"]  # zero debris cannot enter a log scale
    assert pp.select(d, exclude=["a"]).u.tolist() == ["b", "d"] and pp.select(d, exclude=["a"], include=["a"]).u.tolist() == ["a", "b", "d"]
    assert pp.select(d, drop=["a"], include=["a"]).u.tolist() == ["b", "d"]
    assert pp.select(d, wear="wear_2").u.tolist() == ["b"]
    assert pp.select(d, wear="wear_2", fallback=True).W.tolist() == [1.0, 5.0, 4.0]


def test_slope_permutation_test_and_interval_against_brute_force():
    rng = np.random.default_rng(4)
    x = rng.normal(size=6)
    y = 0.8 * x + rng.normal(0, 0.4, 6)
    r = pp.slope(x, y)
    perms = np.array(list(itertools.permutations(range(6))))

    def p_at(b):
        e, xc = y - b * x, x - x.mean()
        t = np.abs(np.array([e[q] @ xc for q in perms]))
        return (t >= abs(e @ xc) - 1e-12).mean()

    assert r["perm_exact"] and r["n_perm"] == 720
    assert r["p_slope_eq_ref"] == pytest.approx(p_at(1.0)) and r["p_slope_eq_0"] == pytest.approx(p_at(0.0))
    for edge, sign in ((r["perm_lo"], -1), (r["perm_hi"], 1)):
        assert p_at(edge - sign * 1e-6) > 0.05 >= p_at(edge + sign * 1e-6)
    assert r["perm_lo"] < r["slope"] < r["perm_hi"] and p_at(r["slope"]) == 1.0
    assert r["slope"] == pytest.approx(np.polyfit(x, y, 1)[0])
    assert abs(r["slope"]) <= abs(r["rma_slope"]) <= abs(r["inverse_slope"])


def test_slope_interval_covers_and_tiny_samples_give_open_intervals():
    rng = np.random.default_rng(0)
    inside = 0
    for _ in range(60):
        x = rng.normal(size=7)
        r = pp.slope(x, x + rng.normal(0, 0.5, 7))
        inside += r["perm_lo"] <= 1 <= r["perm_hi"]
    assert inside >= 52  # nominal 95 %
    assert np.isinf(pp.slope([0.0, 1.0, 2.0], [0.1, 0.9, 2.2])["perm_hi"])  # 3! = 6 relabellings cannot reject at 5 %
    big = pp.slope(np.arange(12.0), np.arange(12.0) * 2 + np.sin(np.arange(12.0)))
    assert not big["perm_exact"] and big["perm_lo"] < 2 < big["perm_hi"]


def test_spearman_and_sign_flip_exact():
    x, y = np.array([1.0, 2, 3, 4, 5]), np.array([1.0, 3, 2, 5, 4])
    r = pp.spearman(x, y)
    want = np.mean([abs(np.corrcoef(x, y[list(q)])[0, 1]) >= 0.8 - 1e-12 for q in itertools.permutations(range(5))])
    assert r["rho"] == pytest.approx(0.8) and r["p"] == pytest.approx(want) and r["exact"]
    assert np.isnan(pp.spearman([1, 1, 1], [1, 2, 3])["rho"])
    assert pp.signflip_p([1, 2, 3, 4, 5]) == pytest.approx(2 / 32) and pp.signflip_p([1, -1]) == 1.0
    assert pp.signflip_p(np.ones(20)) < 0.001  # Monte Carlo branch
    m = pp.mean_diff([-1.0, -2.0, -3.0, -0.5, -1.5])
    assert m["diff"] == pytest.approx(-1.6) and m["hi"] < 0 and m["p"] == pytest.approx(2 / 32)


def test_loo_models_equal_refits_and_have_equal_parameter_counts():
    rng = np.random.default_rng(1)
    unit = np.array(list("aabbcdef"))
    x, Z = rng.normal(size=8), rng.normal(size=(8, 1))
    y = x + 0.7 + 0.5 * Z[:, 0]
    p = pp.loo_predictions(x, y, Z, unit)
    for u in "abcdef":
        te = unit == u
        assert p.M0[te].tolist() == pytest.approx([y[~te].mean()] * te.sum())
        assert p.M2[te].to_numpy() == pytest.approx(x[te] + (y - x)[~te].mean())
    assert np.allclose(p.M3, y)  # the generating model
    assert np.allclose(pp.loo_predictions(x, x + 3.0, Z, unit).M2, x + 3.0)  # exact proportionality -> zero error
    e = pp.unit_errors(p, y, unit)
    assert e.index.tolist() == list("abcdef") and e.loc["a", "M0"] == pytest.approx(np.abs(p.M0 - y)[:2].mean())


def test_conditions_are_cut_and_constants_dropped():
    d = frame([1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5], cond_a=[1.0, 10, 100, 1000, 1e4], cond_b=1.0, cond_c=[0, 1, 0, 1, 0])
    Z, names = pp.conditions_matrix(d, [["cond_b", "id"], ["cond_a", "log10"], ["cond_c", "id"]], 5)
    assert names == ["cond_a"] and Z.shape == (5, 1) and Z[:, 0].tolist() == pytest.approx([0, 1, 2, 3, 4])
    assert pp.conditions_matrix(d, [["cond_a", "log10"], ["cond_c", "id"]], 6)[1] == ["cond_a", "cond_c"]


def test_verdict_rules_follow_the_plan():
    assert pp.verdict_slope(0.6, 1.5) == "for_us strong" and pp.verdict_slope(0.2, 1.5) == "for_us"
    assert pp.verdict_slope(-0.1, 1.2) == "inconclusive" and pp.verdict_slope(0.1, 0.9) == pp.verdict_slope(1.1, 3) == "against_us"
    assert pp.verdict_spearman(0.8, 0.1, 0.01) == "for_us" and pp.verdict_spearman(0.8, -0.1, 0.01) == "inconclusive"
    assert pp.verdict_spearman(-0.1, -0.5, 0.9) == "against_us" and pp.verdict_spearman(0.9, 0.5, 0.06) == "inconclusive"
    assert [pp.verdict_diff(*a) for a in ((-1, -0.1), (-1, 0.1), (0.5, 1), (np.nan, np.nan))] == ["for_us", "inconclusive", "against_us",
                                                                                                "not testable"]
    assert [pp.verdict_gsd(*a) for a in ((1.2, 3), (1.5, 3), (2.5, 3), (1.1, 2))] == ["for_us", "inconclusive", "against_us", "not testable"]


def test_conversion_summary():
    c = pp.conversion(np.log10([1.0, 10.0, 100.0]))
    assert c["gm"] == pytest.approx(10) and c["gsd"] == pytest.approx(10) and c["max_over_min"] == pytest.approx(100)


def _write(root, sid, df, sets, **meta):
    (root / sid).mkdir(parents=True)
    df.to_csv(root / sid / "paired.csv", index=False)
    spec = {"id": sid, "citation": "c", "doi": None, "url": "u", "rights": "r", "added_after_first_run": False, **meta, "sets": sets}
    (root / sid / "study.json").write_text(json.dumps(spec), encoding="utf-8")


def test_analysis_script_accepts_additional_studies(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    D = 10 ** rng.uniform(0, 2, 8)
    W = 3 * D * 10 ** rng.normal(0, 0.05, 8)
    _write(tmp_path / "in", "aa_first", frame(D, W, study="aa_first", cond_load=10 ** rng.uniform(0, 1, 8)),
           {"s": {"conditions": [["cond_load", "log10"]], "resolved_def": None, "absolute": {"factor": 1.0, "note": ""}}})
    _write(tmp_path / "in", "bb_later", frame(D[::-1].copy(), W, study="bb_later"), {"s": {"conditions": [], "resolved_def": None}},
           added_after_first_run=True)
    _write(tmp_path / "in", "cc_small", frame([1.0, 2, 3], [1.0, 2, 4], study="cc_small"), {"s": {"conditions": [], "resolved_def": None}})
    monkeypatch.setattr(pa, "DATA", tmp_path / "in")
    monkeypatch.setattr(pa, "OUT", tmp_path / "out")
    pa.main()
    first = (tmp_path / "out" / "results.json").read_bytes()
    r = json.loads(first)
    assert r["inferential_sets"] == ["aa_first/s"] and r["inferential_sets_added_later"] == ["bb_later/s"]
    assert r["sets"]["cc_small/s"]["role"] == "descriptive"
    p = r["sets"]["aa_first/s"]["primary"]
    assert p["V1"].startswith("for_us") and p["V2"] == "for_us" and p["DoverW_gm"] == pytest.approx(1 / 3, rel=0.1)
    assert r["families"]["with_later_studies"]["n_inferential"] == 2
    pa.main()  # idempotent
    assert (tmp_path / "out" / "results.json").read_bytes() == first


def test_loader_rejects_set_mismatch(tmp_path):
    _write(tmp_path, "x", frame([1.0], [1.0], study="x"), {"other": {}})
    with pytest.raises(ValueError, match="sets"):
        pp.load_studies(tmp_path)


@pytest.mark.skipif(not (DATA / "feup_martins2008" / "paired.csv").exists(), reason="local paired data absent")
def test_digitised_numbers_match_the_printed_ones():
    st = {s["id"]: s["data"] for s in pp.load_studies(DATA)}
    f = st["feup_martins2008"]
    z = f[f["set"] == "fzg"].set_index("row")
    assert len(z) == 20 and z["unit"].nunique() == 14
    assert z.loc["CM_mp:K9-1", ["debris", "wear"]].tolist() == [480, 45] and z.loc["AE_mp:K7", "wear"] == 132
    assert z.loc["E1_pI:power loss I", "alt_debris_last_sample"] == 8780 and z.loc["E2_pI:power loss I", "wear"] == 7
    assert z.loc["CM_pl:power loss F", "debris"] > z.loc["CM_pl:power loss F", "alt_debris_last_sample"]  # removed debris is added back
    assert z.loc["CM_mp:K3+K7", "class_DL"] == pytest.approx(239, abs=1)  # printed DL 23.9 at dilution 0.1
    r = st["phm_raadnui2015"]
    assert r.debris.tolist() == pytest.approx([2.4999, 2.7999, 7.401, 16.2, 32.49])
    assert (r[["class_fine_mg", "class_mid_mg", "class_coarse_mg"]].sum(axis=1) / r.debris).between(0.9, 1.1).all()
    n = st["nasa_tm2011_217117"]
    assert n.groupby("unit").size().tolist() == [25, 24, 21, 26]
    assert n.groupby("unit").wear.last().tolist() == pytest.approx([16.2, 27.8, 15.2, 19.2], abs=0.05)
    assert (n.sort_values("order").groupby("unit").debris.diff().dropna() > 0).all()
    c = st["soton_craig2010"].set_index("unit")
    assert c.loc["T1", "wear"] == pytest.approx(37.16, abs=0.02) and c.loc["T4", "wear"] == pytest.approx(67.26, abs=0.02)
    assert c.class_scored.tolist() == [1, 1, 1, 1, 1, 1, 0]


@pytest.mark.skipif(not (ROOT / "outputs/open_data/paired/results.json").exists(), reason="run scripts/paired_analysis.py first")
def test_committed_results_follow_the_prewritten_rules():
    r = json.loads((ROOT / "outputs/open_data/paired/results.json").read_text(encoding="utf-8"))
    for v in r["sets"].values():
        p = v["primary"]
        if "slope" not in p:
            continue
        assert p["V1"] == pp.verdict_slope(p["slope_lo"], p["slope_hi"]) and p["V2"] == pp.verdict_diff(p["V2_diff"], p["V2_hi"])
        assert (v["role"] == "inferential") <= (p["n_units"] >= pp.MIN_UNITS)
    c = r["families"]["first_run"]["verdict_counts"]
    n = r["families"]["first_run"]["n_inferential"]
    assert sum(c["V1"].values()) == n and (r["families"]["first_run"]["overall"] == "SUPPORTED") == (
        c["V1"].get("for_us", 0) > n / 2 and c["V2"].get("for_us", 0) > n / 2)
