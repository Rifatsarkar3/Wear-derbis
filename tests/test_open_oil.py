"""Tests for wearpg.open_oil (tiny synthetic workbooks; no network, no real data needed)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl")

from wearpg import open_oil as oo  # noqa: E402


def _zenodo_xlsx(path, ages=(2, 6, 10), hours=oo.SAMPLE_HOURS):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(list(oo.ZENODO_HEADER))
    for k, age in enumerate(ages):
        for i, h in enumerate(hours):
            fe = 10.0 * (k + 1) * (i + 1)  # unit k: Fe = 10 (k+1) (i+1) ppm
            row = [age, h, fe] + [1.0 + 0.1 * j for j in range(len(oo.ZENODO_ELEMENTS) - 1)]
            ws.append(row + ["negetive" if i == 0 else "negative", 0.1, 10, 60])
        ws.append([None] * len(oo.ZENODO_HEADER))
    s2 = wb.create_sheet("Sheet2")
    s2.append([None, "Fe", "Cr", "Fuel Dilution"])
    s2.append(["Infant mortality life", 30.0, 1.0, 0.1])  # = unit T01 mean Fe (10..50)
    s2.append(["Wear-out life", 999.0, 2.0, 0.1])
    wb.save(path)
    return path


def test_load_zenodo_layout_and_phase(tmp_path):
    df = oo.load_zenodo_wear_debris(_zenodo_xlsx(tmp_path / "w.xlsx"))
    assert df.shape[0] == 15 and list(df["unit_id"].unique()) == ["T01", "T02", "T03"]
    assert df.groupby("unit_id")["phase"].first().tolist() == list(oo.PHASES)
    assert set(df["water_content"]) == {"negative"}
    assert df["Fe"].dtype == float and df.loc[df.unit_id == "T02", "Fe"].tolist() == [20, 40, 60, 80, 100]


def test_load_zenodo_rejects_bad_layout(tmp_path):
    with pytest.raises(ValueError, match="expected hours"):
        oo.load_zenodo_wear_debris(_zenodo_xlsx(tmp_path / "h.xlsx", hours=(20, 40, 60, 80, 120)))
    with pytest.raises(ValueError, match="contiguous"):
        oo.load_zenodo_wear_debris(_zenodo_xlsx(tmp_path / "o.xlsx", ages=(10, 2, 6)))


def test_phase_summary_and_reproduction(tmp_path):
    p = _zenodo_xlsx(tmp_path / "w.xlsx")
    summ = oo.load_zenodo_phase_summary(p)
    assert set(summ["phase"]) == {"infant_mortality", "wear_out"} and len(summ) == 6
    rep = oo.summary_reproduction(oo.load_zenodo_wear_debris(p), summ)
    fe = rep[(rep.phase == "infant_mortality") & (rep.variable == "Fe")].iloc[0]
    assert fe.matching_units == "T01" and fe.phase_mean_all_units == 30.0
    wo = rep[(rep.phase == "wear_out") & (rep.variable == "Fe")].iloc[0]
    assert wo.n_match == 0 and wo.nearest_unit == "T03" and wo.nearest_abs_diff == pytest.approx(999 - 90)


def test_debris_rates_cumulative_mode():
    s = pd.DataFrame({"unit_id": "U", "phase": "useful_life", "working_hours": [20, 40, 60, 80, 100.0],
                      "Fe": [10, 20, 30, 40, 50.0], "Cu": [0, 0, 0, 0, 0.0]})
    per, unit = oo.debris_rates(s, ["Fe", "Cu"])
    assert unit.loc["U", "rate_cum"] == pytest.approx(0.5) and unit.loc["U", "rate_ols"] == pytest.approx(0.5)
    assert per["running_rate"].tolist() == pytest.approx([0.5] * 5)
    _, unit_q = oo.debris_rates(s, ["Fe"], q=0.1)  # G_last = 50 + 0.1 (10 + 20 + 30 + 40) = 60
    assert unit_q.loc["U", "rate_cum"] == pytest.approx(0.6)


def test_youden_threshold_rules():
    assert oo.youden_threshold([1, 2], [5, 6]) == pytest.approx(3.5)  # midpoint of the gap
    t = oo.youden_threshold([1, 1], [1, 1])  # no information: J = 0 everywhere
    assert (np.array([1, 1]) >= t).all() or (np.array([1, 1]) < t).all()
    t = oo.youden_threshold([1, 2, 4], [3, 5, 6])  # best J = 2/3 at 2.5 (TPR 1, FPR 1/3) or 4.5 (TPR 2/3, FPR 0)
    assert t in (2.5, 4.5)


def test_loo_threshold_alarms_and_summary():
    units = ["a", "b", "c", "d", "e"]
    feat = pd.Series([0.1, 0.2, 5.0, 6.0, 0.15], index=units)
    pos = pd.Series([False, False, True, True], index=units[:4])
    run = pd.DataFrame({"unit_id": np.repeat(units, 2), "t": [20, 40] * 5,
                        "v": [0.3, 0.1, 0.4, 0.2, 2.0, 5.0, 6.0, 6.0, 0.2, 0.15]})
    al = oo.loo_threshold_alarms(feat, pos, run, "t", "v", extra_negatives=["e"])
    s = oo.alarm_summary(al)
    assert s["detection_rate"] == 1 and s["false_alarm_rate"] == 0 and s["extra_negative_false_alarm_rate"] == 0
    assert al.set_index("unit_id").loc["c", "alarm_time"] == 40  # running 2.0 < tau, 5.0 >= tau
    assert al.set_index("unit_id").loc["d", "alarm_time"] == 20


def test_group_contrast_exact_permutation():
    p, kind = oo.permutation_p([1, 2, 3], [4, 5, 6])
    assert kind == "exact" and p == pytest.approx(2 / 20)  # observed split and its mirror among C(6, 3) = 20
    r = oo.group_contrast([4, 5, 6], [1, 2, 3], log10=False, n_boot=50)
    assert r["cliffs_delta"] == 1 and r["cliffs_delta_lo"] == 1 and r["hedges_g"] > 0
    assert oo.auc([1, 2], [3, 4]) == 1 and oo.auc([1, 2], [1, 2]) == 0.5


def test_composition_features_and_loo():
    s = pd.DataFrame({"unit_id": ["u1"] * 2 + ["u2"] * 2, "Fe": [10, 20, 10, 10.0], "Cr": [1, 2, 5, 5.0],
                      "Al": 1.0, "Cu": 1.0, "Pb": 1.0, "Mn": 1.0})
    X = oo.composition_features(s)
    assert X.loc["u1", "alr_Cr"] == pytest.approx(np.log(0.1)) and X.loc["u2", "alr_Cr"] == pytest.approx(np.log(0.5))
    with pytest.raises(ValueError):
        oo.composition_features(s.assign(Cu=0.0))
    rng = np.random.default_rng(0)
    Xs = np.vstack([rng.normal(0, 0.1, (5, 2)), rng.normal(3, 0.1, (5, 2))])
    y = np.repeat([0, 1], 5)
    p = oo.loo_proba(Xs, y, 3)
    assert p.shape == (10, 3) and np.allclose(p[:, 2], 0) and oo.balanced_accuracy(y, p) == 1


def test_parse_damage_and_teeth():
    assert oo.n_teeth("") == 0 and oo.n_teeth("All teeth") == 28 and oo.n_teeth("6, 7,8") == 3
    d = oo.parse_damage("Dr: de 3t, ip allt / Dn: de3t")
    assert d == {"de_driver": 3, "de_driven": 3, "ip_driver": 28, "ip_driven": 0}
    assert oo.parse_damage("No damage") == {"de_driver": 0, "de_driven": 0, "ip_driver": 0, "ip_driven": 0}
    with pytest.raises(ValueError):
        oo.parse_damage("de 3t")


def test_nasa_transcriptions_are_internally_consistent():
    exps = oo.nasa_experiments().set_index("exp")
    assert len(exps) == 24 and (exps.loc[9:17, "stage_end"] == 0).all() and exps.loc[7, "stage_end"] == 1
    raw = oo._table(oo.NASA_TM2003_READINGS)
    raw["n_de"] = [oo.n_teeth(a) + oo.n_teeth(b) for a, b in zip(raw.de_driver, raw.de_driven)]
    last = raw.groupby("exp").last()
    for e, r in last.iterrows():  # Tables 3.10-3.24 completion rows == Table 3.9 completion columns
        assert float(r.mass_mg) == exps.loc[int(e), "mass_end_mg"] and int(r.reading) == exps.loc[int(e), "rdg_end"]
        assert r.n_de == exps.loc[int(e), "n_de_end"]
    cc = oo.nasa_cross_check()
    assert cc.loc[~cc.agree, ["exp", "reading"]].values.tolist() == [[16, 21446]]  # documented TM-2001 discrepancy
    rd = oo.nasa_readings()
    first = rd[rd.stage == 2].groupby("exp")["reading"].min()
    old = cc[cc.first_destructive_tm2001 == 1].set_index("exp")["reading"]
    assert (first.loc[old.index] == old).all()  # primary labels reproduce the TM-2001 highlighted cells
    alt = oo.nasa_readings(exp5_rdg2566="table_3_14")
    assert alt[alt.stage == 2].groupby("exp")["reading"].min().loc[5] == 2566


def test_load_dataverse_flags_excel_dates(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(list(oo.DATAVERSE_COLUMNS))
    ws.append(["description"] * len(oo.DATAVERSE_COLUMNS))
    ws.append([1, "15/13/4"] + [1.0] * (len(oo.DATAVERSE_COLUMNS) - 2))
    ws.append([2, dt.datetime(2005, 10, 15)] + [2.0] * (len(oo.DATAVERSE_COLUMNS) - 3) + [None])
    wb.save(tmp_path / "d.xlsx")
    df = oo.load_dataverse_oil(tmp_path / "d.xlsx")
    assert df["isocode_excel_date"].tolist() == [False, True] and df["icm_isocode"].tolist() == ["15/13/4", ""]
    assert np.isnan(df.loc[1, "oh_od"]) and df.loc[0, "icm_pc14"] == 1.0


def test_fetch_checksums(tmp_path):
    import hashlib
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "open_oil_fetch", Path(__file__).resolve().parents[1] / "scripts" / "open_oil_fetch.py")
    fetch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_mod)
    src = tmp_path / "src.bin"
    src.write_bytes(b"debris")
    url, good = src.as_uri(), hashlib.md5(b"debris").hexdigest()
    dest = tmp_path / "raw" / "f.bin"
    assert fetch_mod.fetch(url, dest, good, strict=True) == "downloaded" and dest.read_bytes() == b"debris"
    assert fetch_mod.fetch(url, dest, good, strict=True) == "kept"
    with pytest.raises(SystemExit, match="md5 mismatch"):
        fetch_mod.fetch(url, tmp_path / "raw" / "g.bin", "0" * 32, strict=True)
    assert not (tmp_path / "raw" / "g.bin").exists()  # only the .part file is left behind
    assert fetch_mod.fetch(url, tmp_path / "raw" / "h.bin", "0" * 32, strict=False) == "md5 differs"
