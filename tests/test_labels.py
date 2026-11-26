import warnings

import numpy as np
import pandas as pd
import pytest

from wearpg.labels import (adjudicate, agreement_report, cohens_kappa, fleiss_kappa, krippendorff_alpha_nominal,
                           label_index, soft_labels, validity_report)
from wearpg.schema import MECHANISMS, WearDataset


def _long(table: dict) -> pd.DataFrame:
    """{rater: [label or None per unit]} -> labels-like long frame."""
    rows = [(f"u{u}", r, str(v)) for r, vals in table.items() for u, v in enumerate(vals) if v is not None]
    return pd.DataFrame(rows, columns=["particle_id", "rater_id", "label"])


def test_fleiss_kappa_wikipedia_example():
    # Fleiss (1971) / Wikipedia worked example: 10 items, 14 raters, 5 categories -> kappa = 0.210
    t = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
                  [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    assert fleiss_kappa(t) == pytest.approx(0.2099, abs=5e-4)
    with pytest.raises(ValueError):
        fleiss_kappa([[2, 0], [1, 2]])  # unequal raters per item


def test_cohens_kappa_known_values():
    a = ["a"] * 4 + ["b"] * 3 + ["c"] * 2
    assert cohens_kappa(a, a) == pytest.approx(1.0)
    # Wikipedia 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po 0.7, pe 0.5, kappa 0.4
    r1 = ["y"] * 25 + ["n"] * 25
    r2 = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(r1, r2) == pytest.approx(0.4)
    assert cohens_kappa(r1 + ["x"], r2 + ["y"], labels=["y", "n"]) == pytest.approx(0.4)  # outside labels dropped
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(cohens_kappa(["a", "a"], ["a", "a"]))  # undefined, no 0/0 warning
        assert np.isnan(cohens_kappa([], []))


def test_krippendorff_alpha_known_values():
    # Krippendorff (2011) "Computing Krippendorff's alpha-reliability": 4 observers x 12 units -> 0.743
    k = {"A": [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None], "B": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
         "C": [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None], "D": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]}
    assert krippendorff_alpha_nominal(_long(k)) == pytest.approx(0.743, abs=5e-4)
    # two complete raters: 1 - alpha = (1 - Scott's pi) (n - 1) / n with n = 2 N pairable values
    r1 = list("aabbbcca")
    r2 = list("abbbcccc")
    po = np.mean(np.array(r1) == np.array(r2))
    pooled = pd.Series(r1 + r2).value_counts(normalize=True).to_numpy()
    pi = (po - (pooled**2).sum()) / (1 - (pooled**2).sum())
    n = 2 * len(r1)
    assert krippendorff_alpha_nominal(_long({"A": r1, "B": r2})) == pytest.approx(1 - (1 - pi) * (n - 1) / n)


def test_agreement_report_two_raters_excludes_reference(tiny_dataset):
    lab = tiny_dataset.labels
    ref = lab[lab.rater_id == "A"].assign(rater_id="SEMREF", label="oxide")  # must be ignored
    rep = agreement_report(pd.concat([lab, ref], ignore_index=True))
    n = tiny_dataset.particles.shape[0]
    assert rep["raters"] == ["A", "B"] and rep["kappa_method"] == "cohen"
    assert rep["n_items"] == n and rep["n_multi_rated"] == n and rep["n_kappa_items"] == n
    w = lab.pivot(index="particle_id", columns="rater_id", values="label")
    assert rep["kappa"] == pytest.approx(cohens_kappa(w.A, w.B, labels=MECHANISMS))
    assert rep["pairwise_kappa"] == {"A|B": pytest.approx(rep["kappa"])}
    conf = np.array(rep["rater_confusion"])
    assert conf.shape == (7, 7) and conf.sum() == n and np.trace(conf) == (w.A == w.B).sum()
    # 2 raters: specific agreement of class c = 2 n_cc / (n_c. + n_.c)
    c = MECHANISMS.index("rubbing")
    assert rep["per_class_agreement"]["rubbing"] == pytest.approx(2 * conf[c, c] / (conf[c].sum() + conf[:, c].sum()))
    assert np.isfinite(rep["krippendorff_alpha"])


def test_agreement_report_fleiss_on_modal_rater_count():
    t = {"A": ["rubbing", "cutting", "oxide", "laminar", "oxide", "cutting"],
         "B": ["rubbing", "cutting", "spherical", "laminar", "oxide", None],
         "C": ["rubbing", "laminar", "oxide", "laminar", None, None]}
    rep = agreement_report(_long(t))
    assert rep["kappa_method"] == "fleiss" and rep["raters"] == ["A", "B", "C"]
    assert rep["n_items"] == 6 and rep["n_multi_rated"] == 5 and rep["n_kappa_items"] == 4  # modal count = 3
    counts = np.array([[3, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 2], [0, 0, 0, 0, 3, 0, 0]])
    assert rep["kappa"] == pytest.approx(fleiss_kappa(counts))
    assert set(rep["pairwise_kappa"]) == {"A|B", "A|C", "B|C"}
    assert np.isnan(rep["per_class_agreement"]["fatigue_chunk"])  # never used


def test_validity_report():
    adj = pd.DataFrame({"particle_id": ["p0", "p1", "p2", "p3"], "label": ["rubbing", "cutting", "oxide", "oxide"]})
    lab = pd.DataFrame({"particle_id": ["p0", "p1", "p2", "p9", "p0"], "rater_id": ["SEMREF"] * 4 + ["A"],
                        "label": ["rubbing", "cutting", "spherical", "oxide", "laminar"]})
    rep = validity_report(lab, adj)
    assert rep["n"] == 3 and rep["accuracy"] == pytest.approx(2 / 3)
    assert rep["kappa"] == pytest.approx(cohens_kappa(["rubbing", "cutting", "oxide"], ["rubbing", "cutting", "spherical"]))
    empty = validity_report(lab[lab.rater_id == "A"], adj)
    assert empty["n"] == 0 and np.isnan(empty["kappa"])


def test_adjudicate_methods_and_schema():
    t = {"A": ["rubbing", "rubbing", "cutting", "oxide", "oxide"],
         "B": ["rubbing", "laminar", "cutting", "laminar", None],
         "C": [None, None, "laminar", None, None]}
    lab = pd.concat([_long(t), pd.DataFrame({"particle_id": ["u0"], "rater_id": ["SEMREF"], "label": ["oxide"]})])
    out = adjudicate(lab, adjudicator={"u1": "laminar", "u3": "spherical"}).set_index("particle_id")
    assert out.loc["u0"].tolist() == ["rubbing", "unanimous", 2, 1.0]  # SEMREF not a vote
    assert out.loc["u1"].tolist() == ["laminar", "adjudicated", 2, 0.5]
    assert out.loc["u2"].tolist() == ["cutting", "majority", 3, pytest.approx(2 / 3)]
    assert out.loc["u3"].tolist() == ["spherical", "adjudicated", 2, 0.0]  # adjudicator overrides both
    assert out.loc["u4"].tolist() == ["oxide", "single", 1, 1.0]
    # adjudicator as a rater id: its rows only resolve disagreements; unresolved ones are omitted
    adj_rows = pd.DataFrame({"particle_id": ["u1"], "rater_id": ["ADJ"], "label": ["rubbing"]})
    out2 = adjudicate(pd.concat([_long(t), adj_rows])).set_index("particle_id")  # ADJ unnamed -> ordinary vote
    assert out2.loc["u1"].tolist() == ["rubbing", "majority", 3, pytest.approx(2 / 3)]
    assert "u3" not in out2.index  # unresolved disagreement stays unlabelled
    out3 = adjudicate(pd.concat([_long(t), adj_rows]), adjudicator="ADJ").set_index("particle_id")
    assert out3.loc["u1", "label"] == "rubbing" and out3.loc["u1", "n_raters"] == 2
    with pytest.raises(ValueError):
        adjudicate(_long(t), adjudicator={"u1": "not_a_class", "u3": "oxide"})


def test_adjudicate_reproduces_tiny_dataset(tiny_dataset):
    ds = tiny_dataset
    ref = ds.labels_adjudicated
    decisions = ref.loc[ref.method == "adjudicated"].set_index("particle_id")["label"]
    out = adjudicate(ds.labels, adjudicator=decisions)
    pd.testing.assert_frame_equal(out.sort_values("particle_id").reset_index(drop=True),
                                  ref.sort_values("particle_id").reset_index(drop=True))
    assert WearDataset(**{t: getattr(ds, t) for t in ("specimens", "runs", "samples", "images", "particles",
                                                       "views", "labels")}, labels_adjudicated=out).validate() == []


def test_soft_labels():
    lab = _long({"A": ["rubbing", "cutting"], "B": ["rubbing", "oxide"]})
    lab = pd.concat([lab, pd.DataFrame({"particle_id": ["u0"], "rater_id": ["SEMREF"], "label": ["oxide"]})])
    p = soft_labels(lab, ["u1", "u0", "zz"])
    assert p.shape == (3, 7) and np.allclose(p.sum(axis=1), 1.0)
    assert p[0, MECHANISMS.index("cutting")] == 0.5 and p[0, MECHANISMS.index("oxide")] == 0.5
    assert p[1, 0] == 1.0  # SEMREF excluded
    assert np.allclose(p[2], 1 / 7)  # unrated -> uniform
    ps = soft_labels(lab, ["u0"], smoothing=0.1)
    assert ps[0, 0] == pytest.approx(0.9 + 0.1 / 7) and ps[0, 1] == pytest.approx(0.1 / 7)


def test_label_index():
    assert label_index(["rubbing", "oxide", "bogus", ""]).tolist() == [0, 6, -1, -1]
    assert label_index([], classes=("a",)).dtype == np.int64
