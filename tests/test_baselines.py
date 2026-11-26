import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import check_grad
from sklearn.model_selection import ParameterGrid
from torch_geometric.data import Batch, Data

from wearpg.baselines.classical import (CLASSICAL_BASELINES, FOURIER_FRACTAL_PREFIXES, DescriptorClassifier,
                                        MultiViewSVM, aligned_proba, macro_f1, make_particle_baseline)
from wearpg.baselines.deep import (DEEP_BASELINES, ContourCNN, PointNetClassifier, SoftVoteEnsemble,
                                   TimmImageClassifier, _mean_loss, make_deep_baseline, predict_particle_proba,
                                   train_particle_classifier)
from wearpg.baselines.run_level import (NOTE_ETA, RUN_BASELINES, ConditionsOnlyBaseline, DescriptorAggregateBaseline,
                                        PhysicsConditionsBaseline,
                                        OrdinalLogistic, PhysicsOnlyBaseline, aggregate_sample_features,
                                        make_run_baseline)
from wearpg.batch import ModelDims, ParticleBatch
from wearpg.config import deep_merge, load_config
from wearpg.features.descriptors import view_descriptors
from wearpg.schema import RUN_PRED_COLUMNS, SAMPLE_PRED_COLUMNS, SEVERITY_PROB_COLUMNS, WearDataset

CFG = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
ETA = {"AISI52100": 0.6, "CuSn12": 0.2}


# ----------------------------------------------------------------------------- classical


def _blobs(n_per: int, classes, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """shape_/efd_/fractal_ columns carry the class (shape_a = 4 * class + noise); tex_leak is the label itself and
    must be ignored by the default prefixes."""
    rng = np.random.default_rng(seed)
    y = np.repeat(classes, n_per)
    X = pd.DataFrame({"shape_a": 4.0 * y + rng.normal(0, 0.5, len(y)), "efd_0": rng.normal(size=len(y)),
                      "fractal_divider_D": rng.normal(size=len(y)), "tex_leak": 10.0 * y})
    return X, y


@pytest.mark.parametrize("kind", ["svm", "cart"])
def test_descriptor_classifier_prefixes_and_absent_classes(kind):
    X, y = _blobs(15, [0, 1, 3], seed=0)
    y_fit = y.copy()
    y_fit[::7] = -1  # unlabelled rows are ignored
    clf = DescriptorClassifier(kind, n_classes=7, seed=0).fit(X, y_fit)
    assert clf.columns_ == ["shape_a", "efd_0", "fractal_divider_D"]
    Xt, yt = _blobs(10, [0, 1, 3], seed=1)
    p = clf.predict_proba(Xt)
    assert p.shape == (30, 7)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(p[:, [2, 4, 5, 6]] == 0)  # classes absent from training keep their column, probability 0
    assert (p.argmax(axis=1) == yt).mean() > 0.9
    assert clf.params_ == {} and math.isnan(clf.val_score_)  # no validation data -> estimator defaults


def test_descriptor_classifier_grid_selected_on_validation():
    def data(n, noise, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 1, n)
        y = (x > 0.5).astype(int)
        flip = rng.random(n) < noise
        y[flip] = 1 - y[flip]
        return pd.DataFrame({"shape_x": x}), y

    X, y = data(200, 0.3, 0)  # noisy training labels: an unrestricted tree memorises the noise
    Xv, yv = data(200, 0.0, 1)  # clean validation labels reward the stump
    grid = {"max_depth": [None, 1]}
    clf = DescriptorClassifier("cart", n_classes=2, seed=0, grid=grid).fit(X, y, Xv, yv)
    assert clf.params_ == {"max_depth": 1}
    assert clf.val_score_ == pytest.approx(macro_f1(yv, clf.predict_proba(Xv))) and clf.val_score_ > 0.9
    # validating on the training rows rewards memorisation: the selection follows the validation data
    assert DescriptorClassifier("cart", n_classes=2, seed=0, grid=grid).fit(X, y, X, y).params_ == {"max_depth": None}
    # unlabelled validation rows -> no search
    assert DescriptorClassifier("cart", n_classes=2, grid=grid).fit(X, y, Xv, np.full(len(yv), -1)).params_ == {}


def _mv_frame(classes, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Per-view descriptors indexed by '{pid}_v{k}': class c particles have 3 - c views, shape_x = 4c + noise."""
    rng = np.random.default_rng(seed)
    rows, ids, labels = [], [], {}
    for j, c in enumerate(classes):
        pid = f"P{seed}_{j}"
        labels[pid] = int(c)
        for k in range(3 - int(c)):
            rows.append({"shape_x": 4.0 * c + rng.normal(0, 0.5), "efd_0": rng.normal(), "tex_y": float(c)})
            ids.append(f"{pid}_v{k}")
    return pd.DataFrame(rows, index=ids), pd.Series(labels)


def test_multiview_svm_soft_vote_over_available_views():
    V, y = _mv_frame(np.tile([0, 1, 2], 12), seed=0)
    mv = MultiViewSVM(n_classes=5, seed=0).fit(V, y)
    assert mv.columns_ == ["shape_x", "efd_0"]
    assert set(mv.models_) == {0, 1}  # view index 2 was only seen with class 0 -> no SVC
    Vt, yt = _mv_frame(np.tile([0, 1, 2], 4), seed=1)
    ids = list(yt.index)
    p = mv.predict_proba(Vt, ids)
    assert p.shape == (12, 5) and np.all(p[:, 3:] == 0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    pid = ids[0]  # class 0: views 0, 1, 2 -> mean of the view-0 and view-1 SVCs
    x0, x1 = (Vt.loc[[f"{pid}_v{k}"], mv.columns_].to_numpy() for k in (0, 1))
    np.testing.assert_allclose(p[0], (aligned_proba(mv.models_[0], x0, 5) + aligned_proba(mv.models_[1], x1, 5))[0] / 2)
    assert (p.argmax(axis=1) == yt.to_numpy()).mean() > 0.9
    with pytest.raises(ValueError, match="no view"):
        mv.predict_proba(Vt.drop(index=[f"{ids[2]}_v0"]), ids)  # the class-2 particle loses its only view
    mv2 = MultiViewSVM(n_classes=5, seed=0, grid={"C": [0.1, 10.0]}).fit(V, y, Vt, yt)
    assert mv2.params_ in list(ParameterGrid({"C": [0.1, 10.0]}))
    assert mv2.val_score_ == pytest.approx(macro_f1(yt.to_numpy(), mv2.predict_proba(Vt, ids)))


def _particle_desc(ds) -> pd.DataFrame:
    """Real view-0 descriptors (module B, contour only) indexed by particle_id."""
    v = ds.views[ds.views.view_idx == 0]
    return pd.DataFrame([view_descriptors(WearDataset.contour(r), r.um_per_px) for r in v.itertuples()],
                        index=v.particle_id.to_numpy())


def _split(ds) -> tuple[list[str], list[str]]:
    """(train runs = repeats 0 and 1, test runs = repeat 2)."""
    r = ds.runs
    return r.run_id[r.repeat_idx < 2].tolist(), r.run_id[r.repeat_idx == 2].tolist()


def test_fourier_fractal_svm_on_real_descriptors(tiny_dataset):
    ds = tiny_dataset
    desc = _particle_desc(ds)
    y = ds.label_indices().loc[desc.index].to_numpy()
    rep = ds.particle_hierarchy().set_index("particle_id").run_id.loc[desc.index].str[-1].astype(int).to_numpy()
    clf = make_particle_baseline("fourier_fractal_svm", 7, 0, CFG)
    clf.fit(desc[rep == 0], y[rep == 0], desc[rep == 1], y[rep == 1])
    assert clf.params_ in list(ParameterGrid(CFG["baselines"]["grids"]["svm"]))
    assert {c.split("_")[0] + "_" for c in clf.columns_} == set(FOURIER_FRACTAL_PREFIXES)
    p = clf.predict_proba(desc[rep == 2])
    assert p.shape == ((rep == 2).sum(), 7) and np.isfinite(p).all()


def test_make_particle_baseline():
    svm = make_particle_baseline("fourier_fractal_svm", 7, 3, CFG)
    assert isinstance(svm, DescriptorClassifier) and svm.kind == "svm" and svm.seed == 3
    assert svm.grid == CFG["baselines"]["grids"]["svm"] and svm.class_weight == "balanced"
    assert make_particle_baseline("fourier_fractal_cart", 7, 0, CFG).grid == CFG["baselines"]["grids"]["cart"]
    assert isinstance(make_particle_baseline("multiview_svm", 7, 0, CFG), MultiViewSVM)
    no_weights = deep_merge(CFG, {"train": {"class_weighting": "none"}})
    assert make_particle_baseline("fourier_fractal_svm", 7, 0, no_weights).class_weight is None
    with pytest.raises(ValueError):
        make_particle_baseline("resnet18", 7, 0, CFG)
    assert set(CLASSICAL_BASELINES) | set(DEEP_BASELINES) == set(CFG["baselines"]["particle"])


# ----------------------------------------------------------------------------- run level


def test_ordinal_logistic_gradient_and_absent_levels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = np.digitize(X[:, 0] + 0.3 * rng.normal(size=60), [-0.5, 0.5])
    ol = OrdinalLogistic()
    w = rng.normal(size=4) * 0.5
    assert check_grad(lambda v: ol._nll(v, X, y)[0], lambda v: ol._nll(v, X, y)[1], w) < 1e-4
    p = ol.fit(X, y).predict_proba(X)
    assert (p >= 0).all() and np.allclose(p.sum(axis=1), 1.0) and list(ol.classes_) == [0, 1, 2]
    assert ol.coef_[0] > 0 and (ol.predict(X) == y).mean() > 0.8
    p2 = OrdinalLogistic().fit(X[y != 1], y[y != 1]).predict_proba(X)  # level 1 absent from training
    assert p2.shape == (60, 3) and np.isfinite(p2).all() and np.allclose(p2.sum(axis=1), 1.0)
    assert p2[:, 1].max() < 0.5


def _w_sample(ds, desc) -> pd.Series:
    """sample_id -> debris volume at eta = 1 [mm^3] = 0.3 sum_table A^1.5 n_total / n_table / f / 1e9 (module B's A)."""
    p = ds.particles.set_index("particle_id")
    g = (desc.shape_area_um2.reindex(p.index) ** 1.5).groupby(p.sample_id).agg(["sum", "size"])
    s = ds.samples.set_index("sample_id")
    return 0.3 * g["sum"] * s.n_particles_total / g["size"] / s.observation_fraction / 1e9


def _consistent(ds, desc, eta: dict, ferromagnetic=("AISI52100",)) -> WearDataset:
    """ds whose gravimetric run volumes equal the (interval, full-coverage) debris volume / eta[material] exactly."""
    w_run = _w_sample(ds, desc).groupby(ds.samples.set_index("sample_id").run_id).sum()
    rt = ds.run_table()
    v = w_run.reindex(rt.run_id).to_numpy() / rt.material_id.map(eta).to_numpy()
    runs = ds.runs.assign(wear_volume_mm3=v, mass_loss_mg=v * rt.density_g_cm3.to_numpy(),
                          wear_rate_mm3_Nm=v / (ds.runs.load_N * ds.runs.sliding_distance_m))
    specimens = ds.specimens.assign(ferromagnetic=ds.specimens.material_id.isin(ferromagnetic))
    return replace(ds, runs=runs, specimens=specimens)


def _true_log10_k(ds, run_ids) -> np.ndarray:
    return np.log10(ds.runs.set_index("run_id").wear_rate_mm3_Nm.loc[list(run_ids)].to_numpy())


def test_aggregate_sample_features_used_particles_and_volume_index(tiny_dataset):
    ds = tiny_dataset
    desc = _particle_desc(ds)
    agg = aggregate_sample_features(ds, desc)
    assert list(agg.index) == list(ds.samples.sample_id) and (agg.n_used == 4).all()
    # ln(sum_table A^1.5 n_total / n_table / f): the views-based shoelace area equals module B's shape_area_um2
    np.testing.assert_allclose(agg.log_volume_index, np.log(_w_sample(ds, desc) / 0.3 * 1e9).loc[agg.index],
                               rtol=1e-9)
    assert not any(c.startswith("mean_efd_") for c in agg.columns) and "mean_fractal_divider_D" in agg.columns
    s0, s1 = agg.index[:2]
    p = ds.particles
    dropped = ((p.sample_id == s0) & p.particle_id.str.endswith(("_p0", "_p1"))) | (p.sample_id == s1)
    sub = aggregate_sample_features(ds, desc, particle_ids=p.particle_id[~dropped])
    assert sub.n_used[s0] == 2 and sub.n_used[s1] == 0 and sub.loc[s1].drop("n_used").isna().all()
    assert sub.log_volume_index[s0] == pytest.approx(agg.log_volume_index[s0])  # table-based: the bag cancels
    used = p.particle_id[(p.sample_id == s0) & ~dropped]
    assert sub.mean_shape_area_um2[s0] == pytest.approx(desc.shape_area_um2[used].mean())
    pd.testing.assert_frame_equal(sub.iloc[2:], agg.iloc[2:])
    assert (aggregate_sample_features(ds, desc, particle_ids=[]).n_used == 0).all()


def test_physics_only_calibrates_eta_per_magnetic_class(tiny_dataset):
    desc = _particle_desc(tiny_dataset)
    ds = _consistent(tiny_dataset, desc, ETA)
    train, test = _split(ds)
    b = PhysicsOnlyBaseline().fit(ds.subset(train), desc)
    assert b.theta_ == pytest.approx({1: np.log(0.6), 0: np.log(0.2)})  # 1 ferrous AISI52100, 0 non-ferrous CuSn12
    assert b.delta_ == pytest.approx({"AISI52100": 0.0, "CuSn12": 0.0})
    samples, runs = b.predict(ds.subset(test), desc)
    assert list(samples.columns) == ["sample_id", "severity_pred", *SEVERITY_PROB_COLUMNS, "log10_rate_pred",
                                     "log_debris_volume_mm3"]
    assert set(samples.columns) <= set(SAMPLE_PRED_COLUMNS) and set(runs.columns) <= set(RUN_PRED_COLUMNS)
    assert list(runs.run_id) == test and list(samples.sample_id) == list(ds.subset(test).samples.sample_id)
    np.testing.assert_allclose(runs.log10_rate_pred, _true_log10_k(ds, test), rtol=0, atol=1e-9)
    np.testing.assert_allclose(runs.log_debris_volume_mm3, np.log(ds.grav_volume_mm3().loc[test]), atol=1e-9)
    assert (runs.coverage == 1).all() and (runs.note == "").all()
    np.testing.assert_allclose(samples[list(SEVERITY_PROB_COLUMNS)].sum(axis=1), 1.0)
    # trained on the ferrous material only: the non-ferrous class has no eta -> NaN outputs + note
    b2 = PhysicsOnlyBaseline().fit(ds.subset([r for r in train if "AISI" in r]), desc)
    assert set(b2.theta_) == {1}
    s2, r2 = b2.predict(ds.subset([r for r in test if "CuSn12" in r]), desc)
    assert (r2.note == NOTE_ETA).all() and r2[["log10_rate_pred", "log_debris_volume_mm3"]].isna().all().all()
    assert (r2.coverage == 1).all()  # coverage is still reported (observation, not eta)
    assert s2[["log10_rate_pred", "log_debris_volume_mm3", "sev_p_mild"]].isna().all().all()
    assert (s2.severity_pred == -1).all()


def test_physics_only_material_offset_within_magnetic_class(tiny_dataset):
    desc = _particle_desc(tiny_dataset)
    ds = _consistent(tiny_dataset, desc, {"AISI52100": 0.5, "CuSn12": 0.2}, ferromagnetic=())  # one class
    train, test = _split(ds)
    b = PhysicsOnlyBaseline().fit(ds.subset(train), desc)
    theta = (np.log(0.5) + np.log(0.2)) / 2
    assert b.theta_ == pytest.approx({0: theta})
    assert b.delta_ == pytest.approx({"AISI52100": np.log(0.5) - theta, "CuSn12": np.log(0.2) - theta})
    np.testing.assert_allclose(b.predict(ds.subset(test), desc)[1].log10_rate_pred, _true_log10_k(ds, test),
                               atol=1e-9)
    # a material unseen in training but sharing the magnetic class: transferable, delta = 0 -> eta = 0.5
    cu = [r for r in test if "CuSn12" in r]
    _, r2 = PhysicsOnlyBaseline().fit(ds.subset([r for r in train if "AISI" in r]), desc).predict(ds.subset(cu), desc)
    assert (r2.note == "").all()
    np.testing.assert_allclose(r2.log10_rate_pred, _true_log10_k(ds, cu) + np.log10(0.2 / 0.5), atol=1e-9)


def test_physics_only_area_ratio_ht_estimator(tiny_dataset):
    desc = _particle_desc(tiny_dataset)
    ds = _consistent(tiny_dataset, desc, ETA)
    train, test = _split(ds)
    b = PhysicsOnlyBaseline().fit(ds.subset(train), desc)
    dt = ds.subset(test)
    s0, _ = b.predict(dt, desc)
    mat = dt.samples.run_id.map(dt.run_table().set_index("run_id").material_id)
    expected = np.log(_w_sample(dt, desc).loc[s0.sample_id].to_numpy()) - np.log(mat.map(ETA).to_numpy())
    np.testing.assert_allclose(s0.log_debris_volume_mm3, expected, atol=1e-9)
    sid = s0.sample_id[0]
    bag = dt.particles.particle_id[dt.particles.sample_id == sid][:3]
    s1, _ = b.predict(dt, desc.drop(index=bag))  # bag = 1 of 4 table rows: the area ratio cancels it
    np.testing.assert_allclose(s1.log_debris_volume_mm3, s0.log_debris_volume_mm3, atol=1e-12)
    samples = dt.samples.copy()
    samples.loc[samples.sample_id == sid, "n_particles_total"] *= 2  # twice as many detected as stored
    s2, _ = b.predict(replace(dt, samples=samples), desc)
    diff = (s2.log_debris_volume_mm3 - s0.log_debris_volume_mm3).to_numpy()
    assert diff[0] == pytest.approx(np.log(2)) and np.allclose(diff[1:], 0)


def test_physics_only_interval_and_cumulative_run_volumes(tiny_dataset):
    desc = _particle_desc(tiny_dataset)
    ds = _consistent(tiny_dataset, desc, ETA)
    train, test = _split(ds)
    b = PhysicsOnlyBaseline().fit(ds.subset(train), desc)
    run = test[0]
    dt = ds.subset([run])
    eta = ETA["AISI52100" if "AISI" in run else "CuSn12"]
    s = dt.samples.set_index("sample_id")
    w0, w1 = _w_sample(dt, desc).loc[s.index].to_numpy()
    length = float(dt.runs.sliding_distance_m.iloc[0])
    q0 = float(s.sampled_oil_ml.iloc[0] / dt.runs.reservoir_volume_ml.iloc[0])
    ds0 = float(s.sliding_distance_end_m.iloc[0] - s.sliding_distance_start_m.iloc[0])
    end0, end1 = s.sliding_distance_end_m.to_numpy(float)
    no_s1 = desc.drop(index=dt.particles.particle_id[dt.particles.sample_id == s.index[1]])
    cum = replace(dt, runs=dt.runs.assign(debris_sampling="cumulative"))
    cases = [(dt, desc, w0 + w1, 1.0),  # interval: sum of interval volumes
             (dt, no_s1, w0 * length / ds0, 0.5),  # interval, s1 unobserved: extrapolated by covered distance
             (cum, desc, (w1 + q0 * w0) * length / end1, 1.0),  # cumulative: (M_last + q M_0) L / s_end(last)
             (cum, no_s1, w0 * length / end0, 0.5)]  # cumulative, last observed = s0
    for d, de, v_eta1, cov in cases:
        _, runs = b.predict(d, de)
        assert runs.coverage.iloc[0] == pytest.approx(cov)
        assert runs.log_debris_volume_mm3.iloc[0] == pytest.approx(np.log(v_eta1 / eta))
    samples, _ = b.predict(cum, desc)  # cumulative: inventories, no interval rate, no severity
    assert samples.log10_rate_pred.isna().all() and (samples.severity_pred == -1).all()
    assert np.isfinite(samples.log_debris_volume_mm3).all()
    samples, _ = b.predict(dt, no_s1)
    assert np.isnan(samples.log_debris_volume_mm3.iloc[1]) and samples.severity_pred.iloc[1] == -1
    assert np.isfinite(samples.log10_rate_pred.iloc[0]) and samples.severity_pred.iloc[0] >= 0


def test_physics_only_severity_tracks_interval_rate(tiny_dataset):
    desc = _particle_desc(tiny_dataset)
    samples = tiny_dataset.samples.copy()
    samples["n_particles_total"] = 4 * 1000 ** samples.severity  # debris amount grows 1000x per severity level
    ds = _consistent(replace(tiny_dataset, samples=samples), desc, ETA)
    train, test = _split(ds)
    b = PhysicsOnlyBaseline().fit(ds.subset(train), desc)
    s, _ = b.predict(ds.subset(test), desc)
    truth = ds.samples.set_index("sample_id").severity.loc[s.sample_id].to_numpy()
    order = np.argsort(s.log10_rate_pred.to_numpy())
    assert b.severity_[-1].coef_[0] > 0
    assert np.all(np.diff(s.severity_pred.to_numpy()[order]) >= 0)  # ordinal in the interval debris rate
    assert np.all(np.diff(1.0 - s.sev_p_mild.to_numpy()[order]) >= 0)  # P(severity >= 1) rises with k-hat
    pred = s.severity_pred.to_numpy()
    assert (pred[truth == 0] == 0).all() and (pred[truth == 2] == 2).all() and (pred == truth).mean() >= 0.75


def test_conditions_only(tiny_dataset):
    ds = tiny_dataset
    desc = _particle_desc(ds)
    train, test = _split(ds)
    b = ConditionsOnlyBaseline().fit(ds.subset(train), desc)
    s, r = b.predict(ds.subset(test), desc)
    assert np.abs(r.log10_rate_pred - _true_log10_k(ds, test)).mean() < 0.3
    severe = s.sample_id.map(ds.samples.set_index("sample_id").severity).to_numpy() > 0
    assert ((s.severity_pred.to_numpy() >= 1) == severe).all()
    assert s[["log10_rate_pred", "log_debris_volume_mm3"]].isna().all().all() and r.log_debris_volume_mm3.isna().all()
    gone = ds.particles.particle_id[ds.particles.sample_id == s.sample_id[0]]
    s2, _ = b.predict(ds.subset(test), desc.drop(index=gone))  # sample outputs only for observed samples
    assert s2.severity_pred[0] == -1 and np.isnan(s2.sev_p_mild[0]) and (s2.severity_pred[1:] >= 0).all()
    cu = [r for r in test if "CuSn12" in r]  # material unseen in training -> one-hot all zero, still predicted
    b3 = ConditionsOnlyBaseline().fit(ds.subset([r for r in train if "AISI" in r]), desc)
    assert np.isfinite(b3.predict(ds.subset(cu), desc)[1].log10_rate_pred).all()


def test_physics_conditions_fusion(tiny_dataset):
    ds = tiny_dataset
    desc = _particle_desc(ds)
    train, test = _split(ds)
    b = PhysicsConditionsBaseline().fit(ds.subset(train), desc)
    m = b.moments_
    assert m["n"] == len(train)  # every train run cross-fitted once
    w = (m["S_cc"] - m["S_pc"]) / (m["S_pp"] + m["S_cc"] - 2 * m["S_pc"])
    assert b.weight_ == pytest.approx(np.clip(w, 0.0, 1.0))
    s, r = b.predict(ds.subset(test), desc)
    sp, rp = b.physics_.predict(ds.subset(test), desc)
    rc = b.conditions_.predict(ds.subset(test), desc)[1]
    np.testing.assert_allclose(r.log10_rate_pred, b.weight_ * rp.log10_rate_pred + (1 - b.weight_) * rc.log10_rate_pred)
    pd.testing.assert_frame_equal(s, sp)
    np.testing.assert_allclose(r.log_debris_volume_mm3, rp.log_debris_volume_mm3)
    exact = _consistent(ds, desc, {"AISI52100": 0.6, "CuSn12": 0.2})  # debris volume = gravimetry: all weight on it
    assert PhysicsConditionsBaseline().fit(exact.subset(train), desc).weight_ == pytest.approx(1.0, abs=1e-6)
    assert isinstance(make_run_baseline("physics_conditions", CFG, seed=1), PhysicsConditionsBaseline)


@pytest.mark.parametrize("kind", ["rf", "ridge"])
def test_descriptor_aggregate(tiny_dataset, kind):
    ds = tiny_dataset
    # tiny_dataset particles are near-identical circles, so most aggregates are pure jitter that would swamp a ridge
    # fitted on 4 runs (p >> n); keep a few real columns and make severe-regime debris coarser
    desc = _particle_desc(ds)[["shape_log_ecd_um", "shape_area_um2", "shape_aspect_ratio", "fractal_divider_D"]]
    severe_p = ds.particle_hierarchy().set_index("particle_id").load_N.loc[desc.index].to_numpy() > 60
    desc.loc[severe_p, "shape_log_ecd_um"] += 3.0
    desc.loc[severe_p, "shape_area_um2"] *= np.exp(6.0)
    runs = ds.runs.run_id
    fit_runs, val, test = (runs[ds.runs.repeat_idx == k].tolist() for k in (0, 1, 2))
    grid = CFG["baselines"]["grids"]["rf"]
    b = DescriptorAggregateBaseline(kind, seed=0, grid=grid).fit(ds.subset(fit_runs), desc, ds.subset(val))
    s, r = b.predict(ds.subset(test), desc)
    severe_r = r.run_id.str.contains("_120_").to_numpy()
    assert r.log10_rate_pred[severe_r].min() > r.log10_rate_pred[~severe_r].max()
    severe_s = s.sample_id.map(ds.samples.set_index("sample_id").severity).to_numpy() > 0
    assert ((s.severity_pred.to_numpy() >= 1) == severe_s).all()
    assert np.isfinite(s.log10_rate_pred).all()  # interval targets exist in tiny_dataset
    assert s.log_debris_volume_mm3.isna().all() and r.log_debris_volume_mm3.isna().all() and (r.coverage == 1).all()
    if kind == "rf":
        assert set(b.params_) == {"run", "sample_rate", "severity"}
        assert all(p in list(ParameterGrid(grid)) for p in b.params_.values())
        assert DescriptorAggregateBaseline("rf", grid=grid).fit(ds.subset(fit_runs), desc).params_["run"] == {}
    else:
        assert b.params_ == {"run": {}, "sample_rate": {}, "severity": {}}
        no_iv = replace(ds, samples=ds.samples.assign(interval_wear_volume_mm3=np.nan))  # no interval targets
        b2 = DescriptorAggregateBaseline(kind).fit(no_iv.subset(fit_runs), desc)
        assert b2.models_["sample_rate"] is None and b2.models_["run"] is not None
        s2, _ = b2.predict(no_iv.subset(test), desc)
        assert s2.log10_rate_pred.isna().all() and (s2.severity_pred >= 0).all()


def test_make_run_baseline():
    assert isinstance(make_run_baseline("physics_only", CFG), PhysicsOnlyBaseline)
    assert isinstance(make_run_baseline("conditions_only", CFG), ConditionsOnlyBaseline)
    rf = make_run_baseline("descriptor_agg_rf", CFG, seed=2)
    assert rf.kind == "rf" and rf.seed == 2 and rf.grid == CFG["baselines"]["grids"]["rf"]
    assert make_run_baseline("descriptor_agg_ridge", CFG).kind == "ridge"
    assert make_run_baseline("physics_only", CFG).min_coverage == CFG["physics"]["min_run_coverage"]
    with pytest.raises(ValueError):
        make_run_baseline("nope", CFG)
    assert set(RUN_BASELINES) == set(CFG["baselines"]["run"])


# ----------------------------------------------------------------------------- deep

DIMS = ModelDims(node_dim=5, edge_dim=3, particle_feat_dim=4, sample_feat_dim=0, cond_dim=0, n_classes=7,
                 signal_channels=3, contour_points=32)
DEEP_CFG = deep_merge(CFG, {"model": {"hidden": 32, "gnn_layers": 2}})


def _batch(n_p: int = 6, seed: int = 0, target=None) -> ParticleBatch:
    """Random ParticleBatch; particle 0 has a second view graph."""
    g = torch.Generator().manual_seed(seed)
    views = list(range(n_p)) + [0]
    graphs = []
    for _ in views:
        n = int(torch.randint(6, 12, (1,), generator=g))
        e = torch.randint(0, n, (2, 2 * n), generator=g)
        graphs.append(Data(x=torch.randn(n, DIMS.node_dim, generator=g), edge_index=e,
                           edge_attr=torch.randn(e.shape[1], DIMS.edge_dim, generator=g)))
    return ParticleBatch(
        graphs=Batch.from_data_list(graphs), view_to_particle=torch.tensor(views),
        contour_points=torch.randn(n_p, 32, 2, generator=g),
        contour_signal=torch.randn(n_p, DIMS.signal_channels, 32, generator=g),
        crops=torch.rand(n_p, 4, 64, 64, generator=g),
        particle_feats=torch.randn(n_p, DIMS.particle_feat_dim, generator=g),
        composition=torch.zeros(n_p, 11), composition_mask=torch.zeros(n_p, dtype=torch.bool),
        mech_target=torch.randint(-1, 7, (n_p,), generator=g) if target is None else target,
        mech_soft=None, particle_ids=[f"p{seed}_{i}" for i in range(n_p)])


@pytest.mark.parametrize("name", ["particle_gnn", "contour_cnn", "pointnet", "fecnn"])
def test_deep_baselines_forward_backward_and_globals(name):
    torch.manual_seed(0)
    model = make_deep_baseline(name, DIMS, DEEP_CFG)
    b = _batch()
    out = model(b)
    assert out.shape == (6, 7) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():  # the standardized globals reach the classifier
        assert not torch.allclose(model(b), model(replace(b, particle_feats=b.particle_feats + 1.0)))


@pytest.mark.parametrize("name,dim", [("resnet18", 512), ("convnext_tiny", 768), ("vit_tiny_patch16_224", 192),
                                      ("efficientnet_b0", 1280)])
def test_timm_backbones_on_cpu(name, dim):
    torch.manual_seed(0)
    model = TimmImageClassifier(name, DIMS, pretrained=False).eval()
    assert model.backbone.num_features == dim
    with torch.no_grad():
        out = model(_batch(n_p=3))
    assert out.shape == (3, 7) and torch.isfinite(out).all()


def test_vit_ensemble_is_a_soft_vote():
    torch.manual_seed(0)
    ens = make_deep_baseline("vit_ensemble", DIMS, DEEP_CFG).eval()
    assert isinstance(ens, SoftVoteEnsemble)
    assert [type(m.backbone).__name__ for m in ens.members] == ["VisionTransformer", "ResNet", "EfficientNet"]
    b = _batch(n_p=3)
    with torch.no_grad():
        mean_p = torch.stack([m(b).softmax(-1) for m in ens.members]).mean(0)
        torch.testing.assert_close(ens(b).softmax(-1), mean_p)


def test_make_deep_baseline_names():
    assert isinstance(make_deep_baseline("resnet18", DIMS, DEEP_CFG), TimmImageClassifier)
    no_rot = deep_merge(DEEP_CFG, {"data": {"augment": {"point_rotation": False}}})
    assert make_deep_baseline("pointnet", DIMS, no_rot).rotate is False
    with pytest.raises(ValueError):
        make_deep_baseline("fourier_fractal_svm", DIMS, DEEP_CFG)


def test_contour_cnn_is_start_point_invariant():
    torch.manual_seed(0)
    model = ContourCNN(DIMS, hidden=32).eval()
    b = _batch()
    with torch.no_grad():
        torch.testing.assert_close(model(b), model(replace(b, contour_signal=b.contour_signal.roll(7, dims=-1))))


def test_pointnet_point_order_invariance_and_rotation_augmentation():
    torch.manual_seed(0)
    model = PointNetClassifier(DIMS, hidden=32, dropout=0.0, rotate=True).eval()
    b = _batch()
    with torch.no_grad():
        torch.testing.assert_close(model(b), model(replace(b, contour_points=b.contour_points[:, torch.randperm(32)])))
        model.train()
        assert not torch.allclose(model(b), model(b))  # a new random rotation per call in training
        model.rotate = False
        torch.testing.assert_close(model(b), model(b))


def _learnable(n: int, seed: int, shift: int = 0) -> ParticleBatch:
    """Labelled batch: class c in 0..2 raises contour-signal channel c; labels are (c + shift) % 3."""
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, 3, (n,), generator=g)
    sig = torch.randn(n, 3, 32, generator=g) * 0.5
    sig[torch.arange(n), c] += 1.5
    return replace(_batch(n, seed, target=(c + shift) % 3), contour_signal=sig)


def test_train_particle_classifier_learns():
    torch.manual_seed(0)
    cfg = deep_merge(DEEP_CFG, {"train": {"particle_epochs": 10, "lr": 1e-2}})
    val = _learnable(48, 10)
    val.mech_target[:5] = -1  # unlabelled particles are ignored
    model = ContourCNN(DIMS, hidden=32)
    out = train_particle_classifier(model, [_learnable(32, s) for s in range(4)], [val], cfg, "cpu",
                                    class_weights=np.ones(7, dtype=np.float32))
    vals = [h["val_loss"] for h in out["history"]]
    assert out["best_epoch"] == int(np.argmin(vals)) and out["best_val_loss"] == pytest.approx(min(vals))
    ids, p = predict_particle_proba(model, [val], "cpu")
    y = val.mech_target.numpy()
    assert ids == val.particle_ids and p.shape == (48, 7) and np.allclose(p.sum(axis=1), 1.0)
    assert (p.argmax(axis=1)[y >= 0] == y[y >= 0]).mean() > 0.8


def test_train_particle_classifier_early_stops_and_restores_best():
    torch.manual_seed(0)
    cfg = deep_merge(DEEP_CFG, {"train": {"particle_epochs": 30, "lr": 1e-2, "early_stopping_patience": 3}})
    val = [_learnable(48, 10, shift=1)]  # validation labels contradict training: learning raises the val loss
    model = ContourCNN(DIMS, hidden=32)
    out = train_particle_classifier(model, [_learnable(32, s) for s in range(4)], val, cfg, "cpu")
    vals = [h["val_loss"] for h in out["history"]]
    assert len(vals) == out["best_epoch"] + 4 < 30  # stopped `patience` epochs after the best one
    assert min(vals) < vals[-1]
    restored = _mean_loss(model, val, "cpu", None, cfg["train"]["label_smoothing"])
    assert restored == pytest.approx(out["best_val_loss"], rel=1e-5)


def test_train_without_validation_keeps_last_epoch():
    cfg = deep_merge(DEEP_CFG, {"train": {"particle_epochs": 3}})
    out = train_particle_classifier(ContourCNN(DIMS, hidden=32), [_learnable(16, 0)], None, cfg, "cpu")
    assert len(out["history"]) == 3 and out["best_epoch"] == 2 and math.isnan(out["best_val_loss"])


def test_soft_vote_members_trained_independently():
    torch.manual_seed(0)
    cfg = deep_merge(DEEP_CFG, {"train": {"particle_epochs": 2}})
    ens = SoftVoteEnsemble([ContourCNN(DIMS, hidden=32), PointNetClassifier(DIMS, hidden=32)])
    out = train_particle_classifier(ens, [_learnable(16, 0)], [_learnable(16, 1)], cfg, "cpu")
    assert len(out["members"]) == 2 and all(len(m["history"]) == 2 for m in out["members"])
    assert np.isfinite(out["best_val_loss"])
