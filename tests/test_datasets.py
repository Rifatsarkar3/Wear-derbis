import copy
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from wearpg.batch import RunBatch
from wearpg.config import load_config
from wearpg.datasets import (PARTICLE_GRAPH_SELECTION, SAMPLE_FEATURES, GroupedRunSampler, RunBagDataset,
                             Standardizer, baseline_global_columns, build_cache, class_weights, collate_runs,
                             dataset_key, fit_scalers, magnetic_index, make_particle_loader, make_run_loader,
                             material_index, model_dims, mono_groups, particle_feature_columns)
from wearpg.features.contour import signed_area
from wearpg.graphs import selected_dims, unpack_graph
from wearpg.schema import WearDataset
from wearpg.splits import particles_for_runs

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(*overrides):
    return load_config(ROOT / "configs" / "default.yaml", list(overrides))


@pytest.fixture(scope="module")
def shared_cache_dir(tmp_path_factory):
    """The tiny dataset is deterministic, so every test's copy has the same dataset_key: one cache build serves all."""
    return tmp_path_factory.mktemp("tiny_cache")


@pytest.fixture
def tiny(tiny_dataset_on_disk, shared_cache_dir):
    """(dataset, cache, cfg) on the tiny on-disk dataset: 12 runs x 2 samples x 4 single-view particles."""
    cfg = make_cfg()
    return tiny_dataset_on_disk, build_cache(tiny_dataset_on_disk, cfg, cache_dir=shared_cache_dir), cfg


def runs_of(ds):
    return sorted(ds.runs.run_id)


def dihedral(c):
    """The 8 rot90/flip variants of a (C, H, W) array."""
    rots = [np.rot90(c, k, axes=(1, 2)) for k in range(4)]
    return rots + [r[:, :, ::-1] for r in rots]


# ----------------------------------------------------------------------------- cache


def test_dataset_key_stable_and_content_sensitive(tiny_dataset_on_disk, tiny_dataset):
    ds = tiny_dataset_on_disk
    key = dataset_key(ds)
    assert key.startswith("tiny_") and key == dataset_key(WearDataset.load(ds.root))
    p = ds.root / "particles.csv"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert dataset_key(ds) != key
    with pytest.raises(ValueError):
        dataset_key(tiny_dataset)  # not saved: no root


def test_cache_contents_and_disk_round_trip(tiny_dataset_on_disk, tmp_path):
    ds, cfg = tiny_dataset_on_disk, make_cfg()
    cache = build_cache(ds, cfg, cache_dir=tmp_path / "cache")
    n, nodes = len(ds.particles), cfg["graph"]["n_nodes"]
    assert cache.particle_ids == ds.particles.particle_id.tolist() == cache.descriptors.index.tolist()
    assert cache.crops.shape == (n, 64, 64, 4) and cache.crops.dtype == np.uint8
    assert cache.contour_points.shape == (n, nodes, 2) and cache.contour_signal.shape == (n, 8, nodes)
    assert set(cache.view_descriptors.index) == set(ds.views.view_id) == set(cache.graphs["view_ids"])
    assert {"particle_id", "view_idx"} <= set(cache.view_descriptors.columns)
    assert not any(c.startswith("efd_") for c in particle_feature_columns(cache.descriptors.columns, cfg))
    assert np.isfinite(cache.descriptors.to_numpy()).all()
    v0 = ds.views.set_index("particle_id")
    area = [signed_area(WearDataset.contour(v0.loc[p])) * v0.at[p, "um_per_px"] ** 2 for p in cache.particle_ids]
    np.testing.assert_allclose(cache.descriptors.shape_area_um2, area, rtol=1e-9)
    assert (cache.crops[..., 3] > 0).mean(axis=(1, 2)).min() > 0.2  # the mask channel marks the particle
    g = unpack_graph(cache.graphs, cache.particle_ids[0] + "_v0")  # cached signal / points = the graph's boundary
    np.testing.assert_allclose(cache.contour_signal[0].T, g["x"][:nodes, :8].numpy(), atol=1e-5)
    np.testing.assert_allclose(cache.contour_points[0], g["pos"][:nodes].numpy(), atol=1e-6)

    again = build_cache(ds, cfg, cache_dir=Path(cache.meta["base_dir"]).parents[1])
    assert cache.meta["rebuilt"] == {"base": True, "graphs": True}
    assert again.meta["rebuilt"] == {"base": False, "graphs": False}
    pd.testing.assert_frame_equal(again.descriptors, cache.descriptors)
    pd.testing.assert_frame_equal(again.view_descriptors, cache.view_descriptors)
    for k in ("crops", "contour_points", "contour_signal"):
        np.testing.assert_array_equal(getattr(again, k), getattr(cache, k))
    for vid in cache.graphs["view_ids"]:
        a, b = unpack_graph(again.graphs, vid), unpack_graph(cache.graphs, vid)
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_cache_key_invalidation(tiny_dataset_on_disk, tmp_path):
    small = ("graph.n_nodes=32", "graph.interior_nodes=4")  # cheaper graphs; the keys are what is tested
    ds, cfg, cdir = tiny_dataset_on_disk, make_cfg(*small), tmp_path / "cache"
    cache = build_cache(ds, cfg, cache_dir=cdir)
    vid = cache.particle_ids[0] + "_v0"
    c2 = build_cache(ds, make_cfg(*small, "graph.interior_nodes=2"), cache_dir=cdir)  # graph parameter: graphs only
    assert c2.meta["rebuilt"] == {"base": False, "graphs": True} and c2.meta["graph_path"] != cache.meta["graph_path"]
    assert int((unpack_graph(c2.graphs, vid)["node_type"] == 1).sum()) == 2
    c3 = build_cache(ds, make_cfg(*small, "data.crop_size=32"), cache_dir=cdir)  # base parameter: graphs shared
    assert c3.meta["rebuilt"] == {"base": True, "graphs": False} and c3.meta["base_dir"] != cache.meta["base_dir"]
    assert c3.crops.shape[1:3] == (32, 32)
    meta = Path(cache.meta["base_dir"]) / "meta.json"  # stale key inputs in meta.json -> rebuild
    m = json.loads(meta.read_text(encoding="utf-8"))
    m["key_inputs"]["efd_order"] = 3
    meta.write_text(json.dumps(m), encoding="utf-8")
    assert build_cache(ds, cfg, cache_dir=cdir).meta["rebuilt"] == {"base": True, "graphs": False}
    gp = Path(cache.meta["graph_path"])
    blob = torch.load(gp, weights_only=True)
    blob["key_inputs"]["n_nodes"] = 1
    torch.save(blob, gp)
    assert build_cache(ds, cfg, cache_dir=cdir).meta["rebuilt"] == {"base": False, "graphs": True}
    assert build_cache(ds, cfg, cache_dir=cdir, force=True).meta["rebuilt"] == {"base": True, "graphs": True}
    assert not list(cdir.rglob("*.tmp"))  # atomic writes leave no temporaries


# ----------------------------------------------------------------------------- columns, scalers, indices


def test_particle_feature_and_baseline_global_columns():
    cols = ["shape_area_um2", "efd_0", "fractal_divider_D", "tex_mean", "color_r", "mv_n_views", "other"]
    assert particle_feature_columns(cols, make_cfg()) == ["shape_area_um2", "fractal_divider_D", "tex_mean",
                                                          "color_r", "mv_n_views"]
    for flag, prefix in [("use_fractal_features", "fractal_"), ("use_texture_branch", "tex_"),
                         ("use_composition", "color_"), ("use_multiview", "mv_")]:
        got = particle_feature_columns(cols, make_cfg(f"model.{flag}=false"))
        assert not any(c.startswith(prefix) for c in got) and len(got) == 4 and "efd_0" not in got
    assert baseline_global_columns(cols) == ["shape_area_um2", "fractal_divider_D"]
    assert baseline_global_columns(cols, "fecnn") == ["shape_area_um2", "fractal_divider_D", "tex_mean"]
    with pytest.raises(ValueError):
        baseline_global_columns(cols, "texture")


def test_standardizer_median_fill_zero_variance_and_subsets():
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 5.0], "b": [3.0] * 4, "c": [np.nan] * 4, "d": [0.0, np.inf, 1.0, 2.0]})
    st = Standardizer().fit(train)
    z = st.transform(train)
    a = np.array([1.0, 2.0, 2.0, 5.0])  # NaN -> train median 2 before scaling
    assert z.dtype == np.float32 and np.isfinite(z).all()
    np.testing.assert_allclose(z[:, 0], (a - a.mean()) / a.std(), rtol=1e-6)
    assert (z[:, 1] == 0).all() and (z[:, 2] == 0).all()  # zero variance / all NaN
    np.testing.assert_allclose(z[1, 3], 0.0, atol=1e-7)  # inf -> median 1 = mean of [0, 1, 1, 2]
    np.testing.assert_allclose(st.transform(pd.DataFrame({"b": [4.0], "a": [np.nan]})),
                               [[1.0, (2 - a.mean()) / a.std()]], rtol=1e-6)


def test_scalers_use_train_runs_only(tiny):
    ds, cache, cfg = tiny
    runs = runs_of(ds)
    train, test = runs[:6], runs[6:]
    sc = fit_scalers(ds, cache, train, cfg)
    train_p, test_p = particles_for_runs(ds, train), particles_for_runs(ds, test)
    pd.testing.assert_series_equal(sc.particle.mean, cache.descriptors.loc[train_p].mean(), check_names=False)
    rt = ds.run_table().set_index("run_id")
    assert sc.cond.mean["log_load_N"] == pytest.approx(np.log(rt.loc[train, "load_N"]).mean())
    assert sc.cond.mean["log_hardness_MPa"] == pytest.approx(np.log(rt.loc[train, "hardness_MPa"]).mean())
    assert sc.feature_columns == particle_feature_columns(cache.descriptors.columns, cfg)
    desc = cache.descriptors.copy()
    desc.loc[test_p] *= 1000.0  # test-run values must not move any statistic
    sc2 = fit_scalers(ds, replace(cache, descriptors=desc), train, cfg)
    for name in ("particle", "sample", "cond"):
        for stat in ("median", "mean", "std"):
            pd.testing.assert_series_equal(getattr(getattr(sc2, name), stat), getattr(getattr(sc, name), stat))


def test_class_weights_indices_and_mono_groups(tiny_dataset):
    ds = tiny_dataset
    pids = ds.particles.particle_id.tolist()
    y = ds.label_indices().loc[pids].to_numpy()
    counts = np.bincount(y, minlength=7)
    expected = np.where(counts > 0, len(y) / ((counts > 0).sum() * np.maximum(counts, 1)), 0.0)
    w = class_weights(ds, pids, make_cfg())
    assert w.dtype == np.float32
    np.testing.assert_allclose(w, expected, rtol=1e-6)
    np.testing.assert_array_equal(class_weights(ds, pids, make_cfg("train.class_weighting=none")), np.ones(7))
    y_true = ds.label_indices("true").loc[pids].to_numpy()
    assert (class_weights(ds, pids, make_cfg("data.label_source=true")) > 0).sum() == len(np.unique(y_true))
    with pytest.raises(ValueError):
        class_weights(ds, pids, make_cfg("data.label_source=majority"))
    assert material_index(ds) == {"AISI52100": 0, "CuSn12": 1}
    spec = ds.specimens.assign(ferromagnetic=ds.specimens.material_id == "AISI52100")
    assert magnetic_index(replace(ds, specimens=spec)) == {"AISI52100": 1, "CuSn12": 0}
    mono = mono_groups(ds)  # temperature is NaN in the tiny data: still one key value
    rt = ds.run_table()
    assert set(mono) == set(rt.run_id) and min(mono.values()) >= 0
    by_mat = rt.assign(g=rt.run_id.map(mono)).groupby("material_id").g.nunique()
    assert (by_mat == 1).all() and rt.run_id.map(mono).nunique() == 2  # loads share a group, materials do not


# ----------------------------------------------------------------------------- run bags


def test_eval_run_batches_fill_every_field(tiny):
    ds, cache, cfg = tiny
    runs = runs_of(ds)
    sc = fit_scalers(ds, cache, runs, cfg)
    dims = model_dims(ds, cache, cfg, sc)
    loader = make_run_loader(ds, cache, runs, cfg, sc, train=False)
    assert len(loader) == len(runs)
    p, smp, rt = (ds.particles.set_index("particle_id"), ds.samples.set_index("sample_id"),
                  ds.run_table().set_index("run_id"))
    labels, grav, mats, mono = ds.label_indices(), ds.grav_volume_mm3(), material_index(ds), mono_groups(ds)
    seen = []
    for b in loader:
        r = b.run_ids[0]
        seen.append(r)
        assert b.n_runs == 1 and b.mech_soft is None
        assert all(getattr(b, f.name) is not None for f in fields(RunBatch) if f.name != "mech_soft")
        assert sorted(b.particle_ids) == sorted(particles_for_runs(ds, [r]))  # evaluation: every particle
        assert [p.at[q, "sample_id"] for q in b.particle_ids] == [b.sample_ids[k] for k in b.particle_to_sample]
        assert all(smp.at[s, "run_id"] == r for s in b.sample_ids) and (b.sample_to_run == 0).all()
        assert b.graphs.num_graphs == len(b.view_to_particle) == b.n_particles  # tiny: one view per particle
        assert (b.graphs.x.shape[1], b.graphs.edge_attr.shape[1]) == (dims.node_dim, dims.edge_dim)
        s = smp.loc[b.sample_ids]
        dist = (s.sliding_distance_end_m - s.sliding_distance_start_m).to_numpy()
        np.testing.assert_allclose(b.log_obs_fraction, np.log(s.observation_fraction), rtol=1e-6)
        np.testing.assert_allclose(b.sample_distance_m, dist, rtol=1e-6)
        np.testing.assert_allclose(b.sample_time_s, s.time_s, rtol=1e-6)
        np.testing.assert_allclose(b.sample_log_q, np.log(s.sampled_oil_ml / 500.0), rtol=1e-6)
        np.testing.assert_allclose(b.log10_rate_sample_target,
                                   np.log10(s.interval_wear_volume_mm3 / (rt.at[r, "load_N"] * dist)), rtol=1e-5)
        assert b.severity_target.tolist() == s.severity.tolist()
        np.testing.assert_allclose(b.log_subsample_factor, 0.0, atol=1e-6)  # full bag and n_total == rows
        np.testing.assert_allclose(b.log_grav_volume_mm3, [np.log(grav[r])], rtol=1e-6)
        np.testing.assert_allclose(b.log10_rate_run_target, [np.log10(rt.at[r, "wear_rate_mm3_Nm"])], rtol=1e-6)
        np.testing.assert_allclose(b.load_N, [rt.at[r, "load_N"]])
        np.testing.assert_allclose(b.sliding_distance_m, [rt.at[r, "sliding_distance_m"]])
        assert b.material_idx.item() == mats[rt.at[r, "material_id"]] and b.magnetic_idx.item() == 0
        assert b.mono_group.item() == mono[r] and not b.run_cumulative.item()
        np.testing.assert_allclose(b.log_area_um2, np.log(cache.descriptors.loc[b.particle_ids, "shape_area_um2"]),
                                   rtol=1e-6)
        assert b.composition_mask.tolist() == p.loc[b.particle_ids, "has_eds"].tolist()
        assert torch.all(b.composition[~b.composition_mask] == 0)
        np.testing.assert_allclose(b.composition[b.composition_mask].sum(1), 1.0, rtol=1e-6)
        assert b.mech_target.tolist() == labels.loc[b.particle_ids].tolist()
        assert torch.isnan(b.log_volume_target).all()  # no measured volumes in the tiny data
        assert b.crops.shape == (b.n_particles, 4, 64, 64) and 0 <= b.crops.min() and b.crops.max() <= 1
        np.testing.assert_allclose(b.crops.numpy(), cache.crops[[cache.particle_ids.index(q) for q in b.particle_ids]]
                                   .transpose(0, 3, 1, 2) / 255.0, atol=1e-6)  # no augmentation in evaluation
        assert torch.isfinite(b.particle_feats).all() and torch.isfinite(b.sample_feats).all()
        assert b.particle_feats.shape[1] == dims.particle_feat_dim and b.run_cond_feats.shape == (1, dims.cond_dim)
    assert sorted(seen) == runs
    batch_cond = [b.condition_idx.item() for b in loader]
    assert len(set(batch_cond)) == ds.runs.condition_id.nunique()


def test_sample_feats_and_ht_from_surviving_particles(tiny):
    ds, cache, cfg = tiny
    runs = runs_of(ds)
    samples = ds.samples.assign(n_particles_total=10)  # 10 detected, 4 stored per sample
    images = ds.images.copy()
    images.loc[images.sample_id.str.endswith("_s1"), "focus_level"] = 1.0
    ds2 = replace(ds, samples=samples, images=images)
    flt = {"column": "focus_level", "train": [[0.0, 0.5]], "test": [[0.5, float("inf")]]}
    sc = fit_scalers(ds2, cache, runs, cfg, image_filter=flt)
    s0 = particles_for_runs(ds2, runs, flt, "train")
    assert all(q.split("_s")[1].startswith("0") for q in s0)
    pd.testing.assert_series_equal(sc.particle.mean, cache.descriptors.loc[s0].mean(), check_names=False)
    for role, suffix in (("train", "_s0"), ("test", "_s1")):
        dset = RunBagDataset(ds2, cache, runs, cfg, sc, train=False, image_filter=flt, role=role)
        assert len(dset) == len(runs)
        for i in range(len(dset)):
            it = dset[i]  # the other sample has no surviving particle: dropped from the bag
            sid = it["sample_ids"][0]
            assert len(it["sample_ids"]) == 1 and sid.endswith(suffix)
            assert all(ds2.particles.set_index("particle_id").at[q, "sample_id"] == sid for q in it["particle_ids"])
            assert it["log_subsample_factor"][0] == pytest.approx(np.log(10 / 4))
            d = cache.descriptors.loc[it["particle_ids"]]
            raw = pd.DataFrame([{
                "log_n_total": np.log(10), "log_obs_fraction": np.log(5.0 / 500.0 * 0.01),
                "log_ecd_q10": d.shape_log_ecd_um.quantile(0.1), "log_ecd_q50": d.shape_log_ecd_um.quantile(0.5),
                "log_ecd_q90": d.shape_log_ecd_um.quantile(0.9), "log_total_area_um2": np.log(d.shape_area_um2.sum())}])
            np.testing.assert_allclose(it["sample_feats"], sc.sample.transform(raw[SAMPLE_FEATURES]), atol=1e-5)
    flt_mag = {"column": "magnification", "train": [200.0], "test": [1000.0]}
    d200 = RunBagDataset(ds, cache, runs, cfg, sc, train=False, image_filter=flt_mag, role="train")
    rt = ds.runs
    assert sorted(d200.run_ids) == sorted(rt.run_id[rt.repeat_idx % 2 == 1])  # 500x runs have no observed sample
    empty = RunBagDataset(ds, cache, runs, cfg, sc, train=True, image_filter=flt_mag, role="test")  # nothing passes
    assert len(empty) == 0 and len(make_run_loader(ds, cache, runs, cfg, sc, True, image_filter=flt_mag, role="test")) == 0
    sc_empty = fit_scalers(ds, cache, runs, cfg, image_filter={**flt_mag, "train": [1000.0]})
    assert np.isfinite(sc_empty.particle.transform(cache.descriptors.iloc[:3])).all()


def test_training_bags_subsample_with_area_ratio_ht_and_augment(tiny):
    ds, cache, cfg = tiny
    runs = runs_of(ds)
    cfg2 = copy.deepcopy(cfg)
    cfg2["data"]["max_particles_per_sample"] = 2
    ds2 = replace(ds, samples=ds.samples.assign(n_particles_total=10))
    sc = fit_scalers(ds2, cache, runs, cfg2)
    dset = RunBagDataset(ds2, cache, runs, cfg2, sc, train=True, seed=0)
    area = cache.descriptors.shape_area_um2
    sample_of = ds.particles.set_index("particle_id").sample_id
    for i in range(len(dset)):
        it = dset[i]
        assert np.bincount(it["particle_to_sample"]).tolist() == [2] * len(it["sample_ids"])
        for k, sid in enumerate(it["sample_ids"]):
            bag = [q for q, kk in zip(it["particle_ids"], it["particle_to_sample"]) if kk == k]
            table = sample_of.index[sample_of == sid]
            ht = np.log((area[table] ** 1.5).sum() / (area[bag] ** 1.5).sum()) + np.log(10 / len(table))
            assert it["log_subsample_factor"][k] == pytest.approx(ht)
        rows = [cache.particle_ids.index(q) for q in it["particle_ids"]]
        for crop, ref in zip(it["crops"], cache.crops[rows].transpose(0, 3, 1, 2) / np.float32(255)):
            assert any(np.allclose(crop, v) for v in dihedral(ref))  # random dihedral augmentation
    full = RunBagDataset(ds2, cache, runs, cfg2, sc, train=False)[0]
    assert len(full["particle_ids"]) == 4 * len(full["sample_ids"])
    np.testing.assert_allclose(full["log_subsample_factor"], np.log(10 / 4))
    a, b = (RunBagDataset(ds2, cache, runs, cfg2, sc, train=True, seed=3) for _ in range(2))
    draws_a, draws_b = [a[0]["particle_ids"] for _ in range(6)], [b[0]["particle_ids"] for _ in range(6)]
    assert draws_a == draws_b and len({tuple(d) for d in draws_a}) > 1  # seeded, new draw each epoch


def test_train_loader_batches_group_runs_and_offset_indices(tiny):
    ds, cache, cfg = tiny
    runs = runs_of(ds)
    sc = fit_scalers(ds, cache, runs, cfg)
    loader = make_run_loader(ds, cache, runs, cfg, sc, train=True, seed=1)
    p, smp = ds.particles.set_index("particle_id"), ds.samples.set_index("sample_id")
    seen = []
    for b in loader:
        seen += b.run_ids
        assert 1 <= b.n_runs <= cfg["train"]["batch_runs"] and len(set(b.mono_group.tolist())) == 1
        assert b.load_N.unique().numel() >= 2  # every tiny mono group has two loads
        assert [p.at[q, "sample_id"] for q in b.particle_ids] == [b.sample_ids[k] for k in b.particle_to_sample]
        assert [smp.at[s, "run_id"] for s in b.sample_ids] == [b.run_ids[k] for k in b.sample_to_run]
        assert torch.equal(b.view_to_particle, torch.arange(b.n_particles))
        assert torch.equal(b.graphs.batch.unique(), torch.arange(b.graphs.num_graphs))
    assert sorted(seen) == runs and len(loader) == 4


def test_grouped_run_sampler_one_group_per_batch_and_mixed_loads():
    groups = np.repeat([0, 1, 2], [7, 4, 1])
    loads = np.array([20, 20, 20, 120, 120, 120, 60, 20, 20, 120, 120, 20])
    s = GroupedRunSampler(groups, loads, batch_runs=3, seed=0)
    epochs = [list(s) for _ in range(4)]
    for batches in epochs:
        assert len(batches) == len(s) == 3 + 2 + 1
        assert sorted(i for b in batches for i in b) == list(range(12))
        for b in batches:
            assert len(b) <= 3 and len(set(groups[b])) == 1
            assert len(b) == 1 or len(set(loads[b])) >= 2  # mixed loads whenever the group allows it
    assert any(e != epochs[0] for e in epochs[1:])  # reshuffled every epoch
    assert [list(GroupedRunSampler(groups, loads, 3, seed=0)) for _ in range(1)][0] == epochs[0]


# ----------------------------------------------------------------------------- particles + model integration


def test_particle_loader_globals_kinds_labels_and_augmentation(tiny):
    ds, cache, cfg = tiny
    sc = fit_scalers(ds, cache, runs_of(ds), cfg)
    unlabelled = ds.particles.particle_id.iloc[:5].tolist()
    ds2 = replace(ds, labels_adjudicated=ds.labels_adjudicated[~ds.labels_adjudicated.particle_id.isin(unlabelled)])
    rows = {q: i for i, q in enumerate(cache.particle_ids)}
    for kind, prefixes in (("geometry", ("shape_", "fractal_")), ("fecnn", ("shape_", "fractal_", "tex_"))):
        loader = make_particle_loader(ds2, cache, cache.particle_ids, cfg, sc, train=False, globals_kind=kind,
                                      batch_size=32)
        b = next(iter(loader))
        cols = baseline_global_columns(cache.descriptors.columns, kind)
        assert {c.split("_")[0] + "_" for c in cols} == set(prefixes)
        np.testing.assert_allclose(b.particle_feats.numpy(),
                                   sc.particle.transform(cache.descriptors.loc[b.particle_ids, cols]), atol=1e-6)
        assert len(loader.dataset) == len(ds.particles) - 5 and not set(unlabelled) & set(loader.dataset.pids)
    idx = [rows[q] for q in b.particle_ids]
    assert b.graphs.num_graphs == b.n_particles and torch.equal(b.view_to_particle, torch.arange(b.n_particles))
    assert b.graphs.x.shape[1] == selected_dims(cache.graph_cfg, PARTICLE_GRAPH_SELECTION)[0] == 9  # no tex_*
    np.testing.assert_allclose(b.contour_points.numpy(), cache.contour_points[idx])
    np.testing.assert_allclose(b.contour_signal.numpy(), cache.contour_signal[idx])
    assert b.crops.shape == (32, 4, 64, 64) and b.mech_soft is None
    assert b.mech_target.tolist() == ds.label_indices().loc[b.particle_ids].tolist()

    soft = copy.deepcopy(cfg)
    soft["data"]["label_source"] = "soft"
    tl = make_particle_loader(ds, cache, cache.particle_ids, soft, sc, train=True, batch_size=16)
    tb = next(iter(tl))
    assert tb.mech_soft.shape == (16, 7) and torch.allclose(tb.mech_soft.sum(1), torch.ones(16))
    assert tb.particle_ids != b.particle_ids[:16]  # shuffled (seeded) for training
    assert tb.particle_ids == next(iter(make_particle_loader(ds, cache, cache.particle_ids, soft, sc, True,
                                                             batch_size=16))).particle_ids
    idx = [rows[q] for q in tb.particle_ids]
    for crop, ref in zip(tb.crops.numpy(), cache.crops[idx].transpose(0, 3, 1, 2) / np.float32(255)):
        assert any(np.allclose(crop, v) for v in dihedral(ref))


@pytest.mark.parametrize("overrides", [
    [], ["model.use_texture_branch=false"], ["model.edges=none", "model.use_interior_nodes=false"],
    ["model.use_fractal_features=false", "model.use_composition=false", "model.use_multiview=false"],
])
def test_model_dims_match_batches_and_model_runs(tiny, overrides):
    from wearpg.models import MechanismLinkedModel
    from wearpg.models.losses import compute_losses

    ds, cache, _ = tiny
    cfg = make_cfg("model.hidden=16", "model.gnn_layers=1", *overrides)
    runs = runs_of(ds)
    sc = fit_scalers(ds, cache, runs, cfg)
    dims = model_dims(ds, cache, cfg, sc)
    b = next(iter(make_run_loader(ds, cache, runs, cfg, sc, train=True)))
    assert (b.graphs.x.shape[1], b.graphs.edge_attr.shape[1]) == (dims.node_dim, dims.edge_dim)
    assert b.particle_feats.shape[1] == dims.particle_feat_dim and b.sample_feats.shape[1] == dims.sample_feat_dim
    assert (b.crops is None) == ("model.use_texture_branch=false" in overrides)
    assert (b.graphs.edge_index.shape[1] == 0) == ("model.edges=none" in overrides)
    if "model.use_composition=false" in overrides:  # luminance x 3
        assert torch.equal(b.crops[:, 0], b.crops[:, 1]) and torch.equal(b.crops[:, 1], b.crops[:, 2])
    torch.manual_seed(0)
    out = MechanismLinkedModel(cfg, dims)(b)
    total, _ = compute_losses(out, b, cfg, torch.from_numpy(class_weights(ds, cache.particle_ids, cfg)))
    assert torch.isfinite(total)
    total.backward()


@pytest.mark.slow  # generator fixture + cache build on generated images: ~15 s
def test_synthetic_small_multiview_batches(synthetic_small, tmp_path):
    ds, cfg = synthetic_small, make_cfg("graph.n_nodes=64", "graph.interior_nodes=8")
    cache = build_cache(ds, cfg, cache_dir=tmp_path)
    assert set(cache.graphs["view_ids"]) == set(ds.views.view_id)  # every view (<= max_views)
    runs = runs_of(ds)
    sc = fit_scalers(ds, cache, runs, cfg)
    dset = RunBagDataset(ds, cache, runs, cfg, sc, train=False)
    b = collate_runs([dset[0], dset[1]])
    p, smp = ds.particles.set_index("particle_id"), ds.samples.set_index("sample_id")
    assert torch.bincount(b.view_to_particle, minlength=b.n_particles).tolist() == p.n_views[b.particle_ids].tolist()
    views = ds.views.sort_values(["particle_id", "view_idx"]).groupby("particle_id").view_id.agg(list)
    expected = [v for q in b.particle_ids for v in views[q]]
    assert len(expected) == b.graphs.num_graphs and p.n_views[b.particle_ids].max() > 1
    for k, vid in enumerate(expected):  # graph k is the cached graph of the k-th (particle, view) pair
        assert torch.equal(b.graphs.get_example(k).pos, unpack_graph(cache.graphs, vid)["pos"])
    assert [p.at[q, "sample_id"] for q in b.particle_ids] == [b.sample_ids[k] for k in b.particle_to_sample]
    assert [smp.at[s, "run_id"] for s in b.sample_ids] == [b.run_ids[k] for k in b.sample_to_run]
    n_rows = p.groupby("sample_id").size()
    np.testing.assert_allclose(b.log_subsample_factor,
                               np.log(smp.loc[b.sample_ids, "n_particles_total"] / n_rows[b.sample_ids]), rtol=1e-5)
    one = RunBagDataset(ds, cache, runs, make_cfg("model.use_multiview=false"), sc, train=False)[0]
    assert len(one["graphs"]) == len(one["particle_ids"])
