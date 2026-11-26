import numpy as np
import pytest

from wearpg.splits import (Split, audit_leakage, grouped_kfold, image_passes, leave_one_group_out, load_splits,
                           particles_for_runs, safe_name, save_splits, transfer_split)


def _roles(s):
    return {"train": set(s.train_runs), "val": set(s.val_runs), "test": set(s.test_runs)}


def _role_of(s, run):
    return next(r for r, runs in _roles(s).items() if run in runs)


@pytest.mark.parametrize("level,k", [("run", 5), ("specimen", 5), ("condition", 3), ("condition", 4)])
def test_grouped_kfold_partitions_and_is_clean(tiny_dataset, level, k):
    ds = tiny_dataset
    splits = grouped_kfold(ds, level=level, n_splits=k, seed=3)
    all_runs = set(ds.runs.run_id)
    assert [s.name for s in splits] == [f"{level}_k{k}_f{i}" for i in range(k)]
    tested = [r for s in splits for r in s.test_runs]
    assert sorted(tested) == sorted(all_runs)  # every run tested exactly once
    for s in splits:
        r = _roles(s)
        assert s.protocol == "cv" and r["train"] and r["test"] and r["train"] | r["val"] | r["test"] == all_runs
        assert audit_leakage(ds, s) == []
    again = grouped_kfold(ds, level=level, n_splits=k, seed=3)
    assert [s.test_runs for s in again] == [s.test_runs for s in splits]  # deterministic


def test_grouped_kfold_stratification_flag(tiny_dataset):
    splits = grouped_kfold(tiny_dataset, level="specimen", n_splits=5, seed=0)  # 6 mild + 6 severe specimens
    assert all(s.meta["stratified"] for s in splits)
    regime = tiny_dataset.runs.set_index("run_id").regime
    assert all(set(regime[s.test_runs]) == {"mild", "severe"} for s in splits)
    cond = grouped_kfold(tiny_dataset, level="condition", n_splits=3)  # 2 conditions per regime < 3 folds
    assert not any(s.meta["stratified"] for s in cond)
    assert not grouped_kfold(tiny_dataset, n_splits=3, stratify=None)[0].meta["stratified"]
    with pytest.raises(ValueError):
        grouped_kfold(tiny_dataset, level="condition", n_splits=5)  # only 4 conditions
    with pytest.raises(ValueError):
        grouped_kfold(tiny_dataset, level="lab")


def test_runs_sharing_a_specimen_stay_together(tiny_dataset):
    ds = tiny_dataset
    r0, r1 = ds.runs.run_id.iloc[0], ds.runs.run_id.iloc[4]  # different conditions
    ds.runs.loc[4, "specimen_id"] = ds.runs.specimen_id.iloc[0]
    assert ds.validate() == []
    for level, k in (("run", 5), ("specimen", 5), ("condition", 3)):
        for seed in range(4):
            for s in grouped_kfold(ds, level=level, n_splits=k, seed=seed):
                assert _role_of(s, r0) == _role_of(s, r1), (level, seed)
                assert audit_leakage(ds, s) == []
    # a split that separates them leaks the specimen and the audit must say so
    s = grouped_kfold(ds, level="run", n_splits=5)[0]
    bad = Split("bad", "cv", sorted((set(s.train_runs) | {r0}) - {r1}), [], sorted((set(s.test_runs) | {r1}) - {r0}),
                {"level": "run"})
    errs = audit_leakage(ds, bad)
    assert any(e.startswith("specimen_id overlap train/test") for e in errs)


def test_audit_catches_injected_overlap(tiny_dataset):
    ds = tiny_dataset
    s = grouped_kfold(ds, level="specimen", n_splits=5)[0]
    leak = Split(s.name, s.protocol, s.train_runs + [s.test_runs[0]], s.val_runs, s.test_runs, s.meta)
    errs = "\n".join(audit_leakage(ds, leak))
    for col in ("run_id", "specimen_id", "sample_id", "image_id", "particle_id"):
        assert f"{col} overlap train/test" in errs
    assert "unknown run ids" in "\n".join(audit_leakage(ds, Split("x", "cv", ["nope"], [], s.test_runs, {})))
    # two repeats of one condition (different specimens): leakage at level condition only
    a, b = "R_AISI52100_20_0", "R_AISI52100_20_1"
    assert audit_leakage(ds, Split("s", "cv", [a], [], [b], {"level": "specimen"})) == []
    assert audit_leakage(ds, Split("c", "cv", [a], [], [b], {"level": "condition"})) == [
        "condition_id overlap train/test: 1 (e.g. ['AISI52100_F20'])"]


def test_leave_one_group_out(tiny_dataset):
    ds = tiny_dataset
    splits = leave_one_group_out(ds, by="material_id")
    assert [s.name for s in splits] == ["logo_material_id_AISI52100", "logo_material_id_CuSn12"]
    mat = ds.run_table().set_index("run_id").material_id
    for s in splits:
        assert s.protocol == "external" and s.val_runs
        assert set(mat[s.test_runs]) == {s.meta["held_out"]} and s.meta["held_out"] not in set(mat[s.train_runs + s.val_runs])
        assert audit_leakage(ds, s) == []
    assert [s.name for s in leave_one_group_out(ds, by="load_N")] == ["logo_load_N_20", "logo_load_N_120"]
    with pytest.raises(ValueError):
        leave_one_group_out(ds, by="nope")


def test_leave_one_group_out_drops_runs_sharing_test_specimens(tiny_dataset):
    ds = tiny_dataset
    ds.runs.loc[ds.runs.run_id == "R_AISI52100_120_0", "specimen_id"] = "S_AISI52100_20_0"  # one specimen, two loads
    for s in leave_one_group_out(ds, by="load_N"):
        assert len(s.meta["dropped_runs"]) == 1 and audit_leakage(ds, s) == []


def test_transfer_split_exact_values(tiny_dataset):
    ds = tiny_dataset  # magnification 200 on odd repeats, 500 on even repeats (per run)
    s = transfer_split(ds, "magnification", [200], [500], seed=1)
    assert s.name == "transfer_magnification_200_to_500" and s.protocol == "transfer"
    assert s.meta["image_filter"] == {"column": "magnification", "train": [200], "test": [500]}
    rep = ds.runs.set_index("run_id").repeat_idx
    assert all(rep[r] % 2 == 1 for r in s.train_runs + s.val_runs) and all(rep[r] % 2 == 0 for r in s.test_runs)
    assert len(s.test_runs) == 8 and len(s.train_runs) + len(s.val_runs) == 4  # one-domain groups go to their side
    h = ds.particle_hierarchy().set_index("particle_id")
    assert set(h.magnification[particles_for_runs(ds, s.test_runs, s.meta["image_filter"], "test")]) == {500.0}
    assert set(h.magnification[particles_for_runs(ds, s.val_runs, s.meta["image_filter"], "val")]) == {200.0}
    assert audit_leakage(ds, s) == []
    with pytest.raises(ValueError):
        transfer_split(ds, "magnification", [200], [1000])  # no run passes the test filter
    with pytest.raises(ValueError):
        transfer_split(ds, "no_such_column", [1], [2])


def test_transfer_split_intervals(tiny_dataset):
    ds = tiny_dataset
    ds.images["focus_level"] = np.where(ds.images.sample_id.str.endswith("_s0"), 0.2, 1.0)  # every run has both
    s = transfer_split(ds, "focus_level", [[0.0, 0.5]], [[0.5, float("inf")]], level="specimen", seed=0)
    assert s.name == "transfer_focus_level_0-0.5_to_0.5-inf"
    assert len(s.test_runs) == 4  # round(0.3 * 12) groups eligible for both roles
    assert not set(s.test_runs) & set(s.train_runs + s.val_runs)
    h = ds.particle_hierarchy().set_index("particle_id")
    flt = s.meta["image_filter"]
    assert set(h.focus_level[particles_for_runs(ds, s.test_runs, flt, "test")]) == {1.0}
    assert set(h.focus_level[particles_for_runs(ds, s.train_runs, flt, "train")]) == {0.2}
    assert len(particles_for_runs(ds, s.test_runs)) == 2 * len(particles_for_runs(ds, s.test_runs, flt, "test"))
    assert audit_leakage(ds, s) == []
    # filter violations: overlapping domains and a run without any passing particle in its role
    overlap = Split(s.name, s.protocol, s.train_runs, s.val_runs, s.test_runs,
                    {**s.meta, "image_filter": {**flt, "train": [[0.0, 2.0]]}})
    assert any("filters overlap" in e for e in audit_leakage(ds, overlap))
    ds.images.loc[ds.images.sample_id.str.startswith(s.test_runs[0]), "focus_level"] = 0.2
    assert any(e.startswith("test runs without a particle passing") for e in audit_leakage(ds, s))


def test_image_passes():
    assert image_passes(200.0, 200) and not image_passes(500.0, 200)
    assert image_passes(0.0, [0.0, 0.5]) and not image_passes(0.5, [0.0, 0.5])  # [lo, hi)
    assert image_passes(9.0, (0.5, float("inf"))) and not image_passes(float("nan"), [0.0, 1.0])
    assert image_passes("sem", "sem") and not image_passes("sem", "ferrograph_optical")


def test_particles_for_runs_without_filter(tiny_dataset):
    ds = tiny_dataset
    run = ds.runs.run_id.iloc[0]
    assert particles_for_runs(ds, [run]) == [p for p in ds.particles.particle_id if p.startswith(run + "_")]


def test_save_load_roundtrip(tiny_dataset, tmp_path):
    ds = tiny_dataset
    ds.images["focus_level"] = np.where(ds.images.sample_id.str.endswith("_s0"), 0.2, 1.0)
    splits = grouped_kfold(ds, n_splits=3) + leave_one_group_out(ds) + [
        transfer_split(ds, "focus_level", [[0.0, 0.5]], [[0.5, float("inf")]])]
    path = save_splits(splits, tmp_path / "splits.json")
    assert load_splits(path) == splits


@pytest.mark.slow
def test_splits_clean_on_generated_data(synthetic_small):  # skipped until wearpg.synthetic exists
    ds = synthetic_small
    for s in grouped_kfold(ds, n_splits=3) + grouped_kfold(ds, level="condition", n_splits=2) + leave_one_group_out(ds):
        assert audit_leakage(ds, s) == [], s.name
    s = transfer_split(ds, "magnification", [200], [500])
    assert audit_leakage(ds, s) == []


def test_safe_name():
    assert safe_name("logo_load_N_120 N/m:x") == "logo_load_N_120-N-m-x"
    assert safe_name("a.b-c_d") == "a.b-c_d"
