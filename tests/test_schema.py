import numpy as np

from wearpg.config import apply_overrides, load_config
from wearpg.schema import WearDataset


def test_tiny_dataset_valid(tiny_dataset):
    assert tiny_dataset.validate() == []


def test_roundtrip(tiny_dataset, tmp_path):
    tiny_dataset.save(tmp_path)
    ds = WearDataset.load(tmp_path)
    assert ds.validate() == []
    assert ds.is_synthetic
    assert len(ds.views) == len(tiny_dataset.views)
    c0 = WearDataset.contour(ds.views.iloc[0])
    assert c0.shape[1] == 2 and np.allclose(c0, WearDataset.contour(tiny_dataset.views.iloc[0]))
    assert ds.samples.severity.dtype.kind == "i" and ds.runs.transition_time_s.dtype.kind == "f"


def test_validation_catches_errors(tiny_dataset):
    ds = tiny_dataset
    ds.samples.loc[0, "observation_fraction"] = 0.5
    ds.particles.loc[0, "n_views"] = 3
    ds.runs.loc[0, "wear_rate_mm3_Nm"] = 1.0
    errs = ds.validate()
    joined = "\n".join(errs)
    assert "observation_fraction" in joined and "n_views" in joined and "wear_rate" in joined


def test_hierarchy_and_subset(tiny_dataset):
    h = tiny_dataset.particle_hierarchy()
    assert len(h) == len(tiny_dataset.particles)
    assert h[["run_id", "specimen_id", "material_id", "magnification"]].notna().all().all()
    runs = tiny_dataset.runs.run_id.iloc[:3].tolist()
    sub = tiny_dataset.subset(runs)
    assert set(sub.runs.run_id) == set(runs) and sub.validate() == []
    lab = tiny_dataset.label_indices("adjudicated")
    assert (lab >= 0).all()


def test_config_inheritance():
    cfg = load_config("configs/smoke.yaml", overrides=["model.hidden=32", "eval.seeds=[0,1]"])
    assert cfg["model"]["hidden"] == 32 and cfg["eval"]["seeds"] == [0, 1]
    assert cfg["graph"]["n_nodes"] == 128  # inherited from default.yaml
    assert apply_overrides({}, ["a.b=1"]) == {"a": {"b": 1}}


def test_tiny_dataset_on_disk(tiny_dataset_on_disk):
    from skimage.io import imread

    ds = tiny_dataset_on_disk
    assert ds.validate() == []
    img = imread(ds.image_path(ds.images.image_id.iloc[0]))
    assert img.shape == (256, 256, 3) and img.min() < 200  # particles rendered darker than background


def test_grav_volume_and_new_columns(tiny_dataset):
    ds = tiny_dataset
    g = ds.grav_volume_mm3()
    assert np.allclose(g.loc[ds.runs.run_id].to_numpy(), ds.runs.wear_volume_mm3.to_numpy())
    ds.runs.loc[0, "counterface_mass_loss_mg"] = 7.81
    ds.specimens.loc[ds.specimens.specimen_id == ds.runs.specimen_id.iloc[0], "counterface_density_g_cm3"] = 7.81
    assert np.isclose(ds.grav_volume_mm3().iloc[0], ds.runs.wear_volume_mm3.iloc[0] + 1.0)
    ds.runs.loc[1, "debris_sampling"] = "bogus"
    assert any("debris_sampling" in e for e in ds.validate())
