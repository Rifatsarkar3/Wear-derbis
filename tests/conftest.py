"""Shared fixtures. `tiny_dataset` is a hand-built, schema-valid dataset WITHOUT image files,
so table-level modules (labels, splits, metrics, gates) can be tested independently of the
synthetic image generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wearpg.schema import EDS_COLUMNS, MECHANISMS, WearDataset


def make_tiny_dataset(n_materials: int = 2, loads=(20.0, 120.0), repeats: int = 3,
                      samples_per_run: int = 2, particles_per_sample: int = 4, seed: int = 0) -> WearDataset:
    rng = np.random.default_rng(seed)
    mats = [("AISI52100", 7500.0, 7.81), ("CuSn12", 1000.0, 8.70), ("AA6061", 1050.0, 2.70)][:n_materials]
    specimens, runs, samples, images, particles, views, labels, adj = [], [], [], [], [], [], [], []
    for m, (mat, hard, rho) in enumerate(mats):
        for load in loads:
            cond = f"{mat}_F{int(load)}"
            for rep in range(repeats):
                spec_id = f"S_{mat}_{int(load)}_{rep}"
                specimens.append(dict(specimen_id=spec_id, material_id=mat, counterface_id="AISI52100",
                                      material_pair=f"{mat}|AISI52100", hardness_MPa=hard, density_g_cm3=rho))
                run_id = f"R_{mat}_{int(load)}_{rep}"
                severe = load > 60
                k = (1.5e-4 if severe else 2e-6) * float(np.exp(rng.normal(0, 0.2)))
                dist = 0.4 * 1800.0
                vol = k * load * dist
                runs.append(dict(run_id=run_id, specimen_id=spec_id, condition_id=cond, rig="pin_on_disc",
                                 load_N=load, speed_m_s=0.4, lubricant_id="PAO8", duration_s=1800.0,
                                 sliding_distance_m=dist, repeat_idx=rep, reservoir_volume_ml=500.0,
                                 mass_loss_mg=vol * rho, wear_volume_mm3=vol, wear_volume_method="gravimetric",
                                 wear_rate_mm3_Nm=vol / (load * dist), regime="severe" if severe else "mild",
                                 transition_time_s=900.0 if severe else np.nan))
                dt = 1800.0 / samples_per_run
                for si in range(samples_per_run):
                    sid = f"{run_id}_s{si}"
                    sev = (2 if si > 0 else 1) if severe else 0
                    samples.append(dict(sample_id=sid, run_id=run_id, t_start_s=si * dt, t_end_s=(si + 1) * dt,
                                        time_s=(si + 1) * dt, sliding_distance_start_m=si * dt * 0.4,
                                        sliding_distance_end_m=(si + 1) * dt * 0.4, sampled_oil_ml=5.0,
                                        imaged_fraction=0.01, observation_fraction=5.0 / 500.0 * 0.01,
                                        n_particles_total=particles_per_sample,
                                        interval_wear_volume_mm3=vol / samples_per_run, severity=sev))
                    iid = f"{sid}_img0"
                    images.append(dict(image_id=iid, sample_id=sid, path=f"images/{iid}.png",
                                       modality="ferrograph_optical", magnification=200.0 if rep % 2 else 500.0,
                                       um_per_px=0.5 if rep % 2 else 0.25, focus_level=0.0,
                                       illumination="reflected", width_px=256, height_px=256))
                    for pi in range(particles_per_sample):
                        pid = f"{sid}_p{pi}"
                        mech = MECHANISMS[(pi + (3 if severe else 0)) % len(MECHANISMS)]
                        has_eds = pi % 2 == 0
                        eds = {c: np.nan for c in EDS_COLUMNS}
                        if has_eds:
                            eds = {c: 0.0 for c in EDS_COLUMNS}
                            eds["eds_Fe"] = 0.9
                            eds["eds_O"] = 0.1
                        particles.append(dict(particle_id=pid, sample_id=sid, image_id=iid, n_views=1,
                                              has_eds=has_eds, **eds, has_sem=pi == 0, thickness_um=np.nan,
                                              volume_um3=np.nan, true_mechanism=mech,
                                              true_volume_um3=float(rng.uniform(10, 1000))))
                        t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                        r = 8 + 2 * pi + rng.normal(0, 0.3, t.size)
                        cx, cy = 40 + 50 * pi, 128
                        x, y = cx + r * np.cos(t), cy + r * np.sin(t)
                        views.append(dict(view_id=f"{pid}_v0", particle_id=pid, view_idx=0, image_id=iid,
                                          bbox_x0=int(np.floor(x.min())), bbox_y0=int(np.floor(y.min())),
                                          bbox_x1=int(np.ceil(x.max())) + 1, bbox_y1=int(np.ceil(y.max())) + 1,
                                          contour_x=x, contour_y=y,
                                          um_per_px=0.5 if rep % 2 else 0.25, source="gt"))
                        lab_a = mech
                        lab_b = mech if pi % 3 else MECHANISMS[(MECHANISMS.index(mech) + 1) % len(MECHANISMS)]
                        labels.append(dict(particle_id=pid, rater_id="A", label=lab_a, confidence=4, evidence="optical"))
                        labels.append(dict(particle_id=pid, rater_id="B", label=lab_b, confidence=3, evidence="optical"))
                        adj.append(dict(particle_id=pid, label=lab_a,
                                        method="unanimous" if lab_a == lab_b else "adjudicated",
                                        n_raters=2, agreement=1.0 if lab_a == lab_b else 0.5))
    return WearDataset(
        specimens=pd.DataFrame(specimens), runs=pd.DataFrame(runs), samples=pd.DataFrame(samples),
        images=pd.DataFrame(images), particles=pd.DataFrame(particles), views=pd.DataFrame(views),
        labels=pd.DataFrame(labels), labels_adjudicated=pd.DataFrame(adj),
        meta={"name": "tiny", "is_synthetic": True},
    )


@pytest.fixture
def tiny_dataset() -> WearDataset:
    return make_tiny_dataset()


@pytest.fixture
def tiny_dataset_on_disk(tmp_path) -> WearDataset:
    """`make_tiny_dataset()` saved to tmp_path with one rendered 256x256 PNG per images row
    (background (215,220,226), each view polygon filled with (120,120,128)). Independent of synthetic.py."""
    from skimage.draw import polygon
    from skimage.io import imsave

    ds = make_tiny_dataset()
    ds.save(tmp_path)
    for row in ds.images.itertuples():
        img = np.empty((row.height_px, row.width_px, 3), np.uint8)
        img[:] = (215, 220, 226)
        for v in ds.views[ds.views.image_id == row.image_id].itertuples():
            rr, cc = polygon(np.asarray(v.contour_y), np.asarray(v.contour_x), shape=img.shape[:2])
            img[rr, cc] = (120, 120, 128)
        out = tmp_path / row.path
        out.parent.mkdir(parents=True, exist_ok=True)
        imsave(out, img, check_contrast=False)
    return WearDataset.load(tmp_path)


@pytest.fixture(scope="session")
def synthetic_small(tmp_path_factory) -> WearDataset:
    """Small generated dataset WITH images (~12 runs x 2 samples, <= 10 stored particles each), built once per
    session. For integration-style tests (mark heavy ones @pytest.mark.slow)."""
    synthetic = pytest.importorskip("wearpg.synthetic")
    from wearpg.config import deep_merge

    cfg = deep_merge(synthetic.load_synthetic_config("smoke"),
                     {"samples_per_run": 2, "duration_s": 1200, "debris": {"max_stored_per_sample": 10}})
    return synthetic.generate_dataset(cfg, tmp_path_factory.mktemp("synthetic_small"), seed=0)
