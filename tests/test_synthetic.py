import hashlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

from wearpg import synthetic as S
from wearpg.config import deep_merge
from wearpg.schema import MECHANISMS, WearDataset


def tiny_cfg(over: dict | None = None) -> dict:
    """2 materials x 2 loads x 1 repeat x 2 samples, small images: fast full generation."""
    base = deep_merge(S.load_synthetic_config("smoke"), {
        "repeats": 1, "samples_per_run": 2, "duration_s": 1200,
        "debris": {"max_stored_per_sample": 8}, "imaging": {"image_size_px": 256}})
    return deep_merge(base, over or {})


@pytest.fixture(scope="module")
def pilot_sim():
    return S._simulate(S.load_synthetic_config("pilot"), 0)


@pytest.fixture(scope="module")
def synthetic_small_v2(tmp_path_factory):
    """conftest synthetic_small rendered with renderer v2 (bichromatic light, ferrographic chains)."""
    cfg = deep_merge(S.load_synthetic_config("smoke"), {"samples_per_run": 2, "duration_s": 1200,
                                                        "debris": {"max_stored_per_sample": 10},
                                                        "imaging": {"renderer": "v2"}})
    return S.generate_dataset(cfg, tmp_path_factory.mktemp("synthetic_small_v2"), seed=0)


RENDERERS = pytest.mark.parametrize("renderer", ["v1", "v2"])
SMALL = pytest.mark.parametrize("small", ["synthetic_small", "synthetic_small_v2"])


def v(renderer: str) -> dict:
    return {"imaging": {"renderer": renderer}}


def _hash_dir(root) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_load_config_names_and_path():
    smoke, pilot = S.load_synthetic_config("smoke"), S.load_synthetic_config("pilot")
    assert smoke["name"] == "synthetic_smoke" and pilot["name"] == "synthetic_pilot"
    assert S.load_synthetic_config("configs/synthetic_smoke.yaml") == smoke
    assert set(S.SEVERITY_MIXTURES) == {0, 1, 2} and set(S.MECHANISM_PRIORS) == set(MECHANISMS)
    for mix in S.SEVERITY_MIXTURES.values():
        assert abs(sum(mix.values()) - 1) < 1e-9


@RENDERERS
def test_tiny_dataset_valid_meta_and_reload(tmp_path, renderer):
    ds = S.generate_dataset(tiny_cfg(v(renderer)), tmp_path, seed=3)
    assert ds.validate() == []
    m = ds.meta
    assert m["is_synthetic"] is True and m["generator"] == "wearpg.synthetic" and m["seed"] == 3
    assert m["warning"] == S.SYNTHETIC_BANNER and m["generator_config"]["name"] == "synthetic_smoke"
    assert {"expected_severe_fraction", "median_n_total", "empty_sample_fraction", "mean_ht_factor",
            "cramers_v_magnification_severity", "cramers_v_magnification_material", "fractal"} <= set(m["stats"])
    re = WearDataset.load(tmp_path)
    assert re.validate() == [] and re.is_synthetic
    assert len(re.particles) == len(ds.particles) > 0
    assert {"true_fractal_D_target", "true_fractal_D"} <= set(re.particles.columns)
    assert (ds.runs.debris_sampling == "interval").all()
    for p in ds.images.path:
        assert (tmp_path / p).exists()


@RENDERERS
def test_determinism(tmp_path, renderer):
    a = S.generate_dataset(tiny_cfg(v(renderer)), tmp_path / "a", seed=5)
    S.generate_dataset(tiny_cfg(v(renderer)), tmp_path / "b", seed=5)
    assert _hash_dir(tmp_path / "a") == _hash_dir(tmp_path / "b")   # tables AND images byte-identical
    c = S.generate_dataset(tiny_cfg(v(renderer)), tmp_path / "c", seed=6)
    assert not a.runs.wear_volume_mm3.equals(c.runs.wear_volume_mm3)


def test_wear_bookkeeping(pilot_sim):
    cfg = S.load_synthetic_config("pilot")
    w = cfg["wear"]
    rt = pilot_sim["runs"].merge(pilot_sim["specimens"], on="specimen_id")
    sm = pilot_sim["samples"].merge(rt, on="run_id")
    # gravimetry: rounded mass, V = m / rho, k = V / (F s)
    res = w["mass_resolution_mg"]
    assert np.allclose(rt.mass_loss_mg / res, np.round(rt.mass_loss_mg / res), atol=1e-6)
    assert np.allclose(rt.wear_volume_mm3, rt.mass_loss_mg / rt.density_g_cm3)
    assert np.allclose(rt.wear_rate_mm3_Nm, rt.wear_volume_mm3 / (rt.load_N * rt.sliding_distance_m))
    assert np.allclose(rt.sliding_distance_m, rt.speed_m_s * rt.duration_s)
    # true interval volumes add up to the gravimetric volume within mass noise + rounding
    v_true = sm.groupby("run_id").true_interval_wear_volume_mm3.sum().loc[rt.run_id].to_numpy()
    assert np.all(np.abs(rt.wear_volume_mm3 - v_true) <= 5 * w["mass_noise_rel"] * v_true + res / rt.density_g_cm3)
    assert sm.interval_wear_volume_mm3.isna().all()          # expose_interval_volume: false in the pilot
    # hardness-scaled Archard: mild-interval k / (k_mild_ref H_ref / H) is the lognormal run x interval effect
    mild = sm[(sm.severity == 0) & (sm.t_start_s > 0)]
    k = mild.true_interval_wear_volume_mm3 / (mild.load_N * mild.speed_m_s * (mild.t_end_s - mild.t_start_s))
    ln_ratio = np.log(k / (w["k_mild_ref"] * w["hardness_ref_MPa"] / mild.hardness_MPa))
    assert abs(ln_ratio.mean()) < 0.15 and 0.15 < ln_ratio.std() < 0.45
    # regime / transition time / severity labels are mutually consistent
    sev = rt.regime == "severe"
    assert (np.isfinite(rt.transition_time_s) == sev).all() and (rt.transition_time_s[sev] < rt.duration_s[sev]).all()
    t_tr = sm.transition_time_s.fillna(np.inf)
    expect = np.where(sm.t_end_s <= t_tr, 0, np.where(sm.t_start_s >= t_tr, 2, 1))
    assert (sm.severity.to_numpy() == expect).all()


def test_exposed_interval_volume_is_noisy_rounded_gravimetry(tmp_path):
    sim = S._simulate(tiny_cfg(), 0)   # smoke config exposes interval volumes
    sm = sim["samples"].merge(sim["runs"], on="run_id").merge(sim["specimens"], on="specimen_id")
    iv, tv = sm.interval_wear_volume_mm3, sm.true_interval_wear_volume_mm3
    mass = iv * sm.density_g_cm3
    ok = iv.notna()
    assert ok.any() and np.allclose(mass[ok] / 0.01, np.round(mass[ok] / 0.01), atol=1e-6)
    assert np.all(np.abs(iv[ok] - tv[ok]) <= 0.1 * tv[ok] + 0.01 / sm.density_g_cm3[ok])


@pytest.mark.parametrize("sampling", ["interval", "cumulative"])
def test_budget_invariant_oracle(sampling):
    cfg = tiny_cfg({"debris": {"budget_sigma": 0.0, "sampling": sampling}})
    sim = S._simulate(cfg, 1)
    sm = sim["samples"].merge(sim["runs"][["run_id", "reservoir_volume_ml"]], on="run_id")
    eta = pd.Series(sim["material"]).map(sim["stats"]["eta_material"]).to_numpy()
    dv = sm.true_interval_wear_volume_mm3.to_numpy()
    if sampling == "cumulative":   # inventory: earlier debris x prod(1 - q) of the samplings in between
        q = (sm.sampled_oil_ml / sm.reservoir_volume_ml).to_numpy()
        inv = dv.copy()
        for i in range(1, len(sm)):
            if sm.run_id[i] == sm.run_id[i - 1]:
                inv[i] = inv[i - 1] * (1 - q[i - 1]) + dv[i]
        dv = inv
    budget, drawn, last, n = sim["budget"].T
    assert np.allclose(budget, eta * sm.observation_fraction * dv * 1e9, rtol=1e-9)
    assert (n >= 1).all()
    assert (drawn >= budget).all() and (drawn - last < budget).all()   # reaches it, overshoot < one particle


@SMALL
def test_detection_rule_and_storage(small, request):
    synthetic_small = request.getfixturevalue(small)
    img = S.load_synthetic_config("smoke")["imaging"]
    upp, lim = 0.5, S._fov_limit_px(img["image_size_px"])
    area = np.array([30, 29.9, 40, 40, 40]) * upp ** 2
    size = np.array([5, 5, 5, 5, lim * upp + 0.1])
    contrast = np.array([0.05, 0.5, 0.0499, 0.3, 0.3])
    assert S._detected(area, size, contrast, upp, img).tolist() == [True, False, False, True, False]
    ds = synthetic_small
    cap = ds.meta["generator_config"]["debris"]["max_stored_per_sample"]
    rows = ds.particles.groupby("sample_id").size().reindex(ds.samples.sample_id, fill_value=0).to_numpy()
    n_tot = ds.samples.n_particles_total.to_numpy()
    assert (rows <= np.minimum(n_tot, cap)).all() and (n_tot >= rows).all()
    assert (rows == np.minimum(n_tot, cap)).sum() >= len(rows) - ds.meta["stats"]["n_fov_dropped"]
    assert (n_tot > cap).any()      # the cap binds somewhere: n_particles_total counts all detected
    v0 = ds.views[ds.views.view_idx == 0]
    area_px = np.array([S._shoelace(WearDataset.contour(r)) for _, r in v0.iterrows()])
    assert (area_px >= img["min_area_px"] * (1 - 1e-9)).all()   # GT contour area == drawn area exactly


def test_detection_contrast_rule_bites():
    cfg = tiny_cfg({"imaging": {"max_luminance_ratio": 1.0, "background_rgb": [125, 125, 133], "min_contrast": 0.1}})
    sim = S._simulate(cfg, 0)
    p = sim["parts"]
    assert sim["stats"]["n_detected"] < sim["stats"]["n_drawn"]
    assert (p.contrast >= 0.1).all() and (p.area / sim["samples"].um_per_px[p.sample_idx].to_numpy() ** 2 >= 30).all()


@RENDERERS
def test_zero_particle_samples_are_legal(tmp_path, renderer):
    ds = S.generate_dataset(tiny_cfg({"imaging": {"min_contrast": 0.99, "renderer": renderer}}), tmp_path, seed=0)
    assert ds.validate() == [] and len(ds.particles) == 0
    assert (ds.samples.n_particles_total == 0).all() and ds.meta["stats"]["empty_sample_fraction"] == 1.0
    assert ds.images.groupby("sample_id").size().reindex(ds.samples.sample_id).eq(1).all()   # one empty field each


def test_pilot_statistics(pilot_sim):
    st = pilot_sim["stats"]
    assert st["empty_sample_fraction"] < 0.05
    assert set(st["expected_severe_fraction"]) == {"AISI52100", "CuSn12", "AA6061"}
    assert all(0.3 <= v <= 0.7 for v in st["expected_severe_fraction"].values())
    assert st["cramers_v_magnification_severity"] < 0.1 and st["cramers_v_magnification_material"] < 0.1
    assert st["n_stored"] > 0 and st["mean_ht_factor"] >= 1


def test_divider_estimator():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.c_[100 * np.cos(t), 100 * np.sin(t)]
    d = S._divider_dimension(circle)
    assert d["valid"] and abs(d["D"] - 1) < 0.01
    assert not S._divider_dimension(circle / 10)["valid"]       # L = 20 px: steps would span < 2x
    rng = np.random.default_rng(0)
    rough = S._contour_px(rng, "circle", 40.0, 40.0, 1.25, 1.2, 0.2, 0.0, 1250.0)
    assert S._divider_dimension(rough)["D"] > 1.08


def test_fractal_calibration_tracks_target():
    """Spearman(measured divider D of GT contours with perimeter >= 100 px, D_target) >= 0.8 at both magnifications."""
    rng = np.random.default_rng(11)
    p = {k: np.concatenate([v1, v2]) for (k, v1), v2 in
         zip(S._draw_particles(rng, 500, 0.0, 60.0, 0.47).items(), S._draw_particles(rng, 500, 1.0, 60.0, 0.47).values())}
    shapes = S._PR["shape"][p["mech"]]
    for upp in (0.5, 0.2):
        dt, dm = [], []
        for i in np.flatnonzero(p["size"] / upp > 25):
            xy = S._contour_px(rng, shapes[i], p["size"][i], p["minor"][i], p["d_target"][i],
                               S._amplitude(p["d_target"][i]), upp, rng.uniform(0, 6.3), p["area"][i])
            d = S._divider_dimension(xy)
            if S._perimeter(xy) >= 100 and d["valid"]:
                dt.append(p["d_target"][i])
                dm.append(d["D"])
        assert len(dt) > 150
        assert spearmanr(dt, dm)[0] >= 0.8, upp


@SMALL
def test_contours_and_rendering_agree(small, request):
    ds = request.getfixturevalue(small)
    fov = ds.images[ds.images.modality == "ferrograph_optical"].iloc[:6]
    for im in fov.itertuples():
        gray = np.asarray(Image.open(ds.root / im.path).convert("L"), float) / 255
        mask = np.zeros(gray.shape, bool)
        for v in ds.views[ds.views.image_id == im.image_id].itertuples():
            xy = WearDataset.contour(v)
            assert len(xy) >= 8 and S._shoelace(xy) > 0 and not np.allclose(xy[0], xy[-1])
            mask |= S._fill(xy, *gray.shape)
        if mask.sum() < 20:
            continue
        inside = ndimage.binary_erosion(mask, iterations=1)
        outside = ~ndimage.binary_dilation(mask, iterations=6)
        if small == "synthetic_small":   # v1: every particle darker than the background
            assert gray[inside].mean() < gray[outside].mean() - 0.08
        else:   # v2: both polarities by design; the particle texture lies inside the GT contours (a mask shifted
            assert gray[inside].std() > 2 * gray[outside].std()   # off the particles gives a ratio near 1)


@SMALL
def test_multiview_frames(small, request):
    ds = request.getfixturevalue(small)
    im = ds.images.set_index("image_id")
    v = ds.views.merge(ds.particles[["particle_id", "image_id"]].rename(columns={"image_id": "primary"}), on="particle_id")
    assert (im.loc[v[v.view_idx == 0].image_id, "modality"] == "ferrograph_optical").all()
    extra = v[v.view_idx > 0]
    assert len(extra) > 0 and extra.image_id.is_unique                       # one silhouette per frame image
    assert (im.loc[extra.image_id, "modality"] == "multiview_frame").all()
    assert set(ds.particles.n_views) <= {1, ds.meta["generator_config"]["multiview"]["n_views"]}
    upp_s = im.loc[extra.primary, "um_per_px"].to_numpy()
    upp_f = im.loc[extra.image_id, "um_per_px"].to_numpy()
    assert np.allclose(extra.um_per_px, upp_f) and (upp_f >= upp_s - 1e-12).all()
    assert np.allclose(im.loc[extra.image_id, "magnification"], im.loc[extra.primary, "magnification"].to_numpy() * upp_s / upp_f)
    frame = ds.meta["generator_config"]["multiview"]["frame_size_px"]
    for r in extra.itertuples():
        xy = WearDataset.contour(r)
        assert xy.min() > 1 and xy.max() < frame - 2      # silhouette never touches the frame border
        assert S._shoelace(xy) > 0


@SMALL
def test_semref_only_with_sem_or_eds(small, request):
    ds = request.getfixturevalue(small)
    p = ds.particles.set_index("particle_id")
    ref = ds.labels[ds.labels.rater_id == "SEMREF"]
    assert len(ref) > 0
    assert set(ref.particle_id) == set(p.index[p.has_sem | p.has_eds])
    assert (ref.evidence != "optical").all()
    assert ds.labels_adjudicated.n_raters.max() <= 2       # adjudication uses raters A, B only
    eds = p[p.has_eds]
    assert np.allclose(eds.filter(like="eds_").sum(axis=1), 1) and p[~p.has_eds].eds_Fe.isna().all()
    assert (eds.eds_O[eds.true_mechanism == "oxide"] > 0.1).all()


def test_rater_accuracy_and_adjudication():
    ann = S.load_synthetic_config("smoke")["annotation"]
    rng = np.random.default_rng(0)
    n = 20000
    true = rng.integers(0, len(MECHANISMS), n)
    sem, eds = rng.random(n) < 0.15, rng.random(n) < 0.3
    pids = np.array([f"p{i}" for i in range(n)], dtype=object)
    labels, adj = S._annotate_labels(rng, pids, true, sem, eds, ann)
    truth = pd.Series(np.array(MECHANISMS)[true], index=pids)
    for rater, acc in (("A", ann["rater_accuracy"][0]), ("B", ann["rater_accuracy"][1]), ("SEMREF", ann["reference_accuracy"])):
        lab = labels[labels.rater_id == rater]
        ok = lab.label.to_numpy() == truth.loc[lab.particle_id].to_numpy()
        assert abs(ok.mean() - acc) < 0.015, rater
        wrong = lab[~ok]
        assert all(l in S.CONFUSABLE[t] for l, t in zip(wrong.label, truth.loc[wrong.particle_id]))
        assert lab.confidence[ok].between(3, 5).all() and lab.confidence[~ok].between(1, 3).all()
    assert len(labels[labels.rater_id == "SEMREF"]) == (sem | eds).sum()
    a = adj.set_index("particle_id")
    acc_adj = (a.label == truth.loc[a.index]).mean()
    assert acc_adj > ann["rater_accuracy"][0]
    assert set(a.method) == {"unanimous", "adjudicated"}           # label/double fractions are 1.0 in smoke
    assert (a.agreement[a.method == "unanimous"] == 1).all() and (a.agreement[a.method == "adjudicated"] < 1).all()


@RENDERERS
def test_cumulative_and_counterface_datasets_valid(tmp_path, renderer):
    cfg = tiny_cfg({"debris": {"sampling": "cumulative", "counterface_wear_ratio": 0.5}, **v(renderer)})
    ds = S.generate_dataset(cfg, tmp_path, seed=2)
    assert ds.validate() == [] and (ds.runs.debris_sampling == "cumulative").all()
    rt = ds.run_table()
    assert rt.counterface_mass_loss_mg.notna().all() and (ds.grav_volume_mm3().to_numpy() > rt.wear_volume_mm3).all()
    sim = S._simulate(cfg, 2)
    assert (sim["parts"].src == 1).any() and (sim["parts"].src == 0).any()
    bad = deep_merge(cfg, {"materials": [dict(cfg["materials"][0], counterface_id="UNKNOWN")]})
    with pytest.raises(ValueError, match="counterface"):
        S._simulate(bad, 0)


@pytest.mark.slow
@RENDERERS
def test_smoke_dataset_fractal_stats(tmp_path, renderer):
    ds = S.generate_dataset(deep_merge(S.load_synthetic_config("smoke"), v(renderer)), tmp_path, seed=0)
    fr = ds.meta["stats"]["fractal"]
    assert fr["spearman_mag200"] >= 0.8 and fr["spearman_mag500"] >= 0.8


@RENDERERS
def test_simulate_tables_matches_the_rendered_generator(tmp_path, renderer):
    from wearpg.baselines.run_level import aggregate_sample_features

    cfg = tiny_cfg(v(renderer))
    fast, desc = S.simulate_tables(cfg, seed=3)
    full = S.generate_dataset(cfg, tmp_path / "d", seed=3)
    pd.testing.assert_frame_equal(fast.runs, full.runs)
    assert (fast.samples.n_particles_total >= full.samples.n_particles_total).all()  # render drops oversized only
    assert set(full.particles.particle_id) <= set(desc.index) == set(fast.particles.particle_id)
    a_fast = aggregate_sample_features(fast, desc).log_volume_index.to_numpy()
    a_full = aggregate_sample_features(full, desc.reindex(full.particles.particle_id)).log_volume_index.to_numpy()
    same = fast.samples.n_particles_total.to_numpy() == full.samples.n_particles_total.to_numpy()
    assert same.sum() >= 0.9 * len(same)
    np.testing.assert_allclose(a_fast[same], a_full[same], rtol=1e-6)  # identical areas -> identical HT index


def test_v2_changes_rendering_only(tmp_path):
    """Renderer v2 keeps v1's physics, budgets, particles and labels (same cfg and seed): only the images, the view-0
    placement (a rigid motion of the identical outline) and the measured D differ; ferromagnetic debris forms chains
    (touching / overlapping particles = fewer GT-raster components than particles)."""
    cfg = tiny_cfg({"debris": {"counterface_wear_ratio": 0.5}})   # steel counterface debris in bronze samples
    a = S.generate_dataset(cfg, tmp_path / "v1", seed=4)
    b = S.generate_dataset(deep_merge(cfg, v("v2")), tmp_path / "v2", seed=4)
    for t in ("specimens", "runs", "samples", "labels", "labels_adjudicated"):
        pd.testing.assert_frame_equal(getattr(a, t), getattr(b, t))
    keep = [c for c in a.particles.columns if c not in ("image_id", "true_fractal_D")]
    pd.testing.assert_frame_equal(a.particles[keep], b.particles[keep])
    va, vb = (d.views[d.views.view_idx == 0].set_index("particle_id") for d in (a, b))
    size = cfg["imaging"]["image_size_px"]
    for pid in va.index:
        xa, xb = WearDataset.contour(va.loc[pid]), WearDataset.contour(vb.loc[pid])
        ra, rb = (np.linalg.norm(x - x.mean(axis=0), axis=1) for x in (xa, xb))
        assert np.allclose(ra, rb) and np.isclose(S._shoelace(xa), S._shoelace(xb))   # same outline, moved rigidly
        assert xb.min() >= 0 and xb.max() <= size - 1                                  # fully inside its field
    fov = b.images[b.images.modality == "ferrograph_optical"]
    assert (fov.illumination == "bichromatic").all()
    ferro = b.particles.merge(b.samples[["sample_id", "run_id"]]).merge(b.runs[["run_id", "specimen_id"]]).merge(
        b.specimens[["specimen_id", "ferromagnetic"]])
    merged = 0
    for image_id, g in ferro[ferro.ferromagnetic].groupby("image_id"):
        mask = np.zeros((size, size), bool)
        for r in vb.loc[g.particle_id].itertuples():
            mask |= S._fill(WearDataset.contour(r), size, size)
        merged += len(g) - ndimage.label(mask)[1]
    assert merged > 0


def test_chains_touch_and_fit():
    """_chains: the cores of one unit's members form one 4-connected component; no unit exceeds the limit."""
    rng = np.random.default_rng(0)
    alphas = []
    for _ in range(14):
        a = float(rng.uniform(5, 30))
        yy, xx = np.mgrid[-a - 2:a + 3, -a / 1.6 - 2:a / 1.6 + 3]
        alphas.append(((yy / a) ** 2 + (xx * 1.6 / a) ** 2 <= 1).astype(np.float32))
    units = S._chains(rng, alphas, list(range(14)), np.array([0.0, 1.0]), 150)
    assert sum(len(m) for _, m in units) == 14 and len(units) < 14
    for mask, mem in units:
        assert max(mask.shape) <= 150 + 6
        core = np.zeros(mask.shape, bool)
        for i, dy, dx in mem:
            h, w = alphas[i].shape
            core[dy:dy + h, dx:dx + w] |= alphas[i] >= 0.5
        assert ndimage.label(core)[1] == 1
