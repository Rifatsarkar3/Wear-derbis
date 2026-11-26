import math
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from wearpg.batch import ModelDims, RunBatch
from wearpg.config import load_config
from wearpg.models import MechanismLinkedModel, ParticleGNN, make_aggregator, make_image_branch, pool_views
from wearpg.models.heads import CoralHead, coral_expected, coral_probs
from wearpg.models.losses import class_weights_from_targets, compute_losses, ordinal_coral_loss

ROOT = Path(__file__).resolve().parents[1]
DIMS = ModelDims(node_dim=9, edge_dim=7, particle_feat_dim=6, sample_feat_dim=5, cond_dim=4, n_classes=7,
                 n_materials=3, crop_size=32)
# DESIGN sec. 11 (experiments.VARIANTS): every override touches model.* or loss.*
VARIANTS = {
    "full": [], "no_graph": ["model.use_graph=false"], "edges_ring": ["model.edges=ring"],
    "edges_none": ["model.edges=none"], "no_interior": ["model.use_interior_nodes=false"],
    "no_fractal": ["model.use_fractal_features=false"], "no_texture": ["model.use_texture_branch=false"],
    "no_multiview": ["model.use_multiview=false"], "no_composition": ["model.use_composition=false"],
    "no_run_aggregation": ["model.use_run_aggregation=false"],
    "no_physics": ["model.physics_head=false", "loss.consistency=0", "loss.monotone=0"],
    "no_consistency": ["loss.consistency=0"], "with_conditions": ["model.use_operating_conditions=true"],
    "no_mechanism_link": ["model.mechanism_link=false"],
}
# graph-selection / feature-group flags only change the inputs (ModelDims, view count, edges) the datasets produce
VARIANT_INPUTS = {
    "edges_none": ({}, {"edge_prob": 0.0}), "no_interior": ({"node_dim": 8}, {}),
    "no_fractal": ({"node_dim": 8, "particle_feat_dim": 4}, {}),
    "no_texture": ({"node_dim": 4, "particle_feat_dim": 3}, {"crops": False}),
    "no_multiview": ({"particle_feat_dim": 5}, {"max_views": 1}), "no_composition": ({"particle_feat_dim": 5}, {}),
}


def make_cfg(*overrides):
    return load_config(ROOT / "configs" / "default.yaml", ["model.hidden=32", "model.gnn_layers=2", *overrides])


def random_graph(n, dims, n_edges):
    return Data(x=torch.randn(n, dims.node_dim), edge_index=torch.randint(0, n, (2, n_edges)),
                edge_attr=torch.randn(n_edges, dims.edge_dim), pos=torch.randn(n, 2),
                node_type=torch.zeros(n, dtype=torch.long))


def make_batch(dims=DIMS, runs=((3, False), (2, True), (1, False)), seed=0, crops=True, soft=False, edge_prob=0.7,
               max_views=3, targets=True) -> RunBatch:
    """RunBatch with random PyG view graphs (some with E = 0); runs = [(n observed samples, cumulative)], a run with 0
    samples has all its samples absent. Interval sample ds = 240 m and L = 240 m * max(n, 3)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    graphs, v2p, p2s, s2r, time = [], [], [], [], []
    for r, (n_samples, _) in enumerate(runs):
        for s in range(n_samples):
            for _ in range(rng.integers(1, 5)):
                for _ in range(rng.integers(1, max_views + 1)):
                    n = int(rng.integers(5, 11))
                    graphs.append(random_graph(n, dims, int(rng.integers(1, 3 * n)) if rng.random() < edge_prob else 0))
                    v2p.append(len(p2s))
                p2s.append(len(s2r))
            s2r.append(r)
            time.append(600.0 * (s + 1))
    n_p, n_s, n_r = len(p2s), len(s2r), len(runs)
    cumulative = torch.tensor([c for _, c in runs])
    s2r_t = torch.tensor(s2r, dtype=torch.long)
    mask = torch.rand(n_p) < 0.5
    comp = torch.rand(n_p, dims.n_elements)
    comp = torch.where(mask[:, None], comp / comp.sum(1, keepdim=True), 0.0)
    log_area = torch.log(torch.empty(n_p).uniform_(5, 500))
    mech = torch.randint(-1, dims.n_classes, (n_p,))
    vol_t = torch.where(torch.rand(n_p) < 0.4, 1.5 * log_area + math.log(0.3) + 0.2 * torch.randn(n_p), math.nan)
    rate_s = torch.where((torch.rand(n_s) < 0.7) & ~cumulative[s2r_t], torch.empty(n_s).uniform_(-6, -4), math.nan)
    sev = torch.randint(-1, 3, (n_s,))
    grav = torch.log(torch.empty(n_r).uniform_(0.01, 1.0))
    rate_r = torch.empty(n_r).uniform_(-6, -4)
    soft_t = None
    if soft:
        soft_t = torch.distributions.Dirichlet(torch.ones(dims.n_classes)).sample((n_p,))
        soft_t[mech < 0] = 0.0
    if not targets:
        mech, sev = torch.full_like(mech, -1), torch.full_like(sev, -1)
        vol_t, rate_s, grav, rate_r = (torch.full_like(x, math.nan) for x in (vol_t, rate_s, grav, rate_r))
    return RunBatch(
        graphs=Batch.from_data_list(graphs), view_to_particle=torch.tensor(v2p, dtype=torch.long),
        crops=torch.rand(n_p, dims.crop_channels, dims.crop_size, dims.crop_size) if crops else None,
        particle_feats=torch.randn(n_p, dims.particle_feat_dim), composition=comp, composition_mask=mask,
        log_area_um2=log_area, mech_target=mech, mech_soft=soft_t, log_volume_target=vol_t,
        particle_to_sample=torch.tensor(p2s, dtype=torch.long),
        sample_to_run=s2r_t, log_obs_fraction=torch.log(torch.empty(n_s).uniform_(1e-4, 1e-2)),
        log_subsample_factor=torch.log(torch.empty(n_s).uniform_(1.0, 2.0)), severity_target=sev,
        log10_rate_sample_target=rate_s, sample_time_s=torch.tensor(time), sample_distance_m=torch.full((n_s,), 240.0),
        sample_distance_end_m=torch.tensor(time) * 0.4,  # 240 m per 600 s interval
        sample_log_q=torch.full((n_s,), math.log(0.01)), sample_feats=torch.randn(n_s, dims.sample_feat_dim),
        log_grav_volume_mm3=grav, log10_rate_run_target=rate_r,
        load_N=torch.tensor([(20.0, 60.0, 120.0)[r % 3] for r in range(n_r)]),
        sliding_distance_m=torch.tensor([240.0 * max(n, 3) for n, _ in runs]),
        run_cond_feats=torch.randn(n_r, dims.cond_dim), material_idx=torch.arange(n_r) % dims.n_materials,
        magnetic_idx=torch.arange(n_r) % dims.n_magnetic, run_cumulative=cumulative,
        mono_group=torch.zeros(n_r, dtype=torch.long), condition_idx=torch.arange(n_r),
        particle_ids=[f"p{i}" for i in range(n_p)], sample_ids=[f"s{i}" for i in range(n_s)],
        run_ids=[f"r{i}" for i in range(n_r)])


def assert_finite_grads(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def forward_loss_backward(cfg, dims=DIMS, **batch_kw):
    torch.manual_seed(0)
    model = MechanismLinkedModel(cfg, dims)
    batch = make_batch(dims, **batch_kw)
    out = model(batch)
    total, logs = compute_losses(out, batch, cfg, class_weights_from_targets(batch.mech_target, dims.n_classes))
    assert torch.isfinite(total) and all(math.isfinite(v) for v in logs.values())
    total.backward()
    assert_finite_grads(model)
    return model, batch, out, logs


# ----------------------------------------------------------------------------- components


@pytest.mark.parametrize("conv", ["gine", "gatv2"])
def test_particle_gnn_shapes_zero_edges_and_grad(conv):
    gnn = ParticleGNN(DIMS.node_dim, DIMS.edge_dim, hidden=16, layers=2, conv=conv)
    for n_edges in ((12, 0, 30), (0, 0, 0)):  # mixed, then every graph with E = 0
        g = Batch.from_data_list([random_graph(8, DIMS, e) for e in n_edges])
        out = gnn(g.x, g.edge_index, g.edge_attr, g.batch, g.num_graphs)
        assert out.shape == (3, gnn.out_dim) and gnn.out_dim == 48 and torch.isfinite(out).all()
    out.sum().backward()
    assert gnn.inp.weight.grad is not None and torch.isfinite(gnn.inp.weight.grad).all()
    assert ParticleGNN(4, 3, hidden=16, pooling=("mean",)).out_dim == 16
    with pytest.raises(ValueError):
        ParticleGNN(4, 3, hidden=18, conv="gatv2")


@pytest.mark.parametrize("conv", ["gine", "gatv2"])
def test_particle_gnn_node_permutation_invariance(conv):
    torch.manual_seed(1)
    gnn = ParticleGNN(DIMS.node_dim, DIMS.edge_dim, hidden=16, layers=3, conv=conv).eval()
    g = random_graph(10, DIMS, 25)
    perm = torch.randperm(10)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(10)
    batch = torch.zeros(10, dtype=torch.long)
    a = gnn(g.x, g.edge_index, g.edge_attr, batch)
    b = gnn(g.x[perm], inv[g.edge_index], g.edge_attr, batch)
    assert torch.allclose(a, b, atol=1e-5)


def test_pool_views_mean_max():
    out = pool_views(torch.tensor([[1.0, 2.0], [3.0, 0.0], [5.0, 5.0]]), torch.tensor([0, 0, 1]), 2)
    assert torch.equal(out, torch.tensor([[2.0, 1.0, 3.0, 2.0], [5.0, 5.0, 5.0, 5.0]]))


@pytest.mark.parametrize("name", ["gated_attention", "mean", "set_transformer"])
def test_aggregators_bag_invariance_weights_and_empty_bag(name):
    torch.manual_seed(0)
    agg = make_aggregator(name, 8, 16).eval()
    x, index = torch.randn(7, 8), torch.tensor([2, 0, 2, 1, 0, 2, 0])  # unsorted, bag 3 empty
    pooled, w = agg(x, index, 4)
    assert pooled.shape == (4, 8) and w.shape == (7,) and torch.isfinite(pooled).all()
    assert torch.allclose(torch.zeros(4).index_add(0, index, w)[:3], torch.ones(3), atol=1e-6)
    perm = torch.randperm(7)
    pooled_p, w_p = agg(x[perm], index[perm], 4)
    assert torch.allclose(pooled, pooled_p, atol=1e-5) and torch.allclose(w[perm], w_p, atol=1e-6)
    if name != "set_transformer":  # SetTransformer decodes a learned seed even for an empty bag (finite)
        assert pooled[3].abs().max() == 0
    with pytest.raises(ValueError):
        make_aggregator("max", 8, 16)


@pytest.mark.parametrize("name", ["small_cnn", "resnet18"])
def test_image_branch_shapes(name):
    net = make_image_branch(name, in_ch=4, out_dim=16, img_size=32)
    assert net(torch.rand(2, 4, 32, 32)).shape == (2, 16)


def test_coral_head_probs_expected_and_loss():
    logits = torch.tensor([[-1.0, 2.0], [3.0, -3.0], [0.0, 0.0]])  # first row not rank-consistent
    p = coral_probs(logits)
    assert (p >= 0).all() and torch.allclose(p.sum(1), torch.ones(3))
    assert torch.allclose(coral_expected(logits), p @ torch.arange(3.0))
    head = CoralHead(4, 3)
    assert head(torch.randn(5, 4)).shape == (5, 2) and head.bias[0] > head.bias[1]
    lg = torch.randn(4, 2, requires_grad=True)
    target = torch.tensor([0, 2, -1, 1])
    assert torch.isclose(ordinal_coral_loss(lg, target), ordinal_coral_loss(lg[[0, 1, 3]], target[[0, 1, 3]]))
    # hand value: y = 2 -> levels [1, 1]
    ref = torch.nn.functional.binary_cross_entropy_with_logits(lg[1], torch.ones(2), reduction="sum")
    assert torch.isclose(ordinal_coral_loss(lg[[1]], target[[1]]), ref)
    empty = ordinal_coral_loss(lg, torch.full((4,), -1))
    empty.backward()
    assert empty.item() == 0.0 and torch.isfinite(lg.grad).all()


def test_class_weights_from_targets():
    w = class_weights_from_targets(np.array([0, 0, 1, -1, 3]), 5)
    assert w.dtype == torch.float32
    assert torch.allclose(w, torch.tensor([4 / 6, 4 / 3, 0.0, 4 / 3, 0.0]))
    assert torch.equal(class_weights_from_targets([-1, -1], 3), torch.zeros(3))


# ----------------------------------------------------------------------------- full model


def test_forward_contract_shapes_and_values():
    cfg = make_cfg()
    model, batch, out, logs = forward_loss_backward(cfg)
    n_p, n_s, n_r, h = batch.n_particles, batch.n_samples, batch.n_runs, 32
    shapes = {"particle_emb": (n_p, h), "mech_logits": (n_p, 7), "log_volume": (n_p,), "sample_emb": (n_s, h),
              "severity_logits": (n_s, 2), "log10_rate_sample": (n_s,), "run_emb": (n_r, h), "log10_rate_run": (n_r,),
              "log_eta_run": (n_r,), "log_debris_volume_sample_mm3": (n_s,), "log_debris_volume_run_mm3": (n_r,),
              "coverage_run": (n_r,), "attention": (n_p,), "eta_offset_l2": ()}
    assert {k: tuple(v.shape) for k, v in out.items()} == shapes
    cum_s = batch.run_cumulative[batch.sample_to_run]
    assert torch.isnan(out["log10_rate_sample"][cum_s]).all() and torch.isfinite(out["log10_rate_sample"][~cum_s]).all()
    assert torch.allclose(out["coverage_run"], torch.tensor([1.0, 2 / 3, 1 / 3]))
    assert torch.isfinite(out["log_debris_volume_run_mm3"]).all() and torch.isfinite(out["log10_rate_run"]).all()
    att_sum = torch.zeros(n_s).index_add(0, batch.particle_to_sample, out["attention"].detach())
    assert torch.allclose(att_sum, torch.ones(n_s), atol=1e-5)
    assert set(logs) == {"mechanism", "severity", "rate_run", "rate_sample", "volume", "consistency", "monotone",
                         "eta_offset_l2", "total"}
    with torch.no_grad():  # s(z) starts at ln 0.3 -> v = 0.3 A^1.5; eta = theta[magnetic] + delta[material]
        model.delta.copy_(torch.tensor([0.1, -0.2, 0.3]))
        out = MechanismLinkedModel.forward(model, batch)
    assert torch.allclose(out["log_volume"], 1.5 * batch.log_area_um2 + math.log(0.3), atol=1e-6)
    assert torch.allclose(out["log_eta_run"], model.theta[batch.magnetic_idx] + model.delta[batch.material_idx])
    assert torch.isclose(out["eta_offset_l2"], torch.tensor(0.14))
    assert out["log_debris_volume_run_mm3"].dtype == torch.float32


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_variant_forward_loss_backward(variant):
    dims_kw, batch_kw = VARIANT_INPUTS.get(variant, ({}, {}))
    model, batch, out, logs = forward_loss_backward(make_cfg(*VARIANTS[variant]), replace(DIMS, **dims_kw), **batch_kw)
    assert (model.gnn is None) == (variant == "no_graph") and (model.image is None) == (variant == "no_texture")
    assert (model.aggregator is None) == (variant == "no_run_aggregation")
    assert (model.a_run is None) == (variant == "no_physics")
    if variant == "no_physics":  # mlp-only rates start near the k prior; debris volumes are still computed
        assert torch.isfinite(out["log_debris_volume_run_mm3"]).all()
        assert abs(float(out["log10_rate_run"].detach().mean()) + 5.0) < 1.0


@pytest.mark.parametrize("overrides", [
    ["model.aggregator=set_transformer"], ["model.aggregator=mean"], ["model.conv=gatv2"],
    ["model.per_material_eta=false"], ["model.pooling=[mean]"], ["model.physics_head=false"],
    ["model.use_run_aggregation=false", "model.physics_head=false", "model.use_composition=false"],
    ["model.use_operating_conditions=true", "model.aggregator=set_transformer", "model.conv=gatv2"],
])
def test_model_flags(overrides):
    model, batch, out, _ = forward_loss_backward(make_cfg(*overrides), soft=True)
    if "model.per_material_eta=false" in overrides:
        assert model.delta is None and out["eta_offset_l2"].item() == 0.0
        assert torch.allclose(out["log_eta_run"], model.theta[batch.magnetic_idx])


def test_no_mechanism_link_volume_is_size_only():
    model, batch, out, _ = forward_loss_backward(make_cfg("model.mechanism_link=false"))
    assert torch.equal(model.c_by_class, torch.ones(7))  # metal fraction 1 for every class (oxide too)
    assert model.volume_head.in_features == 1 and model.volume_head.weight.grad is not None
    with torch.no_grad():  # the particle embedding does not reach the volume: other features, same volumes
        other = model(replace(batch, particle_feats=torch.randn_like(batch.particle_feats)))
        assert torch.allclose(other["log_volume"], out["log_volume"])
        model.volume_head.weight.fill_(0.5)
        assert torch.allclose(model(batch)["log_volume"], 2.0 * batch.log_area_um2 + model.volume_head.bias)
    full = MechanismLinkedModel(make_cfg(), DIMS)
    assert full.c_by_class[6].item() == pytest.approx(0.47) and full.volume_head.in_features == 32


def test_texture_branch_without_crops_raises():
    batch = make_batch(crops=False)
    with pytest.raises(ValueError, match="crops"):
        MechanismLinkedModel(make_cfg(), DIMS)(batch)
    MechanismLinkedModel(make_cfg("model.use_texture_branch=false"), DIMS)(batch)


def test_config_errors_are_explicit():
    with pytest.raises(ValueError, match="oxid"):  # a typo would silently drop the oxide correction
        MechanismLinkedModel(make_cfg("physics.metal_fraction={oxid: 0.47}"), DIMS)
    with pytest.raises(ValueError, match="conv"):
        MechanismLinkedModel(make_cfg("model.conv=gcn"), DIMS)


def test_chunked_gnn_under_no_grad_matches_single_call():
    torch.manual_seed(0)
    model = MechanismLinkedModel(make_cfg(), DIMS).eval()
    batch = make_batch()
    with torch.no_grad():
        full = model(batch)
        model.chunk = 3
        chunked = model(batch)
    assert batch.graphs.num_graphs > 3
    for k in ("particle_emb", "mech_logits", "log10_rate_run", "log_debris_volume_run_mm3"):
        assert torch.allclose(full[k], chunked[k], atol=1e-5)


@pytest.mark.parametrize("physics_head", ["true", "false"])
def test_run_with_all_samples_absent_gives_finite_loss_and_grads(physics_head):
    cfg = make_cfg(f"model.physics_head={physics_head}")
    model, batch, out, logs = forward_loss_backward(cfg, runs=((3, False), (0, False), (2, True), (0, True)))
    assert out["coverage_run"][1] == 0 and out["coverage_run"][3] == 0
    assert torch.isnan(out["log_debris_volume_run_mm3"][[1, 3]]).all()
    if physics_head == "true":
        assert torch.isnan(out["log10_rate_run"][[1, 3]]).all() and model.a_run.grad is not None


@pytest.mark.parametrize("soft", [False, True])
def test_all_targets_missing_gives_zero_supervised_losses(soft):
    cfg = make_cfg()
    model, batch, out, logs = forward_loss_backward(cfg, targets=False, soft=soft, runs=((2, False), (0, True)))
    assert all(logs[k] == 0.0 for k in ("mechanism", "severity", "rate_run", "rate_sample", "volume", "consistency"))
    assert logs["total"] == pytest.approx(cfg["loss"]["monotone"] * logs["monotone"])


def test_losses_mask_low_coverage_runs_and_use_soft_labels():
    cfg = make_cfg()
    torch.manual_seed(0)
    model = MechanismLinkedModel(cfg, DIMS)
    batch = make_batch(soft=True)
    out = model(batch)
    only_low = replace(batch, log10_rate_run_target=torch.tensor([math.nan, math.nan, -5.0]))  # run 2: coverage 1/3
    _, logs = compute_losses(out, only_low, cfg)
    assert logs["rate_run"] == 0.0
    _, logs_soft = compute_losses(out, batch, cfg)
    _, logs_hard = compute_losses(out, replace(batch, mech_soft=None), cfg)
    assert logs_soft["mechanism"] > 0 and logs_soft["mechanism"] != logs_hard["mechanism"]
    zero_w = torch.zeros(DIMS.n_classes)  # validation batch whose classes all have weight 0 -> 0, never NaN
    assert compute_losses(out, replace(batch, mech_soft=None), cfg, zero_w)[1]["mechanism"] == 0.0


def test_volume_path_does_not_train_the_classifier():
    torch.manual_seed(0)
    model = MechanismLinkedModel(make_cfg(), DIMS)
    out = model(make_batch())
    out["log_debris_volume_run_mm3"][torch.isfinite(out["log_debris_volume_run_mm3"])].sum().backward()
    assert model.mech_head.weight.grad is None and model.volume_head.weight.grad is not None


def test_fill_unseen_eta_and_transferability():
    model = MechanismLinkedModel(make_cfg(), DIMS)
    with torch.no_grad():
        model.delta.copy_(torch.tensor([0.5, -0.4, 0.3]))
    model.fill_unseen_eta([0, 2])
    assert torch.allclose(model.delta, torch.tensor([0.5, 0.0, 0.3]))
    model.fill_unseen_eta([])
    assert model.delta.abs().max() == 0
    magnetic = [1, 0, 0]  # material 0 ferrous, 1 and 2 non-ferrous
    assert not model.eta_transferable(1, [0], magnetic)
    assert model.eta_transferable(2, [1], magnetic) and model.eta_transferable(0, [0, 1], {0: 1, 1: 0})


def test_bf16_autocast_keeps_physics_float32():
    torch.manual_seed(0)
    model = MechanismLinkedModel(make_cfg(), DIMS)
    batch = make_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(batch)
    for k in ("log_volume", "log_debris_volume_sample_mm3", "log_debris_volume_run_mm3", "coverage_run",
              "log10_rate_run", "log_eta_run", "mech_logits", "severity_logits"):
        assert out[k].dtype == torch.float32, k
    total, _ = compute_losses(out, batch, make_cfg())
    total.backward()
    assert torch.isfinite(total)
    assert_finite_grads(model)


def test_batch_helper_covers_every_runbatch_field():
    batch = make_batch(soft=True)
    assert all(getattr(batch, f.name) is not None for f in fields(RunBatch))
    assert (batch.graphs.edge_attr.shape[1] == DIMS.edge_dim and batch.run_cumulative.any()
            and not batch.run_cumulative.all())
