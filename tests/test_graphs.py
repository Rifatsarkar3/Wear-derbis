import io
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr
from skimage.draw import polygon
from skimage.measure import points_in_poly

import wearpg.graphs as graphs_mod
from wearpg.config import load_config
from wearpg.features.contour import canonicalize, perimeter
from wearpg.features.fractal import divider_dimension
from wearpg.graphs import (EDGE_CHORD, EDGE_INTERIOR, EDGE_RING1, EDGE_RING_MULTI, GraphConfig, GraphSelection,
                           contour_graph, pack_graphs, select_graph, selected_dims, unpack_graph)
from wearpg.schema import WearDataset

ROOT = Path(__file__).resolve().parents[1]
CFG = GraphConfig()
N = CFG.n_nodes
NAMES = CFG.node_feature_names()
COL = {n: i for i, n in enumerate(NAMES)}
TEX = [i for i, n in enumerate(NAMES) if n.startswith("tex_")]


def ellipse(a=40.0, b=10.0, n=300, c=(100.0, 80.0)):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([c[0] + a * np.cos(t), c[1] + b * np.sin(t)])


def rough(seed=0, r0=50.0, n=400, c=(100.0, 100.0)):
    """Asymmetric rough blob (unique principal axis and max-radius vertex)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    noise = sum(rng.normal() * np.cos(k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(3, 30))
    r = r0 * (1 + 0.3 * np.cos(2 * t + 0.4) + 0.1 * np.cos(3 * t + 1.0) + 0.08 * noise)
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])


def ribbon(r=60.0, w=6.0, arc=2.0, n=200, c=(100.0, 100.0)):
    """Thin curved band (cutting-debris-like)."""
    t = np.linspace(-arc / 2, arc / 2, n)
    outer = np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])
    inner = np.column_stack([c[0] + (r - w) * np.cos(t[::-1]), c[1] + (r - w) * np.sin(t[::-1])])
    return np.vstack([outer, inner])


def disc_image(xy, shape=(200, 200), inside=0.3, outside=0.8):
    img = np.full(shape, outside, np.float32)
    rr, cc = polygon(xy[:, 1], xy[:, 0], shape)
    img[rr, cc] = inside
    return img


def kind_bits(kind):
    return torch.stack([(kind & b) > 0 for b in (EDGE_RING1, EDGE_RING_MULTI, EDGE_CHORD, EDGE_INTERIOR)], 1).float()


def test_full_graph_contract():
    xy = ellipse()
    g = contour_graph(xy, 0.5, CFG, disc_image(xy))
    n = N + CFG.interior_nodes
    e = g["edge_index"].shape[1]
    assert g["x"].shape == (n, len(NAMES)) and g["x"].dtype == torch.float32
    assert g["pos"].shape == (n, 2) and g["edge_index"].dtype == torch.long and g["edge_index"].shape[0] == 2
    assert g["edge_attr"].shape == (e, len(CFG.edge_feature_names())) and g["edge_kind"].shape == (e,)
    assert (g["node_type"][:N] == 0).all() and (g["node_type"][N:] == 1).all()
    assert torch.isfinite(g["x"]).all() and torch.isfinite(g["edge_attr"]).all()
    assert torch.equal(g["x"][:, COL["is_interior"]], g["node_type"].float())
    kind = g["edge_kind"]
    assert (kind > 0).all() and torch.equal(g["edge_attr"][:, 3:], kind_bits(kind))
    touches_interior = (g["edge_index"] >= N).any(0)
    assert torch.equal(touches_interior, (kind & EDGE_INTERIOR) > 0)
    assert (g["edge_attr"][touches_interior, 1:3] == 0).all()  # arc_frac, tangent_align: boundary pairs only
    b = ~touches_interior
    hop = (g["edge_index"][0, b] - g["edge_index"][1, b]).abs()
    assert torch.allclose(g["edge_attr"][b, 1], torch.minimum(hop, N - hop).float() / N)
    # canonical frame: boundary RMS radius ~ 1, centroid ~ 0
    assert abs(float(g["pos"][:N].norm(dim=1).pow(2).mean().sqrt()) - 1) < 0.01
    assert float(g["pos"][:N].mean(0).abs().max()) < 0.01


@pytest.mark.parametrize("transform", ["rotate", "scale", "translate", "cyclic_shift", "combined"])
def test_boundary_node_features_similarity_and_start_point_invariant(transform):
    xy = rough(1)
    a = np.deg2rad(37.0)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = xy.mean(axis=0)
    moved = {"rotate": (xy - c) @ rot.T + c, "scale": xy * 3.7, "translate": xy + [123.4, -56.7],
             "cyclic_shift": np.roll(xy, 97, axis=0),
             "combined": np.roll((xy - c) @ rot.T * 0.4 + [10.0, 20.0], 31, axis=0)}[transform]
    ref = contour_graph(xy, 0.5, CFG)["x"][:N].numpy()
    got = contour_graph(moved, 0.5, CFG)["x"][:N].numpy()
    np.testing.assert_allclose(np.sort(got, axis=0), np.sort(ref, axis=0), rtol=1e-4, atol=1e-4)


def test_width_chords_cross_the_ellipse():
    g = contour_graph(ellipse(40, 10), 0.5, CFG)
    chord = (g["edge_kind"] & EDGE_CHORD) > 0
    i, j = g["edge_index"][:, chord]
    hop = torch.minimum((i - j).abs(), N - (i - j).abs())
    assert chord.sum() >= N  # nearly every node has a chord (each counted in both directions)
    assert hop.float().median() > N / 8
    pos = g["pos"]  # canonical frame: major axis = x, so a width chord joins the two long sides
    assert (pos[i, 1] * pos[j, 1] < 0).float().mean() > 0.8


@pytest.mark.parametrize("shape", ["ellipse", "rough", "ribbon", "tiny"])
def test_no_duplicate_pairs_no_self_loops_and_symmetric(shape):
    xy = {"ellipse": ellipse(), "rough": rough(2), "ribbon": ribbon(), "tiny": ellipse(3, 2, n=8)}[shape]
    g = contour_graph(xy, 0.5, CFG)
    assert torch.isfinite(g["x"]).all() and torch.isfinite(g["edge_attr"]).all()  # incl. straight runs (tiny)
    ei = g["edge_index"]
    pairs = list(map(tuple, ei.T.tolist()))
    assert len(set(pairs)) == len(pairs) and (ei[0] != ei[1]).all()
    half = ei.shape[1] // 2
    assert torch.equal(ei[:, half:], ei[:, :half].flip(0)) and torch.equal(g["edge_attr"][half:], g["edge_attr"][:half])
    assert (ei[0, :half] < ei[1, :half]).all()
    for hop in CFG.ring_hops:  # every ring pair of every hop is present (once)
        assert all((min(i, (i + hop) % N), max(i, (i + hop) % N)) in set(pairs) for i in range(N))


def test_coinciding_kinds_are_or_ed_into_one_pair():
    cfg = GraphConfig(n_nodes=8, ring_hops=(1, 2, 4), interior_nodes=0)
    g = contour_graph(ellipse(20, 18, n=64), 0.5, cfg)
    kind = g["edge_kind"]
    assert g["x"].shape[0] == 8 and (g["node_type"] == 0).all()
    # near-circle: the width chord of node i is the opposite node i + 4, i.e. the hop-4 ring pair
    assert (((kind & EDGE_CHORD) > 0) & ((kind & EDGE_RING_MULTI) > 0)).any()
    assert g["edge_index"].shape[1] == 2 * (8 + 8 + 4)  # ring 1, ring 2, ring 4 (n/2: 4 pairs); chords merged


def test_selection_views_and_dims():
    xy = rough(3)
    g = contour_graph(xy, 0.5, CFG, disc_image(xy))
    n_all = g["x"].shape[0]
    full = select_graph(g, CFG, GraphSelection())
    assert torch.equal(full.x, g["x"]) and torch.equal(full.edge_index, g["edge_index"])
    assert torch.equal(full.edge_attr, g["edge_attr"]) and full.node_type.dtype == torch.long
    ring = select_graph(g, CFG, GraphSelection(edges="ring"))
    assert ring.edge_index.shape[1] == 2 * N  # the hop-1 cycle in both directions
    assert torch.equal(ring.edge_attr[:, 3:], torch.tensor([1.0, 0, 0, 0]).expand(2 * N, 4))
    none = select_graph(g, CFG, GraphSelection(edges="none"))
    assert none.edge_index.shape == (2, 0) and none.edge_attr.shape == (0, 7) and none.num_nodes == n_all
    no_int = select_graph(g, CFG, GraphSelection(interior=False))
    assert no_int.num_nodes == N and (no_int.edge_index < N).all() and (no_int.node_type == 0).all()
    assert (no_int.edge_attr[:, 6] == 0).all()
    drops = [(GraphSelection(roughness=False), {"roughness"}),
             (GraphSelection(boundary_texture=False), {n for n in NAMES if n.startswith("tex_")}),
             (GraphSelection(interior=False), {"is_interior"})]
    for sel, dropped in drops:
        d = select_graph(g, CFG, sel)
        keep = [COL[n] for n in NAMES if n not in dropped]
        assert torch.equal(d.x, g["x"][:d.num_nodes, keep])
    for edges, flags in itertools.product(("full", "ring", "none"), itertools.product((True, False), repeat=3)):
        sel = GraphSelection(edges, *flags)
        d = select_graph(g, CFG, sel)
        assert (d.x.shape[1], d.edge_attr.shape[1]) == selected_dims(CFG, sel)
        assert len(set(map(tuple, d.edge_index.T.tolist()))) == d.edge_index.shape[1]  # never parallel edges
    with pytest.raises(ValueError):
        GraphSelection(edges="knn")


def test_pack_round_trip_weights_only():
    gs = {"a_v0": contour_graph(ellipse(), 0.5, CFG, disc_image(ellipse())),
          "b_v1": contour_graph(rough(4), 0.2, CFG),
          "c_v0": contour_graph(ellipse(3, 2, n=8), 0.5, GraphConfig(interior_nodes=0))}
    packed = pack_graphs(gs)
    buf = io.BytesIO()
    torch.save(packed, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=True)
    for p in (packed, loaded):
        for vid, g in gs.items():
            u = unpack_graph(p, vid)
            assert u.keys() == g.keys()
            for k in g:
                assert u[k].dtype == g[k].dtype and torch.equal(u[k], g[k]), (vid, k)


def test_texture_node_features_sample_the_image():
    xy = ellipse(40, 25)
    x = contour_graph(xy, 0.5, CFG, disc_image(xy, inside=0.3, outside=0.8))["x"].numpy()
    b = x[:N]
    for i in range(len(CFG.texture_depths_frac)):
        assert np.median(b[:, COL[f"tex_in_{i}"]]) == pytest.approx(0.3, abs=0.02)
    assert np.median(b[:, COL["tex_out"]]) == pytest.approx(0.8, abs=0.02)
    assert np.median(b[:, COL["tex_grad"]]) > 0.1
    np.testing.assert_allclose(x[N:, COL["tex_in_0"]], 0.3, atol=1e-6)  # interior nodes: local grey mean
    plain = contour_graph(xy, 0.5, CFG)["x"].numpy()
    assert (plain[:, TEX] == 0).all()
    np.testing.assert_array_equal(np.delete(plain, TEX, 1), np.delete(x, TEX, 1))  # geometry ignores the image
    # thin band (6 px wide): the deepest inward depth 0.3 x RMS radius would exit through the opposite side, but it is
    # clipped to 0.8 x the local width, so the sample stays on the particle
    rib = ribbon()
    assert CFG.texture_depths_frac[-1] * canonicalize(rib)[1]["scale"] > 1.5 * 6.0
    xr = contour_graph(rib, 0.5, CFG, disc_image(rib))["x"].numpy()[:N]
    assert np.median(xr[:, COL["tex_in_2"]]) < 0.45


@pytest.mark.parametrize("shape", ["ellipse", "rough", "ribbon"])
def test_interior_nodes_inside_and_connected(shape):
    xy = {"ellipse": ellipse(), "rough": rough(5), "ribbon": ribbon()}[shape]
    g = contour_graph(xy, 0.5, CFG)
    pos = g["pos"].numpy()
    inner = pos[N:]
    assert len(inner) == CFG.interior_nodes and points_in_poly(inner, pos[:N]).all()
    assert (g["x"][N:, COL["convexity_defect"]] > 0).all()  # distance to the boundary / RMS radius
    ei = g["edge_index"]
    for r in range(N, N + len(inner)):
        nb = ei[1, ei[0] == r]
        assert (nb < N).sum() >= CFG.interior_knn and (nb >= N).sum() >= CFG.interior_knn


def test_graph_config_and_selection_from_cfg(monkeypatch):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    gc = GraphConfig.from_cfg(cfg)
    assert gc == GraphConfig()  # dataclass defaults mirror configs/default.yaml
    key = gc.key()
    assert key == GraphConfig().key() and GraphConfig(interior_nodes=8).key() != key
    monkeypatch.setattr(graphs_mod, "FEATURE_VERSION", graphs_mod.FEATURE_VERSION + 1)
    assert gc.key() != key
    assert NAMES == ["r_norm", "curv_s1", "curv_s2", "curv_s4", "curv_s8", "turning", "convexity_defect", "roughness",
                     "tex_in_0", "tex_in_1", "tex_in_2", "tex_out", "tex_grad", "is_interior"]
    assert CFG.edge_feature_names() == ["dist", "arc_frac", "tangent_align", "is_ring1", "is_ring_multi",
                                        "is_chord", "is_interior"]
    off = load_config(ROOT / "configs" / "default.yaml", ["model.edges=ring", "model.use_interior_nodes=false",
                                                         "model.use_fractal_features=false",
                                                         "model.use_texture_branch=false"])
    assert GraphSelection.from_cfg(cfg) == GraphSelection()
    assert GraphSelection.from_cfg(off) == GraphSelection("ring", False, False, False)
    with pytest.raises(TypeError):
        GraphConfig.from_cfg({"graph": {"n_node": 64}})


def test_synthetic_fractal_calibration_cross_check(synthetic_small):
    """features.fractal divider D of the generator's view-0 GT contours (perimeter >= 100 px) tracks D_target."""
    ds = synthetic_small
    mag = ds.particle_hierarchy().set_index("particle_id").magnification
    v0 = ds.views[ds.views.view_idx == 0].set_index("particle_id")
    rows = []
    for pid, target in zip(ds.particles.particle_id, pd.to_numeric(ds.particles.true_fractal_D_target)):
        xy = WearDataset.contour(v0.loc[pid])
        if perimeter(xy) >= 100:
            d = divider_dimension(xy)
            if d["valid"]:
                rows.append((mag[pid], target, d["D"]))
    df = pd.DataFrame(rows, columns=["magnification", "target", "D"])
    assert set(df.magnification) == {200.0, 500.0}
    for m, g in df.groupby("magnification"):
        rho = spearmanr(g.target, g.D)[0]
        assert len(g) >= 20 and rho >= 0.8, (m, len(g), rho)
