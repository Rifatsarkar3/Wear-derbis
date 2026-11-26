"""[F] Contour / surface graphs of particle views (DESIGN sec. 8).

One FULL graph per view is built once (`contour_graph`) and cached (`pack_graphs`); every ablation is a
`GraphSelection` view of it applied at collate time (`select_graph`), so graph ablations never rebuild the cache.

Boundary nodes = the view contour resampled to `n_nodes` points at uniform arc length (positive area, canonical
start at the max-radius vertex); geometry lives in the canonical frame of `features.contour.canonicalize`
(area centroid 0, RMS boundary radius 1, major axis +x), so geometric node / edge features are similarity
invariant. Boundary nodes come first (indices 0..n_nodes-1), interior ("surface") nodes after them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.measure import points_in_poly
from torch_geometric.data import Data

from wearpg.features.contour import (canonicalize, convexity_defect, curvature, distance_to_boundary, inward_normals,
                                     local_roughness, signed_area, tangents, turning_angle)
from wearpg.features.texture import boundary_band_profile
from wearpg.segmentation import to_gray
from wearpg.utils import stable_hash

FEATURE_VERSION = 1
EDGE_RING1, EDGE_RING_MULTI, EDGE_CHORD, EDGE_INTERIOR = 1, 2, 4, 8  # edge_kind bit flags
_BITS = torch.tensor([EDGE_RING1, EDGE_RING_MULTI, EDGE_CHORD, EDGE_INTERIOR], dtype=torch.uint8)
_EDGE_NAMES = ["dist", "arc_frac", "tangent_align", "is_ring1", "is_ring_multi", "is_chord", "is_interior"]
_N_CONT = 3  # continuous edge features (dist, arc_frac, tangent_align); the rest are the kind bits
_ALLOWED = {"full": EDGE_RING1 | EDGE_RING_MULTI | EDGE_CHORD | EDGE_INTERIOR, "ring": EDGE_RING1, "none": 0}


@dataclass(frozen=True)
class GraphConfig:
    """BUILD parameters of the full graphs (never changed by ablations). Lengths are fractions of the RMS boundary
    radius; curvature scales and the roughness window are in boundary nodes."""

    n_nodes: int = 128
    ring_hops: tuple = (1, 2, 4, 8, 16)
    curvature_scales: tuple = (1, 2, 4, 8)
    roughness_window: int = 9
    texture_depths_frac: tuple = (0.05, 0.15, 0.3)
    tex_out_frac: float = 0.1
    interior_nodes: int = 16
    interior_knn: int = 4

    @classmethod
    def from_cfg(cls, cfg: dict) -> "GraphConfig":
        """From the experiment config's `graph` block (unknown keys raise TypeError)."""
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["graph"].items()})

    def key(self) -> str:
        """Cache key of the build parameters + FEATURE_VERSION."""
        return stable_hash({**asdict(self), "feature_version": FEATURE_VERSION})

    def node_feature_names(self) -> list[str]:
        return ["r_norm", *[f"curv_s{s:g}" for s in self.curvature_scales], "turning", "convexity_defect",
                "roughness", *[f"tex_in_{i}" for i in range(len(self.texture_depths_frac))], "tex_out", "tex_grad",
                "is_interior"]

    def edge_feature_names(self) -> list[str]:
        return list(_EDGE_NAMES)


@dataclass(frozen=True)
class GraphSelection:
    """ABLATION view of a full graph applied at collate time: edges "full" | "ring" (EDGE_RING1 only) | "none";
    interior=False drops interior nodes (and the then constant is_interior column); roughness=False drops the
    roughness column; boundary_texture=False drops the tex_* columns (incl. the interior grey in tex_in_0)."""

    edges: str = "full"
    interior: bool = True
    roughness: bool = True
    boundary_texture: bool = True

    def __post_init__(self):
        if self.edges not in _ALLOWED:
            raise ValueError(f"edges must be one of {tuple(_ALLOWED)}, got {self.edges!r}")

    @classmethod
    def from_cfg(cls, cfg: dict) -> "GraphSelection":
        m = cfg["model"]
        return cls(edges=str(m["edges"]), interior=bool(m["use_interior_nodes"]),
                   roughness=bool(m["use_fractal_features"]), boundary_texture=bool(m["use_texture_branch"]))


def _node_columns(cfg: GraphConfig, sel: GraphSelection) -> list[int]:
    drop = set()
    if not sel.roughness:
        drop.add("roughness")
    if not sel.boundary_texture:
        drop.update(n for n in cfg.node_feature_names() if n.startswith("tex_"))
    if not sel.interior:
        drop.add("is_interior")
    return [i for i, n in enumerate(cfg.node_feature_names()) if n not in drop]


def selected_dims(cfg: GraphConfig, sel: GraphSelection) -> tuple[int, int]:
    """(node_dim, edge_dim) of `select_graph` output; edge_dim is fixed (kind columns of dropped kinds are 0)."""
    return len(_node_columns(cfg, sel)), len(_EDGE_NAMES)


def boundary_signals(pos: np.ndarray, cfg: GraphConfig) -> np.ndarray:
    """(n, 4 + len(curvature_scales)) geometric features of canonical boundary nodes (pos from
    `canonicalize(xy, n)`, RMS radius 1, so lengths are already / RMS radius): r_norm, curv_s* (curvature x RMS
    radius at each Gaussian scale), turning [rad], convexity_defect, roughness. Also the ContourCNN signal."""
    with np.errstate(invalid="ignore"):  # local_roughness gives sqrt(-1e-17) = NaN on straight runs (true value 0)
        rough = np.nan_to_num(local_roughness(pos, cfg.roughness_window))
    return np.column_stack([np.linalg.norm(pos, axis=1), curvature(pos, cfg.curvature_scales), turning_angle(pos),
                            convexity_defect(pos), rough])


def _normal_hits(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(j, t): for each node i the boundary node nearest the first crossing of its inward normal ray with the
    contour (-1 if none) and the ray length t to that crossing (inf if none). The two segments meeting at node i are
    skipped (they start at the ray origin)."""
    n = len(pos)
    d = inward_normals(pos)
    e = np.roll(pos, -1, axis=0) - pos  # segment k: pos[k] + u e[k], u in [0, 1]
    w = pos[None, :, :] - pos[:, None, :]  # (ray i, segment k) start offsets
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / den
        u = (w[..., 0] * d[:, None, 1] - w[..., 1] * d[:, None, 0]) / den
    ok = (t > 1e-9) & (u >= 0) & (u <= 1)
    i = np.arange(n)
    ok[i, i] = ok[i, i - 1] = False
    t = np.where(ok, t, np.inf)
    k = t.argmin(axis=1)
    tk = t[i, k]
    j = np.where(u[i, k] < 0.5, k, (k + 1) % n)
    return np.where(np.isfinite(tk), j, -1), tk


def _interior_points(pos: np.ndarray, m: int) -> np.ndarray:
    """(<= m, 2) farthest-point samples of a grid inside the polygon, seeded with the boundary (the first one is the
    point deepest inside). Grid spacing from the polygon area (~16 m candidates), so it is scale free."""
    if m <= 0:
        return np.zeros((0, 2))
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    h = np.sqrt(abs(signed_area(pos)) / (16 * m))
    for _ in range(4):  # refine for very thin shapes
        gx, gy = np.meshgrid(np.arange(lo[0] + h / 2, hi[0], h), np.arange(lo[1] + h / 2, hi[1], h))
        cand = np.column_stack([gx.ravel(), gy.ravel()])
        cand = cand[points_in_poly(cand, pos)]
        if len(cand) >= 4 * m:
            break
        h /= 2
    if not len(cand):
        return cand
    d = cKDTree(pos).query(cand)[0]  # distance to the nearest boundary node is enough to seed the sampling
    out = []
    for _ in range(min(m, len(cand))):
        k = int(np.argmax(d))
        out.append(cand[k])
        d = np.minimum(d, np.linalg.norm(cand - cand[k], axis=1))
    return np.array(out)


def _sample(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear grey at pts (..., 2) (x = col, y = row); nearest edge value outside the image."""
    v = map_coordinates(gray, [pts[..., 1].ravel(), pts[..., 0].ravel()], order=1, mode="nearest")
    return v.reshape(pts.shape[:-1])


def contour_graph(contour_px, um_per_px, cfg: GraphConfig, image=None) -> dict[str, torch.Tensor]:
    """FULL graph of one view: {"x" (n, F) float32 (F = len(cfg.node_feature_names())), "edge_index" (2, E) long
    (undirected: pairs first, then the same pairs reversed; no self loops, no duplicate pairs), "edge_attr" (E, 7)
    float32, "edge_kind" (E,) uint8 bit flags, "pos" (n, 2) float32 canonical, "node_type" (n,) uint8 (0 boundary,
    1 interior)}, n = n_nodes + interior nodes.

    contour_px: (N, 2) view contour in image px (x, y). image: the view's image (RGB or grey, any dtype; luminance in
    [0, 1] is sampled) for the tex_* node features (zeros if None); texture depths are fractions of the RMS radius in
    px, inward depths
    clipped to 0.8 x the inward distance to the opposite boundary. um_per_px is accepted for the contract
    signature only: every node / edge feature is similarity invariant (absolute size enters via particle globals).
    Raises ValueError for a zero-area contour.
    """
    n = int(cfg.n_nodes)
    pos, info = canonicalize(contour_px, n)
    names = cfg.node_feature_names()
    hit, hit_t = _normal_hits(pos)
    inner = _interior_points(pos, int(cfg.interior_nodes))
    m = len(inner)
    x = np.zeros((n + m, len(names)), np.float32)
    x[:n, :4 + len(cfg.curvature_scales)] = boundary_signals(pos, cfg)
    x[n:, 0] = np.linalg.norm(inner, axis=1)
    x[n:, names.index("convexity_defect")] = distance_to_boundary(pos, inner)
    x[n:, -1] = 1.0
    if image is not None:
        gray = to_gray(image)  # float32 [0, 1] from RGB or grey of any dtype
        ang, scale = info["angle"], info["scale"]
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        b_px = info["centroid"] + scale * pos @ rot.T  # canonical -> image px
        nrm = inward_normals(b_px)
        depth = np.minimum(np.asarray(cfg.texture_depths_frac, float)[None, :] * scale, 0.8 * hit_t[:, None] * scale)
        t0, nd = names.index("tex_in_0"), len(cfg.texture_depths_frac)
        x[:n, t0:t0 + nd] = boundary_band_profile(gray, b_px, nrm, depth)
        x[:n, t0 + nd] = boundary_band_profile(gray, b_px, -nrm, [cfg.tex_out_frac * scale])[:, 0]
        cross = _sample(gray, b_px[:, None, :] + np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]]))
        x[:n, t0 + nd + 1] = 0.5 * np.hypot(cross[:, 0] - cross[:, 1], cross[:, 2] - cross[:, 3])
        if m:  # local grey mean: centre + 8 neighbours at the first texture depth
            r = max(1.0, cfg.texture_depths_frac[0] * scale)
            a = np.arange(8) * np.pi / 4
            offs = np.vstack([[0.0, 0.0], r * np.column_stack([np.cos(a), np.sin(a)])])
            x[n:, t0] = _sample(gray, (info["centroid"] + scale * inner @ rot.T)[:, None, :] + offs).mean(axis=1)

    # candidate pairs (i, j, kind); interior node r has index n + r
    i = np.arange(n)
    pairs, kinds = [], []
    for hop in cfg.ring_hops:
        if hop % n:
            pairs.append(np.stack([i, (i + hop) % n]))
            kinds.append(np.full(n, EDGE_RING1 if hop == 1 else EDGE_RING_MULTI))
    ok = hit >= 0
    pairs.append(np.stack([i[ok], hit[ok]]))
    kinds.append(np.full(int(ok.sum()), EDGE_CHORD))
    if m:  # each interior node -> its interior_knn nearest boundary nodes and interior_knn nearest interior nodes
        knn = int(cfg.interior_knn)
        targets = [np.argsort(np.linalg.norm(inner[:, None] - pos[None], axis=2), axis=1)[:, :knn]]
        if m > 1:
            di = np.linalg.norm(inner[:, None] - inner[None], axis=2)
            np.fill_diagonal(di, np.inf)
            targets.append(n + np.argsort(di, axis=1)[:, :min(knn, m - 1)])
        for tg in targets:
            pairs.append(np.stack([np.repeat(n + np.arange(m), tg.shape[1]), tg.ravel()]))
            kinds.append(np.full(tg.size, EDGE_INTERIOR))
    p, k = np.concatenate(pairs, axis=1), np.concatenate(kinds).astype(np.uint8)
    lo, hi = p.min(axis=0), p.max(axis=0)
    keep = lo != hi
    uniq, inv = np.unique(lo[keep] * (n + m) + hi[keep], return_inverse=True)
    kind = np.zeros(len(uniq), np.uint8)
    np.bitwise_or.at(kind, inv, k[keep])  # a node pair appears once; its kinds OR-ed
    lo, hi = uniq // (n + m), uniq % (n + m)

    allpos = np.vstack([pos, inner])
    bnd = hi < n  # lo < hi, so both ends are boundary nodes
    hop = hi - lo
    tan = tangents(pos)
    tl, th = tan[np.minimum(lo, n - 1)], tan[np.minimum(hi, n - 1)]
    attr = np.column_stack([
        np.linalg.norm(allpos[lo] - allpos[hi], axis=1),
        np.where(bnd, np.minimum(hop, n - hop) / n, 0.0),
        np.where(bnd, np.abs(np.sum(tl * th, axis=1)), 0.0),
        (kind[:, None] & _BITS.numpy()[None, :]) > 0,
    ]).astype(np.float32)
    return {
        "x": torch.from_numpy(x),
        "edge_index": torch.from_numpy(np.concatenate([np.stack([lo, hi]), np.stack([hi, lo])], axis=1)).long(),
        "edge_attr": torch.from_numpy(np.vstack([attr, attr])),
        "edge_kind": torch.from_numpy(np.concatenate([kind, kind])),
        "pos": torch.from_numpy(allpos.astype(np.float32)),
        "node_type": torch.from_numpy(np.r_[np.zeros(n, np.uint8), np.ones(m, np.uint8)]),
    }


def select_graph(g: dict, cfg: GraphConfig, sel: GraphSelection) -> Data:
    """PyG Data(x, edge_index, edge_attr, pos, node_type long) of the selection: feature columns dropped, interior
    nodes (and their edges) dropped if not sel.interior, edges kept by kind ("ring": EDGE_RING1 pairs only, with only
    their is_ring1 flag set; "none": edge_index (2, 0), edge_attr (0, 7)). Never parallel edges."""
    allowed = _ALLOWED[sel.edges]
    nb = g["x"].shape[0] if sel.interior else int((g["node_type"] == 0).sum())
    ei, kind = g["edge_index"], g["edge_kind"]
    keep = (kind & allowed) > 0
    if not sel.interior:
        keep &= (ei < nb).all(dim=0)
    kind = kind[keep] & allowed
    attr = g["edge_attr"][keep].clone()
    attr[:, _N_CONT:] = ((kind[:, None] & _BITS) > 0).float()
    return Data(x=g["x"][:nb, _node_columns(cfg, sel)], edge_index=ei[:, keep], edge_attr=attr, pos=g["pos"][:nb],
                node_type=g["node_type"][:nb].long())


def pack_graphs(graphs: dict[str, dict]) -> dict:
    """Concatenate full graphs {view_id: contour_graph(...)} into a few flat tensors (fast torch.save, loads with
    weights_only=True). Stores each undirected pair once with its continuous edge features; `unpack_graph` rebuilds
    the full graph exactly. Keys: view_ids, index {view_id: i}, x, pos, node_type, pairs (2, P) int32, pair_attr,
    pair_kind, node_ptr (G + 1,), pair_ptr (G + 1,)."""
    ids = list(graphs)
    gs = [graphs[v] for v in ids]
    half = [g["edge_index"].shape[1] // 2 for g in gs]

    def ptr(counts):
        return torch.tensor(np.r_[0, np.cumsum(counts)], dtype=torch.long)

    return {
        "view_ids": ids, "index": {v: i for i, v in enumerate(ids)},
        "x": torch.cat([g["x"] for g in gs]), "pos": torch.cat([g["pos"] for g in gs]),
        "node_type": torch.cat([g["node_type"] for g in gs]),
        "pairs": torch.cat([g["edge_index"][:, :h] for g, h in zip(gs, half)], dim=1).int(),
        "pair_attr": torch.cat([g["edge_attr"][:h, :_N_CONT] for g, h in zip(gs, half)]),
        "pair_kind": torch.cat([g["edge_kind"][:h] for g, h in zip(gs, half)]),
        "node_ptr": ptr([g["x"].shape[0] for g in gs]), "pair_ptr": ptr(half),
    }


def unpack_graph(packed: dict, view_id: str) -> dict:
    """The full graph dict of `view_id` (same keys, dtypes and edge order as contour_graph); tensors share memory
    with `packed` where possible."""
    i = packed["index"][view_id]
    n0, n1 = packed["node_ptr"][i:i + 2].tolist()
    e0, e1 = packed["pair_ptr"][i:i + 2].tolist()
    pairs, kind = packed["pairs"][:, e0:e1].long(), packed["pair_kind"][e0:e1]
    attr = torch.cat([packed["pair_attr"][e0:e1], ((kind[:, None] & _BITS) > 0).float()], dim=1)
    return {"x": packed["x"][n0:n1], "edge_index": torch.cat([pairs, pairs.flip(0)], dim=1),
            "edge_attr": torch.cat([attr, attr]), "edge_kind": torch.cat([kind, kind]),
            "pos": packed["pos"][n0:n1], "node_type": packed["node_type"][n0:n1]}
