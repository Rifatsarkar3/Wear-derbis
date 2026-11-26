"""Draft PNG figures (matplotlib Agg) for the open-data image analyses, read from outputs/open_data/ferro/ (the final paper
figures are made later from the same CSV / JSON files). Run after open_ferro_eval.py, open_ferro_morph.py and
open_ferro_robust.py:   .venv/Scripts/python.exe -I scripts/open_ferro_figs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wearpg.features.contour import canonicalize  # noqa: E402
from wearpg.graphs import EDGE_CHORD, EDGE_INTERIOR, EDGE_RING1, GraphConfig, contour_graph  # noqa: E402
from wearpg.open_data import load_zenodo_ferro, mask_components, read_mask  # noqa: E402
from wearpg.segmentation import load_image, segment_image  # noqa: E402
from wearpg.utils import load_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZEN = ROOT / "data" / "real" / "zenodo_ferro" / "extracted"
OUT = ROOT / "outputs" / "open_data" / "ferro"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NOTE = "Open real data: Zenodo Ferrographic Image Dataset (CC-BY-4.0, doi 10.5281/zenodo.22997504); px uncalibrated"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
METHODS = ["ours_default", "ours_tuned", "otsu", "local_default", "local_tuned"]


def _save(fig, name: str, note: str = NOTE) -> None:
    fig.text(0.01, -0.02, note, fontsize=6.5, color=INK2, ha="left", va="top")  # below rotated tick labels
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_overlay(idx: pd.DataFrame) -> None:
    """Three testO ferrograms (10th / 50th / 90th percentile of ours_tuned IoU): GT vs predicted contours."""
    per = pd.read_csv(OUT / "seg_per_image_test.csv")
    t = per[per.method == "ours_tuned"].sort_values("iou").reset_index(drop=True)
    picks = [t.iloc[int(round(q * (len(t) - 1)))] for q in (0.1, 0.5, 0.9)]
    cfg = {k: v for k, v in load_json(OUT / "seg_tuned_params.json")["best"]["ours"].items()
           if not k.startswith("train_")}
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    for a, r in zip(ax, picks):
        row = idx.set_index("image_id").loc[r.image_id]
        img = load_image(row.image_path)
        a.imshow(img)
        for c in mask_components(read_mask(row.mask_path), 30):
            a.plot(*np.vstack([c["contour"], c["contour"][:1]]).T, color=SLOTS[2], lw=1.0)
        for s in segment_image(img, cfg):
            a.plot(*np.vstack([s["contour"], s["contour"][:1]]).T, color=SLOTS[1], lw=0.8, ls="--")
        a.set_title(f"{r.image_id}: IoU {r.iou:.2f}, F1 {r.f1:.2f}", color=INK, fontsize=9)
        a.set_axis_off()
    ax[0].plot([], [], color=SLOTS[2], label="GT mask components")
    ax[0].plot([], [], color=SLOTS[1], ls="--", label="ours (trainO-tuned)")
    ax[0].legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, fontsize=8)
    _save(fig, "fig_seg_overlay.png")


def fig_seg_metrics() -> None:
    s = load_json(OUT / "seg_summary_test.json")
    mets = [("iou", "pixel IoU"), ("dice", "pixel Dice"), ("f1", "object F1 (IoU>=0.5)"),
            ("abs_area_rel_err", "|rel. error| total area"), ("abs_count_rel_err", "|rel. error| count"),
            ("ccc_log_area", "CCC log area (across images)")]
    fig, ax = plt.subplots(1, len(mets), figsize=(15, 3.4))
    x = np.arange(len(METHODS))
    for a, (m, lab) in zip(ax, mets):
        pt = np.array([s[k][m]["point"] for k in METHODS])
        lo, hi = np.array([s[k][m]["lo"] for k in METHODS]), np.array([s[k][m]["hi"] for k in METHODS])
        a.bar(x, pt, color=SLOTS[:len(METHODS)], width=0.7, edgecolor="white", linewidth=1)
        a.errorbar(x, pt, yerr=[pt - lo, hi - pt], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        a.set_xticks(x, METHODS, rotation=45, ha="right")
        a.set_title(lab, fontsize=9, color=INK)
        a.grid(axis="x", visible=False)
    _save(fig, "fig_seg_metrics.png", NOTE + f"; testO n = {s['n_images']} images, 95% image-bootstrap CIs")


def fig_morph() -> None:
    df = pd.read_csv(OUT / "particle_descriptors.csv.gz")
    sets = [("zenodo_gt", "Zenodo GT masks"), ("zenodo_seg", "Zenodo ours-segmented"),
            ("tcbgy_seg", "TCBGY crops (ours-segmented)"), ("synthetic_raster", "synthetic smoke (SIMULATION)")]
    feats = [("shape_area_um2", "area [px] (log scale)", True), ("shape_aspect_ratio", "aspect ratio", False),
             ("shape_roundness", "roundness 4piA/P^2", False), ("shape_solidity", "solidity", False),
             ("fractal_divider_D", "divider D (fractal_valid only)", False)]
    fig, ax = plt.subplots(1, len(feats) + 1, figsize=(17, 3.4))
    for a, (f, lab, logx) in zip(ax, feats):
        for k, (sname, slab) in enumerate(sets):
            d = df[df.set == sname]
            v = np.sort((d[d.fractal_valid > 0.5] if f == "fractal_divider_D" else d)[f].to_numpy())
            if len(v):
                a.plot(v, np.arange(1, len(v) + 1) / len(v), color=SLOTS[k], lw=2, label=f"{slab} (n={len(v)})")
        a.set_xscale("log" if logx else "linear")
        a.set_xlabel(lab)
        a.set_ylabel("ECDF" if f == feats[0][0] else "")
    frac = [df[df.set == s].fractal_valid.mean() for s, _ in sets]
    ax[-1].barh(np.arange(len(sets)), frac, color=SLOTS[:len(sets)], edgecolor="white")
    ax[-1].set_yticks(np.arange(len(sets)), [lab for _, lab in sets], fontsize=7)
    ax[-1].set_xlabel("fraction fractal_valid (2-px floor)")
    ax[-1].grid(axis="y", visible=False)
    ax[0].legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=4, fontsize=8)
    _save(fig, "fig_morph_real_vs_synthetic.png", NOTE + "; synthetic = simulation (wearpg.synthetic smoke set)")


def fig_robust() -> None:
    st = pd.read_csv(OUT / "robust_stability.csv")
    med = st.groupby(["mode", "transform", "group"]).spearman.median().reset_index()
    groups = ["shape", "efd", "fractal", "graph_geom", "graph_tex"]
    fams = [("rotation", [("rot45", 45), ("rot90", 90), ("rot_random", 135)]),
            ("rescale (magnification)", [("scale0.5", 0.5), ("scale0.25", 0.25)]),
            ("Gaussian blur [px]", [("blur1", 1), ("blur2", 2)]), ("noise SD", [("noise0.02", 0.02),
                                                                          ("noise0.05", 0.05)])]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.2), sharey=True)
    for i, mode in enumerate(["mask", "pipeline"]):
        for j, (fam, ts) in enumerate(fams):
            a = ax[i, j]
            for k, g in enumerate(groups):
                xs, ys = [], []
                ident = {"rotation": None, "rescale (magnification)": 1.0}.get(fam, 0.0)
                if ident is not None:
                    xs, ys = [ident], [1.0]
                for t, xv in ts:
                    v = med[(med["mode"] == mode) & (med["transform"] == t) & (med.group == g)].spearman
                    if len(v):
                        xs.append(xv)
                        ys.append(float(v.iloc[0]))
                if len(xs) > 1 or (fam == "rotation" and xs):
                    a.plot(xs, ys, marker="o", ms=5, lw=2 if fam != "rotation" else 0, color=SLOTS[k],
                           label=g)
            if not a.lines:
                a.text(0.5, 0.5, "n/a: noise leaves the mask unchanged", transform=a.transAxes, ha="center",
                       color=INK2)
            if fam == "rotation":
                a.set_xticks([45, 90, 135], ["45", "90", "random"])
            if fam.startswith("rescale"):
                a.invert_xaxis()
            a.set_title(f"{mode}: {fam}", fontsize=9, color=INK)
            a.set_ylabel("median Spearman vs reference" if j == 0 else "")
    ax[0, 0].legend(loc="lower left", fontsize=8)
    _save(fig, "fig_robustness.png", NOTE + "; mask = contour from transformed GT mask, pipeline = ours re-segmented")


def fig_domain_gap() -> None:
    d = pd.read_csv(OUT / "domain_gap_auc.csv")
    pairs = d[["a", "b"]].drop_duplicates().apply(tuple, axis=1).tolist()
    groups = d.group.unique().tolist()
    fig, a = plt.subplots(figsize=(11, 3.6))
    w = 0.8 / len(pairs)
    for k, (pa, pb) in enumerate(pairs):
        s = d[(d.a == pa) & (d.b == pb)].set_index("group").loc[groups]
        x = np.arange(len(groups)) + (k - (len(pairs) - 1) / 2) * w
        a.bar(x, s.auc, width=w, color=SLOTS[k], edgecolor="white", label=f"{pa} vs {pb}")
        a.errorbar(x, s.auc, yerr=[s.auc - s.lo, s.hi - s.auc], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
    a.axhline(0.5, color=INK2, lw=1, ls=":")
    a.set_xticks(np.arange(len(groups)), groups)
    a.set_ylim(0.4, 1.02)
    a.set_ylabel("cross-validated AUC (real vs other)")
    a.grid(axis="x", visible=False)
    a.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    _save(fig, "fig_domain_gap_auc.png", NOTE + "; AUC 0.5 = indistinguishable; synthetic = simulation")


def fig_graph_montage(idx: pd.DataFrame, n: int = 12) -> None:
    """Real GT particles with their contour graphs: hop-1 ring edges, width chords, interior nodes and edges."""
    df = pd.read_csv(OUT / "particle_descriptors.csv.gz", usecols=["set", "image_id", "particle_id",
                                                                   "shape_area_um2"])
    cand = df[(df.set == "zenodo_gt") & (df.shape_area_um2 >= 600)].sample(n, random_state=0)
    cfg = GraphConfig()
    fig, ax = plt.subplots(2, n // 2, figsize=(2.4 * n / 2, 5.2))
    for a, r in zip(ax.ravel(), cand.itertuples()):
        row = idx.set_index("image_id").loc[r.image_id]
        img = load_image(row.image_path)
        xy = mask_components(read_mask(row.mask_path), 30)[int(r.particle_id.rsplit("_p", 1)[1])]["contour"]
        g = contour_graph(xy, 1.0, cfg, img)
        _, info = canonicalize(xy, cfg.n_nodes)
        c, s = np.cos(info["angle"]), np.sin(info["angle"])
        p = info["centroid"] + info["scale"] * g["pos"].numpy() @ np.array([[c, -s], [s, c]]).T
        lo, hi = xy.min(0) - 6, xy.max(0) + 6
        a.imshow(img, extent=(-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
        ei, kind = g["edge_index"].numpy(), g["edge_kind"].numpy()
        half = ei.shape[1] // 2
        for (i, j), k in zip(ei[:, :half].T, kind[:half]):
            if k & EDGE_INTERIOR:
                a.plot(*p[[i, j]].T, color=SLOTS[3], lw=0.5, alpha=0.8)
            elif k & EDGE_CHORD:
                a.plot(*p[[i, j]].T, color=SLOTS[0], lw=0.6, alpha=0.9)
            elif k & EDGE_RING1:
                a.plot(*p[[i, j]].T, color="white", lw=1.0)
        inner = g["node_type"].numpy() == 1
        a.scatter(*p[inner].T, s=9, color=SLOTS[3], edgecolors="white", linewidths=0.4, zorder=3)
        a.set_xlim(lo[0], hi[0])
        a.set_ylim(hi[1], lo[1])
        a.set_title(r.particle_id, fontsize=6.5, color=INK2)
        a.set_axis_off()
    ax[0, 0].plot([], [], color="white", label="ring (hop 1)")
    ax[0, 0].plot([], [], color=SLOTS[0], label="width chord")
    ax[0, 0].plot([], [], color=SLOTS[3], marker="o", label="interior node / edge")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower right", ncol=3, fontsize=8,
               facecolor="#9a9a9a", frameon=True)
    _save(fig, "fig_graph_montage.png", NOTE + f"; graphs.contour_graph n_nodes={cfg.n_nodes}, multi-hop ring "
                                              "edges not drawn")


def main() -> None:
    idx = load_zenodo_ferro(ZEN)
    for f, args in ((fig_seg_metrics, ()), (fig_overlay, (idx,)), (fig_morph, ()), (fig_robust, ()),
                    (fig_domain_gap, ()), (fig_graph_montage, (idx,))):
        try:
            f(*args)
            print("ok", f.__name__)
        except FileNotFoundError as e:  # analysis not run yet
            print("skip", f.__name__, e)


if __name__ == "__main__":
    main()
