"""[I2] Results report (DESIGN sec. 12): report/report.md, PNG figures in report/figures and booktabs LaTeX tables in
report/tables, built from a results directory with the DESIGN sec. 11 layout.

Reads metrics/*.csv, gates/gates.json, profiling/*, dataset_meta.json and predictions/parts/{level}__*.parquet
directly (never imports experiments.py). Results from synthetic data carry SYNTHETIC_BANNER in report.md, in every
table caption and on every figure, and are never copied into the manuscript. Missing optional inputs are skipped and
logged. Table and figure stems are the ones paper/main.tex includes (\\inputtable / \\inputfigure): every table is a
complete float with \\label{tab:<stem>} that needs only booktabs + graphicx.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from wearpg.config import load_config
from wearpg.gates import PROPOSED
from wearpg.metrics import reliability_curve
from wearpg.schema import MECHANISMS, SYNTHETIC_BANNER, VERIFICATION_NOTICE, WearDataset, prob_columns
from wearpg.utils import get_logger, load_json

log = get_logger()

CLS_METRICS = ("macro_f1", "balanced_accuracy", "accuracy", "ece", "brier", "log_loss")
SEV_METRICS = ("macro_f1", "balanced_accuracy", "ordinal_mae", "qwk")
REG_METRICS = ("mae", "rmse", "r2", "bias", "spearman", "within_condition_spearman")
PER_CLASS_METRICS = ("sensitivity", "specificity", "f1")
LABELS = {"macro_f1": "Macro-F1", "balanced_accuracy": "Bal. acc.", "accuracy": "Acc.", "ece": "ECE",
          "brier": "Brier", "log_loss": "Log loss", "ordinal_mae": "Ordinal MAE", "qwk": "QWK", "mae": "MAE",
          "rmse": "RMSE", "r2": "R2", "bias": "Bias", "spearman": "Spearman",
          "within_condition_spearman": "Within-cond. Spearman", "sensitivity": "Sensitivity",
          "specificity": "Specificity", "f1": "F1", "mae_s": "MAE [s]", "bias_s": "Bias [s]",
          "mae_intervals": "MAE [intervals]", "detection_rate": "Detection", "early_alarm_rate": "Early alarm",
          "false_alarm_rate": "False alarm", "oracle_floor_s": "Oracle floor [s]", "interval_s": "Interval [s]",
          "n_transition_runs": "n transition", "n_stable_runs": "n stable", "mean_log_ratio": "Mean ln ratio",
          "sd_log_ratio": "SD ln ratio", "slope": "Slope", "intercept": "Intercept", "ccc": "CCC"}
# stem -> (report.md section title, plain-text caption; "{ci}" = CI level)
TABLE_SPECS = {
    "dataset_summary": ("Dataset summary", "Dataset summary by material and operating condition."),
    "gates": ("Go/no-go gates", "Outcomes of the go/no-go gates."),
    "classification": ("Particle-level mechanism classification", "Particle-level mechanism classification: "
                       "point estimates with {ci} cluster-bootstrap confidence intervals."),
    "severity": ("Sample-level severity", "Sample-level severity classification ({ci} CIs)."),
    "regression": ("Specific wear rate", "Specific wear rate (log10 k) at sample and run level ({ci} CIs)."),
    "comparisons": ("Paired comparisons", "Paired comparisons: difference with {ci} CI, relative difference and "
                    "multiplicity-adjusted p-value."),
    "per_class": ("Per-class metrics", "Per-class sensitivity, specificity and F1 of the proposed model and the "
                  "strongest baseline (cross-validation, {ci} CIs)."),
    "per_group": ("Metrics by group", "Metrics by material and magnification and for the multi-view and EDS "
                  "subsets (proposed model and strongest baseline)."),
    "transition": ("Transition timing", "Mild-to-severe transition timing."),
    "volume_consistency": ("Volume consistency", "Consistency of the debris-implied run volume with the "
                           "gravimetric volume (natural log)."),
    "profiling": ("Computational cost", "Peak GPU memory and throughput of the particle graph encoder (GNN) and "
                  "the contour CNN."),
}
FIGURES = ("confusion", "reliability", "rate_scatter", "transition_timeline", "volume_consistency", "profiling")
_NUMERIC = re.compile(r"[-<]?\d[\d.]*( \[[^\]]*\])?|-")
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


# ----------------------------------------------------------------------------- formatting


def _cell(v) -> str:
    """Compact text of one value: ints as is, floats with 3 decimals below 10, 1 below 100, else 0; '-' = NaN."""
    if isinstance(v, str):
        return v
    if v is None or isinstance(v, (bool, np.bool_)):
        return "-" if v is None else str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if not np.isfinite(v):
        return "-"
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}"


def _ci(point, lo, hi) -> str:
    return f"{_cell(point)} [{_cell(lo)}, {_cell(hi)}]" if np.isfinite(lo) and np.isfinite(hi) else _cell(point)


def _p(p) -> str:
    return "-" if not np.isfinite(p) else "<0.001" if p < 0.001 else f"{p:.3f}"


def _tex(s) -> str:
    return "".join(_TEX.get(ch, ch) for ch in str(s))


def latex_table(df: pd.DataFrame, caption: str, label: str, synthetic: bool = False) -> str:
    """Complete booktabs ``table`` float (needs booktabs + graphicx only) of `df`: header = column names, one row per
    df row, values formatted compactly and LaTeX-escaped. `caption` is plain text (escaped); with `synthetic` it
    starts with SYNTHETIC_BANNER. `label` is the full label, e.g. "tab:classification". Numeric columns are
    right-aligned, text longer than 50 characters wraps, and tables wider than the text are scaled to \\linewidth."""
    cells = [[_cell(v) for v in row] for row in df.itertuples(index=False)]

    def align(j: int) -> str:
        col = [r[j] for r in cells]
        if col and max(map(len, col)) > 50:
            return r"p{0.5\linewidth}"
        return "r" if col and all(_NUMERIC.fullmatch(c) for c in col) else "l"

    cap = (rf"\textbf{{{SYNTHETIC_BANNER}.}} " if synthetic else "") + _tex(caption)
    return "\n".join([
        "% Generated by wearpg.report from the results directory - do not edit by hand.",
        *([f"% {SYNTHETIC_BANNER}"] if synthetic else []),
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{{cap}}}", rf"\label{{{label}}}", r"\small",
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        rf"\begin{{tabular}}{{{''.join(align(j) for j in range(df.shape[1]))}}}", r"\toprule",
        " & ".join(_tex(c) for c in df.columns) + r" \\", r"\midrule",
        *(" & ".join(_tex(c) for c in r) + r" \\" for r in cells),
        r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def _md(df: pd.DataFrame) -> str:
    rows = [[_cell(v).replace("|", r"\|") for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1],
                      *("| " + " | ".join(r) + " |" for r in rows)])


# ----------------------------------------------------------------------------- inputs


def _csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.info("report: %s not found - skipped", path.as_posix())
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def _json(path: Path) -> dict | None:
    return load_json(path) if path.exists() else None


def _parts(res: Path, level: str, model: str, variant: str, protocol: str = "cv") -> pd.DataFrame | None:
    """Test-split rows of predictions/parts/{level}__{protocol}__{model}__{variant}__*.parquet (all folds/seeds)."""
    files = sorted((res / "predictions" / "parts").glob(f"{level}__{protocol}__{model}__{variant}__*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df.split == "test"]
    return df if len(df) else None


def _dataset(cfg: dict) -> WearDataset | None:
    root = cfg.get("paths", {}).get("data_root")
    if root and Path(root).is_dir():
        return WearDataset.load(root)
    log.info("report: dataset %s not found - dataset summary skipped", root)
    return None


# ----------------------------------------------------------------------------- tables


def _ordered(df: pd.DataFrame, *extra: str) -> pd.DataFrame:
    """Rows by protocol (+ extra keys), then the proposed model, other wearpg variants, baselines (alphabetical)."""
    rank = np.where(df.model == PROPOSED[0], np.where(df.variant == PROPOSED[1], 0, 1), 2)
    keys = ["protocol", *extra, "_rank", "model", "variant"]
    return df.assign(_rank=rank).sort_values(keys, kind="stable").drop(columns="_rank")


def _wide(df: pd.DataFrame, keys: list[str], order, cell) -> pd.DataFrame:
    """One row per unique `keys` (input order) and one text column per metric (`order` first, then any other
    metric), plus n (max over metrics) when present. cell(row) -> text of one long-format row."""
    d = df.assign(_cell=[cell(r) for r in df.itertuples(index=False)])
    w = d.pivot_table(index=keys, columns="metric", values="_cell", aggfunc="first", sort=False)
    w = w[[m for m in order if m in w.columns] + [m for m in w.columns if m not in order]].fillna("-")
    if "n" in d:
        w["n"] = d.groupby(keys, sort=False)["n"].max().map(lambda v: str(int(v)) if np.isfinite(v) else "-")
    return w.reset_index().rename(columns=LABELS)


def _ci_wide(df: pd.DataFrame, keys: list[str], order) -> pd.DataFrame:
    return _wide(df, keys, order, lambda r: _ci(r.point, r.lo, r.hi))


def _headline(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Rows without the synthetic ground-truth diagnostics (metric suffix _gt)."""
    return None if df is None else df[~df.metric.astype(str).str.endswith("_gt")]


def _strongest(df: pd.DataFrame | None, metric: str, higher: bool, level: str | None = None) -> tuple | None:
    """(model, variant) of the best non-wearpg model by the cv point estimate of `metric`, or None."""
    if df is None:
        return None
    d = df[(df.protocol == "cv") & (df.metric == metric) & (df.model != PROPOSED[0])]
    d = d[np.isfinite(d.point.to_numpy(float)) & ((d.level == level) if level else True)]
    if d.empty:
        return None
    r = d.loc[d.point.idxmax() if higher else d.point.idxmin()]
    return r.model, r.variant


def _among(df: pd.DataFrame, models: list[tuple]) -> pd.DataFrame:
    return df[(df.model + "|" + df.variant).isin({f"{m}|{v}" for m, v in models})]


def _dataset_table(ds: WearDataset) -> pd.DataFrame:
    """Counts per (material, condition) + a total row."""
    keys = ["material_id", "condition_id"]
    rt = ds.run_table().assign(severe=lambda d: d.regime == "severe")
    s = ds.samples.merge(rt[["run_id", *keys]], on="run_id")
    p = ds.particles.merge(s[["sample_id", *keys]], on="sample_id")
    p["labelled"] = p.particle_id.isin(set(ds.labels_adjudicated.particle_id))
    t = pd.DataFrame({"load [N]": rt.groupby(keys).load_N.first(), "speed [m/s]": rt.groupby(keys).speed_m_s.first(),
                      "runs": rt.groupby(keys).size(), "severe runs": rt.groupby(keys).severe.sum(),
                      "samples": s.groupby(keys).size(), "particles detected": s.groupby(keys).n_particles_total.sum(),
                      "particles stored": p.groupby(keys).size(), "labelled": p.groupby(keys).labelled.sum()})
    counts = list(t.columns[2:])
    t[counts] = t[counts].fillna(0).astype(np.int64)
    t = t.reset_index()
    return pd.concat([t, pd.DataFrame([{"material_id": "all", "condition_id": "", **t[counts].sum()}])],
                     ignore_index=True)


def _comparisons_table(c: pd.DataFrame) -> pd.DataFrame:
    """Claim-metric rows (those with a p-value; every row if none has one), sorted by protocol/family/level. The
    method (constant text) goes into the caption."""
    c = c[c.p_value.notna()] if c.p_value.notna().any() else c
    c = c.sort_values(["protocol", "family", "level", "metric"], kind="stable")
    return pd.DataFrame({
        "protocol": c.protocol, "level": c.level, "family": c.family, "metric": c.metric,
        "A": c.model_a + "/" + c.variant_a, "B": c.model_b + "/" + c.variant_b,
        "diff [CI]": [_ci(d, lo, hi) for d, lo, hi in zip(c["diff"], c.lo, c.hi)], "rel. diff": c.rel_diff,
        "p": [_p(v) for v in c.p_value], "p adj.": [_p(v) for v in c.p_adj]})


def _profiling_table(gnn: pd.DataFrame | None, cnn: pd.DataFrame | None) -> pd.DataFrame | None:
    names = {"peak_mem_mb": "peak MiB", "fwd_bwd_ms": "ms", "graphs_per_s": "graphs/s",
             "particles_per_s": "particles/s", "status": "status"}
    parts = [df.rename(columns={"batch_size": "graphs_per_batch"}).set_index(["n_nodes", "graphs_per_batch"])
             .rename(columns=names).add_prefix(f"{tag} ") for df, tag in ((gnn, "GNN"), (cnn, "CNN")) if df is not None]
    if not parts:
        return None
    return pd.concat(parts, axis=1).reset_index().rename(columns={"n_nodes": "nodes", "graphs_per_batch": "batch"})


# ----------------------------------------------------------------------------- figures


def _figure(ncols: int, nrows: int, synthetic: bool, w: float = 4.2, h: float = 3.8):
    """Figure + flat axes array; pyplot-free (Agg canvas), so no global backend state is touched."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(w * ncols, h * nrows + (0.3 if synthetic else 0)), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    if synthetic:
        fig.suptitle(SYNTHETIC_BANNER, color="red", fontsize=8, fontweight="bold", wrap=True)
    return fig, axes


def _fig_confusion(panels, path: Path, synthetic: bool) -> None:
    """Row-normalised confusion matrices (rows = adjudicated label), counts annotated."""
    k = len(MECHANISMS)
    fig, axes = _figure(len(panels), 1, synthetic, 4.6, 4.4)
    for ax, (title, d) in zip(axes, panels):
        cm = np.zeros((k, k))
        np.add.at(cm, (d.y_true.to_numpy(int), d.y_pred.to_numpy(int)), 1)
        frac = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(frac, vmin=0, vmax=1, cmap="Blues")
        for i, j in zip(*np.nonzero(cm)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if frac[i, j] > 0.5 else "black")
        ax.set_xticks(range(k), MECHANISMS, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(k), MECHANISMS, fontsize=7)
        ax.set(xlabel="predicted", ylabel="adjudicated label", title=title)
    fig.savefig(path, dpi=300)


def _fig_reliability(panels, path: Path, synthetic: bool, n_bins: int) -> None:
    fig, (ax,) = _figure(1, 1, synthetic)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect calibration")
    for title, d in panels:
        rc = reliability_curve(d.y_true.to_numpy(int), d[prob_columns()].to_numpy(float), n_bins)
        ok = np.isfinite(rc["accuracy"])
        ax.plot(rc["confidence"][ok], rc["accuracy"][ok], marker="o", ms=3, label=title)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="confidence (max probability)", ylabel="accuracy")
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=300)


def _fig_scatter(panels, path: Path, synthetic: bool, x: str, y: str, xlabel: str, ylabel: str) -> bool:
    """Predicted vs reference scatter with the identity line, one panel per model; False if nothing to plot."""
    panels = [(t, d[x].to_numpy(float), d[y].to_numpy(float)) for t, d in panels]
    panels = [(t, xv[ok], yv[ok]) for t, xv, yv in panels if (ok := np.isfinite(xv) & np.isfinite(yv)).any()]
    if not panels:
        return False
    fig, axes = _figure(len(panels), 1, synthetic)
    for ax, (title, xv, yv) in zip(axes, panels):
        lim = [min(xv.min(), yv.min()), max(xv.max(), yv.max())]
        ax.plot(lim, lim, ls="--", color="grey", lw=1)
        ax.scatter(xv, yv, s=12, alpha=0.7)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f"{title} (n={len(xv)})")
    fig.savefig(path, dpi=300)
    return True


def _fig_transition(samples: pd.DataFrame, runs: pd.DataFrame, path: Path, synthetic: bool, threshold: float,
                    max_runs: int = 12) -> None:
    """P(severity >= 1) per sampling interval of up to `max_runs` runs (transition runs first; first seed), with
    the observed (red) and predicted (dashed) transition times and the alarm threshold."""
    s = samples[samples.seed == samples.seed.min()]
    r = runs[runs.seed == runs.seed.min()].drop_duplicates("run_id").set_index("run_id")
    t_true, t_pred = r.transition_time_true, r.transition_time_pred
    ids = sorted(s.run_id.unique(), key=lambda i: (not np.isfinite(t_true.get(i, np.nan)), i))[:max_runs]
    ncols = min(4, len(ids))
    fig, axes = _figure(ncols, -(-len(ids) // ncols), synthetic, 3.0, 2.4)
    for ax, rid in zip(axes, ids):
        g = s[s.run_id == rid].sort_values("time_s")
        ax.hlines(g.sev_p_transition + g.sev_p_severe, g.t_start_s, g.time_s, lw=2)  # one segment per interval
        ax.axhline(threshold, ls=":", color="grey", lw=1, label="alarm threshold")
        if np.isfinite(t_true.get(rid, np.nan)):
            ax.axvline(t_true[rid], color="red", lw=1.2, label="observed transition")
        if np.isfinite(t_pred.get(rid, np.nan)):
            ax.axvline(t_pred[rid], color="black", ls="--", lw=1, label="predicted transition")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(rid, fontsize=7)
        ax.set_xlabel("t [s]", fontsize=7)
        ax.set_ylabel("P(severity >= 1)", fontsize=7)
    axes[0].legend(fontsize=6)
    for ax in axes[len(ids):]:
        ax.set_visible(False)
    fig.savefig(path, dpi=300)


def _fig_profiling(gnn, cnn, budget_mb, path: Path, synthetic: bool) -> None:
    fig, (a1, a2) = _figure(2, 1, synthetic)
    for df, tag, batch, rate, ls in ((gnn, "GNN", "graphs_per_batch", "graphs_per_s", "-"),
                                     (cnn, "CNN", "batch_size", "particles_per_s", "--")):
        if df is None:
            continue
        for b, g in df[df.status == "ok"].groupby(batch):
            a1.plot(g.n_nodes, g.peak_mem_mb, ls=ls, marker="o", ms=3, label=f"{tag}, batch {b}")
            a2.plot(g.n_nodes, g[rate], ls=ls, marker="o", ms=3, label=f"{tag}, batch {b}")
    if budget_mb:
        a1.axhline(budget_mb, color="red", ls=":", lw=1, label="VRAM budget")
    a1.set(xscale="log", yscale="log", xlabel="nodes per graph", ylabel="peak memory [MiB]")
    a2.set(xscale="log", yscale="log", xlabel="nodes per graph", ylabel="throughput [graphs or particles / s]")
    a1.legend(fontsize=6)
    fig.savefig(path, dpi=300)


# ----------------------------------------------------------------------------- report


def build_report(results_dir, cfg: dict | None = None, paper_dir=None) -> Path:
    """Write <results_dir>/report/{report.md, tables/<stem>.tex, figures/<stem>.png}; returns the report.md path.

    cfg defaults to <results_dir>/config.yaml (it gives the dataset root for the dataset summary, eval.ci,
    eval.ece_bins, eval.transition_threshold). The data are synthetic when the dataset, dataset_meta.json or
    gates.json says so, and also when none of them is available. With `paper_dir`, the tables and figures written
    are copied to paper_dir/tables and paper_dir/figures - never for synthetic data.
    """
    res = Path(results_dir)
    if not res.is_dir():
        raise FileNotFoundError(f"results directory not found: {res}")
    if cfg is None and (res / "config.yaml").exists():
        cfg = load_config(res / "config.yaml")
    cfg = cfg or {}
    ev = cfg.get("eval", {})
    ci = f"{100 * ev.get('ci', 0.95):g} %"
    out, tdir, fdir = res / "report", res / "report" / "tables", res / "report" / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(exist_ok=True)
    for stem in TABLE_SPECS:  # no stale outputs from an earlier report
        (tdir / f"{stem}.tex").unlink(missing_ok=True)
    for stem in FIGURES:
        (fdir / f"{stem}.png").unlink(missing_ok=True)

    meta = _json(res / "dataset_meta.json") or {}
    gates = _json(res / "gates" / "gates.json")
    ds = _dataset(cfg)
    known = ([ds.is_synthetic] if ds is not None else []) + ([meta["is_synthetic"]] if "is_synthetic" in meta else [])
    known += [gates["synthetic"]] if gates else []
    synthetic = any(map(bool, known)) if known else True
    if not known:
        log.warning("report: cannot tell whether the data are synthetic - treated as synthetic")

    m = {k: _csv(res / "metrics" / f"{k}.csv") for k in ("classification", "per_class", "severity", "regression",
                                                         "comparisons", "per_group", "transition", "volume_consistency")}
    gnn, cnn = _csv(res / "profiling" / "gnn.csv"), _csv(res / "profiling" / "contour_cnn.csv")
    prof = _json(res / "profiling" / "summary.json") or {}
    cls, sev, reg = _headline(m["classification"]), _headline(m["severity"]), _headline(m["regression"])
    best_cls = _strongest(cls, "macro_f1", higher=True)
    best_run = _strongest(reg, "mae", higher=False, level="run")
    two = [PROPOSED] + ([best_cls] if best_cls else [])
    status = {True: "PASS", False: "FAIL", None: "NOT EVALUABLE"}
    tables = {
        "dataset_summary": None if ds is None else _dataset_table(ds),
        "gates": None if gates is None else pd.DataFrame(
            [{"gate": g["name"], "status": status[g["passed"]], "message": g["message"]} for g in gates["gates"]]),
        "classification": None if cls is None else _ci_wide(_ordered(cls), ["protocol", "model", "variant"],
                                                            CLS_METRICS),
        "severity": None if sev is None else _ci_wide(_ordered(sev), ["protocol", "model", "variant"], SEV_METRICS),
        "regression": None if reg is None else _ci_wide(_ordered(reg, "level"),
                                                        ["protocol", "level", "model", "variant"], REG_METRICS),
        "comparisons": None if m["comparisons"] is None else _comparisons_table(m["comparisons"]),
        "per_class": None if m["per_class"] is None else _ci_wide(
            _ordered(_among(m["per_class"][m["per_class"].protocol == "cv"], two)), ["model", "variant", "class"],
            PER_CLASS_METRICS),
        "per_group": None if m["per_group"] is None else _wide(
            _ordered(_among(m["per_group"], two)), ["protocol", "model", "variant", "group_col", "group"],
            CLS_METRICS, lambda r: _cell(r.value)),
        "transition": None if m["transition"] is None else _ordered(m["transition"]).rename(columns=LABELS),
        "volume_consistency": None if m["volume_consistency"] is None else _ordered(
            m["volume_consistency"]).rename(columns=LABELS),
        "profiling": _profiling_table(gnn, cnn),
    }

    md = ["# wearpg results report", ""]
    if synthetic:
        md += [f"**{SYNTHETIC_BANNER}**", "", f"**{VERIFICATION_NOTICE}**", ""]
    md += [f"Results directory: `{res.as_posix()}`", ""]
    scalars = {k: v for k, v in {**meta, **meta.get("describe", {})}.items() if not isinstance(v, (dict, list))}
    if scalars:
        md += ["## Dataset", "", _md(pd.DataFrame({"key": list(scalars), "value": list(map(str, scalars.values()))})),
               ""]
    captions = {k: c.format(ci=ci) for k, (_, c) in TABLE_SPECS.items()}
    if m["comparisons"] is not None:
        captions["comparisons"] += f" Method: {'; '.join(sorted(set(m['comparisons'].method.astype(str))))}."
    written, skipped = [], []
    for stem, df in tables.items():
        title, caption = TABLE_SPECS[stem][0], captions[stem]
        if df is None or df.empty:
            skipped.append(f"table {stem}")
            continue
        (tdir / f"{stem}.tex").write_text(latex_table(df, caption, f"tab:{stem}", synthetic),
                                          encoding="utf-8")
        written.append(tdir / f"{stem}.tex")
        md += [f"## {title}", ""]
        if stem == "gates":
            md += [f"decision_valid: {str(gates.get('decision_valid')).lower()}; overall: {status[gates.get('passed')]}"
                   " (details: ../gates/gates.md)", ""]
        if stem == "comparisons":
            md += [f"{caption} Claim metrics with p-values; every metric's CI is in metrics/comparisons.csv.", ""]
        if stem in ("per_class", "per_group"):
            md += [f"Models: {', '.join('/'.join(t) for t in two)}.", ""]
        if stem == "profiling" and prof:
            md += [f"Device {prof.get('device_name')}, VRAM budget {_cell(prof.get('vram_budget_mb'))} MiB, "
                   f"recommended max nodes per graphs-per-batch: {prof.get('recommended_max_nodes')}.", ""]
        md += [_md(df), ""]
    gt = [d[d.metric.astype(str).str.endswith("_gt")].assign(table=k) for k, d in m.items()
          if d is not None and "metric" in d and "point" in d]
    gt = pd.concat(gt, ignore_index=True) if gt else pd.DataFrame()
    exc = _csv(res / "metrics" / "exclusions.csv")
    if exc is not None:
        exc = exc[exc.n_runs > 0]
        md += ["## Excluded runs (counted, not scored)", "", _md(exc) if len(exc) else "No run was excluded.", ""]
    if len(gt):
        md += ["## Synthetic ground-truth diagnostics (never headline results)", "",
               _md(gt.reindex(columns=["table", "protocol", "level", "model", "variant", "metric", "point", "lo",
                                       "hi", "n"])), ""]

    # figures
    def panels(level: str, models) -> list:
        found = [(f"{mo}/{va}", _parts(res, level, mo, va)) for mo, va in models]
        return [(t, d) for t, d in found if d is not None]

    figs = {}
    lab = [(t, d[(d.y_true >= 0) & (d.y_pred >= 0)]) for t, d in panels("particles", two)]
    lab = [(t, d) for t, d in lab if len(d)]
    if lab:
        _fig_confusion(lab, fdir / "confusion.png", synthetic)
        _fig_reliability(lab, fdir / "reliability.png", synthetic, int(ev.get("ece_bins", 15)))
        figs.update(confusion=True, reliability=True)
    runs_two = panels("runs", [PROPOSED] + ([best_run] if best_run else []))
    figs["rate_scatter"] = _fig_scatter(runs_two, fdir / "rate_scatter.png", synthetic, "log10_rate_true",
                                        "log10_rate_pred", "measured log10 k [mm^3/(N m)]", "predicted log10 k")
    s_prop, r_prop = _parts(res, "samples", *PROPOSED), _parts(res, "runs", *PROPOSED)
    if s_prop is not None and r_prop is not None:
        _fig_transition(s_prop, r_prop, fdir / "transition_timeline.png", synthetic,
                        float(ev.get("transition_threshold", 0.5)))
        figs["transition_timeline"] = True
    vc = m["volume_consistency"]
    vc_models = [PROPOSED] + ([] if vc is None else sorted(
        {(a, b) for a, b in zip(vc.model, vc.variant) if a != PROPOSED[0] and b == "default"}))
    figs["volume_consistency"] = _fig_scatter(
        panels("runs", vc_models), fdir / "volume_consistency.png", synthetic, "log_grav_volume_mm3",
        "log_debris_volume_mm3", "ln gravimetric volume [mm^3]", "ln debris-implied volume [mm^3]")
    if gnn is not None or cnn is not None:
        _fig_profiling(gnn, cnn, prof.get("vram_budget_mb"), fdir / "profiling.png", synthetic)
        figs["profiling"] = True
    fig_paths = [fdir / f"{s}.png" for s in FIGURES if figs.get(s)]
    skipped += [f"figure {s}" for s in FIGURES if not figs.get(s)]
    if fig_paths:
        md += ["## Figures", "", *(f"![{p.stem}](figures/{p.name})" for p in fig_paths), ""]
    if skipped:
        log.info("report: skipped (missing inputs): %s", ", ".join(skipped))
        md += ["## Skipped (missing inputs)", "", *(f"- {s}" for s in skipped), ""]

    if paper_dir is not None:
        if synthetic:
            log.warning("report: synthetic results are never copied into %s", paper_dir)
            md += [f"Not copied to `{Path(paper_dir).as_posix()}`: synthetic results never enter the manuscript.", ""]
        else:
            for p, sub in [(p, "tables") for p in written] + [(p, "figures") for p in fig_paths]:
                (Path(paper_dir) / sub).mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, Path(paper_dir) / sub / p.name)
            log.info("report: copied %d tables and %d figures to %s", len(written), len(fig_paths), paper_dir)
    path = out / "report.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path
