"""Publication figures for the Wear manuscript, made ONLY from stored results (no analysis is re-run, no number changed).

Each figure has one function `fig_<stem>()`; it reads the stored CSV / JSON files, re-derives any plotted quantity
that is not stored directly from the stored per-item rows (as the analysis docs define it) and asserts that it matches
the stored summary (`check`). Output: paper/figures/<stem>.pdf (vector) + <stem>.png (opaque RGB, 1000 dpi; 600 dpi for
the figures that contain photographs). Figures whose input files do not exist yet are skipped with "WAITING". Every
check and the size, smallest font and resolution of every figure are written to outputs/agent_logs/FIG_checks.log.

    .venv/Scripts/python.exe -I scripts/make_paper_figures.py              # all figures
    .venv/Scripts/python.exe -I scripts/make_paper_figures.py nasa_onset   # selected stems

Style (of the author's published RESS paper, its make_figures_R3.py; `save` asserts width, height and font size):
- every figure 7.2 in wide (two-column text block: 7.26 in): Arial, 7.5 pt tick labels, 8 pt axis labels, 7 pt legends,
  nothing below 6.5 pt; mathematics in STIX Sans;
- every plot in a closed frame with inward ticks on all four sides and minor ticks on continuous axes only, framed
  legends, panel labels (a), (b), ... outside the axes, plain block diagrams with black outlines, no titles;
- colours from ColorBrewer "Paired" (dict C), one meaning each in every figure, always with a second cue (marker, hatch,
  line style or text). Simulation figures say "Simulation" on the figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.image  # noqa: E402
import matplotlib.colors  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Patch, Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "paper" / "figures"
OIL = ROOT / "outputs" / "open_data" / "oil"
FER = ROOT / "outputs" / "open_data" / "ferro"
DIAG = ROOT / "outputs" / "diagnostics"
REND = ROOT / "outputs" / "renderer"
SIM = ROOT / "outputs" / "simulation"
PILOT = ROOT / "outputs" / "pilot"
ZEN = ROOT / "data" / "real" / "zenodo_ferro" / "extracted"
CHECK_LOG = ROOT / "outputs" / "agent_logs" / "FIG_checks.log"

W, H_MAX = 7.2, 6.0                       # in; width of every figure, largest height
DPI_LINE, DPI_PHOTO = 1000, 600           # PNG resolution: line art / figures that contain photographs
PHOTO = {"ferro_overlays", "leakage_audit", "contour_graphs_real", "renderer_v1_v2"}
FS_LETTER, FS_LAB, FS_TICK, FS_NOTE, FS_SMALL, FS_MIN = 9, 8, 7.5, 7.5, 7, 6.5

C = {  # ColorBrewer "Paired" (dark/light pairs), one meaning each in every figure
    "ours": "#1F78B4",        # debris measurement as pre-specified (segmenter, mass, composition, physics)
    "ours_ext": "#A6CEE3",    # its post-hoc, exploratory or fused variant
    "base": "#808080",        # comparison: baseline method or information, second real data set, training twin
    "base_dark": "#404040",   # second dark comparison next to "base"
    "neutral": "#D9D9D9",     # neutral band: range of thresholds, annotation ceiling, interval, pilot setting
    "true": "#000000",        # ground truth (on photographs: white in a black casing), thresholds, reference lines
    "severe": "#E31A1C",      # severe wear state: destructive pitting, wear-out
    "early": "#FF7F00",       # early wear state: initial pitting, infant mortality
    "healthy": "#33A02C",     # normal wear state: no damage, useful life
    "graph": "#6A3D9A",       # particle graphs: graph geometry, width chords, learned particle-graph model
    "interior": "#CAB2D6",    # interior nodes and edges of the particle graphs (light partner of "graph")
    "synth": "#B15928",       # simulated particles and fields
    "dup": "#FB9A99",         # near-duplicate test images and their masks
    "link": "#9A9A9A",        # line that joins paired values
}
DASH, DOT, DASHDOT = (0, (4, 2)), (0, (1, 1.6)), (0, (5, 1.5, 1, 1.5))
EB = dict(capsize=2.2, elinewidth=0.8, mec="black", mew=0.4)          # error bars
EDGE = dict(edgecolors="black", linewidths=0.4)                       # scatter markers
PHASE = {"infant_mortality": ("infant mortality", C["early"], "^"), "useful_life": ("useful life", C["healthy"], "s"),
         "wear_out": ("wear-out", C["severe"], "o")}
CHECKS: list[str] = []
STYLE_LOG: list[str] = []


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "font.size": FS_NOTE, "axes.labelsize": FS_LAB, "axes.titlesize": FS_NOTE,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_SMALL,
        "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.linewidth": 0.7,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.55, "ytick.minor.width": 0.55,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.pad": 3.0, "ytick.major.pad": 3.0,
        "axes.labelpad": 3.0, "axes.titlepad": 4.0, "lines.linewidth": 1.0, "hatch.linewidth": 0.6,
        "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black", "legend.framealpha": 1.0,
        "legend.borderpad": 0.45, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.5,
        "legend.labelspacing": 0.35, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem: str) -> None:
    """Assert the style contract (7.2 in wide, at most 6 in high, no text below 6.5 pt), then write the vector PDF and
    an opaque RGB PNG with the resolution recorded in the file."""
    w, h = fig.get_size_inches()
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    assert abs(w - W) < 1e-9 and h <= H_MAX and min(sizes) >= FS_MIN, (stem, w, h, min(sizes))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf")
    dpi, png = DPI_PHOTO if stem in PHOTO else DPI_LINE, FIG / f"{stem}.png"
    fig.savefig(png, dpi=dpi)
    with Image.open(png) as im:
        rgb = im.convert("RGB")
    rgb.save(png, dpi=(dpi, dpi), optimize=True)
    STYLE_LOG.append(f"STYLE {stem}: {w:.2f} x {h:.2f} in, smallest font {min(sizes):g} pt, {dpi} dpi, "
                     f"{png.stat().st_size / 1e6:.1f} MB")
    plt.close(fig)


def axes_at(fig, x, y, w, h, **kw):
    """Axes placed in inches from the lower left corner of the figure."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh], **kw)


def canvas(fig, zorder=0):
    """Frameless axes over the whole figure in inches, for diagrams and labels."""
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1], zorder=zorder)
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")
    return ax


def letter(fig, x, y, s: str) -> None:
    """Panel label, e.g. (a), at a position in inches (top left corner of the text)."""
    fw, fh = fig.get_size_inches()
    fig.text(x / fw, y / fh, s, fontsize=FS_LETTER, weight="bold", ha="left", va="top")


def framed(leg):
    leg.get_frame().set_linewidth(0.6)
    return leg


def categorical(ax, axis="y") -> None:
    """Category axis: major ticks at the categories only."""
    ax.tick_params(axis=axis, which="minor", left=False, right=False, top=False, bottom=False)


def dotted_grid(ax, axis="x") -> None:
    ax.grid(axis=axis, which="major", color="#B0B0B0", lw=0.45, ls=(0, (1, 2)))
    ax.set_axisbelow(True)


MASK = dict(boxstyle="square,pad=0.12", fc="white", ec="none")       # white box behind a label in a gridded axes


def check(name: str, got, want, tol: float = 1e-6) -> None:
    """Assert a re-derived value equals the stored one (abs tolerance) and log it."""
    got, want = float(got), float(want)
    ok = (np.isnan(got) and np.isnan(want)) or abs(got - want) <= tol
    CHECKS.append(f"{'OK ' if ok else 'BAD'} {name}: re-derived {got:.6g} vs stored {want:.6g} (tol {tol:g})")
    assert ok, CHECKS[-1]


def jload(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def num(v: float, d: int = 2) -> str:
    """Fixed decimals; one more decimal when a non-zero value would round to zero (e.g. -0.001, not -0.00)."""
    s = f"{v:.{d}f}"
    if v != 0 and float(s) == 0:
        s = f"{v:.{d + 1}f}"
    return s.replace("-", "−")


def fmt_ci(p, lo, hi, d: int = 2) -> str:
    """'0.86 [0.69, 0.92]' with true minus signs."""
    return f"{num(p, d)} [{num(lo, d)}, {num(hi, d)}]"


def sci(v: float) -> str:
    """p-value style number: 0.008 -> '0.008', 0.0009995 (bootstrap floor 1/1001) -> '0.001', 4.1e-05 -> '4 x 10^-5'
    (mathtext), as the analysis docs report them."""
    if round(v, 3) >= 0.001:
        return f"{v:.3f}"
    e = int(np.floor(np.log10(v)))
    mant = round(v / 10**e)
    if mant == 10:
        mant, e = 1, e + 1
    return f"{mant}×10$^{{{e}}}$"


def m(s: str) -> str:
    return s.replace("-", "−")


def clopper_pearson(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def lin_ccc(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(2 * ((x - x.mean()) * (y - y.mean())).mean() / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def plain_log(axis, subs=(1.0, 2.0, 5.0)) -> None:
    """Log axis with plain-number labels at 1-2-5 steps (0.1, 0.2, 0.5, 1, ...)."""
    axis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=subs))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


# =============================================================================================== real data: NASA


def fig_nasa_debris_vs_damage() -> None:
    """C1 / C1b: accumulated debris mass vs destructively pitted teeth, run-time-only comparison, LOO MAE."""
    t = pd.read_csv(OIL / "nasa_experiments_tm2003_table3_9.csv")
    res = jload(OIL / "oil_results.json")["nasa"]
    sp, sr = res["spearman_vs_n_destructive_teeth"]["mass_end_mg"], res["spearman_vs_n_destructive_teeth"]["rdg_end"]
    au = res["auc_destructive_vs_not"]
    check("NASA n experiments", len(t), res["n_experiments"])
    check("NASA Spearman mass vs teeth", stats.spearmanr(t.mass_end_mg, t.n_de_end)[0], sp["spearman"])
    check("NASA Spearman run time vs teeth", stats.spearmanr(t.rdg_end, t.n_de_end)[0], sr["spearman"])
    pos, neg = t.stage_end == 2, t.stage_end != 2
    check("NASA AUC mass destructive vs not", auc(t.mass_end_mg[pos], t.mass_end_mg[neg]), au["mass_end_mg"]["auc"])
    check("NASA AUC time destructive vs not", auc(t.rdg_end[pos], t.rdg_end[neg]), au["rdg_end"]["auc"])
    loo = pd.read_csv(OIL / "nasa_loo_regression_predictions.csv")
    lr = res["loo_regression_n_destructive_teeth"]
    mae = loo.groupby("model").abs_err.mean()
    for k in ("time_only", "debris_only", "debris_plus_time"):
        check(f"NASA LOO MAE {k}", mae[k], lr[k]["loo_mae_teeth"])
    d = lr["debris_only_minus_time_only_mae"]

    H, y0, h = 2.68, 0.46, 1.76
    fig = plt.figure(figsize=(W, H))
    a1 = axes_at(fig, 0.46, y0, 2.22, h)
    a2 = axes_at(fig, 2.92, y0, 2.22, h, sharey=a1)
    a3 = axes_at(fig, 5.66, y0, 1.46, h)
    groups = [(0, "no damage", C["healthy"], "s"), (1, "initial pitting", C["early"], "^"),
              (2, "destructive pitting", C["severe"], "o")]
    for a, x, xl, q, au_ in ((a1, t.mass_end_mg, "Accumulated oil-debris mass at test end (mg)", sp, au["mass_end_mg"]),
                             (a2, t.rdg_end / 60, "Test duration (h, log scale)", sr, au["rdg_end"])):
        for st, lab, c, mk in groups:
            s = t.stage_end == st
            a.scatter(x[s], t.n_de_end[s], s=20, marker=mk, color=c, zorder=3, **EDGE,
                      label=f"{lab} ($n$ = {int(s.sum())})")
        a.set_xlabel(xl)
        a.text(0.035, 0.965, f"Spearman $\\rho$ = {fmt_ci(q['spearman'], q['lo'], q['hi'])}\n"
               f"permutation $p$ {'≤' if q['perm_p'] <= 5.1e-5 else '='} {sci(q['perm_p'])}\n"
               f"AUC, destructive pitting or not:\n{fmt_ci(au_['auc'], au_['lo'], au_['hi'])}",
               transform=a.transAxes, ha="left", va="top", fontsize=FS_SMALL, linespacing=1.15)
    a1.set_ylabel("Destructively pitted teeth (count)")
    a1.set_ylim(-1.4, 36)
    a1.set_yticks([0, 10, 20, 30])
    a1.set_xlim(-1.5, 38.5)
    a2.set_xscale("log")
    a2.set_xlim(2.6, 640)
    a2.set_xticks([5, 10, 20, 50, 100, 200, 500], ["5", "10", "20", "50", "100", "200", "500"])
    plt.setp(a2.get_yticklabels(), visible=False)
    framed(fig.legend(*a1.get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(2.80 / W, 0.998),
                      handletextpad=0.3, columnspacing=1.4))
    # c: paired leave-one-out absolute errors
    w = loo.pivot(index="exp", columns="model", values="abs_err")
    for e, r in w.iterrows():
        a3.plot([0, 1], [r.time_only, r.debris_only], color=C["link"], lw=0.5, zorder=1)
    a3.scatter(np.zeros(len(w)), w.time_only, s=15, color=C["base"], marker="s", zorder=2, clip_on=False, **EDGE)
    a3.scatter(np.ones(len(w)), w.debris_only, s=16, color=C["ours"], zorder=2, clip_on=False, **EDGE)
    for x, k in ((0, "time_only"), (1, "debris_only")):
        a3.plot([x - 0.26, x + 0.26], [mae[k]] * 2, color="black", lw=1.8, zorder=3, solid_capstyle="butt")
        a3.text(x + (-0.32 if x == 0 else 0.32), mae[k], f"{mae[k]:.2f}", va="center", ha="right" if x == 0 else "left",
                fontsize=FS_SMALL)
    a3.set_xticks([0, 1], ["run time\nonly", "debris\nmass"])
    categorical(a3, "x")
    a3.set_xlim(-0.95, 1.95)
    a3.set_ylabel("Leave-one-out absolute error (teeth)")
    a3.text(0.5, 0.965, f"ΔMAE = {num(d['diff'])} teeth\n[{num(d['lo'])}, {num(d['hi'])}]\n$p$ = {sci(d['p_value'])}",
            transform=a3.transAxes, ha="center", va="top", fontsize=FS_SMALL, linespacing=1.15)
    a3.set_ylim(0, max(w.max()) * 1.42)
    for x, s in ((0.02, "(a)"), (2.70, "(b)"), (5.20, "(c)")):
        letter(fig, x, y0 + h + 0.16, s)
    save(fig, "nasa_debris_vs_damage")


def fig_nasa_onset() -> None:
    """C4: per-experiment debris mass at inspections, LOO thresholds, first destructive inspection (primary,
    pre-specified) next to the exploratory accumulation-rate detector (F6) and the detection summary."""
    r = pd.read_csv(OIL / "nasa_onset_readings.csv")
    r = r[(r.feature == "mass_mg") & (r.variant == "table_3_9")].copy()
    pe = pd.read_csv(OIL / "nasa_onset_per_experiment_table_3_9_mass_mg.csv")
    od = jload(OIL / "oil_results.json")["nasa"]["onset_detection"]["table_3_9|mass_mg"]
    check("onset readings", len(r), od["n_readings"])
    check("onset detection (mass, primary)", pe[pe.destructive].detected.mean(), od["exp_detection_rate"])
    check("onset false alarm (mass, primary)", pe[~pe.destructive].alarmed.mean(), od["exp_false_alarm_rate"])
    check("onset early alarm (mass, primary)", pe[pe.destructive].early_alarm.mean(), od["exp_early_alarm_rate"])
    check("onset alarm flag = mass >= LOO tau", (r.alarm == (r.mass_mg >= r.tau)).mean(), 1.0)
    f6 = pd.read_csv(DIAG / "f6_onset_readings.csv")
    f6 = f6[(f6.variant == "table_3_9") & (f6.t0_rule == "runin_t0")].reset_index(drop=True)
    s6 = jload(DIAG / "f6_summary.json")
    check("F6 mass alarms reproduce frozen alarms", (f6.alarm_mass.to_numpy() == r.alarm.to_numpy()).mean(), 1.0)
    e6 = pd.read_csv(DIAG / "f6_onset_per_experiment.csv")
    e6 = e6[(e6.variant == "table_3_9") & (e6.t0_rule == "runin_t0")]
    for det in ("mass", "rate", "mass_or_rate"):
        q, sm = e6[e6.detector == det], s6[f"table_3_9|runin_t0|{det}"]
        k, n = int(q[q.destructive].detected.sum()), int(q.destructive.sum())
        check(f"F6 {det} detection k", k, sm["detection"]["k"])
        check(f"F6 {det} false alarms k", q[~q.destructive].alarmed.sum(), sm["false_alarm"]["k"])
        lo, hi = clopper_pearson(k, n)
        check(f"F6 {det} detection CI lo (Clopper-Pearson)", lo, sm["detection"]["lo"], 1e-6)
        check(f"F6 {det} detection CI hi (Clopper-Pearson)", hi, sm["detection"]["hi"], 1e-6)

    destr = [1, 2, 3, 4, 5, 6, 8, 18, 19, 20, 21, 22, 23, 24]
    order = destr + [7] + list(range(9, 18))
    ypos = {e: i for i, e in enumerate(order)}
    stage = r.groupby("exp").stage.max()
    assert sorted(order) == sorted(stage.index) and (stage[destr] == 2).all() and stage[7] == 1
    first = r[r.y == 1].groupby("exp").reading.min()
    missed = pe[pe.destructive & ~pe.detected].exp.tolist()
    early = pe[pe.early_alarm].exp.tolist()
    rate_missed = e6[(e6.detector == "rate") & e6.destructive & ~e6.detected].exp.tolist()
    rate_early = e6[(e6.detector == "rate") & e6.early_alarm].exp.tolist()

    H, y0, h = 4.42, 0.50, 3.66
    fig = plt.figure(figsize=(W, H))
    ax0 = axes_at(fig, 0.46, y0, 2.18, h)
    ax1 = axes_at(fig, 2.90, y0, 2.18, h, sharey=ax0)

    def rows(a, xcol, alarm_col, d, missed_e, early_e, xmin=None):
        for b0 in (len(destr) - 0.5, len(destr) + 0.5):
            a.axhline(b0, color="black", lw=0.5, zorder=1)
        for e, g in d.groupby("exp"):
            y = ypos[e]
            x = g[xcol].to_numpy(float)
            if xmin is not None:
                x = np.maximum(x, xmin)
            a.plot([x.min(), x.max()], [y, y], color=C["link"], lw=0.8, zorder=1)
            for xi, (_, q) in zip(x, g.iterrows()):
                if e in first.index and q.reading == first[e]:
                    a.plot([xi], [y], marker="o", ms=7.6, mfc="none", mec="black", mew=0.8, ls="", zorder=4)
                a.scatter(xi, y, s=20 if q[alarm_col] else 17, marker="^" if q[alarm_col] else "o",
                          facecolors=C["severe"] if q.y == 1 else "white", edgecolors="black", linewidths=0.5, zorder=3)
            if e in missed_e or e in early_e:
                a.text(0.96, y, "missed" if e in missed_e else "early", transform=a.get_yaxis_transform(), ha="right",
                       va="center", fontsize=FS_SMALL, weight="bold" if e in missed_e else "normal", bbox=MASK)

    rows(ax0, "mass_mg", "alarm", r, missed, early)
    for e, g in r.groupby("exp"):
        ax0.plot(g.tau.iloc[0], ypos[e], marker="|", color="black", ms=7.5, mew=1.3, zorder=2.5)
    ax0.set_xlim(-1.4, 47)
    ax0.set_xlabel("Accumulated oil-debris mass at inspection (mg)")
    ax0.set_yticks(range(len(order)), [str(e) for e in order])
    ax0.set_ylim(len(order) - 0.4, -0.6)
    ax0.set_ylabel("Experiment")
    ax0.set_title("Mass detector (pre-specified)")
    for yy, s in ((len(destr), "initial pitting only"), (len(destr) + 5, "no damage (reading\nat completion only)")):
        ax0.text(0.96, yy, s, transform=ax0.get_yaxis_transform(), ha="right", va="center", fontsize=FS_SMALL,
                 linespacing=1.15, bbox=MASK)

    xmin = 2e-6
    rows(ax1, "rate", "alarm_rate", f6, rate_missed, rate_early, xmin=xmin)
    tmm = s6["table_3_9|runin_t0|rate"]["tau_min_median_max"]["rate"]
    ax1.axvspan(tmm[0], tmm[2], color=C["neutral"], lw=0, zorder=0.5)
    ax1.set_xscale("log")
    ax1.set_xlim(xmin / 1.8, 1.6)
    ax1.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=9))
    ax1.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=range(2, 10), numticks=99))
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_xlabel("Accumulation rate since previous\ninspection (mg min$^{-1}$)", linespacing=1.1)
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.set_title("Rate detector (exploratory)")
    z = f6[f6.rate <= 0]
    for _, q in z.iterrows():
        ax1.annotate("rate 0 (at axis limit)", (xmin, ypos[q.exp]), xytext=(6, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=FS_SMALL, bbox=MASK)
    for a in (ax0, ax1):
        categorical(a, "y")
        dotted_grid(a, "x")

    # c: detection / false alarm with exact CIs
    dets = [("mass", "mass", C["ours"], "o"), ("rate", "rate", C["ours_ext"], "s"),
            ("mass_or_rate", "mass or rate", C["ours_ext"], "D")]
    for key, yy, bar, name in (("detection", 3.26, 0.8, "Detection ($n$ = 14)"),
                               ("false_alarm", 1.96, 0.2, "False alarm ($n$ = 10)")):
        ax2 = axes_at(fig, 6.04, yy, 1.06, 0.90)
        for i, (k, lab, c, mk) in enumerate(dets):
            v = s6[f"table_3_9|runin_t0|{k}"][key]
            ax2.errorbar(v["rate"], i, xerr=[[v["rate"] - v["lo"]], [v["hi"] - v["rate"]]], fmt=mk, color=c,
                         ecolor="black", ms=4.6, clip_on=False, zorder=3, **EB)
        ax2.axvline(bar, ls=DASH, color="black", lw=0.7, zorder=1)
        ax2.set_yticks(range(len(dets)), [d[1] for d in dets])
        ax2.set_ylim(len(dets) - 0.45, -0.55)
        ax2.set_xlim(-0.04, 1.04)
        ax2.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
        categorical(ax2, "y")
        dotted_grid(ax2, "x")
        ax2.set_title(name)
    ax2.set_xlabel("Experiment-level rate")

    mark = dict(ls="", mec="black", mew=0.5)
    h_ = [Line2D([], [], marker="o", ms=4.4, mfc="white", label="before destructive pitting", **mark),
          Line2D([], [], marker="o", ms=4.4, mfc=C["severe"], label="destructive pitting present", **mark),
          Line2D([], [], marker="^", ms=4.8, mfc="white", label="alarm (threshold reached)", **mark),
          Line2D([], [], marker="o", ms=7.6, mfc="none", label="first destructive inspection", **{**mark, "mew": 0.8}),
          Line2D([], [], marker="|", ms=7.5, label="leave-one-out mass threshold", **{**mark, "mew": 1.3}),
          Patch(facecolor=C["neutral"], lw=0, label="range of rate thresholds"),
          Line2D([], [], color="black", ls=DASH, lw=0.7, label="pre-specified limit"),
          Line2D([], [], marker="o", ms=4.6, mfc=C["ours"], label="pre-specified detector", **mark),
          Line2D([], [], marker="s", ms=4.6, mfc=C["ours_ext"], label="exploratory detector", **mark)]
    framed(fig.legend(handles=h_, loc="lower right", bbox_to_anchor=(7.12 / W, 0.06 / H), handlelength=1.3,
                      labelspacing=0.3))
    for x, s in ((0.02, "(a)"), (2.66, "(b)"), (5.20, "(c)")):
        letter(fig, x, H - 0.04, s)
    save(fig, "nasa_onset")


# =============================================================================================== real data: field


def fig_field_phase_rates() -> None:
    """C4: cumulative Fe proxy per tractor vs tractor age and its running value over 100 h."""
    u = pd.read_csv(OIL / "zenodo_unit_rates.csv")
    run = pd.read_csv(OIL / "zenodo_running_rates_Fe.csv")
    pc = pd.read_csv(OIL / "zenodo_phase_contrasts.csv")
    pc = pc[(pc.subset == "all_units") & (pc.feature == "rate_cum_Fe_q0")].set_index(["a", "b"])
    med = u.groupby("phase").rate_cum_Fe_q0.median()
    check("field median Fe proxy wear-out", med["wear_out"], pc.loc[("wear_out", "useful_life")].median_a)
    check("field median Fe proxy useful", med["useful_life"], pc.loc[("wear_out", "useful_life")].median_b)
    check("field median Fe proxy infant", med["infant_mortality"],
          pc.loc[("useful_life", "infant_mortality")].median_b)
    last = run[run.working_hours == 100].set_index("unit_id").running_rate
    check("field running rate at 100 h = unit proxy (max abs diff)",
          np.abs(last.reindex(u.unit_id).to_numpy() - u.rate_cum_Fe_q0.to_numpy()).max(), 0.0, 1e-9)
    check("field units", len(u), 27)

    H, y0, h = 2.62, 0.46, 2.00
    fig = plt.figure(figsize=(W, H))
    a = axes_at(fig, 0.60, y0, 3.10, h)
    rng = np.random.default_rng(0)
    for ph, (lab, c, mk) in PHASE.items():
        g = u[u.phase == ph]
        x = g.tractor_age + rng.uniform(-0.18, 0.18, len(g))
        a.scatter(x, g.rate_cum_Fe_q0, s=19, color=c, marker=mk, zorder=3, label=f"{lab} ($n$ = {len(g)})", **EDGE)
        a.plot([g.tractor_age.min() - 0.4, g.tractor_age.max() + 0.4], [med[ph]] * 2, color="black", lw=1.2, zorder=2)
    a.set_yscale("log")
    plain_log(a.yaxis)
    a.set_xlabel("Tractor age (years)")
    a.set_ylabel("Fe proxy $G$(100 h)/100 h (ppm h$^{-1}$)")
    a.set_xticks(range(1, 14))
    a.tick_params(axis="x", which="minor", bottom=False, top=False)
    a.set_xlim(0.2, 13.8)
    a.set_ylim(0.1, 14)
    hd, lab = a.get_legend_handles_labels()
    framed(a.legend(hd + [Line2D([], [], color="black", lw=1.2)], lab + ["phase median"], loc="upper left",
                    handletextpad=0.4, handlelength=1.3))
    wu, ui = pc.loc[("wear_out", "useful_life")], pc.loc[("useful_life", "infant_mortality")]
    for x, y, ha, name, q in ((0.965, 0.60, "right", "wear-out vs useful life", wu),
                              (0.035, 0.46, "left", "useful life vs infant mortality", ui)):
        a.text(x, y, f"{name}:\nCliff's $\\delta$ = "
               f"{fmt_ci(q.cliffs_delta, q.cliffs_delta_lo, q.cliffs_delta_hi)}\nexact $p$ = {sci(q.perm_p)}",
               transform=a.transAxes, ha=ha, va="top", fontsize=FS_SMALL, linespacing=1.15)
    b = axes_at(fig, 4.38, y0, 2.70, h)
    for (unit, ph), g in run.groupby(["unit_id", "phase"]):
        b.plot(g.working_hours, g.running_rate, color=PHASE[ph][1], lw=0.7, marker=PHASE[ph][2], ms=3.4, mec="black",
               mew=0.3)
    b.set_yscale("log")
    plain_log(b.yaxis)
    b.set_xlabel("Working hours since the oil change (h)")
    b.set_ylabel("Running proxy $G$($t$)/$t$ (ppm h$^{-1}$)")
    b.set_xticks([20, 40, 60, 80, 100])
    b.set_xlim(14, 106)
    b.set_ylim(0.14, 9)
    letter(fig, 0.02, H - 0.03, "(a)")
    letter(fig, 3.86, H - 0.03, "(b)")
    save(fig, "field_phase_rates")


def fig_field_composition() -> None:
    """C2: wear-metal composition ln(X/Fe) by phase and LOO balanced accuracy composition vs Fe-only (+ F7)."""
    f = pd.read_csv(OIL / "zenodo_composition_features.csv")
    sh = pd.read_csv(OIL / "zenodo_composition_shift.csv").set_index(["alr", "a", "b"])
    res = pd.read_csv(OIL / "zenodo_composition_results.csv")
    res = res[res.subset == "all_units"].set_index(["task", "features"])
    f7 = pd.read_csv(DIAG / "f7_composition_sensitivity.csv")
    f7 = f7[f7.threshold == 5]
    s7 = jload(DIAG / "f7_summary.json")
    metals = ["Cr", "Al", "Cu", "Pb", "Mn"]
    for mt in metals:
        g = f.groupby("phase")[f"alr_{mt}"].median()
        check(f"field median ln({mt}/Fe) wear-out", g["wear_out"], sh.loc[(f"alr_{mt}", "wear_out", "useful_life")].median_a)
        check(f"field median ln({mt}/Fe) useful", g["useful_life"],
              sh.loc[(f"alr_{mt}", "wear_out", "useful_life")].median_b)
    comp7, fe7 = f7[f7.features == "composition"], f7[f7.features == "fe_only"]
    check("F7 subsets", f7.subset.nunique(), s7["mis_ge5"]["n_subsets"])
    check("F7 composition BA min over subsets", comp7.balanced_accuracy.min(), 1.0)
    check("F7 Fe-only BA max over subsets", fe7.balanced_accuracy.max(), 2 / 3)
    check("F7 Fe-only BA min over subsets", fe7.balanced_accuracy.min(), 2 / 3)

    H, y0, h = 2.80, 0.56, 1.86
    fig = plt.figure(figsize=(W, H))
    a = axes_at(fig, 0.46, y0, 3.72, h)
    rng = np.random.default_rng(1)
    for i, mt in enumerate(metals):
        for j, (ph, (lab, c, mk)) in enumerate(PHASE.items()):
            v = f[f.phase == ph][f"alr_{mt}"]
            x = i + (j - 1) * 0.24 + rng.uniform(-0.06, 0.06, len(v))
            a.scatter(x, v, s=13, color=c, marker=mk, zorder=3, label=lab if i == 0 else None, edgecolors="black",
                      linewidths=0.3)
            a.plot([i + (j - 1) * 0.24 - 0.11, i + (j - 1) * 0.24 + 0.11], [v.median()] * 2, color="black",
                   lw=1.4, zorder=4, solid_capstyle="butt")
        dl = sh.loc[(f"alr_{mt}", "wear_out", "useful_life")].cliffs_delta
        a.text(i, 0.965, m(f"$\\delta$ = {dl:+.1f}"), transform=a.get_xaxis_transform(), ha="center", va="top",
               fontsize=FS_SMALL)
    a.set_xticks(range(len(metals)), [f"ln({mt}/Fe)" for mt in metals])
    a.set_xlim(-0.55, len(metals) - 0.45)
    a.set_ylim(-5.15, -1.05)
    a.set_yticks([-5, -4, -3, -2])
    a.set_ylabel("ln(X/Fe), median of 5 samples")
    hd, lab = a.get_legend_handles_labels()
    top = (y0 + h + 0.07) / H
    framed(fig.legend(hd + [Line2D([], [], color="black", lw=1.4)], lab + ["phase median"], loc="lower center", ncol=4,
                      bbox_to_anchor=((0.46 + 3.72 / 2) / W, top), handletextpad=0.3, columnspacing=1.0,
                      handlelength=1.1, borderaxespad=0))

    b = axes_at(fig, 4.80, y0, 2.30, h)
    tasks = [("3class", "3 phases\n(27 tractors)"), ("infant_vs_useful", "infant vs\nuseful life\n(18 tractors)")]
    feats = (("composition", "composition", C["ours"]), ("fe_only", "iron only", C["base"]))
    bar = dict(width=0.34, edgecolor="black", lw=0.5, zorder=2)
    for i, (task, lab) in enumerate(tasks):
        for j, (feat, flab, c) in enumerate(feats):
            q = res.loc[(task, feat)]
            x = i + (j - 0.5) * 0.36
            b.bar(x, q.balanced_accuracy, color=c, **bar)
            b.errorbar(x, q.balanced_accuracy, yerr=[[q.balanced_accuracy - q.ba_lo], [q.ba_hi - q.balanced_accuracy]],
                       fmt="none", ecolor="black", elinewidth=0.8, capsize=2.2, zorder=3)
        dq = res.loc[(task, "composition_minus_fe_only")]
        b.text(i, 1.53, f"Δ = {num(dq.balanced_accuracy)}\n[{num(dq.ba_lo)}, {num(dq.ba_hi)}]\n"
               f"$p$ = {sci(dq.perm_p_balanced_accuracy)}", ha="center", va="top", fontsize=FS_SMALL, linespacing=1.15)
    x = 2
    for dx, q, c in ((-0.18, comp7, C["ours"]), (0.18, fe7, C["base"])):
        b.bar(x + dx, q.balanced_accuracy.iloc[0], color=c, hatch="////", **bar)
    b.text(x, 1.53, "all 4 subsets\nidentical", ha="center", va="top", fontsize=FS_SMALL, linespacing=1.15)
    b.set_xticks([0, 1, 2], [tasks[0][1], tasks[1][1], "duplicate-\nfree subsets\n(16 tractors)"], linespacing=1.1)
    for x0, x1, ch in ((-0.5, 0.5, 1 / 3), (0.5, 1.5, 0.5), (1.5, 2.5, 1 / 3)):
        b.plot([x0, x1], [ch, ch], color="black", lw=0.7, ls=DASH, zorder=1)
    b.set_ylim(0, 1.58)
    b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    b.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(np.arange(0.05, 1.0, 0.05)))
    b.set_xlim(-0.5, 2.5)
    b.set_ylabel("Balanced accuracy")
    for a_ in (a, b):
        categorical(a_, "x")
        dotted_grid(a_, "y")
    framed(fig.legend(handles=[Patch(facecolor=c, edgecolor="black", lw=0.5, label=lab) for _, lab, c in feats]
                      + [Line2D([], [], color="black", lw=0.7, ls=DASH, label="chance")], loc="lower center", ncol=3,
                      bbox_to_anchor=((4.80 + 2.30 / 2) / W, top), handlelength=1.3, columnspacing=1.0,
                      borderaxespad=0))
    letter(fig, 0.02, H - 0.06, "(a)")
    letter(fig, 4.32, H - 0.06, "(b)")
    save(fig, "field_composition")


# =============================================================================================== real data: ferrograms

METHODS = {  # label, colour, marker, marker face, line style
    "ours_tuned": ("ours (tuned)", C["ours"], "o", C["ours"], "-"),
    "otsu": ("Otsu", C["base"], "s", C["base"], DASH),
    "local_default": ("local (default)", C["base"], "^", "white", DOT),
    "local_tuned": ("local (tuned)", C["base_dark"], "D", C["base_dark"], DASHDOT)}
PRED_C = C["ours"]


def casing(lw: float) -> list:
    """Black casing of a light line drawn on a photograph."""
    return [patheffects.Stroke(linewidth=lw + 0.9, foreground="black"), patheffects.Normal()]


def _zen_index() -> pd.DataFrame:
    from wearpg.open_data import load_zenodo_ferro
    idx = load_zenodo_ferro(ZEN).set_index("image_id")
    for c in ("image_path", "mask_path"):
        idx[c] = [str(ROOT / p) for p in idx[c]]
    return idx


def _near_dups() -> set[str]:
    leak = pd.read_csv(FER / "seg_leak_check.csv")
    return set(leak[leak.near_dup].test_image)


def _closed(c: np.ndarray) -> np.ndarray:
    return np.vstack([c, c[:1]])


def fig_ferro_overlays() -> None:
    """C5: four testO ferrograms (10/40/70/90th percentile of our per-image IoU among the 80 non-duplicate images)
    with GT and predicted contours; the predicted masks are re-created with the frozen trainO-tuned settings and must
    reproduce the stored per-image scores exactly."""
    from wearpg.open_data import mask_components, pixel_scores, read_mask
    from wearpg.segmentation import load_image
    sys.path.insert(0, str(ROOT / "scripts"))
    import open_ferro_eval as ev  # frozen benchmark code, imported read-only

    per = pd.read_csv(FER / "seg_per_image_test.csv")
    t = per[(per.method == "ours_tuned") & ~per.image_id.isin(_near_dups())].sort_values("iou")
    t = t.reset_index(drop=True)
    check("overlay pool = testO without near duplicates", len(t), 80)
    qs = (0.1, 0.4, 0.7, 0.9)
    picks = [t.iloc[int(round(q * (len(t) - 1)))] for q in qs]
    meth = ev.methods(jload(FER / "seg_tuned_params.json"))["ours_tuned"]
    idx = _zen_index()
    s_, x0_, gx, ylo, yhi = 1.70, 0.05, 0.10, 0.40, 2.26  # 512 px over 1.70 in: 301 dpi
    H = 4.32
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)
    win = 150  # zoom window (px), placed where ground truth and prediction cover the most pixels
    for col, (r, q) in enumerate(zip(picks, qs)):
        img, gt = load_image(idx.loc[r.image_id, "image_path"]), read_mask(idx.loc[r.image_id, "mask_path"])
        pred, pc = ev.predict(img, *meth)
        sc = pixel_scores(pred, gt)
        check(f"overlay {r.image_id} IoU reproduces stored", sc["iou"], r.iou, 1e-12)
        check(f"overlay {r.image_id} predicted area reproduces stored", sc["area_pred"], r.area_pred, 0)
        cs = np.pad((gt | pred).astype(float).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        h, w = gt.shape
        y0, x0 = max(((y, x) for y in range(0, h - win + 1, 8) for x in range(0, w - win + 1, 8)),
                     key=lambda p: cs[p[0] + win, p[1] + win] - cs[p[0], p[1] + win] - cs[p[0] + win, p[1]]
                     + cs[p[0], p[1]])
        gcont = [c["contour"] for c in mask_components(gt, 30)]
        xa = x0_ + col * (s_ + gx)
        for yy, lw, zoom in ((yhi, 0.5, False), (ylo, 1.0, True)):
            a = axes_at(fig, xa, yy, s_, s_)
            a.imshow(img, interpolation="none")
            for c in gcont:  # ground truth on photographs: white line in a black casing (visible on any background)
                a.plot(*_closed(c).T, color="white", lw=lw, path_effects=casing(lw))
            for c in pc:
                a.plot(*_closed(c).T, color=PRED_C, lw=lw)
            if zoom:
                a.set_xlim(x0 - 0.5, x0 + win - 0.5)
                a.set_ylim(y0 + win - 0.5, y0 - 0.5)
            else:
                a.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), win, win, fill=False, ec="white", lw=1.3, zorder=3))
                a.set_xlim(-0.5, w - 0.5)
                a.set_ylim(h - 0.5, -0.5)
            a.set_xticks([])
            a.set_yticks([])
        letter(fig, xa - 0.02, H - 0.03, f"({'abcd'[col]})")
        cv.text(xa + 0.26, H - 0.04, m(f"{r.image_id.replace('_', ' ')}, {int(q * 100)}th percentile\nIoU {r.iou:.2f}, "
                                      f"area error {100 * r.area_rel_err:+.0f}%"), fontsize=FS_SMALL, va="top",
                linespacing=1.2)
    framed(fig.legend(handles=[Line2D([], [], color="white", lw=1.2, path_effects=casing(1.2),
                                      label="ground-truth mask"),
                               Line2D([], [], color=PRED_C, lw=1.4, label="ours (frozen settings tuned on trainO)"),
                               Line2D([], [], color="none", marker="s", ms=6, mfc="none", mec="black", mew=0.8,
                                      label=f"white box: detail of {win} × {win} px (lower row)")],
                      loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04 / H), columnspacing=1.6))
    save(fig, "ferro_overlays")


def fig_seg_benchmark() -> None:
    """C5: testO metrics per method (95 % image-bootstrap CIs) and paired differences ours - baseline (Holm p)."""
    s = jload(FER / "seg_summary_test.json")
    per = pd.read_csv(FER / "seg_per_image_test.csv")
    pt = pd.read_csv(FER / "seg_paired_test.csv")
    for k in METHODS:
        d = per[per.method == k]
        check(f"seg {k} n images", len(d), s["n_images"])
        check(f"seg {k} mean IoU", d.iou.mean(), s[k]["iou"]["point"])
        check(f"seg {k} mean |area error|", d.area_rel_err.abs().mean(), s[k]["abs_area_rel_err"]["point"])
        check(f"seg {k} CCC log area", lin_ccc(np.log(d.area_gt), np.log(d.area_pred.clip(lower=1))),
              s[k]["ccc_log_area"]["point"])
    leak = pd.read_csv(FER / "seg_leak_check.csv")
    ceil = leak[leak.near_dup & (leak.thumb_corr > 0.9999)].mask_iou
    check("pixel-identical pairs", len(ceil), 3)

    unet = jload(FER / "unet" / "summary.json")["CLEAN_all86"]["boot"]      # post-hoc learned baseline
    up = pd.read_csv(FER / "unet" / "per_image.csv").query("arm == 'CLEAN'").groupby("image_id")
    check("U-Net mean IoU", up.iou.mean().mean(), unet["iou"]["point"])
    check("U-Net mean |area error|", up.area_rel_err.apply(lambda e: e.abs().mean()).mean(), unet["abs"]["point"])
    H, yt, ht, yb, hb, xl, gap = 3.58, 2.06, 1.22, 0.46, 0.98, 0.94, 0.22
    fig = plt.figure(figsize=(W, H))
    w3, w2 = (7.10 - xl - 2 * gap) / 3, (7.10 - xl - gap) / 2
    ax = [axes_at(fig, xl, yt, w3, ht)]
    ax += [axes_at(fig, xl + i * (w3 + gap), yt, w3, ht, sharey=ax[0]) for i in (1, 2)]
    ax.append(axes_at(fig, xl, yb, w2, hb))
    ax.append(axes_at(fig, xl + w2 + gap, yb, w2, hb, sharey=ax[3]))
    keys = list(METHODS)
    for a, met, lab in ((ax[0], "iou", "Pixel IoU"), (ax[1], "ccc_log_area", "CCC of ln(debris area)"),
                        (ax[2], "abs_area_rel_err", "|Relative area error|")):
        for i, k in enumerate(keys):
            v = s[k][met]
            _, c, mk, fc, _ = METHODS[k]
            a.errorbar(v["point"], i, xerr=[[v["point"] - v["lo"]], [v["hi"] - v["point"]]], fmt=mk, color=c, mfc=fc,
                       ecolor="black", ms=4.8, zorder=3, **EB)
        v = unet[{"iou": "iou", "ccc_log_area": "ccc", "abs_area_rel_err": "abs"}[met]]
        a.errorbar(v["point"], len(keys), xerr=[[v["point"] - v["lo"]], [v["hi"] - v["point"]]], fmt="P", color=C["ours_ext"],
                   mfc=C["ours_ext"], ecolor="black", ms=5.4, zorder=3, **EB)
        a.set_xlabel(lab)
    ax[0].set_yticks(range(len(keys) + 1), [METHODS[k][0] for k in keys] + ["U-Net (post-hoc)"])
    ax[0].set_ylim(len(keys) + 0.55, -0.55)
    ax[0].axvspan(ceil.min(), ceil.max(), color=C["neutral"], lw=0, zorder=0)
    ax[0].text(ceil.min() - 0.01, 2.5, "agreement of\nredrawn masks", fontsize=FS_SMALL, ha="right", va="center",
               linespacing=1.15, bbox=MASK)
    ax[0].set_xlim(0.45, 0.97)
    ax[1].axvline(0.8, color="black", ls=DASH, lw=0.7, zorder=1)
    ax[1].text(0.8, 1.02, "pre-specified threshold", transform=ax[1].get_xaxis_transform(), fontsize=FS_SMALL, ha="center",
               va="bottom")
    ax[1].set_xlim(0.35, 1.02)
    ax[2].set_xlim(0, 1.25)
    ax[2].set_xticks([0, 0.4, 0.8, 1.2])
    base = ["otsu", "local_tuned", "local_default"]
    for a, met, lab in ((ax[3], "iou", "Δ pixel IoU, ours − baseline"),
                        (ax[4], "abs_area_rel_err", "Δ |relative area error|, ours − baseline")):
        for j, b in enumerate(base):
            q = pt[(pt.a == "ours_tuned") & (pt.b == b) & (pt.metric == met)].iloc[0]
            _, c, mk, fc, _ = METHODS[b]
            a.errorbar(q["diff"], j, xerr=[[q["diff"] - q.lo], [q.hi - q["diff"]]], fmt=mk, color=c, mfc=fc,
                       ecolor="black", ms=4.8, zorder=3, **EB)
            a.text(0.955, j, f"$p$ = {sci(q.p_holm)}", transform=a.get_yaxis_transform(), ha="right", va="center",
                   fontsize=FS_SMALL, bbox=MASK)
        a.axvline(0, color="black", lw=0.7, zorder=1)
        a.set_xlabel(lab)
    ax[3].set_yticks(range(len(base)), [f"vs {METHODS[b][0]}" for b in base])
    ax[3].set_ylim(len(base) - 0.5, -0.95)
    ax[3].set_xlim(-0.03, 0.135)
    ax[3].set_xticks([0, 0.04, 0.08, 0.12])
    ax[4].set_xlim(-1.05, 0.75)
    ax[4].set_xticks([-0.8, -0.4, 0, 0.4])
    for a, s_, x, ha in ((ax[3], "ours better →", 0.0035, "left"), (ax[4], "← ours better", -0.035, "right")):
        a.text(x, -0.64, s_, ha=ha, va="center", fontsize=FS_SMALL, bbox=MASK)
    for a in ax:
        categorical(a, "y")
        dotted_grid(a, "x")
    for a in (ax[1], ax[2], ax[4]):
        plt.setp(a.get_yticklabels(), visible=False)
    for i, (x, y) in enumerate([(0.02, yt + ht + 0.24)] + [(xl + j * (w3 + gap) - 0.20, yt + ht + 0.24) for j in (1, 2)]
                               + [(0.02, yb + hb + 0.24), (xl + w2 + gap - 0.20, yb + hb + 0.24)]):
        letter(fig, x, y, f"({'abcde'[i]})")
    save(fig, "seg_benchmark")


def fig_debris_area_agreement() -> None:
    """C5a: per-image debris area predicted vs GT (log-log) with CCC and slope; relative error vs GT area."""
    s = jload(FER / "seg_summary_test.json")["ours_tuned"]
    per = pd.read_csv(FER / "seg_per_image_test.csv")
    d = per[per.method == "ours_tuned"].copy()
    x, y = np.log(d.area_gt), np.log(d.area_pred.clip(lower=1))
    fit = stats.linregress(x, y)
    check("area CCC (ln area)", lin_ccc(x, y), s["ccc_log_area"]["point"])
    check("area OLS slope (ln pred on ln GT)", fit.slope, s["slope_log_area"]["point"])
    check("area Spearman", stats.spearmanr(d.area_pred, d.area_gt)[0], s["spearman_area"]["point"])
    check("area mean relative error", d.area_rel_err.mean(), s["area_rel_err"]["point"])
    dup = d.image_id.isin(_near_dups())

    H, y0, h = 2.92, 0.46, 2.34
    fig = plt.figure(figsize=(W, H))
    a = axes_at(fig, 0.66, y0, h, h)
    b = axes_at(fig, 3.92, y0, 3.16, h)
    dots = (dict(s=15, color=C["ours"], marker="o", zorder=3, **EDGE),
            dict(s=22, color=C["dup"], marker="D", zorder=4, edgecolors="black", linewidths=0.6))
    for ax, y_ in ((a, d.area_pred), (b, 100 * d.area_rel_err)):
        for sel, kw, lab in ((~dup, dots[0], "test images ($n$ = 80)"),
                             (dup, dots[1], "near duplicates of\ntraining images ($n$ = 6)")):
            ax.scatter(d.area_gt[sel], y_[sel], label=lab, **kw)
        ax.set_xscale("log")
        ax.set_xlabel("Ground-truth debris area per image (px)")
    lim = np.array([2e3, 3e5])
    a.plot(lim, lim, color="black", lw=0.7, ls=DASH, label="1:1", zorder=1)
    xx = np.linspace(np.log(lim[0]), np.log(lim[1]), 50)
    a.plot(np.exp(xx), np.exp(fit.intercept + fit.slope * xx), color="black", lw=1.1, label="OLS fit (ln–ln)", zorder=2)
    a.set_yscale("log")
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_ylabel("Segmented debris area per image (px)")
    framed(a.legend(loc="lower right", handlelength=1.7, borderaxespad=0.4))
    c, sl = s["ccc_log_area"], s["slope_log_area"]
    a.text(0.035, 0.965, f"CCC = {fmt_ci(c['point'], c['lo'], c['hi'], 3)}\n(pre-specified: ≥ 0.8)\n"
           f"slope = {fmt_ci(sl['point'], sl['lo'], sl['hi'], 3)}", transform=a.transAxes, ha="left", va="top",
           fontsize=FS_SMALL, linespacing=1.2)
    b.axhline(0, color="black", lw=0.7, ls=DASH, zorder=1)
    e = s["area_rel_err"]
    b.axhspan(100 * e["lo"], 100 * e["hi"], color=C["neutral"], lw=0, zorder=0)
    b.axhline(100 * e["point"], color="black", lw=1.1, zorder=2)
    b.set_xlim(*lim)
    b.set_ylim(-68, 158)
    b.set_ylabel("Relative area error (%)")
    bias = f"mean bias {num(100 * e['point'], 1)}% [{num(100 * e['lo'], 1)}, {num(100 * e['hi'], 1)}]"
    framed(b.legend([Line2D([], [], color="black", lw=0.7, ls=DASH),
                     (Patch(facecolor=C["neutral"], lw=0), Line2D([], [], color="black", lw=1.1))],
                    ["no error", bias], loc="upper right", handlelength=1.7))
    letter(fig, 0.02, H - 0.04, "(a)")
    letter(fig, 3.28, H - 0.04, "(b)")
    save(fig, "debris_area_agreement")


def _thumb(img: np.ndarray, f: int = 2) -> np.ndarray:
    """Block-mean downsample by f (keeps PDF size small; display only)."""
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    x = img[:h, :w].astype(float) / (255.0 if img.dtype == np.uint8 else 1.0)
    return x.reshape(h // f, f, w // f, f, *x.shape[2:]).mean(axis=(1, 3))


def fig_leakage_audit() -> None:
    """C6: best train match of every testO image, the six near-duplicate pairs and their independently drawn
    masks (annotation ceiling), and IoU with / without the duplicates."""
    from wearpg.open_data import read_mask
    from wearpg.segmentation import load_image
    leak = pd.read_csv(FER / "seg_leak_check.csv")
    ls = jload(FER / "seg_leak_summary.json")
    per = pd.read_csv(FER / "seg_per_image_test.csv")
    check("leak n test", len(leak), ls["n_test"])
    check("leak near duplicates", leak.near_dup.sum(), ls["n_near_dup"])
    check("leak exact duplicates", leak.exact_dup.sum(), ls["n_exact_dup"])
    check("leak rule r>=0.95 & IoU>=0.5 reproduces flags",
          ((leak.thumb_corr >= 0.95) & (leak.mask_iou >= 0.5) | leak.exact_dup).eq(leak.near_dup).mean(), 1.0)
    pairs = leak[leak.near_dup].sort_values("thumb_corr", ascending=False)
    idx = _zen_index()
    H, s_, gx, xg, xl, wl = 3.60, 0.745, 0.045, 2.47, 0.56, 1.46
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)
    ah = axes_at(fig, xl, 2.28, wl, 1.12)
    bins = np.arange(0.1, 1.0001, 0.05)
    ah.hist(leak.thumb_corr, bins=bins, color=C["base"], edgecolor="black", lw=0.4)
    ah.hist(pairs.thumb_corr, bins=bins, color=C["dup"], edgecolor="black", lw=0.4, hatch="//////")
    ah.axvline(0.95, color="black", ls=DASH, lw=0.7)
    ah.set_xlabel("Best match in trainO, thumbnail $r$")
    ah.set_ylabel("Test images (count)")
    ah.set_xlim(0.1, 1.04)
    ah.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ah.set_ylim(0, 17)
    ah.text(0.925, 16.0, "$n$ = 6: $r$ ≥ 0.95\nand mask IoU ≥ 0.5", ha="right", va="top", fontsize=FS_SMALL,
            linespacing=1.15)
    rows_y = [2.46, 1.54, 0.62]
    for yy, lab in zip(rows_y, ("test image", "training twin", "masks")):
        cv.text(xg - 0.06, yy + s_ / 2, lab, rotation=90, ha="right", va="center", fontsize=FS_SMALL)
    for j, (_, r) in enumerate(pairs.iterrows()):
        it, ir = load_image(idx.loc[r.test_image, "image_path"]), load_image(idx.loc[r.best_train_image, "image_path"])
        mt, mr = read_mask(idx.loc[r.test_image, "mask_path"]), read_mask(idx.loc[r.best_train_image, "mask_path"])
        assert r.dihedral == 0
        iou = (mt & mr).sum() / (mt | mr).sum()
        check(f"leak mask IoU {r.test_image}-{r.best_train_image}", iou, r.mask_iou, 1e-9)
        same = it.shape == ir.shape and np.array_equal(it, ir)
        if r.thumb_corr > 0.9999:
            check(f"leak {r.test_image} pixel-identical to {r.best_train_image}", same, True)
        agree = np.zeros((*mt.shape, 3))
        agree[mt & mr] = 1.0
        agree[mt & ~mr] = matplotlib.colors.to_rgb(C["dup"])
        agree[~mt & mr] = matplotlib.colors.to_rgb(C["base"])
        x = xg + j * (s_ + gx)
        for yy, im, lab in zip(rows_y, (it, ir, agree), (r.test_image.replace("_", " "),
                                                         r.best_train_image.replace("_", " "),
                                                         f"mask IoU {r.mask_iou:.2f}")):
            a = axes_at(fig, x, yy, s_, s_)
            a.imshow(_thumb(im), interpolation="none")
            a.set_xticks([])
            a.set_yticks([])
            cv.text(x + s_ / 2, yy - 0.03, lab, ha="center", va="top", fontsize=FS_SMALL)
        cv.text(x + s_ / 2, rows_y[0] + s_ + 0.04, f"{'pixel-identical' if same else 'near-identical'}\n"
                f"$r$ = {r.thumb_corr:.4f}", ha="center", va="bottom", fontsize=FS_SMALL, linespacing=1.15)
    framed(fig.legend(handles=[Patch(fc="white", ec="black", lw=0.5, label="both masks"),
                               Patch(fc=C["dup"], ec="black", lw=0.5, label="test mask only"),
                               Patch(fc=C["base"], ec="black", lw=0.5, label="training mask only"),
                               Patch(fc="black", ec="black", lw=0.5, label="background in both")],
                      loc="lower center", ncol=4, bbox_to_anchor=((xg + 3 * s_ + 2.5 * gx) / W, 0.05 / H),
                      handlelength=1.1, columnspacing=1.2))
    # c: IoU with / without the duplicates
    ac = axes_at(fig, xl, 0.48, wl, 1.12)
    nd = pd.DataFrame(ls["test_metrics_by_near_dup"])
    dupset = set(pairs.test_image)
    for i, k in enumerate(METHODS):
        d = per[per.method == k]
        clean = d[~d.image_id.isin(dupset)].iou.mean()
        check(f"leak {k} IoU without duplicates", clean,
              nd[(nd.method == k) & ~nd.near_dup].iou.iloc[0], 5e-5)
        lab, c, mk, fc, st = METHODS[k]
        ac.plot([0, 1], [d.iou.mean(), clean], color=c, marker=mk, mfc=fc, mec="black", mew=0.4, ms=4.2, lw=1.0, ls=st)
        ac.text(1.14, clean, lab, ha="left", va="center", fontsize=FS_SMALL)
    ac.set_xticks([0, 1], ["all\n(86)", "without\n(80)"])
    categorical(ac, "x")
    ac.set_xlim(-0.25, 2.55)
    ac.set_ylabel("Mean pixel IoU")
    ac.set_ylim(0.525, 0.625)
    for x, y, s in ((0.02, 3.58, "(a)"), (2.14, 3.58, "(b)"), (0.02, 1.78, "(c)")):
        letter(fig, x, y, s)
    save(fig, "leakage_audit")


GROUPS = {"shape": ("shape", C["base_dark"], "o"), "graph_geom": ("graph geometry", C["graph"], "s"),
          "efd": ("elliptic Fourier (EFD)", C["base"], "^")}
TRANSFORMS = [("rot45", "45°"), ("rot_random", "random"), ("rot90", "90°"), ("scale0.5", "0.5×"),
              ("scale0.25", "0.25×"), ("blur1", "1 px"), ("blur2", "2 px"), ("noise0.02", "0.02"),
              ("noise0.05", "0.05")]
FAMILY = {"rot": "rotation", "scale": "rescale", "blur": "blur", "noise": "noise SD"}


def fig_invariance_real() -> None:
    """C3 (mask mode) and C5 robustness (pipeline mode): group-median Spearman per transform with the frozen
    thresholds (for us: >= 0.9 rotation, >= 0.8 at 0.5x and blur 1; against us: < 0.7 under rotation or 0.5x)."""
    st = pd.read_csv(FER / "robust_stability.csv")
    rs = jload(FER / "robust_summary.json")
    med = st.groupby(["mode", "transform", "group"]).spearman.median()
    for row in rs["group_median_spearman"]:
        for g in ("shape", "graph_geom", "efd", "fractal", "graph_tex"):
            check(f"robust {row['mode']} {row['transform']} {g}", med[(row["mode"], row["transform"], g)], row[g],
                  5e-5)
    nn = st[st.group == "graph_geom"].groupby(["mode", "transform"]).n.median()
    H, y0, h, xl, gap = 2.78, 0.64, 1.56, 0.56, 0.22
    u = (7.10 - xl - gap) / 16  # in per transform; 7 in (a), 9 in (b)
    fig = plt.figure(figsize=(W, H))
    a0 = axes_at(fig, xl, y0, 7 * u, h)
    ax = [a0, axes_at(fig, xl + 7 * u + gap, y0, 9 * u, h, sharey=a0)]
    for a, mode in zip(ax, ("mask", "pipeline")):
        ts = [t for t in TRANSFORMS if (mode, t[0]) in nn.index]
        x = np.arange(len(ts))
        for i, (t, _) in enumerate(ts):
            limits = [(0.9, DASH)] if t.startswith("rot") else [(0.8, DASH)] if t in ("scale0.5", "blur1") else []
            limits += [(0.7, DOT)] if t.startswith("rot") or t == "scale0.5" else []
            for v, st in limits:
                a.plot([i - 0.42, i + 0.42], [v, v], color="black", ls=st, lw=0.8 if st == DASH else 1.2, zorder=2)
        for k, (g, (lab, c, mk)) in enumerate(GROUPS.items()):
            y = [med[(mode, t, g)] for t, _ in ts]
            a.scatter(x + (k - 1) * 0.22, y, color=c, marker=mk, s=20, zorder=3, label=lab, **EDGE)
        a.set_xticks(x, [f"{lab}\n({int(nn[(mode, t)])})" for t, lab in ts], linespacing=1.2)
        fams = [FAMILY[next(f for f in FAMILY if t.startswith(f))] for t, _ in ts]
        for fam in dict.fromkeys(fams):
            pos = [i for i, f in enumerate(fams) if f == fam]
            a.text(np.mean(pos), -0.245, fam, transform=a.get_xaxis_transform(), fontsize=FS_NOTE, ha="center",
                   va="top", weight="bold")
            if pos[0] > 0:
                a.axvline(pos[0] - 0.5, color="black", lw=0.5, zorder=1)
        a.set_xlim(-0.5, len(ts) - 0.5)
        categorical(a, "x")
        dotted_grid(a, "y")
    ax[0].set_ylim(0.2, 1.05)
    ax[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax[0].set_ylabel("Spearman $\\rho$ with reference")
    ax[0].set_title("Contours from transformed masks")
    ax[1].set_title("Transformed images re-segmented")
    plt.setp(ax[1].get_yticklabels(), visible=False)
    hd, lab = ax[0].get_legend_handles_labels()
    hd += [Line2D([], [], color="black", ls=DASH, lw=0.8), Line2D([], [], color="black", ls=DOT, lw=1.2)]
    lab += ["threshold for the claim", "against the claim"]
    framed(fig.legend(hd, lab, loc="upper center", ncol=5, bbox_to_anchor=((xl + 7.10) / 2 / W, 0.998),
                      handletextpad=0.4, columnspacing=1.3))
    letter(fig, 0.02, y0 + h + 0.22, "(a)")
    letter(fig, xl + 7 * u + gap - 0.20, y0 + h + 0.22, "(b)")
    save(fig, "invariance_real")


def fig_contour_graphs_real() -> None:
    """C3 (cycle-graph objection): real GT particles with their contour graphs, and the width-chord hop distribution
    against the n_nodes/8 line."""
    from wearpg.features.contour import canonicalize
    from wearpg.graphs import EDGE_CHORD, EDGE_INTERIOR, EDGE_RING1, GraphConfig, contour_graph
    from wearpg.open_data import mask_components, read_mask
    from wearpg.segmentation import load_image, to_gray

    hops = pd.read_csv(FER / "robust_chord_hops.csv")
    ch = jload(FER / "robust_summary.json")["chord_hop"]
    cfg = GraphConfig()
    check("graph n_nodes", cfg.n_nodes, jload(FER / "robust_summary.json")["graph_cfg_n_nodes"])
    for mode in ("mask", "pipeline"):
        h = hops[hops["mode"] == mode].chord_median_hop
        check(f"chord hop {mode} n", len(h), ch[mode]["n_particles"])
        check(f"chord hop {mode} median", h.median(), ch[mode]["median_of_median_hop"])
        check(f"chord hop {mode} share > n/8", (h > cfg.n_nodes / 8).mean(), ch[mode]["fraction_median_hop_gt_n_over_8"])
    df = pd.read_csv(FER / "particle_descriptors.csv.gz", usecols=["set", "image_id", "particle_id", "shape_area_um2"])
    cand = df[(df.set == "zenodo_gt") & (df.shape_area_um2 >= 600)].sample(8, random_state=0)
    idx = _zen_index()
    H, s_, gx, x0_, rows_y = 3.28, 1.02, 0.07, 0.30, (1.80, 0.46)
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)
    for k, r in enumerate(cand.itertuples()):
        xa, ya = x0_ + (k % 4) * (s_ + gx), rows_y[k // 4]
        a = axes_at(fig, xa, ya, s_, s_)
        img = load_image(idx.loc[r.image_id, "image_path"])
        comps = mask_components(read_mask(idx.loc[r.image_id, "mask_path"]), 30)
        xy = comps[int(r.particle_id.rsplit("_p", 1)[1])]["contour"]
        g = contour_graph(xy, 1.0, cfg, img)
        _, info = canonicalize(xy, cfg.n_nodes)
        c, sn = np.cos(info["angle"]), np.sin(info["angle"])
        p = info["centroid"] + info["scale"] * g["pos"].numpy() @ np.array([[c, -sn], [sn, c]]).T
        lo, hi = xy.min(0) - 5, xy.max(0) + 5
        side = min(max(hi - lo) / 2, min(img.shape[:2]) / 2)
        cen = np.clip((lo + hi) / 2, side - 0.5, np.array(img.shape[1::-1]) - 0.5 - side)  # window inside image
        a.imshow(to_gray(img), cmap="gray", vmin=-0.4, vmax=1.3, interpolation="none",
                 extent=(-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
        ei, kind = g["edge_index"].numpy(), g["edge_kind"].numpy()
        half = ei.shape[1] // 2
        for (i, j), kd in zip(ei[:, :half].T, kind[:half]):
            if kd & EDGE_INTERIOR:
                a.plot(*p[[i, j]].T, color=C["interior"], lw=0.5, zorder=2.5)
            elif kd & EDGE_CHORD:
                a.plot(*p[[i, j]].T, color=C["graph"], lw=0.5, zorder=2.2)
            elif kd & EDGE_RING1:
                a.plot(*p[[i, j]].T, color="black", lw=0.9, zorder=2.4)
        inner = g["node_type"].numpy() == 1
        a.scatter(*p[inner].T, s=7, color=C["interior"], edgecolors="black", linewidths=0.4, zorder=3)
        a.set_xlim(cen[0] - side, cen[0] + side)
        a.set_ylim(cen[1] + side, cen[1] - side)
        a.set_xticks([])
        a.set_yticks([])
        cv.text(xa + s_ / 2, ya + s_ + 0.03, f"{r.particle_id.replace('_', ' ')}\n{int(r.shape_area_um2)} px",
                ha="center", va="bottom", fontsize=FS_SMALL, linespacing=1.1)
    framed(fig.legend(handles=[Line2D([], [], color="black", lw=1.0, label="ring edge (hop 1)"),
                               Line2D([], [], color=C["graph"], lw=1.0, label="width chord"),
                               Line2D([], [], color=C["interior"], lw=1.0, marker="o", ms=3.6, mec="black", mew=0.4,
                                      label="interior node and edges")],
                      loc="lower center", ncol=3, bbox_to_anchor=((x0_ + 2 * s_ + 1.5 * gx) / W, 0.05 / H)))
    b = axes_at(fig, 5.28, rows_y[1], 1.82, rows_y[0] + s_ - rows_y[1])
    bins = np.arange(0, 65, 2)
    for mode, lab in (("mask", "ground-truth masks"), ("pipeline", "re-segmented")):
        h = hops[hops["mode"] == mode].chord_median_hop
        kw = (dict(color=C["neutral"], edgecolor="black", linewidth=0.5) if mode == "mask" else
              dict(color=C["ours"], histtype="step", linewidth=1.2, hatch="////"))
        edges = bins[(bins > h.min() - 2) & (bins < h.max() + 2)]  # no outline along the axis at empty bins
        b.hist(h, bins=edges, label=f"{lab}\n($n$ = {len(h)}, median {h.median():.0f})", **kw)
    b.axvline(cfg.n_nodes / 8, color="black", ls=DASH, lw=0.8)
    b.text(cfg.n_nodes / 8 - 1.2, 52, "reach of local edges,\n$n$/8 = 16 nodes", ha="right", va="center", rotation=90,
           fontsize=FS_SMALL, linespacing=1.15)
    b.set_xlim(0, 64)
    b.set_ylim(0, 118)
    b.set_yticks([0, 20, 40, 60, 80, 100])
    b.set_xlabel("Median chord hop distance\nper particle (boundary nodes)", linespacing=1.1)
    b.set_ylabel("Particles (count)")
    framed(b.legend(loc="upper right", handlelength=1.2, borderaxespad=0.35))
    letter(fig, 0.02, H - 0.10, "(a)")
    letter(fig, 4.74, H - 0.10, "(b)")
    save(fig, "contour_graphs_real")


MORPH_SETS = {"zenodo_gt": ("Zenodo ground-truth components (real)", C["true"], "-"),
              "zenodo_seg": ("Zenodo, our segments (real)", C["ours"], (0, (5, 1.2))),
              "tcbgy_seg": ("TCBGY single particles (real)", C["base"], DASHDOT),
              "synthetic_raster": ("Simulation (synthetic v1)", C["synth"], DOT)}


def fig_real_vs_synthetic_morphology() -> None:
    """Limitation: real vs synthetic particle morphology and the classifier domain gap per descriptor group."""
    cols = {"aspect": "shape_aspect_ratio", "roundness": "shape_roundness", "solidity": "shape_solidity"}
    df = pd.read_csv(FER / "particle_descriptors.csv.gz", usecols=["set", *cols.values()])
    ms = pd.read_csv(FER / "morph_summary.csv").set_index("set")
    ks = pd.read_csv(FER / "morph_ks.csv").set_index(["a", "b", "feature"])
    gap_ = pd.read_csv(FER / "domain_gap_auc.csv")
    for sname in MORPH_SETS:
        d = df[df.set == sname]
        check(f"morph {sname} n", len(d), ms.loc[sname, "n_particles"])
        for f, c in cols.items():
            check(f"morph {sname} median {f}", d[c].median(), ms.loc[sname, f"{f}_q50"], 1e-9)
    syn = df[df.set == "synthetic_raster"]
    for sname in ("zenodo_gt", "zenodo_seg", "tcbgy_seg"):
        for f, c in cols.items():
            check(f"morph KS {sname} vs synthetic {f}",
                  stats.ks_2samp(df[df.set == sname][c], syn[c]).statistic,
                  ks.loc[(sname, "synthetic_raster", f), "ks_D"], 1e-9)

    H, yt, ht, yb, hb, xl, gap = 4.50, 2.40, 1.50, 0.34, 1.38, 0.56, 0.16
    fig = plt.figure(figsize=(W, H))
    w3 = (7.10 - xl - 2 * gap) / 3
    ax = [axes_at(fig, xl, yt, w3, ht)]
    ax += [axes_at(fig, xl + i * (w3 + gap), yt, w3, ht, sharey=ax[0]) for i in (1, 2)]
    short = {"zenodo_gt": "ground truth", "zenodo_seg": "our segments", "tcbgy_seg": "TCBGY"}
    for a, (f, c), xl_ in zip(ax, cols.items(), ("Aspect ratio (log scale)", "Roundness 4π$A$/$P$$^2$",
                                                 "Solidity $A$/$A_\\mathrm{hull}$")):
        for sname, (lab, colr, ls_) in MORPH_SETS.items():
            v = np.sort(df[df.set == sname][c].to_numpy())
            a.plot(v, np.arange(1, len(v) + 1) / len(v), color=colr, ls=ls_, label=f"{lab}, $n$ = {len(v):,}",
                   lw=1.5 if sname == "synthetic_raster" else 1.2)
        a.set_xlabel(xl_)
        up = f == "solidity"
        a.text(0.04 if up else 0.965, 0.96 if up else 0.05, "KS $D$ vs synthetic\n" + "\n".join(
            f"{short[k]} {ks.loc[(k, 'synthetic_raster', f), 'ks_D']:.2f}" for k in short), transform=a.transAxes,
            ha="left" if up else "right", va="top" if up else "bottom", fontsize=FS_SMALL, linespacing=1.15)
    ax[0].set_ylim(0, 1.03)
    ax[0].set_xscale("log")
    plain_log(ax[0].xaxis, subs=(1.0, 2.0, 5.0))
    ax[0].set_ylabel("Cumulative fraction of particles")
    for a in ax[1:]:
        plt.setp(a.get_yticklabels(), visible=False)
    framed(fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, handlelength=2.4,
                      bbox_to_anchor=((xl + 7.10) / 2 / W, 0.998), columnspacing=1.8))
    # d: domain gap
    a = axes_at(fig, xl, yb, 7.10 - xl, hb)
    pairs = [("zenodo_gt", "synthetic_raster", "ground truth vs synthetic", C["true"], "o"),
             ("zenodo_seg", "synthetic_raster", "our segments vs synthetic", C["ours"], "s"),
             ("tcbgy_seg", "synthetic_raster", "TCBGY vs synthetic", C["base"], "^"),
             ("zenodo_gt", "tcbgy_seg", "ground truth vs TCBGY (real vs real)", "white", "D")]
    groups = [("texture", "texture"), ("colour", "colour"), ("efd", "EFD"), ("fractal", "fractal"),
              ("shape_scale_free", "shape (scale-free)"), ("size_px", "size (px)")]
    w = 0.19
    for k, (pa, pb, lab, c, mk) in enumerate(pairs):
        s_ = gap_[(gap_.a == pa) & (gap_.b == pb)].set_index("group")
        x = np.arange(len(groups)) + (k - 1.5) * w
        v = s_.loc[[g for g, _ in groups]]
        a.errorbar(x, v.auc, yerr=[v.auc - v.lo, v.hi - v.auc], fmt=mk, color=c, ecolor="black", ms=4.4, label=lab,
                   zorder=3, **EB)
    for v, st in ((0.5, DOT), (0.9, DASH)):
        a.axhline(v, color="black", lw=0.8 if st == DASH else 1.1, ls=st, zorder=1)
    a.text(-0.46, 0.89, "0.9: clearly separable (pre-specified)", ha="left", va="top", fontsize=FS_SMALL, bbox=MASK)
    a.text(len(groups) - 0.56, 0.512, "0.5: indistinguishable", ha="right", va="bottom", fontsize=FS_SMALL, bbox=MASK)
    a.set_xticks(range(len(groups)), [lab for _, lab in groups])
    categorical(a, "x")
    dotted_grid(a, "y")
    a.set_xlim(-0.5, len(groups) - 0.5)
    a.set_ylim(0.45, 1.04)
    a.set_ylabel("Cross-validated AUC")
    framed(a.legend(loc="lower left", bbox_to_anchor=(0.005, 0.12), labelspacing=0.3))
    for (x, y), s in zip([(0.02, yt + ht + 0.17)] + [(xl + i * (w3 + gap) - 0.14, yt + ht + 0.17) for i in (1, 2)]
                         + [(0.02, yb + hb + 0.17)], "abcd"):
        letter(fig, x, y, f"({s})")
    save(fig, "real_vs_synthetic_morphology")


# =============================================================================================== post-hoc diagnostics


def fig_diagnostics_posthoc() -> None:
    """F3 / F4 (post-hoc diagnostics) and F5 (secondary analysis) of docs/diagnostics.md; frozen verdicts unchanged."""
    lv = pd.read_csv(DIAG / "f3_per_image_levels.csv.gz")
    lv = lv[(lv.split == "testO") & (lv.method == "ours_tuned")]
    cnt = pd.read_csv(DIAG / "f3_testO_counts.csv")
    cnt = cnt[cnt.method == "ours_tuned"].set_index(["level", "metric"])
    s3 = jload(DIAG / "f3_summary.json")
    rho = s3["chosen_level_trainO"]["ours_tuned"]
    per = pd.read_csv(FER / "seg_per_image_test.csv")
    frozen = per[per.method == "ours_tuned"].set_index("image_id")
    none, merged = lv[lv.level == "none"].set_index("image_id"), lv[lv.level == rho].set_index("image_id")
    check("F3 level none reproduces frozen object counts", (none.n_pred == frozen.n_pred.reindex(none.index)).mean(),
          1.0)
    for lvl, d in (("none", none), (rho, merged)):
        check(f"F3 pooled count ratio level {lvl}", d.n_pred.sum() / d.n_gt.sum(),
              cnt.loc[(lvl, "pooled_count_ratio"), "point"])
        check(f"F3 mean object F1 level {lvl}", d.f1.mean(), cnt.loc[(lvl, "f1"), "point"])
    strata = pd.read_csv(DIAG / "f4_size_strata.csv")
    s4 = jload(DIAG / "f4_summary.json")
    sw = pd.read_csv(DIAG / "f4_min_size_sweep.csv")
    fmin = []
    for sc in (0.5, 0.25):  # F_min rule: smallest grid F with S >= 0.8 for every F' >= F (trainO, adapted)
        q = sw[(sw["mode"] == "pipeline_adapted") & (sw.split == "trainO") & (sw.scale == sc)].sort_values("F")
        ok = (q.S >= 0.8) | q.S.isna()
        tail = ok[::-1].cumprod()[::-1].astype(bool)
        fmin.append(q.F[tail].min())
    check("F4 F_min re-derived from the trainO sweep", max(fmin), s4["F_min"]["pipeline_adapted"]["F_min_trainO"])
    f5 = pd.read_csv(DIAG / "f5_graph_minus_efd.csv")
    s5 = jload(DIAG / "f5_summary.json")
    prim = f5[(f5["mode"] == "mask") & (f5.bootstrap_unit == "image_id")].set_index("transform")
    for row in s5["primary_family"]:
        check(f"F5 diff {row['transform']}", prim.loc[row["transform"], "diff"], row["diff"])
        check(f"F5 {row['transform']} = robust_stability medians",
              prim.loc[row["transform"], "S_graph_geom"] - prim.loc[row["transform"], "S_efd"], row["diff"], 1e-9)

    H = 4.50
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)
    a = axes_at(fig, 0.60, 2.56, 1.92, 1.68)
    rec = pd.read_csv(DIAG / "f3_testO_recall_by_gt_type.csv")
    rec = rec[(rec.method == "ours_tuned") & (rec.level == "none") & (rec.stratify == "type")].set_index("stratum")
    a.scatter(none.n_gt, none.n_pred, s=13, color=C["ours"], label="as segmented", **EDGE)
    a.scatter(merged.n_gt, merged.n_pred, s=13, color=C["ours_ext"], marker="s", **EDGE,
              label=f"merged\n(gaps ≤ {2 * int(rho)} px)")
    lim = [0.6, 2000]
    a.plot(lim, lim, color="black", lw=0.7, ls=DASH, zorder=1, label="1:1")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_xlabel("Ground-truth objects per image")
    a.set_ylabel("Segmented objects per image")
    framed(fig.legend(*a.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(2.60 / W, 4.24 / H),
                      handlelength=1.1, borderaxespad=0))
    f1n, f1m = cnt.loc[("none", "f1")], cnt.loc[(rho, "f1")]
    ratio = [fmt_ci(*cnt.loc[(k, "pooled_count_ratio"), ["point", "lo", "hi"]]) for k in ("none", rho)]
    cv.text(2.60, 3.62, f"post-hoc, {len(none)} test images\npooled count ratio\n{ratio[0]}\n→ {ratio[1]}\nobject F1\n"
            f"{num(f1n.point)} → {fmt_ci(f1m.point, f1m.lo, f1m.hi)}\nrecall (as segmented)\nsingle "
            f"{rec.loc['single_like', 'point']:.2f}, chain {rec.loc['chain_like', 'point']:.2f}", ha="left", va="top",
            fontsize=FS_SMALL, linespacing=1.2)
    labels = ["<16", "16–32", "32–64", "≥64"]
    variants = [("pipeline_frozen", "frozen segmentation", C["ours"], "o"),
                ("pipeline_adapted", "scale-adapted (post-hoc)", C["ours_ext"], "s"),
                ("mask", "ground-truth mask", C["true"], "^")]
    axb = []
    for yy, sc in zip((2.36, 0.56), (0.5, 0.25)):
        b = axes_at(fig, 4.44, yy, 2.66, 1.62, sharex=axb[0] if axb else None)
        axb.append(b)
        for k, (mode, lab, c, mk) in enumerate(variants):
            q = strata[(strata["mode"] == mode) & (strata.scale == sc)].sort_values("feret_reduced_lo")
            x = np.arange(len(q)) + (k - 1) * 0.2
            b.errorbar(x, q.point, yerr=[q.point - q.lo, q.hi - q.point], fmt=mk, color=c, ecolor="black", ms=4.4,
                       label=lab, zorder=3, **EB)
        v = s4["validation_testO"]["pipeline_adapted"]["testO"][str(sc)]
        b.errorbar(4.1, v["point"], yerr=[[v["point"] - v["lo"]], [v["hi"] - v["point"]]], fmt="s", color=C["ours_ext"],
                   ecolor="black", ms=4.4, zorder=3, **EB)
        b.text(4.1, v["lo"] - 0.03, f"$n$ = {v['n']}", ha="center", va="top", fontsize=FS_SMALL, bbox=MASK)
        b.axhline(0.8, color="black", ls=DASH, lw=0.7, zorder=1)
        b.axvline(3.55, color="black", lw=0.5, zorder=1)
        b.set_xlim(-0.5, 4.65)
        b.set_ylim(-0.04, 1.12)
        b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        b.set_ylabel("Graph-geometry $\\rho$ (median)")
        categorical(b, "x")
        dotted_grid(b, "y")
        b.text(0.025, 0.96, f"rescaled {sc:g}×", transform=b.transAxes, ha="left", va="top", fontsize=FS_NOTE,
               weight="bold", bbox=MASK)
        ad = s4["adapted_vs_frozen"][f"scale{sc}"]
        b.text(3.45, 0.02, f"adapted − frozen\n{fmt_ci(ad['diff'], ad['lo'], ad['hi'])}", ha="right", va="bottom",
               fontsize=FS_SMALL, linespacing=1.15, bbox=MASK)
    axb[1].set_xticks([0, 1, 2, 3, 4.1], labels + ["testO\n≥40"])
    plt.setp(axb[0].get_xticklabels(), visible=False)
    axb[1].set_xlabel("Feret diameter after rescaling (px)")
    framed(fig.legend(*axb[0].get_legend_handles_labels(), loc="lower right", ncol=3, handlelength=1.0,
                      bbox_to_anchor=(7.10 / W, 4.06 / H), handletextpad=0.3, columnspacing=0.8, borderaxespad=0))
    order = ["rot45", "rot_random", "rot90", "scale0.5", "scale0.25", "blur1", "blur2"]
    names = {"rot45": "rotation 45°", "rot_random": "rotation random", "rot90": "rotation 90°",
             "scale0.5": "rescale 0.5×", "scale0.25": "rescale 0.25×", "blur1": "blur 1 px", "blur2": "blur 2 px"}
    pipe = f5[(f5["mode"] == "pipeline") & (f5.bootstrap_unit == "image_id")].set_index("transform")
    c_ = axes_at(fig, 1.02, 0.46, 2.02, 1.32)
    for i, t in enumerate(order):
        for off, d, fc, mk, lab in ((-0.16, prim, C["graph"], "s", "ground-truth masks"),
                                    (0.16, pipe, "white", "D", "re-segmented")):
            if t not in d.index:
                continue
            q = d.loc[t]
            c_.errorbar(q["diff"], i + off, xerr=[[q["diff"] - q.lo], [q.hi - q["diff"]]], fmt=mk, color=C["graph"],
                        mfc=fc, ecolor="black", ms=3.8, zorder=3, label=lab if i == 0 else None, **EB)
        q = prim.loc[t]
        c_.text(1.03, i, f"$p$ = {sci(q.p_holm)}" if q.p_holm < 1 else "exact tie", transform=c_.get_yaxis_transform(),
                va="center", ha="left", fontsize=FS_SMALL)
    c_.axvline(0, color="black", lw=0.7, zorder=1)
    c_.set_yticks(range(len(order)), [names[t] for t in order])
    c_.set_ylim(len(order) - 0.4, -0.6)
    c_.set_xlim(-0.25, 0.65)
    c_.set_xlabel("Δ Spearman $\\rho$, graph geometry − EFD (secondary)")
    categorical(c_, "y")
    dotted_grid(c_, "x")
    framed(fig.legend(*c_.get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(2.03 / W, 1.84 / H),
                      handlelength=1.0, handletextpad=0.3, columnspacing=1.0, borderaxespad=0))
    for x, y, s in ((0.02, H - 0.04, "(a)"), (3.86, H - 0.04, "(b)"), (0.02, 2.06, "(c)")):
        letter(fig, x, y, s)
    save(fig, "diagnostics_posthoc")


# =============================================================================================== renderer (simulation)


def fig_renderer_v1_v2() -> None:
    """Simulation realism: real trainO ferrograms vs the same synthetic fields rendered by v1 and v2, and the
    pre-specified domain-gap targets before / after (met and unmet)."""
    ba = pd.read_csv(REND / "before_after.csv")
    for _, r in ba.dropna(subset=["met"]).iterrows():
        want = {"a": (r.after <= r.before - 0.15) or (r.after < 0.85), "b": r.after <= 0.25,
                "c": r.after <= r.before + 0.05}[r.target]
        check(f"renderer target {r.target} {r.pair} {r.measure} 'met' flag re-derived", bool(want),
              str(r.met) == "True")
    img = matplotlib.image.imread(REND / "fig_real_v1_v2.png")
    assert img.shape[:2] == (1152, 1536), img.shape  # 3 rows x 4 tiles of 384 px
    H, s_, gx, xg = 3.22, 0.80, 0.05, 0.34
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)
    rows = ["real\n(Zenodo trainO)", "Simulation, v1", "Simulation, v2"]
    for i in range(3):
        ya = 0.30 + (2 - i) * (s_ + 0.12)
        cv.text(xg - 0.06, ya + s_ / 2, rows[i], rotation=90, ha="right", va="center", fontsize=FS_SMALL,
                linespacing=1.1, multialignment="center")
        for j in range(4):
            a = axes_at(fig, xg + j * (s_ + gx), ya, s_, s_)
            a.imshow(img[i * 384:(i + 1) * 384, j * 384:(j + 1) * 384], interpolation="none")
            a.set_xticks([])
            a.set_yticks([])

    def pair(a, i, r, thr, right):
        """v1 -> v2 of one target with its limit and verdict (text, marker shape)."""
        met = str(r.met) == "True"
        if abs(r.after - r.before) > 0.02:
            a.annotate("", xy=(r.after, i), xytext=(r.before, i),
                       arrowprops=dict(arrowstyle="-|>", color=C["link"], lw=0.9, mutation_scale=7, shrinkA=2,
                                       shrinkB=2))
        a.plot(r.before, i, "o", mfc="white", mec="black", ms=4.8, mew=0.7, zorder=3)
        a.plot(r.after, i, "o" if met else "X", color=C["synth"], mec="black", mew=0.5, ms=4.8 if met else 5.4,
               zorder=4)
        a.plot([thr, thr], [i - 0.32, i + 0.32], color="black", lw=1.5, zorder=2)
        a.text(right, i, "met" if met else "not met", transform=a.get_yaxis_transform(), va="center", ha="left",
               fontsize=FS_SMALL, weight="bold" if met else "normal")

    prim = [("a", "P1", "AUC texture", "texture"), ("a", "P1", "AUC colour", "colour"),
            ("a", "P4", "AUC efd", "EFD, chains"), ("a", "P4", "AUC shape_scale_free", "shape, chains")]
    xr, wr = 4.94, 1.74
    b = axes_at(fig, xr, 2.02, wr, 0.92)
    for i, (t, p, meas, lab) in enumerate(prim):
        r = ba[(ba.target == t) & (ba.pair == p) & (ba.measure == meas)].iloc[0]
        pair(b, i, r, max(r.before - 0.15, 0.85), 1.03)
    b.set_yticks(range(len(prim)), [x[3] for x in prim])
    b.set_ylim(len(prim) - 0.45, -0.55)
    b.set_xlim(0.76, 1.03)
    b.set_xticks([0.8, 0.9, 1.0])
    b.set_xlabel("Classifier AUC, real vs synthetic")
    c_ = axes_at(fig, xr, 0.68, wr, 0.90)
    ks = [("b", "P4", "KS D aspect", "aspect, chains"), ("b", "P4", "KS D roundness", "roundness, chains"),
          ("b", "P4", "KS D solidity", "solidity, chains"), ("c", "P3", "KS D aspect", "aspect, singles"),
          ("c", "P3", "KS D roundness", "roundness, singles"), ("c", "P3", "KS D solidity", "solidity, singles")]
    for i, (t, p, meas, lab) in enumerate(ks):
        r = ba[(ba.target == t) & (ba.pair == p) & (ba.measure == meas)].iloc[0]
        pair(c_, i, r, 0.25 if t == "b" else r.before + 0.05, 1.03)
    c_.axhline(2.5, color="black", lw=0.5)
    c_.set_yticks(range(len(ks)), [x[3] for x in ks])
    c_.set_ylim(len(ks) - 0.45, -0.55)
    c_.set_xlim(0, 0.82)
    c_.set_xlabel("Two-sample KS $D$, real vs synthetic")
    for a in (b, c_):
        categorical(a, "y")
        dotted_grid(a, "x")
    framed(fig.legend(handles=[Line2D([], [], ls="", marker="o", mfc="white", mec="black", mew=0.7, ms=4.8, label="v1"),
                               Line2D([], [], ls="", marker="o", color=C["synth"], mec="black", mew=0.5, ms=4.8,
                                      label="v2, met"),
                               Line2D([], [], ls="", marker="X", color=C["synth"], mec="black", mew=0.5, ms=5.4,
                                      label="v2, not met"),
                               Line2D([], [], ls="", marker="|", color="black", ms=7, mew=1.5,
                                      label="pre-specified target")],
                      loc="lower right", ncol=4, bbox_to_anchor=(7.14 / W, 0.04 / H), handletextpad=0.2,
                      columnspacing=0.9, borderaxespad=0))
    letter(fig, 0.02, H - 0.04, "(a)")
    letter(fig, 3.86, H - 0.04, "(b)")
    letter(fig, 3.86, 1.78, "(c)")
    save(fig, "renderer_v1_v2")


# =============================================================================================== schematic


def connect(ax, pts, color="black", ls="-", lw=0.8) -> None:
    """Orthogonal connector through points (inches) with a solid arrow head at the last point."""
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    d = np.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    head = 0.061                                    # head length in inches: 0.55 x 8 pt
    xs, ys = zip(*(list(pts[:-1]) + [(x1 - 0.8 * head * ux, y1 - 0.8 * head * uy)]))
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_joinstyle="miter", solid_capstyle="butt", zorder=1)
    ax.annotate("", xy=(x1, y1), xytext=(x1 - head * ux, y1 - head * uy), zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.26", color=color, lw=lw, shrinkA=0,
                                shrinkB=0, mutation_scale=8))


def block(ax, x, y, w, h, title=None, body=(), ec="black", fc="white", ls="-", lw=0.7, tcolor="black", tsize=7.2,
          bsize=FS_SMALL, top=None, lead=1.3) -> None:
    """Block of a diagram: rectangle with a bold title and regular body lines, centred. top: inches from the top edge
    to the first line (default: the text block is centred vertically)."""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    lines = ([(t, True) for t in title.split("\n")] if title else []) + [(b, False) for b in body]
    hts = [(tsize if bold else bsize) / 72 * lead for _, bold in lines]
    gap = 0.03 if title and body else 0.0
    yy = y + h - top if top is not None else y + (h + sum(hts) + gap) / 2
    for i, ((s, bold), ht) in enumerate(zip(lines, hts)):
        ax.text(x + w / 2, yy - ht / 2, s, ha="center", va="center", fontsize=tsize if bold else bsize,
                weight="bold" if bold else "normal", color=tcolor if bold else "black", zorder=4)
        yy -= ht + (gap if bold and i + 1 < len(lines) and not lines[i + 1][1] else 0)


def frame_label(ax, x, y, bold, regular) -> None:
    """Group label ending at x (bold lead, regular rest)."""
    t = ax.text(x, y, regular, fontsize=FS_NOTE, ha="right", va="top", bbox=MASK)
    x0 = ax.transData.inverted().transform((t.get_window_extent(renderer=ax.figure.canvas.get_renderer()).x0, 0))[0]
    ax.text(x0 - 0.06, y, bold, fontsize=FS_NOTE, weight="bold", ha="right", va="top", bbox=MASK)


# =============================================================================================== simulation

SIM_MODELS = {  # label, colour, marker, marker face, line style
    "physics_only": ("physics only (debris)", C["ours"], "o", C["ours"], "-"),
    "conditions_only": ("conditions only", C["base"], "s", C["base"], DASHDOT),
    "physics_conditions": ("closed-form fusion", C["ours"], "D", C["ours_ext"], DASH),
    "wearpg|full": ("learned model", C["graph"], "^", C["graph"], "-")}
SIM_BANNER = "Simulation, not real data"


def fig_sim_stress() -> None:
    """S1: stress tests, one factor at a time (20 campaigns per setting)."""
    raw = pd.read_csv(SIM / "stress" / "stress_campaigns.csv")
    summ = pd.read_csv(SIM / "stress" / "stress_summary.csv")
    sys.path.insert(0, str(ROOT / "scripts"))
    from sim_summary import stress_summary  # the producing agent's aggregation, imported read-only
    re_ = stress_summary(raw)
    mrg = summ.merge(re_, on=["factor", "level", "model", "metric"], suffixes=("", "_re"))
    check("stress summary rows re-derived", len(mrg), len(summ))
    for col in ("mean", "q025", "q975"):
        check(f"stress summary {col} max abs diff", (mrg[col] - mrg[f"{col}_re"]).abs().max(), 0.0, 1e-12)
    factors = list(dict.fromkeys(summ.factor))
    flab = {"budget_sigma": "Debris budget\nnoise $\\sigma$", "imaged_fraction_range": "Imaged fraction\nrange",
            "material_sigma": "Capture spread $\\sigma$\n(material)", "min_area_px": "Min. detectable\narea (px)",
            "samples_per_run": "Samples\nper run"}
    mets = [("mae", "Run MAE of log$_{10}$ $k$"), ("within_spearman", "Within-condition\nSpearman $\\rho$"),
            ("ccc", "Volume CCC\n(debris vs true)")]
    ylim = {"mae": ((0.1, 0.62), (0.2, 0.4, 0.6)), "within_spearman": ((-0.8, 1.1), (-0.5, 0, 0.5, 1.0)),
            "ccc": ((0.77, 1.01), (0.8, 0.9, 1.0))}
    H, xl, gx, yb, h, gy = 4.31, 0.66, 0.10, 0.62, 0.98, 0.16
    w = (7.10 - xl - (len(factors) - 1) * gx) / len(factors)
    fig = plt.figure(figsize=(W, H))
    for j, f in enumerate(factors):
        levels = list(dict.fromkeys(summ[summ.factor == f].level))
        x = np.arange(len(levels))
        for i, (met, lab) in enumerate(mets):
            a = axes_at(fig, xl + j * (w + gx), yb + (len(mets) - 1 - i) * (h + gy), w, h)
            for k, mdl in enumerate(("physics_only", "conditions_only", "physics_conditions")):
                d = summ[(summ.factor == f) & (summ.model == mdl) & (summ.metric == met)].set_index("level")
                if d.empty:
                    continue
                d = d.reindex(levels)
                name, c, mk, fc, st = SIM_MODELS[mdl]
                a.errorbar(x + (k - 1) * 0.17, d["mean"], yerr=[d["mean"] - d.q025, d.q975 - d["mean"]], fmt=mk, ls=st,
                           color=c, mfc=fc, ms=3.8, lw=0.9, label=name, zorder=3, **EB)
            pil = [levels.index(lv) for lv in levels
                   if summ[(summ.factor == f) & (summ.level == lv)].pilot_level.any()]
            for p in pil:
                a.axvspan(p - 0.42, p + 0.42, color=C["neutral"], lw=0, zorder=0)
            if met == "within_spearman":
                a.axhline(0, color="black", lw=0.6, zorder=1)
            if met == "ccc":
                a.axhline(0.8, color="black", lw=0.7, ls=DASH, zorder=1)
            a.set_xlim(-0.55, len(levels) - 0.45)
            a.set_xticks(x, [lv.replace("[", "").replace("]", "").replace(", ", "\n–") for lv in levels],
                         linespacing=1.0)
            categorical(a, "x")
            a.set_ylim(*ylim[met][0])
            a.set_yticks(ylim[met][1])
            if j == 0:
                a.set_ylabel(lab, linespacing=1.1)
                letter(fig, 0.02, yb + (len(mets) - 1 - i) * (h + gy) + h + 0.13, f"({'abc'[i]})")
            else:
                a.set_yticklabels([])
            if i < len(mets) - 1:
                a.set_xticklabels([])
            else:
                fig.text((xl + j * (w + gx) + w / 2) / W, 0.04 / H, flab[f], fontsize=FS_LAB, ha="center", va="bottom",
                         linespacing=1.1)
            if i == j == 0:
                hd, lb = a.get_legend_handles_labels()
    framed(fig.legend(hd + [Patch(facecolor=C["neutral"], lw=0)], lb + ["pilot setting"], loc="upper right", ncol=4,
                      bbox_to_anchor=(7.10 / W, 1 - 0.03 / H), handlelength=2.2, columnspacing=1.3, borderaxespad=0))
    fig.text(xl / W, 1 - 0.085 / H, SIM_BANNER, fontsize=FS_NOTE, weight="bold", ha="left", va="top")
    save(fig, "sim_stress")


def fig_sim_power() -> None:
    """S1 design: probability that a campaign beats conditions-only (run-MAE CI below 0) vs design."""
    raw = pd.read_csv(SIM / "power" / "power_campaigns.csv")
    sys.path.insert(0, str(ROOT / "scripts"))
    from sim_summary import power_summary  # the producing agent's aggregation, imported read-only
    p = power_summary(raw)
    stored = pd.read_csv(SIM / "power" / "power_summary.csv")
    key = ["repeats", "n_conditions", "imaged_fraction_range", "model"]
    mrg = stored.merge(p, on=key, suffixes=("", "_re"))
    same_n = (mrg.n_campaigns == mrg.n_campaigns_re).all()
    if same_n:
        for col in ("p_mae_ci_below_0", "p_both", "p_mae_ci_below_0_lo", "p_mae_ci_below_0_hi"):
            check(f"power summary {col} max abs diff", (mrg[col] - mrg[f"{col}_re"]).abs().max(), 0.0, 1e-12)
    else:
        CHECKS.append(f"STALE power_summary.csv (n_campaigns {sorted(stored.n_campaigns.unique())} vs campaigns file "
                      f"{sorted(p.n_campaigns.unique())}): figure uses the re-derived summary of power_campaigns.csv")
    imfs = list(dict.fromkeys(p.sort_values("imaged_fraction_range", key=lambda s: s.map(
        lambda v: -json.loads(v)[1])).imaged_fraction_range))
    models = ["physics_only", "physics_conditions"]
    ncs = sorted(p.n_conditions.unique())
    reps = sorted(p.repeats.unique())
    cols = {6: ("^", DOT), 12: ("s", DASH), 18: ("o", "-")}  # marker, line style; 18 conditions: filled
    H, xl, gx, yb, h, gy = 3.65, 0.84, 0.12, 0.44, 1.26, 0.20
    w = (7.10 - xl - (len(imfs) - 1) * gx) / len(imfs)
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)
    for i, mdl in enumerate(models):
        ya = yb + (len(models) - 1 - i) * (h + gy)
        for j, imf in enumerate(imfs):
            a = axes_at(fig, xl + j * (w + gx), ya, w, h)
            for nc in ncs:
                d = p[(p.model == mdl) & (p.imaged_fraction_range == imf) & (p.n_conditions == nc)].sort_values(
                    "repeats")
                mk, st = cols[nc]
                a.errorbar(d.repeats + (ncs.index(nc) - 1) * 0.16, d.p_mae_ci_below_0,
                           yerr=[d.p_mae_ci_below_0 - d.p_mae_ci_below_0_lo,
                                 d.p_mae_ci_below_0_hi - d.p_mae_ci_below_0],
                           fmt=mk, ls=st, color=C["ours"], mfc=SIM_MODELS[mdl][3] if nc == 18 else "white", ms=4.2,
                           lw=1.0, zorder=3, **EB,
                           label=f"{nc} conditions ({nc * min(reps)}–{nc * max(reps)} runs)")
            a.axhline(0.8, color="black", ls=DASH, lw=0.7, zorder=1)
            a.set_xticks(reps)
            a.tick_params(axis="x", which="minor", bottom=False, top=False)
            a.set_xlim(min(reps) - 0.6, max(reps) + 0.6)
            a.set_ylim(-0.04, 1.06)
            a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            lo, hi = json.loads(imf)
            if i == 0:
                a.set_title(f"imaged fraction {lo:g}–{hi:g}")
                a.set_xticklabels([])
            else:
                a.set_xlabel("Runs per condition")
            if j == 0:
                a.set_ylabel("$P$ (beats conditions only)")
                cv.text(0.17, ya + h / 2, SIM_MODELS[mdl][0].replace(" (debris)", ""), rotation=90, ha="center",
                        va="center", fontsize=FS_NOTE, weight="bold")
                letter(fig, 0.02, ya + h + 0.17, f"({'ab'[i]})")
            else:
                a.set_yticklabels([])
            if i == j == 0:
                hd, lb = a.get_legend_handles_labels()
    framed(fig.legend(hd, lb, loc="upper right", ncol=3, bbox_to_anchor=(7.10 / W, 1 - 0.03 / H), handlelength=2.4,
                      columnspacing=1.4, borderaxespad=0))
    fig.text(xl / W, 1 - 0.085 / H, SIM_BANNER, fontsize=FS_NOTE, weight="bold", ha="left", va="top")
    save(fig, "sim_power")


def _pilot_runs(root: Path) -> pd.DataFrame:
    files = sorted((root / "predictions" / "parts").glob("runs__cv__*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[d.split == "test"].copy()
    d["key"] = np.where(d.model == "wearpg", "wearpg|" + d.variant, d.model)
    return d


def fig_sim_pilot_rate(root: Path | None = None) -> None:
    """S1: predicted vs true run log10 specific wear rate (held-out runs, protocol cv) per estimator."""
    from wearpg.metrics import regression_metrics
    root = root or PILOT
    d = _pilot_runs(root)
    reg = pd.read_csv(root / "metrics" / "regression.csv")
    reg = reg[(reg.protocol == "cv") & (reg.level == "run")]
    keys = [k for k in ("wearpg|full", "physics_conditions", "physics_only", "conditions_only") if k in set(d.key)]
    H, xl, gx, y0 = 2.38, 0.70, 0.13, 0.40
    s_ = (7.10 - xl - 3 * gx) / 4
    fig = plt.figure(figsize=(W, H))
    ax = []
    for j, (k, l) in enumerate(zip(keys, "abcd")):
        a = axes_at(fig, xl + j * (s_ + gx), y0, s_, s_, sharey=ax[0] if ax else None)
        ax.append(a)
        q = d[(d.key == k) & np.isfinite(d.log10_rate_true) & np.isfinite(d.log10_rate_pred)]
        mdl, var = (k.split("|") + ["default"])[:2] if "|" in k else (k, "default")
        r = reg[(reg.model == mdl) & (reg.variant == var)].set_index("metric")
        per_seed = [regression_metrics(g.log10_rate_true, g.log10_rate_pred, g.condition_id) for _, g in q.groupby("seed")]
        check(f"pilot {k} seed-mean run MAE", np.mean([x["mae"] for x in per_seed]), r.loc["mae", "seed_mean"], 1e-9)
        check(f"pilot {k} seed-mean within-condition Spearman",
              np.mean([x["within_condition_spearman"] for x in per_seed]),
              r.loc["within_condition_spearman", "seed_mean"], 1e-9)
        s0 = q[q.seed == q.seed.min()]
        name, c, mk, fc, _ = SIM_MODELS[k]
        a.scatter(s0.log10_rate_true, s0.log10_rate_pred, s=14, color=fc, marker=mk, zorder=3, **EDGE)
        lim = [min(d.log10_rate_true.min(), d.log10_rate_pred.min()) - 0.1,
               max(d.log10_rate_true.max(), d.log10_rate_pred.max()) + 0.1]
        a.plot(lim, lim, color="black", lw=0.7, ls=DASH, zorder=1)
        a.set_xlim(*lim)
        a.set_ylim(*lim)
        a.set_xticks([-7, -6, -5, -4])
        a.set_yticks([-7, -6, -5, -4])
        mae, ws = r.loc["mae"], r.loc["within_condition_spearman"]
        a.text(0.04, 0.96, f"MAE {fmt_ci(mae.point, mae.lo, mae.hi)}\nwithin-condition $\\rho$\n"
               f"{fmt_ci(ws.point, ws.lo, ws.hi)}", transform=a.transAxes, ha="left", va="top", fontsize=FS_SMALL,
               linespacing=1.15)
        a.set_title(name)
        letter(fig, xl + j * (s_ + gx) - (0.68 if j == 0 else 0.16), y0 + s_ + 0.19, f"({l})")
        if j:
            plt.setp(a.get_yticklabels(), visible=False)
    ax[0].set_ylabel("Predicted run log$_{10}$ $k$\n(mm$^3$ N$^{-1}$ m$^{-1}$)", linespacing=1.1)
    fig.text((xl + 7.10) / 2 / W, 0.04 / H, "True run log$_{10}$ $k$ (mm$^3$ N$^{-1}$ m$^{-1}$)", fontsize=FS_LAB,
             ha="center", va="bottom")
    fig.text(xl / W, 1 - 0.05 / H, SIM_BANNER, fontsize=FS_NOTE, weight="bold", ha="left", va="top")
    save(fig, "sim_pilot_rate")


def fig_sim_pilot_volume(root: Path | None = None) -> None:
    """S1: debris-implied vs true run volume (volume consistency) and within-condition tracking."""
    from wearpg.metrics import volume_consistency
    root = root or PILOT
    d = _pilot_runs(root)
    vc = pd.read_csv(root / "metrics" / "volume_consistency.csv")
    vc = vc[vc.protocol == "cv"]
    vkeys = [k for k in ("wearpg|full", "physics_only") if k in set(d.key)]
    tkeys = [k for k in ("wearpg|full", "physics_conditions", "physics_only", "conditions_only") if k in set(d.key)]
    H, y0, s_ = 2.77, 0.46, 1.82
    fig = plt.figure(figsize=(W, H))
    xs = (0.52, 2.52)
    for j, k in enumerate(vkeys):
        a = axes_at(fig, xs[j], y0, s_, s_)
        q = d[d.key == k]
        mdl, var = k.split("|") if "|" in k else (k, "default")
        row = vc[(vc.model == mdl) & (vc.variant == var)].iloc[0]
        per_seed = [volume_consistency(g.log_debris_volume_mm3, g.log_grav_volume_mm3) for _, g in q.groupby("seed")]
        for met in ("ccc", "slope"):
            check(f"pilot {k} volume {met} (seed mean)", np.mean([x[met] for x in per_seed]), row[met], 1e-9)
        s0 = q[(q.seed == q.seed.min()) & np.isfinite(q.log_debris_volume_mm3)]
        name, c, mk, fc, _ = SIM_MODELS[k]
        a.scatter(s0.log_grav_volume_mm3 / np.log(10), s0.log_debris_volume_mm3 / np.log(10), s=15, color=fc, marker=mk,
                  zorder=3, **EDGE)
        lim = np.array([s0.log_grav_volume_mm3.min(), s0.log_grav_volume_mm3.max()]) / np.log(10) + [-0.3, 0.3]
        a.plot(lim, lim, color="black", lw=0.7, ls=DASH, zorder=1)
        a.set_xlim(*lim)
        a.set_ylim(*lim)
        a.set_xticks([-2, -1, 0, 1, 2])
        a.set_yticks([-2, -1, 0, 1, 2])
        a.text(0.04, 0.96, f"CCC {row.ccc:.3f}\nslope {row.slope:.3f}\n(seed means, $n$ = {int(row.n)})",
               transform=a.transAxes, ha="left", va="top", fontsize=FS_SMALL, linespacing=1.15)
        a.set_title(name)
        a.set_xlabel("True run log$_{10}$ $V$ (mm$^3$)")
        if j == 0:
            a.set_ylabel("Debris-implied log$_{10}$ $V$ (mm$^3$)")
        else:
            plt.setp(a.get_yticklabels(), visible=False)
        letter(fig, xs[j] - (0.50 if j == 0 else 0.16), y0 + s_ + 0.19, f"({'ab'[j]})")
    a = axes_at(fig, 5.02, y0, 2.08, s_)
    for k in tkeys:
        q = d[(d.key == k) & (d.seed == d.seed.min())].dropna(subset=["log10_rate_true", "log10_rate_pred"])
        q = q[q.groupby("condition_id").run_id.transform("size") >= 2]
        ct = q.log10_rate_true - q.groupby("condition_id").log10_rate_true.transform("mean")
        cp = q.log10_rate_pred - q.groupby("condition_id").log10_rate_pred.transform("mean")
        name, c, mk, fc, _ = SIM_MODELS[k]
        a.scatter(ct, cp, s=13, color=fc, marker=mk, label=name, **EDGE)
    a.axhline(0, color="black", lw=0.6, zorder=0.5)
    a.axvline(0, color="black", lw=0.6, zorder=0.5)
    a.set_xlabel("True log$_{10}$ $k$ − condition mean")
    a.set_ylabel("Predicted log$_{10}$ $k$ − condition mean")
    letter(fig, 4.52, y0 + s_ + 0.19, f"({'abc'[len(vkeys)]})")
    framed(fig.legend(*a.get_legend_handles_labels(), loc="upper right", ncol=4,
                      bbox_to_anchor=(7.10 / W, 1 - 0.03 / H), handlelength=1.0, handletextpad=0.3, columnspacing=1.1,
                      borderaxespad=0))
    fig.text(xs[0] / W, 1 - 0.085 / H, SIM_BANNER, fontsize=FS_NOTE, weight="bold", ha="left", va="top")
    save(fig, "sim_pilot_volume")


# ================================================================================= published paired debris/wear data

PAIRED = ROOT / "outputs" / "open_data" / "paired"
DATA_PAIRED = ROOT / "data" / "real" / "paired"
PAIR_SETS = [  # summary key, label, inferential, marker
    ("feup_martins2008/fzg", "FZG gears (14)", True), ("feup_martins2008/fourball", "four-ball, oil (6)", True),
    ("phm_raadnui2015/dry_fourball", "four-ball, dry (5)", True), ("soton_craig2010/bearings", "bearings (6)", True),
    ("uct_jordan1995/grease", "grease, fretting (7)", True), ("phm_bolander2009/rig_tests", "bearing rig, relative (7)", False),
    ("jsme_higashi2003/plate_intervals", "one run, 8 intervals", False)]
SCUFFED = {"CM_pl", "E1_pI"}
ADI = {"AM_mp", "AE_mp", "AM_pl", "AE_pl"}


def _paired():
    s = pd.read_csv(PAIRED / "summary.csv")
    s["key"] = s.study + "/" + s.set
    u = pd.read_csv(PAIRED / "units.csv")
    u["key"] = u.study + "/" + u.set
    return s, s[s.analysis == "primary"].set_index("key"), u


POOL = PAIRED / "pooled"
PUB = {  # publications of the FZG gear pairs: label, marker, colour (thesis pairs open)
    "feup_martins2008": ("thesis, 14 pairs", "o", "white"), "jet495": ("carburised, micropitting", "o", C["ours"]), "tribint2009": ("nitrided, micropitting", "s", C["base"]),
    "ilt2011": ("carburised, interrupted", "^", C["graph"]), "pij2011": ("carburised, power loss", "D", C["early"])}
PAIR_SETS_SHOWN = [  # (summary key, label, inferential); the FZG thesis set comes in two rows (23 and 14 pairs) from the pooled file
    ("feup_martins2008/fourball", "four-ball, oil (6)", True), ("phm_raadnui2015/dry_fourball", "four-ball, dry (5)", True), ("soton_craig2010/bearings", "bearings (6)", True),
    ("uct_jordan1995/grease", "grease, fretting (7)", True), ("phm_bolander2009/rig_tests", "bearing rig, relative (7)", False), ("jsme_higashi2003/plate_intervals", "one run, 8 intervals", False)]


def _pooled():
    """Pooled results, the 44 rows and the 23 pair means (log10) of the post-hoc program."""
    return jload(POOL / "results.json"), pd.read_csv(POOL / "rows.csv"), pd.read_csv(POOL / "pairs.csv")


def fig_observation_chain() -> None:
    """Terms between the wear of a test and a debris reading, and the evidence of this article for the calibrated factor.
    Every number printed in the drawing is read from the stored results."""
    r, _, _ = _pooled()
    p23, pt, dec = r["pooled"]["23 pairs, as run"], r["pooled"]["23 pairs, run totals (as run)"], r["decision_table"]["D0 as run"]
    pg = r["two_groups"]["23 pairs, run totals (as run)"]["fold_constant_per_group"]
    sg = r["signed_errors"]["as run"]["debris"]["factor"]
    idt = jload(PAIRED / "extra" / "review.json")["identity"]["per_set"]
    g = sorted(idt[k]["gsd_factor"] for k in ("feup_martins2008/fourball", "phm_raadnui2015/dry_fourball", "soton_craig2010/bearings", "uct_jordan1995/grease"))
    nasa = jload(OIL / "oil_results.json")["nasa"]
    auc_ = nasa["auc_destructive_vs_not"]["mass_end_mg"]["auc"]
    ccc = jload(FER / "seg_summary_test.json")["ours_tuned"]["ccc_log_area"]["point"]
    check("schematic frozen factor error", dec["fold"]["debris"], jload(PAIRED / "holdout" / "results.json")["fold"]["debris"], 1e-9)
    check("schematic pooled constant error", pt["fold_loo"]["constant"], 2.325, 2e-3)
    check("schematic onset AUC", auc_, 0.9929, 1e-3)
    H = 2.72
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)
    for y, h, lead, rest in ((1.42, 1.27, "Terms:", "from the wear of a test to a debris reading"), (0.03, 1.32, "Evidence:", "for one calibrated factor per rig and instrument")):
        cv.add_patch(Rectangle((0.03, y), W - 0.06, h, fc="none", ec=C["base"], lw=0.8, zorder=1))
        frame_label(cv, 7.08, y + h - 0.05, lead, rest)
    stages = [("Wear $W$", ("weighed mass loss", "of both bodies")), ("Released", ("to the oil; not settled,", "not filtered, not held")), ("Sampled", ("share of the oil;", "interval or inventory")),
              ("Captured", ("magnetic recovery,", "dilution")), ("Read", ("index CPUC =", "$(D_L + D_S)/d$"))]
    x0, w, gap, yb, h = 0.15, 1.22, 0.175, 1.67, 0.70
    xs = [x0 + i * (w + gap) for i in range(5)]
    for i, (x, (t, d)) in enumerate(zip(xs, stages)):
        block(cv, x, yb, w, h, t, d, ec="black" if i in (0, 4) else C["base"], lw=0.9 if i in (0, 4) else 0.7)
        if i:
            connect(cv, [(xs[i - 1] + w, yb + h / 2), (x, yb + h / 2)])
    cv.text(xs[1] + (xs[4] - xs[1]) / 2 - 0.1, yb - 0.10, "not recorded separately: one calibrated factor, $D = f\\,\\eta\\,\\kappa\\,W$", ha="center", va="center", fontsize=FS_SMALL, zorder=4)

    def ev(x, w_, title, body, **kw):
        block(cv, x, 0.12, w_, 0.88, title, body, **kw)

    ev(0.13, 1.74, "Planned test", ("factor frozen on 14 gear pairs,", f"nine unseen pairs, run once:", f"error {dec['fold']['debris']:.1f}× (planned);", f"signed error {sg:.1f}× (post hoc);", "planned rule: no-go as run"))
    ev(1.98, 1.74, "23 gear pairs (post hoc)", (f"slope {pt['slope']:.2f}, rank corr. {pt['rho']:.2f};", f"scatter GSD {pt['gsd_D_over_W']:.1f}; error {pt['fold_loo']['debris']:.2f}×", f"(constant {pt['fold_loo']['constant']:.2f}×, one per", f"group of tests {pg:.2f}×;", "whole losses)"))
    ev(3.83, 1.60, "Other contacts (post hoc)", ("four-ball, bearings, grease:", f"GSD {g[0]:.1f}–{g[-1]:.1f}, no significant", "gain in these four sets", "(1.2, 1.5 in two bearing", "sets, descriptive)"))
    ev(5.54, 1.54, "Without measured wear", (f"NASA gears: pitted or not", f"(AUC {auc_:.2f}); how much is", "undecided;", f"ferrogram area: CCC {ccc:.2f}"))
    save(fig, "observation_chain")


def fig_graphical_abstract() -> None:
    """Graphical abstract (Wear: at least 1328 px wide, longest side horizontal, readable at 13 cm, so no text below 9 pt
    at this 7.2-in width): the 23 gear pairs with one factor, and the three findings. Every number from the stored results."""
    r, rows, _ = _pooled()
    pt, dec = r["pooled"]["23 pairs, run totals (as run)"], r["decision_table"]["D0 as run"]
    cv_ = r["cumulative_vs_interval"]["7 new pairs, as run"]
    tot = rows.groupby(["study", "unit"], sort=False)[["D0", "wear"]].sum().reset_index()     # whole loss of each test
    check("graphical abstract factor", 10 ** (np.log10(tot.wear) - np.log10(tot.D0)).mean(), pt["factor_W_over_D"], 1e-6)
    fig = plt.figure(figsize=(W, 3.0))
    a = axes_at(fig, 0.78, 0.62, 2.45, 2.20)
    xx = np.array([40.0, 14000.0])
    f, rg = pt["factor_W_over_D"], pt["range95_factor"]
    a.fill_between(xx, f * xx / rg, f * xx * rg, color=C["neutral"], alpha=0.55, lw=0, zorder=0, label=f"95 % range (×/÷ {rg:.1f})")
    a.plot(xx, f * xx, color="black", lw=1.1, ls=DASH, zorder=2, label="one factor")
    th = (tot.study == "feup_martins2008").to_numpy()
    a.scatter(tot.D0[th], tot.wear[th], s=30, marker="o", color="white", edgecolors="black", linewidths=0.7, zorder=4, label="14 pairs: factor set")
    a.scatter(tot.D0[~th], tot.wear[~th], s=30, marker="o", color=C["ours"], edgecolors="black", linewidths=0.7, zorder=4, label="9 unseen pairs")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(40, 14000)
    a.set_ylim(3, 3000)
    a.set_xlabel("Ferrograph index CPUC", fontsize=9.5)
    a.set_ylabel("Gear mass loss (mg)", fontsize=9.5)
    a.tick_params(labelsize=9)
    framed(a.legend(loc="upper left", fontsize=9, handlelength=1.4, borderpad=0.3, labelspacing=0.2))
    cv = canvas(fig)
    items = (("Planned test, 9 unseen FZG gear pairs", (f"error ×{dec['fold']['debris']:.1f} (frozen constant ×{dec['fold']['constant']:.1f});", "a constant refitted in them did as well:", "no-go by the rule of the plan")),
             ("23 pairs, one laboratory (post hoc)", (f"scatter GSD {pt['gsd_D_over_W']:.1f}; error ×{pt['fold_loo']['debris']:.2f} against", f"×{pt['fold_loo']['constant']:.2f} for one constant; gain = two groups", "of tests and three extreme tests")),
             ("Cumulative debris–wear plots", (f"R² {cv_['mean_r2_cumulative_per_pair']:.2f}, and {cv_['null_within_pair']['mean_r2']:.2f} with shuffled intervals:", "no test of a conversion")))
    y = 2.80
    for head, body in items:
        cv.text(3.62, y, head, fontsize=9.5, fontweight="bold", va="top", ha="left")
        for k, line in enumerate(body):
            cv.text(3.62, y - 0.20 - 0.18 * k, line, fontsize=9, va="top", ha="left")
        y -= 0.24 + 0.18 * len(body) + 0.14
    save(fig, "graphical_abstract")


def fig_paired_pooled() -> None:
    """23 FZG gear pairs: ferrograph index against weighed mass loss with one factor, and the leave-one-pair-out error of every predictor."""
    r, rows, pairs = _pooled()
    P = r["pooled"]
    a0 = P["23 pairs, run totals (as run)"]
    tot = rows.groupby(["study", "unit"], sort=False)[["D0", "D1", "wear"]].sum().apply(np.log10).reset_index()     # whole loss of a test: sums over its weighed stages
    x, y = tot.D0, tot.wear
    fit = stats.linregress(x, y)
    check("pooled slope", fit.slope, a0["slope"], 1e-6)
    check("pooled Spearman", stats.spearmanr(x, y)[0], a0["rho"], 1e-6)
    check("pooled GSD of D/W", 10 ** (x - y).std(ddof=1), a0["gsd_D_over_W"], 1e-6)
    check("pooled factor", 10 ** (y - x).mean(), a0["factor_W_over_D"], 1e-6)
    check("pooled constant error (leave one pair out)", np.log10(a0["fold_loo"]["constant"]), np.mean([abs(v - y.drop(i).mean()) for i, v in y.items()]), 1e-6)
    H, y0, h = 3.55, 0.48, 2.42
    fig = plt.figure(figsize=(W, H))
    a = axes_at(fig, 0.62, y0, 3.15, h)
    xx = np.array([25.0, 14000.0])
    f, s, rg = a0["factor_W_over_D"], a0["gsd_D_over_W"], a0["range95_factor"]
    a.fill_between(xx, f * xx / rg, f * xx * rg, color=C["neutral"], alpha=0.55, lw=0, zorder=0, label=f"×/÷ {rg:.1f} (95 %)")
    a.fill_between(xx, f * xx / s, f * xx * s, color="#BDBDBD", alpha=0.75, lw=0, zorder=1, label=f"×/÷ {s:.2f} (1 SD)")
    a.plot(xx, f * xx, color="black", lw=1.0, ls=DASH, zorder=2, label="one factor, slope 1")
    for st, (lab, mk, col) in PUB.items():
        q = tot[tot.study == st]
        a.scatter(10 ** q.D0, 10 ** q.wear, s=22, marker=mk, color=col, zorder=4, label=lab, edgecolors="black", linewidths=0.6)
    for _, q in tot[(tot.D0 - tot.D1).abs() > 1e-9].iterrows():
        a.plot([10 ** q.D1, 10 ** q.D0], [10 ** q.wear] * 2, color=C["link"], lw=0.7, zorder=3)
        a.plot(10 ** q.D1, 10 ** q.wear, "|", color="black", ms=4.5, zorder=3)
    sc = tot[tot.unit.isin(["feup_martins2008:CM_pl", "feup_martins2008:E1_pI"])]
    a.scatter(10 ** sc.D0, 10 ** sc.wear, s=78, marker="o", facecolors="none", edgecolors="black", linewidths=0.8, zorder=5)
    for dx, dy, u, t in ((1.22, 0.82, "feup_martins2008:CM_pl", "scuffed"), (1.22, 0.80, "feup_martins2008:E1_pI", "scuffed"), (1.22, 0.80, "feup_martins2008:E2_pI", "7 mg")):
        q = tot[tot.unit == u].iloc[0]
        a.text(10 ** q.D0 * dx if t == "7 mg" else 10 ** q.D0 * 0.60, 10 ** q.wear * dy if t == "7 mg" else 10 ** q.wear * 1.30, t, fontsize=FS_MIN, ha="left" if t == "7 mg" else "right", va="center", zorder=6)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(25, 14000)
    a.set_ylim(2, 2500)
    a.set_xlabel("Debris index CPUC of the test oil (whole test)")
    a.set_ylabel("Weighed gear mass loss of the test (mg)")
    a.text(0.975, 0.03, f"slope {fmt_ci(a0['slope'], *a0['perm_ci'])}\nSpearman {fmt_ci(a0['rho'], *a0['rho_ci'])}\n23 gear pairs", transform=a.transAxes, ha="right", va="bottom", fontsize=FS_SMALL, linespacing=1.2)
    framed(a.legend(loc="upper left", fontsize=FS_MIN, handlelength=1.6, borderpad=0.3, labelspacing=0.22))

    b = axes_at(fig, 5.35, y0, 1.75, h)
    G = r["two_groups"]
    sets = (("23 pairs, run totals (as run)", "23, whole loss", "23 pairs, run totals (as run)"), ("23 pairs, as run", "23, stages", "23 pairs, stage means (as run)"),
            ("23 pairs, K6 sample", "23, stages, K6 sample", "23 pairs, stage means (K6 sample)"), ("20 pairs, run totals (as run), without the three extreme pairs", "20, whole loss", "20 pairs, run totals (as run)"),
            ("23 pairs, as run, without the three extreme pairs", "20, stages", "20 pairs, stage means (as run)"), ("14 thesis pairs, run totals", "14 thesis, whole loss", None), ("14 thesis pairs", "14 thesis, stages", None),
            ("9 new pairs, run totals (as run)", "9 later, whole loss", None), ("9 new pairs, as run", "9 later, stages", None))
    ms = (("constant", "s", C["base"], C["base"], "constant"), ("group", "D", C["base_dark"], "white", "constant per group"), ("conditions", "^", C["base"], "white", "conditions"), ("archard", "v", C["base_dark"], C["base_dark"], "Archard"), ("debris", "o", C["ours"], C["ours"], "debris"))
    for i, (key, lab, gk) in enumerate(sets):
        fo = dict(P[key]["fold_loo"])
        fo["group"] = G[gk]["fold_constant_per_group"] if gk else np.nan
        for j, (k, mk, col, fc, name) in enumerate(ms):
            if np.isfinite(fo[k]):
                b.plot(fo[k], i + (j - 2) * 0.14, mk, color=col, mfc=fc, mec="black" if mk != "D" else col, mew=0.4 if mk != "D" else 1.0, ms=4.6, zorder=3, label=name if i == 0 else None)
    check("pooled figure group constant", G["23 pairs, run totals (as run)"]["fold_constant_per_group"], 2.047, 2e-3)
    b.set_yticks(range(len(sets)), [lab for _, lab, _ in sets])
    b.set_ylim(len(sets) - 0.45, -0.6)
    b.set_xlim(1.2, 3.6)
    b.set_xticks([1.5, 2, 2.5, 3, 3.5], ["1.5", "2", "2.5", "3", "3.5"])
    b.set_xlabel("Leave-one-pair-out error (factor)")
    categorical(b, "y")
    dotted_grid(b, "x")
    framed(b.legend(loc="lower right", bbox_to_anchor=(1.04, 1.02), ncol=2, fontsize=FS_MIN, columnspacing=0.6, handletextpad=0.1))
    for x_, s_ in ((0.02, "(a)"), (4.00, "(b)")):
        letter(fig, x_, y0 + h + 0.24, s_)
    save(fig, "paired_pooled")


def fig_paired_holdout() -> None:
    """Frozen factor on nine unseen gear pairs: signed error of every frozen predictor, and the error of every pair."""
    res = jload(PAIRED / "holdout" / "results.json")
    pr = pd.read_csv(PAIRED / "holdout" / "predictions.csv")
    r, _, _ = _pooled()
    se = r["signed_errors"]["as run"]
    check("holdout number of pairs", pr.unit.nunique(), res["n_units"], 1e-9)
    pr["study"] = pr.unit.str.split(":").str[0]
    err = {k: (pr[f"pred_log10_{k}"] - np.log10(pr.wear)).abs().groupby(pr.unit, sort=False).mean() for k in ("debris", "constant")}
    sgn = {k: (pr[f"pred_log10_{k}"] - np.log10(pr.wear)).groupby(pr.unit, sort=False).mean() for k in ("debris", "fused", "conditions", "archard", "constant")}
    lw = np.log10(pr.wear.to_numpy())                                   # constant refitted in the new data, leave one pair out (rows as fitting cases)
    refit = {u: float(np.abs(lw[(pr.unit == u).to_numpy()] - lw[(pr.unit != u).to_numpy()].mean()).mean()) for u in pr.unit.unique()}
    h1c = res["H1c_debris_vs_refitted_constant"]["diff"]
    check("holdout refitted constant", np.mean([refit[u] for u in err["debris"].index]) - err["debris"].mean(), -h1c, 1e-6)
    check("holdout mean error of debris", 10 ** err["debris"].mean(), res["fold"]["debris"], 1e-6)
    check("holdout mean error of constant", 10 ** err["constant"].mean(), res["fold"]["constant"], 1e-6)
    for k in sgn:
        check(f"holdout signed error {k}", 10 ** sgn[k].mean(), se[k]["factor"], 1e-6)
    H, y0, h = 3.30, 0.48, 2.15
    fig = plt.figure(figsize=(W, H))
    a = axes_at(fig, 1.30, y0, 2.35, h)
    names = {"debris": "debris, one factor", "fused": "debris + conditions", "conditions": "conditions", "archard": "Archard", "constant": "constant (71 mg)"}
    rng = np.random.default_rng(1)
    for i, (k, lab) in enumerate(names.items()):
        for u, v in sgn[k].items():
            st = u.split(":")[0]
            a.plot(10 ** v, i + rng.uniform(-0.20, 0.20), PUB[st][1], ms=4.0, color=PUB[st][2], mec="black", mew=0.35, zorder=3, label=PUB[st][0] if (i == 0 and u in {"ilt2011:E2", "jet495:M1", "pij2011:P1", "tribint2009:M1"}) else None)
        f, (lo, hi) = se[k]["factor"], se[k]["ci"]
        a.errorbar(f, i + 0.34, xerr=[[f - lo], [hi - f]], fmt="D", ms=4.2, color="black", mfc="white", mec="black", ecolor="black", elinewidth=0.9, capsize=2.2, zorder=4)
        a.text(0.985, i + 0.34, f"{f:.1f}", transform=a.get_yaxis_transform(), ha="right", va="center", fontsize=FS_SMALL, bbox=MASK)
    a.axvline(1, color="black", ls=DASH, lw=0.8, zorder=1)
    a.set_xscale("log")
    a.set_xlim(0.2, 12)
    plain_log(a.xaxis, (1.0, 2.0, 5.0))
    a.set_yticks(range(len(names)), list(names.values()))
    a.set_ylim(len(names) - 0.42, -0.55)
    a.set_xlabel("Predicted over measured mass loss (pair means)")
    categorical(a, "y")
    dotted_grid(a, "x")
    framed(a.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, fontsize=FS_MIN, columnspacing=0.6, handletextpad=0.1))

    c = axes_at(fig, 4.95, y0, 2.15, h)
    order = err["debris"].sort_values().index
    for i, u in enumerate(order):
        c.plot([10 ** err["debris"][u], 10 ** err["constant"][u]], [i, i], color=C["link"], lw=0.8, zorder=1)
        c.plot(10 ** err["constant"][u], i, "s", ms=4.4, color="white", mec=C["base"], mew=1.0, zorder=3, label="frozen constant" if i == 0 else None)
        c.plot(10 ** refit[u], i, "x", ms=4.6, color=C["base_dark"], mew=1.1, zorder=3, label="constant refitted in the new data" if i == 0 else None)
        st = u.split(":")[0]
        c.plot(10 ** err["debris"][u], i, PUB[st][1], ms=4.8, color=PUB[st][2], mec="black", mew=0.5, zorder=4, label="frozen factor" if i == 0 else None)
    c.axvline(2, color="black", ls=DASH, lw=0.7, zorder=0)
    c.set_xscale("log")
    c.set_xlim(1.0, 40)
    plain_log(c.xaxis, (1.0, 2.0, 5.0))
    c.set_yticks(range(len(order)), [u.replace("jet495:", "carburised ").replace("tribint2009:", "nitrided ").replace("ilt2011:", "interrupted ").replace("pij2011:", "power loss ") for u in order])
    c.set_xlabel("Error of the predicted wear (factor, per pair)")
    categorical(c, "y")
    dotted_grid(c, "x")
    framed(c.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=FS_MIN))
    for x_, s_ in ((0.02, "(a)"), (3.85, "(b)")):
        letter(fig, x_, y0 + h + 0.24, s_)
    save(fig, "paired_holdout")


DECISION_ROWS = [  # key of results.json decision_table, label
    ("D0 as run", "as run (frozen)"), ("D1 K6 sample", "K6 sample instead of K3 + K6"), ("D2 as run, P3 at 1 mg", "as run, pair P3 at 1 mg"), ("D3 K6 sample, P3 at 1 mg", "K6 sample and P3 at 1 mg"),
    ("D4a run totals, as run, frozen row constants", "whole losses, frozen constants"), ("D4a run totals, as run, constants of the thesis run totals", "whole losses, thesis constants"),
    ("D4b run totals, K6 sample, constants of the thesis run totals", "whole losses, K6 sample")]


def fig_paired_decision() -> None:
    """The statistics of the plan under the definitions that a reader could choose, and what the rule of the plan gives."""
    r, _, _ = _pooled()
    d = r["decision_table"]
    f = jload(PAIRED / "holdout" / "results.json")
    check("decision table reproduces the frozen run", d["D0 as run"]["H1c_debris_vs_refitted_constant"]["diff"], f["H1c_debris_vs_refitted_constant"]["diff"], 1e-9)
    n = len(DECISION_ROWS)
    H, y0, h = 2.95, 0.55, 1.85
    fig = plt.figure(figsize=(W, H))
    x0, w, gap = 2.02, 1.04, 0.16
    ax = [axes_at(fig, x0 + i * (w + gap), y0, w, h) for i in range(4)]
    for i, (k, lab) in enumerate(DECISION_ROWS):
        v = d[k]
        ci = [10 ** q for q in v["mean_log10_error_debris_ci"]]
        main = i == 0
        kw = dict(fmt="o", ms=5.0 if main else 4.4, color="black" if main else C["ours"], mfc="black" if main else "white", mec="black" if main else C["ours"], ecolor="black", elinewidth=0.9, capsize=2.2, zorder=3)
        ax[0].errorbar(v["fold"]["debris"], i, xerr=[[v["fold"]["debris"] - ci[0]], [ci[1] - v["fold"]["debris"]]], **kw)
        for j, key in ((1, "H1b_debris_vs_constant"), (2, "H1c_debris_vs_refitted_constant")):
            q = v[key]
            ax[j].errorbar(q["diff"], i, xerr=[[q["diff"] - q["ci"][0]], [q["ci"][1] - q["diff"]]], **kw)
        q = v["H4_slope"]
        ax[3].errorbar(q["slope"], i, xerr=[[q["slope"] - q["ci"][0]], [q["ci"][1] - q["slope"]]], **kw)
        dd = v["decision"]
        ax[3].text(1.03, i, "no-go" if dd.startswith("NO-GO") else "go, strong" if "strong" in dd else "go, limited", transform=ax[3].get_yaxis_transform(), ha="left", va="center", fontsize=FS_SMALL, weight="bold" if main else "normal")
    for j, (xl, lim, lines) in enumerate((("Error of the frozen\nfactor (H1a; limit 2.0)", (1.1, 2.6), (2.0,)), ("Minus frozen\nconstant (H1b)", (-1.15, 0.25), (0.0,)), ("Minus refitted\nconstant (H1c)", (-0.40, 0.14), (0.0,)), ("Slope, new pairs\n(H4)", (-0.8, 2.1), (0.0, 1.0)))):
        for v in lines:
            ax[j].axvline(v, color="black", ls=DASH if v != 1.0 else DOT, lw=0.8, zorder=1)
        ax[j].set_xlim(*lim)
        ax[j].set_ylim(n - 0.45, -0.55)
        ax[j].set_xlabel(xl, linespacing=1.1)
        categorical(ax[j], "y")
        dotted_grid(ax[j], "x")
        if j:
            ax[j].set_yticks(range(n), [""] * n)
        else:
            ax[j].set_yticks(range(n), [lab for _, lab in DECISION_ROWS])
        for k in range(0, n):
            ax[j].axhline(k - 0.5, color="#D0D0D0", lw=0.4, zorder=0)
        ax[j].axhline(0.5, color="black", lw=0.5)
        letter(fig, x0 + j * (w + gap) - 0.14, y0 + h + 0.19, "(abcd)"[0] + "abcd"[j] + ")")
    ax[3].text(1.03, -0.72, "rule of\nthe plan", transform=ax[3].get_yaxis_transform(), ha="left", va="bottom", fontsize=FS_SMALL, linespacing=1.0)
    save(fig, "paired_decision")


def fig_paired_stages() -> None:
    """Debris per unit wear of every weighed stage of the 11 micropitting pairs, as run and with the K6 sample in the first stage."""
    r, rows, _ = _pooled()
    d = rows[rows["mode"] == "interval"].copy()
    d["r0"], d["r1"] = d.D0 / d.wear, d.D1 / d.wear
    s0, s1 = r["stage_ratios"]["all 11, as run"], r["stage_ratios"]["all 11, K6 sample"]
    H, y0, h = 2.80, 0.48, 1.95
    fig = plt.figure(figsize=(W, H))
    a = axes_at(fig, 0.62, y0, 2.15, h)
    b = axes_at(fig, 3.30, y0, 2.15, h, sharey=a)
    for ax, thesis in ((a, True), (b, False)):
        sub = d[d.study.eq("feup_martins2008") == thesis]
        for k, (u, g) in enumerate(sub.groupby("unit", sort=False)):
            g = g.sort_values("order")
            st = g.study.iloc[0]
            lab, mk, col = PUB[st]
            mk = ("o", "s", "^", "D")[k] if thesis else mk
            ls = ("-", DASH, DOT, DASHDOT)[k % 4] if thesis else "-"
            xs = np.arange(1, len(g) + 1)
            ax.plot(xs, g.r0, color=C["link"] if thesis else col, lw=0.9, ls=ls, zorder=2)
            ax.scatter(xs, g.r0, s=20, marker=mk, color="white" if thesis else col, zorder=3, label=(u.split(":")[1].replace("_mp", "") if thesis else (lab if u in {"jet495:M1", "tribint2009:M1", "ilt2011:E2"} else None)), **EDGE)
            if not thesis and g.r1.iloc[0] != g.r0.iloc[0]:
                ax.scatter([1 + 0.14], [g.r1.iloc[0]], s=18, marker=mk, facecolors="none", edgecolors=col, linewidths=0.9, zorder=4, label="K6 sample (same pair)" if u == "jet495:M1" else None)
        ax.set_xlim(0.6, 4.5)
        ax.set_xticks([1, 2, 3, 4], ["1", "2", "3", "4"])
        ax.set_xlabel("Weighed stage of the test")
    a.set_yscale("log")
    a.set_ylim(1.5, 120)
    plain_log(a.yaxis)
    a.set_ylabel("Debris per unit wear (CPUC per mg)")
    plt.setp(b.get_yticklabels(), visible=False)
    for ax in (a, b):
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        dotted_grid(ax, "y")
    framed(a.legend(loc="upper right", fontsize=FS_MIN, title="thesis pairs", title_fontsize=FS_MIN))
    framed(b.legend(loc="upper right", fontsize=FS_MIN))
    c = axes_at(fig, 5.95, y0, 1.15, h)
    va = {"as run": s0, "K6 sample": s1}
    rng = np.random.default_rng(2)
    for i, (k, v) in enumerate(va.items()):
        vals = np.array(list(v["first_over_last"].values()))
        check(f"stage ratios median {k}", np.median(vals), v["median"], 1e-9)
        c.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14, color=C["ours"], zorder=3, **EDGE)
        c.plot([i - 0.25, i + 0.25], [v["median"]] * 2, color="black", lw=1.4, zorder=4)
    c.axhline(1, color="black", ls=DASH, lw=0.8)
    c.set_yscale("log")
    c.set_ylim(0.6, 14)
    plain_log(c.yaxis, (1.0, 2.0, 5.0))
    c.set_xlim(-0.6, 1.6)
    c.set_xticks([0, 1], ["as run", "K6\nsample"])
    c.set_ylabel("First over last stage")
    categorical(c, "x")
    dotted_grid(c, "y")
    for x_, s_ in ((0.02, "(a)"), (3.04, "(b)"), (5.62, "(c)")):
        letter(fig, x_, y0 + h + 0.20, s_)
    save(fig, "paired_stages")


def fig_cumulative_shuffle() -> None:
    """Cumulative curves against intervals: the plots of the sources, the interval rows, and the correlation after shuffling."""
    r, rows, _ = _pooled()
    cv0, cv1 = r["cumulative_vs_interval"]["7 new pairs, as run"], r["cumulative_vs_interval"]["7 new pairs, K6 sample"]
    d = rows[(rows["mode"] == "interval") & (rows.study != "feup_martins2008")]
    r2 = []
    H, y0, h = 2.95, 0.48, 1.92
    fig = plt.figure(figsize=(W, H))
    a = axes_at(fig, 0.60, y0, 1.95, h)
    b = axes_at(fig, 3.22, y0, 1.95, h)
    c = axes_at(fig, 5.80, y0, 1.30, h)
    for u, g in d.groupby("unit", sort=False):
        g = g.sort_values("order")
        st = u.split(":")[0]
        lab, mk, col = PUB[st]
        x, y = np.cumsum(g.D0.to_numpy()), np.cumsum(g.wear.to_numpy())
        r2.append(np.corrcoef(x, y)[0, 1] ** 2)
        a.plot(y, x, color=col, lw=0.9, zorder=2)
        a.scatter(y, x, s=14, marker=mk, color=col, zorder=3, label=lab if u in {"jet495:M1", "tribint2009:M1", "ilt2011:E2"} else None, **EDGE)
    check("mean R2 of the cumulative curves", np.mean(r2), cv0["mean_r2_cumulative_per_pair"], 1e-9)
    a.set_xlabel("Cumulative mass loss (mg)")
    a.set_ylabel("Cumulative CPUC")
    a.set_xlim(0, 100)
    a.set_ylim(0, 900)
    a.text(0.03, 0.97, f"mean $R^2$ {cv0['mean_r2_cumulative_per_pair']:.2f}", transform=a.transAxes, ha="left", va="top", fontsize=FS_SMALL)
    framed(a.legend(loc="lower right", fontsize=FS_MIN))
    for st in ("jet495", "tribint2009", "ilt2011"):
        q = d[d.study == st]
        b.scatter(q.wear, q.D0, s=18, marker=PUB[st][1], color=PUB[st][2], zorder=3, **EDGE)
        m1 = q[(q.D0 - q.D1).abs() > 1e-9]
        b.scatter(m1.wear, m1.D1, s=18, marker=PUB[st][1], facecolors="none", edgecolors=PUB[st][2], linewidths=0.9, zorder=3)
        for _, w in m1.iterrows():
            b.plot([w.wear] * 2, [w.D1, w.D0], color=C["link"], lw=0.6, zorder=2)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(2, 70)
    b.set_ylim(40, 900)
    plain_log(b.xaxis)
    plain_log(b.yaxis)
    b.set_xlabel("Mass loss of the stage (mg)")
    b.set_ylabel("CPUC of the stage")
    b.text(0.03, 0.97, f"rank corr. {cv0['interval_spearman']:.2f} ($p$ = {cv0['interval_spearman_p']:.2f})\nK6 sample {cv1['interval_spearman']:.2f} ($p$ = {cv1['interval_spearman_p']:.2f})", transform=b.transAxes, ha="left", va="top", fontsize=FS_MIN, linespacing=1.2)
    dr = np.array(cv0["null_within_pair"]["draws"])
    c.hist(dr, bins=np.linspace(0.85, 1.0, 26), color=C["neutral"], edgecolor="black", lw=0.4, zorder=2)
    c.axvline(cv0["mean_r2_cumulative_per_pair"], color=C["ours"], lw=1.4, zorder=3, label="observed")
    c.axvline(cv0["null_within_pair"]["mean_r2"], color="black", lw=0.9, ls=DASH, zorder=3, label="mean of the shuffles")
    c.set_xlim(0.85, 1.0)
    c.set_xlabel("Mean $R^2$ of cumulative\ncurves, intervals shuffled", linespacing=1.1)
    c.set_ylabel("Shuffles")
    c.set_yticks([])
    framed(c.legend(loc="lower right", bbox_to_anchor=(1.04, 1.02), fontsize=FS_MIN))
    for x_, s_ in ((0.02, "(a)"), (2.72, "(b)"), (5.46, "(c)")):
        letter(fig, x_, y0 + h + 0.20, s_)
    save(fig, "cumulative_shuffle")


def fig_paired_conversion() -> None:
    """Slopes and leave-one-unit-out errors of every paired set, with the 23 FZG pairs first."""
    _, p, u = _paired()
    r, _, _ = _pooled()
    P = r["pooled"]
    fz = [("FZG gears, all pairs (23)", P["23 pairs, as run"]), ("FZG gears, thesis (14)", P["14 thesis pairs"])]
    loo = pd.read_csv(PAIRED / "loo_predictions.csv")
    for key, lab, inf in PAIR_SETS:
        if not inf:
            continue
        st, se = key.split("/")
        e = loo[(loo.study == st) & (loo.set == se)].copy()
        for mod in ("M0", "M1", "M2", "M3"):
            if e[mod].notna().all():
                err = (np.log10(e.W) - e[mod]).abs().groupby(e.unit).mean().mean()
                check(f"paired {key} LOO MAE {mod}", err, p.loc[key, f"mae_{mod}"], 1e-4)
    check("paired FZG 14 pairs slope", P["14 thesis pairs"]["slope"], p.loc["feup_martins2008/fzg", "slope"], 1e-4)
    rows = [(lab, True, q["slope"], q["perm_ci"][0], q["perm_ci"][1], q["fold_loo"]) for lab, q in fz]
    for key, lab, inf in PAIR_SETS_SHOWN:
        v = p.loc[key]
        rows.append((lab, inf, v.slope, v.slope_lo, v.slope_hi, {"constant": 10 ** v.mae_M0, "conditions": 10 ** v.mae_M1, "debris": 10 ** v.mae_M2, "fused": 10 ** v.mae_M3, "archard": np.nan}))
    n = len(rows)
    H, y0, h = 3.10, 0.48, 2.15
    fig = plt.figure(figsize=(W, H))
    b = axes_at(fig, 1.95, y0, 2.10, h)
    for i, (lab, inf, s, lo, hi, _) in enumerate(rows):
        b.errorbar(s, i, xerr=[[s - lo], [hi - s]], fmt="o" if inf else "D", color=C["ours"], mfc=C["ours"] if inf else "white", ecolor="black", ms=4.8, zorder=3, **EB)
    b.axvline(1, color="black", ls=DASH, lw=0.8, zorder=1)
    b.axvline(0, color="black", ls=DOT, lw=0.8, zorder=1)
    b.set_yticks(range(n), [q[0] for q in rows])
    b.set_ylim(n - 0.45, -0.55)
    b.axhline(1.5, color="black", lw=0.5)
    b.axhline(5.5, color="black", lw=0.5)
    b.set_xlim(-1.25, 3.2)
    b.set_xlabel("Slope of log wear on log debris")
    categorical(b, "y")
    dotted_grid(b, "x")
    c = axes_at(fig, 4.50, y0, 2.60, h, sharey=b)
    ms = (("constant", "s", C["base"], C["base"], "constant"), ("conditions", "^", C["base"], "white", "conditions"), ("archard", "v", C["base_dark"], C["base_dark"], "Archard (FZG)"),
          ("debris", "o", C["ours"], C["ours"], "debris"), ("fused", "D", C["ours_ext"], C["ours_ext"], "debris + conditions"))
    for i, (*_, fo) in enumerate(rows):
        for j, (k, mk, col, fc, name) in enumerate(ms):
            if np.isfinite(fo.get(k, np.nan)):
                c.plot(fo[k], i + (j - 2) * 0.15, mk, color=col, mfc=fc, mec="black", mew=0.4, ms=4.6, zorder=3, label=name if i == 0 else None)
    c.axhline(1.5, color="black", lw=0.5)
    c.axhline(5.5, color="black", lw=0.5)
    c.set_xscale("log")
    c.set_xlim(1.1, 9)
    plain_log(c.xaxis, (1.0, 1.5, 2.0, 3.0, 5.0, 7.0))
    c.set_xlabel("Leave-one-out error (factor)")
    plt.setp(c.get_yticklabels(), visible=False)
    categorical(c, "y")
    dotted_grid(c, "x")
    framed(c.legend(loc="lower right", bbox_to_anchor=(1.02, 1.02), ncol=3, fontsize=FS_MIN, columnspacing=0.6, handletextpad=0.1))
    for x_, s_ in ((0.02, "(a)"), (4.20, "(b)")):
        letter(fig, x_, y0 + h + 0.24, s_)
    save(fig, "paired_conversion")


def fig_paired_criterion() -> None:
    """Debris per unit wear of every test of every set, and the scatter of one factor with the 95 % range of one prediction."""
    _, p, u = _paired()
    r, rows, pairs = _pooled()
    cr = {o["set"]: o for o in jload(PAIRED / "extra" / "results.json")["criterion"]}
    ident = jload(PAIRED / "extra" / "review.json")["identity"]["per_set"]
    later = jload(PAIRED / "later" / "results.json")["A_holzhauer1983"]["primary"]["end_spread"]
    P = r["pooled"]
    a23 = P["23 pairs, as run"]
    k23 = P["23 pairs, K6 sample"]
    s14 = [k for k, *_ in PAIR_SETS if k == "feup_martins2008/fzg"][0]
    lines = [("FZG gears, all pairs (23)", True, a23["gsd_D_over_W"], a23["gsd_ci"], a23["range95_factor"], pairs.D0 - pairs.wear), ("FZG gears, thesis (14)", True, ident[s14]["gsd_factor"], ident[s14]["gsd_factor_ci"], cr[s14]["pi95_factor"], u[u.key == s14].log10_D - u[u.key == s14].log10_W)]
    for key, lab, inf in PAIR_SETS_SHOWN:
        g = u[u.key == key]
        lr = g.log10_D - g.log10_W
        check(f"criterion {key} GSD of D/W", 10 ** lr.std(ddof=1), ident[key]["gsd_factor"], 1e-3)
        lines.append((lab, inf, ident[key]["gsd_factor"], ident[key]["gsd_factor_ci"], cr[key]["pi95_factor"], lr))
    hz = pd.read_csv(DATA_PAIRED / "holzhauer1983" / "later.csv").sort_values(["unit", "order"]).groupby("unit").last()
    lr = np.log10(hz.conc_ppm / hz.wear_mg)
    check("later ball bearings GSD of D/W", 10 ** lr.std(ddof=1), later["gsd"], 1e-3)
    check("criterion 23 pairs GSD", 10 ** (pairs.D0 - pairs.wear).std(ddof=1), a23["gsd_D_over_W"], 1e-9)
    lines.append(("ball bearings, later (4)", False, later["gsd"], later["gsd_ci"], later["pi95_factor"], lr))
    n = len(lines)
    H, y0, h = 3.20, 0.48, 2.40
    fig = plt.figure(figsize=(W, H))
    a = axes_at(fig, 1.95, y0, 2.35, h)
    rng = np.random.default_rng(0)
    for i, (lab, inf, gsd, ci, pi, v) in enumerate(lines):
        w = 10 ** (v - v.mean())
        a.scatter(w, i + rng.uniform(-0.16, 0.16, len(w)), s=13, marker="o" if inf else "D", color=C["ours"] if inf else "white", zorder=3, edgecolors="black", linewidths=0.4)
    a.axvline(1, color="black", ls=DASH, lw=0.8, zorder=1)
    a.set_xscale("log")
    a.set_xlim(0.08, 14)
    plain_log(a.xaxis, (1.0, 2.0, 5.0))
    a.set_yticks(range(n), [q[0] for q in lines])
    a.set_ylim(n - 0.45, -0.55)
    for yy, ls in ((1.5, "-"), (5.5, "-"), (n - 1.5, DOT)):
        a.axhline(yy, color="black", lw=0.5, ls=ls)
    a.set_xlabel("Debris per unit wear / geometric mean of the set")
    categorical(a, "y")
    dotted_grid(a, "x")
    b = axes_at(fig, 4.80, y0, 2.30, h, sharey=a)
    for i, (lab, inf, gsd, ci, pi, _) in enumerate(lines):
        b.plot([1, pi], [i, i], color=C["neutral"], lw=5.0, solid_capstyle="butt", zorder=1)
        b.errorbar(gsd, i, xerr=[[gsd - ci[0]], [ci[1] - gsd]], fmt="o" if inf else "D", color=C["ours"], mfc=C["ours"] if inf else "white", ecolor="black", ms=4.8, zorder=3, **EB)
        b.text(pi * 1.07, i - 0.28, f"×/÷ {pi:.1f}", ha="left", va="center", fontsize=FS_MIN, zorder=4)
    b.set_xscale("log")
    b.set_xlim(1, 70)
    plain_log(b.xaxis)
    for yy, ls in ((1.5, "-"), (5.5, "-"), (n - 1.5, DOT)):
        b.axhline(yy, color="black", lw=0.5, ls=ls)
    b.set_xlabel("GSD of debris per unit wear (point, 95 % interval);\n95 % range of one prediction (grey)", linespacing=1.1)
    plt.setp(b.get_yticklabels(), visible=False)
    categorical(b, "y")
    dotted_grid(b, "x")
    for x_, s_ in ((0.02, "(a)"), (4.45, "(b)")):
        letter(fig, x_, y0 + h + 0.24, s_)
    save(fig, "paired_criterion")


def fig_paired_limits() -> None:
    """Other limits of the conversion (descriptive): four ball bearings, recovered share between rigs, bearing cups, particle thickness."""
    _, p, u = _paired()
    res = jload(PAIRED / "results.json")
    all_ = pd.read_csv(PAIRED / "paired_all.csv")
    cups = pd.read_csv(PAIRED / "nasa_cups.csv")
    th = pd.read_csv(ROOT / "data" / "real" / "paired" / "jsme_higashi2003" / "thickness.csv")
    vp = res["volume_prior_check"]
    rec = res["across_studies"]["recovery_gm_per_set"]
    fit = stats.linregress(np.log10(th.size_um), np.log10(th.thickness_um))
    check("paired thickness slope", fit.slope, vp["slope_log_thickness_on_log_size"], 1e-4)
    d = pd.read_csv(DATA_PAIRED / "holzhauer1983" / "later.csv").sort_values(["unit", "order"])
    rz = jload(PAIRED / "later" / "results.json")["A_holzhauer1983"]["primary"]
    for b_, s in rz["slope_conc_per_mg"].items():
        q = d[d.unit == b_]
        check(f"later Holzhauer {b_} slope", np.polyfit(q.wear_mg, q.conc_ppm, 1)[0], s, 1e-3)
    H = 4.55
    fig = plt.figure(figsize=(W, H))
    a = axes_at(fig, 0.62, 2.82, 2.85, 1.40)
    for b_, mk, col, ls in (("A", "o", C["ours"], "-"), ("B", "s", C["ours_ext"], "--"), ("C", "^", C["base"], "-."), ("D", "D", C["base_dark"], ":")):
        q = d[d.unit == b_]
        ft = np.polyfit(q.wear_mg, q.conc_ppm, 1)
        xx = np.array([0, q.wear_mg.max()])
        a.plot(xx, np.polyval(ft, xx), color=col, lw=0.9, ls=ls, zorder=2)
        a.scatter(q.wear_mg, q.conc_ppm, s=16, marker=mk, color=col, zorder=3, label=f"bearing {b_} ({ft[0]:.1f})", **EDGE)
    a.set_xlim(0, 180)
    a.set_ylim(0, 400)
    a.set_xlabel("Bearing mass loss (mg)")
    a.set_ylabel("Ferrograph concentration (ppm)")
    framed(a.legend(loc="upper left", fontsize=FS_MIN, title="bearing (our slope, ppm/mg)", title_fontsize=FS_MIN))

    b = axes_at(fig, 5.05, 2.82, 2.05, 1.40)
    rows = [("pin, oil", u[u.key == "bu_torres2011/pin_totals"].pipe(lambda g: g.log10_D - g.log10_W), "bu_torres2011/pin_totals"),
            ("plate, oil (one run)", u[u.key == "jsme_higashi2003/plate_intervals"].pipe(lambda g: g.log10_D - g.log10_W), "jsme_higashi2003/plate_intervals"),
            ("four-ball, dry", u[u.key == "phm_raadnui2015/dry_fourball"].pipe(lambda g: g.log10_D - g.log10_W - np.log10(7.80)), "phm_raadnui2015/dry_fourball")]
    for i, (lab, rr, key) in enumerate(rows):
        check(f"paired recovery {key}", 10 ** rr.mean(), rec[key], 2e-3)
        b.scatter(10 ** rr, np.full(len(rr), i, float), s=15, color=C["ours"], zorder=3, **EDGE)
        b.plot([10 ** rr.mean()] * 2, [i - 0.3, i + 0.3], color="black", lw=1.4, zorder=4)
    b.axvline(1, color="black", ls=DASH, lw=0.8, zorder=1)
    b.set_xscale("log")
    b.set_xlim(0.2, 7)
    plain_log(b.xaxis, (1.0, 2.0, 5.0))
    b.set_yticks(range(len(rows)), [q[0] for q in rows])
    b.set_ylim(len(rows) - 0.45, -0.55)
    b.set_xlabel("Debris amount / measured wear")
    categorical(b, "y")
    dotted_grid(b, "x")

    c = axes_at(fig, 0.62, 0.48, 2.85, 1.72)
    cup = all_[all_.study == "nasa_tm2011_217117"]
    for (cid, mk, ls), col in zip((("cup27", "o", "-"), ("cup33", "s", DASH), ("cup35", "^", DOT), ("cup36", "D", DASHDOT)), (C["ours"], C["base"], C["ours_ext"], C["base_dark"])):
        g = cup[cup.unit == cid].sort_values("debris")
        rr = cups.set_index("cup").loc[cid]
        check(f"paired NASA {cid} kappa", stats.linregress(g.debris, g.wear).slope, rr.kappa_mm_per_mg, 1e-4)
        c.plot(g.debris, g.wear, color=col, ls=ls, lw=0.8, zorder=2)
        c.plot(g.debris, g.wear, mk, color=col, mec="black", mew=0.35, ms=3.6, zorder=3, label=f"cup {cid[3:]}: {1000 * rr.kappa_mm_per_mg:.0f} µm mg$^{{-1}}$")
    c.set_xlabel("Accumulated oil-debris mass (mg)")
    c.set_ylabel("Spall length (mm)")
    c.set_xlim(0, 1010)
    c.set_ylim(0, 30)
    framed(c.legend(loc="lower right", fontsize=FS_MIN))

    e = axes_at(fig, 4.35, 0.48, 2.75, 1.72)
    sx = np.geomspace(4, 100, 60)
    e.plot(sx, 0.3 * np.sqrt(np.pi) / 2 * sx, color="black", ls=DASH, lw=0.9, label="prior, $v = 0.3A^{1.5}$", zorder=2)
    e.plot(sx, 10 ** (fit.intercept + fit.slope * np.log10(sx)), color=C["ours"], lw=1.0, zorder=2, label=f"fit, slope {fmt_ci(vp['slope_log_thickness_on_log_size'], vp['slope_lo'], vp['slope_hi'])}")
    e.plot(th.size_um, th.thickness_um, "o", color=C["ours"], mec="black", mew=0.4, ms=4.4, zorder=3, label="measured (16 classes)")
    e.set_xscale("log")
    e.set_yscale("log")
    e.set_xlim(4, 100)
    e.set_ylim(0.7, 40)
    plain_log(e.xaxis)
    plain_log(e.yaxis)
    e.set_xlabel("Particle size (µm)")
    e.set_ylabel("Particle thickness (µm)")
    framed(e.legend(loc="upper left", fontsize=FS_MIN))
    for x_, y_, s_ in ((0.02, 4.44, "(a)"), (3.95, 4.44, "(b)"), (0.02, 2.42, "(c)"), (3.78, 2.42, "(d)")):
        letter(fig, x_, y_, s_)
    save(fig, "paired_limits")


# ============================================================================ composition of the debris (calculation)


def fig_composition_bias() -> None:
    """Bias of a calibrated debris-to-wear factor when the debris composition changes (closed form of
    scripts/bias_study.py, no data), next to the spread of debris per unit wear between tests of one rig."""
    d = pd.read_csv(SIM / "bias" / "bias.csv")
    gsd = jload(POOL / "results.json")["pooled"]["23 pairs, as run"]["gsd_D_over_W"]
    check("bias GSD of the 23 pairs", gsd, 1.674, 2e-3)
    f0, mark = 1.25, dict(color=C["ours"], mec="black", mew=0.4, ms=4.6)

    def curve(ax, term, c, mk, ls):
        """Draw a stored curve and the point where it reaches f0; return its rows and that share (closed form)."""
        g = d[(d.term == term) & np.isclose(d.r_or_c, c)].sort_values("share")
        x = (1 - 1 / f0) / (1 - c)
        check(f"bias {term} {c:.2f} share at {f0} (linear interpolation)", np.interp(f0, g.factor, g.share), x, 1e-3)
        ax.plot(g.share, g.factor, color=C["ours"], ls=ls, zorder=2)
        ax.plot(x, f0, mk, zorder=3, **mark)
        return g, x

    H, y0, w, h = 2.64, 0.46, 2.94, 1.72
    fig = plt.figure(figsize=(W, H))
    a, b = axes_at(fig, 0.56, y0, w, h), axes_at(fig, 7.10 - w, y0, w, h)
    g, x = curve(a, "oxide", 0.47, "o", "-")
    note = dict(textcoords="offset points", fontsize=FS_SMALL)
    a.annotate(f"{f0} at {x:.3f}", (x, f0), xytext=(4, -4), ha="left", va="top", **note)
    a.annotate(f"maximum {g.factor.max():.2f}", (1, g.factor.max()), xytext=(-8, 0), ha="right", va="center", **note)
    hd = []
    for c, name, mk, ls in ((0.1, "0.1", "^", DASHDOT), (1 / 3, "1/3", "o", "-"), (0.5, "0.5", "s", DOT)):
        g, x = curve(b, "capture", c, mk, ls)
        if name == "1/3":
            check("bias r = 1/3 share at 1.25", np.interp(f0, g.factor, g.share), 0.30)
        hd.append(Line2D([], [], marker=mk, ls=ls, label=f"$r$ = {name} ({f0} at {x:.3f})", **mark))
    for ax, xl in ((a, "Oxide share of the debris volume"), (b, "Non-ferrous share of the worn volume")):
        ax.axhspan(1, gsd, color=C["neutral"], lw=0, zorder=0)
        ax.axhline(f0, color="black", ls=DASH, lw=0.7, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.set_xlabel(xl)
        ax.set_ylabel("Bias of the calibrated factor")
    a.set_ylim(1, 2.3)
    a.set_yticks([1, 1.25, 1.5, 1.75, 2, 2.25])
    b.set_yscale("log")
    b.set_ylim(1, 10)
    plain_log(b.yaxis)
    framed(b.legend(handles=hd, loc="upper left", handlelength=2.4, borderaxespad=0.9))
    framed(fig.legend([Patch(facecolor=C["neutral"], lw=0), Line2D([], [], color="black", ls=DASH, lw=0.7)],
                      [f"spread between the 23 FZG pairs (GSD {gsd:.1f})", f"bias of {f0}"], loc="upper center", ncol=2,
                      bbox_to_anchor=((0.56 + 7.10) / 2 / W, 1 - 0.03 / H), handlelength=1.8, columnspacing=1.4,
                      borderaxespad=0))
    letter(fig, 0.02, y0 + h + 0.17, "(a)")
    letter(fig, 7.10 - w - 0.54, y0 + h + 0.17, "(b)")
    save(fig, "composition_bias")


FIGURES = {  # stem -> (function, required input files)
    "graphical_abstract": (fig_graphical_abstract, [POOL / "results.json", POOL / "rows.csv"]),
    "paired_pooled": (fig_paired_pooled, [POOL / "results.json", POOL / "pairs.csv"]),
    "paired_decision": (fig_paired_decision, [POOL / "results.json", PAIRED / "holdout" / "results.json"]),
    "paired_stages": (fig_paired_stages, [POOL / "results.json", POOL / "rows.csv"]),
    "cumulative_shuffle": (fig_cumulative_shuffle, [POOL / "results.json", POOL / "rows.csv"]),
    "paired_conversion": (fig_paired_conversion, [PAIRED / "results.json", PAIRED / "units.csv", POOL / "results.json"]),
    "paired_criterion": (fig_paired_criterion, [PAIRED / "extra" / "review.json", PAIRED / "later" / "results.json", PAIRED / "units.csv", POOL / "results.json"]),
    "paired_holdout": (fig_paired_holdout, [PAIRED / "holdout" / "results.json", PAIRED / "holdout" / "predictions.csv"]),
    "paired_limits": (fig_paired_limits, [PAIRED / "results.json", PAIRED / "nasa_cups.csv"]),
    "composition_bias": (fig_composition_bias, [SIM / "bias" / "bias.csv", POOL / "results.json"]),
    "nasa_debris_vs_damage": (fig_nasa_debris_vs_damage, [OIL / "oil_results.json"]),
    "nasa_onset": (fig_nasa_onset, [OIL / "nasa_onset_readings.csv", DIAG / "f6_summary.json"]),
    "field_phase_rates": (fig_field_phase_rates, [OIL / "zenodo_unit_rates.csv"]),
    "field_composition": (fig_field_composition, [OIL / "zenodo_composition_results.csv",
                                                  DIAG / "f7_composition_sensitivity.csv"]),
    "ferro_overlays": (fig_ferro_overlays, [FER / "seg_per_image_test.csv", ZEN]),
    "seg_benchmark": (fig_seg_benchmark, [FER / "seg_summary_test.json", FER / "seg_paired_test.csv"]),
    "debris_area_agreement": (fig_debris_area_agreement, [FER / "seg_per_image_test.csv"]),
    "leakage_audit": (fig_leakage_audit, [FER / "seg_leak_check.csv", ZEN]),
    "invariance_real": (fig_invariance_real, [FER / "robust_stability.csv", FER / "robust_summary.json"]),
    "contour_graphs_real": (fig_contour_graphs_real, [FER / "robust_chord_hops.csv", ZEN]),
    "real_vs_synthetic_morphology": (fig_real_vs_synthetic_morphology, [FER / "particle_descriptors.csv.gz",
                                                                        FER / "domain_gap_auc.csv"]),
    "diagnostics_posthoc": (fig_diagnostics_posthoc, [DIAG / "f3_summary.json", DIAG / "f4_summary.json",
                                                      DIAG / "f5_summary.json"]),
    "renderer_v1_v2": (fig_renderer_v1_v2, [REND / "before_after.csv", REND / "fig_real_v1_v2.png"]),
    "observation_chain": (fig_observation_chain, [OIL / "oil_results.json", FER / "seg_summary_test.json"]),
    "sim_stress": (fig_sim_stress, [SIM / "stress" / "stress_campaigns.csv", SIM / "stress" / "stress_summary.csv"]),
    "sim_power": (fig_sim_power, [SIM / "power" / "power_campaigns.csv", SIM / "power" / "power_summary.csv"]),
    "sim_pilot_rate": (fig_sim_pilot_rate, [PILOT / "metrics" / "regression.csv"]),
    "sim_pilot_volume": (fig_sim_pilot_volume, [PILOT / "metrics" / "volume_consistency.csv"]),
}


def main(argv=None) -> int:
    style()
    want = list(argv if argv is not None else sys.argv[1:]) or list(FIGURES)
    unknown = [w for w in want if w not in FIGURES]
    assert not unknown, f"unknown stems {unknown}; known: {list(FIGURES)}"
    status = {}
    for stem in want:
        fn, inputs = FIGURES[stem]
        missing = [str(p.relative_to(ROOT)) for p in inputs if not p.exists()]
        if missing:
            status[stem] = "WAITING for " + ", ".join(missing)
        else:
            fn()
            status[stem] = "ok"
        print(f"{stem}: {status[stem]}", flush=True)
    CHECK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CHECK_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"# run {pd.Timestamp.now():%Y-%m-%d %H:%M:%S} stems {want}\n" + "\n".join(CHECKS + STYLE_LOG) + "\n")
    print(f"{len(CHECKS)} checks passed (log {CHECK_LOG.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
