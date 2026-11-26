"""Draft PNG figures from the CSVs of scripts/open_oil_analysis.py (outputs/open_data/oil/). Final paper figures are
made later from the same CSVs.

Usage: .venv/Scripts/python.exe -I scripts/open_oil_figures.py [--root "<project root>"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PHASES = ("infant_mortality", "useful_life", "wear_out")
PHASE_LABEL = {"infant_mortality": "infant mortality", "useful_life": "useful life", "wear_out": "wear-out"}
PHASE_TICK = {"infant_mortality": "infant\nmortality", "useful_life": "useful\nlife", "wear_out": "wear-out"}
PHASE_COLOR = {"infant_mortality": "#2a78d6", "useful_life": "#eb6834", "wear_out": "#1baf7a"}  # validated, all-pairs
STAGE_COLOR = {0: "#86b6ef", 1: "#2a78d6", 2: "#104281"}  # ordinal ramp (validated --ordinal)
STAGE_LABEL = {0: "no pitting", 1: "initial pitting only", 2: "destructive pitting"}
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False})


def _strip(ax, groups, values, colors, rng):
    """Jittered unit dots per group with a median tick."""
    for i, (v, c) in enumerate(zip(values, colors)):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=28, color=c, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, linewidth=1.5, zorder=4)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def fig_phase_rates(d: Path) -> None:
    u = pd.read_csv(d / "zenodo_unit_rates.csv")
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, col, title in ((axs[0], "rate_cum_Fe_q0", "Fe"), (axs[1], "rate_cum_wear_metals_q0", "total wear metals")):
        _strip(ax, [PHASE_TICK[p] for p in PHASES], [u.loc[u.phase == p, col] for p in PHASES],
               [PHASE_COLOR[p] for p in PHASES], rng)
        ax.set_yscale("log")
        ax.set_title(f"{title}: G(100 h) / 100 h", fontsize=9)
        ax.set_ylabel("debris wear-rate proxy [ppm/h], unit = tractor")
    for p in PHASES:
        g = u[u.phase == p]
        axs[2].scatter(g.tractor_age, g.rate_cum_Fe_q0, s=28, color=PHASE_COLOR[p], edgecolor=SURFACE, linewidth=1,
                       label=PHASE_LABEL[p], zorder=3)
    axs[2].set_yscale("log")
    axs[2].set_xlabel("tractor age [years]")
    axs[2].set_ylabel("Fe proxy [ppm/h]")
    axs[2].set_title("phase labels are inferred from age (confound)", fontsize=9)
    axs[2].legend(loc="upper left")
    fig.suptitle("Zenodo 17287193 tractor engine oil: Fe-based debris wear-rate proxy by life phase (27 tractors)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "oil_fig_phase_rates.png", dpi=200)
    plt.close(fig)


def fig_transition_timeline(d: Path) -> None:
    run = pd.read_csv(d / "zenodo_running_rates_Fe.csv")
    al = pd.read_csv(d / "zenodo_transition_alarms.csv")
    al = al[(al.subset == "all_units") & (al.feature == "rate_cum_Fe_q0")].set_index("unit_id")
    s = pd.read_csv(d / "zenodo_transition_summary.csv").set_index(["subset", "feature"]).loc[
        ("all_units", "rate_cum_Fe_q0")]
    units = run.drop_duplicates("unit_id").sort_values(["phase", "unit_id"], key=lambda c: c.map(
        {p: i for i, p in enumerate(PHASES)}) if c.name == "phase" else c).unit_id.tolist()
    fig, ax = plt.subplots(figsize=(6.4, 7))
    for y, unit in enumerate(units):
        r = run[run.unit_id == unit]
        c = PHASE_COLOR[r.phase.iloc[0]]
        on = (r.running_rate >= al.loc[unit, "tau"]).to_numpy()
        ax.scatter(r.working_hours[on], [y] * on.sum(), s=40, color=c, zorder=3)
        ax.scatter(r.working_hours[~on], [y] * (~on).sum(), s=40, facecolor=SURFACE, edgecolor=c, linewidth=1.2,
                   zorder=3)
    ax.set_yticks(range(len(units)), units, fontsize=7)
    ax.invert_yaxis()
    ax.set_xticks([20, 40, 60, 80, 100])
    for p in PHASES:
        ax.scatter([], [], s=40, color=PHASE_COLOR[p], label=PHASE_LABEL[p])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    ax.set_xlabel("working hours [h]   (filled = alarm: running proxy G(t)/t >= tau; hollow = no alarm)")
    ax.set_title(f"Leave-one-tractor-out alarm rule G(t)/t >= tau (Fe, tau {s.tau_min:.2f}-{s.tau_max:.2f} ppm/h)\n"
                 f"detection {s.detection_rate:.2f}, false alarm {s.false_alarm_rate:.2f} (useful life), "
                 f"{s.extra_negative_false_alarm_rate:.2f} (infant); no within-unit transition exists", fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "oil_fig_transition_timeline.png", dpi=200)
    plt.close(fig)


def fig_composition(d: Path) -> None:
    f = pd.read_csv(d / "zenodo_composition_features.csv")
    cols = [c for c in f.columns if c.startswith("alr_")]
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, len(cols), figsize=(2.1 * len(cols), 3.2))
    for ax, c in zip(axs, cols):
        _strip(ax, ["IM", "UL", "WO"], [f.loc[f.phase == p, c] for p in PHASES], [PHASE_COLOR[p] for p in PHASES], rng)
        ax.set_title(f"ln({c[4:]}/Fe)", fontsize=9)
    axs[0].set_ylabel("unit median of additive log-ratio")
    r = pd.read_csv(d / "zenodo_composition_results.csv")
    r = r[(r.subset == "all_units") & (r.task == "3class")].set_index("features")
    fig.suptitle("Wear-metal composition by life phase (IM infant mortality, UL useful life, WO wear-out; 27 tractors)"
                 f"\nLOO balanced accuracy: composition {r.loc['composition', 'balanced_accuracy']:.2f}, "
                 f"Fe-only {r.loc['fe_only', 'balanced_accuracy']:.2f} "
                 f"(difference {r.loc['composition_minus_fe_only', 'balanced_accuracy']:.2f}, 95% CI "
                 f"{r.loc['composition_minus_fe_only', 'ba_lo']:.2f} to {r.loc['composition_minus_fe_only', 'ba_hi']:.2f})",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(d / "oil_fig_composition_shift.png", dpi=200)
    plt.close(fig)


def fig_nasa_mass_damage(d: Path) -> None:
    ex = pd.read_csv(d / "nasa_experiments_tm2003_table3_9.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, col, xl in ((axs[0], "mass_end_mg", "accumulated oil-debris mass at test end [mg]"),
                        (axs[1], "rdg_end", "test duration [min] (time-only baseline)")):
        for st in (0, 1, 2):
            g = ex[ex.stage_end == st]
            ax.scatter(g[col], g.n_de_end, s=32, color=STAGE_COLOR[st], edgecolor=SURFACE, linewidth=1,
                       label=f"{STAGE_LABEL[st]} (n={len(g)})", zorder=3)
        ax.set_xlabel(xl)
    axs[0].set_ylabel("destructively pitted teeth at test end")
    axs[0].legend(loc="upper left", fontsize=8)
    sp = json.loads((d / "oil_results.json").read_text())["nasa"]["spearman_vs_n_destructive_teeth"]
    for ax, col in zip(axs, ("mass_end_mg", "rdg_end")):
        ax.set_title(f"Spearman {sp[col]['spearman']:.2f} (95% CI {sp[col]['lo']:.2f} to {sp[col]['hi']:.2f})",
                     fontsize=9)
    fig.suptitle("NASA spur-gear fatigue rig, 24 experiments (TM-2003-211307 Table 3.9, transcribed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "oil_fig_nasa_mass_vs_damage.png", dpi=200)
    plt.close(fig)


def fig_nasa_onset(d: Path) -> None:
    r = pd.read_csv(d / "nasa_onset_readings.csv")
    r = r[(r.variant == "table_3_9") & (r.feature == "mass_mg")].sort_values(["exp", "reading"])
    r["k"] = r.groupby("exp").cumcount() + 1
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axhspan(r.tau.min(), r.tau.max(), color=GRID, zorder=1, label="LOO threshold range")
    for _, g in r.groupby("exp"):
        ax.plot(g.k, g.mass_mg, color=AXIS, linewidth=1, zorder=2)
    for st in (0, 1, 2):
        g = r[r.stage == st]
        ax.scatter(g.k, g.mass_mg, s=30, color=STAGE_COLOR[st], edgecolor=SURFACE, linewidth=1, zorder=3,
                   label=STAGE_LABEL[st])
    ax.set_xlabel("inspection number within experiment (loaded test, run-in excluded)")
    ax.set_ylabel("accumulated debris mass [mg]")
    ax.legend(loc="upper left", fontsize=8)
    o = json.loads((d / "oil_results.json").read_text())["nasa"]["onset_detection"]["table_3_9|mass_mg"]
    ax.set_title("Onset detection at inspections: mass >= tau (leave-one-experiment-out)\n"
                 f"experiment detection {o['exp_detection_rate']:.3f} (pre-specified >= 0.8: miss), false alarm "
                 f"{o['exp_false_alarm_rate']:.2f}\nreading-level AUC {o['auc_raw']:.2f} (secondary)", fontsize=9)
    fig.tight_layout()
    fig.savefig(d / "oil_fig_nasa_onset.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    d = Path(ap.parse_args().root) / "outputs" / "open_data" / "oil"
    for fn in (fig_phase_rates, fig_transition_timeline, fig_composition, fig_nasa_mass_damage, fig_nasa_onset):
        fn(d)
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
