"""Summaries, figures and README of the SIMULATION stress tests and power analysis (scripts/sim_campaigns.py).

stress: per (factor, level) and estimator the mean and the 2.5 / 97.5 % quantiles over campaigns of run MAE of log10 k,
within-condition Spearman, volume CCC and slope, plus the share of campaigns where physics_conditions has a lower MAE
than each component. power: per cell and debris estimator the share of campaigns whose paired-bootstrap CI of the
run-MAE difference vs conditions_only lies below 0 and whose within-condition Spearman CI lies above 0 (Wilson 95 %
interval of each share). Everything is labelled SIMULATION.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BANNER = ("SIMULATION - parameter-level synthetic campaigns (no rendering, renderer not calibrated to real ferrograms); "
          "design and software verification only, not evidence about real wear")
CAVEAT = ("The generator and the physics_only observation model share their structure (aggregate Archard volume, "
          "observation fraction, capture efficiency by magnetic class), so these numbers measure internal consistency, "
          "identifiability and campaign power of the debris chain, not its external validity.")
MODELS = ("physics_only", "conditions_only", "physics_conditions")
COLORS = {"physics_only": "#2a78d6", "conditions_only": "#eb6834", "physics_conditions": "#1baf7a"}  # fixed order
METRICS = {"mae": "run MAE of log10 k", "within_spearman": "within-condition Spearman", "ccc": "volume CCC",
           "slope": "volume slope"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval of k successes in n trials."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return mid - half, mid + half


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def stress_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (factor, level), g in df.groupby(["factor", "level"], sort=False):
        for m in MODELS:
            for metric in METRICS:
                col = f"{m}.{metric}"
                if col not in g:
                    continue
                v = g[col].to_numpy(float)
                v = v[np.isfinite(v)]
                rows.append({"factor": factor, "level": level, "pilot_level": bool(g.pilot_level.iloc[0]),
                             "model": m, "metric": metric, "mean": v.mean() if v.size else np.nan,
                             "q025": np.quantile(v, 0.025) if v.size else np.nan,
                             "q975": np.quantile(v, 0.975) if v.size else np.nan, "n_campaigns": int(v.size)})
        for b in ("physics_only", "conditions_only"):
            win = (g["physics_conditions.mae"] < g[f"{b}.mae"]).to_numpy()
            lo, hi = wilson(win.sum(), len(win))
            rows.append({"factor": factor, "level": level, "pilot_level": bool(g.pilot_level.iloc[0]),
                         "model": "physics_conditions", "metric": f"share_mae_below_{b}", "mean": win.mean(),
                         "q025": float(lo), "q975": float(hi), "n_campaigns": int(len(win))})
    return pd.DataFrame(rows)


def stress_figure(s: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factors = list(dict.fromkeys(s.factor))
    fig, axes = plt.subplots(len(METRICS), len(factors), figsize=(3.0 * len(factors), 2.4 * len(METRICS)),
                             squeeze=False, facecolor=SURFACE)
    for j, f in enumerate(factors):
        levels = list(dict.fromkeys(s[s.factor == f].level))
        x = np.arange(len(levels))
        for i, (metric, label) in enumerate(METRICS.items()):
            ax = axes[i, j]
            _style(ax)
            for k, m in enumerate(MODELS):
                d = s[(s.factor == f) & (s.model == m) & (s.metric == metric)].set_index("level").reindex(levels)
                if d["mean"].isna().all():
                    continue
                off = (k - 1) * 0.12
                ax.errorbar(x + off, d["mean"], yerr=[d["mean"] - d.q025, d.q975 - d["mean"]], color=COLORS[m],
                            marker="o", markersize=4, linewidth=1.5, elinewidth=1.0, capsize=2, label=m)
            pilot = [lv for lv in levels if s[(s.factor == f) & (s.level == lv)].pilot_level.any()]
            for lv in pilot:
                ax.axvline(levels.index(lv), color=MUTED, linewidth=0.6, linestyle=":")
            if metric in ("ccc", "slope"):
                ax.axhline(0.8 if metric == "ccc" else 1.0, color=MUTED, linewidth=0.8, linestyle="--")
            if metric == "within_spearman":
                ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle="--")
            ax.set_xticks(x, [lv.replace(" ", "") for lv in levels], fontsize=7)
            if j == 0:
                ax.set_ylabel(label, color=INK, fontsize=8)
            if i == 0:
                ax.set_title(f, color=INK, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.text(0.5, 0.005, f"{BANNER}. Points: mean over campaigns; bars: 2.5-97.5 % of campaigns; dotted: pilot level.",
             ha="center", fontsize=6.5, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def power_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rep, nc, imf), g in df.groupby(["repeats", "n_conditions", "imaged_fraction_range"], sort=True):
        for m in ("physics_only", "physics_conditions"):
            ok_mae = (g[f"{m}.mae_diff_hi"] < 0).to_numpy()
            ok_w = (g[f"{m}.within_lo"] > 0).to_numpy()
            both = ok_mae & ok_w
            row = {"repeats": rep, "n_conditions": nc, "n_runs": int(g.n_runs.iloc[0]),
                   "imaged_fraction_range": imf, "model": m, "n_campaigns": int(len(g))}
            for name, ok in (("p_mae_ci_below_0", ok_mae), ("p_within_ci_above_0", ok_w), ("p_both", both)):
                lo, hi = wilson(ok.sum(), len(ok))
                row.update({name: ok.mean(), f"{name}_lo": float(lo), f"{name}_hi": float(hi)})
            row.update({"mean_mae": g[f"{m}.mae"].mean(), "mean_mae_conditions_only": g["conditions_only.mae"].mean(),
                        "mean_within_spearman": g[f"{m}.within_spearman"].mean()})
            rows.append(row)
    return pd.DataFrame(rows)


def power_figure(p: pd.DataFrame, col: str, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    imfs = list(dict.fromkeys(p.imaged_fraction_range))
    models = ("physics_only", "physics_conditions")
    reps, ncs = sorted(p.repeats.unique()), sorted(p.n_conditions.unique())
    fig, axes = plt.subplots(len(models), len(imfs), figsize=(3.1 * len(imfs), 2.9 * len(models)), squeeze=False,
                             facecolor=SURFACE)
    for i, m in enumerate(models):
        for j, imf in enumerate(imfs):
            ax = axes[i, j]
            d = p[(p.model == m) & (p.imaged_fraction_range == imf)].pivot(index="n_conditions", columns="repeats",
                                                                         values=col).reindex(index=ncs, columns=reps)
            ax.imshow(d.to_numpy(float), cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
            for a in range(len(ncs)):
                for b in range(len(reps)):
                    v = d.to_numpy(float)[a, b]
                    ax.text(b, a, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v >= 0.55 else INK)
            ax.set_xticks(range(len(reps)), [str(r) for r in reps], fontsize=8)
            ax.set_yticks(range(len(ncs)), [str(c) for c in ncs], fontsize=8)
            ax.set_xlabel("runs per condition", fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(f"{m}\nnumber of conditions", fontsize=8, color=INK)
            if i == 0:
                ax.set_title(f"imaged fraction {json.loads(imf)}", fontsize=8, color=INK)
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle(title, fontsize=9, color=INK)
    fig.text(0.5, 0.005, BANNER, ha="center", fontsize=6.5, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def summarize(out: Path) -> None:
    out = Path(out)
    sp = out / "stress" / "stress_campaigns.csv"
    if sp.exists():
        df = pd.read_csv(sp)
        first = lambda v: v[0] if isinstance(v, list) else v  # noqa: E731
        df["_order"] = [first(json.loads(v)) for v in df.level]  # numeric level order (CSV rows follow completion)
        df = df.sort_values(["factor", "_order"], kind="stable")
        s = stress_summary(df)
        s.to_csv(out / "stress" / "stress_summary.csv", index=False, lineterminator="\n")
        stress_figure(s, out / "stress" / "stress_metrics.png")
        main = s[s.metric.isin(["mae", "within_spearman", "ccc"])].copy()
        main["value"] = [f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(main["mean"], main.q025, main.q975)]
        wide = main.pivot_table(index=["factor", "level"], columns=["model", "metric"], values="value",
                                aggfunc="first", sort=False)
        wide.columns = [f"{m} {k}" for m, k in wide.columns]
        share = s[s.metric.str.startswith("share")].pivot_table(index=["factor", "level"], columns="metric",
                                                                 values="mean", sort=False)
        (out / "stress" / "README.md").write_text("\n".join([
            "# Stress tests of the run-level debris estimators", "", f"**{BANNER}**", "", CAVEAT, "",
            "One factor at a time around configs/synthetic_pilot.yaml (54 runs = 3 materials x 3 loads x 2 speeds x "
            f"3 repeats, 5 samples per run unless varied); {int(df.groupby(['factor', 'level']).size().min())} "
            "campaigns per setting; each campaign = pipeline-identical grouped 5-fold cv (experiments.make_splits + "
            "run_run_baseline). Cells: mean [2.5 %, 97.5 %] over campaigns. Confocal thickness / volume noise is not a "
            "factor here: physics_only, conditions_only and physics_conditions never use confocal data (only the "
            "wearpg volume head does), so their results are invariant to it by construction.", "",
            "Files: stress_campaigns.csv (one row per campaign), stress_summary.csv, stress_metrics.png.", "",
            _md(wide.reset_index()), "",
            "Share of campaigns where physics_conditions has a lower run MAE than the component (Wilson CIs in "
            "stress_summary.csv):", "", _md(share.reset_index()), ""]), encoding="utf-8")
    pp = out / "power" / "power_campaigns.csv"
    if pp.exists():
        df = pd.read_csv(pp)
        p = power_summary(df)
        p.to_csv(out / "power" / "power_summary.csv", index=False, lineterminator="\n")
        power_figure(p, "p_mae_ci_below_0", "P(paired-bootstrap CI of run-MAE difference vs conditions_only < 0)",
                     out / "power" / "power_mae.png")
        power_figure(p, "p_within_ci_above_0", "P(within-condition Spearman CI > 0)", out / "power" / "power_within.png")
        power_figure(p, "p_both", "P(both criteria)", out / "power" / "power_both.png")
        show = p[["model", "imaged_fraction_range", "n_conditions", "repeats", "n_runs", "n_campaigns",
                  "p_mae_ci_below_0", "p_within_ci_above_0", "p_both", "mean_mae", "mean_mae_conditions_only"]]
        (out / "power" / "README.md").write_text("\n".join([
            "# Campaign power of the debris-based wear-rate estimators", "", f"**{BANNER}**", "", CAVEAT, "",
            "Grid: runs per condition x number of conditions (3 materials x {2 loads x 1 speed, 2 x 2, 3 x 2}) x "
            "imaged fraction range (LogUniform). Per campaign: pipeline-identical grouped 5-fold cv; paired cluster "
            "bootstrap (1000 replicates, clusters = specimens) of the run-MAE difference debris estimator - "
            "conditions_only and cluster bootstrap of the within-condition Spearman. Power = share of campaigns with "
            "the 95 % CI below 0 (MAE difference) / above 0 (Spearman); Wilson 95 % intervals in power_summary.csv.",
            "", "Files: power_campaigns.csv, power_summary.csv, power_mae.png, power_within.png, power_both.png.", "",
            _md(show), ""]), encoding="utf-8")


def overview(out: Path) -> Path | None:
    """outputs/simulation/README.md: what the stress and power simulations say about the paper claims (FRAMING sec. 3)."""
    out = Path(out)
    sp, pp = out / "stress" / "stress_summary.csv", out / "power" / "power_summary.csv"
    if not (sp.exists() and pp.exists()):
        return None
    s, p = pd.read_csv(sp), pd.read_csv(pp)
    base = s[(s.factor == "budget_sigma") & (s.level == "0.3")]

    def v(model, metric, frame=base):
        r = frame[(frame.model == model) & (frame.metric == metric)].iloc[0]
        return f"{r['mean']:.3f} [{r.q025:.3f}, {r.q975:.3f}]"

    worst = s[(s.factor == "imaged_fraction_range") & (s.level == "[0.005, 0.05]")]
    share = s[s.metric == "share_mae_below_physics_only"]
    lines = ["# Simulation studies of the debris -> wear-rate chain", "", f"**{BANNER}**", "", CAVEAT, "",
             "Subfolders: stress/ (one factor at a time, 20 campaigns per setting), power/ (design grid, 50 campaigns "
             "per cell), stress_wearpg/ (wearpg/full under confocal-volume noise on the rendered v1 pilot data). "
             "Values: mean [2.5 %, 97.5 %] over simulated campaigns at the pilot design unless stated.", "",
             "## Mapping to the Wear claims (paper/FRAMING.md sec. 3) - simulation level only", "",
             f"- C1 (debris volume gives the wear rate): the mechanism-agnostic debris estimator physics_only has run "
             f"MAE of log10 k {v('physics_only', 'mae')} vs conditions_only {v('conditions_only', 'mae')}; volume CCC "
             f"{v('physics_only', 'ccc')} (criterion >= 0.8 met in every setting), but volume slope "
             f"{v('physics_only', 'slope')}: the slope-CI-contains-1 criterion fails for a mechanism-agnostic volume "
             "(v = 0.3 A^1.5 ignores that severe debris is thicker: in the generator the true metal volume / 0.3 sum "
             "A^1.5 is 0.59 in mild and 0.90 in severe intervals, campaign seed 1000), which is the gap mechanism "
             "attribution (C2) must close in the pilot.",
             f"- C1b (beyond operating conditions): within-condition Spearman of physics_only "
             f"{v('physics_only', 'within_spearman')} (lowest setting, imaged fraction 0.005-0.05: "
             f"{v('physics_only', 'within_spearman', worst)}); conditions_only {v('conditions_only', 'within_spearman')}"
             " (negative: a run left out of training pulls its own condition estimate away from itself).",
             f"- Fusion hypothesis H-F (C1 support): physics_conditions MAE {v('physics_conditions', 'mae')} beats "
             f"conditions_only in every campaign but beats physics_only in only {share.loc[share.level == '0.3', 'mean'].iloc[0]:.0%} "
             f"of pilot-design campaigns (at most {share['mean'].max():.0%} in any setting): with regime transitions "
             "driving the run-to-run variation, the known conditions add nothing the debris volume does not already "
             "carry. Against the pre-specified 'beats both components' at simulation level.",
             f"- Robustness: imaged fraction is the dominant factor (MAE {v('physics_only', 'mae', worst)} at 0.005-0.05); "
             "budget noise matters little up to sd 0.6; fine-debris loss (min area 30 -> 150 px) and material capture "
             "spread (absorbed by the per-material eta offset in cv) leave the estimate unchanged; more samples per "
             "run did not help in this generator.", "", "## Design guidance (power)", "",
             "Smallest simulated campaign (runs = conditions x runs per condition) whose power for BOTH criteria "
             "(run-MAE CI below conditions_only and within-condition Spearman CI above 0) is >= 0.8:", ""]
    for m in ("physics_only", "physics_conditions"):
        for imf, g in p[p.model == m].groupby("imaged_fraction_range", sort=False):
            ok = g[g.p_both >= 0.8].sort_values(["n_runs", "repeats"])
            best = ("none in the grid" if ok.empty else
                    f"{int(ok.n_conditions.iloc[0])} conditions x {int(ok.repeats.iloc[0])} runs = {int(ok.n_runs.iloc[0])} "
                    f"runs (power {ok.p_both.iloc[0]:.2f} [{ok.p_both_lo.iloc[0]:.2f}, {ok.p_both_hi.iloc[0]:.2f}])")
            lines.append(f"- {m}, imaged fraction {imf}: {best}")
    po = p[(p.model == "physics_only") & (p.imaged_fraction_range == "[0.05, 0.5]")]
    two = po[po.n_conditions < 18]
    fus = p[p.imaged_fraction_range == "[0.005, 0.05]"].pivot_table(index="n_conditions", columns="model",
                                                                     values="mean_mae")
    within5 = p[(p.model == "physics_only") & (p.repeats >= 5)].p_within_ci_above_0.min()
    lines += ["", "Reading the grid (power_summary.csv, power_*.png):",
              f"- The run-MAE criterion depends on the condition design more than on the number of runs: with 2 load "
              f"levels (6 or 12 conditions) the linear conditions_only model reproduces every condition mean (MAE "
              f"{two.mean_mae_conditions_only.min():.2f}-{two.mean_mae_conditions_only.max():.2f}) and physics_only beats "
              f"it with a CI below 0 in at most {two.p_mae_ci_below_0.max():.0%} of campaigns even at the pilot imaged "
              "fraction (at low imaged fractions even fewer with more repeats, which sharpen the conditions model); with "
              "3 load levels (18 "
              "conditions) the transition-driven, non-linear load response defeats the linear conditions model "
              f"(MAE {po[po.n_conditions == 18].mean_mae_conditions_only.mean():.2f}) and the power is "
              f"{po[po.n_conditions == 18].p_mae_ci_below_0.min():.2f}-{po[po.n_conditions == 18].p_mae_ci_below_0.max():.2f}.",
              f"- The within-condition criterion (C1b) has power >= {within5:.2f} for physics_only in every cell with >= 5 "
              "runs per condition; 3 runs per condition is marginal when the imaged fraction is low.",
              "- Fusion helps where the debris is noisy and the conditions model is good: at imaged fraction 0.005-0.05 "
              "the mean MAE of physics_conditions vs physics_only is "
              + ", ".join(f"{fus.at[n, 'physics_conditions']:.2f} vs {fus.at[n, 'physics_only']:.2f} ({n} conditions)"
                          for n in fus.index) + "."]
    wp = out / "stress_wearpg" / "summary.csv"
    if wp.exists():
        lines += ["", "## wearpg/full under confocal-volume noise", "", "See stress_wearpg/README.md."]
    path = out / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
