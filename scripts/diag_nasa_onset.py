"""F6 EXPLORATORY (DIAG; rules fixed beforehand in outputs/agent_logs/DIAG.md): NASA spur-gear onset detection
(Dempsey, NASA/TM-2003-211307, transcribed tables in wearpg.open_oil) with the debris accumulation RATE between
inspections - consistent with the cumulative observation model: the oil-debris monitor accumulates debris mass, so
its time derivative is the debris generation rate - and a mass-OR-rate rule. Leave-one-experiment-out Youden
thresholds exactly as the pre-specified analysis (scripts/open_oil_analysis.py). The pre-specified mass detector
(experiment detection 11/14 = 0.786 < 0.80) stays PRIMARY; this detector was designed after the onset table had
been seen and is exploratory.

Rate r_s = (m_s - m_{s-1}) / (t_s - t_{s-1}) [mg/min] (reading number = run time in min); first loaded inspection
from (t_0, 0 mg): the ODM was reset to zero at the loaded-test start, t_0 = the tabulated run-in reading of that
experiment, else 60 min (1 h run-in, TM-2001-210936 p. 6). Sensitivity: t_0 = 0.
Outputs: outputs/diagnostics/f6_*.   Run: .venv/Scripts/python.exe -I scripts/diag_nasa_onset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir

import numpy as np
import pandas as pd
from scipy import stats as sps

from diag_common import NOTE, OIL, OUT
from wearpg import open_oil as oo
from wearpg.stats import cluster_bootstrap, paired_cluster_bootstrap
from wearpg.utils import save_json

DEFAULT_T0 = 60.0  # min, 1 h run-in before the loaded test (TM-2001-210936 p. 6; TM-2003-211307 p. 15)
DETECTORS = {"mass": ["mass_mg"], "rate": ["rate"], "mass_or_rate": ["mass_mg", "rate"]}
BAR = {"detection": 0.8, "false_alarm": 0.2}  # FRAMING C4 thresholds, as the pre-specified analysis
B = 2000


def run_in_t0() -> dict[int, float]:
    """Tabulated run-in reading per experiment (Tables 3.10-3.24, 'Run-in Wear' rows)."""
    t = oo._table(oo.NASA_TM2003_READINGS)
    r = t[t.description == "Run-in Wear"]
    return {int(e): float(v) for e, v in zip(r.exp, r.reading)}


def add_rate(rd: pd.DataFrame, t0: dict[int, float] | None) -> pd.DataFrame:
    """Interval debris rate [mg/min] between consecutive loaded inspections; t0 None -> origin 0 for every exp."""
    rd = rd.sort_values(["exp", "reading"]).reset_index(drop=True)
    g = rd.groupby("exp")
    origin = rd.exp.map(t0).fillna(DEFAULT_T0) if t0 is not None else pd.Series(0.0, index=rd.index)
    prev_t, prev_m = g.reading.shift().fillna(origin), g.mass_mg.shift().fillna(0.0)
    if not (rd.reading > prev_t).all():
        raise ValueError("non-increasing inspection times")
    return rd.assign(t_prev=prev_t, rate=(rd.mass_mg - prev_m) / (rd.reading - prev_t))


def loo_alarms(rd: pd.DataFrame, feats: list[str], taus: dict | None = None) -> np.ndarray:
    """Per held-out experiment, tau_f = open_oil.youden_threshold on the OTHER experiments' readings (neg = y 0,
    pos = y 1) for each feature; alarm if any feature >= its tau (one feature = the pre-specified rule).
    taus (optional dict) collects the fitted thresholds per feature."""
    alarm = np.zeros(len(rd), bool)
    y = rd.y.to_numpy()
    for e in rd.exp.unique():
        tr, te = (rd.exp != e).to_numpy(), (rd.exp == e).to_numpy()
        for f in feats:
            v = rd[f].to_numpy(float)
            t = oo.youden_threshold(v[tr & (y == 0)], v[tr & (y == 1)])
            alarm[te] |= v[te] >= t
            if taus is not None:
                taus.setdefault(f, []).append(t)
    return alarm


def per_experiment(r: pd.DataFrame) -> pd.DataFrame:
    """Experiment-level outcome exactly as open_oil_analysis.nasa()."""
    rows = []
    for e, g in r.groupby("exp"):
        g = g.sort_values("reading").reset_index(drop=True)
        has = bool(g.y.any())
        first_d = int(np.flatnonzero(g.y)[0]) if has else -1
        first_a = int(np.flatnonzero(g.alarm)[0]) if g.alarm.any() else -1
        rows.append({"exp": e, "destructive": has, "alarmed": bool(g.alarm.any()),
                     "detected": bool((g.alarm & (g.y == 1)).any()),
                     "early_alarm": bool(has and 0 <= first_a < first_d), "n_inspections": len(g)})
    return pd.DataFrame(rows)


def cp(k: int, n: int) -> dict:
    ci = sps.binomtest(k, n).proportion_ci(0.95, method="exact")
    return {"k": k, "n": n, "rate": k / n, "lo": float(ci.low), "hi": float(ci.high)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(OIL / "nasa_onset_readings.csv")
    res: dict = {"note": NOTE, "status": "exploratory", "primary": "pre-specified mass detector: detection 11/14 = "
                 "0.786 (< 0.80, against_us), false alarm 0/10 - unchanged and primary", "bar": BAR,
                 "t0_tabulated": run_in_t0(), "t0_default_min": DEFAULT_T0}
    all_rows, all_exp = [], []
    for variant in ("table_3_9", "table_3_14"):
        base = oo.nasa_readings(exp5_rdg2566=variant)
        base["y"] = (base.stage == 2).astype(int)
        for t0name, t0 in (("runin_t0", run_in_t0()), ("zero_t0", None)):
            rd = add_rate(base, t0)
            out = rd.assign(variant=variant, t0_rule=t0name)
            for det, feats in DETECTORS.items():
                taus: dict = {}
                out[f"alarm_{det}"] = a = loo_alarms(rd, feats, taus)
                pe = per_experiment(rd.assign(alarm=a))
                pos, neg = pe[pe.destructive], pe[~pe.destructive]
                key = f"{variant}|{t0name}|{det}"
                res[key] = {"detection": cp(int(pos.detected.sum()), len(pos)),
                            "false_alarm": cp(int(neg.alarmed.sum()), len(neg)),
                            "early_alarm_rate": float(pos.early_alarm.mean()),
                            "reading_sensitivity": float(a[rd.y == 1].mean()),
                            "reading_specificity": float(1 - a[rd.y == 0].mean()),
                            "tau_min_median_max": {f: [float(np.min(v)), float(np.median(v)), float(np.max(v))]
                                                   for f, v in taus.items()},
                            "missed_exps": pos.exp[~pos.detected].tolist(),
                            "false_alarm_exps": neg.exp[neg.alarmed].tolist(),
                            "meets_bar": bool(pos.detected.mean() >= BAR["detection"]
                                              and neg.alarmed.mean() <= BAR["false_alarm"])}
                all_exp.append(pe.assign(variant=variant, t0_rule=t0name, detector=det))
            # reading-level AUC of the rate (experiment-cluster bootstrap) and rate - mass paired difference
            auc = cluster_bootstrap(lambda d: oo.auc(d.loc[d.y == 0, "rate"], d.loc[d.y == 1, "rate"]), rd, "exp",
                                    n_boot=B, seed=0)["value"]
            pa = paired_cluster_bootstrap(lambda d: oo.auc(d.loc[d.y == 0, "s"], d.loc[d.y == 1, "s"]),
                                          rd.assign(s=rd.rate, key=np.arange(len(rd))),
                                          rd.assign(s=rd.mass_mg, key=np.arange(len(rd))), "exp", "key",
                                          n_boot=B, seed=0)["value"]
            res[f"{variant}|{t0name}|auc_rate"] = auc
            res[f"{variant}|{t0name}|auc_rate_minus_mass"] = pa
            # exact McNemar (destructive experiments) of each new detector vs the mass detector
            for det in ("rate", "mass_or_rate"):
                a = all_exp[-3 + list(DETECTORS).index(det)]
                m = all_exp[-3]
                j = a[a.destructive].merge(m[m.destructive], on="exp", suffixes=("", "_mass"))
                b_, c_ = int((j.detected & ~j.detected_mass).sum()), int((~j.detected & j.detected_mass).sum())
                res[f"{variant}|{t0name}|{det}"]["mcnemar_vs_mass"] = {
                    "new_only": b_, "mass_only": c_,
                    "p_exact": float(sps.binomtest(min(b_, c_), b_ + c_, 0.5).pvalue) if b_ + c_ else 1.0}
            all_rows.append(out)

    # reproduction of the frozen mass detector (table 3.9 alarms per reading)
    fz = frozen[(frozen.variant == "table_3_9") & (frozen.feature == "mass_mg")].sort_values(["exp", "reading"])
    mine = all_rows[0].sort_values(["exp", "reading"])
    res["reproduction_frozen_mass_alarms"] = {"n": len(fz), "equal": bool((fz.alarm.to_numpy()
                                                                           == mine.alarm_mass.to_numpy()).all())}
    pd.concat(all_rows).to_csv(OUT / "f6_onset_readings.csv", index=False)
    pd.concat(all_exp).to_csv(OUT / "f6_onset_per_experiment.csv", index=False)
    save_json(res, OUT / "f6_summary.json")
    for k, v in res.items():
        if isinstance(v, dict) and isinstance(v.get("detection"), dict):
            print(f"{k:38s} det {v['detection']['k']}/{v['detection']['n']} = {v['detection']['rate']:.3f} "
                  f"[{v['detection']['lo']:.2f}, {v['detection']['hi']:.2f}]  FA {v['false_alarm']['k']}/"
                  f"{v['false_alarm']['n']}  early {v['early_alarm_rate']:.2f}  missed {v['missed_exps']}  "
                  f"FA exps {v['false_alarm_exps']}  bar {v['meets_bar']}")
    for k in [k for k in res if "auc" in k]:
        print(k, {a: round(b, 3) for a, b in res[k].items()})
    print("reproduction:", res["reproduction_frozen_mass_alarms"])


if __name__ == "__main__":
    main()
