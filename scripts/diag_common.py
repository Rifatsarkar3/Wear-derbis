"""Shared helpers of the DIAG diagnostics (scripts/diag_*.py; decision rules in outputs/agent_logs/DIAG.md).

Every diagnostic is NEW and labelled post_hoc_diagnostic / exploratory / sensitivity; frozen pre-specified verdicts
(ODI.md, ODA.md) are never recomputed or overwritten. Frozen scripts are imported read-only (load_script) so that
the same scoring code is reused; group_spearman reproduces the statistic of open_ferro_robust.stability().
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "diagnostics"
FERRO = ROOT / "outputs" / "open_data" / "ferro"
OIL = ROOT / "outputs" / "open_data" / "oil"
ZEN = ROOT / "data" / "real" / "zenodo_ferro" / "extracted"
NOTE = "NEW analysis (DIAG); frozen pre-specified verdicts unchanged; rules: outputs/agent_logs/DIAG.md"


def load_script(name: str):
    """Import scripts/<name>.py as a module without running its main() (read-only reuse of frozen code)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # registered so worker processes can unpickle functions defined in it
    spec.loader.exec_module(mod)
    return mod


def feature_group(f: str) -> str:
    """Descriptor group exactly as open_ferro_robust.stability() assigns it."""
    return f.split("_")[0] if not f.startswith("graph_") else ("graph_tex" if "_tex_" in f else "graph_geom")


def group_features(df: pd.DataFrame, group: str) -> list[str]:
    feats = [c for c in df.columns if c.startswith(("shape_", "efd_", "fractal_", "graph_"))]
    return [f for f in feats if feature_group(f) == group]


def wide_pairs(df: pd.DataFrame, mode: str, transform: str, feats: list[str], particles=None) -> pd.DataFrame:
    """One row per particle present (not lost) at identity and at `transform` in `mode` (stability(): reference =
    identity rows not lost): columns particle, image_id, ref__<f>, tr__<f>."""
    g = df[df["mode"] == mode]
    lost = g["lost"].astype(bool)
    ref = g[(g["transform"] == "identity") & ~lost].set_index("particle")
    tr = g[(g["transform"] == transform) & ~lost].set_index("particle")
    ok = tr.index.intersection(ref.index)
    if particles is not None:
        ok = ok.intersection(pd.Index(particles))
    return pd.concat([ref.loc[ok, ["image_id"]], ref.loc[ok, feats].add_prefix("ref__"),
                      tr.loc[ok, feats].add_prefix("tr__")], axis=1).rename_axis("particle").reset_index()


def group_spearman(wide: pd.DataFrame) -> float:
    """Median over the features of Spearman(ref, tr) across rows; a feature is NaN when n < 3 or either side is
    constant (as stability()); NaN if no feature is finite."""
    ref = [c for c in wide.columns if c.startswith("ref__")]
    if len(wide) < 3 or not ref:
        return float("nan")
    a = wide[ref].to_numpy(float)
    b = wide[["tr__" + c[5:] for c in ref]].to_numpy(float)
    ok = (np.ptp(a, axis=0) > 0) & (np.ptp(b, axis=0) > 0)  # NaN ptp -> False, as spearmanr's NaN result
    if not ok.any():
        return float("nan")
    ra, rb = sps.rankdata(a[:, ok], axis=0), sps.rankdata(b[:, ok], axis=0)  # average ranks = scipy spearmanr
    ra, rb = ra - ra.mean(axis=0), rb - rb.mean(axis=0)
    rho = (ra * rb).sum(axis=0) / np.sqrt((ra * ra).sum(axis=0) * (rb * rb).sum(axis=0))
    return float(np.median(rho))
