"""Grouped, leakage-free splits over runs (DESIGN sec. 0 rule 3).

A split assigns RUNS to train / val / test. Runs are grouped into independent units: connected components
of runs linked by a shared level unit (run, specimen or condition) OR a shared specimen_id, so runs on one
specimen never cross roles at any level. Image-domain filters (transfer protocol) act on the particle's
primary (view-0) image via `ds.particle_hierarchy()`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split

from wearpg.schema import WearDataset
from wearpg.utils import get_logger, load_json, save_json

_UNIT = {"run": "run_id", "specimen": "specimen_id", "condition": "condition_id"}


@dataclass
class Split:
    name: str
    protocol: str  # cv | external | transfer
    train_runs: list[str]
    val_runs: list[str]
    test_runs: list[str]
    meta: dict


def safe_name(s) -> str:
    """File-system safe name: every char outside [A-Za-z0-9._-] -> '-'."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", str(s))


def _py(v):
    """numpy scalar -> python scalar (JSON round-trip safe)."""
    return v.item() if isinstance(v, np.generic) else v


def _fmt(v) -> str:
    v = _py(v)
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def _is_interval(spec) -> bool:
    return isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 2


def _specs(values) -> list:
    """Role spec list in JSON-safe form: [lo, hi] intervals as float lists, exact values as python scalars."""
    return [[float(s[0]), float(s[1])] if _is_interval(s) else _py(s) for s in values]


def _pass_mask(values, specs) -> np.ndarray:
    """Vectorized `image_passes` over `values` for ANY of `specs`. NaN never passes."""
    values = pd.Series(np.asarray(values))
    m = np.zeros(len(values), dtype=bool)
    for s in specs:
        if _is_interval(s):
            v = pd.to_numeric(values, errors="coerce").to_numpy(float)
            m |= (v >= float(s[0])) & (v < float(s[1]))
        else:
            m |= (values == s).to_numpy(bool)
    return m


def image_passes(value, spec) -> bool:
    """True if `value` equals the exact-value `spec` or lies in the interval spec [lo, hi)."""
    return bool(_pass_mask([value], [spec])[0])


def _hierarchy_mask(h: pd.DataFrame, image_filter: dict, role: str) -> np.ndarray:
    col = image_filter["column"]
    if col not in h.columns:
        raise ValueError(f"filter column {col!r} is not a column of ds.particle_hierarchy()")
    return _pass_mask(h[col], image_filter["test" if role == "test" else "train"])


def particles_for_runs(ds: WearDataset, runs, image_filter: dict | None = None, role: str = "train") -> list[str]:
    """Particle ids of the given runs (particles-table order). With `image_filter` ({"column", "train",
    "test"}), keep particles whose primary image passes the role's specs; role "test" uses "test", any
    other role (train, val) uses "train"."""
    runs = set(runs)
    if image_filter is None:
        sids = ds.samples.loc[ds.samples.run_id.isin(runs), "sample_id"]
        return ds.particles.loc[ds.particles.sample_id.isin(sids), "particle_id"].tolist()
    h = ds.particle_hierarchy()
    keep = h.run_id.isin(runs).to_numpy() & _hierarchy_mask(h, image_filter, role)
    return h.loc[keep, "particle_id"].tolist()


def _unit_groups(rt: pd.DataFrame, level: str) -> np.ndarray:
    """Group id per run row: connected components of runs sharing the level unit or a specimen."""
    if level not in _UNIT:
        raise ValueError(f"level must be one of {tuple(_UNIT)}, got {level!r}")
    a = pd.factorize(rt[_UNIT[level]])[0]
    b = pd.factorize(rt["specimen_id"])[0] + a.max() + 1
    n = int(b.max()) + 1
    _, comp = connected_components(coo_matrix((np.ones(len(rt)), (a, b)), shape=(n, n)), directed=False)
    return comp[a]


def _group_strata(rt: pd.DataFrame, groups: np.ndarray, col: str | None) -> pd.Series | None:
    """group id -> modal value of `col` over the group's runs (ties -> smallest); None if col is None."""
    if col is None:
        return None
    return rt[col].astype(str).groupby(groups).agg(lambda s: s.mode().iloc[0])


def _carve_val(groups: np.ndarray, strata: pd.Series | None, val_fraction: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Split unique group ids into (train, val); val ~ val_fraction of groups (>= 1 if val_fraction > 0,
    leaving >= 1 train group), stratified by group stratum when every stratum is large enough."""
    n_val = min(max(int(round(val_fraction * len(groups))), int(val_fraction > 0)), len(groups) - 1)
    if n_val <= 0:
        return np.sort(groups), groups[:0]
    rs = int(rng.integers(2**31 - 1))
    try:
        tr, va = train_test_split(groups, test_size=n_val, random_state=rs,
                                  stratify=None if strata is None else strata.loc[groups].to_numpy())
    except ValueError:  # a stratum too small to stratify -> plain random carve
        tr, va = train_test_split(groups, test_size=n_val, random_state=rs)
    return np.sort(tr), np.sort(va)


def _runs(rt: pd.DataFrame, mask) -> list[str]:
    return sorted(rt.loc[np.asarray(mask, dtype=bool), "run_id"].tolist())


def grouped_kfold(ds: WearDataset, level: str = "specimen", n_splits: int = 5, seed: int = 0,
                  val_fraction: float = 0.2, stratify: str | None = "regime") -> list[Split]:
    """K-fold over unit groups (protocol "cv", names f"{level}_k{n_splits}_f{i}").

    StratifiedGroupKFold with each run labelled by its GROUP's modal `stratify` value (a `run_table()`
    column) when every stratum has >= n_splits groups, else shuffled GroupKFold (meta["stratified"]).
    Val groups are carved from each fold's train groups.
    """
    rt = ds.run_table()
    groups = _unit_groups(rt, level)
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"{n_groups} {level} groups < n_splits={n_splits}")
    strata = _group_strata(rt, groups, stratify)
    stratified = strata is not None and bool((strata.value_counts() >= n_splits).all())
    cv = (StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed) if stratified
          else GroupKFold(n_splits, shuffle=True, random_state=seed))
    y = strata.loc[groups].to_numpy() if stratified else np.zeros(len(rt))
    rng = np.random.default_rng(seed)
    out = []
    for i, (tr, te) in enumerate(cv.split(rt, y, groups)):
        tr_g, va_g = _carve_val(np.unique(groups[tr]), strata, val_fraction, rng)
        out.append(Split(
            name=f"{level}_k{n_splits}_f{i}", protocol="cv",
            train_runs=_runs(rt, np.isin(groups, tr_g)), val_runs=_runs(rt, np.isin(groups, va_g)),
            test_runs=_runs(rt, np.isin(np.arange(len(rt)), te)),
            meta={"level": level, "fold": i, "n_splits": n_splits, "seed": seed, "stratify": stratify,
                  "stratified": stratified, "val_fraction": val_fraction}))
    return out


def leave_one_group_out(ds: WearDataset, by: str = "material_id", seed: int = 0, val_fraction: float = 0.2) -> list[Split]:
    """One split per value of `by` (a `run_table()` column, e.g. material_id, load_N, condition_id): test =
    runs with that value (protocol "external", name f"logo_{by}_{value}", meta["held_out"]).

    Non-test runs sharing a specimen group with a test run are dropped (meta["dropped_runs"]); values that
    leave no training runs are skipped (logged). Val groups carved from the remaining specimen groups.
    """
    rt = ds.run_table()
    if by not in rt.columns:
        raise ValueError(f"unknown column {by!r}")
    groups = _unit_groups(rt, "specimen")
    strata = _group_strata(rt, groups, "regime")
    rng = np.random.default_rng(seed)
    out = []
    for v in sorted(rt[by].unique()):
        test = (rt[by] == v).to_numpy()
        blocked = np.isin(groups, groups[test])
        pool = ~blocked
        if not pool.any():
            get_logger().warning("leave_one_group_out: %s=%s leaves no training runs; skipped", by, v)
            continue
        tr_g, va_g = _carve_val(np.unique(groups[pool]), strata, val_fraction, rng)
        out.append(Split(
            name=safe_name(f"logo_{by}_{_fmt(v)}"), protocol="external",
            train_runs=_runs(rt, np.isin(groups, tr_g)), val_runs=_runs(rt, np.isin(groups, va_g)),
            test_runs=_runs(rt, test),
            meta={"level": "specimen", "by": by, "held_out": _py(v), "seed": seed, "val_fraction": val_fraction,
                  "dropped_runs": _runs(rt, blocked & ~test)}))
    return out


def transfer_split(ds: WearDataset, by: str, train_values, test_values, level: str = "specimen", seed: int = 0,
                   test_fraction: float = 0.3, val_fraction: float = 0.2) -> Split:
    """Domain-transfer split (protocol "transfer", name f"transfer_{by}_{train}_to_{test}", sanitized).

    `train_values` / `test_values` are lists of specs: an exact value or a [lo, hi) interval (e.g. focus_level
    [[0.0, 0.5]]). `by` is a `ds.particle_hierarchy()` column (image-level magnification, focus_level, ...).
    A run is eligible for a role iff >= 1 of its particles passes that role's filter. Unit groups are disjoint
    between the two sides: groups eligible for one role only go to that side; groups eligible for both are
    shuffled and `test_fraction` of them go to the test side. Val groups are carved from the train side.
    meta["image_filter"] = {"column": by, "train": [...], "test": [...]} (consumers filter particles with it).
    Raises ValueError if no train or no test runs result.
    """
    flt = {"column": by, "train": _specs(train_values), "test": _specs(test_values)}
    rt = ds.run_table()
    groups = _unit_groups(rt, level)
    h = ds.particle_hierarchy()
    ok_tr = rt.run_id.isin(h.loc[_hierarchy_mask(h, flt, "train"), "run_id"]).to_numpy()
    ok_te = rt.run_id.isin(h.loc[_hierarchy_mask(h, flt, "test"), "run_id"]).to_numpy()
    g_tr, g_te = set(groups[ok_tr]), set(groups[ok_te])
    rng = np.random.default_rng(seed)
    both = rng.permutation(sorted(g_tr & g_te))
    n_te = int(round(test_fraction * len(both)))
    if len(both) and not g_te - g_tr:
        n_te = max(n_te, 1)  # need >= 1 test group
    if len(both) and not g_tr - g_te:
        n_te = min(n_te, len(both) - 1)  # need >= 1 train group
    test_g = list((g_te - g_tr) | set(both[:n_te].tolist()))
    pool = ok_tr & np.isin(groups, list((g_tr - g_te) | set(both[n_te:].tolist())))
    test = ok_te & np.isin(groups, test_g)
    if not pool.any() or not test.any():
        raise ValueError(f"transfer_split on {by!r}: no {'train' if not pool.any() else 'test'} runs pass the filter")
    tr_g, va_g = _carve_val(np.unique(groups[pool]), _group_strata(rt, groups, "regime"), val_fraction, rng)

    def spec_str(specs):
        return "_".join(f"{_fmt(s[0])}-{_fmt(s[1])}" if _is_interval(s) else _fmt(s) for s in specs)

    return Split(
        name=safe_name(f"transfer_{by}_{spec_str(flt['train'])}_to_{spec_str(flt['test'])}"), protocol="transfer",
        train_runs=_runs(rt, pool & np.isin(groups, tr_g)), val_runs=_runs(rt, pool & np.isin(groups, va_g)),
        test_runs=_runs(rt, test),
        meta={"level": level, "by": by, "seed": seed, "test_fraction": test_fraction, "val_fraction": val_fraction,
              "image_filter": flt})


def audit_leakage(ds: WearDataset, split: Split) -> list[str]:
    """Leakage audit; empty list = clean.

    Reports unknown run ids; overlap across train/val/test of run_id and specimen_id (always) and
    condition_id (level "condition"), and of sample/image/particle ids (particles after the image filter);
    with meta["image_filter"]: runs holding no particle that passes their role's filter, and particles that
    pass BOTH the train and test filters (overlapping domains).
    """
    roles = {"train": split.train_runs, "val": split.val_runs, "test": split.test_runs}
    rt = ds.run_table().set_index("run_id")
    flt = split.meta.get("image_filter")
    errs = []
    unknown = sorted(set().union(*map(set, roles.values())) - set(rt.index))
    if unknown:
        errs.append(f"unknown run ids: {len(unknown)} (e.g. {unknown[:5]})")
    cols = ["run_id", "specimen_id"] + (["condition_id"] if split.meta.get("level") == "condition" else [])
    sets = {}
    for role, runs in roles.items():
        r = rt.loc[rt.index.intersection(list(runs))]
        s = ds.samples.loc[ds.samples.run_id.isin(set(runs)), "sample_id"]
        sets[role] = {"run_id": set(runs), "specimen_id": set(r.specimen_id), "condition_id": set(r.condition_id),
                      "sample_id": set(s), "image_id": set(ds.images.loc[ds.images.sample_id.isin(s), "image_id"]),
                      "particle_id": set(particles_for_runs(ds, runs, flt, role))}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for col in cols + ["sample_id", "image_id", "particle_id"]:
            ov = sorted(sets[a][col] & sets[b][col], key=str)
            if ov:
                errs.append(f"{col} overlap {a}/{b}: {len(ov)} (e.g. {ov[:5]})")
    if flt:
        h = ds.particle_hierarchy()
        for role, runs in roles.items():
            bad = sorted(set(runs) - set(h.loc[_hierarchy_mask(h, flt, role), "run_id"]))
            if bad:
                errs.append(f"{role} runs without a particle passing the {role} filter: {len(bad)} (e.g. {bad[:5]})")
        n_both = int((_hierarchy_mask(h, flt, "train") & _hierarchy_mask(h, flt, "test")).sum())
        if n_both:
            errs.append(f"train and test filters overlap: {n_both} particles pass both")
    return errs


def save_splits(splits: list[Split], path) -> Path:
    """Write splits as a JSON list of dicts."""
    return save_json([asdict(s) for s in splits], path)


def load_splits(path) -> list[Split]:
    return [Split(**d) for d in load_json(path)]
