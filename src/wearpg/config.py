"""YAML config loading with `base:` inheritance and dotted overrides (`model.hidden=64`)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into a copy of `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b.c=value` strings; values are YAML-parsed (numbers, lists, bools)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config; `base: other.yaml` (relative to this file) is merged underneath."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return apply_overrides(cfg, overrides)
