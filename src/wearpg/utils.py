"""Small shared helpers: seeding, device, JSON io, stable hashing, logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch optional for pure-data utilities
        pass


def get_device(pref: str = "auto"):
    import torch

    if pref in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default, allow_nan=True), encoding="utf-8")
    return path


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_hash(obj, n: int = 12) -> str:
    """Deterministic short hash of a JSON-able object (cache keys)."""
    blob = json.dumps(obj, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:n]


def get_logger(name: str = "wearpg") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log
