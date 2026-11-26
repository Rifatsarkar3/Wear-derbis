"""Classical particle-classifier baselines on handcrafted descriptors (DESIGN sec. 9).

Every model is Pipeline(SimpleImputer(median), StandardScaler, model), fitted on TRAIN rows only, with
class_weight="balanced" (train.class_weighting) and hyper-parameters from cfg baselines.grids chosen by macro-F1 on
VALIDATION runs, never test folds. The selected candidate was itself fitted on the training rows, so it is the refit.
Probabilities always have one column per class index (0 for classes absent from training).

MultiViewSVM is a re-implementation in the spirit of ref. [8] of the planning review (multi-view moving-debris SVM
ensemble). Verify it against the original article before claiming equivalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FOURIER_FRACTAL_PREFIXES = ("efd_", "fractal_", "shape_")
CLASSICAL_BASELINES = ("fourier_fractal_svm", "fourier_fractal_cart", "multiview_svm")


def _classifier(kind: str, seed: int, class_weight: str | None, **params) -> Pipeline:
    if kind == "svm":
        model = SVC(kernel="rbf", probability=True, class_weight=class_weight, random_state=seed, **params)
    else:
        model = DecisionTreeClassifier(class_weight=class_weight, random_state=seed, **params)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def aligned_proba(model, X, n_classes: int) -> np.ndarray:
    """model.predict_proba(X) as (n, n_classes), column c = class index c (0 for classes absent from training)."""
    out = np.zeros((len(X), n_classes))
    out[:, np.asarray(model.classes_, dtype=np.int64)] = model.predict_proba(X)
    return out


def macro_f1(y_true, proba) -> float:
    """Macro-F1 of argmax(proba) over the classes present in y_true (as metrics.classification_metrics)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    return float(f1_score(y_true, np.argmax(proba, axis=1), labels=np.unique(y_true), average="macro", zero_division=0))


def select_on_val(fit, grid: dict | None, score) -> tuple[object, dict, float]:
    """Grid search on validation data.

    fit(**params) -> model fitted on the TRAINING rows; score(model) -> validation score (higher is better). Every
    grid point is tried in ParameterGrid order and the first best is kept. grid None or {} -> a single fit with the
    estimator defaults and score NaN. Returns (model, params, score).
    """
    best = (None, {}, -np.inf)
    for params in ParameterGrid(grid or {}):
        model = fit(**params)
        s = float(score(model)) if grid else float("nan")
        if best[0] is None or s > best[2]:
            best = (model, params, s)
    return best


def _labelled(X: np.ndarray, y) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    return X[y >= 0], y[y >= 0]


class DescriptorClassifier:
    """RBF SVC (probability=True) | CART on the descriptor columns that start with `prefixes`
    (default shape_* + efd_* + fractal_*: elliptic Fourier + fractal descriptors with absolute size via shape_*).

    Fitted attributes: columns_, model_, params_ (selected grid point, {} = estimator defaults), val_score_
    (validation macro-F1, NaN without validation labels).
    """

    def __init__(self, kind: str = "svm", prefixes=FOURIER_FRACTAL_PREFIXES, n_classes: int = 7, seed: int = 0,
                 grid: dict | None = None, class_weight: str | None = "balanced"):
        if kind not in ("svm", "cart"):
            raise ValueError(f"kind must be svm|cart, got {kind!r}")
        self.kind, self.prefixes, self.n_classes, self.seed = kind, tuple(prefixes), int(n_classes), int(seed)
        self.grid, self.class_weight = grid, class_weight

    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame | None = None, y_val=None) -> "DescriptorClassifier":
        """X: descriptor rows (DataFrame); y: class indices aligned with the rows (-1 = unlabelled, ignored).
        The grid is searched only when labelled validation rows exist."""
        self.columns_ = [c for c in X.columns if c.startswith(self.prefixes)]
        Xt, yt = _labelled(X[self.columns_].to_numpy(float), y)
        Xv, yv = (_labelled(X_val[self.columns_].to_numpy(float), y_val) if X_val is not None
                  else (None, np.zeros(0, np.int64)))
        self.model_, self.params_, self.val_score_ = select_on_val(
            lambda **p: _classifier(self.kind, self.seed, self.class_weight, **p).fit(Xt, yt),
            self.grid if len(yv) else None,
            lambda m: macro_f1(yv, aligned_proba(m, Xv, self.n_classes)))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """(n, n_classes) probabilities aligned to the class index."""
        return aligned_proba(self.model_, X[self.columns_].to_numpy(float), self.n_classes)


class MultiViewSVM:
    """One RBF SVC per view index k (k = 0 primary face-on view, k >= 1 extra views) on that view's descriptor columns
    with `prefixes` (shape_* + efd_* + fractal_*). A particle's probability is the mean (soft vote) over its available
    views whose index has a fitted SVC. A view index seen with fewer than 2 labelled classes gets no SVC. The grid is
    shared by all view indices and selected by the particle-level validation macro-F1.

    Fitted attributes: columns_, models_ {view index: Pipeline}, params_, val_score_.
    """

    def __init__(self, n_classes: int = 7, seed: int = 0, grid: dict | None = None,
                 class_weight: str | None = "balanced", prefixes=FOURIER_FRACTAL_PREFIXES):
        self.n_classes, self.seed, self.grid = int(n_classes), int(seed), grid
        self.class_weight, self.prefixes = class_weight, tuple(prefixes)

    def _views(self, view_desc: pd.DataFrame, y=None):
        """(X (n_v, F), particle ids (n_v,), view index (n_v,), labels (n_v,) or None) from view_id '{pid}_v{k}'."""
        parts = view_desc.index.to_series().str.rsplit("_v", n=1, expand=True)
        pid = parts[0].to_numpy()
        lab = None if y is None else pd.Series(y).reindex(pid).fillna(-1).to_numpy(np.int64)
        return view_desc[self.columns_].to_numpy(float), pid, parts[1].astype(int).to_numpy(), lab

    def _vote(self, models: dict, X, pid, k, particle_ids) -> np.ndarray:
        pos = pd.Index(particle_ids).get_indexer(pid)
        total, count = np.zeros((len(particle_ids), self.n_classes)), np.zeros(len(particle_ids))
        for v, model in models.items():
            m = (pos >= 0) & (k == v)
            if m.any():
                np.add.at(total, pos[m], aligned_proba(model, X[m], self.n_classes))
                np.add.at(count, pos[m], 1.0)
        if (count == 0).any():
            missing = np.asarray(particle_ids, dtype=object)[count == 0][:5].tolist()
            raise ValueError(f"no view with a fitted SVC for {int((count == 0).sum())} particles, e.g. {missing}")
        return total / count[:, None]

    def fit(self, view_desc: pd.DataFrame, y, view_desc_val: pd.DataFrame | None = None, y_val=None) -> "MultiViewSVM":
        """view_desc: per-view descriptors indexed by view_id '{particle_id}_v{view_idx}'; y: particle_id -> class index
        (Series or dict; missing or -1 = unlabelled). Validation views/labels likewise."""
        self.columns_ = [c for c in view_desc.columns if c.startswith(self.prefixes)]
        X, _, k, lab = self._views(view_desc, y)

        def fit_views(**params) -> dict:
            models = {}
            for v in np.unique(k):
                m = (k == v) & (lab >= 0)
                if len(np.unique(lab[m])) > 1:
                    models[int(v)] = _classifier("svm", self.seed, self.class_weight, **params).fit(X[m], lab[m])
            return models

        grid, score = None, None
        if view_desc_val is not None:
            Xv, pv, kv, labv = self._views(view_desc_val, y_val)
            ids = pd.unique(pv[labv >= 0])
            yv = pd.Series(y_val).loc[ids].to_numpy(np.int64)
            grid = self.grid if len(ids) else None
            score = lambda models: macro_f1(yv, self._vote(models, Xv, pv, kv, ids))  # noqa: E731
        self.models_, self.params_, self.val_score_ = select_on_val(fit_views, grid, score)
        return self

    def predict_proba(self, view_desc: pd.DataFrame, particle_ids) -> np.ndarray:
        """(len(particle_ids), n_classes) soft-vote probabilities in particle_ids order."""
        X, pid, k, _ = self._views(view_desc)
        return self._vote(self.models_, X, pid, k, list(particle_ids))


def make_particle_baseline(name: str, n_classes: int, seed: int, cfg: dict):
    """Classical particle baseline by config name (CLASSICAL_BASELINES) with cfg baselines.grids and
    train.class_weighting. Deep baselines are built by deep.make_deep_baseline."""
    grids = cfg["baselines"]["grids"]
    cw = "balanced" if cfg["train"].get("class_weighting", "balanced") == "balanced" else None
    if name == "fourier_fractal_svm":
        return DescriptorClassifier("svm", n_classes=n_classes, seed=seed, grid=grids.get("svm"), class_weight=cw)
    if name == "fourier_fractal_cart":
        return DescriptorClassifier("cart", n_classes=n_classes, seed=seed, grid=grids.get("cart"), class_weight=cw)
    if name == "multiview_svm":
        return MultiViewSVM(n_classes, seed, grids.get("svm"), cw)
    raise ValueError(f"unknown classical baseline {name!r}; expected one of {CLASSICAL_BASELINES}")
