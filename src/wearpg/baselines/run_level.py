"""Run- and sample-level wear baselines (DESIGN sec. 9).

Every baseline has fit(ds_train, descriptors, ds_val=None) and predict(ds_test, descriptors) -> (samples_df, runs_df)
holding the value columns of schema.SAMPLE_PRED_COLUMNS / RUN_PRED_COLUMNS (the caller attaches keys, ids, targets):
    samples: sample_id, severity_pred (argmax sev_p_*, -1 if not predicted), sev_p_mild/transition/severe,
             log10_rate_pred (interval k [mm^3/(N m)]; NaN for cumulative runs), log_debris_volume_mm3
    runs:    run_id, log10_rate_pred, log_debris_volume_mm3 (ln mm^3), coverage, note
Rows follow ds.samples / ds.runs, with NaN where a baseline cannot predict.

`descriptors` = particle descriptors (index particle_id). Its rows that belong to ds are the OBSERVED particles, so pass
the role's surviving particles (e.g. after a transfer image filter). A sample is observed when it has >= 1 of them.
Sample outputs exist only for observed samples, as for the proposed model, and coverage is computed over the observed
samples exactly as in DESIGN sec. 3. Fitting uses what the proposed model trains on: run targets of runs with coverage
>= min_coverage (physics.min_run_coverage) and sample targets of observed samples. Targets: run log10
wear_rate_mm3_Nm; sample log10 interval k = interval_wear_volume_mm3 / (F ds) (interval runs only); severity >= 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wearpg.baselines.classical import aligned_proba, macro_f1, select_on_val
from wearpg.schema import SEVERITY_LEVELS, SEVERITY_PROB_COLUMNS, UM3_PER_MM3

RUN_BASELINES = ("physics_only", "conditions_only", "physics_conditions", "descriptor_agg_rf", "descriptor_agg_ridge")
NOTE_ETA = "eta not transferable"
LN10 = float(np.log(10.0))
K_SEV = len(SEVERITY_LEVELS)
_AGG_PREFIXES = ("shape_", "fractal_", "tex_", "color_", "mv_")  # efd_ left out: means of normalised harmonics
# Ridge(alpha=1) on standardized inputs, not RidgeCV: sklearn 1.8.0's efficient-LOO RidgeCV returns wrong coefficients
# on balanced factorial designs (orthogonal equal-norm columns, e.g. 1.38 instead of 2.0), which wear tests are.


def _log(x) -> np.ndarray:
    """Natural log of positive values, NaN elsewhere (no warnings)."""
    x = np.asarray(x, dtype=float)
    return np.log(np.where(x > 0, x, np.nan))


class OrdinalLogistic(ClassifierMixin, BaseEstimator):
    """Proportional-odds (cumulative logit) model P(y <= j | x) = sigmoid(theta_j - x @ beta), j = 0..n_levels-2.

    Maximum penalised likelihood by L-BFGS with the analytic gradient. The penalty 0.5 alpha |beta|^2 + 0.5e-4 |t|^2
    (t = threshold parameters, theta = cumsum(t_0, exp(t_1), ...) keeps them ordered) keeps the fit finite under
    separation and for levels absent from training. Use behind StandardScaler. classes_ = 0..n_levels-1 always.
    """

    def __init__(self, alpha: float = 1.0, n_levels: int = K_SEV):
        self.alpha = alpha
        self.n_levels = n_levels

    def _nll(self, w, X, y):
        p, r = X.shape[1], np.arange(len(y))
        beta = w[:p]
        theta = np.cumsum(np.r_[w[p], np.exp(w[p + 1:])])
        cdf = np.pad(expit(theta[None, :] - (X @ beta)[:, None]), ((0, 0), (1, 1)), constant_values=(0.0, 1.0))
        dens = np.pad(cdf[:, 1:-1] * (1.0 - cdf[:, 1:-1]), ((0, 0), (1, 1)))  # d cdf / d z, 0 at the open ends
        prob = np.maximum(cdf[r, y + 1] - cdf[r, y], 1e-300)
        hi, lo = dens[r, y + 1] / prob, dens[r, y] / prob
        g_cut = np.zeros_like(cdf)
        np.add.at(g_cut, (r, y + 1), -hi)
        np.add.at(g_cut, (r, y), lo)
        g_theta = g_cut[:, 1:-1].sum(axis=0)  # d nll / d theta_j
        g_t = np.cumsum(g_theta[::-1])[::-1] * np.r_[1.0, np.exp(w[p + 1:])]  # chain rule through cumsum / exp
        t = w[p:]
        nll = -np.log(prob).sum() + 0.5 * self.alpha * beta @ beta + 0.5e-4 * t @ t
        return nll, np.r_[X.T @ (hi - lo) + self.alpha * beta, g_t + 1e-4 * t]

    def fit(self, X, y) -> "OrdinalLogistic":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=np.int64)
        w = minimize(self._nll, np.zeros(X.shape[1] + self.n_levels - 1), args=(X, y), jac=True,
                     method="L-BFGS-B").x
        p = X.shape[1]
        self.coef_, self.thresholds_ = w[:p], np.cumsum(np.r_[w[p], np.exp(w[p + 1:])])
        self.classes_ = np.arange(self.n_levels)
        return self

    def predict_proba(self, X) -> np.ndarray:
        z = self.thresholds_[None, :] - (np.asarray(X, dtype=float) @ self.coef_)[:, None]
        return np.diff(np.pad(expit(z), ((0, 0), (1, 1)), constant_values=(0.0, 1.0)), axis=1)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def _primary_area_um2(ds) -> pd.Series:
    """particle_id -> shoelace area [um^2] of the primary (view_idx 0) contour."""
    v = ds.views[ds.views.view_idx == 0]
    n = np.fromiter((len(c) for c in v.contour_x), np.int64, len(v))
    if not len(v) or (n == 0).any():  # empty contours (invalid data): per-view fallback, area 0
        a = [0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) if len(x) else 0.0
             for x, y in zip(v.contour_x, v.contour_y)]
    else:  # vectorized shoelace over all contours (one concatenation; next vertex wraps within each contour)
        x, y = np.concatenate(list(v.contour_x)), np.concatenate(list(v.contour_y))
        start = np.cumsum(n) - n
        nxt = np.arange(len(x)) + 1
        nxt[start + n - 1] = start
        a = 0.5 * np.abs(np.add.reduceat(x * y[nxt] - y * x[nxt], start))
    return pd.Series(np.asarray(a, dtype=float) * v.um_per_px.to_numpy(float) ** 2, index=v.particle_id.to_numpy())


def aggregate_sample_features(ds, descriptors: pd.DataFrame, particle_ids=None) -> pd.DataFrame:
    """Debris aggregates per sample (index sample_id, ds.samples order); no operating conditions.

    Used particles = particle_ids (default: the ds particles present in descriptors.index). Columns:
      n_used                    number of used particles (0 -> every other column NaN)
      log_n_total, log_obs_fraction    ln n_particles_total, ln observation_fraction f_s
      log_volume_index          ln(sum_table A^1.5 * n_particles_total / n_table / f_s) [ln um^3]. This is the
                                area-ratio HT estimator of DESIGN sec. 3: the bag cancels out, so it uses the
                                primary-view areas A [um^2] (from ds.views) of all n_table particle rows of the sample.
      q10/q50/q90_log_ecd, log_total_area   quantiles of shape_log_ecd_um, ln sum shape_area_um2 of the used particles
      mean_<col>                mean of every descriptor column with a prefix in shape_/fractal_/tex_/color_/mv_
    """
    p = ds.particles.set_index("particle_id")
    used = descriptors.index.intersection(p.index) if particle_ids is None else pd.Index(particle_ids)
    d = descriptors.loc[used]
    g = d.groupby(p.loc[used, "sample_id"].to_numpy())
    ecd = g["shape_log_ecd_um"]
    out = pd.DataFrame({"n_used": g.size(), "q10_log_ecd": ecd.quantile(0.1), "q50_log_ecd": ecd.quantile(0.5),
                        "q90_log_ecd": ecd.quantile(0.9), "log_total_area": np.log(g["shape_area_um2"].sum())})
    out = out.join(g[[c for c in d.columns if c.startswith(_AGG_PREFIXES)]].mean().add_prefix("mean_"))
    s = ds.samples.set_index("sample_id")
    out = out.reindex(s.index)
    table = (_primary_area_um2(ds) ** 1.5).groupby(p["sample_id"]).agg(["sum", "size"]).reindex(s.index)
    n_total, f = s.n_particles_total.to_numpy(float), s.observation_fraction.to_numpy(float)
    out.insert(1, "log_n_total", _log(n_total))
    out.insert(2, "log_obs_fraction", _log(f))
    out.insert(3, "log_volume_index", _log(table["sum"].to_numpy() * n_total / table["size"].to_numpy() / f))
    obs = out.n_used.fillna(0).to_numpy() > 0
    out.loc[~obs, out.columns[1:]] = np.nan
    out["n_used"] = out.n_used.fillna(0).astype(np.int64)
    return out


def _per_run(s: pd.DataFrame, rt: pd.DataFrame, log_d) -> tuple[np.ndarray, np.ndarray]:
    """(ln V_r [mm^3], coverage) per row of rt over the observed samples (s.obs), DESIGN sec. 3; log_d = ln D_s (n_s,).
    interval: V_r = sum V_s * L / sum ds_s, coverage = sum ds_s / L;
    cumulative: V_r = (M_last + sum_{s<last} q_s M_s) * L / s_end(last), coverage = s_end(last) / L.
    Runs without observed samples -> (NaN, 0)."""
    o = s.obs.to_numpy()
    r, t, cum = s.run_pos.to_numpy()[o], s.time_s.to_numpy(float)[o], s.cumulative.to_numpy()[o]
    t_last = np.full(len(rt), -np.inf)
    np.maximum.at(t_last, r, t)
    last = t >= t_last[r]
    w = np.exp(np.asarray(log_d, dtype=float)[o]) * np.where(cum & ~last, s.q.to_numpy(float)[o], 1.0)
    covered = np.where(cum, np.where(last, s.sliding_distance_end_m.to_numpy(float)[o], 0.0), s.ds_m.to_numpy(float)[o])
    cov = np.bincount(r, covered, len(rt)) / rt.sliding_distance_m.to_numpy(float)
    return np.where(cov > 0, _log(np.bincount(r, w, len(rt))) - _log(cov), np.nan), cov


def _tables(ds, descriptors, min_coverage: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(samples, aggregates, runs) of ds. samples = ds.samples + run attributes + run_pos, ds_m [m], q, cumulative, obs,
    log10_k (interval target), fit_rate / fit_sev masks; aggregates = aggregate_sample_features (same rows);
    runs = ds.run_table() + coverage, log10_k, fit_rate mask (finite target and coverage >= min_coverage)."""
    rt = ds.run_table()
    agg = aggregate_sample_features(ds, descriptors)
    s = ds.samples.merge(rt[["run_id", "load_N", "duration_s", "reservoir_volume_ml", "debris_sampling"]],
                         on="run_id", how="left")
    s["run_pos"] = pd.Index(rt.run_id).get_indexer(s.run_id)
    s["ds_m"] = s.sliding_distance_end_m - s.sliding_distance_start_m
    s["q"] = s.sampled_oil_ml / s.reservoir_volume_ml
    s["cumulative"] = s.debris_sampling == "cumulative"
    s["obs"] = agg.n_used.to_numpy() > 0
    s["log10_k"] = np.where(s.cumulative, np.nan, _log(s.interval_wear_volume_mm3 / (s.load_N * s.ds_m)) / LN10)
    s["fit_rate"] = s.obs & np.isfinite(s.log10_k)
    s["fit_sev"] = s.obs & (s.severity >= 0)
    rt["coverage"] = _per_run(s, rt, np.zeros(len(s)))[1]
    rt["log10_k"] = _log(rt.wear_rate_mm3_Nm) / LN10
    rt["fit_rate"] = np.isfinite(rt.log10_k) & (rt.coverage >= min_coverage)
    return s, agg, rt


def _fit(model, X, y, rows):
    """model fitted on the rows (bool) of X, y; None when there is no row."""
    rows = np.asarray(rows, dtype=bool)
    return model.fit(np.asarray(X, dtype=float)[rows], np.asarray(y)[rows]) if rows.any() else None


def _values(model, X, rows) -> np.ndarray:
    rows, out = np.asarray(rows, dtype=bool), np.full(len(rows), np.nan)
    if model is not None and rows.any():
        out[rows] = model.predict(np.asarray(X, dtype=float)[rows])
    return out


def _proba(model, X, rows) -> np.ndarray:
    rows, out = np.asarray(rows, dtype=bool), np.full((len(rows), K_SEV), np.nan)
    if model is not None and rows.any():
        out[rows] = aligned_proba(model, np.asarray(X, dtype=float)[rows], K_SEV)
    return out


def _outputs(s, rt, sev_p, rate_s, log_d_s, rate_r, log_v_r, note) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = np.isfinite(sev_p).all(axis=1)
    samples = pd.DataFrame({
        "sample_id": s.sample_id.to_numpy(),
        "severity_pred": np.where(ok, np.nan_to_num(sev_p).argmax(axis=1), -1).astype(np.int64),
        **{c: sev_p[:, i] for i, c in enumerate(SEVERITY_PROB_COLUMNS)},
        "log10_rate_pred": rate_s, "log_debris_volume_mm3": log_d_s})
    runs = pd.DataFrame({"run_id": rt.run_id.to_numpy(), "log10_rate_pred": rate_r, "log_debris_volume_mm3": log_v_r,
                         "coverage": rt.coverage.to_numpy(float), "note": note})
    return samples, runs


class PhysicsOnlyBaseline:
    """Debris + load and sliding distance only, no learned debris model (DESIGN sec. 3):
        v_i = 0.3 A_i^1.5 [um^3] (A = primary-view area), metal fraction 1, area-ratio HT estimator:
        ln D_s = ln 0.3 + log_volume_index_s - ln eta_r - ln 1e9   [ln mm^3]
    Run volume and coverage use the interval / cumulative estimators; log10 k = log10(V / (F s)).
    ln eta_r = theta[magnetic class] + delta[material]. theta = median, over TRAIN runs of the class with coverage >=
    min_coverage, of ln(V_debris(eta = 1) / V_grav) (V_grav = ds.grav_volume_mm3()). delta = median offset of the
    material within its class (0 for materials unseen in training). When no training material shares the run's
    magnetic class, every output of the run is NaN with note "eta not transferable".
    Severity: OrdinalLogistic on log10 of the interval debris rate k_s = D_s / (F ds_s), fitted on TRAIN interval
    samples; cumulative runs have no interval rate, so their rate and severity are NaN.
    Fitted: theta_ {0 non-ferrous | 1 ferrous: ln eta}, delta_ {material: offset}, severity_ (None without data).
    """

    def __init__(self, min_coverage: float = 0.5):
        self.min_coverage = float(min_coverage)

    def _debris(self, ds, descriptors):
        s, agg, rt = _tables(ds, descriptors, self.min_coverage)
        return s, rt, np.log(0.3) + agg.log_volume_index.to_numpy() - np.log(UM3_PER_MM3)  # ln D_s at eta = 1

    def _log_eta(self, rt) -> np.ndarray:
        """ln eta per run: theta[class] + delta[material] (0 if unseen); NaN when the class is unseen."""
        return (rt.ferromagnetic.astype(int).map(self.theta_)
                + rt.material_id.map(self.delta_).fillna(0.0)).to_numpy(float)

    @staticmethod
    def _log10_k(s, log_d) -> np.ndarray:
        return np.where(s.cumulative, np.nan, (log_d - _log(s.load_N * s.ds_m)) / LN10)

    def fit(self, ds_train, descriptors, ds_val=None) -> "PhysicsOnlyBaseline":
        """ds_val is unused (nothing to tune)."""
        s, rt, log_w = self._debris(ds_train, descriptors)
        resid = pd.Series(_per_run(s, rt, log_w)[0] - _log(ds_train.grav_volume_mm3().reindex(rt.run_id)))
        ok = np.isfinite(resid) & (rt.coverage >= self.min_coverage)
        cls = rt.ferromagnetic.astype(int)
        self.theta_ = resid[ok].groupby(cls[ok]).median().to_dict()
        self.delta_ = (resid - cls.map(self.theta_))[ok].groupby(rt.material_id[ok]).median().to_dict()
        x = self._log10_k(s, log_w - self._log_eta(rt)[s.run_pos.to_numpy()])
        self.severity_ = _fit(make_pipeline(StandardScaler(), OrdinalLogistic()), x[:, None], s.severity,
                              s.fit_sev & np.isfinite(x))
        return self

    def predict(self, ds_test, descriptors) -> tuple[pd.DataFrame, pd.DataFrame]:
        s, rt, log_w = self._debris(ds_test, descriptors)
        log_eta = self._log_eta(rt)
        log_d = log_w - log_eta[s.run_pos.to_numpy()]
        log_v = _per_run(s, rt, log_d)[0]
        x = self._log10_k(s, log_d)
        return _outputs(s, rt, _proba(self.severity_, x[:, None], np.isfinite(x)), x, log_d,
                        (log_v - _log(rt.load_N * rt.sliding_distance_m)) / LN10, log_v,
                        np.where(np.isnan(log_eta), NOTE_ETA, ""))


class ConditionsOnlyBaseline:
    """Operating conditions only, no debris (the shortcut reference): Ridge(alpha=1) on standardized [material one-hot
    (training materials; unseen -> all 0), ln load, ln speed, ln hardness, ln sliding distance] -> run log10 k.
    Severity: OrdinalLogistic on the same + time_s / duration_s. No sample rate and no debris volume (NaN).
    Fitted: materials_, rate_, severity_.
    """

    def __init__(self, min_coverage: float = 0.5):
        self.min_coverage = float(min_coverage)

    def _x(self, rt) -> np.ndarray:
        onehot = (rt.material_id.to_numpy()[:, None] == np.asarray(self.materials_, dtype=object)[None, :])
        return np.column_stack([onehot.astype(float), _log(rt.load_N), _log(rt.speed_m_s), _log(rt.hardness_MPa),
                                _log(rt.sliding_distance_m)])

    @staticmethod
    def _xs(s, xr) -> np.ndarray:
        return np.column_stack([xr[s.run_pos.to_numpy()], (s.time_s / s.duration_s).to_numpy(float)])

    def fit(self, ds_train, descriptors, ds_val=None) -> "ConditionsOnlyBaseline":
        """ds_val is unused (nothing to tune)."""
        s, _, rt = _tables(ds_train, descriptors, self.min_coverage)
        self.materials_ = sorted(rt.material_id.unique())
        xr = self._x(rt)
        self.rate_ = _fit(make_pipeline(StandardScaler(), Ridge(alpha=1.0)), xr, rt.log10_k, rt.fit_rate)
        self.severity_ = _fit(make_pipeline(StandardScaler(), OrdinalLogistic()), self._xs(s, xr), s.severity,
                              s.fit_sev)
        return self

    def predict(self, ds_test, descriptors) -> tuple[pd.DataFrame, pd.DataFrame]:
        s, _, rt = _tables(ds_test, descriptors, self.min_coverage)
        xr, nan_s, nan_r = self._x(rt), np.full(len(s), np.nan), np.full(len(rt), np.nan)
        return _outputs(s, rt, _proba(self.severity_, self._xs(s, xr), s.obs), nan_s, nan_s,
                        _values(self.rate_, xr, np.ones(len(rt), bool)), nan_r, "")


class PhysicsConditionsBaseline:
    """Debris + known operating conditions (DESIGN sec. 15): minimum-variance fusion of the physics_only (debris) and
    conditions_only run log10 k, x = w x_debris + (1 - w) x_conditions with
        w = (S_cc - S_pc) / (S_pp + S_cc - 2 S_pc), clipped to [0, 1],
    S = second moments of the two components' errors (S_pp = mean e_p^2, S_pc = mean e_p e_c, ...) on cross-fitted
    predictions of the TRAIN runs (inner K-fold over specimens, K = min(inner_folds, n specimens); scored runs only).
    With uncorrelated errors this is inverse-variance weighting. Both components are then refit on all train runs.
    Sample outputs (severity, interval rate, debris volume), the run debris volume and the note are physics_only's;
    the fused run rate is NaN where either component is. ds_val is unused (as for both components).
    Fitted: weight_ (w of the debris estimate; 0.5 with < 3 cross-fitted pairs), moments_, physics_, conditions_.
    """

    def __init__(self, min_coverage: float = 0.5, inner_folds: int = 5, seed: int = 0):
        self.min_coverage, self.inner_folds, self.seed = float(min_coverage), int(inner_folds), int(seed)

    def _components(self, ds_train, descriptors):
        return (PhysicsOnlyBaseline(self.min_coverage).fit(ds_train, descriptors),
                ConditionsOnlyBaseline(self.min_coverage).fit(ds_train, descriptors))

    def fit(self, ds_train, descriptors, ds_val=None) -> "PhysicsConditionsBaseline":
        rt = ds_train.run_table()
        spec = np.random.default_rng(self.seed).permutation(sorted(rt.specimen_id.unique()))
        k = min(self.inner_folds, len(spec))
        fold = rt.specimen_id.map(dict(zip(spec, np.arange(len(spec)) % k))).to_numpy()
        pairs = []
        for f in range(k if k >= 2 else 0):
            inner, held = rt.run_id[fold != f].tolist(), rt.run_id[fold == f].tolist()
            phys, cond = self._components(ds_train.subset(inner), descriptors)
            test = ds_train.subset(held)
            xp, xc = (m.predict(test, descriptors)[1].set_index("run_id").log10_rate_pred for m in (phys, cond))
            _, _, t = _tables(test, descriptors, self.min_coverage)
            t = t.set_index("run_id")
            pairs.append(pd.DataFrame({"p": xp, "c": xc, "y": t.log10_k.where(t.fit_rate)}))
        e = pd.concat(pairs).dropna() if pairs else pd.DataFrame(columns=["p", "c", "y"])
        ep, ec = (e.p - e.y).to_numpy(float), (e.c - e.y).to_numpy(float)
        self.moments_ = {"S_pp": float(np.mean(ep**2)) if len(e) else np.nan,
                         "S_cc": float(np.mean(ec**2)) if len(e) else np.nan,
                         "S_pc": float(np.mean(ep * ec)) if len(e) else np.nan, "n": int(len(e))}
        den = self.moments_["S_pp"] + self.moments_["S_cc"] - 2 * self.moments_["S_pc"]
        self.weight_ = (float(np.clip((self.moments_["S_cc"] - self.moments_["S_pc"]) / den, 0.0, 1.0))
                        if len(e) >= 3 and den > 0 else 0.5)
        self.physics_, self.conditions_ = self._components(ds_train, descriptors)
        return self

    def predict(self, ds_test, descriptors) -> tuple[pd.DataFrame, pd.DataFrame]:
        samples, runs = self.physics_.predict(ds_test, descriptors)
        xc = self.conditions_.predict(ds_test, descriptors)[1].log10_rate_pred.to_numpy(float)
        runs["log10_rate_pred"] = self.weight_ * runs.log10_rate_pred.to_numpy(float) + (1 - self.weight_) * xc
        return samples, runs


class DescriptorAggregateBaseline:
    """Debris aggregates only (aggregate_sample_features, no operating conditions). Run log10 k comes from the mean of
    the run's observed-sample aggregates; interval log10 k and severity come from the sample aggregates.
    kind "rf": RandomForestRegressor / RandomForestClassifier(class_weight="balanced"), each with the cfg
    baselines.grids.rf point that is best on the validation data (MAE resp. macro-F1; sklearn defaults without
    ds_val). kind "ridge": Ridge(alpha=1) on standardized inputs and OrdinalLogistic for severity.
    No debris-volume output (NaN). Fitted: columns_, models_ / params_ {"run", "sample_rate", "severity"}.
    """

    def __init__(self, kind: str = "rf", seed: int = 0, grid: dict | None = None, min_coverage: float = 0.5):
        if kind not in ("rf", "ridge"):
            raise ValueError(f"kind must be rf|ridge, got {kind!r}")
        self.kind, self.seed, self.grid, self.min_coverage = kind, int(seed), grid, float(min_coverage)

    def _make(self, clf: bool, **params):
        if self.kind == "rf":
            est = (RandomForestClassifier(class_weight="balanced", random_state=self.seed, **params) if clf
                   else RandomForestRegressor(random_state=self.seed, **params))
            return make_pipeline(SimpleImputer(strategy="median"), est)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             OrdinalLogistic() if clf else Ridge(alpha=1.0))

    def _design(self, s, agg, rt):
        """(samples, runs, sample features, run features) from _tables(), columns in columns_ order."""
        xs = agg.drop(columns="n_used").reindex(columns=self.columns_)
        xr = xs[s.obs.to_numpy()].groupby(s.run_id[s.obs].to_numpy()).mean().reindex(rt.run_id)
        return s, rt, xs.to_numpy(float), xr.to_numpy(float)

    @staticmethod
    def _tasks(s, rt, xs, xr) -> dict:
        """task -> (X, y, training rows)."""
        return {"run": (xr, rt.log10_k.to_numpy(float), rt.fit_rate.to_numpy()),
                "sample_rate": (xs, s.log10_k.to_numpy(float), s.fit_rate.to_numpy()),
                "severity": (xs, s.severity.to_numpy(np.int64), s.fit_sev.to_numpy())}

    def fit(self, ds_train, descriptors, ds_val=None) -> "DescriptorAggregateBaseline":
        s, agg, rt = _tables(ds_train, descriptors, self.min_coverage)
        self.columns_ = list(agg.columns[1:])
        train = self._tasks(*self._design(s, agg, rt))
        val = ({} if ds_val is None else
               self._tasks(*self._design(*_tables(ds_val, descriptors, self.min_coverage))))
        self.models_, self.params_ = {}, {}
        for task, (X, y, rows) in train.items():
            if not rows.any():
                self.models_[task], self.params_[task] = None, {}
                continue
            clf = task == "severity"
            Xv, yv, rv = val.get(task, (None, None, np.zeros(0, bool)))
            if clf:
                score = lambda m: macro_f1(yv[rv], aligned_proba(m, Xv[rv], K_SEV))  # noqa: E731
            else:
                score = lambda m: -float(np.mean(np.abs(m.predict(Xv[rv]) - yv[rv])))  # noqa: E731
            self.models_[task], self.params_[task], _ = select_on_val(
                lambda **p: self._make(clf, **p).fit(X[rows], y[rows]),
                self.grid if self.kind == "rf" and rv.any() else None, score)
        return self

    def predict(self, ds_test, descriptors) -> tuple[pd.DataFrame, pd.DataFrame]:
        s, rt, xs, xr = self._design(*_tables(ds_test, descriptors, self.min_coverage))
        obs, nan_s, nan_r = s.obs.to_numpy(), np.full(len(s), np.nan), np.full(len(rt), np.nan)
        return _outputs(s, rt, _proba(self.models_["severity"], xs, obs),
                        _values(self.models_["sample_rate"], xs, obs & ~s.cumulative.to_numpy()), nan_s,
                        _values(self.models_["run"], xr, rt.coverage.to_numpy() > 0), nan_r, "")


def make_run_baseline(name: str, cfg: dict, seed: int = 0):
    """Run-level baseline by config name (RUN_BASELINES) with physics.min_run_coverage and baselines.grids.rf."""
    cov = float(cfg["physics"]["min_run_coverage"])
    if name == "physics_only":
        return PhysicsOnlyBaseline(cov)
    if name == "conditions_only":
        return ConditionsOnlyBaseline(cov)
    if name == "physics_conditions":
        return PhysicsConditionsBaseline(cov, seed=seed)
    if name == "descriptor_agg_rf":
        return DescriptorAggregateBaseline("rf", seed, cfg["baselines"]["grids"].get("rf"), cov)
    if name == "descriptor_agg_ridge":
        return DescriptorAggregateBaseline("ridge", seed, None, cov)
    raise ValueError(f"unknown run baseline {name!r}; expected one of {RUN_BASELINES}")
