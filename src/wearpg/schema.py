"""Dataset schema: hierarchical tables particle -> image -> sample -> run -> specimen/material.

Single source of truth for table names, column names, dtypes, units and enums.
Every other module imports names from here instead of hard-coding strings.

Storage layout of a dataset directory::

    <root>/dataset.json          metadata (name, schema_version, is_synthetic, ...)
    <root>/specimens.csv
    <root>/runs.csv
    <root>/samples.csv
    <root>/images.csv
    <root>/particles.csv
    <root>/labels.csv
    <root>/labels_adjudicated.csv
    <root>/views.parquet         (list columns contour_x / contour_y)
    <root>/images/...            image files referenced by images.path

Missing-value conventions (plain numpy dtypes, no pandas nullable types):
    optional float -> NaN, optional int -> -1, optional str -> "", optional bool -> False.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"

MECHANISMS: tuple[str, ...] = (
    "rubbing",
    "cutting",
    "spherical",
    "fatigue_chunk",
    "laminar",
    "severe_sliding",
    "oxide",
)
SEVERITY_LEVELS: tuple[str, ...] = ("mild", "transition", "severe")  # ordinal 0, 1, 2
REGIMES: tuple[str, ...] = ("mild", "severe")  # run-level final regime
ELEMENTS: tuple[str, ...] = ("Fe", "Cr", "Ni", "Mo", "Cu", "Sn", "Zn", "Al", "Si", "O", "C")
EDS_COLUMNS: tuple[str, ...] = tuple(f"eds_{e}" for e in ELEMENTS)
MODALITIES: tuple[str, ...] = ("ferrograph_optical", "sem", "confocal", "multiview_frame")
RIGS: tuple[str, ...] = ("pin_on_disc", "gear", "bearing", "four_ball", "other")
WEAR_VOLUME_METHODS: tuple[str, ...] = ("gravimetric", "profilometry", "synthetic")
LABEL_EVIDENCE: tuple[str, ...] = ("optical", "sem", "eds", "sem_eds")
ADJUDICATION_METHODS: tuple[str, ...] = ("unanimous", "majority", "adjudicated", "single")
VIEW_SOURCES: tuple[str, ...] = ("gt", "segmented", "manual")
DEBRIS_SAMPLING: tuple[str, ...] = ("interval", "cumulative")  # "" = interval
REFERENCE_RATERS: tuple[str, ...] = ("SEMREF",)  # independent SEM/EDS reference labels (validity only)

SYNTHETIC_BANNER = "SYNTHETIC DATA - PIPELINE VERIFICATION ONLY - NOT EVIDENCE FOR ANY SCIENTIFIC CLAIM"
VERIFICATION_NOTICE = "VERIFICATION ONLY - NO GO/NO-GO DECISION"

UM3_PER_MM3 = 1e9  # 1 mm^3 = 1e9 um^3
# mass [mg] / density [g/cm^3] = volume [mm^3]  (1 mg / (1 g/cm^3) = 1e-3 cm^3 = 1 mm^3)

# column -> (dtype, required, description with unit). dtype in {str, float, int, bool, list}
ColSpec = tuple[str, bool, str]

TABLES: dict[str, dict[str, ColSpec]] = {
    "specimens": {
        "specimen_id": ("str", True, "Primary key."),
        "material_id": ("str", True, "Wearing-specimen material, e.g. 'AISI52100'."),
        "counterface_id": ("str", True, "Counterface material."),
        "material_pair": ("str", True, "'{material_id}|{counterface_id}'."),
        "hardness_MPa": ("float", True, "Specimen hardness [MPa] (HV * 9.807)."),
        "density_g_cm3": ("float", True, "Specimen density [g/cm^3]; mass loss -> volume."),
        "ferromagnetic": ("bool", False, "Ferromagnetic material (magnetic-capture class for eta)."),
        "counterface_density_g_cm3": ("float", False, "Counterface density [g/cm^3]."),
        "notes": ("str", False, "Free text."),
    },
    "runs": {
        "run_id": ("str", True, "Primary key. One wear test = one run."),
        "specimen_id": ("str", True, "FK -> specimens."),
        "condition_id": ("str", True, "Operating-condition group; repeats share it."),
        "rig": ("str", True, f"One of {RIGS}."),
        "load_N": ("float", True, "Normal load [N]."),
        "speed_m_s": ("float", True, "Sliding speed [m/s]."),
        "lubricant_id": ("str", True, "Lubricant identifier."),
        "temperature_C": ("float", False, "Bulk oil temperature [degC]."),
        "duration_s": ("float", True, "Test duration [s]."),
        "sliding_distance_m": ("float", True, "Total sliding distance [m]."),
        "repeat_idx": ("int", True, "Repeat index within condition (0-based)."),
        "reservoir_volume_ml": ("float", True, "Oil reservoir volume [ml]."),
        "mass_loss_mg": ("float", False, "Gravimetric mass loss over the run [mg]."),
        "wear_volume_mm3": ("float", False, "Measured wear volume over the run [mm^3]."),
        "wear_volume_method": ("str", False, f"One of {WEAR_VOLUME_METHODS}."),
        "wear_rate_mm3_Nm": ("float", False, "Specific wear rate k = V/(F*s) [mm^3/(N m)]."),
        "regime": ("str", False, f"Run-level final regime, one of {REGIMES}."),
        "transition_time_s": ("float", False, "Observed mild->severe transition time [s]; NaN if none."),
        "debris_sampling": ("str", False, f"One of {DEBRIS_SAMPLING}; '' = interval (see DESIGN sec. 3)."),
        "counterface_mass_loss_mg": ("float", False, "Counterface gravimetric mass loss [mg]."),
    },
    "samples": {
        "sample_id": ("str", True, "Primary key. One oil sample / debris-collection interval."),
        "run_id": ("str", True, "FK -> runs."),
        "t_start_s": ("float", True, "Start of the debris-generation interval represented [s]."),
        "t_end_s": ("float", True, "End of interval = sampling time [s]."),
        "time_s": ("float", True, "Sampling time [s] (= t_end_s)."),
        "sliding_distance_start_m": ("float", True, "Cumulative sliding distance at t_start [m]."),
        "sliding_distance_end_m": ("float", True, "Cumulative sliding distance at t_end [m]."),
        "sampled_oil_ml": ("float", True, "Oil volume drawn for the sample [ml]."),
        "imaged_fraction": ("float", True, "(0,1]: fraction of the sample's debris that was imaged = (deposited aliquot / "
                            "sampled_oil_ml) x (imaged deposit area / total deposit area)."),
        "observation_fraction": (
            "float",
            True,
            "Fraction of the interval's generated debris that reached the imaged area: "
            "sampled_oil_ml / reservoir_volume_ml * imaged_fraction (validated).",
        ),
        "n_particles_total": ("int", True, "Particles detected in the imaged area (>= rows in particles)."),
        "interval_wear_volume_mm3": ("float", False, "Wear volume generated in the interval [mm^3]."),
        "severity": ("int", False, "Ordinal severity of the interval: 0 mild, 1 transition, 2 severe; -1 unknown."),
        "true_interval_wear_volume_mm3": ("float", False, "SYNTHETIC ONLY: true interval wear volume [mm^3]."),
    },
    "images": {
        "image_id": ("str", True, "Primary key."),
        "sample_id": ("str", True, "FK -> samples."),
        "path": ("str", True, "Image path relative to dataset root, forward slashes."),
        "modality": ("str", True, f"One of {MODALITIES}."),
        "magnification": ("float", True, "Nominal objective magnification, e.g. 200."),
        "um_per_px": ("float", True, "Calibrated pixel size [um/px]."),
        "focus_level": ("float", False, "Focus/blur indicator (synthetic: Gaussian blur sigma [px])."),
        "illumination": ("str", False, "reflected | transmitted | bichromatic | sem_se | sem_bse."),
        "width_px": ("int", True, "Image width [px]."),
        "height_px": ("int", True, "Image height [px]."),
    },
    "particles": {
        "particle_id": ("str", True, "Primary key. One physical particle."),
        "sample_id": ("str", True, "FK -> samples."),
        "image_id": ("str", True, "FK -> images; image of the primary (view_idx 0) view."),
        "n_views": ("int", True, "Number of rows in views for this particle (>= 1)."),
        "has_eds": ("bool", True, "EDS composition available (eds_* columns valid)."),
        **{c: ("float", False, f"EDS mass fraction of {c[4:]} [0-1]; NaN if no EDS.") for c in EDS_COLUMNS},
        "has_sem": ("bool", True, "Independent SEM evidence available for the label."),
        "thickness_um": ("float", False, "Measured thickness [um] (confocal/SEM); NaN if unknown."),
        "volume_um3": ("float", False, "Measured particle volume [um^3]; NaN if unknown."),
        "true_mechanism": ("str", False, "SYNTHETIC ONLY: ground-truth mechanism. Never a training label."),
        "true_volume_um3": ("float", False, "SYNTHETIC ONLY: ground-truth volume [um^3]. Diagnostics only."),
        "true_fractal_D_target": ("float", False, "SYNTHETIC ONLY: roughness D_target of the generator."),
        "true_fractal_D": ("float", False, "SYNTHETIC ONLY: generator-measured divider D of the view-0 GT contour."),
    },
    "views": {
        "view_id": ("str", True, "Primary key, '{particle_id}_v{view_idx}'."),
        "particle_id": ("str", True, "FK -> particles."),
        "view_idx": ("int", True, "0 = primary / face-on view."),
        "image_id": ("str", True, "FK -> images (same sample as the particle)."),
        "bbox_x0": ("int", True, "Bounding box in image px (inclusive)."),
        "bbox_y0": ("int", True, "Bounding box in image px (inclusive)."),
        "bbox_x1": ("int", True, "Bounding box in image px (exclusive)."),
        "bbox_y1": ("int", True, "Bounding box in image px (exclusive)."),
        "contour_x": ("list", True, "Closed contour x [px, image coords], no repeated end point."),
        "contour_y": ("list", True, "Closed contour y [px, image coords]."),
        "um_per_px": ("float", True, "Pixel size of the view's image [um/px]."),
        "source": ("str", True, f"One of {VIEW_SOURCES}."),
    },
    "labels": {
        "particle_id": ("str", True, "FK -> particles."),
        "rater_id": ("str", True, "Rater identifier."),
        "label": ("str", True, f"Mechanism class, one of {MECHANISMS}."),
        "confidence": ("int", False, "Rater confidence 1-5; -1 unknown."),
        "evidence": ("str", False, f"One of {LABEL_EVIDENCE}."),
    },
    "labels_adjudicated": {
        "particle_id": ("str", True, "Primary key, FK -> particles."),
        "label": ("str", True, f"Final mechanism class, one of {MECHANISMS}."),
        "method": ("str", True, f"One of {ADJUDICATION_METHODS}."),
        "n_raters": ("int", True, "Number of raters who labelled the particle."),
        "agreement": ("float", True, "Fraction of raters agreeing with the final label."),
    },
}

PRIMARY_KEYS: dict[str, str | None] = {
    "specimens": "specimen_id",
    "runs": "run_id",
    "samples": "sample_id",
    "images": "image_id",
    "particles": "particle_id",
    "views": "view_id",
    "labels": None,  # (particle_id, rater_id) unique
    "labels_adjudicated": "particle_id",
}

TABLE_ORDER: tuple[str, ...] = tuple(TABLES)

# ----------------------------------------------------------------------------- predictions

PRED_KEYS: tuple[str, ...] = ("protocol", "model", "variant", "fold", "seed", "split")
# protocol: cv | external | transfer; fold: split name (str); split: test | val


def prob_columns(classes: tuple[str, ...] | list[str] = MECHANISMS) -> list[str]:
    """Per-class probability column names, e.g. ['p_rubbing', ...]."""
    return [f"p_{c}" for c in classes]


SEVERITY_PROB_COLUMNS: tuple[str, ...] = tuple(f"sev_p_{s}" for s in SEVERITY_LEVELS)

PARTICLE_PRED_COLUMNS: list[str] = [
    *PRED_KEYS,
    "particle_id", "image_id", "sample_id", "run_id", "specimen_id", "material_id",
    "condition_id", "magnification",
    "y_true",      # adjudicated label index, -1 unlabeled
    "y_true_gt",   # synthetic ground-truth index, -1 unknown
    "y_pred",
    *prob_columns(),
    "log_volume_pred", "log_volume_true",  # ln um^3, NaN if n/a
]
SAMPLE_PRED_COLUMNS: list[str] = [
    *PRED_KEYS,
    "sample_id", "run_id", "specimen_id", "material_id", "condition_id", "t_start_s", "time_s",
    "severity_true", "severity_pred", *SEVERITY_PROB_COLUMNS,
    "log10_rate_true", "log10_rate_pred",  # interval specific wear rate, log10 mm^3/(N m)
    "log_debris_volume_mm3",               # ln of debris-implied interval wear volume
]
RUN_PRED_COLUMNS: list[str] = [
    *PRED_KEYS,
    "run_id", "specimen_id", "material_id", "condition_id",
    "log10_rate_true", "log10_rate_pred",
    "log_grav_volume_mm3", "log_debris_volume_mm3",  # natural log
    "transition_time_true", "transition_time_pred",  # s, NaN if none
    "coverage",  # fraction of run sliding distance covered by observed samples
    "note",      # e.g. "eta not transferable"
]

# ----------------------------------------------------------------------------- coercion

_MISSING = {"float": np.nan, "int": -1, "str": "", "bool": False}
_PD_DTYPE = {"float": "float64", "int": "int64", "str": object, "bool": bool, "list": object}


def _to_bool(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return str(v).strip().lower() in ("true", "1", "yes", "y", "t")


def coerce_table(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Return a copy with schema dtypes; adds missing optional columns with missing values.

    Missing *required* columns are added empty too (so validate() can report them) - check
    with validate(), never assume coercion makes a table valid.
    """
    spec = TABLES[table]
    out = df.copy()
    n = len(out)
    for col, (dtype, _req, _desc) in spec.items():
        if col not in out.columns:
            vals = [[] for _ in range(n)] if dtype == "list" else [_MISSING[dtype]] * n
            out[col] = pd.Series(vals, index=out.index, dtype=_PD_DTYPE[dtype])
            continue
        s = out[col]
        if dtype == "float":
            out[col] = pd.to_numeric(s.mask(s.astype(str) == ""), errors="coerce").astype("float64")
        elif dtype == "int":
            out[col] = pd.to_numeric(s.mask(s.astype(str) == ""), errors="coerce").fillna(-1).astype("int64")
        elif dtype == "str":
            out[col] = s.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)).astype(object)
        elif dtype == "bool":
            out[col] = s.map(_to_bool).astype(bool)
        elif dtype == "list":
            out[col] = [np.asarray(v, dtype=np.float64).ravel() for v in s]
    cols = list(spec) + [c for c in out.columns if c not in spec]  # keep user extension columns
    return out[cols].reset_index(drop=True)


def empty_table(table: str) -> pd.DataFrame:
    return coerce_table(pd.DataFrame(columns=list(TABLES[table])), table)


# ----------------------------------------------------------------------------- dataset


@dataclass
class WearDataset:
    """In-memory dataset bundle. Construct with keyword DataFrames; dtypes are coerced."""

    specimens: pd.DataFrame = field(default_factory=lambda: empty_table("specimens"))
    runs: pd.DataFrame = field(default_factory=lambda: empty_table("runs"))
    samples: pd.DataFrame = field(default_factory=lambda: empty_table("samples"))
    images: pd.DataFrame = field(default_factory=lambda: empty_table("images"))
    particles: pd.DataFrame = field(default_factory=lambda: empty_table("particles"))
    views: pd.DataFrame = field(default_factory=lambda: empty_table("views"))
    labels: pd.DataFrame = field(default_factory=lambda: empty_table("labels"))
    labels_adjudicated: pd.DataFrame = field(default_factory=lambda: empty_table("labels_adjudicated"))
    meta: dict = field(default_factory=dict)
    root: Path | None = None

    def __post_init__(self) -> None:
        for t in TABLE_ORDER:
            setattr(self, t, coerce_table(getattr(self, t), t))
        self.meta = {"schema_version": SCHEMA_VERSION, "is_synthetic": False, **(self.meta or {})}
        if self.root is not None:
            self.root = Path(self.root)

    # -- io ------------------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path) -> "WearDataset":
        root = Path(root)
        tables = {}
        for t in TABLE_ORDER:
            if t == "views":
                p = root / "views.parquet"
                tables[t] = pd.read_parquet(p) if p.exists() else empty_table(t)
            else:
                p = root / f"{t}.csv"
                tables[t] = (
                    pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
                    if p.exists()
                    else empty_table(t)
                )
        meta_p = root / "dataset.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        return cls(**tables, meta=meta, root=root)

    def save(self, root: str | Path | None = None) -> Path:
        root = Path(root) if root is not None else self.root
        if root is None:
            raise ValueError("No root given and dataset has no root.")
        root.mkdir(parents=True, exist_ok=True)
        for t in TABLE_ORDER:
            df = getattr(self, t)
            if t == "views":
                v = df.copy()
                for c in ("contour_x", "contour_y"):
                    v[c] = [np.asarray(a, dtype=np.float64).tolist() for a in v[c]]
                v.to_parquet(root / "views.parquet", index=False)
            else:
                df.to_csv(root / f"{t}.csv", index=False, encoding="utf-8", lineterminator="\n")
        (root / "dataset.json").write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        self.root = root
        return root

    # -- convenience -----------------------------------------------------------
    @property
    def is_synthetic(self) -> bool:
        return bool(self.meta.get("is_synthetic", False))

    def image_path(self, image_id: str) -> Path:
        if self.root is None:
            raise ValueError("Dataset has no root; cannot resolve image paths.")
        rel = self.images.set_index("image_id").at[image_id, "path"]
        return self.root / rel

    @staticmethod
    def contour(view_row) -> np.ndarray:
        """(N, 2) float64 contour in image px from a views row (Series, dict or itertuples row)."""
        if isinstance(view_row, (pd.Series, dict)):
            x, y = view_row["contour_x"], view_row["contour_y"]
        else:
            x, y = view_row.contour_x, view_row.contour_y
        return np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=1)

    def particle_hierarchy(self) -> pd.DataFrame:
        """One row per particle with every ancestor id and key attributes.

        Columns: particle_id, sample_id, image_id, magnification, um_per_px, modality, focus_level,
        run_id, time_s, specimen_id, condition_id, load_N, speed_m_s, rig, lubricant_id,
        material_id, counterface_id, material_pair.
        """
        p = self.particles[["particle_id", "sample_id", "image_id"]]
        im = self.images[["image_id", "magnification", "um_per_px", "modality", "focus_level"]]
        s = self.samples[["sample_id", "run_id", "time_s"]]
        r = self.runs[["run_id", "specimen_id", "condition_id", "load_N", "speed_m_s", "rig", "lubricant_id"]]
        sp = self.specimens[["specimen_id", "material_id", "counterface_id", "material_pair"]]
        return (
            p.merge(im, on="image_id", how="left")
            .merge(s, on="sample_id", how="left")
            .merge(r, on="run_id", how="left")
            .merge(sp, on="specimen_id", how="left")
        )

    def run_table(self) -> pd.DataFrame:
        """Runs joined with specimen attributes (material_id, hardness, density, ...)."""
        return self.runs.merge(self.specimens, on="specimen_id", how="left")

    def grav_volume_mm3(self) -> pd.Series:
        """run_id -> measured wear volume [mm^3] used by the consistency term: specimen wear_volume_mm3
        plus counterface_mass_loss_mg / counterface_density_g_cm3 when both are given (NaN-safe)."""
        rt = self.run_table()
        cf = (rt.counterface_mass_loss_mg / rt.counterface_density_g_cm3).where(
            rt.counterface_mass_loss_mg.notna() & rt.counterface_density_g_cm3.notna(), 0.0)
        return pd.Series((rt.wear_volume_mm3 + cf).to_numpy(float), index=rt.run_id.to_numpy())

    def label_indices(self, source: str = "adjudicated", classes: tuple[str, ...] = MECHANISMS) -> pd.Series:
        """particle_id -> class index (-1 if unlabeled). source: 'adjudicated' | 'true'."""
        idx = {c: i for i, c in enumerate(classes)}
        if source == "adjudicated":
            lab = self.labels_adjudicated.set_index("particle_id")["label"]
        elif source == "true":
            lab = self.particles.set_index("particle_id")["true_mechanism"]
        else:
            raise ValueError(f"unknown label source {source!r}")
        out = pd.Series(-1, index=self.particles["particle_id"].to_numpy(), dtype="int64")
        common = out.index.intersection(lab.index)
        out.loc[common] = [idx.get(v, -1) for v in lab.loc[common]]
        return out

    def subset(self, run_ids) -> "WearDataset":
        """Dataset restricted to the given runs (all descendant rows; referenced specimens)."""
        run_ids = set(run_ids)
        runs = self.runs[self.runs.run_id.isin(run_ids)]
        samples = self.samples[self.samples.run_id.isin(run_ids)]
        sids = set(samples.sample_id)
        images = self.images[self.images.sample_id.isin(sids)]
        particles = self.particles[self.particles.sample_id.isin(sids)]
        pids = set(particles.particle_id)
        return WearDataset(
            specimens=self.specimens[self.specimens.specimen_id.isin(set(runs.specimen_id))],
            runs=runs,
            samples=samples,
            images=images,
            particles=particles,
            views=self.views[self.views.particle_id.isin(pids)],
            labels=self.labels[self.labels.particle_id.isin(pids)],
            labels_adjudicated=self.labels_adjudicated[self.labels_adjudicated.particle_id.isin(pids)],
            meta=dict(self.meta),
            root=self.root,
        )

    def describe(self) -> dict:
        return {
            "is_synthetic": self.is_synthetic,
            **{f"n_{t}": int(len(getattr(self, t))) for t in TABLE_ORDER},
            "n_materials": int(self.specimens.material_id.nunique()),
            "n_conditions": int(self.runs.condition_id.nunique()),
            "n_labelled_particles": int(len(self.labels_adjudicated)),
        }

    # -- validation ------------------------------------------------------------
    def validate(self) -> list[str]:
        """Return human-readable schema/integrity errors (empty list = valid)."""
        return validate_dataset(self)

    def assert_valid(self) -> None:
        errs = self.validate()
        if errs:
            head = "\n  - ".join(errs[:50])
            raise ValueError(f"Dataset invalid ({len(errs)} errors):\n  - {head}")


def _close(a, b, rtol=1e-3, atol=1e-12) -> np.ndarray:
    return np.isclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=atol)


def validate_dataset(ds: WearDataset) -> list[str]:
    errs: list[str] = []

    def bad(table: str, mask, msg: str, id_col: str | None = None):
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            df = getattr(ds, table)
            col = id_col or PRIMARY_KEYS.get(table) or df.columns[0]
            ids = df.loc[mask, col].astype(str).tolist()[:5]
            errs.append(f"{table}: {msg} ({int(mask.sum())} rows, e.g. {ids})")

    # required columns / missing values / primary keys
    for t, spec in TABLES.items():
        df = getattr(ds, t)
        for col, (dtype, req, _d) in spec.items():
            if not req:
                continue
            s = df[col]
            if dtype == "str":
                bad(t, s.astype(str).str.len() == 0, f"required column {col!r} empty")
            elif dtype == "float":
                bad(t, ~np.isfinite(s.to_numpy(float)), f"required column {col!r} not finite")
            elif dtype == "int":
                bad(t, s.to_numpy() < 0, f"required column {col!r} missing/negative")
            elif dtype == "list":
                bad(t, np.array([len(v) == 0 for v in s], dtype=bool), f"required column {col!r} empty list")
        pk = PRIMARY_KEYS[t]
        if pk is not None:
            bad(t, df[pk].duplicated(keep=False).to_numpy(), f"duplicate primary key {pk!r}")
    if ds.labels[["particle_id", "rater_id"]].duplicated(keep=False).any():
        bad("labels", ds.labels[["particle_id", "rater_id"]].duplicated(keep=False), "duplicate (particle_id, rater_id)")
    if any("duplicate primary key" in e for e in errs):
        return errs  # the key lookups below (set_index().map) need unique keys

    # foreign keys
    def fk(table, col, ref_table, ref_col):
        ref = set(getattr(ds, ref_table)[ref_col])
        df = getattr(ds, table)
        bad(table, ~df[col].isin(ref).to_numpy(), f"{col} not found in {ref_table}.{ref_col}")

    fk("runs", "specimen_id", "specimens", "specimen_id")
    fk("samples", "run_id", "runs", "run_id")
    fk("images", "sample_id", "samples", "sample_id")
    fk("particles", "sample_id", "samples", "sample_id")
    fk("particles", "image_id", "images", "image_id")
    fk("views", "particle_id", "particles", "particle_id")
    fk("views", "image_id", "images", "image_id")
    fk("labels", "particle_id", "particles", "particle_id")
    fk("labels_adjudicated", "particle_id", "particles", "particle_id")

    # hierarchy consistency: particle image & view images belong to the particle's sample
    img_sample = ds.images.set_index("image_id")["sample_id"]
    p = ds.particles
    bad("particles", p.image_id.map(img_sample).to_numpy() != p.sample_id.to_numpy(),
        "primary image belongs to a different sample")
    v = ds.views
    p_sample = p.set_index("particle_id")["sample_id"]
    bad("views", v.image_id.map(img_sample).to_numpy() != v.particle_id.map(p_sample).to_numpy(),
        "view image belongs to a different sample than its particle")

    # enums
    def enum(table, col, allowed, allow_empty=False):
        s = getattr(ds, table)[col].astype(str)
        ok = s.isin(allowed) | ((s == "") if allow_empty else False)
        bad(table, ~ok.to_numpy(), f"{col} not in {allowed}")

    enum("runs", "rig", RIGS)
    enum("runs", "wear_volume_method", WEAR_VOLUME_METHODS, allow_empty=True)
    enum("runs", "regime", REGIMES, allow_empty=True)
    enum("runs", "debris_sampling", DEBRIS_SAMPLING, allow_empty=True)
    enum("images", "modality", MODALITIES)
    enum("views", "source", VIEW_SOURCES)
    enum("labels", "label", MECHANISMS)
    enum("labels", "evidence", LABEL_EVIDENCE, allow_empty=True)
    enum("labels_adjudicated", "label", MECHANISMS)
    enum("labels_adjudicated", "method", ADJUDICATION_METHODS)
    enum("particles", "true_mechanism", MECHANISMS, allow_empty=True)
    bad("samples", ~ds.samples.severity.isin([-1, 0, 1, 2]).to_numpy(), "severity not in {-1,0,1,2}")
    bad("labels", ~ds.labels.confidence.isin([-1, 1, 2, 3, 4, 5]).to_numpy(), "confidence not in {-1,1..5}",
        id_col="particle_id")

    # value ranges
    r = ds.runs
    for c in ("load_N", "speed_m_s", "duration_s", "sliding_distance_m", "reservoir_volume_ml"):
        bad("runs", r[c].to_numpy(float) <= 0, f"{c} must be > 0")
    rt = ds.run_table()
    has_all = rt[["wear_volume_mm3", "wear_rate_mm3_Nm"]].notna().all(axis=1).to_numpy()
    k_expected = rt.wear_volume_mm3 / (rt.load_N * rt.sliding_distance_m)
    bad("runs", has_all & ~_close(rt.wear_rate_mm3_Nm, k_expected, rtol=1e-2),
        "wear_rate_mm3_Nm != wear_volume_mm3 / (load_N * sliding_distance_m)")
    grav = (rt.wear_volume_method == "gravimetric").to_numpy() & rt[["mass_loss_mg", "wear_volume_mm3"]].notna().all(axis=1).to_numpy()
    bad("runs", grav & ~_close(rt.wear_volume_mm3, rt.mass_loss_mg / rt.density_g_cm3, rtol=1e-2),
        "gravimetric wear_volume_mm3 != mass_loss_mg / density_g_cm3")

    s = ds.samples
    bad("samples", (s.t_end_s <= s.t_start_s).to_numpy(), "t_end_s must be > t_start_s")
    bad("samples", (s.sliding_distance_end_m <= s.sliding_distance_start_m).to_numpy(),
        "sliding_distance_end_m must be > sliding_distance_start_m")
    bad("samples", ~_close(s.time_s, s.t_end_s), "time_s must equal t_end_s")
    imf = s.imaged_fraction.to_numpy(float)
    bad("samples", ~((imf > 0) & (imf <= 1)), "imaged_fraction must be in (0, 1]")
    res = s.run_id.map(r.set_index("run_id")["reservoir_volume_ml"])
    bad("samples", ~_close(s.observation_fraction, s.sampled_oil_ml / res * s.imaged_fraction),
        "observation_fraction != sampled_oil_ml / reservoir_volume_ml * imaged_fraction")
    dur = s.run_id.map(r.set_index("run_id")["duration_s"])
    bad("samples", (s.t_end_s > dur * (1 + 1e-6)).to_numpy(), "t_end_s exceeds run duration")
    for run_id, g in s.sort_values("t_start_s").groupby("run_id"):
        if (g.t_start_s.to_numpy()[1:] < g.t_end_s.to_numpy()[:-1] - 1e-9).any():
            errs.append(f"samples: overlapping intervals in run {run_id}")
    counts = p.groupby("sample_id").size()
    bad("samples", s.sample_id.map(counts).fillna(0).to_numpy() > s.n_particles_total.to_numpy(),
        "n_particles_total smaller than number of particle rows")

    im = ds.images
    bad("images", im.um_per_px.to_numpy(float) <= 0, "um_per_px must be > 0")
    bad("images", (im.width_px <= 0).to_numpy() | (im.height_px <= 0).to_numpy(), "image size must be > 0")

    # particles: views count, EDS, volumes
    vcount = v.groupby("particle_id").size()
    bad("particles", p.particle_id.map(vcount).fillna(0).to_numpy() != p.n_views.to_numpy(),
        "n_views != number of views rows")
    has_v0 = set(v.loc[v.view_idx == 0, "particle_id"])
    bad("particles", ~p.particle_id.isin(has_v0).to_numpy(), "no view_idx 0 view")
    v0_img = v.loc[v.view_idx == 0].set_index("particle_id")["image_id"]
    bad("particles", p.particle_id.map(v0_img).fillna("").to_numpy() != p.image_id.to_numpy(),
        "image_id differs from the view_idx 0 view's image_id")
    eds = p[list(EDS_COLUMNS)].to_numpy(float)
    he = p.has_eds.to_numpy(bool)
    if he.any():
        e = eds[he]
        fin = np.isfinite(e).all(axis=1)
        tot = np.nansum(e, axis=1)
        okv = fin & (e >= -1e-9).all(axis=1) & (e <= 1 + 1e-9).all(axis=1) & (np.abs(tot - 1) <= 0.05)
        m = np.zeros(len(p), bool)
        m[np.flatnonzero(he)[~okv]] = True
        bad("particles", m, "has_eds but EDS fractions invalid (need finite, in [0,1], sum 1+-0.05)")
    for c in ("thickness_um", "volume_um3", "true_volume_um3"):
        x = p[c].to_numpy(float)
        bad("particles", np.isfinite(x) & (x <= 0), f"{c} must be > 0 when given")

    # views: contour & bbox
    lens_x = np.array([len(a) for a in v.contour_x])
    lens_y = np.array([len(a) for a in v.contour_y])
    bad("views", lens_x != lens_y, "contour_x/contour_y length mismatch")
    bad("views", lens_x < 8, "contour needs >= 8 points")
    bad("views", (v.bbox_x1 <= v.bbox_x0).to_numpy() | (v.bbox_y1 <= v.bbox_y0).to_numpy(), "empty bbox")
    bad("views", v.um_per_px.to_numpy(float) <= 0, "um_per_px must be > 0")
    bad("views", ~_close(v.um_per_px, v.image_id.map(im.set_index("image_id")["um_per_px"])),
        "um_per_px differs from its image")
    bad("views", v.view_id.to_numpy() != (v.particle_id + "_v" + v.view_idx.astype(str)).to_numpy(),
        "view_id must be '{particle_id}_v{view_idx}'")

    # labels_adjudicated sanity
    la = ds.labels_adjudicated
    ag = la.agreement.to_numpy(float)
    bad("labels_adjudicated", ~((ag >= 0) & (ag <= 1)), "agreement must be in [0, 1]")
    bad("labels_adjudicated", (la.n_raters < 1).to_numpy(), "n_raters must be >= 1")
    return errs
