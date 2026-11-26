"""[I2] Command-line interface: ``python -m wearpg <command>`` or the ``wearpg`` console script (DESIGN sec. 12).

Every command takes ``--set key=value`` (repeatable; dotted key, YAML-parsed value). It overrides the experiment
config (``--config``), the generator config (``synth``) or the flat ``segment_image`` settings (``segment``, keys of
``segmentation.SEG_DEFAULTS``); ``validate``, ``agreement`` and ``rating-package`` have no settings and reject it.
Handlers import their modules lazily, so ``--help`` and the data commands never load torch. Run from the project
root: config paths are relative to it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from wearpg.config import apply_overrides, load_config
from wearpg.schema import SYNTHETIC_BANNER, TABLES, WearDataset
from wearpg.utils import get_logger

log = get_logger()


# ----------------------------------------------------------------------------- helpers


def _print(obj: dict, synthetic: bool = False) -> None:
    """JSON to stdout (ASCII); synthetic results carry the banner as their first key."""
    print(json.dumps({"banner": SYNTHETIC_BANNER, **obj} if synthetic else obj, indent=2, default=str))


def _cfg(args) -> dict:
    return load_config(args.config, args.set)


def _out(cfg: dict) -> Path:
    return Path(cfg["paths"]["output_dir"])


def _data(args) -> WearDataset:
    if not Path(args.data).is_dir():
        raise SystemExit(f"error: dataset directory not found: {args.data}")
    return WearDataset.load(args.data)


def _no_set(args) -> None:
    if args.set:
        raise SystemExit(f"error: --set is not used by '{args.cmd}' (it has no settings)")


def _save_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- data commands


def cmd_synth(args) -> int:
    from wearpg.synthetic import generate_dataset, load_synthetic_config

    ds = generate_dataset(apply_overrides(load_synthetic_config(args.config), args.set), args.out, seed=args.seed)
    _print({"out": Path(args.out).as_posix(), **ds.describe()}, synthetic=True)
    return 0


def cmd_validate(args) -> int:
    """Exit 1 when the dataset has errors. Missing table files are listed, not errors: early onboarding stages
    legitimately lack particles/views/labels."""
    _no_set(args)
    ds = _data(args)
    try:
        errs = ds.validate()
    except Exception as err:  # malformed tables (e.g. unparsable values): report, never crash
        errs = [f"validation aborted: {type(err).__name__}: {err}"]
    files =[f"{t}.parquet" if t == "views" else f"{t}.csv" for t in TABLES] + ["dataset.json"]
    _print({"data": Path(args.data).as_posix(), **ds.describe(),
            "missing_files": [f for f in files if not (Path(args.data) / f).exists()],
            "n_errors": len(errs), "errors": errs}, synthetic=ds.is_synthetic)
    return 1 if errs else 0


def cmd_segment(args) -> int:
    from wearpg.segmentation import SEG_DEFAULTS, segment_dataset

    seg_cfg = apply_overrides({}, args.set)
    unknown = sorted(set(seg_cfg) - set(SEG_DEFAULTS))
    if unknown:
        raise SystemExit(f"error: unknown segmentation settings {unknown}; allowed: {sorted(SEG_DEFAULTS)}")
    ds = _data(args)
    particles, views, quality = segment_dataset(ds, seg_cfg or None, match_gt=args.match_gt)
    out = {"n_particles": int(len(particles)), "n_views": int(len(views)), "quality": quality}
    if args.write:
        out["backup"] = _write_segmentation(ds, particles, views).as_posix()
    _print(out, synthetic=ds.is_synthetic)
    return 0


def _write_segmentation(ds: WearDataset, particles, views) -> Path:
    """Replace particles/views by the segmentation and keep the dataset consistent (labels only for kept
    particles, n_particles_total >= segmented rows per sample). Validates before touching any file and copies the
    replaced table files to <root>/pre_segment_<time>/ first. Returns the backup dir. The feature cache keys on the
    saved files, so build it after this."""
    keep = set(particles.particle_id)
    ds.labels = ds.labels[ds.labels.particle_id.isin(keep)]
    ds.labels_adjudicated = ds.labels_adjudicated[ds.labels_adjudicated.particle_id.isin(keep)]
    n_seg = ds.samples.sample_id.map(particles.groupby("sample_id").size()).fillna(0).to_numpy(np.int64)
    ds.samples["n_particles_total"] = np.maximum(ds.samples.n_particles_total.to_numpy(np.int64), n_seg)
    ds.particles, ds.views = particles, views
    ds.assert_valid()
    backup = ds.root / time.strftime("pre_segment_%Y%m%d-%H%M%S")
    backup.mkdir()
    for f in ds.root.iterdir():
        if f.is_file():
            shutil.copy2(f, backup / f.name)
    ds.save()
    log.info("segment: wrote %d particles / %d views to %s (previous tables in %s)", len(particles), len(views),
             ds.root, backup)
    return backup


def cmd_rating_package(args) -> int:
    from wearpg.segmentation import export_rating_package

    _no_set(args)
    out = export_rating_package(_data(args), args.out)
    log.info("rating package written to %s", out)
    return 0


def cmd_agreement(args) -> int:
    from wearpg.labels import agreement_report, validity_report

    _no_set(args)
    ds = _data(args)
    _print({"agreement": agreement_report(ds.labels), "validity": validity_report(ds.labels, ds.labels_adjudicated)},
           synthetic=ds.is_synthetic)
    return 0


# ----------------------------------------------------------------------------- experiment commands


def cmd_features(args) -> int:
    from wearpg import experiments as ex

    cfg = _cfg(args)
    _, cache = ex.prepare(cfg, force=args.force)
    log.info("feature cache %s: %d particles", cache.meta.get("dataset_key"), len(cache.particle_ids))
    return 0


def cmd_train(args) -> int:
    """Proposed model on cv fold K: training artefacts under training/, predictions as standard parts (so
    evaluate picks them up and cv skips them)."""
    from wearpg import experiments as ex
    from wearpg.splits import safe_name

    cfg = _cfg(args)
    if args.variant not in ex.VARIANTS:
        raise SystemExit(f"error: unknown variant {args.variant!r}; choose from {sorted(ex.VARIANTS)}")
    ds, cache = ex.prepare(cfg)
    splits = ex.make_splits(ds, cfg)
    if not 0 <= args.fold < len(splits):
        raise SystemExit(f"error: --fold must be in [0, {len(splits) - 1}]")
    split = splits[args.fold]
    seed = int(cfg["eval"]["seeds"][0] if args.seed is None else args.seed)
    out, fold = _out(cfg), safe_name(split.name)
    preds = ex.run_proposed(ds, cache, cfg, split, seed, variant=args.variant,
                            out_dir=out / "training" / "wearpg" / args.variant / f"{fold}_seed{seed}")
    for level, df in preds.items():
        _save_parquet(df, out / "predictions" / "parts" /
                      f"{level}__{split.protocol}__wearpg__{args.variant}__{fold}__seed{seed}.parquet")
    log.info("train: %s seed %d variant %s -> %s", split.name, seed, args.variant, out)
    return 0


def cmd_cv(args) -> int:
    from wearpg.experiments import run_cv

    log.info("cv results: %s", run_cv(_cfg(args), models=args.models, variants=args.variants))
    return 0


def cmd_external(args) -> int:
    from wearpg.experiments import run_external

    log.info("external results: %s", run_external(_cfg(args), by=args.by))
    return 0


def cmd_transfer(args) -> int:
    from wearpg.experiments import run_transfer

    # values are typed specs: "200" -> 200 (matches the float column), "[0.5, .inf]" -> [lo, hi) interval
    train = None if args.train is None else [yaml.safe_load(v) for v in args.train]
    test = None if args.test is None else [yaml.safe_load(v) for v in args.test]
    log.info("transfer results: %s", run_transfer(_cfg(args), by=args.by, train_values=train, test_values=test))
    return 0


def cmd_evaluate(args) -> int:
    from wearpg.experiments import evaluate_results

    cfg = _cfg(args)
    evaluate_results(_out(cfg), cfg)
    log.info("metrics written to %s", _out(cfg) / "metrics")
    return 0


def _run_gates(cfg: dict, ds, cache) -> Path:
    """Gates on the test-split predictions of every protocol -> output_dir/gates/gates.{json,md}."""
    from wearpg import experiments as ex
    from wearpg.gates import evaluate_gates, write_gate_report

    out, results = _out(cfg), {}
    for level in ("particles", "samples", "runs"):
        df = ex.load_predictions(out, level)
        results[level] = None if df is None or df.empty else df[df.split == "test"]
    json_path, _ = write_gate_report(evaluate_gates(ds, cache.descriptors, results, cfg), out / "gates",
                                     ds.is_synthetic)
    return json_path


def cmd_gates(args) -> int:
    from wearpg import experiments as ex

    cfg = _cfg(args)
    log.info("gates: %s", _run_gates(cfg, *ex.prepare(cfg)))
    return 0


def cmd_profile(args) -> int:
    from wearpg.profiling import run_profiling

    cfg = _cfg(args)
    log.info("profiling: %s", run_profiling(cfg, _out(cfg))["files"]["summary"])
    return 0


def cmd_report(args) -> int:
    from wearpg.report import build_report

    cfg = _cfg(args)
    log.info("report: %s", build_report(_out(cfg), cfg, paper_dir=args.paper_dir))
    return 0


def cmd_smoke(args) -> int:
    from wearpg.experiments import run_smoke

    path = Path(args.config)
    if args.set:  # run_smoke takes a config path: hand it the resolved config (base merged, overrides applied)
        cfg = _cfg(args)
        path = _out(cfg) / f"cli_{path.name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log.info("smoke report: %s", run_smoke(path))
    return 0


def cmd_run_all(args) -> int:
    """features, cv, external, transfer loop, evaluate, gates, profile, report."""
    from wearpg import experiments as ex
    from wearpg.profiling import run_profiling
    from wearpg.report import build_report

    cfg = _cfg(args)
    out = _out(cfg)
    ds, cache = ex.prepare(cfg)
    ex.run_cv(cfg)
    ex.run_external(cfg)
    ex.run_transfer(cfg)
    ex.evaluate_results(out, cfg)
    _run_gates(cfg, ds, cache)
    run_profiling(cfg, out)
    log.info("run-all done: %s", build_report(out, cfg))
    return 0


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a setting (dotted key, YAML value); repeatable")
    p = argparse.ArgumentParser(prog="wearpg", description="Mechanism-linked multi-view particle graph learning "
                                "for wear debris (DESIGN.md sec. 12). Run from the project root.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    def add(name: str, fn, help: str, arg: str | None = "--config") -> argparse.ArgumentParser:
        s = sub.add_parser(name, parents=[common], help=help, description=help)
        if arg == "--config":
            s.add_argument("--config", required=True, metavar="CFG", help="experiment config (YAML)")
        elif arg == "--data":
            s.add_argument("--data", required=True, metavar="DIR", help="dataset directory")
        s.set_defaults(func=fn)
        return s

    s = add("synth", cmd_synth, "generate a synthetic dataset (software verification only)", arg=None)
    s.add_argument("--config", required=True, metavar="CFG", help="generator config: smoke | pilot | YAML path")
    s.add_argument("--out", required=True, metavar="DIR")
    s.add_argument("--seed", type=int, default=None, metavar="N")
    add("validate", cmd_validate, "schema and integrity check of a dataset directory", arg="--data")
    s = add("segment", cmd_segment, "segment ferrograph images into particle contours", arg="--data")
    s.add_argument("--match-gt", action="store_true", help="keep ids of segments matching existing contours")
    s.add_argument("--write", action="store_true", help="replace the dataset's particles/views (backup kept)")
    s = add("rating-package", cmd_rating_package, "blinded crops + rater templates + README", arg="--data")
    s.add_argument("--out", required=True, metavar="DIR")
    add("agreement", cmd_agreement, "inter-rater agreement and validity vs reference raters", arg="--data")
    s = add("features", cmd_features, "build or refresh the feature cache")
    s.add_argument("--force", action="store_true", help="rebuild even if the cache is current")
    s = add("train", cmd_train, "train and predict the proposed model on one cv fold")
    s.add_argument("--fold", type=int, default=0, metavar="K")
    s.add_argument("--seed", type=int, default=None, metavar="S", help="default: first of eval.seeds")
    s.add_argument("--variant", default="full", metavar="V", help="ablation variant (experiments.VARIANTS)")
    s = add("cv", cmd_cv, "grouped cross-validation (protocol cv)")
    s.add_argument("--models", nargs="+", default=None)
    s.add_argument("--variants", nargs="+", default=None)
    s = add("external", cmd_external, "leave-one-group-out (protocol external)")
    s.add_argument("--by", default=None, metavar="COL", help="default: eval.external_by")
    s = add("transfer", cmd_transfer, "domain-shift protocol (default: every entry of eval.transfer)")
    s.add_argument("--by", default=None, metavar="COL")
    s.add_argument("--train", nargs="+", default=None, metavar="V", help="YAML values, e.g. 200 or '[0.0, 0.5]'")
    s.add_argument("--test", nargs="+", default=None, metavar="V")
    add("evaluate", cmd_evaluate, "metrics, CIs, comparisons -> metrics/")
    add("gates", cmd_gates, "go/no-go gates -> gates/")
    add("profile", cmd_profile, "GNN / contour-CNN memory and throughput -> profiling/")
    s = add("report", cmd_report, "report/ (markdown, figures, LaTeX tables)")
    s.add_argument("--paper-dir", default=None, metavar="DIR",
                   help="copy tables/figures into DIR/tables, DIR/figures (never for synthetic data)")
    s = add("smoke", cmd_smoke, "end-to-end verification on synthetic smoke data", arg=None)
    s.add_argument("--config", default="configs/smoke.yaml", metavar="CFG")
    add("run-all", cmd_run_all, "features, cv, external, transfer, evaluate, gates, profile, report")
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse argv (default sys.argv[1:]) and run the command; returns the exit code."""
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "transfer" and len({args.by is None, args.train is None, args.test is None}) > 1:
        p.error("transfer: give --by, --train and --test together (or none for eval.transfer)")
    return int(args.func(args) or 0)
