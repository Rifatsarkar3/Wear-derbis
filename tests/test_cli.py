"""cli.py: parsing, --help, data commands on small datasets, and dispatch of the experiment commands to fake
experiments / profiling modules (so these tests need neither experiments.py nor torch)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

import wearpg
from wearpg import cli
from wearpg.gates import GateResult
from wearpg.schema import SYNTHETIC_BANNER, WearDataset
from wearpg.splits import Split

SMOKE = (Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml").as_posix()
COMMANDS = ("synth", "validate", "segment", "rating-package", "agreement", "features", "train", "cv", "external",
            "transfer", "evaluate", "gates", "profile", "report", "smoke", "run-all")


def _fake_module(monkeypatch, name: str, calls: list, **funcs) -> types.ModuleType:
    """Install a fake wearpg.<name> whose functions record (name, args, kwargs) in `calls` and return the given value
    (or call it when it is a function)."""
    mod = types.ModuleType(f"wearpg.{name}")
    for fname, ret in funcs.items():
        def f(*a, _n=fname, _r=ret, **k):
            calls.append((_n, a, k))
            return _r(*a, **k) if callable(_r) else _r
        setattr(mod, fname, f)
    monkeypatch.setitem(sys.modules, f"wearpg.{name}", mod)
    monkeypatch.setattr(wearpg, name, mod, raising=False)
    return mod


@pytest.fixture
def fake_ex(monkeypatch, tiny_dataset):
    calls: list = []
    cache = types.SimpleNamespace(descriptors=pd.DataFrame(), particle_ids=["p0"], meta={"dataset_key": "k"})
    splits = [Split(f"specimen_k3_f{i}", "cv", ["R_a"], ["R_b"], ["R_c"], {}) for i in range(3)]
    preds = {lvl: pd.DataFrame({"x": [1.0]}) for lvl in ("particles", "samples", "runs")}
    ex = _fake_module(monkeypatch, "experiments", calls, prepare=(tiny_dataset, cache), make_splits=splits,
                      run_proposed=preds, run_cv=Path("cv"), run_external=Path("ext"), run_transfer=Path("tr"),
                      evaluate_results={},
                      load_predictions=lambda d, lvl: pd.DataFrame({"split": ["test", "val"], "x": [1, 2]}),
                      run_smoke=lambda path, device=None: Path(path))
    ex.VARIANTS = {"full": [], "no_physics": ["model.physics_head=false"]}
    _fake_module(monkeypatch, "profiling", calls, run_profiling={"files": {"summary": "s.json"}})
    monkeypatch.setattr("wearpg.report.build_report", lambda *a, **k: calls.append(("build_report", a, k)) or "r.md")
    monkeypatch.setattr("wearpg.gates.evaluate_gates", lambda ds, desc, results, cfg: calls.append(
        ("evaluate_gates", (results,), {})) or [GateResult("pilot", None, "not evaluable", {}, {})])
    return calls


def _base(out: Path) -> list[str]:
    return ["--config", SMOKE, "--set", f"paths.output_dir={out.as_posix()}"]


def test_help_and_parsing():
    for argv in (["--help"], *([c, "--help"] for c in COMMANDS)):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 0
    a = cli.build_parser().parse_args(["cv", "--config", "c.yaml", "--set", "a.b=1", "--set", "c=[1, 2]",
                                       "--models", "wearpg", "physics_only"])
    assert a.set == ["a.b=1", "c=[1, 2]"] and a.models == ["wearpg", "physics_only"] and a.variants is None
    assert cli.build_parser().parse_args(["smoke"]).config == "configs/smoke.yaml"
    with pytest.raises(SystemExit):  # --by, --train, --test go together
        cli.main(["transfer", "--config", SMOKE, "--by", "magnification"])
    with pytest.raises(SystemExit, match="not used"):  # validate has no settings
        cli.main(["validate", "--data", ".", "--set", "a=1"])


def test_validate_agreement_rating_package(tiny_dataset_on_disk, tmp_path, capsys):
    root = tiny_dataset_on_disk.root
    assert cli.main(["validate", "--data", str(root)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_errors"] == 0 and out["n_runs"] == 12 and out["missing_files"] == []
    assert out["banner"] == SYNTHETIC_BANNER

    assert cli.main(["agreement", "--data", str(root)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["banner"] == SYNTHETIC_BANNER and out["agreement"]["raters"] == ["A", "B"]
    assert out["validity"]["n"] == 0  # the tiny dataset has no SEMREF labels

    pkg = tmp_path / "pkg"
    assert cli.main(["rating-package", "--data", str(root), "--out", str(pkg)]) == 0
    assert len(list((pkg / "crops").glob("*.png"))) == 96 and (pkg / "labels_template_A.csv").exists()

    (root / "labels.csv").unlink()
    runs = pd.read_csv(root / "runs.csv", dtype=str, keep_default_na=False)
    runs.loc[0, "load_N"] = "-5"
    runs.to_csv(root / "runs.csv", index=False)
    assert cli.main(["validate", "--data", str(root)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert any("load_N must be > 0" in e for e in out["errors"]) and out["missing_files"] == ["labels.csv"]
    pd.concat([runs, runs.head(1)]).to_csv(root / "runs.csv", index=False)  # duplicate key: reported, no crash
    assert cli.main(["validate", "--data", str(root)]) == 1
    assert any("duplicate primary key 'run_id'" in e for e in json.loads(capsys.readouterr().out)["errors"])
    with pytest.raises(SystemExit, match="not found"):
        cli.main(["validate", "--data", str(root / "missing")])


def test_synth(tmp_path, capsys):
    out = tmp_path / "syn"
    argv = ["synth", "--config", "smoke", "--out", str(out), "--seed", "1", "--set", "samples_per_run=2",
            "--set", "duration_s=1200", "--set", "debris.max_stored_per_sample=3"]
    assert cli.main(argv) == 0
    res = json.loads(capsys.readouterr().out)
    ds = WearDataset.load(out)
    assert res["banner"] == SYNTHETIC_BANNER and res["n_samples"] == 2 * res["n_runs"] == len(ds.samples)
    assert ds.validate() == [] and ds.meta["seed"] == 1


def test_segment_write_keeps_dataset_consistent(tiny_dataset_on_disk, capsys):
    root = tiny_dataset_on_disk.root
    samples = pd.read_csv(root / "samples.csv", dtype=str, keep_default_na=False)
    samples["n_particles_total"] = "1"  # below the segmented count -> must be raised on --write
    samples.to_csv(root / "samples.csv", index=False)
    before = (root / "views.parquet").read_bytes()
    assert cli.main(["segment", "--data", str(root), "--match-gt"]) == 0  # without --write nothing changes
    out = json.loads(capsys.readouterr().out)
    assert out["banner"] == SYNTHETIC_BANNER and out["quality"]["recall"] > 0.9 and "backup" not in out
    assert (root / "views.parquet").read_bytes() == before

    assert cli.main(["segment", "--data", str(root), "--match-gt", "--write", "--set", "min_area_px=20"]) == 0
    out = json.loads(capsys.readouterr().out)
    ds = WearDataset.load(root)
    assert ds.validate() == [] and (ds.views.source == "segmented").all()
    assert set(ds.labels.particle_id) <= set(ds.particles.particle_id) and len(ds.labels_adjudicated)
    assert (ds.samples.n_particles_total >= ds.samples.sample_id.map(ds.particles.groupby("sample_id").size())).all()
    backup = Path(out["backup"])
    assert (pd.read_parquet(backup / "views.parquet").source == "gt").all()  # replaced tables kept
    with pytest.raises(SystemExit, match="unknown segmentation settings"):
        cli.main(["segment", "--data", str(root), "--set", "bogus=1"])


def test_experiment_commands_dispatch(fake_ex, tmp_path):
    out = tmp_path / "res"
    assert cli.main(["cv", *_base(out), "--models", "wearpg", "--variants", "full", "no_physics"]) == 0
    name, a, k = fake_ex[-1]
    assert name == "run_cv" and k == {"models": ["wearpg"], "variants": ["full", "no_physics"]}
    assert a[0]["paths"]["output_dir"] == out.as_posix() and a[0]["name"] == "smoke"  # base merged, override applied
    cli.main(["transfer", *_base(out), "--by", "focus_level", "--train", "200", "[0.0, 0.5]", "--test", "[0.5, .inf]"])
    assert fake_ex[-1][2] == {"by": "focus_level", "train_values": [200, [0.0, 0.5]],
                              "test_values": [[0.5, float("inf")]]}  # YAML-typed specs
    cli.main(["transfer", *_base(out)])
    assert fake_ex[-1][2] == {"by": None, "train_values": None, "test_values": None}  # eval.transfer loop
    cli.main(["external", *_base(out), "--by", "load_N"])
    assert fake_ex[-1][0] == "run_external" and fake_ex[-1][2] == {"by": "load_N"}
    cli.main(["evaluate", *_base(out)])
    assert fake_ex[-1][0] == "evaluate_results" and fake_ex[-1][1][0] == out
    cli.main(["features", *_base(out), "--force"])
    assert fake_ex[-1][0] == "prepare" and fake_ex[-1][2] == {"force": True}
    cli.main(["profile", *_base(out)])
    assert fake_ex[-1][0] == "run_profiling" and fake_ex[-1][1][1] == out
    cli.main(["report", *_base(out), "--paper-dir", "paper"])
    assert fake_ex[-1] == ("build_report", (out, fake_ex[-1][1][1]), {"paper_dir": "paper"})


def test_train_writes_standard_parts(fake_ex, tmp_path):
    out = tmp_path / "res"
    assert cli.main(["train", *_base(out), "--fold", "1", "--seed", "3", "--variant", "no_physics"]) == 0
    _, a, k = next(c for c in fake_ex if c[0] == "run_proposed")
    assert a[3].name == "specimen_k3_f1" and a[4] == 3 and k["variant"] == "no_physics"
    assert k["out_dir"] == out / "training" / "wearpg" / "no_physics" / "specimen_k3_f1_seed3"
    assert sorted(p.name for p in (out / "predictions" / "parts").iterdir()) == [
        f"{lvl}__cv__wearpg__no_physics__specimen_k3_f1__seed3.parquet" for lvl in ("particles", "runs", "samples")]
    with pytest.raises(SystemExit, match="--fold"):
        cli.main(["train", *_base(out), "--fold", "3"])
    with pytest.raises(SystemExit, match="unknown variant"):
        cli.main(["train", *_base(out), "--variant", "nope"])


def test_gates_and_run_all(fake_ex, tmp_path):
    out = tmp_path / "res"
    assert cli.main(["gates", *_base(out)]) == 0
    results = next(c for c in fake_ex if c[0] == "evaluate_gates")[1][0]
    assert set(results) == {"particles", "samples", "runs"} and all((d.split == "test").all() for d in results.values())
    doc = json.loads((out / "gates" / "gates.json").read_text(encoding="utf-8"))
    assert doc["synthetic"] is True and doc["decision_valid"] is False  # the tiny dataset is synthetic

    fake_ex.clear()
    assert cli.main(["run-all", *_base(out)]) == 0
    order = [c[0] for c in fake_ex if c[0] != "load_predictions"]
    assert order == ["prepare", "run_cv", "run_external", "run_transfer", "evaluate_results", "evaluate_gates",
                     "run_profiling", "build_report"]


def test_smoke_passes_overrides_through_a_resolved_config(fake_ex, tmp_path):
    cli.main(["smoke"])
    assert fake_ex[-1][1][0] == Path("configs/smoke.yaml")
    cli.main(["smoke", *_base(tmp_path), "--set", "train.epochs=1"])
    path = Path(fake_ex[-1][1][0])
    resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "cli_smoke.yaml" and resolved["train"]["epochs"] == 1
    assert "base" not in resolved and resolved["model"]["conv"] == "gine"  # default.yaml merged in
