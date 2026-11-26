import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from wearpg import training
from wearpg.config import load_config
from wearpg.datasets import build_cache, class_weights, fit_scalers, make_run_loader, model_dims, particles_for_runs
from wearpg.models import MechanismLinkedModel
from wearpg.schema import SEVERITY_PROB_COLUMNS, prob_columns
from wearpg.training import fit_model, predict_runs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def shared_cache_dir(tmp_path_factory):
    """The tiny dataset is deterministic: every test's copy has the same dataset_key, so one cache serves all."""
    return tmp_path_factory.mktemp("tiny_cache_training")


@pytest.fixture
def setup(tiny_dataset_on_disk, shared_cache_dir):
    """(ds, cache, cfg, train runs, val runs, scalers) with a small model config."""
    ds = tiny_dataset_on_disk
    cfg = load_config(ROOT / "configs" / "default.yaml",
                      ["model.hidden=16", "model.gnn_layers=1", "train.epochs=3", "train.batch_runs=2"])
    cache = build_cache(ds, cfg, cache_dir=shared_cache_dir)
    runs = sorted(ds.runs.run_id)
    train, val = runs[0::2], runs[1::2]
    return ds, cache, cfg, train, val, fit_scalers(ds, cache, train, cfg)


def make_model(ds, cache, cfg, scalers):
    torch.manual_seed(0)
    return MechanismLinkedModel(cfg, model_dims(ds, cache, cfg, scalers))


def test_fit_model_early_stopping_restores_best_and_writes_files(setup, tmp_path, monkeypatch):
    ds, cache, cfg, train, val, sc = setup
    cfg = {**cfg, "train": {**cfg["train"], "epochs": 6, "early_stopping_patience": 2}}
    model = make_model(ds, cache, cfg, sc)
    script, states = iter([3.0, 2.0, 2.5, 2.6, 9.0, 9.0]), []

    def scripted_val(model, loader, cfg, device, cw, amp):  # best at epoch 1, no improvement for 2 epochs -> stop
        states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return next(script)

    monkeypatch.setattr(training, "_val_loss", scripted_val)
    cw = class_weights(ds, particles_for_runs(ds, train), cfg)
    res = fit_model(model, make_run_loader(ds, cache, train, cfg, sc, True), make_run_loader(ds, cache, val, cfg, sc, False),
                    cfg, "cpu", tmp_path / "fit", cw)
    assert res["epochs_run"] == 4 and res["best_epoch"] == 1 and res["best_val_loss"] == 2.0
    assert not torch.equal(states[0]["mech_head.weight"], states[1]["mech_head.weight"])  # weights were trained
    assert all(torch.equal(v, states[1][k]) for k, v in model.state_dict().items())  # best epoch restored
    best = torch.load(tmp_path / "fit" / "best.pt", weights_only=True)
    assert all(torch.equal(best[k], v) for k, v in model.state_dict().items())
    hist = json.loads((tmp_path / "fit" / "history.json").read_text(encoding="utf-8"))
    assert [h["epoch"] for h in hist["history"]] == [0, 1, 2, 3] and hist["best_epoch"] == 1
    h0 = hist["history"][0]
    assert math.isfinite(h0["train_loss"]) and {"train_mechanism", "train_consistency", "n_skipped"} <= set(h0)
    lrs = [h["lr"] for h in hist["history"]]
    assert lrs[0] == pytest.approx(float(cfg["train"]["lr"])) and all(a > b for a, b in zip(lrs, lrs[1:]))  # cosine
    assert not list((tmp_path / "fit").glob("*.tmp"))


def test_fit_model_real_validation_and_no_validation(setup):
    ds, cache, cfg, train, val, sc = setup
    tl = make_run_loader(ds, cache, train, cfg, sc, True)
    res = fit_model(make_model(ds, cache, cfg, sc), tl, make_run_loader(ds, cache, val, cfg, sc, False), cfg, "cpu")
    assert all(math.isfinite(h["val_loss"]) for h in res["history"])
    assert res["best_val_loss"] == min(h["val_loss"] for h in res["history"])
    res = fit_model(make_model(ds, cache, cfg, sc), tl, None, cfg, "cpu")  # every epoch, last weights kept
    assert res["epochs_run"] == cfg["train"]["epochs"] and math.isnan(res["best_val_loss"])
    assert res["best_epoch"] == cfg["train"]["epochs"] - 1
    with pytest.raises(ValueError):
        fit_model(make_model(ds, cache, cfg, sc), tl, None, {**cfg, "train": {**cfg["train"], "amp": "fp16"}}, "cpu")


def test_predict_runs_value_columns(setup):
    ds, cache, cfg, train, val, sc = setup
    model = make_model(ds, cache, cfg, sc)
    pred = predict_runs(model, make_run_loader(ds, cache, val, cfg, sc, False), "cpu")
    part, smp, runs = pred["particles"], pred["samples"], pred["runs"]
    assert list(part.columns) == training.PARTICLE_VALUE_COLUMNS
    assert list(smp.columns) == training.SAMPLE_VALUE_COLUMNS and list(runs.columns) == training.RUN_VALUE_COLUMNS
    assert sorted(part.particle_id) == sorted(particles_for_runs(ds, val))  # eval bags hold every particle
    assert sorted(smp.sample_id) == sorted(ds.samples.sample_id[ds.samples.run_id.isin(val)])
    assert sorted(runs.run_id) == sorted(val)
    p = part[prob_columns()].to_numpy()
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
    assert (part.y_pred.to_numpy() == p.argmax(1)).all()
    q = smp[list(SEVERITY_PROB_COLUMNS)].to_numpy()
    np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-5)
    assert (smp.severity_pred.to_numpy() == q.argmax(1)).all()
    np.testing.assert_allclose(runs.coverage, 1.0, rtol=1e-5)  # every interval observed
    assert np.isfinite(runs[["log10_rate_pred", "log_debris_volume_mm3"]].to_numpy()).all()
    assert np.isfinite(part.log_volume_pred).all() and np.isfinite(smp.log10_rate_pred).all()
    empty = predict_runs(model, [], "cpu")
    assert all(len(df) == 0 for df in empty.values())
    assert list(empty["runs"].columns) == training.RUN_VALUE_COLUMNS
