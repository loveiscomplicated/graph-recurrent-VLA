"""Tiny training, checkpoint replay and source adapter tests."""

from dataclasses import replace

import pytest
import torch

from data_collection.state21 import audit_splits
from data_collection.state21_batches import PreparedWindows
from data_collection.state21_synthetic import synthetic_episodes
from evaluation.state21 import summarize_errors
from graph.state_estimation.models import MODELS, build_model
from training.state21 import fit, run_epoch


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_split_checks_source_hash_and_upstream_tracker():
    train, _, validation, *_ = synthetic_episodes()
    overlap = replace(validation.samples[0], frame_sha256=train.samples[0].frame_sha256)
    with pytest.raises(ValueError, match="source frame"):
        audit_splits([train, replace(validation, samples=(overlap,))])
    with pytest.raises(ValueError, match="Tracker"):
        audit_splits([train, validation], tracker_parents=[validation.parent_id])


@pytest.mark.parametrize("name", MODELS)
def test_tiny_overfit_and_cpu_gradients(name, record_property):
    import json

    torch.manual_seed(9)
    model = build_model(name, hidden=24, layers=1, heads=4, history=3)
    episodes = synthetic_episodes(steps=3)
    dataset = PreparedWindows([episodes[0], episodes[1]], history=3, max_edge_m=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    settings = {
        "task": {"shape_weight": 0.5},
        "training": {"batch_size": 8, "gradient_clip_norm": 1.0},
    }
    rng = torch.Generator().manual_seed(3)
    device = torch.device("cpu")
    initial = run_epoch(model, dataset, settings, device)
    # Two shape classes and multiple rotations must be fitted in the same batch.
    for iteration in range(260):
        if iteration in (130, 210):
            for group in optimizer.param_groups:
                group["lr"] *= 0.25
        last = run_epoch(
            model, dataset, settings, device, optimizer=optimizer, generator=rng
        )
    final = run_epoch(model, dataset, settings, device)
    record_property("initial_terms", json.dumps(initial))
    record_property("final_terms", json.dumps(final))
    assert all(torch.isfinite(torch.tensor(list(last.values()))))
    assert final["total"] < 0.025 and final["total"] < initial["total"] * 0.1, (
        initial,
        final,
    )
    assert final["peg_position"] + final["hole_position"] < 0.005
    assert final["peg_rotation"] + final["hole_rotation"] < 0.015
    assert final["size"] < 0.005
    if name.startswith("object_"):
        assert final["shape"] < 0.005
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("name", MODELS)
def test_resume_exact_and_validation_selects_checkpoint(tmp_path, monkeypatch, name):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("EXPERIMENTS_DIR", str(tmp_path / "experiments"))

    def config(epochs, resume=None):
        return {
            "seed": 5,
            "device": "cpu",
            "model": {"name": name, "hidden": 16, "layers": 1, "dropout": 0.1},
            "task": {"shape_weight": 0.25},
            "data": {"kind": "synthetic", "history": 3, "steps": 3, "num_workers": 0},
            "training": {
                "epochs": epochs,
                "batch_size": 1,
                "gradient_clip_norm": 1.0,
                "cpu_threads": 1,
            },
            "optimizer": {"lr": 0.003},
            "scheduler": None,
            "output": {"root": str(tmp_path / "runs")},
            "resume": resume,
        }

    full = fit(config(2))
    partial = fit(config(1))
    resumed = fit(config(2, str(partial / "last.pt")))
    expected = torch.load(full / "last.pt", weights_only=True)
    actual = torch.load(resumed / "last.pt", weights_only=True)
    assert actual["epoch"] == 2
    assert expected["best_loss"] == actual["best_loss"]
    for key, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][key], rtol=0, atol=0)
    for key, value in expected["best_model"].items():
        torch.testing.assert_close(value, actual["best_model"][key], rtol=0, atol=0)
    assert (resumed / "last.pt.tracking.json").is_file()


def test_rotation_metric_sign_invariant_and_percentiles():
    target = torch.zeros(2, 21)
    target[:, [3, 13]] = 1.0
    prediction = target.clone()
    prediction[:, 3:7] *= -1
    prediction[1, :3] = torch.tensor([0.003, 0.004, 0])
    result = summarize_errors(prediction, target)
    assert result["peg_rotation_error_deg"]["p95"] == 0
    assert result["peg_position_error_mm"]["mean"] == pytest.approx(2.5)
    assert result["peg_position_error_mm"]["p90"] == pytest.approx(4.5)


@pytest.mark.parametrize("name", MODELS)
def test_cpu_bfloat16_training_and_evaluation(name):
    device = torch.device("cpu")
    model = build_model(name, hidden=16, layers=1, history=2).to(device)
    dataset = PreparedWindows(synthetic_episodes(2)[:2], history=2, max_edge_m=0.08)
    settings = {
        "task": {"shape_weight": 0.25},
        "training": {"batch_size": 2, "gradient_clip_norm": 1.0, "amp": True},
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trained = run_epoch(model, dataset, settings, device, optimizer=optimizer)
    evaluated = run_epoch(model, dataset, settings, device)
    assert torch.isfinite(torch.tensor([trained["total"], evaluated["total"]])).all()
    assert not model.training


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable on the test host"
)
@pytest.mark.parametrize("name", MODELS)
def test_cuda_training(name):
    device = torch.device("cuda")
    model = build_model(name, hidden=16, layers=1, history=2).to(device)
    dataset = PreparedWindows(synthetic_episodes(2)[:2], history=2, max_edge_m=0.08)
    settings = {
        "task": {"shape_weight": 0.25},
        "training": {
            "batch_size": 2,
            "gradient_clip_norm": 1.0,
            "amp": torch.cuda.is_bf16_supported(),
        },
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trained = run_epoch(model, dataset, settings, device, optimizer=optimizer)
    assert torch.isfinite(torch.tensor(trained["total"]))


def test_training_checkpoint_evaluation_and_stress_csv(tmp_path, monkeypatch):
    import csv
    import json
    from pathlib import Path

    from evaluation.state21 import evaluate
    from main import load_config
    from training.state21 import effective_config

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    config = effective_config(
        load_config(Path("graph/configs/state_estimation/spatial.yaml"))
    )
    config["output"]["root"] = str(tmp_path / "runs")
    config["training"]["epochs"] = 1
    config["model"].update(hidden=16, layers=1)
    config["data"]["steps"] = 3
    config["evaluation"].update(warmup=0, stress_steps=8)
    run_path = fit(config)
    config["evaluation"]["checkpoint"] = str(run_path / "last.pt")
    result_path = evaluate(config)
    result = json.loads(result_path.read_text())
    assert result["source_run_id"] == run_path.name
    assert result["split"] == "test"
    test = result["measurements"]["test"]
    assert test["count"] == 6
    assert test["window_limit"] == 1
    assert set(test["latency_ms_by_window"]) == {"1"}

    stress_path = evaluate(config, stress=True)
    stress = json.loads(stress_path.read_text())
    assert len(stress["measurements"]) == 10
    with (stress_path.parent / "reappearance.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["visibility_phase"] for row in rows} == {
        "before_occlusion",
        "occlusion",
        "after_occlusion",
    }
    config["data"]["steps"] = 4
    with pytest.raises(ValueError, match="same dataset and model recipe"):
        evaluate(config)


@pytest.mark.parametrize("name", MODELS)
def test_targets_and_hidden_vertex_coordinates_do_not_enter_model(name):
    from data_collection.state21 import Episode, Sample
    from graph.state_estimation.graph import Source, batch_graphs, build_graph
    from tests.test_state21_temporal_first import frames

    raw = frames()
    hidden_index = 1
    changed = list(raw)
    changed[hidden_index] = replace(
        raw[hidden_index],
        nodes=tuple(
            replace(point, position=(-900.0, 30.0, float("nan")))
            if point.source == Source.MISSING
            else point
            for point in raw[hidden_index].nodes
        ),
    )
    reliable = build_model(name, hidden=16, layers=1, history=3).eval()
    original = [batch_graphs([build_graph(frame)]) for frame in raw]
    modified = [batch_graphs([build_graph(frame)]) for frame in changed]
    torch.testing.assert_close(
        reliable.forward_sequence(original), reliable.forward_sequence(modified)
    )
    observed = Episode(
        "one",
        "origin",
        "train",
        tuple(Sample(frame, frame.index, torch.zeros(21)) for frame in raw),
    )
    different_targets = replace(
        observed,
        samples=tuple(
            replace(sample, target=torch.ones(21) * 19) for sample in observed.samples
        ),
    )
    device = torch.device("cpu")
    original_batches = PreparedWindows([observed], 3, 0.08).batches(1, device)
    changed_batches = PreparedWindows([different_targets], 3, 0.08).batches(1, device)
    for (old_inputs, old_target, _), (new_inputs, new_target, _) in zip(
        original_batches, changed_batches, strict=True
    ):
        assert not torch.equal(old_target, new_target)
        torch.testing.assert_close(
            reliable(old_inputs)["object21"], reliable(new_inputs)["object21"]
        )


def test_training_configuration_errors():
    from training.state21 import effective_config, prepare_dataset, run

    with pytest.raises(ValueError, match="Unknown Object21D model"):
        build_model("other")
    with pytest.raises(ValueError, match="history"):
        build_model("spatial", history=0)
    with pytest.raises(ValueError, match="hidden"):
        build_model("spatial", hidden=1)
    config = {
        "phase": "other",
        "model": {"name": "spatial"},
        "data": {"history": 2},
        "task": {},
        "training": {},
    }
    with pytest.raises(ValueError, match="Unknown Object21D phase"):
        run(config)
    with pytest.raises(ValueError, match="constant learning rate"):
        effective_config(config | {"scheduler": "unavailable"})
    with pytest.raises(ValueError, match="num_workers"):
        effective_config(config | {"data": {"history": 2, "num_workers": 2}})
    with pytest.raises(ValueError, match="no test episodes"):
        prepare_dataset([], effective_config(config), "test")


def test_split_audit_cli_and_empty_preparation(tmp_path, capsys):
    from main import run

    config = tmp_path / "audit.yaml"
    config.write_text(
        "trainer: training.state21:run\n"
        "base_config: graph/configs/state_estimation/compare.yaml\n"
        "phase: audit\n"
    )
    run(config)
    assert '"train": 2' in capsys.readouterr().out
    with pytest.raises(ValueError, match="history"):
        PreparedWindows([], history=0, max_edge_m=0.08)
    empty = PreparedWindows([], history=3, max_edge_m=0.08)
    with pytest.raises(ValueError, match="batch_size"):
        list(empty.batches(0, torch.device("cpu")))
