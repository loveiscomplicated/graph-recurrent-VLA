"""Validation-only orchestration, checkpoint pins and association safety gates."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from data_collection.episode_dataset import file_sha256
from evaluation.spatial_temporal_association_calibration import calibration_indices, selection_key
from training.spatial_temporal_workflow import run


def test_calibration_keeps_complete_validation_seed_groups():
    class Dataset:
        records = [
            {"seed": seed, "policy_mode": mode, "split": "validation"}
            for seed in range(10) for mode in range(5)
        ]
    indices = calibration_indices(Dataset(), 3)
    assert len(indices) == 15
    assert {Dataset.records[index]["seed"] for index in indices} == {0, 4, 9}
    good_identity = {
        "reappearance_same_id_rate": 0.95, "identity_switches": 2,
        "known_track_conflicts": 1, "known_detections": 1000,
        "missing_rmse_mm": 20, "mean_final_tracks": 24,
    }
    bad_identity = {**good_identity, "reappearance_same_id_rate": 0.5, "missing_rmse_mm": 1}
    assert selection_key(good_identity) < selection_key(bad_identity)


@pytest.mark.parametrize("max_tracks", [24, 129])
def test_workflow_pins_selected_weights_and_stops_on_track_growth(tmp_path, monkeypatch, max_tracks):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    names = ["train_tracker", "calibrate_association", "evaluate_tracker", "train_object21", "evaluate_object21"]
    paths = {}
    for name in names:
        recipe = {"trainer": f"mock:{name}", "device": "cpu"}
        if name == "train_tracker":
            recipe.update(seed=0, data={"num_workers": 0})
        if name == "train_object21":
            recipe.update(data={}, training={"max_tracks_per_frame": 128})
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(recipe))
        paths[name] = str(path)
    calls = []
    association = {"distance_limit_m": 0.02, "embedding_weight_m": 0.005}

    def dispatch(recipe):
        name = recipe["trainer"].split(":")[1]
        calls.append((name, copy.deepcopy(recipe)))
        output = tmp_path / "results" / name
        output.mkdir(parents=True)
        if name in ("train_tracker", "train_object21"):
            (output / "best.pt").write_bytes(name.encode())
        if name == "calibrate_association":
            assert recipe["checkpoint_sha256"] == file_sha256(Path(recipe["checkpoint"]))
            (output / "results.json").write_text(json.dumps({"selected_association": association}))
        if name == "evaluate_tracker":
            assert recipe["association"] == association
            (output / "results.json").write_text(json.dumps({
                "detector_only_online": {"max_final_tracks": max_tracks},
            }))
        if name == "train_object21":
            assert recipe["tracker"]["sha256"] == file_sha256(Path(recipe["tracker"]["checkpoint"]))
            assert recipe["association"] == association
            (output / "frozen_tracker_rollouts.pt").write_bytes(b"frozen-cache")
        if name == "evaluate_object21":
            assert Path(recipe["rollout_cache"]).parent == Path(recipe["checkpoint"]).parent
            assert recipe["checkpoint_sha256"] == file_sha256(Path(recipe["checkpoint"]))
        return str(output)

    monkeypatch.setattr("training.spatial_temporal_workflow.load_trainer", lambda _: dispatch)
    config = {"configs": paths, "execution": {"device": "cpu", "num_workers": 3},
              "output": {"root": str(tmp_path / "workflow")}}
    if max_tracks > 128:
        with pytest.raises(ValueError, match="Full validation exceeds"):
            run(config)
        assert [name for name, _ in calls] == names[:3]
    else:
        output = Path(run(config))
        progress = json.loads((output / "progress.json").read_text())
        assert progress["status"] == "completed"
        assert progress["test_episode_files_opened"] == 0
        assert [name for name, _ in calls] == names
    assert calls[0][1]["data"]["num_workers"] == 3
