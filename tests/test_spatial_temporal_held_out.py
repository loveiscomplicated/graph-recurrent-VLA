"""Held-out split isolation, fixed-weight evaluation and shared metric contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.spatial_temporal_held_out import evaluate, validate_test_manifest
from evaluation.spatial_temporal_corner_tracking import evaluate_online
from evaluation.spatial_temporal_object21 import evaluate as evaluate_validation
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.spatial_temporal_corner_tracker import SpatialTemporalCornerTracker, load_tracker_checkpoint
from training.gnn.spatial_temporal_corner_data import CornerTrackingDataset
from tests.test_spatial_temporal_corner_tracker import prepared_episode_arrays


def test_test_manifest_requires_disjoint_complete_release_inventory():
    record = {"seed": 7, "policy_mode": "Fixed_Hole_expert", "split": "test",
              "frames": 3, "batch": "fixture"}
    manifest = {"episodes": [record], "condition_filter": None}
    release = {"records": [record, {**record, "seed": 8, "split": "train"}]}
    expected = {"episodes": 1, "frames": 3}
    validate_test_manifest(manifest, release, expected)
    leaking = copy.deepcopy(release)
    leaking["records"][1]["seed"] = 7
    with pytest.raises(ValueError, match="overlap"):
        validate_test_manifest(manifest, leaking, expected)
    with pytest.raises(ValueError, match="inventory"):
        validate_test_manifest(manifest, release, {"episodes": 1, "frames": 4})
    wrong_split = copy.deepcopy(manifest)
    wrong_split["episodes"] = [{**record, "split": "validation"}]
    with pytest.raises(ValueError, match="only test"):
        validate_test_manifest(wrong_split, release, expected)


def test_validation_evaluator_still_rejects_an_unpinned_test_cache(tmp_path):
    cache = tmp_path / "test.pt"
    cache.write_bytes(b"test cache must not replace validation cache")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"format": "spatial_temporal_object21_v1", "rollout_cache_sha256": "other"}, checkpoint)
    with pytest.raises(ValueError, match="cache checksum mismatch"):
        evaluate_validation({"checkpoint": str(checkpoint), "rollout_cache": str(cache)})


def test_held_out_smoke_keeps_weights_fixed_and_outputs_aligned(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    record = {"episode": "five_mode_7_Fixed_Hole_expert", "seed": 7,
              "policy_mode": "Fixed_Hole_expert", "split": "test", "frames": 3,
              "batch": "fixture", "file": "episode.npz"}
    arrays = prepared_episode_arrays()
    target = np.zeros((3, 21), dtype=np.float32)
    target[:, 3] = 1
    target[:, 13] = 1
    target[:, 17:21] = [0.04, 0.01, 0.01, 0.02]
    np.savez_compressed(tmp_path / "episode.npz", **arrays, object21=target)
    record["sha256"] = file_sha256(tmp_path / "episode.npz")
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps({"records": [record]}))
    release_sha = file_sha256(release_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "episodes": [record], "condition_filter": None,
        "source": {"manifest_sha256": release_sha},
    }))
    tracker_settings = {"hidden": 16, "spatial_layers": 1, "temporal_layers": 1,
                        "heads": 4, "association_dim": 8, "coordinate_frame": "world"}
    tracker = SpatialTemporalCornerTracker(**tracker_settings)
    tracker_path = tmp_path / "tracker.pt"
    torch.save({
        "format": "spatial_temporal_corner_tracker_v1", "model": tracker.state_dict(),
        "identity": {"max_edge_m": 0.08},
        "config": {"seed": 0, "model": tracker_settings, "training": {"log_every_batches": 0}},
    }, tracker_path)
    association = {"distance_limit_m": 0.03, "embedding_weight_m": 0.0}
    gnn_settings = {"num_semantic_ids": 24, "hidden": 16, "layers": 1,
                    "use_semantic_ids": False, "extra_node_dim": 3}
    gnn = VertexEdgeGatedMPNN(**gnn_settings)
    object_path = tmp_path / "object21.pt"
    torch.save({
        "format": "spatial_temporal_object21_v1", "model": gnn.state_dict(),
        "identity": {"manifest_sha256": release_sha,
                     "tracker_checkpoint_sha256": file_sha256(tracker_path),
                     "association": association},
        "config": {"gnn": gnn_settings, "training": {"max_tracks_per_frame": 128},
                   "downstream_features": {"uncertainty_scale_m": 0.01, "age_scale_frames": 10},
                   "loss": {"pos_unit_m": 0.01, "consistency_weight": 0.05,
                            "size_units_m": [0.04, 0.01, 0.01, 0.02]}},
    }, object_path)
    config = {
        "device": "cpu", "tracker": {"checkpoint": str(tracker_path), "sha256": file_sha256(tracker_path)},
        "object21": {"checkpoint": str(object_path), "sha256": file_sha256(object_path)},
        "association": association,
        "data": {"manifest": str(manifest_path), "release_manifest": str(release_path)},
        "expected": {"episodes": 1, "frames": 3},
        "evaluation": {"frame_batch_size": 2}, "output": {"root": str(tmp_path / "results")},
    }
    def forbid_optimizer(*args, **kwargs):
        raise AssertionError("Held-out evaluation must not construct an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", forbid_optimizer)
    output = Path(evaluate(config))
    result = json.loads((output / "results.json").read_text())
    assert result["split"] == "test" and result["frames"] == 3
    assert result["test_used_for_selection"] is False and result["weights_frozen"] is True
    assert result["detector_only_online"]["reappearance_same_id_rate"] == 1
    reference_model, _ = load_tracker_checkpoint(tracker_path, config["tracker"]["sha256"], "cpu")
    reference_dataset = CornerTrackingDataset(manifest_path, "test")
    reference = evaluate_online(
        reference_model, reference_dataset, {**association, "log_every_episodes": 0}, torch.device("cpu")
    )
    assert result["detector_only_online"] == reference
    assert file_sha256(tracker_path) == config["tracker"]["sha256"]
    assert file_sha256(object_path) == config["object21"]["sha256"]
    with np.load(output / "predictions.npz", allow_pickle=False) as predictions:
        assert predictions["object21"].shape == (3, 21)
        np.testing.assert_array_equal(predictions["target"], target)
        assert predictions["sample_ids"].tolist() == [f"7:Fixed_Hole_expert:{i}" for i in range(3)]
    changed = copy.deepcopy(config)
    changed["association"]["distance_limit_m"] = 0.05
    with pytest.raises(ValueError, match="association settings differ"):
        evaluate(changed)
