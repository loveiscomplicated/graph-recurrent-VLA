"""Insertion geometry, differentiable losses, precision metrics and small GRU ablations."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.insertion_sequences import validate_insertion_splits
from data_collection.temporal.pose_sequences import TemporalPoseDataset
from evaluation.gnn.temporal.precision import (
    motion_precision_groups,
    precision_diagnostics,
    precision_errors,
    precision_summary,
    selection_score,
    validation_precision,
)
from graph.pose_geometry import geometric_endpoint, hole_frame_vector, rotvec_quaternion
from graph.temporal.pose_gru import StreamingPoseEstimator, TemporalPoseGRU
from main import run
from tests.test_temporal_hole_gru import write_smoke_fixture
from tests.test_temporal_pose_gru import loss_settings, pose_sequences
from training.gnn.data import canonical_hash, load_cached_sample, save_cached_sample
from training.gnn.temporal.pose_loss import (
    INSERTION_ABLATIONS,
    insertion_loss,
    pose_loss,
)


def insertion_settings(ablation="pose_relative_full"):
    settings = loss_settings()
    settings["ablation"] = ablation
    settings["weights"].update(
        relative_position=0.25, lateral_position=0.25, relative_orientation=0.1
    )
    return settings


def target_pose():
    target = torch.zeros(1, 2, 21)
    target[..., 3] = target[..., 13] = 1
    target[..., 17:21] = torch.tensor([0.05, 0.01, 0.01, 0.013])
    target[..., 7:10] = geometric_endpoint(target)
    return target


class InsertionPrecisionTests(unittest.TestCase):
    def test_endpoint_plus_x_sign_and_rotated_hole_lateral_axes(self):
        target = target_pose()
        prediction = target.clone()
        target[..., 13:17] = rotvec_quaternion(torch.tensor([0.0, 0.0, torch.pi / 2]))
        prediction[..., 13:17] = target[..., 13:17]
        # Hole local [X,Y,Z]=[2,1,2]mm is world [-1,2,2]mm at +90deg yaw.
        prediction[..., :3] += torch.tensor([-0.001, 0.002, 0.002])
        error = precision_errors(prediction, target)
        torch.testing.assert_close(error["signed_gt_hole_x_mm"], torch.full((1, 2), 2.0))
        torch.testing.assert_close(error["signed_gt_hole_y_mm"], torch.full((1, 2), 1.0))
        torch.testing.assert_close(error["lateral_mm"], torch.full((1, 2), 5.0**0.5))
        torch.testing.assert_close(error["relative_3d_mm"], torch.full((1, 2), 3.0))
        torch.testing.assert_close(geometric_endpoint(target)[..., 0], torch.full((1, 2), 0.05))

    def test_relative_loss_recomputes_geometry_and_reaches_both_poses(self):
        target = target_pose()
        prediction = target.clone()
        prediction[..., :3] += torch.tensor([0.002, 0.003, 0.001])
        prediction[..., 7:10] = 999.0  # Must not be used by the training objective.
        prediction.requires_grad_(True)
        settings = insertion_settings("pose_relative")
        settings["weights"] = {name: 0.0 for name in settings["weights"]}
        settings["weights"]["relative_position"] = 1.0
        loss, _ = insertion_loss(
            prediction, target, torch.ones(1, 2, dtype=torch.bool), "joint_peg_hole", settings
        )
        loss.backward()
        self.assertGreater(float(prediction.grad[..., :3].abs().sum()), 0)
        self.assertGreater(float(prediction.grad[..., 10:13].abs().sum()), 0)
        torch.testing.assert_close(prediction.grad[..., :3], -prediction.grad[..., 10:13])
        self.assertEqual(float(prediction.grad[..., 7:10].abs().sum()), 0)
        self.assertGreater(float(prediction.grad[..., 3:7].abs().sum()), 0)

        model = TemporalPoseGRU()
        inputs, target = TemporalPoseDataset(pose_sequences())[0]
        output, _ = model(inputs[None])
        loss, _ = insertion_loss(
            output, target[None], torch.ones(1, 5, dtype=torch.bool), "joint_peg_hole", settings
        )
        loss.backward()
        for name in ("peg", "hole"):
            self.assertGreater(float(model.heads[name][-1].weight.grad.abs().sum()), 0)

    def test_lateral_loss_is_error_not_a_force_to_insert(self):
        target = target_pose()
        target[..., :3] = torch.tensor([0.2, 0.1, 0.15])
        target[..., 7:10] = geometric_endpoint(target)
        mask = torch.ones(1, 2, dtype=torch.bool)
        _, terms = insertion_loss(target, target, mask, "joint_peg_hole", insertion_settings())
        self.assertEqual(float(terms["lateral_position"]), 0.0)
        self.assertEqual(float(terms["relative_position"]), 0.0)

    def test_relative_orientation_q_sign_and_gradient_to_both_heads(self):
        target = target_pose()
        prediction = target.clone()
        prediction[..., 3:7] = rotvec_quaternion(torch.tensor([0.0, 0.0, 0.1]))
        prediction.requires_grad_(True)
        settings = insertion_settings()
        settings["weights"] = {name: 0.0 for name in settings["weights"]}
        settings["weights"]["relative_orientation"] = 1.0
        mask = torch.ones(1, 2, dtype=torch.bool)
        loss, _ = insertion_loss(prediction, target, mask, "joint_peg_hole", settings)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad[..., 3:7].abs().sum()), 0)
        self.assertGreater(float(prediction.grad[..., 13:17].abs().sum()), 0)
        flipped = prediction.detach().clone()
        flipped[..., 3:7] *= -1
        flipped[..., 13:17] *= -1
        same, _ = insertion_loss(flipped, target, mask, "joint_peg_hole", settings)
        torch.testing.assert_close(same, loss)

    def test_streaming_control_geometry_uses_predictions_only(self):
        data = pose_sequences()
        model = TemporalPoseGRU()
        estimator = StreamingPoseEstimator(model)

        def step(row):
            return estimator.step(
                torch.tensor(data["prediction"][row : row + 1]),
                torch.tensor(data["pose_quality"][row : row + 1]),
                torch.tensor(data["tcp_world"][row : row + 1]),
            )

        initial = step(0)
        torch.testing.assert_close(
            initial["relative_hole_frame"],
            hole_frame_vector(
                initial["world_object21"][:, 7:10], initial["world_object21"][:, 13:17]
            ),
        )
        data["target"][:] = 999
        estimator.reset()
        repeated = step(0)
        torch.testing.assert_close(initial["relative_hole_frame"], repeated["relative_hole_frame"])
        torch.testing.assert_close(
            initial["peg_axis_in_hole_frame"], repeated["peg_axis_in_hole_frame"]
        )

    def test_pose_only_matches_previous_loss_and_padding_is_excluded(self):
        target = target_pose()
        prediction = target.clone()
        prediction[..., :3] += 0.003
        mask = torch.tensor([[True, False]])
        prediction[:, 1] = target[:, 1] = float("nan")
        expected, _ = pose_loss(prediction, target, mask, "joint_peg_hole", loss_settings())
        actual, _ = insertion_loss(
            prediction, target, mask, "joint_peg_hole", insertion_settings("pose_only")
        )
        torch.testing.assert_close(actual, expected)
        metrics = validation_precision(
            prediction, target, mask, {"motion_threshold_m": 0.001, "motion_threshold_rad": 0.01}
        )
        self.assertEqual(len(metrics["relative_3d_mm"]), 1)

    def test_precision_thresholds_are_strict_estimation_fractions(self):
        values = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        summary = precision_summary(
            {"lateral_mm": values, "relative_3d_mm": values}, np.ones(5, dtype=bool)
        )
        self.assertEqual(summary["estimation_accuracy_fractions"]["lateral_lt_1_mm"], 0.2)
        self.assertEqual(summary["estimation_accuracy_fractions"]["lateral_lt_2_mm"], 0.4)
        self.assertEqual(summary["estimation_accuracy_fractions"]["lateral_lt_3_mm"], 0.6)
        self.assertEqual(summary["errors"]["relative_3d_mm"]["maximum"], 4.0)
        self.assertIsNotNone(summary["errors"]["relative_3d_mm"]["p95"])
        self.assertNotIn("success_rate", summary)

    def test_historical_endpoint_outputs_are_preserved_in_evaluation(self):
        data = pose_sequences()
        prediction = data["target"].copy()
        prediction[:, 7] += 0.005
        result, _ = precision_diagnostics(
            data, prediction, {"all": np.ones(len(prediction), dtype=bool)}
        )
        errors = result["groups"]["all"]["errors"]
        self.assertAlmostEqual(errors["relative_3d_mm"]["mean"], 5.0, places=4)
        self.assertLess(errors["pose_reconstructed_relative_3d_mm"]["maximum"], 1e-4)
        self.assertEqual(len(result["episodes"]), 2)
        self.assertEqual(len(result["worst_relative_frames"]), 10)

    def test_motion_reversal_and_co_motion_offline_slices(self):
        data = pose_sequences()
        # Override GT world Peg motion only; input predictions remain untouched.
        world_x = np.tile([0.0, 0.01, 0.02, 0.005, -0.01], 2)
        data["target"][:, 0] = world_x - data["tcp_world"][:, 0]
        settings = dict(
            motion_threshold_m=0.001,
            co_motion_tolerance_m=0.001,
            direction_change_cosine=0.0,
            alignment_axial_min_m=-0.05,
            alignment_axial_max_m=0.02,
            alignment_lateral_m=0.01,
        )
        groups = motion_precision_groups(data, settings)
        self.assertEqual(groups["fast_direction_change"].sum(), 2)
        self.assertGreater(groups["peg_tcp_co_motion_proxy"].sum(), 0)

    def test_checkpoint_selection_uses_relative_and_motion_validation_metrics(self):
        settings = {
            "components": {
                "relative_3d_mm_p90": {"scale": 3.0, "weight": 1.0},
                "peg_moving_position_mm_mean": {"scale": 3.0, "weight": 1.0},
            }
        }
        good = {"relative_3d_mm_p90": 2.0, "peg_moving_position_mm_mean": 2.0}
        lagged = {"relative_3d_mm_p90": 1.0, "peg_moving_position_mm_mean": 10.0}
        self.assertLess(selection_score(good, settings)[0], selection_score(lagged, settings)[0])

    def test_independent_parent_reservation(self):
        source = {"train": ["a", "b"], "validation": ["v"], "test": ["t"]}
        external = {"train": ["new_a"], "validation": ["new_b"], "test": []}
        parents = {name: name for name in ["a", "b", "v", "t", "new_a", "new_b"]}
        manifest = {
            "test_role": "independent",
            "splits": {"train": ["a"], "validation": ["b"], "test": ["new_a"]},
            "parent_by_episode": parents,
        }
        validate_insertion_splits(manifest, source, external)
        broken = deepcopy(manifest)
        broken["parent_by_episode"]["new_a"] = "v"
        with self.assertRaisesRegex(ValueError, "previously used"):
            validate_insertion_splits(broken, source, external)

    def test_independent_cache_cli_path_uses_unseen_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_smoke_fixture(root / "source", pose_targets=True)
            external_config = write_smoke_fixture(root / "external", pose_targets=True)
            cache = Path(external_config["data"]["cache_dir"])
            metadata_path = cache / "metadata/data_identity.json"
            inventory_path = cache / "metadata/cache_inventory.json"
            metadata, inventory = (
                json.loads(metadata_path.read_text()),
                json.loads(inventory_path.read_text()),
            )
            mapping = {
                episode: "independent_" + episode
                for episodes in metadata["splits"].values()
                for episode in episodes
            }
            entries = {entry["sample_id"]: entry for entry in inventory["files"]}
            for frame in metadata["frames"]:
                old_id = frame["id"]
                entry = entries[old_id]
                original_identity = dict(
                    frame_sha256=frame["sha256"],
                    sample_id=old_id,
                    preprocessing_sha256=inventory["preprocessing_sha256"],
                )
                batch = load_cached_sample(cache / entry["file"], original_identity, old_id)
                episode, filename = old_id.split("/")
                sample_id = mapping[episode] + "/" + filename
                new_identity = dict(original_identity, sample_id=sample_id)
                path = (
                    cache
                    / inventory["preprocessing_sha256"]
                    / (canonical_hash(new_identity) + ".npz")
                )
                batch.sample_ids = [sample_id]
                save_cached_sample(path, batch, new_identity)
                frame["id"] = sample_id
                entry.update(
                    sample_id=sample_id,
                    file=str(path.relative_to(cache)),
                    sha256=file_sha256(path),
                    size=path.stat().st_size,
                )
            metadata["splits"] = {
                name: [mapping[episode] for episode in episodes]
                for name, episodes in metadata["splits"].items()
            }
            metadata["dataset_sha256"] = canonical_hash(
                {
                    key: value
                    for key, value in metadata.items()
                    if key not in ("dataset_sha256", "sample_counts")
                }
            )
            metadata_path.write_text(json.dumps(metadata))
            inventory_path.write_text(json.dumps(inventory))
            manifest_path = Path(config["split_manifest"])
            manifest = json.loads(manifest_path.read_text())
            manifest["test_role"] = "independent"
            manifest["splits"]["test"] = ["independent_v"]
            manifest["parent_by_episode"].update({episode: episode for episode in mapping.values()})
            manifest_path.write_text(json.dumps(manifest))
            config.update(trainer="training.gnn.temporal.pose_gru:train", mode="joint_peg_hole")
            config["data"]["independent_cache_dir"] = str(cache)
            config["loss"] = insertion_settings("pose_relative")
            path = root / "config.json"
            with patch.dict(
                "os.environ",
                {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(root / "experiments")},
            ):
                path.write_text(json.dumps(config))
                run(path)
                best = next(Path(config["output"]["root"]).glob("*/best.pt"))
                config.update(
                    trainer="evaluation.gnn.temporal.pose_gru:evaluate", pose_checkpoint=str(best)
                )
                config["output"]["root"] = str(root / "evaluation")
                path.write_text(json.dumps(config))
                run(path)
            summary = json.loads(next((root / "evaluation").glob("*/summary.json")).read_text())
            self.assertEqual(summary["split"], "independent_temporal_test")
            self.assertIsNotNone(summary["independent_dataset_sha256"])
            self.assertIn("independent_v", summary["methods"]["gru_joint_peg_hole"]["episodes"])

    def test_tiny_all_ablation_cli_training_evaluation_and_checkpoint_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_smoke_fixture(root, pose_targets=True)
            config.update(trainer="training.gnn.temporal.pose_gru:train", mode="joint_peg_hole")
            original = file_sha256(config["checkpoint"])
            config["loss"] = insertion_settings()
            paths = {}
            config_path = root / "config.json"
            with patch.dict(
                "os.environ",
                {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(root / "experiments")},
            ):
                for ablation in INSERTION_ABLATIONS:
                    config["loss"]["ablation"] = ablation
                    config["output"]["root"] = str(root / ablation)
                    config_path.write_text(json.dumps(config))
                    run(config_path)
                    paths[ablation] = str(next((root / ablation).glob("*/best.pt")))
                config.update(
                    trainer="evaluation.gnn.temporal.pose_gru:evaluate",
                    pose_checkpoint=paths["pose_relative_full"],
                )
                config["comparison"] = {
                    "precision_checkpoints": {
                        key: value for key, value in paths.items() if key != "pose_relative_full"
                    }
                }
                config["output"]["root"] = str(root / "evaluation")
                config_path.write_text(json.dumps(config))
                run(config_path)
            output = next((root / "evaluation").glob("*/summary.json"))
            summary = json.loads(output.read_text())
            self.assertIn("insertion_precision", summary["methods"]["gru_joint_peg_hole"])
            self.assertEqual(len(summary["training_objectives"]), 4)
            payload = torch.load(paths["pose_relative_full"], weights_only=True)
            self.assertIn("selection_score", payload)
            self.assertEqual(
                payload["config"]["loss"]["effective_weights"]["relative_position"], 0.25
            )
            self.assertTrue((output.parent / "precision_trace_000.png").is_file())
            self.assertEqual(original, file_sha256(config["checkpoint"]))


if __name__ == "__main__":
    unittest.main()
