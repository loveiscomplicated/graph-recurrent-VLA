"""Joint temporal pose geometry, masking, GT isolation and small real-cache workflow."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.pose_sequences import (
    TemporalPoseDataset,
    attach_proprioception,
    collate_pose_episodes,
    fit_observation_normalization,
)
from evaluation.gnn.temporal.pose_gru import (
    diagnostic_groups,
    lag_and_drift,
    load_pose_model,
    pose_errors,
    replay_pose,
    score_pose_methods,
)
from graph.pose_geometry import (
    angular_error_rad,
    geometric_endpoint,
    quaternion_product,
    quaternion_rotvec,
    rotvec_quaternion,
    unit_quaternion,
)
from graph.temporal.pose_gru import StreamingPoseEstimator, TemporalPoseGRU
from main import run
from tests.test_temporal_hole_gru import synthetic_sequences, write_smoke_fixture
from training.gnn.temporal.config import resolve_pose_config
from training.gnn.temporal.frozen_source import infer_pose_data, prepare_pose_sequences
from training.gnn.temporal.pose_loss import pose_loss


def pose_sequences(with_robot=False):
    """Generate observed geometry plus separate labels for moving Peg/static Hole."""
    data = synthetic_sequences()
    count = len(data["frame"])
    for name in ("prediction", "target"):
        data[name][:, 3] = data[name][:, 13] = 1.0
        data[name][:, 17:21] = [0.05, 0.01, 0.01, 0.02]
        data[name][:, :3] = np.array([0.0, 0.1, 0.25]) - data["tcp_world"]
        data[name][:, 0] += data["frame"] * 0.003
        data[name][:, 3:7] = rotvec_quaternion(
            torch.tensor(np.c_[np.zeros(count), np.zeros(count), data["frame"] * 0.05])
        ).numpy()
        data[name][:, 7:10] = geometric_endpoint(torch.tensor(data[name])).numpy()
    peg = np.tile(np.array([3, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32), (count, 1))
    peg[[2, 3, 7, 8]] = [0, 0, 1, 0, 0, 0, 0, 0]
    data["pose_quality"] = np.c_[peg, data["features"]]
    if with_robot:
        data["tcp_quaternion"] = (
            rotvec_quaternion(
                torch.tensor(np.c_[np.zeros(count), np.zeros(count), data["frame"] * 0.01])
            )
            .float()
            .numpy()
        )
        data["gripper_qpos"] = np.full((count, 2), 0.02, dtype=np.float32)
    return data


def loss_settings():
    return {
        "position_unit_m": 0.01,
        "rotation_unit_rad": 0.1,
        "weights": {
            name: 1.0 for name in ("peg_position", "peg_rotation", "hole_position", "hole_rotation")
        },
    }


class TemporalPoseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.data = pose_sequences()
        self.model = TemporalPoseGRU()

    def test_quaternion_sign_composition_and_identity_gradients(self):
        vector = torch.zeros(2, 3, requires_grad=True)
        q = rotvec_quaternion(vector)
        q.sum().backward()
        self.assertTrue(torch.isfinite(vector.grad).all())
        torch.testing.assert_close(angular_error_rad(q, -q), torch.zeros(2))
        quarter = rotvec_quaternion(torch.tensor([[0.0, 0.0, torch.pi / 2]]))
        half = quaternion_product(quarter, quarter)
        torch.testing.assert_close(quaternion_rotvec(half).norm(dim=-1), torch.tensor([torch.pi]))
        with self.assertRaises(ValueError):
            unit_quaternion(torch.zeros(1, 4))

    def test_feature_schema_robot_motion_and_episode_reset(self):
        dataset = TemporalPoseDataset(pose_sequences(True))
        inputs, target = dataset[0]
        self.assertEqual(inputs.shape, (5, 62))
        torch.testing.assert_close(inputs[0, 49:52], torch.zeros(3))
        torch.testing.assert_close(inputs[1, 49:52], torch.tensor([0.01, 0, 0]))
        self.assertEqual(inputs[0, 52].item(), 0)
        self.assertEqual(inputs[1, 52].item(), 1)
        torch.testing.assert_close(
            inputs[1, 57:60], torch.tensor([0.0, 0.0, 0.01]), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(dataset[1][0][0, 49:52], torch.zeros(3))
        self.assertEqual(TemporalPoseDataset(self.data)[0][0].shape[-1], 53)
        changed = deepcopy(self.data)
        changed["target"][:] = 1000
        torch.testing.assert_close(
            TemporalPoseDataset(changed)[0][0], TemporalPoseDataset(self.data)[0][0]
        )

    def test_train_only_normalization_ignores_targets_and_validation(self):
        dataset = TemporalPoseDataset(self.data)
        mean, std = fit_observation_normalization(dataset)
        values = torch.cat([dataset[index][0] for index in range(len(dataset))])
        torch.testing.assert_close(mean, values.mean(0), atol=1e-6, rtol=1e-5)
        self.assertTrue((std > 0).all())
        self.data["target"][:] = -999
        again = fit_observation_normalization(TemporalPoseDataset(self.data))
        torch.testing.assert_close(again[0], mean)
        torch.testing.assert_close(again[1], std)
        self.model.set_normalization(mean, std)
        self.assertIn("input_mean", self.model.state_dict())

    def test_full_streaming_causal_and_reset_for_all_modes_and_robot_options(self):
        for with_robot in (False, True):
            data = pose_sequences(with_robot)
            for mode in ("hole_only", "peg_only", "joint_peg_hole"):
                model = TemporalPoseGRU(mode=mode, with_robot=with_robot)
                prediction, difference = replay_pose(model, data, "cpu")
                self.assertLess(difference, 1e-6)
                modified = deepcopy(data)
                modified["prediction"][4, :3] += 10
                modified["target"][:] = 1000
                changed, _ = replay_pose(model, modified, "cpu")
                np.testing.assert_array_equal(prediction[:4], changed[:4])
                np.testing.assert_array_equal(prediction[5:], changed[5:])
                inactive = slice(0, 7) if mode == "hole_only" else slice(10, 17)
                if mode != "joint_peg_hole":
                    np.testing.assert_allclose(
                        prediction[:, inactive], data["prediction"][:, inactive], atol=1e-7
                    )

    def test_padding_masks_loss_gradients_and_last_hidden(self):
        dataset = TemporalPoseDataset(self.data)
        first, second = dataset[0], dataset[1]
        second = (second[0][:2], second[1][:2])
        observations, target, mask = collate_pose_episodes([first, second])
        prediction, hidden = self.model(observations, mask=mask)
        single, single_hidden = self.model(second[0][None])
        torch.testing.assert_close(hidden[:, 1:2], single_hidden, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(prediction[1, :2], single[0], atol=1e-6, rtol=1e-5)
        target[~mask] = float("nan")
        loss, terms = pose_loss(prediction, target, mask, self.model.mode, loss_settings())
        loss.backward()
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None)
        )
        self.assertTrue(all(value.item() >= 0 for value in terms.values()))
        self.assertGreater(float(self.model.heads["peg"][-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.heads["hole"][-1].weight.grad.abs().sum()), 0)
        bad = mask.clone()
        bad[0, 1] = False
        with self.assertRaises(ValueError):
            self.model(observations, mask=bad)

    def test_loss_units_weights_and_q_sign_equivalence(self):
        dataset = TemporalPoseDataset(self.data)
        _, target = dataset[0]
        target = target[None]
        prediction = target.clone()
        prediction[:, :, 3:7] *= -1
        prediction[:, :, 13:17] *= -1
        mask = torch.ones(1, 5, dtype=torch.bool)
        loss, _ = pose_loss(prediction, target, mask, "joint_peg_hole", loss_settings())
        self.assertLess(float(loss), 1e-4)
        prediction[:, :, 0] += 0.01
        settings = loss_settings()
        _, terms = pose_loss(prediction, target, mask, "joint_peg_hole", settings)
        self.assertAlmostEqual(float(terms["peg_position"]), 1 / 6, places=5)
        settings["weights"]["peg_position"] = 0
        zero, _ = pose_loss(prediction, target, mask, "joint_peg_hole", settings)
        self.assertLess(float(zero), 1e-4)

    def test_geometric_endpoint_gt_definition_and_state38(self):
        inputs, target = TemporalPoseDataset(self.data)[0]
        torch.testing.assert_close(geometric_endpoint(target), target[:, 7:10])
        estimator = StreamingPoseEstimator(self.model)
        robot, freshness = torch.randn(1, 15), torch.randn(1, 2)
        result = estimator.step(
            torch.tensor(self.data["prediction"][:1]),
            torch.tensor(self.data["pose_quality"][:1]),
            torch.tensor(self.data["tcp_world"][:1]),
            robot15=robot,
            freshness2=freshness,
        )
        refined = result["object21"]
        torch.testing.assert_close(
            refined[:, 17:21], torch.tensor(self.data["prediction"][:1, 17:21])
        )
        torch.testing.assert_close(refined[:, 7:10], geometric_endpoint(refined))
        torch.testing.assert_close(result["state38"][:, :15], robot)
        torch.testing.assert_close(result["state38"][:, 36:], freshness)
        torch.testing.assert_close(result["state38"][:, 15:36], refined)

    def test_parameter_count_and_position_modes(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 95244)
        self.assertEqual(
            sum(p.numel() for p in TemporalPoseGRU(with_robot=True).parameters()), 95820
        )
        self.assertEqual(
            sum(p.numel() for p in TemporalPoseGRU(mode="peg_only").parameters()), 86598
        )
        inputs, _ = TemporalPoseDataset(self.data)[0]
        for option in ("direct", "residual"):
            model = TemporalPoseGRU(position_mode=option)
            for head in model.heads.values():
                for parameter in head.parameters():
                    torch.nn.init.zeros_(parameter)
            output, _ = model(inputs[None])
            expected = inputs[:, :3] if option == "residual" else torch.zeros(5, 3)
            torch.testing.assert_close(output[0, :, :3], expected)

    def test_missing_frame_gradient_reaches_past_observations(self):
        inputs, target = TemporalPoseDataset(self.data)[0]
        inputs.requires_grad_(True)
        output, _ = self.model(inputs[None])
        (output[0, 3, :3] - target[3, :3]).square().sum().backward()
        self.assertGreater(float(inputs.grad[0].abs().sum()), 0)

    def test_dynamic_lag_groups_and_episode_metrics(self):
        settings = {
            "visibility_labels": None,
            "phase_labels": None,
            "motion_threshold_m": 0.001,
            "motion_threshold_rad": 0.01,
            "long_occlusion_frames": 2,
            "lag_max_frames": 1,
            "bad_error_mm": 10.0,
            "recovery_rotation_deg": 10.0,
        }
        groups, _ = diagnostic_groups(self.data, settings)
        self.assertEqual(groups["peg_moving"].sum(), 8)
        summary, errors, events = score_pose_methods(
            self.data, {"perfect": self.data["target"]}, settings
        )
        self.assertEqual(len(summary["methods"]["perfect"]["episodes"]), 2)
        self.assertLess(
            summary["methods"]["perfect"]["groups"]["all"]["peg_rotation"]["mean_deg"], 1e-4
        )
        self.assertEqual(len(events["perfect"]["peg"]), 2)
        self.assertEqual(
            lag_and_drift(self.data, self.data["target"], groups["peg_moving"], 1)[0]["lag_frames"],
            0,
        )
        self.assertEqual(len(pose_errors(self.data["prediction"], self.data["target"])), 5)
        stationary = deepcopy(self.data)
        stationary["target"][2, :3] = (
            stationary["target"][1, :3] + stationary["tcp_world"][1] - stationary["tcp_world"][2]
        )
        stationary["target"][2, 3:7] = stationary["target"][1, 3:7]
        static_groups, _ = diagnostic_groups(stationary, settings)
        self.assertTrue(static_groups["peg_stationary"][2])
        self.assertFalse(static_groups["peg_moving"][2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase_labels.npz"
            phases = np.repeat(["before_grasp", "after_grasp"], 5)
            np.savez(path, sample_ids=self.data["sample_ids"][::-1], phase=phases[::-1])
            settings["phase_labels"] = str(path)
            phase_groups, _ = diagnostic_groups(self.data, settings)
            self.assertEqual(phase_groups["phase_before_grasp"].sum(), 5)
            self.assertEqual(phase_groups["phase_after_grasp"].sum(), 5)

    def test_observed_robot_join_checksum_and_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_smoke_fixture(root, pose_targets=True, raw_proprioception=True)
            config["data"]["raw_root"] = str(root / "raw")
            effective = resolve_pose_config(config)
            frozen, loaders, identity = prepare_pose_sequences(
                effective, torch.device("cpu"), ("train",)
            )
            data = infer_pose_data(frozen, loaders["train"], "cpu", identity, effective)
            self.assertEqual(TemporalPoseDataset(data)[0][0].shape[-1], 62)
            sample = root / "raw" / str(data["sample_ids"][0])
            sample.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "published graph source"):
                attach_proprioception(data, identity, root / "raw")
            np.savez(sample, **{"ground_truth/ee_pose": np.array([[0, 0, 0, 1, 0, 0, 0]])})
            modified_identity = deepcopy(identity)
            for frame in modified_identity["frames"]:
                if frame["id"] == str(data["sample_ids"][0]):
                    frame["sha256"] = file_sha256(sample)
            with self.assertRaisesRegex(ValueError, "no GT fallback"):
                attach_proprioception(data, modified_identity, root / "raw")

    def test_real_cache_three_modes_train_and_paired_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_smoke_fixture(root, pose_targets=True)
            config.update(trainer="training.gnn.temporal.pose_gru:train")
            config["training"]["episode_batch_size"] = 2
            config["evaluation"].update(lag_max_frames=1)
            original = file_sha256(config["checkpoint"])
            checkpoints = {}
            path = root / "config.json"
            environment = {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(root / "experiments")}
            with patch.dict("os.environ", environment):
                for mode in ("hole_only", "peg_only", "joint_peg_hole"):
                    config["mode"] = mode
                    config["output"]["root"] = str(root / mode)
                    path.write_text(json.dumps(config))
                    run(path)
                    checkpoints[mode] = str(next((root / mode).glob("*/best.pt")))
                config.update(
                    trainer="evaluation.gnn.temporal.pose_gru:evaluate",
                    pose_checkpoint=checkpoints["joint_peg_hole"],
                )
                config["comparison"] = {
                    "gru_checkpoints": {
                        mode: value
                        for mode, value in checkpoints.items()
                        if mode != "joint_peg_hole"
                    }
                }
                config["output"]["root"] = str(root / "evaluation")
                path.write_text(json.dumps(config))
                run(path)
            summary_path = next((root / "evaluation").glob("*/summary.json"))
            summary = json.loads(summary_path.read_text())
            self.assertEqual(len(summary["methods"]), 6)
            self.assertEqual(summary["frames"], 5)
            self.assertIn("learned_gate", summary["unavailable_comparisons"])
            self.assertEqual(set(summary["performance"]), set(summary["methods"]))
            self.assertTrue((summary_path.parent / "pose_trace_000.png").is_file())
            self.assertEqual(original, file_sha256(config["checkpoint"]))
            effective = resolve_pose_config(config)
            _, _, identity = prepare_pose_sequences(effective, torch.device("cpu"), ("test",))
            effective["source_checkpoint_sha256"] = "wrong_source"
            with self.assertRaisesRegex(ValueError, "source_checkpoint_sha256"):
                load_pose_model(checkpoints["hole_only"], effective, identity, torch.device("cpu"))
            from utils.experiment_artifacts import plan_upload

            plan_upload(root / "experiments" / summary_path.parent.name, {"metrics"})


if __name__ == "__main__":
    unittest.main()
