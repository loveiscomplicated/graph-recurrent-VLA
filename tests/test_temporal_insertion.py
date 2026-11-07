"""Insertion geometry, actual selective SSM caches, leakage and root workflow tests."""

import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.insertion_sequences import (
    InsertionEpisodeDataset,
    InsertionWindowDataset,
    validate_insertion_splits,
)
from data_collection.temporal.pose_sequences import (
    collate_pose_episodes,
    fit_observation_normalization,
)
from evaluation.gnn.temporal.insertion import (
    compare_existing_gru,
    existing_gru_data_view,
    insertion_errors,
    load_existing_pose_gru,
    load_insertion_model,
    precision_percentages,
    replay_insertion,
    score_insertion_methods,
)
from graph.pose_geometry import (
    geometric_endpoint,
    hole_frame_vector,
    rotate_vector,
    rotvec_quaternion,
)
from graph.temporal.insertion import (
    ReferenceMamba,
    StreamingInsertionEstimator,
    TemporalInsertionEstimator,
)
from graph.temporal.pose_gru import TemporalPoseGRU
from graph.temporal.pose_observations import (
    BASE_NAMES,
    INSERTION_CONTRACT,
    POSE_CONTRACT,
)
from main import run
from tests.test_temporal_hole_gru import write_smoke_fixture
from tests.test_temporal_pose_gru import pose_sequences
from training.gnn.temporal.config import resolve_insertion_config
from training.gnn.temporal.frozen_source import (
    infer_insertion_data,
    prepare_insertion_sequences,
)
from training.gnn.temporal.insertion import (
    check_matched_training_config,
    train,
    train_and_compare,
)
from training.gnn.temporal.pose_loss import INSERTION_ABLATIONS, insertion_loss


def insertion_settings(ablation="pose_relative_full"):
    return {
        "ablation": ablation,
        "position_unit_m": 0.01,
        "rotation_unit_rad": 0.1,
        "rotation_objective": "geodesic_huber",
        "weights": {
            name: 1.0
            for name in (
                "peg_position",
                "peg_rotation",
                "hole_position",
                "hole_rotation",
                "relative_position",
                "lateral_position",
                "relative_orientation",
            )
        },
    }


def model_settings(architecture="mamba"):
    return dict(
        architecture=architecture,
        d_model=16,
        d_state=4,
        d_conv=3,
        head_hidden=8,
        n_layers=2,
        with_robot=True,
    )


def workflow_fixture(root):
    """Use real checksummed graph serialization and observed raw proprioception."""
    config = write_smoke_fixture(root, pose_targets=True, raw_proprioception=True)
    manifest_path = Path(config["split_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["test_role"] = "gnn_validation_reference"
    manifest_path.write_text(json.dumps(manifest))
    config["data"]["raw_root"] = str(root / "raw")
    config["model"] = dict(d_model=16, d_state=4, d_conv=3, head_hidden=8, n_layers=1)
    config["training"].update(episode_batch_size=2, sequence_length=3)
    config["evaluation"].update(lag_max_frames=1, plot_episodes=1, benchmark_training=True)
    config["trainer"] = "training.gnn.temporal.insertion:train"
    return config


class TemporalMambaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.data = pose_sequences(True)
        self.dataset = InsertionEpisodeDataset(self.data)

    def test_reference_selective_scan_streaming_and_gradients(self):
        mixer = ReferenceMamba(d_model=8, d_state=3, d_conv=4)
        sequence = torch.randn(2, 7, 8, requires_grad=True)
        expected = mixer(sequence)
        convolution, selective = mixer.allocate_inference_cache(2)
        outputs = []
        for index in range(7):
            output, convolution, selective = mixer.step(
                sequence[:, index : index + 1], convolution, selective
            )
            outputs.append(output)
        torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-6, rtol=1e-5)
        expected[:, -1].sum().backward()
        self.assertGreater(float(sequence.grad[:, 0].abs().sum()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in mixer.parameters()))
        for name in ("A_log", "D", "x_proj.weight", "dt_proj.weight"):
            self.assertGreater(float(dict(mixer.named_parameters())[name].grad.abs().sum()), 0)
        # Changing the future cannot change the causal prefix.
        modified = sequence.detach().clone()
        modified[:, 4:] += 100
        torch.testing.assert_close(mixer(modified)[:, :4], expected.detach()[:, :4])

    def test_shared_features_tcp_coordinates_age_and_gt_isolation(self):
        observations, target = self.dataset[0]
        self.assertEqual(observations.shape, (5, 64))
        np.testing.assert_array_equal(
            observations[:, :3].numpy(),
            self.data["prediction"][:5, :3] + self.data["tcp_world"][:5],
        )
        torch.testing.assert_close(observations[0, 49:52], torch.zeros(3))
        torch.testing.assert_close(observations[1, 49:52], torch.tensor([0.01, 0.0, 0.0]))
        torch.testing.assert_close(
            observations[:, -2:],
            torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]]),
        )
        torch.testing.assert_close(target[:, 7:10], geometric_endpoint(target))
        changed = deepcopy(self.data)
        changed["target"][:] = 999
        torch.testing.assert_close(InsertionEpisodeDataset(changed)[0][0], observations)
        mean, std = fit_observation_normalization(self.dataset)
        altered = fit_observation_normalization(InsertionEpisodeDataset(changed))
        torch.testing.assert_close(mean, altered[0])
        torch.testing.assert_close(std, altered[1])
        self.assertEqual(InsertionEpisodeDataset(pose_sequences(False))[0][0].shape[-1], 55)

    def test_geometry_sign_hole_axes_and_quaternion_equivalence(self):
        quarter_turn = rotvec_quaternion(torch.tensor([[0.0, 0.0, torch.pi / 2]]))
        world_vector = torch.tensor([[0.0, 0.004, 0.002]])
        local = hole_frame_vector(world_vector, quarter_turn)
        torch.testing.assert_close(local, torch.tensor([[0.004, 0.0, 0.002]]), atol=1e-8, rtol=1e-5)
        torch.testing.assert_close(
            rotate_vector(quarter_turn, local), world_vector, atol=1e-8, rtol=1e-5
        )
        _, target = self.dataset[0]
        torch.testing.assert_close(geometric_endpoint(target), target[:, 7:10])
        shifted = target.clone()
        shifted[:, 10] += 0.01
        torch.testing.assert_close(
            geometric_endpoint(shifted), target[:, 7:10] - torch.tensor([0.01, 0, 0])
        )
        flipped = target.clone()
        flipped[:, 3:7] *= -1
        flipped[:, 13:17] *= -1
        mask = torch.ones(1, 5, dtype=torch.bool)
        loss, _ = insertion_loss(
            flipped[None], target[None], mask, "joint_peg_hole", insertion_settings()
        )
        self.assertLess(float(loss), 1e-9)
        errors = insertion_errors(flipped.numpy(), target.numpy())
        self.assertLess(errors["relative_rotation"].max(), 1e-4)

    def test_relative_loss_reaches_both_pose_heads_and_lateral_is_error(self):
        observations, target = self.dataset[0]
        model = TemporalInsertionEstimator(**model_settings())
        mask = torch.ones(1, 5, dtype=torch.bool)
        prediction = model(observations[None])
        settings = insertion_settings("pose_relative")
        settings["weights"] = {
            name: float(name == "relative_position") for name in settings["weights"]
        }
        loss, _ = insertion_loss(prediction, target[None], mask, model.mode, settings)
        loss.backward()
        for name in ("peg", "hole"):
            self.assertGreater(float(model.heads[name][-1].weight.grad.abs().sum()), 0)
        # Perfectly predicted but off-axis arrangements must have zero lateral loss.
        _, terms = insertion_loss(
            target[None], target[None], mask, model.mode, insertion_settings()
        )
        self.assertEqual(float(terms["lateral_position"]), 0.0)
        target[0, 0] += 10
        for ablation in INSERTION_ABLATIONS:
            loss, terms = insertion_loss(
                prediction.detach(), target[None], mask, model.mode, insertion_settings(ablation)
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(len(terms), 7)

    def test_padding_causal_reset_streaming_state38_and_all_modes(self):
        for architecture in ("gru", "mamba"):
            for mode in ("hole_only", "peg_only", "joint_peg_hole"):
                model = TemporalInsertionEstimator(mode=mode, **model_settings(architecture))
                predictions, _, difference = replay_insertion(model, self.data, torch.device("cpu"))
                self.assertLess(difference, 1e-5)
                if mode != "joint_peg_hole":
                    inactive = slice(0, 7) if mode == "hole_only" else slice(10, 17)
                    np.testing.assert_allclose(
                        predictions[:, inactive], self.data["prediction"][:, inactive], atol=1e-7
                    )
                first, second = self.dataset[0], self.dataset[1]
                short = (second[0][:2], second[1][:2])
                inputs, target, mask = collate_pose_episodes([first, short])
                outputs = model(inputs, mask)
                torch.testing.assert_close(
                    outputs[1, :2], model(short[0][None])[0], atol=1e-6, rtol=1e-5
                )
                self.assertTrue((outputs[~mask] == 0).all())
                target[~mask] = float("nan")
                loss, _ = insertion_loss(outputs, target, mask, mode, insertion_settings())
                self.assertTrue(torch.isfinite(loss))
                modified = inputs.clone()
                modified[:, 3:, :3] += 10
                torch.testing.assert_close(model(modified, mask)[:, :3], outputs[:, :3])
        model = TemporalInsertionEstimator(**model_settings())
        estimator = StreamingInsertionEstimator(model)
        from evaluation.gnn.temporal.pose_gru import stream_arguments

        observation = stream_arguments(self.data, 0, "cpu")
        observation.update(robot15=torch.randn(1, 15), freshness2=torch.randn(1, 2))
        estimator.reset("a")
        first = estimator.step(observation)
        estimator.step(stream_arguments(self.data, 1, "cpu"))
        estimator.reset("b")
        torch.testing.assert_close(estimator.step(observation)["object21"], first["object21"])
        estimator.reset("a")
        torch.testing.assert_close(estimator.step(observation)["object21"], first["object21"])
        self.assertEqual(set(estimator.episodes), {"a", "b"})
        torch.testing.assert_close(first["state38"][:, :15], observation["robot15"])
        torch.testing.assert_close(first["state38"][:, 15:36], first["object21"])
        torch.testing.assert_close(first["state38"][:, 36:], observation["freshness2"])
        with self.assertRaisesRegex(ValueError, "forbidden"):
            estimator.step(dict(observation, target=torch.zeros(1, 21)))
        with self.assertRaises(ValueError):
            estimator.step(observation, "unknown")

    def test_windows_never_cross_episodes_and_keep_actual_tcp_deltas(self):
        windows = InsertionWindowDataset(self.dataset, 3)
        self.assertEqual(len(windows), 4)
        torch.testing.assert_close(windows[1][0], self.dataset[0][0][3:])
        torch.testing.assert_close(windows[2][0], self.dataset[1][0][:3])
        self.assertGreater(float(windows[1][0][0, 49:52].abs().sum()), 0)
        for architecture in ("gru", "mamba"):
            model = TemporalInsertionEstimator(**model_settings(architecture))
            inputs, _ = self.dataset[0]
            for head in model.heads.values():
                torch.nn.init.zeros_(head[-1].weight)
                torch.nn.init.zeros_(head[-1].bias)
            output = model(inputs[None])[0]
            torch.testing.assert_close(output[:, :7], inputs[:, :7])
            torch.testing.assert_close(output[:, 10:17], inputs[:, 10:17])

    def test_parent_leakage_independent_and_reference_roles(self):
        source = {"train": ["a", "b"], "validation": ["v"], "test": ["t"]}
        manifest = {
            "test_role": "independent",
            "splits": {"train": ["a"], "validation": ["b"], "test": ["fresh"]},
            "parent_by_episode": {name: name for name in ("a", "b", "v", "t", "fresh")},
        }
        independent = {"train": ["fresh"], "validation": ["extra"], "test": []}
        manifest["parent_by_episode"]["extra"] = "extra"
        validate_insertion_splits(manifest, source, independent)
        for reserved in ("a", "b", "v", "t"):
            bad = deepcopy(manifest)
            bad["parent_by_episode"]["fresh"] = reserved
            with self.assertRaisesRegex(ValueError, "parent"):
                validate_insertion_splits(bad, source, independent)
        bad = deepcopy(manifest)
        bad["splits"]["train"] = ["v"]
        with self.assertRaises(ValueError):
            validate_insertion_splits(bad, source, independent)
        with self.assertRaises(ValueError):
            validate_insertion_splits(manifest, source)
        reference = deepcopy(manifest)
        reference["test_role"] = "gnn_validation_reference"
        reference["splits"]["test"] = ["v"]
        validate_insertion_splits(reference, source)

    def test_insertion_thresholds_p95_episode_pairing_and_motion(self):
        settings = {
            "visibility_labels": None,
            "phase_labels": None,
            "motion_threshold_m": 0.001,
            "motion_threshold_rad": 0.01,
            "long_occlusion_frames": 2,
            "lag_max_frames": 1,
        }
        target = self.data["target"].copy()
        perturbed = target.copy()
        perturbed[:, 7:10] += [0.0, 0.004, 0.0]
        summary, errors = score_insertion_methods(
            self.data, {"perfect": target, "off": perturbed}, settings
        )
        perfect = summary["methods"]["perfect"]["groups"]["all"]
        self.assertEqual(perfect["precision"]["lateral_lt_1mm_percent"], 100.0)
        self.assertEqual(perfect["peg_position"]["p95_mm"], 0.0)
        self.assertFalse(summary["physical_success_evaluated"])
        self.assertEqual(len(summary["paired_episode_mean_differences"]["perfect__minus__off"]), 2)
        self.assertTrue(
            all(row["lag_frames"] == 0 for row in summary["methods"]["perfect"]["peg_dynamics"])
        )
        thresholds = precision_percentages(
            {
                "lateral": np.array([0.5, 1.0, 2.0, 3.0]),
                "relative_3d": np.array([0.5, 1.0, 2.0, 3.0]),
            },
            np.ones(4, bool),
        )
        self.assertEqual(thresholds["lateral_lt_1mm_percent"], 25.0)
        self.assertEqual(thresholds["lateral_lt_3mm_percent"], 75.0)
        self.assertAlmostEqual(errors["off"]["lateral"].mean(), 4.0, places=4)

    def test_root_workflow_train_mamba_and_gru_then_paired_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = workflow_fixture(root)
            original = file_sha256(config["checkpoint"])
            config_path = root / "config.yaml"
            paths = {}
            environment = {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(root / "experiments")}
            with patch.dict("os.environ", environment):
                for architecture in ("mamba", "gru"):
                    config["model"]["architecture"] = architecture
                    config["output"]["root"] = str(root / architecture)
                    config_path.write_text(json.dumps(config))
                    run(config_path)
                    paths[architecture] = str(next((root / architecture).glob("*/best.pt")))
                config["model"]["architecture"] = "mamba"
                config.update(
                    trainer="evaluation.gnn.temporal.insertion:evaluate",
                    temporal_checkpoint=paths["mamba"],
                )
                config["comparison"] = {"checkpoints": {"gru_joint": paths["gru"]}}
                config["output"]["root"] = str(root / "evaluation")
                config_path.write_text(json.dumps(config))
                run(config_path)
            summary_path = next((root / "evaluation").glob("*/summary.json"))
            summary = json.loads(summary_path.read_text())
            self.assertFalse(summary["independent_test"])
            self.assertEqual(summary["split"], "validation")
            self.assertEqual(summary["frames"], 5)
            self.assertEqual(len(summary["methods"]), 6)
            self.assertTrue(
                all(summary["comparison_protocols"]["gru_joint"]["matches_primary"].values())
            )
            self.assertIn("learned_gate", summary["unavailable_comparisons"])
            self.assertGreater(summary["performance"]["primary"]["cache_bytes_batch_one"], 0)
            self.assertGreater(
                summary["performance"]["primary"]["training_throughput"]["frames_per_second"], 0
            )
            self.assertEqual(original, file_sha256(config["checkpoint"]))
            effective = resolve_insertion_config(config)
            _, _, identity = prepare_insertion_sequences(
                effective, torch.device("cpu"), ("validation",)
            )
            effective["feature_names"][0] = "wrong"
            with self.assertRaisesRegex(ValueError, "feature contract"):
                load_insertion_model(paths["mamba"], effective, identity, "cpu")
            history = json.loads(next((root / "mamba").glob("*/history.json")).read_text())
            self.assertEqual(len(history[0]["train"]), 8)

    def test_matched_gru_training_rejects_drift_and_evaluates_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = workflow_fixture(root)
            config["evaluation"]["benchmark_training"] = False
            with patch.dict(
                "os.environ",
                {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(root / "experiments")},
            ):
                reference_output = train(config)
                reference_path = reference_output / "best.pt"
                reference = torch.load(reference_path, map_location="cpu", weights_only=True)
                matched = deepcopy(config)
                matched["trainer"] = "training.gnn.temporal.insertion:train_and_compare"
                matched["model"].update(architecture="gru", backend="torch_reference")
                matched["comparison"] = {
                    "checkpoints": {"mamba": str(reference_path)},
                    "reference_sha256": file_sha256(reference_path),
                }
                matched["output"] = {
                    "root": str(root / "gru_training"),
                    "comparison_root": str(root / "comparison"),
                }
                resolved = resolve_insertion_config(matched)
                for field in ("feature_names", "loss", "training"):
                    changed = deepcopy(resolved)
                    if field == "feature_names":
                        changed[field] = changed[field][:-1]
                    elif field == "loss":
                        changed[field]["weights"]["lateral_position"] = 0.25
                    else:
                        changed[field]["sequence_length"] += 1
                    with self.assertRaisesRegex(ValueError, field):
                        check_matched_training_config(changed, reference)
                evaluation_output = train_and_compare(matched)
            summary = json.loads((evaluation_output / "summary.json").read_text())
            self.assertTrue(
                all(summary["comparison_protocols"]["mamba"]["matches_primary"].values())
            )
            self.assertFalse(summary["independent_test"])
            self.assertEqual(summary["frames"], 5)
            selected_path = next((root / "gru_training").glob("*/best.pt"))
            selected = torch.load(selected_path, map_location="cpu", weights_only=True)
            self.assertEqual(
                selected["config"]["normalization"], reference["config"]["normalization"]
            )
            self.assertEqual(
                selected["config"]["feature_names"], reference["config"]["feature_names"]
            )
            self.assertEqual(selected["config"]["model"]["architecture"], "gru")
            self.assertEqual(
                selected["config"]["matched_training_reference"]["sha256"],
                file_sha256(reference_path),
            )

    def test_existing_gru_comparison_keeps_same_frames_and_declares_training_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_dir = root / "reference"
            reference_dir.mkdir()
            frozen_path = root / "frozen.pt"
            frozen_path.write_bytes(b"unchanged frozen source")
            manifest = {
                "splits": {"train": ["train"], "validation": ["a", "b"], "test": ["test"]},
                "parent_by_episode": {name: name for name in ("train", "a", "b", "test")},
            }
            settings = {
                "visibility_labels": None,
                "phase_labels": None,
                "motion_threshold_m": 0.001,
                "motion_threshold_rad": 0.01,
                "long_occlusion_frames": 2,
                "lag_max_frames": 1,
            }
            model = TemporalInsertionEstimator(**model_settings())
            mamba_prediction, _, difference = replay_insertion(
                model, self.data, torch.device("cpu")
            )
            reference = {
                "checkpoint": str(frozen_path),
                "source_checkpoint_sha256": file_sha256(frozen_path),
                "temporal_split_manifest": manifest,
                "feature_names": list(self.dataset.feature_names),
                "observation_contract": INSERTION_CONTRACT,
                "with_robot": True,
                "evaluation": settings,
                "seed": 12,
                "mode": "joint_peg_hole",
                "model": model_settings(),
                "loss": insertion_settings(),
                "training": {"context_policy": "reset_windows", "sequence_length": 3},
            }
            primary_path = root / "mamba.pt"
            torch.save(
                {
                    "kind": "temporal_insertion",
                    "config": reference,
                    "model": model.state_dict(),
                    "dataset_sha256": "same_dataset",
                    "epoch": 1,
                },
                primary_path,
            )
            reference["temporal_checkpoint"] = str(primary_path)
            predictions = {
                "gnn": self.data["prediction"],
                "learned_gate": self.data["prediction"],
                "primary": mamba_prediction,
            }
            summary, _ = score_insertion_methods(self.data, predictions, settings)
            summary.update(split="validation", streaming_max_difference={"primary": difference})
            (reference_dir / "summary.json").write_text(json.dumps(summary))
            (reference_dir / "effective_config.json").write_text(json.dumps(reference))
            (reference_dir / "provenance.json").write_text(
                json.dumps({"checkpoints": {"primary": {"sha256": file_sha256(primary_path)}}})
            )
            np.savez_compressed(
                reference_dir / "predictions.npz",
                **self.data,
                **{name + "_object21": value for name, value in predictions.items()},
            )
            gru_settings = {
                "hidden": 16,
                "projection": 8,
                "head_hidden": 8,
                "layers": 1,
                "position_mode": "direct",
                "position_scale_m": 0.1,
            }
            gru = TemporalPoseGRU(**gru_settings)
            trained = dict(
                reference,
                feature_names=list(BASE_NAMES),
                observation_contract=POSE_CONTRACT,
                with_robot=False,
                gru=gru_settings,
                normalization={"source_split": "train", "mean": [0.0] * 53, "std": [1.0] * 53},
                training={"epochs": 20},
                sequence_contract="complete_episode_zero_hidden_right_padding_v1",
            )
            trained["loss"] = dict(insertion_settings(), rotation_objective="pose_surrogate")
            gru_path = root / "gru.pt"
            payload = {
                "kind": "temporal_pose_gru",
                "config": trained,
                "model": gru.state_dict(),
                "dataset_sha256": "same_dataset",
                "epoch": 19,
                "selection_score": 2.0,
            }
            torch.save(payload, gru_path)
            original = file_sha256(gru_path)
            view = existing_gru_data_view(self.data, False)
            self.assertNotIn("tcp_quaternion", view)
            self.assertIn("tcp_quaternion", self.data)
            np.testing.assert_array_equal(view["sample_ids"], self.data["sample_ids"])
            with patch.dict(
                "os.environ",
                {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(root / "experiments")},
            ):
                output = compare_existing_gru(
                    {
                        "reference_evaluation": str(reference_dir),
                        "gru_checkpoint": str(gru_path),
                        "device": "cpu",
                        "output": {"root": str(root / "comparison")},
                    }
                )
            measured = json.loads((output / "summary.json").read_text())
            self.assertEqual(measured["frames"], 10)
            self.assertEqual(set(measured["methods"]), {"gnn", "learned_gate", "gru", "mamba"})
            self.assertFalse(measured["comparison_audit"]["controlled_backbone_comparison"])
            self.assertFalse(measured["comparison_audit"]["training_features_equal"])
            self.assertEqual(measured["comparison_audit"]["gru"]["feature_count"], 53)
            self.assertEqual(original, file_sha256(gru_path))
            self.assertEqual(
                measured["methods"]["mamba"]["groups"]["all"],
                summary["methods"]["primary"]["groups"]["all"],
            )
            np.testing.assert_array_equal(
                np.load(output / "predictions.npz")["sample_ids"], self.data["sample_ids"]
            )
            payload["config"]["normalization"]["source_split"] = "validation"
            torch.save(payload, gru_path)
            with self.assertRaisesRegex(ValueError, "normalization"):
                load_existing_pose_gru(
                    gru_path, reference, {"dataset_sha256": "same_dataset"}, "cpu"
                )
            payload["config"]["normalization"]["source_split"] = "train"
            payload["config"]["temporal_split_manifest"] = deepcopy(manifest)
            payload["config"]["temporal_split_manifest"]["parent_by_episode"]["a"] = "wrong_parent"
            torch.save(payload, gru_path)
            with self.assertRaisesRegex(ValueError, "parent provenance"):
                load_existing_pose_gru(
                    gru_path, reference, {"dataset_sha256": "same_dataset"}, "cpu"
                )

    def test_independent_cache_end_to_end_and_no_test_reads_during_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            external_root = root / "external"
            source_root.mkdir()
            external_root.mkdir()
            config = workflow_fixture(source_root)
            external_config = workflow_fixture(external_root)
            # Distinct source IDs/parents, while preserving the same preprocessing.
            from training.gnn.cached_data import build_data
            from training.gnn.data import (
                TrainingBatch,
                canonical_hash,
                save_cached_sample,
            )

            external_cache = Path(external_config["data"]["cache_dir"])
            source_config = torch.load(external_config["checkpoint"], weights_only=True)["config"]
            bundle = build_data(source_config, torch.Generator())
            materialized = []
            for loader in (bundle.train, bundle.validation):
                for index in range(len(loader.dataset)):
                    materialized.append(loader.dataset[index])
            metadata_path = external_cache / "metadata/data_identity.json"
            inventory_path = external_cache / "metadata/cache_inventory.json"
            metadata = json.loads(metadata_path.read_text())
            namespace = canonical_hash(metadata["preprocessing"])
            metadata["splits"] = {
                split: ["fresh_" + ep for ep in ids] for split, ids in metadata["splits"].items()
            }
            metadata["frames"] = []
            inventory = {"preprocessing_sha256": namespace, "files": []}
            for batch in materialized:
                sample_id = "fresh_" + batch.sample_ids[0]
                raw_old = external_root / "raw" / batch.sample_ids[0]
                raw_new = external_root / "raw" / sample_id
                raw_new.parent.mkdir(exist_ok=True)
                raw_new.write_bytes(raw_old.read_bytes())
                digest = file_sha256(raw_new)
                identity = {
                    "sample_id": sample_id,
                    "frame_sha256": digest,
                    "preprocessing_sha256": namespace,
                }
                relative = f"{namespace}/{canonical_hash(identity)}.npz"
                save_cached_sample(
                    external_cache / relative,
                    TrainingBatch(batch.graph, batch.targets, [sample_id]),
                    identity,
                )
                metadata["frames"].append({"id": sample_id, "sha256": digest})
                inventory["files"].append(
                    {
                        "sample_id": sample_id,
                        "file": relative,
                        "sha256": file_sha256(external_cache / relative),
                        "size": (external_cache / relative).stat().st_size,
                    }
                )
            metadata.pop("dataset_sha256")
            counts = metadata.pop("sample_counts")
            metadata["dataset_sha256"] = canonical_hash(metadata)
            metadata["sample_counts"] = counts
            metadata_path.write_text(json.dumps(metadata))
            inventory_path.write_text(json.dumps(inventory))
            manifest_path = Path(config["split_manifest"])
            manifest = json.loads(manifest_path.read_text())
            manifest["test_role"] = "independent"
            manifest["splits"]["test"] = ["fresh_v"]
            manifest["parent_by_episode"].update({f"fresh_{ep}": f"fresh_{ep}" for ep in "abvt"})
            manifest_path.write_text(json.dumps(manifest))
            config["data"].update(
                independent_cache_dir=str(external_cache),
                independent_raw_root=str(external_root / "raw"),
            )
            effective = resolve_insertion_config(config)
            # Training should need only independent metadata, not its inventory/frames.
            inventory_bytes = inventory_path.read_bytes()
            inventory_path.unlink()
            prepare_insertion_sequences(effective, torch.device("cpu"), ("train", "validation"))
            inventory_path.write_bytes(inventory_bytes)
            frozen, loaders, identity = prepare_insertion_sequences(
                effective, torch.device("cpu"), ("test",)
            )
            data = infer_insertion_data(frozen, loaders["test"], "cpu", identity, effective, "test")
            self.assertEqual(set(data["episode"]), {"fresh_v"})
            self.assertIn("independent_dataset_sha256", identity)
            self.assertEqual(InsertionEpisodeDataset(data)[0][0].shape[-1], 64)

    @unittest.skipUnless(importlib.util.find_spec("mani_skill"), "requires installed ManiSkill")
    def test_live_cpu_episode_geometry_and_rgbd_streaming(self):
        """Three actual frames verify plumbing, not trained-checkpoint accuracy."""
        from dataclasses import asdict

        from envs.maniskill import get_gt_state, make_peg_insertion_env
        from graph.build_graph import GraphBuilder
        from graph.gated_mpnn import VertexEdgeGatedMPNN
        from graph.temporal.observations import observation_features
        from graph.temporal.pose_observations import (
            observation_age_step,
            pose_input_step,
        )
        from graph.utils.object21_targets import object21_from_gt
        from visual.corner_detection import detect_tensors

        frozen = (
            VertexEdgeGatedMPNN(num_semantic_ids=1, hidden=16, layers=1)
            .eval()
            .requires_grad_(False)
        )
        model = TemporalInsertionEstimator(**model_settings()).eval()
        stream = StreamingInsertionEstimator(model)
        stream.reset("actual_cpu_episode")
        env = make_peg_insertion_env(
            device="cpu",
            obs_mode="rgb+depth",
            sensor_configs={"base_camera": {"width": 512, "height": 512}},
        )
        sequence, streamed = [], []
        previous_tcp = previous_quaternion = age = None
        try:
            observation, _ = env.reset(seed=12)
            for _ in range(3):
                gt = get_gt_state(env)
                target = object21_from_gt(asdict(gt))
                task = env.unwrapped
                expected_relative = task.peg_head_pose.p - task.box_hole_pose.p
                torch.testing.assert_close(
                    geometric_endpoint(target), expected_relative, atol=1e-7, rtol=1e-5
                )
                expected_local = (task.box_hole_pose.inv() * task.peg_head_pose).p
                torch.testing.assert_close(
                    hole_frame_vector(target[:, 7:10], target[:, 13:17]),
                    expected_local,
                    atol=1e-7,
                    rtol=1e-5,
                )
                camera = observation["sensor_data"]["base_camera"]
                calibration = observation["sensor_param"]["base_camera"]
                corners = detect_tensors(
                    camera["rgb"][0],
                    camera["depth"][0],
                    calibration["intrinsic_cv"][0],
                    calibration["extrinsic_cv"][0],
                    device="cpu",
                )
                tcp_pose = observation["extra"]["tcp_pose"]
                tcp, quaternion = tcp_pose[:, :3], tcp_pose[:, 3:]
                gripper = observation["agent"]["qpos"][:, -2:]
                graph = GraphBuilder()([corners], tcp)
                with torch.inference_mode():
                    prediction = frozen(graph)["object21"]
                quality = observation_features(graph, include_peg=True)
                inputs = pose_input_step(
                    prediction, quality, tcp, previous_tcp, quaternion, previous_quaternion, gripper
                )
                age = observation_age_step(quality, age)
                sequence.append(torch.cat((inputs, age), dim=-1))
                result = stream.step(
                    dict(
                        object21=prediction,
                        quality25=quality,
                        tcp_world=tcp,
                        tcp_quaternion=quaternion,
                        gripper_qpos=gripper,
                    )
                )
                streamed.append(result["world_object21"])
                previous_tcp, previous_quaternion = tcp.clone(), quaternion.clone()
                action = torch.zeros(env.action_space.shape)
                observation, _, _, _, _ = env.step(action)
            with torch.inference_mode():
                expected = model.forward_sequence(torch.stack(sequence, dim=1))
            torch.testing.assert_close(torch.stack(streamed, dim=1), expected, atol=1e-6, rtol=1e-5)
        finally:
            env.close()

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("mamba_ssm"),
        "requires installed official mamba_ssm and CUDA",
    )
    def test_official_kernel_reference_forward_gradient_and_streaming_parity(self):
        from mamba_ssm import Mamba

        reference = ReferenceMamba(16, d_state=4, d_conv=4).cuda()
        official = Mamba(16, d_state=4, d_conv=4, use_fast_path=True).cuda()
        official.load_state_dict(reference.state_dict(), strict=True)
        inputs = torch.randn(2, 7, 16, device="cuda", requires_grad=True)
        copied = inputs.detach().clone().requires_grad_(True)
        actual = official(inputs)
        expected = reference(copied)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(inputs.grad, copied.grad, atol=2e-5, rtol=1e-4)
        model = TemporalInsertionEstimator(backend="mamba_ssm", **model_settings()).cuda()
        replay_insertion(model, self.data, torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
