"""Experiment-matrix contracts for Surface-Mamba training and evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.surface_mamba_controller import (
    _aggregate_predictions,
    _bootstrap_episode_metrics,
    _candidate_specs,
    _episode_metric_rows,
    _load_eval_seeds,
    _paired_bootstrap_differences,
)
from main import load_config
from recurrent.surface_mamba import ACTION_DIM, OBSERVATION_DIM, MambaController
from training.surface_mamba_controller import _validate_config, train
from training.surface_mamba_parallel import _build_jobs, _resolve_gpu_ids


def training_config(*, seeds, alphas):
    """Return the smallest valid controller-training config for contract tests."""
    return {
        "seeds": seeds,
        "mixing": {"alphas": alphas},
        "training": {
            "max_steps": 10,
            "batch_size": 2,
            "validation_interval": 5,
            "checkpoint_interval": 5,
            "gradient_clip_norm": 1.0,
            "optimizer": {
                "type": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
            },
            "scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "patience_validations": 2,
                "min_learning_rate": 1e-5,
            },
            "early_stopping": {
                "patience_validations": 3,
                "min_delta": 1e-4,
            },
        },
    }


class TrainingConfigTests(unittest.TestCase):
    def test_accepts_five_seed_mixture_and_single_graph_only(self):
        _validate_config(
            training_config(
                seeds=[0, 1, 2, 3, 4],
                alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
            )
        )
        _validate_config(training_config(seeds=[0], alphas=[1.0]))

    def test_rejects_duplicate_seeds_or_missing_graph_validation_level(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            _validate_config(training_config(seeds=[0, 0], alphas=[1.0]))
        with self.assertRaisesRegex(ValueError, "include 1.0"):
            _validate_config(training_config(seeds=[0], alphas=[0.0, 0.5]))

    def test_rejects_unsorted_or_out_of_range_alphas(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _validate_config(training_config(seeds=[0], alphas=[1.0, 0.5]))
        with self.assertRaisesRegex(ValueError, "between 0.0 and 1.0"):
            _validate_config(training_config(seeds=[0], alphas=[1.25]))

    def test_rejects_invalid_optimizer_scheduler_and_stopping_settings(self):
        config = training_config(seeds=[0], alphas=[1.0])
        config["training"]["optimizer"]["type"] = "sgd"
        with self.assertRaisesRegex(ValueError, "optimizer.type"):
            _validate_config(config)

        config = training_config(seeds=[0], alphas=[1.0])
        config["training"]["scheduler"]["factor"] = 1.0
        with self.assertRaisesRegex(ValueError, "scheduler.factor"):
            _validate_config(config)

        config = training_config(seeds=[0], alphas=[1.0])
        config["training"]["early_stopping"]["patience_validations"] = 0
        with self.assertRaisesRegex(ValueError, "patience_validations"):
            _validate_config(config)


class GraphOnlyTrainingIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_split(path: Path, episode_count: int, seed_start: int) -> None:
        graph = np.zeros((episode_count, 36), dtype=np.float32)
        graph[:, 3] = 1
        graph[:, 18] = 1
        graph[:, 28] = 1
        ground_truth = graph.copy()
        ground_truth[:, 0] = 1000
        action = np.zeros((episode_count, 7), dtype=np.float32)
        action[:, 6] = 1
        np.savez(
            path,
            state36_gt=ground_truth,
            state36_graph=graph,
            action_target=action,
            episode_offsets=np.arange(episode_count + 1, dtype=np.int64),
            reset_seed=np.arange(seed_start, seed_start + episode_count, dtype=np.int64),
        )

    @staticmethod
    def _config(dataset: Path, output: Path, *, max_steps: int) -> dict:
        return {
            "trainer": "training.surface_mamba_controller:train",
            "dataset": str(dataset),
            "output": str(output),
            "resume": True,
            "device": "cpu",
            "cpu_threads": 1,
            "seeds": [0],
            "mixing": {"alphas": [1.0]},
            "model": {
                "d_model": 8,
                "d_state": 2,
                "d_conv": 2,
                "expand": 1,
                "layers": 1,
            },
            "training": {
                "max_steps": max_steps,
                "batch_size": 8,
                "gradient_clip_norm": 1.0,
                "validation_interval": 1,
                "checkpoint_interval": 2,
                "optimizer": {
                    "type": "adamw",
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                },
                "scheduler": {
                    "type": "reduce_on_plateau",
                    "factor": 0.5,
                    "patience_validations": 1,
                    "min_learning_rate": 1e-5,
                },
                "early_stopping": {
                    "patience_validations": max(max_steps + 1, 2),
                    "min_delta": 1e-6,
                },
            },
        }

    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        dataset.mkdir()
        (dataset / "dataset_manifest.json").write_text(
            json.dumps({"format": "surface_mamba_state36_v1"})
        )
        self._write_split(dataset / "train.npz", 720, 0)
        self._write_split(dataset / "validation.npz", 90, 1000)
        return dataset

    @staticmethod
    def _warm_start(config: dict) -> dict[int, dict]:
        """Build a valid source checkpoint payload without a preparatory run."""
        model_config = {
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            **config["model"],
        }
        torch.manual_seed(91)
        model = MambaController(**model_config)
        statistics = {
            "obs_mean": np.zeros(OBSERVATION_DIM, dtype=np.float32).tolist(),
            "obs_std": np.ones(OBSERVATION_DIM, dtype=np.float32).tolist(),
            "obs_keep": np.ones(OBSERVATION_DIM, dtype=np.float32).tolist(),
            "act_mean": np.zeros(6, dtype=np.float32).tolist(),
            "act_std": np.ones(6, dtype=np.float32).tolist(),
        }
        payload = {
            "seed": 0,
            "model_config": model_config,
            "model": {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            },
            "statistics": statistics,
            "identity": {"alphas": [1.0]},
        }
        return {0: {"payload": payload, "sha256": "warm-start-test"}}

    def test_warm_start_step_zero_remains_best_when_updates_regress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            output = root / "training"
            config = self._config(dataset, output, max_steps=2)
            config["training"]["early_stopping"]["patience_validations"] = 2
            warm_starts = self._warm_start(config)
            source_model = warm_starts[0]["payload"]["model"]

            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False), patch(
                "training.surface_mamba_controller._validation_loss",
                side_effect=[0.5, 1.0, 1.0],
            ):
                train(config, warm_starts=warm_starts)

            run = output / "mamba_seed_0"
            summary = json.loads((run / "summary.json").read_text())
            history = json.loads((run / "history.json").read_text())["history"]
            best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)

        self.assertEqual(summary["best_step"], 0)
        self.assertEqual(summary["best_graph_loss"], 0.5)
        self.assertEqual(summary["stop_reason"], "early_stopping")
        self.assertEqual(best["step"], 0)
        self.assertEqual(last["step"], 2)
        self.assertTrue(history[0]["initial_checkpoint"])
        self.assertIsNone(history[0]["training_loss"])
        for name, expected in source_model.items():
            torch.testing.assert_close(best["model"][name], expected, rtol=0, atol=0)

    def test_warm_start_step_zero_is_replaced_by_a_real_improvement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            output = root / "training"
            config = self._config(dataset, output, max_steps=2)
            warm_starts = self._warm_start(config)

            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False), patch(
                "training.surface_mamba_controller._validation_loss",
                side_effect=[1.0, 0.5, 0.6],
            ):
                train(config, warm_starts=warm_starts)

            run = output / "mamba_seed_0"
            summary = json.loads((run / "summary.json").read_text())
            best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)

        self.assertEqual(summary["best_step"], 1)
        self.assertEqual(summary["best_graph_loss"], 0.5)
        self.assertEqual(best["step"], 1)

    def test_one_step_graph_only_training_never_reports_gt_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            output = root / "training"
            config = self._config(dataset, output, max_steps=1)

            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False):
                train(config)

            history = json.loads(
                (output / "mamba_seed_0" / "history.json").read_text()
            )["history"]
            checkpoint = output / "mamba_seed_0" / "best.pt"
            checkpoint_exists = checkpoint.is_file()

        self.assertEqual(len(history), 1)
        self.assertIsNone(history[0]["validation_gt_loss"])
        self.assertEqual(list(history[0]["validation_by_alpha"]), ["1.0"])
        self.assertTrue(checkpoint_exists)

    def test_plateau_scheduler_early_stopping_and_periodic_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            output = root / "training"
            config = self._config(dataset, output, max_steps=10)
            config["training"]["early_stopping"]["patience_validations"] = 2

            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False), patch(
                "training.surface_mamba_controller._validation_loss",
                return_value=1.0,
            ):
                train(config)

            run = output / "mamba_seed_0"
            summary = json.loads((run / "summary.json").read_text())
            checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)

            self.assertEqual(summary["stop_reason"], "early_stopping")
            self.assertEqual(summary["final_step"], 3)
            self.assertLess(summary["final_learning_rate"], 1e-3)
            self.assertTrue((run / "checkpoints" / "step_000002.pt").is_file())
            self.assertTrue((run / "best.pt").is_file())
            self.assertTrue((run / "segment_best.pt").is_file())
            self.assertIn("scheduler", checkpoint)
            self.assertIn("rng", checkpoint)

    def test_interrupted_resume_matches_uninterrupted_model_and_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self._dataset(root)
            full_config = self._config(dataset, root / "full", max_steps=4)
            resumed_config = self._config(dataset, root / "resumed", max_steps=4)

            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False):
                train(full_config)

            from training import surface_mamba_controller as trainer

            original_save = trainer._save_checkpoint

            def interrupt_after_step_two(path, payload):
                original_save(path, payload)
                if path.name == "last.pt" and payload["step"] == 2:
                    raise RuntimeError("simulated interruption")

            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False), patch(
                "training.surface_mamba_controller._save_checkpoint",
                side_effect=interrupt_after_step_two,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train(resumed_config)

            with patch.dict(os.environ, {"WANDB_MODE": "disabled"}, clear=False):
                train(resumed_config)

            full = torch.load(
                root / "full" / "mamba_seed_0" / "last.pt",
                map_location="cpu",
                weights_only=False,
            )
            resumed = torch.load(
                root / "resumed" / "mamba_seed_0" / "last.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(resumed["step"], 4)
            self.assertEqual(resumed["scheduler"], full["scheduler"])
            self.assertEqual(resumed["optimizer"]["param_groups"], full["optimizer"]["param_groups"])
            for name, expected in full["model"].items():
                torch.testing.assert_close(resumed["model"][name], expected, rtol=0, atol=0)
            for parameter, expected_state in full["optimizer"]["state"].items():
                actual_state = resumed["optimizer"]["state"][parameter]
                for name, expected in expected_state.items():
                    if torch.is_tensor(expected):
                        torch.testing.assert_close(actual_state[name], expected, rtol=0, atol=0)
                    else:
                        self.assertEqual(actual_state[name], expected)


class EvaluationCandidateTests(unittest.TestCase):
    def test_closed_loop_seed_slices_partition_frozen_seed_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation_seeds.json"
            path.write_text(json.dumps({"seeds": list(range(300))}))

            self.assertEqual(_load_eval_seeds(path, 30), list(range(30)))
            first_shard = _load_eval_seeds(path, 150, seed_start=0)
            second_shard = _load_eval_seeds(path, 150, seed_start=150)
            self.assertEqual(first_shard + second_shard, list(range(300)))
            self.assertEqual(len(_load_eval_seeds(path)), 300)
            with self.assertRaisesRegex(ValueError, "number of seeds remaining"):
                _load_eval_seeds(path, 0)
            with self.assertRaisesRegex(ValueError, "between 0 and 299"):
                _load_eval_seeds(path, 1, seed_start=300)
            with self.assertRaisesRegex(ValueError, "number of seeds remaining"):
                _load_eval_seeds(path, 151, seed_start=150)

    def test_candidate_specs_accept_single_and_five_checkpoint_models(self):
        specs = _candidate_specs(
            {
                "candidates": {
                    "mixed_single": {
                        "input": "graph",
                        "checkpoints": ["mixed/seed_0.pt"],
                    },
                    "mixed_ensemble": {
                        "input": "graph",
                        "checkpoints": [f"mixed/seed_{seed}.pt" for seed in range(5)],
                    },
                    "graph_only_single": {
                        "input": "graph",
                        "checkpoints": ["graph_only/seed_0.pt"],
                    },
                }
            }
        )

        self.assertEqual(list(specs), ["mixed_single", "mixed_ensemble", "graph_only_single"])
        self.assertEqual(len(specs["mixed_single"]["checkpoints"]), 1)
        self.assertEqual(len(specs["mixed_ensemble"]["checkpoints"]), 5)

    def test_candidate_specs_reject_unsupported_counts_and_inputs(self):
        with self.assertRaisesRegex(ValueError, "one or five"):
            _candidate_specs(
                {
                    "candidates": {
                        "bad": {"input": "graph", "checkpoints": ["a.pt", "b.pt"]}
                    }
                }
            )
        with self.assertRaisesRegex(ValueError, "graph.*gt"):
            _candidate_specs(
                {
                    "candidates": {
                        "bad": {"input": "camera", "checkpoints": ["a.pt"]}
                    }
                }
            )

    def test_prediction_aggregation_uses_mean_and_majority_vote(self):
        predictions = []
        for index in range(5):
            prediction = np.zeros((3, 7), dtype=np.float32)
            prediction[:, :6] = index
            prediction[:, 6] = 1 if index < 3 else -1
            predictions.append(prediction)

        ensemble = _aggregate_predictions(predictions)

        np.testing.assert_allclose(ensemble[:, :6], 2.0)
        np.testing.assert_array_equal(ensemble[:, 6], 1.0)
        np.testing.assert_array_equal(_aggregate_predictions(predictions[:1]), predictions[0])

    def test_episode_bootstrap_and_paired_differences_are_deterministic(self):
        target = np.zeros((4, 7), dtype=np.float32)
        target[:, 6] = 1
        reference_prediction = target.copy()
        candidate_prediction = target.copy()
        candidate_prediction[:, 0] = 0.5
        candidate_prediction[:, 6] = -1
        offsets = np.asarray([0, 2, 4], dtype=np.int64)
        seeds = np.asarray([10, 11], dtype=np.int64)
        reference_rows = _episode_metric_rows(
            "reference", reference_prediction, target, offsets, seeds
        )
        candidate_rows = _episode_metric_rows(
            "candidate", candidate_prediction, target, offsets, seeds
        )

        first = _bootstrap_episode_metrics(candidate_rows, samples=100, seed=7)
        second = _bootstrap_episode_metrics(candidate_rows, samples=100, seed=7)
        paired = _paired_bootstrap_differences(
            {"reference": reference_rows, "candidate": candidate_rows},
            reference="reference",
            samples=100,
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["gripper_accuracy"]["episode_mean"], 0.0)
        self.assertGreater(
            paired["candidate"]["motion_mse"]["mean_difference"], 0.0
        )
        self.assertEqual(
            paired["candidate"]["gripper_accuracy"]["mean_difference"], -1.0
        )


class ParallelLauncherTests(unittest.TestCase):
    def test_generated_json_preserves_scientific_notation_as_float(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "job.json"
            path.write_text('{"min_learning_rate": 1e-05}')

            config = load_config(path)

        self.assertEqual(config["min_learning_rate"], 1e-5)
        self.assertIsInstance(config["min_learning_rate"], float)

    def test_builds_six_independent_jobs_without_retraining_mixed_single(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "output": str(root / "launcher"),
                "base_training": {
                    "trainer": "training.surface_mamba_controller:train",
                    "dataset": "/data/state36",
                    "resume": True,
                    "device": "cuda:0",
                    "cpu_threads": 2,
                    "model": {"d_model": 96},
                    "training": {
                        "max_steps": 20000,
                        "batch_size": 8,
                        "gradient_clip_norm": 1.0,
                        "validation_interval": 100,
                        "checkpoint_interval": 200,
                        "optimizer": {
                            "type": "adamw",
                            "learning_rate": 1e-3,
                            "weight_decay": 0.0,
                        },
                        "scheduler": {
                            "type": "reduce_on_plateau",
                            "factor": 0.5,
                            "patience_validations": 5,
                            "min_learning_rate": 1e-5,
                        },
                        "early_stopping": {
                            "patience_validations": 15,
                            "min_delta": 1e-4,
                        },
                    },
                },
                "experiments": [
                    {
                        "name": "mixed",
                        "output": str(root / "mixed"),
                        "seeds": [0, 1, 2, 3, 4],
                        "mixing": {"alphas": [0.0, 0.25, 0.5, 0.75, 1.0]},
                    },
                    {
                        "name": "graph_only",
                        "output": str(root / "graph_only"),
                        "seeds": [0],
                        "mixing": {"alphas": [1.0]},
                    },
                ],
            }

            jobs = _build_jobs(config)

        self.assertEqual(
            [job.name for job in jobs],
            [
                "mixed_seed_0",
                "mixed_seed_1",
                "mixed_seed_2",
                "mixed_seed_3",
                "mixed_seed_4",
                "graph_only_seed_0",
            ],
        )
        self.assertTrue(all(job.config["device"] == "cuda:0" for job in jobs))
        self.assertTrue(all(len(job.config["seeds"]) == 1 for job in jobs))
        self.assertEqual(jobs[-1].config["mixing"]["alphas"], [1.0])
        self.assertEqual(
            jobs[0].checkpoint,
            Path(config["experiments"][0]["output"])
            / "seed_0"
            / "mamba_seed_0"
            / "best.pt",
        )

    def test_warm_start_sources_are_verified_and_added_to_each_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "source.pt"
            checkpoint.write_bytes(b"source checkpoint")
            config = {
                "output": str(root / "launcher"),
                "warm_start_sources": [
                    {
                        "seed": 0,
                        "checkpoint": str(checkpoint),
                        "sha256": file_sha256(checkpoint),
                    }
                ],
                "base_training": {
                    "trainer": "training.surface_mamba_moving:fine_tune",
                    "dataset": "/data/moving_state36",
                    "resume": True,
                    "device": "cuda:0",
                    "cpu_threads": 2,
                    "model": {"d_model": 8},
                    "training": training_config(seeds=[0], alphas=[1.0])["training"],
                },
                "experiments": [
                    {
                        "name": "graph_only",
                        "output": str(root / "graph_only"),
                        "seeds": [0],
                        "mixing": {
                            "alphas": [1.0],
                            "allow_warm_start_alpha_change": True,
                        },
                    }
                ],
            }

            jobs = _build_jobs(config)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            jobs[0].config["trainer"],
            "training.surface_mamba_moving:fine_tune",
        )
        self.assertEqual(jobs[0].config["initial_checkpoint"], str(checkpoint))
        self.assertEqual(
            jobs[0].config["initial_checkpoint_sha256"],
            config["warm_start_sources"][0]["sha256"],
        )

    def test_gpu_auto_respects_parent_cuda_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}, clear=False):
            self.assertEqual(_resolve_gpu_ids("auto"), ["2", "5"])
        self.assertEqual(_resolve_gpu_ids([3, "GPU-abcd"]), ["3", "GPU-abcd"])


if __name__ == "__main__":
    unittest.main()
