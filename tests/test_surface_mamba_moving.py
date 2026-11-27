"""Contracts for exogenous motion and paired policy comparisons."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data_collection.surface_mamba_motion import make_motion_plan
from data_collection.episode_dataset import file_sha256
from evaluation.surface_mamba_moving import (
    _verify_candidate_checkpoints,
    paired_comparison,
)
from graph.surface_context_frontend import (
    EAGER_CUDA_VALIDATION_ENV,
    _detector_main_options,
)
from main import load_config
from recurrent.run_surface_mamba_moving import (
    _child_environment,
    _prepare_collection_protocol,
)
from training.surface_mamba_moving import (
    _evaluation_seeds,
    _validate_warm_start_mixing,
)


MOTION = {
    "start_step": 10,
    "end_step": 40,
    "distance_m": 0.06,
    "seed_salt": 20261120,
}


class MovingSurfaceTests(unittest.TestCase):
    def test_graph_only_single_configs_preserve_fixed_hole_lineage(self):
        repository = Path(__file__).resolve().parents[1]
        config_root = repository / "recurrent" / "configs" / "surface_mamba"
        collection = load_config(
            config_root / "moving_hole_graph_only_single_collect.yaml"
        )
        training = load_config(
            config_root / "moving_hole_graph_only_single_train.yaml"
        )

        self.assertEqual(
            collection["trainer"],
            "recurrent.run_surface_mamba_moving:collect_dataset",
        )
        self.assertEqual(collection["dataset_counts"], [160, 40])
        self.assertEqual(training["mixing"], {"alphas": [1.0]})
        self.assertEqual(training["seeds"], [0])
        self.assertNotIn("allow_warm_start_alpha_change", training["mixing"])
        self.assertEqual(
            training["initial_checkpoint_sha256"],
            "48d07bbc9c86cc4476cd3e32c5b13aa4a6270af80fe924b804cf783dfc64e4ec",
        )
        self.assertIn(
            "/training_v2/graph_only/seed_0/",
            training["initial_checkpoint"],
        )
        self.assertEqual(training["dataset_contract"]["counts"], [160, 40])
        self.assertEqual(
            training["dataset_contract"]["motion"], collection["motion"]
        )

        evaluations = [
            load_config(
                config_root
                / f"moving_hole_graph_only_single_evaluate_shard_{index}.yaml"
            )
            for index in range(2)
        ]
        self.assertEqual(
            [(config["seed_start"], config["seed_limit"]) for config in evaluations],
            [(0, 150), (150, 150)],
        )
        for evaluation in evaluations:
            self.assertEqual(evaluation["motion"], collection["motion"])
            self.assertEqual(
                evaluation["comparison_reference"], "fixed_graph_only_single"
            )
            self.assertEqual(
                set(evaluation["candidates"]),
                {
                    "fixed_graph_only_single",
                    "moving_adapted_graph_only_single",
                },
            )
            self.assertEqual(
                evaluation["checkpoint_sha256"]["fixed_graph_only_single"],
                [
                    "48d07bbc9c86cc4476cd3e32c5b13aa4a6270af80fe924b804cf783dfc64e4ec"
                ],
            )
            self.assertEqual(
                evaluation["checkpoint_sha256"][
                    "moving_adapted_graph_only_single"
                ],
                [
                    "5e35d7e68c2dbf9fa4cf98377d9f39aeaabba659d1ebfd3979cab31da10714e4"
                ],
            )

    def test_collection_protocol_excludes_fixed_and_evaluation_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seeds_path = root / "evaluation_seeds.json"
            evaluation_seeds = list(range(300))
            seeds_path.write_text(json.dumps({"seeds": evaluation_seeds}))
            tap_checkpoint = root / "tap.pt"
            tap_checkpoint.write_bytes(b"tap")

            fixed_dataset = root / "fixed_dataset"
            fixed_dataset.mkdir()
            for name, seeds in {
                "train": [1000, 1001],
                "validation": [2000],
                "test": [3000],
            }.items():
                np.savez(fixed_dataset / f"{name}.npz", reset_seed=seeds)

            fixed_config = root / "fixed.json"
            fixed_config.write_text(
                json.dumps(
                    {
                        "frontend": {
                            "tapnextpp": {"checkpoint": str(tap_checkpoint)}
                        },
                        "environment": {"device": "cpu"},
                        "horizon": 250,
                        "seeds_file": str(seeds_path),
                        "controller_device": "cpu",
                    }
                )
            )
            config = {
                "fixed_evaluation_config": str(fixed_config),
                "fixed_dataset": str(fixed_dataset),
                "tap_checkpoint_sha256": file_sha256(tap_checkpoint),
                "motion": MOTION,
            }
            expected_seed_sha256 = file_sha256(seeds_path)

            protocol, excluded = _prepare_collection_protocol(config)

        self.assertEqual(protocol["motion"], MOTION)
        self.assertEqual(protocol["seeds_sha256"], expected_seed_sha256)
        self.assertEqual(
            excluded,
            evaluation_seeds + [1000, 1001, 2000, 3000],
        )

    def test_named_moving_candidates_require_exact_checkpoint_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.pt"
            second = root / "second.pt"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            config = {
                "candidates": {
                    "first": {"checkpoints": [str(first)]},
                    "second": {"checkpoints": [str(second)]},
                },
                "checkpoint_sha256": {
                    "first": [file_sha256(first)],
                    "second": [file_sha256(second)],
                },
            }
            self.assertEqual(
                _verify_candidate_checkpoints(config),
                config["checkpoint_sha256"],
            )
            config["checkpoint_sha256"]["second"] = ["0" * 64]
            with self.assertRaisesRegex(ValueError, "digests differ"):
                _verify_candidate_checkpoints(config)

    def test_warm_start_alpha_change_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "alpha changes require"):
            _validate_warm_start_mixing(
                [0.0, 0.25, 0.5, 0.75, 1.0], {"alphas": [1.0]}
            )

        selected = _validate_warm_start_mixing(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            {"alphas": [1.0], "allow_warm_start_alpha_change": True},
        )
        self.assertEqual(selected, [1.0])

    def test_file_backed_evaluation_seeds_are_hash_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation_seeds.json"
            path.write_text(json.dumps({"seeds": [10, 20, 30]}))
            contract = {
                "evaluation_seeds_file": str(path),
                "evaluation_seeds_sha256": file_sha256(path),
            }
            self.assertEqual(_evaluation_seeds(contract), [10, 20, 30])
            path.write_text(json.dumps({"seeds": [10, 20]}))
            with self.assertRaisesRegex(ValueError, "digest changed"):
                _evaluation_seeds(contract)

    def test_collection_eager_validation_is_explicit_and_scoped(self):
        config = {"device": "cuda", "detector": {"optimized": True}}

        with patch.dict("os.environ", {EAGER_CUDA_VALIDATION_ENV: "1"}):
            options = _detector_main_options(config)
            collection_environment = _child_environment(
                1, {EAGER_CUDA_VALIDATION_ENV: "1"}
            )
            evaluation_environment = _child_environment(0)

        self.assertFalse(options["compile_cuda_validation"])
        self.assertEqual(collection_environment["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(collection_environment[EAGER_CUDA_VALIDATION_ENV], "1")
        self.assertNotIn(EAGER_CUDA_VALIDATION_ENV, evaluation_environment)

    def test_detector_config_takes_precedence_over_recovery_environment(self):
        config = {
            "device": "cuda",
            "detector": {"compile_cuda_validation": True},
        }
        with patch.dict("os.environ", {EAGER_CUDA_VALIDATION_ENV: "1"}):
            options = _detector_main_options(config)

        self.assertTrue(options["compile_cuda_validation"])

    def test_invalid_eager_validation_environment_is_rejected(self):
        with patch.dict("os.environ", {EAGER_CUDA_VALIDATION_ENV: "yes"}):
            with self.assertRaisesRegex(ValueError, "must be '0' or '1'"):
                _detector_main_options({"device": "cuda"})

    def test_motion_is_seeded_and_matches_endpoints(self):
        first = make_motion_plan(4, MOTION, 250)
        second = make_motion_plan(4, MOTION, 250)
        np.testing.assert_array_equal(first.offsets, second.offsets)
        np.testing.assert_array_equal(first.offsets[:11], 0)
        np.testing.assert_allclose(first.offsets[25], first.offsets[-1] / 2)
        np.testing.assert_array_equal(first.offsets[40], first.offsets[-1])
        self.assertAlmostEqual(np.linalg.norm(first.offsets[-1]), 0.06)
        self.assertNotEqual(first.sha256, make_motion_plan(5, MOTION, 250).sha256)

    def test_comparison_verifies_motion_and_counts_paired_gains(self):
        metadata = {
            "trajectory_sha256": "plan",
            "control_frequency_hz": 20.0,
            "motion_completed": True,
        }
        baseline = {seed: dict(metadata, success=seed == 0) for seed in range(4)}
        adapted = {seed: dict(metadata, success=seed in (0, 1, 2)) for seed in range(4)}
        comparison = paired_comparison(baseline, adapted, MOTION, 250)
        self.assertEqual(comparison["delta_percentage_points"], 50.0)
        self.assertEqual(comparison["adapted_only_success"], 2)
        self.assertEqual(comparison["baseline_only_success"], 0)
        adapted[1]["trajectory_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Motion or control frequency differs"):
            paired_comparison(baseline, adapted, MOTION, 250)
