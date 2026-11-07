"""Peg-only optimization, frozen causal parity, geometry, and parent isolation."""

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.finetuning_split import make_finetuning_split
from data_collection.temporal.hybrid_features import (
    cache_frozen_features,
    predict_cached,
    replay_hybrid,
)
from evaluation.gnn.temporal.hybrid_finetuning import selection_score
from evaluation.gnn.temporal.hybrid_finetuning_analysis import (
    analyze,
    approach_proxy,
    tracking_summary,
)
from evaluation.gnn.temporal.insertion import score_insertion_methods
from graph.pose_geometry import geometric_endpoint, hole_frame_vector, peg_axis
from graph.temporal.hybrid import StreamingHybridEstimator
from graph.temporal.hybrid_finetuning import (
    HybridPegFineTuner,
    parameter_changes,
    snapshot_state,
)
from graph.temporal.insertion import TemporalInsertionEstimator
from graph.temporal_gate import LearnedTemporalGate
from tests.test_temporal_pose_gru import pose_sequences
from training.gnn.temporal.hybrid_finetuning import (
    calibrate_weights,
    fit_ablation,
    validate_config,
)
from training.gnn.temporal.hybrid_finetuning_loss import geometry_terms, gradient_vector
from utils.experiment_logger import ExperimentLogger


class HybridFineTuningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        torch.set_num_threads(1)
        self.device = torch.device("cpu")
        self.data = pose_sequences(True)
        self.model = TemporalInsertionEstimator(d_model=16, d_state=4, head_hidden=8)
        self.gate = LearnedTemporalGate()
        self.original = snapshot_state(self.model)
        self.cache, self.h1 = cache_frozen_features(self.model, self.gate, self.data, self.device)
        self.adapter = HybridPegFineTuner(self.model, self.gate)
        self.config = yaml.safe_load(
            Path("graph/configs/temporal/hybrid/finetune_peg.yaml").read_text()
        )
        self.config["evaluation"] = {
            "phase_labels": None,
            "visibility_labels": None,
            "motion_threshold_m": 0.001,
            "motion_threshold_rad": 0.01,
            "long_occlusion_frames": 2,
            "lag_max_frames": 1,
            "alignment_axial_min_m": -0.05,
            "alignment_axial_max_m": 0.02,
            "alignment_lateral_m": 0.01,
        }
        self.config["training"].update(epochs=2, batch_size=4, calibration_frames=8)

    def prediction(self):
        return self.adapter(
            *(
                self.cache[name]
                for name in ("representation", "frozen_world", "mamba_world", "gate_world")
            )
        )

    def test_initialization_causal_cache_and_no_target_leakage(self):
        prediction = predict_cached(self.adapter, self.cache, self.data, self.device, 4)
        np.testing.assert_allclose(prediction, self.h1, atol=2e-6, rtol=1e-5)
        altered = deepcopy(self.data)
        altered["target"][:] = 123
        changed, unchanged = cache_frozen_features(self.model, self.gate, altered, self.device)
        torch.testing.assert_close(
            changed["representation"], self.cache["representation"], atol=0, rtol=0
        )
        np.testing.assert_array_equal(unchanged, self.h1)
        future = deepcopy(self.data)
        future["prediction"][3:5, :3] += 1
        _, changed_output = cache_frozen_features(self.model, self.gate, future, self.device)
        np.testing.assert_array_equal(changed_output[:3], self.h1[:3])
        # Each episode's frame 0 comes from a cold reset, including repeated replay.
        np.testing.assert_array_equal(
            unchanged[self.data["frame"] == 0], self.h1[self.data["frame"] == 0]
        )

    def test_relative_gradient_reaches_only_peg_and_size_is_predicted(self):
        prediction = self.prediction()
        target = self.cache["target"].clone()
        target[:, 17:21] *= 3
        terms = geometry_terms(prediction, target, self.config["loss"])
        parameters = list(self.model.heads["peg"].parameters())
        gradient = gradient_vector(terms["relative_position"], parameters)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)
        output_gradient = torch.autograd.grad(
            terms["relative_position"], self.model.heads["peg"][2].bias, retain_graph=True
        )[0]
        self.assertGreater(float(output_gradient[:3].norm()), 0)
        self.assertGreater(float(output_gradient[3:].norm()), 0)
        self.assertGreater(float(gradient_vector(terms["lateral_position"], parameters).norm()), 0)
        terms["relative_position"].backward()
        for name, parameter in self.adapter.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("mamba.heads.peg."))
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad)
        torch.testing.assert_close(prediction[:, 17:21], self.cache["frozen_world"][:, 17:21])
        torch.testing.assert_close(prediction[:, 7:10], geometric_endpoint(prediction))
        torch.testing.assert_close(prediction[:, 3:7].norm(dim=-1), torch.ones(len(prediction)))
        opposite = target.clone()
        opposite[:, 3:7] *= -1
        sign_terms = geometry_terms(prediction, opposite, self.config["loss"])
        torch.testing.assert_close(sign_terms["peg_rotation"], terms["peg_rotation"])
        original_terms = geometry_terms(prediction, self.cache["target"], self.config["loss"])
        torch.testing.assert_close(terms["relative_position"], original_terms["relative_position"])

    def test_lateral_axis_decomposition_matches_scorer(self):
        target = self.cache["target"]
        error = geometric_endpoint(self.prediction()) - target[:, 7:10]
        axis = peg_axis(target[:, 13:17])
        axial = (error * axis).sum(-1)
        lateral = error - axial[:, None] * axis
        transformed = hole_frame_vector(error, target[:, 13:17])
        torch.testing.assert_close(lateral.norm(dim=-1), transformed[:, 1:].norm(dim=-1))
        torch.testing.assert_close(
            error.square().sum(-1), axial.square() + lateral.square().sum(-1)
        )

    def test_split_uses_only_original_train_parents(self):
        original = {
            "splits": {"train": ["a", "a_sibling", "b", "c"], "validation": ["v"], "test": ["t"]},
            "parent_by_episode": {
                "a": "p",
                "a_sibling": "p",
                "b": "q",
                "c": "r",
                "v": "s",
                "t": "u",
            },
        }
        result = make_finetuning_split(original, seed=0, development_fraction=0.5)
        train, development = result["splits"]["train"], result["splits"]["development"]
        self.assertEqual(set(train + development), set(original["splits"]["train"]))
        parents = original["parent_by_episode"]
        self.assertFalse({parents[e] for e in train} & {parents[e] for e in development})
        invalid = deepcopy(original)
        invalid["parent_by_episode"]["v"] = "p"
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_finetuning_split(invalid, seed=0, development_fraction=0.5)

    def test_post_selection_approach_and_tracking_keep_episode_boundaries(self):
        data = deepcopy(self.data)
        data["target"][:, 7:10] = 0.0
        data["target"][:, 7] = np.tile([-0.06, -0.04, -0.03, -0.02, 0.01], 2)
        data["target"][:, 13:17] = [1.0, 0.0, 0.0, 0.0]
        mask = approach_proxy(
            data,
            {
                "alignment_axial_min_m": -0.05,
                "alignment_axial_max_m": 0.02,
                "alignment_lateral_m": 0.01,
            },
        )
        self.assertEqual(int(mask.sum()), 8)
        self.assertFalse(mask[self.data["frame"] == 0].any())
        baseline = {
            "peg_dynamics": [
                {"episode": "a", "lag_frames": 2, "rotation_lag_frames": 0},
                {"episode": "b", "lag_frames": None, "rotation_lag_frames": None},
            ]
        }
        candidate = deepcopy(baseline)
        candidate["peg_dynamics"][0]["lag_frames"] = 3
        result = tracking_summary(candidate, baseline)
        self.assertEqual(result["lag_frames"]["paired_episodes"], 1)
        self.assertEqual(result["lag_frames"]["greater_delay_episodes"], 1)

    def test_all_ablations_optimization_frozen_checks_and_streaming_state38(self):
        baseline = score_insertion_methods(
            self.data, {"original_h1": self.h1}, self.config["evaluation"]
        )[0]
        weights, calibration = calibrate_weights(self.adapter, self.cache, self.config, self.device)
        self.assertGreater(calibration["initial_gradients"]["pose_gradient_norm"], 0)
        gate_before = snapshot_state(self.gate)
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {"WANDB_MODE": "disabled"}),
        ):
            output = Path(temporary)
            predictions = {"original_h1": self.h1}
            with ExperimentLogger(
                team="graph", root=output / "experiments", config=self.config
            ) as experiment:
                for name in ("pose_only", "pose_relative", "pose_relative_lateral"):
                    current = dict(weights)
                    if name == "pose_only":
                        current["relative_position"] = 0.0
                    if name != "pose_relative_lateral":
                        current["lateral_position"] = 0.0
                    result = fit_ablation(
                        self.adapter,
                        {"train": self.cache, "development": self.cache},
                        {"development": self.data},
                        baseline["methods"]["original_h1"],
                        self.original,
                        self.config,
                        name,
                        current,
                        output,
                        experiment,
                        self.device,
                    )
                    self.assertTrue(Path(result["best_checkpoint"]).is_file())
                    last = torch.load(output / name / "last.pt", weights_only=True)
                    self.assertTrue(
                        any(
                            not torch.equal(value, last["peg_head"][key])
                            for key, value in snapshot_state(self.model.heads["peg"]).items()
                        )
                        or result["best_epoch"] > 0
                    )
                    cached = predict_cached(self.adapter, self.cache, self.data, self.device, 4)
                    streamed, _, _ = replay_hybrid(self.model, self.gate, self.data, self.device)
                    predictions[name] = streamed["hybrid_h1_a"]
                    np.testing.assert_allclose(
                        cached, streamed["hybrid_h1_a"], atol=2e-6, rtol=1e-5
                    )
                    np.testing.assert_array_equal(
                        streamed["hybrid_h1_a"][:, 10:17], self.h1[:, 10:17]
                    )
                    parameter_changes(self.original, self.model)
            for key, value in gate_before.items():
                torch.testing.assert_close(value, self.gate.state_dict()[key], atol=0, rtol=0)
            summary, errors = score_insertion_methods(
                self.data, predictions, self.config["evaluation"]
            )
            np.savez_compressed(output / "reference.npz", **self.data)
            self.config["reference_predictions_sha256"] = file_sha256(output / "reference.npz")
            metadata = {
                "summary.json": summary,
                "effective_config.json": self.config,
                "execution_status.json": {"status": "completed_ablations_no_independent_test"},
                "finetuning_split.json": {
                    "parent_by_episode": {
                        str(episode): str(episode) for episode in np.unique(self.data["episode"])
                    }
                },
            }
            for name, value in metadata.items():
                (output / name).write_text(json.dumps(value))
            np.savez_compressed(
                output / "frame_errors.npz",
                sample_ids=self.data["sample_ids"],
                **{
                    name + "__" + metric: value
                    for name, metrics in errors.items()
                    for metric, value in metrics.items()
                },
            )
            with patch.dict(os.environ, {"EXPERIMENTS_DIR": str(output / "analysis_experiments")}):
                report = analyze(
                    {
                        "source_run": str(output),
                        "reference_predictions": str(output / "reference.npz"),
                        "seed": 0,
                        "bootstrap_resamples": 100,
                        "output": {"root": str(output / "analysis")},
                    }
                )
            self.assertTrue((report / "comparison.json").is_file())
            self.assertGreater((report / "precision_cdf.png").stat().st_size, 10000)
        from evaluation.gnn.temporal.pose_gru import stream_arguments

        observation = stream_arguments(self.data, 0, self.device)
        observation.update(
            quality17=torch.from_numpy(self.data["features"][:1]),
            robot15=torch.zeros(1, 15),
            freshness2=torch.ones(1, 2),
        )
        estimator = StreamingHybridEstimator(self.model, self.gate)
        estimator.reset("new")
        result = estimator.step(observation, episode_id="new", frame_index=0)
        self.assertEqual(tuple(result["state38"].shape), (1, 38))
        torch.testing.assert_close(result["state38"][:, 15:36], result["object21"])

    def test_frozen_change_and_selection_guard_rejected(self):
        with torch.no_grad():
            self.model.input_mean.add_(1)
        with self.assertRaisesRegex(AssertionError, "Frozen Mamba"):
            parameter_changes(self.original, self.model)
        baseline = score_insertion_methods(self.data, {"h1": self.h1}, self.config["evaluation"])[
            0
        ]["methods"]["h1"]
        candidate = deepcopy(baseline)
        candidate["groups"]["all"]["peg_position"]["mean_mm"] += 10
        self.assertIn(
            "peg_position", selection_score(candidate, baseline, self.config["selection"])[1]
        )
        invalid = deepcopy(self.config)
        invalid["training"]["lr"] = float("nan")
        with self.assertRaises(ValueError):
            validate_config(invalid)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable on this host")
    def test_cpu_streaming_cache_cuda_head_training_and_return_to_cpu(self):
        self.adapter.to("cuda")
        self.adapter.enable_peg_head()
        arguments = [
            self.cache[name].to("cuda")
            for name in ("representation", "frozen_world", "mamba_world", "gate_world")
        ]
        prediction = self.adapter(*arguments)
        terms = geometry_terms(prediction, self.cache["target"].to("cuda"), self.config["loss"])
        optimizer = torch.optim.AdamW(self.model.heads["peg"].parameters(), lr=3e-5)
        sum(terms.values()).backward()
        optimizer.step()
        self.adapter.to("cpu")
        changes = parameter_changes(self.original, self.model)
        self.assertTrue(any(item["changed"] for item in changes.values()))
        cached = predict_cached(self.adapter, self.cache, self.data, self.device, 4)
        streamed, _, _ = replay_hybrid(self.model, self.gate, self.data, self.device)
        np.testing.assert_allclose(cached, streamed["hybrid_h1_a"], atol=2e-6, rtol=1e-5)
        np.testing.assert_array_equal(streamed["hybrid_h1_a"][:, 10:17], self.h1[:, 10:17])


if __name__ == "__main__":
    unittest.main()
