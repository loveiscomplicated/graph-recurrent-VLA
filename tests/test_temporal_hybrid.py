"""Hybrid contracts, causal branch independence and explicit identity/reset checks."""

import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from data_collection.temporal.hybrid_features import replay_hybrid
from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset
from data_collection.temporal.pose_sequences import stream_arguments
from evaluation.gnn.temporal.hybrid_diagnostics import analyze_hybrid
from evaluation.gnn.temporal.hybrid_inputs import verify_model_sources
from evaluation.gnn.temporal.insertion import insertion_errors, replay_insertion
from evaluation.gnn.temporal_gate import replay_gate
from graph.pose_geometry import geometric_endpoint, rotvec_quaternion
from graph.temporal.hybrid import (
    StreamingHybridEstimator,
    assemble_hybrid,
    world_object21,
)
from graph.temporal.insertion import TemporalInsertionEstimator
from graph.temporal_gate import LearnedTemporalGate
from tests.test_temporal_pose_gru import pose_sequences


class HybridTemporalTests(unittest.TestCase):
    def test_source_relocation_preserves_hash_checks_and_rejects_calculation_changes(self):
        paths = {
            "graph/temporal_mamba.py": "graph/temporal/insertion.py",
            "graph/temporal_gate.py": "graph/temporal_gate.py",
            "graph/pose_geometry.py": "graph/pose_geometry.py",
            "graph/gated_mpnn.py": "graph/gated_mpnn.py",
            "graph/temporal_pose_observations.py": "graph/temporal/pose_observations.py",
        }
        original = b"from graph.temporal_pose import MODES\nposition_scale_m = 0.01\n"
        relocated = b"from graph.temporal.pose_gru import MODES\nposition_scale_m = 0.01\n"
        original_hash = hashlib.sha256(original).hexdigest()
        current_hash = hashlib.sha256(relocated).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            for current_path in paths.values():
                path = repository / current_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relocated)
            historical = {name: original_hash for name in paths}
            audit = verify_model_sources(historical, repository=repository)
            self.assertTrue(
                all(
                    details["verification"] == "original_bytes_after_import_relocation"
                    for details in audit.values()
                )
            )
            current = {name: current_hash for name in paths.values()}
            audit = verify_model_sources(current, repository=repository)
            self.assertTrue(
                all(details["verification"] == "exact_bytes" for details in audit.values())
            )
            changed_model = repository / "graph/temporal/insertion.py"
            changed_model.write_bytes(relocated.replace(b"0.01", b"0.02"))
            for hashes in (historical, current, {}):
                with (
                    self.subTest(reference=hashes),
                    self.assertRaisesRegex(ValueError, "Model source differs"),
                ):
                    verify_model_sources(hashes, repository=repository)

    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.data = pose_sequences(True)
        self.mamba = TemporalInsertionEstimator(
            with_robot=True, d_model=16, d_state=4, head_hidden=8
        )
        self.gate = LearnedTemporalGate()

    def observation(self, row):
        result = stream_arguments(self.data, row, torch.device("cpu"))
        result["quality17"] = torch.tensor(self.data["features"][row : row + 1])
        return result

    def test_indices_normalization_sizes_and_both_endpoint_methods(self):
        frozen = torch.tensor(self.data["prediction"])
        mamba, gate = frozen.clone(), frozen.clone()
        mamba[:, :3] += 0.003
        mamba[:, 3:7] = rotvec_quaternion(torch.tensor([[0.0, 0.0, torch.pi / 2]])).expand(
            len(frozen), 4
        )
        mamba[:, 13:17] = rotvec_quaternion(torch.tensor([[0.1, 0.2, 0.0]])).expand(len(frozen), 4)
        mamba[:, 7:10] = geometric_endpoint(mamba)
        gate[:, 10:13] += torch.tensor([0.01, -0.002, 0.005])
        a = assemble_hybrid(mamba, gate, frozen)
        b = assemble_hybrid(mamba, gate, frozen, endpoint_method="preserve_endpoint")
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(a[:, :7], mamba[:, :7])
        torch.testing.assert_close(a[:, 10:13], gate[:, 10:13])
        torch.testing.assert_close(a[:, 13:17], mamba[:, 13:17])
        torch.testing.assert_close(a[:, 17:21], frozen[:, 17:21])
        torch.testing.assert_close(a[:, 7:10], geometric_endpoint(a))
        torch.testing.assert_close(a[:, 7:10], mamba[:, 7:10] + mamba[:, 10:13] - gate[:, 10:13])
        h2 = assemble_hybrid(mamba, gate, frozen, hole_rotation="gnn")
        torch.testing.assert_close(h2[:, 13:17], frozen[:, 13:17])
        torch.testing.assert_close(h2[:, 7:10], a[:, 7:10])
        scaled = mamba.clone()
        scaled[:, 3:7] *= -3
        scaled[:, 13:17] *= 2
        torch.testing.assert_close(assemble_hybrid(scaled, gate, frozen), a)
        # Explicitly distinguish endpoint preservation from pose reconstruction.
        mamba[:, 7] += 0.01
        preserved = assemble_hybrid(mamba, gate, frozen, endpoint_method="preserve_endpoint")
        torch.testing.assert_close(
            preserved[:, 7:10] + gate[:, 10:13], mamba[:, 7:10] + mamba[:, 10:13]
        )
        self.assertGreater(float((preserved - a).abs().max()), 0.009)

    def test_tcp_world_translation_and_vector_sign(self):
        frozen = torch.tensor(self.data["prediction"])
        tcp = torch.tensor(self.data["tcp_world"])
        gate = frozen.clone()
        gate[:, 10] += 0.01
        relative = assemble_hybrid(frozen, gate, frozen)
        world = assemble_hybrid(
            world_object21(frozen, tcp), world_object21(gate, tcp), world_object21(frozen, tcp)
        )
        torch.testing.assert_close(world, world_object21(relative, tcp))
        torch.testing.assert_close(
            relative[:, 7:10], geometric_endpoint(frozen) - torch.tensor([0.01, 0.0, 0.0])
        )
        torch.testing.assert_close(world[:, 7:10], relative[:, 7:10])

    def test_episode_reset_interleaving_and_state38(self):
        estimator = StreamingHybridEstimator(self.mamba, self.gate)
        estimator.reset("first")
        observation = self.observation(0)
        observation.update(robot15=torch.zeros(1, 15), freshness2=torch.ones(1, 2))
        first = estimator.step(observation, episode_id="first", frame_index=0)
        self.assertEqual(tuple(first["state38"].shape), (1, 38))
        torch.testing.assert_close(first["state38"][:, 15:36], first["object21"])
        self.assertEqual(float(first["gate_alpha"]), 1.0)
        estimator.reset("other")
        estimator.step(self.observation(2), episode_id="other", frame_index=0)
        estimator.step(self.observation(1), episode_id="first", frame_index=1)
        estimator.reset("first")
        repeated = estimator.step(observation, episode_id="first", frame_index=0)
        torch.testing.assert_close(first["object21"], repeated["object21"])
        self.assertEqual(float(repeated["gate_alpha"]), 1.0)
        estimator.discard("first")
        self.assertNotIn("first", estimator.mamba.episodes)
        self.assertNotIn("first", estimator.episodes)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.mamba.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.mamba.parameters()))

    def test_frame_alignment_unknown_gt_and_quality_mismatch_rejected(self):
        estimator = StreamingHybridEstimator(self.mamba, self.gate)
        estimator.reset("episode")
        with self.assertRaisesRegex(ValueError, "next frame"):
            estimator.step(self.observation(0), episode_id="episode", frame_index=1)
        observation = self.observation(0)
        observation["target"] = torch.zeros(1, 21)
        with self.assertRaisesRegex(ValueError, "GT/labels"):
            estimator.step(observation, episode_id="episode", frame_index=0)
        observation = self.observation(0)
        observation["quality17"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "same BOX/HOLE"):
            estimator.step(observation, episode_id="episode", frame_index=0)
        shifted = deepcopy(self.data)
        shifted["sample_ids"] = np.roll(shifted["sample_ids"], 1)
        with self.assertRaises(ValueError):
            replay_hybrid(self.mamba, self.gate, shifted, torch.device("cpu"))
        shifted = deepcopy(self.data)
        shifted["frame"][1] += 1
        with self.assertRaises(ValueError):
            InsertionEpisodeDataset(shifted)

    def test_streaming_reproduces_standalone_without_gt_or_future_leakage(self):
        device = torch.device("cpu")
        combined, _, _ = replay_hybrid(self.mamba, self.gate, self.data, device)
        solo, _, _ = replay_insertion(self.mamba, self.data, device)
        gate = replay_gate(self.gate, self.data, device)["object21"]
        np.testing.assert_allclose(combined["mamba"], solo, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(combined["learned_gate"], gate, atol=1e-6, rtol=1e-5)
        altered = deepcopy(self.data)
        altered["target"][:] = 123
        without_gt, _, _ = replay_hybrid(self.mamba, self.gate, altered, device)
        for name in combined:
            np.testing.assert_array_equal(combined[name], without_gt[name])
        altered = deepcopy(self.data)
        altered["prediction"][3:, :3] += 20
        changed, _, _ = replay_hybrid(self.mamba, self.gate, altered, device)
        np.testing.assert_array_equal(combined["hybrid_h1_a"][:3], changed["hybrid_h1_a"][:3])

    def test_cancellation_identity_and_cluster_bootstrap(self):
        predictions, _, _ = replay_hybrid(self.mamba, self.gate, self.data, torch.device("cpu"))
        errors = {
            name: insertion_errors(value, self.data["target"])
            for name, value in predictions.items()
        }
        manifest = {
            "parent_by_episode": {
                str(episode): "shared_parent" for episode in np.unique(self.data["episode"])
            }
        }
        diagnostics = analyze_hybrid(self.data, predictions, errors, manifest, resamples=100)
        self.assertLess(
            diagnostics["error_cancellation"]["squared_identity_max_residual_mm2"], 1e-8
        )
        bootstrap = diagnostics["comparisons"]["hybrid_h1_a"]["relative_3d"]["paired_bootstrap"]
        self.assertEqual(bootstrap["parents"], 1)
        self.assertEqual(
            bootstrap["episode_equal_percentile_ci_mm"][0],
            bootstrap["episode_equal_percentile_ci_mm"][1],
        )


if __name__ == "__main__":
    unittest.main()
