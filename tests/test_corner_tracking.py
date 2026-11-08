"""Causality, lifecycle, masking and paired workflow contracts on CPU fixtures."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data_collection.temporal.corner_tracks import (
    confirmed_eligibility, load_dataset, mask_plan, masked_frame, prepare, save_dataset,
)
from data_collection.temporal.corner_tracking_synthetic import synthetic_dataset
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.corner_tracking.model import (
    ConstantVelocityMotion, CornerMotionModel, observed_rotation_change, rotation_change,
)
from graph.temporal.corner_tracking.tracker import CornerTracker, Observation, TrackingSettings
from main import load_config, run
from tests.fixtures.corner_tracking import calibration, observation, write_corner_family_source
from training.gnn.temporal.corner_tracking import (
    build_models, compare, resolve_config, shared_initialization_digest,
)
from training.gnn.temporal.corner_tracking_rollout import replay_episode


class CornerLifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tracker = CornerTracker(ConstantVelocityMotion(), calibration())

    def test_confirmation_anchor_missing_dormancy_and_retirement(self):
        first = self.tracker.step(observation([[0, 0, 0]]))
        self.assertEqual(first.statuses[0], "tentative")
        self.tracker.step(observation([[0, 0, 0]]))
        memory_length = self.tracker.tracks[0].memory["length"]
        for age in range(1, 34):
            result = self.tracker.step(observation())
            if age <= 8:
                self.assertEqual(result.statuses[0], "missing")
                self.assertEqual(result.adopted_predictions, [0])
            elif age < 33:
                self.assertEqual(result.statuses[0], "dormant")
                self.assertEqual(result.adopted_predictions, [])
            else:
                self.assertEqual(result.retired, [0])
                self.assertNotIn(0, self.tracker.tracks)
            if age < 33:
                self.assertEqual(self.tracker.tracks[0].anchor_frame, 1)
                np.testing.assert_array_equal(self.tracker.tracks[0].anchor, [0, 0, 0])
                self.assertEqual(self.tracker.tracks[0].memory["length"], memory_length + age)

    def test_tentative_expiration_and_no_id_reuse(self):
        self.tracker.step(observation([[0, 0, 0]]))
        self.tracker.step(observation())
        self.tracker.step(observation())
        result = self.tracker.step(observation([[0, 0, 0]]))
        self.assertEqual(result.retired, [0])
        self.assertEqual(result.births, {1: 0})

    def test_detection_order_does_not_define_identity(self):
        points = [[0, 0, 0], [0.05, 0, 0]]
        self.tracker.step(observation(points))
        result = self.tracker.step(observation(points, [1, 0]))
        self.assertEqual(result.assignments, {0: 1, 1: 0})

    def test_ambiguous_detection_does_not_create_duplicate_or_change_anchor(self):
        points = [[-0.005, 0, 0], [0.005, 0, 0]]
        self.tracker.step(observation(points))
        self.tracker.step(observation(points))
        result = self.tracker.step(observation([[0, 0, 0]]))
        self.assertFalse(result.assignments)
        self.assertFalse(result.births)
        self.assertEqual(result.ambiguous_detections, {0})
        self.assertFalse(result.adopted_predictions)
        self.assertEqual(int(result.graph.geometry.node_valid.sum()), 1)
        self.assertEqual(self.tracker.tracks[0].anchor_frame, 1)

    def test_outside_gate_creates_new_candidate_instead_of_forced_match(self):
        self.tracker.step(observation([[0, 0, 0]]))
        result = self.tracker.step(observation([[1, 0, 0]]))
        self.assertFalse(result.assignments)
        self.assertEqual(result.births, {1: 0})

    def test_one_frame_fallback_preserves_identity_when_prediction_misses_gate(self):
        self.tracker.step(observation([[0, 0, 0]]))
        track = self.tracker.tracks[0]
        self.tracker.frame = 1
        predicted = np.array([[0.1, 0, 0]], dtype=np.float32)
        detection = observation([[0.001, 0, 0]])
        positions = self.tracker._association_positions([track], predicted, detection)
        np.testing.assert_array_equal(positions[0], track.anchor)
        np.testing.assert_array_equal(predicted[0], np.array([0.1, 0, 0], dtype=np.float32))
        self.assertEqual(self.tracker._associate([track], positions, detection)[0], {0: 0})

        # A valid motion prediction remains preferred; the fallback never
        # reconnects a track across a gap using its stale observed position.
        valid_prediction = np.array([[0.01, 0, 0]], dtype=np.float32)
        np.testing.assert_array_equal(
            self.tracker._association_positions([track], valid_prediction, detection), valid_prediction
        )
        self.tracker.frame = 2
        np.testing.assert_array_equal(
            self.tracker._association_positions([track], predicted, detection), predicted
        )

    def test_dormant_reconnection_needs_two_consecutive_matches(self):
        self.tracker.step(observation([[0, 0, 0]]))
        self.tracker.step(observation([[0, 0, 0]]))
        for _ in range(9):
            self.tracker.step(observation())
        first = self.tracker.step(observation([[0.001, 0, 0]]))
        self.assertFalse(first.assignments)
        self.assertFalse(first.births)
        self.assertEqual(self.tracker.tracks[0].anchor_frame, 1)
        length = self.tracker.tracks[0].memory["length"]
        second = self.tracker.step(observation([[0.001, 0, 0]]))
        self.assertEqual(second.assignments, {0: 0})
        self.assertEqual(second.statuses[0], "observed")
        self.assertEqual(self.tracker.tracks[0].memory["length"], length + 1)

    def test_uncalibrated_uncertainty_disables_generated_nodes(self):
        self.tracker.calibration.variance_scale[:] = np.inf
        self.tracker.step(observation([[0, 0, 0]]))
        self.tracker.step(observation([[0, 0, 0]]))
        result = self.tracker.step(observation())
        self.assertFalse(result.adopted_predictions)
        self.assertFalse(result.graph.geometry.node_valid.any())

    def test_age_specific_uncalibrated_bin_cannot_admit_a_point(self):
        scales = np.full((3, 33), np.inf)
        scales[:, 1] = 0.1
        self.tracker.calibration.variance_scale = scales
        self.tracker.step(observation([[0, 0, 0]]))
        self.tracker.step(observation([[0, 0, 0]]))
        self.assertEqual(self.tracker.step(observation()).adopted_predictions, [0])
        self.assertEqual(self.tracker.step(observation()).adopted_predictions, [])

    def test_dormant_confirmation_is_broken_by_a_missing_frame(self):
        self.tracker.step(observation([[0, 0, 0]]))
        self.tracker.step(observation([[0, 0, 0]]))
        for _ in range(9):
            self.tracker.step(observation())
        self.tracker.step(observation([[0, 0, 0]]))
        self.tracker.step(observation())
        self.assertFalse(self.tracker.step(observation([[0, 0, 0]])).assignments)
        self.assertEqual(self.tracker.step(observation([[0, 0, 0]])).assignments, {0: 0})


class CornerMotionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cpu_observed_rotation_features_match_torch_reference(self):
        rng = np.random.default_rng(78)
        current = rng.normal(size=(32, 4)).astype(np.float32)
        previous = rng.normal(size=(32, 4)).astype(np.float32)
        current /= np.linalg.norm(current, axis=-1, keepdims=True)
        previous /= np.linalg.norm(previous, axis=-1, keepdims=True)
        actual = observed_rotation_change(current, previous)
        expected = rotation_change(torch.from_numpy(current), torch.from_numpy(previous))
        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6, rtol=1e-5)

    def test_both_backbones_have_identical_common_initialization(self):
        config = resolve_config(load_config("graph/configs/temporal/corner_tracking/smoke.yaml"))
        a, ga = build_models(config, "mamba", 3, torch.device("cpu"))
        b, gb = build_models(config, "transformer", 3, torch.device("cpu"))
        self.assertEqual(shared_initialization_digest(a, ga), shared_initialization_digest(b, gb))

    def test_dense_context_pooling_matches_per_type_mean_and_gradients(self):
        torch.manual_seed(22)
        model = CornerMotionModel(d_model=16, d_state=4, d_conv=2)
        reference = deepcopy(model)
        inputs = torch.randn(9, 5)
        kinds = torch.tensor([0, 0, 0, 2, 2, 2, 2, 2, 2])
        membership = torch.nn.functional.one_hot(kinds, 3).float()
        actual = model._context_values(inputs, membership)
        values = reference.context_encoder(inputs)
        expected = torch.stack([values[kinds == kind].mean(0) if (kinds == kind).any()
                                else torch.zeros(16) for kind in range(3)])
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for left, right in zip(model.context_encoder.parameters(), reference.context_encoder.parameters()):
            torch.testing.assert_close(left.grad, right.grad)

    def test_batched_frame_graphs_preserve_predictions_loss_and_gradients(self):
        groups, _ = synthetic_dataset(frames=7, training_episodes=1)
        episode = groups["train"][0]
        plan = dict(mode="single", start=3, stop=5, references=[0], kind=0)
        config = resolve_config(load_config("graph/configs/temporal/corner_tracking/smoke.yaml"))
        gate = calibration()
        gate.variance_scale[:] = 0.1
        for architecture, readout in (("mamba", "attention"), ("transformer", "multiseed")):
            with self.subTest(architecture=architecture, readout=readout):
                config["gnn"].update(global_geometry=True, readout=readout)
                motion, gnn = build_models(config, architecture, 0, torch.device("cpu"))
                batched_motion, batched_gnn = deepcopy(motion), deepcopy(gnn)
                batched_gnn.corner_graph_batch_size = 3
                scalar = replay_episode(motion, gnn, episode, plan, gate, TrackingSettings())
                batched = replay_episode(batched_motion, batched_gnn, episode, plan, gate, TrackingSettings())
                self.assertEqual(scalar["trace"], batched["trace"])
                np.testing.assert_allclose(scalar["prediction"], batched["prediction"], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(scalar["object_loss"], batched["object_loss"])
                for result in (scalar, batched):
                    (result["object_loss"] + 0.1 * result["node_loss"]).backward()
                for left_module, right_module in ((motion, batched_motion), (gnn, batched_gnn)):
                    for (name, left), (_, right) in zip(left_module.named_parameters(), right_module.named_parameters()):
                        if left.grad is None or right.grad is None:
                            self.assertIs(left.grad, right.grad, name)
                        else:
                            torch.testing.assert_close(left.grad, right.grad, rtol=1e-3, atol=1e-5, msg=name)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Inductor")
    def test_compiled_replay_matches_eager_predictions_and_gradients(self):
        groups, _ = synthetic_dataset(frames=7, training_episodes=1)
        episode = groups["train"][0]
        config = resolve_config(load_config("graph/configs/temporal/corner_tracking/smoke.yaml"))
        device = torch.device("cuda")
        plan = dict(mode="natural", start=0, stop=0)
        for architecture in ("mamba", "transformer"):
            with self.subTest(architecture=architecture):
                config["execution"]["compile"] = False
                config["execution"]["graph_batch_size"] = 1
                eager, eager_gnn = build_models(config, architecture, 0, device)
                config["execution"]["compile"] = True
                config["execution"]["graph_batch_size"] = 3
                compiled, compiled_gnn = build_models(config, architecture, 0, device)
                torch.nn.init.normal_(eager.motion_head[-1].weight, std=0.01)
                compiled.load_state_dict(eager.state_dict())
                self.assertEqual(eager.state_dict().keys(), compiled.state_dict().keys())
                full = replay_episode(eager, eager_gnn, episode, plan, calibration(), TrackingSettings())
                optimized = replay_episode(compiled, compiled_gnn, episode, plan, calibration(), TrackingSettings())
                self.assertEqual(full["trace"], optimized["trace"])
                self.assertEqual(full["node_supervision_count"], optimized["node_supervision_count"])
                np.testing.assert_allclose(full["prediction"], optimized["prediction"], rtol=1e-4, atol=1e-5)
                for result in (full, optimized):
                    (result["object_loss"] + 0.1 * result["node_loss"]).backward()
                for left_module, right_module in ((eager, compiled), (eager_gnn, compiled_gnn)):
                    for (name, left), (_, right) in zip(left_module.named_parameters(), right_module.named_parameters()):
                        if left.grad is None or right.grad is None:
                            self.assertIs(left.grad, right.grad, name)
                        else:
                            torch.testing.assert_close(left.grad, right.grad, rtol=2e-3, atol=2e-5, msg=name)
                for module in (eager, eager_gnn, compiled, compiled_gnn):
                    module.eval()
                with torch.no_grad():
                    reference = replay_episode(eager, eager_gnn, episode, plan, calibration(), TrackingSettings())
                    evaluation = replay_episode(compiled, compiled_gnn, episode, plan, calibration(), TrackingSettings())
                np.testing.assert_allclose(reference["prediction"], evaluation["prediction"], rtol=1e-4, atol=1e-5)

    def test_functional_mamba_step_matches_existing_reference_and_gradients(self):
        from graph.temporal.insertion import ReferenceMamba
        from graph.temporal.corner_tracking.model import differentiable_mamba_step

        torch.manual_seed(9)
        mixer = ReferenceMamba(16, d_state=4, d_conv=2)
        cache = mixer.allocate_inference_cache(1)
        reference_cache = mixer.allocate_inference_cache(1)
        outputs = []
        for value in torch.randn(1, 5, 16).split(1, dim=1):
            actual, convolution, state = differentiable_mamba_step(mixer, value, *cache)
            expected, ref_convolution, ref_state = mixer.step(value, *reference_cache)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(convolution, ref_convolution)
            torch.testing.assert_close(state, ref_state)
            cache, reference_cache = (convolution, state), (ref_convolution, ref_state)
            outputs.append(actual)
        torch.cat(outputs, dim=1).square().sum().backward()
        self.assertTrue(torch.isfinite(mixer.A_log.grad).all())

    def test_batched_tracks_match_single_track_replay_and_gradients(self):
        self.assert_batched_tracks_match_single_track(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_batched_tracks_match_single_track_on_cuda(self):
        self.assert_batched_tracks_match_single_track(torch.device("cuda:0"))

    def assert_batched_tracks_match_single_track(self, device):
        """Check assignment, predictions and gradients across ragged track births."""
        groups, _ = synthetic_dataset(frames=14, training_episodes=1)
        episode = groups["train"][0]
        # Force different birth times so Transformer history lengths differ.
        keep = np.arange(len(episode.frames[0].physical_ids)) < 4
        episode.frames[0].observation = episode.frames[0].observation.subset(keep)
        episode.frames[0].physical_ids = episode.frames[0].physical_ids[keep]
        plan = dict(mode="single", start=4, stop=7, references=[0], kind=0)
        for architecture in ("mamba", "transformer"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(81)
                batched = CornerMotionModel(architecture=architecture, d_model=16, d_state=4, d_conv=2)
                torch.nn.init.normal_(batched.motion_head[-1].weight, std=0.01)
                batched = batched.to(device)
                single = deepcopy(batched)
                gnn_batched = VertexEdgeGatedMPNN(16, hidden=16, layers=1, extra_node_dim=5).to(device)
                gnn_single = deepcopy(gnn_batched)
                gate = calibration()
                gate.variance_scale[:] = 0.1
                result_batched = replay_episode(batched, gnn_batched, episode, plan, gate, TrackingSettings())
                with patch("training.gnn.temporal.corner_tracking_rollout.CornerTracker",
                           side_effect=lambda model, calibration, settings: CornerTracker(
                               model, calibration, settings, batch_tracks=False)):
                    result_single = replay_episode(single, gnn_single, episode, plan, gate, TrackingSettings())
                np.testing.assert_allclose(result_batched["prediction"], result_single["prediction"],
                                           rtol=2e-5, atol=2e-6)
                self.assertEqual(result_batched["trace"], result_single["trace"])
                loss_batched = result_batched["object_loss"] + 0.1 * result_batched["node_loss"]
                loss_single = result_single["object_loss"] + 0.1 * result_single["node_loss"]
                loss_batched.backward()
                loss_single.backward()
                for (name, left), (_, right) in zip(batched.named_parameters(), single.named_parameters()):
                    if left.grad is None or right.grad is None:
                        self.assertIs(left.grad, right.grad, name)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, rtol=1e-3, atol=1e-5, msg=name)

    def test_training_without_diagnostic_exports_preserves_loss_and_gradients(self):
        groups, _ = synthetic_dataset(frames=7, training_episodes=1)
        episode = groups["train"][0]
        plan = dict(mode="natural", start=0, stop=0)
        for architecture in ("mamba", "transformer"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(12)
                full_model = CornerMotionModel(architecture=architecture, d_model=16, d_state=4, d_conv=2)
                lean_model = deepcopy(full_model)
                full_gnn = VertexEdgeGatedMPNN(16, hidden=16, layers=1, extra_node_dim=5)
                lean_gnn = deepcopy(full_gnn)
                full = replay_episode(full_model, full_gnn, episode, plan, calibration(), TrackingSettings())
                lean = replay_episode(lean_model, lean_gnn, episode, plan, calibration(), TrackingSettings(),
                                      collect_diagnostics=False)
                self.assertIsNone(lean["prediction"])
                self.assertEqual(lean["node_supervision_count"], full["node_supervision_count"])
                for name in ("node_loss", "object_loss"):
                    torch.testing.assert_close(lean[name], full[name], rtol=0, atol=0)
                (full["object_loss"] + full["node_loss"]).backward()
                (lean["object_loss"] + lean["node_loss"]).backward()
                for left, right in zip(full_model.parameters(), lean_model.parameters()):
                    if left.grad is None or right.grad is None:
                        self.assertIs(left.grad, right.grad)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_node_only_pretraining_skips_graph_without_changing_gradients(self):
        self.assert_node_only_pretraining_matches(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_node_only_pretraining_matches_on_cuda(self):
        self.assert_node_only_pretraining_matches(torch.device("cuda"))

    def assert_node_only_pretraining_matches(self, device):
        groups, _ = synthetic_dataset(frames=7, training_episodes=1)
        episode = groups["train"][0]
        plan = dict(mode="natural", start=0, stop=0)
        for architecture in ("mamba", "transformer"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(31)
                full_model = CornerMotionModel(architecture=architecture, d_model=16, d_state=4, d_conv=2).to(device)
                lean_model = deepcopy(full_model)
                full_gnn = VertexEdgeGatedMPNN(16, hidden=16, layers=1, extra_node_dim=5).to(device)
                lean_gnn = deepcopy(full_gnn)
                full = replay_episode(full_model, full_gnn, episode, plan, calibration(), TrackingSettings())
                with patch("graph.temporal.corner_tracking.tracker.build_completed_graph",
                           side_effect=AssertionError("node-only pretraining built a graph")):
                    lean = replay_episode(lean_model, lean_gnn, episode, plan, calibration(),
                                          TrackingSettings(), collect_diagnostics=False, compute_object_loss=False)
                self.assertIsNone(lean["object_loss"])
                self.assertEqual(lean["node_supervision_count"], full["node_supervision_count"])
                rtol, atol = (1e-5, 1e-6) if device.type == "cuda" else (0, 0)
                torch.testing.assert_close(lean["node_loss"], full["node_loss"], rtol=rtol, atol=atol)
                full["node_loss"].backward()
                lean["node_loss"].backward()
                for left, right in zip(full_model.parameters(), lean_model.parameters()):
                    if left.grad is None or right.grad is None:
                        self.assertIs(left.grad, right.grad)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol)

    def test_complete_episode_backward_and_gt_input_separation(self):
        groups, _ = synthetic_dataset(frames=7, training_episodes=1)
        episode = groups["train"][0]
        plan = dict(mode="natural", start=0, stop=0)
        for architecture in ("mamba", "transformer"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(4)
                model = CornerMotionModel(architecture=architecture, d_model=16, heads=4,
                                          d_state=4, d_conv=2)
                gnn = VertexEdgeGatedMPNN(16, hidden=16, layers=1, extra_node_dim=5)
                result = replay_episode(model, gnn, episode, plan, calibration(), TrackingSettings())
                loss = result["node_loss"] + result["object_loss"]
                loss.backward()
                self.assertGreater(result["node_supervision_count"], 0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(float(model.motion_head[-1].weight.grad.abs().sum()), 0)
                changed = deepcopy(episode)
                for frame in changed.frames:
                    frame.reference_xyz += 100
                    frame.physical_ids[:] = -1
                    frame.target_object21[:3] += 20
                with torch.no_grad():
                    alternative = replay_episode(model, gnn, changed, plan, calibration(), TrackingSettings())
                np.testing.assert_array_equal(result["prediction"], alternative["prediction"])

    def test_position_error_matches_certified_diagnostics(self):
        groups, _ = synthetic_dataset(frames=7, training_episodes=1)
        model = CornerMotionModel(architecture="mamba", d_model=16, d_state=4, d_conv=2)
        gnn = VertexEdgeGatedMPNN(16, hidden=16, layers=1, extra_node_dim=5)
        result = replay_episode(model, gnn, groups["train"][0], dict(mode="natural", start=0, stop=0),
                                calibration(), TrackingSettings())
        errors = [row["error_m"] for row in result["nodes"] if row["identity_consistent"]]
        self.assertEqual(len(errors), result["node_supervision_count"])
        self.assertGreater(len(errors), 0)
        self.assertAlmostEqual(float(result["position_error_m"]), float(np.mean(errors)), places=6)

    def test_generated_coordinates_receive_gnn_gradient(self):
        model = CornerMotionModel(d_model=16, d_state=4, d_conv=2)
        calibrated = calibration()
        calibrated.variance_scale[:] = 0.1
        tracker = CornerTracker(model, calibrated)
        gnn = VertexEdgeGatedMPNN(16, hidden=16, layers=1, extra_node_dim=5)
        tracker.step(observation([[0, 0, 0]]))
        tracker.step(observation([[0, 0, 0]]))
        result = tracker.step(observation())
        self.assertEqual(result.adopted_predictions, [0])
        prediction = result.priors[0].prediction.delta
        prediction.retain_grad()
        gnn(result.graph.geometry, result.graph.features())["object21"].square().sum().backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0)

    def test_current_observation_cannot_change_prior_prediction(self):
        for architecture in ("mamba", "transformer"):
            model = CornerMotionModel(architecture=architecture, d_model=16, d_state=4, d_conv=2)
            torch.nn.init.normal_(model.motion_head[-1].weight, std=0.05)
            trackers = [CornerTracker(model, calibration()) for _ in range(2)]
            for tracker in trackers:
                tracker.step(observation([[0, 0, 0]]))
                tracker.step(observation([[0, 0, 0]]))
            a = trackers[0].step(observation([[0, 0, 0]]))
            b = trackers[1].step(observation([[0.005, 0, 0]]))
            torch.testing.assert_close(a.priors[0].position, b.priors[0].position)

    def test_extra_gnn_features_obey_dtype_and_padding_contract(self):
        from types import SimpleNamespace
        from graph.build_graph import GraphBuilder

        item = SimpleNamespace(xyz=torch.zeros(1, 3), object_id=torch.tensor([0]), metadata=[{}])
        builder = GraphBuilder()
        plain = builder([item], torch.zeros(1, 3))
        padded = builder([item], torch.zeros(1, 3), pad_to=3)
        gnn = VertexEdgeGatedMPNN(16, hidden=16, layers=1, extra_node_dim=5)
        expected = gnn(plain, torch.ones(1, 1, 5))["object21"]
        features = torch.full((1, 3, 5), 999.0)
        features[:, :1] = 1
        torch.testing.assert_close(expected, gnn(padded, features)["object21"])
        with self.assertRaisesRegex(ValueError, "shape"):
            gnn(plain)
        with self.assertRaisesRegex(ValueError, "dtype"):
            gnn(plain, torch.ones(1, 1, 5, dtype=torch.float64))


class CornerDataTests(unittest.TestCase):
    def test_family_adapter_reuses_real_cache_and_raw_observation_contract(self):
        from data_collection.episode_dataset import canonical_hash, file_sha256

        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {
                "WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(Path(directory) / "experiments")}):
            root = Path(directory) / "family"
            manifest = write_corner_family_source(root)
            original_hashes = {path: file_sha256(path) for path in root.rglob("*.npz")}
            output = prepare(dict(source={"family_dataset": str(root)}, seed=0,
                                  output={"root": str(Path(directory) / "prepared")}))
            packed, _ = load_dataset(output)
            self.assertEqual([len(packed[key]) for key in ("train", "calibration", "validation")], [1, 1, 1])
            self.assertTrue(all(np.any(frame.physical_ids >= 0) for frame in packed["train"][0].frames))
            self.assertEqual(original_hashes, {path: file_sha256(path) for path in original_hashes})
            first_episode = Path(manifest["samples"][0]["sample_id"]).parent
            missing_frame = str(first_episode / "frame_000001.npz")
            manifest["samples"] = [row for row in manifest["samples"] if row["sample_id"] != missing_frame]
            manifest.pop("dataset_sha256")
            manifest["dataset_sha256"] = canonical_hash(manifest)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "contiguous full episodes"):
                prepare(dict(source={"family_dataset": str(root)}, output={"root": str(Path(directory) / "rejected")}))

    def test_packing_parent_checks_and_corruption(self):
        groups, _ = synthetic_dataset(frames=5, training_episodes=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            episodes = [episode for group in groups.values() for episode in group]
            save_dataset(root, episodes, {})
            loaded, manifest = load_dataset(root)
            self.assertEqual(len(loaded["train"][0].frames), 5)
            path = root / manifest["episodes"][0]["file"]
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_dataset(root)

    def test_masks_are_paired_and_remove_context_evidence(self):
        groups, _ = synthetic_dataset(frames=40, training_episodes=1)
        episode = groups["train"][0]
        for mode, probabilities in (("single", [0, 1, 0, 0]), ("group", [0, 0, 1, 0]),
                                    ("object", [0, 0, 0, 1])):
            plan = mask_plan(episode, 7, probabilities)
            self.assertEqual(plan, mask_plan(episode, 7, probabilities))
            self.assertEqual(plan["mode"], mode)
            self.assertGreaterEqual(plan["start"], 2)
            frame = episode.frames[plan["start"]]
            obs, identities, _ = masked_frame(frame, plan, plan["start"])
            if mode == "object":
                self.assertFalse(np.any(obs.kind == plan["kind"]))
            else:
                self.assertFalse(set(identities) & set(plan["references"]))
            self.assertEqual(len(frame.physical_ids), len(frame.observation.xyz))

    def test_earlier_unknown_corner_is_not_eligible_for_masking(self):
        groups, _ = synthetic_dataset(frames=5, training_episodes=1)
        episode = groups["train"][0]
        for frame in episode.frames:
            frame.physical_ids[:] = -1
        plan = mask_plan(episode, 0, [0, 1, 0, 0])
        self.assertIn("skipped", plan)
        self.assertEqual(plan["actual_length"], 0)

    def test_common_mask_eligibility_requires_a_confirmed_prior_track(self):
        groups, _ = synthetic_dataset(frames=8, training_episodes=1)
        episode = groups["train"][0]
        eligible = confirmed_eligibility(episode, calibration(), TrackingSettings())
        self.assertFalse(eligible[0])
        self.assertFalse(eligible[1])
        self.assertTrue(eligible[2])
        plan = mask_plan(episode, 4, [0, 1, 0, 0], eligible_references=eligible)
        self.assertTrue(set(plan["references"]) <= eligible[plan["start"]])

    def test_eligibility_without_graph_matches_full_graph_replay(self):
        groups, _ = synthetic_dataset(frames=12, training_episodes=1)
        episode = groups["train"][0]
        original_step = CornerTracker.step

        def step_with_graph(tracker, observation, *, include_graph=True):
            return original_step(tracker, observation, include_graph=True)

        with patch.object(CornerTracker, "step", step_with_graph):
            original = confirmed_eligibility(deepcopy(episode), calibration(), TrackingSettings())
        optimized = confirmed_eligibility(deepcopy(episode), calibration(), TrackingSettings())
        self.assertEqual(original, optimized)

    def test_configuration_rejects_unpaired_or_invalid_conditions(self):
        for override in ({"masking": {"mode_probabilities": [1, 1, 1, 1]}},
                         {"training": {"joint_epochs": 0}}, {"gnn": {"extra_node_dim": 0}},
                         {"seeds": [0, 0]}):
            with self.assertRaises(ValueError):
                resolve_config(dict(data={"kind": "synthetic"}, **override))


class CornerWorkflowTests(unittest.TestCase):
    def test_small_paired_training_checkpoint_evaluation_and_tracking(self):
        from training.gnn.temporal.corner_tracking import evaluate_checkpoint

        config = load_config("graph/configs/temporal/corner_tracking/smoke.yaml")
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"WANDB_MODE": "disabled"}):
            config["output"] = {"root": str(Path(directory) / "runs")}
            config["data"]["frames"] = 8
            with patch.dict("os.environ", {"EXPERIMENTS_DIR": str(Path(directory) / "experiments")}):
                report_path = compare(config)
                report = json.loads(Path(report_path).read_text())
                self.assertEqual(len(report["runs"]), 2)
                a, b = report["runs"]
                for key in ("shared_initialization_sha256", "validation_mask_sha256", "training_mask_sha256"):
                    self.assertEqual(a[key], b[key])
                for row in report["runs"]:
                    checkpoint = torch.load(row["checkpoint"], weights_only=True)
                    self.assertEqual(checkpoint["kind"], "corner_tracking")
                    last = torch.load(Path(row["output"]) / "last.pt", weights_only=True)
                    self.assertIn("optimizer", last)
                    self.assertEqual(last["epoch"], 2)
                    self.assertTrue(Path(row["checkpoint"] + ".tracking.json").is_file())
                    target = evaluate_checkpoint(dict(checkpoint=row["checkpoint"], device="cpu",
                                                       output={"root": str(Path(directory) / "evaluation")}))
                    self.assertTrue((Path(target) / "predictions.npz").is_file())

                # Extending an epoch-boundary checkpoint reproduces a fresh run
                # with the same longer budget, including calibrated graph policy.
                from training.gnn.temporal.corner_tracking import train
                resume = deepcopy(config)
                resume.update(architectures=["mamba"], seeds=[0],
                              resume_checkpoint=str(Path(a["output"]) / "last.pt"))
                resume["training"]["joint_epochs"] = 2
                resume["output"]["root"] = str(Path(directory) / "resumed")
                resumed = train(resume)
                fresh = deepcopy(resume)
                fresh.pop("resume_checkpoint")
                fresh["output"]["root"] = str(Path(directory) / "fresh")
                uninterrupted = train(fresh)
                resumed_state = torch.load(Path(resumed["output"]) / "last.pt", weights_only=True)
                fresh_state = torch.load(Path(uninterrupted["output"]) / "last.pt", weights_only=True)
                for module in ("motion", "gnn"):
                    for key in resumed_state[module]:
                        torch.testing.assert_close(resumed_state[module][key], fresh_state[module][key], rtol=0, atol=0)
                self.assertEqual(resumed_state["mask_ledger"], fresh_state["mask_ledger"])


if __name__ == "__main__":
    unittest.main()
