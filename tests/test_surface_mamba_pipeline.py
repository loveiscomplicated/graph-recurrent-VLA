"""Contract tests for the frozen Surface frontend and Mamba controller pipeline."""

import unittest

import numpy as np
import torch

from data_collection.surface_mamba_dataset import apply_dart_noise
from graph.surface_context_frontend import SurfaceTrackerStream
from graph.temporal.surface_spatial_corner import SurfaceSpatialCornerTracker
from recurrent.surface_mamba import (
    ControllerEnsemble,
    MambaController,
    mix_state36,
    state36_to_controller_features,
)


class SurfaceTrackerStreamingTests(unittest.TestCase):
    def test_streaming_matches_full_causal_forward(self):
        torch.manual_seed(4)
        generator = np.random.default_rng(4)
        model = SurfaceSpatialCornerTracker(width=16, heads=4).eval()
        fallback = generator.normal(size=(20, 3)).astype(np.float32) * 0.05
        length = 6
        detected = np.zeros((length, 20, 3), dtype=np.float32)
        observed = generator.random((length, 20)) > 0.65
        detected[observed] = generator.normal(size=(observed.sum(), 3)).astype(np.float32) * 0.05
        tcp = generator.normal(size=(length, 3)).astype(np.float32) * 0.02
        surface_xyz = generator.normal(size=(length, 32, 3)).astype(np.float32) * 0.05
        surface_valid = generator.random((length, 32)) > 0.25
        surface_xyz[~surface_valid] = 0

        time = np.arange(length)[:, None]
        last = np.maximum.accumulate(np.where(observed, time, -1), axis=0)
        seen = last >= 0
        last_xyz = detected[np.maximum(last, 0), np.arange(20)]
        anchor = np.where(seen[..., None], last_xyz, fallback[None])
        age = time - last
        features = np.concatenate(
            (
                (anchor - fallback[None]) / 0.1,
                (tcp[:, None] - anchor) / 0.1,
                observed[..., None],
                seen[..., None],
                (np.minimum(age, 40) / 40)[..., None],
                np.broadcast_to(time[..., None] / 200, (length, 20, 1)),
            ),
            axis=-1,
        ).astype(np.float32)
        center = fallback[:8].mean(axis=0)
        surface_features = np.concatenate(
            (
                (surface_xyz - center) / 0.1,
                (tcp[:, None] - surface_xyz) / 0.1,
            ),
            axis=-1,
        ).astype(np.float32)
        surface_features[~surface_valid] = 0
        with torch.inference_mode():
            full = model(
                torch.from_numpy(features[None]),
                torch.from_numpy(anchor[None].astype(np.float32)),
                torch.from_numpy(detected[None]),
                torch.from_numpy(observed[None]),
                torch.ones((1, length), dtype=torch.bool),
                torch.from_numpy(surface_features[None]),
                torch.from_numpy(surface_valid[None]),
            )[0].numpy()

        stream = SurfaceTrackerStream(model, fallback, torch.device("cpu"))
        streamed = []
        streamed_seen = []
        for index in range(length):
            prediction, valid = stream.step(
                detected[index],
                observed[index],
                tcp[index],
                surface_xyz[index],
                surface_valid[index],
            )
            streamed.append(prediction)
            streamed_seen.append(valid.copy())
        np.testing.assert_allclose(np.stack(streamed), full, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(np.stack(streamed_seen), seen)
        np.testing.assert_array_equal(np.stack(streamed)[observed], detected[observed])


class MambaContractTests(unittest.TestCase):
    def test_full_and_incremental_mamba_are_equal(self):
        torch.manual_seed(7)
        model = MambaController(d_model=16, d_state=4, d_conv=3, layers=2).eval()
        sequence = torch.randn(2, 5, 44)
        with torch.inference_mode():
            full, _ = model(sequence)
            state = None
            steps = []
            for index in range(sequence.shape[1]):
                prediction, state = model(sequence[:, index : index + 1], state)
                steps.append(prediction)
        torch.testing.assert_close(torch.cat(steps, dim=1), full)

    def test_state_mixing_and_44d_conversion_boundaries(self):
        ground_truth = np.zeros((2, 36), dtype=np.float32)
        graph = np.ones((2, 36), dtype=np.float32)
        for values in (ground_truth, graph):
            values[:, 3] = 1
            values[:, 18] = 1
            values[:, 28] = 1
        np.testing.assert_array_equal(mix_state36(ground_truth, graph, 0), ground_truth)
        np.testing.assert_array_equal(mix_state36(ground_truth, graph, 1), graph)
        features = state36_to_controller_features(mix_state36(ground_truth, graph, 0.25))
        self.assertEqual(features.shape, (2, 44))
        self.assertTrue(np.isfinite(features).all())

    def test_ensemble_averages_motion_and_votes_gripper(self):
        class FixedController:
            def __init__(self, action):
                self.action = np.asarray(action, dtype=np.float32)
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1

            def act(self, _):
                return self.action.copy()

        controllers = [
            FixedController([index] * 6 + ([1] if index < 3 else [-1]))
            for index in range(5)
        ]
        ensemble = ControllerEnsemble(controllers)
        ensemble.reset()
        action = ensemble.act(np.zeros(36, dtype=np.float32))
        np.testing.assert_allclose(action[:6], 2)
        self.assertEqual(action[6], 1)
        self.assertTrue(all(controller.reset_count == 1 for controller in controllers))


class DartCollectionTests(unittest.TestCase):
    def test_dart_noise_preserves_gripper_and_is_deterministic(self):
        clean = np.asarray([[0.99, 0, 0, 0, 0, 0, -1]], dtype=np.float32)
        first = apply_dart_noise(clean, np.random.default_rng(9), 0.02)
        second = apply_dart_noise(clean, np.random.default_rng(9), 0.02)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, 6], clean[:, 6])
        self.assertLessEqual(np.abs(first[:, :6]).max(), 1)


if __name__ == "__main__":
    unittest.main()
