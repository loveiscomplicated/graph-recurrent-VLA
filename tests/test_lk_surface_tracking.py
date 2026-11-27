"""Check optical flow, RGB-D masking and episode reset for the LK backend."""

import unittest

import cv2
import numpy as np
import torch

from graph.surface_context_frontend import _default_surface_tracking
from visual.lk_surface_tracking import LKSurfaceTracker
from visual.peg_surface_points import choose_points, lift


class SurfaceTrackingDefaultTests(unittest.TestCase):
    def test_device_defaults_preserve_legacy_cuda_evaluations(self):
        cases = (
            ("cuda:0", "cuda:0", {}, "lk"),
            ("cpu", "cuda:1", {}, "lk"),
            ("cuda:0", "cuda:0", {"tapnextpp": {}}, "tapnextpp"),
            ("cpu", "mps", {}, "lk"),
            ("cpu", "cpu", {}, "tapnextpp"),
        )
        for model, detector, config, expected in cases:
            with self.subTest(model=model, detector=detector, config=config):
                actual = _default_surface_tracking(
                    config, torch.device(model), torch.device(detector)
                )
                self.assertEqual(actual, expected)


class LKSurfaceTrackingTests(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(21)
        texture = random.integers(0, 256, (192, 192, 3), dtype=np.uint8)
        self.rgb = cv2.GaussianBlur(texture, (7, 7), 1.4)
        self.mask = np.zeros(self.rgb.shape[:2], dtype=bool)
        self.mask[56:136, 56:136] = True
        self.depth = np.full(self.mask.shape, 600, dtype=np.int16)

    def test_retains_and_moves_tracks_with_visible_texture(self):
        tracker = LKSurfaceTracker(8)
        initial, valid = tracker.step(self.rgb, self.depth, self.mask)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(initial, choose_points(self.mask, self.depth, 8))

        shift = np.array([[1, 0, 1], [0, 1, -1]], dtype=np.float32)
        moved = cv2.warpAffine(self.rgb, shift, self.mask.shape[::-1])
        current, valid = tracker.step(moved, self.depth, self.mask)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(current, initial + [1, -1], atol=0.1)

    def test_invalid_tracks_reseed_only_where_depth_and_mask_allow(self):
        tracker = LKSurfaceTracker(8)
        initial, _ = tracker.step(self.rgb, self.depth, self.mask)
        hidden_mask = np.zeros_like(self.mask)
        limited_depth = np.zeros_like(self.depth)
        limited_depth[60, 60] = 600

        _, hidden_valid = tracker.step(self.rgb, self.depth, hidden_mask)
        self.assertFalse(hidden_valid.any())
        current, valid = tracker.step(self.rgb, limited_depth, self.mask)
        np.testing.assert_array_equal(current[valid], [[60, 60]])
        indices, xyz = lift(current, limited_depth, self.mask, np.eye(3), valid)
        np.testing.assert_array_equal(indices, [0])
        np.testing.assert_allclose(xyz, [[36.0, 36.0, 0.6]])

        tracker.reset()
        reset, reset_valid = tracker.step(self.rgb, self.depth, self.mask)
        self.assertTrue(reset_valid.all())
        np.testing.assert_array_equal(reset, initial)
