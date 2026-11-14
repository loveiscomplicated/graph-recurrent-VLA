"""Synthetic depth evidence checks for opening recovery."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.hole_rim import opening_corners


class HoleRimRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.intrinsic = np.array([[256, 0, 32], [0, 256, 32], [0, 0, 1.0]])
        self.planes = [
            (np.array([0.0, 0.0, 1.0]), -1.0),
            (np.array([1.0, 0.0, 0.0]), -0.1),
        ]
        self.depth = np.ones((65, 65))
        self.support = np.zeros_like(self.depth, dtype=bool)
        self.support[14:51, 14:51] = True
        self.opening = np.s_[27:38, 27:38]

    def proposals_on_front_face(self):
        return [
            row
            for row in opening_corners(
                self.depth, self.support, self.planes, {0, 1}, self.intrinsic
            )
            if row[0] == 0
        ]

    def test_reconstructs_four_depth_supported_opening_corners(self):
        self.depth[self.opening] = 1.1
        _, corners, rectangular, verified = self.proposals_on_front_face()[0]
        self.assertTrue(rectangular)
        self.assertTrue(verified)
        np.testing.assert_allclose(np.abs(corners[:, :2]), 5.5 / 256)
        np.testing.assert_allclose(corners[:, 2], 1)

    def test_rejects_foreground_occluder_even_with_rectangular_shape(self):
        self.depth[self.opening] = 0.9
        _, _, rectangular, verified = self.proposals_on_front_face()[0]
        self.assertTrue(rectangular)
        self.assertFalse(verified)

    def test_rejects_color_mask_gap_at_the_same_depth(self):
        self.support[self.opening] = False
        _, _, rectangular, verified = self.proposals_on_front_face()[0]
        self.assertTrue(rectangular)
        self.assertFalse(verified)

    def test_refined_merge_replaces_only_nearby_concave_box_points(self):
        detector = TorchCornerDetector("cpu", recover_hole_rim=True)
        detector.xyz = torch.empty((0, 3))
        detector.blocks, detector.rows = [], []
        box = SimpleNamespace(oid=1)
        peg = SimpleNamespace(oid=0)
        points = torch.tensor([[0.0, 0.0, 1.0], [0.040, 0.0, 1.0], [0.080, 0.0, 1.0]])
        detector.add(
            box,
            points,
            points[:, :2],
            torch.ones(3, dtype=torch.bool),
            [
                {"method": "boundary", "line_residual_px": 0.2, "contour_hole": True},
                {"method": "boundary", "line_residual_px": 0.2, "contour_hole": False},
                {"method": "boundary", "line_residual_px": 0.2, "contour_hole": True},
            ],
        )
        detector.add(
            peg,
            points[:1],
            points[:1, :2],
            torch.ones(1, dtype=torch.bool),
            [{"method": "boundary", "line_residual_px": 0.2, "contour_hole": True}],
        )
        opening = torch.tensor([[0.007, 0.0, 1.0], [0.042, 0.0, 1.0]])
        detector.add(
            box,
            opening,
            opening[:, :2],
            torch.ones(2, dtype=torch.bool),
            [{"method": "box_opening", "contour_hole": True}] * 2,
            stage=3,
        )
        result, _, ids, _ = detector.merge()
        torch.testing.assert_close(
            result, torch.cat((points[1:], points[:1], opening[:1]))
        )
        self.assertEqual(ids.tolist(), [1, 1, 0, 1])
