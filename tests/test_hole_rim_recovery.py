"""Synthetic depth evidence checks for opening recovery."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.hole_rim import _depth_supported_square, opening_corners
from visual.corner_detection.peg_prism import prism_corners


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

    def test_square_depth_check_rejects_foreground_and_missing_face(self):
        corners = np.array(
            [[-0.021, -0.021, 1], [-0.021, 0.021, 1],
             [0.021, -0.021, 1], [0.021, 0.021, 1]]
        )
        self.depth[self.opening] = 1.04
        self.support[self.opening] = False
        args = corners, self.depth, self.support, self.intrinsic, self.planes[0]
        self.assertIsNotNone(_depth_supported_square(*args))

        self.depth[self.opening] = 0.96
        self.assertIsNone(_depth_supported_square(*args))

        self.depth[self.opening] = 1.04
        self.support[:, 38:] = False
        self.assertIsNone(_depth_supported_square(*args))

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

    def test_grid_hole_fallback_queues_verified_ransac_opening(self):
        detector = TorchCornerDetector(
            "cpu",
            recover_hole_rim=True,
            recover_small_geometry=True,
            recover_fragmented_rim=True,
            hole_rim_ransac_fallback=True,
            plane_fit_method="grid",
        )
        detector.xyz = torch.zeros((65, 65, 3))
        detector.depth_cpu = np.ones((65, 65))
        detector.k = torch.as_tensor(self.intrinsic, dtype=torch.float32)
        detector.k_cpu = self.intrinsic
        detector.blocks, detector.rows = [], []
        corners = np.array(
            [
                [-0.02, -0.02, 1],
                [-0.02, 0.02, 1],
                [0.02, -0.02, 1],
                [0.02, 0.02, 1],
            ]
        )
        rejected = [(0, corners, True, False)]
        recovered = [(0, corners, True, True)]
        box = SimpleNamespace(
            oid=1,
            support_cpu=self.support,
            planes_cpu=self.planes[:1],
            eligible_cpu={0},
            points=torch.empty((0, 3)),
            support=torch.as_tensor(self.support),
        )

        with patch.object(
            detector, "_opening_proposals", side_effect=(rejected, recovered)
        ) as proposals, patch.object(
            detector,
            "_ransac_hole_planes",
            return_value=(self.planes, {0, 1}),
        ) as fallback:
            detector.box_openings(box)

        self.assertEqual(proposals.call_count, 2)
        fallback.assert_called_once_with(box)
        self.assertEqual(len(detector.blocks), 2)
        self.assertEqual(detector.blocks[0][3].tolist(), [False] * 4)
        self.assertEqual(detector.blocks[1][3].tolist(), [True] * 4)
        fallback_rows = [row for row in detector.rows if row["block"] == 1]
        self.assertTrue(
            all(
                row["meta"]["plane_fit_method"] == "ransac_fallback"
                for row in fallback_rows
            )
        )

    def test_main_enables_fallback_only_for_grid_plane_fit_by_default(self):
        grid = TorchCornerDetector.main(device="cpu", plane_fit_method="grid")
        ransac = TorchCornerDetector.main(device="cpu", plane_fit_method="ransac")
        historical_grid = TorchCornerDetector.main(
            device="cpu",
            plane_fit_method="grid",
            hole_rim_ransac_fallback=False,
        )

        self.assertTrue(grid.hole_rim_ransac_fallback)
        self.assertFalse(ransac.hole_rim_ransac_fallback)
        self.assertFalse(historical_grid.hole_rim_ransac_fallback)

    def test_main_cuda_preset_enables_grid_hole_fallback(self):
        with patch(
            "visual.corner_detection.detector.K.resolve_device",
            return_value=torch.device("cuda"),
        ), patch(
            "visual.corner_detection.detector.validation_torch.validation_function",
            return_value=lambda *args: None,
        ):
            detector = TorchCornerDetector.main(device="cuda")

        self.assertEqual(detector.plane_fit_method, "grid")
        self.assertTrue(detector.hole_rim_ransac_fallback)

    def test_peg_prism_requires_two_planes_and_depth_near_each_vertex(self):
        x, y = np.meshgrid(np.linspace(-0.1, 0.1, 11), np.linspace(-0.02, 0.02, 5))
        front = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
        x, z = np.meshgrid(np.linspace(-0.1, 0.1, 11), np.linspace(1, 1.04, 5))
        side = np.column_stack((x.ravel(), np.full(x.size, -0.02), z.ravel()))
        cloud = np.concatenate((front, side))
        planes = [(np.array([0.0, 0.0, 1.0]), -1.0), (np.array([0.0, 1.0, 0.0]), 0.02)]
        self.assertEqual(len(prism_corners(cloud, planes[:1])), 0)
        vertices = prism_corners(cloud, planes)
        self.assertEqual(len(vertices), 6)
        self.assertFalse(np.any((vertices[:, 1] > 0) & (vertices[:, 2] > 1.01)))

    def test_depth_component_recovers_rim_when_mask_has_open_seam(self):
        self.depth[self.opening] = 1.1
        self.support[14:27, 32] = False
        enclosed = self.proposals_on_front_face()
        self.assertFalse(any(row[3] for row in enclosed))
        recovered = [
            row
            for row in opening_corners(
                self.depth,
                self.support,
                self.planes,
                {0, 1},
                self.intrinsic,
                small_opening=True,
                fragmented=True,
            )
            if row[0] == 0 and row[3]
        ]
        self.assertEqual(len(recovered), 1)
        np.testing.assert_allclose(np.abs(recovered[0][1][:, :2]), 5.5 / 256)

    def test_component_windows_preserve_enclosed_and_open_seam_proposals(self):
        self.depth[self.opening] = 1.1
        inputs = self.depth, self.support, self.planes, {0, 1}, self.intrinsic
        options = dict(small_opening=True, fragmented=True, verify_rejected_square=True)
        for open_seam in (False, True):
            with self.subTest(open_seam=open_seam):
                if open_seam:
                    self.support[14:27, 32] = False
                reference = list(opening_corners(*inputs, **options))
                cropped = list(opening_corners(*inputs, **options, crop_components=True))
                self.assertTrue(any(row[3] for row in reference))
                self.assertEqual(len(cropped), len(reference))
                for actual, expected in zip(cropped, reference, strict=True):
                    self.assertEqual((actual[0], *actual[2:]), (expected[0], *expected[2:]))
                    np.testing.assert_array_equal(actual[1], expected[1])
