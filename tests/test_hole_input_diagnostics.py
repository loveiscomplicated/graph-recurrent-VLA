"""Check aperture projection and depth probes used by the case study."""

import unittest

import numpy as np

from evaluation.hole_input_diagnostics import aperture_geometry, probe_aperture_centers


class HoleInputDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.zeros((24, 3), dtype=np.float32)
        corners = np.array([[-.1, -.1], [-.1, .1], [.1, -.1], [.1, .1]])
        self.truth[16:20, :2] = corners
        self.truth[16:20, 2] = 2
        self.truth[20:24, :2] = corners
        self.truth[20:24, 2] = 3
        self.intrinsic = np.array([[100, 0, 256], [0, 100, 256], [0, 0, 1]])
        self.extrinsic = np.column_stack((np.eye(3), np.zeros(3)))

    def test_frontal_aperture_projection(self):
        result = aperture_geometry(self.truth, self.intrinsic, self.extrinsic)
        np.testing.assert_allclose(result['bounding_box_px'], [10, 10], atol=1e-5)
        self.assertAlmostEqual(result['normal_view_angle_degrees'], 0)

    def test_depth_residual_compares_surface_with_gt_plane(self):
        depth = np.full((512, 512, 1), 2100, dtype=np.int16)
        result = probe_aperture_centers(self.truth, depth, self.intrinsic, self.extrinsic)
        self.assertAlmostEqual(result[0]['measured_minus_expected_m'], .1)
        self.assertAlmostEqual(result[1]['measured_minus_expected_m'], -.9)


if __name__ == '__main__':
    unittest.main()
