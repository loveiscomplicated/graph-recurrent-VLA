"""Check physical vector errors and frame pooling in the held-out GNN report."""

import unittest

import numpy as np

from evaluation.frozen_surface_tracker_gnn import position_errors_mm
from evaluation.five_mode_corner_tracker_comparison import summarize


class Object21ErrorTest(unittest.TestCase):
    def test_vector_norm_and_frame_weighted_percentile(self):
        target = np.zeros((3, 21))
        prediction = target.copy()
        prediction[:, 0:3] = [[0.003, 0.004, 0], [0, 0, 0], [0, 0, 0.010]]
        prediction[:, 10:13] = [0, 0.006, 0.008]
        prediction[:, 7:10] = [-0.003, -0.004, 0]
        errors = position_errors_mm(prediction, target)
        np.testing.assert_allclose(errors["peg_position"], [5, 0, 10])
        np.testing.assert_allclose(errors["hole_position"], [10, 10, 10])
        np.testing.assert_allclose(errors["tip_minus_hole"], [5, 5, 5])
        self.assertEqual(summarize(errors["peg_position"])["mean_mm"], 5)
        self.assertEqual(summarize(errors["peg_position"])["p90_mm"], 9)

    def test_nonfinite_predictions_fail_instead_of_dropping_frames(self):
        prediction = np.zeros((2, 21))
        prediction[1, 7] = np.nan
        with self.assertRaises(ValueError):
            position_errors_mm(prediction, np.zeros_like(prediction))


if __name__ == "__main__":
    unittest.main()
