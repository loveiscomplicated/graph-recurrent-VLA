"""Regression checks for paired post-observation error aggregation."""

import unittest

import numpy as np

from evaluation.five_mode_corner_tracker_comparison import observation_history
from evaluation.observed_corner_surface_comparison import paired_metrics


class ObservedCornerSurfaceComparisonTests(unittest.TestCase):
    def test_only_past_observed_currently_missing_points_are_scored(self):
        observed = np.zeros((5, 20), dtype=bool)
        observed[2, 0] = True
        episode = {
            "observed": observed,
            "detected": np.zeros((5, 20, 3)),
            "gt": np.zeros((5, 20, 3)),
            "mode": "fixed",
        }
        history = observation_history([episode])
        eligible = history["ever"] & ~observed
        error = np.full((5, 20), 100000, dtype=np.float32)
        error[3:, 0] = [2, 4]
        surface_present = np.ones(5, dtype=bool)
        result = paired_metrics(error, eligible, [episode], history, surface_present)
        self.assertEqual(result["body"]["all"]["count"], 2)
        self.assertEqual(result["body"]["peg"]["mean_mm"], 3)
        self.assertAlmostEqual(result["body"]["peg"]["p90_mm"], 3.8)
        self.assertEqual(result["body"]["hole"]["count"], 0)
        self.assertEqual(result["peg_surface_support"]["present"]["count"], 2)
        error[~eligible] *= 100
        changed = paired_metrics(error, eligible, [episode], history, surface_present)
        self.assertEqual(changed, result)


if __name__ == "__main__":
    unittest.main()
