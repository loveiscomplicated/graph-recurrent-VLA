"""Contracts for exogenous motion and paired policy comparisons."""

import unittest

import numpy as np

from data_collection.surface_mamba_motion import make_motion_plan
from evaluation.surface_mamba_moving import paired_comparison


MOTION = {
    "start_step": 10,
    "end_step": 40,
    "distance_m": 0.06,
    "seed_salt": 20261120,
}


class MovingSurfaceTests(unittest.TestCase):
    def test_motion_is_seeded_and_matches_endpoints(self):
        first = make_motion_plan(4, MOTION, 250)
        second = make_motion_plan(4, MOTION, 250)
        np.testing.assert_array_equal(first.offsets, second.offsets)
        np.testing.assert_array_equal(first.offsets[:11], 0)
        np.testing.assert_allclose(first.offsets[25], first.offsets[-1] / 2)
        np.testing.assert_array_equal(first.offsets[40], first.offsets[-1])
        self.assertAlmostEqual(np.linalg.norm(first.offsets[-1]), 0.06)
        self.assertNotEqual(first.sha256, make_motion_plan(5, MOTION, 250).sha256)

    def test_comparison_verifies_motion_and_counts_paired_gains(self):
        metadata = {
            "trajectory_sha256": "plan",
            "control_frequency_hz": 20.0,
            "motion_completed": True,
        }
        baseline = {seed: dict(metadata, success=seed == 0) for seed in range(4)}
        adapted = {seed: dict(metadata, success=seed in (0, 1, 2)) for seed in range(4)}
        comparison = paired_comparison(baseline, adapted, MOTION, 250)
        self.assertEqual(comparison["delta_percentage_points"], 50.0)
        self.assertEqual(comparison["adapted_only_success"], 2)
        self.assertEqual(comparison["baseline_only_success"], 0)
        adapted[1]["trajectory_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Motion or control frequency differs"):
            paired_comparison(baseline, adapted, MOTION, 250)
