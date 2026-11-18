"""Regression checks for held-out masks and diagnostic interpretation."""

import unittest

import numpy as np

from evaluation.five_mode_corner_tracker_comparison import (
    evaluation_mask,
    model_metrics,
    observation_history,
    summarize,
)


class TrackerComparisonTests(unittest.TestCase):
    def test_primary_metrics_exclude_first_observation_and_future_only_sightings(self):
        observed = np.zeros((5, 20), dtype=bool)
        observed[2, 0] = True
        observed[0, 8] = True
        observed[1, 16] = True
        truth = np.zeros((5, 20, 3), dtype=np.float32)
        episode = {
            "seed": 1,
            "observed": observed,
            "detected": truth.copy(),
            "gt": truth,
            "mode": "fixed",
            "hole_initialized": np.ones(5, dtype=bool),
        }
        episodes = [episode]
        history = observation_history(episodes)
        mask = evaluation_mask(observed, history["ever"], "after_first_observation")
        self.assertEqual(int(mask.sum()), 9)
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[2, 0])
        self.assertFalse(mask[:, 17].any())
        arrays = {
            "observed": observed,
            "gt": truth,
            "depth_mm": np.zeros(5),
            "raw_nearest_mm": np.zeros((5, 20)),
            "hole_initialized": np.ones(5, dtype=bool),
        }
        prediction = truth + 0.001
        metrics, _ = model_metrics(
            prediction, arrays, history, episodes, "after_first_observation"
        )
        prediction[~mask] = 10000
        altered, _ = model_metrics(
            prediction, arrays, history, episodes, "after_first_observation"
        )
        self.assertEqual(metrics, altered)
        self.assertEqual(metrics["body"]["all"]["count"], 9)
        self.assertEqual(metrics["body"]["peg"]["count"], 2)
        self.assertEqual(metrics["body"]["box"]["count"], 4)
        self.assertEqual(metrics["body"]["hole"]["count"], 3)
        self.assertEqual(metrics["history"]["never_seen_yet"]["all"]["count"], 0)

    def test_legacy_evaluation_population_is_preserved(self):
        observed = np.array([[False, True], [True, False]])
        seen = np.logical_or.accumulate(observed, axis=0)
        np.testing.assert_array_equal(
            evaluation_mask(observed, seen, "all_missing"), ~observed
        )

    def test_never_seen_is_not_counted_as_post_observation_occlusion(self):
        observed = np.zeros((5, 20), dtype=bool)
        observed[2, 16] = True
        detected = np.zeros((5, 20, 3), dtype=np.float32)
        truth = np.zeros_like(detected)
        detected[2, 16, 0] = 0.003
        episode = {"observed": observed, "detected": detected, "gt": truth}
        history = observation_history([episode])
        np.testing.assert_array_equal(
            history["ever"][:, 16], [False, False, True, True, True]
        )
        np.testing.assert_array_equal(history["age"][:, 16], [1, 2, 0, 1, 2])
        self.assertFalse(history["ever"][:, 17].any())
        self.assertTrue(history["seen_in_episode"][:, 16].all())
        self.assertFalse(history["seen_in_episode"][:, 17].any())
        np.testing.assert_allclose(history["first_detection_error_mm"][:, 16], 3)
        self.assertTrue(np.isnan(history["first_detection_error_mm"][:, 17]).all())

    def test_point_frame_weighted_metric_ignores_absent_diagnostics(self):
        result = summarize(np.array([1, 2, 3, np.nan, np.inf], dtype=np.float32))
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean_mm"], 2)
        self.assertAlmostEqual(result["p90_mm"], 2.8, places=5)


if __name__ == "__main__":
    unittest.main()
