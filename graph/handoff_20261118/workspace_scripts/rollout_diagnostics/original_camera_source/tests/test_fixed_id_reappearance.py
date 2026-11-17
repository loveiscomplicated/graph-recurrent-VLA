"""Natural reappearance and distractor scoring contracts."""

import unittest

import numpy as np

from data_collection.temporal.fixed_id_corner_sequences import CornerSequence
from evaluation.fixed_id_reappearance import event_measurement, reappearance_events, summary


class ReappearanceTests(unittest.TestCase):
    def test_reappearance_uses_current_detection_only_as_target(self):
        sequence = CornerSequence(
            episode="episode", parent="parent", vertex_id=0, start_frame=0,
            inputs=np.zeros((4, 13), dtype=np.float32),
            targets=np.array([[0, 0, 0], [9, 9, 9], [9, 9, 9], [1, 0, 0]], dtype=np.float32),
            observed=np.array([True, False, False, True]), prediction_valid=np.array([False, True, True, True]),
            last_observed_xyz_world=np.zeros((4, 3), dtype=np.float32),
            target_delta_xyz=np.zeros((4, 3), dtype=np.float32),
        )
        xyz = np.zeros((4, 3, 3), dtype=np.float32)
        xyz[3] = [[1, 0, 0], [1.01, 0, 0], [0, 0, 0]]
        source = dict(xyz=xyz, valid=np.ones((4, 3), dtype=bool),
                      kind=np.tile(np.array([0, 0, 1]), (4, 1)),
                      physical_ids=np.tile(np.array([0, -1, 8]), (4, 1)))
        events = reappearance_events(sequence, source)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["gap_frames"], 2)
        self.assertEqual(len(events[0]["inputs"]), 4)
        self.assertEqual(events[0]["competitor_ids"].tolist(), [-1])
        result = event_measurement(events[0], np.array([1.009, 0, 0]))
        self.assertTrue(result["closer_competitor"])
        self.assertAlmostEqual(result["true_detection_distance_mm"], 9, places=3)
        self.assertEqual(result["unknown_competitor_count"], 1)

    def test_threshold_score_requires_true_point_to_be_nearest(self):
        rows = [dict(gap_frames=1, true_detection_distance_mm=4., closer_competitor=False),
                dict(gap_frames=1, true_detection_distance_mm=3., closer_competitor=True)]
        result = summary(rows)["all"]
        self.assertEqual(result["thresholds_mm"]["5"]["true_within_fraction"], 1.)
        self.assertEqual(result["thresholds_mm"]["5"]["true_within_and_nearest_fraction"], .5)
        self.assertEqual(result["competitor_closer_fraction"], .5)


if __name__ == "__main__":
    unittest.main()
