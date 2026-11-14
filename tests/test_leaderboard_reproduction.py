"""Check numerical and unordered-stage comparison contracts."""

import unittest

import numpy as np

from evaluation.leaderboard_detector_audit import compare_detections
from evaluation.leaderboard_raw_replay import track_difference
from evaluation.leaderboard_reproduction import difference


class LeaderboardReproductionTests(unittest.TestCase):
    def test_difference_requires_same_shape_and_reports_nonexact(self):
        result = difference(np.array([0.0, .2]), np.array([0.0, .1]))
        self.assertFalse(result['exact'])
        self.assertEqual(result['differing_elements'], 1)
        self.assertAlmostEqual(result['max_abs'], .1)
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            difference(np.zeros(1), np.zeros(2))

    def test_detector_matching_ignores_order_and_respects_kind(self):
        xyz = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        kinds = np.array([0, 1])
        result = compare_detections(xyz, kinds, xyz[::-1], kinds[::-1])
        self.assertEqual((result['unmatched_actual'], result['unmatched_recorded']), (0, 0))
        self.assertEqual(result['distance_mm'], [0, 0])
        mismatch = compare_detections(xyz, kinds, xyz, np.array([0, 0]))
        self.assertEqual((mismatch['unmatched_actual'], mismatch['unmatched_recorded']), (1, 1))

    def test_tracker_matching_respects_detection_provenance(self):
        xyz = np.array([[0, 0, 0]], dtype=np.float32)
        actual = {'positions': xyz, 'observed': np.array([True])}
        result = track_difference(actual, np.array([0]), xyz, np.array([0]), np.array([False]))
        self.assertEqual((result['unmatched_actual'], result['unmatched_reference']), (1, 1))


if __name__ == '__main__':
    unittest.main()
