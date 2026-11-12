"""Contracts for diagnostic-only corner matching and aggregation."""

import unittest

import numpy as np

from evaluation.random_action_diagnostics import episode_summary, nearest_references


class RandomActionDiagnosticsTests(unittest.TestCase):
    def test_nearest_reference_respects_detector_kind(self):
        truth = np.zeros((24, 3), dtype=np.float32)
        truth[:8, 0] = 1.0
        truth[8:, 0] = 2.0
        ids, distances = nearest_references(np.array([[1.0, 0, 0], [1.0, 0, 0]]),
                                            np.array([0, 1]), truth)
        self.assertEqual(ids, [0, 8])
        self.assertAlmostEqual(distances[0], 0)
        self.assertAlmostEqual(distances[1], 1000)

    def test_summary_excludes_switch_frame_and_unassigned_tracks(self):
        metrics = {name: {'peg': 4.0, 'hole': 2.0} for name in
                   ('original', 'correct_observed', 'correct_predicted', 'correct_all',
                    'assigned_gt_only', 'assigned_gt_as_observed', 'complete_gt')}
        frame = dict(relative_step=1, grasp=True, errors=metrics, track_fixed_mm=[3.0, None, 4.0],
                     track_observed=[True, True, False], detection_nearest_mm=[5.0],
                     track_reference_ids=[0, None, 1], tracks=3)
        episode = {'frames': [dict(frame, relative_step=0), frame]}
        result = episode_summary(episode)
        self.assertEqual(result['frames'], 1)
        self.assertEqual(result['detected_fixed_rmse_mm'], 3)
        self.assertEqual(result['predicted_fixed_rmse_mm'], 4)
        self.assertEqual(result['unassigned_track_nodes'], 1)
        self.assertEqual(result['detection_nearest_rmse_mm'], 5)


if __name__ == '__main__':
    unittest.main()
