"""Contracts for diagnostic-only corner matching and aggregation."""

import unittest

import numpy as np

from evaluation.random_action_diagnostics import episode_summary, make_sensor_configs, nearest_references
from evaluation.compare_camera_diagnostics import validate_pair


class RandomActionDiagnosticsTests(unittest.TestCase):
    def test_paired_comparison_rejects_different_actions(self):
        frame = {'frame': 0, 'relative_step': 0, 'grasp': True, 'peg_gt': [0, 0, 0],
                 'hole_gt': [1, 0, 0], 'action': [0, 0]}
        episode = {'environment_seed': 0, 'action_seed': 73, 'switch_frame': 0, 'frames': [frame]}
        self.assertEqual(validate_pair(episode, episode), 0)
        changed = dict(episode, frames=[dict(frame, action=[0, 1])])
        with self.assertRaisesRegex(ValueError, 'action mismatch'):
            validate_pair(episode, changed)

    def test_camera_position_is_optional_and_validated(self):
        self.assertNotIn('pose', make_sensor_configs(None)['base_camera'])
        sensors = make_sensor_configs({'eye': [0.65, -0.30, 0.65], 'target': [0.065, 0.297, 0.11]})
        np.testing.assert_allclose(sensors['base_camera']['pose'].p[0], [0.65, -0.30, 0.65])
        with self.assertRaisesRegex(ValueError, 'camera_view.eye'):
            make_sensor_configs({'eye': [0, 0], 'target': [0, 0, 1]})
        with self.assertRaisesRegex(ValueError, 'must differ'):
            make_sensor_configs({'eye': [0, 0, 0], 'target': [0, 0, 0]})

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
