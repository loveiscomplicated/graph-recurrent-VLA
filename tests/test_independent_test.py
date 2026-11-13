"""Contracts for preserving unscored frames and paired expert prefixes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.independent_test import summarize, verify_expert_random_prefix


def model_row(value=None):
    return {'available': value is not None,
            'errors': {'peg_position_error_mm': value, 'hole_position_error_mm': value} if value is not None else None}


class IndependentTestContracts(unittest.TestCase):
    def test_missing_model_output_is_counted_not_hidden(self):
        valid = {'after_lift': 1, 'models': {str(seed): model_row(float(seed+1)) for seed in range(3)}}
        missing = {'after_lift': 2, 'models': {'0': model_row(1), '1': model_row(None), '2': model_row(3)}}
        result = summarize([{'mode': 'expert', 'frames': [valid, missing]}], [0, 1, 2])
        summary = result['expert']['post_lift']
        self.assertEqual((summary['frames'], summary['scored_frames'], summary['missing_any_model']), (2, 1, 1))
        self.assertEqual(summary['metrics']['peg_position_error_mm']['mean'], 2)

    def test_prefix_checker_rejects_different_actions(self):
        fields = ('observation/sensor_data/base_camera/rgb', 'observation/sensor_data/base_camera/depth',
                  'observation/sensor_param/base_camera/intrinsic_cv',
                  'observation/sensor_param/base_camera/extrinsic_cv', 'observation/agent/qpos',
                  'observation/extra/tcp_pose', 'ground_truth/peg_pose', 'ground_truth/hole_pose')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ('expert', 'random_after_lift'):
                (root/mode).mkdir()
                for index in (0, 1):
                    values = {key: np.array([0]) for key in fields}
                    values['action'] = np.array([index])
                    np.savez(root/mode/f'frame_{index:06d}.npz', **values)
            manifest = {'episodes': [dict(seed=1, mode=mode, path=mode, switch_frame=1, observations=2)
                                     for mode in ('expert', 'random_after_lift')]}
            self.assertEqual(verify_expert_random_prefix(manifest, root, [1]), 2)
            values['action'] = np.array([99])
            np.savez(root/'random_after_lift'/'frame_000000.npz', **values)
            with self.assertRaisesRegex(ValueError, 'prefix action differs'):
                verify_expert_random_prefix(manifest, root, [1])


if __name__ == '__main__':
    unittest.main()
