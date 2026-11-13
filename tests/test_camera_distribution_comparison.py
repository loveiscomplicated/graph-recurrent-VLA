"""Check per-seed visibility pairing and trajectory-phase aggregation."""

import unittest

from evaluation.camera_distribution_comparison import summarize


def row(seed, frame, relative_step, visible):
    return dict(seed=seed, action_seed=73, frame=frame, relative_step=relative_step,
                visible_fraction=visible, robot_blocked_fraction=1-visible)


class CameraDistributionComparisonTests(unittest.TestCase):
    def test_paired_seed_improvement_and_phases(self):
        baseline = {'frames': [row(1, 0, -1, .2), row(1, 1, 0, .4),
                               row(2, 0, -1, .8), row(2, 1, 0, .6)]}
        candidate = {'name': 'candidate', 'frames': [row(1, 0, -1, .6), row(1, 1, 0, .6),
                                                      row(2, 0, -1, .6), row(2, 1, 0, .6)]}
        result = summarize(candidate, baseline, [1, 2])
        self.assertEqual((result['improved_seeds'], result['worse_seeds']), (1, 1))
        self.assertAlmostEqual(result['mean_visibility'], .6)
        self.assertEqual(result['phases']['initial']['samples'], 2)
        self.assertEqual(result['phases']['post_lift_random']['samples'], 2)

    def test_rejects_unpaired_frames(self):
        baseline = {'frames': [row(1, 0, 0, .2)]}
        candidate = {'name': 'candidate', 'frames': [row(1, 1, 0, .2)]}
        with self.assertRaisesRegex(ValueError, 'Sampled states differ'):
            summarize(candidate, baseline, [1])


if __name__ == '__main__':
    unittest.main()
