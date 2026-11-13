"""Check camera ranking against the predeclared robust selection rule."""

import unittest

from evaluation.camera_visibility_search import rank_candidates


def candidate(name, visibility, robot, widths):
    return {'name': name, 'frames': [dict(seed=seed, visible_fraction=value,
                                         robot_blocked_fraction=robot,
                                         aperture_bbox_px=[width, 10])
                                     for seed, value, width in zip((0, 1, 2), visibility, widths)]}


class CameraVisibilitySearchTests(unittest.TestCase):
    def test_rank_favors_worst_seed_then_projected_width(self):
        records = [candidate('fragile', [1, 1, .4], 0, [20,20,20]),
                   candidate('narrow', [.8,.8,.8], 0, [4,4,4]),
                   candidate('wide', [.8,.8,.8], 0, [10,10,10])]
        self.assertEqual([row['name'] for row in rank_candidates(records, [0,1,2])],
                         ['wide','narrow','fragile'])

    def test_mean_rule_rewards_overall_visibility(self):
        records = [candidate('consistent', [.7, .7, .7], 0, [10, 10, 10]),
                   candidate('higher_mean', [1, 1, .4], 0, [10, 10, 10])]
        ranked = rank_candidates(records, [0, 1, 2], 'mean_then_p10')
        self.assertEqual(ranked[0]['name'], 'higher_mean')


if __name__ == '__main__':
    unittest.main()
