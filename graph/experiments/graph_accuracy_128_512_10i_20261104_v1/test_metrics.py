"""Controls for purity labels, common-reference credit and paired denominators."""
import unittest
import numpy as np
from .evaluate import actor_labels
from .common import world_points
from graph.task10i_stage4a2.evaluation_metrics import correct_source_coverage, hierarchical_paired


class MetricsControls(unittest.TestCase):
    def test_record_coordinates_use_current_tcp_translation_and_rotation(self):
        q = np.sqrt(.5)
        tcp = [1., 2., 3., q, 0., 0., q]
        r = dict(xyz=[1., 0., 0.], coordinate_frame='ee', reference_ee_pose=tcp)
        np.testing.assert_allclose(world_points([r], tcp), [[1., 3., 3.]], atol=1e-7)
        with self.assertRaises(ValueError):
            world_points([r], [0., 0., 0., 1., 0., 0., 0.])

    def test_representative_identity_uses_native_pixel_and_camera(self):
        ids = dict(peg=[18], box=[19], robot=[3])
        a = np.array([[18, 19], [3, 0]])
        labels = actor_labels([a, a.T], [ids, ids], 0, [0, 0, 1, 0], [0, 1, 1, 3])
        self.assertEqual(labels.tolist(), ['correct', 'wrong_object', 'robot', 'background'])

    def test_false_point_near_surface_gets_no_coverage_credit(self):
        p = np.zeros((1, 3))
        self.assertEqual(correct_source_coverage(p, [False], p), 0.)
        self.assertEqual(correct_source_coverage(p, [True], p), 1.)
        self.assertIsNone(correct_source_coverage(p, [True], np.empty((0,3))))

    def test_missing_precision_not_zero_or_success(self):
        rows = [dict(parent_id='p', episode='e', timestep=0, object_id=0, condition=c,
                     precision=v, missing=m) for c, v, m in [('B0', None, 1.), ('N512', 1., 0.)]]
        self.assertEqual(hierarchical_paired(rows, 'precision', group_keys=('object_id',), candidate='N512'), [])
        result = hierarchical_paired(rows, 'missing', group_keys=('object_id',), candidate='N512')
        self.assertEqual(result[0]['delta'], -1.)


if __name__ == '__main__':
    unittest.main()
