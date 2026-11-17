"""Physical identity counting and face rejection for frame selection."""

import unittest
import numpy as np

from data_collection.physical_vertex_filter import peg_corner_ids, physical_vertex_conditions
from data_collection.vertex_family_labels import SIGNS


class PhysicalVertexFilterTests(unittest.TestCase):
    def test_distinct_and_same_face(self):
        family = [1, 1, 1, 0, 1]
        ids = [8, 9, 10, 4, 10]
        face = physical_vertex_conditions([0, 1, 2, 3, 3], family, ids)
        self.assertFalse(face['peg_pass'])
        self.assertTrue(face['peg_single_face'])
        self.assertTrue(face['hole_pass'])
        mixed = physical_vertex_conditions([0, 1, 2, 4, 4], family, ids)
        self.assertTrue(mixed['both_pass'])
        self.assertEqual(len(mixed['unique_peg_ids']), 4)
        self.assertEqual(len(mixed['unique_rim_ids']), 3)
        self.assertFalse(physical_vertex_conditions([0, 1, 2, 2], [0, 1, 1], [0, 8, 9])['both_pass'])

    def test_projection_and_unique_geometry(self):
        gt = {'peg_pose': np.array([0, 0, 1, 1, 0, 0, 0]),
              'peg_half_sizes': np.array([.1, .02, .02])}
        observation = {'intrinsic_cv': np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]),
                       'extrinsic_cv': np.eye(4)[:3]}
        points = SIGNS * gt['peg_half_sizes'] + gt['peg_pose'][:3]
        pixels = points[:, :2] / points[:, 2, None] * 100 + 50
        pixels[1] += 10
        ids = peg_corner_ids(points, pixels, np.zeros(8), gt, observation)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[1], -1)
        self.assertEqual(ids[7], 7)


if __name__ == '__main__':
    unittest.main()
