"""Contracts protecting physical geometry, replay identity and graph rebuilding."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from evaluation.gnn.box_hole_geometry import (
    SIGNS, physical_vertices, rigid_registration, transform_points, verify_replay,
)
from evaluation.gnn.box_hole_diagnostics import graph_from_arrays, oracle_arrays


class BoxHoleGeometryTests(unittest.TestCase):
    def test_outer_geometry_does_not_determine_random_hole_offset(self):
        box = np.array([.01, .3, .1, 1, 0, 0, 0])
        first = box.copy()
        second = box.copy()
        first[:3] += [0, .02, .01]
        second[:3] += [0, -.02, -.01]
        outer_a, rim_a = physical_vertices(box, first, .1, .023)
        outer_b, rim_b = physical_vertices(box, second, .1, .023)
        np.testing.assert_array_equal(outer_a, outer_b)
        self.assertGreater(np.linalg.norm(rim_a.mean(0) - rim_b.mean(0)), .04)

    def test_planar_labelled_fit_and_hole_midpoint(self):
        quaternion = Rotation.from_euler('xyz', [.1, -.2, .8]).as_quat()
        pose = np.r_[[.01, .3, .1], quaternion[[3, 0, 1, 2]]]
        local = SIGNS * [.1, .025, .025]
        observed = transform_points(local, pose)
        rotation, translation = rigid_registration(local[:4], observed[:4])
        np.testing.assert_allclose(translation, pose[:3], atol=1e-12)
        np.testing.assert_allclose(rotation, Rotation.from_quat(quaternion).as_matrix(), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(observed[:4].mean(0) - pose[:3]), .1)

    def test_collinear_or_duplicated_points_are_rejected(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
        with self.assertRaisesRegex(ValueError, 'Collinear'):
            rigid_registration(points, points)

    def test_replay_cannot_silently_accept_wrong_seed(self):
        state = dict(hole_pose=[0, 0, 0, 1, 0, 0, 0], peg_half_sizes=[.1, .02, .02], hole_radius=.023)
        target = np.zeros((2, 21))
        target[:, 13] = 1
        target[:, 17:20] = state['peg_half_sizes']
        target[:, 20] = state['hole_radius']
        verify_replay(state, target, np.zeros((2, 3)), 1e-6)
        target[1, 10] = .01
        with self.assertRaisesRegex(ValueError, 'does not match'):
            verify_replay(state, target, np.zeros((2, 3)), 1e-6)

    def test_edges_respect_padding_and_current_coordinates(self):
        xyz = np.array([[[1, 2, 3], [4, 6, 3], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        labels = np.array([[0, 1, -1], [-1, -1, -1]])
        graph = graph_from_arrays(xyz, labels, np.zeros((2, 3)))
        self.assertEqual(graph.edge_index.shape[1], 2)
        np.testing.assert_array_equal(graph.distance.numpy(), [[5], [5]])
        self.assertFalse(graph.semantic_valid.any())
        self.assertFalse(graph.confidence_valid.any())
        self.assertEqual(graph.node_valid.sum(), 2)

    def test_gt_target_oracle_preserves_original_peg_but_changes_visibility(self):
        geometry = dict(sample_ids=np.array(['ep/frame.npz']), xyz_world=np.array([[[.7, .8, .9], [9, 9, 9]]]),
                        object_type=np.array([[0, 1]]))
        states = {'ep': dict(box_pose=[0, 0, .1, 1, 0, 0, 0], hole_pose=[0, .02, .11, 1, 0, 0, 0],
                             peg_half_sizes=[.1, .02, .02], hole_radius=.023)}
        xyz, labels = oracle_arrays(geometry, states, 'gt_target_original_peg')
        self.assertEqual(xyz.shape, (1, 17, 3))
        np.testing.assert_allclose(xyz[0, 0], geometry['xyz_world'][0, 0])
        self.assertEqual((labels == 1).sum(), 16)
        self.assertEqual((labels == 2).sum(), 0)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            oracle_arrays(geometry, states, 'nearest_vertex_replacement')


if __name__ == '__main__':
    unittest.main()
