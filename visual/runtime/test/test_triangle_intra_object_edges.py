"""Exact raster oracle and observation-only triangle contracts (no simulator)."""
import copy
from fractions import Fraction
from itertools import combinations
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import QhullError

from visual.runtime.edge_evidence import pair_evidence
from visual.runtime.geometry import OrganizedGeometry
from visual.runtime.intra_object_edges import observed_components
from visual.runtime.triangle_intra_object_edges import (
    TRIANGLE_POLICY, supercover, adjacent_pairs, prepare_validation,
    validate_cover, build_triangle_intra_object_edges, triangles_to_edges,
)


def fixture(depth=None, mask=None, overlap=None):
    z = np.ones((9, 9), np.float32) if depth is None else np.asarray(depth, np.float32)
    v, u = np.indices(z.shape)
    valid = np.isfinite(z) & (z > 0)
    xyz = np.stack((u*z/100, v*z/100, z), -1).astype(np.float32)
    xyz[~valid] = np.nan
    mask = np.ones(z.shape, bool) if mask is None else mask.copy()
    overlap = np.zeros(z.shape, bool) if overlap is None else overlap
    g = OrganizedGeometry(xyz, valid, {0: mask, 1: overlap.copy()}, overlap)
    normals = np.zeros((*z.shape, 3), np.float32); normals[..., 2] = 1
    pairs = {o: pair_evidence(g, o, normals, valid) for o in (0, 1)}
    return SimpleNamespace(estimator='pca5', frame=SimpleNamespace(geometry=g), pairs=pairs)


def build(evidence, uv, oid=0):
    uv = np.asarray(uv, np.int64).reshape(-1, 2)
    w = evidence.frame.geometry.valid_depth.shape[1]
    source = uv[:, 1]*w+uv[:, 0]
    maps, _ = observed_components(evidence)
    return build_triangle_intra_object_edges(
        evidence.frame.geometry.xyz.reshape(-1, 3)[source],
        np.full(len(source), oid, np.int64), source, maps, evidence)


def oracle(vertices, u, v):
    """Independent Fraction slab clipping + point-in-triangle oracle."""
    verts = [tuple(map(Fraction, p)) for p in vertices]
    lo = (Fraction(2*u-1, 2), Fraction(2*v-1, 2))
    hi = (lo[0]+1, lo[1]+1)
    if any(all(lo[k] <= p[k] <= hi[k] for k in (0, 1)) for p in verts):
        return True
    for a, b in zip(verts, verts[1:]+verts[:1]):
        t0, t1 = Fraction(0), Fraction(1)
        for k in (0, 1):
            d = b[k]-a[k]
            if d == 0:
                if not lo[k] <= a[k] <= hi[k]:
                    t1 = -1; break
            else:
                x, y = sorted(((lo[k]-a[k])/d, (hi[k]-a[k])/d))
                t0, t1 = max(t0, x), min(t1, y)
        if t0 <= t1:
            return True
    if len(verts) == 3:
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                cross = [(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
                         for a, b in zip(verts, verts[1:]+verts[:1])]
                if min(cross) >= 0 or max(cross) <= 0:
                    return True
    return False


class RasterOracleTests(unittest.TestCase):
    def test_closed_pixel_centres_diagonal_corner_touch(self):
        expected = {(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)}
        self.assertEqual(set(map(tuple, supercover(np.array([[0, 0], [2, 2]])))), expected)
        tri = supercover(np.array([[0, 0], [2, 0], [0, 2]]))
        self.assertEqual(len(tri), 8)
        self.assertNotIn((2, 2), set(map(tuple, tri)))

    def test_small_exhaustive_fraction_oracle(self):
        grid = list(np.ndindex(3, 3))
        cases = list(combinations(grid, 2))
        cases += [p for p in combinations(grid, 3)
                  if (p[1][0]-p[0][0])*(p[2][1]-p[0][1]) !=
                     (p[1][1]-p[0][1])*(p[2][0]-p[0][0])]
        for vertices in cases:
            expected = {(u, v) for v in range(-1, 4) for u in range(-1, 4)
                        if oracle(vertices, u, v)}
            self.assertEqual(set(map(tuple, supercover(np.array(vertices)))), expected, vertices)

    def test_pair_enumeration_only_both_touched_cells(self):
        cover = supercover(np.array([[0, 0], [3, 0], [0, 3]]))
        actual = set(map(tuple, adjacent_pairs(cover, (5, 5))))
        cells = set(map(tuple, cover))
        expected = {(v*5+u, y*5+x) for u, v in cells
                    for x, y in ((u+1, v), (u, v+1)) if (x, y) in cells}
        self.assertEqual(actual, expected)
        self.assertNotIn((18, 19), actual)  # bbox corner is not triangle support

    def test_boundary_interior_and_external_pair_are_distinct(self):
        e = fixture(); ctx = prepare_validation(e)
        tri = np.array([[1, 1], [7, 1], [1, 7]])
        interior = set(map(tuple, supercover(tri)))
        boundary = set().union(*(set(map(tuple, supercover(tri[[i, (i+1)%3]]))) for i in range(3)))
        self.assertIn((3, 3), interior-boundary)
        # Outside the triangle but inside its bbox: a jump must not reject it.
        e.pairs[0].depth_jump[:] = False
        px = e.pairs[0].pixels
        e.pairs[0].depth_jump[(px[:, 0] == 7*9+7) & (px[:, 1] == 7*9+8)] = True
        ctx = prepare_validation(e)
        self.assertEqual(validate_cover(supercover(tri), ctx[0])['reasons'], [])

    def test_independent_support_reasons_and_missing_pair(self):
        for name in ('mask_outside', 'invalid_depth', 'overlap', 'image_outside'):
            mask = np.ones((9, 9), bool); z = np.ones((9, 9), np.float32)
            overlap = np.zeros((9, 9), bool); uv = np.array([[4, 4]])
            if name == 'mask_outside': mask[4, 4] = False
            if name == 'invalid_depth': z[4, 4] = np.nan
            if name == 'overlap': overlap[4, 4] = True
            if name == 'image_outside': uv = np.array([[-1, 0]])
            result = validate_cover(uv, prepare_validation(fixture(z, mask, overlap))[0])
            self.assertIn(name, result['reasons'])
            self.assertNotIn('pair_missing', result['reasons'])
        e = fixture(); ctx = prepare_validation(e)
        ctx[0]['right'][1, 1] = -1
        result = validate_cover(np.array([[1, 1], [2, 1]]), ctx[0])
        self.assertIn('pair_missing', result['reasons'])
        self.assertNotIn('mask_outside', result['reasons'])


class TriangleContractTests(unittest.TestCase):
    def test_triangle_rejection_no_salvage_shared_edge(self):
        triangles = np.array([[0, 1, 2], [1, 2, 3]], np.int64)
        edges = triangles_to_edges(triangles[[False, True]])
        self.assertEqual(set(map(tuple, edges.T)), {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)})
        mask = np.ones((9, 9), bool); mask[3, 3] = False
        e = fixture(mask=mask)
        edges, _, audit = build(e, [[1, 1], [7, 1], [1, 7]])
        self.assertEqual(edges.shape, (2, 0))
        self.assertEqual(audit['support_pass'].tolist(), [False])
        self.assertIn('mask_outside', audit['triangle_checks'][0]['interior']['reasons'])

    def test_ramp_total_delta_large_and_middle_depth_jump(self):
        ramp = np.ones((9, 9), np.float32)+np.arange(9, dtype=np.float32)[None, :]*.02
        e = fixture(ramp)
        edges, _, a = build(e, [[1, 1], [7, 1], [1, 7]])
        self.assertEqual(edges.shape[1], 6)
        self.assertTrue(a['depth_pass'].all())
        e = fixture(); px = e.pairs[0].pixels
        e.pairs[0].depth_jump[(px[:, 0] == 1*9+4) & (px[:, 1] == 1*9+5)] = True
        edges, _, a = build(e, [[1, 1], [7, 1], [1, 7]])
        self.assertEqual(edges.shape[1], 0)
        self.assertIn('depth_jump', a['triangle_checks'][0]['edges'][0]['reasons'])

    def test_interior_only_jump(self):
        e = fixture(); px = e.pairs[0].pixels
        e.pairs[0].depth_jump[(px[:, 0] == 3*9+3) & (px[:, 1] == 3*9+4)] = True
        edges, _, a = build(e, [[1, 1], [7, 1], [1, 7]])
        self.assertEqual(edges.shape[1], 0)
        self.assertTrue(all('depth_jump' not in r['reasons'] for r in a['triangle_checks'][0]['edges']))
        self.assertIn('depth_jump', a['triangle_checks'][0]['interior']['reasons'])

    def test_normal_blind_corner_gt_independence_and_object_rule(self):
        # Continuous V-shaped depth: adjacent dz=.02 stays below the old gate.
        z = np.ones((9, 9), np.float32)+.02*np.abs(np.arange(9)-4)[None, :]
        e = fixture(z); before = build(e, [[1, 1], [7, 1], [1, 7]])
        for p in e.pairs.values():p.normal_angle_deg[:] = 90
        class ObservationOnly:
            def __getattr__(self, key):
                if key in ('frame', 'pairs', 'estimator'): return getattr(e, key)
                raise AssertionError('unexpected GT or feature access: '+key)
        after = build(ObservationOnly(), [[1, 1], [7, 1], [1, 7]])
        np.testing.assert_array_equal(before[0], after[0])
        self.assertEqual(after[0].shape[1], 6)
        e.frame.geometry.object_masks[1][:] = e.frame.geometry.object_masks[0]
        e.frame.geometry.object_masks[0][:] = False
        e.pairs[0] = pair_evidence(e.frame.geometry, 0, np.zeros((9, 9, 3)), np.zeros((9, 9), bool))
        e.pairs[1] = pair_evidence(e.frame.geometry, 1, np.zeros((9, 9, 3)), np.zeros((9, 9), bool))
        np.testing.assert_array_equal(build(e, [[1, 1], [7, 1], [1, 7]], 1)[0], before[0])

    def test_distinct_observed_components_do_not_bridge(self):
        z = np.ones((9, 9), np.float32); z[:, 4:] = 2
        e = fixture(z)
        edge, comp, a = build(e, [[0, 1], [3, 1], [1, 7], [4, 1], [8, 1], [7, 7]])
        self.assertEqual(len(a['components']), 2)
        self.assertTrue((comp[edge[0]] == comp[edge[1]]).all())

    def test_small_empty_collinear_duplicate_and_cocircular(self):
        e = fixture()
        for uv, status in (([], 'empty'), ([[1, 1]], 'insufficient'),
                           ([[1, 1], [2, 2]], 'insufficient'),
                           ([[1, 1], [2, 2], [3, 3]], 'collinear')):
            edge, comp, a = build(e, uv)
            self.assertEqual(edge.shape, (2, 0)); self.assertEqual(len(comp), len(uv))
            if uv:self.assertEqual(a['components'][0]['status'], status)
        with self.assertRaises(ValueError): build(e, [[1, 1], [1, 1], [3, 4]])
        uv = [[1, 1], [7, 1], [7, 7], [1, 7]]
        a = build(e, uv); b = build(e, uv)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(len(a[2]['triangles']), 2)
        self.assertEqual(a[2]['components'][0]['omitted_rows'], [])
        self.assertEqual(build(e, [[0, 0], [8, 1], [7, 1]])[2]['components'][0]['status'], 'ok')

    def test_error_and_omitted_vertices_distinct(self):
        e = fixture(); uv = [[1, 1], [7, 1], [1, 7], [7, 7]]
        with patch('visual.runtime.triangle_intra_object_edges.Delaunay', side_effect=QhullError('test')):
            edge, _, a = build(e, uv)
        self.assertEqual(edge.shape[1], 0)
        self.assertEqual(a['components'][0]['status'], 'qhull_error')
        fake = SimpleNamespace(simplices=np.array([[0, 1, 2]]), coplanar=np.empty((0, 3), int))
        with patch('visual.runtime.triangle_intra_object_edges.Delaunay', return_value=fake):
            _, _, a = build(e, uv)
        self.assertEqual(a['components'][0]['status'], 'omitted_vertices')
        self.assertEqual(a['components'][0]['omitted_rows'], [3])

    def test_components_and_receiver_union_deepcopy(self):
        from visual.runtime.test.test_vision_features import feature_fixture, assert_tree_equal
        from visual.runtime.vision_packet import complete_vision_packet
        from graph.vision_input import receive_vision_packet
        source, evidence, frame = feature_fixture()
        packet = complete_vision_packet(source, evidence, frame)  # synthetic only
        maps, _ = observed_components(evidence)
        edge, comp, a = build_triangle_intra_object_edges(packet['camera_points'], packet['object_id'],
            packet['provenance']['source_index'], maps, evidence)
        packet.update(edge_status='supplied', edge_policy=copy.deepcopy(TRIANGLE_POLICY),
                      observed_component_id=comp, intra_object_edge_index=edge,
                      accepted_triangles=a['triangles'][a['accepted']].copy())
        original = copy.deepcopy(packet)
        received = receive_vision_packet(packet)
        assert_tree_equal(self, received.packet, original)
        np.testing.assert_array_equal(received.edge_index, edge)
        np.testing.assert_array_equal(receive_vision_packet(packet,
            robot_positions=np.zeros((3, 3), np.float32)).edge_index, edge+3)
        received.packet['points'][:] = 0
        assert_tree_equal(self, packet, original)
        for mutate in (lambda p:p.update(intra_object_edge_index=p['intra_object_edge_index'][:, 2:]),
                       lambda p:p.update(accepted_triangles=p['accepted_triangles'][:0]),
                       lambda p:p.update(edge_policy={'name': 'unknown'}),
                       lambda p:p['accepted_triangles'].__setitem__((0, 0), len(p['points'])),
                       lambda p:p['observed_component_id'].__setitem__(edge[0, 0], 999)):
            bad = copy.deepcopy(packet); mutate(bad)
            with self.assertRaises(ValueError):receive_vision_packet(bad)


if __name__ == '__main__': unittest.main()
