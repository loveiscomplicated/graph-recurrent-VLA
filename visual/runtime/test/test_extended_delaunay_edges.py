"""DE/EDE observation contracts; real evidence extraction is never invoked."""
import copy
from itertools import combinations
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import QhullError

from graph.vision_input import receive_vision_packet
from visual.runtime.extended_delaunay_edges import (
    DE_POLICY, EDE_POLICY, candidate_edges, collinear_order, directed_edges,
    build_extended_delaunay_edges, edge_metadata,
)
from visual.runtime.triangle_intra_object_edges import (
    supercover, adjacent_pairs, prepare_validation, validate_cover, triangles_to_edges,
)
from visual.runtime.test.test_triangle_intra_object_edges import fixture, oracle, build as triangle_build


def build(evidence, uv, *, extended=True, components=None, **kwargs):
    uv = np.asarray(uv, np.int64).reshape(-1, 2)
    w = evidence.frame.geometry.valid_depth.shape[1]
    source = uv[:, 1]*w+uv[:, 0]
    oid = np.zeros(len(source), np.int64)
    component = np.zeros(len(source), np.int64) if components is None else np.asarray(components, np.int64)
    return build_extended_delaunay_edges(evidence.frame.geometry.xyz.reshape(-1, 3)[source],
        oid, source, component, evidence, extended=extended, **kwargs)


def source_pairs(uv, edges):
    return {tuple(sorted((tuple(uv[a]), tuple(uv[b])))) for a, b in edges.T}


class ExtendedDelaunayContracts(unittest.TestCase):
    def test_empty_singleton_two_and_component_separation(self):
        e = fixture()
        for points, count in (([], 0), ([[1, 1]], 0), ([[1, 1], [7, 1]], 2)):
            edge, a = build(e, points)
            self.assertEqual(edge.shape, (2, count))
            self.assertEqual(build(e, points, extended=False)[0].shape, (2, 0))
        self.assertEqual(build(e, [[1, 1], [7, 1]], components=[0, 1])[0].shape[1], 0)
        self.assertTrue(any(s['status'] == 'empty' for s in a['components']))
        oid = np.array([0, 1], np.int64); src = np.array([10, 16], np.int64)
        a = candidate_edges(oid, src, np.zeros(2, np.int64), (9, 9), extended=True)
        self.assertEqual(len(a['candidates']), 0)

    def test_chain_directions_spacing_source_permutation_and_no_input_sort(self):
        e = fixture()
        cases = ([[1, 2], [2, 2], [5, 2], [8, 2]],
                 [[2, 1], [2, 2], [2, 5], [2, 8]],
                 [[1, 1], [2, 2], [5, 5], [8, 8]],
                 [[1, 8], [2, 7], [5, 4], [8, 1]])
        for points in cases:
            points = np.array(points, np.int64)
            edge, a = build(e, points)
            self.assertEqual(edge.shape[1], 6)
            self.assertEqual(set(a['origins']), {'collinear_chain'})
            expected = source_pairs(points, edge)
            for order in ([3, 2, 1, 0], [2, 0, 3, 1]):
                permuted = points[order].copy(); before = permuted.copy()
                other, audit = build(e, permuted)
                self.assertEqual(source_pairs(permuted, other), expected)
                np.testing.assert_array_equal(permuted, before)
                chain = next(s['chain_rows'] for s in audit['components'] if s['N'])
                np.testing.assert_array_equal(permuted[chain], points)

    def test_exact_cross_python_int_and_nonzero_nearly_collinear(self):
        uv = np.array([[0, 0], [2_000_000_000, 2_000_000_000],
                       [4_000_000_000, 4_000_000_000]], np.int64)
        np.testing.assert_array_equal(collinear_order(uv, np.arange(3), np.arange(3)), [0, 1, 2])
        uv[-1, 1] += 1
        self.assertIsNone(collinear_order(uv, np.arange(3), np.arange(3)))

    def test_noncollinear_matches_existing_delaunay_and_has_no_extra_chain(self):
        for uv in ([[1, 1], [7, 1], [7, 7], [1, 7]],
                   [[1, 1], [3, 1], [7, 1], [3, 7]], [[0, 0], [8, 1], [7, 1]]):
            e = fixture(); _, _, old = triangle_build(e, uv)
            de, d = build(e, uv, extended=False, stored_triangles=old['triangles'])
            ede, a = build(e, uv)
            np.testing.assert_array_equal(d['candidates'], a['candidates'])
            np.testing.assert_array_equal(de, ede)
            t0 = triangles_to_edges(old['triangles'])
            np.testing.assert_array_equal(a['candidates'], t0[:, t0[0] < t0[1]].T)
            self.assertEqual(set(a['origins']), {'delaunay_edge'})

    def test_qhull_error_omitted_vertices_and_no_automatic_fallback(self):
        uv = [[1, 1], [7, 1], [1, 7], [7, 7]]
        with patch('visual.runtime.extended_delaunay_edges.Delaunay', side_effect=QhullError('test')):
            edge, a = build(fixture(), uv)
        self.assertEqual(edge.shape[1], 0)
        self.assertEqual(a['components'][0]['status'], 'qhull_error')
        fake = SimpleNamespace(simplices=np.array([[0, 1, 2]]), coplanar=np.array([[3, 0, 2]]))
        with patch('visual.runtime.extended_delaunay_edges.Delaunay', return_value=fake):
            _, a = build(fixture(), uv)
        self.assertEqual(a['components'][0]['status'], 'omitted_vertices')
        self.assertEqual(a['components'][0]['omitted_rows'], [3])
        self.assertEqual(set(a['origins']), {'delaunay_edge'})

    def test_edge_independence_failed_bc_keeps_ab_ac(self):
        mask = np.ones((9, 9), bool); mask[4, 4] = False
        e = fixture(mask=mask); uv = [[1, 1], [7, 1], [1, 7]]
        edge, a = build(e, uv)
        self.assertEqual(set(map(tuple, edge.T)), {(0, 1), (1, 0), (0, 2), (2, 0)})
        self.assertIn('mask_outside', a['checks'][2]['reasons'])
        self.assertNotIn('mask_outside', a['checks'][0]['reasons'])
        self.assertEqual(triangle_build(e, uv)[0].shape[1], 0)

    def test_triangle_interior_failure_does_not_reject_valid_edges(self):
        mask = np.ones((9, 9), bool); mask[3, 3] = False
        e = fixture(mask=mask); uv = [[1, 1], [7, 1], [1, 7]]
        self.assertEqual(build(e, uv)[0].shape[1], 6)
        self.assertEqual(triangle_build(e, uv)[0].shape[1], 0)

    def test_fallback_rejects_mask_gap_depth_jump_with_same_stored_component(self):
        for n in (2, 3):
            points = [[1, 1], [7, 1]] if n == 2 else [[1, 1], [3, 1], [7, 1]]
            mask = np.ones((9, 9), bool); mask[1, 5] = False
            _, a = build(fixture(mask=mask), points)
            self.assertIn('mask_outside', a['checks'][-1]['reasons'])
            e = fixture(); px = e.pairs[0].pixels
            e.pairs[0].depth_jump[(px[:, 0] == 13) & (px[:, 1] == 14)] = True
            _, a = build(e, points)
            self.assertIn('depth_jump', a['checks'][-1]['reasons'])

    def test_segment_fraction_oracle_boundary_and_corner_pairs(self):
        for points in combinations(list(np.ndindex(3, 3)), 2):
            expected = {(u, v) for u in range(-1, 4) for v in range(-1, 4) if oracle(points, u, v)}
            self.assertEqual(set(map(tuple, supercover(np.array(points)))), expected)
        cover = supercover(np.array([[0, 0], [1, 1]]))
        self.assertEqual(len(cover), 4)
        self.assertEqual(len(adjacent_pairs(cover, (9, 9))), 4)
        self.assertIn('image_outside', validate_cover(supercover(np.array([[-1, 0], [1, 0]])),
                                                        prepare_validation(fixture())[0])['reasons'])

    def test_bbox_off_cover_failure_is_irrelevant(self):
        e = fixture(); px = e.pairs[0].pixels
        e.pairs[0].depth_jump[(px[:, 0] == 7*9+1) & (px[:, 1] == 7*9+2)] = True
        self.assertEqual(build(e, [[1, 1], [7, 7]])[0].shape[1], 2)

    def test_reasons_independent_and_gate_strict_boundary(self):
        for name in ('mask_outside', 'invalid_depth', 'overlap', 'pair_unsupported',
                     'pair_missing', 'delta_invalid', 'depth_jump'):
            ctx = prepare_validation(fixture())[0]
            if name == 'mask_outside':ctx['mask'][1, 2] = False
            elif name == 'invalid_depth':ctx['valid'][1, 2] = False
            elif name == 'overlap':ctx['overlap'][1, 2] = True
            elif name == 'pair_unsupported':ctx['support'][1, 2] = False
            elif name == 'pair_missing':ctx['right'][1, 1] = -1
            else:
                k = ctx['right'][1, 1]
                if name == 'delta_invalid':ctx['pairs'].signed_depth_delta_m[k] = np.nan
                else:ctx['pairs'].depth_jump[k] = True
            self.assertIn(name, validate_cover(np.array([[1, 1], [2, 1]]), ctx)['reasons'])
        # Exactly representable equality in the installed float32 arithmetic.
        za = np.float32(.001); gate = np.float32(.01+.02*za); zb = np.float32(za+gate)
        self.assertEqual(np.float32(zb-za), gate)
        z = np.full((9, 9), za, np.float32); z[:, 1:] = zb
        e = fixture(z); self.assertFalse(e.pairs[0].depth_jump[e.pairs[0].pixels[:, 0] == 0][0])
        z[:, 1:] = np.nextafter(zb, np.float32(np.inf))
        e = fixture(z); self.assertTrue(e.pairs[0].depth_jump[e.pairs[0].pixels[:, 0] == 0][0])

    def test_normal_and_gt_independent_ramp_and_input_immutable(self):
        z = np.ones((9, 9), np.float32)+np.arange(9, dtype=np.float32)[None, :]*.02
        e = fixture(z); before = copy.deepcopy(e)
        uv = [[1, 1], [7, 1], [1, 7]]
        first = build(e, uv)[0]
        for p in e.pairs.values():p.normal_angle_deg[:] = 90
        class ObservationOnly:
            def __getattr__(self, key):
                if key in ('frame', 'pairs', 'estimator'):return getattr(e, key)
                raise AssertionError('GT/feature access: '+key)
        np.testing.assert_array_equal(first, build(ObservationOnly(), uv)[0])
        np.testing.assert_array_equal(e.frame.geometry.xyz, before.frame.geometry.xyz)
        self.assertEqual(first.shape[1], 6)

    def test_subset_relations(self):
        for mask in (np.ones((9, 9), bool), np.eye(9, dtype=bool) == 0):
            mask[1, 1] = mask[7, 1] = mask[1, 7] = True
            e = fixture(mask=mask); uv = [[1, 1], [7, 1], [1, 7]]
            tmd, _, t = triangle_build(e, uv)
            de, d = build(e, uv, extended=False, stored_triangles=t['triangles'])
            ede, a = build(e, uv)
            self.assertLessEqual(set(map(tuple, tmd.T)), set(map(tuple, de.T)))
            self.assertLessEqual(set(map(tuple, de.T)), set(map(tuple, ede.T)))

    def test_invalid_sources_and_directed_unique(self):
        with self.assertRaises(ValueError):build(fixture(), [[1, 1], [1, 1]])
        e, a = build(fixture(), [[1, 1], [7, 1], [7, 7], [1, 7]])
        pairs = set(map(tuple, e.T))
        self.assertEqual(len(pairs), e.shape[1])
        self.assertTrue(all((b, a) in pairs and a != b for a, b in pairs))
        np.testing.assert_array_equal(e, directed_edges(a['candidates'][a['accepted']]))

    def test_receiver_policies_metadata_deepcopy_robot_and_immutable_features(self):
        from visual.runtime.test.test_vision_features import feature_fixture, assert_tree_equal
        from visual.runtime.vision_packet import complete_vision_packet
        source, evidence, frame = feature_fixture()
        packet = complete_vision_packet(source, evidence, frame)  # synthetic only
        component = np.zeros(len(packet['points']), np.int64)
        for extended in (False, True):
            edge, audit = build_extended_delaunay_edges(packet['camera_points'], packet['object_id'],
                packet['provenance']['source_index'], component, evidence, extended=extended)
            p = copy.deepcopy(packet)
            p.update(edge_status='supplied', observed_component_id=component,
                     intra_object_edge_index=edge, **edge_metadata(audit, extended=extended))
            original = copy.deepcopy(p); received = receive_vision_packet(p)
            assert_tree_equal(self, received.packet, original)
            for key in packet:
                if key not in ('edge_status', 'intra_object_edge_index'):
                    assert_tree_equal(self, packet[key], original[key])
            np.testing.assert_array_equal(receive_vision_packet(p,
                robot_positions=np.zeros((3, 3), np.float32)).edge_index, edge+3)
            received.packet['points'][:] = 0; assert_tree_equal(self, p, original)
            mutations = [lambda b:b.update(intra_object_edge_index=edge[:, 2:]),
                         lambda b:b.update(accepted_edge_pairs=b['accepted_edge_pairs'][:0]),
                         lambda b:b['accepted_edge_origins'].__setitem__(0, 'unknown'),
                         lambda b:b['accepted_edge_origins'].__setitem__(0, 'two_point_fallback'),
                         lambda b:b.update(edge_policy={'name': 'unknown'}),
                         lambda b:b['accepted_edge_pairs'].__setitem__((0, 0), len(p['points'])),
                         lambda b:b.update(intra_object_edge_index=np.c_[edge, edge[:, :1]]),
                         lambda b:b['observed_component_id'].__setitem__(edge[0, 0], 999)]
            for mutate in mutations:
                bad = copy.deepcopy(p); mutate(bad)
                with self.assertRaises(ValueError):receive_vision_packet(bad)

    def test_receiver_accepts_two_point_chain_empty_and_rejects_chain_shortcut(self):
        from visual.runtime.test.test_vision_features import feature_fixture
        from visual.runtime.vision_packet import complete_vision_packet
        for height, empty in ((2, False), (4, False), (2, True)):
            source, evidence, frame = feature_fixture(np.ones((height, 3), np.float32), empty=empty)
            packet = complete_vision_packet(source, evidence, frame)
            component = np.zeros(len(packet['points']), np.int64)
            edge, audit = build_extended_delaunay_edges(packet['camera_points'], packet['object_id'],
                packet['provenance']['source_index'], component, evidence)
            packet.update(edge_status='supplied', observed_component_id=component,
                          intra_object_edge_index=edge, **edge_metadata(audit, extended=True))
            receive_vision_packet(packet)
            if height == 4:
                bad = copy.deepcopy(packet)
                bad['accepted_edge_pairs'] = np.array([[0, 3]], np.int64)
                bad['accepted_edge_origins'] = ['collinear_chain']
                bad['intra_object_edge_index'] = directed_edges(bad['accepted_edge_pairs'])
                with self.assertRaises(ValueError):receive_vision_packet(bad)
            elif not empty:
                self.assertEqual(packet['accepted_edge_origins'], ['two_point_fallback'])
                bad = copy.deepcopy(packet); bad['edge_policy'] = dict(DE_POLICY)
                with self.assertRaises(ValueError):receive_vision_packet(bad)


class OptimizedExtendedDelaunayContracts(ExtendedDelaunayContracts):
    """Run every original contract with opt-in, preserving all expectations."""
    def setUp(self):
        from visual.runtime.extended_delaunay_edges import _segment_supercover_single_sat
        original = build_extended_delaunay_edges
        self.addCleanup(patch.stopall)
        patch(__name__+'.build_extended_delaunay_edges',
              side_effect=lambda *a, **kw: original(*a, **dict(kw, optimized=True))).start()
        patch(__name__+'.supercover', side_effect=_segment_supercover_single_sat).start()


class SingleSATExactContracts(unittest.TestCase):
    def test_fraction_negative_boundary_degenerate_reverse_and_order(self):
        from visual.runtime.extended_delaunay_edges import _segment_supercover_single_sat as optimized
        points = list(np.ndindex(5, 5))
        for a in points:
            for b in points:
                uv = np.asarray([a, b], np.int64)-2
                baseline = supercover(uv)
                result = optimized(uv)
                self.assertEqual(result.dtype, baseline.dtype)
                self.assertEqual(result.shape, baseline.shape)
                self.assertEqual(result.tobytes(), baseline.tobytes())
                self.assertEqual(optimized(uv[::-1]).tobytes(), result.tobytes())
                expected = [(u, v) for v in range(-3, 4) for u in range(-3, 4)
                            if oracle(uv, u, v)]
                np.testing.assert_array_equal(result, np.asarray(expected, np.int64))

    def test_guard_large_signed_unsigned_and_baseline_delegation(self):
        from visual.runtime.extended_delaunay_edges import _segment_supercover_single_sat as optimized
        for sign in (-1, 1):
            m = sign*((1 << 30)-1)
            uv = np.array([[m, m], [m+1, m]], np.int64)
            np.testing.assert_array_equal(optimized(uv), supercover(uv))
        sentinel = np.empty((0, 2), np.int64)
        for uv in (np.array([[2**40, 0], [2**40, 1]], np.int64),
                   np.array([[2**63, 0], [2**63, 1]], np.uint64),
                   np.array([[-2**63, 0], [-2**63, 1]], np.int64),
                   np.array([[0, 0], [1, 0], [0, 1]], np.int64),
                   np.zeros((2, 2), float), np.zeros((1, 2), np.int64)):
            with patch('visual.runtime.extended_delaunay_edges.supercover', return_value=sentinel) as fallback:
                self.assertIs(optimized(uv), sentinel)
                fallback.assert_called_once_with(uv)

    def test_multiple_reasons_counts_witnesses_pairs_and_order(self):
        from visual.runtime.extended_delaunay_edges import _segment_supercover_single_sat as optimized
        from visual.runtime.test.test_vision_features import assert_tree_equal
        ctx = prepare_validation(fixture())[0]
        ctx['mask'][1, 2:8] = False
        ctx['valid'][1, 2:8] = False
        ctx['overlap'][1, 2:8] = True
        ctx['support'][1, 2:4] = False
        ctx['right'][1, 4] = -1
        ctx['pairs'].signed_depth_delta_m[ctx['right'][1, 5]] = np.nan
        ctx['pairs'].depth_jump[ctx['right'][1, 6]] = True
        uv = np.array([[-1, 1], [8, 1]], np.int64)
        a, b = supercover(uv), optimized(uv)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(adjacent_pairs(a, (9, 9)).tobytes(), adjacent_pairs(b, (9, 9)).tobytes())
        first, second = validate_cover(a, ctx), validate_cover(b, ctx)
        assert_tree_equal(self, first, second)
        self.assertEqual(first['reasons'], ['image_outside', 'mask_outside', 'invalid_depth',
                         'overlap', 'pair_unsupported', 'pair_missing', 'delta_invalid', 'depth_jump'])
        self.assertEqual(first['counts']['mask_outside'], 6)
        self.assertEqual(len(first['witnesses']['mask_outside']), 4)


if __name__ == '__main__':unittest.main()
