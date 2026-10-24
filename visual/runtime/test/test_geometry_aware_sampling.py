"""Small contract fixtures, not geometry/noise generalization performance tests."""
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.edge_evidence import PairEvidence
from visual.runtime.sampling import fps_indices
from visual.runtime.test import run_geometry_aware_sampling as g


def fixture(shape=(3, 4), candidate=None, seed=7, xyz=None):
    x = np.arange(np.prod(shape), dtype=np.int64)
    if xyz is None:
        xyz = np.column_stack((x % shape[1], x // shape[1], np.ones(len(x)))).astype(float)
    c = x if candidate is None else np.asarray(candidate, np.int64)
    a = np.arange(len(x)).reshape(shape)
    pixels = np.concatenate((np.column_stack((a[:, :-1].ravel(), a[:, 1:].ravel())),
                             np.column_stack((a[:-1].ravel(), a[1:].ravel()))))
    n = len(pixels)
    pair = PairEvidence(shape, pixels, np.ones((n,2),bool), np.zeros(n,bool),
        np.zeros(n,bool), np.zeros((n,2),bool), np.zeros(n,np.float32),
        np.zeros(n,bool), np.zeros(n,np.float32), np.full(n,-1), np.full(n,-1))
    case = dict(support_flat=x, support_xyz=xyz, candidate_flat=c, candidate_xyz=xyz[c],
                seed=seed, first=int(c[np.random.default_rng(seed).integers(len(c))]) if len(c) else -1,
                budget=len(x))
    return case, pair


class GeometrySamplingContracts(unittest.TestCase):
    def test_connected_and_inclusive_normal_threshold(self):
        c, p = fixture(); p.normal_angle_deg[:] = 20.
        labels, gate = g.patch_labels(c['support_flat'], p)
        self.assertEqual(len(np.unique(labels)), 1); self.assertTrue(gate.all())

    def test_depth_normal_nonfinite_and_overlap_veto(self):
        c, p = fixture(shape=(1,5))
        p.depth_jump[0] = True; p.normal_angle_deg[1] = 20.001
        p.normal_angle_deg[2] = np.nan; p.overlap[3,0] = True
        labels, gate = g.patch_labels(c['support_flat'], p)
        self.assertFalse(gate.any()); np.testing.assert_array_equal(labels, np.arange(5))

    def test_support_invalid_endpoint_and_no_wrap(self):
        c,p=fixture(); support=c['support_flat'][c['support_flat'] != 1]
        p.supported[:] = np.isin(p.pixels,support)
        labels,gate=g.patch_labels(support,p)
        self.assertFalse(gate[np.any(p.pixels == 1, axis=1)].any())
        p.pixels[0] = [3,4]
        with self.assertRaises(ValueError): g.patch_labels(support,p)

    def test_support_mismatch_fails(self):
        c,p=fixture();p.supported[0,0]=False
        with self.assertRaises(AssertionError):g.prepare(c,p)

    def test_surfaces_separate_and_candidate_free_X_retained(self):
        c,p=fixture(candidate=[0,1,2,3])
        p.depth_jump[:] = (p.pixels[:,0] < 4) & (p.pixels[:,1] >= 4)
        a=g.prepare(c,p)
        self.assertEqual(a['diagnostics']['X_patches'],2)
        self.assertEqual(a['diagnostics']['candidate_patches'],1)
        self.assertEqual(len(a['support_patch']),12)
        np.testing.assert_array_equal(a['candidate_patch'],np.zeros(4))

    def test_span_proportional_and_stable_ties(self):
        q,left=g.allocate_span(np.array([20,20,20]),[1,2,4],7)
        np.testing.assert_array_equal(q,[1,2,4]);self.assertEqual(left,0)
        q,left=g.allocate_span(np.array([8,8,8]),[1,1,1],2)
        np.testing.assert_array_equal(q,[1,1,0])

    def test_capacity_saturation_zero_and_exhaustion(self):
        q,left=g.allocate_span(np.array([1,10,10]),[100,1,0],8)
        np.testing.assert_array_equal(q,[1,7,0]);self.assertEqual(left,0)
        q,left=g.allocate_span(np.array([1,2,10]),[100,1,0],8)
        np.testing.assert_array_equal(q,[1,2,0]);self.assertEqual(left,5)

    def test_singletons_zero_span_anchor_and_input_only_fallback(self):
        c,p=fixture(shape=(2,20));p.normal_angle_deg[:]=np.nan
        a=g.prepare(c,p);s=g.select_geometry(c,a)
        self.assertEqual(a['diagnostics']['zero_span_candidates'],40)
        self.assertEqual(a['diagnostics']['quota_sum'],1)
        self.assertEqual(a['fallback_slots'],31)
        self.assertEqual(s['fallback_mode'],'generated_current_C_global_order')
        metric=g.prior.metric_coordinates(c['candidate_xyz'],0.)
        expected=fps_indices(metric,32,seed=c['seed'])
        np.testing.assert_array_equal(s['P'],c['candidate_flat'][expected])
        self.assertEqual(s['P'][0],c['first']);self.assertEqual(len(s['P']),32)
        np.testing.assert_array_equal(s['xyz'],c['support_xyz'][s['F']])

    def test_coincident_xyz_is_zero_span_and_unique_source(self):
        c,p=fixture(xyz=np.ones((12,3)))
        a=g.prepare(c,p);s=g.select_geometry(c,a)
        self.assertEqual(a['patches'][0]['weight'],0.)
        self.assertEqual(a['fallback_slots'],11)
        self.assertEqual(len(np.unique(s['F'])),12)

    def test_saved_prefix_optional_and_short_prefix_generates(self):
        c,p=fixture(shape=(2,20));p.normal_angle_deg[:]=np.nan
        a=g.prepare(c,p);s=g.select_geometry(c,a)
        with patch.object(g,'fps_indices',side_effect=AssertionError('no global replay')):
            rows,mode=g.fallback_rows(c['candidate_flat'],c['candidate_xyz'],c['seed'],c['first'],
                np.array([np.searchsorted(c['candidate_flat'],c['first'])]),31,s['P'])
        np.testing.assert_array_equal(c['candidate_flat'][rows],s['P'][1:])
        self.assertEqual(mode,'verified_saved_global_prefix')
        t=g.select_geometry(c,a,s['P'][:1]);np.testing.assert_array_equal(t['P'],s['P'])

    def test_determinism_alignment_and_GT_independence(self):
        c,p=fixture(shape=(3,16));a=g.prepare(c,p)
        one=g.select_geometry(g.selection_input(c),a)
        c.update(GT_annotation=np.array([99]),pose='changed',future_frame='poison',evaluation_score=-999)
        two=g.select_geometry(g.selection_input(c),g.prepare(c,p))
        for key in ('P','S','F','xyz'):np.testing.assert_array_equal(one[key],two[key])
        self.assertEqual(len(np.intersect1d(one['P'],one['S'])),0)
        np.testing.assert_array_equal(one['F'],np.r_[one['P'],one['S']])
        np.testing.assert_array_equal(c['support_xyz'][np.searchsorted(c['support_flat'],one['P'])],
                                      c['candidate_xyz'][one['priority_rows']])

    def test_C_outside_X_and_xyz_mismatch_stop(self):
        c,p=fixture();c['candidate_xyz']=c['candidate_xyz'].copy();c['candidate_xyz'][0,2]+=1
        with self.assertRaises(AssertionError):g.prepare(c,p)
        c,p=fixture();c['candidate_flat']=np.array([99])
        with self.assertRaises(ValueError):g.prepare(c,p)

    def test_empty_C_retains_X_for_surface(self):
        c,p=fixture(candidate=[]);a=g.prepare(c,p);s=g.select_geometry(c,a)
        self.assertEqual(len(s['P']),0);self.assertEqual(len(s['S']),12)
        self.assertEqual(a['diagnostics']['candidate_patches'],0)


if __name__=='__main__':unittest.main()
