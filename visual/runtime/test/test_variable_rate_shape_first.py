"""03-J observation/selection contracts; no performance threshold is a PASS rule."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.geometry import OrganizedGeometry
from visual.runtime.test import run_variable_rate_shape_first as task


def fixture(n=400, pca=None):
    shape = (20, 20)
    x = np.arange(n, dtype=np.int64)
    xyz = np.column_stack((x*.01, (x % 7)*.001, np.ones(n))).astype(np.float32)
    raw = np.zeros(shape, bool); raw.ravel()[x] = True
    valid = np.ones(shape, bool); overlap = np.zeros(shape, bool)
    pca = x.copy() if pca is None else np.asarray(pca, np.int64)
    prep = task.prepare_object(x, xyz, pca, raw, valid, overlap, np.diag([10000., 10000., 1.]))
    return prep


def feature_case(prep):
    flat = prep['X'][::3]
    normal = np.tile(np.array([0., 0., 1.], np.float32), (len(flat), 1))
    if len(normal): normal[0] = np.nan
    return dict(saved_flat=flat.copy(), saved_features=dict(normal=normal, flag=np.ones(len(flat), bool)))


def packet_fixture(n=40):
    prep = {0: fixture(n), 1: fixture(n)}
    selections = {o: task.select_object(p, .5, 0, 3) for o,p in prep.items()}
    cases = {o: feature_case(p) for o,p in prep.items()}
    packet, _ = task.make_packet(0, prep, selections, cases)
    return packet, prep, selections, cases


class ContourTests(unittest.TestCase):
    def contours(self, raw, observed=None, invalid=None, overlap=None):
        valid = np.ones_like(raw) if invalid is None else ~invalid
        overlap = np.zeros_like(raw) if overlap is None else overlap
        observed = raw & valid & ~overlap if observed is None else observed
        return task.outer_contours(raw, observed, valid, overlap)

    def test_raw_external_ignores_internal_mask_and_depth_holes(self):
        raw = np.zeros((15, 15), bool); raw[2:13, 2:13] = True; raw[6:9, 6:9] = False
        invalid = np.zeros_like(raw); invalid[4:6, 4:6] = True
        c = self.contours(raw, invalid=invalid)
        self.assertEqual(len(c['components']), 1)
        v,u = np.divmod(c['raw_unique'], 15)
        self.assertTrue(((v == 2)|(v == 12)|(u == 2)|(u == 12)).all())
        np.testing.assert_array_equal(np.sort(c['C']), np.sort(c['raw_unique']))

    def test_canonical_start_direction_and_repeated_minimum(self):
        walk = np.array([9, 5, 1, 2, 3, 5])
        expected = task.canonical_walk(walk)
        for w in (walk, walk[::-1]):
            for k in range(len(w)):
                np.testing.assert_array_equal(task.canonical_walk(np.roll(w, k)), expected)
        self.assertEqual(expected[0], 1)
        self.assertEqual(len(expected)-len(np.unique(expected)), 1)
        np.testing.assert_array_equal(task.canonical_walk(np.array([1,2,3,1])), [1,2,3])

    def test_component_sort_and_global_thinning_no_quota(self):
        raw = np.zeros((8, 12), bool); raw[1:4, 1:4] = True; raw[6, 10] = True
        c = self.contours(raw)
        self.assertEqual(len(c['components']), 2)
        self.assertLess(c['components'][0]['walk'][0], c['components'][1]['walk'][0])
        np.testing.assert_array_equal(c['C'][task.thin_indices(len(c['C']), .5)], c['C'][::2])
        # Last one-pixel component has no separate minimum reservation.
        kept = c['C'][task.thin_indices(len(c['C']), .75)]
        np.testing.assert_array_equal(kept, c['C'][[0,1,2,4,5,6,8]])

    def test_one_pixel_line_empty_and_image_boundary(self):
        for coords in ([], [(0,0)], [(2,2),(2,3),(2,4),(2,5)]):
            raw = np.zeros((8,8), bool)
            for v,u in coords: raw[v,u] = True
            c = self.contours(raw)
            self.assertEqual(len(c['C']), len(coords))
            self.assertEqual(len(c['C']), len(np.unique(c['C'])))
            if len(coords)==1: self.assertTrue(c['components'][0]['image_boundary'][0])
            if len(coords)>1: self.assertGreater(c['components'][0]['repeated_source_visits'], 0)

    def test_invalid_overlap_exclusions_and_segment_markers(self):
        raw = np.zeros((10,10), bool); raw[2:8,2:8] = True
        invalid = np.zeros_like(raw); invalid[2,4] = True
        overlap = np.zeros_like(raw); overlap[7,5] = True; overlap[2,4] = True
        c = self.contours(raw, invalid=invalid, overlap=overlap)
        self.assertNotIn(24, c['C']); self.assertNotIn(75, c['C'])
        comp = c['components'][0]
        self.assertEqual((~comp['observed']).sum(), 2)
        self.assertTrue((comp['observed_segment_id'][~comp['observed']] == -1).all())
        self.assertEqual(len(task.observed_runs(comp['observed'])), 2)

    def test_occlusion_shaped_outer_kept_without_GT(self):
        raw = np.zeros((12,12), bool); raw[2:10,2:10] = True; raw[2:7,5:8] = False
        c = self.contours(raw)
        self.assertTrue(any((s//12==7 and 5<=s%12<=7) for s in c['C']))
        self.assertEqual(c['canonical_component_count'], 1)

    def test_no_interpolation_over_missing_or_components(self):
        prep = fixture(20)
        contours = dict(components=[dict(component=0,walk=np.array([0,1,2,3,4]),observed=np.array([1,1,0,1,1],bool)),
            dict(component=1,walk=np.array([10,11]),observed=np.ones(2,bool))])
        result, data = task.gap_metrics(contours, [], prep)
        self.assertEqual(result['max_missing_walk_visits'], 4)  # valid run wraps original closure
        self.assertEqual(result['gap_count'], 2)
        # The 2→3 exclusion is not traversed. The real 4→0 contour closure is.
        self.assertFalse(any(2 in v for k,v in data.items() if k.endswith('/source')))
        lengths = data['internal_length_mm']
        expected = sum(np.linalg.norm(prep['xyz'][a].astype(float)-prep['xyz'][b].astype(float))*1000 for a,b in ((3,4),(4,0),(0,1)))
        self.assertAlmostEqual(lengths.max(), expected, places=5)

    def test_closed_loop_all_missing_and_closed_boundary_merge(self):
        prep=fixture(4)
        contours=dict(components=[dict(component=0,walk=np.arange(4),observed=np.ones(4,bool))])
        result,_=task.gap_metrics(contours,[],prep)
        self.assertTrue(result['gaps'][0]['closed_missing_loop'])
        result,_=task.gap_metrics(contours,[1,2],prep)
        self.assertEqual(result['max_missing_walk_visits'],2)
        self.assertFalse(result['gaps'][0]['closed_missing_loop'])


class ThinningTests(unittest.TestCase):
    def test_fixed_rational_sequences_odd_empty(self):
        for m in (0,1,3,8,11):
            for r in (1., .75, .5, .25, .3, .6):
                f=task.Fraction(str(r)); expected=np.array([(k*f.denominator)//f.numerator for k in range((m*f.numerator+f.denominator-1)//f.denominator)],np.int64)
                np.testing.assert_array_equal(task.thin_indices(m,r),expected)
        np.testing.assert_array_equal(task.thin_indices(9,.75),[0,1,2,4,5,6,8])
        np.testing.assert_array_equal(task.thin_indices(9,.5,1),[1,3,5,7])
        self.assertEqual(len(task.thin_indices(9,.5,0)),5)

    def test_bad_ratios_and_phase(self):
        for r in (0,-1,1.1,np.inf,np.nan):
            with self.assertRaises(ValueError):task.thin_indices(10,r)
        for r,p in ((.75,1),(.5,2),(.5,-1)):
            with self.assertRaises(ValueError):task.thin_indices(10,r,p)
        with self.assertRaises(ValueError):task.thin_indices(-1,.5)


class SelectionTests(unittest.TestCase):
    def test_optional_fps_backend_receives_shape_prefix(self):
        prep = fixture()
        reference = task.select_object(prep, .5, 0, 17)
        calls = []
        def backend(points, count, **kwargs):
            calls.append((count, kwargs))
            return task.fps_indices(points, count, **kwargs)
        selected = task.select_object(prep, .5, 0, 17, fps_fn=backend)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(prep['X'][calls[0][1]['initial']], selected['P'])
        np.testing.assert_array_equal(selected['F'], reference['F'])

    def test_no_total_or_structural_cap_original_XYZ(self):
        prep=fixture(); s=task.select_object(prep,1.,0,97)
        self.assertGreater(len(s['P']),32); self.assertGreater(len(s['F']),256)
        np.testing.assert_array_equal(s['xyz'],prep['xyz'][task.source_rows(prep['X'],s['F'])])
        self.assertEqual(len(np.unique(s['F'])),len(s['F']))
        self.assertEqual(len(s['F']),len(s['P'])+len(s['S']))

    def test_surface_initial_and_count_existing_API(self):
        prep=fixture()
        with patch.object(task,'fps_indices',wraps=task.fps_indices) as fps:
            s=task.select_object(prep,.25,0,17)
        self.assertEqual(fps.call_count,1)
        args,kwargs=fps.call_args
        self.assertIs(args[0],prep['xyz']);self.assertEqual(args[1],len(s['P'])+17)
        np.testing.assert_array_equal(prep['X'][kwargs['initial']],s['P'])
        self.assertEqual(len(s['S']),17)

    def test_empty_priority_first_is_surface_and_no_rng(self):
        prep=fixture(40,pca=[]);prep['R']['sil']=np.empty(0,np.int64)
        with patch('numpy.random.default_rng',side_effect=AssertionError('RNG forbidden')):
            a=task.select_object(prep,.5,0,5);b=task.select_object(prep,.5,0,5)
        self.assertEqual(len(a['P']),0);self.assertEqual(len(a['S']),5);self.assertEqual(a['S'][0],prep['X'][0])
        np.testing.assert_array_equal(a['F'],b['F'])

    def test_empty_shortfall_zero_surface_no_padding(self):
        prep=fixture(0);s=task.select_object(prep,.5,0,97)
        self.assertEqual(len(s['F']),0);self.assertEqual(s['shortfall'],97)
        prep=fixture(4);s=task.select_object(prep,1.,0,97)
        self.assertEqual(len(s['F']),4);self.assertEqual(s['shortfall'],97)
        s=task.select_object(prep,.5,0,0);self.assertEqual(len(s['S']),0)

    def test_source_union_order_and_duplicates(self):
        prep=fixture(40);prep['R']={'pca':np.array([1,4,8]),'sil':np.array([8,2,4,5])}
        s=task.select_object(prep,1.,0,2)
        np.testing.assert_array_equal(s['P'],[1,4,8,2,5])
        self.assertFalse(np.intersect1d(s['P'],s['S']).size)

    def test_independent_same_voxel_different_representatives_survive(self):
        x=np.arange(4,dtype=np.int64);xyz=np.array([[.0001,0,1],[.0002,0,1],[.0003,0,1],[.0004,0,1]],np.float32)
        raw=np.ones((2,2),bool)
        prep=task.prepare_object(x,xyz,np.array([0],np.int64),raw,raw,np.zeros_like(raw),np.eye(3))
        self.assertEqual(len(prep['R']['pca']),1);self.assertEqual(len(prep['R']['sil']),1)
        self.assertNotEqual(prep['R']['pca'][0],prep['R']['sil'][0])
        s=task.select_object(prep,1.,0,0);self.assertEqual(len(s['P']),2)
        geometry=OrganizedGeometry(xyz.reshape(2,2,3),raw,{0:raw,1:np.zeros_like(raw)},np.zeros_like(raw))
        row,_,_=task.evaluate_object(prep,s,geometry,0)
        self.assertEqual(row['counts']['common_occupied_voxels'],1)
        self.assertEqual(row['counts']['structural_multiple_source_voxels'],1)
        self.assertEqual(row['counts']['structural_points_in_multiple_source_voxels'],2)

    def test_sil_representatives_restore_walk_order_and_bounds(self):
        prep=fixture()
        sil=prep['C']['sil'];r=prep['R']['sil']
        np.testing.assert_array_equal(r,sil[np.isin(sil,r)])
        for g in prep['groups'].values():
            self.assertLessEqual(g['same_cell_distance_m'].max(),prep['scale']['epsilon_m']+1e-10)

    def test_input_mutation_and_determinism(self):
        prep=fixture();before=task.packet_digest(prep)
        a=task.select_object(prep,.75,0,9);b=task.select_object(prep,.75,0,9)
        for k in ('P','S','F','xyz'):np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(before,task.packet_digest(prep))

    def test_bad_candidate_order_membership(self):
        for pca in ([1,1],[3,1],[500]):
            with self.assertRaises(ValueError):fixture(40,pca)

    def test_coincident_different_sources_allowed(self):
        prep=fixture(20);prep['xyz'][:]=[0,0,1];prep['R']={'pca':np.array([2]),'sil':np.array([3])}
        s=task.select_object(prep,1.,0,5)
        self.assertEqual(len(s['F']),7);self.assertEqual(len(np.unique(s['F'])),7)
        self.assertEqual(s['S'][0],0) # first-row FPS tie


class ProvenanceAndIsolationTests(unittest.TestCase):
    def test_both_evidence_independent_selection_and_surface(self):
        packet,prep,selected,cases=packet_fixture(40)
        # Source0 belongs to both C; remove its PCA role but retain its silhouette role.
        for oid in (0,1):
            selected[oid]['T']['pca']=selected[oid]['T']['pca'][selected[oid]['T']['pca']!=0]
            self.assertIn(0,selected[oid]['T']['sil'])
        packet,_=task.make_packet(0,prep,selected,cases)
        p=packet['provenance'];row=np.flatnonzero((p['source_index']==0)&(p['object_id']==0))[0]
        self.assertTrue(p['both_evidence'][row]);np.testing.assert_array_equal(p['selected_by_structural_source'][row],[False,True])
        surface=p['surface_selected'];self.assertTrue(p['candidate_evidence'][surface,0].all())
        self.assertFalse(p['selected_by_structural_source'][surface].any())

    def test_features_exact_available_only_no_fill(self):
        packet,prep,selected,cases=packet_fixture(40)
        p=packet['provenance'];self.assertTrue((~p['feature_available']).any())
        self.assertTrue(np.isnan(packet['geometric_features'][~p['feature_available']]).all())
        for name,compact in packet['compact_features'].items():
            for pos,value in zip(compact['positions'],compact['values']):
                oid=packet['object_id'][pos];source=p['source_index'][pos]
                row=np.flatnonzero(cases[oid]['saved_flat']==source)[0]
                np.testing.assert_array_equal(value,cases[oid]['saved_features'][name][row])

    def test_no_cell_evidence_synthesis(self):
        packet,prep,selected,cases=packet_fixture(40)
        for oid in (0,1):
            s=selected[oid];f=s['F'];prep[oid]['C']['pca']=np.array([f[-1]],np.int64)
            s['T']['pca']=np.empty(0,np.int64);s['T']['sil']=s['P'].copy()
            prep[oid]['C']['sil']=s['P'].copy()
        packet,_=task.make_packet(0,prep,selected,cases)
        self.assertFalse(packet['provenance']['both_evidence'].any())
        self.assertTrue(packet['provenance']['candidate_evidence'][packet['provenance']['surface_selected'],0].any())

    def test_selection_all_under_GT_reader_sentinel_and_freeze(self):
        prep={f:{0:fixture(40),1:fixture(40)} for f in task.FRAMES}
        cases={f:{o:feature_case(p) for o,p in v.items()} for f,v in prep.items()}
        with patch.object(task,'read_gt',side_effect=AssertionError('GT reader opened')):
            selections,packets,_=task.select_all(prep,cases)
        self.assertEqual(len(selections),15)
        with tempfile.TemporaryDirectory() as d:
            digest,arrays=task.freeze(Path(d),prep,selections,packets)
            self.assertEqual(digest,task.prior.sha(Path(d)/'selection_before_gt.npz'))
            self.assertGreater(len(arrays),100)
            with self.assertRaises(ValueError):task.read_gt({},Path(d)/'selection_before_gt.npz','wrong hash')

    def test_selection_signatures_and_no_existing_cap_wrapper(self):
        for fn in (task.prepare_object,task.select_object,task.make_packet):
            signature=str(inspect.signature(fn)).lower()
            for forbidden in ('gt','reference','roi','budget','cap'):
                self.assertNotIn(forbidden,signature)
        code=inspect.getsource(task.select_object)
        self.assertNotIn('choose_priority',code);self.assertNotIn('sample_points',code)

    def test_distance_empty_denominators_and_joint_matcher(self):
        prep=fixture(4)
        d=task.distances(prep,np.array([1]),np.empty(0,np.int64))
        stats=task.distance_stats(d)
        self.assertIsNone(stats['p95']);self.assertEqual(stats['unavailable_count'],1)
        self.assertEqual(stats['within_3mm']['value'],0)
        self.assertIsNone(task.distance_stats([])['within_3mm']['value'])

    def test_observation_loader_never_reads_GT_or_evaluation_keys(self):
        original=np.load
        forbidden={'peg_id','box_id','peg_pose','box_pose','box_hole_pose','peg_half_sizes','box_hole_radii','seed'}
        accesses=[]
        class Guard:
            def __init__(self,path,*args,**kwargs):self.a=original(path,*args,**kwargs)
            def __getitem__(self,k):
                if k in forbidden or k.endswith('/seg') or k.startswith('evaluation/'):
                    raise AssertionError('GT before freeze: '+k)
                accesses.append(k);return self.a[k]
            def __getattr__(self,k):return getattr(self.a,k)
            def __contains__(self,k):return k in self.a
            def __enter__(self):return self
            def __exit__(self,*args):self.a.close()
        with patch.object(task.np,'load',side_effect=Guard):
            frames,_,prep,_,_,_=task.load_observations()
        self.assertEqual(tuple(frames),task.FRAMES)
        self.assertTrue(accesses)
        self.assertEqual(len(prep),3)


if __name__=='__main__':unittest.main()
