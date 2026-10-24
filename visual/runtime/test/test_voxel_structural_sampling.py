"""Task03-I contracts. Performance improvement is not a PASS condition."""
import copy
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.test import run_voxel_structural_sampling as voxel
from visual.runtime.sampling import fps_indices
from visual.runtime.geometry import OrganizedGeometry
from visual.runtime.test.sampling_evaluation import CoverageMatcher


def case(n=100):
    flat=np.arange(n,dtype=np.int64)
    xyz=np.column_stack((flat*.01,np.sin(flat)*.002,np.ones(n))).astype(np.float32)
    return dict(candidate_flat=flat[::2],candidate_xyz=xyz[::2].copy(),support_flat=flat,
        support_xyz=xyz,shape=(10,10),intrinsic=np.diag([100.,100.,1.]),seed=47,anchor=0,final_target=64)


class GridTests(unittest.TestCase):
    def test_fixed_origin_negative_floor_and_boundaries(self):
        q=np.array([[-1.1,-1.,-.1],[0.,1.,1.1],[np.nextafter(1.,0),np.nextafter(1.,2),-0.]])
        np.testing.assert_array_equal(voxel.voxel_keys(q,1.),[[-2,-1,-1],[0,1,1],[0,1,0]])
        # Adding a distant observation never recentres existing cells.
        np.testing.assert_array_equal(voxel.voxel_keys(np.vstack([q,[1000,1000,1]]),1.)[:-1],voxel.voxel_keys(q,1.))

    def test_representative_centroid_is_not_output_and_tie_source(self):
        flat=np.array([2,4,7],np.int64);xyz=np.array([[.125,.125,.125],[.375,.125,.125],[1.5,.125,.125]])
        g=voxel.representatives(flat,xyz,1.)
        np.testing.assert_array_equal(flat[g['representative_rows']],[2,7])
        np.testing.assert_array_equal(g['same_representative_source'],[2,2,7])
        self.assertNotIn(.2,xyz[g['representative_rows'],0])
        np.testing.assert_array_equal(g['capacity'],[2,1])

    def test_coincident_xyz_sources_collapse_without_duplicate(self):
        g=voxel.representatives(np.array([1,3,5]),np.ones((3,3)),1.)
        np.testing.assert_array_equal(g['representative_rows'],[0])
        np.testing.assert_array_equal(g['same_cell_distance_m'],0.)

    def test_determinism_input_unchanged_and_bound(self):
        c=case();snapshot=copy.deepcopy(c)
        a=voxel.select(**c);b=voxel.select(**c)
        for key in ('R','P','F','xyz'): np.testing.assert_array_equal(a[key],b[key])
        for key,value in c.items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,snapshot[key])
        self.assertLessEqual(a['grouping']['same_cell_distance_m'].max(),a['scale']['epsilon_m']+1e-12)

    def test_invalid_grid_and_source_inputs_fail_closed(self):
        for a in (0,-1,np.inf,np.nan):
            with self.assertRaises(ValueError):voxel.voxel_keys(np.ones((1,3)),a)
        for xyz in (np.array([[np.nan,1,1]]),np.array([[2.**63,0,0]]),np.array([[-2.**64,0,0]])):
            with self.assertRaises(ValueError):voxel.voxel_keys(xyz,1.)
        for flat in (np.array([1,1]),np.array([2,1]),np.array([-1,2]),np.array([1.,2.])):
            with self.assertRaises(ValueError):voxel.representatives(flat,np.ones((2,3)),1.)

    def test_scale_median_robust_to_extreme_max_and_uses_only_X(self):
        x=np.array([[0,0,1],[0,0,2],[0,0,30.726]])
        s=voxel.nominal_scale(x,np.diag([256.,128.,1.]))
        self.assertEqual(s['z_ref_m'],2.)
        self.assertAlmostEqual(s['epsilon_m'],2*np.hypot(2/256,2/128))
        self.assertAlmostEqual(s['a_m']*np.sqrt(3),s['epsilon_m'])
        self.assertEqual(voxel.nominal_scale(np.empty((0,3)),np.eye(3))['status'],'N/A')
        for x,k in ((np.array([[0,0,-1]]),np.eye(3)),(x,np.diag([0,1,1]))):
            with self.assertRaises(ValueError):voxel.nominal_scale(x,k)

    def test_dyadic_parent_nesting_first_count_and_nonexact32(self):
        keys=np.column_stack((np.arange(-40,40),np.zeros(80,int),np.zeros(80,int))).astype(np.int64)
        d=voxel.dyadic_counts(keys,.01)
        self.assertEqual([v['occupied'] for v in d['levels']],[80,40,20])
        self.assertEqual(d['ratio'],4);self.assertFalse(d['used_for_selection'])
        self.assertEqual(voxel.dyadic_counts(keys,.01,0)['status'],'impossible')
        self.assertEqual(voxel.dyadic_counts(np.empty((0,3),np.int64),.01)['occupied'],0)

    def test_dyadic_signed_int64_extremes_terminate(self):
        vals=np.array([np.iinfo(np.int64).min,np.iinfo(np.int64).max],np.int64)
        keys=np.array(np.meshgrid(vals,vals,vals)).reshape(3,-1).T
        d=voxel.dyadic_counts(keys,.001)
        self.assertLessEqual(d['occupied'],32)
        # General, non-nested voxel sizes do not imply monotone cell counts.
        xyz=np.array([[.9,0,0],[1.1,0,0],[1.9,0,0]])
        self.assertEqual(len(np.unique(voxel.voxel_keys(xyz,1.),axis=0)),2)


class SelectionTests(unittest.TestCase):
    def test_more_equal_fewer_occupied_and_zero_cap(self):
        for n in (0,1,10,32,45):
            flat=np.arange(n,dtype=np.int64);xyz=np.column_stack((flat*2.,flat*0,flat*0+1))
            g=voxel.choose_priority(flat,xyz,1.,47,0 if n else None,32,(10,10))
            self.assertEqual(len(g['P']),min(n,32));self.assertEqual(len(g['R']),n)
            self.assertLessEqual(len(np.unique(g['P'])),32)
            if n<=32: np.testing.assert_array_equal(np.sort(g['P']),flat)
        z=voxel.choose_priority(flat,xyz,1.,47,0,0)
        self.assertEqual(len(z['P']),0)

    def test_anchor_maps_to_representative_records_distance_pixels(self):
        f=np.array([0,1,2],np.int64);x=np.array([[.1,0,1],[.2,0,1],[.4,0,1]])
        g=voxel.choose_priority(f,x,1.,0,0,32,(1,3))
        self.assertEqual(g['anchor']['original_source'],0);self.assertEqual(g['anchor']['mapped_source'],1)
        self.assertAlmostEqual(g['anchor']['distance_m'],.1)
        self.assertEqual(g['anchor']['delta_pixel_uv'],[1,0]);self.assertEqual(g['anchor']['pixel_distance_px'],1)
        np.testing.assert_array_equal(g['P'],[1])

    def test_cap_reuses_existing_3d_fps_and_exact_source(self):
        c=case();c['intrinsic']=np.diag([10000.,10000.,1.])
        with patch.object(voxel,'fps_indices',wraps=fps_indices) as spy:
            s=voxel.select(**c)
        self.assertEqual(spy.call_count,2)
        rxyz=c['candidate_xyz'][np.searchsorted(c['candidate_flat'],s['R'])]
        np.testing.assert_array_equal(spy.call_args_list[0].args[0],rxyz)
        self.assertEqual(spy.call_args_list[0].kwargs['initial'],[int(np.searchsorted(s['R'],s['grouping']['anchor']['mapped_source']))])
        np.testing.assert_array_equal(s['F'][:len(s['P'])],s['P'])
        np.testing.assert_array_equal(s['xyz'],c['support_xyz'][np.searchsorted(c['support_flat'],s['F'])])
        self.assertTrue(set(s['P'])<=set(s['R'])<=set(c['candidate_flat'])<=set(c['support_flat']))
        self.assertEqual(len(s['F']),len(np.unique(s['F'])))

    def test_underfill_surface_empty_candidate_small_support_and_empty(self):
        for n,nc in ((10,2),(10,0),(0,0)):
            c=case(n);c['candidate_flat']=c['support_flat'][:nc];c['candidate_xyz']=c['support_xyz'][:nc]
            c['anchor']=0 if nc else None
            s=voxel.select(**c)
            self.assertEqual(len(s['F']),n);self.assertLessEqual(len(s['P']),nc)
            self.assertEqual(len(s['S']),n-len(s['P']));self.assertEqual(s['shortfall'],64-n)

    def test_invalid_membership_coordinates_anchor_target(self):
        c=case();c['candidate_xyz'][0,0]=42
        with self.assertRaises(AssertionError):voxel.select(**c)
        c=case();c['anchor']=99
        with self.assertRaises(ValueError):voxel.select(**c)
        c=case();c['final_target']=-1
        with self.assertRaises(ValueError):voxel.select(**c)

    def test_decomposition_nn_extra_and_ties(self):
        c=case();s=voxel.select(**c)
        metrics,a=voxel.distance_decomposition(c['candidate_flat'],c['candidate_xyz'],s,c['shape'])
        self.assertTrue((a['budget_extra_m']>=-1e-12).all())
        np.testing.assert_allclose(a['C_to_R_m']+a['budget_extra_m'],a['C_to_P_m'])
        self.assertTrue((a['C_to_R_m']<=a['same_cell_m']+1e-12).all())
        d,near=voxel.nearest_sources(np.array([[0.,0,1]]),np.array([9,1]),np.array([[1,0,1],[-1,0,1]]))
        self.assertEqual(near[0],1);self.assertEqual(d[0],1)
        self.assertEqual(metrics['nominal_occupied']-metrics['selected_occupied'],metrics['discarded_occupied'])

    def test_empty_P_distance_is_missing_not_zero(self):
        c=case();g=voxel.choose_priority(c['candidate_flat'],c['candidate_xyz'],.01,0,0,0,c['shape'])
        s=dict(grouping=g,R=g['R'],P=g['P'],scale=dict(epsilon_m=.01,a_m=.01))
        row,a=voxel.distance_decomposition(c['candidate_flat'],c['candidate_xyz'],s,c['shape'])
        self.assertIsNone(row['C_to_P_m']['mean']);self.assertEqual(row['C_to_P_m']['missing_count'],50)
        self.assertTrue(np.isnan(a['C_to_P_m']).all())


class EvaluationContracts(unittest.TestCase):
    def test_selection_signature_has_no_annotations_and_freeze_before_GT(self):
        forbidden=('gt','roi','pose','score','label','reference','annotation')
        for f in (voxel.select,voxel.choose_priority,voxel.representatives,voxel.nominal_scale):
            self.assertFalse(any(bad in p.lower() for p in inspect.signature(f).parameters for bad in forbidden))
        c=case();s=voxel.select(**c)
        case_data=dict(c,budget=64)
        with tempfile.TemporaryDirectory() as temp,patch.object(voxel.prior,'evaluation_inputs',side_effect=AssertionError('GT forbidden')):
            sha,arrays=voxel.freeze_selection(Path(temp),{'peg':case_data},{'peg':s},{})
            self.assertEqual(voxel.prior.sha(Path(temp)/'selection_before_gt.npz'),sha)
            self.assertIn('peg/R',arrays)
            with self.assertRaises(FileExistsError):voxel.freeze_selection(Path(temp),{'peg':case_data},{'peg':s},{})

    def test_available_feature_exact_gather_variable_priority(self):
        c=dict(saved_flat=np.array([1,3,7]),saved_features={'score':np.array([.1,.3,.7]),'normal_valid':np.array([True,False,True])})
        mask,pos,fields=voxel.prior.feature_rows(c,np.array([3,4,1,9]))
        np.testing.assert_array_equal(mask,[True,False,True,False]);np.testing.assert_array_equal(pos,[0,2])
        np.testing.assert_array_equal(fields['score'],[.3,.1]);np.testing.assert_array_equal(fields['normal_valid'],[False,True])

    def test_GT_stage_unique_node_reference_same_pair_and_subset_losses(self):
        xyz=np.array([[[0,0,1],[.001,0,1],[.002,0,1]]],np.float32)
        geometry=OrganizedGeometry(xyz,np.ones((1,3),bool),{0:np.ones((1,3),bool)},np.zeros((1,3),bool))
        reference=(np.array([[0,0],[1,0],[2,0]]),xyz.reshape(-1,3).copy())
        matcher=CoverageMatcher(geometry,0,reference)
        context=dict(matcher=matcher,reference=reference,observable=np.ones(3,bool))
        row,arrays=voxel.stage_gt(np.array([0]),context,np.array(['a']*3),{},None)
        self.assertEqual(row['matching_node_count'],1);self.assertEqual(row['unique_reference_count'],3)
        self.assertEqual(len(arrays['joint_reference_id']),3)
        bad=CoverageMatcher(geometry,0,(np.array([[0,0],[99,0]]),np.array([[0,0,2],[0,0,1]])))
        context.update(matcher=bad,reference=(bad.ref_uv,bad.ref_xyz),observable=np.ones(2,bool))
        row,_=voxel.stage_gt(np.array([0]),context,np.array(['a','a']),{},None)
        self.assertEqual(row['unique_reference_count'],0)

    def test_variable_P_adapter_and_separate_reference_losses(self):
        flat=np.array([0,3,6],np.int64);xyz=np.zeros((1,7,3),np.float32)
        xyz[0,flat]=np.column_stack((np.arange(3)*.01,np.zeros(3),np.ones(3)))
        support=np.zeros((1,7),bool);support[0,flat]=True
        geometry=OrganizedGeometry(xyz,support,{0:support},np.zeros_like(support))
        reference=(np.column_stack((flat,np.zeros(3))),xyz[0,flat])
        matcher=CoverageMatcher(geometry,0,reference)
        context=dict(matcher=matcher,reference=reference,observable=np.ones(3,bool),negatives={'test':np.zeros_like(support)})
        c=dict(candidate_flat=flat,candidate_xyz=xyz[0,flat],support_flat=flat,support_xyz=xyz[0,flat],
            saved_flat=flat,saved_features={'test_feature':np.arange(3)},oid=0)
        chosen=dict(R=flat[:2],P=flat[:1],S=flat[2:],F=flat[[0,2]],xyz=xyz[0,flat[[0,2]]])
        arrays={};row=voxel.evaluate_method(c,chosen,context,SimpleNamespace(depth=np.ones((1,7))),np.array(['a']*3),{},None,arrays,'test')
        self.assertEqual([row['stages'][k]['unique_reference_count'] for k in ('C','R','P','S','F')],[3,2,1,1,2])
        self.assertEqual(row['reference_differences']['C_minus_R']['reference_ids'],[2])
        self.assertEqual(row['reference_differences']['R_minus_P']['reference_ids'],[1])
        self.assertEqual(row['reference_differences']['F_minus_P']['reference_ids'],[2])
        np.testing.assert_array_equal(arrays['test/F_selection_role'],['Priority','Surface'])

    def test_protected_hash_failure_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'original';p.write_bytes(b'original')
            baseline=dict(hashes={str(p):voxel.prior.sha(p)},head='unused')
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):voxel.preserved(baseline)


if __name__=='__main__': unittest.main()
