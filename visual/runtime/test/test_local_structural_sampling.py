"""Task03-F contracts use synthetic inputs; performance gains are never assertions."""
import copy
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.sampling import fps_indices
from visual.runtime.test import run_local_structural_sampling as local


def case():
    flat=np.arange(160,dtype=np.int64)
    xyz=np.column_stack((np.sin(flat*.1),np.cos(flat*.13),1+flat*.01)).astype(np.float32)
    c=flat[::2]
    return dict(candidate_flat=c,candidate_xyz=xyz[::2].copy(),support_flat=flat,
        support_xyz=xyz,shape=(16,10),seed=47,anchor=18,budget=100)


class GroupTests(unittest.TestCase):
    def test_disjoint_exhaustive_partition_capacity_and_order(self):
        c=case();flat=c['candidate_flat'];groups,membership,splits=local.spatial_groups(flat,c['shape'])
        self.assertEqual(len(groups),8)
        np.testing.assert_array_equal(np.sort(np.concatenate(groups)),np.arange(len(flat)))
        self.assertEqual(len(np.unique(np.concatenate(groups))),len(flat))
        self.assertTrue(all(len(g)>=4 for g in groups))
        self.assertEqual([int(flat[g[0]]) for g in groups],sorted(int(flat[g[0]]) for g in groups))
        for gid,g in enumerate(groups):
            self.assertTrue(np.all(membership[g]==gid));self.assertTrue(np.all(np.diff(flat[g])>0))
        self.assertTrue(all(min(s['child_capacity'])>=4 for s in splits))

    def test_axis_and_balanced_cut_ties(self):
        # Equal u/v spans, balanced u=1.5, u axis wins.
        flat=np.arange(16,dtype=np.int64)
        _,_,splits=local.spatial_groups(flat,(4,4))
        self.assertEqual(splits[0]['axis'],'u');self.assertEqual(splits[0]['boundary'],1.5)
        # One horizontal line of 9: cuts 3.5 and 4.5 have equal imbalance.
        _,_,splits=local.spatial_groups(np.arange(9,dtype=np.int64),(1,9))
        self.assertEqual(splits[0]['boundary'],3.5)

    def test_invalid_longer_axis_uses_valid_shorter_axis(self):
        # u=0 only one candidate, u=100 seven: u has no valid split.
        flat=np.array([0]+[100+101*v for v in range(7)],np.int64)
        flat.sort()
        _,_,s=local.spatial_groups(flat,(7,101))
        self.assertEqual(s[0]['axis'],'v')

    def test_leaf_tie_and_equal_coordinate_not_divided(self):
        flat=np.arange(32,dtype=np.int64)
        _,_,s=local.spatial_groups(flat,(4,8))
        # After first equal-capacity split, smallest original flat leaf is first.
        self.assertEqual(s[1]['parent_min_flat'],0)
        for split in s:self.assertTrue(all(n>=4 for n in split['child_capacity']))
        a=local.spatial_groups(flat,(4,8));b=local.spatial_groups(flat.copy(),(4,8))
        np.testing.assert_array_equal(a[1],b[1]);self.assertEqual(a[2],b[2])

    def test_invalid_sources_fail_closed(self):
        for flat in (np.array([1,1]),np.array([2,1]),np.array([-1,2]),np.array([1.2,2.])):
            with self.assertRaises(ValueError):local.spatial_groups(flat,(2,4))

    def test_quota_capacity_round_robin_and_zero(self):
        np.testing.assert_array_equal(local.allocate([1,4,100],8),[1,4,3])
        np.testing.assert_array_equal(local.allocate([4]*8,32),[4]*8)
        np.testing.assert_array_equal(local.allocate([4]*8,3),[1,1,1,0,0,0,0,0])
        np.testing.assert_array_equal(local.allocate([1,2],32),[1,2])
        self.assertEqual(local.allocate([],32).size,0)
        with self.assertRaises(ValueError):local.allocate([4],-1)


class SelectionTests(unittest.TestCase):
    def test_budgets_alignment_uniqueness_original_geometry_and_calls(self):
        c=case();before=copy.deepcopy(c)
        with patch.object(local,'fps_indices',wraps=fps_indices) as calls:
            chosen=local.select_local(**c)
        self.assertEqual(calls.call_count,len(chosen['grouping']['quota'])+1)
        self.assertEqual((len(chosen['P']),len(chosen['S']),len(chosen['F'])),(32,68,100))
        self.assertEqual(chosen['P'][0],c['anchor'])
        self.assertEqual(len(np.unique(chosen['F'])),100)
        self.assertEqual(len(np.intersect1d(chosen['P'],chosen['S'])),0)
        self.assertTrue(np.isin(chosen['P'],c['candidate_flat']).all())
        np.testing.assert_array_equal(chosen['F'],np.r_[chosen['P'],chosen['S']])
        np.testing.assert_array_equal(chosen['xyz'],c['support_xyz'][chosen['F']])
        np.testing.assert_array_equal(calls.call_args_list[-1].args[0],c['support_xyz'])
        for call in calls.call_args_list[:-1]:self.assertTrue(np.all(call.args[0][:,2]==0))
        for k in ('candidate_flat','candidate_xyz','support_flat','support_xyz'):np.testing.assert_array_equal(c[k],before[k])

    def test_anchor_and_nonanchor_group_rng_and_order(self):
        c=case();p=local.choose_priority(c['candidate_flat'],c['candidate_xyz'],c['shape'],c['seed'],c['anchor'])
        leaves,_,_=local.spatial_groups(c['candidate_flat'],c['shape'])
        for gid,rows in enumerate(leaves):
            expected=c['anchor'] if c['anchor'] in c['candidate_flat'][rows] else int(c['candidate_flat'][rows][np.random.default_rng(c['seed']).integers(len(rows))])
            self.assertEqual(p['initial_flat'][gid],expected)
        again=local.choose_priority(c['candidate_flat'],c['candidate_xyz'],c['shape'],c['seed'],c['anchor'])
        np.testing.assert_array_equal(p['P'],again['P'])

    def test_small_budget_zero_anchor_quota_does_not_force_selection(self):
        c=case();groups,_,_=local.spatial_groups(c['candidate_flat'],c['shape'])
        anchor=int(c['candidate_flat'][groups[-1][0]])
        with patch.object(local,'fps_indices',wraps=fps_indices) as calls:
            p=local.choose_priority(c['candidate_flat'],c['candidate_xyz'],c['shape'],c['seed'],anchor,count=1)
        self.assertEqual(calls.call_count,1);self.assertNotEqual(p['P'][0],anchor)
        self.assertEqual(p['quota'].sum(),1)

    def test_single_group_matches_global_priority_and_final_exact(self):
        flat=np.arange(7,dtype=np.int64);xyz=np.column_stack((flat,flat**2,flat*.2)).astype(float)
        p=local.choose_priority(flat,xyz,(1,7),6,anchor=3,count=4)
        expected=fps_indices(local.prior.metric_coordinates(xyz,0.),4,seed=6,initial=[3])
        np.testing.assert_array_equal(p['P'],flat[expected])
        support=np.arange(20,dtype=np.int64);sx=np.column_stack((support,support**2,support*.2)).astype(float)
        chosen=local.select_local(flat,xyz,support,sx,(1,20),6,3,15)
        allp=fps_indices(local.prior.metric_coordinates(xyz,0.),7,seed=6,initial=[3])
        final=fps_indices(sx,15,seed=6,initial=allp)
        np.testing.assert_array_equal(chosen['F'],support[final])

    def test_empty_candidates_shortfall_and_no_padding(self):
        flat=np.arange(6,dtype=np.int64);xyz=np.column_stack((flat,flat,flat)).astype(float)
        with patch.object(local,'fps_indices',wraps=fps_indices) as calls:
            chosen=local.select_local(np.empty(0,np.int64),np.empty((0,3)),flat,xyz,(1,6),2,None,10)
        self.assertEqual(calls.call_count,1)
        self.assertEqual(len(chosen['P']),0);self.assertEqual(len(chosen['F']),6);self.assertEqual(chosen['shortfall'],4)

    def test_xy_ties_depth_invariance_and_same_xyz_distinct_sources(self):
        flat=np.arange(40,dtype=np.int64);xyz=np.zeros((40,3));xyz[:,2]=np.arange(40)
        a=local.choose_priority(flat,xyz,(5,8),0,anchor=19)
        xyz[:,2]**=3
        b=local.choose_priority(flat,xyz,(5,8),0,anchor=19)
        np.testing.assert_array_equal(a['P'],b['P']);self.assertEqual(len(np.unique(a['P'])),32)
        for gid in range(len(a['quota'])):
            selected=a['rows'][a['membership'][a['rows']]==gid]
            rest=selected[1:]
            self.assertTrue(np.all(np.diff(rest)>0))

    def test_support_mapping_mismatch_and_bad_anchor_fail(self):
        c=case();c['candidate_xyz'][0,0]+=1
        with self.assertRaises(AssertionError):local.select_local(**c)
        c=case();c['anchor']=-1
        with self.assertRaises(ValueError):local.select_local(**c)

    def test_annotation_change_delete_does_not_affect_selection(self):
        c=case();annotations={'pose':np.eye(4),'rim':np.ones((4,3)),'face_pairs':['a'],'roi':np.ones(10)}
        a=local.select_local(**c)
        annotations.update(pose=np.full((4,4),999),rim=None,face_pairs=['noise']);annotations.clear()
        b=local.select_local(**c)
        for k in ('P','S','F','xyz'):np.testing.assert_array_equal(a[k],b[k])
        self.assertNotIn('annotation',inspect.signature(local.select_local).parameters)
        with self.assertRaises(TypeError):local.select_local(**c,annotation={})
        # No module GT state or evaluator is read by either selection helper.
        for function in (local.choose_priority,local.select_local,local.spatial_groups):
            self.assertNotIn('evaluation_inputs',function.__code__.co_names)
            self.assertNotIn('score_case',function.__code__.co_names)

    def test_feature_unavailable_exact_compact_only(self):
        c=dict(saved_flat=np.array([3,7,11]),saved_features={'value':np.array([[1,2],[3,4],[5,6]]),'valid':np.array([True,False,True])})
        a,pos,features=local.prior.feature_rows(c,np.array([11,12,3,0]))
        np.testing.assert_array_equal(a,[True,False,True,False]);np.testing.assert_array_equal(pos,[0,2])
        np.testing.assert_array_equal(features['value'],[[5,6],[1,2]])
        self.assertEqual(features['valid'].shape,(2,))


class SpatialEvaluationTests(unittest.TestCase):
    def test_known_distances_raw_normalized_and_worst_source(self):
        flat=np.array([0,3,6],np.int64);xyz=np.array([[0,0,1],[3,0,1],[6,0,1]],float)
        m,r=local.spatial_metrics(flat,xyz,flat[:1],xyz[:1],flat,xyz,(1,7))
        np.testing.assert_allclose(r['distance_mm'],[0,3000,6000]);np.testing.assert_allclose(r['distance_px'],[0,3,6])
        self.assertEqual(m['mm']['raw']['p50'],3000);self.assertAlmostEqual(m['mm']['raw']['p95'],5700)
        self.assertEqual(m['mm']['worst']['flat_index'],6)
        np.testing.assert_allclose(r['normalized_mm'],[0,.5,1])
        np.testing.assert_allclose(r['ecdf_px'],[0,3,6])

    def test_grid_boundary_max_endpoint_zero_span_and_same_definition(self):
        points=np.array([[0,0],[1,0],[2,0],[3,0],[4,0]],float)
        definition=local.grid_definition(points,[4,4])
        self.assertEqual(definition['bins'],[4,1])
        np.testing.assert_array_equal(local.grid_ids(points,definition),[0,1,2,3,3])
        flat=np.array([0,1,2,3,4],np.int64);xyz=np.column_stack((flat,np.zeros(5),np.ones(5)))
        a,_=local.spatial_metrics(flat,xyz,flat[:1],xyz[:1],flat,xyz,(1,5))
        b,_=local.spatial_metrics(flat,xyz,flat[-1:],xyz[-1:],flat,xyz,(1,5))
        self.assertEqual(a['grids']['uv']['definition'],b['grids']['uv']['definition'])
        self.assertEqual(a['grids']['uv']['empty_priority_region_ids'],[1,2,3])

    def test_region_equal_weight_differs_from_candidate_weight_and_no_group_input(self):
        # Dense near cluster and one distant candidate: region macro exposes sparse loss.
        flat=np.arange(11,dtype=np.int64);xyz=np.column_stack((np.r_[np.zeros(10),10.],np.zeros(11),np.ones(11)))
        m,_=local.spatial_metrics(flat,xyz,flat[:1],xyz[:1],flat,xyz,(1,11))
        cells=m['grids']['xyz']['cells'];self.assertEqual([c['candidate_count'] for c in cells],[10,1])
        self.assertEqual(m['grids']['xyz']['equal_region_mean_p95_mm'],5000)
        self.assertNotIn('group',inspect.signature(local.spatial_metrics).parameters)

    def test_empty_and_zero_bbox_na(self):
        flat=np.array([0],np.int64);xyz=np.array([[0.,0.,1.]])
        m,r=local.spatial_metrics(flat,xyz,flat,xyz,flat,xyz,(1,1))
        self.assertIsNone(m['mm']['normalized']['p95']);self.assertTrue(np.isnan(r['normalized_mm']).all())
        m,_=local.spatial_metrics(flat,xyz,np.empty(0,np.int64),np.empty((0,3)),flat,xyz,(1,1))
        self.assertIsNone(m['mm']['raw']['p95']);self.assertIsNone(m['mm']['worst'])


class FixtureContracts(unittest.TestCase):
    def test_exact_seven_constants_and_reproducible_source_correspondence(self):
        self.assertEqual(len(local.FIXTURES['names']),7)
        self.assertEqual(local.FIXTURES['focal_px'],1600.)
        for name in local.FIXTURES['names']:
            d,a,c=local.synthetic_fixture(name);e,b,k=local.synthetic_fixture(name)
            self.assertEqual(c,k);np.testing.assert_array_equal(d['flat'],e['flat']);np.testing.assert_array_equal(d['xyz'],e['xyz'])
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            self.assertTrue(np.all(np.diff(d['flat'])>0));self.assertGreaterEqual(len(d['flat']),32)
            self.assertEqual(len(a['generated_id']),len(d['flat']))
            self.assertEqual(c['in_view_before_collision']-c['collision_count'],len(d['flat']))

    def test_density_noise_camera_and_occlusion_are_input_changes_only(self):
        d,a,c=local.synthetic_fixture('line_dense');self.assertEqual(c['generated_count'],256)
        d,a,c=local.synthetic_fixture('separated_noise');self.assertEqual(c['generated_count'],200)
        self.assertEqual((a['structure_id']==-1).sum(),8)
        near,an,cn=local.synthetic_fixture('camera_near');far,af,cf=local.synthetic_fixture('camera_far')
        self.assertTrue(np.all(near['xyz'][:,2]==.8));self.assertTrue(np.all(far['xyz'][:,2]==1.6))
        self.assertEqual(cn['generated_count'],cf['generated_count'])
        d,a,c=local.synthetic_fixture('camera_occluded');uv=local.uv_from_flat(d['flat'],d['shape'])
        self.assertFalse(((uv[:,0]>=384)&(uv[:,0]<640)).any());self.assertGreater(c['excluded_references'],0)

    def test_annotation_not_shared_or_required_by_priority(self):
        d,a,_=local.synthetic_fixture('separated_noise')
        seed=local.prior.fps_seed(0,0)
        p=local.choose_priority(d['flat'],d['xyz'],d['shape'],seed)
        a['structure_id'][:]=999;a['reference_xyz'][:]=1e9;a.clear()
        q=local.choose_priority(d['flat'],d['xyz'],d['shape'],seed)
        np.testing.assert_array_equal(p['P'],q['P'])

    def test_no_baseline_replay_or_production_control_entrypoints(self):
        forbidden={'load_inputs','execute_controls','baseline_gate','select'}
        self.assertFalse(forbidden.intersection(local.load_actual.__code__.co_names))
        self.assertNotIn('fps_indices',local.load_actual.__code__.co_names)


if __name__=='__main__':unittest.main()
