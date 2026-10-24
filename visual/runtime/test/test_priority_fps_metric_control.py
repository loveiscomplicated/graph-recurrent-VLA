"""Task03-E contracts; FPS inputs are synthetic, real replay belongs to runner."""
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.geometry import OrganizedGeometry
from visual.runtime.sampling import fps_indices
from visual.runtime.test import run_priority_fps_metric_control as control
from visual.runtime.test.sampling_evaluation import CoverageMatcher, staged_reference_coverage


def synthetic_case():
    flat=np.arange(120,dtype=np.int64)
    points=np.column_stack((np.sin(flat*.21),np.cos(flat*.15),flat*.02)).astype(np.float32)
    candidates=flat[::2]
    return dict(candidate_flat=candidates,candidate_xyz=points[::2].copy(),support_flat=flat,
        support_xyz=points,first=8,seed=0,budget=70,saved_flat=flat[:70],
        saved_features={'normals':points[:70].copy(),'flag':flat[:70]%3==0})


class MetricSelectionTests(unittest.TestCase):
    def test_metric_squared_distance_formula_and_original_unchanged(self):
        points=np.array([[0,0,1],[3,4,9],[1,5,2]],np.float32); before=points.copy()
        for weight in control.LAMBDAS:
            metric=control.metric_coordinates(points,weight)
            self.assertFalse(np.shares_memory(metric,points))
            diff=points.astype(float)[:,None]-points.astype(float)[None,:]
            expected=diff[...,0]**2+diff[...,1]**2+weight*diff[...,2]**2
            np.testing.assert_array_equal(((metric[:,None]-metric[None,:])**2).sum(-1),expected)
        np.testing.assert_array_equal(points,before)
        with self.assertRaises(ValueError): control.metric_coordinates(points,.5)

    def test_zero_weight_z_change_does_not_change_priority_with_xy_ties(self):
        case=synthetic_case(); points=case['candidate_xyz'].copy(); points[2,:2]=points[3,:2]
        altered=points.copy(); altered[:,2]=np.arange(len(points))**3
        a=fps_indices(control.metric_coordinates(points,0.),32,initial=[4])
        b=fps_indices(control.metric_coordinates(altered,0.),32,initial=[4])
        np.testing.assert_array_equal(a,b)
        # Exact XY ties retain existing source-row argmax behaviour.
        tie=np.zeros((40,3)); tie[:,2]=np.arange(40)
        np.testing.assert_array_equal(fps_indices(control.metric_coordinates(tie,0.),4,initial=[8]),[8,0,1,2])

    def test_lambda_one_matches_existing_priority_and_seeded_surface(self):
        case=synthetic_case(); original=copy.deepcopy(case)
        p=fps_indices(case['candidate_xyz'],32,initial=[4])
        initial=np.searchsorted(case['support_flat'],case['candidate_flat'][p])
        final=fps_indices(case['support_xyz'],70,initial=initial)
        selected=control.select(case,1.)
        np.testing.assert_array_equal(selected['P'],case['candidate_flat'][p])
        np.testing.assert_array_equal(selected['F'],case['support_flat'][final])
        for field in ('candidate_flat','candidate_xyz','support_flat','support_xyz'):
            np.testing.assert_array_equal(case[field],original[field])

    def test_all_weights_fixed_initial_quota_original_xyz_surface_metric(self):
        case=synthetic_case(); original=copy.deepcopy(case)
        for w in control.LAMBDAS:
            with patch.object(control,'fps_indices',wraps=fps_indices) as calls:
                chosen=control.select(case,w)
            self.assertEqual(calls.call_count,2)
            np.testing.assert_array_equal(calls.call_args_list[0].args[0],control.metric_coordinates(case['candidate_xyz'],w))
            np.testing.assert_array_equal(calls.call_args_list[1].args[0],case['support_xyz'])
            self.assertEqual(chosen['P'][0],case['first']); self.assertEqual(len(chosen['P']),32)
            self.assertEqual(len(chosen['S']),38); self.assertEqual(len(chosen['F']),70)
            self.assertEqual(len(np.unique(chosen['F'])),70)
            self.assertFalse(np.intersect1d(chosen['P'],chosen['S']).size)
            np.testing.assert_array_equal(chosen['xyz'],case['support_xyz'][chosen['F']])
        for field in ('candidate_flat','candidate_xyz','support_flat','support_xyz'):
            np.testing.assert_array_equal(case[field],original[field])

    def test_every_gate_required_before_either_control(self):
        cases={k:synthetic_case() for k in ('S2/peg','S2/box','S3/peg','S3/box')}
        for key in cases:
            gate={k:dict(status='PASS') for k in cases}; gate[key]['status']='FAIL'
            with patch.object(control,'select',side_effect=AssertionError('must not select')):
                with self.assertRaises(ValueError): control.execute_controls(cases,gate)
        with patch.object(control,'select',side_effect=AssertionError('must not select')):
            with self.assertRaises(ValueError): control.execute_controls(cases,{})

    def test_gt_annotations_and_segment_labels_cannot_affect_selection(self):
        case=synthetic_case(); before=control.select(case,.25)
        # Change annotations/segment labels while holding selection inputs fixed.
        annotations=[dict(gt=np.ones((4,3)),segments=['a']*4),dict(gt=np.full((4,3),999),segments=['ambiguous']*4)]
        for annotation in annotations:
            case_with_extra=dict(case,**annotation)
            after=control.select(case_with_extra,.25)
            np.testing.assert_array_equal(before['F'],after['F'])
            np.testing.assert_array_equal(before['xyz'],after['xyz'])
        self.assertEqual(tuple(inspect.signature(control.select).parameters),('case','weight'))
        with self.assertRaises(TypeError): control.select(case,.25,gt={})

    def test_feature_gather_exact_only_available_rows_no_placeholder(self):
        case=synthetic_case(); before=copy.deepcopy(case)
        available,positions,features=control.feature_rows(case,np.array([8,100,4,119,20]))
        np.testing.assert_array_equal(available,[True,False,True,False,True])
        np.testing.assert_array_equal(positions,[0,2,4])
        for name,value in features.items():
            np.testing.assert_array_equal(value,case['saved_features'][name][[8,4,20]])
            np.testing.assert_array_equal(case['saved_features'][name],before['saved_features'][name])


class LocalEvaluationTests(unittest.TestCase):
    def test_joint_matching_same_pair_dedup_and_staged_union(self):
        xyz=np.zeros((1,10,3),np.float32); xyz[0,3]=[.1,0,0]
        geometry=OrganizedGeometry(xyz,np.ones((1,10),bool),{0:np.ones((1,10),bool)},np.zeros((1,10),bool))
        matcher=CoverageMatcher(geometry,0,(np.array([[0.,0.],[7.,0.],[7.5,0.]]),np.array([[.1,0,0],[0,0,0],[0,0,0]])))
        ids,hit=matcher.reference_hits(np.array([0,3]))
        self.assertFalse(len(ids)); self.assertFalse(len(hit))
        # One node covers two refs, and multiple nodes cover those same two refs.
        p=np.array([7]); s=np.array([6,8]); f=np.r_[p,s]
        row=staged_reference_coverage(matcher,f,p,s,f)
        self.assertEqual(row['stages']['P']['matched_node_count'],1)
        self.assertEqual(row['stages']['P']['unique_reference_count'],2)
        self.assertEqual(row['stages']['S']['matched_node_count'],2)
        self.assertEqual(row['differences']['S_minus_P']['count'],0)
        self.assertEqual(row['differences']['P_intersect_S']['count'],2)
        self.assertEqual(row['reference_ids']['F'],[1,2])

    def test_peg_face_pair_endpoint_ambiguity_and_nonphysical_order_na(self):
        face=np.array([[0,0,2],[0,5,2]],np.int8); visible=np.ones_like(face,bool)
        seam=np.array([[False,True,True],[True,True,True]])
        uv=np.column_stack(np.nonzero(seam))[:,::-1]
        labels,members=control.peg_labels(face,visible,seam,uv)
        self.assertIn('ambiguous',labels)
        self.assertTrue(any(len(m)>1 for m in members))
        r=control.segment_spacing([.03,.01,.02],[False,True,True],[.015],order_verified=False)
        self.assertEqual(r['status'],'N/A')
        self.assertNotIn('actual_inter_node_gap_mm',r)

    def test_rim_line_identity_from_existing_transform_and_corner_ambiguity(self):
        # Known line points with shuffled reference order. Corner stays ambiguous.
        xyz=np.array([[.1,-.01,0],[.1,.005,-.01],[.1,-.01,-.01],[.1,-.01,.005]])
        pose=np.array([0,0,0,1,0,0,0],float)
        labels,specs,_=control.rim_labels(xyz,np.eye(4),pose,pose,.01)
        self.assertEqual(labels.tolist(),['rim_axis1_-1','rim_axis2_-1','ambiguous','rim_axis1_-1'])
        self.assertTrue(specs['rim_axis1_-1']['order_verified'])

    def test_spacing_uses_actual_node_projections_not_covered_reference_centres(self):
        row=control.segment_spacing(np.array([.03,.01,.02,.04]),np.array([False,True,False,True]),
            np.array([.012,.038]),order_verified=True)
        np.testing.assert_allclose(row['actual_inter_node_gap_mm'],[26])
        self.assertAlmostEqual(row['start_tail_to_first_node_mm'],2)
        self.assertAlmostEqual(row['end_tail_from_last_node_mm'],2)
        self.assertEqual(row['missing_reference_runs'][0]['reference_count'],2)
        self.assertAlmostEqual(row['missing_reference_run_total_span_mm'],10)
        self.assertIsNone(row['continuous_unpreserved_length_mm'])
        self.assertEqual(control.segment_spacing([.01],[True],[],order_verified=True)['status'],'N/A')

    def test_actual_gap_includes_matching_node_outside_visible_reference_span(self):
        # A joint-matching node may lie just beyond the first reference centre.
        # Keep that actual node when measuring adjacent selected-node intervals.
        row=control.segment_spacing([.01,.02,.03],[True,False,True],
            [.009,.025],order_verified=True)
        np.testing.assert_allclose(row['actual_inter_node_gap_mm'],[16.])
        self.assertEqual(row['selected_nodes_within_reference_span'],1)
        self.assertAlmostEqual(row['start_tail_to_first_node_mm'],15.)


if __name__=='__main__': unittest.main()
