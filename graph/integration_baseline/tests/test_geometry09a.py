"""Geometry diagnostic failure injection; fixtures are not experiment data."""
from itertools import product
from unittest.mock import patch
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from graph.integration_baseline.geometry09a import convex_gap, expert_reference, protocol


class GeometryContracts(unittest.TestCase):
    def test_registered_candidates_and_scope(self):
        p=protocol()
        self.assertEqual(p['timesteps'],[0,10,25])
        self.assertEqual(sum(s['group']=='primary' for s in p['primary']),8)
        self.assertEqual(sum(s['group']=='box_control' for s in p['primary']),4)
        self.assertEqual(len(p['additional']),6)
        self.assertTrue(p['no_student_step'] and p['no_training'])

    def test_collision_sat_separation_touch_and_penetration(self):
        a=np.array(list(product((-.01,.01),repeat=3)))
        self.assertAlmostEqual(convex_gap(a,a+[.025,0,0]),.005)
        self.assertAlmostEqual(convex_gap(a,a+[.02,0,0]),0.)
        self.assertLess(convex_gap(a,a+[.015,0,0]),-.0049)

    def test_rotated_collision_handles_aabb_false_positive(self):
        a=np.array(list(product((-.05,.05),(-.002,.002),(-.002,.002))))
        rot=Rotation.from_euler('z',45,degrees=True).as_matrix()
        a=a@rot.T;b=a+np.array([-.004,.004,0])
        self.assertGreater(convex_gap(a,b),.001)
        self.assertLess(convex_gap(a,a),0)

    def test_unverified_expert_stage_is_excluded(self):
        from types import SimpleNamespace
        pose=SimpleNamespace(p=np.zeros(3),q=np.array([1.,0,0,0]))
        baseline=SimpleNamespace(phase='align',counter=2,grasp=pose)
        with patch('graph.integration_baseline.geometry09a.ScriptedExpert',return_value=SimpleNamespace(grasp=pose)):
            action,detail=expert_reference(None,baseline,'peg',dict(yaw_deg=0,xyz_mm=[5,0,0]))
        self.assertEqual(detail['status'],'INVALID_EXPERT_REFERENCE')
        np.testing.assert_array_equal(action,np.zeros(7))
        self.assertEqual(baseline.phase,'align')

    def test_response_direction_requires_discriminative_reference(self):
        from graph.integration_baseline.geometry09a_report import response_metrics
        base=dict(action=np.zeros((1,7)),controller=np.zeros((1,7)),readout=np.ones((1,320)),
            presence=np.ones((1,2)),previous_hidden=np.zeros((1,1,144)),hidden=np.zeros((1,1,144)),
            gru_input=np.zeros((1,1,128)),encoded=np.zeros((1,7)))
        alt={k:v.copy() for k,v in base.items()};alt['controller'][0,0]=.001
        eb=np.zeros(7);ea=np.zeros(7);ea[0]=.002
        m=response_metrics(base,alt,eb,ea)
        self.assertAlmostEqual(m['translation_direction_cosine'],1.)
        self.assertAlmostEqual(m['translation_magnitude_ratio'],.5)
        self.assertAlmostEqual(m['translation_response_error_mm'],1.)
        ea[0]=1e-8
        m=response_metrics(base,alt,eb,ea)
        self.assertEqual(m['reference_class'],'NON_DISCRIMINATIVE_REFERENCE')
        self.assertIsNone(m['translation_direction_cosine'])
        self.assertIsNone(m['translation_magnitude_ratio'])

    def test_opposite_response_is_not_useful_sensitivity(self):
        from graph.integration_baseline.geometry09a_report import response_metrics
        base=dict(action=np.zeros((1,7)),controller=np.zeros((1,7)),readout=np.ones((1,320)),
            presence=np.ones((1,2)),previous_hidden=np.zeros((1,1,144)),hidden=np.zeros((1,1,144)),
            gru_input=np.zeros((1,1,128)),encoded=np.zeros((1,7)))
        alt={k:v.copy() for k,v in base.items()};alt['controller'][0,0]=-.003
        eb=np.zeros(7);ea=np.zeros(7);ea[0]=.003
        m=response_metrics(base,alt,eb,ea)
        self.assertAlmostEqual(m['translation_direction_cosine'],-1.)
        self.assertAlmostEqual(m['translation_magnitude_ratio'],1.)
        self.assertAlmostEqual(m['translation_response_error_mm'],6.)


if __name__=='__main__':unittest.main()
