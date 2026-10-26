"""Intervention isolation and physical-outcome rejection contracts."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from graph.integration_baseline.grasp_intervention import (
    close_action, retention, compare_prefix, atomic, validate_config, CONDITIONS)
from graph.integration_baseline.direct_runtime import read


class InterventionTests(unittest.TestCase):
    def test_four_conditions_replace_only_requested_components(self):
        a = np.arange(7, dtype=np.float32)/10
        original = a.copy()
        expected = [np.r_[np.zeros(6), -1], a, np.r_[a[:6], -1], np.r_[np.zeros(6), a[6]]]
        for name, value in zip(CONDITIONS, expected):
            actual = close_action(a, name)
            np.testing.assert_array_equal(actual, value)
            self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(a, original)
        with self.assertRaises(ValueError):close_action(np.zeros(7, np.float64), 'model')

    def fixture(self):
        cfg = dict(close_steps=8, hold_steps=20, minimum_peg_lift_m=.04, minimum_tcp_lift_m=.06)
        anchor = dict(peg_pose=np.r_[0,0,.02,1,0,0,0], ee_pose=np.r_[0,0,.02,1,0,0,0])
        def row(stage):return dict(stage=stage, after=dict(peg_pose=np.r_[0,0,.1,1,0,0,0],
            ee_pose=np.r_[0,0,.1,1,0,0,0], grasp=True, finger_forces=np.ones((2,3)),gripper_qpos=np.zeros(2)))
        rows=[row('close') for _ in range(8)]+[row('lift') for _ in range(20)]+[row('hold') for _ in range(20)]
        return cfg,anchor,rows

    def test_contact_on_table_or_failed_lift_cannot_count_as_retained(self):
        c,a,r=self.fixture()
        self.assertTrue(retention(r,a,c)['retained'])
        r[-1]['after']['peg_pose'][2]=.02
        self.assertFalse(retention(r,a,c)['retained'])
        c,a,r=self.fixture();r[-1]['after']['ee_pose'][2]=.02
        outcome=retention(r,a,c)
        self.assertFalse(outcome['retained']);self.assertFalse(outcome['lift_execution_valid'])

    def test_one_hold_contact_loss_fails_and_incomplete_run_has_no_verdict(self):
        c,a,r=self.fixture();r[-4]['after']['grasp']=False
        self.assertFalse(retention(r,a,c)['retained'])
        with self.assertRaises(ValueError):retention(r[:-1],a,c)

    def test_pair_rejects_wrong_observation_hidden_or_reset(self):
        c=dict(atol=1e-6,rtol=1e-5)
        a=dict(prefix=[dict(timestep=0,rgbd_calibration_robot_graph_mask_sha256='same',
            expert_action_sha256='a',action=[0]*7,hidden=[0]*144)],
            anchor={k:np.zeros(n) for k,n in [('peg_pose',7),('ee_pose',7),('gripper_qpos',2)]},reset_group='same')
        self.assertEqual(compare_prefix(a,a,c)['status'],'PASS')
        for key,value in [('hidden',[1]*144),('rgbd_calibration_robot_graph_mask_sha256','changed')]:
            b=deepcopy(a);b['prefix'][0][key]=value
            with self.assertRaises((ValueError,AssertionError)):compare_prefix(a,b,c)
        b=deepcopy(a);b['reset_group']='other'
        with self.assertRaises(ValueError):compare_prefix(a,b,c)

    def test_numpy_diagnostic_serialization_and_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json';atomic(p,dict(after=np.ones((2,3)),flag=np.bool_(True)))
            self.assertEqual(read(p),dict(after=[[1]*3,[1]*3],flag=True))
        c=read(Path(__file__).parents[1]/'grasp_intervention_config.json')
        validate_config(c);c['close_steps']=7
        with self.assertRaises(ValueError):validate_config(c)


if __name__ == '__main__':unittest.main()
