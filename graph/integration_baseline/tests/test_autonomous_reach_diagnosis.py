"""Fixed causal boundary, geometry invariance and pair mismatch rejection."""
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
from graph.integration_baseline.autonomous_reach_diagnosis import (
    rotation_distance,relative_pose,box_surface_distance,pad_segment_intersects_peg,
    treatment_action,validate_close_event,validate_config,compare_prefix,CONDITIONS)
from graph.integration_baseline.direct_runtime import read


class AutonomousReachTests(unittest.TestCase):
    def config(self):
        return read(Path(__file__).parents[1]/'autonomous_reach_diagnosis_config.json')

    def test_protocol_rejects_seed_event_or_outcome_threshold_change(self):
        c=self.config();validate_config(c)
        for k,v in [('seed',41006),('model_close_timestep',28),('minimum_peg_lift_m',.03),('physical_atol',1e-5)]:
            bad=deepcopy(c);bad[k]=v
            with self.assertRaises(ValueError):validate_config(bad)

    def test_fixed_event_fails_without_moving_boundary(self):
        c=self.config();a=np.zeros(7,np.float32)
        a[6]=.1;validate_close_event(26,a,c)
        with self.assertRaises(ValueError):validate_close_event(27,a,c)
        a[6]=-.1;validate_close_event(27,a,c)
        with self.assertRaises(ValueError):validate_close_event(26,a,c)
        # Negative outputs after the prescribed intervention need not persist.
        a[6]=1;validate_close_event(28,a,c)

    def test_treatment_only_changes_B_and_positive_control(self):
        a=np.arange(7,dtype=np.float32)/10;original=a.copy()
        for c in CONDITIONS:
            np.testing.assert_array_equal(treatment_action(a,c),np.r_[np.zeros(6),-1] if c in ('B','positive_control') else a)
        np.testing.assert_array_equal(a,original)
        with self.assertRaises(ValueError):treatment_action(a.astype(np.float64),'A')

    def test_pose_distance_handles_quaternion_sign_and_known_rotation(self):
        q=np.array([1.,0,0,0]);self.assertEqual(rotation_distance(q,-q),0.)
        self.assertAlmostEqual(rotation_distance(q,[0,0,0,1]),np.pi)
        parent=np.r_[1,2,3,np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
        child=parent.copy();child[:3]+=[0,1,0]
        np.testing.assert_allclose(relative_pose(parent,child)[:3],[1,0,0],atol=1e-12)

    def test_surface_distance_and_between_pads_have_geometric_meaning(self):
        pose=np.r_[0,0,0,1,0,0,0];half=np.array([.03,.01,.01])
        np.testing.assert_allclose(box_surface_distance([[0,0,.05],[0,0,0]],pose,half),[.04,-.01])
        self.assertTrue(pad_segment_intersects_peg([[0,-.04,0],[0,.04,0]],pose,half))
        self.assertFalse(pad_segment_intersects_peg([[0,-.04,.05],[0,.04,.05]],pose,half))

    def test_prefix_rejects_physical_or_vision_mismatch(self):
        keys=('qpos','qvel','ee_pose','peg_pose','box_pose','left_finger_pose','right_finger_pose',
              'left_fingertip_pos','right_fingertip_pos','tcp_linear_velocity','tcp_angular_velocity',
              'peg_linear_velocity','peg_angular_velocity','box_linear_velocity','box_angular_velocity')
        a=dict(reset_group='same',prefix=[dict(timestep=0,rgbd_calibration_robot_graph_mask_sha256='same',
            expert_action_sha256='close_boundary',action=[0]*7,hidden=[0]*144,physical={k:[0]*3 for k in keys})])
        self.assertEqual(compare_prefix(a,a,self.config())['status'],'PASS')
        b=deepcopy(a);b['prefix'][0]['physical']['peg_pose'][0]=1e-4
        with self.assertRaises(AssertionError):compare_prefix(a,b,self.config())
        b=deepcopy(a);b['prefix'][0]['rgbd_calibration_robot_graph_mask_sha256']='wrong'
        with self.assertRaises(ValueError):compare_prefix(a,b,self.config())


if __name__=='__main__':unittest.main()


class ReachReportTests(unittest.TestCase):
    def test_vision_presence_can_coexist_with_wrong_identity(self):
        from graph.integration_baseline.autonomous_reach_report import vision_metrics
        masks=np.zeros((2,2,2),bool);masks[0,0,0]=True;masks[1,1,1]=True
        seg=np.array([[9,0],[0,8]])
        packet=dict(points=np.array([[0,0,0]],np.float32),object_id=np.array([0]),
                    provenance=dict(source_index=np.array([0])),intra_object_edge_index=np.zeros((2,0),np.int64))
        state=dict(ee_pose=[0,0,0,1,0,0,0],peg_pose=[0,0,0,1,0,0,0],peg_half_sizes=[1,1,1])
        v=vision_metrics(masks,seg,[8,9],packet,state)
        self.assertEqual(v['peg']['selected_point_count'],1)
        self.assertEqual(v['peg']['iou'],0)
        self.assertEqual(v['peg']['selected_on_correct_gt_fraction'],0)
        self.assertEqual(v['peg']['predicted_on_other_gt_pixels'],1)
