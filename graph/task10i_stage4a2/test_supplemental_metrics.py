import unittest
import numpy as np
from .geometry_drift_audit import residuals
from .matching_phase_audit import phase_clock
from .final_dino_diagnostic import rectangle_support
from .recovery_event_audit import first_correct


class SupplementalMetricContracts(unittest.TestCase):
    def test_equal_world_motion_has_zero_centroid_step_residual(self):
        points=np.array([[0.,0.,0.],[1.,0.,0.]])
        _,previous=residuals(points,[True,True],points,t=0)
        row,_=residuals(points+[0,1,0],[True,True],points+[0,1,0],previous,1)
        self.assertEqual(row['visible_centroid_step_residual_m'],0)

    def test_wrong_surface_and_drifting_centroid_are_detected(self):
        points=np.array([[0.,0.,0.],[1.,0.,0.]])
        _,previous=residuals(points,[True,True],points,t=0)
        row,_=residuals(points+[0,1,0],[False,False],points,previous,1)
        self.assertEqual(row['all_point_surface_mean_m'],1)
        self.assertEqual(row['visible_centroid_step_residual_m'],1)
        self.assertIsNone(row['correct_point_surface_mean_m'])

    def test_no_surface_or_prediction_does_not_become_zero_error(self):
        for points,correct,reference in [([],[],[[0,0,0]]),([[0,0,0]],[True],[])]:
            row,state=residuals(points,correct,reference)
            self.assertIsNone(row['visible_centroid_error_m'])
            self.assertIsNone(state)

    def test_gap_does_not_bridge_temporal_drift(self):
        points=np.array([[0.,0.,0.]])
        _,old=residuals(points,[True],points,t=0)
        row,_=residuals(points+[1,0,0],[False],points,old,2)
        self.assertIsNone(row['visible_centroid_step_residual_m'])

    def test_native_grasp_is_after_action_not_before(self):
        values=phase_clock([dict(phase='close',physical=dict(is_grasping=True)),
                            dict(phase='lift',physical=dict(is_grasping=False))],3)
        self.assertIsNone(values[0]['native_grasp_at_frame'])
        self.assertTrue(values[1]['native_grasp_at_frame'])
        self.assertFalse(values[2]['native_grasp_at_frame'])
        self.assertEqual(values[0]['native_grasp_role'],'BEFORE_FIRST_OBSERVED_GRASP')

    def test_expert_phase_is_not_promoted_to_actual_grasp(self):
        values=phase_clock([dict(expert_phase='reach'),dict(expert_phase='close')],2)
        self.assertTrue(values[1]['expert_after_close_phase'])
        self.assertEqual(values[1]['native_grasp_role'],'GRASP_NOT_RECORDED')
        self.assertIsNone(values[1]['native_grasp_at_frame'])

    def test_rectangle_can_cover_target_while_being_mixed_with_robot(self):
        target=np.zeros((4,4),bool);target[1,1]=True
        robot=np.zeros_like(target);robot[1,2]=True
        rows=rectangle_support([[1,1,3,2],[-9,-9,-1,-1]],target,robot,np.zeros_like(target))
        self.assertEqual(rows[0]['target_coverage'],1)
        self.assertEqual(rows[0]['target_fraction'],.5)
        self.assertEqual(rows[0]['nonbackground_target_dominance'],.5)
        self.assertIsNone(rows[1]['target_fraction'])

    def test_recovery_outside_uncertain_interval_is_censored(self):
        rows=[dict(timestep=t,reference_kind='EXACT_GT',correct_mask_availability=int(t==5)) for t in range(6)]
        self.assertIsNone(first_correct(rows,1,4))
        self.assertEqual(first_correct(rows,1,6),5)


if __name__=='__main__':unittest.main()
