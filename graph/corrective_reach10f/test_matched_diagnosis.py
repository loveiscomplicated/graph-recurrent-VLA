"""Temporal counterexamples for the post-hoc Reach diagnosis."""
import json
import tempfile
import unittest
from pathlib import Path

from graph.corrective_reach10f.matched_diagnosis import diagnose


class TemporalDiagnosisTests(unittest.TestCase):
    def trace(self, pairs, unsafe_last=False):
        rows=[]
        for t,(p,r) in enumerate(pairs):
            valid=not (unsafe_last and t==len(pairs)-1)
            rows.append(dict(timestep=t,simulator_time_s=(t+1)/20,
                segment='expert_prefix' if t==0 else 'student_recovery',
                target_position_error_m=p,target_orientation_error_rad=r,
                approach_position_error_m=0 if t==0 else .05,
                approach_orientation_error_rad=0,opening_m=.08,peg_translation_m=0,
                physical_safety=valid,tcp_pose=[0,0,p,1,0,0,0],
                physical_details=dict(shape_violations=[] if valid else [dict(pair=['robot:panda_hand','peg_0'],gap_m=.0001,minimum_m=.0002)],
                    substep_contact_violations=[],joint_limit_violation=False,max_qvel=.1,
                    object_translation_m=dict(peg_pose=0,box_pose=0,hole_pose=0),
                    object_rotation_rad=dict(peg_pose=0,box_pose=0,hole_pose=0))))
        return rows

    def evaluate(self, rows):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            episode=dict(actions=len(rows),condition='R0',kind='handoff',seed=65000,requested_offset_mm=[5,0,0],
                first_approach_t=1,completion_t=None,corrective_reach_success=False,disturbance_ok=True,
                physical_safety=all(r['physical_safety'] for r in rows),reset_sha256='test',recovery_start_t=1,
                disturbance_steps=1,actual_offset_mm=[5,0,0],
                target_error_reduction_m=0,recovery_error_overshoot_m=0)
            (folder/'episode.json').write_text(json.dumps(episode));(folder/'trace.json').write_text(json.dumps(rows))
            (folder/'rollout.mp4').write_bytes(b'fixture-only')
            return diagnose(folder,'fixture',dict(safety=dict(max_joint_velocity_rad_s=5)))

    def test_independent_minima_do_not_imply_simultaneous_success(self):
        summary,timeline,_=self.evaluate(self.trace([(.05,0),(.003,.1),(.02,.01)]))
        self.assertLess(summary['minimum_position_error_mm'],4)
        self.assertLess(summary['minimum_orientation_error_rad'],.03)
        self.assertIsNone(summary['preclose_pose_first_post_t'])
        self.assertFalse(summary['diagnostic_without_safety'])
        self.assertEqual(timeline[1]['pre_state_t'],1)
        self.assertEqual(timeline[1]['post_state_t'],2)
        self.assertEqual(timeline[1]['pre_position_error_mm'],50)

    def test_safety_counterfactual_remains_separate_from_official(self):
        summary,_,safety=self.evaluate(self.trace([(.05,0),(.003,.02),(.003,.02),(.003,.02)],True))
        self.assertTrue(summary['diagnostic_without_safety'])
        self.assertFalse(summary['official_success'])
        self.assertEqual(summary['official_max_consecutive'],2)
        self.assertEqual(summary['first_safety_failure_post_t'],4)
        self.assertEqual(safety['events'][0]['substep_contact_violations'],[])
        self.assertIn('NOT_AVAILABLE',safety['transient_or_persistent'])

    def test_three_steps_required_even_when_pose_and_opening_pass(self):
        summary,_,_=self.evaluate(self.trace([(.05,0),(.003,.02),(.003,.02)]))
        self.assertEqual(summary['no_safety_max_consecutive'],2)
        self.assertFalse(summary['diagnostic_without_safety'])
        self.assertIn('STABILITY_FAILURE',summary['failure_categories'])


if __name__=='__main__':unittest.main()
