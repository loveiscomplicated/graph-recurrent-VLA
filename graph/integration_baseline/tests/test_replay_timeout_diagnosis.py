"""Diagnostic math only; no simulator fixtures or production-policy changes."""
import unittest
import numpy as np
from graph.integration_baseline.diagnose_replay_timeout import summarize


class DiagnosticSummaryTests(unittest.TestCase):
    def test_exact_joint_state_does_not_invent_correlation(self):
        ref=np.zeros((4,43),np.float32);ref[:,21]=1
        report=summarize(ref,ref)
        self.assertIsNone(report['qpos_qvel_max_error_correlation'])
        self.assertIsNone(report['first_failure_timestep'])
        self.assertEqual(report['tcp_orientation_rad']['max'],0)

    def test_failure_reports_observation_and_preceding_action(self):
        ref=np.zeros((6,43),np.float32);ref[:,21]=1
        actual=ref.copy();actual[4,14]=.0011
        report=summarize(actual,ref)
        self.assertEqual(report['failure']['observation_t'],4)
        self.assertEqual(report['failure']['action_index'],3)
        self.assertEqual(report['qvel']['max_location'],[4,5])
        self.assertEqual(report['qvel_by_joint'][5]['units'],'rad/s')
        self.assertEqual(report['qvel_by_joint'][7]['units'],'m/s')

    def test_quaternion_sign_is_same_orientation(self):
        ref=np.zeros((3,43),np.float32);ref[:,21]=1
        actual=ref.copy();actual[:,21]=-1
        report=summarize(actual,ref)
        self.assertEqual(report['tcp_orientation_rad']['max'],0)
        self.assertEqual(report['tcp_quaternion_component']['max'],2)


if __name__=='__main__':unittest.main()
