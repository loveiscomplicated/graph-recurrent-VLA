"""Integration checks for the guarded executor's supported scene."""

import unittest

import sapien
import torch

from envs.maniskill.roarm import make_roarm_env
from envs.maniskill.roarm.execution import RoArmGuardedExecutor
from roarm.collision import MotionRejected, RoArmCollisionModel
from roarm.motion import RoArmMotionPlanner


class RoArmExecutionTest(unittest.TestCase):
    def test_changed_table_is_rejected_before_stepping(self):
        env = make_roarm_env(control_mode="pd_joint_pos_guarded")
        try:
            env.reset(seed=0)
            guard = RoArmGuardedExecutor(env)
            elapsed = env.unwrapped.elapsed_steps.clone()
            env.unwrapped.table.set_pose(sapien.Pose([0, 0, 0.4]))
            with self.assertRaisesRegex(MotionRejected, "table geometry changed"):
                guard.execute([[0, 0, 1.57, 0.75], [0, 0, 1.57, 0.75]])
            self.assertTrue(torch.equal(elapsed, env.unwrapped.elapsed_steps))
            self.assertTrue(guard.fault)
        finally:
            env.close()


    def test_mismatched_static_box_is_rejected(self):
        planned = [dict(center=[0.31, 0, 0.25], half_size=[0.025, 0.035, 0.025])]
        actual = [dict(center=[0.31, 0, 0.25], half_size=[0.025, 0.035, 0.03])]
        env = make_roarm_env(control_mode="pd_joint_pos_guarded", obstacles=actual)
        self.addCleanup(env.close)
        env.reset(seed=0)
        planner = RoArmMotionPlanner(RoArmCollisionModel(obstacles=planned))

        with self.assertRaisesRegex(MotionRejected, "Static obstacle geometry changed"):
            RoArmGuardedExecutor(env, planner)
        self.assertEqual(env.unwrapped.elapsed_steps.item(), 0)


if __name__ == "__main__":
    unittest.main()
