"""Small CPU integration checks against the installed simulator."""

import unittest

import torch

from envs.maniskill import get_gt_state, make_peg_insertion_env


class ManiSkillEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_peg_insertion_env(device="cpu", obs_mode="state")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_environment_creation(self):
        self.assertEqual(self.env.unwrapped.backend.sim_backend, "physx_cpu")
        self.assertEqual(self.env.unwrapped.control_mode, "pd_joint_delta_pos")

    def test_reset(self):
        obs, info = self.env.reset(seed=0)
        self.assertEqual(tuple(obs.shape), (1, 43))
        self.assertIn("success", info)

    def test_step(self):
        self.env.reset(seed=0)
        action = torch.zeros((1, self.env.action_space.shape[-1]))
        for _ in range(3):
            _, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual(tuple(reward.shape), (1,))
        self.assertEqual(tuple(terminated.shape), (1,))
        self.assertEqual(tuple(truncated.shape), (1,))
        self.assertIn("success", info)

    def test_gt_state(self):
        self.env.reset(seed=0)
        state = get_gt_state(self.env)
        self.assertEqual(tuple(state.qpos.shape), (1, 9))
        self.assertEqual(tuple(state.qvel.shape), (1, 9))
        self.assertEqual(tuple(state.ee_pose.shape), (1, 7))
        self.assertEqual(tuple(state.left_fingertip_pos.shape), (1, 3))
        self.assertEqual(tuple(state.peg_pose.shape), (1, 7))
        self.assertEqual(tuple(state.hole_pose.shape), (1, 7))
        self.assertEqual(tuple(state.success.shape), (1,))

    def test_rgb_and_render(self):
        env = make_peg_insertion_env(obs_mode="rgb", render_mode="rgb_array")
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(tuple(obs["sensor_data"]["base_camera"]["rgb"].shape), (1, 128, 128, 3))
            self.assertEqual(tuple(env.render().shape), (1, 512, 512, 3))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
