"""Small CPU integration checks against the installed simulator."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from envs.maniskill import get_gt_state, make_peg_insertion_env


class FactoryBackendTest(unittest.TestCase):
    def test_cpu_simulation_uses_cpu_renderer_when_torch_sees_cuda(self):
        with patch("envs.maniskill.peg_insertion.torch.cuda.is_available", return_value=True), \
             patch("envs.maniskill.peg_insertion.gym.make") as make:
            make_peg_insertion_env(device="cpu", obs_mode="state")
        self.assertEqual(make.call_args.kwargs["sim_backend"], "physx_cpu")
        self.assertEqual(make.call_args.kwargs["render_backend"], "cpu")


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
        env = make_peg_insertion_env(obs_mode="rgb+depth", render_mode="rgb_array")
        try:
            obs, _ = env.reset(seed=0)
            camera = obs["sensor_data"]["base_camera"]
            params = obs["sensor_param"]["base_camera"]
            self.assertEqual(tuple(camera["rgb"].shape), (1, 512, 512, 3))
            self.assertEqual(tuple(camera["depth"].shape), (1, 512, 512, 1))
            self.assertEqual(camera["rgb"].dtype, torch.uint8)
            self.assertEqual(camera["depth"].dtype, torch.int16)
            transform = params["cam2world_gl"][0].cpu().numpy()
            eye = np.array([0.65, -0.30, 0.65])
            target = np.array([0.065, 0.297, 0.11])
            direction = (target - eye) / np.linalg.norm(target - eye)
            np.testing.assert_allclose(transform[:3, 3], eye, atol=1e-6)
            np.testing.assert_allclose(-transform[:3, 2], direction, atol=1e-6)
            intrinsic = params["intrinsic_cv"][0].cpu().numpy()
            np.testing.assert_allclose([intrinsic[0, 0], intrinsic[1, 1]], [256, 256], atol=1e-4)
            self.assertEqual(tuple(env.render().shape), (1, 512, 512, 3))
        finally:
            env.close()

    def test_camera_overrides(self):
        configs = {"width": 96, "base_camera": {"height": 64}}
        env = make_peg_insertion_env(obs_mode="rgb", sensor_configs=configs)
        try:
            obs, _ = env.reset(seed=0)
            self.assertEqual(tuple(obs["sensor_data"]["base_camera"]["rgb"].shape), (1, 64, 96, 3))
            self.assertEqual(tuple(obs["sensor_data"]["hand_camera"]["rgb"].shape), (1, 128, 96, 3))
            self.assertEqual(configs, {"width": 96, "base_camera": {"height": 64}})
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
