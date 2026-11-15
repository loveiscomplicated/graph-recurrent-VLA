"""CPU and Apple Silicon MPS checks for raw RoArm RGB-D observations."""

import unittest
from unittest.mock import patch

import torch

from envs.maniskill.roarm import make_roarm_env


class RoArmObservationTest(unittest.TestCase):
    def assert_raw_observation(self, observation, device):
        """Check the camera and state contract for a single default bench."""
        camera = observation["sensor_data"]["base_camera"]
        calibration = observation["sensor_param"]["base_camera"]
        agent = observation["agent"]
        fields = (
            (camera["rgb"], (1, 128, 128, 3), torch.uint8),
            (camera["depth"], (1, 128, 128, 1), torch.int16),
            (calibration["intrinsic_cv"], (1, 3, 3), torch.float32),
            (calibration["extrinsic_cv"], (1, 3, 4), torch.float32),
            (calibration["cam2world_gl"], (1, 4, 4), torch.float32),
            (agent["qpos"], (1, 4), torch.float32),
            (agent["qvel"], (1, 4), torch.float32),
            (observation["extra"]["tcp_pose"], (1, 7), torch.float32),
        )
        for tensor, shape, dtype in fields:
            self.assertEqual(tensor.device.type, device)
            self.assertEqual(tensor.shape, shape)
            self.assertEqual(tensor.dtype, dtype)

    def assert_observation_transfer(self, raw, moved):
        """Check exact value preservation across the raw observation tree."""
        if isinstance(raw, dict):
            self.assertEqual(raw.keys(), moved.keys())
            for key in raw:
                self.assert_observation_transfer(raw[key], moved[key])
        elif isinstance(raw, torch.Tensor):
            self.assertEqual(raw.device.type, "cpu")
            self.assertEqual(moved.device.type, "mps")
            torch.testing.assert_close(raw, moved.cpu(), rtol=0, atol=0, equal_nan=True)
        else:
            self.assertEqual(raw, moved)

    def test_default_cpu_reset_and_step(self):
        env = make_roarm_env(obs_mode="rgb+depth")
        self.addCleanup(env.close)
        observation, _ = env.reset(seed=0)
        self.assert_raw_observation(observation, "cpu")
        observation, *_ = env.step(torch.zeros((1, 4)))
        self.assert_raw_observation(observation, "cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_reset_and_step_preserve_raw_observations(self):
        env = make_roarm_env(obs_mode="rgb+depth", observation_device="mps")
        self.addCleanup(env.close)
        self.assertEqual(env.unwrapped.device.type, "cpu")
        action = torch.zeros((1, 4), device="cpu")
        with patch.object(env, "observation", wraps=env.observation) as transfer:
            observation, _ = env.reset(seed=0)
            self.assert_raw_observation(observation, "mps")
            self.assert_observation_transfer(transfer.call_args.args[0], observation)

            observation, reward, terminated, truncated, _ = env.step(action)
            self.assert_raw_observation(observation, "mps")
            self.assert_observation_transfer(transfer.call_args.args[0], observation)
            for tensor in (reward, terminated, truncated):
                self.assertEqual(tensor.device.type, "cpu")
