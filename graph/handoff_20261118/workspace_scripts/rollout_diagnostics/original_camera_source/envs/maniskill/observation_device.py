"""Move raw reset/step observations without changing the simulator backend."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


def observation_to_device(observation, device):
    """Preserve the raw tree, shapes and dtypes; transfer each array once.

    Already resident tensors are returned as-is by Tensor.to. No NumPy/CPU
    materialization, feature computation or sensor-value conversion is added.
    """
    if isinstance(observation, torch.Tensor):
        return observation.to(device=device)
    if isinstance(observation, np.ndarray):
        return torch.as_tensor(observation, device=device)
    if isinstance(observation, dict):
        return {k: observation_to_device(v, device) for k, v in observation.items()}
    if isinstance(observation, tuple):
        return tuple(observation_to_device(v, device) for v in observation)
    if isinstance(observation, list):
        return [observation_to_device(v, device) for v in observation]
    return observation


class ObservationDevice(gym.ObservationWrapper):
    """Transfer only observations immediately after reset/step returns them."""

    def __init__(self, env, device):
        super().__init__(env)
        self.observation_device = torch.device(device)

    def observation(self, observation):
        return observation_to_device(observation, self.observation_device)
