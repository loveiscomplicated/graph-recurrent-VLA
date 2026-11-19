"""Mamba controller and feature contract for the frozen Surface frontend."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


OBSERVATION_DIM = 44
ACTION_DIM = 7
STATE36_DIM = 36
DEFAULT_MODEL_CONFIG = {
    "d_model": 96,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "layers": 2,
}


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """Return the conjugate of wxyz quaternions."""
    return quaternion * np.asarray([1, -1, -1, -1], dtype=quaternion.dtype)


def quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate row-vector coordinates with wxyz quaternions."""
    scalar, xyz = quaternion[..., :1], quaternion[..., 1:]
    cross = 2.0 * np.cross(xyz, vector)
    return vector + scalar * cross + np.cross(xyz, cross)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply wxyz quaternions."""
    aw, ax, ay, az = np.moveaxis(left, -1, 0)
    bw, bx, by, bz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized wxyz quaternions to canonical rotation vectors."""
    quaternion = np.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    sine = np.linalg.norm(quaternion[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sine, quaternion[..., :1])
    return quaternion[..., 1:] / np.maximum(sine, 1e-9) * angle


def quaternion_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    """Return the first two columns of the quaternion rotation matrix."""
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("State36 contains a zero-norm quaternion")
    w, x, y, z = np.moveaxis(quaternion / norm, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
            2 * (x * y - w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z + w * x),
        ),
        axis=-1,
    )


def state36_to_controller_features(state36: np.ndarray) -> np.ndarray:
    """Convert canonical ``[Robot15, Object21]`` rows to the 44D controller input."""
    state36 = np.asarray(state36, dtype=np.float32)
    if state36.ndim != 2 or state36.shape[1] != STATE36_DIM:
        raise ValueError("state36 must be a finite float array with shape [T,36]")
    if not np.isfinite(state36).all():
        raise ValueError("state36 contains NaN or infinity")

    base = np.concatenate(
        (state36[:, :3], quaternion_to_rotation_6d(state36[:, 3:7]), state36[:, 7:]),
        axis=-1,
    )
    object21 = state36[:, 15:]
    peg_quaternion = object21[:, 3:7]
    tip_relative_hole = object21[:, 7:10]
    hole_quaternion = object21[:, 13:17]
    peg_quaternion = peg_quaternion / np.linalg.norm(peg_quaternion, axis=-1, keepdims=True)
    hole_quaternion = hole_quaternion / np.linalg.norm(hole_quaternion, axis=-1, keepdims=True)
    tip_in_hole = quaternion_rotate(quaternion_conjugate(hole_quaternion), tip_relative_hole)
    relative_rotation = quaternion_to_rotation_vector(
        quaternion_multiply(quaternion_conjugate(hole_quaternion), peg_quaternion)
    )
    features = np.concatenate((base, tip_in_hole, relative_rotation), axis=-1)
    if features.shape[1] != OBSERVATION_DIM or not np.isfinite(features).all():
        raise ValueError("State36 conversion did not produce finite 44D features")
    return features.astype(np.float32, copy=False)


def mix_state36(ground_truth: np.ndarray, graph: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate a whole episode between GT and Graph observations."""
    ground_truth = np.asarray(ground_truth, dtype=np.float32)
    graph = np.asarray(graph, dtype=np.float32)
    if ground_truth.shape != graph.shape or ground_truth.ndim != 2 or ground_truth.shape[1] != 36:
        raise ValueError("GT and Graph state arrays must have matching shape [T,36]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    return (ground_truth + np.float32(alpha) * (graph - ground_truth)).astype(np.float32)


def compute_normalization(
    episodes: list[tuple[np.ndarray, np.ndarray]],
    *,
    standard_deviation_floor: float = 0.01,
    constant_threshold: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Fit observation/action statistics from training episodes only."""
    if not episodes:
        raise ValueError("At least one training episode is required")
    observations = np.concatenate([episode[0] for episode in episodes])
    actions = np.concatenate([episode[1] for episode in episodes])[:, :6]
    raw_std = observations.std(axis=0)
    action_std = actions.std(axis=0)
    return {
        "obs_mean": observations.mean(axis=0).astype(np.float32),
        "obs_std": np.maximum(raw_std, standard_deviation_floor).astype(np.float32),
        "obs_keep": (raw_std >= constant_threshold).astype(np.float32),
        "act_mean": actions.mean(axis=0).astype(np.float32),
        "act_std": (action_std + 1e-6).astype(np.float32),
    }


def normalize_episode(
    observations: np.ndarray,
    actions: np.ndarray,
    statistics: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize one feature/action sequence and encode the gripper target."""
    normalized_observation = (
        (observations - statistics["obs_mean"])
        / statistics["obs_std"]
        * statistics["obs_keep"]
    ).astype(np.float32)
    normalized_action = (
        (actions[:, :6] - statistics["act_mean"]) / statistics["act_std"]
    ).astype(np.float32)
    gripper = (actions[:, 6] > 0).astype(np.float32)
    return normalized_observation, normalized_action, gripper


def pad_episodes(
    episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad complete episodes for behavior-cloning updates."""
    if not episodes:
        raise ValueError("Cannot pad an empty episode batch")
    maximum_length = max(len(observation) for observation, _, _ in episodes)
    batch_size = len(episodes)
    observation_dim = episodes[0][0].shape[1]
    observation = np.zeros((batch_size, maximum_length, observation_dim), np.float32)
    action = np.zeros((batch_size, maximum_length, 6), np.float32)
    gripper = np.zeros((batch_size, maximum_length), np.float32)
    mask = np.zeros((batch_size, maximum_length), np.float32)
    for index, (episode_observation, episode_action, episode_gripper) in enumerate(episodes):
        length = len(episode_observation)
        observation[index, :length] = episode_observation
        action[index, :length] = episode_action
        gripper[index, :length] = episode_gripper
        mask[index, :length] = 1
    return tuple(torch.from_numpy(value).to(device) for value in (observation, action, gripper, mask))


def behavior_cloning_loss(
    prediction: torch.Tensor,
    action: torch.Tensor,
    gripper: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Combine normalized six-axis MSE and binary gripper cross entropy."""
    if prediction.shape[:-1] != action.shape[:-1] or prediction.shape[-1] != ACTION_DIM:
        raise ValueError("prediction and action batch shapes differ")
    motion_loss = ((prediction[..., :6] - action) ** 2).mean(dim=-1)
    gripper_loss = F.binary_cross_entropy_with_logits(
        prediction[..., 6], gripper, reduction="none"
    )
    denominator = mask.sum()
    if denominator <= 0:
        raise ValueError("Behavior-cloning batch contains no valid timesteps")
    return ((motion_loss + gripper_loss) * mask).sum() / denominator


class MambaBlock(nn.Module):
    """Small reference Mamba block used by the established recurrent prototype."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        inner_width = expand * d_model
        self.inner_width = inner_width
        self.state_width = d_state
        self.convolution_width = d_conv
        self.time_rank = math.ceil(d_model / 16)
        self.norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(d_model, 2 * inner_width)
        self.convolution = nn.Conv1d(
            inner_width, inner_width, d_conv, groups=inner_width
        )
        self.state_projection = nn.Linear(
            inner_width, self.time_rank + 2 * d_state, bias=False
        )
        self.time_projection = nn.Linear(self.time_rank, inner_width)
        state_indices = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
            inner_width, 1
        )
        self.log_transition = nn.Parameter(torch.log(state_indices))
        self.skip = nn.Parameter(torch.ones(inner_width))
        self.output_projection = nn.Linear(inner_width, d_model)

    def forward(self, values: torch.Tensor, state: Any = None):
        batch_size, length, _ = values.shape
        projected = self.input_projection(self.norm(values))
        input_values, gate = projected.chunk(2, dim=-1)
        input_values = input_values.transpose(1, 2)
        if state is None:
            convolution_buffer = input_values.new_zeros(
                batch_size, self.inner_width, self.convolution_width - 1
            )
            recurrent_state = input_values.new_zeros(
                batch_size, self.inner_width, self.state_width
            )
        else:
            convolution_buffer, recurrent_state = state
        convolution_sequence = torch.cat((convolution_buffer, input_values), dim=-1)
        input_values = F.silu(self.convolution(convolution_sequence)).transpose(1, 2)
        time_step, input_state, output_state = self.state_projection(input_values).split(
            (self.time_rank, self.state_width, self.state_width), dim=-1
        )
        time_step = F.softplus(self.time_projection(time_step))
        transition = -self.log_transition.exp()
        outputs = []
        for timestep in range(length):
            discrete_transition = torch.exp(
                time_step[:, timestep, :, None] * transition
            )
            recurrent_state = (
                discrete_transition * recurrent_state
                + time_step[:, timestep, :, None]
                * input_state[:, timestep, None, :]
                * input_values[:, timestep, :, None]
            )
            outputs.append(
                (recurrent_state * output_state[:, timestep, None, :]).sum(dim=-1)
            )
        output = (torch.stack(outputs, dim=1) + input_values * self.skip) * F.silu(gate)
        next_state = (
            convolution_sequence[..., -(self.convolution_width - 1) :],
            recurrent_state,
        )
        return values + self.output_projection(output), next_state


class MambaController(nn.Module):
    """44D recurrent controller with a normalized seven-action head."""

    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        d_model: int = 96,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layers: int = 2,
    ):
        super().__init__()
        self.encoder = nn.Linear(observation_dim, d_model)
        self.blocks = nn.ModuleList(
            MambaBlock(d_model, d_state, d_conv, expand) for _ in range(layers)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )

    def forward(self, observation: torch.Tensor, state: Any = None):
        hidden = self.encoder(observation)
        next_state = []
        for index, block in enumerate(self.blocks):
            hidden, block_state = block(
                hidden, None if state is None else state[index]
            )
            next_state.append(block_state)
        return self.head(hidden), next_state


def detach_recurrent_state(state: Any) -> Any:
    """Detach arbitrarily nested recurrent tensors between online steps."""
    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    return type(state)(detach_recurrent_state(item) for item in state)


@dataclass
class ControllerCheckpoint:
    """Validated runtime contents of one trained controller checkpoint."""

    model: MambaController
    statistics: dict[str, np.ndarray]
    seed: int
    path: Path


def load_controller_checkpoint(path: str | Path, device: torch.device) -> ControllerCheckpoint:
    """Load one checkpoint without accepting an incompatible feature contract."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "surface_mamba_controller_v1":
        raise ValueError(f"Unsupported controller checkpoint format: {path}")
    if payload.get("observation_dim") != OBSERVATION_DIM:
        raise ValueError(f"Controller checkpoint does not use 44D observations: {path}")
    model = MambaController(**payload["model_config"]).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    statistics = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in payload["statistics"].items()
    }
    return ControllerCheckpoint(model, statistics, int(payload["seed"]), path)


class OnlineController:
    """Stateful decoder for one Mamba checkpoint."""

    def __init__(self, checkpoint: ControllerCheckpoint):
        self.checkpoint = checkpoint
        self.device = next(checkpoint.model.parameters()).device
        self.reset()

    def reset(self) -> None:
        self.state = None

    @torch.inference_mode()
    def act(self, state36: np.ndarray) -> np.ndarray:
        state36 = np.asarray(state36, dtype=np.float32).reshape(1, STATE36_DIM)
        features = state36_to_controller_features(state36)
        stats = self.checkpoint.statistics
        normalized = (
            (features - stats["obs_mean"]) / stats["obs_std"] * stats["obs_keep"]
        ).astype(np.float32)
        tensor = torch.from_numpy(normalized)[None].to(self.device)
        prediction, self.state = self.checkpoint.model(tensor, self.state)
        self.state = detach_recurrent_state(self.state)
        prediction = prediction[0, 0].float().cpu().numpy()
        motion = prediction[:6] * stats["act_std"] + stats["act_mean"]
        gripper = 1.0 if prediction[6] > 0 else -1.0
        return np.clip(np.concatenate((motion, [gripper])), -1, 1).astype(np.float32)


class ControllerEnsemble:
    """Five independent recurrent controllers with action-space aggregation."""

    def __init__(self, controllers: list[OnlineController]):
        if len(controllers) != 5:
            raise ValueError("The primary ensemble requires exactly five controllers")
        self.controllers = controllers

    def reset(self) -> None:
        for controller in self.controllers:
            controller.reset()

    def act(self, state36: np.ndarray) -> np.ndarray:
        actions = np.stack([controller.act(state36) for controller in self.controllers])
        motion = actions[:, :6].mean(axis=0)
        gripper = 1.0 if np.count_nonzero(actions[:, 6] > 0) >= 3 else -1.0
        return np.concatenate((motion, [gripper])).astype(np.float32)
