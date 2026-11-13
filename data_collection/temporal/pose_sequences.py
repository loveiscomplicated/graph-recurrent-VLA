"""Extend aligned frozen episodes with causal motion and optional raw proprioception."""

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.frozen_episodes import FrozenEpisodeDataset
from graph.pose_geometry import unit_quaternion
from graph.temporal.pose_observations import BASE_NAMES, ROBOT_NAMES, pose_input_step


def attach_proprioception(data, identity, raw_root):
    """Join observed TCP pose and finger qpos by exact published raw-frame checksum.

    Only observation/extra/tcp_pose and observation/agent/qpos are read. GT object
    poses, GT visibility and GT gripper fields are never used as input fallback.
    The cache already contains observed TCP XYZ; reject a raw/cache disagreement.
    """
    if raw_root is None:
        return data
    root = Path(raw_root).expanduser().resolve()
    hashes = {frame["id"]: frame["sha256"] for frame in identity["frames"]}
    poses, grippers = [], []
    for index, sample_id in enumerate(data["sample_ids"]):
        path = (root / str(sample_id)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"Missing raw proprioception frame: {path}")
        if file_sha256(path) != hashes.get(str(sample_id)):
            raise ValueError(f"Raw frame differs from published graph source: {sample_id}")
        with np.load(path, allow_pickle=False) as saved:
            required = ("observation/extra/tcp_pose", "observation/agent/qpos")
            if any(key not in saved for key in required):
                raise ValueError(f"{sample_id}: need observed tcp_pose and qpos; no GT fallback")
            pose, qpos = (saved[key].copy() for key in required)
        if (
            pose.shape != (1, 7)
            or qpos.shape != (1, 9)
            or not np.isfinite(pose).all()
            or not np.isfinite(qpos).all()
        ):
            raise ValueError(f"{sample_id}: expected finite tcp_pose[1,7] and panda qpos[1,9]")
        if not np.allclose(pose[0, :3], data["tcp_world"][index], atol=1e-6, rtol=0):
            raise ValueError(f"{sample_id}: observed TCP differs from cached TCP")
        poses.append(unit_quaternion(torch.tensor(pose[0, 3:], dtype=torch.float32)).numpy())
        grippers.append(qpos[0, -2:])
    return dict(
        data,
        tcp_quaternion=np.asarray(poses, dtype=np.float32),
        gripper_qpos=np.asarray(grippers, dtype=np.float32),
    )


class TemporalPoseDataset(FrozenEpisodeDataset):
    """Reuse full-episode validation; return separate observation and Object21 labels."""

    def __init__(self, data):
        super().__init__(data)
        count = len(data["frame"])
        self.with_robot = "tcp_quaternion" in data
        if self.with_robot != ("gripper_qpos" in data):
            raise ValueError("Observed TCP orientation and gripper fields must appear together")
        fields = [("pose_quality", 25)]
        if self.with_robot:
            fields.extend((("tcp_quaternion", 4), ("gripper_qpos", 2)))
        for name, width in fields:
            if data[name].shape != (count, width) or not np.isfinite(data[name]).all():
                raise ValueError(f"{name} must be finite [{count},{width}]")
        self.feature_names = BASE_NAMES + (ROBOT_NAMES if self.with_robot else ())

    def __getitem__(self, index):
        rows = self.indices[index]
        tensors = {
            name: torch.as_tensor(self.data[name][rows], dtype=torch.float32)
            for name in ("prediction", "target", "tcp_world", "pose_quality")
        }
        observations = []
        for local, row in enumerate(rows):
            robot = {}
            if self.with_robot:
                previous_quaternion = None
                if local:
                    previous_row = rows[local - 1]
                    previous_quaternion = torch.tensor(
                        self.data["tcp_quaternion"][previous_row : previous_row + 1]
                    )
                robot = {
                    "tcp_quaternion": torch.tensor(self.data["tcp_quaternion"][row : row + 1]),
                    "gripper_qpos": torch.tensor(self.data["gripper_qpos"][row : row + 1]),
                    "previous_quaternion": previous_quaternion,
                }
            observations.append(
                pose_input_step(
                    tensors["prediction"][local : local + 1],
                    tensors["pose_quality"][local : local + 1],
                    tensors["tcp_world"][local : local + 1],
                    tensors["tcp_world"][local - 1 : local] if local else None,
                    **robot,
                )[0]
            )
        target = tensors["target"].clone()
        target[:, :3] += tensors["tcp_world"]
        target[:, 10:13] += tensors["tcp_world"]
        return torch.stack(observations), target


def collate_pose_episodes(episodes):
    """Right-pad variable episodes and return a boolean supervision mask."""
    observations, targets = zip(*episodes)
    lengths = torch.tensor([len(value) for value in observations], dtype=torch.long)
    mask = torch.arange(int(lengths.max()))[None] < lengths[:, None]
    return (
        pad_sequence(observations, batch_first=True),
        pad_sequence(targets, batch_first=True),
        mask,
    )


def fit_observation_normalization(training_dataset):
    """Compute population mean/std on train observations only, never labels/padding."""
    count = 0
    total = torch.zeros(len(training_dataset.feature_names), dtype=torch.float64)
    squared = torch.zeros_like(total)
    for index in range(len(training_dataset)):
        observations, _ = training_dataset[index]
        observations = observations.double()
        count += len(observations)
        total += observations.sum(0)
        squared += observations.square().sum(0)
    mean = total / count
    variance = (squared / count - mean.square()).clamp_min(0)
    # Constant training features have unit scale: new nonzero observations remain bounded.
    std = torch.where(variance < 1e-12, 1.0, variance.sqrt())
    return mean.float(), std.float()


def stream_arguments(data, row, device):
    """Select only current observation fields; labels cannot enter streaming calls."""
    result = {
        name: torch.as_tensor(data[field][row : row + 1], dtype=torch.float32, device=device)
        for name, field in (
            ("object21", "prediction"),
            ("quality25", "pose_quality"),
            ("tcp_world", "tcp_world"),
        )
    }
    if "tcp_quaternion" in data:
        result.update(
            tcp_quaternion=torch.tensor(data["tcp_quaternion"][row : row + 1], device=device),
            gripper_qpos=torch.tensor(data["gripper_qpos"][row : row + 1], device=device),
        )
    return result
