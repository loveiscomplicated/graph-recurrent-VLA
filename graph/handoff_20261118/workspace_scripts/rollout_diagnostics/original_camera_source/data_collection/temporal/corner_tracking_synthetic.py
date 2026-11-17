"""Small synthetic geometry fixtures, never a simulator or accuracy benchmark."""

from itertools import product

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash
from graph.temporal.corner_tracking.tracker import Observation
from graph.utils.object21_targets import object21_from_gt
from .corner_tracks import Episode, Frame, FORMAT


def synthetic_dataset(seed=19, frames=24, training_episodes=2):
    """Make noisy, reordered partial corners with disjoint synthetic parents."""
    groups = {split: [] for split in ("train", "calibration", "validation", "test")}
    signs = np.asarray(list(product((-1, 1), repeat=3)), dtype=np.float32)
    for number, split in enumerate(["train"] * training_episodes + ["calibration", "validation"]):
        rng = np.random.default_rng(seed + number)
        name = f"synthetic_episode_{number:03d}"
        phase = float(rng.uniform(-0.01, 0.01))
        rows = []
        for time in range(frames):
            peg = np.asarray([0.10 + 0.001 * time, phase, 0.1], dtype=np.float32)
            hole = np.asarray([0.25, phase + 0.02, 0.1], dtype=np.float32)
            half = np.asarray([0.05, 0.015, 0.015], dtype=np.float32)
            references = np.concatenate((peg + signs * half,
                                         hole + signs * 0.05,
                                         hole + signs * np.asarray([0.05, 0.018, 0.018]))).astype(np.float32)
            kinds = np.asarray([0] * 8 + [1] * 16, dtype=np.int64)
            visible = np.asarray([0, 2, 4, 6, 8, 10, 16, 18])
            if 10 <= time < 13:
                visible = visible[kinds[visible] != 0]
            rng.shuffle(visible)
            xyz = references[visible] + rng.normal(0, 0.0002, (len(visible), 3)).astype(np.float32)
            robot = np.asarray([*peg, 1, 0, 0, 0, 0.015, 0.015], dtype=np.float32)
            observation = Observation(xyz, kinds[visible], np.zeros(len(visible), dtype=np.float32),
                                      np.zeros(len(visible), dtype=bool), robot)
            gt = dict(ee_pose=torch.tensor(robot[:7])[None],
                      peg_pose=torch.tensor([*peg, 1, 0, 0, 0])[None],
                      hole_pose=torch.tensor([*hole, 1, 0, 0, 0])[None],
                      peg_half_sizes=torch.tensor(half)[None], hole_radius=torch.tensor([0.018]))
            target = object21_from_gt(gt)[0].numpy()
            rows.append(Frame(f"{name}/frame_{time:06d}.npz", observation, visible.copy(),
                              references, kinds, target))
        groups[split].append(Episode(name, name, split, rows))
    identity = dict(format=FORMAT, synthetic=True, seed=seed, frames=frames,
                    training_episodes=training_episodes, time_unit="saved_frame")
    identity["dataset_sha256"] = canonical_hash(identity)
    return groups, identity
