"""Episode splits, strict T/T+1 alignment and geometry-only dataset."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .geometry import GeometryState, inverse_transform, transform


PHASES = ("approach", "grasp", "transport", "alignment", "insertion")


def episode_split(ids, seed=0, groups=None):
    if len(set(ids)) != len(ids) or len(ids) < 10:
        raise ValueError("need >=10 unique episodes for nonempty 80/10/10 split")
    groups = ids if groups is None else groups
    if len(groups) != len(ids):
        raise ValueError("one group per episode required")
    unique_groups = list(dict.fromkeys(groups))
    if len(unique_groups) < 10:
        raise ValueError("need >=10 unique reset groups")
    order = np.random.default_rng(seed).permutation(unique_groups).tolist()
    nval = max(1, int(len(order)*.1))
    ntest = nval
    split_groups = {"train": order[:-nval-ntest], "validation": order[-nval-ntest:-ntest], "test": order[-ntest:]}
    return {split: [eid for group in selected for eid, label in zip(ids, groups) if label == group]
            for split, selected in split_groups.items()}


def state_at(group, index):
    return {k: state_at(v, index) if isinstance(v, h5py.Group) else np.array(v[index])
            for k, v in group.items()}


def validate_trajectory(group):
    actions = np.array(group["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("expected finite [T,7] pd_ee_delta_pose actions")
    if np.abs(actions).max() > 1.00001:
        raise ValueError("expert action exceeds normalized action space")
    obs = np.array(group["obs"])
    if obs.shape != (len(actions)+1, 43):
        raise ValueError("installed state observation must be [T+1,43]")
    lengths = []
    group["env_states"].visititems(lambda _, v: lengths.append(v.shape[0]) if isinstance(v, h5py.Dataset) else None)
    if not lengths or any(n != len(actions)+1 for n in lengths):
        raise ValueError("all env state tensors must have T+1 rows")
    return actions, obs


def reset_episode(env, episode):
    kwargs = json.loads(json.dumps(episode["reset_kwargs"]))
    seed = kwargs.get("seed", episode["episode_seed"])
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError("CPU extraction requires one episode seed")
        seed = seed[0]
    if seed != episode["episode_seed"]:
        raise ValueError("reset seed does not match episode metadata")
    kwargs["seed"] = seed
    kwargs.setdefault("options", {})["reconfigure"] = True
    return env.reset(**kwargs)


def phase_label(env, g):
    """Heuristic diagnostic only; never a policy feature or loss input."""
    # Restoring state does not restore PhysX contact caches. A geometric grasp
    # proxy keeps snapshot phase labels free of stale/absent contact reports.
    pads = torch.stack((g.left_fingertip, g.right_fingertip), 1)
    near = torch.cdist(pads, g.peg_points).min(-1).values
    grasped = bool((near.max() < .03) and (g.gripper.max() < .035))
    head_local = torch.zeros_like(g.peg_pose[:, None, :3])
    head_local[:, 0, 0] = g.peg_half_size[:, 0]
    head = transform(head_local, g.peg_pose)
    local = inverse_transform(head, g.hole_pose)[0, 0]
    transverse = float(local[1:].norm())
    depth = float(g.box_half_size[0, 0])
    if transverse < float(g.hole_radius[0, 0])*2 and float(local[0]) > -depth:
        return 4
    if grasped and transverse < .06 and float(local[0]) > -depth-.12:
        return 3
    if grasped and float(g.peg_pose[0, 2]) > float(g.peg_half_size[0, 2])+.03:
        return 2
    return 1 if grasped or float(near.mean()) < .055 else 0


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path):
    data = torch.load(path, weights_only=True)
    geometry = GeometryState(**data["geometry"])
    return data, geometry


def split_indices(data, split):
    ids = torch.tensor(data["manifest"]["splits"][split])
    return torch.where(torch.isin(data["episode_id"], ids))[0]
