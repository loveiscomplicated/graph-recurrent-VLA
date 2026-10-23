"""Collect legacy CPU reset fixtures in a dedicated process."""

import argparse
import dataclasses
import time
from pathlib import Path

import torch

from envs.maniskill import get_gt_state
from recurrent.common import make_env, versions, environment_record, HORIZON
from recurrent.execution import writer_lock
from recurrent.integrity import package_versions


def tree_map(fn, obj):
    return (
        {k: tree_map(fn, v) for k, v in obj.items()}
        if isinstance(obj, dict)
        else fn(obj)
    )


def validate(path, seeds, horizon=HORIZON):
    seeds = list(seeds)
    fixtures = torch.load(path, map_location="cpu", weights_only=False)
    if (fixtures.get("source") != "legacy physx_cpu reset(seed)"
            or fixtures.get("environment") != environment_record(horizon)
            or fixtures.get("seeds") != list(seeds)
            or set(fixtures.get("episodes", {})) != set(seeds)
            or any(fixtures.get("versions", {}).get(k) != v for k, v in package_versions().items())):
        raise ValueError(f"CPU fixture seed/environment/package identity mismatch: {path}")
    required = {"qpos": 9, "qvel": 9, "ee_pose": 7, "left_finger_pose": 7,
                "right_finger_pose": 7, "left_fingertip_pos": 3, "right_fingertip_pos": 3,
                "peg_pose": 7, "box_pose": 7, "hole_pose": 7, "gripper_qpos": 2,
                "peg_half_sizes": 3, "hole_radius": None, "success": None}
    for entry in fixtures["episodes"].values():
        if set(entry) != {"gt", "state", "hole_offset"} or not entry["state"]:
            raise ValueError("Invalid fixture state")
        for key, dim in required.items():
            tensor = entry["gt"][key]
            shape = (1, dim) if dim is not None else (1,)
            if tensor.shape != shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid fixture GT {key}")
        if entry["hole_offset"].shape != (1, 7):
            raise ValueError("Invalid fixture hole offset")
        def check(tensor):
            if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != 1 or not torch.isfinite(tensor).all():
                raise ValueError("Invalid fixture tensor")
            return tensor
        tree_map(check, entry["state"])
        check(entry["hole_offset"])
    return fixtures


def collect(path, seeds, horizon=HORIZON):
    seeds = list(seeds)
    path = Path(path)
    with writer_lock(path.with_name(path.name + ".lock")):
        if path.exists():
            validate(path, seeds, horizon)
            return
        _collect(path, seeds, horizon)
        validate(path, seeds, horizon)


def _collect(path, seeds, horizon):
    start = time.perf_counter()
    env = make_env(horizon)
    states = {}
    try:
        if env.unwrapped.backend.sim_backend != "physx_cpu" or str(env.unwrapped.device) != "cpu":
            raise RuntimeError("Fixture collection requires actual CPU physics")
        for seed in seeds:
            env.reset(seed=seed)
            task = env.unwrapped
            states[seed] = {
                "state": tree_map(
                    lambda x: x.detach().cpu().clone(), task.get_state_dict()
                ),
                "gt": {
                    f.name: getattr(get_gt_state(env), f.name).cpu().clone()
                    for f in dataclasses.fields(get_gt_state(env))
                },
                "hole_offset": task.box_hole_offsets.raw_pose.cpu().clone(),
            }
    finally:
        env.close()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(".tmp.pt")
    torch.save(
        {
            "versions": versions(),
            "environment": environment_record(horizon),
            "seeds": seeds,
            "source": "legacy physx_cpu reset(seed)",
            "collection_s": time.perf_counter() - start,
            "episodes": states,
        },
        temporary,
    )
    temporary.replace(path)


def restore(env, fixtures, seeds):
    task = env.unwrapped
    env.reset(seed=list(seeds), options={"reconfigure": True})
    entries = [fixtures["episodes"][s] for s in seeds]

    def stack(path, nodes):
        if isinstance(nodes[0], dict):
            return {k: stack(path + [k], [n[k] for n in nodes]) for k in nodes[0]}
        return torch.cat(nodes, 0).to(task.device)

    # Geometry must agree before state restoration.
    for name, actual in [
        ("peg_half_sizes", task.peg_half_sizes),
        ("hole_radius", task.box_hole_radii),
        ("hole_offset", task.box_hole_offsets.raw_pose),
    ]:
        expected = torch.cat(
            [
                e["hole_offset"] if name == "hole_offset" else e["gt"][name]
                for e in entries
            ]
        ).to(task.device)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    task.set_state_dict(stack([], [e["state"] for e in entries]))
    gt = get_gt_state(env)
    errors = {}
    for field in dataclasses.fields(gt):
        name = field.name
        actual = getattr(gt, name)
        expected = torch.cat([e["gt"][name] for e in entries]).to(actual.device)
        if actual.dtype == torch.bool:
            continue
        if name.endswith("pose"):
            torch.testing.assert_close(
                actual[:, :3], expected[:, :3], atol=1e-5, rtol=0
            )
            # Relative rotation angle; abs(w) identifies q with -q.
            aq = torch.nn.functional.normalize(actual[:, 3:], dim=-1)
            eq = torch.nn.functional.normalize(expected[:, 3:], dim=-1)
            scalar = (aq * eq).sum(-1).abs()
            vector = (
                -aq[:, :1] * eq[:, 1:]
                + eq[:, :1] * aq[:, 1:]
                - torch.linalg.cross(aq[:, 1:], eq[:, 1:], dim=-1)
            )
            angle = 2 * torch.atan2(vector.norm(dim=-1), scalar)
            if (angle > 1e-5).any():
                raise AssertionError(f"{name} rotation differs (radians): {angle}")
            errors[name + "_rotation_rad"] = float(angle.max())
            errors[name] = float((actual[:, :3] - expected[:, :3]).abs().max())
        else:
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
            errors[name] = float((actual - expected).abs().max())
    return errors


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--start-seed", type=int, default=2000)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--horizon", type=int, default=HORIZON)
    a = p.parse_args()
    collect(a.out, range(a.start_seed, a.start_seed + a.episodes), a.horizon)
