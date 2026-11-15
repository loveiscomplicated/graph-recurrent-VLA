"""Bounded CPU/CUDA RoArm checks. Run each renderer/backend in a fresh process."""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image

from envs.maniskill.roarm import make_roarm_env
from examples.roarm_checks import save_frames, write_report
from roarm.assets import (
    HOME_QPOS,
    JOINT_NAMES,
    PHYSICAL_LINK_NAMES,
    load_collision_exclusions,
)
from roarm.kinematics import RoArmKinematics


def contact_forces(task):
    """Query enabled non-adjacent pairs plus moving links against table/ground."""
    links = task.agent.robot.links_map
    excluded = load_collision_exclusions()
    pairs = []
    for first, second in itertools.combinations(PHYSICAL_LINK_NAMES, 2):
        if frozenset((first, second)) not in excluded:
            pairs.append((first, second, links[first], links[second]))
    for name in PHYSICAL_LINK_NAMES[1:]:
        for surface in ("table", "ground"):
            pairs.append((name, surface, links[name], getattr(task, surface)))
    forces = {}
    for first_name, second_name, first, second in pairs:
        contact_force = task.scene.get_pairwise_contact_forces(first, second)
        forces[f"{first_name}/{second_name}"] = torch.linalg.vector_norm(
            contact_force, dim=-1
        )
    return forces


def set_test_configuration(task, qpos, mount_height=None):
    """Use BaseEnv's public state API so teleports are applied to CUDA PhysX too."""
    robot = task.agent.robot
    state = task.get_state_dict()
    articulation = state["articulations"][robot.name]
    # Installed Articulation state: xyz, wxyz, root velocities, qpos, qvel.
    articulation[:, 7:13] = 0
    articulation[:, 13:17] = qpos
    articulation[:, 17:21] = 0
    if mount_height is not None:
        articulation[:, 2] = mount_height
    task.set_state_dict(state)
    assert torch.allclose(robot.get_qpos(), qpos, atol=1e-6)
    if mount_height is not None:
        assert torch.allclose(
            robot.pose.p[:, 2], torch.full_like(robot.pose.p[:, 2], mount_height)
        )
    task.agent.controller.reset()


def validate_state(task):
    """Reject nonfinite states, runaway velocities and joint-stop penetration."""
    robot = task.agent.robot
    qpos, qvel, limits = robot.get_qpos(), robot.get_qvel(), robot.get_qlimits()
    assert torch.isfinite(qpos).all() and torch.isfinite(qvel).all()
    assert qvel.abs().max() < 100, "Runaway joint velocity"
    violation = torch.maximum(limits[:, :, 0] - qpos, qpos - limits[:, :, 1]).clamp_min(
        0
    )
    assert violation.max() < 0.015, f"Joint limit violation {violation.max().item()}"
    return float(violation.max())


def run_core(env, report, output):
    """Motion, reaching, partial resets, raw CUDA RGB-D and independent batched FK."""
    task = env.unwrapped
    robot = task.agent.robot
    batch = task.num_envs
    obs, _ = env.reset(seed=0)
    kinematics = RoArmKinematics()
    home = torch.as_tensor(HOME_QPOS, device=task.device).repeat(batch, 1)
    assert tuple(obs["agent"]["qpos"].shape) == (batch, 4)
    assert tuple(task.agent.tcp_pose.p.shape) == (batch, 3)
    images = []
    max_fk = 0.0
    max_contact = 0.0
    max_limit = 0.0
    steps = 0

    def check_camera(obs):
        values = obs["sensor_data"]["base_camera"]
        rgb, depth = values["rgb"], values["depth"]
        assert tuple(rgb.shape) == (batch, 128, 128, 3) and rgb.dtype == torch.uint8
        assert tuple(depth.shape) == (batch, 128, 128, 1) and depth.dtype == torch.int16
        expected_device = (
            "cuda" if task.backend.render_backend == "sapien_cuda" else "cpu"
        )
        assert (
            rgb.device.type == expected_device and depth.device.type == expected_device
        )
        assert rgb.float().std() > 1 and (depth > 0).any()
        report["native_camera"] = {
            "status": "PASS",
            "rgb_device": str(rgb.device),
            "depth_device": str(depth.device),
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
        }
        return rgb

    check_camera(obs)

    def move(command, count=50):
        nonlocal max_fk, max_contact, max_limit, steps
        for index in range(count):
            obs, *_ = env.step(command)
            steps += 1
            max_limit = max(max_limit, validate_state(task))
            if index % 10 == 0 or index == count - 1:
                rgb = check_camera(obs)
                images.append(Image.fromarray(rgb[0].cpu().numpy()))
                actual = (robot.pose.inv() * task.agent.tcp_pose).raw_pose.cpu().numpy()
                for qpos, pose in zip(robot.get_qpos().cpu().numpy(), actual):
                    expected = kinematics.forward(qpos)
                    error = float(np.linalg.norm(expected.p - pose[:3]))
                    max_fk = max(max_fk, error)
                    assert error < 1e-5
                    assert abs(float(np.dot(expected.q, pose[3:]))) > 1 - 1e-5
                max_contact = max(
                    max_contact,
                    max(float(f.max()) for f in contact_forces(task).values()),
                )
                assert max_contact < 1e-3, f"Unexpected path contact: {max_contact} N"
        return robot.get_qpos()

    error = float((move(home) - home).abs().max())
    assert error < 0.03
    report["gravity_hold"] = {
        "status": "PASS",
        "joint_error_rad": error,
        "gravity_enabled": all(
            not bool(link.disable_gravity.any()) for link in robot.links
        ),
    }
    assert report["gravity_hold"]["gravity_enabled"]
    errors = []
    for axis in range(4):
        for sign in (-1, 1):
            env.reset(seed=0)
            command = home.clone()
            # Distinct batch commands also test scene isolation.
            delta = sign * torch.linspace(0.08, 0.15, batch, device=task.device)
            command[:, axis] += delta
            actual = move(command)
            assert ((actual[:, axis] - home[:, axis]) * sign > 0).all()
            error = float((actual - command).abs().max())
            assert error < 0.03
            errors.append(error)
    report["joint_motion"] = {"status": "PASS", "max_error_rad": max(errors)}
    env.reset(seed=0)
    reaching = []
    for target in ([0.30, 0.06, 0.33], [0.25, -0.08, 0.38], [0.35, 0, 0.28]):
        command = kinematics.solve_position(target, HOME_QPOS)
        move(torch.as_tensor(command, device=task.device).repeat(batch, 1), count=70)
        actual = (robot.pose.inv() * task.agent.tcp_pose).p
        error = torch.linalg.vector_norm(
            actual - torch.tensor(target, device=task.device), dim=-1
        )
        assert error.max() < 0.015
        reaching.append({"target_base_m": target, "max_error_m": float(error.max())})
    report["reaching"] = {"status": "PASS", "trials": reaching}
    before_qpos = robot.get_qpos().clone()
    before_qvel = robot.get_qvel().clone()
    before_tcp = task.agent.tcp_pose.raw_pose.clone()
    if batch > 1:
        indices = torch.arange(0, batch, 2, device=task.device)
        untouched = torch.arange(1, batch, 2, device=task.device)
        env.reset(options={"env_idx": indices})
        assert torch.allclose(robot.get_qpos()[indices], home[indices], atol=1e-6)
        assert torch.count_nonzero(robot.get_qvel()[indices]) == 0
        assert torch.allclose(
            robot.get_qpos()[untouched], before_qpos[untouched], atol=1e-6
        )
        assert torch.allclose(
            robot.get_qvel()[untouched], before_qvel[untouched], atol=1e-6
        )
        assert torch.allclose(
            task.agent.tcp_pose.raw_pose[untouched], before_tcp[untouched], atol=1e-6
        )
        report["partial_reset"] = "PASS"
    else:
        report["partial_reset"] = "NOT APPLICABLE (single environment)"
    env.reset(seed=0)
    assert torch.allclose(robot.get_qpos(), home, atol=1e-6)
    assert torch.count_nonzero(robot.get_qvel()) == 0
    report["reset"] = "PASS"
    report["motion_physics"] = {
        "status": "PASS",
        "steps_per_env": steps,
        "max_fk_error_m": max_fk,
        "max_path_force_n": max_contact,
        "max_limit_violation_rad": max_limit,
    }
    save_frames(output, images, duration=120)


def run_limits(env, report, output):
    """Test all physical joint stops with collisions retained and table avoided."""
    task = env.unwrapped
    robot = task.agent.robot
    trials = []
    for axis in range(4):
        for side, direction in ((0, -1), (1, 1)):
            env.reset(seed=0)
            limits = robot.get_qlimits()
            qpos = torch.as_tensor(HOME_QPOS, device=task.device).repeat(
                task.num_envs, 1
            )
            qpos[:, axis] = limits[:, axis, side]
            set_test_configuration(task, qpos, mount_height=2)
            action = torch.zeros_like(qpos)
            action[:, axis] = direction
            max_violation = 0.0
            forces = {}
            for _ in range(20):
                env.step(action)
                max_violation = max(max_violation, validate_state(task))
                for name, force in contact_forces(task).items():
                    forces[name] = max(forces.get(name, 0.0), float(force.max()))
            gap = float((robot.get_qpos()[:, axis] - limits[:, axis, side]).abs().max())
            trials.append(
                {
                    "joint": JOINT_NAMES[axis],
                    "side": "lower" if side == 0 else "upper",
                    "status": "PASS" if gap < 0.03 else "BLOCKED",
                    "max_stop_gap_rad": gap,
                    "max_violation_rad": max_violation,
                    "contacts_n": {
                        name: value for name, value in forces.items() if value > 1e-4
                    },
                }
            )
    report["joint_stops"] = trials
    report["stop_bounds"] = "PASS"
    report["stop_reaching"] = (
        "PASS" if all(t["status"] == "PASS" for t in trials) else "BLOCKED"
    )


def run_workspace(env, report, output, grid_size):
    """Contact survey of a finite joint grid, not a collision-free motion planner."""
    task = env.unwrapped
    robot = task.agent.robot
    env.reset(seed=0)
    limits = robot.get_qlimits()[0].cpu().numpy()
    axes = [np.linspace(lo + 0.01, hi - 0.01, grid_size) for lo, hi in limits[:3]]
    samples = list(itertools.product(*axes))
    records = []
    for start in range(0, len(samples), task.num_envs):
        arm = samples[start : start + task.num_envs]
        valid = len(arm)
        arm += [arm[-1]] * (task.num_envs - valid)
        qpos = torch.tensor(
            [list(q) + [0.75] for q in arm], device=task.device, dtype=torch.float32
        )
        env.reset(seed=0)
        set_test_configuration(task, qpos)
        env.step(qpos)
        # Teleported penetrating poses deliberately exercise collision response;
        # record deviations without claiming those poses form safe trajectories.
        actual = robot.get_qpos()
        assert torch.isfinite(actual).all() and torch.isfinite(robot.get_qvel()).all()
        forces = contact_forces(task)
        for index in range(valid):
            records.append(
                {
                    "qpos_rad": qpos[index].cpu().tolist(),
                    "contacts_n": {
                        name: float(force[index])
                        for name, force in forces.items()
                        if float(force[index]) > 1e-4
                    },
                }
            )
    (output / "workspace_samples.json").write_text(json.dumps(records, indent=2) + "\n")
    report["workspace_survey"] = {
        "status": "PASS",
        "grid_size": grid_size,
        "sample_count": len(records),
        "contact_samples": sum(bool(row["contacts_n"]) for row in records),
        "contact_pairs": sorted(
            {name for row in records for name in row["contacts_n"]}
        ),
        "all_workspace_safe": "NOT PROVEN",
        "scope": "instantaneous contacts after one control step; not swept-path coverage",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--render-backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--check", choices=["core", "limits", "workspace"], default="core"
    )
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.grid_size < 2 or args.grid_size > 15:
        parser.error("grid-size must be between 2 and 15")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "RUNNING",
        "arguments": {**vars(args), "output": str(args.output)},
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("mani-skill", "sapien", "torch", "gymnasium", "pin", "numpy")
        },
        "hardware": "NOT TESTED",
        "grasping": "NOT TESTED",
        "payload_fidelity": "NOT TESTED",
    }
    env = None
    start = time.monotonic()
    try:
        env = make_roarm_env(
            device=args.device,
            num_envs=args.num_envs,
            render_backend=args.render_backend,
            obs_mode="rgb+depth" if args.check == "core" else "state_dict",
            control_mode="pd_joint_delta_pos"
            if args.check == "limits"
            else "pd_joint_pos",
        )
        task = env.unwrapped
        report["backend"] = {
            "physics": task.backend.sim_backend,
            "renderer": task.backend.render_backend,
            "device": str(task.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if args.check == "core":
            run_core(env, report, args.output)
        elif args.check == "limits":
            run_limits(env, report, args.output)
        else:
            run_workspace(env, report, args.output, args.grid_size)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        try:
            if env is not None:
                env.close()
        finally:
            write_report(args.output, report, display=True)


if __name__ == "__main__":
    main()
