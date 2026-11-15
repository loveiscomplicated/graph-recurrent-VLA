"""Reproducible CPU asset, motion, FK, reaching and collision checks; no training."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import sapien
import torch
from PIL import Image

from envs.maniskill.roarm import make_roarm_env
from examples.roarm_checks import save_frames, write_report
from roarm.assets import HOME_QPOS, JOINT_NAMES, audit_assets
from roarm.kinematics import RoArmKinematics


def check_loading(env, observation):
    """Check joint order, observation shape and live render/collision geometry."""
    robot = env.unwrapped.agent.robot
    assert [j.name for j in robot.active_joints] == JOINT_NAMES
    assert observation["agent"]["qpos"].shape == (1, 4)
    assert len(robot.links) == 6 and env.action_space.shape == (4,)
    # Geometry must be present in PhysX and the renderer, not only the XML.
    for link in robot.links:
        if link.name != "hand_tcp":
            assert len(link._objs[0].collision_shapes) > 0
            assert link._objs[0].entity.find_component_by_type(
                sapien.render.RenderBodyComponent
            )
    assert all(not bool(link.disable_gravity.any()) for link in robot.links)
    return {
        "status": "PASS",
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "gravity_enabled": True,
    }


def check_joint_motion(env, move):
    """Exercise positive and negative commands on each of the four axes."""
    motion = []
    for index in range(4):
        for delta in (-0.15, 0.15):
            env.reset(seed=0)
            target = HOME_QPOS.copy()
            target[index] += delta
            actual = move(target)
            assert (actual[index] - HOME_QPOS[index]) * delta > 0
            error = float(np.abs(actual - target).max())
            assert error < 0.03, error
            motion.append(
                {"joint": JOINT_NAMES[index], "delta_rad": delta, "error_rad": error}
            )
    return {"status": "PASS", "trials": motion}


def check_reaching(env, move, kinematics):
    """Reach three base-frame targets and reject an unreachable position."""
    task = env.unwrapped
    robot = task.agent.robot
    env.reset(seed=0)
    reaching = []
    for target in ([0.30, 0.06, 0.33], [0.25, -0.08, 0.38], [0.35, 0.0, 0.28]):
        initial = robot.get_qpos()[0].cpu().numpy()
        command = kinematics.solve_position(target, initial)
        move(command, steps=70)
        actual = (robot.pose.inv() * task.agent.tcp_pose).p[0].cpu().numpy()
        error = float(np.linalg.norm(actual - target))
        assert error < 0.015, error
        reaching.append(
            {
                "target_base_m": target,
                "actual_base_m": actual.tolist(),
                "error_m": error,
                "command_rad": command.tolist(),
            }
        )
    try:
        kinematics.solve_position([2, 0, 0.5], HOME_QPOS)
    except ValueError:
        pass
    else:
        raise AssertionError("Unreachable target accepted")
    return {"status": "PASS", "trials": reaching}


def check_collision_probes(env):
    """Measure dynamic-sphere contact forces on table, ground and link3."""
    task = env.unwrapped
    robot = task.agent.robot
    # Static table vs fixed robot base cannot produce dynamic contact forces.
    # Use an optional dynamic sphere to positively exercise all three surfaces.
    probes = [
        ("table", [0.5, 0, 0.55]),
        ("ground", [1, 0, 0.06]),
        ("link3", [0.20, 0, 0.86]),
    ]
    contacts = {}
    for expected, position in probes:
        env.reset(seed=0)
        task.probe.set_pose(sapien.Pose(position))
        found = False
        peak_force = 0.0
        surface = (
            robot.links_map[expected]
            if expected == "link3"
            else getattr(task, expected)
        )
        for _ in range(40):
            env.step(torch.as_tensor(HOME_QPOS[None]))
            force = task.scene.get_pairwise_contact_forces(task.probe, surface)
            peak_force = max(peak_force, float(torch.linalg.vector_norm(force)))
            for contact in task.scene.get_contacts():
                names = {
                    body.entity.name.removeprefix("scene-0_") for body in contact.bodies
                }
                if {"collision_probe", expected} <= names:
                    found = True
        contacts[expected] = {
            "status": "PASS" if found and peak_force > 0 else "FAIL",
            "peak_force_n": peak_force,
        }
        assert found and peak_force > 0, (
            f"No positive collision response with {expected}"
        )
    return contacts


def check_delta_control(env):
    """Check normalized direction and both base/jaw physical stops."""
    env.reset(seed=0)
    before = env.unwrapped.agent.robot.get_qpos().clone()
    env.step(torch.tensor([[1.0, 0, 0, 0]]))
    after = env.unwrapped.agent.robot.get_qpos()
    assert 0 < float(after[0, 0] - before[0, 0]) < 0.07
    # Exercise physical limits at both ends of base and jaw travel.
    robot = env.unwrapped.agent.robot
    limits = robot.get_qlimits()[0]
    for index in (0, 3):
        for side, direction in ((0, -1.0), (1, 1.0)):
            env.reset(seed=0)
            qpos = torch.as_tensor(HOME_QPOS[None].copy())
            qpos[0, index] = limits[index, side]
            robot.set_qpos(qpos)
            env.unwrapped.agent.controller.reset()
            action = torch.zeros((1, 4))
            action[0, index] = direction
            for _ in range(10):
                env.step(action)
                current = robot.get_qpos()[0]
                assert torch.isfinite(current).all()
                assert (current >= limits[:, 0] - 0.015).all()
                assert (current <= limits[:, 1] + 0.015).all()


def run_checks(output, frames=False):
    """Run assertions against the live CPU articulation, writing structured evidence."""
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("mani-skill", "sapien", "torch", "pin", "scipy")
        },
        "assets": audit_assets(),
        "cuda": "NOT TESTED",
        "hardware": "NOT TESTED",
    }
    env = make_roarm_env(
        render_mode="rgb_array" if frames else None, collision_probe=True
    )
    task = env.unwrapped
    robot = task.agent.robot
    kinematics = RoArmKinematics()
    frame_list = []
    peak_speed = 0.0
    max_fk_error = 0.0
    observed_contacts = set()
    limits = robot.get_qlimits()[0].cpu().numpy()
    step_count = 0

    def inspect():
        nonlocal peak_speed, max_fk_error
        qpos = robot.get_qpos()[0].cpu().numpy()
        qvel = robot.get_qvel()[0].cpu().numpy()
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        assert (qpos >= limits[:, 0] - 0.015).all() and (
            qpos <= limits[:, 1] + 0.015
        ).all()
        peak_speed = max(peak_speed, float(np.abs(qvel).max()))
        expected = robot.pose.sp * kinematics.forward(qpos)
        actual = task.agent.tcp_pose.sp
        max_fk_error = max(max_fk_error, float(np.linalg.norm(expected.p - actual.p)))
        assert np.linalg.norm(expected.p - actual.p) < 1e-5
        assert abs(float(np.dot(expected.q, actual.q))) > 1 - 1e-5
        for contact in task.scene.get_contacts():
            pair = tuple(
                sorted(
                    [
                        contact.bodies[0].entity.name.removeprefix("scene-0_"),
                        contact.bodies[1].entity.name.removeprefix("scene-0_"),
                    ]
                )
            )
            observed_contacts.add(pair)
            # The mounted base may touch its table; moving links must not collide.
            assert not (set(pair) & {"link1", "link2", "link3", "gripper_link"}), pair
        return qpos

    def move(target, steps=50):
        nonlocal step_count
        for _ in range(steps):
            env.step(torch.as_tensor(target[None], dtype=torch.float32))
            step_count += 1
            inspect()
            if frames and step_count % 10 == 0:
                rgb = env.render()[0].cpu().numpy()
                assert rgb.std() > 1
                frame_list.append(Image.fromarray(rgb))
        return inspect()

    try:
        observation, _ = env.reset(seed=0)
        report["loading"] = check_loading(env, observation)
        hold_error = float(np.abs(move(HOME_QPOS) - HOME_QPOS).max())
        assert hold_error < 0.03
        report["home_hold"] = {"status": "PASS", "max_joint_error_rad": hold_error}
        report["joint_motion"] = check_joint_motion(env, move)
        report["reaching"] = check_reaching(env, move, kinematics)
        report["unreachable_rejected"] = "PASS"
        env.reset(seed=0)
        assert np.allclose(inspect(), HOME_QPOS, atol=1e-6)
        assert torch.count_nonzero(robot.get_qvel()) == 0
        report["reset"] = "PASS"
        report["contact_probes"] = check_collision_probes(env)
        report["physics"] = {
            "status": "PASS",
            "steps": step_count,
            "peak_speed_rad_s": peak_speed,
            "max_fk_position_error_m": max_fk_error,
            "path_contacts": sorted(observed_contacts),
            "full_workspace_collision_safety": "NOT TESTED",
        }
    except Exception as exc:
        report["failure"] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        try:
            save_frames(output, frame_list, duration=100)
        finally:
            try:
                env.close()
            finally:
                write_report(output, report)
    # A separate instance checks normalized delta control and limit behavior.
    env = make_roarm_env(control_mode="pd_joint_delta_pos")
    try:
        check_delta_control(env)
        report["delta_control"] = "PASS"
        report["base_jaw_limit_stops"] = "PASS"
    except Exception as exc:
        report["delta_control"] = {"status": "FAIL", "error": str(exc)}
        raise
    finally:
        try:
            env.close()
        finally:
            write_report(output, report, display=True)


def main():
    """Parse CPU check options and execute the smoke suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/roarm_check"))
    parser.add_argument("--frames", action="store_true")
    args = parser.parse_args()
    run_checks(args.output.resolve(), args.frames)


if __name__ == "__main__":
    main()
