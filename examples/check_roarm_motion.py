"""Exercise guarded RoArm reaching, detours, unsafe rejection and fault latching."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image

from envs.maniskill.roarm import make_roarm_env
from envs.maniskill.roarm.execution import RoArmGuardedExecutor
from examples.roarm_checks import save_frames, write_report
from roarm.assets import HOME_QPOS
from roarm.collision import MotionRejected, RoArmCollisionModel
from roarm.motion import RoArmMotionPlanner


def check_geometry(env, collision):
    """Compare independent hull FK with every live physical link."""
    robot = env.unwrapped.agent.robot
    # Match the independent geometric model against every live physical link.
    geometry = collision.geometry(HOME_QPOS)
    for name, (vertices, _) in geometry.items():
        live = (
            robot.pose.inv() * robot.links_map[name].pose
        ).sp.to_transformation_matrix()
        vertices_local = collision.meshes[name][0]
        expected = vertices_local @ live[:3, :3].T + live[:3, 3]
        assert np.max(np.abs(vertices - expected)) < 2e-6


def check_invalid_motion(env, planner, unsafe_elbow):
    """Reject invalid goals without moving the articulation."""
    robot = env.unwrapped.agent.robot
    before = robot.get_qpos().clone()
    for invalid in (unsafe_elbow, [float("nan"), 0, 1, 0.75], [4, 0, 1, 0.75]):
        try:
            planner.plan_joints(HOME_QPOS, invalid)
        except MotionRejected:
            pass
        else:
            raise AssertionError("Unsafe command was accepted")
    assert torch.equal(before, robot.get_qpos())


def check_reaching(env, planner, capture):
    """Check command timing and execute five successive base-frame goals."""
    collision = planner.collision
    robot = env.unwrapped.agent.robot
    trials = []
    for target in (
        [0.30, 0.06, 0.33],
        [0.25, -0.08, 0.38],
        [0.35, 0, 0.28],
        [0.27, 0.12, 0.32],
        [0.25, -0.12, 0.32],
    ):
        start = robot.get_qpos()[0].cpu().numpy()
        path = planner.plan_position(start, target)
        trajectory = planner.trajectory(path, 1 / env.unwrapped.control_freq)
        commands = np.diff(trajectory, axis=0) * env.unwrapped.control_freq
        assert np.max(np.abs(commands)) <= collision.settings.command_speed_rad_s + 1e-6
        assert (
            np.max(np.abs(np.diff(commands, axis=0) * env.unwrapped.control_freq))
            <= collision.settings.command_acceleration_rad_s2 + 1e-4
        )
        stats = RoArmGuardedExecutor(env, planner).execute(path, on_step=capture)
        actual = (robot.pose.inv() * env.unwrapped.agent.tcp_pose).p[0].cpu().numpy()
        error = float(np.linalg.norm(actual - target))
        assert error < 0.015
        trials.append({"target_base_m": target, "error_m": error, **stats})
    return trials


def check_obstacle_detour(env, planner, capture):
    """Execute a route around a physical box whose direct path is blocked."""
    collision = planner.collision
    obstacles = [
        {key: value.tolist() for key, value in obstacle.items()}
        for obstacle in collision.obstacles
    ]
    env.reset(seed=0)
    robot = env.unwrapped.agent.robot
    first = collision.kinematics.solve_position([0.30, -0.16, 0.23], HOME_QPOS)
    second = collision.kinematics.solve_position([0.30, 0.16, 0.23], HOME_QPOS)
    assert collision.rejection(first) is None and collision.rejection(second) is None
    assert not collision.edge_valid(first, second)
    detour = planner.plan_joints(first, second)
    assert len(detour) > 2
    # Move there through certified paths, without teleporting the robot.
    env.reset(seed=0)
    start = robot.get_qpos()[0].cpu().numpy()
    approach = planner.plan_joints(start, detour[0])
    RoArmGuardedExecutor(env, planner).execute(approach, on_step=capture)
    # Replan from measured start, preserving the tracking-margin accounting.
    measured = robot.get_qpos()[0].cpu().numpy()
    path = planner.plan_joints(measured, detour[-1])
    stats = RoArmGuardedExecutor(env, planner).execute(path, on_step=capture)
    final = robot.get_qpos()[0].cpu().numpy()
    assert np.max(np.abs(final - path[-1])) < collision.settings.tracking_error_rad
    return {
        "status": "PASS",
        "direct_path_rejected": True,
        "obstacles": obstacles,
        "waypoints": path.tolist(),
        **stats,
    }


def check_stale_plan(env, planner):
    """Latch a stale-plan fault before advancing simulation."""
    # A stale plan must latch a fault without advancing the environment.
    guard = RoArmGuardedExecutor(env, planner)
    elapsed = env.unwrapped.elapsed_steps.clone()
    try:
        guard.execute(np.array([HOME_QPOS, HOME_QPOS]))
    except MotionRejected:
        assert guard.fault and torch.equal(elapsed, env.unwrapped.elapsed_steps)
    else:
        raise AssertionError("Stale plan was accepted")


def check_tracking_fault(env, planner):
    """Inject a joint disturbance and verify the persistent execution fault."""
    robot = env.unwrapped.agent.robot
    # Inject a joint-state disturbance and require a latched stop before the
    # next physics step, on both CPU and CUDA state-buffer implementations.
    env.reset(seed=0)
    guard = RoArmGuardedExecutor(env, planner)
    path = planner.plan_position(HOME_QPOS, [0.25, 0, 0.45])
    injected_at = None
    callback_steps = 0

    def disturb(obs):
        nonlocal injected_at, callback_steps
        callback_steps += 1
        if callback_steps == 5:
            state = env.unwrapped.get_state_dict()
            # Articulation state: pose (7), root velocities (6), then qpos.
            first_joint_index = 13
            state["articulations"][robot.name][:, first_joint_index] += 0.03
            env.unwrapped.set_state_dict(state)
            injected_at = env.unwrapped.elapsed_steps.clone()

    try:
        guard.execute(path, on_step=disturb)
    except MotionRejected:
        assert injected_at is not None and guard.fault
        assert torch.equal(injected_at, env.unwrapped.elapsed_steps)
    else:
        raise AssertionError("Tracking disturbance was ignored")
    try:
        guard.execute(path)
    except MotionRejected:
        assert torch.equal(injected_at, env.unwrapped.elapsed_steps)
    else:
        raise AssertionError("Latched fault allowed motion")


def main():
    """Run the reaching, obstacle and fault checks, then save their evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/roarm_motion"))
    parser.add_argument("--observation-device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--frames", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    observation_device = args.observation_device or args.device
    report = {
        "status": "RUNNING",
        "device": args.device,
        "observation_device": observation_device,
        "hardware": "NOT TESTED",
    }
    env = None
    images = []
    frame_counter = 0

    def capture(observation):
        nonlocal frame_counter
        assert observation["agent"]["qpos"].device.type == observation_device
        frame_counter += 1
        if args.frames and frame_counter % 4 == 0:
            images.append(Image.fromarray(env.render()[0].cpu().numpy()))

    try:
        collision = RoArmCollisionModel()
        planner = RoArmMotionPlanner(collision)
        collision.export_hulls(HOME_QPOS, args.output / "home_collision.glb")
        unsafe_elbow = HOME_QPOS.copy()
        unsafe_elbow[2] = 3.12
        collision.export_hulls(unsafe_elbow, args.output / "elbow_collision.glb")
        env = make_roarm_env(
            device=args.device,
            observation_device=args.observation_device,
            control_mode="pd_joint_pos_guarded",
            render_mode="rgb_array" if args.frames else None,
        )
        env.reset(seed=0)
        check_geometry(env, collision)
        report["geometry_fk"] = "PASS"
        check_invalid_motion(env, planner, unsafe_elbow)
        report["unsafe_rejected_without_step"] = "PASS"
        report["guarded_reaching"] = check_reaching(env, planner, capture)

        # A second known scene tests a real physical obstacle detour.
        env.close()
        env = None
        obstacles = [{"center": [0.31, 0, 0.25], "half_size": [0.025, 0.035, 0.025]}]
        planner = RoArmMotionPlanner(RoArmCollisionModel(obstacles=obstacles))
        env = make_roarm_env(
            device=args.device,
            observation_device=args.observation_device,
            control_mode="pd_joint_pos_guarded",
            obstacles=obstacles,
            render_mode="rgb_array" if args.frames else None,
        )
        report["detour"] = check_obstacle_detour(env, planner, capture)
        check_stale_plan(env, planner)
        report["stale_plan_fault"] = "PASS"
        check_tracking_fault(env, planner)
        report["tracking_fault_and_latch"] = "PASS"
        report["settings"] = vars(planner.collision.settings)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            save_frames(args.output, images, duration=120)
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                write_report(args.output, report, display=True)


if __name__ == "__main__":
    main()
