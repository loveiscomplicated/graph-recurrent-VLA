"""Monitor guarded motion in the fixed RoArm simulation bench."""

import numpy as np
import torch

from roarm.collision import MotionRejected
from roarm.motion import RoArmMotionPlanner
from roarm.scene import MOUNT_POSITION, TABLE_HALF_SIZE, TABLE_POSITION


class RoArmGuardedExecutor:
    """Monitor one known static bench, latch faults, and hold measured qpos."""

    def __init__(self, env, planner=None):
        self.env = env
        self.task = env.unwrapped
        self.planner = planner or RoArmMotionPlanner()
        self.fault = None
        self._check_scene()
        self.last_valid_qpos = self.task.agent.robot.get_qpos().clone()
        if not torch.isfinite(self.last_valid_qpos).all():
            raise MotionRejected("Initial joint state is not finite")

    def _check_scene(self):
        task = self.task
        if task.num_envs != 1 or task.control_mode != "pd_joint_pos_guarded":
            raise MotionRejected(
                "Guard requires one environment and pd_joint_pos_guarded"
            )
        expected_actors = {"ground", "table"} | {
            f"guard_obstacle_{i}" for i in range(len(self.planner.collision.obstacles))
        }
        if set(task.scene.actors) != expected_actors or task.agent.uid != "roarm_m2_s":
            raise MotionRejected(
                "Guard supports only the unmodified static RoArm bench"
            )
        for index, item in enumerate(self.planner.collision.obstacles):
            obstacle = task.scene.actors[f"guard_obstacle_{index}"]
            pose = obstacle.pose.raw_pose[0].cpu().numpy()
            expected = np.r_[item["center"] + MOUNT_POSITION, 1, 0, 0, 0]
            if not np.allclose(pose, expected, atol=1e-6) or not np.allclose(
                obstacle._bodies[0].collision_shapes[0].half_size, item["half_size"]
            ):
                raise MotionRejected("Static obstacle geometry changed")
        table_pose = task.table.pose.raw_pose[0].cpu().numpy()
        shape = task.table._bodies[0].collision_shapes[0]
        if not np.allclose(
            table_pose, [*TABLE_POSITION, 1, 0, 0, 0], atol=1e-6
        ) or not np.allclose(shape.half_size, TABLE_HALF_SIZE):
            raise MotionRejected("Bench table geometry changed")
        pose = task.agent.robot.pose.raw_pose[0].cpu().numpy()
        if not np.allclose(pose, [*MOUNT_POSITION, 1, 0, 0, 0], atol=1e-6):
            raise MotionRejected("Bench mounting pose changed")

    def _hold_and_raise(self, reason):
        self.fault = str(reason)
        measured = self.task.agent.robot.get_qpos().clone()
        hold = measured if torch.isfinite(measured).all() else self.last_valid_qpos
        self.task.agent.controller.set_action(hold)
        raise MotionRejected(self.fault)

    def execute(self, path, on_step=None):
        """Latch any execution failure and hold the last finite measured state."""
        try:
            return self._execute(path, on_step)
        except MotionRejected as exc:
            if not self.fault:
                self._hold_and_raise(exc)
            raise
        except Exception as exc:
            self._hold_and_raise(f"Execution failed: {type(exc).__name__}: {exc}")

    def _execute(self, path, on_step=None):
        """Execute a preflighted trajectory; reject before stepping on any fault."""
        if self.fault:
            raise MotionRejected(
                f"Fault latched: {self.fault}; reset and create a new guard"
            )
        self._check_scene()
        settings = self.planner.collision.settings
        robot = self.task.agent.robot
        dt = 1 / self.task.control_freq
        trajectory = self.planner.trajectory(path, dt)
        start = robot.get_qpos()[0].cpu().numpy()
        if np.max(np.abs(start - trajectory[0])) > settings.tracking_error_rad:
            self._hold_and_raise("Start state changed since planning")
        peak_speed = 0.0
        peak_error = 0.0
        peak_contact = 0.0
        previous = trajectory[0]
        for command in trajectory[1:]:
            try:
                self._check_scene()
            except MotionRejected as exc:
                self._hold_and_raise(exc)
            actual = robot.get_qpos()[0].cpu().numpy()
            error = float(np.max(np.abs(actual - previous)))
            speed = float(robot.get_qvel().abs().max())
            if (
                not np.isfinite(actual).all()
                or not np.isfinite(speed)
                or error > settings.tracking_error_rad
                or speed > settings.measured_speed_limit_rad_s
            ):
                self._hold_and_raise(
                    "Tracking, finite-state or measured-speed limit exceeded"
                )
            if self.planner.collision.rejection(actual) is not None:
                self._hold_and_raise("Measured state lost clearance")
            obs, _, _, truncated, _ = self.env.step(
                torch.as_tensor(
                    command[None], device=self.task.device, dtype=torch.float32
                )
            )
            if bool(truncated.any()):
                self._hold_and_raise("Episode horizon reached")
            measured = robot.get_qpos()[0].cpu().numpy()
            error = float(np.max(np.abs(measured - command)))
            speed = float(robot.get_qvel().abs().max())
            if (
                not np.isfinite(error)
                or not np.isfinite(speed)
                or error > settings.tracking_error_rad
                or speed > settings.measured_speed_limit_rad_s
            ):
                self._hold_and_raise(
                    f"Post-step limit: error={error:.6f} rad, speed={speed:.6f} rad/s"
                )
            if self.planner.collision.rejection(measured) is not None:
                self._hold_and_raise("Measured state lost clearance")
            peak_error = max(peak_error, error)
            peak_speed = max(peak_speed, speed)
            for contact in self._contact_values():
                peak_contact = max(peak_contact, contact)
                if not np.isfinite(contact):
                    self._hold_and_raise("Contact force is not finite")
                if contact > settings.contact_limit_n:
                    self._hold_and_raise("Unexpected contact force")
            if on_step:
                on_step(obs)
            self.last_valid_qpos = robot.get_qpos().clone()
            previous = command
        return {
            "steps": len(trajectory) - 1,
            "peak_tracking_error_rad": peak_error,
            "peak_measured_speed_rad_s": peak_speed,
            "peak_contact_force_n": peak_contact,
        }

    def _contact_values(self):
        robot = self.task.agent.robot
        for first, second in self.planner.collision.pairs:
            yield float(
                torch.linalg.vector_norm(
                    self.task.scene.get_pairwise_contact_forces(
                        robot.links_map[first], robot.links_map[second]
                    )
                )
            )
        for name in self.planner.collision.link_names[1:]:
            obstacles = [
                self.task.scene.actors[f"guard_obstacle_{i}"]
                for i in range(len(self.planner.collision.obstacles))
            ]
            for actor in (self.task.table, self.task.ground, *obstacles):
                yield float(
                    torch.linalg.vector_norm(
                        self.task.scene.get_pairwise_contact_forces(
                            robot.links_map[name], actor
                        )
                    )
                )
