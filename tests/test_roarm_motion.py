"""Geometric, path and timing contracts for the conservative RoArm guard."""

import unittest

import numpy as np

from roarm.assets import HOME_QPOS
from roarm.collision import MotionRejected, RoArmCollisionModel, SafetySettings
from roarm.motion import RoArmMotionPlanner


class RoArmMotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collision = RoArmCollisionModel()
        cls.planner = RoArmMotionPlanner(cls.collision)

    def test_home_and_reaching_are_accepted(self):
        self.assertIsNone(self.collision.rejection(HOME_QPOS))
        path = self.planner.plan_position(HOME_QPOS, [0.30, 0.06, 0.33])
        self.assertTrue(self.collision.edge_valid(path[0], path[-1]))

    def test_physx_collision_examples_are_rejected(self):
        elbow = HOME_QPOS.copy()
        elbow[2] = 3.12
        # These table-contact examples were independently observed in PhysX.
        for qpos in (elbow, [0, -1.5608, 1.0708, 0.75], [0, -1.5608, 1.7577, 0.75]):
            with self.subTest(qpos=qpos):
                self.assertIsNotNone(self.collision.rejection(qpos))
                with self.assertRaises(MotionRejected):
                    self.planner.plan_joints(HOME_QPOS, qpos)

    def test_invalid_and_limit_margin_commands_are_rejected(self):
        for qpos in (
            [float("nan"), 0, 1, 0.75],
            [0, 0, 1],
            [4, 0, 1, 0.75],
            [3.14, 0, 1.57, 0.75],
        ):
            with self.subTest(qpos=qpos), self.assertRaises(MotionRejected):
                self.planner.plan_joints(HOME_QPOS, qpos)

    def test_safe_endpoints_do_not_imply_safe_path(self):
        collision = RoArmCollisionModel(
            obstacles=[dict(center=[0.31, 0, 0.25], half_size=[0.025, 0.035, 0.025])]
        )
        planner = RoArmMotionPlanner(collision)
        first = collision.kinematics.solve_position([0.30, -0.16, 0.23], HOME_QPOS)
        second = collision.kinematics.solve_position([0.30, 0.16, 0.23], HOME_QPOS)
        self.assertIsNone(collision.rejection(first))
        self.assertIsNone(collision.rejection(second))
        self.assertFalse(collision.edge_valid(first, second))
        route = planner.plan_joints(first, second, timeout=10)
        self.assertGreater(len(route), 2)
        self.assertTrue(
            all(collision.edge_valid(a, b) for a, b in zip(route[:-1], route[1:]))
        )
        with self.assertRaises(MotionRejected):
            planner.trajectory([first, second], 0.05)

    def test_time_scaling_respects_command_limits(self):
        path = self.planner.plan_position(HOME_QPOS, [0.25, -0.08, 0.38])
        dt = 0.05
        positions = self.planner.trajectory(path, dt)
        velocities = np.diff(positions, axis=0) / dt
        accelerations = np.diff(velocities, axis=0) / dt
        self.assertLessEqual(np.abs(velocities).max(), 0.25 + 1e-6)
        self.assertLessEqual(np.abs(accelerations).max(), 0.5 + 1e-4)
        np.testing.assert_allclose(positions[0], path[0])
        np.testing.assert_allclose(positions[-1], path[-1])

    def test_motion_bound_contains_transformed_vertices(self):
        start = HOME_QPOS.copy()
        end = start + [0.04, -0.03, 0.02, 0.01]
        first, second = self.collision.geometry(start), self.collision.geometry(end)
        for name in self.collision.link_names:
            displacement = np.linalg.norm(
                first[name][0] - second[name][0], axis=1
            ).max()
            bound = self.collision.motion_bounds[name] @ np.abs(end - start)
            self.assertLessEqual(displacement, bound + 1e-6)

    def test_nonpositive_or_nonfinite_settings_fail_closed(self):
        for kwargs in (
            {"clearance_m": 0},
            {"tracking_error_rad": -1},
            {"command_speed_rad_s": float("nan")},
            {"max_edge_depth": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SafetySettings(**kwargs)


if __name__ == "__main__":
    unittest.main()
