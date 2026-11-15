"""RoArm asset provenance and CPU position-IK regression checks."""

import unittest

import numpy as np

from roarm.assets import HOME_QPOS, audit_assets
from roarm.kinematics import RoArmKinematics


class RoArmKinematicsTest(unittest.TestCase):
    def setUp(self):
        self.solver = RoArmKinematics()

    def test_vendored_asset_integrity(self):
        self.assertEqual(audit_assets()["status"], "PASS")

    def test_home_tcp_geometry(self):
        # Independently specified from the official home chain dimensions.
        np.testing.assert_allclose(
            self.solver.forward(HOME_QPOS).p, [0.3202024, 0, 0.3578744], atol=1e-5
        )

    def test_position_ik_preserves_jaw(self):
        for target in ([0.30, 0.06, 0.33], [0.25, -0.08, 0.38], [0.35, 0, 0.28]):
            with self.subTest(target=target):
                command = self.solver.solve_position(target, HOME_QPOS)
                np.testing.assert_allclose(
                    self.solver.forward(command).p, target, atol=1e-4
                )
                self.assertEqual(command[3], HOME_QPOS[3])
                self.assertTrue((command[:3] >= self.solver.lower).all())
                self.assertTrue((command[:3] <= self.solver.upper).all())

    def test_jaw_does_not_change_official_tcp(self):
        open_jaw = HOME_QPOS.copy()
        open_jaw[3] = 1.5
        np.testing.assert_allclose(
            self.solver.forward(open_jaw).p, self.solver.forward(HOME_QPOS).p
        )

    def test_unreachable_and_invalid_targets_are_rejected(self):
        for target in ([2, 0, 0.5], [float("nan"), 0, 0], [0, 0]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.solver.solve_position(target, HOME_QPOS)


if __name__ == "__main__":
    unittest.main()
