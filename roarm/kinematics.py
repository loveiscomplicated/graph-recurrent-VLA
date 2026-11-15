"""CPU position-only IK using SAPIEN's installed Pinocchio FK and SciPy."""

import xml.etree.ElementTree as ET

import numpy as np
from sapien.wrapper.pinocchio_model import PinocchioModel
from scipy.optimize import least_squares

from .assets import JOINT_NAMES, URDF_PATH


class RoArmKinematics:
    """Base-frame kinematics; preserve jaw angle and reject unreachable targets."""

    def __init__(self):
        self.model = PinocchioModel(URDF_PATH.read_text(), [0, 0, -9.81])
        self.model.set_joint_order(JOINT_NAMES)
        self.model.set_link_order(["hand_tcp"])
        root = ET.parse(URDF_PATH).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        limits = []
        for name in JOINT_NAMES:
            limit = joints[name].find("limit")
            limits.append([float(limit.get("lower")), float(limit.get("upper"))])
        limits = np.asarray(limits)
        self.limits = limits
        self.lower, self.upper = limits[:3, 0], limits[:3, 1]

    def forward(self, qpos):
        """Return the official TCP pose relative to base_link (metres, wxyz)."""
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (4,) or not np.isfinite(qpos).all():
            raise ValueError("qpos must contain four finite joint angles")
        self.model.compute_forward_kinematics(qpos)
        return self.model.get_link_pose(0)

    def solve_position(self, target, initial_qpos, tolerance=1e-4):
        """Fit three arm angles within URDF bounds; no orientation/collision planning."""
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target must be a finite base-frame xyz vector")
        initial = np.asarray(initial_qpos, dtype=float)
        self.forward(initial)  # Validate before slicing.
        if not self.limits[3, 0] <= initial[3] <= self.limits[3, 1]:
            raise ValueError("Initial jaw angle is outside the URDF limits")

        def residual(arm):
            return self.forward(np.r_[arm, initial[3]]).p - target

        def position_jacobian(arm):
            # SAPIEN Pose returns float32. Relative finite differences near a
            # zero angle can round to zero on Linux; use an absolute radian step.
            step = 1e-3
            columns = []
            for axis in range(3):
                offset = np.zeros(3)
                offset[axis] = step
                columns.append(
                    (residual(arm + offset) - residual(arm - offset)) / (2 * step)
                )
            return np.column_stack(columns)

        initial_arm = np.clip(initial[:3], self.lower, self.upper)
        result = least_squares(
            residual,
            initial_arm,
            bounds=(self.lower, self.upper),
            jac=position_jacobian,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=300,
        )
        error = np.linalg.norm(result.fun)
        if not result.success or error > tolerance:
            raise ValueError(f"Position IK failed: residual={error:.6f} m")
        return np.r_[result.x, initial[3]].astype(np.float32)
