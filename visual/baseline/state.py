"""Robot state fed to the baseline policies (numpy only, no simulator import).

Layout, 17 dims: tcp_pos 3, tcp_rot6 6, left_tip - tcp 3, right_tip - tcp 3,
gripper_qpos 2.

The rotation is the first two columns of the TCP rotation matrix, the same 6D
form the Recurrent team adopted in experiment 2 (``recurrent/exp2/exp2_train.py``).
The earlier robot15 layout stored a quaternion made non-negative in w. In
top-down grasps w is near 0, so that canonicalization flipped the whole
quaternion's sign from frame to frame; the 6D form is identical for q and -q.
"""

from __future__ import annotations

import numpy as np

ROBOT_DIM = 17


def rot6(quat: np.ndarray) -> np.ndarray:
    """wxyz quaternions (..., 4) → first two rotation-matrix columns (..., 6)."""
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
        2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x),
    ], axis=-1)


def robot_state(ee_pose, left_tip, right_tip, gripper_qpos) -> np.ndarray:
    """Build the 17D measured robot state; inputs may carry leading batch/time dims."""
    ee_pose = np.asarray(ee_pose, dtype=np.float64)
    tcp_position = ee_pose[..., :3]
    parts = [
        tcp_position,
        rot6(ee_pose[..., 3:7]),
        np.asarray(left_tip) - tcp_position,
        np.asarray(right_tip) - tcp_position,
        np.asarray(gripper_qpos),
    ]
    state = np.concatenate(parts, axis=-1).astype(np.float32)
    assert state.shape[-1] == ROBOT_DIM
    return state
