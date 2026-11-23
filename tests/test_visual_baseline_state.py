"""Robot-state layout of the Visual baselines (numpy only, no simulator)."""

import importlib.util
from pathlib import Path

import numpy as np

# Load state.py directly so the test does not import torch or ManiSkill.
STATE_PATH = Path(__file__).resolve().parents[1] / "visual" / "baseline" / "state.py"
spec = importlib.util.spec_from_file_location("baseline_state", STATE_PATH)
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


def random_quaternions(count: int, seed: int = 0) -> np.ndarray:
    quaternions = np.random.default_rng(seed).normal(size=(count, 4))
    return quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)


def rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Reference wxyz quaternion → 3x3 rotation matrix."""
    w, x, y, z = quaternions.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def test_rot6_is_first_two_rotation_columns():
    quaternions = random_quaternions(100)
    matrices = rotation_matrices(quaternions)
    expected = np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)
    np.testing.assert_allclose(state.rot6(quaternions), expected, atol=1e-12)


def test_rot6_ignores_quaternion_sign():
    quaternions = random_quaternions(100, seed=1)
    np.testing.assert_allclose(state.rot6(quaternions), state.rot6(-quaternions), atol=1e-12)


def test_top_down_grasp_sign_flip_no_longer_jumps():
    # Two nearly identical top-down orientations with w just above and below 0:
    # the old w >= 0 canonicalization mapped them to opposite quaternions.
    before = np.array([0.001, 1.0, 0.0, 0.0])
    after = np.array([-0.001, 1.0, 0.0, 0.0])
    assert np.abs(state.rot6(before) - state.rot6(after)).max() < 1e-2


def test_robot_state_layout_with_time_axis():
    steps = 5
    ee_pose = np.concatenate([np.full((steps, 3), 0.5), random_quaternions(steps)], axis=1)
    left_tip = ee_pose[:, :3] + 0.01
    right_tip = ee_pose[:, :3] - 0.01
    gripper = np.full((steps, 2), 0.04)

    robot = state.robot_state(ee_pose, left_tip, right_tip, gripper)

    assert robot.shape == (steps, state.ROBOT_DIM) == (steps, 17)
    assert robot.dtype == np.float32
    np.testing.assert_allclose(robot[:, :3], 0.5)
    np.testing.assert_allclose(robot[:, 9:12], 0.01, atol=1e-7)
    np.testing.assert_allclose(robot[:, 12:15], -0.01, atol=1e-7)
    np.testing.assert_allclose(robot[:, 15:], 0.04)
