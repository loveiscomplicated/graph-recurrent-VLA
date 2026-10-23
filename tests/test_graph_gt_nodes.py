"""Unit tests for TCP-relative GT point extraction."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from graph.gt_nodes import get_gt_node_positions


def _fake_state(
    *,
    ee_pose: torch.Tensor,
    left_fingertip_pos: torch.Tensor,
    right_fingertip_pos: torch.Tensor,
    peg_pose: torch.Tensor,
    peg_half_sizes: torch.Tensor,
    hole_pose: torch.Tensor,
    hole_radius: torch.Tensor,
) -> SimpleNamespace:
    batch = ee_pose.shape[0]
    return SimpleNamespace(
        ee_pose=ee_pose,
        left_fingertip_pos=left_fingertip_pos,
        right_fingertip_pos=right_fingertip_pos,
        peg_pose=peg_pose,
        peg_half_sizes=peg_half_sizes,
        hole_pose=hole_pose,
        hole_radius=hole_radius,
        # Other fields are not read by get_gt_node_positions.
        qpos=torch.zeros((batch, 9)),
        qvel=torch.zeros((batch, 9)),
        box_pose=torch.zeros((batch, 7)),
        gripper_qpos=torch.zeros((batch, 2)),
        success=torch.zeros((batch,), dtype=torch.bool),
    )


def _identity_pose(position: tuple[float, float, float]) -> list[float]:
    return [*position, 1.0, 0.0, 0.0, 0.0]


class GetGTNodePositionsTest(unittest.TestCase):
    def test_returns_expected_tcp_relative_positions_and_corners(self):
        state = _fake_state(
            ee_pose=torch.tensor([_identity_pose((10.0, 20.0, 30.0))]),
            left_fingertip_pos=torch.tensor([[10.1, 20.2, 30.3]]),
            right_fingertip_pos=torch.tensor([[9.9, 19.8, 29.7]]),
            peg_pose=torch.tensor([_identity_pose((11.0, 20.0, 30.0))]),
            peg_half_sizes=torch.tensor([[1.0, 2.0, 3.0]]),
            hole_pose=torch.tensor([_identity_pose((10.0, 22.0, 30.0))]),
            hole_radius=torch.tensor([0.5]),
        )

        with patch("graph.gt_nodes.get_gt_state", return_value=state):
            nodes = get_gt_node_positions(env=object())

        self.assertEqual(
            set(nodes),
            {"gripper_tcp", "left_fingertip", "right_fingertip", "peg_corners", "hole_corners"},
        )
        torch.testing.assert_close(nodes["gripper_tcp"], torch.zeros((1, 3)))
        torch.testing.assert_close(nodes["left_fingertip"], torch.tensor([[0.1, 0.2, 0.3]]))
        torch.testing.assert_close(nodes["right_fingertip"], torch.tensor([[-0.1, -0.2, -0.3]]))
        self.assertEqual(tuple(nodes["peg_corners"].shape), (1, 8, 3))
        self.assertEqual(tuple(nodes["hole_corners"].shape), (1, 4, 3))
        torch.testing.assert_close(nodes["peg_corners"][0, 0], torch.tensor([0.0, -2.0, -3.0]))
        torch.testing.assert_close(nodes["peg_corners"][0, -1], torch.tensor([2.0, 2.0, 3.0]))
        torch.testing.assert_close(nodes["hole_corners"][0, 0], torch.tensor([0.0, 1.5, -0.5]))
        torch.testing.assert_close(nodes["hole_corners"][0, -1], torch.tensor([0.0, 2.5, 0.5]))

    def test_coordinates_follow_tcp_orientation(self):
        half_sqrt_two = 2**-0.5
        rotated_tcp = [0.0, 0.0, 0.0, half_sqrt_two, 0.0, 0.0, half_sqrt_two]
        state = _fake_state(
            ee_pose=torch.tensor([rotated_tcp]),
            left_fingertip_pos=torch.tensor([[0.0, 1.0, 0.0]]),
            right_fingertip_pos=torch.tensor([[0.0, -1.0, 0.0]]),
            peg_pose=torch.tensor([rotated_tcp]),
            peg_half_sizes=torch.zeros((1, 3)),
            hole_pose=torch.tensor([rotated_tcp]),
            hole_radius=torch.zeros((1,)),
        )

        with patch("graph.gt_nodes.get_gt_state", return_value=state):
            nodes = get_gt_node_positions(env=object())

        # The TCP is rotated +90 degrees around world Z, so world +Y is TCP +X.
        torch.testing.assert_close(nodes["left_fingertip"], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(nodes["right_fingertip"], torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)

    def test_preserves_batch_dimension(self):
        poses = torch.tensor(
            [
                _identity_pose((0.0, 0.0, 0.0)),
                _identity_pose((10.0, 0.0, 0.0)),
            ]
        )
        state = _fake_state(
            ee_pose=poses,
            left_fingertip_pos=poses[:, :3] + torch.tensor([[1.0, 0.0, 0.0]]),
            right_fingertip_pos=poses[:, :3] - torch.tensor([[1.0, 0.0, 0.0]]),
            peg_pose=poses,
            peg_half_sizes=torch.ones((2, 3)),
            hole_pose=poses,
            hole_radius=torch.ones((2,)),
        )

        with patch("graph.gt_nodes.get_gt_state", return_value=state):
            nodes = get_gt_node_positions(env=object())

        self.assertEqual(tuple(nodes["gripper_tcp"].shape), (2, 3))
        self.assertEqual(tuple(nodes["left_fingertip"].shape), (2, 3))
        self.assertEqual(tuple(nodes["peg_corners"].shape), (2, 8, 3))
        self.assertEqual(tuple(nodes["hole_corners"].shape), (2, 4, 3))
        torch.testing.assert_close(nodes["left_fingertip"], torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
