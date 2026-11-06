"""Current geometric goals, without planner waypoints or contact labels."""
import torch
from ..geometry import rotation, transform, inverse_transform


def rotation6(matrix):
    """First two COLUMNS, column-major [c0, c1], not quaternion subtraction."""
    return matrix[..., :, :2].transpose(-1, -2).flatten(-2)


def goal_geometry(g):
    rp, rh, re = [rotation(p[:, 3:]) for p in (g.peg_pose, g.hole_pose, g.ee_pose)]
    # Fixed tail grasp: TCP z approaches downward in the peg frame, fingers
    # close along -peg Y. Parallel-jaw pi yaw symmetry is handled in diagnostics.
    offset = torch.zeros_like(g.peg_half_size)
    offset[:, 0] = -torch.maximum(torch.full_like(offset[:, 0], .05), g.peg_half_size[:, 0]/2+.01)
    offset[:, 2] = g.peg_half_size[:, 2]-.025
    grasp_position = transform(offset[:, None], g.peg_pose)[:, 0]
    grasp_rotation = rp * rp.new_tensor([1., -1., -1.])[None, None, :]
    head_local = torch.zeros_like(g.peg_half_size)
    head_local[:, 0] = g.peg_half_size[:, 0]
    head = transform(head_local[:, None], g.peg_pose)
    return dict(grasp_position=grasp_position, grasp_rotation=grasp_rotation,
                ee_to_grasp=(grasp_position-g.ee_pose[:, :3])[:, None] @ re,
                grasp_relative_rotation=re.transpose(-1,-2) @ grasp_rotation,
                head_in_hole=inverse_transform(head, g.hole_pose)[:, 0],
                peg_relative_rotation=rh.transpose(-1,-2) @ rp,
                insertion_axis=rh[:, :, 0])


def angle(matrix):
    return torch.acos(((matrix.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1,1))
