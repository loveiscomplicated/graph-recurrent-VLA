"""World-axis pose geometry: metre positions and unit wxyz quaternions."""

import torch


def unit_quaternion(quaternion):
    """Normalize and choose w >= 0, respecting the q/-q rotation equivalence."""
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if quaternion.shape[-1] != 4 or not torch.isfinite(quaternion).all() or (norm < 1e-8).any():
        raise ValueError('Expected finite nonzero wxyz quaternions[...,4]')
    quaternion = quaternion / norm
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def quaternion_product(left, right):
    """Compose wxyz rotations; left * right applies right then left."""
    lw, lv = left[..., :1], left[..., 1:]
    rw, rv = right[..., :1], right[..., 1:]
    return torch.cat((lw * rw - (lv * rv).sum(-1, keepdim=True),
                      lw * rv + rw * lv + torch.linalg.cross(lv, rv)), dim=-1)


def conjugate(quaternion):
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def rotvec_quaternion(rotation_vector):
    """Exponential map, with a finite derivative at the identity rotation."""
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    factor = .5 * torch.sinc(angle / (2 * torch.pi))
    return torch.cat((torch.cos(angle / 2), rotation_vector * factor), dim=-1)


def quaternion_rotvec(quaternion):
    """Shortest rotation vector, in radians, stable at zero and near pi."""
    quaternion = unit_quaternion(quaternion)
    magnitude = torch.linalg.vector_norm(quaternion[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(magnitude, quaternion[..., :1])
    factor = torch.where(magnitude > 1e-7, angle / magnitude.clamp_min(1e-8), 2.)
    return quaternion[..., 1:] * factor


def angular_error_rad(prediction, target):
    """Geodesic angular distance in [0,pi], invariant to quaternion signs."""
    relative = quaternion_product(unit_quaternion(prediction), conjugate(unit_quaternion(target)))
    return 2 * torch.atan2(torch.linalg.vector_norm(relative[..., 1:], dim=-1), relative[..., 0].abs())


def peg_axis(quaternion):
    """Rotate local +X into world axes, matching object21_from_gt()."""
    w, x, y, z = unit_quaternion(quaternion).unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                        2 * (x * z - w * y)), dim=-1)


def geometric_endpoint(object21):
    """Peg +X tip minus Hole, using predicted half-length and world axes."""
    return object21[..., :3] + peg_axis(object21[..., 3:7]) * object21[..., 17:18] - object21[..., 10:13]


def rotate_vector(quaternion, vector):
    """Rotate world/local vectors by unit wxyz quaternions, with broadcasting."""
    quaternion = unit_quaternion(quaternion)
    imaginary = quaternion[..., 1:]
    imaginary, vector = torch.broadcast_tensors(imaginary, vector)
    cross = torch.linalg.cross(imaginary, vector)
    return vector + 2 * (quaternion[..., :1] * cross + torch.linalg.cross(imaginary, cross))


def hole_frame_vector(vector_world, hole_quaternion):
    """Express a vector in Hole coordinates: X axial, YZ lateral."""
    return rotate_vector(conjugate(hole_quaternion), vector_world)


def relative_rotation(peg_quaternion, hole_quaternion):
    """Peg orientation in Hole coordinates; retain the task's directed +X axis."""
    return quaternion_product(conjugate(unit_quaternion(hole_quaternion)),
                              unit_quaternion(peg_quaternion))


def assemble_refined_object21(frozen, corrections, position_mode, position_scale_m):
    """Decode selected 6D heads and reconstruct the +X endpoint from predicted size.

    Positions use the world frame here. The endpoint is endpoint minus Hole in
    world axes, which is invariant to the later TCP translation of both poses.
    Inactive poses and all Size4D values retain the Frozen GNN prediction.
    """
    poses = {'peg': frozen[..., :7], 'hole': frozen[..., 10:17]}
    for name, correction in corrections.items():
        start = 0 if name == 'peg' else 10
        position = correction[..., :3] * position_scale_m
        if position_mode == 'residual':
            position = position + frozen[..., start:start + 3]
        rotation = quaternion_product(rotvec_quaternion(correction[..., 3:]),
                                      unit_quaternion(frozen[..., start + 3:start + 7]))
        poses[name] = torch.cat((position, unit_quaternion(rotation)), dim=-1)
    composed = torch.cat((poses['peg'], frozen[..., 7:10], poses['hole'], frozen[..., 17:21]), dim=-1)
    return torch.cat((poses['peg'], geometric_endpoint(composed), poses['hole'], frozen[..., 17:21]), dim=-1)
