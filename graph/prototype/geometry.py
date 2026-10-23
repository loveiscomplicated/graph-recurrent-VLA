"""Geometry-only tensors, deliberately excluding rewards, success and actions."""
from dataclasses import dataclass, fields

import torch
from torch import Tensor


def rotation(q: Tensor) -> Tensor:
    """wxyz quaternion -> rotation matrix, including arbitrary batch dimensions."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def transform(points: Tensor, pose: Tensor) -> Tensor:
    return points @ rotation(pose[..., 3:]).transpose(-1, -2) + pose[..., None, :3]


def inverse_transform(points: Tensor, pose: Tensor) -> Tensor:
    return (points - pose[..., None, :3]) @ rotation(pose[..., 3:])


@dataclass(frozen=True)
class GeometryState:
    """All float32 with leading B; metres, wxyz poses, world axes.

    Point/normal tensors [B,P,3]; poses [B,7]; qpos [B,9]; gripper [B,2];
    half sizes [B,3], radius [B,1], hole offset [B,3]. Hole points use BOX
    local coordinates (the aperture is offset from the box centre).
    Fingertips mean rubber pad collision centres, not finger link origins.
    """
    ee_pose: Tensor
    left_fingertip: Tensor
    right_fingertip: Tensor
    gripper: Tensor
    qpos: Tensor
    root_pose: Tensor
    peg_pose: Tensor
    peg_half_size: Tensor
    peg_local: Tensor
    peg_points: Tensor
    peg_normals: Tensor
    box_pose: Tensor
    box_half_size: Tensor
    hole_pose: Tensor
    hole_radius: Tensor
    hole_offset: Tensor
    hole_local: Tensor
    hole_points: Tensor
    hole_normals: Tensor

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to(self, device):
        return GeometryState(**{k: v.to(device) for k, v in self.as_dict().items()})

    def select(self, indices):
        return GeometryState(**{k: v[indices] for k, v in self.as_dict().items()})


def concatenate(states):
    return GeometryState(**{k: torch.cat([s.as_dict()[k] for s in states]) for k in states[0].as_dict()})
