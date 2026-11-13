"""Convert raw world-frame GT to the gated MPNN's Object21 target."""

from collections.abc import Mapping

import torch
from torch import Tensor


def object21_from_gt(ground_truth: Mapping[str, Tensor]) -> Tensor:
    """Return [B,21], keeping world axes and expressing positions from TCP.

    Quaternions are world-frame wxyz. The endpoint is the peg's local +X tip
    minus the hole position, not a hole-frame rotation of that vector.
    """
    shapes = {"ee_pose": 7, "peg_pose": 7, "hole_pose": 7, "peg_half_sizes": 3}
    values = {}
    batch_size = None
    for name, width in shapes.items():
        if name not in ground_truth:
            raise ValueError(f"Missing GT field: {name}; collect with save_step_gt=true")
        value = torch.as_tensor(ground_truth[name], dtype=torch.float32)
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"GT {name} must have shape [B,{width}]")
        if batch_size is None:
            batch_size = value.shape[0]
        if value.shape[0] != batch_size or not torch.isfinite(value).all():
            raise ValueError(f"GT {name} must be finite with the same B")
        values[name] = value
    radius = torch.as_tensor(ground_truth.get("hole_radius", []), dtype=torch.float32)
    if radius.shape != (batch_size,) or not torch.isfinite(radius).all():
        raise ValueError("GT hole_radius must be finite [B]")
    device = values["ee_pose"].device
    if any(value.device != device for value in (*values.values(), radius)):
        raise ValueError("All GT tensors must be on the same device")
    if torch.any(values["peg_half_sizes"] <= 0) or torch.any(radius <= 0):
        raise ValueError("GT peg_half_sizes and hole_radius must be positive")
    quaternions = []
    for name in ("peg_pose", "hole_pose"):
        quat = values[name][:, 3:]
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        if torch.any(norm < 1e-8):
            raise ValueError(f"GT {name} contains a zero quaternion")
        quaternions.append(quat / norm)
    peg_quat, hole_quat = quaternions
    w, x, y, z = peg_quat.unbind(-1)
    peg_axis = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z),
                            2 * (x*z - w*y)), dim=-1)
    peg = values["peg_pose"][:, :3]
    hole = values["hole_pose"][:, :3]
    tcp = values["ee_pose"][:, :3]
    endpoint = peg + peg_axis * values["peg_half_sizes"][:, :1] - hole
    return torch.cat((peg - tcp, peg_quat, endpoint, hole - tcp, hole_quat,
                      values["peg_half_sizes"], radius[:, None]), dim=-1)
