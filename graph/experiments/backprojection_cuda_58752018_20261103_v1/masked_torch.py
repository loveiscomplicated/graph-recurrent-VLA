"""Compact Torch projection prototype; no dense XYZ or robot filtering.

    Inputs: positive/invalid metre depth [H,W], two boolean masks [2,H,W], K.
    Output: float32 camera XYZ [N,3], int64 source pixels [N], object IDs [N].

Ordering is object 0 then 1, each in original raster order. No sampling or
downsampling occurs. Float64 arithmetic preserves the NumPy reference policy.
"""
import torch


def project(depth, masks, intrinsic):
    if depth.ndim != 2 or min(depth.shape) < 1 or not depth.is_floating_point():
        raise ValueError('nonempty floating [H,W] depth required')
    if masks.shape != (2, *depth.shape) or masks.dtype != torch.bool:
        raise ValueError('two bool masks aligned with depth required')
    if intrinsic.shape != (3, 3) or not intrinsic.is_floating_point():
        raise ValueError('floating [3,3] CV intrinsic required')
    if masks.device != depth.device or intrinsic.device != depth.device:
        raise ValueError('all inputs must share a device')
    k = intrinsic.to(torch.float64)
    # Match the old calibration checks. This scalar host decision is included
    # in the measured resident wall time, as is nonzero's dynamic compaction.
    standard = torch.stack((k[0, 1], k[1, 0], k[2, 0], k[2, 1], k[2, 2]))
    expected = torch.zeros_like(standard); expected[-1] = 1
    calibration_ok = (torch.isfinite(k).all() & (k[0, 0] > 0) & (k[1, 1] > 0)
                      & torch.isclose(standard, expected).all())
    if not bool(calibration_ok):
        raise ValueError('invalid zero-skew CV intrinsic')
    valid = torch.isfinite(depth) & (depth > 0)
    overlap = masks[0] & masks[1]
    selected = torch.nonzero(masks & valid[None] & ~overlap[None], as_tuple=False)
    oid, v, u = selected.unbind(1)
    z = depth[v, u].to(torch.float64)
    x = (u.to(torch.float64)+.5-k[0, 2])*z/k[0, 0]
    y = (v.to(torch.float64)+.5-k[1, 2])*z/k[1, 1]
    xyz = torch.stack((x, y, z), dim=1).to(torch.float32)
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError('selected XYZ is not representable as float32')
    return dict(points=xyz, pixels=v*depth.shape[1]+u, object_id=oid)
