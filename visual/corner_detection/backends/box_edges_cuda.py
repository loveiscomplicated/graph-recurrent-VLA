"""CUDA verification of Box edges using only the sampled RGB-D pixels."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _edge_samples(
    center_u, center_v, side_u, side_v, side_sign,
    xyz, support, nx, ny, nz, offset,
):
    u = libdevice.rint(center_u + 2.0 * side_sign * side_u).to(tl.int32)
    v = libdevice.rint(center_v + 2.0 * side_sign * side_v).to(tl.int32)
    in_bounds = (u >= 1) & (u <= 510) & (v >= 1) & (v <= 510)
    u = tl.maximum(0, tl.minimum(u, 511))
    v = tl.maximum(0, tl.minimum(v, 511))
    pixel = v * 512 + u
    x = tl.load(xyz + pixel * 3)
    y = tl.load(xyz + pixel * 3 + 1)
    z = tl.load(xyz + pixel * 3 + 2)
    residual = tl.abs(((x * nx + y * ny) + z * nz) + offset)
    member = tl.load(support + pixel) & (residual < 0.0012)
    return u, v, pixel, x, y, z, residual, member, in_bounds


@triton.jit
def _validate_box_edge_legs(
    uv, adjacent, planes, plane_ids, support, depth, valid, xyz, k, neutral,
    output, uv_row: tl.constexpr, uv_col: tl.constexpr,
    adj_row: tl.constexpr, adj_leg: tl.constexpr, adj_col: tl.constexpr,
    plane_row: tl.constexpr, plane_col: tl.constexpr,
    k_row: tl.constexpr, k_col: tl.constexpr,
):
    # One program handles the six taps on both sides of one candidate leg.
    leg_index = tl.program_id(0)
    candidate = leg_index // 2
    leg = leg_index % 2
    tap = tl.arange(0, 8)
    active = tap < 6

    u = tl.load(uv + candidate * uv_row)
    v = tl.load(uv + candidate * uv_row + uv_col)
    end = adjacent + candidate * adj_row + leg * adj_leg
    du = tl.load(end) - u
    dv = tl.load(end + adj_col) - v
    length = tl.sqrt_rn(du * du + dv * dv)
    safe_length = tl.maximum(length, 1e-12)
    direction_u = tl.div_rn(du, safe_length)
    direction_v = tl.div_rn(dv, safe_length)
    # torch.linspace fills the upper half by subtracting from the endpoint.
    step = 0.2
    fraction = tl.where(tap < 3, tap * step, 1.0 - (5 - tap) * step)
    distance = 1.5 + (tl.minimum(length * 0.75, 8.0) - 1.5) * fraction
    center_u = u + distance * direction_u
    center_v = v + distance * direction_v

    plane = planes + tl.load(plane_ids + candidate) * plane_row
    nx = tl.load(plane)
    ny = tl.load(plane + plane_col)
    nz = tl.load(plane + 2 * plane_col)
    offset = tl.load(plane + 3 * plane_col)
    left_u, left_v, left, lx, ly, lz, left_residual, left_member, left_bounds = _edge_samples(
        center_u, center_v, -direction_v, direction_u, -1.0, xyz, support, nx, ny, nz, offset,
    )
    right_u, right_v, right, rx, ry, rz, right_residual, right_member, right_bounds = _edge_samples(
        center_u, center_v, -direction_v, direction_u, 1.0, xyz, support, nx, ny, nz, offset,
    )
    outside_right = tl.sum(left_member & active, 0) >= tl.sum(right_member & active, 0)
    outside = tl.where(outside_right, right, left)
    inside = tl.where(outside_right, left, right)
    inside_member = tl.where(outside_right, left_member, right_member)
    sample_u = tl.where(outside_right, right_u, left_u)
    sample_v = tl.where(outside_right, right_v, left_v)
    away = tl.where(outside_right, right_residual, left_residual) > 0.0012

    ray_u = tl.div_rn(sample_u - tl.load(k + 2 * k_col), tl.load(k))
    ray_v = tl.div_rn(sample_v - tl.load(k + k_row + 2 * k_col), tl.load(k + k_row + k_col))
    denominator = (ray_u * nx + ray_v * ny) + nz
    predicted = tl.div_rn(-offset, tl.where(tl.abs(denominator) > 0.02, denominator, float("nan")))
    delta = tl.load(depth + outside) - predicted
    x_gap, y_gap, z_gap = lx - rx, ly - ry, lz - rz
    gap = (x_gap * x_gap + y_gap * y_gap) + z_gap * z_gap

    outside_valid = tl.load(valid + outside)
    crease = (tl.load(support + outside) | ~tl.load(neutral + outside)) & away & (gap < 0.012**2)
    background = ~outside_valid | ((delta > 0.002) & away)
    occluding = outside_valid & (delta < -0.002) & ~crease
    inside_ok = tl.load(valid + inside) & inside_member
    support_count = tl.sum((crease | background) & inside_ok & active, 0)
    occluding_count = tl.sum(occluding & active, 0)
    in_bounds = tl.sum(left_bounds & right_bounds & active, 0) == 6
    accepted = in_bounds & (length >= 3.0) & (support_count >= 3) & (occluding_count <= 1)
    tl.store(output + leg_index, accepted)


def validate_edge_legs(uv, adjacent, planes, plane_ids, support, depth, valid, xyz, k, neutral):
    """Launch one kernel for all Box vertices, axes, sides and depth tests."""
    result = torch.empty((len(uv), 2), dtype=torch.bool, device=uv.device)
    _validate_box_edge_legs[(2 * len(uv),)](
        uv, adjacent, planes, plane_ids.contiguous(), support.contiguous(),
        depth.contiguous(), valid.contiguous(), xyz.contiguous(), k, neutral.contiguous(),
        result, *uv.stride(), *adjacent.stride(), *planes.stride(), *k.stride(),
        num_warps=1, enable_fp_fusion=False,
    )
    return result
