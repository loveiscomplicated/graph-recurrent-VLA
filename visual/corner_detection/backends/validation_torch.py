"""Tensor-only candidate validation shared by eager and compiled detectors."""

from functools import lru_cache, partial

import torch

from . import torch_ops as K


def boundary(
    uv, adjacent, pi, boundary, short, data, object_mask, depth, valid, xyz, k,
    neutral, maximum_support_gap_px, maximum_angle_cosine, physical_edges,
):
    """Validate contour-segment corners against depth and physical edges."""
    normal, offset = data.planes[pi, :3], data.planes[pi, 3]
    ray = K.rays(uv, k)
    denom = (ray * normal).sum(-1)
    points = ray * (-offset / denom)[:, None]
    aray = K.rays(adjacent, k)
    arms = (
        aray * (-offset[:, None] / (aray * normal[:, None]).sum(-1))[..., None]
        - points[:, None]
    )
    cosine = (arms[:, 0] * arms[:, 1]).sum(-1).abs() / (
        torch.linalg.vector_norm(arms[:, 0], dim=-1)
        * torch.linalg.vector_norm(arms[:, 1], dim=-1)
        + 1e-12
    )
    plane_patch, inside, pix = K.sample_patch(data.masks[0], uv, 3)
    plane_patch = data.masks[pi[:, None, None], pix[..., 1], pix[..., 0]] & inside
    distance = (pix.to(uv.dtype) - uv[:, None, None]).square().sum(-1)
    gap = (
        torch.where(plane_patch, distance, torch.inf)
        .flatten(1)
        .min(1)
        .values.sqrt()
    )
    patch_valid = valid[pix[..., 1], pix[..., 0]] & inside
    occluded = (
        patch_valid
        & (depth[pix[..., 1], pix[..., 0]] < points[:, 2, None, None] - 0.020)
    ).sum((1, 2)) >= 3
    cloud = xyz[pix[..., 1], pix[..., 0]]
    contact = (
        patch_valid
        & neutral[pix[..., 1], pix[..., 0]]
        & ~object_mask[pix[..., 1], pix[..., 0]]
    )
    contact &= (cloud - points[:, None, None]).square().sum(-1) < 0.006**2
    pixel = torch.round(uv)
    base = (
        (denom.abs() >= 0.08)
        & (points[:, 2] > 0)
        & ((pixel >= 3) & (pixel < 510)).all(1)
    )
    base &= (
        (plane_patch.sum((1, 2)) >= 5)
        & (gap <= maximum_support_gap_px)
        & (contact.sum((1, 2)) < 3)
    )
    snapped, supporting = K.snap(points, pi, data.planes, data.eligible)
    image = K.project(snapped, k)
    core = base & boundary & (cosine <= 0.28) & ~occluded
    if data.recovery:
        physical, _ = physical_edges(
            uv,
            adjacent,
            data.planes,
            pi,
            data.masks,
            object_mask,
            depth,
            valid,
            xyz,
            k,
            neutral,
        )
        observed = K.visible(points, uv, data.support, xyz)
        selected = base & (cosine <= maximum_angle_cosine)
        selected &= (boundary & (cosine <= 0.28)) | physical
        selected &= ~occluded | (physical & observed)
        selected &= boundary | (
            ((image - uv).square().sum(-1) <= 4)
            & K.visible(snapped, image, data.support, xyz)
        )
        selected &= ~short | ((uv >= 8) & (uv < 504)).all(1)
        selected &= K.unoccluded(snapped, image, depth, valid)
        return snapped, image, supporting, core, selected
    return snapped, image, supporting, core, core


def junctions(
    points, indices, data, depth, valid, xyz, k,
):
    """Check depth support at intersections of three visible planes."""
    uv = K.project(points, k)
    selected = (points[:, 2] > 0) & (
        (torch.round(uv) >= 6) & (torch.round(uv) < 506)
    ).all(1)
    support, inside, pix = K.sample_patch(data.support, uv, 4)
    nearby = (
        xyz[pix[..., 1], pix[..., 0]] - points[:, None, None]
    ).square().sum(-1) < 0.006**2
    for i in range(3):
        ids = indices[:, i]
        selected &= (
            data.masks[ids[:, None, None], pix[..., 1], pix[..., 0]]
            & support
            & inside
            & nearby
        ).sum((1, 2)) >= 3
        _, _, corepix = K.sample_patch(data.support, uv, 6)
        selected &= (
            data.cores[ids[:, None, None], corepix[..., 1], corepix[..., 0]].sum(
                (1, 2)
            )
            >= 2
        )
    local_depth, _, pix = K.sample_patch(depth, uv, 1)
    selected &= (
        valid[pix[..., 1], pix[..., 0]]
        & (local_depth < points[:, 2, None, None] - 0.004)
    ).sum((1, 2)) < 2
    recovered = selected
    if data.recovery:
        recovered = selected & K.unoccluded(points, uv, depth, valid)
    return uv, selected, recovered


def envelopes(
    points, uv, adjacent, pi, data, object_mask, depth, valid, xyz, k,
    neutral, physical_edges,
):
    """Validate rectangular envelope candidates on the observed face."""
    physical, _ = physical_edges(
        uv,
        adjacent,
        data.planes,
        pi,
        data.masks,
        object_mask,
        depth,
        valid,
        xyz,
        k,
        neutral,
    )
    selected = physical & K.visible(points, uv, data.support, xyz)
    _, _, pix = K.sample_patch(depth, uv, 2)
    contact = (
        valid[pix[..., 1], pix[..., 0]]
        & neutral[pix[..., 1], pix[..., 0]]
        & ~object_mask[pix[..., 1], pix[..., 0]]
    )
    contact &= (
        xyz[pix[..., 1], pix[..., 0]] - points[:, None, None]
    ).square().sum(-1) < 0.006**2
    points, count = K.snap(points, pi, data.planes, data.eligible)
    image = K.project(points, k)
    selected &= (contact.sum((1, 2)) < 3) & ((image >= 8) & (image < 504)).all(1)
    selected &= ((image - uv).square().sum(-1) <= 4) & K.visible(
        points, image, data.support, xyz
    )
    selected &= K.unoccluded(points, image, depth, valid)
    return points, image, count, selected


def box_rgb(
    uv, pi, axes, signs, distances, data, depth, valid, xyz, k, neutral,
    physical_edges,
):
    """Validate RGB-feature corners against plane and quadrant evidence."""
    n, d = data.planes[pi, :3], data.planes[pi, 3]
    ray = K.rays(uv, k)
    denom = (ray * n).sum(-1)
    points = ray * (-d / denom)[:, None]
    selected = (
        (denom.abs() > 0.1)
        & (points[:, 2] > 0)
        & K.visible(points, uv, data.support, xyz)
    )
    selected &= K.unoccluded(points, uv, depth, valid)
    s = signs
    delta = s[:, None] * distances[None]
    probe = points[:, None, None] + torch.einsum("sdt,ntc->nsdc", delta, axes)
    pix0 = K.project(probe, k).round().to(torch.int64)
    in_image = ((pix0 >= 1) & (pix0 <= 510)).all((2, 3))
    pix = pix0.clamp(0, 511)
    cloud = xyz[pix[..., 1], pix[..., 0]]
    evidence = data.support[pix[..., 1], pix[..., 0]] & (
        (cloud * n[:, None, None]).sum(-1) + d[:, None, None]
    ).abs().lt(0.0012)
    fractions = torch.where(in_image, evidence.float().mean(-1), 0.5)
    occupied = fractions >= 0.75
    empty = fractions <= 0.25
    exterior = (occupied.sum(1) == 1) & (empty.sum(1) == 3)
    interior = (occupied.sum(1) == 3) & (empty.sum(1) == 1)
    selected &= exterior | interior
    sign_id = torch.where(
        exterior,
        occupied.to(torch.int64).argmax(1),
        empty.to(torch.int64).argmax(1),
    )
    chosen = s[sign_id]
    adjacent = K.project(points[:, None] + chosen[..., None] * 0.015 * axes, k)
    physical, _ = physical_edges(
        uv,
        adjacent,
        data.planes,
        pi,
        data.masks,
        data.support,
        depth,
        valid,
        xyz,
        k,
        neutral,
    )
    selected &= physical
    _, _, patch = K.sample_patch(depth, uv, 2)
    contact = (
        valid[patch[..., 1], patch[..., 0]]
        & neutral[patch[..., 1], patch[..., 0]]
        & ~data.support[patch[..., 1], patch[..., 0]]
    )
    contact &= (
        xyz[patch[..., 1], patch[..., 0]] - points[:, None, None]
    ).square().sum(-1) < 0.006**2
    selected &= contact.sum((1, 2)) < 3
    return points, selected, interior


def box_cube_coordinates(points, features, eligible, data, k, *, optimize_cuda=False):
    """Preserve eager plane selection and RGB snapping for Box vertices."""
    if optimize_cuda:
        eligible_planes = data.planes.index_select(0, eligible)
        normals, offsets = eligible_planes[:, :3], eligible_planes[:, 3]
        distances = (points @ normals.T + offsets).abs()
        distance, nearest_index = distances.min(1)
    else:
        distances = (
            points @ data.planes[eligible, :3].T + data.planes[eligible, 3]
        ).abs()
        nearest_index = distances.argmin(1)
    nearest = eligible[nearest_index]
    snapped, _ = K.snap(
        points, nearest, data.planes, data.eligible, check_errors=not optimize_cuda
    )
    if not optimize_cuda:
        distance = distances.min(1).values
    points = torch.where((distance < 0.003)[:, None], snapped, points)
    uv = K.project(points, k)
    if len(features):
        f = features
        diff = (uv[:, None] - f[None]).square().sum(-1)
        if optimize_cuda:
            distance, closest_index = diff.min(1)
            closest = f[closest_index]
            close = distance < 2.5**2
            distances = (points @ normals.T + offsets).abs()
        else:
            closest = f[diff.argmin(1)]
            close = diff.min(1).values < 2.5**2
            distances = (
                points @ data.planes[eligible, :3].T + data.planes[eligible, 3]
            ).abs()
        pi = eligible[distances.argmin(1)]
        normal = data.planes[pi, :3]
        ray = K.rays(closest, k)
        denom = (ray * normal).sum(-1)
        refined = ray * (-data.planes[pi, 3] / denom)[:, None]
        refine = (
            close
            & (denom.abs() > 0.1)
            & ((refined - points).square().sum(-1) < 0.005**2)
        )
        points = torch.where(refine[:, None], refined, points)
        uv = torch.where(refine[:, None], closest, uv)
        image_corner = (uv[:, None] - f[None]).square().sum(-1).min(
            1
        ).values < 2.5**2
    else:
        image_corner = torch.zeros(8, dtype=torch.bool, device=points.device)
    return points, uv, image_corner


def box_cube(
    points, uv, image_corner, rotation, signs, data, depth, valid, xyz, k,
    neutral, physical_edges, *, validate_edge_legs=None,
):
    """Validate the eight fitted Box vertices against visible geometry."""
    visible = K.visible(points, uv, data.support, xyz)
    observed, inside, pix = K.sample_patch(data.support, uv, 3)
    nearby = (
        xyz[pix[..., 1], pix[..., 0]] - points[:, None, None]
    ).square().sum(-1) < 0.006**2
    selected = (
        visible | (image_corner & (observed & inside & nearby).any((1, 2)))
    ) & K.unoccluded(points, uv, depth, valid)
    physical_any = torch.zeros(8, dtype=torch.bool, device=points.device)
    if validate_edge_legs is None:
        actual_masks = (
            (xyz.reshape(-1, 3) @ data.planes[:, :3].T + data.planes[:, 3]).abs()
            < 0.0012
        ).T.reshape(-1, 512, 512) & data.support
    else:
        edge_endpoints = []
        plane_ids = []
        matched = []
    for ai in range(3):
        normals = data.planes[:, :3]
        target = rotation[:, ai]
        match = (
            ((normals @ target).abs() > 0.97)[None]
            & ((points @ normals.T + data.planes[:, 3]).abs() < 0.0025)
            & data.eligible[None]
        )
        pi = match.to(torch.int64).argmax(1)
        tangent = [j for j in range(3) if j != ai]
        adjacent = (
            points[:, None]
            - signs[:, tangent][..., None]
            * 0.015
            * rotation[:, tangent].T[None]
        )
        adjacent_uv = K.project(adjacent, k)
        if validate_edge_legs is None:
            physical, _ = physical_edges(
                uv,
                adjacent_uv,
                data.planes,
                pi,
                actual_masks,
                data.support,
                depth,
                valid,
                xyz,
                k,
                neutral,
            )
            physical_any |= match.any(1) & physical
        else:
            edge_endpoints.append(adjacent_uv)
            plane_ids.append(pi)
            matched.append(match.any(1))
    if validate_edge_legs is not None:
        legs = validate_edge_legs(
            uv.repeat(3, 1), torch.cat(edge_endpoints), data.planes,
            torch.cat(plane_ids), data.support, depth, valid, xyz, k, neutral,
        )
        physical = legs.reshape(3, 8, 2).all(-1)
        physical_any = (torch.stack(matched) & physical).any(0)
    selected &= physical_any | image_corner
    return selected


@lru_cache(maxsize=None)
def validation_function(name, compiled):
    """Resolve one validation stage and share its compiled CUDA graph."""
    stages = {
        "boundary": boundary,
        "junctions": junctions,
        "envelopes": envelopes,
        "box_rgb": box_rgb,
        "box_cube": box_cube,
    }
    if name == "box_cube_sampled":
        from .box_edges_cuda import validate_edge_legs

        function = partial(box_cube, validate_edge_legs=validate_edge_legs)
    else:
        function = stages[name]
    if not compiled:
        return function
    return torch.compile(function, dynamic=True, fullgraph=True)


@lru_cache(maxsize=None)
def contour_frame_function(kind):
    """Compile one validation kind across the frame's plane groups."""
    if kind == "boundary":
        def validate(groups, depth, valid, xyz, k, neutral, support_gap, max_angle, physical):
            return tuple(
                boundary(
                    *args, data, object_mask, depth, valid, xyz, k, neutral,
                    support_gap, max_angle, physical,
                )
                for args, data, object_mask in groups
            )
    elif kind == "junctions":
        def validate(groups, depth, valid, xyz, k, neutral, support_gap, max_angle, physical):
            return tuple(
                junctions(*args, data, depth, valid, xyz, k)
                for args, data, _ in groups
            )
    else:
        def validate(groups, depth, valid, xyz, k, neutral, support_gap, max_angle, physical):
            return tuple(
                envelopes(
                    *args, data, object_mask, depth, valid, xyz, k, neutral, physical
                )
                for args, data, object_mask in groups
            )
    return torch.compile(validate, dynamic=True, fullgraph=True)
