"""Fit planar RGB-D surfaces from organized pixel neighborhoods."""

import torch
import torch.nn.functional as F


def _sample_normals(xyz, support, *, stride=2):
    """Sample normals on a pixel grid with neighbors two pixels from each center."""
    center = xyz[2:-2:stride, 2:-2:stride]
    left, right = xyz[2:-2:stride, :-4:stride], xyz[2:-2:stride, 4::stride]
    top, bottom = xyz[:-4:stride, 2:-2:stride], xyz[4::stride, 2:-2:stride]
    normal = torch.linalg.cross(right - left, bottom - top)
    magnitude = torch.linalg.vector_norm(normal, dim=-1)
    valid = (
        support[2:-2:stride, 2:-2:stride]
        & support[2:-2:stride, :-4:stride] & support[2:-2:stride, 4::stride]
        & support[:-4:stride, 2:-2:stride] & support[4::stride, 2:-2:stride]
        & (magnitude > 1e-8)
    )
    # Reject stencils crossing a depth edge before accumulating their votes.
    for neighbor in (left, right, top, bottom):
        valid &= (neighbor[..., 2] - center[..., 2]).abs() <= 0.020
    unit = normal[valid] / magnitude[valid].unsqueeze(1)
    points = center[valid]
    unit = torch.where(((unit * points).sum(1) > 0)[:, None], -unit, unit)
    return points, unit


def _plane_seeds(samples, normals, origin, tolerance, maximum):
    """Group local planes by quantized normal and object-centered offset."""
    offsets = -((samples - origin) * normals).sum(1)
    angle_bins = ((normals + 1) / 0.08).round().to(torch.int64)
    depth_bins = (offsets / (4 * tolerance)).round().to(torch.int64)
    keys = (((depth_bins * 27 + angle_bins[:, 0]) * 27 + angle_bins[:, 1]) * 27
            + angle_bins[:, 2])
    _, inverse, counts = torch.unique(keys, return_inverse=True, return_counts=True)
    coefficients = torch.cat((normals, offsets[:, None]), 1)
    sums = samples.new_zeros((len(counts), 4))
    sums.index_add_(0, inverse, coefficients)
    strongest = counts.argsort(descending=True, stable=True)[:32]
    seeds = sums[strongest] / counts[strongest, None]
    seeds = seeds / torch.linalg.vector_norm(seeds[:, :3], dim=1, keepdim=True).clamp_min(1e-8)
    similar = (
        (seeds[:, :3] @ seeds[:, :3].T > 0.996)
        & ((seeds[:, 3, None] - seeds[:, 3]).abs() <= 4 * tolerance)
    )
    distinct = ~similar.tril(diagonal=-1).any(1)
    keep = distinct & (counts[strongest] >= 3)
    return seeds[keep][:maximum]


def fit_planes(xyz, support, points, parameters, maximum):
    """Assign points to the strongest local plane groups and fit each once."""
    if len(points) < parameters.minimum_plane_pixels:
        return points.new_empty((0, 4))
    samples, normals = _sample_normals(xyz, support)
    return fit_from_normals(points, samples, normals, parameters, maximum)


def fit_pca_planes(points, pixels, shape, parameters, maximum):
    """Use the edge detector's CUDA PCA5 normals to seed the shared plane fit."""
    if len(points) < parameters.minimum_plane_pixels:
        return points.new_empty((0, 4))
    from visual.edge_first.cuda_evidence import DepthJump, compact_rows, fused_normals_compact

    rows = compact_rows(pixels, shape)
    # All compact points belong to the current object mask.
    objects = torch.zeros_like(pixels)
    normals, valid = fused_normals_compact(points, pixels, objects, rows, DepthJump())
    samples_and_normals = torch.cat((points, normals), dim=1)[valid]
    samples, normals = samples_and_normals.split(3, dim=1)
    return fit_from_normals(points, samples, normals, parameters, maximum)


def fit_from_normals(points, samples, normals, parameters, maximum):
    """Group oriented samples and update each plane with one batched fit."""
    if not len(samples):
        return points.new_empty((0, 4))
    origin = points.mean(0)
    tolerance = parameters.plane_tolerance_m
    seeds = _plane_seeds(samples, normals, origin, tolerance, maximum)
    if not len(seeds):
        return points.new_empty((0, 4))

    residual = ((points - origin) @ seeds[:, :3].T + seeds[:, 3]).abs()
    error, group = residual.min(dim=1)
    membership = F.one_hot(group, num_classes=len(seeds)).T.bool()
    membership &= error <= 2 * tolerance
    count = membership.sum(1)
    enough = count >= parameters.minimum_plane_pixels
    if not bool(enough.any()):
        return points.new_empty((0, 4))

    weight = membership[enough].to(points.dtype)
    sizes = count[enough]
    centers = (weight @ points) / sizes[:, None]
    centered = points[None] - centers[:, None]
    covariance = centered.transpose(1, 2) @ (centered * weight[:, :, None])
    _, vectors = torch.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    planes = torch.cat((normals, -(normals * centers).sum(1, keepdim=True)), 1)
    return planes[sizes.argsort(descending=True, stable=True)]
