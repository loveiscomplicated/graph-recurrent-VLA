"""Tensor geometry and Box occlusion checks for observed Peg ends."""

import torch


def _median(points):
    """Match the midpoint convention of NumPy's median for even counts."""
    ordered = points.sort(dim=0).values
    return (ordered[(len(points) - 1) // 2] + ordered[len(points) // 2]) * 0.5


def _orthogonal_bases(normals, pairs):
    """Build a local orthogonal basis per observed plane pair."""
    first, other = normals[pairs[:, 0]], normals[pairs[:, 1]]
    dot = (first * other).sum(dim=1)
    tangent = other - first * dot[:, None]
    tangent_norm = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    tangent = tangent / tangent_norm.clamp_min(1e-7)
    length = torch.linalg.cross(first, tangent)
    return torch.stack((first, tangent, length), dim=1), dot.abs() < 0.10


def _observed_bounds(coordinates, planes, basis, centered, *, optimized=False):
    """Snap measured cloud bounds to aligned planes in input order."""
    lower, upper = coordinates.amin(dim=0), coordinates.amax(dim=0)
    center = (lower + upper) * 0.5
    alignments = planes[:, :3] @ basis.T
    strongest = alignments.abs().amax(dim=1, keepdim=True)
    fitted_lower = torch.zeros_like(lower, dtype=torch.bool)
    if optimized:
        fits = (alignments.abs() == strongest) & (alignments.abs() >= 0.98)
        denominator = torch.where(fits, alignments, 1)
        if centered:
            delta = planes[:, 3] + (alignments * center).sum(dim=1)
            bounds = center - delta[:, None] / denominator
        else:
            bounds = -planes[:, 3:4] / denominator
        for bound, fit in zip(bounds, fits, strict=True):
            use_lower = (bound - lower).abs() < (bound - upper).abs()
            lower = torch.where(fit & use_lower, bound, lower)
            upper = torch.where(fit & ~use_lower, bound, upper)
            fitted_lower = fitted_lower | (fit & use_lower)
        return lower, upper, fitted_lower

    for alignment, offset, best in zip(alignments, planes[:, 3], strongest, strict=True):
        fits = (alignment.abs() == best) & (alignment.abs() >= 0.98)
        denominator = torch.where(fits, alignment, 1)
        if centered:
            bound = center - (offset + (alignment * center).sum()) / denominator
        else:
            bound = -offset / denominator
        use_lower = (bound - lower).abs() < (bound - upper).abs()
        lower = torch.where(fits & use_lower, bound, lower)
        upper = torch.where(fits & ~use_lower, bound, upper)
        fitted_lower = fitted_lower | (fits & use_lower)

    return lower, upper, fitted_lower


def prism_basis(planes, pairs):
    """Select the first perpendicular plane pair used by the prism fit."""
    bases, perpendicular = _orthogonal_bases(planes[:, :3], pairs)
    first = perpendicular.to(torch.int32).argmax().reshape(1)
    basis = bases.index_select(0, first).squeeze(0)
    return basis, perpendicular.any()


def prism_candidates(cloud, planes, pairs, sides, *, centered, optimized=False, frame_basis=None):
    """Return eight prism vertices and their local depth support flags."""
    basis, perpendicular = prism_basis(planes, pairs) if frame_basis is None else frame_basis
    lower, upper, _ = _observed_bounds(
        cloud @ basis.T, planes, basis, centered, optimized=optimized
    )
    widths = (upper - lower).sort().values
    valid = perpendicular & (widths[2] >= 0.12) & (widths[0] > 0)
    valid = valid & (widths[1] <= 0.06)
    points = torch.where(sides, upper, lower) @ basis
    return points, valid & depth_supported(points, cloud, 0.006)


def rear_candidates(cloud, planes, box_cloud, pairs, transverse_sides, *, optimized=False):
    """Return four rear candidates and a flag for credible end geometry."""
    toward_box = _median(box_cloud) - _median(cloud)
    toward_box = toward_box / torch.linalg.vector_norm(toward_box).clamp_min(1e-7)
    bases, perpendicular = _orthogonal_bases(planes[:, :3], pairs)
    agreement = (bases[:, 2] * toward_box).sum(dim=1)
    ranks = torch.where(perpendicular, agreement.abs(), -1)
    chosen = ranks.argmax().reshape(1)
    basis = bases.index_select(0, chosen).squeeze(0)
    direction = agreement.index_select(0, chosen)
    basis = torch.cat((basis[:2], basis[2:] * torch.where(direction >= 0, 1, -1)))

    coordinates = cloud @ basis.T
    lower, upper, fitted_lower = _observed_bounds(coordinates, planes, basis, True, optimized=optimized)
    widths = upper - lower
    valid = (ranks.max() > 0.9) & (widths[2] < 0.12)
    valid = valid & ((widths[:2] >= 0.025) & (widths[:2] <= 0.06)).all()
    side_bounds = torch.where(fitted_lower, lower, upper)
    end_band = (coordinates[:, 2] - lower[2]).abs() < 0.004
    side_edges = (coordinates[:, :2] - side_bounds[:2]).abs() < 0.0015
    side_edges = side_edges & end_band[:, None]
    across = coordinates[:, :2].flip(dims=(1,))
    edge_low = torch.where(side_edges, across, torch.inf).amin(dim=0)
    edge_high = torch.where(side_edges, across, -torch.inf).amax(dim=0)
    required_span = 0.55 * widths[:2].flip(dims=(0,))
    shared_end = (side_edges.sum(dim=0) >= 3) & (edge_high - edge_low > required_span)
    valid = valid & (fitted_lower[2] | shared_end.all())

    rear = torch.where(transverse_sides, upper[:2], lower[:2])
    points = torch.cat((rear, lower[2].expand(len(rear), 1)), dim=1) @ basis
    return points, valid


def depth_supported(points, cloud, radius):
    """Check four candidate neighborhoods without transferring the cloud to CPU."""
    squared_distances = (points[:, None, :] - cloud[None, :, :]).square().sum(dim=2)
    return squared_distances.amin(dim=1) < radius ** 2


def unoccluded(points, uv, depth, box_support):
    """Reject rays outside the image or covered by closer Box depth."""
    height, width = depth.shape
    u, v = uv.round().long().T
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u.clamp(0, width - 1), v.clamp(0, height - 1)
    covered = box_support[v, u] & (depth[v, u] < points[:, 2])
    return inside & ~covered

def box_cut_vertices(
    points, planes, basis, box_points, depth, peg_support, box_support, intrinsic
):
    """Find Box-facing prism vertices on an occlusion cut through the Peg."""
    coordinates = points @ basis.T
    span = coordinates.amax(dim=0) - coordinates.amin(dim=0)
    length_index = span.argmax().reshape(1)
    axis = basis.index_select(0, length_index).squeeze(0)
    toward_box = box_points.mean(dim=0) - points.mean(dim=0)
    axis = axis * torch.where(axis @ toward_box >= 0, 1, -1)
    axial = points @ axis
    at_end = (span.max() >= 0.12) & (axial >= axial.max() - 0.004)

    end_plane = (planes[:, :3] @ axis).abs() >= 0.98
    residuals = points @ planes[:, :3].T + planes[:, 3]
    observed_end = (end_plane & (residuals.abs() < 0.004)).any(dim=1)
    unverified = at_end & ~observed_end

    # Probe either side of the candidate and just beyond the apparent end.
    # A calibrated vertex can project onto Peg even when Box clips the edge.
    offsets = points.new_tensor([-0.001, 0.00075, 0.0015, 0.003])
    probes = points[:, None, :] + offsets[:, None] * axis
    uv = probes[..., :2] / probes[..., 2:] * intrinsic.diagonal()[:2] + intrinsic[:2, 2]
    height, width = depth.shape
    u, v = uv.round().long().unbind(dim=-1)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u.clamp(0, width - 1), v.clamp(0, height - 1)
    gap = probes[..., 2] - depth[v, u]
    on_box = inside & box_support[v, u]
    straddles_box = (
        on_box[:, 0] & (gap[:, 0] > 0.003)
        & on_box[:, 2] & (gap[:, 2] > 0.011)
    )
    peg_beyond = inside[:, 1] & peg_support[v[:, 1], u[:, 1]] & (gap[:, 1].abs() < 0.003)
    covered_beyond = on_box[:, 3] & (gap[:, 3] > 0.012)
    # For a shared cut, require corroboration from multiple vertices. Preserve a
    # partially visible end when another corner still has Peg depth beyond it.
    shared_cut = (
        (unverified.sum() >= 2)
        & (~at_end | (unverified & covered_beyond)).all()
        & ~(at_end & peg_beyond).any()
    )
    return unverified & (straddles_box | shared_cut)
