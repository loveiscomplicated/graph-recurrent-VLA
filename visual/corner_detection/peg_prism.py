"""Estimate supported Peg vertices from observed planes and silhouette bounds."""

from itertools import product

import numpy as np
from scipy.spatial import cKDTree


def prism_corners(
    cloud,
    planes,
    *,
    size_memory=None,
    silhouette=None,
    require_end_face=False,
    preserve_transverse_extent=False,
    preserve_lower_silhouette=False,
    trace_bounds=None,
    center_plane_bounds=False,
):
    """Fit a rectangular prism in the axes of two perpendicular observed faces."""
    if len(cloud) < 40:
        return np.empty((0, 3))
    for i, (normal, _) in enumerate(planes):
        for other, _ in planes[i + 1 :]:
            if abs(normal @ other) < 0.10:
                tangent = other - normal * (normal @ other)
                tangent /= np.linalg.norm(tangent)
                basis = np.array([normal, tangent, np.cross(normal, tangent)])
                return _supported_vertices(
                    cloud,
                    planes,
                    basis,
                    size_memory,
                    silhouette,
                    require_end_face,
                    preserve_transverse_extent,
                    trace_bounds,
                    preserve_lower_silhouette,
                    center_plane_bounds,
                )
    return np.empty((0, 3))


def _supported_vertices(
    cloud,
    planes,
    basis,
    size_memory,
    silhouette,
    require_end_face,
    preserve_transverse_extent,
    trace_bounds,
    preserve_lower_silhouette,
    center_plane_bounds,
):
    """Anchor observed bounds to planes and require local depth for each vertex."""
    lower, upper, plane_bounds = _observed_bounds(cloud, planes, basis, center_plane_bounds)
    widths = np.sort(upper - lower)
    if widths[2] < 0.12 or widths[0] <= 0 or widths[1] > 0.06:
        return np.empty((0, 3))
    length_axis = np.argmax(upper - lower)
    if silhouette is not None and (
        not require_end_face or plane_bounds[:, length_axis].any()
    ):
        before = np.stack((lower, upper))
        _refine_silhouette(lower, upper, plane_bounds, basis, silhouette)
        after = np.stack((lower, upper))
        if trace_bounds is not None:
            trace_bounds(basis, before, after, plane_bounds)
        if preserve_transverse_extent:
            transverse = np.arange(3) != length_axis
            lower[transverse] = np.minimum(lower, before[0])[transverse]
            upper[transverse] = np.maximum(upper, before[1])[transverse]
        if preserve_lower_silhouette:
            # CV +Y points down. Leave longitudinal and side-facing bounds eligible.
            down = basis[:, 1]
            transverse = np.arange(3) != length_axis
            lower_side = transverse & (down < -0.5) & (lower > before[0])
            upper_side = transverse & (down > 0.5) & (upper < before[1])
            lower[lower_side] = before[0, lower_side]
            upper[upper_side] = before[1, upper_side]
    if size_memory is not None:
        size_memory.adjust(lower, upper, plane_bounds)
    sides = np.array(list(product((False, True), repeat=3)))
    corners = np.where(sides, upper, lower) @ basis
    distances, _ = cKDTree(cloud).query(corners)
    return corners[distances < 0.006]


def _observed_bounds(cloud, planes, basis, center_plane_bounds):
    """Fit the prism bounds in a given frame using the measured planes."""
    coordinates = cloud @ basis.T
    lower, upper = coordinates.min(0), coordinates.max(0)
    center = (lower + upper) / 2
    plane_bounds = np.zeros((2, 3), dtype=bool)
    for normal, offset in planes:
        alignment = basis @ normal
        axis = np.abs(alignment).argmax()
        if abs(alignment[axis]) < 0.98:
            continue
        if center_plane_bounds:
            # Intersect along this prism axis through the observed cloud center.
            # A fitted normal need not be exactly parallel to the prism axis.
            bound = center[axis] - (offset + alignment @ center) / alignment[axis]
        else:
            bound = -offset / alignment[axis]
        if abs(bound - lower[axis]) < abs(bound - upper[axis]):
            lower[axis] = bound
            plane_bounds[0, axis] = True
        else:
            upper[axis] = bound
            plane_bounds[1, axis] = True
    return lower, upper, plane_bounds


def rear_end_corners(cloud, planes, box_cloud):
    """Recover the end facing away from Box using a plane or two side edges."""
    if len(cloud) < 40 or len(planes) < 2 or not len(box_cloud):
        return np.empty((0, 3))
    toward_box = np.median(box_cloud, axis=0) - np.median(cloud, axis=0)
    toward_box /= max(np.linalg.norm(toward_box), 1e-7)
    basis = None
    alignment = 0.9
    for i, (normal, _) in enumerate(planes):
        for other, _ in planes[i + 1 :]:
            dot = normal @ other
            if abs(dot) >= 0.10:
                continue
            tangent = other - normal * dot
            tangent /= np.linalg.norm(tangent)
            length = np.cross(normal, tangent)
            agreement = length @ toward_box
            if abs(agreement) > alignment:
                alignment = abs(agreement)
                basis = np.array([normal, tangent, np.copysign(1, agreement) * length])
    if basis is None:
        return np.empty((0, 3))

    lower, upper, fitted = _observed_bounds(cloud, planes, basis, True)
    widths = upper - lower
    if widths[2] >= 0.12 or (widths[:2] < 0.025).any() or (widths[:2] > 0.06).any():
        return np.empty((0, 3))
    if not fitted[0, 2]:
        # End-on views can lack enough face pixels for a fitted plane. Require
        # both visible side faces to terminate at the same longitudinal bound.
        coordinates = cloud @ basis.T
        side_bounds = np.where(fitted[0], lower, upper)
        for axis in range(2):
            side = side_bounds[axis]
            edge = coordinates[
                (np.abs(coordinates[:, axis] - side) < 0.0015)
                & (np.abs(coordinates[:, 2] - lower[2]) < 0.004)
            ]
            if len(edge) < 3 or np.ptp(edge[:, 1 - axis]) < 0.55 * widths[1 - axis]:
                return np.empty((0, 3))
    rear_bounds = [(lower[0], upper[0]), (lower[1], upper[1]), (lower[2],)]
    vertices = np.array(list(product(*rear_bounds))) @ basis
    distances, _ = cKDTree(cloud).query(vertices)
    return vertices[distances < 0.006]


class PegSizeMemory:
    """Reuse observed dimensions while estimating pose from the current frame."""

    def __init__(self):
        self.size = None

    def adjust(self, lower, upper, plane_bounds):
        """Extend shortened bounds from a supported face using earlier dimensions."""
        widths = upper - lower
        order = np.argsort(widths)
        if self.size is not None:
            for axis, expected in zip(order, self.size, strict=True):
                missing = expected - widths[axis]
                if not 0.001 < missing <= 0.012:
                    continue
                if plane_bounds[0, axis] and not plane_bounds[1, axis]:
                    upper[axis] += missing
                elif plane_bounds[1, axis] and not plane_bounds[0, axis]:
                    lower[axis] -= missing
        # Only learn from this frame's original bounds.
        observed = widths[order]
        if observed[0] >= 0.025 and observed[2] >= 0.16:
            self.size = (
                observed if self.size is None else np.maximum(self.size, observed)
            )


def _refine_silhouette(lower, upper, plane_bounds, basis, silhouette):
    """Fit free bounds using half-pixel edges with visible background."""
    support, depth, intrinsic = silhouette
    focal = np.diag(intrinsic)[:2]
    center = intrinsic[:2, 2]
    samples = []
    for axis in (0, 1):
        foreground = np.moveaxis(support, axis, 0)
        image_depth = np.moveaxis(depth, axis, 0)
        for direction in (-1, 1):
            inside = slice(1, None) if direction < 0 else slice(None, -1)
            outside = slice(None, -1) if direction < 0 else slice(1, None)
            edge = foreground[inside] & ~foreground[outside]
            observed = image_depth[inside]
            edge &= image_depth[outside] > observed + 0.003
            row, col = np.nonzero(edge)
            boundary = row + 0.5
            u, v = (col, boundary) if axis == 0 else (boundary, col)
            z = observed[row, col]
            uv = np.column_stack((u, v))
            samples.append(np.column_stack(((uv - center) / focal * z[:, None], z)))
    coordinates = np.concatenate(samples) @ basis.T
    for index, bounds in enumerate((lower, upper)):
        for axis in np.flatnonzero(~plane_bounds[index]):
            values = coordinates[:, axis]
            nearby = values[np.abs(values - bounds[axis]) < 0.003]
            if len(nearby) >= 3:
                bounds[axis] = np.median(nearby)
