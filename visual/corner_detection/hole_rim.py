"""Recover corners of enclosed rectangular openings on observed Box faces."""

import cv2
import numpy as np


def _opening_boundary(label_image, label):
    """Return midpoint samples across pixel-cell edges on an opening boundary."""
    opening = label_image == label
    horizontal = opening[:, 1:] != opening[:, :-1]
    vertical = opening[1:] != opening[:-1]
    y_h, x_h = np.nonzero(horizontal)
    y_v, x_v = np.nonzero(vertical)
    return np.column_stack((np.r_[x_h + 0.5, x_v], np.r_[y_h, y_v + 0.5]))


def _plane_points(uv, intrinsic, normal, offset):
    """Intersect integer-center CV pixel rays with a metric camera-space plane."""
    rays = (uv - intrinsic[:2, 2]) / np.diag(intrinsic)[:2]
    rays = np.column_stack((rays, np.ones(len(uv))))
    denominator = rays @ normal
    depth = np.full(len(rays), np.nan)
    np.divide(-offset, denominator, out=depth, where=np.abs(denominator) > 1e-6)
    return rays * depth[:, None]


def _rectangle(boundary):
    """Fit four axis-aligned sides, with independent evidence on each side."""
    low, high = np.quantile(boundary, [0.08, 0.92], axis=0)
    if np.any(high <= low):
        return None, False

    scaled = (boundary - low) / (high - low)
    side_distance = np.column_stack((scaled, 1 - scaled))
    nearest = np.abs(side_distance).argmin(axis=1)
    sides = np.r_[low, high]
    supported = True
    for side in range(4):
        axis = side % 2
        # Use the middle of each leg to avoid mixing samples at the corner.
        cross = scaled[:, 1 - axis]
        values = boundary[(nearest == side) & (cross > 0.18) & (cross < 0.82), axis]
        if len(values) < 3:
            supported = False
            continue
        sides[side] = np.median(values)
        supported &= bool(np.median(np.abs(values - sides[side])) <= 0.002)
    width = sides[2:] - sides[:2]
    supported &= bool(
        np.all((width >= 0.030) & (width <= 0.065))
        and abs(width[0] - width[1]) <= 0.009
    )
    corners = sides[[[0, 1], [0, 3], [2, 1], [2, 3]]]
    return corners, supported


def opening_corners(depth, support, planes, eligible, intrinsic):
    """Yield rim proposals with cumulative rectangularity and depth checks.

    Depth is in metres, and plane normals are unit length. Each row contains
    a plane index, four camera-space corners, and two cumulative gate flags.
    The Box model has a square opening. Its width and position are fit from
    image evidence, using a perpendicular observed Box face only for axes.
    """
    y, x = np.nonzero(support)
    if not len(x):
        return
    height, width = depth.shape
    x0, x1 = max(0, x.min() - 1), min(width, x.max() + 2)
    y0, y1 = max(0, y.min() - 1), min(height, y.max() + 2)
    y_grid, x_grid = np.mgrid[y0:y1, x0:x1]
    uv = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    crop = np.s_[y0:y1, x0:x1]
    for plane in sorted(eligible):
        normal, offset = planes[plane]
        perpendicular = next(
            (
                index
                for index in sorted(eligible)
                if abs(normal @ planes[index][0]) < 0.15
            ),
            None,
        )
        if perpendicular is None:
            continue
        axis = planes[perpendicular][0].copy()
        axis -= normal * (normal @ axis)
        axis /= np.linalg.norm(axis)
        basis = np.stack((axis, np.cross(normal, axis)))

        plane_depth = _plane_points(uv, intrinsic, normal, offset)[:, 2]
        depth_delta = depth[crop] - plane_depth.reshape(x_grid.shape)
        face = support[crop] & (np.abs(depth_delta) <= 0.002)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (~face).astype(np.uint8), connectivity=4
        )
        for label in range(1, count):
            left, top, w, h, area = stats[label]
            if (
                not 18 <= area <= 450
                or left == 0
                or top == 0
                or left + w == face.shape[1]
                or top + h == face.shape[0]
            ):
                continue
            boundary = _opening_boundary(labels, label) + [x0, y0]
            points = _plane_points(boundary, intrinsic, normal, offset)
            if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
                continue
            rectangle, rectangular = _rectangle(points @ basis.T)
            if rectangle is None:
                continue
            corners = rectangle @ basis - offset * normal
            component = labels == label
            opening_delta = depth_delta[component]
            opening_depth = depth[crop][component]
            behind = opening_delta > 0.003
            occluding = (opening_depth > 0) & (opening_delta < -0.003)
            verified = (
                rectangular and behind.mean() >= 0.45 and occluding.mean() <= 0.12
            )
            yield plane, corners, rectangular, verified
