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


def _rectangle(boundary, *, small_opening=False, fragmented=False, square_span="mean"):
    """Fit four axis-aligned sides, with independent evidence on each side."""
    low, high = np.quantile(boundary, [0.08, 0.92], axis=0)
    if np.any(high <= low):
        return None, False

    scaled = (boundary - low) / (high - low)
    side_distance = np.column_stack((scaled, 1 - scaled))
    nearest = np.abs(side_distance).argmin(axis=1)
    sides = np.r_[low, high]
    minimum_support = 2 if small_opening else 3
    residual_limit = 0.0028 if fragmented else 0.002
    supported = True
    for side in range(4):
        axis = side % 2
        # Use the middle of each leg to avoid mixing samples at the corner.
        cross = scaled[:, 1 - axis]
        margin = 0.18 if square_span == "mean" else 0.08
        values = boundary[
            (nearest == side) & (cross > margin) & (cross < 1 - margin), axis
        ]
        if len(values) < minimum_support:
            supported = False
            continue
        sides[side] = np.median(values)
        supported &= bool(np.median(np.abs(values - sides[side])) <= residual_limit)
    width = sides[2:] - sides[:2]
    size_tolerance = 0.015 if small_opening else 0.009
    if fragmented:
        size_tolerance = 0.019
    supported &= bool(
        np.all((width >= 0.030) & (width <= 0.065))
        and abs(width[0] - width[1]) <= size_tolerance
    )
    if small_opening:
        center = (sides[:2] + sides[2:]) / 2
        span = {"mean": np.mean, "short": np.min, "long": np.max}[square_span]
        half_size = span(width) / 2
        sides = np.r_[center - half_size, center + half_size]
    corners = sides[[[0, 1], [0, 3], [2, 1], [2, 3]]]
    return corners, supported


def _depth_supported_square(corners, depth, support, intrinsic, plane):
    """Check a fitted opening using depth on both sides of all four edges."""
    center = corners.mean(axis=0)
    axes = corners[[2, 1]] - corners[0]
    width = np.linalg.norm(axes, axis=1)
    if np.any((width < 0.029) | (width > 0.058)):
        return None
    unit = axes / width[:, None]
    points = []
    for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
        edge = center + sign * axes[axis] / 2
        along = np.linspace(-0.35, 0.35, 7)[:, None] * axes[1 - axis]
        offsets = sign * np.array([-0.008, -0.005, 0.005, 0.008])
        points.append(edge + along[None] + offsets[:, None, None] * unit[axis])
    camera = np.asarray(points).reshape(-1, 3)
    uv = camera[:, :2] / camera[:, 2:] * intrinsic.diagonal()[:2] + intrinsic[:2, 2]
    pixels = np.rint(uv).astype(int)
    u, v = pixels.T
    if np.any((u < 0) | (v < 0) | (u >= depth.shape[1]) | (v >= depth.shape[0])):
        return None
    expected_depth = _plane_points(pixels, intrinsic, *plane)[:, 2]
    observed_depth = depth[v, u]
    delta = observed_depth - expected_depth
    # Sample the two offsets inside and outside each edge independently.
    face_samples = support[v, u] & (np.abs(delta) <= 0.002)
    recess_samples = (delta > 0.003) & (observed_depth < 4)
    face = face_samples.reshape(4, 4, 7)[:, 2:].mean(axis=(1, 2))
    recess = recess_samples.reshape(4, 4, 7)[:, :2].mean(axis=(1, 2))
    if face.min() < 0.40 or face.mean() < 0.65:
        return None
    if recess.min() < 0.40 or recess.mean() < 0.65:
        return None
    return float(face.mean() + recess.mean())


def opening_corners(
    depth,
    support,
    planes,
    eligible,
    intrinsic,
    *,
    small_opening=False,
    fragmented=False,
    square_span="mean",
    verify_rejected_square=False,
):
    """Try enclosed contours first, then depth components with real face edges."""
    inputs = (depth, support, planes, eligible, intrinsic)
    proposals = list(_opening_corners(*inputs, small_opening, fragmented, square_span))
    yield from proposals
    if fragmented and not any(row[3] for row in proposals):
        depth_proposals = list(
            _opening_corners(*inputs, small_opening, fragmented, square_span, depth_component=True)
        )
        yield from depth_proposals
        if verify_rejected_square and not any(row[3] for row in depth_proposals):
            supported = []
            for plane, corners, _, _ in [*proposals, *depth_proposals]:
                evidence = _depth_supported_square(
                    corners, depth, support, intrinsic, planes[plane]
                )
                if evidence is not None:
                    supported.append((evidence, plane, corners))
            if supported:
                _, plane, corners = max(supported, key=lambda row: row[0])
                yield plane, corners, True, True


def _opening_corners(
    depth,
    support,
    planes,
    eligible,
    intrinsic,
    small_opening,
    fragmented,
    square_span,
    *,
    depth_component=False,
):
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
    minimum_area = 9 if small_opening else 18
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
        region = ~face
        if depth_component:
            # Split the recess from exterior missing pixels using measured depth.
            nearby_face = cv2.dilate(face.astype(np.uint8), np.ones((9, 9), np.uint8))
            region = (depth_delta > 0.003) & (depth[crop] < 4) & (nearby_face != 0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            region.astype(np.uint8), connectivity=4
        )
        for label in range(1, count):
            left, top, w, h, area = stats[label]
            if (
                not minimum_area <= area <= 450
                or left == 0
                or top == 0
                or left + w == face.shape[1]
                or top + h == face.shape[0]
            ):
                continue
            boundary = _opening_boundary(labels, label)
            if depth_component:
                # Reject crop/dilation boundaries that border no observed face.
                lower = np.floor(boundary).astype(int)
                upper = np.ceil(boundary).astype(int)
                touches_face = (
                    face[lower[:, 1], lower[:, 0]] | face[upper[:, 1], upper[:, 0]]
                )
                if touches_face.sum() < 9 or touches_face.mean() < 0.5:
                    continue
                boundary = boundary[touches_face]
            boundary += [x0, y0]
            points = _plane_points(boundary, intrinsic, normal, offset)
            if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
                continue
            rectangle, rectangular = _rectangle(
                points @ basis.T,
                small_opening=small_opening,
                fragmented=fragmented,
                square_span=square_span,
            )
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


def verified_long_rim(depth, support, planes, eligible, intrinsic):
    """Fit the longer observed span and keep the strongest four-edge depth fit."""
    inputs = (depth, support, planes, eligible, intrinsic)
    best_corners = None
    best_evidence = -np.inf
    for depth_component in (False, True):
        for plane, corners, _, _ in _opening_corners(
            *inputs, True, True, "long", depth_component=depth_component
        ):
            evidence = _depth_supported_square(
                corners, depth, support, intrinsic, planes[plane]
            )
            if evidence is not None and evidence > best_evidence:
                best_corners = corners
                best_evidence = evidence
    return best_corners
