"""NumPy camera geometry, observed planes, visibility and physical-edge checks.

No simulator, ground truth or model inputs are used by these inference helpers.
Pixel centers follow the detector's integer-u/v convention.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Parameters:
    plane_tolerance_m: float = 0.0010
    minimum_plane_pixels: int = 35
    ransac_trials: int = 160
    polygon_epsilon_px: float = 2.0
    minimum_edge_px: float = 7.0
    maximum_line_residual_px: float = 0.8
    maximum_support_gap_px: float = 2.5
    merge_radius_m: float = 0.0045
    maximum_angle_cosine: float = 0.45
    additional_merge_radius_m: float = 0.010


def color_masks(rgb, minimum_component_pixels=35):
    """Known orange/blue-white peg and yellow box materials, from RGB only."""
    hsv = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = np.moveaxis(hsv, -1, 0)
    orange = (h <= 8) & (s >= 85) & (s <= 190) & (v >= 60)
    white = (h >= 90) & (h <= 105) & (s >= 8) & (s <= 22) & (v >= 65)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (orange | white).astype(np.uint8), 8
    )
    peg = np.zeros(h.shape, bool)
    # White-only visible pieces are valid even when the orange end is occluded.
    # Requiring an orange seed discarded the entire visible tail in v1.
    if n > 1:
        peg = np.isin(labels, 1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= 35))
    box = (h >= 14) & (h <= 30) & (s >= 65) & (s <= 195) & (v >= 50)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(box.astype(np.uint8), 8)
    if n > 1:
        # Disconnected visible pieces are retained: robot occlusion can split a box.
        box = np.isin(
            labels,
            1 + np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= minimum_component_pixels),
        )
    return np.stack((peg, box))


def unproject(depth_m, intrinsic):
    v, u = np.indices(depth_m.shape)
    return np.stack(
        (
            (u - intrinsic[0, 2]) * depth_m / intrinsic[0, 0],
            (v - intrinsic[1, 2]) * depth_m / intrinsic[1, 1],
            depth_m,
        ),
        -1,
    )


def _plane_fit(xyz):
    center = xyz.mean(0)
    _, _, vh = np.linalg.svd(xyz - center, full_matrices=False)
    normal = vh[-1]
    return normal, -normal @ center


def _planes(points, params, rng, maximum):
    remaining = np.arange(len(points))
    found = []
    for _ in range(maximum):
        if len(remaining) < params.minimum_plane_pixels:
            break
        sample = remaining[
            rng.choice(len(remaining), min(len(remaining), 1300), replace=False)
        ]
        triples = rng.choice(len(sample), (params.ransac_trials, 3), replace=True)
        a, b, c = np.moveaxis(points[sample[triples]], 1, 0)
        normals = np.cross(b - a, c - a)
        lengths = np.linalg.norm(normals, axis=1)
        ok = lengths > 1e-7
        normals = normals[ok] / lengths[ok, None]
        if not len(normals):
            break
        offsets = -np.einsum("ij,ij->i", normals, a[ok])
        residual = np.abs(points[sample] @ normals.T + offsets)
        _best = np.argmin(residual, axis=0)
        counts = (residual < params.plane_tolerance_m).sum(0)
        winner = int(counts.argmax())
        mask = (
            np.abs(points[remaining] @ normals[winner] + offsets[winner])
            < params.plane_tolerance_m
        )
        selected = remaining[mask]
        for _ in range(3):
            if len(selected) < params.minimum_plane_pixels:
                break
            n, d = _plane_fit(points[selected])
            selected = remaining[
                np.abs(points[remaining] @ n + d) < params.plane_tolerance_m
            ]
        if len(selected) < params.minimum_plane_pixels:
            break
        n, d = _plane_fit(points[selected])
        # Opposite parallel surfaces remain distinct planes.
        found.append((n, float(d)))
        remaining = remaining[~np.isin(remaining, selected)]
    return found


def _line(points):
    center = points.mean(0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]])
    residual = np.abs((points - center) @ normal)
    keep = residual < 1.3
    if keep.sum() >= 5 and not keep.all():
        center = points[keep].mean(0)
        _, _, vh = np.linalg.svd(points[keep] - center, full_matrices=False)
        direction = vh[0]
        normal = np.array([-direction[1], direction[0]])
        residual = np.abs((points[keep] - center) @ normal)
    return normal, -normal @ center, float(np.sqrt(np.mean(residual**2)))


def _snap_planes(point, primary, planes, eligible=None):
    """Use nearby independently observed orthogonal faces to refine a vertex."""
    picked = [primary]
    for i in sorted(
        range(len(planes)), key=lambda i: abs(planes[i][0] @ point + planes[i][1])
    ):
        normal, offset = planes[i]
        if (
            i in picked
            or (eligible is not None and i not in eligible)
            or abs(normal @ point + offset) > 0.004
        ):
            continue
        if all(abs(normal @ planes[j][0]) < 0.25 for j in picked):
            picked.append(i)
        if len(picked) == 3:
            break
    if len(picked) == 1:
        return point, 1
    normals = np.stack([planes[i][0] for i in picked])
    offsets = np.array([planes[i][1] for i in picked])
    refined = point - np.linalg.lstsq(normals, normals @ point + offsets, rcond=None)[0]
    return (
        (refined, len(picked))
        if np.linalg.norm(refined - point) < 0.0045
        else (point, 1)
    )


def _visible_at_corner(point, uv, support, xyz):
    """A neighbouring nearer face does not occlude an observed corner ray.

    Require current target depth within the raster footprint of this corner,
    rather than comparing all neighbourhood depths to one scalar centre depth.
    """
    u, v = np.round(uv).astype(int)
    if not 2 <= u < 510 or not 2 <= v < 510:
        return False
    yy, xx = np.indices((5, 5))
    footprint = np.hypot(xx + u - 2 - uv[0], yy + v - 2 - uv[1]) <= 1.5
    observed = support[v - 2 : v + 3, u - 2 : u + 3] & footprint
    return bool(
        np.any(
            observed
            & (
                np.linalg.norm(xyz[v - 2 : v + 3, u - 2 : u + 3] - point, axis=-1)
                < 0.004
            )
        )
    )


def _unoccluded_footprint(point, uv, depth, valid):
    """A supplementary junction cannot sit behind the front aperture/robot."""
    u, v = np.floor(uv).astype(int)
    if not 1 <= u < 510 or not 1 <= v < 510:
        return False
    foreground = valid[v : v + 2, u : u + 2] & (
        depth[v : v + 2, u : u + 2] < point[2] - 0.020
    )
    return int(foreground.sum()) < 2


def _physical_edges(
    uv,
    adjacent,
    normal,
    offset,
    plane_mask,
    object_mask,
    depth,
    valid,
    xyz,
    intrinsic,
    neutral=None,
):
    """Check both edge legs against current depth, not a nearby occluder.

    Physical silhouette edges have a farther background; a crease has another
    observed object face. A boundary cut by a nearer robot is an occlusion edge.
    """
    scores = []
    for end in adjacent:
        vector = np.asarray(end) - uv
        length = np.linalg.norm(vector)
        if length < 3:
            return False, []
        direction = vector / length
        side = np.array([-direction[1], direction[0]])
        along = np.linspace(1.5, min(8.0, length * 0.75), 6)
        center = uv + along[:, None] * direction
        offsets = []
        for sign in (-1, 1):
            pix = np.round(center + sign * 2.0 * side).astype(int)
            if np.any((pix < 1) | (pix > 510)):
                return False, []
            offsets.append((pix, plane_mask[pix[:, 1], pix[:, 0]].sum()))
        outside_index = 0 if offsets[0][1] < offsets[1][1] else 1
        outside = offsets[outside_index][0]
        inside = offsets[1 - outside_index][0]
        u, v = outside[:, 0], outside[:, 1]
        observed = xyz[v, u]
        ray = np.c_[
            (u - intrinsic[0, 2]) / intrinsic[0, 0],
            (v - intrinsic[1, 2]) / intrinsic[1, 1],
            np.ones(len(u)),
        ]
        denominator = ray @ normal
        predicted = -offset / np.where(abs(denominator) > 0.02, denominator, np.nan)
        delta = depth[v, u] - predicted
        local_gap = np.linalg.norm(observed - xyz[inside[:, 1], inside[:, 0]], axis=1)
        # A supported object/table contact edge is visible too. Neutral robot
        # surfaces are handled conservatively by the separate contact filter.
        non_robot = (
            object_mask[v, u] if neutral is None else object_mask[v, u] | ~neutral[v, u]
        )
        crease = (
            non_robot
            & (np.abs(observed @ normal + offset) > 0.0012)
            & (local_gap < 0.012)
        )
        # Another part of the same object can occlude this face. It is not an
        # adjacent physical edge merely because the actor/material is the same.
        # At grazing incidence a sub-mm depth quantisation step can become a
        # large ray-depth difference. Require actual distance from the plane.
        background = ~valid[v, u] | (
            (delta > 0.002) & (np.abs(observed @ normal + offset) > 0.0012)
        )
        physical = crease | background
        occluding = valid[v, u] & (delta < -0.002) & ~crease
        inside_ok = valid[inside[:, 1], inside[:, 0]] & (
            plane_mask[inside[:, 1], inside[:, 0]] > 0
        )
        physical &= inside_ok
        scores.append(
            dict(physical=int(physical.sum()), occluding=int(occluding.sum()))
        )
    return all(s["physical"] >= 3 and s["occluding"] <= 1 for s in scores), scores


def _local_boundary_corners(contour, params, is_hole=False):
    """Fit short edge legs locally when a global polygon loses the junction."""
    points = contour[:, 0].astype(float)
    if len(points) < 18:
        return []
    rough = cv2.approxPolyDP(points.astype(np.float32), 0.8, True)[:, 0]
    result = []
    for q in rough:
        index = int(np.linalg.norm(points - q, axis=1).argmin())
        choices = []
        for span in (7, 11):
            if len(points) < 2 * span + 2:
                continue
            previous = points[(index - np.arange(2, span + 1)) % len(points)]
            following = points[(index + np.arange(2, span + 1)) % len(points)]
            lines = []
            for part in (previous, following):
                n, d, res = _line(part)
                d += (-1 if is_hole else 1) * 0.5 * np.sign(n @ points.mean(0) + d)
                lines.append((n, d, res))
            matrix = np.stack([x[0] for x in lines])
            if abs(np.linalg.det(matrix)) < 0.15:
                continue
            uv = np.linalg.solve(matrix, -np.array([x[1] for x in lines]))
            residual = max(x[2] for x in lines)
            if residual <= 0.4 and np.linalg.norm(uv - q) <= 2.0:
                choices.append((uv, residual, (previous[-1], following[-1])))
        if len(choices) >= 2 and np.linalg.norm(choices[0][0] - choices[1][0]) <= 0.8:
            result.append(min(choices, key=lambda c: c[1]))
    return result


def _face_envelopes(plane_mask, primary, planes, eligible, intrinsic):
    """Rectangular corners of observed face/rim boundaries, with no known size.

    Other observed face normals provide directions only. Current contour extents
    determine all edge locations; each output still needs direct visibility and
    two physical edge legs, so a completed hidden rectangle is never emitted.
    """
    normal, offset = planes[primary]
    other = next(
        (
            i
            for i in range(len(planes))
            if i != primary and i in eligible and abs(normal @ planes[i][0]) < 0.2
        ),
        None,
    )
    if other is None:
        return []
    axis = planes[other][0] - normal * (normal @ planes[other][0])
    axis /= np.linalg.norm(axis)
    basis = np.stack((axis, np.cross(normal, axis)), axis=1)

    def lift(uv):
        rays = np.c_[
            (uv[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
            (uv[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1],
            np.ones(len(uv)),
        ]
        return rays * (-offset / (rays @ normal))[:, None]

    contours, hierarchy = cv2.findContours(
        plane_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    result = []
    for ci, contour in enumerate(contours):
        if abs(cv2.contourArea(contour)) < 30:
            continue
        uv = contour[:, 0].astype(float)
        if np.any((uv <= 1) | (uv >= 510)):
            continue
        cloud = lift(uv)
        coords = cloud @ basis
        center = np.median(uv, axis=0)
        probe = lift(np.array([center, center + [1, 0], center + [0, 1]])) @ basis
        half = 0.5 * (abs(probe[1] - probe[0]) + abs(probe[2] - probe[0]))
        hole = bool(hierarchy[0, ci, 3] >= 0)
        sign = -1 if hole else 1
        lo = coords.min(0) - sign * half
        hi = coords.max(0) + sign * half
        if np.any(hi - lo < 0.005):
            continue
        corners2 = np.array(
            [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
        )
        points = corners2 @ basis.T - offset * normal
        if np.any(points[:, 2] <= 0):
            continue
        image = (
            points[:, :2]
            / points[:, 2, None]
            * np.array([intrinsic[0, 0], intrinsic[1, 1]])
            + intrinsic[:2, 2]
        )
        for i in range(4):
            result.append(
                (image[i], points[i], (image[i - 1], image[(i + 1) % 4]), hole)
            )
    return result


def _polygon_corners(contour, params, is_hole=False):
    contour = contour[:, 0].astype(np.float64)
    if len(contour) < 16:
        return []
    approx = cv2.approxPolyDP(
        contour.astype(np.float32), params.polygon_epsilon_px, True
    )[:, 0]
    if not 3 <= len(approx) <= 30:
        return []
    indices = [int(np.linalg.norm(contour - p, axis=1).argmin()) for p in approx]
    lines = []
    for i, start in enumerate(indices):
        end = indices[(i + 1) % len(indices)]
        interval = (
            contour[start : end + 1]
            if end >= start
            else np.concatenate((contour[start:], contour[: end + 1]))
        )
        length = np.linalg.norm(approx[i] - approx[(i + 1) % len(approx)])
        if len(interval) < 5 or length < params.minimum_edge_px:
            lines.append(None)
            continue
        n, d, residual = _line(interval)
        # Contour coordinates are foreground pixel centres. The physical image
        # boundary lies between foreground and background pixel centres.
        side = np.sign(n @ contour.mean(0) + d)
        d += (-1 if is_hole else 1) * 0.5 * side
        lines.append((n, d, residual))
    result = []
    for i, rough in enumerate(approx):
        left, right = lines[i - 1], lines[i]
        if left is None or right is None:
            continue
        a = np.stack((left[0], right[0]))
        if abs(np.linalg.det(a)) < 0.18:
            continue
        uv = np.linalg.solve(a, -np.array([left[1], right[1]]))
        residual = max(left[2], right[2])
        if (
            residual <= params.maximum_line_residual_px
            and np.linalg.norm(uv - rough) <= 3
        ):
            result.append(
                (uv, residual, (approx[i - 1], approx[(i + 1) % len(approx)]))
            )
    return result
