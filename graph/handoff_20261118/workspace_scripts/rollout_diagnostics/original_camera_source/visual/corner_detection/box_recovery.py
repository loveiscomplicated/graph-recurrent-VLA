"""Current RGB corner proposals validated by observed Box planes and depth."""

from itertools import product
import cv2
import numpy as np
from . import geometry
from .candidates import _corner_features


def _project(points, k):
    return points[:, :2] / points[:, 2, None] * np.array([k[0, 0], k[1, 1]]) + k[:2, 2]


def _observed_box_corners(
    mask, xyz, planes, eligible, k, depth, valid, neutral, features=(), debug=False
):
    """Cube exterior prior; size comes only from two observed face spans.

    Proposed corners need current corner-depth and physical edge or RGB-corner
    evidence. A clipped/insufficient face does not establish a box size.
    """
    y, x = np.nonzero(mask)
    if not len(x) or x.min() < 2 or y.min() < 2 or x.max() > 509 or y.max() > 509:
        return []
    n, d = planes[0]
    other = next((i for i in eligible if i != 0 and abs(n @ planes[i][0]) < 0.15), None)
    if other is None:
        return []
    axis = planes[other][0] - n * (n @ planes[other][0])
    axis /= np.linalg.norm(axis)
    rotation = np.stack((n, axis, np.cross(n, axis)), axis=1)
    coords = xyz[mask] @ rotation
    lo, hi = np.percentile(coords, [0.1, 99.9], axis=0)
    widths = hi - lo
    # No hidden per-episode dimensions are supplied. Bounds are a class prior
    # from the installed task, not an oracle instance parameter.
    size = float(max(widths[1:])) + 0.0015
    if not 0.16 <= size <= 0.27 or min(widths[1:]) < 0.8 * size:
        return []
    center = (lo + hi) / 2
    center[0] = -d + (size / 2 if hi[0] + d > -d - lo[0] else -size / 2)
    result = []
    for sign in product((-1, 1), repeat=3):
        coordinate = center + np.array(sign) * size / 2
        point = coordinate @ rotation.T
        nearest = min(eligible, key=lambda i: abs(planes[i][0] @ point + planes[i][1]))
        if abs(planes[nearest][0] @ point + planes[nearest][1]) < 0.003:
            point, _ = geometry._snap_planes(point, nearest, planes, set(eligible))
        uv = _project(point[None], k)[0]
        if features:
            feature, kind = min(features, key=lambda f: np.linalg.norm(f[0] - uv))
            if np.linalg.norm(feature - uv) < 2.5:
                pi = min(
                    eligible, key=lambda i: abs(planes[i][0] @ point + planes[i][1])
                )
                normal, offset = planes[pi]
                ray = np.r_[(feature - k[:2, 2]) / np.array([k[0, 0], k[1, 1]]), 1.0]
                if abs(normal @ ray) > 0.1:
                    refined = ray * (-offset / (normal @ ray))
                    if np.linalg.norm(refined - point) < 0.005:
                        point = refined
                        uv = feature.copy()
        if debug:
            result.append(
                (
                    uv,
                    point,
                    dict(
                        visible=geometry._visible_at_corner(point, uv, mask, xyz),
                        unoccluded=geometry._unoccluded_footprint(
                            point, uv, depth, valid
                        ),
                    ),
                    size,
                )
            )
            continue
        image_corner = any(
            np.linalg.norm(uv - feature[0]) < 2.5 for feature in features
        )
        observed = geometry._visible_at_corner(point, uv, mask, xyz)
        if not observed and image_corner:
            u, v = np.round(uv).astype(int)
            if 3 <= u < 509 and 3 <= v < 509:
                observed = bool(
                    np.any(
                        mask[v - 3 : v + 4, u - 3 : u + 4]
                        & (
                            np.linalg.norm(
                                xyz[v - 3 : v + 4, u - 3 : u + 4] - point, axis=-1
                            )
                            < 0.006
                        )
                    )
                )
        if not observed:
            continue
        if not geometry._unoccluded_footprint(point, uv, depth, valid):
            continue
        # At least one actual observed incident face must support both edge legs.
        evidence = None
        for axis_index in range(3):
            matching = next(
                (
                    i
                    for i in eligible
                    if abs(planes[i][0] @ rotation[:, axis_index]) > 0.97
                    and abs(planes[i][0] @ point + planes[i][1]) < 0.0025
                ),
                None,
            )
            if matching is None:
                continue
            tangents = [i for i in range(3) if i != axis_index]
            adjacent = []
            for tangent in tangents:
                adjacent.append(point - sign[tangent] * 0.015 * rotation[:, tangent])
            observed = (
                np.abs(xyz @ planes[matching][0] + planes[matching][1]) < 0.0012
            ) & mask
            physical, checks = geometry._physical_edges(
                uv,
                _project(np.array(adjacent), k),
                *planes[matching],
                observed.astype(np.uint8),
                mask,
                depth,
                valid,
                xyz,
                k,
                neutral,
            )
            if physical:
                evidence = checks
                break
        if evidence is None and not image_corner:
            continue
        if evidence is None:
            evidence = []
        result.append((uv, point, evidence, size))
    return result


def recover(rgb, raw_depth, k, ext, existing, masks=None):
    """Cube class prior; no instance GT/dimensions, memory or other camera."""
    depth = np.asarray(raw_depth).squeeze().astype(float) * 0.001
    valid = (depth > 0) & (depth < 4)
    masks = geometry.color_masks(rgb, 8) if masks is None else np.asarray(masks, bool)
    mask = masks[1] & valid
    if mask.sum() < 80:
        return [], dict(features=0, planes=0)
    xyz = geometry.unproject(depth, k)
    vv, uu = np.nonzero(mask)
    points = xyz[vv, uu]
    planes = geometry._planes(
        points, geometry.Parameters(), np.random.default_rng(43), 8
    )
    if not planes:
        return [], dict(features=0, planes=0)
    distances = np.stack([np.abs(points @ n + d) for n, d in planes], -1)
    labels = distances.argmin(1)
    plane_masks = []
    eligible = []
    for i in range(len(planes)):
        pm = np.zeros(mask.shape, np.uint8)
        chosen = (labels == i) & (distances[:, i] < 0.001)
        pm[vv[chosen], uu[chosen]] = 1
        pm = cv2.morphologyEx(pm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        plane_masks.append(pm)
        if cv2.erode(pm, np.ones((3, 3), np.uint8)).sum() >= 20:
            eligible.append(i)
    features = _corner_features(rgb, mask)
    neutral = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1] < 8
    proposals = []
    for uv, point, evidence, size in _observed_box_corners(
        mask, xyz, planes, eligible, k, depth, valid, neutral, features
    ):
        proposals.append(
            dict(
                object_id=1,
                uv=uv.tolist(),
                xyz=((point - ext[:, 3]) @ ext[:, :3]).tolist(),
                camera_xyz=point.tolist(),
                plane=0,
                line_residual_px=0.25,
                contour_hole=False,
                sources=1,
                supporting_planes=2,
                method="box_observed_cube",
                estimated_size_m=size,
                edge_evidence=evidence,
            )
        )
    for uv, feature in features:
        hypotheses = []
        ray = np.r_[(uv - k[:2, 2]) / np.array([k[0, 0], k[1, 1]]), 1.0]
        for pi in eligible:
            n, d = planes[pi]
            if abs(n @ ray) < 0.1:
                continue
            point = ray * (-d / (n @ ray))
            if point[2] <= 0 or not geometry._visible_at_corner(point, uv, mask, xyz):
                continue
            if not geometry._unoccluded_footprint(point, uv, depth, valid):
                continue
            other = next(
                (j for j in eligible if j != pi and abs(n @ planes[j][0]) < 0.2), None
            )
            if other is None:
                continue
            axis = planes[other][0] - n * (n @ planes[other][0])
            axis /= np.linalg.norm(axis)
            axes = np.stack((axis, np.cross(n, axis)))
            fractions = []
            signs = list(product((-1, 1), repeat=2))
            for sx, sy in signs:
                probe = np.array(
                    [
                        point + sx * a * axes[0] + sy * b * axes[1]
                        for a, b in product((0.002, 0.004), repeat=2)
                    ]
                )
                pix = np.round(_project(probe, k)).astype(int)
                if np.any((pix < 1) | (pix > 510)):
                    fractions.append(0.5)
                    continue
                x, y = pix[:, 0], pix[:, 1]
                fractions.append(
                    float(np.mean(mask[y, x] & (np.abs(xyz[y, x] @ n + d) < 0.0012)))
                )
            occupied = np.array(fractions) >= 0.75
            empty = np.array(fractions) <= 0.25
            if occupied.sum() == 1 and empty.sum() == 3:
                sx, sy = signs[int(occupied.argmax())]
            elif occupied.sum() == 3 and empty.sum() == 1:
                sx, sy = signs[int(empty.argmax())]
            else:
                continue
            adjacent = _project(
                np.array([point + sx * 0.015 * axes[0], point + sy * 0.015 * axes[1]]),
                k,
            )
            physical, evidence = geometry._physical_edges(
                uv, adjacent, n, d, plane_masks[pi], mask, depth, valid, xyz, k, neutral
            )
            if not physical:
                continue
            u, v = np.round(uv).astype(int)
            contact = (
                valid[v - 2 : v + 3, u - 2 : u + 3]
                & neutral[v - 2 : v + 3, u - 2 : u + 3]
                & ~mask[v - 2 : v + 3, u - 2 : u + 3]
            )
            contact &= (
                np.linalg.norm(xyz[v - 2 : v + 3, u - 2 : u + 3] - point, axis=-1)
                < 0.006
            )
            if contact.sum() >= 3:
                continue
            world = (point - ext[:, 3]) @ ext[:, :3]
            hypotheses.append(
                dict(
                    object_id=1,
                    uv=uv.tolist(),
                    xyz=world.tolist(),
                    camera_xyz=point.tolist(),
                    plane=pi,
                    line_residual_px=0.3,
                    contour_hole=bool(empty.sum() == 1),
                    sources=1,
                    supporting_planes=1,
                    method="box_rgb_quadrants",
                    feature=feature,
                    quadrant_fractions=fractions,
                    edge_evidence=evidence,
                )
            )
        if hypotheses:
            # Reject hidden-wall interpretations if a supported front face exists.
            proposals.append(min(hypotheses, key=lambda c: c["camera_xyz"][2]))
    result = []
    for p in proposals:
        if any(
            c["object_id"] == 1
            and np.linalg.norm(np.array(c["xyz"]) - p["xyz"]) < 0.010
            for c in existing + result
        ):
            continue
        result.append(p)
    return result, dict(
        features=len(features), planes=len(planes), validated_hypotheses=len(proposals)
    )
