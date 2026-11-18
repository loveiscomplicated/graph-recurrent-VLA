"""Select depth-supported queries along detected object edges."""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from data_collection.box_shape_first import fps_rows


def prism_edge_pairs(camera_corners):
    """Recover the three dominant orthogonal directions from detected corner pairs."""
    a, b = np.triu_indices(len(camera_corners), k=1)
    delta = camera_corners[b] - camera_corners[a]
    lengths = np.linalg.norm(delta, axis=1)
    usable = lengths > 0.005
    pairs = np.column_stack((a[usable], b[usable]))
    directions = delta[usable] / lengths[usable, None]
    best = np.zeros(len(pairs), dtype=bool)
    for i, j in zip(*np.nonzero(np.triu(np.abs(directions @ directions.T) < 0.15, 1))):
        first, second = directions[i], directions[j]
        second = second - (first @ second) * first
        second /= np.linalg.norm(second)
        axes = np.stack((first, second, np.cross(first, second)))
        alignment = np.abs(directions @ axes.T)
        aligned = alignment.max(axis=1) > 0.985
        if aligned.sum() > best.sum():
            best = aligned
    return pairs[best]


def supported_edges(pixels, candidate_depth, corner_uv, corner_camera, radius, pairs):
    """Find projected edge segments and pixels supported by their expected depth."""
    depth_m = candidate_depth[pixels[:, 1], pixels[:, 0]] * 0.001
    edges = []
    for a, b in pairs:
        start, end = corner_uv[[a, b]]
        delta = end - start
        length = np.linalg.norm(delta)
        span = length - 2 * radius
        if span <= 0:
            continue
        direction = delta / length
        relative = pixels - start
        along = relative @ direction
        cross = np.linalg.norm(relative - along[:, None] * direction, axis=1)
        start_z, end_z = corner_camera[[a, b], 2]
        # Linear steps in image space interpolate inverse camera depth.
        fraction = np.clip(along / length, 0, 1)
        expected_depth = 1 / ((1 - fraction) / start_z + fraction / end_z)
        supported = (
            (along > radius) & (along < length - radius) & (cross <= 2)
            & (np.abs(depth_m - expected_depth) <= 0.008)
        )
        if supported.any():
            edges.append((start, direction, span, supported))
    return edges


def assign_edge_pixels(pixels, edges, slots, radius, *, tolerance=3, line_tolerance=2):
    """Assign spaced targets to distinct pixels and skip unsupported targets."""
    costs = []
    for (start, direction, span, supported), slots_on_edge in zip(edges, slots, strict=True):
        relative = pixels - start
        perpendicular = np.abs(relative[:, 0] * direction[1] - relative[:, 1] * direction[0])
        near_line = supported & (perpendicular <= line_tolerance)
        for index in range(slots_on_edge):
            target = start + direction * (radius + span * (index + 0.5) / slots_on_edge)
            distance = np.linalg.norm(pixels - target, axis=1)
            costs.append(np.where(near_line & (distance <= tolerance), distance, np.inf))
    if not costs:
        return np.empty((0, 2), dtype=pixels.dtype)
    costs = np.asarray(costs)
    # Dummy columns let targets across gaps remain unmatched.
    padded = np.column_stack((costs, np.full((len(costs), len(costs)), 100.0)))
    _, assignment = linear_sum_assignment(padded)
    return pixels[assignment[assignment < len(pixels)]]


def append_peg_edge_queries(mask, candidate_depth, corner_uv, corner_camera, surface_uv,
                            *, radius=2, spacing=2, line_tolerance=2):
    """Keep the original surface queries and append dense, unique edge pixels."""
    pairs = prism_edge_pairs(corner_camera)
    return append_edge_queries(
        mask, candidate_depth, corner_uv, corner_camera, surface_uv, pairs,
        radius=radius, spacing=spacing, line_tolerance=line_tolerance,
    )


def append_box_edge_queries(mask, candidate_depth, corner_uv, corner_camera, rim, surface_uv,
                            *, radius=2, spacing=3, line_tolerance=2):
    """Infer Box exterior and Hole rim edges separately using the Peg pairing rule."""
    outer_ids = np.flatnonzero(~rim)
    rim_ids = np.flatnonzero(rim)
    outer_pairs = outer_ids[prism_edge_pairs(corner_camera[outer_ids])]
    rim_pairs = rim_ids[prism_edge_pairs(corner_camera[rim_ids])]
    pairs = np.concatenate((outer_pairs, rim_pairs))
    return append_edge_queries(
        mask, candidate_depth, corner_uv, corner_camera, surface_uv, pairs,
        radius=radius, spacing=spacing, line_tolerance=line_tolerance,
    )


def append_edge_queries(mask, candidate_depth, corner_uv, corner_camera, surface_uv, pairs,
                        *, radius, spacing, line_tolerance):
    """Append distinct supported pixels at the requested target spacing."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Peg edge spacing must be a finite positive distance in pixels")
    valid = mask & np.isfinite(candidate_depth) & (candidate_depth > 0)
    x, y = surface_uv.astype(int).T
    valid[y, x] = False
    pixels = np.argwhere(valid)[:, ::-1]
    edges = supported_edges(pixels, candidate_depth, corner_uv, corner_camera, radius, pairs)
    slots = [max(1, int(span / spacing)) for _, _, span, _ in edges]
    added = assign_edge_pixels(
        pixels, edges, slots, radius, tolerance=2, line_tolerance=line_tolerance,
    )
    return np.concatenate((surface_uv, added.astype(np.float32)))


def choose_peg_edge_queries(mask, candidate_depth, corner_uv, corner_camera, count, *, radius=2):
    """Place evenly spaced edge targets and snap them to distinct supported pixels."""
    valid = mask & np.isfinite(candidate_depth) & (candidate_depth > 0)
    pixels = np.argwhere(valid)[:, ::-1]
    pairs = prism_edge_pairs(corner_camera)
    edges = supported_edges(pixels, candidate_depth, corner_uv, corner_camera, radius, pairs)
    if not edges or count < 4:
        raise ValueError("No depth-supported Peg edges available for the query budget")

    slots = np.zeros(len(edges), dtype=int)
    for _ in range(count // 4):
        # Give each visible edge a slot before adding more to longer edges.
        score = [span / (slots[i] + 1) if slots[i] else np.inf
                 for i, (_, _, span, _) in enumerate(edges)]
        slots[np.argmax(score)] += 1
    priority = assign_edge_pixels(pixels, edges, slots, radius)
    if not len(priority):
        raise ValueError("No Peg edge target matched a valid-depth pixel")
    valid[priority[:, 1], priority[:, 0]] = False
    core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    surface = np.argwhere(valid & (core > 0))[:, ::-1]
    candidates = np.concatenate((priority, surface)).astype(np.float32)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible Peg pixels for {count} queries")
    rows = fps_rows(candidates, count, seed=0, initial=range(len(priority)))
    return candidates[rows], len(priority)
