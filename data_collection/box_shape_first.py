"""Object query placement using FD2 and surface sampling."""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def crease_pixels(mask, depth_m, xyz):
    """Flag endpoints of adjacent camera-facing FD2 normals over 20 degrees."""
    support = mask & np.isfinite(depth_m) & (depth_m > 0)
    height, width = support.shape
    v, u = np.nonzero(support)
    interior = (v >= 2) & (v < height - 2) & (u >= 2) & (u < width - 2)
    v, u = v[interior], u[interior]
    valid = np.ones(len(v), dtype=bool)
    for dv, du in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for step in (1, 2):
            previous = v + dv * (step - 1), u + du * (step - 1)
            current = v + dv * step, u + du * step
            near = np.minimum(depth_m[previous], depth_m[current])
            delta = np.abs(depth_m[previous] - depth_m[current])
            valid &= support[current] & (delta <= 0.01 + 0.02 * near)
    v, u = v[valid], u[valid]
    horizontal = xyz[v, u + 2] - xyz[v, u - 2]
    vertical = xyz[v + 2, u] - xyz[v - 2, u]
    vectors = np.cross(horizontal, vertical)
    lengths = np.linalg.norm(vectors, axis=1)
    usable = np.isfinite(lengths) & (lengths > 0)
    v, u = v[usable], u[usable]
    vectors = vectors[usable] / lengths[usable, None]
    vectors[np.einsum("ij,ij->i", vectors, xyz[v, u]) > 0] *= -1
    normals = np.full_like(xyz, np.nan)
    normals[v, u] = vectors

    crease = np.zeros_like(support)
    horizontal = np.einsum("ijk,ijk->ij", normals[:, :-1], normals[:, 1:])
    vertical = np.einsum("ijk,ijk->ij", normals[:-1], normals[1:])
    cosine_threshold = np.cos(np.deg2rad(20))
    crease[:, :-1] |= horizontal < cosine_threshold
    crease[:, 1:] |= horizontal < cosine_threshold
    crease[:-1] |= vertical < cosine_threshold
    crease[1:] |= vertical < cosine_threshold
    return crease


def fps_rows(xyz, count, seed, initial=()):
    """Select rows by 3D distance, keeping any preselected rows first."""
    selected = list(initial)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if not selected:
        selected.append(int(np.random.default_rng(seed).integers(len(xyz))))
    distance = np.full(len(xyz), np.inf)
    for index in range(count):
        if index == len(selected):
            selected.append(int(distance.argmax()))
        row = selected[index]
        squared = np.sum((xyz - xyz[row]) ** 2, axis=1)
        np.minimum(distance, squared, out=distance)
        distance[row] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def choose_box_rim_queries(mask, candidate_depth, rim_uv, count, *, radius=2):
    """Place equal numbers along the four detected rim sides, then fill by UV FPS."""
    if count < 16:
        raise ValueError("Box rim queries require at least 16 total queries")
    if len(rim_uv) != 4:
        raise ValueError("Box rim queries require four detected rim corners")
    hull = cv2.convexHull(np.asarray(rim_uv, dtype=np.float32)).reshape(-1, 2)
    if len(hull) != 4:
        raise ValueError("Box rim queries require four detected rim corners forming a convex quadrilateral")
    per_side = count // 16
    targets = []
    for start, end in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        direction = end - start
        length = np.linalg.norm(direction)
        if length <= 2 * radius:
            raise ValueError("Detected rim sides must be longer than twice the corner exclusion radius")
        fractions = (np.arange(per_side) + 0.5) / per_side
        distances = radius + fractions * (length - 2 * radius)
        targets.extend(start + distances[:, None] * (direction / length))

    valid = mask & np.isfinite(candidate_depth) & (candidate_depth > 0)
    pixels = np.argwhere(valid)[:, ::-1]
    if len(pixels) < len(targets):
        raise ValueError("Too few valid Box pixels for the rim queries")
    distances = np.linalg.norm(np.asarray(targets).reshape(-1, 2)[:, None] - pixels[None], axis=2)
    target_ids, pixel_ids = linear_sum_assignment(distances)
    if np.any(distances[target_ids, pixel_ids] > 3):
        raise ValueError("Cannot place distinct valid Box pixels within 3 px of each rim target")
    rim = pixels[pixel_ids]

    core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    valid[rim[:, 1], rim[:, 0]] = False
    surface = np.argwhere(valid & (core > 0))[:, ::-1]
    candidates = np.concatenate((rim, surface)).astype(np.float32)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible rim and surface pixels for {count} queries")
    rows = fps_rows(candidates, count, seed=0, initial=range(len(rim)))
    return candidates[rows], len(rim)


def camera_point_grid(depth, intrinsic):
    """Backproject millimetre depth to a dense camera-space point grid in metres."""
    depth_m = depth.astype(np.float64) * 0.001
    yy, xx = np.indices(depth.shape)
    rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(intrinsic).T
    return depth_m, rays * depth_m[..., None]


def append_fd2_edge_queries(mask, depth, candidate_depth, intrinsic, surface_uv):
    """Preserve surface queries and append distinct FD2 candidates at least 3 px apart."""
    depth_m, xyz = camera_point_grid(depth, intrinsic)
    eligible = crease_pixels(mask, depth_m, xyz)
    eligible &= np.isfinite(candidate_depth) & (candidate_depth > 0)
    x, y = surface_uv.astype(int).T
    eligible[y, x] = False

    added = []
    for v, u in np.argwhere(eligible):
        if eligible[v, u]:
            added.append((u, v))
            # On an integer grid, all offsets with distance < 3 px fit this square.
            eligible[max(0, v - 2):v + 3, max(0, u - 2):u + 3] = False
    return np.concatenate((surface_uv, np.asarray(added, dtype=np.float32).reshape(-1, 2)))


def choose_shape_first(mask, depth, candidate_depth, intrinsic, count, *, object_id):
    """Reserve up to a quarter of queries for FD2 creases, then fill by 3D FPS."""
    core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    eligible = (core > 0) & np.isfinite(candidate_depth) & (candidate_depth > 0)
    v, u = np.nonzero(eligible)
    if len(v) < count:
        raise ValueError(f"Only {len(v)} interior object pixels for {count} queries")

    depth_m, xyz = camera_point_grid(depth, intrinsic)
    points = xyz[v, u]
    creases = crease_pixels(mask, depth_m, xyz)
    rows = np.flatnonzero(creases[v, u])
    seed = int(np.random.SeedSequence([0, object_id]).generate_state(1)[0])
    priority_count = min(count // 4, len(rows))
    priority = rows[fps_rows(points[rows], priority_count, seed)]
    selected = fps_rows(points, count, seed, initial=priority)
    return np.column_stack((u[selected], v[selected])).astype(np.float32), priority_count


def choose_box_shape_first(mask, depth, candidate_depth, intrinsic, count):
    """Select FD2 Shape-First queries using the Box sampling seed."""
    return choose_shape_first(mask, depth, candidate_depth, intrinsic, count, object_id=1)
