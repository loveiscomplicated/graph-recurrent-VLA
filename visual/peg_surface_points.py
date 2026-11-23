"""Select image-space Peg surface points and lift them using measured depth."""

import cv2
import numpy as np


def interior_pixels(mask, depth):
    """Return XY pixels with positive depth inside the eroded Peg mask."""
    core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    return np.argwhere((core > 0) & (depth > 0))[:, ::-1].astype(np.float32)


def spread_points(candidates, count, *, occupied=(), min_distance=0.0):
    """Select spatially spread pixels, keeping clear of retained tracks."""
    if not len(candidates):
        return candidates

    index = np.linalg.norm(candidates - candidates.mean(0), axis=1).argmin()
    distance = np.full(len(candidates), np.inf)
    for point in occupied:
        distance = np.minimum(distance, np.sum((candidates - point) ** 2, axis=1))
    if len(occupied):
        index = distance.argmax()

    selected = []
    for _ in range(count):
        if distance[index] < min_distance ** 2:
            break
        point = candidates[index]
        selected.append(point)
        distance = np.minimum(distance, np.sum((candidates - point) ** 2, axis=1))
        index = distance.argmax()
    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


def choose_points(mask, depth, count):
    """Initialize a full set of spatially spread surface queries."""
    candidates = interior_pixels(mask, depth)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} interior surface pixels for {count} queries")
    return spread_points(candidates, count)


def supported_pixels(xy, depth, mask, visible):
    """Return valid track indices and their rounded pixel coordinates."""
    finite = np.isfinite(xy).all(axis=1)
    uv = np.rint(np.where(finite[:, None], xy, 0)).astype(int)
    u, v = uv.T
    h, w = depth.shape
    usable = finite & visible & (u >= 0) & (v >= 0) & (u < w) & (v < h)
    ids = np.flatnonzero(usable)
    ids = ids[(depth[v[ids], u[ids]] > 0) & mask[v[ids], u[ids]]]
    return ids, u[ids], v[ids]


def lift(xy, depth, mask, intrinsic, visible):
    """Recover camera XYZ for track coordinates supported by current RGB-D."""
    ids, u, v = supported_pixels(xy, depth, mask, visible)
    rays = np.column_stack((xy[ids], np.ones(len(ids)))) @ np.linalg.inv(intrinsic).T
    return ids, rays * (depth[v, u] * .001)[:, None]
