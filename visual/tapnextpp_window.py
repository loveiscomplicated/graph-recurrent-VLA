"""Backend-independent observation windows and state row mappings."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveState:
    """Patch coordinates and recent observations for one tracking stream."""

    crop: tuple[int, int, int, int]
    model_state: tuple
    visible_xy: np.ndarray
    misses: int
    frame_shape: tuple[int, int]


def _union(first, second):
    if first is None:
        return second
    if second is None:
        return first
    return np.r_[np.minimum(first[:2], second[:2]), np.maximum(first[2:], second[2:])]


def _point_bounds(points):
    if points is None:
        return None
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1)]
    return None if not len(points) else np.r_[points.min(axis=0), points.max(axis=0) + 1]


def _mask_bounds(mask):
    if mask is None:
        return None
    rows = np.flatnonzero(np.any(mask, axis=1))
    if not len(rows):
        return None
    columns = np.flatnonzero(np.any(mask, axis=0))
    return np.array([columns[0], rows[0], columns[-1] + 1, rows[-1] + 1])


def _select_window(bounds, prior, misses, frame_shape, grid):
    """Quantize a padded box to a few window sizes and keep a stable crop when possible."""
    if bounds is None or misses >= 3:
        return (0, 0, grid, grid)
    height, width = frame_shape
    scale = grid / np.array([width, height])
    lower = np.clip(np.floor(bounds[:2] * scale).astype(int) - 2, 0, grid)
    upper = np.clip(np.ceil(bounds[2:] * scale).astype(int) + 2, 0, grid)
    minimum_side = max(int(np.max(upper - lower)), 8 + 8 * misses)
    side = next((size for size in (12, 16, 24, grid) if size >= minimum_side), grid)
    if prior is not None:
        previous_lower, previous_upper = np.array(prior[:2]), np.array(prior[2:])
        if (previous_upper[0] - previous_lower[0] == side
                and np.all(previous_lower <= lower) and np.all(previous_upper >= upper)):
            return prior
    origin = np.clip((lower + upper - side) // 2, 0, grid - side)
    return tuple(int(edge) for edge in np.r_[origin, origin + side])


def patch_ids(crop, grid):
    left, top, right, bottom = crop
    return (np.arange(top, bottom)[:, None] * grid + np.arange(left, right)).ravel()


def remap_rows(previous, selected, query_count, grid):
    """Match rows by absolute patch ID and keep every query row in order."""
    old_ids = patch_ids(previous, grid)
    old_rows = np.full(grid * grid, -1, dtype=np.int32)
    old_rows[old_ids] = np.arange(len(old_ids))
    image_rows = old_rows[selected]
    query_rows = np.arange(len(old_ids), len(old_ids) + query_count, dtype=np.int32)
    rows = np.concatenate((image_rows, query_rows))
    return rows


def select_window(query_points_xy, state, observation_mask, frame_shape, grid):
    """Select a window using only the current observations and prior state."""
    if observation_mask is not None and observation_mask.shape != frame_shape:
        raise ValueError("Expected an object mask with the source frame height and width")
    if state is not None and frame_shape != state.frame_shape:
        raise ValueError("Keep the same frame size within one adaptive tracking stream")

    bounds = _mask_bounds(observation_mask)
    prior_crop = None if state is None else state.crop
    if state is None:
        bounds = _union(bounds, _point_bounds(query_points_xy))
        misses = 0
    elif bounds is not None:
        misses = 0
    else:
        bounds = _point_bounds(state.visible_xy) if len(state.visible_xy) >= 4 else None
        misses = state.misses + 1 if observation_mask is not None or bounds is None else 0

    return _select_window(bounds, prior_crop, misses, frame_shape, grid), misses
