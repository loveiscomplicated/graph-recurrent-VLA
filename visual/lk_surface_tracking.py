"""Online pyramidal Lucas–Kanade tracking for visible Peg surface points."""

import cv2
import numpy as np

from visual.peg_surface_points import interior_pixels, spread_points, supported_pixels


class LKSurfaceTracker:
    """Track up to a fixed number of surface points and reseed lost slots."""

    def __init__(self, count: int):
        self.count = count
        self.reset()

    def reset(self) -> None:
        """Clear all inter-frame state at the start of an episode."""
        self.previous_gray = None
        self.previous_xy = np.empty((0, 2), dtype=np.float32)
        self.previous_slots = np.empty(0, dtype=np.intp)

    def _track(self, gray, depth, mask):
        """Keep only supported tracks that agree in both temporal directions."""
        if not len(self.previous_slots) or self.previous_gray.shape != gray.shape:
            return self.previous_slots[:0], self.previous_xy[:0]

        options = {"winSize": (15, 15), "maxLevel": 3}
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, self.previous_xy, None, **options
        )
        if forward is None or status is None:
            return self.previous_slots[:0], self.previous_xy[:0]

        forward = forward.reshape(-1, 2)
        candidates, _, _ = supported_pixels(forward, depth, mask, status.ravel() != 0)
        if not len(candidates):
            return self.previous_slots[:0], self.previous_xy[:0]

        current_xy = forward[candidates]
        backward, status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, current_xy, None, **options
        )
        if backward is None or status is None:
            return self.previous_slots[:0], self.previous_xy[:0]

        error = np.linalg.norm(backward.reshape(-1, 2) - self.previous_xy[candidates], axis=1)
        keep = (status.ravel() != 0) & (error <= 1.0)
        return self.previous_slots[candidates[keep]], current_xy[keep]

    def step(self, rgb, depth, peg_mask):
        """Return source-image XY and a validity mask for the current frame."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        depth = np.where(np.isfinite(depth), depth, 0)
        tracked_slots, tracked_xy = self._track(gray, depth, peg_mask)
        xy = np.zeros((self.count, 2), dtype=np.float32)
        valid = np.zeros(self.count, dtype=bool)
        xy[tracked_slots] = tracked_xy
        valid[tracked_slots] = True

        free_slots = np.flatnonzero(~valid)
        if len(free_slots):
            candidates = interior_pixels(peg_mask, depth)
            new_xy = spread_points(
                candidates, len(free_slots), occupied=tracked_xy, min_distance=3.0
            )
            seeded_slots = free_slots[:len(new_xy)]
            xy[seeded_slots] = new_xy
            valid[seeded_slots] = True

        self.previous_gray = gray
        self.previous_slots = np.flatnonzero(valid)
        self.previous_xy = xy[self.previous_slots]
        return xy, valid
