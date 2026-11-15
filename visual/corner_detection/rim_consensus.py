"""Fit a single stationary square to sequential RGB-D opening hypotheses."""

import numpy as np
from scipy.optimize import linear_sum_assignment


class RimConsensus:
    """Align four-corner observations and jointly fit their square geometry."""

    def __init__(self):
        self.observations = []

    def update(self, squares):
        """Accept the current frame's verified squares and return the running fit."""
        for square in squares:
            square = np.asarray(square)
            if not self.observations:
                self.observations.append(square)
                continue
            anchor = self.observations[0]
            distances = np.linalg.norm(anchor[:, None] - square[None], axis=-1)
            _, indices = linear_sum_assignment(distances)
            if np.max(distances[np.arange(4), indices]) <= 0.015:
                self.observations.append(square[indices])
        if not self.observations:
            return []
        median = np.median(self.observations, axis=0)
        center = median.mean(axis=0)
        signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        axes = signs.T @ (median - center)
        left, singular, right = np.linalg.svd(axes, full_matrices=False)
        return center + signs @ (left @ right) * singular.mean() / 4
