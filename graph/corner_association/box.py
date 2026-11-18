"""Causal Box exterior corner IDs from detector candidates and workspace axes."""

from dataclasses import dataclass
from itertools import product

import numpy as np

# Mirrors corners() in visual/corner_detection/evaluation.py. The matcher never reads its GT.
SIGNS = np.asarray(list(product((-1, 1), repeat=3)))
LOCAL_XY = (SIGNS[:, 0] + 1j * SIGNS[:, 1]) / 2
TOP = SIGNS[:, 2] > 0
SIDE_MIN, SIDE_MAX = 0.17, 0.25
YAW_CENTER = np.pi / 2
YAW_MARGIN = np.pi / 8


def canonical_yaw(angle):
    """Use the task's world-yaw convention to distinguish the four cube symmetries."""
    return YAW_CENTER + (angle - YAW_CENTER + np.pi / 4) % (np.pi / 2) - np.pi / 4


def complex_xy(xyz):
    return xyz[..., 0] + 1j * xyz[..., 1]


@dataclass(frozen=True)
class BoxPose:
    center_xy: complex
    side_m: float
    yaw: float

    def corners(self):
        points = self.center_xy + LOCAL_XY * self.side_m * np.exp(1j * self.yaw)
        return np.stack((points.real, points.imag, TOP * self.side_m), axis=-1)


def candidate_indices(expected, points, threshold):
    """Nearest candidate per vertex, with disjoint gates for the 170 mm cube."""
    if not len(points):
        return np.full(8, -1, dtype=int)
    assert 2 * threshold < SIDE_MIN
    dist = np.linalg.norm(expected[:, None] - points[None], axis=-1)
    candidates = dist.argmin(axis=1)
    return np.where(dist[np.arange(8), candidates] <= threshold, candidates, -1)


def fit_initial(points, min_support=3):
    level = (abs(points[:, 2]) <= 0.02) | (
        (points[:, 2] >= SIDE_MIN - 0.02) & (points[:, 2] <= SIDE_MAX + 0.02)
    )
    vertices = points[level]
    if len(vertices) < min_support:
        return None
    xy = complex_xy(vertices)
    left, right = np.triu_indices(len(vertices), k=1)
    edges = xy[right] - xy[left]
    runs = []
    for ratio, turn in ((1.0, 0.0), (np.sqrt(2.0), np.pi / 4)):
        side = abs(edges) / ratio
        yaw = canonical_yaw(np.angle(edges) - turn)
        keep = (
            (SIDE_MIN - 0.012 <= side)
            & (side <= SIDE_MAX + 0.012)
            & (abs(yaw - YAW_CENTER) <= YAW_MARGIN + 0.08)
        )
        runs.extend(zip(side[keep], yaw[keep]))
    if not runs:
        return None
    scales, angles = np.asarray(runs).T
    horizontal = scales[:, None] * np.exp(1j * angles[:, None]) * LOCAL_XY
    # All four horizontal sign combinations occur at both z levels.
    centers = xy[None, :, None] - horizontal[:, None, ::2]
    expected = (centers[..., None] + horizontal[:, None, None, :]).reshape(-1, 8)
    sizes = np.repeat(scales, len(vertices) * 4)
    hypotheses = np.stack(
        (expected.real, expected.imag, TOP[None, :] * sizes[:, None]), axis=-1
    )
    nearest = np.linalg.norm(
        hypotheses[:, :, None] - vertices[None, None], axis=-1
    ).min(axis=-1)
    gate = 0.017
    matches = (nearest <= gate).sum(axis=1)
    costs = np.minimum(nearest, gate).sum(axis=1)
    winner = np.lexsort((costs, -matches))[0]
    if matches[winner] < min_support:
        return None
    phase = winner // (len(vertices) * 4)
    pose = BoxPose(centers.ravel()[winner], sizes[winner], angles[phase])
    selected = candidate_indices(pose.corners(), vertices, gate)
    has = selected >= 0
    local, found = LOCAL_XY[has], vertices[selected[has]]
    local_shifted = local - local.mean()
    energy = np.vdot(local_shifted, local_shifted).real
    if energy < 0.25:
        return None
    scale = (
        np.vdot(local_shifted, complex_xy(found) - complex_xy(found).mean()) / energy
    )
    yaw = np.angle(scale)
    size = np.median(np.r_[abs(scale), found[TOP[has], 2]])
    if not (
        SIDE_MIN - 0.012 <= size <= SIDE_MAX + 0.012
        and abs(yaw - YAW_CENTER) <= YAW_MARGIN + 0.08
    ):
        return None
    center = np.median((complex_xy(found) - local * size * np.exp(1j * yaw)).real)
    center += 1j * np.median((complex_xy(found) - local * size * np.exp(1j * yaw)).imag)
    return BoxPose(center, float(size), float(yaw))


class BoxCornerMatcher:
    """Initialize a cubical Box once and track translation using exterior votes."""

    def __init__(self, assignment_m=0.010):
        self.assignment_m = assignment_m
        self.pose = None
        self.velocity = 0j
        self.age = 0

    def step(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite Box candidate coordinates [N, 3]")
        if self.pose is None:
            self.pose = fit_initial(points)
        if self.pose is None:
            return np.full(8, -1, dtype=int)
        previous = self.pose
        guess = BoxPose(
            previous.center_xy + self.velocity, previous.side_m, previous.yaw
        )
        association = candidate_indices(guess.corners(), points, 0.028)
        keep = association >= 0
        if keep.sum() >= 3:
            residual = complex_xy(points[association[keep]]) - complex_xy(
                guess.corners()[keep]
            )
            consensus = abs(residual[:, None] - residual[None]) <= 0.009
            support = consensus[consensus.sum(axis=1).argmax()]
            if support.sum() >= 3:
                agreed = residual[support]
                correction = np.median(agreed.real) + 1j * np.median(agreed.imag)
                new_center = guess.center_xy + correction
                displacement = new_center - previous.center_xy
                if abs(displacement) <= 0.023:
                    self.pose = BoxPose(new_center, previous.side_m, previous.yaw)
                    self.velocity = 0.45 * self.velocity + 0.55 * displacement
                    self.age = 0
                    return candidate_indices(
                        self.pose.corners(), points, self.assignment_m
                    )
        self.age += 1
        if self.age <= 3:
            self.pose = guess
        if self.age >= 5:
            recovered = fit_initial(points)
            if (
                recovered is not None
                and abs(recovered.side_m - previous.side_m) < 0.016
            ):
                self.pose, self.velocity, self.age = recovered, 0j, 0
        return candidate_indices(self.pose.corners(), points, self.assignment_m)
