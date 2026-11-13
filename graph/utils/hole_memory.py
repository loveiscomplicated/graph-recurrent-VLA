"""Causal world-position memory for a stationary hole, without GT inputs."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MemorySettings:
    """Observation gates are heuristics, not calibrated prediction confidence."""

    min_target_nodes: int = 5
    min_mean_confidence: float = 0.82
    min_confidence_fraction: float = 1.0
    consecutive_frames: int = 3
    max_spread_m: float = 0.01

    def __post_init__(self):
        for name in ("min_target_nodes", "consecutive_frames"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"memory.{name} must be an integer >= 1")
        for name in ("min_mean_confidence", "min_confidence_fraction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"memory.{name} must be finite in [0,1]")
        if (type(self.max_spread_m) not in (int, float)
                or not math.isfinite(self.max_spread_m) or self.max_spread_m <= 0):
            raise ValueError("memory.max_spread_m must be finite and positive")


@dataclass(frozen=True)
class ObservationQuality:
    """Counts/quality of observed BOX or HOLE nodes; no rim identity is assumed."""

    target_nodes: int
    mean_confidence: float
    confidence_fraction: float

    def __post_init__(self):
        if type(self.target_nodes) is not int or self.target_nodes < 0:
            raise ValueError("quality.target_nodes must be an integer >= 0")
        for name in ("mean_confidence", "confidence_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"quality.{name} must be finite in [0,1]")

    def accepts(self, settings: MemorySettings) -> bool:
        return (self.target_nodes >= settings.min_target_nodes
                and self.mean_confidence >= settings.min_mean_confidence
                and self.confidence_fraction >= settings.min_confidence_fraction)


@dataclass(frozen=True)
class MemoryEstimate:
    hole_relative: np.ndarray
    initialized: bool
    updated: bool
    held: bool
    quality_accepted: bool
    age_frames: int


class HolePositionMemory:
    """One episode's position memory. Construct a new instance at every reset.

    Store relative_prediction + current_tcp in world coordinates. Even when TCP
    moves, the returned relative position is stored_world - current_tcp. A valid
    update replaces memory with the coordinate-wise median of the causal window.
    An uninitialized memory returns the current prediction, never a fabricated 0.
    """

    def __init__(self, settings: MemorySettings):
        self.settings = settings
        self.candidates = deque(maxlen=settings.consecutive_frames)
        self.position_world = None
        self.last_update_frame = None
        self.previous_frame = None

    def step(self, frame: int, hole_relative, tcp_world,
             quality: ObservationQuality) -> MemoryEstimate:
        if type(frame) is not int or frame < 0:
            raise ValueError("frame must be an integer >= 0")
        if self.previous_frame is not None and frame <= self.previous_frame:
            raise ValueError("Memory needs strictly increasing frame indices within one episode")
        vectors = []
        for name, value in (("hole_relative", hole_relative), ("tcp_world", tcp_world)):
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f"{name} must contain three finite metre coordinates")
            vectors.append(vector)
        hole_relative, tcp_world = vectors
        accepted = quality.accepts(self.settings)
        consecutive = self.previous_frame is None or frame == self.previous_frame + 1
        if not accepted or not consecutive:
            self.candidates.clear()
        updated = False
        if accepted:
            self.candidates.append(hole_relative + tcp_world)
            if len(self.candidates) == self.settings.consecutive_frames:
                positions = np.stack(self.candidates)
                center = np.median(positions, axis=0)
                spread = np.linalg.norm(positions - center, axis=1).max()
                if spread <= self.settings.max_spread_m:
                    self.position_world = center.copy()
                    self.last_update_frame = frame
                    updated = True
        self.previous_frame = frame
        initialized = self.position_world is not None
        relative = self.position_world - tcp_world if initialized else hole_relative
        age = frame - self.last_update_frame if initialized else -1
        return MemoryEstimate(relative.copy(), initialized, updated,
                              initialized and not updated, accepted, age)


def replace_hole_position(object21, hole_relative):
    """Preserve the estimated peg tip when changing the Object21 hole position.

    Object21[7:10] is peg_tip - hole, in world axes. Rotations and sizes retain
    the current model outputs; this baseline remembers only the hole position.
    """
    output = np.asarray(object21).copy()
    if output.shape != (21,) or not np.isfinite(output).all():
        raise ValueError("object21 must contain 21 finite values")
    hole = np.asarray(hole_relative)
    if hole.shape != (3,) or not np.isfinite(hole).all():
        raise ValueError("hole_relative must contain three finite coordinates")
    output[7:10] += output[10:13] - hole
    output[10:13] = hole
    return output
