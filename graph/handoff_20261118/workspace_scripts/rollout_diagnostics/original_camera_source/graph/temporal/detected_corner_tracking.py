"""Simple detector-only, one-to-one matching for shared corner predictors."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def match_positions(predictions, track_kinds, detections, detection_kinds, distance_limit_m):
    """Maximize valid matches, then minimize distance; unmatched is explicit."""
    tracks, points = len(predictions), len(detections)
    if distance_limit_m <= 0:
        raise ValueError("distance_limit_m must be positive")
    if not tracks or not points:
        return []
    distances = np.linalg.norm(predictions[:, None, :] - detections[None, :, :], axis=-1)
    allowed = (track_kinds[:, None] == detection_kinds[None, :]) & (distances <= distance_limit_m)
    # A dummy costs more than every possible distance saving, so the assignment
    # first maximizes the number of valid pairs, then minimizes their distance.
    dummy_cost = (tracks + 1) * distance_limit_m
    cost = np.full((tracks, points + tracks), dummy_cost, dtype=np.float64)
    cost[:, :points] = np.where(allowed, distances, dummy_cost * (tracks + 2))
    row, col = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(row, col) if j < points and allowed[i, j]]


@dataclass
class CornerTrack:
    """One internally assigned ID with actual detector history only."""

    track_id: int
    kind: int
    last_observed: np.ndarray
    previous_observed: bool
    previous_xyz: np.ndarray
    inputs: list = field(default_factory=list)


class DetectedCornerTracker:
    """Track every observed corner without GT identities or retirement rules."""

    def __init__(self, model, distance_limit_m, device):
        self.model = model.eval()
        self.distance_limit_m = distance_limit_m
        self.device = torch.device(device)
        self.tracks = []

    def predict(self, robot):
        """Append current causal input, then forecast each existing corner."""
        if not self.tracks:
            return np.empty((0, 3), dtype=np.float32)
        for track in self.tracks:
            previous = track.previous_xyz if track.previous_observed else np.zeros(3, dtype=np.float32)
            track.inputs.append(np.concatenate((previous, np.array([track.previous_observed], dtype=np.float32), robot)))
        maximum = max(len(track.inputs) for track in self.tracks)
        inputs = np.zeros((len(self.tracks), maximum, 13), dtype=np.float32)
        mask = np.zeros((len(self.tracks), maximum), dtype=bool)
        for index, track in enumerate(self.tracks):
            length = len(track.inputs)
            inputs[index, :length] = track.inputs
            mask[index, :length] = True
        context = torch.as_tensor(mask, device=self.device)
        with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            delta = self.model(torch.as_tensor(inputs, device=self.device), context).float()
            lengths = context.sum(dim=1)
            displacement = delta[torch.arange(len(self.tracks), device=self.device), lengths - 1].cpu().numpy()
        return np.stack([track.last_observed for track in self.tracks]) + displacement

    def step(self, detections, kinds, robot):
        """Predict, match and update from one frame; return actual/forecast XYZ."""
        detections = np.asarray(detections, dtype=np.float32)
        kinds = np.asarray(kinds, dtype=np.int64)
        robot = np.asarray(robot, dtype=np.float32)
        if detections.ndim != 2 or detections.shape[1] != 3 or kinds.shape != (len(detections),):
            raise ValueError("detections must be [N,3] and kinds [N]")
        if robot.shape != (9,) or not np.isfinite(robot).all() or not np.isfinite(detections).all():
            raise ValueError("robot [9] and detections must be finite")
        if np.any((kinds != 0) & (kinds != 1)):
            raise ValueError("Detector kind must be 0=peg or 1=box")
        prediction = self.predict(robot)
        track_kinds = np.asarray([track.kind for track in self.tracks], dtype=np.int64)
        matches = match_positions(prediction, track_kinds, detections, kinds, self.distance_limit_m)
        output_positions = [position.copy() for position in prediction]
        observed = [False] * len(self.tracks)
        detection_tracks = np.full(len(detections), -1, dtype=np.int64)
        for track in self.tracks:
            track.previous_observed = False
        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            position = detections[detection_index].copy()
            track.last_observed = position
            track.previous_xyz = position
            track.previous_observed = True
            output_positions[track_index] = position
            observed[track_index] = True
            detection_tracks[detection_index] = track.track_id
        for index in np.flatnonzero(detection_tracks < 0):
            position = detections[index].copy()
            track = CornerTrack(len(self.tracks), int(kinds[index]), position, True, position.copy())
            # A first sighting has no previous observation, just as in training.
            track.inputs.append(np.concatenate((np.zeros(4, dtype=np.float32), robot)))
            self.tracks.append(track)
            detection_tracks[index] = track.track_id
            output_positions.append(position)
            observed.append(True)
        return dict(positions=np.asarray(output_positions, dtype=np.float32).reshape(-1, 3),
                    observed=np.asarray(observed, dtype=bool), detection_tracks=detection_tracks,
                    matches=len(matches), new_tracks=len(detections) - len(matches))
