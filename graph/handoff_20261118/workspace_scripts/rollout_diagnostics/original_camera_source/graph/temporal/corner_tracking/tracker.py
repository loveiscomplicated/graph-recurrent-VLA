"""GT-free online association, track lifecycle and observed-anchor management."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .graph import build_completed_graph
from .model import MotionPrediction


@dataclass
class Observation:
    """One frame's inference inputs only; no physical IDs, GT or mask reasons."""

    xyz: np.ndarray
    kind: np.ndarray
    confidence: np.ndarray
    confidence_valid: np.ndarray
    robot: np.ndarray  # observed TCP world xyz/wxyz and two finger positions

    def validate(self):
        """Validate inference fields without consulting any external annotation."""
        count = len(self.xyz)
        if (self.xyz.shape != (count, 3) or self.robot.shape != (9,)
                or self.kind.shape != (count,) or self.confidence.shape != (count,)
                or self.confidence_valid.shape != (count,)):
            raise ValueError("Need xyz[N,3], kind/confidence/quality_valid[N], robot[9]")
        if (not np.isfinite(self.xyz).all() or not np.isfinite(self.robot).all()
                or not np.isfinite(self.confidence).all()
                or np.any((self.kind < 0) | (self.kind > 2))
                or np.any((self.confidence < 0) | (self.confidence > 1))):
            raise ValueError("Observation must be finite with types 0..2 and quality in [0,1]")
        if self.confidence_valid.dtype != np.bool_:
            raise ValueError("confidence_valid must be boolean")
        if (self.xyz.dtype != np.float32 or self.robot.dtype != np.float32
                or self.confidence.dtype != np.float32 or self.kind.dtype != np.int64):
            raise ValueError("Observation coordinates/robot/quality must be float32 and kind int64")
        if np.any(~self.confidence_valid & (self.confidence != 0)):
            raise ValueError("Unknown observation quality must be zero")
        if not np.isclose(np.linalg.norm(self.robot[3:7]), 1.0, atol=1e-4):
            raise ValueError("Observed TCP quaternion must be unit wxyz")

    def subset(self, keep):
        """Remove points before any association, pooling or graph construction."""
        return Observation(self.xyz[keep], self.kind[keep], self.confidence[keep],
                           self.confidence_valid[keep], self.robot.copy())


@dataclass(frozen=True)
class TrackingSettings:
    confirmation_hits: int = 2
    confirmation_window: int = 3
    graph_max_missing_frames: int = 8
    retirement_missing_frames: int = 33
    ambiguity_ratio: float = 0.8
    dormant_confirmation_hits: int = 2
    graph_radius95_m: float = 0.005

    def __post_init__(self):
        integer_settings = (self.confirmation_hits, self.confirmation_window,
                            self.graph_max_missing_frames, self.retirement_missing_frames,
                            self.dormant_confirmation_hits)
        if any(type(value) is not int for value in integer_settings):
            raise ValueError("Track confirmation counts and frame ages must be integers")
        if not np.isfinite([self.ambiguity_ratio, self.graph_radius95_m]).all():
            raise ValueError("Track ambiguity ratio and uncertainty bound must be finite")
        if (not 1 <= self.confirmation_hits <= self.confirmation_window
                or not 0 < self.graph_max_missing_frames < self.retirement_missing_frames
                or self.dormant_confirmation_hits < 2
                or not 0 < self.ambiguity_ratio < 1 or self.graph_radius95_m <= 0):
            raise ValueError("Invalid confirmation, age, ambiguity or graph uncertainty settings")


@dataclass
class AssociationCalibration:
    """Train-calibration-only distance gates and empirically scaled uncertainty."""

    radii: np.ndarray  # [3, max_age+1], metres
    variance_scale: np.ndarray = field(default_factory=lambda: np.full(3, np.inf))

    def __post_init__(self):
        if (self.radii.ndim != 2 or self.radii.shape[0] != 3 or self.radii.shape[1] < 2
                or not np.isfinite(self.radii).all() or np.any(self.radii <= 0)):
            raise ValueError("Association radii must be positive finite [3, ages>=2]")
        if (self.variance_scale.shape not in ((3,), self.radii.shape)
                or np.isnan(self.variance_scale).any() or np.any(self.variance_scale <= 0)):
            raise ValueError("variance_scale must be positive [3] or [3,ages]; infinity disables support")

    def radius(self, kind, age):
        """Read the fixed empirical association gate in metres."""
        return float(self.radii[kind, min(age, self.radii.shape[1] - 1)])

    def scale(self, kind, age):
        """Missing-age-specific calibration; unsupported bins cannot emit nodes."""
        if self.variance_scale.ndim == 1:
            return float(self.variance_scale[kind])
        return float(self.variance_scale[kind, min(age, self.radii.shape[1] - 1)])

    def as_dict(self):
        """Serialize disabled bins as null rather than nonstandard JSON infinity."""
        scales = self.variance_scale.astype(object)
        scales[~np.isfinite(self.variance_scale)] = None
        return dict(radii=self.radii.tolist(), variance_scale=scales.tolist())

    @classmethod
    def from_dict(cls, value):
        """Restore both object-wide and age-specific calibration formats."""
        scales = np.asarray(value["variance_scale"], dtype=float)
        scales[np.isnan(scales)] = np.inf
        return cls(np.asarray(value["radii"]), scales)


@dataclass
class Track:
    identifier: int
    kind: int
    anchor: np.ndarray
    anchor_robot: np.ndarray
    anchor_frame: int
    birth_frame: int
    quality: float
    quality_valid: bool
    memory: dict
    hits: int = 1
    confirmed: bool = False
    pending_frame: int = -2
    pending_hits: int = 0
    status: str = "tentative"


@dataclass
class Prior:
    position: torch.Tensor
    prediction: MotionPrediction
    anchor: np.ndarray
    age: int
    kind: int


@dataclass
class TrackingResult:
    graph: object
    priors: dict
    assignments: dict
    births: dict
    retired: list
    ambiguous_detections: set
    adopted_predictions: list
    eligible_predictions: list
    statuses: dict


class CornerTracker:
    """One instance per episode; model weights can be shared between episodes."""

    def __init__(self, model, calibration, settings=None, *, batch_tracks=True):
        self.model = model
        self.calibration = calibration
        self.settings = settings or TrackingSettings()
        self.batch_tracks = batch_tracks
        self.tracks = {}
        self.next_identifier = 0
        self.frame = -1
        self.previous_robot = None

    def _associate(self, tracks, positions, observation):
        """Allow unmatched rows and require a distinct best choice in both directions."""
        rows, columns = len(tracks), len(observation.xyz)
        if not rows or not columns:
            return {}, set(), set()
        costs = np.full((rows, columns), 1e6)
        for row, track in enumerate(tracks):
            distances = np.linalg.norm(observation.xyz - positions[row], axis=-1)
            radius = self.calibration.radius(track.kind, self.frame - track.anchor_frame)
            allowed = (observation.kind == track.kind) & (distances <= radius)
            costs[row, allowed] = distances[allowed] / radius
        # Private dummy columns mean no-match competes in the optimization.
        augmented = np.concatenate((costs, np.ones((rows, rows))), axis=1)
        row_indices, column_indices = linear_sum_assignment(augmented)
        accepted = {}
        ambiguous_rows, ambiguous_columns = set(), set()
        for row, column in zip(row_indices, column_indices):
            if column >= columns or costs[row, column] >= 1:
                continue
            cost = costs[row, column]
            other_row = np.delete(costs[row], column)
            other_column = np.delete(costs[:, column], row)
            second = min(1.0, float(other_row.min(initial=1.0)),
                         float(other_column.min(initial=1.0)))
            distinct = cost + 1e-8 <= self.settings.ambiguity_ratio * second
            if distinct:
                accepted[tracks[row].identifier] = int(column)
            else:
                ambiguous_rows.add(tracks[row].identifier)
                ambiguous_columns.add(int(column))
        # Every unaccepted detection near an existing hypothesis is held, not
        # reborn as a duplicate. Suppress competing predicted graph support too.
        matched_columns = set(accepted.values())
        for column in range(columns):
            candidates = np.flatnonzero(costs[:, column] <= 1)
            if column not in matched_columns and len(candidates):
                ambiguous_columns.add(column)
                ambiguous_rows.update(tracks[row].identifier for row in candidates)
        return accepted, ambiguous_rows, ambiguous_columns

    def _association_positions(self, tracks, positions, observation):
        """Use the last observation if a one-frame prediction has no gated candidate.

        This only supplies a fallback for association. The predicted position
        and its gradient remain unchanged for supervision and graph completion.
        Longer gaps still use the motion prediction and normal ambiguity rules.
        """
        positions = positions.copy()
        for index, track in enumerate(tracks):
            age = self.frame - track.anchor_frame
            if age != 1:
                continue
            candidates = observation.xyz[observation.kind == track.kind]
            if not len(candidates):
                continue
            radius = self.calibration.radius(track.kind, age)
            predicted_distance = np.linalg.norm(candidates - positions[index], axis=1).min()
            observed_distance = np.linalg.norm(candidates - track.anchor, axis=1).min()
            if predicted_distance > radius and observed_distance <= radius:
                positions[index] = track.anchor
        return positions

    def step(self, observation, *, include_graph=True, validate_graph=True):
        """Predict -> associate -> construct graph -> commit this frame once."""
        observation.validate()
        self.frame += 1
        retired = []
        for identifier, track in list(self.tracks.items()):
            expired = self.frame - track.anchor_frame >= self.settings.retirement_missing_frames
            unconfirmed = (not track.confirmed
                           and self.frame - track.birth_frame >= self.settings.confirmation_window)
            if expired or unconfirmed:
                retired.append(identifier)
                del self.tracks[identifier]

        tracks = list(self.tracks.values())
        priors = {}
        if self.batch_tracks and hasattr(self.model, "predict_many"):
            predictions = self.model.predict_many(tracks, observation.robot, self.frame)
        else:
            predictions = [self.model.predict(track.memory, track.anchor, track.anchor_robot,
                                               observation.robot, self.frame - track.anchor_frame,
                                               track.kind, track.quality, track.quality_valid)
                           for track in tracks]
        position_tensors = (self.model.tensor(np.stack([track.anchor for track in tracks]))
                            + torch.stack([prediction.delta for prediction in predictions])
                            if tracks else None)
        for index, (track, prediction) in enumerate(zip(tracks, predictions)):
            age = self.frame - track.anchor_frame
            if track.confirmed:
                priors[track.identifier] = Prior(position_tensors[index], prediction,
                                                track.anchor.copy(), age, track.kind)
        # Transfer all predictions once per frame instead of synchronizing for
        # every node before the CPU/SciPy assignment.
        positions = (position_tensors.detach().cpu().numpy()
                     if position_tensors is not None else np.empty((0, 3), dtype=np.float32))
        association_positions = self._association_positions(tracks, positions, observation)
        proposed, ambiguous_tracks, ambiguous_detections = self._associate(
            tracks, association_positions, observation
        )
        assignments = {}
        pending_detections = set()
        for track in tracks:
            index = proposed.get(track.identifier)
            age = self.frame - track.anchor_frame
            dormant = track.confirmed and age > self.settings.graph_max_missing_frames
            if index is None:
                track.pending_hits = 0
                continue
            if dormant:
                track.pending_hits = (track.pending_hits + 1
                                      if track.pending_frame == self.frame - 1 else 1)
                track.pending_frame = self.frame
                if track.pending_hits < self.settings.dormant_confirmation_hits:
                    pending_detections.add(index)
                    continue
            assignments[track.identifier] = index
            track.anchor = observation.xyz[index].copy()
            track.anchor_robot = observation.robot.copy()
            track.anchor_frame = self.frame
            track.quality = float(observation.confidence[index])
            track.quality_valid = bool(observation.confidence_valid[index])
            track.hits += 1
            track.confirmed = track.hits >= self.settings.confirmation_hits
            track.pending_hits = 0

        births = {}
        occupied = set(proposed.values()) | ambiguous_detections | pending_detections
        for index in range(len(observation.xyz)):
            if index in occupied:
                continue
            identifier = self.next_identifier
            self.next_identifier += 1
            track = Track(identifier, int(observation.kind[index]), observation.xyz[index].copy(),
                          observation.robot.copy(), self.frame, self.frame,
                          float(observation.confidence[index]), bool(observation.confidence_valid[index]),
                          self.model.new_memory(), confirmed=self.settings.confirmation_hits == 1)
            self.tracks[identifier] = track
            births[identifier] = index

        adopted, eligible = [], []
        graph = None
        if include_graph:
            # Real observations remain anonymous geometric evidence when their
            # identity is pending; they do not overwrite an uncertain anchor.
            xyz = [self.model.tensor(point) for point in observation.xyz]
            kinds = observation.kind.tolist()
            observed = [True] * len(xyz)
            ages = [0] * len(xyz)
            deviations = [torch.zeros(3, device=self.model.device) for _ in xyz]
            qualities = observation.confidence.tolist()
            quality_valid = observation.confidence_valid.tolist()
            unmatched = [identifier for identifier in priors if identifier not in assignments]
            std_norms = (torch.stack([priors[identifier].prediction.variance for identifier in unmatched])
                         .sum(dim=-1).sqrt().detach().cpu().numpy()
                         if unmatched else np.empty(0, dtype=np.float32))
            std_by_id = dict(zip(unmatched, std_norms))
            for identifier, prior in priors.items():
                if identifier in assignments:
                    continue
                eligible.append(identifier)
                scale = self.calibration.scale(prior.kind, prior.age)
                radius95 = float(std_by_id[identifier]) * scale
                if (prior.age <= self.settings.graph_max_missing_frames
                        and identifier not in ambiguous_tracks
                        and radius95 <= self.settings.graph_radius95_m):
                    xyz.append(prior.position)
                    kinds.append(prior.kind)
                    observed.append(False)
                    ages.append(prior.age)
                    deviations.append(prior.prediction.variance.sqrt() * scale)
                    qualities.append(0.0)
                    quality_valid.append(False)
                    adopted.append(identifier)
            graph = build_completed_graph(xyz, kinds, observed, ages, deviations, qualities,
                                          quality_valid, observation.robot[:3], self.model.device,
                                          validate=validate_graph)

        contexts = self.model.contexts(observation)
        records = []
        record_tracks = []
        for identifier, track in self.tracks.items():
            index = assignments.get(identifier, births.get(identifier))
            age = self.frame - track.anchor_frame
            if not track.confirmed:
                track.status = "tentative"
            elif index is not None:
                track.status = "observed"
            elif age <= self.settings.graph_max_missing_frames:
                track.status = "missing"
            else:
                track.status = "dormant"
            records.append(dict(memory=track.memory, kind=track.kind, age=age, context=contexts[track.kind],
                                xyz=None if index is None else observation.xyz[index],
                                quality=0.0 if index is None else float(observation.confidence[index]),
                                quality_valid=False if index is None else bool(observation.confidence_valid[index])))
            record_tracks.append(track)
        if self.batch_tracks and hasattr(self.model, "record_many"):
            memories = self.model.record_many(records, observation.robot, self.previous_robot)
        else:
            memories = [self.model.record(row["memory"], observation.robot, self.previous_robot,
                                          row["kind"], row["age"], row["context"], row["xyz"],
                                          row["quality"], row["quality_valid"]) for row in records]
        for track, memory in zip(record_tracks, memories):
            track.memory = memory
        self.previous_robot = observation.robot.copy()
        return TrackingResult(graph, priors, assignments, births, retired,
                              ambiguous_detections, adopted, eligible,
                              {key: track.status for key, track in self.tracks.items()})
