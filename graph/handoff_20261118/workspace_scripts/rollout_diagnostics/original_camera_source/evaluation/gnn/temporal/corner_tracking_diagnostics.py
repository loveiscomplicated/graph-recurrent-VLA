"""Read-only GT audit and tracking-policy probes on frozen corner checkpoints.

GT labels classify outcomes after each online tracking decision. Neither the
tracker nor the tested association/admission policies receive those labels.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.temporal.corner_tracks import masked_frame, read_episode
from graph.temporal.corner_tracking.tracker import AssociationCalibration, CornerTracker, TrackingSettings
from training.gnn.temporal.corner_tracking import build_models, plans_for
from utils.experiment_logger import ExperimentLogger, write


def distribution(values):
    """Report absent populations explicitly, and distances in millimetres."""
    if not values:
        return dict(count=0, mean_mm=None, median_mm=None, p90_mm=None)
    values = np.asarray(values) * 1000
    return dict(count=len(values), mean_mm=float(values.mean()), median_mm=float(np.median(values)),
                p90_mm=float(np.quantile(values, 0.9)))


class AuditTracker(CornerTracker):
    """Expose candidate geometry and optionally test a recent-track cascade."""

    def __init__(self, *args, association="baseline", **kwargs):
        super().__init__(*args, **kwargs)
        self.association = association
        self.ambiguous_tracks = set()
        self.positions = {}

    def _associate(self, tracks, positions, observation):
        self.positions = {track.identifier: position.copy() for track, position in zip(tracks, positions)}
        if self.association == "baseline":
            output = super()._associate(tracks, positions, observation)
        elif self.association == "absolute_distance":
            output = self._associate_absolute(tracks, positions, observation)
        else:
            recent = [i for i, track in enumerate(tracks)
                      if track.confirmed and self.frame - track.anchor_frame <= 2]
            remaining = [i for i in range(len(tracks)) if i not in recent]
            accepted, ambiguous_rows, ambiguous_columns = {}, set(), set()
            available = list(range(len(observation.xyz)))
            for group in (recent, remaining):
                if not group or not available:
                    continue
                subset = observation.subset(np.asarray(available, dtype=int))
                matches, rows, columns = super()._associate([tracks[i] for i in group], positions[group], subset)
                accepted.update({identifier: available[column] for identifier, column in matches.items()})
                held = {available[column] for column in columns}
                ambiguous_rows.update(rows)
                ambiguous_columns.update(held)
                occupied = set(accepted.values()) | held
                available = [column for column in available if column not in occupied]
            output = accepted, ambiguous_rows, ambiguous_columns
        self.ambiguous_tracks = output[1]
        return output

    def _associate_absolute(self, tracks, positions, observation):
        """Keep age-specific feasibility gates, rank candidates in common metres."""
        rows, columns = len(tracks), len(observation.xyz)
        if not rows or not columns:
            return {}, set(), set()
        radii = np.asarray([self.calibration.radius(track.kind, self.frame - track.anchor_frame)
                            for track in tracks])
        distances = np.linalg.norm(positions[:, None] - observation.xyz[None], axis=-1)
        kinds = np.asarray([track.kind for track in tracks])
        allowed = (distances <= radii[:, None]) & (kinds[:, None] == observation.kind[None])
        costs = np.where(allowed, distances, 1e6)
        unmatched_cost = float(radii.max()) + 1e-6
        augmented = np.concatenate((costs, np.full((rows, rows), unmatched_cost)), axis=1)
        row_indices, column_indices = linear_sum_assignment(augmented)
        accepted, ambiguous_rows, ambiguous_columns = {}, set(), set()
        for row, column in zip(row_indices, column_indices):
            if column >= columns or not allowed[row, column]:
                continue
            second = min(radii[row], float(np.delete(costs[row], column).min(initial=np.inf)),
                         float(np.delete(costs[:, column], row).min(initial=np.inf)))
            if costs[row, column] + 1e-8 <= self.settings.ambiguity_ratio * second:
                accepted[tracks[row].identifier] = int(column)
            else:
                ambiguous_rows.add(tracks[row].identifier)
                ambiguous_columns.add(int(column))
        for column in range(columns):
            candidates = np.flatnonzero(allowed[:, column])
            if column not in accepted.values() and len(candidates):
                ambiguous_columns.add(column)
                ambiguous_rows.update(tracks[row].identifier for row in candidates)
        return accepted, ambiguous_rows, ambiguous_columns


def audit_episode(motion, episode, plan, calibration, settings, association):
    """Partition failure denominators and distinguish swaps from fragmentation."""
    tracker = AuditTracker(motion, calibration, settings, association=association)
    bindings, last_certified, last_track, last_detection = {}, {}, {}, {}
    ever_confirmed, seen = set(), set()
    previous_missing = set()
    counts, switches, missing_ages = Counter(), Counter(), Counter()
    admission = defaultdict(Counter)
    errors = defaultdict(list)
    probes = {"current": (settings.graph_max_missing_frames, settings.graph_radius95_m),
              "age16_5mm": (16, .005), "age32_5mm": (32, .005),
              "age8_10mm": (8, .010), "age16_10mm": (16, .010)}
    for frame_index, frame in enumerate(episode.frames):
        observation, identities, _ = masked_frame(frame, plan, frame_index)
        old_tracks = {key: dict(anchor_frame=track.anchor_frame, confirmed=track.confirmed,
                                anchor=track.anchor.copy()) for key, track in tracker.tracks.items()}
        result = tracker.step(observation, include_graph=True, validate_graph=False)
        accepted = dict(result.assignments)
        accepted.update(result.births)
        current_ids = list(map(int, identities[identities >= 0]))
        counts["duplicate_labeled_detections"] += len(current_ids) - len(set(current_ids))
        for identifier, index in accepted.items():
            reference = int(identities[index])
            if reference < 0:
                continue
            previous_reference = last_certified.get(identifier)
            if previous_reference is not None and previous_reference != reference:
                counts["track_identity_changes"] += 1
            last_certified[identifier] = reference
            bindings.setdefault(identifier, reference)
            if identifier in previous_missing:
                counts["reobservations"] += 1
                counts["false_reconnections"] += int(bindings[identifier] != reference)
            if tracker.tracks[identifier].confirmed:
                ever_confirmed.add(reference)
                previous = last_track.get(reference)
                if previous is not None and previous[0] != identifier:
                    counts["id_switches"] += 1
                    old_identifier, previous_frame = previous
                    if previous_frame == frame_index:
                        switches["same_frame_duplicate"] += 1
                    elif previous_reference is not None and previous_reference != reference:
                        switches["existing_track_changed_physical_corner"] += 1
                    elif old_identifier not in tracker.tracks:
                        switches["new_track_after_previous_retired"] += 1
                    else:
                        switches["another_track_while_previous_alive"] += 1
                    if old_identifier in old_tracks:
                        old = old_tracks[old_identifier]
                        age = frame_index - old["anchor_frame"]
                        switches[f"previous_age_{'1_2' if age <= 2 else '3_8' if age <= 8 else '9_32' if age <= 32 else '33_plus'}"] += 1
                        position = tracker.positions.get(old_identifier)
                        candidates = np.flatnonzero(identities == reference)
                        if position is not None and len(candidates):
                            distance = np.linalg.norm(observation.xyz[candidates] - position, axis=-1).min()
                            radius = calibration.radius(int(frame.reference_kind[reference]), age)
                            reason = "true_detection_outside_previous_gate" if distance > radius else "true_detection_inside_previous_gate"
                            switches[reason] += 1
                last_track[reference] = (identifier, frame_index)
        active_refs = [bindings[key] for key, track in tracker.tracks.items()
                       if track.confirmed and key in bindings]
        counts["duplicate_track_frames"] += len(active_refs) - len(set(active_refs))
        counts["ambiguous_detection_frames"] += len(result.ambiguous_detections)
        missing = seen - set(current_ids)
        for reference in missing:
            age = frame_index - last_detection[reference]
            missing_ages["1_8" if age <= 8 else "9_32" if age <= 32 else "33_plus"] += 1
        eligible = {key: prior for key, prior in result.priors.items() if key not in result.assignments}
        prior_errors = {}
        for identifier, prior in result.priors.items():
            reference = bindings.get(identifier)
            if reference is None:
                continue
            error = float((prior.position - motion.tensor(frame.reference_xyz[reference])).norm().cpu())
            prior_errors[identifier] = error
            if reference in missing and last_certified.get(identifier) == reference:
                errors["all_missing_priors"].append(error)
        for name, (maximum_age, maximum_radius) in probes.items():
            adopted = set()
            covered, covered_accurately = set(), set()
            for identifier, prior in eligible.items():
                radius = float(prior.prediction.variance.sum().sqrt().cpu()) * calibration.scale(prior.kind, prior.age)
                if (prior.age <= maximum_age and identifier not in tracker.ambiguous_tracks
                        and radius <= maximum_radius):
                    adopted.add(identifier)
                    reference = bindings.get(identifier)
                    if reference is None:
                        admission[name]["unknown_or_inconsistent_adoptions"] += 1
                        admission[name]["unknown_adoptions"] += 1
                        continue
                    consistent = last_certified.get(identifier) == reference
                    if not consistent:
                        admission[name]["unknown_or_inconsistent_adoptions"] += 1
                        admission[name]["inconsistent_adoptions"] += 1
                    error = prior_errors[identifier]
                    admission[name]["auditable_adoptions"] += 1
                    admission[name]["accurate_adoptions_5mm"] += int(consistent and error <= .005)
                    if consistent and reference in missing:
                        covered.add(reference)
                        if error <= .005:
                            covered_accurately.add(reference)
                    if name == "current":
                        errors["adopted_predictions"].append(error)
            admission[name]["missing_corner_frames"] += len(missing)
            admission[name]["covered_missing_corner_frames"] += len(covered & missing)
            admission[name]["accurate_missing_corner_frames_5mm"] += len(covered_accurately & missing)
            if name == "current":
                assert adopted == set(result.adopted_predictions)
                for reference in missing - covered:
                    candidates = [(key, prior) for key, prior in result.priors.items()
                                  if bindings.get(key) == reference and last_certified.get(key) == reference]
                    if not candidates:
                        if reference not in ever_confirmed:
                            reason = "never_confirmed"
                        elif reference in last_track and last_track[reference][0] not in tracker.tracks:
                            reason = "previous_confirmed_track_retired"
                        else:
                            reason = "no_consistent_live_confirmed_track"
                    else:
                        candidates = [(key, prior) for key, prior in candidates if key in eligible]
                        if not candidates:
                            reason = "assigned_to_uncertified_detection"
                        else:
                            short = [(key, prior) for key, prior in candidates
                                     if prior.age <= settings.graph_max_missing_frames]
                            distinct = [(key, prior) for key, prior in short if key not in tracker.ambiguous_tracks]
                            calibrated = [(key, prior) for key, prior in distinct if np.isfinite(calibration.scale(prior.kind, prior.age))]
                            if not short:
                                reason = "age_over_limit"
                            elif not distinct:
                                reason = "association_ambiguity"
                            elif not calibrated:
                                reason = "unsupported_calibration"
                            else:
                                reason = "calibrated_radius_over_5mm"
                    counts[f"missing_excluded:{reason}"] += 1
        for reference in current_ids:
            last_detection[reference] = frame_index
        seen.update(current_ids)
        previous_missing = {identifier for identifier, status in result.statuses.items()
                            if status in ("missing", "dormant")}
    return dict(counts=dict(counts), switches=dict(switches), missing_ages=dict(missing_ages),
                admission={key: dict(value) for key, value in admission.items()}, errors=dict(errors))


def merge_reports(reports):
    """Aggregate counts without averaging ratios or hiding error tails."""
    merged = {key: Counter() for key in ("counts", "switches", "missing_ages")}
    admission, errors = defaultdict(Counter), defaultdict(list)
    for report in reports:
        for key in merged:
            merged[key].update(report[key])
        for key, values in report["admission"].items():
            admission[key].update(values)
        for key, values in report["errors"].items():
            errors[key].extend(values)
    output = {key: dict(value) for key, value in merged.items()}
    for key, value in admission.items():
        denominator = value["missing_corner_frames"]
        value["coverage"] = value["covered_missing_corner_frames"] / denominator if denominator else None
        value["accurate_coverage_5mm"] = value["accurate_missing_corner_frames_5mm"] / denominator if denominator else None
        count = value["auditable_adoptions"]
        value["adoption_precision_5mm"] = value["accurate_adoptions_5mm"] / count if count else None
    output["admission"] = {key: dict(value) for key, value in admission.items()}
    output["errors"] = {key: distribution(value) for key, value in errors.items()}
    return output


def evaluate(config):
    """Root-main diagnostic entrypoint, saving frozen checkpoint identities."""
    requested_episodes = config.get("episodes", 8)
    associations = config.get("associations", ["baseline", "recent_first"])
    if type(requested_episodes) is not int or requested_episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if (not associations or len(set(associations)) != len(associations)
            or any(name not in ("baseline", "recent_first", "absolute_distance") for name in associations)):
        raise ValueError("associations must contain unique baseline/recent_first/absolute_distance names")
    torch.set_num_threads(1)
    checkpoint_path = Path(config["checkpoint"]).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") not in ("corner_tracking_last", "corner_tracking"):
        raise ValueError("checkpoint must be a corner tracking best.pt or last.pt")
    effective = deepcopy(checkpoint["config"])
    effective["execution"]["compile"] = False
    effective["device"] = "cpu"
    root = Path(effective["data"]["root"])
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["dataset_sha256"] != effective["dataset_sha256"]:
        raise ValueError("Diagnostic dataset differs from checkpoint")
    rows = [row for row in manifest["episodes"] if row["split"] == "validation"]
    count = min(requested_episodes, len(rows))
    if not count:
        raise ValueError("Diagnostic dataset contains no validation episodes")
    selected = np.linspace(0, len(rows) - 1, count, dtype=int)
    episodes = []
    for index in selected:
        row = rows[index]
        path = root / row["file"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"Diagnostic episode checksum mismatch: {path}")
        episodes.append(read_episode(path, row))
    motion, _ = build_models(effective, effective["architecture"], effective["seed"], torch.device("cpu"))
    motion.load_state_dict(checkpoint["motion"])
    motion.eval()
    calibration = AssociationCalibration.from_dict(checkpoint["calibration"])
    settings = TrackingSettings(**effective["tracking"])
    plans = plans_for(episodes, effective["seed"], "validation", effective, calibration)
    report = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=file_sha256(checkpoint_path),
                  epoch=checkpoint["epoch"], architecture=effective["architecture"], seed=effective["seed"],
                  dataset_sha256=effective["dataset_sha256"], episodes=[episode.name for episode in episodes],
                  frames=sum(len(episode.frames) for episode in episodes),
                  mask_sha256=canonical_hash(plans), variants={})
    with ExperimentLogger(team="graph", config=config, seed=effective["seed"],
                          model="corner_tracking_diagnostics", job_type="evaluation") as logger:
        logger.link_output(checkpoint_path, checkpoint=True)
        logger.record(checkpoint_sha256=report["checkpoint_sha256"],
                      diagnostic_source_sha256=file_sha256(Path(__file__)), device="cpu")
        for association in associations:
            started = time.perf_counter()
            with torch.no_grad():
                results = [audit_episode(motion, episode, plan, calibration, settings, association)
                           for episode, plan in zip(episodes, plans)]
            report["variants"][association] = merge_reports(results)
            if association == "baseline" and count == len(rows) and checkpoint.get("history"):
                saved = checkpoint["history"][-1]["validation"]
                audited = report["variants"][association]
                names = {"id_switches": "id_switches", "track_identity_changes": "track_identity_changes",
                         "duplicate_tracks": "duplicate_track_frames", "reobservations": "reobservations",
                         "false_reconnections": "false_reconnections"}
                report["baseline_reproduction"] = dict(
                    count_differences={key: audited["counts"].get(value, 0) - saved["tracking"][key]
                                       for key, value in names.items()},
                    coverage_difference=audited["admission"]["current"]["coverage"] - saved["certified_missing_coverage"],
                )
            print(f"{effective['architecture']} {association}: {len(episodes)} episodes, "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
        output = Path(config["output"]["root"]) / logger.run_id / "diagnostics.json"
        write(output, report)
        logger.link_output(output)
        print(f"Tracking diagnostics: {output}", flush=True)
        return str(output)
