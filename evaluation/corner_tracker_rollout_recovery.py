"""Recover causal scoring bindings from an exact, frozen online rollout.

Observed track rows contain the unchanged detector coordinate. An exact,
one-to-one coordinate/type join recovers detection-to-track assignments when
there are no duplicates. Forecasts are never compared with GT to assign IDs.
"""

import json

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256


def detection_tracks(points, kinds, output, output_kinds, observed):
    """Join actual detections to observed output rows exactly, refusing ambiguity."""
    input_keys = [(int(kind), *point.tolist()) for kind, point in zip(kinds, points)]
    rows = np.flatnonzero(observed)
    output_keys = [(int(output_kinds[row]), *output[row].tolist()) for row in rows]
    if len(set(input_keys)) != len(input_keys) or len(set(output_keys)) != len(output_keys):
        raise ValueError("Ambiguous duplicate detector coordinates; replay online inference instead")
    if set(input_keys) != set(output_keys):
        raise ValueError("Frozen observed outputs differ from current-frame detector inputs")
    assignments = dict(zip(output_keys, rows.tolist()))
    return np.asarray([assignments[key] for key in input_keys], dtype=np.int64)


def recover(config, comparison, reference):
    """Verify a pinned test trace and reconstruct all online scoring identities."""
    # Import here to keep the coordinate join independently testable.
    from evaluation.corner_tracker_comparison import checked_file, member_file, validate_online_replay

    path = checked_file(config["online_rollout"], reference["rollout_cache_sha256"])
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if (cache.get("format") != "spatial_temporal_test_rollouts_v1" or cache.get("split") != "test"
            or cache["tracker_checkpoint_sha256"] != config["tracker"]["sha256"]
            or cache["association"] != config["association"]):
        raise ValueError("Frozen online rollout identity differs from the fixed test protocol")
    arrays = cache["arrays"]
    np.testing.assert_array_equal(arrays["sample_ids"], comparison["sample_ids"])
    manifest_path = checked_file(config["data"]["manifest"], reference["prepared_manifest_sha256"])
    records = json.loads(manifest_path.read_text())["episodes"]
    predictions, available, reference_observed = [], [], []
    count_names = ("known_detections", "association_targets", "association_correct", "identity_switches",
                   "known_track_conflicts", "reappearances", "reappearance_same_id", "missing_targets")
    counts = dict.fromkeys(count_names, 0)
    squared_error = 0.0
    frame_global = 0
    final_track_counts = []
    new_track_count = 0
    for index, record in enumerate(records):
        episode_path = member_file(manifest_path.parent, record["file"], record["sha256"])
        with np.load(episode_path, allow_pickle=False) as archive:
            episode = dict(archive)
        bindings, track_identity, last_seen = {}, {}, {}
        previous_count = 0
        for frame, (start, stop) in enumerate(zip(episode["offsets"][:-1], episode["offsets"][1:])):
            first, last = arrays["offsets"][frame_global:frame_global + 2]
            positions = arrays["xyz"][first:last]
            observed = arrays["observed"][first:last]
            if len(positions) < previous_count or len(positions) > 128:
                raise ValueError("Frozen online track lifecycle/guard was violated")
            new_track_count += len(positions) - previous_count
            previous_count = len(positions)
            corners = episode["is_corner"][start:stop]
            tracks = detection_tracks(episode["xyz"][start:stop][corners],
                                      episode["kinds"][start:stop][corners], positions,
                                      arrays["kinds"][first:last], observed)
            physical_ids = episode["vertex_ids"][start:stop][corners]
            current = set()
            for physical, track in zip(physical_ids.tolist(), tracks.tolist()):
                if physical < 0:
                    continue
                current.add(physical)
                counts["known_detections"] += 1
                prior = bindings.get(physical)
                if track in track_identity and track_identity[track] != physical:
                    counts["known_track_conflicts"] += 1
                if prior is not None:
                    counts["association_targets"] += 1
                    counts["association_correct"] += int(prior == track)
                    counts["identity_switches"] += int(prior != track)
                if physical in last_seen and frame - last_seen[physical] > 1:
                    counts["reappearances"] += 1
                    counts["reappearance_same_id"] += int(prior == track)
                track_identity.setdefault(track, physical)
                bindings[physical] = track
                last_seen[physical] = frame
            prediction = np.full((20, 3), np.nan, dtype=np.float32)
            present = np.zeros(20, dtype=bool)
            physical_observed = np.zeros(20, dtype=bool)
            for physical, track in bindings.items():
                if physical < 20:
                    prediction[physical] = positions[track]
                    present[physical] = True
                if physical not in current:
                    delta = positions[track] - episode["targets"][frame, physical]
                    squared_error += float(delta @ delta)
                    counts["missing_targets"] += 1
            for physical in current:
                if physical < 20:
                    physical_observed[physical] = True
            predictions.append(prediction)
            available.append(present)
            reference_observed.append(physical_observed)
            frame_global += 1
        final_track_counts.append(previous_count)
        if index % 50 == 0:
            print(f"Recovered exact online bindings {index + 1}/{len(records)}", flush=True)
    np.testing.assert_array_equal(comparison["observed"], reference_observed)
    original = reference["detector_only_online"]
    online = dict(original)
    online.update(counts, episodes=len(records), frames=frame_global,
                  new_tracks=new_track_count, matches=int(arrays["observed"].sum()) - new_track_count,
                  max_final_tracks=max(final_track_counts), mean_final_tracks=float(np.mean(final_track_counts)),
                  missing_rmse_mm=1000 * np.sqrt(squared_error / counts["missing_targets"]))
    validate_online_replay(online, reference)
    for key in ("mean_final_tracks", "association_accuracy", "reappearance_same_id_rate"):
        actual = {"mean_final_tracks": float(np.mean(final_track_counts)),
                  "association_accuracy": counts["association_correct"] / counts["association_targets"],
                  "reappearance_same_id_rate": counts["reappearance_same_id"] / counts["reappearances"]}[key]
        if actual != original[key]:
            raise ValueError(f"Recovered online {key} differs from the published result")
    comparison["online_prediction"] = np.asarray(predictions)
    comparison["online_available"] = np.asarray(available)
    return online, {
        "mode": "exact_frozen_rollout_recovery", "frames": frame_global,
        "ambiguous_detection_frames": 0, "all_original_integer_counts_preserved": True,
        "original_missing_rmse_reproduced": True, "rollout_sha256": file_sha256(path),
        "reappearance_forecast_rmse": "reused original result; pre-update forecasts are not in this cache",
    }
