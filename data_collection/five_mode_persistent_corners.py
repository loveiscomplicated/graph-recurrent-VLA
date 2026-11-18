"""Prepare persistent detector tracks from the pinned Five-mode release."""

from contextlib import ExitStack
import json

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from data_collection.detected_corner_visibility import CONDITIONS
from data_collection.episode_dataset import canonical_hash
from data_collection.five_mode_recent_corners import (
    corner_families, read_episode_tables, transport_labels, verify_input,
)
from graph.temporal.detected_corner_tracking import match_positions
from training.gnn.packed_five_mode_data import indexed_frames


def associate(anchors, carried, kinds, points, detected, match_m):
    """Match Box points in world space and test two motion hypotheses for Peg."""
    matches = []
    for family in range(3):
        tracks = np.flatnonzero(kinds == family)
        detections = np.flatnonzero(detected == family)
        if not len(tracks) or not len(detections):
            continue
        forecasts = (anchors, carried) if family == 0 else (anchors,)
        best, best_score = [], (0, 0.0)
        for forecast in forecasts:
            pairs = match_positions(forecast[tracks], kinds[tracks], points[detections],
                                    detected[detections], match_m)
            pairs = [(tracks[left], detections[right]) for left, right in pairs]
            score = (len(pairs), -sum(np.linalg.norm(forecast[i] - points[j]) for i, j in pairs))
            if score > best_score:
                best, best_score = pairs, score
        matches.extend(best)
    return matches


def episode_tracks(corners, robot, *, match_m):
    """Emit every track after birth, using only detections and robot kinematics."""
    xyz = corners["corner_world_xyz"]
    offsets = corners["corner_offsets"]
    detected = corner_families(corners)
    tcp = robot[:, :3]
    rotation = Rotation.from_quat(robot[:, 3:7], scalar_first=True)
    gripper = robot[:, 13:15]
    kinds = np.empty(len(xyz), dtype=np.int8)
    last = np.empty(len(xyz), dtype=np.int32)
    hits = np.zeros(len(xyz), dtype=np.int32)
    anchors = np.empty_like(xyz)
    tool_local = np.empty_like(xyz)
    velocity = np.zeros_like(xyz)
    size = 0
    rows = []
    for frame, (left, right) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        present = np.zeros(size + right - left, dtype=bool)
        carried = anchors[:size].copy()
        peg = kinds[:size] == 0
        if peg.any():
            carried[peg] = rotation[frame].apply(tool_local[:size][peg]) + tcp[frame]
        points, tags = xyz[left:right], detected[left:right]
        matches = associate(anchors[:size], carried, kinds[:size], points, tags, match_m)
        assigned = np.full(len(points), -1, dtype=np.int32)
        for track, detection in matches:
            velocity[track] = (points[detection] - anchors[track]) / (frame - last[track])
            assigned[detection] = track
        new = assigned < 0
        stop = size + np.count_nonzero(new)
        assigned[new] = np.arange(size, stop)
        size = stop
        present[assigned] = True
        kinds[assigned] = tags
        anchors[assigned] = points
        last[assigned] = frame
        hits[assigned] += 1
        if len(points):
            tool_local[assigned] = rotation[frame].inv().apply(points - tcp[frame])

        carried = anchors[:size].copy()
        peg = kinds[:size] == 0
        if peg.any():
            carried[peg] = rotation[frame].apply(tool_local[:size][peg]) + tcp[frame]
        age = frame - last[:size]
        features = np.column_stack((
            (anchors[:size] - tcp[frame]) / 0.15,
            (carried - anchors[:size]) / 0.15,
            np.clip(velocity[:size] / 0.02, -2, 2),
            np.log1p(age) / 4.0,
            hits[:size].clip(max=8) / 8.0,
            np.broadcast_to(gripper[frame] / 0.04, (size, 2)),
            (gripper[frame] - gripper[last[:size]]) / 0.04,
        )).astype(np.float32)
        rows.append((anchors[:size].copy(), kinds[:size].copy(), last[:size].copy(),
                     present[:size], features))
    return rows


def selected_frames(index, manifest, split, limits, seed):
    """Keep split membership and take a reproducible per-condition training sample."""
    episodes = {(row["seed"], row["policy_mode"]): row["split"] for row in manifest["records"]}
    selections = [indexed_frames(index, name, (split,), episodes) for name in CONDITIONS]
    random = np.random.default_rng(seed)
    by_episode = {}
    for label, frames in enumerate(selections, start=1):
        entries = [(key, frame) for key, numbers in frames.items() for frame in numbers]
        if split == "train":
            count = limits[label - 1]
            if not 0 <= count <= len(entries):
                raise ValueError(f"Invalid training budget for {CONDITIONS[label - 1]}: {count}")
            chosen = random.choice(len(entries), size=count, replace=False)
            entries = [entries[i] for i in chosen]
        for key, frame in entries:
            labels = by_episode.setdefault(key, {})
            if frame in labels:
                raise ValueError(f"Frame {key}:{frame} belongs to multiple conditions")
            labels[frame] = label
    return by_episode


def load_split(config, split):
    """Use only supervised corner and target members, retaining original frame IDs."""
    settings = config["data"]
    path = verify_input(settings["manifest"])
    manifest = json.loads(path.read_text())
    index = json.loads(verify_input(settings["condition_index"]).read_text())
    if manifest["format"] != "five_mode_supervised_v1" or split not in ("train", "validation"):
        raise ValueError(f"Expected a Five-mode training or validation split, got {split}")
    chosen = selected_frames(index, manifest, split, settings["train_frames"], config["seed"])
    shard_names = {row["file"] for row in manifest["archives"]}
    rows, ids = [], []
    with ExitStack() as stack:
        archives = {}
        for episode in manifest["records"]:
            key = episode["seed"], episode["policy_mode"]
            if key not in chosen:
                continue
            frames = np.asarray(sorted(chosen[key]))
            labels = np.asarray([chosen[key][frame] for frame in frames], dtype=np.int8)
            tables = read_episode_tables(stack, archives, path, shard_names, episode)
            arrays = pack_tracks(tables, frames, labels, **config["association"])
            rows.append(arrays)
            ids.extend(f"{key[0]}:{key[1]}:{frame}" for frame in frames)
            if len(rows) % 450 == 0:
                print(f"{split}: tracked {len(rows):,} episodes", flush=True)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicated {split} frame IDs")
    if split == "validation" and canonical_hash(ids) != settings["validation_frame_ids_sha256"]:
        raise ValueError("Validation frame IDs or order differ from the previous experiment")
    width = max(row["kinds"].shape[1] for row in rows)
    result = {}
    for name in rows[0]:
        columns = [row[name] for row in rows]
        if name in ("anchors", "kinds", "age", "observed", "features", "motion_targets"):
            pad_value = -1 if name == "kinds" else 0
            columns = [np.pad(col, [(0, 0), (0, width - col.shape[1])] + [(0, 0)] * (col.ndim - 2),
                              constant_values=pad_value) for col in columns]
        result[name] = torch.from_numpy(np.concatenate(columns))
    counts = torch.bincount(result["condition"], minlength=4).tolist()[1:]
    print(f"{split}: {len(ids):,} frames {counts}, maximum {width} tracks", flush=True)
    return result, ids


def pack_tracks(tables, frames, labels, *, match_m):
    """Pack a selected frame cohort, adding rigidly transported labels separately."""
    corners, targets = tables["corners"], tables["targets"]
    tcp, target, robot = (targets[name] for name in ("tcp", "object21", "robot15"))
    steps = len(tcp)
    if (not all(np.array_equal(table["frame_indices"], np.arange(steps)) for table in tables.values())
            or robot.shape != (steps, 15) or not np.isfinite(robot).all()
            or not np.allclose(tcp, robot[:, :3], atol=1e-6, rtol=0)):
        raise ValueError("Detector frames and robot kinematics are unaligned")
    tracks = episode_tracks(corners, robot, match_m=match_m)
    width = max(1, max(len(tracks[frame][1]) for frame in frames))
    anchors = np.zeros((len(frames), width, 3), dtype=np.float32)
    kinds = np.full((len(frames), width), -1, dtype=np.int8)
    last = np.zeros((len(frames), width), dtype=np.int32)
    observed = np.zeros((len(frames), width), dtype=bool)
    features = np.zeros((len(frames), width, tracks[0][-1].shape[-1]), dtype=np.float32)
    for row, frame in enumerate(frames):
        history = tracks[frame]
        span = slice(0, len(history[1]))
        for destination, value in zip((anchors, kinds, last, observed, features), history, strict=True):
            destination[row, span] = value
    supervision = transport_labels(anchors, last, kinds, tcp, target, frames)
    supervision[observed] = anchors[observed]
    return dict(tcp=tcp[frames], target=target[frames], condition=labels, anchors=anchors,
                kinds=kinds, age=(frames[:, None] - last).astype(np.int16), observed=observed,
                features=features, motion_targets=supervision)
