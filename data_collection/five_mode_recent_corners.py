"""Build causal recent-corner examples from Five-mode detector and target members."""

from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from data_collection.detected_corner_visibility import CONDITIONS, classify_episode
from data_collection.episode_dataset import canonical_hash, file_sha256
from graph.temporal.detected_corner_tracking import match_positions
from training.gnn.packed_five_mode_data import indexed_frames, pack_episode, read_archive_member


@dataclass
class RecentCorner:
    """Keep the last two real observations of one detector track."""

    kind: int
    anchor: np.ndarray
    frame: int
    velocity: np.ndarray
    hits: int = 1

    def update(self, point, frame):
        self.velocity = (point - self.anchor) / (frame - self.frame)
        self.anchor = point
        self.frame = frame
        self.hits += 1


def episode_candidates(corners, tcp, *, max_age, max_candidates, match_m, duplicate_m):
    """Propose missing tracks in most-recent-first order using detector observations only."""
    tracks = []
    offsets = corners["corner_offsets"]
    world = corners["corner_world_xyz"]
    families = corner_families(corners)
    features = np.zeros((len(tcp), max_candidates, 14), np.float32)
    anchors = np.zeros((len(tcp), max_candidates, 3), np.float32)
    anchor_frames = np.full((len(tcp), max_candidates), -1, np.int32)
    kinds = np.full_like(anchor_frames, -1)
    for frame, (left, right) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        xyz, current_kinds = world[left:right], families[left:right]
        tracks = [track for track in tracks if frame - track.frame <= max_age]
        track_kinds = np.asarray([track.kind for track in tracks])
        forecast = np.asarray([track.anchor + min(frame - track.frame, 2)
                               * np.clip(track.velocity, -0.012, 0.012) for track in tracks])
        matches = match_positions(forecast, track_kinds, xyz, current_kinds, match_m)
        matched_tracks = {track_id for track_id, _ in matches}
        matched_points = {point_id for _, point_id in matches}
        motions = {kind: [] for kind in range(3)}
        for track_id, point_id in matches:
            track = tracks[track_id]
            track.update(xyz[point_id], frame)
            motions[track.kind].append(track.velocity)
        median_motion = {kind: np.median(values, axis=0) if values else np.zeros(3)
                         for kind, values in motions.items()}

        missing = [idx for idx in range(len(tracks)) if idx not in matched_tracks]
        missing.sort(key=lambda idx: (frame - tracks[idx].frame, -tracks[idx].hits, idx))
        selected = 0
        for idx in missing:
            track = tracks[idx]
            same_kind = xyz[current_kinds == track.kind]
            if np.any(np.linalg.norm(same_kind - forecast[idx], axis=1) <= duplicate_m):
                continue
            age = frame - track.frame
            anchors[frame, selected] = track.anchor
            anchor_frames[frame, selected] = track.frame
            kinds[frame, selected] = track.kind
            motion = np.clip(median_motion[track.kind], -0.02, 0.02)
            features[frame, selected] = np.concatenate((
                (track.anchor - tcp[frame]) / 0.15,
                np.clip(track.velocity, -0.02, 0.02) / 0.02,
                (tcp[frame] - tcp[track.frame]) / 0.05,
                motion / 0.02,
                [age / max_age, min(track.hits, 8) / 8],
            ))
            selected += 1
            if selected == max_candidates:
                break
        for idx, point in enumerate(xyz):
            if idx not in matched_points:
                tracks.append(RecentCorner(int(current_kinds[idx]), point, frame, np.zeros(3)))
    return features, anchors, anchor_frames, kinds


def corner_families(corners):
    """Use detector tags 0=Peg, 1=Box exterior, and 2=Hole rim."""
    kinds = corners["corner_object_id"]
    return np.where((kinds == 1) & corners["corner_contour_hole"], 2, kinds)


def transport_labels(anchors, anchor_frames, kinds, tcp, object21, frames):
    """Move each detector anchor with its object's measured GT rigid transform."""
    targets = np.zeros_like(anchors)
    for mask, center, rotation in ((kinds == 0, slice(0, 3), slice(3, 7)),
                                   (kinds > 0, slice(10, 13), slice(13, 17))):
        if not mask.any():
            continue
        last = anchor_frames[mask]
        current = np.broadcast_to(frames[:, None], kinds.shape)[mask]
        centers = object21[:, center] + tcp
        rotations = Rotation.from_quat(object21[:, rotation], scalar_first=True)
        local = rotations[last].inv().apply(anchors[mask] - centers[last])
        targets[mask] = rotations[current].apply(local) + centers[current]
    return targets


def cohort_frames(manifest, index, split, settings):
    """Keep all Condition 2 frames and a frozen sample of Condition 1 for training."""
    records = {(row["seed"], row["policy_mode"]): row["split"] for row in manifest["records"]}
    if len(records) != manifest["episodes"]:
        raise ValueError("Five-mode release has duplicate episode identities")
    selections = [indexed_frames(index, condition, (split,), records) for condition in CONDITIONS]
    if split != "train":
        return selections
    count = sum(len(frames) for frames in selections[1].values())
    if count != settings["condition_2_train_frames"]:
        raise ValueError(f"Unexpected Condition 2 training frame count: {count}")
    primary = [(key, frame) for key, frames in selections[0].items() for frame in frames]
    random = np.random.default_rng(settings["sampling_seed"])
    sampled = random.choice(len(primary), size=settings["condition_1_train_frames"], replace=False)
    keep = {key: [] for key in selections[0]}
    for item in sampled:
        key, frame = primary[item]
        keep[key].append(frame)
    return keep, selections[1], {}


def read_episode_tables(stack, archives, manifest_path, shard_names, episode):
    """Decode only checksum-pinned detector corners and Object21 targets."""
    tables = {}
    for kind in ("corners", "targets"):
        receipt = episode["files"][kind]
        name = receipt["archive"]
        path = (manifest_path.parent / name).resolve()
        if name not in shard_names or not path.is_relative_to(manifest_path.parent) or path.suffix != ".tar":
            raise ValueError(f"Unexpected Five-mode supervised shard: {path}")
        if path not in archives:
            archives[path] = stack.enter_context(tarfile.open(path, "r:"))
        tables[kind] = read_archive_member(archives[path], receipt, kind)
    return tables


def load_cohort(config, split):
    """Pack detector histories with supervision and a stable per-frame identity."""
    settings = config["data"]
    path = verify_input(settings["manifest"])
    manifest = json.loads(path.read_text())
    index = json.loads(verify_input(settings["condition_index"]).read_text())
    if manifest["format"] != "five_mode_supervised_v1" or split not in ("train", "validation", "test"):
        raise ValueError(f"Invalid Five-mode release or split: {split}")
    subsets = cohort_frames(manifest, index, split, settings)
    shard_names = {row["file"] for row in manifest["archives"]}
    data, ids = [], []
    motion_targets = []
    with ExitStack() as stack:
        archives = {}
        for index, episode in enumerate(manifest["records"]):
            if episode["split"] != split:
                continue
            key = episode["seed"], episode["policy_mode"]
            by_condition = {frame: label for label, rows in enumerate(subsets, start=1)
                            for frame in rows.get(key, ())}
            if not by_condition:
                continue
            frames = np.asarray(sorted(by_condition))
            tables = read_episode_tables(stack, archives, path, shard_names, episode)
            if split == "test":
                groups = classify_episode(tables["corners"], episode["frames"],
                                          tolerance_m=settings["coplanarity_tolerance_m"])
                if any(groups[name] != selection[key] for name, selection in zip(CONDITIONS, subsets, strict=True)):
                    raise ValueError(f"Detector visibility does not match the test index for {key}")
            xyz, node_kinds, tcp, target = pack_episode(episode, tables, include_surface=False)
            observed_families = node_kinds.copy()
            offsets = tables["corners"]["corner_offsets"]
            families = corner_families(tables["corners"])
            for frame in frames:
                start, stop = offsets[frame:frame + 2]
                observed_families[frame, :stop - start] = families[start:stop]
            features, anchors, anchor_frames, kinds = episode_candidates(
                tables["corners"], tcp, **config["association"],
            )
            labels = np.asarray([by_condition[frame] for frame in frames])
            if split == "train":
                eligible = (kinds[frames] >= 0).any(axis=1)
                frames, labels = frames[eligible], labels[eligible]
                motion_targets.append(transport_labels(anchors[frames], anchor_frames[frames],
                                                       kinds[frames], tcp, target, frames))
            anchors, kinds = anchors[frames], kinds[frames]
            data.append((xyz[frames], node_kinds[frames], tcp[frames], target[frames], labels,
                         features[frames], anchors, kinds, observed_families[frames]))
            ids.extend(f"{key[0]}:{key[1]}:{frame}" for frame in frames)
            if len(data) % 450 == 0:
                print(f"{split}: prepared {len(data):,} episodes at release row {index + 1:,}", flush=True)

    if len(set(ids)) != len(ids) or not ids:
        raise ValueError(f"Empty or duplicated {split} frame IDs")
    width = max(row[0].shape[1] for row in data)
    xyz = np.concatenate([np.pad(row[0], ((0, 0), (0, width - row[0].shape[1]), (0, 0))) for row in data])
    node_kinds = np.concatenate([np.pad(row[1], ((0, 0), (0, width - row[1].shape[1])),
                                        constant_values=-1) for row in data])
    names = ("tcp", "target", "condition", "features", "anchors", "candidate_kinds")
    columns = list(zip(*data, strict=True))
    arrays = dict(zip(names, (np.concatenate(rows) for rows in columns[2:-1]), strict=True))
    observed_families = np.concatenate([
        np.pad(values, ((0, 0), (0, width - values.shape[1])), constant_values=-1)
        for values in columns[-1]
    ])
    arrays.update(xyz=xyz, node_kinds=node_kinds, observed_families=observed_families)
    if split == "train":
        arrays["motion_targets"] = np.concatenate(motion_targets)
    elif canonical_hash(ids) != settings[f"{split}_frame_ids_sha256"]:
        raise ValueError(f"Unexpected {split} frame IDs or order")
    print(f"{split}: {len(ids):,} frames by condition {np.bincount(arrays['condition'], minlength=4)[1:].tolist()}",
          flush=True)
    return {name: torch.from_numpy(array) for name, array in arrays.items()}, ids


def verify_input(spec):
    """Check the pinned supervised release before accessing its selected members."""
    path = Path(spec["path"]).resolve()
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"Unexpected input checksum: {path}")
    return path
