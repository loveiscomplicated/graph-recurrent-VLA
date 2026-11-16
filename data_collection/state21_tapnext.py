"""Verified Object21D adapter for the saved TAPNext++ surface replay."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256, read_frame
from data_collection.state21 import Episode, Sample, audit_splits, is_sha256
from graph.state_estimation.codec import object21_from_gt
from graph.state_estimation.graph import Frame, Node, Source


def _inside(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Expected a relative path inside the dataset: {relative}")
    return path


def load_tapnext_surface_manifest(path, *, source_manifest=None, camera="base_camera"):
    """Restore ordered observations from TAPNext++ and verify targets against raw RGB-D records."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    tracker_hash = manifest.get("tracker_sha256")
    if not is_sha256(tracker_hash):
        raise ValueError("Expected a TAPNext++ surface manifest with tracker_sha256")
    source_path = Path(source_manifest or manifest["source"])
    source_hash = file_sha256(source_path)
    if source_hash != manifest["source_sha256"]:
        raise ValueError(
            "Raw trajectory manifest SHA-256 differs from the TAPNext++ source"
        )
    raw = json.loads(source_path.read_text())
    if raw.get("status") != "COMPLETE":
        raise ValueError("TAPNext++ replay requires a COMPLETE raw trajectory manifest")
    raw_episodes = {row["path"]: row for row in raw["episodes"]}
    if len(raw_episodes) != len(raw["episodes"]):
        raise ValueError("Duplicate paths in raw trajectory manifest")
    episodes = []
    for row in manifest["episodes"]:
        episode_id = row["episode"]
        if episode_id not in raw_episodes:
            raise ValueError(
                f"TAPNext++ episode {episode_id} is not listed in the raw manifest"
            )
        source = raw_episodes[episode_id]
        if (
            row["seed"] != source["seed"]
            or row["source_frames"] != source["observations"]
        ):
            raise ValueError(f"TAPNext++ and raw source disagree for {episode_id}")
        cache = _inside(path.parent, row["file"])
        if file_sha256(cache) != row["sha256"]:
            raise ValueError(f"TAPNext++ cache checksum mismatch: {cache}")
        directory = _inside(source_path.parent, episode_id)
        samples = _load_episode(cache, row, source, directory, camera)
        # Each policy derived from the same pilot reset seed shares the parent.
        episodes.append(
            Episode(
                episode_id,
                f"reset_seed:{source['seed']}",
                row["split"],
                samples,
                source.get("initial_fingerprint"),
            )
        )
    report = audit_splits(episodes)
    provenance = {
        "tracker": "TAPNext++",
        "tracker_sha256": tracker_hash,
        "source_manifest_sha256": source_hash,
        "tracker_training_or_selection_parents": None,
        "surface": "TAPNext++ query accepted by visibility, object mask and depth checks",
        "vertices": "RGB-D detector observations with unverified semantic IDs",
        "confidence": "unavailable",
    }
    return episodes, {
        "manifest_sha256": file_sha256(path),
        "provenance": provenance,
        **report,
    }


def _load_episode(cache, row, source, directory, camera):
    with np.load(cache, allow_pickle=False) as archive:
        frames = archive["frame_indices"]
        offsets = archive["node_offsets"]
        positions = archive["xyz_world"]
        kinds, corners, track_ids = (
            archive[name] for name in ("object_type", "is_corner", "track_id")
        )
        tcp, targets = archive["tcp_xyz_world"], archive["object21"]
    count = row["graphs"]
    if frames.shape != (count,) or frames.dtype.kind not in "iu":
        raise ValueError(
            "TAPNext++ frame_indices must be integer and match the graph count"
        )
    if (
        offsets.shape != (count + 1,)
        or offsets.dtype.kind not in "iu"
        or offsets[0] != 0
    ):
        raise ValueError("Invalid ragged node offsets in TAPNext++ cache")
    if np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(positions):
        raise ValueError("TAPNext++ offsets do not match saved node rows")
    if (
        positions.shape != (len(kinds), 3)
        or corners.shape != kinds.shape
        or track_ids.shape != kinds.shape
    ):
        raise ValueError("TAPNext++ node arrays must share the same rows")
    if tcp.shape != (count, 3) or targets.shape != (count, 21):
        raise ValueError("TAPNext++ frames need one TCP and Object21D target")
    if (
        kinds.dtype.kind not in "iu"
        or corners.dtype.kind != "b"
        or track_ids.dtype.kind not in "iu"
    ):
        raise ValueError(
            "TAPNext++ requires integer object/query IDs and a boolean corner mask"
        )
    if np.any(track_ids[corners] != -1) or np.any(track_ids[~corners] < 0):
        raise ValueError("Expected tracker query IDs for TAPNext++ surface rows")

    samples = []
    for row_index, frame_index in enumerate(frames.tolist()):
        if not 0 <= frame_index < len(source["frame_sha256"]):
            raise ValueError(
                f"TAPNext++ frame index outside source trajectory: {frame_index}"
            )
        raw_frame = directory / f"frame_{frame_index:06d}.npz"
        digest = source["frame_sha256"][frame_index]
        if file_sha256(raw_frame) != digest:
            raise ValueError(f"Raw frame checksum mismatch: {raw_frame}")
        _, gt = read_frame(raw_frame, camera)
        target = object21_from_gt(gt)[0]
        with np.load(raw_frame, allow_pickle=False) as raw:
            observed_tcp = raw["observation/extra/tcp_pose"][0, :3]
        if not (
            np.allclose(observed_tcp, tcp[row_index], rtol=0, atol=1e-6)
            and np.allclose(observed_tcp, gt["ee_pose"][0, :3], rtol=0, atol=1e-6)
            and torch.allclose(
                target, torch.as_tensor(targets[row_index]), rtol=0, atol=1e-6
            )
        ):
            raise ValueError(f"TCP or Object21D target is not aligned with {raw_frame}")
        selected = slice(offsets[row_index], offsets[row_index + 1])
        # Saved surface rows already passed TAPNext++ visibility and current-depth checks.
        nodes = tuple(
            Node(tuple(xyz), int(kind), bool(corner), None, True, None, Source.OBSERVED)
            for xyz, kind, corner in zip(
                positions[selected], kinds[selected], corners[selected]
            )
        )
        observation = Frame(row["episode"], frame_index, tuple(observed_tcp), nodes)
        samples.append(Sample(observation, frame_index, target, digest))
    return tuple(samples)
