"""Audit prepared chronological episodes against current Five-mode detections."""

from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.spatial_corner_tracking import VERTEX_ID, VERTICES
from data_collection.spatial_temporal_corner_tracking import (
    _open_release_archives,
    _reject_forbidden_cache_reference,
    _validate_cache_tables,
)
from data_collection.state21_vertex_targets import match_vertices
from training.gnn.packed_five_mode_data import read_archive_member
from training.gnn.spatial_temporal_corner_data import _validate_tracker_arrays
from utils.experiment_logger import ExperimentLogger, write


def audit_episode(arrays, record, tables, matching):
    """Check every frame's original rows, chronological offsets and certificates."""
    episode_id = f"five_mode_{record['seed']}_{record['policy_mode']}"
    _validate_tracker_arrays(arrays, episode_id)
    _validate_cache_tables(record, tables)
    corners, surface = tables["corners"], tables["surface"]
    expected_offsets = corners["corner_offsets"] + surface["surface_offsets"]
    np.testing.assert_array_equal(arrays["offsets"], expected_offsets)
    np.testing.assert_array_equal(arrays["tcp"], tables["targets"]["tcp"])
    surface_kinds = surface.get("surface_object_id")
    if surface_kinds is None:
        surface_kinds = np.zeros(len(surface["surface_world_xyz"]), dtype=np.int8)

    certified = 0
    for frame_index in range(record["frames"]):
        first, last = arrays["offsets"][frame_index : frame_index + 2]
        corner_slice = slice(*corners["corner_offsets"][frame_index : frame_index + 2])
        surface_slice = slice(*surface["surface_offsets"][frame_index : frame_index + 2])
        detector_xyz = corners["corner_world_xyz"][corner_slice]
        detector_kinds = corners["corner_object_id"][corner_slice]
        surface_xyz = surface["surface_world_xyz"][surface_slice]
        expected_xyz = np.concatenate((detector_xyz, surface_xyz)).astype(np.float32)
        expected_kinds = np.concatenate((detector_kinds, surface_kinds[surface_slice]))
        expected_corner = np.arange(last - first) < len(detector_xyz)
        np.testing.assert_array_equal(arrays["xyz"][first:last], expected_xyz)
        np.testing.assert_array_equal(arrays["kinds"][first:last], expected_kinds)
        np.testing.assert_array_equal(arrays["is_corner"][first:last], expected_corner)

        observed = {
            (int(kind), index): xyz
            for index, (kind, xyz) in enumerate(zip(detector_kinds, detector_xyz, strict=True))
        }
        reference = dict(zip(VERTICES, arrays["targets"][frame_index], strict=True))
        matches = match_vertices(observed, reference, **matching)
        expected_ids = np.full(last - first, -1, dtype=np.int16)
        for detection, vertex in matches.items():
            expected_ids[detection[1]] = VERTEX_ID[vertex]
        np.testing.assert_array_equal(arrays["vertex_ids"][first:last], expected_ids)
        certified += len(matches)

    return {
        "frames": record["frames"],
        "detector_rows": len(corners["corner_world_xyz"]),
        "surface_rows": len(surface["surface_world_xyz"]),
        "certified_rows": certified,
    }


def verify(config):
    """Verify all train/validation episode hashes and current-frame lineage."""
    manifest_path = Path(config["data"]["manifest"]).resolve()
    _reject_forbidden_cache_reference(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "spatial_temporal_corner_tracking_v1":
        raise ValueError("Audit requires a spatial_temporal_corner_tracking_v1 manifest")
    if manifest.get("condition_filter") is not None:
        raise ValueError("Prepared data must not filter frames by visibility")
    source = manifest["source"]
    if not source.get("raw_archives_verified"):
        raise ValueError("Preparation must verify raw archive checksums")
    release_path = Path(config["data"].get("source_manifest", source["manifest"])).resolve()
    _reject_forbidden_cache_reference(release_path)
    if file_sha256(release_path) != source["manifest_sha256"]:
        raise ValueError("Source release manifest checksum differs from preparation")
    release = json.loads(release_path.read_text())
    records = {
        (row["seed"], row["policy_mode"]): row
        for row in release["records"]
        if row["split"] in ("train", "validation")
    }
    episodes = manifest["episodes"]
    keys = [(row["seed"], row["policy_mode"]) for row in episodes]
    if len(set(keys)) != len(keys) or set(keys) != set(records):
        raise ValueError("Prepared episodes must uniquely cover the full train/validation release")
    if len({row["episode"] for row in episodes}) != len(episodes):
        raise ValueError("Prepared episode IDs must be unique")
    seed_splits = {}
    totals = Counter()
    episode_counts, frame_counts, surface_modes = Counter(), Counter(), Counter()
    with ExitStack() as stack:
        archives = _open_release_archives(stack, release_path.parent, release)
        for number, row in enumerate(episodes, 1):
            record = records[row["seed"], row["policy_mode"]]
            if any(row[key] != record[key] for key in ("split", "frames", "batch")):
                raise ValueError(f"Prepared episode metadata differs from release: {row['episode']}")
            prior_split = seed_splits.setdefault(row["seed"], row["split"])
            if prior_split != row["split"]:
                raise ValueError(f"Seed occurs in multiple splits: {row['seed']}")
            _reject_forbidden_cache_reference(row["file"])
            episode_path = (root / row["file"]).resolve()
            if not episode_path.is_relative_to(root) or file_sha256(episode_path) != row["sha256"]:
                raise ValueError(f"Prepared episode path/checksum mismatch: {row['episode']}")
            with np.load(episode_path, allow_pickle=False) as archive:
                arrays = dict(archive)
            tables = {}
            for kind in ("corners", "surface", "targets"):
                receipt = record["files"][kind]
                tables[kind] = read_archive_member(archives[receipt["archive"]], receipt, kind)
            totals.update(audit_episode(arrays, record, tables, manifest["matching"]))
            episode_counts[row["split"]] += 1
            frame_counts[row["split"]] += row["frames"]
            surface_modes[row["surface_identity_mode"]] += 1
            if number == 1 or number % 100 == 0 or number == len(episodes):
                print(f"audited episodes={number}/{len(episodes)} frames={totals['frames']}", flush=True)

    expected_digest = canonical_hash({
        "source_manifest_sha256": source["manifest_sha256"],
        "episode_sha256": [row["sha256"] for row in episodes],
        "matching": manifest["matching"],
        "condition_filter": None,
    })
    if expected_digest != manifest["dataset_sha256"]:
        raise ValueError("Prepared dataset identity hash differs")
    for name, actual in (("episodes_by_split", episode_counts), ("frames_by_split", frame_counts)):
        if dict(actual) != manifest[name]:
            raise ValueError(f"Prepared {name} differs from verified inventory")
    if dict(surface_modes) != manifest["surface_identity_modes"]:
        raise ValueError("Prepared surface identity mode inventory differs")

    result = {
        "status": "passed",
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": expected_digest,
        "episodes_by_split": dict(episode_counts),
        "frames_by_split": dict(frame_counts),
        "surface_identity_modes": dict(surface_modes),
        "condition_filter": None,
        "test_episode_files_opened": 0,
        "gt_role": "offline_identity_and_coordinate_supervision_only",
        "raw_pose_verification": "checksum-verified streaming preparation; not repeated by audit",
        **totals,
    }
    with ExperimentLogger(team="graph", config=config, job_type="dataset_verification") as logger:
        destination = Path(config["output"]["result"])
        write(destination, result)
        logger.link_output(destination)
        logger.record(**result)
    return result
