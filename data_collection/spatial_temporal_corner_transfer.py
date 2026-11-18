"""Transfer supervision while reusing checksum-identical remote observations.

The GPU host already has the Five-mode release. Only targets and offline
certificates need uploading; detector/surface coordinates are reconstructed from
that pinned release. Reconstructed episode bytes must match the local SHA-256.
"""

from contextlib import ExitStack
import json
from pathlib import Path
import shutil

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.spatial_temporal_corner_tracking import (
    _open_release_archives,
    _reject_forbidden_cache_reference,
    _validate_cache_tables,
    _write_episode,
)
from training.gnn.packed_five_mode_data import read_archive_member
from utils.experiment_logger import write


def export_labels(config):
    """Export only reference corners and certificates from verified episodes."""
    return _export_labels(config, allowed_splits={"train", "validation"})


def export_test_labels(config):
    """Export explicitly prepared test supervision, including Object21 labels."""
    return _export_labels(config, allowed_splits={"test"})


def _export_labels(config, *, allowed_splits):
    """Keep split authorization explicit while sharing the transfer format."""
    manifest_path = Path(config["data"]["manifest"]).resolve()
    _reject_forbidden_cache_reference(manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "spatial_temporal_corner_tracking_v1":
        raise ValueError("Label export requires a prepared spatial-temporal dataset")
    output = Path(config["output"]["root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for number, row in enumerate(manifest["episodes"], 1):
        if row["split"] not in allowed_splits:
            raise ValueError(f"Supervision transfer accepts only {sorted(allowed_splits)}")
        path = (root / row["file"]).resolve()
        _reject_forbidden_cache_reference(path)
        if not path.is_relative_to(root) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Prepared transfer source checksum/path mismatch: {path}")
        relative = Path("labels") / Path(row["file"]).name
        destination = output / relative
        destination.parent.mkdir(exist_ok=True)
        with np.load(path, allow_pickle=False) as episode:
            # Leave labels uncompressed here; the transport archive compresses
            # across episodes without duplicating the remote observed geometry.
            labels = {name: episode[name] for name in ("targets", "vertex_ids")}
            if "object21" in episode.files:
                labels["object21"] = episode["object21"]
            np.savez(destination, **labels)
        receipts.append({
            "episode": row["episode"],
            "file": str(relative),
            "sha256": file_sha256(destination),
        })
        if number % 500 == 0:
            print(f"exported labels={number}/{len(manifest['episodes'])}", flush=True)
    shutil.copyfile(manifest_path, output / "prepared_manifest.json")
    write(output / "transfer_manifest.json", {
        "prepared_manifest_sha256": file_sha256(manifest_path),
        "labels": receipts,
    })


def reconstruct_episode(tables, labels):
    """Rebuild the exact prepared array order using unchanged cached observations."""
    corners, surface = tables["corners"], tables["surface"]
    frames = len(tables["targets"]["frame_indices"])
    xyz, kinds, is_corner = [], [], []
    surface_kinds = surface.get("surface_object_id")
    if surface_kinds is None:
        surface_kinds = np.zeros(len(surface["surface_world_xyz"]), dtype=np.int8)
    for frame in range(frames):
        corner_slice = slice(*corners["corner_offsets"][frame : frame + 2])
        surface_slice = slice(*surface["surface_offsets"][frame : frame + 2])
        detector_xyz = corners["corner_world_xyz"][corner_slice]
        surface_xyz = surface["surface_world_xyz"][surface_slice]
        xyz.append(np.concatenate((detector_xyz, surface_xyz)))
        kinds.append(np.concatenate((
            corners["corner_object_id"][corner_slice], surface_kinds[surface_slice],
        )))
        is_corner.append(np.concatenate((
            np.ones(len(detector_xyz), bool), np.zeros(len(surface_xyz), bool),
        )))
    arrays = {
        "offsets": (corners["corner_offsets"] + surface["surface_offsets"]).astype(np.int64),
        "xyz": np.concatenate(xyz).astype(np.float32),
        "kinds": np.concatenate(kinds).astype(np.int8),
        "vertex_ids": labels["vertex_ids"],
        "is_corner": np.concatenate(is_corner),
        "tcp": tables["targets"]["tcp"].astype(np.float32),
        "targets": labels["targets"],
    }
    if "object21" in labels:
        arrays["object21"] = labels["object21"]
    return arrays


def restore(config):
    """Restore byte-identical train/validation episodes from a supervision packet."""
    return _restore(config, allowed_splits={"train", "validation"})


def restore_test(config):
    """Restore explicitly requested test episodes without altering training data."""
    return _restore(config, allowed_splits={"test"})


def _restore(config, *, allowed_splits):
    """Restore only allowed episodes and require their original byte hashes."""
    packet = Path(config["data"]["packet"]).resolve()
    _reject_forbidden_cache_reference(packet)
    manifest_path = packet / "prepared_manifest.json"
    transfer = json.loads((packet / "transfer_manifest.json").read_text())
    if file_sha256(manifest_path) != transfer["prepared_manifest_sha256"]:
        raise ValueError("Transferred preparation manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "spatial_temporal_corner_tracking_v1":
        raise ValueError("Label restoration requires a prepared spatial-temporal dataset")
    release_path = Path(config["data"]["release_manifest"]).resolve()
    _reject_forbidden_cache_reference(release_path)
    if file_sha256(release_path) != manifest["source"]["manifest_sha256"]:
        raise ValueError("Remote release differs from the local prepared dataset")
    release = json.loads(release_path.read_text())
    records = {(row["seed"], row["policy_mode"]): row for row in release["records"]}
    label_receipts = {row["episode"]: row for row in transfer["labels"]}
    output = Path(config["output"]["root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        archives = _open_release_archives(stack, release_path.parent, release)
        for number, row in enumerate(manifest["episodes"], 1):
            if row["split"] not in allowed_splits:
                raise ValueError(f"Supervision restoration accepts only {sorted(allowed_splits)}")
            destination = (output / row["file"]).resolve()
            _reject_forbidden_cache_reference(destination)
            if not destination.is_relative_to(output):
                raise ValueError("Prepared destination must remain within output.root")
            if not destination.exists() or file_sha256(destination) != row["sha256"]:
                receipt = label_receipts[row["episode"]]
                label_path = (packet / receipt["file"]).resolve()
                _reject_forbidden_cache_reference(label_path)
                valid_labels = (
                    label_path.is_relative_to(packet)
                    and file_sha256(label_path) == receipt["sha256"]
                )
                if not valid_labels:
                    raise ValueError(f"Transferred supervision checksum/path mismatch: {label_path}")
                record = records[row["seed"], row["policy_mode"]]
                tables = {}
                for kind in ("corners", "surface", "targets"):
                    source = record["files"][kind]
                    tables[kind] = read_archive_member(archives[source["archive"]], source, kind)
                _validate_cache_tables(record, tables)
                with np.load(label_path, allow_pickle=False) as labels:
                    arrays = reconstruct_episode(tables, labels)
                _write_episode(destination, arrays)
                if file_sha256(destination) != row["sha256"]:
                    raise ValueError(
                        f"Restored episode bytes differ from local preparation: {destination}"
                    )
            if number == 1 or number % 100 == 0 or number == len(manifest["episodes"]):
                print(f"restored episodes={number}/{len(manifest['episodes'])}", flush=True)
    shutil.copyfile(manifest_path, output / "manifest.json")
