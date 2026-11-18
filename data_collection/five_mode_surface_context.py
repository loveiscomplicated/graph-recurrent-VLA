"""Prepare aligned Peg surface context without changing the corner-only dataset."""

import json
import tarfile
from contextlib import ExitStack
from itertools import pairwise
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_corner_tracking import read_member, sequence_arrays
from utils.experiment_logger import write


def prepare(config):
    """Prepare a verified development sidecar or an explicitly requested test sidecar."""
    release = Path(config["release"])
    source = release / "manifest.json"
    if file_sha256(source) != config["manifest_sha256"]:
        raise ValueError("Surface source release manifest checksum differs")
    manifest = json.loads(source.read_text())
    base = Path(config["base_dataset"])
    base_manifest = json.loads((base / "manifest.json").read_text())
    splits = tuple(config.get("splits", ("train", "validation")))
    if splits not in (("train", "validation"), ("test",)):
        raise ValueError(
            "Surface preparation splits must be train/validation or test only"
        )
    if any(row["split"] not in splits for row in base_manifest["records"]):
        raise ValueError(
            "Base corner manifest contains a split outside the requested surface sidecar"
        )
    if (
        config.get("base_manifest_sha256")
        and file_sha256(base / "manifest.json") != config["base_manifest_sha256"]
    ):
        raise ValueError(
            "Base corner manifest checksum differs from the frozen evaluation input"
        )
    if base_manifest["source_manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError(
            "Corner and surface datasets must share the same source release"
        )
    inventory = {(row["seed"], row["policy_mode"]): row for row in manifest["records"]}
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for split in splits:
        (output / split).mkdir()
    with ExitStack() as stack:
        archives = {}
        for index, episode in enumerate(base_manifest["records"], 1):
            source_record = inventory[episode["seed"], episode["mode"]]
            if (
                source_record["split"] != episode["split"]
                or source_record["frames"] != episode["frames"]
            ):
                raise ValueError(
                    f"Surface/corner episode identity mismatch: {episode['path']}"
                )
            receipt = source_record["files"]["surface"]
            archive_name = receipt["archive"]
            if archive_name not in archives:
                archives[archive_name] = stack.enter_context(
                    tarfile.open(release / archive_name)
                )
            arrays = read_member(archives[archive_name], receipt)
            length = episode["frames"]
            if not np.array_equal(arrays["frame_indices"], np.arange(length)):
                raise ValueError(f"Surface frame order differs: {receipt['member']}")
            offsets = arrays["surface_offsets"]
            xyz, ids = arrays["surface_world_xyz"], arrays["surface_track_id"]
            if (
                offsets.shape != (length + 1,)
                or offsets[0] != 0
                or offsets[-1] != len(ids)
                or np.any(np.diff(offsets) < 0)
                or xyz.shape != (len(ids), 3)
                or not np.isfinite(xyz).all()
            ):
                raise ValueError(
                    f"Invalid surface coordinates or offsets: {receipt['member']}"
                )
            count = config["maximum_points"]
            if len(arrays["surface_query_xy"]) > count or np.any(
                (ids < 0) | (ids >= count)
            ):
                raise ValueError(f"Surface query IDs exceed configured maximum {count}")
            if "surface_object_id" in arrays and np.any(
                arrays["surface_object_id"] != 0
            ):
                raise ValueError("This experiment accepts Peg-only surface context")
            dense = np.zeros((length, count, 3), dtype=np.float32)
            valid = np.zeros((length, count), dtype=bool)
            for frame, (start, stop) in enumerate(pairwise(offsets)):
                frame_ids = ids[start:stop]
                if len(np.unique(frame_ids)) != len(frame_ids):
                    raise ValueError("Duplicate surface track ID within one frame")
                dense[frame, frame_ids] = xyz[start:stop]
                valid[frame, frame_ids] = True
            destination = output / episode["path"]
            np.savez_compressed(destination, xyz=dense, valid=valid)
            records.append(
                {
                    "seed": episode["seed"],
                    "mode": episode["mode"],
                    "split": episode["split"],
                    "frames": length,
                    "path": episode["path"],
                    "sha256": file_sha256(destination),
                    "source_surface_sha256": receipt["sha256"],
                    "valid_points": int(valid.sum()),
                }
            )
            if index % 250 == 0:
                print(
                    f"Prepared surface context {index}/{len(base_manifest['records'])}",
                    flush=True,
                )
    write(
        output / "manifest.json",
        {
            "format": "five_mode_peg_surface_context_v1",
            "records": records,
            "base_manifest_sha256": file_sha256(base / "manifest.json"),
            "source_manifest_sha256": config["manifest_sha256"],
            "maximum_points": config["maximum_points"],
            "object_identity": "Peg: pinned no-plan producer provenance; IDs absent in original cache",
            "test_payloads_read": "test" in splits,
            "config": config,
        },
    )


def attach_surface_context(groups, root, base_root):
    """Validate sidecar identity and attach finite current-frame surface points."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != "five_mode_peg_surface_context_v1":
        raise ValueError("Expected a prepared Peg-only surface context dataset")
    if manifest["base_manifest_sha256"] != file_sha256(
        Path(base_root) / "manifest.json"
    ):
        raise ValueError(
            "Surface sidecar was prepared against a different corner dataset"
        )
    records = {(row["seed"], row["mode"]): row for row in manifest["records"]}
    for split, episodes in groups.items():
        for episode in episodes:
            record = records.pop((episode["seed"], episode["mode"]))
            if record["split"] != split or record["frames"] != len(episode["gt"]):
                raise ValueError(
                    "Surface episode does not match the corner split/length"
                )
            path = root / record["path"]
            if file_sha256(path) != record["sha256"]:
                raise ValueError(f"Surface sidecar checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as source:
                xyz, valid = source["xyz"], source["valid"]
            if (
                xyz.shape != (record["frames"], manifest["maximum_points"], 3)
                or valid.shape != xyz.shape[:2]
                or valid.dtype != bool
                or not np.isfinite(xyz).all()
            ):
                raise ValueError(f"Invalid surface sidecar arrays: {path}")
            episode.update(surface_xyz=xyz, surface_valid=valid)
    if records:
        raise ValueError("Surface sidecar contains unmatched episodes")
    return file_sha256(root / "manifest.json")


def surface_sequence_arrays(episodes, fallback):
    """Add only current XYZ features; no surface history, prediction or GT input."""
    arrays, lengths, modes = sequence_arrays(episodes, fallback)
    count = episodes[0]["surface_xyz"].shape[1]
    shape = (len(episodes), int(lengths.max()), count)
    arrays["surface_features"] = np.zeros((*shape, 6), dtype=np.float32)
    arrays["surface_valid"] = np.zeros(shape, dtype=bool)
    center = fallback[:8].mean(axis=0)
    for index, episode in enumerate(episodes):
        xyz, valid = episode["surface_xyz"], episode["surface_valid"]
        length = len(xyz)
        features = np.concatenate(
            ((xyz - center) / 0.1, (episode["tcp"][:, None] - xyz) / 0.1), axis=-1
        )
        arrays["surface_features"][index, :length] = np.where(
            valid[..., None], features, 0
        )
        arrays["surface_valid"][index, :length] = valid
    return arrays, lengths, modes
