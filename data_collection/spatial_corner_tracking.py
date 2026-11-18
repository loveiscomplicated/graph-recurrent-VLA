"""Prepare complete corner-tracking episodes from direct corners and surface tracks."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.state21_vertex_targets import POSE_FIELDS, corners_from_pose, match_vertices
from data_collection.storage import write_json

VERTICES = tuple((kind, name, index) for kind, name in (
    (0, "peg"), (1, "box_outer"), (1, "hole_rim")
) for index in range(8))
VERTEX_ID = {key: index for index, key in enumerate(VERTICES)}


def pack_episode(raw, row, corners, surface, output, split, episode_id):
    """Store frame-aligned observations and separate GT corner coordinates."""
    count = row["observations"]
    offsets = [0]
    xyz, kinds, ids, is_corner, tcp, targets = [], [], [], [], [], []
    for index, expected_hash in enumerate(row["frame_sha256"]):
        path = raw / f"frame_{index:06d}.npz"
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Frame hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as frame:
            pose = np.concatenate([frame[f"ground_truth/{key}"][0].reshape(size) for key, size in POSE_FIELDS])
            tcp.append(frame["ground_truth/ee_pose"][0, :3])
        reference = corners_from_pose(pose)
        targets.append([reference[key] for key in VERTICES])
        corner_slice = slice(*corners["corner_offsets"][index:index + 2])
        surface_slice = slice(*surface["surface_offsets"][index:index + 2])
        corner_xyz = corners["corner_world_xyz"][corner_slice]
        corner_kind = corners["corner_object_id"][corner_slice]
        surface_xyz = surface["surface_world_xyz"][surface_slice]
        surface_kinds = surface.get("surface_object_id")
        surface_kinds = (np.zeros(len(surface_xyz), dtype=np.int8) if surface_kinds is None
                         else surface_kinds[surface_slice])
        observed = {(int(kind), j): point for j, (kind, point) in enumerate(zip(corner_kind, corner_xyz, strict=True))}
        matches = match_vertices(observed, reference, gate_m=0.005, margin_m=0.002)
        identity = [VERTEX_ID[matches[key]] if key in matches else -1 for key in observed]
        xyz.append(np.concatenate((corner_xyz, surface_xyz)))
        kinds.append(np.concatenate((corner_kind, surface_kinds)))
        ids.extend(identity + [-1] * len(surface_xyz))
        is_corner.extend([True] * len(corner_xyz) + [False] * len(surface_xyz))
        offsets.append(offsets[-1] + len(xyz[-1]))
    if len(tcp) != count:
        raise ValueError(f"Unexpected frame count: {raw}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, offsets=np.asarray(offsets), xyz=np.concatenate(xyz).astype(np.float32),
            kinds=np.concatenate(kinds), vertex_ids=np.asarray(ids), is_corner=np.asarray(is_corner),
            tcp=np.asarray(tcp, dtype=np.float32), targets=np.asarray(targets, dtype=np.float32),
        )
    temporary.replace(output)
    return {"episode": episode_id, "file": output.name, "frames": count,
            "split": split, "sha256": file_sha256(output), "seed": row["seed"]}


def read_cache(root, receipt, fields, frames, *, optional=()):
    """Verify cached arrays and their episode-local frame sequence."""
    path = root / receipt["file"]
    if file_sha256(path) != receipt["sha256"]:
        raise ValueError(f"Cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["frame_indices"], np.arange(frames)):
            raise ValueError(f"Frame index mismatch: {path}")
        return {field: archive[field] for field in (*fields, *(key for key in optional if key in archive))}


def prepare_imported(settings, fallback=None):
    """Read complete pre-condition caches; retain all train and validation frames."""
    root = Path(settings["root"])
    manifest = json.loads((root / "manifest.json").read_text())
    cache = Path(settings["cache_root"])
    destination = Path(settings["output"])
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["archives"]:
        name = Path(item["path"]).name
        stem = name.removesuffix(".tar.zst")
        output = destination / f"{stem}.json"
        if output.exists() or item["split"] == "test":
            continue
        source = root / item["path"]
        if file_sha256(source) != item["sha256"]:
            raise ValueError(f"Raw archive mismatch: {source}")
        cache_receipt = cache / f"{stem}.json"
        cached = json.loads(cache_receipt.read_text()) if cache_receipt.exists() else None
        if cached is not None and cached["raw_archive"]["sha256"] != item["sha256"]:
            raise ValueError(f"Raw/cache lineage mismatch: {cache_receipt}")
        index = {(row["seed"], row["policy_mode"]): row for row in cached["episodes"]} if cached else {}
        receipts = []
        with TemporaryDirectory(prefix="corner_tracks_", dir=destination) as scratch:
            subprocess.run(["tar", "--zstd", "-xf", str(source), "-C", scratch], check=True)
            for case in item["episode_index"]:
                raw = Path(scratch) / case["path"]
                row = json.loads((raw / "episode.json").read_text())
                record = index.get((case["seed"], case["policy_mode"]))
                if record is None:
                    if fallback is None:
                        raise ValueError(f"Missing complete corner and surface cache: {cache_receipt}")
                    corners, surface = fallback(raw, row)
                else:
                    if any(record[key] != case[key] for key in ("path", "observations", "split")):
                        raise ValueError(f"Cache episode mismatch: {case['path']}")
                    corners = read_cache(cache, record["files"]["corners"],
                                         ("corner_offsets", "corner_world_xyz", "corner_object_id"), case["observations"])
                    surface = read_cache(cache, record["files"]["surface"],
                                          ("surface_offsets", "surface_world_xyz"), case["observations"],
                                          optional=("surface_object_id",))
                episode_id = f"five_mode_{case['seed']}_{case['policy_mode']}"
                receipts.append(pack_episode(raw, row, corners, surface, destination / f"{episode_id}.npz",
                                             case["split"], episode_id))
        write_json(output, {"episodes": receipts, "archive_sha256": item["sha256"]})
        print(stem, len(receipts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("cache_root")
    parser.add_argument("output")
    prepare_imported(vars(parser.parse_args()))
