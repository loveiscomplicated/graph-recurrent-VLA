"""Prepare full chronological B2-3 corner-tracking supervision.

The input observations come from the current Five-mode ``corners`` and
``surface`` cache members.  Simulator poses are read only to create offline
training labels and never become model inputs.  Train and validation episodes
are retained in their original split and frame order without a visibility
condition filter.
"""

from collections import Counter, defaultdict
from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import numpy as np

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.spatial_corner_tracking import VERTEX_ID, VERTICES
from data_collection.state21_vertex_targets import (
    POSE_FIELDS,
    corners_from_pose,
    match_vertices,
)
from data_collection.storage import write_json
from training.gnn.packed_five_mode_data import read_archive_member


FORBIDDEN_CACHE_NAMES = {
    "tapnext_state36.npz",
    "tapnext_recovery_state36.npz",
}


def _reject_forbidden_cache_reference(value):
    """Reject the legacy shared State36 caches before opening any input."""
    if Path(str(value)).name in FORBIDDEN_CACHE_NAMES:
        raise PermissionError(f"Forbidden legacy cache reference: {value}")


def _raw_archive_path(raw_root, batch):
    """Return the pinned local RGB-D archive path for one collection batch."""
    if not isinstance(batch, str) or "_" not in batch or Path(batch).name != batch:
        raise ValueError(f"Invalid Five-mode batch name: {batch!r}")
    suffix = batch.split("_", 1)[1]
    return raw_root / batch / f"episodes_{suffix}.tar.zst"


def _read_raw_episode_index(path):
    """Find the episode index even when archive frames precede it.

    Read metadata in a separate streaming pass so no RGB-D frames need to be
    buffered while waiting for their episode directory mapping.
    """
    process = subprocess.Popen(
        ["zstd", "-qdc", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                is_index = PurePosixPath(member.name) == PurePosixPath("episode_index.json")
                if is_index and member.isfile():
                    return json.load(archive.extractfile(member))
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
    raise ValueError(f"Raw Five-mode archive has no episode_index.json: {path}")


def _read_raw_pose_sequences(path, digest, records, *, verify_checksum=True):
    """Stream only the GT pose fields needed for canonical corner labels."""
    _reject_forbidden_cache_reference(path)
    if verify_checksum and file_sha256(path) != digest:
        raise ValueError(f"Raw Five-mode archive checksum differs: {path}")

    expected = {(row["seed"], row["policy_mode"]): row for row in records}
    if len(expected) != len(records):
        raise ValueError(f"Duplicate episodes requested from raw archive: {path}")

    pose_width = sum(width for _, width in POSE_FIELDS)
    poses = {}
    directories = {}
    seen_frames = {key: set() for key in expected}
    saw_index = False
    index = _read_raw_episode_index(path)
    for row in index:
        key = row.get("seed"), row.get("policy_mode")
        if key not in expected:
            continue
        directory = PurePosixPath(row["path"])
        record = expected[key]
        if (
            directory.is_absolute()
            or ".." in directory.parts
            or directory in directories
            or key in poses
            or row.get("observations") != record["frames"]
            or row.get("split") != record["split"]
        ):
            raise ValueError(f"Invalid raw episode index for {key}: {path}")
        poses[key] = np.empty((record["frames"], pose_width), dtype=np.float32)
        directories[directory] = key
    if poses.keys() != expected.keys():
        raise ValueError(f"Raw Five-mode index is missing requested episodes: {path}")

    process = subprocess.Popen(["zstd", "-qdc", str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if name == PurePosixPath("episode_index.json") and member.isfile():
                    if saw_index:
                        raise ValueError(f"Duplicate episode_index.json: {path}")
                    saw_index = True
                    if json.load(archive.extractfile(member)) != index:
                        raise ValueError(f"Raw Five-mode index changed between reads: {path}")
                    continue

                key = directories.get(name.parent)
                if key is None or not member.isfile() or not name.name.startswith("frame_"):
                    continue
                try:
                    frame_index = int(name.stem.removeprefix("frame_"))
                except ValueError as error:
                    raise ValueError(f"Invalid raw Five-mode frame name: {name}") from error
                if (
                    name.suffix != ".npz"
                    or not 0 <= frame_index < len(poses[key])
                    or frame_index in seen_frames[key]
                ):
                    raise ValueError(f"Unexpected raw Five-mode frame: {name}")
                content = archive.extractfile(member).read()
                with np.load(BytesIO(content), allow_pickle=False) as frame:
                    poses[key][frame_index] = np.concatenate(
                        [
                            frame[f"ground_truth/{field}"][0].reshape(width)
                            for field, width in POSE_FIELDS
                        ]
                    )
                seen_frames[key].add(frame_index)

        # Read tar padding before waiting so zstd can validate the complete stream.
        while process.stdout.read(65536):
            pass
        if process.wait() != 0:
            raise ValueError(f"Could not decompress raw Five-mode archive: {path}")
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()

    if not saw_index or poses.keys() != expected.keys():
        raise ValueError(f"Raw Five-mode archive is missing requested episodes: {path}")
    if any(len(seen_frames[key]) != len(values) for key, values in poses.items()):
        raise ValueError(f"Raw Five-mode frames are incomplete: {path}")
    return poses


def _validate_cache_tables(record, tables):
    """Validate alignment and shapes before matching detector rows to GT labels."""
    frames = record["frames"]
    frame_indices = np.arange(frames)
    if any(
        not np.array_equal(table.get("frame_indices"), frame_indices)
        for table in tables.values()
    ):
        raise ValueError(
            f"Unaligned Five-mode cache frames: {record['seed']}:{record['policy_mode']}"
        )

    corners = tables["corners"]
    surface = tables["surface"]
    targets = tables["targets"]
    corner_rows = int(corners["corner_offsets"][-1])
    surface_rows = int(surface["surface_offsets"][-1])
    if (
        corners["corner_offsets"].shape != (frames + 1,)
        or surface["surface_offsets"].shape != (frames + 1,)
        or np.any(np.diff(corners["corner_offsets"]) < 0)
        or np.any(np.diff(surface["surface_offsets"]) < 0)
        or corners["corner_world_xyz"].shape != (corner_rows, 3)
        or corners["corner_object_id"].shape != (corner_rows,)
        or surface["surface_world_xyz"].shape != (surface_rows, 3)
        or targets["tcp"].shape != (frames, 3)
    ):
        raise ValueError(
            f"Invalid Five-mode cache shapes: {record['seed']}:{record['policy_mode']}"
        )
    surface_object_id = surface.get("surface_object_id")
    if surface_object_id is not None and surface_object_id.shape != (surface_rows,):
        raise ValueError("surface_object_id must align with every TAPNext++ surface row")
    values = (
        corners["corner_world_xyz"],
        surface["surface_world_xyz"],
        targets["tcp"],
    )
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("Five-mode cache coordinates must be finite")
    if not np.isin(corners["corner_object_id"], (0, 1)).all():
        raise ValueError("Detector corner object IDs must be Peg=0 or Box/Hole=1")
    if surface_object_id is not None and not np.isin(surface_object_id, (0, 1)).all():
        raise ValueError("Surface object IDs must be Peg=0 or Box/Hole=1")


def _pack_episode_arrays(record, tables, poses, *, gate_m, margin_m):
    """Create one chronological episode with IDs isolated as supervision metadata."""
    _validate_cache_tables(record, tables)
    frames = record["frames"]
    if poses.shape != (frames, sum(width for _, width in POSE_FIELDS)):
        raise ValueError("Raw pose sequence does not match the cached episode")

    corners = tables["corners"]
    surface = tables["surface"]
    stored_surface_kinds = surface.get("surface_object_id")
    surface_identity_mode = (
        "explicit_per_row" if stored_surface_kinds is not None else "historical_peg_only"
    )

    offsets = [0]
    all_xyz = []
    all_kinds = []
    all_vertex_ids = []
    all_is_corner = []
    target_rows = []

    for frame_index, pose in enumerate(poses):
        reference = corners_from_pose(pose)
        target_rows.append([reference[vertex] for vertex in VERTICES])

        corner_slice = slice(*corners["corner_offsets"][frame_index : frame_index + 2])
        surface_slice = slice(*surface["surface_offsets"][frame_index : frame_index + 2])
        corner_xyz = corners["corner_world_xyz"][corner_slice]
        corner_kinds = corners["corner_object_id"][corner_slice]
        surface_xyz = surface["surface_world_xyz"][surface_slice]
        if stored_surface_kinds is None:
            surface_kinds = np.zeros(len(surface_xyz), dtype=np.int8)
        else:
            surface_kinds = stored_surface_kinds[surface_slice]

        observed = {
            (int(kind), detection_index): point
            for detection_index, (kind, point) in enumerate(
                zip(corner_kinds, corner_xyz, strict=True)
            )
        }
        matches = match_vertices(
            observed,
            reference,
            gate_m=gate_m,
            margin_m=margin_m,
        )
        vertex_ids = np.asarray(
            [VERTEX_ID[matches[key]] if key in matches else -1 for key in observed],
            dtype=np.int16,
        )

        frame_xyz = np.concatenate((corner_xyz, surface_xyz), axis=0).astype(
            np.float32, copy=False
        )
        frame_kinds = np.concatenate((corner_kinds, surface_kinds), axis=0).astype(
            np.int8, copy=False
        )
        all_xyz.append(frame_xyz)
        all_kinds.append(frame_kinds)
        all_vertex_ids.append(
            np.concatenate(
                (vertex_ids, np.full(len(surface_xyz), -1, dtype=np.int16))
            )
        )
        all_is_corner.append(
            np.concatenate(
                (
                    np.ones(len(corner_xyz), dtype=bool),
                    np.zeros(len(surface_xyz), dtype=bool),
                )
            )
        )
        offsets.append(offsets[-1] + len(frame_xyz))

    arrays = {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "xyz": np.concatenate(all_xyz, axis=0),
        "kinds": np.concatenate(all_kinds, axis=0),
        "vertex_ids": np.concatenate(all_vertex_ids, axis=0),
        "is_corner": np.concatenate(all_is_corner, axis=0),
        "tcp": tables["targets"]["tcp"].astype(np.float32, copy=False),
        "targets": np.asarray(target_rows, dtype=np.float32),
    }
    return arrays, surface_identity_mode


def _write_episode(path, arrays):
    """Atomically write and read back one compressed supervision episode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
    with np.load(temporary, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError(f"Prepared episode fields differ after write: {path}")
        for name, expected in arrays.items():
            np.testing.assert_array_equal(saved[name], expected)
    temporary.replace(path)


def _load_completed_batch(receipt_path, output_root, identity):
    """Reuse a completed batch only after checking lineage and episode hashes."""
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("identity") != identity:
        return None
    episodes = receipt.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return None
    expected = [tuple(row[:2]) for row in identity["episodes"]]
    actual = [(row.get("seed"), row.get("policy_mode")) for row in episodes]
    if actual != expected:
        return None
    for episode in episodes:
        path = (output_root / episode["file"]).resolve()
        if (
            not path.is_relative_to(output_root)
            or not path.is_file()
            or file_sha256(path) != episode["sha256"]
        ):
            return None
    return episodes


def _open_release_archives(stack, release_root, manifest):
    """Verify and open every listed uncompressed cache shard once."""
    archives = {}
    for row in manifest.get("archives", []):
        name = row.get("file")
        _reject_forbidden_cache_reference(name)
        path = (release_root / name).resolve()
        if not path.is_relative_to(release_root) or file_sha256(path) != row.get("sha256"):
            raise ValueError(f"Five-mode cache shard checksum/path mismatch: {path}")
        archives[name] = stack.enter_context(tarfile.open(path, "r:"))
    return archives


def _expected_inventory(config, episodes):
    """Check configured split counts before publishing a dataset manifest."""
    episode_counts = Counter(row["split"] for row in episodes)
    frame_counts = Counter()
    for row in episodes:
        frame_counts[row["split"]] += row["frames"]
    expected = config.get("expected", {})
    for split, count in expected.get("episodes", {}).items():
        if episode_counts[split] != int(count):
            raise ValueError(
                f"Expected {count} {split} episodes, found {episode_counts[split]}"
            )
    for split, count in expected.get("frames", {}).items():
        if frame_counts[split] != int(count):
            raise ValueError(f"Expected {count} {split} frames, found {frame_counts[split]}")
    return dict(episode_counts), dict(frame_counts)


def prepare(config):
    """Prepare train/validation episodes without opening reserved test members."""
    return _prepare(config, allowed_splits={"train", "validation"})


def prepare_test(config):
    """Prepare explicitly requested held-out test labels in a separate dataset."""
    return _prepare(config, allowed_splits={"test"}, include_object21=True)


def _prepare(config, *, allowed_splits, include_object21=False):
    """Stream the requested, explicitly permitted splits into episode arrays."""
    source = config.get("source", {})
    output = config.get("output", {})
    manifest_path = Path(source["manifest"]).expanduser().resolve()
    raw_root = Path(source["raw_root"]).expanduser().resolve()
    output_root = Path(output["root"]).expanduser().resolve()
    for path in (manifest_path, raw_root, output_root):
        _reject_forbidden_cache_reference(path)

    release_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "five_mode_supervised_v1":
        raise ValueError("Expected a five_mode_supervised_v1 source manifest")
    manifest_sha256 = file_sha256(manifest_path)
    splits = tuple(config.get("splits", ("train", "validation")))
    if not splits or not set(splits).issubset(allowed_splits):
        raise ValueError(f"This preparation entrypoint accepts only {sorted(allowed_splits)}")
    records = [row for row in manifest.get("records", []) if row.get("split") in splits]
    if not records:
        raise ValueError("No requested Five-mode episodes were found")
    episode_keys = [(row.get("seed"), row.get("policy_mode")) for row in records]
    if len(set(episode_keys)) != len(episode_keys):
        raise ValueError("Five-mode source manifest contains duplicate episodes")
    episode_counts, frame_counts = _expected_inventory(config, records)
    if "test" in splits:
        test_seeds = {row["seed"] for row in records}
        development_seeds = {
            row["seed"] for row in manifest["records"] if row["split"] != "test"
        }
        if test_seeds & development_seeds:
            raise ValueError("Held-out test seeds overlap train/validation")

    matching = config.get("matching", {})
    gate_m = float(matching.get("gate_m", 0.005))
    margin_m = float(matching.get("margin_m", 0.002))
    if not 0 < margin_m < gate_m:
        raise ValueError("matching.margin_m must be positive and smaller than gate_m")
    verify_raw = bool(source.get("verify_raw_archives", True))

    records_by_batch = defaultdict(list)
    for row in records:
        records_by_batch[row["batch"]].append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    receipt_root = output_root / "_batches"
    receipt_root.mkdir(exist_ok=True)
    episodes = []
    surface_modes = Counter()

    with ExitStack() as stack:
        cache_archives = _open_release_archives(stack, release_root, manifest)
        for batch_index, (batch, batch_records) in enumerate(records_by_batch.items(), 1):
            raw_digests = {row.get("raw_sha256") for row in batch_records}
            if len(raw_digests) != 1 or None in raw_digests:
                raise ValueError(f"Inconsistent raw archive lineage for batch {batch}")
            raw_digest = raw_digests.pop()
            batch_identity = {
                "source_manifest_sha256": manifest_sha256,
                "batch": batch,
                "raw_sha256": raw_digest,
                "episodes": [
                    [row["seed"], row["policy_mode"], row["split"], row["frames"]]
                    for row in batch_records
                ],
                "matching": {"gate_m": gate_m, "margin_m": margin_m},
            }
            if include_object21:
                batch_identity["include_object21_targets"] = True
            receipt_path = receipt_root / f"{batch}.json"
            completed = _load_completed_batch(receipt_path, output_root, batch_identity)
            if completed is not None:
                episodes.extend(completed)
                surface_modes.update(row["surface_identity_mode"] for row in completed)
                print(f"[{batch_index}/{len(records_by_batch)}] reused {batch}", flush=True)
                continue

            raw_path = _raw_archive_path(raw_root, batch)
            poses = _read_raw_pose_sequences(
                raw_path,
                raw_digest,
                batch_records,
                verify_checksum=verify_raw,
            )
            batch_episodes = []
            for row in batch_records:
                tables = {}
                for kind in ("corners", "surface", "targets"):
                    receipt = row["files"][kind]
                    archive = cache_archives.get(receipt["archive"])
                    if archive is None:
                        raise ValueError(f"Unlisted cache archive: {receipt['archive']}")
                    tables[kind] = read_archive_member(archive, receipt, kind)
                key = row["seed"], row["policy_mode"]
                arrays, surface_mode = _pack_episode_arrays(
                    row,
                    tables,
                    poses[key],
                    gate_m=gate_m,
                    margin_m=margin_m,
                )
                if include_object21:
                    object21 = tables["targets"]["object21"]
                    if object21.shape != (row["frames"], 21) or not np.isfinite(object21).all():
                        raise ValueError(f"Invalid held-out Object21 targets: {key}")
                    arrays["object21"] = object21.astype(np.float32, copy=False)
                episode_id = f"five_mode_{row['seed']}_{row['policy_mode']}"
                relative = Path("episodes") / f"{episode_id}.npz"
                destination = output_root / relative
                _write_episode(destination, arrays)
                episode = {
                    "episode": episode_id,
                    "file": relative.as_posix(),
                    "frames": row["frames"],
                    "split": row["split"],
                    "sha256": file_sha256(destination),
                    "seed": row["seed"],
                    "policy_mode": row["policy_mode"],
                    "batch": batch,
                    "surface_identity_mode": surface_mode,
                }
                batch_episodes.append(episode)
                surface_modes[surface_mode] += 1
            write_json(
                receipt_path,
                {"identity": batch_identity, "episodes": batch_episodes},
            )
            episodes.extend(batch_episodes)
            print(
                f"[{batch_index}/{len(records_by_batch)}] prepared {batch}: "
                f"{len(batch_episodes)} episodes",
                flush=True,
            )

    order = {
        (row["seed"], row["policy_mode"]): index for index, row in enumerate(records)
    }
    episodes.sort(key=lambda row: order[(row["seed"], row["policy_mode"])])
    result = {
        "format": "spatial_temporal_corner_tracking_v1",
        "episodes": episodes,
        "episodes_by_split": episode_counts,
        "frames_by_split": frame_counts,
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "raw_root": str(raw_root),
            "raw_archives_verified": verify_raw,
            "corner_source": "current_rgbd_detector",
            "surface_source": "tapnext_non_corner",
            "gt_role": "offline_identity_and_coordinate_supervision_only",
        },
        "matching": {"gate_m": gate_m, "margin_m": margin_m},
        "surface_identity_modes": dict(surface_modes),
        "condition_filter": None,
    }
    result["dataset_sha256"] = canonical_hash(
        {
            "source_manifest_sha256": manifest_sha256,
            "episode_sha256": [row["sha256"] for row in episodes],
            "matching": result["matching"],
            "condition_filter": None,
        }
    )
    write_json(output_root / "manifest.json", result)
    print(
        f"Prepared {len(episodes)} episodes and {sum(frame_counts.values())} frames "
        f"at {output_root}",
        flush=True,
    )

