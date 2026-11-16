"""Load the shared TAPNext++ cache and measured depth for Graph Object21D."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.state21 import (
    Episode,
    Sample,
    audit_splits,
    check_split_owner,
    is_sha256,
)
from graph.state_estimation.codec import camera_to_world
from graph.state_estimation.graph import Frame, Node, Source
from recurrent.features import FEATURE_LAYOUT

SPLITS = ("train", "validation", "test")


def split_reset_seeds(seeds, *, seed, validation_count, test_count):
    """Permute sorted reset seeds once, keeping all policy variants together."""
    unique = np.unique(seeds)
    if (
        type(validation_count) is not int
        or type(test_count) is not int
        or min(validation_count, test_count) < 1
        or validation_count + test_count >= len(unique)
    ):
        raise ValueError(
            "Validation and test each need reset seeds, with at least one left for train"
        )
    shuffled = np.random.default_rng(seed).permutation(unique).tolist()
    test_start = validation_count
    train_start = test_start + test_count
    return {
        "train": shuffled[train_start:],
        "validation": shuffled[:test_start],
        "test": shuffled[test_start:train_start],
    }


def _check_offsets(offsets, entries, final, name):
    """Validate half-open row ranges before indexing the packed arrays."""
    if (
        offsets.shape != (entries + 1,)
        or offsets.dtype.kind not in "iu"
        or offsets[0] != 0
        or offsets[-1] != final
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"Invalid {name} in shared TAPNext++ cache")


def _verify_cache(arrays, metadata, source):
    """Check archive schema and alignment against every source episode."""
    episode_ids = arrays["episode_ids"].tolist()
    entries = source["episodes"]
    if episode_ids != [row["path"] for row in entries] or len(set(episode_ids)) != len(
        episode_ids
    ):
        raise ValueError("Shared cache and source manifest episode order differ")
    frame_indices = arrays["frame_indices"]
    seeds = arrays["reset_seeds"]
    if seeds.dtype.kind not in "iu" or seeds.tolist() != [
        row["seed"] for row in entries
    ]:
        raise ValueError("Shared cache and source manifest reset seeds differ")
    if frame_indices.ndim != 1 or frame_indices.dtype.kind not in "iu":
        raise ValueError("Shared cache frame indices must be a vector of integers")
    total = len(frame_indices)
    states = arrays["state36_gt"]
    expected_layout = [list(field) for field in FEATURE_LAYOUT[:-2]]
    if (
        metadata["state36_layout"] != expected_layout
        or states.shape != (total, 36)
        or states.dtype != np.float32
        or not np.isfinite(states).all()
    ):
        raise ValueError(
            "Expected finite float32 State36 with the Recurrent feature layout"
        )

    surface_xy, surface_visible = arrays["surface_xy"], arrays["surface_visible"]
    corner_xy, corner_visible = arrays["corner_xy"], arrays["corner_visible"]
    corner_objects = arrays["corner_query_object_id"]
    if (
        surface_xy.ndim != 3
        or surface_xy.shape[0] != total
        or surface_xy.shape[-1] != 2
        or surface_visible.shape != surface_xy.shape[:2]
        or surface_visible.dtype != np.bool_
        or corner_xy.ndim != 2
        or corner_xy.shape[-1] != 2
        or corner_visible.shape != (len(corner_xy),)
        or corner_visible.dtype != np.bool_
        or corner_objects.ndim != 1
        or corner_objects.dtype.kind not in "iu"
        or not np.isin(corner_objects, [0, 1]).all()
    ):
        raise ValueError("Invalid TAPNext++ coordinates, visibility or object IDs")
    episode_offsets = arrays["episode_offsets"]
    query_offsets = arrays["corner_query_offsets"]
    frame_offsets = arrays["corner_frame_offsets"]
    _check_offsets(episode_offsets, len(entries), total, "episode offsets")
    _check_offsets(
        query_offsets, len(entries), len(corner_objects), "corner query offsets"
    )
    _check_offsets(frame_offsets, total, len(corner_xy), "corner frame offsets")
    for index, row in enumerate(entries):
        start, stop = episode_offsets[index : index + 2]
        length = int(stop - start)
        if (
            length != row["observations"]
            or length != len(row["frame_sha256"])
            or length != row["steps"] + 1
            or length < 1
            or not np.array_equal(frame_indices[start:stop], np.arange(length))
        ):
            raise ValueError(
                f"Shared cache frame order or count differs for {row['path']}"
            )
        if not np.all(
            np.diff(frame_offsets[start : stop + 1])
            == query_offsets[index + 1] - query_offsets[index]
        ):
            raise ValueError(f"Corner query count changes within {row['path']}")


def _audit_full_split(entries, groups):
    """Check complete source lineage before selecting any smoke frames."""
    split_by_seed = {seed: split for split, seeds in groups.items() for seed in seeds}
    owners = {}
    episode_counts = dict.fromkeys(SPLITS, 0)
    frame_counts = dict.fromkeys(SPLITS, 0)
    for row in entries:
        split = split_by_seed[row["seed"]]
        if row.get("initial_fingerprint"):
            check_split_owner(
                owners, "initial scene fingerprint", row["initial_fingerprint"], split
            )
        for digest in row["frame_sha256"]:
            check_split_owner(owners, "source frame", digest, split)
        episode_counts[split] += 1
        frame_counts[split] += row["observations"]
    return {
        "reset_seed_counts": {split: len(seeds) for split, seeds in groups.items()},
        "reset_seeds": groups,
        "episode_counts": episode_counts,
        "frame_counts": frame_counts,
    }


def _selection(groups, smoke):
    """Restrict smoke runs to causal prefixes from train and validation only."""
    if smoke is None:
        return groups, None
    selected = {}
    for split in ("train", "validation"):
        limit = smoke[f"{split}_seed_count"]
        if type(limit) is not int or not 1 <= limit <= len(groups[split]):
            raise ValueError(f"Invalid smoke {split}_seed_count")
        selected[split] = groups[split][:limit]
    length = smoke["frames_per_episode"]
    if type(length) is not int or length < 1:
        raise ValueError("Expected positive smoke frames_per_episode")
    return selected, length


def _read_observed_depth(path, digest, camera):
    """Read only sensor geometry and measured TCP from the source observation."""
    if file_sha256(path) != digest:
        raise ValueError(f"Raw frame checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as raw:
        depth = raw[f"observation/sensor_data/{camera}/depth"][0]
        intrinsic = raw[f"observation/sensor_param/{camera}/intrinsic_cv"][0]
        extrinsic = raw[f"observation/sensor_param/{camera}/extrinsic_cv"][0]
        tcp = raw["observation/extra/tcp_pose"][0, :3]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.dtype != np.int16:
        raise ValueError(f"Expected signed int16 depth in millimetres: {path}")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"Expected finite OpenCV camera intrinsics: {path}")
    return depth, intrinsic, extrinsic, tuple(tcp.tolist())


def _lift_tracks(uv, visible, depth, intrinsic, extrinsic):
    """Sample current depth at visible TAPNext++ pixels and backproject to world metres."""
    height, width = depth.shape
    finite = np.isfinite(uv).all(axis=1)
    pixels = np.rint(np.where(finite[:, None], uv, -1))
    keep = (
        visible
        & finite
        & (pixels >= 0).all(axis=1)
        & (pixels < [width, height]).all(axis=1)
    )
    rows = np.flatnonzero(keep)
    x, y = pixels[rows].astype(np.int64).T
    depth_mm = depth[y, x]
    valid_depth = depth_mm > 0
    rows = rows[valid_depth]
    rays = np.column_stack((uv[rows], np.ones(len(rows)))) @ np.linalg.inv(intrinsic).T
    world = camera_to_world(rays * (depth_mm[valid_depth] * 0.001)[:, None], extrinsic)
    return {
        int(row): tuple(position) for row, position in zip(rows, world, strict=True)
    }


def _load_episode(
    arrays, index, entry, split, source_root, camera, max_frames, *, first_frame=0
):
    """Use stable episode-local query IDs and keep Object21D outside the Frame."""
    name = entry["path"]
    directory = (source_root / name).resolve()
    if Path(name).is_absolute() or not directory.is_relative_to(source_root):
        raise ValueError(f"Source episode path must stay within the dataset: {name}")
    start, stop = arrays["episode_offsets"][index : index + 2]
    start += first_frame
    if max_frames is not None:
        stop = min(stop, start + max_frames)
    queries = slice(*arrays["corner_query_offsets"][index : index + 2])
    kinds = arrays["corner_query_object_id"][queries].tolist()

    samples = []
    for frame_index, flat_index in enumerate(range(start, stop), start=first_frame):
        path = directory / f"frame_{frame_index:06d}.npz"
        digest = entry["frame_sha256"][frame_index]
        depth, intrinsic, extrinsic, tcp = _read_observed_depth(path, digest, camera)
        corner_rows = slice(
            *arrays["corner_frame_offsets"][flat_index : flat_index + 2]
        )
        corners = _lift_tracks(
            arrays["corner_xy"][corner_rows],
            arrays["corner_visible"][corner_rows],
            depth,
            intrinsic,
            extrinsic,
        )
        surface = _lift_tracks(
            arrays["surface_xy"][flat_index],
            arrays["surface_visible"][flat_index],
            depth,
            intrinsic,
            extrinsic,
        )
        # Only depth-backed points supply geometry. An absent corner retains its
        # episode-local query ID for causal binding without a coordinate.
        nodes = [
            Node(corners[query], kind, True, query, True, None, Source.OBSERVED)
            if query in corners
            else Node(None, kind, True, query, False, None, Source.MISSING)
            for query, kind in enumerate(kinds)
        ]
        nodes.extend(
            Node(position, 0, False, None, True, None, Source.OBSERVED)
            for position in surface.values()
        )
        target = torch.from_numpy(arrays["state36_gt"][flat_index, 15:].copy())
        samples.append(
            Sample(
                Frame(name, frame_index, tcp, tuple(nodes)), frame_index, target, digest
            )
        )
    return Episode(
        name,
        f"reset_seed:{entry['seed']}",
        split,
        tuple(samples),
        entry.get("initial_fingerprint"),
    )


@dataclass(frozen=True)
class SharedSource:
    """One verified archive paired with its original camera recordings."""

    arrays: dict
    metadata: dict
    entries: list
    raw_root: Path
    archive_sha256: str
    source_sha256: str
    smoke_center: str | None


def _open_source(settings):
    """Validate one cache independently of the split and frame selection."""
    path = Path(settings["cache"])
    source_path = Path(settings["source_manifest"]).resolve()
    source_hash = file_sha256(source_path)
    source = json.loads(source_path.read_text())
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        arrays = {
            name: archive[name] for name in archive.files if name != "metadata_json"
        }
    if (
        metadata.get("format") != "shared_tapnext_state36"
        or metadata.get("format_version") != 1
        or not is_sha256(metadata.get("tapnext_checkpoint_sha256"))
        or metadata.get("source_manifest_sha256") != source_hash
        or source.get("status") != "COMPLETE"
    ):
        raise ValueError(
            "Expected a shared TAPNext++ State36 archive and its complete raw source manifest"
        )
    _verify_cache(arrays, metadata, source)
    center = settings.get("smoke_center")
    if center not in (None, "recovery_step"):
        raise ValueError("smoke_center must be recovery_step or omitted")
    return SharedSource(
        arrays,
        metadata,
        source["episodes"],
        source_path.parent,
        file_sha256(path),
        source_hash,
        center,
    )


def _first_smoke_frame(entry, center, length):
    """Pick a contiguous window that includes both sides of the expert return."""
    if center is None or length is None:
        return 0
    step = entry.get(center)
    count = entry["observations"]
    if type(step) is not int or not 0 < step < count or length < 2:
        raise ValueError(f"Cannot include both sides of {center} in {entry['path']}")
    return min(max(0, step - length // 2), max(0, count - length))


def _reference_split(settings, sources, groups):
    """Check the computed grouping against any recorded split assignment."""
    reference = settings.get("split_reference")
    if reference is not None:
        reference_path = Path(reference)
        pinned = json.loads(reference_path.read_text())["full_split"]["reset_seeds"]
        if groups != pinned:
            raise ValueError("Reset-seed assignment differs from the pinned split")
    for source in sources.values():
        recorded = source.metadata.get("reset_seeds_by_split")
        if recorded is not None and recorded != groups:
            raise ValueError(
                "Reset-seed assignment differs from the shared cache metadata"
            )
    return file_sha256(reference_path) if reference is not None else None


def load_shared_tapnext(settings, *, splits=None):
    """Join one or more TAPNext++ archives on a common reset-seed partition."""
    inputs = settings.get("sources")
    if inputs is None:
        inputs = {"pilot": settings}
    elif not isinstance(inputs, dict) or not inputs:
        raise ValueError("sources must be a nonempty mapping of named TAPNext++ caches")
    sources = {name: _open_source(config) for name, config in inputs.items()}
    entries = [row for source in sources.values() for row in source.entries]
    if len({row["path"] for row in entries}) != len(entries):
        raise ValueError("Shared TAPNext++ sources contain duplicate episode IDs")
    tracker_hashes = {
        source.metadata["tapnext_checkpoint_sha256"] for source in sources.values()
    }
    if len(tracker_hashes) != 1:
        raise ValueError(
            "Shared TAPNext++ sources must use the same tracker checkpoint"
        )

    groups = split_reset_seeds(
        [row["seed"] for row in entries],
        seed=settings["split_seed"],
        validation_count=settings["validation_seed_count"],
        test_count=settings["test_seed_count"],
    )
    split_reference_sha256 = _reference_split(settings, sources, groups)
    full_split = _audit_full_split(entries, groups)
    selected, max_frames = _selection(groups, settings.get("smoke"))
    if splits is not None:
        if not splits or not set(splits) <= set(selected):
            raise ValueError(
                "Requested splits are outside the configured frame selection"
            )
        selected = {split: selected[split] for split in splits}
    split_by_seed = {seed: split for split, seeds in selected.items() for seed in seeds}
    camera = settings.get("camera", "base_camera")
    if camera != "base_camera":
        raise ValueError(
            "Shared TAPNext++ State36 v1 tracks use the base_camera pixel coordinates"
        )

    episodes = []
    source_report = {}
    for name, source in sources.items():
        windows = []
        for index, row in enumerate(source.entries):
            split = split_by_seed.get(row["seed"])
            if split is None:
                continue
            first = _first_smoke_frame(row, source.smoke_center, max_frames)
            episode = _load_episode(
                source.arrays,
                index,
                row,
                split,
                source.raw_root,
                camera,
                max_frames,
                first_frame=first,
            )
            episodes.append(episode)
            window = {
                "episode_id": episode.episode_id,
                "split": split,
                "first_frame": first,
                "frame_count": len(episode.samples),
            }
            if source.smoke_center:
                window[source.smoke_center] = row[source.smoke_center]
            windows.append(window)
        source_report[name] = {
            "cache_sha256": source.archive_sha256,
            "source_manifest_sha256": source.source_sha256,
            "selected_frame_counts": {
                split: sum(
                    row["frame_count"] for row in windows if row["split"] == split
                )
                for split in selected
            },
            "selected_windows": windows,
        }

    first_source = next(iter(sources.values()))
    single_cache = len(sources) == 1 and split_reference_sha256 is None
    identity = (
        first_source.archive_sha256
        if single_cache
        else canonical_hash(
            {
                "sources": {
                    name: source.archive_sha256 for name, source in sources.items()
                },
                "split_reference": split_reference_sha256,
            }
        )
    )
    provenance = {
        "tracker": "TAPNext++",
        "tracker_sha256": next(iter(tracker_hashes)),
        "tracker_training_or_selection_parents": None,
        "vertices": "episode-local tracked query IDs, visible with current measured depth",
        "surface": "Peg tracked pixels, visible with current measured depth",
        "tcp": "raw observation/extra/tcp_pose",
        "supervision": "State36 columns 15:36 (zero-based)",
    }
    if len(sources) == 1:
        provenance["source_manifest_sha256"] = first_source.source_sha256
    return episodes, {
        "manifest_sha256": identity,
        "full_split": full_split,
        "split_reference_sha256": split_reference_sha256,
        "selected_reset_seeds": selected,
        "sources": source_report,
        "selected_frame_counts": {
            split: sum(
                len(episode.samples) for episode in episodes if episode.split == split
            )
            for split in selected
        },
        "provenance": provenance,
        **audit_splits(episodes),
    }
