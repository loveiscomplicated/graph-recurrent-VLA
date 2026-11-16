"""Index recorded frames by geometrically verified TAPNext++ corner observations."""

import json
from pathlib import Path

import numpy as np

from data_collection.state21_shared_tapnext import (
    _lift_tracks,
    _open_source,
    split_reset_seeds,
)
from data_collection.state21_vertex_targets import (
    POSE_FIELDS,
    corners_from_pose,
    match_vertices,
)
from data_collection.vertex_family_labels import SIGNS


def classify(matches):
    """Return the exact visibility group for distinct physical corner matches."""
    peg = sorted({identity for _, kind, identity in matches if kind == "peg"})
    # Insertion approaches the negative-X mouth (corner IDs 0–3).
    rim = {
        identity for _, kind, identity in matches
        if kind == "hole_rim" and identity < 4
    }
    if len(peg) >= 4 and len(rim) >= 3:
        signs = SIGNS[peg]
        if np.linalg.matrix_rank(signs[1:] - signs[0]) == 3:
            return "condition_1"
    if len(peg) == 3 and len(rim) == 2:
        return "condition_2"
    if len(peg) < 2 and len(rim) < 1:
        return "condition_3"
    return None


def _frame_matches(path, pixels, visible, kinds, camera, gate_m, margin_m):
    """Lift visible pixels with measured depth and match them to GT geometry."""
    if not visible.any():
        return ()
    with np.load(path, allow_pickle=False) as raw:
        depth = raw[f"observation/sensor_data/{camera}/depth"][0, ..., 0]
        intrinsic = raw[f"observation/sensor_param/{camera}/intrinsic_cv"][0]
        extrinsic = raw[f"observation/sensor_param/{camera}/extrinsic_cv"][0]
        pose = np.concatenate(
            [raw[f"ground_truth/{field}"][0].reshape(width) for field, width in POSE_FIELDS]
        )
    points = _lift_tracks(pixels, visible, depth, intrinsic, extrinsic)
    observed = {(int(kinds[query]), query): xyz for query, xyz in points.items()}
    if not observed:
        return ()
    return match_vertices(
        observed, corners_from_pose(pose), gate_m=gate_m, margin_m=margin_m
    ).values()


def _index_source(source, groups, split_by_seed, camera, gate_m, margin_m):
    """Scan the cache in original episode and frame order."""
    arrays = source.arrays
    repo_root = Path.cwd().resolve()
    frames = len(arrays["frame_indices"])
    for index, entry in enumerate(source.entries):
        split = split_by_seed[entry["seed"]]
        directory = (source.raw_root / entry["path"]).resolve()
        if not directory.is_relative_to(source.raw_root):
            raise ValueError(f"Invalid episode directory: {directory}")
        first, last = arrays["episode_offsets"][index : index + 2]
        query_first, query_last = arrays["corner_query_offsets"][index : index + 2]
        kinds = arrays["corner_query_object_id"][query_first:query_last]
        for row in range(first, last):
            frame = arrays["frame_indices"][row]
            path = directory / f"frame_{frame:06d}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            start, stop = arrays["corner_frame_offsets"][row : row + 2]
            matches = _frame_matches(
                path,
                arrays["corner_xy"][start:stop],
                arrays["corner_visible"][start:stop],
                kinds,
                camera,
                gate_m,
                margin_m,
            )
            condition = classify(matches)
            if condition:
                location = (
                    path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
                )
                groups[condition][split].append(location.as_posix())
        if (index + 1) % 50 == 0:
            print(
                f"{source.raw_root.name}: {index + 1}/{len(source.entries)} episodes",
                flush=True,
            )
    return len(source.entries), frames


def run(config):
    """Write a new JSON containing only lookup lists of original frame paths."""
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Choose a new output index path: {output}")
    gate_m = float(config["gate_m"])
    margin_m = float(config["margin_m"])
    if not np.isfinite([gate_m, margin_m]).all() or not 0 < margin_m <= gate_m:
        raise ValueError("Expected 0 < margin_m <= gate_m with finite metre values")
    sources = [_open_source(settings) for settings in config["sources"]]
    seeds = [entry["seed"] for source in sources for entry in source.entries]
    splits = split_reset_seeds(
        seeds,
        seed=config["split_seed"],
        validation_count=config["validation_seed_count"],
        test_count=config["test_seed_count"],
    )
    split_by_seed = {seed: split for split, seeds in splits.items() for seed in seeds}
    groups = {
        condition: {split: [] for split in splits}
        for condition in ("condition_1", "condition_2", "condition_3")
    }
    total_episodes = total_frames = 0
    for source in sources:
        episodes, frames = _index_source(
            source, groups, split_by_seed, config["camera"], gate_m, margin_m
        )
        total_episodes += episodes
        total_frames += frames
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(groups, stream, indent=2)
        stream.write("\n")
    counts = {
        condition: {split: len(paths) for split, paths in split_paths.items()}
        for condition, split_paths in groups.items()
    }
    indexed_frames = sum(sum(split_counts.values()) for split_counts in counts.values())
    print(f"{total_episodes} episodes, {total_frames} frames: {counts}")
    print(f"Unmatched: {total_frames - indexed_frames}. Index: {output}")
