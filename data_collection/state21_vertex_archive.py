"""Pack verified raw GT poses for the B2 comparison on a separate CUDA host."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.state21_curriculum import STAGES, frame_limit
from data_collection.state21_shared_tapnext import (
    _audit_full_split,
    _open_source,
    _reference_split,
    split_reset_seeds,
)
from data_collection.state21_vertex_targets import read_reference_pose

FORMAT = "state21_vertex_reference_poses_v1"


def _episode_poses(task):
    """Extract the selected original frame prefix after verifying each raw SHA."""
    root, row, count = task
    directory = (root / row["path"]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f"Invalid source episode: {row['path']}")
    return np.stack(
        [
            read_reference_pose(directory / f"frame_{frame:06d}.npz", digest)
            for frame, digest in enumerate(row["frame_sha256"][:count])
        ]
    )


def pack_vertex_references(config, output, *, workers=6):
    """Write the existing train/validation episode pose prefixes and their provenance."""
    data = config["data"]
    sources = {name: _open_source(item) for name, item in data["sources"].items()}
    entries = [row for source in sources.values() for row in source.entries]
    groups = split_reset_seeds(
        [row["seed"] for row in entries],
        seed=data["split_seed"],
        validation_count=data["validation_seed_count"],
        test_count=data["test_seed_count"],
    )
    pin = _reference_split(data, sources, groups)
    full_split = _audit_full_split(entries, groups)
    selected = set(groups["train"]) | set(groups["validation"])
    tasks, archives = [], {}
    random_frames = config["curriculum"]["auxiliary_random_frames"]
    for source in sources.values():
        for index, row in enumerate(source.entries):
            if row["seed"] not in selected:
                continue
            stage_row = {"policy_mode": row["mode"], "frames": row["observations"]}
            count = frame_limit(stage_row, STAGES[-1], 0, random_frames)
            key = row["path"]
            archives[key] = {
                "anchor": int(source.arrays["surface_anchor_frame"][index]),
                "frames": count,
            }
            tasks.append((source.raw_root, row, count))
    arrays = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (key, poses) in enumerate(
            zip(archives, pool.map(_episode_poses, tasks), strict=True), 1
        ):
            arrays[key] = poses
            if index % 60 == 0:
                print(
                    f"Packed reference poses for {index}/{len(tasks)} episodes",
                    flush=True,
                )
    arrays["metadata_json"] = np.asarray(
        json.dumps(
            {
                "format": FORMAT,
                "full_split": full_split,
                "split_reference_sha256": pin,
                "source_caches": {
                    name: source.archive_sha256 for name, source in sources.items()
                },
                "episodes": archives,
            }
        )
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{file_sha256(output)}  {output}", flush=True)
