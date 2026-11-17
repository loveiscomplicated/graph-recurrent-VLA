"""Convert the published 512px expert archives into compact per-episode arrays.

Each shard TAR is read sequentially (nothing is extracted to disk). For every
episode we keep resized RGB of both cameras, low-resolution depth, robot15, the
raw GT needed for state baselines, and the actions aligned with observations:
``actions[t]`` is the command applied at observation ``t`` (stored in frame t+1).
The terminal observation (after the last action) is dropped.

Example:
    python -m visual.baseline.prepare_data \
        --archives data/hf/integration/c555bed7462d4fc09e55ae3bb04421a1 \
        --output data/visual_baseline/episodes
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import tarfile
from pathlib import Path

import numpy as np
import torch

from visual.baseline.common import resize_depth, resize_rgb, robot15

RGB_SIZE = 256
DEPTH_SIZE = 128
CAMERA_KEYS = {"main": "base_camera", "wrist": "hand_camera"}
GT_KEYS = ("qpos", "qvel", "ee_pose", "peg_pose", "hole_pose", "box_pose",
           "peg_half_sizes", "hole_radius", "left_fingertip_pos",
           "right_fingertip_pos", "gripper_qpos")


def read_frame(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict:
    """Load one NPZ frame from the archive into a plain dict of arrays."""
    payload = archive.extractfile(member).read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as frame:
        return {key: frame[key] for key in frame.files}


def convert_episode(frames: list[dict]) -> dict:
    """Turn the T+1 raw frames of one episode into aligned training arrays."""
    observations = frames[:-1]
    arrays = {}
    for name, camera in CAMERA_KEYS.items():
        rgb = np.concatenate([f[f"observation/sensor_data/{camera}/rgb"] for f in observations])
        depth = np.concatenate([f[f"observation/sensor_data/{camera}/depth"] for f in observations])
        arrays[f"{name}_rgb"] = resize_rgb(rgb, RGB_SIZE).numpy()
        arrays[f"{name}_depth"] = resize_depth(depth, DEPTH_SIZE).numpy()

    def gt(key):
        return np.concatenate([f[f"ground_truth/{key}"].reshape(1, -1) for f in observations])

    for key in GT_KEYS:
        arrays[f"gt_{key}"] = gt(key).astype(np.float32)
    arrays["robot15"] = robot15(
        gt("ee_pose"), gt("left_fingertip_pos"), gt("right_fingertip_pos"), gt("gripper_qpos")
    )
    arrays["actions"] = np.stack([f["action"].reshape(-1) for f in frames[1:]]).astype(np.float32)
    assert len(arrays["actions"]) == len(arrays["robot15"]) == len(observations)
    return arrays


def convert_shard(task: tuple[Path, Path]) -> list[dict]:
    """Convert every episode of one shard; return index rows for the episodes."""
    archive_path, output = task
    torch.set_num_threads(2)
    rows = []
    with tarfile.open(archive_path) as archive:
        current, frames, metadata = None, [], None

        def flush():
            if current is None:
                return
            episode_id = current.replace("/", "_")
            target = output / f"{episode_id}.npz"
            arrays = convert_episode(frames)
            np.savez(target, **arrays)
            rows.append(dict(episode=episode_id, file=target.name, seed=metadata["seed"],
                             steps=len(arrays["actions"]), success=metadata["success"]))

        # TAR members of an episode are stored contiguously in frame order.
        for member in archive:
            parts = member.name.split("/")
            if len(parts) != 3 or not parts[1].startswith("episode_"):
                continue
            episode = "/".join(parts[:2])
            if episode != current:
                flush()
                current, frames, metadata = episode, [], None
            if parts[2] == "episode.json":
                metadata = json.loads(archive.extractfile(member).read())
            elif parts[2].endswith(".npz"):
                frames.append(read_frame(archive, member))
        flush()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()

    shards = sorted(args.archives.glob("artifacts/trajectory/*/shard_*.tar"), key=lambda p: p.name)
    if len(shards) != 30:
        raise SystemExit(f"expected 30 shard archives, found {len(shards)}")
    args.output.mkdir(parents=True, exist_ok=False)

    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [row for part in pool.map(convert_shard, [(s, args.output) for s in shards]) for row in part]

    index = dict(rgb_size=RGB_SIZE, depth_size=DEPTH_SIZE, episodes=len(rows),
                 steps=sum(r["steps"] for r in rows), rows=rows)
    (args.output / "index.json").write_text(json.dumps(index, indent=1))
    print(f"episodes={len(rows)} steps={index['steps']}")


if __name__ == "__main__":
    main()
