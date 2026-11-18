"""Convert 512px expert shard archives into compact per-episode arrays.

Each shard TAR is read sequentially (nothing is extracted to disk). For every
episode we keep resized RGB of both cameras, low-resolution depth, the 17D robot
state (state.py), the raw GT needed for state baselines, and the actions aligned
with observations: ``actions[t]`` is the command applied at observation ``t``
(stored in frame t+1). The terminal observation (after the last action) is dropped.

``--archives`` accepts either layout written by data_collection.collect_dataset:
the published HF run (``<run>/artifacts/trajectory/*/shard_*.tar``) or a local,
unpublished collection output (``<output>/archives/shard_*.tar``).

Example (local collection with the shared camera):
    python -m visual.baseline.prepare_data \
        --archives outputs/visible_peg_shared_camera \
        --output data/visual_baseline/episodes_shared_camera
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

from visual.baseline.common import ROBOT_DIM, resize_depth, resize_rgb, robot_state

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
    arrays["robot_state"] = robot_state(
        gt("ee_pose"), gt("left_fingertip_pos"), gt("right_fingertip_pos"), gt("gripper_qpos")
    )
    arrays["actions"] = np.stack([f["action"].reshape(-1) for f in frames[1:]]).astype(np.float32)
    assert len(arrays["actions"]) == len(arrays["robot_state"]) == len(observations)
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


def find_shards(root: Path) -> list[Path]:
    """Shard TARs of a published HF run or of a local collection output, by name."""
    published = list(root.glob("artifacts/trajectory/*/shard_*.tar"))
    local = list(root.glob("archives/shard_*.tar"))
    if published and local:
        raise SystemExit(f"{root} contains both published and local shard layouts; pass one of them")
    return sorted(published or local, key=lambda path: path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--expected-shards", type=int, default=30,
                        help="900 episodes / 30 per shard in visible_peg configs")
    args = parser.parse_args()

    shards = find_shards(args.archives)
    if len(shards) != args.expected_shards:
        raise SystemExit(
            f"expected {args.expected_shards} shard archives under {args.archives}, found {len(shards)}; "
            "pass --expected-shards for a smaller collection"
        )
    args.output.mkdir(parents=True, exist_ok=False)

    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = [row for part in pool.map(convert_shard, [(s, args.output) for s in shards]) for row in part]

    index = dict(rgb_size=RGB_SIZE, depth_size=DEPTH_SIZE, robot_dim=ROBOT_DIM, episodes=len(rows),
                 steps=sum(r["steps"] for r in rows), rows=rows)
    (args.output / "index.json").write_text(json.dumps(index, indent=1))
    print(f"episodes={len(rows)} steps={index['steps']}")


if __name__ == "__main__":
    main()
