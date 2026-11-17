"""Read approved raw episode frames without model or training dependencies."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    """Hash source bytes in bounded chunks for frame and manifest identity."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Episode:
    """An approved manifest entry backed by an identical committed receipt."""

    identity: str
    seed: int
    frames: tuple[Path, ...]


def read_episodes(root: str | Path, manifest_name: str = "manifest.json") -> list[Episode]:
    """Use only manifest-listed complete episodes; never glob rejected data.

    For downloaded TAR shards, extract into root and pass dataset_manifest.json.
    Paths are relative to root, e.g. shard_000000/episode_000000.
    """
    root = Path(root).expanduser().resolve()
    manifest_path = root / manifest_name
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"Dataset manifest must be COMPLETE: {manifest_path}")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Dataset manifest needs a nonempty episodes list: {manifest_path}")
    episodes = []
    paths, seeds = set(), set()
    for entry in entries:
        relative = Path(entry["path"])
        directory = (root / relative).resolve()
        if relative.is_absolute() or not directory.is_relative_to(root):
            raise ValueError(f"Episode path must stay within dataset root: {relative}")
        if any(part.endswith(".partial") or part.startswith("rejected_") for part in relative.parts):
            raise ValueError(f"Unapproved episode path: {relative}")
        receipt = json.loads((directory / "episode.json").read_text())
        # Aggregate manifests prefix path with a shard, unlike the local receipt.
        for name in ("seed", "observations", "steps"):
            value = entry.get(name)
            if type(value) is not int or value < 0 or receipt.get(name) != value:
                raise ValueError(f"Episode {relative}: invalid or inconsistent {name}")
        if entry["observations"] != entry["steps"] + 1:
            raise ValueError(f"Episode {relative}: observations must equal steps + 1")
        if entry["path"] in paths or entry["seed"] in seeds:
            raise ValueError("Duplicate episode paths or reset seeds in manifest")
        paths.add(entry["path"])
        seeds.add(entry["seed"])
        frames = tuple(directory / f"frame_{index:06d}.npz"
                       for index in range(entry["observations"]))
        if not all(path.is_file() for path in frames):
            raise ValueError(f"Episode {relative} has missing frames; extract its shard first")
        if not all(path.resolve().is_relative_to(root) for path in frames):
            raise ValueError(f"Episode {relative} contains external frame symlinks")
        episodes.append(Episode(entry["path"], entry["seed"], frames))
    return sorted(episodes, key=lambda episode: episode.identity)


def read_frame(path: Path, camera: str) -> tuple[dict, dict]:
    """Read one native camera's arrays and raw GT, preserving image dtypes."""
    image_prefix = f"observation/sensor_data/{camera}/"
    sensor_prefix = f"observation/sensor_param/{camera}/"
    keys = {"rgb": image_prefix + "rgb", "depth": image_prefix + "depth",
            "intrinsic_cv": sensor_prefix + "intrinsic_cv",
            "extrinsic_cv": sensor_prefix + "extrinsic_cv"}
    with np.load(path, allow_pickle=False) as frame:
        missing = set(keys.values()) - set(frame.files)
        if missing:
            raise ValueError(f"Frame {path} is missing camera fields: {sorted(missing)}")
        observation = {}
        for name, key in keys.items():
            value = frame[key]
            if value.shape[0] != 1:
                raise ValueError(f"Frame {path}: {key} needs one environment")
            observation[name] = value[0].copy()
        ground_truth = {key.removeprefix("ground_truth/"): frame[key].copy()
                        for key in frame.files if key.startswith("ground_truth/")}
    if not ground_truth:
        raise ValueError(f"Frame {path} has no GT; collect with save_step_gt=true")
    if observation["rgb"].shape != (512, 512, 3) or observation["rgb"].dtype != np.uint8:
        raise ValueError(f"Frame {path}: native uint8 RGB [512,512,3] required")
    if observation["depth"].shape not in ((512, 512), (512, 512, 1)):
        raise ValueError(f"Frame {path}: native millimetre depth [512,512,1] required")
    if observation["depth"].dtype != np.int16:
        raise ValueError(f"Frame {path}: raw signed int16 millimetre depth required")
    return observation, ground_truth


def canonical_hash(value) -> str:
    """Hash JSON settings without depending on dictionary insertion order."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
