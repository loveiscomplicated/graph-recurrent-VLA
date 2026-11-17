"""Adapt paired Object21 demonstrations to the Recurrent policy's feature order."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tarfile
import time

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.object21_bc import ready_episodes, write_npz
from data_collection.storage import write_json
from recurrent import features
from utils.experiment_logger import ExperimentLogger


def robot15(frame):
    """Read the Recurrent baseline's robot kinematics from a saved observation."""
    tcp = frame["ground_truth/ee_pose"][0]
    left = frame["ground_truth/left_fingertip_pos"][0]
    right = frame["ground_truth/right_fingertip_pos"][0]
    gripper = frame["ground_truth/gripper_qpos"][0]
    return np.concatenate((tcp[:3], features.canon_quat(tcp[3:]),
                           left - tcp[:3], right - tcp[:3], gripper))


def extract_episode(raw_root, row, cache):
    """Compute Robot15 for action-bearing steps while the detector export runs."""
    path = cache / f"{row['seed']}.npz"
    if path.is_file():
        return
    directory = raw_root / row["path"]
    states = []
    for step in range(row["steps"]):
        with np.load(directory / f"frame_{step:06d}.npz", allow_pickle=False) as frame:
            states.append(robot15(frame))
    states = np.stack(states)
    if states.shape != (row["steps"], 15) or states.dtype != np.float32 or not np.isfinite(states).all():
        raise ValueError(f"Invalid Robot15 for reset seed {row['seed']}")
    write_npz(path, robot15=states)


def adapt_split(source, output, cache, expected_seeds):
    """Add Robot15 and both State36 views without reordering episode or action rows."""
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "robot_state"}
        tracker_robot = data["robot_state"]
    seeds = arrays["reset_seed"].tolist()
    if seeds != expected_seeds:
        raise ValueError(f"Unexpected episode order in {source}")
    states = []
    for seed, left, right in zip(seeds, arrays["episode_offsets"], arrays["episode_offsets"][1:]):
        with np.load(cache / f"{seed}.npz", allow_pickle=False) as data:
            robot = data["robot15"]
        if robot.shape != (right - left, 15):
            raise ValueError(f"Robot/action frame counts differ for seed {seed}")
        np.testing.assert_allclose(robot[:, :3], tracker_robot[left:right, :3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(robot[:, 3:7], features.canon_quat(tracker_robot[left:right, 3:7]),
                                   atol=1e-6, rtol=0)
        np.testing.assert_allclose(robot[:, -2:], tracker_robot[left:right, -2:], atol=1e-6, rtol=0)
        states.append(robot)
    arrays["robot15"] = np.concatenate(states)
    for name in ("transformer", "mamba"):
        geometry = arrays[f"object21_{name}"]
        arrays[f"state36_{name}"] = np.concatenate((arrays["robot15"], geometry), axis=1)
    write_npz(output, **arrays)


def package(output, manifest):
    """Create one self-contained NPZ release for the Recurrent team."""
    (output / "README.md").write_text(
        """# Recurrent BC demonstrations with paired Graph features

Each split contains the same expert observation/action pairs for both frozen
seed-0 perception models. Arrays use the Recurrent baseline's column order.

```python
import numpy as np

with np.load("train.npz", allow_pickle=False) as data:
    start, stop = data["episode_offsets"][:2]
    x = data["state36_transformer"][start:stop]  # or state36_mamba
    y = data["expert_action"][start:stop]

# Legacy Recurrent models expect State38. The clean-observation convention
# appends obs_fresh=1 and normalized obs_age=0 to each row.
freshness = np.broadcast_to(np.array([1, 0], dtype=np.float32), (len(x), 2))
x38 = np.concatenate((x, freshness), axis=1)
```

- `robot15` uses TCP world XYZ, canonical wxyz TCP quaternion, left then right
  fingertip XYZ relative to TCP, and the two gripper joints. It is the recorded
  simulator robot kinematics used by the Recurrent baseline.
- `object21_transformer` and `object21_mamba` use Peg position (0:3), Peg
  rotation (3:7), Peg +X tip minus Hole (7:10), Hole position (10:13), Hole
  rotation (13:17), Peg half extents (17:20), and Hole radius (20:21).
  Positions and sizes are metres. Peg and Hole positions are relative to TCP
  in world axes. Quaternions are world-frame wxyz with nonnegative w.
- `state36_*` concatenates `robot15` and the corresponding predicted Object21.
  It excludes the two synthetic dropout channels. The constant State38 adapter
  indicates that an estimate is supplied each frame. It does not signal detector
  visibility, tracker confidence, or GT accuracy.
- `expert_action` is float32 [N,7], the normalized ManiSkill
  `pd_ee_delta_pose` command that follows the observation on that row.
- `episode_offsets[i:i+2]` gives the half-open row interval for `episode_id[i]`
  and `reset_seed[i]`. `step` is zero-based inside the episode.

Both camera sensors were native 512x512. The first Main frame passed the
depth-based peg ends-and-grasp gate. Later frames can have natural occlusions.
Every expert trajectory reached simulator success. Graph's development episodes
and all other seeds from the original collection are excluded. Normalize using
the training split only, and measure insertion success in separate closed-loop
rollouts. See `dataset_manifest.json` for checksums, splits and model provenance.
""",
        encoding="utf-8",
    )
    path = output / "recurrent_bc_dataset.tar"
    temporary = path.with_suffix(".partial")
    with tarfile.open(temporary, "w") as bundle:
        for name in ("README.md", "dataset_manifest.json", *(row["file"] for row in manifest["files"].values())):
            bundle.add(output / name, arcname=name)
    temporary.replace(path)
    return path


def export(config):
    """Assemble a Recurrent-compatible dataset after the paired Object21 release."""
    raw = Path(config["raw_root"]).resolve()
    source = Path(config["object21_root"]).resolve()
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "release.json").exists():
        raise ValueError("Recurrent BC release is already complete")
    cache = output / "robot_cache"
    cache.mkdir(exist_ok=True)
    identity = {"raw_root": str(raw), "object21_root": str(source),
                "adapter_sha256": file_sha256(Path(__file__))}
    identity_path = output / "export_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Robot15 cache was created by a different export")
    write_json(identity_path, identity)
    with ThreadPoolExecutor(max_workers=config.get("workers", 2)) as pool:
        while True:
            ready = (source / "release.json").is_file()
            jobs = [pool.submit(extract_episode, raw, row, cache) for row in ready_episodes(raw)
                    if not (cache / f"{row['seed']}.npz").is_file()]
            for job in jobs:
                job.result()
            if jobs:
                print(f"Robot15 ready for {len(list(cache.glob('*.npz')))} episodes", flush=True)
            if ready:
                break
            time.sleep(15)
    original = json.loads((source / "dataset_manifest.json").read_text())
    with ExperimentLogger(team="integration", model="recurrent_bc_state36", config=config,
                          seed=original["split_seed"], job_type="data_preparation") as logger:
        files = {}
        for split, info in original["files"].items():
            source_file = source / info["file"]
            if file_sha256(source_file) != info["sha256"]:
                raise ValueError(f"Object21 source hash mismatch: {split}")
            result = output / source_file.name
            adapt_split(source_file, result, cache, original["splits"][split])
            files[split] = dict(info, sha256=file_sha256(result))
        manifest = dict(original, run_id=logger.run_id, files=files,
                        object21_source_run_id=original["run_id"],
                        object21_source_manifest_sha256=file_sha256(source / "dataset_manifest.json"),
                        recurrent_features_sha256=file_sha256(Path(features.__file__)),
                        state36_layout=features.FEATURE_LAYOUT[:-2],
                        optional_state38_suffix={"obs_fresh": 1, "obs_age": 0})
        manifest_path = output / "dataset_manifest.json"
        write_json(manifest_path, manifest)
        archive = package(output, manifest)
        logger.link_output(archive, kind="trajectory")
        logger.link_output(manifest_path, kind="episode_metadata")
        logger.log({"episodes": sum(row["episodes"] for row in files.values()),
                    "action_rows": sum(row["action_rows"] for row in files.values())})
    write_json(output / "release.json", {"run_id": logger.run_id, "file": archive.name,
                                        "sha256": file_sha256(archive)})
