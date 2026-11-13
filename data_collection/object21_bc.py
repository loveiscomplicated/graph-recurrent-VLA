"""Export paired, frozen Object21 predictions and aligned expert actions."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import subprocess
import tarfile
import time

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from training.detected_corner_gnn import graph_batch
from training.fixed_id_gnn import load_temporal
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


ARCHITECTURES = ("transformer", "mamba")
FIELDS = ("robot_state", "expert_action", "object21_transformer", "object21_mamba")
_worker = None


def load_models(config, device):
    """Verify the selected GNN and load its matching frozen corner predictor."""
    models = {}
    sources = {}
    for name in ARCHITECTURES:
        paths = config["models"][name]
        if file_sha256(Path(paths["gnn"])) != paths["sha256"]:
            raise ValueError(f"Unexpected GNN checkpoint hash for {name}")
        checkpoint = torch.load(paths["gnn"], map_location="cpu", weights_only=False)
        model_config = deepcopy(checkpoint["config"])
        if model_config["seed"] != 0 or model_config["temporal"]["architecture"] != name:
            raise ValueError(f"Expected the recorded seed-0 {name} pair")
        if checkpoint["identity"]["threshold_mm"] != 30:
            raise ValueError("Expected the recorded 30 mm association threshold")
        model_config["temporal"]["checkpoint"] = paths["corner"]
        temporal, corner = load_temporal(model_config, device)
        gnn = VertexEdgeGatedMPNN(**model_config["gnn"]).to(device).eval().requires_grad_(False)
        gnn.load_state_dict(checkpoint["model"], strict=True)
        models[name] = (temporal, gnn, model_config["training"]["batch_size"])
        sources[name] = {
            "gnn_run_id": checkpoint["run_id"], "gnn_epoch": checkpoint["epoch"],
            "gnn_sha256": paths["sha256"], "corner_run_id": corner["run_id"],
            "corner_sha256": model_config["temporal"]["sha256"],
        }
    return models, sources


def read_observation(frame, detector):
    """Read the observed robot vector and detect Main camera corners once."""
    image = "observation/sensor_data/base_camera/"
    camera = "observation/sensor_param/base_camera/"
    rgb, depth = frame[image + "rgb"], frame[image + "depth"]
    if rgb.shape != (1, 512, 512, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected native 512x512 uint8 Main RGB")
    if depth.shape != (1, 512, 512, 1) or depth.dtype != np.int16:
        raise ValueError("Expected native 512x512 signed millimetre Main depth")
    result = detector(rgb[0], depth[0], frame[camera + "intrinsic_cv"][0],
                      frame[camera + "extrinsic_cv"][0])
    robot = np.concatenate((frame["observation/extra/tcp_pose"][0],
                            frame["observation/agent/qpos"][0, -2:])).astype(np.float32)
    return robot, result.xyz.cpu().numpy(), result.object_id.cpu().numpy()


def predict_graphs(gnn, track_frames, tcp, batch_size, device):
    """Build the original detector-only graphs and regress 21 values per frame."""
    predictions = []
    for start in range(0, len(track_frames), batch_size):
        rows = track_frames[start:start + batch_size]
        shape = (len(rows), max(1, max(len(row[0]) for row in rows)))
        xyz = np.zeros((*shape, 3), dtype=np.float32)
        kinds = np.full(shape, -1, dtype=np.int64)
        valid = np.zeros(shape, dtype=bool)
        observed = np.zeros(shape, dtype=bool)
        for index, (positions, labels, detections) in enumerate(rows):
            n = len(positions)
            xyz[index, :n] = positions
            kinds[index, :n] = labels
            valid[index, :n] = True
            observed[index, :n] = detections
        tensors = [torch.as_tensor(value, device=device)
                   for value in (xyz, kinds, valid, observed, tcp[start:start + len(rows)])]
        graph, features = graph_batch(*tensors)
        predictions.append(gnn(graph, features)["object21"].cpu().numpy())
    return np.concatenate(predictions)


@torch.no_grad()
def encode_episode(directory, episode, detector, models, device):
    """Encode o[t] against the expert action stored with o[t+1]."""
    frames = sorted(directory.glob("frame_*.npz"))
    expected = [f"frame_{i:06d}.npz" for i in range(episode["steps"] + 1)]
    if [path.name for path in frames] != expected or len(frames) < 2:
        raise ValueError(f"Missing or nonconsecutive observations in {directory}")
    trackers = {name: DetectedCornerTracker(models[name][0], 0.030, device)
                for name in ARCHITECTURES}
    tracks = {name: [] for name in ARCHITECTURES}
    robot_states, expert_actions = [], []
    for step, path in enumerate(frames):
        with np.load(path, allow_pickle=False) as frame:
            if step:
                expert_actions.append(frame["action"].reshape(-1))
            if step == len(frames) - 1:
                break
            robot, points, kinds = read_observation(frame, detector)
        robot_states.append(robot)
        for name, tracker in trackers.items():
            result = tracker.step(points, kinds, robot)
            tracks[name].append((result["positions"], [track.kind for track in tracker.tracks],
                                 result["observed"]))
    arrays = dict(robot_state=np.stack(robot_states), expert_action=np.stack(expert_actions))
    for name, (_, gnn, batch_size) in models.items():
        arrays[f"object21_{name}"] = predict_graphs(
            gnn, tracks[name], arrays["robot_state"][:, :3], batch_size, device
        )
    check_episode(arrays, episode["steps"])
    return arrays


def check_episode(arrays, steps):
    """Validate model output, state and action arrays before saving or merging."""
    for name, width in zip(FIELDS, (9, 7, 21, 21)):
        array = arrays[name]
        if array.shape != (steps, width) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name}: expected finite float32 [{steps},{width}]")
    if np.abs(arrays["expert_action"]).max() > 1:
        raise ValueError("Expert action is outside the normalized control range")


def write_npz(path, **arrays):
    """Publish a complete NumPy artifact atomically."""
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def split_positions(counts, seed):
    """Fix split membership by sorted episode position before inference."""
    if set(counts) != {"train", "validation", "test"} or min(counts.values()) < 1:
        raise ValueError("Expected positive train, validation and test episode counts")
    order = np.random.default_rng(seed).permutation(sum(counts.values()))
    positions = {}
    start = 0
    for split in ("train", "validation", "test"):
        stop = start + counts[split]
        positions[split] = sorted(order[start:stop].tolist())
        start = stop
    return positions


def pack_split(path, seeds, episodes, encoded):
    """Pack flat columns with T+1 style episode offsets for easy sequence access."""
    rows = []
    offsets = [0]
    for seed in seeds:
        steps = episodes[seed]["steps"]
        with np.load(encoded / f"{seed}.npz", allow_pickle=False) as data:
            arrays = {name: data[name] for name in FIELDS}
        check_episode(arrays, steps)
        rows.append(arrays)
        offsets.append(offsets[-1] + steps)
    columns = {name: np.concatenate([row[name] for row in rows]) for name in FIELDS}
    columns.update(episode_id=np.asarray([f"reset_seed_{seed}" for seed in seeds]),
                   reset_seed=np.asarray(seeds, dtype=np.int64),
                   episode_offsets=np.asarray(offsets, dtype=np.int64),
                   step=np.concatenate([np.arange(row["expert_action"].shape[0], dtype=np.int32)
                                        for row in rows]))
    write_npz(path, **columns)
    return {"file": path.name, "episodes": len(seeds), "action_rows": offsets[-1],
            "sha256": file_sha256(path)}


def package_release(output, split_files, dataset_manifest):
    """Bundle the three split files, manifest and a runnable data-loading example."""
    readme = output / "README.md"
    readme.write_text(
        """# Paired Object21 features for behavior cloning

Each split is a compressed NumPy file. All arrays load with `allow_pickle=False`.
The same expert observation/action pairs carry both frozen seed-0 model outputs.

```python
import numpy as np

with np.load("train.npz", allow_pickle=False) as data:
    start, stop = data["episode_offsets"][:2]
    robot = data["robot_state"][start:stop]
    geometry = data["object21_transformer"][start:stop]  # or object21_mamba
    x = np.concatenate((robot, geometry), axis=-1)
    y = data["expert_action"][start:stop]
```

- `robot_state` is float32 [N,9] with TCP world XYZ, wxyz quaternion, and two finger joints.
- `object21_transformer` and `object21_mamba` are float32 [N,21]. Layout is Peg
  position (0:3), Peg rotation (3:7), Peg +X tip minus Hole (7:10), Hole position
  (10:13), Hole rotation (13:17), Peg half extents (17:20), Hole radius (20:21).
  Positions use world axes and metres. Peg and Hole positions are relative to TCP.
  Quaternions use world-frame wxyz.
- `expert_action` is float32 [N,7], the normalized `pd_ee_delta_pose` command
  that was applied to the observation on that row.
- `episode_offsets[i:i+2]` gives a half-open row interval for `episode_id[i]`
  and `reset_seed[i]`. `step` is zero-based within the episode.

Use training data for normalization statistics. Validation and test are disjoint
by reset seed from training and from the original collection. Checkpoint hashes,
counts and per-split hashes are in `dataset_manifest.json`.

The Main camera input was native 512x512 RGB-D. The initial Main frame passed
the existing ends-and-grasp depth gate. Later frames can have natural occlusions.
Every trajectory reached official simulator success using the scripted expert.
This dataset measures action agreement on expert trajectories. Evaluate actual
insertion success separately in closed-loop simulator rollouts.
""",
        encoding="utf-8",
    )
    archive = output / "object21_bc_dataset.tar"
    temporary = archive.with_suffix(".partial")
    with tarfile.open(temporary, "w") as bundle:
        for file in (readme, dataset_manifest, *(output / row["file"] for row in split_files.values())):
            bundle.add(file, arcname=file.name)
    temporary.replace(archive)
    return archive


def ready_episodes(raw_root):
    """Expose only shards whose full raw validation already passed."""
    for verified in sorted((raw_root / "shards").glob("*/verification.json")):
        directory = verified.parent
        raw = json.loads((directory / "manifest.json").read_text())
        audit = json.loads(verified.read_text())
        if audit["status"] != "PASS" or raw["status"] != "COMPLETE":
            raise ValueError(f"Unverified shard {directory}")
        for row in raw["episodes"]:
            yield dict(row, path=f"shards/{directory.name}/{row['path']}")


def initialize_worker(config):
    """Load one isolated detector and tracker-model pair per worker process."""
    global _worker
    device = torch.device(config["device"])
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    models, _ = load_models(config, device)
    detector = TorchCornerDetector(config.get("detector_device", config["device"]))
    _worker = detector, models, device


def encode_to_disk(raw_root, row, encoded):
    """Encode one complete episode in a worker and atomically save its arrays."""
    seed = row["seed"]
    values = encode_episode(raw_root / row["path"], row, *_worker)
    write_npz(encoded / f"{seed}.npz", **values)
    return seed


def export(config):
    """Encode verified shards, then publish split files and a local manifest."""
    raw_root = Path(config["raw_root"]).resolve()
    output = Path(config["output"]).resolve()
    encoded = output / "encoded"
    device = torch.device(config["device"])
    positions = split_positions(config["split_counts"], config["split_seed"])
    _, sources = load_models(config, torch.device("cpu"))
    identity = dict(config=config, models=sources, split_positions=positions,
                    exporter_sha256=file_sha256(Path(__file__)))
    identity_file = output / "export_identity.json"
    if output.exists():
        if (output / "release.json").exists() or json.loads(identity_file.read_text()) != identity:
            raise ValueError("Existing export is complete or belongs to a different extraction")
    else:
        output.mkdir(parents=True)
        encoded.mkdir()
        write_json(identity_file, identity)
    completed = {int(path.stem) for path in encoded.glob("*.npz")}
    started = time.monotonic()
    manifest_path = raw_root / "dataset_manifest.json"
    pool_settings = dict(max_workers=config.get("workers", 1),
                         mp_context=multiprocessing.get_context("spawn"),
                         initializer=initialize_worker, initargs=(config,))
    with ExperimentLogger(team="integration", model="object21_bc_features", seed=config["split_seed"],
                          config=config, job_type="data_preparation") as logger:
        write_json(output / "run_identity.json", {"run_id": logger.run_id})
        with ProcessPoolExecutor(**pool_settings) as executor:
            while True:
                collection_finished = manifest_path.is_file()
                rows = list(ready_episodes(raw_root))
                if len({row["seed"] for row in rows}) != len(rows):
                    raise ValueError("Duplicate reset seeds in verified shards")
                pending = [row for row in rows if row["seed"] not in completed]
                jobs = [executor.submit(encode_to_disk, raw_root, row, encoded) for row in pending]
                for job in as_completed(jobs):
                    seed = job.result()
                    completed.add(seed)
                    write_json(output / "progress.json", {"episodes": len(completed),
                                                          "seconds": time.monotonic() - started})
                    print(f"Encoded {len(completed)} episodes (seed {seed})", flush=True)
                if collection_finished and not pending:
                    raw = json.loads(manifest_path.read_text())
                    if completed != {row["seed"] for row in raw["episodes"]}:
                        raise ValueError("Verified shards and raw manifest differ")
                    break
                if not pending:
                    time.sleep(10)
        if len(completed) != raw["episode_count"]:
            raise ValueError("Raw manifest and encoded episode counts differ")
        episodes = {row["seed"]: row for row in raw["episodes"]}
        sorted_seeds = sorted(episodes)
        if len(sorted_seeds) != sum(config["split_counts"].values()):
            raise ValueError("Split sizes must cover the full collection")
        splits = {name: [sorted_seeds[index] for index in indices]
                  for name, indices in positions.items()}
        split_files = {}
        for split, seeds in splits.items():
            path = output / f"{split}.npz"
            split_files[split] = pack_split(path, seeds, episodes, encoded)
        provenance = {
            "run_id": logger.run_id,
            "raw_run_id": raw["run_id"], "raw_manifest_sha256": file_sha256(manifest_path),
            "graph_dataset_sha256": raw["graph_dataset_sha256"],
            "original_collection_run_id": raw["original_collection_run_id"],
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "exporter_sha256": identity["exporter_sha256"], "device": str(device),
            "detector_device": config.get("detector_device", config["device"]),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "mamba-ssm", "numpy", "scipy", "opencv-python")},
            "checkpoints": sources, "threshold_mm": 30, "control_mode": "pd_ee_delta_pose",
            "camera": "base_camera", "resolution": [512, 512], "visibility_mode": "ends_and_grasp",
            "split_seed": config["split_seed"], "splits": splits, "files": split_files,
            "episode_lengths": dict(Counter(row["steps"] for row in raw["episodes"])),
        }
        dataset_manifest = output / "dataset_manifest.json"
        write_json(dataset_manifest, provenance)
        archive = package_release(output, split_files, dataset_manifest)
        logger.link_output(archive, kind="trajectory")
        logger.link_output(dataset_manifest, kind="episode_metadata")
        logger.log({"episodes": len(completed), "action_rows": sum(row["steps"] for row in episodes.values())})
    write_json(output / "release.json", {"run_id": logger.run_id, "file": archive.name,
                                        "sha256": file_sha256(archive)})
