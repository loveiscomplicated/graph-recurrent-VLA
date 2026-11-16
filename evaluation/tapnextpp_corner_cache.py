"""Add frame-wise RGB-D corner detections to an immutable TAPNext++ cache."""

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.corner_detection_holdout import (
    GROUND_TRUTH,
    INPUTS,
    THRESHOLDS,
    cuda_graphs,
    detector_source_hashes,
    hardware,
    summarize,
)
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import score, visible_reference


def load_frame(path, digest):
    """Read detector inputs and scoring-only references from a verified raw frame."""
    if file_sha256(path) != digest:
        raise ValueError(f"Raw frame checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as frame:
        sensor = {
            name: frame[f"observation/sensor_data/base_camera/{name}"][0]
            for name in ("rgb", "depth", "segmentation")
        }
        sensor.update({
            name: frame[f"observation/sensor_param/base_camera/{name}"][0]
            for name in ("intrinsic_cv", "extrinsic_cv")
        })
        gt = {name: frame[f"ground_truth/{name}"][0] for name in GROUND_TRUTH}
    return sensor, gt


def detect_frame(detector, sensor):
    """Measure a GPU-resident detector call and the standard host export."""
    inputs = tuple(torch.as_tensor(sensor[key], device=detector.device) for key in INPUTS)
    torch.cuda.synchronize(detector.device)
    before = cuda_graphs(detector.device)
    started = perf_counter()
    result = detector(*inputs)
    predictions = result.as_dict()["corners"]
    elapsed = (perf_counter() - started) * 1000
    return predictions, dict(detector_ms=result.elapsed_ms, host_ready_ms=elapsed,
                             new_cuda_graphs=cuda_graphs(detector.device) - before)


def augment_episode(detector, episode, actor_ids, source, track_path, expected_hash):
    """Copy the original track arrays and append flat, frame-indexed corner arrays."""
    if file_sha256(track_path) != expected_hash:
        raise ValueError(f"TAPNext++ cache checksum mismatch: {track_path}")
    count = episode["observations"]
    with np.load(track_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if not np.array_equal(arrays["frame_indices"], np.arange(count)):
        raise ValueError(f"Track frame indices do not match {episode['path']}")

    offsets = [0]
    corners = []
    rows = []
    for index, digest in enumerate(episode["frame_sha256"]):
        path = source / episode["path"] / f"frame_{index:06d}.npz"
        sensor, gt = load_frame(path, digest)
        predictions, timing = detect_frame(detector, sensor)
        corners.extend(predictions)
        offsets.append(len(corners))

        references = visible_reference(gt, sensor, actor_ids)
        scores = {name: score(predictions, references, [], threshold)
                  for name, threshold in THRESHOLDS.items()}
        rows.append(dict(seed=episode["seed"], mode=episode["mode"], scores=scores, **timing))

    if len(rows) != count:
        raise ValueError(f"Raw frame count does not match {episode['path']}")
    arrays.update(
        corner_offsets=np.asarray(offsets, dtype=np.int64),
        corner_object_id=np.asarray([p["object_id"] for p in corners], dtype=np.int8),
        corner_uv=np.asarray([p["uv"] for p in corners], dtype=np.float32).reshape(-1, 2),
        corner_world_xyz=np.asarray([p["xyz"] for p in corners], dtype=np.float32).reshape(-1, 3),
        corner_camera_xyz=np.asarray([p["camera_xyz"] for p in corners], dtype=np.float32).reshape(-1, 3),
        corner_method=np.asarray([p["method"] for p in corners], dtype=str),
        corner_detector_ms=np.asarray([row["detector_ms"] for row in rows], dtype=np.float32),
    )
    return arrays, rows


def build(config):
    """Build a new episode cache and benchmark the exact detections it contains."""
    torch.set_num_threads(config.get("threads", 2))
    cv2.setNumThreads(1)
    source_path = Path(config["source_manifest"])
    track_path = Path(config["track_manifest"])
    source = json.loads(source_path.read_text())
    tracks = json.loads(track_path.read_text())
    ids = json.loads(Path(config["actor_ids"]).read_text())
    source_hash = file_sha256(source_path)
    if source_hash != tracks["source_manifest_sha256"] or source_hash != ids["source_manifest_sha256"]:
        raise ValueError("Tracks and verified actor IDs must refer to the same raw dataset")
    by_path = {entry["source"]: entry for entry in tracks["episodes"]}
    episodes = source["episodes"]
    requested = config.get("episodes")
    if requested is not None:
        episodes = [row for row in episodes if row["path"] in requested]
        if set(requested) != {row["path"] for row in episodes}:
            raise ValueError("Configured episodes must match source manifest paths")
    if not episodes or any(row["path"] not in by_path for row in episodes):
        raise ValueError("Every selected source episode must have a track cache")

    detector = TorchCornerDetector.main(device=config["device"])
    with ExperimentLogger(team="visual", model="tapnextpp_corner_cache", config=config,
                          job_type="evaluate") as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        track_hash = file_sha256(track_path)
        detector_hashes = detector_source_hashes()
        logger.record(source_manifest_sha256=source_hash, track_manifest_sha256=track_hash,
                      detector_source_sha256=detector_hashes, hardware=hardware(detector.device),
                      torch_version=str(torch.__version__))
        print(f"Output directory: {output}", flush=True)
        # Prime early, middle, and final observations of each policy before timing.
        first_by_mode = {}
        for episode in episodes:
            first_by_mode.setdefault(episode["mode"], episode)
        for episode in first_by_mode.values():
            for index in sorted({0, episode["steps"] // 2, episode["steps"]}):
                frame = source_path.parent / episode["path"] / f"frame_{index:06d}.npz"
                sensor, _ = load_frame(frame, episode["frame_sha256"][index])
                detect_frame(detector, sensor)

        records = []
        metrics_by_mode = defaultdict(list)
        for episode in episodes:
            entry = by_path[episode["path"]]
            actor_ids = {int(key): value for key, value in ids["actor_ids"][str(episode["seed"])].items()}
            arrays, rows = augment_episode(detector, episode, actor_ids, source_path.parent,
                                           track_path.parent / entry["file"], entry["sha256"])
            path = output / entry["file"]
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            path.with_suffix(".tmp").replace(path)
            records.append({**entry, "source_track_sha256": entry["sha256"], "sha256": file_sha256(path)})
            metrics_by_mode[episode["mode"]].extend(rows)
            summary = summarize(rows)
            with (output / "episode_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(source=episode["path"], mode=episode["mode"], **summary)) + "\n")
            print(f"{len(records)}/{len(episodes)} {episode['path']} frames={len(rows)} "
                  f"corners={len(arrays['corner_object_id'])} p50={summary['detector_ms']['p50']:.2f} ms", flush=True)

        index = output / "manifest.json"
        metrics = output / "metrics.json"
        write_json(index, dict(source_manifest_sha256=source_hash, track_manifest_sha256=track_hash,
                               detector_source_sha256=detector_hashes, episodes=records))
        write_json(metrics, dict(all=summarize([r for rows in metrics_by_mode.values() for r in rows]),
                                 by_mode={mode: summarize(rows) for mode, rows in metrics_by_mode.items()}))
        for path in (index, metrics, output / "episode_metrics.jsonl"):
            logger.link_output(path)
