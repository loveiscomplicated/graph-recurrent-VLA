"""Cache causal TAPNext++ Peg surface tracks from recorded RGB-D episodes."""

import json
from pathlib import Path
from time import monotonic, sleep

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.tapnextpp_surface_tracking import (
    choose_points,
    detector_inputs,
    load_tracker,
    read_frames,
    resolve_detector_input_mode,
    track_surface_frame,
)
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector


def track_episode(frames, tracker, detector, config):
    """Start at the first usable Peg mask and preserve every source frame index."""
    count = config["surface_queries"]
    xy = np.full((len(frames), count, 2), np.nan, dtype=np.float32)
    visible = np.zeros((len(frames), count), dtype=bool)
    query = None
    anchor = -1
    state = None

    for index, frame in enumerate(frames):
        rgb, depth, _, _ = frame
        detection = detector(*detector_inputs(frame, detector.device, config["detector_inputs"]))
        mask = detection.masks[0].cpu().numpy().astype(bool)
        initial_query = None
        if state is None:
            try:
                initial_query = choose_points(mask, depth, count)
            except ValueError:
                continue
            query = initial_query
            anchor = index

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        xy[index], visible[index], state = track_surface_frame(
            tracker, config, bgr, initial_query, state, observation_mask=mask,
        )

    if query is None:
        query = np.full((count, 2), np.nan, dtype=np.float32)
    return dict(frame_indices=np.arange(len(frames)), anchor_frame=anchor,
                query_xy=query, surface_xy=xy, surface_visible=visible)


def available_episodes(source_root, episodes, wait_seconds):
    """Yield complete episodes while a transfer is delivering source frames."""
    if not wait_seconds:
        yield from episodes
        return

    pending = list(episodes)
    deadline = monotonic() + wait_seconds
    while pending:
        waiting = []
        for episode in pending:
            directory = source_root / episode["path"]
            if sum(1 for _ in directory.glob("frame_*.npz")) < episode["observations"]:
                waiting.append(episode)
                continue
            yield episode
            deadline = monotonic() + wait_seconds
        pending = waiting
        if pending:
            if monotonic() >= deadline:
                raise TimeoutError(f"Waiting for source frames for {len(pending)} episodes in {source_root}")
            sleep(5)


def build(config):
    """Write one track archive per episode and a checksum-linked index."""
    config = {
        "tap_input_resolution": 256,
        "tap_cuda_autocast": False,
        "tap_cache_cuda_autocast": False,
        "tap_mps_fp16_vit": False,
        "tap_mps_fp16_recurrent_mlp": False,
        "tap_mps_skip_past_query_embeddings": False,
        "tap_mps_compile_recurrent": False,
        **config,
    }
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    source_manifest = Path(config["source_manifest"])
    episodes = json.loads(source_manifest.read_text())["episodes"]
    if "episodes" in config:
        requested = set(config["episodes"])
        episodes = [episode for episode in episodes if episode["path"] in requested]
        if {episode["path"] for episode in episodes} != requested:
            raise ValueError("Configured episode names must match the source manifest")
    if not episodes:
        raise ValueError("The source manifest must contain at least one selected episode")
    wait_seconds = config.get("source_wait_timeout_seconds", 0)
    if type(wait_seconds) is not int or wait_seconds < 0:
        raise ValueError("source_wait_timeout_seconds must be a non-negative integer")

    tracker = load_tracker(config)
    detector = TorchCornerDetector.main(device=config["detector_device"])
    config["detector_inputs"] = resolve_detector_input_mode("auto", detector.device)
    with ExperimentLogger(team="visual", model="tapnextpp_cache", config=config,
                          job_type="evaluate") as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        source_hash = file_sha256(source_manifest)
        checkpoint_hash = file_sha256(Path(config["checkpoint"]))
        logger.record(source_manifest_sha256=source_hash, checkpoint_sha256=checkpoint_hash,
                      tracker_device=str(tracker.device), detector_device=str(detector.device))
        print(f"Output directory: {output}", flush=True)
        records = []
        for episode in available_episodes(source_manifest.parent, episodes, wait_seconds):
            source = source_manifest.parent / episode["path"]
            frames = read_frames({"source": source}, episode)
            if len(frames) != episode["observations"]:
                raise ValueError(f"Frame count differs in {source}")
            tracks = track_episode(frames, tracker, detector, config)
            path = output / f"{episode['path']}.npz"
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez_compressed(stream, **tracks)
            path.with_suffix(".tmp").replace(path)
            records.append(dict(source=episode["path"], seed=episode["seed"],
                                mode=episode["mode"], frames=len(frames),
                                anchor_frame=tracks["anchor_frame"], file=path.name,
                                sha256=file_sha256(path)))
            print(f"{len(records)}/{len(episodes)} {episode['path']} "
                  f"frames={len(frames)} anchor={tracks['anchor_frame']}", flush=True)

        index = output / "manifest.json"
        index.write_text(json.dumps(dict(source_manifest=str(source_manifest),
                                         source_manifest_sha256=source_hash,
                                         checkpoint_sha256=checkpoint_hash,
                                         episodes=records), indent=2) + "\n")
        logger.link_output(index)
