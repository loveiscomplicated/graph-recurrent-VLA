"""Measure causal MLX SAM2 tracking through completed Metal masks."""

import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import mlx.core as mx
import numpy as np

from utils.experiment_logger import ExperimentLogger
from visual.sam2_mlx.object_stream import MLXObjectStream


def sha256(path):
    """Hash a local input without loading it into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def evaluate(config):
    """Replay fixed initial peg and box prompts and record frame latency."""
    clip = Path(config["clip"])
    checkpoint = Path(config["checkpoint"])
    warmup = int(config["warmup_frames"])
    repeats = int(config["repeats"])
    with np.load(clip) as recorded:
        rgb = recorded["rgb"]
        boxes = recorded["boxes"]
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("clip must contain uint8 RGB of shape [frames, height, width, 3]")
    if boxes.shape != (2, 4) or not np.isfinite(boxes).all():
        raise ValueError("clip must provide finite XYXY boxes for peg and box")
    if repeats < 1 or not 1 <= warmup < len(rgb):
        raise ValueError("repeats must be positive and warmup_frames must leave measured frames")

    with ExperimentLogger(team="visual", model="sam2_mlx", job_type="eval", config=config) as log:
        stream = MLXObjectStream(checkpoint)
        log.record(host=platform.platform(), mlx=mx.__version__, device=str(mx.default_device()),
                   checkpoint_sha256=sha256(checkpoint), clip_sha256=sha256(clip),
                   objects=2, image_size=512, spatial_memories=3, appearance_guard=False)
        runs = []
        for repeat in range(repeats):
            stream.reset()
            latency = []
            for index, frame in enumerate(rgb):
                started = perf_counter()
                if index == 0:
                    stream.start(frame, {1: boxes[0], 2: boxes[1]})
                else:
                    stream.track(frame, index)
                latency.append((perf_counter() - started) * 1000)
            median, p95 = np.percentile(latency[warmup:], [50, 95]).tolist()
            runs.append({"p50_ms": median, "p95_ms": p95, "frame_ms": latency})
            log.log({"p50_ms": median, "p95_ms": p95}, step=repeat)
            print(f"Replay {repeat + 1}: p50 {median:.2f} ms, p95 {p95:.2f} ms", flush=True)
        output = log.directory / "sam2_mlx_replay.json"
        output.write_text(json.dumps({"frames": len(rgb), "runs": runs}, indent=2) + "\n")
        log.link_output(output, kind="metrics")
        print(f"Results: {output}", flush=True)
