"""Paired accuracy and latency replay for accelerated Peg rear-end recovery."""

import json
import platform
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.box_exterior_recovery import metrics
from evaluation.trajectory_rim_recovery import load_frame
from utils.experiment_logger import ExperimentLogger
from utils.timer import Timer
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.backends import torch_ops
from visual.corner_detection.evaluation import visible_reference


def _inputs(sensor):
    return tuple(sensor[key] for key in ("rgb", "depth", "intrinsic_cv", "extrinsic_cv"))


def _latency(samples):
    return {
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def _totals(rows, devices):
    result = {"frames": len(rows)}
    for device in devices:
        counts = Counter()
        for row in rows:
            counts.update(row[device]["metrics"])
        result[device] = {
            **counts,
            "latency": _latency([row[device]["elapsed_ms"] for row in rows]),
            "host_ready_latency": _latency([
                row[device]["elapsed_ms"] + row[device]["host_output_ms"] for row in rows
            ]),
        }
    return result


def evaluate(config):
    """Time normal execution and a separate stage-synchronized replay."""
    torch.set_num_threads(config["threads"])
    source = Path(config["source"])
    manifest = json.loads(source.read_text())
    frames = [row for row in manifest["rows"] if row["phase"] == "insert"]
    last = {(row["seed"], row["mode"]): row["frame"] for row in frames}
    accelerator = torch_ops.resolve_device(config["accelerator"])
    if accelerator.type not in ("mps", "cuda"):
        raise ValueError("accelerator must be mps or cuda[:index]")
    devices = ("cpu", str(accelerator))
    timers = {
        "cpu": Timer(),
        str(accelerator): Timer(synchronize=partial(torch_ops.synchronize, accelerator)),
    }
    resident = config.get("resident_geometry", False)
    detectors = {
        device: TorchCornerDetector.main(
            device=device if resident else "cpu", peg_rear_end_device=device
        )
        for device in devices
    }
    output_times = {device: Timer() for device in devices}

    with ExperimentLogger(
        team="visual",
        model="gpu_resident_corner_geometry" if resident else "peg_rear_end_accelerator",
        job_type="evaluation",
        config={
            **config, "source_sha256": file_sha256(source),
            "torch": str(torch.__version__), "platform": platform.platform(),
        },
        seed=manifest["config"]["seed_start"],
    ) as logger:
        for row in frames[-config["warmup_frames"]:]:
            sensor, _, _ = load_frame(source.parent, row)
            for detector in detectors.values():
                detector(*_inputs(sensor))

        rows = []
        for index, row in enumerate(frames):
            sensor, gt, _ = load_frame(source.parent, row)
            args = _inputs(sensor)
            refs = visible_reference(gt, sensor, {int(k): v for k, v in row["actor_ids"].items()})
            paired = {
                **{key: row[key] for key in ("seed", "mode", "frame", "sha256")},
                "last_insert": row["frame"] == last[row["seed"], row["mode"]],
            }
            order = devices if index % 2 == 0 else devices[::-1]
            for device in order:
                result = detectors[device](*args)
                assert result.xyz.device == detectors[device].device
                timer = output_times[device]
                timer.reset()
                with timer.measure("export"):
                    predictions = result.as_dict()["corners"]
                paired[device] = {
                    "elapsed_ms": result.elapsed_ms,
                    "host_output_ms": 1000 * timer.summary()["export"]["total_seconds"],
                    "metrics": metrics(predictions, refs, []),
                }
            rows.append(paired)
        totals = {
            "insertion": _totals(rows, devices),
            "last_insert": _totals([row for row in rows if row["last_insert"]], devices),
        }
        print("Unprofiled", json.dumps(totals), flush=True)

        for device, detector in detectors.items():
            detector.timer = timers[device]
        for index, row in enumerate(frames):
            sensor, _, _ = load_frame(source.parent, row)
            order = devices if index % 2 == 0 else devices[::-1]
            for device in order:
                detectors[device](*_inputs(sensor))
        stages = {device: timer.summary() for device, timer in timers.items()}
        output = Path(config["output_root"]) / logger.run_id / "scores.json"
        output.parent.mkdir(parents=True)
        write_json(output, {
            "source_sha256": file_sha256(source),
            "rows": rows, "totals": totals, "stage_timing_seconds": stages,
        })
        logger.link_output(output)
        logger.record(totals=totals, stage_timing_seconds=stages)
        print(output, json.dumps(stages), flush=True)
