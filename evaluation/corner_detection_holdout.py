"""Benchmark the default frame detector on a frozen RGB-D holdout."""

import json
from collections import Counter
from pathlib import Path
import platform
import subprocess

import cv2
import numpy as np
import scipy
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger
from utils.timer import Timer
from visual import corner_detection
from visual.corner_detection.backends import torch_ops
from visual.corner_detection.evaluation import score, visible_reference

INPUTS = ("rgb", "depth", "intrinsic_cv", "extrinsic_cv")
GROUND_TRUTH = ("peg_pose", "box_pose", "hole_pose", "peg_half_sizes", "hole_radius")
FAMILIES = ("peg", "box_outer", "hole_rim")
THRESHOLDS = {"3mm": 0.003, "5mm": 0.005}


def load_episode(root, episode):
    """Verify the immutable episode archive before loading images and GT."""
    path = root / episode["file"]
    if file_sha256(path) != episode["sha256"]:
        raise ValueError(f"Holdout archive SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in (*INPUTS, "segmentation", *GROUND_TRUTH)}


def detector_source_hashes():
    """Identify the exact detector source, including working-tree files."""
    root = Path(corner_detection.__file__).parent
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob("*.py"))}


def hardware(device):
    """Identify both the accelerator and the CPU serving host-side stages."""
    if device.type == "cuda":
        cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
        cpu = next(line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name"))
        return {"gpu": torch.cuda.get_device_name(device), "cpu": cpu, "cuda": torch.version.cuda}
    model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return {"gpu": model, "cpu": model}


def cuda_graphs(device):
    """Read TorchDynamo's unique-graph counter for CUDA timing provenance."""
    if device.type == "cuda":
        from torch._dynamo.utils import counters
        return int(counters["stats"]["unique_graphs"])
    return None


def resident_inputs(arrays, index, device):
    """Move all four inputs before starting the detector timer."""
    return tuple(torch.as_tensor(arrays[key][index], device=device) for key in INPUTS)


def warmup(detector, root, episodes, samples_per_episode):
    """Compile and prime the device using frames spread across policies and phases."""
    for index, episode in enumerate(episodes):
        arrays = load_episode(root, episode)
        count = len(episode["frames"])
        fractions = (np.arange(samples_per_episode) + (index % 7 + 1) / 8) / samples_per_episode
        sample = sorted(set((fractions * count).astype(int)))
        for frame_index in sample:
            detector(*resident_inputs(arrays, frame_index, detector.device)).as_dict()


def warmup_extra_frames(detector, root, episodes, selected_frames):
    """Prime frame shapes that triggered compilation in an earlier replay."""
    by_key = {(episode["seed"], episode["mode"]): episode for episode in episodes}
    for selected in selected_frames:
        episode = by_key[selected["seed"], selected["mode"]]
        index = next(
            index for index, frame in enumerate(episode["frames"])
            if frame["step"] == selected["step"]
        )
        arrays = load_episode(root, episode)
        detector(*resident_inputs(arrays, index, detector.device)).as_dict()


def score_frame(detector, timer, arrays, index, actor_ids):
    """Time the native call and CPU export, then score against independent GT."""
    args = resident_inputs(arrays, index, detector.device)
    torch_ops.synchronize(detector.device)
    graphs_before = cuda_graphs(detector.device)
    timer.reset()
    with timer.measure("host_ready"):
        result = detector(*args)
        predictions = result.as_dict()["corners"]
    if result.xyz.device != detector.device:
        raise ValueError(f"Expected detector output on {detector.device}, got {result.xyz.device}")
    new_graphs = cuda_graphs(detector.device)
    compile_count = 0 if graphs_before is None else new_graphs - graphs_before
    elapsed = timer.summary()["host_ready"]["total_seconds"] * 1000

    sensor = {key: arrays[key][index] for key in (*INPUTS, "segmentation")}
    gt = {key: arrays[key][index] for key in GROUND_TRUTH}
    supported = visible_reference(gt, sensor, actor_ids)
    matches = {
        label: score(predictions, supported, [], threshold)
        for label, threshold in THRESHOLDS.items()
    }
    return {
        "detector_ms": result.elapsed_ms, "host_ready_ms": elapsed,
        "new_cuda_graphs": compile_count,
        "scores": matches,
        "predictions": [{"id": point["object_id"], "xyz": point["xyz"]} for point in predictions],
    }


def latency(samples):
    """Report milliseconds per frame from all timed calls in a subset."""
    return {"mean": float(np.mean(samples)),
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "max": float(np.max(samples))}


def summarize(rows):
    """Aggregate frame-level one-to-one matching and latency distributions."""
    summary = {
        "frames": len(rows),
        "episodes": len({(row["seed"], row["mode"]) for row in rows}),
        "detector_ms": latency([row["detector_ms"] for row in rows]),
        "host_ready_ms": latency([row["host_ready_ms"] for row in rows]),
        "new_cuda_graphs": sum(row["new_cuda_graphs"] for row in rows),
    }
    for threshold in THRESHOLDS:
        objects = {key: Counter() for key in ("peg", "box")}
        families = {family: Counter(supported=0, matched=0) for family in FAMILIES}
        errors = {family: [] for family in FAMILIES}
        for row in rows:
            for kind, result in zip(objects, row["scores"][threshold], strict=True):
                objects[kind].update(
                    predictions=result["predictions"],
                    matched=result["matched_3mm"],
                    unmatched=result["predictions"] - result["matched_3mm"],
                    frames_without_predictions=int(result["predictions"] == 0),
                )
                for family, counts in result["by_kind"].items():
                    families[family].update(supported=counts["visible"], matched=counts["matched"])
                for hit in result["matches"]:
                    errors[hit["kind"]].append(hit["error_mm"])
        for counts in objects.values():
            n = counts["predictions"]
            counts["precision"] = counts["matched"] / n if n else None
            counts["unmatched_per_frame"] = counts["unmatched"] / len(rows)
        for family, counts in families.items():
            n = counts["supported"]
            counts["recall"] = counts["matched"] / n if n else None
            distances = errors[family]
            counts["matched_error_p50_mm"] = float(np.median(distances)) if distances else None
        summary[threshold] = {"objects": objects, "families": families}
    return summary


def evaluate(config):
    """Replay the main preset or an explicit ablation on archived observations."""
    torch.set_num_threads(config["threads"])
    cv2.setNumThreads(config["threads"])
    device = torch_ops.resolve_device(config["device"])
    if device.type not in ("cuda", "mps"):
        raise ValueError("Holdout evaluation requires an explicit CUDA or MPS device")
    if config["warmup_frames_per_episode"] < 1:
        raise ValueError("warmup_frames_per_episode must be positive")
    options = {
        key: config[key] for key in (
            "plane_fit_method", "compile_cuda_validation", "batch_cuda_readbacks",
        )
        if key in config
    }
    detector = corner_detection.TorchCornerDetector.main(device=device, **options)
    source = Path(config["source"])
    manifest = json.loads(source.read_text())
    episodes = manifest["episodes"]
    provenance = {
        "source_sha256": file_sha256(source), "detector_sources": detector_source_hashes(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": str(torch.__version__), "opencv": cv2.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__, "hardware": hardware(device),
        "plane_fit_method": detector.plane_fit_method,
    }
    with ExperimentLogger(team="visual", model="corner_detection_holdout",
                          config={**config, **provenance}, job_type="evaluate",
                          seed=manifest["config"]["protocol_seed"]) as logger:
        print(f"Warmup on {device}: {len(episodes)} episodes", flush=True)
        warmup(detector, source.parent, episodes, config["warmup_frames_per_episode"])
        warmup_extra_frames(detector, source.parent, episodes, config.get("warmup_extra_frames", []))
        warmup_graphs = cuda_graphs(device)
        print(f"Warmup complete, CUDA graphs: {warmup_graphs}", flush=True)
        timer = Timer()
        rows = []
        for index, episode in enumerate(episodes, 1):
            arrays = load_episode(source.parent, episode)
            actor_ids = {int(key): value for key, value in episode["actor_ids"].items()}
            for frame_index, frame in enumerate(episode["frames"]):
                result = score_frame(detector, timer, arrays, frame_index, actor_ids)
                rows.append({"seed": episode["seed"], "mode": episode["mode"], **frame, **result})
            print(f"{device}: {index}/{len(episodes)} episodes, {len(rows)} frames", flush=True)

        groups = {"all": rows, "grasp": [row for row in rows if row["grasp"]]}
        mixed = [row for row in rows if row["mode"] == "expert_random"]
        groups.update({mode: [row for row in rows if row["mode"] == mode]
                       for mode in sorted({row["mode"] for row in rows})})
        groups["before_switch"] = [row for row in mixed if not row["after_switch"]]
        groups["after_switch"] = [row for row in mixed if row["after_switch"]]
        summaries = {name: summarize(group) for name, group in groups.items() if group}
        output = Path(config["output_root"]) / logger.run_id / "results.json"
        output.parent.mkdir(parents=True)
        write_json(output, {"config": config, "provenance": provenance, "summaries": summaries,
                            "warmup_cuda_graphs": warmup_graphs, "rows": rows})
        logger.link_output(output)
        logger.log(summaries, prefix="eval")
        print(output, flush=True)
