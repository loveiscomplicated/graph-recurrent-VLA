"""Evaluate saved Object21D estimators and measure bounded-window inference."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.state21 import load_dataset
from data_collection.state21_batches import PreparedWindows
from data_collection.state21_synthetic import STRESS_CASES, synthetic_episodes
from graph.state_estimation.models import build_model
from training.gnn.tasks import geometry_errors
from training.state21 import dataset_identity, prepare_dataset
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def _errors(prediction, target):
    results = {}
    for name, values in geometry_errors(prediction, target).items():
        if name.endswith("_m"):
            results[name + "m"] = values * 1000
        else:
            results[name] = values
    return results


def _percentiles(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        **{
            f"p{percentile}": float(np.percentile(array, percentile))
            for percentile in (90, 95, 99)
        },
    }


def summarize_errors(prediction, target):
    """Aggregate physical errors with quantiles and sign-invariant rotations."""
    return {
        name: _percentiles(values.cpu().numpy())
        for name, values in _errors(prediction, target).items()
    }


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _context(window):
    """Describe frame observability and label-derived motion for metric grouping only."""
    frame = window.sample.observation
    visible_vertices = sum(node.visible and node.is_vertex for node in frame.nodes)
    hole_vertices = sum(
        node.visible and node.is_vertex and node.object_id == 1 for node in frame.nodes
    )
    hole_support = any(node.visible and node.object_id == 1 for node in frame.nodes)
    hole_delta_mm = None
    if window.previous is not None:
        previous = window.previous
        world = window.sample.target[10:13] + torch.tensor(frame.tcp)
        previous_world = previous.target[10:13] + torch.tensor(previous.observation.tcp)
        gap = frame.index - previous.observation.index
        hole_delta_mm = float((world - previous_world).norm() * 1000 / gap)
    return {
        "episode_id": frame.episode_id,
        "frame_index": frame.index,
        "visible_vertices": visible_vertices,
        "hole_visible_vertices": hole_vertices,
        "hole_supported": hole_support,
        "hole_motion_mm_per_source_frame": hole_delta_mm,
        **(window.sample.diagnostics or {}),
    }


def _recovery(rows):
    """Mark loss/reappearance phases and elapsed source frames per episode."""
    episodes = {}
    for row in rows:
        episodes.setdefault(row["episode_id"], []).append(row)
    for episode in episodes.values():
        missing_since, reappeared_at = None, None
        ever_visible = False
        for row in sorted(episode, key=lambda item: item["frame_index"]):
            index = row["frame_index"]
            if not row["hole_supported"]:
                if missing_since is None:
                    missing_since = index
                reappeared_at = None
                row["visibility_phase"] = (
                    "occlusion" if ever_visible else "initially_hidden"
                )
                row["frames_since_visibility_loss"] = index - missing_since
            elif missing_since is not None:
                missing_since, reappeared_at = None, index
            if row["hole_supported"]:
                ever_visible = True
                row["visibility_phase"] = (
                    "before_occlusion" if reappeared_at is None else "after_occlusion"
                )
            row["frames_since_reappearance"] = (
                None if reappeared_at is None else index - reappeared_at
            )
            row.setdefault("frames_since_visibility_loss", None)


def _stratify(rows, names):
    groups = {
        "visibility_phase": {},
        "visible_vertices": {},
        "hole_motion": {},
        "tracker_drift_mm": {},
        "source_frames_since_visibility_loss": {},
        "source_frames_since_reappearance": {},
    }
    for row in rows:
        speed = row["hole_motion_mm_per_source_frame"]
        motion_bin = (
            "unknown"
            if speed is None
            else ("below_1_mm" if speed < 1 else "at_least_1_mm")
        )
        for axis, key in (
            ("visibility_phase", row["visibility_phase"]),
            ("visible_vertices", str(row["visible_vertices"])),
            ("hole_motion", motion_bin),
            ("tracker_drift_mm", row.get("injected_tracker_drift_mm")),
            (
                "source_frames_since_visibility_loss",
                row["frames_since_visibility_loss"],
            ),
            ("source_frames_since_reappearance", row["frames_since_reappearance"]),
        ):
            if key is not None:
                groups[axis].setdefault(str(key), []).append(row)
    return {
        axis: {
            key: {
                "count": len(values),
                **{name: _percentiles([row[name] for row in values]) for name in names},
            }
            for key, values in buckets.items()
        }
        for axis, buckets in groups.items()
    }


@torch.inference_mode()
def score(model, dataset, device, *, warmup=3):
    """Measure batch-one forward latency after graph construction and device transfer."""
    if type(warmup) is not int or warmup < 0:
        raise ValueError("evaluation.warmup must be nonnegative")
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows, latencies, predictions, targets = [], [], [], []
    for current_index, (frames, target, windows) in enumerate(
        dataset.batches(1, device)
    ):
        if current_index == 0:
            for _ in range(warmup):
                model(frames)
        _sync(device)
        started = perf_counter()
        prediction = model(frames)["object21"]
        _sync(device)
        elapsed = (perf_counter() - started) * 1000
        latencies.append(elapsed)
        result = prediction.cpu()
        target = target.cpu()
        predictions.append(result)
        targets.append(target)
        values = {name: value.item() for name, value in _errors(result, target).items()}
        history_frames = min(len(frames), getattr(model, "history", 1))
        spatial_frames = frames[-1:] if model.spatial_frame_limit == 1 else frames
        rows.append(
            {
                **_context(windows[0]),
                **values,
                "latency_ms": elapsed,
                "history_frames": history_frames,
                "spatial_frames": len(spatial_frames),
                "graph_edges": sum(
                    frame.edge_index.shape[1] for frame in spatial_frames
                ),
                "graph_nodes": sum(int(frame.valid.sum()) for frame in spatial_frames),
            }
        )

    _recovery(rows)
    prediction, target = torch.cat(predictions), torch.cat(targets)
    counts = {
        count: _percentiles(
            [row["latency_ms"] for row in rows if row["history_frames"] == count]
        )
        for count in sorted({row["history_frames"] for row in rows})
    }
    result = {
        "accuracy": summarize_errors(prediction, target),
        "stratified": _stratify(rows, tuple(_errors(prediction, target))),
        "latency_ms": _percentiles(latencies),
        "latency_ms_by_window": counts,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_graph_edges_per_prediction": float(
            np.mean([row["graph_edges"] for row in rows])
        ),
        "mean_graph_nodes_per_prediction": float(
            np.mean([row["graph_nodes"] for row in rows])
        ),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "window_limit": getattr(model, "history", 1),
        "count": len(rows),
    }
    return result, rows


def evaluate(config, *, stress=False):
    """Load the validation-selected model and save score, timing and per-frame CSV."""
    options = config["evaluation"]
    device = set_device(config["device"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    checkpoint_path = Path(options["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    episodes, report = load_dataset(config["data"])
    identifier = dataset_identity(config["data"], report)
    if (
        checkpoint["dataset_sha256"] != identifier
        or checkpoint["config"]["model"] != config["model"]
    ):
        raise ValueError(
            "Evaluation requires the same dataset and model recipe as the checkpoint"
        )
    model = build_model(**config["model"]).to(device)
    model.load_state_dict(checkpoint["best_model"])
    split = options.get("split", "validation")
    if stress:
        scenarios = {
            case: PreparedWindows(
                [
                    episode
                    for episode in synthetic_episodes(
                        options.get("stress_steps", 12), scenario=case
                    )
                    if episode.split == "test"
                ],
                config["data"]["history"],
                config["data"]["max_edge_m"],
            )
            for case in STRESS_CASES
        }
    else:
        scenarios = {split: prepare_dataset(episodes, config, split)}

    with ExperimentLogger(
        team="graph",
        model=config["model"]["name"],
        config=config,
        seed=checkpoint["config"]["seed"],
        job_type="evaluate",
    ) as experiment:
        output = Path(config["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        experiment.link_output(checkpoint_path, checkpoint=True)
        digest = file_sha256(checkpoint_path)
        experiment.record(
            dataset_sha256=identifier,
            checkpoint_sha256=digest,
            data_audit=report,
            device=str(device),
        )
        measurements = {}
        for name, dataset in scenarios.items():
            summary, rows = score(
                model, dataset, device, warmup=options.get("warmup", 3)
            )
            measurements[name] = summary
            experiment.log(summary, prefix=f"eval/{name}")
            csv_path = output / f"{name}.csv"
            columns = sorted(set().union(*(row.keys() for row in rows)))
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            experiment.link_output(csv_path, kind="metrics")
        summary_path = output / "evaluation.json"
        summary_path.write_text(
            json.dumps(
                {
                    "run_id": experiment.run_id,
                    "source_run_id": experiment.metadata.get("source_run_id"),
                    "git_commit": experiment.metadata.get("git_commit"),
                    "data": config["data"]["kind"],
                    "dataset_sha256": identifier,
                    "checkpoint_sha256": digest,
                    "model": config["model"]["name"],
                    "seed": checkpoint["config"]["seed"],
                    "split": "synthetic_stress_test" if stress else split,
                    "device": str(device),
                    "validation_review_required": bool(
                        report["near_duplicate_tcp_trajectories"]
                    ),
                    "tracker_training_lineage_verified": report["provenance"].get(
                        "tracker_training_or_selection_parents"
                    )
                    is not None,
                    "verified_visible_vertex_rows": report[
                        "verified_visible_vertex_rows"
                    ],
                    "distinct_size_targets": report["distinct_size_targets"],
                    "measurements": measurements,
                },
                indent=2,
            )
            + "\n"
        )
        experiment.link_output(summary_path, kind="metrics")
        print(summary_path)
        return summary_path
