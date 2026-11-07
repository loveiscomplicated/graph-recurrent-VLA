"""Paired frozen GNN/rule/GRU diagnostics and measured streaming performance."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal.frozen_episodes import (
    FrozenEpisodeDataset,
    infer_observations,
)
from evaluation.gnn.temporal_gate import recovery_events, update_audit
from evaluation.gnn.temporal_memory import error_summary, replay_memory, scoring_groups
from graph.temporal.hole_gru import StreamingHoleEstimator, TemporalHoleGRU
from graph.temporal.observations import observation_features, temporal_inputs
from graph.utils.hole_memory import (
    HolePositionMemory,
    MemorySettings,
    ObservationQuality,
)
from training.gnn.temporal.config import resolve_gru_config
from training.gnn.temporal.frozen_source import prepare_sequences
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


@torch.inference_mode()
def replay_gru(model, data, device):
    """Check sequence/streaming equivalence on every frame with independent resets."""
    dataset = FrozenEpisodeDataset(data)
    estimator = StreamingHoleEstimator(model)
    world, hole_only, consistent = [], [], []
    maximum_difference = 0.0
    for index, rows in enumerate(dataset.indices):
        observations, _ = dataset[index]
        expected, expected_hidden = model(observations.to(device).unsqueeze(0))
        estimator.reset()
        episode_world = []
        for row in rows:
            prediction, tcp, quality = (
                torch.as_tensor(data[name][row : row + 1], dtype=torch.float32, device=device)
                for name in ("prediction", "tcp_world", "features")
            )
            result = estimator.step(prediction, tcp, quality)
            episode_world.append(result["hole_world"])
            hole_only.append(result["object21"].cpu().numpy())
            consistent.append(result["consistent_object21"].cpu().numpy())
        streamed = torch.stack(episode_world, dim=1)
        torch.testing.assert_close(streamed, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(estimator.hidden, expected_hidden, atol=1e-5, rtol=1e-4)
        maximum_difference = max(maximum_difference, float((streamed - expected).abs().max()))
        world.append(streamed[0].cpu().numpy())
    return {
        "world": np.concatenate(world),
        "object21": np.concatenate(hole_only),
        "consistent_object21": np.concatenate(consistent),
        "streaming_max_difference_m": maximum_difference,
    }


def absence_diagnostics(data, absent, minimum_length):
    """Mark complete absent runs and their causal ages for scoring only."""
    age = np.zeros(len(absent), dtype=np.int64)
    long_runs = np.zeros(len(absent), dtype=bool)
    events = []
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        start = 0
        while start < len(rows):
            if not absent[rows[start]]:
                start += 1
                continue
            stop = start + 1
            while stop < len(rows) and absent[rows[stop]]:
                stop += 1
            segment = rows[start:stop]
            age[segment] = np.arange(1, len(segment) + 1)
            if len(segment) >= minimum_length:
                long_runs[segment] = True
            events.append(
                {
                    "episode": str(episode),
                    "start_frame": int(data["frame"][segment[0]]),
                    "length_frames": len(segment),
                }
            )
            start = stop
    return age, long_runs, events


def temporal_metrics(data, world, error, baseline_error, threshold):
    """Report episode drift, equal-progress bins and fixed bad-GNN-start cohorts."""
    episodes, bins, recovery = [], [[] for _ in range(4)], []
    target_world = data["target"][:, 10:13] + data["tcp_world"]
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        frames = data["frame"][rows]
        local_error = error[rows]
        displacement_error = (world[rows] - world[rows[0]]) - (
            target_world[rows] - target_world[rows[0]]
        )
        episodes.append(
            {
                "episode": str(episode),
                "frames": len(rows),
                "first_error_mm": float(local_error[0]),
                "last_error_mm": float(local_error[-1]),
                "error_slope_mm_per_frame": float(np.polyfit(frames, local_error, 1)[0])
                if len(rows) > 1
                else None,
                "displacement_drift": error_summary(
                    np.linalg.norm(displacement_error, axis=1) * 1000
                ),
            }
        )
        for bucket, segment in zip(bins, np.array_split(local_error, 4)):
            bucket.extend(segment.tolist())
        if baseline_error[rows[0]] > threshold:
            recovered = rows[local_error <= threshold]
            recovery.append(
                {
                    "episode": str(episode),
                    "latency_frames": int(data["frame"][recovered[0]] - frames[0])
                    if len(recovered)
                    else None,
                }
            )
    return {
        "episode_traces": episodes,
        "progress_quarters": [error_summary(values) for values in bins],
        "bad_gnn_initialization_recovery": recovery,
        "bad_gnn_initialization_unrecovered": sum(
            row["latency_frames"] is None for row in recovery
        ),
    }


def synchronize(device):
    """Include asynchronous accelerator work in wall-clock latency."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def measure_callback(callback, device, warmup, steps):
    """Time batch-one resident inference, including Python and validation overhead."""
    for _ in range(warmup):
        callback()
    synchronize(device)
    baseline = None
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(steps):
        synchronize(device)
        started = perf_counter()
        callback()
        synchronize(device)
        durations.append((perf_counter() - started) * 1000)
    return {
        "steps": steps,
        "warmup_steps": warmup,
        "batch_size": 1,
        "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)),
        "p90_ms": float(np.percentile(durations, 90)),
        "cuda_baseline_allocated_bytes": baseline,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "cuda_peak_incremental_bytes": torch.cuda.max_memory_allocated(device) - baseline
        if device.type == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
    }


@torch.inference_mode()
def benchmark(model, frozen, sample, data, device, settings, rule_settings):
    """Measure separate components and GNN→observations→GRU pipeline on one frame.

    These repeated-frame latency probes are not accuracy evaluation. CUDA counters
    are process allocated/reserved peaks, with both models resident, not nvidia-smi
    usage. The incremental peak subtracts post-warmup live allocations.
    """
    model.eval()
    graph = sample.graph.to(device)
    # The loader's original inventory order need not match sorted replay order.
    row = int(np.flatnonzero(data["sample_ids"] == sample.sample_ids[0])[0])
    prediction = torch.as_tensor(data["prediction"][row : row + 1], device=device)
    tcp = torch.as_tensor(data["tcp_world"][row : row + 1], device=device)
    quality = torch.as_tensor(data["features"][row : row + 1], device=device)
    inputs = temporal_inputs(prediction, tcp, quality)
    hidden = None
    estimator = StreamingHoleEstimator(model)
    memory = HolePositionMemory(rule_settings)
    rule_quality = ObservationQuality(
        int(data["quality"][row, 0]), float(data["quality"][row, 1]), float(data["quality"][row, 2])
    )
    frame = 0

    def gru_step():
        nonlocal hidden
        _, hidden = model.step(inputs, hidden)

    def rule_step():
        nonlocal frame
        memory.step(frame, data["prediction"][row, 10:13], data["tcp_world"][row], rule_quality)
        frame += 1

    def pipeline_step():
        output = frozen(graph)["object21"]
        estimator.step(output, graph.tcp_xyz_world, observation_features(graph))

    callbacks = {
        "gnn": lambda: frozen(graph),
        "rule_only": rule_step,
        "gru_only": gru_step,
        "gnn_gru_pipeline": pipeline_step,
    }
    measured = {}
    for name, callback in callbacks.items():
        timing_device = torch.device("cpu") if name == "rule_only" else device
        measured[name] = measure_callback(
            callback, timing_device, settings["latency_warmup_steps"], settings["latency_steps"]
        )
    return {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "gru_parameters": sum(p.numel() for p in model.parameters()),
        "frozen_gnn_parameters": sum(p.numel() for p in frozen.parameters()),
        "scope": "batch-one repeated frame; resident tensors; no IO/detection/robot assembly",
        "cuda_memory_scope": "process allocator peaks with both models resident; after warmup",
        "measurements": measured,
    }


def save_plots(output, data, errors, world, maximum):
    """Plot paired errors, world XYZ and observed support on the same causal time axis."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for number, episode in enumerate(np.unique(data["episode"])[:maximum]):
        mask = data["episode"] == episode
        frames = data["frame"][mask]
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
        for name, error in errors.items():
            axes[0].plot(frames, error[mask], label=name)
        axes[0].set_ylabel("Hole error (mm)")
        axes[0].set_title(str(episode))
        axes[0].legend()
        target_world = data["target"][mask, 10:13] + data["tcp_world"][mask]
        for coordinate, label in enumerate("xyz"):
            axes[1].plot(frames, world[mask, coordinate], label=f"GRU {label}")
            axes[1].plot(frames, target_world[:, coordinate], "--", label=f"GT {label}")
        axes[1].set_ylabel("World position (m)")
        axes[1].legend(ncol=3)
        axes[2].plot(frames, data["features"][mask, 0], label="BOX count")
        axes[2].plot(frames, data["features"][mask, 8], label="HOLE count")
        axes[2].set_ylabel("Observed vertices")
        axes[2].set_xlabel("Frame")
        axes[2].legend()
        for axis in axes:
            axis.fill_between(
                frames,
                0,
                1,
                where=data["features"][mask, 8] == 0,
                transform=axis.get_xaxis_transform(),
                alpha=0.12,
                color="gray",
            )
        figure.tight_layout()
        figure.savefig(output / f"episode_trace_{number:03d}.png", dpi=140)
        plt.close(figure)


def score_predictions(data, learned, rule, settings):
    """Consume GT/visibility only after all three causal predictions are fixed."""
    predictions = {"gnn": data["prediction"], "rule": rule["object21"], "gru": learned["object21"]}
    errors = {
        name: np.linalg.norm(value[:, 10:13] - data["target"][:, 10:13], axis=1) * 1000
        for name, value in predictions.items()
    }
    groups = scoring_groups(data, settings["visibility_labels"])
    groups["no_hole_vertices"] = data["features"][:, 8] == 0
    groups["no_box_or_hole_vertices"] = data["quality"][:, 0] == 0
    absence_events = {}
    for name in ("no_hole_vertices", "no_box_or_hole_vertices", "strong_occlusion_proxy"):
        if name not in groups:
            continue
        age, long_runs, events = absence_diagnostics(
            data, groups[name], settings["long_occlusion_frames"]
        )
        groups[name + "_long_runs"] = long_runs
        groups[name + "_long_age"] = age >= settings["long_occlusion_frames"]
        absence_events[name] = events
    summary = {
        "split": "temporal_test_original_gnn_validation",
        "frames": len(data["frame"]),
        "episodes": len(np.unique(data["episode"])),
        "visibility_labels_available": settings["visibility_labels"] is not None,
        "streaming_max_difference_m": learned["streaming_max_difference_m"],
        "methods": {},
    }
    first = np.r_[True, data["episode"][1:] != data["episode"][:-1]]
    recovery = {}
    for name, prediction in predictions.items():
        error = errors[name]
        world = prediction[:, 10:13] + data["tcp_world"]
        method = {
            "groups": {group: error_summary(error[mask]) for group, mask in groups.items()},
            "episode_balanced_mean_mm": float(
                np.mean(
                    [
                        error[data["episode"] == episode].mean()
                        for episode in np.unique(data["episode"])
                    ]
                )
            ),
            "first_frame_error": error_summary(error[first]),
            "first_frame_failures": int(np.sum(first & (error > settings["bad_error_mm"]))),
            "endpoint_vector_error": error_summary(
                np.linalg.norm(prediction[:, 7:10] - data["target"][:, 7:10], axis=1) * 1000
            ),
            "update_audit": update_audit(data, world, error, settings["bad_error_mm"]),
            "temporal": temporal_metrics(
                data, world, error, errors["gnn"], settings["bad_error_mm"]
            ),
        }
        recovery[name] = {}
        for group in absence_events:
            recovery[name][group] = recovery_events(
                data, error, settings["bad_error_mm"], ~groups[group]
            )
        summary["methods"][name] = method
    original = data["prediction"]
    consistent = learned["consistent_object21"]
    original_tip = original[:, 7:10] + original[:, 10:13]
    summary["assembly_ablation"] = {
        "hole_only_endpoint_vector_error": summary["methods"]["gru"]["endpoint_vector_error"],
        "consistent_endpoint_vector_error": error_summary(
            np.linalg.norm(consistent[:, 7:10] - data["target"][:, 7:10], axis=1) * 1000
        ),
        "consistent_tip_preservation_error": error_summary(
            np.linalg.norm(consistent[:, 7:10] + consistent[:, 10:13] - original_tip, axis=1) * 1000
        ),
        "hole_only_tip_shift": error_summary(
            np.linalg.norm(
                learned["object21"][:, 7:10] + learned["object21"][:, 10:13] - original_tip, axis=1
            )
            * 1000
        ),
    }
    return summary, errors, {"absence_runs": absence_events, "recovery": recovery}


def evaluate(config):
    """Evaluate a selected GRU without fitting, searching thresholds or reading GNN test."""
    effective = resolve_gru_config(config)
    checkpoint = Path(effective["gru_checkpoint"]).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("kind") != "temporal_hole_gru":
        raise ValueError("gru_checkpoint must be a temporal_hole_gru best.pt")
    device = set_device(effective["device"])
    frozen, loaders, identity = prepare_sequences(effective, device, ("test",))
    for name in (
        "source_checkpoint_sha256",
        "split_manifest_sha256",
        "gru",
        "feature_names",
        "observation_contract",
    ):
        if effective[name] != payload["config"][name]:
            raise ValueError(f"Evaluation differs from trained GRU provenance: {name}")
    if identity["dataset_sha256"] != payload["dataset_sha256"]:
        raise ValueError("GRU training and evaluation dataset identities differ")
    model = TemporalHoleGRU(**effective["gru"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    settings = effective["evaluation"]
    rule_settings = MemorySettings(**effective["rule_memory"])
    with ExperimentLogger(
        team="graph",
        model="temporal_hole_gru",
        config=effective,
        seed=effective["seed"],
        job_type="evaluate",
    ) as experiment:
        output = Path(effective["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        experiment.link_output(effective["checkpoint"], checkpoint=True)
        experiment.link_output(checkpoint, checkpoint=True)
        data = infer_observations(frozen, loaders["test"], device)
        learned = replay_gru(model, data, device)
        rule = replay_memory(
            data["prediction"],
            data["tcp_world"],
            data["quality"],
            data["episode"],
            data["frame"],
            rule_settings,
        )
        summary, errors, events = score_predictions(data, learned, rule, settings)
        performance = benchmark(
            model, frozen, loaders["test"].dataset[0], data, device, settings, rule_settings
        )
        summary["performance"] = performance
        np.savez_compressed(
            output / "predictions.npz",
            **data,
            gru_world=learned["world"],
            gru_object21=learned["object21"],
            consistent_object21=learned["consistent_object21"],
            rule_object21=rule["object21"],
        )
        save_plots(output, data, errors, learned["world"], settings["plot_episodes"])
        write_json(output / "summary.json", summary)
        write_json(output / "events.json", events)
        write_json(output / "effective_config.json", effective)
        write_json(
            output / "provenance.json",
            {
                "gru_checkpoint_sha256": file_sha256(checkpoint),
                "source_checkpoint_sha256": effective["source_checkpoint_sha256"],
                "split_manifest_sha256": effective["split_manifest_sha256"],
                "visibility_sha256": file_sha256(Path(settings["visibility_labels"]))
                if settings["visibility_labels"]
                else None,
            },
        )
        experiment.record(
            device=str(device),
            dataset_sha256=identity["dataset_sha256"],
            teacher_forcing=False,
            memory_uses_gt=False,
            performance=performance,
        )
        for name, method in summary["methods"].items():
            experiment.log(method["groups"]["all"], prefix=name)
        for path in list(output.iterdir()):
            # Publisher metrics are JSON snapshots; NPZ is an explicit trajectory
            # selection, while PNG traces stay linked as local outputs.
            experiment.link_output(path, kind="metrics" if path.suffix == ".json" else None)
        print(f"Temporal GRU comparison: {output}", flush=True)
        return output
