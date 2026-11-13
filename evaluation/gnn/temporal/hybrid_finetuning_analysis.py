"""Summarize a completed H1 study without selecting or updating any model."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal.hybrid_features import (
    cache_frozen_features,
    predict_cached,
    replay_hybrid,
)
from evaluation.gnn.temporal.hybrid_diagnostics import paired_bootstrap
from evaluation.gnn.temporal.hybrid_finetuning import analyze_finetuning
from evaluation.gnn.temporal.hybrid_inputs import load_reference, restore_branches
from evaluation.gnn.temporal.insertion import distribution, score_insertion_methods
from graph.pose_geometry import hole_frame_vector
from graph.temporal.hybrid_finetuning import HybridPegFineTuner
from utils.experiment_logger import ExperimentLogger

LABELS = {
    "original_h1": "Original H1",
    "pose_only": "FT-A",
    "pose_relative": "FT-B",
    "pose_relative_lateral": "FT-C",
}


def tracking_summary(method, original):
    """Pair translational/rotational lag estimates on the same eligible episodes."""
    baseline = {row["episode"]: row for row in original["peg_dynamics"]}
    result = {}
    for field in ("lag_frames", "rotation_lag_frames"):
        pairs = [
            (row[field], baseline[row["episode"]][field])
            for row in method["peg_dynamics"]
            if row[field] is not None and baseline[row["episode"]][field] is not None
        ]
        values = np.asarray(pairs, dtype=float)
        result[field] = {"paired_episodes": len(pairs)}
        if pairs:
            result[field].update(
                mean=float(values[:, 0].mean()),
                median=float(np.median(values[:, 0])),
                mean_absolute=float(np.abs(values[:, 0]).mean()),
                mean_positive_delay=float(np.maximum(values[:, 0], 0).mean()),
                greater_delay_episodes=int(np.sum(values[:, 0] > values[:, 1])),
                greater_absolute_lag_episodes=int(
                    np.sum(np.abs(values[:, 0]) > np.abs(values[:, 1]))
                ),
            )
    return result


def approach_proxy(data, settings):
    """GT-only offline +X approach proxy; this is not a contact/phase annotation."""
    local = hole_frame_vector(
        torch.from_numpy(data["target"][:, 7:10]), torch.from_numpy(data["target"][:, 13:17])
    ).numpy()
    aligned = (
        (local[:, 0] >= settings["alignment_axial_min_m"])
        & (local[:, 0] <= settings["alignment_axial_max_m"])
        & (np.linalg.norm(local[:, 1:], axis=1) <= settings["alignment_lateral_m"])
    )
    approaching = np.zeros(len(local), dtype=bool)
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        approaching[rows[1:]] = np.diff(local[rows, 0]) > 0.0005
    return aligned & approaching


def save_plots(output, errors, source):
    """Export precision CDFs and development curves using the saved evidence only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(
        original_h1="#667085",
        pose_only="#3366cc",
        pose_relative="#00897b",
        pose_relative_lateral="#d2691e",
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, metric, title in zip(axes, ("relative_3d", "lateral"), ("Relative 3D", "Lateral")):
        for name, label in LABELS.items():
            values = np.sort(errors[name + "__" + metric])
            indices = np.linspace(0, len(values) - 1, min(2000, len(values))).astype(int)
            axis.plot(
                values[indices], (indices + 1) / len(values) * 100, label=label, color=colors[name]
            )
        axis.axvline(3, color="#888888", linestyle=":", linewidth=1)
        axis.set(
            xlabel="Estimation error (mm)",
            ylabel="Frames (%)",
            title=title,
            xlim=(0, 25),
            ylim=(0, 100),
        )
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output / "precision_cdf.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name in tuple(LABELS)[1:]:
        history = json.loads((source / name / "history.json").read_text())
        epochs = [row["epoch"] for row in history]
        axes[0].plot(
            epochs,
            [row["development"]["all"]["relative_3d"]["mean_mm"] for row in history],
            label=LABELS[name],
            color=colors[name],
        )
        axes[1].plot(
            epochs,
            [row["development"]["all"]["peg_position"]["mean_mm"] for row in history],
            label=LABELS[name],
            color=colors[name],
        )
    for axis, title in zip(axes, ("Development relative 3D", "Development Peg position")):
        axis.set(xlabel="Epoch", ylabel="Mean error (mm)", title=title)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output / "development_curves.png", dpi=160)
    plt.close(figure)


def additional_loss_comparisons(data, errors, manifest, seed, resamples):
    """Compare added objectives against their matched fine-tuning controls."""
    result = {}
    for candidate, control in (
        ("pose_relative", "pose_only"),
        ("pose_relative_lateral", "pose_relative"),
    ):
        comparison = {}
        for metric in ("peg_position", "relative_3d", "lateral", "axial"):
            delta = errors[candidate + "__" + metric] - errors[control + "__" + metric]
            comparison[metric] = paired_bootstrap(data, delta, manifest, seed, resamples)
        result[candidate + "__minus__" + control] = comparison
    return result


def analyze_last_heads(source, baseline_config, settings, manifest, seed, resamples):
    """Explain rejected updates after selection; never replace selected models.

    One fresh original H1 replay supplies frozen activations. Last heads are
    decoded on those same causal activations, preserving normalization and Hole
    outputs on development. All four heads also receive fresh full-episode
    streaming benchmark replay. No training or checkpoint search occurs here.
    """
    config = yaml.safe_load(Path(baseline_config).read_text())
    config.update(device="cpu", mamba_backend="torch_reference")
    benchmark, _, reference, _, provenance, identity, _ = load_reference(config)
    mamba, gate, audit = restore_branches(
        config, reference, provenance, identity, torch.device("cpu")
    )
    hashes = {name: value["sha256"] for name, value in audit["checkpoints"].items()}
    if hashes != settings["source_checkpoint_hashes"]:
        raise ValueError("Diagnostic original checkpoint hashes differ from the source study")
    with np.load(source / "development_observations.npz", allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    if set(data["episode"]) != set(manifest["splits"]["development"]):
        raise ValueError("Diagnostic frames must be the fine-tuning development episodes only")
    cache, streamed_h1 = cache_frozen_features(mamba, gate, data, torch.device("cpu"))
    adapter = HybridPegFineTuner(mamba, gate)
    predictions = {"original_h1": predict_cached(adapter, cache, data, torch.device("cpu"), 512)}
    np.testing.assert_allclose(predictions["original_h1"], streamed_h1, atol=2e-6, rtol=1e-5)
    benchmark_predictions = {
        "original_h1": replay_hybrid(mamba, gate, benchmark, torch.device("cpu"))[0]["hybrid_h1_a"]
    }
    checkpoints = {}
    for name in tuple(LABELS)[1:]:
        path = source / name / "last.pt"
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if (
            payload["kind"] != "hybrid_peg_finetuning"
            or payload["ablation"] != name
            or payload["config"]["source_checkpoint_hashes"] != hashes
            or payload["config"]["finetuning_split"] != manifest
        ):
            raise ValueError(f"{name}: last-head checkpoint provenance differs")
        mamba.heads["peg"].load_state_dict(payload["peg_head"], strict=True)
        predictions[name] = predict_cached(adapter, cache, data, torch.device("cpu"), 512)
        print(f"Replaying unselected last head: {name}, epoch {payload['epoch']}", flush=True)
        benchmark_predictions[name] = replay_hybrid(mamba, gate, benchmark, torch.device("cpu"))[0][
            "hybrid_h1_a"
        ]
        checkpoints[name] = {"epoch": payload["epoch"], "sha256": file_sha256(path)}
    summary, errors = score_insertion_methods(data, predictions, settings["evaluation"])
    diagnostics = analyze_finetuning(
        data, predictions, errors, manifest, seed=seed, resamples=resamples
    )
    benchmark_summary, benchmark_errors = score_insertion_methods(
        benchmark, benchmark_predictions, settings["evaluation"]
    )
    benchmark_diagnostics = analyze_finetuning(
        benchmark, benchmark_predictions, benchmark_errors, manifest, seed=seed, resamples=resamples
    )
    flat_errors = {
        name + "__" + metric: values
        for name, metrics in benchmark_errors.items()
        for metric, values in metrics.items()
    }
    return {
        "scope": "unselected LAST heads; post-selection diagnosis only; no model updates or selection",
        "development_observations_sha256": file_sha256(source / "development_observations.npz"),
        "initial_cached_streaming_max_difference": float(
            np.abs(predictions["original_h1"] - streamed_h1).max()
        ),
        "checkpoints": checkpoints,
        "development": {"summary": summary, "diagnostics": diagnostics},
        "benchmark": {
            "summary": benchmark_summary,
            "diagnostics": benchmark_diagnostics,
            "additional_loss_comparisons": additional_loss_comparisons(
                benchmark, flat_errors, manifest, seed, resamples
            ),
        },
    }


def analyze(config):
    """Report additional-loss effects, tracking and approach after selection is fixed."""
    torch.set_num_threads(config.get("math_threads", 1))
    source = Path(config["source_run"])
    status = json.loads((source / "execution_status.json").read_text())
    if status["status"] != "completed_ablations_no_independent_test":
        raise ValueError("source_run must be a completed three-ablation H1 study")
    summary = json.loads((source / "summary.json").read_text())
    settings = json.loads((source / "effective_config.json").read_text())
    manifest = json.loads((source / "finetuning_split.json").read_text())
    snapshot = Path(config["reference_predictions"])
    if file_sha256(snapshot) != settings["reference_predictions_sha256"]:
        raise ValueError("Reference snapshot differs from the source study")
    with np.load(snapshot, allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in ("sample_ids", "episode", "frame", "target")}
    with np.load(source / "frame_errors.npz", allow_pickle=False) as saved:
        errors = {name: saved[name].copy() for name in saved.files}
    np.testing.assert_array_equal(errors["sample_ids"], data["sample_ids"])
    if set(summary["methods"]) != set(LABELS):
        raise ValueError("Expected Original H1 and all three fixed ablations on the same frames")
    result = {
        "scope": "post-selection reused development benchmark; no model updates or test selection",
        "source_run_id": source.name,
        "models": {},
        "additional_loss_comparisons": {},
    }
    baseline = summary["methods"]["original_h1"]
    approach = approach_proxy(data, settings["evaluation"])
    for name, method in summary["methods"].items():
        result["models"][name] = {
            "label": LABELS[name],
            "all": method["groups"]["all"],
            "tracking": tracking_summary(method, baseline),
            "approach_proxy": {
                "definition": "GT Hole local +X motion >0.5mm/frame within the existing alignment geometry window",
                "frames": int(approach.sum()),
                "metrics": {
                    metric: distribution(errors[name + "__" + metric][approach], "mm")
                    for metric in ("peg_position", "relative_3d", "lateral", "axial")
                },
                "lateral_lt_3mm_percent": (
                    float(np.mean(errors[name + "__lateral"][approach] < 3) * 100)
                    if approach.any()
                    else None
                ),
                "relative_3d_lt_3mm_percent": (
                    float(np.mean(errors[name + "__relative_3d"][approach] < 3) * 100)
                    if approach.any()
                    else None
                ),
            },
        }
    result["additional_loss_comparisons"] = additional_loss_comparisons(
        data, errors, manifest, config["seed"], config["bootstrap_resamples"]
    )
    with ExperimentLogger(
        team="graph",
        model="hybrid_geometry_analysis",
        config=config,
        seed=config["seed"],
        job_type="analyze",
    ) as experiment:
        output = Path(config["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "comparison.json", result)
        write_json(output / "effective_config.json", config)
        if config.get("analyze_last_heads", False):
            last = analyze_last_heads(
                source,
                config["baseline_config"],
                settings,
                manifest,
                config["seed"],
                config["bootstrap_resamples"],
            )
            write_json(output / "last_head_diagnostics.json", last)
        save_plots(output, errors, source)
        experiment.record(
            source_run_id=source.name,
            independent_test=False,
            source_summary_sha256=file_sha256(source / "summary.json"),
            source_errors_sha256=file_sha256(source / "frame_errors.npz"),
            reference_predictions_sha256=file_sha256(snapshot),
            analysis_source_sha256=file_sha256(__file__),
        )
        for path in output.iterdir():
            experiment.link_output(path)
        print(f"H1 post-selection analysis: {output}", flush=True)
        return output
