"""Compare fixed-Hole Object21D checkpoints on the preserved reset-seed test split."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.state21_curriculum import MODES, STAGES
from data_collection.state21_shared_tapnext import load_shared_tapnext
from evaluation.state21 import _errors, _percentiles
from graph.state_estimation.models import build_model
from main import load_config
from training.state21 import effective_config, prepare_dataset
from training.state21_loss import state21_loss
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger, write


def _checkpoints(names, training, bundle_path):
    """Verify stage, recipe and file identity before touching the test inputs."""
    root = Path(training["output"]["root"])
    bundle_sha256 = file_sha256(bundle_path)
    expected_data = {**training["data"], "curriculum_stage": STAGES[-1]}
    sources = {}
    for name in names:
        status = json.loads((root / name / "status.json").read_text())
        stages = status["completed"]
        if (
            status["status"] != "complete"
            or status["model"] != name
            or tuple(row["name"] for row in stages) != STAGES
            or status["graph_manifest_sha256"] != bundle_sha256
        ):
            raise ValueError(
                f"Expected a complete curriculum with the shared cache: {name}"
            )
        final = stages[-1]
        path = Path(final["checkpoint"])
        digest = file_sha256(path)
        if digest != final["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA-256 differs for {name}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected_model = {**training["model"], "name": name}
        if (
            checkpoint["config"]["model"] != expected_model
            or checkpoint["config"]["data"] != expected_data
            or checkpoint["config"]["seed"] != training["seed"]
            or checkpoint["epoch"] != status["epochs_per_stage"]
        ):
            raise ValueError(f"Checkpoint recipe differs for {name}")
        sources[name] = (path, digest, checkpoint)
    if len({source[2]["dataset_sha256"] for source in sources.values()}) != 1:
        raise ValueError("Final checkpoints must share one training dataset identity")
    return sources


def _continuation(spec, original):
    """Read a resumed checkpoint and verify its original training recipe."""
    original_path, _, baseline = original
    path = Path(spec["path"])
    digest = file_sha256(path)
    if digest != spec["sha256"]:
        raise ValueError(f"Resumed checkpoint SHA-256 differs: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint["config"].get("resume") != str(original_path)
        or checkpoint["dataset_sha256"] != baseline["dataset_sha256"]
        or checkpoint["resume_signature"] != baseline["resume_signature"]
        or checkpoint["epoch"] <= baseline["epoch"]
    ):
        raise ValueError(f"Expected an exact continuation of {original_path}")
    return path, digest, checkpoint


def _test_data(training, bundle_path):
    """Read full test episodes and verify disjoint train/validation lineage."""
    data = training["data"]
    episodes, report = load_shared_tapnext(data, splits=("test",))
    bundle = json.loads(bundle_path.read_text())
    cache_hashes = {
        name: source["cache_sha256"] for name, source in report["sources"].items()
    }
    if (
        cache_hashes != bundle["source_caches"]
        or report["full_split"] != bundle["full_split"]
        or report["split_reference_sha256"] != bundle["split_reference_sha256"]
        or data["max_edge_m"] != bundle["max_edge_m"]
    ):
        raise ValueError("Test inputs must match the training sources and pinned split")
    test_ids = {episode.episode_id for episode in episodes}
    training_ids = {row["episode_id"] for row in bundle["episodes"]}
    if len(test_ids) != 40 or not test_ids.isdisjoint(training_ids):
        raise ValueError(
            "Expected 40 reserved episodes outside the training graph archive"
        )
    entries = {}
    for name, source in data["sources"].items():
        path = Path(source["source_manifest"])
        if file_sha256(path) != report["sources"][name]["source_manifest_sha256"]:
            raise ValueError(f"Source manifest changed: {path}")
        entries.update(
            {
                row["path"]: row
                for row in json.loads(path.read_text())["episodes"]
                if row["path"] in test_ids
            }
        )
    if set(entries) != test_ids or Counter(
        row["mode"] for row in entries.values()
    ) != dict.fromkeys(MODES, 10):
        raise ValueError("Expected ten test episodes for each of the four policies")
    print(f"Building shared test graphs for {len(episodes)} episodes", flush=True)
    dataset = prepare_dataset(episodes, training, "test")
    if len(dataset.windows) != report["full_split"]["frame_counts"]["test"]:
        raise ValueError(
            "Expected the full reserved episodes without prefix truncation"
        )
    return dataset, entries, report


@torch.inference_mode()
def _score(model, dataset, device, batch_size, task):
    """Collect physical errors per frame and the common Object21D loss."""
    errors = defaultdict(dict)
    loss_sum = 0.0
    model.eval()
    for frames, targets, windows in dataset.batches(batch_size, device):
        output = model(frames)
        prediction = output["object21"]
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite held-out Object21D prediction")
        loss_sum += float(state21_loss(output, targets, **task)["state21"]) * len(
            windows
        )
        values = _errors(prediction, targets)
        rows = {metric: tensor.cpu().tolist() for metric, tensor in values.items()}
        for index, window in enumerate(windows):
            frame = window.sample.observation
            errors[frame.episode_id][frame.index] = {
                metric: samples[index] for metric, samples in rows.items()
            }
    return errors, loss_sum / len(dataset.windows)


def _stability_frame(errors, threshold, consecutive):
    """Count post-action frames through the first confirmed stable window."""
    run = 0
    for frame_count, error in enumerate(errors, start=1):
        run = run + 1 if error <= threshold else 0
        if run == consecutive:
            return frame_count
    return None


def _threshold_result(values, tolerance):
    """Return the exact frame count and fraction at or below the tolerance."""
    count = int(np.count_nonzero(np.asarray(values) <= tolerance))
    return {"within": count, "total": len(values), "rate": count / len(values)}


def _summarize(errors, entries, options, tolerance=3):
    """Summarize full trajectories and align recovery to the first Expert action."""
    thresholds = options["thresholds_mm"]
    consecutive = options["consecutive_frames"]
    recoveries = {}
    by_mode = defaultdict(list)
    for name, entry in entries.items():
        frames = errors[name]
        if set(frames) != set(range(entry["observations"])):
            raise ValueError(f"Missing or misaligned test predictions: {name}")
        rows = [frames[index] for index in range(len(frames))]
        by_mode[entry["mode"]].extend(rows)
        if entry["mode"] != "expert_random_expert":
            continue
        # recovery_step is the observation before the first renewed Expert action.
        switch = entry["recovery_step"]
        trace = [row["endpoint_error_mm"] for row in rows[switch + 1 :]]
        if (
            not entry["restarted_expert"]
            or len(trace) != entry["expert_recovery_actions"]
            or not trace
        ):
            raise ValueError(f"Missing observations after the Expert return: {name}")
        recoveries[name] = {
            "pre_action_frame": switch,
            "post_action_frames": len(trace),
            "first_post_action_mm": trace[0],
            "first_five_mean_mm": float(np.mean(trace[:5])),
            "stable_at_frame": {
                str(threshold): _stability_frame(trace, threshold, consecutive)
                for threshold in thresholds
            },
        }
    accuracy = {
        metric: _percentiles([row[metric] for rows in by_mode.values() for row in rows])
        for metric in next(iter(errors.values()))[0]
    }
    endpoint = [row["endpoint_error_mm"] for rows in by_mode.values() for row in rows]
    first_after_return = [row["first_post_action_mm"] for row in recoveries.values()]
    stability = {}
    for threshold in map(str, thresholds):
        times = [
            row["stable_at_frame"][threshold]
            for row in recoveries.values()
            if row["stable_at_frame"][threshold] is not None
        ]
        stability[threshold] = {
            "stabilized": len(times),
            "episodes": len(recoveries),
            "median_frames_among_stabilized": float(np.median(times))
            if times
            else None,
        }
    return {
        "all_frames": accuracy,
        "endpoint_rmse_mm": float(np.sqrt(np.mean(np.square(endpoint)))),
        "endpoint_tolerance_mm": tolerance,
        "endpoint_within_tolerance": _threshold_result(endpoint, tolerance),
        "endpoint_by_policy_mm": {
            mode: _percentiles([row["endpoint_error_mm"] for row in rows])
            for mode, rows in by_mode.items()
        },
        "recovery": {
            "first_post_action_mm": _percentiles(first_after_return),
            "first_post_action_within_tolerance": _threshold_result(
                first_after_return, tolerance
            ),
            "first_five_mean_mm": _percentiles(
                [row["first_five_mean_mm"] for row in recoveries.values()]
            ),
            "stability": stability,
            "episodes": recoveries,
        },
    }


def evaluate(config):
    """Run all specified frozen checkpoints against identical held-out windows."""
    recovery = config["recovery"]
    if (
        type(recovery["consecutive_frames"]) is not int
        or recovery["consecutive_frames"] < 1
        or not recovery["thresholds_mm"]
        or any(
            not np.isfinite(value) or value <= 0 for value in recovery["thresholds_mm"]
        )
    ):
        raise ValueError(
            "Recovery needs a positive frame count and positive mm thresholds"
        )
    training = effective_config(load_config(config["training_config"]))
    tolerance = config.get("endpoint_tolerance_mm", 3)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("endpoint_tolerance_mm must be a positive finite distance")
    device = set_device(config["device"])
    torch.set_num_threads(training["training"]["cpu_threads"])
    bundle_path = Path(training["data"]["curriculum_cache"]) / "manifest.json"
    checkpoints = _checkpoints(config["models"], training, bundle_path)
    for label, spec in config.get("continuations", {}).items():
        if label in checkpoints:
            raise ValueError(f"Checkpoint result labels must be distinct: {label}")
        checkpoints[label] = _continuation(spec, checkpoints[spec["model"]])
    dataset, entries, report = _test_data(training, bundle_path)
    with ExperimentLogger(
        team="graph",
        job_type="evaluate",
        model="state21_holdout",
        config={**config, "training": training},
        seed=training["seed"],
    ) as comparison:
        output = Path(config["output"]) / comparison.run_id / "comparison.json"
        results = {}
        for name, (path, digest, checkpoint) in checkpoints.items():
            print(
                f"Evaluating {name} on {len(dataset.windows)} test frames", flush=True
            )
            with ExperimentLogger(
                team="graph",
                model=checkpoint["config"]["model"]["name"],
                job_type="evaluate",
                seed=training["seed"],
                config={**config, "model": checkpoint["config"]["model"]},
            ) as experiment:
                experiment.link_output(path, checkpoint=True)
                experiment.record(checkpoint_sha256=digest, data_audit=report)
                model = build_model(**checkpoint["config"]["model"]).to(device)
                model.load_state_dict(checkpoint["best_model"], strict=True)
                errors, loss = _score(
                    model, dataset, device, config["batch_size"], training["task"]
                )
                result = _summarize(errors, entries, recovery, tolerance)
                results[name] = {
                    "run_id": experiment.run_id,
                    "checkpoint_sha256": digest,
                    "checkpoint_epoch": checkpoint["epoch"],
                    "best_validation_loss": checkpoint["best_loss"],
                    "test_state21_loss": loss,
                    **result,
                }
                experiment.log(
                    {
                        "state21_loss": loss,
                        "endpoint_within_tolerance": result[
                            "endpoint_within_tolerance"
                        ],
                        **result["all_frames"],
                    },
                    prefix="test",
                )
        write(
            output,
            {
                "run_id": comparison.run_id,
                "reset_seeds": report["selected_reset_seeds"]["test"],
                "episodes": len(entries),
                "frames": len(dataset.windows),
                "recovery_options": config["recovery"],
                "models": results,
            },
        )
        comparison.link_output(output, kind="metrics")
        print(output, flush=True)
