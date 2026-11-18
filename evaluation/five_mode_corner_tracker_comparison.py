"""Paired held-out comparison of frozen graph2 and Five-mode corner trackers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_corner_tracking import sequence_arrays
from graph.temporal.spatial_temporal_corner import SpatialTemporalCornerTracker
from utils.experiment_logger import ExperimentLogger, write

BODIES = {
    "peg": slice(0, 8),
    "box": slice(8, 16),
    "hole": slice(16, 20),
    "all": slice(0, 20),
}


def summarize(values):
    """Return point-frame weighted Euclidean errors; no episode averaging."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean_mm": float(values.mean()),
        "median_mm": float(np.median(values)),
        "p90_mm": float(np.percentile(values, 90)),
        "p95_mm": float(np.percentile(values, 95)),
        "max_mm": float(values.max()),
    }


def load_test(root, expected_hash):
    """Read only the previously frozen test manifest's verified payloads."""
    root = Path(root)
    path = root / "manifest.json"
    if file_sha256(path) != expected_hash:
        raise ValueError("Prepared test manifest checksum mismatch")
    manifest = json.loads(path.read_text())
    if manifest["episodes"] != 450 or manifest["seeds"] != 90:
        raise ValueError("Comparison requires all 450 held-out test episodes")
    episodes = []
    for row in manifest["records"]:
        if row["split"] != "test":
            raise ValueError("Comparison manifest contains non-test episodes")
        path = root / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"Test episode checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            episodes.append(dict(arrays, seed=row["seed"], mode=row["mode"]))
    return episodes, manifest


@torch.inference_mode()
def predict(episodes, checkpoint, batch_size, device):
    """Use each model's own train-fitted fallback, with identical FP32 execution."""
    model = SpatialTemporalCornerTracker().to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    fallback = checkpoint["fallback"].cpu().numpy()
    arrays, lengths, _ = sequence_arrays(episodes, fallback)
    tensors = {
        key: torch.as_tensor(arrays[key], device=device)
        for key in ("features", "anchor", "detected", "observed", "valid")
    }
    predictions = []
    for start in range(0, len(episodes), batch_size):
        stop = min(start + batch_size, len(episodes))
        length = int(lengths[start:stop].max())
        batch = [tensor[start:stop, :length] for tensor in tensors.values()]
        output = model(*batch)
        if not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite test predictions")
        torch.testing.assert_close(output[batch[3]], batch[2][batch[3]], rtol=0, atol=0)
        for index, row in enumerate(output):
            predictions.append(row[: lengths[start + index]].cpu().numpy())
    return np.concatenate(predictions)


def observation_history(episodes):
    """Keep causal ever-seen/age masks distinct from retrospective episode diagnostics."""
    ever, age, any_episode, first_error, episode_index = [], [], [], [], []
    for index, episode in enumerate(episodes):
        observed = episode["observed"]
        time = np.arange(len(observed))[:, None]
        last = np.maximum.accumulate(np.where(observed, time, -1), axis=0)
        ever.append(last >= 0)
        age.append(time - last)
        any_episode.append(np.broadcast_to(observed.any(axis=0), observed.shape))
        error = np.linalg.norm(episode["detected"] - episode["gt"], axis=-1) * 1000
        first = np.argmax(observed, axis=0)
        first_values = error[first, np.arange(20)]
        first_values = np.where(observed.any(axis=0), first_values, np.nan)
        first_error.append(np.broadcast_to(first_values, observed.shape))
        episode_index.append(np.full(len(observed), index, dtype=np.int32))
    return {
        "ever": np.concatenate(ever),
        "age": np.concatenate(age),
        "seen_in_episode": np.concatenate(any_episode),
        "first_detection_error_mm": np.concatenate(first_error),
        "episode_index": np.concatenate(episode_index),
    }


def evaluation_mask(observed, ever_seen, policy):
    """Select the same causal point-frame population for every compared model."""
    missing = ~observed
    if policy == "after_first_observation":
        return missing & ever_seen
    if policy == "all_missing":
        return missing
    raise ValueError("supervision must be 'all_missing' or 'after_first_observation'")


def model_metrics(prediction, arrays, history, episodes, supervision="all_missing"):
    """Use exactly the same missing point-frame masks for both models."""
    error = np.linalg.norm(prediction - arrays["gt"], axis=-1) * 1000
    missing = evaluation_mask(arrays["observed"], history["ever"], supervision)
    metrics = {
        "body": {
            name: summarize(error[:, slots][missing[:, slots]])
            for name, slots in BODIES.items()
        }
    }
    metrics["mode"] = {}
    for mode in sorted({episode["mode"] for episode in episodes}):
        frames = np.array(
            [episodes[i]["mode"] == mode for i in history["episode_index"]]
        )
        metrics["mode"][mode] = {
            name: summarize(error[frames, slots][missing[frames, slots]])
            for name, slots in BODIES.items()
        }
    depth = arrays["depth_mm"]
    stages = {
        "pre": depth <= 0,
        "entry": (depth > 0) & (depth <= 20),
        "deep": depth > 20,
    }
    metrics["stage"] = {
        stage: {
            name: summarize(error[frames, slots][missing[frames, slots]])
            for name, slots in BODIES.items()
        }
        for stage, frames in stages.items()
    }
    metrics["history"] = {}
    for name, support in (
        ("never_seen_yet", ~history["ever"]),
        ("previously_seen", history["ever"]),
        ("never_seen_in_episode", ~history["seen_in_episode"]),
        ("seen_somewhere_in_episode", history["seen_in_episode"]),
    ):
        mask = missing & support
        metrics["history"][name] = {
            body: summarize(error[:, slots][mask[:, slots]])
            for body, slots in BODIES.items()
        }
    metrics["occlusion_age"] = {}
    for name, lower, upper in (
        ("1_10", 1, 10),
        ("11_40", 11, 40),
        ("41_100", 41, 100),
        ("over_100", 101, 100000),
    ):
        mask = (
            missing
            & history["ever"]
            & (history["age"] >= lower)
            & (history["age"] <= upper)
        )
        metrics["occlusion_age"][name] = {
            body: summarize(error[:, slots][mask[:, slots]])
            for body, slots in BODIES.items()
        }
    hole_mask = missing[:, 16:20]
    support = arrays["raw_nearest_mm"][:, 16:20] <= 5
    initialized = arrays["hole_initialized"][:, None]
    subsets = {
        "raw_candidate_within_5mm_but_unassigned": support,
        "no_raw_candidate_within_5mm": ~support,
        "matcher_uninitialized": np.broadcast_to(~initialized, hole_mask.shape),
        "matcher_initialized": np.broadcast_to(initialized, hole_mask.shape),
        "first_detection_within_5mm": history["first_detection_error_mm"][:, 16:20]
        <= 5,
        "first_detection_over_5mm": history["first_detection_error_mm"][:, 16:20] > 5,
    }
    metrics["hole_subsets"] = {
        name: summarize(error[:, 16:20][hole_mask & mask])
        for name, mask in subsets.items()
    }
    residual = prediction[:, 16:20] - arrays["gt"][:, 16:20]
    center = residual.mean(axis=1)
    shape = residual - center[:, None]
    all_missing = hole_mask.all(axis=1)
    metrics["hole_all_four_missing"] = {
        "frames": int(all_missing.sum()),
        "center_error": summarize(np.linalg.norm(center[all_missing], axis=-1) * 1000),
        "center_removed_corner_error": summarize(
            np.linalg.norm(shape[all_missing], axis=-1).ravel() * 1000
        ),
    }
    ranked = []
    for index, episode in enumerate(episodes):
        frames = history["episode_index"] == index
        row = summarize(error[frames, 16:20][hole_mask[frames]])
        ranked.append(
            {
                "seed": episode["seed"],
                "mode": episode["mode"],
                **row,
                "hole_observed_points": int(episode["observed"][:, 16:20].sum()),
                "matcher_initialized": bool(episode["hole_initialized"].any()),
            }
        )
    metrics["worst_hole_episodes"] = sorted(
        ranked, key=lambda row: row.get("mean_mm", -1), reverse=True
    )[:15]
    return metrics, error


def input_metrics(arrays, history, episodes):
    """Separate candidate availability, association, and first-observation quality."""
    observed = arrays["observed"]
    raw = arrays["raw_nearest_mm"] <= 5
    error = np.linalg.norm(arrays["detected"] - arrays["gt"], axis=-1) * 1000
    bodies = {}
    for name, slots in BODIES.items():
        mask = observed[:, slots]
        bodies[name] = {
            "possible_point_frames": int(mask.size),
            "assigned_point_frames": int(mask.sum()),
            "raw_supported_5mm": int(raw[:, slots].sum()),
            "raw_supported_but_missing": int((raw[:, slots] & ~mask).sum()),
            "assigned_error": summarize(error[:, slots][mask]),
            "assigned_over_5mm": int((mask & (error[:, slots] > 5)).sum()),
        }
    hole_observed = observed[:, 16:20]
    raw_three = raw[:, 16:20].sum(axis=1) >= 3
    hint_three = (arrays["hint_nearest_mm"] <= 5).sum(axis=1) >= 3
    uninitialized = ~arrays["hole_initialized"]
    any_hole = np.array([episode["observed"][:, 16:20].any() for episode in episodes])
    radius = np.array(
        [float(np.median(episode["hole_radius_mm"])) for episode in episodes]
    )
    return {
        "body": bodies,
        "hole": {
            "episodes_with_zero_hole_observations": int((~any_hole).sum()),
            "episodes_without_all_four_slots_ever_observed": sum(
                not episode["observed"][:, 16:20].any(axis=0).all()
                for episode in episodes
            ),
            "never_seen_point_frames": int(
                (~hole_observed & ~history["ever"][:, 16:20]).sum()
            ),
            "previously_seen_missing_point_frames": int(
                (~hole_observed & history["ever"][:, 16:20]).sum()
            ),
            "matcher_uninitialized_frames": int(uninitialized.sum()),
            "three_raw_supported_but_uninitialized_frames": int(
                (raw_three & uninitialized).sum()
            ),
            "three_hint_supported_but_uninitialized_frames": int(
                (hint_three & uninitialized).sum()
            ),
            "three_raw_supported_but_under_three_assigned_frames": int(
                (raw_three & (hole_observed.sum(axis=1) < 3)).sum()
            ),
            "fit_center_error": summarize(arrays["hole_fit_center_error_mm"]),
            "fit_radius_error": summarize(arrays["hole_fit_radius_error_mm"]),
            "radius_over_matcher_30mm_episodes": int((radius > 30).sum()),
            "radius_over_30mm_with_no_observations": int(
                ((radius > 30) & ~any_hole).sum()
            ),
        },
    }


def evaluate(config):
    """Compare the frozen best checkpoints; never select models on held-out scores."""
    device = torch.device(config["device"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    episodes, manifest = load_test(config["data_root"], config["test_manifest_sha256"])
    fields = (
        "gt",
        "detected",
        "observed",
        "depth_mm",
        "raw_nearest_mm",
        "hint_nearest_mm",
        "hole_initialized",
        "box_initialized",
        "hole_fit_center_error_mm",
        "hole_fit_radius_error_mm",
    )
    arrays = {
        key: np.concatenate([episode[key] for episode in episodes]) for key in fields
    }
    history = observation_history(episodes)
    supervision = config.get("supervision", "all_missing")
    mask = evaluation_mask(arrays["observed"], history["ever"], supervision)
    counts = {name: int(mask[:, slots].sum()) for name, slots in BODIES.items()}
    if config.get("expected_supervised_counts", counts) != counts:
        raise ValueError(
            f"Evaluation mask counts differ from the frozen expected counts: {counts}"
        )
    with ExperimentLogger(
        team="graph",
        model="paired_five_mode_corner_tracker_test",
        job_type="evaluate",
        config=config,
        root=output / "experiments",
    ) as logger:
        logger.record(
            runtime={
                "torch": str(torch.__version__),
                "device": str(device),
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                "precision": "FP32; TF32 disabled; identical batch size",
            }
        )
        result = {
            "run_id": logger.run_id,
            "episodes": len(episodes),
            "seeds": manifest["seeds"],
            "frames": len(arrays["gt"]),
            "input_audit": input_metrics(arrays, history, episodes),
            "models": {},
            "config": config,
            "evaluation_mask": {
                "supervision": supervision,
                "counts": counts,
                "excluded_missing_point_frames": int(
                    (~arrays["observed"] & ~mask).sum()
                ),
                "packed_mask_sha256": hashlib.sha256(
                    np.packbits(mask, bitorder="little").tobytes()
                ).hexdigest(),
            },
        }
        saved = dict(arrays, **history)
        for name, settings in config["models"].items():
            path = Path(settings["checkpoint"])
            if file_sha256(path) != settings["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
            # Our hash-pinned training checkpoint also stores NumPy RNG/history
            # objects. It is a trusted artifact produced by this training run.
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=not settings.get("trusted_training_checkpoint", False),
            )
            prediction = predict(episodes, checkpoint, config["batch_size"], device)
            metrics, error = model_metrics(
                prediction, arrays, history, episodes, supervision
            )
            result["models"][name] = {
                "epoch": checkpoint["epoch"],
                "sha256": settings["sha256"],
                **metrics,
            }
            saved[f"prediction_{name}"] = prediction
            saved[f"error_{name}_mm"] = error
            logger.link_output(path, checkpoint=True)
            logger.log(metrics["body"], prefix=f"test/{name}")
            print(name, json.dumps(metrics["body"]), flush=True)
        np.savez_compressed(output / "predictions.npz", **saved)
        result["predictions_sha256"] = file_sha256(output / "predictions.npz")
        write(output / "results.json", result)
        for path in (output / "results.json", output / "predictions.npz"):
            logger.link_output(path)
        print("COMPARISON_COMPLETE", logger.run_id, flush=True)
