"""Offline and paired closed-loop evaluation for Surface-Mamba candidates."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from recurrent.surface_mamba import (
    ControllerEnsemble,
    OnlineController,
    load_controller_checkpoint,
    state36_to_controller_features,
)
from utils.experiment_logger import ExperimentLogger


def _candidate_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate named single-model and five-model evaluation candidates."""
    raw_candidates = config.get("candidates")
    if not isinstance(raw_candidates, dict) or not raw_candidates:
        raise ValueError("candidates must be a nonempty mapping")

    candidates = {}
    for name, raw_spec in raw_candidates.items():
        if not isinstance(name, str) or not name or not isinstance(raw_spec, dict):
            raise ValueError("Each candidate needs a nonempty name and mapping")
        input_source = raw_spec.get("input")
        if input_source not in {"graph", "gt"}:
            raise ValueError(f"Candidate {name!r} input must be 'graph' or 'gt'")
        raw_paths = raw_spec.get("checkpoints")
        if not isinstance(raw_paths, list) or len(raw_paths) not in {1, 5}:
            raise ValueError(f"Candidate {name!r} must contain one or five checkpoints")
        if any(not isinstance(path, str) or not path for path in raw_paths):
            raise ValueError(f"Candidate {name!r} checkpoint paths must be nonempty strings")
        if len(set(raw_paths)) != len(raw_paths):
            raise ValueError(f"Candidate {name!r} checkpoint paths must be distinct")
        candidates[name] = {
            "input": input_source,
            "checkpoints": [Path(path) for path in raw_paths],
        }
    return candidates


def _load_candidates(config: dict[str, Any], device: torch.device):
    """Load independent recurrent state for every configured candidate."""
    loaded = {}
    for name, spec in _candidate_specs(config).items():
        checkpoints = [
            load_controller_checkpoint(path, device) for path in spec["checkpoints"]
        ]
        seeds = [checkpoint.seed for checkpoint in checkpoints]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Candidate {name!r} contains duplicate checkpoint seeds")
        controllers = [OnlineController(checkpoint) for checkpoint in checkpoints]
        controller = (
            controllers[0]
            if len(controllers) == 1
            else ControllerEnsemble(controllers)
        )
        loaded[name] = {
            **spec,
            "checkpoints": checkpoints,
            "controller": controller,
        }
    return loaded


def _aggregate_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    """Aggregate one model directly or five models by mean and majority vote."""
    if len(predictions) not in {1, 5}:
        raise ValueError("Prediction aggregation requires one or five models")
    if len(predictions) == 1:
        return predictions[0].copy()
    if any(prediction.shape != predictions[0].shape for prediction in predictions[1:]):
        raise ValueError("Candidate predictions must have matching shapes")

    stacked = np.stack(predictions)
    ensemble = np.empty_like(predictions[0])
    ensemble[:, :6] = stacked[:, :, :6].mean(axis=0)
    positive_votes = (stacked[:, :, 6] > 0).sum(axis=0)
    ensemble[:, 6] = np.where(positive_votes >= 3, 1.0, -1.0)
    return ensemble


def _read_test_split(dataset: Path) -> dict[str, np.ndarray]:
    path = dataset / "test.npz"
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "state36_gt",
            "state36_graph",
            "action_target",
            "episode_offsets",
            "reset_seed",
        }
        if not required <= set(archive.files):
            raise ValueError("Test split is missing required State36 arrays")
        arrays = {key: archive[key] for key in required}
    if len(arrays["reset_seed"]) != 90 or arrays["episode_offsets"][-1] != len(arrays["action_target"]):
        raise ValueError("Expected the clean 90-episode test split")
    if len(np.unique(arrays["reset_seed"])) != 90:
        raise ValueError("Test reset seeds must be distinct")
    if not all(
        np.isfinite(arrays[name]).all()
        for name in ("state36_gt", "state36_graph", "action_target")
    ):
        raise ValueError("Test split contains nonfinite values")
    for split, expected_count in (("train", 720), ("validation", 90)):
        with np.load(dataset / f"{split}.npz", allow_pickle=False) as archive:
            reset_seeds = archive["reset_seed"]
        if len(reset_seeds) != expected_count:
            raise ValueError(f"Expected {expected_count} {split} reset seeds")
        if set(reset_seeds.tolist()) & set(arrays["reset_seed"].tolist()):
            raise ValueError(f"{split} and test reset seeds overlap")
    return arrays


@torch.inference_mode()
def _predict_episodes(
    checkpoint, state: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Predict held-out episodes while resetting recurrent state at every boundary."""
    stats = checkpoint.statistics
    device = next(checkpoint.model.parameters()).device
    predictions = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        features = state36_to_controller_features(state[start:stop])
        normalized = (
            (features - stats["obs_mean"]) / stats["obs_std"] * stats["obs_keep"]
        ).astype(np.float32)
        prediction, _ = checkpoint.model(
            torch.from_numpy(normalized)[None].to(device)
        )
        predictions.append(prediction[0].float().cpu().numpy())
    prediction = np.concatenate(predictions)
    action = np.empty_like(prediction, dtype=np.float32)
    action[:, :6] = prediction[:, :6] * stats["act_std"] + stats["act_mean"]
    action[:, 6] = np.where(prediction[:, 6] > 0, 1.0, -1.0)
    return np.clip(action, -1, 1)


def _action_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    motion_error = prediction[:, :6] - target[:, :6]
    return {
        "motion_mse": float(np.mean(motion_error**2)),
        "translation_rmse": float(np.sqrt(np.mean(motion_error[:, :3] ** 2))),
        "rotation_rmse": float(np.sqrt(np.mean(motion_error[:, 3:6] ** 2))),
        "gripper_accuracy": float(np.mean((prediction[:, 6] > 0) == (target[:, 6] > 0))),
    }


def _episode_metric_rows(
    candidate_name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    offsets: np.ndarray,
    reset_seeds: np.ndarray,
) -> list[dict[str, Any]]:
    """Return equal-episode metrics while retaining the timestep-weighted summary."""
    rows = []
    for seed, start, stop in zip(
        reset_seeds, offsets[:-1], offsets[1:], strict=True
    ):
        rows.append(
            {
                "candidate": candidate_name,
                "reset_seed": int(seed),
                "steps": int(stop - start),
                **_action_metrics(prediction[start:stop], target[start:stop]),
            }
        )
    return rows


def _bootstrap_episode_metrics(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, dict[str, float]]:
    """Bootstrap equal-episode metric means with a deterministic RNG seed."""
    metric_names = (
        "motion_mse",
        "translation_rmse",
        "rotation_rmse",
        "gripper_accuracy",
    )
    values = np.asarray(
        [[row[name] for name in metric_names] for row in rows], dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(rows), size=(samples, len(rows)))
    bootstrap_means = values[indices].mean(axis=1)
    return {
        name: {
            "episode_mean": float(values[:, index].mean()),
            "bootstrap_95_low": float(np.percentile(bootstrap_means[:, index], 2.5)),
            "bootstrap_95_high": float(np.percentile(bootstrap_means[:, index], 97.5)),
        }
        for index, name in enumerate(metric_names)
    }


def _paired_bootstrap_differences(
    rows_by_candidate: dict[str, list[dict[str, Any]]],
    reference: str,
    samples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Bootstrap paired candidate-minus-reference episode metric differences."""
    if reference not in rows_by_candidate:
        raise ValueError("comparison_reference must name an offline candidate")
    metric_names = (
        "motion_mse",
        "translation_rmse",
        "rotation_rmse",
        "gripper_accuracy",
    )
    reference_rows = rows_by_candidate[reference]
    reference_seeds = [row["reset_seed"] for row in reference_rows]
    result = {}
    for candidate_index, (name, rows) in enumerate(rows_by_candidate.items()):
        if name == reference:
            continue
        if [row["reset_seed"] for row in rows] != reference_seeds:
            raise ValueError("Offline paired metrics require identical episode order")
        differences = np.asarray(
            [
                [row[metric] - reference_row[metric] for metric in metric_names]
                for row, reference_row in zip(rows, reference_rows, strict=True)
            ],
            dtype=np.float64,
        )
        generator = np.random.default_rng(seed + candidate_index)
        indices = generator.integers(
            0, len(rows), size=(samples, len(rows))
        )
        bootstrap_means = differences[indices].mean(axis=1)
        result[name] = {
            metric: {
                "mean_difference": float(differences[:, index].mean()),
                "bootstrap_95_low": float(
                    np.percentile(bootstrap_means[:, index], 2.5)
                ),
                "bootstrap_95_high": float(
                    np.percentile(bootstrap_means[:, index], 97.5)
                ),
            }
            for index, metric in enumerate(metric_names)
        }
    return result


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one stable, inspectable row per candidate and test episode."""
    fieldnames = (
        "candidate",
        "reset_seed",
        "steps",
        "motion_mse",
        "translation_rmse",
        "rotation_rmse",
        "gripper_accuracy",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_offline(config: dict[str, Any]) -> None:
    """Score each named candidate on clean held-out expert trajectories."""
    device = torch.device(config.get("device", "cpu"))
    candidates = _load_candidates(config, device)
    dataset = Path(config["dataset"]).resolve()
    arrays = _read_test_split(dataset)
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "effective_config.json", config)
    bootstrap_samples = int(config.get("bootstrap_samples", 2000))
    bootstrap_seed = int(config.get("bootstrap_seed", 0))
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    results = {}
    rows_by_candidate = {}
    with ExperimentLogger(
        team="recurrent",
        model="surface_mamba_offline",
        job_type="evaluate",
        seed=0,
        config=config,
        root=output / "experiments",
    ) as logger:
        for candidate_name, candidate in candidates.items():
            state_key = (
                "state36_graph" if candidate["input"] == "graph" else "state36_gt"
            )
            predictions = [
                _predict_episodes(
                    checkpoint, arrays[state_key], arrays["episode_offsets"]
                )
                for checkpoint in candidate["checkpoints"]
            ]
            prediction = _aggregate_predictions(predictions)
            metrics = _action_metrics(prediction, arrays["action_target"])
            episode_rows = _episode_metric_rows(
                candidate_name,
                prediction,
                arrays["action_target"],
                arrays["episode_offsets"],
                arrays["reset_seed"],
            )
            rows_by_candidate[candidate_name] = episode_rows
            results[candidate_name] = {
                "input": candidate["input"],
                "checkpoint_seeds": [
                    checkpoint.seed for checkpoint in candidate["checkpoints"]
                ],
                "metrics": metrics,
                "episode_bootstrap": _bootstrap_episode_metrics(
                    episode_rows, bootstrap_samples, bootstrap_seed
                ),
            }
            logger.log(metrics, step=0, prefix=candidate_name)
        comparison_reference = config.get("comparison_reference", "mixed_ensemble")
        all_episode_rows = [
            row for rows in rows_by_candidate.values() for row in rows
        ]
        _write_episode_csv(output / "offline_episode_metrics.csv", all_episode_rows)
        write_json(output / "offline_episode_metrics.json", all_episode_rows)
        summary = {
            "run_id": logger.run_id,
            "episodes": len(arrays["reset_seed"]),
            "rows": len(arrays["action_target"]),
            "test_sha256": file_sha256(dataset / "test.npz"),
            "dataset_manifest_sha256": file_sha256(
                dataset / "dataset_manifest.json"
            ),
            "checkpoint_sha256": {
                name: [file_sha256(checkpoint.path) for checkpoint in candidate["checkpoints"]]
                for name, candidate in candidates.items()
            },
            "results": results,
            "comparison_reference": comparison_reference,
            "paired_candidate_minus_reference": _paired_bootstrap_differences(
                rows_by_candidate,
                comparison_reference,
                bootstrap_samples,
                bootstrap_seed,
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        }
        write_json(output / "offline_summary.json", summary)
        logger.link_output(output / "effective_config.json")
        logger.link_output(output / "offline_episode_metrics.csv")
        logger.link_output(output / "offline_episode_metrics.json")
        logger.link_output(output / "offline_summary.json")


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        raise ValueError("Wilson interval requires a positive sample count")
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / count + z * z / (4 * count * count)
    ) / denominator
    return [center - radius, center + radius]


def _is_success(info: dict[str, Any]) -> bool:
    value = info.get("success", False)
    if torch.is_tensor(value):
        return bool(value.detach().cpu().reshape(-1).any().item())
    return bool(np.asarray(value).reshape(-1).any())


def _episode(
    env,
    seed: int,
    controller,
    frontend: Any | None,
    horizon: int,
) -> dict[str, Any]:
    from envs.maniskill import get_gt_state
    from recurrent.surface_state import from_gt_state, state36

    observation, _ = env.reset(seed=seed)
    controller.reset()
    if frontend is not None:
        frontend.reset()
    success = False
    frontend_times = []
    for step in range(horizon):
        robot, object21_gt = from_gt_state(get_gt_state(env))
        if frontend is None:
            object21 = object21_gt
        else:
            result = frontend.step(observation, robot[:3])
            object21 = result.object21
            frontend_times.append(result.timing_ms)
        action = controller.act(state36(robot, object21))
        observation, _, terminated, truncated, info = env.step(
            torch.as_tensor(action[None], device=env.unwrapped.device)
        )
        success = success or _is_success(info)
        if success or bool(terminated.reshape(-1)[0]) or bool(truncated.reshape(-1)[0]):
            break
    timing = np.concatenate(frontend_times) if frontend_times else np.asarray([], dtype=np.float32)
    result = {
        "success": success,
        "steps": step + 1,
        "frontend_mean_ms": float(timing.mean()) if len(timing) else None,
    }
    if hasattr(env, "motion_receipt"):
        result.update(env.motion_receipt())
    return result


def _load_eval_seeds(
    path: str | Path,
    seed_limit: int | None = None,
    seed_start: int = 0,
) -> list[int]:
    """Load a contiguous slice from the frozen 300-seed evaluation list."""
    payload = json.loads(Path(path).read_text())
    seeds = payload["seeds"] if isinstance(payload, dict) else payload
    if len(seeds) != 300 or len(set(seeds)) != 300 or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Closed-loop seed file must contain 300 distinct nonnegative integers")
    if type(seed_start) is not int or not 0 <= seed_start < len(seeds):
        raise ValueError("seed_start must be an integer between 0 and 299")
    if seed_limit is not None:
        remaining = len(seeds) - seed_start
        if type(seed_limit) is not int or not 1 <= seed_limit <= remaining:
            raise ValueError(
                "seed_limit must be an integer between 1 and the number of "
                "seeds remaining after seed_start"
            )
        return seeds[seed_start : seed_start + seed_limit]
    return seeds[seed_start:]


def evaluate_closed_loop(config: dict[str, Any], *, wrap_environment=None) -> None:
    """Run named graph or GT-input candidates on the same 300 reset seeds."""
    from envs.maniskill import make_peg_insertion_env
    from graph.surface_context_frontend import SurfaceContextFrontend

    device = torch.device(config["controller_device"])
    loaded_candidates = _load_candidates(config, device)
    candidates = {
        name: candidate["controller"] for name, candidate in loaded_candidates.items()
    }
    comparison_reference = config.get("comparison_reference")
    if comparison_reference not in candidates:
        raise ValueError("comparison_reference must name one configured candidate")
    seeds = _load_eval_seeds(
        config["seeds_file"],
        config.get("seed_limit"),
        config.get("seed_start", 0),
    )
    output = Path(config["output"]).resolve()
    effective_path = output / "effective_config.json"
    trace_path = output / "rollouts.jsonl"
    rows = []
    completed = set()
    if output.exists():
        if not config.get("resume") or (output / "closed_loop_summary.json").exists():
            raise FileExistsError(f"Closed-loop output already exists: {output}")
        if json.loads(effective_path.read_text()) != config:
            raise ValueError("Closed-loop resume requires an identical effective config")
        if trace_path.exists():
            for line_number, line in enumerate(trace_path.read_text().splitlines(), 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid closed-loop JSONL at line {line_number}; preserve and repair it explicitly"
                    ) from error
                key = (row.get("candidate"), row.get("seed"))
                if key in completed:
                    raise ValueError(f"Duplicate closed-loop result: {key}")
                completed.add(key)
                rows.append(row)
        expected_keys = {(candidate, seed) for candidate in candidates for seed in seeds}
        unexpected = completed - expected_keys
        if unexpected:
            raise ValueError(f"Closed-loop trace contains unexpected jobs: {sorted(unexpected)[:5]}")
    else:
        output.mkdir(parents=True)
        write_json(effective_path, config)
    frontend = SurfaceContextFrontend(config["frontend"])
    environment = config["environment"]
    if environment["max_episode_steps"] != config["horizon"]:
        raise ValueError("Environment max_episode_steps must equal the evaluation horizon")
    env = make_peg_insertion_env(
        device=environment["device"],
        render_backend=environment["render_backend"],
        observation_device=environment["observation_device"],
        obs_mode="rgb+depth",
        control_mode="pd_ee_delta_pose",
        sensor_configs={
            name: {"width": 512, "height": 512}
            for name in ("base_camera", "hand_camera")
        },
        max_episode_steps=environment["max_episode_steps"],
    )
    if wrap_environment is not None:
        env = wrap_environment(env)
    started = monotonic()
    with ExperimentLogger(
        team="recurrent",
        model="surface_mamba_closed_loop",
        job_type="evaluate",
        seed=seeds[0],
        config=config,
        root=output / "experiments",
    ) as logger:
        logger.record(
            backend=str(env.unwrapped.backend),
            controller_device=str(device),
            checkpoint_sha256={
                name: [file_sha256(checkpoint.path) for checkpoint in candidate["checkpoints"]]
                for name, candidate in loaded_candidates.items()
            },
            paired_seed_count=len(seeds),
            kinematic_filter=False,
        )
        try:
            with trace_path.open("a") as stream:
                for candidate_name, controller in candidates.items():
                    use_ground_truth = loaded_candidates[candidate_name]["input"] == "gt"
                    for index, seed in enumerate(seeds, 1):
                        if (candidate_name, seed) in completed:
                            continue
                        result = _episode(
                            env,
                            seed,
                            controller,
                            None if use_ground_truth else frontend,
                            config["horizon"],
                        )
                        row = {"candidate": candidate_name, "seed": seed, **result}
                        rows.append(row)
                        completed.add((candidate_name, seed))
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        print(
                            f"{candidate_name} {index}/{len(seeds)} seed={seed} "
                            f"success={result['success']} steps={result['steps']}",
                            flush=True,
                        )
        finally:
            env.close()
        expected_results = len(candidates) * len(seeds)
        if len(rows) != expected_results or len(completed) != expected_results:
            raise RuntimeError(
                f"Closed-loop result count differs: expected {expected_results}, got {len(rows)}"
            )
        summaries = {}
        for name in candidates:
            selected = [row for row in rows if row["candidate"] == name]
            successes = sum(row["success"] for row in selected)
            summaries[name] = {
                "successes": successes,
                "episodes": len(selected),
                "success_rate": successes / len(selected),
                "wilson_95": _wilson_interval(successes, len(selected)),
                "mean_steps": float(np.mean([row["steps"] for row in selected])),
            }
            logger.log(
                {
                    "successes": successes,
                    "episodes": len(selected),
                    "success_rate": successes / len(selected),
                    "mean_steps": summaries[name]["mean_steps"],
                },
                step=0,
                prefix=name,
            )
        reference_by_seed = {
            row["seed"]: row["success"]
            for row in rows
            if row["candidate"] == comparison_reference
        }
        paired = {}
        for name in candidates:
            if name == comparison_reference:
                continue
            candidate_by_seed = {
                row["seed"]: row["success"]
                for row in rows
                if row["candidate"] == name
            }
            paired[name] = {
                "reference_only_success": sum(
                    reference_by_seed[seed] and not candidate_by_seed[seed] for seed in seeds
                ),
                "candidate_only_success": sum(
                    candidate_by_seed[seed] and not reference_by_seed[seed] for seed in seeds
                ),
            }
        summary = {
            "run_id": logger.run_id,
            "seconds": monotonic() - started,
            "measurement_gate": "all paired rollouts complete; no precommitted success-rate floor",
            "summaries": summaries,
            "comparison_reference": comparison_reference,
            "paired_vs_reference": paired,
        }
        write_json(output / "closed_loop_summary.json", summary)
        logger.link_output(trace_path)
        logger.link_output(output / "closed_loop_summary.json")


def evaluate(config: dict[str, Any]) -> None:
    """Dispatch the explicitly selected Surface-Mamba evaluation mode."""
    mode = config.get("mode")
    if mode == "offline":
        evaluate_offline(config)
    elif mode == "closed_loop":
        evaluate_closed_loop(config)
    else:
        raise ValueError("Evaluation mode must be 'offline' or 'closed_loop'")
