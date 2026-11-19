"""Offline and paired closed-loop evaluation for the Surface-Mamba ensemble."""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from envs.maniskill import get_gt_state, make_peg_insertion_env
from graph.surface_context_frontend import SurfaceContextFrontend
from recurrent.surface_mamba import (
    ControllerEnsemble,
    OnlineController,
    load_controller_checkpoint,
    state36_to_controller_features,
)
from recurrent.surface_state import from_gt_state, state36
from utils.experiment_logger import ExperimentLogger


def _load_controllers(config: dict[str, Any], device: torch.device):
    paths = [Path(path) for path in config["checkpoints"]]
    if len(paths) != 5:
        raise ValueError("Evaluation requires exactly five Mamba checkpoints")
    checkpoints = [load_controller_checkpoint(path, device) for path in paths]
    if [checkpoint.seed for checkpoint in checkpoints] != [0, 1, 2, 3, 4]:
        raise ValueError("Controller checkpoints must be ordered by seeds 0..4")
    return checkpoints, [OnlineController(checkpoint) for checkpoint in checkpoints]


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


def evaluate_offline(config: dict[str, Any]) -> None:
    """Score each model and the ensemble on clean held-out expert trajectories."""
    device = torch.device(config.get("device", "cpu"))
    checkpoints, _ = _load_controllers(config, device)
    dataset = Path(config["dataset"]).resolve()
    arrays = _read_test_split(dataset)
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    with ExperimentLogger(
        team="recurrent",
        model="surface_mamba_offline",
        job_type="evaluate",
        seed=0,
        config=config,
        root=output / "experiments",
    ) as logger:
        for input_name, state_key in (("graph", "state36_graph"), ("gt_oracle", "state36_gt")):
            predictions = [
                _predict_episodes(
                    checkpoint, arrays[state_key], arrays["episode_offsets"]
                )
                for checkpoint in checkpoints
            ]
            condition = {
                f"single_seed_{index}": _action_metrics(prediction, arrays["action_target"])
                for index, prediction in enumerate(predictions)
            }
            stacked = np.stack(predictions)
            ensemble = np.empty_like(predictions[0])
            ensemble[:, :6] = stacked[:, :, :6].mean(axis=0)
            ensemble[:, 6] = np.where((stacked[:, :, 6] > 0).sum(axis=0) >= 3, 1.0, -1.0)
            condition["ensemble"] = _action_metrics(ensemble, arrays["action_target"])
            results[input_name] = condition
            logger.log(condition["ensemble"], step=0, prefix=input_name)
        summary = {
            "run_id": logger.run_id,
            "episodes": len(arrays["reset_seed"]),
            "rows": len(arrays["action_target"]),
            "test_sha256": file_sha256(dataset / "test.npz"),
            "checkpoint_sha256": [file_sha256(Path(path)) for path in config["checkpoints"]],
            "results": results,
        }
        write_json(output / "offline_summary.json", summary)
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
    frontend: SurfaceContextFrontend | None,
    horizon: int,
) -> dict[str, Any]:
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
    return {
        "success": success,
        "steps": step + 1,
        "frontend_mean_ms": float(timing.mean()) if len(timing) else None,
    }


def _load_eval_seeds(path: str | Path) -> list[int]:
    payload = json.loads(Path(path).read_text())
    seeds = payload["seeds"] if isinstance(payload, dict) else payload
    if len(seeds) != 300 or len(set(seeds)) != 300 or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Closed-loop seed file must contain 300 distinct nonnegative integers")
    return seeds


def evaluate_closed_loop(config: dict[str, Any]) -> None:
    """Run five singles, graph ensemble and GT-input oracle on the same 300 seeds."""
    device = torch.device(config["controller_device"])
    checkpoints, controllers = _load_controllers(config, device)
    ensemble = ControllerEnsemble(controllers)
    candidates = {
        **{f"single_seed_{index}": controller for index, controller in enumerate(controllers)},
        "ensemble_graph": ensemble,
        "ensemble_gt_oracle": ensemble,
    }
    seeds = _load_eval_seeds(config["seeds_file"])
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
            checkpoint_sha256=[file_sha256(checkpoint.path) for checkpoint in checkpoints],
            paired_seed_count=len(seeds),
            kinematic_filter=False,
        )
        try:
            with trace_path.open("a") as stream:
                for candidate_name, controller in candidates.items():
                    oracle = candidate_name == "ensemble_gt_oracle"
                    for index, seed in enumerate(seeds, 1):
                        if (candidate_name, seed) in completed:
                            continue
                        result = _episode(
                            env,
                            seed,
                            controller,
                            None if oracle else frontend,
                            config["horizon"],
                        )
                        row = {"candidate": candidate_name, "seed": seed, **result}
                        rows.append(row)
                        completed.add((candidate_name, seed))
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        print(
                            f"{candidate_name} {index}/300 seed={seed} "
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
        ensemble_by_seed = {
            row["seed"]: row["success"]
            for row in rows
            if row["candidate"] == "ensemble_graph"
        }
        paired = {}
        for name in candidates:
            if name == "ensemble_graph":
                continue
            candidate_by_seed = {
                row["seed"]: row["success"]
                for row in rows
                if row["candidate"] == name
            }
            paired[name] = {
                "ensemble_only_success": sum(
                    ensemble_by_seed[seed] and not candidate_by_seed[seed] for seed in seeds
                ),
                "candidate_only_success": sum(
                    candidate_by_seed[seed] and not ensemble_by_seed[seed] for seed in seeds
                ),
            }
        summary = {
            "run_id": logger.run_id,
            "seconds": monotonic() - started,
            "measurement_gate": "all paired rollouts complete; no precommitted success-rate floor",
            "summaries": summaries,
            "paired_vs_graph_ensemble": paired,
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
