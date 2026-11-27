"""Train scratch Mamba controllers on configurable episode-level state mixtures."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic, time_ns
from typing import Any

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from recurrent.surface_mamba import (
    ACTION_DIM,
    DEFAULT_MODEL_CONFIG,
    OBSERVATION_DIM,
    MambaController,
    behavior_cloning_loss,
    compute_normalization,
    mix_state36,
    normalize_episode,
    pad_episodes,
    state36_to_controller_features,
)
from utils.experiment_logger import ExperimentLogger


CHECKPOINT_FORMAT = "surface_mamba_controller_v2"


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    return device


def _read_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "state36_gt",
            "state36_graph",
            "action_target",
            "episode_offsets",
            "reset_seed",
        }
        if not required <= set(archive.files):
            raise ValueError(f"Dataset split is missing arrays: {path}")
        arrays = {key: archive[key] for key in required}
    gt = arrays["state36_gt"]
    graph = arrays["state36_graph"]
    action = arrays["action_target"]
    offsets = arrays["episode_offsets"]
    if (
        gt.shape != graph.shape
        or gt.ndim != 2
        or gt.shape[1] != 36
        or action.shape != (len(gt), ACTION_DIM)
        or offsets.ndim != 1
        or offsets[0] != 0
        or offsets[-1] != len(gt)
        or np.any(np.diff(offsets) <= 0)
        or len(offsets) != len(arrays["reset_seed"]) + 1
    ):
        raise ValueError(f"Dataset split has invalid State36/action offsets: {path}")
    if not all(np.isfinite(array).all() for array in (gt, graph, action)):
        raise ValueError(f"Dataset split contains nonfinite values: {path}")
    return arrays


def _raw_episodes(
    arrays: dict[str, np.ndarray], alphas: tuple[float, ...]
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Return one aligned episode list per alpha without crossing episode boundaries."""
    levels = [[] for _ in alphas]
    offsets = arrays["episode_offsets"]
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        ground_truth = arrays["state36_gt"][start:stop]
        graph = arrays["state36_graph"][start:stop]
        action = arrays["action_target"][start:stop]
        for index, alpha in enumerate(alphas):
            state = mix_state36(ground_truth, graph, alpha)
            levels[index].append((state36_to_controller_features(state), action))
    return levels


def _normalized_levels(
    levels: list[list[tuple[np.ndarray, np.ndarray]]],
    statistics: dict[str, np.ndarray],
) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    return [
        [normalize_episode(observation, action, statistics) for observation, action in level]
        for level in levels
    ]


@torch.inference_mode()
def _validation_loss(
    model: MambaController,
    episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    timesteps = 0.0
    for start in range(0, len(episodes), batch_size):
        observation, action, gripper, mask = pad_episodes(
            episodes[start : start + batch_size], device
        )
        prediction, _ = model(observation)
        count = float(mask.sum().item())
        weighted_loss += float(
            behavior_cloning_loss(prediction, action, gripper, mask).item()
        ) * count
        timesteps += count
    return weighted_loss / timesteps


def _validation_metrics(
    model: MambaController,
    validation_levels: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    alphas: list[float],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[float, float], float | None, float, float]:
    """Evaluate every configured observation mixture and identify Graph loss."""
    validation_by_alpha = [
        _validation_loss(model, episodes, batch_size, device)
        for episodes in validation_levels
    ]
    loss_by_alpha = dict(zip(alphas, validation_by_alpha, strict=True))
    ground_truth_loss = loss_by_alpha.get(0.0)
    graph_loss = loss_by_alpha[1.0]
    mixed_loss = float(np.mean(validation_by_alpha))
    return loss_by_alpha, ground_truth_loss, graph_loss, mixed_loss


def _checkpoint_payload(
    model: MambaController,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    statistics: dict[str, np.ndarray],
    seed: int,
    step: int,
    model_config: dict[str, Any],
    identity: dict[str, Any],
    generator: np.random.Generator,
    progress: dict[str, Any],
) -> dict[str, Any]:
    """Capture everything required to reproduce the next optimizer update."""
    rng = {
        "numpy_generator": generator.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        rng["torch_mps"] = torch.mps.get_rng_state()

    return {
        "format": CHECKPOINT_FORMAT,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "model_config": model_config,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "statistics": {
            name: np.asarray(value).tolist() for name, value in statistics.items()
        },
        "seed": seed,
        "step": step,
        "identity": identity,
        "rng": rng,
        "progress": progress,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _build_optimizer(
    model: MambaController, settings: dict[str, Any]
) -> torch.optim.Optimizer:
    """Construct the configured optimizer without hard-coding Adam behavior."""
    optimizer_settings = settings["optimizer"]
    arguments = {
        "lr": float(optimizer_settings["learning_rate"]),
        "weight_decay": float(optimizer_settings["weight_decay"]),
    }
    if optimizer_settings["type"] == "adam":
        return torch.optim.Adam(model.parameters(), **arguments)
    return torch.optim.AdamW(model.parameters(), **arguments)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, settings: dict[str, Any]
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build the validation-loss scheduler used by every comparison run."""
    scheduler_settings = settings["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_settings["factor"]),
        patience=int(scheduler_settings["patience_validations"]),
        threshold=float(settings["early_stopping"]["min_delta"]),
        threshold_mode="abs",
        min_lr=float(scheduler_settings["min_learning_rate"]),
    )


def _initial_progress() -> dict[str, Any]:
    """Return serializable fresh-run convergence and segment state."""
    return {
        "history": [],
        "run_ids": [],
        "global_best_loss": float("inf"),
        "global_best_step": 0,
        "early_stopping_best_loss": float("inf"),
        "validations_without_improvement": 0,
        "completed_segments": [],
        "current_segment": {
            "index": 0,
            "start_step": 0,
            "best_graph_loss": float("inf"),
            "best_step": 0,
        },
    }


def _restore_checkpoint(
    path: Path,
    model: MambaController,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    generator: np.random.Generator,
    expected_seed: int,
    expected_model_config: dict[str, Any],
    expected_identity: dict[str, Any],
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    """Restore an exact v2 training boundary."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported continuation checkpoint format: {path}")
    if checkpoint.get("seed") != expected_seed:
        raise ValueError(f"Continuation checkpoint seed differs: {path}")
    if checkpoint.get("model_config") != expected_model_config:
        raise ValueError(f"Continuation checkpoint model config differs: {path}")
    if checkpoint.get("identity") != expected_identity:
        raise ValueError(f"Continuation checkpoint experiment identity differs: {path}")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    generator.bit_generator.state = checkpoint["rng"]["numpy_generator"]
    torch.set_rng_state(checkpoint["rng"]["torch_cpu"].cpu())
    if device.type == "cuda":
        cuda_states = checkpoint["rng"].get("torch_cuda")
        if cuda_states is None:
            raise ValueError(f"CUDA RNG state is missing from continuation checkpoint: {path}")
        torch.cuda.set_rng_state_all(cuda_states)
    if device.type == "mps" and "torch_mps" in checkpoint["rng"]:
        torch.mps.set_rng_state(checkpoint["rng"]["torch_mps"])

    step = int(checkpoint["step"])
    progress = checkpoint["progress"]
    return step, progress


def _begin_resume_segment(progress: dict[str, Any], step: int) -> None:
    """Close the interrupted segment and begin a separately tracked continuation."""
    previous_segment = dict(progress["current_segment"])
    previous_segment.update(end_step=step, stop_reason="interrupted")
    progress["completed_segments"].append(previous_segment)
    progress["current_segment"] = {
        "index": previous_segment["index"] + 1,
        "start_step": step,
        "best_graph_loss": float("inf"),
        "best_step": 0,
    }


def _history_document(
    progress: dict[str, Any], final_step: int, stop_reason: str | None = None
) -> dict[str, Any]:
    """Create the human-readable view of checkpointed training progress."""
    document = {
        "run_ids": progress["run_ids"],
        "global_best_step": progress["global_best_step"],
        "global_best_graph_loss": progress["global_best_loss"],
        "early_stopping_best_graph_loss": progress["early_stopping_best_loss"],
        "validations_without_improvement": progress[
            "validations_without_improvement"
        ],
        "completed_segments": progress["completed_segments"],
        "current_segment": progress["current_segment"],
        "final_step": final_step,
        "history": progress["history"],
    }
    if stop_reason is not None:
        document["stop_reason"] = stop_reason
    return document


def _save_training_state(
    run_directory: Path,
    model: MambaController,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    statistics: dict[str, np.ndarray],
    seed: int,
    step: int,
    model_config: dict[str, Any],
    identity: dict[str, Any],
    generator: np.random.Generator,
    progress: dict[str, Any],
    *,
    save_periodic: bool,
    save_global_best: bool,
    save_segment_best: bool,
) -> None:
    """Atomically save requested aliases from one consistent state snapshot."""
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        statistics,
        seed,
        step,
        model_config,
        identity,
        generator,
        progress,
    )
    if save_global_best:
        _save_checkpoint(run_directory / "best.pt", payload)
    if save_segment_best:
        segment_index = progress["current_segment"]["index"]
        _save_checkpoint(
            run_directory / "segments" / f"segment_{segment_index:03d}_best.pt",
            payload,
        )
        _save_checkpoint(run_directory / "segment_best.pt", payload)
    if save_periodic:
        _save_checkpoint(
            run_directory / "checkpoints" / f"step_{step:06d}.pt", payload
        )
    # Save last after the archival aliases so last.pt always points to a state whose
    # corresponding best/periodic files have already been committed.
    _save_checkpoint(run_directory / "last.pt", payload)


def _train_seed(
    config: dict[str, Any],
    seed: int,
    train_levels: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    validation_levels: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    statistics: dict[str, np.ndarray],
    dataset_identity: dict[str, Any],
    device: torch.device,
    output: Path,
    initial_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = config["training"]
    run_directory = output / f"mamba_seed_{seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = np.random.default_rng(seed)
    model_settings = {**DEFAULT_MODEL_CONFIG, **config.get("model", {})}
    model_config = {
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        **model_settings,
    }
    model = MambaController(**model_config).to(device)
    optimizer = _build_optimizer(model, settings)
    scheduler = _build_scheduler(optimizer, settings)
    alphas = [float(alpha) for alpha in config["mixing"]["alphas"]]
    identity = {
        **dataset_identity,
        "seed": seed,
        "alphas": alphas,
        "alpha_sampling": "one alpha per complete episode",
        "normalization": "training split across every configured alpha",
        "model_config": model_config,
        "training": settings,
        "batch_size": settings["batch_size"],
        "kinematic_filter": False,
    }
    if initial_checkpoint is not None:
        identity["warm_start_sha256"] = initial_checkpoint["sha256"]
        identity["normalization"] = "fixed from the source checkpoint"
        source = initial_checkpoint["payload"]
        if source["seed"] != seed or source["model_config"] != model_config:
            raise ValueError("Warm-start checkpoint has a different seed or model configuration")
        identity["source_alphas"] = source.get("identity", {}).get("alphas")
        identity["fine_tuning_alphas"] = alphas
        model.load_state_dict(source["model"], strict=True)
    started = monotonic()
    progress = _initial_progress()
    start_step = 0
    last_path = run_directory / "last.pt"
    if last_path.is_file():
        start_step, progress = _restore_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            generator,
            seed,
            model_config,
            identity,
            device,
        )
        print(f"Resuming Mamba seed={seed} from exact step={start_step}", flush=True)

    stopping_settings = settings["early_stopping"]
    stopped_before_restart = (
        progress["validations_without_improvement"]
        >= stopping_settings["patience_validations"]
    )
    work_remains = start_step < settings["max_steps"] and not stopped_before_restart
    if start_step > 0 and work_remains:
        _begin_resume_segment(progress, start_step)

    stop_reason = "early_stopping" if stopped_before_restart else "max_steps_safety_cap"
    final_step = start_step
    with ExperimentLogger(
        team="recurrent",
        model="surface_mamba_controller",
        job_type="train",
        seed=seed,
        config=identity,
        root=run_directory / "experiments",
    ) as logger:
        progress["run_ids"].append(logger.run_id)
        logger.record(
            device=str(device),
            torch_version=str(torch.__version__),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        )
        if initial_checkpoint is not None and not last_path.is_file():
            (
                loss_by_alpha,
                ground_truth_loss,
                graph_loss,
                mixed_loss,
            ) = _validation_metrics(
                model,
                validation_levels,
                alphas,
                settings["batch_size"],
                device,
            )
            scheduler.step(graph_loss)
            progress["global_best_loss"] = graph_loss
            progress["global_best_step"] = 0
            progress["early_stopping_best_loss"] = graph_loss
            progress["validations_without_improvement"] = 0
            progress["current_segment"]["best_graph_loss"] = graph_loss
            progress["current_segment"]["best_step"] = 0

            current_learning_rate = float(optimizer.param_groups[0]["lr"])
            initial_row = {
                "step": 0,
                "training_loss": None,
                "validation_graph_loss": graph_loss,
                "validation_gt_loss": ground_truth_loss,
                "validation_mixed_loss": mixed_loss,
                "validation_by_alpha": {
                    str(alpha): value for alpha, value in loss_by_alpha.items()
                },
                "elapsed_seconds": monotonic() - started,
                "learning_rate_before_scheduler": current_learning_rate,
                "learning_rate": current_learning_rate,
                "global_best_step": 0,
                "validations_without_improvement": 0,
                "initial_checkpoint": True,
            }
            progress["history"].append(initial_row)
            logger.log(
                {"graph_loss": graph_loss, "mixed_loss": mixed_loss},
                step=0,
                prefix="validation",
            )
            _save_training_state(
                run_directory,
                model,
                optimizer,
                scheduler,
                statistics,
                seed,
                0,
                model_config,
                identity,
                generator,
                progress,
                save_periodic=False,
                save_global_best=True,
                save_segment_best=True,
            )
            write_json(
                run_directory / "history.json",
                _history_document(progress, 0),
            )
            print(
                f"Mamba seed={seed} step=0 warm_start "
                f"graph_val={graph_loss:.5f} mixed_val={mixed_loss:.5f} "
                f"lr={current_learning_rate:.3g}",
                flush=True,
            )

        final_training_step = settings["max_steps"] if work_remains else start_step
        for step in range(start_step + 1, final_training_step + 1):
            final_step = step
            model.train()
            count = min(settings["batch_size"], len(train_levels[0]))
            episode_indices = generator.choice(len(train_levels[0]), count, replace=False)
            alpha_indices = generator.integers(len(train_levels), size=count)
            batch = [
                train_levels[alpha_index][episode_index]
                for episode_index, alpha_index in zip(
                    episode_indices, alpha_indices, strict=True
                )
            ]
            observation, action, gripper, mask = pad_episodes(batch, device)
            prediction, _ = model(observation)
            loss = behavior_cloning_loss(prediction, action, gripper, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(settings["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()

            validate = (
                step % settings["validation_interval"] == 0
                or step == settings["max_steps"]
            )
            if not validate:
                if step % settings["checkpoint_interval"] == 0:
                    _save_training_state(
                        run_directory,
                        model,
                        optimizer,
                        scheduler,
                        statistics,
                        seed,
                        step,
                        model_config,
                        identity,
                        generator,
                        progress,
                        save_periodic=True,
                        save_global_best=False,
                        save_segment_best=False,
                    )
                continue
            (
                loss_by_alpha,
                ground_truth_loss,
                graph_loss,
                mixed_loss,
            ) = _validation_metrics(
                model,
                validation_levels,
                alphas,
                settings["batch_size"],
                device,
            )
            previous_learning_rate = float(optimizer.param_groups[0]["lr"])
            scheduler.step(graph_loss)
            current_learning_rate = float(optimizer.param_groups[0]["lr"])

            global_best_improved = graph_loss < progress["global_best_loss"]
            if global_best_improved:
                progress["global_best_loss"] = graph_loss
                progress["global_best_step"] = step

            minimum_improvement = float(settings["early_stopping"]["min_delta"])
            convergence_improved = (
                graph_loss
                < progress["early_stopping_best_loss"] - minimum_improvement
            )
            if convergence_improved:
                progress["early_stopping_best_loss"] = graph_loss
                progress["validations_without_improvement"] = 0
            else:
                progress["validations_without_improvement"] += 1

            segment = progress["current_segment"]
            segment_best_improved = graph_loss < segment["best_graph_loss"]
            if segment_best_improved:
                segment["best_graph_loss"] = graph_loss
                segment["best_step"] = step

            row = {
                "step": step,
                "training_loss": float(loss.item()),
                "validation_graph_loss": graph_loss,
                "validation_gt_loss": ground_truth_loss,
                "validation_mixed_loss": mixed_loss,
                "validation_by_alpha": {
                    str(alpha): value
                    for alpha, value in loss_by_alpha.items()
                },
                "elapsed_seconds": monotonic() - started,
                "learning_rate_before_scheduler": previous_learning_rate,
                "learning_rate": current_learning_rate,
                "global_best_step": progress["global_best_step"],
                "validations_without_improvement": progress[
                    "validations_without_improvement"
                ],
            }
            progress["history"].append(row)
            logger.log(
                {"loss": row["training_loss"]}, step=step, prefix="train"
            )
            validation_metrics = {
                "graph_loss": graph_loss,
                "mixed_loss": mixed_loss,
            }
            if ground_truth_loss is not None:
                validation_metrics["gt_loss"] = ground_truth_loss
            logger.log(validation_metrics, step=step, prefix="validation")
            _save_training_state(
                run_directory,
                model,
                optimizer,
                scheduler,
                statistics,
                seed,
                step,
                model_config,
                identity,
                generator,
                progress,
                save_periodic=step % settings["checkpoint_interval"] == 0,
                save_global_best=global_best_improved,
                save_segment_best=segment_best_improved,
            )
            write_json(
                run_directory / "history.json",
                _history_document(progress, step),
            )
            ground_truth_text = (
                "n/a" if ground_truth_loss is None else f"{ground_truth_loss:.5f}"
            )
            print(
                f"Mamba seed={seed} step={step} train={loss.item():.5f} "
                f"graph_val={graph_loss:.5f} "
                f"gt_val={ground_truth_text} "
                f"mixed_val={mixed_loss:.5f} lr={current_learning_rate:.3g}",
                flush=True,
            )
            if (
                progress["validations_without_improvement"]
                >= settings["early_stopping"]["patience_validations"]
            ):
                stop_reason = "early_stopping"
                break
        current_segment = progress["current_segment"]
        current_segment["end_step"] = final_step
        current_segment["stop_reason"] = stop_reason
        write_json(
            run_directory / "history.json",
            _history_document(progress, final_step, stop_reason),
        )
        logger.link_output(run_directory / "best.pt", checkpoint=True)
        logger.link_output(run_directory / "last.pt", checkpoint=True)
        logger.link_output(run_directory / "segment_best.pt", checkpoint=True)
        logger.link_output(run_directory / "history.json")

    summary = {
        "seed": seed,
        "stop_reason": stop_reason,
        "final_step": final_step,
        "best_step": progress["global_best_step"],
        "best_graph_loss": progress["global_best_loss"],
        "global_best_step": progress["global_best_step"],
        "global_best_graph_loss": progress["global_best_loss"],
        "segment_index": current_segment["index"],
        "segment_best_step": current_segment["best_step"],
        "segment_best_graph_loss": current_segment["best_graph_loss"],
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "best_sha256": file_sha256(run_directory / "best.pt"),
        "last_sha256": file_sha256(run_directory / "last.pt"),
        "segment_best_sha256": file_sha256(run_directory / "segment_best.pt"),
        "seconds": monotonic() - started,
    }
    write_json(run_directory / "summary.json", summary)
    return summary


def _validate_config(config: dict[str, Any]) -> None:
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain distinct values")

    alphas = config.get("mixing", {}).get("alphas")
    if not isinstance(alphas, list) or not alphas:
        raise ValueError("mixing.alphas must be a nonempty list")
    if any(
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not np.isfinite(alpha)
        or not 0.0 <= float(alpha) <= 1.0
        for alpha in alphas
    ):
        raise ValueError("mixing.alphas values must be finite and between 0.0 and 1.0")
    numeric_alphas = [float(alpha) for alpha in alphas]
    if any(left >= right for left, right in zip(numeric_alphas[:-1], numeric_alphas[1:])):
        raise ValueError("mixing.alphas must be strictly increasing")
    if 1.0 not in numeric_alphas:
        raise ValueError("mixing.alphas must include 1.0 for graph-input checkpoint selection")

    settings = config.get("training", {})
    for name in (
        "max_steps",
        "batch_size",
        "validation_interval",
        "checkpoint_interval",
    ):
        if type(settings.get(name)) is not int or settings[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    gradient_clip_norm = settings.get("gradient_clip_norm")
    if (
        not isinstance(gradient_clip_norm, (int, float))
        or isinstance(gradient_clip_norm, bool)
        or not np.isfinite(gradient_clip_norm)
        or gradient_clip_norm <= 0
    ):
        raise ValueError("training.gradient_clip_norm must be finite and positive")

    optimizer = settings.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("type") not in {"adam", "adamw"}:
        raise ValueError("training.optimizer.type must be 'adam' or 'adamw'")
    learning_rate = optimizer.get("learning_rate")
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("training.optimizer.learning_rate must be finite and positive")
    weight_decay = optimizer.get("weight_decay")
    if (
        not isinstance(weight_decay, (int, float))
        or isinstance(weight_decay, bool)
        or not np.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("training.optimizer.weight_decay must be finite and nonnegative")

    early_stopping = settings.get("early_stopping")
    if not isinstance(early_stopping, dict):
        raise ValueError("training.early_stopping must be a mapping")
    if (
        type(early_stopping.get("patience_validations")) is not int
        or early_stopping["patience_validations"] < 1
    ):
        raise ValueError(
            "training.early_stopping.patience_validations must be a positive integer"
        )
    minimum_delta = early_stopping.get("min_delta")
    if (
        not isinstance(minimum_delta, (int, float))
        or isinstance(minimum_delta, bool)
        or not np.isfinite(minimum_delta)
        or minimum_delta < 0
    ):
        raise ValueError("training.early_stopping.min_delta must be finite and nonnegative")

    scheduler = settings.get("scheduler")
    if not isinstance(scheduler, dict) or scheduler.get("type") != "reduce_on_plateau":
        raise ValueError("training.scheduler.type must be 'reduce_on_plateau'")
    factor = scheduler.get("factor")
    if (
        not isinstance(factor, (int, float))
        or isinstance(factor, bool)
        or not np.isfinite(factor)
        or not 0 < factor < 1
    ):
        raise ValueError("training.scheduler.factor must be between 0 and 1")
    if (
        type(scheduler.get("patience_validations")) is not int
        or scheduler["patience_validations"] < 0
    ):
        raise ValueError(
            "training.scheduler.patience_validations must be a nonnegative integer"
        )
    minimum_learning_rate = scheduler.get("min_learning_rate")
    if (
        not isinstance(minimum_learning_rate, (int, float))
        or isinstance(minimum_learning_rate, bool)
        or not np.isfinite(minimum_learning_rate)
        or not 0 <= minimum_learning_rate < learning_rate
    ):
        raise ValueError(
            "training.scheduler.min_learning_rate must be nonnegative and below the optimizer learning rate"
        )


def _reuse_or_preserve_run(output: Path, seed: int) -> dict[str, Any] | None:
    """Reuse a completed seed or leave an exact v2 checkpoint ready to resume."""
    run_directory = output / f"mamba_seed_{seed}"
    if not run_directory.exists():
        return None
    summary_path = run_directory / "summary.json"
    best_path = run_directory / "best.pt"
    last_path = run_directory / "last.pt"
    segment_best_path = run_directory / "segment_best.pt"
    completed_files_exist = all(
        path.is_file()
        for path in (summary_path, best_path, last_path, segment_best_path)
    )
    if completed_files_exist:
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("seed") == seed
            and summary.get("best_sha256") == file_sha256(best_path)
            and summary.get("last_sha256") == file_sha256(last_path)
            and summary.get("segment_best_sha256")
            == file_sha256(segment_best_path)
        ):
            print(f"Reusing completed Mamba seed={seed}", flush=True)
            return summary
        raise ValueError(f"Completed training artifacts fail identity checks: {run_directory}")
    if summary_path.exists():
        raise ValueError(f"Completed training artifacts are incomplete: {run_directory}")
    if last_path.is_file():
        print(f"Found resumable incomplete Mamba seed={seed}", flush=True)
        return None
    incomplete_root = output / "incomplete"
    incomplete_root.mkdir(exist_ok=True)
    destination = incomplete_root / f"mamba_seed_{seed}_{time_ns()}"
    run_directory.rename(destination)
    print(f"Preserved incomplete Mamba seed={seed} at {destination}", flush=True)
    return None


def train(
    config: dict[str, Any],
    *,
    dataset_format: str = "surface_mamba_state36_v1",
    expected_counts: tuple[int, int] = (720, 90),
    warm_starts: dict[int, dict[str, Any]] | None = None,
) -> None:
    """Train configured independent models and select graph-validation bests."""
    _validate_config(config)
    dataset = Path(config["dataset"]).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != dataset_format:
        raise ValueError(f"Expected a {dataset_format} dataset")
    train_arrays = _read_split(dataset / "train.npz")
    validation_arrays = _read_split(dataset / "validation.npz")
    counts = (len(train_arrays["reset_seed"]), len(validation_arrays["reset_seed"]))
    if counts != expected_counts:
        raise ValueError(f"Training/validation episode counts must be {expected_counts}")
    if set(train_arrays["reset_seed"]) & set(validation_arrays["reset_seed"]):
        raise ValueError("Training and validation reset seeds overlap")

    alphas = tuple(float(value) for value in config["mixing"]["alphas"])
    raw_train_levels = _raw_episodes(train_arrays, alphas)
    if warm_starts is None:
        statistics = compute_normalization(
            [episode for level in raw_train_levels for episode in level]
        )
    else:
        if set(warm_starts) != set(config["seeds"]):
            raise ValueError("A warm-start checkpoint is required for each configured seed")
        sources = [entry["payload"]["statistics"] for entry in warm_starts.values()]
        if any(source != sources[0] for source in sources):
            raise ValueError("Warm-start checkpoints use different normalization")
        statistics = {name: np.asarray(value, dtype=np.float32) for name, value in sources[0].items()}
    train_levels = _normalized_levels(raw_train_levels, statistics)
    validation_levels = _normalized_levels(
        _raw_episodes(validation_arrays, alphas), statistics
    )
    output = Path(config["output"]).resolve()
    effective_path = output / "effective_config.json"
    if output.exists():
        if not config.get("resume"):
            raise FileExistsError(f"Training output already exists: {output}")
        if json.loads(effective_path.read_text()) != config:
            raise ValueError("Training resume requires an identical effective config")
    else:
        output.mkdir(parents=True)
        write_json(effective_path, config)
        write_json(
            output / "normalization.json",
            {name: value.tolist() for name, value in statistics.items()},
        )
    identity = {
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "train_sha256": file_sha256(dataset / "train.npz"),
        "validation_sha256": file_sha256(dataset / "validation.npz"),
        "train_episodes": len(train_arrays["reset_seed"]),
        "validation_episodes": len(validation_arrays["reset_seed"]),
    }
    device = _device(config.get("device", "auto"))
    torch.set_num_threads(int(config.get("cpu_threads", 2)))
    summaries = []
    for seed in config["seeds"]:
        summary = _reuse_or_preserve_run(output, seed)
        if summary is None:
            summary = _train_seed(
                config,
                seed,
                train_levels,
                validation_levels,
                statistics,
                identity,
                device,
                output,
                initial_checkpoint=None if warm_starts is None else warm_starts[seed],
            )
        summaries.append(summary)
    write_json(output / "ensemble_summary.json", {"models": summaries, **identity})
