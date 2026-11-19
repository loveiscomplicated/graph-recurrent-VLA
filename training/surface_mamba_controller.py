"""Train five scratch Mamba controllers on episode-level GT/Graph mixtures."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic
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


CHECKPOINT_FORMAT = "surface_mamba_controller_v1"


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


def _checkpoint_payload(
    model: MambaController,
    optimizer: torch.optim.Optimizer,
    statistics: dict[str, np.ndarray],
    seed: int,
    step: int,
    model_config: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "model_config": model_config,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "statistics": {
            name: np.asarray(value).tolist() for name, value in statistics.items()
        },
        "seed": seed,
        "step": step,
        "identity": identity,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _train_seed(
    config: dict[str, Any],
    seed: int,
    train_levels: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    validation_levels: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    statistics: dict[str, np.ndarray],
    dataset_identity: dict[str, Any],
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    settings = config["training"]
    run_directory = output / f"mamba_seed_{seed}"
    run_directory.mkdir(parents=True, exist_ok=False)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    identity = {
        **dataset_identity,
        "seed": seed,
        "alphas": config["mixing"]["alphas"],
        "alpha_sampling": "one alpha per complete episode",
        "normalization": "training split across every configured alpha",
        "model_config": model_config,
        "steps": settings["steps"],
        "batch_size": settings["batch_size"],
        "learning_rate": settings["learning_rate"],
        "validation_interval": settings["validation_interval"],
        "kinematic_filter": False,
    }
    started = monotonic()
    history = []
    best_loss = float("inf")
    best_step = 0
    with ExperimentLogger(
        team="recurrent",
        model="surface_mamba_controller",
        job_type="train",
        seed=seed,
        config=identity,
        root=run_directory / "experiments",
    ) as logger:
        logger.record(
            device=str(device),
            torch_version=str(torch.__version__),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        )
        for step in range(1, settings["steps"] + 1):
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

            validate = step % settings["validation_interval"] == 0 or step == settings["steps"]
            if not validate:
                continue
            validation_by_alpha = [
                _validation_loss(model, episodes, settings["batch_size"], device)
                for episodes in validation_levels
            ]
            ground_truth_loss = validation_by_alpha[0]
            graph_loss = validation_by_alpha[-1]
            mixed_loss = float(np.mean(validation_by_alpha))
            row = {
                "step": step,
                "training_loss": float(loss.item()),
                "validation_graph_loss": graph_loss,
                "validation_gt_loss": ground_truth_loss,
                "validation_mixed_loss": mixed_loss,
                "validation_by_alpha": {
                    str(alpha): value
                    for alpha, value in zip(
                        config["mixing"]["alphas"], validation_by_alpha, strict=True
                    )
                },
                "elapsed_seconds": monotonic() - started,
            }
            history.append(row)
            logger.log(
                {"loss": row["training_loss"]}, step=step, prefix="train"
            )
            logger.log(
                {
                    "graph_loss": graph_loss,
                    "gt_loss": ground_truth_loss,
                    "mixed_loss": mixed_loss,
                },
                step=step,
                prefix="validation",
            )
            payload = _checkpoint_payload(
                model, optimizer, statistics, seed, step, model_config, identity
            )
            _save_checkpoint(run_directory / "last.pt", payload)
            if graph_loss < best_loss:
                best_loss = graph_loss
                best_step = step
                _save_checkpoint(run_directory / "best.pt", payload)
            write_json(
                run_directory / "history.json",
                {
                    "run_id": logger.run_id,
                    "best_step": best_step,
                    "best_graph_loss": best_loss,
                    "history": history,
                },
            )
            print(
                f"Mamba seed={seed} step={step} train={loss.item():.5f} "
                f"graph_val={graph_loss:.5f} gt_val={ground_truth_loss:.5f} "
                f"mixed_val={mixed_loss:.5f}",
                flush=True,
            )
        logger.link_output(run_directory / "best.pt", checkpoint=True)
        logger.link_output(run_directory / "last.pt", checkpoint=True)
        logger.link_output(run_directory / "history.json")
    summary = {
        "seed": seed,
        "best_step": best_step,
        "best_graph_loss": best_loss,
        "best_sha256": file_sha256(run_directory / "best.pt"),
        "last_sha256": file_sha256(run_directory / "last.pt"),
        "seconds": monotonic() - started,
    }
    write_json(run_directory / "summary.json", summary)
    return summary


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("The primary ensemble requires seeds [0,1,2,3,4]")
    alphas = config.get("mixing", {}).get("alphas")
    if alphas != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("GT/Graph mixing alphas must be [0,.25,.5,.75,1]")
    settings = config.get("training", {})
    for name in ("steps", "batch_size", "validation_interval"):
        if type(settings.get(name)) is not int or settings[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    for name in ("learning_rate", "gradient_clip_norm"):
        if not isinstance(settings.get(name), (int, float)) or settings[name] <= 0:
            raise ValueError(f"training.{name} must be positive")


def _reuse_or_preserve_run(output: Path, seed: int) -> dict[str, Any] | None:
    """Reuse a verified completed seed or preserve an incomplete run before retraining."""
    run_directory = output / f"mamba_seed_{seed}"
    if not run_directory.exists():
        return None
    summary_path = run_directory / "summary.json"
    best_path = run_directory / "best.pt"
    last_path = run_directory / "last.pt"
    if summary_path.is_file() and best_path.is_file() and last_path.is_file():
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("seed") == seed
            and summary.get("best_sha256") == file_sha256(best_path)
            and summary.get("last_sha256") == file_sha256(last_path)
        ):
            print(f"Reusing completed Mamba seed={seed}", flush=True)
            return summary
        raise ValueError(f"Completed training artifacts fail identity checks: {run_directory}")
    incomplete_root = output / "incomplete"
    incomplete_root.mkdir(exist_ok=True)
    destination = incomplete_root / f"mamba_seed_{seed}_{int(monotonic() * 1000)}"
    run_directory.rename(destination)
    print(f"Preserved incomplete Mamba seed={seed} at {destination}", flush=True)
    return None


def train(config: dict[str, Any]) -> None:
    """Train five independent models from scratch and select graph-validation bests."""
    _validate_config(config)
    dataset = Path(config["dataset"]).resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "surface_mamba_state36_v1":
        raise ValueError("Expected a derived Surface-Mamba State36 dataset")
    train_arrays = _read_split(dataset / "train.npz")
    validation_arrays = _read_split(dataset / "validation.npz")
    if len(train_arrays["reset_seed"]) != 720 or len(validation_arrays["reset_seed"]) != 90:
        raise ValueError("Training/validation episode counts must be 720/90")
    if set(train_arrays["reset_seed"]) & set(validation_arrays["reset_seed"]):
        raise ValueError("Training and validation reset seeds overlap")

    alphas = tuple(float(value) for value in config["mixing"]["alphas"])
    raw_train_levels = _raw_episodes(train_arrays, alphas)
    statistics = compute_normalization(
        [episode for level in raw_train_levels for episode in level]
    )
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
            )
        summaries.append(summary)
    write_json(output / "ensemble_summary.json", {"models": summaries, **identity})
