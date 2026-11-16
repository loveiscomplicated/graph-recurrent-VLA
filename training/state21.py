"""Train Object21D estimators and select checkpoints using validation alone."""

import json
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.state21 import load_dataset
from data_collection.state21_batches import PreparedWindows
from graph.state_estimation.models import MODELS, build_model
from training.gnn.checkpoint import load_checkpoint, save_checkpoint
from training.state21_loss import state21_loss
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def effective_config(config):
    """Resolve team defaults and an optional shared YAML before logging settings."""
    config = deepcopy(config)
    base_path = config.pop("base_config", None)
    if base_path is not None:
        from main import load_config

        base_file = Path(base_path)
        base = load_config(base_file)
        if "base_config" in base:
            raise ValueError("Object21D config supports one base_config")
        overrides = deepcopy(config)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        config = base
        config["config_overrides"] = {"base_config": str(base_file), **overrides}
    model = config["model"]
    for key, value in {
        "hidden": 64,
        "layers": 2,
        "dropout": 0.0,
        "heads": 4,
        "tbptt_steps": 0,
    }.items():
        model.setdefault(key, value)
    data = config["data"]
    data.setdefault("max_edge_m", 0.08)
    data.setdefault("num_workers", 0)
    if data["num_workers"] != 0:
        raise ValueError("Prepared graph batches require data.num_workers=0")
    model["history"] = data["history"]
    config["task"].setdefault("pos_unit_m", 0.01)
    config["task"].setdefault("shape_weight", 0.0)
    config["training"].setdefault("amp", False)
    config["training"].setdefault("cpu_threads", 1)
    config.setdefault("scheduler", None)
    if config["scheduler"] is not None:
        raise ValueError("Object21D currently uses constant learning rate")
    return config


def prepare_dataset(episodes, config, split):
    """Create independently indexed windows for one split."""
    selected = [episode for episode in episodes if episode.split == split]
    if not selected:
        raise ValueError(f"Dataset has no {split} episodes")
    data = config["data"]
    # A is stateless, but using the same batches preserves the matched training budget.
    return PreparedWindows(
        selected, history=data["history"], max_edge_m=data["max_edge_m"]
    )


def run_epoch(model, data, settings, device, *, optimizer=None, generator=None):
    """Train or validate all windows, with metrics weighted by actual sample counts."""
    training = optimizer is not None
    model.train(training)
    options = settings["training"]
    progress_interval = options.get("progress_every_batches", 0)
    mixed_precision = options.get("amp", False)
    if mixed_precision and device.type not in ("cpu", "cuda"):
        raise ValueError("BF16 autocast is supported by this workflow on CPU or CUDA")
    if mixed_precision and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support BF16")
    totals = {}
    count = 0
    with torch.set_grad_enabled(training):
        batches = data.batches(options["batch_size"], device, generator)
        for batch_index, (frames, targets, _) in enumerate(batches, start=1):
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=mixed_precision
            ):
                output = model(frames)
            terms = state21_loss(output, targets, **settings["task"])
            loss = terms["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Object21D loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    options["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(
                    targets
                )
            count += len(targets)
            if progress_interval and (
                batch_index == 1
                or batch_index % progress_interval == 0
                or count == len(data.windows)
            ):
                stage = "train" if training else "validation"
                print(
                    f"{stage} batch {batch_index} frames {count}/{len(data.windows)}",
                    flush=True,
                )
    if count == 0:
        raise ValueError("No Object21D windows to process")
    return {name: value / count for name, value in totals.items()}


def dataset_identity(data, report):
    """Lock source and graph preparation settings when resuming or evaluating."""
    return canonical_hash({"manifest": report["manifest_sha256"], "data": data})


@dataclass(frozen=True)
class TrainingData:
    """One prepared dataset shared by runs with the same data configuration."""

    settings: dict
    train: PreparedWindows
    validation: PreparedWindows
    report: dict


def _prepare_training_data(config):
    """Load and graph the train/validation splits once for the comparison."""
    settings = config["data"]
    episodes, report = load_dataset(settings, splits=("train", "validation"))
    if report["near_duplicate_tcp_trajectories"]:
        report["review_status"] = "near_duplicate_diagnostic_requires_review"
    train = prepare_dataset(episodes, config, "train")
    validation = prepare_dataset(episodes, config, "validation")
    return TrainingData(deepcopy(settings), train, validation, report)


def _early_stopping_settings(options):
    """Validate the optional stopping policy without changing the loss recipe."""
    settings = options.get("early_stopping")
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) != {
        "patience",
        "min_epochs",
        "min_delta",
    }:
        raise ValueError(
            "training.early_stopping requires patience, min_epochs, and min_delta"
        )
    for name in ("patience", "min_epochs"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(
                f"training.early_stopping.{name} must be a positive integer"
            )
    delta = settings["min_delta"]
    if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
        raise ValueError(
            "training.early_stopping.min_delta must be finite and nonnegative"
        )
    return settings


def fit(config, *, prepared=None):
    """Run one new or resumed experiment and return its checkpoint directory."""
    config = effective_config(config)
    started = perf_counter()
    seed = config["seed"]
    device = set_device(config["device"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    if prepared is None:
        prepared = _prepare_training_data(config)
    if prepared.settings != config["data"]:
        raise ValueError("Prepared dataset and run data configuration differ")
    train, validation, report = prepared.train, prepared.validation, prepared.report
    identifier = dataset_identity(config["data"], report)
    model = build_model(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    first_epoch, best_loss, best_state = 1, float("inf"), None
    resume = config.get("resume")
    initial_checkpoint = config.get("initial_checkpoint")
    if initial_checkpoint and resume:
        raise ValueError("Select an initial checkpoint or a resume checkpoint")
    if initial_checkpoint:
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if initial["config"]["model"] != config["model"]:
            raise ValueError("Initial checkpoint model settings differ")
        model.load_state_dict(initial["best_model"], strict=True)
    options = config["training"]
    epochs = options["epochs"]
    if type(epochs) is not int or epochs < 1:
        raise ValueError("training.epochs must be a positive integer")
    early = _early_stopping_settings(options)
    early_state = None
    if early:
        from training.gnn.temporal.insertion import update_early_stopping

        early_state = {"reference_loss": best_loss, "stale_epochs": 0}

    with ExperimentLogger(
        team="graph", model=config["model"]["name"], config=config, seed=seed
    ) as experiment:
        output = Path(config["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        last_path = output / "last.pt"
        experiment.record(
            dataset_sha256=identifier,
            data_audit=report,
            device=str(device),
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            graph_methods={
                "train": dict(train.graph_counts),
                "validation": dict(validation.graph_counts),
            },
            long_edges_rejected={
                "train": train.rejected_edges,
                "validation": validation.rejected_edges,
            },
            history_saved_frames=1
            if config["model"]["name"] == "spatial"
            else config["data"]["history"],
        )
        if initial_checkpoint:
            experiment.link_output(initial_checkpoint, checkpoint=True)
            experiment.record(initial_checkpoint_sha256=file_sha256(initial_checkpoint))
        if resume:
            # Include mixed precision in the existing checkpoint recipe signature.
            loaded = load_checkpoint(
                resume,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                config=_checkpoint_config(config),
                dataset_sha256=identifier,
                device=device,
                shuffle_generator=generator,
            )
            first_epoch = loaded["epoch"] + 1
            best_loss, best_state = loaded["best_loss"], loaded["best_model"]
            experiment.link_output(resume, checkpoint=True)
            if early:
                # A checkpoint from before early stopping provides the best known
                # validation loss. The newly configured patience starts here.
                if (
                    "early_stopping" in loaded["resume_signature"]
                    and "early_stopping_state" not in loaded
                ):
                    raise ValueError("Resume checkpoint lacks its early stopping state")
                early_state = loaded.get(
                    "early_stopping_state",
                    {"reference_loss": best_loss, "stale_epochs": 0},
                )
                if (
                    loaded["epoch"] >= early["min_epochs"]
                    and early_state["stale_epochs"] >= early["patience"]
                ):
                    raise ValueError(
                        "Resume checkpoint has already met its early stopping criterion"
                    )

        stopped_early = False
        for epoch in range(first_epoch, epochs + 1):
            if options.get("progress_every_batches"):
                print(f"{config['model']['name']} epoch {epoch}/{epochs}", flush=True)
            train_terms = run_epoch(
                model, train, config, device, optimizer=optimizer, generator=generator
            )
            val_terms = run_epoch(model, validation, config, device)
            experiment.log({"train": train_terms, "validation": val_terms}, step=epoch)
            validation_loss = val_terms["state21"]
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if early:
                early_state = update_early_stopping(early_state, validation_loss, early)
                stale = early_state["stale_epochs"]
                print(
                    f"Epoch {epoch} validation Object21D {validation_loss:.6f}, "
                    f"best {best_loss:.6f}, patience {stale}/{early['patience']}",
                    flush=True,
                )
                stopped_early = (
                    epoch >= early["min_epochs"] and stale >= early["patience"]
                )
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                best_loss=best_loss,
                best_model_state=best_state,
                config=_checkpoint_config(config),
                dataset_sha256=identifier,
                device=device,
                shuffle_generator=generator,
                early_stopping_state=early_state,
            )
            if stopped_early:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
        experiment.link_output(last_path)
        summary = {
            "epochs": epoch,
            "train_frames_per_epoch": len(train.windows),
            "validation_frames_per_epoch": len(validation.windows),
            "checkpoint_selection": "minimum validation Object21D loss",
            "best_validation_loss": best_loss,
            "training_wall_seconds": perf_counter() - started,
            "run_id": experiment.run_id,
            "dataset_sha256": identifier,
            "validation_review_required": bool(
                report["near_duplicate_tcp_trajectories"]
            ),
        }
        if early:
            summary.update(
                epoch_limit=epochs,
                early_stopped=stopped_early,
                early_stopping_state=early_state,
            )
        report_path = output / "training.json"
        report_path.write_text(json.dumps(summary, indent=2) + "\n")
        experiment.link_output(report_path, kind="metrics")
        experiment.log(summary, prefix="summary")
        print(output)
        return output


def _checkpoint_config(config):
    """Extend the shared signature with precision without altering the fit objective."""
    snapshot = deepcopy(config)
    snapshot["task"]["checkpoint_amp"] = snapshot["training"]["amp"]
    snapshot["task"]["checkpoint_selection"] = "state21"
    return snapshot


def run(config):
    """Dispatch train, evaluate, stress or audit via the standard root entrypoint."""
    config = effective_config(config)
    phase = config.get("phase", "train")
    if phase == "train":
        models = config.get("models", [config["model"]["name"]])
        if (
            not isinstance(models, list)
            or not models
            or any(not isinstance(name, str) or name not in MODELS for name in models)
            or len(set(models)) != len(models)
        ):
            raise ValueError("models must list distinct Object21D model names")
        seeds = config.get("seeds")
        prepared = None
        if len(models) > 1:
            torch.set_num_threads(config["training"]["cpu_threads"])
            if config["training"].get("progress_every_batches"):
                print("Preparing shared train and validation graphs", flush=True)
            prepared = _prepare_training_data(config)
        for name in models:
            model_config = {**config, "model": {**config["model"], "name": name}}
            if seeds is None:
                fit(model_config, prepared=prepared)
            else:
                for seed in seeds:
                    fit(
                        {**model_config, "seed": seed, "seed_override": seed},
                        prepared=prepared,
                    )
    elif phase in ("evaluate", "stress"):
        from evaluation.state21 import evaluate

        evaluate(config, stress=phase == "stress")
    elif phase == "audit":
        _, report = load_dataset(config["data"])
        print(json.dumps(report, indent=2))
    else:
        raise ValueError(f"Unknown Object21D phase: {phase}")
