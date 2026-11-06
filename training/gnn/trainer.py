"""Task-independent GNN optimization, validation and experiment orchestration."""

from copy import deepcopy
from dataclasses import asdict
import importlib
import math
from pathlib import Path
import random

import numpy as np

import torch
from torch import nn

from graph.basic_gnn import BasicGNN
from graph.gated_mpnn import VertexEdgeGatedMPNN
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger
from data_collection.storage import write_json
from graph.build_graph import ConfidenceSettings, GraphSettings
from .checkpoint import load_checkpoint, save_checkpoint
from .tasks import StepResult
from .progress import TrainingProgress


def validate_integer_settings(settings: dict, minimums: dict[str, int]) -> None:
    """Check constructor dimensions before constructing a model."""
    for name, minimum in minimums.items():
        value = settings.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"model.{name} must be an integer >= {minimum}")


def validate_positive_scale(name: str, value: object) -> None:
    """Reject nonnumeric, nonfinite and nonpositive physical scales."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def resolve_config(config: dict) -> dict:
    """Validate the selected model and record all defaults in a config copy."""
    effective = deepcopy(config)
    effective.setdefault("seed", 0)
    effective.setdefault("device", "auto")
    if type(effective["seed"]) is not int or not 0 <= effective["seed"] < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")

    model_config = effective.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("model must be a YAML mapping of model settings")

    if "factory" in model_config:
        import_callable(model_config["factory"])
        model_config.setdefault("name", model_config["factory"])
        return resolve_training_settings(effective)

    model_name = model_config.setdefault("name", "basic_gnn")
    if model_name == "basic_gnn":
        minimums = dict(n_layers=1, in_dim=1, h_dim=1, out_dim=1, edge_dim=1)
        defaults = dict(dropout=0.1, use_confidence=False)
    elif model_name == "vertex_edge_gated_mpnn":
        minimums = dict(num_semantic_ids=1, hidden=8, layers=1, edge_embedding_dim=1)
        defaults = dict(
            hidden=128,
            layers=3,
            edge_embedding_dim=16,
            dropout=0.0,
            use_semantic_ids=False,
            use_confidence=False,
            coord_scale_m=0.05,
            size_scales_m=[0.04, 0.01, 0.01, 0.02],
        )
    else:
        raise ValueError(
            "model.name must be 'basic_gnn' or 'vertex_edge_gated_mpnn'; "
            f"got {model_name!r}"
        )

    unknown = set(model_config) - set(minimums) - set(defaults) - {"name"}
    if unknown:
        raise ValueError(f"Unknown {model_name} model settings: {sorted(unknown)}")
    for name, value in defaults.items():
        model_config.setdefault(name, value)
    validate_integer_settings(model_config, minimums)

    dropout = model_config["dropout"]
    if (type(dropout) not in (int, float) or not math.isfinite(dropout)
            or not 0 <= dropout < 1):
        raise ValueError("model.dropout must be a finite number in [0, 1)")

    for name in ("use_semantic_ids", "use_confidence"):
        if name in model_config and type(model_config[name]) is not bool:
            raise ValueError(f"model.{name} must be bool")

    if model_name == "vertex_edge_gated_mpnn":
        validate_positive_scale("model.coord_scale_m", model_config["coord_scale_m"])
        scales = model_config["size_scales_m"]
        if not isinstance(scales, (list, tuple)) or len(scales) != 4:
            raise ValueError("model.size_scales_m must contain exactly four positive scales")
        for index, scale in enumerate(scales):
            validate_positive_scale(f"model.size_scales_m[{index}]", scale)

    return resolve_training_settings(effective)


def build_model(model_config: dict, device: torch.device) -> nn.Module:
    """Construct a model from resolved settings without changing their contents."""
    constructor_settings = dict(model_config)
    model_name = constructor_settings.pop("name", "basic_gnn")
    if "factory" in constructor_settings:
        factory = import_callable(constructor_settings.pop("factory"))
        model = factory(**constructor_settings)
    elif model_name == "basic_gnn":
        model = BasicGNN(**constructor_settings)
    elif model_name == "vertex_edge_gated_mpnn":
        model = VertexEdgeGatedMPNN(**constructor_settings)
    else:
        raise ValueError(f"Unsupported model.name: {model_name!r}")
    return model.to(device)


def import_callable(path):
    """Resolve an explicit plugin path, e.g. training.gnn.tasks:Object21Task."""
    if not isinstance(path, str) or path.count(":") != 1:
        raise ValueError("Factory must have the form 'importable.module:callable'")
    module_name, callable_name = path.split(":")
    factory = getattr(importlib.import_module(module_name), callable_name, None)
    if not callable(factory):
        raise ValueError(f"Factory is not callable: {path}")
    return factory


def positive_integer(name, value, minimum=1):
    """Validate counts without accepting bool as an integer."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def resolve_training_settings(config):
    """Record defaults and reject malformed workflow inputs before reading data."""
    config.setdefault("mode", "train")
    if config["mode"] not in ("train", "initialize"):
        raise ValueError("mode must be train or initialize")
    sections = {
        "training": {"epochs": 30, "batch_size": 32, "gradient_clip_norm": None,
                     "log_interval_seconds": 10.0},
        "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0001},
        "scheduler": {"name": "none"},
        "task": {"factory": "training.gnn.tasks:Object21Task", "settings": {}},
        "data": {"factory": "training.gnn.data:build_data", "manifest": "manifest.json",
                 "camera": "base_camera", "use_cache": True, "cache_dir": "outputs/gnn_cache",
                 "validation_fraction": 0.2, "test_fraction": 0.1, "split_file": None,
                 "frame_stride": 1, "max_frames_per_episode": None, "num_workers": 0,
                 "empty_inputs": "keep"},
        "detector": {"device": "cpu", "parameters": {}},
        "graph": asdict(GraphSettings()),
        "confidence": asdict(ConfidenceSettings()),
        "output": {"root": "outputs/gnn_training"},
    }
    for name, defaults in sections.items():
        settings = config.setdefault(name, {})
        if not isinstance(settings, dict):
            raise ValueError(f"{name} must be a YAML mapping")
        for key, default in defaults.items():
            settings.setdefault(key, deepcopy(default))
    GraphSettings(**config["graph"])
    ConfidenceSettings(**config["confidence"])
    training = config["training"]
    validate_positive_scale("training.log_interval_seconds", training["log_interval_seconds"])
    for name in ("epochs", "batch_size"):
        positive_integer(f"training.{name}", training[name])
    if training["gradient_clip_norm"] is not None:
        validate_positive_scale("training.gradient_clip_norm", training["gradient_clip_norm"])
    optimizer = config["optimizer"]
    if optimizer["name"] not in ("adam", "adamw", "sgd"):
        raise ValueError("optimizer.name must be adam, adamw or sgd")
    validate_positive_scale("optimizer.lr", optimizer["lr"])
    decay = optimizer["weight_decay"]
    if type(decay) not in (int, float) or not math.isfinite(decay) or decay < 0:
        raise ValueError("optimizer.weight_decay must be finite and nonnegative")
    scheduler = config["scheduler"]
    if scheduler["name"] not in ("none", "step"):
        raise ValueError("scheduler.name must be none or step")
    if scheduler["name"] == "step":
        scheduler.setdefault("step_size", 10)
        scheduler.setdefault("gamma", 0.5)
        positive_integer("scheduler.step_size", scheduler["step_size"])
        validate_positive_scale("scheduler.gamma", scheduler["gamma"])
    data = config["data"]
    positive_integer("data.frame_stride", data["frame_stride"])
    positive_integer("data.num_workers", data["num_workers"], minimum=0)
    if data["max_frames_per_episode"] is not None:
        positive_integer("data.max_frames_per_episode", data["max_frames_per_episode"])
    for name in ("validation_fraction", "test_fraction"):
        value = data[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"data.{name} must be finite in [0,1)")
    if data["validation_fraction"] <= 0 or data["validation_fraction"] + data["test_fraction"] >= 1:
        raise ValueError("Reserve a positive validation fraction and a nonempty training fraction")
    if data["empty_inputs"] not in ("keep", "skip"):
        raise ValueError("data.empty_inputs must be keep or skip")
    for section in ("task", "data"):
        import_callable(config[section]["factory"])
    if not isinstance(config["task"]["settings"], dict):
        raise ValueError("task.settings must be a mapping of task constructor arguments")
    if type(data["use_cache"]) is not bool:
        raise ValueError("data.use_cache must be bool")
    if not isinstance(data["cache_dir"], str) or not data["cache_dir"]:
        raise ValueError("data.cache_dir must be a nonempty path")
    if config["task"]["factory"] == "training.gnn.tasks:Object21Task":
        config["task"]["settings"].setdefault("pos_unit_m", 0.01)
        config["task"]["settings"].setdefault("consistency_weight", 0.05)
        config["task"]["settings"].setdefault("size_units_m", [0.04, 0.01, 0.01, 0.02])
    config.setdefault("resume", None)
    return config


def seed_everything(seed, device):
    """Seed model initialization and user-defined task/data randomness."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def run_epoch(model, loader, task, device, optimizer=None, gradient_clip_norm=None,
              progress=None):
    """Execute one epoch and aggregate each metric by its explicit denominator."""
    is_training = optimizer is not None
    model.train(is_training)
    totals, denominators = {}, {}
    batch_count = 0
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            if batch is None:
                if progress is not None:
                    progress.batch_completed()
                continue
            batch = batch.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            result = task.compute(model, batch)
            if not isinstance(result, StepResult) or result.loss.ndim != 0:
                raise ValueError("Task must return StepResult with a scalar loss Tensor")
            if not torch.isfinite(result.loss):
                raise ValueError("Task returned a nonfinite loss")
            if "loss" not in result.metrics or result.metrics.keys() != result.metric_weights.keys():
                raise ValueError("Task metrics need loss and matching metric_weights")
            for name, value in result.metrics.items():
                weight = result.metric_weights[name]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"Task metric {name} must be a finite Python scalar")
                positive_integer(f"metric_weights.{name}", weight)
                totals[name] = totals.get(name, 0.0) + value * weight
                denominators[name] = denominators.get(name, 0) + weight
            if is_training:
                result.loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm,
                                                   error_if_nonfinite=True)
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    raise ValueError("Nonfinite model gradients; optimizer step cancelled")
                optimizer.step()
            batch_count += 1
            if progress is not None:
                progress.batch_completed()
    if not batch_count:
        raise ValueError("Cannot run an epoch with an empty DataLoader")
    return {name: total / denominators[name] for name, total in totals.items()}


def atomic_torch_save(path, payload):
    """Store a weights-only best snapshot independently from resumable last state."""
    from uuid import uuid4
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(config: dict):
    """Dispatch data/task factories and run the common GNN training lifecycle."""
    effective = resolve_config(config)
    device = set_device(effective["device"])
    seed_everything(effective["seed"], device)
    model_name = effective["model"]["name"]
    job_type = "initialize" if effective["mode"] == "initialize" else "train"
    with ExperimentLogger(team="graph", config=effective, seed=effective["seed"],
                          model=model_name, job_type=job_type) as experiment:
        if effective["mode"] == "initialize":
            model = build_model(effective["model"], device)
            experiment.record(initialization_only=True, device=str(device),
                              parameter_count=sum(p.numel() for p in model.parameters()))
            return experiment.directory
        # UUID-specific directories avoid collisions across fresh and resumed runs.
        output = Path(effective["output"]["root"]).expanduser().resolve() / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        task = import_callable(effective["task"]["factory"])(**effective["task"]["settings"])
        shuffle_generator = torch.Generator().manual_seed(effective["seed"])
        if effective["data"]["factory"] == "training.gnn.data:build_data" and not effective["data"].get("root"):
            raise ValueError("data.root must point to extracted approved episodes")
        bundle = import_callable(effective["data"]["factory"])(effective, shuffle_generator)
        if not isinstance(bundle.identity.get("dataset_sha256"), str):
            raise ValueError("Data factory identity must contain dataset_sha256")
        write_json(output / "effective_config.json", effective)
        write_json(output / "data_identity.json", bundle.identity)
        experiment.record(dataset_sha256=bundle.identity["dataset_sha256"],
                          device=str(device), output_directory=str(output))
        for name in ("effective_config.json", "data_identity.json"):
            experiment.link_output(output / name)
        # Dataset indexing must not change initialization or training RNG.
        seed_everything(effective["seed"], device)
        model = build_model(effective["model"], device)
        optimizer_classes = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        optimizer_settings = dict(effective["optimizer"])
        optimizer_class = optimizer_classes[optimizer_settings.pop("name")]
        optimizer = optimizer_class(model.parameters(), **optimizer_settings)
        scheduler = None
        if effective["scheduler"]["name"] == "step":
            settings = dict(effective["scheduler"])
            settings.pop("name")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **settings)
        start_epoch, best_loss, best_state = 0, math.inf, None
        if effective["resume"]:
            source = Path(effective["resume"]).expanduser().resolve()
            experiment.link_output(source, checkpoint=True)
            payload = load_checkpoint(source, model=model, optimizer=optimizer, scheduler=scheduler,
                                      config=effective, dataset_sha256=bundle.identity["dataset_sha256"],
                                      device=device, shuffle_generator=shuffle_generator)
            start_epoch = payload["epoch"]
            best_loss, best_state = payload["best_loss"], payload["best_model"]
            experiment.record(resume_checkpoint=str(source), resume_epoch=start_epoch)
        experiment.record(parameter_count=sum(p.numel() for p in model.parameters()),
                          initialization_only=False)
        history = []
        progress = TrainingProgress(
            experiment, effective["training"]["epochs"], bundle.train, bundle.validation,
            interval_seconds=effective["training"]["log_interval_seconds"],
        )
        for epoch in range(start_epoch + 1, effective["training"]["epochs"] + 1):
            progress.begin_epoch(epoch)
            learning_rate = optimizer.param_groups[0]["lr"]
            progress.begin_phase("train")
            training_metrics = run_epoch(model, bundle.train, task, device, optimizer,
                                         effective["training"]["gradient_clip_norm"], progress=progress)
            progress.finish_phase()
            progress.begin_phase("validation")
            validation_metrics = run_epoch(model, bundle.validation, task, device, progress=progress)
            progress.finish_phase()
            if validation_metrics["loss"] < best_loss:
                best_loss = validation_metrics["loss"]
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if scheduler is not None:
                scheduler.step()
            row = {"epoch": epoch, "train": training_metrics, "validation": validation_metrics,
                   "learning_rate": learning_rate, "best_validation_loss": best_loss}
            history.append(row)
            experiment.log(training_metrics, step=epoch, prefix="train")
            experiment.log(validation_metrics, step=epoch, prefix="val")
            experiment.log({"learning_rate": learning_rate, "best_validation_loss": best_loss}, step=epoch)
            last_path = output / "last.pt"
            save_checkpoint(last_path, model=model, optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch, best_loss=best_loss, best_model_state=best_state,
                            config=effective, dataset_sha256=bundle.identity["dataset_sha256"],
                            device=device, shuffle_generator=shuffle_generator)
            best_path = output / "best.pt"
            atomic_torch_save(best_path, {"kind": "best_weights", "model": best_state,
                                         "validation_loss": best_loss, "config": effective,
                                         "dataset_sha256": bundle.identity["dataset_sha256"]})
            for path in (last_path, best_path):
                experiment.link_output(path)
            row["seconds"] = progress.finish_epoch()
            write_json(output / "history.json", history)
            experiment.link_output(output / "history.json")
            print(f"Epoch {epoch}/{effective['training']['epochs']}: train={training_metrics['loss']:.6f} "
                  f"val={validation_metrics['loss']:.6f}; seconds={row['seconds']:.1f}; output={output}", flush=True)
        experiment.record(completed_epoch=effective["training"]["epochs"], best_validation_loss=best_loss)
        return output
