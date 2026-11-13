"""Atomic epoch checkpoints and reproducible training continuation."""

from copy import deepcopy
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch

from graph.vertex_aggregation import AGGREGATION_DEFAULTS


def canonical_resume_signature(signature):
    """Treat omitted legacy aggregation flags as the unchanged baseline defaults."""
    canonical = deepcopy(signature)
    if canonical["model"].get("name") == "vertex_edge_gated_mpnn":
        for name, value in AGGREGATION_DEFAULTS.items():
            canonical["model"].setdefault(name, value)
    return canonical


def capture_rng(device, shuffle_generator):
    """Store Python, NumPy, Torch and selected accelerator RNG states safely."""
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "shuffle": shuffle_generator.get_state(),
        "device_type": device.type,
    }
    if device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device, shuffle_generator):
    """Restore after model/data construction, before creating the next iterator."""
    if state["device_type"] != device.type:
        raise ValueError("Resume requires the same device type as the source checkpoint")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    shuffle_generator.set_state(state["shuffle"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["accelerator"])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["accelerator"])


def resume_signature(config):
    """Lock the objective and optimization recipe while permitting longer runs."""
    signature = {name: config[name] for name in ("model", "task", "optimizer", "scheduler", "seed")} | {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["data"]["num_workers"],
        "gradient_clip_norm": config["training"]["gradient_clip_norm"],
    }

    if config["training"].get("early_stopping") is not None:
        signature["early_stopping"] = config["training"]["early_stopping"]
    return signature


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, best_loss,
                    best_model_state, config, dataset_sha256, device, shuffle_generator,
                    early_stopping_state=None):
    """Save last and best states together so resume can preserve earlier bests."""
    payload = {
        "format_version": 1,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch, "best_loss": best_loss, "best_model": best_model_state,
        "config": config, "resume_signature": resume_signature(config),
        "dataset_sha256": dataset_sha256,
        "rng": capture_rng(device, shuffle_generator),
    }
    if early_stopping_state is not None:
        payload["early_stopping_state"] = early_stopping_state
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, model, optimizer, scheduler, config, dataset_sha256,
                    device, shuffle_generator, resume_from_dataset_sha256=None):
    """Reject incompatible data/recipes before restoring model or optimizer."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported resume checkpoint; use a GNN training last.pt file")
    if payload["dataset_sha256"] not in (dataset_sha256, resume_from_dataset_sha256):
        raise ValueError("Resume dataset/split/preparation hash differs from checkpoint")
    saved_signature = canonical_resume_signature(payload["resume_signature"])
    requested_signature = canonical_resume_signature(resume_signature(config))
    # Enabling early stopping on a legacy run does not alter its training recipe.
    # Once enabled, its settings and counter must remain unchanged on resume.
    if "early_stopping" not in saved_signature:
        requested_signature.pop("early_stopping", None)
    if saved_signature != requested_signature:
        raise ValueError("Resume model/task/optimizer/scheduler/seed or batch settings differ")
    if payload["epoch"] >= config["training"]["epochs"]:
        raise ValueError("training.epochs must exceed the completed checkpoint epoch")
    if payload["rng"]["device_type"] != device.type:
        raise ValueError("Resume requires the same device type as the source checkpoint")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"], device, shuffle_generator)
    return payload
