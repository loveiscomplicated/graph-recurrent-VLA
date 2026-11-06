"""Atomic epoch checkpoints and reproducible training continuation."""

from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch


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
    return {name: config[name] for name in ("model", "task", "optimizer", "scheduler", "seed")} | {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["data"]["num_workers"],
        "gradient_clip_norm": config["training"]["gradient_clip_norm"],
    }


def save_checkpoint(path, *, model, optimizer, scheduler, epoch, best_loss,
                    best_model_state, config, dataset_sha256, device, shuffle_generator):
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
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, model, optimizer, scheduler, config, dataset_sha256,
                    device, shuffle_generator):
    """Reject incompatible data/recipes before restoring model or optimizer."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported resume checkpoint; use a GNN training last.pt file")
    if payload["dataset_sha256"] != dataset_sha256:
        raise ValueError("Resume dataset/split/preparation hash differs from checkpoint")
    if payload["resume_signature"] != resume_signature(config):
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
