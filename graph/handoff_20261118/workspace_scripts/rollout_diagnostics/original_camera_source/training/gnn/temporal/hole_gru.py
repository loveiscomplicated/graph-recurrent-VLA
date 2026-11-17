"""Frozen GNN sequence preparation and full-episode GRU training orchestration."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data_collection.storage import write_json
from data_collection.temporal.frozen_episodes import (
    FrozenEpisodeDataset,
    infer_observations,
)
from graph.temporal.hole_gru import TemporalHoleGRU
from training.gnn.temporal.config import resolve_gru_config
from training.gnn.temporal.frozen_source import prepare_sequences
from training.gnn.trainer import seed_everything
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def run_gru_epoch(model, dataset, device, settings, optimizer=None, generator=None):
    """Full causal BPTT, with every frame supervised and zero hidden per episode.

    Shuffle episodes only. Each episode has one optimizer step; the reported loss
    is a frame-weighted coordinate mean (L1: m, Huber: m²), including initialization
    and absent-vertex frames. Labels are passed only to the loss.
    """
    if len(dataset) == 0:
        raise ValueError("GRU epoch needs at least one episode")
    model.train(optimizer is not None)
    indices = np.arange(len(dataset))
    if optimizer is not None:
        if generator is None:
            raise ValueError("Training requires a seeded episode-shuffle generator")
        indices = generator.permutation(indices)
    total_loss, total_frames = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for index in indices:
            observations, target = dataset[index]
            observations = observations.to(device).unsqueeze(0)
            target = target.to(device)
            prediction, _ = model(observations, hidden=None)
            if settings["loss"] == "huber":
                loss = F.huber_loss(prediction[0], target, delta=settings["huber_delta_m"])
            else:
                loss = F.l1_loss(prediction[0], target)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite GRU loss in episode {dataset.episodes[index]}")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["gradient_clip_norm"], error_if_nonfinite=True
                )
                optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_frames += len(target)
    return total_loss / total_frames


def train(config):
    """Train only GRU/head; select best on held-out GNN-train parents, never test."""
    effective = resolve_gru_config(config)
    device = set_device(effective["device"])
    seed_everything(effective["seed"], device)
    frozen, loaders, identity = prepare_sequences(effective, device, ("train", "validation"))
    with ExperimentLogger(
        team="graph", model="temporal_hole_gru", config=effective, seed=effective["seed"]
    ) as experiment:
        output = Path(effective["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "effective_config.json", effective)
        experiment.link_output(effective["checkpoint"], checkpoint=True)
        datasets = {}
        for split, loader in loaders.items():
            datasets[split] = FrozenEpisodeDataset(infer_observations(frozen, loader, device))
        model = TemporalHoleGRU(**effective["gru"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=effective["training"]["lr"])
        generator = np.random.default_rng(effective["seed"])
        best, history = float("inf"), []
        for epoch in range(1, effective["training"]["epochs"] + 1):
            training_loss = run_gru_epoch(
                model, datasets["train"], device, effective["training"], optimizer, generator
            )
            validation_loss = run_gru_epoch(
                model, datasets["validation"], device, effective["training"]
            )
            row = dict(epoch=epoch, train_loss=training_loss, validation_loss=validation_loss)
            history.append(row)
            experiment.log(row, step=epoch)
            if validation_loss < best:
                best = validation_loss
                torch.save(
                    {
                        "kind": "temporal_hole_gru",
                        "model": model.state_dict(),
                        "config": effective,
                        "dataset_sha256": identity["dataset_sha256"],
                        "epoch": epoch,
                        "validation_loss": best,
                    },
                    output / "best.pt",
                )
                experiment.link_output(output / "best.pt")
            print(
                f"GRU epoch {epoch}: train={training_loss:.8f}, val={validation_loss:.8f}",
                flush=True,
            )
        write_json(output / "history.json", history)
        experiment.record(
            parameter_count=sum(p.numel() for p in model.parameters()),
            frozen_parameter_count=sum(p.numel() for p in frozen.parameters()),
            dataset_sha256=identity["dataset_sha256"],
            device=str(device),
            teacher_forcing=False,
            gnn_validation_used_for_training=False,
            source_checkpoint_sha256=effective["source_checkpoint_sha256"],
            sequence_counts={name: len(dataset) for name, dataset in datasets.items()},
        )
        for name in ("history.json", "effective_config.json"):
            experiment.link_output(output / name)
        return output
