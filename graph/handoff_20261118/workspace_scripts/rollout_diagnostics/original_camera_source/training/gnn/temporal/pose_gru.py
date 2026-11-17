"""Train selected Peg/Hole temporal heads over the existing frozen GNN sequence path."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_collection.storage import write_json
from data_collection.temporal.pose_sequences import (
    TemporalPoseDataset,
    collate_pose_episodes,
    fit_observation_normalization,
)
from graph.temporal.pose_gru import TemporalPoseGRU
from training.gnn.temporal.config import resolve_pose_config
from training.gnn.temporal.frozen_source import (
    infer_pose_data,
    prepare_pose_sequences,
)
from training.gnn.temporal.pose_loss import (
    insertion_loss,
    pose_loss,
)
from training.gnn.trainer import seed_everything
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def run_pose_epoch(model, dataset, device, settings, optimizer=None, generator=None):
    """Batch complete episodes, mask right padding and reset hidden for every batch."""
    loader = DataLoader(
        dataset,
        batch_size=settings["training"]["episode_batch_size"],
        shuffle=optimizer is not None,
        generator=generator,
        collate_fn=collate_pose_episodes,
    )
    model.train(optimizer is not None)
    sums, count = {}, 0
    precision = {}
    with torch.set_grad_enabled(optimizer is not None):
        for observations, target, mask in loader:
            observations, target, mask = (
                value.to(device) for value in (observations, target, mask)
            )
            prediction, _ = model(observations, hidden=None, mask=mask)
            objective = insertion_loss if "ablation" in settings["loss"] else pose_loss
            loss, terms = objective(prediction, target, mask, model.mode, settings["loss"])
            if "ablation" in settings["loss"]:
                from evaluation.gnn.temporal.precision import validation_precision

                arrays = validation_precision(prediction, target, mask, settings["evaluation"])
                for name, values in arrays.items():
                    precision.setdefault(name, []).append(values)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    settings["training"]["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
            frames = int(mask.sum())
            for name, value in dict(loss=loss, **terms).items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * frames
            count += frames
    if count == 0:
        raise ValueError("Pose epoch needs nonempty episodes")
    result = {name: value / count for name, value in sums.items()}
    for name, arrays in precision.items():
        values = np.concatenate(arrays)
        result[name + "_mean"] = float(values.mean()) if len(values) else None
        result[name + "_p90"] = float(np.percentile(values, 90)) if len(values) else None
        result[name + "_frames"] = len(values)
    return result


def train(config):
    """Train a pose mode; fit feature normalization exclusively on temporal train parents."""
    effective = resolve_pose_config(config)
    device = set_device(effective["device"])
    seed_everything(effective["seed"], device)
    frozen, loaders, identity = prepare_pose_sequences(effective, device, ("train", "validation"))
    with ExperimentLogger(
        team="graph", model="temporal_pose_gru", config=effective, seed=effective["seed"]
    ) as experiment:
        output = Path(effective["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        experiment.link_output(effective["checkpoint"], checkpoint=True)
        datasets = {
            name: TemporalPoseDataset(infer_pose_data(frozen, loader, device, identity, effective))
            for name, loader in loaders.items()
        }
        mean, std = fit_observation_normalization(datasets["train"])
        effective["normalization"] = {
            "source_split": "train",
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        effective["loss_definition"] = (
            "pose_relative_geometry_v1"
            if "ablation" in effective["loss"]
            else "normalized_huber_position_plus_sign_invariant_rotation_v1"
        )
        write_json(output / "effective_config.json", effective)
        experiment.record(normalization=effective["normalization"])
        model = TemporalPoseGRU(
            mode=effective["mode"], with_robot=effective["with_robot"], **effective["gru"]
        ).to(device)
        model.set_normalization(mean.to(device), std.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=effective["training"]["lr"])
        generator = torch.Generator().manual_seed(effective["seed"])
        best, history = float("inf"), []
        for epoch in range(1, effective["training"]["epochs"] + 1):
            trained = run_pose_epoch(
                model, datasets["train"], device, effective, optimizer, generator
            )
            validation = run_pose_epoch(model, datasets["validation"], device, effective)
            row = {"epoch": epoch, "train": trained, "validation": validation}
            score = validation["loss"]
            if "ablation" in effective["loss"]:
                from evaluation.gnn.temporal.precision import selection_score

                score, components = selection_score(validation, effective["selection"])
                row["selection"] = {"score": score, "components": components}
            history.append(row)
            experiment.log(row, step=epoch)
            if score < best:
                best = score
                torch.save(
                    {
                        "kind": "temporal_pose_gru",
                        "model": model.state_dict(),
                        "config": effective,
                        "dataset_sha256": identity["dataset_sha256"],
                        "epoch": epoch,
                        "validation_loss": validation["loss"],
                        "selection_score": best,
                    },
                    output / "best.pt",
                )
                experiment.link_output(output / "best.pt")
            print(
                f"Pose GRU {effective['mode']} epoch {epoch}: train_loss={trained['loss']:.5f}, "
                f"validation_loss={validation['loss']:.5f}, selection_score={score:.5f}",
                flush=True,
            )
        write_json(output / "history.json", history)
        for name in ("history.json", "effective_config.json"):
            experiment.link_output(output / name)
        experiment.record(
            parameter_count=sum(value.numel() for value in model.parameters()),
            dataset_sha256=identity["dataset_sha256"],
            device=str(device),
            source_checkpoint_sha256=effective["source_checkpoint_sha256"],
            mode=effective["mode"],
            teacher_forcing=False,
        )
        return output
