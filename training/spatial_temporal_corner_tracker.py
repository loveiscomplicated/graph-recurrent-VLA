"""Train the B2-3 Delaunay-GNN to causal-Transformer corner tracker."""

from itertools import combinations
import copy
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from graph.temporal.spatial_temporal_corner_tracker import (
    CORNER_OBJECT_IDS,
    SpatialTemporalCornerTracker,
)
from data_collection.episode_dataset import file_sha256
from training.gnn.spatial_temporal_corner_data import build_corner_tracking_loaders
from utils.experiment_logger import ExperimentLogger, write


def association_objective(output, batch, temperature):
    """Contrast forecast embeddings with current certified detections by object."""
    if temperature <= 0:
        raise ValueError("association_temperature must be positive")
    predicted = output["track_embedding"]
    detected = output["detection_embedding"]
    width = predicted.shape[-1]
    rows = batch.rows.clamp_min(0)
    detected_slots = torch.gather(
        detected, 2, rows[..., None].expand(*rows.shape, width)
    )
    eligible = batch.visible & batch.seen_before & batch.frame_mask[..., None]
    object_ids = torch.as_tensor(CORNER_OBJECT_IDS, device=predicted.device)
    same_object = object_ids[:, None] == object_ids[None, :]
    allowed_keys = eligible[..., None, :] & same_object
    logits = predicted @ detected_slots.transpose(-1, -2) / temperature
    # A finite sentinel keeps rows without eligible keys finite. Such queries
    # are ignored by cross entropy and never contribute to loss or gradients.
    logits = logits.masked_fill(~allowed_keys, torch.finfo(logits.dtype).min)
    slots = rows.shape[-1]
    labels = torch.arange(slots, device=predicted.device).expand_as(rows)
    labels = labels.masked_fill(~eligible, -100)
    total = F.cross_entropy(
        logits.reshape(-1, slots), labels.reshape(-1), reduction="sum"
    )
    count = int(eligible.sum())
    correct = int(((logits.argmax(dim=-1) == labels) & eligible).sum())
    return total / max(count, 1), correct, count


def rigidity_objective(prediction, target, supervised, visible, frame_mask, scale_m):
    """Preserve within-object corner distances whenever at least one is hidden."""
    if scale_m <= 0:
        raise ValueError("rigidity_scale_m must be positive")
    terms = []
    for first, last in ((0, 8), (8, 24)):
        pairs = list(combinations(range(first, last), 2))
        left = prediction[..., [a for a, _ in pairs], :]
        right = prediction[..., [b for _, b in pairs], :]
        target_left = target[..., [a for a, _ in pairs], :]
        target_right = target[..., [b for _, b in pairs], :]
        valid = (
            supervised[..., [a for a, _ in pairs]]
            & supervised[..., [b for _, b in pairs]]
            & (~visible[..., [a for a, _ in pairs]] | ~visible[..., [b for _, b in pairs]])
            & frame_mask[..., None]
        )
        difference = (
            torch.linalg.vector_norm(left - right, dim=-1)
            - torch.linalg.vector_norm(target_left - target_right, dim=-1)
        ) / scale_m
        if valid.any():
            terms.append(F.smooth_l1_loss(difference[valid], torch.zeros_like(difference[valid])))
    return sum(terms) / len(terms) if terms else prediction.new_zeros(())


def tracker_objective(model, output, batch, settings):
    """Combine forecast, association, shape and calibrated-uncertainty losses."""
    supervised = batch.seen_before & batch.frame_mask[..., None]
    if not supervised.any():
        raise ValueError("A tracker batch has no post-initialization targets")
    visible = batch.visible & supervised
    missing = ~batch.visible & supervised
    normalized_error = (output["position"] - batch.targets) / model.position_scale_m
    point_loss = F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        reduction="none",
    ).mean(dim=-1)
    weights = torch.where(
        batch.visible,
        point_loss.new_tensor(settings.get("visible_weight", 1.0)),
        point_loss.new_tensor(settings.get("missing_weight", 2.0)),
    )
    weights = torch.where(
        batch.reappearance,
        weights * settings.get("reappearance_weight", 2.0),
        weights,
    )
    position = (point_loss * weights)[supervised].sum() / weights[supervised].sum()

    squared = normalized_error.square().mean(dim=-1)
    log_variance = output["log_variance"]
    uncertainty = 0.5 * (squared * torch.exp(-log_variance) + log_variance)
    uncertainty = uncertainty[supervised].mean()
    association, correct, association_count = association_objective(
        output, batch, settings.get("association_temperature", 0.1)
    )
    rigidity = rigidity_objective(
        output["position"],
        batch.targets,
        supervised,
        batch.visible,
        batch.frame_mask,
        settings.get("rigidity_scale_m", 0.01),
    )
    total = (
        position
        + settings.get("association_weight", 0.25) * association
        + settings.get("rigidity_weight", 0.25) * rigidity
        + settings.get("uncertainty_weight", 0.05) * uncertainty
    )
    return {
        "total": total,
        "position": position,
        "association": association,
        "rigidity": rigidity,
        "uncertainty": uncertainty,
        "association_correct": correct,
        "association_count": association_count,
        "supervised": supervised,
        "visible": visible,
        "missing": missing,
    }


def run_epoch(model, loader, settings, device, optimizer=None):
    """Run full chronological episodes and aggregate target-weighted metrics."""
    training = optimizer is not None
    model.train(training)
    totals = {
        "squared": 0.0,
        "hold_squared": 0.0,
        "missing_hold_squared": 0.0,
        "visible_squared": 0.0,
        "missing_squared": 0.0,
        "reappearance_squared": 0.0,
        "targets": 0,
        "visible_targets": 0,
        "missing_targets": 0,
        "reappearance_targets": 0,
        "association_correct": 0,
        "association_count": 0,
        "loss": 0.0,
        "batches": 0,
        "episodes": 0,
        "frames": 0,
        "position_loss": 0.0,
        "association_loss": 0.0,
        "rigidity_loss": 0.0,
        "uncertainty_loss": 0.0,
    }
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress_interval = settings.get("log_every_batches", 25)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            output = model(
                batch.graph,
                batch.rows,
                batch.frame_mask,
                batch.last_observed,
                batch.age,
            )
            losses = tracker_objective(model, output, batch, settings)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite spatial-temporal corner loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    settings.get("gradient_clip", 1.0),
                    error_if_nonfinite=True,
                )
                optimizer.step()

            error = (output["position"].detach() - batch.targets).square().sum(dim=-1)
            hold_error = (batch.last_observed - batch.targets).square().sum(dim=-1)
            supervised = losses["supervised"]
            visible = losses["visible"]
            missing = losses["missing"]
            reappearance = batch.reappearance & supervised
            totals["squared"] += float(error[supervised].sum())
            totals["hold_squared"] += float(hold_error[supervised].sum())
            totals["missing_hold_squared"] += float(hold_error[missing].sum())
            totals["visible_squared"] += float(error[visible].sum())
            totals["missing_squared"] += float(error[missing].sum())
            totals["reappearance_squared"] += float(error[reappearance].sum())
            totals["targets"] += int(supervised.sum())
            totals["visible_targets"] += int(visible.sum())
            totals["missing_targets"] += int(missing.sum())
            totals["reappearance_targets"] += int(reappearance.sum())
            totals["association_correct"] += losses["association_correct"]
            totals["association_count"] += losses["association_count"]
            totals["loss"] += float(losses["total"].detach())
            totals["batches"] += 1
            totals["episodes"] += len(batch.episode_ids)
            totals["frames"] += int(batch.frame_mask.sum())
            for name in ("position", "association", "rigidity", "uncertainty"):
                totals[f"{name}_loss"] += float(losses[name].detach())
            if progress_interval and (
                totals["batches"] == 1 or totals["batches"] % progress_interval == 0
            ):
                elapsed = time.monotonic() - started
                phase = "train" if training else "validation"
                print(
                    f"{phase} batches={totals['batches']}/{len(loader)} "
                    f"episodes={totals['episodes']} frames={totals['frames']} "
                    f"loss={totals['loss'] / totals['batches']:.6f} "
                    f"elapsed={elapsed:.1f}s episodes_per_second={totals['episodes'] / elapsed:.3f}",
                    flush=True,
                )

    def rmse(name, count_name):
        count = totals[count_name]
        return 1000 * (totals[name] / count) ** 0.5 if count else None

    if not totals["targets"]:
        raise ValueError("No supervised corner forecasts were evaluated")
    elapsed = time.monotonic() - started
    metrics = {
        "loss": totals["loss"] / totals["batches"],
        **{
            f"{name}_loss": totals[f"{name}_loss"] / totals["batches"]
            for name in ("position", "association", "rigidity", "uncertainty")
        },
        "elapsed_seconds": elapsed,
        "episodes_per_second": totals["episodes"] / elapsed,
        "frames_per_second": totals["frames"] / elapsed,
        "episodes": totals["episodes"],
        "frames": totals["frames"],
        "forecast_rmse_mm": rmse("squared", "targets"),
        "hold_rmse_mm": rmse("hold_squared", "targets"),
        "missing_hold_rmse_mm": rmse("missing_hold_squared", "missing_targets"),
        "visible_rmse_mm": rmse("visible_squared", "visible_targets"),
        "missing_rmse_mm": rmse("missing_squared", "missing_targets"),
        "reappearance_rmse_mm": rmse("reappearance_squared", "reappearance_targets"),
        "association_accuracy": (
            totals["association_correct"] / totals["association_count"]
            if totals["association_count"]
            else None
        ),
        "targets": totals["targets"],
        "missing_targets": totals["missing_targets"],
        "reappearance_targets": totals["reappearance_targets"],
    }
    if device.type == "cuda":
        metrics["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return metrics


def train(config):
    """Train B2-3 stage A and select by validation missing-corner RMSE."""
    config = copy.deepcopy(config)
    config["model"].setdefault("coordinate_frame", "world")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    settings = config["training"]
    torch.set_num_threads(settings.get("cpu_threads", 4))
    device = torch.device(config["device"])
    generator = torch.Generator().manual_seed(seed)
    loaders, identity = build_corner_tracking_loaders(config, generator)
    model = SpatialTemporalCornerTracker(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    patience = settings.get("patience", settings["epochs"])

    with ExperimentLogger(
        team="graph",
        model="spatial_temporal_corner_tracker",
        seed=seed,
        config=config,
        job_type="training",
    ) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write(output / "effective_config.yaml", config)
        logger.record(**identity, device=str(device), tracker_stage="A")
        history = []
        best = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(1, settings["epochs"] + 1):
            train_metrics = run_epoch(model, loaders["train"], settings, device, optimizer)
            validation = run_epoch(model, loaders["validation"], settings, device)
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation})
            logger.log({"train": train_metrics, "validation": validation}, step=epoch)
            score = validation["missing_rmse_mm"]
            if score is None:
                raise ValueError("Validation needs missing-corner targets for checkpoint selection")
            state = {
                "format": "spatial_temporal_corner_tracker_v1",
                "epoch": epoch,
                "config": config,
                "identity": identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "run_id": logger.run_id,
            }
            if score < best:
                best = score
                best_epoch = epoch
                stale = 0
                torch.save(state, output / "best.pt")
            else:
                stale += 1
            torch.save(state, output / "last.pt")
            write(output / "history.json", history)
            print(
                f"epoch={epoch} train_missing={train_metrics['missing_rmse_mm']:.3f}mm "
                f"validation_missing={score:.3f}mm association={validation['association_accuracy']} "
                f"best_epoch={best_epoch} stale={stale}/{patience}",
                flush=True,
            )
            if stale >= patience:
                break
        summary = {
            "epochs": epoch,
            "best_epoch": best_epoch,
            "best_missing_rmse_mm": best,
            "validation": history[best_epoch - 1]["validation"],
            "checkpoint_sha256": file_sha256(output / "best.pt"),
            "last_checkpoint_sha256": file_sha256(output / "last.pt"),
        }
        write(output / "completed.json", summary)
        for name in (
            "effective_config.yaml",
            "best.pt",
            "last.pt",
            "history.json",
            "completed.json",
        ):
            logger.link_output(output / name, kind="checkpoint" if name.endswith(".pt") else None)
        logger.record(**summary)
        return str(output)
