"""Train matched Five-mode trackers with and without the rigid shape objective."""

from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from data_collection.episode_dataset import canonical_hash
from data_collection.five_mode_persistent_corners import load_split
from data_collection.storage import write_json
from evaluation.gnn.five_mode_routed import load_model
from evaluation.gnn.persistent_corner_tracking import batch_rows, compare, validation_loss
from graph.temporal.persistent_corner_tracking import (
    PersistentCornerTracking, predict_object21, shape_errors, shape_means,
)
from training.gnn.loss import object21_loss
from training.gnn.trainer import atomic_torch_save
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def objective(gnn, tracker, batch, settings):
    """Combine downstream pose, absolute motion, and object-wise distance errors."""
    positions = tracker(batch)
    output = predict_object21(gnn, batch, positions)
    downstream = object21_loss(output, batch["target"])["total"]
    scale = settings["loss_scale_m"]
    hidden = (batch["kinds"] >= 0) & ~batch["observed"]
    labels = batch["motion_targets"]
    distance = torch.linalg.vector_norm(positions - labels, dim=-1) / scale
    pose = F.smooth_l1_loss(distance, torch.zeros_like(distance), reduction="none")
    motion = (pose * hidden).sum() / hidden.sum().clamp_min(1)
    errors, pairs = shape_errors(positions, labels, batch["kinds"], batch["observed"])
    residual = errors / scale
    distortion = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
    frame_shape, counts = shape_means(distortion, pairs)
    shape = frame_shape.sum() / (counts > 0).sum().clamp_min(1)
    loss = downstream + settings["motion_weight"] * motion + settings["shape_weight"] * shape
    terms = torch.stack((loss, downstream, motion, shape)).detach()
    return loss, terms


def train_epoch(gnn, tracker, dataset, optimizer, generator, device, settings):
    """Update the tracker only, using the same frame shuffle in both runs."""
    tracker.train()
    order = torch.randperm(len(dataset["condition"]), generator=generator)
    totals = np.zeros(4)
    for rows in order.split(settings["batch_size"]):
        batch = batch_rows(dataset, rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = objective(gnn, tracker, batch, settings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tracker.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        totals += terms.cpu().numpy() * len(rows)
    names = ("loss", "object21", "motion", "shape")
    return dict(zip(names, (totals / len(order)).tolist(), strict=True))


def fit(config, gnn, train, validation, frame_ids, device, shape_weight):
    """Fit one configuration and retain its best validation Object21 checkpoint."""
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    settings = {**config["training"], "shape_weight": shape_weight}
    effective = {**config, "training": settings}
    tracker = PersistentCornerTracking(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(tracker.parameters(), lr=settings["learning_rate"], weight_decay=0.0001)
    with ExperimentLogger(team="graph", model="persistent_corner_tracking", config=effective,
                          job_type="train") as logger:
        root = Path(config["output_root"]) / logger.run_id
        root.mkdir(parents=True, exist_ok=False)
        logger.link_output(config["frozen_gnn"]["path"], checkpoint=True)
        logger.record(device=str(device), torch_version=str(torch.__version__),
                      trainable_parameters=sum(parameter.numel() for parameter in tracker.parameters()),
                      dataset_sha256=config["data"]["manifest"]["sha256"],
                      train_frame_ids_sha256=canonical_hash(frame_ids[0]),
                      validation_frame_ids_sha256=canonical_hash(frame_ids[1]))
        print(f"Train run {logger.run_id}, shape weight {shape_weight}", flush=True)
        history = []
        best_score, best_epoch = float("inf"), 0
        best_path, last_path, history_path = (root / name for name in ("best.pt", "last.pt", "history.json"))
        for epoch in range(1, settings["epochs"] + 1):
            start = monotonic()
            scalars = train_epoch(gnn, tracker, train, optimizer, generator, device, settings)
            val = validation_loss(gnn, tracker, validation, device, config["evaluation"]["batch_size"])
            score = float(np.mean(list(val.values())))
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation Object21 loss")
            row = dict(epoch=epoch, train=scalars, validation={**val, "object21_macro": score},
                       elapsed_s=monotonic() - start)
            history.append(row)
            logger.log(row, step=epoch)
            write_json(history_path, history)
            snapshot = dict(epoch=epoch, config=effective, validation_loss=score,
                            model=tracker.state_dict(), optimizer=optimizer.state_dict())
            atomic_torch_save(last_path, snapshot)
            if score < best_score:
                best_epoch, best_score = epoch, score
                atomic_torch_save(best_path, snapshot)
            print(f"Epoch {epoch}: train {scalars['loss']:.4f}, val {score:.4f}, "
                  f"best {best_epoch}, {row['elapsed_s']:.0f}s", flush=True)
            if epoch - best_epoch >= settings["patience"]:
                break
        for path in (best_path, last_path, history_path):
            logger.link_output(path, kind="metrics" if path == history_path else "checkpoint")
        logger.record(best_epoch=best_epoch, best_validation_loss=best_score)
    snapshot = torch.load(best_path, map_location=device, weights_only=True)
    tracker.load_state_dict(snapshot["model"], strict=True)
    return tracker, best_path


def train_and_compare(config):
    """Run the shape-loss ablation without updating the frozen position GNN."""
    device = set_device(config["device"])
    torch.set_num_threads(config["cpu_threads"])
    gnn = load_model(config["frozen_gnn"], device).requires_grad_(False)
    frozen = {name: value.clone() for name, value in gnn.state_dict().items()}
    train, train_ids = load_split(config, "train")
    validation, validation_ids = load_split(config, "validation")
    fitted = {}
    for name, weight in config["shape_weights"].items():
        fitted[name] = fit(config, gnn, train, validation, (train_ids, validation_ids), device, weight)
    if not all(torch.equal(value, frozen[name]) for name, value in gnn.state_dict().items()):
        raise RuntimeError("The Object21 GNN changed during tracking training")
    compare(config, gnn, fitted, validation, validation_ids, device)
