"""Train recent-corner recovery and selection through a frozen Object21 GNN."""

from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.five_mode_recent_corners import load_cohort
from data_collection.storage import write_json
from evaluation.gnn.five_mode_routed import load_model
from evaluation.gnn.recent_corner_completion import compare, evaluate_checkpoint, per_frame_loss, select_rows
from graph.temporal.recent_corner_completion import RecentCornerCompletion, run_actions
from training.gnn.trainer import atomic_torch_save
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def objective(gnn, completion, batch, settings):
    """Use one Object21 loss per option and differentiate its expected value."""
    recovered, logits, valid = completion(batch)
    rows, choices = valid.nonzero(as_tuple=True)
    options = completion.actions[choices]
    output = run_actions(gnn, batch, recovered, options, rows)
    losses = per_frame_loss(output, batch["target"][rows])
    points = (options >= 0).sum(dim=-1)
    probability = logits.softmax(dim=-1)[rows, choices]
    exploration = settings["uniform_exploration"]
    weight = (1 - exploration) * probability + exploration / valid.sum(dim=-1)[rows]
    object21 = (weight * losses).sum() / len(logits)
    point_cost = (probability * points).sum() / len(logits) * settings["point_cost"]
    mask = batch["candidate_kinds"] >= 0
    scale = settings["motion_loss_scale_m"]
    motion = F.smooth_l1_loss(recovered[mask] / scale, batch["motion_targets"][mask] / scale)
    total = object21 + settings["motion_loss_weight"] * motion + point_cost
    return total, torch.stack((object21, motion, point_cost)).detach()


def train_epoch(gnn, completion, dataset, optimizer, generator, device, settings):
    """Update only the recovery and subset model, with one shuffle per epoch."""
    completion.train()
    indices = torch.randperm(len(dataset["target"]), generator=generator)
    chunks = indices.split(settings["batch_size"])
    totals = np.zeros(4)
    started = monotonic()
    for step, rows in enumerate(chunks, start=1):
        batch = select_rows(dataset, rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = objective(gnn, completion, batch, settings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(completion.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        totals += torch.cat((loss.detach().reshape(1), parts)).cpu().numpy() * len(rows)
        if step % 60 == 0:
            print(f"  batches {step}/{len(chunks)} at {monotonic() - started:.0f}s", flush=True)
    return dict(zip(("loss", "object21_loss", "motion_loss", "point_cost"),
                    (totals / len(indices)).tolist(), strict=True))


def train(config):
    """Train on the Condition 2 focused cohort and run the matched validation."""
    device = set_device(config["device"])
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    settings = config["training"]
    gnn = load_model(config["frozen_gnn"], device).requires_grad_(False)
    frozen_state = {name: value.clone() for name, value in gnn.state_dict().items()}
    completion = RecentCornerCompletion(config["association"]["max_candidates"], **config["model"]).to(device)
    optimizer = torch.optim.AdamW(completion.parameters(), lr=settings["learning_rate"], weight_decay=0.0001)

    with ExperimentLogger(team="graph", model="recent_corner_completion", config=config,
                          job_type="train") as logger:
        root = Path(config["output_root"]) / logger.run_id
        root.mkdir(parents=True, exist_ok=False)
        logger.link_output(config["frozen_gnn"]["path"], checkpoint=True)
        logger.record(dataset_sha256=config["data"]["manifest"]["sha256"], device=str(device),
                      trainable_parameters=sum(parameter.numel() for parameter in completion.parameters()),
                      torch_version=str(torch.__version__), trainer_sha256=file_sha256(Path(__file__)))
        print(f"Training run {logger.run_id}", flush=True)
        dataset, train_ids = load_cohort(config, "train")
        validation, validation_ids = load_cohort(config, "validation")
        selection = (validation["condition"] == 2).nonzero(as_tuple=True)[0]
        logger.record(train_frames_sha256=canonical_hash(train_ids),
                      validation_frames_sha256=canonical_hash(validation_ids),
                      training_frames_by_condition=torch.bincount(dataset["condition"], minlength=4)[1:].tolist())

        best_loss, best_epoch, history = float("inf"), 0, []
        best_path, last_path, history_path = (root / name for name in ("best.pt", "last.pt", "history.json"))
        for epoch in range(1, settings["epochs"] + 1):
            start = monotonic()
            scalars = train_epoch(gnn, completion, dataset, optimizer, generator, device, settings)
            trace = compare(gnn, completion, validation, selection, device,
                            config["evaluation"]["batch_size"], learned_only=True)["learned"]
            validation_loss, peg, hole, endpoint, selected = trace.mean(axis=0, dtype=np.float64).tolist()
            val = dict(object21_loss=validation_loss, peg_mm=peg, hole_mm=hole,
                       endpoint_mm=endpoint, average_points=selected)
            row = dict(epoch=epoch, train=scalars, validation_condition_2=val, elapsed_s=monotonic() - start)
            history.append(row)
            logger.log(row, step=epoch)
            write_json(history_path, history)
            snapshot = dict(epoch=epoch, config=config, model=completion.state_dict(),
                            optimizer=optimizer.state_dict(), validation_loss=validation_loss)
            atomic_torch_save(last_path, snapshot)
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                atomic_torch_save(best_path, snapshot)
            print(f"Epoch {epoch}: train {scalars['object21_loss']:.4f}, condition_2 val {validation_loss:.4f}, "
                  f"endpoint {endpoint:.2f} mm, mean points {selected:.2f}, best {best_epoch}, "
                  f"{row['elapsed_s']:.0f}s", flush=True)
            if epoch - best_epoch >= settings["patience"]:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

        if any(parameter.grad is not None for parameter in gnn.parameters()) or any(
            not torch.equal(value, frozen_state[name]) for name, value in gnn.state_dict().items()
        ):
            raise RuntimeError("Frozen GNN changed during corner completion training")
        for path in (best_path, last_path, history_path):
            logger.link_output(path, kind="metrics" if path == history_path else "checkpoint")
        logger.record(best_epoch=best_epoch, best_validation_loss=best_loss, frozen_gnn_unchanged=True)

    best = torch.load(best_path, map_location=device, weights_only=True)
    completion.load_state_dict(best["model"], strict=True)
    evaluate_checkpoint(config, gnn, completion, validation, validation_ids, device, best_path)
