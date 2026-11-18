"""Fine-tune the Peg surface GNN with paired validation and staged frame mixing."""

from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.storage import write_json
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger
from .checkpoint import save_checkpoint
from .packed_five_mode_data import BalancedFrameBatches, PackedFrameBatches, read_splits
from .trainer import atomic_torch_save, build_model, import_callable, resolve_config, run_epoch, seed_everything


CONDITIONS = ("condition_1", "condition_2")
ENDPOINT = "endpoint_error_m"


def load_cohort(config, device):
    """Pin the detector index and pack only the train and validation splits."""
    data = config["data"]
    manifest_path = Path(data["manifest"]).resolve()
    index_path = Path(data["condition_index"]).resolve()
    manifest_hash = file_sha256(manifest_path)
    index_hash = file_sha256(index_path)
    if (manifest_hash, index_hash) != (data["manifest_sha256"], data["condition_index_sha256"]):
        raise ValueError("Expected the pinned full Five-mode manifest and detector visibility index")
    manifest = json.loads(manifest_path.read_text())
    index = json.loads(index_path.read_text())
    train, validation = {}, {}
    for condition in CONDITIONS:
        splits = read_splits(manifest_path, manifest, ("train", "validation"), condition,
                             include_surface=True, condition_index=index)
        for split, episodes in splits.items():
            count = sum(len(frames) for _, _, frames in episodes)
            if count != data["frame_counts"][condition][split]:
                raise ValueError(f"Unexpected {condition}/{split} frame count: {count}")
            print(f"Loaded {condition}/{split}: {count:,} frames", flush=True)
        train[condition] = splits["train"]
        validation[condition] = PackedFrameBatches(
            splits["validation"], device, config["training"]["batch_size"], None,
        )
    return train, validation


def evaluate_conditions(model, validation, task, device):
    """Keep condition metrics separate, with frame-weighted errors in metres."""
    return {condition: run_epoch(model, loader, task, device)
            for condition, loader in validation.items()}


def accept_candidate(metrics, incumbent, condition_1_limit):
    """Require both the fixed retention guard and improvement over the incumbent."""
    return (metrics["condition_1"][ENDPOINT] <= condition_1_limit
            and metrics["condition_2"][ENDPOINT] < incumbent["condition_2"][ENDPOINT])


def train_stage(config, ratio, source_path, source_metrics, condition_1_limit,
                model, task, train, validation, device):
    """Run a stage until the guarded Condition 2 improvement plateaus."""
    training = config["training"]
    early = training["early_stopping"]
    ratio_name = f"mix_{ratio[0]}_{ratio[1]}"
    seed_everything(config["seed"], device)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = BalancedFrameBatches(
        train["condition_1"], train["condition_2"], device, training["batch_size"],
        generator, ratio=ratio, frames_per_epoch=config["curriculum"]["frames_per_epoch"],
    )
    effective = deepcopy(config)
    effective["stage"] = dict(ratio=ratio, source_checkpoint=str(source_path),
                              source_checkpoint_sha256=file_sha256(source_path),
                              source_validation=source_metrics, condition_1_limit_m=condition_1_limit)
    data = effective["data"]
    identity = dict(manifest_sha256=data["manifest_sha256"],
                    condition_index_sha256=data["condition_index_sha256"], include_surface=True,
                    ratio=ratio, frames_per_epoch=config["curriculum"]["frames_per_epoch"])
    dataset_hash = canonical_hash(identity)
    model.load_state_dict(torch.load(source_path, map_location="cpu", weights_only=True)["model"])
    optimizer_settings = {k: v for k, v in effective["optimizer"].items() if k != "name"}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_settings)
    scheduler_settings = {k: v for k, v in effective["scheduler"].items() if k != "name"}
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_settings)

    selected_metrics, selected_path = source_metrics, source_path
    selected_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    selected_epoch, stale = 0, 0
    reference = source_metrics["condition_2"][ENDPOINT]
    history = []
    with ExperimentLogger(team="graph", model=effective["model"]["name"], config=effective,
                          seed=effective["seed"], job_type="train") as logger:
        output = Path(effective["output"]["root"]) / ratio_name / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.link_output(source_path, checkpoint=True)
        logger.record(dataset_sha256=dataset_hash, output_directory=str(output), device=str(device))
        write_json(output / "data_identity.json", dict(identity, dataset_sha256=dataset_hash))
        logger.link_output(output / "data_identity.json")
        print(f"Started {ratio_name}: {loader.primary_frames:,}/{loader.secondary_frames:,} frames "
              f"per epoch; source={source_path}; output={output}", flush=True)
        for epoch in range(1, training["epochs"] + 1):
            started = perf_counter()
            train_metrics = run_epoch(model, loader, task, device, optimizer,
                                      training["gradient_clip_norm"])
            metrics = evaluate_conditions(model, validation, task, device)
            condition_1_ok = metrics["condition_1"][ENDPOINT] <= condition_1_limit
            endpoint = metrics["condition_2"][ENDPOINT]
            if condition_1_ok and endpoint <= reference - early["min_delta"]:
                reference, stale = endpoint, 0
            else:
                stale += 1
            if accept_candidate(metrics, selected_metrics, condition_1_limit):
                selected_metrics, selected_path, selected_epoch = metrics, output / "best.pt", epoch
                selected_state = {name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()}
                atomic_torch_save(selected_path, dict(
                    kind="best_weights", model=selected_state, config=effective,
                    dataset_sha256=dataset_hash, validation_loss=metrics["condition_2"]["loss"],
                    validation=metrics, selected_epoch=epoch,
                ))
                logger.link_output(selected_path)
            learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step()
            save_checkpoint(
                output / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_loss=selected_metrics["condition_2"]["loss"],
                best_model_state=selected_state, config=effective, dataset_sha256=dataset_hash,
                device=device, shuffle_generator=generator,
                early_stopping_state=dict(reference_endpoint_m=reference, stale_epochs=stale),
            )
            logger.link_output(output / "last.pt")
            history.append(dict(epoch=epoch, train=train_metrics, validation=metrics,
                                condition_1_within_limit=condition_1_ok, stale_epochs=stale,
                                selected_epoch=selected_epoch, learning_rate=learning_rate,
                                seconds=perf_counter() - started))
            write_json(output / "history.json", history)
            logger.link_output(output / "history.json")
            logger.log(train_metrics, step=epoch, prefix="train")
            for condition, values in metrics.items():
                logger.log(values, step=epoch, prefix=f"val/{condition}")
            logger.log(dict(stale_epochs=stale, condition_1_within_limit=int(condition_1_ok)),
                       step=epoch, prefix="selection")
            print(f"{ratio_name} epoch {epoch}: condition_1={metrics['condition_1'][ENDPOINT]*1000:.3f} mm, "
                  f"condition_2={endpoint*1000:.3f} mm, eligible={condition_1_ok}, "
                  f"stale={stale}/{early['patience']}, selected_epoch={selected_epoch}", flush=True)
            if epoch >= early["min_epochs"] and stale >= early["patience"]:
                break
        else:
            raise RuntimeError(f"{ratio_name} reached the safety cap before early stopping")
        selection = dict(selected_checkpoint=str(selected_path), selected_epoch=selected_epoch,
                         selected_checkpoint_sha256=file_sha256(selected_path),
                         validation=selected_metrics, stopped_epoch=epoch,
                         inherited_source=selected_epoch == 0, run_id=logger.run_id)
        write_json(output / "selection.json", selection)
        logger.link_output(output / "selection.json")
        logger.record(completed_epoch=epoch, early_stopped=True, selection=selection)
        print(f"Early stopped {ratio_name} at epoch {epoch}; selected={selected_path}", flush=True)
    return selected_path, selected_metrics


def train(config):
    """Execute the pinned Peg-surface curriculum from the Condition 1 validation best."""
    effective = resolve_config(config)
    device = set_device(effective["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision(effective["data"]["matmul_precision"])
    curriculum = effective["curriculum"]
    source = Path(effective["initial_weights"]).resolve()
    if file_sha256(source) != curriculum["initial_weights_sha256"]:
        raise ValueError("Expected the pinned Condition 1 Peg-surface best checkpoint")
    best = torch.load(source, map_location="cpu", weights_only=True)
    if best["kind"] != "best_weights" or best["config"]["model"] != effective["model"]:
        raise ValueError("Initial checkpoint must contain matching validation-best model weights")

    seed_everything(effective["seed"], device)
    model = build_model(effective["model"], device)
    model.load_state_dict(best["model"])
    task = import_callable(effective["task"]["factory"])(**effective["task"]["settings"])
    train_rows, validation = load_cohort(effective, device)
    metrics = evaluate_conditions(model, validation, task, device)
    limit = metrics["condition_1"][ENDPOINT] * (1 + curriculum["condition_1_max_degradation"])
    print(f"Baseline condition_1={metrics['condition_1'][ENDPOINT]*1000:.3f} mm, "
          f"condition_2={metrics['condition_2'][ENDPOINT]*1000:.3f} mm, "
          f"condition_1 limit={limit*1000:.3f} mm", flush=True)
    for ratio in curriculum["ratios"]:
        source, metrics = train_stage(effective, tuple(ratio), source, metrics, limit,
                                      model, task, train_rows, validation, device)
    print(f"Curriculum complete; final selected checkpoint: {source}", flush=True)
