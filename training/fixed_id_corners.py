"""Train independent fixed-ID corner sequences with one shared temporal model."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import iter_sequences, read_manifest
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from utils.experiment_logger import ExperimentLogger, write


def load_splits(root, expected_hash, device):
    """Read each split once and retain complete padded histories on the GPU."""
    manifest = read_manifest(root)
    if manifest["dataset_sha256"] != expected_hash:
        raise ValueError("Dataset identity differs from data.expected_dataset_sha256")
    samples = {}
    for split in ("train", "validation"):
        samples[split] = list(iter_sequences(root, split))
        if not samples[split]:
            raise ValueError(f"No eligible corner sequences in {split}")
        print(f"Loaded {split}: {len(samples[split])} corner sequences", flush=True)
    parents = [{sample.parent for sample in samples[split]} for split in ("train", "validation")]
    if parents[0] & parents[1]:
        raise ValueError("Train and validation parents overlap")
    maximum_length = max(len(sample.inputs) for values in samples.values() for sample in values)
    splits = {}
    for split, values in samples.items():
        shape = (len(values), maximum_length)
        arrays = dict(
            inputs=np.zeros((*shape, 13), dtype=np.float32),
            targets=np.zeros((*shape, 3), dtype=np.float32),
            context=np.zeros(shape, dtype=bool),
            supervised=np.zeros(shape, dtype=bool),
            observed=np.zeros(shape, dtype=bool),
        )
        for index, sample in enumerate(values):
            length = len(sample.inputs)
            arrays["inputs"][index, :length] = sample.inputs
            arrays["targets"][index, :length] = sample.target_delta_xyz
            arrays["context"][index, :length] = True
            arrays["supervised"][index, :length] = sample.prediction_valid
            arrays["observed"][index, :length] = sample.observed
        for name in ("inputs", "targets"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"{split} {name} contains nonfinite values")
        if not arrays["supervised"].any(axis=1).all():
            raise ValueError(f"{split} contains a sequence with no prediction targets")
        splits[split] = {name: torch.as_tensor(value, device=device) for name, value in arrays.items()}
        print(f"Cached {split} on {device}: shape={shape} supervised={int(arrays['supervised'].sum())}", flush=True)
    return splits, manifest


def input_statistics(data):
    """Fit input normalization on train context only, including real missing flags."""
    values = data["inputs"][data["context"]].double()
    mean = values.mean(0).float()
    std = values.std(0, correction=0).float().clamp_min(0.001)
    return mean, std


def error_totals(predictions, batch):
    """Accumulate squared 3-D error and counts for all/visible/missing targets."""
    errors = (predictions.float() - batch["targets"]).square().sum(-1)
    valid = batch["supervised"]
    visible = valid & batch["observed"]
    missing = valid & ~batch["observed"]
    totals = []
    for mask in (valid, visible, missing):
        totals.append((errors * mask).sum())
        totals.append(mask.sum().float())
    return torch.stack(totals)


def metrics_from_totals(totals, displacement_scale_m):
    """Report normalized MSE loss and physical 3-D RMSE without negative NLLs."""
    values = totals.detach().cpu().tolist()
    metrics = {}
    for index, name in enumerate(("all", "visible", "missing")):
        squared_error, count = values[2 * index:2 * index + 2]
        metrics[f"{name}_targets"] = int(count)
        metrics[f"{name}_rmse_mm"] = (squared_error / count) ** 0.5 * 1000 if count else None
    metrics["loss"] = values[0] / (3 * values[1] * displacement_scale_m ** 2)
    return metrics


def execute_epoch(model, data, order, config, output, epoch, optimizer=None):
    """Run full histories in batches; synchronize metrics only when logging."""
    training = optimizer is not None
    model.train(training)
    settings = config["training"]
    batch_size = settings["batch_size"]
    batch_count = (len(order) + batch_size - 1) // batch_size
    device = data["inputs"].device
    totals = torch.zeros(6, device=device)
    started = time.monotonic()
    phase = "train" if training else "validation"
    gradient_context = nullcontext if training else torch.no_grad
    with gradient_context():
        for batch_index, left in enumerate(range(0, len(order), batch_size), start=1):
            indices = order[left:left + batch_size]
            batch = {name: value[indices] for name, value in data.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            use_amp = settings["precision"] == "bfloat16"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                predictions = model(batch["inputs"], batch["context"])
                squared = ((predictions - batch["targets"]) / model.displacement_scale_m).square().sum(-1)
                count = batch["supervised"].sum()
                loss = (squared * batch["supervised"]).sum() / (3 * count)
            if training:
                loss.backward()
                # Nonfinite gradients fail immediately; no silent bad checkpoints.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["gradient_clip"], error_if_nonfinite=True,
                )
                optimizer.step()
            totals += error_totals(predictions.detach(), batch)
            if batch_index == 1 or batch_index % settings["log_every_batches"] == 0 or batch_index == batch_count:
                metrics = metrics_from_totals(totals, model.displacement_scale_m)
                if not np.isfinite(metrics["loss"]):
                    raise FloatingPointError(f"Nonfinite {phase} loss at epoch {epoch}, batch {batch_index}")
                progress = dict(
                    epoch=epoch, total_epochs=settings["epochs"], phase=phase,
                    batch=batch_index, batches=batch_count, elapsed_seconds=time.monotonic() - started,
                    metrics=metrics, gpu_allocated_mb=torch.cuda.memory_allocated(device) / 2 ** 20,
                )
                write(output / "progress.json", progress)
                print(
                    f"epoch={epoch}/{settings['epochs']} {phase} batch={batch_index}/{batch_count} "
                    f"loss={metrics['loss']:.6f} targets={metrics['all_targets']} "
                    f"visible={metrics['visible_targets']} missing={metrics['missing_targets']} "
                    f"elapsed={progress['elapsed_seconds']:.1f}s", flush=True,
                )
    return metrics_from_totals(totals, model.displacement_scale_m)


def save_checkpoint(path, model, optimizer, config, dataset_hash, run_id, epoch, history, shuffle):
    """Atomically save model, normalization, optimizer and reproducibility metadata."""
    state = dict(
        format="fixed_id_corner_predictor_v1", model=model.state_dict(),
        optimizer=optimizer.state_dict(), config=config, dataset_sha256=dataset_hash,
        run_id=run_id, epoch=epoch, history=history,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        shuffle_state=shuffle.bit_generator.state,
    )
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_resume_checkpoint(config, dataset_hash):
    """Validate the previous last/best checkpoints before modifying any state."""
    path = Path(config["resume"]["checkpoint"]).expanduser().resolve()
    # Checkpoints are locally produced training artifacts, including optimizer state.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "fixed_id_corner_predictor_v1":
        raise ValueError(f"Unsupported resume checkpoint: {path}")
    if checkpoint["dataset_sha256"] != dataset_hash:
        raise ValueError("Resume checkpoint dataset hash does not match the requested dataset")
    old_config = checkpoint["config"]
    if old_config["seed"] != config["seed"] or old_config["model"] != config["model"]:
        raise ValueError("Resume seed and model configuration must match the checkpoint")
    for name in ("device", "batch_size", "learning_rate", "weight_decay", "gradient_clip", "precision"):
        if old_config["training"][name] != config["training"][name]:
            raise ValueError(f"Resume training.{name} must match the checkpoint")
    epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    if type(epoch) is not int or epoch < 1 or [row["epoch"] for row in history] != list(range(1, epoch + 1)):
        raise ValueError("Resume history must contain all completed epochs in order")
    if config["training"]["epochs"] <= epoch:
        raise ValueError("training.epochs must exceed the checkpoint epoch")
    losses = [row["validation"]["loss"] for row in history]
    if not np.isfinite(losses).all():
        raise ValueError("Resume history contains nonfinite validation losses")
    best_path = Path(config["resume"].get("best_checkpoint") or path.with_name("best.pt")).expanduser().resolve()
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    best_epoch = history[int(np.argmin(losses))]["epoch"]
    if (best.get("format") != checkpoint["format"] or best.get("dataset_sha256") != dataset_hash
            or best.get("epoch") != best_epoch or best.get("history") != history[:best_epoch]
            or best.get("config", {}).get("seed") != old_config["seed"]
            or best.get("config", {}).get("model") != old_config["model"]):
        raise ValueError("Resume best checkpoint must match the source run and best validation epoch")
    return checkpoint, path, best_path


def restore_training_state(checkpoint, model, optimizer, shuffle, train_size):
    """Restore weights, optimizer, RNG and the exact next epoch's batch order."""
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    # Original checkpoints did not store NumPy's generator state. Reconstruct
    # it from the seed and verify each historical permutation before continuing.
    for row in checkpoint["history"]:
        order = shuffle.permutation(train_size)
        digest = hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest()
        if digest != row["batch_order_sha256"]:
            raise ValueError(f"Cannot reproduce batch order for epoch {row['epoch']}")
    if "shuffle_state" in checkpoint and shuffle.bit_generator.state != checkpoint["shuffle_state"]:
        raise ValueError("Stored NumPy shuffle state differs from reconstructed history")
    torch.set_rng_state(checkpoint["torch_rng"])
    if next(model.parameters()).device.type == "cuda":
        if len(checkpoint["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume checkpoint CUDA device count does not match this host")
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def update_early_stopping(loss, epoch, best_loss, best_epoch, patience):
    """Update the global best and count epochs since it, including resumed history."""
    if not np.isfinite(loss):
        raise ValueError("Validation loss must be finite")
    improved = loss < best_loss
    if improved:
        best_loss = loss
        best_epoch = epoch
    stale_epochs = epoch - best_epoch
    stop = patience is not None and stale_epochs >= patience
    return best_loss, best_epoch, stale_epochs, improved, stop


def train(config):
    """Train one architecture/seed until the configured stop, never read test."""
    settings = config["training"]
    device = torch.device(settings["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This training recipe requires CUDA; CPU fallback is disabled")
    if settings["precision"] not in ("float32", "bfloat16"):
        raise ValueError("training.precision must be float32 or bfloat16")
    if settings["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested bfloat16 is not supported by this GPU")
    for name in ("epochs", "batch_size", "log_every_batches", "cpu_threads"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    early_stopping = settings.get("early_stopping")
    patience = None
    if early_stopping is not None:
        if not isinstance(early_stopping, dict):
            raise ValueError("training.early_stopping must be null or a mapping with patience")
        patience = early_stopping.get("patience")
        if type(patience) is not int or patience < 1:
            raise ValueError("training.early_stopping.patience must be a positive integer")
    torch.set_num_threads(settings["cpu_threads"])
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    shuffle = np.random.default_rng(seed)
    architecture = config["model"]["architecture"]
    with ExperimentLogger(
        team="graph", model=f"fixed_id_corner_{architecture}", config=config,
        seed=seed, job_type="training",
    ) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write(output / "effective_config.yaml", config)
        print(f"run_id={logger.run_id} architecture={architecture} seed={seed} output={output}", flush=True)
        splits, manifest = load_splits(
            Path(config["data"]["root"]), config["data"]["expected_dataset_sha256"], device,
        )
        model = FixedIdCornerPredictor(**config["model"]).to(device)
        mean, std = input_statistics(splits["train"])
        model.input_mean.copy_(mean)
        model.input_std.copy_(std)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"], fused=True,
        )
        history = []
        best = float("inf")
        best_epoch = 0
        previous_elapsed = 0.0
        resume = config.get("resume", {}).get("checkpoint")
        if resume:
            checkpoint, resume_path, best_path = load_resume_checkpoint(config, manifest["dataset_sha256"])
            restore_training_state(checkpoint, model, optimizer, shuffle, len(splits["train"]["inputs"]))
            if not torch.equal(model.input_mean, mean) or not torch.equal(model.input_std, std):
                raise ValueError("Resumed input normalization does not match the training data")
            history = list(checkpoint["history"])
            best_row = min(history, key=lambda row: row["validation"]["loss"])
            best = best_row["validation"]["loss"]
            best_epoch = best_row["epoch"]
            previous_elapsed = history[-1]["elapsed_seconds"]
            shutil.copyfile(best_path, output / "best.pt")
            logger.link_output(resume_path, checkpoint=True)
            logger.record(source_checkpoint_sha256=file_sha256(resume_path),
                          source_best_checkpoint_sha256=file_sha256(best_path), resumed_epoch=checkpoint["epoch"])
            write(output / "history.json", history)
            print(f"Resumed epoch={checkpoint['epoch']} source={resume_path} prior_best_loss={best:.6f} "
                  f"best_epoch={best_epoch} epochs_without_improvement={checkpoint['epoch'] - best_epoch}", flush=True)
        baseline = metrics_from_totals(error_totals(torch.zeros_like(splits["validation"]["targets"]),
                                                   splits["validation"]), model.displacement_scale_m)
        source_hashes = {
            name: file_sha256(Path(__file__).resolve().parents[1] / name)
            for name in ("training/fixed_id_corners.py", "graph/temporal/fixed_id_corner.py",
                         "graph/temporal/causal_transformer.py", "data_collection/temporal/fixed_id_corner_sequences.py")
        }
        logger.record(
            dataset_sha256=manifest["dataset_sha256"], source_hashes=source_hashes,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            device=str(device), gpu_name=torch.cuda.get_device_name(device),
            baseline_last_detection=baseline, calibrated_split_used=False,
            synthetic_occlusion=False, input_width=13, target="last-observation displacement",
        )
        write(output / "baseline.json", baseline)
        print(f"Baseline last detection: {json.dumps(baseline)}", flush=True)
        started = time.monotonic()
        stop_reason = "maximum_epochs" if patience is not None else "epoch_budget"
        for epoch in range(len(history) + 1, settings["epochs"] + 1):
            order = shuffle.permutation(len(splits["train"]["inputs"]))
            order_sha256 = hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest()
            train_metrics = execute_epoch(
                model, splits["train"], torch.as_tensor(order, device=device), config, output, epoch, optimizer,
            )
            validation_metrics = execute_epoch(
                model, splits["validation"], torch.arange(len(splits["validation"]["inputs"]), device=device),
                config, output, epoch,
            )
            best, best_epoch, stale_epochs, improved, should_stop = update_early_stopping(
                validation_metrics["loss"], epoch, best, best_epoch, patience,
            )
            row = dict(epoch=epoch, train=train_metrics, validation=validation_metrics,
                       batch_order_sha256=order_sha256, elapsed_seconds=previous_elapsed + time.monotonic() - started,
                       best_epoch=best_epoch, epochs_without_improvement=stale_epochs)
            history.append(row)
            write(output / "history.json", history)
            logger.log(dict(train=train_metrics, validation=validation_metrics,
                            best_epoch=best_epoch, epochs_without_improvement=stale_epochs), step=epoch)
            if improved:
                save_checkpoint(output / "best.pt", model, optimizer, config, manifest["dataset_sha256"],
                                logger.run_id, epoch, history, shuffle)
            save_checkpoint(output / "last.pt", model, optimizer, config, manifest["dataset_sha256"],
                            logger.run_id, epoch, history, shuffle)
            print(
                f"EPOCH {epoch}/{settings['epochs']} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={validation_metrics['loss']:.6f} "
                f"visible_rmse_mm={validation_metrics['visible_rmse_mm']:.3f} "
                f"missing_rmse_mm={validation_metrics['missing_rmse_mm']:.3f} "
                f"best_epoch={best_epoch} no_improvement={stale_epochs}/{patience if patience is not None else 'off'}",
                flush=True,
            )
            if should_stop:
                stop_reason = "early_stopping"
                break
        for name in ("last.pt", "best.pt", "history.json", "baseline.json", "effective_config.yaml"):
            logger.link_output(output / name)
        completed_epochs = history[-1]["epoch"]
        logger.record(completed_epochs=completed_epochs, best_validation_loss=best,
                      best_epoch=best_epoch, stop_reason=stop_reason)
        write(output / "completed.json", dict(epochs=completed_epochs, best_validation_loss=best,
                                               best_epoch=best_epoch, stop_reason=stop_reason))
        print(f"Completed epoch {completed_epochs}, reason={stop_reason}, best_epoch={best_epoch}: {output}", flush=True)
        return str(output)
