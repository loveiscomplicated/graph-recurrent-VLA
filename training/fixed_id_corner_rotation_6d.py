"""Train the sign-invariant corner predictor without changing legacy runs."""

import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.temporal.corner_rotation_6d import (
    CHECKPOINT_FORMAT, INPUT_SCHEMA, Rotation6DCornerPredictor, encode_inputs,
)
from training.fixed_id_corners import execute_epoch, load_splits, update_early_stopping
from utils.experiment_logger import ExperimentLogger, write


def training_statistics(data):
    """Fit normalization on valid training rows only, after the 6D conversion."""
    values = encode_inputs(data["inputs"], data["context"])[data["context"]].double()
    return values.mean(0).float(), values.std(0, correction=0).float().clamp_min(0.001)


def save_checkpoint(path, model, optimizer, config, dataset_hash, run_id, history, shuffle):
    """Save a separate checkpoint contract and its training normalization."""
    state = dict(format=CHECKPOINT_FORMAT, input_schema=INPUT_SCHEMA, model=model.state_dict(),
                 optimizer=optimizer.state_dict(), config=config, dataset_sha256=dataset_hash,
                 run_id=run_id, epoch=history[-1]["epoch"], history=history,
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                 shuffle_state=shuffle.bit_generator.state)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config):
    """Train a new 6D model on existing fixed-ID sequences; never read test data."""
    settings = config["training"]
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    if settings["precision"] not in ("float32", "bfloat16"):
        raise ValueError("training.precision must be float32 or bfloat16")
    if settings["precision"] == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 training requires a supported CUDA device")
    for name in ("epochs", "batch_size", "log_every_batches", "cpu_threads"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    patience = settings.get("early_stopping_patience")
    if patience is not None and (type(patience) is not int or patience < 1):
        raise ValueError("training.early_stopping_patience must be null or a positive integer")
    torch.set_num_threads(settings["cpu_threads"])
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    shuffle = np.random.default_rng(seed)
    with ExperimentLogger(team="graph", model="fixed_id_corner_rotation_6d", config=config,
                          seed=seed, job_type="training") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write(output / "effective_config.yaml", config)
        splits, manifest = load_splits(config["data"]["root"], config["data"]["expected_dataset_sha256"], device)
        model = Rotation6DCornerPredictor(**config["model"]).to(device)
        mean, std = training_statistics(splits["train"])
        model.input_mean.copy_(mean)
        model.input_std.copy_(std)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"])
        root = Path(__file__).resolve().parents[1]
        source_names = ("training/fixed_id_corner_rotation_6d.py", "graph/temporal/corner_rotation_6d.py",
                        "training/fixed_id_corners.py", "graph/temporal/fixed_id_corner.py",
                        "graph/temporal/causal_transformer.py", "data_collection/temporal/fixed_id_corner_sequences.py")
        logger.record(dataset_sha256=manifest["dataset_sha256"], input_schema=INPUT_SCHEMA,
                      source_hashes={name: file_sha256(root / name) for name in source_names})
        history = []
        best, best_epoch = float("inf"), 0
        started = time.monotonic()
        for epoch in range(1, settings["epochs"] + 1):
            order = shuffle.permutation(len(splits["train"]["inputs"]))
            train_metrics = execute_epoch(model, splits["train"], torch.as_tensor(order, device=device),
                                          config, output, epoch, optimizer)
            validation = execute_epoch(model, splits["validation"],
                                       torch.arange(len(splits["validation"]["inputs"]), device=device),
                                       config, output, epoch)
            best, best_epoch, stale, improved, stop = update_early_stopping(
                validation["loss"], epoch, best, best_epoch, patience)
            history.append(dict(epoch=epoch, train=train_metrics, validation=validation,
                                batch_order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),
                                elapsed_seconds=time.monotonic() - started, best_epoch=best_epoch,
                                epochs_without_improvement=stale))
            write(output / "history.json", history)
            logger.log(dict(train=train_metrics, validation=validation), step=epoch)
            if improved:
                save_checkpoint(output / "best.pt", model, optimizer, config, manifest["dataset_sha256"],
                                logger.run_id, history, shuffle)
            save_checkpoint(output / "last.pt", model, optimizer, config, manifest["dataset_sha256"],
                            logger.run_id, history, shuffle)
            if stop:
                break
        for name in ("best.pt", "last.pt", "history.json", "effective_config.yaml"):
            logger.link_output(output / name)
        logger.record(completed_epochs=len(history), best_epoch=best_epoch, best_validation_loss=best)
        return str(output)
