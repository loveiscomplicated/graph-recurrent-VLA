"""Continue the fixed-Hole Expert Transformer and retain periodic checkpoints."""

import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from training.fixed_id_corners import (
    execute_epoch, input_statistics, load_resume_checkpoint, load_splits,
    restore_training_state, save_checkpoint, update_early_stopping,
)
from utils.experiment_logger import ExperimentLogger, write


def train(config):
    """Restore the source run and train until validation loss stops improving."""
    settings = config["training"]
    patience = settings["early_stopping"]["patience"]
    interval = settings["checkpoint_interval_epochs"]
    for name, value in (("early_stopping.patience", patience), ("checkpoint_interval_epochs", interval)):
        if type(value) is not int or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")

    device = torch.device(settings["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The fixed-Hole Expert Transformer continuation requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(settings["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset_hash = config["data"]["expected_dataset_sha256"]
    checkpoint, source, source_best = load_resume_checkpoint(config, dataset_hash)
    if config["model"]["architecture"] != "transformer" or checkpoint["epoch"] != config["resume"]["epoch"]:
        raise ValueError("Expected the configured source Transformer epoch")

    with ExperimentLogger(team="graph", model="fixed_id_corner_transformer", config=config,
                          seed=config["seed"], job_type="training") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True)
        print(f"Transformer early stopping output: {output}", flush=True)
        write(output / "effective_config.yaml", config)
        logger.link_output(source, checkpoint=True)
        logger.record(dataset_sha256=dataset_hash, resumed_epoch=checkpoint["epoch"],
                      source_checkpoint_sha256=file_sha256(source), source_best_checkpoint_sha256=file_sha256(source_best))

        splits, _ = load_splits(Path(config["data"]["root"]), dataset_hash, device)
        train_data, validation_data = splits["train"], splits["validation"]
        train_size = len(train_data["inputs"])
        validation_order = torch.arange(len(validation_data["inputs"]), device=device)
        model = FixedIdCornerPredictor(**config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"], fused=True)
        shuffle = np.random.default_rng(config["seed"])
        restore_training_state(checkpoint, model, optimizer, shuffle, train_size)
        mean, std = input_statistics(train_data)
        if not torch.equal(model.input_mean, mean) or not torch.equal(model.input_std, std):
            raise ValueError("Resumed input normalization differs from the training data")
        history = list(checkpoint["history"])
        best_row = min(history, key=lambda row: row["validation"]["loss"])
        best_loss, best_epoch = best_row["validation"]["loss"], best_row["epoch"]
        start = checkpoint["epoch"]
        replay = execute_epoch(model, validation_data, validation_order, config, output, start)
        original = history[-1]["validation"]
        np.testing.assert_allclose([replay[key] for key in original], list(original.values()), rtol=1e-4, atol=1e-5)
        write(output / "validation_replay.json", dict(epoch=start, source=original, replay=replay))
        write(output / "history.json", history)
        shutil.copyfile(source_best, output / "best.pt")
        stop_reason = "maximum_epochs"
        previous_elapsed = history[-1]["elapsed_seconds"]
        started = time.monotonic()

        for epoch in range(start + 1, settings["epochs"] + 1):
            order = shuffle.permutation(train_size)
            train_metrics = execute_epoch(model, train_data, torch.as_tensor(order, device=device),
                                          config, output, epoch, optimizer)
            validation = execute_epoch(model, validation_data, validation_order, config, output, epoch)
            best_loss, best_epoch, stale, improved, should_stop = update_early_stopping(
                validation["loss"], epoch, best_loss, best_epoch, patience,
            )
            history.append(dict(epoch=epoch, train=train_metrics, validation=validation,
                                batch_order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),
                                best_epoch=best_epoch, epochs_without_improvement=stale,
                                elapsed_seconds=previous_elapsed + time.monotonic() - started))
            save_checkpoint(output / "last.pt", model, optimizer, config, dataset_hash,
                            logger.run_id, epoch, history, shuffle)
            if improved:
                save_checkpoint(output / "best.pt", model, optimizer, config, dataset_hash,
                                logger.run_id, epoch, history, shuffle)
            if epoch % interval == 0:
                snapshot = output / f"epoch_{epoch:03d}.pt"
                save_checkpoint(snapshot, model, optimizer, config, dataset_hash, logger.run_id, epoch, history, shuffle)
                logger.link_output(snapshot)
            write(output / "history.json", history)
            logger.log(dict(train=train_metrics, validation=validation, best_epoch=best_epoch,
                            epochs_without_improvement=stale), step=epoch)
            print(f"Epoch {epoch}/{settings['epochs']} validation={validation['loss']:.6f} "
                  f"best_epoch={best_epoch} no_improvement={stale}/{patience}", flush=True)
            if should_stop:
                stop_reason = "early_stopping"
                break

        summary = dict(epochs=history[-1]["epoch"], best_validation_loss=best_loss,
                       best_epoch=best_epoch, stop_reason=stop_reason)
        write(output / "completed.json", summary)
        logger.record(**summary)
        for name in ("last.pt", "best.pt", "completed.json", "history.json", "validation_replay.json", "effective_config.yaml"):
            logger.link_output(output / name)
        print(f"Completed epoch {summary['epochs']}, reason={stop_reason}, best_epoch={best_epoch}", flush=True)
        return str(output)
