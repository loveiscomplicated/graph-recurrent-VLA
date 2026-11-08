"""Resume selected-vertex M0 training without overwriting its original runs."""

from copy import deepcopy
import json
from pathlib import Path

import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from training.gnn.trainer import atomic_torch_save, train as train_gnn
from utils.experiment_logger import ExperimentLogger


def train(config):
    """Continue one completed run, retaining optimizer and best weights."""
    threads = config.get("torch_threads", 4)
    if type(threads) is not int or threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    torch.set_num_threads(threads)
    seed = config["seed"]
    target_epoch = config["target_epoch"]
    if type(seed) is not int or seed < 0 or type(target_epoch) is not int or target_epoch < 60:
        raise ValueError("seed must be nonnegative and target_epoch must be at least 60")
    source_matrix = Path(config["source_matrix"]) / "matrix.json"
    matrix = json.loads(source_matrix.read_text())
    matches = [row for row in matrix["runs"] if row["mode"] == "baseline" and row["seed"] == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one baseline source run for seed {seed}")
    row = matches[0]
    source_epoch = row["epochs"]
    if target_epoch <= source_epoch or file_sha256(row["checkpoint"]) != row["checkpoint_sha256"]:
        raise ValueError("The original source run must be complete and its best checkpoint unchanged")
    source = Path(row["output"]) / "last.pt"
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload["epoch"] != source_epoch or payload["rng"]["device_type"] != "cuda":
        raise ValueError(f"Expected a complete epoch-{source_epoch} CUDA last.pt checkpoint")
    effective = deepcopy(payload["config"])
    effective["training"]["epochs"] = target_epoch
    if "early_stopping" in config:
        effective["training"]["early_stopping"] = deepcopy(config["early_stopping"])
    if "validation_selection" in config:
        effective["data"]["validation_selection"] = config["validation_selection"]
    effective["resume"] = str(source)
    with ExperimentLogger(team="graph", config=config, seed=seed,
                          model="selected_vertex_continuation", job_type="train") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        effective["output"] = {"root": str(output / "training")}
        write_json(output / "effective_config.json", config)
        path = train_gnn(effective)
        last = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        history = json.loads((path / "history.json").read_text())
        completed_epoch = last["epoch"]
        if (not history or completed_epoch > target_epoch
                or [h["epoch"] for h in history] != list(range(source_epoch + 1, completed_epoch + 1))
                or (completed_epoch < target_epoch and "early_stopping" not in config)):
            raise ValueError("Continuation history does not match the completed checkpoint")
        best_epoch = min(history, key=lambda item: item["validation"]["loss"])["epoch"]
        baseline_path = path / "baseline_validation.json"
        if baseline_path.exists() and last["best_loss"] == json.loads(baseline_path.read_text())["loss"]:
            best_epoch = source_epoch
        elif not baseline_path.exists() and last["best_loss"] == row["validation_loss"]:
            best_epoch = row["best_epoch"]
        final_checkpoint = path / "final_epoch.pt"
        atomic_torch_save(final_checkpoint, {
            "kind": "final_weights", "model": last["model"], "config": last["config"],
            "validation_loss": history[-1]["validation"]["loss"],
            "dataset_sha256": last["dataset_sha256"],
        })
        result = dict(row, output=str(path), checkpoint=str(path / "best.pt"),
                      checkpoint_sha256=file_sha256(path / "best.pt"),
                      validation_loss=last["best_loss"], best_epoch=best_epoch,
                      epochs=completed_epoch, updates=completed_epoch * (row["updates"] // row["epochs"]),
                      early_stopped=completed_epoch < target_epoch,
                      source_last=str(source), source_last_sha256=file_sha256(source),
                      final_checkpoint=str(final_checkpoint),
                      final_checkpoint_sha256=file_sha256(final_checkpoint),
                      final_validation_loss=history[-1]["validation"]["loss"])
        write_json(output / "matrix.json", {"core_dataset_sha256": matrix["core_dataset_sha256"],
                                            "runs": [result]})
        logger.link_output(output / "matrix.json")
        logger.link_output(final_checkpoint)
        print(f"Continuation completed for seed {seed}: {output}", flush=True)
    return output
