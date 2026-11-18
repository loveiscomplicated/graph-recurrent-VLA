"""Compare final Five-mode checkpoints on the same held-out frames."""

import json
from pathlib import Path

import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.storage import write_json
from graph.build_graph import NodeType
from training.gnn.packed_five_mode_data import PackedFrameBatches, read_splits
from training.gnn.trainer import build_model, import_callable, run_epoch
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def final_checkpoint(path, manifest_hash, condition, surface, *, condition_index_hash=None):
    """Check that the chosen weights are the completed run's validation best."""
    best = torch.load(path, map_location="cpu", weights_only=True)
    last = torch.load(path.with_name("last.pt"), map_location="cpu", weights_only=True)
    history = json.loads(path.with_name("history.json").read_text())
    config = best["config"]
    identity = dict(manifest_sha256=manifest_hash, include_surface=surface,
                    surface_object="peg" if surface else None, condition=condition)
    if config["data"].get("condition_index") is not None:
        if condition_index_hash is None:
            raise ValueError(f"Expected the training visibility index for checkpoint: {path}")
        identity["condition_index_sha256"] = condition_index_hash
    balanced_condition = config["data"].get("balanced_train_condition")
    if balanced_condition:
        identity["balanced_train_condition"] = balanced_condition
        if "train_mix" in config["data"]:
            identity["train_mix"] = config["data"]["train_mix"]
        else:
            recorded = json.loads(path.with_name("data_identity.json").read_text())
            identity["frames_per_condition"] = recorded["frames_per_condition"]
    start_epoch = 0
    candidates = history
    if config.get("resume"):
        parent_path = Path(config["resume"])
        parent_last = torch.load(parent_path, map_location="cpu", weights_only=True)
        parent_best, parent_epoch = final_checkpoint(
            parent_path.with_name("best.pt"), manifest_hash, condition, surface,
            condition_index_hash=condition_index_hash,
        )
        start_epoch = parent_last["epoch"]
        candidates = [
            dict(epoch=parent_epoch, validation=dict(loss=parent_best["validation_loss"])),
            *history,
        ]
    selected = min(candidates, key=lambda row: row["validation"]["loss"])
    early = config["training"].get("early_stopping")
    stopped = last.get("early_stopping_state", {})
    completed = last["epoch"] == config["training"]["epochs"]
    if early:
        completed |= (
            early["min_epochs"] <= last["epoch"] < config["training"]["epochs"]
            and stopped.get("stale_epochs", 0) >= early["patience"]
        )
    if (
        [row["epoch"] for row in history] != list(range(start_epoch + 1, last["epoch"] + 1))
        or not completed
        or best["kind"] != "best_weights"
        or best["dataset_sha256"] != canonical_hash(identity)
        or best["dataset_sha256"] != last["dataset_sha256"]
        or best["validation_loss"] != selected["validation"]["loss"]
        or best["validation_loss"] != last["best_loss"]
        or best["model"].keys() != last["best_model"].keys()
        or any(not torch.equal(value, last["best_model"][name])
               for name, value in best["model"].items())
    ):
        raise ValueError(f"Expected the validation-best checkpoint of a completed run: {path}")
    return best, selected["epoch"]


def verify_paired_frames(direct, fused, expected_frames):
    """Confirm paired targets, corner observations and original frame IDs."""
    if (len(direct.ids) != expected_frames or len(set(direct.ids)) != expected_frames
            or direct.ids != fused.ids or not torch.equal(direct.target, fused.target)
            or not torch.equal(direct.tcp, fused.tcp)):
        raise ValueError("Test frame identity, count or supervision differs between models")
    width = direct.xyz.size(1)
    corners = direct.kinds != NodeType.PAD
    if (not torch.equal(direct.kinds[corners], fused.kinds[:, :width][corners])
            or not torch.equal(direct.xyz[corners], fused.xyz[:, :width][corners])):
        raise ValueError("Detector corners differ between the paired test inputs")
    surface = fused.kinds != NodeType.PAD
    surface[:, :width] &= ~corners
    if not surface.any() or not (fused.kinds[surface] == NodeType.PEG).all():
        raise ValueError("Added test nodes must be Peg surface nodes")


def evaluate(config):
    """Evaluate the two models and record their paired test metrics."""
    path = Path(config["manifest"]).resolve()
    manifest = json.loads(path.read_text())
    manifest_hash = file_sha256(path)
    index_path = config.get("condition_index")
    condition_index = None
    condition_index_hash = None
    if index_path is not None:
        index_path = Path(index_path)
        condition_index = json.loads(index_path.read_text())
        condition_index_hash = file_sha256(index_path)
    condition = config["condition"]
    checkpoint_condition = config.get("checkpoint_condition", condition)
    variants = config.get("variants", {"detector_corners": False, "peg_surface": True})
    checkpoint_conditions = config.get("checkpoint_conditions", {})
    device = set_device(config["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision(config["matmul_precision"])
    loaders = {}
    for surface in set(variants.values()):
        rows = read_splits(path, manifest, ("test",), condition, surface,
                          condition_index=condition_index)["test"]
        loaders[surface] = PackedFrameBatches(rows, device, config["batch_size"], None)
    if False in loaders and True in loaders:
        verify_paired_frames(loaders[False], loaders[True], config["expected_frames"])
    reference = next(iter(loaders.values()))
    if len(reference.ids) != config["expected_frames"] or len(set(reference.ids)) != len(reference.ids):
        raise ValueError("Unexpected test frame count or duplicate frame IDs")
    frame_hash = canonical_hash(reference.ids)
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    results = {}
    with ExperimentLogger(team="graph", model="vertex_edge_gated_mpnn",
                          config=config, job_type="evaluate") as logger:
        for step, (name, surface) in enumerate(variants.items(), start=1):
            checkpoint = Path(config["checkpoints"][name])
            trained_on = checkpoint_conditions.get(name, checkpoint_condition)
            best, epoch = final_checkpoint(checkpoint, manifest_hash, trained_on, surface,
                                           condition_index_hash=condition_index_hash)
            logger.link_output(checkpoint, checkpoint=True)
            training = best["config"]
            model = build_model(training["model"], device)
            model.load_state_dict(best["model"], strict=True)
            task = import_callable(training["task"]["factory"])(**training["task"]["settings"])
            metrics = run_epoch(model, loaders[surface], task, device)
            logger.log(metrics, step=step, prefix=name)
            results[name] = dict(best_epoch=epoch, validation_loss=best["validation_loss"],
                                 checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                                 metrics=metrics)
            print(name, results[name], flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, dict(config=config, manifest_sha256=manifest_hash,
                               condition_index_sha256=condition_index_hash,
                               test_frame_ids_sha256=frame_hash, frames=config["expected_frames"],
                               split="test", results=results, evaluation_run_id=logger.run_id))
        logger.link_output(output)
