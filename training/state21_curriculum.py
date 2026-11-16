"""Train and extend Object21D models through the fixed-Hole curriculum."""

import json
from pathlib import Path

import torch

from data_collection.episode_dataset import file_sha256
from data_collection.state21_curriculum import (
    STAGES,
    prepare_cache,
    prepare_stage,
    read_graph_episode,
)
from main import load_config
from training.state21 import TrainingData, effective_config, fit
from utils.experiment_logger import write


def _training_bundle(data):
    """Load the saved graph manifest and check the pinned split and geometry."""
    manifest_path = Path(data["curriculum_cache"]) / "manifest.json"
    bundle = json.loads(manifest_path.read_text())
    if (
        bundle["format"] != "state21_fixed_hole_graphs_v1"
        or bundle["max_edge_m"] != data["max_edge_m"]
        or bundle["split_reference_sha256"] != file_sha256(data["split_reference"])
    ):
        raise ValueError("Curriculum cache settings or split differ from this run")
    return manifest_path, bundle


def _run_stages(config, status_path, status):
    """Train the available stages in order, with a new optimizer at each stage."""
    data = config["data"]
    manifest_path, bundle = _training_bundle(data)
    status["graph_manifest_sha256"] = file_sha256(manifest_path)
    write(status_path, status)
    episodes = [
        read_graph_episode(manifest_path.parent, row) for row in bundle["episodes"]
    ]
    previous = None
    for stage in STAGES:
        print(f"Stage {stage}", flush=True)
        train, validation, report = prepare_stage(bundle, episodes, stage, config)
        stage_root = status_path.parent / stage
        options = {
            **config,
            "data": {**data, "curriculum_stage": stage},
            "output": {"root": str(stage_root)},
            "initial_checkpoint": str(previous) if previous else None,
        }
        current = {
            "name": stage,
            "train_frames": len(train.windows),
            "validation_frames": len(validation.windows),
            "policy_counts": report["policy_counts"],
        }
        status.update(status="training", current_stage=current)
        write(status_path, status)
        output = fit(
            options, prepared=TrainingData(options["data"], train, validation, report)
        )
        previous = output / "last.pt"
        current.update(
            checkpoint=str(previous),
            checkpoint_sha256=file_sha256(previous),
            metrics=str(output / "training.json"),
        )
        status["completed"].append(current)
        write(status_path, status)


def run(config):
    """Prepare portable input graphs or execute one independent A–C2 workflow."""
    config = effective_config(config)
    if config.get("phase") == "prepare":
        prepare_cache(config)
        return
    if config.get("phase") != "train":
        raise ValueError("Curriculum phase must be prepare or train")
    if config.get("initial_checkpoint") or config.get("resume"):
        raise ValueError("The first curriculum stage starts from fresh model weights")
    torch.set_num_threads(config["training"]["cpu_threads"])
    name = config["model"]["name"]
    output = Path(config["output"]["root"]) / name
    output.mkdir(parents=True, exist_ok=False)
    path = output / "status.json"
    status = {
        "model": name,
        "device": config["device"],
        "status": "loading_graphs",
        "epochs_per_stage": config["training"]["epochs"],
        "stages": list(STAGES),
        "completed": [],
    }
    write(path, status)
    try:
        _run_stages(config, path, status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__)
        write(path, status)
        raise
    status.update(status="complete", current_stage=None)
    write(path, status)


def resume_final(config):
    """Extend the final stage with the existing optimizer, RNG and data recipe."""
    source = effective_config(load_config(config["source_config"]))
    name = config["model"]
    output_root = Path(source["output"]["root"])
    status_path = output_root / name / "status.json"
    status = json.loads(status_path.read_text())
    manifest_path, bundle = _training_bundle(source["data"])
    completed = status["completed"]
    if (
        status["model"] != name
        or status["status"] != "complete"
        or tuple(row["name"] for row in completed) != STAGES
        or status["graph_manifest_sha256"] != file_sha256(manifest_path)
    ):
        raise ValueError(
            f"Expected a complete curriculum with its original graph cache: {name}"
        )
    final = completed[-1]
    checkpoint = final["checkpoint"]
    if file_sha256(checkpoint) != final["checkpoint_sha256"]:
        raise ValueError(f"Final curriculum checkpoint checksum differs: {checkpoint}")
    if "resume" in config:
        resume = config["resume"]
        checkpoint = resume["checkpoint"]
        if file_sha256(checkpoint) != resume["sha256"]:
            raise ValueError(f"Continuation checkpoint checksum differs: {checkpoint}")
    epochs = config["epochs"]
    if type(epochs) is not int or epochs <= status["epochs_per_stage"]:
        raise ValueError("epochs must exceed the completed final-stage epoch count")

    training = {**source["training"], "epochs": epochs}
    if "early_stopping" in config:
        training["early_stopping"] = config["early_stopping"]
    options = {
        **source,
        "trainer": config["trainer"],
        "model": {**source["model"], "name": name},
        "data": {**source["data"], "curriculum_stage": final["name"]},
        "training": training,
        "resume": checkpoint,
        "output": config["output"],
        "continuation": config,
    }
    torch.set_num_threads(options["training"]["cpu_threads"])
    print(
        f"Preparing the original {name} final-stage windows from {checkpoint}",
        flush=True,
    )
    episodes = [
        read_graph_episode(manifest_path.parent, row) for row in bundle["episodes"]
    ]
    train, validation, report = prepare_stage(bundle, episodes, final["name"], options)
    if (len(train.windows), len(validation.windows)) != (
        final["train_frames"],
        final["validation_frames"],
    ):
        raise ValueError("Prepared frame counts differ from the completed final stage")
    print(
        f"Verified {len(train.windows)} train and {len(validation.windows)} validation windows",
        flush=True,
    )
    return fit(
        options, prepared=TrainingData(options["data"], train, validation, report)
    )
