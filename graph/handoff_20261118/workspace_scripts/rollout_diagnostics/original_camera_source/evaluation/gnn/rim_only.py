"""Evaluate a filtered M0 on the same development frames as an optional M0 control."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_family import infer
from evaluation.gnn.vertex_metrics import statistics
from graph.build_graph import NodeType
from training.gnn.data import canonical_hash
from training.gnn.trainer import build_model
from training.gnn.vertex_family_data import VertexFamilyDataset
from data_collection.vertex_family_manifest import read_manifest
from utils.experiment_logger import ExperimentLogger


def read_runs(directories, dataset_hash, expected_filter):
    """Validate checkpoints, preprocessing identity, and unique M0 seeds."""
    runs = {}
    for directory in directories:
        matrix = json.loads((Path(directory) / "matrix.json").read_text())
        if matrix["core_dataset_sha256"] != dataset_hash:
            raise ValueError("Matrix and evaluation dataset differ")
        for row in matrix["runs"]:
            if row["mode"] != "baseline":
                continue
            seed = row["seed"]
            if seed in runs:
                raise ValueError(f"Duplicate baseline seed: {seed}")
            path = Path(row["checkpoint"])
            if file_sha256(path) != row["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            settings = payload["config"]
            actual_filter = settings["data"].get("target_filter", "none")
            if actual_filter != expected_filter or settings["model"]["use_semantic_ids"]:
                raise ValueError("Expected M0 checkpoint with the configured target filter")
            identity = [dataset_hash, "baseline"]
            if expected_filter != "none":
                identity.append({"target_filter": expected_filter, "version": 1})
            if payload["dataset_sha256"] != canonical_hash(identity):
                raise ValueError("Checkpoint preprocessing identity does not match")
            runs[seed] = (path, payload)
    if not runs:
        raise ValueError("No baseline checkpoints found in the supplied matrices")
    return runs


def evaluate(config):
    """Write per-frame metrics and paired control comparisons for filtered M0."""
    torch.set_num_threads(config.get("torch_threads", 4))
    root = Path(config["family_dataset"])
    manifest = read_manifest(root)
    target_filter = config["target_filter"]
    if target_filter not in ("detector_rim", "oracle_rim"):
        raise ValueError("target_filter must be detector_rim or oracle_rim")
    filtered = VertexFamilyDataset(root, manifest, "validation", "baseline", target_filter)
    original = VertexFamilyDataset(root, manifest, "validation", "baseline")
    datasets = {"filtered": filtered, "original": original}
    runs = {"filtered": read_runs(config["training_matrices"], manifest["dataset_sha256"], target_filter)}
    if config.get("reference_matrices"):
        runs["original"] = read_runs(config["reference_matrices"], manifest["dataset_sha256"], "none")
        if set(runs["filtered"]) != set(runs["original"]):
            raise ValueError("Filtered and reference checkpoints must have the same seeds")
        for seed in runs["filtered"]:
            filtered_config = runs["filtered"][seed][1]["config"]
            reference_config = runs["original"][seed][1]["config"]
            for key in ("model", "task", "training", "optimizer", "scheduler", "graph", "confidence"):
                if filtered_config[key] != reference_config[key]:
                    raise ValueError(f"Filtered and reference seed {seed} differ in {key}")
    device = torch.device(config["device"])
    with ExperimentLogger(team="graph", config=config, seed=min(runs["filtered"]),
                          model="rim_only_m0", job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "effective_config.json", config)
        original_counts = np.array([
            int((sample.graph.node_valid & (sample.graph.object_type == NodeType.BOX)).sum())
            for sample in original.samples
        ])
        filtered_counts = np.array([
            int((sample.graph.node_valid & (sample.graph.object_type == NodeType.BOX)).sum())
            for sample in filtered.samples
        ])
        groups = {
            "all": np.ones(len(filtered_counts), dtype=bool),
            "original_target_present": original_counts > 0,
            "filtered_target_present": filtered_counts > 0,
            "filtered_target_missing": filtered_counts == 0,
        }
        summaries = {"groups": {name: int(mask.sum()) for name, mask in groups.items()}}
        errors_by_mode = {}
        for mode, mode_runs in runs.items():
            errors_by_mode[mode] = {}
            for seed, (checkpoint, payload) in sorted(mode_runs.items()):
                model = build_model(payload["config"]["model"], device)
                model.load_state_dict(payload["model"], strict=True)
                tensors, targets, errors = infer(model, datasets[mode], device, config.get("batch_size", 128), "natural")
                ids = np.array([sample.sample_ids[0] for sample in datasets[mode].samples])
                counts = np.array([int((sample.graph.node_valid & (sample.graph.object_type == NodeType.BOX)).sum())
                                   for sample in datasets[mode].samples])
                np.savez_compressed(output / f"{mode}_seed_{seed}.npz", sample_ids=ids,
                                    target=targets, target_count=counts, **tensors, **errors)
                errors_by_mode[mode][seed] = errors
                summaries[f"{mode}_seed_{seed}"] = {
                    "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                    "frames": len(ids), "target_present": int((counts > 0).sum()),
                    "metrics_by_group": {
                        group: {key: statistics(value[mask]) for key, value in errors.items()}
                        for group, mask in groups.items()
                    },
                }
        if "original" in runs:
            first, second = datasets["filtered"].samples, datasets["original"].samples
            if [s.sample_ids for s in first] != [s.sample_ids for s in second]:
                raise ValueError("Filtered and original frame IDs differ")
            paired = {}
            for seed in sorted(runs["filtered"]):
                paired[str(seed)] = {
                    group: {
                        key: statistics((errors_by_mode["filtered"][seed][key] - values)[mask])
                        for key, values in errors_by_mode["original"][seed].items()
                    }
                    for group, mask in groups.items()
                }
            summaries["paired_filtered_minus_original"] = paired
        write_json(output / "summary.json", summaries)
        logger.link_output(output / "effective_config.json")
        logger.link_output(output / "summary.json")
        print(f"Rim-only evaluation: {output}", flush=True)
    return output
