"""Compare selected-training M0 with reference M0 on identical validation frames."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_metrics import (
    cluster_interval, statistics,
)
from evaluation.gnn.vertex_family import infer
from graph.build_graph import NodeType
from training.gnn.data import canonical_hash
from training.gnn.trainer import build_model
from training.gnn.vertex_family_data import VertexFamilyDataset
from data_collection.vertex_family_manifest import read_manifest
from utils.experiment_logger import ExperimentLogger


def load_baseline_runs(matrix_dir, manifest):
    """Check M0 checkpoint data identities, source hashes, and seed uniqueness."""
    matrix = json.loads((Path(matrix_dir) / "matrix.json").read_text())
    if matrix["core_dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("Matrix and dataset identity differ")
    runs = {}
    for row in matrix["runs"]:
        if row["mode"] != "baseline":
            continue
        seed = row["seed"]
        if seed in runs:
            raise ValueError(f"Duplicate M0 seed {seed}")
        path = Path(row["checkpoint"])
        if file_sha256(path) != row["checkpoint_sha256"]:
            raise ValueError(f"M0 checkpoint checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        if cfg["model"]["use_semantic_ids"] or cfg["data"].get("target_filter", "none") != "none":
            raise ValueError("Expected an unfiltered M0 checkpoint without semantic IDs")
        if payload["dataset_sha256"] != canonical_hash([manifest["dataset_sha256"], "baseline"]):
            raise ValueError("M0 checkpoint dataset identity mismatch")
        runs[seed] = (path, payload)
    if not runs:
        raise ValueError("No baseline checkpoints in matrix")
    return runs


def evaluate(config):
    """Save all Object21 metrics, aligned predictions, and paired parent bootstrap."""
    torch.set_num_threads(config.get("torch_threads", 4))
    selected_root = Path(config["selected_dataset"])
    reference_root = Path(config["reference_dataset"])
    selected_manifest = read_manifest(selected_root)
    reference_manifest = read_manifest(reference_root)
    selected_data = VertexFamilyDataset(selected_root, selected_manifest, "validation", "baseline")
    reference_data = VertexFamilyDataset(reference_root, reference_manifest, "validation", "baseline")
    selected_rows = [row for row in selected_manifest["samples"] if row["split"] == "validation"]
    reference_rows = [row for row in reference_manifest["samples"] if row["split"] == "validation"]
    identities = lambda rows: [(row["sample_id"], row["parent_id"], row["graph_sha256"]) for row in rows]
    if identities(selected_rows) != identities(reference_rows):
        raise ValueError("Validation graph IDs, parents, or graph hashes differ")
    runs = {
        "selected": load_baseline_runs(config["selected_matrix"], selected_manifest),
        "reference": load_baseline_runs(config["reference_matrix"], reference_manifest),
    }
    if set(runs["selected"]) != set(runs["reference"]):
        raise ValueError("Selected and reference seeds must match")
    for seed in runs["selected"]:
        candidate = runs["selected"][seed][1]["config"]
        original = runs["reference"][seed][1]["config"]
        for key in ("model", "task", "training", "optimizer", "scheduler", "graph", "confidence"):
            candidate_settings = candidate[key]
            reference_settings = original[key]
            if key == "training" and config.get("allow_different_epochs", False):
                candidate_settings = {name: value for name, value in candidate_settings.items() if name != "epochs"}
                reference_settings = {name: value for name, value in reference_settings.items() if name != "epochs"}
            if candidate_settings != reference_settings:
                raise ValueError(f"Seed {seed} differs in {key}")
    datasets = {"selected": selected_data, "reference": reference_data}
    parents = np.array([row["parent_id"] for row in selected_rows])
    counts = np.array([int((sample.graph.node_valid & (sample.graph.object_type == NodeType.BOX)).sum())
                       for sample in selected_data.samples])
    groups = {"all": np.ones(len(counts), dtype=bool), "target_present": counts > 0,
              "target_missing": counts == 0}
    selection_path = config.get("validation_selection")
    if selection_path:
        records = [json.loads(line) for line in Path(selection_path).read_text().splitlines()]
        if [(r["sample_id"], r["parent_id"]) for r in records] != [
            (r["sample_id"], r["parent_id"]) for r in selected_rows
        ]:
            raise ValueError("Physical validation selection does not match validation frame IDs and parents")
        for rule in ("peg_pass", "hole_pass", "both_pass"):
            groups["physical_" + rule] = np.array([record[rule] for record in records], dtype=bool)
    device = torch.device(config["device"])
    with ExperimentLogger(team="graph", config=config, seed=min(runs["selected"]),
                          model="selected_vertex_m0", job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "effective_config.json", config)
        all_errors = {}
        targets_by_mode = {}
        for mode, mode_runs in runs.items():
            all_errors[mode] = {}
            for seed, (checkpoint, payload) in sorted(mode_runs.items()):
                model = build_model(payload["config"]["model"], device)
                model.load_state_dict(payload["model"], strict=True)
                tensors, target, errors = infer(model, datasets[mode], device, config.get("batch_size", 128), "natural")
                if mode in targets_by_mode and not np.array_equal(targets_by_mode[mode], target):
                    raise ValueError("Targets changed across seeds")
                targets_by_mode[mode] = target
                np.savez_compressed(output / f"{mode}_seed_{seed}.npz",
                                    sample_ids=np.array([r["sample_id"] for r in selected_rows]),
                                    target=target, target_count=counts, **tensors, **errors)
                all_errors[mode][seed] = errors
                print(f"Evaluated {mode} seed {seed}: {checkpoint}", flush=True)
        if not np.array_equal(targets_by_mode["selected"], targets_by_mode["reference"]):
            raise ValueError("Validation targets differ between datasets")
        summary = {"groups": {name: int(mask.sum()) for name, mask in groups.items()},
                   "metrics": {}, "seeds": sorted(runs["selected"]),
                   "validation_selection_sha256": file_sha256(selection_path) if selection_path else None,
                   "note": "Check effective configs: datasets and/or optimizer update counts may differ."}
        for group, mask in groups.items():
            summary["metrics"][group] = {}
            for key in next(iter(all_errors["selected"].values())):
                candidate = np.stack([all_errors["selected"][seed][key] for seed in sorted(runs["selected"])])
                reference = np.stack([all_errors["reference"][seed][key] for seed in sorted(runs["reference"])])
                avg_candidate, avg_reference = candidate.mean(0)[mask], reference.mean(0)[mask]
                summary["metrics"][group][key] = {
                    "selected": statistics(avg_candidate), "reference": statistics(avg_reference),
                    "selected_minus_reference": cluster_interval(avg_candidate - avg_reference, parents[mask]),
                }
        write_json(output / "summary.json", summary)
        logger.link_output(output / "summary.json")
        logger.link_output(output / "effective_config.json")
        print(f"Selected M0 evaluation: {output}", flush=True)
    return output
