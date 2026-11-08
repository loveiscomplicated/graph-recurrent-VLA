"""Run matched family-information variants through the existing GNN trainer."""

from copy import deepcopy
import hashlib
from pathlib import Path
import json

import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from graph.vertex_family import FAMILY_MODES
from training.gnn.trainer import build_model, seed_everything, train as train_gnn
from data_collection.vertex_family_manifest import read_manifest
from utils.experiment_logger import ExperimentLogger


def initialization_hash(model_settings, seed):
    """Fingerprint fresh weights to verify identical initialization across modes."""
    seed_everything(seed, torch.device("cpu"))
    model = build_model(model_settings, torch.device("cpu"))
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(config):
    """Execute fresh seeded comparisons without modifying production architecture.

    The same seed initializes identical tensors and generates the same shuffled
    sample order for every variant. The optional M0 filter changes target nodes.
    No source checkpoint is loaded; loss and checkpoint selection are unchanged.
    """
    torch.set_num_threads(config.get("torch_threads", 4))
    manifest = read_manifest(config["family_dataset"])
    modes = config.get("modes", list(FAMILY_MODES))
    if not modes or any(mode not in FAMILY_MODES for mode in modes):
        raise ValueError(f"modes must select from {FAMILY_MODES}")
    target_filter = config.get("target_filter", "none")
    if target_filter not in ("none", "detector_rim", "oracle_rim"):
        raise ValueError("target_filter must be none, detector_rim, or oracle_rim")
    if target_filter != "none" and modes != ["baseline"]:
        raise ValueError("A target_filter requires modes: [baseline] (M0)")
    seeds = config["seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    with ExperimentLogger(team="graph", config=config, seed=seeds[0],
                          model="vertex_family_matrix", job_type="train") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        completed = []
        train_frame_count = sum(row["split"] == "train" for row in manifest["samples"])
        batch_size = config["recipe"]["training"]["batch_size"]
        updates_per_epoch = (train_frame_count + batch_size - 1) // batch_size
        write_json(output / "effective_config.json", config)
        for seed in seeds:
            expected_initialization = None
            for mode in modes:
                effective = deepcopy(config["recipe"])
                effective["seed"] = seed
                effective["device"] = config["device"]
                effective["model"]["use_semantic_ids"] = mode != "baseline"
                effective["data"] = dict(factory="training.gnn.vertex_family_data:build_data",
                                          family_dataset=config["family_dataset"], family_mode=mode,
                                          num_workers=0, empty_inputs="keep")
                if target_filter != "none":
                    effective["data"]["target_filter"] = target_filter
                effective["output"] = dict(root=str(output / mode / f"seed_{seed}"))
                effective["resume"] = None
                initial_hash = initialization_hash(effective["model"], seed)
                if expected_initialization is not None and initial_hash != expected_initialization:
                    raise ValueError("Family modes must initialize identical model tensors for each seed")
                expected_initialization = initial_hash
                effective["family_study"] = dict(mode=mode, core_dataset_sha256=manifest["dataset_sha256"],
                                                  parent_matrix_run_id=logger.run_id)
                path = train_gnn(effective)
                payload = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
                history = json.loads((path / "history.json").read_text())
                best_epoch = min(history, key=lambda row: row["validation"]["loss"])["epoch"]
                completed.append(dict(mode=mode, seed=seed, output=str(path), checkpoint=str(path / "best.pt"),
                                      checkpoint_sha256=file_sha256(path / "best.pt"), best_epoch=best_epoch,
                                      validation_loss=payload["validation_loss"],
                                      initialization_sha256=initial_hash,
                                      epochs=len(history), updates=len(history) * updates_per_epoch,
                                      early_stop_enabled=bool(effective["training"].get("early_stopping")),
                                      early_stopped=len(history) < effective["training"]["epochs"]))
                write_json(output / "matrix.json", dict(core_dataset_sha256=manifest["dataset_sha256"], runs=completed))
                logger.link_output(output / "matrix.json")
                print(f"Family matrix finished {mode}/seed={seed}; checkpoint={path / 'best.pt'}", flush=True)
        logger.record(core_dataset_sha256=manifest["dataset_sha256"], completed_runs=len(completed))
        logger.link_output(output / "effective_config.json")
        print(f"Vertex family training matrix: {output}", flush=True)
    return output
