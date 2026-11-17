"""Train the four vertex-aggregation variants through the existing GNN trainer."""

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import multiprocessing
from pathlib import Path
import traceback

import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from main import load_config
from training.gnn.cached_data import build_data
from training.gnn.trainer import build_model, resolve_config, train as train_gnn
from utils.experiment_logger import ExperimentLogger


VARIANTS = {
    "baseline": {"global_geometry": False, "readout": "attention"},
    "geometry": {"global_geometry": True, "readout": "attention"},
    "multiseed": {"global_geometry": False, "readout": "multiseed"},
    "combined": {"global_geometry": True, "readout": "multiseed"},
}
AGGREGATION_SETTINGS = {
    "global_geometry", "geometry_dim", "readout", "readout_seeds", "readout_dim", "readout_heads",
}


def comparison_protocol(config):
    """Extract settings that must match across all four newly trained models."""
    names = ("seed", "model", "data", "graph", "confidence", "task", "training",
             "optimizer", "scheduler", "detector", "device", "mode")
    protocol = {name: deepcopy(config[name]) for name in names}
    for name in ("global_geometry", "readout"):
        protocol["model"].pop(name, None)
    return protocol


def checkpoint_audit(path):
    """Read actual saved settings and strictly load weights, never infer defaults."""
    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = resolve_config(payload["config"])
    model = build_model(config["model"], torch.device("cpu"))
    model.load_state_dict(payload["model"], strict=True)
    audit = {
        "path": str(source), "sha256": file_sha256(source),
        "model": config["model"], "config": config,
        "dataset_sha256": payload["dataset_sha256"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "strict_load": True,
    }
    return audit


def prepare_ablation(config):
    """Resolve one recipe and verify its source architecture and fixed cache."""
    parallel_models = config.get("parallel_models", 1)
    if type(parallel_models) is not int or not 1 <= parallel_models <= len(VARIANTS):
        raise ValueError("parallel_models must be an integer from 1 to 4")
    base = resolve_config(load_config(config["base_config"]))
    aggregation = config.get("aggregation", {})
    dimension_names = {"geometry_dim", "readout_seeds", "readout_dim", "readout_heads"}
    if not isinstance(aggregation, dict) or set(aggregation) - dimension_names:
        raise ValueError(f"aggregation must contain only dimension settings: {sorted(dimension_names)}")
    base["model"].update(aggregation)
    base["device"] = config.get("device", base["device"])
    if "num_workers" in config["data"]:
        base["data"]["num_workers"] = config["data"]["num_workers"]
    base["data"]["cache_dir"] = str(Path(config["data"]["cache_dir"]).expanduser().resolve())
    base = resolve_config(base)
    if base["data"]["factory"] != "training.gnn.cached_data:build_data":
        raise ValueError("Vertex ablation requires the existing published cache-only data factory")
    if base["mode"] != "train" or base["resume"] is not None:
        raise ValueError("Vertex ablation requires mode=train and resume=null for all four models")
    if base["model"]["name"] != "vertex_edge_gated_mpnn":
        raise ValueError("Vertex ablation requires model.name=vertex_edge_gated_mpnn")
    if base["task"]["factory"] != "training.gnn.tasks:Object21Task":
        raise ValueError("Vertex ablation requires the unchanged Object21Task")
    if base["data"]["empty_inputs"] != "keep":
        raise ValueError("Vertex ablation requires data.empty_inputs=keep to score missing groups")
    audit = checkpoint_audit(config["reference_checkpoint"])
    if audit["model"]["global_geometry"] or audit["model"]["readout"] != "attention":
        raise ValueError("reference_checkpoint must be the original attention-pool baseline")
    original_model = {name: value for name, value in audit["model"].items()
                      if name not in AGGREGATION_SETTINGS}
    selected_model = {name: value for name, value in base["model"].items()
                      if name not in AGGREGATION_SETTINGS}
    if selected_model != original_model:
        raise ValueError("Base model settings differ from the actual reference checkpoint")
    # Reuse all cache checksum/split checks. No detection or data regeneration.
    bundle = build_data(base, torch.Generator().manual_seed(base["seed"]))
    if bundle.identity["dataset_sha256"] != audit["dataset_sha256"]:
        raise ValueError("Selected cache differs from the reference checkpoint dataset identity")
    return base, audit


def train_variant(config, log_path=None):
    """Run the existing trainer, optionally keeping one process's console log."""
    if log_path is None:
        return train_gnn(config)
    with Path(log_path).open("x") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            return train_gnn(config)
        except BaseException:
            traceback.print_exc()
            raise


def train_variants(configs, parallel_models, output):
    """Yield completed independent runs; spawn keeps each CUDA RNG/context local."""
    if parallel_models == 1:
        for name, config in configs.items():
            print(f"Training vertex ablation: {name}", flush=True)
            yield name, train_variant(config)
        return

    log_directory = output / "model_logs"
    log_directory.mkdir()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=parallel_models, mp_context=context) as executor:
        futures = {}
        for name, config in configs.items():
            log_path = log_directory / f"{name}.log"
            print(f"Training vertex ablation: {name}; console={log_path}", flush=True)
            future = executor.submit(train_variant, config, log_path)
            futures[future] = name
        for future in as_completed(futures):
            yield futures[future], future.result()


def train(config):
    """Train and record every variant from scratch after a common preflight.

    The source checkpoint supplies provenance, not initialization weights or a
    baseline score. All models use the existing loss-based best-checkpoint rule.
    """
    base, audit = prepare_ablation(config)
    effective = deepcopy(config)
    effective.setdefault("parallel_models", 1)
    effective["base_config_effective"] = base
    effective["variants"] = VARIANTS
    with ExperimentLogger(team="graph", config=effective, seed=base["seed"],
                          model="vertex_aggregation_ablation", job_type="train") as experiment:
        output = Path(config["output"]["root"]).expanduser().resolve() / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "complete": False, "reference_checkpoint": audit,
            "protocol": comparison_protocol(base), "variants": {},
            "reference_training_differences": {
                name: {"reference": audit["config"][name], "selected": base[name]}
                for name in ("seed", "task", "training", "optimizer", "scheduler")
                if audit["config"][name] != base[name]
            },
        }
        manifest_path = output / "ablation_manifest.json"
        write_json(manifest_path, manifest)
        experiment.link_output(manifest_path)
        configs = {}
        for name, flags in VARIANTS.items():
            variant = deepcopy(base)
            variant["model"].update(flags)
            variant["output"]["root"] = str(output / name)
            configs[name] = variant
        for name, directory in train_variants(configs, effective["parallel_models"], output):
            checkpoint = directory / "best.pt"
            trained = checkpoint_audit(checkpoint)
            manifest["variants"][name] = {
                "checkpoint": str(checkpoint), "sha256": trained["sha256"],
                "parameter_count": trained["parameter_count"], "model": trained["model"],
            }
            write_json(manifest_path, manifest)
            print(f"Completed vertex ablation: {name}; checkpoint={checkpoint}", flush=True)
        manifest["complete"] = True
        write_json(manifest_path, manifest)
        experiment.record(manifest=str(manifest_path), dataset_sha256=audit["dataset_sha256"])
        print(f"Four-model training complete: {manifest_path}", flush=True)
        return manifest_path
