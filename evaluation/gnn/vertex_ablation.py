"""Evaluate all four newly trained vertex models under an audited paired protocol."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_metrics import (
    comparison_markdown, observation_groups, paired_improvements,
    per_frame_errors, summarize_errors,
)
from graph.build_graph import NodeType
from graph.pose_geometry import geometric_endpoint
from training.gnn.cached_data import build_data
from training.gnn.trainer import build_model, positive_integer
from training.gnn.vertex_ablation import VARIANTS, checkpoint_audit, comparison_protocol
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def infer(model, loader, device):
    """Collect ordered predictions and observed counts; never pass targets to GNN."""
    collected = {name: [] for name in (
        "sample_ids", "target", "counts", "object21", "endpoint_geom", "endpoint_residual",
    )}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            graph = batch.graph
            counts = torch.stack([
                (graph.node_valid & (graph.object_type == kind)).sum(dim=1)
                for kind in (NodeType.PEG, NodeType.BOX, NodeType.HOLE)
            ], dim=1)
            output = model(graph.to(device))
            for name in ("object21", "endpoint_geom", "endpoint_residual"):
                values = output[name].cpu()
                if not torch.isfinite(values).all():
                    raise ValueError(f"Nonfinite ablation output: {name}")
                collected[name].append(values)
            collected["target"].append(batch.targets["object21"])
            collected["counts"].append(counts)
            collected["sample_ids"].extend(batch.sample_ids)
    if not collected["sample_ids"] or len(set(collected["sample_ids"])) != len(collected["sample_ids"]):
        raise ValueError("Validation must contain nonempty, unique sample IDs")
    result = {name: torch.cat(values) for name, values in collected.items() if name != "sample_ids"}
    result["sample_ids"] = np.asarray(collected["sample_ids"])
    for start in (3, 13):
        torch.testing.assert_close(result["object21"][:, start:start + 4].norm(dim=-1),
                                   torch.ones(len(result["target"])), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result["endpoint_geom"], geometric_endpoint(result["object21"]))
    torch.testing.assert_close(result["object21"][:, 7:10],
                               result["endpoint_geom"] + result["endpoint_residual"])
    return result


def audit_matrix(manifest):
    """Reject incomplete, modified or differently trained checkpoint matrices."""
    if not manifest["complete"] or set(manifest["variants"]) != set(VARIANTS):
        raise ValueError("Evaluation requires all four newly trained ablation checkpoints")
    audits = {}
    for name, flags in VARIANTS.items():
        entry = manifest["variants"][name]
        audit = checkpoint_audit(entry["checkpoint"])
        if audit["sha256"] != entry["sha256"]:
            raise ValueError(f"Checkpoint bytes changed since matrix training: {name}")
        if comparison_protocol(audit["config"]) != manifest["protocol"]:
            raise ValueError(f"Ablation training protocol mismatch: {name}")
        if any(audit["model"][key] != value for key, value in flags.items()):
            raise ValueError(f"Checkpoint flags disagree with ablation variant: {name}")
        if audit["dataset_sha256"] != manifest["reference_checkpoint"]["dataset_sha256"]:
            raise ValueError(f"Ablation dataset identity mismatch: {name}")
        audits[name] = audit
    return audits


def evaluate(config):
    """Score all four matched models on every published validation frame."""
    effective = deepcopy(config)
    settings = effective.setdefault("evaluation", {})
    for name, default in (("batch_size", 64), ("sufficient_peg_vertices", 4),
                          ("sufficient_target_vertices", 4)):
        settings.setdefault(name, default)
        positive_integer(f"evaluation.{name}", settings[name])
    manifest_path = Path(config["manifest"]).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    audits = audit_matrix(manifest)
    device = set_device(effective.get("device", "auto"))
    data_config = deepcopy(audits["baseline"]["config"])
    data_config["training"]["batch_size"] = settings["batch_size"]
    # The trained cache path is authoritative; moving the cache needs an explicit override.
    if "data" in effective:
        data_config["data"]["cache_dir"] = effective["data"]["cache_dir"]
    bundle = build_data(data_config, torch.Generator().manual_seed(data_config["seed"]))
    if bundle.identity["dataset_sha256"] != audits["baseline"]["dataset_sha256"]:
        raise ValueError("Evaluation cache differs from the trained dataset")
    effective["training_protocol"] = manifest["protocol"]
    effective["checkpoint_audits"] = audits
    with ExperimentLogger(team="graph", config=effective, seed=data_config["seed"],
                          model="vertex_aggregation_ablation", job_type="evaluate") as experiment:
        output_dir = Path(effective["output"]["root"]).expanduser().resolve() / experiment.run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        summary = {
            "split": "validation", "checkpoint_selection": "minimum validation loss",
            "manifest_sha256": file_sha256(manifest_path), "device": str(device),
            "dataset_sha256": bundle.identity["dataset_sha256"],
            "observation_thresholds": settings, "models": {}, "paired_vs_baseline": {},
            "visibility_limit": "Count proxies do not establish full visibility or occlusion cause",
        }
        baseline = None
        baseline_errors = None
        baseline_count = audits["baseline"]["parameter_count"]
        for name, audit in audits.items():
            model = build_model(audit["model"], device)
            payload = torch.load(audit["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            experiment.link_output(audit["path"], checkpoint=True)
            data = infer(model, bundle.validation, device)
            if baseline is None:
                baseline = data
            else:
                if not np.array_equal(data["sample_ids"], baseline["sample_ids"]):
                    raise ValueError("Validation frame ordering differs between models")
                for key in ("target", "counts"):
                    if not torch.equal(data[key], baseline[key]):
                        raise ValueError(f"Validation {key} differs between models")
            groups = observation_groups(data["counts"].numpy(),
                                        settings["sufficient_peg_vertices"],
                                        settings["sufficient_target_vertices"])
            errors = per_frame_errors(data, data["target"])
            if baseline_errors is None:
                baseline_errors = errors
            else:
                summary["paired_vs_baseline"][name] = paired_improvements(baseline_errors, errors, groups)
            summary["models"][name] = {
                "parameter_count": audit["parameter_count"],
                "parameter_ratio_to_baseline": audit["parameter_count"] / baseline_count,
                "checkpoint_sha256": audit["sha256"],
                "groups": summarize_errors(errors, groups),
            }
            arrays = {key: value.numpy() if isinstance(value, torch.Tensor) else value
                      for key, value in data.items()}
            prediction_path = output_dir / f"{name}_predictions.npz"
            np.savez_compressed(prediction_path, **arrays, **errors)
            experiment.link_output(prediction_path, kind="metrics")
            experiment.log({metric: values["mean"] for metric, values in
                            summary["models"][name]["groups"]["all"]["metrics"].items()}, prefix=name)
        write_json(output_dir / "summary.json", summary)
        (output_dir / "comparison.md").write_text(comparison_markdown(summary), encoding="utf-8")
        write_json(output_dir / "effective_config.json", effective)
        for filename in ("summary.json", "comparison.md", "effective_config.json"):
            experiment.link_output(output_dir / filename, kind="metrics")
        experiment.record(dataset_sha256=bundle.identity["dataset_sha256"],
                          frames=len(baseline["target"]), device=str(device))
        print(f"Paired four-model evaluation complete: {output_dir}", flush=True)
        return output_dir
