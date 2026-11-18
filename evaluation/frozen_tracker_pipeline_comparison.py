"""Compare the trained corner-only and surface-context pipelines on one test set."""

import gc
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_surface_context import attach_surface_context
from evaluation.five_mode_corner_tracker_comparison import load_test, predict, summarize
from evaluation.frozen_surface_tracker_gnn import load_targets, position_errors_mm
from evaluation.observed_corner_surface_comparison import predict_surface
from graph.frozen_tracker_gnn import completed_graph
from graph.frozen_tracker_graph import build_tracker_graph
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.frozen_surface_tracker_gnn import state_digest
from utils.experiment_logger import ExperimentLogger, write


def load_pipeline(name, settings, development_hash):
    """Check frozen weights and validation selection while retaining each trained schema."""
    payloads = {}
    for component in ("tracker", "gnn"):
        receipt = settings[component]
        if file_sha256(receipt["path"]) != receipt["sha256"]:
            raise ValueError(f"{name} {component} checkpoint checksum differs")
        payloads[component] = torch.load(receipt["path"], map_location="cpu", weights_only=False)
    tracker, gnn = payloads["tracker"], payloads["gnn"]
    if (tracker["config"]["training"]["supervision"] != "after_first_observation"
            or tracker["config"]["dataset_manifest_sha256"] != development_hash):
        raise ValueError(f"{name} tracker supervision or development identity differs")
    if name == "corner_only":
        selected = min(gnn["history"], key=lambda row: row["validation"]["mode_balanced_loss"])
        if (gnn["epoch"] != selected["epoch"] or gnn["epoch"] != gnn["best_epoch"]
                or gnn["config"]["tracker"]["sha256"] != settings["tracker"]["sha256"]
                or gnn["config"]["cache_manifest"]["identity"]["tracking_manifest_sha256"] != development_hash):
            raise ValueError("Corner-only GNN selection or training input differs")
    elif name == "surface_context":
        history = json.loads(Path(settings["gnn"]["history"]).read_text())
        selected = min(history, key=lambda row: row["validation"]["loss"])
        if (selected["validation"]["loss"] != gnn["validation_loss"]
                or gnn["dataset_sha256"] != development_hash
                or gnn["config"]["data"]["tracker_sha256"] != settings["tracker"]["sha256"]):
            raise ValueError("Surface GNN selection or training input differs")
    else:
        raise ValueError(f"Unknown pipeline {name}")
    if selected["epoch"] != settings["gnn"]["epoch"]:
        raise ValueError(f"{name} selected epoch differs from the pinned recipe")
    return payloads


@torch.inference_mode()
def infer_pipeline(name, episodes, payloads, settings, config, device, return_residual=False):
    """Use each pipeline's original graph features with common numerical execution."""
    predictor = predict if name == "corner_only" else predict_surface
    xyz = predictor(episodes, payloads["tracker"], config["tracker_batch_size"], device)
    observed = np.concatenate([episode["observed"] for episode in episodes])
    seen = np.concatenate([np.logical_or.accumulate(episode["observed"], axis=0) for episode in episodes])
    detected = np.concatenate([episode["detected"] for episode in episodes])
    if not np.isfinite(xyz).all() or not np.array_equal(xyz[observed], detected[observed]):
        raise ValueError(f"{name} tracker failed finite/measurement pass-through checks")
    tcp = np.concatenate([episode["tcp"] for episode in episodes])
    inputs = {key: torch.as_tensor(value, device=device) for key, value in
              dict(xyz=xyz, observed=observed, seen=seen, tcp=tcp).items()}
    model_settings = dict(payloads["gnn"]["config"]["model"])
    for key in ("factory", "name"):
        model_settings.pop(key, None)
    model = VertexEdgeGatedMPNN(**model_settings).to(device).eval().requires_grad_(False)
    model.load_state_dict(payloads["gnn"]["model"], strict=True)
    before = state_digest(model)
    prediction = []
    residual = []
    for start in range(0, len(xyz), config["gnn_batch_size"]):
        segment = slice(start, start + config["gnn_batch_size"])
        batch = {key: value[segment] for key, value in inputs.items()}
        if name == "corner_only":
            graph, flags = completed_graph(batch["xyz"], batch["seen"], batch["observed"], batch["tcp"])
        else:
            graph, flags = build_tracker_graph(batch["xyz"], batch["observed"], batch["seen"], batch["tcp"])
        prediction_output = model(graph, flags)
        prediction.append(prediction_output["object21"].cpu().numpy())
        if return_residual:
            residual.append(prediction_output["endpoint_residual"].cpu().numpy())
    if state_digest(model) != before:
        raise RuntimeError(f"{name} GNN state changed during inference")
    for component in ("tracker", "gnn"):
        if file_sha256(settings[component]["path"]) != settings[component]["sha256"]:
            raise RuntimeError(f"{name} {component} checkpoint changed during inference")
    result = (np.concatenate(prediction), int((~seen).sum()), model_settings)
    if return_residual:
        return (*result, np.concatenate(residual))
    return result


def evaluate(config):
    """Run both frozen pipelines against exactly aligned test targets without tuning."""
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(config["device"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    development_path = Path(config["training_data_root"]) / "manifest.json"
    development = json.loads(development_path.read_text())
    development_hash = file_sha256(development_path)
    episodes, manifest = load_test(config["data_root"], config["test_manifest_sha256"])
    if (manifest["source_manifest_sha256"] != development["source_manifest_sha256"]
            or {e["seed"] for e in episodes} & {r["seed"] for r in development["records"]}):
        raise ValueError("Test source release differs or test seeds overlap development")
    if file_sha256(Path(config["surface_context"]) / "manifest.json") != config["surface_manifest_sha256"]:
        raise ValueError("Test surface manifest checksum differs")
    attach_surface_context({"test": episodes}, config["surface_context"], config["data_root"])
    target = load_targets(episodes, config["target_root"], config["target_manifest_sha256"],
                          manifest["source_manifest_sha256"])
    if len(target) != manifest["frames"]:
        raise ValueError("Test target count differs from manifest")
    episode_index = np.repeat(np.arange(len(episodes)), [len(episode["tcp"]) for episode in episodes])
    modes = np.asarray([episodes[index]["mode"] for index in episode_index])
    with ExperimentLogger(team="graph", model="frozen_tracker_pipeline_comparison", job_type="evaluate",
                          config=config, root=output / "experiments") as logger:
        result = dict(run_id=logger.run_id, episodes=len(episodes), frames=len(target), config=config,
                      precision="strict FP32; TF32 disabled for both trackers and GNNs",
                      aggregation="all test frames pooled; 3D Euclidean mm; linear P90", models={},
                      interpretation="as-trained pipeline comparison; graph features and training settings also differ")
        for name, settings in config["models"].items():
            print(f"Evaluating frozen pipeline: {name}", flush=True)
            for receipt in settings.values():
                logger.link_output(receipt["path"], checkpoint=True)
            payloads = load_pipeline(name, settings, development_hash)
            prediction, excluded, model_settings, residual = infer_pipeline(
                name, episodes, payloads, settings, config, device, return_residual=True
            )
            errors = position_errors_mm(prediction, target)
            metrics = {metric: summarize(values) for metric, values in errors.items()}
            path = output / f"{name}_predictions.npz"
            np.savez_compressed(path, object21=prediction, endpoint_residual=residual, target=target,
                                episode_index=episode_index, mode=modes, **errors)
            result["models"][name] = dict(metrics=metrics, epoch=settings["gnn"]["epoch"],
                model_settings=model_settings, never_seen_node_frames_excluded=excluded,
                predictions_sha256=file_sha256(path),
                by_mode={mode: {metric: summarize(values[modes == mode]) for metric, values in errors.items()}
                         for mode in sorted(set(modes))})
            logger.link_output(path)
            logger.log(metrics, step=0, prefix=f"test/{name}")
            del payloads, prediction, errors
            gc.collect()
            torch.cuda.empty_cache()
        result["reduction_percent"] = {}
        for metric, baseline in result["models"]["corner_only"]["metrics"].items():
            variant = result["models"]["surface_context"]["metrics"][metric]
            result["reduction_percent"][metric] = {
                key: (baseline[key] - variant[key]) / baseline[key] * 100 for key in ("mean_mm", "p90_mm")
            }
        write(output / "results.json", result)
        logger.link_output(output / "results.json")
        logger.record(dataset_sha256=config["test_manifest_sha256"], device=str(device),
                      gpu=torch.cuda.get_device_name(device), episodes=len(episodes), frames=len(target),
                      precision=result["precision"], model_selection="validation only")
        print(json.dumps({name: value["metrics"] for name, value in result["models"].items()}, indent=2), flush=True)
