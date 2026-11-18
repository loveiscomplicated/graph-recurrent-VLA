"""Evaluate a validation-selected GNN on the fixed 450-episode test release."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_surface_context import attach_surface_context
from evaluation.five_mode_corner_tracker_comparison import load_test, summarize
from evaluation.observed_corner_surface_comparison import predict_surface
from graph.frozen_tracker_graph import build_tracker_graph
from training.frozen_surface_tracker_gnn import state_digest
from training.gnn.trainer import build_model
from utils.experiment_logger import ExperimentLogger, write


def position_errors_mm(prediction, target):
    """Measure per-frame Euclidean errors, including the complete tip−Hole vector."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (prediction.ndim != 2 or prediction.shape[1] != 21 or prediction.shape != target.shape
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError("Expected matching finite Object21 arrays [frames,21]")
    return {
        name: np.linalg.norm(prediction[:, start:start + 3] - target[:, start:start + 3], axis=1) * 1000
        for name, start in (("peg_position", 0), ("hole_position", 10), ("tip_minus_hole", 7))
    }


def load_targets(episodes, root, expected_hash, source_hash):
    """Verify held-out target receipts, episode membership and per-frame TCP."""
    root = Path(root)
    path = root / "manifest.json"
    if file_sha256(path) != expected_hash:
        raise ValueError("Test Object21 target manifest checksum differs")
    manifest = json.loads(path.read_text())
    if manifest["source_manifest_sha256"] != source_hash or not manifest["test_payloads_read"]:
        raise ValueError("Expected labels from the same held-out release")
    records = {(row["seed"], row["mode"]): row for row in manifest["records"]}
    if len(records) != 450 or len(manifest["records"]) != 450:
        raise ValueError("Test target manifest must contain 450 distinct episodes")
    targets = []
    for episode in episodes:
        row = records.pop((episode["seed"], episode["mode"]))
        path = root / row["path"]
        if row["split"] != "test" or row["frames"] != len(episode["tcp"]) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Test target episode identity/checksum differs: {path}")
        with np.load(path, allow_pickle=False) as archive:
            target, tcp = archive["object21"], archive["tcp"]
        if target.shape != (len(tcp), 21) or not np.allclose(tcp, episode["tcp"], atol=1e-6, rtol=0):
            raise ValueError(f"Test target frame/TCP alignment differs: {path}")
        targets.append(target)
    if records:
        raise ValueError("Unused test target episodes")
    return np.concatenate(targets)


def evaluate(config):
    """Freeze both selected models; evaluate all test frames without tuning."""
    device = torch.device(config["device"])
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_paths = {name: Path(config[name]["path"]) for name in ("gnn", "tracker")}
    for name, path in checkpoint_paths.items():
        if file_sha256(path) != config[name]["sha256"]:
            raise ValueError(f"Frozen {name} checkpoint checksum differs")
    gnn_checkpoint = torch.load(checkpoint_paths["gnn"], map_location="cpu", weights_only=False)
    tracker_checkpoint = torch.load(checkpoint_paths["tracker"], map_location="cpu", weights_only=False)
    history = json.loads(Path(config["gnn"]["history"]).read_text())
    selected = min(history, key=lambda row: row["validation"]["loss"])
    if (selected["epoch"] != config["gnn"]["epoch"]
            or selected["validation"]["loss"] != gnn_checkpoint["validation_loss"]):
        raise ValueError("GNN checkpoint does not match the validation-selected epoch")
    development_path = Path(config["training_data_root"]) / "manifest.json"
    development_hash = file_sha256(development_path)
    if (gnn_checkpoint["dataset_sha256"] != development_hash
            or tracker_checkpoint["config"]["dataset_manifest_sha256"] != development_hash
            or gnn_checkpoint["config"]["data"]["tracker_sha256"] != config["tracker"]["sha256"]):
        raise ValueError("GNN and tracker training input identities differ")
    if tracker_checkpoint["config"]["training"]["supervision"] != "after_first_observation":
        raise ValueError("Expected observed-only surface tracker")
    episodes, manifest = load_test(config["data_root"], config["test_manifest_sha256"])
    development = json.loads(development_path.read_text())
    if ({episode["seed"] for episode in episodes}
            & {row["seed"] for row in development["records"]}):
        raise ValueError("Test seeds overlap training/validation")
    if manifest["source_manifest_sha256"] != development["source_manifest_sha256"]:
        raise ValueError("Test and development source releases differ")
    surface_root = Path(config["surface_context"])
    if file_sha256(surface_root / "manifest.json") != config["surface_manifest_sha256"]:
        raise ValueError("Test surface manifest checksum differs")
    attach_surface_context({"test": episodes}, surface_root, config["data_root"])
    target = load_targets(episodes, config["target_root"], config["target_manifest_sha256"],
                          manifest["source_manifest_sha256"])
    with ExperimentLogger(team="graph", model="frozen_surface_tracker_gnn", job_type="evaluate",
                          config=config, root=output / "experiments") as logger:
        for path in checkpoint_paths.values():
            logger.link_output(path, checkpoint=True)
        tracker_before = {key: value.clone() for key, value in tracker_checkpoint["model"].items()}
        xyz = predict_surface(episodes, tracker_checkpoint, config["tracker_batch_size"], device)
        if any(not torch.equal(value, tracker_before[key]) for key, value in tracker_checkpoint["model"].items()):
            raise RuntimeError("Tracker state changed during evaluation")
        observed = np.concatenate([episode["observed"] for episode in episodes])
        valid = np.concatenate([np.logical_or.accumulate(episode["observed"], axis=0) for episode in episodes])
        detected = np.concatenate([episode["detected"] for episode in episodes])
        if not np.array_equal(xyz[observed], detected[observed]) or not np.isfinite(xyz).all():
            raise ValueError("Tracker measurement pass-through/finite check failed")
        tcp = np.concatenate([episode["tcp"] for episode in episodes])
        arrays = {key: torch.as_tensor(value, device=device) for key, value in
                  dict(xyz=xyz, observed=observed, valid=valid, tcp=tcp).items()}
        model = build_model(gnn_checkpoint["config"]["model"], device).eval().requires_grad_(False)
        model.load_state_dict(gnn_checkpoint["model"], strict=True)
        before = state_digest(model)
        # Match the GNN's validation precision; tracker rollout above stays FP32.
        torch.backends.cuda.matmul.allow_tf32 = True
        prediction = []
        with torch.inference_mode():
            for start in range(0, len(xyz), config["gnn_batch_size"]):
                stop = start + config["gnn_batch_size"]
                graph, flags = build_tracker_graph(*(arrays[key][start:stop] for key in ("xyz", "observed", "valid", "tcp")))
                prediction.append(model(graph, flags)["object21"].cpu().numpy())
        prediction = np.concatenate(prediction)
        if state_digest(model) != before:
            raise RuntimeError("GNN state changed during evaluation")
        if len(prediction) != manifest["frames"] or len(target) != len(prediction):
            raise ValueError("Test prediction/label/frame counts differ")
        errors = position_errors_mm(prediction, target)
        episode_index = np.repeat(np.arange(len(episodes)), [len(e["tcp"]) for e in episodes])
        result = dict(run_id=logger.run_id, episodes=len(episodes), frames=len(prediction),
                      selected_epoch=selected["epoch"], best_validation_loss=gnn_checkpoint["validation_loss"],
                      training_completed_epoch=history[-1]["epoch"],
                      aggregation="all test frames pooled; Euclidean 3D error in mm; NumPy linear P90",
                      metrics={name: summarize(values) for name, values in errors.items()},
                      never_observed_node_frames_excluded=int((~valid).sum()), config=config)
        result["by_mode"] = {}
        for mode in sorted({episode["mode"] for episode in episodes}):
            mask = np.asarray([episodes[index]["mode"] == mode for index in episode_index])
            result["by_mode"][mode] = {name: summarize(values[mask]) for name, values in errors.items()}
        prediction_path = output / "predictions.npz"
        np.savez_compressed(prediction_path, object21=prediction, target=target,
                            episode_index=episode_index, **errors)
        result["predictions_sha256"] = file_sha256(prediction_path)
        write(output / "results.json", result)
        logger.link_output(prediction_path)
        logger.link_output(output / "results.json")
        logger.record(dataset_sha256=config["test_manifest_sha256"], device=str(device),
                      gpu=torch.cuda.get_device_name(device), selected_epoch=selected["epoch"],
                      episodes=len(episodes), frames=len(prediction), tracker_frozen=True,
                      precision="tracker FP32 TF32 disabled; GNN FP32 TF32 enabled, matching validation")
        logger.log(result["metrics"], step=0, prefix="test")
        print(json.dumps({key: result[key] for key in ("episodes", "frames", "selected_epoch", "metrics")}, indent=2), flush=True)
