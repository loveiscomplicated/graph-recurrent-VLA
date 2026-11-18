"""Compare frozen GNNs on shared or separately paired held-out tracker graphs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.five_mode_corner_tracker_comparison import load_test, predict
from graph.frozen_tracker_gnn import completed_graph
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.gnn.loss import object21_loss
from utils.experiment_logger import ExperimentLogger, write


def state_digest(model):
    """Hash parameters and buffers to detect any inference-time update."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tracker_settings(config):
    """Pair each GNN with its tracker; default to the shared-input comparison."""
    if set(config["models"]) != {"existing", "retrained"}:
        raise ValueError("Comparison requires existing and retrained GNN settings")
    return {"existing": config.get("existing_tracker", config["tracker"]),
            "retrained": config["tracker"]}


def load_test_targets(episodes, config, source_hash):
    """Verify the held-out label identities, checksums and per-frame TCP."""
    root = Path(config["targets_root"])
    path = root / "manifest.json"
    if file_sha256(path) != config["targets_manifest_sha256"]:
        raise ValueError("Held-out Object21 target manifest checksum mismatch")
    manifest = json.loads(path.read_text())
    if manifest["source_manifest_sha256"] != source_hash or not manifest["test_payloads_read"]:
        raise ValueError("Object21 labels must come from the same held-out release")
    records = {(row["seed"], row["mode"]): row for row in manifest["records"]}
    if len(records) != 450 or len(manifest["records"]) != 450:
        raise ValueError("Expected 450 distinct test target episodes")
    targets = []
    for episode in episodes:
        identity = (episode["seed"], episode["mode"])
        row = records.pop(identity)
        path = root / row["path"]
        if row["split"] != "test" or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Test target split/checksum mismatch: {identity}")
        with np.load(path, allow_pickle=False) as archive:
            target, tcp = archive["object21"], archive["tcp"]
        length = len(episode["tcp"])
        if row["frames"] != length or target.shape != (length, 21):
            raise ValueError(f"Test target frame count mismatch: {identity}")
        if not np.isfinite(target).all() or not np.allclose(tcp, episode["tcp"], rtol=0, atol=1e-6):
            raise ValueError(f"Test target TCP alignment/nonfinite labels: {identity}")
        np.testing.assert_allclose(
            target[:, :3], episode["gt"][:, :8].mean(axis=1) - tcp, rtol=0, atol=2e-6
        )
        targets.append(target.astype(np.float32))
    if records:
        raise ValueError("Unmatched held-out target episodes")
    return np.concatenate(targets)


def summarize_predictions(prediction, residual, target, mode_ids, loss_config):
    """Retain both pooled and equal-mode metrics using the original Object21 loss."""
    for name, values, width in (("prediction", prediction, 21), ("target", target, 21),
                                ("endpoint residual", residual, 3)):
        if values.shape != (len(target), width) or not np.isfinite(values).all():
            raise ValueError(f"Expected finite {name} [frames,{width}]")
    errors = {
        name: np.linalg.norm(
            prediction[:, start:start + 3].astype(np.float64)
            - target[:, start:start + 3].astype(np.float64), axis=1,
        ) * 1000
        for name, start in (("peg_position_mm", 0), ("hole_position_mm", 10),
                            ("tip_minus_hole_mm", 7))
    }

    def score(mask):
        output = {"object21": torch.from_numpy(prediction[mask]),
                  "endpoint_residual": torch.from_numpy(residual[mask])}
        loss = object21_loss(output, torch.from_numpy(target[mask]), **loss_config)
        return {"object21_loss": float(loss["total"]),
                **{name: float(values[mask].mean()) for name, values in errors.items()}}

    by_mode = {mode: score(mode_ids == mode) for mode in sorted(set(mode_ids))}
    balanced = {name: float(np.mean([row[name] for row in by_mode.values()]))
                for name in next(iter(by_mode.values()))}
    return {"mode_balanced": balanced, "pooled": score(np.ones(len(target), dtype=bool)),
            "by_mode": by_mode}, errors


@torch.inference_mode()
def evaluate(config):
    """Freeze all models, infer the configured tracker pairings and score without tuning."""
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(config["device"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[1]
    source = json.loads(Path(config["source_hash_manifest"]).read_text())
    for path, expected in source["files"].items():
        if file_sha256(repository / path) != expected:
            raise ValueError(f"Evaluation source checksum mismatch: {path}")
    checkpoints = {}
    trackers = tracker_settings(config)
    receipts = {f"{name}_tracker": receipt for name, receipt in trackers.items()}
    receipts.update(config["models"])
    for name, receipt in receipts.items():
        if file_sha256(receipt["checkpoint"]) != receipt["sha256"]:
            raise ValueError(f"Frozen checkpoint checksum mismatch: {name}")
        checkpoints[name] = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
    tracker = checkpoints["retrained_tracker"]
    retrained = checkpoints["retrained"]
    development_path = Path(config["training_data_root"]) / "manifest.json"
    development = json.loads(development_path.read_text())
    if (retrained["epoch"] != config["models"]["retrained"]["epoch"]
            or retrained["best_epoch"] != retrained["epoch"]
            or retrained["config"]["tracker"]["sha256"] != config["tracker"]["sha256"]
            or retrained["config"]["cache_manifest"]["identity"]["tracking_manifest_sha256"]
            != file_sha256(development_path)):
        raise ValueError("Retrained GNN does not match the frozen selected training run")
    selected = min(retrained["history"], key=lambda row: row["validation"]["mode_balanced_loss"])
    if selected["epoch"] != retrained["epoch"]:
        raise ValueError("New GNN checkpoint is not the validation minimum")
    if tracker["config"]["training"]["supervision"] != "after_first_observation":
        raise ValueError("Expected the corrected observed-only tracker")
    episodes, manifest = load_test(config["data_root"], config["test_manifest_sha256"])
    if manifest["source_manifest_sha256"] != development["source_manifest_sha256"]:
        raise ValueError("Held-out and training source releases differ")
    if {e["seed"] for e in episodes} & {row["seed"] for row in development["records"]}:
        raise ValueError("Held-out reset seeds overlap training/validation")
    target = load_test_targets(episodes, config, manifest["source_manifest_sha256"])
    models, before = {}, {}
    for name in config["models"]:
        checkpoint = checkpoints[name]
        settings = checkpoint["config"].get("gnn", checkpoint["config"].get("model"))
        if settings != config["model"]:
            raise ValueError(f"GNN graph/model configuration differs: {name}")
        model = VertexEdgeGatedMPNN(**settings).to(device).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["model"], strict=True)
        models[name], before[name] = model, state_digest(model)
    with ExperimentLogger(team="graph", model="frozen_tracker_gnn_comparison",
                          config=config, job_type="evaluate", root=output / "experiments") as logger:
        for receipt in receipts.values():
            logger.link_output(receipt["checkpoint"], checkpoint=True)
        observed = np.concatenate([e["observed"] for e in episodes])
        seen = np.concatenate([np.logical_or.accumulate(e["observed"], axis=0) for e in episodes])
        tcp = np.concatenate([e["tcp"] for e in episodes])
        episode_index = np.repeat(np.arange(len(episodes)), [len(e["tcp"]) for e in episodes])
        mode_ids = np.asarray([episodes[index]["mode"] for index in episode_index])
        inputs_by_hash, input_receipts = {}, {}
        shared_tracker = trackers["existing"]["sha256"] == trackers["retrained"]["sha256"]
        for name, receipt in trackers.items():
            tracker_hash = receipt["sha256"]
            if tracker_hash in inputs_by_hash:
                continue
            print(f"Inferring {name} frozen tracker on 450 held-out episodes", flush=True)
            xyz = predict(episodes, checkpoints[f"{name}_tracker"], config["tracker_batch_size"], device)
            if len(xyz) != manifest["frames"] or len(target) != len(xyz):
                raise ValueError("Test prediction/label counts differ")
            inputs_by_hash[tracker_hash] = {
                key: torch.as_tensor(value, device=device) for key, value in
                dict(xyz=xyz, seen=seen, observed=observed, tcp=tcp).items()
            }
            filename = "tracker_inputs.npz" if shared_tracker else f"{name}_tracker_inputs.npz"
            input_path = output / filename
            np.savez_compressed(input_path, xyz=np.where(seen[..., None], xyz, 0), seen=seen,
                                observed=observed, tcp=tcp, target=target,
                                episode_index=episode_index, mode=mode_ids)
            input_receipts[tracker_hash] = {"path": filename, "sha256": file_sha256(input_path)}
            logger.link_output(input_path)
        predictions = {name: {"object21": [], "endpoint_residual": []} for name in models}
        for start in range(0, len(target), config["gnn_batch_size"]):
            stop = start + config["gnn_batch_size"]
            graphs = {}
            for name, model in models.items():
                tracker_hash = trackers[name]["sha256"]
                if tracker_hash not in graphs:
                    inputs = inputs_by_hash[tracker_hash]
                    # Shared trackers still produce one exact graph for both GNNs.
                    graphs[tracker_hash] = completed_graph(
                        *(inputs[key][start:stop] for key in ("xyz", "seen", "observed", "tcp"))
                    )
                graph, flags = graphs[tracker_hash]
                result = model(graph, flags)
                for key in predictions[name]:
                    predictions[name][key].append(result[key].cpu().numpy())
        summary = {"run_id": logger.run_id, "episodes": len(episodes), "frames": len(target),
                   "primary_aggregation": "equal mean of five mode frame-means",
                   "precision": "strict FP32, TF32 disabled for tracker and both GNNs",
                   "comparison": "shared_tracker" if shared_tracker else "paired_trackers_and_gnns",
                   "tracker_inputs": input_receipts,
                   "tracker_input_sha256": next(iter(input_receipts.values()))["sha256"] if shared_tracker else None,
                   "never_seen_point_frames_excluded": int((~seen).sum()),
                   "models": {}, "config": config, "source_files": source["files"]}
        for name, values in predictions.items():
            if state_digest(models[name]) != before[name]:
                raise RuntimeError(f"GNN state changed during evaluation: {name}")
            arrays = {key: np.concatenate(value) for key, value in values.items()}
            metrics, errors = summarize_predictions(
                arrays["object21"], arrays["endpoint_residual"], target, mode_ids, config["loss"]
            )
            path = output / f"{name}_predictions.npz"
            np.savez_compressed(path, **arrays, target=target, episode_index=episode_index, **errors)
            summary["models"][name] = dict(metrics, checkpoint_epoch=checkpoints[name]["epoch"],
                                           tracker_sha256=trackers[name]["sha256"],
                                           predictions_sha256=file_sha256(path))
            logger.link_output(path)
            logger.log(metrics, step=0, prefix=f"test/{name}")
        for name, receipt in receipts.items():
            if file_sha256(receipt["checkpoint"]) != receipt["sha256"]:
                raise RuntimeError(f"Frozen checkpoint changed: {name}")
        write(output / "results.json", summary)
        logger.link_output(output / "results.json")
        logger.record(episodes=len(episodes), frames=len(target), identical_gnn_input=shared_tracker,
                      device=str(device), torch=str(torch.__version__),
                      gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
        print(json.dumps({name: row["mode_balanced"] for name, row in summary["models"].items()}, indent=2), flush=True)
