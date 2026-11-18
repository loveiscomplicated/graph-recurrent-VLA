"""Evaluate visibility-based routing on the frozen supervised Five-mode release."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.storage import write_json
from training.gnn.packed_five_mode_data import PackedFrameBatches, read_splits
from training.gnn.trainer import build_model
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


CONDITIONS = ("condition_1", "condition_2")
ENDPOINT = slice(7, 10)


def pinned_path(settings):
    """Resolve an input only when its bytes match the recorded identity."""
    path = Path(settings["path"]).resolve()
    if file_sha256(path) != settings["sha256"]:
        raise ValueError(f"Input checksum does not match the evaluation config: {path}")
    return path


def load_model(settings, device):
    """Load the current weights from either a best or a resume checkpoint."""
    checkpoint = torch.load(pinned_path(settings), map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"]["model"], device)
    # Resume checkpoints store the last epoch in model and the selected weights in best_model.
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def load_frames(path, manifest, index, split, condition, settings, device):
    """Keep the original frame IDs and the leaderboard loader's packing order."""
    episodes = read_splits(
        path, manifest, (split,), condition, True, condition_index=index,
    )[split]
    loader = PackedFrameBatches(episodes, device, settings["batch_size"][split], None)
    cohort = settings["cohorts"][split][condition]
    if len(loader.ids) != cohort["frames"] or len(set(loader.ids)) != len(loader.ids):
        raise ValueError(f"Unexpected frame count or duplicate frame IDs in {condition}/{split}")
    if "frame_ids_sha256" in cohort and canonical_hash(loader.ids) != cohort["frame_ids_sha256"]:
        raise ValueError(f"Frame IDs or order differ from the reference for {condition}/{split}")
    return loader


@torch.inference_mode()
def predict_endpoints(loader, models):
    """Run all models on the same batches and return aligned Hole-relative vectors."""
    predictions = {name: [] for name in models}
    for batch in loader:
        for name, model in models.items():
            predicted = model(batch.graph)["object21"][:, ENDPOINT]
            predictions[name].append(predicted.cpu().numpy())
    arrays = {name: np.concatenate(batches) for name, batches in predictions.items()}
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("The GNN returned a nonfinite relative endpoint")
    return arrays


def summarize_mm(values):
    """Summarize a nonempty array already expressed in millimetres."""
    return {
        "count": len(values),
        "mean_mm": float(np.mean(values, dtype=np.float64)),
        "p95_mm": float(np.quantile(values, 0.95, method="linear")),
    }


def endpoint_errors_mm(predicted, target):
    return np.linalg.norm(predicted - target, axis=1) * 1000


def adjacent_rows(frame_ids):
    """Join only consecutive frame numbers within the same seed and policy mode."""
    rows = {frame_id: row for row, frame_id in enumerate(frame_ids)}
    if len(rows) != len(frame_ids):
        raise ValueError("Frame IDs must be unique across the two test conditions")
    previous, current = [], []
    for row, frame_id in enumerate(frame_ids):
        # Example: 8644707:Fixed_Hole_expert:69 identifies one episode and frame.
        episode, _, frame = frame_id.rpartition(":")
        previous_row = rows.get(f"{episode}:{int(frame) - 1}")
        if previous_row is not None:
            previous.append(previous_row)
            current.append(row)
    return np.asarray(previous, dtype=np.int64), np.asarray(current, dtype=np.int64)


def summarize_transitions(frame_ids, conditions, target, baseline, routed):
    """Compare GT-adjusted jumps against the fixed checkpoint on identical pairs."""
    previous, current = adjacent_rows(frame_ids)
    gt_delta = target[current] - target[previous]
    baseline_delta = baseline[current] - baseline[previous]
    routed_delta = routed[current] - routed[previous]
    baseline_jump = endpoint_errors_mm(baseline_delta, gt_delta)
    routed_jump = endpoint_errors_mm(routed_delta, gt_delta)
    routing_change = endpoint_errors_mm(routed_delta, baseline_delta)
    from_condition, to_condition = conditions[previous], conditions[current]
    groups = {"switch": from_condition != to_condition}
    for start in CONDITIONS:
        for stop in CONDITIONS:
            groups[f"{start}_to_{stop}"] = (from_condition == start) & (to_condition == stop)
    report = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        fixed, switched = baseline_jump[mask], routed_jump[mask]
        report[name] = {
            "baseline": summarize_mm(fixed),
            "routed": summarize_mm(switched),
            "routing_change": summarize_mm(routing_change[mask]),
            "mean_change_mm": float(np.mean(switched - fixed, dtype=np.float64)),
            "baseline_over_10_mm_fraction": float(np.mean(fixed > 10)),
            "routed_over_10_mm_fraction": float(np.mean(switched > 10)),
        }
    return report


def evaluate(config):
    """Select on condition_2 validation, then evaluate once on held-out episodes."""
    manifest_path = pinned_path(config["manifest"])
    manifest = json.loads(manifest_path.read_text())
    index = json.loads(pinned_path(config["condition_index"]).read_text())
    device = set_device(config["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision(config["matmul_precision"])
    baseline_spec = config["checkpoints"]["condition_1"]
    candidates = config["checkpoints"]["condition_2_candidates"]
    baseline = load_model(baseline_spec, device)
    models = {name: load_model(settings, device) for name, settings in candidates.items()}
    with ExperimentLogger(
        team="graph", model="vertex_edge_gated_mpnn", config=config, job_type="evaluate",
    ) as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.record(
            dataset_sha256=config["manifest"]["sha256"],
            condition_index_sha256=config["condition_index"]["sha256"],
            device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
            torch_version=str(torch.__version__),
            evaluator_sha256=file_sha256(Path(__file__)),
            output_directory=str(output.resolve()),
        )
        for settings in (baseline_spec, *candidates.values()):
            logger.link_output(settings["path"], checkpoint=True)

        validation = load_frames(
            manifest_path, manifest, index, "validation", "condition_2", config, device,
        )
        target = validation.target[:, ENDPOINT].cpu().numpy()
        val_predictions = predict_endpoints(validation, models)
        validation_metrics = {
            name: summarize_mm(endpoint_errors_mm(predictions, target))
            for name, predictions in val_predictions.items()
        }
        winner = min(validation_metrics, key=lambda name: validation_metrics[name]["mean_mm"])
        selection = {
            "condition_1": baseline_spec,
            "condition_2": candidates[winner],
            "selected_candidate": winner,
            "criterion": "minimum condition_2 validation mean endpoint error",
            "validation_frame_ids_sha256": canonical_hash(validation.ids),
            "validation_metrics": validation_metrics,
        }
        selection_path = output / "selection.json"
        write_json(selection_path, selection)
        logger.link_output(selection_path, kind="metrics")
        logger.log(validation_metrics, step=1, prefix="validation/condition_2")
        print(f"Selected {winner} from validation: {validation_metrics}", flush=True)
        del validation

        frame_ids, condition_labels, targets, baseline_predictions, routed_predictions = [], [], [], [], []
        test = {}
        for condition in CONDITIONS:
            loader = load_frames(manifest_path, manifest, index, "test", condition, config, device)
            test_models = {"baseline": baseline}
            if condition == "condition_2":
                test_models["routed"] = models[winner]
            predictions = predict_endpoints(loader, test_models)
            fixed = predictions["baseline"]
            routed = predictions.get("routed", fixed)
            target = loader.target[:, ENDPOINT].cpu().numpy()
            test[condition] = {
                "baseline": summarize_mm(endpoint_errors_mm(fixed, target)),
                "routed": summarize_mm(endpoint_errors_mm(routed, target)),
            }
            frame_ids.extend(loader.ids)
            condition_labels.extend([condition] * len(loader.ids))
            targets.append(target)
            baseline_predictions.append(fixed)
            routed_predictions.append(routed)
            print(f"Evaluated {condition}/test: {test[condition]}", flush=True)

        labels = np.asarray(condition_labels)
        target = np.concatenate(targets)
        fixed = np.concatenate(baseline_predictions)
        routed = np.concatenate(routed_predictions)
        test["pooled"] = {
            "baseline": summarize_mm(endpoint_errors_mm(fixed, target)),
            "routed": summarize_mm(endpoint_errors_mm(routed, target)),
        }
        transitions = summarize_transitions(frame_ids, labels, target, fixed, routed)
        trace_path = output / "test_endpoint_predictions.npz"
        np.savez_compressed(
            trace_path, frame_ids=np.asarray(frame_ids), conditions=labels,
            target_m=target, baseline_m=fixed, routed_m=routed,
        )
        logger.link_output(trace_path)
        result_path = output / "results.json"
        write_json(result_path, {
            "run_id": logger.run_id, "config": config, "selection": selection,
            "test_endpoint_error": test, "test_gt_adjusted_endpoint_jumps": transitions,
        })
        logger.link_output(result_path, kind="metrics")
        logger.log(test, step=2, prefix="test/endpoint_error")
        logger.log(transitions, step=3, prefix="test/gt_adjusted_jump")
        print(f"Completed {result_path}", flush=True)
