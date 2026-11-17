"""Compare matched completed fixed-ID temporal-to-GNN training runs."""

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import read_manifest
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.fixed_id_gnn import load_temporal, make_graph, predict_frozen
from utils.experiment_logger import ExperimentLogger, write


def comparable_config(config):
    """Remove only the expected temporal source and output path differences."""
    result = deepcopy(config)
    result.pop("temporal")
    result.pop("output")
    return result


def validation_data(config, first_checkpoint, device):
    """Prove frame IDs, ordering and Object21 GT match the leaderboard reference."""
    root = Path(config["corner_dataset"]).resolve()
    manifest = read_manifest(root)
    if manifest["dataset_sha256"] != first_checkpoint["dataset_sha256"]:
        raise ValueError("Evaluation corner dataset hash mismatch")
    frame_ids = []
    for row in manifest["episodes"]:
        if row["split"] != "validation":
            continue
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Evaluation episode hash mismatch: {row['episode']}")
        with np.load(path, allow_pickle=False) as archive:
            frame_ids.extend(archive["sample_ids"].tolist())
    cache_path = Path(first_checkpoint["config"]["data"]["cache"])
    if file_sha256(cache_path) != first_checkpoint["cache_sha256"]:
        raise ValueError("Evaluation frame cache hash mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    reference_path = Path(config["reference_validation"])
    if file_sha256(reference_path) != config["reference_sha256"]:
        raise ValueError("Leaderboard reference hash mismatch")
    with np.load(reference_path, allow_pickle=False) as reference:
        ref_ids, ref_target = reference["sample_ids"], reference["target"]
    val = cache["arrays"]["split"]
    if not np.array_equal(np.asarray(frame_ids), ref_ids):
        raise ValueError("Validation IDs/order differ from leaderboard reference")
    if not np.array_equal(cache["arrays"]["target"][val], ref_target):
        raise ValueError("Validation Object21 labels differ from leaderboard reference")
    arrays = {key: torch.as_tensor(value, device=device) for key, value in cache["arrays"].items()}
    sequences = [item for item in cache["sequences"] if val[item["frame_offset"]]]
    return frame_ids, arrays, sequences


def replay_validation(row, arrays, sequences, device):
    """Replay the frozen temporal checkpoint and the selected trained GNN."""
    checkpoint = row["checkpoint"]
    run_config = checkpoint["config"]
    temporal, _ = load_temporal(run_config, device)
    delta = predict_frozen(temporal, sequences, len(arrays["split"]), device,
                           run_config["training"]["temporal_batch_size"])
    xyz = torch.where(arrays["observed"].unsqueeze(-1), arrays["measurement"], arrays["last"] + delta)
    gnn = VertexEdgeGatedMPNN(**run_config["gnn"]).to(device).eval()
    gnn.load_state_dict(checkpoint["model"], strict=True)
    indices = torch.nonzero(arrays["split"], as_tuple=True)[0]
    prediction, errors = [], {}
    with torch.no_grad():
        batch_size = run_config["training"]["batch_size"]
        for left in range(0, len(indices), batch_size):
            index = indices[left:left + batch_size]
            graph, feature = make_graph(xyz[index], arrays["seen"][index], arrays["observed"][index],
                                        arrays["tcp"][index])
            output = gnn(graph, feature)
            prediction.append(output["object21"].cpu().numpy())
            for name, value in per_frame_errors(output, arrays["target"][index]).items():
                errors.setdefault(name, []).append(value)
    return np.concatenate(prediction), {name: np.concatenate(parts) for name, parts in errors.items()}


def leaderboard_statistics(frame_errors, seeds):
    """Average seed errors per frame before computing median and percentiles."""
    result = {}
    for name in ("mamba", "transformer"):
        sample = frame_errors[(name, seeds[0])]
        result[name] = {}
        for metric in sample:
            values = np.stack([frame_errors[(name, seed)][metric] for seed in seeds])
            result[name][metric] = dict(
                three_seed_frame_mean=statistics(values.mean(axis=0)),
                seed_means=[float(row.mean()) for row in values],
            )
    return result


def compare(config):
    """Verify every paired run and summarize best validation physical errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for seed in config["seeds"]:
        pair = []
        for architecture in ("mamba", "transformer"):
            root = Path(config["runs_root"]) / architecture / f"seed_{seed}"
            checkpoints = list(root.glob("*/best.pt"))
            if len(checkpoints) != 1:
                raise ValueError(f"Expected exactly one completed checkpoint under {root}")
            path = checkpoints[0]
            summary_path = path.parent / "completed.json"
            if not summary_path.exists():
                raise ValueError(f"Run did not complete: {path.parent}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            summary = json.loads(summary_path.read_text())
            history = json.loads((path.parent / "history.json").read_text())
            if (checkpoint["epoch"] != summary["best_epoch"]
                    or checkpoint["config"]["seed"] != seed
                    or checkpoint["config"]["temporal"]["architecture"] != architecture
                    or summary["best_validation_loss"] != min(row["validation"]["loss"] for row in history)):
                raise ValueError(f"Checkpoint/history identity mismatch: {path}")
            pair.append(dict(architecture=architecture, seed=seed, checkpoint=checkpoint,
                             path=path, history=history, summary=summary))
        first, second = pair
        if (comparable_config(first["checkpoint"]["config"]) != comparable_config(second["checkpoint"]["config"])
                or first["checkpoint"]["cache_sha256"] != second["checkpoint"]["cache_sha256"]
                or first["checkpoint"]["gnn_initial_sha256"] != second["checkpoint"]["gnn_initial_sha256"]):
            raise ValueError(f"Paired data, initial GNN weights or settings differ for seed {seed}")
        rows.extend(pair)

    with ExperimentLogger(team="graph", model="fixed_id_gnn_comparison", config=config,
                          seed=0, job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        device = torch.device(config.get("device", "cuda:0"))
        frame_ids, arrays, sequences = validation_data(config, rows[0]["checkpoint"], device)
        results = []
        frame_errors = {}
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for row in rows:
            name, seed, summary = row["architecture"], row["seed"], row["summary"]
            prediction, errors = replay_validation(row, arrays, sequences, device)
            frame_errors[(name, seed)] = errors
            predictions_path = output / f"{name}_seed_{seed}_predictions.npz"
            np.savez_compressed(predictions_path, sample_ids=np.asarray(frame_ids), prediction=prediction,
                                target=arrays["target"][arrays["split"]].cpu().numpy(), **errors)
            logger.link_output(predictions_path)
            results.append(dict(architecture=name, seed=seed, checkpoint=str(row["path"]),
                                checkpoint_sha256=file_sha256(row["path"]),
                                predictions=str(predictions_path), predictions_sha256=file_sha256(predictions_path),
                                **summary))
            axis = axes[0 if name == "mamba" else 1]
            axis.plot([item["epoch"] for item in row["history"]],
                      [item["validation"]["loss"] for item in row["history"]], label=f"seed {seed}")
            axis.scatter([summary["best_epoch"]], [summary["best_validation_loss"]], s=20)
            logger.link_output(row["path"], checkpoint=True)
        aggregate = {}
        keys = rows[0]["summary"]["validation"].keys()
        for name in ("mamba", "transformer"):
            samples = [row["summary"]["validation"] for row in rows if row["architecture"] == name]
            aggregate[name] = {key: dict(mean=float(np.mean([sample[key] for sample in samples])),
                                         std=float(np.std([sample[key] for sample in samples])))
                               for key in keys if key != "frames"}
            aggregate[name]["frames"] = samples[0]["frames"]
            if any(sample["frames"] != samples[0]["frames"] for sample in samples):
                raise ValueError("Validation frame counts differ between seeds")
        if aggregate["mamba"]["frames"] != aggregate["transformer"]["frames"]:
            raise ValueError("Validation frame counts differ between architectures")
        leaderboard_metrics = leaderboard_statistics(frame_errors, config["seeds"])
        for axis, name in zip(axes, ("Mamba", "Transformer")):
            axis.set_title(name)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
            axis.legend()
        axes[0].set_ylabel("Validation Object21 loss")
        figure.tight_layout()
        figure.savefig(output / "learning_curves.png", dpi=150)
        plt.close(figure)
        write(output / "comparison.json", dict(results=results, aggregate=aggregate,
              leaderboard_metrics=leaderboard_metrics, reference_sha256=config["reference_sha256"],
              validation_frames=len(frame_ids), same_ids_and_targets_as_reference=True,
              checkpoint_selection="minimum development validation loss", paired_by_seed=True,
              online_id_matching=False, temporal_frozen=True, reserved_test_evaluated=False))
        logger.link_output(output / "comparison.json")
        logger.link_output(output / "learning_curves.png")
        logger.record(aggregate=aggregate)
        print(f"Comparison: {output}", flush=True)
        return str(output)
