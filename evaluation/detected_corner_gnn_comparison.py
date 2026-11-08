"""Score six detector-only rollout GNNs against the 21D leaderboard frames."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.detected_corner_gnn import graph_batch
from utils.experiment_logger import ExperimentLogger, write


def comparable_config(config):
    """Ignore per-model source paths/hashes while retaining shared settings."""
    value = deepcopy(config)
    for name in ("temporal", "source_validation", "output", "resume", "rollout_source"):
        value.pop(name, None)
    return value


def load_run(root, architecture, seed):
    paths = list((Path(root) / architecture / f"seed_{seed}").glob("*/best.pt"))
    if len(paths) != 1:
        raise ValueError(f"Expected one trained checkpoint for {architecture} seed {seed}")
    path = paths[0]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    history = json.loads((path.parent / "history.json").read_text())
    summary = json.loads((path.parent / "completed.json").read_text())
    if (checkpoint["config"]["seed"] != seed
            or checkpoint["config"]["temporal"]["architecture"] != architecture
            or checkpoint["epoch"] != summary["best_epoch"]
            or (summary["epochs"] != checkpoint["config"]["training"]["epochs"]
                and not (summary.get("early_stopped")
                         and summary["epochs"] < checkpoint["config"]["training"]["epochs"]))
            or summary["best_validation_loss"] != min(item["validation"]["loss"] for item in history)):
        raise ValueError(f"Incomplete or inconsistent trained run: {path}")
    cache_path = path.parent / "rollout_inputs.pt"
    if file_sha256(cache_path) != checkpoint["rollout_cache_sha256"]:
        raise ValueError(f"Frozen rollout input checksum mismatch: {cache_path}")
    return dict(path=path, checkpoint=checkpoint, history=history, summary=summary, cache_path=cache_path)


def evaluate_run(run, reference_ids, reference_target, device):
    """Evaluate the selected GNN against exact, aligned leaderboard frames."""
    cache = torch.load(run["cache_path"], map_location="cpu", weights_only=False)
    arrays = cache["arrays"]
    validation = arrays["split"]
    if (not np.array_equal(arrays["sample_ids"][validation], reference_ids)
            or not np.array_equal(arrays["target"][validation], reference_target)):
        raise ValueError("Validation frame IDs/order or Object21 GT differ from leaderboard reference")
    values = {key: torch.as_tensor(value[validation], device=device) for key, value in arrays.items()
              if key not in ("sample_ids", "split")}
    model = VertexEdgeGatedMPNN(**run["checkpoint"]["config"]["gnn"]).to(device).eval()
    model.load_state_dict(run["checkpoint"]["model"], strict=True)
    prediction, errors = [], {}
    batch_size = run["checkpoint"]["config"]["training"]["batch_size"]
    with torch.no_grad():
        for left in range(0, len(reference_ids), batch_size):
            sl = slice(left, left + batch_size)
            graph, features = graph_batch(values["xyz"][sl], values["kinds"][sl], values["valid"][sl],
                                           values["observed"][sl], values["tcp"][sl])
            output = model(graph, features)
            prediction.append(output["object21"].cpu().numpy())
            for name, array in per_frame_errors(output, values["target"][sl]).items():
                errors.setdefault(name, []).append(array)
    return np.concatenate(prediction), {name: np.concatenate(parts) for name, parts in errors.items()}


def compare(config):
    """Enforce paired settings and save per-frame and 3-seed 21D scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = Path(config["reference_validation"])
    if file_sha256(reference) != config["reference_sha256"]:
        raise ValueError("Leaderboard reference checksum mismatch")
    with np.load(reference, allow_pickle=False) as archive:
        reference_ids, reference_target = archive["sample_ids"].copy(), archive["target"].copy()
    runs = {}
    for seed in (0, 1, 2):
        pair = [load_run(config["runs_root"], architecture, seed) for architecture in ("mamba", "transformer")]
        left, right = (run["checkpoint"] for run in pair)
        left_identity = {key: value for key, value in left["identity"].items() if key != "max_nodes"}
        right_identity = {key: value for key, value in right["identity"].items() if key != "max_nodes"}
        if (comparable_config(left["config"]) != comparable_config(right["config"])
                or left["gnn_initial_sha256"] != right["gnn_initial_sha256"]
                or left_identity != right_identity):
            raise ValueError(f"Paired GNN initialization, data or settings differ for seed {seed}")
        for name, run in zip(("mamba", "transformer"), pair):
            runs[(name, seed)] = run
    with ExperimentLogger(team="graph", model="detected_corner_gnn_comparison", config=config,
                          seed=0, job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        device = torch.device(config["device"])
        errors_by_run = {}
        results = []
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for name in ("mamba", "transformer"):
            for seed in (0, 1, 2):
                run = runs[(name, seed)]
                prediction, errors = evaluate_run(run, reference_ids, reference_target, device)
                errors_by_run[(name, seed)] = errors
                prediction_path = output / f"{name}_seed_{seed}_predictions.npz"
                np.savez_compressed(prediction_path, sample_ids=reference_ids, prediction=prediction,
                                    target=reference_target, **errors)
                logger.link_output(prediction_path)
                logger.link_output(run["path"], checkpoint=True)
                results.append(dict(architecture=name, seed=seed, checkpoint=str(run["path"]),
                                    checkpoint_sha256=file_sha256(run["path"]),
                                    prediction_sha256=file_sha256(prediction_path), **run["summary"]))
                axis = axes[0 if name == "mamba" else 1]
                axis.plot([entry["epoch"] for entry in run["history"]],
                          [entry["validation"]["loss"] for entry in run["history"]], label=f"seed {seed}")
                print(f"Evaluated Object21 {name} seed {seed}: {len(reference_ids)} frames", flush=True)
        metrics = {}
        for name in ("mamba", "transformer"):
            metrics[name] = {}
            for metric in errors_by_run[(name, 0)]:
                values = np.stack([errors_by_run[(name, seed)][metric] for seed in (0, 1, 2)])
                metrics[name][metric] = dict(three_seed_frame_mean=statistics(values.mean(axis=0)),
                                             seed_means=[float(value.mean()) for value in values])
        for axis, name in zip(axes, ("Mamba", "Transformer")):
            axis.set_title(name)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=.2)
            axis.legend()
        axes[0].set_ylabel("Validation Object21 loss")
        figure.tight_layout()
        figure.savefig(output / "learning_curves.png", dpi=150)
        plt.close(figure)
        write(output / "comparison.json", dict(results=results, metrics=metrics,
              validation_frames=len(reference_ids), reference_sha256=config["reference_sha256"],
              same_frame_ids_and_gt_as_reference=True, temporal_frozen=True,
              gt_in_tracking=False, natural_visibility=True,
              checkpoint_selection="minimum development-validation loss",
              seed_aggregation="mean errors per frame, then frame statistics"))
        logger.link_output(output / "comparison.json")
        logger.link_output(output / "learning_curves.png")
        logger.record(validation_frames=len(reference_ids), metrics=metrics)
        print(f"21D comparison: {output}", flush=True)
        return str(output)
