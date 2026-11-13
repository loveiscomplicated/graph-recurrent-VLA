"""Compare matched completed fixed-ID temporal-to-GNN training runs."""

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write


def comparable_config(config):
    """Remove only the expected temporal source and output path differences."""
    result = deepcopy(config)
    result.pop("temporal")
    result.pop("output")
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
        results = []
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for row in rows:
            name, seed, summary = row["architecture"], row["seed"], row["summary"]
            results.append(dict(architecture=name, seed=seed, checkpoint=str(row["path"]),
                                checkpoint_sha256=file_sha256(row["path"]), **summary))
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
              checkpoint_selection="minimum development validation loss", paired_by_seed=True,
              online_id_matching=False, temporal_frozen=True, reserved_test_evaluated=False))
        logger.link_output(output / "comparison.json")
        logger.link_output(output / "learning_curves.png")
        logger.record(aggregate=aggregate)
        print(f"Comparison: {output}", flush=True)
        return str(output)
