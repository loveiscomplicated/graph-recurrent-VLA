"""Evaluate pinned Peg-surface checkpoints on fixed Five-mode test cohorts."""

import json
from pathlib import Path

import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.storage import write_json
from training.gnn.packed_five_mode_data import PackedFrameBatches, read_splits
from training.gnn.trainer import build_model, import_callable, run_epoch
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def pinned_path(settings, label):
    """Resolve a configured input only when its bytes match the pinned hash."""
    path = Path(settings["path"]).resolve()
    if file_sha256(path) != settings["sha256"]:
        raise ValueError(f"{label} checksum does not match the evaluation config: {path}")
    return path


def load_checkpoint_model(settings, device):
    """Build one model and task from a best-weights or resumable checkpoint."""
    path = pinned_path(settings, "Checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_key = settings.get("state_key", "model")
    if state_key not in checkpoint:
        raise ValueError(f"Checkpoint does not contain state_key={state_key!r}: {path}")

    checkpoint_config = checkpoint["config"]
    model = build_model(checkpoint_config["model"], device)
    model.load_state_dict(checkpoint[state_key], strict=True)
    task_settings = checkpoint_config["task"]
    task = import_callable(task_settings["factory"])(**task_settings["settings"])
    return path, checkpoint, model, task


def load_test_cohort(manifest_path, manifest, condition_index, condition, config, device):
    """Pack one fixed test cohort and verify its ordered frame identities."""
    episodes = read_splits(
        manifest_path,
        manifest,
        ("test",),
        condition,
        include_surface=True,
        condition_index=condition_index,
    )["test"]
    loader = PackedFrameBatches(episodes, device, config["batch_size"], None)
    expected = config["cohorts"][condition]
    if len(loader.ids) != expected["frames"] or len(set(loader.ids)) != len(loader.ids):
        raise ValueError(f"Unexpected frame count or duplicate IDs in {condition}/test")
    if canonical_hash(loader.ids) != expected["frame_ids_sha256"]:
        raise ValueError(f"Frame IDs or order differ from the reference for {condition}/test")
    return loader


def evaluate(config):
    """Measure every checkpoint on both test cohorts without selecting on test."""
    manifest_path = pinned_path(config["manifest"], "Manifest")
    index_path = pinned_path(config["condition_index"], "Condition index")
    manifest = json.loads(manifest_path.read_text())
    condition_index = json.loads(index_path.read_text())

    device = set_device(config["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision(config["matmul_precision"])

    checkpoint_inputs = {
        name: load_checkpoint_model(settings, device)
        for name, settings in config["checkpoints"].items()
    }
    with ExperimentLogger(
        team="graph",
        model="vertex_edge_gated_mpnn",
        config=config,
        job_type="evaluate",
    ) as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.record(
            dataset_sha256=config["manifest"]["sha256"],
            condition_index_sha256=config["condition_index"]["sha256"],
            device=str(device),
            device_name=(
                torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
            ),
            torch_version=str(torch.__version__),
            evaluator_sha256=file_sha256(Path(__file__)),
            output_directory=str(output.resolve()),
        )
        for path, _, _, _ in checkpoint_inputs.values():
            logger.link_output(path, checkpoint=True)

        results = {}
        for condition_step, condition in enumerate(config["cohorts"], start=1):
            loader = load_test_cohort(
                manifest_path, manifest, condition_index, condition, config, device,
            )
            results[condition] = {}
            for checkpoint_step, (name, (path, checkpoint, model, task)) in enumerate(
                checkpoint_inputs.items(), start=1
            ):
                metrics = run_epoch(model, loader, task, device)
                logger.log(
                    metrics,
                    step=(condition_step - 1) * len(checkpoint_inputs) + checkpoint_step,
                    prefix=f"test/{condition}/{name}",
                )
                results[condition][name] = {
                    "role": config["checkpoints"][name]["role"],
                    "checkpoint": str(path),
                    "checkpoint_sha256": file_sha256(path),
                    "epoch": checkpoint.get("epoch"),
                    "metrics": metrics,
                }
                print(f"{condition} {name}: {metrics}", flush=True)
            del loader

        result_path = output / "results.json"
        write_json(
            result_path,
            {
                "run_id": logger.run_id,
                "config": config,
                "split": "test",
                "results": results,
            },
        )
        logger.link_output(result_path, kind="metrics")
        print(f"Completed {result_path}", flush=True)
        return output
