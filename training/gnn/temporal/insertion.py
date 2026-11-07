"""Fair Mamba/GRU insertion-state training over frozen, checksummed GNN episodes."""

from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal.insertion_sequences import (
    InsertionEpisodeDataset,
    InsertionWindowDataset,
)
from data_collection.temporal.pose_sequences import (
    collate_pose_episodes,
    fit_observation_normalization,
)
from graph.temporal.insertion import TemporalInsertionEstimator
from training.gnn.temporal.config import LOSS_TERMS, resolve_insertion_config
from training.gnn.temporal.frozen_source import (
    infer_insertion_data,
    prepare_insertion_sequences,
)
from training.gnn.temporal.pose_loss import insertion_loss
from training.gnn.trainer import seed_everything
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def build_insertion_model(effective):
    """Construct either backbone with the exact same feature/pose/output contract."""
    return TemporalInsertionEstimator(
        mode=effective["mode"], with_robot=effective["with_robot"], **effective["model"]
    )


def configure_training_memory(device, settings):
    """Optionally cap this process's CUDA allocator for concurrent experiments."""
    fraction = settings["cuda_memory_fraction"]
    if device.type == "cuda" and fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(fraction), device)


def run_insertion_epoch(model, dataset, device, settings, optimizer=None, generator=None):
    """Shuffle windows, mask padding, supervise each real frame, reset each window."""
    loader = DataLoader(
        dataset,
        batch_size=settings["training"]["episode_batch_size"],
        shuffle=optimizer is not None,
        generator=generator,
        collate_fn=collate_pose_episodes,
    )
    model.train(optimizer is not None)
    sums = {}
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for observations, target, mask in loader:
            observations, target, mask = (
                value.to(device) for value in (observations, target, mask)
            )
            prediction = model.forward_sequence(observations, mask)
            loss, terms = insertion_loss(prediction, target, mask, model.mode, settings["loss"])
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    settings["training"]["gradient_clip_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
            frames = int(mask.sum())
            for name, value in dict(loss=loss, **terms).items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * frames
            count += frames
    if count == 0:
        raise ValueError("Insertion epoch requires nonempty windows")
    return {name: value / count for name, value in sums.items()}


def save_loss_curves(output, history):
    """Save all seven train/validation terms plus the weighted objective."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(12, 12))
    epochs = [row["epoch"] for row in history]
    for axis, term in zip(axes.flat, ("loss",) + LOSS_TERMS):
        for split in ("train", "validation"):
            axis.plot(epochs, [row[split][term] for row in history], label=split)
        axis.set_title(term)
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "loss_curves.png", dpi=120)
    plt.close(figure)


def check_matched_training_config(effective, reference):
    """Reject changes to the reference Mamba's data, heads or training protocol."""
    trained = reference["config"]
    if (
        reference.get("kind") != "temporal_insertion"
        or trained["model"]["architecture"] != "mamba"
        or effective["model"]["architecture"] != "gru"
    ):
        raise ValueError(
            "Matched training requires a temporal_insertion Mamba reference and a new GRU"
        )
    for name in (
        "seed",
        "mode",
        "with_robot",
        "feature_names",
        "observation_contract",
        "training",
        "loss",
    ):
        if effective[name] != trained[name]:
            raise ValueError(f"Matched GRU training differs from Mamba: {name}")
    shared_model_fields = set(trained["model"]) - {"architecture", "backend"}
    for name in shared_model_fields:
        if effective["model"].get(name) != trained["model"][name]:
            raise ValueError(f"Matched GRU pose/projection setting differs: model.{name}")
    for name in ("cache_dir", "raw_root", "require_robot"):
        if effective["data"][name] != trained["data"][name]:
            raise ValueError(f"Matched GRU input source differs: data.{name}")
    for path_name, hash_name in (
        ("checkpoint", "source_checkpoint_sha256"),
        ("split_manifest", "split_manifest_sha256"),
    ):
        if file_sha256(effective[path_name]) != trained[hash_name]:
            raise ValueError(f"Matched GRU source identity differs: {path_name}")


def matched_train_normalization(mean, std, effective, identity, reference):
    """Verify freshly fitted train statistics, then reuse the exact reference buffers."""
    trained = reference["config"]
    if (
        identity["dataset_sha256"] != reference["dataset_sha256"]
        or effective["temporal_split_manifest"] != trained["temporal_split_manifest"]
        or effective["feature_names"] != trained["feature_names"]
    ):
        raise ValueError(
            "Matched normalization requires identical train data, parents and features"
        )
    saved = trained["normalization"]
    if saved["source_split"] != "train":
        raise ValueError("Matched normalization reference must contain train-only statistics")
    reference_mean = torch.tensor(saved["mean"], dtype=mean.dtype)
    reference_std = torch.tensor(saved["std"], dtype=std.dtype)
    # CUDA index_add reductions in Frozen GNN inference can differ by roundoff.
    # Verify the independent train fit before using identical normalization values.
    torch.testing.assert_close(mean, reference_mean, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(std, reference_std, atol=1e-6, rtol=1e-5)
    return reference_mean, reference_std


def train_and_compare(config):
    """Train a matched GRU, then evaluate its best checkpoint and frozen baselines.

    A single sequential job performs training and validation; no polling process
    is needed. All data and optimization settings must match the saved Mamba.
    Evaluation starts only after successful training and normalization checks.
    """
    from evaluation.gnn.temporal.insertion import evaluate

    effective = resolve_insertion_config(config)
    comparison = effective["comparison"]
    reference_path = Path(comparison["checkpoints"]["mamba"]).expanduser().resolve()
    reference_hash = file_sha256(reference_path)
    if reference_hash != comparison["reference_sha256"]:
        raise ValueError("The reference Mamba checksum differs from comparison.reference_sha256")
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    check_matched_training_config(effective, reference)
    if effective["evaluation"]["split"] != "validation":
        raise ValueError("Matched development comparison must use evaluation.split=validation")
    comparison_root = Path(effective["output"]["comparison_root"]).expanduser().resolve()
    effective["matched_training_reference"] = {
        "checkpoint": str(reference_path),
        "sha256": reference_hash,
        "selection": "minimum validation weighted objective within each run",
        "matching_fields": ["data", "seed", "features", "pose_heads", "training", "loss"],
        "normalization_policy": "verify fresh train fit, reuse exact Mamba train-only statistics",
        "evaluation_cudnn_allow_tf32": False,
    }
    print("Matched GRU protocol verified against the selected Mamba checkpoint", flush=True)
    training_output = train(effective, normalization_reference=reference)
    checkpoint_path = training_output / "best.pt"
    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if selected["dataset_sha256"] != reference["dataset_sha256"]:
        raise ValueError("Trained GRU and reference Mamba dataset identities differ")
    for name in ("temporal_split_manifest", "normalization"):
        if selected["config"][name] != reference["config"][name]:
            raise ValueError(f"Trained GRU and reference Mamba differ: {name}")
    if file_sha256(reference_path) != reference_hash:
        raise ValueError("Reference Mamba changed during GRU training")

    evaluation_config = deepcopy(effective)
    evaluation_config["trainer"] = "evaluation.gnn.temporal.insertion:evaluate"
    evaluation_config["temporal_checkpoint"] = str(checkpoint_path)
    evaluation_config["output"]["root"] = str(comparison_root)
    print(f"Matched GRU training completed: {training_output}", flush=True)
    print(
        "Starting paired full-episode validation: Frozen GNN / Learned Gate / GRU / Mamba",
        flush=True,
    )
    # Disable reduced-precision CuDNN inference to retain sequence/step parity.
    # The training kernels and optimization settings above remain unchanged.
    with torch.backends.cudnn.flags(allow_tf32=False):
        return evaluate(evaluation_config)


def train(config, normalization_reference=None):
    """Train only temporal weights; select best by validation, never read test frames."""
    effective = resolve_insertion_config(config)
    device = set_device(effective["device"])
    configure_training_memory(device, effective["training"])
    seed_everything(effective["seed"], device)
    print("Preparing Frozen GNN checkpoint and parent-disjoint development split", flush=True)
    frozen, loaders, identity = prepare_insertion_sequences(
        effective, device, ("train", "validation")
    )
    with ExperimentLogger(
        team="graph",
        model="temporal_insertion_" + effective["model"]["architecture"],
        config=effective,
        seed=effective["seed"],
    ) as experiment:
        output = Path(effective["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        experiment.link_output(effective["checkpoint"], checkpoint=True)
        episodes = {}
        for split, loader in loaders.items():
            print(f"Frozen GNN inference and observed robot join: {split}", flush=True)
            data = infer_insertion_data(frozen, loader, device, identity, effective, split)
            episodes[split] = InsertionEpisodeDataset(data)
            print(
                f"{split}: {len(episodes[split])} episodes, {len(data['frame'])} frames", flush=True
            )
        mean, std = fit_observation_normalization(episodes["train"])
        if normalization_reference is not None:
            mean, std = matched_train_normalization(
                mean, std, effective, identity, normalization_reference
            )
        effective["normalization"] = {
            "source_split": "train",
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
        model = build_insertion_model(effective).to(device)
        model.set_normalization(mean.to(device), std.to(device))
        windows = {
            split: InsertionWindowDataset(dataset, effective["training"]["sequence_length"])
            for split, dataset in episodes.items()
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=effective["training"]["lr"])
        generator = torch.Generator().manual_seed(effective["seed"])
        history = []
        best = float("inf")
        write_json(output / "effective_config.json", effective)
        experiment.record(
            normalization=effective["normalization"],
            parameter_count=sum(p.numel() for p in model.parameters()),
            frozen_parameter_count=sum(p.numel() for p in frozen.parameters()),
            dataset_sha256=identity["dataset_sha256"],
            device=str(device),
            backend=model.backend,
            context_policy="reset_windows",
            cuda_memory_fraction=effective["training"]["cuda_memory_fraction"],
            sequence_counts={split: len(value) for split, value in episodes.items()},
            test_frames_read=False,
            teacher_forcing=False,
        )
        for epoch in range(1, effective["training"]["epochs"] + 1):
            print(f"Insertion {model.architecture} epoch {epoch} starting", flush=True)
            trained = run_insertion_epoch(
                model, windows["train"], device, effective, optimizer, generator
            )
            validation = run_insertion_epoch(model, windows["validation"], device, effective)
            row = {"epoch": epoch, "train": trained, "validation": validation}
            history.append(row)
            experiment.log(row, step=epoch)
            if validation["loss"] < best:
                best = validation["loss"]
                torch.save(
                    {
                        "kind": "temporal_insertion",
                        "model": model.state_dict(),
                        "config": effective,
                        "dataset_sha256": identity["dataset_sha256"],
                        "epoch": epoch,
                        "validation_loss": best,
                    },
                    output / "best.pt",
                )
                experiment.link_output(output / "best.pt")
            print(
                f"Insertion {model.architecture}/{model.mode} epoch {epoch}: "
                f"train={trained['loss']:.5f}, validation={validation['loss']:.5f}",
                flush=True,
            )
        write_json(output / "history.json", history)
        save_loss_curves(output, history)
        for name in ("effective_config.json", "history.json", "loss_curves.png"):
            experiment.link_output(output / name)
        return output
