"""Train the graph2 spatial/temporal tracker from scratch on Five-mode sequences."""

import gc
import json
from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_corner_tracking import (
    load_training_episodes,
    sequence_arrays,
    training_fallback,
)
from graph.temporal.spatial_temporal_corner import SpatialTemporalCornerTracker
from utils.experiment_logger import ExperimentLogger, write

MODEL_INPUTS = ("features", "anchor", "detected", "observed", "valid")


def supervision_mask(batch):
    """Select missing valid slots, optionally restricted to causal observation history."""
    missing = batch["valid"][..., None] & ~batch["observed"]
    if "ever_seen" in batch:
        missing = missing & batch["ever_seen"]
    return missing


def add_supervision_history(arrays, policy):
    """Add a causal mask for the observed-only run; preserve historical configs."""
    if policy == "after_first_observation":
        arrays["ever_seen"] = np.logical_or.accumulate(arrays["observed"], axis=1)
    elif policy != "all_missing":
        raise ValueError(
            "training.supervision must be 'all_missing' or 'after_first_observation'"
        )


def supervision_counts(data):
    """Record eligible and excluded point-frame counts before optimizing weights."""
    missing = data["valid"][..., None] & ~data["observed"]
    supervised = supervision_mask(data)
    return {
        "missing_point_frames": int(missing.sum().item()),
        "supervised_point_frames": int(supervised.sum().item()),
        "excluded_before_first_observation": int((missing & ~supervised).sum().item()),
        "body": {
            name: int(supervised[..., start:stop].sum().item())
            for name, start, stop in (("peg", 0, 8), ("box", 8, 16), ("hole", 16, 20))
        },
    }


def tensor_batch(data, indices, lengths):
    """Trim each episode batch to its longest valid sequence."""
    length = int(lengths[indices].max())
    selected = torch.as_tensor(indices, device="cuda", dtype=torch.long)
    return {key: values[selected, :length] for key, values in data.items()}


def forward(model, batch):
    inputs = tuple(batch[key] for key in MODEL_INPUTS)
    if "surface_features" in batch:
        return model(*inputs, batch["surface_features"], batch["surface_valid"])
    return model(*inputs)


def missing_loss(prediction, batch):
    """Apply Smooth L1 only to supervised slots; empty batches have zero gradient."""
    missing = supervision_mask(batch)
    selected = prediction[missing].float()
    if selected.numel() == 0:
        return selected.sum()
    return F.smooth_l1_loss(selected * 100, batch["gt"][missing] * 100)


def cuda_preflight(model, data, lengths):
    """Check nontrivial causality, padding invariance and exact detection passthrough."""
    model.eval()
    saved = deepcopy(model.state_dict())
    with torch.no_grad():
        model.output[1].weight.normal_(std=0.03)
    batch = tensor_batch(data, np.array([0]), lengths)
    pivot = min(12, int(lengths[0]) // 2)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["features"][:, pivot:] += 2
    changed["anchor"][:, pivot:] += 0.2
    if "surface_features" in changed:
        changed["surface_features"][:, pivot:] += 3
        changed["surface_valid"][:, pivot:] = ~changed["surface_valid"][:, pivot:]
    prefix = {key: value[:, :pivot] for key, value in batch.items()}
    with torch.inference_mode():
        full = forward(model, batch)
        perturbed = forward(model, changed)
        truncated = forward(model, prefix)
    torch.testing.assert_close(full[:, :pivot], perturbed[:, :pivot], rtol=0, atol=2e-6)
    torch.testing.assert_close(full[:, :pivot], truncated, rtol=0, atol=3e-6)
    torch.testing.assert_close(
        full[batch["observed"]], batch["detected"][batch["observed"]], rtol=0, atol=0
    )
    if not torch.isfinite(full).all():
        raise RuntimeError("Nonfinite CUDA preflight predictions")
    model.load_state_dict(saved)
    return {
        "causal_future_check": True,
        "prefix_padding_check": True,
        "detections_preserved": True,
    }


def autotune_batch(model, data, lengths, options):
    """Measure full training steps at worst-case length and keep memory headroom."""
    initial = deepcopy(model.state_dict())
    records = []
    longest = int(np.argmax(lengths))
    eligible_episodes = supervision_mask(data).flatten(1).any(dim=1).cpu().numpy()
    if not eligible_episodes.any():
        raise ValueError("Training data contains no eligible missing-corner targets")
    longest_supervised = int(np.argmax(np.where(eligible_episodes, lengths, -1)))
    capacity = torch.cuda.get_device_properties(0).total_memory
    for size in options["candidates"]:
        batch = optimizer = prediction = loss = None
        try:
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
            indices = np.full(size, longest)
            # Keep the longest sequence in the stress test, even if it has no
            # eligible target, and include one supervised episode for backward.
            indices[-1] = longest_supervised
            batch = tensor_batch(data, indices, lengths)
            torch.cuda.reset_peak_memory_stats()
            for step in range(2 + options["timed_steps"]):
                if step == 2:
                    torch.cuda.synchronize()
                    started = monotonic()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = forward(model, batch)
                    loss = missing_loss(prediction, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            torch.cuda.synchronize()
            elapsed = monotonic() - started
            peak = torch.cuda.max_memory_reserved()
            throughput = size * options["timed_steps"] / elapsed
            row = {
                "batch_size": size,
                "episodes_per_second": throughput,
                "worst_sequence_frames": int(lengths.max()),
                "peak_reserved_bytes": peak,
                "within_memory_budget": peak <= capacity * options["memory_fraction"],
                "supervised_point_frames": int(supervision_mask(batch).sum().item()),
            }
            records.append(row)
            print("BATCH_PROBE", json.dumps(row), flush=True)
        except torch.OutOfMemoryError:
            records.append({"batch_size": size, "out_of_memory": True})
            print(f"BATCH_PROBE batch_size={size} out_of_memory", flush=True)
            break
        finally:
            batch = optimizer = prediction = loss = None
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    model.load_state_dict(initial)
    eligible = [row for row in records if row.get("within_memory_budget")]
    if not eligible:
        raise RuntimeError("No autotune batch fits the requested GPU memory headroom")
    fastest = max(row["episodes_per_second"] for row in eligible)
    selected = max(
        row["batch_size"]
        for row in eligible
        if row["episodes_per_second"] >= fastest * options["throughput_fraction"]
    )
    return selected, records


def summarize_errors(error, mask):
    values = error[mask]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean_mm": float(values.mean()),
        "p90_mm": float(np.percentile(values, 90)),
        "max_mm": float(values.max()),
    }


@torch.inference_mode()
def validate(model, data, lengths, modes, batch_size):
    """Run predictions and score the same supervision mask used by the loss."""
    model.eval()
    shape = data["observed"].shape
    errors = np.zeros(shape, dtype=np.float32)
    for start in range(0, len(lengths), batch_size):
        indices = np.arange(start, min(start + batch_size, len(lengths)))
        batch = tensor_batch(data, indices, lengths)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = forward(model, batch)
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Nonfinite validation predictions")
        torch.testing.assert_close(
            prediction[batch["observed"]],
            batch["detected"][batch["observed"]],
            rtol=0,
            atol=0,
        )
        distances = (
            torch.linalg.vector_norm(prediction.float() - batch["gt"], dim=-1) * 1000
        )
        errors[indices, : distances.shape[1]] = distances.cpu().numpy()
    return validation_metrics(errors, data, modes)


def validation_metrics(errors, data, modes):
    """Use one supervision mask for every validation metric and selection score."""
    missing = supervision_mask(data).cpu().numpy()
    metrics = {"all": summarize_errors(errors, missing)}
    modes = np.asarray(modes)
    metrics["mode"] = {
        mode: summarize_errors(errors, missing & (modes == mode)[:, None, None])
        for mode in sorted(set(modes.tolist()))
    }
    for mode, result in metrics["mode"].items():
        if not result["count"]:
            raise ValueError(
                f"Validation mode {mode!r} has no supervised missing corners"
            )
    metrics["mode_balanced_mean_mm"] = float(
        np.mean([row["mean_mm"] for row in metrics["mode"].values()])
    )
    metrics["body"] = {}
    for name, start, stop in (("peg", 0, 8), ("box", 8, 16), ("hole", 16, 20)):
        metrics["body"][name] = summarize_errors(
            errors[..., start:stop], missing[..., start:stop]
        )
    age = data["age"].cpu().numpy()
    metrics["age"] = {}
    for name, lower, upper in (
        ("1_10", 1, 10),
        ("11_40", 11, 40),
        ("41_100", 41, 100),
        ("over_100", 101, 100000),
    ):
        metrics["age"][name] = summarize_errors(
            errors, missing & (age >= lower) & (age <= upper)
        )
    depth = data["depth_mm"].cpu().numpy()
    metrics["stage"] = {}
    for name, frames in (
        ("pre", depth <= 0),
        ("entry", (depth > 0) & (depth <= 20)),
        ("deep", depth > 20),
    ):
        metrics["stage"][name] = summarize_errors(errors, missing & frames[..., None])
    if "surface_valid" in data:
        supported = data["surface_valid"].any(dim=-1).cpu().numpy()
        metrics["peg_surface_support"] = {
            "present": summarize_errors(
                errors[..., :8], missing[..., :8] & supported[..., None]
            ),
            "absent": summarize_errors(
                errors[..., :8], missing[..., :8] & ~supported[..., None]
            ),
        }
    return metrics


def save_checkpoint(path, state):
    """Atomically replace a checkpoint so interruptions retain the preceding epoch."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config):
    """Randomly initialize one tracker, tune throughput and train until validation plateaus."""
    if not torch.cuda.is_available():
        raise RuntimeError("Five-mode tracker training requires CUDA")
    policy = config["training"].get("supervision", "all_missing")
    if policy not in ("all_missing", "after_first_observation"):
        raise ValueError(
            "training.supervision must be 'all_missing' or 'after_first_observation'"
        )
    source_path = Path(config["source"]["source_hash_manifest"])
    source_hashes = json.loads(source_path.read_text())
    repository = Path(__file__).resolve().parents[1]
    for relative_path, expected_hash in source_hashes["files"].items():
        if file_sha256(repository / relative_path) != expected_hash:
            raise ValueError(f"Deployed source checksum mismatch: {relative_path}")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.get("cpu_threads", 4))
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("Loading checksum-verified train/validation episodes", flush=True)
    episodes, manifest = load_training_episodes(config["data_root"])
    fallback = training_fallback(episodes["train"])
    prepare_arrays = sequence_arrays
    model_class = SpatialTemporalCornerTracker
    surface_manifest_hash = None
    if config.get("surface_context"):
        from data_collection.five_mode_surface_context import (
            attach_surface_context,
            surface_sequence_arrays,
        )
        from graph.temporal.surface_spatial_corner import SurfaceSpatialCornerTracker

        surface_manifest_hash = attach_surface_context(
            episodes, config["surface_context"], config["data_root"]
        )
        prepare_arrays = surface_sequence_arrays
        model_class = SurfaceSpatialCornerTracker
    datasets, lengths, modes = {}, {}, {}
    counts = {}
    for split in ("train", "validation"):
        arrays, lengths[split], modes[split] = prepare_arrays(
            episodes.pop(split), fallback
        )
        add_supervision_history(arrays, policy)
        datasets[split] = {
            key: torch.as_tensor(value, device="cuda") for key, value in arrays.items()
        }
        counts[split] = supervision_counts(datasets[split])
        if not counts[split]["supervised_point_frames"]:
            raise ValueError(f"{split} contains no supervised missing corners")
        print(f"SUPERVISION {split} {json.dumps(counts[split])}", flush=True)
        print(
            f"Loaded {split}: {len(lengths[split])} episodes, {int(lengths[split].sum())} frames, max_length={int(lengths[split].max())}",
            flush=True,
        )
        del arrays
    gc.collect()
    model = model_class(**config["model"]).cuda()
    initial = deepcopy(model.state_dict())
    preflight = cuda_preflight(model, datasets["validation"], lengths["validation"])
    batch_size, probes = autotune_batch(
        model, datasets["train"], lengths["train"], config["batch_tuning"]
    )
    model.load_state_dict(initial)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    effective = deepcopy(config)
    effective["training"]["supervision"] = policy
    effective["supervision_counts"] = counts
    effective["source"]["files"] = source_hashes["files"]
    if surface_manifest_hash:
        effective["surface_manifest_sha256"] = surface_manifest_hash
    effective.update(
        batch_size=batch_size,
        initialization="random; no checkpoint loaded",
        dataset_manifest_sha256=file_sha256(
            Path(config["data_root"]) / "manifest.json"
        ),
    )
    write(
        output / "batch_tuning.json",
        {"selected_batch_size": batch_size, "probes": probes, "preflight": preflight},
    )
    write(output / "effective_config.json", effective)
    write(output / "supervision.json", {"policy": policy, "counts": counts})
    print(f"SELECTED_BATCH_SIZE={batch_size}", flush=True)
    rates = config["training"]["learning_rates"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rates[0], weight_decay=0.01, fused=True
    )
    best = threshold = float("inf")
    best_epoch = last_gain = rate_index = 0
    history = []
    started = monotonic()
    stopping = "maximum_epochs"
    with ExperimentLogger(
        team="graph",
        model=(
            "five_mode_surface_spatial_corner_tracker"
            if surface_manifest_hash
            else "five_mode_spatial_temporal_corner_tracker"
        ),
        config=effective,
        seed=seed,
        job_type="train",
        root=output / "experiments",
    ) as logger:
        logger.record(
            runtime={
                "gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "precision": "BF16 autocast; FP32 loss",
            },
            dataset={
                "episodes": manifest["episodes"],
                "source_hf": manifest["source_hf"],
                "test_payloads_read": False,
            },
            source=effective["source"],
            supervision={"policy": policy, "counts": counts},
        )
        initial_metrics = validate(
            model,
            datasets["validation"],
            lengths["validation"],
            modes["validation"],
            batch_size,
        )
        if (
            initial_metrics["all"]["count"]
            != counts["validation"]["supervised_point_frames"]
        ):
            raise RuntimeError(
                "Validation eligibility count differs from the startup audit"
            )
        logger.log(initial_metrics, step=0, prefix="validation")
        write(output / "initial_validation.json", initial_metrics)
        print("INITIAL_VALIDATION", json.dumps(initial_metrics), flush=True)
        for epoch in range(1, config["training"]["maximum_epochs"] + 1):
            epoch_started = monotonic()
            model.train()
            total_loss = torch.zeros((), device="cuda")
            total_count = 0
            skipped_batches = 0
            order = rng.permutation(len(lengths["train"]))
            for start in range(0, len(order), batch_size):
                batch = tensor_batch(
                    datasets["train"],
                    order[start : start + batch_size],
                    lengths["train"],
                )
                count = supervision_mask(batch).sum()
                if not count.item():
                    skipped_batches += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = forward(model, batch)
                    loss = missing_loss(prediction, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                total_loss += loss.detach() * count
                total_count += count
            metrics = validate(
                model,
                datasets["validation"],
                lengths["validation"],
                modes["validation"],
                batch_size,
            )
            if int(total_count.item()) != counts["train"]["supervised_point_frames"]:
                raise RuntimeError(
                    "Epoch supervision count differs from the startup audit"
                )
            if (
                metrics["all"]["count"]
                != counts["validation"]["supervised_point_frames"]
            ):
                raise RuntimeError(
                    "Validation eligibility count differs from the startup audit"
                )
            score = metrics["mode_balanced_mean_mm"]
            row = {
                "epoch": epoch,
                "train_loss": float((total_loss / total_count).item()),
                "validation": metrics,
                "learning_rate": rates[rate_index],
                "seconds": monotonic() - epoch_started,
                "supervised_point_frames": int(total_count.item()),
                "skipped_empty_batches": skipped_batches,
            }
            history.append(row)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "fallback": torch.from_numpy(fallback),
                "config": effective,
                "run_id": logger.run_id,
                "history": history,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            }
            if score < best:
                best, best_epoch = score, epoch
                save_checkpoint(output / "best.pt", state)
            if score <= threshold - config["training"]["minimum_improvement_mm"]:
                threshold, last_gain = score, epoch
            should_stop = False
            if epoch - last_gain >= config["training"]["patience"]:
                if rate_index + 1 == len(rates):
                    stopping = "early_stopping_at_lowest_learning_rate"
                    should_stop = True
                else:
                    rate_index += 1
                    optimizer.param_groups[0]["lr"] = rates[rate_index]
                    threshold, last_gain = best, epoch
            state.update(
                best_epoch=best_epoch,
                best_score_mm=best,
                rate_index=rate_index,
                plateau_threshold_mm=threshold,
                last_gain_epoch=last_gain,
            )
            save_checkpoint(output / "last.pt", state)
            write(output / "history.json", history)
            write(
                output / "progress.json",
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_score_mm": best,
                    "learning_rate": rates[rate_index],
                    "batch_size": batch_size,
                    "no_improvement_epochs": epoch - last_gain,
                    "seconds": monotonic() - started,
                    "status": "completed" if should_stop else "running",
                },
            )
            logger.log(row, step=epoch)
            print(
                f"EPOCH {epoch} loss={row['train_loss']:.6f} val_balanced={score:.4f}mm "
                f"val_p90={metrics['all']['p90_mm']:.4f}mm best={best_epoch}/{best:.4f}mm "
                f"lr={rates[rate_index]:.3g} batch={batch_size} seconds={row['seconds']:.2f}",
                flush=True,
            )
            if should_stop:
                break
        result = {
            "run_id": logger.run_id,
            "reason": stopping,
            "epochs": epoch,
            "best_epoch": best_epoch,
            "best_validation_mode_balanced_mean_mm": best,
            "batch_size": batch_size,
            "seconds": monotonic() - started,
            "best_sha256": file_sha256(output / "best.pt"),
        }
        write(output / "completed.json", result)
        for filename in (
            "best.pt",
            "last.pt",
            "history.json",
            "completed.json",
            "batch_tuning.json",
            "effective_config.json",
            "supervision.json",
        ):
            logger.link_output(output / filename)
        print("TRAINING_COMPLETED", json.dumps(result), flush=True)
