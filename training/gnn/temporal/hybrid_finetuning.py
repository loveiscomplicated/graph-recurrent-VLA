"""Run the three controlled Peg-head H1 ablations through the root entrypoint."""

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal.finetuning_split import (
    make_finetuning_split,
    select_episode_data,
)
from data_collection.temporal.hybrid_features import (
    cache_frozen_features,
    predict_cached,
    replay_hybrid,
)
from data_collection.temporal.insertion_sequences import select_insertion_loaders
from evaluation.gnn.temporal.hybrid_diagnostics import check_reproduction
from evaluation.gnn.temporal.hybrid_finetuning import (
    analyze_finetuning,
    selection_score,
)
from evaluation.gnn.temporal.hybrid_inputs import load_reference, restore_branches
from evaluation.gnn.temporal.insertion import score_insertion_methods
from graph.temporal.hybrid_finetuning import (
    HybridPegFineTuner,
    parameter_changes,
    snapshot_state,
)
from training.gnn.temporal.frozen_source import infer_insertion_data, load_frozen_source
from training.gnn.temporal.hybrid_finetuning_loss import (
    TERMS,
    geometry_terms,
    measure_gradients,
)
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger

ABLATIONS = {
    "pose_only": (),
    "pose_relative": ("relative_position",),
    "pose_relative_lateral": ("relative_position", "lateral_position"),
}
CACHE_INPUTS = ("representation", "frozen_world", "mamba_world", "gate_world")


def validate_config(config):
    """Reject invalid settings before opening data or creating an optimizer."""
    effective = deepcopy(config)
    if effective.get("action") not in ("audit", "run"):
        raise ValueError("action must be audit or run")
    training = effective["training"]
    for name in ("epochs", "batch_size", "patience", "calibration_frames"):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    for section, names in (
        ("training", ("lr", "gradient_clip_norm")),
        (
            "loss",
            (
                "position_unit_m",
                "rotation_unit_rad",
                "relative_gradient_ratio",
                "lateral_gradient_ratio",
            ),
        ),
    ):
        for name in names:
            value = effective[section][name]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{name} must be finite and positive")
    for section, name in (
        ("training", "weight_decay"),
        ("loss", "peg_position_weight"),
        ("loss", "peg_rotation_weight"),
    ):
        value = effective[section][name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{section}.{name} must be finite and nonnegative")
    if min(effective["loss"]["peg_position_weight"], effective["loss"]["peg_rotation_weight"]) <= 0:
        raise ValueError("Both Peg pose supervision weights must be positive")
    for name in (
        "peg_position_tolerance_mm",
        "peg_rotation_tolerance_deg",
        "axial_tolerance_mm",
        "moving_peg_tolerance_mm",
    ):
        value = effective["selection"][name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"selection.{name} must be finite and nonnegative")
    expected = {"relative_mean", "lateral_mean", "relative_p95", "lateral_p95"}
    weights = effective["selection"]["score_weights"]
    if (
        set(weights) != expected
        or any(not math.isfinite(v) or v < 0 for v in weights.values())
        or not any(weights.values())
    ):
        raise ValueError(
            "selection.score_weights must define positive geometry selection contributions"
        )
    limits = effective["loss"]["auxiliary_weight_bounds"]
    if len(limits) != 2 or not 0 < limits[0] <= limits[1] < math.inf:
        raise ValueError("loss.auxiliary_weight_bounds must be [positive minimum, maximum]")
    return effective


def cached_batch(adapter, cache, indices, device):
    """Move only a valid-frame minibatch; GT is returned separately from inputs."""
    inputs = [cache[name][indices].to(device) for name in CACHE_INPUTS]
    return adapter(*inputs), cache["target"][indices].to(device)


def calibrate_weights(adapter, cache, config, device):
    """Fix auxiliary weights using only initialization gradients on training frames.

    The requested ratios are small fractions of the combined pose gradient.
    Clipping bounds and measured relative/lateral overlap are recorded, so a
    nominal coefficient is never assumed to have a universal scale.
    """
    adapter.enable_peg_head()
    generator = torch.Generator().manual_seed(config["seed"])
    indices = torch.randperm(len(cache["target"]), generator=generator)[
        : config["training"]["calibration_frames"]
    ]
    prediction, target = cached_batch(adapter, cache, indices, device)
    terms = geometry_terms(prediction, target, config["loss"])
    weights = {name: 1.0 for name in TERMS}
    weights.update(
        peg_position=config["loss"]["peg_position_weight"],
        peg_rotation=config["loss"]["peg_rotation_weight"],
    )
    parameters = list(adapter.mamba.heads["peg"].parameters())
    measured = measure_gradients(terms, parameters, weights)
    if measured["pose_gradient_norm"] <= 1e-12:
        raise ValueError("Training pose gradient is zero; auxiliary weights cannot be calibrated")
    for term, ratio in (
        ("relative_position", "relative_gradient_ratio"),
        ("lateral_position", "lateral_gradient_ratio"),
    ):
        norm = measured[term]["gradient_norm"]
        if norm <= 1e-12:
            raise ValueError(f"Training {term} gradient is zero")
        requested = config["loss"][ratio] * measured["pose_gradient_norm"] / norm
        weights[term] = float(np.clip(requested, *config["loss"]["auxiliary_weight_bounds"]))
    measured = measure_gradients(terms, parameters, weights)
    return weights, {
        "frames": len(indices),
        "source_split": "fine_tuning_train",
        "initial_gradients": measured,
        "weights": weights,
        "sample_indices": indices.tolist(),
    }


def train_epoch(adapter, cache, optimizer, config, weights, generator, device):
    """Shuffle cached valid frames; each retains its frozen full-episode context."""
    adapter.enable_peg_head()
    adapter.train()
    parameters = list(adapter.mamba.heads["peg"].parameters())
    order = torch.randperm(len(cache["target"]), generator=generator)
    totals = {name: 0.0 for name in ("loss", *TERMS)}
    gradients = None
    for indices in order.split(config["training"]["batch_size"]):
        prediction, target = cached_batch(adapter, cache, indices, device)
        terms = geometry_terms(prediction, target, config["loss"])
        loss = sum(weights[name] * value for name, value in terms.items())
        if not torch.isfinite(loss):
            raise ValueError("Fine-tuning produced a nonfinite loss")
        if gradients is None:
            gradients = measure_gradients(terms, parameters, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, config["training"]["gradient_clip_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        for name, value in dict(loss=loss, **terms).items():
            totals[name] += float(value.detach()) * len(indices)
    return {name: value / len(order) for name, value in totals.items()}, gradients, float(norm)


def save_head(path, adapter, config, name, epoch, score, weights):
    """Write a new head artifact; original Mamba/GNN/Gate files remain read-only."""
    torch.save(
        {
            "kind": "hybrid_peg_finetuning",
            "peg_head": snapshot_state(adapter.mamba.heads["peg"]),
            "config": config,
            "ablation": name,
            "epoch": epoch,
            "selection_score": score,
            "weights": weights,
        },
        path,
    )


def fit_ablation(
    adapter,
    caches,
    data,
    baseline,
    original_state,
    config,
    name,
    weights,
    output,
    experiment,
    device,
):
    """Start independently from epoch 19 with a fresh optimizer and identical RNG."""
    adapter.mamba.load_state_dict(original_state, strict=True)
    adapter.enable_peg_head()
    trainable = [(key, value) for key, value in adapter.named_parameters() if value.requires_grad]
    if not trainable or any(not key.startswith("mamba.heads.peg.") for key, _ in trainable):
        raise AssertionError("Only mamba.heads.peg parameters may be trainable")
    optimizer = torch.optim.AdamW(
        [value for _, value in trainable],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    optimized = {id(value) for group in optimizer.param_groups for value in group["params"]}
    if optimized != {id(value) for _, value in trainable}:
        raise AssertionError("Optimizer does not contain exactly the trainable Peg head")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    generator = torch.Generator().manual_seed(config["seed"])
    directory = output / name
    directory.mkdir()
    best_score, failures = selection_score(baseline, baseline, config["selection"])
    assert not failures
    best_epoch = 0
    save_head(directory / "best.pt", adapter, config, name, 0, best_score, weights)
    history = []
    for epoch in range(1, config["training"]["epochs"] + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        trained, gradients, unclipped = train_epoch(
            adapter, caches["train"], optimizer, config, weights, generator, device
        )
        prediction = predict_cached(
            adapter,
            caches["development"],
            data["development"],
            device,
            config["training"]["batch_size"],
        )
        summary, _ = score_insertion_methods(
            data["development"], {"candidate": prediction}, config["evaluation"]
        )
        method = summary["methods"]["candidate"]
        score, failures = selection_score(method, baseline, config["selection"])
        parameter_changes(original_state, adapter.mamba)
        row = {
            "epoch": epoch,
            "lr": learning_rate,
            "train": trained,
            "gradient_contributions": gradients,
            "last_unclipped_gradient_norm": unclipped,
            "selection_score": score,
            "guard_failures": failures,
            "development": method["groups"],
            "development_peg_dynamics": method["peg_dynamics"],
        }
        history.append(row)
        experiment.log(row, step=epoch, prefix=name)
        if not failures and score < best_score:
            best_score, best_epoch = score, epoch
            save_head(directory / "best.pt", adapter, config, name, epoch, score, weights)
        scheduler.step()
        write_json(directory / "history.json", history)
        print(
            f"{name} epoch {epoch}: loss={trained['loss']:.6f}, "
            f"development score={score:.4f}, guard failures={failures}",
            flush=True,
        )
        if epoch - best_epoch >= config["training"]["patience"]:
            break
    save_head(directory / "last.pt", adapter, config, name, epoch, score, weights)
    write_json(
        directory / "last_parameter_diff.json", parameter_changes(original_state, adapter.mamba)
    )
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    adapter.mamba.heads["peg"].load_state_dict(best["peg_head"], strict=True)
    write_json(
        directory / "best_parameter_diff.json", parameter_changes(original_state, adapter.mamba)
    )
    for path in directory.iterdir():
        experiment.link_output(path)
    return {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epochs_run": epoch,
        "weights": weights,
        "best_checkpoint": str(directory / "best.pt"),
    }


def prepare_training_data(config, trained, output, device):
    """Reuse the existing checksummed GNN cache and observed-proprioception join."""
    effective = deepcopy(trained)
    effective.update(checkpoint=config["gnn_checkpoint"], data=config["data"])
    effective["split_manifest"] = str(output / "original_split_manifest.json")
    write_json(effective["split_manifest"], trained["temporal_split_manifest"])
    frozen, source, expected_hash = load_frozen_source(effective, device)
    frozen_before = snapshot_state(frozen)
    loaders, identity, _ = select_insertion_loaders(
        source,
        effective["split_manifest"],
        ("train",),
        torch.Generator().manual_seed(config["seed"]),
    )
    if identity["dataset_sha256"] != expected_hash:
        raise ValueError("GNN training cache identity differs from original checkpoint")
    data = infer_insertion_data(frozen, loaders["train"], device, identity, effective, "train")
    if any(
        not torch.equal(value, frozen.state_dict()[name].cpu())
        for name, value in frozen_before.items()
    ):
        raise AssertionError("Frozen GNN changed while preparing fine-tuning observations")
    return data


def run(config):
    """Audit exact H1; train only with available authorized original training data."""
    config = validate_config(config)
    device = set_device(config["device"])
    inference_device = set_device(config.get("inference_device", config["device"]))
    torch.set_num_threads(config["math_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    benchmark, baselines, reference, previous, provenance, identity, frames = load_reference(config)
    mamba, gate, source_audit = restore_branches(
        config, reference, provenance, identity, inference_device
    )
    original_state, original_gate = snapshot_state(mamba), snapshot_state(gate)
    hashes_before = {
        name: file_sha256(config[name + "_checkpoint"]) for name in ("gnn", "mamba", "gate")
    }
    split = make_finetuning_split(
        reference["temporal_split_manifest"],
        seed=config["seed"],
        development_fraction=config["development_fraction"],
    )
    adapter = HybridPegFineTuner(mamba, gate)
    trainable = {
        name: value.numel() for name, value in adapter.named_parameters() if value.requires_grad
    }
    print("Trainable parameters:", trainable, "total:", sum(trainable.values()), flush=True)
    config["evaluation"] = deepcopy(reference["evaluation"])
    config["source_checkpoint_hashes"] = hashes_before
    config["finetuning_split"] = split
    config["context_policy"] = (
        "full_episode_streaming_frozen_activations; framewise Peg head; no padding"
    )
    source_files = (
        "graph/temporal/hybrid_finetuning.py",
        "data_collection/temporal/finetuning_split.py",
        "training/gnn/temporal/hybrid_finetuning.py",
        "training/gnn/temporal/hybrid_finetuning_loss.py",
        "evaluation/gnn/temporal/hybrid_finetuning.py",
    )
    config["finetuning_source_sha256"] = {name: file_sha256(name) for name in source_files}
    with ExperimentLogger(
        team="graph",
        model="hybrid_peg_finetuning",
        config=config,
        seed=config["seed"],
        job_type="audit" if config["action"] == "audit" else "train",
    ) as experiment:
        output = Path(config["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "effective_config.json", config)
        write_json(output / "finetuning_split.json", split)
        write_json(output / "benchmark_frame_manifest.json", frames)
        audit = {
            "source": source_audit,
            "trainable_parameters": trainable,
            "trainable_parameter_count": sum(trainable.values()),
            "implementation": "A: existing heads.peg",
            "independent_test_executed": False,
            "optimizer_created": False,
            "training_device": str(device),
            "frozen_streaming_device": str(inference_device),
        }
        missing = [
            str(path)
            for path in (
                Path(config["data"]["cache_dir"]) / "metadata/data_identity.json",
                Path(config["data"]["raw_root"]),
            )
            if not path.exists()
        ]
        print(
            "Replaying unchanged H1 benchmark and capturing frozen causal activations", flush=True
        )
        benchmark_cache, original_h1 = cache_frozen_features(
            mamba, gate, benchmark, inference_device
        )
        adapter.enable_peg_head()
        initialized = predict_cached(
            adapter, benchmark_cache, benchmark, inference_device, config["training"]["batch_size"]
        )
        np.testing.assert_allclose(initialized, original_h1, atol=2e-6, rtol=1e-5)
        audit["initialization_max_component_difference"] = float(
            np.abs(initialized - original_h1).max()
        )
        cached_baselines = {
            "gnn": benchmark["prediction"],
            "learned_gate": benchmark_cache["gate_world"].clone(),
            "mamba": benchmark_cache["mamba_world"].clone(),
        }
        for name in ("learned_gate", "mamba"):
            value = cached_baselines[name]
            value[:, :3] -= torch.from_numpy(benchmark["tcp_world"])
            value[:, 10:13] -= torch.from_numpy(benchmark["tcp_world"])
            cached_baselines[name] = value.numpy()
        audit["reproduction"] = check_reproduction(
            cached_baselines, baselines, previous, config["evaluation"], benchmark
        )
        predictions = {"original_h1": original_h1}
        results = {}
        if config["action"] == "run" and not missing:
            all_training = prepare_training_data(config, reference, output, device)
            datasets = {
                name: select_episode_data(all_training, split["splits"][name])
                for name in ("train", "development")
            }
            caches = {}
            development_baseline = None
            for name, data in datasets.items():
                print(f"Caching {name}: {len(np.unique(data['episode']))} episodes", flush=True)
                caches[name], h1 = cache_frozen_features(mamba, gate, data, inference_device)
                np.savez_compressed(output / (name + "_observations.npz"), **data)
                if name == "development":
                    development_baseline = score_insertion_methods(
                        data, {"original_h1": h1}, config["evaluation"]
                    )[0]
                    write_json(output / "development_baseline.json", development_baseline)
            adapter.to(device)
            weights, calibration = calibrate_weights(adapter, caches["train"], config, device)
            write_json(output / "gradient_calibration.json", calibration)
            audit["optimizer_created"] = True
            for name, auxiliary in ABLATIONS.items():
                adapter.to(device)
                ablation_weights = dict(weights)
                for term in ("relative_position", "lateral_position"):
                    if term not in auxiliary:
                        ablation_weights[term] = 0.0
                results[name] = fit_ablation(
                    adapter,
                    caches,
                    datasets,
                    development_baseline["methods"]["original_h1"],
                    original_state,
                    config,
                    name,
                    ablation_weights,
                    output,
                    experiment,
                    device,
                )
                # Selection is complete. Benchmark results are never fed back into fit_ablation.
                adapter.to(inference_device)
                for role, data, cache in (
                    ("development", datasets["development"], caches["development"]),
                    ("benchmark", benchmark, benchmark_cache),
                ):
                    cached = predict_cached(
                        adapter, cache, data, inference_device, config["training"]["batch_size"]
                    )
                    streamed, _, _ = replay_hybrid(mamba, gate, data, inference_device)
                    fresh = streamed["hybrid_h1_a"]
                    np.testing.assert_allclose(cached, fresh, atol=2e-6, rtol=1e-5)
                    np.testing.assert_array_equal(
                        fresh[:, 10:17], cache["original_h1"].numpy()[:, 10:17]
                    )
                    results[name][role + "_cached_streaming_max_difference"] = float(
                        np.abs(cached - fresh).max()
                    )
                    if role == "benchmark":
                        predictions[name] = fresh
                    else:
                        write_json(
                            output / name / "development_summary.json",
                            score_insertion_methods(data, {name: fresh}, config["evaluation"])[0],
                        )
            status = "completed_ablations_no_independent_test"
        else:
            status = "audit_only" if config["action"] == "audit" else "missing_training_data"
        summary, errors = score_insertion_methods(benchmark, predictions, config["evaluation"])
        summary.update(
            split="reused_development_benchmark",
            independent_test=False,
            selection_data="fine-tuning development within original train only",
        )
        diagnostics = analyze_finetuning(
            benchmark,
            predictions,
            errors,
            reference["temporal_split_manifest"],
            seed=config["seed"],
            resamples=config["bootstrap_resamples"],
        )
        write_json(output / "summary.json", summary)
        write_json(output / "diagnostics.json", diagnostics)
        np.savez_compressed(
            output / "predictions.npz",
            sample_ids=benchmark["sample_ids"],
            **{name: value for name, value in predictions.items()},
        )
        np.savez_compressed(
            output / "frame_errors.npz",
            sample_ids=benchmark["sample_ids"],
            **{
                name + "__" + metric: value
                for name, values in errors.items()
                for metric, value in values.items()
            },
        )
        audit["final_mamba_parameter_diff"] = parameter_changes(original_state, mamba)
        audit["gate_tensors_bitwise_unchanged"] = all(
            torch.equal(value, gate.state_dict()[name].cpu())
            for name, value in original_gate.items()
        )
        if not audit["gate_tensors_bitwise_unchanged"]:
            raise AssertionError("Frozen Gate tensor changed")
        audit["source_checkpoint_hashes_before"] = hashes_before
        audit["source_checkpoint_hashes_after"] = {
            name: file_sha256(config[name + "_checkpoint"]) for name in hashes_before
        }
        if audit["source_checkpoint_hashes_after"] != hashes_before:
            raise AssertionError("An original checkpoint file changed")
        write_json(output / "audit.json", audit)
        write_json(
            output / "execution_status.json",
            {
                "status": status,
                "missing_training_inputs": missing,
                "ablations": results,
                "independent_test_executed": False,
            },
        )
        experiment.record(
            dataset_sha256=identity["dataset_sha256"],
            status_detail=status,
            independent_test=False,
            optimizer_created=audit["optimizer_created"],
            trainable_parameter_count=sum(trainable.values()),
            device=str(device),
        )
        for path in output.iterdir():
            if path.is_file():
                experiment.link_output(path)
        print(f"H1 fine-tuning {status}: {output}", flush=True)
        return output
