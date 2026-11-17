"""Paired insertion-precision, tracking and compute evaluation after causal replay."""

import hashlib
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset
from data_collection.temporal.pose_sequences import stream_arguments
from evaluation.gnn.temporal.hole_gru import measure_callback, synchronize
from evaluation.gnn.temporal.pose_gru import (
    diagnostic_groups,
    first_observation_hold,
    lag_and_drift,
    pose_errors,
    replay_pose,
    save_pose_plots,
)
from evaluation.gnn.temporal.precision import precision_errors
from evaluation.gnn.temporal_gate import replay_gate
from evaluation.gnn.temporal_memory import replay_memory
from graph.pose_geometry import (
    geometric_endpoint,
    peg_axis,
)
from graph.temporal.insertion import StreamingInsertionEstimator
from graph.temporal.observations import QUALITY_NAMES
from graph.temporal.pose_gru import TemporalPoseGRU
from graph.temporal.pose_observations import BASE_NAMES, POSE_CONTRACT, ROBOT_NAMES
from graph.temporal_gate import LearnedTemporalGate
from graph.utils.hole_memory import MemorySettings
from training.gnn.temporal.config import resolve_insertion_config
from training.gnn.temporal.frozen_source import (
    infer_insertion_data,
    prepare_insertion_sequences,
)
from training.gnn.temporal.insertion import build_insertion_model
from training.gnn.temporal.pose_loss import insertion_loss
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def load_existing_pose_gru(path, reference, identity, device):
    """Restore an older 53D/62D GRU without changing its learned input contract.

    The development parents and Frozen GNN must match. Different feature/loss
    settings are retained and reported rather than disguised as a controlled
    backbone comparison. Validation data never fits normalization statistics.
    """
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("kind") != "temporal_pose_gru":
        raise ValueError("gru_checkpoint must be a temporal_pose_gru checkpoint")
    check_development_provenance(payload, reference, identity)
    trained = payload["config"]
    names = BASE_NAMES + (ROBOT_NAMES if trained["with_robot"] else ())
    if trained["feature_names"] != list(names) or trained["observation_contract"] != POSE_CONTRACT:
        raise ValueError("Existing GRU has an unsupported observation contract")
    normalization = trained["normalization"]
    if normalization["source_split"] != "train":
        raise ValueError("Existing GRU normalization must come from train only")
    model = TemporalPoseGRU(
        mode=trained["mode"], with_robot=trained["with_robot"], **trained["gru"]
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    torch.testing.assert_close(model.input_mean.cpu(), torch.tensor(normalization["mean"]))
    torch.testing.assert_close(model.input_std.cpu(), torch.tensor(normalization["std"]))
    return model.eval(), payload


def existing_gru_data_view(data, with_robot):
    """Keep exact frame/target/GNN arrays; expose only the GRU's trained observations."""
    result = dict(data)
    robot_fields = ("tcp_quaternion", "gripper_qpos")
    if with_robot:
        if any(name not in result for name in robot_fields):
            raise ValueError(
                "The existing GRU requires observed robot fields absent from the reference"
            )
    else:
        for name in robot_fields:
            result.pop(name, None)
    return result


def check_development_provenance(payload, effective, identity):
    """Allow legacy baseline manifests only when actual development parents match."""
    trained = payload["config"]
    if (
        trained["source_checkpoint_sha256"] != effective["source_checkpoint_sha256"]
        or payload["dataset_sha256"] != identity["dataset_sha256"]
    ):
        raise ValueError("Comparison Frozen GNN/cache provenance differs")
    manifest = trained.get("temporal_split_manifest", trained.get("gate_split_manifest"))
    if manifest is None:
        raise ValueError("Comparison checkpoint has no explicit parent split provenance")
    expected = effective["temporal_split_manifest"]
    for split in ("train", "validation"):
        if set(manifest["splits"][split]) != set(expected["splits"][split]):
            raise ValueError(f"Comparison development split differs: {split}")
        for episode in expected["splits"][split]:
            if manifest["parent_by_episode"][episode] != expected["parent_by_episode"][episode]:
                raise ValueError(f"Comparison parent provenance differs: {episode}")


def load_insertion_model(path, effective, identity, device):
    """Restore schema and normalization without touching evaluation statistics."""
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if payload.get("kind") != "temporal_insertion":
        raise ValueError("Expected a temporal_insertion checkpoint")
    check_development_provenance(payload, effective, identity)
    trained = payload["config"]
    for name in ("feature_names", "observation_contract", "with_robot"):
        if trained[name] != effective[name]:
            raise ValueError(f"Comparison feature contract differs: {name}")
    model = build_insertion_model(trained).to(device)
    model.load_state_dict(payload["model"], strict=True)
    normalization = trained["normalization"]
    if normalization["source_split"] != "train":
        raise ValueError("Normalization must be fitted on train observations only")
    torch.testing.assert_close(model.input_mean.cpu(), torch.tensor(normalization["mean"]))
    torch.testing.assert_close(model.input_std.cpu(), torch.tensor(normalization["std"]))
    return model.eval(), payload


@torch.inference_mode()
def replay_insertion(model, data, device):
    """Check full causal forward against real per-step caches, episode by episode."""
    dataset = InsertionEpisodeDataset(data)
    estimator = StreamingInsertionEstimator(model)
    outputs = []
    control_geometry = []
    maximum_difference = 0.0
    for index, rows in enumerate(dataset.indices):
        observations, _ = dataset[index]
        expected = model.forward_sequence(observations[None].to(device))
        estimator.reset(str(dataset.episodes[index]))
        streamed = []
        for row in rows:
            result = estimator.step(stream_arguments(data, row, device))
            streamed.append(result["world_object21"])
            outputs.append(result["object21"].cpu().numpy())
            control_geometry.append(result["relative_hole_frame"].cpu().numpy())
        actual = torch.stack(streamed, dim=1)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
        maximum_difference = max(maximum_difference, float((actual - expected).abs().max()))
        estimator.discard(str(dataset.episodes[index]))
    return np.concatenate(outputs), np.concatenate(control_geometry), maximum_difference


def insertion_errors(prediction, target):
    """Score estimation error, with GT-frame and prediction-frame quantities distinct."""
    errors = pose_errors(prediction, target)
    prediction, target = torch.as_tensor(prediction), torch.as_tensor(target)
    shared = precision_errors(prediction, target, endpoint_from_pose=False)
    metric_names = {
        "relative_3d": "relative_3d_mm",
        "lateral": "lateral_mm",
        "axial": "axial_mm",
        "control_frame_relative": "relative_in_predicted_hole_frame_mm",
        "relative_rotation": "relative_orientation_deg",
        "relative_axis_direction_rotation": "relative_axis_direction_deg",
    }
    errors.update({name: shared[field].numpy() for name, field in metric_names.items()})

    def alignment(object21):
        peg = peg_axis(object21[:, 3:7])
        hole = peg_axis(object21[:, 13:17])
        cross = torch.linalg.vector_norm(torch.linalg.cross(peg, hole), dim=-1)
        return torch.atan2(cross, (peg * hole).sum(-1))

    errors["axis_alignment_rotation"] = torch.rad2deg(
        (alignment(prediction) - alignment(target)).abs()
    ).numpy()
    return errors


def distribution(values, unit):
    """Frame-weighted precision statistics; empty diagnostic groups remain explicit."""
    values = np.asarray(values)
    result = {"frames": len(values)}
    for name, value in (
        ("mean", np.mean),
        ("median", np.median),
        ("p90", lambda x: np.percentile(x, 90)),
        ("p95", lambda x: np.percentile(x, 95)),
        ("worst", np.max),
    ):
        result[name + "_" + unit] = float(value(values)) if len(values) else None
    return result


def precision_percentages(errors, mask):
    """Strict estimation thresholds; these never imply physical insertion success."""
    result = {"frames": int(mask.sum())}
    for threshold in (1, 2, 3):
        result[f"lateral_lt_{threshold}mm_percent"] = (
            float(np.mean(errors["lateral"][mask] < threshold) * 100) if mask.any() else None
        )
    result["relative_3d_lt_3mm_percent"] = (
        float(np.mean(errors["relative_3d"][mask] < 3) * 100) if mask.any() else None
    )
    return result


def score_insertion_methods(data, predictions, settings):
    """Pair every method on identical episodes, groups and frame indices."""
    groups, missing_runs = diagnostic_groups(data, settings)
    summary = {
        "frames": len(data["frame"]),
        "episodes": len(np.unique(data["episode"])),
        "physical_success_evaluated": False,
        "methods": {},
        "missing_runs": missing_runs,
        "phase_labels_available": settings["phase_labels"] is not None,
        "occlusion_scope": "observed vertex absence; optional depth proxy, not GT visibility",
    }
    errors = {}
    for name, prediction in predictions.items():
        values = insertion_errors(prediction, data["target"])
        errors[name] = values

        def summarize(mask):
            metrics = {
                metric: distribution(error[mask], "deg" if "rotation" in metric else "mm")
                for metric, error in values.items()
            }
            metrics["precision"] = precision_percentages(values, mask)
            return metrics

        summary["methods"][name] = {
            "groups": {group: summarize(mask) for group, mask in groups.items()},
            "episodes": {
                str(episode): summarize(data["episode"] == episode)
                for episode in np.unique(data["episode"])
            },
            "peg_dynamics": lag_and_drift(
                data, prediction, groups["peg_moving"], settings["lag_max_frames"]
            ),
        }
    paired = {}
    for left, right in combinations(predictions, 2):
        paired[left + "__minus__" + right] = {
            str(episode): {
                metric: float(
                    errors[left][metric][data["episode"] == episode].mean()
                    - errors[right][metric][data["episode"] == episode].mean()
                )
                for metric in errors[left]
            }
            for episode in np.unique(data["episode"])
        }
    summary["paired_episode_mean_differences"] = paired
    tradeoffs = {}
    for candidate, baseline in combinations(predictions, 2):
        moving = groups["peg_moving"]
        relative_improvement = float(
            errors[candidate]["relative_3d"].mean() - errors[baseline]["relative_3d"].mean()
        )
        moving_degradation = (
            float(
                errors[candidate]["peg_position"][moving].mean()
                - errors[baseline]["peg_position"][moving].mean()
            )
            if moving.any()
            else None
        )
        tradeoffs[candidate + "__versus__" + baseline] = {
            "candidate_minus_baseline_relative_mean_mm": relative_improvement,
            "candidate_minus_baseline_moving_peg_mean_mm": moving_degradation,
            "relative_improved_but_moving_peg_worsened": (
                relative_improvement < 0 and moving_degradation > 0
                if moving_degradation is not None
                else None
            ),
            "reverse_relative_improved_but_moving_peg_worsened": (
                relative_improvement > 0 and moving_degradation < 0
                if moving_degradation is not None
                else None
            ),
        }
    summary["relative_geometry_motion_tradeoffs"] = tradeoffs
    return summary, errors


def cache_bytes(cache):
    """Count actual tensor bytes in nested recurrent state, excluding model weights."""
    if isinstance(cache, torch.Tensor):
        return cache.numel() * cache.element_size()
    if cache is None:
        return 0
    return sum(cache_bytes(value) for value in cache)


def benchmark_model(model, data, device, settings, training_settings, loss_settings):
    """Measure synchronized FP32 wall time; report the scope and actual device.

    Optional training throughput uses forward+backward without optimizer updates.
    Inputs repeat one real episode window to the configured batch size. This is a
    compute benchmark, not an additional training run or a control-frequency claim.
    """
    estimator = StreamingInsertionEstimator(model)
    arguments = stream_arguments(data, 0, device)
    estimator.reset("benchmark")
    step = lambda: estimator.step(arguments)
    timing = measure_callback(
        step, device, settings["latency_warmup_steps"], settings["latency_steps"]
    )
    state = estimator.episodes["benchmark"]
    memory = cache_bytes(state["cache"])
    reset = measure_callback(
        lambda: estimator.reset("benchmark"),
        device,
        settings["latency_warmup_steps"],
        settings["latency_steps"],
    )
    # reset() releases history immediately; allocation occurs lazily at first step.
    cold_step = measure_callback(
        lambda: (estimator.reset("benchmark"), estimator.step(arguments)),
        device,
        settings["latency_warmup_steps"],
        settings["latency_steps"],
    )
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "precision": "float32",
        "backend": model.backend,
        "streaming_step": timing,
        "episode_reset": reset,
        "reset_plus_first_step": cold_step,
        "cache_bytes_batch_one": memory,
        "cache_history_frames": (
            int(state["cache"][0].shape[1]) if model.architecture == "transformer" else None
        ),
        "scope": "feature construction + temporal model + geometry; excludes Frozen GNN, IO, simulation",
        "training_throughput": None,
    }
    observations, target = InsertionEpisodeDataset(data)[0]
    length = min(len(observations), training_settings["sequence_length"])
    batch_size = training_settings["episode_batch_size"]
    observations = observations[:length][None].repeat(batch_size, 1, 1).to(device)
    target = target[:length][None].repeat(batch_size, 1, 1).to(device)
    mask = torch.ones((batch_size, length), device=device, dtype=torch.bool)

    @torch.inference_mode()
    def sequence_forward():
        return model.forward_sequence(observations)

    sequence_timing = measure_callback(
        sequence_forward, device, settings["latency_warmup_steps"], settings["latency_steps"]
    )
    sequence_timing.update(batch_size=batch_size, sequence_length=length)
    result["sequence_forward"] = sequence_timing
    if settings["benchmark_training"]:

        def backward():
            model.zero_grad(set_to_none=True)
            prediction = model(observations)
            loss, _ = insertion_loss(prediction, target, mask, model.mode, loss_settings)
            loss.backward()

        model.train()
        try:
            for _ in range(settings["latency_warmup_steps"]):
                backward()
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = perf_counter()
            for _ in range(settings["latency_steps"]):
                backward()
            synchronize(device)
            duration = perf_counter() - started
            result["training_throughput"] = {
                "scope": "forward+loss+backward, no optimizer step",
                "batch_size": batch_size,
                "sequence_length": length,
                "steps": settings["latency_steps"],
                "warmup_steps": settings["latency_warmup_steps"],
                "frames_per_second": batch_size * length * settings["latency_steps"] / duration,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
            }
        finally:
            model.zero_grad(set_to_none=True)
            model.eval()
    return result


def evaluate(config):
    """Replay configured baselines; keep final independent testing an explicit action."""
    effective = resolve_insertion_config(config)
    device = set_device(effective["device"])
    split = effective["evaluation"]["split"]
    frozen, loaders, identity = prepare_insertion_sequences(effective, device, (split,))
    primary_path = Path(effective["temporal_checkpoint"]).expanduser().resolve()
    primary, payload = load_insertion_model(primary_path, effective, identity, device)
    if payload["config"]["model"] != effective["model"] or primary.mode != effective["mode"]:
        raise ValueError("Primary model settings/mode must match its training checkpoint")
    with ExperimentLogger(
        team="graph",
        model="temporal_insertion_" + primary.architecture,
        config=effective,
        seed=effective["seed"],
        job_type="evaluate",
    ) as experiment:
        output = Path(effective["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        data = infer_insertion_data(frozen, loaders[split], device, identity, effective, split)
        predictions = {
            "gnn": data["prediction"],
            "first_observation_hold": first_observation_hold(data),
        }
        predictions["rule"] = replay_memory(
            data["prediction"],
            data["tcp_world"],
            data["quality"],
            data["episode"],
            data["frame"],
            MemorySettings(**effective["rule_memory"]),
        )["object21"]
        geometric = data["prediction"].copy()
        geometric[:, 7:10] = geometric_endpoint(torch.tensor(geometric)).numpy()
        predictions["gnn_geometric_endpoint"] = geometric
        models = {"primary": primary}
        paths = {"primary": primary_path}
        protocols = {"primary": payload["config"]}
        for name, path in effective["comparison"]["checkpoints"].items():
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or name in predictions
                or name in models
            ):
                raise ValueError(f"Comparison names must be unique identifiers: {name}")
            model, compared = load_insertion_model(path, effective, identity, device)
            models[name], paths[name], protocols[name] = model, Path(path), compared["config"]
        geometry, streaming = {}, {}
        for name, model in models.items():
            predictions[name], geometry[name], streaming[name] = replay_insertion(
                model, data, device
            )
        unavailable = []
        gate_path = effective["comparison"]["learned_gate"]
        if gate_path:
            gate_payload = torch.load(gate_path, weights_only=True, map_location="cpu")
            check_development_provenance(gate_payload, effective, identity)
            if gate_payload.get("kind") != "learned_temporal_gate" or gate_payload["config"][
                "feature_names"
            ] != list(QUALITY_NAMES):
                raise ValueError("Expected the existing Learned Gate quality feature contract")
            gate = LearnedTemporalGate(**gate_payload["config"]["gate"]).to(device)
            gate.load_state_dict(gate_payload["gate"], strict=True)
            predictions["learned_gate"] = replay_gate(gate, data, device)["object21"]
            paths["learned_gate"] = Path(gate_path)
        else:
            unavailable.append("learned_gate")
        if not any(model.architecture == "gru" for model in models.values()):
            unavailable.append("gru_same_features")
        # Labels are consulted only after every method has completed causal replay.
        summary, errors = score_insertion_methods(data, predictions, effective["evaluation"])
        summary.update(
            split=split,
            test_role=effective["temporal_split_manifest"]["test_role"],
            independent_test=split == "test"
            and effective["temporal_split_manifest"]["test_role"] == "independent",
            streaming_max_difference=streaming,
            unavailable_comparisons=unavailable,
            geometry_frame_definition="lateral/axial estimation error uses GT Hole axes; control geometry uses predicted axes",
        )
        fairness = {}
        for name, trained in protocols.items():
            matches = {
                key: trained[key] == payload["config"][key]
                for key in ("seed", "training", "loss", "normalization")
            }
            # Loss ablations deliberately differ; report the protocol rather than
            # silently presenting a different loss or training budget as matched.
            fairness[name] = {
                "matches_primary": matches,
                "mode": trained["mode"],
                "model": trained["model"],
                "loss": trained["loss"],
                "selection": "minimum validation weighted objective within each run",
            }
        summary["comparison_protocols"] = fairness
        summary["performance"] = {
            name: benchmark_model(
                model,
                data,
                device,
                effective["evaluation"],
                protocols[name]["training"],
                protocols[name]["loss"],
            )
            for name, model in models.items()
        }
        np.savez_compressed(
            output / "predictions.npz",
            **data,
            **{name + "_object21": values for name, values in predictions.items()},
            **{name + "_predicted_hole_frame": values for name, values in geometry.items()},
        )
        save_pose_plots(output, data, errors, effective["evaluation"]["plot_episodes"])
        write_json(output / "summary.json", summary)
        write_json(output / "effective_config.json", effective)
        write_json(
            output / "provenance.json",
            {
                "source_checkpoint_sha256": effective["source_checkpoint_sha256"],
                "split_manifest_sha256": effective["split_manifest_sha256"],
                "independent_dataset_sha256": identity.get("independent_dataset_sha256"),
                "checkpoints": {
                    name: {"path": str(path), "sha256": file_sha256(path)}
                    for name, path in paths.items()
                },
                "diagnostic_labels_sha256": {
                    name: file_sha256(effective["evaluation"][name])
                    if effective["evaluation"][name]
                    else None
                    for name in ("phase_labels", "visibility_labels")
                },
            },
        )
        experiment.link_output(effective["checkpoint"], checkpoint=True)
        for path in paths.values():
            experiment.link_output(path, checkpoint=True)
        for name, values in summary["methods"].items():
            experiment.log(values["groups"]["all"], prefix=name)
        experiment.record(
            dataset_sha256=identity["dataset_sha256"],
            independent_dataset_sha256=identity.get("independent_dataset_sha256"),
            performance=summary["performance"],
            device=str(device),
            split=split,
            independent_test=summary["independent_test"],
        )
        for path in list(output.iterdir()):
            experiment.link_output(path)
        print(f"Insertion precision evaluation: {output}", flush=True)
        return output


def load_comparison_reference(reference_dir):
    """Validate a saved Mamba evaluation before comparing an existing GRU."""
    required = ("summary.json", "effective_config.json", "provenance.json", "predictions.npz")
    for name in required:
        if not (reference_dir / name).is_file():
            raise FileNotFoundError(f"Missing reference evaluation file: {reference_dir / name}")
    reference = json.loads((reference_dir / "effective_config.json").read_text())
    reference_summary = json.loads((reference_dir / "summary.json").read_text())
    provenance = json.loads((reference_dir / "provenance.json").read_text())
    if reference_summary["split"] != "validation":
        raise ValueError(
            "This comparison requires a development-validation reference, not test data"
        )
    primary_path = Path(reference["temporal_checkpoint"])
    if file_sha256(primary_path) != provenance["checkpoints"]["primary"]["sha256"]:
        raise ValueError("Reference Mamba checkpoint has changed since its evaluation")
    if file_sha256(reference["checkpoint"]) != reference["source_checkpoint_sha256"]:
        raise ValueError("Reference Frozen GNN checkpoint has changed")
    primary = torch.load(primary_path, weights_only=True, map_location="cpu")
    if (
        primary.get("kind") != "temporal_insertion"
        or primary["config"]["model"]["architecture"] != "mamba"
    ):
        raise ValueError("The reference must contain a temporal_insertion Mamba checkpoint")
    identity = {"dataset_sha256": primary["dataset_sha256"]}
    check_development_provenance(primary, reference, identity)
    return reference, reference_summary, primary, identity


def compare_existing_gru(config):
    """Remeasure a preserved GRU on an existing Mamba validation frame snapshot.

    All four systems share exact Frozen GNN outputs, targets, sample order,
    episode resets, endpoint sign, Hole axes and frame-weighted scoring. Each
    trained temporal model retains its own input schema and train-only statistics.
    This compares completed systems; unequal training settings remain explicit.
    """
    reference_dir = Path(config["reference_evaluation"]).expanduser().resolve()
    gru_path = Path(config["gru_checkpoint"]).expanduser().resolve()
    if not gru_path.is_file():
        raise FileNotFoundError(f"Missing existing GRU checkpoint: {gru_path}")
    reference, reference_summary, primary, identity = load_comparison_reference(reference_dir)
    device = set_device(config.get("device", "cpu"))
    fraction = config.get("cuda_memory_fraction", 0.4)
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError("cuda_memory_fraction must be a number in (0,1]")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(fraction), device)
    model, payload = load_existing_pose_gru(gru_path, reference, identity, device)
    fields = (
        "sample_ids",
        "episode",
        "frame",
        "prediction",
        "target",
        "tcp_world",
        "features",
        "quality",
        "pose_quality",
        "tcp_quaternion",
        "gripper_qpos",
    )
    with np.load(reference_dir / "predictions.npz", allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in fields if name in saved}
        predictions = {
            name: saved[field].copy()
            for name, field in (
                ("gnn", "gnn_object21"),
                ("learned_gate", "learned_gate_object21"),
                ("mamba", "primary_object21"),
            )
        }
    # Reuse complete-episode validation before giving any observations to the GRU.
    dataset = InsertionEpisodeDataset(data)
    if list(dataset.feature_names) != reference["feature_names"]:
        raise ValueError("Reference archive features differ from its recorded Mamba schema")
    expected_episodes = set(reference["temporal_split_manifest"]["splits"]["validation"])
    if (
        set(dataset.episodes) != expected_episodes
        or len(data["frame"]) != reference_summary["frames"]
        or len(dataset) != reference_summary["episodes"]
    ):
        raise ValueError("Reference frames do not match the development-validation split")
    trained = payload["config"]
    observed = existing_gru_data_view(data, model.with_robot)
    print(
        f"Replaying existing GRU on {len(dataset)} episodes / {len(data['frame'])} identical frames",
        flush=True,
    )
    # CuDNN's default TF32 path can differ between sequence and one-step calls.
    # Use FP32 recurrent kernels so the existing strict streaming check applies.
    with torch.backends.cudnn.flags(allow_tf32=False):
        predictions["gru"], difference = replay_pose(model, observed, device)
    summary, _ = score_insertion_methods(data, predictions, reference["evaluation"])
    metrics = (
        "peg_position",
        "peg_rotation",
        "hole_position",
        "hole_rotation",
        "relative_3d",
        "lateral",
        "axial",
    )
    for name, source_name in (
        ("gnn", "gnn"),
        ("learned_gate", "learned_gate"),
        ("mamba", "primary"),
    ):
        for metric in metrics:
            unit = "deg" if "rotation" in metric else "mm"
            current = summary["methods"][name]["groups"]["all"][metric]["mean_" + unit]
            previous = reference_summary["methods"][source_name]["groups"]["all"][metric][
                "mean_" + unit
            ]
            np.testing.assert_allclose(current, previous, atol=1e-6, rtol=1e-6)
    primary_path = Path(reference["temporal_checkpoint"])
    primary_config = primary["config"]
    gru_context = {
        "policy": trained.get("sequence_contract"),
        "sequence_length": trained["training"].get("sequence_length"),
    }
    mamba_context = {
        "policy": primary_config["training"].get("context_policy"),
        "sequence_length": primary_config["training"].get("sequence_length"),
    }
    audit = {
        "same_frozen_checkpoint": True,
        "same_dataset": True,
        "same_development_parents": True,
        "same_frame_ids_order_targets_and_gnn_outputs": True,
        "same_world_wxyz_endpoint_minus_hole_geometry": True,
        "same_full_episode_causal_inference_and_reset": True,
        "same_frame_weighted_scoring": True,
        "normalization_source": "train for each model",
        "training_features_equal": trained["feature_names"] == primary_config["feature_names"],
        "training_loss_equal": trained["loss"] == primary_config["loss"],
        "training_context_equal": gru_context == mamba_context,
        "training_contexts": {"gru": gru_context, "mamba": mamba_context},
        "controlled_backbone_comparison": False,
        "scope": "completed trained systems on identical validation frames",
        "gru": {
            "checkpoint": str(gru_path),
            "epoch": payload["epoch"],
            "feature_count": len(trained["feature_names"]),
            "model": trained["gru"],
            "loss": trained["loss"],
            "training": trained["training"],
            "sequence_contract": trained.get("sequence_contract"),
            "selection": trained.get("selection", {"criterion": "validation_loss"}),
            "selection_score": payload.get("selection_score"),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "mamba": {
            "checkpoint": str(primary_path),
            "epoch": primary["epoch"],
            "feature_count": len(primary_config["feature_names"]),
            "model": primary_config["model"],
            "loss": primary_config["loss"],
            "training": primary_config["training"],
            "selection": {"criterion": "minimum validation weighted loss"},
        },
    }
    summary.update(
        split="validation",
        independent_test=False,
        comparison_audit=audit,
        gru_streaming_max_difference=difference,
        source_mamba_streaming_max_difference=reference_summary["streaming_max_difference"][
            "primary"
        ],
    )
    reference_files = (
        "summary.json",
        "effective_config.json",
        "provenance.json",
        "predictions.npz",
    )
    reference_hashes = {name: file_sha256(reference_dir / name) for name in reference_files}
    checkpoint_hashes = {"gru": file_sha256(gru_path), "mamba": file_sha256(primary_path)}
    frame_hash = hashlib.sha256("\n".join(data["sample_ids"].tolist()).encode()).hexdigest()
    repository = Path(__file__).resolve().parents[3]
    source_files = (
        "evaluation/gnn/temporal/insertion.py",
        "evaluation/gnn/temporal/pose_gru.py",
        "evaluation/gnn/temporal/precision.py",
        "graph/temporal/pose_gru.py",
        "graph/pose_geometry.py",
        "graph/temporal/pose_observations.py",
        "data_collection/temporal/pose_sequences.py",
        "data_collection/temporal/frozen_episodes.py",
    )
    effective = dict(
        config,
        reference_evaluation=str(reference_dir),
        gru_checkpoint=str(gru_path),
        comparison_audit=audit,
        reference_files_sha256=reference_hashes,
        checkpoint_sha256=checkpoint_hashes,
        sample_ids_sha256=frame_hash,
        gru_cudnn_allow_tf32=False,
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        source_sha256={name: file_sha256(repository / name) for name in source_files},
    )
    with ExperimentLogger(
        team="graph",
        model="gru_mamba_validation_comparison",
        config=effective,
        seed=reference["seed"],
        job_type="evaluate",
    ) as experiment:
        output = Path(config["output"]["root"]).expanduser().resolve() / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "summary.json", summary)
        write_json(output / "effective_config.json", effective)
        np.savez_compressed(
            output / "predictions.npz",
            sample_ids=data["sample_ids"],
            **{name + "_object21": values for name, values in predictions.items()},
        )
        experiment.link_output(gru_path, checkpoint=True)
        experiment.link_output(primary_path, checkpoint=True)
        experiment.record(
            dataset_sha256=identity["dataset_sha256"],
            sample_ids_sha256=frame_hash,
            checkpoint_sha256=checkpoint_hashes,
            comparison_scope=audit["scope"],
            controlled_backbone_comparison=False,
            device=str(device),
        )
        for name, method in summary["methods"].items():
            experiment.log(method["groups"]["all"], prefix=name)
        for path in list(output.iterdir()):
            experiment.link_output(path)
        print(f"GRU/Mamba validation comparison: {output}", flush=True)
        return output
