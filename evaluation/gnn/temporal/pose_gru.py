"""Paired pose, occlusion, motion-lag and streaming evaluation on reserved episodes."""

from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from data_collection.temporal.pose_sequences import (
    TemporalPoseDataset,
    stream_arguments,
)
from evaluation.gnn.temporal.hole_gru import (
    absence_diagnostics,
    measure_callback,
    replay_gru,
    temporal_metrics,
)
from evaluation.gnn.temporal.precision import (
    motion_precision_groups,
    precision_diagnostics,
)
from evaluation.gnn.temporal_gate import replay_gate
from evaluation.gnn.temporal_memory import error_summary, replay_memory, scoring_groups
from graph.pose_geometry import (
    angular_error_rad,
    conjugate,
    geometric_endpoint,
    quaternion_product,
)
from graph.temporal.hole_gru import StreamingHoleEstimator, TemporalHoleGRU
from graph.temporal.observations import (
    OBSERVATION_CONTRACT,
    OBSERVATION_NAMES,
    QUALITY_NAMES,
)
from graph.temporal.pose_gru import StreamingPoseEstimator, TemporalPoseGRU
from graph.temporal_gate import LearnedTemporalGate, correct_object21
from graph.utils.hole_memory import (
    HolePositionMemory,
    MemorySettings,
    ObservationQuality,
)
from training.gnn.temporal.config import resolve_pose_config
from training.gnn.temporal.frozen_source import infer_pose_data, prepare_pose_sequences
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


def check_provenance(payload, effective, identity):
    """Reject comparisons across perception checkpoints, parent splits or datasets."""
    for name in ("source_checkpoint_sha256", "split_manifest_sha256"):
        if payload["config"][name] != effective[name]:
            if (
                name != "split_manifest_sha256"
                or effective["temporal_split_manifest"].get("test_role") != "independent"
            ):
                raise ValueError(f"Comparison provenance mismatch: {name}")
            trained = payload["config"].get(
                "temporal_split_manifest", payload["config"].get("gate_split_manifest")
            )
            expected = effective["temporal_split_manifest"]
            if trained is None:
                raise ValueError(
                    "Independent comparison needs actual development parent provenance"
                )
            for split in ("train", "validation"):
                if set(trained["splits"][split]) != set(expected["splits"][split]):
                    raise ValueError(f"Comparison development {split} split differs")
                for episode in expected["splits"][split]:
                    if (
                        trained["parent_by_episode"][episode]
                        != expected["parent_by_episode"][episode]
                    ):
                        raise ValueError(f"Comparison development parent differs: {episode}")
    if payload["dataset_sha256"] != identity["dataset_sha256"]:
        raise ValueError("Comparison dataset identity differs")


def load_pose_model(path, effective, identity, device):
    """Load exact mode/schema/normalization; do not fit or adapt on evaluation data."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "temporal_pose_gru":
        raise ValueError("Expected a temporal_pose_gru checkpoint")
    check_provenance(payload, effective, identity)
    trained = payload["config"]
    for name in ("feature_names", "observation_contract", "with_robot"):
        if trained[name] != effective[name]:
            raise ValueError(f"Comparison observation mismatch: {name}")
    model = TemporalPoseGRU(
        mode=trained["mode"], with_robot=trained["with_robot"], **trained["gru"]
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    expected = trained["normalization"]
    if expected["source_split"] != "train":
        raise ValueError("Normalization must come from the temporal train split")
    torch.testing.assert_close(model.input_mean.cpu(), torch.tensor(expected["mean"]))
    torch.testing.assert_close(model.input_std.cpu(), torch.tensor(expected["std"]))
    return model.eval(), payload


@torch.inference_mode()
def replay_pose(model, data, device):
    """Replay every episode and verify pose outputs/final hidden against full forward."""
    dataset = TemporalPoseDataset(data)
    estimator = StreamingPoseEstimator(model)
    outputs, maximum = [], 0.0
    for index, rows in enumerate(dataset.indices):
        observations, _ = dataset[index]
        full, last_hidden = model(observations.to(device)[None])
        estimator.reset()
        streamed = []
        for row in rows:
            result = estimator.step(**stream_arguments(data, row, device))
            outputs.append(result["object21"].cpu().numpy())
            streamed.append(result["world_object21"])
        streamed = torch.stack(streamed, 1)
        torch.testing.assert_close(streamed, full, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(estimator.hidden, last_hidden, atol=1e-5, rtol=1e-4)
        maximum = max(maximum, float((streamed - full).abs().max()))
    return np.concatenate(outputs), maximum


def first_observation_hold(data):
    """Hold the initial GNN Hole world pose per episode, with no GT initialization."""
    output = torch.tensor(data["prediction"])
    tcp = torch.tensor(data["tcp_world"])
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        initial = rows[0]
        world = output[initial, 10:13] + tcp[initial]
        corrected = correct_object21(output[rows], world - tcp[rows])
        corrected[:, 13:17] = output[initial, 13:17]
        output[rows] = corrected
    return output.numpy()


def scalar_summary(values, unit):
    """Reuse frame-weighted summary with explicit rotation units in field names."""
    summary = error_summary(values)
    return {key.replace("_mm", "_" + unit): value for key, value in summary.items()}


def pose_errors(prediction, target):
    """Compute all five GT metrics after causal inference, without symmetry assumptions."""
    errors = {}
    for name, start in (("peg", 0), ("hole", 10)):
        errors[name + "_position"] = (
            np.linalg.norm(prediction[:, start : start + 3] - target[:, start : start + 3], axis=1)
            * 1000
        )
        errors[name + "_rotation"] = torch.rad2deg(
            angular_error_rad(
                torch.tensor(prediction[:, start + 3 : start + 7]),
                torch.tensor(target[:, start + 3 : start + 7]),
            )
        ).numpy()
    errors["endpoint"] = np.linalg.norm(prediction[:, 7:10] - target[:, 7:10], axis=1) * 1000
    return errors


def load_phase_groups(path, data):
    """Read supplied actual phase labels solely for scoring; never infer grasp from GT."""
    if path is None:
        return {}
    with np.load(path, allow_pickle=False) as labels:
        ids, phases = labels["sample_ids"].tolist(), labels["phase"].copy()
    if (
        len(set(ids)) != len(ids)
        or set(ids) != set(data["sample_ids"])
        or phases.shape != (len(ids),)
    ):
        raise ValueError("Phase labels must list exactly the evaluation frames")
    mapping = {sample: index for index, sample in enumerate(ids)}
    phases = phases[[mapping[str(sample)] for sample in data["sample_ids"]]]
    return {f"phase_{phase}": phases == phase for phase in np.unique(phases)}


def diagnostic_groups(data, settings):
    """Define GT motion and observation-absence slices; none controls learned updates."""
    groups = scoring_groups(data, settings["visibility_labels"])
    moving = np.zeros(len(data["frame"]), dtype=bool)
    valid = np.zeros_like(moving)
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        target = data["target"][rows]
        world = target[:, :3] + data["tcp_world"][rows]
        translation = np.linalg.norm(np.diff(world, axis=0), axis=1)
        rotation = angular_error_rad(
            torch.tensor(target[1:, 3:7]), torch.tensor(target[:-1, 3:7])
        ).numpy()
        valid[rows[1:]] = True
        moving[rows[1:]] = (translation > settings["motion_threshold_m"]) | (
            rotation > settings["motion_threshold_rad"]
        )
    groups["peg_moving"], groups["peg_stationary"] = moving, valid & ~moving
    groups["peg_missing"] = data["pose_quality"][:, 0] == 0
    groups["hole_missing"] = data["pose_quality"][:, 16] == 0
    groups["target_support_missing"] = data["quality"][:, 0] == 0
    events = {}
    for name in ("peg_missing", "hole_missing", "target_support_missing", "strong_occlusion_proxy"):
        if name in groups:
            age, long_run, runs = absence_diagnostics(
                data, groups[name], settings["long_occlusion_frames"]
            )
            groups[name + "_long_run"] = long_run
            groups[name + "_long_age"] = age >= settings["long_occlusion_frames"]
            events[name] = runs
    groups.update(load_phase_groups(settings["phase_labels"], data))
    precision_settings = dict(
        co_motion_tolerance_m=0.002,
        direction_change_cosine=0.0,
        alignment_axial_min_m=-0.05,
        alignment_axial_max_m=0.02,
        alignment_lateral_m=0.01,
        **{name: settings[name] for name in ("motion_threshold_m",)},
    )
    precision_settings.update(settings)
    groups.update(motion_precision_groups(data, precision_settings))
    return groups, events


def lag_and_drift(data, prediction, moving, maximum_lag):
    """Estimate translational lag using a common interior window for all lag candidates.

    Positive lag means predictions resemble earlier GT positions. Future GT is
    used only here, after inference, to also detect lead (negative lag). Short or
    static episodes have an explicit null lag. Quaternion dynamics are scored
    separately via angular errors and the motion groups.
    """
    results = []
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        world = prediction[rows, :3] + data["tcp_world"][rows]
        target = data["target"][rows, :3] + data["tcp_world"][rows]
        margin = min(maximum_lag, max(0, (len(rows) - 2) // 3))
        times = np.arange(margin, len(rows) - margin)
        active = times[moving[rows[times]]]
        translates = np.linalg.norm(target - target[0], axis=1).max() > 1e-8
        scores = (
            {
                lag: float(
                    np.linalg.norm(world[active] - target[active - lag], axis=1).mean() * 1000
                )
                for lag in range(-margin, margin + 1)
            }
            if margin and len(active) and translates
            else {}
        )
        best = min(scores, key=lambda lag: (scores[lag], abs(lag))) if scores else None
        target_q = torch.tensor(data["target"][rows, 3:7])
        predicted_q = torch.tensor(prediction[rows, 3:7])
        rotates = float(angular_error_rad(target_q, target_q[:1]).max()) > 1e-7
        rotation_scores = (
            {
                lag: float(
                    torch.rad2deg(
                        angular_error_rad(predicted_q[active], target_q[active - lag])
                    ).mean()
                )
                for lag in range(-margin, margin + 1)
            }
            if margin and len(active) and rotates
            else {}
        )
        rotation_best = (
            min(rotation_scores, key=lambda lag: (rotation_scores[lag], abs(lag)))
            if rotation_scores
            else None
        )
        displacement_error = (world - world[0]) - (target - target[0])
        results.append(
            {
                "episode": str(episode),
                "lag_frames": best,
                "lag_compensated_error_mm": scores[best] if best is not None else None,
                "lag_scores_mm": {str(key): value for key, value in scores.items()},
                "rotation_lag_frames": rotation_best,
                "rotation_lag_scores_deg": {
                    str(key): value for key, value in rotation_scores.items()
                },
                "displacement_drift": error_summary(
                    np.linalg.norm(displacement_error, axis=1) * 1000
                ),
            }
        )
    return results


def pose_recovery(data, errors, absent, settings):
    """Report pose correction after missing runs, requiring position AND rotation recovery."""
    events = []
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        local = absent[rows]
        for start in np.flatnonzero(local[:-1] & ~local[1:]) + 1:
            missing_start = start - 1
            while missing_start > 0 and local[missing_start - 1]:
                missing_start -= 1
            stop = start + 1
            while stop < len(rows) and not local[stop]:
                stop += 1
            segment = rows[start:stop]
            recovered = segment[
                (errors[0][segment] <= settings["bad_error_mm"])
                & (errors[1][segment] <= settings["recovery_rotation_deg"])
            ]
            events.append(
                {
                    "episode": str(episode),
                    "return_frame": int(data["frame"][rows[start]]),
                    "missing_length_frames": int(start - missing_start),
                    "long_occlusion": bool(
                        start - missing_start >= settings["long_occlusion_frames"]
                    ),
                    "before_position_mm": float(errors[0][rows[start - 1]]),
                    "before_rotation_deg": float(errors[1][rows[start - 1]]),
                    "return_position_mm": float(errors[0][rows[start]]),
                    "return_rotation_deg": float(errors[1][rows[start]]),
                    "end_position_mm": float(errors[0][segment[-1]]),
                    "end_rotation_deg": float(errors[1][segment[-1]]),
                    "recovery_latency_frames": int(
                        data["frame"][recovered[0]] - data["frame"][rows[start]]
                    )
                    if len(recovered)
                    else None,
                }
            )
    return events


def hole_rotation_drift(data, prediction):
    """Compare change from initial Hole orientation to GT change, after inference."""
    results = []
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        predicted = torch.tensor(prediction[rows, 13:17])
        target = torch.tensor(data["target"][rows, 13:17])
        predicted_change = quaternion_product(predicted, conjugate(predicted[:1]))
        target_change = quaternion_product(target, conjugate(target[:1]))
        error = torch.rad2deg(angular_error_rad(predicted_change, target_change)).numpy()
        results.append(
            {"episode": str(episode), "orientation_change_error": scalar_summary(error, "deg")}
        )
    return results


def score_pose_methods(data, predictions, settings):
    """Score identical frames for all methods and save per-episode breakdowns."""
    groups, runs = diagnostic_groups(data, settings)
    first = np.r_[True, data["episode"][1:] != data["episode"][:-1]]
    groups["first_frame"] = first
    summary = {
        "frames": len(data["frame"]),
        "episodes": len(np.unique(data["episode"])),
        "split": "temporal_test_original_gnn_validation",
        "phase_labels_available": settings["phase_labels"] is not None,
        "visibility_labels_available": settings["visibility_labels"] is not None,
        "methods": {},
        "missing_runs": runs,
    }
    hole_translation, hole_rotation = [], []
    for episode in np.unique(data["episode"]):
        rows = np.flatnonzero(data["episode"] == episode)
        world = data["target"][rows, 10:13] + data["tcp_world"][rows]
        hole_translation.append(float(np.linalg.norm(world - world[0], axis=1).max()))
        quaternion = torch.tensor(data["target"][rows, 13:17])
        hole_rotation.append(
            float(torch.rad2deg(angular_error_rad(quaternion, quaternion[:1])).max())
        )
    summary["hole_gt_max_translation_m"] = max(hole_translation)
    summary["hole_gt_max_rotation_deg"] = max(hole_rotation)
    all_errors, recovery = {}, {}
    for name, prediction in predictions.items():
        errors = pose_errors(prediction, data["target"])
        all_errors[name] = errors

        def summarize(mask):
            return {
                metric: scalar_summary(values[mask], "deg" if "rotation" in metric else "mm")
                for metric, values in errors.items()
            }

        method = {
            "groups": {group: summarize(mask) for group, mask in groups.items()},
            "episodes": {
                str(episode): summarize(data["episode"] == episode)
                for episode in np.unique(data["episode"])
            },
            "peg_dynamics": lag_and_drift(
                data, prediction, groups["peg_moving"], settings["lag_max_frames"]
            ),
            "hole_orientation_drift": hole_rotation_drift(data, prediction),
        }
        world = prediction[:, 10:13] + data["tcp_world"]
        method["hole_temporal"] = temporal_metrics(
            data,
            world,
            errors["hole_position"],
            np.linalg.norm(data["prediction"][:, 10:13] - data["target"][:, 10:13], axis=1) * 1000,
            settings["bad_error_mm"],
        )
        method["insertion_precision"], _ = precision_diagnostics(data, prediction, groups)
        recovery[name] = {
            kind: pose_recovery(
                data,
                (errors[kind + "_position"], errors[kind + "_rotation"]),
                groups[kind + "_missing"],
                settings,
            )
            for kind in ("peg", "hole")
        }
        recovery[name]["hole_support"] = pose_recovery(
            data,
            (errors["hole_position"], errors["hole_rotation"]),
            groups["target_support_missing"],
            settings,
        )
        summary["methods"][name] = method
    if "gnn" in summary["methods"]:
        baseline = summary["methods"]["gnn"]["insertion_precision"]["groups"]["all"]["errors"]
        for method in summary["methods"].values():
            measured = method["insertion_precision"]["groups"]["all"]["errors"]
            method["absolute_pose_change_vs_frozen"] = {
                name: {
                    "mean_change": measured[name]["mean"] - baseline[name]["mean"],
                    "mean_ratio": measured[name]["mean"] / baseline[name]["mean"]
                    if baseline[name]["mean"]
                    else None,
                }
                for name in (
                    "peg_position_mm",
                    "hole_position_mm",
                    "peg_rotation_deg",
                    "hole_rotation_deg",
                )
            }
    return summary, all_errors, recovery


def save_pose_plots(output, data, errors, maximum):
    """Plot pose errors and observation support without mixing metre/degree axes."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for number, episode in enumerate(np.unique(data["episode"])[:maximum]):
        rows = data["episode"] == episode
        frames = data["frame"][rows]
        figure, axes = plt.subplots(3, 2, sharex=True, figsize=(14, 9))
        for column, kind in enumerate(("peg", "hole")):
            for name, values in errors.items():
                axes[0, column].plot(frames, values[kind + "_position"][rows], label=name)
                axes[1, column].plot(frames, values[kind + "_rotation"][rows], label=name)
            axes[0, column].set_title(f"{episode}: {kind}")
            axes[0, column].set_ylabel("Position error (mm)")
            axes[1, column].set_ylabel("Angular error (deg)")
            count = data["pose_quality"][rows, 0 if kind == "peg" else 16]
            axes[2, column].step(frames, count, where="post")
            axes[2, column].set_ylabel("Observed vertices")
            axes[2, column].set_xlabel("Frame")
            for axis in axes[:, column]:
                axis.fill_between(
                    frames,
                    0,
                    1,
                    where=count == 0,
                    transform=axis.get_xaxis_transform(),
                    color="gray",
                    alpha=0.12,
                )
        axes[0, 0].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output / f"pose_trace_{number:03d}.png", dpi=130)
        plt.close(figure)


@torch.inference_mode()
def benchmark_baselines(frozen, sample, gate, legacy_model, data, device, effective):
    """Record baseline parameters/latency with component/device scope made explicit."""
    settings = effective["evaluation"]
    graph = sample.graph.to(device)
    previous, first = None, None
    rule = HolePositionMemory(MemorySettings(**effective["rule_memory"]))
    quality = ObservationQuality(
        int(data["quality"][0, 0]), float(data["quality"][0, 1]), float(data["quality"][0, 2])
    )
    prediction = torch.tensor(data["prediction"][:1], device=device)
    tcp = torch.tensor(data["tcp_world"][:1], device=device)
    features = torch.tensor(data["features"][:1], device=device)
    frame = 0

    def gate_step():
        nonlocal previous
        previous, _ = gate.step(prediction[0, 10:13], tcp[0], features[0], previous)

    def rule_step():
        nonlocal frame
        rule.step(frame, data["prediction"][0, 10:13], data["tcp_world"][0], quality)
        frame += 1

    def hold_step():
        nonlocal first
        if first is None:
            first = data["prediction"][0, 10:13] + data["tcp_world"][0]
        return first - data["tcp_world"][0]

    callbacks = {
        "gnn": (
            lambda: frozen(graph),
            device,
            sum(p.numel() for p in frozen.parameters()),
            "frozen graph forward, resident batch-one graph",
        ),
        "rule": (rule_step, torch.device("cpu"), 0, "CPU memory position update only"),
        "first_observation_hold": (hold_step, torch.device("cpu"), 0, "CPU position hold only"),
    }
    if gate is not None:
        callbacks["learned_gate"] = (
            gate_step,
            device,
            sum(p.numel() for p in gate.parameters()),
            "gate step, excludes frozen GNN",
        )
    if legacy_model is not None:
        legacy = StreamingHoleEstimator(legacy_model)
        callbacks["gru_legacy_hole_only"] = (
            lambda: legacy.step(prediction, tcp, features),
            device,
            sum(p.numel() for p in legacy_model.parameters()),
            "legacy features/GRU/assembly, excludes frozen GNN",
        )
    result = {}
    for name, (callback, timing_device, parameters, scope) in callbacks.items():
        result[name] = {
            "parameters": parameters,
            "scope": scope,
            "latency": measure_callback(
                callback, timing_device, settings["latency_warmup_steps"], settings["latency_steps"]
            ),
        }
    return result


def collect_pose_predictions(model, payload, checkpoint, data, effective, identity, device):
    """Restore configured baselines and replay them before reading GT diagnostics."""
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
    models = {"gru_" + model.mode: model}
    primary = "gru_" + model.mode
    objectives = {primary: payload["config"]["loss"]}
    predictions[primary], difference = replay_pose(model, data, device)
    streaming = {primary: difference}
    source_paths = {primary: checkpoint}
    unavailable = []
    legacy_model = None
    comparison = effective["comparison"]
    for ablation, path in comparison.get("precision_checkpoints", {}).items():
        if not path:
            unavailable.append("ablation_" + ablation)
            continue
        estimator, compared = load_pose_model(path, effective, identity, device)
        if compared["config"]["loss"].get("ablation") != ablation:
            raise ValueError(f"Comparison loss ablation differs from {ablation}")
        if compared["config"]["mode"] != model.mode:
            raise ValueError("Loss-ablation comparisons must use the same pose mode")
        torch.testing.assert_close(estimator.input_mean, model.input_mean)
        torch.testing.assert_close(estimator.input_std, model.input_std)
        label = "ablation_" + ablation
        predictions[label], streaming[label] = replay_pose(estimator, data, device)
        models[label], source_paths[label] = estimator, Path(path)
        objectives[label] = compared["config"]["loss"]
    for mode, path in comparison.get("gru_checkpoints", {}).items():
        if not path:
            if "gru_" + mode not in predictions:
                unavailable.append("gru_" + mode)
            continue
        if "gru_" + mode in predictions:
            raise ValueError(f"Duplicate comparison mode: {mode}")
        if mode not in ("hole_only", "peg_only", "joint_peg_hole", "legacy_hole_only"):
            raise ValueError(f"Unknown comparison mode: {mode}")
        if mode == "legacy_hole_only":
            legacy = torch.load(path, weights_only=True, map_location="cpu")
            check_provenance(legacy, effective, identity)
            if legacy.get("kind") != "temporal_hole_gru":
                raise ValueError(
                    "legacy_hole_only requires the original temporal_hole_gru checkpoint"
                )
            if (
                legacy["config"]["feature_names"] != list(OBSERVATION_NAMES)
                or legacy["config"]["observation_contract"] != OBSERVATION_CONTRACT
            ):
                raise ValueError("Legacy Hole GRU observation contract differs")
            estimator = TemporalHoleGRU(**legacy["config"]["gru"]).to(device)
            estimator.load_state_dict(legacy["model"])
            legacy_model = estimator
            predictions["gru_" + mode] = replay_gru(estimator, data, device)["consistent_object21"]
        else:
            estimator, compared = load_pose_model(path, effective, identity, device)
            if compared["config"]["mode"] != mode:
                raise ValueError(f"Comparison checkpoint mode differs from {mode}")
            predictions["gru_" + mode], streaming["gru_" + mode] = replay_pose(
                estimator, data, device
            )
            models["gru_" + mode] = estimator
            objectives["gru_" + mode] = compared["config"]["loss"]
        source_paths["gru_" + mode] = Path(path)
    gate_path = comparison.get("learned_gate")
    gate = None
    if gate_path:
        gate_payload = torch.load(gate_path, weights_only=True, map_location="cpu")
        check_provenance(gate_payload, effective, identity)
        if gate_payload.get("kind") != "learned_temporal_gate":
            raise ValueError("comparison.learned_gate requires a learned_temporal_gate checkpoint")
        if gate_payload["config"]["feature_names"] != list(QUALITY_NAMES):
            raise ValueError("Learned gate quality feature contract differs")
        gate = LearnedTemporalGate(**gate_payload["config"]["gate"]).to(device)
        gate.load_state_dict(gate_payload["gate"])
        predictions["learned_gate"] = replay_gate(gate, data, device)["object21"]
        source_paths["learned_gate"] = Path(gate_path)
    else:
        unavailable.append("learned_gate")
    return {
        "predictions": predictions,
        "models": models,
        "objectives": objectives,
        "streaming": streaming,
        "source_paths": source_paths,
        "unavailable": unavailable,
        "legacy_model": legacy_model,
        "gate": gate,
    }


def evaluate(config):
    """Compare configured checkpoints; unavailable baselines stay explicitly unmeasured."""
    effective = resolve_pose_config(config)
    device = set_device(effective["device"])
    frozen, loaders, identity = prepare_pose_sequences(effective, device, ("test",))
    checkpoint = Path(effective["pose_checkpoint"]).expanduser().resolve()
    model, payload = load_pose_model(checkpoint, effective, identity, device)
    if (
        payload["config"]["mode"] != effective["mode"]
        or payload["config"]["gru"] != effective["gru"]
    ):
        raise ValueError("Primary evaluation mode/GRU settings must match its checkpoint")
    with ExperimentLogger(
        team="graph",
        model="temporal_pose_gru",
        config=effective,
        seed=effective["seed"],
        job_type="evaluate",
    ) as experiment:
        output = Path(effective["output"]["root"]) / experiment.run_id
        output.mkdir(parents=True, exist_ok=False)
        experiment.link_output(effective["checkpoint"], checkpoint=True)
        data = infer_pose_data(frozen, loaders["test"], device, identity, effective)
        comparisons = collect_pose_predictions(
            model, payload, checkpoint, data, effective, identity, device
        )
        predictions = comparisons["predictions"]
        models = comparisons["models"]
        objectives = comparisons["objectives"]
        streaming = comparisons["streaming"]
        source_paths = comparisons["source_paths"]
        unavailable = comparisons["unavailable"]
        legacy_model = comparisons["legacy_model"]
        gate = comparisons["gate"]
        # All outputs exist before GT motion, phase or visibility diagnostics are read.
        summary, errors, recovery = score_pose_methods(data, predictions, effective["evaluation"])
        role = effective["temporal_split_manifest"].get("test_role", "gnn_validation_reference")
        summary["split"] = (
            "independent_temporal_test"
            if role == "independent"
            else "reused_gnn_validation_reference"
        )
        summary["generalization_scope"] = (
            "new parents disjoint from every original GNN parent"
            if role == "independent"
            else "repeatedly inspected GNN validation; not evidence of final independent generalization"
        )
        summary["independent_dataset_sha256"] = identity.get("independent_dataset_sha256")
        summary.update(unavailable_comparisons=unavailable, streaming_max_difference=streaming)
        summary["training_objectives"] = objectives
        summary["fairness_contract"] = {
            "features": effective["feature_names"],
            "observation_contract": effective["observation_contract"],
            "sequence_contract": effective["sequence_contract"],
            "normalization_definition": effective["normalization_definition"],
            "precision_evaluation_contract": effective["precision_evaluation_contract"],
            "split_manifest_sha256": effective["split_manifest_sha256"],
            "source_checkpoint_sha256": effective["source_checkpoint_sha256"],
        }
        summary["primary_selection_rule"] = payload["config"].get(
            "selection", {"criterion": "validation loss"}
        )
        performance = {}
        for name, estimator_model in models.items():
            estimator = StreamingPoseEstimator(estimator_model)
            arguments = stream_arguments(data, 0, device)
            performance[name] = {
                "parameters": sum(p.numel() for p in estimator_model.parameters()),
                "latency": measure_callback(
                    lambda: estimator.step(**arguments),
                    device,
                    effective["evaluation"]["latency_warmup_steps"],
                    effective["evaluation"]["latency_steps"],
                ),
            }
        performance.update(
            benchmark_baselines(
                frozen, loaders["test"].dataset[0], gate, legacy_model, data, device, effective
            )
        )
        summary["performance"] = performance
        summary["performance_scope"] = (
            "batch-one resident pose feature construction/GRU/assembly; excludes frozen GNN and IO"
        )
        summary["gnn_geometric_endpoint_ablation"] = error_summary(
            np.linalg.norm(
                geometric_endpoint(torch.tensor(data["prediction"])).numpy()
                - data["target"][:, 7:10],
                axis=1,
            )
            * 1000
        )
        np.savez_compressed(
            output / "predictions.npz",
            **data,
            **{name + "_object21": values for name, values in predictions.items()},
        )
        geometry_arrays, precision_traces = {}, {}
        for name, values in predictions.items():
            _, per_frame = precision_diagnostics(
                data, values, {"all": np.ones(len(values), dtype=bool)}
            )
            precision_traces[name] = per_frame
            geometry_arrays.update(
                {name + "_" + field: array for field, array in per_frame.items()}
            )
        np.savez_compressed(
            output / "precision_errors.npz", sample_ids=data["sample_ids"], **geometry_arrays
        )
        from evaluation.gnn.temporal.precision import save_precision_plots

        save_precision_plots(
            output, data, precision_traces, effective["evaluation"]["plot_episodes"]
        )
        save_pose_plots(output, data, errors, effective["evaluation"]["plot_episodes"])
        write_json(output / "summary.json", summary)
        write_json(output / "recovery_events.json", recovery)
        write_json(output / "effective_config.json", effective)
        write_json(
            output / "provenance.json",
            {
                "checkpoints": {
                    name: {"path": str(path), "sha256": file_sha256(path)}
                    for name, path in source_paths.items()
                },
                "source_checkpoint_sha256": effective["source_checkpoint_sha256"],
                "phase_labels_sha256": file_sha256(effective["evaluation"]["phase_labels"])
                if effective["evaluation"]["phase_labels"]
                else None,
            },
        )
        for path in source_paths.values():
            experiment.link_output(path, checkpoint=True)
        experiment.link_output(checkpoint, checkpoint=True)
        for name, metrics in summary["methods"].items():
            experiment.log(metrics["groups"]["all"], prefix=name)
        experiment.record(
            dataset_sha256=identity["dataset_sha256"], device=str(device), performance=performance
        )
        for path in list(output.iterdir()):
            experiment.link_output(path, kind="metrics" if path.suffix == ".json" else None)
        print(f"Temporal pose comparison: {output}", flush=True)
        return output
