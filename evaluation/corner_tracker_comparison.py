"""Compare frozen surface-slot and online trackers on one physical-corner cohort.

The surface model's hash-pinned saved FP32 predictions are reused. Online
association remains detector-only. Certificates define evaluation targets and
causal scoring bindings, and never enter either model's inference inputs.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.spatial_temporal_corner_tracking import evaluate_online
from evaluation.spatial_temporal_held_out import validate_test_manifest
from graph.temporal.spatial_temporal_corner_tracker import load_tracker_checkpoint
from training.gnn.spatial_temporal_corner_data import CornerTrackingDataset
from utils.experiment_logger import ExperimentLogger, write


BODIES = {"peg": slice(0, 8), "box": slice(8, 16), "hole": slice(16, 20), "all": slice(0, 20)}
FORBIDDEN_NAMES = {"tapnext_state36.npz", "tapnext_recovery_state36.npz"}


def checked_file(path, expected_hash=None):
    """Reject forbidden inputs and check an explicitly pinned file before loading."""
    path = Path(path).expanduser().resolve()
    if path.name in FORBIDDEN_NAMES:
        raise PermissionError(f"Forbidden legacy cache input: {path}")
    if not path.is_file():
        raise ValueError(f"Expected an existing input file: {path}")
    if expected_hash is not None and file_sha256(path) != expected_hash:
        raise ValueError(f"Input SHA-256 mismatch: {path}")
    return path


def member_file(root, name, expected_hash):
    """Resolve a checksummed manifest member without allowing directory traversal."""
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest member escapes its root: {name}")
    return checked_file(path, expected_hash)


def causal_masks(observed):
    """Derive eligibility solely from actual, reference-certified detections."""
    observed = np.asarray(observed, dtype=bool)
    ever_seen = np.maximum.accumulate(observed, axis=0)
    return ever_seen, ever_seen & ~observed


def physical_observations(arrays):
    """Read common physical IDs 0..19; IDs 20..23 are the rear Hole rim."""
    frames = len(arrays["offsets"]) - 1
    observed = np.zeros((frames, 20), dtype=bool)
    for frame, (start, stop) in enumerate(zip(arrays["offsets"][:-1], arrays["offsets"][1:])):
        ids = arrays["vertex_ids"][start:stop]
        common = ids[(ids >= 0) & (ids < 20)]
        if len(np.unique(common)) != len(common):
            raise ValueError(f"Duplicate certified physical IDs in frame {frame}")
        observed[frame, common] = True
    return observed


def sorted_points(points):
    """Compare point sets without depending on surface-track storage order."""
    points = np.asarray(points)
    if not len(points):
        return points.reshape(0, 3)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return points[order]


def audit_episode_inputs(common, slots, surfaces, saved, episode_id):
    """Verify exact GT, TCP and raw coordinate lineage before comparing outputs."""
    np.testing.assert_array_equal(common["targets"][:, :20], slots["gt"], err_msg=episode_id)
    np.testing.assert_array_equal(slots["gt"], saved["gt"], err_msg=episode_id)
    np.testing.assert_array_equal(common["tcp"], slots["tcp"], err_msg=episode_id)
    np.testing.assert_array_equal(slots["observed"], saved["observed"], err_msg=episode_id)
    ever_seen, eligible = causal_masks(slots["observed"])
    np.testing.assert_array_equal(ever_seen, saved["ever_seen"], err_msg=episode_id)
    np.testing.assert_array_equal(eligible, saved["eligible"], err_msg=episode_id)
    for frame, (start, stop) in enumerate(zip(common["offsets"][:-1], common["offsets"][1:])):
        points = common["xyz"][start:stop]
        is_corner = common["is_corner"][start:stop]
        surface_points = points[~is_corner]
        if np.any(common["kinds"][start:stop][~is_corner] != 0):
            raise ValueError("Pinned comparison expects the original Peg-only surface release")
        np.testing.assert_array_equal(
            sorted_points(surface_points),
            sorted_points(surfaces["xyz"][frame, surfaces["valid"][frame]]),
            err_msg=f"Surface inputs differ: {episode_id}:{frame}",
        )
        # Native slot association can discard or permute detector points. Every
        # coordinate it accepts must still be an unchanged current-frame corner.
        detections = points[is_corner]
        assigned = slots["detected"][frame, slots["observed"][frame]]
        for point in assigned:
            if not np.any(np.all(detections == point, axis=1)):
                raise ValueError(f"Slot input is not a raw detector point: {episode_id}:{frame}")


def load_comparison_inputs(config):
    """Align saved surface predictions to the complete, independently certified test."""
    settings = config["surface"]
    results_path = checked_file(settings["results"], settings["results_sha256"])
    source = json.loads(results_path.read_text())
    predictions_path = checked_file(settings["predictions"], source["predictions_sha256"])
    model = source["models"]["surface_context"]
    if model["sha256"] != settings["checkpoint_sha256"] or model["epoch"] != 443:
        raise ValueError("Expected the frozen validation-selected epoch-443 surface checkpoint")
    if source["config"]["batch_size"] != 16:
        raise ValueError("Saved surface predictions must use the pinned batch-16 evaluation")
    slot_manifest_path = checked_file(settings["manifest"], source["test_manifest_sha256"])
    surface_manifest_path = checked_file(settings["surface_manifest"], source["surface_manifest_sha256"])
    common_manifest_path = checked_file(config["data"]["manifest"], config["data"]["manifest_sha256"])
    release_path = checked_file(config["data"]["release_manifest"])
    common_manifest = json.loads(common_manifest_path.read_text())
    slot_manifest = json.loads(slot_manifest_path.read_text())
    surface_manifest = json.loads(surface_manifest_path.read_text())
    release = json.loads(release_path.read_text())
    release_hash = file_sha256(release_path)
    if any(value != release_hash for value in (
        common_manifest["source"]["manifest_sha256"],
        slot_manifest["source_manifest_sha256"], surface_manifest["source_manifest_sha256"],
    )):
        raise ValueError("Both model inputs must originate from the identical source release")
    validate_test_manifest(common_manifest, release, config["expected"])
    records = common_manifest["episodes"]
    slot_records = slot_manifest["records"]
    slot_index = {(row["seed"], row["mode"]): index for index, row in enumerate(slot_records)}
    surface_index = {(row["seed"], row["mode"]): row for row in surface_manifest["records"]}
    release_index = {(row["seed"], row["policy_mode"]): row for row in release["records"]}
    common_keys = {(row["seed"], row["policy_mode"]) for row in records}
    if (len(slot_index) != len(slot_records) or set(slot_index) != common_keys
            or len(surface_index) != len(records) or set(surface_index) != common_keys):
        raise ValueError("Surface and online test episode inventories differ or contain duplicates")
    with np.load(predictions_path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in (
            "gt", "observed", "ever_seen", "eligible", "episode_offsets", "prediction_surface_context"
        )}
    offsets = saved["episode_offsets"]
    expected_offsets = np.r_[0, np.cumsum([row["frames"] for row in slot_records])]
    np.testing.assert_array_equal(offsets, expected_offsets)
    truth, observations, predictions, initialized, slot_observations, sample_ids = [], [], [], [], [], []
    for index, row in enumerate(records):
        key = row["seed"], row["policy_mode"]
        slot_number = slot_index[key]
        slot_row = slot_records[slot_number]
        surface_row = surface_index[key]
        if any(item["split"] != "test" or item["frames"] != row["frames"]
               for item in (slot_row, surface_row)):
            raise ValueError(f"Test split/frame mismatch: {key}")
        release_row = release_index[key]
        if (slot_row["detector_sha256"] != release_row["files"]["corners"]["sha256"]
                or surface_row["source_surface_sha256"] != release_row["files"]["surface"]["sha256"]):
            raise ValueError(f"Raw detector/surface lineage differs: {key}")
        common_path = member_file(common_manifest_path.parent, row["file"], row["sha256"])
        slot_path = member_file(slot_manifest_path.parent, slot_row["path"], slot_row["sha256"])
        surface_path = member_file(surface_manifest_path.parent, surface_row["path"], surface_row["sha256"])
        with np.load(common_path, allow_pickle=False) as archive:
            common = dict(archive)
        with np.load(slot_path, allow_pickle=False) as archive:
            slots = dict(archive)
        with np.load(surface_path, allow_pickle=False) as archive:
            surfaces = dict(archive)
        frame_slice = slice(offsets[slot_number], offsets[slot_number + 1])
        saved_episode = {key: value[frame_slice] for key, value in saved.items() if key != "episode_offsets"}
        audit_episode_inputs(common, slots, surfaces, saved_episode, row["episode"])
        truth.append(common["targets"][:, :20])
        observations.append(physical_observations(common))
        predictions.append(saved_episode["prediction_surface_context"])
        initialized.append(saved_episode["ever_seen"])
        slot_observations.append(saved_episode["observed"])
        sample_ids.extend(f"{row['seed']}:{row['policy_mode']}:{frame}" for frame in range(row["frames"]))
        if index % 50 == 0:
            print(f"Audited common inputs {index + 1}/{len(records)}", flush=True)
    return {
        "gt": np.concatenate(truth), "observed": np.concatenate(observations),
        "surface_prediction": np.concatenate(predictions), "surface_initialized": np.concatenate(initialized),
        "surface_observed": np.concatenate(slot_observations),
        "sample_ids": np.asarray(sample_ids),
        "episode_offsets": np.r_[0, np.cumsum([row["frames"] for row in records])],
    }, source, common_manifest


class PhysicalTrackCollector:
    """Score causal track bindings without feeding certificates back to inference.

    Bind a physical corner to the track that accepted its latest certified real
    detection, as in the existing online evaluator. Never rematch forecasts to
    GT. A wrong detection accepted by that track therefore remains an error.
    """

    def __init__(self):
        self.bindings = {}
        self.predictions = []
        self.available = []
        self.observed = []

    def __call__(self, episode, frame_index, result):
        if frame_index == 0:
            self.bindings = {}
        ids = episode.graphs[frame_index].vertex_id.cpu().numpy()
        rows = result["detection_rows"].cpu().numpy()
        tracks = result["detection_tracks"].cpu().numpy()
        for row, track in zip(rows, tracks):
            physical = int(ids[row])
            if 0 <= physical < 20:
                self.bindings[physical] = int(track)
        positions = result["positions"].cpu().numpy()
        prediction = np.full((20, 3), np.nan, dtype=np.float32)
        available = np.zeros(20, dtype=bool)
        for physical, track in self.bindings.items():
            if 0 <= track < len(positions):
                prediction[physical] = positions[track]
                available[physical] = True
        self.predictions.append(prediction)
        self.available.append(available)
        self.observed.append(episode.visible[frame_index, :20])


def error_summary(error, eligible, available):
    """Report missing output explicitly; pool 3D distances over point-frames."""
    count = int(eligible.sum())
    scored = eligible & available & np.isfinite(error)
    values = error[scored].astype(np.float64)
    result = {
        "eligible_count": count, "scored_count": len(values),
        "missing_prediction_count": count - len(values),
        "coverage": len(values) / count if count else None,
    }
    if len(values):
        result.update(mean_mm=float(values.mean()), rmse_mm=float(np.sqrt(np.mean(values ** 2))),
                      p90_mm=float(np.percentile(values, 90)), p95_mm=float(np.percentile(values, 95)))
    return result


def comparison_metrics(arrays):
    """Compute identical-cohort summaries and a separately labelled initialization view."""
    mask = arrays["eligible"]
    errors = {}
    availability = {}
    for name in ("surface", "online"):
        prediction = arrays[f"{name}_prediction"]
        errors[name] = np.linalg.norm(prediction.astype(np.float64) - arrays["gt"], axis=-1) * 1000
        availability[name] = np.isfinite(prediction).all(axis=-1)
    initialized_mask = mask & arrays["surface_initialized"] & arrays["online_available"]
    cohorts = {
        "common_reference": mask,
        "both_initialized": initialized_mask,
        "both_native_missing": initialized_mask & ~arrays["surface_observed"],
    }
    result = {}
    for cohort, eligible in cohorts.items():
        result[cohort] = {
            name: {
                body: error_summary(errors[name][:, slots], eligible[:, slots], availability[name][:, slots])
                for body, slots in BODIES.items()
            } for name in errors
        }
    result["surface_uninitialized_on_reference"] = int((mask & ~arrays["surface_initialized"]).sum())
    result["surface_observed_on_reference_missing"] = int((mask & arrays["surface_observed"]).sum())
    return result


def validate_online_reference(config):
    """Keep the published model and association identity fixed before replay."""
    settings = config["online_reference"]
    path = checked_file(settings["results"], settings["sha256"])
    reference = json.loads(path.read_text())
    if (reference["tracker_checkpoint_sha256"] != config["tracker"]["sha256"]
            or reference["association"] != config["association"]):
        raise ValueError("Online checkpoint/association differ from the frozen held-out run")
    return reference


def validate_online_replay(online, reference):
    """Require the unchanged online implementation to preserve published counts."""
    expected = reference["detector_only_online"]
    for key, value in expected.items():
        if isinstance(value, int) and online[key] != value:
            raise ValueError(f"Online replay changed {key}: {online[key]} != {value}")
    for key in ("missing_rmse_mm", "reappearance_rmse_mm"):
        if not np.isclose(online[key], expected[key], rtol=0, atol=0.001):
            raise ValueError(f"Online replay changed {key} by more than 0.001 mm")


def evaluate(config):
    """Audit the frozen paired inputs, replay online tracking, and save raw comparisons."""
    torch.set_num_threads(config["evaluation"].get("cpu_threads", 4))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config["device"])
    tracker_path = checked_file(config["tracker"]["checkpoint"], config["tracker"]["sha256"])
    reference = validate_online_reference(config)
    output = Path(config["output"]["root"])
    output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team="graph", model="corner_tracker_common_test", job_type="evaluate",
                          config=config, root=output / "experiments") as logger:
        arrays, surface_source, manifest = load_comparison_inputs(config)
        mask_parts = []
        for first, last in zip(arrays["episode_offsets"][:-1], arrays["episode_offsets"][1:]):
            _, eligible = causal_masks(arrays["observed"][first:last])
            mask_parts.append(eligible)
        arrays["eligible"] = np.concatenate(mask_parts)
        logger.record(
            dataset_hash=file_sha256(config["data"]["manifest"]),
            runtime={"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
                     "precision": "saved FP32 XYZ; float64 scoring" if config.get("online_rollout") else "FP32; TF32 disabled",
                     "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            surface_evaluation_run=surface_source["run_id"],
            implementation_sha256={
                name: file_sha256(Path(name)) for name in (
                    "evaluation/corner_tracker_comparison.py",
                    "evaluation/corner_tracker_rollout_recovery.py",
                    "evaluation/spatial_temporal_corner_tracking.py",
                    "graph/temporal/spatial_temporal_corner_tracker.py",
                    "training/gnn/spatial_temporal_corner_data.py",
                    "graph/state_estimation/graph.py",
                )
            },
        )
        if config.get("online_rollout"):
            from evaluation.corner_tracker_rollout_recovery import recover

            online, online_audit = recover(config, arrays, reference)
        else:
            model, checkpoint = load_tracker_checkpoint(tracker_path, config["tracker"]["sha256"], device)
            dataset = CornerTrackingDataset(
                config["data"]["manifest"], "test", max_edge_m=checkpoint["identity"]["max_edge_m"],
                graph_cache_root=config["data"].get("graph_cache_root"),
            )
            collector = PhysicalTrackCollector()
            online = evaluate_online(
                model, dataset, {**config["association"], "max_tracks_per_frame": 128, "log_every_episodes": 10},
                device, frame_callback=collector,
            )
            arrays["online_prediction"] = np.asarray(collector.predictions)
            arrays["online_available"] = np.asarray(collector.available)
            np.testing.assert_array_equal(arrays["observed"], np.asarray(collector.observed))
            online_audit = {"mode": "fresh_online_replay"}
        if online["frames"] != config["expected"]["frames"]:
            raise ValueError("Online replay did not process the complete test inventory")
        validate_online_replay(online, reference)
        metrics = comparison_metrics(arrays)
        predictions_path = output / "predictions.npz"
        np.savez_compressed(predictions_path, **arrays)
        result = {
            "run_id": logger.run_id, "episodes": online["episodes"], "frames": online["frames"],
            "common_corner_ids": list(range(20)), "common_eligible_point_frames": int(arrays["eligible"].sum()),
            "mask_sha256": hashlib.sha256(arrays["eligible"].tobytes()).hexdigest(),
            "metrics": metrics, "online_original_protocol": online,
            "online_trace_audit": online_audit,
            "surface_checkpoint_sha256": config["surface"]["checkpoint_sha256"],
            "tracker_checkpoint_sha256": config["tracker"]["sha256"],
            "surface_prediction_source_run": surface_source["run_id"],
            "predictions_sha256": file_sha256(predictions_path),
            "association": config["association"], "weights_unchanged": file_sha256(tracker_path) == config["tracker"]["sha256"],
            "test_used_for_selection": False,
            "online_published_counts_preserved": True,
            "input_audit": "all episode hashes, exact GT/TCP/surfaces, raw detector membership passed",
            "protocol": "reference-certified first-observation mask; native model association; no GT forecast rematching",
        }
        write(output / "results.json", result)
        logger.log(metrics, prefix="test")
        logger.link_output(tracker_path, checkpoint=True)
        logger.link_output(predictions_path)
        logger.link_output(output / "results.json")
        print(json.dumps({"run_id": logger.run_id, "metrics": metrics}, indent=2), flush=True)
    return str(output)
