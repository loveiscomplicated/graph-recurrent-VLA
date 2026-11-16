"""Isolate the frozen corner Transformer's forecasts on expert and random actions."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.vertex_family_labels import pose_rotation
from evaluation.corner_rigidity import TrackIdentities, representatives
from evaluation.gnn.vertex_metrics import statistics
from evaluation.independent_test import (
    initialize_models,
    predict_frame,
    read_observation,
    verify_expert_random_prefix,
)
from evaluation.random_action_diagnostics import references
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from graph.utils.object21_targets import object21_from_gt
from utils.experiment_logger import ExperimentLogger


MODES = ("expert", "random_after_lift")
ERROR_NAMES = ("transformer", "hold_point", "previous_robot", "transport_tcp", "other_branch_robot")


class ForecastProbe(DetectedCornerTracker):
    """Capture forecasts before association and score alternate final tokens."""

    def __init__(self, model, gate_m, device):
        super().__init__(model, gate_m, device)
        self.robot_variants = {}
        self.forecasts = {}

    def predict(self, robot):
        """Only the original forecast goes back into the tracker."""
        forecast = super().predict(robot)
        self.forecasts = {"transformer": forecast}
        if not self.tracks or not self.robot_variants:
            return forecast
        lengths = np.asarray([len(track.inputs) for track in self.tracks])
        inputs = np.zeros((len(self.tracks), int(lengths.max()), 13), dtype=np.float32)
        for index, track in enumerate(self.tracks):
            inputs[index, :lengths[index]] = track.inputs
        valid = np.arange(inputs.shape[1])[None, :] < lengths[:, None]
        rows = np.arange(len(self.tracks))
        anchor = np.stack([track.last_observed for track in self.tracks])
        context = torch.as_tensor(valid, device=self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda",
        ):
            for name, replacement in self.robot_variants.items():
                inputs[rows, lengths - 1, 4:] = replacement
                delta = self.model(torch.as_tensor(inputs, device=self.device), context).float()
                self.forecasts[name] = anchor + delta[rows, lengths - 1].cpu().numpy()
        return forecast


def transport_point(point, previous_robot, robot):
    """Move a world point as if it were rigidly attached to the observed TCP."""
    previous_rotation = pose_rotation(previous_robot[:7])
    rotation = pose_rotation(robot[:7])
    return rotation @ previous_rotation.T @ (point - previous_robot[:3]) + robot[:3]


def score_forecasts(tracker, previous_labels, previous_observed, anchor, tracked, truth,
                    previous_truth, robot, previous_robot, grasped, seed, step):
    """Label prediction targets with only the identity ledger from the prior frame."""
    rows = []
    kinds = np.asarray([track.kind for track in tracker.tracks[:len(previous_labels)]], dtype=np.int64)
    for kind in (0, 1):
        selected = representatives(previous_labels, previous_observed, kinds, kind)
        for index in selected:
            identity = int(previous_labels[index])
            fresh = bool(previous_observed[index])
            target = truth[identity]
            positions = {"hold_point": anchor[index], **{
                name: points[index] for name, points in tracker.forecasts.items()
            }}
            if kind == 0 and fresh and grasped:
                positions["transport_tcp"] = transport_point(anchor[index], previous_robot, robot)
            errors = {name: float(np.linalg.norm(point - target) * 1000)
                      for name, point in positions.items()}
            rows.append(dict(step=step, model_seed=seed, identity=identity, kind=kind, fresh=fresh,
                             missed=not bool(tracked["observed"][index]), grasped=grasped,
                             movement_mm=float(np.linalg.norm(target - previous_truth[identity]) * 1000),
                             errors=errors))
    return rows


def verify_baseline(prediction, gt, prior, seed, index, path):
    """Confirm the probe leaves the original detector/tracker/GNN path intact."""
    if (prediction is not None) != prior[f"available_{seed}"][index]:
        raise ValueError(f"Baseline availability changed for seed {seed} at {path}")
    if not np.array_equal(gt, prior[f"target_{seed}"][index]):
        raise ValueError(f"Baseline GT changed for seed {seed} at {path}")
    if prediction is not None and not np.array_equal(
        prediction["object21"][0].cpu().numpy(), prior[f"prediction_{seed}"][index],
    ):
        raise ValueError(f"Baseline prediction changed for seed {seed} at {path}")


def audit_episode(receipt, paired_receipt, root, models, detector, device, prior_root, gate_mm):
    """Replay an episode and score each eligible forecast before current detection."""
    name = f'{receipt["mode"]}_seed_{receipt["seed"]}'
    with np.load(prior_root / f"{name}.npz", allow_pickle=False) as archive:
        prior = {key: archive[key] for key in archive.files}
    trackers = {seed: ForecastProbe(model["temporal"], model["gate_m"], device)
                for seed, model in models.items()}
    ledgers = {seed: TrackIdentities(gate_mm) for seed in models}
    paired_index = paired_receipt["switch_frame"] + 1
    paired_path = root / paired_receipt["path"] / f"frame_{paired_index:06d}.npz"
    if file_sha256(paired_path) != paired_receipt["frame_sha256"][paired_index]:
        raise ValueError(f"Paired first-action frame checksum mismatch: {paired_path}")
    paired_observation, _, paired_step, _ = read_observation(paired_path)
    if paired_step != 1:
        raise ValueError(f"Expected first action result at {paired_path}")
    paired_robot = paired_observation[4]
    rows = []
    previous_robot = previous_truth = None
    previous_grasp = False
    for index, expected_sha in enumerate(receipt["frame_sha256"]):
        path = root / receipt["path"] / f"frame_{index:06d}.npz"
        if file_sha256(path) != expected_sha:
            raise ValueError(f"Raw frame checksum mismatch: {path}")
        obs, gt, step, grasp = read_observation(path)
        detection = detector(*obs[:4])
        robot = obs[4]
        truth = references(SimpleNamespace(**gt))
        target = object21_from_gt(gt)[0].numpy()
        for seed, model in models.items():
            tracker = trackers[seed]
            tracker.robot_variants = {}
            if step > 0:
                tracker.robot_variants["previous_robot"] = previous_robot
                if step == 1:
                    tracker.robot_variants["other_branch_robot"] = paired_robot
                previous_labels = np.asarray(ledgers[seed].ids, dtype=np.int64)
                seen = np.asarray([track.previous_observed for track in tracker.tracks], dtype=bool)
                anchor = [track.last_observed for track in tracker.tracks]
            prediction, tracked = predict_frame(model["gnn"], tracker, detection, robot, device)
            verify_baseline(prediction, target, prior, seed, index, path)
            if step > 0:
                rows.extend(score_forecasts(tracker, previous_labels, seen, anchor, tracked, truth,
                                            previous_truth, robot, previous_robot,
                                            previous_grasp and grasp, seed, step))
            kinds = np.asarray([track.kind for track in tracker.tracks], dtype=np.int64)
            ledgers[seed].update(tracked, kinds, truth)
        previous_robot, previous_truth, previous_grasp = robot, truth, grasp
    if any(len(prior[f"available_{seed}"]) != receipt["observations"] for seed in models):
        raise ValueError(f"Unexpected frame count for {name}")
    return dict(seed=receipt["seed"], mode=receipt["mode"], frames=receipt["observations"], rows=rows)


def describe(rows):
    """Report scalar errors on identical samples and include variant coverage."""
    result = dict(samples=len(rows), resets=len({row["reset"] for row in rows}),
                  movement_mm=statistics([row["movement_mm"] for row in rows]))
    for name in ERROR_NAMES:
        available = [row["errors"] for row in rows if name in row["errors"]]
        result[name] = dict(samples=len(available), mm=statistics([row[name] for row in available]),
                            improvement_vs_hold_mm=statistics([row["hold_point"] - row[name]
                                                               for row in available]))
    return result


def summarize(episodes):
    """Pair forecasts by reset, post-lift step, model seed and physical corner."""
    by_mode = {mode: {} for mode in MODES}
    for episode in episodes:
        reset = episode["seed"]
        rows = by_mode[episode["mode"]]
        for row in episode["rows"]:
            key = (reset, row["step"], row["model_seed"], row["identity"])
            if key in rows:
                raise ValueError(f"Duplicate forecast target at {key}")
            rows[key] = dict(reset=reset, **row)
    expert, random = (by_mode[mode] for mode in MODES)
    common = sorted(expert.keys() & random.keys())
    result = {}
    for kind, name in enumerate(("peg", "box")):
        paired = [(expert[key], random[key]) for key in common if expert[key]["kind"] == kind]
        fresh = [(left, right) for left, right in paired if left["fresh"] and right["fresh"]]
        grasped = [(left, right) for left, right in fresh if left["grasped"] and right["grasped"]]
        groups = dict(one_step=fresh, one_step_grasped=grasped,
                      first_action=[(left, right) for left, right in fresh if left["step"] == 1],
                      one_step_missed=[(left, right) for left, right in fresh
                                       if left["missed"] and right["missed"]])
        result[name] = {
            group: {mode: describe([pair[index] for pair in pairs]) for index, mode in enumerate(MODES)}
            for group, pairs in groups.items()
        }
        result[name]["natural_missed"] = {
            mode: describe([row for row in rows.values() if row["kind"] == kind and row["missed"]])
            for mode, rows in by_mode.items()
        }
        if kind == 0:
            result[name]["grasped_movement_bins_mm"] = {}
            for low, high in ((0, 2), (2, 5), (5, 10), (10, 20), (20, np.inf)):
                subset = [(left, right) for left, right in grasped
                          if all(low <= row["movement_mm"] < high for row in (left, right))]
                label = f"{low}-{high}"
                result[name]["grasped_movement_bins_mm"][label] = {
                    mode: describe([pair[index] for pair in subset]) for index, mode in enumerate(MODES)
                }
    return result


def evaluate(config):
    """Run the action probe on a protocol-pinned held-out cohort."""
    if file_sha256(config["protocol"]) != config["protocol_sha256"]:
        raise ValueError("Frozen held-out protocol SHA256 mismatch")
    protocol = json.loads(Path(config["protocol"]).read_text())
    root = Path(config["raw_root"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "COMPLETE" or manifest["protocol_sha256"] != config["protocol_sha256"]:
        raise ValueError("Incomplete raw dataset or mismatched held-out protocol")
    receipts = {(row["seed"], row["mode"]): row for row in manifest["episodes"]}
    expected = {(seed, mode) for seed in protocol["seeds"] for mode in MODES}
    if set(receipts) != expected or len(receipts) != len(manifest["episodes"]):
        raise ValueError("Raw dataset must include each held-out case exactly once")
    prefix_frames = verify_expert_random_prefix(manifest, root, protocol["seeds"])

    for item in protocol["models"]:
        for role in ("temporal", "gnn"):
            item[role] = str(Path(config["checkpoint_root"]) / f'{role}_{item["seed"]}.pt')
    device = torch.device(config["device"])
    torch.set_num_threads(config["cpu_threads"])
    models = initialize_models(protocol, device, config["gate_m"])
    detector = importlib.import_module(config["detector_module"]).TorchCornerDetector(device=device)
    prior_root = Path(config["prior_evaluation"])
    with ExperimentLogger(team="graph", model="corner_action_generalization", config=config,
                          job_type="evaluation", seed=protocol["generation"]["seed"]) as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        episodes = []
        for (seed, mode), receipt in receipts.items():
            other_mode = MODES[1] if mode == MODES[0] else MODES[0]
            episodes.append(audit_episode(receipt, receipts[(seed, other_mode)], root, models,
                                          detector, device, prior_root, config["identity_gate_mm"]))
            print(f"Action replay {len(episodes)}/{len(receipts)}: {mode} {seed}", flush=True)
        result = dict(config=config, models=protocol["models"], torch_version=torch.__version__,
                      raw_manifest_sha256=file_sha256(manifest_path), baseline_predictions_exact=True,
                      prefix_frames_verified=prefix_frames, summary=summarize(episodes))
        for name, body in (("summary.json", result), ("forecasts.json", episodes)):
            path = output / name
            path.write_text(json.dumps(body, indent=2 if name == "summary.json" else None, allow_nan=False))
            logger.link_output(path)
        logger.record(frames=sum(ep["frames"] for ep in episodes), baseline_predictions_exact=True)
        print(f"Corner action generalization results: {output}", flush=True)
