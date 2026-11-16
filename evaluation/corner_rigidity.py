"""Measure within-object distortion before the frozen leaderboard GNN."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.gnn.box_hole_geometry import rigid_registration
from evaluation.gnn.vertex_metrics import statistics
from evaluation.independent_test import initialize_models, predict_frame, read_observation
from evaluation.random_action_diagnostics import nearest_references, predict, references
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from graph.utils.object21_targets import object21_from_gt
from utils.experiment_logger import ExperimentLogger


OBJECTS = ("peg", "box")
PROVENANCE = ("observed_observed", "observed_predicted", "predicted_predicted")
CONDITIONS = ("original", "last_detection", "rigid")


class TrackIdentities:
    """Audit detector-only tracks against GT in a separate evaluation ledger."""

    def __init__(self, gate_mm):
        self.gate_mm = gate_mm
        self.ids = []

    def update(self, tracked, kinds, truth):
        """Fix identities at first sight and flag later contradictory detections."""
        observed = tracked["observed"]
        indices = np.flatnonzero(observed)
        self.ids.extend([-1] * (len(observed) - len(self.ids)))
        nearest, distances = nearest_references(tracked["positions"][indices], kinds[indices], truth)
        for index, identity, distance in zip(indices, nearest, distances):
            if self.ids[index] == -1:
                self.ids[index] = identity if distance <= self.gate_mm else -2
            elif self.ids[index] >= 0 and (identity != self.ids[index] or distance > self.gate_mm):
                self.ids[index] = -3
        return np.asarray(self.ids, dtype=np.int64)


def representatives(identities, observed, kinds, kind):
    """Choose one clean track per physical corner, preferring a current detection."""
    candidates = np.flatnonzero((identities >= 0) & (kinds == kind))
    ordered = sorted(candidates, key=lambda index: (not observed[index], index))
    selected = {}
    for index in ordered:
        selected.setdefault(identities[index], index)
    return np.asarray(list(selected.values()), dtype=np.int64)


def rigid_forecasts(positions, observed, selected, truth, candidates, identities):
    """Fit one pose to the selected graph points and move only clean forecasts."""
    if len(selected) < 3 or not len(candidates):
        return positions, np.empty(0, dtype=np.int64)
    # Centering removes the current GT translation. A proper rotation fit
    # absorbs its orientation, so only the episode's shape informs projection.
    template = truth[identities[selected]]
    template_origin = template.mean(axis=0)
    template = template - template_origin
    if np.linalg.matrix_rank(template, tol=1e-7) < 2:
        return positions, np.empty(0, dtype=np.int64)
    rotation, translation = rigid_registration(template, positions[selected])
    corrected = positions.copy()
    corrected[candidates] = (truth[identities[candidates]] - template_origin) @ rotation.T + translation
    if not np.array_equal(corrected[observed], positions[observed]):
        raise RuntimeError("Rigid intervention must preserve observed coordinates")
    return corrected, candidates


def shape_errors(position_sets, truth, observed):
    """Return per-frame mean absolute GT distance errors by pair provenance."""
    left, right = np.triu_indices(len(truth), k=1)
    pair_kind = (~observed[left]).astype(int) + (~observed[right]).astype(int)
    gt_length = np.linalg.norm(truth[left] - truth[right], axis=1)
    errors = {}
    for name, xyz in position_sets.items():
        length = np.linalg.norm(xyz[left] - xyz[right], axis=1)
        errors[name] = np.abs(length - gt_length) * 1000
    result = {}
    for kind, name in enumerate(PROVENANCE):
        mask = pair_kind == kind
        if mask.any():
            result[name] = {"pairs": int(mask.sum()),
                            **{key: float(value[mask].mean()) for key, value in errors.items()}}
    return result


def position_error(prediction, target, kind):
    """Read the object's position error from Object21 in millimetres."""
    start = 0 if kind == 0 else 10
    return float(np.linalg.norm(prediction[start:start + 3] - target[start:start + 3]) * 1000)


def audit_object(model, tracker, tracked, kinds, identities, truth, output, target, robot, kind, device):
    """Measure one object and perturb only its forecast nodes at GNN input."""
    positions = tracked["positions"]
    observed = tracked["observed"]
    selected = representatives(identities, observed, kinds, kind)
    candidates = np.flatnonzero((identities >= 0) & (kinds == kind) & ~observed)
    rigid, changed = rigid_forecasts(positions, observed, selected, truth, candidates, identities)

    last_detection = positions[selected].copy()
    for row, index in enumerate(selected):
        if not observed[index]:
            last_detection[row] = tracker.tracks[index].last_observed
    selected_truth = truth[identities[selected]]
    pairs = shape_errors(
        {"original": positions[selected], "last_detection": last_detection, "rigid": rigid[selected]},
        selected_truth, observed[selected],
    )
    errors = {"original": position_error(output, target, kind)}
    if len(changed):
        gt_forecasts = positions.copy()
        gt_forecasts[changed] = truth[identities[changed]]
        for name, xyz in (("rigid", rigid), ("gt_forecast", gt_forecasts)):
            prediction = predict(model["gnn"], xyz, kinds, observed, robot[:3], device)
            errors[name] = position_error(prediction, target, kind)
    return dict(pairs=pairs, position_mm=errors, fitted_forecasts=len(changed),
                observed_nodes=int(((kinds == kind) & observed).sum()),
                predicted_nodes=int(((kinds == kind) & ~observed).sum()),
                clean_predicted_nodes=len(candidates),
                rejected_nodes=int(((kinds == kind) & (identities < 0)).sum()))


def audit_episode(receipt, root, models, detector, device, prior_root, identity_gate_mm):
    """Replay in causal order and compare the unmodified path to the saved outputs."""
    trackers = {seed: DetectedCornerTracker(model["temporal"], model["gate_m"], device)
                for seed, model in models.items()}
    identities = {seed: TrackIdentities(identity_gate_mm) for seed in models}
    name = f'{receipt["mode"]}_seed_{receipt["seed"]}'
    with np.load(prior_root / f"{name}.npz", allow_pickle=False) as archive:
        prior = {key: archive[key] for key in archive.files}
    frames = []
    for index, expected_sha in enumerate(receipt["frame_sha256"]):
        path = root / receipt["path"] / f"frame_{index:06d}.npz"
        if file_sha256(path) != expected_sha:
            raise ValueError(f"Raw frame checksum mismatch: {path}")
        observation, gt, after_lift, _ = read_observation(path)
        detection = detector(*observation[:4])
        robot = observation[4]
        target = object21_from_gt(gt)[0].numpy()
        truth = references(SimpleNamespace(**gt))
        frame = dict(after_lift=after_lift, models={})
        for seed, model in models.items():
            tracker = trackers[seed]
            result, tracked = predict_frame(model["gnn"], tracker, detection, robot, device)
            if (result is not None) != prior[f"available_{seed}"][index]:
                raise ValueError(f"Saved prediction availability differs at {path}, model seed {seed}")
            if not np.array_equal(target, prior[f"target_{seed}"][index]):
                raise ValueError(f"Saved target differs at {path}, model seed {seed}")
            if result is None:
                continue
            output = result["object21"][0].cpu().numpy()
            if not np.array_equal(output, prior[f"prediction_{seed}"][index]):
                raise ValueError(f"Baseline prediction differs at {path}, model seed {seed}")
            kinds = np.asarray([track.kind for track in tracker.tracks], dtype=np.int64)
            labels = identities[seed].update(tracked, kinds, truth)
            frame["models"][seed] = {
                obj: audit_object(model, tracker, tracked, kinds, labels, truth, output, target, robot, kind, device)
                for kind, obj in enumerate(OBJECTS)
            }
        frames.append(frame)
    if len(frames) != receipt["observations"] or any(len(prior[f"available_{s}"]) != len(frames) for s in models):
        raise ValueError(f"Unexpected frame count for {name}")
    return dict(seed=receipt["seed"], mode=receipt["mode"], frames=frames)


def summarize_group(frames, obj):
    """Average within each physical frame over eligible model seeds."""
    pairs = {pair: [] for pair in PROVENANCE}
    position = []
    fitted_position = []
    fitted_pair = []
    coverage = dict(observed_nodes=0, predicted_nodes=0, clean_predicted_nodes=0, rejected_nodes=0,
                    fitted_forecasts=0)
    for frame in frames:
        rows = [model[obj] for model in frame["models"].values()]
        if not rows:
            continue
        for name in coverage:
            coverage[name] += sum(row[name] for row in rows)
        position.append(np.mean([row["position_mm"]["original"] for row in rows]))
        fitted = [row for row in rows if row["fitted_forecasts"]]
        if fitted:
            fitted_position.append({key: np.mean([row["position_mm"][key] for row in fitted])
                                    for key in ("original", "rigid", "gt_forecast")})
        predicted_pairs = [row["pairs"]["predicted_predicted"] for row in fitted
                           if "predicted_predicted" in row["pairs"]]
        if predicted_pairs:
            fitted_pair.append({key: np.mean([row[key] for row in predicted_pairs])
                                for key in ("original", "rigid")})
        for pair in PROVENANCE:
            scored = [row["pairs"][pair] for row in rows if pair in row["pairs"]]
            if scored:
                pairs[pair].append({**{key: np.mean([row[key] for row in scored]) for key in CONDITIONS},
                                    "pairs": sum(row["pairs"] for row in scored)})
    return dict(
        baseline_position_mm=statistics(position),
        fitted_position={"frames": len(fitted_position), **{
            key: statistics([row[key] for row in fitted_position])
            for key in ("original", "rigid", "gt_forecast")}},
        fitted_predicted_pairs={"frames": len(fitted_pair), **{
            key: statistics([row[key] for row in fitted_pair]) for key in ("original", "rigid")}},
        pair_distance_mm={pair: {"frames": len(rows), "pairs": sum(row["pairs"] for row in rows),
                                 **{key: statistics([row[key] for row in rows]) for key in CONDITIONS}}
                          for pair, rows in pairs.items()},
        coverage=coverage,
    )


def summarize(episodes):
    """Keep full expert trajectories and the paired post-lift modes separate."""
    result = {}
    for mode, post_lift in (("expert", False), ("expert", True), ("random_after_lift", True)):
        frames = [frame for episode in episodes if episode["mode"] == mode for frame in episode["frames"]
                  if not post_lift or frame["after_lift"] > 0]
        name = f"{mode}_post_lift" if post_lift else f"{mode}_all"
        result[name] = dict(frames=len(frames), **{obj: summarize_group(frames, obj) for obj in OBJECTS})
    return result


def evaluate(config):
    """Verify protocol, raw test frames and checkpoints, and save the paired audit."""
    if file_sha256(config["protocol"]) != config["protocol_sha256"]:
        raise ValueError("Frozen held-out protocol SHA256 mismatch")
    protocol = json.loads(Path(config["protocol"]).read_text())
    root = Path(config["raw_root"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "COMPLETE" or manifest["protocol_sha256"] != config["protocol_sha256"]:
        raise ValueError("Incomplete raw dataset or mismatched held-out protocol")
    expected = {(seed, mode) for seed in protocol["seeds"] for mode in protocol["policy"]["modes"]}
    actual = {(row["seed"], row["mode"]) for row in manifest["episodes"]}
    if expected != actual or len(actual) != len(manifest["episodes"]):
        raise ValueError("Raw dataset must include each held-out case exactly once")

    checkpoint_root = Path(config["checkpoint_root"])
    for item in protocol["models"]:
        for role in ("temporal", "gnn"):
            item[role] = str(checkpoint_root / f'{role}_{item["seed"]}.pt')
    device = torch.device(config["device"])
    torch.set_num_threads(config["cpu_threads"])
    models = initialize_models(protocol, device, config["gate_m"])
    detector_class = importlib.import_module(config["detector_module"]).TorchCornerDetector
    detector = detector_class(device=device)
    prior_root = Path(config["prior_evaluation"])
    with ExperimentLogger(team="graph", model="corner_rigidity", config=config, job_type="evaluation",
                          seed=protocol["generation"]["seed"]) as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        episodes = []
        for receipt in manifest["episodes"]:
            episodes.append(audit_episode(receipt, root, models, detector, device, prior_root,
                                          config["identity_gate_mm"]))
            print(f'Rigidity replay {len(episodes)}/{len(expected)}: {receipt["mode"]} {receipt["seed"]}',
                  flush=True)
        result = dict(config=config, models=protocol["models"], torch_version=torch.__version__,
                      raw_manifest_sha256=file_sha256(manifest_path),
                      baseline_predictions_exact=True, summary=summarize(episodes))
        results_path = output / "summary.json"
        results_path.write_text(json.dumps(result, indent=2, allow_nan=False))
        frames_path = output / "per_frame.json"
        frames_path.write_text(json.dumps(episodes, allow_nan=False))
        logger.link_output(results_path)
        logger.link_output(frames_path)
        logger.record(frames=sum(len(ep["frames"]) for ep in episodes),
                      protocol_sha256=config["protocol_sha256"], baseline_predictions_exact=True)
        print(f"Corner rigidity results: {output}", flush=True)
