"""Association gates and uncertainty scaling fitted on calibration parents only."""

import numpy as np
import torch

from graph.temporal.corner_tracking.tracker import AssociationCalibration
from .corner_tracking_rollout import replay_episode


def bootstrap_gates(episodes, maximum_age=32):
    """Measure constant-velocity innovations plus observed detector error by age.

    Future GT is used only to fit offline thresholds. There is no GT motion rule
    or physical ID in the deployed tracker. Unsupported types retain disabled
    uncertainty admission until sufficient calibration evidence is available.
    """
    errors = [[[] for _ in range(maximum_age + 1)] for _ in range(3)]
    noise = [[] for _ in range(3)]
    for episode in episodes:
        history = {}
        for time, frame in enumerate(episode.frames):
            for index, reference in enumerate(frame.physical_ids):
                if reference < 0:
                    continue
                kind = int(frame.observation.kind[index])
                anchor = frame.observation.xyz[index]
                noise[kind].append(float(np.linalg.norm(anchor - frame.reference_xyz[reference])))
                previous = history.get(int(reference))
                velocity = np.zeros(3)
                if previous is not None and previous[0] != time:
                    velocity = (anchor - previous[1]) / (time - previous[0])
                history[int(reference)] = (time, anchor)
                for age in range(1, min(maximum_age + 1, len(episode.frames) - time)):
                    predicted = anchor + velocity * age
                    target = episode.frames[time + age].reference_xyz[reference]
                    errors[kind][age].append(float(np.linalg.norm(predicted - target)))
    radii = np.full((3, maximum_age + 1), 0.001)
    for kind in range(3):
        detector_noise = float(np.quantile(noise[kind], 0.99)) if noise[kind] else 0.0
        for age in range(1, maximum_age + 1):
            if errors[kind][age]:
                radii[kind, age] = max(0.001, float(np.quantile(errors[kind][age], 0.99)) + detector_noise)
            else:
                radii[kind, age] = radii[kind, age - 1]
        radii[kind, 0] = radii[kind, 1]
    return AssociationCalibration(radii), dict(
        estimator="constant velocity; Q99 position error plus Q99 detector error",
        gate_counts=[[len(values) for values in group] for group in errors],
        detector_counts=list(map(len, noise)),
    )


@torch.no_grad()
def calibrate_uncertainty(motion, gnn, episodes, plans, calibration, settings, minimum_samples=32):
    """Fit Q95 error/std ratios without consulting development validation."""
    previous_motion_mode, previous_gnn_mode = motion.training, gnn.training
    motion.eval()
    gnn.eval()
    ranges = ((1, 2), (3, 8), (9, 16), (17, 32))
    ratios = [[[] for _ in ranges] for _ in range(3)]
    for episode, plan in zip(episodes, plans):
        result = replay_episode(motion, gnn, episode, plan, calibration, settings)
        for row in result["nodes"]:
            if row["identity_consistent"] and row["std_norm_m"] > 0:
                for bin_index, (low, high) in enumerate(ranges):
                    if low <= row["age"] <= high:
                        ratios[row["kind"]][bin_index].append(row["error_m"] / row["std_norm_m"])
    scales = np.full_like(calibration.radii, np.inf)
    unsupported = []
    for kind, bins in enumerate(ratios):
        for bin_index, ((low, high), values) in enumerate(zip(ranges, bins)):
            if len(values) >= minimum_samples:
                scales[kind, low:high + 1] = max(0.001, float(np.quantile(values, 0.95)))
            else:
                unsupported.append(dict(kind=kind, age_range=[low, high]))
    motion.train(previous_motion_mode)
    gnn.train(previous_gnn_mode)
    return AssociationCalibration(calibration.radii.copy(), scales), dict(
        method="Q95 Euclidean error / predicted std norm; calibration parents only",
        counts=[[len(values) for values in bins] for bins in ratios], minimum_samples=minimum_samples,
        unsupported_bins=unsupported,
    )
