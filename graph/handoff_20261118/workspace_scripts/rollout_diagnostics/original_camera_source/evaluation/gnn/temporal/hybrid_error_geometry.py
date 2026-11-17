"""Diagnostic-only endpoint interventions and exact signed error decompositions."""

from itertools import combinations, permutations

import numpy as np
import torch

from graph.pose_geometry import hole_frame_vector, peg_axis

FACTORS = ("translation", "rotation", "size")


def endpoint_position(object21):
    """Reconstruct Method A in the input translation frame, using only half-size X."""
    value = torch.as_tensor(object21, dtype=torch.float64)
    return (value[:, :3] + peg_axis(value[:, 3:7]) * value[:, 17:18]).numpy()


def local_vectors(vectors, target):
    """Express signed world-axis vectors in GT Hole X/Y/Z, strictly after inference."""
    return hole_frame_vector(
        torch.as_tensor(vectors, dtype=torch.float64),
        torch.as_tensor(target[:, 13:17], dtype=torch.float64),
    ).numpy()


def squared_decomposition(endpoint_error, hole_error):
    """Return per-frame terms of ||E-H||² = ||E||² + ||H||² - 2 E·H.

    Vectors are metres; squared terms use mm². Positive dot cancels error;
    negative dot amplifies it. A zero vector has no defined direction cosine.
    """
    endpoint_squared = np.sum(endpoint_error**2, axis=1) * 1e6
    hole_squared = np.sum(hole_error**2, axis=1) * 1e6
    dot = np.sum(endpoint_error * hole_error, axis=1) * 1e6
    relative_squared = np.sum((endpoint_error - hole_error) ** 2, axis=1) * 1e6
    denominator = np.sqrt(endpoint_squared * hole_squared)
    cosine = np.full(len(dot), np.nan)
    np.divide(dot, denominator, out=cosine, where=denominator > 1e-12)
    cosine = np.clip(cosine, -1, 1)
    residual = relative_squared - endpoint_squared - hole_squared + 2 * dot
    np.testing.assert_allclose(residual, 0, atol=1e-8, rtol=0)
    return dict(
        endpoint_squared_mm2=endpoint_squared,
        hole_squared_mm2=hole_squared,
        dot_mm2=dot,
        cross_term_mm2=-2 * dot,
        relative_squared_mm2=relative_squared,
        cosine=cosine,
        identity_residual_mm2=residual,
    )


def endpoint_interventions(prediction, target):
    """Enumerate all eight GT replacement sets, retaining signed endpoint errors.

    Bit 0 replaces translation, bit 1 rotation, bit 2 Size3. Size Y/Z do not
    enter this endpoint definition. These are oracle diagnostics, never inputs
    to a model or proposed inference procedures.
    """
    target_endpoint = endpoint_position(target)
    errors = {}
    for replacement in range(8):
        value = prediction.copy()
        for bit, region in enumerate((slice(0, 3), slice(3, 7), slice(17, 20))):
            if replacement & (1 << bit):
                value[:, region] = target[:, region]
        errors[replacement] = endpoint_position(value) - target_endpoint
    return errors


def endpoint_factor_vectors(prediction, target):
    """Exactly separate translation, rotation, size and rotation-size coupling.

    E = delta_p + L_GT*delta_axis + axis_GT*delta_L + delta_axis*delta_L.
    This is an algebraic vector identity; its squared terms also need cross terms.
    """
    predicted_axis = peg_axis(torch.as_tensor(prediction[:, 3:7], dtype=torch.float64)).numpy()
    target_axis = peg_axis(torch.as_tensor(target[:, 3:7], dtype=torch.float64)).numpy()
    axis_difference = predicted_axis - target_axis
    size_difference = prediction[:, 17:18] - target[:, 17:18]
    return dict(
        translation=prediction[:, :3] - target[:, :3],
        rotation=axis_difference * target[:, 17:18],
        size=target_axis * size_difference,
        rotation_size_interaction=axis_difference * size_difference,
    )


def factor_squared_terms(vectors):
    """Keep every self and pair term; do not call norm averages contributions."""
    result = {
        name + "_squared_mm2": np.sum(value**2, axis=1) * 1e6 for name, value in vectors.items()
    }
    for left, right in combinations(vectors, 2):
        result[left + "__" + right + "_cross_mm2"] = (
            2 * np.sum(vectors[left] * vectors[right], axis=1) * 1e6
        )
    return result


def ordered_oracle_reductions(interventions, mask):
    """Average MSE reductions over all six orders; these remain oracle allocations."""
    if not mask.any():
        return {"orders": [], "order_averaged_mse_reduction_mm2": {}}
    mse = {
        key: float(np.mean(np.sum(value[mask] ** 2, axis=1)) * 1e6)
        for key, value in interventions.items()
    }
    orders = []
    reductions = {factor: [] for factor in FACTORS}
    for order in permutations(range(3)):
        replaced = 0
        row = {"order": [FACTORS[bit] for bit in order], "reductions_mm2": {}}
        for bit in order:
            next_replaced = replaced | (1 << bit)
            gain = mse[replaced] - mse[next_replaced]
            row["reductions_mm2"][FACTORS[bit]] = gain
            reductions[FACTORS[bit]].append(gain)
            replaced = next_replaced
        orders.append(row)
    return {
        "orders": orders,
        "order_averaged_mse_reduction_mm2": {
            name: dict(
                mean=float(np.mean(values)), minimum=float(min(values)), maximum=float(max(values))
            )
            for name, values in reductions.items()
        },
    }


def geometry_groups(target, settings):
    """Partition offline GT geometry without claiming grasp/contact phase labels.

    Priority is head-region, approach, alignment, pre-alignment, distant. These
    are instantaneous spatial proxies, not ordered task events or contact GT.
    """
    for name, value in settings.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"geometry_proxies.{name} must be a finite number")
    if not (
        settings["alignment_min_x_m"]
        <= settings["approach_min_x_m"]
        < settings["head_min_x_m"]
        < settings["head_max_x_m"]
    ):
        raise ValueError("Expected alignment_min_x <= approach_min_x < head_min_x < head_max_x")
    if not (
        0
        < settings["approach_lateral_m"]
        <= settings["alignment_lateral_m"]
        < settings["near_distance_m"]
    ):
        raise ValueError("Expected 0 < approach_lateral <= alignment_lateral < near_distance")
    if not (0 < settings["approach_axis_deg"] <= settings["alignment_axis_deg"] <= 180):
        raise ValueError("Expected 0 < approach_axis_deg <= alignment_axis_deg <= 180")
    local = local_vectors(target[:, 7:10], target)
    peg = peg_axis(torch.as_tensor(target[:, 3:7], dtype=torch.float64)).numpy()
    hole = peg_axis(torch.as_tensor(target[:, 13:17], dtype=torch.float64)).numpy()
    angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(peg, hole), axis=1), (peg * hole).sum(1)))
    lateral = np.linalg.norm(local[:, 1:], axis=1)
    fine = (
        (local[:, 0] >= settings["approach_min_x_m"])
        & (local[:, 0] <= settings["head_max_x_m"])
        & (lateral <= settings["approach_lateral_m"])
        & (angle <= settings["approach_axis_deg"])
    )
    head = fine & (local[:, 0] >= settings["head_min_x_m"])
    approach = fine & ~head
    coarse = (
        (local[:, 0] >= settings["alignment_min_x_m"])
        & (local[:, 0] <= settings["head_max_x_m"])
        & (lateral <= settings["alignment_lateral_m"])
        & (angle <= settings["alignment_axis_deg"])
    )
    alignment = coarse & ~fine
    near = np.linalg.norm(local, axis=1) <= settings["near_distance_m"]
    groups = {
        "all": np.ones(len(local), dtype=bool),
        "distant_geometry_proxy": ~(near | coarse | fine),
        "pre_alignment_geometry_proxy": near & ~(coarse | fine),
        "alignment_geometry_proxy": alignment,
        "insertion_approach_geometry_proxy": approach,
        "head_in_region_geometry_proxy": head,
        "historical_alignment_geometry_proxy": (
            (local[:, 0] >= -0.05) & (local[:, 0] <= 0.02) & (lateral <= 0.01)
        ),
    }
    partition = [
        value
        for key, value in groups.items()
        if key not in ("all", "historical_alignment_geometry_proxy")
    ]
    np.testing.assert_array_equal(np.sum(partition, axis=0), 1)
    geometry = dict(gt_local_relative_m=local, gt_axis_alignment_deg=angle)
    return groups, geometry
