"""Procedural rigid bodies for Object21D correctness and stress checks."""

import math

import numpy as np

from data_collection.episode_dataset import canonical_hash
from data_collection.state21 import Episode, Sample
from graph.state_estimation.codec import object21_from_gt
from graph.state_estimation.graph import Frame, Node, Source

STRESS_CASES = (
    "fixed_hole",
    "moving_hole",
    "moving_peg",
    "partial_occlusion",
    "long_occlusion",
    "reappearance",
    "sparse_surface",
    "vertex_permutation",
    "tracker_drift",
    "unseen_shape",
)

VERTEX_SIGNS = np.array(
    [
        [-1, -1, -1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
    ]
)


def synthetic_episodes(steps=12, *, scenario="independent_motion"):
    """Build disjoint deterministic train, validation and test episodes."""
    if type(steps) is not int or steps < 1:
        raise ValueError("Synthetic episodes require a positive step count")
    if scenario not in (*STRESS_CASES, "independent_motion"):
        raise ValueError(f"Unknown synthetic scenario: {scenario}")
    splits = ("train", "validation", "test")
    return [
        _episode(
            f"{scenario}_{split}_{variant}", split, variant + group * 2, steps, scenario
        )
        for group, split in enumerate(splits)
        for variant in range(2)
    ]


def _episode(identity, split, variant, steps, scenario):
    samples = []
    size = np.array([0.027, 0.009, 0.010]) + (variant % 2) * np.array(
        [0.020, 0.006, 0.007]
    )
    radius = 0.014 + 0.012 * (variant % 2)
    if scenario == "unseen_shape":
        size = size * 1.4
        radius *= 0.7

    for index in range(steps):
        time = index / max(steps - 1, 1)
        tcp = np.array([0.004 * variant, 0.002 * time, 0.45])
        peg = np.array([-0.035 + 0.007 * variant, -0.036 + 0.025 * time, 0.46])
        hole = np.array([0.044 - 0.005 * variant, 0.04 - 0.018 * time, 0.46])
        if scenario in ("fixed_hole", "moving_peg"):
            hole[1] = 0.04
        if scenario == "moving_hole":
            peg[1] = -0.036
        if scenario == "moving_peg":
            peg[1] = -0.039 + 0.043 * time
            peg[2] += 0.006 * math.sin(math.pi * time)
        angles = (
            0.16 + variant * 0.24 + time * 0.4,
            -0.24 - variant * 0.17 + time * 0.21,
        )
        nodes = []
        for kind, (center, extent, angle) in enumerate(
            (
                (peg, size, angles[0]),
                (hole, np.array([0.012, radius, radius]), angles[1]),
            )
        ):
            nodes.extend(
                _body_nodes(kind, center, extent, angle, index, steps, scenario)
            )
        if scenario == "vertex_permutation" and index % 2:
            nodes.reverse()
        target = object21_from_gt(
            {
                "ee_pose": [[*tcp, 1, 0, 0, 0]],
                "peg_pose": [
                    [*peg, math.cos(angles[0] / 2), 0, 0, math.sin(angles[0] / 2)]
                ],
                "hole_pose": [
                    [*hole, math.cos(angles[1] / 2), 0, 0, math.sin(angles[1] / 2)]
                ],
                "peg_half_sizes": [size.tolist()],
                "hole_radius": [radius],
            }
        )[0]
        frame = Frame(identity, index, tuple(tcp), tuple(nodes))
        digest = canonical_hash([identity, index])
        diagnostics = (
            {"injected_tracker_drift_mm": _drift_m(index) * 1000}
            if scenario == "tracker_drift"
            else None
        )
        samples.append(Sample(frame, index, target, digest, diagnostics))
    return Episode(identity, identity, split, tuple(samples))


def _body_nodes(kind, center, extent, angle, index, steps, scenario):
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    vertices = VERTEX_SIGNS * extent @ rotation.T + center
    surfaces = (
        np.array([[0.8, 0.15, 1], [-0.8, -0.15, 1], [0, 1, 0.2]]) * extent @ rotation.T
        + center
    )
    occluded = kind == 1 and steps // 4 <= index < 3 * steps // 4
    partial = scenario == "partial_occlusion" and occluded
    long_occlusion = scenario == "long_occlusion" and kind == 1 and index >= steps // 4
    fully_hidden = long_occlusion or (scenario == "reappearance" and occluded)
    nodes = []
    for vertex_id, point in enumerate(vertices):
        missing = fully_hidden or (partial and vertex_id % 2 == 0)
        if missing:
            nodes.append(Node(None, kind, True, vertex_id, False, None, Source.MISSING))
        else:
            nodes.append(
                Node(tuple(point), kind, True, vertex_id, True, 0.9, Source.OBSERVED)
            )
    if not fully_hidden:
        for point in surfaces[:1] if scenario == "sparse_surface" else surfaces:
            if scenario == "tracker_drift":
                drift = np.array([_drift_m(index), 0.0, 0.0])
                nodes.append(
                    Node(
                        tuple(point + drift),
                        kind,
                        False,
                        None,
                        False,
                        0.7,
                        Source.TRACKER,
                    )
                )
            else:
                nodes.append(
                    Node(tuple(point), kind, False, None, True, 0.85, Source.OBSERVED)
                )
    return nodes


def _drift_m(frame_index):
    return min(0.028, frame_index * 0.0025)
