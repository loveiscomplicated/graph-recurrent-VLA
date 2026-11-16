"""Certify TAPNext++ corner tracks against recorded simulator geometry offline."""

import hashlib
import io
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from visual.corner_detection.evaluation import corners

POSE_FIELDS = (
    ("peg_pose", 7),
    ("box_pose", 7),
    ("hole_pose", 7),
    ("peg_half_sizes", 3),
    ("hole_radius", 1),
)


def read_reference_pose(path, digest):
    """Read the raw, checksum-verified pose and size labels in one frame."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Raw frame checksum mismatch: {path}")
    with np.load(io.BytesIO(content), allow_pickle=False) as raw:
        return np.concatenate(
            [
                raw[f"ground_truth/{field}"][0].reshape(width)
                for field, width in POSE_FIELDS
            ]
        )


def corners_from_pose(pose):
    """Reconstruct physical corners from the archived GT fields."""
    start = 0
    gt = {}
    for field, width in POSE_FIELDS:
        gt[field] = pose[start : start + width]
        start += width
    return {
        (row["object_id"], row["kind"], row["corner_id"]): row["xyz"]
        for row in corners(gt)
    }


def reference_corners(path, digest):
    """Reconstruct physical corners from the poses and dimensions in one recording."""
    return corners_from_pose(read_reference_pose(path, digest))


def match_vertices(observed, references, *, gate_m, margin_m):
    """Accept one-to-one nearest matches with a margin to the next candidate."""
    matches = {}
    for kind in (0, 1):
        track_ids = [key for key in observed if key[0] == kind]
        corner_ids = [key for key in references if key[0] == kind]
        if not track_ids or not corner_ids:
            continue
        points = np.asarray([observed[key] for key in track_ids])
        targets = np.asarray([references[key] for key in corner_ids])
        distances = np.linalg.norm(points[:, None] - targets[None, :], axis=-1)
        nearest = np.argmin(distances, axis=1)
        valid = distances <= gate_m
        if len(corner_ids) > 1:
            closest = np.partition(distances, 1, axis=1)[:, :2]
            valid &= (closest[:, 1] - closest[:, 0] >= margin_m)[:, None]
        valid &= np.arange(len(corner_ids))[None, :] == nearest[:, None]
        cost = np.where(valid, distances, 1.0 + gate_m * len(track_ids))
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols, strict=True):
            if valid[row, col]:
                matches[track_ids[row]] = corner_ids[col]
    return matches


def annotate_vertices(episode, raw_root, anchor_index, *, gate_m, margin_m):
    """Freeze identities at the track anchor and recheck visible points per frame."""
    root = Path(raw_root).resolve()
    directory = (root / episode.episode_id).resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f"Invalid episode location: {episode.episode_id}")

    def observed_at(sample):
        return {
            (node.object_id, node.vertex_id): node.position
            for node in sample.observation.nodes
            if node.visible and node.is_vertex and node.vertex_id is not None
        }

    def references_at(sample):
        frame_path = directory / f"frame_{sample.observation.index:06d}.npz"
        return reference_corners(frame_path, sample.frame_sha256)

    return _annotate(
        episode, anchor_index, gate_m, margin_m, observed_at, references_at
    )


def annotate_graph_vertices(episode, graphs, poses, anchor_index, *, gate_m, margin_m):
    """Label the same observations from a saved graph bundle and GT pose archive."""

    def observed_at(sample):
        graph = graphs[sample.observation.index]
        rows = (graph.visible & (graph.vertex_id >= 0)).nonzero(as_tuple=True)[0]
        return {
            (kind, identity): xyz
            for kind, identity, xyz in zip(
                graph.object_id[rows].tolist(),
                graph.vertex_id[rows].tolist(),
                graph.position[rows].tolist(),
                strict=True,
            )
        }

    def references_at(sample):
        return corners_from_pose(poses[sample.observation.index])

    return _annotate(
        episode, anchor_index, gate_m, margin_m, observed_at, references_at
    )


def _annotate(episode, anchor_index, gate_m, margin_m, observed_at, references_at):
    """Keep geometric matches with consistent identity across the episode."""
    anchor_matches = {}
    samples = []
    counts = {kind: Counter() for kind in ("peg", "hole_assembly")}
    family_counts = Counter()
    for sample in episode.samples:
        frame = sample.observation
        observed = observed_at(sample)
        for kind, _ in observed:
            counts[("peg", "hole_assembly")[kind]]["visible"] += 1
        targets = {}
        if frame.index == anchor_index or observed.keys() & anchor_matches.keys():
            references = references_at(sample)
            confirmed = match_vertices(
                observed, references, gate_m=gate_m, margin_m=margin_m
            )
            if frame.index == anchor_index:
                anchor_matches = confirmed
                for kind, _ in confirmed:
                    counts[("peg", "hole_assembly")[kind]]["anchored"] += 1
            for key, reference in confirmed.items():
                if anchor_matches.get(key) == reference:
                    targets[key] = tuple(references[reference].tolist())
                    counts[("peg", "hole_assembly")[key[0]]]["matched"] += 1
                    family_counts[reference[1]] += 1
        samples.append(replace(sample, vertex_targets=targets))
    return replace(episode, samples=tuple(samples)), counts, family_counts
