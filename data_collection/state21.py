"""Episode-aligned Object21D datasets and source lineage checks."""

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.state_estimation.graph import Frame, Node, Source


@dataclass(frozen=True)
class Sample:
    """Targets are stored separately from the observation accepted by inference."""

    observation: Frame
    target_index: int
    target: torch.Tensor
    frame_sha256: str | None = None
    diagnostics: dict | None = None


@dataclass(frozen=True)
class Episode:
    """A trajectory with an explicit pre-window parent and split."""

    episode_id: str
    parent_id: str
    split: str
    samples: tuple[Sample, ...]
    reset_fingerprint: str | None = None


def audit_splits(episodes, *, tracker_parents=(), near_threshold_m=0.002):
    """Reject cross-split origins and report similar TCP trajectories separately."""
    owners = {}
    ids = set()
    counts = {}
    signatures = []
    observed_ids = {}
    shapes = {}
    tracker_parents = set(tracker_parents)
    for episode in episodes:
        if (
            not episode.episode_id
            or not episode.parent_id
            or episode.split not in ("train", "validation", "test")
        ):
            raise ValueError("Episode needs an ID, original parent and valid split")
        if episode.episode_id in ids:
            raise ValueError(f"duplicate episode ID: {episode.episode_id}")
        ids.add(episode.episode_id)
        if not episode.samples:
            raise ValueError(f"Empty episode: {episode.episode_id}")
        if episode.split != "train" and episode.parent_id in tracker_parents:
            raise ValueError(
                f"Tracker training or selection overlaps held-out parent {episode.parent_id}"
            )
        counts[episode.split] = counts.get(episode.split, 0) + 1
        previous = -1
        tcp = []
        for sample in episode.samples:
            frame = sample.observation
            if frame.episode_id != episode.episode_id or frame.index <= previous:
                raise ValueError(
                    f"Invalid episode/frame ordering in {episode.episode_id}"
                )
            if sample.target_index != frame.index:
                raise ValueError(
                    f"Misaligned target frame in {episode.episode_id}:{frame.index}"
                )
            if sample.target.shape != (21,) or not torch.isfinite(sample.target).all():
                raise ValueError("Expected a finite Object21 target vector")
            if sample.frame_sha256 is not None:
                check_split_owner(
                    owners, "source frame", sample.frame_sha256, episode.split
                )
            observed_ids[episode.split] = observed_ids.get(episode.split, 0) + sum(
                node.visible and node.vertex_id is not None for node in frame.nodes
            )
            shapes.setdefault(episode.split, set()).add(
                tuple(sample.target[17:].round(decimals=6).tolist())
            )
            tcp.append(frame.tcp)
            previous = frame.index
        check_split_owner(owners, "parent", episode.parent_id, episode.split)
        if episode.reset_fingerprint:
            check_split_owner(
                owners,
                "initial scene fingerprint",
                episode.reset_fingerprint,
                episode.split,
            )
        positions = np.asarray(tcp, dtype=float)
        if positions.shape != (len(tcp), 3) or not np.isfinite(positions).all():
            raise ValueError("Audit requires finite observed TCP xyz for every frame")
        grid = np.linspace(0, 1, len(tcp))
        signature = np.stack(
            [np.interp(np.linspace(0, 1, 16), grid, axis) for axis in positions.T],
            axis=-1,
        )
        signatures.append((episode, signature))

    similar = []
    for (left, a), (right, b) in combinations(signatures, 2):
        if left.split != right.split:
            distance = np.linalg.norm(a - b, axis=-1).mean()
            if distance <= near_threshold_m:
                similar.append(
                    {
                        "episodes": [left.episode_id, right.episode_id],
                        "mean_tcp_distance_m": float(distance),
                    }
                )
    return {
        "episode_counts": counts,
        "near_duplicate_tcp_trajectories": similar,
        "verified_visible_vertex_rows": observed_ids,
        "distinct_size_targets": {
            split: len(values) for split, values in shapes.items()
        },
    }


def check_split_owner(owners, kind, identity, split):
    key = (kind, identity)
    if owners.setdefault(key, split) != split:
        raise ValueError(f"The same {kind} crosses splits: {identity}")


def sequence_windows(episodes, history):
    """Emit causal windows of saved frames without crossing episode boundaries."""
    if type(history) is not int or history < 1:
        raise ValueError("history must be a positive number of saved frames")
    for episode in episodes:
        for end, sample in enumerate(episode.samples):
            start = max(0, end + 1 - history)
            frames = [item.observation for item in episode.samples[start : end + 1]]
            yield frames, sample.target


def load_observation_manifest(path):
    """Load explicit world-metre nodes and synchronized targets from a JSON manifest."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (
        manifest.get("format") != "state21_observations"
        or manifest.get("coordinates") != "world_m"
    ):
        raise ValueError("Expected state21_observations with world_m coordinates")
    provenance = manifest["provenance"]
    if provenance.get("tracker") != "TAPNext++" or not is_sha256(
        provenance.get("tracker_sha256")
    ):
        raise ValueError(
            "Observation manifest requires TAPNext++ tracker provenance and checkpoint SHA-256"
        )
    tracker_parents = provenance.get("tracker_training_or_selection_parents")
    if not isinstance(tracker_parents, list) or not all(
        isinstance(parent, str) and parent for parent in tracker_parents
    ):
        raise ValueError(
            "Observation manifest requires the tracker training or selection parent list"
        )

    episodes = []
    for row in manifest["episodes"]:
        episode_id = row["episode_id"]
        samples = []
        for frame in row["frames"]:
            if not is_sha256(frame.get("source_frame_sha256")):
                raise ValueError(
                    "Observation manifest requires a source frame SHA-256 for every frame"
                )
            nodes = tuple(_read_node(node) for node in frame["nodes"])
            observation = Frame(
                episode_id, frame["frame_index"], frame["tcp_world_m"], nodes
            )
            samples.append(
                Sample(
                    observation,
                    frame["target_frame_index"],
                    torch.tensor(frame["object21"], dtype=torch.float32),
                    frame["source_frame_sha256"],
                )
            )
        episodes.append(
            Episode(
                episode_id,
                row["parent_id"],
                row["split"],
                tuple(samples),
                row.get("reset_fingerprint"),
            )
        )
    report = audit_splits(episodes, tracker_parents=tracker_parents)
    return episodes, {
        "manifest_sha256": file_sha256(path),
        "provenance": provenance,
        **report,
    }


def _read_node(row):
    """Bind an upstream vertex only when the observation certifies its identity."""
    verified = row.get("identity_verified", False)
    if type(verified) is not bool or (verified and row.get("vertex_id") is None):
        raise ValueError(
            "identity_verified requires a boolean flag and a vertex ID when true"
        )
    try:
        source = Source[row["source"].upper()]
    except (AttributeError, KeyError) as error:
        raise ValueError("Expected source observed, tracker or missing") from error
    return Node(
        row.get("position"),
        row["object_id"],
        row["is_vertex"],
        row["vertex_id"] if verified else None,
        row["visible"],
        row.get("confidence"),
        source,
    )


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def load_dataset(settings, *, splits=None):
    """Load the configured data source and return its auditable identity."""
    kind = settings["kind"]
    if kind == "synthetic":
        from data_collection.episode_dataset import canonical_hash
        from data_collection.state21_synthetic import synthetic_episodes

        steps = settings.get("steps", 12)
        scenario = settings.get("scenario", "independent_motion")
        episodes = synthetic_episodes(steps, scenario=scenario)
        report = audit_splits(episodes)
        identity = canonical_hash(
            {
                "generator_sha256": file_sha256(
                    Path(__file__).with_name("state21_synthetic.py")
                ),
                "steps": steps,
                "scenario": scenario,
            }
        )
        return episodes, {
            "manifest_sha256": identity,
            "provenance": {
                "tracker": "synthetic",
                "tracker_training_or_selection_parents": [],
            },
            **report,
        }
    if kind == "observations":
        return load_observation_manifest(settings["manifest"])
    if kind == "tapnext_state36":
        from data_collection.state21_shared_tapnext import load_shared_tapnext

        return load_shared_tapnext(settings, splits=splits)
    if kind == "tapnext_surface_cache":
        from data_collection.state21_tapnext import load_tapnext_surface_manifest

        return load_tapnext_surface_manifest(
            settings["manifest"],
            source_manifest=settings.get("source_manifest"),
            camera=settings.get("camera", "base_camera"),
        )
    raise ValueError(f"Unknown state estimation data kind: {kind}")
