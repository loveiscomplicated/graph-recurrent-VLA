"""Restore frozen perception and prepare identity-checked temporal observations."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.frozen_episodes import infer_observations
from data_collection.temporal.insertion_sequences import select_insertion_loaders
from data_collection.temporal.pose_sequences import attach_proprioception
from data_collection.temporal_gate_data import selected_loaders
from graph.temporal.observations import OBSERVATION_CONTRACT, OBSERVATION_NAMES
from graph.temporal.pose_observations import BASE_NAMES, POSE_CONTRACT, ROBOT_NAMES
from training.gnn.trainer import build_model, resolve_config


def load_frozen_source(effective, device):
    """Load exact best.pt architecture/state head and disable all parameter gradients."""
    checkpoint = torch.load(effective["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "best_weights":
        raise ValueError("checkpoint must be the original best_weights best.pt")
    source = resolve_config(checkpoint["config"])
    if source["model"]["name"] != "vertex_edge_gated_mpnn":
        raise ValueError("Expected the existing vertex_edge_gated_mpnn checkpoint")
    if source["data"]["factory"] != "training.gnn.cached_data:build_data":
        raise ValueError("Expected a published cache-only GNN checkpoint")
    if source["data"]["empty_inputs"] != "keep":
        raise ValueError("Temporal gate requires empty_inputs=keep")
    source["data"]["cache_dir"] = effective["data"]["cache_dir"]
    source["training"]["batch_size"] = effective["evaluation"]["batch_size"]
    model = build_model(source["model"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, source, checkpoint["dataset_sha256"]


def prepare_sequences(effective, device, requested, loader_factory=selected_loaders):
    """Verify frozen checkpoint/cache/parent identity and record the feature contract."""
    frozen, source, expected = load_frozen_source(effective, device)
    loaders, identity, manifest = loader_factory(
        source,
        torch.Generator().manual_seed(effective["seed"]),
        effective["split_manifest"],
        requested,
    )
    if identity["dataset_sha256"] != expected:
        raise ValueError("Frozen checkpoint and graph cache dataset identities differ")
    effective.update(
        source_training_config=source,
        source_checkpoint_sha256=file_sha256(Path(effective["checkpoint"])),
        split_manifest_sha256=file_sha256(Path(effective["split_manifest"])),
        temporal_split_manifest=manifest,
        observation_contract=OBSERVATION_CONTRACT,
        feature_names=list(OBSERVATION_NAMES),
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
    )
    repository = Path(__file__).resolve().parents[3]
    sources = (
        "graph/temporal/hole_gru.py",
        "graph/temporal/observations.py",
        "graph/temporal_gate.py",
        "graph/gated_mpnn.py",
        "graph/utils/hole_memory.py",
        "data_collection/temporal/frozen_episodes.py",
        "data_collection/temporal_gate_data.py",
        "training/gnn/temporal/hole_gru.py",
        "training/gnn/temporal_gate.py",
        "training/gnn/cached_data.py",
        "evaluation/gnn/temporal/hole_gru.py",
        "evaluation/gnn/temporal_gate.py",
        "evaluation/gnn/temporal_memory.py",
    )
    sources += ("training/gnn/temporal/config.py", "training/gnn/temporal/frozen_source.py")
    effective["source_sha256"] = {name: file_sha256(repository / name) for name in sources}
    return frozen, loaders, identity


def prepare_pose_sequences(effective, device, requested):
    """Preserve checkpoint/cache/split validation and record expanded observation identity."""
    manifest = json.loads(Path(effective["split_manifest"]).read_text())
    if manifest.get("test_role") == "independent":
        from data_collection.temporal.insertion_sequences import (
            select_insertion_loaders,
        )

        def independent_loaders(source, generator, path, splits):
            return select_insertion_loaders(
                source, path, splits, generator, effective["data"]["independent_cache_dir"]
            )

        frozen, loaders, identity = prepare_sequences(
            effective, device, requested, independent_loaders
        )
    else:
        frozen, loaders, identity = prepare_sequences(effective, device, requested)
    effective["feature_names"] = list(BASE_NAMES + (ROBOT_NAMES if effective["with_robot"] else ()))
    effective["observation_contract"] = POSE_CONTRACT
    effective["sequence_contract"] = "complete_episode_zero_hidden_right_padding_v1"
    effective["normalization_definition"] = "train_population_mean_std_constant_scale_one_v1"
    effective["precision_evaluation_contract"] = "peg_head_plus_x_hole_x_axial_yz_lateral_v1"
    repository = Path(__file__).resolve().parents[3]
    for name in (
        "graph/temporal/pose_gru.py",
        "graph/temporal/pose_observations.py",
        "graph/pose_geometry.py",
        "data_collection/temporal/pose_sequences.py",
        "training/gnn/temporal/pose_gru.py",
        "training/gnn/temporal/pose_loss.py",
        "evaluation/gnn/temporal/pose_gru.py",
    ):
        effective["source_sha256"][name] = file_sha256(repository / name)
    effective["source_sha256"]["evaluation/gnn/temporal/precision.py"] = file_sha256(
        repository / "evaluation/gnn/temporal/precision.py"
    )
    return frozen, loaders, identity


def infer_pose_data(frozen, loader, device, identity, effective):
    """Capture expanded vertex quality and optionally join verified observed robot data."""
    data = infer_observations(frozen, loader, device, include_pose_quality=True)
    raw_root = effective["data"]["raw_root"]
    if "independent_frames" in identity:
        independent_ids = {frame["id"] for frame in identity["independent_frames"]}
        if set(data["sample_ids"]) <= independent_ids:
            identity = dict(identity, frames=identity["independent_frames"])
            if effective["with_robot"]:
                raw_root = effective["data"]["independent_raw_root"]
                if raw_root is None:
                    raise ValueError(
                        "Independent observed-robot evaluation requires data.independent_raw_root"
                    )
    return attach_proprioception(data, identity, raw_root)


def prepare_insertion_sequences(effective, device, requested):
    """Verify frozen source identity and separate reference from independent testing."""
    frozen, source, expected = load_frozen_source(effective, device)
    loaders, identity, manifest = select_insertion_loaders(
        source,
        effective["split_manifest"],
        requested,
        torch.Generator().manual_seed(effective["seed"]),
        effective["data"]["independent_cache_dir"],
    )
    if identity["dataset_sha256"] != expected:
        raise ValueError("Frozen GNN and source cache dataset identities differ")
    effective.update(
        source_checkpoint_sha256=file_sha256(effective["checkpoint"]),
        split_manifest_sha256=file_sha256(effective["split_manifest"]),
        temporal_split_manifest=manifest,
        source_training_config=source,
    )
    repository = Path(__file__).resolve().parents[3]
    sources = (
        "graph/temporal/insertion.py",
        "graph/temporal/pose_observations.py",
        "graph/pose_geometry.py",
        "graph/gated_mpnn.py",
        "graph/temporal_gate.py",
        "graph/utils/hole_memory.py",
        "data_collection/temporal/insertion_sequences.py",
        "data_collection/temporal/pose_sequences.py",
        "data_collection/temporal/frozen_episodes.py",
        "data_collection/temporal_gate_data.py",
        "training/gnn/temporal/insertion.py",
        "training/gnn/temporal/pose_loss.py",
        "training/gnn/cached_data.py",
        "evaluation/gnn/temporal/insertion.py",
        "evaluation/gnn/temporal/precision.py",
    )
    sources += ("training/gnn/temporal/config.py", "training/gnn/temporal/frozen_source.py")
    effective["source_sha256"] = {name: file_sha256(repository / name) for name in sources}
    return frozen, loaders, identity


def infer_insertion_data(frozen, loader, device, identity, effective, split):
    """Use observed proprioception from the matching source, never GT robot fallback."""
    settings = effective
    if split == "test" and effective["temporal_split_manifest"]["test_role"] == "independent":
        settings = deepcopy(effective)
        settings["data"]["raw_root"] = effective["data"]["independent_raw_root"]
        if effective["with_robot"] and settings["data"]["raw_root"] is None:
            raise ValueError(
                "Independent test requires matching observed data.independent_raw_root"
            )
        identity = dict(identity, frames=identity["independent_frames"])
    data = infer_pose_data(frozen, loader, device, identity, settings)
    if ("tcp_quaternion" in data) != effective["with_robot"]:
        raise ValueError("Source and independent feature schemas must match")
    return data
