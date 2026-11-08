"""Prepare a provenance-preserving development cohort from original TRAIN only.

Storage and sample selection stay in data_collection. Graph definitions and
training remain team-owned. This does not alter the published original split.
"""

from collections import Counter
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256, read_frame
from data_collection.storage import write_json
from data_collection.vertex_family_cache import FamilyArchiveCache, ExistingGraphCache
from data_collection.vertex_family_labels import (
    CertificationSettings, detector_families, oracle_families, physical_references,
)
from graph.build_graph import GraphBuilder
from graph.utils.object21_targets import object21_from_gt
from training.gnn.data import TrainingBatch, canonical_hash, save_cached_sample
from utils.experiment_logger import ExperimentLogger


def select_development_frames(config, identity, parent_manifest):
    """Freeze parent selection/split and uniform temporal sampling before detection."""
    source_splits = identity["splits"]
    parents = parent_manifest["parent_by_episode"]
    parent_sets = {name: {parents[episode] for episode in episodes}
                   for name, episodes in source_splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if parent_sets[left] & parent_sets[right]:
            raise ValueError(f"Original parent leakage: {left} overlaps {right}")
    train_parent_count = config["selection"]["train_parents"]
    validation_parent_count = config["selection"]["validation_parents"]
    frames_per_parent = config["selection"]["frames_per_parent"]
    for name, value in (("train_parents", train_parent_count), ("validation_parents", validation_parent_count)):
        if type(value) is not int or value < 1:
            raise ValueError(f"selection.{name} must be a positive integer")
    if frames_per_parent is not None and (type(frames_per_parent) is not int or frames_per_parent < 1):
        raise ValueError("selection.frames_per_parent must be null or a positive integer")
    train_episodes = source_splits["train"]
    if len({parents[episode] for episode in train_episodes}) != len(train_episodes):
        raise ValueError("This cohort sampler requires one original episode per parent")
    selection_seed = config["selection"]["seed"]
    ranked = sorted(train_episodes, key=lambda episode: canonical_hash([selection_seed, parents[episode]]))
    requested = train_parent_count + validation_parent_count
    if requested > len(ranked):
        raise ValueError("Requested development parents exceed the original TRAIN pool")
    validation_selection = config["selection"].get("validation_selection")
    if validation_selection is None:
        selected = ranked[:requested]
        validation_episodes = set(selected[train_parent_count:])
    else:
        previous = json.loads(Path(validation_selection).read_text())
        validation_episodes = set(previous["splits"]["validation"])
        if len(validation_episodes) != validation_parent_count or not validation_episodes <= set(ranked):
            raise ValueError("validation_selection must identify the requested original TRAIN episodes")
        training_episodes = [episode for episode in ranked if episode not in validation_episodes]
        selected = training_episodes[:train_parent_count] + [
            episode for episode in ranked if episode in validation_episodes
        ]
    frame_hashes = {frame["id"]: frame["sha256"] for frame in identity["frames"]}
    by_episode = {episode: [] for episode in selected}
    for identifier in frame_hashes:
        episode = str(Path(identifier).parent)
        if episode in by_episode:
            by_episode[episode].append(identifier)
    rows = []
    splits = {"train": [], "validation": []}
    for index, episode in enumerate(selected):
        split = "validation" if episode in validation_episodes else "train"
        splits[split].append(episode)
        identifiers = sorted(by_episode[episode])
        if not identifiers:
            raise ValueError(f"Original TRAIN episode has no recorded frame identities: {episode}")
        if frames_per_parent is None:
            positions = range(len(identifiers))
        else:
            positions = np.unique(np.rint(np.linspace(
                0, len(identifiers) - 1, min(frames_per_parent, len(identifiers))
            )).astype(int))
        for position in positions:
            identifier = identifiers[position]
            rows.append(dict(sample_id=identifier, episode=episode, parent_id=parents[episode],
                             split=split, source_sha256=frame_hashes[identifier]))
    return rows, splits


def prepare(config):
    """Preserve raw observations, retained proposal evidence, graphs and GT annotations."""
    from visual.corner_detection import Parameters, TorchCornerDetector

    identity = json.loads(Path(config["source_identity"]).read_text())
    parent_manifest = json.loads(Path(config["source_parent_manifest"]).read_text())
    rows, splits = select_development_frames(config, identity, parent_manifest)
    detector = TorchCornerDetector(config["detector"]["device"], Parameters(**config["detector"].get("parameters", {})))
    certification = CertificationSettings(**config.get("certification", {}))
    builder = GraphBuilder()
    family_cache = FamilyArchiveCache(config.get("reuse_family", []), identity, detector, certification)
    graph_cache = ExistingGraphCache(config.get("reuse_graph_cache"), identity, builder)
    source_root = Path(config["source_root"]).resolve()
    raw_storage = config.get("raw_storage", "copy")
    if raw_storage not in ("copy", "hardlink"):
        raise ValueError("raw_storage must be copy or hardlink")
    with ExperimentLogger(team="graph", config=config, seed=config["selection"]["seed"],
                          model="vertex_family_development", job_type="data_preparation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        for directory in ("graphs", "labels", "raw"):
            (output / directory).mkdir()
        write_json(output / "selection.json", dict(splits=splits, samples=rows, original_pool="train_only"))
        write_json(output / "effective_config.json", config)
        manifest_rows = []
        reason_counts = Counter()
        detector_counts = Counter()
        oracle_counts = Counter()
        start = time.monotonic()
        reuse_counts = Counter()
        for index, row in enumerate(rows):
            source = source_root / row["sample_id"]
            if not source.is_file() or file_sha256(source) != row["source_sha256"]:
                raise ValueError(f"Original TRAIN raw frame missing or hash mismatch: {source}")
            key = canonical_hash(row["sample_id"])
            cache_path = Path("graphs") / f"{key}.npz"
            label_path = Path("labels") / f"{key}.npz"
            raw_path = Path("raw") / row["sample_id"]
            sample_identity = {name: row[name] for name in ("sample_id", "parent_id", "source_sha256")}
            cached = family_cache.find(row)
            if cached is not None:
                os.link(cached["graph"], output / cache_path)
                os.link(cached["labels"], output / label_path)
                object_ids = cached["object_ids"]
                detector_labels = cached["detector_labels"]
                oracle_labels = cached["oracle_labels"]
                reasons = cached["reasons"]
                reuse_counts["complete_family"] += 1
            else:
                observation, gt = read_frame(source, config.get("camera", "base_camera"))
                result = detector(observation["rgb"], observation["depth"], observation["intrinsic_cv"], observation["extrinsic_cv"])
                xyz = result.xyz.cpu().numpy()
                uv = result.uv.cpu().numpy()
                object_ids = result.object_id.cpu().numpy()
                detector_labels = detector_families(object_ids, result.metadata)
                oracle_labels, physical_id, reasons = oracle_families(xyz, uv, object_ids, observation, gt, certification)
                tcp = torch.as_tensor(gt["ee_pose"][:, :3], dtype=torch.float32)
                target = object21_from_gt(gt)
                sample = graph_cache.find(row, result, tcp, target)
                if sample is None:
                    sample = TrainingBatch(builder([result], tcp), {"object21": target}, [row["sample_id"]])
                    reuse_counts["new_graph"] += 1
                else:
                    reuse_counts["existing_graph"] += 1
                save_cached_sample(output / cache_path, sample, sample_identity)
                references = physical_references(gt)
                np.savez_compressed(output / label_path,
                    detector_family=detector_labels, oracle_family=oracle_labels, physical_id=physical_id,
                    xyz_world=xyz, uv=uv, object_type=object_ids,
                    metadata_json=np.array(json.dumps(result.metadata)), reason_json=np.array(json.dumps(reasons)),
                    detector_diagnostics_json=np.array(json.dumps(result.diagnostics)),
                    physical_xyz_world=np.asarray([reference["world"] for reference in references]),
                    physical_family=np.asarray([reference["family"] for reference in references]),
                    box_pose=gt["box_pose"], hole_pose=gt["hole_pose"],
                    peg_half_sizes=gt["peg_half_sizes"], hole_radius=gt["hole_radius"])
            raw_destination = output / raw_path
            raw_destination.parent.mkdir(parents=True, exist_ok=True)
            if raw_storage == "hardlink":
                os.link(source, raw_destination)
            else:
                shutil.copyfile(source, raw_destination)
            if file_sha256(raw_destination) != row["source_sha256"]:
                raise ValueError("Raw RGB-D preservation failed its byte-for-byte hash check")
            manifest_rows.append(dict(**row, graph_path=str(cache_path), labels_path=str(label_path),
                                      raw_path=str(raw_path), sample_identity=sample_identity,
                                      graph_sha256=file_sha256(output / cache_path), labels_sha256=file_sha256(output / label_path)))
            target_nodes = object_ids == 1
            detector_counts.update(map(int, detector_labels[target_nodes]))
            oracle_counts.update(map(int, oracle_labels[target_nodes]))
            reason_counts.update(reasons[node] for node in np.flatnonzero(target_nodes))
            if index % 24 == 0 or index + 1 == len(rows):
                print(f"Family dataset: {index + 1}/{len(rows)} frames; {time.monotonic() - start:.1f}s; "
                      f"reuse={dict(reuse_counts)}", flush=True)
        source_paths = [Path(__file__), Path("data_collection/vertex_family_cache.py"),
                        Path("data_collection/vertex_family_labels.py"),
                        *Path("visual/corner_detection").rglob("*.py"), *Path("graph/build_graph").glob("*.py")]
        manifest = dict(schema_version=1, original_pool="train_only", splits=splits, samples=manifest_rows,
                        source_dataset_sha256=identity["dataset_sha256"],
                        source_identity_sha256=file_sha256(config["source_identity"]),
                        source_parent_manifest_sha256=file_sha256(config["source_parent_manifest"]),
                        selection=config["selection"], raw_storage=raw_storage, certification=asdict(certification),
                        detector=dict(device=str(detector.device), parameters=asdict(detector.parameters)),
                        sources={str(path): file_sha256(path) for path in source_paths},
                        packages={name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "opencv-python-headless")},
                        reuse_counts=dict(reuse_counts),
                        label_counts=dict(detector=dict(detector_counts), oracle=dict(oracle_counts)),
                        uncertified_reasons=dict(reason_counts), reserved_test_frames_read=0)
        manifest["dataset_sha256"] = canonical_hash(manifest)
        write_json(output / "manifest.json", manifest)
        write_json(output / "effective_config.json", config)
        logger.record(dataset_sha256=manifest["dataset_sha256"], frames=len(rows),
                      development_train_parents=len(splits["train"]), development_validation_parents=len(splits["validation"]),
                      reserved_test_frames_read=0)
        for name in ("manifest.json", "effective_config.json", "selection.json"):
            logger.link_output(output / name)
        print(f"Vertex family development dataset: {output}", flush=True)
    return output
