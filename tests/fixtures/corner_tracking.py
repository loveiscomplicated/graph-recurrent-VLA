"""Synthetic observations and native-shaped, checksummed corner-family files."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.temporal.corner_tracking_synthetic import synthetic_dataset
from data_collection.vertex_family_labels import project_world
from graph.build_graph import GraphBuilder
from graph.temporal.corner_tracking.tracker import AssociationCalibration, Observation
from training.gnn.data import TrainingBatch, save_cached_sample


def observation(points=(), order=None):
    """Create a finite peg-only observation with a fixed observed robot pose."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if order is not None:
        points = points[order]
    return Observation(points, np.zeros(len(points), dtype=np.int64),
                       np.zeros(len(points), dtype=np.float32), np.zeros(len(points), dtype=bool),
                       np.asarray([0, 0, 0, 1, 0, 0, 0, 0.02, 0.02], dtype=np.float32))


def calibration(radius=0.02):
    """Use the same distance gate for each object kind and missing age."""
    return AssociationCalibration(np.full((3, 33), radius), np.ones(3))


def _write_corner_family_frame(root, episode, frame, intrinsic, extrinsic):
    """Write one matched raw frame, graph cache and physical-certificate archive."""
    raw_path = root / "raw" / frame.sample_id
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    peg = frame.observation.robot[:3]
    hole = frame.reference_xyz[8] + 0.05
    raw = {
        "observation/extra/tcp_pose": frame.observation.robot[None, :7],
        "observation/agent/qpos": np.asarray([[0] * 7 + [0.015, 0.015]], dtype=np.float32),
        "ground_truth/ee_pose": frame.observation.robot[None, :7],
        "ground_truth/peg_pose": np.asarray([[*peg, 1, 0, 0, 0]], dtype=np.float32),
        "ground_truth/box_pose": np.asarray([[*hole, 1, 0, 0, 0]], dtype=np.float32),
        "ground_truth/hole_pose": np.asarray([[*hole, 1, 0, 0, 0]], dtype=np.float32),
        "ground_truth/peg_half_sizes": np.asarray([[0.05, 0.015, 0.015]], dtype=np.float32),
        "ground_truth/hole_radius": np.asarray([0.018], dtype=np.float32),
        "observation/sensor_data/base_camera/rgb": np.zeros((1, 512, 512, 3), dtype=np.uint8),
        "observation/sensor_data/base_camera/depth": np.zeros((1, 512, 512, 1), dtype=np.int16),
        "observation/sensor_param/base_camera/intrinsic_cv": intrinsic[None],
        "observation/sensor_param/base_camera/extrinsic_cv": extrinsic[None],
    }
    np.savez_compressed(raw_path, **raw)

    source_sha256 = file_sha256(raw_path)
    identity = dict(sample_id=frame.sample_id, parent_id=episode.parent, source_sha256=source_sha256)
    detection = SimpleNamespace(xyz=torch.from_numpy(frame.observation.xyz),
                                object_id=torch.from_numpy(frame.observation.kind),
                                metadata=[{} for _ in frame.physical_ids])
    graph = GraphBuilder()([detection], torch.from_numpy(peg)[None])
    stem = canonical_hash(frame.sample_id)
    graph_path = root / f"{stem}_graph.npz"
    targets = {"object21": torch.from_numpy(frame.target_object21)[None]}
    sample = TrainingBatch(graph, targets, [frame.sample_id])
    save_cached_sample(graph_path, sample, identity)

    label_path = root / f"{stem}_labels.npz"
    uv, _ = project_world(frame.observation.xyz, intrinsic, extrinsic)
    physical_ids = np.where(frame.observation.kind == 1, frame.physical_ids - 8, -1)
    np.savez_compressed(label_path, xyz_world=frame.observation.xyz, uv=uv, physical_id=physical_ids)
    return dict(**identity, split=episode.split, sample_identity=identity,
                graph_path=graph_path.name, graph_sha256=file_sha256(graph_path),
                labels_path=label_path.name, labels_sha256=file_sha256(label_path),
                raw_path=str(raw_path.relative_to(root)))


def write_corner_family_source(root):
    """Write three complete episodes with original-shaped RGB-D, graph and labels.

    The two training parents allow the adapter to reserve one for calibration.
    Return the manifest so a test can independently alter its own source files.
    """
    groups, _ = synthetic_dataset(frames=4, training_episodes=2)
    root = Path(root)
    root.mkdir()
    rows = []
    intrinsic = np.asarray([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=np.float32)
    extrinsic = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5]], dtype=np.float32)
    for episode in groups["train"] + groups["validation"]:
        for frame in episode.frames:
            rows.append(_write_corner_family_frame(root, episode, frame, intrinsic, extrinsic))
    manifest = dict(original_pool="train_only", reserved_test_frames_read=0, samples=rows,
                    selection={"frames_per_parent": None})
    manifest["dataset_sha256"] = canonical_hash(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest
