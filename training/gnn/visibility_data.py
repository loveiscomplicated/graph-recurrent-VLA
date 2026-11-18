"""Load tracked-corner graphs for a single visibility condition."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data_collection.direct_corner_samples import read_direct_samples
from data_collection.episode_dataset import canonical_hash, file_sha256, read_frame
from data_collection.state21_shared_tapnext import _lift_tracks, _open_source
from data_collection.state21_vertex_targets import POSE_FIELDS, corners_from_pose, match_vertices
from graph.build_graph import NodeType
from graph.utils.object21_targets import object21_from_gt
from training.detected_corner_gnn import graph_batch
from training.gnn.data import DataBundle, TrainingBatch
from training.gnn.corner_jitter import fixed_validation_rows, jitter_positions, jitter_std_mm, read_std_mm, training_collate
from training.gnn.imported_direct_data import read_imported_train


def matched_observations(points, kinds, gt, settings):
    """Keep one node per matched physical vertex, using measured or GT coordinates."""
    pose = np.concatenate([gt[field][0].reshape(width) for field, width in POSE_FIELDS])
    observed = {(int(kinds[query]), query): xyz for query, xyz in points.items()}
    references = corners_from_pose(pose)
    matches = match_vertices(
        observed, references, gate_m=settings["gate_m"], margin_m=settings["margin_m"]
    )
    oracle = settings.get("coordinates", "observed") == "ground_truth"
    return {
        query: references[matches[key]] if oracle else xyz
        for query, xyz in points.items()
        if (key := (int(kinds[query]), query)) in matches
    }


def read_samples(source, selected, camera, samples, *, vertex_match=None, use_tcp=True):
    """Lift selected corner observations once and keep raw GT for supervision."""
    arrays = source.arrays
    for episode, entry in enumerate(source.entries):
        start, end = arrays["episode_offsets"][episode:episode + 2]
        first, last = arrays["corner_query_offsets"][episode:episode + 2]
        kinds = arrays["corner_query_object_id"][first:last]
        directory = source.raw_root / entry["path"]
        for row in range(start, end):
            path = directory / f"frame_{arrays['frame_indices'][row]:06d}.npz"
            split = selected.pop(path, None)
            if split is None:
                continue

            left, right = arrays["corner_frame_offsets"][row:row + 2]
            frame_index = arrays["frame_indices"][row]
            if file_sha256(path) != entry["frame_sha256"][frame_index]:
                raise ValueError(f"Raw frame checksum mismatch: {path}")
            obs, gt = read_frame(path, camera)
            depth = obs["depth"]
            if depth.ndim == 3:
                depth = depth[..., 0]
            points = _lift_tracks(
                arrays["corner_xy"][left:right],
                arrays["corner_visible"][left:right],
                depth,
                obs["intrinsic_cv"],
                obs["extrinsic_cv"],
            )
            if vertex_match is not None:
                points = matched_observations(points, kinds, gt, vertex_match)
            if not points:
                raise ValueError(f"No depth-valid tracked corners for {path}")
            xyz = torch.tensor(list(points.values()), dtype=torch.float32)
            types = torch.tensor(kinds[list(points)], dtype=torch.long)
            if not use_tcp:
                # Both the graph and Object21 position targets use the world origin.
                gt["ee_pose"][:, :3] = 0
            tcp = torch.as_tensor(gt["ee_pose"][0, :3], dtype=torch.float32)
            target = object21_from_gt(gt)[0]
            samples[split].append((xyz, types, tcp, target, str(path)))
        if (episode + 1) % 50 == 0:
            print(f"{source.raw_root.name}: loaded {episode + 1}/{len(source.entries)} episodes",
                  flush=True)


def collate(samples, *, jitter_std_m=0.0, corner_counts=None):
    """Build complete graphs using the existing observed-corner topology."""
    xyz, kinds, tcp, target, paths = zip(*samples)
    xyz = pad_sequence(xyz, batch_first=True)
    kinds = pad_sequence(kinds, batch_first=True, padding_value=NodeType.PAD)
    if kinds.shape[1] == 0:
        xyz = xyz.new_zeros((len(samples), 1, 3))
        kinds = kinds.new_full((len(samples), 1), NodeType.PAD)
    valid = kinds != NodeType.PAD
    xyz = jitter_positions(xyz, valid, paths, jitter_std_m, corner_counts)
    graph, _ = graph_batch(xyz, kinds, valid, valid, torch.stack(tcp))
    return TrainingBatch(graph, {"object21": torch.stack(target)}, list(paths))


def build_data(config, shuffle_generator):
    """Prepare only the indexed training and validation frames."""
    settings = config["data"]
    std_mm = jitter_std_mm(settings)
    validation_std_mm = read_std_mm(settings, "validation_corner_jitter_std_mm")
    if (std_mm or validation_std_mm) and (settings.get("vertex_match") is not None or any(
        "detection_manifest" not in source or "cache" in source for source in settings["sources"]
    )):
        raise ValueError("Corner jitter requires direct detector inputs")
    index_path = Path(settings["visibility_index"])
    groups = json.loads(index_path.read_text())[settings["condition"]]
    samples = {split: [] for split in ("train", "validation")}
    selected = {}
    for split in samples:
        if not groups[split]:
            raise ValueError(f"No {split} frames for {settings['condition']}")
        for name in groups[split]:
            path = Path(name).resolve()
            if path in selected:
                raise ValueError(f"Duplicate indexed frame: {path}")
            selected[path] = split

    identity = {
        "index_sha256": file_sha256(index_path),
        "condition": settings["condition"],
        "camera": settings["camera"],
        "sources": [],
    }
    vertex_match = settings.get("vertex_match")
    use_tcp = settings.get("use_tcp", True)
    if type(use_tcp) is not bool:
        raise ValueError("data.use_tcp must be a boolean")
    if not use_tcp:
        identity["use_tcp"] = False
    if vertex_match is not None:
        gate = vertex_match["gate_m"]
        margin = vertex_match["margin_m"]
        if not np.isfinite([gate, margin]).all() or not 0 < margin <= gate:
            raise ValueError("Expected 0 < vertex_match.margin_m <= vertex_match.gate_m")
        if vertex_match.get("coordinates", "observed") not in ("observed", "ground_truth"):
            raise ValueError("vertex_match.coordinates must be observed or ground_truth")
        identity["vertex_match"] = vertex_match
    for source_settings in settings["sources"]:
        if "detection_manifest" in source_settings:
            if vertex_match is not None:
                raise ValueError("vertex_match is only implemented for tracked corner inputs")
            source_identity = None
            for split, path, xyz, kinds, gt, source_identity in read_direct_samples(
                source_settings, selected, settings["camera"]
            ):
                if not use_tcp:
                    gt["ee_pose"][:, :3] = 0
                tcp = torch.as_tensor(gt["ee_pose"][0, :3], dtype=torch.float32)
                samples[split].append((
                    torch.as_tensor(xyz, dtype=torch.float32),
                    torch.as_tensor(kinds, dtype=torch.long),
                    tcp,
                    object21_from_gt(gt)[0],
                    str(path),
                ))
            if source_identity is not None:
                identity["sources"].append(source_identity)
            continue
        source = _open_source(source_settings)
        read_samples(
            source, selected, settings["camera"], samples,
            vertex_match=vertex_match, use_tcp=use_tcp,
        )
        identity["sources"].append({
            "cache_sha256": source.archive_sha256,
            "manifest_sha256": source.source_sha256,
        })
    if selected:
        raise ValueError(f"{len(selected)} indexed frames were missing from the input caches")
    if "imported_direct_train" in settings:
        if vertex_match is not None or not use_tcp:
            raise ValueError("Imported compact targets use observed coordinates and the TCP reference")
        imported, receipt = read_imported_train(settings["imported_direct_train"])
        samples["train"].extend(imported)
        identity["imported_direct_train"] = receipt
    identity["frames"] = {split: len(rows) for split, rows in samples.items()}
    identity["dataset_sha256"] = canonical_hash(identity)
    print(f"Loaded {settings['condition']} frames: {identity['frames']}", flush=True)

    train_collate = training_collate(collate, identity, std_mm)
    validation = fixed_validation_rows(samples["validation"], identity, settings)
    validation_generator = torch.Generator().manual_seed(0) if validation_std_mm else None
    batch_size = config["training"]["batch_size"]
    return DataBundle(
        train=DataLoader(samples["train"], batch_size=batch_size, collate_fn=train_collate,
                         shuffle=True, generator=shuffle_generator),
        validation=DataLoader(validation, batch_size=batch_size, collate_fn=collate,
                              generator=validation_generator),
        identity=identity,
    )
