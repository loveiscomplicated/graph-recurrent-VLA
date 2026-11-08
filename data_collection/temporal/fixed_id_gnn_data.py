"""Prepare synchronized fixed-ID corner histories and Object21 labels."""

from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import (
    load_source_episode, read_manifest, read_source_manifest, sequences_from_arrays,
    validate_packed_episode,
)
from data_collection.vertex_family_manifest import read_manifest as read_family_manifest
from utils.experiment_logger import ExperimentLogger, write


def episode_arrays(packed, row, source):
    """Reassemble frame-wise data, including IDs first seen on the last frame."""
    length = len(packed["robot_state"])
    measured = np.zeros((length, 24, 3), dtype=np.float32)
    last = np.zeros_like(measured)
    observed = np.zeros((length, 24), dtype=bool)
    seen = np.zeros_like(observed)
    sequences = list(sequences_from_arrays(packed, row))
    for sequence in sequences:
        start, vertex_id = sequence.start_frame, sequence.vertex_id
        seen[start:, vertex_id] = True
        observed[start:, vertex_id] = sequence.observed
        measured[start:, vertex_id] = np.where(sequence.observed[:, None], sequence.targets, 0)
        last[start:, vertex_id] = sequence.last_observed_xyz_world

    # Terminal-only sightings are absent from the sequence dataset because they
    # have no prediction target. They still belong in the final frame's graph.
    known = source["valid"][-1] & (source["physical_ids"][-1] >= 0)
    for vertex_id in np.unique(source["physical_ids"][-1, known]):
        if seen[-1, vertex_id]:
            continue
        candidates = np.flatnonzero(known & (source["physical_ids"][-1] == vertex_id))
        errors = np.linalg.norm(source["xyz"][-1, candidates] - source["reference_xyz"][-1, vertex_id], axis=1)
        selected = candidates[int(np.argmin(errors))]
        measured[-1, vertex_id] = source["xyz"][-1, selected]
        observed[-1, vertex_id] = True
        seen[-1, vertex_id] = True
    return measured, last, observed, seen, sequences


def prepare(config):
    """Verify lineage and save frame labels plus causal per-corner inputs once."""
    data = config["data"]
    root = Path(data["corner_dataset"]).resolve()
    family_root = Path(data["family_dataset"]).resolve()
    manifest = read_manifest(root)
    family = read_family_manifest(family_root)
    source_root = Path(manifest["config"]["source"]["corner_dataset"]).resolve()
    source = read_source_manifest(source_root)
    if manifest["dataset_sha256"] != data["dataset_sha256"]:
        raise ValueError("Fixed-ID dataset hash mismatch")
    if (source["dataset_sha256"] != manifest["source_dataset_sha256"]
            or source["provenance"]["source_dataset_sha256"] != family["dataset_sha256"]):
        raise ValueError("Corner and 21D family dataset lineage mismatch")
    source_rows = {row["episode"]: row for row in source["episodes"]}
    family_rows = {row["sample_id"]: row for row in family["samples"]}
    episodes = [row for row in manifest["episodes"] if row["split"] in ("train", "validation")]
    parents = {split: {row["parent"] for row in episodes if row["split"] == split}
               for split in ("train", "validation")}
    if not parents["train"] or not parents["validation"] or parents["train"] & parents["validation"]:
        raise ValueError("Training and validation parents must be nonempty and disjoint")

    with ExperimentLogger(team="graph", model="fixed_id_gnn_data", config=config,
                          seed=0, job_type="data_preparation") as logger:
        output = Path(config["output"]["cache_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        data_parts = {key: [] for key in ("measurement", "last", "observed", "seen", "tcp", "target", "split")}
        sequences = []
        frame_offset = 0
        for episode_index, row in enumerate(episodes):
            path = (root / row["file"]).resolve()
            if not path.is_relative_to(root) or file_sha256(path) != row["sha256"]:
                raise ValueError(f"Corner dataset file mismatch: {row['episode']}")
            with np.load(path, allow_pickle=False) as archive:
                packed = {name: archive[name].copy() for name in archive.files}
            validate_packed_episode(packed)
            original = load_source_episode(source_root, source_rows[row["episode"]])
            measurement, last, observed, seen, corner_sequences = episode_arrays(packed, row, original)
            labels = []
            for frame, sample_id in enumerate(packed["sample_ids"]):
                record = family_rows[str(sample_id)]
                if record["split"] != row["split"] or record["parent_id"] != row["parent"]:
                    raise ValueError(f"Frame split or parent mismatch: {sample_id}")
                graph_path = (family_root / record["graph_path"]).resolve()
                if not graph_path.is_relative_to(family_root) or file_sha256(graph_path) != record["graph_sha256"]:
                    raise ValueError(f"Object21 graph cache hash mismatch: {sample_id}")
                with np.load(graph_path, allow_pickle=False) as archive:
                    target = archive["target_object21"].copy()
                    tcp = archive["tcp_xyz_world"].copy()
                if (target.shape != (1, 21) or not np.isfinite(target).all()
                        or not np.allclose(tcp[0], packed["robot_state"][frame, :3], atol=1e-6, rtol=0)):
                    raise ValueError(f"Object21/TCP label mismatch: {sample_id}")
                labels.append(target[0])
            length = len(labels)
            for key, value in (("measurement", measurement), ("last", last), ("observed", observed),
                               ("seen", seen), ("tcp", packed["robot_state"][:, :3]),
                               ("target", np.asarray(labels, dtype=np.float32)),
                               ("split", np.full(length, row["split"] == "validation", dtype=bool))):
                data_parts[key].append(value)
            for sequence in corner_sequences:
                sequences.append(dict(inputs=sequence.inputs, frame_offset=frame_offset + sequence.start_frame,
                                      vertex_id=sequence.vertex_id))
            frame_offset += length
            if episode_index == 0 or (episode_index + 1) % 25 == 0 or episode_index + 1 == len(episodes):
                print(f"Prepared {episode_index + 1}/{len(episodes)} episodes; frames={frame_offset}", flush=True)
        arrays = {key: np.concatenate(value) for key, value in data_parts.items()}
        cache = dict(arrays=arrays, sequences=sequences, dataset_sha256=manifest["dataset_sha256"],
                     family_dataset_sha256=family["dataset_sha256"], episodes=len(episodes))
        cache_path = output / "frames.pt"
        torch.save(cache, cache_path)
        write(output / "manifest.json", dict(dataset_sha256=manifest["dataset_sha256"],
              family_dataset_sha256=family["dataset_sha256"], cache_sha256=file_sha256(cache_path),
              frames=frame_offset, sequences=len(sequences), episodes=len(episodes),
              train_frames=int((~arrays["split"]).sum()), validation_frames=int(arrays["split"].sum())))
        logger.link_output(cache_path)
        logger.link_output(output / "manifest.json")
        logger.record(dataset_sha256=manifest["dataset_sha256"], family_dataset_sha256=family["dataset_sha256"])
        print(f"Prepared cache: {cache_path}", flush=True)
        return str(cache_path)
