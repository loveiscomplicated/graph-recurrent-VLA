"""Load immutable graph coordinates with experiment-specific semantic views."""

from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_collection.episode_dataset import file_sha256
from data_collection.vertex_family_manifest import read_manifest
from graph.build_graph import NodeType, filter_nodes
from graph.vertex_family import FAMILY_MODES, with_family_tokens
from training.gnn.data import (
    DataBundle, TrainingBatch, canonical_hash, collate_graph_samples, load_cached_sample,
)


class VertexFamilyDataset(Dataset):
    """Eagerly verify graph/label hashes, then retain small graph samples in RAM.

    RGB-D and GT physical geometry are not loaded into the Student data path.
    Oracle semantic mode and the analysis-only oracle_rim filter select GT-family labels.
    """

    def __init__(self, root, manifest, split, mode, target_filter="none"):
        if mode not in FAMILY_MODES:
            raise ValueError(f"Unknown family mode: {mode}")
        if target_filter not in ("none", "detector_rim", "oracle_rim"):
            raise ValueError("target_filter must be none, detector_rim, or oracle_rim")
        if target_filter != "none" and mode != "baseline":
            raise ValueError("target_filter is only supported for the M0 baseline")
        self.samples = []
        self.rows = [row for row in manifest["samples"] if row["split"] == split]
        self.mode = mode
        root = Path(root).resolve()
        for row in self.rows:
            graph_path = (root / row["graph_path"]).resolve()
            label_path = (root / row["labels_path"]).resolve()
            if not graph_path.is_relative_to(root) or not label_path.is_relative_to(root):
                raise ValueError("Development graph/label paths must stay inside their dataset root")
            for path, expected in ((graph_path, row["graph_sha256"]), (label_path, row["labels_sha256"])):
                if file_sha256(path) != expected:
                    raise ValueError(f"Development graph/label hash mismatch: {path}")
            sample = load_cached_sample(graph_path, row["sample_identity"], row["sample_id"])
            with np.load(label_path, allow_pickle=False) as archive:
                use_oracle = mode == "oracle" or target_filter == "oracle_rim"
                key = "oracle_family" if use_oracle else "detector_family"
                labels = archive[key].copy()
                xyz = archive["xyz_world"]
                count = int(sample.graph.node_valid.sum())
                np.testing.assert_array_equal(xyz, sample.graph.xyz_world[0, :count].numpy())
                if len(labels) != count:
                    raise ValueError("Family label count must equal the original observed node count")
            padded_labels = np.full(sample.graph.node_valid.shape[1], -1, dtype=np.int64)
            padded_labels[:len(labels)] = labels
            graph = with_family_tokens(sample.graph, padded_labels, mode)
            if target_filter != "none":
                family = torch.as_tensor(padded_labels, device=graph.xyz_world.device)[None]
                keep = (graph.object_type != NodeType.BOX) | (family == 1)
                graph = filter_nodes(graph, keep)
            graph.validate(num_semantic_ids=16)
            self.samples.append(TrainingBatch(graph, sample.targets, sample.sample_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def build_data(config, generator):
    """Load fixed parent splits with the configured family view and node filter."""
    root = config["data"]["family_dataset"]
    mode = config["data"]["family_mode"]
    manifest = read_manifest(root)
    selection_path = config["data"].get("validation_selection")
    selection_sha256 = None
    if selection_path:
        selection_path = Path(selection_path)
        labels = [json.loads(line) for line in selection_path.read_text().splitlines() if line.strip()]
        by_id = {row["sample_id"]: row for row in labels}
        validation_ids = {row["sample_id"] for row in manifest["samples"] if row["split"] == "validation"}
        if len(by_id) != len(labels) or set(by_id) != validation_ids:
            raise ValueError("Validation selection must label every validation sample exactly once")
        if any(row["both_pass"] != (row["peg_pass"] and row["hole_pass"]) for row in labels):
            raise ValueError("Validation selection has inconsistent physical condition labels")
        selected_ids = {sample_id for sample_id, row in by_id.items() if row["both_pass"]}
        if not selected_ids:
            raise ValueError("Validation selection contains no frames satisfying both conditions")
        manifest = dict(manifest, samples=[row for row in manifest["samples"]
                        if row["split"] != "validation" or row["sample_id"] in selected_ids])
        selection_sha256 = file_sha256(selection_path)
    target_filter = config["data"].get("target_filter", "none")
    datasets = {split: VertexFamilyDataset(root, manifest, split, mode, target_filter)
                for split in ("train", "validation")}
    common = dict(batch_size=config["training"]["batch_size"], num_workers=0,
                  collate_fn=partial(collate_graph_samples, empty_inputs="keep"))
    identity_parts = [manifest["dataset_sha256"], mode]
    if target_filter != "none":
        identity_parts.append({"target_filter": target_filter, "version": 1})
    original_dataset_sha256 = canonical_hash(identity_parts)
    if selection_sha256:
        identity_parts.append({"validation_selection_sha256": selection_sha256, "rule": "both_pass"})
    identity = dict(dataset_sha256=canonical_hash(identity_parts),
                    core_dataset_sha256=manifest["dataset_sha256"], family_mode=mode,
                    target_filter=target_filter,
                    source_dataset_sha256=manifest["source_dataset_sha256"],
                    manifest_sha256=file_sha256(Path(root) / "manifest.json"), splits=manifest["splits"],
                    sample_counts={split: len(dataset) for split, dataset in datasets.items()})
    if selection_sha256:
        identity.update(resume_from_dataset_sha256=original_dataset_sha256,
                        validation_selection_sha256=selection_sha256)
    return DataBundle(DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
                      DataLoader(datasets["validation"], shuffle=False, **common), identity)
