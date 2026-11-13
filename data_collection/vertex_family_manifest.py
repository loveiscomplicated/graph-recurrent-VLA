"""Verify the vertex-family development manifest without loading training code."""

import json
from pathlib import Path

from data_collection.episode_dataset import canonical_hash


def read_manifest(root):
    """Reject altered manifests and any overlap in the development parent split."""
    path = Path(root) / "manifest.json"
    manifest = json.loads(path.read_text())
    expected = manifest.pop("dataset_sha256")
    if canonical_hash(manifest) != expected:
        raise ValueError("Development manifest content does not match dataset_sha256")
    manifest["dataset_sha256"] = expected
    if manifest["original_pool"] != "train_only" or manifest["reserved_test_frames_read"] != 0:
        raise ValueError("This study requires original TRAIN-only development data")
    sample_ids = [row["sample_id"] for row in manifest["samples"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Development sample IDs must be unique")
    parents = {split: {row["parent_id"] for row in manifest["samples"] if row["split"] == split}
               for split in ("train", "validation")}
    if not parents["train"] or not parents["validation"] or parents["train"] & parents["validation"]:
        raise ValueError("Development train/validation parents must be disjoint and nonempty")
    return manifest
