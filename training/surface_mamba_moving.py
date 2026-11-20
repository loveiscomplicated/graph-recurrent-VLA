"""Warm-start checkpoint weights on a disjoint moving-hole dataset."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from training.surface_mamba_controller import CHECKPOINT_FORMAT, train


def fine_tune(config):
    """Retain source normalization and start a fresh optimizer on moving data."""
    seed, = config["seeds"]
    path = Path(config["initial_checkpoint"])
    digest = file_sha256(path)
    if digest != config["initial_checkpoint_sha256"]:
        raise ValueError(f"Unexpected source-checkpoint digest: {path}")
    source = torch.load(path, map_location="cpu", weights_only=True)
    if source.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Expected a v2 controller checkpoint: {path}")
    if source["identity"]["alphas"] != config["mixing"]["alphas"]:
        raise ValueError("Fine-tuning must retain the source GT/Graph mixture")

    dataset = Path(config["dataset"])
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    validation = config["dataset_contract"]
    if manifest["motion"] != validation["motion"]:
        raise ValueError("Fine-tuning motion differs from the evaluation protocol")
    reserved = set(validation["evaluation_seeds"])
    for split in ("train", "validation"):
        path = dataset / f"{split}.npz"
        if file_sha256(path) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Moving data digest changed: {path}")
        with np.load(path, allow_pickle=False) as data:
            seeds = data["reset_seed"].tolist()
        if len(set(seeds)) != len(seeds) or reserved.intersection(seeds):
            raise ValueError(f"Duplicate or evaluation seed in {split}")

    train(
        config,
        dataset_format="surface_mamba_moving_state36_v1",
        expected_counts=tuple(validation["counts"]),
        warm_starts={seed: {"payload": source, "sha256": digest}},
    )
