"""Warm-start checkpoint weights on a disjoint moving-hole dataset."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from training.surface_mamba_controller import CHECKPOINT_FORMAT, train


def _validate_warm_start_mixing(source_alphas, mixing):
    """Require an explicit opt-in when fine-tuning changes source mixtures."""
    fine_tuning_alphas = mixing["alphas"]
    mixture_changed = source_alphas != fine_tuning_alphas
    change_allowed = mixing.get("allow_warm_start_alpha_change", False)
    if mixture_changed and not change_allowed:
        raise ValueError(
            "Fine-tuning alpha changes require "
            "mixing.allow_warm_start_alpha_change=true"
        )
    return fine_tuning_alphas


def _evaluation_seeds(contract):
    """Resolve inline or hash-pinned file-backed reserved evaluation seeds."""
    if "evaluation_seeds" in contract:
        seeds = contract["evaluation_seeds"]
    else:
        path = Path(contract["evaluation_seeds_file"])
        expected_sha256 = contract["evaluation_seeds_sha256"]
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"Evaluation seed file digest changed: {path}")
        document = json.loads(path.read_text())
        seeds = document.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Reserved evaluation seeds must be distinct integers")
    return seeds


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
    _validate_warm_start_mixing(
        source["identity"]["alphas"], config["mixing"]
    )

    dataset = Path(config["dataset"])
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    validation = config["dataset_contract"]
    if manifest["motion"] != validation["motion"]:
        raise ValueError("Fine-tuning motion differs from the evaluation protocol")
    reserved = set(_evaluation_seeds(validation))
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
