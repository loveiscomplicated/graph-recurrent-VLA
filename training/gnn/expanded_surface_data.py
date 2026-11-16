"""Load the original and added detector-corner plus surface graphs."""

import torch
from torch.utils.data import DataLoader

from data_collection.episode_dataset import canonical_hash
from main import load_config
from training.gnn.corner_jitter import corner_prefix_counts, jitter_std_mm, training_collate
from training.gnn.data import DataBundle
from training.gnn.imported_direct_data import read_compact_splits, read_imported_train
from training.gnn.visibility_data import collate
from training.gnn.visibility_data import build_data as build_direct_data


def read_corner_counts(settings, fused_rows):
    """Recover the detector prefix using the matching direct-only training recipe."""
    reference = load_config(settings["corner_reference_config"])
    data = reference["data"]
    sources = data.get("sources", [])
    if (not sources or data.get("vertex_match") is not None or data.get("corner_jitter_std_mm", 0)
            or any("detection_manifest" not in source or "cache" in source for source in sources)):
        raise ValueError("corner_reference_config must use clean direct detector inputs")
    direct = build_direct_data(reference, torch.Generator())
    return corner_prefix_counts(fused_rows, direct.train.dataset)


def build_data(config, shuffle_generator):
    """Append the new train graphs while using only the original validation graphs."""
    settings = config["data"]
    original, original_identity = read_compact_splits(settings["original_surface"], ("train", "validation"))
    imported, imported_identity = read_imported_train(settings["imported_surface"])
    original["train"].extend(imported)
    frames = {split: len(rows) for split, rows in original.items()}
    if frames != settings["expected_frames"]:
        raise ValueError(f"Expected frame counts {settings['expected_frames']}, got {frames}")
    identity = {
        "original_surface": original_identity,
        "imported_surface": imported_identity,
        "frames": frames,
    }
    identity["dataset_sha256"] = canonical_hash(identity)
    std_mm = jitter_std_mm(settings)
    corner_counts = read_corner_counts(settings, original["train"]) if std_mm else None
    train_collate = training_collate(collate, identity, std_mm, corner_counts=corner_counts)
    print(f"Loaded detector + surface graphs: {frames}", flush=True)
    batch_size = config["training"]["batch_size"]
    return DataBundle(
        train=DataLoader(original["train"], batch_size=batch_size, collate_fn=train_collate,
                         shuffle=True, generator=shuffle_generator),
        validation=DataLoader(original["validation"], batch_size=batch_size, collate_fn=collate),
        identity=identity,
    )
