"""Load the original and added detector-corner plus surface graphs."""

import torch
from torch.utils.data import DataLoader

from data_collection.episode_dataset import canonical_hash
from main import load_config
from training.gnn.corner_jitter import (
    corner_prefix_counts, fixed_validation_rows, jitter_std_mm, read_std_mm, training_collate,
)
from training.gnn.data import DataBundle
from training.gnn.imported_direct_data import read_compact_splits, read_imported_train
from training.gnn.visibility_data import collate
from training.gnn.visibility_data import build_data as build_direct_data


def read_corner_counts(settings, fused_rows, splits):
    """Recover the detector prefix using the matching direct-only training recipe."""
    reference = load_config(settings["corner_reference_config"])
    data = reference["data"]
    sources = data.get("sources", [])
    if (not sources or data.get("vertex_match") is not None or data.get("corner_jitter_std_mm", 0)
            or any("detection_manifest" not in source or "cache" in source for source in sources)):
        raise ValueError("corner_reference_config must use clean direct detector inputs")
    direct = build_direct_data(reference, torch.Generator())
    return {split: corner_prefix_counts(fused_rows[split], getattr(direct, split).dataset)
            for split in splits}


def build_data(config, shuffle_generator):
    """Load the selected surface graphs and any configured train-only additions."""
    settings = config["data"]
    condition = settings.get("condition", "condition_1")
    original, original_identity = read_compact_splits(
        settings["original_surface"], ("train", "validation"), condition=condition
    )
    identity = {"original_surface": original_identity}
    if "imported_surface" in settings:
        imported, imported_identity = read_imported_train(settings["imported_surface"], condition=condition)
        original["train"].extend(imported)
        identity["imported_surface"] = imported_identity
    additional_train = settings.get("additional_train", {})
    if additional_train:
        seen = {row[-1] for rows in original.values() for row in rows}
        identity["additional_train"] = {}
        for train_condition, sources in additional_train.items():
            receipts = []
            for source in sources:
                cache_splits = tuple(source.get("cache_splits", ["train"]))
                cached, receipt = read_compact_splits(
                    source, cache_splits, condition=train_condition
                )
                rows = cached["train"]
                receipt["used_train_frames"] = len(rows)
                frame_ids = {row[-1] for row in rows}
                if len(frame_ids) != len(rows) or seen.intersection(frame_ids):
                    raise ValueError(f"Duplicate frame in additional {train_condition} training data")
                seen.update(frame_ids)
                original["train"].extend(rows)
                receipts.append(receipt)
            identity["additional_train"][train_condition] = receipts
    frames = {split: len(rows) for split, rows in original.items()}
    if frames != settings["expected_frames"]:
        raise ValueError(f"Expected frame counts {settings['expected_frames']}, got {frames}")
    identity["frames"] = frames
    if condition != "condition_1":
        identity["condition"] = condition
    identity["dataset_sha256"] = canonical_hash(identity)
    std_mm = jitter_std_mm(settings)
    validation_std_mm = read_std_mm(settings, "validation_corner_jitter_std_mm")
    jitter_splits = [split for split, value in (("train", std_mm), ("validation", validation_std_mm)) if value]
    corner_counts = read_corner_counts(settings, original, jitter_splits) if jitter_splits else {}
    train_collate = training_collate(collate, identity, std_mm, corner_counts=corner_counts.get("train"))
    validation = fixed_validation_rows(original["validation"], identity, settings,
                                       corner_counts=corner_counts.get("validation"))
    validation_generator = torch.Generator().manual_seed(0) if validation_std_mm else None
    print(f"Loaded detector + surface graphs: {frames}", flush=True)
    batch_size = config["training"]["batch_size"]
    return DataBundle(
        train=DataLoader(original["train"], batch_size=batch_size, collate_fn=train_collate,
                         shuffle=True, generator=shuffle_generator),
        validation=DataLoader(validation, batch_size=batch_size, collate_fn=collate,
                              generator=validation_generator),
        identity=identity,
    )
