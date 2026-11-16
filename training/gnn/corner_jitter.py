"""Apply Gaussian coordinate noise to detector corners during training."""

from functools import partial
from math import isfinite
from pathlib import Path

import torch

from data_collection.episode_dataset import canonical_hash


def jitter_std_mm(settings):
    """Read a nonnegative per-axis standard deviation in millimetres."""
    value = settings.get("corner_jitter_std_mm", 0.0)
    if type(value) not in (float, int) or not isfinite(value) or value < 0:
        raise ValueError("data.corner_jitter_std_mm must be a finite nonnegative number")
    return float(value)


def training_collate(collate, identity, std_mm, *, corner_counts=None):
    """Configure train-only jitter and preserve the clean dataset for warm starts."""
    if not std_mm:
        return collate
    clean_hash = identity.pop("dataset_sha256")
    identity["resume_from_dataset_sha256"] = clean_hash
    identity["corner_jitter"] = {"std_mm_per_axis": std_mm, "distribution": "gaussian"}
    identity["dataset_sha256"] = canonical_hash(identity)
    return partial(collate, jitter_std_m=std_mm / 1000, corner_counts=corner_counts)


def jitter_positions(xyz, valid, paths, std_m, corner_counts):
    """Perturb valid corner slots, leaving surface points and padding intact."""
    if not std_m:
        return xyz
    corners = valid
    if corner_counts is not None:
        lengths = torch.tensor([corner_counts[path] for path in paths], device=xyz.device)
        corners = valid & (torch.arange(xyz.shape[1], device=xyz.device) < lengths[:, None])
    updated = xyz.clone()
    selected = updated[corners]
    updated[corners] = selected + torch.randn_like(selected) * std_m
    return updated


def corner_prefix_counts(fused_rows, direct_rows):
    """Recover corner masks from aligned direct detector rows and verify the prefix."""
    direct = {row[-1]: row for row in direct_rows}
    if len(direct) != len(direct_rows):
        raise ValueError("Duplicate direct detector sample IDs")
    counts = {}
    for xyz, kinds, tcp, target, sample_id in fused_rows:
        # Original compact rows have IDs such as original::outputs/pilot/frame_000000.npz.
        key = sample_id
        if sample_id.startswith("original::"):
            key = str(Path(sample_id.removeprefix("original::")).resolve())
        if key not in direct:
            raise ValueError(f"No direct detector reference for {sample_id}")
        direct_xyz, direct_kinds, direct_tcp, direct_target, _ = direct.pop(key)
        count = len(direct_xyz)
        if (len(xyz) < count or not torch.equal(xyz[:count], direct_xyz)
                or not torch.equal(kinds[:count], direct_kinds)
                or not torch.equal(tcp, direct_tcp)
                or not torch.equal(target, direct_target)):
            raise ValueError(f"Detector prefix or target differs for {sample_id}")
        counts[sample_id] = count
    if direct:
        raise ValueError(f"Found {len(direct)} unmatched direct detector references")
    return counts
