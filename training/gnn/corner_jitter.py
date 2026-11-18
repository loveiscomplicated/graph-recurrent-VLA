"""Apply Gaussian coordinate noise to detector corners during training."""

from functools import partial
import hashlib
from math import isfinite
from os.path import relpath
from pathlib import Path

import torch

from data_collection.episode_dataset import canonical_hash


def jitter_std_mm(settings):
    """Read a nonnegative per-axis standard deviation in millimetres."""
    return read_std_mm(settings, "corner_jitter_std_mm")


def read_std_mm(settings, key):
    """Validate a configured noise magnitude."""
    value = settings.get(key, 0.0)
    if type(value) not in (float, int) or not isfinite(value) or value < 0:
        raise ValueError(f"data.{key} must be a finite nonnegative number")
    return float(value)


def fixed_validation_rows(rows, identity, settings, *, corner_counts=None):
    """Freeze one draw per validation frame, shared across graph variants."""
    std_mm = read_std_mm(settings, "validation_corner_jitter_std_mm")
    if not std_mm:
        return rows
    seed = settings.get("validation_corner_jitter_seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("data.validation_corner_jitter_seed must be an integer in [0, 2**64)")
    updated = []
    for xyz, kinds, tcp, target, sample_id in rows:
        path = relpath(Path(sample_id.removeprefix("original::")).resolve())
        frame_digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(frame_digest[:8], "big"))
        length = len(xyz) if corner_counts is None else corner_counts[sample_id]
        jittered = xyz.clone()
        offset = torch.randn((length, 3), generator=generator, dtype=xyz.dtype)
        jittered[:length] += offset * (std_mm / 1000)
        updated.append((jittered, kinds, tcp, target, sample_id))
    prior_identity = identity.pop("dataset_sha256")
    identity["resume_from_dataset_sha256"] = prior_identity
    identity["validation_corner_jitter"] = {"std_mm_per_axis": std_mm, "seed": seed}
    identity["dataset_sha256"] = canonical_hash(identity)
    return updated


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
