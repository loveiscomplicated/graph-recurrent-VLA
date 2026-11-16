"""Load compact observed-corner graphs for the imported training frames."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256


def read_compact_splits(settings, splits):
    """Verify each compact archive and enforce exact frame coverage by split."""
    index_path = Path(settings["visibility_index"])
    manifest_path = Path(settings["cache_manifest"])
    index = json.loads(index_path.read_text())["condition_1"]
    manifest = json.loads(manifest_path.read_text())
    index_hash = file_sha256(index_path)
    if manifest["visibility_index_sha256"] != index_hash:
        raise ValueError("Imported corner cache and visibility index differ")
    selected = {(row["archive"], row["member"]): split for split in splits for row in index[split]}
    if len(selected) != sum(len(index[split]) for split in splits):
        raise ValueError("Duplicate imported train frame in visibility index")

    rows = {split: [] for split in splits}
    for entry in manifest["archives"]:
        path = manifest_path.parent / entry["file"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Imported corner cache checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as cache:
            members = cache["frame_members"]
            xyz = torch.from_numpy(cache["corner_world_xyz"])
            kinds = torch.from_numpy(cache["corner_object_id"])
            offsets = cache["corner_offsets"]
            tcp = torch.from_numpy(cache["tcp"])
            target = torch.from_numpy(cache["object21"])
        count = len(members)
        if (
            count != entry["frames"] or offsets.shape != (count + 1,)
            or offsets[0] != 0 or offsets[-1] != len(kinds)
            or np.any(np.diff(offsets) < 0) or xyz.shape != (len(kinds), 3)
            or tcp.shape != (count, 3) or target.shape != (count, 21)
            or not np.isin(kinds.numpy(), (0, 1)).all()
            or not all(torch.isfinite(value).all() for value in (xyz, tcp, target))
        ):
            raise ValueError(f"Invalid compact corner arrays: {path}")
        for row, member in enumerate(members):
            key = entry["archive"], str(member)
            if key not in selected:
                raise ValueError(f"Unexpected or duplicate imported train frame: {key}")
            split = selected.pop(key)
            nodes = slice(offsets[row], offsets[row + 1])
            rows[split].append((xyz[nodes], kinds[nodes], tcp[row], target[row], "::".join(key)))
    if selected:
        raise ValueError(f"Missing {len(selected)} compact frames")
    return rows, {
        "visibility_index_sha256": index_hash,
        "manifest_sha256": file_sha256(manifest_path),
        "frames": sum(map(len, rows.values())),
    }


def read_imported_train(settings):
    """Load the imported training frames without changing the original splits."""
    rows, identity = read_compact_splits(settings, ("train",))
    return rows["train"], identity
