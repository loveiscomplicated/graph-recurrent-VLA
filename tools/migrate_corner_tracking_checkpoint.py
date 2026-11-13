"""Migrate an epoch-boundary checkpoint to the one-frame association recovery.

Run from the revised source tree with --previous-source, --checkpoint and
--output. The original checkpoint remains untouched and its digest is recorded.
"""

import argparse
from pathlib import Path

import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from training.gnn.temporal.corner_tracking import implementation_hashes


def migrate_checkpoint(previous_source, checkpoint_path, output_path):
    """Allow only the reviewed tracker change and retain complete provenance."""
    previous_source = previous_source.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output_path}")

    new_hashes = implementation_hashes()
    old_hashes = {name: file_sha256(previous_source / name) for name in new_hashes}
    changed = {name for name in new_hashes if old_hashes[name] != new_hashes[name]}
    expected = {"graph/temporal/corner_tracking/tracker.py"}
    if changed != expected:
        raise ValueError(f"Expected only the association tracker to differ; found {sorted(changed)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "corner_tracking_last":
        raise ValueError("Checkpoint must be an epoch-boundary corner_tracking_last checkpoint")
    old_implementation = canonical_hash(old_hashes)
    if checkpoint["config"]["implementation_sha256"] != old_implementation:
        raise ValueError("Checkpoint implementation fingerprint does not match previous source")
    if checkpoint["epoch"] < 1:
        raise ValueError("Checkpoint must contain a completed epoch")

    new_implementation = canonical_hash(new_hashes)
    checkpoint["config"]["implementation_sha256"] = new_implementation
    checkpoint["association_recovery_migration"] = dict(
        original_checkpoint=str(checkpoint_path),
        original_checkpoint_sha256=file_sha256(checkpoint_path),
        original_implementation_sha256=old_implementation,
        revised_implementation_sha256=new_implementation,
        changed_source_files=sorted(changed),
        reason="one-frame association fallback after loss of all certified supervision",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary checkpoint already exists: {temporary}")
    torch.save(checkpoint, temporary)
    temporary.replace(output_path)
    print(f"Migrated epoch {checkpoint['epoch']} checkpoint: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    migrate_checkpoint(options.previous_source, options.checkpoint, options.output)


if __name__ == "__main__":
    main()
