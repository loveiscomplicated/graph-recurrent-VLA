"""Copy a corner-tracking checkpoint for the metrics-only rollout change."""

import argparse
from pathlib import Path

import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from training.gnn.temporal.corner_tracking import implementation_hashes


def migrate(previous_source, checkpoint_path, output_path):
    """Preserve model, optimizer and RNG state while recording source provenance."""
    previous_source = previous_source.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    new_hashes = implementation_hashes()
    old_hashes = {name: file_sha256(previous_source / name) for name in new_hashes}
    changed = {name for name in new_hashes if old_hashes[name] != new_hashes[name]}
    if changed != {"training/gnn/temporal/corner_tracking_rollout.py"}:
        raise ValueError(f"Expected only metrics rollout changes, found {sorted(changed)}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    old_hash = canonical_hash(old_hashes)
    if checkpoint.get("kind") != "corner_tracking_last" or checkpoint["config"]["implementation_sha256"] != old_hash:
        raise ValueError("Checkpoint is not an epoch-boundary checkpoint from the previous source")
    new_hash = canonical_hash(new_hashes)
    checkpoint["config"]["implementation_sha256"] = new_hash
    checkpoint["logging_migration"] = dict(original_checkpoint=str(checkpoint_path),
        original_checkpoint_sha256=file_sha256(checkpoint_path), original_implementation_sha256=old_hash,
        revised_implementation_sha256=new_hash, changed_source_files=sorted(changed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {temporary}")
    torch.save(checkpoint, temporary)
    temporary.replace(output_path)
    print(f"Migrated epoch {checkpoint['epoch']} to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    migrate(args.previous_source, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
