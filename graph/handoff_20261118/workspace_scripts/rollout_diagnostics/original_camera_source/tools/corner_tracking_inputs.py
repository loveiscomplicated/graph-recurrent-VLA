"""Validate inputs shared by the Graph corner-tracking profiling tools."""

import json
from pathlib import Path

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.corner_tracks import read_episode
from utils.device_set import set_device


def require_cuda_device(device):
    """Reject unsupported profiling devices before reading data or building models."""
    selected = set_device(device)
    if selected.type != 'cuda':
        raise ValueError(f'Corner-tracking GPU profiling requires a CUDA device; got {selected}')
    return selected


def load_training_episode(data_root):
    """Read the first complete training episode after path and checksum validation."""
    root = Path(data_root).expanduser().resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    row = next((row for row in manifest['episodes'] if row['split'] == 'train'), None)
    if row is None:
        raise ValueError(f'Prepared corner dataset contains no training episode: {root}')
    path = (root / row['file']).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Profile episode path must stay inside the prepared dataset: {row["file"]}')
    if file_sha256(path) != row['sha256']:
        raise ValueError(f'Profile episode checksum mismatch: {row["file"]}')
    return read_episode(path, row), manifest
