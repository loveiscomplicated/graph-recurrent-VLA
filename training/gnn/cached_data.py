"""Read published graph caches without raw frames or a detector runtime."""

from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path, PurePosixPath
import re

from torch.utils.data import DataLoader, Dataset

from data_collection.episode_dataset import file_sha256
from .data import DataBundle, canonical_hash, collate_graph_samples, load_cached_sample


def require_sha256(value, name):
    """Reject malformed identifiers before using them in cache paths."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class CachedSample:
    path: Path
    sample_id: str
    identity: dict
    sha256: str
    size: int


class PublishedGraphDataset(Dataset):
    """Check stored bytes and schema on every read; never regenerate missing data."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if sample.path.stat().st_size != sample.size or file_sha256(sample.path) != sample.sha256:
            raise ValueError(f"Cache byte checksum mismatch: {sample.sample_id}")
        return load_cached_sample(sample.path, sample.identity, sample.sample_id)


def validate_splits(splits):
    """Use published assignments unchanged and reject overlap or invalid IDs."""
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Cache splits must contain train, validation and test lists")
    memberships = {}
    for name, episodes in splits.items():
        if not isinstance(episodes, list):
            raise ValueError(f"Cache split {name} must be a list")
        for episode in episodes:
            if not isinstance(episode, str) or not episode:
                raise ValueError("Cache episode IDs must be nonempty strings")
            if episode in memberships:
                raise ValueError(f"Duplicate or overlapping cache episode: {episode}")
            memberships[episode] = name
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Cache needs nonempty train and validation episode splits")
    return memberships


def index_samples(root, metadata, inventory):
    """Match every selected frame to its published cache checksum and split."""
    preprocessing = metadata["preprocessing"]
    if preprocessing.get("format_version") != 1:
        raise ValueError("Unsupported graph cache format_version")
    namespace = canonical_hash(preprocessing)
    if inventory.get("preprocessing_sha256") != namespace:
        raise ValueError("Cache inventory preprocessing hash differs from metadata")
    memberships = validate_splits(metadata["splits"])
    entries = {}
    for entry in inventory["files"]:
        sample_id = entry["sample_id"]
        if sample_id in entries:
            raise ValueError(f"Duplicate cache inventory sample: {sample_id}")
        entries[sample_id] = entry
    samples = {name: [] for name in ("train", "validation")}
    seen = set()
    for frame in metadata["frames"]:
        sample_id = frame["id"]
        if not isinstance(sample_id, str):
            raise ValueError("Cached sample ID must be a relative frame path")
        relative_id = PurePosixPath(sample_id)
        if relative_id.is_absolute() or ".." in relative_id.parts or len(relative_id.parts) < 2:
            raise ValueError(f"Invalid cached sample ID: {sample_id}")
        if sample_id in seen:
            raise ValueError(f"Duplicate cached frame: {sample_id}")
        seen.add(sample_id)
        episode_id = str(relative_id.parent)
        split = memberships.get(episode_id)
        if split not in samples:
            raise ValueError(f"Cached frame is unassigned or in reserved test split: {sample_id}")
        expected = {
            "frame_sha256": require_sha256(frame["sha256"], "frame.sha256"),
            "sample_id": sample_id,
            "preprocessing_sha256": namespace,
        }
        relative_path = f"{namespace}/{canonical_hash(expected)}.npz"
        entry = entries.get(sample_id)
        if entry is None or entry.get("file") != relative_path:
            raise ValueError(f"Missing or inconsistent cache inventory entry: {sample_id}")
        digest = require_sha256(entry.get("sha256"), "inventory.sha256")
        size = entry.get("size")
        if type(size) is not int or size <= 0:
            raise ValueError(f"Invalid cached file size: {sample_id}")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached frame; re-extract the archive: {path}")
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Cached file must stay within cache_dir: {path}")
        if path.stat().st_size != size:
            raise ValueError(f"Cached file size mismatch: {sample_id}")
        samples[split].append(CachedSample(path, sample_id, expected, digest, size))
    if seen != set(entries):
        raise ValueError("Cache inventory and frame metadata must list identical samples")
    for name, rows in samples.items():
        if not rows or metadata["sample_counts"].get(name) != len(rows):
            raise ValueError(f"Cached {name} frame count differs from metadata or is empty")
    return samples


def build_data(config, shuffle_generator) -> DataBundle:
    """Load the published split/index, bypassing raw data and preprocessing entirely.

    This data factory is selected as training.gnn.cached_data:build_data.
    The existing graph schema is validated when NPZ entries are read. Original
    detector code/library versions remain provenance, not execution requirements.
    """
    settings = config["data"]
    if not settings["use_cache"]:
        raise ValueError("Cache-only data factory requires data.use_cache=true")
    if settings["frame_stride"] != 1 or settings["max_frames_per_episode"] is not None:
        raise ValueError("Cache-only mode uses all published frames; frame selection overrides are unsupported")
    if settings["split_file"] is not None:
        raise ValueError("Cache-only mode uses the published splits; data.split_file must be null")
    root = Path(settings["cache_dir"]).expanduser().resolve()
    metadata_path = root / "metadata/data_identity.json"
    inventory_path = root / "metadata/cache_inventory.json"
    metadata = json.loads(metadata_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    original_identity = {name: value for name, value in metadata.items()
                         if name not in ("dataset_sha256", "sample_counts")}
    if canonical_hash(original_identity) != metadata.get("dataset_sha256"):
        raise ValueError("Published cache data_identity checksum mismatch")
    if metadata["preprocessing"].get("camera") != settings["camera"]:
        raise ValueError("data.camera differs from the published cache camera")
    for name in ("graph", "confidence"):
        if metadata["preprocessing"].get(name) != config[name]:
            raise ValueError(f"{name} settings differ from the published cache; replay cannot transform cached inputs")
    samples = index_samples(root, metadata, inventory)
    # Record stable byte inventory identity for resume without absolute paths.
    selection = {
        "source_dataset_sha256": metadata["dataset_sha256"],
        "inventory_sha256": file_sha256(inventory_path),
        "empty_inputs": settings["empty_inputs"],
    }
    identity = dict(metadata, source_dataset_sha256=metadata["dataset_sha256"],
                    dataset_sha256=canonical_hash(selection),
                    cache_inventory_sha256=selection["inventory_sha256"], cache_only=True,
                    cache_preprocessing_sha256=canonical_hash(metadata["preprocessing"]))
    common = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": settings["num_workers"],
        "collate_fn": partial(collate_graph_samples, empty_inputs=settings["empty_inputs"]),
    }
    training = DataLoader(PublishedGraphDataset(samples["train"]), shuffle=True,
                          generator=shuffle_generator, **common)
    validation = DataLoader(PublishedGraphDataset(samples["validation"]), shuffle=False, **common)
    return DataBundle(training, validation, identity)
