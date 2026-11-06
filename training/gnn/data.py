"""Read raw episodes and build observed graphs on demand, with optional on-demand caching."""

from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_collection.episode_dataset import file_sha256, read_episodes, read_frame
from graph.build_graph import ConfidenceSettings, GraphBatch, GraphBuilder, GraphSettings
from graph.utils.object21_targets import object21_from_gt
from utils.device_set import set_device


@dataclass
class TrainingBatch:
    """Keep supervision separate from the GT-free model inputs."""

    graph: GraphBatch
    targets: dict[str, torch.Tensor]
    sample_ids: list[str]

    def to(self, device):
        """Transfer a complete batch while keeping sample identifiers on the host."""
        return TrainingBatch(
            self.graph.to(device),
            {name: value.to(device) for name, value in self.targets.items()},
            self.sample_ids,
        )


@dataclass
class DataBundle:
    """Data factories supply loaders and a stable dataset/preprocessing identity."""

    train: DataLoader
    validation: DataLoader
    identity: dict


def canonical_hash(value) -> str:
    """Hash JSON settings without depending on dictionary insertion order."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def split_episodes(episodes, settings, seed):
    """Assign episodes before frame selection; optional split files reserve test data."""
    names = [episode.identity for episode in episodes]
    split_file = settings.get("split_file")
    if split_file:
        splits = json.loads(Path(split_file).read_text())
        if set(splits) != {"train", "validation", "test"}:
            raise ValueError("split_file must contain train, validation and test lists")
        flattened = []
        for name, values in splits.items():
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"split_file {name} must be a list of episode IDs")
            flattened.extend(values)
        if len(flattened) != len(set(flattened)) or set(flattened) != set(names):
            raise ValueError("split_file must assign every episode exactly once")
    else:
        random.Random(seed).shuffle(names)
        count = len(names)
        val_count = max(1, int(count * settings["validation_fraction"]))
        test_fraction = settings["test_fraction"]
        test_count = max(1, int(count * test_fraction)) if test_fraction else 0
        if val_count + test_count >= count:
            raise ValueError("Dataset needs disjoint nonempty train/validation episodes and requested test split")
        splits = {"validation": names[:val_count],
                  "test": names[val_count:val_count + test_count],
                  "train": names[val_count + test_count:]}
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Both train and validation splits must contain episodes")
    return splits


def preprocessing_identity(config):
    """Fingerprint transform code/settings for reproducibility, without storing samples."""
    from visual.corner_detection import Parameters

    root = Path(__file__).resolve().parents[2]
    sources = []
    for directory in ("visual/corner_detection", "graph/build_graph", "graph/utils"):
        sources.extend(sorted((root / directory).rglob("*.py")))
    sources.extend((Path(__file__), root / "data_collection/episode_dataset.py"))
    versions = {"torch": str(torch.__version__), "numpy": np.__version__}
    for package in ("scipy", "opencv-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "format_version": 1,
        "sources": {str(path.relative_to(root)): file_sha256(path) for path in sources},
        "camera": config["data"]["camera"],
        "detector_parameters": asdict(Parameters(**config["detector"]["parameters"])),
        "detector_device": config["detector"]["device"],
        "resolved_device": str(set_device(config["detector"]["device"])),
        "versions": versions,
        "graph": config["graph"],
        "confidence": config["confidence"],
    }


class EpisodeGraphDataset(Dataset):
    """Run current-frame detection and graph construction for every access.

    Only the detector's reusable workspace is retained, never frame results.
    GT supplies TCP proprioception and the target, not corner identities/edges.
    """

    def __init__(self, samples, config):
        self.samples = samples
        self.camera = config["data"]["camera"]
        self.detector_settings = config["detector"]
        self.builder = GraphBuilder(
            GraphSettings(**config["graph"]), ConfidenceSettings(**config["confidence"]),
        )
        self._detector = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from visual.corner_detection import Parameters, TorchCornerDetector

        sample = self.samples[index]
        if self._detector is None:
            self._detector = TorchCornerDetector(
                self.detector_settings["device"],
                Parameters(**self.detector_settings["parameters"]),
            )
        observation, gt = read_frame(sample["path"], self.camera)
        result = self._detector(observation["rgb"], observation["depth"],
                                observation["intrinsic_cv"], observation["extrinsic_cv"])
        tcp = torch.as_tensor(gt["ee_pose"][:, :3], dtype=torch.float32)
        graph = self.builder([result], tcp)
        targets = {"object21": object21_from_gt(gt)}
        return TrainingBatch(graph, targets, [sample["id"]])


def collate_graph_samples(samples, empty_inputs="keep"):
    """Pad graphs and remap flattened edges without changing their topology."""
    if empty_inputs == "skip":
        samples = [sample for sample in samples if sample.graph.node_valid.any()]
    if not samples:
        return None
    node_count = max(sample.graph.xyz_world.shape[1] for sample in samples)
    node_fields = ("xyz_world", "object_type", "node_valid", "semantic_id",
                   "semantic_valid", "confidence", "confidence_valid")
    values = {}
    for name in node_fields:
        padded = []
        for sample in samples:
            value = getattr(sample.graph, name)
            shape = (1, node_count, *value.shape[2:])
            fill = -1 if name in ("object_type", "semantic_id") else 0
            target = torch.full(shape, fill, dtype=value.dtype)
            target[:, :value.shape[1]] = value
            padded.append(target)
        values[name] = torch.cat(padded)
    values["tcp_xyz_world"] = torch.cat([sample.graph.tcp_xyz_world for sample in samples])
    values["edge_index"] = torch.cat(
        [sample.graph.edge_index + index * node_count for index, sample in enumerate(samples)], dim=1,
    )
    for name in ("edge_type", "relative_xyz", "distance"):
        values[name] = torch.cat([getattr(sample.graph, name) for sample in samples])
    graph = GraphBatch(**values)
    graph.validate()
    targets = {name: torch.cat([sample.targets[name] for sample in samples])
               for name in samples[0].targets}
    return TrainingBatch(graph, targets, [sample.sample_ids[0] for sample in samples])


def build_data(config, shuffle_generator) -> DataBundle:
    """Index approved raw frames, record splits/hashes, and create raw or cached loaders."""
    settings = config["data"]
    root = Path(settings["root"]).expanduser().resolve()
    episodes = read_episodes(root, settings["manifest"])
    splits = split_episodes(episodes, settings, config["seed"])
    detector_device = set_device(config["detector"]["device"])
    if detector_device.type != "cpu" and settings["num_workers"] != 0:
        raise ValueError("GPU detector execution requires data.num_workers=0")
    split_for_episode = {episode: name for name, ids in splits.items() for episode in ids}
    samples = {name: [] for name in splits}
    frame_identity = []
    for episode in episodes:
        split = split_for_episode[episode.identity]
        if split == "test":
            continue
        selected = episode.frames[::settings["frame_stride"]]
        if settings["max_frames_per_episode"] is not None:
            selected = selected[:settings["max_frames_per_episode"]]
        for frame_path in selected:
            sample_id = f"{episode.identity}/{frame_path.name}"
            digest = file_sha256(frame_path)
            frame_identity.append({"id": sample_id, "sha256": digest})
            samples[split].append({"id": sample_id, "path": frame_path, "sha256": digest})
    for name in ("train", "validation"):
        if not samples[name]:
            raise ValueError(f"No usable {name} frames after frame selection")
    identity = {
        "manifest_sha256": file_sha256(root / settings["manifest"]),
        "splits": splits, "frames": frame_identity,
        "preprocessing": preprocessing_identity(config),
        "detector_resolved_device": str(detector_device),
        "frame_stride": settings["frame_stride"],
        "max_frames_per_episode": settings["max_frames_per_episode"],
        "empty_inputs": settings["empty_inputs"],
    }
    identity["dataset_sha256"] = canonical_hash(identity)
    identity["sample_counts"] = {name: len(rows) for name, rows in samples.items()}
    common = {"batch_size": config["training"]["batch_size"],
              "num_workers": settings["num_workers"],
              "collate_fn": partial(collate_graph_samples, empty_inputs=settings["empty_inputs"])}
    datasets = {}
    for name in ("train", "validation"):
        if settings["use_cache"]:
            datasets[name] = CachedEpisodeGraphDataset(samples[name], config, identity["preprocessing"])
        else:
            datasets[name] = EpisodeGraphDataset(samples[name], config)
    train = DataLoader(datasets["train"], shuffle=True, generator=shuffle_generator, **common)
    validation = DataLoader(datasets["validation"], shuffle=False, **common)
    return DataBundle(train, validation, identity)


def save_cached_sample(path, sample, identity):
    """Atomically persist a graph/target sample with an explicit source identity."""
    from dataclasses import fields
    from uuid import uuid4

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {field.name: getattr(sample.graph, field.name).cpu().numpy()
              for field in fields(GraphBatch)}
    arrays.update({f'target_{name}': value.cpu().numpy() for name, value in sample.targets.items()})
    arrays['identity'] = np.array(json.dumps(identity, sort_keys=True))
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cached_sample(path, identity, sample_id):
    """Reject stale or malformed entries rather than silently training on them."""
    from dataclasses import fields

    with np.load(path, allow_pickle=False) as arrays:
        if json.loads(str(arrays['identity'])) != identity:
            raise ValueError(f'Graph cache identity mismatch: {path}')
        graph = GraphBatch(**{field.name: torch.from_numpy(arrays[field.name].copy())
                              for field in fields(GraphBatch)})
        targets = {name.removeprefix('target_'): torch.from_numpy(arrays[name].copy())
                   for name in arrays.files if name.startswith('target_')}
    graph.validate()
    target = targets.get('object21')
    if target is None or target.shape != (1, 21) or not torch.isfinite(target).all():
        raise ValueError(f'Invalid Object21 cache target: {path}')
    return TrainingBatch(graph, targets, [sample_id])


class CachedEpisodeGraphDataset(EpisodeGraphDataset):
    """Lazily cache each frame on first access; do not create files at construction."""

    def __init__(self, samples, config, preprocessing):
        super().__init__(samples, config)
        self.cache_root = Path(config['data']['cache_dir']).expanduser().resolve()
        self.preprocessing_sha256 = canonical_hash(preprocessing)

    def __getitem__(self, index):
        source = self.samples[index]
        # Verify current bytes even on hits, so modified raw data cannot reuse a
        # result silently. The run identity is fixed during indexing as well.
        if file_sha256(source['path']) != source['sha256']:
            raise ValueError(f"Raw frame changed after indexing: {source['path']}")
        identity = {'frame_sha256': source['sha256'], 'sample_id': source['id'],
                    'preprocessing_sha256': self.preprocessing_sha256}
        path = self.cache_root / self.preprocessing_sha256 / (canonical_hash(identity) + '.npz')
        if path.is_file():
            return load_cached_sample(path, identity, source['id'])
        sample = super().__getitem__(index)
        save_cached_sample(path, sample, identity)
        return sample
