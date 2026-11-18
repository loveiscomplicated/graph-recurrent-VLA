"""Read the complete Five-mode training split as time-ordered graph episodes."""

from contextlib import ExitStack
from dataclasses import dataclass, fields
from functools import partial
import json
from pathlib import Path
import tarfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_collection.episode_dataset import canonical_hash, file_sha256
from graph.build_graph import NodeType
from graph.state_estimation.graph import (
    Frame,
    GraphBatch,
    Node,
    Source,
    batch_graphs,
    build_graph,
)
from training.gnn.packed_five_mode_data import pack_episode, read_archive_member


@dataclass
class FiveModeGraphEpisode:
    """One complete episode whose first dimension is chronological time."""

    episode_id: str
    xyz: np.ndarray
    object_type: np.ndarray
    is_vertex: np.ndarray
    tcp: np.ndarray
    object21: np.ndarray

    @property
    def length(self):
        """Return the number of consecutive frames in this episode."""
        return len(self.xyz)


@dataclass
class FiveModeGraphSequenceBatch:
    """Padded graph episodes with a mask identifying real frames.

    ``graph`` contains ``B * T`` graphs in episode-major, time-minor order. A
    downstream frame encoder can reshape its output with ``reshape_frames``.
    Padded frames contain no valid graph nodes and have frame index ``-1``.
    """

    graph: GraphBatch
    targets: dict[str, torch.Tensor]
    frame_mask: torch.Tensor
    frame_indices: torch.Tensor
    lengths: torch.Tensor
    episode_ids: list[str]

    def to(self, device):
        """Move tensor fields while retaining episode identifiers on the host."""
        return FiveModeGraphSequenceBatch(
            graph=self.graph.to(device),
            targets={name: value.to(device) for name, value in self.targets.items()},
            frame_mask=self.frame_mask.to(device),
            frame_indices=self.frame_indices.to(device),
            lengths=self.lengths.to(device),
            episode_ids=self.episode_ids,
        )

    def pin_memory(self):
        """Pin every CPU tensor when DataLoader memory pinning is enabled."""
        pinned_graph = GraphBatch(
            **{
                field.name: (
                    getattr(self.graph, field.name).pin_memory()
                    if isinstance(getattr(self.graph, field.name), torch.Tensor)
                    else getattr(self.graph, field.name)
                )
                for field in fields(GraphBatch)
            }
        )
        return FiveModeGraphSequenceBatch(
            graph=pinned_graph,
            targets={name: value.pin_memory() for name, value in self.targets.items()},
            frame_mask=self.frame_mask.pin_memory(),
            frame_indices=self.frame_indices.pin_memory(),
            lengths=self.lengths.pin_memory(),
            episode_ids=self.episode_ids,
        )

    def reshape_frames(self, values):
        """Restore a frame encoder output from ``[B*T,...]`` to ``[B,T,...]``."""
        batch_size, time_steps = self.frame_mask.shape
        if values.shape[0] != batch_size * time_steps:
            raise ValueError(
                f"Expected {batch_size * time_steps} flattened frames, got {values.shape[0]}"
            )
        return values.reshape(batch_size, time_steps, *values.shape[1:])


class FiveModeTrainSequenceDataset(Dataset):
    """Decode complete train episodes from checksum-pinned release shards."""

    def __init__(self, manifest_path, *, include_surface):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text())
        records = manifest.get("records", [])
        identities = [(row.get("seed"), row.get("policy_mode")) for row in records]
        if (
            manifest.get("format") != "five_mode_supervised_v1"
            or len(records) != manifest.get("episodes")
            or len(identities) != len(set(identities))
        ):
            raise ValueError(
                f"Expected a unique Five-mode supervised manifest: {self.manifest_path}"
            )

        self.archive_names = {row["file"] for row in manifest.get("archives", [])}
        if len(self.archive_names) != len(manifest.get("archives", [])):
            raise ValueError("Five-mode manifest contains duplicate archive names")

        self.episodes = [row for row in records if row.get("split") == "train"]
        expected_episodes = manifest.get("episodes_by_split", {}).get("train")
        if not self.episodes or (
            expected_episodes is not None and len(self.episodes) != expected_episodes
        ):
            raise ValueError(
                f"Five-mode train inventory differs: found {len(self.episodes)} episodes"
            )
        if any(type(row.get("frames")) is not int or row["frames"] < 1 for row in self.episodes):
            raise ValueError("Every Five-mode train episode must contain at least one frame")

        self.include_surface = include_surface
        self.frame_count = sum(row["frames"] for row in self.episodes)
        self._stack = None
        self._archives = {}

    def __len__(self):
        return len(self.episodes)

    def close(self):
        """Close shard handles owned by the current process."""
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self._archives.clear()

    def __getstate__(self):
        """Start each DataLoader worker with independent TAR handles."""
        state = dict(self.__dict__)
        state["_stack"] = None
        state["_archives"] = {}
        return state

    def _archive(self, name):
        if name not in self.archive_names:
            raise ValueError(f"Unlisted Five-mode archive: {name}")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Invalid Five-mode archive path: {name}")
        if self._stack is None:
            self._stack = ExitStack()
        if name not in self._archives:
            self._archives[name] = self._stack.enter_context(tarfile.open(path, "r:"))
        return self._archives[name]

    def __getitem__(self, index):
        episode = self.episodes[index]
        kinds = ("corners", "targets", "surface") if self.include_surface else (
            "corners",
            "targets",
        )
        tables = {}
        for kind in kinds:
            receipt = episode["files"][kind]
            tables[kind] = read_archive_member(
                self._archive(receipt["archive"]), receipt, kind
            )
        xyz, object_type, tcp, object21 = pack_episode(
            episode, tables, self.include_surface
        )
        is_vertex = np.zeros(object_type.shape, dtype=bool)
        corner_counts = np.diff(tables["corners"]["corner_offsets"])
        for frame_index, corner_count in enumerate(corner_counts):
            is_vertex[frame_index, :corner_count] = True
        episode_id = f"{episode['seed']}:{episode['policy_mode']}"
        return FiveModeGraphEpisode(
            episode_id, xyz, object_type, is_vertex, tcp, object21
        )


def build_episode_frame_graphs(episode, max_edge_m):
    """Build chronological B2-2 spatial graphs without inventing fixed corner IDs."""
    graphs = []
    for frame_index in range(episode.length):
        valid_rows = np.flatnonzero(
            episode.object_type[frame_index] != NodeType.PAD
        )
        nodes = []
        for row in valid_rows:
            vertex = bool(episode.is_vertex[frame_index, row])
            source = Source.OBSERVED if vertex else Source.TRACKER
            nodes.append(
                Node(
                    position=tuple(episode.xyz[frame_index, row]),
                    object_id=int(episode.object_type[frame_index, row]),
                    is_vertex=vertex,
                    vertex_id=None,
                    visible=source == Source.OBSERVED,
                    confidence=None,
                    source=source,
                )
            )
        frame = Frame(
            episode_id=episode.episode_id,
            index=frame_index,
            tcp=tuple(episode.tcp[frame_index]),
            nodes=tuple(nodes),
        )
        graphs.append(build_graph(frame, max_edge_m=max_edge_m))
    return graphs


def collate_episode_sequences(episodes, *, max_edge_m=0.08):
    """Pad chronological episodes and build B2-2 object-internal Delaunay graphs."""
    if not episodes:
        raise ValueError("Cannot collate an empty Five-mode episode batch")
    batch_size = len(episodes)
    time_steps = max(episode.length for episode in episodes)

    object21 = torch.zeros(batch_size, time_steps, 21, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)
    frame_indices = torch.full((batch_size, time_steps), -1, dtype=torch.long)
    lengths = torch.tensor([episode.length for episode in episodes], dtype=torch.long)
    frame_graphs = []

    for batch_index, episode in enumerate(episodes):
        length = episode.length
        graphs = build_episode_frame_graphs(episode, max_edge_m)
        frame_graphs.extend(graphs)
        padded_frame = Frame(
            episode_id=episode.episode_id,
            index=length,
            tcp=(0.0, 0.0, 0.0),
            nodes=(),
        )
        frame_graphs.extend(
            build_graph(padded_frame, max_edge_m=max_edge_m)
            for _ in range(time_steps - length)
        )
        object21[batch_index, :length] = torch.from_numpy(episode.object21)
        frame_mask[batch_index, :length] = True
        frame_indices[batch_index, :length] = torch.arange(length)

    return FiveModeGraphSequenceBatch(
        graph=batch_graphs(frame_graphs),
        targets={"object21": object21},
        frame_mask=frame_mask,
        frame_indices=frame_indices,
        lengths=lengths,
        episode_ids=[episode.episode_id for episode in episodes],
    )


def build_train_loader(config, shuffle_generator=None):
    """Build the unfiltered full-train episode loader and its stable identity."""
    settings = config["data"]
    forbidden = {
        name
        for name in ("condition", "condition_index", "balanced_train_condition", "train_mix")
        if name in settings
    }
    if forbidden:
        fields = ", ".join(sorted(forbidden))
        raise ValueError(f"Full-train sequence loading does not accept condition filters: {fields}")

    include_surface = settings.get("include_surface", False)
    if type(include_surface) is not bool:
        raise ValueError("data.include_surface must be true or false")
    dataset = FiveModeTrainSequenceDataset(
        settings["manifest"], include_surface=include_surface
    )
    expected_episodes = settings.get("expected_train_episodes")
    expected_frames = settings.get("expected_train_frames")
    if expected_episodes is not None and (
        type(expected_episodes) is not int or expected_episodes < 1
    ):
        dataset.close()
        raise ValueError("data.expected_train_episodes must be a positive integer")
    if expected_frames is not None and (
        type(expected_frames) is not int or expected_frames < 1
    ):
        dataset.close()
        raise ValueError("data.expected_train_frames must be a positive integer")
    if expected_episodes is not None and len(dataset) != expected_episodes:
        dataset.close()
        raise ValueError(
            f"Expected {expected_episodes} train episodes, found {len(dataset)}"
        )
    if expected_frames is not None and dataset.frame_count != expected_frames:
        dataset.close()
        raise ValueError(
            f"Expected {expected_frames} train frames, found {dataset.frame_count}"
        )

    episode_batch_size = config["training"]["episode_batch_size"]
    if type(episode_batch_size) is not int or episode_batch_size < 1:
        dataset.close()
        raise ValueError("training.episode_batch_size must be a positive integer")
    num_workers = settings.get("num_workers", 0)
    if type(num_workers) is not int or num_workers < 0:
        dataset.close()
        raise ValueError("data.num_workers must be a nonnegative integer")
    shuffle_episodes = settings.get("shuffle_episodes", True)
    if type(shuffle_episodes) is not bool:
        dataset.close()
        raise ValueError("data.shuffle_episodes must be true or false")
    pin_memory = settings.get("pin_memory", False)
    if type(pin_memory) is not bool:
        dataset.close()
        raise ValueError("data.pin_memory must be true or false")
    max_edge_m = settings.get("max_edge_m", 0.08)
    if (
        isinstance(max_edge_m, bool)
        or not isinstance(max_edge_m, (int, float))
        or not np.isfinite(max_edge_m)
        or max_edge_m <= 0
    ):
        dataset.close()
        raise ValueError("data.max_edge_m must be a positive number")

    loader = DataLoader(
        dataset,
        batch_size=episode_batch_size,
        shuffle=shuffle_episodes,
        generator=shuffle_generator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        collate_fn=partial(collate_episode_sequences, max_edge_m=float(max_edge_m)),
    )
    identity = {
        "manifest_sha256": file_sha256(dataset.manifest_path),
        "split": "train",
        "episodes": len(dataset),
        "frames": dataset.frame_count,
        "include_surface": include_surface,
        "graph_topology": "b2_2_object_delaunay_bidirectional",
        "max_edge_m": float(max_edge_m),
        "frame_order": "ascending_contiguous_within_episode",
    }
    identity["dataset_sha256"] = canonical_hash(identity)
    return loader, identity
