"""Episode datasets for spatial-context corner tracking and Object21 rollouts.

The tracker-training dataset reads the GT-assisted B2-2 preparation.  Canonical
IDs and reference coordinates are retained as supervision only.  The Five-mode
rollout dataset reads the current detector ``corners`` cache and TAPNext++
non-corner ``surface`` cache without visibility-condition filtering.
"""

from contextlib import ExitStack
from dataclasses import dataclass, fields
from functools import partial
import json
from pathlib import Path
import tarfile
from uuid import uuid4

import numpy as np
import torch
from scipy import __version__ as scipy_version
from torch.utils.data import DataLoader, Dataset

from data_collection.episode_dataset import canonical_hash, file_sha256
from graph.state_estimation.graph import (
    Frame,
    Graph,
    GraphBatch,
    Node,
    Source,
    batch_graphs,
    build_graph,
)
from training.gnn.packed_five_mode_data import read_archive_member


CORNER_COUNT = 24
CORNER_OBJECT_IDS = np.asarray([0] * 8 + [1] * 16, dtype=np.int64)
FORBIDDEN_CACHE_NAMES = {
    "tapnext_state36.npz",
    "tapnext_recovery_state36.npz",
}


def _reject_forbidden_cache_reference(value):
    """Fail before opening a legacy shared TAPNext++/State36 artifact."""
    path = Path(str(value))
    if path.name in FORBIDDEN_CACHE_NAMES:
        raise PermissionError(f"Forbidden legacy cache reference: {value}")


@dataclass
class CornerTrackingEpisode:
    """One chronological episode with offline labels kept outside model inputs."""

    episode_id: str
    graphs: list[Graph]
    rows: np.ndarray
    measurements: np.ndarray
    targets: np.ndarray
    last_observed: np.ndarray
    visible: np.ndarray
    seen_before: np.ndarray
    reappearance: np.ndarray
    age: np.ndarray
    tcp: np.ndarray

    @property
    def length(self):
        return len(self.graphs)


@dataclass
class CornerTrackingBatch:
    """Padded complete episodes with frame graphs flattened in B-major order."""

    graph: GraphBatch
    rows: torch.Tensor
    measurements: torch.Tensor
    targets: torch.Tensor
    last_observed: torch.Tensor
    visible: torch.Tensor
    seen_before: torch.Tensor
    reappearance: torch.Tensor
    age: torch.Tensor
    tcp: torch.Tensor
    frame_mask: torch.Tensor
    lengths: torch.Tensor
    episode_ids: list[str]

    def to(self, device):
        """Move both the frame graph and padded episode tensors together."""
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values["graph"] = self.graph.to(device)
        return CornerTrackingBatch(
            **{
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in values.items()
            }
        )


@dataclass
class FiveModeRolloutEpisode:
    """Detector corners, surface context and Object21 labels for one episode."""

    episode_id: str
    graphs: list[Graph]
    tcp: np.ndarray
    object21: np.ndarray

    @property
    def length(self):
        return len(self.graphs)


def _validate_tracker_arrays(arrays, episode_id):
    required = {
        "offsets",
        "xyz",
        "kinds",
        "vertex_ids",
        "is_corner",
        "tcp",
        "targets",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"{episode_id} is missing arrays: {sorted(missing)}")
    offsets = arrays["offsets"]
    frames = len(offsets) - 1
    rows = int(offsets[-1])
    if (
        offsets.ndim != 1
        or offsets.dtype.kind not in "iu"
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or arrays["xyz"].shape != (rows, 3)
        or arrays["kinds"].shape != (rows,)
        or arrays["vertex_ids"].shape != (rows,)
        or arrays["is_corner"].shape != (rows,)
        or arrays["tcp"].shape != (frames, 3)
        or arrays["targets"].shape != (frames, CORNER_COUNT, 3)
    ):
        raise ValueError(f"Invalid spatial corner episode shapes: {episode_id}")
    if not np.isfinite(arrays["xyz"]).all() or not np.isfinite(arrays["targets"]).all():
        raise ValueError(f"Nonfinite corner coordinates: {episode_id}")
    if not np.isin(arrays["kinds"], (0, 1)).all():
        raise ValueError(f"Corner/surface object IDs must be Peg=0 or Box=1: {episode_id}")
    identities = arrays["vertex_ids"]
    if np.any((identities < -1) | (identities >= CORNER_COUNT)):
        raise ValueError(f"Canonical corner IDs must be -1 or 0..23: {episode_id}")


def _tracking_episode(arrays, episode_id, max_edge_m):
    """Build Delaunay graphs and causal labels without exposing IDs as features."""
    _validate_tracker_arrays(arrays, episode_id)
    offsets = arrays["offsets"]
    frames = len(offsets) - 1
    rows = np.full((frames, CORNER_COUNT), -1, dtype=np.int64)
    measurements = np.zeros((frames, CORNER_COUNT, 3), dtype=np.float32)
    graphs = []

    for frame_index in range(frames):
        first, last = offsets[frame_index : frame_index + 2]
        nodes = []
        for local_row, source_row in enumerate(range(first, last)):
            is_corner = bool(arrays["is_corner"][source_row])
            vertex_id = int(arrays["vertex_ids"][source_row])
            if vertex_id >= 0:
                if rows[frame_index, vertex_id] >= 0:
                    raise ValueError(
                        f"Duplicate canonical corner {vertex_id} in {episode_id}/{frame_index}"
                    )
                rows[frame_index, vertex_id] = local_row
                measurements[frame_index, vertex_id] = arrays["xyz"][source_row]
            nodes.append(
                Node(
                    position=tuple(arrays["xyz"][source_row]),
                    object_id=int(arrays["kinds"][source_row]),
                    is_vertex=is_corner,
                    vertex_id=vertex_id if vertex_id >= 0 else None,
                    visible=is_corner,
                    confidence=None,
                    source=Source.OBSERVED if is_corner else Source.TRACKER,
                )
            )
        graphs.append(
            build_graph(
                Frame(
                    episode_id=episode_id,
                    index=frame_index,
                    tcp=tuple(arrays["tcp"][frame_index]),
                    nodes=tuple(nodes),
                ),
                max_edge_m=max_edge_m,
            )
        )

    visible = rows >= 0
    seen_before = np.zeros_like(visible)
    reappearance = np.zeros_like(visible)
    last_observed = np.zeros_like(measurements)
    age = np.zeros_like(rows)
    last_position = np.zeros((CORNER_COUNT, 3), dtype=np.float32)
    last_frame = np.full(CORNER_COUNT, -1, dtype=np.int64)
    for frame_index in range(frames):
        previously_seen = last_frame >= 0
        seen_before[frame_index] = previously_seen
        last_observed[frame_index] = last_position
        age[frame_index, previously_seen] = frame_index - last_frame[previously_seen]
        reappearance[frame_index] = visible[frame_index] & (age[frame_index] > 1)
        current = np.flatnonzero(visible[frame_index])
        last_position[current] = measurements[frame_index, current]
        last_frame[current] = frame_index

    targets = np.where(visible[..., None], measurements, arrays["targets"])
    return CornerTrackingEpisode(
        episode_id=episode_id,
        graphs=graphs,
        rows=rows,
        measurements=measurements,
        targets=targets.astype(np.float32, copy=False),
        last_observed=last_observed,
        visible=visible,
        seen_before=seen_before,
        reappearance=reappearance,
        age=age,
        tcp=arrays["tcp"].astype(np.float32, copy=False),
    )


class CornerTrackingDataset(Dataset):
    """Read one split of the checksummed B2-2/B2-3 prepared episodes."""

    def __init__(self, manifest_path, split, *, max_edge_m=0.08, graph_cache_root=None):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        _reject_forbidden_cache_reference(self.manifest_path)
        self.root = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text())
        records = manifest.get("episodes", [])
        if not records or len({row.get("episode") for row in records}) != len(records):
            raise ValueError("Corner tracking manifest needs unique episodes")
        self.records = [row for row in records if row.get("split") == split]
        if not self.records:
            raise ValueError(f"Corner tracking manifest has no {split} episodes")
        self.split = split
        self.max_edge_m = float(max_edge_m)
        if not np.isfinite(self.max_edge_m) or self.max_edge_m <= 0:
            raise ValueError("max_edge_m must be positive")
        self.graph_cache_root = None
        self.graph_cache_identity = None
        if graph_cache_root is not None:
            self.graph_cache_root = Path(graph_cache_root).expanduser().resolve()
            _reject_forbidden_cache_reference(self.graph_cache_root)
            self.graph_cache_root.mkdir(parents=True, exist_ok=True)
            graph_source = Path(__file__).resolve().parents[2] / "graph/state_estimation/graph.py"
            codec_source = graph_source.with_name("codec.py")
            self.graph_cache_identity = {
                "loader_sha256": file_sha256(Path(__file__)),
                "graph_sha256": file_sha256(graph_source),
                "codec_sha256": file_sha256(codec_source),
                "max_edge_m": self.max_edge_m,
                "numpy": np.__version__,
                "scipy": scipy_version,
                "torch": torch.__version__,
            }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        _reject_forbidden_cache_reference(row["file"])
        path = (self.root / row["file"]).resolve()
        if not path.is_relative_to(self.root) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Corner tracking episode checksum/path mismatch: {path}")
        cache_identity = {
            "implementation": self.graph_cache_identity,
            "episode_sha256": row["sha256"],
            "episode": row["episode"],
            "frames": row["frames"],
        }
        cache_path = None
        if self.graph_cache_root is not None:
            cache_path = self.graph_cache_root / f"{canonical_hash(cache_identity)}.pt"
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if cached.get("identity") != cache_identity:
                    raise ValueError(f"Corner graph cache identity mismatch: {cache_path}")
                return cached["episode"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        if len(arrays["offsets"]) - 1 != row["frames"]:
            raise ValueError(f"Corner tracking episode length differs: {row['episode']}")
        episode = _tracking_episode(arrays, row["episode"], self.max_edge_m)
        if cache_path is not None:
            # Workers may request the same episode. Publish complete CPU-only
            # entries atomically without exposing another worker's partial file.
            temporary = cache_path.with_suffix(f".{uuid4().hex}.partial")
            try:
                torch.save({"identity": cache_identity, "episode": episode}, temporary)
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        return episode


def collate_corner_tracking(episodes):
    """Pad episode ends while preserving every chronological frame and graph."""
    if not episodes:
        raise ValueError("Cannot collate an empty corner-tracking batch")
    batch_size = len(episodes)
    time_steps = max(episode.length for episode in episodes)
    shape = (batch_size, time_steps, CORNER_COUNT)
    rows = np.full(shape, -1, dtype=np.int64)
    measurements = np.zeros((*shape, 3), dtype=np.float32)
    targets = np.zeros_like(measurements)
    last_observed = np.zeros_like(measurements)
    visible = np.zeros(shape, dtype=bool)
    seen_before = np.zeros(shape, dtype=bool)
    reappearance = np.zeros(shape, dtype=bool)
    age = np.zeros(shape, dtype=np.int64)
    tcp = np.zeros((batch_size, time_steps, 3), dtype=np.float32)
    frame_mask = np.zeros((batch_size, time_steps), dtype=bool)
    graphs = []

    for batch_index, episode in enumerate(episodes):
        length = episode.length
        graphs.extend(episode.graphs)
        empty = build_graph(
            Frame(episode.episode_id, length, (0.0, 0.0, 0.0), ()), max_edge_m=0.08
        )
        graphs.extend(empty for _ in range(time_steps - length))
        rows[batch_index, :length] = episode.rows
        measurements[batch_index, :length] = episode.measurements
        targets[batch_index, :length] = episode.targets
        last_observed[batch_index, :length] = episode.last_observed
        visible[batch_index, :length] = episode.visible
        seen_before[batch_index, :length] = episode.seen_before
        reappearance[batch_index, :length] = episode.reappearance
        age[batch_index, :length] = episode.age
        tcp[batch_index, :length] = episode.tcp
        frame_mask[batch_index, :length] = True

    tensor = torch.from_numpy
    return CornerTrackingBatch(
        graph=batch_graphs(graphs),
        rows=tensor(rows),
        measurements=tensor(measurements),
        targets=tensor(targets),
        last_observed=tensor(last_observed),
        visible=tensor(visible),
        seen_before=tensor(seen_before),
        reappearance=tensor(reappearance),
        age=tensor(age),
        tcp=tensor(tcp),
        frame_mask=tensor(frame_mask),
        lengths=torch.tensor([episode.length for episode in episodes]),
        episode_ids=[episode.episode_id for episode in episodes],
    )


def build_corner_tracking_loaders(config, generator=None):
    """Build unfiltered train/validation episode loaders and their identity."""
    settings = config["data"]
    forbidden = {name for name in ("condition", "condition_index", "train_mix") if name in settings}
    if forbidden:
        raise ValueError(f"Corner tracking does not accept condition filters: {sorted(forbidden)}")
    max_edge_m = float(settings.get("max_edge_m", 0.08))
    dataset = partial(
        CornerTrackingDataset,
        settings["manifest"],
        max_edge_m=max_edge_m,
        graph_cache_root=settings.get("graph_cache_root"),
    )
    train = dataset("train")
    validation = dataset("validation")
    expected = settings.get("expected_episodes", {})
    for split, dataset in (("train", train), ("validation", validation)):
        count = expected.get(split)
        if count is not None and (type(count) is not int or count < 1 or len(dataset) != count):
            raise ValueError(
                f"Expected {count} {split} corner-tracking episodes, found {len(dataset)}"
            )
    batch_size = config["training"]["episode_batch_size"]
    workers = settings.get("num_workers", 0)
    if type(batch_size) is not int or batch_size < 1 or type(workers) is not int or workers < 0:
        raise ValueError("episode_batch_size must be positive and num_workers nonnegative")
    loader = partial(
        DataLoader,
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
        collate_fn=collate_corner_tracking,
    )
    loaders = {
        "train": loader(train, shuffle=True, generator=generator),
        "validation": loader(validation, shuffle=False),
    }
    identity = {
        "manifest_sha256": file_sha256(train.manifest_path),
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "graph_topology": "object_internal_delaunay_bidirectional",
        "surface_role": "tapnext_non_corner_context",
        "max_edge_m": max_edge_m,
        "condition_filter": None,
    }
    identity["dataset_sha256"] = canonical_hash(identity)
    return loaders, identity


class FiveModeRolloutDataset(Dataset):
    """Read current-frame detector and TAPNext++ graphs for a requested split."""

    def __init__(self, manifest_path, split, *, max_edge_m=0.08):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        _reject_forbidden_cache_reference(self.manifest_path)
        self.root = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text())
        records = manifest.get("records", [])
        if manifest.get("format") != "five_mode_supervised_v1":
            raise ValueError("Expected a five_mode_supervised_v1 manifest")
        self.archive_names = {row["file"] for row in manifest.get("archives", [])}
        self.records = [row for row in records if row.get("split") == split]
        if not self.records:
            raise ValueError(f"Five-mode manifest has no {split} episodes")
        self.max_edge_m = float(max_edge_m)
        self._stack = None
        self._archives = {}

    def __len__(self):
        return len(self.records)

    def close(self):
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self._archives.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_stack"] = None
        state["_archives"] = {}
        return state

    def _archive(self, name):
        _reject_forbidden_cache_reference(name)
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
        row = self.records[index]
        tables = {}
        for kind in ("corners", "surface", "targets"):
            receipt = row["files"][kind]
            _reject_forbidden_cache_reference(receipt["member"])
            tables[kind] = read_archive_member(
                self._archive(receipt["archive"]), receipt, kind
            )
        frame_indices = np.arange(row["frames"])
        if any(not np.array_equal(table["frame_indices"], frame_indices) for table in tables.values()):
            raise ValueError(f"Five-mode episode is not contiguous: {row['seed']}")

        corners, surface, targets = tables["corners"], tables["surface"], tables["targets"]
        surface_kinds = surface.get("surface_object_id")
        if surface_kinds is None:
            # Historical Five-mode releases tracked Peg surfaces only and
            # therefore omitted a redundant object-ID array.  New two-object
            # caches carry explicit IDs and take the branch below unchanged.
            surface_kinds = np.zeros(len(surface["surface_world_xyz"]), dtype=np.int8)
        elif not np.isin(surface_kinds, (0, 1)).all():
            raise ValueError("TAPNext++ surface object IDs must be Peg=0 or Box/Hole=1")
        graphs = []
        episode_id = f"{row['seed']}:{row['policy_mode']}"
        for frame_index in range(row["frames"]):
            corner_slice = slice(*corners["corner_offsets"][frame_index : frame_index + 2])
            surface_slice = slice(*surface["surface_offsets"][frame_index : frame_index + 2])
            nodes = [
                Node(tuple(xyz), int(kind), True, None, True, None, Source.OBSERVED)
                for xyz, kind in zip(
                    corners["corner_world_xyz"][corner_slice],
                    corners["corner_object_id"][corner_slice],
                    strict=True,
                )
            ]
            nodes.extend(
                Node(tuple(xyz), int(kind), False, None, False, None, Source.TRACKER)
                for xyz, kind in zip(
                    surface["surface_world_xyz"][surface_slice],
                    surface_kinds[surface_slice],
                    strict=True,
                )
            )
            graphs.append(
                build_graph(
                    Frame(episode_id, frame_index, tuple(targets["tcp"][frame_index]), tuple(nodes)),
                    max_edge_m=self.max_edge_m,
                )
            )
        object21 = targets["object21"].astype(np.float32, copy=False)
        tcp = targets["tcp"].astype(np.float32, copy=False)
        if object21.shape != (row["frames"], 21) or tcp.shape != (row["frames"], 3):
            raise ValueError(f"Invalid Five-mode targets: {episode_id}")
        return FiveModeRolloutEpisode(episode_id, graphs, tcp, object21)


def rollout_dataset_identity(dataset):
    """Return the frozen input identity used by downstream rollout creation."""
    value = {
        "manifest_sha256": file_sha256(dataset.manifest_path),
        "episodes": len(dataset),
        "graph_topology": "object_internal_delaunay_bidirectional",
        "corner_source": "current_rgbd_detector",
        "surface_source": "tapnext_non_corner",
        "condition_filter": None,
    }
    value["dataset_sha256"] = canonical_hash(value)
    return value
