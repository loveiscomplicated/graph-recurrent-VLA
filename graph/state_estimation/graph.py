"""Build per-object, distance-limited Delaunay graphs from observations."""

from dataclasses import dataclass, fields
from enum import IntEnum
from itertools import combinations, pairwise
from numbers import Integral, Real

import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from graph.state_estimation.codec import COORDINATE_SCALE_M

NODE_FEATURE_DIM = 13
EDGE_FEATURE_DIM = 7


class Source(IntEnum):
    OBSERVED = 0
    TRACKER = 1
    MISSING = 2


@dataclass(frozen=True)
class Node:
    """One observed, estimated or missing node with an optional verified ID."""

    position: tuple[float, float, float] | None
    object_id: int
    is_vertex: bool
    vertex_id: int | None
    visible: bool
    confidence: float | None
    source: Source


@dataclass(frozen=True)
class Frame:
    """Inference inputs at the original, episode-local control frame index."""

    episode_id: str
    index: int
    tcp: tuple[float, float, float]
    nodes: tuple[Node, ...]


@dataclass
class Graph:
    """One unpadded frame graph. Missing vertices occupy slots without geometry."""

    features: Tensor
    position: Tensor
    object_id: Tensor
    vertex_id: Tensor
    visible: Tensor
    geometry: Tensor
    quality: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    methods: tuple[str, str]
    rejected_edges: int
    episode_id: str
    frame_index: int


@dataclass
class GraphBatch:
    """Padded node rows with flattened, sample-local sparse edges."""

    features: Tensor
    position: Tensor
    object_id: Tensor
    vertex_id: Tensor
    visible: Tensor
    geometry: Tensor
    quality: Tensor
    valid: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    episode_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    vertex_rows: tuple[dict[tuple[int, int], int], ...]
    observed_vertex_rows: tuple[dict[tuple[int, int], int], ...]

    def to(self, device):
        """Move every model input to the same device."""
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        return GraphBatch(
            **{
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in values.items()
            }
        )


def _position(value):
    """Validate metre-valued geometry against the workspace scale."""
    position = np.asarray(value, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Position must be a finite xyz vector")
    if np.max(np.abs(position)) > 10:
        raise ValueError("Position exceeds 10 metres; expected world xyz in metres")
    return position


def _validate_node(node):
    if type(node.object_id) is not int or node.object_id not in (0, 1):
        raise ValueError("Expected object ID 0 (Peg) or 1 (Hole assembly)")
    if type(node.is_vertex) is not bool:
        raise ValueError("Expected a boolean vertex indicator")
    if node.vertex_id is not None and (
        not node.is_vertex or type(node.vertex_id) is not int or node.vertex_id < 0
    ):
        raise ValueError(
            "Expected a nonnegative vertex ID on a semantic vertex, or None"
        )
    if not isinstance(node.source, Source):
        raise TypeError("Expected an observation source")
    if type(node.visible) is not bool or node.visible != (
        node.source == Source.OBSERVED
    ):
        raise ValueError("Expected visibility to agree with direct observation source")
    if node.confidence is not None and (
        isinstance(node.confidence, bool)
        or not isinstance(node.confidence, Real)
        or not 0 <= node.confidence <= 1
    ):
        raise ValueError("Expected confidence in [0,1], or None")
    if node.source == Source.MISSING and node.confidence is not None:
        raise ValueError("Missing observations cannot supply confidence")


def _triangulate(points):
    """Triangulate unique positions in their affine hull and expand duplicates."""
    unique, inverse = np.unique(points, axis=0, return_inverse=True)
    if len(unique) < 2:
        return (), "point" if len(unique) else "empty"

    centered = unique - unique.mean(axis=0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    rank = int((singular > max(singular[0] * 1e-6, 1e-9)).sum())
    if rank == 0:
        return (), "point"
    projected = centered @ basis[:rank].T
    if rank == 1:
        order = projected[:, 0].argsort()
        links = list(pairwise(order))
        method = "line"
    else:
        method = f"delaunay_{rank}d"
        try:
            simplices = Delaunay(projected).simplices
            links = {
                tuple(sorted(edge))
                for simplex in simplices
                for edge in combinations(simplex, 2)
            }
        except QhullError:
            # With no reliable triangulation, preserve isolated self-features.
            return (), "qhull_isolated"

    groups = [[] for _ in unique]
    for row, group in enumerate(inverse):
        groups[group].append(row)
    edges = {
        tuple(sorted((i, j))) for a, b in links for i in groups[a] for j in groups[b]
    }
    return sorted(edges), method


def build_graph(frame: Frame, *, max_edge_m=0.08) -> Graph:
    """Keep edges inside each object and exclude unsupported or long links."""
    if (
        not isinstance(frame.index, Integral)
        or isinstance(frame.index, bool)
        or frame.index < 0
    ):
        raise ValueError("Frame index must be a nonnegative integer")
    if not frame.episode_id or not np.isfinite(max_edge_m) or max_edge_m <= 0:
        raise ValueError(
            "Expected a nonempty episode ID and positive maximum edge length"
        )
    tcp = _position(frame.tcp)
    positions, features, kinds, vertices, visible, geometry, qualities = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    bound_ids = set()
    for node in frame.nodes:
        _validate_node(node)
        if node.vertex_id is not None:
            key = (node.object_id, node.vertex_id)
            if key in bound_ids:
                raise ValueError(
                    f"duplicate vertex identity {key} in frame {frame.index}"
                )
            bound_ids.add(key)
        supported = node.source != Source.MISSING
        xyz = _position(node.position) if supported else np.zeros(3)
        relative = (xyz - tcp) / COORDINATE_SCALE_M if supported else xyz
        observed_quality = node.confidence if node.confidence is not None else 0.0
        features.append(
            [
                *relative,
                node.object_id == 0,
                node.object_id == 1,
                node.is_vertex,
                not node.is_vertex,
                node.visible,
                observed_quality,
                node.confidence is not None,
                *(node.source == source for source in Source),
            ]
        )
        positions.append(xyz)
        kinds.append(node.object_id)
        vertices.append(-1 if node.vertex_id is None else node.vertex_id)
        visible.append(node.visible)
        geometry.append(supported)
        # Neutral weight for unavailable confidence, with a separate validity feature.
        qualities.append(1.0 if node.confidence is None else node.confidence)

    positions = np.asarray(positions).reshape(-1, 3)
    edges, attributes, methods = [], [], []
    rejected = 0
    for object_id in (0, 1):
        rows = [i for i, kind in enumerate(kinds) if kind == object_id and geometry[i]]
        links, method = _triangulate(positions[rows])
        methods.append(method)
        for a, b in links:
            src, dst = rows[a], rows[b]
            delta = positions[dst] - positions[src]
            distance = np.linalg.norm(delta)
            if distance > max_edge_m:
                rejected += 1
                continue
            relation = (
                2 - int(frame.nodes[src].is_vertex) - int(frame.nodes[dst].is_vertex)
            )
            one_hot = [relation == index for index in range(3)]
            edges.extend([(src, dst), (dst, src)])
            attributes.extend(
                [(*delta, distance, *one_hot), (*-delta, distance, *one_hot)]
            )

    return Graph(
        torch.tensor(features, dtype=torch.float32).reshape(-1, NODE_FEATURE_DIM),
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(kinds, dtype=torch.long),
        torch.tensor(vertices, dtype=torch.long),
        torch.tensor(visible, dtype=torch.bool),
        torch.tensor(geometry, dtype=torch.bool),
        torch.tensor(qualities, dtype=torch.float32),
        torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
        torch.tensor(attributes, dtype=torch.float32).reshape(-1, EDGE_FEATURE_DIM),
        tuple(methods),
        rejected,
        frame.episode_id,
        frame.index,
    )


def batch_graphs(graphs):
    """Batch arbitrary node counts and retain a padding slot for an all-empty batch."""
    if not graphs:
        raise ValueError("Cannot batch zero frames")
    width = max(1, *(len(graph.features) for graph in graphs))
    lengths = torch.tensor([len(graph.features) for graph in graphs])
    valid = torch.arange(width)[None, :] < lengths[:, None]

    def pad(name, value=0):
        values = [getattr(graph, name) for graph in graphs]
        padded = pad_sequence(values, batch_first=True, padding_value=value)
        if padded.shape[1] == 0:
            return padded.new_full((len(graphs), 1, *padded.shape[2:]), value)
        return padded

    edges = torch.cat(
        [graph.edge_index + i * width for i, graph in enumerate(graphs)], dim=1
    )
    vertex_rows = tuple(
        {
            (int(kind), int(vertex)): row
            for row, (kind, vertex) in enumerate(
                zip(graph.object_id.tolist(), graph.vertex_id.tolist())
            )
            if vertex >= 0
        }
        for graph in graphs
    )
    observed_vertex_rows = tuple(
        {key: row for key, row in lookup.items() if graph.visible[row]}
        for lookup, graph in zip(vertex_rows, graphs)
    )
    return GraphBatch(
        pad("features"),
        pad("position"),
        pad("object_id", -1),
        pad("vertex_id", -1),
        pad("visible"),
        pad("geometry"),
        pad("quality"),
        valid,
        edges,
        torch.cat([graph.edge_attr for graph in graphs]),
        tuple(graph.episode_id for graph in graphs),
        tuple(graph.frame_index for graph in graphs),
        vertex_rows,
        observed_vertex_rows,
    )
