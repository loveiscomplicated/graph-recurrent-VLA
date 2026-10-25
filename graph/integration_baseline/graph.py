"""Exact interactions and disjoint-union packing of unchanged Raw EDE packets."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from graph.vision_input import receive_vision_packet, RAW_EDE_POLICY

NODE_TYPES = ('TCP', 'Left', 'Right', 'Peg', 'Box')
EDGE_TYPES = ('Intra-object', 'Robot-Object', 'Peg-Box')


def _integer(value, minimum, name):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be integer >= {minimum}')


def nearest_robot_pairs(positions, object_id, source_index, k=4):
    _integer(k, 1, 'k')
    pairs = []
    for robot in range(3):
        for oid in (0, 1):
            rows = np.flatnonzero(object_id == oid)
            delta = positions[rows+3].astype(np.float64) - positions[robot].astype(np.float64)
            d2 = np.sum(delta*delta, axis=1)
            order = np.lexsort((rows, source_index[rows], d2))[:k]
            pairs.extend((robot, int(row)+3) for row in rows[order])
    return np.asarray(pairs, np.int64).reshape(-1, 2)


def global_peg_box_pairs(points, object_id, source_index, M=16, block_rows=64):
    """Exact all-pair block scan; O(PQ) distances, O(block_rows*Q+M) scratch.

    Total order is (squared distance, peg source, box source, peg row, box
    row). Keeping each block's first M cannot discard a global first-M pair.
    No cutoff, approximation, normal/depth gate or geometry filter is used.
    """
    _integer(M, 0, 'M')
    _integer(block_rows, 1, 'block_rows')
    peg, box = (np.flatnonzero(object_id == oid) for oid in (0, 1))
    best = np.empty((0, 2), np.int64)
    best_dist = np.empty(0, np.float64)
    if M == 0 or not len(peg) or not len(box):
        return best
    xyz = points.astype(np.float64)
    for start in range(0, len(peg), block_rows):
        rows = peg[start:start+block_rows]
        delta = xyz[rows, None] - xyz[box][None]
        d2 = np.sum(delta*delta, axis=-1).reshape(-1)
        pairs = np.column_stack((np.repeat(rows, len(box)), np.tile(box, len(rows))))
        pairs = np.concatenate((best, pairs))
        d2 = np.r_[best_dist, d2]
        order = np.lexsort((pairs[:, 1], pairs[:, 0], source_index[pairs[:, 1]],
                            source_index[pairs[:, 0]], d2))[:M]
        best, best_dist = pairs[order], d2[order]
    return best


def bidirectional(pairs):
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    return np.concatenate((pairs, pairs[:, ::-1]), axis=0).T.copy()


def edge_attributes(positions, edges):
    delta = positions[edges[1]].astype(np.float64) - positions[edges[0]].astype(np.float64)
    distance = np.linalg.norm(delta, axis=1)
    direction = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0)
    return np.column_stack((distance, direction)).astype(np.float32)


@dataclass
class Graph:
    node_input: torch.Tensor  # pos3, finite F5, available3, valid3
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_type: torch.Tensor
    packet: dict  # immutable independent deep-copy from the existing receiver
    timings_ms: dict


def build_graph(packet, robot_positions, *, k=4, M=16, block_rows=64):
    tick = perf_counter()
    if (packet.get('schema_version') != 2 or packet.get('edge_status') != 'supplied'
            or packet.get('edge_policy') != RAW_EDE_POLICY or packet.get('coordinate_frame') != 'EE_metres'):
        raise ValueError('EE v2 supplied Raw EDE packet required')
    received = receive_vision_packet(packet, robot_positions=robot_positions)
    after_receive = perf_counter()
    p = received.packet
    oid, source = p['object_id'], p['provenance']['source_index']
    ro = bidirectional(nearest_robot_pairs(received.positions, oid, source, k))
    after_robot = perf_counter()
    pb = bidirectional(global_peg_box_pairs(p['points'], oid, source, M, block_rows)+3)
    after_pairs = perf_counter()
    edges = np.concatenate((received.edge_index, ro, pb), axis=1)
    types = np.repeat(np.arange(3, dtype=np.int64), [received.edge_index.shape[1], ro.shape[1], pb.shape[1]])
    n = len(received.positions)
    features = np.zeros((n, 5), np.float32)
    available, valid = np.zeros((n, 3), bool), np.zeros((n, 3), bool)
    available[3:] = p['feature_field_available']
    valid[3:] = p['feature_field_valid']
    usable = (available[3:] & valid[3:])[:, [0, 0, 0, 1, 2]]
    # Preserve original NaNs/dtypes/provenance; create a separate network view.
    features[3:] = np.where(usable, p['geometric_features'], 0).astype(np.float32)
    node_input = np.column_stack((received.positions, features, available, valid)).astype(np.float32)
    if not np.isfinite(node_input).all():
        raise ValueError('nonfinite network input')
    graph = Graph(torch.from_numpy(node_input), torch.from_numpy(np.r_[np.arange(3), oid+3]),
                  torch.from_numpy(edges), torch.from_numpy(edge_attributes(received.positions, edges)),
                  torch.from_numpy(types), p, {})
    graph.timings_ms = dict(receiver_validation_copy=(after_receive-tick)*1000,
                            robot_object_selection=(after_robot-after_receive)*1000,
                            peg_box_global_selection=(after_pairs-after_robot)*1000,
                            tensor_packing=(perf_counter()-after_pairs)*1000,
                            graph_construction_total=(perf_counter()-tick)*1000)
    return graph


@dataclass
class PackedGraphs:
    node_input: torch.Tensor
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_type: torch.Tensor
    node_graph: torch.Tensor
    ptr: torch.Tensor
    frame_map: torch.Tensor  # G,2: sequence batch index, timestep slot

    @property
    def count(self):
        return self.ptr.numel()-1

    def to(self, device):
        return PackedGraphs(**{name: value.to(device) for name, value in vars(self).items()})


def pack_graphs(graphs, frame_map=None):
    if not graphs:
        raise ValueError('at least one graph required')
    if frame_map is None:
        frame_map = [(0, i) for i in range(len(graphs))]
    mapping = torch.as_tensor(frame_map, dtype=torch.long)
    if mapping.shape != (len(graphs), 2) or (mapping < 0).any() or len(torch.unique(mapping, dim=0)) != len(graphs):
        raise ValueError('unique nonnegative sequence/frame mapping required')
    ptr = np.r_[0, np.cumsum([len(g.node_type) for g in graphs])]
    edges = torch.cat([g.edge_index + int(ptr[i]) for i, g in enumerate(graphs)], dim=1)
    node_graph = torch.repeat_interleave(torch.arange(len(graphs)), torch.tensor(np.diff(ptr)))
    if edges.numel() and not torch.equal(node_graph[edges[0]], node_graph[edges[1]]):
        raise ValueError('cross-graph edge')
    return PackedGraphs(*(torch.cat([getattr(g, key) for g in graphs], dim=0) for key in ('node_input', 'node_type')),
                        edges, torch.cat([g.edge_attr for g in graphs]), torch.cat([g.edge_type for g in graphs]),
                        node_graph, torch.tensor(ptr), mapping)
