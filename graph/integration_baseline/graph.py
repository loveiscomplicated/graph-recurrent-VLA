"""Exact interactions and disjoint-union packing of unchanged Raw EDE packets."""
from vla.contracts.graph import Graph, PackedGraphs, pack_graphs, NODE_TYPES, EDGE_TYPES
from time import perf_counter

import numpy as np
import torch

from graph.vision_input import receive_vision_packet, RAW_EDE_POLICY



from graph.geometry.interactions import _integer, bidirectional, edge_attributes, global_peg_box_pairs, nearest_robot_pairs












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
