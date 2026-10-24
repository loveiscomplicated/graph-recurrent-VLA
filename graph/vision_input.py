"""Opt-in experimental Vision packet receiver; no GNN encoder or topology.

Rows are copied without sorting/resampling. Coordinates retain the packet's frame.
Robot positions, when supplied, must already use that same coordinate frame.
Incomplete feature/edge packets are audit containers, not runnable GNN inputs.
"""
from dataclasses import dataclass
import copy

import numpy as np


def _array(value, dtype, shape, name):
    a = np.asarray(value)
    if a.dtype != np.dtype(dtype) or a.shape != shape:
        raise ValueError(f'{name}: expected {dtype} {shape}, got {a.dtype} {a.shape}')
    return a


def validate_packet(packet):
    """Validate this experiment's explicit schema, never infer missing metadata."""
    for key in ('episode', 'camera', 'coordinate_frame'):
        if not isinstance(packet.get(key), str) or not packet[key]:
            raise ValueError('missing packet identity: '+key)
    if type(packet.get('source_frame')) is not int or packet['source_frame'] < 0:
        raise ValueError('invalid source_frame')
    points = np.asarray(packet['points'])
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype != np.float32 or not np.isfinite(points).all():
        raise ValueError('finite float32 [N,3] points required')
    n = len(points)
    oid = _array(packet['object_id'], np.int64, (n,), 'object_id')
    if not np.isin(oid, [0, 1]).all():
        raise ValueError('unknown object_id')
    p = packet['provenance']
    source = _array(p['source_index'], np.int64, (n,), 'source_index')
    if (source < 0).any() or len(set(zip(oid.tolist(), source.tolist()))) != n:
        raise ValueError('invalid/duplicate source identity')
    if not np.array_equal(_array(p['object_id'], np.int64, (n,), 'provenance.object_id'), oid):
        raise ValueError('provenance object_id row mismatch')
    evidence = _array(p['candidate_evidence'], bool, (n, 2), 'candidate_evidence')
    both = _array(p['both_evidence'], bool, (n,), 'both_evidence')
    chosen = _array(p['selected_by_structural_source'], bool, (n, 2), 'selected_by_structural_source')
    surface = _array(p['surface_selected'], bool, (n,), 'surface_selected')
    stage = _array(p['selection_stage'], np.uint8, (n,), 'selection_stage')
    if (not np.array_equal(both, evidence.all(axis=1)) or (chosen & ~evidence).any()
            or not np.array_equal(stage == 0, chosen.any(axis=1))
            or not np.array_equal(surface, stage == 1) or not np.isin(stage, [0, 1]).all()):
        raise ValueError('evidence/selection role mismatch')
    features = np.asarray(packet['geometric_features'])
    fields = packet['feature_fields']
    width = sum(int(np.prod(f['shape'])) if f['shape'] else 1 for f in fields)
    if features.shape != (n, width) or features.dtype != np.float64 or np.isinf(features).any():
        raise ValueError('invalid feature view')
    available = _array(p['feature_available'], bool, (n,), 'feature_available')
    fa = _array(p['feature_field_available'], bool, (n, len(fields)), 'feature_field_available')
    if not np.array_equal(available, fa.all(axis=1)):
        raise ValueError('feature availability mismatch')
    if len({f['name'] for f in fields}) != len(fields):
        raise ValueError('duplicate feature field')
    offset = 0
    for j, field in enumerate(fields):
        w = int(np.prod(field['shape'])) if field['shape'] else 1
        if not np.isnan(features[~fa[:, j], offset:offset+w]).all():
            raise ValueError('unavailable feature must be NaN')
        compact = packet['compact_features'][field['name']]
        pos = np.asarray(compact['positions'])
        values = np.asarray(compact['values'])
        if (pos.dtype != np.int64 or not np.array_equal(pos, np.flatnonzero(fa[:, j]))
                or values.shape != (len(pos), *field['shape']) or values.dtype.str != field['dtype']):
            raise ValueError('compact feature alignment/dtype mismatch')
        if not np.array_equal(features[pos, offset:offset+w], values.reshape(len(pos), w), equal_nan=True):
            raise ValueError('compact feature values mismatch')
        offset += w
    edge = np.asarray(packet['intra_object_edge_index'])
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype != np.int64:
        raise ValueError('int64 [2,E] edges required')
    if edge.size and ((edge < 0).any() or (edge >= n).any() or (oid[edge[0]] != oid[edge[1]]).any()):
        raise ValueError('edge range/object mismatch')
    if packet['edge_status'] not in ('supplied', 'not_implemented'):
        raise ValueError('unknown edge status')
    if packet['edge_status'] == 'not_implemented' and edge.size:
        raise ValueError('unimplemented edges must be empty')


@dataclass(frozen=True)
class VisionInput:
    packet: dict
    positions: np.ndarray
    point_node_index: np.ndarray
    node_kind: np.ndarray  # 0=point, 1=TCP, 2=left, 3=right
    node_source_index: np.ndarray
    node_candidate_evidence: np.ndarray
    robot_mask: np.ndarray
    edge_index: np.ndarray


def receive_vision_packet(packet, *, robot_positions=None):
    """Preserve supplied point rows; optionally prepend three supplied robot rows."""
    validate_packet(packet)
    saved = copy.deepcopy(packet)
    n = len(saved['points'])
    if robot_positions is None:
        offset = 0
        positions = saved['points'].copy()
    else:
        robot = _array(robot_positions, np.float32, (3, 3), 'robot_positions')
        if not np.isfinite(robot).all():
            raise ValueError('nonfinite robot positions')
        offset = 3
        positions = np.concatenate((robot, saved['points']))
    kinds = np.r_[np.arange(1, offset+1, dtype=np.uint8), np.zeros(n, np.uint8)]
    return VisionInput(saved, positions, np.arange(n, dtype=np.int64)+offset, kinds,
                       np.r_[np.full(offset, -1, np.int64), saved['provenance']['source_index']],
                       np.vstack((np.zeros((offset, 2), bool), saved['provenance']['candidate_evidence'])),
                       kinds != 0, saved['intra_object_edge_index'].copy()+offset)
