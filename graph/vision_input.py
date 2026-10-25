"""Opt-in experimental Vision packet receiver; no GNN encoder or topology.

Rows are copied without sorting/resampling. Coordinates retain the packet's frame.
Robot positions, when supplied, must already use that same coordinate frame.
Incomplete feature/edge packets are audit containers, not runnable GNN inputs.
"""
from dataclasses import dataclass
import copy

import numpy as np

# Evaluation contract only. The receiver trusts the Vision candidate algorithm;
# it checks supplied row algebra without Qhull or collinearity reconstruction.
RAW_EDE_POLICY = dict(name='raw_extended_delaunay', version=1,
                      candidate_mode='extended_delaunay', edge_validation='none',
                      evaluation_only=True)


def _array(value, dtype, shape, name):
    a = np.asarray(value)
    if a.dtype != np.dtype(dtype) or a.shape != shape:
        raise ValueError(f'{name}: expected {dtype} {shape}, got {a.dtype} {a.shape}')
    return a


def validate_packet(packet):
    """Validate this experiment's explicit schema, never infer missing metadata."""
    version = packet.get('schema_version', 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError('unsupported packet schema_version')
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
    feature_namespace = packet if version == 2 else p
    available = _array(feature_namespace['feature_available'], bool, (n,), 'feature_available')
    fa = _array(feature_namespace['feature_field_available'], bool, (n, len(fields)), 'feature_field_available')
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
    if version == 2:
        _validate_v2_features(packet, n, source, oid, fa)
        if packet['edge_status'] == 'supplied':
            _validate_v2_edges(packet, n, oid, edge)


def _validate_v2_edges(packet, n, oid, edge):
    from visual.runtime.intra_object_edges import EDGE_POLICY
    from visual.runtime.triangle_intra_object_edges import TRIANGLE_POLICY, triangles_to_edges
    from visual.runtime.extended_delaunay_edges import DE_POLICY, EDE_POLICY
    policy = packet.get('edge_policy')
    if policy not in (EDGE_POLICY, TRIANGLE_POLICY, DE_POLICY, EDE_POLICY, RAW_EDE_POLICY):
        raise ValueError('unknown v2 intra-object edge policy')
    component = _array(packet['observed_component_id'], np.int64, (n,), 'observed_component_id')
    if ((component < 0).any() or (policy == EDGE_POLICY and edge.shape[1] > 16*n)
            or (edge[0] == edge[1]).any()
            or (component[edge[0]] != component[edge[1]]).any()):
        raise ValueError('self-loop/component/edge-count violation')
    if policy == RAW_EDE_POLICY:
        if type(policy['version']) is not int or policy['evaluation_only'] is not True:
            raise ValueError('raw evaluation policy type mismatch')
        _validate_raw_ede_metadata(packet, n, oid, component, edge)
        return
    pairs = set(map(tuple, edge.T.tolist()))
    if len(pairs) != edge.shape[1] or any((j,i) not in pairs for i,j in pairs):
        raise ValueError('duplicate/asymmetric supplied edge')
    if policy in (DE_POLICY, EDE_POLICY):
        _validate_edgewise_metadata(packet, n, oid, component, pairs, policy == EDE_POLICY)
    if policy == TRIANGLE_POLICY:
        # Packet algebra only: dense RGB-D validation belongs to the helper.
        t = np.asarray(packet.get('accepted_triangles'))
        if (t.ndim != 2 or t.shape[1] != 3 or t.dtype != np.int64
                or (t < 0).any() or (t >= n).any()):
            raise ValueError('int64 accepted triangle final indices required')
        if (np.any(np.diff(np.sort(t, axis=1), axis=1) == 0)
                or len(np.unique(np.sort(t, axis=1), axis=0)) != len(t)
                or (oid[t] != oid[t[:, :1]]).any()
                or (component[t] != component[t[:, :1]]).any()):
            raise ValueError('invalid duplicate/object/component triangle')
        source = packet['provenance']['source_index']; w = int(packet['image_shape'][1])
        uv = np.stack((source[t] % w, source[t] // w), axis=-1)
        d1, d2 = uv[:, 1]-uv[:, 0], uv[:, 2]-uv[:, 0]
        if (d1[:, 0]*d2[:, 1]-d1[:, 1]*d2[:, 0] == 0).any():
            raise ValueError('zero-area accepted triangle')
        if pairs != set(map(tuple, triangles_to_edges(t).T.tolist())):
            raise ValueError('edges must exactly equal accepted triangle side union')


def _validate_raw_ede_metadata(packet, n, oid, component, edge):
    """Structural trust boundary; deliberately no candidate regeneration."""
    pairs = np.asarray(packet.get('raw_candidate_pairs'))
    origins = packet.get('raw_candidate_origins')
    if (pairs.dtype != np.int64 or pairs.ndim != 2 or pairs.shape[1] != 2
            or (pairs < 0).any() or (pairs >= n).any()
            or (pairs[:, 0] >= pairs[:, 1]).any()
            or not np.array_equal(pairs, np.unique(pairs, axis=0))
            or not isinstance(origins, list) or len(origins) != len(pairs)
            or any(type(s) is not str or s not in
                   ('delaunay_edge', 'two_point_fallback', 'collinear_chain') for s in origins)):
        raise ValueError('canonical raw candidate rows/origins required')
    if ((oid[pairs[:, 0]] != oid[pairs[:, 1]]).any()
            or (component[pairs[:, 0]] != component[pairs[:, 1]]).any()):
        raise ValueError('raw candidate object/component mismatch')
    expected = np.unique(np.concatenate((pairs, pairs[:, ::-1])), axis=0).T
    if not np.array_equal(edge, expected):
        raise ValueError('raw directed edges must exactly equal ordered candidate union')
    source = packet['provenance']['source_index']
    for name, value in (('source_pairs', source[pairs]),
                        ('object_id', oid[pairs[:, 0]]),
                        ('component_id', component[pairs[:, 0]])):
        supplied = _array(packet['raw_candidate_'+name], np.int64, value.shape, name)
        if not np.array_equal(supplied, value):
            raise ValueError('raw candidate row metadata mismatch: '+name)


def _validate_edgewise_metadata(packet, n, oid, component, directed, extended):
    """Evaluation packet algebra/origin geometry; dense RGB-D is helper-owned."""
    from visual.runtime.extended_delaunay_edges import collinear_order, directed_edges, ORIGINS
    accepted = np.asarray(packet.get('accepted_edge_pairs'))
    origins = packet.get('accepted_edge_origins')
    if (accepted.dtype != np.int64 or accepted.ndim != 2 or accepted.shape[1] != 2
            or (accepted < 0).any() or (accepted >= n).any()
            or (accepted[:, 0] >= accepted[:, 1]).any()
            or not np.array_equal(accepted, np.unique(accepted, axis=0))
            or not isinstance(origins, list) or len(origins) != len(accepted)
            or any(type(s) is not str or s not in ORIGINS for s in origins)):
        raise ValueError('canonical accepted edge rows/origins required')
    if directed != set(map(tuple, directed_edges(accepted).T.tolist())):
        raise ValueError('edges must exactly equal accepted edge pair union')
    source = packet['provenance']['source_index']; w = int(packet['image_shape'][1])
    uv = np.column_stack((source % w, source//w))
    groups = {}
    for (a, b), origin in zip(accepted, origins):
        if oid[a] != oid[b] or component[a] != component[b]:
            raise ValueError('accepted edge object/component mismatch')
        key = (int(oid[a]), int(component[a]))
        if key not in groups:
            rows = np.flatnonzero((oid == key[0]) & (component == key[1]))
            order = collinear_order(uv[rows], source[rows], rows)
            chain = set() if order is None else {
                tuple(sorted((int(i), int(j)))) for i, j in zip(order[:-1], order[1:])}
            groups[key] = (len(rows), order is not None, chain)
        count, collinear, chain = groups[key]
        if origin == 'delaunay_edge':
            valid = count >= 3 and not collinear
        elif origin == 'two_point_fallback':
            valid = extended and count == 2
        else:
            valid = extended and count >= 3 and collinear and (int(a), int(b)) in chain
        if not valid:
            raise ValueError('candidate origin/whole-component geometry mismatch')


def _validate_v2_features(packet, n, source, oid, availability):
    """Measured fields and observation frame only; no encoder/imputation."""
    from visual.runtime.vision_packet import (
        FEATURE_FIELDS, DIAGNOSTIC_DTYPES, NUMERIC_CONTRACT,
        validate_transform, pose_matrix, float32_tolerance,
    )
    if packet.get('feature_mode') != 'continuous':
        raise ValueError('v2 supports continuous mode only')
    if packet['coordinate_frame'] != 'EE_metres' or packet['feature_fields'] != FEATURE_FIELDS:
        raise ValueError('v2 feature/frame schema mismatch')
    validity = _array(packet['feature_field_valid'], bool, (n, 3), 'feature_field_valid')
    valid = _array(packet['feature_valid'], bool, (n,), 'feature_valid')
    if (validity & ~availability).any() or not np.array_equal(valid, validity.all(1)):
        raise ValueError('feature validity mismatch')
    diag = packet['feature_diagnostics']
    for name, dtype in DIAGNOSTIC_DTYPES.items():
        _array(diag[name], dtype, (n,), name)
    if not diag['valid_points'].all():
        raise ValueError('unobserved point in v2')
    for j, name in enumerate(('normal_valid', 'normal_variation_valid', 'depth_pair_valid')):
        if not np.array_equal(validity[:, j], diag[name]):
            raise ValueError('diagnostic validity mismatch')
    if ((diag['normal_variation_valid'] & ~diag['normal_valid']).any()
            or not np.array_equal(diag['normal_variation_valid'],
                                  diag['normal_valid'] & (diag['normal_neighbor_count'] > 0))
            or (diag['normal_neighbor_count'] > 4).any()):
        raise ValueError('normal variation support mismatch')
    features = packet['geometric_features']
    for j, (a, b) in enumerate(((0, 3), (3, 4), (4, 5))):
        values = features[:, a:b]
        if not np.isfinite(values[validity[:, j]]).all() or not np.isnan(values[~validity[:, j]]).all():
            raise ValueError('invalid measured value/validity')
    normals = features[validity[:, 0], :3]
    if (np.abs(np.linalg.norm(normals, axis=1)-1) > NUMERIC_CONTRACT['normal_norm_atol']).any():
        raise ValueError('invalid measured unit normal')
    if ((features[validity[:, 1], 3] < 0).any()
            or (features[validity[:, 1], 3] > np.pi+1e-6).any()
            or (features[validity[:, 2], 4] < 0).any()):
        raise ValueError('invalid scalar measurement range')
    old = packet['provenance']
    if 'feature_available' in old or 'feature_field_available' in old:
        old_fields = packet['source_artifact_feature_fields']
        old_fa = _array(old['feature_field_available'], bool, (n, len(old_fields)), 'artifact availability')
        old_a = _array(old['feature_available'], bool, (n,), 'artifact availability')
        if not np.array_equal(old_a, old_fa.all(1)):
            raise ValueError('source artifact availability mismatch')
    image = _array(packet['image_shape'], np.int64, (2,), 'image_shape')
    if (image <= 0).any() or (source >= int(np.prod(image))).any():
        raise ValueError('source outside image')
    object_valid = _array(packet['valid'], bool, (2,), 'valid')
    if not object_valid[oid].all():
        raise ValueError('point belonging to unobserved object')
    camera = _array(packet['camera_points'], np.float32, (n, 3), 'camera_points')
    if not np.isfinite(camera).all() or (camera[:, 2] <= 0).any():
        raise ValueError('invalid camera points')
    transform = _array(packet['T_EE_from_camera'], np.float64, (4, 4), 'T_EE_from_camera')
    validate_transform(transform)
    c2w = _array(packet['camera_to_world'], np.float64, (4, 4), 'camera_to_world')
    validate_transform(c2w)
    pose = _array(packet['observation_ee_pose'], np.float64, (7,), 'observation_ee_pose')
    reference_transform = np.linalg.inv(pose_matrix(pose)) @ c2w
    eps = NUMERIC_CONTRACT['reference_eps_factor'] * np.finfo(np.float64).eps
    if not np.allclose(transform, reference_transform, rtol=eps, atol=eps):
        raise ValueError('source observation transform mismatch')
    k = _array(packet['camera_intrinsic'], np.float64, (3, 3), 'camera_intrinsic')
    if (not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or k[0, 1] != 0 or k[1, 0] != 0 or not np.array_equal(k[2], [0, 0, 1])):
        raise ValueError('invalid source camera intrinsic')
    reference = camera.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    if (np.abs(packet['points'].astype(np.float64)-reference) > float32_tolerance(reference)).any():
        raise ValueError('EE point transform exceeds fixed float32 tolerance')


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
