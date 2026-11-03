"""Versioned 3D surface graph. Never masquerades as a Raw EDE packet."""
from copy import deepcopy
import numpy as np
import torch
from scipy.spatial import cKDTree

from graph.integration_baseline.graph import (Graph, nearest_robot_pairs,
    global_peg_box_pairs, bidirectional, edge_attributes)
from visual.runtime.vision_packet import pose_matrix
from graph.integration_baseline.robot import geometry_in_ee

from . import SCHEMA_ID, EDGE_POLICY
from .contracts import transform
from .coordinates import record_xyz, record_normal, validate_record_frames
from visual.runtime.coordinate_frames import check_frame, camera_from_frame, FRAME_NAMES
from .map import CURRENTLY_OBSERVED, PERSISTENT_ESTIMATED, LOW_CONFIDENCE, INVALID_REMOVED

EXTRA_FIELDS = ('currently_observed', 'persistent_estimated', 'low_confidence',
                'confidence', 'age_seconds', 'uncertainty_m', 'pose_confidence')
FEATURE_DIMENSION = 14 + len(EXTRA_FIELDS)


def farthest_rows(xyz, ids, count):
    """Deterministic spatial budget; independent of targets, task and node order."""
    xyz = np.asarray(xyz, float)
    if not isinstance(count,(int,np.integer)) or count<0:raise ValueError('nonnegative point count required')
    if count==0:return np.empty(0,np.int64)
    order = np.argsort(np.asarray(ids).astype(str),kind='stable')
    if len(xyz) <= count:
        return order
    selected = [int(order[0])]
    distance = np.full(len(xyz), np.inf)
    for _ in range(1, count):
        distance = np.minimum(distance, np.sum((xyz-xyz[selected[-1]])**2, axis=1))
        distance[selected] = -1
        best = distance.max()
        selected.append(int(next(i for i in order if distance[i] == best)))
    return np.asarray(selected)


def free_space_violation(world_samples, cameras, tolerance=.006, *, coordinate_frame='world'):
    bad = np.zeros(len(world_samples), bool)
    for camera in cameras:
        p = transform(world_samples, camera_from_frame(camera, coordinate_frame))
        k = camera.intrinsic
        uv = np.floor(p[:, :2]/np.maximum(p[:, 2, None], 1e-12)*np.array([k[0, 0], k[1, 1]])+k[:2, 2]).astype(int)
        h, w = camera.depth.shape
        inside = (p[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        rows = np.flatnonzero(inside)
        u, v = uv[rows].T
        z = camera.depth[v, u]
        bad[rows] |= np.isfinite(z) & (p[rows, 2] < z-tolerance)
    return bad


def surface_edges(xyz, normals, normal_valid, object_id, reference, cameras=(), *, k=8, radius=.02,
                  coordinate_frame='world'):
    """Distance/normal/tangency gates plus intermediate surface witnesses.

    This conservative policy can disconnect creases. Its audits report that
    cost; component count is never interpreted as true object connectivity.
    """
    pairs = set()
    visited=set()
    audit = dict(candidates=0, rejected_normal=0, rejected_witness=0, rejected_free_space=0)
    for oid in (0, 1):
        rows = np.flatnonzero(object_id == oid)
        if len(rows) < 2:
            continue
        tree = cKDTree(xyz[rows])
        distances, near = tree.query(xyz[rows], k=min(k+1, len(rows)))
        reference_tree = cKDTree(reference[oid])
        for ii, indices in enumerate(near):
            for jj in np.atleast_1d(indices):
                a, b = sorted((int(rows[ii]), int(rows[jj])))
                if a == b or (a, b) in visited:
                    continue
                visited.add((a,b))
                delta = xyz[b]-xyz[a]
                length = np.linalg.norm(delta)
                if length > radius or length < 1e-9:
                    continue
                audit['candidates'] += 1
                if normal_valid[a] and normal_valid[b]:
                    if (normals[a] @ normals[b] < .5
                            or max(abs(normals[a] @ delta), abs(normals[b] @ delta)) > .002+length*.25):
                        audit['rejected_normal'] += 1
                        continue
                samples = xyz[a] + np.array([.25, .5, .75])[:, None]*delta
                if (reference_tree.query(samples)[0] > .004).any():
                    audit['rejected_witness'] += 1
                    continue
                if free_space_violation(samples, cameras, coordinate_frame=coordinate_frame).any():
                    audit['rejected_free_space'] += 1
                    continue
                pairs.add((a, b))
    return bidirectional(np.asarray(sorted(pairs), np.int64).reshape(-1, 2)), audit


def build_graph(records, robot_world, tcp_pose, timestep, *, cameras=(), point_cap=128,
                include_observation_status=True, coordinate_frame=None):
    if coordinate_frame is not None:
        check_frame(coordinate_frame)
        if any(not np.array_equal(c.tcp_pose,tcp_pose) for c in cameras):
            raise ValueError('Graph/camera EE reference mismatch')
    elif any('coordinate_frame' in r for r in records):
        raise ValueError('frame-aware records require an explicit coordinate_frame')
    validate_record_frames(records, coordinate_frame or 'world', tcp_pose)
    records = [r for r in records if r['status'] != INVALID_REMOVED]
    selected = []
    for oid in (0, 1):
        rows = [r for r in records if r['object_id'] == oid]
        if rows:
            cap=point_cap[oid] if isinstance(point_cap,dict) else point_cap
            chosen = farthest_rows([record_xyz(r) for r in rows], [r['persistent_node_id'] for r in rows],
                                   len(rows) if cap is None else cap)
            selected.extend(rows[i] for i in chosen)
    object_id = np.array([r['object_id'] for r in selected], np.int64)
    world = np.array([record_xyz(r) for r in selected]).reshape(-1, 3)
    inverse_ee = np.linalg.inv(pose_matrix(tcp_pose)) if coordinate_frame is None else None
    xyz = transform(world, inverse_ee) if coordinate_frame is None else world.copy()
    if coordinate_frame == 'world':
        robots = np.asarray(robot_world)
        if robots.shape != (3, 3) or not np.isfinite(robots).all() or not np.allclose(robots[0], pose_matrix(tcp_pose)[:3, 3], rtol=0, atol=1e-7):
            raise ValueError('same-time World robot geometry mismatch')
        robots = robots.astype(np.float32)
    else:
        robots = geometry_in_ee(robot_world,tcp_pose)
    normals_world = np.array([record_normal(r) for r in selected]).reshape(-1, 3)
    normal_valid = np.array([r['normal_valid'] for r in selected], bool)
    features = np.zeros((len(selected), 5), np.float32)
    available = np.zeros((len(selected), 3), bool)
    valid = np.zeros_like(available)
    extras = np.zeros((len(selected), len(EXTRA_FIELDS)), np.float32)
    for i, row in enumerate(selected):
        current = row['currently_observed']
        # Historical camera-z jump and local patch variation are unavailable
        # now. Only a rotated measured surface normal can persist.
        available[i] = [True, current, current]
        measured_normal=current and row['last_feature_valid'][0]
        historical_normal=(row['normal_valid'] and row['position_uncertainty']<=.02 and row['pose_confidence']>=.15)
        valid[i] = [measured_normal or historical_normal,
                    current and row['last_feature_valid'][1],
                    current and row['last_feature_valid'][2]]
        normal=row['last_f5'][:3] if measured_normal else normals_world[i]
        features[i, :3] = normal @ inverse_ee[:3, :3].T if coordinate_frame is None else normal
        if current:
            features[i, 3:] = np.where(valid[i, 1:], row['last_f5'][3:], 0)
        features[i, :3] *= valid[i, 0]
        extras[i, :3] = np.eye(3)[row['status']]
        extras[i, 3:] = [row['confidence'], (timestep-row['last_seen_timestep'])/20,
                          row['position_uncertainty'], row['pose_confidence']]
    if not include_observation_status:
        extras[:] = 0
    n = len(selected)+3
    inputs = np.zeros((n, FEATURE_DIMENSION), np.float32)
    positions = np.vstack((robots, xyz)).astype(np.float32)
    inputs[:, :3] = positions
    inputs[3:, 3:14] = np.column_stack((features, available, valid))
    inputs[3:, 14:] = extras
    reference = {oid: np.array([record_xyz(r) for r in records if r['object_id'] == oid]).reshape(-1, 3) for oid in (0, 1)}
    intra, audit = surface_edges(world, normals_world, normal_valid, object_id, reference, cameras,
                                 coordinate_frame=coordinate_frame or 'world')
    ids = np.arange(len(selected), dtype=np.int64)
    ro = bidirectional(nearest_robot_pairs(positions, object_id, ids, 4))
    pb = bidirectional(global_peg_box_pairs(xyz.astype(np.float32), object_id, ids, 16)+3)
    edges = np.concatenate((intra+3, ro, pb), axis=1)
    types = np.repeat(np.arange(3, dtype=np.int64), [intra.shape[1], ro.shape[1], pb.shape[1]])
    metadata = dict(schema_id=SCHEMA_ID, edge_policy=EDGE_POLICY, feature_fields=list(EXTRA_FIELDS),
                    node_ids=[r['persistent_node_id'] for r in selected], surface_edge_audit=audit,
                    timestep=timestep, point_cap_per_object=point_cap,
                    point_budget_policy='stable_identity_seeded_farthest_v1',
                    status_features=include_observation_status,
                    coordinate_frame=FRAME_NAMES[coordinate_frame or 'ee'],
                    normal_frame=coordinate_frame or 'ee')
    if coordinate_frame == 'ee':
        metadata['reference_ee_pose'] = np.asarray(tcp_pose, float).tolist()
    if not np.isfinite(inputs).all():
        raise ValueError('nonfinite network features')
    return Graph(torch.from_numpy(inputs), torch.from_numpy(np.r_[np.arange(3), object_id+3]),
                 torch.from_numpy(edges), torch.from_numpy(edge_attributes(positions, edges)),
                 torch.from_numpy(types), metadata, {})


def common_baseline(original, *, point_cap=128, include_observation_status=True):
    """Keep original G0 XYZ/F5 and induced Raw EDE; only control graph budget.

    The full original packet and native graph are stored separately. k4/M16
    interactions are recomputed at the selected budget using existing helpers.
    """
    graph = deepcopy(original)
    chosen = list(range(3))
    for oid in (0, 1):
        rows = np.flatnonzero(graph.node_type.numpy() == oid+3)
        if len(rows):
            cap=point_cap[oid] if isinstance(point_cap,dict) else point_cap
            chosen.extend(rows[farthest_rows(graph.node_input[rows, :3].numpy(), rows,
                len(rows) if cap is None else cap)])
    chosen = np.asarray(chosen, np.int64)
    remap = np.full(len(graph.node_type), -1, np.int64)
    remap[chosen] = np.arange(len(chosen))
    old = graph.edge_index[:, graph.edge_type == 0].numpy()
    intra = remap[old[:, (remap[old] >= 0).all(0)]]
    inputs = np.zeros((len(chosen), FEATURE_DIMENSION), np.float32)
    inputs[:, :14] = graph.node_input[chosen].numpy()
    if include_observation_status:
        inputs[3:, 14:] = [1, 0, 0, 1, 0, 0, 1]
    node_type = graph.node_type[chosen].numpy()
    object_id = node_type[3:]-3
    ids = chosen[3:]-3
    ro = bidirectional(nearest_robot_pairs(inputs[:, :3], object_id, ids, 4))
    pb = bidirectional(global_peg_box_pairs(inputs[3:, :3], object_id, ids, 16)+3)
    edges = np.concatenate((intra, ro, pb), axis=1)
    types = np.repeat(np.arange(3, dtype=np.int64), [intra.shape[1], ro.shape[1], pb.shape[1]])
    return Graph(torch.from_numpy(inputs), torch.from_numpy(node_type), torch.from_numpy(edges),
        torch.from_numpy(edge_attributes(inputs[:, :3], edges)), torch.from_numpy(types),
        dict(schema_id=SCHEMA_ID, edge_policy=original.packet['edge_policy'],
             origin='unchanged G0 native graph with explicitly induced budget subset',
             point_budget_policy='stable_identity_seeded_farthest_v1',
             selected_original_rows=chosen.tolist(), status_features=include_observation_status), {})
