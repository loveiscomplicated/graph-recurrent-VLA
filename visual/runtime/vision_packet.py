"""Opt-in measured continuous Vision packet, using existing dense PCA5 evidence.

No estimator, sampling, semantic classifier, topology or GNN encoder lives here.
All source rows and generation provenance are preserved; invalid measurements
remain NaN. The old artifact's availability stays in its provenance namespace.
"""
from __future__ import annotations

import copy
from time import perf_counter
import numpy as np


# Fixed before Task04 data validation. Calibration matrices were stored float32:
# their measured orthogonality defect is reported separately from arithmetic.
NUMERIC_CONTRACT = dict(
    reference_eps_factor=64, distance_eps_factor=128,
    float32_point_abs_floor_m=1e-6, float32_ulp_factor=4,
    normal_abs_floor=1e-7, normal_norm_atol=1e-5,
    calibration_rotation_atol=1e-5, quaternion_norm_atol=1e-5,
    cache_normal_atol=1e-6, cache_variation_atol_rad=1e-6,
    cache_depth_atol_m=1e-6,
)
FEATURE_FIELDS = [
    dict(name='normals', shape=[3], dtype=np.dtype('float32').str,
         frame='EE', unit='unit_vector'),
    dict(name='normal_variation_rad', shape=[], dtype=np.dtype('float32').str,
         frame='rotation_invariant', unit='radian'),
    dict(name='depth_jump_m', shape=[], dtype=np.dtype('float32').str,
         frame='CV_camera_z', unit='metre'),
]
DIAGNOSTIC_DTYPES = {
    'valid_points': bool, 'normal_valid': bool,
    'normal_variation_valid': bool, 'depth_pair_valid': bool,
    'depth_discontinuity': bool, 'mask_boundary': bool,
    'depth_validity_boundary': bool, 'image_boundary': bool,
    'normal_neighbor_count': np.uint8,
}


def float32_tolerance(reference, *, normal=False):
    """Componentwise max(fixed floor, 4 ULP) at the stored float32 magnitude."""
    a = np.asarray(reference, np.float64).astype(np.float32)
    ulp = np.abs(np.spacing(np.abs(a)).astype(np.float64))
    floor = NUMERIC_CONTRACT['normal_abs_floor' if normal else 'float32_point_abs_floor_m']
    return np.maximum(floor, NUMERIC_CONTRACT['float32_ulp_factor'] * ulp)


def validate_transform(transform):
    t = np.asarray(transform)
    if t.shape != (4, 4) or not np.issubdtype(t.dtype, np.floating) or not np.isfinite(t).all():
        raise ValueError('finite floating [4,4] transform required')
    if not np.array_equal(t[3], [0, 0, 0, 1]):
        raise ValueError('invalid homogeneous row')
    r = t[:3, :3].astype(np.float64)
    atol = NUMERIC_CONTRACT['calibration_rotation_atol']
    if (np.max(np.abs(r.T @ r - np.eye(3))) > atol
            or abs(np.linalg.det(r) - 1) > atol):
        raise ValueError('calibration rotation is not rigid within recorded precision')
    return t.astype(np.float64)


def pose_matrix(pose):
    """World-from-EE pose [x,y,z,qw,qx,qy,qz], with no identity fallback."""
    p = np.asarray(pose, np.float64)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError('finite source observation TCP pose [7] required')
    norm = np.linalg.norm(p[3:])
    if abs(norm - 1) > NUMERIC_CONTRACT['quaternion_norm_atol']:
        raise ValueError('TCP wxyz quaternion is not unit')
    w, x, y, z = p[3:] / norm
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = p[:3]
    return t


def camera_to_ee(frame):
    return validate_transform(np.linalg.inv(pose_matrix(frame.tcp_pose))
                              @ validate_transform(frame.cam2world))


def check_source_alignment(source_packet, evidence, frame):
    """Check identity, raw masks/support, source bounds and original camera XYZ."""
    from graph.vision_input import validate_packet
    validate_packet(source_packet)
    if source_packet.get('schema_version', 1) != 1:
        raise ValueError('expected camera-frame v1 selection packet')
    if (source_packet['coordinate_frame'] != 'CV_camera_metres'
            or evidence.estimator != 'pca5'):
        raise ValueError('camera selection and existing pca5 evidence required')
    for field in ('source_frame', 'camera'):
        if source_packet[field] != getattr(frame, field) or getattr(evidence.frame, field) != getattr(frame, field):
            raise ValueError('source observation identity mismatch: ' + field)
    for field in ('frame_idx', 'timestamp'):
        if getattr(evidence.frame, field) != getattr(frame, field):
            raise ValueError('source observation timing mismatch: ' + field)
    g = evidence.frame.geometry
    if (g.valid_depth.shape != frame.depth.shape
            or not np.array_equal(g.valid_depth, np.isfinite(frame.depth) & (frame.depth > 0))):
        raise ValueError('source depth validity mismatch')
    for oid in (0, 1):
        if not np.array_equal(g.object_masks[oid], frame.masks[oid]):
            raise ValueError('source object mask mismatch')
        if not np.array_equal(evidence.frame.objects[oid].valid_points, g.observed_mask(oid)):
            raise ValueError('source observed support mismatch')
    if not np.array_equal(g.overlap_mask, frame.masks[0] & frame.masks[1]):
        raise ValueError('source overlap mismatch')
    source = source_packet['provenance']['source_index']
    if (source >= g.valid_depth.size).any():
        raise ValueError('source outside original raster')
    for oid in (0, 1):
        s = source[source_packet['object_id'] == oid]
        if not g.observed_mask(oid).ravel()[s].all():
            raise ValueError('source is not observed for this object')
    xyz = g.xyz.reshape(-1, 3)[source]
    if not np.array_equal(xyz, source_packet['points']):
        raise ValueError('selected original camera XYZ mismatch')
    # Independent source depth/K check at selected pixels (not a second estimator).
    v, u = np.divmod(source, frame.depth.shape[1])
    z = frame.depth[v, u]
    k = np.asarray(frame.intrinsic, np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.array_equal(k[2], [0, 0, 1])
            or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError('invalid source CV calibration')
    original = np.column_stack(((u+.5-k[0, 2])*z/k[0, 0],
                                (v+.5-k[1, 2])*z/k[1, 1], z)).astype(np.float32)
    if not np.array_equal(original, xyz):
        raise ValueError('source calibration/depth/XYZ mismatch')
    return xyz


def complete_vision_packet(source_packet, evidence, frame, *, mode='continuous', timings=None):
    """Gather existing dense measurements into an independent v2 packet.

    Edges are explicitly not implemented at this Feature stage. A separate,
    subsequently validated Vision topology helper supplies them later.
    """
    if mode != 'continuous':
        raise NotImplementedError('unsupported geometric feature mode: ' + str(mode))
    start = perf_counter()
    xyz = check_source_alignment(source_packet, evidence, frame)
    after_alignment = perf_counter()
    n = len(xyz)
    source = source_packet['provenance']['source_index']
    oid = source_packet['object_id']
    diagnostics = {k: np.empty(n, dtype) for k, dtype in DIAGNOSTIC_DTYPES.items()}
    values = {f['name']: np.full((n, *f['shape']), np.nan, np.float32) for f in FEATURE_FIELDS}
    for obj in (0, 1):
        rows = np.flatnonzero(oid == obj)
        s = source[rows]
        dense = evidence.frame.objects[obj]
        for name in diagnostics:
            diagnostics[name][rows] = getattr(dense, name).ravel()[s]
        for name in values:
            a = getattr(dense, name)
            values[name][rows] = a.reshape((-1, 3) if name == 'normals' else (-1,))[s]
    validity = np.column_stack([diagnostics[k] for k in
                               ('normal_valid', 'normal_variation_valid', 'depth_pair_valid')])
    # Do not silently repair an evidence/validity contradiction.
    for j, field in enumerate(FEATURE_FIELDS):
        v = values[field['name']].reshape(n, 3 if j == 0 else 1)
        if not np.isfinite(v[validity[:, j]]).all() or not np.isnan(v[~validity[:, j]]).all():
            raise ValueError('dense measurement validity/value mismatch: ' + field['name'])
    after_gather = perf_counter()
    transform = camera_to_ee(frame)
    ee64 = xyz.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    nv = diagnostics['normal_valid']
    values['normals'][nv] = (values['normals'][nv].astype(np.float64) @ transform[:3, :3].T).astype(np.float32)
    points = ee64.astype(np.float32)
    after_transform = perf_counter()
    availability = np.ones((n, 3), bool)
    compact = {f['name']: dict(positions=np.arange(n, dtype=np.int64), values=values[f['name']].copy())
               for f in FEATURE_FIELDS}
    packet = dict(
        schema_version=2, feature_mode=mode, episode=source_packet['episode'],
        camera=frame.camera, source_frame=int(frame.source_frame), frame_idx=frame.frame_idx,
        timestamp=frame.timestamp, coordinate_frame='EE_metres',
        points=points, camera_points=xyz.copy(), object_id=oid.copy(),
        geometric_features=np.column_stack([values['normals'], values['normal_variation_rad'],
                                            values['depth_jump_m']]).astype(np.float64),
        feature_fields=copy.deepcopy(FEATURE_FIELDS), compact_features=compact,
        feature_available=availability.all(1), feature_field_available=availability,
        feature_valid=validity.all(1), feature_field_valid=validity,
        feature_diagnostics=diagnostics, provenance=copy.deepcopy(source_packet['provenance']),
        source_artifact_feature_fields=copy.deepcopy(source_packet['feature_fields']),
        T_EE_from_camera=transform, observation_ee_pose=np.asarray(frame.tcp_pose, np.float64).copy(),
        camera_to_world=np.asarray(frame.cam2world, np.float64).copy(),
        camera_intrinsic=np.asarray(frame.intrinsic, np.float64).copy(),
        image_shape=np.array(frame.depth.shape, np.int64),
        valid=np.array([evidence.frame.geometry.observed_mask(i).any() for i in (0, 1)], bool),
        edge_status='not_implemented', intra_object_edge_index=np.empty((2, 0), np.int64),
    )
    if timings is not None:
        timings.update(source_alignment_ms=(after_alignment-start)*1000,
                       feature_gather_ms=(after_gather-after_alignment)*1000,
                       ee_transform_ms=(after_transform-after_gather)*1000,
                       packet_packaging_ms=(perf_counter()-after_transform)*1000)
    return packet
