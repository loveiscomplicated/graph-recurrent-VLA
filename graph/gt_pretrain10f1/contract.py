"""GT38 units/order and sensor-only state assembly. No simulator GT imports.

The object head estimates 18 independent values. The three head-to-hole values
are derived, never a separate target. Unique peg orientation is not identifiable
under all geometric symmetries; this API produces an explicitly estimated state,
not an assertion that all 38 fields are observable in every frame.
"""
import numpy as np
import torch

from recurrent.features import FEATURE_LAYOUT, canon_quat, quat_rotate

OBJECT_FIELDS = (('peg_pos_rel_tcp', 3), ('peg_quat', 4),
                 ('hole_pos_rel_tcp', 3), ('hole_quat', 4),
                 ('peg_half_sizes', 3), ('hole_radius', 1))
OBJECT_DIM = sum(n for _, n in OBJECT_FIELDS)
OBJECT_SLICES = {}
_offset = 0
for _name, _size in OBJECT_FIELDS:
    OBJECT_SLICES[_name] = slice(_offset, _offset + _size)
    _offset += _size
ROBOT_KEYS = frozenset(('tcp_pose', 'left_tip', 'right_tip', 'gripper'))


def measured_robot(robot76, schema):
    """Reconstruct measured tips from finger links and installed pad offsets."""
    robot76 = np.asarray(robot76, np.float32)
    if robot76.shape[-1] != 76 or schema['dimension'] != 76 or not np.isfinite(robot76).all():
        raise ValueError('finite Robot76 required')
    fields = {f['name']: robot76[..., f['offset']:f['offset'] + f['dimension']]
              for f in schema['fields']}
    out = dict(tcp_pose=fields['tcp_pose'].copy(), gripper=fields['gripper_position'].copy())
    for side, offset in (('left', [0, .00758, .04525]), ('right', [0, -.00758, .04525])):
        pose = fields[side + '_finger_pose']
        out[side + '_tip'] = pose[..., :3] + quat_rotate(pose[..., 3:], np.broadcast_to(
            np.asarray(offset, np.float32), pose[..., :3].shape))
    return out


def object_target(raw):
    """Supervision branch only: independent GT objects in world-aligned axes."""
    tcp = raw['tcp_pose'][..., :3]
    return np.concatenate((raw['peg_pose'][..., :3]-tcp,
                           canon_quat(raw['peg_pose'][..., 3:]),
                           raw['hole_pose'][..., :3]-tcp,
                           canon_quat(raw['hole_pose'][..., 3:]),
                           raw['peg_half'], raw['hole_radius']), -1).astype(np.float32)


def split_fields(objects):
    if objects.shape[-1] != OBJECT_DIM:
        raise ValueError('object target dimension/order')
    return {k: objects[..., s] for k, s in OBJECT_SLICES.items()}


def assemble_gt38(predicted_fields, measured_robot_obs):
    """Raw GT38-compatible order; only whitelisted estimates and robot sensors.

    No freshness, task dimensions, object pose, or sidecar is accepted as an
    auxiliary input. Clean freshness is explicitly fixed by this contract.
    Callers must retain the observability audit when using the estimated values.
    """
    if set(predicted_fields) != set(OBJECT_SLICES) or set(measured_robot_obs) != ROBOT_KEYS:
        raise ValueError('state assembly input whitelist')
    o = {k: np.asarray(v, np.float32) for k, v in predicted_fields.items()}
    r = {k: np.asarray(v, np.float32) for k, v in measured_robot_obs.items()}
    shape = r['tcp_pose'].shape[:-1]
    for k, n in OBJECT_FIELDS:
        if o[k].shape != (*shape, n) or not np.isfinite(o[k]).all():
            raise ValueError('nonfinite or malformed predicted ' + k)
    for k, n in [('tcp_pose', 7), ('left_tip', 3), ('right_tip', 3), ('gripper', 2)]:
        if r[k].shape != (*shape, n) or not np.isfinite(r[k]).all():
            raise ValueError('nonfinite or malformed measured ' + k)
    for k in ('peg_quat', 'hole_quat'):
        if not np.allclose(np.linalg.norm(o[k], axis=-1), 1, atol=2e-5, rtol=0):
            raise ValueError('unit wxyz estimated quaternion required')
    tcp = r['tcp_pose'][..., :3]
    head_local = np.concatenate((o['peg_half_sizes'][..., :1], np.zeros((*shape, 2), np.float32)), -1)
    # Same arithmetic/order as official encoder, including float32 rounding.
    peg_world = o['peg_pos_rel_tcp'] + tcp
    hole_world = o['hole_pos_rel_tcp'] + tcp
    head = peg_world + quat_rotate(o['peg_quat'], head_local)
    return np.concatenate((tcp, canon_quat(r['tcp_pose'][..., 3:]),
        r['left_tip']-tcp, r['right_tip']-tcp, r['gripper'],
        o['peg_pos_rel_tcp'], canon_quat(o['peg_quat']), head-hole_world,
        o['hole_pos_rel_tcp'], canon_quat(o['hole_quat']), o['peg_half_sizes'],
        o['hole_radius'], np.ones((*shape, 1), np.float32), np.zeros((*shape, 1), np.float32)), -1).astype(np.float32)


def normalize_gt38(state38, stats):
    """Official normalizer, distinct from pretraining target scales."""
    return ((state38-np.asarray(stats['obs_mean'], np.float32)) /
            np.asarray(stats['obs_std'], np.float32) *
            np.asarray(stats['obs_keep'], np.float32)).astype(np.float32)


def feature_contract():
    rows = []
    index = 0
    for name, size in FEATURE_LAYOUT:
        robot = index < 15
        freshness = index >= 36
        quat = 'quat' in name
        uncertain = name in ('peg_quat', 'hole_quat', 'peg_half_sizes', 'hole_radius', 'peg_head_rel_hole')
        source = ('Robot76 / measured FK' if robot else 'Clean observation contract' if freshness else
                  'derived from predicted peg/hole/size' if name == 'peg_head_rel_hole' else 'GNN State Head estimate')
        for j in range(size):
            rows.append(dict(index=index+j, feature_name=f'{name}[{j}]', group=name,
                unit='unitless' if quat or freshness else 'm',
                coordinate_frame=('world wxyz' if quat else 'finger joint' if name == 'gripper_qpos' else
                    'peg local half extent' if name == 'peg_half_sizes' else 'world-aligned displacement' if 'rel' in name else
                    'world' if name == 'tcp_pos' else 'scalar'),
                gt_source=('recurrent.common.raw_gt -> get_gt_state' if not freshness else 'recurrent.corruption Clean'),
                measured_sensor=robot, gnn_prediction_target=name in OBJECT_SLICES,
                classification='A' if robot else 'clean_contract' if freshness else 'C' if uncertain else 'B',
                observability_uncertain=uncertain,
                observability_status=('CONDITIONAL_ESTIMATE_NOT_UNIQUELY_OBSERVABLE' if uncertain else
                                      'MEASURED' if robot else 'CONTRACT' if freshness else 'PARTIAL_VISIBLE_ESTIMATE'),
                unsupported_cases=('missing geometry; peg rotational symmetries; unobserved rim/extent' if uncertain else
                                   'missing object geometry' if not robot and not freshness else None),
                normalization='((x-obs_mean)/obs_std)*obs_keep; checkpoint-specific; target scales separate',
                quaternion_rule='wxyz, source unit quaternion, canonical w>=0; q and -q equivalent' if quat else None,
                inference_source=source))
        index += size
    assert index == 38 and OBJECT_DIM == 18
    return dict(dimension=38, independent_prediction_dimension=18, dimensions=rows,
        relative_coordinate_rule='subtraction of world positions; NOT inverse TCP rotation',
        peg_head_rule='peg_p + R(peg_q)*[peg_half_x,0,0] - hole_p, world displacement',
        derived_field_not_independent_target='peg_head_rel_hole', clean_freshness=[1, 0],
        no_gt_inference=True, exact_full_observability=False,
        scope='estimated compatible values; unsupported/ambiguous cases remain explicitly documented')


def quaternion_geodesic(q, target):
    dot = (q * target).sum(-1).abs().clamp(0, 1)
    return 2 * torch.atan2(torch.sqrt((1-dot.square()).clamp_min(0)), dot)
