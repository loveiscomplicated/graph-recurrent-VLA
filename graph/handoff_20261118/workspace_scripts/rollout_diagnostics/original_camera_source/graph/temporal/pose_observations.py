"""Named joint-pose observations, using only current inference and causal TCP motion."""

import torch

from graph.pose_geometry import conjugate, quaternion_product, quaternion_rotvec, unit_quaternion

QUALITY_NAMES = tuple(f'{kind}_{stat}' for kind in ('peg', 'box', 'hole')
                      for stat in ('count', 'present', 'missing', 'confidence_fraction',
                                   'confidence_present', 'confidence_mean', 'confidence_min', 'confidence_max'))
BASE_NAMES = tuple(f'world_object21_{index}' for index in range(21)) + QUALITY_NAMES + (
    'all_vertices_missing', 'tcp_world_x', 'tcp_world_y', 'tcp_world_z',
    'tcp_delta_x', 'tcp_delta_y', 'tcp_delta_z', 'has_previous_frame')
ROBOT_NAMES = ('tcp_qw', 'tcp_qx', 'tcp_qy', 'tcp_qz',
               'tcp_delta_rot_x', 'tcp_delta_rot_y', 'tcp_delta_rot_z',
               'left_gripper_qpos', 'right_gripper_qpos')
POSE_CONTRACT = 'object21_world_vertex_quality_tcp_motion_wxyz_v1'
INSERTION_CONTRACT = 'object21_world_vertex_quality_tcp_motion_observation_age_v1'
FRESHNESS_NAMES = ('peg_unobserved_age_frames', 'target_unobserved_age_frames')


def observation_age_step(quality25, previous_age=None):
    """Count consecutive missing observations; no GT visibility or pose is used.

    Target support means BOX or HOLE vertices. An initially missing object has
    age one; a present object has age zero. These are observed-support ages,
    not a claim that an occluded object has a reliable remembered pose.
    """
    if quality25.ndim != 2 or quality25.shape[-1] != 25:
        raise ValueError('Expected quality25[B,25]')
    missing = torch.stack((quality25[:, 0] == 0,
                           quality25[:, 8] + quality25[:, 16] == 0), dim=-1)
    if previous_age is None:
        previous_age = quality25.new_zeros((len(quality25), 2))
    if previous_age.shape != missing.shape:
        raise ValueError('Previous observation age must have shape [B,2]')
    return torch.where(missing, previous_age + 1, 0.)


def pose_input_step(object21, quality25, tcp_world, previous_tcp=None,
                    tcp_quaternion=None, previous_quaternion=None, gripper_qpos=None):
    """Build one batched causal observation; zero motion at reset is a boundary value."""
    batch = object21.shape[0]
    if object21.shape != (batch, 21) or quality25.shape != (batch, 25) or tcp_world.shape != (batch, 3):
        raise ValueError('Expected object21[B,21], quality25[B,25], tcp_world[B,3]')
    world = torch.cat((object21[:, :3] + tcp_world, unit_quaternion(object21[:, 3:7]),
                       object21[:, 7:10], object21[:, 10:13] + tcp_world,
                       unit_quaternion(object21[:, 13:17]), object21[:, 17:21]), dim=-1)
    delta = torch.zeros_like(tcp_world) if previous_tcp is None else tcp_world - previous_tcp
    initialized = object21.new_full((batch, 1), float(previous_tcp is not None))
    features = [world, quality25, tcp_world, delta, initialized]
    if (tcp_quaternion is None) != (gripper_qpos is None):
        raise ValueError('Supply observed TCP quaternion and gripper qpos together, or neither')
    if tcp_quaternion is not None:
        if tcp_quaternion.shape != (batch, 4) or gripper_qpos.shape != (batch, 2):
            raise ValueError('Expected tcp_quaternion[B,4], gripper_qpos[B,2]')
        current = unit_quaternion(tcp_quaternion)
        if previous_tcp is not None and previous_quaternion is None:
            raise ValueError('Previous observed TCP quaternion is required after initialization')
        rotation = (torch.zeros_like(tcp_world) if previous_quaternion is None else
                    quaternion_rotvec(quaternion_product(current, conjugate(unit_quaternion(previous_quaternion)))))
        features.extend((current, rotation, gripper_qpos))
    result = torch.cat(features, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError('Temporal pose observations must be finite')
    return result
