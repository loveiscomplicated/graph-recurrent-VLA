"""Robot-only current-observation schema, shared by recording and inference.

No task GT accessor is called: in particular no actor/hole/goal/contact inputs.
The two pad offsets are the existing raw accessor's collision-centre definition.
"""
from dataclasses import dataclass

import numpy as np
import torch

from envs.maniskill.gt_state import _finger_pad_origin, _LEFT_TIP_LOCAL, _RIGHT_TIP_LOCAL
from graph.prototype.v2.state import StateNormalizer
from visual.runtime.vision_packet import pose_matrix


SCHEMA_ID = 'panda_observable_robot_state_v1'


def _np(tensor):
    return tensor.detach().cpu().numpy()[0].astype(np.float32, copy=True)


def robot_observation(env):
    """Read only agent kinematics available identically during inference."""
    agent = env.unwrapped.agent
    proprio = agent.get_proprioception()
    if set(proprio) != {'qpos', 'qvel'} or agent.get_controller_state():
        raise ValueError('proprioception schema changed; explicitly audit new fields')
    qpos, qvel = _np(proprio['qpos']), _np(proprio['qvel'])
    fields = dict(qpos=qpos, qvel=qvel,
                  gripper_position=qpos[-2:].copy(), gripper_velocity=qvel[-2:].copy(),
                  gripper_opening=np.array([qpos[-2:].sum()], np.float32),
                  gripper_opening_velocity=np.array([qvel[-2:].sum()], np.float32))
    for name, link in [('tcp', agent.tcp), ('left_finger', agent.finger1_link),
                       ('right_finger', agent.finger2_link), ('robot_base', agent.robot.get_root())]:
        fields[name + '_pose'] = _np(link.pose.raw_pose)
        fields[name + '_linear_velocity'] = _np(link.linear_velocity)
        fields[name + '_angular_velocity'] = _np(link.angular_velocity)
    pads = np.stack((_np(agent.tcp.pose.p),
                     _np(_finger_pad_origin(agent.finger1_link, _LEFT_TIP_LOCAL)),
                     _np(_finger_pad_origin(agent.finger2_link, _RIGHT_TIP_LOCAL))))
    return fields, pads


def state_schema(fields):
    result, start = [], 0
    for name, value in fields.items():
        size = value.size
        if name in ('qpos', 'qvel'):
            units = (['rad'] * (size - 2) + ['m'] * 2 if name == 'qpos'
                     else ['rad/s'] * (size - 2) + ['m/s'] * 2)
        elif name.endswith('_pose'):
            units = ['m'] * 3 + ['unitless_wxyz'] * 4
        elif 'angular_velocity' in name:
            units = ['rad/s'] * size
        else:
            units = [('m/s' if 'velocity' in name else 'm')] * size
        result.append(dict(name=name, offset=start, dimension=size, units=units,
                           frame='joint' if name in ('qpos', 'qvel') or name.startswith('gripper') else 'world',
                           source=('agent.get_proprioception()' if name in ('qpos', 'qvel') else
                                   'same-time robot link state / gripper joint observation')))
        start += size
    return dict(id=SCHEMA_ID, dimension=start, fields=result, dtype='float32',
                pose_order='xyz,qw,qx,qy,qz', controller_state={},
                duplicates='gripper fields repeat final two joints; TCP/finger positions also appear in graph',
                scope='all get_proprioception fields + current TCP/finger/base kinematics; no task state')


def flatten_state(fields, schema):
    if schema['id'] != SCHEMA_ID or list(fields) != [f['name'] for f in schema['fields']]:
        raise ValueError('robot state field/order mismatch')
    parts = []
    for spec in schema['fields']:
        part = np.asarray(fields[spec['name']], np.float32).reshape(-1)
        if part.size != spec['dimension'] or not np.isfinite(part).all():
            raise ValueError('invalid robot state field')
        parts.append(part)
    result = np.concatenate(parts)
    if result.shape != (schema['dimension'],):
        raise ValueError('robot schema dimension mismatch')
    return result


def geometry_in_ee(world_positions, tcp_pose):
    world_positions = np.asarray(world_positions)
    if world_positions.shape != (3, 3) or not np.isfinite(world_positions).all():
        raise ValueError('same-time TCP and two real pad centres required')
    matrix = pose_matrix(tcp_pose)
    if not np.allclose(world_positions[0], matrix[:3, 3], rtol=0, atol=1e-7):
        raise ValueError('TCP geometry/transform mismatch')
    return ((world_positions.astype(np.float64) - matrix[:3, 3]) @ matrix[:3, :3]).astype(np.float32)


@dataclass
class RobotNormalizer:
    """Reuse v2 StateNormalizer statistics/storage, with schema-sized features."""
    schema: dict
    stats: StateNormalizer
    fitted_episode_ids: list

    @classmethod
    def fit(cls, rows, episode_ids, splits, schema):
        ids = set(map(int, episode_ids))
        if not ids or not ids.issubset(set(splits['train'])):
            raise ValueError('normalizer fit accepts train episodes only')
        x = torch.as_tensor(np.asarray(rows), dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != schema['dimension'] or not torch.isfinite(x).all():
            raise ValueError('finite schema-sized train robot matrix required')
        return cls(schema, StateNormalizer(x.mean(0), x.std(0, correction=0).clamp_min(.001)), sorted(ids))

    def apply(self, values):
        if values.shape[-1] != self.schema['dimension'] or not torch.isfinite(values).all():
            raise ValueError('normalizer schema mismatch')
        return (values - self.stats.mean.to(values)) / self.stats.std.to(values)
