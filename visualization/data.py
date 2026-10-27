"""Read saved Task07 graph inputs and list Task09-C diagnostic graphs."""

import base64
import json
from pathlib import Path
import re

import numpy as np

from graph.integration_baseline.graph import EDGE_TYPES, NODE_TYPES
from visual.runtime.vision_packet import pose_matrix


REQUIRED = {'node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type', 'source_index'}


def read_frame(path, tcp_pose, robot_ee, robot_world):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != REQUIRED:
            raise ValueError(f'{path}: unexpected graph fields: {sorted(saved.files)}')
        data = {name: saved[name] for name in REQUIRED}
    nodes, kinds = data['node_input'], data['node_type']
    edges, attrs, edge_kinds = data['edge_index'], data['edge_attr'], data['edge_type']
    n, e = len(nodes), edges.shape[1]
    if (nodes.shape != (n, 14) or nodes.dtype != np.float32 or
            kinds.shape != (n,) or not np.issubdtype(kinds.dtype, np.integer) or
            edges.shape != (2, e) or not np.issubdtype(edges.dtype, np.integer) or
            attrs.shape != (e, 4) or attrs.dtype != np.float32 or
            edge_kinds.shape != (e,) or not np.issubdtype(edge_kinds.dtype, np.integer) or
            data['source_index'].shape != (n - 3,) or
            not np.issubdtype(data['source_index'].dtype, np.integer)):
        raise ValueError(f'{path}: invalid graph array shape or dtype')
    if (n < 3 or not np.isfinite(nodes).all() or not np.isfinite(attrs).all() or
            (edges < 0).any() or (edges >= n).any() or
            (kinds < 0).any() or (kinds >= len(NODE_TYPES)).any() or
            (edge_kinds < 0).any() or (edge_kinds >= len(EDGE_TYPES)).any() or
            (data['source_index'] < 0).any() or
            (data['source_index'] > np.iinfo(np.int32).max).any()):
        raise ValueError(f'{path}: invalid graph values')
    transform = pose_matrix(tcp_pose)
    world_positions = (nodes[:, :3].astype(np.float64) @ transform[:3, :3].T
                       + transform[:3, 3])
    if (not np.array_equal(nodes[:3, :3], robot_ee) or
            not np.allclose(world_positions[:3], robot_world, rtol=0, atol=1e-6)):
        raise ValueError(f'{path}: graph and same-timestep robot observation disagree')
    timestep = int(path.stem.removeprefix('graph_'))
    def packed(array, dtype):
        return base64.b64encode(np.asarray(array, dtype=dtype, order='C').tobytes()).decode('ascii')

    return dict(t=timestep, n=n, e=e,
                node_input=packed(nodes, '<f4'), node_type=packed(kinds, 'u1'),
                world_position=packed(world_positions, '<f4'),
                edge_index=packed(edges, '<i4'), edge_attr=packed(attrs, '<f4'),
                edge_type=packed(edge_kinds, 'u1'),
                source_index=packed(data['source_index'], '<i4'))


def observation_rows(episode):
    episode = Path(episode)
    observations = episode/'observations.npz'
    if not observations.is_file():
        raise ValueError(f'{observations}: required for EE/world coordinate toggle')
    with np.load(observations, allow_pickle=False) as saved:
        required = {'timestep', 'tcp_pose', 'robot_ee_geometry', 'robot_world_geometry'}
        if not required.issubset(saved.files):
            raise ValueError(f'{observations}: missing pose or robot geometry')
        timesteps, poses = saved['timestep'], saved['tcp_pose']
        robot_ee, robot_world = saved['robot_ee_geometry'], saved['robot_world_geometry']
    count = len(timesteps)
    if (timesteps.shape != (count,) or not np.issubdtype(timesteps.dtype, np.integer) or
            len(np.unique(timesteps)) != count or
            poses.shape != (count, 7) or robot_ee.shape != (count, 3, 3) or
            robot_world.shape != (count, 3, 3) or
            not np.isfinite(poses).all() or not np.isfinite(robot_ee).all() or
            not np.isfinite(robot_world).all()):
        raise ValueError(f'{observations}: invalid timestep or pose arrays')
    return {int(step): (poses[i], robot_ee[i], robot_world[i])
            for i, step in enumerate(timesteps)}


def graph_steps(episode):
    paths = [path for path in Path(episode).glob('graph_*.npz')
             if re.fullmatch(r'graph_\d{4}\.npz', path.name)]
    steps = sorted(int(path.stem[6:]) for path in paths)
    if not steps or len(steps) != len(set(steps)):
        raise ValueError(f'{episode}: no graph frames or duplicate timesteps')
    return steps


def outcome(episode):
    for name in ('episode.json', 'trial.json', 'summary.json'):
        path = episode/name
        if path.is_file():
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                result = {key: value.get(key) for key in
                          ('status', 'success', 'first_success_step', 'end_reason')}
                result['result'] = ('success' if result['success'] is True else
                                    'failure' if result['success'] is False and
                                    result['status'] == 'COMPLETE' else 'unknown')
                return result
    return dict(status=None, success=None, first_success_step=None,
                end_reason=None, result='unknown')


def episode_info(episode, root):
    episode = Path(episode).resolve()
    root = Path(root).resolve()
    if not episode.is_relative_to(root):
        raise ValueError('episode outside experiments root')
    steps = graph_steps(episode)
    result = dict(id=episode.relative_to(root).as_posix(), name=episode.name,
                  frames=len(steps), first_timestep=steps[0],
                  last_timestep=steps[-1], **outcome(episode))
    return result


def list_episodes(root):
    root = Path(root).resolve()
    episodes = []
    for observations in root.rglob('observations.npz'):
        folder = observations.parent
        if not next(folder.glob('graph_[0-9][0-9][0-9][0-9].npz'), None):
            continue
        try:
            info = episode_info(folder, root)
            frame_for_step(folder, info['first_timestep'])
            episodes.append(info)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    rank = {'success': 0, 'failure': 1, 'unknown': 2}
    from visualization.persistent import list_persistent_episodes
    episodes.extend(list_persistent_episodes(root))
    return sorted(episodes, key=lambda item: (rank[item['result']], item['id']))


def frame_for_step(episode, timestep):
    if type(timestep) is not int or timestep < 0:
        raise ValueError('nonnegative integer timestep required')
    episode = Path(episode)
    path = episode/f'graph_{timestep:04d}.npz'
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = observation_rows(episode)
    if timestep not in rows:
        raise ValueError(f'{path}: matching observation timestep is missing')
    return read_frame(path, *rows[timestep])
