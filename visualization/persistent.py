"""Read saved Task09-C graphs, preserving their schema and capture identity."""

import json
from pathlib import Path
import re

import numpy as np

from visualization.captures import _archive, _packed, capture_episode, within
from visual.runtime.vision_packet import pose_matrix


VARIANTS = {
    'G0': 'G0 · Main 원본 / Raw EDE',
    'G1': 'G1 · Main 현재 관측 + 로봇 제거',
    'G2': 'G2 · Main 누적 지도',
    'G3': 'G3 · Main + Wrist 현재 관측',
    'G4': 'G4 · Main + Wrist 누적 지도',
    'O1': 'O1 · GT 마스크 + 추정 이동 · Oracle',
    'O2': 'O2 · 예측 마스크 + GT 이동 · Oracle',
    'O3': 'O3 · GT 마스크 + GT 이동 · Oracle',
}
EXTRA_FIELDS = ['currently_observed', 'persistent_estimated', 'low_confidence',
                'confidence', 'age_seconds', 'uncertainty_m', 'pose_confidence']
FIELDS = {'node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type'}


def is_persistent_id(episode_id):
    return Path(episode_id).parent.name == 'geometry'


def persistent_episode(root, episode_id, *, check_complete=False):
    root = Path(root).resolve()
    folder = within(root, root / episode_id)
    if folder.parent.name != 'geometry' or not re.fullmatch(r'seed\d+', folder.name):
        raise ValueError('geometry id must identify geometry/seedN')
    run = folder.parent.parent
    manifest = json.loads(within(root, run / 'geometry_manifest.json').read_text())
    capture_run = manifest['config']['capture_run_id']
    if not isinstance(capture_run, str) or Path(capture_run).name != capture_run:
        raise ValueError('invalid capture run id')
    capture_id = f'{capture_run}/collection/{folder.name}'
    _, capture, capture_info = capture_episode(root, capture_id)
    records = [r for r in manifest['frames'] if r['seed'] == capture['seed']]
    steps = [r['timestep'] for r in records]
    if steps != capture_info['timesteps']:
        raise ValueError('geometry and capture timelines disagree')
    for record, observation in zip(records, capture['frames']):
        expected = folder / f"f{record['timestep']:04d}"
        if within(root, run / record['folder']) != expected:
            raise ValueError('geometry frame folder disagrees with timestep')
        if record['capture_input_sha256'] != observation['input_sha256']:
            raise ValueError('geometry belongs to a different capture observation')
    variants = [v for v in VARIANTS if v in manifest['config']['variants'] and
                (not check_complete or all(within(root, folder / f'f{t:04d}' / f'{v}_graph.npz').is_file()
                    and within(root, folder / f'f{t:04d}' / f'{v}_graph.json').is_file()
                    for t in steps))]
    if not variants:
        raise ValueError('no complete saved graph variant')
    info = dict(id=folder.relative_to(root).as_posix(), name=f'Task09-C · {folder.name}',
                frames=len(steps), first_timestep=steps[0], last_timestep=steps[-1],
                status='LIVE_DIAGNOSTIC', success=None, first_success_step=None,
                end_reason=None, result='unknown', kind='persistent_geometry',
                split=capture['split'], capture_id=capture_id,
                revision=manifest['config'].get('geometry_revision', 'unspecified'),
                variants=[dict(id=v, label=VARIANTS[v]) for v in variants],
                default_variant='G4' if 'G4' in variants else variants[0])
    return folder, records, info


def list_persistent_episodes(root):
    root = Path(root).resolve()
    result = []
    for path in sorted(root.rglob('geometry_manifest.json')):
        try:
            manifest = json.loads(within(root, path).read_text())
            seeds = sorted({r['seed'] for r in manifest['frames']})
            for seed in seeds:
                episode_id = (path.parent / 'geometry' / f'seed{seed}').relative_to(root).as_posix()
                _, records, info = persistent_episode(root, episode_id, check_complete=True)
                persistent_arrays(root, episode_id, records[0]['timestep'], info['default_variant'])
                result.append(info)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def persistent_arrays(root, episode_id, timestep, variant, *, context=None):
    if variant not in VARIANTS or type(timestep) is not int or timestep < 0:
        raise ValueError('invalid geometry variant or timestep')
    folder, records, info = context or persistent_episode(root, episode_id)
    if variant not in [v['id'] for v in info['variants']]:
        raise ValueError('graph variant unavailable')
    record = next((r for r in records if r['timestep'] == timestep), None)
    if record is None:
        raise FileNotFoundError('geometry timestep missing')
    capture_folder, capture, _ = capture_episode(root, info['capture_id'])
    observation = next(r for r in capture['frames'] if r['timestep'] == timestep)
    sensor = _archive(root, capture_folder, observation, 'input')
    if sensor['timestep'].item() != timestep or sensor['episode'].item() != capture['episode']:
        raise ValueError('graph/capture observation identity mismatch')
    path = within(root, folder / f'f{timestep:04d}' / f'{variant}_graph.npz')
    metadata = json.loads(within(root, path.with_suffix('.json')).read_text())
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != FIELDS:
            raise ValueError('unexpected Task09-C graph fields')
        arrays = {key: saved[key] for key in FIELDS}
    nodes, kinds = arrays['node_input'], arrays['node_type']
    edges, attrs, types = arrays['edge_index'], arrays['edge_attr'], arrays['edge_type']
    if nodes.ndim != 2 or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('invalid graph dimensions')
    n, width, e = len(nodes), nodes.shape[1], edges.shape[1]
    if (n < 3 or width != 21 or nodes.dtype != np.float32 or
            kinds.shape != (n,) or not np.issubdtype(kinds.dtype, np.integer) or
            not np.issubdtype(edges.dtype, np.integer) or
            attrs.shape != (e, 4) or attrs.dtype != np.float32 or
            types.shape != (e,) or not np.issubdtype(types.dtype, np.integer) or
            not np.isfinite(nodes).all() or not np.isfinite(attrs).all() or
            (edges < 0).any() or (edges >= n).any() or
            (kinds < 0).any() or (kinds > 4).any() or
            (types < 0).any() or (types > 2).any() or
            not np.array_equal(kinds[:3], np.arange(3)) or (kinds[3:] < 3).any()):
        raise ValueError('invalid Task09-C graph arrays')
    if metadata.get('schema_id') != 'task09c_persistent_surface_v1':
        raise ValueError('unsupported saved graph schema')
    if 'timestep' in metadata and metadata['timestep'] != timestep:
        raise ValueError('graph metadata timestep mismatch')
    node_ids = metadata.get('node_ids')
    source_rows = metadata.get('selected_original_rows')
    if node_ids is not None:
        if len(node_ids) != n-3 or len(set(node_ids)) != n-3:
            raise ValueError('invalid persistent node identities')
    elif source_rows is None or len(source_rows) != n:
        raise ValueError('saved node identities missing')
    transform = pose_matrix(sensor['tcp_pose'])
    world = nodes[:, :3].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    if not np.allclose(world[:3], sensor['robot_world_geometry'], rtol=0, atol=1e-6):
        raise ValueError('graph and same-timestep robot geometry disagree')
    arrays.update(world_position=world, metadata=metadata, node_ids=node_ids,
                  source_rows=source_rows, sensor=sensor, info=info, record=record)
    return arrays


def persistent_frame(root, episode_id, timestep, variant):
    data = persistent_arrays(root, episode_id, timestep, variant)
    nodes, edges = data['node_input'], data['edge_index']
    rows = data['source_rows']
    return dict(t=timestep, n=len(nodes), e=edges.shape[1], feature_dimension=nodes.shape[1],
                node_input=_packed(nodes, '<f4'), node_type=_packed(data['node_type'], 'u1'),
                world_position=_packed(data['world_position'], '<f4'),
                edge_index=_packed(edges, '<i4'), edge_attr=_packed(data['edge_attr'], '<f4'),
                edge_type=_packed(data['edge_type'], 'u1'),
                source_index=_packed(np.asarray(rows[3:])-3 if rows is not None else [], '<i4'),
                source_label='original object row' if rows is not None else 'source index',
                node_ids=data['node_ids'], feature_fields=EXTRA_FIELDS,
                variant=variant, phase=data['record'].get('phase'),
                timestamp=float(data['sensor']['timestamp']),
                schema_id=data['metadata']['schema_id'], edge_policy=data['metadata'].get('edge_policy'))
