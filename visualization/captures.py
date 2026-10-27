"""Read dual-camera live diagnostic captures without executing the simulator."""

import base64
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def within(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('capture path outside experiments root')
    return path


def capture_episode(root, episode_id):
    root = Path(root).resolve()
    folder = within(root, root / episode_id)
    if folder.parent.name != 'collection' or not re.fullmatch(r'seed\d+', folder.name):
        raise ValueError('capture id must identify collection/seedN')
    manifest = within(root, folder.parent.parent / 'capture_manifest.json')
    value = json.loads(manifest.read_text())
    matches = [e for e in value['episodes'] if f"seed{e['seed']}" == folder.name]
    if len(matches) != 1 or not folder.is_dir():
        raise ValueError('capture episode missing or ambiguous')
    episode = matches[0]
    frames = episode['frames']
    steps = [f['timestep'] for f in frames]
    if (not frames or any(type(t) is not int or t < 0 for t in steps)
            or steps != sorted(set(steps))
            or any(f['stem'] != f"f{f['timestep']:04d}" for f in frames)):
        raise ValueError('invalid capture frame timeline')
    contract = json.loads(within(root, folder / 'sensor_contract.json').read_text())
    roles = contract['roles']
    if set(roles) != {'main', 'wrist'} or len(set(roles.values())) != 2:
        raise ValueError('main and wrist camera roles required')
    info = dict(id=folder.relative_to(root).as_posix(), name=folder.name,
                run_id=folder.parent.parent.name, seed=episode['seed'],
                split=episode['split'], frames=len(frames),
                complete=episode.get('complete') is True,
                source_kind=episode.get('source_kind', 'LIVE_DIAGNOSTIC'),
                roles=roles, backend=contract.get('backend'),
                timesteps=steps, phases=[f.get('phase', 'unknown') for f in frames])
    return folder, episode, info


def list_captures(root):
    root = Path(root).resolve()
    result = []
    for manifest in sorted(root.rglob('capture_manifest.json')):
        try:
            within(root, manifest)
            value = json.loads(manifest.read_text())
            for episode in value['episodes']:
                folder = manifest.parent / 'collection' / f"seed{episode['seed']}"
                _, _, info = capture_episode(root, folder.relative_to(root).as_posix())
                result.append(info)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def _archive(root, folder, record, kind):
    path = within(root, folder / f"{record['stem']}_{kind}.npz")
    expected = record.get({'input': 'input_sha256', 'masks': 'mask_sha256',
                           'gt': 'gt_sha256'}[kind])
    if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'{path.name}: capture manifest hash mismatch')
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _packed(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype, order='C').tobytes()).decode('ascii')


def capture_frame(root, episode_id, timestep):
    if type(timestep) is not int or timestep < 0:
        raise ValueError('nonnegative integer timestep required')
    folder, episode, info = capture_episode(root, episode_id)
    record = next((f for f in episode['frames'] if f['timestep'] == timestep), None)
    if record is None:
        raise FileNotFoundError('capture timestep missing')
    sensor = _archive(root, folder, record, 'input')
    masks = _archive(root, folder, record, 'masks')
    gt = _archive(root, folder, record, 'gt')
    if (sensor['timestep'].item() != timestep or
            sensor['episode'].item() != episode['episode'] or
            not np.isfinite(sensor['timestamp'].item())):
        raise ValueError('capture observation identity mismatch')
    actor_ids = gt['actor_ids']
    if actor_ids.shape != (2,) or actor_ids[0] == actor_ids[1]:
        raise ValueError('two distinct evaluation actor ids required')
    cameras = []
    for role, name in info['roles'].items():
        rgb, depth = sensor[name + '/rgb'], sensor[name + '/depth_raw_mm']
        predicted, segmentation = masks[name], gt[name + '/segmentation']
        h, w = depth.shape
        if (rgb.shape != (h, w, 3) or rgb.dtype != np.uint8 or
                depth.dtype != np.int16 or predicted.shape != (2, h, w) or
                predicted.dtype != np.bool_ or segmentation.shape != (h, w)):
            raise ValueError('invalid RGB/depth/mask shape or dtype')
        truth = segmentation[None] == actor_ids[:, None, None]
        objects = []
        for i, label in enumerate(('Peg', 'Box')):
            pred, target = predicted[i], truth[i]
            intersection = int(np.count_nonzero(pred & target))
            union = int(np.count_nonzero(pred | target))
            objects.append(dict(name=label, predicted_pixels=int(pred.sum()),
                                gt_pixels=int(target.sum()),
                                iou=intersection / union if union else None))
        valid = depth > 0
        cameras.append(dict(role=role, name=name, width=w, height=h,
                            rgb=_packed(rgb, 'u1'), depth_mm=_packed(depth, '<i2'),
                            predicted=_packed(predicted, 'u1'), gt=_packed(truth, 'u1'),
                            valid_depth_fraction=float(valid.mean()),
                            depth_min_mm=int(depth[valid].min()) if valid.any() else None,
                            depth_max_mm=int(depth[valid].max()) if valid.any() else None,
                            objects=objects))
    return dict(t=timestep, timestamp=float(sensor['timestamp']), phase=record.get('phase'),
                hashes_verified=True, cameras=cameras)
