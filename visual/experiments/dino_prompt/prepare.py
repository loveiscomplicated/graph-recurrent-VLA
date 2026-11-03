"""Read existing source artifacts and build the frozen Task10-J input bundle.

No simulator, model, or ground-truth generation is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def build(workspace, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    diag = workspace / 'task10g_f1_abc_20261031/diagnostics'
    specs = [('98202', 'base_camera', 0, 'search', False)]
    specs += [('0355', 'base_camera', t, 'search', False) for t in range(0, 31, 5)]
    specs += [('0645', 'base_camera', 0, 'search', True),
              ('0645', 'hand_camera', 6, 'search', True)]
    specs += [(p, 'hand_camera', 0, 'confirm', False) for p in ('0362', '0666', '0810')]
    specs += [('98200', 'base_camera', 0, 'confirm', True)]
    frames, arrays, sources = [], {}, {}
    def identity(p):
        p = Path(p)
        if str(p) not in sources:
            sources[str(p)] = sha(p)
        return sources[str(p)]
    for parent, camera, t, split, control in specs:
        key = f'{parent}_{camera}_{t:03d}'
        row = dict(id=key, parent=parent, camera=camera, timestep=t,
                   split=split, control=control, gt_available=False)
        if parent.startswith('98'):
            d = diag / ('seed98200_recognition_v2' if parent == '98200' else 'seed98202_initialization_v1')
            rgb_path = d / f'{camera}_rgb_t{t:03d}.png'
            if not rgb_path.exists():
                row['missing'] = 'original RGB unavailable'
                frames.append(row)
                continue
            rgb = np.array(Image.open(rgb_path).convert('RGB'))
            row.update(rgb_source=str(rgb_path), rgb_source_sha256=identity(rgb_path))
            gt = d / 'all_masks_segmentation.npz'
            if gt.exists():
                with np.load(gt) as z:
                    camera_index = list(z['camera_names']).index(camera)
                    seg = z['segmentation'][t, camera_index]
                    ids = z['object_ids'].tolist()
                row.update(gt_available=True, gt_source=str(gt), gt_source_sha256=identity(gt),
                           object_ids=dict(zip(('peg', 'box'), [[int(x)] for x in ids])))
            else:
                row['missing'] = 'stored exact actor segmentation unavailable; RGB-only diagnostic'
        else:
            raw = workspace / f'task10i_stage3_20261031_v3/run/parents/10i_s1_{parent}/full_task/episode.h5'
            episode = f'10i_stage3_10i_s1_{parent}_full_task'
            roots = [workspace / 'task10i_stage4a2_newdev_reference_20261101_v1/run/references',
                     workspace / 'task10i_stage4a_20261031_v5/run/references',
                     workspace / 'task10i_stage4a_20261031_v3/run/replay_reference']
            roots += sorted(workspace.glob('task10*/run/references'))
            roots += sorted(workspace.glob('task10*/run/replay_reference'))
            ref = next((r / episode for r in roots if (r / episode / 'receipt.json').exists()
                        and (r / episode / 'segmentation.npz').exists()), None)
            if not raw.exists() or ref is None:
                row['missing'] = 'original RGB or reference unavailable'
                frames.append(row)
                continue
            receipt = json.loads((ref / 'receipt.json').read_text())
            assert identity(raw) == receipt['raw_sha256'], str(raw)
            with h5py.File(raw) as f:
                rgb = f[camera]['rgb'][t]
            with np.load(ref / 'segmentation.npz') as z:
                exact = bool(z['exact_frame'][t])
                seg = z['actor_segmentation'][t, ('base_camera', 'hand_camera').index(camera)]
            row.update(rgb_source=str(raw), rgb_source_sha256=identity(raw),
                       gt_source=str(ref / 'segmentation.npz'),
                       gt_source_sha256=identity(ref / 'segmentation.npz'),
                       receipt_sha256=identity(ref / 'receipt.json'), gt_available=exact,
                       object_ids=receipt['object_ids'])
            assert row['gt_source_sha256'] == receipt['segmentation_sha256']
            if not exact:
                row['missing'] = 'reference is not EXACT_GT'
        assert rgb.dtype == np.uint8 and rgb.shape == (128, 128, 3)
        arrays[key + '_rgb'] = rgb
        row['rgb_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
        if row['gt_available']:
            arrays[key + '_seg'] = seg
            row['gt_pixels'] = {obj: int(np.isin(seg, ids).sum()) for obj, ids in row['object_ids'].items()}
        Image.fromarray(rgb).save(out / 'images' / (key + '.png'))
        frames.append(row)
    np.savez_compressed(out / 'frames.npz', **arrays)
    manifest = dict(task='10j_dino_prompt_feasibility', frames=frames, sources=sources,
                    bundle_sha256=sha(out / 'frames.npz'),
                    missing=[dict(id=f['id'], reason=f['missing']) for f in frames if 'missing' in f])
    write(out / 'manifest.json', manifest)
    print(json.dumps(dict(frames=len(frames), exact_gt=sum(f['gt_available'] for f in frames),
                         missing=manifest['missing']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path('/workspace'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    build(a.workspace, a.out)
