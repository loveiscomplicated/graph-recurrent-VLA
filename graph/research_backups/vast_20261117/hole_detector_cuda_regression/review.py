"""Matched-frame audit of the detector presets with the team's holdout scorer."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.initial_corner_detection import verify_camera
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import score, visible_reference

CAMERA = dict(camera_eye_m=[0.65, -0.30, 0.65], camera_target_m=[0.065, 0.297, 0.11],
              resolution=[512, 512], vertical_fov_deg=90)
FAMILIES = ('peg', 'box_outer', 'hole_front', 'hole_rear')


def previous_detector(path):
    spec = importlib.util.spec_from_file_location('original_corner_detection', path / '__init__.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TorchCornerDetector.main(device='cuda:0', optimized=False, compile_cuda_edges=False)


def detectors(old):
    yield 'previous', previous_detector(old), False
    for name, fit, optimized, compiled, tensor in (
        ('latest_legacy', 'ransac', False, False, False),
        ('ransac_eager', 'ransac', True, False, False),
        ('ransac_compiled', 'ransac', True, True, False),
        ('grid_eager', 'grid', True, False, False),
        ('grid_compiled_numpy', 'grid', True, True, False),
        ('grid_compiled_cuda', 'grid', True, True, True),
    ):
        yield name, TorchCornerDetector.main(device='cuda:0', plane_fit_method=fit,
                                             optimized=optimized, compile_cuda_validation=compiled), tensor


def frame_score(result, refs):
    predictions = result.as_dict()['corners']
    scores = score(predictions, refs, [], .003)
    return dict(supported={name: sum(r['kind'] == name for r in refs) for name in FAMILIES},
                matched={name: sum(m['kind'] == name for s in scores for m in s['matches'])
                         for name in FAMILIES},
                diagnostics=result.diagnostics,
                counts={'peg': scores[0]['predictions'], 'box': scores[1]['predictions'],
                        'rim_flagged': sum(p['object_id'] == 1 and
                                           (p.get('contour_hole') or p.get('method') == 'box_opening')
                                           for p in predictions)})


def summarize(rows):
    out = {}
    for scope in ('systematic', 'targeted'):
        group = [row for row in rows if row['scope'] == scope]
        if not group:
            continue
        out[scope] = {}
        for name in group[0]['detectors']:
            xs = [r['detectors'][name] for r in group]
            groups = {f: (f,) for f in FAMILIES}
            groups['hole_all'] = ('hole_front', 'hole_rear')
            out[scope][name] = dict(frames=len(xs))
            for family, parts in groups.items():
                supports = [sum(x['supported'][f] for f in parts) for x in xs]
                hits = [sum(x['matched'][f] for f in parts) for x in xs]
                out[scope][name][family] = dict(matched=sum(hits), supported=sum(supports),
                                               supported_frames=sum(bool(s) for s in supports),
                                               zero_matched=sum(s > 0 and h == 0 for s, h in zip(supports, hits)))
    return out


def evaluate(args):
    config = dict(camera=CAMERA, systematic=list(range(200, 1400, 10)), targeted=args.targeted,
                  source='same initial seed cohort as paired Object21 experiment',
                  torch_version=str(torch.__version__))
    torch.set_num_threads(2)
    models = list(detectors(args.old))
    env = make_peg_insertion_env(device='cpu', render_backend='cuda',
                                 obs_mode='rgb+depth+segmentation', control_mode='pd_ee_delta_pose',
                                 sensor_configs={'hand_camera': dict(width=16, height=16)})
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        with ExperimentLogger(team='graph', job_type='evaluate', model='hole_detector_cuda_regression',
                              config=config, seed=200, root=args.output/'experiments') as logger:
            logger.record(backend=str(env.unwrapped.backend), gpu=torch.cuda.get_device_name())
            rows = []
            with (args.output/'frames.jsonl').open('w') as trace:
                for scope in ('systematic','targeted'):
                    for seed in config[scope]:
                        obs, _ = env.reset(seed=seed)
                        sensor = {k: v[0].cpu().numpy() for k,v in obs['sensor_data']['base_camera'].items()}
                        calib = {k: v[0].cpu().numpy() for k,v in obs['sensor_param']['base_camera'].items()}
                        verify_camera(sensor, calib, CAMERA)
                        sensor.update(calib)
                        state = {k:v[0].cpu().numpy() for k,v in vars(get_gt_state(env)).items()}
                        task = env.unwrapped
                        ids = {0:int(task.peg.per_scene_id[0]), 1:int(task.box.per_scene_id[0])}
                        refs = visible_reference(state, sensor, ids)
                        for ref in refs:
                            if ref['kind'] == 'hole_rim':
                                ref['kind'] = 'hole_front' if ref['corner_id'] < 4 else 'hole_rear'
                        host = tuple(sensor[k] for k in ('rgb','depth','intrinsic_cv','extrinsic_cv'))
                        gpu = tuple(torch.as_tensor(x, device='cuda:0') for x in host)
                        row = dict(seed=seed,scope=scope,detectors={name:frame_score(d(*(gpu if resident else host)), refs)
                                                                  for name,d,resident in models})
                        trace.write(json.dumps(row,allow_nan=False)+'\n')
                        trace.flush()
                        rows.append(row)
                        if len(rows) == 1 or len(rows) % 20 == 0:
                            print(len(rows), scope, seed, flush=True)
            summary = summarize(rows)
            path = args.output/'summary.json'
            path.write_text(json.dumps(dict(run_id=logger.run_id,config=config,summary=summary), indent=2)+'\n')
            logger.log(summary, prefix='eval')
            logger.link_output(path)
            logger.link_output(args.output/'frames.jsonl')
            print('Completed', logger.run_id, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--targeted',type=int,nargs='*',default=[])
    evaluate(parser.parse_args())
