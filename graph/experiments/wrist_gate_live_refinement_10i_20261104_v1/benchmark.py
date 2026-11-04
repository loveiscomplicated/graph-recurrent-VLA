"""Paired before/after timing at the live prepare_tensor_gate boundary."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import triton

from graph.experiments.wrist_gate_cuda_10i_20261104_v1.benchmark import load_case, sha, array_sha, write
from graph.persistent_geometry import tensor_gate, tensor_fusion, contracts
from graph.persistent_geometry.tensor_camera import tensor_camera
from utils.experiment_logger import ExperimentLogger


def load_baseline(folder):
    def load(stem):
        name = 'graph.persistent_geometry._before_live_' + stem
        spec = importlib.util.spec_from_file_location(name, folder / (stem + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    fusion, gate = load('tensor_fusion'), load('tensor_gate')
    gate.prepare_tensor_views = fusion.prepare_tensor_views
    gate.consistency_keep = fusion.consistency_keep
    return gate, fusion


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA execution required')
    if min(args.warmups, args.repeats) < 1: raise ValueError('positive warmups/repeats required')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    baseline = Path(args.baseline)
    before, before_fusion = load_baseline(baseline)
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    cameras, raw, clean, gate_cfg, prior, input_path = load_case(Path(args.input))
    cameras = tuple(tensor_camera(c) for c in cameras)
    raw = {k: torch.as_tensor(v, device='cuda') for k, v in raw.items()}
    clean = {k: torch.as_tensor(v, device='cuda') for k, v in clean.items()}
    depths = {c.camera: tensor_fusion.depth_from_mm(c.depth_raw_mm) for c in cameras}
    data = dict(episode=cameras[0].episode, timestep=cameras[0].timestep)
    config = dict(seed=42, warmups=args.warmups, repeats=args.repeats,
        boundary='CurrentGeometry._build_prepared -> prepare_tensor_gate; native CUDA cameras, raw/clean masks and float64 depths ready -> C3 CUDA masks, full Python gate trace and Graph-frame TensorViews; synchronized wall time',
        excluded='SAM2, robot filtering/depth preparation, Graph construction, policy, simulator, verification readback, file I/O',
        pairing='adjacent GPU before/after calls; alternate order each repeat; correctness outside timer',
        input_sha256=sha(input_path), clean_points=29775, gate_config=gate_cfg,
        gpu=torch.cuda.get_device_name(), python=platform.python_version(),
        numpy=np.__version__, torch=torch.__version__, triton=triton.__version__,
        cpu=next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
        local_before_identity=json.loads((baseline.parent / 'before_identity.json').read_text()),
        source_sha256={str(p): sha(p) for p in (Path(__file__), Path(tensor_gate.__file__), Path(tensor_fusion.__file__))},
        baseline_sha256={p.name: sha(p) for p in baseline.glob('*.py')})
    write(output / 'config.json', config)
    with ExperimentLogger(team='integration', model='task10i_wrist_gate_live_refinement', seed=42,
            job_type='evaluate', root=output / 'experiments', config=config) as logger:
        logger.record(input_sha256=config['input_sha256'], source_run_id='46439da34962499a989535b737b71488')
        results = {}
        masks = {}
        for frame in ('ee', 'world'):
            def call(module):
                return module.prepare_tensor_gate(cameras, data, raw, clean, {'gate': gate_cfg},
                                                  coordinate_frame=frame, depths=depths)
            def check(value):
                prepared, views = value
                if prepared['gate'] != prior['trace']: raise AssertionError('full trace differs from frozen CPU reference')
                mask = prepared['conditions']['C3']['hand_camera'][0].cpu().numpy()
                if array_sha(mask) != prior['accepted_mask_sha256']: raise AssertionError('accepted mask differs')
                if views.coordinate_frame != frame: raise AssertionError('Graph frame differs')
                for name, view in views.by_name.items():
                    if view.mask is not prepared['conditions']['C3'][name]: raise AssertionError('mask snapshot differs')
                return mask
            for _ in range(args.warmups):
                for module in (before, tensor_gate):
                    check(call(module))
                    torch.cuda.synchronize()
            rows = []
            for repeat in range(args.repeats):
                order = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
                row = dict(repeat=repeat, order=order, wall_ms={})
                for name in order:
                    torch.cuda.synchronize(); start = time.perf_counter_ns()
                    value = call(before if name == 'before' else tensor_gate)
                    torch.cuda.synchronize()
                    row['wall_ms'][name] = (time.perf_counter_ns()-start)/1e6
                    masks[frame+'_'+name] = check(value)
                rows.append(row)
                print('MEASURE', frame, json.dumps(row), flush=True)
            stats = {}
            for name in ('before', 'after'):
                values = [row['wall_ms'][name] for row in rows]
                stats[name] = dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values))
            audits = {}
            for name, module, fusion in (('before', before, before_fusion), ('after', tensor_gate, tensor_fusion)):
                calls = []
                original = contracts.synchronized
                def validate(c):
                    calls.append(1); return original(c)
                with patch.object(contracts, 'synchronized', side_effect=validate), \
                        patch.object(fusion, 'synchronized', side_effect=validate), \
                        patch('torch.equal', wraps=torch.equal) as equal:
                    check(call(module))
                audits[name] = dict(camera_validation_passes=len(calls), connectivity_host_checks=equal.call_count)
            results[frame] = dict(statistics=stats, rows=rows, audits=audits,
                mask_exact=True, full_trace_exact=True, accepted_points=prior['trace']['accepted_points'],
                speedup=stats['before']['median_ms']/stats['after']['median_ms'],
                reduction_percent=100*(1-stats['after']['median_ms']/stats['before']['median_ms']))
            logger.log({frame: {name+'_ms': stats[name]['median_ms'] for name in stats}})
        result = dict(status='PASS', run_id=logger.run_id, frames=results,
                      accepted_mask_sha256=prior['accepted_mask_sha256'], trace=prior['trace'])
        write(output / 'results.json', result)
        np.savez_compressed(output / 'accepted_masks.npz', **masks)
        for p in (output/'config.json', output/'results.json', output/'accepted_masks.npz'):
            logger.link_output(p)
        print('COMPLETE', json.dumps(dict(run_id=logger.run_id,
            frames={f:{k:v for k,v in r.items() if k!='rows'} for f,r in results.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    run(parser.parse_args())
