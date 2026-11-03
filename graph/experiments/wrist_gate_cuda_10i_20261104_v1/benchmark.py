"""Compare the existing CPU/CUDA gates on the same nonempty frozen input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
from unittest.mock import patch

import numpy as np
import scipy
import torch
import triton

from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.tensor_fusion import prepare_tensor_views
from graph.persistent_geometry.tensor_gate import wrist_gate_tensor
from graph.plan_a_geometry import core
from utils.experiment_logger import ExperimentLogger

SCENE = '10i_s1_0531_track_t011'
NAMES = ('base_camera', 'hand_camera')
DIAMETER_ATOL_M = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compare_trace(actual, expected):
    if actual.keys() != expected.keys():
        raise AssertionError('trace keys differ')
    for key in actual:
        if key == 'component_diameter_m':
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=DIAMETER_ATOL_M)
        elif actual[key] != expected[key]:
            raise AssertionError(f'trace field differs: {key}: {actual[key]} != {expected[key]}')
    return dict(trace_bitwise_equal=actual == expected,
                diameter_abs_delta_m=abs(actual['component_diameter_m'] - expected['component_diameter_m']))


def load_case(folder):
    config = json.loads((folder / 'config.json').read_text())
    prior = next(c for c in json.loads((folder / 'results.json').read_text())['cases'] if c['scene'] == SCENE)
    path = folder / prior['packed_input']
    if sha(path) != prior['packed_input_sha256']:
        raise AssertionError('frozen input file hash differs')
    with np.load(path, allow_pickle=False) as z:
        cameras = tuple(CameraObservation(SCENE, prior['timestep'], name, prior['timestep'] / 20,
            *[z[name + '__' + key].copy() for key in
              ('rgb', 'depth_raw_mm', 'intrinsic', 'extrinsic_cv', 'cam2world_gl', 'tcp_pose')]).validate()
            for name in NAMES)
        raw = {name: z[name + '__raw'].copy() for name in NAMES}
        clean = {name: z[name + '__clean'].copy() for name in NAMES}
    for name in NAMES:
        for kind, masks in (('raw', raw), ('clean', clean)):
            a = masks[name]
            if a.dtype != bool or a.shape != (2, 512, 512):
                raise AssertionError('original bool 512 masks required')
            if array_sha(a) != prior['masks'][name][kind + '_sha256']:
                raise AssertionError('mask hash differs')
    if int((clean['hand_camera'][0] & np.isfinite(cameras[1].depth)).sum()) != 29775:
        raise AssertionError('only the nonempty 29,775-point case is in scope')
    return cameras, raw, clean, config['gate_config'], prior, path


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA execution required; CPU fallback is not a GPU validation')
    if args.repeats < 1 or args.warmups < 1:
        raise ValueError('positive repeats and warmups required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    cameras, raw, clean, gate_cfg, prior, input_path = load_case(Path(args.input))
    wrist = cameras[1]
    # Upload each immutable input before measuring the resident gate boundary.
    raw_cuda = {name: torch.as_tensor(a, device='cuda') for name, a in raw.items()}
    clean_cuda = {name: torch.as_tensor(a, device='cuda') for name, a in clean.items()}
    views = prepare_tensor_views(cameras, clean_cuda, coordinate_frame='world')
    torch.cuda.synchronize()
    config = dict(scene=SCENE, seed=42, warmups=args.warmups, repeats=args.repeats,
        cpu_boundary='core.wrist_gate only, original CPU inputs ready -> NumPy mask and Python trace',
        cuda_boundary='wrist_gate_tensor only, raw/clean masks and float64 depth resident -> CUDA mask and Python trace; synchronized wall time',
        included='XYZ, exact radius connectivity, local support, cross-view consistency, host metadata/calibration checks/uploads, scalar synchronization and trace/mask construction',
        excluded='initial image/mask/depth upload, verification/output mask readback, file I/O, FK/robot filtering, SAM2/detection, Graph, policy, simulator',
        diameter_comparison_atol_m=DIAMETER_ATOL_M, gate_config=gate_cfg,
        input_sha256=sha(input_path), source_run_id='46439da34962499a989535b737b71488',
        cpu=next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), 'unknown'),
        gpu=torch.cuda.get_device_name(0), instance_id=os.getenv('CONTAINER_ID'),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        torch=torch.__version__, triton=triton.__version__, torch_threads=torch.get_num_threads(),
        environment={k: os.getenv(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        affinity=sorted(os.sched_getaffinity(0)), cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        source_sha256={str(p): sha(p) for p in (Path(__file__), Path(core.__file__),
            Path('graph/persistent_geometry/tensor_gate.py'), Path('graph/persistent_geometry/tensor_gate_kernels.py'),
            Path('graph/persistent_geometry/tensor_fusion.py'), Path('graph/persistent_geometry/contracts.py'))})
    write(output / 'config.json', config)
    with ExperimentLogger(team='integration', model='task10i_wrist_gate_cuda_first_step',
            seed=42, job_type='evaluate', root=output / 'experiments', config=config) as logger:
        logger.record(source_run_id=config['source_run_id'], input_sha256=config['input_sha256'])
        mask_cpu, trace_cpu = core.wrist_gate(wrist, raw[wrist.camera], clean[wrist.camera], cameras, clean, gate_cfg)
        if array_sha(mask_cpu) != prior['accepted_mask_sha256'] or trace_cpu != prior['trace']:
            raise AssertionError('current CPU output differs from the frozen reference')

        def gpu_call():
            return wrist_gate_tensor(cameras, raw_cuda, clean_cuda, gate_cfg, views=views)

        def cpu_call():
            return core.wrist_gate(wrist, raw[wrist.camera], clean[wrist.camera], cameras, clean, gate_cfg)

        # A guard verifies that GPU execution does not fall back to CPU geometry.
        with (patch.object(core, 'components', side_effect=AssertionError('CPU components called')),
              patch.object(CameraObservation, 'world_points', side_effect=AssertionError('CPU world_points called'))):
            mask_gpu, trace_gpu = gpu_call()
            torch.cuda.synchronize()
        np.testing.assert_array_equal(mask_gpu.cpu().numpy(), mask_cpu)
        equivalence = compare_trace(trace_gpu, trace_cpu)
        print('REFERENCE', json.dumps(dict(accepted_mask_exact=True, **equivalence)), flush=True)

        def check(name, mask, trace):
            host = mask.cpu().numpy() if name == 'cuda' else mask
            np.testing.assert_array_equal(host, mask_cpu)
            compare_trace(trace, trace_cpu)
            # Repeats must also reproduce their own reference trace exactly.
            if trace != (trace_gpu if name == 'cuda' else trace_cpu):
                raise AssertionError('repeat trace changed')

        calls = {'cpu': cpu_call, 'cuda': gpu_call}
        for _ in range(args.warmups):
            for name, call in calls.items():
                mask, trace = call()
                torch.cuda.synchronize()
                check(name, mask, trace)
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for repeat in range(args.repeats):
            order = ('cpu', 'cuda') if repeat % 2 == 0 else ('cuda', 'cpu')
            row = dict(repeat=repeat, order=list(order), wall_ms={})
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                mask, trace = calls[name]()
                if name == 'cuda':
                    torch.cuda.synchronize()
                row['wall_ms'][name] = (time.perf_counter_ns() - start) / 1e6
                check(name, mask, trace)
            row['accepted_mask_exact'] = True
            row['repeat_trace_exact'] = True
            rows.append(row)
            print('MEASURE', json.dumps(row), flush=True)
        stats = {}
        for name in calls:
            values = [row['wall_ms'][name] for row in rows]
            stats[name] = dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values))
        result = dict(status='PASS', run_id=logger.run_id, scene=SCENE, clean_points=29775,
            accepted_points=int(mask_cpu.sum()), accepted_mask_sha256=array_sha(mask_cpu),
            cpu_trace=trace_cpu, cuda_trace=trace_gpu, equivalence=equivalence,
            cpu_geometry_fallback_guard_passed=True, rows=rows, statistics=stats,
            median_speedup=stats['cpu']['median_ms'] / stats['cuda']['median_ms'],
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        np.savez_compressed(output / 'accepted_masks.npz', cpu=mask_cpu, cuda=mask_gpu.cpu().numpy())
        write(output / 'results.json', result)
        logger.log({'latency': {name: value['median_ms'] for name, value in stats.items()},
                    'median_speedup': result['median_speedup'], 'accepted_points': result['accepted_points']})
        for path in (output / 'config.json', output / 'results.json', output / 'accepted_masks.npz'):
            logger.link_output(path)
        print('COMPLETE', json.dumps(dict(run_id=logger.run_id, statistics=stats,
            median_speedup=result['median_speedup'], equivalence=equivalence)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Folder containing the original config/results/packed input')
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=15)
    run(parser.parse_args())
