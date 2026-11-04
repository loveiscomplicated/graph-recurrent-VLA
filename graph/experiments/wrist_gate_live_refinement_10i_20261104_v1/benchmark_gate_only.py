"""Time only the current wrist_gate_tensor on the saved nonempty input."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch
import triton

from graph.experiments.wrist_gate_cuda_10i_20261104_v1.benchmark import load_case, sha, array_sha, write
from graph.persistent_geometry import tensor_gate, tensor_fusion
from graph.persistent_geometry.tensor_camera import tensor_camera
from graph.plan_a_geometry import core
from utils.experiment_logger import ExperimentLogger


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is not a GPU measurement')
    if min(args.warmups, args.repeats) < 1:
        raise ValueError('positive warmups/repeats required')
    identity = json.loads(Path(args.identity).read_text())
    for path, expected in identity['source_sha256'].items():
        if sha(path) != expected:
            raise AssertionError('source snapshot differs: ' + path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    cameras, raw, clean, gate_cfg, prior, input_path = load_case(Path(args.input))
    cameras = tuple(tensor_camera(c) for c in cameras)
    raw = {name: torch.as_tensor(value, device='cuda') for name, value in raw.items()}
    clean = {name: torch.as_tensor(value, device='cuda') for name, value in clean.items()}
    # Only World views are prepared, outside the measured region. No Graph
    # frame/view, EE matrix, C3 preparation or Graph construction is executed.
    views = tensor_fusion.prepare_tensor_views(cameras, clean, coordinate_frame='world')
    torch.cuda.synchronize()
    config = dict(seed=42, warmups=args.warmups, repeats=args.repeats,
        boundary='wrist_gate_tensor(cameras, raw_cuda, clean_cuda, gate_cfg, views=world_views) only; CUDA synchronized wall time',
        ready_inputs='native CUDA cameras, raw/clean masks, float64 depth, intrinsic and World calibration tensors',
        included='World 3D reconstruction, radius connectivity, local support, cross-camera consistency, snapshot identity check, all scalar synchronization, full Python trace and accepted CUDA mask',
        excluded='input upload, World view/depth/calibration preparation, prepare_tensor_gate, Graph-frame view/EE matrix preparation, C3 clone/update, SAM2, robot filter, Graph, policy, simulator, verification readback and file I/O',
        gate_frame='world', graph_executed=False, ee_preparation_executed=False,
        clean_points=29775, input_sha256=sha(input_path), gate_config=gate_cfg,
        local_source_identity=identity, gpu=torch.cuda.get_device_name(),
        cpu=next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        triton=triton.__version__, torch_threads=torch.get_num_threads(),
        affinity=sorted(os.sched_getaffinity(0)), cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip())
    write(output / 'config.json', config)
    with ExperimentLogger(team='integration', model='task10i_wrist_gate_gpu_only_timing', seed=42,
            job_type='evaluate', root=output/'experiments', config=config) as logger:
        logger.record(input_sha256=config['input_sha256'], source_run_id='46439da34962499a989535b737b71488')

        def call():
            return tensor_gate.wrist_gate_tensor(cameras, raw, clean, gate_cfg, views=views)

        def check(value):
            mask, trace = value
            host = mask.cpu().numpy()
            if array_sha(host) != prior['accepted_mask_sha256']:
                raise AssertionError('accepted mask differs from frozen CPU reference')
            if trace != prior['trace']:
                raise AssertionError('full trace differs from frozen CPU reference')
            return host

        # Untimed guards ensure the measured function neither retries CPU
        # geometry nor prepares any World/Graph/EE views inside this boundary.
        with patch.object(core, 'wrist_gate', side_effect=AssertionError('CPU gate called')), \
                patch.object(core, 'components', side_effect=AssertionError('CPU components called')), \
                patch.object(tensor_gate, 'prepare_tensor_views', side_effect=AssertionError('view preparation called')):
            check(call())
        for _ in range(args.warmups):
            value = call()
            torch.cuda.synchronize()
            check(value)
        rows = []
        for repeat in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            value = call()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns()-start)/1e6
            mask = check(value)
            rows.append(dict(repeat=repeat, wall_ms=elapsed_ms, mask_exact=True, full_trace_exact=True))
            print('MEASURE', json.dumps(rows[-1]), flush=True)
        values = [row['wall_ms'] for row in rows]
        stats = dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values))
        result = dict(status='PASS', run_id=logger.run_id, scene=prior['scene'], clean_points=29775,
            accepted_points=prior['trace']['accepted_points'], accepted_mask_sha256=array_sha(mask),
            trace=value[1], rows=rows, statistics=stats, mask_exact=True, full_trace_exact=True,
            cpu_fallback_guard_passed=True, no_view_preparation_guard_passed=True)
        write(output/'results.json', result)
        np.savez_compressed(output/'accepted_mask.npz', accepted=mask)
        logger.log(dict(gate_only_latency_ms=stats['median_ms'], accepted_points=result['accepted_points']))
        for p in (output/'config.json', output/'results.json', output/'accepted_mask.npz'):
            logger.link_output(p)
        print('COMPLETE', json.dumps(dict(run_id=logger.run_id, statistics=stats,
            mask_exact=True, full_trace_exact=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    run(parser.parse_args())
