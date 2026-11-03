"""Time the unchanged CPU wrist_gate on frozen real 512 camera/mask inputs."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import scipy
import torch

from graph.experiments.robot_sam_overlap_live_10i_20261103_v1.run import Inputs, NAMES, ROOT
from graph.plan_a_geometry import core
from utils.experiment_logger import ExperimentLogger


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_case(inputs, scene, frame, output):
    cameras, surfaces = inputs.load(scene)
    masks = {}
    hashes = {}
    for name in NAMES:
        if frame is None:
            folder = ROOT / 'masks/actual/static' / f'r0__{scene}__512__{name}'
        else:
            folder = ROOT / 'masks/actual/tracking' / f'r0__10i_s1_0531_track_t000__512__{name}'
        path = folder / 'masks.npz'
        hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            masks[name] = data['masks'].copy() if frame is None else data['masks'][frame].copy()
        assert masks[name].dtype == bool and masks[name].shape == (2, 512, 512)
    # Exact old CPU filter, outside the gate timer. No model or CUDA operation.
    clean = {}
    for camera in cameras:
        classification = core.classify_robot(camera, surfaces)
        clean[camera.camera], _ = core.filtered_masks(masks[camera.camera], classification)
    packed = {}
    for camera in cameras:
        for key in ('rgb', 'depth_raw_mm', 'intrinsic', 'extrinsic_cv', 'cam2world_gl', 'tcp_pose'):
            packed[camera.camera + '__' + key] = getattr(camera, key)
        packed[camera.camera + '__raw'] = masks[camera.camera]
        packed[camera.camera + '__clean'] = clean[camera.camera]
    destination = output / (scene + '_gate_input.npz')
    np.savez_compressed(destination, **packed)
    wrist = next(c for c in cameras if c.camera == 'hand_camera')
    metadata = dict(scene=scene, timestep=wrist.timestep, saved_mask_frame=frame,
                    mask_files_sha256=hashes, packed_input=destination.name,
                    packed_input_sha256=sha(destination),
                    masks={name: dict(raw_sha256=array_sha(masks[name]),
                                      clean_sha256=array_sha(clean[name]),
                                      raw_pixels=masks[name].sum(axis=(1, 2)).tolist(),
                                      clean_pixels=clean[name].sum(axis=(1, 2)).tolist())
                           for name in NAMES})
    return (wrist, masks[wrist.camera], clean[wrist.camera], cameras, clean), metadata


def measure(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    assert core.components.__module__ == 'graph.plan_a_geometry.core'
    assert 'query_pairs' in inspect.getsource(core.components)
    assert 'connected_components' in inspect.getsource(core.components)
    inputs = Inputs()
    protocol_path = Path(core.__file__).with_name('protocol.json')
    gate_config = json.loads(protocol_path.read_text())['gate']
    cases = [load_case(inputs, '10i_s1_0921_t080', None, output),
             load_case(inputs, '10i_s1_0531_track_t011', 11, output)]
    source_paths = [Path(__file__), Path(core.__file__)]
    from graph.persistent_geometry import contracts, fusion, self_filter
    source_paths += [Path(module.__file__) for module in (contracts, fusion, self_filter)]
    source_paths = list(dict.fromkeys(source_paths))
    (output / 'source').mkdir()
    source_hashes = {}
    for index, path in enumerate(source_paths):
        name = f'{index:02d}_{path.name}'
        (output / 'source' / name).write_bytes(path.read_bytes())
        source_hashes[str(path)] = sha(path)
    config = dict(boundary='Unchanged CPU core.wrist_gate call only: full camera world_points, radius components, local support, cross-camera consistency, reasons/trace and accepted mask',
                  excluded='Sensor/mask loading, FK, robot filtering, input snapshots, SAM2/detection, D2H/H2D, Shape-First/Graph/action/simulator',
                  input='Frozen actual SAM2 masks r0 and CPU-filtered clean masks, original 512 two-camera sensors; no object GT',
                  warmups=args.warmups, repeats=args.repeats,
                  seed=42, gate_config=gate_config, protocol_sha256=sha(protocol_path),
                  cpu=next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), 'unknown'),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  torch=torch.__version__, torch_threads=torch.get_num_threads(),
                  environment={k: os.getenv(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                  cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                  affinity=sorted(os.sched_getaffinity(0)),
                  source_sha256=source_hashes, cases=[metadata for _, metadata in cases])
    write(output / 'config.json', config)
    for _, metadata in cases:
        print('INPUT', json.dumps(metadata), flush=True)
    with ExperimentLogger(team='integration', model='task10i_old_cpu_wrist_gate',
                          seed=42, job_type='evaluate', root=output / 'experiments', config=config) as logger:
        results = []
        for arguments, metadata in cases:
            call_args = (*arguments, gate_config)
            baseline_mask, baseline_trace = core.wrist_gate(*call_args)
            for _ in range(args.warmups):
                mask, trace = core.wrist_gate(*call_args)
                np.testing.assert_array_equal(mask, baseline_mask)
                assert trace == baseline_trace
            rows = []
            for repeat in range(args.repeats):
                start = time.perf_counter_ns()
                mask, trace = core.wrist_gate(*call_args)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                # Full-array/trace verification is intentionally outside the timer.
                np.testing.assert_array_equal(mask, baseline_mask)
                assert trace == baseline_trace
                rows.append(dict(repeat=repeat, wall_ms=elapsed_ms, exact=True))
                print(json.dumps(dict(scene=metadata['scene'], repeat=repeat, wall_ms=elapsed_ms)), flush=True)
            times = [row['wall_ms'] for row in rows]
            result = dict(**metadata, rows=rows, median_ms=statistics.median(times),
                          min_ms=min(times), max_ms=max(times),
                          accepted_mask_sha256=array_sha(baseline_mask), trace=baseline_trace)
            results.append(result)
            write(output / (metadata['scene'] + '_result.json'), result)
            logger.log({metadata['scene']: dict(median_ms=result['median_ms'], clean_points=baseline_trace['clean_points'])})
        write(output / 'results.json', dict(status='PASS', run_id=logger.run_id, cases=results))
        write(output / 'input_hashes.json', inputs.hashes)
        for path in [output / 'config.json', output / 'results.json', output / 'input_hashes.json']:
            logger.link_output(path)
        print('COMPLETE', json.dumps(dict(run_id=logger.run_id,
              medians={r['scene']: r['median_ms'] for r in results})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=15)
    measure(parser.parse_args())
