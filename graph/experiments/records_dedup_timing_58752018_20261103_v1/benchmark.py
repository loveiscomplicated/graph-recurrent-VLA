"""Unchanged PlanA SF Python records and 3mm deduplication CPU baseline."""
import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.map import current_records
from graph.persistent_geometry.tensor_fusion import (
    prepare_tensor_views, observations_from_tensor_packet,
    consistent_views_tensor, numpy_observations)
from graph.plan_a_geometry.core import PlanA, deduplicate, key
from visual.runtime.graph_perception import make_packet_cuda


def same(a, b):
    if is_dataclass(a):
        assert type(a) is type(b)
        same(asdict(a), asdict(b))
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else:
        assert a == b, (a, b)


def stats(values):
    return dict(median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                min_ms=float(np.min(values)), max_ms=float(np.max(values)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmups', type=int, default=20)
    p.add_argument('--repeats', type=int, default=100)
    a = p.parse_args()
    assert os.environ.get('CONTAINER_ID') == '58752018' and torch.cuda.is_available()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip()
    if active: raise RuntimeError('another CUDA process is active; do not stop it')
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    # Source root is resolved from the imported production module, not the runner location.
    import graph.plan_a_geometry.core as production
    root = Path(production.__file__).resolve().parents[2]
    protocol = json.loads((root/'graph/plan_a_geometry/protocol.json').read_text())
    distance = protocol['merge_distance_m']
    assert distance == .003
    def load(name):
        with np.load(a.inputs/name, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}
    sensor = load('hand_camera.npz'); robot = load('robot.npz')
    mask = load('masks.npz')['masks']
    hand = CameraObservation('10i_s1_0844', 80, 'hand_camera', 4.,
        sensor['rgb'], sensor['depth_raw_mm'], sensor['intrinsic_cv'],
        sensor['extrinsic_cv'], sensor['cam2world_gl'], robot['tcp_pose']).validate()
    assert hand.depth.shape == (512, 512)
    cameras = (replace(hand, camera='base_camera'), hand)
    cm = dict(base_camera=np.zeros_like(mask), hand_camera=mask)
    frames = ('ee', 'world')
    inputs, packets = {}, {}
    for frame in frames:
        ctx = prepare_tensor_views(cameras, cm, coordinate_frame=frame)
        inputs[frame] = {}; packets[frame] = {}
        for camera in cameras:
            v = ctx.by_name[camera.camera]
            packet = make_packet_cuda(camera, v.mask, depth_cuda=v.depth.to(torch.float32),
                intrinsic_cuda=v.intrinsic, transform_cuda=v.frame_from_camera,
                coordinate_frame=frame)
            observations = observations_from_tensor_packet(packet, camera)
            observations, _ = consistent_views_tensor(observations, ctx)
            inputs[frame][camera.camera] = numpy_observations(observations, include_structural=True)
            packets[frame][camera.camera] = packet
    torch.cuda.synchronize()

    def records(frame):
        # Exact statements from PlanA.represent's SF branch after numpy_observations.
        sf = []
        for camera in cameras:
            observations, flags = inputs[frame][camera.camera]
            structural_pixels = {int(p) for o, f in zip(observations, flags) for p in o.pixels[f]}
            rows = current_records(observations, fuse_views=False, coordinate_frame=frame)
            for r in rows: r['structural'] = key(r)[3] in structural_pixels
            sf.extend(rows)
        return sf

    def whole(frame):
        return deduplicate(records(frame), distance)

    checks = {}; before_inputs = deepcopy(inputs)
    for frame in frames:
        before_rows = records(frame); saved_rows = deepcopy(before_rows)
        expected, audit = deduplicate(before_rows, distance)
        same(before_rows, saved_rows)
        system = PlanA('base_camera', 'hand_camera', protocol, coordinate_frame=frame,
            tensor_packet_builder=lambda c, m, **kw: packets[frame][c.camera])
        pools, audits, _, _ = system.represent(cameras, {'conditions': {'C3': cm}}, 'C3', ('SF',))
        same(expected, pools['SF']); same(audit, audits['SF'])
        checks[frame] = dict(public_plan_a_sf_all_record_fields_exact=True,
            public_plan_a_sf_audit_exact=True, dedup_input_unchanged=True,
            input_points=len(before_rows), input_structural=sum(r['structural'] for r in before_rows),
            input_surface=sum(not r['structural'] for r in before_rows),
            retained_structural=sum(r['structural'] for r in expected),
            retained_source_ids=sum(len(r['source_observation_ids']) for r in expected),
            audit=audit)
    same(inputs, before_inputs)
    # No GPU operation or readback is part of the CPU timed endpoints.
    torch.cuda.synchronize()
    config = dict(task='10i_records_dedup_timing_20261103', warmups=a.warmups,
        repeats=a.repeats, frames=list(frames), merge_distance_m=distance,
        endpoint='CPU observations + aligned structural flags ready -> SF deduplicated records',
        includes=['structural pixel sets', 'current_records', 'structural marking',
                  'camera row concatenation', 'SF deduplicate'],
        excludes=['GPU preparation', 'numpy_observations readback', 'final 128-point FPS',
                  'legacy/hybrid diagnostic branches', 'graph edges', 'robot filtering/gate'],
        actual_sensor_views=1, empty_cloned_base=True,
        input_sha256={x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in a.inputs.glob('*.npz')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    logger = ExperimentLogger(team='graph', job_type='benchmark', seed=42,
        root=a.output/'experiments', config=config)
    try:
        for _ in range(a.warmups):
            for frame in frames: whole(frame)
        timings = {f'{frame}__{mode}': [] for frame in frames
                   for mode in ('whole', 'records_diagnostic', 'dedup_diagnostic')}
        # The primary whole endpoint has no intermediate clocks.
        # Outputs from previous repetitions are released OUTSIDE the next timed call.
        for i in range(a.repeats):
            for frame in (frames if i % 2 == 0 else frames[::-1]):
                start = perf_counter(); result = whole(frame)
                elapsed = (perf_counter() - start) * 1000
                timings[f'{frame}__whole'].append(elapsed)
                del result
        for _ in range(a.warmups):
            for frame in frames:
                rows = records(frame); deduplicate(rows, distance)
        del rows
        for i in range(a.repeats):
            for frame in (frames[::-1] if i % 2 == 0 else frames):
                start = perf_counter(); rows = records(frame)
                split = perf_counter(); result = deduplicate(rows, distance)
                end = perf_counter()
                timings[f'{frame}__records_diagnostic'].append((split-start)*1000)
                timings[f'{frame}__dedup_diagnostic'].append((end-split)*1000)
                del result, rows
        same(inputs, before_inputs)
        for frame in frames:
            got, audit = whole(frame)
            same(audit, checks[frame]['audit'])
            checks[frame]['observations_and_structural_flags_unchanged'] = True
            # Preserve representative order and every merged source ID for later comparison.
            (a.output/f'{frame}_provenance.json').write_text(json.dumps([
                dict(persistent_node_id=r['persistent_node_id'], object_id=r['object_id'],
                    structural=r['structural'], sources=r['source_observation_ids']) for r in got], indent=2)+'\n')
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), 'unknown')
        summary = dict(run_id=logger.run_id, instance='58752018', resolution=[512,512],
            config=config, verification=checks,
            timings={k: stats(v) for k,v in timings.items()},
            camera_input_counts={frame: {c.camera: sum(len(o.xyz) for o in inputs[frame][c.camera][0])
                                       for c in cameras} for frame in frames},
            cpu=cpu, gpu_preparation=torch.cuda.get_device_name(), python=platform.python_version(),
            torch=torch.__version__, numpy=np.__version__, torch_threads=torch.get_num_threads(),
            environment_threads={k: os.environ.get(k) for k in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            timed_device='cpu', timer='perf_counter wall clock',
            cpu_affinity_fixed=False, garbage_collection_default=True)
        (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        np.savez(a.output/'timings.npz', **{k: np.asarray(v) for k,v in timings.items()})
        logger.record(summary=summary)
        for path in a.output.glob('*'):
            if path.is_file(): logger.link_output(path)
        logger.finish(); print(json.dumps(summary, indent=2))
    except BaseException:
        logger.finish(failed=True); raise


if __name__ == '__main__': main()
