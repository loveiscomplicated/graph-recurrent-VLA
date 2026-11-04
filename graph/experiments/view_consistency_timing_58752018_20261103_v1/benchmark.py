"""Unmodified consistent_views timing on the preceding real512 input boundary.

Only one recorded sensor is available: base is an explicitly empty cloned view.
Do not describe this case as two real cameras with nonempty observations.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
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
from graph.persistent_geometry.fusion import observations_from_packet, consistent_views
from visual.runtime.graph_perception import make_packet


def same(a, b):
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
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
    args = p.parse_args()
    assert os.environ.get('CONTAINER_ID') == '58752018' and torch.cuda.is_available()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('another CUDA compute process is active; do not stop it')
    args.output.mkdir(parents=True, exist_ok=False)
    def load(name):
        with np.load(args.inputs/name, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}
    sensor = load('hand_camera.npz'); tcp = load('robot.npz')['tcp_pose']; mask = load('masks.npz')['masks']
    hand = CameraObservation('10i_s1_0844', 80, 'hand_camera', 4., sensor['rgb'], sensor['depth_raw_mm'],
        sensor['intrinsic_cv'], sensor['extrinsic_cv'], sensor['cam2world_gl'], tcp).validate()
    cameras = (replace(hand, camera='base_camera'), hand)
    masks = dict(base_camera=np.zeros_like(mask), hand_camera=mask)
    observations = {}
    # All CUDA sampling, D2H, packet completion and observation conversion are
    # outside every timer. Keep the regular, uncached CameraObservation class.
    for frame in ('world', 'ee'):
        observations[frame] = {}
        for c in cameras:
            packet, _ = make_packet(c, masks[c.camera], coordinate_frame=frame)
            observations[frame][c.camera] = observations_from_packet(packet, c, coordinate_frame=frame)
    torch.cuda.synchronize()
    before = deepcopy(observations)
    source = Path(consistent_views.__code__.co_filename).resolve()
    logger = ExperimentLogger(team='graph', job_type='benchmark', seed=42,
        root=args.output/'experiments', config=dict(task='10i_view_consistency_timing_20261103',
        scope='ready CPU observations -> existing consistent_views for base then hand, matching PlanA call pattern',
        actual_sensor_views=1, empty_cloned_base=True, robot_filter_excluded=True,
        warmups=args.warmups, repeats=args.repeats,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs.glob('*.npz')}))
    def run(frame):
        # Same two existing calls as PlanA.represent, no combined/replaced API.
        return [consistent_views(observations[frame][c.camera], cameras, masks) for c in cameras]
    try:
        for _ in range(args.warmups):
            for frame in observations:
                run(frame)
        times = {frame: [] for frame in observations}
        for i in range(args.repeats):
            for frame in (('world', 'ee') if i % 2 == 0 else ('ee', 'world')):
                start = perf_counter()
                output = run(frame)
                times[frame].append((perf_counter()-start)*1000)
        details = {}
        for frame in observations:
            output = run(frame)
            audit = []
            for c, (checked, trace) in zip(cameras, output):
                expected = observations[frame][c.camera]
                same([asdict(o) for o in checked], [asdict(o) for o in expected])
                same([asdict(o) for o in expected], [asdict(o) for o in before[frame][c.camera]])
                assert all(r['unknown_conflict'] == 0 for r in trace)
                audit.extend(trace)
            details[frame] = dict(timing=stats(times[frame]),
                input_points_by_camera={name: [len(o.xyz) for o in os] for name, os in observations[frame].items()},
                audit=audit, all_output_fields_exact=True, input_observations_unchanged=True)
        summary = dict(run_id=logger.run_id, instance=os.environ['CONTAINER_ID'],
            resolution=list(hand.depth.shape), actual_sensor_views=1, empty_cloned_base=True,
            coordinate_frames=details, calls_per_sample=2, warmups=args.warmups, repeats=args.repeats,
            cpu=next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
            python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
            measured_backend='existing CPU NumPy consistent_views', runtime_modified=False,
            includes_synchronized_camera_validation=True, includes_full_depth_property_conversion=True,
            includes_result_subset_copy=True, excludes_projection_sampling_packet_transfer_observation_creation=True)
        path = args.output/'summary.json'; path.write_text(json.dumps(summary, indent=2)+'\n')
        np.savez(args.output/'timings.npz', **{key: np.array(t) for key, t in times.items()})
        logger.record(summary=summary)
        logger.link_output(path); logger.link_output(args.output/'timings.npz')
        logger.finish(); print(json.dumps(summary, indent=2))
    except BaseException:
        logger.finish(failed=True); raise


if __name__ == '__main__':
    main()
