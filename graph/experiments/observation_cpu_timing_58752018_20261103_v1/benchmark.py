"""Existing ready EE packet -> per-object World SurfaceObservation baseline."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import torch

from graph.persistent_geometry.contracts import CameraObservation, SurfaceObservation, transform
from graph.persistent_geometry import fusion
from graph.persistent_geometry.fusion import observations_from_packet
from utils.experiment_logger import ExperimentLogger
from visual.runtime.graph_perception import make_packet
from visual.runtime.vision_packet import pose_matrix


def load_case(inputs):
    with np.load(inputs/'hand_camera.npz', allow_pickle=False) as a:
        sensor = {k: a[k].copy() for k in a.files}
    with np.load(inputs/'robot.npz', allow_pickle=False) as a:
        tcp = a['tcp_pose'].copy()
    with np.load(inputs/'masks.npz', allow_pickle=False) as a:
        masks = a['masks'].copy()
    camera = CameraObservation('10i_s1_0844', 80, 'hand_camera', 4., sensor['rgb'],
        sensor['depth_raw_mm'], sensor['intrinsic_cv'], sensor['extrinsic_cv'],
        sensor['cam2world_gl'], tcp).validate()
    return camera, masks


def assert_same(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b, (a, b)


def diagnostic(packet, camera):
    """Same expressions/order as the public helper; clocks only in this copy.

    Exact parity of every returned field is checked outside primary timing.
    Transform's own SE(3) validation belongs to the point-transform phase.
    """
    times = dict(setup=0., split_and_gather=0., normal_transform=0.,
                 point_transform=0., construct_and_validate=0.)
    start = perf_counter()
    if packet['camera'] != camera.camera or packet['source_frame'] != camera.timestep:
        raise ValueError('packet/sensor identity mismatch')
    world_from_ee = pose_matrix(camera.tcp_pose)
    result = []
    times['setup'] = perf_counter()-start
    for oid in (0, 1):
        start = perf_counter()
        rows = packet['object_id'] == oid
        f5 = np.asarray(packet['geometric_features'][rows]).copy()
        valid = packet['feature_field_valid'][rows].copy()
        times['split_and_gather'] += perf_counter()-start
        start = perf_counter()
        normals = np.zeros((int(rows.sum()), 3))
        normals[valid[:, 0]] = f5[valid[:, 0], :3] @ world_from_ee[:3, :3].T
        f5[:, :3] = np.where(valid[:, :1], normals, np.nan)
        times['normal_transform'] += perf_counter()-start
        start = perf_counter()
        xyz = transform(packet['points'][rows], world_from_ee)
        times['point_transform'] += perf_counter()-start
        start = perf_counter()
        observation = SurfaceObservation(camera.episode, camera.timestep, camera.camera, oid,
            xyz, normals, valid[:, 0].copy(), f5, valid,
            packet['provenance']['source_index'][rows].copy(),
            camera.world_from_camera[:3, 3].copy())
        result.append(observation.validate())
        times['construct_and_validate'] += perf_counter()-start
    return result, {key: value*1000 for key, value in times.items()}


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
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('another CUDA compute process is active; do not stop it')
    args.output.mkdir(parents=True, exist_ok=False)
    camera, masks = load_case(args.inputs)
    # Projection, sampling, transfer, packet completion and receiver all excluded.
    packet, _ = make_packet(camera, masks)
    torch.cuda.synchronize()
    packet_before = deepcopy(packet)
    root = Path(fusion.__file__).resolve().parents[2]
    names = ('graph/persistent_geometry/fusion.py', 'graph/persistent_geometry/contracts.py',
             'visual/runtime/graph_perception.py', 'visual/runtime/vision_packet.py',
             'graph/vision_input.py')
    config = dict(task='10i_observation_cpu_timing_58752018_20261103_v1',
        scope='ready existing CPU EE packet -> two validated World SurfaceObservations',
        warmups=args.warmups, repeats=args.repeats,
        input_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.inputs.glob('*.npz')},
        source_hashes={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        robot_filter_excluded=True, multi_view_consistency_excluded=True)
    logger = ExperimentLogger(team='graph', job_type='benchmark', config=config,
        seed=42, root=args.output/'experiments')
    try:
        for _ in range(args.warmups):
            observations_from_packet(packet, camera)
        primary = []
        for _ in range(args.repeats):
            start = perf_counter()
            observations = observations_from_packet(packet, camera)
            primary.append((perf_counter()-start)*1000)
        phases = {key: [] for key in diagnostic(packet, camera)[1]}
        for _ in range(args.repeats):
            actual, times = diagnostic(packet, camera)
            for name, value in times.items():
                phases[name].append(value)
        assert_same([asdict(o) for o in actual], [asdict(o) for o in observations])
        assert_same(packet_before, packet)
        # Independent geometry sanity check; EE float32 rounding is retained.
        for o in observations:
            rows = packet['object_id'] == o.object_id
            expected = transform(packet['camera_points'][rows], camera.world_from_camera)
            np.testing.assert_allclose(o.xyz, expected, atol=2e-6, rtol=0)
        summary = dict(run_id=logger.run_id, instance=os.environ['CONTAINER_ID'],
            cpu=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                     if line.startswith('model name')), gpu=torch.cuda.get_device_name(),
            python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
            resolution=list(camera.depth.shape), input_camera=camera.camera,
            mask_pixels=masks.sum((1, 2)).tolist(), selected_points=len(packet['points']),
            output_points_by_object={str(o.object_id): len(o.xyz) for o in observations},
            input_frame=packet['coordinate_frame'], output_frame='World_metres',
            warmups=args.warmups, repeats=args.repeats,
            primary_observations_from_packet=stats(primary),
            diagnostic_phases={name: stats(v) for name, v in phases.items()},
            public_vs_instrumented_all_fields_exact=True, input_packet_unchanged=True,
            camera_to_world_check_atol_m=2e-6, includes_empty_box_path=True,
            multi_view_consistency_executed=False, runtime_modified=False)
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        np.savez(args.output/'timings.npz', observations_from_packet=np.asarray(primary))
        np.savez(args.output/'phase_timings.npz', **{name: np.asarray(v) for name, v in phases.items()})
        np.savez_compressed(args.output/'observations.npz', **{
            f'{o.object_id}_{name}': getattr(o, name) for o in observations
            for name in ('xyz', 'normals', 'normal_valid', 'f5', 'feature_valid', 'pixels', 'camera_origin')})
        logger.record(summary=summary)
        for path in args.output.glob('*.*'):
            logger.link_output(path)
        logger.finish()
        print(json.dumps(summary, indent=2))
    except BaseException:
        logger.finish(failed=True)
        raise


if __name__ == '__main__':
    main()
