"""Paired unchanged NumPy / Torch CUDA view checks; explicit endpoint costs."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.persistent_geometry.contracts import CameraObservation
from graph.persistent_geometry.fusion import observations_from_packet, consistent_views
from graph.persistent_geometry.tensor_fusion import (prepare_tensor_views, tensor_observations,
    consistent_views_tensor, numpy_observations)
from graph.plan_a_bc.geometry import CurrentGeometry
from visual.runtime.graph_perception import make_packet


def same_observations(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        for key, value in asdict(y).items():
            other = getattr(x, key)
            if isinstance(value, np.ndarray):
                assert other.dtype == value.dtype
                np.testing.assert_array_equal(other, value)
            else:
                assert other == value, key


def stats(values):
    return dict(median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                min_ms=float(np.min(values)), max_ms=float(np.max(values)))


def verify_graph(cameras, cm, data, assets, schema):
    result = {}
    for frame in ('world', 'ee'):
        new = CurrentGeometry(assets, schema, coordinate_frame=frame)
        old = CurrentGeometry(assets, schema, coordinate_frame=frame)
        old.system.tensor_packet_builder = None  # Exact prior SF/NumPy-view route.
        new.fk.verify_robot(data['robot76'], data['robot_world_geometry'], data['tcp_pose'])
        surfaces = new.fk.surfaces(data['robot76'])
        for geo in (new, old):
            geo.system.prepare = lambda *args: dict(conditions={'C3': cm}, gate=dict(accepted=True))
        b, br, bg = old.build(cameras, data, surfaces, cm)
        a, ar, ag = new.build(cameras, data, surfaces, cm)
        assert a.packet['view_consistency_backend'] == 'torch_cuda'
        assert b.packet['view_consistency_backend'] == 'numpy_cpu'
        assert a.packet['node_ids'] == b.packet['node_ids'] and ag == bg
        assert [r['source_observation_ids'] for r in ar] == [r['source_observation_ids'] for r in br]
        for name in ('node_type', 'edge_index', 'edge_type'):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        for name in ('node_input', 'edge_attr'):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=0, atol=2e-5)
        result[frame] = dict(passed=True, nodes=len(a.node_type), edges=a.edge_index.shape[1],
            discrete_and_provenance_exact=True, max_node_abs=float((a.node_input-b.node_input).abs().max()),
            max_edge_abs=float((a.edge_attr-b.edge_attr).abs().max()))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True); p.add_argument('--schema', type=Path, required=True)
    p.add_argument('--warmups', type=int, default=20); p.add_argument('--repeats', type=int, default=100)
    a = p.parse_args()
    assert os.environ.get('CONTAINER_ID') == '58752018' and torch.cuda.is_available()
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    if active: raise RuntimeError('another CUDA process is active; do not stop it')
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    def load(name):
        with np.load(a.inputs/name, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
    s = load('hand_camera.npz'); data = load('robot.npz'); mask = load('masks.npz')['masks']
    c = CameraObservation('10i_s1_0844', 80, 'hand_camera', 4., s['rgb'], s['depth_raw_mm'],
        s['intrinsic_cv'], s['extrinsic_cv'], s['cam2world_gl'], data['tcp_pose']).validate()
    cameras = (replace(c, camera='base_camera'), c)
    cm = dict(base_camera=np.zeros_like(mask), hand_camera=mask)
    observations = {}; gpu = {}; views = {}
    for frame in ('world','ee'):
        observations[frame] = {}
        for camera in cameras:
            packet, _ = make_packet(camera, cm[camera.camera], coordinate_frame=frame)
            observations[frame][camera.camera] = observations_from_packet(packet, camera, coordinate_frame=frame)
        gpu[frame] = {name: tensor_observations(os, device='cuda') for name, os in observations[frame].items()}
        views[frame] = prepare_tensor_views(cameras, cm, coordinate_frame=frame)
    torch.cuda.synchronize()
    modes = ('numpy', 'torch_resident', 'torch_with_sensor_upload', 'torch_upload_and_cpu_output')
    def run(frame, mode):
        if mode == 'numpy':
            return [consistent_views(observations[frame][c.camera], cameras, cm) for c in cameras]
        ctx = views[frame] if mode == 'torch_resident' else prepare_tensor_views(cameras, cm, coordinate_frame=frame)
        output = []
        for c in cameras:
            out, audit = consistent_views_tensor(gpu[frame][c.camera], ctx)
            if mode == 'torch_upload_and_cpu_output': out = numpy_observations(out)
            output.append((out, audit))
        return output
    logger = ExperimentLogger(team='graph', job_type='benchmark', seed=42, root=a.output/'experiments',
        config=dict(task='10i_view_consistency_cuda_20261103', warmups=a.warmups, repeats=a.repeats,
        modes=list(modes), actual_sensor_views=1, empty_cloned_base=True, robot_filter_excluded=True,
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.inputs.glob('*.npz')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        endpoint='two per-camera calls; CUDA endpoint synchronized; input GPU observations ready for CUDA modes'))
    try:
        for _ in range(a.warmups):
            for frame in observations:
                for mode in modes: run(frame, mode)
        torch.cuda.synchronize()
        times = {frame+'__'+mode: [] for frame in observations for mode in modes}
        combinations = list(times)
        for i in range(a.repeats):
            order = combinations[i%len(combinations):]+combinations[:i%len(combinations)]
            if i%2: order = order[::-1]
            for key in order:
                frame, mode = key.split('__')
                torch.cuda.synchronize(); start = perf_counter(); output = run(frame, mode)
                if mode != 'numpy': torch.cuda.synchronize()
                times[key].append((perf_counter()-start)*1000)
        for frame in observations:
            expected = run(frame, 'numpy')
            for mode in modes[1:]:
                actual = run(frame, mode)
                for (selected, trace), (ref, audit) in zip(actual, expected):
                    host = selected if mode == 'torch_upload_and_cpu_output' else numpy_observations(selected)
                    same_observations(host, ref)
                    assert [int(r['unknown_conflict'].item()) for r in trace] == [r['unknown_conflict'] for r in audit]
                    if mode != 'torch_upload_and_cpu_output': assert all(o.xyz.is_cuda for o in selected)
        data.update(episode=np.array(c.episode), timestep=np.int64(c.timestep), timestamp=np.float64(c.timestamp))
        graph = verify_graph(cameras, cm, data, a.assets, json.loads(a.schema.read_text())['schema'])
        summary = dict(run_id=logger.run_id, instance='58752018', resolution=[512,512],
            actual_sensor_views=1, empty_cloned_base=True,
            hand_points=sum(len(o.xyz) for o in observations['ee']['hand_camera']),
            base_points=sum(len(o.xyz) for o in observations['ee']['base_camera']),
            calls_per_sample=2, all_observation_fields_and_audits_exact=True, graph_verification=graph,
            timings={name: stats(t) for name,t in times.items()}, warmups=a.warmups, repeats=a.repeats,
            gpu=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__,
            resident_preparation_excluded=True, upload_modes_include_sensor_validation_and_upload=True,
            cuda_modes_input_observations_already_on_gpu=True)
        (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        np.savez(a.output/'timings.npz', **{k: np.asarray(v) for k,v in times.items()})
        logger.record(summary=summary)
        logger.link_output(a.output/'summary.json'); logger.link_output(a.output/'timings.npz')
        logger.finish(); print(json.dumps(summary, indent=2))
    except BaseException:
        logger.finish(failed=True); raise


if __name__ == '__main__': main()
