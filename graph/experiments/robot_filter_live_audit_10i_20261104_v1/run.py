"""Audit the actual CurrentGeometry robot-filter boundary; stop before the gate."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from threading import Event
from unittest.mock import patch
import unittest

import numpy as np
import torch

from graph.experiments.robot_sam_overlap_live_10i_20261103_v1.run import Inputs, ROOT, NAMES
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_gpu_io import NoReadback
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.persistent_geometry.tensor_camera import tensor_camera
from graph.persistent_geometry.tensor_fusion import depth_from_mm
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.plan_a_bc.tensor_ingress import TensorRobotPreparation
from utils.experiment_logger import ExperimentLogger


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(inputs, scene, frame):
    cameras, surfaces = inputs.load(scene)
    raw = {}
    for name in NAMES:
        group = 'static' if frame is None else 'tracking'
        origin = scene if frame is None else '10i_s1_0531_track_t000'
        with np.load(ROOT / f'masks/actual/{group}/r0__{origin}__512__{name}/masks.npz') as z:
            value = z['masks'] if frame is None else z['masks'][frame]
            raw[name] = torch.as_tensor(value.copy(), device='cuda')
    with np.load(ROOT / f'sensors/{scene}/512/robot.npz') as z:
        data = {k: z[k].copy() for k in z.files}
    data.update(episode=np.array(scene), timestep=np.int64(cameras[0].timestep),
                timestamp=np.float64(cameras[0].timestamp), qpos=data['robot76'][:9],
                robot_link_world=np.stack([s.world_from_link for s in surfaces]))
    reference = CudaRobotFilter().filter_cameras(cameras, surfaces, raw)
    geometry = CurrentGeometry('/workspace/task10g_f1_abc_20261031/full/inputs', inputs.fk.schema)
    assert geometry.robot_filter_backend == 'cuda'
    # Native image buffers match the new live sensors contract.
    native = tuple(tensor_camera(c) for c in cameras)
    started = Event()
    events = []
    captured = []
    original_prepare = CudaRobotFilter.prepare_surfaces
    original_filter = CudaRobotFilter.filter_cameras

    def prepare(shapes, copied_cameras=(), **kwargs):
        events.append('prepare_started')
        started.set()
        return original_prepare(shapes, copied_cameras, **kwargs)

    def filtering(self, copied_cameras, shapes, masks, **kwargs):
        assert kwargs.get('prepared') is not None
        assert all(kwargs['camera_tensors'][c.camera]['depth_cuda'].is_cuda for c in copied_cameras)
        events.append('filter')
        captured.append(kwargs['prepared'])
        return original_filter(self, copied_cameras, shapes, masks, **kwargs)

    def vision():
        events.append('vision_entered')
        assert started.wait(5), 'worker was not submitted before vision'
        return raw

    def handoff(copied_cameras, copied_data, shapes, masks, clean_masks=None, *, depths=None):
        assert clean_masks is not None and depths is not None
        assert all(masks[n] is raw[n] and clean_masks[n].is_cuda for n in NAMES)
        assert copied_cameras[0].episode == scene
        return clean_masks

    geometry._build_prepared = handoff  # Intentionally excludes wrist gate and Graph.
    try:
        with patch.object(CudaRobotFilter, 'prepare_surfaces', staticmethod(prepare)), \
                patch.object(CudaRobotFilter, 'filter_cameras', filtering), \
                patch('graph.plan_a_geometry.core.classify_robot', side_effect=AssertionError('duplicate CPU robot filter')):
            # No dense D2H, CUDA scalar materialization, dynamic nonzero or global sync.
            with NoReadback(), patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('global sync')):
                actual = geometry.build(native, data, surfaces, vision)
        assert events.count('prepare_started') == 1 and events.count('filter') == 1
        assert events.index('vision_entered') < events.index('filter')
        assert geometry.last_robot_schedule == 'before_vision'
        prep = geometry.robot_preparation
        assert isinstance(prep, TensorRobotPreparation) and prep._pending is None
        filt = prep.robot_filter
        assert (filt._compact, filt._clip_pixels, filt._depth_from_mm) == (True, 32, True)
        for name in NAMES:
            torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
        # A second build must create fresh pose-dependent preparation, reusing rays.
        ray_ids = {k: v.data_ptr() for k, v in filt._rays.items()}
        with patch.object(CudaRobotFilter, 'prepare_surfaces', staticmethod(prepare)), \
                patch.object(CudaRobotFilter, 'filter_cameras', filtering):
            again = geometry.build(native, data, surfaces, vision)
        assert captured[0] is not captured[1]
        assert ray_ids == {k: v.data_ptr() for k, v in filt._rays.items()}
        for name in NAMES:
            torch.testing.assert_close(again[name], reference[name], rtol=0, atol=0)
        # Vision failures drain pending work; the same live object remains usable.
        def failure():
            raise RuntimeError('injected vision failure')
        try:
            geometry.build(native, data, surfaces, failure)
        except RuntimeError as exc:
            assert str(exc) == 'injected vision failure'
        else:
            raise AssertionError('expected vision failure')
        assert prep._pending is None
        recovered = geometry.build(native, data, surfaces, vision)
        for name in NAMES:
            torch.testing.assert_close(recovered[name], reference[name], rtol=0, atol=0)
        return dict(scene=scene, size=512, cameras=2, raw_clean_cuda=True, output_bitwise_exact=True,
                    single_shared_preparation=True, filter_batch_once=True, compact=True, clip_pixels=32,
                    prepared_depth_cuda=True, rays_reused=True, pose_preparation_refreshed=True,
                    vision_failure_recovered=True, no_readback_guard=True, gate_and_graph_executed=False)
    finally:
        geometry.close()


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = dict(scope='robot filter only; actual default CurrentGeometry native CUDA input to pre-gate handoff',
                  source_sha256={str(Path(p).resolve()): sha(p) for p in [__file__, inspect.getfile(CudaRobotFilter),
                      inspect.getfile(TensorRobotPreparation), inspect.getfile(CurrentGeometry), inspect.getfile(depth_from_mm)]},
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
    with ExperimentLogger(team='integration', model='task10i_robot_filter_live_audit',
                          job_type='evaluate', seed=42, root=output / 'experiments', config=config) as logger:
        names = ['graph.plan_a_bc.test_robot_overlap.DefaultRobotOverlap',
                 'graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_gpu_io',
                 'graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_preparation',
                 'graph.integration_baseline.tests.test_online_vision']
        with (output / 'tests.log').open('w') as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
        assert tests.wasSuccessful()
        inputs = Inputs()
        rows = [probe(inputs, '10i_s1_0921_t080', None), probe(inputs, '10i_s1_0531_track_t011', 11)]
        # The shared live lookup matches the filter's fused lookup for every int16 value.
        raw = torch.arange(-32768, 32768, device='cuda', dtype=torch.int32).to(torch.int16).reshape(256, 256)
        filt = CudaRobotFilter()
        a, b = depth_from_mm(raw), filt.depth_tensor(raw)
        assert torch.equal(torch.isnan(a), torch.isnan(b))
        assert torch.equal(a[torch.isfinite(a)], b[torch.isfinite(b)])
        # Reproduce the legacy live caller's CUDA-RGB / CPU-adapter mismatch before model inference.
        legacy = OnlineVisionAdapter(None, None)
        try:
            try:
                legacy.process(torch.zeros((4, 4, 3), device='cuda', dtype=torch.uint8), legacy.identity, 0)
            except TypeError as exc:
                legacy_error = str(exc)
                assert 'numpy' in legacy_error.lower()
            else:
                raise AssertionError('legacy CPU adapter unexpectedly accepted CUDA RGB')
        finally:
            legacy.close()
        result = dict(status='PASS', run_id=logger.run_id, config=config,
                      tests=dict(run=tests.testsRun, skipped=len(tests.skipped), failures=len(tests.failures), errors=len(tests.errors)),
                      native_cases=rows, shared_vs_fused_depth_all_int16_exact=True,
                      legacy_cuda_rgb_cpu_adapter_error=legacy_error)
        path = output / 'results.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        logger.link_output(path)
        logger.link_output(output / 'tests.log')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
