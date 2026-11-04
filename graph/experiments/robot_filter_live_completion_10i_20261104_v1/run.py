"""Validate the four live robot-filter integration fixes on existing CUDA inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from graph.experiments.robot_sam_overlap_live_10i_20261103_v1.run import Inputs, ROOT, NAMES
from graph.experiments.robot_filter_live_audit_10i_20261104_v1.run import probe
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
from graph.persistent_geometry.self_filter import classify_robot
from graph.persistent_geometry.fusion import filtered_masks
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.task10i_stage4a.perframe_cache import install
from graph.task10i_stage4a2.frozen_inference import infer_frame
from utils.experiment_logger import ExperimentLogger


def correction_case(inputs, scene, index):
    cameras, surfaces = inputs.load(scene)
    with np.load(ROOT / f'sensors/{scene}/512/robot.npz') as z:
        data = {k: z[k].copy() for k in z.files}
    data.update(episode=np.array(scene), timestep=np.int64(cameras[0].timestep),
                timestamp=np.float64(cameras[0].timestamp), qpos=data['robot76'][:9],
                robot_link_world=np.stack([s.world_from_link for s in surfaces]))
    raw = {}
    for name in NAMES:
        group = 'static' if index is None else 'tracking'
        origin = scene if index is None else '10i_s1_0531_track_t000'
        with np.load(ROOT / f'masks/actual/{group}/r0__{origin}__512__{name}/masks.npz') as z:
            raw[name] = z['masks'].copy() if index is None else z['masks'][index].copy()
    proposals = {name: value.copy() for name, value in raw.items()}
    proposals['hand_camera'][0, :4, :4] = True
    diagnostics = {c.camera: classify_robot(c, surfaces) for c in cameras}
    geometry = CurrentGeometry('/workspace/task10g_f1_abc_20261031/full/inputs', inputs.fk.schema)
    expected = geometry.system.prepare(cameras, data, proposals, surfaces)
    memo = install(geometry)
    native_raw = {name: torch.as_tensor(value, device='cuda') for name, value in raw.items()}
    started = Event()
    counts = dict(prepare=0, batch=0, fused_depth=0)
    native_prepare = CudaRobotFilter.prepare_surfaces
    native_filter = CudaRobotFilter.filter_cameras
    native_depth = CudaRobotFilter.depth_tensor
    def prepare(*args, **kwargs):
        counts['prepare'] += 1
        started.set()
        return native_prepare(*args, **kwargs)
    def filtering(self, *args, **kwargs):
        counts['batch'] += 1
        assert kwargs['diagnostics'] is True
        return native_filter(self, *args, **kwargs)
    def depth(self, value):
        counts['fused_depth'] += 1
        return native_depth(self, value)
    class Adapter:
        found = {}
        def __init__(self, name):
            self.name = name
        def process(self, rgb, episode, t, *, mask_device):
            assert mask_device == 'cuda' and started.wait(5)
            return native_raw[self.name], dict(new_object_ids=[])
    max_depth_error = []
    owner_differences = {}
    class Correction:
        def process(self, supplied_cameras, supplied_data, supplied_surfaces, host, found, *, robot_frame):
            assert robot_frame is not None
            for name in raw:
                np.testing.assert_array_equal(host[name], raw[name])
                assert not host[name].flags.writeable
            first = robot_frame.prepare(supplied_cameras, supplied_data, host, supplied_surfaces)
            proposed = robot_frame.prepare(supplied_cameras, supplied_data, proposals, supplied_surfaces)
            repeated = robot_frame.prepare(supplied_cameras, supplied_data, proposals, supplied_surfaces)
            for name in raw:
                for key in ('confirmed_robot', 'uncertain', 'candidate', 'inside_known_link_volume'):
                    np.testing.assert_array_equal(first['filters'][name][key], diagnostics[name][key])
                # Correction never reads link_owner. Retain differences against
                # NumPy's rounding/order as diagnostic evidence; GPU owner/tie
                # behavior is separately covered by the original filter suite.
                owner_differences[name] = int(np.count_nonzero(
                    first['filters'][name]['link_owner'] != diagnostics[name]['link_owner']))
                actual_depth = first['filters'][name]['predicted_robot_depth']
                target_depth = diagnostics[name]['predicted_robot_depth']
                np.testing.assert_allclose(actual_depth, target_depth, rtol=0, atol=1e-12)
                finite = np.isfinite(target_depth)
                max_depth_error.append(float(np.max(np.abs(actual_depth[finite]-target_depth[finite]))) if finite.any() else 0.)
                np.testing.assert_array_equal(proposed['clean'][name], expected['clean'][name])
                np.testing.assert_array_equal(repeated['clean'][name], expected['clean'][name])
            assert proposed['gate'] == expected['gate']
            return proposals
    def handoff(copied_cameras, copied_data, copied_surfaces, masks, clean_masks=None, **kwargs):
        for name in raw:
            assert masks[name].is_cuda and clean_masks[name].is_cuda
            np.testing.assert_array_equal(masks[name].cpu().numpy(), proposals[name])
            target, _ = filtered_masks(proposals[name], diagnostics[name])
            np.testing.assert_array_equal(clean_masks[name].cpu().numpy(), target)
        return 'pre-gate handoff', [], {}
    geometry._build_prepared = handoff
    try:
        with patch.object(CudaRobotFilter, 'prepare_surfaces', staticmethod(prepare)), \
                patch.object(CudaRobotFilter, 'filter_cameras', filtering), \
                patch.object(CudaRobotFilter, 'depth_tensor', depth), \
                patch('graph.plan_a_geometry.core.classify_robot', side_effect=AssertionError('CPU robot classification')):
            _, _, _, state = infer_frame(geometry, {name: Adapter(name) for name in NAMES}, cameras,
                data, surfaces, scene, cameras[0].timestep, SimpleNamespace(), 'correction_fixture', policy=Correction())
        assert counts == dict(prepare=1, batch=1, fused_depth=2), counts
        assert memo.misses == 0 and geometry.robot_preparation._pending is None
        assert state['robot_preparation'] == 'before_vision' and state['shared_robot_classification']
        return dict(scene=scene, size=512, cameras=2, counts=counts, CPU_classifier_calls=0,
                    full_boolean_diagnostics_exact=True, owner_differences_vs_CPU=owner_differences,
                    raw_and_clean_masks_exact=True,
                    new_pixels_outside_original_mask_checked=True, correction_gate_trace_exact=True,
                    max_predicted_depth_error_m=max(max_depth_error), final_graph_executed=False)
    finally:
        geometry.close()


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  scope='four robot-filter integration fixes; no simulator/models/API/latency benchmark',
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with ExperimentLogger(team='integration', model='task10i_robot_filter_live_completion',
            job_type='evaluate', seed=42, root=output / 'experiments', config=config) as logger:
        names = ['graph.plan_a_bc.test_robot_live_integration',
                 'graph.plan_a_bc.test_robot_overlap.DefaultRobotOverlap',
                 'graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_preparation',
                 'graph.experiments.robot_filter_gpu_io_10i_20261103_v1.test_gpu_io',
                 'graph.integration_baseline.tests.test_online_vision',
                 'graph.task10i_stage4a2.test_contracts', 'graph.task10i_stage4a.test_perframe_cache',
                 'graph.plan_a_geometry.tests']
        with (output / 'tests.log').open('w') as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
        assert tests.wasSuccessful(), 'see tests.log'
        inputs = Inputs()
        cases = [('10i_s1_0921_t080', None), ('10i_s1_0531_track_t011', 11)]
        normal = [probe(inputs, *case) for case in cases]
        corrected = [correction_case(inputs, *case) for case in cases]
        result = dict(status='PASS', run_id=logger.run_id, config=config,
                      tests=dict(run=tests.testsRun, skipped=len(tests.skipped), failures=len(tests.failures), errors=len(tests.errors)),
                      native_normal_cases=normal, shared_correction_cases=corrected)
        (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        for path in (output / 'tests.log', output / 'results.json'):
            logger.link_output(path)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
