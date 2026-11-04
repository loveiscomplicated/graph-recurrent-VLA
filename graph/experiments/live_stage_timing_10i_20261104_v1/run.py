"""Actual main run_episode, real vision/filter/gate, per-stage wall latency."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from .profile import Profiler, instrument_graph_edges


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    if not values:
        return dict(n=0)
    return dict(n=len(values), median_ms=statistics.median(values),
                p95_ms=float(np.quantile(values, .95)), min_ms=min(values), max_ms=max(values))


class ZeroNativePolicy(torch.nn.Module):
    """Only the post-Graph policy is replaced; main preprocessing is unmodified."""
    def __init__(self, codec):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), device='cuda'))
        self.variant = 'STAGE_TIMING'
        self.codec = codec

    def step(self, packed, robot, norm, hidden):
        assert packed.node_input.is_cuda and packed.edge_index.is_cuda and packed.edge_attr.is_cuda
        result = torch.zeros((1, 7), device='cuda')
        if self.codec.method != 'A':
            result[:, :6] = -torch.tensor(self.codec.mean, device='cuda')/torch.tensor(self.codec.std, device='cuda')
        return result, torch.zeros((1, 1), device='cuda')


class LiveMeasurement:
    def __init__(self, detailed, mode):
        self.prof = Profiler(torch.cuda.synchronize, detailed=detailed)
        self.mode = mode
        self.rows = []
        self.compilations = []
        self.FPS_captures = []
        self.pending_render = None
        self.current = None

    def render(self, function):
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            self.pending_render = dict(start_ns=start, end_ns=end, ms=(end-start)/1e6, adapter_ms=0.)
            return result
        return wrapped

    def observation(self, function):
        def wrapped(*args, **kwargs):
            start = time.perf_counter_ns()
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            if self.pending_render is not None:
                self.pending_render['adapter_ms'] += (end-start)/1e6
                self.pending_render['end_ns'] = end
            return result
        return wrapped

    def sensors(self, function):
        def wrapped(env, obs, identity, t, schema):
            self.prof.frame = t
            assert self.pending_render is not None
            torch.cuda.synchronize()
            self.current = dict(timestep=t, sensor_start_ns=time.perf_counter_ns(),
                                rendering=self.pending_render)
            self.pending_render = None
            with self.prof.span('2', 'rollout.sensors'):
                result = function(env, obs, identity, t, schema)
            return result
        return wrapped

    def build(self, function):
        def wrapped(*args, **kwargs):
            start = time.perf_counter_ns()
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            row = self.current
            row['build_ms'] = (end-start)/1e6
            row['sensors_to_graph_ms'] = (end-row['sensor_start_ns'])/1e6
            row['image_start_to_graph_ms'] = (end-row['rendering']['start_ns'])/1e6
            row['render_and_sensor_graph_work_ms'] = row['rendering']['ms']+row['rendering']['adapter_ms']+row['sensors_to_graph_ms']
            graph, records, gate = result
            from graph.persistent_geometry.tensor_records import TensorRecords
            from graph.persistent_geometry.tensor_graph import FIELDS
            assert isinstance(records, TensorRecords) and records.xyz.is_cuda
            assert all(getattr(graph, name).is_cuda for name in FIELDS)
            assert graph.packet['graph_backend'] == 'torch_cuda'
            row.update(nodes=len(graph.node_type), edges=graph.edge_index.shape[1],
                       object_points=[int((records.object_id==i).sum()) for i in (0, 1)],
                       gate=gate, backends={k:graph.packet[k] for k in
                           ('perception_backend', 'graph_backend', 'records_dedup_backend', 'final_fps_backend', 'wrist_gate_backend')})
            # Verification exports are AFTER the measured Graph completion.
            row['graph_sha256'] = {}
            for name in FIELDS:
                value = getattr(graph, name).detach().cpu().numpy()
                row['graph_sha256'][name] = hashlib.sha256(value.tobytes()).hexdigest()
            self.rows.append(row)
            print('FRAME', json.dumps(dict(mode=self.mode,
                t=row['timestep'], graph_ms=row['sensors_to_graph_ms'], nodes=row['nodes'], edges=row['edges'],
                accepted=gate['accepted'])), flush=True)
            return result
        return wrapped

    def install(self, stack):
        from mani_skill.envs.sapien_env import BaseEnv
        from envs.maniskill.observation_device import ObservationDevice
        from graph.plan_a_bc import rollout, geometry, tensor_ingress
        from graph.plan_a_geometry import core
        from graph.persistent_geometry import tensor_gate, tensor_fusion, tensor_records, tensor_graph
        from graph.integration_baseline import tensor_online_vision
        from graph.experiments.robot_filter_gpu_io_10i_20261103_v1 import gpu_filter
        from visual.runtime import tracking, graph_perception, cuda_sampling
        from visual.runtime import tensor_packet
        from visual.runtime.live_grounding import LiveGrounding

        def replace(owner, name, value):
            stack.enter_context(patch.object(owner, name, value))
        replace(BaseEnv, '_get_obs_with_sensor_data', self.render(BaseEnv._get_obs_with_sensor_data))
        replace(ObservationDevice, 'observation', self.observation(ObservationDevice.observation))
        replace(rollout, 'sensors', self.sensors(rollout.sensors))
        replace(geometry.CurrentGeometry, 'build', self.build(geometry.CurrentGeometry.build))
        from triton.runtime.jit import JITFunction
        original_compile = JITFunction._do_compile
        def compile_kernel(kernel, *args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return original_compile(kernel, *args, **kwargs)
            finally:
                self.compilations.append(dict(frame=self.prof.frame, kernel=kernel.fn.__name__,
                    start_ns=start, end_ns=time.perf_counter_ns(),
                    host_ms=(time.perf_counter_ns()-start)/1e6))
        replace(JITFunction, '_do_compile', compile_kernel)
        original_capture = cuda_sampling._FPSPlan.__init__
        def capture(plan, n, seeds, extra, points):
            start = time.perf_counter_ns()
            try:
                return original_capture(plan, n, seeds, extra, points)
            finally:
                self.FPS_captures.append(dict(frame=self.prof.frame, n=n, seeds=seeds, extra=extra,
                    start_ns=start, end_ns=time.perf_counter_ns(),
                    host_ms=(time.perf_counter_ns()-start)/1e6))
        replace(cuda_sampling._FPSPlan, '__init__', capture)
        # Synchronous API/scan wall spans do not need CUDA barriers. Keep them
        # even in the natural pass so actual requests are distinguished from
        # exact-RGB cache hits during the paired detailed pass.
        replace(LiveGrounding, '_request', self.prof.wrap(LiveGrounding._request,
            '3-A', 'Terra_actual_request_encode_HTTP_parse', cpu=True, always=True))
        replace(tensor_online_vision, 'detect_initial', self.prof.wrap(tensor_online_vision.detect_initial,
            '3-A', 'grounding_scan_RGB_to_boxes', cpu=True, always=True))
        replace(gpu_filter.CudaRobotFilter, 'prepare_surfaces', staticmethod(
            self.prof.wrap(gpu_filter.CudaRobotFilter.prepare_surfaces, '3-B-worker',
                           'CPU_hull_prepare', cpu=True, always=True)))
        if not self.prof.detailed:
            return
        wrappers = [
            (tensor_ingress.TensorRobotPreparation, 'submit', '3-B', 'snapshot_submit_depth'),
            (tensor_ingress.TensorRobotPreparation, 'filter', '6', 'join_and_robot_filter'),
            (tensor_online_vision.TensorFrames, 'append', '4', 'resize_normalize_image'),
            (tensor_online_vision.TensorOnlineVisionAdapter, 'process', '5', 'vision_process'),
            # start_frames includes real SAM2 feature/prompt inference.
            (tracking.SAM2Tracker, 'start_frames', '5', 'SAM2_prompt_inference'),
            (tensor_gate, 'prepare_tensor_gate', '8', 'gate_input_checks_and_handoff'),
            (tensor_gate, 'prepare_tensor_views', '8', 'shared_depth_calibration_views'),
            (tensor_gate, 'wrist_gate_tensor', '7', 'actual_wrist_gate'),
            (graph_perception, 'prepare_cuda', '9', 'mask_projection_and_row_index'),
            (graph_perception, 'features_cuda', '10', 'fused_PCA_and_features'),
            (cuda_sampling, 'build_sampling_info_cuda', '11', 'sampling_evidence'),
            (cuda_sampling, 'sample_cuda', '12', 'ShapeFirst_and_output_provenance'),
            (tensor_packet, 'selected_packet', '13', 'transform_packet_validate'),
            (tensor_fusion, 'observations_from_tensor_packet', '14', 'packet_to_observations'),
            (tensor_fusion, 'consistent_views_tensor', '14', 'same_frame_view_check'),
            (tensor_records, 'current_records_tensor', '15', 'records'),
            (tensor_records, 'deduplicate_tensor', '15', 'greedy_3mm_dedup'),
            (tensor_records, 'budget_tensor', '16', 'final128_FPS'),
            (core.PlanA, 'graph', '18', 'PlanA_graph_metadata_validation'),
            (geometry.CurrentGeometry, '_finish', '18', 'final_geometry_validation')]
        replace(tensor_graph, 'build_graph_tensor', self.prof.wrap(
            instrument_graph_edges(tensor_graph.build_graph_tensor, self.prof.span),
            '18', 'Graph_nodes_metadata_return'))
        for owner, name, stage, label in wrappers:
            replace(owner, name, self.prof.wrap(getattr(owner, name), stage, label))

    def finish(self):
        for row in self.rows:
            events = [e for e in self.prof.events if e['frame']==row['timestep']]
            stages = {'1': row['rendering']['ms']+row['rendering']['adapter_ms']}
            for event in events:
                stages[event['stage']] = stages.get(event['stage'], 0.)+event['exclusive_ms']
            row['stage_ms'] = stages
            main = sum(v for k, v in stages.items() if k not in ('1', '3-B-worker'))
            row['unassigned_sensor_graph_ms'] = row['sensors_to_graph_ms']-main
        for row in self.rows:
            row['Triton_compile_host_ms'] = sum(e['host_ms'] for e in self.compilations if e['frame']==row['timestep'])
            row['FPS_capture_host_ms'] = sum(e['host_ms'] for e in self.FPS_captures if e['frame']==row['timestep'])
        return dict(rows=self.rows, spans=self.prof.events, compilations=self.compilations, FPS_captures=self.FPS_captures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--dotenv', type=Path)
    parser.add_argument('--frames', type=int, default=36)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    assert os.environ.get('CONTAINER_ID') == '58752018'
    assert torch.cuda.is_available() and args.frames > args.warmup+1
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'another CUDA process is active'
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[3]
    for file, digest in read(source/'source_manifest.json').items():
        assert sha(source/file) == digest, file
    from graph.plan_a_bc import rollout
    from graph.integration_baseline.bc08_codec import ActionCodec
    from visual.runtime.live_grounding import LiveGrounding, policy_identity
    from visual.runtime.tracking import SAM2Tracker
    runtime = read(args.runtime/'runtime_paths.json')
    schema = read(args.runtime/'normalizer.json')['schema']
    codec = ActionCodec(read(args.runtime/'action_codec.json'))
    detector = LiveGrounding(device='cuda', model_id=runtime['dino'],
                             **({'env_file':args.dotenv} if args.dotenv else {}))
    # A DINO run would not answer the requested Terra timing comparison.
    if detector.backend != 'terra':
        raise RuntimeError('Terra credential unavailable; do not silently benchmark DINO')
    load_start = time.perf_counter_ns()
    tracker = SAM2Tracker(device='cuda', checkpoint=runtime['sam'])
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter_ns()-load_start)/1e6
    config = dict(task='10i_main_live_stage_timing_20261104', instance='58752018',
                  size=512, cameras=2, coordinate_frame='ee', seed=args.seed,
                  frames_per_pass=args.frames, warmup_tracking_frames=args.warmup,
                  passes=['natural', 'detailed', 'natural_warm'], grounding=policy_identity(),
                  SAM2_checkpoint_sha256=sha(runtime['sam']), SAM2_load_ms=model_load_ms,
                  source_manifest_sha256=sha(source/'source_manifest.json'),
                  real='original run_episode, simulator, Terra, SAM2, robot filter, wrist gate, default CUDA geometry',
                  policy='zero native motion after Graph, original codec/gripper0',
                  publishing=False, versions={n:importlib.metadata.version(n) for n in
                      ('torch','numpy','scipy','triton','mani_skill','sapien','gymnasium')},
                  gpu=torch.cuda.get_device_name(), python=__import__('sys').version)
    with ExperimentLogger(team='graph', model='main_live_stage_timing', seed=args.seed,
                          job_type='diagnostic', config=config, root=args.output/'experiments') as log:
        results = {}
        for mode in ('natural', 'detailed', 'natural_warm'):
            root = args.output/mode
            root.mkdir()
            (root/'inputs').symlink_to(args.runtime/'inputs', target_is_directory=True)
            cfg = deepcopy(read(args.runtime/'config.json'))
            cfg['evaluation'].update(horizon=args.frames, smoke_horizon=args.frames)
            cfg.setdefault('geometry', {})['coordinate_frame'] = 'ee'
            write(root/'config.json', cfg)
            measurement = LiveMeasurement(detailed=mode=='detailed', mode=mode)
            before = detector.status()
            with ExitStack() as stack:
                measurement.install(stack)
                episode = rollout.run_episode(root, root/'episode', ZeroNativePolicy(codec),
                    SimpleNamespace(schema=schema), codec, detector, tracker,
                    dict(seed=args.seed, group='timing_diagnostic'), 0, smoke=True)
            result = measurement.finish()
            result.update(episode=episode, grounding_before=before, grounding_after=detector.status())
            assert len(result['rows']) == args.frames
            result['summary'] = {}
            steady = result['rows'][args.warmup+1:]
            for key in ('build_ms','sensors_to_graph_ms','render_and_sensor_graph_work_ms','image_start_to_graph_ms'):
                result['summary'][key] = stats([row[key] for row in steady])
            result['stage_summary'] = {stage:stats([r['stage_ms'].get(stage, 0.) for r in steady])
                                       for stage in sorted({k for row in steady for k in row['stage_ms']})}
            write(root/'measurement.json', result)
            results[mode] = result
            log.link_output(root/'measurement.json')
            print('PASS_SUMMARY', mode, json.dumps(result['summary']), flush=True)
            gc.collect()
        matched = {mode:[dict(timestep=a['timestep'], equal=a['graph_sha256']==b['graph_sha256'])
                   for a, b in zip(results['natural']['rows'], results[mode]['rows'])]
                   for mode in ('detailed', 'natural_warm')}
        exact = all(row['equal'] for values in matched.values() for row in values)
        summary = dict(status='PASS' if exact else 'GRAPH_MISMATCH',
                       run_id=log.run_id, config=config, per_frame_graph_exact=matched,
                       grounding=detector.status(), summaries={k:dict(summary=v['summary'],
                           stage_summary=v['stage_summary'], first_frame=v['rows'][0]) for k,v in results.items()})
        write(args.output/'summary.json', summary)
        log.record(stage_timing_status=summary['status'], per_frame_graph_exact=exact)
        log.link_output(args.output/'summary.json')
        assert summary['status'] == 'PASS', 'timing-only instrumentation changed Graph'
        assert detector.backend == 'terra' and detector.api_requests > 0
        print('COMPLETE', log.run_id, flush=True)


if __name__ == '__main__':
    main()
