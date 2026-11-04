"""Paired historical/current/live RF timing, then a real causal SAM2 overlap trace."""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from graph.experiments.robot_sam_overlap_live_10i_20261103_v1.run import Inputs, ROOT, NAMES, CHECKPOINT
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
from graph.persistent_geometry.tensor_camera import tensor_camera
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.integration_baseline.tensor_online_vision import TensorOnlineVisionAdapter
from visual.runtime.tracking import SAM2Tracker
from utils.experiment_logger import ExperimentLogger

HISTORICAL = Path('/workspace/task10i_robot_filter_gpu_depth_20261103_v1/final_512')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def load_reference():
    path = HISTORICAL / 'source/gpu_filter.py'
    assert sha(path) == '4c7402e66752473b19e6412982e6b6e353f5c2f1877ea23b472f1b65ef63f8df'
    spec = importlib.util.spec_from_file_location('frozen_7159ms_filter', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CudaRobotFilter


def load_data(scene, cameras, surfaces):
    with np.load(ROOT / f'sensors/{scene}/512/robot.npz') as z:
        data = {key: z[key].copy() for key in z.files}
    data.update(episode=np.array(scene), timestep=np.int64(cameras[0].timestep),
                timestamp=np.float64(cameras[0].timestamp), qpos=data['robot76'][:9],
                robot_link_world=np.stack([s.world_from_link for s in surfaces]))
    return data


def geometry(inputs):
    result = CurrentGeometry('/workspace/task10g_f1_abc_20261031/full/inputs', inputs.fk.schema)
    assert result.robot_filter_backend == 'cuda'
    # The actual RF path runs; the following gate/Graph is outside this timing.
    result._build_prepared = lambda cameras, data, surfaces, masks, clean_masks=None, **_: clean_masks
    return result


def check(actual, expected):
    for name in NAMES:
        assert actual[name].is_cuda
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def summary(values):
    return dict(n=len(values), median_ms=statistics.median(values),
                q25_ms=float(np.quantile(values, .25)), q75_ms=float(np.quantile(values, .75)),
                min_ms=min(values), max_ms=max(values))


def paired(inputs, reference, output):
    scene = '10i_s1_0921_t080'
    cameras, surfaces = inputs.load(scene)
    data = load_data(scene, cameras, surfaces)
    masks = {}
    for name in NAMES:
        with np.load(ROOT / f'masks/actual/static/r0__{scene}__512__{name}/masks.npz') as z:
            masks[name] = z['masks'].copy()
    fingerprint = dict(masks={n: array_sha(v) for n, v in masks.items()},
                       cameras={c.camera: {k: array_sha(getattr(c, k)) for k in
                                ('depth_raw_mm', 'intrinsic', 'extrinsic_cv')} for c in cameras},
                       surfaces=[dict(link=s.link, vertices=array_sha(s.world_vertices)) for s in surfaces])
    historical_config = json.loads((HISTORICAL / 'config.json').read_text())
    assert fingerprint == historical_config['input_sha256'], 'historical inputs changed'
    native = tuple(tensor_camera(c) for c in cameras)
    raw = {name: torch.as_tensor(value, device='cuda') for name, value in masks.items()}
    old, current = reference(), CudaRobotFilter()
    live, full = geometry(inputs), geometry(inputs)
    expected = old.filter_cameras(cameras, surfaces, raw)
    variants = ['historical_direct', 'current_direct', 'live_ready_masks', 'live_full_classification']
    def run(name):
        if name == 'historical_direct':
            return old.filter_cameras(cameras, surfaces, raw)
        if name == 'current_direct':
            return current.filter_cameras(cameras, surfaces, raw)
        if name == 'live_ready_masks':
            return live.build(native, data, surfaces, lambda: raw)
        return full.build(native, data, surfaces, lambda: raw, mask_transform=lambda masks, frame: masks)
    rows = []
    try:
        for _ in range(3):
            for name in variants:
                check(run(name), expected)
        for repeat in range(15):
            row = dict(repeat=repeat, ms={})
            for name in variants if repeat % 2 == 0 else variants[::-1]:
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                clean = run(name)
                torch.cuda.synchronize()
                row['ms'][name] = (time.perf_counter_ns() - start) / 1e6
                check(clean, expected)  # Outside every timer.
            row['exact'] = True
            rows.append(row)
            print('PAIRED', json.dumps(row), flush=True)
        result = dict(status='PASS', scene=scene, size=512, cameras=2, rows=rows,
                      summary={name: summary([r['ms'][name] for r in rows]) for name in variants},
                      historical_input_hashes_match=True, input_sha256=fingerprint,
                      boundaries=dict(historical_direct='original CPU sensors/calibration/FK surfaces + resident CUDA masks -> CUDA clean, including current hulls/uploads; old 7.159ms boundary',
                          current_direct='identical old boundary with the current filter class',
                          live_ready_masks='native CUDA RGB-D/calibration/current FK/raw masks ready -> actual CurrentGeometry RF handoff; checks, snapshots, worker submission, depth conversion, prepare/join/filter included, no SAM2 to hide preparation',
                          live_full_classification='same native live boundary, full-frame diagnostics and final clean reuse for correction; excludes CPU correction/evidence export/gate/Graph'))
        write(output / 'paired.json', result)
        return result
    finally:
        live.close()
        full.close()


def real_sam2(inputs, reference, output):
    parent = '10i_s1_0531'
    scenes = sorted((s for s in inputs.protocol['scenes'] if s.get('cohort') == 'tracking' and s.get('parent') == parent),
                    key=lambda s: s['timestep'])
    assert [s['timestep'] for s in scenes] == list(range(21))
    loaded = []
    for s in scenes:
        cpu, surfaces = inputs.load(s['id'])
        loaded.append((cpu, tuple(tensor_camera(c) for c in cpu), load_data(s['id'], cpu, surfaces), surfaces))
    assert sha(CHECKPOINT) == '6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
    tracker = SAM2Tracker('cuda', checkpoint=CHECKPOINT)
    tracker.predictor.eval()
    adapters = {name: TensorOnlineVisionAdapter(None, tracker) for name in NAMES}
    for adapter in adapters.values():
        adapter.reset(parent)
    boxes = inputs.boxes(scenes[0]['id'])
    live, old = geometry(inputs), reference()
    rows = []
    try:
        for t, (cpu, native, data, surfaces) in enumerate(loaded):
            marks = {}
            def produce():
                marks['vision_start'] = time.perf_counter_ns()
                raw = {}
                for camera in native:
                    initial = {obj: dict(box=box, frame=0 if box is not None else None,
                               score=1. if box is not None else None)
                               for obj, box in zip(('peg', 'box'), boxes[camera.camera])}
                    # Existing recorded RGB-derived t0 prompts, no DINO/API.
                    with patch('graph.integration_baseline.tensor_online_vision.detect_initial', return_value=initial):
                        raw[camera.camera] = adapters[camera.camera].process(camera.rgb, parent, t, mask_device='cuda')[0]
                marks['vision_end'] = time.perf_counter_ns()
                marks['cpu_preparation_done'] = live.robot_preparation._pending.future.done()
                return raw
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            clean = live.build(native, data, surfaces, produce)
            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            whole = (end - start) / 1e6
            vision = (marks['vision_end'] - marks['vision_start']) / 1e6
            # All values are paired CPU-wall spans from THIS call, not sums of medians.
            row = dict(timestep=t, whole_sam_and_RF_ms=whole, vision_host_call_ms=vision,
                       prevision_ms=(marks['vision_start'] - start) / 1e6,
                       postvision_to_clean_ms=(end - marks['vision_end']) / 1e6,
                       visible_RF_ms=whole - vision,
                       cpu_preparation_done_at_vision_return=marks['cpu_preparation_done'])
            check(clean, old.filter_cameras(cpu, surfaces, live.last_raw_masks))
            row['exact'] = True
            rows.append(row)
            print('SAM2', json.dumps(row), flush=True)
        steady = rows[6:21]
        result = dict(status='PASS', parent=parent, size=512, cameras=2, rows=rows,
                      precision=dict(weights=str(next(tracker.predictor.parameters()).dtype), autocast=False,
                                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
                      checkpoint_sha256=sha(CHECKPOINT), boxes=boxes,
                      summary={k: summary([row[k] for row in steady]) for k in
                               ('visible_RF_ms', 'prevision_ms', 'postvision_to_clean_ms', 'whole_sam_and_RF_ms')},
                      aggregation='t0 initialization; t1..5 warm progression; t6..20 fifteen matched steady frames, not repeated identical inputs',
                      boundary='actual default native CUDA video adapter/SAM2 + CurrentGeometry RF; no extra interstage GPU synchronization; gate/Graph excluded; visible RF is same-call whole minus vision host span and not standalone RF latency')
        write(output / 'sam2.json', result)
        return result
    finally:
        live.close()
        for adapter in adapters.values():
            adapter.close()
        del tracker
        gc.collect()
        torch.cuda.empty_cache()


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    gc.enable()
    resource.setrlimit(resource.RLIMIT_DATA, (80 * 1024**3, 80 * 1024**3))
    reference = load_reference()
    inputs = Inputs()
    config = dict(warmups=3, repeats=15, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, torch_threads=torch.get_num_threads(),
                  environment={k: os.getenv(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                  cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(), affinity=sorted(os.sched_getaffinity(0)),
                  source_sha256={str(Path(p).resolve()): sha(p) for p in
                                 (__file__, inspect.getfile(reference), inspect.getfile(CudaRobotFilter), inspect.getfile(CurrentGeometry))},
                  scope='RF timing only: no production edits, gate/Graph/action/simulator/grounding API; real SAM2 is run only for the separately labelled overlap trace')
    with ExperimentLogger(team='integration', model='task10i_robot_filter_live_timing', seed=42,
                          job_type='evaluate', root=output / 'experiments', config=config) as log:
        write(output / 'config.json', config)
        paired_result = paired(inputs, reference, output)
        log.link_output(output / 'paired.json')
        sam_result = real_sam2(inputs, reference, output)
        log.link_output(output / 'sam2.json')
        write(output / 'input_hashes.json', inputs.hashes)
        log.link_output(output / 'input_hashes.json')
        write(output / 'complete.json', dict(status='PASS', run_id=log.run_id,
              paired=paired_result['summary'], sam2_RF=sam_result['summary']))
        log.link_output(output / 'config.json')
        log.link_output(output / 'complete.json')
        print('COMPLETE', (output / 'complete.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
