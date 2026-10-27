"""Isolated, recorded-input CUDA comparison of legacy and reused Visual paths.

Each condition runs in a fresh process. Profiling, warmup, serialization and
model loading are outside steady latency. Never trains or publishes artifacts.
"""
import argparse
import csv
from copy import deepcopy
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
import shutil
import time
from time import perf_counter

import numpy as np
import torch

from graph.integration_baseline.direct_runtime import atomic, read, sha


ROOT = Path(__file__).resolve().parents[2]
FAST = dict(path='direct', bf16=True, compile=True, preprocess='gpu', prune=True,
            reuse_preflight=False, separate_states=False, deterministic=False,
            cudnn_tf32=True, frozen_dino=False)
VARIANTS = {
    'direct': dict(FAST),
    'adapter': dict(FAST, path='adapter'),
    'legacy_offline': dict(FAST, path='legacy_offline', bf16=False, compile=False,
                           preprocess='jpeg_pil', prune=False, separate_states=True,
                           deterministic=True, cudnn_tf32=False),
    'legacy_online': dict(FAST, path='legacy_online', bf16=False, compile=False,
                          preprocess='jpeg_pil', prune=False, separate_states=True,
                          deterministic=True, cudnn_tf32=False),
    'tracking_reference': dict(FAST, frozen_dino=True),
    'float32': dict(FAST, frozen_dino=True, bf16=False),
    'pil_preprocess': dict(FAST, frozen_dino=True, preprocess='pil'),
    'no_compile': dict(FAST, frozen_dino=True, compile=False),
    'reuse_preflight': dict(FAST, frozen_dino=True, reuse_preflight=True),
    'separate_states': dict(FAST, frozen_dino=True, separate_states=True),
    'no_prune': dict(FAST, frozen_dino=True, prune=False),
    'legacy_numerics': dict(FAST, frozen_dino=True, deterministic=True, cudnn_tf32=False),
}
CANONICAL = {'direct', 'adapter', 'legacy_offline', 'legacy_online'}


def statistics(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), maximum=float(a.max()),
                deadline_50ms_fraction=float(np.mean(a <= 50)))


def fingerprint(value):
    """Stable content digest, including nested packet arrays and their dtypes."""
    h = hashlib.sha256()
    def update(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            h.update(x.dtype.str.encode()); h.update(str(x.shape).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for key in sorted(x):
                h.update(str(key).encode()); update(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x:
                update(item)
        else:
            h.update(json.dumps(x, sort_keys=True, default=str, allow_nan=True).encode())
        h.update(b'\0')
    update(value)
    return h.hexdigest()


def detection_key(rgb, prompt):
    return fingerprint([rgb, prompt])


def prepare_normalizer_input(dataset, output):
    """Export only verified TRAIN robot rows; retain hashes of their raw sources."""
    import h5py
    from graph.integration_baseline.direct_runtime import finalize
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    proof_path = output.with_name('train_robot_provenance.json')
    if output.exists() or proof_path.exists():
        raise FileExistsError('new normalizer output paths required')
    manifest, split = read(dataset/'dataset_manifest.json'), read(dataset/'split_manifest.json')
    if split['dataset_sha256'] != sha(dataset/'dataset_manifest.json'):
        raise ValueError('split/dataset checksum mismatch')
    rows, proof = [], []
    for e in manifest['episodes']:
        if e['episode_id'] in split['splits']['train']:
            path = dataset/e['path']
            if sha(path) != e['sha256']:
                raise ValueError('normalizer source checksum mismatch')
            with h5py.File(path) as f:
                rows.append(f['robot_state'][:-1])
            proof.append(dict(episode_id=e['episode_id'], sha256=e['sha256']))
    if sorted(e['episode_id'] for e in proof) != sorted(split['splits']['train']):
        raise ValueError('incomplete TRAIN input')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem+'.partial.npz')
    np.savez_compressed(tmp, robot=np.concatenate(rows), episode_ids=np.array([e['episode_id'] for e in proof]))
    finalize(tmp, output)
    atomic(proof_path, dict(source_dataset_id=manifest['dataset_id'], inputs=proof), immutable=True)


class FrozenDetector:
    is_frozen = True
    last_ms = {}

    def __init__(self, trace):
        self.trace = trace

    def __call__(self, rgb, prompt):
        return deepcopy(self.trace[detection_key(rgb, prompt)])


class DinoMeter:
    def __init__(self, detector):
        self.detector = detector
        self.calls, self.requests, self.wall_ms, self.parts, self.detections = 0, 0, 0., {}, {}

    def __call__(self, rgb, prompt):
        tick = perf_counter()
        result = self.detector(rgb, prompt)
        self.wall_ms += (perf_counter() - tick) * 1000
        self.calls += not getattr(self.detector, 'is_frozen', False)
        self.requests += 1
        for k, v in self.detector.last_ms.items():
            self.parts[k] = self.parts.get(k, 0.) + v
        # Store JSON-native pairs; the detector API itself returns a tuple.
        self.detections[detection_key(rgb, prompt)] = [deepcopy(result[0]), deepcopy(result[1])]
        return result


class GroupedStream:
    """Benchmark-only state ablation: shared pixels, independent feature caches."""
    def __init__(self, tracker):
        self.template, self.p = tracker, tracker.p
        self.groups = []

    def reset(self):
        for group in self.groups:
            group.reset()
        self.groups = []
        self.frames = None

    def advance(self, image, t, boxes):
        from visual.scripts.stream_v1 import CausalFrames, StreamTracker
        if t == 0:
            self.rgb0 = image
            self.frames = CausalFrames(self.p.image_size, device=self.p.device, keep=2,
                                       preprocess=self.template.pre, observed_horizon=True)
        self.frames.push(image)
        if boxes:
            group = StreamTracker(self.p, preprocess=self.template.pre, prune=self.template.prune,
                                  observed_horizon=True)
            group.state = group._init_state(self.rgb0)
            group.frames = self.frames
            group.state['images'] = self.frames
            group.state['num_frames'] = t + 1
            for oid, box in boxes.items():
                group.add_box(t, oid, box)
            self.groups.append(group)
        values = {}
        for group in self.groups:
            group.state['num_frames'] = t + 1
            values.update(group._propagate(t))
        return values


class StageProfiler:
    """Synchronized inclusive intervals; nested rows must not be summed."""
    def __init__(self):
        self.originals, self.rows = [], []

    def wrap(self, obj, attr, name):
        original = getattr(obj, attr)
        def measured(*args, **kwargs):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tick = perf_counter(); start.record()
            value = original(*args, **kwargs)
            end.record(); end.synchronize()
            self.rows.append(dict(stage=name, wall_ms=(perf_counter()-tick)*1000,
                                  cuda_interval_ms=start.elapsed_time(end)))
            return value
        self.originals.append((obj, attr, original))
        setattr(obj, attr, measured)

    def close(self):
        for obj, attr, original in reversed(self.originals):
            setattr(obj, attr, original)


def gpu_snapshot():
    def query(flag):
        return subprocess.check_output(['nvidia-smi', flag, '--format=csv,noheader'], text=True).strip()
    return dict(gpu=query('--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu'),
                processes=query('--query-compute-apps=pid,process_name,used_memory'))


def gpu_process_ids(snapshot):
    """NVML reports host PIDs, which must never be used as container kill PIDs."""
    return {int(row[0].strip()) for row in csv.reader(snapshot['processes'].splitlines()) if row}


def reserve_gpu(folder, fraction):
    if gpu_process_ids(gpu_snapshot()):
        raise RuntimeError('GPU became occupied before worker initialization')
    tick = perf_counter()
    torch.cuda.set_per_process_memory_fraction(fraction)
    sentinel = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    snapshot = gpu_snapshot()
    pids = gpu_process_ids(snapshot)
    if len(pids) != 1:
        raise RuntimeError('exclusive worker GPU ownership could not be established')
    atomic(folder/'gpu_owner.json', dict(container_pid=os.getpid(), nvml_host_pid=next(iter(pids)),
        initialization_ms=(perf_counter()-tick)*1000, memory_fraction_limit=fraction,
        scope='single NVML process immediately after an initially idle GPU allocation'))
    del sentinel


def monitor_worker(process, folder, minimum_free_bytes=0):
    """Stop only our new process group when another GPU process appears."""
    interference = None
    while process.poll() is None:
        owner_path = folder/'gpu_owner.json'
        if minimum_free_bytes and shutil.disk_usage(folder).free < minimum_free_bytes:
            interference = dict(status='ABORTED_FOR_DISK_RESERVE', worker_container_pid=process.pid,
                minimum_free_bytes=minimum_free_bytes,
                action='SIGTERM to this newly created worker process group only')
        elif owner_path.exists():
            owner = read(owner_path)['nvml_host_pid']
            snapshot = gpu_snapshot()
            foreign = gpu_process_ids(snapshot) - {owner}
            if foreign:
                interference = dict(status='ABORTED_FOR_OTHER_GPU_WORK',
                    foreign_nvml_host_pids=sorted(foreign), worker_container_pid=process.pid,
                    action='SIGTERM to this newly created worker process group only')
        if interference:
            atomic(folder/'resource_interference.json', interference)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        time.sleep(1)
    return process.wait(), interference


def load_inputs(plan):
    from graph.integration_baseline.direct_dataset import validate_receipt
    source = Path(plan['dataset'])
    manifest, split = read(source/'dataset_manifest.json'), read(source/'split_manifest.json')
    schema = read(source/'robot_state_schema.json')
    if sha(source/'dataset_manifest.json') != plan['dataset_sha256']:
        raise ValueError('dataset manifest changed')
    selected = [e for e in manifest['episodes'] if e['episode_id'] in plan['episodes']]
    if set(e['episode_id'] for e in selected) != set(plan['episodes']):
        raise ValueError('missing selected episode')
    if not set(plan['episodes']) <= set(split['splits']['train']):
        raise ValueError('benchmark requires TRAIN episodes')
    return [(e['episode_id'], validate_receipt(source, e, schema)) for e in selected], schema, split


def policy_callback(plan, arrays, schema, split, folder):
    from graph.integration_baseline.graph import build_graph, pack_graphs
    from graph.integration_baseline.model import IntegrationPolicy
    from graph.integration_baseline.robot import RobotNormalizer, geometry_in_ee
    from graph.integration_baseline.vision import packet_from_frame
    from visual.runtime.rgbd import TrackedRGBDFrame
    data = Path(plan['train_robot'])
    if sha(data) != plan['train_robot_sha256']:
        raise ValueError('normalizer input changed')
    with np.load(data, allow_pickle=False) as fit:
        ids = fit['episode_ids'].tolist()
        if sorted(ids) != sorted(split['splits']['train']):
            raise ValueError('normalizer must cover exactly frozen TRAIN split')
        normalizer = RobotNormalizer.fit(fit['robot'], ids, split['splits'], schema)
    torch.manual_seed(42)
    model = IntegrationPolicy(schema, d=64, L=2, gru_hidden=144, gru_layers=1, enc_hidden=128).cuda().eval()
    hidden, previous = None, None
    rows, actions = [], []

    def callback(eid, t, masks, mask_ms):
        nonlocal hidden, previous
        if previous != eid or t == 0:
            hidden, previous = None, eid
        a = arrays[eid]
        tick = perf_counter()
        frame = TrackedRGBDFrame(t, 'base_camera', a['base_camera/rgb'][t], a['base_camera/depth'][t],
            a['base_camera/intrinsic'][t], a['base_camera/cam2world'][t], a['tcp_pose'][t],
            {0: masks[0], 1: masks[1]}, t, float(a['timestamp'][t]))
        packet, detail = packet_from_frame(frame, plan['source_dataset_id'] + f':ep{eid}', keep_ratio=.5)
        packet_done = perf_counter()
        graph = build_graph(packet, geometry_in_ee(a['robot_world_geometry'][t], a['tcp_pose'][t]))
        cpu_packed = pack_graphs([graph], [(0, 0)])
        packed = cpu_packed.to('cuda')
        robot = torch.from_numpy(a['robot_state'][t].copy())[None].cuda()
        torch.cuda.synchronize(); graph_done = perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            action, hidden = model.step(packed, robot, normalizer, hidden)
        action = action.cpu().numpy().copy()
        torch.cuda.synchronize(); done = perf_counter()
        if action.shape != (1, 7) or not np.isfinite(action).all():
            raise ValueError('invalid CPU Action7')
        rows.append(dict(episode_id=eid, timestep=t, mask_ms=mask_ms,
            vision_packet_ms=(packet_done-tick)*1000, graph_pack_transfer_ms=(graph_done-packet_done)*1000,
            policy_action_cpu_ms=(done-graph_done)*1000, total_ms=mask_ms+(done-tick)*1000,
            vision_detail_ms=detail, packet_digest=fingerprint(packet), graph_digest=fingerprint(vars(cpu_packed)),
            point_count=len(packet['points']), object_counts=[int((packet['object_id']==o).sum()) for o in (0, 1)]))
        actions.append(action[0])
    return callback, rows, actions


def worker(plan_path, name):
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from graph.integration_baseline.streaming_vision_adapter import (
        PrefixInitialization, StreamVisionAdapter, build_stream_tracker)
    from graph.integration_baseline.vision import causal_masks
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker

    plan = read(plan_path); settings = plan['variants'][name]
    folder = Path(plan_path).parent/name; folder.mkdir(exist_ok=True)
    np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(plan['threads'])
    torch.use_deterministic_algorithms(settings['deterministic'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = settings['cudnn_tf32']
    torch.backends.cudnn.benchmark = False
    data, schema, split = load_inputs(plan)
    reserve_gpu(folder, plan.get('gpu_memory_fraction', .35))
    tick = perf_counter()
    detector = (FrozenDetector(read(Path(plan_path).parent/'direct/detections.json'))
                if settings['frozen_dino'] else GDino(device='cuda'))
    meter = DinoMeter(detector)
    if settings['path'].startswith('legacy'):
        tracker = SAM2Tracker(device='cuda')
    else:
        tracker = build_stream_tracker(compile_image_encoder=settings['compile'],
                                       preprocess=settings['preprocess'], prune=settings['prune'])
    if settings['reuse_preflight']:
        from graph.integration_baseline.vision_speed import guard_preflight
        preflight = guard_preflight(tracker.p)
    else:
        preflight = None
    model_load_ms = (perf_counter()-tick)*1000

    def episode(eid, a, callback=None):
        rgb = a['base_camera/rgb']
        samples, values = [], []
        previous_calls = meter.calls
        previous_requests = meter.requests
        def record(t, masks, elapsed, found):
            nonlocal previous_calls, previous_requests
            calls = meter.calls - previous_calls; previous_calls = meter.calls
            requests = meter.requests - previous_requests; previous_requests = meter.requests
            samples.append(dict(episode_id=eid, timestep=t, mask_ms=elapsed, dino_calls=calls,
                                dino_requests=requests,
                                initialized_objects=len(found),
                                mask_pixels=masks.sum(axis=(1,2)).tolist()))
            values.append(masks.copy())
            if callback:
                callback(eid, t, masks, elapsed)
        if settings['path'] == 'legacy_offline':
            start = perf_counter()
            masks, meta = causal_masks(rgb, meter, tracker,
                lambda t, m, timing: record(t, m, timing['elapsed_ms'], {}))
            elapsed_ms = (perf_counter()-start)*1000
            found = meta['initialization']
            for row in samples:
                row['initialized_objects'] = sum(v['decision_observation'] <= row['timestep'] for v in found.values())
            overhead = elapsed_ms - sum(r['mask_ms'] for r in samples)
            if callback:
                overhead -= sum(r['total_ms']-r['mask_ms'] for r in pipeline_rows[-len(rgb):])
            return np.stack(values), samples, found, overhead
        if settings['path'] == 'legacy_online':
            online = OnlineVisionAdapter(meter, tracker)
        elif settings['path'] == 'adapter':
            online = StreamVisionAdapter(meter, tracker, bf16=settings['bf16'])
        else:
            online = None
            initialization = PrefixInitialization(meter)
            stream = GroupedStream(tracker) if settings['separate_states'] else tracker
            stream.reset()
        if online:
            online.reset(f'ep{eid}')
        try:
            for t, image in enumerate(rgb):
                torch.cuda.synchronize(); start = perf_counter()
                if online:
                    masks, _ = online.process(image, f'ep{eid}', t)
                    found = online.found
                else:
                    boxes = initialization.push(image)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=settings['bf16']):
                        if settings['separate_states']:
                            output = stream.advance(image, t, boxes)
                        else:
                            output = stream.start(image, boxes) if t == 0 else stream.step(image, boxes=boxes)[1]
                    masks = np.zeros((2, *image.shape[:2]), dtype=bool)
                    for oid, value in output.items():
                        masks[oid-1] = value.cpu().numpy()
                    found = initialization.found
                torch.cuda.synchronize()
                record(t, masks, (perf_counter()-start)*1000, found)
            return np.stack(values), samples, deepcopy(found), 0.
        finally:
            if online:
                online.close()
            else:
                stream.reset()

    with torch.inference_mode():
        start = perf_counter()
        warmup_frames = []
        for eid, a in data:
            _, rows, _, _ = episode(eid, a)
            warmup_frames.extend(rows)
        warmup_ms = (perf_counter()-start)*1000
        atomic(folder/'warmup_frames.json', warmup_frames)
        torch.cuda.reset_peak_memory_stats()
        frames, first, initializations, overheads = [], {}, {}, []
        repeats_exact = True
        for repeat in range(plan['repeats']):
            for eid, a in data:
                masks, samples, found, overhead = episode(eid, a)
                if repeat == 0:
                    first[eid], initializations[str(eid)] = masks, found
                    np.savez_compressed(folder/f'ep{eid}_masks.npz', masks=masks)
                else:
                    repeats_exact &= np.array_equal(first[eid], masks)
                frames.extend(dict(r, repeat=repeat) for r in samples)
                overheads.append(overhead)
                print(f'{name} repeat={repeat} episode={eid} frames={len(samples)}', flush=True)
        atomic(folder/'detections.json', meter.detections)
        pipeline_rows, actions = [], []
        if name in CANONICAL:
            # Preserve the existing FP32 policy numerics independently of Vision
            # autocast. This phase measures actual synchronous mask->Action7.
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.allow_tf32 = False
            callback, pipeline_rows, actions = policy_callback(plan, dict(data), schema, split, folder)
            eid, a = data[0]
            short = {k: v[:16] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(a['base_camera/rgb']) else v
                     for k, v in a.items()}
            episode(eid, short, callback)
            pipeline_rows.clear(); actions.clear()
            for eid, a in data:
                episode(eid, a, callback)
            np.savez_compressed(folder/'actions.npz', actions=np.stack(actions))
            atomic(folder/'pipeline_frames.json', pipeline_rows)
            torch.use_deterministic_algorithms(settings['deterministic'])
            torch.backends.cudnn.allow_tf32 = settings['cudnn_tf32']
        checks = None
        if name == 'adapter':
            from graph.integration_baseline.streaming_vision_checks import compare_stream_sequence
            checks = {str(eid): compare_stream_sequence(a['base_camera/rgb'], meter, tracker, folder/f'checks_ep{eid}')
                      for eid, a in data}
        if name == 'direct':
            from visual.scripts.stream_v1 import CausalFrames
            profiler = StageProfiler()
            for obj, attr, label in [(CausalFrames, 'push', 'preprocess'),
                (tracker.p.image_encoder, 'forward', 'image_encoder'),
                (tracker.p, 'propagate_in_video_preflight', 'preflight'),
                (tracker.p, '_run_memory_encoder', 'memory_encoder'),
                (tracker.p, '_run_single_frame_inference', 'tracking_inference_inclusive'),
                (tracker.p, '_get_orig_video_res_output', 'mask_resize')]:
                profiler.wrap(obj, attr, label)
            try:
                episode(*data[0])
            finally:
                profiler.close()
            atomic(folder/'profile.json', dict(scope='separate synchronized inclusive pass; do not sum nested intervals',
                   stages={stage: dict(wall_ms=statistics([r['wall_ms'] for r in profiler.rows if r['stage']==stage]),
                                       cuda_interval_ms=statistics([r['cuda_interval_ms'] for r in profiler.rows if r['stage']==stage]))
                           for stage in sorted({r['stage'] for r in profiler.rows})}, frames=profiler.rows))
    tracking = [r for r in frames if r['timestep'] > 0 and not r['dino_requests'] and r['initialized_objects'] > 0]
    groups = {'initialization': [r for r in frames if r['timestep']==0],
              'dino_scan': [r for r in frames if r['timestep']>0 and r['dino_requests']],
              'sam2_tracking_only': tracking}
    result = dict(status='COMPLETE', variant=name, settings=settings, model_load_ms=model_load_ms,
                  warmup_ms=warmup_ms, warmup='both full episodes; excluded; covers late objects and compilation',
                  cold_initial_frames=[r for r in warmup_frames if r['timestep']==0],
                  cold_scope='first observed inference includes native compile, DINO and initialization; not compile-only time',
                  groups={k:statistics([r['mask_ms'] for r in v]) for k,v in groups.items()},
                  tracking_by_initialized_objects={str(n): statistics([r['mask_ms'] for r in tracking if r['initialized_objects']==n]) for n in (1,2)},
                  all_frames=statistics([r['mask_ms'] for r in frames]), repeats_exact=bool(repeats_exact),
                  initializations=initializations, gpu=gpu_snapshot(), torch=torch.__version__,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  offline_preparation_and_reset_overhead_ms=overheads,
                  dino=dict(actual_calls=meter.calls, wall_ms=meter.wall_ms, parts_ms=meter.parts),
                  preflight=preflight, checks=checks,
                  boundary='in-memory raw RGB -> CPU boolean masks, with DINO when requested',
                  legacy_offline_boundary_exception='JPEG directory preparation outside frame timing; recorded separately',
                  pipeline_scope='separate actual synchronous pass; FP32 random policy, exact TRAIN96 normalizer, no optimizer/closed-loop')
    from graph.integration_baseline.rollout_checkpoint import vision_runtime_identity
    import importlib.metadata
    result['runtime'] = dict(vision_runtime=vision_runtime_identity(), cuda_runtime=torch.version.cuda,
        packages={package: importlib.metadata.version(package) for package in
                  ('torch','torchvision','numpy','pillow','transformers','sam-2','h5py')})
    atomic(folder/'frames.json', frames)
    atomic(folder/'result.json', result)


def summarize(run, plan):
    results = {name: read(run/name/'result.json') for name in plan['variants'] if (run/name/'result.json').exists()}
    comparisons = {}
    if 'direct' in results:
        for name in results:
            differing, presence, totals = 0, 0, 0
            for eid in plan['episodes']:
                with np.load(run/'direct'/f'ep{eid}_masks.npz') as a, np.load(run/name/f'ep{eid}_masks.npz') as b:
                    differing += int(np.count_nonzero(a['masks'] != b['masks']))
                    presence += int(np.count_nonzero(a['masks'].any(axis=(2,3)) != b['masks'].any(axis=(2,3))))
                    totals += int(a['masks'].size)
            comparison = dict(differing_mask_pixels=differing, compared_pixels=totals,
                              presence_differences=presence, masks_exact=differing==0,
                              initialization_equal=results[name]['initializations']==results['direct']['initializations'])
            if (run/name/'actions.npz').exists():
                with np.load(run/'direct/actions.npz') as a, np.load(run/name/'actions.npz') as b:
                    comparison['action_max_abs_diff'] = float(np.max(np.abs(a['actions']-b['actions'])))
                a, b = read(run/'direct/pipeline_frames.json'), read(run/name/'pipeline_frames.json')
                for field in ('packet_digest','graph_digest'):
                    comparison[field+'_different_frames'] = sum(x[field]!=y[field] for x,y in zip(a,b))
            comparisons[name] = comparison
    gates = dict(status='INCOMPLETE', masks_exact=False, adapter_overhead_pass=False,
                 tracking_20hz_pass=False, real_weight_checks_pass=False,
                 repeatability_pass=False, downstream_exact=False)
    if 'direct' in results and 'adapter' in results:
        a, b = (results[n]['groups']['sam2_tracking_only'] for n in ('direct','adapter'))
        overhead = {k: b[k]/a[k]-1 for k in ('median','p95')}
        gates.update(masks_exact=comparisons['adapter']['masks_exact'],
                     adapter_overhead_fraction=overhead, adapter_overhead_pass=all(v<=.10 for v in overhead.values()),
                     tracking_20hz_pass=b['p95']<=50,
                     real_weight_checks_pass=all(v['status']=='PASS' for v in results['adapter']['checks'].values()),
                     repeatability_pass=results['direct']['repeats_exact'] and results['adapter']['repeats_exact'],
                     downstream_exact=all(comparisons['adapter'].get(k) == 0 for k in (
                         'packet_digest_different_frames','graph_digest_different_frames','action_max_abs_diff')))
        restored = all(gates[k] for k in ('masks_exact','adapter_overhead_pass','real_weight_checks_pass',
                                        'repeatability_pass','downstream_exact'))
        gates['status'] = 'RESTORED' if restored else 'GATE_FAILED'
    summary = dict(run_id=run.name, status='COMPLETE' if len(results)==len(plan['variants']) else 'INCOMPLETE',
                   gates=gates, results=results, comparisons_to_direct=comparisons,
                   limits='Recorded TRAIN RGB-D; random FP32 policy; no task success, live-camera, or full-path 20Hz claim.')
    atomic(run/'summary.json', summary)
    lines = ['# Visual streaming integration validation', '', f'Run: `{run.name}`', '',
             f'Execution-path gate: **{gates["status"]}**. Tracking 20Hz: **{gates["tracking_20hz_pass"]}**.', '',
             '| Condition | Track p50 ms | Track p95 ms | DINO-scan mean ms |', '|---|---:|---:|---:|']
    for name, result in results.items():
        track, scan = result['groups']['sam2_tracking_only'], result['groups']['dino_scan']
        lines.append(f'| {name} | {track.get("median",0):.3f} | {track.get("p95",0):.3f} | {scan.get("mean",0):.3f} |')
    lines += ['', 'Tracking includes preprocessing and CPU mask return. Initial frames, compile and warmup are separate.',
              'Frozen-DINO ablations compare to tracking_reference; they do not measure detection latency.',
              'Ablation deltas have interactions and must not be added together.', '',
              'The profiling pass is separate and inclusive; nested intervals must not be summed.', '',
              summary['limits'], '', '[Full summary](summary.json) · [Frozen plan](plan.json)', '']
    (run/'report.md').write_text('\n'.join(lines))
    table = ''.join('<tr><td>'+escape(n)+'</td><td>'+f'{r["groups"]["sam2_tracking_only"].get("median",0):.3f}'+'</td><td>'+f'{r["groups"]["sam2_tracking_only"].get("p95",0):.3f}'+'</td></tr>' for n,r in results.items())
    (run/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Visual streaming validation</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}td,th{padding:10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>'
        '<h1>Visual streaming integration</h1><p>Run '+escape(run.name)+'</p><p>'+escape(summary['limits'])+'</p>'
        '<h2>Adoption gates</h2><pre>'+escape(json.dumps(gates,indent=2))+'</pre><table><tr><th>Condition</th><th>Track p50 ms</th><th>Track p95 ms</th></tr>'+table+
        '</table><p>Preprocess through CPU mask return. Warmup and profiling are excluded. Frozen-DINO conditions are controlled ablations; compare them to tracking_reference.</p>'
        '<h2>Output differences</h2><pre>'+escape(json.dumps(comparisons,indent=2))+'</pre><p><a href="summary.json">Full summary</a> · <a href="plan.json">Plan</a> · <a href="report.md">Markdown</a></p></html>')
    return summary


def run(args):
    from utils.experiment_logger import ExperimentLogger
    dataset = args.dataset.resolve()
    plan = dict(dataset=str(dataset), source_dataset_id=read(dataset/'dataset_manifest.json')['dataset_id'],
                dataset_sha256=sha(dataset/'dataset_manifest.json'), episodes=[30002,30047],
                train_robot=str(args.train_robot.resolve()), train_robot_sha256=sha(args.train_robot),
                repeats=args.repeats, threads=args.threads,
                variants={n:v for n,v in VARIANTS.items() if n in args.variants}, seed=42,
                adoption=dict(mask_pixel_exact=True, adapter_overhead_max_fraction=.10, tracking_p95_budget_ms=50),
                task='reused Visual streaming validation', instance_id=args.instance_id,
                gpu_memory_fraction=args.gpu_memory_fraction,
                minimum_free_bytes=int(args.minimum_free_gb*1024**3),
                resource_policy='one worker; monitor foreign GPU PIDs each second; stop our worker only')
    provenance = args.train_robot.with_name('train_robot_provenance.json')
    proof = read(provenance)
    manifest, split = read(dataset/'dataset_manifest.json'), read(dataset/'split_manifest.json')
    expected = {e['episode_id']:e['sha256'] for e in manifest['episodes'] if e['episode_id'] in split['splits']['train']}
    if proof['source_dataset_id'] != plan['source_dataset_id'] or {e['episode_id']:e['sha256'] for e in proof['inputs']} != expected:
        raise ValueError('normalizer provenance must match the frozen TRAIN96 inputs')
    plan['train_robot_provenance_sha256'] = sha(provenance)
    with ExperimentLogger(team='integration', model='visual_stream_reuse', config=plan,
                          seed=42, job_type='latency', root=args.output_root) as logger:
        run = logger.directory.resolve(); print('RUN_DIR='+str(run), flush=True)
        atomic(run/'plan.json', plan)
        atomic(run/'train_robot_provenance.json', proof)
        paths = [p for p in subprocess.check_output(['git','ls-files'],text=True).splitlines()
                 if p.endswith('.py') and p.startswith(('visual/runtime/','visual/scripts/stream_v1','graph/','recurrent/','utils/','envs/'))]
        atomic(run/'source_hashes.json', {p:sha(ROOT/p) for p in paths})
        changed = ['visual/scripts/stream_v1.py', 'graph/integration_baseline/streaming_vision_adapter.py',
                   'graph/integration_baseline/streaming_vision_checks.py',
                   'graph/integration_baseline/streaming_vision_benchmark.py',
                   'graph/integration_baseline/direct_dataset.py', 'graph/integration_baseline/direct_errors.py',
                   'graph/integration_baseline/rollout_checkpoint.py', 'graph/integration_baseline/rollout_stop_clock.py',
                   'visual/pyproject.toml', 'visual/uv.lock']
        for relative in changed:
            target = run/'source_snapshot'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, target)
        failures = {}
        for name in plan['variants']:
            if shutil.disk_usage(run).free < plan['minimum_free_bytes']:
                raise OSError('disk reserve reached; existing files will not be removed')
            snapshot = gpu_snapshot()
            if snapshot['processes']:
                raise RuntimeError('GPU is occupied; no existing process will be stopped')
            folder = run/name; folder.mkdir()
            atomic(folder/'before_gpu.json', snapshot)
            with (folder/'worker.log').open('w') as log:
                process = subprocess.Popen([sys.executable,'-m',__spec__.name,'--worker',name,'--plan',str(run/'plan.json')],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                returncode, interference = monitor_worker(process, folder, plan['minimum_free_bytes'])
            if returncode:
                failures[name] = returncode
                atomic(run/'failures.json', failures)
            else:
                print('CONDITION_COMPLETE='+name, flush=True)
            if interference or (name == 'direct' and returncode):
                break
        summary = summarize(run, plan)
        for path in run.rglob('*'):
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in {
                    'metadata.json','config.yaml','manifest.json','metrics.json'} and 'source_snapshot' not in path.parts:
                logger.link_output(path, kind='trajectory' if path.suffix=='.npz' else 'metrics' if path.suffix=='.json' else 'reports')
        logger.record(validation_status=summary['status'], gates=summary['gates'], optimizer_updates=0,
                      source_dataset_id=dataset.name, published=False)
        logger.log({name: result['groups']['sam2_tracking_only'] for name,result in summary['results'].items()}, prefix='mask_latency_ms')
        if failures:
            logger.finish(failed=True)
        return 1 if failures or summary['gates']['status']!='RESTORED' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--train-robot', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('experiments'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--instance-id', type=int)
    parser.add_argument('--minimum-free-gb', type=float, default=2.)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.35)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--worker', choices=VARIANTS)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--prepare-robot-from', type=Path)
    parser.add_argument('--write-robot', type=Path)
    args = parser.parse_args()
    if args.prepare_robot_from:
        if args.write_robot is None:
            parser.error('--write-robot required with --prepare-robot-from')
        prepare_normalizer_input(args.prepare_robot_from, args.write_robot)
    elif args.worker:
        worker(args.plan, args.worker)
    else:
        if args.dataset is None or args.train_robot is None or args.repeats < 1:
            parser.error('--dataset, --train-robot and positive --repeats required')
        if not 0 < args.gpu_memory_fraction <= 1 or args.minimum_free_gb < 0:
            parser.error('valid GPU memory fraction and nonnegative disk reserve required')
        if 'direct' not in args.variants or 'adapter' not in args.variants:
            parser.error('direct and adapter conditions are required')
        raise SystemExit(run(args))


if __name__ == '__main__':
    main()
