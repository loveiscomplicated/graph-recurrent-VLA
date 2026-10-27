"""Opt-in integration of Visual's existing causal GPU StreamTracker.

Detection remains the prefix initializer. Only raw DINO results are memoized;
prefix-dependent assignment is never cached. SAM2 owns one streaming state and
receives new DINO boxes on the current observation before propagation.
"""
from time import perf_counter

import numpy as np
import torch

from graph.integration_baseline.vision_speed import MemoizedDetector
from visual.runtime.initialization import detect_initial, SCAN_EVERY


BACKEND = 'visual_stream'


class PrefixInitialization:
    """Reuse existing prefix decisions without rerunning past DINO inference."""

    def __init__(self, detector):
        self.detector = MemoizedDetector(detector)
        self.reset()

    def reset(self):
        self.detector.reset_episode()
        self.rgb, self.found = [], {}
        self.scan_events = 0

    def push(self, rgb):
        t = len(self.rgb)
        self.rgb.append(rgb.copy())
        new = {}
        if t % SCAN_EVERY == 0 and len(self.found) < 2:
            self.scan_events += 1
            initial = detect_initial(self.rgb, self.detector)
            new = {i + 1: initial[name]['box'] for i, name in enumerate(('peg', 'box'))
                   if name not in self.found and initial[name]['frame'] == t}
            for i, name in enumerate(('peg', 'box')):
                if i + 1 in new:
                    self.found[name] = dict(initial[name], decision_observation=t, prefix_length=t + 1)
        return new


def build_stream_tracker(device='cuda', *, compile_image_encoder=True,
                         preprocess='gpu', prune=True):
    """Use Visual's SAM2 builder and streaming implementation, without a fork."""
    from visual.runtime.tracking import SAM2Tracker
    from visual.scripts.stream_v1 import StreamTracker

    if torch.device(device).type != 'cuda':
        raise ValueError('visual_stream requires CUDA; legacy_fp32 remains available on CPU')
    predictor = SAM2Tracker(device=device, hydra_overrides=[
        f'++model.compile_image_encoder={bool(compile_image_encoder)}']).predictor
    return StreamTracker(predictor, preprocess=preprocess, prune=prune,
                         keep_mem=16, keep_frames=2, observed_horizon=True)


class StreamVisionAdapter:
    """Same reset/process/close and boolean mask boundary as OnlineVisionAdapter."""

    def __init__(self, detector, tracker, *, bf16=True):
        if not tracker.observed_horizon:
            raise ValueError('visual_stream requires observed_horizon=True')
        if bf16 and torch.device(tracker.p.device).type != 'cuda':
            raise ValueError('BF16 streaming requires CUDA')
        self.tracker, self.bf16 = tracker, bf16
        self.initializer = PrefixInitialization(detector)
        self.reset(None)

    @property
    def found(self):
        return self.initializer.found

    def reset(self, episode_identity):
        self.tracker.reset()
        self.initializer.reset()
        self.identity = episode_identity
        self.records = []
        self.closed = self.failed = False

    def close(self):
        self.tracker.reset()
        self.initializer.reset()
        self.closed = True

    @torch.inference_mode()
    def process(self, rgb, episode_identity, timestep):
        if self.closed or self.failed:
            raise RuntimeError('reset required after close or failed streaming inference')
        history = self.initializer.rgb
        if episode_identity != self.identity or timestep != len(history):
            raise ValueError('online vision episode/timestep discontinuity; explicit reset required')
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('uint8 H,W,3 live RGB required')
        if history and rgb.shape != history[0].shape:
            raise ValueError('camera dimensions changed within episode')
        tick = perf_counter()
        try:
            # DINO is outside the SAM2 autocast scope, as in Visual's callers.
            new = self.initializer.push(rgb)
            detected = perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
                if timestep == 0:
                    values = self.tracker.start(rgb, new)
                else:
                    frame, values = self.tracker.step(rgb, boxes=new)
                    if frame != timestep:
                        raise ValueError('SAM2 current frame identity mismatch')
            tracked = perf_counter()
            expected = {i + 1 for i, name in enumerate(('peg', 'box')) if name in self.found}
            if set(values) != expected:
                raise ValueError('SAM2 current object identity mismatch')
            masks = np.zeros((2, *rgb.shape[:2]), dtype=bool)
            for oid, value in values.items():
                mask = value.detach().cpu().numpy()
                if mask.dtype != bool or mask.shape != rgb.shape[:2]:
                    raise ValueError('SAM2 mask dtype/shape mismatch')
                masks[oid - 1] = mask
        except Exception:
            self.failed = True
            raise
        done = perf_counter()
        statistics = self.initializer.detector.statistics()
        record = dict(timestep=timestep, prefix_length=timestep + 1,
                      elapsed_ms=(done - tick) * 1000,
                      # Host intervals; use the separate synchronized profiler
                      # for CUDA stage attribution, not these enqueue intervals.
                      host_intervals_ms=dict(detection=(detected - tick) * 1000,
                                             tracking_enqueue=(tracked - detected) * 1000,
                                             mask_return=(done - tracked) * 1000),
                      mask_pixels=masks.sum(axis=(1, 2)).tolist(),
                      new_object_ids=list(new), detector_calls=statistics['actual_inferences'],
                      detector_requests=statistics['requests'],
                      initialization_scan_events=self.initializer.scan_events,
                      tracking_groups=[list(self.tracker.obj_ids)] if expected else [],
                      backend=BACKEND)
        self.records.append(record)
        return masks, record


def make_online_vision(config, *, detector=None, tracker=None):
    """Missing backend preserves every existing run's legacy constructor."""
    backend = config.get('vision', {}).get('backend', 'legacy_fp32')
    if backend not in ('legacy_fp32', BACKEND):
        raise ValueError('unknown vision backend: ' + str(backend))
    device = config['vision_device']
    if backend == BACKEND and torch.device(device).type != 'cuda':
        raise ValueError('visual_stream requires CUDA')
    if detector is None:
        from visual.runtime.grounding import GDino
        detector = GDino(device=device)
    if backend == BACKEND:
        return StreamVisionAdapter(detector, tracker if tracker is not None else build_stream_tracker(device))
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    if tracker is None:
        from visual.runtime.tracking import SAM2Tracker
        tracker = SAM2Tracker(device=device)
    return OnlineVisionAdapter(detector, tracker)


def stream_masks(rgb, detector, tracker, callback=None, *, episode_identity='recorded'):
    """Offline caller revealing RGB through exactly the online adapter boundary."""
    from graph.integration_baseline.io import sha256
    from visual.runtime.tracking import DEFAULT_CHECKPOINT

    adapter = StreamVisionAdapter(detector, tracker)
    adapter.reset(episode_identity)
    masks = np.zeros((len(rgb), 2, *rgb.shape[1:3]), dtype=bool)
    try:
        for t, image in enumerate(rgb):
            masks[t], record = adapter.process(image, episode_identity, t)
            if callback is not None:
                callback(t, masks[t], record)
        provenance = dict(source='RGB-only DINO + Visual StreamTracker', backend=BACKEND,
                          checkpoint_sha256=sha256(DEFAULT_CHECKPOINT),
                          gt_segmentation_used=False, future_observations_used=False,
                          sam2_horizon='num_frames=t+1; one-frame forward; shared object state',
                          precision='BF16 autocast; FP32 weights', preprocess=tracker.pre,
                          keep_mem=tracker.keep_mem, keep_frames=tracker.keep_frames,
                          initialization=dict(adapter.found), frames=list(adapter.records),
                          detector=adapter.initializer.detector.statistics())
    finally:
        adapter.close()
    return masks, provenance
