"""Append-only RGB input for the existing prefix-causal DINO/SAM2 algorithm.

SAM2's directory loader is not live. Replace only its image container with an
append-only, bounds-checked sequence, using its own JPEG preprocessing. Predictor
features, object memories, prompts and propagation code are left unchanged.
Separate late-object states deliberately match vision.causal_masks().
"""
from pathlib import Path
import tempfile
from time import perf_counter

import cv2
import numpy as np
import torch

from vla.perception.initialization import detect_initial, SCAN_EVERY


class ObservedFrames:
    def __init__(self):
        self.frames = []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError('unobserved SAM2 frame requested')
        return self.frames[index]

    def append_jpeg(self, path, predictor):
        from sam2.utils.misc import _load_img_as_tensor
        image, height, width = _load_img_as_tensor(str(path), predictor.image_size)
        image = image.to(predictor.device)
        mean = image.new_tensor((.485, .456, .406))[:, None, None]
        std = image.new_tensor((.229, .224, .225))[:, None, None]
        self.frames.append((image - mean) / std)
        return height, width


class OnlineVisionAdapter:
    """One call per newly observed RGB; no episode video/length input exists."""
    def __init__(self, detector, tracker):
        bind = getattr(detector, 'new_stream', None)
        self.detector = bind() if callable(bind) else detector
        self.tracker = tracker
        self.directory = None
        self.states = {}
        self.reset(None)

    def reset(self, episode_identity):
        reset_initialization = getattr(self.detector, 'reset_initialization', None)
        if callable(reset_initialization):reset_initialization()
        for state in self.states.values():
            self.tracker.reset(state)
        if self.directory is not None:
            self.directory.cleanup()
        self.directory = tempfile.TemporaryDirectory(prefix='task07_observed_')
        self.identity = episode_identity
        self.states, self.found, self.rgb = {}, {}, []
        self.images = ObservedFrames()
        self.records = []
        self.detector_calls = 0
        self.scan_events = 0

    def close(self):
        for state in self.states.values():
            self.tracker.reset(state)
        self.states.clear()
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    @torch.inference_mode()
    def process(self, rgb, episode_identity, timestep, *, mask_device='cpu'):
        """Existing causal vision, with optional CUDA raw masks for robot filtering.

        CPU remains the legacy output default. Current PlanA cache/live frontends
        request CUDA so no dense mask is downloaded before their GPU filter.
        Only the existing two pixel-count scalars are read for the record.
        """
        if mask_device not in ('cpu','cuda'):raise ValueError('mask_device must be cpu or cuda')
        if episode_identity != self.identity or timestep != len(self.rgb):
            raise ValueError('online vision episode/timestep discontinuity; explicit reset required')
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('uint8 H,W,3 live RGB required')
        if self.rgb and rgb.shape != self.rgb[0].shape:
            raise ValueError('camera dimensions changed within episode')
        tick = perf_counter()
        path = Path(self.directory.name) / f'{timestep:06d}.jpg'
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError('SAM2 JPEG write failed')
        self.rgb.append(rgb.copy())
        if self.images.append_jpeg(path, self.tracker.predictor) != rgb.shape[:2]:
            raise ValueError('SAM2 image dimensions differ')

        def detect(image, prompt):
            self.detector_calls += 1
            return self.detector(image, prompt)

        new = {}
        if timestep % SCAN_EVERY == 0 and len(self.found) < 2:
            self.scan_events += 1
            joint = callable(getattr(self.detector, 'initialize', None))
            if joint:
                self.detector_calls += 1
            initial = detect_initial(self.rgb, self.detector if joint else detect)
            new = {i+1: initial[name]['box'] for i, name in enumerate(('peg', 'box'))
                   if name not in self.found and initial[name]['frame'] == timestep}
            if new:
                # This directory contains observed frames only. init_state may
                # warm frame zero, exactly as the offline baseline does.
                state = self.tracker.start(self.directory.name, new, frame_idx=timestep)
                required = {'images', 'num_frames', 'cached_features', 'output_dict_per_obj',
                            'frames_tracked_per_obj', 'obj_ids'}
                if not required.issubset(state):
                    raise RuntimeError('BLOCKED_ONLINE_VISION: unsupported SAM2 state schema')
                state['images'] = self.images
                state['num_frames'] = timestep+1
                self.states[tuple(new)] = state
                for i, name in enumerate(('peg', 'box')):
                    if i+1 in new:
                        self.found[name] = dict(initial[name], decision_observation=timestep,
                                                prefix_length=timestep+1)
        masks = (np.zeros((2, *rgb.shape[:2]), bool) if mask_device=='cpu' else
            torch.zeros((2,*rgb.shape[:2]),device='cuda',dtype=torch.bool))
        for group, state in self.states.items():
            state['num_frames'] = timestep+1
            iterator = self.tracker.track(state, start_frame=timestep)
            try:
                frame, values = next(iterator)
            finally:
                iterator.close()
            if frame != timestep or set(values) != set(group):
                raise ValueError('SAM2 current frame/object identity mismatch')
            for oid in group:
                mask = (values[oid].detach().cpu().numpy() if mask_device=='cpu' else values[oid].detach().to(masks.device))
                if mask.dtype != (bool if mask_device=='cpu' else torch.bool) or mask.shape != rgb.shape[:2]:
                    raise ValueError('SAM2 mask dtype/shape mismatch')
                masks[oid-1] = mask
        record = dict(timestep=timestep, prefix_length=len(self.images),
                      elapsed_ms=(perf_counter()-tick)*1000,
                      mask_pixels=(masks.sum(axis=(1,2)).tolist() if mask_device=='cpu' else masks.sum(dim=(1,2)).tolist()),
                      new_object_ids=list(new), detector_calls=self.detector_calls,
                      initialization_scan_events=self.scan_events,
                      tracking_groups=[list(g) for g in self.states])
        if callable(getattr(self.detector, 'status', None)):
            record['grounding'] = self.detector.status()
        self.records.append(record)
        return masks, record
