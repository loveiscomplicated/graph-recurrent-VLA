"""Real-weight checks for Visual streaming; no simulator or policy actions."""
from pathlib import Path

import numpy as np
import torch

from graph.integration_baseline.direct_runtime import atomic
from graph.integration_baseline.streaming_vision_adapter import PrefixInitialization, StreamVisionAdapter
from visual.scripts.stream_v1 import FutureFrameAccess


def direct_sequence(rgb, detector, tracker):
    """Independent direct StreamTracker loop using the same causal DINO rules."""
    tracker.reset()
    initialization = PrefixInitialization(detector)
    masks = np.zeros((len(rgb), 2, *rgb.shape[1:3]), dtype=bool)
    try:
        with torch.inference_mode():
            for t, image in enumerate(rgb):
                boxes = initialization.push(image)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    values = tracker.start(image, boxes) if t == 0 else tracker.step(image, boxes=boxes)[1]
                if tracker.state['num_frames'] != t + 1:
                    raise AssertionError('non-prefix SAM2 horizon')
                for oid, value in values.items():
                    masks[t, oid - 1] = value.cpu().numpy()
        return masks, initialization.found
    finally:
        tracker.reset()


def compare_stream_sequence(rgb, detector, tracker, output):
    """Recorded RGB plus controlled late prompts, explicitly distinct from live QA."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reference, found = direct_sequence(rgb, detector, tracker)
    adapter = StreamVisionAdapter(detector, tracker)

    def online(sequence, identity):
        adapter.reset(identity)
        masks = np.stack([adapter.process(image, identity, t)[0] for t, image in enumerate(sequence)])
        return masks, dict(adapter.found)

    try:
        current, current_found = online(rgb, 'recorded')
        state = tracker.state
        bounds = dict(prefix_horizon=state['num_frames'] == len(rgb),
                      retained_frames=len(tracker.frames.buf),
                      max_noncond_rows=max((len(o['non_cond_frame_outputs'])
                                          for o in state['output_dict_per_obj'].values()), default=0))
        future_guard = False
        try:
            tracker.frames[len(rgb)]
        except FutureFrameAccess:
            future_guard = True
        split = max(1, len(rgb) // 2)
        changed = rgb[:min(len(rgb), split + 2)].copy()
        changed[split:] = 255 - changed[split:]
        suffix_values, _ = online(changed, 'changed_suffix')
        prefix_values, _ = online(rgb[:split], 'prefix_only')
        differences = np.count_nonzero(current != reference, axis=(2, 3))
        result = dict(status='PASS', frames=len(rgb), masks_exact=not bool(differences.any()),
                      initialization_equal=current_found == found,
                      suffix_prefix_exact=np.array_equal(current[:split], suffix_values[:split]),
                      prefix_only_exact=np.array_equal(current[:split], prefix_values),
                      future_access_rejected=future_guard, memory=bounds,
                      scope='REAL_SAM2_RECORDED_RGB; not simulator/live-camera or task success')
        # Controlled prompts test SAM2 registration independently of detection
        # accuracy. Their source boxes were obtained from observed RGB above.
        H, W = rgb.shape[1:3]
        boxes = {i + 1: found[name]['box'] if name in found else
                 [W * (.1 + .4 * i), H * .2, W * (.25 + .4 * i), H * .4]
                 for i, name in enumerate(('peg', 'box'))}
        tracker.reset()
        fixture = [rgb[t % len(rgb)] for t in range(24)]
        counter = [0]
        original = tracker.p.image_encoder.forward

        def count(*args, **kwargs):
            counter[0] += 1
            return original(*args, **kwargs)

        tracker.p.image_encoder.forward = count
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                values = tracker.start(fixture[0], {1: boxes[1]})
                first_state = tracker.state
                first_memory = first_state['output_dict_per_obj'][0]
                first_conditioning = dict(first_memory['cond_frame_outputs'])
                before = set()
                retained = True
                for t in range(1, len(fixture)):
                    if t == 5:
                        before = set(first_memory['non_cond_frame_outputs'])
                    _, values = tracker.step(fixture[t], boxes={2: boxes[2]} if t == 5 else None)
                    if t == 5:
                        retained = (tracker.state is first_state and
                                    tracker.state['output_dict_per_obj'][0] is first_memory and
                                    before <= set(first_memory['non_cond_frame_outputs']))
                late = dict(existing_memory_retained=retained,
                            conditioning_retained=all(first_memory['cond_frame_outputs'].get(k) is v
                                                      for k, v in first_conditioning.items()),
                            both_objects_registered=set(values) == {1, 2},
                            image_encoder_calls=counter[0], frames=len(fixture),
                            one_image_encode_per_frame=counter[0] == len(fixture),
                            max_noncond_rows=max(len(o['non_cond_frame_outputs'])
                                                for o in tracker.state['output_dict_per_obj'].values()),
                            scope='CONTROLLED_LATE_PROMPT_FIXTURE; not detection-quality evidence')
        finally:
            tracker.p.image_encoder.forward = original
            tracker.reset()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            empty = not tracker.start(rgb[0], {})
            for image in rgb[1:min(6, len(rgb))]:
                empty = empty and not tracker.step(image)[1]
        result['late_object'] = late
        result['empty_initialization'] = bool(empty)
        gates = [result[k] for k in ('masks_exact', 'initialization_equal', 'suffix_prefix_exact',
                                    'prefix_only_exact', 'future_access_rejected', 'empty_initialization')]
        gates.extend(late[k] for k in ('existing_memory_retained', 'conditioning_retained',
                                      'both_objects_registered', 'one_image_encode_per_frame'))
        gates.extend([bounds['retained_frames'] <= tracker.keep_frames,
                      bounds['max_noncond_rows'] <= tracker.keep_mem,
                      late['max_noncond_rows'] <= tracker.keep_mem])
        if not all(gates):
            result['status'] = 'CONTRACT_FAILURE'
        np.savez_compressed(output / 'stream_comparison.npz', direct=reference, adapter=current,
                            differing_pixels=differences)
        atomic(output / 'stream_comparison.json', result)
        return result
    finally:
        adapter.close()
