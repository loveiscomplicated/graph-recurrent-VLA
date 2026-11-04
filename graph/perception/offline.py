"""Prefix-causal recorded masks and CPU Graph packet construction."""
from pathlib import Path
import tempfile
from time import perf_counter
import cv2
import numpy as np
from graph.vision_input import receive_vision_packet
from vla.data.legacy_io import sha256
from vla.perception.initialization import detect_initial, SCAN_EVERY
from vla.perception.tracking import DEFAULT_CHECKPOINT
from graph.perception.sampling import extract_sampling_evidence, extract_candidates, SamplingConfig
from graph.perception.vision_packet import complete_vision_packet
from graph.perception.intra_object_edges import observed_components
import graph.perception.shape_first as shape_first
from graph.perception.raw_ede import build_raw_ede

def causal_masks(rgb, detector, tracker, callback=None):
    """Invoke existing initialization on prefixes, never on a future suffix.

    The old episode-wide initializer may use the other object's *future*
    detection as an exclusion box for an earlier late initialization. Prefix
    evaluation + accepting only current-frame decisions removes that leakage.
    SAM2 iterators propagate forwards only; no backward/post-hoc refinement.
    """
    bind = getattr(detector, 'new_stream', None)
    if callable(bind):detector = bind()
    found, states = {}, {}
    records = []
    masks = np.zeros((len(rgb), 2, *rgb.shape[1:3]), bool)
    with tempfile.TemporaryDirectory(prefix='task06_sam2_') as directory:
        for t, image in enumerate(rgb):
            cv2.imwrite(str(Path(directory)/f'{t:06d}.jpg'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        try:
            for t in range(len(rgb)):
                tick = perf_counter()
                if t % SCAN_EVERY == 0 and len(found) < 2:
                    initial = detect_initial(rgb[:t+1], detector)
                    new = {i+1: initial[name]['box'] for i, name in enumerate(('peg', 'box'))
                           if name not in found and initial[name]['frame'] == t}
                    if new:
                        state = tracker.start(directory, new, frame_idx=t)
                        key = tuple(new)
                        states[key] = state
                        for i, name in enumerate(('peg', 'box')):
                            if i+1 in new:
                                found[name] = dict(initial[name], decision_observation=t, prefix_length=t+1)
                for group, state in states.items():
                    # SAM2 normalizes temporal object-pointer encodings with
                    # min(num_frames,max_obj_ptrs). A full-video num_frames
                    # therefore leaks the future horizon even in forward mode.
                    # Use a fresh one-frame propagation on the current horizon.
                    # The existing predictor/cache/weights remain unchanged.
                    state['num_frames'] = t+1
                    iterator = tracker.track(state, start_frame=t)
                    try:
                        frame_id, values = next(iterator)
                    finally:
                        iterator.close()
                    if frame_id != t:
                        raise ValueError('SAM2 forward propagation frame mismatch')
                    for oid in group:
                        mask = values[oid].detach().cpu().numpy()
                        if mask.shape != rgb.shape[1:3]:
                            raise ValueError('mask shape mismatch; no implicit resizing')
                        masks[t, oid-1] = mask
                records.append(dict(timestep=t, prefix_length=t+1, elapsed_ms=(perf_counter()-tick)*1000,
                                    mask_pixels=masks[t].sum(axis=(1, 2)).tolist()))
                if callback is not None:
                    callback(t, masks[t], records[-1])
                if t % 20 == 0:
                    print(f'causal masks t={t}/{len(rgb)} found={list(found)}', flush=True)
        finally:
            for state in states.values():
                tracker.reset(state)
    return masks, dict(source='RGB-only Grounding DINO + forward SAM2',
                       checkpoint_sha256=sha256(DEFAULT_CHECKPOINT),
                       gt_segmentation_used=False, future_observations_used=False,
                       sam2_horizon='num_frames=t+1; one-frame forward propagation; no future total-length normalization',
                       initialization=found, frames=records,
                       jpeg_input='existing SAM2 directory interface, cv2 default JPEG codec')

def packet_from_frame(frame, episode, keep_ratio=.5):
    tick = perf_counter()
    evidence = extract_sampling_evidence(frame, estimator='pca5')
    after_evidence = perf_counter()
    candidates = extract_candidates(evidence, SamplingConfig(), method='S3')
    prepared, selected, cases = {}, {}, {}
    geometry = evidence.frame.geometry
    for oid in (0, 1):
        obj = candidates[oid]
        orientation = evidence.pairs[oid].candidates(20)['orientation']
        pca = obj.flat_index[orientation.ravel()[obj.flat_index]]
        prepared[oid] = shape_first.prepare_object(obj.flat_index, obj.points, pca, frame.masks[oid],
            geometry.valid_depth, geometry.overlap_mask, frame.intrinsic)
        selected[oid] = shape_first.select_object(prepared[oid], keep_ratio, 0, shape_first.SURFACE[oid])
        cases[oid] = dict(saved_flat=obj.flat_index, saved_features=obj.metadata)
    selection, _ = shape_first.make_packet(frame.source_frame, prepared, selected, cases)
    selection.update(episode=episode, camera=frame.camera)
    after_selection = perf_counter()
    packet = complete_vision_packet(selection, evidence, frame)
    after_feature = perf_counter()
    maps, _ = observed_components(evidence)
    component = np.empty(len(packet['points']), np.int64)
    for oid in (0, 1):
        rows = packet['object_id'] == oid
        component[rows] = maps[oid].ravel()[packet['provenance']['source_index'][rows]]
    packet['observed_component_id'] = component
    edge, metadata, _ = build_raw_ede(packet)
    packet.update(metadata, edge_status='supplied', intra_object_edge_index=edge)
    receive_vision_packet(packet)
    return packet, dict(pca5_evidence_ms=(after_evidence-tick)*1000,
                         shape_first_ms=(after_selection-after_evidence)*1000,
                         feature_ee_ms=(after_feature-after_selection)*1000,
                         component_raw_ede_ms=(perf_counter()-after_feature)*1000,
                         total_offline_graph_ms=(perf_counter()-tick)*1000)
