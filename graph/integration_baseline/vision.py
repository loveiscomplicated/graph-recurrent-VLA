"""Offline causal masks -> unchanged Shape-First/PCA5/EE/Raw EDE helpers.

Existing research helpers remain read-only dependencies. This opt-in builder
does not promote their policies to a production default or use GT masks.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import cv2
import numpy as np
import torch

from graph.vision_input import receive_vision_packet
from graph.integration_baseline.io import write_json, sha256, save_npz, save_packet
from visual.runtime.grounding import GDino
from visual.runtime.initialization import detect_initial, SCAN_EVERY
from visual.runtime.tracking import SAM2Tracker, DEFAULT_CHECKPOINT
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.sampling import extract_sampling_evidence, extract_candidates, SamplingConfig
from visual.runtime.vision_packet import complete_vision_packet
from visual.runtime.intra_object_edges import observed_components
from visual.runtime.test import run_variable_rate_shape_first as shape_first
from visual.runtime.test.raw_ede_comparison_adapter import build_raw_ede


def causal_masks(rgb, detector, tracker, callback=None):
    """Invoke existing initialization on prefixes, never on a future suffix.

    The old episode-wide initializer may use the other object's *future*
    detection as an exclusion box for an earlier late initialization. Prefix
    evaluation + accepting only current-frame decisions removes that leakage.
    SAM2 iterators propagate forwards only; no backward/post-hoc refinement.
    """
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


def run(replay, output, device='cpu', camera='base_camera', episode_ids=None):
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise ValueError('HF_HUB_OFFLINE=1 required; use installed cached weights only')
    replay, output = Path(replay).resolve(), Path(output).resolve()
    source = json.loads((replay/'manifest.json').read_text())
    if source.get('robot_state_schema') is None or not source['episodes']:
        raise ValueError('no verified replay observations available')
    output.mkdir(parents=True, exist_ok=False)
    (output/'packets').mkdir()
    summary = dict(schema_version=1, status='RUNNING', replay_manifest=str(replay/'manifest.json'),
                   replay_manifest_sha256=sha256(replay/'manifest.json'), robot_state_schema=source['robot_state_schema'],
                   splits=source['splits'], source_sha256=source['source_sha256'],
                   policy=dict(keep_ratio=.5, phase=0, surface_additional=shape_first.SURFACE,
                               intra_object='raw_extended_delaunay', edge_validation='none'),
                   camera=camera, device=device, episodes=[], failures=[],
                   frames_expected=sum(e['observations'] for e in source['episodes']),
                   input_scope='current RGB-D, prefix-causal predicted masks, same-timestep robot only')
    write_json(output/'manifest.json', summary)
    try:
        torch.set_num_threads(1)
        detector = GDino(device=device)
        tracker = SAM2Tracker(device=device)
        for ep in source['episodes']:
            eid = ep['episode_id']
            if episode_ids is not None and eid not in episode_ids:
                continue
            path = replay/ep['path']
            if sha256(path) != ep['sha256']:
                raise ValueError('replay episode hash changed')
            episode_key = source['source_sha256'] + f':traj_{eid}'
            folder = output/'packets'/f'ep{eid:03d}'
            folder.mkdir()
            result = dict(episode_id=eid, episode_identity=episode_key, replay_path=str(path),
                          replay_sha256=ep['sha256'], packets=[], failures=[], mask_provenance=None,
                          source_episode_status=ep['status'])
            summary['episodes'].append(result)
            with np.load(path, allow_pickle=False) as arrays:
                rgb = arrays[camera+'/rgb']
                def process(t, masks, timing):
                    try:
                        frame = TrackedRGBDFrame(t, camera, rgb[t], arrays[camera+'/depth'][t],
                            arrays[camera+'/intrinsic'][t], arrays[camera+'/cam2world'][t], arrays['tcp_pose'][t],
                            {0: masks[0], 1: masks[1]}, int(arrays['timestep'][t]), float(arrays['timestamp'][t]))
                        packet, stages = packet_from_frame(frame, episode_key)
                        packet['observation_provenance'] = dict(source_sha256=source['source_sha256'], episode_id=eid,
                            timestep=t, replay_sha256=ep['sha256'], mask_source='prefix_causal_rgb_gdino_forward_sam2',
                            rgb_sha256=__import__('hashlib').sha256(rgb[t].tobytes()).hexdigest())
                        packet_path = folder/f'f{t:04d}.npz'
                        save_packet(packet_path, packet)
                        result['packets'].append(dict(timestep=t, path=str(packet_path.relative_to(output)),
                            sha256=sha256(packet_path), point_count=len(packet['points']),
                            object_counts=[int((packet['object_id']==o).sum()) for o in (0, 1)],
                            timings_ms=dict(**stages, mask_ms=timing['elapsed_ms']),
                            has_action_target=t < len(arrays['actions'])))
                    except Exception as exc:
                        result['failures'].append(dict(timestep=t, type=type(exc).__name__, reason=str(exc)))
                        print(f'graph failure eid={eid} t={t}: {type(exc).__name__}: {exc}', flush=True)
                    write_json(output/'manifest.json', summary)
                masks, provenance = causal_masks(rgb, detector, tracker, process)
                mask_path = output/f'ep{eid:03d}_masks.npz'
                save_npz(mask_path, peg=masks[:, 0], box=masks[:, 1], source_frame=arrays['timestep'],
                         replay_sha256=np.array(ep['sha256']), episode_id=np.array(eid, np.int64))
                result['mask_provenance'] = provenance
                result.update(mask_path=str(mask_path.relative_to(output)), mask_sha256=sha256(mask_path))
                write_json(output/'manifest.json', summary)
    except Exception as exc:
        summary['failures'].append(dict(type=type(exc).__name__, reason=str(exc)))
        print(f'vision blocked: {type(exc).__name__}: {exc}', flush=True)
    count = sum(len(e['packets']) for e in summary['episodes'])
    summary.update(status='REAL_ALIGNED' if count == summary['frames_expected'] else 'PARTIAL_REAL_ALIGNED' if count else 'BLOCKED_DATA',
                   frames_built=count, invalid_frames=sum(len(e['failures']) for e in summary['episodes']),
                   missing_frames=summary['frames_expected']-count)
    write_json(output/'manifest.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run(args.replay, args.output, args.device)


if __name__ == '__main__':
    main()
