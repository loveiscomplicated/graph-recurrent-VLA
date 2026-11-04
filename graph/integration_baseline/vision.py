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


from graph.perception.offline import causal_masks, packet_from_frame




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
