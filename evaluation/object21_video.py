"""Render saved Object21 validation predictions on original camera frames."""

import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.vertex_family_manifest import read_manifest
from evaluation.fixed_id_corner_video import project_world, put_text, read_camera
from utils.experiment_logger import ExperimentLogger, write


def quaternion_matrix(value):
    """Convert a wxyz quaternion to a local-to-world rotation matrix."""
    quaternion = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Object21 contains an invalid quaternion')
    w, x, y, z = quaternion / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def geometry(object21, tcp):
    """Decode TCP-relative world-axis Object21 into world-space primitives."""
    obj = np.asarray(object21)
    if obj.shape != (21,) or not np.isfinite(obj).all() or np.any(obj[17:21] <= 0):
        raise ValueError('Expected a finite Object21 with positive half-sizes and radius')
    peg = obj[:3] + tcp
    hole = obj[10:13] + tcp
    peg_rotation = quaternion_matrix(obj[3:7])
    hole_rotation = quaternion_matrix(obj[13:17])
    signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    corners = peg + (signs * obj[17:20]) @ peg_rotation.T
    angles = np.linspace(0, 2*np.pi, 49)
    local_ring = np.stack((np.zeros_like(angles), np.cos(angles), np.sin(angles)), axis=1)
    ring = hole + obj[20] * local_ring @ hole_rotation.T
    axes = hole + obj[20] * hole_rotation.T
    endpoint = hole + obj[7:10]
    return dict(corners=corners, ring=ring, hole=hole, peg=peg, axes=axes, endpoint=endpoint)


def segment(image, a, b, intrinsic, extrinsic, color, thickness=1):
    """Draw a projected segment only if both endpoints lie in view."""
    first = project_world(a, intrinsic, extrinsic)
    second = project_world(b, intrinsic, extrinsic)
    if first is not None and second is not None:
        cv2.line(image, first, second, color, thickness, cv2.LINE_AA)


def draw_geometry(image, values, intrinsic, extrinsic, color, predicted):
    """Draw peg wireframe, hole circle and the independently predicted endpoint."""
    corners = values['corners']
    for left in range(8):
        for right in range(left+1, 8):
            if (left ^ right) in (1, 2, 4):
                segment(image, corners[left], corners[right], intrinsic, extrinsic, color, 2 if predicted else 1)
    for a, b in zip(values['ring'][:-1], values['ring'][1:]):
        segment(image, a, b, intrinsic, extrinsic, color, 2 if predicted else 1)
    # A ring alone does not show roll about the hole axis: show local X/Y/Z.
    for axis in values['axes']:
        segment(image, values['hole'], axis, intrinsic, extrinsic, color, 1)
    for key in ('peg', 'hole', 'endpoint'):
        pixel = project_world(values[key], intrinsic, extrinsic)
        if pixel is not None:
            if key == 'endpoint':
                cv2.drawMarker(image, pixel, color, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
            else:
                cv2.circle(image, pixel, 2, color, -1, cv2.LINE_AA)


def load_data(config):
    """Verify stored leaderboard predictions, shared targets, and TCP alignment."""
    comparison_path = Path(config['comparison'])
    if file_sha256(comparison_path) != config['comparison_sha256']:
        raise ValueError('Comparison checksum mismatch')
    comparison = json.loads(comparison_path.read_text())
    if not comparison['same_frame_ids_and_gt_as_reference']:
        raise ValueError('Predictions are not aligned to the reference validation set')
    predictions = {}
    for model in ('mamba', 'transformer'):
        result = next(item for item in comparison['results']
                      if item['architecture'] == model and item['seed'] == config['seed'])
        path = comparison_path.parent / f"{model}_seed_{config['seed']}_predictions.npz"
        if file_sha256(path) != result['prediction_sha256']:
            raise ValueError(f'{model} prediction checksum mismatch')
        with np.load(path, allow_pickle=False) as data:
            predictions[model] = {key: data[key].copy() for key in data.files}
    first, second = predictions['mamba'], predictions['transformer']
    if not np.array_equal(first['sample_ids'], second['sample_ids']) or not np.array_equal(first['target'], second['target']):
        raise ValueError('Model predictions do not share frame IDs and target')
    cache_path = Path(config['rollout_cache'])
    if file_sha256(cache_path) != config['rollout_cache_sha256']:
        raise ValueError('Rollout cache checksum mismatch')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)['arrays']
    mask = cache['split']
    if not np.array_equal(cache['sample_ids'][mask], first['sample_ids']) or not np.array_equal(cache['target'][mask], first['target']):
        raise ValueError('Rollout TCP cache does not align with validation predictions')
    return predictions, cache['tcp'][mask], comparison


def render(config):
    """Write one side-by-side video per episode from existing evaluated outputs."""
    family_root = Path(config['family_dataset']).resolve()
    family = read_manifest(family_root)
    if family['dataset_sha256'] != config['family_dataset_sha256']:
        raise ValueError('Family dataset checksum mismatch')
    predictions, tcp, comparison = load_data(config)
    ids = predictions['mamba']['sample_ids']
    rows = {row['sample_id']: row for row in family['samples']}
    if any(rows[str(sample_id)]['split'] != 'validation' for sample_id in ids):
        raise ValueError('Expected validation frames only')
    with ExperimentLogger(team='graph', model='object21_video', config=config,
                          seed=config['seed'], job_type='visualization') as logger:
        output = Path(config['output']['root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        summaries = []
        for item in config['episodes']:
            indices = [i for i, sample_id in enumerate(ids) if str(sample_id).startswith(item['episode'] + '/')]
            if not indices:
                raise ValueError(f"No validation frames for {item['episode']}")
            path = output / (item['name'] + '.mp4')
            command = ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
                       '-pixel_format','bgr24','-video_size','1024x704','-framerate',str(config['fps']),
                       '-i','-','-an','-c:v','libx264','-preset','fast','-crf','18',
                       '-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            try:
                for frame, index in enumerate(indices):
                    image, intrinsic, extrinsic = read_camera(family_root, rows[str(ids[index])])
                    target = predictions['mamba']['target'][index]
                    canvas = np.full((704,1024,3), (28,25,23), dtype=np.uint8)
                    put_text(canvas, 'OBJECT21 VALIDATION  |  SEED 0  |  ORIGINAL CAMERA', (15,25), .56, thickness=2)
                    put_text(canvas, 'Green: GT    Magenta: prediction    Cross: endpoint    Lines: hole axes', (15,50), .49)
                    put_text(canvas, 'Peg box / hole circle use predicted position, rotation and size; units mm / deg', (15,72), .43)
                    for panel, model in enumerate(('mamba','transformer')):
                        offset = panel * 512
                        view = image.copy()
                        data = predictions[model]
                        draw_geometry(view, geometry(data['prediction'][index],tcp[index]),intrinsic,extrinsic,(230,80,240),True)
                        draw_geometry(view, geometry(target,tcp[index]),intrinsic,extrinsic,(65,225,75),False)
                        canvas[100:612,offset:offset+512] = view
                        put_text(canvas, f'{model.upper()} + GNN  |  seed {config["seed"]}', (offset+10,93), .51, thickness=2)
                        put_text(canvas, f'Frame {frame:03d}  Peg {data["peg_position_error_mm"][index]:.1f} mm  Hole {data["hole_position_error_mm"][index]:.1f} mm', (offset+10,638), .43)
                        put_text(canvas, f'Peg rot {data["peg_rotation_error_deg"][index]:.1f} deg  Hole rot {data["hole_rotation_error_deg"][index]:.1f} deg', (offset+10,660), .43)
                        put_text(canvas, f'End-Hole relative {data["relative_3d_error_mm"][index]:.1f} mm', (offset+10,682), .43)
                    cv2.line(canvas,(512,82),(512,690),(95,95,95),1)
                    process.stdin.write(canvas.tobytes())
            finally:
                process.stdin.close()
                code = process.wait()
            if code:
                raise RuntimeError(f'ffmpeg exited {code}')
            summary = dict(episode=item['episode'], frames=len(indices), fps=config['fps'],
                           first_sample_id=str(ids[indices[0]]), last_sample_id=str(ids[indices[-1]]),
                           video=path.name, video_sha256=file_sha256(path))
            summaries.append(summary)
            logger.link_output(path, kind='rollouts')
            print(f'Rendered {path}: {len(indices)} frames',flush=True)
        result = dict(config=config, episodes=summaries, selection='development validation examples',
                      comparison_reference_sha256=comparison['reference_sha256'],
                      note='Saved seed-specific evaluation predictions, no new inference')
        write(output/'manifest.json',result)
        logger.link_output(output/'manifest.json')
        logger.record(episodes=summaries, source_comparison_sha256=config['comparison_sha256'])
        return str(output)
