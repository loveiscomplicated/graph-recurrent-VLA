"""Render the original Visual v0 extractor on saved Task09-C Main observations.

Only selected input/mask archives are read, never object GT or model weights.
The existing graph videos and graph schema are retained. FPS is a media variant.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from utils.experiment_artifacts import register_existing_output
from visualization.captures import _archive, capture_episode, within
from visualization.render_captures import BG, COLORS, font, projection, sha256, view_matrix, with_rgb


def legacy_module():
    source = Path(__file__).resolve().parents[1] / 'visual/scripts/pointcloud.py'
    spec = importlib.util.spec_from_file_location('visual_legacy_pointcloud', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


def extract_frame(sensor, masks, camera, rng, extractor):
    """Adapt signed-mm depth and world-to-CV extrinsics to the original API."""
    raw = sensor[camera + '/depth_raw_mm']
    mask = masks[camera]
    if raw.dtype != np.int16 or mask.dtype != np.bool_ or mask.shape != (2, *raw.shape):
        raise ValueError('invalid saved depth or prediction mask')
    depth = raw.astype(np.float32) * np.float32(.001)
    world_to_cv = np.eye(4, dtype=np.float32)
    world_to_cv[:3] = sensor[camera + '/extrinsic_cv']
    cv_to_world = np.linalg.inv(world_to_cv)
    clouds, stats = [], []
    for predicted in mask:
        _, world, info = extractor(predicted, depth, sensor[camera + '/intrinsic'], cv_to_world, rng)
        if not np.isfinite(world).all() or world.shape[1:] != (3,) or len(world) > 1024:
            raise ValueError('invalid legacy point cloud output')
        clouds.append(world)
        stats.append(dict(**info, points=len(world)))
    return clouds, stats


def cloud_image(clouds, sensor, record, seed, bounds, fonts):
    rotation, center, scale, all_center, all_scale = bounds
    world = np.concatenate(clouds)
    colors = [COLORS[3]] * len(clouds[0]) + [COLORS[4]] * len(clouds[1])
    projected = projection(world, rotation, center, scale)
    visible = ((projected[:, 0] >= 18) & (projected[:, 0] <= 942)
               & (projected[:, 1] >= 90) & (projected[:, 1] <= 640))
    image = Image.new('RGB', (960, 720), BG)
    draw = ImageDraw.Draw(image)
    if not len(world):
        draw.text((160, 335), 'No valid Main points in this frame', font=fonts[0], fill='#b5c2d8')
    for i in np.argsort(projected[:, 2])[::-1]:
        if visible[i]:
            x, y = projected[i, :2]
            draw.ellipse((x-2, y-2, x+2, y+2), fill=colors[i])
    draw.rectangle((0, 0, 960, 84), fill=BG)
    draw.text((24, 14), 'point cloud prototype(FPS) | WORLD [m]', font=fonts[0], fill='white')
    draw.text((24, 48), f"seed {seed} | t={record['timestep']:03d} | {float(sensor['timestamp']):.2f} s | {record['phase']} | Peg={len(clouds[0])} Box={len(clouds[1])}", font=fonts[1], fill='#b5c2d8')
    draw.rectangle((0, 648, 960, 720), fill=BG)
    draw.text((24, 656), 'Main prediction | erosion > SOR > FPS | <=1024 / object | no edges', font=fonts[1], fill='#c8d4e8')
    draw.text((24, 686), f'Visual v0 reapplied to Task09-C | outside focus: {int((~visible).sum())} points', font=fonts[2], fill='#8e9fb8')
    overview = world @ rotation.T
    xy = np.column_stack((822+(overview[:, 0]-all_center[0])*all_scale,
                          181-(overview[:, 1]-all_center[1])*all_scale))
    draw.rectangle((704, 94, 944, 270), fill='#172132', outline='#455673')
    for (x, y), color in zip(xy, colors):
        draw.ellipse((x-1, y-1, x+1, y+1), fill=color)
    draw.text((714, 100), 'ALL FPS points / full extent', font=fonts[2], fill='#c8d4e8')
    return image


def render(root, geometry_run, replace=False):
    root = Path(root).resolve()
    run = within(root, root / geometry_run)
    output = within(root, run / 'viewer_media')
    index_path = output / 'index.json'
    index = json.loads(index_path.read_text())
    if index['geometry_run_id'] != geometry_run:
        raise ValueError('media run identity mismatch')
    if not replace and any('FPS' in e['variants'] for e in index['episodes'].values()):
        raise ValueError('FPS videos already indexed; use --replace-videos')
    legacy, source = legacy_module()
    provenance = dict(schema='task09c_legacy_fps_media_v1', source=str(source.relative_to(source.parents[2])),
        source_sha256=sha256(source), renderer_sha256=sha256(__file__), camera_role='main',
        method='original Visual v0 extract; erosion2, random pre-cap2048, SOR16/2, FPS<=1024; cluster off',
        seed=0, rng_order='new RNG per episode; timesteps ascending; peg then box',
        depth='original signed int16 mm to float32 metres; <=0 excluded',
        transform='inverse saved world-to-CV extrinsic; original pixel centers +0.5',
        gt_input=False, robot_filter=False, accumulation=False, edges=False, episodes={})
    fonts = [font(23), font(18), font(14)]
    rotation = view_matrix()
    pending_outputs = []
    for seed, entry in index['episodes'].items():
        folder, episode, info = capture_episode(root, entry['capture_id'])
        if info['timesteps'] != entry['timesteps']:
            raise ValueError('capture and video timelines disagree')
        rng = np.random.default_rng(0)
        loaded, records, cloud_arrays = [], [], {}
        for record in episode['frames']:
            sensor = _archive(root, folder, record, 'input')
            masks = _archive(root, folder, record, 'masks')
            if sensor['episode'].item() != episode['episode'] or sensor['timestep'].item() != record['timestep']:
                raise ValueError('capture observation identity mismatch')
            clouds, stats = extract_frame(sensor, masks, info['roles']['main'], rng, legacy.extract)
            loaded.append((sensor, clouds, record))
            records.append(dict(timestep=record['timestep'], input_sha256=record['input_sha256'],
                                mask_sha256=record['mask_sha256'], objects=stats))
            for obj, points in zip(('peg', 'box'), clouds):
                cloud_arrays[f"{record['stem']}_{obj}"] = points
        timestamps = np.array([float(sensor['timestamp']) for sensor, _, _ in loaded])
        if not np.allclose(timestamps, entry['timestamps'], rtol=0, atol=1e-8) or not np.allclose(np.diff(timestamps), 1/entry['fps'], rtol=0, atol=1e-6):
            raise ValueError('capture timestamps must match graph videos')
        lo, hi = np.array(entry['focus_projected_bounds'])
        all_lo, all_hi = np.array(entry['overview_projected_bounds'])
        points = np.concatenate([p for _, clouds, _ in loaded for p in clouds]) @ rotation.T
        if len(points):
            all_lo = np.minimum(all_lo, points[:, :2].min(0))
            all_hi = np.maximum(all_hi, points[:, :2].max(0))
        bounds = (rotation, (lo+hi)/2, min(860/max(hi[0]-lo[0], .08), 510/max(hi[1]-lo[1], .08)),
                  (all_lo+all_hi)/2, min(204/max(all_hi[0]-all_lo[0], .08), 124/max(all_hi[1]-all_lo[1], .08)))
        paths = {kind: output / seed / f'FPS_{kind}.mp4' for kind in ('graph_rgb', 'graph_only')}
        if not replace and any(p.exists() for p in paths.values()):
            raise ValueError('FPS video exists; use --replace-videos')
        kwargs = dict(fps=entry['fps'], codec='libx264', quality=7, macro_block_size=16,
                      ffmpeg_params=['-movflags', '+faststart'])
        with imageio.get_writer(paths['graph_rgb'], **kwargs) as rgb, imageio.get_writer(paths['graph_only'], **kwargs) as only:
            for sensor, clouds, record in loaded:
                image = cloud_image(clouds, sensor, record, info['seed'], bounds, fonts)
                only.append_data(np.asarray(image))
                rgb.append_data(np.asarray(with_rgb(image, dict(sensor=sensor), info['roles'], fonts)))
        videos = {}
        for kind, path in paths.items():
            reader = imageio.get_reader(path)
            count = sum(1 for _ in reader)
            reader.close()
            if count != len(loaded):
                raise ValueError('FPS encoded frame count mismatch')
            videos[kind] = dict(path=path.relative_to(run).as_posix(), sha256=sha256(path), frames=count, fps=entry['fps'])
            pending_outputs.append((path, 'rollouts'))
        archive = output / seed / 'FPS_points.npz'
        np.savez_compressed(archive, **cloud_arrays)
        pending_outputs.append((archive, 'trajectory'))
        provenance['episodes'][seed] = dict(capture_id=info['id'], frames=records,
            points_path=archive.relative_to(run).as_posix(), points_sha256=sha256(archive),
            overview_projected_bounds=[all_lo.tolist(), all_hi.tolist()])
        entry['variants']['FPS'] = videos
        print(f'{seed}: FPS pair encoded and fully decoded ({len(loaded)} frames)', flush=True)
    provenance_path = output / 'fps_sources.json'
    provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    index['fps_sources_sha256'] = sha256(provenance_path)
    index_path.write_text(json.dumps(index, indent=2)+'\n')
    for path, kind in [*pending_outputs, (provenance_path, 'reports'), (index_path, 'reports')]:
        register_existing_output(run, path, kind=kind)
    print('COMPLETE: FPS media variant added', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments-root', type=Path, default=Path('experiments'))
    parser.add_argument('--geometry-run', required=True)
    parser.add_argument('--replace-videos', action='store_true')
    args = parser.parse_args()
    render(args.experiments_root, args.geometry_run, args.replace_videos)


if __name__ == '__main__':
    main()
