"""Render saved Task09-C graphs into paired graph/RGB and graph-only MP4s.

No simulator, graph construction or model inference is run. Source graph arrays
and their camera observations stay unchanged; only viewer_media is created.
"""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.experiment_artifacts import register_existing_output
from visualization.captures import capture_episode, within
from visualization.persistent import persistent_arrays, persistent_episode


COLORS = ['#f5f5f5', '#facc68', '#f194ff', '#f16e72', '#61bbff']
EDGE_COLORS = ['#465369', '#b28b4d', '#9275b0']
BG = '#101622'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    for path in ('/System/Library/Fonts/Supplemental/Arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def view_matrix():
    yaw, pitch = -.68, .55
    return np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [-np.sin(yaw)*np.sin(pitch), -np.cos(yaw)*np.sin(pitch), np.cos(pitch)],
                     [np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(pitch), np.sin(pitch)]])


def projection(points, rotation, center, scale):
    p = np.asarray(points) @ rotation.T
    return np.column_stack((480+(p[:, 0]-center[0])*scale,
                            380-(p[:, 1]-center[1])*scale, p[:, 2]))


def graph_image(data, variant, timestep, seed, bounds, fonts):
    rotation, center, scale, all_center, all_scale = bounds
    image = Image.new('RGB', (960, 720), BG)
    draw = ImageDraw.Draw(image)
    world = data['world_position']
    points = projection(world, rotation, center, scale)
    visible = (points[:, 0]>=18)&(points[:, 0]<=942)&(points[:, 1]>=90)&(points[:, 1]<=640)
    edges, edge_types = data['edge_index'].T, data['edge_type']
    # Draw each saved bidirectional edge once. No edge policy is recomputed.
    seen = set()
    for (a, b), kind in zip(edges, edge_types):
        key = (int(kind), min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        if not (visible[a] and visible[b]):
            continue
        draw.line([tuple(points[a, :2]), tuple(points[b, :2])], fill=EDGE_COLORS[kind], width=1)
    nodes, types = data['node_input'], data['node_type']
    for i in np.argsort(points[:, 2])[::-1]:
        if not visible[i]:
            continue
        x, y = points[i, :2]
        radius = 6 if i < 3 else 3
        if i >= 3:
            halo = '#55d9e6' if nodes[i, 14] > .5 else '#f2a55a' if nodes[i, 15] > .5 else '#d191ff'
            draw.ellipse((x-radius-1, y-radius-1, x+radius+1, y+radius+1), fill=halo)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=COLORS[types[i]])
        if i < 3:
            draw.text((x+9, y-12), ['TCP', 'L finger', 'R finger'][i], font=fonts[1], fill=COLORS[i])
    origin = world[:3].mean(0)
    axes = projection(np.vstack((origin, origin+np.eye(3)*.04)), rotation, center, scale)
    for i, (label, color) in enumerate(zip('XYZ', ['#ff7777', '#7bdb99', '#7dabff'])):
        draw.line([tuple(axes[0, :2]), tuple(axes[i+1, :2])], fill=color, width=2)
        draw.text(tuple(axes[i+1, :2]), label, font=fonts[1], fill=color)
    # Headers have opaque backgrounds so every source node is visually distinct
    # from the title; all points remain within the common episode framing.
    draw.rectangle((0, 0, 960, 84), fill=BG)
    draw.text((24, 14), f'{variant} | saved graph | WORLD [m]' + (' | ORACLE (GT)' if variant.startswith('O') else ''), font=fonts[0], fill='white')
    draw.text((24, 48), f"seed {seed} | t={timestep:03d} | {float(data['sensor']['timestamp']):.2f} s | {data['record']['phase']} | N={len(nodes)} E={len(edges)}", font=fonts[1], fill='#b5c2d8')
    draw.rectangle((0, 648, 960, 720), fill=BG)
    for x, color, label in [(24, COLORS[3], 'Peg'), (110, COLORS[4], 'Box'),
                             (210, '#55d9e6', 'current'), (340, '#f2a55a', 'estimated'),
                             (500, '#d191ff', 'low confidence')]:
        draw.ellipse((x, 665, x+9, 674), fill=color)
        draw.text((x+15, 658), label, font=fonts[1], fill='#c8d4e8')
    draw.text((24, 691), f"{data['info']['revision']} | focus P1-P99 | outside focus: {int((~visible).sum())} nodes | overview: ALL nodes", font=fonts[2], fill='#8e9fb8')
    # Keep extreme source points visible without shrinking the useful scene.
    # This inset uses the full bounds and draws ALL saved nodes and edges.
    overview = world @ rotation.T
    xy = np.column_stack((822+(overview[:,0]-all_center[0])*all_scale,
                          181-(overview[:,1]-all_center[1])*all_scale))
    draw.rectangle((704, 94, 944, 270), fill='#172132', outline='#455673')
    for (a,b),kind in zip(edges,edge_types):
        draw.line([tuple(xy[a]),tuple(xy[b])],fill=EDGE_COLORS[kind],width=1)
    for i,(x,y) in enumerate(xy):
        draw.ellipse((x-1,y-1,x+1,y+1),fill=COLORS[types[i]])
    draw.text((714, 100), 'ALL nodes / full extent', font=fonts[2], fill='#c8d4e8')
    return image


def with_rgb(graph, data, roles, fonts):
    image = Image.new('RGB', (1344, 720), BG)
    image.paste(graph, (0, 0))
    draw = ImageDraw.Draw(image)
    for row, role in enumerate(('main', 'wrist')):
        camera = roles[role]
        rgb = Image.fromarray(data['sensor'][camera + '/rgb']).resize((320, 320), Image.Resampling.NEAREST)
        y = row*354
        draw.text((984, y+9), f'{role.upper()} RGB | same timestep', font=fonts[1], fill='#d6e2f5')
        image.paste(rgb, (984, y+35))
    return image


def render(root, geometry_run, fps=None, replace_videos=False):
    root = Path(root).resolve()
    run = within(root, root / geometry_run)
    manifest = json.loads((run / 'geometry_manifest.json').read_text())
    output = run / 'viewer_media'
    output.mkdir(exist_ok=replace_videos)
    fonts = [font(23), font(18), font(14)]
    rotation = view_matrix()
    index = dict(schema='task09c_viewer_media_v1', geometry_run_id=geometry_run,
                 geometry_manifest_sha256=sha256(run/'geometry_manifest.json'),
                 revision=manifest['config'].get('geometry_revision'), episodes={})
    sources = {}
    for seed in sorted({r['seed'] for r in manifest['frames']}):
        episode_id = f'{geometry_run}/geometry/seed{seed}'
        context = persistent_episode(root, episode_id, check_complete=True)
        folder, records, info = context
        _, _, capture_info = capture_episode(root, info['capture_id'])
        steps = [r['timestep'] for r in records]
        variants = [v['id'] for v in info['variants']]
        loaded = {}
        for variant in variants:
            loaded[variant] = [persistent_arrays(root, episode_id, t, variant, context=context) for t in steps]
            for t in steps:
                for extension in ('npz', 'json'):
                    path = folder / f'f{t:04d}' / f'{variant}_graph.{extension}'
                    sources[path.relative_to(root).as_posix()] = sha256(path)
        timestamps = np.array([float(d['sensor']['timestamp']) for d in loaded[variants[0]]])
        delta = np.diff(timestamps)
        frame_rate = fps or (1/float(np.median(delta)) if len(delta) else 20.)
        if len(delta) and not np.allclose(delta, 1/frame_rate, rtol=0, atol=1e-6):
            raise ValueError('MP4 requires uniform captured timestamps matching fps')
        projected = np.concatenate([d['world_position'] @ rotation.T for frames in loaded.values() for d in frames])
        all_lo, all_hi = projected[:, :2].min(0), projected[:, :2].max(0)
        lo, hi = np.percentile(projected[:, :2], [1,99], axis=0)
        robots = np.concatenate([d['world_position'][:3] @ rotation.T for d in loaded[variants[0]]])
        lo = np.minimum(lo, robots[:,:2].min(0))
        hi = np.maximum(hi, robots[:,:2].max(0))
        center = (lo+hi)/2
        # One robust focus and one full-extent inset shared by all variants.
        scale = min(860/max(hi[0]-lo[0], .08), 510/max(hi[1]-lo[1], .08))
        all_center = (all_lo+all_hi)/2
        all_scale = min(204/max(all_hi[0]-all_lo[0], .08), 124/max(all_hi[1]-all_lo[1], .08))
        bounds = rotation, center, scale, all_center, all_scale
        target = output / f'seed{seed}'
        target.mkdir(exist_ok=replace_videos)
        entry = dict(capture_id=info['capture_id'], geometry_id=episode_id,
                     frames=len(steps), fps=frame_rate, timesteps=steps,
                     timestamps=timestamps.tolist(), roles=capture_info['roles'], variants={},
                     focus_percentiles=[1,99], all_nodes_in_overview=True,
                     focus_projected_bounds=[lo.tolist(),hi.tolist()],
                     overview_projected_bounds=[all_lo.tolist(),all_hi.tolist()])
        for variant in variants:
            paths = {kind: target / f'{variant}_{kind}.mp4' for kind in ('graph_rgb', 'graph_only')}
            kwargs = dict(fps=frame_rate, codec='libx264', quality=7,
                          macro_block_size=16, ffmpeg_params=['-movflags', '+faststart'])
            with imageio.get_writer(paths['graph_rgb'], **kwargs) as combined, imageio.get_writer(paths['graph_only'], **kwargs) as graph_only:
                for t, data in zip(steps, loaded[variant]):
                    graph = graph_image(data, variant, t, seed, bounds, fonts)
                    graph_only.append_data(np.asarray(graph))
                    combined.append_data(np.asarray(with_rgb(graph, data, capture_info['roles'], fonts)))
            entry['variants'][variant] = {}
            for kind, path in paths.items():
                # Decode every frame after encoding, not just the MP4 header.
                reader = imageio.get_reader(path)
                count = sum(1 for _ in reader)
                reader.close()
                if count != len(steps):
                    raise ValueError(f'{path.name}: video frame count mismatch')
                entry['variants'][variant][kind] = dict(path=path.relative_to(run).as_posix(),
                    sha256=sha256(path), frames=count, fps=frame_rate)
                register_existing_output(run, path, kind='rollouts')
            print(f'seed={seed} {variant}: both MP4s encoded and decoded ({len(steps)} frames)', flush=True)
        index['episodes'][f'seed{seed}'] = entry
    # Record sensor inputs independently of GT; each graph frame is already
    # joined against this hash in persistent_arrays.
    for record in manifest['frames']:
        capture = manifest['config']['capture_run_id']
        path = f"{capture}/collection/seed{record['seed']}/f{record['timestep']:04d}_input.npz"
        sources[path] = record['capture_input_sha256']
    (output/'sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    index['sources_sha256'] = sha256(output/'sources.json')
    (output/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    for path in (output/'sources.json', output/'index.json'):
        register_existing_output(run, path, kind='reports')
    print(f'COMPLETE: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments-root', type=Path, default=Path('experiments'))
    parser.add_argument('--geometry-run', required=True)
    parser.add_argument('--fps', type=float, help='Must match captured timestamps; inferred by default')
    parser.add_argument('--replace-videos', action='store_true', help='Replace only generated viewer media')
    args = parser.parse_args()
    render(args.experiments_root, args.geometry_run, args.fps, args.replace_videos)


if __name__ == '__main__':
    main()
