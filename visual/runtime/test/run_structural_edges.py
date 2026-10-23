"""Task 02-C offline CPU sweep. Run from repository root with WANDB_MODE=disabled.

GT is read by this evaluator only, after observation-only extraction. No changes
to the production extractor, representation, tracker, sampling or graph modules.
"""
import argparse
from dataclasses import fields
import ctypes.util
import glob
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from utils.experiment_artifacts import digests
from utils.experiment_logger import ExperimentLogger
from visual.runtime.rgbd import PEG, BOX, iter_saved_rgbd
from visual.runtime.test.normal_diagnostics import erode, dilate, endpoints, pose_rotation, rays, stats
from visual.runtime.test.run_normal_comparison import real_truth, bounds
from visual.runtime.test.run_normal_diagnostics import save_json
from visual.runtime.test.structural_edges import METHODS, THRESHOLDS, compare

OBJECTS = {'peg': PEG, 'box': BOX}
LABELS = {'baseline': 'A: baseline', 'fd2': 'B: FD 2px', 'pca5': 'C: PCA 5x5'}
TOL_PX, TOL_M = 2., .003


def pcl_status():
    return dict(status='NOT RUN', python_modules={m: importlib.util.find_spec(m) is not None for m in ('pcl', 'pclpy')},
                libraries={m: ctypes.util.find_library(m) for m in ('pcl_common', 'pcl_features')},
                headers=[p for root in ('/opt/homebrew/include', '/usr/local/include', '/usr/include')
                         for p in glob.glob(root+'/pcl*/pcl/features/organized_edge_detection.h')],
                pkg_config=shutil.which('pkg-config'), installed_or_built=False, rgb_canny=False,
                reason='No usable PCL dependency on this host; no installation or binding attempted')


def raster_reference(mask, truth, intrinsic):
    vu = np.column_stack(np.nonzero(mask))
    return vu[:, ::-1].astype(float), (rays(mask.shape, intrinsic)*truth['depth'][..., None])[mask]


def rim_reference(frame, gt, truth, legacy_region):
    """Physical line samples, visibility restricted by the unchanged Task02-A ROI.

    At most one reference per rounded projected pixel. Samples must be within
    1.5px of a GT-visible legacy rim surface AND within its 4mm physical gate.
    This is evaluation discretization, not sampling predicted structural points.
    """
    if not legacy_region.any():
        return np.empty((0, 2)), np.empty((0, 3))
    s = frame.source_frame
    pose = gt['box_pose'][s]
    rot = pose_rotation(pose)
    centre = ((gt['box_hole_pose'][s, :3]-pose[:3]) @ rot)[1:]
    radius, half = float(gt['box_hole_radii']), float(gt['peg_half_sizes'][0])
    segments = []
    for axis, other, c, oc in ((1, 2, centre[0], centre[1]), (2, 1, centre[1], centre[0])):
        for sign in (-1, 1):
            p = np.zeros((2001, 3)); p[:, 0] = np.sign(truth['origin'][0])*half
            p[:, axis] = c+sign*radius
            p[:, other] = np.linspace(oc-radius, oc+radius, len(p))
            segments.append(p)
    world_to_cv = np.linalg.inv(frame.cam2world)
    xyz = (np.concatenate(segments) @ rot.T+pose[:3]) @ world_to_cv[:3, :3].T+world_to_cv[:3, 3]
    xyz = xyz[xyz[:, 2] > 0]
    uv = xyz[:, :2]/xyz[:, 2:]*np.diag(frame.intrinsic)[:2]+frame.intrinsic[:2, 2]-.5
    visible_uv, visible_xyz = raster_reference(legacy_region, truth, frame.intrinsic)
    distance, nearest = cKDTree(visible_uv).query(uv)
    keep = (distance <= 1.5) & (np.linalg.norm(xyz-visible_xyz[nearest], axis=1) <= .004)
    uv, xyz = uv[keep], xyz[keep]
    if not len(uv):
        return uv, xyz
    rounded = np.rint(uv).astype(int)
    # Choose the subpixel line point nearest its raster centre, deterministically.
    order = np.argsort(np.linalg.norm(uv-rounded, axis=1), kind='stable')
    _, unique = np.unique(rounded[order], axis=0, return_index=True)
    indices = order[unique]
    return uv[indices], xyz[indices]


class Matcher:
    """Whole observed support, no GT corridor restriction on predictions.

    Tolerance matches require the SAME reference/prediction pair to satisfy both
    <=2px AND <=3mm. One-to-one maximum matching penalizes thick duplicate bands.
    Nearest 2D and nearest 3D errors are separately reported in both directions.
    """
    def __init__(self, geometry, object_id, reference):
        self.support = np.flatnonzero(geometry.observed_mask(object_id))
        vu = np.column_stack(np.unravel_index(self.support, geometry.valid_depth.shape))
        self.uv, self.xyz = vu[:, ::-1], geometry.xyz.reshape(-1, 3)[self.support]
        self.ref_uv, self.ref_xyz = reference
        self.nref = len(self.ref_uv)
        self.near2 = np.full(len(vu), np.nan)
        self.near3 = self.near2.copy()
        if len(vu) and self.nref:
            self.near2 = cKDTree(self.ref_uv).query(self.uv)[0]
            self.near3 = cKDTree(self.ref_xyz).query(self.xyz)[0]*1000
            matrix = cKDTree(self.uv).sparse_distance_matrix(cKDTree(self.ref_uv), TOL_PX, output_type='coo_matrix')
            ok = np.linalg.norm(self.xyz[matrix.row]-self.ref_xyz[matrix.col], axis=1) <= TOL_M
            self.adjacency = csr_matrix((np.ones(ok.sum(), bool), (matrix.row[ok], matrix.col[ok])), shape=(len(vu), self.nref))
        else:
            self.adjacency = csr_matrix((len(vu), self.nref), dtype=bool)

    def score(self, prediction, distances=False):
        choose = prediction.ravel()[self.support]
        adjacency = self.adjacency[choose]
        count = int(choose.sum())
        tp = int(np.count_nonzero(adjacency.getnnz(axis=1)))
        hit = int(np.count_nonzero(adjacency.getnnz(axis=0)))
        one = int(np.count_nonzero(maximum_bipartite_matching(adjacency, perm_type='column') >= 0)) if count and self.nref else 0
        row = dict(candidates=count, references=self.nref, matched_candidates=tp, covered_references=hit, one_to_one=one,
                   candidates_without_visible_reference=count if not self.nref else 0)
        arrays = {}
        if distances:
            arrays = dict(candidate_to_gt_px=self.near2[choose], candidate_to_gt_mm=self.near3[choose])
            arrays['gt_to_candidate_px'] = cKDTree(self.uv[choose]).query(self.ref_uv)[0] if count and self.nref else np.empty(0)
            arrays['gt_to_candidate_mm'] = cKDTree(self.xyz[choose]).query(self.ref_xyz)[0]*1000 if count and self.nref else np.empty(0)
            row['location'] = {k: stats(v) for k, v in arrays.items()}
        return row, arrays


def ratios(row):
    n, r = row['candidates'], row['references']
    row.update(precision=row['matched_candidates']/n if n else None,
               recall=row['covered_references']/r if r else None,
               one_to_one_precision=row['one_to_one']/n if n else None,
               one_to_one_recall=row['one_to_one']/r if r else None,
               one_to_one_f1=2*row['one_to_one']/(n+r) if n+r else None)
    return row


class Pool:
    def __init__(self):
        self.items = {}

    def add(self, key, row, arrays):
        item = self.items.setdefault(key, dict(counts={k: 0 for k in row if k != 'location'}, arrays={}))
        for k, v in row.items():
            if k != 'location': item['counts'][k] += v
        for k, v in arrays.items(): item['arrays'].setdefault(k, []).append(v.astype(np.float32))

    def summary(self):
        return {k: dict(**ratios(item['counts']), location={a: stats(np.concatenate(v)) for a, v in item['arrays'].items()})
                for k, item in self.items.items()}


def negative_regions(regions, maps, reference, shape):
    # Exclude known physical structure neighbourhood before calling a boundary FP.
    raster = np.zeros(shape, bool)
    uv = np.rint(reference[0]).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
    uv = uv[inside]; raster[uv[:, 1], uv[:, 0]] = True
    away = ~dilate(raster, 3)
    result = {k: regions[k] & away for k in ('silhouette_2px', 'occlusion_2px', 'same_plane_tracker_cut_2px', 'invalid_depth_2px')}
    if 'robot_occlusion_2px' in regions: result['robot_occlusion_2px'] = regions['robot_occlusion_2px'] & away
    result['contamination'] = regions['contamination'] & away
    if 'plane_interior' in regions: result['plane_interior'] = regions['plane_interior'] & away
    return result


def comparison_plot(output, frame, evidence, regions, maps, references):
    union_mask = frame.masks[PEG] | frame.masks[BOX]
    views = [('Full image', None, (0, frame.depth.shape[1]-1, frame.depth.shape[0]-1, 0)),
             ('Peg', PEG, bounds(regions['peg']['gt_visible'], union_mask)),
             ('Box / physical hole entrance', BOX, bounds(regions['box']['hole_rim'], regions['box']['gt_visible'], 12))]
    colors = dict(orientation='#ff3939', depth='#ffba18', occluding='#d737ef', occluded='#29c5ff', invalid='#ffffff', mask='#67e535')
    fig, axes = plt.subplots(3, 4, figsize=(19, 13), constrained_layout=True)
    for row, (view, oid, extent) in enumerate(views):
        for col in range(4):
            ax = axes[row, col]; ax.imshow(frame.rgb)
            if col == 0:
                for name in OBJECTS:
                    uv = references[name]['all'][0]
                    ax.scatter(uv[:, 0], uv[:, 1], s=3, c='#ffff00', linewidths=0)
                    for cause, color in (('occlusion', '#d737ef'), ('cut', '#67e535')):
                        v, u = np.nonzero(maps[name][cause]); ax.scatter(u, v, s=3, c=color, linewidths=0)
                ax.set_title(view+' | GT only (yellow structure)')
            else:
                method = METHODS[col-1]
                for i in OBJECTS.values() if oid is None else (oid,):
                    candidates = evidence[method][i].candidates(20)
                    for cause, color in colors.items():
                        v, u = np.nonzero(candidates[cause])
                        if cause in ('depth', 'mask'):
                            ax.scatter(u, v, s=14 if cause == 'depth' else 8, marker='s' if cause == 'depth' else 'o',
                                       facecolors='none', edgecolors=color, linewidths=.45)
                        else:
                            ax.scatter(u, v, s=3, c=color, linewidths=0)
                ax.set_title(LABELS[method]+' | angle >20 deg')
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect('equal')
            ax.set_xlabel('source u (px)'); ax.set_ylabel('source v (px)')
    fig.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], loc='outside lower center', ncol=6)
    fig.suptitle(f'Source {frame.source_frame}: same crop / CV source pixels. Colours are diagnostic; causes overlap.\nNear/far depth order is potential occlusion only; mask/occlusion is not a structural label.')
    fig.savefig(output/f'f{frame.source_frame:04d}_comparison.png', dpi=135); plt.close(fig)


def sweep_plot(output, summary):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for method in METHODS:
        for channel, style in (('union', '-'), ('orientation', '--')):
            rows = [summary[f'{method}/{channel}/{t}/all'] for t in THRESHOLDS]
            axes[0, 0].plot([r['recall'] for r in rows], [r['precision'] for r in rows], style, marker='.', label=LABELS[method]+' '+channel)
            axes[0, 1].plot([r['one_to_one_recall'] for r in rows], [r['one_to_one_precision'] for r in rows], style, marker='.')
        rows = [summary[f'{method}/union/{t}/all'] for t in THRESHOLDS]
        axes[0, 2].plot(THRESHOLDS[:-1], [r['candidates'] for r in rows[:-1]], marker='.', label=LABELS[method])
        for ax, key in zip(axes[1], ('peg_seam', 'box_rim', 'peg_plane')):
            if key == 'peg_plane':
                y = [summary[f'{method}/union/{t}/negative/peg/plane_interior']['rate'] for t in THRESHOLDS]
            else:
                y = [summary[f'{method}/union/{t}/{key}']['recall'] for t in THRESHOLDS]
            ax.plot(THRESHOLDS[:-1], y[:-1], marker='.', label=LABELS[method]); ax.set_title(key); ax.set_xlabel('angle threshold (deg)')
    axes[0, 0].set(xlabel='coverage / recall', ylabel='candidate precision', title='Whole support: <=2px AND <=3mm')
    axes[0, 1].set(xlabel='one-to-one recall', ylabel='one-to-one precision', title='Maximum one-to-one matching')
    axes[0, 2].set(xlabel='angle threshold (deg)', ylabel='candidates / 121 frames', title='Density (depth-only at 180 in JSON)')
    for ax in axes.flat: ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8); axes[0, 2].legend(fontsize=8)
    fig.savefig(output/'tradeoffs.png', dpi=140); plt.close(fig)


def density_plot(output, density, shape):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    vmax = max(max(v['grid_8x8'][i][j] for i in range(8) for j in range(8))
               for k, v in density.items() if '/union/20/' in k)
    for row, name in enumerate(OBJECTS):
        for col, method in enumerate(METHODS):
            d = density[f'{method}/union/20/{name}']; ax = axes[row, col]
            im = ax.imshow(d['grid_8x8'], extent=(0, shape[1], shape[0], 0), vmin=0, vmax=vmax, cmap='magma')
            fraction = f'{d["fraction"]:.1%}' if d['fraction'] is not None else 'no observation'
            ax.set(title=f'{LABELS[method]} {name}: {fraction}', xlabel='source u (px)', ylabel='source v (px)')
    fig.colorbar(im, ax=axes, label='Candidate pixel-frames per image tile (8x8 grid); same scale')
    fig.savefig(output/'density.png', dpi=140); plt.close(fig)


def evaluate(episode_path, masks_path, output_root, camera='base_camera', selected=(0, 15, 45, 80, 120)):
    episode_path, masks_path = Path(episode_path).resolve(), Path(masks_path).resolve()
    with np.load(episode_path, allow_pickle=False) as archive:
        required = [f'{camera}/seg', 'peg_id', 'box_id', 'peg_pose', 'box_pose', 'box_hole_pose', 'peg_half_sizes', 'box_hole_radii', 'seed']
        if set(required)-set(archive.files): raise ValueError('GT fields missing')
        gt = {k: archive[k] for k in required}
    with np.load(masks_path, allow_pickle=False) as archive: sources = archive['source_frame']
    if not np.array_equal(sources, np.arange(len(gt[f'{camera}/seg']))):
        raise ValueError('Task02-C requires every episode frame exactly once in source order')
    tracker_meta = masks_path.with_name('tracking.json')
    if tracker_meta.exists():
        track = json.loads(tracker_meta.read_text())
        if track.get('camera', camera) != camera: raise ValueError('tracker camera mismatch')
        track_input = Path(track.get('input') or episode_path)
        if track_input.exists() and digests(track_input)[1] != digests(episode_path)[1]: raise ValueError('tracker episode mismatch')
    segmap_path = episode_path.with_name('segmentation_id_map.json')
    segmap = json.loads(segmap_path.read_text()) if segmap_path.exists() else {}
    robot_ids = [int(k) for k, v in segmap.items() if v.startswith('panda_')]
    config = dict(task='02-C', input=str(episode_path), masks=str(masks_path), camera=camera,
                  input_sha256=digests(episode_path)[1], masks_sha256=digests(masks_path)[1],
                  methods=METHODS, angle_thresholds_deg=THRESHOLDS, depth_gate_abs_m=.01, depth_gate_rel=.02,
                  tolerance_px=TOL_PX, tolerance_m=TOL_M, pcl=pcl_status(), seed=int(gt['seed']),
                  device='cpu', python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                  scipy=importlib.metadata.version('scipy'), matplotlib=matplotlib.__version__,
                  rgb_canny=False, gt_detector_input=False, full_resolution=True,
                  robot_segmentation_ids=robot_ids, segmentation_map_sha256=digests(segmap_path)[1] if segmap else None,
                  selected_sources=selected, timing='warmup excluded; rotated order; both objects; shared XYZ charged to each; no I/O, GT, sweep or plotting',
                  source_sha256={str(p.relative_to(REPO)): digests(p)[1] for p in [Path(__file__), Path(__file__).with_name('structural_edges.py'),
                      REPO/'visual/runtime/geometry.py', Path(__file__).with_name('normal_estimators.py'),
                      Path(__file__).with_name('normal_diagnostics.py'), Path(__file__).with_name('run_normal_comparison.py')]})
    with ExperimentLogger(team='visual', config=config, seed=int(gt['seed']), model='task02c_structural_edges', job_type='evaluate') as logger:
        output = Path(output_root).resolve()/logger.run_id; output.mkdir(parents=True, exist_ok=False)
        print('Output:', output, flush=True)
        for source in config['source_sha256']:
            destination = output/'evaluated_source'/source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO/source, destination)
        pool, negative, records, density, legacy_rim = Pool(), {}, [], {}, {}
        iterator = iter_saved_rgbd(episode_path, masks_path, camera=camera)
        for index, frame in enumerate(iterator):
            if index == 0: compare(frame)  # one warmup per method
            results, evidence = compare(frame, index)
            truths, regions, maps = real_truth(frame, results['baseline'], gt, camera)
            for name in OBJECTS:
                robot_occluder = (np.isin(gt[f'{camera}/seg'][frame.source_frame], robot_ids)
                                  & np.isfinite(truths[name]['depth'])
                                  & results['baseline'].geometry.valid_depth
                                  & (frame.depth < truths[name]['depth']-.0015))
                regions[name]['robot_occlusion_2px'] = (dilate(endpoints(regions[name]['gt_visible'], robot_occluder), 2)
                                                       & regions[name]['gt_visible'])
            references, matchers, negatives = {}, {}, {}
            record = dict(source_frame=frame.source_frame, methods={}, gt={}, observations={})
            for name, oid in OBJECTS.items():
                references[name] = {'seam': raster_reference(maps[name]['seam'], truths[name], frame.intrinsic)}
                if name == 'box':
                    references[name]['rim'] = rim_reference(frame, gt, truths[name], regions[name]['hole_rim'])
                    # Additional box plane control; fixed face-wise erosion, no new GT detector input.
                    regions[name]['plane_interior'] = np.logical_or.reduce([erode(regions[name]['gt_visible'] & (truths[name]['face'] == f), 3) for f in range(6)])
                    regions[name]['plane_interior'] &= ~dilate(regions[name]['hole_rim'], 4)
                references[name]['all'] = tuple(np.concatenate([r[k] for r in references[name].values()]) for k in (0, 1))
                _, unique = np.unique(np.rint(references[name]['all'][0]).astype(int), axis=0, return_index=True)
                references[name]['all'] = tuple(x[unique] for x in references[name]['all'])
                matchers[name] = {key: Matcher(results['baseline'].geometry, oid, ref) for key, ref in references[name].items()}
                negatives[name] = negative_regions(regions[name], maps[name], references[name]['all'], frame.depth.shape)
                record['gt'][name] = {k: len(v[0]) for k, v in references[name].items()}
                if name == 'box': record['gt'][name]['legacy_rim_band'] = int(regions[name]['hole_rim'].sum())
                f = results['baseline'].objects[oid]
                record['observations'][name] = dict(mask=int(f.object_mask.sum()), points=int(f.valid_points.sum()),
                    gt_visible=int(regions[name]['gt_visible'].sum()), contamination=int((f.valid_points & regions[name]['contamination']).sum()),
                    legacy_rim_observed=int((f.valid_points & regions[name].get('hole_rim', np.zeros_like(f.valid_points))).sum()))
            for method in METHODS:
                record['methods'][method] = dict(timing_ms=results[method].timings_ms, thresholds={})
                for threshold in THRESHOLDS:
                    detail = {}
                    for name, oid in OBJECTS.items():
                        candidates = evidence[method][oid].candidates(threshold)
                        f = results[method].objects[oid]
                        # Full-raster equivalence to the requested original union.
                        expected = f.valid_points & (f.depth_discontinuity | (np.rad2deg(f.normal_variation_rad) > threshold))
                        if not np.array_equal(candidates['union'], expected): raise AssertionError('pair union differs from original feature maps')
                        for channel in ('union', 'orientation'):
                            prediction = candidates[channel]
                            key = f'{method}/{channel}/{threshold}'
                            for target, matcher in matchers[name].items():
                                row, arrays = matcher.score(prediction, distances=threshold in (10, 20, 30))
                                label = name+'_'+target if target != 'all' else 'all'
                                pool.add(key+'/'+label, row, arrays)
                                detail[name+'/'+target+'/'+channel] = row
                            if name == 'box':
                                lr = legacy_rim.setdefault(key, dict(gt_band=0, observed_band=0, detected_band=0))
                                rim = regions[name]['hole_rim']
                                lr['gt_band'] += int(rim.sum()); lr['observed_band'] += int((rim & f.valid_points).sum())
                                lr['detected_band'] += int((rim & prediction).sum())
                            for region, mask in negatives[name].items():
                                supported = mask & f.valid_points
                                nr = negative.setdefault(key+'/negative/'+name+'/'+region, dict(points=0, candidates=0))
                                nr['points'] += int(supported.sum()); nr['candidates'] += int((supported & prediction).sum())
                                detail[name+'/negative/'+region+'/'+channel] = dict(points=int(supported.sum()), candidates=int((supported & prediction).sum()))
                            v, u = np.nonzero(prediction)
                            grid = np.histogram2d(v, u, bins=(8, 8), range=((0, frame.depth.shape[0]), (0, frame.depth.shape[1])))[0].astype(int)
                            den = density.setdefault(key+'/'+name, dict(observed=0, candidates=0, grid_8x8=np.zeros((8, 8), int)))
                            den['observed'] += int(f.valid_points.sum()); den['candidates'] += len(v); den['grid_8x8'] += grid
                        if threshold == 20:
                            detail[name+'/causes'] = {k: int(v.sum()) for k, v in candidates.items()}
                    record['methods'][method]['thresholds'][str(threshold)] = detail
            if frame.source_frame in selected:
                comparison_plot(output, frame, evidence, regions, maps, references)
            # Every frame's pair evidence is preserved, not just selected plots.
            arrays = dict(source_frame=np.array(frame.source_frame), shape=np.array(frame.depth.shape), intrinsic=frame.intrinsic)
            for method in METHODS:
                for name, oid in OBJECTS.items():
                    pair = evidence[method][oid]
                    for field in fields(pair): arrays[f'{method}/{name}/{field.name}'] = getattr(pair, field.name)
                    if frame.source_frame in selected:
                        arrays.update({f'{method}/{name}/candidate20/{k}': v for k, v in pair.candidates(20).items()})
            if frame.source_frame in selected:
                for name in OBJECTS:
                    arrays.update({f'evaluation/{name}/{k}': v for k, v in regions[name].items()})
                    for target, ref in references[name].items():
                        arrays[f'evaluation/{name}/{target}_uv'] = ref[0]; arrays[f'evaluation/{name}/{target}_xyz'] = ref[1]
                arrays['depth_m'] = frame.depth
            np.savez_compressed(output/f'f{frame.source_frame:04d}_pairs.npz', **arrays)
            records.append(record)
            if index % 10 == 0: print(f'Frames {index+1}/{len(sources)}', flush=True)
        summary = pool.summary()
        summary.update({k: dict(**v, rate=v['candidates']/v['points'] if v['points'] else None) for k, v in negative.items()})
        density = {k: dict(observed=v['observed'], candidates=v['candidates'], grid_8x8=v['grid_8x8'].tolist(),
                           fraction=v['candidates']/v['observed'] if v['observed'] else None) for k, v in density.items()}
        legacy_rim = {k: dict(**v, coverage=v['detected_band']/v['gt_band'] if v['gt_band'] else None) for k, v in legacy_rim.items()}
        latency = {m: {k: stats([r['methods'][m]['timing_ms'][k] for r in records]) for k in records[0]['methods'][m]['timing_ms']} for m in METHODS}
        data = dict(run_id=logger.run_id, config=config, evaluated_frames=len(records),
                    git_commit=logger.metadata['git_commit'], git_dirty=logger.metadata['git_dirty'],
                    summary=summary, density=density, legacy_rim=legacy_rim, latency_ms=latency, frames=records)
        save_json(output/'comparison.json', data)
        sweep_plot(output, summary)
        density_plot(output, density, frame.depth.shape)
        body = ''.join(f'<p><a href="{p.name}"><img src="{p.name}"></a></p>' for p in [output/'tradeoffs.png', output/'density.png']+[output/f'f{s:04d}_comparison.png' for s in selected])
        (output/'comparison.html').write_text(f'''<!doctype html><html><meta charset="utf-8"><title>Task02-C</title>
<style>body{{font:15px system-ui;max-width:1800px;margin:24px auto}}img{{width:100%}}</style>
<h1>Task 02-C — structural edge candidate comparison</h1><p>Run {logger.run_id}; all {len(records)} frames; CPU only.
<a href="comparison.json">Full sweep / counts / errors / latency / density / frame metrics</a>.</p>
<p>PCL: NOT RUN (absent). RGB Canny: disabled. GT never used by detectors. Union is a proposal, not a structural class.
Whole-support precision uses <=2px AND <=3mm to visible physical references, without GT corridor filtering.
One-to-one matching additionally penalizes thick bands. Errors to GT are conditional on references being visible;
empty predictions still give zero recall. Curves sweep angle while preserving the original depth gate and normal support.
Rim line samples use the unchanged visible physical rim ROI for visibility; legacy band coverage remains in JSON.
No independent episode or final production threshold selection. Colours are debugging only; causes overlap.</p>{body}</html>''', encoding='utf-8')
        for p in sorted(p for p in output.rglob('*') if p.is_file()):
            logger.link_output(p, kind='metrics' if p.suffix == '.json' else 'trajectory' if p.suffix == '.npz' else 'reports')
        logger.log({m: summary[f'{m}/union/20/all'] for m in METHODS}, prefix='eval/task02c')
        logger.record(evaluated_frames=len(records), evaluation_device='cpu', gpu_executed=False, pcl_executed=False,
                      dataset_sha256=config['input_sha256'], masks_sha256=config['masks_sha256'])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=REPO/'visual/data/step2/ep000.npz')
    parser.add_argument('--masks', type=Path, default=REPO/'visual/results/runtime_tracking/ep000/masks.npz')
    parser.add_argument('--camera', default='base_camera')
    parser.add_argument('--out', type=Path, default=REPO/'visual/results/structural_edges')
    args = parser.parse_args()
    print(evaluate(args.input, args.masks, args.out, args.camera))
