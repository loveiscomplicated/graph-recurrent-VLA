"""Task 02-B: fixed full-resolution normal comparison, CPU/offline diagnostics.

WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_normal_comparison
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import html
import importlib.metadata
from itertools import chain
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from utils.experiment_artifacts import digests
from utils.experiment_logger import ExperimentLogger
from visual.runtime.geometry import (
    GeometryFrame, GeometryParameters, build_organized_geometry, compute_geometric_features,
    estimate_surface_normals, process_geometry,
)
from visual.runtime.rgbd import BOX, PEG, iter_saved_rgbd
from visual.runtime.test.normal_diagnostics import (
    THRESHOLDS, angular_error, box_primitives, dilate, metrics, oracle, pose_rotation,
    quantize, rays, real_regions, rim_region, stats, synthetic_case, synthetic_frame,
)
from visual.runtime.test.normal_estimators import METHODS, finite_difference, local_plane
from visual.runtime.test.run_normal_diagnostics import save_json, summarize_rows

OBJECTS = {'peg': PEG, 'box': BOX}
PARAMETERS = GeometryParameters()


def compare(frame, rotation=0):
    """Shared observed backprojection; separately timed normal + feature passes.

    Rotate execution order to reduce fixed ordering effects. Baseline uses the
    exact existing functions; the standard process_geometry is checked in tests.
    Timers exclude GT, loading, plotting, saving, metrics and the other methods.
    """
    start = perf_counter()
    geometry = build_organized_geometry(frame)
    geometry_ms = (perf_counter()-start)*1000
    results = {}
    order = METHODS[rotation % len(METHODS):]+METHODS[:rotation % len(METHODS)]
    for method in order:
        start = perf_counter()
        if method == 'baseline':
            normals = {i: estimate_surface_normals(geometry, i, PARAMETERS) for i in geometry.object_masks}
        elif method.startswith('fd'):
            normals = {i: finite_difference(geometry, i, int(method[2:]), PARAMETERS) for i in geometry.object_masks}
        else:
            normals = {i: local_plane(geometry, i, int(method[3:]), PARAMETERS) for i in geometry.object_masks}
        after_normals = perf_counter()
        features = {i: compute_geometric_features(geometry, i, *normals[i], PARAMETERS) for i in geometry.object_masks}
        end = perf_counter()
        timings = dict(geometry=geometry_ms, normals=(after_normals-start)*1000,
                       discontinuity_features=(end-after_normals)*1000,
                       total=geometry_ms+(end-start)*1000)
        results[method] = GeometryFrame(frame.source_frame, frame.camera, frame.frame_idx,
                                       frame.timestamp, geometry, features, timings)
    return results


def response_metrics(feature, region, truth=None):
    row = metrics(feature, region, truth)
    point = region & feature.valid_points
    jump = point & feature.depth_discontinuity
    row['jump_without_normal'] = int((jump & ~feature.normal_valid).sum())
    for t in THRESHOLDS:
        high = point & feature.normal_variation_valid & (np.rad2deg(feature.normal_variation_rad) > t)
        row[f'union_{t}'] = int((jump | high).sum())
        row[f'variation_only_{t}'] = int((high & ~jump).sum())
        row[f'both_{t}'] = int((high & jump).sum())
        row[f'jump_only_{t}'] = int((jump & ~high).sum())
    return row


class Pool:
    def __init__(self):
        self.items = {}

    def add(self, key, feature, region, truth=None):
        item = self.items.setdefault(key, dict(rows=[], variation=[], errors=[]))
        row = response_metrics(feature, region, truth)
        item['rows'].append(row)
        vv = region & feature.valid_points & feature.normal_variation_valid
        item['variation'].append(np.rad2deg(feature.normal_variation_rad[vv]))
        if truth is not None:
            nv = region & feature.valid_points & feature.normal_valid
            item['errors'].append(angular_error(feature.normals[nv], truth[nv]))
        return row

    def summary(self):
        result = {}
        for key, item in self.items.items():
            rows = item['rows']
            r = summarize_rows(rows, item['variation'], item['errors'] or None)
            extras = ['jump_without_normal']+[f'{k}_{t}' for t in THRESHOLDS
                                                for k in ('union', 'variation_only', 'both', 'jump_only')]
            for k in extras:
                r[k] = sum(v[k] for v in rows)
                r[k+'_fraction_of_region'] = r[k]/r['region_pixels'] if r['region_pixels'] else None
                r[k+'_fraction_of_points'] = r[k]/r['points'] if r['points'] else None
            result[key] = r
        return result


def localization(feature, reference, visible, threshold):
    """Raster seam endpoint matching, fixed 6px search corridor, no GT input.

    Report both directions: nearest-response distance alone can reward planar
    noise. Precision is deliberately not claimed in this restricted corridor.
    Missing predictions give zero coverage and no finite distance statistic.
    """
    n = int(reference.sum())
    if not n:
        return dict(reference=0, predictions=0, hit_1=0, hit_2=0, hit_3=0), np.array([]), np.array([])
    to_ref = distance_transform_edt(~reference)
    prediction = (to_ref <= 6) & visible & feature.valid_points & feature.normal_variation_valid
    prediction &= np.rad2deg(feature.normal_variation_rad) > threshold
    p = int(prediction.sum())
    a = to_ref[prediction]
    b = distance_transform_edt(~prediction)[reference] if p else np.array([])
    row = dict(reference=n, predictions=p,
               **{f'hit_{t}': int((b <= t).sum()) for t in (1, 2, 3)})
    return row, a, b


def localization_summary(items):
    result = {}
    for key, values in items.items():
        total = {k: sum(v[0][k] for v in values) for k in values[0][0]}
        total['response_to_seam_px'] = stats(np.concatenate([v[1] for v in values]))
        total['seam_to_response_px_when_any_prediction'] = stats(np.concatenate([v[2] for v in values]))
        for t in (1, 2, 3):
            total[f'coverage_{t}px'] = total[f'hit_{t}']/total['reference'] if total['reference'] else None
        result[key] = total
    return result


def real_truth(frame, baseline, gt, camera):
    s, half = frame.source_frame, gt['peg_half_sizes']
    centre = ((gt['box_hole_pose'][s, :3]-gt['box_pose'][s, :3]) @ pose_rotation(gt['box_pose'][s]))[1:]
    truths = dict(peg=oracle(frame, gt['peg_pose'][s], [(np.zeros(3), half)]),
                  box=oracle(frame, gt['box_pose'][s], box_primitives(float(half[0]), float(gt['box_hole_radii']), centre)))
    regions, maps = {}, {}
    for name, object_id in OBJECTS.items():
        seg = gt[f'{camera}/seg'][s] == int(gt[f'{name}_id'])
        regions[name], _, maps[name] = real_regions(frame, baseline, seg, truths[name], object_id,
                                                  half=half if name == 'peg' else None)
        if name == 'box':
            regions[name]['hole_rim'], maps[name]['rim_projection'] = rim_region(
                frame, truths[name], regions[name]['gt_visible'], gt['box_pose'][s],
                float(half[0]), float(gt['box_hole_radii']), centre)
    return truths, regions, maps


def profile(frame, results, seam):
    """A horizontal cross-section selected from GT seam alone, never responses."""
    # Prefer seams with horizontal face transitions to make a horizontal profile.
    v, u = np.nonzero(seam)
    if not len(v):
        return None
    h, w = seam.shape
    idx = np.argmin((v-np.median(v))**2+(u-np.median(u))**2)
    v, u = int(v[idx]), int(u[idx])
    x = np.arange(max(0, u-12), min(w, u+13))
    row = dict(v=v, reference_u=u, u=x.tolist(), methods={})
    for method, result in results.items():
        f = result.objects[PEG]
        a = np.rad2deg(f.normal_variation_rad[v, x])
        values = [float(t) if np.isfinite(t) else None for t in a]
        peak = int(x[np.nanargmax(a)]) if np.isfinite(a).any() else None
        row['methods'][method] = dict(variation_deg=values, peak_u=peak,
                                      peak_offset_px=abs(peak-u) if peak is not None else None,
                                      widths_above_deg={str(t): int(np.count_nonzero(a > t)) for t in THRESHOLDS})
    return row


def bounds(mask, fallback, margin=8):
    y, x = np.nonzero(mask if mask.any() else fallback)
    if not len(x):
        return (0, mask.shape[1]-1, mask.shape[0]-1, 0)
    return (max(0, x.min()-margin), min(mask.shape[1]-1, x.max()+margin),
            min(mask.shape[0]-1, y.max()+margin), max(0, y.min()-margin))


def comparison_plot(output, frame, results, regions, maps):
    base = results['baseline']
    union = base.objects[PEG].valid_points | base.objects[BOX].valid_points
    views = [('Full view', None, (0, 511, 511, 0)),
             ('Peg planes / actual seam', PEG, bounds(regions['peg']['gt_visible'], union)),
             ('Box physical hole rim', BOX, bounds(regions['box']['hole_rim'], regions['box']['gt_visible'], 12))]
    colors = [('peg', 'plane_interior', '#00ffcc'), ('peg', 'face_seam_2px', '#ff8000'),
              ('box', 'hole_rim', '#ffff00'), ('peg', 'occlusion_2px', '#ff55cc'),
              ('peg', 'same_plane_tracker_cut_2px', '#4488ff')]
    for validity in (False, True):
        fig, axes = plt.subplots(3, 6, figsize=(22, 12), constrained_layout=True)
        for axs, (title, oid, extent) in zip(axes, views):
            axs[0].imshow(frame.rgb)
            for name, key, color in colors:
                if regions[name][key].any():
                    axs[0].contour(regions[name][key], [.5], colors=[color], linewidths=.6)
            axs[0].set_title(title+'\nGT evaluation regions')
            for ax, method in zip(axs[1:], METHODS):
                image = np.full(frame.depth.shape, np.nan)
                for object_id in OBJECTS.values() if oid is None else (oid,):
                    f = results[method].objects[object_id]
                    if validity:
                        image[f.valid_points] = f.normal_valid[f.valid_points].astype(float)
                    else:
                        image[f.normal_variation_valid] = np.rad2deg(f.normal_variation_rad[f.normal_variation_valid])
                cmap = plt.get_cmap('viridis' if validity else 'turbo').with_extremes(bad='#cccccc')
                im = ax.imshow(image, cmap=cmap, vmin=0, vmax=1 if validity else 45)
                ax.set_facecolor('#cccccc')
                ax.set_title(method+(' normal validity' if validity else ' variation (deg)'))
                for name, key, color in colors[:3]:
                    if (oid is None or oid == OBJECTS[name]) and regions[name][key].any():
                        ax.contour(regions[name][key], [.5], colors=[color], linewidths=.5)
            for ax in axs:
                ax.set(xlim=extent[:2], ylim=extent[2:]); ax.axis('off')
        fig.colorbar(im, ax=axes[:, 1:], shrink=.6, label='Normal valid (0/1)' if validity else 'Degrees, clipped at 45')
        fig.suptitle(f'Source {frame.source_frame}: identical source pixels / masks; fixed scales\n'
                     'Cyan=plane ROI, orange=face seam, yellow=visible physical rim, pink=occlusion, blue=tracker cut. '
                     + ('Dark=invalid, gray=unobserved.' if validity else 'Gray=unavailable variation; see separate validity panels.'))
        fig.savefig(output/f'f{frame.source_frame:04d}_{"validity" if validity else "comparison"}.png', dpi=120)
        plt.close(fig)
    p = profile(frame, results, maps['peg']['seam'])
    if p:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for method, row in p['methods'].items():
            ax.plot(np.array(p['u'])-p['reference_u'], row['variation_deg'], '.-', label=method)
        ax.axvline(0, color='k', ls=':'); ax.legend(ncol=5, fontsize=8)
        ax.set(xlabel='u offset from selected GT seam pixel', ylabel='Variation (deg)',
               title=f'Source {frame.source_frame}, v={p["v"]}: same horizontal cross-section')
        fig.savefig(output/f'f{frame.source_frame:04d}_profile.png', dpi=140); plt.close(fig)
    return p


def synthetic_experiment(output, intrinsic):
    rows, primary = [], {}
    for kind in ('front', 'tilted', 'crease', 'step'):
        for z in (.2, .25, .35, .6, 1.):
            for phase in (0., .00025, .0005, .00075):
                frame, truth, regions = synthetic_case(kind, intrinsic, z0=z, phase_m=phase)
                for mode in ('float', 'floor', 'nearest'):
                    obs = frame if mode == 'float' else replace(frame, depth=quantize(frame.depth, mode))
                    results = compare(obs, len(rows))
                    row = dict(geometry=kind, z0_m=z, phase_m=phase, quantization=mode, methods={})
                    for method in METHODS:
                        f = results[method].objects[PEG]
                        row['methods'][method] = {k: response_metrics(f, m, truth) for k, m in regions.items()}
                        if kind in ('crease', 'step'):
                            ref = np.zeros(frame.depth.shape, bool); ref[131:381, 255:257] = True
                            row['methods'][method]['localization'] = {}
                            for t in THRESHOLDS:
                                a, b, c = localization(f, ref, frame.masks[PEG], t)
                                row['methods'][method]['localization'][str(t)] = dict(
                                    **a, response_to_seam_px=stats(b), seam_to_response_px=stats(c))
                    rows.append(row)
                    if z == .25 and phase == .00025 and mode in ('float', 'floor'):
                        # Keep only cropped maps for the figure, not dense results.
                        primary[f'{kind}/{mode}'] = {m: np.rad2deg(results[m].objects[PEG].normal_variation_rad[126:386, 226:286]).copy() for m in METHODS}
        print(f'Synthetic geometry complete: {kind}', flush=True)
    # Supplementary surface-direction sweep; keep original Task 02-A ROIs above.
    directions = []
    for z in (.25, .6):
        for slope in (.2, .8):
            for azimuth in (0, 45, 90, 135):
                frame, _, regions = synthetic_case('tilted', intrinsic, z0=z, phase_m=.00025)
                a, b = slope*np.cos(np.deg2rad(azimuth)), slope*np.sin(np.deg2rad(azimuth))
                ray = rays(frame.depth.shape, intrinsic)
                depth = (z+.00025)/(1-a*ray[..., 0]-b*ray[..., 1])
                frame = replace(frame, depth=depth.astype(np.float32))
                truth = np.broadcast_to(np.array([a, b, -1.])/np.sqrt(a*a+b*b+1), (*depth.shape, 3))
                for mode in ('float', 'floor', 'nearest'):
                    obs = frame if mode == 'float' else replace(frame, depth=quantize(frame.depth, mode))
                    results = compare(obs)
                    directions.append(dict(z0_m=z, slope=slope, azimuth_deg=azimuth, quantization=mode,
                                           methods={m: response_metrics(results[m].objects[PEG], regions['plane'], truth) for m in METHODS}))
    controls = {}
    frame, truth, _ = synthetic_case('front', intrinsic, z0=.25025)
    mask = frame.masks[PEG].copy(); mask[:, 256:] = False
    depth = frame.depth.copy(); depth[240:272, 240:272] = 0
    for label, obs in [('mask_cut', replace(frame, masks={PEG: mask})), ('invalid_depth_hole', replace(frame, depth=depth))]:
        results = compare(obs)
        controls[label] = {m: response_metrics(results[m].objects[PEG], obs.masks[PEG], truth) for m in METHODS}
    fig, axes = plt.subplots(len(primary), 5, figsize=(17, 20), constrained_layout=True)
    for axs, (label, images) in zip(axes, primary.items()):
        for ax, method in zip(axs, METHODS):
            ax.imshow(images[method], cmap='turbo', vmin=0, vmax=45)
            ax.set_title(label+' '+method, fontsize=9); ax.axis('off')
    fig.suptitle('Same crop: flat false responses and preservation of crease / depth step. Gray/white=missing variation.')
    fig.savefig(output/'synthetic.png', dpi=120); plt.close(fig)
    return dict(cases=rows, direction_sweep=directions, controls=controls,
                original_task02a_conditions=240, supplementary_direction_conditions=len(directions))


def summary_plot(output, summary, local, synthetic):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for method in METHODS:
        for ax, region in zip(axes[0, :2], ('peg/plane_interior', 'box/hole_rim')):
            r = summary[f'{method}/{region}']
            ax.plot(THRESHOLDS, [r[f'high_{t}_fraction_of_points'] for t in THRESHOLDS], 'o-', label=method)
            ax.set(title=region, xlabel='Variation threshold (deg)', ylabel='Response / observed points')
        r = summary[f'{method}/peg/plane_interior']
        axes[0, 2].plot(np.arange(2.5, 180, 5), np.array(r['variation_hist_5deg'])/r['variation_valid'], label=method)
        axes[0, 2].set(title='Plane variation distribution', xlim=(0, 60), xlabel='Degrees', ylabel='Fraction / 5deg bin')
        for ax, key in zip(axes[1, :2], ('coverage_2px', 'response_to_seam_px')):
            ys = [local[f'{method}/{t}'][key] for t in THRESHOLDS]
            if key == 'response_to_seam_px': ys = [v['p95'] for v in ys]
            ax.plot(THRESHOLDS, ys, 'o-', label=method)
            ax.set(title='Peg seam '+key, xlabel='Threshold (deg)')
        cases = [r for r in synthetic['cases'] if r['geometry']=='crease' and r['z0_m']==.25 and r['phase_m']==.00025 and r['quantization']=='float']
        r = cases[0]['methods'][method]['edge_band']
        axes[1, 2].plot(THRESHOLDS, [r[f'high_{t}_fraction_of_points'] for t in THRESHOLDS], 'o-', label=method)
        axes[1, 2].set(title='Unquantized synthetic crease (same 6px band)', xlabel='Threshold (deg)', ylabel='Response / observed points')
    for ax in axes.ravel(): ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(output/'tradeoffs.png', dpi=140); plt.close(fig)


def evaluate(episode_path, masks_path, *, output_root, camera='base_camera', selected=(0, 15, 45, 80, 120), synthetic=True):
    episode_path, masks_path = Path(episode_path).resolve(), Path(masks_path).resolve()
    tracking_path = masks_path.with_name('tracking.json')
    if tracking_path.exists():
        tracking = json.loads(tracking_path.read_text())
        if tracking.get('camera') not in (None, camera): raise ValueError('camera differs from tracker')
        original = Path(tracking.get('input') or episode_path)
        if original.is_file() and original.resolve() != episode_path and digests(original)[1] != digests(episode_path)[1]:
            raise ValueError('episode differs from tracker')
    with np.load(episode_path, allow_pickle=False) as archive:
        required = [f'{camera}/seg', 'peg_id', 'box_id', 'peg_pose', 'box_pose', 'box_hole_pose', 'peg_half_sizes', 'box_hole_radii', 'seed']
        missing = set(required)-set(archive.files)
        if missing: raise ValueError(f'GT missing: {sorted(missing)}')
        gt = {k: archive[k] for k in required}
    with np.load(masks_path, allow_pickle=False) as archive: sources = archive['source_frame']
    if sources.ndim != 1 or not len(sources) or len(np.unique(sources)) != len(sources):
        raise ValueError('source_frame must be unique/nonempty')
    if set(selected)-set(sources.tolist()): raise ValueError('selected frames absent')
    iterator = iter_saved_rgbd(episode_path, masks_path, camera=camera)
    first = next(iterator)
    config = dict(task='02-B', input=str(episode_path), masks=str(masks_path), camera=camera,
                  input_sha256=digests(episode_path)[1], masks_sha256=digests(masks_path)[1],
                  baseline_geometry_sha256=digests(REPO/'visual/runtime/geometry.py')[1], methods=list(METHODS),
                  intrinsic=first.intrinsic.tolist(), parameters=vars(PARAMETERS), selected_sources=list(selected),
                  full_resolution=True, variation_neighbour_spacing_px=1, pca_min_points_including_centre=6,
                  pca_batch_size=4096, pca_rank_relative_tolerance=1e-8, pca_rank_absolute_tolerance_m2=1e-16,
                  gate='original unit-pixel adjacent gate; FD complete axis paths; PCA patch-local centre component',
                  roi_source='Task 02-A unchanged real_regions / rim_region and original synthetic_case',
                  roi_depth_residual_m=.0015, roi_metric_edge_margin_m=.004, roi_raster_erosion_px=3,
                  rim_metric_radius_m=.004, rim_raster_radius_px=2, seam_search_corridor_px=6,
                  high_variation_degrees=list(THRESHOLDS), device='cpu', gpu_executed=False,
                  python=platform.python_version(), numpy=np.__version__, scipy=importlib.metadata.version('scipy'),
                  matplotlib=matplotlib.__version__, platform=platform.platform(), synthetic_executed=synthetic,
                  timing='two objects; one warmup per method; rotated order; shared backprojection charged to every total; no I/O/GT/metrics',
                  feature_inputs='recorded depth/intrinsic/original tracker masks; GT strictly after extraction')
    config['source_sha256'] = {str(p.relative_to(REPO)): digests(p)[1] for p in (
        Path(__file__), Path(__file__).with_name('normal_estimators.py'), Path(__file__).with_name('normal_diagnostics.py'))}
    with ExperimentLogger(team='visual', config=config, seed=int(gt['seed']), model='task02b_normal_comparison', job_type='evaluate') as logger:
        output = Path(output_root).resolve()/logger.run_id; output.mkdir(parents=True, exist_ok=False)
        print(f'Output: {output}', flush=True)
        synthetic_data = synthetic_experiment(output, first.intrinsic) if synthetic else None
        compare(first)  # Warmup excluded from every reported real latency.
        pool, records, local = Pool(), [], {}
        for frame in chain((first,), iterator):
            results = compare(frame, len(records))  # GT unavailable to estimators.
            truths, regions, maps = real_truth(frame, results['baseline'], gt, camera)
            record = dict(source_frame=frame.source_frame, methods={}, paired={})
            plane = regions['peg']['plane_interior']
            common_n = np.logical_and.reduce([results[m].objects[PEG].normal_valid for m in METHODS]) & plane
            common_v = np.logical_and.reduce([results[m].objects[PEG].normal_variation_valid for m in METHODS]) & plane
            for method in METHODS:
                result = results[method]
                mr = dict(timings_ms=result.timings_ms, regions={}, seam_localization={})
                for name, object_id in OBJECTS.items():
                    f = result.objects[object_id]
                    for key, region in regions[name].items():
                        mr['regions'][f'{name}/{key}'] = pool.add(f'{method}/{name}/{key}', f, region,
                                                               truths[name]['normals'] if key.startswith('plane_') else None)
                f = result.objects[PEG]
                for label, region in [('common_normal', common_n), ('common_variation', common_v)]:
                    pool.add(f'{method}/paired/{label}', f, region, truths['peg']['normals'])
                pair = plane & f.normal_valid & results['baseline'].objects[PEG].normal_valid
                for which in ('baseline', method) if method != 'baseline' else ('baseline',):
                    pool.add(f'{method}/paired_with_baseline/{which}', results[which].objects[PEG], pair, truths['peg']['normals'])
                for t in THRESHOLDS:
                    values = localization(f, maps['peg']['seam'], regions['peg']['gt_visible'], t)
                    local.setdefault(f'{method}/{t}', []).append(values)
                    mr['seam_localization'][str(t)] = dict(**values[0], response_to_seam_px=stats(values[1]), seam_to_response_px=stats(values[2]))
                record['methods'][method] = mr
            if frame.source_frame in selected:
                record['profile'] = comparison_plot(output, frame, results, regions, maps)
                arrays = dict(source_frame=np.array(frame.source_frame), depth_m=frame.depth, intrinsic=frame.intrinsic)
                for method in METHODS:
                    for name, object_id in OBJECTS.items():
                        f = results[method].objects[object_id]
                        for field in fields(f): arrays[f'{method}/{name}/{field.name}'] = getattr(f, field.name)
                for name in OBJECTS:
                    arrays.update({f'evaluation/{name}/{k}': v for k, v in regions[name].items()})
                    for k in ('face', 'normals'): arrays[f'evaluation/{name}/{k}'] = truths[name][k]
                arrays['evaluation/peg/seam_endpoints'] = maps['peg']['seam']
                np.savez_compressed(output/f'f{frame.source_frame:04d}.npz', **arrays)
            records.append(record)
            if len(records) == 1 or len(records) % 20 == 0: print(f'Real frames: {len(records)}/{len(sources)}', flush=True)
        summary, ls = pool.summary(), localization_summary(local)
        latency = {m: {k: stats([r['methods'][m]['timings_ms'][k] for r in records])
                        for k in ('geometry', 'normals', 'discontinuity_features', 'total')} for m in METHODS}
        data = dict(run_id=logger.run_id, config=config, git_commit=logger.metadata['git_commit'],
                    git_dirty=logger.metadata['git_dirty'], evaluated_frames=len(records), summary=summary,
                    seam_localization=ls, latency_ms=latency, synthetic=synthetic_data, frames=records)
        save_json(output/'comparison.json', data)
        if synthetic: summary_plot(output, summary, ls, synthetic_data)
        images = ['tradeoffs.png', 'synthetic.png']+[f'f{s:04d}_{k}.png' for s in selected for k in ('comparison', 'validity', 'profile')]
        body = ''.join(f'<p><a href="{p}"><img src="{p}" alt="{p}"></a></p>' for p in images if (output/p).exists())
        (output/'comparison.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Task 02-B</title>
<style>body{{font:15px system-ui;max-width:1800px;margin:24px auto}}img{{width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>Task 02-B — quantization-robust normals</h1><p>Run {logger.run_id}; {len(records)} ep000 frames. CPU NumPy.
<a href="comparison.json">Complete counts, distributions, paired metrics, threshold sweeps and latency</a>.</p>
<p>Baseline unchanged. FD2/FD4 and PCA5/PCA7 preserve original resolution and use the same 1px variation comparison.
GT is evaluation-only. Fixed Task 02-A regions and 5/10/20/30 degree thresholds. Boundary bands may overlap.
Seam distance statistics use a fixed 6px corridor, alongside 1/2/3px coverage and planar false responses.
Pooled pixel-frame statistics are temporally dependent. No final estimator/threshold adopted.</p>{body}
<details><summary>Configuration</summary><pre>{html.escape(json.dumps(config, indent=2))}</pre></details></html>''', encoding='utf-8')
        for p in sorted(output.iterdir()): logger.link_output(p, kind='metrics' if p.suffix=='.json' else None)
        logger.log(summary, prefix='eval/task02b'); logger.log(latency, prefix='eval/task02b/latency_ms')
        logger.record(dataset_sha256=config['input_sha256'], masks_sha256=config['masks_sha256'],
                      evaluation_device='cpu', evaluated_frames=len(records), gpu_executed=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=REPO/'visual/data/step2/ep000.npz')
    parser.add_argument('--masks', type=Path, default=REPO/'visual/results/runtime_tracking/ep000/masks.npz')
    parser.add_argument('--camera', default='base_camera')
    parser.add_argument('--out', type=Path, default=REPO/'visual/results/normal_comparison')
    args = parser.parse_args()
    print(evaluate(args.input, args.masks, output_root=args.out, camera=args.camera))


if __name__ == '__main__': main()
