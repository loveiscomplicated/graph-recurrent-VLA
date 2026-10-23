"""Task 02-A: paired quantization and saved RGB-D normal reliability diagnostics.

CPU only, fresh UUID output directory, no inference, estimator changes or uploads.
Run with WANDB_MODE=disabled for entirely local experiment tracking.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from utils.experiment_artifacts import digests
from utils.experiment_logger import ExperimentLogger
from visual.runtime.geometry import GeometryParameters, process_geometry
from visual.runtime.rgbd import BOX, PEG, iter_saved_rgbd
from visual.runtime.test.normal_diagnostics import (
    angular_error, box_primitives, dilate, endpoints, erode, metrics, normal_blockers,
    oracle, pose_rotation, quantize, real_regions, rim_region, scale_probe, stats, synthetic_case,
)

OBJECTS = {'peg': PEG, 'box': BOX}
COLORS = {'plane': '#00ffcc', 'seam': '#ff8000', 'rim': '#ffdd00',
          'occlusion': '#ff55cc', 'contamination': '#ff2222', 'mask_cut': '#4488ff'}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def summarize_rows(rows, values, errors=None):
    """Micro averages and pooled quantiles; no averaging per-frame percentiles."""
    keys = ('region_pixels', 'points', 'normals', 'variation_valid', 'depth_discontinuity',
            'mask_boundary', 'invalid_depth_boundary', 'incomplete_normal_neighbours',
            'high_5_count', 'high_10_count', 'high_20_count', 'high_30_count', 'jump_or_high20_count')
    total = {k: sum(r[k] for r in rows) for k in keys}
    n, nv, vv = total['points'], total['normals'], total['variation_valid']
    total.update(frames=len(rows), normal_valid_fraction=nv/n if n else None,
                 variation_valid_fraction=vv/n if n else None,
                 variation_deg=stats(np.concatenate(values)))
    rn = total['region_pixels']
    total.update(observed_fraction_of_region=n/rn if rn else None,
                 normal_fraction_of_region=nv/rn if rn else None,
                 jump_or_high20_fraction_of_region=total['jump_or_high20_count']/rn if rn else None)
    total['variation_hist_5deg'] = np.sum([r['variation_hist_5deg'] for r in rows], axis=0).tolist()
    for t in (5, 10, 20, 30):
        total[f'high_{t}_fraction_of_valid'] = total[f'high_{t}_count']/vv if vv else None
        total[f'high_{t}_fraction_of_points'] = total[f'high_{t}_count']/n if n else None
    if errors is not None:
        total['angular_error_deg'] = stats(np.concatenate(errors))
    return total


def synthetic_experiment(output, intrinsic):
    rows, primaries, scales = [], {}, []
    for kind in ('front', 'tilted', 'crease', 'step'):
        for z in (.2, .25, .35, .6, 1.):
            for phase in (0., .00025, .0005, .00075):
                frame, truth, regions = synthetic_case(kind, intrinsic, z0=z, phase_m=phase)
                original = process_geometry(frame)
                float_feature = original.objects[PEG]
                runs = {'float': (frame, original)}
                for mode in ('floor', 'nearest'):
                    changed = replace(frame, depth=quantize(frame.depth, mode))
                    runs[mode] = (changed, process_geometry(changed))
                row = dict(geometry=kind, z0_m=z, phase_m=phase, conditions={})
                for mode, (observation, result) in runs.items():
                    f = result.objects[PEG]
                    row['conditions'][mode] = dict(
                        regions={k: metrics(f, m, truth) for k, m in regions.items()},
                        all_support=metrics(f, f.valid_points, truth),
                        depth_change_mm=stats((observation.depth-frame.depth)[frame.masks[PEG]]*1000))
                rows.append(row)
                if z == .25 and phase == .00025:
                    primaries[kind] = (frame, truth, regions, runs)
                if kind == 'tilted' and phase == .00025:
                    qframe, qresult = runs['floor']
                    scales.append(dict(z0_m=z, rows=scale_probe(qframe, qresult.objects[PEG], PEG,
                                                              regions['plane'], truth)))
    # Controls: observation cuts of an exactly flat surface do not create a corner.
    frame, truth, _ = synthetic_case('front', intrinsic, z0=.25025)
    mask = frame.masks[PEG].copy(); mask[:, 256:] = False
    cut = replace(frame, masks={PEG: mask})
    hole_depth = frame.depth.copy(); hole_depth[240:272, 240:272] = 0
    hole = replace(frame, depth=hole_depth)
    controls = {}
    for name, observation in [('mask_cut', cut), ('invalid_depth_hole', hole)]:
        result = process_geometry(observation)
        controls[name] = dict(metrics=metrics(result.objects[PEG], observation.masks[PEG], truth),
                              blockers=normal_blockers(result, PEG))
    synthetic_plot(output/'synthetic.png', primaries)
    return dict(primary_z0_m=.25, primary_phase_m=.00025, primary_quantization='floor',
                depth_sweep_m=[.2, .25, .35, .6, 1.], phase_sweep_m=[0, .00025, .0005, .00075],
                slope_xy=[.4, -.25], crease_slopes_xy=[[-.6, .15], [.6, .15]], step_m=.05,
                mask_bounds_vu=[128, 384, 128, 384], edge_band_half_width_px=3,
                planar_border_margin_px=3, controls=controls, cases=rows, scale_sweep=scales)


def synthetic_plot(path, primary):
    fig, axes = plt.subplots(4, 5, figsize=(18, 13), constrained_layout=True)
    for axs, (kind, (frame, truth, regions, runs)) in zip(axes, primary.items()):
        clean, q = runs['float'][1].objects[PEG], runs['floor'][1].objects[PEG]
        images = [np.where(frame.masks[PEG], frame.depth, np.nan)*1000,
                  np.rad2deg(clean.normal_variation_rad), np.rad2deg(q.normal_variation_rad),
                  angular_error(q.normals, truth),
                  np.where(q.valid_points, q.normal_valid.astype(float)+2*q.depth_discontinuity, np.nan)]
        titles = [f'{kind}: float depth (mm)', 'Float variation (deg)', '1mm floor variation (deg)',
                  '1mm floor normal error (deg)', 'Normal valid=1, depth jump=+2']
        for j, (ax, a, title) in enumerate(zip(axs, images, titles)):
            im = ax.imshow(a, vmin=0 if j in (1, 2, 3, 4) else None,
                           vmax=45 if j in (1, 2, 3) else 3 if j == 4 else None, cmap='turbo')
            ax.set(xlim=(122, 390), ylim=(390, 122), title=title); ax.axis('off')
            fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle('Paired synthetic observations: same camera, masks and extractor; only depth quantized\n'
                 'fx=fy=256; z0=250.25mm; no noise or filtering', fontsize=14)
    fig.savefig(path, dpi=130); plt.close(fig)


def real_plot(path, frame, result, regions, maps, oracles):
    fig, axes = plt.subplots(3, 4, figsize=(18, 14), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(frame.rgb); axes[0].set_title('RGB with evaluation regions')
    plane = regions['peg']['plane_interior']
    overlays = dict(plane=plane, seam=regions['peg']['face_seam_2px'],
                    rim=regions['box']['hole_rim'], occlusion=regions['peg']['occlusion_2px']|regions['box']['occlusion_2px'],
                    contamination=(result.objects[PEG].valid_points & regions['peg']['contamination'])|
                                  (result.objects[BOX].valid_points & regions['box']['contamination']),
                    mask_cut=regions['peg']['same_plane_tracker_cut_2px'])
    for name, mask in overlays.items():
        if mask.any():
            axes[0].contour(mask, levels=[.5], colors=[COLORS[name]], linewidths=.65)
    axes[0].legend(handles=[Patch(color=COLORS[k], label=k) for k in COLORS], fontsize=8, loc='upper right')
    support = result.objects[PEG].valid_points | result.objects[BOX].valid_points
    d = np.where(support, frame.depth*1000, np.nan)
    im = axes[1].imshow(d, cmap='turbo', vmin=150, vmax=700)
    axes[1].set_title('Observed depth (mm), fixed scale'); fig.colorbar(im, ax=axes[1], shrink=.7)
    for name, base in [('peg', 2), ('box', 7)]:
        f = result.objects[OBJECTS[name]]
        var = np.rad2deg(f.normal_variation_rad)
        error = angular_error(f.normals, oracles[name]['normals'])
        error[~regions[name]['gt_visible']] = np.nan
        panel = [(var, 'normal variation (deg)', 45, 'turbo'),
                 (np.where(f.valid_points, f.normal_valid, np.nan), 'normal validity', 1, 'viridis'),
                 (error, 'normal error vs GT (deg)', 45, 'turbo')]
        for j, (a, title, vmax, cmap) in enumerate(panel):
            im = axes[base+j].imshow(a, cmap=cmap, vmin=0, vmax=vmax)
            axes[base+j].set_title(f'{name}: {title}'); fig.colorbar(im, ax=axes[base+j], shrink=.7)
        boundary = np.zeros((*frame.depth.shape, 3))
        boundary[..., 0] = f.depth_discontinuity
        boundary[..., 1] = f.mask_boundary
        boundary[..., 2] = f.depth_validity_boundary
        axes[base+3].imshow(boundary)
        axes[base+3].set_title(f'{name}: jump / mask / invalid = R/G/B')
    # Peg-only zoom shows quantization terraces and selected broad planar interior.
    y, x = np.nonzero(plane)
    if len(x):
        lo_x, hi_x, lo_y, hi_y = max(0, x.min()-8), x.max()+8, max(0, y.min()-8), y.max()+8
    else:
        lo_x, hi_x, lo_y, hi_y = 0, 511, 0, 511
    f = result.objects[PEG]
    axes[6].imshow(np.where(f.valid_points, frame.depth*1000, np.nan), cmap='turbo')
    if plane.any(): axes[6].contour(plane, [.5], colors=['#00ffcc'], linewidths=.6)
    axes[6].set_title('Peg depth terraces; cyan=verified plane ROI')
    axes[6].set(xlim=(lo_x, hi_x), ylim=(hi_y, lo_y))
    axes[11].imshow(np.rad2deg(f.normal_variation_rad), cmap='turbo', vmin=0, vmax=45)
    if plane.any(): axes[11].contour(plane, [.5], colors=['#00ffcc'], linewidths=.6)
    axes[11].set_title('Same Peg crop: variation (0-45 deg)')
    axes[11].set(xlim=(lo_x, hi_x), ylim=(hi_y, lo_y))
    # Full panels share a fixed object viewport for comparisons between frames.
    for i, ax in enumerate(axes):
        if i not in (6, 11): ax.set(xlim=(0, 511), ylim=(511, 0))
        ax.axis('off')
    fig.suptitle(f'Source {frame.source_frame}: observed features; GT used only to evaluate regions\n'
                 'Black validity=unavailable; NaN variation is missing, not zero. Scales fixed across frames.', fontsize=14)
    fig.savefig(path, dpi=130); plt.close(fig)


def boundary_plot(path, frame, result, regions):
    fig, axes = plt.subplots(2, 4, figsize=(16, 10), constrained_layout=True)
    for axs, (name, object_id) in zip(axes, OBJECTS.items()):
        f = result.objects[object_id]
        region = regions[name]
        axs[0].imshow(frame.rgb)
        for key, color in [('face_seam_2px', '#ff8000'), ('occlusion_2px', '#ff55cc'),
                           ('same_plane_tracker_cut_2px', '#4488ff'), ('hole_rim', '#ffff00')]:
            if key in region and region[key].any():
                axs[0].contour(region[key], [.5], colors=[color], linewidths=.8)
        axs[0].set_title(f'{name}: GT seam=orange, rim=yellow\nocclusion=pink, same-face mask cut=blue')
        im = axs[1].imshow(np.rad2deg(f.normal_variation_rad), cmap='turbo', vmin=0, vmax=45)
        axs[1].set_title('Observed normal variation (deg)'); fig.colorbar(im, ax=axs[1], shrink=.65)
        axs[2].imshow(np.where(f.valid_points, f.normal_valid, np.nan), cmap='viridis', vmin=0, vmax=1)
        axs[2].set_title('Normal valid: yellow=1, dark=0')
        boundary = np.stack((f.depth_discontinuity, f.mask_boundary, f.depth_validity_boundary), -1).astype(float)
        axs[3].imshow(boundary)
        axs[3].set_title('Observed jump / mask / invalid = R/G/B')
        contour = region.get('hole_rim', region['face_seam_2px'])
        if contour.any():
            for ax in axs[1:]: ax.contour(contour, [.5], colors=['white'], linewidths=.65)
        y, x = np.nonzero(f.object_mask | region['gt_visible'])
        bounds = (max(0, x.min()-8), min(511, x.max()+8), min(511, y.max()+8), max(0, y.min()-8)) if len(x) else (0, 511, 511, 0)
        for ax in axs: ax.set(xlim=bounds[:2], ylim=bounds[2:]); ax.axis('off')
    fig.suptitle(f'Frame {frame.source_frame}: boundary detail; evaluation regions may overlap\n'
                 'White outline in feature maps: visible physical hole rim (Box) or face seam (Peg).', fontsize=13)
    fig.savefig(path, dpi=150); plt.close(fig)


def profile_plot(path, frame, result, regions, truth):
    # Dominant face and its longer image axis, selected independently of features.
    plane = regions['plane_interior']
    if not plane.any(): return
    labels, counts = np.unique(truth['face'][plane], return_counts=True)
    face = labels[counts.argmax()]
    face_roi = plane & (truth['face'] == face)
    row_counts, col_counts = face_roi.sum(axis=1), face_roi.sum(axis=0)
    along_v = col_counts.max() > row_counts.max()
    fixed = int(np.argmax(col_counts if along_v else row_counts))
    line_roi = face_roi[:, fixed] if along_v else face_roi[fixed]
    coords = np.flatnonzero(line_roi)
    lo, hi = int(coords.min()), int(coords.max())
    coord = np.arange(lo, hi+1)
    vi, ui = (coord, fixed) if along_v else (fixed, coord)
    good = line_roi[coord]
    f = result.objects[PEG]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(coord, frame.depth[vi, ui]*1000, '.-', label='observed 1mm depth', ms=3)
    axes[0].plot(coord, truth['depth'][vi, ui]*1000, label='analytic GT face, evaluation only', lw=1)
    axes[0].set_ylabel('Depth (mm)'); axes[0].legend(fontsize=8)
    for c, label in enumerate(('nx', 'ny', 'nz')):
        axes[1].plot(coord, np.where(good, f.normals[vi, ui, c], np.nan), label=label)
    axes[1].set_ylabel('CV unit normal'); axes[1].legend(fontsize=8, ncol=3)
    axes[2].plot(coord, np.where(good, np.rad2deg(f.normal_variation_rad[vi, ui]), np.nan), label='variation')
    axis_name, fixed_name = ('v', 'u') if along_v else ('u', 'v')
    axes[2].axhline(20, color='r', ls=':', lw=1)
    axes[2].set(ylabel='Variation (deg)', xlabel=f'Source {axis_name}')
    fig.suptitle(f'Frame {frame.source_frame}, {fixed_name}={fixed}, GT signed face {face}: verified planar pixels only')
    fig.savefig(path, dpi=140); plt.close(fig)
    return dict(varying_axis=axis_name, fixed_axis=fixed_name, fixed_index=fixed,
                signed_face=int(face), axis_min=lo, axis_max=hi, valid_planar_coordinates=coord[good].tolist())


def evaluate(episode_path, masks_path, *, output_root, camera='base_camera', selected=(0, 15, 45, 80, 120)):
    episode_path, masks_path = Path(episode_path).resolve(), Path(masks_path).resolve()
    # Check source identity before producing a new evaluation run.
    tracking_path = masks_path.with_name('tracking.json')
    if tracking_path.exists():
        tracking = json.loads(tracking_path.read_text())
        if tracking.get('camera') not in (None, camera): raise ValueError('camera differs from tracking.json')
        original = Path(tracking.get('input') or episode_path)
        if original.is_file() and original.resolve() != episode_path and digests(original)[1] != digests(episode_path)[1]:
            raise ValueError('episode differs from tracking.json')
    with np.load(episode_path, allow_pickle=False) as archive:
        required = [f'{camera}/seg', 'peg_id', 'box_id', 'peg_pose', 'box_pose', 'box_hole_pose',
                    'peg_half_sizes', 'box_hole_radii', 'seed']
        missing = set(required)-set(archive.files)
        if missing: raise ValueError(f'evaluation GT missing: {sorted(missing)}')
        gt = {k: archive[k] for k in required}
    with np.load(masks_path, allow_pickle=False) as archive:
        sources = archive['source_frame']
    if sources.ndim != 1 or not len(sources) or len(np.unique(sources)) != len(sources):
        raise ValueError('source_frame must be unique and nonempty')
    if set(selected)-set(sources.tolist()): raise ValueError('selected sources absent from tracking archive')
    iterator = iter_saved_rgbd(episode_path, masks_path, camera=camera)
    first = next(iterator)
    from itertools import chain
    config = dict(task='02-A', input=str(episode_path), masks=str(masks_path), camera=camera,
                  input_sha256=digests(episode_path)[1], masks_sha256=digests(masks_path)[1],
                  baseline_geometry_sha256=digests(REPO/'visual/runtime/geometry.py')[1],
                  selected_sources=list(selected), intrinsic=first.intrinsic.tolist(),
                  parameters=vars(GeometryParameters()), device='cpu', gpu_executed=False,
                  python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                  scipy=importlib.metadata.version('scipy'), matplotlib=matplotlib.__version__,
                  roi_depth_residual_m=.0015, roi_metric_edge_margin_m=.004, roi_raster_erosion_px=3,
                  rim_metric_radius_m=.004, rim_raster_radius_px=2, high_variation_degrees=[5, 10, 20, 30],
                  actual_feature_inputs='recorded depth, intrinsic and original tracker masks only',
                  gt_use='evaluation regions, oriented angular truth, analytic depth residuals only')
    config['diagnostic_source_sha256'] = {str(p.relative_to(REPO)): digests(p)[1] for p in
        (Path(__file__), Path(__file__).with_name('normal_diagnostics.py'), Path(__file__).with_name('test_normal_diagnostics.py'))}
    with ExperimentLogger(team='visual', config=config, seed=int(gt['seed']),
                          model='task02a_unchanged_organized_geometry', job_type='evaluate') as logger:
        output = Path(output_root).resolve()/logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        print(f'Output: {output}', flush=True)
        synthetic = synthetic_experiment(output, first.intrinsic)
        records, pooled, pooled_errors = [], {}, {}
        for frame in chain((first,), iterator):
            s = frame.source_frame
            result = process_geometry(frame)  # Actual feature computation: no GT.
            half = gt['peg_half_sizes']
            box_rotation = pose_rotation(gt['box_pose'][s])
            hole_centre = ((gt['box_hole_pose'][s, :3]-gt['box_pose'][s, :3]) @ box_rotation)[1:]
            boxes = box_primitives(float(half[0]), float(gt['box_hole_radii']), hole_centre)
            truths = dict(peg=oracle(frame, gt['peg_pose'][s], [(np.zeros(3), half)]),
                          box=oracle(frame, gt['box_pose'][s], boxes))
            record = dict(source_frame=s, frame_idx=frame.frame_idx, timestamp=frame.timestamp,
                          overlap_pixels=int(result.geometry.overlap_mask.sum()), objects={})
            selected_regions, maps, plane_base = {}, {}, None
            for name, object_id in OBJECTS.items():
                f = result.objects[object_id]
                seg = gt[f'{camera}/seg'][s] == int(gt[f'{name}_id'])
                regions, base, boundary_maps = real_regions(frame, result, seg, truths[name], object_id,
                                                            half=half if name == 'peg' else None)
                if name == 'peg': plane_base = base
                if name == 'box':
                    rim, projected = rim_region(frame, truths[name], regions['gt_visible'], gt['box_pose'][s],
                                                 float(half[0]), float(gt['box_hole_radii']), hole_centre)
                    regions['hole_rim'] = rim
                    regions['hole_rim_normal_valid'] = rim & f.normal_valid
                    regions['hole_rim_normal_invalid'] = rim & ~f.normal_valid
                    boundary_maps['rim_projection'] = projected
                selected_regions[name], maps[name] = regions, boundary_maps
                rows = {k: metrics(f, m, truths[name]['normals'] if k.startswith('plane_') else None)
                        for k, m in regions.items()}
                visible = regions['gt_visible']
                residual = (frame.depth-truths[name]['depth'])*1000
                observed_z, analytic_z = frame.depth[visible], truths[name]['depth'][visible]
                expected_floor = (np.floor(truths[name]['depth']*1000)/1000).astype(np.float32)
                floor_agree_map = visible & (frame.depth == expected_floor)
                floor_agree = floor_agree_map[visible]
                row = dict(mask_pixels=int(f.object_mask.sum()), gt_pixels=int(seg.sum()),
                           analytic_hit_gt_fraction=float(np.mean(np.isfinite(truths[name]['depth'][seg]))) if seg.any() else None,
                           depth_consistent_gt_fraction=float(visible.sum()/seg.sum()) if seg.any() else None,
                           depth_residual_mm=stats(residual[visible]),
                           mm_grid_residual_m=stats(np.abs(observed_z-np.rint(observed_z.astype(float)*1000)/1000)),
                           analytic_floor_match_fraction=float(floor_agree.mean()) if visible.any() else None,
                           analytic_floor_match_count=int(floor_agree.sum()),
                           regions=rows, blockers=normal_blockers(result, object_id))
                if name == 'peg':
                    plane = regions['plane_interior']
                    # Derivative geometry: 1-pixel lateral spacing at the same z.
                    row['plane_depth_m'] = stats(frame.depth[plane])
                    row['plane_lateral_pixel_spacing_mm'] = stats(frame.depth[plane]/frame.intrinsic[0, 0]*1000)
                    row['q_over_2pixel_lateral_span'] = stats(.001/(2*frame.depth[plane]/frame.intrinsic[0, 0]))
                    vv = plane & f.normal_variation_valid
                    matched = vv & erode(floor_agree_map, 2)
                    high = vv & (np.rad2deg(f.normal_variation_rad) > 20)
                    row['plane_floor_dependency_match'] = dict(
                        variation_valid=int(vv.sum()), matched=int(matched.sum()),
                        fraction=float(matched.sum()/vv.sum()) if vv.any() else None,
                        high20=int(high.sum()), high20_matched=int((high & matched).sum()),
                        high20_matched_fraction=float((high & matched).sum()/high.sum()) if high.any() else None)
                    # Same-plane mask cuts and occlusions are evaluated separately.
                record['objects'][name] = row
                for region_name, roi in regions.items():
                    key = f'{name}/{region_name}'
                    vv = roi & f.normal_variation_valid & f.valid_points
                    pooled.setdefault(key, []).append(np.rad2deg(f.normal_variation_rad[vv]))
                    if region_name.startswith('plane_'):
                        nv = roi & f.normal_valid & f.valid_points
                        pooled_errors.setdefault(key, []).append(angular_error(f.normals[nv], truths[name]['normals'][nv]))
            if s in selected:
                record['scale_probe'] = scale_probe(frame, result.objects[PEG], PEG, plane_base, truths['peg']['normals'])
                record['profile'] = profile_plot(output/f'f{s:04d}_profile.png', frame, result, selected_regions['peg'], truths['peg'])
                real_plot(output/f'f{s:04d}.png', frame, result, selected_regions, maps, truths)
                boundary_plot(output/f'f{s:04d}_boundaries.png', frame, result, selected_regions)
                arrays = dict(source_frame=np.array(s), depth_m=frame.depth, intrinsic=frame.intrinsic)
                for name, object_id in OBJECTS.items():
                    f = result.objects[object_id]
                    arrays.update({f'{name}/{field.name}': getattr(f, field.name) for field in fields(f)})
                    arrays.update({f'evaluation/{name}/{k}': v for k, v in selected_regions[name].items()})
                    arrays[f'evaluation/{name}/analytic_depth_m'] = truths[name]['depth']
                    arrays[f'evaluation/{name}/face_id'] = truths[name]['face']
                np.savez_compressed(output/f'f{s:04d}.npz', **arrays)
            records.append(record)
            if len(records) == 1 or len(records) % 20 == 0: print(f'Real frames: {len(records)}/{len(sources)}', flush=True)
        summary = {}
        for key, values in pooled.items():
            name, region = key.split('/')
            summary[key] = summarize_rows([r['objects'][name]['regions'][region] for r in records], values,
                                          pooled_errors.get(key))
        data = dict(run_id=logger.run_id, config=config, git_commit=logger.metadata['git_commit'],
                    git_dirty=logger.metadata['git_dirty'], synthetic=synthetic, evaluated_frames=len(records),
                    selected_source_frames=list(selected), summary=summary, frames=records)
        save_json(output/'diagnostics.json', data)
        distribution_plot(output/'distributions.png', data)
        scale_plot(output/'scale_probe.png', data)
        write_html(output/'diagnostics.html', data)
        for path in sorted(output.iterdir()):
            logger.link_output(path, kind='metrics' if path.suffix == '.json' else None)
        logger.log(summary, prefix='eval/task02a')
        logger.record(dataset_sha256=config['input_sha256'], masks_sha256=config['masks_sha256'],
                      evaluation_device='cpu', evaluated_frames=len(records), gpu_executed=False)
    return output


def distribution_plot(path, data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    xs = np.arange(2.5, 180, 5)
    for name, ax in zip(('peg', 'box'), axes[0]):
        keys = ['all', 'plane_interior', 'contamination', 'face_seam_2px', 'occlusion_2px', 'hole_rim']
        for key in keys:
            r = data['summary'].get(f'{name}/{key}')
            if r and r['variation_valid']:
                ax.plot(xs, np.array(r['variation_hist_5deg'])/r['variation_valid'], label=f'{key} (n={r["variation_valid"]})')
        ax.set(xlim=(0, 90), xlabel='Normal variation (deg)', ylabel='Fraction per 5 deg bin', title=f'{name}: pooled 121 frames')
        ax.legend(fontsize=7)
    sources = [r['source_frame'] for r in data['frames']]
    for key in ('all', 'plane_interior', 'contamination', 'face_seam_2px', 'occlusion_2px'):
        ys = [r['objects']['peg']['regions'][key]['high_20_fraction_of_valid'] for r in data['frames']]
        axes[1, 0].plot(sources, [np.nan if v is None else v for v in ys], label=key, lw=1)
    axes[1, 0].set(xlabel='Source frame', ylabel='Fraction >20 deg / valid variation', title='Peg: time distribution')
    axes[1, 0].legend(fontsize=7)
    for name, key in [('peg', 'all'), ('box', 'all'), ('box', 'hole_rim')]:
        ys = [r['objects'][name]['regions'][key]['normal_valid_fraction'] for r in data['frames']]
        axes[1, 1].plot(sources, [np.nan if v is None else v for v in ys], label=f'{name}/{key}')
    axes[1, 1].set(xlabel='Source frame', ylabel='Normals / observed points', title='Coverage (missing remains missing)', ylim=(-.05, 1.05))
    axes[1, 1].legend(fontsize=8)
    fig.savefig(path, dpi=140); plt.close(fig)


def scale_plot(path, data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for r in data['frames']:
        if 'scale_probe' not in r: continue
        rows = r['scale_probe']
        x = [v['stride_px'] for v in rows]
        for ax, key in zip(axes[:2], ('angular_error_deg', 'variation_deg')):
            y = [v['larger_spacing'][key]['p95'] for v in rows]
            ax.plot(x, y, 'o-', label=f'f{r["source_frame"]}')
    axes[0].set(ylabel='p95 normal error (deg)', xlabel='Neighbour spacing (px)', title='Verified flat Peg ROI, shared safe region')
    axes[1].set(ylabel='p95 variation (deg)', xlabel='Neighbour spacing (px)', title='No averaging; same baseline extractor')
    for s in (1, 2, 4):
        rows = data['synthetic']['scale_sweep']
        axes[2].plot([r['z0_m'] for r in rows],
                     [next(v for v in r['rows'] if v['stride_px']==s)['larger_spacing']['variation_deg']['p95'] for r in rows],
                     'o-', label=f'synthetic stride={s}')
    axes[2].set(xlabel='Central depth (m)', ylabel='p95 variation (deg)', title='Tilted plane, fixed K and 1mm quantization')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(path, dpi=140); plt.close(fig)


def write_html(path, data):
    images = ['synthetic.png', 'distributions.png', 'scale_probe.png']
    images += [f'f{s:04d}{suffix}.png' for s in data['selected_source_frames'] for suffix in ('', '_boundaries', '_profile')]
    body = ''.join(f'<p><a href="{p}"><img src="{p}" alt="{p}"></a></p>' for p in images if (path.parent/p).exists())
    path.write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task 02-A</title>
<style>body{{font:15px system-ui;max-width:1600px;margin:24px auto;padding:16px}}img{{width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>Task 02-A — normal reliability and depth quantization</h1>
<p>Run {data['run_id']}; all {data['evaluated_frames']} saved frames; unchanged CPU NumPy extractor.
<a href="diagnostics.json">Full numeric metrics, counts, histograms and parameters</a>.</p>
<p>Actual feature inputs are recorded depth, intrinsic and tracker masks only. GT defines evaluation regions
and oriented normal truth. Plane ROI: visible cuboid face, depth residual ≤1.5mm, 4mm from other faces,
then 3px erosion independently per signed face. No selection by measured normal or variation.
Hole rim requires analytic physical line proximity (4mm, 2px), not just a segmentation contour.</p>
<p>20° is a diagnostic threshold, not an edge class. All 5/10/20/30° rates and missing support are reported.
Boundary ROIs can overlap; counts are not additive. Pooled values are pixel-frame weighted.
Subsampling probes preserve pixel centres and compare baseline values on the same centres; they change
both stencil span and normal-comparison separation. Only verified planar regions are compared.</p>
<p>GT annotations and analytic depth in NPZ are under evaluation/. They never replace observed depth.
No smoothing, PCL, new production estimator, structural classification, sampling or Graph changes.</p>
{body}<details><summary>Configuration</summary><pre>{html.escape(json.dumps(data['config'], indent=2))}</pre></details></html>''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=REPO/'visual/data/step2/ep000.npz')
    parser.add_argument('--masks', type=Path, default=REPO/'visual/results/runtime_tracking/ep000/masks.npz')
    parser.add_argument('--camera', default='base_camera')
    parser.add_argument('--out', type=Path, default=REPO/'visual/results/normal_diagnostics')
    args = parser.parse_args()
    print(evaluate(args.input, args.masks, output_root=args.out, camera=args.camera))


if __name__ == '__main__': main()
