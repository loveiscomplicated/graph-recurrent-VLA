"""Replay saved Frame 0 Box S2/S3 Priority FPS, then explain the verified order.

WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_priority_fps_trace
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.experiment_artifacts import digests
from utils.experiment_logger import ExperimentLogger
from visual.runtime.geometry import build_organized_geometry
from visual.runtime.rgbd import BOX, iter_saved_rgbd
from visual.runtime.sampling import fps_indices
from visual.runtime.test.run_rim_bottleneck import (
    SOURCE_RUN, geometry_evidence, require_equal, source_contract,
    validate_saved_selection, write_json,
)
from visual.runtime.test.sampling_evaluation import CoverageMatcher, evaluation_context

REPO = Path(__file__).resolve().parents[3]
METHODS = {'S2': ('fd2', 'S2_n256_r0.25_t20_s0', 297, 57, 13),
           'S3': ('pca5', 'S3_n256_r0.25_t20_s0', 599, 91, 3)}
FRAME = 0
PRIORITY_COUNT = 32


def fps_seed(seed=0, object_id=BOX):
    return int(np.random.SeedSequence([seed, object_id]).generate_state(1)[0])


def verify_fps_source(source, expected_sha):
    relative = 'visual/runtime/sampling.py'
    saved = source/'evaluated_source'/relative
    current = REPO/relative
    require_equal(digests(saved)[1], expected_sha, 'evaluated sampling.py SHA256')
    def definitions(path):
        tree = ast.parse(path.read_text())
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ('fps_indices', 'sample_points', 'extract_candidates', 'allocate_budget')}
    require_equal(definitions(current), definitions(saved), 'FPS/candidate/allocation definitions')
    return {'saved_sha256': expected_sha, 'current_sha256': digests(current)[1],
            'fps_candidate_allocation_ast_exact': True}


def trace_order(points, selected, flat, depth, candidate_references, negatives=None):
    """Explain an already verified FPS order; no selection or RNG occurs here."""
    xyz = np.asarray(points, dtype=np.float64)
    selected = np.asarray(selected, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.int64)
    refs = np.asarray(candidate_references, dtype=bool)
    if xyz.shape != (len(flat), 3) or refs.ndim != 2 or refs.shape[0] != len(flat):
        raise ValueError('trace input shape mismatch')
    if not len(selected) or len(np.unique(selected)) != len(selected) or ((selected < 0) | (selected >= len(flat))).any():
        raise ValueError('invalid FPS order')
    distance = np.full(len(flat), np.inf, dtype=np.float64)
    x, y, z = (np.ascontiguousarray(xyz[:, i]) for i in range(3))
    covered = np.zeros(refs.shape[1], dtype=bool)
    remaining = np.ones(len(flat), dtype=bool)
    rows = []
    for step, local in enumerate(selected):
        local = int(local)
        a = remaining & refs.any(axis=1)
        b = remaining & refs[:, ~covered].any(axis=1)
        if step == 0:
            if not np.isinf(distance).all():
                raise AssertionError('initial distances must all be +inf')
            maximum = None
            ties = []
        else:
            maximum = float(distance[remaining].max())
            ties = np.flatnonzero(remaining & (distance == maximum)).tolist()
            if local != ties[0]:
                raise AssertionError(f'step {step}: selected {local}, first argmax {ties[0]}')
        def group(mask):
            if not mask.any() or step == 0:
                return {'remaining_count': int(mask.sum()), 'max_min_squared_distance_m2': None,
                        'gap_to_winner_m2': None, 'ties_winner': None}
            best = float(distance[mask].max())
            return {'remaining_count': int(mask.sum()), 'max_min_squared_distance_m2': best,
                    'gap_to_winner_m2': maximum-best, 'ties_winner': best == maximum}
        new_ids = np.flatnonzero(refs[local] & ~covered).tolist()
        roi = {name: bool(mask.ravel()[flat[local]]) for name, mask in (negatives or {}).items()}
        roi['known_negative_union'] = any(roi.values())
        item = dict(step=step+1, initial_rng_choice=step == 0,
            candidate_local_index=local, flat_index=int(flat[local]), xyz_m=xyz[local].tolist(),
            depth_m=float(depth[local]), min_squared_distance_m2=None if step == 0 else float(distance[local]),
            argmax_value_m2=maximum, exact_tie_count=None if step == 0 else len(ties),
            tied_local_indices=None if step == 0 else ties,
            first_tied_local_index=None if step == 0 else ties[0], rule_satisfied=True,
            all_remaining=group(remaining), rim_A=group(a), uncovered_rim_B=group(b),
            matched_rim_reference_ids=np.flatnonzero(refs[local]).tolist(),
            rim_reference_matched=bool(refs[local].any()), new_reference_ids=new_ids,
            new_reference_count=len(new_ids), selected_negative_roi=roi)
        covered |= refs[local]
        item['cumulative_reference_count'] = int(covered.sum())
        rows.append(item)
        new_dist = (x-x[local])**2+(y-y[local])**2+(z-z[local])**2
        np.minimum(distance, new_dist, out=distance)
        distance[local] = -np.inf
        remaining[local] = False
    return rows


def _frame_and_context(comparison, episode, masks):
    config = comparison['config']
    frame = next(f for f in iter_saved_rgbd(episode, masks, camera=config['camera']) if f.source_frame == FRAME)
    geometry = build_organized_geometry(frame)
    np.testing.assert_array_equal(geometry_evidence(frame).frame.geometry.xyz, geometry.xyz)
    with np.load(episode, allow_pickle=False) as ep:
        keys = [f'{config["camera"]}/seg', 'peg_id', 'box_id', 'peg_pose', 'box_pose',
                'box_hole_pose', 'peg_half_sizes', 'box_hole_radii']
        gt = {key: ep[key] for key in keys}
    return frame, geometry, gt


def create_plots(output, frame, methods, refs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, (name, row) in zip(axes, methods.items()):
        c = np.asarray(row['candidate_flat_indices'])
        v, u = np.unravel_index(c, frame.depth.shape)
        ax.imshow(frame.rgb)
        ax.scatter(u, v, s=8, c='#ff6a40', alpha=.55, label='orientation candidates')
        ax.scatter(refs[:, 0], refs[:, 1], s=12, c='#e927d8', marker='+', label='GT rim (evaluation only)')
        for item in row['trace']:
            vv, uu = np.unravel_index(item['flat_index'], frame.depth.shape)
            ax.scatter(uu, vv, s=38, c='#ffe34d', edgecolors='black', linewidths=.4)
            ax.text(uu+2, vv-2, str(item['step']), fontsize=6, color='white',
                    bbox=dict(facecolor='black', alpha=.6, pad=.5))
        ax.set(title=f'{name}: {len(c)} candidates, 32 priority points', xlim=(min(u.min(),refs[:,0].min())-12,max(u.max(),refs[:,0].max())+12),
               ylim=(max(v.max(),refs[:,1].max())+12,min(v.min(),refs[:,1].min())-12))
        ax.set_aspect('equal'); ax.legend(fontsize=7, loc='upper right')
    fig.savefig(output/'priority_overlay.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for name, row in methods.items():
        steps = [t['step'] for t in row['trace']][1:]
        for key, label, style in [('argmax_value_m2','winner','-'),('rim_A','rim A','--'),('uncovered_rim_B','uncovered rim B',':')]:
            vals = [t[key] if key == 'argmax_value_m2' else t[key]['max_min_squared_distance_m2'] for t in row['trace'][1:]]
            axes[0].plot(steps, vals, style, marker='.', ms=2, label=f'{name} {label}')
        axes[1].step([t['step'] for t in row['trace']], [t['cumulative_reference_count'] for t in row['trace']],
                     where='post', label=name)
    axes[0].set_yscale('log')
    axes[0].set(ylabel='Preselection min squared distance (m², log scale)',
                title='FPS distance competition; step 1 is RNG and omitted')
    axes[1].set(xlabel='Priority selection step', ylabel='Unique GT rim references covered', ylim=(-1, 16))
    for ax in axes: ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=8)
    fig.savefig(output/'priority_distances_coverage.png', dpi=150); plt.close(fig)


def create_html(output, run_id):
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Priority FPS trace</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}}img{{width:100%;height:auto}}a{{margin-right:1rem}}</style>
<h1>Task03-D Phase 2 — Frame 0 Box Priority FPS</h1><p>Run {run_id}; S2 FD2 and S3 PCA5; 32 points each. GT rim marks and coverage are evaluation only; they did not enter FPS.</p>
<p><a href="priority_fps_trace.json">Full step trace JSON</a><a href="verification.json">Verification</a><a href="tests.log">Test log</a><a href="source_hashes.json">Source hashes</a><a href="implementation.patch">Implementation diff</a></p>
<h2>Candidate locations and numbered selection overlay</h2><img src="priority_overlay.png" alt="S2 and S3 candidates and numbered priority selections">
<h2>Step distance competition and cumulative GT coverage</h2><img src="priority_distances_coverage.png" alt="FPS distance competition and coverage">
<p><a href="priority_overlay.png">Overlay PNG</a><a href="priority_distances_coverage.png">Distance and coverage PNG</a></p></html>'''
    (output/'priority_fps_trace.html').write_text(body, encoding='utf-8')


def diagnose(source, output_root):
    if os.environ.get('WANDB_MODE') != 'disabled':
        raise ValueError('set WANDB_MODE=disabled for this local-only diagnostic')
    source = Path(source).resolve()
    comparison_path = source/'comparison.json'
    comparison = json.loads(comparison_path.read_text())
    require_equal(comparison['run_id'], SOURCE_RUN, 'Task03 source run')
    config = comparison['config']
    source_checks = source_contract(source, config)
    fps_source_check = verify_fps_source(source, config['source_sha256']['visual/runtime/sampling.py'])
    episode, masks = Path(config['input']), Path(config['masks'])
    segmap = episode.with_name('segmentation_id_map.json')
    tracked = masks.with_name('tracking.json')
    inputs = [comparison_path, episode, masks, segmap, tracked, source/'f0000_samples.npz', source/'f0000_metrics.json']
    for path, key in [(episode, 'input_sha256'), (masks, 'masks_sha256'), (segmap, 'segmentation_map_sha256')]:
        require_equal(digests(path)[1], config[key], key)
    before = {str(p): digests(p)[1] for p in inputs}
    current_sources = [Path(__file__), Path(__file__).with_name('run_rim_bottleneck.py'),
                       Path(__file__).with_name('sampling_evaluation.py'), REPO/'visual/runtime/sampling.py']
    source_hashes = {str(p.relative_to(REPO)): digests(p)[1] for p in current_sources}
    diagnostic_config = dict(task='03-D', phase=2, source_run_id=SOURCE_RUN, source_directory=str(source),
        frame=FRAME, methods=list(METHODS), priority_count=PRIORITY_COUNT, fps_seed=fps_seed(),
        tolerance_px_m=[2., .003], input_sha256=before, source_sha256=source_hashes,
        device='cpu', gpu_executed=False, simulator_executed=False, sample_points_executed=False,
        candidate_extraction_executed=False, feature_extraction_executed=False, surface_fps_executed=False,
        python=platform.python_version(), numpy=np.__version__)
    frame, geometry, gt = _frame_and_context(comparison, episode, masks)
    phase1 = json.loads((REPO/'experiments/d3f14ee494894daabb8ba2fab8e1b00c/diagnostic.json').read_text())
    phase1_box_ids = {m: phase1['frames'][0]['conditions'][m]['reference_ids']['P'] for m in METHODS}
    metrics = json.loads((source/'f0000_metrics.json').read_text())
    results, replayed = {}, {}
    with np.load(source/'f0000_samples.npz', allow_pickle=False) as archive:
        for method, (estimator, key, expected_candidates, expected_candidate_refs, expected_coverage) in METHODS.items():
            alignment = validate_saved_selection(archive, key, frame, geometry)
            box = archive[key+'/object_id'] == BOX
            priority = archive[key+'/priority_selected'][box]
            stored_flat = archive[key+'/flat_index'][box][priority]
            require_equal(len(stored_flat), PRIORITY_COUNT, method+' saved priority count')
            metadata_names = [name for name in archive.files if name.startswith(key+'/metadata/')]
            if not metadata_names or any(len(archive[name]) != len(archive[key+'/flat_index']) for name in metadata_names):
                raise ValueError(method+': saved feature row alignment mismatch')
            if not archive[key+'/metadata/orientation_candidate'][box][priority].all():
                raise ValueError(method+': saved priority feature role mismatch')
            if archive[key+'/metadata/depth_only_candidate'][box][priority].any():
                raise ValueError(method+': unexpected depth-only priority row')
            candidate_flat = np.asarray(archive[f'evidence/{estimator}/1/20/orientation_flat'], dtype=np.int64)
            require_equal(len(candidate_flat), expected_candidates, method+' candidates')
            if not np.all(candidate_flat[:-1] < candidate_flat[1:]) or not geometry.observed_mask(BOX).ravel()[candidate_flat].all():
                raise ValueError(method+': candidate raster order or Box support mismatch')
            points = geometry.xyz.reshape(-1, 3)[candidate_flat].astype(np.float64)
            depth = frame.depth.ravel()[candidate_flat].astype(np.float64)
            positions = np.searchsorted(candidate_flat, stored_flat)
            np.testing.assert_array_equal(candidate_flat[positions], stored_flat)
            seed = fps_seed(config['conditions'][key]['seed'], BOX)
            order = fps_indices(points, PRIORITY_COUNT, seed=seed)  # exactly one live replay per method
            np.testing.assert_array_equal(candidate_flat[order], stored_flat)
            np.testing.assert_array_equal(points[order], archive[key+'/points'][box][priority])
            require_equal(int(order[0]), int(np.random.default_rng(seed).integers(len(points))), method+' initial RNG local row')
            counts = metrics[key]['objects']['box']['counts']
            require_equal(counts['priority'], PRIORITY_COUNT, method+' quota')
            require_equal(counts['candidates'], expected_candidates, method+' candidate count')
            replayed[method] = dict(estimator=estimator, candidate_flat=candidate_flat, points=points,
                depth=depth, order=order, stored_flat=stored_flat, seed=seed, alignment=alignment,
                metadata_fields=len(metadata_names), counts=counts,
                expected_candidate_refs=expected_candidate_refs, expected_coverage=expected_coverage)
        # Both FPS replays and exact saved-order checks have finished. GT starts here.
        context = evaluation_context(frame, geometry_evidence(frame), gt, config['camera'], config['robot_ids'])
        for suffix, index in [('uv', 0), ('xyz', 1)]:
            np.testing.assert_array_equal(archive[f'evaluation/box/reference_{suffix}'], context['references']['box'][index])
        refs = tuple(archive[f'evaluation/box/reference_{suffix}'] for suffix in ('uv', 'xyz'))
        matcher = CoverageMatcher(geometry, BOX, refs)
        for method, replay in replayed.items():
            candidate_flat = replay['candidate_flat']
            points, depth, order = replay['points'], replay['depth'], replay['order']
            stored_flat = replay['stored_flat']
            matched_ids, _ = matcher.reference_hits(candidate_flat)
            require_equal(len(matched_ids), replay['expected_candidate_refs'], method+' candidate rim coverage')
            matrix = matcher.adjacencies[matcher.key(2., .003)][np.searchsorted(matcher.flat, candidate_flat)].toarray().astype(bool)
            trace = trace_order(points, order, candidate_flat, depth, matrix, context['negatives']['box'])
            priority_hits, _ = matcher.reference_hits(stored_flat)
            require_equal(len(priority_hits), replay['expected_coverage'], method+' priority coverage')
            require_equal(priority_hits.tolist(), phase1_box_ids[method], method+' Phase1 priority IDs')
            require_equal(trace[-1]['cumulative_reference_count'], replay['expected_coverage'], method+' trace coverage')
            results[method] = dict(estimator=replay['estimator'], candidate_count=len(candidate_flat),
                candidate_flat_indices=candidate_flat.tolist(), candidate_rim_reference_count=len(matched_ids),
                candidate_rim_source_node_count=int(matrix.any(axis=1).sum()),
                stored_priority_flat_indices=stored_flat.tolist(), replay_priority_flat_indices=candidate_flat[order].tolist(),
                exact_saved_order_match=True, rng_seed=replay['seed'], initial_rng_local_index=int(order[0]),
                priority_reference_ids=priority_hits.tolist(), priority_reference_count=len(priority_hits),
                source_alignment=replay['alignment'], saved_feature_rows_aligned=True,
                saved_feature_fields=replay['metadata_fields'], saved_allocation=replay['counts'], trace=trace)
    require_equal({str(p): digests(p)[1] for p in inputs}, before, 'source input hashes after read')
    result = dict(source_run_id=SOURCE_RUN, frame=FRAME, config=diagnostic_config,
        source_contract=source_checks, fps_source_check=fps_source_check, input_hashes_unchanged=True,
        methods=results, pass_all=True, interpretation=[
            'Distances are reconstructed from the verified saved prefix, without a second FPS selection algorithm.',
            'GT rim and negative ROI annotations are added after exact FPS replay.',
            'Distance competition describes the observed selection; causal contributions remain untested.'])
    with ExperimentLogger(team='visual', config=diagnostic_config, seed=0,
            model='task03d_priority_fps_trace', job_type='evaluate', root=output_root) as logger:
        result['run_id'] = logger.run_id
        output = logger.directory
        write_json(output/'priority_fps_trace.json', result)
        write_json(output/'source_hashes.json', {'inputs': before, 'diagnostic_sources': source_hashes,
                                                  'evaluated_fps': fps_source_check})
        create_plots(output, frame, results, refs[0])
        create_html(output, logger.run_id)
        for path in current_sources + [Path(__file__).with_name('test_priority_fps_trace.py')]:
            dest = output/'diagnostic_source'/path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            logger.link_output(dest, kind='reports')
        for filename, kind in [('priority_fps_trace.json','metrics'), ('source_hashes.json','metrics'),
             ('priority_overlay.png','reports'), ('priority_distances_coverage.png','reports'),
             ('priority_fps_trace.html','reports')]:
            logger.link_output(output/filename, kind=kind)
        logger.log({m: {'candidate_rim_references': r['candidate_rim_reference_count'],
                        'priority_rim_references': r['priority_reference_count']}
                    for m, r in results.items()}, step=FRAME, prefix='eval/task03d')
        logger.record(source_run_id=SOURCE_RUN, frame=FRAME, exact_replays=True,
                      inputs_unchanged=True, gpu_executed=False, simulator_executed=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO/'visual/results/sampling'/SOURCE_RUN)
    parser.add_argument('--out', type=Path, default=REPO/'experiments')
    args = parser.parse_args()
    print(diagnose(args.source, args.out))


if __name__ == '__main__':
    main()
