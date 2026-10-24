"""Task03-D Phase 3: saved Priority evaluation and one shared-initial FPS pair.

WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_priority_fps_seed_control \
    --baseline /path/to/baseline.json

Only the two control selections call FPS on real data. No full sampler, feature,
normal or candidate extraction runs. GT annotations enter control scoring only
when both selections are complete. Tests repeat FPS only on synthetic inputs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import html
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils.experiment_artifacts import digests
from utils.experiment_logger import ExperimentLogger
from visual.runtime.rgbd import BOX, PEG
from visual.runtime.sampling import fps_indices
from visual.runtime.test.run_priority_fps_trace import (
    REPO, METHODS, _frame_and_context, fps_seed, verify_fps_source,
)
from visual.runtime.test.run_rim_bottleneck import (
    SOURCE_RUN, compare_ids, depth_summary, geometry_evidence, require_equal,
    source_contract, validate_saved_selection, write_json,
)
from visual.runtime.test.sampling_evaluation import CoverageMatcher, evaluation_context

PHASE1 = 'd3f14ee494894daabb8ba2fab8e1b00c'
PHASE2 = 'aa5d71078e0841fd886b5026d6030ebc'
REPORT = REPO/'docs/visual-graph-integration/03d_priority_fps_seed_control_phase3.md'
FINAL = {'S2': [16, 10, 10], 'S3': [3, 5, 3]}
FIRST = {'S2': [101598, 119061, 112919], 'S3': [96553, 117010, 102621]}
COUNT = 32
TEST_MODULES = [f'visual.runtime.test.{name}' for name in (
    'test_priority_fps_seed_control', 'test_priority_fps_trace', 'test_sampling',
    'test_sampling_geometry', 'test_sampling_evaluation')]


def hashes(paths):
    return {str(p.relative_to(REPO)): digests(p)[1] for p in sorted(set(paths))}


def verify_preserved(baseline):
    actual = hashes([REPO/p for p in baseline['hashes']])
    require_equal(actual, baseline['hashes'], 'pre-existing workspace/artifact hashes')
    return {'status': 'PASS', 'protected_file_count': len(actual), 'all_sha256_unchanged': True}


def forbid_sampling():
    """Scoped tripwires, including both the imported FPS alias and its source."""
    stack = ExitStack()
    for target in ('visual.runtime.sampling.fps_indices', __name__+'.fps_indices',
                   'visual.runtime.sampling.sample_points',
                   'visual.runtime.sampling.extract_sampling_evidence',
                   'visual.runtime.sampling.extract_candidates'):
        stack.enter_context(patch(target, side_effect=AssertionError('forbidden in saved analysis: '+target)))
    return stack


def load_saved(archive, config, metrics, frame, geometry, phase2):
    """Read/validate saved arrays; never regenerate features, candidates or FPS."""
    candidates, selections, checks = {}, {}, {}
    for method, (estimator, _, n_candidates, _, _) in METHODS.items():
        candidate = np.asarray(archive[f'evidence/{estimator}/1/20/orientation_flat'])
        if candidate.ndim != 1 or candidate.dtype.kind not in 'iu':
            raise ValueError('candidate flat indices must be one-dimensional integers')
        require_equal(len(candidate), n_candidates, method+' candidate count')
        if not np.all(candidate[:-1] < candidate[1:]):
            raise ValueError('candidate raster order changed')
        if not geometry.observed_mask(BOX).ravel()[candidate].all():
            raise ValueError('candidate outside observed Box')
        np.testing.assert_array_equal(candidate, phase2['methods'][method]['candidate_flat_indices'])
        candidates[method] = dict(flat=candidate.copy(),
            points=geometry.xyz.reshape(-1, 3)[candidate].copy())
        selections[method] = {}
        previous = []
        for seed in (0, 1, 2):
            key = f'{method}_n256_r0.25_t20_s{seed}'
            expected = dict(method=method, num_points=256, structural_ratio=.25,
                            orientation_threshold_deg=20, seed=seed)
            require_equal(config['conditions'][key], expected, key+' condition')
            alignment = validate_saved_selection(archive, key, frame, geometry)
            flat, ids = archive[key+'/flat_index'], archive[key+'/object_id']
            priority = archive[key+'/priority_selected']
            require_equal(len(flat), 256, key+' total nodes')
            for oid, name, budget in ((PEG, 'peg', 129), (BOX, 'box', 127)):
                choose = ids == oid
                require_equal(int(choose.sum()), budget, key+' '+name+' budget')
                np.testing.assert_array_equal(priority[choose], np.arange(budget) < COUNT)
                counts = metrics[key]['objects'][name]['counts']
                for field, value in dict(budget=budget, selected=budget, priority=COUNT,
                                         orientation_priority=COUNT, depth_priority=0,
                                         surface_fps=budget-COUNT).items():
                    require_equal(counts[field], value, key+' '+name+' '+field)
            names = sorted(k for k in archive.files if k.startswith(key+'/metadata/'))
            require_equal(len(names), 23, key+' metadata fields')
            features = {k.split('/metadata/')[1]: archive[k] for k in names}
            if any(len(v) != len(flat) for v in features.values()):
                raise ValueError('feature row alignment mismatch')
            box = ids == BOX
            np.testing.assert_array_equal(features['orientation_candidate'][box], np.isin(flat[box], candidate))
            np.testing.assert_array_equal(features['structural_candidate'][box], np.isin(flat[box], candidate))
            if not features['orientation_candidate'][box & priority].all() or features['depth_only_candidate'][box & priority].any():
                raise ValueError('Priority orientation/depth role mismatch')
            if not (features['object_mask'].all() and features['valid_points'].all()):
                raise ValueError('saved metadata support mismatch')
            # Source-keyed equality of every saved feature wherever seeds overlap.
            overlap_counts = []
            for old_flat, old_features in previous:
                shared, a, b = np.intersect1d(flat, old_flat, return_indices=True)
                for name, value in features.items():
                    np.testing.assert_array_equal(value[a], old_features[name][b], err_msg=key+'/'+name)
                overlap_counts.append(len(shared))
            previous.append((flat.copy(), features))
            selected = flat[box & priority]
            require_equal(len(selected), COUNT, key+' Priority count')
            require_equal(int(selected[0]), FIRST[method][seed], key+' first flat')
            positions = np.searchsorted(candidate, selected)
            if (positions >= len(candidate)).any():
                raise ValueError('saved Priority outside candidates')
            np.testing.assert_array_equal(candidate[positions], selected)
            np.testing.assert_array_equal(candidates[method]['points'][positions], archive[key+'/points'][box & priority])
            final = metrics[key]['objects']['box']['structural']['tolerance']['2px_3mm']
            require_equal(final['gt_visible'], 103, key+' final denominator')
            require_equal(final['sampled'], FINAL[method][seed], key+' stored Final coverage')
            require_equal(final['gt_coverage'], FINAL[method][seed]/103, key+' stored Final fraction')
            selections[method][seed] = dict(flat=selected.copy(), local=positions,
                xyz=archive[key+'/points'][box & priority].copy(), final=dict(final))
            checks[key] = dict(**alignment, saved_feature_fields=23,
                source_keyed_feature_overlap_counts=overlap_counts,
                feature_and_role_alignment=True, candidate_local_global_exact=True,
                priority_is_saved_prefix=True, saved_final_coverage_unchanged=True)
    return candidates, selections, checks


def point_summary(flat, frame, negatives, reference_ids, matching_flat, xyz):
    flat = np.asarray(flat, np.int64)
    masks = dict(negatives)
    masks['known_negative_union'] = np.logical_or.reduce(list(negatives.values()))
    occupancy = {name: dict(count=int(mask.ravel()[flat].sum()), denominator=len(flat),
                           fraction=float(mask.ravel()[flat].mean())) for name, mask in masks.items()}
    ids = list(map(int, reference_ids))
    if ids != sorted(set(ids)):
        raise ValueError('reference IDs must be sorted and deduplicated')
    return dict(node_count=len(flat), selected_flat_indices=flat.tolist(),
        selected_xyz_m=np.asarray(xyz).tolist(),
        initial_flat_index=int(flat[0]), initial_xyz_m=np.asarray(xyz)[0].tolist(),
        rim_matching_node_count=len(matching_flat), rim_matching_flat_indices=list(map(int, matching_flat)),
        priority_reference_ids=ids, priority_reference_count=len(ids),
        priority_coverage=dict(numerator=len(ids), denominator=103, value=len(ids)/103),
        priority_depth_m=depth_summary(frame.depth.ravel()[flat]), negative_roi=occupancy)


def summarize_saved(selections, phase1, phase2, matcher, frame, negatives):
    """Seed 0 IDs/hits come ONLY from Phase2; only seed 1/2 call the matcher."""
    results = {}
    for method in METHODS:
        results[method] = {}
        for seed, row in selections[method].items():
            flat, xyz = row['flat'], row['xyz']
            if seed == 0:
                original = phase2['methods'][method]
                np.testing.assert_array_equal(flat, original['stored_priority_flat_indices'])
                np.testing.assert_array_equal(xyz, [t['xyz_m'] for t in original['trace']])
                np.testing.assert_array_equal(frame.depth.ravel()[flat], [t['depth_m'] for t in original['trace']])
                ref_ids = original['priority_reference_ids']
                require_equal(ref_ids, phase1['frames'][0]['conditions'][method]['reference_ids']['P'], method+' seed0 Phase1/2 IDs')
                require_equal(sorted({i for t in original['trace'] for i in t['matched_rim_reference_ids']}), ref_ids,
                              method+' seed0 trace reference union')
                require_equal(len(ref_ids), original['priority_reference_count'], method+' seed0 coverage')
                hits = [t['flat_index'] for t in original['trace'] if t['rim_reference_matched']]
                origin = 'Phase2 reused; seed0 matcher not called'
            else:
                ref_ids, hits = matcher.reference_hits(flat, px=2., m=.003)
                origin = 'saved Priority evaluated by existing CoverageMatcher'
            result = point_summary(flat, frame, negatives, ref_ids, hits, xyz)
            if seed == 0:
                for name in result['negative_roi']:
                    require_equal(result['negative_roi'][name]['count'],
                        sum(t['selected_negative_roi'][name] for t in original['trace']), method+' seed0 ROI '+name)
            if result['priority_reference_count'] > row['final']['sampled']:
                raise ValueError('Priority coverage exceeds stored Final coverage')
            result.update(seed=seed, stage='B_saved_priority', priority_result_origin=origin,
                          final_sampling=dict(stage='A_saved_final', source='Task03 f0000_metrics.json',
                            numerator=row['final']['sampled'], denominator=103, value=row['final']['gt_coverage']))
            results[method][str(seed)] = result
    return results


def select_control(candidates, *, expected_shared=252, expected_initial=114957):
    """GT-free selection boundary. Preserve candidate raster order and XYZ."""
    require_equal(set(candidates), {'S2', 'S3'}, 'control methods')
    for method, row in candidates.items():
        flat, xyz = row['flat'], row['points']
        if flat.ndim != 1 or flat.dtype.kind not in 'iu' or not np.all(flat[:-1] < flat[1:]):
            raise ValueError('control candidates must keep ascending raster order')
        if xyz.shape != (len(flat), 3) or not np.isfinite(xyz).all() or len(flat) < COUNT:
            raise ValueError('invalid control candidate XYZ/count')
    shared = sorted(set(candidates['S2']['flat']) & set(candidates['S3']['flat']))
    require_equal(len(shared), expected_shared, 'candidate intersection')
    initial = int(shared[len(shared)//2])
    require_equal(initial, expected_initial, 'deterministic shared initial')
    local = {m: int(np.searchsorted(c['flat'], initial)) for m, c in candidates.items()}
    xyz = {m: c['points'][local[m]] for m, c in candidates.items()}
    np.testing.assert_array_equal(xyz['S2'], xyz['S3'])
    orders = {}
    # No RNG factory or integer draw may occur in the real initial path.
    with patch('visual.runtime.sampling.np.random.default_rng',
               side_effect=AssertionError('initial control must not consume RNG')):
        for method, candidate in candidates.items():
            order = fps_indices(candidate['points'], COUNT, seed=fps_seed(0, BOX), initial=[local[method]])
            if len(order) != COUNT or len(np.unique(order)) != COUNT or ((order < 0) | (order >= len(candidate['flat']))).any():
                raise ValueError('control requires 32 unique candidate rows')
            require_equal(int(order[0]), local[method], method+' initial included')
            selected = candidate['flat'][order]
            require_equal(len(np.unique(selected)), COUNT, method+' unique source nodes')
            np.testing.assert_array_equal(np.searchsorted(candidate['flat'], selected), order)
            orders[method] = dict(flat=selected.copy(), xyz=candidate['points'][order].copy(),
                                  local=order.copy())
    return orders, dict(rule='sorted(set(C_S2) & set(C_S3))[len(shared)//2]',
        shared_count=len(shared), shared_flat_indices=list(map(int, shared)),
        initial_flat_index=initial, initial_xyz_m=xyz['S2'].tolist(), candidate_local_index=local,
        exact_initial_xyz_equal=True, initial_rng_draws=0, real_fps_calls={'S2': 1, 'S3': 1},
        total_nodes_including_initial=COUNT, gt_selection_input=False)


def score_control(orders, matcher, frame, negatives):
    results = {}
    for method, row in orders.items():
        ids, hits = matcher.reference_hits(row['flat'], px=2., m=.003)
        results[method] = dict(stage='C_shared_initial_priority',
            selected_candidate_local_indices=row['local'].tolist(),
            **point_summary(row['flat'], frame, negatives, ids, hits, row['xyz']))
    return results


def create_plots(output, frame, saved, control, refs):
    colors = {'S2': '#1976b9', 'S3': '#df6d16'}
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.imshow(frame.rgb)
    initial_vu = []
    offsets = {'S2': [(8, 18), (8, -18), (-65, -15)],
               'S3': [(-65, -18), (-65, 12), (8, -14)]}
    for method, rows in saved.items():
        for seed, row in rows.items():
            v, u = divmod(row['initial_flat_index'], frame.depth.shape[1])
            initial_vu.append((v, u))
            ax.scatter(u, v, s=90, color=colors[method], marker=['o', 's', '^'][int(seed)], edgecolors='white')
            ax.annotate(f'{method} seed {seed}', (u, v), xytext=offsets[method][int(seed)],
                        textcoords='offset points', arrowprops=dict(arrowstyle='-', color=colors[method], lw=.6),
                        color=colors[method], fontsize=9, bbox=dict(facecolor='white', alpha=.85, pad=1))
    initial_vu = np.asarray(initial_vu)
    ax.set(title='Saved seed 0/1/2: first Box Priority point',
           xlim=(initial_vu[:, 1].min()-25, initial_vu[:, 1].max()+25),
           ylim=(initial_vu[:, 0].max()+20, initial_vu[:, 0].min()-20), xlabel='u (px)', ylabel='v (px)')
    fig.savefig(output/'seed_initial_points.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        x = np.arange(3); rows = list(saved[method].values())
        for dx, field, label, color in [(-.18, 'priority_reference_count', 'B: saved Priority (32)', colors[method]),
                                       (.18, 'final_sampling', 'A: stored Final (127 Box)', '#999999')]:
            values = [r[field] if field != 'final_sampling' else r[field]['numerator'] for r in rows]
            bars = ax.bar(x+dx, values, .34, label=label, color=color)
            ax.bar_label(bars, labels=[f'{n}/103' for n in values], fontsize=9)
        ax.set(title=method, xticks=x, xticklabels=['seed 0', 'seed 1', 'seed 2'], ylim=(0, 20), ylabel='Unique GT rim reference IDs (of 103)')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.savefig(output/'seed_priority_final_coverage.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    all_vu = np.column_stack(np.unravel_index(
        [flat for row in control.values() for flat in row['selected_flat_indices']], frame.depth.shape))
    xlim = (min(all_vu[:, 1].min(), refs[:, 0].min())-14,
            max(all_vu[:, 1].max(), refs[:, 0].max())+14)
    ylim = (max(all_vu[:, 0].max(), refs[:, 1].max())+14,
            min(all_vu[:, 0].min(), refs[:, 1].min())-14)
    for ax, method in zip(axes, METHODS):
        row = control[method]; vu = np.column_stack(np.unravel_index(row['selected_flat_indices'], frame.depth.shape))
        ax.imshow(frame.rgb)
        ax.scatter(refs[:, 0], refs[:, 1], marker='+', s=18, color='#e226bd', label='GT rim: evaluation only')
        ax.scatter(vu[:, 1], vu[:, 0], s=50, edgecolors='white', linewidths=.6, color=colors[method], label='32 Priority selections')
        ax.scatter(vu[0, 1], vu[0, 0], s=180, color='#ffe243', edgecolors='black', marker='*', label='shared initial')
        ax.set(title=f"{method}: C Priority {row['priority_reference_count']}/103", xlim=xlim, ylim=ylim, xlabel='u (px)', ylabel='v (px)')
        ax.legend(fontsize=8, loc='upper right')
    fig.savefig(output/'shared_initial_overlay.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    values = [control[m]['priority_reference_count'] for m in METHODS]
    bars = axes[0].bar(list(METHODS), values, color=list(colors.values()))
    axes[0].bar_label(bars, labels=[f'{n}/103 ({100*n/103:.2f}%)' for n in values])
    axes[0].set(title='C: same initial point, 32 Priority nodes', ylabel='Unique GT rim references', ylim=(0, max(values)+5))
    axes[1].scatter(refs[:, 0], refs[:, 1], color='#cccccc', s=24, label='103 GT refs: evaluation only')
    for method in METHODS:
        selected = refs[control[method]['priority_reference_ids']]
        axes[1].scatter(selected[:, 0], selected[:, 1], color=colors[method], marker='o' if method == 'S2' else 'x', s=50, label=method+' covered refs')
    axes[1].invert_yaxis(); axes[1].set_aspect('equal'); axes[1].legend(fontsize=8)
    axes[1].set(title='Unique reference coverage after control FPS', xlabel='u (px)', ylabel='v (px)')
    fig.savefig(output/'shared_initial_coverage.png', dpi=160); plt.close(fig)


def create_html(output, result, source):
    def link(path, label):
        return f'<a href="{html.escape(os.path.relpath(path, output))}">{html.escape(label)}</a>'
    links = ' | '.join(link(output/name, name) for name in (
        'priority_fps_seed_control.json', 'source_hashes.json', 'verification.json', 'tests.log',
        'implementation.patch', 'git_start.json', 'git_end.json', 'report_snapshot.md'))
    originals = ' | '.join(link(path, label) for path, label in (
        (source/'f0000_samples.npz', 'Task03 samples NPZ'), (source/'f0000_metrics.json', 'Task03 frame metrics'),
        (source/'comparison.json', 'Task03 config'),
        (REPO/'experiments'/PHASE1/'diagnostic.json', 'Phase1 JSON'),
        (REPO/'experiments'/PHASE2/'priority_fps_trace.json', 'Phase2 JSON'), (REPORT, 'Markdown report')))
    rows = ''.join(f"<tr><td>{m}</td><td>{s}</td><td>{r['initial_flat_index']}</td><td>{r['initial_xyz_m']}</td>"
        f"<td>{r['priority_reference_count']}/103</td><td>{r['final_sampling']['numerator']}/103</td></tr>"
        for m, seeds in result['saved'].items() for s, r in seeds.items())
    control_rows = ''.join(f"<tr><td>{m}</td><td>{r['node_count']}</td><td>{r['rim_matching_node_count']}</td>"
        f"<td>{r['priority_reference_count']}/103</td><td>{r['priority_reference_ids']}</td></tr>" for m, r in result['control'].items())
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Task03-D Phase 3 — FPS initial control</title><style>body{{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.5rem;border:1px solid #bbb;text-align:left;overflow-wrap:anywhere}}.scroll{{overflow-x:auto}}</style>
<h1>Task03-D Phase 3 — FPS initialization vs candidate distribution</h1>
<p>Run {result['run_id']}. Frame 0 Box; FD2/PCA5; n=256, ratio=.25, threshold=20°, Box budget=127, Priority quota=32. Same-pair 2px AND 3mm matching; 103 GT rim line references.</p>
<p>GT rim marks and negative ROIs are evaluation only. Neither GT nor ROI enters initial-point choice or FPS. Only one shared initial point was tested; no new sampling policy was adopted.</p>
<p>{links}</p><p>Original read-only results: {originals}</p>
<h2>Saved seeds: A Final and B Priority are separate stages</h2><p>A is stored final sampling (127 Box nodes in 256 total); B is the saved 32-node Priority subset. Seed 0 B reuses Phase2; seed 1/2 B use the existing matcher, without FPS replay.</p>
<div class="scroll"><table><tr><th>Method</th><th>Seed</th><th>First flat</th><th>First XYZ (m)</th><th>B Priority</th><th>A Final</th></tr>{rows}</table></div>
<h2>First selected points</h2><a href="seed_initial_points.png"><img src="seed_initial_points.png" alt="First saved Priority point of each seed"></a>
<h2>Saved coverage by stage</h2><a href="seed_priority_final_coverage.png"><img src="seed_priority_final_coverage.png" alt="Separate bars for B Priority and A Final coverage"></a>
<h2>C: common initial point control</h2><p>Shared intersection: {result['initial_control']['shared_count']}; initial flat {result['initial_control']['initial_flat_index']}; XYZ {result['initial_control']['initial_xyz_m']}. Each method executes one existing FPS call, including the initial point in 32 unique nodes. C is Priority only; A and C are different sampling stages.</p>
<table><tr><th>Method</th><th>Nodes</th><th>Rim matching nodes</th><th>C Priority</th><th>Unique reference IDs</th></tr>{control_rows}</table>
<a href="shared_initial_overlay.png"><img src="shared_initial_overlay.png" alt="S2 and S3 2D Priority locations with the same initial point"></a>
<a href="shared_initial_coverage.png"><img src="shared_initial_coverage.png" alt="Control Priority coverage and unique rim reference positions"></a>
<p>{html.escape(result['interpretation'])}</p><p>CPU NumPy offline analysis only. Simulator, renderer, CUDA, other frames/episodes and initialization sweeps were not executed.</p></html>'''
    (output/'priority_fps_seed_control.html').write_text(body, encoding='utf-8')


def create_markdown(output, result, test_summary):
    rel = lambda path: os.path.relpath(path, REPORT.parent)
    saved, control = result['saved'], result['control']
    lines = [
        '# Task 03-D Phase 3 — FPS Initialization vs Candidate Distribution', '',
        '**상태: 완료 / 승인 범위 종료.**', '',
        result['interpretation_ko'], '',
        '## 실행 조건과 재사용 artifact', '',
        f"- 새 run: `{result['run_id']}`. 기존 ExperimentLogger / artifact manifest·sidecar 계약 재사용, `WANDB_MODE=disabled` 로컬 기록.",
        '- Frame 0, Box, S2(FD2) / S3(PCA5), num_points=256, structural_ratio=.25, threshold=20°, Box budget=127, Priority quota=32.',
        '- 동일 GT rim line reference 103개. 동일 point/reference pair에서 `2px AND 3mm`, reference ID 중복 제거. GT rim/negative ROI는 평가 전용이다.',
        f'- [Task03 samples]({rel(REPO/"visual/results/sampling"/SOURCE_RUN/"f0000_samples.npz")}), [frame metrics]({rel(REPO/"visual/results/sampling"/SOURCE_RUN/"f0000_metrics.json")}), [config]({rel(REPO/"visual/results/sampling"/SOURCE_RUN/"comparison.json")}) (`{SOURCE_RUN}`).',
        f'- [Phase1 JSON]({rel(REPO/"experiments"/PHASE1/"diagnostic.json")}) (`{PHASE1}`), [Phase2 JSON]({rel(REPO/"experiments"/PHASE2/"priority_fps_trace.json")}) (`{PHASE2}`). Seed 0 Priority ID/coverage 및 matched node는 Phase2를 그대로 사용하고 Phase1과 대조했다.',
        '- 저장 결과 분석은 FPS·candidate·normal·feature를 재실행하지 않았다. Source geometry와 기존 matcher/ROI만 재사용했다. Final coverage는 저장 metrics 값을 그대로 보존했다.', '',
        '## A. 저장된 Final / B. 저장된 Priority', '',
        '| 방법 | Seed | 첫 flat | 첫 XYZ (m) | B rim node /32 | B Priority refs /103 | A Final refs /103 | Negative union /32 |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for method, seeds in saved.items():
        for seed, row in seeds.items():
            xyz = ', '.join(f'{v:.9g}' for v in row['initial_xyz_m'])
            lines.append(f"| {method} | {seed} | {row['initial_flat_index']} | ({xyz}) | {row['rim_matching_node_count']} | {row['priority_reference_count']} ({100*row['priority_coverage']['value']:.2f}%) | {row['final_sampling']['numerator']} | {row['negative_roi']['known_negative_union']['count']} |")
    lines += ['', 'A는 최종 256점 중 Box 127점의 저장 coverage, B는 그중 실제 Priority 32점의 coverage다. B의 새 진단값으로 A를 바꾸지 않았다.', '',
        '| 방법 / Seed | Priority depth min / median / p95 / max (m) | Unique GT reference IDs (B) |', '| --- | --- | --- |']
    for method, seeds in saved.items():
        for seed, row in seeds.items():
            d = row['priority_depth_m']
            lines.append(f"| {method} / {seed} | {' / '.join(f'{d[k]:.9g}' for k in ('min','median','p95','max'))} | {row['priority_reference_ids']} |")
    initial = result['initial_control']
    lines += ['', '## C. 동일 초기점 offline control', '',
        f"후보 교집합은 {initial['shared_count']}개다. GT와 무관한 `shared = sorted(set(C_S2) & set(C_S3)); initial_flat = shared[len(shared)//2]` 규칙으로 **{initial['initial_flat_index']}**를 선택했다. XYZ는 `{initial['initial_xyz_m']}` m이며 S2/S3에서 exact equality다. Candidate-local index는 `{initial['candidate_local_index']}`다.", '',
        '- 각 후보 입력은 저장된 오름차순 raster 순서(S2 297개 / S3 599개)를 유지했다. 기존 `fps_indices(..., 32, seed=fps_seed(0, BOX), initial=[local])`를 방법당 정확히 1회 호출했다.',
        '- `seed`는 기존 SeedSequence 계약에 맞게 전달했으며 initial 경로에서 RNG 생성/최초 draw가 없음을 실행 중 차단 검증했다. 초기점 포함 32개의 고유 source index다.',
        '- 두 FPS가 끝난 후에만 control을 GT matcher 및 negative ROI로 평가했다. 전체 32-step trace는 생성하지 않았다.', '',
        '| 방법 | 실제 node | Rim matching node | C Priority refs /103 | Unique GT reference IDs | Negative union /32 | Depth min / median / p95 / max (m) |',
        '| --- | ---: | ---: | ---: | --- | ---: | --- |']
    for method, row in control.items():
        d = row['priority_depth_m']
        lines.append(f"| {method} | {row['node_count']} | {row['rim_matching_node_count']} | {row['priority_reference_count']} ({100*row['priority_coverage']['value']:.2f}%) | {row['priority_reference_ids']} | {row['negative_roi']['known_negative_union']['count']} | {' / '.join(f'{d[k]:.9g}' for k in ('min','median','p95','max'))} |")
    lines += ['', '공통 초기점 control의 실제 선택 flat index는 선택 순서대로 다음과 같다.', '']
    for method, row in control.items():
        lines += [f"- {method}: `{row['selected_flat_indices']}`"]
    lines += ['', 'Reference 집합 비교(C):', '']
    for name, row in result['control_reference_comparison'].items():
        lines += [f"- {name}: {row['count']}개, `{row['reference_ids']}`"]
    node_comparison = result['control_node_comparison']
    lines += ['', f"선택 source index 교집합은 {node_comparison['intersection']['count']}개 "
        f"(`{node_comparison['intersection']['flat_indices']}`), S2-only / S3-only는 각각 "
        f"{node_comparison['S2_only']['count']} / {node_comparison['S3_only']['count']}개다. "
        '각 선택 XYZ 및 candidate-local index와 전체 차집합은 아래 JSON에 보존했다.', '',
        '## 해석과 연구적 판단', '', result['interpretation_ko'], '',
        '- 저장 seed의 S3 Priority 변동 폭은 2 references(1.94%p), S2는 6 references(5.83%p)다. S3의 덮인 ID 집합도 seed에 따라 달라진다. 이는 관측된 초기점 민감성의 범위이며 초기점의 독립적인 인과 효과를 정량화한 값은 아니다.',
        '- 같은 seed끼리의 B S2−S3 차이는 seed 0/1/2에서 10/2/4 references다. C의 차이 5는 seed 0보다 작고 seed 1/2보다 크다. 따라서 seed 0의 큰 차이 전부가 지속되는 것은 아니지만, 이 공통 초기점에서도 S3의 낮은 coverage는 남는다.',
        '- 관측된 seed별 B Priority 범위 및 같은 초기점 C를 함께 읽어야 한다. A와 C는 sampling 단계가 다르므로 동일 조건 성능 비교로 사용하지 않는다.',
        '- 한 공통 초기점 결과를 전체 seed·episode에 일반화하지 않는다. 후보 수·위치·오염·원거리 depth의 독립적인 인과 기여율은 분해하지 않았다. Phase2의 기존 경로 분석을 재사용하며 새 trace를 만들지 않았다.',
        '- 다음 연구 판단은 초기점 민감성과 후보 집합 효과를 함께 검토하는 것이다. 새 sampling 정책, 후보 제거, depth cutoff, GT quota, FPS 변형의 채택 근거로 사용하지 않는다. 추가 초기점/episode 검증은 별도 승인 대상이다.', '',
        '## 구현·검증·source hash', '',
        '- 추가 구현: [run_priority_fps_seed_control.py](../../visual/runtime/test/run_priority_fps_seed_control.py), [test_priority_fps_seed_control.py](../../visual/runtime/test/test_priority_fps_seed_control.py). 기존 production·Phase1/2·run_sampling 파일은 보존했다.',
        '- Phase1 source geometry/matcher helper와 Phase2 `verify_fps_source` 검증 경로를 재사용했다. 새 sampler나 logger/publisher 구현은 추가하지 않았다.',
        f'- **PASS**: 신규 및 관련 기존 테스트. `{test_summary}`. [test log]({rel(output/"tests.log")}). 테스트의 FPS/전체 sampler 반복은 합성 입력에만 적용했다.',
        '- **PASS**: evaluated source와 현재 FPS·candidate·allocation 정의 일치, seed0 Phase1/2 reference ID 및 저장 순서 일치, seed1/2 source/feature/role alignment, 23 metadata row 및 seed 간 공통 source feature equality.',
        '- **PASS**: 저장 분석 FPS 호출 차단, 후보 교집합 252 / initial 114957, initial XYZ exact equality, 초기 RNG draw 차단, 32 unique indices·초기점 포함·실제 후보 membership·local/global 대응·raster 순서 유지.',
        '- **PASS**: 합성 GT annotation 변경 시 선택 불변, 기존 same-pair joint matching 및 reference 중복 제거, 저장 negative ROI union 점유 일치, 원본 artifact·입력·기존 파일 hash 불변, HTML/Markdown 상대 링크와 PNG 생성 확인.',
        '- **FAIL**: 없음. 불일치 시 assertion·tolerance·GT·seed 규칙을 변경하지 않고 중단하도록 구현했다.',
        '- **SKIP / 미검증**: simulator·renderer·CUDA/GPU backend, 다른 frame/episode, 추가 공통 초기점 sweep, PCL·Graph·BC·정책/학습 실험, 외부 upload·dependency 설치·commit. 현재 호스트에서는 CPU NumPy offline 진단만 실행했다.',
        f'- Git 시작/종료 및 보존한 기존 변경: [start]({rel(output/"git_start.json")}), [end]({rel(output/"git_end.json")}), [diff]({rel(output/"implementation.patch")}). 시작 Git dirty 상태이므로 commit만으로 재현할 수 없으며 source snapshot과 hash를 함께 보존했다.', '',
        '| Source / input | SHA-256 |', '| --- | --- |']
    for path, digest in result['source_hashes'].items():
        lines.append(f'| `{path}` | `{digest}` |')
    lines += ['', '위 표는 실제 진단 실행 시점의 source hash다. 전체 입력/evaluated source hash와 원본 보존 기록은 아래 JSON에 있다. '
              '그림·보고서만 저장 결과로 갱신한 경우 실제 실행 source는 `execution_source/`, 최종 source는 `diagnostic_source/` 및 '
              '`source_hashes.json`의 `report_generation_sources`로 구분한다. FPS 결과는 재실행하지 않는다.', '', '## 산출물', '']
    for filename in ('priority_fps_seed_control.html', 'priority_fps_seed_control.json', 'seed_initial_points.png',
        'seed_priority_final_coverage.png', 'shared_initial_overlay.png', 'shared_initial_coverage.png',
        'source_hashes.json', 'verification.json', 'tests.log', 'implementation.patch'):
        lines.append(f'- [{filename}]({rel(output/filename)})')
    lines += ['', '**Phase 3 결과 제출 후 중단한다. 후속 변경은 별도 승인 후 진행한다.**', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    shutil.copyfile(REPORT, output/'report_snapshot.md')


def diagnose(source, output_root, baseline_path):
    if os.environ.get('WANDB_MODE') != 'disabled':
        raise ValueError('set WANDB_MODE=disabled for local-only execution')
    if REPORT.exists():
        raise FileExistsError('refuse to overwrite Phase3 report')
    baseline = json.loads(Path(baseline_path).read_text())
    verify_preserved(baseline)
    source = Path(source).resolve()
    comparison = json.loads((source/'comparison.json').read_text())
    require_equal(comparison['run_id'], SOURCE_RUN, 'Task03 run')
    config = comparison['config']
    source_checks = source_contract(source, config)
    fps_checks = verify_fps_source(source, config['source_sha256']['visual/runtime/sampling.py'])
    phase1_path = REPO/'experiments'/PHASE1/'diagnostic.json'
    phase2_path = REPO/'experiments'/PHASE2/'priority_fps_trace.json'
    phase1, phase2 = (json.loads(p.read_text()) for p in (phase1_path, phase2_path))
    for obj, run, phase in ((phase1, PHASE1, 1), (phase2, PHASE2, 2)):
        require_equal(obj['run_id'], run, f'Phase{phase} run')
        require_equal(obj['source_run_id'], SOURCE_RUN, f'Phase{phase} Task03 source')
    require_equal(phase1['inputs_unchanged'], True, 'Phase1 inputs unchanged')
    for field in ('saved_reference_order_and_geometry_exact', 'target_coverage_exact',
                  'saved_negative_roi_counts_exact'):
        require_equal(phase1['frames'][0][field], True, 'Phase1 Frame0 '+field)
    require_equal(phase2['pass_all'], True, 'Phase2 pass_all')
    for relative, expected in phase2['config']['source_sha256'].items():
        require_equal(digests(REPO/relative)[1], expected, 'Phase2 diagnostic source '+relative)
    episode, masks = Path(config['input']), Path(config['masks'])
    inputs = [source/'comparison.json', source/'f0000_samples.npz', source/'f0000_metrics.json',
              episode, masks, episode.with_name('segmentation_id_map.json'), masks.with_name('tracking.json'),
              phase1_path, phase2_path]
    for name, expected in phase2['config']['input_sha256'].items():
        require_equal(digests(Path(name))[1], expected, 'Phase2 input '+name)
    before = hashes(inputs)
    current_sources = [Path(__file__), Path(__file__).with_name('test_priority_fps_seed_control.py'),
        Path(__file__).with_name('run_priority_fps_trace.py'), Path(__file__).with_name('run_rim_bottleneck.py'),
        Path(__file__).with_name('sampling_evaluation.py'), REPO/'visual/runtime/sampling.py',
        REPO/'visual/runtime/geometry.py', REPO/'visual/runtime/rgbd.py']
    source_hashes = hashes(current_sources)
    run_config = dict(task='03-D', phase=3, source_run_id=SOURCE_RUN, phase1_run_id=PHASE1,
        phase2_run_id=PHASE2, frame=0, methods=list(METHODS), num_points=256,
        structural_ratio=.25, orientation_threshold_deg=20, box_budget=127, priority_count=32,
        matching_px_m=[2., .003], reference_count=103, input_sha256=before, source_sha256=source_hashes,
        real_fps_calls_per_method=1, additional_initial_sweeps=0, simulator_executed=False,
        gpu_executed=False, device='cpu', full_sampling_executed=False,
        candidate_normal_feature_recalculation=False, python=platform.python_version(), numpy=np.__version__)
    with ExperimentLogger(team='visual', config=run_config, seed=0,
        model='task03d_priority_fps_seed_control', job_type='evaluate', root=output_root) as logger:
        output = logger.directory
        write_json(output/'git_start.json', baseline)
        shutil.copyfile(Path(baseline_path).with_name('git_start.diff'), output/'git_start.diff')
        # Tests precede real control, so any failing contract stops the experiment.
        with (output/'tests.log').open('w') as log:
            test = subprocess.run([sys.executable, '-m', 'unittest', *TEST_MODULES, '-v'],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False)
        logger.link_output(output/'tests.log', kind='reports')
        if test.returncode:
            write_json(output/'verification.json', dict(status='FAIL', reason='required tests failed', real_fps_calls=0))
            logger.link_output(output/'verification.json', kind='reports')
            raise RuntimeError('required tests failed; see tests.log; no real FPS executed')
        test_lines = (output/'tests.log').read_text().splitlines()
        test_summary = next(line for line in test_lines if line.startswith('Ran '))+'; '+test_lines[-1]
        frame, geometry, gt = _frame_and_context(comparison, episode, masks)
        metrics = json.loads((source/'f0000_metrics.json').read_text())
        with np.load(source/'f0000_samples.npz', allow_pickle=False) as archive, forbid_sampling():
            candidates, selections, alignment = load_saved(archive, config, metrics, frame, geometry, phase2)
            context = evaluation_context(frame, geometry_evidence(frame), gt, config['camera'], config['robot_ids'])
            refs = tuple(archive[f'evaluation/box/reference_{suffix}'] for suffix in ('uv', 'xyz'))
            require_equal(len(refs[0]), 103, 'GT reference count')
            for a, b in zip(refs, context['references']['box']):
                np.testing.assert_array_equal(a, b)
            matcher = CoverageMatcher(geometry, BOX, refs)
            saved = summarize_saved(selections, phase1, phase2, matcher, frame, context['negatives']['box'])
            for method, seeds in saved.items():
                for seed, row in seeds.items():
                    key = f'{method}_n256_r0.25_t20_s{seed}'
                    for name, roi in row['negative_roi'].items():
                        require_equal(roi['count'], metrics[key]['objects']['box']['negative'][name]['priority'], key+' ROI '+name)
        # No GT/ROI/matcher parameter crosses this selection boundary.
        with patch(__name__+'.fps_indices', wraps=fps_indices) as calls:
            orders, initial = select_control(candidates)
        require_equal(calls.call_count, 2, 'total real FPS call count')
        # Only now may selected control points enter the matcher/negative ROIs.
        with forbid_sampling():
            control = score_control(orders, matcher, frame, context['negatives']['box'])
        require_equal(hashes(inputs), before, 'input hashes after evaluation')
        preservation = verify_preserved(baseline)
        difference = control['S2']['priority_reference_count']-control['S3']['priority_reference_count']
        ranges = {m: [min(r['priority_reference_count'] for r in seeds.values()),
                      max(r['priority_reference_count'] for r in seeds.values())] for m, seeds in saved.items()}
        interpretation = (f"Saved Priority ranges: S2 {ranges['S2']}, S3 {ranges['S3']} references of 103. "
            f"At shared initial 114957, S2 covers {control['S2']['priority_reference_count']}/103 and S3 "
            f"{control['S3']['priority_reference_count']}/103 (S2 minus S3: {difference}). "
            'This single initialization constrains interpretation to the fixed candidate sets and FPS conditions; '
            'candidate count, position, contamination and far depth have not been causally separated.')
        interpretation_ko = (f"저장된 seed 0/1/2의 B Priority coverage는 S2 {ranges['S2'][0]}–{ranges['S2'][1]}/103, "
            f"S3 {ranges['S3'][0]}–{ranges['S3'][1]}/103이다. 공통 초기점 114957의 C Priority는 "
            f"S2 **{control['S2']['priority_reference_count']}/103**, S3 **{control['S3']['priority_reference_count']}/103** "
            f"(S2−S3 {difference} refs)이다. "
            + ('이 초기점과 기존 FPS 조건에서도 후보 집합 차이에 따른 결과 차이가 남는다. '
               if difference > 0 else '저장 seed에서 관측한 차이가 이 공통 초기점에서는 축소되거나 역전되어 초기점 민감성의 근거가 된다. ')
            + '단일 공통 초기점으로 초기점 효과의 일반적 크기나 각 후보 특성의 독립적 인과 기여율을 추정하지 않는다.')
        result = dict(run_id=logger.run_id, source_run_id=SOURCE_RUN, config=run_config,
            source_contract=source_checks, fps_source_check=fps_checks, source_hashes=source_hashes,
            saved=saved, saved_alignment=alignment, initial_control=initial, control=control,
            control_reference_comparison=compare_ids(*(control[m]['priority_reference_ids'] for m in METHODS)),
            control_node_comparison={name: dict(count=row['count'], flat_indices=row['reference_ids'])
                for name, row in compare_ids(*(control[m]['selected_flat_indices'] for m in METHODS)).items()},
            interpretation=interpretation, interpretation_ko=interpretation_ko,
            input_hashes_unchanged=True, pass_all=True)
        write_json(output/'priority_fps_seed_control.json', result)
        write_json(output/'source_hashes.json', dict(inputs_before=before, inputs_after=hashes(inputs),
            diagnostic_sources=source_hashes, evaluated_fps=fps_checks, original_preservation=preservation))
        create_plots(output, frame, saved, control, refs[0])
        create_markdown(output, result, test_summary)
        create_html(output, result, source)
        for path in current_sources:
            dest = output/'diagnostic_source'/path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            logger.link_output(dest, kind='reports')
        diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=REPO)
        for path in (Path(__file__), Path(__file__).with_name('test_priority_fps_seed_control.py'), REPORT):
            delta = subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', str(path.relative_to(REPO))],
                cwd=REPO, stdout=subprocess.PIPE, check=False)
            if delta.returncode != 1:
                raise ValueError('cannot preserve added-file diff')
            diff += delta.stdout
        (output/'implementation.patch').write_bytes(diff)
        write_json(output/'git_end.json', dict(git_status=subprocess.check_output(['git', 'status', '--short'], cwd=REPO).decode(),
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip(),
            original_preservation=verify_preserved(baseline)))
        verification = dict(status='PASS', required_tests=dict(status='PASS', summary=test_summary, modules=TEST_MODULES),
            source_and_input_contract='PASS', seed0_phase2_ids_reused='PASS', saved_source_feature_role_alignment='PASS',
            saved_analysis_fps_calls=0, shared_candidate_count=252, shared_initial=114957,
            initial_xyz_exact='PASS', initial_rng_draws=0, real_fps_calls=initial['real_fps_calls'],
            unique_count_includes_initial=COUNT, raster_local_global_membership='PASS',
            gt_annotation_independence='PASS (synthetic tests; no GT argument in selection)',
            same_pair_joint_matching_and_reference_dedup='PASS', negative_roi_union='PASS',
            inputs_unchanged='PASS', original_preservation=preservation,
            skipped=['simulator', 'renderer', 'GPU/CUDA', 'other frame/episode', 'initialization sweep',
                     'new sampling policy', 'external upload', 'dependency installation', 'commit'])
        write_json(output/'verification.json', verification)
        # Check all local href/src targets without browser/network dependencies.
        from html.parser import HTMLParser
        class Links(HTMLParser):
            def handle_starttag(self, tag, attrs):
                for attr, value in attrs:
                    if attr in ('href', 'src') and value and not (output/value).is_file():
                        raise ValueError('broken local HTML link: '+value)
        Links().feed((output/'priority_fps_seed_control.html').read_text())
        markdown_links = re.findall(r'\]\(([^)]+)\)', REPORT.read_text())
        for target in markdown_links:
            if not (REPORT.parent/target).is_file():
                raise ValueError('broken local Markdown link: '+target)
        for png in output.glob('*.png'):
            pixels = plt.imread(png)
            if pixels.ndim != 3 or min(pixels.shape[:2]) < 400:
                raise ValueError('invalid report PNG: '+str(png))
        verification['html_and_markdown_local_links'] = dict(status='PASS', markdown_link_count=len(markdown_links))
        verification['png_readback'] = 'PASS (4 PNGs, dimensions and pixel decode)'
        write_json(output/'verification.json', verification)
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name not in ('metadata.json', 'manifest.json', 'config.yaml', 'metrics.json') and not path.name.endswith('.tracking.json'):
                logger.link_output(path, kind='metrics' if path.name in ('priority_fps_seed_control.json','source_hashes.json') else 'reports')
        logger.log({m: {'saved_priority_seed'+s: r['priority_reference_count'] for s, r in seeds.items()}
                    for m, seeds in saved.items()}, step=0, prefix='eval/task03d/saved')
        logger.log({m: dict(priority_references=r['priority_reference_count'], rim_matching_nodes=r['rim_matching_node_count'])
                    for m, r in control.items()}, step=0, prefix='eval/task03d/control')
        logger.record(source_run_id=SOURCE_RUN, phase1_run_id=PHASE1, phase2_run_id=PHASE2,
                      real_fps_calls=initial['real_fps_calls'], inputs_unchanged=True,
                      simulator_executed=False, gpu_executed=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO/'visual/results/sampling'/SOURCE_RUN)
    parser.add_argument('--out', type=Path, default=REPO/'experiments')
    parser.add_argument('--baseline', type=Path, required=True, help='pre-change workspace/artifact SHA256 and git status snapshot')
    args = parser.parse_args()
    print(diagnose(args.source, args.out, args.baseline))


if __name__ == '__main__':
    main()
