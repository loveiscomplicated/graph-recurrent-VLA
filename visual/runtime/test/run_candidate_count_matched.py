"""Task03-H: saved Frame 0 Box candidates; no detection, normals or FPS.

WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_candidate_count_matched \
    --baseline /tmp/task03h_start.json
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from operator import index
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
import scipy
from scipy.sparse import csr_matrix

from utils.experiment_artifacts import digests
from utils.experiment_logger import ExperimentLogger
from visual.runtime.geometry import OrganizedGeometry
from visual.runtime.rgbd import BOX, PEG, iter_saved_rgbd
from visual.runtime.test.run_rim_bottleneck import occupancy, require_equal, source_contract, write_json
from visual.runtime.test.run_structural_edges import negative_regions
from visual.runtime.test.sampling_evaluation import CoverageMatcher

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO/'visual/results/sampling/2551448b0fae4c6bbce3caf48774275f'
DENSE = REPO/'visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a'
EDGES = REPO/'visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0000_pairs.npz'
PHASE1 = REPO/'experiments/d3f14ee494894daabb8ba2fab8e1b00c'
PHASE2 = REPO/'experiments/aa5d71078e0841fd886b5026d6030ebc'
PHASE3 = REPO/'experiments/f36f1e9e2bb64866b8d6c792db158635'
TASK03E = REPO/'experiments/7f657cf88baf4f1f89e594252e243e1c'
REPORT = REPO/'docs/visual-graph-integration/03h_candidate_count_matched.md'
PLAN = REPORT.with_name('03h_candidate_count_matched_plan.md')
STEM = 'candidate_count_matched'
K = 297
SEEDS = tuple(range(20261024, 20261040))
STREAMS = {'pca5_matched': 0, 'observed_random': 1}
FIGURES = ('candidate_original.png', 'candidate_matched.png', 'reference_multiplicity.png',
           'expected_coverage.png', 'roi_depth_spatial.png')
# Explicit existing contracts; omit tests that run a sampler, sweep or GT generation.
EXISTING_TESTS = [
    'SamplingEvaluationTests.test_joint_tolerance_requires_same_pair',
    'SamplingEvaluationTests.test_tracking_and_sampling_losses_have_separate_denominators',
    'SamplingEvaluationTests.test_empty_reference_is_undefined_not_perfect',
    'StagedCoverageTests.test_reference_overlap_union_and_surface_outside_candidates',
    'StagedCoverageTests.test_two_different_nodes_cannot_satisfy_one_reference_pair',
    'StagedCoverageTests.test_empty_points_zero_observation_and_zero_visible',
    'StagedCoverageTests.test_invalid_nodes_and_non_partition_are_rejected',
    'StagedCoverageTests.test_negative_union_deduplicates_and_depth_bands_cover_paths',
    'StagedCoverageTests.test_historical_evaluation_changes_fail_closed',
]


def sha(path):
    return digests(Path(path))[1]


def hash_files(paths):
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(set(map(Path, paths)))}


def array_sha(a):
    a = np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def uniform_rows(n, k, seed, stream):
    """Selection sees ONLY population size, count and RNG identity, never GT/XYZ."""
    n, k, seed, stream = map(index, (n, k, seed, stream))
    if not 0 <= k <= n or min(seed, stream) < 0:
        raise ValueError('invalid uniform subset arguments')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, stream])))
    return np.sort(rng.choice(n, size=k, replace=False))


def coverage_probability(n, h, k):
    """Exact hypergeometric coverage formula, evaluated without huge binomials."""
    n, h, k = map(index, (n, h, k))
    if not 0 <= h <= n or not 0 <= k <= n:
        raise ValueError('require 0 <= h,k <= N')
    if h == 0 or k == 0:
        return 0.
    if k > n-h:
        return 1.
    count, excluded = min(k, h), max(k, h)
    logq = math.fsum(math.log1p(-excluded/(n-i)) for i in range(count))
    return -math.expm1(logq)


def probabilities(n, multiplicity, k):
    return np.array([coverage_probability(n, int(h), k) for h in multiplicity], np.float64)


def checked_rows(support, selected):
    """Validate exact source membership before searchsorted indexing."""
    support, selected = np.asarray(support), np.asarray(selected)
    if support.ndim != 1 or selected.ndim != 1 or selected.dtype.kind not in 'iu':
        raise ValueError('one-dimensional integer source indices required')
    if len(support) and not np.all(support[:-1] < support[1:]):
        raise ValueError('support must be strictly sorted')
    if len(np.unique(selected)) != len(selected):
        raise ValueError('duplicate source indices')
    rows = np.searchsorted(support, selected)
    if (rows >= len(support)).any() or not np.array_equal(support[rows], selected):
        raise ValueError('source indices outside population')
    return rows


def describe(adj):
    adj = adj.astype(bool).tocsr(copy=True)
    adj.sum_duplicates(); adj.eliminate_zeros()
    h = adj.getnnz(axis=0)
    degree = adj.getnnz(axis=1)
    covered = h > 0
    n, nref = adj.shape
    return dict(node_count=n, matching_node_count=int((degree > 0).sum()),
                rim_matching_node_fraction=float((degree > 0).mean()) if n else None,
                unique_reference_count=int(covered.sum()), reference_denominator=nref,
                reference_coverage=float(covered.mean()) if nref else None,
                pair_count=int(adj.nnz), reference_multiplicity=h.tolist(),
                node_reference_degree=degree.tolist(), covered_reference_ids=np.flatnonzero(covered).tolist(),
                missing_reference_ids=np.flatnonzero(~covered).tolist(),
                multiplicity_histogram={label: int(mask.sum()) for label, mask in
                    [('zero', h == 0), ('one', h == 1), ('two_or_more', h >= 2)]})


def sets(first, second):
    a, b = set(map(int, first)), set(map(int, second))
    return {k: dict(count=len(v), ids=sorted(v)) for k, v in
            [('intersection', a & b), ('fd2_only', a-b), ('pca5_only', b-a)]}


def summary(values):
    a = np.asarray(values, float)
    return dict(mean=float(a.mean()), sample_sd=float(a.std(ddof=1)), min=float(a.min()), max=float(a.max()))


def expectation(adj, k, labels, roi_masks, flat):
    row = describe(adj); n = row['node_count']
    p = probabilities(n, row['reference_multiplicity'], k)
    return dict(population_count=n, subset_count=k, reference_denominator=adj.shape[1],
                expected_unique_reference_count=math.fsum(p), expected_reference_coverage=math.fsum(p)/adj.shape[1],
                reference_coverage_probability=p.tolist(),
                expected_reference_multiplicity=(k*np.asarray(row['reference_multiplicity'])/n).tolist(),
                expected_matching_node_count=k*row['matching_node_count']/n,
                expected_rim_matching_node_fraction=row['rim_matching_node_fraction'],
                segments={str(s): dict(reference_count=int((labels == s).sum()),
                    expected_unique_reference_count=math.fsum(p[labels == s]),
                    expected_reference_coverage=math.fsum(p[labels == s])/adj.shape[1]) for s in np.unique(labels)},
                roi={name: dict(expected_node_count=k*int(mask.ravel()[flat].sum())/n,
                     expected_node_fraction=float(mask.ravel()[flat].mean())) for name, mask in roi_masks.items()})


def load_geometry():
    comparison = json.loads((SOURCE/'comparison.json').read_text())
    config = comparison['config']
    require_equal(comparison['run_id'], SOURCE.name, 'source UUID')
    require_equal(config['camera'], 'base_camera', 'camera')
    require_equal(Path(config['input']).name, 'ep000.npz', 'episode')
    for field in ('input', 'masks'):
        require_equal(sha(config[field]), config[field+'_sha256'], field+' SHA256')
    prior = json.loads((DENSE/'geometry.json').read_text())['config']
    for field in ('input_sha256', 'masks_sha256', 'camera'):
        require_equal(prior[field], config[field], 'Task01/03 '+field)
    contracts = source_contract(SOURCE, config)
    frame = next(iter_saved_rgbd(config['input'], config['masks'], camera=config['camera']))
    require_equal(frame.source_frame, 0, 'source frame')
    with np.load(DENSE/'f0000.npz', allow_pickle=False) as a:
        require_equal(int(a['source_frame']), 0, 'dense source frame')
        require_equal(str(a['camera']), frame.camera, 'dense camera')
        for key in ('intrinsic', 'cam2world'):
            np.testing.assert_array_equal(a[key], getattr(frame, key))
        geometry = OrganizedGeometry(a['xyz'], a['valid_depth'],
            {oid: a[name+'/object_mask'] for oid, name in [(PEG, 'peg'), (BOX, 'box')]}, a['overlap_mask'])
        for oid in (PEG, BOX):
            np.testing.assert_array_equal(geometry.object_masks[oid], frame.masks[oid])
        np.testing.assert_array_equal(a['box/valid_points'], geometry.observed_mask(BOX))
        np.testing.assert_array_equal(geometry.valid_depth, np.isfinite(frame.depth) & (frame.depth > 0))
        np.testing.assert_array_equal(geometry.overlap_mask, frame.masks[PEG] & frame.masks[BOX])
        np.testing.assert_array_equal(geometry.xyz[..., 2][geometry.valid_depth], frame.depth[geometry.valid_depth])
    populations = {}
    with np.load(TASK03E/'priority_fps_metric_control.npz', allow_pickle=False) as e, np.load(SOURCE/'f0000_samples.npz', allow_pickle=False) as a:
        require_equal(int(a['source_frame']), 0, 'candidate source frame')
        np.testing.assert_array_equal(a['shape'], frame.depth.shape)
        support = np.flatnonzero(geometry.observed_mask(BOX))
        require_equal(len(support), 11665, 'observed count')
        for method, estimator, count in [('S2', 'fd2', 297), ('S3', 'pca5', 599)]:
            prefix = method+'/box/'
            flat = e[prefix+'candidate_flat']
            require_equal(len(flat), count, estimator+' count')
            checked_rows(support, flat)
            np.testing.assert_array_equal(flat, a[f'evidence/{estimator}/1/20/orientation_flat'])
            np.testing.assert_array_equal(e[prefix+'candidate_xyz'], geometry.xyz.reshape(-1, 3)[flat])
            np.testing.assert_array_equal(e[prefix+'support_flat'], support)
            np.testing.assert_array_equal(e[prefix+'support_xyz'], geometry.xyz.reshape(-1, 3)[support])
            populations[estimator] = flat.copy()
    populations['observed'] = support
    return frame, geometry, populations, config, contracts


def annotate(frame, geometry, populations):
    """Called only after the fixed draws have been persisted."""
    with np.load(SOURCE/'f0000_samples.npz', allow_pickle=False) as a, np.load(TASK03E/'priority_fps_metric_control.npz', allow_pickle=False) as e, np.load(EDGES, allow_pickle=False) as edges:
        ref = tuple(a['evaluation/box/reference_'+s] for s in ('uv', 'xyz'))
        for suffix, value in zip(('uv', 'xyz'), ref):
            np.testing.assert_array_equal(value, e['box/reference_'+suffix])
            np.testing.assert_array_equal(value, edges['evaluation/box/rim_'+suffix])
        require_equal((ref[0].shape, ref[1].shape), ((103, 2), (103, 3)), 'reference shapes')
        labels = e['box/reference_segment'].copy()
        matcher = CoverageMatcher(geometry, BOX, ref, tolerances=((2., .003),))
        np.testing.assert_array_equal(matcher.flat, populations['observed'])
        full = matcher.adjacencies['2px_3mm']
        np.testing.assert_array_equal(full.getnnz(axis=0) > 0, e['box/observable'])
        matrices = {name: full[checked_rows(matcher.flat, flat)] for name, flat in populations.items()}
        for method, estimator in [('S2', 'fd2'), ('S3', 'pca5')]:
            prefix = method+'/box/1.0/C/'
            adj = matrices[estimator]
            saved = csr_matrix((np.ones(len(e[prefix+'joint_node_row']), bool),
                (e[prefix+'joint_node_row'], e[prefix+'joint_reference_id'])), shape=adj.shape)
            require_equal((adj != saved).nnz, 0, estimator+' saved joint pairs')
            np.testing.assert_array_equal(adj.getnnz(axis=0) > 0, e[prefix+'covered'])
            np.testing.assert_array_equal(adj.getnnz(axis=1) > 0, e[prefix+'matched_node'])
            np.testing.assert_array_equal(adj.getnnz(axis=1), e[prefix+'reference_count_per_node'])
        regions = {k.split('/')[-1]: edges[k] for k in edges.files
                   if k.startswith('evaluation/box/') and edges[k].shape == frame.depth.shape}
        all_ref = tuple(np.concatenate([ref[i], edges['evaluation/box/seam_'+s]]) for i, s in enumerate(('uv', 'xyz')))
        negatives = negative_regions(regions, {}, all_ref, frame.depth.shape)
        roi_masks = dict(negatives, known_negative_union=np.logical_or.reduce(list(negatives.values())))
        # Fixed GT rim mask crop, before evaluating subset outcomes.
        v, u = np.nonzero(regions['hole_rim'])
        crop = (max(0, int(u.min())-12), min(frame.depth.shape[1]-1, int(u.max())+12),
                min(frame.depth.shape[0]-1, int(v.max())+12), max(0, int(v.min())-12))
    original = {name: describe(adj) for name, adj in matrices.items()}
    for name, count, matching, refs in [('observed', 11665, 153, 103), ('fd2', 297, 101, 57), ('pca5', 599, 127, 91)]:
        row = original[name]
        require_equal((row['node_count'], row['matching_node_count'], row['unique_reference_count']),
                      (count, matching, refs), name+' original evidence')
        ids, nodes = matcher.reference_hits(populations[name])
        require_equal((len(ids), len(nodes)), (refs, matching), name+' helper agreement')
    differences = dict(source_indices=sets(populations['fd2'], populations['pca5']),
                       references=sets(original['fd2']['covered_reference_ids'], original['pca5']['covered_reference_ids']))
    require_equal([differences['references'][k]['count'] for k in ('intersection', 'fd2_only', 'pca5_only')], [56, 1, 35], 'reference set relation')
    phase = json.loads((PHASE1/'diagnostic.json').read_text())['frames'][0]
    require_equal(phase['source_frame'], 0, 'Phase1 frame')
    for new, old in [('intersection', 'intersection'), ('fd2_only', 'S2_only'), ('pca5_only', 'S3_only')]:
        require_equal(differences['references'][new]['ids'], phase['S2_vs_S3']['C'][old]['reference_ids'], 'Phase1 reference IDs')
    occ = occupancy(frame, dict(X=populations['observed'], fd2=populations['fd2'], pca5=populations['pca5']), negatives)
    prior = json.loads((TASK03E/'priority_fps_metric_control.json').read_text())
    for method, name in [('S2', 'fd2'), ('S3', 'pca5')]:
        old = prior['cases'][method+'/box']
        require_equal(array_sha(populations[name]), old['candidate_sha256'], name+' ordered hash')
        require_equal(array_sha(populations['observed']), old['support_sha256'], name+' support hash')
        old = old['conditions']['1.0']
        require_equal(occ['depth_m'][name], old['occupancy']['depth_m']['C'], name+' saved depth')
        for roi in roi_masks:
            require_equal(occ['negative_roi'][roi][name], old['occupancy']['negative_roi'][roi]['C'], name+' saved ROI '+roi)
        for s in np.unique(labels):
            ids = np.flatnonzero((labels == s) & (np.asarray(original[name]['reference_multiplicity']) > 0)).tolist()
            require_equal(ids, old['stages']['C']['segments'][str(s)]['covered_reference_ids'], name+' saved segment '+str(s))
    return matcher, matrices, original, differences, ref, labels, negatives, roi_masks, crop


def evaluate_set(flat, matcher, labels, frame, negatives, arrays, prefix):
    rows = checked_rows(matcher.flat, flat)
    adj = matcher.adjacencies['2px_3mm'][rows]
    row = describe(adj)
    occ = occupancy(frame, dict(X=matcher.flat, selected=flat), negatives)
    row['occupancy'] = dict(negative_roi={k: v['selected'] for k, v in occ['negative_roi'].items()},
        depth_m=occ['depth_m']['selected'], depth_quantile_occupancy=occ['depth_quantile_occupancy']['selected'],
        depth_band_bounds_m=occ['depth_band_bounds_m'])
    row['segments'] = {str(s): dict(reference_count=int((labels == s).sum()),
        covered_reference_ids=np.flatnonzero((labels == s) & (np.asarray(row['reference_multiplicity']) > 0)).tolist(),
        missing_reference_ids=np.flatnonzero((labels == s) & (np.asarray(row['reference_multiplicity']) == 0)).tolist(),
        coverage_count=int(((labels == s) & (np.asarray(row['reference_multiplicity']) > 0)).sum()),
        coverage_denominator=103) for s in np.unique(labels)}
    coo = adj.tocoo()
    arrays.update({prefix+'/flat_index': flat, prefix+'/xyz': matcher.xyz[rows], prefix+'/uv': matcher.uv[rows],
        prefix+'/joint_node_row': coo.row, prefix+'/joint_reference_id': coo.col,
        prefix+'/reference_multiplicity': np.asarray(row['reference_multiplicity']),
        prefix+'/matched_node': adj.getnnz(axis=1) > 0,
        prefix+'/covered': adj.getnnz(axis=0) > 0})
    return row


def repeated_summary(rows):
    out = {key: summary([r[key] for r in rows]) for key in
           ('matching_node_count', 'rim_matching_node_fraction', 'unique_reference_count', 'reference_coverage')}
    out['roi'] = {roi: {key: summary([r['occupancy']['negative_roi'][roi][key] if key == 'count'
                 else r['occupancy']['negative_roi'][roi]['fraction_of_path']['value'] for r in rows])
                 for key in ('count', 'fraction')} for roi in rows[0]['occupancy']['negative_roi']}
    out['depth_m'] = {key: summary([r['occupancy']['depth_m'][key] for r in rows]) for key in ('min', 'median', 'p95', 'max')}
    out['segments'] = {s: summary([r['segments'][s]['coverage_count'] for r in rows]) for s in rows[0]['segments']}
    out['reference_multiplicity_mean'] = np.mean([r['reference_multiplicity'] for r in rows], axis=0).tolist()
    out['reference_coverage_frequency'] = np.mean([np.asarray(r['reference_multiplicity']) > 0 for r in rows], axis=0).tolist()
    return out


def save_figure(out, filename, fig):
    fig.savefig(out/filename, dpi=170, bbox_inches='tight'); plt.close(fig)


def plots(out, frame, matcher, populations, draws, original, expected, labels, negatives, roi_masks, crop, arrays):
    def overlay(ax, flat, title, extent):
        rows = checked_rows(matcher.flat, flat)
        adj = matcher.adjacencies['2px_3mm'][rows]
        hit = adj.getnnz(axis=1) > 0; covered = adj.getnnz(axis=0) > 0
        uv = matcher.uv[rows]; refs = matcher.ref_uv
        ax.imshow(frame.rgb)
        ax.scatter(*refs[~covered].T, s=25, marker='x', color='#ffdb6e', linewidths=.8, label='missing ref', zorder=3)
        ax.scatter(*refs[covered].T, s=22, facecolors='none', edgecolors='#58ef9b', linewidths=.8, label='covered ref', zorder=3)
        ax.scatter(*uv[~hit].T, s=9, color='#ff6347', alpha=.55, linewidths=0, label='candidate', zorder=4)
        ax.scatter(*uv[hit].T, s=9, color='#43b5ff', alpha=.55, linewidths=0, label='rim matching node', zorder=4)
        ax.set(xlim=extent[:2], ylim=extent[2:], title=title, xlabel='source u (px)', ylabel='source v (px)')
    u, v = matcher.uv.T
    full = (max(0, int(u.min())-12), min(511, int(u.max())+12), min(511, int(v.max())+12), max(0, int(v.min())-12))
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for col, name in enumerate(('fd2', 'pca5')):
        r = original[name]
        for row, extent in enumerate((full, crop)):
            overlay(axes[row, col], populations[name], f'{name.upper()} original | {len(populations[name])} nodes | {r["unique_reference_count"]}/103 refs', extent)
    axes[0, 0].legend(fontsize=8); fig.tight_layout(); save_figure(out, FIGURES[0], fig)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    choices = [('FD2 original', populations['fd2']), ('PCA5 matched r=0', draws['pca5_matched'][0]), ('Observed random r=0', draws['observed_random'][0])]
    for col, (title, flat) in enumerate(choices):
        r = describe(matcher.adjacencies['2px_3mm'][checked_rows(matcher.flat, flat)])
        for row, extent in enumerate((full, crop)):
            overlay(axes[row, col], flat, f'{title} | 297 nodes | {r["unique_reference_count"]}/103 refs', extent)
    axes[0, 0].legend(fontsize=8); fig.tight_layout(); save_figure(out, FIGURES[1], fig)
    # Physical order is established by the saved line label and varying XYZ axis.
    order = []
    for label in sorted(np.unique(labels)):
        ids = np.flatnonzero(labels == label)
        axis = int(np.argmax(np.ptp(matcher.ref_xyz[ids], axis=0)))
        order.extend(ids[np.argsort(matcher.ref_xyz[ids, axis], kind='stable')])
    order = np.asarray(order); arrays['reference_plot_order'] = order
    h = np.array([original[n]['reference_multiplicity'] for n in ('fd2', 'pca5')])
    empirical = [describe(matcher.adjacencies['2px_3mm'][checked_rows(matcher.flat, flat)])['reference_multiplicity'] for _, flat in choices[1:]]
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), constrained_layout=True)
    im = axes[0].imshow(np.vstack([h, empirical])[:, order], aspect='auto', interpolation='nearest', cmap='magma', vmin=0)
    axes[0].set_yticks(range(4), ['FD2 original h', 'PCA5 original h', 'PCA5 matched r=0 h', 'Observed r=0 h']); fig.colorbar(im, ax=axes[0], label='actual matching node multiplicity')
    p = np.array([expected[n]['reference_coverage_probability'] for n in ('pca5_matched', 'observed_random')])
    im = axes[1].imshow(p[:, order], aspect='auto', interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
    axes[1].set_yticks(range(2), ['PCA5 k=297 exact P(covered)', 'Observed k=297 exact P(covered)']); fig.colorbar(im, ax=axes[1], label='coverage probability')
    for name in ('fd2', 'pca5'):
        axes[2].plot(np.arange(103), np.asarray(original[name]['reference_multiplicity'])[order], label=name)
    for ax in axes:
        offset = 0
        for label in sorted(np.unique(labels)):
            count = int((labels == label).sum())
            ax.axvline(offset-.5, color='gray', alpha=.5)
            offset += count
        ax.set_xlabel('references grouped by saved rim line, sorted along verified varying XYZ axis')
    axes[2].legend(); axes[0].set_title('Zero = missing; candidate nodes and reference counts are different quantities')
    save_figure(out, FIGURES[2], fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name in ('fd2', 'pca5', 'observed'):
        x, y = arrays[name+'/curve_k'], arrays[name+'/curve_expected_unique']
        axes[0].plot(x, y, label=name); axes[1].plot(x[1:], np.diff(y), label=name)
    axes[0].axhline(57, color='gray', ls=':', label='FD2 original 57/103')
    for ax in axes:
        ax.axvline(K, color='black', ls='--', alpha=.5); ax.legend(); ax.set_xlabel('uniform subset count k (analytic, no sampling sweep)')
    axes[0].set(ylabel='expected unique references (denominator 103)', ylim=(0, 103))
    axes[1].set(ylabel='expected additional references per additional node')
    save_figure(out, FIGURES[3], fig)
    names = ['fd2', 'pca5', 'pca5_matched', 'observed_random']
    flats = [populations['fd2'], populations['pca5'], draws['pca5_matched'][0], draws['observed_random'][0]]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    union = roi_masks['known_negative_union']
    axes[0, 0].imshow(frame.rgb); axes[0, 0].imshow(np.ma.masked_where(~union, union), cmap='Reds', alpha=.4, vmin=0, vmax=1)
    axes[0, 0].set(xlim=full[:2], ylim=full[2:], title='Existing negative ROI union (not an error label)')
    roi_names = list(roi_masks); positions = np.arange(len(roi_names))
    for i, (name, flat) in enumerate(zip(names, flats)):
        axes[0, 1].bar(positions+(i-1.5)*.19, [roi_masks[r].ravel()[flat].mean() for r in roi_names], width=.19, label=name)
        z = frame.depth.ravel()[flat]
        axes[0, 2].step(np.sort(z), np.arange(1, len(z)+1)/len(z), where='post', label=name)
        xyz = matcher.xyz[checked_rows(matcher.flat, flat)]
        axes[1, 1].scatter(xyz[:, 0], xyz[:, 1], s=8, alpha=.4, label=name)
        axes[1, 2].scatter(xyz[:, 0], xyz[:, 2], s=8, alpha=.4, label=name)
    axes[0, 1].set_xticks(positions, roi_names, rotation=65, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('fraction of nodes; r=0 for random conditions')
    axes[0, 2].set(xlabel='positive camera depth (m)', ylabel='ECDF', title='Far positive depth is not a contamination truth')
    median, p95 = np.percentile(frame.depth.ravel()[matcher.flat], [50, 95])
    for j, (name, flat) in enumerate(zip(names, flats)):
        z = frame.depth.ravel()[flat]; counts = [np.mean(z <= median), np.mean((z > median) & (z <= p95)), np.mean(z > p95)]
        axes[1, 0].bar(np.arange(3)+(j-1.5)*.19, counts, width=.19, label=name)
    axes[1, 0].set_xticks(range(3), ['z <= X median', 'X median < z <= X p95', 'z > X p95'], rotation=15)
    axes[1, 0].set_ylabel('fraction of nodes; fixed observed-depth bands')
    axes[1, 1].set(xlabel='camera X (m)', ylabel='camera Y (m)', title='XY distribution')
    axes[1, 2].set(xlabel='camera X (m)', ylabel='camera Z (m)', title='XZ distribution')
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=7)
    save_figure(out, FIGURES[4], fig)


def table(result):
    lines = ['| 조건 | 후보 node 수 | rim matching node | rim matching fraction | unique rim reference /103 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in ('fd2', 'pca5'):
        r = result['original'][name]
        lines.append(f'| {name} original | {r["node_count"]} | {r["matching_node_count"]} | {r["rim_matching_node_fraction"]:.2%} | {r["unique_reference_count"]}/103 |')
    for name in STREAMS:
        r = result['expected'][name]
        lines.append(f'| {name} exact expectation | 297 | {r["expected_matching_node_count"]:.4f} | {r["expected_rim_matching_node_fraction"]:.2%} | {r["expected_unique_reference_count"]:.6f}/103 |')
    return lines


def write_reports(out, result):
    p = result['expected']['pca5_matched']; o = result['expected']['observed_random']
    delta = p['expected_unique_reference_count']-57
    conclusion = ('동일 후보 수에서도 PCA5의 기대 rim coverage가 FD2보다 높다. 후보 수 증가만으로 원래 coverage 차이를 설명할 수 없다.' if delta > 0 else
                  '동일 후보 수에서 PCA5의 기대 rim coverage 우위가 사라진다. 원래 높은 coverage는 후보 수에 상당히 의존한다.')
    result['conclusion'] = conclusion
    def link(name): return os.path.relpath(out/name, REPORT.parent)
    lines = ['# Task03-H — Candidate Count-Matched Structural Evidence Analysis', '',
        f'실행 UUID: `{out.name}`. {conclusion}', '',
        f'Primary: PCA5 297개 정확한 expected unique reference는 **{p["expected_unique_reference_count"]:.6f}/103 ({p["expected_reference_coverage"]:.2%})**, FD2 original은 **57/103**이며 차이는 **{delta:+.6f} reference**다.', '',
        *table(result), '',
        '정확한 기대값 행의 matching node/reference 수는 비정수 기대값이다. 실제 subset의 정수 count 및 16회 표본 평균과 구분한다. Rim matching fraction은 이 특정 rim에 대한 후보 집중도이며 일반적인 detector precision이 아니다.', '',
        '## 고정 조건과 선택 독립성', '',
        'ep000 Frame 0 Box, base_camera, 동일 관측 11,665개, 저장 FD2/PCA5 orientation-only 20° 후보 297/599개, 같은 103개 reference, 동일 pair의 ≤2px AND ≤3mm다. 원본 source index·XYZ·reference를 그대로 사용했으며 normal/candidate/backprojection/FPS를 재실행하지 않았다.', '',
        'GT·ROI·score·XYZ를 받지 않는 uniform_rows(N,k,seed,stream)으로 비복원 선택했다. PCG64 + SeedSequence([seed,stream]), seed 20261024–20261039, PCA5 stream=0/observed stream=1, 조건별 16회, 대표 overlay는 사전 고정 r=0이다. 선택 배열은 GT matching annotation 전에 저장했다. 전체 관측 control은 candidate/normal-valid filtering을 하지 않았다.', '',
        '## 후보 수 효과와 공간 분포', '',
        f'PCA5 original 91개와 FD2 original 57개의 차이는 34 reference다. 같은 PCA5 모집단을 297개로 줄일 때 기대 coverage 감소는 **{91-p["expected_unique_reference_count"]:.6f} reference**이고, 줄인 뒤 FD2와의 차이는 **{delta:+.6f} reference**다. 이는 고정 모집단에서의 기술적 비교이며 독립적인 인과 기여율이나 estimator 효과의 비율 분해가 아니다.', '',
        f'Observed random 297개의 기대값은 **{o["expected_unique_reference_count"]:.6f}/103**, 기대 matching node는 **{o["expected_matching_node_count"]:.4f}/297**다. 두 estimator의 rim 집중도와 임의 관측점 선택의 차이를 보는 대조군이다.', '',
        'Expected unique coverage는 reference별 multiplicity h에 대한 1−C(N−h,k)/C(N,k)를 합산했다. 짧은 유한곱의 log1p, math.fsum 및 −expm1을 사용했다. Reference 간 독립성을 가정하지 않는다. 정확한 분산은 계산하지 않았고, 아래 변동은 고정 후보 집합의 부분추출 변동이다. Curve는 고정 후보의 analytic rarefaction으로 추가 실험/threshold sweep이 아니다.', '',
        '| 조건 | 16회 실제 unique refs mean | sample SD | min–max | matching nodes mean |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, r in result['repeated_summary'].items():
        u = r['unique_reference_count']
        lines.append(f'| {name} | {u["mean"]:.4f} | {u["sample_sd"]:.4f} | {u["min"]:.0f}–{u["max"]:.0f} | {r["matching_node_count"]["mean"]:.4f} |')
    lines += ['', '## 중복·누락과 reference 집합', '',
        '| 모집단 | h=0 refs | h=1 refs | h≥2 refs | joint pairs | matching nodes |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in ('fd2', 'pca5', 'observed'):
        r = result['original'][name]; h = r['multiplicity_histogram']
        lines.append(f'| {name} | {h["zero"]} | {h["one"]} | {h["two_or_more"]} | {r["pair_count"]} | {r["matching_node_count"]} |')
    lines += ['', '한 node가 여러 reference를 덮을 수 있고 여러 node가 같은 reference에 집중될 수 있다. Pair 수를 node 수나 unique reference 수로 읽지 않는다. h가 높은 reference는 작은 k에서 포화되고, h=0인 위치는 원래 후보 전체를 사용해도 누락된다.', '',
        f'Original reference 교집합 56개, FD2-only 1개, PCA5-only 35개를 ID까지 재확인했다. Candidate source index 집합 관계는 {result["set_comparison"]["source_indices"] ["intersection"]["count"]}개 공통, FD2-only {result["set_comparison"]["source_indices"]["fd2_only"]["count"]}개, PCA5-only {result["set_comparison"]["source_indices"]["pca5_only"]["count"]}개다. 두 종류의 집합 관계는 서로 다르다.', '',
        '| 기존 rim 구간 | 구간 reference 수 | FD2 original covered | PCA5 original covered | PCA5 297 expected covered | Observed 297 expected covered |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for s in sorted(result['original']['fd2']['segments']):
        a, b = (result['original'][n]['segments'][s] for n in ('fd2', 'pca5'))
        lines.append(f'| {s} | {a["reference_count"]} | {a["coverage_count"]}/103 | {b["coverage_count"]}/103 | {p["segments"][s]["expected_unique_reference_count"]:.4f}/103 | {o["segments"][s]["expected_unique_reference_count"]:.4f}/103 |')
    lines += ['', '구간 크기는 별도 count이며 모든 coverage 비율의 분모는 103이다. 원래 56/1/35 reference 그룹별 기대 보존과 16회 실제 보존, reference 103행별 multiplicity/확률/누락 ID는 JSON/NPZ에 보존했다.', '',
        '## 기존 ROI와 depth', '',
        '| 조건 | negative union nodes / 전체 | depth median m | depth p95 m | max m |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in ('fd2', 'pca5'):
        r = result['original'][name]['occupancy']; z = r['depth_m']
        lines.append(f'| {name} original | {r["negative_roi"]["known_negative_union"]["count"]}/{result["original"][name]["node_count"]} | {z["median"]:.4f} | {z["p95"]:.4f} | {z["max"]:.4f} |')
    for name in STREAMS:
        r = result['repeated_summary'][name]; z = r['depth_m']
        lines.append(f'| {name} 16回 mean | {r["roi"]["known_negative_union"]["count"]["mean"]:.4f}/297 | {z["median"]["mean"]:.4f} | {z["p95"]["mean"]:.4f} | {z["max"]["mean"]:.4f} |')
    lines += ['', '個別 ROI는 중첩되므로 합계와 union을 구분한다. ROI 점유는 오검출 정답이 아니며, 원거리 positive depth 자체도 오염 정답이 아니다. X의 median/p95로 고정한 depth bands, 개별 ROI count/fraction과 camera XY/XZ/source UV 분포를 별도 저장·시각화했다.', '',
        '## 확인된 사실과 미확인 범위', '', conclusion,
        '이 결과는 고정 Frame 0 Box에서 후보 수를 통제한 rim coverage와 후보 위치·중복 분포의 근거다. PCA5의 모든 구조/형상에 대한 우월성, 모든 edge의 precision, 추가 후보가 noise인지 여부, 독립 episode 일반화, Graph/GNN/BC 효용은 검증하지 않았다. 후속 sampling 정책이나 Voxel 설계는 채택하지 않는다.', '',
        '## 검증·재현성·산출물', '',
        '작은 N 전수열거·exact rational 대조, same-pair tolerance·중복/집합·GT 독립성 계약 테스트를 통과했다. 입력 support/camera/XYZ와 저장 pair, 원래 103/57/91 coverage 및 56/1/35 집합을 재확인했다. 기존 source/input/artifact/sidecar와 초기 dirty 파일의 SHA-256 불변을 확인했다. 상세 PASS와 테스트 목록은 verification.json/tests.log를 따른다.', '',
        'ExperimentLogger(team=visual, job_type=evaluate), WANDB_MODE=disabled와 기존 로컬 artifact 등록을 재사용했다. UUID/project/group 규칙은 유지했다. Git dirty 상태는 source snapshot/input hashes/implementation diff와 함께 보존했다. CPU NumPy/SciPy offline 분석이며 simulator·renderer·CUDA/MPS/GPU, 외부 업로드·dependency 설치·Git commit은 실행하지 않았다.', '',
        f'[승인된 계획](03h_candidate_count_matched_plan.md) · [정적 HTML]({link(STEM+".html")}) · [결과 JSON]({link(STEM+".json")}) · [결과 NPZ]({link(STEM+".npz")}) · [검증]({link("verification.json")}) · [테스트]({link("tests.log")}) · [hash]({link("source_hashes.json")}) · [implementation diff]({link("implementation.patch")})', '']
    for filename in FIGURES: lines.append(f'![{filename}]({link(filename)})\n')
    lines.append('구현·진단·검증·보고를 완료했으며 추가 알고리즘 변경 없이 중단한다.\n')
    REPORT.write_text('\n'.join(lines).replace('個別', '개별'), encoding='utf-8')
    table_html = '<table><tr><th>Condition</th><th>Nodes</th><th>Rim matching nodes</th><th>Rim fraction</th><th>Unique refs /103</th></tr>'
    for line in table(result)[2:]:
        table_html += '<tr>'+''.join('<td>'+html.escape(v.strip())+'</td>' for v in line.strip('|').split('|'))+'</tr>'
    table_html += '</table>'
    groups = result['reference_groups']
    group_html = '<table><tr><th>Original reference group</th><th>Refs</th><th>PCA5 k=297 expected retained</th></tr>'
    for name, group in groups.items(): group_html += f'<tr><td>{name}</td><td>{group["count"]}</td><td>{group["pca5_expected_retained"]:.6f}</td></tr>'
    group_html += '</table>'
    links = [(STEM+'.json', 'Result JSON'), (STEM+'.npz', 'Source/subsets/pairs NPZ'), ('verification.json', 'Verification'),
             ('tests.log', 'Tests'), ('source_hashes.json', 'Source/input hashes'), ('implementation.patch', 'Implementation diff'),
             (os.path.relpath(REPORT, out), 'Korean Markdown report')]
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task03-H candidate count matching</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#172235;background:#f8fafc}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #cbd5e1;text-align:left}img{max-width:100%;height:auto;background:white}a{color:#0759b0}section{margin:32px 0}.note{padding:16px;background:#e5eefb}code{overflow-wrap:anywhere}</style><body>'
    page += f'<h1>Task03-H — Candidate count-matched analysis</h1><p>Run <code>{out.name}</code> · ep000 / Frame 0 / Box · orientation-only 20° · same-pair ≤2px AND ≤3mm</p><p class="note">{html.escape(conclusion)}<br>Primary exact expectation: {p["expected_unique_reference_count"]:.6f}/103 vs FD2 57/103 (Δ {delta:+.6f} references).</p>'+table_html
    page += '<p>Expectation rows are analytic, not sampled means. Rim matching fraction describes concentration near this specific rim; it is not general edge precision. GT/ROI are evaluation annotations only.</p><h2>Original coverage groups</h2>'+group_html
    page += '<p>PCG64, SeedSequence([20261024+r,stream]), r=0…15; streams PCA5=0, observed=1. All overlays use preselected r=0. Sixteen-repeat SD/range describes subset variability, not generalization uncertainty.</p>'
    page += '<table><tr><th>16 fixed repeats</th><th>Actual unique refs mean</th><th>Sample SD</th><th>Min–max</th><th>Matching nodes mean</th><th>ROI union nodes mean</th></tr>'
    for name, r in result['repeated_summary'].items():
        u = r['unique_reference_count']
        page += f'<tr><td>{name}</td><td>{u["mean"]:.4f}/103</td><td>{u["sample_sd"]:.4f}</td><td>{u["min"]:.0f}–{u["max"]:.0f}</td><td>{r["matching_node_count"]["mean"]:.4f}/297</td><td>{r["roi"]["known_negative_union"]["count"]["mean"]:.4f}/297</td></tr>'
    page += '</table>'
    for filename, title in zip(FIGURES, ['Original candidate locations: full Box and common rim crop', 'Equal 297 nodes: FD2 / PCA5 matched / observed random', 'Reference multiplicity, missing positions and coverage probability', 'Exact expected coverage saturation; no threshold sweep', 'Existing ROI union, individual ROI occupancy, depth and spatial distribution']):
        page += f'<section><h2>{title}</h2><a href="{filename}"><img src="{filename}" alt="{title}"></a></section>'
    page += '<p>Overlapping ROI occupancy is not a false-edge label. Far positive depth is not contamination truth. Node counts, matching nodes, pair counts and unique references are distinct. All reference coverage ratios use the same 103 references. Single-frame Box evidence does not establish all-shape detector superiority or Graph/GNN/BC utility.</p><h2>Artifacts and reproducibility</h2><ul>'
    page += ''.join(f'<li><a href="{html.escape(path)}">{title}</a></li>' for path, title in links)+'</ul><pre id="browser-check">image load pending</pre>'
    page += '<script>window.addEventListener("load",()=>{document.getElementById("browser-check").textContent=JSON.stringify(Array.from(document.images).map(i=>({src:i.getAttribute("src"),complete:i.complete,width:i.naturalWidth,height:i.naturalHeight})))})</script></body></html>'
    (out/(STEM+'.html')).write_text(page, encoding='utf-8')


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.targets = []
    def handle_starttag(self, tag, attrs):
        self.targets += [v for k, v in attrs if k in ('href', 'src')]


def verify_links(out):
    parser = Links(); parser.feed((out/(STEM+'.html')).read_text())
    for path in parser.targets:
        if '://' in path or Path(path).is_absolute(): raise ValueError('non-relative HTML link')
        if not (out/path).is_file(): raise ValueError('missing HTML target: '+path)
    targets = re.findall(r'\]\(([^)]+)\)', REPORT.read_text())
    for path in targets:
        if not (REPORT.parent/path).is_file(): raise ValueError('missing Markdown target: '+path)
    for name in FIGURES:
        a = plt.imread(out/name)
        if min(a.shape[:2]) == 0 or not np.isfinite(a).all(): raise ValueError('invalid PNG '+name)
    return dict(html_relative_targets=len(parser.targets), markdown_relative_targets=len(targets), png_decodes=len(FIGURES))


def verify_browser(out):
    chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    if not chrome.is_file(): return dict(status='not executed', reason='Chrome unavailable')
    command = [str(chrome), '--headless', '--disable-gpu', '--no-sandbox', '--allow-file-access-from-files',
               '--user-data-dir='+str(out/'chrome_profile'), '--dump-dom', (out/(STEM+'.html')).as_uri()]
    process = subprocess.run(command, capture_output=True, text=True, timeout=45)
    require_equal(process.returncode, 0, 'headless browser return code')
    match = re.search(r'<pre id="browser-check">(.*?)</pre>', process.stdout, re.S)
    if match is None: raise ValueError('browser image check absent')
    rows = json.loads(html.unescape(match.group(1)))
    require_equal(len(rows), len(FIGURES), 'browser image count')
    if not all(r['complete'] and r['width'] > 0 and r['height'] > 0 for r in rows): raise ValueError('browser image load failed')
    shutil.rmtree(out/'chrome_profile')
    return dict(status='PASS', images=rows)


def preserved(hashes):
    require_equal(hash_files([REPO/p for p in hashes]), hashes, 'protected files SHA256')
    return dict(status='PASS', protected_file_count=len(hashes), all_sha256_unchanged=True)


def diagnose(baseline_path):
    if os.environ.get('WANDB_MODE') != 'disabled': raise ValueError('WANDB_MODE=disabled required')
    if REPORT.exists(): raise FileExistsError('refuse to overwrite result Markdown')
    baseline = json.loads(Path(baseline_path).read_text())
    preserved(baseline['hashes'])
    c = json.loads((SOURCE/'comparison.json').read_text())['config']
    protected = [Path(c[k]) for k in ('input', 'masks')]
    for root in (SOURCE, DENSE, PHASE1, PHASE2, PHASE3, TASK03E, EDGES.parent):
        # Only designated evidence directories, not a repository search.
        protected += [p for p in root.rglob('*') if p.is_file()]
    source_paths = [Path(module.__file__).resolve() for module in list(sys.modules.values())
                    if getattr(module, '__file__', None) and str(module.__file__).endswith('.py')
                    and Path(module.__file__).resolve().is_relative_to(REPO)
                    and not Path(module.__file__).resolve().is_relative_to(REPO/'.venv')]
    source_paths += [REPO/'visual/runtime/test/test_candidate_count_matched.py',
                     REPO/'visual/runtime/test/test_sampling_evaluation.py', PLAN]
    input_hashes = hash_files(protected)
    source_hashes = hash_files(source_paths)
    protected_hashes = {**input_hashes, **source_hashes, **baseline['hashes']}
    config = dict(task='03-H', source_run_id=SOURCE.name, source_geometry_run_id=TASK03E.name,
        frame=0, episode='ep000', camera='base_camera', object='box', orientation_only=True, threshold_deg=20,
        observed_count=11665, candidate_counts=dict(fd2=297, pca5=599), reference_count=103,
        tolerances_px_m=[[2., .003]], k=K, seeds=list(SEEDS), streams=STREAMS, repeats=16,
        rng='PCG64(SeedSequence([seed,stream])); sorted choice(N,k,replace=False)',
        gt_selection_input=False, representative_repeat=0, analytic_curves=True,
        input_sha256=input_hashes, source_sha256=source_hashes,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        device='cpu', simulator_executed=False, gpu_executed=False)
    with ExperimentLogger(team='visual', config=config, seed=SEEDS[0], model='task03h_candidate_count_matched', job_type='evaluate') as logger:
        out = logger.directory
        try:
            write_json(out/'git_start.json', baseline)
            preflight_note = Path(baseline_path).with_name('task03h_preflight_failure.json')
            if preflight_note.is_file(): shutil.copyfile(preflight_note, out/preflight_note.name)
            (out/'git_start.diff').write_text(baseline['git_diff'])
            for p in source_paths:
                dest = out/'source_snapshot'/p.relative_to(REPO); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
            additions = [REPO/'visual/runtime/test/run_candidate_count_matched.py', REPO/'visual/runtime/test/test_candidate_count_matched.py']
            diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=REPO, text=True)
            for p in additions:
                proc = subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', str(p.relative_to(REPO))], cwd=REPO, capture_output=True, text=True)
                if proc.returncode not in (0, 1): raise RuntimeError('implementation diff failed')
                diff += proc.stdout
            (out/'implementation.patch').write_text(diff)
            tests = ['visual.runtime.test.test_candidate_count_matched'] + ['visual.runtime.test.test_sampling_evaluation.'+t for t in EXISTING_TESTS]
            with (out/'tests.log').open('w') as log:
                proc = subprocess.run([sys.executable, '-m', 'unittest', *tests, '-v'], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            require_equal(proc.returncode, 0, 'required contract tests')
            # Guard actual data path: any accidental regeneration or FPS aborts.
            with ExitStack() as stack:
                for module in list(sys.modules.values()):
                    if getattr(module, '__name__', '').startswith('visual.'):
                        for name in ('fps_indices', 'sample_points', 'extract_sampling_evidence', 'build_organized_geometry', 'rim_reference'):
                            if callable(getattr(module, name, None)):
                                stack.enter_context(patch.object(module, name, side_effect=AssertionError('forbidden actual-data call: '+name)))
                frame, geometry, populations, old_config, contracts = load_geometry()
                arrays = {}
                draws, draw_records = {}, {}
                for condition, population in [('pca5_matched', populations['pca5']), ('observed_random', populations['observed'])]:
                    draws[condition] = []; draw_records[condition] = []
                    for repeat, seed in enumerate(SEEDS):
                        rows = uniform_rows(len(population), K, seed, STREAMS[condition]); flat = population[rows]
                        require_equal((len(flat), len(np.unique(flat))), (K, K), 'unique subset count')
                        checked_rows(population, flat)
                        draws[condition].append(flat)
                        arrays[f'{condition}/r{repeat:02d}/draw_rows'] = rows
                        arrays[f'{condition}/r{repeat:02d}/flat_index'] = flat
                        draw_records[condition].append(dict(repeat=repeat, seed=seed, stream=STREAMS[condition], source_sha256=array_sha(flat),
                            population_sha256=array_sha(population), selected_source_indices=flat.tolist()))
                # Persist draws before matcher, reference/ROI access or outcome evaluation.
                np.savez_compressed(out/'selection_before_annotation.npz', **arrays)
                write_json(out/'selection_before_annotation.json', draw_records)
                matcher, matrices, original, differences, ref, labels, negatives, roi_masks, crop = annotate(frame, geometry, populations)
                result = dict(run_id=logger.run_id, source_run_id=SOURCE.name, config=config, source_contract=contracts,
                    set_comparison=differences, original={}, expected={}, repeats={}, repeated_summary={},
                    representative_repeat=0, fixed_crop=list(crop), reference_denominator=103)
                arrays.update(reference_uv=ref[0], reference_xyz=ref[1], reference_segment=labels)
                for name, flat in populations.items():
                    result['original'][name] = evaluate_set(flat, matcher, labels, frame, negatives, arrays, name+'/original')
                    max_k = len(flat) if name != 'observed' else K
                    curve = np.array([math.fsum(probabilities(len(flat), original[name]['reference_multiplicity'], k)) for k in range(max_k+1)])
                    if (np.diff(curve) < -1e-12).any(): raise ValueError('nonmonotone expectation')
                    arrays[name+'/curve_k'] = np.arange(max_k+1); arrays[name+'/curve_expected_unique'] = curve
                    arrays[name+'/curve_marginal_gain'] = np.diff(curve)
                require_equal(float(arrays['fd2/curve_expected_unique'][-1]), 57., 'FD2 full expectation')
                require_equal(float(arrays['pca5/curve_expected_unique'][-1]), 91., 'PCA5 full expectation')
                for condition, population_name in [('pca5_matched', 'pca5'), ('observed_random', 'observed')]:
                    exp = expectation(matrices[population_name], K, labels, roi_masks, populations[population_name])
                    result['expected'][condition] = exp
                    arrays[condition+'/expected_reference_coverage_probability'] = np.asarray(exp['reference_coverage_probability'])
                    arrays[condition+'/expected_reference_multiplicity'] = np.asarray(exp['expected_reference_multiplicity'])
                    rows = []
                    for repeat, flat in enumerate(draws[condition]):
                        r = evaluate_set(flat, matcher, labels, frame, negatives, arrays, f'{condition}/r{repeat:02d}')
                        if not set(r['covered_reference_ids']) <= set(original[population_name]['covered_reference_ids']): raise ValueError('subset covers outside population')
                        np.testing.assert_array_equal(arrays[f'{condition}/r{repeat:02d}/xyz'], geometry.xyz.reshape(-1, 3)[flat])
                        r.update(draw_records[condition][repeat]); rows.append(r)
                    result['repeats'][condition] = rows; result['repeated_summary'][condition] = repeated_summary(rows)
                p = result['expected']['pca5_matched']['expected_unique_reference_count']; o = result['expected']['observed_random']['expected_unique_reference_count']
                result['comparisons'] = dict(original_pca5_minus_fd2=34, pca5_original_minus_matched_expected=91-p,
                    pca5_matched_expected_minus_fd2=p-57, fd2_minus_observed_expected=57-o, pca5_matched_minus_observed_expected=p-o,
                    pca5_original_minus_observed_expected=91-o)
                result['reference_groups'] = {}
                for group, entry in differences['references'].items():
                    ids = entry['ids']
                    result['reference_groups'][group] = dict(count=len(ids), ids=ids,
                        pca5_expected_retained=math.fsum(np.asarray(result['expected']['pca5_matched']['reference_coverage_probability'])[ids]),
                        observed_expected_retained=math.fsum(np.asarray(result['expected']['observed_random']['reference_coverage_probability'])[ids]),
                        retained_per_repeat={condition: [len(set(ids) & set(r['covered_reference_ids'])) for r in result['repeats'][condition]] for condition in STREAMS})
                plots(out, frame, matcher, populations, draws, result['original'], result['expected'], labels, negatives, roi_masks, crop, arrays)
            np.savez_compressed(out/(STEM+'.npz'), **arrays)
            with np.load(out/(STEM+'.npz'), allow_pickle=False) as saved:
                require_equal(set(saved.files), set(arrays), 'NPZ keys')
                for key, value in arrays.items():
                    np.testing.assert_array_equal(saved[key], value, err_msg=key)
                    require_equal(saved[key].dtype, value.dtype, 'NPZ dtype '+key)
            result['tests_summary'] = next(line for line in (out/'tests.log').read_text().splitlines() if line.startswith('Ran '))+'; OK'
            verification = dict(status='PASS', original_evidence_exact=True, same_pair_joint_tolerance=True,
                draw_count=32, all_draws_unique_k297_and_original_membership=True, selections_saved_before_gt_annotation=True,
                gt_blind_selector_signature=['n', 'k', 'seed', 'stream'], original_reference_relation=[56, 1, 35],
                exact_formula_exhaustive_synthetic_tests=True, npz_readback_exact=True,
                forbidden_actual_data_calls_guarded=True, preservation=preserved(protected_hashes), tests=result['tests_summary'],
                simulator_executed=False, renderer_executed=False, gpu_executed=False)
            write_json(out/'verification.json', verification)
            write_json(out/'source_hashes.json', dict(inputs_before=input_hashes, sources_before=source_hashes,
                protected_before=protected_hashes, protected_after=hash_files([REPO/p for p in protected_hashes])))
            write_reports(out, result)
            write_json(out/(STEM+'.json'), result)
            verification['links'] = verify_links(out)
            verification['browser'] = verify_browser(out)
            verification['preservation'] = preserved(protected_hashes)
            write_json(out/'verification.json', verification)
            write_json(out/'git_end.json', dict(status=subprocess.check_output(['git', 'status', '--short'], cwd=REPO, text=True),
                initial_dirty_files_preserved=preserved(baseline['hashes'])))
            logger.record(source_run_id=SOURCE.name, source_geometry_run_id=TASK03E.name,
                simulator_executed=False, gpu_executed=False, preserved=True)
            logger.log(dict(pca5_matched_expected=p, fd2_original=57, observed_random_expected=o, primary_delta=p-57))
            for path in sorted(out.iterdir()):
                if path.is_file() and path.name not in ('config.yaml', 'metadata.json', 'metrics.json', 'manifest.json'):
                    logger.link_output(path, kind='trajectory' if path.suffix == '.npz' else 'reports')
            # The new Markdown is the only registered output outside this UUID.
            logger.link_output(REPORT, kind='reports')
            print(json.dumps(dict(run_id=logger.run_id, expected_pca5=p, expected_observed=o, primary_delta=p-57,
                tests=result['tests_summary'], verification='PASS', directory=str(out)), indent=2))
        except Exception as exc:
            write_json(out/'verification.json', dict(status='FAIL', exception_type=type(exc).__name__, reason=str(exc),
                stopped_without_source_or_tolerance_adjustment=True))
            logger.link_output(out/'verification.json', kind='reports')
            raise
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.baseline)


if __name__ == '__main__': main()
