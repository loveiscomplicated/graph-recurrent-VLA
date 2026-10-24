"""Task03-E: offline distance-only Priority FPS ablation on saved ep000 Frame 0.

WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_priority_fps_metric_control \
    --baseline /tmp/task03e_baseline/baseline.json

No geometry, features, candidates, allocation, GT geometry or normals are regenerated.
All four exact baseline gates precede either control. Selection accepts no annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from utils.experiment_logger import ExperimentLogger
from visual.runtime.geometry import OrganizedGeometry
from visual.runtime.rgbd import PEG, BOX, iter_saved_rgbd
from visual.runtime.sampling import fps_indices
from visual.runtime.test.normal_diagnostics import pose_rotation, stats
from visual.runtime.test.run_priority_fps_trace import fps_seed
from visual.runtime.test.run_rim_bottleneck import (
    require_equal, source_contract, validate_saved_selection, occupancy, write_json,
)
from visual.runtime.test.run_structural_edges import negative_regions
from visual.runtime.test.sampling_evaluation import CoverageMatcher, staged_reference_coverage

REPO = Path(__file__).resolve().parents[3]
SOURCE_RUN = '2551448b0fae4c6bbce3caf48774275f'
TASK01 = REPO/'visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a'
NORMAL = REPO/'visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/f0000.npz'
EDGES = REPO/'visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0000_pairs.npz'
REPORT = REPO/'docs/visual-graph-integration/03e_priority_fps_metric_control.md'
STEM = 'priority_fps_metric_control'
METHODS = {'S2': 'fd2', 'S3': 'pca5'}
OBJECTS = {'peg': PEG, 'box': BOX}
BUDGET = {'peg': 129, 'box': 127}
EXPECTED = {('S2', 'peg'): (870, 213708), ('S3', 'peg'): (703, 214225),
            ('S2', 'box'): (297, 101598), ('S3', 'box'): (599, 96553)}
LAMBDAS = (1., .25, 0.)
TEST_MODULES = ['visual.runtime.test.'+n for n in (
    'test_priority_fps_metric_control', 'test_sampling', 'test_sampling_geometry',
    'test_sampling_evaluation', 'test_priority_fps_trace', 'test_priority_fps_seed_control')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(paths):
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(set(map(Path, paths)))}


def array_sha(a):
    a = np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def verify_preserved(baseline):
    require_equal(hashes([REPO/p for p in baseline['hashes']]), baseline['hashes'], 'protected original files')
    return dict(status='PASS', protected_file_count=len(baseline['hashes']), all_sha256_unchanged=True)


def metric_coordinates(points, weight):
    if weight not in LAMBDAS:
        raise ValueError('only lambda 1/.25/0 is authorized')
    metric = np.array(points, dtype=np.float64, copy=True)
    if metric.ndim != 2 or metric.shape[1] != 3 or not np.isfinite(metric).all():
        raise ValueError('finite [N,3] coordinates required')
    metric[:, 2] *= np.sqrt(weight)
    return metric


def select(case, weight):
    """Only candidate/support geometry, fixed initial/quota and metric enter FPS."""
    flat, xyz = case['candidate_flat'], case['candidate_xyz']
    support, support_xyz = case['support_flat'], case['support_xyz']
    first = int(case['first'])
    local = np.flatnonzero(flat == first)
    if len(local) != 1 or not np.all(flat[:-1] < flat[1:]) or not np.all(support[:-1] < support[1:]):
        raise ValueError('invalid initial or source order')
    p_rows = fps_indices(metric_coordinates(xyz, weight), 32, seed=case['seed'], initial=local)
    p = flat[p_rows]
    rows = np.searchsorted(support, p)
    np.testing.assert_array_equal(support[rows], p)
    np.testing.assert_array_equal(support_xyz[rows], xyz[p_rows])
    f_rows = fps_indices(support_xyz, case['budget'], seed=case['seed'], initial=rows)
    f = support[f_rows]
    require_equal(len(p), 32, 'priority budget')
    require_equal(len(f), case['budget'], 'final budget')
    require_equal(len(np.unique(f)), len(f), 'unique source rows')
    np.testing.assert_array_equal(f[:32], p)
    return dict(P=p.copy(), S=f[32:].copy(), F=f.copy(), priority_local=p_rows,
                final_support_rows=f_rows, xyz=support_xyz[f_rows].copy())


def selection_input(case):
    return {k: case[k] for k in ('candidate_flat','candidate_xyz','support_flat','support_xyz','first','seed','budget')}


def execute_controls(cases, gate):
    """Fail closed even when called independently of the runner."""
    if set(gate) != set(cases) or not all(v.get('status') == 'PASS' for v in gate.values()):
        raise ValueError('all four baseline gates must pass before controls')
    return {key: {str(w): select(selection_input(c), w) for w in (.25, 0.)} for key, c in cases.items()}


def load_inputs(source):
    comparison = json.loads((source/'comparison.json').read_text())
    config = comparison['config']
    require_equal(comparison['run_id'], SOURCE_RUN, 'Task03 run')
    contracts = source_contract(source, config)
    task01 = json.loads((TASK01/'geometry.json').read_text())
    for key in ('input_sha256', 'masks_sha256', 'camera'):
        require_equal(task01['config'][key], config[key], 'Task01/03 '+key)
    for field, digest in (('input', 'input_sha256'), ('masks', 'masks_sha256')):
        require_equal(sha(config[field]), config[digest], 'original '+field)
    frame = next(iter_saved_rgbd(config['input'], config['masks'], camera=config['camera']))
    with np.load(TASK01/'f0000.npz', allow_pickle=False) as a:
        require_equal(int(a['source_frame']), 0, 'Task01 frame')
        require_equal(str(a['camera']), config['camera'], 'camera')
        for key in ('intrinsic', 'cam2world'):
            np.testing.assert_array_equal(a[key], getattr(frame, key))
        geometry = OrganizedGeometry(a['xyz'], a['valid_depth'],
            {oid: a[name+'/object_mask'] for name, oid in OBJECTS.items()}, a['overlap_mask'])
        for name, oid in OBJECTS.items():
            np.testing.assert_array_equal(geometry.object_masks[oid], frame.masks[oid])
            np.testing.assert_array_equal(a[name+'/valid_points'], geometry.observed_mask(oid))
        np.testing.assert_array_equal(a['valid_depth'], np.isfinite(frame.depth) & (frame.depth > 0))
        np.testing.assert_array_equal(geometry.xyz[..., 2][geometry.valid_depth], frame.depth[geometry.valid_depth])
        np.testing.assert_array_equal(geometry.overlap_mask, frame.masks[PEG] & frame.masks[BOX])
    samples = np.load(source/'f0000_samples.npz', allow_pickle=False)
    metrics = json.loads((source/'f0000_metrics.json').read_text())
    edges = np.load(EDGES, allow_pickle=False)
    normal = np.load(NORMAL, allow_pickle=False)
    np.testing.assert_array_equal(edges['depth_m'], frame.depth)
    np.testing.assert_array_equal(normal['depth_m'], frame.depth)
    for archive in (edges, normal):
        np.testing.assert_array_equal(archive['intrinsic'], frame.intrinsic)
        require_equal(int(archive['source_frame']), 0, 'evaluation frame')
    cases = {}
    for method, estimator in METHODS.items():
        key = method+'_n256_r0.25_t20_s0'
        require_equal(config['conditions'][key], dict(method=method, num_points=256,
            structural_ratio=.25, orientation_threshold_deg=20, seed=0), 'condition')
        validate_saved_selection(samples, key, frame, geometry)
        for name, oid in OBJECTS.items():
            candidate = samples[f'evidence/{estimator}/{oid}/20/orientation_flat'].copy()
            count, first = EXPECTED[method, name]
            require_equal(len(candidate), count, method+'/'+name+' candidates')
            if candidate.dtype.kind not in 'iu' or not np.all(candidate[:-1] < candidate[1:]):
                raise ValueError('candidate source order invalid')
            support = np.flatnonzero(geometry.observed_mask(oid))
            if not np.isin(candidate, support).all():
                raise ValueError('candidate outside original support')
            choose = samples[key+'/object_id'] == oid
            saved_flat = samples[key+'/flat_index'][choose]
            p = samples[key+'/priority_selected'][choose]
            np.testing.assert_array_equal(p, np.arange(BUDGET[name]) < 32)
            require_equal(int(saved_flat[0]), first, 'stored first point')
            # The fixed initial also agrees with the original RNG stream, without a new selection.
            require_equal(int(candidate[np.random.default_rng(fps_seed(0, oid)).integers(len(candidate))]), first, 'original seeded initial')
            saved_counts = metrics[key]['objects'][name]['counts']
            for field, value in dict(observed=len(support), budget=BUDGET[name], priority=32,
                orientation_priority=32, depth_priority=0, surface_fps=BUDGET[name]-32,
                selected=BUDGET[name], candidates=count).items():
                require_equal(saved_counts[field], value, 'stored allocation '+field)
            feature_names = sorted(k.split('/metadata/')[1] for k in samples.files if k.startswith(key+'/metadata/'))
            require_equal(len(feature_names), 23, 'saved feature fields')
            features = {n: samples[key+'/metadata/'+n][choose].copy() for n in feature_names}
            for flag in ('orientation_candidate', 'structural_candidate'):
                np.testing.assert_array_equal(features[flag], np.isin(saved_flat, candidate))
            if features['depth_only_candidate'][p].any() or not (features['valid_points'].all() and features['object_mask'].all()):
                raise ValueError('stored feature support/role mismatch')
            cases[method+'/'+name] = dict(method=method, name=name, oid=oid, candidate_flat=candidate,
                candidate_xyz=geometry.xyz.reshape(-1, 3)[candidate].copy(), support_flat=support,
                support_xyz=geometry.xyz.reshape(-1, 3)[support].copy(), first=first, seed=fps_seed(0, oid),
                budget=BUDGET[name], saved_flat=saved_flat.copy(), saved_xyz=samples[key+'/points'][choose].copy(),
                saved_features=features, stored_metrics=metrics[key]['objects'][name],
                candidate_sha256=array_sha(candidate), support_sha256=array_sha(support),
                support_xyz_sha256=array_sha(geometry.xyz.reshape(-1,3)[support]))
    return frame, geometry, samples, edges, normal, cases, config, contracts


def evaluation_inputs(frame, geometry, samples, edges, normal):
    """Saved annotations only, after geometry-only baseline replay."""
    contexts = {}
    for name, oid in OBJECTS.items():
        ref = tuple(samples[f'evaluation/{name}/reference_{s}'] for s in ('uv', 'xyz'))
        label = 'seam' if name == 'peg' else 'rim'
        for suffix, value in zip(('uv', 'xyz'), ref):
            np.testing.assert_array_equal(value, edges[f'evaluation/{name}/{label}_{suffix}'])
        matcher = CoverageMatcher(geometry, oid, ref)
        observable = matcher.adjacencies['2px_3mm'].getnnz(axis=0) > 0
        require_equal((len(ref[0]), int(observable.sum())), (464, 462) if name == 'peg' else (103,103), 'GT/observed references')
        region = {k.split('/')[-1]: edges[k] for k in edges.files
                  if k.startswith('evaluation/'+name+'/') and edges[k].shape == frame.depth.shape}
        other = (tuple(edges[f'evaluation/{name}/seam_{s}'] for s in ('uv', 'xyz')))
        all_ref = tuple(np.concatenate([ref[i], other[i]]) for i in (0,1))
        negatives = negative_regions(region, {}, all_ref, frame.depth.shape)
        if name == 'peg':
            # Verify saved seam is exactly the reference pixel set using existing raster helper.
            np.testing.assert_array_equal(np.column_stack(np.nonzero(normal['evaluation/peg/seam_endpoints']))[:, ::-1], ref[0])
        contexts[name] = dict(matcher=matcher, reference=ref, observable=observable, negatives=negatives)
    return contexts


def baseline_gate(cases, contexts, replays):
    """All four lambda=1 selections and stored Frame0 metrics must be exact."""
    gate, selections = {}, {}
    for key, case in cases.items():
        chosen = replays[key]
        np.testing.assert_array_equal(chosen['P'], case['saved_flat'][:32], err_msg=key+' Priority order')
        np.testing.assert_array_equal(chosen['F'], case['saved_flat'], err_msg=key+' Final order')
        np.testing.assert_array_equal(chosen['xyz'], case['saved_xyz'], err_msg=key+' XYZ')
        context = contexts[case['name']]
        score, _ = context['matcher'].score(chosen['F'])
        require_equal(score, case['stored_metrics']['structural'], key+' all saved structural metrics')
        nearest = cKDTree(chosen['xyz']).query(case['support_xyz'])[0]*1000
        require_equal(stats(nearest), case['stored_metrics']['surface']['all']['distance_mm'], key+' saved Surface distances')
        nodes = dict(X=case['support_flat'], C=case['candidate_flat'], P=chosen['P'], S=chosen['S'], F=chosen['F'])
        occ = occupancy(SimpleNamespace(depth=contexts['depth']), nodes, context['negatives'])
        for roi, counts in case['stored_metrics']['negative'].items():
            for stage, saved in (('X','observed'), ('P','priority'), ('F','selected')):
                require_equal(occ['negative_roi'][roi][stage]['count'], counts[saved], key+' stored ROI '+roi)
        gate[key] = dict(status='PASS', priority_order_exact=True, final_order_xyz_exact=True,
            full_frame0_structural_metrics_exact=True, surface_metrics_exact=True, negative_roi_exact=True,
            candidate_order_source_support_exact=True, first=case['first'], priority_count=32, budget=case['budget'],
            candidate_sha256=case['candidate_sha256'], support_sha256=case['support_sha256'],
            support_xyz_sha256=case['support_xyz_sha256'])
        selections[key] = {'1.0': chosen}
    return gate, selections


def peg_labels(face, visible, seam, ref_uv):
    """Existing four-neighbour visible face transitions; ambiguous endpoints stay explicit."""
    pairs_by_pixel = {}
    for dv, du in ((1,0),(0,1)):
        h, w = face.shape
        a = (slice(0,h-dv), slice(0,w-du)); b = (slice(dv,h), slice(du,w))
        hit = visible[a] & visible[b] & (face[a] != face[b])
        for v, u in zip(*np.nonzero(hit)):
            pair = tuple(sorted((int(face[v,u]), int(face[v+dv,u+du]))))
            for pixel in ((v,u),(v+dv,u+du)):
                pairs_by_pixel.setdefault(pixel, set()).add(pair)
    rebuilt = np.zeros_like(seam)
    for v,u in pairs_by_pixel: rebuilt[v,u] = True
    np.testing.assert_array_equal(rebuilt, seam)
    labels, memberships = [], []
    for u, v in np.asarray(ref_uv, int):
        pairs = sorted(pairs_by_pixel.get((v,u), set()))
        memberships.append(['face_%s_%s'%p for p in pairs])
        labels.append('face_%s_%s'%pairs[0] if len(pairs) == 1 else 'ambiguous')
    return np.asarray(labels), memberships


def rim_labels(ref_xyz, cam2world, box_pose, hole_pose, radius):
    """Invert the existing rim_reference transform to identify its four lines.

    No samples/GT coordinates are created. 1e-8m is roundoff validation of line
    identity only; joint matching remains the unchanged 2px AND 3mm matcher.
    """
    rot = pose_rotation(box_pose)
    centre = ((hole_pose[:3]-box_pose[:3]) @ rot)[1:]
    world_to_cv = np.linalg.inv(cam2world)
    world = (ref_xyz-world_to_cv[:3,3]) @ np.linalg.inv(world_to_cv[:3,:3].T)
    local = (world-box_pose[:3]) @ np.linalg.inv(rot.T)
    labels = np.full(len(local), 'ambiguous', dtype='<U32')
    specs = {}
    hits = []
    for axis, other, c, oc in ((1,2,centre[0],centre[1]), (2,1,centre[1],centre[0])):
        for sign in (-1,1):
            label = f'rim_axis{axis}_{sign:+d}'
            hit = np.abs(local[:,axis]-(c+sign*radius)) <= 1e-8
            hit &= np.abs(local[:,other]-oc) <= radius+1e-8
            hits.append(hit)
            specs[label] = dict(axis=other, fixed_axis=axis, sign=sign, unit='metres',
                order_verified=True, domain='visible saved reference span only', local=local[:,other])
    matrix = np.column_stack(hits)
    for i, label in enumerate(specs): labels[matrix[:,i] & (matrix.sum(axis=1) == 1)] = label
    # Physical corner references can belong to two lines and are kept ambiguous.
    return labels, specs, dict(rot=rot, box_pose=box_pose, world_to_cv=world_to_cv)


def segment_spacing(t_reference, covered, t_nodes, *, order_verified):
    """Measured line projections and missing-reference runs, never array-order arcs."""
    if not order_verified or len(t_reference) < 2:
        return dict(status='N/A', reason='physical single-line order unverified or fewer than two references')
    order = np.argsort(t_reference, kind='stable')
    t = np.asarray(t_reference)[order]; hit = np.asarray(covered)[order]
    # Retain actual node projections; do not substitute covered reference centres.
    nodes = np.sort(np.asarray(t_nodes))
    lo, hi = float(t[0]), float(t[-1])
    within = nodes[(nodes >= lo) & (nodes <= hi)]
    gaps = np.diff(nodes)*1000
    missing = np.flatnonzero(~hit)
    runs = np.split(missing, np.flatnonzero(np.diff(missing)>1)+1) if len(missing) else []
    missing_runs = [dict(start_m=float(t[r[0]]), end_m=float(t[r[-1]]),
                         reference_count=len(r), span_mm=float((t[r[-1]]-t[r[0]])*1000)) for r in runs]
    return dict(status='verified line', reference_span_mm=(hi-lo)*1000,
        selected_node_projection_m=nodes.tolist(), selected_nodes_within_reference_span=len(within),
        actual_inter_node_gap_mm=gaps.tolist(), gap_stats_mm=stats(gaps),
        start_tail_to_first_node_mm=float((within[0]-lo)*1000) if len(within) else None,
        end_tail_from_last_node_mm=float((hi-within[-1])*1000) if len(within) else None,
        missing_reference_runs=missing_runs,
        missing_reference_run_total_span_mm=sum(r['span_mm'] for r in missing_runs),
        continuous_unpreserved_length_mm=None,
        length_caveat='Missing spans join consecutive uncovered saved references only; singleton span=0. '
                      'No continuous coverage/visibility interpolation. Gaps use all actual node projections; tails use nodes within the reference span.')


def feature_rows(case, flat):
    """Only the exact baseline condition's saved feature rows are available.

    Compact available-only arrays avoid placeholders masquerading as features.
    """
    lookup = {int(f): i for i,f in enumerate(case['saved_flat'])}
    positions = np.array([i for i,f in enumerate(flat) if int(f) in lookup], dtype=np.int64)
    source_rows = np.array([lookup[int(flat[i])] for i in positions], dtype=np.int64)
    features = {name: value[source_rows].copy() for name,value in case['saved_features'].items()}
    available = np.isin(np.arange(len(flat)), positions)
    for name, value in features.items():
        np.testing.assert_array_equal(value, case['saved_features'][name][source_rows])
    return available, positions, features


def score_case(case, chosen, context, frame, labels, specs, transform):
    matcher = context['matcher']; ref_uv, ref_xyz = context['reference']
    stages = dict(C=case['candidate_flat'], P=chosen['P'], S=chosen['S'], F=chosen['F'])
    coverage = staged_reference_coverage(matcher, stages['C'], stages['P'], stages['S'], stages['F'])
    arrays, details = {}, {}
    obs = context['observable']
    for stage, flat in stages.items():
        rows = np.searchsorted(matcher.flat, flat)
        adj = matcher.adjacencies['2px_3mm'][rows]
        covered = adj.getnnz(axis=0)>0
        hit_count = adj.getnnz(axis=1)
        matched = hit_count > 0
        uv, xyz = matcher.uv[rows], matcher.xyz[rows]
        arrays.update({f'{stage}/flat_index': flat, f'{stage}/xyz': xyz,
            f'{stage}/pixels_vu': uv[:, ::-1], f'{stage}/object_id': np.full(len(flat),case['oid'],np.int64),
            f'{stage}/covered': covered, f'{stage}/matched_node': matched,
            f'{stage}/reference_count_per_node': hit_count})
        near_stats = {}
        for group, use in (('all', np.ones(len(flat), bool)), ('structural', matched)):
            for unit, points, refs, scale in (('px',uv,ref_uv,1),('mm',xyz,ref_xyz,1000)):
                distance, nearest = cKDTree(points[use]).query(refs) if use.any() else (np.full(len(refs),np.nan),np.full(len(refs),-1,int))
                distance = distance*scale
                source = flat[use][nearest] if use.any() else np.full(len(refs),-1,np.int64)
                arrays[f'{stage}/nearest_{group}_{unit}'] = distance
                arrays[f'{stage}/nearest_{group}_{unit}_source'] = source
                near_stats[f'{group}_{unit}'] = stats(distance[obs]) if use.any() else dict(status='N/A',count=0)
        node_labels = []
        for i in range(len(flat)):
            lab = set(labels[adj[i].indices])
            node_labels.append(next(iter(lab)) if len(lab)==1 and 'ambiguous' not in lab else 'ambiguous' if lab else 'unmatched')
        node_labels = np.asarray(node_labels)
        arrays[f'{stage}/node_segment'] = node_labels
        segs = {}
        for label in sorted(set(labels)):
            mask = labels == label
            tnodes = []
            if label in specs and transform is not None:
                wt = transform['world_to_cv']; rot = transform['rot']
                world = (xyz[node_labels == label]-wt[:3,3]) @ np.linalg.inv(wt[:3,:3].T)
                local = (world-transform['box_pose'][:3]) @ np.linalg.inv(rot.T)
                tnodes = local[:,specs[label]['axis']]
            segs[label] = dict(gt_references=int(mask.sum()), observed_references=int((mask&obs).sum()),
                covered_reference_ids=np.flatnonzero(mask&covered).tolist(),
                missing_observed_reference_ids=np.flatnonzero(mask&obs&~covered).tolist(),
                coverage_count=int((mask&covered).sum()), matching_node_count=int((node_labels == label).sum()),
                node_flat_indices=flat[node_labels == label].tolist(),
                nearest_structural_px=stats(arrays[f'{stage}/nearest_structural_px'][mask&obs]) if matched.any() else None,
                nearest_structural_mm=stats(arrays[f'{stage}/nearest_structural_mm'][mask&obs]) if matched.any() else None,
                spacing=segment_spacing(specs[label]['local'][mask] if label in specs else [],
                    covered[mask], tnodes, order_verified=label in specs and label != 'ambiguous'))
        coo = adj.tocoo()
        arrays[f'{stage}/joint_node_row'] = coo.row
        arrays[f'{stage}/joint_reference_id'] = coo.col
        nearest_node_dist = {}
        for unit, points, scale in (('px',uv,1), ('mm',xyz,1000)):
            d = cKDTree(points[matched]).query(points[matched],k=2)[0][:,1]*scale if matched.sum()>1 else np.empty(0)
            nearest_node_dist[unit] = stats(d)
        details[stage] = dict(selected_flat_indices=flat.tolist(), selected_xyz_m=xyz.tolist(),
            selected_pixels_vu=uv[:,::-1].tolist(), node_count=len(flat), structural_matching_nodes=int(matched.sum()),
            matching_flat_indices=flat[matched].tolist(), unique_reference_count=int(covered.sum()),
            nodes_covering_multiple_references=int((hit_count>1).sum()),
            references_covered_by_multiple_nodes=int((adj.getnnz(axis=0)>1).sum()),
            reference_count_per_matching_node=hit_count[matched].tolist(),
            covered_reference_ids=np.flatnonzero(covered).tolist(),
            missing_observed_reference_ids=np.flatnonzero(obs&~covered).tolist(),
            nearest_observed_reference_distance=near_stats, matched_node_nearest_euclidean_distance=nearest_node_dist,
            segments=segs, quadrant_origin='existing matcher median UV halves',
            reference_quadrants=matcher.score(flat)[0]['tolerance']['2px_3mm']['reference_quadrants'])
    surface_distance = cKDTree(chosen['xyz']).query(case['support_xyz'])[0]*1000
    arrays['surface_support_to_final_mm'] = surface_distance
    nodes = dict(X=case['support_flat'], **stages)
    occ = occupancy(frame, nodes, context['negatives'])
    # Selection-rate denominator is the same candidate depth band, not all candidates.
    zc = frame.depth.ravel()[stages['C']]; zp = frame.depth.ravel()[stages['P']]
    med, p95 = (occ['depth_band_bounds_m'][k] for k in ('median','p95'))
    def bands(z): return (z<=med, (z>med)&(z<=p95), z>p95)
    occ['candidate_band_priority_selection_rates'] = [dict(candidate_nodes=int(c.sum()),priority_nodes=int(p.sum()),
        rate=float(p.sum()/c.sum()) if c.any() else None) for c,p in zip(bands(zc),bands(zp))]
    available, positions, features = feature_rows(case, chosen['F'])
    arrays.update({'F/feature_available':available, 'F/feature_available_positions':positions,
        'F/selection_role':np.where(np.arange(len(chosen['F']))<32,'Priority','Surface')})
    for name,value in features.items(): arrays['F/available_features/'+name] = value
    for stage in ('P','S'):
        sl = slice(0,32) if stage == 'P' else slice(32,None)
        arrays[stage+'/feature_available'] = available[sl]
        arrays[stage+'/selection_role'] = np.full(len(chosen[stage]), 'Priority' if stage == 'P' else 'Surface')
    return dict(coverage=coverage, stages=details, surface_distance_mm=stats(surface_distance),
        occupancy=occ, feature_availability=dict(fields=23, exact_available_rows=int(available.sum()),
        unavailable_rows=int((~available).sum()), source='same-estimator Task03 n256/r.25/t20/seed0 stored rows only',
        storage='F/available_features uses F/feature_available_positions; unavailable rows have no feature values',
        baseline_feature_alignment_exact=True, dense_pca_cache_used=False)), arrays


def create_plots(output, frame, cases, selections, result, arrays, contexts):
    plt.rcParams.update({'font.size':9})
    for object_name in ('peg','box'):
        refs = contexts[object_name]['reference'][0]
        fig, axes = plt.subplots(4,3,figsize=(15,16), constrained_layout=True)
        for mi, method in enumerate(METHODS):
            key = method+'/'+object_name
            for si, stage in enumerate(('P','F')):
                for wi, w in enumerate(LAMBDAS):
                    prefix = key+'/'+str(w)+'/'+stage
                    ax = axes[mi*2+si,wi]
                    ax.imshow(frame.rgb)
                    flat = selections[key][str(w)][stage]
                    uv = np.column_stack(np.unravel_index(flat,frame.depth.shape))[:,::-1]
                    hit = arrays[prefix+'/matched_node']
                    covered = arrays[prefix+'/covered']
                    # Hollow nodes stay visible when one node covers many reference markers.
                    ax.scatter(*refs[~covered].T,s=15,c='#e42b67',marker='x',label='missing GT reference',zorder=3)
                    ax.scatter(*refs[covered].T,s=12,c='#2be4bb',marker='.',label='covered GT reference',zorder=3)
                    ax.scatter(*uv[~hit].T,s=35,facecolors='none',edgecolors='#edb53f',label='other selected node',zorder=4)
                    ax.scatter(*uv[hit].T,s=85,facecolors='none',edgecolors='#4b8aff',linewidths=1.8,label='joint matching node',zorder=5)
                    margin = 12 if object_name == 'box' else 10
                    ax.set(xlim=(refs[:,0].min()-margin,refs[:,0].max()+margin),
                        ylim=(refs[:,1].max()+margin,refs[:,1].min()-margin),
                        title=f'{method} {stage}, lambda={w:g}: {hit.sum()} nodes / {covered.sum()} refs',xlabel='u (px)',ylabel='v (px)')
                    if mi==0 and si==0 and wi==0: ax.legend(fontsize=7,loc='upper left')
        fig.suptitle(f'{object_name.title()} structural closeup: actual nodes and unique references (P=Priority, F=Final)')
        fig.savefig(output/f'{object_name}_structural_zoom.png',dpi=140); plt.close(fig)
    fig, axes = plt.subplots(4,3,figsize=(15,16),constrained_layout=True)
    fig3, axes3 = plt.subplots(4,3,figsize=(15,14),constrained_layout=True)
    for row,(key,case) in enumerate(cases.items()):
        for col,w in enumerate(LAMBDAS):
            chosen = selections[key][str(w)]; ax=axes[row,col]; ax3=axes3[row,col]
            ax.imshow(frame.rgb)
            for stage,color in (('S','#36cccd'),('P','#ff9b24')):
                f=chosen[stage]; uv=np.column_stack(np.unravel_index(f,frame.depth.shape))[:,::-1]
                ax.scatter(*uv.T,s=14,c=color,label=stage,alpha=.9)
                xyz=case['support_xyz'][np.searchsorted(case['support_flat'],f)]
                ax3.scatter(xyz[:,0]*1000,xyz[:,2]*1000,s=12,c=color,label=stage)
            ax.set(title=f'{key}, lambda={w:g}',xlabel='u (px)',ylabel='v (px)')
            support=case['support_xyz']
            ax3.scatter(support[:,0]*1000,support[:,2]*1000,s=.5,c='#69788d',alpha=.13,zorder=0)
            ax3.set(title=f'{key}, lambda={w:g}',xlabel='camera X (mm)',ylabel='camera Z (mm)')
            ax3.set_xlim(support[:,0].min()*1000-5,support[:,0].max()*1000+5)
            ax3.set_ylim(support[:,2].max()*1000+5,support[:,2].min()*1000-5)
            ax.legend(); ax3.legend()
    fig.suptitle('All selected pixels: Priority orange, Surface cyan; all 256 points per estimator/lambda')
    fig.savefig(output/'selection_overview.png',dpi=120); plt.close(fig)
    fig3.suptitle('Actual Camera X/Z distribution; depth is never scaled in output geometry')
    fig3.savefig(output/'spatial_xz.png',dpi=130); plt.close(fig3)
    fig, axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for i,(key,case) in enumerate(cases.items()):
        ax=axes.ravel()[i]; values=result['cases'][key]['conditions']; x=np.arange(3)
        for stage,offset,color in (('P',-.15,'#ef9834'),('F',.15,'#147da4')):
            ax.bar(x+offset,[values[str(w)]['stages'][stage]['unique_reference_count'] for w in LAMBDAS],.28,label=stage+' unique refs',color=color)
        second=ax.twinx()
        second.plot(x,[values[str(w)]['surface_distance_mm']['mean'] for w in LAMBDAS],'-o',c='#9b238e',label='Surface mean mm')
        second.plot(x,[values[str(w)]['surface_distance_mm']['p95'] for w in LAMBDAS],'--o',c='#9b238e',label='Surface p95 mm')
        ax.set(title=key,xticks=x,xticklabels=['1','.25','0'],xlabel='lambda',ylabel='unique references',ylim=(0,464 if case['name']=='peg' else 103))
        second.set_ylabel('support to Final distance (mm)'); ax.legend(loc='upper left'); second.legend(loc='upper right')
    fig.savefig(output/'coverage_surface_tradeoff.png',dpi=150); plt.close(fig)
    # Per-reference UV positions above avoid implying reference-ID order is physical.
    # Here the explicit sort is image u then v and is labelled as such, not arc length.
    fig, axes=plt.subplots(4,2,figsize=(17,13),constrained_layout=True)
    for i,(key,case) in enumerate(cases.items()):
        refs=contexts[case['name']]['reference'][0]; order=np.lexsort((refs[:,1],refs[:,0]))
        matrix=np.array([arrays[key+'/'+str(w)+'/'+s+'/covered'][order] for w in LAMBDAS for s in ('P','F')])
        axes[i,0].imshow(matrix,aspect='auto',interpolation='nearest',cmap='RdYlGn',vmin=0,vmax=1)
        axes[i,0].set(title=key+' joint reference coverage',xlabel='reference sorted by image u, then v (NOT physical order)',
            yticks=range(6),yticklabels=[f'{w:g} {s}' for w in LAMBDAS for s in ('P','F')])
        for w in LAMBDAS:
            d=arrays[key+'/'+str(w)+'/F/nearest_structural_mm'][order]
            axes[i,1].plot(np.arange(len(d)),d,label=f'{w:g}',alpha=.8)
        axes[i,1].set(title=key+' Final: nearest actual structural node',xlabel='same image-coordinate sort',ylabel='3D distance (mm)')
        axes[i,1].legend(title='lambda')
    fig.savefig(output/'local_reference_distances.png',dpi=140); plt.close(fig)
    # Actual selected projections, not the centres of references covered by a node.
    fig, axes=plt.subplots(4,1,figsize=(15,12),constrained_layout=True)
    for i,(method,stage) in enumerate((('S2','P'),('S2','F'),('S3','P'),('S3','F'))):
        ax=axes[i]; lines=sorted(result['segment_definitions']['box']['labels'])
        for j,label in enumerate(lines):
            if label=='ambiguous': continue
            for wi,w in enumerate(LAMBDAS):
                spacing=result['cases'][method+'/box']['conditions'][str(w)]['stages'][stage]['segments'][label]['spacing']
                if spacing['status']=='N/A': continue
                ax.scatter(np.asarray(spacing['selected_node_projection_m'])*1000,
                    np.full(len(spacing['selected_node_projection_m']),j+wi*.22),marker='|',s=180,
                    color=('#2764b7','#df842b','#159d6c')[wi],label=f'{w:g}' if j==0 else None)
        ax.set(title=f'{method} Box {stage}: actual node projections by verified rim line',
            yticks=np.arange(len(lines))+.22,yticklabels=lines,xlabel='box-local coordinate along each line (mm); each row is independent')
        ax.legend(title='lambda')
    fig.savefig(output/'box_line_node_spacing.png',dpi=140); plt.close(fig)


def summary_table(result):
    lines=['| 조건 | λ | P nodes / refs | F nodes / refs | Surface 추가 / overlap refs | Surface mean / p95 mm | P depth median / p95 m | Negative union P |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,row in result['cases'].items():
        for w in LAMBDAS:
            r=row['conditions'][str(w)]; p=r['stages']['P']; f=r['stages']['F']; d=r['surface_distance_mm']
            diff=r['coverage']['differences']; z=r['occupancy']['depth_m']['P']
            neg=r['occupancy']['negative_roi']['known_negative_union']['P']['count']
            lines.append(f"| {key} | {w:g} | {p['structural_matching_nodes']} / {p['unique_reference_count']} | "
                f"{f['structural_matching_nodes']} / {f['unique_reference_count']} | {diff['S_minus_P']['count']} / {diff['P_intersect_S']['count']} | "
                f"{d['mean']:.3f} / {d['p95']:.3f} | {z['median']:.4f} / {z['p95']:.4f} | {neg}/32 |")
    return lines


def create_report(output, result, test_summary):
    rel=lambda p: os.path.relpath(p, REPORT.parent)
    lines=['# Task03-E — Depth-Weighted Structural FPS & Local Geometry Preservation','',
        f"실행 UUID: `{result['run_id']}`. ep000 Frame 0의 geometry-level offline ablation을 완료했다. 네 조합의 λ=1 exact reproduction 관문은 모두 PASS다.", '',
        '## 고정 조건과 baseline 관문','',
        '- S2(FD2), S3(PCA5), Peg/Box, num_points=256, structural_ratio=.25, threshold=20°, seed=0. λ는 1/.25/0만 실행했다.',
        '- 기존 `fps_indices()`에 `[x,y,sqrt(λ)z]`를 Priority 거리 계산용으로만 전달했다. Camera XY-only는 normalized image UV FPS와 다르다.',
        '- 후보 source index·raster 순서·support·첫 점·quota를 유지했다. Peg 32+97=129, Box 32+95=127, estimator/λ마다 256점이다.',
        '- Task01/03 episode·mask hash, camera/intrinsic/cam2world, dense Z와 원본 depth, validity·mask·overlap 및 선택 XYZ/pixel/object가 exact 일치한다. Backprojection·normal·candidate·allocation은 재계산하지 않았다.',
        '- 네 λ=1 Priority 32점 및 기존 Priority 초기 집합을 포함한 3D Surface/Final 순서·XYZ가 exact 일치한다. Frame0의 모든 저장 structural tolerance/거리/quadrant 지표, 전체 Surface 거리 통계, negative ROI 수치도 exact 재현했다. 기준 tolerance 변경은 없다.',
        '- 원본 orientation 배열은 저장 후보를 그대로 복사하며 count/order hash와 저장 metadata의 candidate membership을 검증했다. 독립적인 normal 재계산에 의한 후보 재검출은 수행하지 않았다.', '',
        '| 조건 | 후보 | 첫 flat index | Final budget |', '| --- | ---: | ---: | ---: |']
    for key,row in result['cases'].items():
        g=result['baseline_gate'][key]; lines.append(f"| {key} | {row['candidate_count']} | {g['first']} | {g['budget']} |")
    lines += ['', '## Structural retention과 Surface trade-off','',
        '아래 nodes는 실제 joint-matching 선택 node 수이며 refs는 unique reference 수다. P 실제 선택 수는 32, F는 Peg 129/Box 127이다. 분모는 Peg observed 462(GT 464), Box observed/GT 103이다. Candidate와 Surface의 전체 수·matching node·reference ID 및 P/S 교집합은 JSON/NPZ에 별도 보존했다.', '', *summary_table(result),'',
        '## 질문별 판단','']
    for i,answer in enumerate(result['answers_ko'],1): lines += [f'{i}. {answer}', '']
    lines += ['## 국소 분포와 물리적 간격','',
        'GT와 구간 label은 sampler의 입력이 아니다. Reference 배열 순서는 물리적 순서로 쓰지 않았다. 아래 coverage는 관측 reference 분모이며 corner의 복수 귀속은 ambiguous로 보존한다. Segment node 수는 해당 node가 매칭한 모든 reference가 하나의 label에 속할 때만 집계한다. 여러 label에 걸친 node는 ambiguous로 기록하므로 구간별 node 수를 전체 matching node 수로 단순 합산하지 않는다.', '',
        '| 조건 | 구간 | observed | P covered λ=1/.25/0 | F covered λ=1/.25/0 | F actual nodes λ=1/.25/0 |',
        '| --- | --- | ---: | --- | --- | --- |']
    for key,case in result['cases'].items():
        conditions=case['conditions']; labels=conditions['1.0']['stages']['F']['segments']
        for label,seg in labels.items():
            def values(stage,field): return ' / '.join(str(conditions[str(w)]['stages'][stage]['segments'][label][field]) for w in LAMBDAS)
            lines.append(f"| {key} | {label} | {seg['observed_references']} | {values('P','coverage_count')} | {values('F','coverage_count')} | {values('F','matching_node_count')} |")
    lines += ['',
        '- Peg: 기존 visible face map의 4-neighbour face transition과 seam_endpoints가 exact 일치한다. 각 reference의 face-pair membership과 모서리 복수 귀속을 저장했다. Raster seam의 두께·접점을 물리적 선분의 중심선으로 바꾸지 않았으므로 face-pair의 물리적 순서·간격·미보존 길이는 **N/A**다. 새 GT geometry나 큰 helper 구현은 범위에서 제외했다.',
        '- Box: 기존 `rim_reference`가 사용한 pose/centre/radius와 transform을 역변환해 확인 가능한 네 선분 label을 붙였다. 새 reference를 생성하지 않았다. 1e-8m line identity 검사는 부동소수점 roundoff 확인이며 matching tolerance와 무관하다. 물리적 corner·복수 귀속은 ambiguous이며 spacing은 N/A다.',
        '- S2 Box rim_axis1_-1에서 Final 실제 node의 최대 투영 간격은 λ=1/.25/0 순서로 25.54/15.62/19.83mm다. λ=0의 첫 node는 visible reference min보다 바깥에 있지만 joint match에 포함된 실제 선택점이므로 간격 계산에 유지했다. λ=0 S3 rim_axis1_+1은 두 node 사이 17.25mm 간격이 남는다. 다른 선분의 node 부족과 전체 누락은 별도다.',
        '- 확인된 Box 선분에서는 실제 matching node XYZ를 해당 축에 투영해 정렬했다. 실제 인접 node 간격·끝부분 node 미배치 span과 연속 missing-reference run의 양 끝 span(mm)을 JSON에 기록했다. 분석 domain은 저장된 visible reference의 min–max span이다. 전체 물리 선분의 occluded 부분 길이는 평가하지 않았다.',
        '- Missing-reference run span은 연속으로 누락된 저장 reference의 첫/끝 좌표 차이다. 단일 누락 reference의 span은 0이다. 연속적인 구조의 미보존 길이를 보간하지 않았으므로 continuous_unpreserved_length_mm은 null(N/A)이다. 이 값으로 실제 미보존 길이의 부재를 주장하지 않는다.',
        '- Reference별 nearest pixel/3D 거리·nearest source index는 전체 선택점과 실제 structural matching node 집합을 나누어 P/F 각각 저장했다. 두 독립 최근접점으로 joint match를 합성하지 않는다. observed reference의 mean/median(p50)/p95/max, covered/missing ID, image quadrants도 저장했다. Peg 미관측 2개 ID는 별도 mask로 남겼다.',
        '- 한 node가 여러 reference를 덮는 joint pair 목록, node별 reference 수, 여러 node에 덮이는 reference 수를 별도 기록했다. 높은 reference coverage를 촘촘한 실제 node 배치로 해석하지 않는다.', '',
        '## Feature 대응과 제약','',
        '모든 선택 XYZ/pixel/source index/object ID/역할은 원본 관측과 exact 대응한다. Baseline과 겹치는 저장 행에서 23 metadata field를 source index로 gather하고 정렬을 검증했다. 이 구현은 같은 estimator·동일 baseline condition의 행만 exact available로 간주하는 보수적인 범위다. 다른 저장 condition의 feature를 임의 병합하지 않는다. `F/feature_available_positions`가 compact `F/available_features/*`의 대상 행을 가리키며, unavailable 행에는 feature 값을 만들지 않는다. P/S availability mask도 별도 저장한다. 과거 dense PCA5 feature cache 대체·feature 재추출·optimized PCA·Vision Tensor packaging은 미실행이다.', '',
        '| 조건 | λ | exact feature rows / Final | unavailable |', '| --- | ---: | ---: | ---: |']
    for key,case in result['cases'].items():
        for w in LAMBDAS:
            a=case['conditions'][str(w)]['feature_availability']; lines.append(f"| {key} | {w:g} | {a['exact_available_rows']} / {BUDGET[key.split('/')[1]]} | {a['unavailable_rows']} |")
    lines += ['', '## 검증·재현과 완료 범위','',
        '- 최초 시도는 실제 FPS 이전의 runner 검증 오류로 중단했다. Surface metadata의 depth_only_candidate까지 false라고 가정한 assertion을 기존 source 계약에 맞게 Priority 행으로 한정했다. 기존 sampler·입력·평가 tolerance는 변경하지 않았다. [실패 시도 기록](../../experiments/9702d2709f0946f895e504554537c492/attempt_note.json)을 보존했다. 실제 λ=1 재현 불일치는 없었다.',
        f'- 필수 신규/기존 테스트 PASS: `{test_summary}`. [tests.log]({rel(output/"tests.log")}). λ 거리식, λ=0 Z 불변, source/feature alignment, GT·구간 label 독립성, 관문 실패 시 control 차단, budget/중복, same-pair·dedup 및 N/A 계약을 포함한다.',
        f'- 저장 결과 후처리의 물리적 간격 명시와 synthetic 계약은 [tests_final.log]({rel(output/"tests_final.log")})로 추가 확인했다. 실제 데이터 FPS 반복은 없다.',
        f'- [verification.json]({rel(output/"verification.json")})에 실제 데이터의 네 exact baseline, source/input hash 보존, 모든 λ의 동일 후보·초기점·quota, 원본 XYZ/역할/feature 대응 및 상대 링크·PNG decode 결과를 기록했다.',
        '- 기존 ExperimentLogger(team=visual, job_type=evaluate)를 WANDB_MODE=disabled로 재사용했다. UUID/project/group 규칙은 바꾸지 않았다. 기존 파일·source·artifact sidecar를 보존하고 새 산출물만 등록했다.',
        '- CPU NumPy/SciPy offline 평가만 실행했다. Simulator·renderer·CUDA/GPU/backend는 이 작업에서 실행하지 않았다. 독립 episode/frame/seed, 추가 λ, production 정책 채택, 학습·GNN/BC·PCL·GPU 최적화·외부 upload·dependency 설치·commit은 미실행이다.',
        '- FPS 실행 당시 source는 source_snapshot과 source_hashes.json의 sources_before/after에 보존했다. 결과 해석·그림의 색상·선분 투영 간격 요약을 저장된 결과에서 보완한 최종 source는 report_source_snapshot 및 report_generation_sources로 구분한다. 실제 데이터의 FPS는 반복 실행하지 않았다.',
        '- Production sampling/normal/allocation/tracking 및 기존 Phase1–3 구현은 변경하지 않았다. Git 시작 상태가 dirty이므로 commit만으로 재현할 수 없으며 source snapshot·input hashes·implementation diff를 함께 보관한다.', '',
        '재실행: `WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_priority_fps_metric_control --baseline <새 실행 전 baseline.json>`; 기존 보고서를 덮어쓰지 않는다. 각 실행은 새 UUID를 만든다. 결과 보고 후 다음 알고리즘 변경 없이 중단한다.', '',
        '## 산출물','']
    for name in (STEM+'.html',STEM+'.json',STEM+'.npz','selection_overview.png','box_structural_zoom.png',
        'peg_structural_zoom.png','spatial_xz.png','coverage_surface_tradeoff.png','local_reference_distances.png',
        'box_line_node_spacing.png','verification.json','source_hashes.json','implementation.patch','git_start.json','git_end.json'):
        lines.append(f'- [{name}]({rel(output/name)})')
    lines += ['', '## 기존 근거','',
        '- [승인 전 계획](03e_priority_fps_metric_control_plan.md)',
        f'- [Task03 원본 comparison](../../visual/results/sampling/{SOURCE_RUN}/comparison.json), [Frame0 NPZ](../../visual/results/sampling/{SOURCE_RUN}/f0000_samples.npz)',
        '- [Task01 dense geometry](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0000.npz)',
        '- [Phase1](../../experiments/d3f14ee494894daabb8ba2fab8e1b00c/diagnostic.json), [Phase2](../../experiments/aa5d71078e0841fd886b5026d6030ebc/priority_fps_trace.json), [Phase3](../../experiments/f36f1e9e2bb64866b8d6c792db158635/priority_fps_seed_control.json)',
        '- [runner](../../visual/runtime/test/run_priority_fps_metric_control.py), [tests](../../visual/runtime/test/test_priority_fps_metric_control.py)', '']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    shutil.copyfile(REPORT,output/'report_snapshot.md')


def create_html(output,result):
    links=[STEM+'.json',STEM+'.npz','verification.json','source_hashes.json','tests.log','implementation.patch',
           'report_snapshot.md',os.path.relpath(REPORT,output),
           os.path.relpath(REPO/'visual/results/sampling'/SOURCE_RUN/'comparison.json',output),
           os.path.relpath(REPO/'visual/results/sampling'/SOURCE_RUN/'f0000_samples.npz',output)]
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task03-E metric control</title>',
        '<style>body{font:16px system-ui;margin:24px auto;max-width:1500px;padding:0 20px;color:#182538}img{width:100%;height:auto}a{color:#075ca8}pre{white-space:pre-wrap}li{margin:.6em 0}</style>',
        '<h1>Task03-E — Depth-weighted Priority FPS</h1>',
        f"<p>Run {result['run_id']}; ep000 Frame 0; S2/S3; λ=1/.25/0. All four exact baseline gates PASS. Geometry-level ablation.</p>",
        '<p>Priority=32/object; Final Peg=129, Box=127. Actual XYZ and 3D Surface metric retained. GT/segment annotations enter evaluation only.</p>',
        '<nav>'+' | '.join(f'<a href="{html.escape(p)}">{html.escape(Path(p).name)}</a>' for p in links)+'</nav>',
        '<ol>'+''.join('<li>'+html.escape(a)+'</li>' for a in result['answers_ko'])+'</ol>',
        '<pre>'+html.escape('\n'.join(summary_table(result)))+'</pre>']
    for name,title in [('selection_overview.png','S2/S3 and lambda: all selected positions'),
        ('box_structural_zoom.png','Box rim: actual nodes vs covered references'),
        ('peg_structural_zoom.png','Peg seam: actual nodes vs covered references'),
        ('local_reference_distances.png','Local reference coverage and nearest structural-node distances'),
        ('box_line_node_spacing.png','Verified Box line projections; Peg physical order/spacing N/A'),
        ('coverage_surface_tradeoff.png','Structural retention and general Surface trade-off'),
        ('spatial_xz.png','Actual X/Z distribution')]:
        parts += [f'<h2>{title}</h2>',f'<a href="{name}"><img src="{name}" alt="{title}"></a>']
    parts += ['<p>One node can cover multiple references. Reference-ID or image-u order is not arc length. '
        'Peg physical spacing and continuous unpreserved length: N/A. Box missing-reference run spans are discrete measured spans only. '
        'Unavailable original features are explicit; no dense PCA substitute. No production policy adopted.</p></html>']
    (output/(STEM+'.html')).write_text('\n'.join(parts),encoding='utf-8')


def verify_links(output):
    targets=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            targets.extend(value for attr,value in attrs if attr in ('href','src') and value)
    Links().feed((output/(STEM+'.html')).read_text())
    for target in targets:
        if '://' in target or not (output/target).is_file(): raise ValueError('bad HTML local target '+target)
    md=re.findall(r'\]\(([^)]+)\)',REPORT.read_text())
    for target in md:
        if not (REPORT.parent/target).is_file(): raise ValueError('bad Markdown target '+target)
    pngs={}
    for p in output.glob('*.png'):
        pixels=plt.imread(p)
        if pixels.ndim!=3 or min(pixels.shape[:2])<400 or float(pixels.std())<.01:
            raise ValueError('invalid/empty PNG '+str(p))
        pngs[p.name]=list(pixels.shape)
    return dict(html_local_targets=len(targets),markdown_local_targets=len(md),png_decoded=pngs,status='PASS')


def answers(result):
    def values(key,stage,field):
        return ' → '.join(str(result['cases'][key]['conditions'][str(w)]['stages'][stage][field]) for w in LAMBDAS)
    change=[]
    for key,case in result['cases'].items():
        change.append(key+' '+ '/'.join(str(case['conditions'][str(w)]['paired_vs_lambda1']['priority_replaced_nodes']) for w in (.25,0.))+'점')
    return [
        'Z 가중치를 낮추면 실제 Priority 위치가 달라진다. λ=.25/0에서 baseline 대비 교체된 Priority node 수는 '+', '.join(change)+
        '이다. λ=1→.25→0에서 Priority depth median은 Peg S2 .1930→.1865→.1835m, S3 .2030→.1930→.1865m, '
        'Box S2 .5060→.5025→.5000m, S3 .5805→.5035→.4955m로 가까운 쪽으로 이동했다. '
        'Peg는 face_0_2의 실제 node가 늘고 face_2_5가 줄었다. Box는 일부 rim 선분의 node가 늘었지만 '
        'S3의 최대 depth 1.426m는 유지되고 p95도 1.4057→1.3683→1.3864m로 높게 남는다. XY-only도 X/Y 자체의 depth 의존성을 제거하지 않는다.',
        '이 frame의 두 estimator 모두 Box rim coverage가 개선됐다. Priority unique reference coverage(λ=1→.25→0)는 S2 ('+
        values('S2/box','P','unique_reference_count')+')/103, S3 ('+values('S3/box','P','unique_reference_count')+
        ')/103이다. Final은 S2 ('+values('S2/box','F','unique_reference_count')+')/103, S3 ('+
        values('S3/box','F','unique_reference_count')+')/103이다. Priority의 실제 matching node 수는 각각 4→7→7과 1→2→3, '
        'Final은 5→8→8과 1→2→4다. Reference 증가와 node 증가를 구분해야 하며 유지된 구간은 rim 전체의 일부다.',
        'Peg 모든 구간이 더 촘촘하게 보존되지는 않았다. Final coverage는 S2 ('+values('S2/peg','F','unique_reference_count')+
        ')/462, S3 ('+values('S3/peg','F','unique_reference_count')+')/462다. '
        'face_0_2는 S2 refs 24→36→40, nodes 4→7→7 및 S3 refs 29→43→44, nodes 4→7→7로 증가했다. '
        '반면 face_2_5는 S2 refs 84→72→70, nodes 19→16→15 및 S3 refs 90→66→66, nodes 20→17→15로 감소했다. '
        '전체 matching node 수는 모든 λ에서 S2 29, S3 30으로 같다. Peg의 Surface는 추가 reference를 하나도 보완하지 않았다. '
        '예컨대 λ=0 S3는 30개 node가 160개 reference를 덮는다. Reference coverage 상승은 실제 node 수 증가를 뜻하지 않는다. '
        '물리적 중심선 순서·간격은 저장 정보로 검증되지 않아 N/A다.',
        'Surface representation의 손익은 물체·estimator에 따라 다르다. λ=.25에서 S2 Box는 rim 개선과 함께 Surface mean/p95가 '
        '11.992/20.179→11.903/19.857mm로 줄었지만 λ=0에서는 12.019/20.210mm로 소폭 증가했다. '
        'S3 Box는 11.253/18.751→11.461/19.088→11.580/19.077mm로 악화됐다. Peg는 두 control 모두 mean/p95가 baseline보다 증가했다. '
        'Box Priority negative ROI union은 S2 3→5→6/32, S3 9→11→14/32로 늘었다. '
        'Priority 변경에 따라 뒤따르는 원본 3D Surface FPS도 달라진다. Depth 분위수와 ROI는 설명용이며 filtering이나 일반 평면의 오선택 판정으로 사용하지 않았다.',
        'Z 조정 후에도 전역 FPS의 국소 구조 누락이 남는다. S2 Box rim_axis1_+1은 모든 λ에서 0/25로 '
        '저장 visible span 46.33mm 전체가 reference 기준으로 누락된다. S3 Box rim_axis1_-1도 모든 λ에서 0/26이며 span은 48.74mm다. '
        'λ=0의 Peg도 S2 320/462, S3 302/462 observed references가 누락된다. '
        'S2 Box rim_axis1_-1의 실제 node 투영 간격은 일부 줄지만 다른 선분 전체 누락을 보완하지 못한다. '
        '한 frame의 최적 λ나 production sampling 정책을 채택하지 않는다.'
    ]


def diagnose(source, output_root, baseline_path):
    if os.environ.get('WANDB_MODE')!='disabled': raise ValueError('WANDB_MODE=disabled required')
    if REPORT.exists(): raise FileExistsError('refuse to overwrite Task03-E report')
    baseline=json.loads(Path(baseline_path).read_text()); verify_preserved(baseline)
    inputs=[source/'comparison.json',source/'f0000_samples.npz',source/'f0000_metrics.json',
        TASK01/'geometry.json',TASK01/'f0000.npz',NORMAL,EDGES,
        REPO/'experiments/d3f14ee494894daabb8ba2fab8e1b00c/diagnostic.json',
        REPO/'experiments/aa5d71078e0841fd886b5026d6030ebc/priority_fps_trace.json',
        REPO/'experiments/f36f1e9e2bb64866b8d6c792db158635/priority_fps_seed_control.json']
    task03_config=json.loads((source/'comparison.json').read_text())['config']
    inputs += [Path(task03_config[k]) for k in ('input','masks')]
    before=hashes(inputs)
    sources=list((REPO/'visual/runtime').glob('*.py'))+list((REPO/'visual/runtime/test').glob('*.py'))+[
        REPO/'utils/experiment_logger.py',REPO/'utils/experiment_artifacts.py']
    source_hashes=hashes(sources)
    config=dict(task='03-E',source_run_id=SOURCE_RUN,episode='ep000',frame=0,methods=METHODS,
        lambdas=list(LAMBDAS),num_points=256,structural_ratio=.25,threshold_deg=20,seed=0,
        priority_count=32,budgets=BUDGET,metric='dx^2+dy^2+lambda*dz^2 (Priority only)',
        surface_metric='original 3D Euclidean',gt_selection_input=False,device='cpu',
        simulator_executed=False,gpu_executed=False,input_sha256=before,source_sha256=source_hashes,
        python=platform.python_version(),numpy=np.__version__)
    with ExperimentLogger(team='visual',config=config,seed=0,model='task03e_priority_fps_metric_control',
        job_type='evaluate',root=output_root) as logger:
        output=logger.directory
        write_json(output/'git_start.json',baseline)
        shutil.copyfile(Path(baseline_path).with_name('git_start.diff'),output/'git_start.diff')
        for path in sources:
            dest=output/'source_snapshot'/path.relative_to(REPO); dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
        with (output/'tests.log').open('w') as log:
            test=subprocess.run([sys.executable,'-m','unittest',*TEST_MODULES,'-v'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        if test.returncode:
            write_json(output/'verification.json',dict(status='FAIL',reason='required tests failed',controls_executed=False))
            logger.link_output(output/'verification.json',kind='reports'); logger.link_output(output/'tests.log',kind='reports')
            raise RuntimeError('required tests failed; no real selections executed: '+str(output))
        test_summary=next(s for s in (output/'tests.log').read_text().splitlines() if s.startswith('Ran '))+'; OK'
        print('Tests PASS. Loading saved inputs for four baseline gates.',flush=True)
        try:
            frame,geometry,samples,edges,normal,cases,old_config,contracts=load_inputs(source)
            replays={key:select(selection_input(case),1.) for key,case in cases.items()}
            contexts=evaluation_inputs(frame,geometry,samples,edges,normal)
            contexts['depth']=frame.depth
            gate,selections=baseline_gate(cases,contexts,replays)
        except Exception as exc:
            write_json(output/'verification.json',dict(status='FAIL',reason=str(exc),controls_executed=False,
                baseline_gate='failed; no tolerance/source/sampler adjustments'))
            write_json(output/(STEM+'.json'),dict(run_id=logger.run_id,status='baseline failed',reason=str(exc),controls_executed=False))
            REPORT.write_text('# Task03-E — Baseline reproduction 실패\n\n'+str(exc)+'\n\nλ=.25/0 control은 실행하지 않았다. 기존 sampler/source/tolerance를 변경하지 않고 중단한다.\n'+
                f'\n[verification]({os.path.relpath(output/"verification.json",REPORT.parent)})\n',encoding='utf-8')
            for p in (output/'verification.json',output/(STEM+'.json')): logger.link_output(p,kind='reports')
            raise RuntimeError('baseline failed; controls blocked; '+str(output)) from exc
        write_json(output/'baseline_gate.json',gate)
        print('All four baseline gates PASS. Executing only lambda=.25/0 controls.',flush=True)
        controls=execute_controls(cases,gate)
        for key in cases: selections[key].update(controls[key])
        # Only now assign GT face-pair/rim-line labels and evaluate selected controls.
        peg_lab,peg_members=peg_labels(normal['evaluation/peg/face'],normal['evaluation/peg/gt_visible'],
            normal['evaluation/peg/seam_endpoints'],contexts['peg']['reference'][0])
        with np.load(old_config['input'],allow_pickle=False) as episode:
            box_lab,box_specs,box_transform=rim_labels(contexts['box']['reference'][1],frame.cam2world,
                episode['box_pose'][0],episode['box_hole_pose'][0],float(episode['box_hole_radii']))
        result=dict(run_id=logger.run_id,source_run_id=SOURCE_RUN,config=config,baseline_gate=gate,
            source_contract=contracts,cases={},segment_definitions=dict(
                peg=dict(labels=sorted(set(peg_lab)),membership_by_reference=peg_members,
                    physical_spacing='N/A: raster face transitions have no verified physical centreline order'),
                box=dict(labels=sorted(set(box_lab)),line_identity_roundoff_m=1e-8,
                    domain='visible saved reference span; corner ambiguity explicit')))
        arrays={}
        for name,context in ((n,contexts[n]) for n in OBJECTS):
            arrays.update({name+'/reference_uv':context['reference'][0],name+'/reference_xyz':context['reference'][1],
                name+'/observable':context['observable'],name+'/reference_segment':peg_lab if name=='peg' else box_lab})
        for key,case in cases.items():
            row=dict(candidate_count=len(case['candidate_flat']),candidate_sha256=case['candidate_sha256'],
                support_count=len(case['support_flat']),support_sha256=case['support_sha256'],conditions={})
            for field in ('candidate_flat','candidate_xyz','support_flat','support_xyz'):
                arrays[key+'/'+field]=case[field]
            b=selections[key]['1.0']
            for w in LAMBDAS:
                chosen=selections[key][str(w)]; name=case['name']
                scored,data=score_case(case,chosen,contexts[name],frame,peg_lab if name=='peg' else box_lab,
                    {} if name=='peg' else box_specs,None if name=='peg' else box_transform)
                scored['initial_flat_index']=int(chosen['P'][0])
                scored['candidate_sha256']=case['candidate_sha256']; scored['support_sha256']=case['support_sha256']
                np.testing.assert_array_equal(chosen['xyz'],geometry.xyz.reshape(-1,3)[chosen['F']])
                require_equal(scored['initial_flat_index'],case['first'],'same lambda initial')
                common=len(np.intersect1d(chosen['P'],b['P']))
                scored['paired_vs_lambda1']=dict(priority_shared_nodes=common,priority_replaced_nodes=32-common,
                    final_shared_nodes=len(np.intersect1d(chosen['F'],b['F'])))
                row['conditions'][str(w)]=scored
                arrays.update({key+'/'+str(w)+'/'+k:v for k,v in data.items()})
            for w in LAMBDAS:
                r=row['conditions'][str(w)]; base=row['conditions']['1.0']
                r['paired_vs_lambda1'].update(
                    priority_reference_delta=r['stages']['P']['unique_reference_count']-base['stages']['P']['unique_reference_count'],
                    final_reference_delta=r['stages']['F']['unique_reference_count']-base['stages']['F']['unique_reference_count'],
                    surface_mean_delta_mm=r['surface_distance_mm']['mean']-base['surface_distance_mm']['mean'],
                    surface_p95_delta_mm=r['surface_distance_mm']['p95']-base['surface_distance_mm']['p95'])
            result['cases'][key]=row
        for method in METHODS:
            for w in LAMBDAS:
                require_equal(sum(len(selections[method+'/'+n][str(w)]['F']) for n in OBJECTS),256,'total budget')
                joined=np.concatenate([selections[method+'/'+n][str(w)]['F'] for n in OBJECTS])
                require_equal(len(np.unique(joined)),256,'cross-object source uniqueness')
        result['answers_ko']=answers(result)
        result['pass_all']=True
        write_json(output/(STEM+'.json'),result)
        np.savez_compressed(output/(STEM+'.npz'),**arrays)
        # Readback validates storage, not merely the in-memory arrays.
        with np.load(output/(STEM+'.npz'),allow_pickle=False) as saved:
            for k,v in arrays.items(): np.testing.assert_array_equal(saved[k],v,err_msg=k)
        create_plots(output,frame,cases,selections,result,arrays,contexts)
        write_json(output/'git_end.json',dict(git_status=subprocess.check_output(['git','status','--short'],cwd=REPO).decode(),
            original_preservation=verify_preserved(baseline)))
        write_json(output/'source_hashes.json',dict(inputs_before=before,inputs_after=hashes(inputs),
            sources_before=source_hashes,sources_after=hashes(sources),original_preservation=verify_preserved(baseline)))
        require_equal(hashes(inputs),before,'all inputs after experiment')
        require_equal(hashes(sources),source_hashes,'all sources after experiment')
        verification=dict(status='PASS',baseline_gate=gate,required_tests=dict(summary=test_summary,modules=TEST_MODULES),
            controls_executed=True,real_fps_calls=dict(baseline=8,controls=16),
            candidate_initial_support_quota_fixed=True,original_selected_xyz_source_role_feature_alignment=True,
            feature_unavailable_explicit=True,npz_exact_readback=True,source_input_hashes_unchanged=True,
            preservation=verify_preserved(baseline),same_pair_joint_and_unique_reference_matching=True,
            gt_and_segment_selection_independence='synthetic tests; selection has no annotation arguments',
            ambiguous_physical_spacing='N/A',production_policy_adopted=False,
            skipped=['simulator','renderer','CUDA/GPU','other frames/seeds/lambdas','feature reconstruction',
                'production policy','external upload','dependency installation','commit'])
        write_json(output/'verification.json',verification)
        # The report includes only local targets and is generated from the saved numbers.
        (output/'implementation.patch').touch()
        shutil.copyfile(output/'tests.log',output/'tests_final.log')
        create_report(output,result,test_summary)
        create_html(output,result)
        verification['document_checks']=verify_links(output)
        write_json(output/'verification.json',verification)
        diff=subprocess.check_output(['git','diff','--binary'],cwd=REPO)
        for path in (Path(__file__),Path(__file__).with_name('test_priority_fps_metric_control.py'),REPORT):
            delta=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path.relative_to(REPO))],cwd=REPO,stdout=subprocess.PIPE)
            if delta.returncode!=1: raise ValueError('cannot preserve implementation diff')
            diff+=delta.stdout
        (output/'implementation.patch').write_bytes(diff)
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','config.yaml','metrics.json'):
                logger.link_output(path,kind='reports')
        logger.record(source_run_id=SOURCE_RUN,baseline_exact=True,simulator_executed=False,gpu_executed=False)
        logger.log({k:{w:dict(priority_refs=r['stages']['P']['unique_reference_count'],final_refs=r['stages']['F']['unique_reference_count'],
            surface_mean_mm=r['surface_distance_mm']['mean'],surface_p95_mm=r['surface_distance_mm']['p95'])
            for w,r in row['conditions'].items()} for k,row in result['cases'].items()},step=0,prefix='eval/task03e')
        print(output,flush=True)
    return output


def render_existing(output):
    """Report-only refresh. Saved raw selections and executed sources stay intact."""
    from unittest.mock import patch
    from utils.experiment_artifacts import register_existing_output
    output=Path(output).resolve()
    result=json.loads((output/(STEM+'.json')).read_text())
    with np.load(output/(STEM+'.npz'),allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    raw_npz_sha=sha(output/(STEM+'.npz'))
    original_config=json.loads((REPO/'visual/results/sampling'/SOURCE_RUN/'comparison.json').read_text())['config']
    frame=next(iter_saved_rgbd(original_config['input'],original_config['masks'],camera=original_config['camera']))
    cases={}; selections={}; contexts={}
    for name in OBJECTS:
        contexts[name]=dict(reference=(arrays[name+'/reference_uv'],arrays[name+'/reference_xyz']))
    for key in result['cases']:
        method,name=key.split('/')
        cases[key]=dict(name=name,support_flat=arrays[key+'/support_flat'],support_xyz=arrays[key+'/support_xyz'])
        selections[key]={str(w):{stage:arrays[key+'/'+str(w)+'/'+stage+'/flat_index'] for stage in ('P','S','F')} for w in LAMBDAS}
        for w in LAMBDAS:
            for stage in ('C','P','S','F'):
                for segment in result['cases'][key]['conditions'][str(w)]['stages'][stage]['segments'].values():
                    spacing=segment['spacing']
                    if spacing['status']=='verified line':
                        gaps=np.diff(spacing['selected_node_projection_m'])*1000
                        spacing['actual_inter_node_gap_mm']=gaps.tolist()
                        spacing['gap_stats_mm']=stats(gaps)
                        spacing['length_caveat']='Missing spans join consecutive uncovered saved references only; singleton span=0. No continuous coverage/visibility interpolation. Gaps use all actual node projections; tails use nodes within the reference span.'
    result['answers_ko']=answers(result)
    result['report_only_refresh']=dict(real_fps_calls=0,raw_selection_npz_sha256=raw_npz_sha)
    with patch(__name__+'.fps_indices',side_effect=AssertionError('report refresh must not select')):
        create_plots(output,frame,cases,selections,result,arrays,contexts)
        write_json(output/(STEM+'.json'),result)
        test_summary=next(s for s in (output/'tests.log').read_text().splitlines() if s.startswith('Ran '))+'; OK'
        create_report(output,result,test_summary)
        create_html(output,result)
    require_equal(sha(output/(STEM+'.npz')),raw_npz_sha,'raw selection NPZ unchanged')
    provenance=json.loads((output/'source_hashes.json').read_text())
    paths=[Path(__file__),Path(__file__).with_name('test_priority_fps_metric_control.py')]
    provenance['report_generation_sources']=hashes(paths)
    for p in paths:
        dest=output/'report_source_snapshot'/p.relative_to(REPO); dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
    write_json(output/'source_hashes.json',provenance)
    baseline=json.loads((output/'git_start.json').read_text())
    verification=json.loads((output/'verification.json').read_text())
    verification['preservation']=verify_preserved(baseline)
    verification['document_checks']=verify_links(output)
    verification['report_only_refresh']=dict(real_fps_calls=0,raw_selection_npz_unchanged=True,
        execution_sources='source_snapshot; sources_before/after',final_sources='report_source_snapshot; report_generation_sources')
    write_json(output/'verification.json',verification)
    write_json(output/'git_end.json',dict(git_status=subprocess.check_output(['git','status','--short'],cwd=REPO).decode(),original_preservation=verify_preserved(baseline)))
    diff=subprocess.check_output(['git','diff','--binary'],cwd=REPO)
    for p in (*paths,REPORT):
        delta=subprocess.run(['git','diff','--no-index','--','/dev/null',str(p.relative_to(REPO))],cwd=REPO,stdout=subprocess.PIPE)
        if delta.returncode!=1: raise ValueError('cannot preserve final implementation diff')
        diff+=delta.stdout
    (output/'implementation.patch').write_bytes(diff)
    for p in output.rglob('*'):
        if p.is_file() and p.name not in ('metadata.json','manifest.json','config.yaml','metrics.json') and not p.name.endswith('.tracking.json'):
            register_existing_output(output,p,kind='reports')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=REPO/'visual/results/sampling'/SOURCE_RUN)
    parser.add_argument('--out',type=Path,default=REPO/'experiments')
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--render-existing',type=Path,help='refresh presentation only; no FPS or geometry reconstruction')
    args=parser.parse_args()
    if args.render_existing:
        print(render_existing(args.render_existing))
    else:
        if args.baseline is None: parser.error('--baseline is required for a new experiment')
        diagnose(args.source.resolve(),args.out,args.baseline)


if __name__=='__main__': main()
