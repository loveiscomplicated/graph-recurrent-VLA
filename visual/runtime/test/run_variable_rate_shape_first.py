"""Task03-J fixed offline feasibility sweep. No production defaults are changed.

WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task03j-mpl .venv/bin/python -m \
visual.runtime.test.run_variable_rate_shape_first --baseline /tmp/task03j_baseline.json
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
from time import perf_counter

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.spatial import cKDTree

from graph.vision_input import receive_vision_packet
from utils.experiment_logger import ExperimentLogger
from visual.runtime.geometry import OrganizedGeometry, build_organized_geometry
from visual.runtime.rgbd import iter_saved_rgbd
from visual.runtime.sampling import fps_indices
from visual.runtime.test import run_voxel_structural_sampling as voxel
from visual.runtime.test import run_priority_fps_metric_control as prior
from visual.runtime.test.run_rim_bottleneck import occupancy, require_equal

REPO = prior.REPO
SOURCE = REPO/'visual/results/sampling'/prior.SOURCE_RUN
STEM = 'variable_rate_shape_first'
REPORT = REPO/'docs/visual-graph-integration/03j_variable_rate_shape_first.md'
PLAN = REPORT.with_name('03j_variable_rate_shape_first_plan.md')
FRAMES = (0, 45, 80)
ARMS = ((1., 0), (.75, 0), (.5, 0), (.25, 0), (.5, 1))
SURFACE = {0: 97, 1: 95}
NAMES = {0: 'peg', 1: 'box'}
TESTS = ['visual.runtime.test.'+n for n in (
    'test_variable_rate_shape_first', 'test_variable_rate_graph_contract',
    'test_voxel_structural_sampling', 'test_sampling', 'test_sampling_geometry',
    'test_sampling_evaluation')] + ['tests.test_experiment_logger', 'tests.test_experiment_artifacts']
RULES = dict(episode='ep000', camera='base_camera', frames=FRAMES, arms=ARMS,
    pca='stored optimized PCA5 orientation only, strict >20 degrees', voxel='Task03-I nominal metric, multiplier 1, original centroid-nearest',
    contour='raw mask RETR_EXTERNAL CHAIN_APPROX_NONE; canonical global component walk',
    thinning='rational floor(k*q/p), ceil(M*p/q); phase1 only .5, no wrap/refill',
    surface_additional=SURFACE, global_cap=None, structural_cap=None,
    extra_outlier_removals=0, production_policy_adopted=False,
    selection_order='object 0 then 1; per-object P_pca + new P_sil + Surface',
    phase_policy='diagnostic only; never choose phase after results',
    scale_interpretation='PCA5 spatial-support design proxy; no measured silhouette uncertainty calibration')


def write_json(path, data):
    voxel.write_json(path, data)


def thin_indices(m, ratio, phase=0):
    if type(m) is not int or m < 0:
        raise ValueError('nonnegative integer M required')
    if not np.isfinite(float(ratio)) or not 0 < float(ratio) <= 1:
        raise ValueError('ratio must be finite in (0,1]')
    r = Fraction(str(ratio))
    if type(phase) is not int or phase not in (0, 1) or (phase == 1 and r != Fraction(1, 2)):
        raise ValueError('phase1 diagnostic is defined only at ratio .5')
    p, q = r.numerator, r.denominator
    count = (m*p+q-1)//q
    rows = np.array([(k*q)//p+phase for k in range(count)], dtype=np.int64)
    return rows[rows < m]


def canonical_walk(walk):
    walk = np.asarray(walk, np.int64)
    if len(walk) > 1 and walk[0] == walk[-1]:
        walk = walk[:-1]
    if not len(walk):
        return walk
    # Degenerate thin contours may revisit their minimum source.
    options = []
    for sequence in (walk, walk[::-1]):
        for start in np.flatnonzero(sequence == sequence.min()):
            options.append(tuple(np.roll(sequence, -int(start)).tolist()))
    return np.asarray(min(options), np.int64)


def stable_unique(values):
    return np.asarray(list(dict.fromkeys(map(int, values))), np.int64)


def observed_runs(good):
    """Walk positions; bridge closure only on a completely observed contour."""
    good = np.asarray(good, bool)
    if not len(good): return []
    if good.all(): return [(np.arange(len(good), dtype=np.int64), True)]
    # Start after a real exclusion, so a valid boundary-crossing run stays intact.
    order = np.roll(np.arange(len(good)), -int(np.flatnonzero(~good)[0])-1)
    runs, current = [], []
    for pos in order:
        if good[pos]: current.append(int(pos))
        elif current:
            runs.append((np.asarray(current, np.int64), False)); current = []
    if current: runs.append((np.asarray(current, np.int64), False))
    return runs


def outer_contours(raw_mask, observed, valid_depth, overlap):
    raw_mask, observed = np.asarray(raw_mask), np.asarray(observed)
    if raw_mask.ndim != 2 or raw_mask.dtype != bool:
        raise ValueError('raw bool [H,W] mask required')
    if any(np.asarray(a).shape != raw_mask.shape or np.asarray(a).dtype != bool
           for a in (observed, valid_depth, overlap)) or (observed & ~raw_mask).any():
        raise ValueError('invalid contour support masks')
    contours, _ = cv2.findContours(raw_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = raw_mask.shape
    walks = [canonical_walk(c[:, 0, 1].astype(np.int64)*w+c[:, 0, 0]) for c in contours]
    walks.sort(key=lambda x: tuple(x.tolist()))
    components = []
    ordered = []
    for component, walk in enumerate(walks):
        good = observed.ravel()[walk]
        v, u = np.divmod(walk, w)
        segments = np.full(len(walk), -1, np.int64)
        for sid, (positions, _) in enumerate(observed_runs(good)): segments[positions] = sid
        components.append(dict(component=component, walk=walk, observed=good, observed_segment_id=segments,
            invalid_depth=~valid_depth.ravel()[walk], overlap=overlap.ravel()[walk],
            image_boundary=(v == 0) | (u == 0) | (v == h-1) | (u == w-1),
            walk_position=np.arange(len(walk), dtype=np.int64),
            repeated_source_visits=len(walk)-len(np.unique(walk))))
        ordered.extend(walk[good].tolist())
    return dict(components=components, C=stable_unique(ordered),
        raw_unique=stable_unique([s for walk in walks for s in walk]),
        walk_count=sum(len(w) for w in walks), canonical_component_count=len(walks))


def source_rows(x, source):
    x, source = np.asarray(x), np.asarray(source)
    rows = np.searchsorted(x, source)
    if ((rows >= len(x)).any() or not np.array_equal(x[rows], source)):
        raise ValueError('candidate/selected source outside observed X')
    return rows


def prepare_object(x, xyz, pca, raw_mask, valid_depth, overlap, intrinsic):
    """Observation-only preprocessing, shared once by all five arms."""
    t0 = perf_counter()
    x, xyz = voxel.sources(x, xyz)
    pca = np.asarray(pca)
    voxel.sources(pca, xyz[source_rows(x, pca)])
    if xyz.dtype != np.float32:
        raise ValueError('original float32 geometry required')
    observed = np.zeros(raw_mask.shape, bool)
    observed.ravel()[x] = True
    contours = outer_contours(raw_mask, observed, valid_depth, overlap)
    t1 = perf_counter()
    scale = voxel.nominal_scale(xyz, intrinsic)
    t2 = perf_counter()
    candidates = {'pca': pca.copy(), 'sil': contours['C']}
    groups, reps = {}, {}
    for name, c in candidates.items():
        asc = np.sort(c)
        if len(x):
            group = voxel.representatives(asc, xyz[source_rows(x, asc)], scale['a_m'])
        else:
            group = dict(keys=np.empty((0, 3), np.int64), cells=np.empty((0, 3), np.int64),
                membership=np.empty(0, np.int64), capacity=np.empty(0, np.int64),
                cell_rep_rows=np.empty(0, np.int64), representative_rows=np.empty(0, np.int64),
                same_representative_source=np.empty(0, np.int64), same_cell_distance_m=np.empty(0))
        r = asc[group['representative_rows']]
        # Independent compression is ascending; restore contour order only afterwards.
        reps[name] = c[np.isin(c, r)]
        group['candidate_source_ascending'] = asc
        groups[name] = group
    t3 = perf_counter()
    return dict(X=x.copy(), xyz=xyz.copy(), C=candidates, R=reps, groups=groups,
        contours=contours, scale=scale, shape=raw_mask.shape,
        timings_ms=dict(contour=(t1-t0)*1000, nominal_scale=(t2-t1)*1000, voxel=(t3-t2)*1000))


def select_object(prepared, ratio, phase, requested_surface):
    """No GT, ROI, reference, cap, score, or random first-point input."""
    if type(requested_surface) is not int or requested_surface < 0:
        raise ValueError('nonnegative Surface request required')
    t0 = perf_counter()
    ts = {name: r[thin_indices(len(r), ratio, phase)] for name, r in prepared['R'].items()}
    t1 = perf_counter()
    p = stable_unique(np.r_[ts['pca'], ts['sil']])
    t2 = perf_counter()
    x, xyz = prepared['X'], prepared['xyz']
    count = min(requested_surface, len(x)-len(p))
    initial = source_rows(x, p)
    if not len(initial) and count:
        initial = np.array([0], np.int64)  # deterministic first Surface, inside request
    rows = fps_indices(xyz, len(p)+count, initial=initial)
    f = x[rows]
    np.testing.assert_array_equal(f[:len(p)], p)
    if len(np.unique(f)) != len(f):
        raise AssertionError('duplicate source after Surface')
    t3 = perf_counter()
    return dict(T=ts, P=p, S=f[len(p):], F=f, xyz=xyz[rows].copy(),
        shortfall=requested_surface-count, requested_surface=requested_surface,
        timings_ms=dict(thinning=(t1-t0)*1000, union=(t2-t1)*1000, surface_fps=(t3-t2)*1000))


def make_packet(source_frame, prepared, selected, feature_cases):
    """Exact stored-condition feature join; provenance stays separate from features."""
    t0 = perf_counter()
    fields = None
    object_rows = []
    compact_by_object = []
    for oid in (0, 1):
        prep, s, case = prepared[oid], selected[oid], feature_cases[oid]
        f = s['F']
        available, positions, features = prior.feature_rows(case, f)
        schema = [dict(name=name, shape=list(value.shape[1:]), dtype=value.dtype.str)
                  for name, value in sorted(case['saved_features'].items())]
        if fields is None:
            fields = schema
        elif fields != schema:
            raise ValueError('feature schema differs across objects')
        view = np.full((len(f), sum(int(np.prod(k['shape'])) if k['shape'] else 1 for k in fields)), np.nan)
        offset = 0
        for field in fields:
            values = features[field['name']]
            width = int(np.prod(field['shape'])) if field['shape'] else 1
            view[positions, offset:offset+width] = values.reshape(len(positions), width)
            offset += width
        evidence = np.column_stack([np.isin(f, prep['C'][n]) for n in ('pca', 'sil')])
        chosen = np.column_stack([np.isin(f, s['T'][n]) for n in ('pca', 'sil')])
        surface = np.arange(len(f)) >= len(s['P'])
        object_rows.append(dict(points=s['xyz'], object_id=np.full(len(f), oid, np.int64),
            geometric_features=view, provenance=dict(source_index=f.copy(), object_id=np.full(len(f), oid, np.int64),
                candidate_evidence=evidence, both_evidence=evidence.all(axis=1), selected_by_structural_source=chosen,
                selection_stage=surface.astype(np.uint8), surface_selected=surface, feature_available=available,
                feature_field_available=np.repeat(available[:, None], len(fields), axis=1))))
        compact_by_object.append((positions, features))
    packet = dict(episode='ep000', camera='base_camera', source_frame=int(source_frame),
        coordinate_frame='CV_camera_metres', feature_fields=fields, edge_status='not_implemented',
        intra_object_edge_index=np.empty((2, 0), np.int64))
    for name in ('points', 'object_id', 'geometric_features'):
        packet[name] = np.concatenate([o[name] for o in object_rows])
    packet['provenance'] = {name: np.concatenate([o['provenance'][name] for o in object_rows])
                            for name in object_rows[0]['provenance']}
    offset = len(selected[0]['F'])
    packet['compact_features'] = {name: dict(
        positions=np.r_[compact_by_object[0][0], compact_by_object[1][0]+offset],
        values=np.concatenate([v[name] for _, v in compact_by_object])) for name in compact_by_object[0][1]}
    received = receive_vision_packet(packet)
    if packet_digest(packet) != packet_digest(received.packet):
        raise AssertionError('receiver changed packet rows/identity')
    return packet, (perf_counter()-t0)*1000


def flatten_arrays(data, prefix=''):
    out = {}
    if isinstance(data, np.ndarray):
        out[prefix] = data
    elif isinstance(data, dict):
        for key, value in data.items():
            out.update(flatten_arrays(value, prefix+'/'+str(key) if prefix else str(key)))
    elif isinstance(data, (list, tuple)):
        for i, value in enumerate(data):
            out.update(flatten_arrays(value, prefix+'/'+str(i)))
    return out


def packet_digest(packet):
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        elif isinstance(value, dict):
            for k in sorted(value):
                digest.update(k.encode()); visit(value[k])
        elif isinstance(value, (list, tuple)):
            for v in value: visit(v)
        else: digest.update(json.dumps(value, sort_keys=True).encode())
    visit(packet)
    return digest.hexdigest()


def load_observations():
    """Only observation/candidate/metadata keys are read here, never evaluation/*."""
    config = json.loads((SOURCE/'comparison.json').read_text())['config']
    if config['camera'] != 'base_camera': raise ValueError('camera mismatch')
    for name in ('input', 'masks'):
        require_equal(prior.sha(config[name]), config[name+'_sha256'], name+' hash')
    dense_config = json.loads((prior.TASK01/'geometry.json').read_text())['config']
    for name in ('input_sha256', 'masks_sha256', 'camera'):
        require_equal(dense_config[name], config[name], 'dense source '+name)
    frames = {f.source_frame: f for f in iter_saved_rgbd(config['input'], config['masks']) if f.source_frame in FRAMES}
    if tuple(frames) != FRAMES: raise ValueError('fixed frames missing or out of order')
    prepared, feature_cases, geometries = {}, {}, {}
    paths = [Path(config['input']), Path(config['masks']), SOURCE/'comparison.json', prior.TASK01/'geometry.json']
    for index, frame in frames.items():
        dense_path, sample_path = prior.TASK01/f'f{index:04d}.npz', SOURCE/f'f{index:04d}_samples.npz'
        paths.extend([dense_path, sample_path])
        with np.load(dense_path, allow_pickle=False) as a, np.load(sample_path, allow_pickle=False) as samples:
            require_equal(int(a['source_frame']), index, 'dense source frame')
            require_equal(str(a['camera']), frame.camera, 'dense camera')
            require_equal(int(samples['source_frame']), index, 'candidate source frame')
            np.testing.assert_array_equal(samples['shape'], frame.depth.shape)
            for key in ('intrinsic', 'cam2world'):
                np.testing.assert_array_equal(a[key], getattr(frame, key))
            g = OrganizedGeometry(a['xyz'].copy(), a['valid_depth'].copy(),
                {oid: a[name+'/object_mask'].copy() for oid, name in NAMES.items()}, a['overlap_mask'].copy())
            expected = build_organized_geometry(frame)
            for name in ('xyz', 'valid_depth', 'overlap_mask'):
                np.testing.assert_array_equal(getattr(g, name), getattr(expected, name))
            prepared[index], feature_cases[index] = {}, {}
            for oid, name in NAMES.items():
                np.testing.assert_array_equal(g.object_masks[oid], frame.masks[oid])
                np.testing.assert_array_equal(a[name+'/valid_points'], g.observed_mask(oid))
                x = np.flatnonzero(g.observed_mask(oid))
                c = samples[f'evidence/pca5/{oid}/20/orientation_flat'].copy()
                key = 'S3_n256_r0.25_t20_s0'
                obj = samples[key+'/object_id'] == oid
                stored_source = samples[key+'/flat_index'][obj].copy()
                np.testing.assert_array_equal(samples[key+'/points'][obj], g.xyz.reshape(-1, 3)[stored_source])
                fields = {k.split('/metadata/')[1]: samples[k][obj].copy()
                          for k in samples.files if k.startswith(key+'/metadata/')}
                require_equal(len(fields), 23, 'stored feature count')
                np.testing.assert_array_equal(fields['orientation_candidate'], np.isin(stored_source, c))
                prepared[index][oid] = prepare_object(x, g.xyz.reshape(-1, 3)[x], c, frame.masks[oid],
                    g.valid_depth, g.overlap_mask, frame.intrinsic)
                feature_cases[index][oid] = dict(saved_flat=stored_source, saved_features=fields)
            geometries[index] = g
    return frames, geometries, prepared, feature_cases, prior.hashes(paths), config


def arm_key(frame, ratio, phase):
    return f'f{frame:04d}/r{ratio:g}_phase{phase}'


def select_all(prepared, feature_cases):
    selected, packets, timings = {}, {}, {}
    for frame in FRAMES:
        for ratio, phase in ARMS:
            key = arm_key(frame, ratio, phase)
            selected[key] = {oid: select_object(prepared[frame][oid], ratio, phase, SURFACE[oid]) for oid in (0, 1)}
            for oid, s in selected[key].items():
                p = prepared[frame][oid]
                require_equal(len(s['P']), len(s['T']['pca'])+len(s['T']['sil'])-len(np.intersect1d(s['T']['pca'], s['T']['sil'])), 'Structural union count')
                require_equal(len(s['S']), min(SURFACE[oid], len(p['X'])-len(s['P'])), 'Surface added count')
                np.testing.assert_array_equal(s['F'], np.r_[s['P'], s['S']])
                np.testing.assert_array_equal(s['xyz'], p['xyz'][source_rows(p['X'], s['F'])])
            packets[key], timings[key] = make_packet(frame, prepared[frame], selected[key], feature_cases[frame])
    return selected, packets, timings


def save_readback(path, arrays):
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        require_equal(set(saved.files), set(arrays), 'archive keys')
        for k, v in arrays.items(): np.testing.assert_array_equal(saved[k], v, err_msg=k)
    return prior.sha(path)


def freeze(output, prepared, selected, packets):
    arrays = flatten_arrays(dict(observations=prepared, selections=selected, packets=packets))
    # Packet identity/schema and contour scalar flags accompany the exact arrays.
    metadata = dict(gt_opened=False, packets={k: dict(hash=packet_digest(p), episode=p['episode'], camera=p['camera'],
        source_frame=p['source_frame'], coordinate_frame=p['coordinate_frame'], feature_fields=p['feature_fields'],
        edge_status=p['edge_status']) for k, p in packets.items()},
        object_scales={str(f): {str(o): prepared[f][o]['scale'] for o in (0, 1)} for f in FRAMES})
    digest = save_readback(output/'selection_before_gt.npz', arrays)
    metadata['npz_sha256'] = digest
    write_json(output/'selection_before_gt.json', metadata)
    require_equal(json.loads((output/'selection_before_gt.json').read_text()), voxel.serial(metadata), 'freeze JSON readback')
    return digest, arrays


def fraction(n, d):
    return dict(numerator=int(n), denominator=int(d), value=float(n/d) if d else None)


def distance_stats(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return dict(query_count=len(values), available_count=len(finite), unavailable_count=len(values)-len(finite),
        mean=float(finite.mean()) if len(finite) else None,
        median=float(np.median(finite)) if len(finite) else None,
        p95=float(np.percentile(finite, 95, method='linear')) if len(finite) else None,
        max=float(finite.max()) if len(finite) else None,
        within_3mm=fraction(int((finite <= 3).sum()), len(values)))


def distances(prep, query, target):
    q = prep['xyz'][source_rows(prep['X'], query)]
    t = prep['xyz'][source_rows(prep['X'], target)]
    if not len(t): return np.full(len(q), np.nan)
    return cKDTree(t).query(q)[0]*1000


def gap_metrics(contours, matched_sources, prep):
    """Consecutive missing walk visits; internal adjacent missing XYZ lengths only.

    No distance to retained endpoints is charged. Closed all-missing loops include
    their closure; exclusions/components never contribute connecting lengths.
    """
    gaps, arrays, runs_report = [], {}, []
    for comp in contours['components']:
        walk = comp['walk']
        for run_id, (positions, closed) in enumerate(observed_runs(comp['observed'])):
            sources = walk[positions]
            missing = ~np.isin(sources, matched_sources)
            chunks = []
            if missing.all() and len(missing):
                chunks = [(np.arange(len(missing)), closed)]
            else:
                order = np.arange(len(missing))
                if closed and (~missing).any(): order = np.roll(order, -int(np.flatnonzero(~missing)[0])-1)
                current = []
                for pos in order:
                    if missing[pos]: current.append(int(pos))
                    elif current: chunks.append((np.asarray(current), False)); current = []
                if current: chunks.append((np.asarray(current), False))
            for chunk, loop in chunks:
                ids = sources[chunk]
                xyz = prep['xyz'][source_rows(prep['X'], ids)].astype(float)
                length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()*1000)
                if loop and len(xyz)>1: length += float(np.linalg.norm(xyz[-1]-xyz[0])*1000)
                g = dict(component=comp['component'], observed_run=run_id, missing_walk_visits=len(ids),
                    internal_length_mm=length, closed_missing_loop=loop,
                    start_walk_position=int(positions[chunk[0]]), end_walk_position=int(positions[chunk[-1]]))
                arrays[f'gap{len(gaps)}/source'] = ids.copy()
                arrays[f'gap{len(gaps)}/walk_position'] = positions[chunk].copy()
                gaps.append(g)
            runs_report.append(dict(component=comp['component'], run=run_id, visits=len(sources), closed=closed))
    counts = np.array([g['missing_walk_visits'] for g in gaps], np.int64)
    lengths = np.array([g['internal_length_mm'] for g in gaps], float)
    arrays.update(pixel_counts=counts, internal_length_mm=lengths)
    return dict(max_missing_walk_visits=int(counts.max()) if len(counts) else 0,
        max_internal_length_mm=float(lengths.max()) if len(lengths) else 0., gap_count=len(gaps),
        gaps=gaps, observed_runs=runs_report,
        definition='internal adjacent missing visits only; closure only fully observed loops; no excluded/component bridges'), arrays


def evaluate_object(prep, selection, geometry, oid):
    from visual.runtime.test.sampling_evaluation import CoverageMatcher
    c, r, t = prep['C'], prep['R'], selection['T']
    union = lambda d: np.union1d(d['pca'], d['sil'])
    stages = dict(X=prep['X'], C_pca=c['pca'], C_sil=c['sil'], C_both=np.intersect1d(c['pca'], c['sil']),
        C=union(c), R_pca=r['pca'], R_sil=r['sil'], R=union(r), T_pca=t['pca'], T_sil=t['sil'], T=union(t),
        P=selection['P'], S=selection['S'], F=selection['F'])
    common_cells = set(map(tuple, prep['groups']['pca']['cells'])) & set(map(tuple, prep['groups']['sil']['cells']))
    if len(selection['P']):
        keys = voxel.voxel_keys(prep['xyz'][source_rows(prep['X'], selection['P'])], prep['scale']['a_m'])
        _, multiplicity = np.unique(keys, axis=0, return_counts=True)
    else: multiplicity = np.empty(0, np.int64)
    counts = {name: len(v) for name, v in stages.items()}
    counts.update(pca_occupied_voxels=len(r['pca']), silhouette_occupied_voxels=len(r['sil']),
        common_occupied_voxels=len(common_cells), representative_source_intersection=len(np.intersect1d(r['pca'], r['sil'])),
        representative_source_union=len(stages['R']), selected_source_intersection=len(np.intersect1d(t['pca'], t['sil'])),
        structural_unique_sources=len(selection['P']), structural_points_in_multiple_source_voxels=int(multiplicity[multiplicity>1].sum()),
        structural_multiple_source_voxels=int((multiplicity>1).sum()), surface_shortfall=selection['shortfall'],
        surface_requested=selection['requested_surface'])
    arrays, metrics = {}, {}
    pairs = []
    for name in ('pca', 'sil'):
        pairs.extend([(f'C_{name}_to_R_{name}', c[name], r[name]),
                      (f'C_{name}_to_R_union', c[name], stages['R']),
                      (f'C_{name}_to_T_{name}', c[name], t[name]),
                      (f'C_{name}_to_P', c[name], selection['P']),
                      (f'C_{name}_to_F', c[name], selection['F'])])
    pairs.extend([('C_union_to_R_union', stages['C'], stages['R']), ('C_union_to_P', stages['C'], selection['P']),
                  ('C_union_to_F', stages['C'], selection['F']), ('X_to_F', prep['X'], selection['F'])])
    for name, query, target in pairs:
        d = distances(prep, query, target)
        arrays['distance_mm/'+name] = d
        arrays['distance_query_source/'+name] = query
        metrics[name] = distance_stats(d)
    uv = np.column_stack(np.unravel_index(c['sil'], prep['shape']))[:, ::-1]
    xyz = prep['xyz'][source_rows(prep['X'], c['sil'])]
    matcher = CoverageMatcher(geometry, oid, (uv, xyz), tolerances=((2., .003),))
    silhouette = {}
    for stage in ('P', 'F'):
        ids, matched = matcher.reference_hits(stages[stage])
        arrays[f'silhouette/{stage}/tolerance_reference_ids'] = ids
        arrays[f'silhouette/{stage}/matching_node_sources'] = matched
        exact_sources = np.intersect1d(c['sil'], stages[stage])
        tolerance_sources = c['sil'][ids]
        exact_gaps, ga = gap_metrics(prep['contours'], exact_sources, prep)
        arrays.update({f'silhouette/{stage}/exact_gaps/{k}': v for k, v in ga.items()})
        tolerance_gaps, ga = gap_metrics(prep['contours'], tolerance_sources, prep)
        arrays.update({f'silhouette/{stage}/tolerance_gaps/{k}': v for k, v in ga.items()})
        silhouette[stage] = dict(exact_unique=fraction(len(exact_sources), len(c['sil'])),
            tolerance_2px_AND_3mm=fraction(len(ids), len(c['sil'])), exact_missing=exact_gaps, tolerance_missing=tolerance_gaps,
            same_pair_count=int(matcher.adjacencies['2px_3mm'][source_rows(matcher.flat, stages[stage])].nnz))
    contour = prep['contours']
    raw = contour['raw_unique']
    invalid = np.unique(np.concatenate([v['walk'][v['invalid_depth']] for v in contour['components']])) if contour['components'] else np.empty(0, np.int64)
    overlap = np.unique(np.concatenate([v['walk'][v['overlap']] for v in contour['components']])) if contour['components'] else np.empty(0, np.int64)
    exclusions = dict(raw_outer_unique=len(raw), raw_walk_visits=contour['walk_count'], observed_unique=len(c['sil']),
        excluded_unique=len(raw)-len(c['sil']), invalid_depth_unique=len(invalid), overlap_unique=len(overlap),
        invalid_AND_overlap=len(np.intersect1d(invalid, overlap)), reason_counts_overlap=True,
        components=contour['canonical_component_count'], repeated_walk_visits=sum(v['repeated_source_visits'] for v in contour['components']),
        image_boundary_visits=sum(int(v['image_boundary'].sum()) for v in contour['components']))
    result = dict(counts=counts, distance_mm=metrics, silhouette=silhouette, contour_exclusions=exclusions,
        keep={name: fraction(len(t[name]), len(r[name])) for name in ('pca', 'sil')},
        structural_retention=dict(original_candidate_union=fraction(len(selection['P']), len(stages['C'])),
                                  representative_union=fraction(len(selection['P']), len(stages['R']))),
        silhouette_source_selection=fraction(len(t['sil']), len(c['sil'])),
        nominal=prep['scale'], same_cell={n: distance_stats(prep['groups'][n]['same_cell_distance_m']*1000) for n in ('pca', 'sil')},
        timings_ms=selection['timings_ms'])
    return result, arrays, stages


def read_gt(config, frozen_path, expected_sha):
    """The sole GT-reader gate; called only after all selection arrays are frozen."""
    require_equal(prior.sha(frozen_path), expected_sha, 'pre-GT selection hash')
    keys = ['base_camera/seg', 'peg_id', 'box_id', 'peg_pose', 'box_pose',
            'box_hole_pose', 'peg_half_sizes', 'box_hole_radii', 'seed']
    with np.load(config['input'], allow_pickle=False) as a:
        return {k: a[k] for k in keys}


def evaluate_after_freeze(output, digest, frames, geometries, prepared, selected, packets, config):
    from visual.runtime.test.sampling_evaluation import evaluation_context, staged_reference_coverage
    from visual.runtime.test.run_rim_bottleneck import geometry_evidence
    gt = read_gt(config, output/'selection_before_gt.npz', digest)
    results, arrays = {}, {}
    for frame_id in FRAMES:
        frame, geometry = frames[frame_id], geometries[frame_id]
        context = evaluation_context(frame, geometry_evidence(frame), gt, frame.camera, config['robot_ids'])
        with np.load(SOURCE/f'f{frame_id:04d}_samples.npz', allow_pickle=False) as saved:
            for oid, name in NAMES.items():
                for axis, value in zip(('uv', 'xyz'), context['references'][name]):
                    np.testing.assert_array_equal(value, saved[f'evaluation/{name}/reference_{axis}'])
                    arrays[f'f{frame_id:04d}/reference/{name}/{axis}'] = value
        for ratio, phase in ARMS:
            key = arm_key(frame_id, ratio, phase)
            objects = {}
            for oid, name in NAMES.items():
                row, data, stages = evaluate_object(prepared[frame_id][oid], selected[key][oid], geometry, oid)
                matcher = context['matchers'][name]
                gt_stages = {}
                hits = {}
                for stage, source in stages.items():
                    ids, matched = matcher.reference_hits(source)
                    hits[stage] = set(ids.tolist())
                    pair_count = matcher.adjacencies['2px_3mm'][source_rows(matcher.flat, source)].nnz
                    gt_stages[stage] = dict(unique_reference_ids=ids, matching_node_sources=matched,
                        reference_count=len(ids), matching_node_count=len(matched), same_pair_count=int(pair_count))
                    data[f'gt/{stage}/reference_ids'] = ids
                    data[f'gt/{stage}/matching_node_sources'] = matched
                row['gt_auxiliary'] = dict(stages=gt_stages,
                    preservation=staged_reference_coverage(matcher, stages['C'], stages['P'], stages['S'], stages['F']),
                    surface_restored_reference_ids=sorted(hits['S']-hits['P']),
                    surface_recovered_candidate_reference_ids=sorted((hits['C']-hits['P']) & hits['S']),
                    surface_structural_duplicate_reference_ids=sorted(hits['S'] & hits['P']))
                row['occupancy'] = occupancy(frame, stages, context['negatives'][name])
                packet = packets[key]
                rows = packet['object_id'] == oid
                row['features'] = dict(available=int(packet['provenance']['feature_available'][rows].sum()),
                    unavailable=int((~packet['provenance']['feature_available'][rows]).sum()),
                    fields={f['name']: dict(available=int(packet['provenance']['feature_field_available'][rows, j].sum()),
                        unavailable=int((~packet['provenance']['feature_field_available'][rows, j]).sum()),
                        stored_nonfinite_count=int((~np.isfinite(packet['compact_features'][f['name']]['values'][
                            np.isin(packet['compact_features'][f['name']]['positions'], np.flatnonzero(rows))])).sum()))
                        for j, f in enumerate(packet['feature_fields'])})
                objects[name] = row
                arrays.update({key+'/'+name+'/'+k: v for k, v in data.items()})
            n = len(packets[key]['points'])
            results[key] = dict(frame=frame_id, ratio=ratio, phase=phase, diagnostic=phase == 1,
                objects=objects, point_nodes=n, expected_graph_nodes_with_robots=n+3,
                edge_count=0, edge_status='not_implemented', packet_hash=packet_digest(packets[key]),
                theoretical_N_squared_float32_bytes=4*n*n,
                packet_array_nbytes=sum(a.nbytes for a in flatten_arrays(packets[key]).values()))
    return results, arrays


def aggregate_results(rows):
    aggregate = {}
    for ratio, phase in ARMS:
        selected = [rows[arm_key(f, ratio, phase)] for f in FRAMES]
        obj = {}
        for name in NAMES.values():
            metrics = {}
            values = [r['objects'][name] for r in selected]
            functions = dict(P=lambda r: r['counts']['P'], F=lambda r: r['counts']['F'],
                candidate_to_F_p95_mm=lambda r: r['distance_mm']['C_union_to_F']['p95'],
                X_to_F_p95_mm=lambda r: r['distance_mm']['X_to_F']['p95'],
                silhouette_exact_F=lambda r: r['silhouette']['F']['exact_unique']['value'],
                silhouette_tolerance_F=lambda r: r['silhouette']['F']['tolerance_2px_AND_3mm']['value'],
                missing_walk_visits_F=lambda r: r['silhouette']['F']['exact_missing']['max_missing_walk_visits'])
            for metric, getter in functions.items():
                v = [getter(r) for r in values]
                defined = [x for x in v if x is not None]
                metrics[metric] = dict(equal_frame_mean=float(np.mean(defined)) if defined else None,
                    defined_frame_count=len(defined), total_frame_count=3, per_frame=v)
            obj[name] = metrics
        aggregate[f'r{ratio:g}_phase{phase}'] = dict(point_nodes_equal_frame_mean=float(np.mean([r['point_nodes'] for r in selected])), objects=obj)
    return aggregate


def phase_diagnostics(rows, selected):
    out = {}
    for f in FRAMES:
        a, b = arm_key(f, .5, 0), arm_key(f, .5, 1)
        objects = {}
        for oid, name in NAMES.items():
            sa, sb = selected[a][oid], selected[b][oid]
            compare = {}
            for stage, x, y in [('T_pca', sa['T']['pca'], sb['T']['pca']), ('T_sil', sa['T']['sil'], sb['T']['sil']),
                                ('P', sa['P'], sb['P']), ('S', sa['S'], sb['S']), ('F', sa['F'], sb['F'])]:
                common, union = np.intersect1d(x, y), np.union1d(x, y)
                compare[stage] = dict(intersection=len(common), union=len(union), jaccard=fraction(len(common), len(union)),
                    phase0_count=len(x), phase1_count=len(y))
            ra, rb = rows[a]['objects'][name], rows[b]['objects'][name]
            compare['changes_phase1_minus_phase0'] = dict(
                silhouette_exact_F=rb['silhouette']['F']['exact_unique']['value']-ra['silhouette']['F']['exact_unique']['value']
                    if ra['silhouette']['F']['exact_unique']['value'] is not None else None,
                silhouette_tolerance_F=rb['silhouette']['F']['tolerance_2px_AND_3mm']['value']-ra['silhouette']['F']['tolerance_2px_AND_3mm']['value']
                    if ra['silhouette']['F']['tolerance_2px_AND_3mm']['value'] is not None else None,
                max_exact_missing_visits=rb['silhouette']['F']['exact_missing']['max_missing_walk_visits']-ra['silhouette']['F']['exact_missing']['max_missing_walk_visits'],
                max_tolerance_missing_visits=rb['silhouette']['F']['tolerance_missing']['max_missing_walk_visits']-ra['silhouette']['F']['tolerance_missing']['max_missing_walk_visits'])
            objects[name] = compare
        out[str(f)] = objects
    return out


def plot_results(output, frames, prepared, selected, rows):
    figures = []
    colors = dict(pca='#f59e0b', sil='#06b6d4')
    for frame_id in FRAMES:
        for ratio, phase in ARMS:
            key = arm_key(frame_id, ratio, phase)
            filename = key.replace('/', '_')+'.png'
            fig = plt.figure(figsize=(19, 9), layout='constrained')
            for oid, name in NAMES.items():
                prep, sel = prepared[frame_id][oid], selected[key][oid]
                for col in range(3):
                    ax = fig.add_subplot(2, 4, oid*4+col+1)
                    ax.imshow(frames[frame_id].rgb)
                    if col < 2:
                        data = prep['C'] if col == 0 else prep['R']
                        for path in ('pca', 'sil'):
                            v, u = np.divmod(data[path], prep['shape'][1])
                            ax.scatter(u, v, s=6 if col == 0 else 10, c=colors[path], label=f'{path}: {len(data[path])}', alpha=.8)
                        title = 'raw candidates' if col == 0 else 'independent voxel representatives'
                    else:
                        for path, sources, color, size in [('P', sel['P'], '#e11d48', 11), ('S', sel['S'], '#2563eb', 15)]:
                            v, u = np.divmod(sources, prep['shape'][1])
                            ax.scatter(u, v, s=size, c=color, label=f'{path}: {len(sources)}', alpha=.85)
                        title = 'Structural union + Surface'
                    ax.set_title(f'{name}: {title}'); ax.legend(fontsize=8, loc='upper right')
                    # Display raw object bounds with fixed padding, independent of metric outcomes.
                    raw = prep['contours']['raw_unique']
                    if len(raw):
                        v, u = np.divmod(raw, prep['shape'][1]); h, w = prep['shape']
                        ax.set_xlim(max(0, u.min()-12), min(w, u.max()+12)); ax.set_ylim(min(h, v.max()+12), max(0, v.min()-12))
                    ax.set_xlabel('u (px)'); ax.set_ylabel('v (px)')
                ax = fig.add_subplot(2, 4, oid*4+4, projection='3d')
                for stage, color in [('P', '#e11d48'), ('S', '#2563eb')]:
                    xyz = prep['xyz'][source_rows(prep['X'], sel[stage])]
                    ax.scatter(*xyz.T, s=7, c=color, label=stage)
                ax.set_title(name+': original CV XYZ (full extent)'); ax.set_xlabel('X m'); ax.set_ylabel('Y m'); ax.set_zlabel('Z m'); ax.legend(fontsize=8)
            fig.suptitle(f'ep000 base_camera frame {frame_id} | keep {ratio:g}, phase {phase} | N={rows[key]["point_nodes"]} | 3D depth tails retained')
            fig.savefig(output/filename, dpi=110); plt.close(fig); figures.append(filename)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    for oid, name in NAMES.items():
        for frame, color in zip(FRAMES, ('#c2410c', '#1d4ed8', '#047857')):
            main = [rows[arm_key(frame, r, 0)] for r in (1., .75, .5, .25)]
            n = [r['point_nodes'] for r in main]
            metrics = [[r['objects'][name]['silhouette']['F']['tolerance_2px_AND_3mm']['value'] for r in main],
                       [r['objects'][name]['distance_mm']['X_to_F']['p95'] for r in main],
                       [r['objects'][name]['silhouette']['F']['exact_missing']['max_missing_walk_visits'] for r in main]]
            for j, y in enumerate(metrics):
                axes[oid, j].plot(n, y, 'o-', color=color, label=f'frame {frame}')
                for x, value, ratio in zip(n, y, (1., .75, .5, .25)):
                    if value is not None: axes[oid, j].annotate(str(ratio), (x, value), fontsize=8)
        for j, title in enumerate(('silhouette same-pair coverage (F)', 'X to F p95 (mm)', 'max exact missing walk visits (F)')):
            axes[oid, j].set(title=name+': '+title, xlabel='Total Point N (both objects)'); axes[oid, j].grid(alpha=.2); axes[oid, j].legend()
    fig.savefig(output/'tradeoffs.png', dpi=130); plt.close(fig); figures.append('tradeoffs.png')
    return figures


def fmt(v):
    if v is None: return 'N/A'
    if isinstance(v, float): return f'{v:.4g}'
    return str(v)


def report_blocks(result):
    blocks = []
    p = lambda x: blocks.append(('p', x))
    h = lambda x: blocks.append(('h', x))
    table = lambda head, rows: blocks.append(('table', (head, rows)))
    h('실행 범위와 고정 조건')
    p(f"Run {result['run_id']}. 승인 계획의 최소 구현과 CPU offline 평가를 완료했다. ep000/base_camera source frame 0,45,80 전체에서 "
      'main ratio 1/.75/.5/.25 phase0와 .5 phase1 진단을 전부 실행했다. 결과에 따른 frame/ratio/phase 추가·교체·선택은 없다. '
      'Surface 추가 요청은 Peg97/Box95로 고정했으며 이번 비교의 실험 설정이다. 새로운 최종 정책으로 채택하지 않았다.')
    p('PCA5는 저장 optimized orientation-only strict >20° 후보·원본 source/XYZ를 사용한다. Raw tracker mask의 RETR_EXTERNAL/CHAIN_APPROX_NONE 외부 contour만 '
      '추출하고 canonical 시작점·방향·component·walk position·observed segment·invalid/overlap exclusion을 보존했다. Internal hole과 invalid-depth hole을 새 외부 contour로 만들지 않는다. '
      'Silhouette는 관측 mask 외곽이며 occlusion 외곽을 포함할 수 있다. Physical edge/crease 정답이 아니다. 추가 outlier 제거는 0개다.')
    p('두 경로는 물체별 동일 nominal metric grid에서 centroid-nearest 원본 대표점을 독립적으로 고른다. 동일 voxel의 서로 다른 source는 유지한다. '
      'Nominal scale은 PCA5 spatial support의 설계 proxy이며 silhouette localization uncertainty를 실측·보정한 값이 아니다. '
      'Representatives helper는 기존 Task03-I의 pure helper를 import했다. Same-cell bound는 voxel 압축에만 적용되며 thinning 이후 localization 상한이 아니다.')
    p('Thinning은 p/q rational floor(kq/p), count ceil(Mp/q) 규칙이다. PCA는 ascending 대표 source, silhouette는 canonical component walk의 대표 순서다. '
      '독립 component 사이에도 전역 목록 규칙을 이어 적용하며 quota는 없다. 이는 index 규칙으로 3D 거리·물리적 contour 길이 등간격을 보장하지 않는다. '
      'Phase1은 .5의 odd indices이며 wrap/refill하지 않는다. 홀수 M에서 count 차이를 그대로 보고한다.')
    p('P는 T_pca 먼저, 이어 새 T_sil source의 안정 합집합이다. 전체 X에 기존 3D Euclidean FPS를 P initial, count=len(P)+Surface_added로 호출한다. '
      'P가 비면 최소 X source를 첫 Surface로 세어 요청 개수 안에 포함한다. 256 전체 cap, 물체별 32 cap, structural_ratio=.25, Structural FPS는 실행하지 않았다. '
      '물체 0→1, 물체별 P→S 순서를 유지한다. Capacity 부족은 shortfall이며 padding·중복 source·새 좌표 생성은 없다.')
    h('관측된 trade-off')
    table(['source frame','Point N: ratio 1 / .75 / .5 / .25','ratio .5 silhouette tol F: Peg / Box','ratio .25 silhouette tol F: Peg / Box'],
        [[f, ' / '.join(str(result['rows'][arm_key(f,r,0)]['point_nodes']) for r in (1.,.75,.5,.25)),
          *[' / '.join(fmt(result['rows'][arm_key(f,r,0)]['objects'][name]['silhouette']['F']['tolerance_2px_AND_3mm']['value'])
              for name in ('peg','box')) for r in (.5,.25)]] for f in FRAMES])
    p('이 세 관측에서는 ratio가 작아질수록 Point N이 줄고 silhouette tolerance coverage와 최대 연속 누락이 악화되는 경향을 보였다. '
      'X→F p95 변화는 frame·물체마다 다르며 후보 coverage와 같은 비율로 변하지 않는다. .5의 phase 변화도 coverage와 최종 Surface 집합을 바꾼다. '
      '어느 ratio/phase도 최종 정책으로 선택하지 않았다. 아래에 모든 개별 결과와 고정 phase 진단을 함께 공개한다.')
    h('가변 Node 수와 GT-free coverage — 모든 frame/arm/object')
    p('표 거리 단위는 mm이며 p95는 NumPy linear다. Silhouette exact/tolerance 분모는 관측 가능한 원본 unique contour source다. '
      'Tolerance는 같은 reference-node pair에서 ≤2px AND ≤3mm를 만족한 비율이다. P와 F를 나누며 Surface가 후보를 다시 선택할 수 있다. '
      'Empty query는 분모0/N/A, nonempty query→empty target은 coverage0·distance unavailable로 저장한다. Undefined distance는 0으로 평균하지 않는다.')
    table(['frame','ratio','phase','object','C pca/sil/both','R pca/sil','T pca/sil','P','S','F','Point N / expected N+3',
           'C→F p95','X→F mean/p95/max','sil exact P/F','sil tol P/F','max missing P/F (visits)'],
        [[r['frame'],r['ratio'],r['phase'],name,
          '/'.join(str(row['counts'][k]) for k in ('C_pca','C_sil','C_both')),
          '/'.join(str(row['counts'][k]) for k in ('R_pca','R_sil')),
          '/'.join(str(row['counts'][k]) for k in ('T_pca','T_sil')),
          *[row['counts'][k] for k in ('P','S','F')],f"{r['point_nodes']} / {r['expected_graph_nodes_with_robots']}",
          fmt(row['distance_mm']['C_union_to_F']['p95']),
          '/'.join(fmt(row['distance_mm']['X_to_F'][k]) for k in ('mean','p95','max')),
          '/'.join(fmt(row['silhouette'][s]['exact_unique']['value']) for s in ('P','F')),
          '/'.join(fmt(row['silhouette'][s]['tolerance_2px_AND_3mm']['value']) for s in ('P','F')),
          '/'.join(str(row['silhouette'][s]['exact_missing']['max_missing_walk_visits']) for s in ('P','F'))]
         for r in result['rows'].values() for name,row in r['objects'].items()])
    blocks.append(('image', ('tradeoffs.png', 'Main phase0 전체 결과. 점 옆 숫자는 ratio이며 정책 선택을 뜻하지 않는다.')))
    h('독립 Voxel 예산 효율 진단')
    table(['frame','ratio','phase','object','PCA cells','sil cells','common cells','R source intersection','R source union','P unique',
           'P points in multi-source cells','multi-source cells','a (mm)','keep pca/sil','P/C union','P/R union'],
          [[r['frame'],r['ratio'],r['phase'],name,*[row['counts'][k] for k in ('pca_occupied_voxels','silhouette_occupied_voxels',
            'common_occupied_voxels','representative_source_intersection','representative_source_union','structural_unique_sources',
            'structural_points_in_multiple_source_voxels','structural_multiple_source_voxels')],fmt(row['nominal']['a_m']*1000) if row['nominal']['a_m'] else 'N/A',
            '/'.join(fmt(row['keep'][n]['value']) for n in ('pca','sil')),
            fmt(row['structural_retention']['original_candidate_union']['value']),fmt(row['structural_retention']['representative_union']['value'])]
           for r in result['rows'].values() for name,row in r['objects'].items()])
    p('공통 cell/대표 source 교집합과 최종 multi-source cell 수는 예산 효율 진단이다. 추가 voxel 병합이나 ratio/phase 선택 입력으로 사용하지 않았다. '
      '각 source의 candidate→cell→representative, capacity, same-cell distance, contour/representative 순서를 NPZ에 저장했다.')
    h('동일 가중 frame aggregate와 phase 민감도')
    p('각 frame을 동일 가중치로 평균한다. 각 metric에 defined_frame_count/total_frame_count=3과 per_frame 원값을 JSON에 저장했다. '
      '객체·frame의 점 수를 합쳐 분모를 바꾸지 않는다. 아래 평균은 phase 진단을 포함해 모든 사전 고정 arm을 공개한다.')
    table(['arm','object','mean Point N','mean P','mean F','mean X→F p95','mean silhouette tol F','defined frames'],
        [[arm,name,fmt(a['point_nodes_equal_frame_mean']),fmt(o['P']['equal_frame_mean']),fmt(o['F']['equal_frame_mean']),
          fmt(o['X_to_F_p95_mm']['equal_frame_mean']),fmt(o['silhouette_tolerance_F']['equal_frame_mean']),
          o['silhouette_tolerance_F']['defined_frame_count']] for arm,a in result['aggregate'].items() for name,o in a['objects'].items()])
    table(['frame','object','T_pca Jaccard','T_sil Jaccard','P Jaccard','S Jaccard','F Jaccard','Δ exact F','Δ tol F','Δ max exact missing'],
        [[f,name,*[fmt(o[k]['jaccard']['value']) for k in ('T_pca','T_sil','P','S','F')],
          *[fmt(o['changes_phase1_minus_phase0'][k]) for k in ('silhouette_exact_F','silhouette_tolerance_F','max_exact_missing_visits')]]
         for f,objects in result['phase_diagnostic'].items() for name,o in objects.items()])
    h('Contour 누락·관측 제외·깊이 tail')
    p('연속 누락은 원본 canonical walk의 유효 관측 run별로 계산한다. Pixel 수는 missing walk visit 수이며 반복 source 재방문은 유지한다. '
      '길이(mm)는 연속 누락 관측점 사이 원본 CV XYZ 인접 거리 합이다. 보존 endpoint까지의 거리는 넣지 않는다. 완전 관측 closed loop만 closure를 잇고, '
      'component/invalid/overlap 틈은 합산하지 않는다. 각 gap의 source/position·분포·최대값과 tolerance-missing도 JSON/NPZ에 보존했다. '
      '이 길이는 물체의 실제 둘레가 아니다. Raw contour 내부 hole은 후보에서 제외되며 경계 truncation flag는 유지한다.')
    table(['frame','object','raw unique / walk','observed','excluded unique','invalid / overlap / both','components / revisits','border visits','X depth median/p95/max m'],
        [[f,name,f"{e['raw_outer_unique']} / {e['raw_walk_visits']}",e['observed_unique'],e['excluded_unique'],
          f"{e['invalid_depth_unique']} / {e['overlap_unique']} / {e['invalid_AND_overlap']}",f"{e['components']} / {e['repeated_walk_visits']}",e['image_boundary_visits'],
          '/'.join(fmt(row['occupancy']['depth_m']['X'][k]) for k in ('median','p95','max'))]
          for f in FRAMES for name,row in result['rows'][arm_key(f,1.,0)]['objects'].items() for e in [row['contour_exclusions']]])
    p('큰 finite-positive depth는 임의 제거하지 않았다. 전체 X→F raw distance tail/max와 X 고정 median/p95 bands의 각 단계 점유를 JSON/NPZ에 공개한다. '
      'ROI 점유는 기존 occupancy 함수를 선택 완료 후에만 적용했다. 후보/대표/선택의 source별 및 union, P/S/F를 구분한다.')
    h('Peg seam / Box rim 보조 GT 평가')
    p('모든 arm을 freeze한 뒤 기존 physical reference evaluator를 호출했다. 세 frame의 reference UV/XYZ가 기존 Task03 저장 reference와 exact 일치했다. '
      'Same-pair ≤2px AND ≤3mm의 visible/observed/unique reference 수, matching node 수와 pair 수를 구분한다. '
      'C_pca/C_sil/both/union, source별 R/T, P/S/F의 reference ID와 손실·Surface 복구·중복을 저장했다. GT는 선택 목적함수로 사용하지 않았다.')
    table(['frame','ratio','phase','object','visible / observed','P refs','F refs','F matching nodes / pairs','Surface restored refs'],
        [[r['frame'],r['ratio'],r['phase'],name,
          f"{row['gt_auxiliary']['preservation']['coverage']['observation']['denominator']} / {row['gt_auxiliary']['preservation']['coverage']['observation']['numerator']}",
          row['gt_auxiliary']['stages']['P']['reference_count'],row['gt_auxiliary']['stages']['F']['reference_count'],
          f"{row['gt_auxiliary']['stages']['F']['matching_node_count']} / {row['gt_auxiliary']['stages']['F']['same_pair_count']}",
          len(row['gt_auxiliary']['surface_restored_reference_ids'])] for r in result['rows'].values() for name,row in r['objects'].items()])
    h('Provenance와 opt-in Graph 수신 계약')
    p('graph/vision_input.py의 receive_vision_packet은 points/object_id/geometric_features/edge/provenance를 검증하고 row를 그대로 복사한다. '
      'packet identity는 episode/camera/source_frame/coordinate_frame이다. provenance는 source_index:int64[N], object_id:int64[N], '
      'candidate_evidence:bool[N,2] 열 순서 PCA,sil, both_evidence:bool[N], selected_by_structural_source:bool[N,2], '
      'selection_stage:uint8[N] 값 0 Structural/1 Surface, surface_selected:bool[N], feature_available:bool[N], feature_field_available:bool[N,23]이다. '
      'Evidence는 원본 후보 membership이고 selection role과 독립이다. 다른 voxel member의 evidence를 대표 node에 합성하지 않는다.')
    p('원본 S3 조건에 저장된 23 metadata field만 source exact join했다. compact values+positions+원본 dtype와 명시적 schema의 float64[N,F] audit view를 함께 보존한다. '
      'Unavailable은 NaN/false mask이며 저장 field 자체의 invalid normal/score NaN과 구분한다. Availability는 저장 row의 존재이지 feature validity 보장이 아니다. '
      '완성된 GNN encoding이 아니며 provenance를 기존 GNN x에 concatenate하지 않는다. 실제 audit packet은 edge [2,0], edge_status=not_implemented다.')
    table(['frame','ratio','phase','object','stored feature rows / unavailable','packet array bytes','Point N','expected N+3'],
        [[r['frame'],r['ratio'],r['phase'],name,f"{row['features']['available']} / {row['features']['unavailable']}",
          r['packet_array_nbytes'],r['point_nodes'],r['expected_graph_nodes_with_robots']] for r in result['rows'].values() for name,row in r['objects'].items()])
    p('Robot-first 테스트에서는 supplied TCP/left/right를 0/1/2, point i를 3+i로 보존한다. point_node_index/node_kind/robot_mask를 제공하고 robot source=-1/evidence=false다. '
      'Supplied edge만 +3 remap하며 입력 edge는 바꾸지 않는다. 실관측 robot 좌표는 수집하지 않았다. N+3은 계약상 예상 node 수다. '
      '실제 topology/EE 좌표 결합/GNN end-to-end/BC/stale cache는 미구현이며 수신 container 통과와 구분한다.')
    h('시간·메모리·검증·재현')
    p(f"필수 CPU 테스트: {result['tests_summary']}. 선택 전 관측 alignment, source/XYZ, independent voxel, rational thinning, cap 부재, "
      'Surface initial/count/shortfall, provenance/feature join, 가변N/N=0 수신·supplied edge offset, GT-reader sentinel을 검증했다. '
      '성능 우월성이나 ratio 채택은 PASS 조건이 아니다. JSON/NPZ exact readback·PNG decode·Markdown↔HTML 상대 링크를 검증했다. '
      '실제 browser 표시 여부는 verification.json의 별도 항목을 따른다.')
    p(f"선택 전 freeze SHA-256 {result['selection_before_gt_sha256']}. 선택 후 selector 재호출 없이 GT 평가했고 종료 해시 불변을 확인했다. "
      f"기존 파일 {result['protected_file_count']}개의 SHA-256와 HEAD를 보호했다. 시작 dirty 상태·diff·소스 snapshot/hash·실제 dependency version·입력 hash를 같은 UUID에 저장했다. "
      'ExperimentLogger(team=visual, job_type=evaluate, seed=0), disabled W&B와 기존 artifact manifest/link_output을 재사용했다. 온라인 sync/publisher/upload/commit은 실행하지 않았다.')
    p('보호 manifest의 OS 메타데이터 예외는 experiments/.DS_Store 한 개다. Finder 변경을 UUID/선택 실행 전에 발견했고 초기·변경 해시와 제외 사유를 보존했다. '
      '기존 source/run/sidecar를 이 예외로 제외하지 않았다. 수치 평가 후 최초 보고서 링크 검증은 shape 표기가 링크로 해석되어 실패했다. '
      '같은 UUID의 freeze·수치 JSON/NPZ·PNG·test log를 유지한 report-only 복구에서 표기를 수정하고 링크·readback·보호 해시를 재검증했다. '
      '원래 실행 source_snapshot과 보고용 report_source_snapshot을 구분한다. attempt_failure.json에 최초 실패를 보존하며 새 선택·GT 평가·UUID는 만들지 않았다.')
    table(['frame','object','contour ms','nominal ms','voxel ms'],[[f,NAMES[o],*[fmt(t[k]) for k in ('contour','nominal_scale','voxel')]]
        for f,v in result['preprocessing_ms'].items() for o,t in [(int(o),t) for o,t in v.items()]])
    table(['frame','ratio','phase','object','thinning ms','union ms','Surface FPS ms','packet + receiver ms (both objects)','N² float32 bytes (theory)'],
        [[r['frame'],r['ratio'],r['phase'],name,*[fmt(row['timings_ms'][k]) for k in ('thinning','union','surface_fps')],
          fmt(result['packet_ms'][key]),r['theoretical_N_squared_float32_bytes']] for key,r in result['rows'].items() for name,row in r['objects'].items()])
    p('공통 contour/scale/voxel은 frame/object당 1회, thinning/union/Surface/packet은 arm별 비용이다. '
      'I/O/GT·evaluation/plot·HTML 등 wall time과 NumPy payload nbytes를 JSON에 구분한다. Peak RSS는 resource.getrusage(RUSAGE_SELF).ru_maxrss이며 '
      f"이 host의 단위는 bytes다. 최종 process 누적 peak는 verification.json에 기록하며 ratio별 peak로 비교하지 않는다. "
      'N² float32 bytes는 이론값이며 실제 Graph/GNN latency·GPU memory·20Hz 가능 여부를 측정하지 않았다. 실제 edge E=0는 미구현 상태다.')
    p('Task03-I/Task03 S3는 historical reference이며 같은 node budget 비교가 아니다. 이번 결과는 한 episode의 세 저장 관측 feasibility다. '
      'Simulator/renderer/CUDA/MPS/GPU/PCL은 이번 host에서 실행하지 않았다. 새 detector/normal/edge topology/모델 학습/추가 episode/파라미터 최적화/의존성 설치는 범위 밖이다. '
      '최종 sampling 정책과 architecture는 확정하지 않았다.')
    h('동일 시점 overlay와 3D — 모든 사전 고정 arm')
    for filename in result['figures']:
        if filename != 'tradeoffs.png': blocks.append(('image', (filename, filename.removesuffix('.png')+' — 원본 후보 / source별 R / P+S / 원본 CV XYZ')))
    return blocks


def write_reports(output, result):
    artifacts = [STEM+'.html', STEM+'.json', STEM+'.npz', 'preregistered_config.json',
        'selection_before_gt.npz', 'selection_before_gt.json', 'verification.json', 'tests.log',
        'source_hashes.json', 'protected_baseline.json', 'implementation.patch', 'git_start.diff']
    relative = '../../experiments/'+output.name+'/'
    md = ['# Task03-J — Variable-Rate Shape-First Sampling', '',
        ' · '.join(f'[{f}]({relative+f})' for f in artifacts), '',
        '[승인 계획](03j_variable_rate_shape_first_plan.md) · [Task03-I historical reference](03i_voxel_structural_sampling.md)', '']
    page = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Task03-J — Variable-Rate Shape-First Sampling</title>',
        '<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1600px;margin:auto;padding:28px;color:#182333;background:#f4f7fa}h1,h2{line-height:1.3}h2{margin-top:2.5em}nav{background:#e0f2fe;padding:18px}a{color:#075ea8}table{border-collapse:collapse;width:100%;background:white;font-size:13px}td,th{border:1px solid #b8c4d2;padding:7px;text-align:left;white-space:nowrap}th{background:#e2e8f0}tr:nth-child(even){background:#f1f5f9}.scroll{overflow:auto}img{max-width:100%;background:white}figure{margin:28px 0}figcaption{color:#475569}p{max-width:1250px}details{background:white;padding:12px;margin:15px 0}pre{white-space:pre-wrap;font-size:12px}</style>',
        '<h1>Task03-J — Variable-Rate Shape-First Sampling</h1>',
        f'<nav>Run {output.name}<br><a href="../../docs/visual-graph-integration/03j_variable_rate_shape_first.md">결과 Markdown</a> · '
        '<a href="../../docs/visual-graph-integration/03j_variable_rate_shape_first_plan.md">승인 계획</a><br>'+ ' · '.join(f'<a href="{f}">{f}</a>' for f in artifacts if not f.endswith('.html'))+'</nav>']
    for kind, value in report_blocks(result):
        if kind == 'h': md.extend(['## '+value, '']); page.append('<h2>'+html.escape(value)+'</h2>')
        elif kind == 'p': md.extend([value, '']); page.append('<p>'+html.escape(value)+'</p>')
        elif kind == 'image':
            filename, caption = value
            md.extend([f'![{caption}]({relative+filename})', ''])
            page.append(f'<figure><a href="{filename}"><img loading="lazy" src="{filename}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>')
        else:
            headers, rows = value
            escape = lambda x: str(x).replace('|', '/').replace('\n', ' ')
            md.extend(['| '+' | '.join(map(escape, headers))+' |', '| '+' | '.join(['---']*len(headers))+' |'])
            md.extend('| '+' | '.join(map(escape, row))+' |' for row in rows); md.append('')
            page.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+
                '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>')
    page.append('<h2>전체 condition별 상세 수치</h2>')
    for key, row in result['rows'].items():
        page.append('<details><summary>'+html.escape(key)+' — source별 거리, contour gaps, GT ID, ROI/depth, feature availability</summary><pre>'+html.escape(json.dumps(voxel.serial(row), ensure_ascii=False, indent=2))+'</pre></details>')
    page.append('</html>')
    REPORT.write_text('\n'.join(md))
    (output/(STEM+'.html')).write_text('\n'.join(page))


def verify_documents(output, figures):
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.targets = []
        def handle_starttag(self, tag, attrs): self.targets.extend(v for k, v in attrs if k in ('href', 'src'))
    parser = Links(); page = (output/(STEM+'.html')).read_text(); parser.feed(page)
    if re.search(r'<script|\bfetch\s*\(|https?://', page): raise AssertionError('non-static HTML')
    for target in parser.targets:
        if Path(target).is_absolute() or not (output/target).is_file(): raise AssertionError('bad HTML link '+target)
    links = re.findall(r'\]\(([^)]+)\)', REPORT.read_text())
    for target in links:
        if Path(target).is_absolute() or not (REPORT.parent/target).is_file(): raise AssertionError('bad Markdown link '+target)
    decoded = {}
    for filename in figures:
        a = plt.imread(output/filename)
        if min(a.shape[:2]) < 400 or not np.isfinite(a).all() or a.std() < .01: raise AssertionError('bad PNG '+filename)
        decoded[filename] = list(a.shape)
    if not any((REPORT.parent/t).resolve() == (output/(STEM+'.html')).resolve() for t in links):
        raise AssertionError('missing Markdown→HTML link')
    if not any((output/t).resolve() == REPORT.resolve() for t in parser.targets):
        raise AssertionError('missing HTML→Markdown link')
    return dict(status='PASS', html_relative_links=len(parser.targets), markdown_relative_links=len(links),
        png_decoded=decoded, static_no_script_no_fetch_no_cdn=True, bidirectional_report_links=True)


def preserved(baseline):
    current = {path: prior.sha(REPO/path) for path in baseline['hashes']}
    changed = [p for p in current if current[p] != baseline['hashes'][p]]
    if changed: raise ValueError('protected source/artifact hashes changed: '+', '.join(changed))
    require_equal(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(), baseline['head'], 'HEAD')
    return dict(status='PASS', protected_file_count=len(current), all_protected_sha256_unchanged=True, HEAD_unchanged=True,
        non_source_OS_metadata_exclusions=baseline.get('non_source_OS_metadata_exclusions', []))


def run(baseline_path):
    if REPORT.exists(): raise FileExistsError('refusing to overwrite existing 03-J report')
    if os.environ.get('WANDB_MODE') != 'disabled': raise ValueError('WANDB_MODE=disabled required')
    baseline = json.loads(baseline_path.read_text()); preserved(baseline)
    sources = sorted(set([Path(__file__), REPO/'graph/vision_input.py', PLAN,
        *[p for p in (REPO/'visual/runtime').rglob('*.py')],
        REPO/'utils/experiment_logger.py', REPO/'utils/experiment_artifacts.py']))
    source_hashes = prior.hashes(sources)
    config = dict(task='03-J', rules=RULES, source_sha256=source_hashes,
        baseline_manifest_sha256=prior.sha(baseline_path), source_run_id=prior.SOURCE_RUN,
        historical_voxel_run_id='92f29716e76f4b998fe924f08e80357d',
        dependencies=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                          opencv=cv2.__version__, matplotlib=matplotlib.__version__),
        platform=platform.platform(), device='cpu', simulator_executed=False, gpu_executed=False)
    with ExperimentLogger(team='visual', config=config, seed=0, model='task03j_variable_rate_shape_first', job_type='evaluate') as logger:
        output = logger.directory
        print('Run:', output, flush=True)
        try:
            write_json(output/'preregistered_config.json', config)
            config_sha = prior.sha(output/'preregistered_config.json')
            shutil.copyfile(baseline_path, output/'protected_baseline.json')
            (output/'git_start.diff').write_text(baseline['git_diff'])
            write_json(output/'git_start.json', dict(head=baseline['head'], status=baseline['git_status']))
            for path in sources:
                dest = output/'source_snapshot'/path.relative_to(REPO); dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
            with (output/'tests.log').open('w') as log:
                test = subprocess.run([sys.executable, '-m', 'unittest', *TESTS, '-v'], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if test.returncode: raise RuntimeError('contracts failed; no fixed sweep selection executed')
            test_text = (output/'tests.log').read_text()
            summary = next(s for s in test_text.splitlines() if s.startswith('Ran '))+'; '+test_text.splitlines()[-1]
            print(summary, flush=True)
            t0 = perf_counter()
            frames, geometries, prepared, cases, input_hashes, old_config = load_observations()
            load_ms = (perf_counter()-t0)*1000
            t0 = perf_counter()
            selected, packets, packet_ms = select_all(prepared, cases)
            selection_ms = (perf_counter()-t0)*1000
            t0 = perf_counter()
            digest, frozen_arrays = freeze(output, prepared, selected, packets)
            freeze_ms = (perf_counter()-t0)*1000
            print('All 15 frame-conditions frozen before GT:', digest, flush=True)
            logger.record(input_sha256=input_hashes, selection_before_gt_sha256=digest,
                          packet_hashes={k: packet_digest(p) for k,p in packets.items()})
            t0 = perf_counter()
            rows, arrays = evaluate_after_freeze(output, digest, frames, geometries, prepared, selected, packets, old_config)
            evaluation_ms = (perf_counter()-t0)*1000
            print('GT-free and auxiliary GT evaluation complete', flush=True)
            result = dict(run_id=logger.run_id, config=config, input_sha256=input_hashes, rows=rows,
                aggregate=aggregate_results(rows), phase_diagnostic=phase_diagnostics(rows, selected),
                tests_summary=summary, selection_before_gt_sha256=digest,
                protected_file_count=len(baseline['hashes']), packet_ms=packet_ms,
                preprocessing_ms={str(f): {str(o): p['timings_ms'] for o,p in v.items()} for f,v in prepared.items()},
                timing_ms=dict(observation_load_and_common_preprocessing=load_ms, common_preprocessing=sum(sum(p['timings_ms'].values()) for v in prepared.values() for p in v.values()),
                    selection_all=selection_ms, freeze_io=freeze_ms, gt_and_evaluation=evaluation_ms),
                memory=dict(input_and_common_array_payload_bytes=sum(v.nbytes for v in flatten_arrays(prepared).values()),
                    selection_array_payload_bytes=sum(v.nbytes for v in flatten_arrays(selected).values()),
                    frozen_array_payload_bytes=sum(v.nbytes for v in frozen_arrays.values()),
                    rss_method='resource.getrusage(RUSAGE_SELF).ru_maxrss; process cumulative; macOS bytes'),
                policy_adopted=False, feature_schema=next(iter(packets.values()))['feature_fields'])
            arrays.update(frozen_arrays)
            result['evaluation_npz_sha256'] = save_readback(output/(STEM+'.npz'), arrays)
            t0 = perf_counter()
            result['figures'] = plot_results(output, frames, prepared, selected, rows)
            result['timing_ms']['plots'] = (perf_counter()-t0)*1000
            write_json(output/(STEM+'.json'), result)
            # All report links exist before their resolution checks.
            write_json(output/'verification.json', dict(status='report checks pending'))
            patch = ''
            for path in (Path(__file__), REPO/'graph/vision_input.py', Path(__file__).with_name('test_variable_rate_shape_first.py'),
                         Path(__file__).with_name('test_variable_rate_graph_contract.py')):
                patch += subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', str(path.relative_to(REPO))], cwd=REPO, capture_output=True, text=True).stdout
            (output/'implementation.patch').write_text(patch)
            preservation = preserved(baseline)
            write_json(output/'source_hashes.json', dict(executed_sources=source_hashes, executed_sources_after=prior.hashes(sources), input_hashes=input_hashes, preservation=preservation))
            t0 = perf_counter(); write_reports(output, result)
            result['timing_ms']['markdown_and_html'] = (perf_counter()-t0)*1000
            result['memory']['process_peak_rss_bytes_after_plots'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            write_json(output/(STEM+'.json'), result)
            documents = verify_documents(output, result['figures'])
            require_equal(prior.sha(output/'selection_before_gt.npz'), digest, 'selection immutable after GT')
            require_equal(prior.hashes(sources), source_hashes, 'executed source immutable')
            require_equal(prior.hashes([REPO/k for k in input_hashes]), input_hashes, 'inputs immutable')
            require_equal(prior.sha(output/'preregistered_config.json'), config_sha, 'preregistered config immutable')
            for key,p in packets.items(): require_equal(packet_digest(p), rows[key]['packet_hash'], 'packet unchanged')
            saved = json.loads((output/(STEM+'.json')).read_text())
            require_equal(saved['run_id'], logger.run_id, 'JSON readback identity')
            skipped = sum(' ... skipped' in s for s in test_text.splitlines())
            count = int(re.search(r'Ran (\d+) tests', summary).group(1))
            verification = dict(status='PASS for executed checks', tests=dict(summary=summary, modules=TESTS, passed=count-skipped, skipped=skipped),
                documents=documents, source_preservation=preservation, input_exact_alignment=True,
                structural_and_surface_counts_exact=True, original_source_XYZ_exact=True, packet_receiver_hashes_exact=True,
                all_15_conditions_saved_before_GT=True, pre_GT_npz_sha256=digest, pre_GT_json_sha256=prior.sha(output/'selection_before_gt.json'),
                selection_hash_unchanged=True, config_hash_unchanged=True, NPZ_exact_readback=True, JSON_readback=True,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                browser_display=dict(status='NOT_EXECUTED', reason='static file link and PNG decode checks executed; actual browser display separate'),
                simulator_executed=False, renderer_executed=False, CUDA_MPS_GPU_executed=False,
                existing_GNN_end_to_end=False, edge_status='not_implemented', policy_adopted=False,
                performance_is_PASS_condition=False, external_upload=False)
            write_json(output/'verification.json', verification)
            write_json(output/'git_end.json', dict(status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),preservation=preserved(baseline)))
            logger.log({key: dict(point_nodes=r['point_nodes'], objects={n: dict(P=o['counts']['P'], F=o['counts']['F'],
                candidate_p95=o['distance_mm']['C_union_to_F']['p95'], surface_p95=o['distance_mm']['X_to_F']['p95']) for n,o in r['objects'].items()})
                for key,r in rows.items()}, step=0, prefix='eval')
            for path in sorted(output.rglob('*')):
                if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','metrics.json','config.yaml'):
                    logger.link_output(path, kind='metrics' if path.name == STEM+'.json' else 'reports')
            logger.link_output(REPORT, kind='reports')
            print('Completed:', output, flush=True)
        except Exception as exc:
            write_json(output/'attempt_failure.json', dict(error_type=type(exc).__name__, message=str(exc), selection_frozen=(output/'selection_before_gt.npz').exists()))
            raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--finalize-saved', type=Path, help='Report-only recovery; no selection, GT, new UUID or tests')
    args = parser.parse_args()
    if args.finalize_saved: finalize_saved(args.finalize_saved, args.baseline)
    else: run(args.baseline)


def read_saved_packet(archive, metadata, key):
    m = metadata['packets'][key]
    prefix = 'packets/'+key+'/'
    packet = {k: m[k] for k in ('episode','camera','source_frame','coordinate_frame','feature_fields','edge_status')}
    for name in ('points','object_id','geometric_features','intra_object_edge_index'):
        packet[name] = archive[prefix+name].copy()
    packet['provenance'] = {name.removeprefix(prefix+'provenance/'): archive[name].copy()
                            for name in archive.files if name.startswith(prefix+'provenance/')}
    packet['compact_features'] = {f['name']: {n: archive[prefix+'compact_features/'+f['name']+'/'+n].copy()
        for n in ('positions','values')} for f in packet['feature_fields']}
    return packet


def finalize_saved(output, baseline_path):
    """Recover only report generation after a completed, frozen numerical sweep.

    Never instantiate another logger/UUID and never invoke selection/evaluation.
    Local scalar-event recovery uses the shared serializer; no W&B transport.
    """
    from datetime import datetime, timezone
    from utils.experiment_artifacts import register_existing_output
    from utils.experiment_logger import scalars, write
    output = output.resolve()
    result = json.loads((output/(STEM+'.json')).read_text())
    require_equal(output.name, result['run_id'], 'saved run identity')
    baseline = json.loads(baseline_path.read_text()); preservation = preserved(baseline)
    immutable = [output/(STEM+'.json'), output/(STEM+'.npz'), output/'selection_before_gt.npz',
        output/'selection_before_gt.json', output/'preregistered_config.json', output/'tests.log', *[output/f for f in result['figures']]]
    before = {p.name: prior.sha(p) for p in immutable}
    require_equal(before['selection_before_gt.npz'], result['selection_before_gt_sha256'], 'frozen selection hash')
    require_equal(before[STEM+'.npz'], result['evaluation_npz_sha256'], 'saved numerical archive hash')
    require_equal(json.loads((output/'preregistered_config.json').read_text()), result['config'], 'saved config')
    for path, digest in result['config']['source_sha256'].items():
        require_equal(prior.sha(output/'source_snapshot'/path), digest, 'executed source snapshot '+path)
    require_equal(prior.hashes([REPO/k for k in result['input_sha256']]), result['input_sha256'], 'saved input hashes')
    freeze_metadata = json.loads((output/'selection_before_gt.json').read_text())
    packet_hashes = {}
    with np.load(output/'selection_before_gt.npz', allow_pickle=False) as frozen, np.load(output/(STEM+'.npz'), allow_pickle=False) as evaluated:
        for key in frozen.files: np.testing.assert_array_equal(frozen[key], evaluated[key], err_msg=key)
        for key, row in result['rows'].items():
            packet = read_saved_packet(frozen, freeze_metadata, key)
            received = receive_vision_packet(packet)
            digest = packet_digest(packet)
            require_equal(digest, freeze_metadata['packets'][key]['hash'], 'saved packet freeze hash')
            require_equal(digest, row['packet_hash'], 'saved packet result hash')
            require_equal(digest, packet_digest(received.packet), 'readback receiver hash')
            packet_hashes[key] = digest
            for oid in (0,1):
                prefix = 'selections/'+key+'/'+str(oid)+'/'
                p,s,f = [frozen[prefix+n] for n in ('P','S','F')]
                tp,ts = [frozen[prefix+'T/'+n] for n in ('pca','sil')]
                np.testing.assert_array_equal(p, stable_unique(np.r_[tp,ts]))
                np.testing.assert_array_equal(f, np.r_[p,s])
                prep = f'observations/{row["frame"]}/{oid}/'
                x,xyz = frozen[prep+'X'], frozen[prep+'xyz']
                np.testing.assert_array_equal(frozen[prefix+'xyz'], xyz[source_rows(x,f)])
                require_equal(len(s), min(SURFACE[oid], len(x)-len(p)), 'saved Surface count')
                require_equal(len(np.unique(f)),len(f),'saved unique sources')
    # Retain original numerical sources and expose report-only changes separately.
    current = [Path(__file__), REPO/'graph/vision_input.py', Path(__file__).with_name('test_variable_rate_shape_first.py'),
               Path(__file__).with_name('test_variable_rate_graph_contract.py')]
    patch_text = ''
    for path in current:
        dest = output/'report_source_snapshot'/path.relative_to(REPO); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path,dest)
        patch_text += subprocess.run(['git','diff','--no-index','--',str(output/'source_snapshot'/path.relative_to(REPO)),str(path)],cwd=REPO,capture_output=True,text=True).stdout
    (output/'report_refinement.patch').write_text(patch_text)
    write_reports(output,result)
    documents = verify_documents(output,result['figures'])
    after = {p.name:prior.sha(p) for p in immutable}; require_equal(after,before,'report-only immutable artifacts')
    tests = (output/'tests.log').read_text(); count = int(re.search(r'Ran (\d+) tests',result['tests_summary']).group(1))
    skipped = sum(' ... skipped' in line for line in tests.splitlines())
    verification = dict(status='PASS for executed checks', tests=dict(summary=result['tests_summary'],modules=TESTS,passed=count-skipped,skipped=skipped),
        documents=documents, source_preservation=preservation, input_exact_alignment=True,
        structural_and_surface_counts_exact=True, original_source_XYZ_exact=True, packet_receiver_hashes_exact=packet_hashes,
        all_15_conditions_saved_before_GT=True, pre_GT_npz_sha256=result['selection_before_gt_sha256'],
        pre_GT_json_sha256=before['selection_before_gt.json'], selection_hash_unchanged=True, config_hash_unchanged=True,
        NPZ_exact_readback=True, JSON_readback=True, peak_rss_bytes=result['memory']['process_peak_rss_bytes_after_plots'],
        browser_display=dict(status='NOT_EXECUTED',reason='actual browser display not run; relative-link checks, PNG decode and visual PNG inspection are separate'),
        visual_PNG_inspection=['tradeoffs.png'], simulator_executed=False,renderer_executed=False,CUDA_MPS_GPU_executed=False,
        existing_GNN_end_to_end=False,edge_status='not_implemented',policy_adopted=False,performance_is_PASS_condition=False,external_upload=False,
        report_only_recovery=dict(new_UUID=False,new_selection=False,new_GT_evaluation=False,numerical_artifacts_unchanged=True,
            initial_failure='attempt_failure.json', original_source='source_snapshot/', report_source='report_source_snapshot/',
            local_scalar_recovery='shared experiment_logger.scalars/write from saved numerical rows; no W&B transport'))
    write_json(output/'verification.json',verification)
    hashes = json.loads((output/'source_hashes.json').read_text())
    hashes.update(report_generation_sources=prior.hashes(current),numerical_artifacts_before_after=dict(before=before,after=after),final_preservation=preserved(baseline))
    write_json(output/'source_hashes.json',hashes)
    metadata = json.loads((output/'metadata.json').read_text())
    # The failed report attempt remains explicit; identity/seed/project/group stay intact.
    metadata['initial_report_failure'] = dict(status=metadata['status'], exception_type=metadata.get('exception_type'),
                                            finished_at=metadata.get('finished_at'), evidence='attempt_failure.json')
    metadata.update(status='finished', finished_at=datetime.now(timezone.utc).isoformat(), report_only_recovery=True)
    metadata.pop('exception_type',None)
    write(output/'metadata.json',metadata)
    metrics = json.loads((output/'metrics.json').read_text())
    if not metrics:
        values = {key: dict(point_nodes=r['point_nodes'],objects={n:dict(P=o['counts']['P'],F=o['counts']['F'],
            candidate_p95=o['distance_mm']['C_union_to_F']['p95'],surface_p95=o['distance_mm']['X_to_F']['p95']) for n,o in r['objects'].items()})
            for key,r in result['rows'].items()}
        write(output/'metrics.json',[dict(run_id=output.name,step=0,metrics=scalars(values,'eval'))])
    write_json(output/'git_end.json',dict(status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),preservation=preserved(baseline)))
    for path in sorted(output.rglob('*')):
        if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','metrics.json','config.yaml'):
            register_existing_output(output,path,kind='metrics' if path.name==STEM+'.json' else 'reports')
    register_existing_output(output,REPORT,kind='reports')
    print(json.dumps(dict(run_id=output.name,status=verification['status'],tests=verification['tests'],documents=documents,
        no_new_numerical_work=True,preservation=verification['source_preservation']),ensure_ascii=False))


if __name__ == '__main__': main()
