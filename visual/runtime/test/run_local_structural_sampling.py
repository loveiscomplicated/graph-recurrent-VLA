"""Task03-F offline spatial-stratification control; production code is read-only.

WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_local_structural_sampling --baseline PATH
Only S3 ep000 Frame0 Local is selected. Actual Global is always read from Task03-E.
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
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from utils.experiment_logger import ExperimentLogger
from visual.runtime.geometry import OrganizedGeometry
from visual.runtime.rgbd import iter_saved_rgbd
from visual.runtime.sampling import fps_indices
from visual.runtime.test import run_priority_fps_metric_control as prior
from visual.runtime.test.run_rim_bottleneck import require_equal, source_contract, write_json

REPO = prior.REPO
SOURCE = REPO / 'visual/results/sampling' / prior.SOURCE_RUN
BASELINE_ID = '7f657cf88baf4f1f89e594252e243e1c'
BASELINE = REPO / 'experiments' / BASELINE_ID
REPORT = REPO / 'docs/visual-graph-integration/03f_local_structural_sampling.md'
STEM = 'local_structural_sampling'
TESTS = ['visual.runtime.test.' + n for n in (
    'test_local_structural_sampling', 'test_sampling', 'test_sampling_geometry',
    'test_sampling_evaluation', 'test_priority_fps_metric_control')]

# Preregistered constants: written to config + snapshot BEFORE any run FPS/tests.
RULES = dict(max_leaves=8, min_child_capacity=4, priority=32, seed=0,
    split_leaf='largest capacity; tie smallest flat', axis='largest valid UV span; tie u',
    cut='minimum count imbalance; tie smaller coordinate; never split equal coordinates',
    group_order='minimum flat ascending; rows flat ascending',
    quota='capacity-capped round robin in group ID order',
    anchor='whole-C seeded first row; actual stored anchor verified without FPS replay',
    rng='same object seed per group; anchor uses initial without RNG draw',
    ties='existing FPS first input row', metric='Camera XY lambda=0',
    surface='existing actual 3D FPS initialized with all P',
    fallback='one leaf=global contract; N<32 all C in group FPS order; N=0 no P; no padding',
    zero_quota='skip group; if anchor quota=0 no forced anchor',
    uv_bins=[4, 4], xyz_bins=[2, 2, 2],
    grid_bounds='actual X / synthetic observed C bbox; half-open except final boundary',
    distances='independent Euclidean C->P 3D mm and UV px; percentiles numpy linear',
    normalized='divide raw metres/pixels by corresponding bbox diagonal',
    na='null scalar summaries for empty/zero diagonal; raw unavailable distance NaN in NPZ only',
    gt='evaluation only after selection; existing same-pair 2px AND 3mm',
    causal_caveat='grouping, quota and group initialization change together; no factorial attribution')

FIXTURES = dict(shape=[1024, 1024], focal_px=1600., principal_uv=[511.5, 511.5],
    near_depth_m=.8, far_depth_m=1.6, seed=0, synthetic_object_id=0,
    line_endpoints_m=[[-.12, 0., 0.], [.12, 0., 0.]], line_candidates=128,
    dense_candidates=128, dense_fraction=.25,
    segments_m=[[[ -.105, -.06, 0.],[-.075, -.06, 0.]],
                [[ -.03, 0., 0.],[.03, 0., 0.]],
                [[ -.06, .06, 0.],[.06, .06, 0.]]],
    candidates_per_segment=64, reference_per_segment=128,
    noise_centre_m=[.10, .02, 0.], noise_step_m=.001, noise_count=8,
    parameter_samples='midpoints (i+.5)/N; extra dense points offset by quarter cell',
    projection='pinhole, np.rint (ties to even), image bounds; retain original camera XYZ',
    collision='one row per pixel: keep smallest generated source ID, then sort flat; never refill',
    occlusion='near image central quarter width: 384 <= u < 640; same rule for references',
    boundary='outer 10% of observed bbox on nonzero axes; occlusion boundary <=1px',
    names=['line_uniform', 'line_dense', 'separated_clean', 'separated_noise',
           'camera_near', 'camera_far', 'camera_occluded'])


def uv_from_flat(flat, shape):
    flat = np.asarray(flat)
    if flat.ndim != 1 or flat.dtype.kind not in 'iu' or len(np.unique(flat)) != len(flat):
        raise ValueError('unique integer source indices required')
    if len(shape) != 2 or min(shape) <= 0 or ((flat < 0) | (flat >= np.prod(shape))).any():
        raise ValueError('invalid raster source')
    return np.column_stack((flat % shape[1], flat // shape[1]))


def spatial_groups(flat, shape):
    """UV only, no XYZ, score, object name or annotation arguments."""
    flat = np.asarray(flat)
    uv = uv_from_flat(flat, shape)
    if len(flat) and not np.all(flat[:-1] < flat[1:]):
        raise ValueError('candidate rows must be original flat ascending')
    leaves = [np.arange(len(flat), dtype=np.int64)] if len(flat) else []
    splits = []
    while len(leaves) < RULES['max_leaves']:
        chosen = None
        for li in sorted(range(len(leaves)), key=lambda i: (-len(leaves[i]), int(flat[leaves[i][0]]))):
            rows = leaves[li]
            axes = []
            for axis in (0, 1):
                values, counts = np.unique(uv[rows, axis], return_counts=True)
                left = np.cumsum(counts)[:-1]
                valid = np.flatnonzero((left >= 4) & (len(rows)-left >= 4))
                if len(valid):
                    cut = min(valid, key=lambda j: (abs(int(2*left[j]-len(rows))), int(values[j])))
                    axes.append((int(values[-1]-values[0]), axis, float(values[cut]+values[cut+1])/2))
            if axes:
                _, axis, boundary = min(axes, key=lambda a: (-a[0], a[1]))
                chosen = li, rows, axis, boundary
                break
        if chosen is None:
            break
        li, rows, axis, boundary = chosen
        low = rows[uv[rows, axis] < boundary]
        high = rows[uv[rows, axis] >= boundary]
        splits.append(dict(parent_min_flat=int(flat[rows[0]]), capacity=len(rows),
                           axis='uv'[axis], boundary=boundary, child_capacity=[len(low), len(high)]))
        leaves[li:li+1] = [low, high]
    leaves.sort(key=lambda rows: int(flat[rows[0]]))
    membership = np.empty(len(flat), np.int64)
    for gid, rows in enumerate(leaves):
        membership[rows] = gid
    return leaves, membership, splits


def allocate(capacities, count):
    n = np.asarray(capacities, dtype=np.int64)
    if count < 0 or (n < 0).any():
        raise ValueError('nonnegative budget/capacity required')
    remaining = min(int(count), int(n.sum()))
    q = np.zeros_like(n)
    while remaining:
        for g in range(len(n)):
            if remaining and q[g] < n[g]:
                q[g] += 1
                remaining -= 1
    return q


def choose_priority(flat, xyz, shape, seed, anchor=None, count=32):
    """Selection inputs only. Source rows remain exact; transformed XYZ is temporary."""
    flat, xyz = np.asarray(flat), np.asarray(xyz)
    if xyz.shape != (len(flat), 3) or not np.isfinite(xyz).all():
        raise ValueError('finite aligned candidate XYZ required')
    leaves, membership, splits = spatial_groups(flat, shape)
    quota = allocate([len(g) for g in leaves], min(count, len(flat)))
    if not len(flat):
        return dict(P=flat.copy(), rows=np.empty(0, np.int64), membership=membership,
                    quota=quota, splits=splits, initial_flat=[], capacities=[])
    if anchor is None:
        anchor = int(flat[np.random.default_rng(seed).integers(len(flat))])
    ar = np.flatnonzero(flat == anchor)
    if len(ar) != 1:
        raise ValueError('anchor must occur exactly once')
    ag = int(membership[ar[0]])
    order = ([ag] if quota[ag] else []) + [g for g in range(len(leaves)) if g != ag and quota[g]]
    selected, first = [], [None] * len(leaves)
    for gid in order:
        rows = leaves[gid]
        initial = np.flatnonzero(rows == ar[0]) if gid == ag else ()
        local = fps_indices(prior.metric_coordinates(xyz[rows], 0.), int(quota[gid]), seed=seed, initial=initial)
        selected.extend(rows[local].tolist())
        first[gid] = int(flat[rows[local[0]]])
    rows = np.asarray(selected, np.int64)
    return dict(P=flat[rows].copy(), rows=rows, membership=membership, quota=quota,
                splits=splits, initial_flat=first, capacities=[len(g) for g in leaves])


def select_local(candidate_flat, candidate_xyz, support_flat, support_xyz, shape, seed, anchor, budget):
    """No annotation arguments and no access to GT globals."""
    support_flat, support_xyz = np.asarray(support_flat), np.asarray(support_xyz)
    uv_from_flat(support_flat, shape)
    if len(support_flat) and not np.all(support_flat[:-1] < support_flat[1:]):
        raise ValueError('support order required')
    rows = np.searchsorted(support_flat, candidate_flat)
    if (rows >= len(support_flat)).any() or not np.array_equal(support_flat[rows], candidate_flat):
        raise ValueError('candidate outside support')
    np.testing.assert_array_equal(support_xyz[rows], candidate_xyz)
    p = choose_priority(candidate_flat, candidate_xyz, shape, seed, anchor)
    initial = np.searchsorted(support_flat, p['P'])
    frows = fps_indices(support_xyz, budget, seed=seed, initial=initial)
    f = support_flat[frows]
    return dict(P=p['P'], S=f[len(initial):], F=f, xyz=support_xyz[frows].copy(),
                grouping=p, shortfall=max(0, budget-len(f)))


def summary(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(count=0, mean=None, p50=None, p95=None, max=None, status='N/A')
    return dict(count=len(a), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), max=float(a.max()), status='defined')


def nearest(query, points):
    return cKDTree(points).query(query)[0] if len(points) else np.full(len(query), np.nan)


def grid_definition(points, bins):
    points = np.asarray(points, float)
    if not len(points):
        return dict(lo=[0.] * len(bins), hi=[0.] * len(bins), bins=[1] * len(bins), empty=True)
    lo, hi = points.min(0), points.max(0)
    return dict(lo=lo.tolist(), hi=hi.tolist(), bins=[b if hi[i] > lo[i] else 1 for i,b in enumerate(bins)], empty=False)


def grid_ids(points, definition):
    lo, hi, bins = (np.asarray(definition[k]) for k in ('lo', 'hi', 'bins'))
    span = hi-lo
    scaled = np.divide(np.asarray(points)-lo, span, out=np.zeros_like(np.asarray(points), dtype=float), where=span>0)
    index = np.minimum(np.maximum(np.floor(scaled*bins).astype(int), 0), bins-1)
    return np.ravel_multi_index(index.T, bins) if len(index) else np.empty(0, np.int64)


def spatial_metrics(flat, xyz, pflat, pxyz, support_flat, support_xyz, shape):
    uv, puv, xuv = (uv_from_flat(f, shape) for f in (flat, pflat, support_flat))
    dmm, dpx = nearest(xyz, pxyz)*1000, nearest(uv, puv)
    definitions = dict(uv=grid_definition(xuv, [4,4]), xyz=grid_definition(support_xyz, [2,2,2]))
    raw = dict(distance_mm=dmm, distance_px=dpx)
    metrics = dict(candidate_count=len(flat), priority_count=len(pflat),
                   selected_fraction=len(pflat)/len(flat) if len(flat) else None, grids={})
    for unit, distance, coordinates, definition in (
        ('mm', dmm, xyz, definitions['xyz']), ('px', dpx, uv, definitions['uv'])):
        diagonal = float(np.linalg.norm(np.asarray(definition['hi'])-definition['lo']))
        denom = diagonal * (1000 if unit == 'mm' else 1)
        norm = distance/denom if denom > 0 else np.full(len(distance), np.nan)
        raw['normalized_'+unit] = norm
        raw['ecdf_'+unit] = np.sort(distance[np.isfinite(distance)])
        worst = int(np.nanargmax(distance)) if np.isfinite(distance).any() else None
        metrics[unit] = dict(raw=summary(distance), normalized=summary(norm), bbox_diagonal=denom,
            worst=None if worst is None else dict(flat_index=int(flat[worst]), xyz_m=xyz[worst].tolist(), uv=uv[worst].tolist()))
    for kind, cpoints, ppoints in (('uv',uv,puv), ('xyz',xyz,pxyz)):
        definition = definitions[kind]
        cid, pid = grid_ids(cpoints, definition), grid_ids(ppoints, definition)
        raw[kind+'_candidate_region'] = cid
        raw[kind+'_priority_region'] = pid
        cells = []
        for rid in range(int(np.prod(definition['bins']))):
            mask = cid == rid
            nc, np_ = int(mask.sum()), int((pid == rid).sum())
            area = float(np.prod((np.asarray(definition['hi'])-definition['lo'])/definition['bins'])) if kind == 'uv' else None
            cells.append(dict(id=rid, candidate_count=nc, priority_count=np_,
                priority_share=np_/len(pflat) if len(pflat) else None, selection_rate=np_/nc if nc else None,
                candidate_density_per_px2=nc/area if area else None,
                distance_mm=summary(dmm[mask]), distance_px=summary(dpx[mask])))
        occupied = [r for r in cells if r['candidate_count']]
        metrics['grids'][kind] = dict(definition=definition, cells=cells,
            candidate_occupied_regions=len(occupied),
            empty_priority_region_ids=[r['id'] for r in occupied if not r['priority_count']],
            selected_region_fraction=sum(r['priority_count'] > 0 for r in occupied)/len(occupied) if occupied else None,
            max_priority_share=max((r['priority_count']/len(pflat) for r in cells), default=0.) if len(pflat) else None,
            equal_region_mean_p95_mm=summary([r['distance_mm']['p95'] for r in occupied if r['distance_mm']['p95'] is not None])['mean'],
            equal_region_mean_p95_px=summary([r['distance_px']['p95'] for r in occupied if r['distance_px']['p95'] is not None])['mean'])
    return metrics, raw


def synthetic_fixture(name):
    """Generate fixed candidates and separate evaluator annotations, no FPS/detector."""
    if name not in FIXTURES['names']:
        raise ValueError('unknown fixed fixture')
    isline = name.startswith('line_')
    endpoints = np.array([FIXTURES['line_endpoints_m']] if isline else FIXTURES['segments_m'], float)
    count = FIXTURES['line_candidates'] if isline else FIXTURES['candidates_per_segment']
    points, labels, references, ref_labels = [], [], [], []
    for sid, (start, end) in enumerate(endpoints):
        t = (np.arange(count)+.5)/count
        points.extend(start+(end-start)*t[:,None]); labels.extend([sid]*count)
        tr = (np.arange(128)+.5)/128
        references.extend(start+(end-start)*tr[:,None]); ref_labels.extend([sid]*128)
    if name == 'line_dense':
        # Quarter-cell offset prevents coincident generated XYZ with the base line.
        t = (np.arange(128)+.25)/128*.25
        points.extend(endpoints[0,0]+(endpoints[0,1]-endpoints[0,0])*t[:,None]); labels.extend([0]*128)
    if name == 'separated_noise':
        centre = np.array(FIXTURES['noise_centre_m'])
        offsets = np.array([(i%4, i//4, 0.) for i in range(8)])*FIXTURES['noise_step_m']
        points.extend(centre+offsets); labels.extend([-1]*8)
    depth = FIXTURES['far_depth_m'] if name == 'camera_far' else FIXTURES['near_depth_m']
    points, references = np.asarray(points), np.asarray(references)
    points[:,2] += depth; references[:,2] += depth
    def project(xyz):
        return np.rint(xyz[:,:2]/xyz[:,2,None]*FIXTURES['focal_px']+FIXTURES['principal_uv']).astype(np.int64)
    uv, ruv = project(points), project(references)
    h,w = FIXTURES['shape']
    keep = ((uv >= 0) & (uv < [w,h])).all(1)
    visible = ((ruv >= 0) & (ruv < [w,h])).all(1)
    if name == 'camera_occluded':
        keep &= ~((uv[:,0] >= 384) & (uv[:,0] < 640))
        visible &= ~((ruv[:,0] >= 384) & (ruv[:,0] < 640))
    generated = np.flatnonzero(keep)
    projected = uv[generated,1]*w+uv[generated,0]
    _, first = np.unique(projected, return_index=True)  # unique values sorted flat; first generated ID wins
    generated = generated[first]
    flat = uv[generated,1]*w+uv[generated,0]
    annotation = dict(structure_id=np.array(labels)[generated], reference_xyz=references,
        reference_structure_id=np.array(ref_labels), reference_visible=visible,
        lengths_m=np.linalg.norm(endpoints[:,1]-endpoints[:,0],axis=1), generated_id=generated)
    return dict(flat=flat, xyz=points[generated], shape=(h,w)), annotation, dict(
        generated_count=len(points), in_view_before_collision=int(keep.sum()), observed_count=len(flat),
        collision_count=int(keep.sum()-len(flat)), excluded_references=int((~visible).sum()))


def evaluate_synthetic(data, annotation, chosen):
    flat, xyz, shape = data['flat'], data['xyz'], data['shape']
    rows = np.searchsorted(flat, chosen)
    pxyz = xyz[rows]
    metrics, raw = spatial_metrics(flat, xyz, chosen, pxyz, flat, xyz, shape)
    visible = annotation['reference_visible']
    distances = nearest(annotation['reference_xyz'][visible], pxyz)*1000
    labels = annotation['reference_structure_id'][visible]
    metrics['reference_mm'] = summary(distances)
    metrics['structures'] = {str(s):dict(length_m=float(length), selected=int((annotation['structure_id'][rows] == s).sum()),
        reference_mm=summary(distances[labels == s])) for s,length in enumerate(annotation['lengths_m'])}
    metrics['noise_selected'] = int((annotation['structure_id'][rows] == -1).sum())
    metrics['noise_priority_fraction'] = metrics['noise_selected']/len(chosen) if len(chosen) else None
    uv = uv_from_flat(flat, shape); lo,hi = uv.min(0),uv.max(0); span=hi-lo
    border = (((uv <= lo+.1*span) | (uv >= hi-.1*span)) & (span > 0)).any(1)
    # Boundary annotation is evaluator-only; near/far/occlusion all use the same definition.
    occ = ((np.abs(uv[:,0]-384) <= 1) | (np.abs(uv[:,0]-640) <= 1)) if annotation.get('occluded', False) else np.zeros(len(flat),bool)
    metrics['boundary'] = {name:dict(candidates=int(mask.sum()), priority=int(mask[rows].sum())) for name,mask in [('outer_10pct',border),('occlusion_1px',occ)]}
    raw['visible_reference_distance_mm'] = distances
    return metrics, raw


def load_actual():
    """Read only S3 geometry/source/evidence; no GT evaluation before Local is fixed."""
    comparison = json.loads((SOURCE/'comparison.json').read_text())
    config = comparison['config']
    contracts = source_contract(SOURCE, config)
    for key in ('input','masks'):
        require_equal(prior.sha(config[key]), config[key+'_sha256'], 'original '+key)
    frame = next(iter_saved_rgbd(config['input'], config['masks'], camera=config['camera']))
    require_equal(frame.source_frame, 0, 'Frame0')
    samples = np.load(SOURCE/'f0000_samples.npz', allow_pickle=False)
    old = json.loads((BASELINE/(prior.STEM+'.json')).read_text())
    require_equal(old['run_id'], BASELINE_ID, 'baseline identity')
    cases, baselines = {}, {}
    with np.load(prior.TASK01/'f0000.npz', allow_pickle=False) as dense, np.load(BASELINE/(prior.STEM+'.npz'), allow_pickle=False) as saved:
        for field in ('intrinsic','cam2world'):
            np.testing.assert_array_equal(dense[field], getattr(frame,field))
        np.testing.assert_array_equal(dense['xyz'][...,2][dense['valid_depth']], frame.depth[dense['valid_depth']])
        geometry = OrganizedGeometry(dense['xyz'].copy(),dense['valid_depth'].copy(),
            {oid:dense[name+'/object_mask'].copy() for name,oid in prior.OBJECTS.items()},dense['overlap_mask'].copy())
        for name,oid in prior.OBJECTS.items():
            np.testing.assert_array_equal(geometry.object_masks[oid], frame.masks[oid])
            prefix = 'S3/'+name
            case = {k:saved[prefix+'/'+k].copy() for k in ('candidate_flat','candidate_xyz','support_flat','support_xyz')}
            candidate, support = case['candidate_flat'], case['support_flat']
            np.testing.assert_array_equal(candidate, samples[f'evidence/pca5/{oid}/20/orientation_flat'])
            np.testing.assert_array_equal(support,np.flatnonzero(geometry.observed_mask(oid)))
            for kind in ('candidate','support'):
                np.testing.assert_array_equal(case[kind+'_xyz'],geometry.xyz.reshape(-1,3)[case[kind+'_flat']])
            count,anchor = prior.EXPECTED['S3',name]
            require_equal(len(candidate),count,'candidate count')
            seed = prior.fps_seed(0,oid)
            require_equal(int(candidate[np.random.default_rng(seed).integers(len(candidate))]),anchor,'seeded anchor')
            key='S3_n256_r0.25_t20_s0'
            mask=samples[key+'/object_id'] == oid
            features={k.split('/metadata/')[1]:samples[k][mask].copy() for k in samples.files if k.startswith(key+'/metadata/')}
            require_equal(len(features),23,'feature field count')
            case.update(name=name,oid=oid,method='S3',first=anchor,seed=seed,budget=prior.BUDGET[name],
                        saved_flat=samples[key+'/flat_index'][mask].copy(),saved_features=features)
            chosen={s:saved[prefix+'/0.0/'+s+'/flat_index'].copy() for s in ('P','S','F')}
            chosen['xyz']=saved[prefix+'/0.0/F/xyz'].copy()
            require_equal((len(chosen['P']),len(chosen['F']),int(chosen['P'][0])),(32,case['budget'],anchor),'stored budget/anchor')
            np.testing.assert_array_equal(chosen['F'],np.r_[chosen['P'],chosen['S']])
            np.testing.assert_array_equal(chosen['xyz'],case['support_xyz'][np.searchsorted(support,chosen['F'])])
            cases[name],baselines[name]=case,chosen
    return frame,geometry,samples,cases,baselines,old,config,contracts


def plots(output, frame, cases, selections, result, arrays, synthetic):
    plt.rcParams.update({'font.size':9})
    fig, axes=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    for i,name in enumerate(prior.OBJECTS):
        case=cases[name]; uv=uv_from_flat(case['candidate_flat'],frame.depth.shape)
        group=selections[name]['local']['grouping']
        axes[i,0].scatter(*uv.T,c=group['membership'],cmap='tab10',s=8)
        for j,method in enumerate(('global','local'),1):
            ax=axes[i,j]; ax.scatter(*uv.T,c='#9ca3af',s=3,alpha=.5)
            selected=uv_from_flat(selections[name][method]['P'],frame.depth.shape)
            ax.scatter(*selected.T,c='#ef7b22',s=25,label='Priority')
            definition=result['actual'][name][method]['spatial']['grids']['uv']['definition']
            for axis in (0,1):
                boundaries=np.linspace(definition['lo'][axis],definition['hi'][axis],definition['bins'][axis]+1)
                for b in boundaries:
                    (ax.axvline if axis==0 else ax.axhline)(b,c='#416b9c',alpha=.3,lw=.8)
            worst=result['actual'][name][method]['spatial']['px']['worst']['uv']
            ax.scatter(*worst,c='red',marker='x',s=80,label='max C->P pixel gap')
            ax.legend(fontsize=7)
        for j,ax in enumerate(axes[i]):
            ax.set(title=f'{name}: '+('UV candidate strata (not physical edges)' if j==0 else ('global','local')[j-1]+' / independent 4x4 grid'),xlabel='u (px)',ylabel='v (px)')
            ax.invert_yaxis()
    fig.savefig(output/'candidate_groups.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for i,name in enumerate(prior.OBJECTS):
        for j,unit in enumerate(('mm','px')):
            for method in ('global','local'):
                values=arrays[f'{name}/{method}/spatial/ecdf_{unit}']
                axes[i,j].plot(values,np.arange(1,len(values)+1)/len(values),label=method)
            axes[i,j].set(title=f'{name}: all C -> P',xlabel=unit,ylabel='ECDF');axes[i,j].legend()
    fig.savefig(output/'candidate_distance_ecdf.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    for i,name in enumerate(prior.OBJECTS):
        for j,kind in enumerate(('uv','xyz')):
            for method,color in [('global','#2878b5'),('local','#e67f22')]:
                cells=result['actual'][name][method]['spatial']['grids'][kind]['cells']
                x=np.arange(len(cells));off=-.18 if method=='global' else .18
                vals=[c['distance_mm']['p95'] if c['distance_mm']['p95'] is not None else np.nan for c in cells]
                axes[i,j].bar(x+off,vals,.34,label=method,color=color)
            axes[i,j].set(title=f'{name}: independent {kind} regions (empty C = blank)',xlabel='region ID',ylabel='C -> P p95 (mm)');axes[i,j].legend()
    fig.savefig(output/'independent_regions.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(17,13),constrained_layout=True)
    for ax,(name,(data,annotation,chosen)) in zip(axes.flat,synthetic.items()):
        uv=uv_from_flat(data['flat'],data['shape'])
        ax.scatter(*uv.T,c=chosen['local']['membership'],cmap='tab10',s=8,alpha=.55)
        for method,marker,color in [('global','o','#1666a2'),('local','x','#d76012')]:
            p=uv_from_flat(chosen[method]['P'],data['shape'])
            ax.scatter(*p.T,marker=marker,facecolors='none' if marker=='o' else color,edgecolors=color,s=40,label=method)
        ax.set(title=f'{name} (N={len(uv)})',xlabel='u (px)',ylabel='v (px)');ax.invert_yaxis();ax.legend(fontsize=7)
    for ax in list(axes.flat)[7:]:ax.axis('off')
    fig.suptitle('Synthetic sampler stress tests: supplied candidates / UV strata / Priority (no GT labels in selection)')
    fig.savefig(output/'synthetic_seven.png',dpi=140);plt.close(fig)
    for name in prior.OBJECTS:
        fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
        refs=arrays[name+'/reference_uv']
        for i,method in enumerate(('global','local')):
            for j,stage in enumerate(('P','F')):
                ax=axes[i,j];ax.imshow(frame.rgb)
                prefix=f'{name}/{method}/{stage}'
                covered=arrays[prefix+'/covered'];matched=arrays[prefix+'/matched_node']
                uv=arrays[prefix+'/pixels_vu'][:,::-1]
                ax.scatter(*refs[~covered].T,c='#df2b68',marker='x',s=15,label='missing GT reference')
                ax.scatter(*refs[covered].T,c='#16b38b',s=12,label='covered GT reference')
                ax.scatter(*uv[~matched].T,facecolors='none',edgecolors='#e6a534',s=28,label='other selected node')
                ax.scatter(*uv[matched].T,facecolors='none',edgecolors='#2972dc',s=80,label='joint matching node')
                ax.set(xlim=(refs[:,0].min()-12,refs[:,0].max()+12),ylim=(refs[:,1].max()+12,refs[:,1].min()-12),
                       title=f'{name} {method} {stage}: {matched.sum()} nodes / {covered.sum()} references')
                ax.legend(fontsize=7)
        fig.suptitle('GT auxiliary evaluation (separate from GT-free UV strata)')
        fig.savefig(output/f'{name}_gt_auxiliary.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for i,name in enumerate(prior.OBJECTS):
        for j,method in enumerate(('global','local')):
            case=cases[name];ax=axes[i,j];support=case['support_xyz']
            ax.scatter(support[:,0]*1000,support[:,2]*1000,c='#a9b4bd',s=.5,alpha=.3)
            for stage,color in [('P','#e87a20'),('S','#2aa3ab')]:
                xyz=support[np.searchsorted(case['support_flat'],selections[name][method][stage])]
                ax.scatter(xyz[:,0]*1000,xyz[:,2]*1000,c=color,s=12,label=stage)
            ax.set(title=f'{name} {method} unfiltered X / P / S',xlabel='camera X (mm)',ylabel='camera Z (mm)');ax.legend()
    fig.savefig(output/'unfiltered_depth_xz.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    names=[n+' '+m for n in prior.OBJECTS for m in ('global','local')]
    rows=[result['actual'][n][m]['gt'] for n in prior.OBJECTS for m in ('global','local')]
    x=np.arange(4)
    axes[0].bar(x-.17,[r['surface_distance_mm']['mean'] for r in rows],.34,label='mean')
    axes[0].bar(x+.17,[r['surface_distance_mm']['p95'] for r in rows],.34,label='p95');axes[0].legend()
    axes[0].set(title='X -> Final actual 3D',ylabel='mm')
    for i,(field,title) in enumerate([('negative','Known-negative union P'),('depth','P beyond X depth p95')],1):
        v=[r['occupancy']['negative_roi']['known_negative_union']['P']['count'] if field=='negative' else
           r['occupancy']['depth_quantile_occupancy']['P']['z_gt_X_p95']['numerator'] for r in rows]
        axes[i].bar(x,v);axes[i].set(title=title,ylabel='nodes / 32')
    for ax in axes:ax.set_xticks(x,names,rotation=25,ha='right')
    fig.savefig(output/'surface_negative_tradeoff.png',dpi=140);plt.close(fig)


def tables(result):
    lines=['| 대상 | 방식 | C→P 3D p50 / p95 / max mm | UV p95 px | UV 점유 구역 선택 / 후보 구역 | UV 구역 동등 평균 p95 mm | Surface mean / p95 mm |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    f=lambda x:'N/A' if x is None else f'{x:.3f}'
    for name,row in result['actual'].items():
        for method in ('global','local'):
            r=row[method];s=r['spatial'];d=s['mm']['raw'];g=s['grids']['uv'];surface=r['gt']['surface_distance_mm']
            occupied=g['candidate_occupied_regions'];selected=occupied-len(g['empty_priority_region_ids'])
            lines.append(f"| {name} | {method} | {' / '.join(f(d[k]) for k in ('p50','p95','max'))} | {f(s['px']['raw']['p95'])} | {selected}/{occupied} | {f(g['equal_region_mean_p95_mm'])} | {f(surface['mean'])} / {f(surface['p95'])} |")
    return lines


def answers(result):
    out=[]
    for name,row in result['actual'].items():
        g,l=(row[m]['spatial'] for m in ('global','local'))
        out.append(f"{name}: C→P p95 {g['mm']['raw']['p95']:.3f}→{l['mm']['raw']['p95']:.3f}mm, max {g['mm']['raw']['max']:.3f}→{l['mm']['raw']['max']:.3f}mm; UV 후보 점유 구역 중 P=0은 {len(g['grids']['uv']['empty_priority_region_ids'])}→{len(l['grids']['uv']['empty_priority_region_ids'])}개다.")
    return out


def create_documents(output, result, test_summary):
    rel=lambda p:os.path.relpath(p,REPORT.parent)
    f=lambda x:'N/A' if x is None else f'{x:.3f}'
    lines=['# Task03-F — Generalizable Local Geometry Preservation','',
        f"실행 UUID: `{result['run_id']}`. ep000 Frame 0 S3의 저장 Global λ=0과 Local UV spatial stratification control을 비교했다.", '',
        '제한된 예산에서 후보의 국소 공간 보존과 실제 구조·Surface의 trade-off를 검증한다. 후보 수를 균형 있게 나누는 UV 분할이며 structural component detection, geometry-aware grouping, density-invariant sampling이 아니다. Grouping·quota·group별 초기화가 함께 달라지므로 효과를 quota 하나에 귀속하지 않는다.', '',
        '## 실행 조건과 사전 고정','',
        '- num_points=256, structural_ratio=.25, threshold=20°, seed=0. C/X 및 원본 XYZ·pixel·source 불변. P 각 32, Peg F=129, Box F=127. 실제 Global FPS 호출은 0회다.',
        '- 최대 8 leaf, split 최소 4후보, 최대 capacity leaf부터 UV 유효 span 축/균형 cut, capacity-capped round robin과 기존 Camera XY FPS를 사용했다. Surface는 원본 3D FPS다.',
        '- 8/4는 이번 K=32 control 설정이다. 일반 budget의 성능을 보장하지 않는다. 규칙·평가 구역·7개 fixture·N/A 정의를 config/source snapshot에 고정한 뒤 테스트·선택을 수행했다.',
        '- GT pose·rim·face-pair·ROI는 실제 선택 저장 후에만 평가에 사용했다. Task-Relevance Scoring은 구현하지 않았다.', '',
        '## GT-free 공간 보존','',*tables(result),'',*answers(result),'',
        '전체 p95는 후보 밀도에 가중된다. 독립 UV 4×4 및 XYZ 최대 2×2×2 구역의 count·밀도·선택률·p95/max·동등 구역 통계와 raw/normalized 거리·ECDF는 JSON/NPZ에 저장했다. C에는 false response가 섞이므로 지표 변화를 구조 검출 정확도 변화로 치환하지 않는다. 최댓값은 관측 C 위의 미피복 위치이며 연속 빈 공간의 크기를 복원한 값이 아니다.', '',
        '## Peg seam / Box rim 보조 평가','',
        '같은 point/reference pair의 2px AND 3mm matcher를 재사용했다. Peg observed 462(GT visible 464), Box 103. nodes는 실제 matching node 수, refs는 unique reference 수다. 구간 복수 귀속은 ambiguous이며 Peg 물리적 간격은 N/A다.', '',
        '| 대상 | 방식 | C nodes / refs | P nodes / refs | S nodes / refs | F nodes / refs | Surface 추가 refs | Negative union P / S / F |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for name,row in result['actual'].items():
        for method in ('global','local'):
            r=row[method]['gt'];st=r['stages'];counts=['%d / %d'%(st[s]['structural_matching_nodes'],st[s]['unique_reference_count']) for s in ('C','P','S','F')]
            neg=r['occupancy']['negative_roi']['known_negative_union']
            lines.append(f"| {name} | {method} | "+' | '.join(counts)+f" | {r['coverage']['differences']['S_minus_P']['count']} | "+' / '.join(str(neg[s]['count']) for s in ('P','S','F'))+' |')
    lines+=['','| 대상 | 후보 C / support X | Leaf capacity (ID 순서) | Quota | Global / Local exact feature rows |',
            '| --- | ---: | --- | --- | ---: |']
    for name,row in result['actual'].items():
        s=row['global']['spatial'];group=row['grouping']
        available=[row[m]['gt']['feature_availability']['exact_available_rows'] for m in ('global','local')]
        lines.append(f"| {name} | {s['candidate_count']} / {row['global']['gt']['occupancy']['depth_m']['X']['count']} | {group['capacities']} | {group['quota']} | {available[0]} / {available[1]} |")
    lines+=['','| 대상 | 구간 | observed | C refs | Global P / F refs | Local P / F refs | Global / Local F nodes |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,row in result['actual'].items():
        g,l=(row[m]['gt']['stages'] for m in ('global','local'))
        for label,c in g['C']['segments'].items():
            v=lambda stages,s:stages[s]['segments'][label]
            lines.append(f"| {name} | {label} | {c['observed_references']} | {c['coverage_count']} | {v(g,'P')['coverage_count']} / {v(g,'F')['coverage_count']} | {v(l,'P')['coverage_count']} / {v(l,'F')['coverage_count']} | {v(g,'F')['matching_node_count']} / {v(l,'F')['matching_node_count']} |")
    lines+=['', '`rim_axis1_-1`의 기존 0/26은 평가 전용 실패 사례다. X/C의 지원과 P/S/F의 보존 reference·matching node·source 목록을 별도 저장했다. 이 구간 개선을 필수 성공 조건으로 삼거나 이를 보고 재선택하지 않았다.', '',
        '## Synthetic sampler stress test','',
        '3종 총 7개 고정 fixture, K=32, 같은 C·anchor·Camera XY λ=0·기존 FPS다. Synthetic에서는 P만 비교하며 detector/normal/tracker/Surface를 만들지 않았다. 구조/noise/reference annotation은 평가 전용이다. 투영 충돌은 생성 ID가 작은 한 점만 남기고 flat 순으로 정렬하며 refill하지 않았다. 실제 camera XYZ를 보존했다.', '',
        '| Fixture | 생성 / 충돌 전 / 관측 C | Global / Local C→P p95 mm | Global / Local 정규화 p95 | Global / Local reference→P p95 mm | Global / Local noise P |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,row in result['synthetic'].items():
        g,l=(row[m] for m in ('global','local'));c=row['counts']
        lines.append(f"| {name} | {c['generated_count']} / {c['in_view_before_collision']} / {c['observed_count']} | {f(g['mm']['raw']['p95'])} / {f(l['mm']['raw']['p95'])} | {f(g['mm']['normalized']['p95'])} / {f(l['mm']['normalized']['p95'])} | {f(g['reference_mm']['p95'])} / {f(l['reference_mm']['p95'])} | {g['noise_selected']} / {l['noise_selected']} |")
    lines+=['','| Fixture | 구조 | 길이 m | Global / Local 선택 수 | Global / Local reference p95 mm |',
            '| --- | --- | ---: | ---: | ---: |']
    for name,row in result['synthetic'].items():
        for sid,g in row['global']['structures'].items():
            l=row['local']['structures'][sid]
            lines.append(f"| {name} | {sid} | {g['length_m']:.3f} | {g['selected']} / {l['selected']} | {f(g['reference_mm']['p95'])} / {f(l['reference_mm']['p95'])} |")
    lines+=['','| Fixture | 외곽 띠 후보 수 | Global / Local 외곽 띠 P | 가림 경계 후보 / Global P / Local P |',
            '| --- | ---: | ---: | ---: |']
    for name,row in result['synthetic'].items():
        g,l=(row[m]['boundary'] for m in ('global','local'))
        lines.append(f"| {name} | {g['outer_10pct']['candidates']} | {g['outer_10pct']['priority']} / {l['outer_10pct']['priority']} | {g['occlusion_1px']['candidates']} / {g['occlusion_1px']['priority']} / {l['occlusion_1px']['priority']} |")
    lines+=['','가림 fixture에서는 짧은 선분 하나만 관측에 남고, 지정한 가림 경계 1px 띠에는 후보가 없다. 이 사례의 동률과 경계 점유 0은 복잡한 부분 가림 또는 경계 noise에 대한 강건성 증거가 아니다. Fixture를 결과에 맞게 바꾸지 않았다.','']
    lines+=['','## 최종 질문별 판단','',*result['interpretation_ko'],'',
        '## 원거리 depth와 Surface·ROI trade-off','',
        '원거리 positive depth를 필터링하지 않았다. 평가 bbox는 전체 observed X에서 정하므로 depth outlier가 3D bbox 대각선을 키우고 XYZ 격자를 넓혀 후보를 일부 구역에 몰아넣을 수 있다. 따라서 정규화 거리·구역 점유만으로 개선을 판단하지 않으며 raw 거리와 UV 결과를 함께 본다. 아래 X depth와 bbox는 두 방식에 동일하다.', '',
        '| 대상 | X depth median / p95 / max m | XYZ bbox 대각선 mm | P depth median / p95 Global → Local | P z>X p95 Global / Local |',
        '| --- | ---: | ---: | --- | ---: |']
    for name,row in result['actual'].items():
        g,l=(row[m]['gt']['occupancy'] for m in ('global','local'));x=g['depth_m']['X']
        lines.append(f"| {name} | {x['median']:.4f} / {x['p95']:.4f} / {x['max']:.4f} | {row['global']['spatial']['mm']['bbox_diagonal']:.3f} | {g['depth_m']['P']['median']:.4f}/{g['depth_m']['P']['p95']:.4f} → {l['depth_m']['P']['median']:.4f}/{l['depth_m']['P']['p95']:.4f} | {g['depth_quantile_occupancy']['P']['z_gt_X_p95']['numerator']} / {l['depth_quantile_occupancy']['P']['z_gt_X_p95']['numerator']} |")
    lines+=['','| 대상 | 방식 | 3D / UV bbox 정규화 p95 | XYZ 후보 점유 구역 / P=0 구역 | 최대 3D 미피복 후보 source / UV / XYZ m |',
            '| --- | --- | ---: | --- | --- |']
    for name,row in result['actual'].items():
        for method in ('global','local'):
            s=row[method]['spatial'];w=s['mm']['worst'];g=s['grids']['xyz']
            lines.append(f"| {name} | {method} | {s['mm']['normalized']['p95']:.6f} / {s['px']['normalized']['p95']:.6f} | {g['candidate_occupied_regions']} / {g['empty_priority_region_ids']} | {w['flat_index']} / {w['uv']} / {w['xyz_m']} |")
    lines+=['','Box의 X depth max 30.726m를 유지한 XYZ 격자에서는 C가 단 한 구역에 들어갔다. 이 coarse 격자는 이 관측에서 국소 분산을 구별하지 못한다. 작은 정규화 거리 역시 큰 bbox 분모의 영향이 있으므로 raw 거리와 UV 격자를 병기했다.','']
    lines+=['','Known-negative ROI는 겹칠 수 있어 union으로 중복을 제거했다. 일반 Surface나 관측 경계 자체를 자동으로 오선택이라 판정하지 않는다.','',
        '## Feature·검증·재현 범위','',
        '- 같은 S3 Task03 baseline metadata가 있는 source에만 23 field를 exact gather했다. Availability mask·compact positions를 저장하고 나머지 feature는 생성하지 않았다. Geometry-level offline 검증이며 완전한 Vision Tensor 출력 검증은 아니다.',
        f'- 필수 계약 테스트: {test_summary}. PASS/FAIL/SKIP와 원본 hash·NPZ readback·상대 링크·PNG 검증은 verification.json에 기록했다. 성능 개선은 테스트 PASS 조건이 아니다.',
        '- 기존 ExperimentLogger(team=visual, job_type=evaluate), UUID/project/group 및 artifact inventory를 재사용했다. WANDB_MODE=disabled이며 업로드는 하지 않았다. Git 시작 상태가 dirty이므로 source snapshot·변경 전 diff·implementation diff를 함께 보관했다.',
        '- CPU NumPy/SciPy offline 실행만 수행했다. Simulator·renderer·CUDA/GPU/backend, 새 episode, 학습/GNN/BC, production sampler/normal/feature/tracking 변경, dependency 설치·외부 업로드·Git commit은 미실행이다.',
        '- 실험 완료와 알고리즘 채택은 별개다. 한 frame과 7개 synthetic fixture로 일반화를 입증하지 않는다. 일반 sampling 방식 채택에는 독립 형상·관측·episode·센서에서 동일 설정 검증이 필요하다. Task-Relevance Scoring은 향후 독립 비교로 남긴다.', '',
        '## 산출물','']
    for p in sorted(output.iterdir()):
        if p.is_file() and not p.name.endswith('.tracking.json') and p.name not in ('metadata.json','metrics.json','config.yaml','manifest.json'):
            lines.append(f'- [{p.name}]({rel(p)})')
    if 'report_recovery' in result:
        recovery=result['report_recovery']
        lines+=['','## 문서 생성 오류와 재실행 구분','',
            f"최초 수치 실행 `{recovery['selection_run_id']}`는 계약 테스트·선택·평가·PNG 생성 후 HTML의 원본 PNG 파일명 링크 검사에서 실패했다. 실패 기록과 당시 source/config/선택/수치는 보존했다. 현재 UUID는 그 저장 결과를 읽은 **문서 후처리 실행**이며 실제·synthetic FPS 및 성능 평가를 다시 실행하지 않았다. Group·quota·fixture·threshold·seed는 바꾸지 않았다.", '',
            f"- [실패 기록]({rel(REPO/'experiments'/recovery['selection_run_id']/'attempt_failure.json')})",
            f"- [최초 수치 JSON]({rel(REPO/'experiments'/recovery['selection_run_id']/(STEM+'.json'))})",
            f"- [최초 선택/source 보존 자료]({rel(REPO/'experiments'/recovery['selection_run_id']/'source_hashes.json')})"]
    lines+=['','## 기존 근거','',
        '- [승인된 계획](03f_local_structural_sampling_plan.md)',
        '- [Task03-E 보고서](03e_priority_fps_metric_control.md)',
        f'- [Task03-E 원본 JSON]({rel(BASELINE/(prior.STEM+".json"))}), [원본 NPZ]({rel(BASELINE/(prior.STEM+".npz"))})',
        f'- [Task03 원본 comparison]({rel(SOURCE/"comparison.json")}), [원본 samples]({rel(SOURCE/"f0000_samples.npz")})',
        '- [runner](../../visual/runtime/test/run_local_structural_sampling.py), [계약 테스트](../../visual/runtime/test/test_local_structural_sampling.py)', '',
        '재실행은 새 보호 baseline.json과 새 UUID에서만 수행한다. 기존 보고서가 있으면 overwrite를 거부한다. 결과 제출 후 다음 알고리즘 선택 없이 중단한다.','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    shutil.copyfile(REPORT,output/'report.md')
    images=[('candidate_groups.png','GT-free candidate strata and independent evaluation grid'),
        ('candidate_distance_ecdf.png','Global / Local C to P ECDF'),('independent_regions.png','Independent spatial regions'),
        ('synthetic_seven.png','Seven fixed synthetic fixtures'),('peg_gt_auxiliary.png','Peg seam: GT auxiliary evaluation'),
        ('box_gt_auxiliary.png','Box rim: GT auxiliary evaluation'),('unfiltered_depth_xz.png','Unfiltered positive depth'),
        ('surface_negative_tradeoff.png','Surface and known-negative trade-offs')]
    links=[p for p in output.iterdir() if p.is_file() and not p.name.endswith('.tracking.json')]
    def html_table(rows):
        cells=[line.strip('|').split('|') for line in rows]
        return '<div style="overflow-x:auto"><table><thead><tr>'+''.join('<th>'+html.escape(c.strip())+'</th>' for c in cells[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in row)+'</tr>' for row in cells[2:])+'</tbody></table></div>'
    syn_table=['| Fixture | N | Reference p95 Global mm | Reference p95 Local mm | Noise P Global/Local |','| --- | --- | --- | --- | --- |']
    for name,row in result['synthetic'].items():
        syn_table.append(f"| {name} | {row['counts']['observed_count']} | {row['global']['reference_mm']['p95']:.3f} | {row['local']['reference_mm']['p95']:.3f} | {row['global']['noise_selected']} / {row['local']['noise_selected']} |")
    blocks=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Task03-F Local geometry sampling</title><style>body{max-width:1250px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;color:#203044;background:#f6f8fb}img{width:100%;height:auto;background:white}section{margin:32px 0}pre{white-space:pre-wrap;background:white;padding:20px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #dde3eb;text-align:left}a{color:#146bc0}h2{margin-top:40px}</style>',
        '<h1>Task03-F · Local geometry sampling</h1>',
        '<p>UV spatial stratification control. Candidate-count splits, not physical components. Grouping, quota and per-group initialization change together. GT annotations are evaluation-only. No production policy is adopted.</p>',
        f'<p>Run: {result["run_id"]} · <a href="{html.escape(os.path.relpath(REPORT,output))}">Markdown report</a> · <a href="{html.escape(os.path.relpath(BASELINE/(prior.STEM+".json"),output))}">Original Global JSON</a> · <a href="{html.escape(os.path.relpath(BASELINE/(prior.STEM+".npz"),output))}">Original Global NPZ</a> · <a href="{html.escape(os.path.relpath(SOURCE/"f0000_samples.npz",output))}">Task03 samples</a> · <a href="{html.escape(os.path.relpath(SOURCE/"f0000_overlay.png",output))}">Task03 PNG</a></p>',
        '<h2>Measured comparison</h2>'+html_table(tables(result)),
        '<h2>Seven fixed synthetic fixtures</h2>'+html_table(syn_table),
        '<h2>Findings and failures</h2><pre>'+html.escape('\n\n'.join(result['interpretation_ko']))+'</pre>']
    for filename,title in images:
        blocks.append(f'<section><h2>{html.escape(title)}</h2><a href="{filename}"><img src="{filename}" alt="{html.escape(title)}"></a></section>')
    blocks.append('<p>Occlusion fixture leaves only one short segment visible; no candidates fall in the fixed 1px occlusion boundary band. It does not establish robustness to boundary noise. Box support includes positive depth up to 30.726m: all candidates lie in one coarse XYZ evaluation region. Raw distances and UV regions remain necessary.</p>')
    blocks+=['<h2>Raw files / provenance</h2><ul>']+[f'<li><a href="{html.escape(p.name)}">{html.escape(p.name)}</a></li>' for p in sorted(links)]+['</ul>',
        '<script>window.addEventListener("load",()=>{document.body.dataset.imagesComplete=String([...document.images].every(i=>i.complete&&i.naturalWidth>0&&i.naturalHeight>0));document.body.dataset.imageCount=document.images.length;});</script></html>']
    (output/(STEM+'.html')).write_text('\n'.join(blocks),encoding='utf-8')


def interpretation(result):
    """Describe measured failures without tuning or a composite success score."""
    a=result['actual'];s=result['synthetic']
    selection=lambda fixture,method:[r['selected'] for r in s[fixture][method]['structures'].values()]
    p95=lambda fixture,method:s[fixture][method]['reference_mm']['p95']
    bg,bl=(a['box'][m]['spatial'] for m in ('global','local'))
    items=['1. 공간 누락: '+' '.join(answers(result))+
        f" Box pixel p95는 {bg['px']['raw']['p95']:.3f}→{bl['px']['raw']['p95']:.3f}px로 좋아졌지만 3D tail과 실제 rim 보존은 악화됐다. "
        '독립 UV 구역의 점유 개수도 같아 점유율만으로 차이를 설명할 수 없다. Group quota 충족은 성능 근거가 아니다.']
    items.append('2. 밀도·길이 편향: 균일→조밀 직선의 reference p95는 Global '
        f"{p95('line_uniform','global'):.3f}→{p95('line_dense','global'):.3f}mm, Local {p95('line_uniform','local'):.3f}→{p95('line_dense','local'):.3f}mm다. "
        f"길이 비 1:2:4 clean 구조의 선택 수는 Global {selection('separated_clean','global')}, Local {selection('separated_clean','local')}다. "
        '후보 수 기반 분할과 균등 quota는 실제 길이나 면적을 균등 보존하는 규칙이 아니다.')
    items.append('3. Noise: 8개 noise 후보의 선택 수는 '
        f"Global {s['separated_noise']['global']['noise_selected']}/32, Local {s['separated_noise']['local']['noise_selected']}/32다. "
        f"실제 세 구조 선택 수는 clean→noise에서 Global {selection('separated_clean','global')}→{selection('separated_noise','global')}, "
        f"Local {selection('separated_clean','local')}→{selection('separated_noise','local')}다. "
        '이 fixture에서는 noise에 더 많은 예산을 쓰는 현상은 관찰되지 않았다. Noise 추가에 따른 구조별 배분 변화는 남으며, 이를 일반적인 noise 강건성으로 확대하지 않는다.')
    camera=[]
    for n in ('camera_near','camera_far','camera_occluded'):
        r=s[n]
        camera.append(f"{n}: C={r['counts']['observed_count']}, reference p95 G/L={p95(n,'global'):.3f}/{p95(n,'local'):.3f}mm, "
            f"C→P 정규화 p95 G/L={r['global']['mm']['normalized']['p95']:.4f}/{r['local']['mm']['normalized']['p95']:.4f}")
    items.append('4. 거리·가림 민감도: '+'; '.join(camera)+
        '. 가림은 남은 visible reference만 평가하며, 관측 불가능 geometry 복원을 요구하지 않는다. 정수 투영 충돌·후보 순서·seeded anchor도 조건 간 달라질 수 있어 scale 하나의 인과 효과로 분리하지 않는다.')
    actual=[]
    for name,row in a.items():
        g,l=(row[m]['gt'] for m in ('global','local'))
        actual.append(f"{name} Final refs {g['stages']['F']['unique_reference_count']}→{l['stages']['F']['unique_reference_count']}, "
            f"Surface mean/p95 {g['surface_distance_mm']['mean']:.3f}/{g['surface_distance_mm']['p95']:.3f}→{l['surface_distance_mm']['mean']:.3f}/{l['surface_distance_mm']['p95']:.3f}mm, "
            f"negative P {g['occupancy']['negative_roi']['known_negative_union']['P']['count']}→{l['occupancy']['negative_roi']['known_negative_union']['P']['count']}/32")
    items.append('5. 실제 구조와 Surface: '+'; '.join(actual)+
        '. 구간별 증가와 감소는 보조 평가 표에 모두 제시했다. Box 한 구간의 개선으로 다른 구조 손실을 상쇄했다고 판단하지 않는다.')
    items.append('6. 추가 검증 판단: 실제 Peg/Box의 3D tail과 GT 보존, 긴 구조·밀도 변화의 reference 보존에서 전반적 열세다. '
        '현재 설정은 공간 배분의 단순 control로 남길 수 있으나 일반 geometry preservation 정책 후보로 적극 채택할 근거는 얻지 못했다. '
        '더 구조적인 grouping이 필요한지 검토할 이유는 생겼지만, 이 실험은 그 대안의 우월성을 검증하지 않았다. '
        '동일 설정의 독립 형상·관측·episode 검증 전에는 채택하지 않는다. GNN/BC 성능 개선은 주장하지 않는다.')
    return items


def verify_documents(output):
    class Links(HTMLParser):
        def __init__(self): super().__init__();self.targets=[]
        def handle_starttag(self,tag,attrs):
            self.targets.extend(v for k,v in attrs if k in ('href','src'))
    parser=Links();parser.feed((output/(STEM+'.html')).read_text())
    for target in parser.targets:
        if not (output/target).is_file():raise ValueError('missing HTML target '+target)
    import re
    targets=re.findall(r'\]\(([^)]+)\)',REPORT.read_text())
    for target in targets:
        if not (REPORT.parent/target).is_file():raise ValueError('missing Markdown target '+target)
    decoded={}
    for path in output.glob('*.png'):
        pixels=plt.imread(path)
        if pixels.ndim!=3 or min(pixels.shape[:2])<400 or pixels.std()<.01:raise ValueError('empty PNG '+path.name)
        decoded[path.name]=list(pixels.shape)
    return dict(status='PASS',html_targets=len(parser.targets),markdown_targets=len(targets),png_decoded=decoded)


def add_arrays(arrays,prefix,data):
    arrays.update({prefix+'/'+k:np.asarray(v) for k,v in data.items()})


def protected(baseline):
    current=prior.hashes([REPO/p for p in baseline['hashes']])
    require_equal(current,baseline['hashes'],'all original source/input/artifact/dirty files')
    return dict(status='PASS',file_count=len(current),all_sha256_unchanged=True)


def run(baseline_path):
    if os.environ.get('WANDB_MODE')!='disabled':raise ValueError('WANDB_MODE=disabled required')
    if REPORT.exists():raise FileExistsError('refuse to overwrite result report')
    baseline_path=Path(baseline_path)
    baseline=json.loads(baseline_path.read_text());protected(baseline)
    sources=list((REPO/'visual/runtime').glob('*.py'))+list((REPO/'visual/runtime/test').glob('*.py'))+[
        REPO/'utils/experiment_logger.py',REPO/'utils/experiment_artifacts.py']
    source_hashes=prior.hashes(sources)
    old=json.loads((BASELINE/(prior.STEM+'.json')).read_text())
    input_hashes=prior.hashes([REPO/p for p in old['config']['input_sha256']]+[
        BASELINE/(prior.STEM+'.json'),BASELINE/(prior.STEM+'.npz'),BASELINE/'verification.json'])
    for path,digest in old['config']['input_sha256'].items():require_equal(prior.sha(REPO/path),digest,'Task03-E input '+path)
    config=dict(task='03-F',source_run_id=BASELINE_ID,task03_source_run_id=prior.SOURCE_RUN,
        episode='ep000',frame=0,method='S3(PCA5)',num_points=256,structural_ratio=.25,threshold_deg=20,
        seed=0,priority_count=32,final_budgets=prior.BUDGET,rules=RULES,synthetic_fixtures=FIXTURES,
        source_sha256=source_hashes,input_sha256=input_hashes,python=platform.python_version(),
        numpy=np.__version__,device='cpu',simulator_executed=False,gpu_executed=False,
        actual_global_fps_replayed=False,production_policy_adopted=False)
    with ExperimentLogger(team='visual',config=config,seed=0,model='task03f_uv_spatial_stratification',job_type='evaluate') as logger:
        output=logger.directory
        try:
            write_json(output/'git_start.json',baseline)
            shutil.copyfile(baseline_path.with_name('git_start.diff'),output/'git_start.diff')
            for path in sources:
                dest=output/'source_snapshot'/path.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
            write_json(output/'preregistered_config.json',config)
            # Materialize fixtures and their identities BEFORE any FPS in tests or performance comparison.
            fixtures={name:synthetic_fixture(name) for name in FIXTURES['names']}
            write_json(output/'fixture_manifest.json',{name:dict(counts=counts,
                flat_sha256=prior.array_sha(data['flat']),xyz_sha256=prior.array_sha(data['xyz']),
                annotation_sha256={k:prior.array_sha(v) for k,v in ann.items()}) for name,(data,ann,counts) in fixtures.items()})
            frozen_hash=prior.sha(output/'preregistered_config.json')
            with (output/'tests.log').open('w') as log:
                process=subprocess.run([sys.executable,'-m','unittest',*TESTS,'-v'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:raise RuntimeError('required contract tests failed; no actual Local FPS executed')
            test_text=(output/'tests.log').read_text()
            test_summary=next(line for line in test_text.splitlines() if line.startswith('Ran '))+'; '+test_text.splitlines()[-1]
            print(test_summary,flush=True)
            frame,geometry,samples,cases,global_selection,old,old_config,contracts=load_actual()
            selections={};arrays={};result=dict(run_id=logger.run_id,source_run_id=BASELINE_ID,config=config,
                source_contract=contracts,actual={},synthetic={},performance_success_required=False)
            # This is the ONLY actual sampling block; no Global FPS call.
            for name,case in cases.items():
                local=select_local(case['candidate_flat'],case['candidate_xyz'],case['support_flat'],case['support_xyz'],
                    frame.depth.shape,case['seed'],case['first'],case['budget'])
                require_equal((len(local['P']),len(local['F'])),(32,case['budget']),'actual fixed budgets')
                selections[name]={'global':global_selection[name],'local':local}
                for field in ('candidate_flat','candidate_xyz','support_flat','support_xyz'):arrays[name+'/'+field]=case[field]
                group=local['grouping']
                add_arrays(arrays,name+'/group',dict(membership=group['membership'],quota=group['quota'],capacities=group['capacities']))
                for method,chosen in selections[name].items():
                    np.testing.assert_array_equal(chosen['F'],np.r_[chosen['P'],chosen['S']])
                    require_equal(len(np.unique(chosen['F'])),case['budget'],'unique selection')
                    require_equal(int(chosen['P'][0]),case['first'],'same initial anchor')
                    for stage in ('P','S','F'):
                        add_arrays(arrays,f'{name}/{method}/{stage}',dict(flat_index=chosen[stage],
                            xyz=case['support_xyz'][np.searchsorted(case['support_flat'],chosen[stage])]))
            for method in ('global','local'):
                joined=np.concatenate([selections[name][method]['F'] for name in prior.OBJECTS])
                require_equal((len(joined),len(np.unique(joined))),(256,256),'cross-object budget and unique sources')
            # Persist selection boundary before loading annotations; failures cannot trigger reselection.
            np.savez_compressed(output/'selection_before_gt.npz',**arrays)
            print('Only S3 Peg/Box Local selected and persisted; Global replay calls=0. Loading evaluation annotations.',flush=True)
            with np.load(prior.EDGES,allow_pickle=False) as edges,np.load(prior.NORMAL,allow_pickle=False) as normal:
                contexts=prior.evaluation_inputs(frame,geometry,samples,edges,normal)
                peg_labels,peg_members=prior.peg_labels(normal['evaluation/peg/face'],normal['evaluation/peg/gt_visible'],
                    normal['evaluation/peg/seam_endpoints'],contexts['peg']['reference'][0])
            with np.load(old_config['input'],allow_pickle=False) as episode:
                box_labels,box_specs,box_transform=prior.rim_labels(contexts['box']['reference'][1],frame.cam2world,
                    episode['box_pose'][0],episode['box_hole_pose'][0],float(episode['box_hole_radii']))
            gate={}
            for name,case in cases.items():
                context=contexts[name];labels=peg_labels if name=='peg' else box_labels
                add_arrays(arrays,name,dict(reference_uv=context['reference'][0],reference_xyz=context['reference'][1],
                    observable=context['observable'],reference_segment=labels))
                row={}
                for method,chosen in selections[name].items():
                    pxyz=case['support_xyz'][np.searchsorted(case['support_flat'],chosen['P'])]
                    spatial,raw=spatial_metrics(case['candidate_flat'],case['candidate_xyz'],chosen['P'],pxyz,
                        case['support_flat'],case['support_xyz'],frame.depth.shape)
                    gt,data=prior.score_case(case,chosen,context,frame,labels,{} if name=='peg' else box_specs,None if name=='peg' else box_transform)
                    add_arrays(arrays,f'{name}/{method}/spatial',raw);add_arrays(arrays,f'{name}/{method}',data)
                    row[method]=dict(spatial=spatial,gt=gt)
                    if method=='global':
                        previous=old['cases']['S3/'+name]['conditions']['0.0']
                        for field in ('coverage','stages','surface_distance_mm','occupancy','feature_availability'):
                            require_equal(gt[field],previous[field],'stored baseline evaluation '+name+'/'+field)
                        gate[name]=dict(status='PASS',all_saved_baseline_evaluation_exact=True,global_fps_calls=0)
                group=selections[name]['local']['grouping']
                row['grouping']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in group.items() if k not in ('P','rows')}
                row['paired']=dict(priority_shared=len(np.intersect1d(selections[name]['global']['P'],selections[name]['local']['P'])),
                    final_shared=len(np.intersect1d(selections[name]['global']['F'],selections[name]['local']['F'])))
                # Reference support and exact match sources on the designated auxiliary failure case.
                if name=='box':
                    columns=np.flatnonzero(labels=='rim_axis1_-1');matcher=context['matcher'];audit={}
                    for method in ('global','local'):
                        audit[method]={}
                        for stage,flat in dict(X=case['support_flat'],C=case['candidate_flat'],**{k:selections[name][method][k] for k in ('P','S','F')}).items():
                            adj=matcher.adjacencies['2px_3mm'][np.searchsorted(matcher.flat,flat)][:,columns]
                            audit[method][stage]=dict(matching_flat_indices=flat[adj.getnnz(axis=1)>0].tolist(),
                                covered_reference_ids=columns[adj.getnnz(axis=0)>0].tolist(),denominator=26)
                    row['rim_axis1_-1_audit']=audit
                result['actual'][name]=row
            result['baseline_gate']=gate
            result['peg_memberships']=peg_members
            synthetic={}
            for name,(data,annotation,counts) in fixtures.items():
                seed=prior.fps_seed(0,0);flat,xyz=data['flat'],data['xyz']
                anchor=int(flat[np.random.default_rng(seed).integers(len(flat))]);ar=int(np.flatnonzero(flat==anchor)[0])
                global_rows=fps_indices(prior.metric_coordinates(xyz,0.),32,seed=seed,initial=[ar])
                chosen={'global':dict(P=flat[global_rows]),'local':choose_priority(flat,xyz,data['shape'],seed,anchor)}
                # Performance evaluator sees annotations only after both selections are fixed.
                annotation['occluded']=name=='camera_occluded'
                row=dict(counts=counts,anchor=anchor)
                for method in ('global','local'):
                    metric,raw=evaluate_synthetic(data,annotation,chosen[method]['P'])
                    row[method]=metric;add_arrays(arrays,f'synthetic/{name}/{method}',dict(P=chosen[method]['P'],**raw))
                add_arrays(arrays,'synthetic/'+name,dict(candidate_flat=flat,candidate_xyz=xyz,
                    group_membership=chosen['local']['membership'],group_quota=chosen['local']['quota'],**annotation))
                row['grouping']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in chosen['local'].items() if k not in ('P','rows')}
                result['synthetic'][name]=row;synthetic[name]=(data,annotation,chosen)
            require_equal(prior.sha(output/'preregistered_config.json'),frozen_hash,'frozen config')
            result['interpretation_ko']=interpretation(result)
            result['contract_status']='PASS'
            write_json(output/(STEM+'.json'),result);np.savez_compressed(output/(STEM+'.npz'),**arrays)
            with np.load(output/(STEM+'.npz'),allow_pickle=False) as saved:
                for k,v in arrays.items():np.testing.assert_array_equal(saved[k],v,err_msg=k)
            plots(output,frame,cases,selections,result,arrays,synthetic)
            preservation=protected(baseline)
            require_equal(prior.hashes(sources),source_hashes,'executed sources unchanged')
            require_equal(prior.hashes([REPO/p for p in input_hashes]),input_hashes,'input arrays/artifacts unchanged')
            write_json(output/'source_hashes.json',dict(sources_before=source_hashes,sources_after=prior.hashes(sources),
                inputs_before=input_hashes,inputs_after=prior.hashes([REPO/p for p in input_hashes]),preservation=preservation,
                historical_task03e_sources=old['config']['source_sha256']))
            write_json(output/'git_end.json',dict(git_status=subprocess.check_output(['git','status','--short'],cwd=REPO).decode(),original_preservation=preservation))
            verification=dict(status='PASS',required_tests=dict(summary=test_summary,modules=TESTS,
                pass_count=sum(' ... ok' in l or l=='ok' for l in test_text.splitlines()),fail_count=0,skip_count=sum(' ... skipped' in l for l in test_text.splitlines())),
                baseline_gate=gate,original_preservation=preservation,config_frozen_sha256=frozen_hash,
                real_fps_calls=dict(global_baseline=0,local_priority=sum(int((selections[n]['local']['grouping']['quota']>0).sum()) for n in prior.OBJECTS),local_surface=2),
                synthetic_fps_calls=dict(global_priority=7,local_priority=sum(int((c['local']['quota']>0).sum()) for _,_,c in synthetic.values())),
                original_xyz_source_pixel_object_role_alignment=True,npz_exact_readback=True,
                source_input_hashes_unchanged=True,feature_unavailable_explicit=True,
                same_pair_joint_reference_denominators=True,gt_independence='annotation-free selection helpers + synthetic contract tests',
                performance_improvement_is_not_pass_criterion=True,production_policy_adopted=False,
                skipped=['simulator','renderer','CUDA/GPU','independent episodes/objects/sensors','normal/candidate recalculation',
                    'feature reconstruction','production changes','task relevance scoring','external upload','dependency install','commit'])
            write_json(output/'verification.json',verification)
            # Include placeholder targets before documents are generated; overwrite only files owned by this new run.
            (output/'implementation.patch').touch();(output/(STEM+'.html')).touch()
            create_documents(output,result,test_summary)
            diff=b''
            for path in (Path(__file__),Path(__file__).with_name('test_local_structural_sampling.py'),REPORT):
                proc=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path.relative_to(REPO))],cwd=REPO,stdout=subprocess.PIPE)
                if proc.returncode!=1:raise ValueError('failed to capture implementation patch')
                diff+=proc.stdout
            (output/'implementation.patch').write_bytes(diff)
            verification['document_checks']=verify_documents(output)
            write_json(output/'verification.json',verification)
            for path in sorted(output.rglob('*')):
                if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','config.yaml','metrics.json'):
                    logger.link_output(path,kind='reports')
            logger.link_output(REPORT,kind='reports')
            logger.record(source_run_id=BASELINE_ID,task03_source_run_id=prior.SOURCE_RUN,actual_global_fps_replayed=False,
                          simulator_executed=False,gpu_executed=False,production_policy_adopted=False)
            logger.log({n:{m:dict(candidate_p95_mm=r[m]['spatial']['mm']['raw']['p95'],
                final_refs=r[m]['gt']['stages']['F']['unique_reference_count'],surface_mean_mm=r[m]['gt']['surface_distance_mm']['mean'])
                for m in ('global','local')} for n,r in result['actual'].items()},step=0,prefix='eval/task03f')
            print(output,flush=True)
        except Exception as exc:
            write_json(output/'attempt_failure.json',dict(status='FAIL',exception_type=type(exc).__name__,reason=str(exc),
                no_automatic_reselection=True,existing_attempt_preserved=True))
            print('Failed attempt preserved at '+str(output),flush=True)
            raise
    return output


def postprocess(previous,baseline_path):
    """New, explicitly linked reporting execution. Never calls sampling/evaluation."""
    import ast
    if os.environ.get('WANDB_MODE')!='disabled':raise ValueError('WANDB_MODE=disabled required')
    previous=Path(previous).resolve()
    original=json.loads((previous/(STEM+'.json')).read_text())
    if REPORT.exists() and REPORT.read_bytes()!=(previous/'report.md').read_bytes():
        raise FileExistsError('refuse to replace a report not generated by this numeric attempt')
    baseline=json.loads(Path(baseline_path).read_text());protected(baseline)
    old_tree=ast.parse((previous/'source_snapshot'/Path(__file__).relative_to(REPO)).read_text())
    new_tree=ast.parse(Path(__file__).read_text())
    frozen_names=['spatial_groups','allocate','choose_priority','select_local','spatial_metrics','synthetic_fixture','evaluate_synthetic','load_actual']
    for name in frozen_names:
        old=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        require_equal(ast.dump(new),ast.dump(old),'unchanged selection/metric/fixture definition '+name)
    require_equal(original['config']['rules'],RULES,'unchanged rules')
    require_equal(original['config']['synthetic_fixtures'],FIXTURES,'unchanged fixture constants')
    previous_hashes=prior.hashes([p for p in previous.rglob('*') if p.is_file()])
    config=dict(task='03-F-report-recovery',source_run_id=BASELINE_ID,
        task03_source_run_id=prior.SOURCE_RUN,selection_run_id=original['run_id'],
        numeric_payload_sha256=prior.sha(previous/(STEM+'.npz')),
        fixed_sampling_config=original['config'],this_execution_fps_calls=0,
        source_sha256=prior.hashes([Path(__file__),Path(__file__).with_name('test_local_structural_sampling.py')]))
    with ExperimentLogger(team='visual',config=config,seed=0,model='task03f_uv_report_recovery',job_type='evaluate') as logger:
        output=logger.directory
        for filename in (STEM+'.npz','selection_before_gt.npz','tests.log','fixture_manifest.json',
                'preregistered_config.json','git_start.json','git_start.diff'):
            shutil.copyfile(previous/filename,output/filename)
        for path in previous.glob('*.png'):shutil.copyfile(path,output/path.name)
        shutil.copytree(previous/'source_snapshot',output/'selection_source_snapshot')
        for path in (Path(__file__),Path(__file__).with_name('test_local_structural_sampling.py')):
            dest=output/'source_snapshot'/path.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        if REPORT.exists():
            shutil.copyfile(REPORT,output/'previous_failed_report.md')
            REPORT.unlink()  # Own just-created report, exact archived copy checked above; no pre-task report existed.
        result=dict(original,run_id=logger.run_id,report_recovery=dict(selection_run_id=original['run_id'],
            reason='original Task03 PNG filename link corrected; stored selection/evaluation reused unchanged',
            this_execution_fps_calls=0,performance_recomputed=False))
        write_json(output/(STEM+'.json'),result)
        old_verification=json.loads((previous/'verification.json').read_text())
        verification=dict(old_verification,report_recovery=result['report_recovery'],this_execution_fps_calls=0,
            required_tests=dict(old_verification['required_tests'],execution_run_id=original['run_id']),
            numeric_npz_byte_identical=prior.sha(output/(STEM+'.npz'))==prior.sha(previous/(STEM+'.npz')),
            frozen_algorithm_ast_unchanged=frozen_names,original_preservation=protected(baseline))
        write_json(output/'verification.json',verification)
        write_json(output/'source_hashes.json',dict(selection_execution=original['run_id'],
            selection_source_hashes=json.loads((previous/'source_hashes.json').read_text()),
            report_sources=prior.hashes([Path(__file__),Path(__file__).with_name('test_local_structural_sampling.py')]),
            previous_execution_files_before=previous_hashes))
        write_json(output/'git_end.json',dict(git_status=subprocess.check_output(['git','status','--short'],cwd=REPO).decode()))
        (output/'implementation.patch').touch();(output/(STEM+'.html')).touch()
        create_documents(output,result,old_verification['required_tests']['summary'])
        diff=b''
        for path in (Path(__file__),Path(__file__).with_name('test_local_structural_sampling.py'),REPORT):
            delta=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path.relative_to(REPO))],cwd=REPO,stdout=subprocess.PIPE)
            if delta.returncode!=1:raise ValueError('implementation diff capture failed')
            diff+=delta.stdout
        (output/'implementation.patch').write_bytes(diff)
        verification['document_checks']=verify_documents(output)
        require_equal(prior.hashes([REPO/p for p in previous_hashes]),previous_hashes,'failed numeric execution retained byte-exact')
        verification['original_preservation']=protected(baseline)
        write_json(output/'verification.json',verification)
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','config.yaml','metrics.json'):
                logger.link_output(path,kind='reports')
        logger.link_output(REPORT,kind='reports')
        logger.record(source_run_id=BASELINE_ID,selection_run_id=original['run_id'],
            report_only=True,numeric_npz_byte_identical=True,this_execution_fps_calls=0)
        print(output,flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,type=Path,help='pre-implementation protected hashes and git_start.diff')
    parser.add_argument('--render-saved',type=Path,help='new report-only UUID from a preserved completed numeric attempt; no FPS/evaluation')
    args=parser.parse_args()
    if args.render_saved:postprocess(args.render_saved,args.baseline)
    else:run(args.baseline)


if __name__=='__main__':main()
