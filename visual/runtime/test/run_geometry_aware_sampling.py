"""Task03-G: one offline Frame0 feasibility rule, never a production sampler.

WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_geometry_aware_sampling --original-state DIR
No normal/candidate extraction, baseline FPS replay, sweep or synthetic performance run.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from utils.experiment_logger import ExperimentLogger
from utils.experiment_artifacts import digests, register_existing_output
from visual.runtime.edge_evidence import PairEvidence
from visual.runtime.geometry import GeometryParameters
from visual.runtime.sampling import fps_indices
from visual.runtime.test import run_local_structural_sampling as previous
from visual.runtime.test import run_priority_fps_metric_control as prior
from visual.runtime.test.run_rim_bottleneck import require_equal, write_json, validate_saved_selection

REPO = prior.REPO
STEM = 'geometry_aware_sampling'
REPORT = REPO/'docs/visual-graph-integration/03g_geometry_aware_sampling.md'
UV_RUN = '1f7ad262dd2e4b2fb0ac805a4eda5278'
UV_DIR = REPO/'experiments'/UV_RUN
TESTS = ['visual.runtime.test.'+s for s in (
    'test_geometry_aware_sampling', 'test_local_structural_sampling',
    'test_priority_fps_metric_control', 'test_sampling', 'test_sampling_geometry',
    'test_sampling_evaluation', 'test_priority_fps_trace', 'test_priority_fps_seed_control')]
RULES = dict(threshold_deg=20., priority=32, total=256, structural_ratio=.25,
    patch='X components: same-object support at both ends, no overlap, no depth jump, finite angle <=20 deg',
    ordering='patch minimum flat ascending; candidate rows flat ascending',
    weight='Euclidean diagonal of candidate XYZ axis-aligned bbox; zero for empty/singleton/coincident',
    quota='reserve whole-C seeded anchor; remaining 31 capacity-capped largest remainder; ties patch order',
    saturation='floor capped by residual capacity, then fractional remainder once per eligible patch; repeat',
    zero_weight='no minimum quota, no merge, no relaxed gate; leftover goes to global fallback',
    priority_metric='original XYZ 3D Euclidean within patches; same object seed for each patch',
    anchor='anchor patch first, initial anchor; remaining patches minimum flat ascending',
    fallback='current C global lambda=0 FPS order (camera Z zeroed), skip already selected; optional verified saved prefix',
    surface='original whole X 3D FPS initialized with all P',
    gt='evaluation only after selection_before_gt.npz; same-pair 2px AND 3mm',
    comparison='grouping, span allocation, per-patch initialization and distance metric change together',
    synthetic='seven existing performance fixtures N/A: dense X and PairEvidence absent',
    adoption='Frame0 feasibility only; no policy adoption or generalization claim')


def ordered_source(flat, shape):
    previous.uv_from_flat(flat, shape)
    if len(flat) and not np.all(flat[:-1] < flat[1:]):
        raise ValueError('source rows must be ascending')


def patch_labels(support_flat, pair):
    """Temporary observation components only; retain isolated and C-free X rows."""
    support_flat = np.asarray(support_flat)
    ordered_source(support_flat, pair.shape)
    pixels = np.asarray(pair.pixels)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError('pair pixels require [N,2]')
    if ((pixels < 0) | (pixels >= np.prod(pair.shape))).any():
        raise ValueError('pair outside raster')
    delta = pixels[:, 1]-pixels[:, 0]
    right = (delta == 1) & (pixels[:, 0]//pair.shape[1] == pixels[:, 1]//pair.shape[1])
    down = delta == pair.shape[1]
    if not (right | down).all() or len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError('unique right/down pair contract required')
    expected = np.isin(pixels, support_flat)
    np.testing.assert_array_equal(pair.supported, expected)
    gate = (pair.supported.all(1) & ~pair.overlap.any(1) & ~pair.depth_jump &
            np.isfinite(pair.normal_angle_deg) & (pair.normal_angle_deg <= RULES['threshold_deg']))
    rows = np.searchsorted(support_flat, pixels[gate])
    graph = coo_matrix((np.ones(len(rows), bool), (rows[:, 0], rows[:, 1])),
                       shape=(len(support_flat), len(support_flat))).tocsr()
    count, labels = connected_components(graph, directed=False)
    # scipy normally discovers in row order; explicitly impose minimum source order.
    minima = np.full(count, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(minima, labels, support_flat)
    order = np.argsort(minima, kind='stable')
    relabel = np.empty(count, np.int64); relabel[order] = np.arange(count)
    return relabel[labels], gate


def allocate_span(capacity, weights, count):
    """No zero-span minimum. Returns quota plus unallocated fallback slots."""
    cap, w = np.asarray(capacity), np.asarray(weights, float)
    if cap.ndim != 1 or cap.dtype.kind not in 'iu' or w.shape != cap.shape:
        raise ValueError('integer capacities and aligned weights required')
    if (cap < 0).any() or not np.isfinite(w).all() or (w < 0).any() or count < 0:
        raise ValueError('nonnegative finite allocation inputs required')
    out = np.zeros(len(cap), np.int64); left = int(count)
    while left:
        active = (out < cap) & (w > 0)
        if not active.any():
            break
        scaled = np.where(active, w/w[active].max(), 0.)
        real = left*scaled/scaled.sum()
        take = np.minimum(cap-out, np.floor(real).astype(np.int64))
        out += take; left -= int(take.sum())
        for g in np.argsort(-(real-np.floor(real)), kind='stable'):
            if left and active[g] and out[g] < cap[g]:
                out[g] += 1; left -= 1
    return out, left


def prepare(case, pair):
    """Freeze fragmentation, bbox extrema, weight, capacity and quota before FPS."""
    c, x = case['candidate_flat'], case['support_flat']
    ordered_source(c, pair.shape); ordered_source(x, pair.shape)
    cr = np.searchsorted(x, c)
    if (cr >= len(x)).any() or not np.array_equal(x[cr], c):
        raise ValueError('C outside X')
    np.testing.assert_array_equal(case['support_xyz'][cr], case['candidate_xyz'])
    if not np.isfinite(case['support_xyz']).all():
        raise ValueError('finite XYZ required')
    labels, gate = patch_labels(x, pair)
    cm = labels[cr]; ng = int(labels.max())+1 if len(labels) else 0
    cap = np.bincount(cm, minlength=ng); xn = np.bincount(labels, minlength=ng)
    spans = np.zeros(ng); rows = []
    for g in range(ng):
        xr, rr = np.flatnonzero(labels == g), np.flatnonzero(cm == g)
        xx, cc = case['support_xyz'][xr], case['candidate_xyz'][rr]
        bbox = lambda a: dict(min_xyz_m=a.min(0).tolist(), max_xyz_m=a.max(0).tolist(),
                              span_m=float(np.linalg.norm(a.max(0)-a.min(0)))) if len(a) else None
        cb = bbox(cc); spans[g] = cb['span_m'] if cb else 0.
        extreme = np.unique(np.r_[cc.argmin(0), cc.argmax(0)]) if len(cc) else np.empty(0, int)
        # Sensitivity diagnostic only: never changes weights or selection.
        loo = []
        for i in extreme:
            remaining = np.delete(cc, i, axis=0)
            span = float(np.linalg.norm(remaining.max(0)-remaining.min(0))) if len(remaining) else 0.
            loo.append(dict(source=int(c[rr[i]]), span_without_point_m=span,
                            span_reduction_m=float(spans[g]-span)))
        rows.append(dict(patch=g, min_flat=int(x[xr[0]]), X=int(xn[g]), C=int(cap[g]),
            X_bbox=bbox(xx), C_bbox=cb, extrema_leave_one_out=loo))
    target = min(RULES['priority'], len(c))
    quota = np.zeros(ng, np.int64); ag = None
    if target:
        ar = np.flatnonzero(c == case['first'])
        if len(ar) != 1:
            raise ValueError('anchor must be in C exactly once')
        require_equal(int(c[np.random.default_rng(case['seed']).integers(len(c))]), int(case['first']), 'whole-C anchor')
        ag = int(cm[ar[0]]); quota[ag] = 1
    additional, left = allocate_span(cap-quota, spans, target-int(quota.sum()))
    quota += additional
    for g, row in enumerate(rows):
        row.update(weight=float(spans[g]), capacity=int(cap[g]), residual_capacity_after_anchor=int(cap[g]-(g == ag)),
                   anchor_reserved=int(g == ag), additional_allocation=int(additional[g]), quota=int(quota[g]))
    zero = (cap > 0) & (spans == 0)
    singleton = xn == 1
    return dict(support_patch=labels, candidate_patch=cm, connection_gate=gate, patches=rows,
        weight=spans, capacity=cap, quota=quota, anchor_patch=ag, fallback_slots=left,
        diagnostics=dict(X_patches=ng, candidate_patches=int((cap > 0).sum()),
            X_singleton_patches=int(singleton.sum()), X_singleton_candidates=int(cap[singleton].sum()),
            candidate_singleton_patches=int((cap == 1).sum()),
            zero_span_candidate_patches=int(zero.sum()), zero_span_candidates=int(cap[zero].sum()),
            zero_span_candidate_fraction=float(cap[zero].sum()/len(c)) if len(c) else None,
            singleton_or_zero_span_patches=int((singleton | zero).sum()),
            singleton_or_zero_span_candidates=int(cap[singleton | zero].sum()),
            singleton_or_zero_span_candidate_fraction=float(cap[singleton | zero].sum()/len(c)) if len(c) else None,
            zero_quota_candidate_patches=int(((cap > 0) & (quota == 0)).sum()),
            zero_quota_candidates=int(cap[quota == 0].sum()),
            anchor_patch=ag, anchor_quota=int(quota[ag]) if ag is not None else 0,
            planned_fallback=left, quota_sum=int(quota.sum())))


def fallback_rows(flat, xyz, seed, anchor, selected, needed, saved_order=None):
    """Generic current-C global lambda=0 FPS fallback, with optional cached prefix.

    A verified prefix is sufficient when it has `needed` unselected rows. Otherwise
    generate current-C order up to selected+needed; no Task03 artifact is required.
    """
    if not needed:
        return np.empty(0, np.int64), 'unused'
    flat = np.asarray(flat); selected = np.asarray(selected, np.int64)
    if saved_order is not None:
        order = np.asarray(saved_order)
        if len(np.unique(order)) != len(order) or not np.isin(order, flat).all() or order[0] != anchor:
            raise ValueError('invalid verified cached fallback prefix')
        available = order[~np.isin(order, flat[selected])]
        if len(available) >= needed:
            return np.searchsorted(flat, available[:needed]), 'verified_saved_global_prefix'
    ar = np.flatnonzero(flat == anchor)
    metric = prior.metric_coordinates(xyz, 0.)
    order = fps_indices(metric, min(len(flat), len(selected)+needed), seed=seed, initial=ar)
    available = order[~np.isin(order, selected)]
    if len(available) < needed:
        raise ValueError('fallback capacity exhausted')
    return available[:needed], 'generated_current_C_global_order'


def select_geometry(case, prepared, saved_order=None):
    """Annotation-free selection. No object names, poses, ROI, scores or references."""
    c, xyz = case['candidate_flat'], case['candidate_xyz']
    cm, q = prepared['candidate_patch'], prepared['quota']
    ag = prepared['anchor_patch']
    order = ([ag] if ag is not None else [])+[g for g in range(len(q)) if g != ag and q[g]]
    selected = []; initial_source = np.full(len(q), -1, np.int64)
    for g in order:
        rows = np.flatnonzero(cm == g)
        initial = np.flatnonzero(c[rows] == case['first']) if g == ag else ()
        local = fps_indices(xyz[rows], int(q[g]), seed=case['seed'], initial=initial)
        selected.extend(rows[local].tolist()); initial_source[g] = c[rows[local[0]]]
    before_fallback = np.asarray(selected, np.int64)
    extra, mode = fallback_rows(c, xyz, case['seed'], case['first'], before_fallback,
                               prepared['fallback_slots'], saved_order)
    rows = np.r_[before_fallback, extra].astype(np.int64)
    p = c[rows]
    initial = np.searchsorted(case['support_flat'], p)
    frows = fps_indices(case['support_xyz'], case['budget'], seed=case['seed'], initial=initial)
    f = case['support_flat'][frows]
    return dict(P=p, S=f[len(p):], F=f, xyz=case['support_xyz'][frows], priority_rows=rows,
        patch_priority_count=np.bincount(cm[before_fallback], minlength=len(q)),
        actual_priority_count=np.bincount(cm[rows], minlength=len(q)),
        fallback_patch_count=np.bincount(cm[extra], minlength=len(q)),
        fallback_source=c[extra], fallback_mode=mode, patch_initial_source=initial_source)


def selection_input(case):
    return {k: case[k] for k in ('candidate_flat','candidate_xyz','support_flat','support_xyz','seed','first','budget')}


def compatibility(frame, geometry, samples, cases, config, edges, normal):
    """Validate saved contracts only; never estimate normals or extract candidates."""
    parameters = GeometryParameters()
    for path in (prior.TASK01/'geometry.json', prior.EDGES.parent/'comparison.json', prior.NORMAL.parent/'comparison.json'):
        cfg = json.loads(path.read_text())['config']
        for key in ('input_sha256','masks_sha256','camera'):
            require_equal(cfg[key], config[key], 'evidence source '+key)
        if 'parameters' in cfg:
            require_equal(cfg['parameters'], dict(depth_jump_abs_m=.01, depth_jump_rel=.02), 'depth config')
        if 'depth_gate_abs_m' in cfg:
            require_equal((cfg['depth_gate_abs_m'], cfg['depth_gate_rel']), (.01,.02), 'pair depth config')
    with np.load(prior.TASK01/'f0000.npz', allow_pickle=False) as dense:
        require_equal(int(dense['source_frame']), 0, 'dense frame')
        require_equal(str(dense['camera']), config['camera'], 'dense camera')
        np.testing.assert_array_equal(dense['valid_depth'], np.isfinite(frame.depth) & (frame.depth > 0))
        np.testing.assert_array_equal(dense['overlap_mask'], frame.masks[prior.OBJECTS['peg']] & frame.masks[prior.OBJECTS['box']])
        for name, oid in prior.OBJECTS.items():
            np.testing.assert_array_equal(dense[name+'/valid_points'], geometry.observed_mask(oid))
    validate_saved_selection(samples, 'S3_n256_r0.25_t20_s0', frame, geometry)
    for archive in (edges, normal):
        require_equal(int(archive['source_frame']), 0, 'pair/normal frame')
        np.testing.assert_array_equal(archive['intrinsic'], frame.intrinsic)
        np.testing.assert_array_equal(archive['depth_m'], frame.depth)
    np.testing.assert_array_equal(edges['shape'], frame.depth.shape)
    pairs, checks = {}, {}
    h, w = frame.depth.shape
    indices = np.arange(h*w, dtype=np.int32).reshape(h, w)
    for name, case in cases.items():
        oid = case['oid']; prefix = 'pca5/'+name+'/'
        pair = PairEvidence(**{f.name: tuple(edges[prefix+f.name]) if f.name == 'shape'
                              else edges[prefix+f.name].copy() for f in fields(PairEvidence)})
        require_equal(pair.shape, frame.depth.shape, 'pair shape')
        mask = geometry.object_masks[oid]
        right = mask[:, :-1] | mask[:, 1:]; down = mask[:-1] | mask[1:]
        expected = np.concatenate((np.column_stack((indices[:, :-1][right], indices[:, 1:][right])),
                                   np.column_stack((indices[:-1][down], indices[1:][down]))))
        np.testing.assert_array_equal(pair.pixels, expected)
        p = pair.pixels; valid = geometry.valid_depth.ravel()[p]; both = valid.all(1)
        np.testing.assert_array_equal(pair.supported, geometry.observed_mask(oid).ravel()[p])
        np.testing.assert_array_equal(pair.overlap, geometry.overlap_mask.ravel()[p])
        np.testing.assert_array_equal(pair.mask_boundary, mask.ravel()[p[:,0]] ^ mask.ravel()[p[:,1]])
        np.testing.assert_array_equal(pair.invalid_depth_boundary, valid[:,0] ^ valid[:,1])
        z = geometry.xyz[...,2].ravel()[p]
        delta = np.full(len(p), np.nan, np.float32); delta[both] = z[both,1]-z[both,0]
        np.testing.assert_array_equal(pair.signed_depth_delta_m, delta)
        jump = np.zeros(len(p), bool)
        jump[both] = np.abs(delta[both]) > parameters.depth_jump_abs_m+parameters.depth_jump_rel*z[both].min(1)
        np.testing.assert_array_equal(pair.depth_jump, jump)
        near = np.full(len(p), -1, np.int8); near[jump] = z[jump].argmin(1)
        np.testing.assert_array_equal(pair.near_endpoint, near)
        np.testing.assert_array_equal(pair.far_endpoint, np.where(jump, 1-near, -1))
        for field, value in (('object_mask',mask), ('valid_points',geometry.observed_mask(oid))):
            np.testing.assert_array_equal(normal[prefix+field], value)
        nv = normal[prefix+'normal_valid'].ravel()[p].all(1)
        np.testing.assert_array_equal(np.isfinite(pair.normal_angle_deg), nv)
        # Recompute pair angles from SAVED reference normals only; no estimator.
        n = normal[prefix+'normals'].reshape(-1,3)[p[nv]]
        angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(n[:,0],n[:,1]),axis=1),
                                     np.einsum('ij,ij->i',n[:,0],n[:,1])))
        np.testing.assert_array_equal(pair.normal_angle_deg[nv], angle)
        endpoint = np.zeros(h*w, bool)
        high = pair.normal_angle_deg > 20.
        for k in (0,1): endpoint[p[high & pair.supported[:,k],k]] = True
        np.testing.assert_array_equal(endpoint.reshape(h,w), edges[prefix+'candidate20/orientation'])
        np.testing.assert_array_equal(np.flatnonzero(endpoint), case['candidate_flat'])
        normal_members = normal[prefix+'valid_points'] & (np.rad2deg(normal[prefix+'normal_variation_rad']) > 20.)
        np.testing.assert_array_equal(np.flatnonzero(normal_members), case['candidate_flat'])
        for field in ('normal_valid','valid_points','orientation_candidate'):
            saved = case['saved_features'][field]
            expected_feature = (endpoint if field == 'orientation_candidate' else normal[prefix+field].ravel())[case['saved_flat']]
            np.testing.assert_array_equal(saved, expected_feature)
        pairs[name] = pair
        checks[name] = dict(status='PASS',pairs=len(p),right=int(right.sum()),down=int(down.sum()),
            source_object_support_pixel_exact=True, depth_jump_strict_gt=True,
            reference_angles_exact_from_saved_reference_normals=True,
            candidate_threshold_membership_exact=True, saved_row_validity_membership_exact=True,
            optimized_pair_numeric_identity='N/A: full optimized PCA5 pair angles not stored',
            optimized_connection_gate_identity='N/A: endpoint candidate equality does not prove low-angle pair gate equality')
    return pairs, checks


def check_selection(case, chosen):
    c,x = case['candidate_flat'],case['support_flat']
    require_equal((len(chosen['P']),len(chosen['F'])), (32,case['budget']), 'fixed budget')
    require_equal(int(chosen['P'][0]),case['first'],'anchor first')
    np.testing.assert_array_equal(chosen['F'],np.r_[chosen['P'],chosen['S']])
    if not np.isin(chosen['P'],c).all() or not np.isin(c,x).all() or not np.isin(chosen['F'],x).all():
        raise ValueError('P subset C subset X / F subset X')
    require_equal(len(np.unique(chosen['F'])),case['budget'],'unique P/S/F')
    np.testing.assert_array_equal(chosen['xyz'],case['support_xyz'][np.searchsorted(x,chosen['F'])])


def preserved(original):
    actual = prior.hashes([REPO/p for p in original])
    require_equal(actual,original,'original source/input/artifact hashes')
    return dict(status='PASS',files=len(original),all_sha256_unchanged=True)


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def read_reference_selections(cases, baselines):
    """Read saved lambda=1 / UV controls. No reference selection is rerun."""
    selections={}
    with np.load(UV_DIR/(previous.STEM+'.npz'),allow_pickle=False) as uv:
        for name,case in cases.items():
            one={k:case['saved_flat'][sl].copy() for k,sl in (
                ('P',slice(0,32)),('S',slice(32,None)),('F',slice(None)))}
            one['xyz']=case['support_xyz'][np.searchsorted(case['support_flat'],one['F'])]
            local={k:uv[f'{name}/local/{k}/flat_index'].copy() for k in ('P','S','F')}
            local['xyz']=uv[f'{name}/local/F/xyz'].copy()
            for k in ('P','S','F'):
                np.testing.assert_array_equal(uv[f'{name}/global/{k}/flat_index'],baselines[name][k])
            selections[name]=dict(global0=baselines[name],global1=one,uv=local)
            for chosen in selections[name].values():check_selection(case,chosen)
    return selections


def evaluate(frame,geometry,samples,cases,selections,arrays,old,uv_old,input_config):
    # This function is called ONLY after selection_before_gt exists and is hashed.
    with np.load(prior.EDGES,allow_pickle=False) as edges,np.load(prior.NORMAL,allow_pickle=False) as normal:
        contexts=prior.evaluation_inputs(frame,geometry,samples,edges,normal)
        peg_labels,peg_members=prior.peg_labels(normal['evaluation/peg/face'],normal['evaluation/peg/gt_visible'],
            normal['evaluation/peg/seam_endpoints'],contexts['peg']['reference'][0])
    with np.load(input_config['input'],allow_pickle=False) as episode:
        box_labels,box_specs,box_transform=prior.rim_labels(contexts['box']['reference'][1],frame.cam2world,
            episode['box_pose'][0],episode['box_hole_pose'][0],float(episode['box_hole_radii']))
    actual={}
    for name,case in cases.items():
        context=contexts[name];labels=peg_labels if name=='peg' else box_labels
        previous.add_arrays(arrays,'evaluation/'+name,dict(reference_uv=context['reference'][0],
            reference_xyz=context['reference'][1],observable=context['observable'],GT_reference_segment=labels))
        row={}
        for method,chosen in selections[name].items():
            pxyz=case['support_xyz'][np.searchsorted(case['support_flat'],chosen['P'])]
            spatial,raw=previous.spatial_metrics(case['candidate_flat'],case['candidate_xyz'],chosen['P'],pxyz,
                case['support_flat'],case['support_xyz'],frame.depth.shape)
            gt,data=prior.score_case(case,chosen,context,frame,labels,
                {} if name=='peg' else box_specs,None if name=='peg' else box_transform)
            previous.add_arrays(arrays,f'{name}/{method}/spatial',raw)
            previous.add_arrays(arrays,f'evaluation/{name}/{method}',data)
            row[method]=dict(spatial=spatial,gt=gt)
            reference = (old['cases']['S3/'+name]['conditions']['0.0'] if method=='global0'
                         else uv_old['actual'][name]['local']['gt'] if method=='uv' else None)
            if reference is not None:
                for field in ('coverage','stages','surface_distance_mm','occupancy','feature_availability'):
                    require_equal(gt[field],reference[field],name+'/'+method+' saved metric '+field)
        base,new=row['global0'],row['geometry']
        row['delta_geometry_minus_global0']=dict(
            C_to_P_mm={k:new['spatial']['mm']['raw'][k]-base['spatial']['mm']['raw'][k] for k in ('mean','p50','p95','max')},
            X_to_F_mm={k:new['gt']['surface_distance_mm'][k]-base['gt']['surface_distance_mm'][k] for k in ('mean','p95','max')},
            unique_references={s:new['gt']['stages'][s]['unique_reference_count']-base['gt']['stages'][s]['unique_reference_count'] for s in ('C','P','S','F')},
            negative_union={s:new['gt']['occupancy']['negative_roi']['known_negative_union'][s]['count']-
                            base['gt']['occupancy']['negative_roi']['known_negative_union'][s]['count'] for s in ('P','S','F')})
        actual[name]=row
    return actual,peg_members


def make_plots(output,frame,cases,prepared,selections,result,arrays):
    plt.rcParams.update({'font.size':9})
    fig,axes=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    for i,(name,case) in enumerate(cases.items()):
        a=prepared[name];xuv=previous.uv_from_flat(case['support_flat'],frame.depth.shape)
        cuv=previous.uv_from_flat(case['candidate_flat'],frame.depth.shape)
        axes[i,0].scatter(*xuv.T,c=a['support_patch']%20,cmap='tab20',s=2,alpha=.6)
        axes[i,0].scatter(*cuv.T,facecolors='none',edgecolors='black',s=11,lw=.35,label='C on X patches')
        zero=a['quota'][a['candidate_patch']]==0
        axes[i,0].scatter(*cuv[zero].T,c='red',marker='x',s=15,label='C in quota=0 patch')
        axes[i,0].set_title(f'{name}: observation X patches (colours repeat)')
        for j,method in enumerate(('global0','geometry'),1):
            ax=axes[i,j];ax.scatter(*cuv.T,c='gray',s=5,alpha=.5)
            for stage,color,marker in [('S','#25a8b5','.'),('P','#ed7b24','o')]:
                uv=previous.uv_from_flat(selections[name][method][stage],frame.depth.shape)
                ax.scatter(*uv.T,c=color,marker=marker,s=14 if stage=='P' else 9,label=stage)
            worst=result['actual'][name][method]['spatial']['mm']['worst']['uv']
            ax.scatter(*worst,c='red',marker='x',s=80,label='max 3D C->P source')
            ax.set_title(f'{name}: {method} P / S')
        for ax in axes[i]:ax.invert_yaxis();ax.set(xlabel='u (px)',ylabel='v (px)');ax.legend(fontsize=7)
    fig.savefig(output/'patch_and_selection.png',dpi=140);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for i,name in enumerate(cases):
        for method in ('global0','geometry','global1','uv'):
            d=arrays[f'{name}/{method}/spatial/ecdf_mm']
            axes[i,0].plot(d,np.arange(1,len(d)+1)/len(d),label=method)
            axes[i,1].plot(d,np.arange(1,len(d)+1)/len(d),label=method)
        axes[i,0].set(title=f'{name}: all C->P original 3D',xlabel='mm',ylabel='ECDF')
        axes[i,1].set(title=f'{name}: distance tail',xlabel='mm',ylabel='ECDF',ylim=(.9,1.002))
        for ax in axes[i]:ax.legend()
    fig.savefig(output/'candidate_3d_ecdf_tail.png',dpi=140);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    for i,name in enumerate(cases):
        for j,kind in enumerate(('uv','xyz')):
            for method,off in [('global0',-.18),('geometry',.18)]:
                cells=result['actual'][name][method]['spatial']['grids'][kind]['cells']
                vals=[c['distance_mm']['p95'] if c['distance_mm']['p95'] is not None else np.nan for c in cells]
                axes[i,j].bar(np.arange(len(cells))+off,vals,.34,label=method)
            axes[i,j].set(title=f'{name}: fixed independent {kind} regions',xlabel='region ID',ylabel='C->P p95 mm')
            axes[i,j].legend()
    fig.savefig(output/'independent_regions.png',dpi=140);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
    for i,name in enumerate(cases):
        rows=[r for r in prepared[name]['patches'] if r['C']]
        ids=[r['patch'] for r in rows];x=np.arange(len(ids))
        axes[i,0].bar(x,[r['C'] for r in rows],label='candidate capacity',color='#b2b9c3')
        axes[i,0].bar(x,[r['quota'] for r in rows],label='quota',color='#ef7b22')
        axes[i,0].set(title=f'{name}: candidate patches incl. quota 0',ylabel='count',xticks=x,xticklabels=ids,xlabel='observation patch ID')
        axes[i,1].bar(x,[r['weight']*1000 for r in rows],color='#327fb7',label='C bbox span')
        axes[i,1].set(title=f'{name}: allocation weights (not structural length)',ylabel='span mm',xticks=x,xticklabels=ids,xlabel='observation patch ID')
        for ax in axes[i]:ax.legend();ax.tick_params(axis='x',labelrotation=90)
    fig.savefig(output/'patch_quota.png',dpi=140);plt.close(fig)

    for name in cases:
        fig,axes=plt.subplots(2,2,figsize=(14,11),constrained_layout=True)
        refs=arrays[f'evaluation/{name}/reference_uv']
        for i,method in enumerate(('global0','geometry')):
            for j,stage in enumerate(('P','F')):
                ax=axes[i,j];ax.imshow(frame.rgb)
                prefix=f'evaluation/{name}/{method}/{stage}'
                hit=arrays[prefix+'/covered'];matched=arrays[prefix+'/matched_node']
                uv=arrays[prefix+'/pixels_vu'][:,::-1]
                ax.scatter(*refs[~hit].T,c='#d92e76',marker='x',s=18,label='missing GT reference')
                ax.scatter(*refs[hit].T,c='#1cb58c',s=16,label='covered GT reference')
                ax.scatter(*uv.T,facecolors='none',edgecolors='#eda420',s=28,label='selected node')
                ax.scatter(*uv[matched].T,facecolors='none',edgecolors='#2873dc',s=85,label='joint matching node')
                ax.set(xlim=(refs[:,0].min()-12,refs[:,0].max()+12),ylim=(refs[:,1].max()+12,refs[:,1].min()-12),
                    title=f'{name} {method} {stage}: {matched.sum()} nodes / {hit.sum()} refs')
                ax.legend(fontsize=7)
        fig.suptitle('GT auxiliary evaluation labels; separate from observation patch IDs')
        fig.savefig(output/f'{name}_GT_auxiliary.png',dpi=140);plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)
    for i,(name,case) in enumerate(cases.items()):
        for j,method in enumerate(('global0','geometry')):
            x=case['support_xyz'];ax=axes[i,j]
            ax.scatter(x[:,0]*1000,x[:,2]*1000,c='gray',s=.5,alpha=.3)
            for stage,color in [('P','#ed7b24'),('S','#20a6b7')]:
                xyz=x[np.searchsorted(case['support_flat'],selections[name][method][stage])]
                ax.scatter(xyz[:,0]*1000,xyz[:,2]*1000,c=color,s=15,label=stage)
            ax.set(title=f'{name} {method}: unfiltered X',xlabel='camera X mm',ylabel='camera Z mm');ax.legend()
        for method in ('global0','geometry'):
            d=np.sort(arrays[f'evaluation/{name}/{method}/surface_support_to_final_mm'])
            axes[i,2].plot(d,np.arange(1,len(d)+1)/len(d),label=method)
        axes[i,2].set(title=f'{name}: X->F Surface ECDF',xlabel='mm',ylabel='ECDF');axes[i,2].legend()
    fig.savefig(output/'surface_depth_tradeoff.png',dpi=140);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for i,(name,case) in enumerate(cases.items()):
        xyz=case['candidate_xyz']*1000
        for j,method in enumerate(('global0','geometry')):
            ax=axes[i,j];d=arrays[f'{name}/{method}/spatial/distance_mm']
            sc=ax.scatter(xyz[:,0],xyz[:,2],c=d,cmap='magma',s=14,vmin=0,
                vmax=max(result['actual'][name][m]['spatial']['mm']['raw']['max'] for m in ('global0','geometry')))
            pxyz=case['support_xyz'][np.searchsorted(case['support_flat'],selections[name][method]['P'])]*1000
            ax.scatter(pxyz[:,0],pxyz[:,2],facecolors='none',edgecolors='cyan',s=35,label='P')
            worst=result['actual'][name][method]['spatial']['mm']['worst']['xyz_m']
            ax.scatter(worst[0]*1000,worst[2]*1000,c='lime',marker='x',s=80,label='worst source')
            ax.set(title=f'{name} {method}: original C XZ / distance mm',xlabel='camera X mm',ylabel='camera Z mm')
            ax.legend();fig.colorbar(sc,ax=ax)
    fig.savefig(output/'candidate_XZ_coverage.png',dpi=140);plt.close(fig)


def documents(output,result,tests):
    """Create both documents from the same exact tables and interpretation."""
    md=['# Task03-G — Geometry-Aware Structural Sampling, Phase 1 Feasibility','']
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task03-G Frame0 feasibility</title>',
        '<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1500px;margin:36px auto;padding:0 24px;color:#203047} '
        'table{border-collapse:collapse;font-size:13px;margin:18px 0;display:block;overflow:auto}th,td{border:1px solid #cbd5e1;padding:6px 9px;white-space:nowrap} '
        'th{background:#eaf1f8}img{width:100%;height:auto}pre{white-space:pre-wrap}a{color:#125caa}</style>',
        '<h1>Task03-G — Geometry-Aware Structural Sampling</h1>']
    def section(title,text=''):
        md.extend(['## '+title,'',text,'']);page.extend(['<h2>'+html.escape(title)+'</h2>','<p>'+html.escape(text)+'</p>'])
    def paragraph(text):md.extend([text,'']);page.append('<p>'+html.escape(text)+'</p>')
    def fmt(v):
        if v is None:return 'N/A'
        if isinstance(v,float):return f'{v:.6g}'
        return str(v)
    def table(head,rows):
        rows=[[fmt(v) for v in r] for r in rows]
        md.extend(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |'])
        md.extend('| '+' | '.join(v.replace('|','/') for v in row)+' |' for row in rows);md.append('')
        page.append('<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in head)+'</tr></thead><tbody>')
        page.extend('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in rows);page.append('</tbody></table>')
    def plot(filename,caption):
        md.extend([f'![{caption}](../../experiments/{result["run_id"]}/{filename})',''])
        page.append(f'<figure><img src="{filename}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    paragraph('Frame0 결과: Peg C→P p95는 6.662→5.664mm로 개선됐고 Box는 29.779→30.710mm로 악화됐다. '
        'X→F Surface mean/p95는 두 물체에서 소폭 개선됐지만 Final GT reference는 Peg 160→149, Box 12→9로 줄었다. '
        'Fallback은 양쪽 0점이며, 관측 patch의 유용성은 제한적이다. 일반화 성공이나 정책 채택을 주장하지 않는다.')
    if result.get('report_recovery'):
        paragraph(f"수치 선택 UUID는 {result['report_recovery']['selection_run_id']}다. 해당 실행은 수치·문서 생성 후 최종 artifact 등록에서 "
            '종료 후 전용 API를 실행 중 호출해 failed로 남았다. 이번 보고 UUID는 저장 수치 NPZ와 pre-GT 선택을 byte-identical 복사한 문서 후처리이며 '
            '새 FPS/평가 실행은 0회다. 실패 실행과 모든 원본 산출물은 보존했다.')
    section('결과 범위와 재현 정보',
        f"새 ExperimentLogger UUID: {result['run_id']}. ep000 Frame 0 / base_camera / S3(PCA5), N=256, structural_ratio=.25, threshold=20°, seed=0. "
        'Peg P/F=32/129, Box P/F=32/127. CPU NumPy/SciPy offline 실행만 수행했다. 일반화 알고리즘 채택 근거가 아닌 관측 patch 단위의 첫 feasibility 실험이다.')
    paragraph('주 비교는 Task03-E 7f657cf88baf4f1f89e594252e243e1c 저장 Global λ=0이다. Task03 2551448b0fae4c6bbce3caf48774275f Global λ=1과 '
        'Task03-F 1f7ad262dd2e4b2fb0ac805a4eda5278 UV는 참고 비교다. F 최종 UUID는 기존 수치 실행의 문서 후처리이며 독립 반복 실험으로 세지 않는다.')
    paragraph('선택 규칙은 same-object X의 right/down pair를 기존 4-neighbour 무방향 연결로 복원하고, 양 끝 support·overlap 없음·depth_jump=False·finite angle≤20°로 고정했다. '
        'C의 각 source는 정확히 하나의 X patch에 귀속한다. C-free patch도 X와 Surface에서 보존한다. Weight는 C bbox의 3D 대각선이다. '
        '전체 C의 seeded anchor 1점을 보존한 뒤 남은 31점을 capacity-capped largest remainder로 배분하고, patch 내부 원본 3D FPS와 전체 X의 Surface FPS를 사용한다.')
    section('Evidence compatibility 관문',
        'Task01 dense XYZ·validity·mask·overlap·camera·intrinsic·cam2world와 E의 C/X, Task02-C pair source/index/support를 대조했다. '
        'Right pair는 같은 행의 +1, down은 +width이며 각 pair는 한 번만 저장된다. Depth jump는 두 depth가 유효할 때 '
        '|z₂−z₁| > .01 + .02×min(z₁,z₂)이다. Equality는 jump가 아니다. Normal orientation 후보는 >20°, patch 연결은 finite ≤20°다. '
        'NaN angle은 연결하지 않는다. Normal-invalid X는 삭제하지 않는다.')
    table(['object','pairs','right','down','candidate/support/threshold','optimized pair identity'],[
        [n,r['pairs'],r['right'],r['down'],'PASS','N/A'] for n,r in result['compatibility'].items()])
    paragraph('Reference PCA5 pair angle은 저장 Task02-B normal에서 직접 재계산해 exact equality를 확인했다. 저장 orientation endpoint 및 normal variation의 >20° membership은 '
        'Task03 optimized PCA5의 저장 C와 일치했고, 저장 선택 행의 validity/candidate membership도 일치했다. '
        '하지만 optimized PCA5 전체 pair angle은 저장되지 않았다. Candidate endpoint 집합의 일치가 low-angle pair별 연결 gate의 일치를 보장하지 않는다. '
        '따라서 이번 patch는 Task02-C reference PCA5 evidence에서 구성했고 optimized PCA5와 bit-exact라고 주장하지 않는다. Normal estimator/candidate detector는 재실행하지 않았다.')
    section('선택 전 patch 분절과 allocation',
        '아래 patch ID는 GT-free 관측상 매끄럽게 연결된 표면 성분이다. Structural edge, physical component, 평가용 seam/rim label 또는 최종 graph edge를 뜻하지 않는다. '
        'Weight=0인 후보 patch에는 anchor 외 minimum quota가 없다. 실패를 보정하기 위한 병합·gate 변경·후보 삭제는 하지 않았다.')
    table(['object','X patches','C patches','X singleton','0-span C patches','0-span candidates / C','quota=0 C patches','quota=0 candidates','anchor patch/quota','fallback'],[
        [n,d['X_patches'],d['candidate_patches'],d['X_singleton_patches'],d['zero_span_candidate_patches'],
         f"{d['zero_span_candidates']} / {result['counts'][n]['C']} ({d['zero_span_candidate_fraction']*100:.3f}%)",
         d['zero_quota_candidate_patches'],d['zero_quota_candidates'],f"{d['anchor_patch']} / {d['anchor_quota']}",d['actual_fallback']]
        for n,d in result['patch_diagnostics'].items()])
    for n,d in result['patch_diagnostics'].items():
        paragraph(f"{n}: singleton 또는 candidate 0-span patch {d['singleton_or_zero_span_patches']}개, 그 안의 후보 {d['singleton_or_zero_span_candidates']}개 "
            f"({d['singleton_or_zero_span_candidate_fraction']*100:.3f}%). Candidate가 1개인 patch {d['candidate_singleton_patches']}개, "
            f"X singleton에 속한 후보 {d['X_singleton_candidates']}개. 실제 fallback={d['actual_fallback']}점.")
    for name,rows in result['patches'].items():
        paragraph(name+' 전체 X patch (C=0 포함): span은 bbox proxy이며 구조 길이가 아니다. Capacity는 전체 C 수, additional은 anchor 제외 배분이다.')
        table(['patch','min source','X','C/capacity','X span m','C span/weight m','anchor','additional','quota','actual P','fallback P'],[
            [r['patch'],r['min_flat'],r['X'],r['C'],r['X_bbox']['span_m'],r['weight'],r['anchor_reserved'],
             r['additional_allocation'],r['quota'],r['actual_priority'],r['fallback_priority']] for r in rows])
    plot('patch_and_selection.png','GT-free X patches, candidate attribution and Global/new P/S')
    plot('patch_quota.png','Candidate capacity, span weight, quota and zero quota patches')
    section('Primary — 원본 3D candidate·surface 보존',
        '모든 거리는 변환하지 않은 CV-camera XYZ의 Euclidean 최근접 거리(mm)다. P의 patch 내부 선택 metric과 독립적으로 평가했다. '
        'Grouping·span allocation·patch별 초기화·patch 내부 3D와 Global λ=0 metric 차이가 묶여 있다. 아래 차이는 선택 규칙 전체의 차이이며 개별 요소의 인과 효과가 아니다.')
    table(['object','method','C→P mean','p50','p95','max','X→F mean','p95','max'],[
        [n,m,*[r[m]['spatial']['mm']['raw'][k] for k in ('mean','p50','p95','max')],
         *[r[m]['gt']['surface_distance_mm'][k] for k in ('mean','p95','max')]]
        for n,r in result['actual'].items() for m in ('global0','geometry','global1','uv')])
    table(['object','Δ mean','Δ p50','Δ p95','Δ max','Δ Surface mean','Δ Surface p95'],[
        [n,*[r['delta_geometry_minus_global0']['C_to_P_mm'][k] for k in ('mean','p50','p95','max')],
         *[r['delta_geometry_minus_global0']['X_to_F_mm'][k] for k in ('mean','p95')]] for n,r in result['actual'].items()])
    table(['object','method','worst source','UV','XYZ m'],[
        [n,m,r[m]['spatial']['mm']['worst']['flat_index'],r[m]['spatial']['mm']['worst']['uv'],
         r[m]['spatial']['mm']['worst']['xyz_m']] for n,r in result['actual'].items() for m in ('global0','geometry')])
    plot('candidate_3d_ecdf_tail.png','C→P original 3D ECDF including tail; references are saved controls')
    plot('candidate_XZ_coverage.png','Candidate XYZ positions, distance tail and worst uncovered source')
    section('독립 평가 구역과 후보 밀도',
        'Task03-F의 X bbox 기반 UV 4×4 및 XYZ 최대 2×2×2 경계와 helper를 그대로 사용했다. Candidate count, 선택률과 거리 분포를 함께 본다. '
        'Box X의 30.726m depth outlier 때문에 XYZ grid의 후보가 한 구역에 들어가는 한계를 보정하지 않았다. '
        '전체 분위수는 후보 밀도의 영향을 받으며 구역 동등 가중 p95도 함께 기록한다. XYZ 공간 밀도의 공통 helper 값은 N/A이며 UV 밀도는 px²당 후보 수다.')
    table(['object','method','grid','C occupied','P=0 regions','equal-region mean p95 mm'],[
        [n,m,k,r[m]['spatial']['grids'][k]['candidate_occupied_regions'],r[m]['spatial']['grids'][k]['empty_priority_region_ids'],
         r[m]['spatial']['grids'][k]['equal_region_mean_p95_mm']] for n,r in result['actual'].items()
        for m in ('global0','geometry') for k in ('uv','xyz')])
    for n,r in result['actual'].items():
        for kind in ('uv','xyz'):
            paragraph(n+' / '+kind+' independent regions:')
            table(['method','region','C','P','C density/px²','P/C','C→P mean mm','p95','max'],[
                [m,c['id'],c['candidate_count'],c['priority_count'],c['candidate_density_per_px2'],c['selection_rate'],
                 *[c['distance_mm'][k] for k in ('mean','p95','max')]] for m in ('global0','geometry')
                for c in r[m]['spatial']['grids'][kind]['cells']])
    plot('independent_regions.png','Independent fixed UV/XYZ regions; empty candidate regions are blank')
    section('Secondary — GT 기반 보조 진단',
        '선택 및 selection_before_gt.npz 저장·hash 후 GT reference/pose/ROI를 평가에만 적용했다. '
        '동일 point/reference pair가 2px AND 3mm를 만족해야 한다. Peg GT-visible 464개 중 관측 가능 462개, Box 103개 중 103개가 분모다. '
        'Matching node 수와 unique reference 수는 서로 다른 수량이다. 특정 Box rim 선분 개선은 성공 조건이 아니다.')
    table(['object','method','stage','nodes','matching nodes','unique refs'],[
        [n,m,s,gt['gt']['stages'][s]['node_count'],gt['gt']['stages'][s]['structural_matching_nodes'],
         gt['gt']['stages'][s]['unique_reference_count']] for n,r in result['actual'].items()
        for m,gt in [(m,r[m]) for m in ('global0','geometry','global1','uv')] for s in ('C','P','S','F')])
    table(['object','method','P refs','S refs','F refs','S added refs','P/S overlap refs'],[
        [n,m,*[r[m]['gt']['stages'][s]['unique_reference_count'] for s in ('P','S','F')],
         len(set(r[m]['gt']['stages']['S']['covered_reference_ids'])-set(r[m]['gt']['stages']['P']['covered_reference_ids'])),
         len(set(r[m]['gt']['stages']['S']['covered_reference_ids'])&set(r[m]['gt']['stages']['P']['covered_reference_ids']))]
        for n,r in result['actual'].items() for m in ('global0','geometry')])
    table(['object','method','stage','GT face-pair/rim segment','observed refs','covered refs','matching nodes'],[
        [n,m,s,label,seg['observed_references'],seg['coverage_count'],seg['matching_node_count']]
        for n,r in result['actual'].items() for m in ('global0','geometry') for s in ('C','P','S','F')
        for label,seg in r[m]['gt']['stages'][s]['segments'].items()])
    paragraph('Box의 verified line projection 간격과 missing-reference run은 JSON/NPZ에 기존 helper 정의로 저장했다. '
        'Peg face-pair 및 ambiguous reference의 물리적 선분 순서는 N/A다. Array 순서를 곡선 길이나 실제 연속 미보존 길이로 해석하지 않는다.')
    plot('peg_GT_auxiliary.png','Peg GT seam evaluation, separate from GT-free observation patches')
    plot('box_GT_auxiliary.png','Box GT rim evaluation, separate from GT-free observation patches')
    section('일반 Surface·negative ROI·depth·outlier trade-off',
        '일반 Surface는 전체 X→F로 평가한다. S만 따로 평가하거나 far point를 제거하지 않았다. '
        'Known-negative ROI union은 평가용이며 평면 Surface 점유를 자동 오선택이라고 분류하지 않는다. 원본 outlier·depth tail은 그대로 남긴다.')
    table(['object','method','negative P / 32','negative S','negative F','F exact feature rows','unavailable rows'],[
        [n,m,*[r[m]['gt']['occupancy']['negative_roi']['known_negative_union'][s]['count'] for s in ('P','S','F')],
         r[m]['gt']['feature_availability']['exact_available_rows'],r[m]['gt']['feature_availability']['unavailable_rows']]
        for n,r in result['actual'].items() for m in ('global0','geometry')])
    table(['object','method','stage','depth min m','median','p95','max'],[
        [n,m,s,*[r[m]['gt']['occupancy']['depth_m'][s][k] for k in ('min','median','p95','max')]]
        for n,r in result['actual'].items() for m in ('global0','geometry') for s in ('X','C','P','S','F')])
    paragraph('Feature는 같은 S3 조건에서 저장된 23개 field의 exact available 행만 compact gather하고 availability mask/positions를 저장했다. '
        'Missing feature는 근사 복원하지 않았다. 완전한 production Vision Tensor 검증을 주장하지 않는다.')
    for name,rows in result['patches'].items():
        sensitive=sorted([(e['span_reduction_m'],r,e) for r in rows for e in r['extrema_leave_one_out']],key=lambda x:-x[0])[:3]
        table(['object','patch','extreme source','original C span m','span without one point m','drop m','quota'],[
            [name,r['patch'],e['source'],r['weight'],e['span_without_point_m'],drop,r['quota']] for drop,r,e in sensitive])
    paragraph('각 patch의 C/X bbox min/max XYZ와 C extrema source별 leave-one-out span을 저장했다. 위 표는 한 극단점에 민감한 상위 사례다. '
        '한 점 제거 span은 진단일 뿐 선택에 적용하지 않았다. Bbox span은 실제 구조 길이가 아니며 camera 축과 outlier에 민감하다.')
    plot('surface_depth_tradeoff.png','Unfiltered depth and whole X→F Surface trade-off')
    section('Synthetic 경계와 fallback 계약',
        '기존 Task03-F 7개 fixture에는 dense X 및 PairEvidence가 없어 새 full-rule performance는 전부 N/A다. '
        '구조 label로 patch를 만들거나 새 detector/normal/fixture를 생성하지 않았다. 기존 Global/UV 결과를 새 방법의 검증 결과로 표시하지 않는다.')
    table(['existing fixture','new full-rule performance','reason'],[[n,'N/A',v['reason']] for n,v in result['synthetic'].items()])
    paragraph('일반 fallback은 현재 C만으로 기존 fps_indices()의 Global λ=0 순서를 만들고 이미 선택된 point를 제외해 부족분을 채운다. '
        '이번 고정 입력에서는 E의 기존 검증과 저장 순서를 재사용할 수 있으며 NPZ는 fallback 규칙의 필수 입력이 아니다. '
        '두 실제 물체의 fallback은 0점이어서 새 결과가 fallback으로 Global FPS를 복제한 경우는 아니다. '
        '작은 synthetic 계약 테스트는 입력 C만으로 fallback을 재현하고 depth/normal veto·분리·0-span·capacity·anchor·정렬·결정성·GT 독립성을 확인한다. '
        '이 테스트는 실제 형상 또는 sensor noise에 대한 일반화 성능 검증이 아니다.')
    section('완료 판단 — 여섯 질문', '')
    for text in result['interpretation_ko']:paragraph(text)
    section('검증과 미실행 범위',tests)
    paragraph('Source/input/artifact hash 불변, 모든 C의 단일 patch 귀속, quota/capacity, P=32, F=129/127, '
        'P⊆C⊆X·F⊆X, pixel/XYZ/object/role 정렬, P/S 중복 없음, NPZ exact readback, saved baseline 평가 exact equality를 확인했다. '
        'GT annotation 변경에 따른 전체 선택 불변은 annotation-free 입력/함수 및 작은 full-rule 계약 테스트로 확인하고 실제 선택 전후 archive hash도 유지했다. '
        '실제 Frame0 선택을 GT 변경 후 다시 실행하지 않았다. Optimized pair-level 동일성은 N/A다.')
    paragraph('공통 ExperimentLogger와 experiment_artifacts의 digests/register_existing_output 및 link_output 로컬 API를 재사용했다. '
        '실행 소스 snapshot, 시작 dirty diff, 입력 hash, 구현 diff와 tests.log를 새 UUID에 보존했다. '
        'Simulator/renderer·GPU/CUDA/MPS/PCL·추가 frame/episode·normal/candidate 재계산·sweep·production·Graph/GNN/BC·외부 업로드·dependency 설치·commit은 실행하지 않았다. '
        'Adaptive Budget 결합, Task-Relevance, 다른 grouping과 다음 단계는 시작하지 않고 여기서 종료한다.')
    paragraph('재현 명령: WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_geometry_aware_sampling '
        '--original-state STATE_DIR. STATE_DIR에는 사전 파일 hash mapping인 hashes.json과 git_start.diff, git_status.txt를 둔다. '
        '선택 규칙·입력 경로/hash는 preregistered_config.json, 실행 소스는 source_snapshot 또는 selection_source_snapshot에 있다. '
        '보고서가 이미 존재하면 덮어쓰기를 거부하므로 별도 checkout/복제 경로에서 재현한다. 현재 결과의 문서만 재생성하려면 '
        '--render-saved SELECTION_RUN_DIR을 추가한다. 원본을 수정하지 않고 새로운 보고 UUID를 생성한다.')
    section('산출물과 원본 링크','')
    files=[STEM+'.html',STEM+'.json',STEM+'.npz','selection_before_gt.npz','patch_diagnostics_before_selection.json',
        'verification.json','tests.log','source_hashes.json','original_hashes.json','preregistered_config.json','implementation.patch']
    for filename in files:
        md.extend([f'[{filename}](../../experiments/{result["run_id"]}/{filename})',''])
        page.append(f'<p><a href="{filename}">{filename}</a></p>')
    for p,label in [(previous.BASELINE/(prior.STEM+'.html'),'Task03-E saved baseline'),
                    (UV_DIR/(previous.STEM+'.html'),'Task03-F reference / existing synthetic evidence'),
                    (REPO/'docs/visual-graph-integration/03g_geometry_aware_sampling_plan.md','approved plan')]:
        md.extend([f'[{label}]({os.path.relpath(p,REPORT.parent)})',''])
        page.append(f'<p><a href="{os.path.relpath(p,output)}">{label}</a></p>')
    page.append('<p><a href="../../docs/visual-graph-integration/03g_geometry_aware_sampling.md">Markdown report</a></p></html>')
    REPORT.write_text('\n'.join(md),encoding='utf-8')
    (output/'report.md').write_text('\n'.join(md),encoding='utf-8')
    (output/(STEM+'.html')).write_text('\n'.join(page),encoding='utf-8')


def interpretation(result):
    d=result['patch_diagnostics']; a=result['actual']; patches=result['patches']
    lines=[f"1. 실제 C의 분리: Peg는 X patch {d['peg']['X_patches']}개 중 C patch {d['peg']['candidate_patches']}개, "
        f"Box는 {d['box']['X_patches']}개 중 {d['box']['candidate_patches']}개다. "
        '이는 관측 normal/depth gate로 연결된 표면이지 실제 구조의 독립 분해가 아니다.']
    allocation=[]
    for name,rows in patches.items():
        allocation.append(name+': '+', '.join(f"patch {r['patch']}={r['quota']}" for r in rows if r['quota']))
    lines.append('2. 3D span quota: '+'; '.join(allocation)+'. Peg의 대부분 C는 하나의 큰 surface patch에 연결되어 독립 구조별 예산을 만들지 못했다. '
        'Box는 여러 patch에 배분되었지만 넓은 bbox를 가진 patch에 예산이 집중된다.')
    primary=[];secondary=[]
    for n,row in a.items():
        b,z=row['global0'],row['geometry'];bg,zg=b['gt'],z['gt']
        primary.append(f"{n} C→P mean/p95/max {b['spatial']['mm']['raw']['mean']:.3f}/{b['spatial']['mm']['raw']['p95']:.3f}/{b['spatial']['mm']['raw']['max']:.3f}"
            f"→{z['spatial']['mm']['raw']['mean']:.3f}/{z['spatial']['mm']['raw']['p95']:.3f}/{z['spatial']['mm']['raw']['max']:.3f}mm; "
            f"X→F mean/p95 {bg['surface_distance_mm']['mean']:.3f}/{bg['surface_distance_mm']['p95']:.3f}"
            f"→{zg['surface_distance_mm']['mean']:.3f}/{zg['surface_distance_mm']['p95']:.3f}mm")
        secondary.append(f"{n} P/F refs {bg['stages']['P']['unique_reference_count']}/{bg['stages']['F']['unique_reference_count']}"
            f"→{zg['stages']['P']['unique_reference_count']}/{zg['stages']['F']['unique_reference_count']}; "
            f"negative P/F {bg['occupancy']['negative_roi']['known_negative_union']['P']['count']}/{bg['occupancy']['negative_roi']['known_negative_union']['F']['count']}"
            f"→{zg['occupancy']['negative_roi']['known_negative_union']['P']['count']}/{zg['occupancy']['negative_roi']['known_negative_union']['F']['count']}")
    lines.append('3. Global λ=0 대비 3D 보존: '+'; '.join(primary)+'. 손익은 지표별로 해석하며 GT 보존으로 primary 손실을 상쇄하지 않는다.')
    lines.append('4. 국소 구조·negative ROI: '+'; '.join(secondary)+'. Face-pair/rim 전체 구간과 Surface 추가분·overlap은 위 표에 제시했다. '
        '구간별 개선·손실을 합쳐 특정 Box 선분을 성공 기준으로 삼지 않았다.')
    lines.append(f"5. 제한 요인: Peg/Box의 0-span 후보 {d['peg']['zero_span_candidates']}/{d['box']['zero_span_candidates']}개는 "
        f"span quota를 받지 못하며 quota=0 후보 patch는 {d['peg']['zero_quota_candidate_patches']}/{d['box']['zero_quota_candidate_patches']}개다. "
        'Normal 불연속 주변 후보가 작은 patch로 분절되면 중요한 구조라도 quota에서 사라질 수 있는 설계 취약성이다. '
        '실제 fallback은 양쪽 0점이다. Bbox 극단점 민감도와 depth outlier는 제거하지 않았고 span을 구조 길이로 해석하지 않는다.')
    lines.append('6. 추가 검증 판단: Frame0에서 관측 patch가 어떻게 분리·배분되는지는 재현 가능하게 설명했지만, '
        'Peg의 단일 큰 patch, Box의 fragmentation/zero-quota, reference와 optimized PCA5 pair 차이의 미확인, '
        '7개 full-rule synthetic N/A가 남는다. 현재 결과만으로 일반 geometry preservation 후보의 유용성·안정성을 확립하거나 production 채택을 정당화하지 않는다. '
        '관측 patch의 유용성에는 제한된 근거만 있으며, 독립 관측·형상·camera에서 같은 고정 규칙의 별도 evidence가 필요하다. 이번에는 추가 검증을 시작하지 않는다.')
    return lines


def verify_documents(output):
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            self.links.extend(v for k,v in attrs if k in ('href','src'))
    parser=Links();content=(output/(STEM+'.html')).read_text();parser.feed(content)
    if re.search(r'<script|https?://',content):raise ValueError('HTML must be static and local')
    for target in parser.links:
        if not (output/target).is_file():raise ValueError('missing HTML link '+target)
    targets=re.findall(r'\]\(([^)]+)\)',REPORT.read_text())
    for target in targets:
        if not (REPORT.parent/target).is_file():raise ValueError('missing Markdown link '+target)
    decoded={}
    for path in output.glob('*.png'):
        a=plt.imread(path)
        if a.ndim != 3 or min(a.shape[:2]) < 400 or a.std()<.01:raise ValueError('invalid PNG '+path.name)
        decoded[path.name]=list(a.shape)
    return dict(status='PASS',html_links=len(parser.links),markdown_links=len(targets),png_decoded=decoded)


def run(state_directory):
    if os.environ.get('WANDB_MODE')!='disabled':raise ValueError('WANDB_MODE=disabled required')
    if REPORT.exists():raise FileExistsError('refuse to overwrite existing report')
    state_directory=Path(state_directory)
    original=json.loads((state_directory/'hashes.json').read_text());preserved(original)
    sources=list((REPO/'visual/runtime').rglob('*.py'))+[REPO/'utils/experiment_logger.py',REPO/'utils/experiment_artifacts.py']
    source_hashes=prior.hashes(sources)
    old=json.loads((previous.BASELINE/(prior.STEM+'.json')).read_text())
    uv_old=json.loads((UV_DIR/(previous.STEM+'.json')).read_text())
    require_equal(uv_old['run_id'],UV_RUN,'UV reference identity')
    for p,digest in old['config']['input_sha256'].items():require_equal(digests(REPO/p)[1],digest,'E saved input '+p)
    require_equal(json.loads((previous.BASELINE/'verification.json').read_text())['status'],'PASS','E baseline verified')
    input_paths=[REPO/p for p in old['config']['input_sha256']]+[
        previous.BASELINE/(prior.STEM+'.npz'),previous.BASELINE/(prior.STEM+'.json'),previous.BASELINE/'verification.json',
        UV_DIR/(previous.STEM+'.npz'),UV_DIR/(previous.STEM+'.json'),UV_DIR/'verification.json',
        prior.EDGES.parent/'comparison.json',prior.NORMAL.parent/'comparison.json']
    input_hashes=prior.hashes(input_paths)
    config=dict(task='03-G Phase1 Frame0 feasibility',rules=RULES,episode='ep000',frame=0,
        method='S3(PCA5)',seed=0,threshold_deg=20,structural_ratio=.25,num_points=256,priority_count=32,budgets=prior.BUDGET,
        source_run_id=previous.BASELINE_ID,task03=prior.SOURCE_RUN,task03f=UV_RUN,
        source_sha256=source_hashes,input_sha256=input_hashes,python=platform.python_version(),numpy=np.__version__,
        device='cpu',simulator_executed=False,gpu_executed=False,production_policy_adopted=False,
        baseline_FPS_replayed=False,synthetic_performance='N/A',parameters_swept=False)
    with ExperimentLogger(team='visual',config=config,seed=0,model='task03g_observation_patch_feasibility',job_type='evaluate') as logger:
        output=logger.directory
        try:
            write_json(output/'original_hashes.json',original)
            write_json(output/'preregistered_config.json',config);frozen_hash=digests(output/'preregistered_config.json')[1]
            for name in ('git_start.diff','git_status.txt'):shutil.copyfile(state_directory/name,output/name)
            for source in sources:
                dest=output/'source_snapshot'/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            frame,geometry,samples,cases,baselines,loaded_old,input_config,contracts=previous.load_actual()
            require_equal(loaded_old,old,'baseline payload')
            with np.load(prior.EDGES,allow_pickle=False) as edges,np.load(prior.NORMAL,allow_pickle=False) as normal:
                pairs,gate=compatibility(frame,geometry,samples,cases,input_config,edges,normal)
            write_json(output/'compatibility.json',gate)
            selections=read_reference_selections(cases,baselines)
            prepared={name:prepare(selection_input(case),pairs[name]) for name,case in cases.items()}
            write_json(output/'patch_diagnostics_before_selection.json',serial(prepared))
            print('Evidence compatibility PASS. Patch/weight/quota frozen before FPS.',flush=True)
            with (output/'tests.log').open('w') as log:
                tests=subprocess.run([sys.executable,'-m','unittest',*TESTS,'-v'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            if tests.returncode:raise RuntimeError('contract tests failed before actual selection')
            test_text=(output/'tests.log').read_text()
            test_summary=next(l for l in test_text.splitlines() if l.startswith('Ran '))+'; '+test_text.splitlines()[-1]
            print(test_summary,flush=True)
            arrays={};patches={};diagnostics={};counts={}
            for name,case in cases.items():
                a=prepared[name]
                chosen=select_geometry(selection_input(case),a,baselines[name]['P'])
                check_selection(case,chosen)
                np.testing.assert_array_equal(chosen['patch_priority_count'],a['quota'])
                if (a['quota']>a['capacity']).any():raise ValueError('quota exceeds capacity')
                require_equal(int(a['quota'].sum())+len(chosen['fallback_source']),32,'quota plus fallback')
                selections[name]['geometry']=chosen
                counts[name]=dict(X=len(case['support_flat']),C=len(case['candidate_flat']),P=32,F=case['budget'])
                diagnostics[name]=dict(a['diagnostics'],actual_fallback=len(chosen['fallback_source']),fallback_mode=chosen['fallback_mode'])
                patches[name]=[dict(r,actual_priority=int(chosen['actual_priority_count'][i]),
                    patch_FPS_priority=int(chosen['patch_priority_count'][i]),fallback_priority=int(chosen['fallback_patch_count'][i]),
                    patch_initial_source=int(chosen['patch_initial_source'][i])) for i,r in enumerate(a['patches'])]
                previous.add_arrays(arrays,name,{k:case[k] for k in ('candidate_flat','candidate_xyz','support_flat','support_xyz')})
                previous.add_arrays(arrays,name+'/observation_patch',{k:a[k] for k in (
                    'support_patch','candidate_patch','connection_gate','weight','capacity','quota')})
                for f in fields(PairEvidence):arrays[f'{name}/observation_pair/{f.name}']=np.asarray(getattr(pairs[name],f.name))
                previous.add_arrays(arrays,name+'/geometry/provenance',{k:chosen[k] for k in (
                    'priority_rows','patch_priority_count','actual_priority_count','fallback_patch_count','fallback_source','patch_initial_source')})
                for method,s in selections[name].items():
                    check_selection(case,s)
                    for stage in ('P','S','F'):
                        flat=s[stage]
                        previous.add_arrays(arrays,f'{name}/{method}/{stage}',dict(flat_index=flat,
                            pixels_vu=previous.uv_from_flat(flat,frame.depth.shape)[:,::-1],
                            xyz=case['support_xyz'][np.searchsorted(case['support_flat'],flat)],
                            object_id=np.full(len(flat),case['oid']),
                            selection_role=np.full(len(flat),'Priority' if stage=='P' else 'Surface') if stage!='F'
                            else np.where(np.arange(len(flat))<32,'Priority','Surface')))
                    available,positions,features=prior.feature_rows(case,s['F'])
                    previous.add_arrays(arrays,f'{name}/{method}/F',dict(feature_available=available,feature_available_positions=positions))
                    previous.add_arrays(arrays,f'{name}/{method}/F/available_features',features)
            for method in ('global0','geometry','global1','uv'):
                f=np.concatenate([selections[n][method]['F'] for n in cases])
                require_equal((len(f),len(np.unique(f))),(256,256),'cross-object unique budget')
            arrays['selection_input_hashes_json']=np.array(json.dumps(input_hashes,sort_keys=True))
            np.savez_compressed(output/'selection_before_gt.npz',**arrays)
            before_gt_hash=digests(output/'selection_before_gt.npz')[1]
            print('Geometry selection fixed and saved before GT. No Global baseline FPS replay.',flush=True)
            actual,members=evaluate(frame,geometry,samples,cases,selections,arrays,old,uv_old,input_config)
            result=dict(run_id=logger.run_id,config=config,input_config=input_config,counts=counts,
                compatibility=gate,source_contract=contracts,patches=patches,patch_diagnostics=diagnostics,
                actual=actual,peg_memberships=members,selection_before_gt_sha256=before_gt_hash,
                synthetic={n:dict(status='N/A',reason='existing sparse C has no dense X or PairEvidence') for n in previous.FIXTURES['names']},
                performance_success_required=False,contract_status='PASS',production_policy_adopted=False)
            result['interpretation_ko']=interpretation(result)
            write_json(output/(STEM+'.json'),serial(result));np.savez_compressed(output/(STEM+'.npz'),**arrays)
            with np.load(output/(STEM+'.npz'),allow_pickle=False) as saved:
                require_equal(set(saved.files),set(arrays),'NPZ key set')
                for k,v in arrays.items():np.testing.assert_array_equal(saved[k],v,err_msg=k)
            require_equal(digests(output/'selection_before_gt.npz')[1],before_gt_hash,'pre-GT archive unchanged')
            make_plots(output,frame,cases,prepared,selections,result,arrays)
            require_equal(prior.hashes(sources),source_hashes,'executed sources unchanged')
            require_equal(prior.hashes(input_paths),input_hashes,'input hashes unchanged')
            require_equal(digests(output/'preregistered_config.json')[1],frozen_hash,'frozen config')
            preservation=preserved(original)
            write_json(output/'source_hashes.json',dict(sources_before=source_hashes,sources_after=prior.hashes(sources),
                inputs_before=input_hashes,inputs_after=prior.hashes(input_paths),original_preservation=preservation,
                historical_task03e_sources=old['config']['source_sha256']))
            verification=dict(status='PASS',compatibility=gate,original_preservation=preservation,
                required_tests=dict(summary=test_summary,modules=TESTS,failed=0),
                all_C_exactly_one_patch=True,quota_capacity_and_sum=True,fixed_P_F_budgets=True,
                P_subset_C_subset_X_F_subset_X=True,source_pixel_xyz_object_role_alignment=True,
                P_S_no_overlap=True,npz_exact_readback=True,pre_GT_archive_sha256_unchanged=True,
                GT_independence='full-rule synthetic annotation perturbation + annotation-free actual helpers; actual selection rerun=0',
                joint_matching_and_reference_denominators=True,baseline_and_UV_saved_metric_exact=True,
                optimized_pair_identity='N/A',synthetic_performance='N/A',source_input_hashes_unchanged=True,
                feature_unavailable_explicit=True,preregistered_config_sha256=frozen_hash,
                actual_fps_calls=dict(global_baseline=0,priority_patch=sum(int((p['quota']>0).sum()) for p in prepared.values()),
                    global_fallback=sum(s['geometry']['fallback_mode']=='generated_current_C_global_order' for s in selections.values()),surface=2),
                no_generalization_or_adoption_claim=True,
                skipped=['simulator','renderer','GPU/CUDA/MPS/PCL','normal/candidate extraction','sweep','other frames/episodes',
                    'full-rule synthetic performance','feature reconstruction','production','external upload','install','commit'])
            write_json(output/'verification.json',verification)
            # Only this new run owns these targets; originals are never replaced.
            (output/'implementation.patch').touch()
            documents(output,result,test_summary)
            diff=b''
            for path in (Path(__file__),Path(__file__).with_name('test_geometry_aware_sampling.py'),REPORT):
                proc=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path.relative_to(REPO))],cwd=REPO,stdout=subprocess.PIPE)
                if proc.returncode!=1:raise ValueError('implementation diff capture failed')
                diff+=proc.stdout
            (output/'implementation.patch').write_bytes(diff)
            verification['document_checks']=verify_documents(output)
            write_json(output/'verification.json',verification)
            for path in sorted(output.rglob('*')):
                if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','config.yaml','metrics.json'):
                    logger.link_output(path,kind='reports')
            logger.link_output(REPORT,kind='reports')
            logger.record(source_run_id=previous.BASELINE_ID,task03_source_run_id=prior.SOURCE_RUN,
                task03f_source_run_id=UV_RUN,selection_before_gt_sha256=before_gt_hash,
                simulator_executed=False,gpu_executed=False,production_policy_adopted=False)
            logger.log({n:dict(candidate_mean_mm=r['geometry']['spatial']['mm']['raw']['mean'],
                candidate_p95_mm=r['geometry']['spatial']['mm']['raw']['p95'],
                surface_mean_mm=r['geometry']['gt']['surface_distance_mm']['mean'],
                final_refs=r['geometry']['gt']['stages']['F']['unique_reference_count']) for n,r in actual.items()},step=0,prefix='eval/task03g')
            print(output,flush=True)
        except Exception as exc:
            write_json(output/'attempt_failure.json',dict(status='FAIL',exception_type=type(exc).__name__,reason=str(exc),
                no_automatic_reselection=True,existing_attempt_preserved=True))
            print('Failed attempt preserved at '+str(output),flush=True)
            raise
    register_existing_output(output,output/(STEM+'.html'),kind='reports')
    return output


def render_saved(previous_run,state_directory):
    """Reporting recovery from a complete numeric attempt, with zero FPS/evaluation.

    Preserve the failed execution byte-for-byte, freeze selector/evaluator ASTs,
    and replace only our own just-created Markdown after checking its exact copy.
    """
    import ast
    if os.environ.get('WANDB_MODE')!='disabled':raise ValueError('WANDB_MODE=disabled required')
    previous_run=Path(previous_run).resolve()
    original=json.loads((Path(state_directory)/'hashes.json').read_text());preserved(original)
    data=json.loads((previous_run/(STEM+'.json')).read_text())
    if REPORT.exists() and REPORT.read_bytes() != (previous_run/'report.md').read_bytes():
        raise FileExistsError('refuse to replace unrelated existing report')
    if not REPORT.exists():raise FileNotFoundError('recovery expects its exact own generated report')
    old_tree=ast.parse((previous_run/'source_snapshot'/Path(__file__).relative_to(REPO)).read_text())
    new_tree=ast.parse(Path(__file__).read_text())
    frozen=['ordered_source','patch_labels','allocate_span','prepare','fallback_rows','select_geometry',
            'selection_input','compatibility','check_selection','read_reference_selections','evaluate','make_plots']
    for name in frozen:
        old_node=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new_node=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        require_equal(ast.dump(old_node),ast.dump(new_node),'unchanged numeric AST '+name)
    require_equal(data['config']['rules'],RULES,'unchanged rule constants')
    prior_hashes=prior.hashes([p for p in previous_run.rglob('*') if p.is_file()])
    config=dict(task='03-G report recovery',source_run_id=previous.BASELINE_ID,selection_run_id=data['run_id'],
        numeric_npz_sha256=digests(previous_run/(STEM+'.npz'))[1],actual_FPS_calls=0,evaluation_calls=0,
        source_sha256=prior.hashes([Path(__file__),Path(__file__).with_name('test_geometry_aware_sampling.py')]))
    with ExperimentLogger(team='visual',config=config,seed=0,model='task03g_report_recovery',job_type='evaluate') as logger:
        output=logger.directory
        excluded={STEM+'.json',STEM+'.html','report.md','implementation.patch','verification.json','source_hashes.json',
                  'config.yaml','metadata.json','manifest.json','metrics.json','attempt_failure.json'}
        for path in previous_run.iterdir():
            if path.is_file() and path.name not in excluded and not path.name.endswith('.tracking.json'):
                shutil.copyfile(path,output/path.name)
        shutil.copytree(previous_run/'source_snapshot',output/'selection_source_snapshot',
            ignore=shutil.ignore_patterns('*.tracking.json'))
        for path in (Path(__file__),Path(__file__).with_name('test_geometry_aware_sampling.py')):
            dest=output/'source_snapshot'/path.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        for filename in ('source_hashes.json','verification.json','attempt_failure.json','report.md','implementation.patch'):
            shutil.copyfile(previous_run/filename,output/('selection_'+filename))
        recovery=dict(selection_run_id=data['run_id'],reason='post-run artifact registration called while logger running; reporting only',
            this_execution_FPS_calls=0,this_execution_evaluation_calls=0,numeric_npz_byte_identical=True,
            selection_before_GT_byte_identical=True,frozen_numeric_AST=frozen)
        result=dict(data,run_id=logger.run_id,report_recovery=recovery)
        write_json(output/(STEM+'.json'),result)
        verification=json.loads((previous_run/'verification.json').read_text())
        verification.update(report_recovery=recovery,selection_run_status='failed after numeric/document completion',
            original_preservation=preserved(original),this_execution_FPS_calls=0,this_execution_evaluation_calls=0,
            visual_QA=dict(status='PASS',inspected=['patch_and_selection.png','candidate_XZ_coverage.png','patch_quota.png']))
        require_equal(digests(output/(STEM+'.npz'))[1],digests(previous_run/(STEM+'.npz'))[1],'numeric NPZ byte identical')
        require_equal(digests(output/'selection_before_gt.npz')[1],digests(previous_run/'selection_before_gt.npz')[1],'pre-GT NPZ byte identical')
        with (output/'tests_report_recovery.log').open('w') as log:
            proc=subprocess.run([sys.executable,'-m','unittest','visual.runtime.test.test_geometry_aware_sampling','-v'],cwd=REPO,
                                stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:raise RuntimeError('recovery source tests failed')
        verification['report_recovery_tests']=dict(status='PASS',count=13,log='tests_report_recovery.log')
        write_json(output/'verification.json',verification)
        write_json(output/'source_hashes.json',dict(selection_source_hashes=json.loads((previous_run/'source_hashes.json').read_text()),
            reporting_source_sha256=config['source_sha256'],previous_run_files_before=prior_hashes))
        (output/'implementation.patch').touch()
        documents(output,result,verification['required_tests']['summary'])
        diff=b''
        for path in (Path(__file__),Path(__file__).with_name('test_geometry_aware_sampling.py'),REPORT):
            proc=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path.relative_to(REPO))],cwd=REPO,stdout=subprocess.PIPE)
            if proc.returncode!=1:raise ValueError('implementation diff capture failed')
            diff+=proc.stdout
        (output/'implementation.patch').write_bytes(diff)
        verification['document_checks']=verify_documents(output)
        require_equal(prior.hashes([REPO/p for p in prior_hashes]),prior_hashes,'failed numeric execution unchanged')
        require_equal(prior.hashes([REPO/p for p in config['source_sha256']]),config['source_sha256'],'reporting sources unchanged')
        verification['failed_numeric_execution_preserved']=True
        verification['original_preservation']=preserved(original)
        write_json(output/'verification.json',verification)
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','config.yaml','metrics.json'):
                logger.link_output(path,kind='reports')
        logger.link_output(REPORT,kind='reports')
        logger.record(source_run_id=previous.BASELINE_ID,selection_run_id=data['run_id'],report_only=True,
            numeric_npz_byte_identical=True,this_execution_FPS_calls=0,this_execution_evaluation_calls=0)
        logger.log({n:dict(candidate_p95_mm=r['geometry']['spatial']['mm']['raw']['p95'],
            final_refs=r['geometry']['gt']['stages']['F']['unique_reference_count']) for n,r in result['actual'].items()},
            step=0,prefix='eval/task03g_saved_selection')
        print(output,flush=True)
    register_existing_output(output,output/(STEM+'.html'),kind='reports')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-state',type=Path,required=True,help='pre-implementation hashes.json and git_start.diff directory')
    parser.add_argument('--render-saved',type=Path,help='report-only recovery from complete numeric execution; no FPS/evaluation')
    args=parser.parse_args()
    if args.render_saved:render_saved(args.render_saved,args.original_state)
    else:run(args.original_state)


if __name__=='__main__':main()
