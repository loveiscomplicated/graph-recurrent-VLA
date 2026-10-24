"""Task03-I: one approved nominal voxel precompression + 3D FPS cap control.

Offline saved ep000 Frame0 only. Production, detectors and saved FPS are read-only.
WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_voxel_structural_sampling --baseline PATH
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
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from utils.experiment_logger import ExperimentLogger
from visual.runtime.sampling import fps_indices
from visual.runtime.test import run_local_structural_sampling as local
from visual.runtime.test import run_priority_fps_metric_control as prior
from visual.runtime.test.run_rim_bottleneck import occupancy, require_equal
from visual.runtime.test.sampling_evaluation import staged_reference_coverage

REPO = prior.REPO
STEM = 'voxel_structural_sampling'
REPORT = REPO/'docs/visual-graph-integration/03i_voxel_structural_sampling.md'
PLAN = REPO/'docs/visual-graph-integration/03i_voxel_structural_sampling_plan.md'
FIXTURE_RUN = REPO/'experiments/1f7ad262dd2e4b2fb0ac805a4eda5278'
FIGURES = ('primary_selection.png', 'resolution_budget.png', 'primary_distances.png',
           'peg_gt.png', 'box_gt.png', 'depth_roi_surface.png', 'synthetic.png')
TESTS = ['visual.runtime.test.'+n for n in ('test_voxel_structural_sampling', 'test_sampling',
    'test_sampling_geometry', 'test_sampling_evaluation', 'test_priority_fps_metric_control',
    'test_local_structural_sampling')]
RULES = dict(frame=0, episode='ep000', camera='base_camera', method='S3 optimized PCA5',
    orientation='strict >20 degrees, stored C only', seed=0, priority_cap_per_object=32,
    final_targets=dict(peg=129, box=127), total_cap=256, structural_ratio=.25,
    scale='median Z of original X; epsilon=z*sqrt((2/fx)^2+(2/fy)^2); a=epsilon/sqrt(3)',
    multiplier=1., grid='CV camera metres; origin 0; float64 floor(xyz/a); int64; half-open',
    representative='float64 centroid nearest original C; exact squared-distance tie smallest source',
    order='C/R source ascending; mapped anchor first in P; rest FPS or source ascending',
    anchor='original whole-C seeded anchor cell representative', cap='existing Euclidean 3D fps_indices',
    surface='existing whole-X 3D FPS initial=P; unused Priority becomes Surface; no padding',
    dyadic='count-only; floor_divide nominal keys by 2; first count<=32; never used for selection',
    gt='only after all real/synthetic selections saved, read back and hashed; same-pair <=2px AND <=3mm',
    score='none; no optimized dense C score available', uv_bins=[4,4], xyz_bins=[2,2,2],
    synthetic_names=local.FIXTURES['names'], synthetic_scale='stored C median Z; stored focal=1600px',
    synthetic_uncertainty_calibration='N/A: no actual PCA5 estimator/localization evidence',
    interpretation='first feasibility of voxel precompression + 3D FPS cap; anchor also changes')


def serial(value):
    if isinstance(value, dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [serial(v) for v in value]
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sources(flat, xyz):
    flat, xyz = np.asarray(flat), np.asarray(xyz)
    if flat.ndim != 1 or flat.dtype.kind not in 'iu' or np.any(flat < 0):
        raise ValueError('nonnegative integer sources required')
    if len(flat)>1 and not np.all(flat[1:]>flat[:-1]): raise ValueError('strict source order required')
    if xyz.shape != (len(flat),3) or not np.isfinite(xyz).all(): raise ValueError('finite source XYZ required')
    return flat, xyz


def nominal_scale(xyz, intrinsic):
    xyz, k = np.asarray(xyz), np.asarray(intrinsic, float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all() or (xyz[:,2]<=0).any():
        raise ValueError('original positive-depth X required')
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('valid intrinsic required')
    if not len(xyz): return dict(status='N/A', z_ref_m=None, epsilon_m=None, a_m=None)
    z = float(np.median(xyz[:,2].astype(float)))
    epsilon = z*np.hypot(2/k[0,0], 2/k[1,1])
    a = epsilon/np.sqrt(3)
    if not np.isfinite(a) or a<=0: raise ValueError('nonpositive nominal scale')
    return dict(status='defined', z_ref_m=z, fx_px=float(k[0,0]), fy_px=float(k[1,1]),
        epsilon_m=float(epsilon), a_m=float(a), multiplier=1.,
        pair_footprint_m=float(z*max(np.hypot(3/k[0,0],2/k[1,1]),np.hypot(2/k[0,0],3/k[1,1]))))


def voxel_keys(xyz, a):
    xyz=np.asarray(xyz, float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all() or not np.isfinite(a) or a<=0:
        raise ValueError('finite XYZ and positive size required')
    with np.errstate(over='ignore',invalid='ignore'):
        q=np.floor(xyz/a)
    # float64(int64.max) rounds to 2**63, so the upper comparison is strict.
    if not np.isfinite(q).all() or (q < -(2.**63)).any() or (q >= 2.**63).any():
        raise ValueError('voxel key int64 overflow')
    return q.astype(np.int64)


def representatives(flat, xyz, a):
    flat,xyz=sources(flat,xyz)
    key=voxel_keys(xyz,a)
    cells,membership,capacity=np.unique(key,axis=0,return_inverse=True,return_counts=True)
    rep_rows=np.empty(len(cells),np.int64)
    for cell in range(len(cells)):
        rows=np.flatnonzero(membership==cell)  # already source ascending
        points=xyz[rows].astype(float)
        centroid=points.mean(axis=0)
        distance=np.sum((points-centroid)**2,axis=1)
        rep_rows[cell]=rows[int(distance.argmin())]
    same_rows=rep_rows[membership]
    same_distance=np.linalg.norm(xyz.astype(float)-xyz[same_rows].astype(float),axis=1)
    if len(flat) and np.max(same_distance)>np.sqrt(3)*a+64*np.finfo(float).eps*max(1.,np.max(np.abs(xyz))):
        raise AssertionError('same-cell bound failed')
    return dict(keys=key,cells=cells,membership=membership,capacity=capacity,
        cell_rep_rows=rep_rows, representative_rows=np.sort(rep_rows),
        same_representative_source=flat[same_rows], same_cell_distance_m=same_distance)


def dyadic_counts(keys, a, cap=32):
    keys=np.asarray(keys)
    if keys.ndim!=2 or keys.shape[1]!=3 or keys.dtype!=np.int64 or not np.isfinite(a) or a<=0:
        raise ValueError('int64 keys and positive size required')
    if not isinstance(cap,int) or cap<0: raise ValueError('invalid cap')
    if cap==0 and len(keys): return dict(status='impossible',levels=[],a_budget_m=None,ratio=None)
    if cap<8 and len(keys): raise ValueError('termination only guaranteed for cap>=8')
    levels=[]; key=keys.copy()
    for level in range(65):
        count=len(np.unique(key,axis=0))
        if levels and count>levels[-1]['occupied']: raise AssertionError('non-nested parent count')
        size=float(np.ldexp(a,level))
        if not np.isfinite(size): raise ValueError('dyadic size overflow')
        levels.append(dict(level=level,occupied=count,a_m=size,ratio=2**level,bound_m=np.sqrt(3)*size))
        if count<=cap: return dict(status='defined',levels=levels,a_budget_m=size,ratio=2**level,
            occupied=count,used_for_selection=False)
        key=np.floor_divide(key,2)
    raise AssertionError('dyadic iteration failed to terminate')


def choose_priority(flat, xyz, a, seed, anchor=None, cap=32, shape=None):
    if not isinstance(cap,int) or cap<0: raise ValueError('nonnegative integer cap required')
    t0=perf_counter(); g=representatives(flat,xyz,a); t1=perf_counter()
    flat,xyz=sources(flat,xyz); rrows=g['representative_rows']; rflat=flat[rrows]
    mapped=None; mapped_row=None
    if len(flat):
        if anchor is None: anchor=int(flat[np.random.default_rng(seed).integers(len(flat))])
        ar=np.flatnonzero(flat==anchor)
        if len(ar)!=1: raise ValueError('anchor outside C')
        mapped=int(g['same_representative_source'][ar[0]])
        mapped_row=int(np.searchsorted(rflat,mapped))
    if len(rrows)>cap:
        chosen=fps_indices(xyz[rrows],cap,seed=seed,initial=[mapped_row] if cap else ())
    elif len(rrows) and cap:
        chosen=np.r_[mapped_row,np.flatnonzero(rflat!=mapped)].astype(np.int64)
    else: chosen=np.empty(0,np.int64)
    prows=rrows[chosen]; t2=perf_counter()
    anchor_info=dict(original_source=anchor if len(flat) else None,mapped_source=mapped,distance_m=None,
                     delta_pixel_uv=None,pixel_distance_px=None)
    if len(flat):
        anchor_info['distance_m']=float(np.linalg.norm(xyz[int(ar[0])].astype(float)-xyz[rrows[mapped_row]].astype(float)))
        if shape is not None:
            uvs=np.column_stack(np.unravel_index([anchor,mapped],shape))[:,::-1]
            delta=uvs[1]-uvs[0]
            anchor_info.update(delta_pixel_uv=delta.tolist(),pixel_distance_px=float(np.linalg.norm(delta)))
    g.update(R=rflat.copy(),P=flat[prows].copy(),priority_rows=prows,anchor=anchor_info,
        timings_ms=dict(grid_representatives=(t1-t0)*1000,cap_fps=(t2-t1)*1000))
    return g


def select(candidate_flat,candidate_xyz,support_flat,support_xyz,shape,intrinsic,seed,anchor,final_target,cap=32):
    if not isinstance(cap,int) or cap<0: raise ValueError('invalid Priority cap')
    c,cxyz=sources(candidate_flat,candidate_xyz); x,xxyz=sources(support_flat,support_xyz)
    local.uv_from_flat(c,shape);local.uv_from_flat(x,shape)
    rows=np.searchsorted(x,c)
    if (rows>=len(x)).any() or not np.array_equal(x[rows],c): raise ValueError('C outside X')
    np.testing.assert_array_equal(xxyz[rows],cxyz)
    if not isinstance(final_target,int) or final_target<0: raise ValueError('invalid final target')
    t0=perf_counter();scale=nominal_scale(xxyz,intrinsic);t1=perf_counter()
    if len(x):
        g=choose_priority(c,cxyz,scale['a_m'],seed,anchor,min(cap,final_target),shape)
        dyadic=dyadic_counts(g['keys'],scale['a_m'],cap)
    else:
        g=dict(R=c.copy(),P=c.copy(),keys=np.empty((0,3),np.int64),cells=np.empty((0,3),np.int64),
            membership=np.empty(0,np.int64),capacity=np.empty(0,np.int64),cell_rep_rows=np.empty(0,np.int64),
            representative_rows=np.empty(0,np.int64),same_representative_source=c.copy(),same_cell_distance_m=np.empty(0),
            anchor=dict(original_source=None,mapped_source=None,distance_m=None,delta_pixel_uv=None,pixel_distance_px=None),
            timings_ms=dict(grid_representatives=0.,cap_fps=0.))
        dyadic=dict(status='N/A',levels=[],a_budget_m=None,ratio=None)
    t2=perf_counter(); initial=np.searchsorted(x,g['P'])
    frows=fps_indices(xxyz,final_target,seed=seed,initial=initial);f=x[frows];t3=perf_counter()
    return dict(P=g['P'],R=g['R'],S=f[len(initial):],F=f,xyz=xxyz[frows].copy(),grouping=g,scale=scale,
        dyadic=dyadic,shortfall=max(0,final_target-len(f)),
        timings_ms=dict(scale=(t1-t0)*1000,selection_and_dyadic=(t2-t1)*1000,surface_fps=(t3-t2)*1000,**g['timings_ms']))


def nearest_sources(query,flat,xyz):
    """Exact tie resolution by ascending source, independent of cKDTree ordering."""
    flat=np.asarray(flat); xyz=np.asarray(xyz,float);query=np.asarray(query,float)
    if not len(flat): return np.full(len(query),np.nan),np.full(len(query),-1,np.int64)
    order=np.argsort(flat,kind='stable');flat=flat[order];xyz=xyz[order]
    distances=np.empty(len(query));near=np.empty(len(query),np.int64)
    for i,q in enumerate(query):
        d=np.sum((xyz-q)**2,axis=1);j=int(d.argmin());distances[i]=np.sqrt(d[j]);near[i]=flat[j]
    return distances,near


def distance_decomposition(c,cxyz,chosen,shape):
    g=chosen['grouping']; r=chosen['R'];p=chosen['P']
    rxyz=cxyz[np.searchsorted(c,r)]; pxyz=cxyz[np.searchsorted(c,p)]
    dr,nr=nearest_sources(cxyz,r,rxyz);dp,np_=nearest_sources(cxyz,p,pxyz)
    budget=dp-dr
    if np.isfinite(budget).any() and np.nanmin(budget)<-1e-12: raise AssertionError('P not subset of R distance')
    arrays=dict(same_cell_m=g['same_cell_distance_m'],C_to_R_m=dr,C_to_P_m=dp,budget_extra_m=budget,
        C_to_R_source=nr,C_to_P_source=np_,same_cell_source=g['same_representative_source'])
    result={}
    for key in ('same_cell_m','C_to_R_m','C_to_P_m','budget_extra_m'):
        d=arrays[key];finite=np.isfinite(d);worst=int(np.nanargmax(d)) if finite.any() else None
        result[key]=dict(**local.summary(d*1000),unit='mm',missing_count=int((~finite).sum()),
            worst=None if worst is None else dict(source=int(c[worst]),xyz_m=cxyz[worst].tolist(),
                pixel_uv=local.uv_from_flat(c[[worst]],shape)[0].tolist(),distance_mm=float(d[worst]*1000)))
        arrays[key+'_ecdf']=np.sort(d[finite])
    eps=chosen['scale']['epsilon_m']
    selected_cells=np.unique(g['membership'][np.searchsorted(c,p)]) if len(p) else np.empty(0,np.int64)
    discarded=np.setdiff1d(np.arange(len(r)),selected_cells)
    above=dp>eps if eps is not None else np.zeros(len(c),bool)
    result.update(candidate_count=len(c),nominal_occupied=len(r),selected_occupied=len(selected_cells),
        discarded_occupied=len(discarded),occupied_over_32=len(r)/32,priority_count=len(p),
        resolution_budget_conflict=len(r)>32,above_epsilon_count=int(above.sum()),
        above_epsilon_fraction=float(above.mean()) if len(c) else None,
        a_used_m=chosen['scale']['a_m'],used_over_nominal=1. if eps is not None else None,
        discarded_cells=[dict(cell=int(i),key=g['cells'][i].tolist(),candidate_count=int(g['capacity'][i]),
            C_to_P_mm=local.summary(dp[g['membership']==i]*1000)) for i in discarded])
    arrays.update(selected_cell_ids=selected_cells,discarded_cell_ids=discarded,above_epsilon=above)
    return result,arrays


def add_arrays(target,prefix,values):
    for k,v in values.items(): target[prefix+'/'+k]=np.asarray(v)


def selection_arrays(cases,selections,synthetic):
    arrays={}
    for name,case in cases.items():
        chosen=selections[name]
        add_arrays(arrays,name,{k:case[k] for k in ('candidate_flat','candidate_xyz','support_flat','support_xyz')})
        add_arrays(arrays,name,{k:chosen[k] for k in ('R','P','S','F','xyz')})
        add_arrays(arrays,name+'/voxel',{k:chosen['grouping'][k] for k in ('keys','cells','membership','capacity',
            'cell_rep_rows','representative_rows','same_representative_source','same_cell_distance_m')})
    for name,data in synthetic.items():
        add_arrays(arrays,'synthetic/'+name,dict(candidate_flat=data['flat'],candidate_xyz=data['xyz'],
            R=data['chosen']['R'],P=data['chosen']['P'],keys=data['chosen']['grouping']['keys']))
    return arrays


def freeze_selection(output,cases,selections,synthetic):
    path=output/'selection_before_gt.npz'
    if path.exists(): raise FileExistsError('selection already frozen')
    arrays=selection_arrays(cases,selections,synthetic)
    np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False) as saved:
        require_equal(set(saved.files),set(arrays),'frozen keys')
        for key,value in arrays.items(): np.testing.assert_array_equal(saved[key],value,err_msg=key)
    return prior.sha(path),arrays


def check_selection(case,chosen):
    c,x=case['candidate_flat'],case['support_flat'];p,r,s,f=(chosen[k] for k in ('P','R','S','F'))
    if not set(p)<=set(r)<=set(c)<=set(x): raise AssertionError('P subset R subset C subset X')
    for key in ('R','P','S','F'):
        if len(np.unique(chosen[key]))!=len(chosen[key]): raise AssertionError('duplicate '+key)
    if len(p)>32 or len(f)>case['budget'] or len(f)!=min(len(x),case['budget']): raise AssertionError('budget')
    np.testing.assert_array_equal(f,np.r_[p,s]);np.testing.assert_array_equal(chosen['xyz'],case['support_xyz'][np.searchsorted(x,f)])


def load_baselines(cases,zero,samples):
    baselines={}
    with np.load(local.BASELINE/(prior.STEM+'.npz'),allow_pickle=False) as e:
        for name,case in cases.items():
            one={stage:case['saved_flat'][sl].copy() for stage,sl in
                (('P',slice(0,32)),('S',slice(32,None)),('F',slice(None)))}
            one['xyz']=case['support_xyz'][np.searchsorted(case['support_flat'],one['F'])].copy()
            for stage in ('P','S','F'):
                np.testing.assert_array_equal(one[stage],e[f'S3/{name}/1.0/{stage}/flat_index'])
            for method,chosen in dict(global1=one,global0=zero[name]).items():
                require_equal((len(chosen['P']),len(chosen['F']),int(chosen['P'][0])),
                    (32,case['budget'],case['first']),'stored baseline count/anchor')
                np.testing.assert_array_equal(chosen['F'],np.r_[chosen['P'],chosen['S']])
                np.testing.assert_array_equal(chosen['xyz'],case['support_xyz'][np.searchsorted(case['support_flat'],chosen['F'])])
            key='S3_n256_r0.25_t20_s0';mask=samples[key+'/object_id']==case['oid']
            np.testing.assert_array_equal(samples[key+'/points'][mask],one['xyz'])
            np.testing.assert_array_equal(samples[key+'/priority_selected'][mask],np.arange(case['budget'])<32)
            for flag in ('orientation_candidate','structural_candidate'):
                np.testing.assert_array_equal(case['saved_features'][flag],np.isin(one['F'],case['candidate_flat']))
            baselines[name]=dict(global1=one,global0=zero[name])
    return baselines


def load_synthetic_observations():
    """Read observations and saved baseline selections only; no annotation field."""
    manifest=json.loads((FIXTURE_RUN/'fixture_manifest.json').read_text())
    output={}
    with np.load(FIXTURE_RUN/(local.STEM+'.npz'),allow_pickle=False) as saved:
        for name in RULES['synthetic_names']:
            pre='synthetic/'+name+'/'
            flat,xyz=saved[pre+'candidate_flat'].copy(),saved[pre+'candidate_xyz'].copy()
            sources(flat,xyz)
            if len(flat)!=manifest[name]['counts']['observed_count']: raise AssertionError('fixture count')
            require_equal(prior.array_sha(flat),manifest[name]['flat_sha256'],'saved fixture source hash')
            require_equal(prior.array_sha(xyz),manifest[name]['xyz_sha256'],'saved fixture XYZ hash')
            shape=tuple(local.FIXTURES['shape']);local.uv_from_flat(flat,shape)
            seed=prior.fps_seed(0,0);anchor=int(flat[np.random.default_rng(seed).integers(len(flat))])
            k=np.array([[1600.,0,511.5],[0,1600.,511.5],[0,0,1]])
            scale=nominal_scale(xyz,k)
            g=choose_priority(flat,xyz,scale['a_m'],seed,anchor,32,shape)
            chosen=dict(R=g['R'],P=g['P'],grouping=g,scale=scale,dyadic=dyadic_counts(g['keys'],scale['a_m']))
            output[name]=dict(flat=flat,xyz=xyz,shape=shape,chosen=chosen,
                saved_global=saved[pre+'global/P'].copy(),saved_uv=saved[pre+'local/P'].copy())
    return output


def stage_gt(flat,context,labels,specs,transform):
    """Thin variable-cardinality adapter over the existing joint matcher."""
    matcher=context['matcher'];rows=np.searchsorted(matcher.flat,flat)
    ids,matched=matcher.reference_hits(flat);adj=matcher.adjacencies['2px_3mm'][rows]
    covered=np.zeros(matcher.nref,bool);covered[ids]=True
    xyz=matcher.xyz[rows];uv=matcher.uv[rows]
    segments={}
    for label in sorted(set(labels)):
        mask=labels==label;hit=adj[:,mask].getnnz(axis=1)>0
        spacing=dict(status='N/A',reason='physical single-line order unverified')
        if label in specs and transform is not None:
            wt=transform['world_to_cv'];rot=transform['rot']
            world=(xyz[hit]-wt[:3,3])@np.linalg.inv(wt[:3,:3].T)
            loc=(world-transform['box_pose'][:3])@np.linalg.inv(rot.T)
            spacing=prior.segment_spacing(specs[label]['local'][mask],covered[mask],loc[:,specs[label]['axis']],order_verified=True)
        segments[str(label)]=dict(reference_count=int(mask.sum()),observed_reference_count=int((mask&context['observable']).sum()),
            unique_reference_count=int((covered&mask).sum()),covered_reference_ids=np.flatnonzero(covered&mask).tolist(),
            matching_node_count=int(hit.sum()),matching_sources=flat[hit].tolist(),spacing=spacing)
    distance,nearest=nearest_sources(xyz,np.arange(matcher.nref,dtype=np.int64),context['reference'][1])
    coo=adj.tocoo()
    return dict(node_count=len(flat),matching_node_count=len(matched),unique_reference_count=len(ids),
        covered_reference_ids=ids.tolist(),matching_sources=matched.tolist(),segments=segments,
        candidate_to_GT_reference_mm=local.summary(distance*1000)),dict(flat_index=flat,xyz=xyz,pixel_uv=uv,
        covered=covered,matching_source=matched,joint_node_row=coo.row,joint_reference_id=coo.col,
        to_GT_reference_m=distance,to_GT_reference_id=nearest)


def evaluate_method(case,chosen,context,frame,labels,specs,transform,arrays,prefix):
    c=case['candidate_flat'];x=case['support_flat'];stages=dict(C=c,P=chosen['P'],S=chosen['S'],F=chosen['F'])
    if 'R' in chosen: stages['R']=chosen['R']
    details={}
    for stage,flat in stages.items():
        details[stage],raw=stage_gt(flat,context,labels,specs,transform)
        add_arrays(arrays,prefix+'/'+stage,raw)
        available,positions,features=prior.feature_rows(case,flat)
        add_arrays(arrays,prefix+'/'+stage,dict(feature_available=available,feature_available_positions=positions))
        add_arrays(arrays,prefix+'/'+stage+'/available_features',features)
        details[stage]['feature_available_count']=int(available.sum())
        details[stage]['feature_unavailable_count']=int((~available).sum())
    coverage=staged_reference_coverage(context['matcher'],c,chosen['P'],chosen['S'],chosen['F'])
    sets={k:set(v['covered_reference_ids']) for k,v in details.items()}
    differences={}
    for label,first,second in [('C_minus_R','C','R'),('R_minus_P','R','P'),('F_minus_P','F','P'),('P_intersect_S','P','S')]:
        if first in sets and second in sets:
            ids=sets[first]&sets[second] if label=='P_intersect_S' else sets[first]-sets[second]
            differences[label]=dict(count=len(ids),reference_ids=sorted(ids))
    occ=occupancy(frame,dict(X=x,**stages),context['negatives'])
    bounds=occ['depth_band_bounds_m'];zc=frame.depth.ravel()[c];zp=frame.depth.ravel()[chosen['P']]
    band=lambda z:(z<=bounds['median'],(z>bounds['median'])&(z<=bounds['p95']),z>bounds['p95'])
    occ['candidate_band_priority_selection_rates']=[dict(candidate_count=int(a.sum()),priority_count=int(b.sum()),
        rate=float(b.sum()/a.sum()) if a.any() else None) for a,b in zip(band(zc),band(zp))]
    spatial,raw=local.spatial_metrics(c,case['candidate_xyz'],chosen['P'],
        case['support_xyz'][np.searchsorted(x,chosen['P'])],x,case['support_xyz'],frame.depth.shape)
    add_arrays(arrays,prefix+'/spatial',raw)
    ds=local.nearest(case['support_xyz'],chosen['xyz'])*1000
    add_arrays(arrays,prefix,dict(X_to_F_mm=ds,F_selection_role=np.where(np.arange(len(chosen['F']))<len(chosen['P']),'Priority','Surface')))
    return dict(stages=details,coverage=coverage,reference_differences=differences,occupancy=occ,
        spatial=spatial,surface_mm=local.summary(ds))


def evaluate_after_freeze(output,selection_sha,frame,geometry,samples,cases,selections,baselines,synthetic,input_config):
    require_equal(prior.sha(output/'selection_before_gt.npz'),selection_sha,'pre-GT selection hash')
    arrays={};actual={}
    with np.load(prior.EDGES,allow_pickle=False) as edges,np.load(prior.NORMAL,allow_pickle=False) as normal:
        contexts=prior.evaluation_inputs(frame,geometry,samples,edges,normal)
        peg_labels,_=prior.peg_labels(normal['evaluation/peg/face'],normal['evaluation/peg/gt_visible'],
            normal['evaluation/peg/seam_endpoints'],contexts['peg']['reference'][0])
    with np.load(input_config['input'],allow_pickle=False) as episode:
        box_labels,specs,transform=prior.rim_labels(contexts['box']['reference'][1],frame.cam2world,
            episode['box_pose'][0],episode['box_hole_pose'][0],float(episode['box_hole_radii']))
    for name,case in cases.items():
        context=contexts[name];labels=peg_labels if name=='peg' else box_labels
        add_arrays(arrays,'evaluation/'+name,dict(reference_uv=context['reference'][0],reference_xyz=context['reference'][1],
            observable=context['observable'],segment=labels))
        methods={}
        for method,chosen in dict(**baselines[name],voxel=selections[name]).items():
            methods[method]=evaluate_method(case,chosen,context,frame,labels,{} if name=='peg' else specs,
                None if name=='peg' else transform,arrays,name+'/'+method)
        original_metrics=json.loads((local.SOURCE/'f0000_metrics.json').read_text())['S3_n256_r0.25_t20_s0']['objects'][name]
        score=context['matcher'].score(baselines[name]['global1']['F'])[0]
        require_equal(score,original_metrics['structural'],'original structural exact')
        require_equal(prior.stats(arrays[name+'/global1/X_to_F_mm']),original_metrics['surface']['all']['distance_mm'],'original Surface exact')
        saved_e=json.loads((local.BASELINE/(prior.STEM+'.json')).read_text())['cases']['S3/'+name]['conditions']['0.0']
        for stage in ('C','P','S','F'):
            require_equal(methods['global0']['stages'][stage]['unique_reference_count'],saved_e['stages'][stage]['unique_reference_count'],'E saved refs')
            require_equal(methods['global0']['stages'][stage]['matching_node_count'],saved_e['stages'][stage]['structural_matching_nodes'],'E saved nodes')
        compression,raw=distance_decomposition(case['candidate_flat'],case['candidate_xyz'],selections[name],frame.depth.shape)
        add_arrays(arrays,name+'/compression',raw)
        r=selections[name]['R'];rxyz=case['candidate_xyz'][np.searchsorted(case['candidate_flat'],r)]
        rspatial,rraw=local.spatial_metrics(case['candidate_flat'],case['candidate_xyz'],r,rxyz,case['support_flat'],case['support_xyz'],frame.depth.shape)
        add_arrays(arrays,name+'/R_spatial',rraw)
        support_scales=case['candidate_xyz'][:,2].astype(float)*np.hypot(2/frame.intrinsic[0,0],2/frame.intrinsic[1,1])
        add_arrays(arrays,name,dict(candidate_r5_m=support_scales))
        actual[name]=dict(scale=selections[name]['scale'],dyadic=selections[name]['dyadic'],anchor=selections[name]['grouping']['anchor'],
            compression=compression,R_spatial=rspatial,candidate_r5_mm=local.summary(support_scales*1000),methods=methods,
            timings_ms=selections[name]['timings_ms'])
    synth_result={}
    with np.load(FIXTURE_RUN/(local.STEM+'.npz'),allow_pickle=False) as saved:
        for name,data in synthetic.items():
            pre='synthetic/'+name+'/'
            annotation={k:saved[pre+k].copy() for k in ('structure_id','reference_xyz','reference_structure_id','reference_visible','lengths_m','generated_id')}
            annotation['occluded']=name=='camera_occluded'
            compression,raw=distance_decomposition(data['flat'],data['xyz'],data['chosen'],data['shape'])
            add_arrays(arrays,pre.rstrip('/')+'/compression',raw)
            row={}
            for method,p in dict(global0=data['saved_global'],uv=data['saved_uv'],voxel=data['chosen']['P']).items():
                row[method],raw=local.evaluate_synthetic(data,annotation,p)
                add_arrays(arrays,pre+method,raw)
            synth_result[name]=dict(scale=data['chosen']['scale'],dyadic=data['chosen']['dyadic'],compression=compression,
                anchor=data['chosen']['grouping']['anchor'],methods=row,uncertainty_calibration='N/A',surface='N/A')
            data['annotation']=annotation
    require_equal(prior.sha(output/'selection_before_gt.npz'),selection_sha,'post-GT selection unchanged')
    return actual,synth_result,arrays,contexts


def plot_results(output,frame,cases,selections,result,arrays,synthetic,contexts):
    def save(fig,name):
        fig.savefig(output/name,dpi=150,bbox_inches='tight');plt.close(fig)
    def points(ax,flat,color,label,marker='o',size=10):
        uv=local.uv_from_flat(flat,frame.depth.shape)
        ax.scatter(uv[:,0],uv[:,1],s=size,c=color,label=label,marker=marker,linewidths=.5)
    def view(ax,case):
        uv=local.uv_from_flat(case['support_flat'],frame.depth.shape);lo,hi=uv.min(0),uv.max(0)
        ax.imshow(frame.rgb);ax.set_xlim(lo[0]-8,hi[0]+8);ax.set_ylim(hi[1]+8,lo[1]-8)
        ax.set_xlabel('source u [px]');ax.set_ylabel('source v [px]')
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for row,(name,case) in enumerate(cases.items()):
        chosen=selections[name]
        for ax in axes[row]:view(ax,case)
        points(axes[row,0],case['candidate_flat'],'#38bdf8','C',size=8)
        points(axes[row,1],case['candidate_flat'],'#64748b','C',size=5)
        points(axes[row,1],chosen['R'],'#ff9f1c','R',size=12)
        points(axes[row,2],chosen['S'],'#a3e635','S',size=9)
        points(axes[row,2],chosen['P'],'#e879f9','P','*',35)
        for ax,title in zip(axes[row],(f'{name}: original C={len(case["candidate_flat"])}',
            f'nominal R={len(chosen["R"])} (not budget matched)',f'P={len(chosen["P"])} + S={len(chosen["S"])} = F={len(chosen["F"])}')):
            ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Observed source points: C / nominal representatives R / actual P + S = F',fontsize=15)
    save(fig,'primary_selection.png')

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,(name,row) in enumerate(result['actual'].items()):
        co=row['compression'];dy=row['dyadic'];s=row['scale']
        ax=axes[0,col];counts=[co['candidate_count'],co['nominal_occupied'],co['selected_occupied'],co['discarded_occupied']]
        ax.bar(['C','R occupied','P occupied','discarded'],counts,color=['#38bdf8','#ff9f1c','#d946ef','#64748b'])
        ax.axhline(32,color='red',ls='--',label='Priority cap=32');ax.legend()
        ax.set_title(f'{name}: epsilon={s["epsilon_m"]*1000:.3f} mm; a={s["a_m"]*1000:.3f} mm')
        for i,c in enumerate(counts):ax.text(i,c+3,str(c),ha='center')
        ax=axes[1,col];lev=dy['levels'];ax.plot([r['ratio'] for r in lev],[r['occupied'] for r in lev],'o-')
        ax.axhline(32,color='red',ls='--');ax.set_xscale('log',base=2)
        ax.set_xlabel('dyadic metric size / nominal (count only; not selected)');ax.set_ylabel('occupied cells')
        ax.set_title(f'First <=32: {dy["occupied"]} cells at {dy["ratio"]}x; actual selection stays 1x')
    fig.tight_layout();save(fig,'resolution_budget.png')

    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for col,(name,row) in enumerate(result['actual'].items()):
        ax=axes[0,col]
        for field,label in [('same_cell_m','same-cell'),('C_to_R_m','C -> R'),('C_to_P_m','C -> P'),('budget_extra_m','budget extra')]:
            d=arrays[f'{name}/compression/{field}_ecdf']*1000
            ax.step(d,np.arange(1,len(d)+1)/max(1,len(d)),where='post',label=label)
        ax.axvline(row['scale']['epsilon_m']*1000,color='grey',ls='--',label='nominal epsilon (R cell only)')
        ax.set_xlabel('Euclidean distance [mm]');ax.set_ylabel('fraction');ax.set_title(name+' distance decomposition');ax.legend(fontsize=8)
        ax=axes[1,col]
        for method in ('global1','global0','voxel'):
            d=arrays[f'{name}/{method}/spatial/distance_mm'];d=np.sort(d)
            ax.step(d,np.arange(1,len(d)+1)/len(d),where='post',label=method)
        ax.set_xlabel('C -> actual Priority [mm]');ax.set_ylabel('fraction');ax.set_title(name+' fixed-cap comparison');ax.legend()
    fig.tight_layout();save(fig,'primary_distances.png')

    for name,case in cases.items():
        row=result['actual'][name]['methods']['voxel'];context=contexts[name];uv=context['reference'][0]
        fig,axes=plt.subplots(2,3,figsize=(15,9))
        for ax,stage in zip(axes.flat,('C','R','P','S','F')):
            view(ax,case);covered=np.asarray(arrays[f'{name}/voxel/{stage}/covered'])
            ax.scatter(uv[:,0],uv[:,1],c='#ef4444',s=10,label='uncovered reference')
            ax.scatter(uv[covered,0],uv[covered,1],c='#22c55e',s=12,label='covered reference')
            points(ax,arrays[f'{name}/voxel/{stage}/flat_index'],'#c084fc',stage,'x',15)
            st=row['stages'][stage]
            ax.set_title(f'{stage}: nodes={st["node_count"]}; matching={st["matching_node_count"]}; unique refs={st["unique_reference_count"]}')
        ax=axes.flat[-1];view(ax,case)
        ax.scatter(uv[:,0],uv[:,1],s=8,c='#64748b',label='all refs')
        for key,color,label in [('C_minus_R','#ef4444','lost C -> R'),('R_minus_P','#f59e0b','lost R -> P'),('F_minus_P','#22c55e','added Surface')]:
            ids=row['reference_differences'][key]['reference_ids']
            ax.scatter(uv[ids,0],uv[ids,1],c=color,s=24,label=f'{label}: {len(ids)}')
        ax.set_title('Loss at compression / cap; Surface additions');ax.legend(fontsize=8)
        fig.suptitle(name+' auxiliary GT only: same-pair <=2px AND <=3mm; references != matching nodes')
        fig.tight_layout();save(fig,name+'_gt.png')

    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for row,(name,data) in enumerate(result['actual'].items()):
        ax=axes[row,0]
        for stage,color in [('C','#38bdf8'),('R','#ff9f1c'),('P','#d946ef'),('S','#84cc16'),('F','#8b5cf6')]:
            flat=arrays[f'{name}/voxel/{stage}/flat_index'];z=np.sort(frame.depth.ravel()[flat])
            ax.step(z,np.arange(1,len(z)+1)/len(z),where='post',label=stage,color=color)
        ax.set_xscale('log');ax.set_xlabel('original depth Z [m], log scale');ax.set_title(name+' retained depth tails');ax.legend()
        methods=list(data['methods']);x=np.arange(len(methods));ax=axes[row,1]
        for offset,stage in [(-.2,'P'),(0,'S'),(.2,'F')]:
            vals=[data['methods'][m]['occupancy']['negative_roi']['known_negative_union'][stage]['fraction_of_path']['value'] for m in methods]
            ax.bar(x+offset,vals,width=.2,label=stage)
        ax.set_xticks(x,methods);ax.set_ylabel('ROI fraction (not noise ground truth)');ax.set_title(name+' existing negative union');ax.legend()
        ax=axes[row,2]
        for offset,metric in [(-.2,'mean'),(0,'p95'),(.2,'max')]:
            ax.bar(x+offset,[data['methods'][m]['surface_mm'][metric] for m in methods],width=.2,label=metric)
        ax.set_xticks(x,methods);ax.set_ylabel('X -> F [mm]');ax.set_title(name+' Surface trade-off');ax.legend()
    fig.tight_layout();save(fig,'depth_roi_surface.png')

    fig,axes=plt.subplots(7,3,figsize=(15,23))
    for row,(name,data) in enumerate(synthetic.items()):
        for col,(method,p) in enumerate(dict(global0=data['saved_global'],uv=data['saved_uv'],voxel=data['chosen']['P']).items()):
            ax=axes[row,col];xyz=data['xyz'];pxyz=xyz[np.searchsorted(data['flat'],p)]
            ax.scatter(xyz[:,0]*1000,xyz[:,1]*1000,s=7,c='#94a3b8',label='stored C')
            if method=='voxel':
                r=xyz[np.searchsorted(data['flat'],data['chosen']['R'])]
                ax.scatter(r[:,0]*1000,r[:,1]*1000,s=13,c='#ff9f1c',label='nominal R')
            ax.scatter(pxyz[:,0]*1000,pxyz[:,1]*1000,s=28,c='#d946ef',marker='*',label='P')
            m=result['synthetic'][name]['methods'][method]
            ax.set_title(f'{name} / {method}: P={len(p)}, p95={m["mm"]["raw"]["p95"]:.3f}mm, noise={m["noise_selected"]}')
            ax.set_xlabel('camera X [mm]');ax.set_ylabel('camera Y [mm]');ax.legend(fontsize=7)
    fig.suptitle('Stored seven fixtures only; PCA5 localization calibration and Surface: N/A',y=1.001)
    fig.tight_layout();save(fig,'synthetic.png')


def report_blocks(result):
    b=[]
    p=lambda text:b.append(('p',text))
    h=lambda text:b.append(('h',text))
    table=lambda headers,rows:b.append(('table',(headers,rows)))
    image=lambda name,caption:b.append(('image',(name,caption)))
    fmt=lambda x:'N/A' if x is None else f'{x:.4f}'
    h('두 연구 질문에 대한 판단')
    for name,row in result['actual'].items():
        co=row['compression'];v=row['methods']['voxel'];g1=row['methods']['global1'];g0=row['methods']['global0']
        p(f'{name}: C={co["candidate_count"]} → nominal R={co["nominal_occupied"]} → P={co["priority_count"]}. '
          f'Nominal occupied/cap={co["occupied_over_32"]:.3f}배이며 {co["discarded_occupied"]}개 cell이 cap에서 제거됐다. '
          f'C→P p95는 voxel {fmt(v["spatial"]["mm"]["raw"]["p95"])}mm, 저장 Global λ=1 {fmt(g1["spatial"]["mm"]["raw"]["p95"])}mm, '
          f'λ=0 {fmt(g0["spatial"]["mm"]["raw"]["p95"])}mm다. '
          f'GT unique refs C/R/P/F={v["stages"]["C"]["unique_reference_count"]}/{v["stages"]["R"]["unique_reference_count"]}/'
          f'{v["stages"]["P"]["unique_reference_count"]}/{v["stages"]["F"]["unique_reference_count"]}다.')
    p('이번 결과는 Voxel precompression + 3D FPS cap의 첫 feasibility 평가다. 순수 Voxel Sampling이나 일반적으로 유효한 Shape-First Sampling의 검증이 아니다. '
      'Nominal R 전체는 cap 없는 해상도 진단 집합이므로 32점 기준선과 동등 budget 성능으로 비교하지 않는다. '
      '32점 cap 뒤의 거리·local coverage와 Surface/GT trade-off를 함께 해석한다.')
    p('A — 전반적 우월성을 지지하지 않는다. Peg는 λ=0보다 C→P tail이 개선됐지만 λ=1보다 p95/max가 소폭 악화했고, '
      'Final seam은 151/160(λ=1/0)에서 142로 줄었다. Box는 p95가 두 기준선보다 작지만 mean은 두 기준선보다 크고, '
      'Final rim은 3(λ=1)보다 많은 6이지만 12(λ=0)보다 적다. Surface p95는 개선됐으나 λ=1 대비 Surface mean은 두 물체 모두 소폭 악화했다. '
      'Box 한 지표의 개선만으로 형상 보존에 유리하다고 채택하지 않는다.')
    p('B — nominal 해상도와 32점은 양립하지 않았다. Peg/Box의 R은 432/356개로 cap의 13.5/11.125배다. '
      'C→R의 GT 손실은 0/9 reference였고 R→P에서 320/76개가 추가로 소실됐다. '
      '따라서 이 고정 경로에서는 cap 단계의 손실이 압축 단계보다 훨씬 컸다. '
      '다만 cap 수 자체의 효과와 그 32개를 고른 FPS 정책의 효과를 분리한 budget sweep은 하지 않았다.')
    p('N_occupied>32인 물체는 원래 해상도의 모든 voxel을 최종 보존하지 못했다. '
      'R 단계의 작은 오차와 R→P 추가 손실은 이 관측에서 예산 충돌의 크기를 보여주지만, '
      '32점이 유일한 인과적 병목이거나 더 큰 budget이 일반적인 최적값임을 증명하지 않는다. '
      '후보 오염, FPS 거리 경쟁과 anchor 이동이 남아 있다. 결과에 따른 재선택·파라미터 변경은 하지 않았다.')
    h('고정 조건·scale·실제 PCA5 지원 범위')
    p('ep000 Frame 0 / base_camera / S3 optimized PCA5 / orientation-only >20° / seed0. '
      'Final Peg129/Box127, 총256; Priority 상한 각32/총64, structural_ratio=.25는 현재 조건이다. '
      '전체 원본 X의 camera Z median, 저장 fx=fy=256px, multiplier=1로 epsilon=z_ref*sqrt((2/fx)^2+(2/fy)^2), a=epsilon/sqrt(3)를 정했다.')
    p('PCA5 normal은 중심 ±2px 5×5 내부의 same-object 4-connected support를 사용한다. '
      'Depth jump >.01+.02*min(z1,z2)에서 연결을 끊고, 최소6점·rank2·unique smallest eigenvector를 요구한다. '
      '불완전 patch를 허용하며 부족한 normal은 invalid다. Pair는 인접 normal을 비교해 >20°일 때 양 끝을 후보로 표시한다. '
      '두 normal support 합집합은 6×5 또는 5×6, 한 endpoint의 모든 비교 footprint는 7×7 bbox의 네 모서리를 제외한다. '
      '단순 5×5 window와 전체 반응 띠를 혼동하지 않는다.')
    p('Spatial support, 실제 edge localization error, voxel approximation error는 서로 다르다. '
      'Nominal epsilon은 동일 depth 평면의 half-window 근사에서 정한 compression scale이며 edge 오차의 추정값·상한이 아니다. '
      '경사·깊이 불연속에서는 근사가 깨진다. Same-cell 후보→대표점 bound sqrt(3)*a는 R 단계에서만 보장된다. '
      'Cell을 버린 P와 2px AND 3mm matcher 통과에는 보장되지 않는다. 새 normal/score/GT calibration은 없다.')
    table(['object','X median m','epsilon mm','a nominal=used mm','R occupied','R/32','P cells','removed','P>epsilon count/fraction','dyadic a mm / ratio / cells'],[
        [name,fmt(row['scale']['z_ref_m']),fmt(row['scale']['epsilon_m']*1000),fmt(row['scale']['a_m']*1000),
         row['compression']['nominal_occupied'],fmt(row['compression']['occupied_over_32']),row['compression']['selected_occupied'],row['compression']['discarded_occupied'],
         f'{row["compression"]["above_epsilon_count"]}/{row["compression"]["candidate_count"]} ({row["compression"]["above_epsilon_fraction"]:.2%})',
         f'{row["dyadic"]["a_budget_m"]*1000:.4f} / {row["dyadic"]["ratio"]}x / {row["dyadic"]["occupied"]}'] for name,row in result['actual'].items()])
    p('실제 used/nominal ratio는 1이다. Dyadic은 nominal integer key를 음수에도 floor_divide(2)하여 만든 nested count-only 진단이다. '
      '최초 occupied≤32에서 멈췄고 그 크기를 선택에 사용하지 않았다. 임의 크기 grid의 단조성이나 정확히32 cell 존재를 가정하지 않았다. '
      '이 dyadic family의 첫 크기를 전역 최소·물리적 최적 해상도로 읽지 않는다.')
    image('resolution_budget.png','Nominal 해상도와 현재 Priority 상한; coarse 진단은 선택에 미사용')
    h('선택 규칙·anchor 이동·원본 provenance')
    p('Camera origin=(0,0,0), 원본 CV XYZ(m)의 float64 floor(XYZ/a) int64 key, half-open cell을 사용했다. '
      'BBox 최소점으로 이동하지 않았다. Cell centroid에 가장 가까운 원본 C 하나, exact squared-distance tie는 최소 source다. '
      'R은 source ascending, original anchor cell의 대표점을 initial로 기존 Euclidean 3D FPS cap32를 수행했다. '
      'R≤32면 전부 사용하고 남는 예산은 전체 X의 기존 3D Surface FPS로 넘긴다. Padding·복제·좌표 평균 node는 없다.')
    table(['object','original anchor','mapped R source','3D shift mm','delta(u,v) px','pixel shift px'],[
        [name,row['anchor']['original_source'],row['anchor']['mapped_source'],fmt(row['anchor']['distance_m']*1000),
         str(row['anchor']['delta_pixel_uv']),fmt(row['anchor']['pixel_distance_px'])] for name,row in result['actual'].items()])
    p('Anchor source가 바뀌면 grouping/precompression과 FPS 시작점 효과가 함께 달라진다. '
      '추가 anchor-only arm은 실행하지 않았으므로 두 효과를 인과적으로 분해하지 않는다. '
      'C 전체 optimized score가 저장돼 있지 않아 score-free만 사용했다. Task02 reference PCA5 angle을 optimized score로 대체하지 않았다.')
    image('primary_selection.png','PCA5 원본 C / nominal 대표점 R / 실제 Priority P와 Surface S=Final F')
    h('Primary: 압축 오차와 budget 추가 손실')
    table(['object','distance (mm)','mean','p50','p95','max','worst source'],[
        [name,key,*[fmt(row['compression'][key][k]) for k in ('mean','p50','p95','max')],row['compression'][key]['worst']['source']]
        for name,row in result['actual'].items() for key in ('same_cell_m','C_to_R_m','C_to_P_m','budget_extra_m')])
    p('same_cell은 자기 cell의 대표점 거리, C→R은 모든 대표점 중 최근접 거리다. '
      'Budget extra=C→P−C→R≥0이며 same_cell을 빼지 않았다. 네 raw 배열, ECDF와 worst source/XYZ/pixel을 NPZ/JSON에 저장했다. '
      'Same-cell bound 확인과 성능 개선은 별개다.')
    table(['object','method','P','C→P mean/p50/p95/max mm','X→F mean/p95/max mm','Final unique GT refs'],[
        [name,method,m['stages']['P']['node_count'],' / '.join(fmt(m['spatial']['mm']['raw'][k]) for k in ('mean','p50','p95','max')),
         ' / '.join(fmt(m['surface_mm'][k]) for k in ('mean','p95','max')),m['stages']['F']['unique_reference_count']]
        for name,row in result['actual'].items() for method,m in row['methods'].items()])
    table(['object','method','UV selected/occupied; P=0 IDs','XYZ selected/occupied; P=0 IDs','UV equal-region mean p95 mm'],[
        [name,method,*[f'{g["candidate_occupied_regions"]-len(g["empty_priority_region_ids"])}/{g["candidate_occupied_regions"]}; {g["empty_priority_region_ids"]}'
            for g in (m['spatial']['grids']['uv'],m['spatial']['grids']['xyz'])],fmt(m['spatial']['grids']['uv']['equal_region_mean_p95_mm'])]
        for name,row in result['actual'].items() for method,m in row['methods'].items()])
    p('독립 UV4×4/XYZ 최대2×2×2 구역은 같은 X bbox로 고정했다. 구역별 후보 밀도·선택률·거리 tail과 R 단계 구역 통계는 JSON/NPZ에 있다. '
      'Box의 30.726m outlier는 그대로 남아 coarse XYZ 구역과 bbox 정규화를 왜곡하므로 raw mm와 UV를 함께 본다. '
      '단순 구역 점유가 같아도 국소 geometry가 같다는 뜻은 아니다.')
    image('primary_distances.png','Same-cell / C→R / C→P / budget extra ECDF와 저장 FPS 비교')
    h('Secondary: 선택 후 GT, C/R/P/S/F reference 손실')
    p('Same-pair ≤2px AND ≤3mm를 유지했다. Peg observed462/visible464, Box observed103/visible103이다. '
      '한 node는 여러 reference를 덮고 한 reference는 여러 node에 매칭될 수 있다. Pair count·matching node·unique reference는 서로 다르다.')
    table(['object','voxel stage','nodes','matching nodes','unique refs','available/unavailable feature rows'],[
        [name,stage,m['node_count'],m['matching_node_count'],m['unique_reference_count'],f'{m["feature_available_count"]}/{m["feature_unavailable_count"]}']
        for name,row in result['actual'].items() for stage,m in row['methods']['voxel']['stages'].items()])
    table(['object','C→R lost refs','R→P additional lost refs','Surface added refs','P/S overlapping refs'],[
        [name,*[row['methods']['voxel']['reference_differences'][key]['count'] for key in ('C_minus_R','R_minus_P','F_minus_P','P_intersect_S')]]
        for name,row in result['actual'].items()])
    table(['Box segment','reference count','C','R','P','S','F','Global λ1 P/F','Global λ0 P/F'],[
        [label,seg['reference_count'],*[result['actual']['box']['methods']['voxel']['stages'][stage]['segments'][label]['unique_reference_count'] for stage in ('C','R','P','S','F')],
         *[' / '.join(str(result['actual']['box']['methods'][method]['stages'][stage]['segments'][label]['unique_reference_count']) for stage in ('P','F')) for method in ('global1','global0')]]
        for label,seg in result['actual']['box']['methods']['voxel']['stages']['C']['segments'].items()])
    p('각 단계 reference ID 집합·joint pairs·matching source·구간별 matching node/reference·Box line spacing을 저장했다. '
      'Surface 추가분은 refs(F)−refs(P)이며 S 전체 hit 수와 다르다. C→R과 R→P 손실은 P⊆R⊆C에서 구분했다. '
      'Peg 물리적 line 순서/간격은 N/A다. GT 후보→reference 거리에는 depth noise, reference 이산화, 가시성·matcher 정의가 섞여 '
      '순수 detector localization error가 아니다. 이를 epsilon calibration에 사용하지 않았다.')
    image('peg_gt.png','Peg seam: C/R/P/S/F coverage와 압축/예산 손실')
    image('box_gt.png','Box rim: C/R/P/S/F coverage와 압축/예산 손실')
    h('Depth·ROI·일반 Surface trade-off')
    table(['object','stage','depth median/p95/max m','negative union count / nodes'],[
        [name,stage,' / '.join(fmt(row['methods']['voxel']['occupancy']['depth_m'][stage][k]) for k in ('median','p95','max')),
         f'{row["methods"]["voxel"]["occupancy"]["negative_roi"]["known_negative_union"][stage]["count"]} / '
         f'{row["methods"]["voxel"]["occupancy"]["negative_roi"]["known_negative_union"][stage]["fraction_of_path"]["denominator"]}']
        for name,row in result['actual'].items() for stage in ('X','C','R','P','S','F')])
    p('원거리 positive depth를 제거하지 않았다. ROI는 중첩되므로 union을 별도 집계했다. '
      'ROI/far는 실제 noise 정답이 아니며 일반 Surface 점유를 자동 오선택으로 분류하지 않는다. '
      'X median/p95 기반 depth bands와 candidate별 P 선택률도 저장했다. '
      'Median은 단일 extreme max 영향은 줄이지만 mask 오염과 물체 내 depth 차이를 해결하지 않는다. '
      'Peg/Box는 같은 공식으로 서로 다른 metric a를 사용하며 동일 절대 해상도 비교가 아니다.')
    image('depth_roi_surface.png','원본 depth tail, 기존 negative union, 전체 X→F Surface 비교')
    h('기존 7개 synthetic의 저장 입력 재사용')
    table(['fixture','C/R/P','a mm','dyadic ratio','Global/UV/Voxel C→P p95 mm','Global/UV/Voxel ref→P p95 mm','noise P G/UV/V'],[
        [name,f'{row["compression"]["candidate_count"]}/{row["compression"]["nominal_occupied"]}/{row["compression"]["priority_count"]}',
         fmt(row['scale']['a_m']*1000),row['dyadic']['ratio'],
         ' / '.join(fmt(row['methods'][m]['mm']['raw']['p95']) for m in ('global0','uv','voxel')),
         ' / '.join(fmt(row['methods'][m]['reference_mm']['p95']) for m in ('global0','uv','voxel')),
         ' / '.join(str(row['methods'][m]['noise_selected']) for m in ('global0','uv','voxel'))]
        for name,row in result['synthetic'].items()])
    p('Task03-F 저장 C/source/XYZ와 Global/UV 선택을 읽었으며 새 fixture·기준선 FPS를 만들지 않았다. '
      'Dense X가 없는 stress test에서 관측 C median과 기존 focal1600px로 같은 scale 공식을 적용했다. '
      'Annotation은 모든 선택 저장 후에만 noise/reference/구조별 평가에 사용했다. '
      '실제 PCA5 5×5 localization uncertainty calibration과 Synthetic Surface는 N/A다. '
      '계약/geometry stress test 통과가 uncertainty calibration 타당성을 뜻하지 않는다.')
    image('synthetic.png','기존 일곱 fixture의 저장 Global/UV와 단일 voxel precompression control')
    h('검증·재현·한계·완료 범위')
    p(f'실행 UUID {result["run_id"]}. 필수 계약 테스트: {result["tests_summary"]}. '
      '성능 개선을 PASS 조건으로 두지 않았다. selection_before_gt.npz를 real/synthetic 선택 전체의 저장·readback·hash 이후에만 '
      'GT evaluator/annotation에 연결했다. 평가 후 선택 hash 불변, 원본 C/X·baseline exact alignment, 원본 feature available-only exact gather를 확인했다.')
    p('기존 ExperimentLogger(team=visual, job_type=evaluate), WANDB_MODE=disabled 및 로컬 artifact inventory를 재사용했다. '
      '기존 UUID/project/group 규칙을 바꾸지 않았다. Git 시작 상태는 dirty이며 source snapshot·input/artifact hashes·시작 diff·implementation diff를 함께 보존했다. '
      'Production sampling/normal/tracking/shared simulator와 기존 dirty/untracked 파일은 변경하지 않았다. '
      '같은 S3 조건에서 저장된 feature가 없는 source는 unavailable로 남겼으며 feature를 추정하지 않았다.')
    p('보호 대상 기존 파일 3,092개의 SHA-256와 HEAD 불변을 확인했다. Markdown↔HTML 및 JSON/NPZ/PNG 상대 링크, '
      'PNG 7개 decode는 PASS다. 실제 브라우저 로딩은 computer-use 권한이 없어 진행하지 못했고, 사용자의 후속 지시에 따라 생략했다. '
      '이 미실행 항목을 검증 PASS로 계산하지 않는다. 수치 실행 source_snapshot과 선택·JSON/NPZ는 유지하고 '
      '최종 해석 문구/검증 상태를 보완한 보고 source는 report_source_snapshot과 report_generation_sources로 구분한다.')
    p('CPU NumPy/SciPy offline 실행이다. 단계별 wall time은 JSON에 기록했으며 Graph/GNN 20Hz 성능을 주장하지 않는다. '
      'Simulator/renderer/CUDA/MPS/GPU/PCL, 새 episode/fixture/score/normal, threshold/scale sweep, 두 번째 selection arm, '
      'GT calibration/quota, depth filtering, ratio/Adaptive Budget, Graph/GNN/BC, dependency 설치·외부 업로드·Git commit은 미실행이다. '
      'Task03-H의 동일 count 기대 coverage는 후보 해석 근거, Task03-F/G는 기존 실패 비교군으로만 유지했다. '
      '보고서 제출 후 추가 알고리즘 실행 없이 중단한다.')
    return b


def write_reports(output,result):
    blocks=report_blocks(result);relative='../../experiments/'+output.name+'/'
    artifacts=[STEM+'.html',STEM+'.json',STEM+'.npz','selection_before_gt.npz','selection_before_gt.json',
        'verification.json','tests.log','source_hashes.json','preregistered_config.json','implementation.patch','git_start.diff']
    md=['# Task03-I — Shape-First Voxel Precompression + 3D FPS Cap','',
        ' · '.join(f'[{f}]({relative+f})' for f in artifacts),'',
        '[승인 계획](03i_voxel_structural_sampling_plan.md) · [Task03-H](03h_candidate_count_matched.md) · '
        '[Task03-F](03f_local_structural_sampling.md) · [Task03-G](03g_geometry_aware_sampling.md)','']
    page=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Task03-I — Voxel precompression + FPS cap</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f9;color:#17212e;margin:auto;max-width:1450px;padding:30px}h1,h2{line-height:1.25}h2{margin-top:2.5em;border-bottom:2px solid #cbd5e1;padding-bottom:12px}p{max-width:1150px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{border:1px solid #cbd5e1;padding:9px;text-align:left}th{background:#e2e8f0}tr:nth-child(even){background:#f8fafc}.scroll{overflow:auto}img{width:100%;height:auto;background:white;border:1px solid #cbd5e1}figure{margin:25px 0}figcaption{color:#475569}a{color:#155eab}nav{padding:18px;background:#e0f2fe}code,pre{overflow:auto}</style>',
        '<h1>Task03-I — Shape-First Voxel Precompression + 3D FPS Cap</h1>',
        f'<nav>Run {output.name}<br><a href="../../docs/visual-graph-integration/03i_voxel_structural_sampling.md">결과 Markdown</a> · '
        '<a href="../../docs/visual-graph-integration/03i_voxel_structural_sampling_plan.md">승인 계획</a><br>'+ ' · '.join(f'<a href="{f}">{f}</a>' for f in artifacts if not f.endswith('.html'))+'</nav>']
    for kind,value in blocks:
        if kind=='h':md.extend(['## '+value,'']);page.append('<h2>'+html.escape(value)+'</h2>')
        elif kind=='p':md.extend([value,'']);page.append('<p>'+html.escape(value)+'</p>')
        elif kind=='image':
            filename,caption=value;md.extend([f'![{caption}]({relative+filename})',''])
            page.append(f'<figure><a href="{filename}"><img src="{filename}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>')
        else:
            headers,rows=value
            esc=lambda x:str(x).replace('|','/').replace('\n',' ')
            md.extend(['| '+' | '.join(map(esc,headers))+' |','| '+' | '.join(['---']*len(headers))+' |'])
            md.extend('| '+' | '.join(map(esc,row))+' |' for row in rows);md.append('')
            page.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>')
    page.append('</html>')
    REPORT.write_text('\n'.join(md));(output/(STEM+'.html')).write_text('\n'.join(page))


def verify_documents(output):
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.targets=[]
        def handle_starttag(self,tag,attrs):self.targets.extend(v for k,v in attrs if k in ('href','src'))
    parser=Links();page=(output/(STEM+'.html')).read_text();parser.feed(page)
    if re.search(r'<script[^>]+src=|\bfetch\s*\(|https?://',page):raise AssertionError('external/dynamic HTML dependency')
    for target in parser.targets:
        if Path(target).is_absolute() or not (output/target).is_file():raise AssertionError('bad HTML link '+target)
    mdlinks=re.findall(r'\]\(([^)]+)\)',REPORT.read_text())
    for target in mdlinks:
        if Path(target).is_absolute() or not (REPORT.parent/target).is_file():raise AssertionError('bad Markdown link '+target)
    decoded={}
    for filename in FIGURES:
        a=plt.imread(output/filename)
        if min(a.shape[:2])<400 or not np.isfinite(a).all() or a.std()<.01:raise AssertionError('bad PNG '+filename)
        decoded[filename]=list(a.shape)
    return dict(status='PASS',html_relative_links=len(parser.targets),markdown_relative_links=len(mdlinks),png_decoded=decoded,
        static_no_cdn_no_fetch=True)


def preserved(baseline):
    current={p:prior.sha(Path(p)) for p in baseline['hashes']}
    require_equal(current,baseline['hashes'],'protected original hashes')
    require_equal(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),baseline['head'],'HEAD unchanged')
    return dict(status='PASS',protected_file_count=len(current),all_sha256_unchanged=True,HEAD_unchanged=True)


def run(baseline_path):
    if REPORT.exists():raise FileExistsError('existing result report; refusing overwrite')
    if os.environ.get('WANDB_MODE')!='disabled':raise ValueError('WANDB_MODE=disabled required')
    baseline=json.loads(baseline_path.read_text());preserved(baseline)
    source_paths=sorted(set([Path(__file__),Path(__file__).with_name('test_voxel_structural_sampling.py'),PLAN,
        *[p for p in (REPO/'visual/runtime').rglob('*.py')],REPO/'utils/experiment_logger.py',REPO/'utils/experiment_artifacts.py']))
    source_hash=prior.hashes(source_paths)
    config=dict(task='03-I',rules=RULES,source_sha256=source_hash,protected_baseline_sha256=prior.sha(baseline_path),
        source_run_id=prior.SOURCE_RUN,baseline_run_id=local.BASELINE_ID,fixture_run_id=FIXTURE_RUN.name,
        python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),device='cpu',
        simulator_executed=False,gpu_executed=False,baseline_FPS_executed=False,production_policy_adopted=False)
    with ExperimentLogger(team='visual',config=config,seed=0,model='task03i_voxel_precompression_3d_fps_cap',job_type='evaluate') as logger:
        output=logger.directory
        def registered(path,kind='reports'):logger.link_output(path,kind=kind)
        try:
            write_json(output/'git_start.json',dict(head=baseline['head'],status=baseline['git_status']))
            (output/'git_start.diff').write_text(baseline['git_diff'])
            shutil.copyfile(baseline_path,output/'protected_baseline.json')
            write_json(output/'preregistered_config.json',config);config_sha=prior.sha(output/'preregistered_config.json')
            for path in source_paths:
                dest=output/'source_snapshot'/path.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
            for filename in ('git_start.json','git_start.diff','protected_baseline.json','preregistered_config.json'):registered(output/filename)
            with (output/'tests.log').open('w') as log:
                process=subprocess.run([sys.executable,'-m','unittest',*TESTS,'-v'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            registered(output/'tests.log')
            if process.returncode:raise RuntimeError('required contracts failed; no real selection performed')
            tests=(output/'tests.log').read_text();summary=next(s for s in tests.splitlines() if s.startswith('Ran '))+'; '+tests.splitlines()[-1]
            print(summary,flush=True)
            frame,geometry,samples,cases,zero,old,input_config,contracts=local.load_actual()
            baselines=load_baselines(cases,zero,samples)
            selections={}
            for name,case in cases.items():
                selections[name]=select(case['candidate_flat'],case['candidate_xyz'],case['support_flat'],case['support_xyz'],
                    frame.depth.shape,frame.intrinsic,case['seed'],case['first'],case['budget'])
                check_selection(case,selections[name])
            require_equal(sum(len(s['F']) for s in selections.values()),256,'total Final budget')
            synthetic=load_synthetic_observations()
            selection_sha,selection_data=freeze_selection(output,cases,selections,synthetic)
            registered(output/'selection_before_gt.npz')
            write_json(output/'selection_before_gt.json',dict(sha256=selection_sha,gt_connected=False,
                actual={name:dict(scale=s['scale'],dyadic=s['dyadic'],anchor=s['grouping']['anchor'],
                    counts={k:len(s[k]) for k in ('R','P','S','F')}) for name,s in selections.items()},
                synthetic={name:dict(scale=s['chosen']['scale'],anchor=s['chosen']['grouping']['anchor']) for name,s in synthetic.items()}))
            registered(output/'selection_before_gt.json')
            print('All selections frozen before GT:',output.name,flush=True)
            t0=perf_counter()
            actual,synth_result,arrays,contexts=evaluate_after_freeze(output,selection_sha,frame,geometry,samples,cases,selections,baselines,synthetic,input_config)
            result=dict(run_id=logger.run_id,config=config,tests_summary=summary,actual=actual,synthetic=synth_result,
                selection_before_gt_sha256=selection_sha,evaluation_ms=(perf_counter()-t0)*1000,source_contract=contracts,
                real_FPS_calls=dict(baseline=0,priority_cap=sum(len(s['R'])>32 for s in selections.values()),surface=2),
                synthetic_FPS_calls=dict(baseline=0,priority_cap=sum(len(d['chosen']['R'])>32 for d in synthetic.values())),
                policy_adopted=False)
            arrays.update(selection_data)
            np.savez_compressed(output/(STEM+'.npz'),**arrays);registered(output/(STEM+'.npz'))
            with np.load(output/(STEM+'.npz'),allow_pickle=False) as saved:
                for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value,err_msg=key)
            write_json(output/(STEM+'.json'),result);registered(output/(STEM+'.json'),'metrics')
            plot_results(output,frame,cases,selections,result,arrays,synthetic,contexts)
            for filename in FIGURES:registered(output/filename)
            preservation=preserved(baseline)
            require_equal(prior.hashes(source_paths),source_hash,'executed source hashes')
            require_equal(prior.sha(output/'preregistered_config.json'),config_sha,'frozen config')
            require_equal(prior.sha(output/'selection_before_gt.npz'),selection_sha,'frozen selection after evaluation')
            write_json(output/'source_hashes.json',dict(sources_before=source_hash,sources_after=prior.hashes(source_paths),
                protected_before=baseline['hashes'],protected_after={p:prior.sha(Path(p)) for p in baseline['hashes']},preservation=preservation))
            registered(output/'source_hashes.json')
            patch_text=''
            for path in (Path(__file__),Path(__file__).with_name('test_voxel_structural_sampling.py')):
                patch_text+=subprocess.run(['git','diff','--no-index','--','/dev/null',str(path.relative_to(REPO))],cwd=REPO,capture_output=True,text=True).stdout
            (output/'implementation.patch').write_text(patch_text);registered(output/'implementation.patch')
            verification=dict(status='pending report verification',required_tests=dict(summary=summary,modules=TESTS,
                pass_count=sum(' ... ok' in line for line in tests.splitlines()),fail_count=0,skip_count=sum(' ... skipped' in line for line in tests.splitlines())),
                baseline_exact=True,C_X_source_XYZ_exact=True,P_subset_R_subset_C_subset_X=True,no_padding_duplicates=True,
                fixed_origin_floor=True,centroid_representatives_original_only=True,variable_P_contract=True,dyadic_nested_counts=True,
                pre_GT_saved_readback_hash=selection_sha,selection_hash_unchanged_after_GT=True,feature_exact_available_only=True,
                same_pair_2px_AND_3mm=True,source_preservation=preservation,config_sha256=config_sha,NPZ_exact_readback=True,
                simulator_executed=False,gpu_executed=False,performance_improvement_is_PASS_condition=False,
                source_snapshots='source_snapshot/',static_browser_loading='pending')
            write_json(output/'verification.json',verification)
            write_reports(output,result)
            verification['documents']=verify_documents(output)
            verification['status']='PASS; static browser loading pending'
            write_json(output/'verification.json',verification);registered(output/'verification.json')
            registered(output/(STEM+'.html'));registered(REPORT)
            write_json(output/'git_end.json',dict(status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),preservation=preserved(baseline)))
            registered(output/'git_end.json')
            logger.log({name:dict(compression=row['compression'],surface=row['methods']['voxel']['surface_mm']) for name,row in actual.items()},step=0,prefix='eval')
            print(output,flush=True)
        except Exception as exc:
            write_json(output/'attempt_failure.json',dict(error_type=type(exc).__name__,message=str(exc),real_selection_saved=(output/'selection_before_gt.npz').exists()))
            registered(output/'attempt_failure.json');raise
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--finalize-saved',type=Path,help='Report/verification only; no FPS, evaluation, tests or new UUID')
    args=parser.parse_args()
    if args.finalize_saved:finalize_saved(args.finalize_saved,args.baseline)
    else:run(args.baseline)


def finalize_saved(output,baseline_path):
    """Finalize the completed local run after browser validation was waived."""
    from utils.experiment_artifacts import register_existing_output
    result=json.loads((output/(STEM+'.json')).read_text())
    require_equal(output.name,result['run_id'],'saved identity')
    metadata=json.loads((output/'metadata.json').read_text())
    require_equal(metadata['status'],'finished','completed numerical run')
    immutable=[output/(STEM+'.json'),output/(STEM+'.npz'),output/'selection_before_gt.npz',output/'tests.log']
    before={str(p):prior.sha(p) for p in immutable}
    require_equal(before[str(output/'selection_before_gt.npz')],result['selection_before_gt_sha256'],'saved pre-GT hash')
    baseline=json.loads(baseline_path.read_text());preservation=preserved(baseline)
    current=[Path(__file__),Path(__file__).with_name('test_voxel_structural_sampling.py')]
    for path in current:
        destination=output/'report_source_snapshot'/path.relative_to(REPO)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    report_patch=''
    for path in current:
        report_patch+=subprocess.run(['git','diff','--no-index','--',str(output/'source_snapshot'/path.relative_to(REPO)),str(path)],
            cwd=REPO,capture_output=True,text=True).stdout
    (output/'report_refinement.patch').write_text(report_patch)
    write_reports(output,result)
    verification=json.loads((output/'verification.json').read_text())
    verification['documents']=verify_documents(output)
    verification['static_browser_loading']=dict(status='SKIP',reason='Computer-use permissions unavailable; user explicitly requested completion without browser execution',images_verified_by_decode=7)
    verification['status']='PASS for executed checks; browser loading SKIP by user instruction'
    ran=int(re.search(r'Ran (\d+) tests',result['tests_summary']).group(1))
    verification['required_tests']['pass_count']=ran-verification['required_tests']['skip_count']
    verification['source_preservation']=preservation
    verification['report_only_postprocessing']=dict(new_UUID=False,new_FPS=False,new_evaluation=False,numerical_artifacts_unchanged=True)
    write_json(output/'verification.json',verification)
    hashes=json.loads((output/'source_hashes.json').read_text())
    hashes['report_generation_sources']=prior.hashes(current)
    hashes['final_protected_preservation']=preservation
    hashes['numerical_artifacts_before_after']=dict(before=before,after={str(p):prior.sha(p) for p in immutable})
    require_equal(hashes['numerical_artifacts_before_after']['before'],hashes['numerical_artifacts_before_after']['after'],'report only preserves numerical artifacts')
    write_json(output/'source_hashes.json',hashes)
    for path in (REPORT,output/(STEM+'.html'),output/'verification.json',output/'source_hashes.json',output/'report_refinement.patch',
                 *[(output/'report_source_snapshot'/p.relative_to(REPO)) for p in current]):
        register_existing_output(output,path,kind='reports')
    write_json(output/'git_end.json',dict(status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),preservation=preserved(baseline)))
    register_existing_output(output,output/'git_end.json',kind='reports')
    print(json.dumps(dict(run_id=output.name,documents=verification['documents'],preservation=preservation,browser='SKIP',no_new_numerical_work=True)))


if __name__=='__main__':main()
