"""Task05-E fixed offline comparison; no simulator, sampling or publishing.

Run with WANDB_MODE=disabled and all THREADS=1 before interpreter startup.
--review-stdin keeps the same ExperimentLogger open for external visual review;
write reviewed evidence into review.json, then send one newline to finalize.
"""
import argparse
from collections import Counter
import copy
import difflib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter, process_time
from unittest.mock import patch

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from graph.vision_input import receive_vision_packet
from utils.experiment_logger import ExperimentLogger
from visual.runtime.extended_delaunay_edges import (
    candidate_edges, directed_edges, build_extended_delaunay_edges, edge_metadata)
from visual.runtime.test.raw_ede_comparison_adapter import build_raw_ede
from visual.runtime.test import ede_exact_optimization_eval as c

io = c.io
REPO = io.REPO
B = c.SOURCE
C = REPO/'experiments/185bc034567d429daf3f1d09f94dc56c'
D = REPO/'experiments/be206b04f7b245c4a9436317b8f87987'
REPORT = REPO/'docs/visual-graph-integration/05e_raw_ede_comparison.md'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
NEW = ('visual/runtime/test/raw_ede_comparison_adapter.py',
       'visual/runtime/test/run_raw_ede_comparison.py', 'visual/runtime/test/test_raw_ede_comparison.py')
EDIT = ('graph/vision_input.py', 'docs/visual-graph-integration/00_interface.md')
REASONS = ('mask_outside', 'invalid_depth', 'overlap', 'pair_unsupported', 'pair_missing', 'delta_invalid', 'depth_jump')


def execute(policy, original, evidence=None):
    """Measured boundary includes fresh Strict preparation and actual receiver."""
    start, cpu = perf_counter(), process_time()
    tick = perf_counter(); p = copy.deepcopy(original)
    timing = dict(common_point_component_preparation_ms=(perf_counter()-tick)*1000)
    if policy == 'Raw':
        edge, metadata, audit = build_raw_ede(p)
    else:
        edge, audit = build_extended_delaunay_edges(p['camera_points'], p['object_id'],
            p['provenance']['source_index'], p['observed_component_id'], evidence,
            extended=True, optimized=True)  # contexts=None: fresh lookup each call
        tick = perf_counter(); metadata = edge_metadata(audit, extended=True)
        timing['metadata_packaging_ms'] = (perf_counter()-tick)*1000
    timing.update(audit['timing_ms'])
    tick = perf_counter(); p.update(metadata, intra_object_edge_index=edge)
    timing['packet_packaging_ms'] = (perf_counter()-tick)*1000
    tick = perf_counter(); received = receive_vision_packet(p)
    timing['receiver_ms'] = (perf_counter()-tick)*1000
    timing['total_wall_ms'] = (perf_counter()-start)*1000
    timing['total_cpu_ms'] = (process_time()-cpu)*1000
    return p, audit, timing, received


def offline_oracle(packet, expected=None):
    """Algorithm oracle, deliberately outside receiver and benchmark timers."""
    a = candidate_edges(packet['object_id'], packet['provenance']['source_index'],
        packet['observed_component_id'], packet['image_shape'], extended=True)
    for field, value in [('raw_candidate_pairs', a['candidates']),
                         ('raw_candidate_origins', a['origins'].tolist())]:
        c.exact(packet[field], value, field+'/offline_oracle')
    c.exact(packet['intra_object_edge_index'], directed_edges(a['candidates']), 'oracle directed')
    if expected is not None:
        for field in ('candidates', 'origins', 'triangles'):
            c.exact(a[field], expected[field], 'stored oracle/'+field)
    io.require(not [s for s in a['components'] if s['status'] in ('qhull_error', 'omitted_vertices')],
               'incomplete candidates: stop comparison')
    return a


def protect(initial):
    for rel, sha in initial['hashes'].items():
        if rel not in EDIT:c.exact(io.sha(REPO/rel), sha, 'protected '+rel)
    return dict(status='PASS', files=len(initial['hashes']), approved_edits=list(EDIT))


def capture_initial():
    """Fresh per-invocation preservation; replay never depends on a /tmp setup."""
    import tempfile
    status=subprocess.check_output(['git','status','--porcelain','-uall'],cwd=REPO,text=True)
    source_runs=(io.SELECTION.name,io.BASE.name,io.OLD.name,B.name,C.name,D.name)
    files={REPO/line[3:] for line in status.splitlines() if (REPO/line[3:]).is_file()
           and line[3:] not in (*NEW,str(REPORT.relative_to(REPO)))}
    for run in source_runs:
        files.update(p for p in (REPO/'experiments'/run).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    old=json.loads((B/'extended_delaunay_edgewise.json').read_text())
    for rel,sha in old['input_hashes'].items():
        c.exact(io.sha(REPO/rel),sha,'source input hash')
        files.add(REPO/rel)
        sidecar=REPO/(rel+'.tracking.json')
        if sidecar.is_file():files.add(sidecar)
    before=Path(tempfile.mkdtemp(prefix='task05e-preservation-'))/'snapshot'
    for line in status.splitlines():
        rel=line[3:];path=REPO/rel
        if path.is_file() and rel not in NEW and not path.is_relative_to(REPO/'experiments'):
            dest=before/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    return dict(status=status,source_runs=list(source_runs),snapshot_directory=str(before),
        hashes={str(path.relative_to(REPO)):io.sha(path) for path in sorted(files)})


def source_save(out, initial):
    paths = set(NEW+EDIT) | {'visual/runtime/extended_delaunay_edges.py',
        'visual/runtime/triangle_intra_object_edges.py', 'visual/runtime/vision_packet.py',
        'visual/runtime/test/ede_exact_optimization_eval.py',
        'visual/runtime/test/run_extended_delaunay_edges.py', 'utils/experiment_logger.py',
        'utils/experiment_artifacts.py', 'docs/visual-graph-integration/05e_raw_ede_comparison_plan.md'}
    hashes, diff = {}, []
    for rel in sorted(paths):
        dest = out/'source_snapshot'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/rel, dest); hashes[rel] = io.sha(dest)
        if rel in NEW+EDIT:
            before = Path(initial.get('snapshot_directory','/tmp/task05e_initial/snapshot'))/rel
            old = before.read_text().splitlines(True) if before.exists() else []
            if before.exists():
                q=out/'before_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(before,q)
            diff.extend(difflib.unified_diff(old, dest.read_text().splitlines(True), fromfile='before/'+rel, tofile='after/'+rel))
    (out/'implementation.diff').write_text(''.join(diff))
    io.write(out/'source_hashes.json', dict(evaluated_sources=hashes, protected_before=initial))


def input_gate(out):
    tick=perf_counter(); packets, evidences, old=c.inputs()
    for rel, sha in old['input_hashes'].items():c.exact(io.sha(REPO/rel),sha,'original input')
    # Verify unchanged numerical helpers against the optimized experiment snapshot.
    h=json.loads((C/'source_hashes.json').read_text())
    binding={}
    for rel in ('visual/runtime/extended_delaunay_edges.py','visual/runtime/triangle_intra_object_edges.py'):
        snapshot=(C/'source_snapshot'/rel) if (C/'source_snapshot'/rel).exists() else (B/'source_snapshot'/rel)
        c.exact(io.sha(REPO/rel),io.sha(snapshot),'Task05-C/B helper '+rel)
        binding[rel]=io.sha(REPO/rel)
    gate=dict(status='PASS', conditions=len(packets), phase=0, upstream_recomputed=False,
        input_hashes=old['input_hashes'], numerical_helpers=binding, IO_setup_wall_ms=(perf_counter()-tick)*1000,
        upstream_component_generation='NOT_EXECUTED; stored observed_component_id only')
    io.write(out/'input_gate.json',gate)
    return packets,evidences,old,gate


def geometries(p,pairs):
    src=p['provenance']['source_index']; w=int(p['image_shape'][1]);uv=np.column_stack((src%w,src//w))
    xyz=p['camera_points'].astype(float)
    return dict(uv_length_px=np.linalg.norm(uv[pairs[:,0]]-uv[pairs[:,1]],axis=1),
        camera_length_m=np.linalg.norm(xyz[pairs[:,0]]-xyz[pairs[:,1]],axis=1),
        signed_depth_difference_m=xyz[pairs[:,1],2]-xyz[pairs[:,0],2],
        absolute_depth_difference_m=np.abs(xyz[pairs[:,1],2]-xyz[pairs[:,0],2]))


def topology(n,pairs):
    edges=directed_edges(pairs);degree=np.bincount(edges[0],minlength=n)
    adj=coo_array((np.ones(edges.shape[1],bool),(edges[0],edges[1])),shape=(n,n)).tocsr()
    labels=connected_components(adj,directed=False)[1] if n else np.empty(0,np.int64)
    return degree,labels


def analyze(key,p,a,accepted,checks):
    pairs=a['candidates'];oid=p['object_id'];src=p['provenance']['source_index'];n=len(oid)
    sd,sl=topology(n,pairs[accepted]);rd,rl=topology(n,pairs)
    geo=geometries(p,pairs);slices={};case_ids=set()
    for obj in (0,1):
        rows=np.flatnonzero(oid==obj); ids=np.flatnonzero(oid[pairs[:,0]]==obj);added=ids[~accepted[ids]]
        single=np.concatenate([np.asarray(s['rows'],np.int64) for s in a['components']
            if s['object_id']==obj and s['status']=='singleton']) if any(s['object_id']==obj and s['status']=='singleton' for s in a['components']) else np.empty(0,np.int64)
        io.require((sd[single]==0).all() and (rd[single]==0).all(),'singleton isolated identity')
        merges=[]
        for label in np.unique(rl[rows]):
            members=rows[rl[rows]==label];strict_ids=np.unique(sl[members])
            if len(strict_ids)>1:
                cross=added[(rl[pairs[added,0]]==label)&(sl[pairs[added,0]]!=sl[pairs[added,1]])]
                merges.append(dict(raw_label=int(label),rows=members,strict_labels=strict_ids,
                    strict_sizes=[int((sl[rows]==x).sum()) for x in strict_ids],cross_added_candidate_ids=cross))
        selected={}
        for metric in ('uv_length_px','camera_length_m','absolute_depth_difference_m'):
            top=added[np.argsort(-geo[metric][added],kind='stable')[:5]]
            for i in top:selected.setdefault(int(i),[]).append('top5_'+metric)
        for combo in sorted({tuple(checks[i]['reasons']) for i in added}):
            i=next(int(i) for i in added if tuple(checks[i]['reasons'])==combo)
            selected.setdefault(i,[]).append('first_reason_combination')
        case_ids.update(selected)
        metrics={}
        for label,d,l,subset in [('Strict',sd,sl,ids[accepted[ids]]),('Raw',rd,rl,ids),('Added',rd,rl,added)]:
            metrics[label]=dict(U=len(subset),E=2*len(subset),geometry={k:io.stats(v[subset]) for k,v in geo.items()})
            if label!='Added':metrics[label].update(isolated=int((d[rows]==0).sum()),
                isolated_rows=rows[d[rows]==0],graph_CC=len(np.unique(l[rows])),
                component_sizes=sorted(np.unique(l[rows],return_counts=True)[1].tolist(),reverse=True),
                degree=io.stats(d[rows]))
        slices[str(obj)]=dict(object_id=obj,N=len(rows),candidate_U=len(ids),metrics=metrics,
            singleton_rows=single,singleton_source=src[single],newly_connected_rows=rows[(sd[rows]==0)&(rd[rows]>0)],
            added_incident_rows=np.unique(pairs[added]),merges=merges,selected_case_ids=selected,
            reasons={r:sum(r in checks[i]['reasons'] for i in added) for r in REASONS},
            reason_combinations=dict(Counter('|'.join(checks[i]['reasons']) for i in added)))
    cases=[]
    for i in sorted(case_ids):
        pair=pairs[i];srcpair=src[pair];w=int(p['image_shape'][1]);obj=int(oid[pair[0]])
        cases.append(dict(id=key+':'+str(i),candidate_id=i,object_id=obj,rows=pair,source=srcpair,
            uv=np.column_stack((srcpair%w,srcpair//w)),camera_xyz=p['camera_points'][pair],ee_xyz=p['points'][pair],
            origin=str(a['origins'][i]),geometry={k:float(v[i]) for k,v in geo.items()},
            selection=slices[str(obj)]['selected_case_ids'][i],stored_strict_check=checks[i],
            judgment='NOT_REVIEWED',visually_reviewed=False))
    return dict(key=key,N=n,candidate_U=len(pairs),strict_U=int(accepted.sum()),added_U=int((~accepted).sum()),
        slices=slices,cases=cases),dict(strict_degree=sd,raw_degree=rd,strict_CC=sl,raw_CC=rl,**geo)


def fixed_comparison(out,packets,old):
    records,refs,arrays={}, {}, {};totals=Counter()
    with np.load(B/'extended_delaunay_edgewise.npz',allow_pickle=False) as b, np.load(C/'ede_exact_optimization.npz',allow_pickle=False) as cp:
        for key,p in packets.items():
            raw,a,t,received=execute('Raw',p)
            ref={f:b[key+'/EDE_'+f].copy() for f in ('candidates','origins','triangles','accepted')}
            offline_oracle(raw,ref)
            for f in ref:c.exact(ref[f],cp[key+'/optimized/audit/'+f],key+'/B_C_'+f)
            strict=io.read_packet(B/f'packets/{key}_EDE.npz')
            c.exact(strict,io.read_packet(C/f'packets/{key}_optimized.npz'),key+'/B_C_packet')
            io.preserved(p,strict)
            for f in p:
                if f not in io.MUTABLE:c.exact(p[f],raw[f],key+'/immutable/'+f)
            c.exact(strict['intra_object_edge_index'],directed_edges(ref['candidates'][ref['accepted']]),'strict ordered')
            c.exact(raw['intra_object_edge_index'],directed_edges(ref['candidates']),'raw ordered')
            checks=old['records'][key]['EDE']['checks']
            c.exact(io.serial(ref['candidates']),old['records'][key]['EDE']['candidates'],'B JSON candidates')
            c.exact(ref['origins'].tolist(),old['records'][key]['EDE']['origins'],'B JSON origins')
            c.exact(ref['accepted'].tolist(),old['records'][key]['EDE']['accepted'],'B JSON accepted')
            crec=json.loads((C/f'records/{key}_optimized.json').read_text())
            c.exact(checks,crec['audit']['checks'],'stored Strict checks B/C')
            # Black-box guard proves raw normal path needs no raster/validator/lookup.
            with patch('visual.runtime.extended_delaunay_edges.supercover',side_effect=AssertionError('pixel call')), \
                 patch('visual.runtime.extended_delaunay_edges._segment_supercover_single_sat',side_effect=AssertionError('SAT call')), \
                 patch('visual.runtime.extended_delaunay_edges.validate_cover',side_effect=AssertionError('validator call')), \
                 patch('visual.runtime.extended_delaunay_edges.prepare_validation',side_effect=AssertionError('lookup call')):
                guarded=execute('Raw',p);c.exact(raw,guarded[0],'raw guarded packet')
            record,metrics=analyze(key,p,a,ref['accepted'],checks)
            record['stored_strict_checks']=checks
            if p['source_frame']==0:
                remote=old['remote'][key]
                rows=np.asarray(remote['rows'],np.int64);pair=ref['candidates']
                incident=np.isin(pair,rows).any(1);new=incident&~ref['accepted']
                record['plateau']=dict(stored=remote,raw_added_incident_ids=np.flatnonzero(new),
                    strict_chain_U=remote['accepted_count'],raw_chain_U=int(incident.sum()),
                    merges_other_strict_CC=bool((metrics['strict_CC'][pair[new,0]]!=metrics['strict_CC'][pair[new,1]]).any()))
            io.save_packet(out/f'packets/{key}_raw.npz',raw)
            io.write(out/f'records/{key}.json',record);records[key]=record;refs[key]=(raw,strict,ref)
            fields=dict(candidates=a['candidates'],origins=a['origins'],accepted=ref['accepted'],
                raw_edges=raw['intra_object_edge_index'],strict_edges=strict['intra_object_edge_index'],
                object_id=p['object_id'],source=p['provenance']['source_index'],component=p['observed_component_id'],
                singleton_rows=np.concatenate([s['singleton_rows'] for s in record['slices'].values()]),**metrics)
            for f,v in fields.items():arrays[key+'/'+f]=v
            totals.update(candidate_U=record['candidate_U'],strict_U=record['strict_U'],added_U=record['added_U'],N=record['N'])
            for s in record['slices'].values():
                totals.update(strict_isolated=s['metrics']['Strict']['isolated'],raw_isolated=s['metrics']['Raw']['isolated'],
                    strict_CC=s['metrics']['Strict']['graph_CC'],raw_CC=s['metrics']['Raw']['graph_CC'],
                    singleton=len(s['singleton_rows']),newly_connected=len(s['newly_connected_rows']),
                    merge_groups=len(s['merges']))
    for k,v in dict(candidate_U=30143,strict_U=25046,added_U=5097,strict_isolated=302,singleton=270).items():c.exact(totals[k],v,'reference total/'+k)
    np.savez_compressed(out/'raw_ede_comparison.npz',**arrays)
    return records,refs,dict(totals)


def benchmark(out,packets,evidences,refs):
    rows=[]
    for ci,(key,p) in enumerate(packets.items()):
        evidence=evidences[p['source_frame']]
        for rep in range(40):
            order=('Strict','Raw') if (rep+ci)%2==0 else ('Raw','Strict')
            results={}
            for policy in order:results[policy]=execute(policy,p,evidence)
            # Assertions are outside execute's wall/process timers for both policies.
            for policy,result in results.items():
                expected=refs[key][0 if policy=='Raw' else 1]
                c.exact(result[0],expected,key+'/'+policy+'/benchmark packet')
                c.exact(result[1]['candidates'],refs[key][2]['candidates'],'benchmark candidate order')
                c.exact(result[1]['origins'],refs[key][2]['origins'],'benchmark origin order')
                if rep>=10:rows.append(dict(key=key,policy=policy,sample=rep-10,order=list(order),**result[2]))
        print('BENCH',key,flush=True)
    io.write(out/'timings_raw.json',rows)
    stages=('total_wall_ms','total_cpu_ms','common_point_component_preparation_ms','candidate_total_ms',
            'validation_lookup_ms','supercover_ms','support_validation_ms','depth_validation_ms',
            'edge_packaging_ms','metadata_packaging_ms','raw_packaging_ms','packet_packaging_ms','receiver_ms')
    def stats(selected):return {p:{s:io.stats([r.get(s,0.) for r in selected if r['policy']==p]) for s in stages} for p in ('Strict','Raw')}
    aggregate=stats(rows);conditions={}
    for key in packets:
        q=stats([r for r in rows if r['key']==key]);q['speedup']=q['Strict']['total_wall_ms']['mean']/q['Raw']['total_wall_ms']['mean'];conditions[key]=q
    return dict(aggregate=aggregate,conditions=conditions,
        aggregate_wall_speedup=aggregate['Strict']['total_wall_ms']['mean']/aggregate['Raw']['total_wall_ms']['mean'],
        aggregate_cpu_speedup=aggregate['Strict']['total_cpu_ms']['mean']/aggregate['Raw']['total_cpu_ms']['mean'],
        worst_sample={p:max([r for r in rows if r['policy']==p],key=lambda r:r['total_wall_ms']) for p in ('Strict','Raw')},
        total_measurements=len(rows),return_assertions=16*40*2)


def plots(out,summary,packets,refs,evidences):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image
    plt.rcParams.update({'font.size':8})
    with np.load(REPO/'visual/data/step2/ep000.npz',allow_pickle=False) as raw:
        rgb={f:raw['base_camera/rgb'][f].copy() for f in io.FRAMES}
        depths={f:raw['base_camera/depth'][f].copy() for f in io.FRAMES}
    with np.load(REPO/'visual/results/runtime_tracking/ep000/masks.npz',allow_pickle=False) as masks:
        idx={int(f):i for i,f in enumerate(masks['source_frame'])}
        mask={f:np.stack([masks[o][idx[f]] for o in ('peg','box')]) for f in io.FRAMES}
    coverage={};review_cases=[]
    for key,p in packets.items():
        r=summary['records'][key];fid=p['source_frame'];pairs=refs[key][2]['candidates'];accepted=refs[key][2]['accepted']
        source=p['provenance']['source_index'];w=int(p['image_shape'][1]);uv=np.column_stack((source%w,source//w));xyz=p['camera_points']
        depth=np.where(depths[fid]>0,depths[fid],np.nan)
        depthmax=float(np.nanmax(depth))
        # Only original observation dependencies; no GT arrays or path diagnostics.
        np.savez_compressed(out/f'inputs/{key}.npz',rgb=rgb[fid],raw_mask=mask[fid],depth=depth,camera_xyz=xyz,uv=uv)
        for obj in (0,1):
            rows=np.flatnonzero(p['object_id']==obj);ids=np.flatnonzero(p['object_id'][pairs[:,0]]==obj)
            strict=pairs[ids[accepted[ids]]];added=pairs[ids[~accepted[ids]]];rawpairs=pairs[ids]
            sid=key+'_o'+str(obj);names=[]
            # Four mandatory backgrounds plus near depth and near 3D views.
            fig=plt.figure(figsize=(15,23))
            for row,bg in enumerate(('RGB','Raw mask','Depth full (original range)','Depth near (0–1.2m)','Camera XYZ full','Camera XYZ near (z<1.2m)')):
                for col,policy in enumerate(('Strict','Raw','Difference')):
                    is3d=row>=4;ax=fig.add_subplot(6,3,row*3+col+1,projection='3d' if is3d else None)
                    ax.set_title(f'{sid} {policy} | {bg} | N={len(rows)} U={len(strict) if col==0 else len(rawpairs)}')
                    collections=[(strict,'#168bd2')] if col==0 else [(rawpairs,'#ef7924')] if col==1 else [(strict,'#168bd2'),(added,'#ef7924')]
                    if is3d:
                        for edges,color in collections:
                            if len(edges):ax.add_collection3d(Line3DCollection(xyz[edges],colors=color,linewidths=.55,alpha=.8))
                        ax.scatter(*xyz[rows].T,s=3,c='black')
                        viewrows=rows if row==4 else rows[xyz[rows,2]<1.2]
                        if len(viewrows):
                            lo=xyz[viewrows].min(0);hi=xyz[viewrows].max(0);span=np.maximum(hi-lo,.01);pad=span*.07
                            ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_zlim(lo[2]-pad[2],hi[2]+pad[2]);ax.set_box_aspect(span)
                        ax.view_init(22,-62);ax.set_xlabel('camera X m');ax.set_ylabel('camera Y m');ax.set_zlabel('camera Z m')
                    else:
                        if row==0:ax.imshow(rgb[fid])
                        elif row==1:ax.imshow(mask[fid][obj],cmap='gray',vmin=0,vmax=1)
                        else:
                            ax.imshow(np.ma.masked_invalid(depth),cmap='viridis',vmin=0,vmax=depthmax if row==2 else 1.2)
                        for edges,color in collections:ax.add_collection(LineCollection(uv[edges],colors=color,linewidths=.65,alpha=.85))
                        ax.scatter(*uv[rows].T,s=3,c='cyan',edgecolors='black',linewidths=.15)
                        if len(rows):
                            lo=uv[rows].min(0)-12;hi=uv[rows].max(0)+12
                            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
                        ax.set_xlabel('u pixels');ax.set_ylabel('v pixels');ax.set_aspect('equal')
            fig.suptitle('Blue = Strict retained; orange = Raw / added in Difference. Same points, axes and camera view.\nMask=raw tracker; full depth original range; near depth clipped 0–1.2m. Near 3D view explicitly clips distant plateau.',fontsize=11)
            fig.tight_layout(rect=(0,0,1,.965));path=f'plots/{sid}_comparison.png';fig.savefig(out/path,dpi=120);plt.close(fig);names.append(path)
            # Keep all scene context, including empty objects.
            fig,ax=plt.subplots(figsize=(6,6));ax.imshow(rgb[fid]);ax.scatter(*uv[rows].T,s=3,c='cyan');ax.set_title(sid+' full original RGB context');fig.tight_layout();path=f'plots/{sid}_context.png';fig.savefig(out/path,dpi=110);plt.close(fig);names.append(path)
            cases=[copy.deepcopy(x) for x in r['cases'] if x['object_id']==obj]
            for page in range(0,len(cases),4):
                chosen=cases[page:page+4];fig=plt.figure(figsize=(16,3.6*len(chosen)))
                for j,case in enumerate(chosen):
                    pair=np.asarray(case['rows']);xy=uv[pair];lo=xy.min(0)-5;hi=xy.max(0)+5
                    for col,bg in enumerate(('RGB','Raw mask','Depth full (original range)','Camera XYZ')):
                        ax=fig.add_subplot(len(chosen),4,j*4+col+1,projection='3d' if col==3 else None)
                        if col==3:
                            ax.scatter(*xyz[rows].T,s=2,c='gray',alpha=.3);ax.plot(*xyz[pair].T,c='#ef7924',marker='o');ax.view_init(22,-62);ax.set_xlabel('X m');ax.set_ylabel('Y m');ax.set_zlabel('Z m')
                        else:
                            ax.imshow(rgb[fid] if col==0 else mask[fid][obj] if col==1 else np.ma.masked_invalid(depth),cmap=None if col==0 else 'gray' if col==1 else 'viridis',vmin=None if col==0 else 0,vmax=None if col==0 else 1 if col==1 else depthmax)
                            ax.plot(*xy.T,c='#ef7924',marker='o',markersize=4,lw=1.5);ax.text(*xy[0],'A',color='red');ax.text(*xy[1],'B',color='red');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
                        ax.set_title(f"{case['candidate_id']} {bg}\n{','.join(case['stored_strict_check']['reasons'])}\nL={case['geometry']['camera_length_m']:.4f}m dz={case['geometry']['absolute_depth_difference_m']:.4f}m",fontsize=8)
                fig.tight_layout();path=f'plots/{sid}_cases_{page//4:02d}.png';fig.savefig(out/path,dpi=110);plt.close(fig);names.append(path)
                for case in chosen:case['plot']=path;review_cases.append(case)
            coverage[sid]=dict(condition=key,object_id=obj,N=len(rows),added_U=len(added),plots=names,overview_visually_reviewed=False)
        print('PLOTS',key,flush=True)
    # Per-condition graph summaries and pooled edge distributions.
    fig,axes=plt.subplots(2,2,figsize=(13,9));keys=list(summary['records']);x=np.arange(32)
    for ax,metric,title in [(axes[0,0],'isolated','Isolated nodes'),(axes[0,1],'graph_CC','Graph connected components')]:
        for policy,shift,color in [('Strict',-.18,'#168bd2'),('Raw',.18,'#ef7924')]:
            vals=[r['slices'][str(o)]['metrics'][policy][metric] for r in summary['records'].values() for o in (0,1)]
            ax.bar(x+shift,vals,.36,label=policy,color=color)
        ax.set_title(title);ax.legend();ax.set_xlabel('16 conditions × Peg/Box')
    with np.load(out/'raw_ede_comparison.npz',allow_pickle=False) as ar:
        for ax,field,title in [(axes[1,0],'camera_length_m','Camera length m (ECDF)'),(axes[1,1],'absolute_depth_difference_m','Endpoint |dz| m (ECDF)')]:
            for label,color in [('Strict','#168bd2'),('Raw','#ef7924'),('Added','#bb3355')]:
                vals=np.concatenate([ar[k+'/'+field][ar[k+'/accepted'] if label=='Strict' else ~ar[k+'/accepted'] if label=='Added' else np.ones(len(ar[k+'/accepted']),bool)] for k in keys]);vals=np.sort(vals)
                ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=label,color=color)
            ax.set_title(title);ax.set_xscale('symlog',linthresh=.001);ax.legend()
    fig.tight_layout();fig.savefig(out/'plots/connectivity_geometry.png',dpi=130);plt.close(fig)
    b=summary['benchmark'];fig,ax=plt.subplots(figsize=(13,5))
    for policy,shift,color in [('Strict',-.18,'#168bd2'),('Raw',.18,'#ef7924')]:
        ax.bar(np.arange(16)+shift,[b['conditions'][k][policy]['total_wall_ms']['mean'] for k in keys],.36,label=policy,color=color)
    ax.set_xticks(np.arange(16),keys,rotation=60,ha='right');ax.set_ylabel('CPU host wall ms');ax.set_title('Fresh candidates + policy work + packaging + actual receiver');ax.legend();fig.tight_layout();fig.savefig(out/'plots/cpu_latency.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    with np.load(out/'raw_ede_comparison.npz',allow_pickle=False) as ar:
        for policy,color in [('Strict','#168bd2'),('Raw','#ef7924')]:
            degree=np.concatenate([ar[k+'/'+policy.lower()+'_degree'] for k in keys]);values,counts=np.unique(degree,return_counts=True)
            axes[0].plot(values,counts,marker='.',label=policy,color=color)
            vals=np.concatenate([ar[k+'/uv_length_px'][ar[k+'/accepted'] if policy=='Strict' else np.ones(len(ar[k+'/accepted']),bool)] for k in keys]);vals=np.sort(vals)
            axes[1].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=policy,color=color)
        axes[0].set_title('Degree distribution (all Point rows)');axes[0].set_yscale('symlog');axes[0].legend()
        axes[1].set_title('UV length pixels (ECDF)');axes[1].set_xscale('symlog',linthresh=1);axes[1].legend()
    stage_names=['common_point_component_preparation_ms','candidate_total_ms','validation_lookup_ms','supercover_ms','support_validation_ms','depth_validation_ms','raw_packaging_ms','edge_packaging_ms','metadata_packaging_ms','packet_packaging_ms','receiver_ms']
    for j,policy in enumerate(('Strict','Raw')):
        bottom=0
        for name in stage_names:
            height=b['aggregate'][policy][name]['mean'];axes[2].bar(j,height,bottom=bottom,label=name if j==0 else None);bottom+=height
    axes[2].set_xticks([0,1],['Strict','Raw']);axes[2].set_ylabel('Wall ms');axes[2].set_title('Stages (instrumented, non-overlapping)');axes[2].legend(fontsize=5)
    fig.tight_layout();fig.savefig(out/'plots/degree_uv_stages.png',dpi=130);plt.close(fig)
    io.write(out/'coverage.json',dict(status='GENERATED',slices=coverage,mandatory_slices=32))
    io.write(out/'review.json',dict(status='NOT_EXECUTED',scope='Selected edge visual judgments are not physical truth labels',
        total_added_U=5097,reviewed_unique_U=0,cases=review_cases,findings=[],browser=dict(status='NOT_EXECUTED')))


def reports(out,s):
    review=json.loads((out/'review.json').read_text());coverage=json.loads((out/'coverage.json').read_text())
    t=s['totals'];b=s['benchmark'];agg=b['aggregate'];back=os.path.relpath(REPORT,out);forward=os.path.relpath(out/'raw_ede_comparison.html',REPORT.parent)
    a=agg['Strict']['total_wall_ms'];z=agg['Raw']['total_wall_ms']
    lines=['# Task05-E — Raw Extended Delaunay vs Strict EDE', '', f"Run UUID: `{out.name}`. [정적 HTML]({forward}).",'',
        '## 결과와 책임 경계','',
        f"고정 입력 16조건 / 32 object slice / {t['N']:,} Point-row. Candidate U={t['candidate_U']:,}, Strict U={t['strict_U']:,}, Raw U={t['candidate_U']:,}, 복원 U={t['added_U']:,}. Strict ⊆ Raw와 저장 거부 후보의 exact 차집합을 모든 조건에서 확인했다.",
        f"Strict isolated {t['strict_isolated']} → Raw {t['raw_isolated']}; source identity로 재확인한 singleton {t['singleton']}. 새 연결을 얻은 Point {t['newly_connected']}; Graph CC 합계 {t['strict_CC']} → {t['raw_CC']}. 조건 간 중복 관측을 포함한 합계이며 정확성 향상 지표가 아니다.",'',
        'Vision은 기존 candidate_edges(extended=True)를 독립 실행한다. Raw는 모든 후보를 그대로 사용하며 Supercover·Mask/Depth/overlap 경로·PairEvidence·depth-jump·reason/witness를 생성하지 않는다. Raw producer의 validator 금지 guard와 회귀 테스트를 통과했다.',
        'Raw Receiver는 v2 Point/Feature/좌표/provenance, row/source metadata, 범위·object/component·canonical pair·origin enum·directed 양방향 exact 순서·deep-copy·실제 robot 3행 공급 시 +3만 검증한다. Qhull·Delaunay·collinear chain을 재실행하지 않는다. 구조적으로 일관된 metadata를 위조한 임의 chord는 Receiver 단독으로 검출하지 못한다. 후보 누락·추가 chord·잘못된 chain/origin은 별도 offline oracle이 거부한다. 기존 Strict 및 다른 정책의 수신 계약과 production default는 유지했다.','',
        '## CPU 비용','',
        '표의 정책상 존재하지 않는 stage의 0은 실제 수행 시간이 아니라 수행하지 않은 작업을 뜻한다.',
        f"현재 host 동일 interpreter/thread에서 조건별 정책당 warmup 10회 + 30표본, AB/BA 교대. 총 측정 {b['total_measurements']}개. perf_counter wall / process_time CPU. IO·통계·assertion·그림은 측정 밖이다.",
        f"Aggregate wall 평균 Strict {a['mean']:.3f}ms → Raw {z['mean']:.3f}ms, {b['aggregate_wall_speedup']:.2f}×; CPU {b['aggregate_cpu_speedup']:.2f}×. 16조건의 30표본씩 동일 가중 pooling이다.",
        '정책이 실제로 제거한 작업량에 따른 비용 차이다. 동일 연산의 구현 최적화 speedup이 아니다. Strict는 Task05-C optimized single-normal SAT, fresh prepare_validation, reason/count/top-k witness까지 포함한다. Raw는 fresh 후보와 packaging만 필요하다. 양쪽 receiver의 검증 책임이 다르므로 receiver 시간은 별도 공개한다.',
        '공통 준비는 packet deepcopy와 저장 component 접근이다. upstream observed component 생성·sampling·PCA5·Feature·EE 변환은 재실행하지 않았으며 비용에 포함하지 않는다. 전체 Vision→Graph→GNN의 20Hz 실측이 아니다. 기존 GPU B1 평균 69.254ms는 다른 측정 범위의 참고값이고 이번 GPU 실행은 없다.','',
        '| Policy/stage ms | Mean | Median | p95 | Max |','|---|---:|---:|---:|---:|']
    for policy in ('Strict','Raw'):
        for stage,v in agg[policy].items():lines.append(f"| {policy} {stage} | {v['mean']:.3f} | {v['median']:.3f} | {v['p95']:.3f} | {v['max']:.3f} |")
    lines+=['','| Condition | N | Strict mean ms | Raw mean ms | speedup |','|---|---:|---:|---:|---:|']
    for key,r in s['records'].items():
        q=b['conditions'][key];lines.append(f"| {key} | {r['N']} | {q['Strict']['total_wall_ms']['mean']:.3f} | {q['Raw']['total_wall_ms']['mean']:.3f} | {q['speedup']:.2f}× |")
    maxkey=max(s['records'],key=lambda k:s['records'][k]['N']);lines+=['',f"최대 N 조건: {maxkey} (N={s['records'][maxkey]['N']}). 상세 p95/max와 최악 표본은 JSON benchmark.conditions / worst_sample에 보존했다.",'','## 연결성·시각 검토','',
        '모든 slice에 Strict/Raw/Difference의 동일 Point·축·camera view PNG, 원본 RGB 전체 context, raw mask, 전체 depth(원본 범위, frame별 최대값), 근거리 depth(0–1.2m clipping), 전체/근거리 Camera XYZ를 제공한다. 파랑=Strict 유지, 주황=Raw 추가. 빈 object도 보존한다.',
        '추가 Edge별 저장 Strict reason/count/witness는 records JSON에 보존했다. 사유 합계는 중복을 포함한다. Top5 UV/3D 길이/endpoint depth 차이 및 reason 조합 첫 사례는 확대 PNG와 review.json에 연결했다. 이는 선택 기준이며 필터가 아니다.',
        f"실제 확대 검토: {review['reviewed_unique_U']} / 전체 추가 {t['added_U']} Edge. 전체 panel 검토: {sum(x['overview_visually_reviewed'] for x in coverage['slices'].values())}/32. 선택 사례 비율을 전체 오류율로 일반화하지 않는다. 물리적 정답 라벨은 아니다.",'',
        '| Slice | N | Candidate U | Strict/Raw isolated | Strict/Raw CC | newly connected | Added U |','|---|---:|---:|---:|---:|---:|---:|']
    for key,r in s['records'].items():
        for obj,sl in r['slices'].items():
            st=sl['metrics']['Strict'];rw=sl['metrics']['Raw'];lines.append(f"| {key} o{obj} | {sl['N']} | {sl['candidate_U']} | {st['isolated']}/{rw['isolated']} | {st['graph_CC']}/{rw['graph_CC']} | {len(sl['newly_connected_rows'])} | {sl['metrics']['Added']['U']} |")
    reason=Counter();combos=Counter()
    for r in s['records'].values():
        for sl in r['slices'].values():reason.update(sl['reasons']);combos.update(sl['reason_combinations'])
    lines+=['','저장 Strict 사유별 복원 Edge 수(중복 가능): `'+json.dumps(dict(reason),ensure_ascii=False)+'`.','사유 조합별 unique 수: `'+json.dumps(dict(combos),ensure_ascii=False)+'`.','',
        'Frame0 Box 8.431m plateau는 기존 component 20의 12 Point, 11 chain Edge가 Strict/Raw 모두 유지된다. Raw added incident Edge 및 다른 Strict CC 병합 여부는 records.plateau에 별도 기록한다. 기존 오염 plateau를 Raw 신규 실패나 정상 복원으로 해석하지 않는다.','',
        '## 다섯 질문에 대한 관측상 답변','']
    lines+=review.get('findings',[]) or ['시각 검토 대기. 정성 결론 미확정.']
    lines+=['','## 검증·재현','',f"테스트: {s['tests']['count']}개 PASS. JSON/NPZ dtype·shape·bytes·order/NaN, packet 재수신, PNG decode, 통계 재계산, 상대 링크와 source 보호 hash를 verification.json에 기록한다. 실제 브라우저 검증: {review.get('browser',{}).get('status','NOT_EXECUTED')}.",
        'source snapshot / before snapshot / implementation.diff / source_hashes.json / input_gate.json / records / packets / timings_raw.json / review.json / coverage.json / tests.log를 동일 UUID에 보존했다. W&B disabled, ExperimentLogger(team=visual, job_type=evaluate) 사용. 외부 업로드·dependency 설치·commit 없음.',
        '시각화의 빈 Edge 3D 패널 오류를 수정한 report-only recovery는 같은 UUID와 저장 측정값을 유지한다. 원본 계산 source snapshot, report source snapshot/diff, 실패 metadata와 numerical 보호 hash를 보존한다.',
        'GPU·simulator·GNN/GRU/BC 학습·rollout: NOT_EXECUTED. 최종 정책 채택·GNN 통합은 별도 결정한다.','',
        '재현 명령: `WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -m visual.runtime.test.run_raw_ede_comparison`']
    REPORT.write_text('\n'.join(lines)+'\n')
    html=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task05-E Raw EDE</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px;background:#f7f8fa;color:#182536}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:5px}details{margin:15px 0}a{color:#0969aa}pre{white-space:pre-wrap}</style>',
        '<h1>Task05-E Raw EDE vs Strict EDE</h1>',f'<p>UUID {out.name} · <a href="{back}">Markdown report</a></p>',
        '<p>Blue: Strict retained. Orange: Raw added in Difference. Visual judgments are not physical truth labels.</p>',
        '<nav>'+''.join(f'<a href="#{key}">{key}</a> · ' for key in s['records'])+'</nav>',
        '<h2>Findings</h2>'+''.join('<p>'+escape(x)+'</p>' for x in review.get('findings',[])),
        '<h2>Connectivity / geometry / CPU</h2><img src="plots/connectivity_geometry.png"><img src="plots/cpu_latency.png"><img src="plots/degree_uv_stages.png">',
        '<details><summary>All stage / condition CPU statistics</summary><pre>'+escape(json.dumps(b,indent=2))+'</pre></details>',
        '<h2>All 16 conditions / 32 object slices</h2>']
    byid={x['id']:x for x in review['cases']}
    for key,r in s['records'].items():
        html.append(f'<section id="{key}"><h2>{key}</h2><p><a href="records/{key}.json">Record: all stored Strict reasons / graph statistics</a> · <a href="packets/{key}_raw.npz">Raw packet</a> · <a href="inputs/{key}.npz">Original RGB/masks/depth + coordinates</a></p>')
        for obj,sl in r['slices'].items():
            item=coverage['slices'][key+'_o'+obj];html.append('<h3>'+('Peg' if obj=='0' else 'Box')+f" N={sl['N']}; candidate U={sl['candidate_U']}; added U={sl['metrics']['Added']['U']}</h3>")
            for path in item['plots']:
                html.append(f'<details open><summary><a href="{path}">{path}</a></summary><img loading="lazy" src="{path}" alt="{path}"></details>')
            html.append('<table><tr><th>Candidate / row / source</th><th>Stored reasons</th><th>Visual judgment</th></tr>')
            for case in r['cases']:
                if str(case['object_id'])!=obj:continue
                v=byid[case['id']];html.append('<tr><td>'+escape(f"{case['id']} rows={case['rows']} source={case['source']}")+'</td><td>'+escape(','.join(case['stored_strict_check']['reasons']))+'</td><td>'+escape(v['judgment'] if v['visually_reviewed'] else 'NOT_REVIEWED')+' '+escape(v.get('note',''))+'</td></tr>')
            html.append('</table>')
        html.append('</section>')
    for name in ('raw_ede_comparison.json','raw_ede_comparison.npz','timings_raw.json','review.json','coverage.json','input_gate.json','verification.json','tests.log','source_hashes.json','implementation.diff'):
        html.append(f'<p><a href="{name}">{name}</a></p>')
    html.append('</html>');(out/'raw_ede_comparison.html').write_text('\n'.join(html))


def verify(out,s,initial):
    from PIL import Image
    arrays=np.load(out/'raw_ede_comparison.npz',allow_pickle=False);seen=Counter()
    review=json.loads((out/'review.json').read_text());cov=json.loads((out/'coverage.json').read_text())
    for key,r in s['records'].items():
        c.exact(io.serial(r),json.loads((out/f'records/{key}.json').read_text()),'record JSON')
        p=io.read_packet(out/f'packets/{key}_raw.npz');received=receive_vision_packet(p)
        c.exact(p['intra_object_edge_index'],arrays[key+'/raw_edges'],'packet NPZ')
        c.exact(p['raw_candidate_pairs'],arrays[key+'/candidates'],'pairs NPZ');offline_oracle(p)
        source=io.read_packet(io.BASE/f'packets/{key}.npz')
        for f in source:
            if f not in io.MUTABLE:c.exact(source[f],p[f],'readback immutable/'+f)
        pairs=arrays[key+'/candidates'];accepted=arrays[key+'/accepted'];geo=geometries(p,pairs)
        sd,sl=topology(len(p['points']),pairs[accepted]);rd,rl=topology(len(p['points']),pairs)
        for k,v in dict(strict_degree=sd,raw_degree=rd,strict_CC=sl,raw_CC=rl,**geo).items():c.exact(v,arrays[key+'/'+k],'recomputed '+k)
        for obj,ss in r['slices'].items():
            rows=np.flatnonzero(p['object_id']==int(obj));ids=np.flatnonzero(p['object_id'][pairs[:,0]]==int(obj));added=ids[~accepted[ids]]
            for policy,d,l,subset in [('Strict',sd,sl,ids[accepted[ids]]),('Raw',rd,rl,ids),('Added',rd,rl,added)]:
                st=ss['metrics'][policy];c.exact(st['U'],len(subset),'U recount')
                for field,values in geo.items():c.exact(io.stats(values[subset]),st['geometry'][field],'geometry stats')
                if policy!='Added':
                    c.exact(st['isolated'],int((d[rows]==0).sum()),'isolated');c.exact(st['graph_CC'],len(np.unique(l[rows])),'CC')
            seen.update(added_U=len(added),candidate_U=len(ids),strict_U=int(accepted[ids].sum()),singleton=len(ss['singleton_rows']),strict_isolated=int((sd[rows]==0).sum()),raw_isolated=int((rd[rows]==0).sum()))
    for k,v in seen.items():c.exact(s['totals'][k],v,'aggregate recount/'+k)
    timings=json.loads((out/'timings_raw.json').read_text())
    c.exact(len(timings),960,'timing count')
    for policy in ('Strict','Raw'):
        for stage,v in s['benchmark']['aggregate'][policy].items():c.exact(v,io.stats([r.get(stage,0.) for r in timings if r['policy']==policy]),'timing stats')
    for key in s['records']:
        for policy in ('Strict','Raw'):
            c.exact(sum(r['key']==key and r['policy']==policy for r in timings),30,'per-condition samples')
    pngs=list((out/'plots').glob('*.png'))
    for p in pngs:
        with Image.open(p) as im:im.verify()
    links=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for name,value in attrs:
                if name in ('href','src') and value and not value.startswith('#'):links.append((out,value))
    Links().feed((out/'raw_ede_comparison.html').read_text())
    for value in re.findall(r'\]\(([^)]+)\)',REPORT.read_text()):links.append((REPORT.parent,value))
    for parent,path in links:io.require(not path.startswith(('http:','https:','/')) and (parent/path).exists(),'relative link '+path)
    c.exact(len(cov['slices']),32,'coverage slices')
    for v in cov['slices'].values():
        for path in v['plots']:io.require((out/path).exists(),'missing panel')
    reviewed=[x for x in review['cases'] if x['visually_reviewed']]
    c.exact(review['reviewed_unique_U'],len({x['id'] for x in reviewed}),'review denominator')
    for x in reviewed:io.require(x['judgment'] in ('문제가 있어 보임','자연스러워 보임','판단 불가') and x.get('note'),'review evidence')
    hashes=json.loads((out/'source_hashes.json').read_text())
    for rel,sha in hashes['evaluated_sources'].items():
        c.exact(io.sha(out/'source_snapshot'/rel),sha,'snapshot')
        refreshed=hashes.get('report_source_changes',{}).get(rel)
        expected=refreshed['report_sha256'] if refreshed else sha
        c.exact(io.sha(REPO/rel),expected,'source current')
        if refreshed:c.exact(io.sha(out/'report_source_snapshot'/rel),expected,'report snapshot')
    return dict(status='PASS',exact_conditions=16,object_slices=32,tests=s['tests'],
        JSON_readback='PASS',NPZ_dtype_shape_bytes_order='PASS',packet_receiver='PASS',statistics_recalculation='PASS',
        PNG_decode='PASS',png_count=len(pngs),relative_links='PASS',links_checked=len(links),
        markdown_html_bidirectional='PASS',protected=protect(initial),reviewed_unique_U=len(reviewed),
        overview_reviewed=sum(x['overview_visually_reviewed'] for x in cov['slices'].values()),
        browser=review['browser'],simulator='NOT_EXECUTED',GPU='NOT_EXECUTED',training='NOT_EXECUTED',
        upstream_component_generation='NOT_EXECUTED',physical_accuracy='NOT_ESTABLISHED',production_default_changed=False)


def report_recovery(out, review_stdin):
    """Reuse measured bytes after a rendering failure; same shared run identity."""
    import ast
    from datetime import datetime, timezone
    from utils.experiment_artifacts import register_existing_output, save_manifest
    from utils.experiment_logger import write as shared_write
    initial=json.loads((out/'source_hashes.json').read_text())['protected_before'];protect(initial)
    immutable=[out/'timings_raw.json',out/'raw_ede_comparison.npz',out/'tests.log',out/'input_gate.json',
               *sorted((out/'records').glob('*.json')),*sorted((out/'packets').glob('*.npz'))]
    before={str(p.relative_to(out)):io.sha(p) for p in immutable}
    io.write(out/'numerical_protection.json',dict(before=before))
    if not (out/'failure.json').exists():
        io.write(out/'failure.json',dict(metadata=json.loads((out/'metadata.json').read_text()),
            phase='3D plot of empty object; numerical evaluation completed',
            error='ValueError: need at least one array to concatenate',
            measured_timings_preserved=True))
    rel='visual/runtime/test/run_raw_ede_comparison.py';snapshot=out/'source_snapshot'/rel
    oldtree=ast.parse(snapshot.read_text());newtree=ast.parse((REPO/rel).read_text())
    functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    oldf,newf=functions(oldtree),functions(newtree)
    for name in oldf:
        if name not in ('plots','reports','verify','main','source_save'):c.exact(oldf[name],newf[name],'numerical source AST/'+name)
    h=json.loads((out/'source_hashes.json').read_text())
    for name,sha in h['evaluated_sources'].items():
        if name!=rel:c.exact(io.sha(REPO/name),sha,'unchanged evaluated source')
    dest=out/'report_source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dest)
    h['report_source_changes']={rel:dict(evaluated_sha256=io.sha(snapshot),report_sha256=io.sha(dest),
        unchanged_numerical_functions=[name for name in oldf if name not in ('plots','reports','verify','main','source_save')])}
    io.write(out/'source_hashes.json',h)
    (out/'report_recovery.diff').write_text(''.join(difflib.unified_diff(snapshot.read_text().splitlines(True),dest.read_text().splitlines(True),fromfile='evaluated/'+rel,tofile='report/'+rel)))
    packets,evidences,old=c.inputs();records={};refs={};totals=Counter()
    arrays=np.load(out/'raw_ede_comparison.npz',allow_pickle=False)
    for key,p in packets.items():
        record=json.loads((out/f'records/{key}.json').read_text());records[key]=record
        refs[key]=(io.read_packet(out/f'packets/{key}_raw.npz'),io.read_packet(B/f'packets/{key}_EDE.npz'),
            {f:arrays[key+'/'+f] for f in ('candidates','origins','accepted')})
        totals.update(candidate_U=record['candidate_U'],strict_U=record['strict_U'],added_U=record['added_U'],N=record['N'])
        for ss in record['slices'].values():totals.update(strict_isolated=ss['metrics']['Strict']['isolated'],raw_isolated=ss['metrics']['Raw']['isolated'],
            strict_CC=ss['metrics']['Strict']['graph_CC'],raw_CC=ss['metrics']['Raw']['graph_CC'],singleton=len(ss['singleton_rows']),newly_connected=len(ss['newly_connected_rows']),merge_groups=len(ss['merges']))
    rows=json.loads((out/'timings_raw.json').read_text())
    stages=('total_wall_ms','total_cpu_ms','common_point_component_preparation_ms','candidate_total_ms',
            'validation_lookup_ms','supercover_ms','support_validation_ms','depth_validation_ms',
            'edge_packaging_ms','metadata_packaging_ms','raw_packaging_ms','packet_packaging_ms','receiver_ms')
    def stats(selected):return {p:{stage:io.stats([r.get(stage,0.) for r in selected if r['policy']==p]) for stage in stages} for p in ('Strict','Raw')}
    agg=stats(rows);conditions={}
    for key in packets:
        q=stats([r for r in rows if r['key']==key]);q['speedup']=q['Strict']['total_wall_ms']['mean']/q['Raw']['total_wall_ms']['mean'];conditions[key]=q
    bench=dict(aggregate=agg,conditions=conditions,aggregate_wall_speedup=agg['Strict']['total_wall_ms']['mean']/agg['Raw']['total_wall_ms']['mean'],
        aggregate_cpu_speedup=agg['Strict']['total_cpu_ms']['mean']/agg['Raw']['total_cpu_ms']['mean'],
        worst_sample={p:max([r for r in rows if r['policy']==p],key=lambda r:r['total_wall_ms']) for p in ('Strict','Raw')},
        total_measurements=len(rows),return_assertions=1280)
    summary=dict(run_id=out.name,config=json.loads((out/'config.yaml').read_text()),environment=io.environment(),
        tests=dict(status='PASS',count=int(re.search(r'Ran (\d+) tests',(out/'tests.log').read_text()).group(1))),
        totals=dict(totals),input_gate=json.loads((out/'input_gate.json').read_text()),records=records,benchmark=bench,
        report_recovery=dict(new_uuid=False,new_benchmark=False,new_graphs=False,numerical_hashes=before))
    try:
        from threadpoolctl import threadpool_info
        summary['native_threadpools']=threadpool_info()
    except ImportError:summary['native_threadpools']='NOT_EXECUTED'
    io.write(out/'raw_ede_comparison.json',summary)
    if not (out/'review.json').exists() or json.loads((out/'review.json').read_text()).get('status')!='EXECUTED_SUBSET':
        plots(out,summary,packets,refs,evidences)
    io.write(out/'verification.json',dict(status='PENDING'));reports(out,summary)
    if review_stdin:print('REVIEW_READY',str(out),flush=True);input()
    summary['verification']=verify(out,summary,initial)
    io.write(out/'verification.json',summary['verification']);io.write(out/'raw_ede_comparison.json',summary);reports(out,summary)
    c.exact(io.serial(summary),json.loads((out/'raw_ede_comparison.json').read_text()),'summary JSON')
    verify(out,summary,initial)
    after={str(p.relative_to(out)):io.sha(p) for p in immutable};c.exact(before,after,'report recovery numerical protection')
    io.write(out/'numerical_protection.json',dict(status='PASS',before=before,after=after))
    for path in list(out.rglob('*'))+[REPORT]:
        if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','metrics.json','config.yaml'):
            register_existing_output(out,path,kind='trajectory' if path.suffix=='.npz' else 'metrics' if path.suffix=='.json' else 'reports')
    metadata=json.loads((out/'metadata.json').read_text())
    metadata['initial_report_failure']=json.loads((out/'failure.json').read_text())['metadata']
    metadata.update(status='finished',finished_at=datetime.now(timezone.utc).isoformat(),report_recovery_completed=True,
        source_runs=initial['source_runs'],verification=summary['verification'],production_default_changed=False)
    metadata.pop('exception_type',None);shared_write(out/'metadata.json',metadata)
    shared_write(out/'metrics.json',[dict(run_id=out.name,step=0,metrics=dict(candidate_U=30143,added_U=5097,wall_speedup=bench['aggregate_wall_speedup']))])
    save_manifest(out,metadata);protect(initial);print('DONE',out.name,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--review-stdin',action='store_true');parser.add_argument('--reports-only',type=Path);args=parser.parse_args()
    if args.reports_only:return report_recovery(args.reports_only.resolve(),args.review_stdin)
    for key in THREADS:io.require(os.getenv(key)=='1','launch with '+key+'=1')
    io.require(os.getenv('WANDB_MODE')=='disabled','WANDB_MODE must be disabled')
    initial=capture_initial();protect(initial)
    config=dict(task='05-E',source_runs=initial['source_runs'],frames=io.FRAMES,ratios=io.RATIOS,phase=0,
        policies=['Strict optimized single-normal SAT','Raw all EDE candidates'],warmups=10,samples=30,
        order='AB/BA alternating; condition start alternates',thread_env={k:os.getenv(k) for k in THREADS},
        boundary='packet copy/stored component + fresh candidate + policy preparation/validation + packaging + actual receiver',
        excluded=['IO','upstream observed component generation','sampling/PCA5/features/EE extraction','assertions','plots'],
        raw_receiver_candidate_regeneration=False)
    with ExperimentLogger(team='visual',job_type='evaluate',model='task05e_raw_ede_comparison',config=config,seed=0) as logger:
        out=logger.directory;Path('/tmp/task05e_run_id').write_text(out.name);print('RUN_ID',out.name,flush=True)
        for d in ('records','packets','plots','inputs'):(out/d).mkdir()
        source_save(out,initial)
        packets,evidences,old,gate=input_gate(out)
        test=subprocess.run([sys.executable,'-m','unittest','visual.runtime.test.test_raw_ede_comparison',*io.TESTS,
            'tests.test_experiment_logger','tests.test_experiment_artifacts','-v'],cwd=REPO,capture_output=True,text=True)
        (out/'tests.log').write_text(test.stdout+test.stderr);io.require(test.returncode==0,'tests failed')
        tests=dict(status='PASS',count=int(re.search(r'Ran (\d+) tests',test.stdout+test.stderr).group(1)))
        records,refs,totals=fixed_comparison(out,packets,old);print('EXACT PASS',totals,flush=True)
        bench=benchmark(out,packets,evidences,refs)
        summary=dict(run_id=out.name,config=config,environment=io.environment(),tests=tests,totals=totals,
            input_gate=gate,records=records,benchmark=bench)
        try:
            from threadpoolctl import threadpool_info
            summary['native_threadpools']=threadpool_info()
        except ImportError:summary['native_threadpools']='NOT_EXECUTED; dependency not installed'
        io.write(out/'raw_ede_comparison.json',summary)
        plots(out,summary,packets,refs,evidences)
        io.write(out/'verification.json',dict(status='PENDING'))
        io.write(out/'raw_ede_comparison.json',summary);reports(out,summary)
        if args.review_stdin:
            print('REVIEW_READY',str(out),flush=True);input()
        summary['verification']=verify(out,summary,initial)
        io.write(out/'verification.json',summary['verification']);io.write(out/'raw_ede_comparison.json',summary);reports(out,summary)
        c.exact(io.serial(summary),json.loads((out/'raw_ede_comparison.json').read_text()),'full summary JSON')
        # Verify final regenerated links too.
        verify(out,summary,initial)
        logger.record(source_runs=initial['source_runs'],exact_conditions=16,production_default_changed=False,
                      physical_accuracy='NOT_ESTABLISHED',verification=summary['verification'])
        logger.log(dict(candidate_U=totals['candidate_U'],added_U=totals['added_U'],wall_speedup=bench['aggregate_wall_speedup']))
        for path in list(out.rglob('*'))+[REPORT]:
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','metrics.json','config.yaml'):
                logger.link_output(path,kind='trajectory' if path.suffix=='.npz' else 'metrics' if path.suffix=='.json' else 'reports')
        protect(initial);print('DONE',out.name,flush=True)


if __name__=='__main__':main()
