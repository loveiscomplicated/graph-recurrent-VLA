"""Task05-C evaluation only: isolated snapshot, exact gates and separate profiling.

Kept outside the Task05-B runner so its original default/report path stays intact.
No sampler, feature estimator, simulator, training or publishing is invoked.
"""
import ast
from collections import defaultdict
import copy
import cProfile
import difflib
import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
from time import perf_counter, process_time
import tracemalloc
import types

import numpy as np

from visual.runtime.test import run_extended_delaunay_edges as io
from visual.runtime.test.run_triangle_intra_object_edges import rss

REPO = io.REPO
SOURCE = REPO/'experiments/9f36f7c89dbd4097a8be9358138d1537'
REPORT = REPO/'docs/visual-graph-integration/05c_ede_exact_optimization.md'
EDITABLE = ('visual/runtime/extended_delaunay_edges.py',
            'visual/runtime/test/test_extended_delaunay_edges.py',
            'visual/runtime/test/run_extended_delaunay_edges.py',
            'visual/runtime/test/ede_exact_optimization_eval.py',
            'docs/visual-graph-integration/05c_ede_exact_optimization.md')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def exact(a, b, path='root'):
    """Field assertions including dtype/shape/bytes/order; no sorting/tolerance."""
    if isinstance(a, np.ndarray):
        io.require(isinstance(b, np.ndarray), path+' array type')
        io.require(a.dtype == b.dtype and a.shape == b.shape, path+' dtype/shape')
        io.require(a.tobytes() == b.tobytes(), path+' bytes/order')
    elif isinstance(a, dict):
        io.require(isinstance(b, dict) and a.keys() == b.keys(), path+' keys')
        for k in a:exact(a[k], b[k], path+'/'+str(k))
    elif isinstance(a, (tuple, list)):
        io.require(type(a) is type(b) and len(a) == len(b), path+' sequence')
        for i, (x, y) in enumerate(zip(a, b)):exact(x, y, path+'/'+str(i))
    elif isinstance(a, (float, np.floating)) and np.isnan(a):
        io.require(np.isnan(b), path+' NaN')
    else:io.require(a == b, path+' value')


def semantic(audit):
    # Timing diagnostics alone are excluded; raster_work is retained.
    return {k:v for k,v in audit.items() if k not in
            ('timing_ms', 'candidate_object_ms', 'validation_object_ms')}


def load_module(name, relative, remap):
    path = SOURCE/'source_snapshot'/relative
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in remap:
            node.module = remap[node.module]
    module = types.ModuleType(name); module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def snapshot():
    hashes = json.loads((SOURCE/'source_hashes.json').read_text())
    checked = {}
    for rel, expected in hashes['evaluated_sources'].items():
        snap = SOURCE/'source_snapshot'/rel
        io.require(io.sha(snap) == expected, 'snapshot hash '+rel)
        if rel not in EDITABLE and not rel.endswith('00_interface.md'):
            io.require(io.sha(REPO/rel) == expected, 'shared dependency changed '+rel)
        checked[rel] = dict(snapshot_sha256=expected, current_sha256=io.sha(REPO/rel))
    names = {'visual.runtime.triangle_intra_object_edges':'_task05c_snapshot_triangle',
             'visual.runtime.extended_delaunay_edges':'_task05c_snapshot_ede',
             'graph.vision_input':'_task05c_snapshot_receiver'}
    tri = load_module(names['visual.runtime.triangle_intra_object_edges'],
                      'visual/runtime/triangle_intra_object_edges.py', names)
    ede = load_module(names['visual.runtime.extended_delaunay_edges'],
                      'visual/runtime/extended_delaunay_edges.py', names)
    receiver = load_module(names['graph.vision_input'], 'graph/vision_input.py', names)
    runner = load_module('_task05c_snapshot_runner',
                         'visual/runtime/test/run_extended_delaunay_edges.py', names)
    io.require(ede.supercover is tri.supercover and ede.validate_cover is tri.validate_cover,
               'snapshot raster/validator binding')
    io.require(runner.build_extended_delaunay_edges is ede.build_extended_delaunay_edges and
               runner.receive_vision_packet is receiver.receive_vision_packet,
               'snapshot execution/receiver binding')
    bindings = {}
    for key, fn in dict(execute=runner.execute, build=ede.build_extended_delaunay_edges,
                        raster=ede.supercover, validator=ede.validate_cover,
                        receiver=runner.receive_vision_packet).items():
        path = Path(fn.__code__.co_filename)
        io.require(path.is_relative_to(SOURCE/'source_snapshot'), 'foreign binding '+key)
        bindings[key] = dict(module=fn.__module__, path=str(path.relative_to(REPO)), sha256=io.sha(path))
    return runner, ede, tri, receiver, dict(sources=checked, bindings=bindings)


def inputs():
    old = json.loads((SOURCE/'extended_delaunay_edgewise.json').read_text())
    io.require(old['run_id'] == SOURCE.name, 'Task05-B identity')
    for rel, expected in old['input_hashes'].items():
        io.require(io.sha(REPO/rel) == expected, 'protected input hash '+rel)
    freeze = json.loads((io.SELECTION/'selection_before_gt.json').read_text())
    io.require(io.sha(io.SELECTION/'selection_before_gt.npz') == freeze['sha256'], 'selection freeze')
    packets, evidences = {}, {}
    for fid in io.FRAMES:
        evidences[fid] = io.load_evidence(io.BASE/f'runtime_evidence/f{fid:04d}.npz')
        for ratio in io.RATIOS:
            key = f'f{fid:04d}_r{ratio:g}'
            p = io.read_packet(io.BASE/f'packets/{key}.npz')
            io.require(p['source_frame'] == fid, 'frame identity')
            io.preserved(p, io.read_packet(SOURCE/f'packets/{key}_EDE.npz'))
            packets[key] = p
    io.require(len(packets) == 16, '16 fixed conditions')
    return packets, evidences, old


def saved_gate(key, p, audit, old):
    saved = io.read_packet(SOURCE/f'packets/{key}_EDE.npz')
    exact(saved, p, key+'/saved_packet')
    record = old['records'][key]
    for field in ('candidates','origins','accepted','checks'):
        exact(io.serial(audit[field]), record['EDE'][field], key+'/'+field)
    for a, b in zip(audit['components'], record['component_diagnostics']['components']):
        exact(io.serial(a), {k:b[k] for k in a}, key+'/component')
    with np.load(SOURCE/'extended_delaunay_edgewise.npz', allow_pickle=False) as ar:
        exact(p['intra_object_edge_index'], ar[key+'/EDE_edges'], key+'/saved_npz_edges')
        for field in ('candidates','origins','accepted','triangles'):
            exact(audit[field],ar[key+'/EDE_'+field],key+'/saved_npz_'+field)
        for name, value in io.flatten_arrays(io.read_packet(io.BASE/f'packets/{key}.npz'), prefix=key+'/original').items():
            exact(value, ar[name], name)
    return dict(status='PASS', packet_hash=io.packet_digest(p), audit_hash=io.packet_digest(semantic(audit)))


def baseline_gate():
    runner, ede, tri, receiver, binding = snapshot()
    packets, evidences, old = inputs()
    references, gates, common = {}, {}, {}
    for key,p in packets.items():
        e = evidences[p['source_frame']]
        tick = perf_counter(); maps = io.lookup_maps(p); map_ms=(perf_counter()-tick)*1000
        tick = perf_counter(); ctx=tri.prepare_validation(e); ctx_ms=(perf_counter()-tick)*1000
        result=runner.execute('EDE', p,e,maps,ctx)
        gates[key]=saved_gate(key,*result[:2],old)
        references[key]=result
        common[key]=dict(maps=maps,contexts=ctx,map_ms=map_ms,prepare_ms=ctx_ms)
        print('baseline exact',key,flush=True)
    return runner,ede,tri,receiver,binding,packets,evidences,old,references,gates,common


def raster_profile(p, total):
    """Identical original SAT operations with stage timers; never benchmark core."""
    tick=perf_counter(); p=np.asarray(p)
    if p.shape not in ((2,2),(3,2)) or p.dtype.kind not in 'iu':raise ValueError('integer UV segment or triangle required')
    p=p.astype(np.int64);lo,hi=p.min(0),p.max(0)
    v,u=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
    total['bbox_array_allocation_ms']+=(perf_counter()-tick)*1000
    tick=perf_counter();centres=np.column_stack((u.ravel(),v.ravel()))
    total['integer_centres_ms']+=(perf_counter()-tick)*1000
    tick=perf_counter();keep=np.ones(len(centres),bool);vertices=2*p;directions=np.roll(p,-1,axis=0)-p
    total['keep_vertices_roll_ms']+=(perf_counter()-tick)*1000
    for i,d in enumerate(directions):
        tick=perf_counter();axis=np.array([-d[1],d[0]],np.int64)
        if axis.any():
            projection=vertices@axis;c=2*(centres@axis);radius=np.abs(axis).sum()
            keep &= (c+radius>=projection.min()) & (c-radius<=projection.max())
        total[f'SAT_normal_{i+1}_ms']+=(perf_counter()-tick)*1000
    tick=perf_counter();result=centres[keep]
    total['membership_row_major_output_ms']+=(perf_counter()-tick)*1000
    total['calls']+=1;total['bbox_cells']+=len(centres);total['cover_cells']+=len(result)
    # Original logical allocations, not allocator/native churn.
    total['bbox_peak_cells']=max(total['bbox_peak_cells'],len(centres))
    return result


def line_profile(call, functions):
    """Separate intrusive line pass; child time is inclusive (never sum parents)."""
    codes={f.__code__:f.__name__ for f in functions}; totals=defaultdict(float); states={}
    def trace(frame,event,arg):
        if frame.f_code not in codes:return trace
        now=perf_counter(); k=id(frame)
        if event in ('line','return'):
            if k in states:
                line,tick=states[k];totals[(codes[frame.f_code],line)]+=now-tick
            if event=='line':states[k]=(frame.f_lineno,now)
            else:states.pop(k,None)
        return trace
    tick=perf_counter();sys.settrace(trace)
    try:call()
    finally:sys.settrace(None)
    wall=(perf_counter()-tick)*1000
    sources={f.__name__:(Path(f.__code__.co_filename).read_text().splitlines(),f.__code__.co_filename) for f in functions}
    return dict(wall_ms=wall, attribution='inclusive child time; intrusive tracer overhead; do not sum nested functions',
        lines=[dict(function=n,line=l,wall_ms=t*1000,source=sources[n][0][l-1],path=sources[n][1]) for (n,l),t in totals.items()])


def profiling(state):
    runner,ede,tri,receiver,_,packets,evidences,_,refs,_,common=state
    output={}; aggregate=defaultdict(float)
    for key,p in packets.items():
        e=evidences[p['source_frame']];c=common[key];t=defaultdict(float)
        def profiled(uv):return raster_profile(uv,t)
        original=ede.supercover;ede.supercover=profiled
        tick=perf_counter()
        try:r=runner.execute('EDE',p,e,c['maps'],c['contexts'])
        finally:ede.supercover=original
        wall=(perf_counter()-tick)*1000
        exact(semantic(refs[key][1]),semantic(r[1]));exact(refs[key][0],r[0])
        total=sum(v for k,v in t.items() if k.endswith('_ms'))
        output[key]=dict(raster=dict(t),instrumented_core_ms=wall,
                         uninstrumented_gate_core_ms=refs[key][2]['core_wall_ms'],
                         stage_ms=r[2],raster_timed_substages_sum_ms=total)
        for k,v in t.items():aggregate[k]+=v
    def all_calls():
        for key,p in packets.items():
            c=common[key];runner.execute('EDE',p,evidences[p['source_frame']],c['maps'],c['contexts'])
    cp=cProfile.Profile();tick=perf_counter();cp.enable();all_calls();cp.disable()
    cpwall=(perf_counter()-tick)*1000;ps=pstats.Stats(cp)
    functions=[dict(file=f,line=l,function=n,primitive_calls=v[0],calls=v[1],self_s=v[2],cumulative_s=v[3])
               for (f,l,n),v in ps.stats.items()]
    lines=line_profile(all_calls,[ede.candidate_edges,ede.build_extended_delaunay_edges,
                tri.adjacent_pairs,tri.validate_cover,receiver.receive_vision_packet])
    reference_total=sum(r[2]['core_wall_ms'] for r in refs.values())
    axis2=aggregate['SAT_normal_2_ms'];raster=sum(v for k,v in aggregate.items() if k.endswith('_ms'))
    # Gate: duplicate is at least 5% of raster and 1% of original core. No input/repeat changes.
    passed=axis2/raster>=.05 and axis2/reference_total>=.01
    return dict(status='PASS' if passed else 'STOP_REPLAN',conditions=output,raster_aggregate=dict(aggregate),
        duplicate_axis_share_raster=axis2/raster,duplicate_axis_share_reference_core=axis2/reference_total,
        gate='duplicate axis >=5% raster and >=1% reference core; descriptive instrumented evidence only',
        mathematical_equivalence='For a segment d and -d yield a and -a. P[-a]=-P[a], c[-a]=-c[a], r[-a]=r[a]. The two inclusive inequalities exchange under negation. Safe integer arithmetic is required.',
        overhead=dict(standalone_baseline_core_ms=reference_total,cprofile_wall_ms=cpwall,
                      cprofile_ratio=cpwall/reference_total,line_trace_ratio=lines['wall_ms']/reference_total,
                      raster_timer_overhead_ms=sum(v['instrumented_core_ms'] for v in output.values())-reference_total),
        cprofile=functions,line_profile=lines,
        boundaries='cProfile and line profile separate from stage-instrumented pass; no nested totals added; all outside timed benchmark')


def gate_main():
    state=baseline_gate();profile=profiling(state)
    io.write('/tmp/task05c_gate.json',dict(binding=state[4],baseline=state[9],profile=profile))
    print('PROFILE GATE',profile['status'],profile['duplicate_axis_share_raster'],profile['duplicate_axis_share_reference_core'],flush=True)




def optimized_runner():
    import functools
    from visual.runtime import extended_delaunay_edges as current
    names={'visual.runtime.triangle_intra_object_edges':'_task05c_snapshot_triangle',
           'graph.vision_input':'_task05c_snapshot_receiver'}
    runner=load_module('_task05c_optimized_runner','visual/runtime/test/run_extended_delaunay_edges.py',names)
    runner.build_extended_delaunay_edges=functools.partial(current.build_extended_delaunay_edges,optimized=True)
    io.require(runner.build_extended_delaunay_edges.func is current.build_extended_delaunay_edges,
               'optimized binding')
    io.require(runner.receive_vision_packet.__module__=='_task05c_snapshot_receiver','same receiver')
    return runner,current


def detailed(packet,audit,raster,tri,contexts):
    """Materialize full ordered cover/pairs and PairEvidence state outside core."""
    src=packet['provenance']['source_index'];w=int(packet['image_shape'][1])
    uv=np.column_stack((src%w,src//w));covers=[];pairs=[];offsets=[0];poff=[0]
    indices=[];supports=[];deltas=[];jumps=[];rights=[]
    for pair in audit['candidates']:
        cover=raster(uv[pair]);ps=tri.adjacent_pairs(cover,packet['image_shape'])
        c=contexts[int(packet['object_id'][pair[0]])]
        support=c['support'].ravel()[ps].all(1)
        right=((ps[:,1]-ps[:,0])==1)&(ps[:,1]//w==ps[:,0]//w)
        ix=np.where(right,c['right'].ravel()[ps[:,0]],c['down'].ravel()[ps[:,0]])
        delta=np.full(len(ps),np.nan,c['pairs'].signed_depth_delta_m.dtype)
        jump=np.zeros(len(ps),bool);present=ix>=0
        delta[present]=c['pairs'].signed_depth_delta_m[ix[present]]
        jump[present]=c['pairs'].depth_jump[ix[present]]
        covers.append(cover);pairs.append(ps);offsets.append(offsets[-1]+len(cover));poff.append(poff[-1]+len(ps))
        indices.append(ix);supports.append(support);deltas.append(delta);jumps.append(jump);rights.append(right)
    def cat(xs,shape,dtype):return np.concatenate(xs) if xs else np.empty(shape,dtype)
    return dict(cover_uv=cat(covers,(0,2),np.int64),cover_offsets=np.array(offsets,np.int64),
        pairs=cat(pairs,(0,2),np.int64),pair_offsets=np.array(poff,np.int64),
        pair_evidence_index=cat(indices,(0,),np.int64),pair_supported=cat(supports,(0,),bool),
        pair_delta=cat(deltas,(0,),np.float32),pair_jump=cat(jumps,(0,),bool),
        pair_is_right=cat(rights,(0,),bool))


def real_exact(state,out):
    runner,ede,tri,receiver,binding,packets,evidences,old,refs,gates,common=state
    opt,current=optimized_runner();arrays={};records={};details={}
    for key,p in packets.items():
        c=common[key];e=evidences[p['source_frame']]
        a=refs[key];b=opt.execute('EDE',p,e,c['maps'],c['contexts'])
        exact(a[0],b[0],key+'/packet');exact(semantic(a[1]),semantic(b[1]),key+'/audit')
        saved_gate(key,*b[:2],old)
        tick=perf_counter();da=detailed(p,a[1],ede.supercover,tri,c['contexts']);ad_ms=(perf_counter()-tick)*1000
        tick=perf_counter();db=detailed(p,b[1],current._segment_supercover_single_sat,tri,c['contexts']);bd_ms=(perf_counter()-tick)*1000
        exact(da,db,key+'/all_cover_pairs_evidence')
        ra=receiver.receive_vision_packet(a[0],robot_positions=np.zeros((3,3),np.float32))
        rb=receiver.receive_vision_packet(b[0],robot_positions=np.zeros((3,3),np.float32))
        for field in ra.__dataclass_fields__:exact(getattr(ra,field),getattr(rb,field),key+'/receiver/'+field)
        exact(ra.edge_index,a[0]['intra_object_edge_index']+3,key+'/robot_offset')
        ra.packet['points'][:]=0;exact(a[0],b[0],key+'/deepcopy')
        details[key]=da
        r=dict(N=len(p['points']),U=len(a[1]['candidates']),A=int(a[1]['accepted'].sum()),
            E=a[0]['intra_object_edge_index'].shape[1],exact='PASS',saved_vs_baseline='PASS',
            saved_vs_optimized='PASS',baseline_vs_optimized='PASS',
            input_hash=io.packet_digest(p),output_hash=io.packet_digest(a[0]),
            semantic_audit_hash=io.packet_digest(semantic(a[1])),detailed_hash=io.packet_digest(da),
            detailed_audit_ms=dict(baseline=ad_ms,optimized=bd_ms),
            detailed_array_bytes=io.array_bytes(da),
            pair_occurrences=len(da['pairs']),unique_observation_pairs=len(np.unique(da['pairs'],axis=0)))
        for variant,res,detail in (('baseline',a,da),('optimized',b,db)):
            io.save_packet(out/f'packets/{key}_{variant}.npz',res[0])
            exact(res[0],io.read_packet(out/f'packets/{key}_{variant}.npz'),key+'/packet_readback')
            prefix=key+'/'+variant
            arrays.update(io.flatten_arrays(semantic(res[1]),prefix=prefix+'/audit'))
            arrays.update({prefix+'/'+k:v for k,v in detail.items()})
            arrays[prefix+'/rejected']=res[1]['candidates'][~res[1]['accepted']].copy()
            arrays[prefix+'/accepted']=res[1]['candidates'][res[1]['accepted']].copy()
            arrays[prefix+'/edge_index']=res[0]['intra_object_edge_index'].copy()
            io.write(out/f'records/{key}_{variant}.json',dict(summary=r,audit=semantic(res[1]),
                receiver_packet_hash=io.packet_digest(res[0]),receiver_robot_edges=rb.edge_index,
                diagnostics=dict(timing_ms=res[2]),detailed_hash=io.packet_digest(detail)))
        arrays.update(io.flatten_arrays(p,prefix=key+'/original'))
        records[key]=r
        print('three-way exact',key,flush=True)
    tick=perf_counter();np.savez_compressed(out/'ede_exact_optimization.npz',**arrays);save_ms=(perf_counter()-tick)*1000
    with np.load(out/'ede_exact_optimization.npz',allow_pickle=False) as ar:
        io.require(set(ar.files)==set(arrays),'NPZ keys')
        for key,value in arrays.items():exact(value,ar[key],key+'/aggregate_readback')
    return opt,current,records,details,dict(npz_arrays=len(arrays),npz_save_ms=save_ms,
                                           ragged_offsets='exact for both paths; each candidate offset preserved')


def benchmark(state,opt,records,out):
    runner,_,_,_,_,packets,evidences,_,refs,_,common=state
    raw=[];beg=perf_counter()
    for ci,(key,p) in enumerate(packets.items()):
        c=common[key];e=evidences[p['source_frame']]
        for warmup in (True,False):
            for repeat in range(3 if warmup else 30):
                order=('baseline','optimized') if (repeat+ci)%2==0 else ('optimized','baseline')
                for position,variant in enumerate(order):
                    r=(runner if variant=='baseline' else opt).execute('EDE',p,e,c['maps'],c['contexts'])
                    tick=perf_counter();exact(refs[key][0],r[0],key+'/timed_packet')
                    exact(semantic(refs[key][1]),semantic(r[1]),key+'/timed_audit')
                    verify_ms=(perf_counter()-tick)*1000
                    raw.append(dict(condition=key,variant=variant,warmup=warmup,repeat=repeat,
                        order='AB' if order[0]=='baseline' else 'BA',position=position,
                        N=records[key]['N'],U=records[key]['U'],A=records[key]['A'],E=records[key]['E'],
                        wall_ms=r[2]['core_wall_ms'],cpu_ms=r[2]['core_process_cpu_ms'],stage_ms=r[2],
                        exact_assertion_ms=verify_ms,output_hash=records[key]['output_hash'],
                        audit_hash=records[key]['semantic_audit_hash']))
                    del r
        print('benchmark 3+30 paired',key,flush=True)
        io.write(out/'timings_raw.json',dict(samples=raw,partial=len(raw)!=1056))
    for key,r in records.items():
        r['timing']={}
        for variant in ('baseline','optimized'):
            samples=[s for s in raw if s['condition']==key and s['variant']==variant and not s['warmup']]
            io.require(len(samples)==30,'measurement count')
            io.require(sum(s['order']=='AB' for s in samples)==15,'balanced AB/BA')
            r['timing'][variant]=dict(wall_ms=io.stats([s['wall_ms'] for s in samples]),
                cpu_ms=io.stats([s['cpu_ms'] for s in samples]),
                stage_ms={k:io.stats([s['stage_ms'][k] for s in samples]) for k in samples[0]['stage_ms']})
        r['wall_speedup']=r['timing']['baseline']['wall_ms']['mean']/r['timing']['optimized']['wall_ms']['mean']
        r['cpu_speedup']=r['timing']['baseline']['cpu_ms']['mean']/r['timing']['optimized']['cpu_ms']['mean']
        for variant in ('baseline','optimized'):
            r['common_plus_core_ms_'+variant]=common[key]['map_ms']+common[key]['prepare_ms']+r['timing'][variant]['wall_ms']['mean']
    measured=[s for s in raw if not s['warmup']]
    totals={v:{k:sum(s[k] for s in measured if s['variant']==v) for k in ('wall_ms','cpu_ms')} for v in ('baseline','optimized')}
    aggregate=dict(totals=totals,wall_speedup=totals['baseline']['wall_ms']/totals['optimized']['wall_ms'],
        cpu_speedup=totals['baseline']['cpu_ms']/totals['optimized']['cpu_ms'],
        mean_ms={v:{k:totals[v][k]/480 for k in ('wall_ms','cpu_ms')} for v in totals},
        worst={v:max((s for s in measured if s['variant']==v),key=lambda s:s['wall_ms']) for v in totals},
        pipeline_wall_ms=(perf_counter()-beg)*1000,
        frame_speedup={str(fid):sum(s['wall_ms'] for s in measured if s['condition'].startswith(f'f{fid:04d}_') and s['variant']=='baseline')/
                      sum(s['wall_ms'] for s in measured if s['condition'].startswith(f'f{fid:04d}_') and s['variant']=='optimized') for fid in io.FRAMES})
    io.write(out/'timings_raw.json',dict(samples=raw,partial=False,aggregate=aggregate,
        clocks='execute perf_counter wall/process_time CPU; verification outside core',
        profiler=False,tracemalloc=False,gc_enabled=gc.isenabled(),percentile='numpy.percentile(...,95), default linear',
        samples_per_variant=480,warmups_per_variant=48))
    return aggregate


def memory_worker(key,variant):
    runner,_,tri,_,_=snapshot()
    if variant=='optimized':runner,_=optimized_runner()
    imported=rss();p=io.read_packet(io.BASE/f'packets/{key}.npz')
    e=io.load_evidence(io.BASE/f'runtime_evidence/f{int(key[1:5]):04d}.npz');loaded=rss()
    maps=io.lookup_maps(p);ctx=tri.prepare_validation(e);prepared=rss()
    result=runner.execute('EDE',p,e,maps,ctx);peak=rss()
    arrays=dict(packet=io.array_bytes(p),evidence=io.array_bytes(e),lookup=io.array_bytes(dict(maps=maps,contexts=ctx)),
                audit=io.array_bytes(result[1]),output_packet=io.array_bytes(result[0]))
    raster=result[1]['raster_work'];del result
    tracemalloc.start();tick=perf_counter();result=runner.execute('EDE',p,e,maps,ctx)
    traced_wall=(perf_counter()-tick)*1000;current,tracedpeak=tracemalloc.get_traced_memory()
    snap=tracemalloc.take_snapshot();alloc=snap.statistics('filename');tracemalloc.stop()
    print(json.dumps(dict(variant=variant,condition=key,import_peak_bytes=imported,input_peak_bytes=loaded,
        common_peak_bytes=prepared,core_peak_bytes=peak,core_high_water_increase_bytes=max(0,peak-prepared),
        common_high_water_increase_bytes=max(0,prepared-loaded),array_bytes=arrays,raster_work=raster,
        traced_current_bytes=current,traced_peak_bytes=tracedpeak,traced_live_count=sum(s.count for s in alloc),
        traced_live_bytes=sum(s.size for s in alloc),traced_core_wall_ms=traced_wall,
        trace_scope='second separate call; no tracing in RSS/core benchmark; snapshots are live allocations, not churn',
        ru_maxrss_unit='Darwin bytes; Linux KiB converted to bytes')),flush=True)


def memory_pass(records,out):
    result={}
    for key in records:
        result[key]={}
        for variant in ('baseline','optimized'):
            proc=subprocess.run([sys.executable,'-m','visual.runtime.test.run_extended_delaunay_edges',
                '--exact-memory-worker',key,variant],cwd=REPO,capture_output=True,text=True)
            io.require(proc.returncode==0,'memory worker failed '+proc.stderr[-2000:])
            result[key][variant]=json.loads(proc.stdout.strip().splitlines()[-1])
        print('fresh memory workers',key,flush=True)
    output=dict(conditions=result,
        scope='32 fresh independent workers; same import/input/common/core boundaries; RSS before tracing',
        limitations='RSS high water includes native Qhull/allocator; zero increase does not mean zero allocation. tracemalloc is not total native memory or allocation churn.',
        temporary_estimates=dict(measured=False,
            common_bbox_arrays='mgrid 16B + centres 16B + keep B; B=bbox cells',
            SAT_axis='projection 16 bytes; centre projection 8B; inclusive comparisons temporary approx 8B+3B',
            peak_live_upper_estimate='64B + 512 bytes (conservative raster arrays only, not allocator/native overhead)',
            baseline_axes=2,optimized_axes=1,
            consequence='peak raster live arrays mostly unchanged; second axis allocation churn removed, not a 50% peak memory claim'))
    io.write(out/'memory.json',output)
    return output


def protect(initial):
    checked=0
    for rel,expected in initial['hashes'].items():
        if rel in EDITABLE:continue
        io.require((REPO/rel).is_file() and io.sha(REPO/rel)==expected,'protected file changed '+rel)
        checked+=1
    return dict(status='PASS',checked=checked,allowed_changes=list(EDITABLE))


def source_save(out,initial,binding):
    hashes={};diff=[]
    for rel in EDITABLE[:-1]:
        path=REPO/rel
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes());hashes[rel]=io.sha(path)
        old=SOURCE/'source_snapshot'/rel
        before=old.read_text() if old.exists() else ''
        diff.extend(difflib.unified_diff(before.splitlines(True),path.read_text().splitlines(True),
                                        fromfile='Task05-B/'+rel,tofile='Task05-C/'+rel))
    (out/'implementation.diff').write_text(''.join(diff))
    io.write(out/'source_hashes.json',dict(evaluated_sources=hashes,baseline_binding=binding,
        protected_before=initial,protected_after=protect(initial),
        implementation='single opposite normal removed, no policy/receiver/validator changes'))

def figures(out,summary,state,details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    from visual.runtime.test.test_triangle_intra_object_edges import oracle,fixture
    _,ede,tri,_,_,packets,_,_,refs,_,common=state
    records=summary['records'];keys=list(records);x=np.arange(len(keys));labels=[k.replace('f00','f') for k in keys]
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/'plots'/name,dpi=150);plt.close(fig)
    for key in keys:
        if int(key[1:5]) not in (0,45,120):continue
        p=packets[key];src=p['provenance']['source_index'];w=int(p['image_shape'][1]);uv=np.column_stack((src%w,src//w))
        edges=io.undirected(refs[key][0]['intra_object_edge_index']);fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,label in zip(axes,('Baseline snapshot','Optimized single SAT')):
            ax.add_collection(LineCollection(uv[edges],colors='#297fa3',linewidths=.4,alpha=.6))
            ax.scatter(*uv.T,s=2,c=p['object_id'],cmap='coolwarm',vmin=0,vmax=1)
            ax.set(xlim=(0,int(p['image_shape'][1])-1),ylim=(int(p['image_shape'][0])-1,0),aspect='equal',title=label,xlabel='u',ylabel='v')
        fig.suptitle(f'{key}: same points / edges / viewport; exact edge byte diff = 0')
        save(fig,key+'_same_edges.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,uv in zip(axes,(np.array([[0,0],[2,2]]),np.array([[-2,1],[2,-1]]),np.array([[0,0],[0,0]]))):
        cover=ede.supercover(uv);expected=np.array([(u,v) for v in range(-3,4) for u in range(-3,4) if oracle(uv,u,v)],np.int64)
        exact(cover,expected,'figure Fraction oracle')
        for u,v in cover:ax.add_patch(Rectangle((u-.5,v-.5),1,1,fc='#d7eafa',ec='#4b7999'))
        ax.plot(*uv.T,'o-',color='#bc3c4b');ax.set(xlim=(-3,3),ylim=(-3,3),aspect='equal',title='Fraction = SAT; exact row-major')
        ax.grid(alpha=.2)
    save(fig,'supercover_fraction.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,kind in zip(axes,('mask_outside','depth_jump')):
        ctx=tri.prepare_validation(fixture())[0];uv=np.array([[1,1],[7,1]])
        if kind=='mask_outside':ctx['mask'][1,4]=False
        else:ctx['pairs'].depth_jump[ctx['right'][1,4]]=True
        cover=ede.supercover(uv);check=tri.validate_cover(cover,ctx)
        ax.imshow(ctx['mask'],vmin=0,vmax=1,cmap='Greys',origin='upper');ax.plot(*uv.T,'o-',color='#b9324b')
        ax.scatter([4],[1],s=140,facecolors='none',edgecolors='red');ax.set_title(kind+'; '+str(check['counts']))
    save(fig,'mask_depth_witness.png')
    stages=('candidate_total_ms','supercover_ms','support_validation_ms','depth_validation_ms','receiver_ms','edge_packaging_ms')
    fig,ax=plt.subplots(figsize=(10,4));bottom=np.zeros(2)
    for stage in stages:
        vals=[np.mean([r['timing'][v]['stage_ms'][stage]['mean'] for r in records.values()]) for v in ('baseline','optimized')]
        ax.bar(['Baseline','Optimized'],vals,bottom=bottom,label=stage.replace('_ms',''));bottom+=vals
    core=np.array([summary['aggregate']['mean_ms'][v]['wall_ms'] for v in ('baseline','optimized')])
    ax.bar(['Baseline','Optimized'],core-bottom,bottom=bottom,label='copy / packaging / bookkeeping residual')
    ax.set(ylabel='Mean wall ms per core',title='Non-overlapping stage boundaries; nested timers excluded');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
    save(fig,'cpu_stages.png')
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    for v,color in (('baseline','#4a7b9d'),('optimized','#df9346')):
        mean=[records[k]['timing'][v]['wall_ms']['mean'] for k in keys]
        p95=[records[k]['timing'][v]['wall_ms']['p95'] for k in keys]
        axes[0].plot(x,mean,'o-',color=color,label=v+' mean');axes[0].plot(x,p95,':',color=color,label=v+' p95')
    axes[0].set(ylabel='Wall ms',xticks=x,xticklabels=labels,title='All 16 conditions; 30 samples each, no outlier removal');axes[0].legend(ncol=2)
    axes[1].bar(x,[records[k]['wall_speedup'] for k in keys],color='#368b8c');axes[1].axhline(1,color='black',lw=1)
    axes[1].set(ylabel='Baseline / optimized',xticks=x,xticklabels=labels,title='Condition speedup; frame aggregates in JSON')
    for ax in axes:ax.tick_params(axis='x',rotation=45)
    save(fig,'frame_latency_speedup.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for v in ('baseline','optimized'):
        cpu=[r['timing'][v]['cpu_ms']['mean'] for r in records.values()]
        axes[0].scatter([r['N'] for r in records.values()],cpu,label=v)
        axes[1].scatter([r['U'] for r in records.values()],cpu,label=v)
        r=records['f0000_r1']['timing'][v]['wall_ms'];axes[2].plot(['mean','median','p95','max'],[r[k] for k in ('mean','median','p95','max')],'o-',label=v)
    axes[0].set(xlabel='N',ylabel='CPU mean ms');axes[1].set(xlabel='Candidate U',ylabel='CPU mean ms')
    axes[2].set(title='Maximum N: f0000_r1 (N=1694)',ylabel='Wall ms')
    for ax in axes:ax.legend()
    save(fig,'node_candidate_max_latency.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for v in ('baseline','optimized'):
        axes[0].plot(x,[records[k]['timing'][v]['wall_ms']['max'] for k in keys],'o-',label=v+' max')
        vals=[s['wall_ms'] for s in json.loads((out/'timings_raw.json').read_text())['samples'] if s['variant']==v and not s['warmup']]
        axes[1].hist(vals,bins=25,alpha=.5,label=v)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Wall ms',title='Worst sample per condition');axes[0].tick_params(axis='x',rotation=60)
    axes[1].set(xlabel='Wall ms',ylabel='Samples',title='All 480 measured samples per path')
    for ax in axes:ax.legend()
    save(fig,'tail_latency.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    mem=summary['memory']['conditions']
    for v in ('baseline','optimized'):
        axes[0].plot(x,[mem[k][v]['core_peak_bytes']/2**20 for k in keys],'o-',label=v)
        axes[1].plot(x,[mem[k][v]['traced_peak_bytes']/2**20 for k in keys],'o-',label=v)
    axes[0].set(title='Fresh-worker RSS high water (includes imports)',ylabel='MiB');axes[1].set(title='Separate traced peak (not native/churn)',ylabel='MiB')
    for ax in axes:ax.set(xticks=x,xticklabels=labels);ax.tick_params(axis='x',rotation=60);ax.legend()
    save(fig,'memory.png')
    fig,ax=plt.subplots(figsize=(8,4))
    names=[];vals=[]
    for v in ('baseline','optimized'):
        names.extend([v+' core',v+' detailed audit'])
        vals.extend([summary['aggregate']['mean_ms'][v]['wall_ms'],np.mean([r['detailed_audit_ms'][v] for r in records.values()])])
    ax.bar(names,vals,color=['#4a7b9d','#b7cbd9','#df9346','#efcfaa']);ax.set(ylabel='Mean wall ms',title='Full reasons/counts/top4 witnesses in both cores; detailed materialization outside')
    ax.tick_params(axis='x',rotation=15);save(fig,'audit_core_boundaries.png')
    fig,ax=plt.subplots(figsize=(10,4));profile=summary['profiling']['raster_aggregate']
    names=[k for k in profile if k.endswith('_ms')];ax.barh(names,[profile[k]/16 for k in names]);ax.set(xlabel='Instrumented ms / condition',title='Original raster substages; profiler overhead excluded from A/B benchmark')
    save(fig,'raster_profile.png')


def reports(out,s):
    from html import escape
    r=s['records'];a=s['aggregate'];run=out.name
    table=['| Condition | N | U | A | Baseline mean / p95 / max ms | Optimized mean / p95 / max ms | Wall speedup | Exact |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for key,v in r.items():
        def cell(variant):
            t=v['timing'][variant]['wall_ms'];return ' / '.join(f'{t[k]:.3f}' for k in ('mean','p95','max'))
        table.append(f"| {key} | {v['N']} | {v['U']} | {v['A']} | {cell('baseline')} | {cell('optimized')} | {v['wall_speedup']:.4f}× | PASS |")
    mean=a['mean_ms'];profile=s['profiling'];mem=s['memory']['conditions']
    memory_range={v:(min(m[v]['core_peak_bytes'] for m in mem.values())/2**20,max(m[v]['core_peak_bytes'] for m in mem.values())/2**20) for v in ('baseline','optimized')}
    stage_table=['| Stage | Baseline mean ms | Optimized mean ms | Change ms |','|---|---:|---:|---:|']
    for stage in ('candidate_total_ms','supercover_ms','support_validation_ms','depth_validation_ms','edge_packaging_ms','receiver_ms'):
        av,bv=[float(np.mean([x['timing'][variant]['stage_ms'][stage]['mean'] for x in r.values()])) for variant in ('baseline','optimized')]
        stage_table.append(f'| {stage} | {av:.3f} | {bv:.3f} | {bv-av:+.3f} |')
    tail_note='최대 N 조건 p95는 baseline '+format(r['f0000_r1']['timing']['baseline']['wall_ms']['p95'],'.3f')+'ms → optimized '+format(r['f0000_r1']['timing']['optimized']['wall_ms']['p95'],'.3f')+'ms로 악화됐다. 전체 평균 개선과 tail 개선은 별개다. Peak RSS도 개선으로 판단하지 않는다.'
    links=['ede_exact_optimization.json','ede_exact_optimization.npz','timings_raw.json','profiling.json','memory.json','verification.json','tests.log','source_hashes.json','implementation.diff','config.yaml','preimplementation_gate.json','saved_npz_exact.json','audit_io_refresh.json','report_verification_refresh.diff','source_snapshot/visual/runtime/extended_delaunay_edges.py','source_snapshot/visual/runtime/test/ede_exact_optimization_eval.py','report_source_snapshot/visual/runtime/test/ede_exact_optimization_eval.py']
    links=[name for name in links if (out/name).is_file()]
    figures_list=sorted((out/'plots').glob('*.png'))
    v=s.get('verification',{})
    md=f'''# Task05-C — Exact-Equivalent EDE CPU Optimization

새 평가 UUID: `{run}`. **Exact Equivalence PASS (16/16)**. CPU 성능 판정은 계약과 별개다.

[정적 HTML](../../experiments/{run}/ede_exact_optimization.html) · [전체 JSON](../../experiments/{run}/ede_exact_optimization.json) · [NPZ](../../experiments/{run}/ede_exact_optimization.npz) · [검증](../../experiments/{run}/verification.json)

## 결과

동일 host/interpreter에서 원본 snapshot baseline mean **{mean['baseline']['wall_ms']:.3f}ms**, optimized mean **{mean['optimized']['wall_ms']:.3f}ms**, aggregate wall speedup **{a['wall_speedup']:.4f}×** ({(1-1/a['wall_speedup'])*100:.2f}% 감소). Process CPU mean은 **{mean['baseline']['cpu_ms']:.3f} → {mean['optimized']['cpu_ms']:.3f}ms**, speedup **{a['cpu_speedup']:.4f}×**다. 과거 Task05-B `116.208ms`는 참고값이며 이번 speedup 분모로 사용하지 않았다.

{'성능 개선 관측' if a['wall_speedup']>1 else '성능 개선 미관측'}. 최악 sample baseline `{a['worst']['baseline']['condition']}` **{a['worst']['baseline']['wall_ms']:.3f}ms**, optimized `{a['worst']['optimized']['condition']}` **{a['worst']['optimized']['wall_ms']:.3f}ms**. 최대 Node 조건은 `f0000_r1`, N=1694이며 아래 표와 최대 N 그림에 별도 표시한다. 30표본 p95는 기술통계이며 안정적인 tail 추정이 아니다. Outlier 삭제 없음.

{chr(10).join(table)}

{tail_note}

{chr(10).join(stage_table)}

## 기준선·profiling 관문

Task05-B `9f36f7c89dbd4097a8be9358138d1537`의 execution source snapshot을 `_task05c_snapshot_*` 별도 module namespace로 적재했다. ImportFrom을 AST에서 명시적으로 연결하고 raster/validator/receiver/execute의 module·code filename·SHA 및 object binding을 assertion으로 확인했다. Snapshot receiver 내부 import도 같은 namespace에 연결했다. 변경 없는 dependency의 현재 SHA도 확인했다. Optimized runner는 같은 snapshot execute·receiver에 현재 builder의 `optimized=True`만 명시적으로 연결했다. Python cache나 sys.path 우선순위에 의존하지 않는다.

원본 snapshot은 먼저 저장 packet·JSON 후보/origins/accepted/checks·component 상태·NPZ 최종 Edge·원본 배열을 16/16 exact 재현했다. 저장 JSON에 없는 전체 cover/pair/PairEvidence 순서는 동일 입력으로 재생성하여 optimized와 대조했다. Candidate triangles와 component 상태, chain/fallback, accepted/rejected, reason/count/witness 순서, final int64 dtype/shape/bytes/order, receiver metadata/+3/deep-copy, F=5·Camera/EE XYZ·validity·provenance/source alignment를 필드별 assertion으로 검사했다.

선분 normal a와 -a는 `P(-a)=-P(a)`, `c(-a)=-c(a)`, `r(-a)=r(a)`다. 따라서 closed-cell inclusive 두 부등식은 부호 반전 시 서로 대응한다. 계측된 두 번째 SAT normal 합계 **{profile['raster_aggregate']['SAT_normal_2_ms']:.3f}ms**, raster 비중 **{profile['duplicate_axis_share_raster']*100:.2f}%**, 원본 gate Core 대비 **{profile['duplicate_axis_share_reference_core']*100:.2f}%**로 profiling 관문을 통과했다. Gate 기준은 raster 5% 이상 및 Core 1% 이상이었다. 이는 절감량 예측이나 speedup 수치가 아니다.

별도 raster stage timer, cProfile, intrusive line tracing을 각기 실행했다. [profiling.json](../../experiments/{run}/profiling.json)은 bbox allocation·centres·첫/반대 SAT·membership, candidate/Qhull/unique/sort, Mask gather/사유, pair set/이웃/sort, support/right-down/delta/jump/witness, receiver validate/deepcopy/output 및 per-edge bookkeeping source-line 비용을 보존한다. Line time은 child 포함이며 nested 시간을 합산하지 않는다. CProfile overhead 비율 **{profile['overhead']['cprofile_ratio']:.3f}×**, line tracing **{profile['overhead']['line_trace_ratio']:.3f}×**; raster timer pass와 uninstrumented gate 차이는 **{profile['overhead']['raster_timer_overhead_ms']:.3f}ms**다. 이 차이에는 실행 변동도 포함된다.

## 구현과 overflow 안전성

`build_extended_delaunay_edges(..., optimized=True)`만 opt-in이다. 기본 `optimized=False`는 기존 raster를 사용한다. 정책 metadata·후보/검증/정렬/threshold는 동일하다. 새 traversal/근사/cache/early exit는 없다.

M을 모든 endpoint 절대 좌표의 최댓값으로 두면 각 normal 성분 ≤2M, doubled vertex/centre 성분 ≤2M, 각 dot product와 그 partial sum 절댓값 ≤8M², radius ≤4M다. **Python integer `8M²+4M <= int64_max`** guard는 차분·부호 반전·doubling·dot·`c±radius`·bbox `hi+1`까지 안전하게 제한한다. Guard 전 signed/unsigned extrema를 Python int로 읽어 wrap을 피한다. 두 normal 모두 안전한 domain에서만 하나를 생략한다. 증명 범위 밖/triangle/unsupported dtype·shape는 기존 `supercover`로 위임한다. 실제 이미지 UV는 이 범위 안이다. Large signed/unsigned fallback과 경계 부근도 테스트했다.

## 공정한 측정·memory·audit

Python `{s['environment']['python']}`, NumPy `{s['environment']['numpy']}`, SciPy `{s['environment']['scipy']}`, `{s['environment']['platform']}`. Qhull extension SHA와 interpreter·thread 값은 JSON environment에 보존했다.

16조건·phase 0, 구현별 warmup 3회와 조건별 30쌍, AB/BA 각 15회, 조건 시작 순서 교대. Thread 환경 OMP/OpenBLAS/VECLIB=1을 NumPy import 전 고정했다. 실제 native thread pool 조회는 `{s['native_threadpools']['status']}`다. 같은 Python GC 설정과 반환값 lifetime을 사용했다. Warmup 96회 + measured 960회 모두 timed 종료 후 exact assertion을 실행했다. Hash/assertion 비용은 timed Core 밖이며 raw에 별도 기록했다.

양쪽 Core는 packet deepcopy·fresh candidate·동일 cover/validation·모든 reason/count/top4 witness·bookkeeping·canonical Edge/metadata/packet packaging·전체 receiver 검증과 deepcopy를 포함한다. `core_audit_mode=full_reasons_counts_top4_witness`, `detailed_audit_in_core=false`. 공통 selected map/prepare_validation은 별도이며 JSON에 common+core도 기록했다. Detailed cover/pair materialization·plot·IO·tracemalloc·cProfile/line tracer는 timed Core 밖이다. Detailed audit는 전체 cover와 ordered right/down pair 및 evidence index/support/delta/jump를 양쪽 동일하게 생성한다.

RSS high-water baseline {memory_range['baseline'][0]:.2f}–{memory_range['baseline'][1]:.2f}MiB, optimized {memory_range['optimized'][0]:.2f}–{memory_range['optimized'][1]:.2f}MiB. 32 fresh workers에서 import/input/common/Core 경계를 동일하게 측정했다. Darwin bytes/Linux KiB 변환 적용. 0 증분은 allocation 0이 아니다. 별도 tracemalloc current/peak·live snapshot count/bytes는 native 전체나 allocation churn 측정이 아니다. Raster logical temporary 추정과 measured RSS/trace를 구분했다. 축 중복 제거로 allocation 반복은 줄지만 bbox/centres와 peak live 배열 크기는 대체로 그대로다.

## 검증·산출물

68 tests PASS: 기존 Task05/05-B·k-NN·feature·receiver 회귀, 전체 EDE fixture의 optimized 재실행, Fraction oracle의 음수·역방향·corner·degenerate, int64 guard, 여러 reason/count/top4 witness를 포함한다. 기존 기대값은 변경하지 않았다. 실제 16조건 three-way exact와 매 benchmark 반환값 exact를 별도 확인했다.

NPZ `allow_pickle=False` readback은 모든 key/dtype/shape/bytes와 ragged offsets를 원본 메모리 값과 대조했다. Packet readback·receiver 및 JSON parse/통계 재계산·기존 입력/artifact/sidecar hash를 검증했다. JSON/NPZ/PNG 상대 링크 및 Markdown↔HTML 양방향 경로도 확인했다. PNG decode `{v.get('png_decode','PENDING')}`, 실제 PNG 시각 검토 `{v.get('png_visual_review',{}).get('status','PENDING')}`, 실제 browser 검증 `{v.get('browser',{}).get('status','NOT_EXECUTED')}`. 미실행 항목을 PASS로 표시하지 않는다.

네 source run 및 기존 dirty/untracked 파일 보존 결과: `{v.get('protected',{}).get('status','PENDING')}`. 공통 `ExperimentLogger(team='visual', job_type='evaluate', seed=0)`를 사용했고 새 run만 link_output으로 등록했다. Sampling/PCA5/feature extraction/EE transform·simulator·GPU/PCL·학습·dependency 설치·업로드·Git commit은 실행하지 않았다. 관측 계약 PASS는 물리 연결 정확성 또는 GNN 20Hz를 확정하지 않는다. 기존 frame0 오염 plateau 연결도 그대로다.

측정 후 보강한 NPZ/JSON readback assertion·stage/tail 설명·그림 label은 `report_source_snapshot`과 별도 diff에 보존했다. 측정 source snapshot, 모든 timing sample·packet·record·수치 NPZ·profiling/memory는 보존했고 benchmark Core 함수는 바뀌지 않았다. Packet/record 직렬화의 별도 보고서 IO pass는 `audit_io_refresh.json`에 기록했다.

재현 (같은 저장 입력과 설치 환경; 새 평가는 새 UUID 생성):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task05c-mpl .venv/bin/python -m visual.runtime.test.run_extended_delaunay_edges --exact-optimization
```

'''+ '\n'.join(f'[{name}](../../experiments/{run}/{name})' for name in links)+'\n\n'+ '\n\n'.join(f'![{p.stem}](../../experiments/{run}/plots/{p.name})' for p in figures_list)+'\n'
    REPORT.write_text(md)
    def html_table(rows):
        cells=[line.strip('|').split('|') for line in rows if not line.startswith('|---')]
        return '<table>'+''.join('<tr>'+''.join('<td>'+escape(c.strip())+'</td>' for c in row)+'</tr>' for row in cells)+'</table>'
    html=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Task05-C Exact EDE CPU</title>
<style>body{{font:16px system-ui;max-width:1350px;margin:32px auto;padding:0 20px;color:#183047;background:#f8fafc}}p{{line-height:1.65}}table{{border-collapse:collapse;width:100%;background:white;font-size:12px}}td{{border:1px solid #cbd4dc;padding:8px}}img{{max-width:100%;height:auto}}figure{{margin:25px 0;padding:12px;background:white}}code{{overflow-wrap:anywhere}}a{{color:#086c9a}}</style></head><body>
<h1>Task05-C — Exact EDE CPU optimization</h1><p><code>{run}</code></p><p><strong>Exact Equivalence PASS: 16/16 fixed inputs; zero semantic / final edge byte differences.</strong></p>
<p>Full CPU core: {mean['baseline']['wall_ms']:.3f} → {mean['optimized']['wall_ms']:.3f}ms; wall {a['wall_speedup']:.4f}×; CPU {a['cpu_speedup']:.4f}×. Task05-B historical 116.208ms is reference only. All core reasons/counts/top4 witnesses retained on both paths.</p>
<p><a href="../../docs/visual-graph-integration/05c_ede_exact_optimization.md">결과 Markdown / full methodology and limits</a></p>
<p>'''+ ' · '.join(f'<a href="{name}">{name}</a>' for name in links)+f'''</p>{html_table(table)}
<p>{tail_note}</p>{html_table(stage_table)}<h2>Original profiling gate and exact checks</h2><p>Opposite SAT normal = {profile['duplicate_axis_share_raster']*100:.2f}% of instrumented raster. Separate stage timer, cProfile and intrusive source-line passes are in profiling.json, including allocation/centres/SAT, Mask/Depth, pair enumeration/sort and receiver costs. These profilers were disabled for the 30-pair benchmark. Snapshot imports are explicitly isolated and verified by object binding, source filename and SHA-256.</p>
<p>Normal negation swaps the same inclusive inequalities; guard 8M²+4M ≤ int64 max proves arithmetic safety. Otherwise original raster is called. Opt-in optimized=True only; production default unchanged. No threshold, tolerance, ordering, candidates, masks, depth evidence, points or F=5 fields changed.</p>
<h2>Same work / memory / verification</h2><p>16 inputs, 3 warmups per path, 30 paired measurements each, AB/BA balanced. Fresh candidates and full validation / witness / packaging / receiver each invocation. Common preparation and detailed audit / plots / IO are outside core on both paths. 32 fresh RSS workers and separate traced-allocation passes. RSS includes imports/native allocators; tracing is not total churn. Exact input/output/audit hashes and full field assertions retained.</p>
<p>68 tests PASS. Worst baseline {a['worst']['baseline']['wall_ms']:.3f}ms; optimized {a['worst']['optimized']['wall_ms']:.3f}ms. No outliers removed. Thirty-sample p95 is descriptive. PNG visual review: {v.get('png_visual_review',{}).get('status','PENDING')}; browser: {v.get('browser',{}).get('status','NOT_EXECUTED')}. Physical connection accuracy / GNN 20Hz not established. No simulator/GPU/PCL/training/upload/install/commit.</p>
<h2>Every condition packet / record</h2><ul>'''+''.join(f'<li>{key}: '+ ' · '.join(f'<a href="{folder}/{key}_{variant}.{ext}">{variant} {folder}</a>' for variant in ('baseline','optimized') for folder,ext in (('records','json'),('packets','npz')) )+'</li>' for key in r)+'</ul><h2>Identical edge overlays, raster validation and CPU evidence</h2>'+''.join(f'<figure><figcaption>{escape(p.stem)}</figcaption><img src="plots/{p.name}" alt="{escape(p.stem)}"></figure>' for p in figures_list)+'</body></html>'
    (out/'ede_exact_optimization.html').write_text(html)

def verify_outputs(out,summary,initial):
    import re
    from html.parser import HTMLParser
    from PIL import Image
    raw=json.loads((out/'timings_raw.json').read_text())
    io.require(len(raw['samples'])==1056 and not raw['partial'],'warmup/measured schedule count')
    for key,r in summary['records'].items():
        for variant in ('baseline','optimized'):
            p=io.read_packet(out/f'packets/{key}_{variant}.npz')
            io.receive_vision_packet(p);io.preserved(io.read_packet(io.BASE/f'packets/{key}.npz'),p)
            io.require(io.packet_digest(p)==r['output_hash'],'packet readback semantic hash')
            record=json.loads((out/f'records/{key}_{variant}.json').read_text())
            io.require(record['detailed_hash']==r['detailed_hash'],'record detailed hash')
            io.require(record['summary']['output_hash']==r['output_hash'],'record output hash')
            samples=[s for s in raw['samples'] if s['condition']==key and s['variant']==variant]
            io.require(len(samples)==33 and sum(s['warmup'] for s in samples)==3,'warmups')
            measured=[s for s in samples if not s['warmup']]
            io.require(sum(s['order']=='AB' for s in measured)==15,'balanced order')
            for clock in ('wall_ms','cpu_ms'):
                exact(io.stats([s[clock] for s in measured]),r['timing'][variant][clock],key+'/statistics')
            for sample in samples:
                io.require(sample['output_hash']==r['output_hash'] and sample['audit_hash']==r['semantic_audit_hash'],'raw exact hash')
    pngs=sorted((out/'plots').glob('*.png'))
    for p in pngs:
        with Image.open(p) as im:im.verify()
        with Image.open(p) as im:im.load();io.require(min(im.size)>0,'empty PNG')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))
    parser=Links();parser.feed((out/'ede_exact_optimization.html').read_text())
    links=[(out,link) for link in parser.links]+[(REPORT.parent,link) for link in re.findall(r'\]\(([^)]+)\)',REPORT.read_text())]
    for root,link in links:
        io.require(not link.startswith(('http:','https:','//')),'external HTML dependency')
        io.require((root/link.split('#')[0]).is_file(),'relative link missing '+link)
    io.require('../../docs/visual-graph-integration/05c_ede_exact_optimization.md' in parser.links,'HTML to Markdown')
    io.require('../../experiments/'+out.name+'/ede_exact_optimization.html' in REPORT.read_text(),'Markdown to HTML')
    for rel,h in summary['input_hashes'].items():io.require(io.sha(REPO/rel)==h,'input artifact changed '+rel)
    # Numerical JSON/NPZ and record checks are supplemented with fresh full field
    # assertions in real_exact and every one of the 1056 core return values.
    with np.load(out/'ede_exact_optimization.npz',allow_pickle=False) as ar:
        for key in summary['records']:
            for variant in ('baseline','optimized'):
                prefix=key+'/'+variant
                exact(ar[prefix+'/edge_index'],io.read_packet(out/f'packets/{key}_{variant}.npz')['intra_object_edge_index'])
                record=json.loads((out/f'records/{key}_{variant}.json').read_text())
                saved_record=json.loads((SOURCE/f'records/{key}.json').read_text())
                for field in ('candidates','origins','accepted','checks'):
                    exact(record['audit'][field],saved_record['EDE'][field],key+'/record_saved/'+field)
                for name in ar.files:
                    if not name.startswith(prefix+'/audit/'):continue
                    value=record['audit']
                    for part in name[len(prefix+'/audit/'):].split('/'):
                        value=value[int(part)] if isinstance(value,list) else value[part]
                    exact(io.serial(ar[name]),value,name+'/JSON_NPZ')
                with np.load(SOURCE/'extended_delaunay_edgewise.npz',allow_pickle=False) as saved:
                    for field in ('candidates','origins','accepted','triangles'):
                        exact(ar[prefix+'/audit/'+field],saved[key+'/EDE_'+field],key+'/saved_npz/'+field)
                for name in ('cover_offsets','pair_offsets'):
                    offsets=ar[prefix+'/'+name]
                    io.require(offsets.dtype==np.int64 and offsets[0]==0 and np.all(np.diff(offsets)>=0),'ragged offsets')
                    io.require(len(offsets)==summary['records'][key]['U']+1,'offset candidates')
                    target='cover_uv' if name=='cover_offsets' else 'pairs'
                    io.require(offsets[-1]==len(ar[prefix+'/'+target]),'ragged terminal')
    protected=protect(initial)
    source_hashes=json.loads((out/'source_hashes.json').read_text())
    refresh=source_hashes.get('report_verification_source_changes',{})
    for p in (out/'source_snapshot').rglob('*.py'):
        rel=str(p.relative_to(out/'source_snapshot'))
        if rel in refresh:
            info=refresh[rel]
            io.require(io.sha(p)==info['evaluated_sha256'],'evaluated source snapshot changed')
            io.require(io.sha(REPO/rel)==info['report_verification_sha256'],'report/verification source changed')
            io.require(io.sha(out/'report_source_snapshot'/rel)==info['report_verification_sha256'],'report snapshot mismatch')
        else:io.require(io.sha(p)==io.sha(REPO/rel),'evaluated source changed')
    previous=summary.get('verification',{})
    return dict(status='PASS',exact_equivalence='PASS',three_way_conditions=16,
        benchmark_return_assertions=1056,tests=summary['tests'],
        semantic_exclusions=['timing_ms','candidate_object_ms','validation_object_ms'],
        array_contract='dtype/shape/bytes/order; NaN payload bytes retained in arrays',
        json_readback='PASS',statistics_recalculation='PASS',packet_receiver_readback='PASS',
        npz_readback='PASS',ragged_offsets='PASS',png_decode='PASS',png_count=len(pngs),
        links='PASS',relative_links_checked=len(links),markdown_html_bidirectional='PASS',protected=protected,
        png_visual_review=previous.get('png_visual_review',dict(status='NOT_EXECUTED',files=[])),
        browser=previous.get('browser',dict(status='NOT_EXECUTED')),
        simulator='NOT_EXECUTED',GPU='NOT_EXECUTED',PCL='NOT_EXECUTED',training='NOT_EXECUTED',
        sampling_calls=0,feature_extraction_calls=0,ee_transform_calls=0,
        physical_connection_accuracy='NOT_ESTABLISHED',production_default_changed=False)


def main():
    import re
    from utils.experiment_logger import ExperimentLogger
    for key in THREADS:io.require(os.getenv(key)=='1','launch with '+key+'=1 before NumPy import')
    io.require(os.getenv('WANDB_MODE')=='disabled','local only WANDB_MODE=disabled')
    initial_path=Path('/tmp/task05c_initial.json')
    if initial_path.exists():initial=json.loads(initial_path.read_text())
    else:
        paths=[p for root in (SOURCE,io.BASE,io.OLD,io.SELECTION) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        initial=dict(status=subprocess.check_output(['git','status','--porcelain','-uall'],cwd=REPO,text=True),
                     hashes={str(p.relative_to(REPO)):io.sha(p) for p in paths})
    protect(initial)
    state=baseline_gate()
    print('original snapshot exact PASS: 16/16; start separate profiling',flush=True)
    profile=profiling(state)
    io.require(profile['status']=='PASS','profiling STOP_REPLAN: do not implement/benchmark another optimization')
    config=dict(task='05-C',frames=io.FRAMES,keep_ratios=io.RATIOS,phase=0,
        source_runs=[SOURCE.name,io.OLD.name,io.BASE.name,io.SELECTION.name],seed=0,
        warmups=3,measurements=30,paired_order='AB/BA alternating; condition start alternates',
        core_audit_mode='full_reasons_counts_top4_witness',detailed_audit_in_core=False,
        common_outside_core=['selected-source maps','prepare_validation'],
        detailed_outside_core=['full cover/pair materialization','plots','JSON/NPZ/PNG/HTML IO','profiling','memory tracing'],
        thread_env={k:os.getenv(k) for k in THREADS},optimization='one duplicate opposite segment SAT normal',
        default_optimized=False,baseline='Task05-B execution snapshot explicit isolated namespace',
        gc_enabled=gc.isenabled(),historical_Task05B_wall_mean_ms=116.208)
    with ExperimentLogger(team='visual',job_type='evaluate',model='task05c_ede_exact_optimization',config=config,seed=0) as logger:
        out=logger.directory;print('RUN_ID',out.name,flush=True)
        Path('/tmp/task05c_run_id').write_text(out.name)
        for folder in ('records','packets','plots'): (out/folder).mkdir()
        source_save(out,initial,state[4]);io.write(out/'profiling.json',profile)
        if Path('/tmp/task05c_gate.json').exists():
            (out/'preimplementation_gate.json').write_bytes(Path('/tmp/task05c_gate.json').read_bytes())
        test=subprocess.run([sys.executable,'-m','unittest',*io.TESTS,'-v'],cwd=REPO,capture_output=True,text=True)
        (out/'tests.log').write_text(test.stdout+test.stderr)
        io.require(test.returncode==0,'tests failed; do not benchmark')
        tests=dict(status='PASS',count=int(re.search(r'Ran (\d+) tests',test.stdout+test.stderr).group(1)),modules=io.TESTS)
        opt,current,records,details,readback=real_exact(state,out)
        print('EXACT EQUIVALENCE PASS: 16/16; benchmark now permitted',flush=True)
        aggregate=benchmark(state,opt,records,out)
        memory=memory_pass(records,out)
        try:
            from threadpoolctl import threadpool_info
            pools=dict(status='EXECUTED',pools=threadpool_info())
        except ImportError:pools=dict(status='NOT_EXECUTED',reason='threadpoolctl not installed; no dependencies installed')
        summary=dict(run_id=out.name,config=config,environment=io.environment(),
            source_environment=json.loads((SOURCE/'extended_delaunay_edgewise.json').read_text())['environment'],
            native_threadpools=pools,tests=tests,records=records,aggregate=aggregate,
            input_hashes=state[7]['input_hashes'],baseline_gate=state[9],baseline_binding=state[4],
            common={k:{n:v for n,v in c.items() if n not in ('maps','contexts')} for k,c in state[10].items()},
            profiling=profile,memory=memory,readback=readback,exact_equivalence='PASS',
            performance='IMPROVED' if aggregate['wall_speedup']>1 else 'NOT_IMPROVED',
            semantic_exclusions=['timing_ms','candidate_object_ms','validation_object_ms'])
        tick=perf_counter();figures(out,summary,state,details);summary['plot_wall_ms']=(perf_counter()-tick)*1000
        io.write(out/'verification.json',dict(status='PENDING'))
        io.write(out/'ede_exact_optimization.json',summary);reports(out,summary)
        summary['verification']=verify_outputs(out,summary,initial)
        io.write(out/'verification.json',summary['verification']);io.write(out/'ede_exact_optimization.json',summary);reports(out,summary)
        exact(json.loads((out/'ede_exact_optimization.json').read_text()),io.serial(summary),'summary JSON full readback')
        logger.record(source_runs=config['source_runs'],correctness='PASS',performance=summary['performance'],
                      production_default_changed=False,physical_connection_accuracy='NOT_ESTABLISHED')
        logger.log(dict(exact_conditions=16,wall_speedup=aggregate['wall_speedup'],cpu_speedup=aggregate['cpu_speedup']))
        for path in list(out.rglob('*'))+[REPORT]:
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','metrics.json','manifest.json','config.yaml'):
                kind='trajectory' if path.suffix=='.npz' else 'metrics' if path.suffix=='.json' else 'reports'
                logger.link_output(path,kind=kind)
        protect(initial)
        print('DONE',out.name,json.dumps(aggregate['mean_ms']),aggregate['wall_speedup'],flush=True)


if __name__=='__main__':main()
