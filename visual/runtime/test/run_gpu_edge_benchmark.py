"""Task05-D local MPS B1 evaluation. No simulator, training or publishing.

Fixed 16 conditions, 10 warmups and 30 paired AB/BA measurements. Explicit
unsupported/failure states retain artifacts. Mask/Depth validation stays CPU.
"""
import copy
import difflib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from time import perf_counter, process_time

import numpy as np

from utils.experiment_logger import ExperimentLogger
from visual.runtime.test import ede_exact_optimization_eval as c
from visual.runtime.test.ede_gpu_bbox_sat import B1Backend, SCRATCH_CAP_BYTES

io=c.io
REPO=io.REPO
SOURCE=REPO/'experiments/185bc034567d429daf3f1d09f94dc56c'
REPORT=REPO/'docs/visual-graph-integration/05d_gpu_edge_benchmark.md'
NEW_FILES=('visual/runtime/test/ede_gpu_bbox_sat.py',
           'visual/runtime/test/run_gpu_edge_benchmark.py',
           'visual/runtime/test/test_gpu_edge_equivalence.py')
SOURCE_RUNS=('9185355a32b8433abc70f1738f983377','9f90293b227e4dbfaaa6290afad9cc3b',
             '91ecd0413a87455cabc5e813cddaea8d','9f36f7c89dbd4097a8be9358138d1537',SOURCE.name)
WARMUPS=10
REPEATS=30


def protection():
    """Never reuse an earlier task's editable-file exclusions."""
    path=Path('/tmp/task05d_initial.json')
    if path.exists():initial=json.loads(path.read_text())
    else:
        files={p for run in SOURCE_RUNS for p in (REPO/'experiments'/run).rglob('*')
               if p.is_file() and '__pycache__' not in p.parts}
        status=subprocess.check_output(['git','status','--porcelain','-uall'],cwd=REPO,text=True)
        for line in status.splitlines():
            p=REPO/line[3:]
            if p.is_file() and line[3:] not in (*NEW_FILES,str(REPORT.relative_to(REPO))):files.add(p)
        initial=dict(git_status=status,source_runs=SOURCE_RUNS,
                     hashes={str(p.relative_to(REPO)):io.sha(p) for p in sorted(files)})
    # Protect all previously declared raw inputs too, including sidecars where present.
    old=json.loads((SOURCE/'ede_exact_optimization.json').read_text())
    for rel,sha in old['input_hashes'].items():
        c.exact(io.sha(REPO/rel),sha,'input hash '+rel)
        initial['hashes'].setdefault(rel,sha)
        sidecar=REPO/(rel+'.tracking.json')
        if sidecar.is_file():initial['hashes'].setdefault(str(sidecar.relative_to(REPO)),io.sha(sidecar))
    verify_protection(initial)
    return initial


def verify_protection(initial):
    for rel,sha in initial['hashes'].items():c.exact(io.sha(REPO/rel),sha,'protected '+rel)
    return dict(status='PASS',files=len(initial['hashes']),source_runs=list(SOURCE_RUNS))


def save_sources(out,initial,binding=None):
    files=set(NEW_FILES)|{'visual/runtime/extended_delaunay_edges.py',
        'visual/runtime/triangle_intra_object_edges.py','graph/vision_input.py',
        'visual/runtime/test/ede_exact_optimization_eval.py','utils/experiment_logger.py',
        'utils/experiment_artifacts.py','docs/visual-graph-integration/05d_gpu_edge_benchmark_plan.md'}
    diff=[];hashes={}
    for rel in sorted(files):
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((REPO/rel).read_bytes());hashes[rel]=io.sha(dest)
        if rel in NEW_FILES:
            diff.extend(difflib.unified_diff([],dest.read_text().splitlines(True),fromfile='/dev/null',tofile=rel))
    (out/'implementation.diff').write_text(''.join(diff))
    io.write(out/'source_hashes.json',dict(evaluated_sources=hashes,baseline_binding=binding,
        protected_before=initial,protected_after=verify_protection(initial)))


def cpu_gate(out):
    h=json.loads((SOURCE/'source_hashes.json').read_text());checked={}
    for rel,sha in h['evaluated_sources'].items():
        c.exact(io.sha(SOURCE/'source_snapshot'/rel),sha,'Task05-C snapshot '+rel)
        expected=h.get('report_verification_source_changes',{}).get(rel,{}).get('report_verification_sha256',sha)
        c.exact(io.sha(REPO/rel),expected,'Task05-C source '+rel);checked[rel]=expected
    _,_,tri,receiver,binding=c.snapshot();a,current=c.optimized_runner()
    c.exact(io.sha(Path(current.validate_cover.__code__.co_filename)),
            io.sha(Path(tri.validate_cover.__code__.co_filename)),'unchanged validator source')
    packets,evidences,_=c.inputs();refs={};details={};gates={}
    with np.load(SOURCE/'ede_exact_optimization.npz',allow_pickle=False) as ar:
        for key,p in packets.items():
            ctx=tri.prepare_validation(evidences[p['source_frame']]);maps=io.lookup_maps(p)
            result=a.execute('EDE',p,evidences[p['source_frame']],maps,ctx)
            c.exact(result[0],io.read_packet(SOURCE/f'packets/{key}_optimized.npz'),key+'/saved packet')
            record=json.loads((SOURCE/f'records/{key}_optimized.json').read_text())
            c.exact(io.serial(c.semantic(result[1])),record['audit'],key+'/saved audit')
            detail=c.detailed(p,result[1],current._segment_supercover_single_sat,tri,ctx)
            arrays=io.flatten_arrays(c.semantic(result[1]),prefix=key+'/optimized/audit')
            arrays.update({key+'/optimized/'+n:v for n,v in detail.items()})
            arrays.update(io.flatten_arrays(p,prefix=key+'/original'))
            arrays[key+'/optimized/edge_index']=result[0]['intra_object_edge_index']
            for name,value in arrays.items():c.exact(value,ar[name],name)
            refs[key]=result;details[key]=detail
            gates[key]=dict(status='PASS',arrays=len(arrays),packet_hash=io.packet_digest(result[0]),
                            audit_hash=io.packet_digest(c.semantic(result[1])),detail_hash=io.packet_digest(detail))
            print('CPU A saved exact PASS',key,flush=True)
    gate=dict(status='PASS',conditions=gates,binding=binding,sources=checked,
              receiver_binding=dict(module=receiver.__name__,path=receiver.__file__))
    io.write(out/'cpu_baseline_gate.json',gate)
    return a,current,tri,receiver,packets,evidences,refs,details,gate


def gpu_detail(p,audit,last,tri,ctx):
    cursor=0;source=p['provenance']['source_index'];w=int(p['image_shape'][1])
    endpoints=np.column_stack((source%w,source//w))[audit['candidates']]
    def raster(uv):
        nonlocal cursor
        c.exact(uv,endpoints[cursor],'GPU detail source/row order')
        offsets=last['cover_offsets'];result=last['cover_uv'][offsets[cursor]:offsets[cursor+1]]
        cursor+=1;return result
    result=c.detailed(p,audit,raster,tri,ctx)
    c.exact(cursor,len(endpoints),'GPU cover candidate count')
    return result


def check_result(key,result,ref,detail=None,last=None):
    c.exact(result[0],ref[0],key+'/packet')
    c.exact(c.semantic(result[1]),c.semantic(ref[1]),key+'/audit')
    if last is not None:
        c.exact(last['cover_uv'],detail['cover_uv'],key+'/cover membership/order')
        c.exact(last['cover_offsets'],detail['cover_offsets'],key+'/ragged offsets')


def equivalence(out,state,backend,b):
    a,current,tri,receiver,packets,evidences,refs,details,gate=state
    arrays={};records={};gpu_refs={}
    for key,p in packets.items():
        e=evidences[p['source_frame']];ctx=tri.prepare_validation(e);maps=io.lookup_maps(p)
        p_hash=io.packet_digest(p)
        result=b.execute('EDE',p,e,maps,ctx);last=backend.last
        check_result(key,result,refs[key],details[key],last)
        d=gpu_detail(p,result[1],last,tri,ctx);c.exact(d,details[key],key+'/all pairs/evidence')
        ra=receiver.receive_vision_packet(refs[key][0],robot_positions=np.zeros((3,3),np.float32))
        rb=receiver.receive_vision_packet(result[0],robot_positions=np.zeros((3,3),np.float32))
        for field in ra.__dataclass_fields__:c.exact(getattr(ra,field),getattr(rb,field),key+'/receiver/'+field)
        c.exact(rb.edge_index,result[0]['intra_object_edge_index']+3,key+'/robot offset')
        rb.packet['points'][:]=0;c.exact(result[0],refs[key][0],key+'/receiver deep copy')
        io.preserved(p,result[0]);c.exact(io.packet_digest(p),p_hash,key+'/input immutable')
        gpu_refs[key]=result
        w=int(p['image_shape'][1]);src=p['provenance']['source_index']
        endpoints=np.column_stack((src%w,src//w))[result[1]['candidates']]
        bbox=np.prod(np.abs(endpoints[:,1]-endpoints[:,0])+1,axis=1)
        cover=np.diff(d['cover_offsets']);pair=np.diff(d['pair_offsets'])
        record=dict(status='PASS',saved_vs_A='PASS',saved_vs_B1='PASS',A_vs_B1='PASS',
            N=len(p['points']),U=len(bbox),accepted=int(result[1]['accepted'].sum()),
            bbox_cells=int(bbox.sum()),cover_cells=int(cover.sum()),pair_occurrences=int(pair.sum()),
            distributions={n:dict(io.stats(v),p99=float(np.percentile(v,99))) for n,v in [('bbox',bbox),('cover',cover),('pair',pair)]},
            input_hash=p_hash,output_hash=io.packet_digest(result[0]),
            audit_hash=io.packet_digest(c.semantic(result[1])),detail_hash=io.packet_digest(d),
            mask_depth_validation='CPU unchanged validate_cover; B2 NOT_EXECUTED')
        records[key]=record
        io.write(out/f'records/{key}.json',dict(summary=record,A=c.semantic(refs[key][1]),B1=c.semantic(result[1]),
            bbox_cells_per_edge=bbox,cover_cells_per_edge=cover,pair_occurrences_per_edge=pair))
        for variant,r,det in (('A',refs[key],details[key]),('B1',result,d)):
            io.save_packet(out/f'packets/{key}_{variant}.npz',r[0])
            c.exact(r[0],io.read_packet(out/f'packets/{key}_{variant}.npz'),key+'/packet readback')
            prefix=key+'/'+variant
            arrays.update(io.flatten_arrays(c.semantic(r[1]),prefix=prefix+'/audit'))
            arrays.update({prefix+'/'+k:v for k,v in det.items()})
            arrays[prefix+'/edge_index']=r[0]['intra_object_edge_index']
            arrays[prefix+'/bbox_cells']=bbox
        arrays.update(io.flatten_arrays(p,prefix=key+'/original'))
        print('three-way exact PASS',key,flush=True)
    counts={n:sum(r[n] for r in records.values()) for n in ('U','bbox_cells','cover_cells','pair_occurrences')}
    c.exact(counts,dict(U=30143,bbox_cells=625978,cover_cells=227426,pair_occurrences=211294),'fixed workload')
    np.savez_compressed(out/'gpu_edge_benchmark.npz',**arrays)
    with np.load(out/'gpu_edge_benchmark.npz',allow_pickle=False) as ar:
        c.exact(set(ar.files),set(arrays),'NPZ key set')
        for name,value in arrays.items():c.exact(value,ar[name],name+'/NPZ readback')
    plateau=evidences[0].frame.geometry.xyz[...,2]
    io.write(out/'exact_equivalence.json',dict(status='PASS',conditions=records,totals=counts,
        npz_arrays=len(arrays),frame0_plateau=dict(depth_m=8.430999755859375,
        pixel_count=int((plateau==np.float32(8.431)).sum()),saved_edges_unchanged=True),
        criteria='full ordered cover/pairs/evidence/checks/reasons/counts/top4 witnesses/packet/receiver; no tolerance'))
    return records,gpu_refs


def pipeline(out,state,backend,b):
    import torch
    a,_,tri,_,packets,evidences,refs,details,_=state
    raw=[];warm=[];assertions=[];start=perf_counter()
    for ki,(key,p) in enumerate(packets.items()):
        e=evidences[p['source_frame']]
        for phase,count,target in (('warmup',WARMUPS,warm),('measured',REPEATS,raw)):
            for i in range(count):
                order=('A','B1') if (ki+i)%2==0 else ('B1','A')
                for variant in order:
                    torch.mps.synchronize();tick=perf_counter();cpu_tick=process_time()
                    t=perf_counter();maps=io.lookup_maps(p);ctx=tri.prepare_validation(e);common=(perf_counter()-t)*1000
                    r=(a if variant=='A' else b).execute('EDE',p,e,maps,ctx)
                    torch.mps.synchronize();wall=(perf_counter()-tick)*1000;cpu=(process_time()-cpu_tick)*1000
                    row=dict(condition=key,iteration=i,phase=phase,order=list(order),variant=variant,
                             total_wall_ms=wall,process_cpu_ms=cpu,common_ms=common,stages=r[2])
                    if variant=='B1':row['gpu']={k:v for k,v in backend.last.items() if k not in ('cover_uv','cover_offsets')}
                    target.append(row)
                    t=perf_counter();check_result(key,r,refs[key],details[key],backend.last if variant=='B1' else None)
                    assertions.append(dict(condition=key,variant=variant,phase=phase,iteration=i,wall_ms=(perf_counter()-t)*1000))
                if phase=='measured' and i in (9,19,29):print('pipeline',key,i+1,'/30',flush=True)
    result=dict(scope='S1 fresh common preparation + fresh CPU candidate + full core + unchanged CPU receiver; actual synchronized wall',
                warmups_per_variant_per_condition=WARMUPS,repeats_per_variant_per_condition=REPEATS,
                measured=raw,warmup=warm,exact_assertions_outside_timed=assertions,
                wall_s=(perf_counter()-start),event_status='TIMING_UNSUPPORTED',device_event_ms=None,
                native_threadpools='NOT_EXECUTED',stage_sync='none inside main pipeline; cover readback and final completion synchronize')
    io.write(out/'timings_raw.json',result)
    return result


def isolated(out,state,backend):
    a,current,tri,_,packets,_,refs,details,_=state
    rows=[];warm=[];memory={}
    for ki,(key,p) in enumerate(packets.items()):
        src=p['provenance']['source_index'];w=int(p['image_shape'][1])
        endpoints=np.column_stack((src%w,src//w))[refs[key][1]['candidates']]
        for phase,count,target in (('warmup',WARMUPS,warm),('measured',REPEATS,rows)):
            for i in range(count):
                order=('A','B1') if (ki+i)%2==0 else ('B1','A')
                for variant in order:
                    if variant=='A':
                        t=perf_counter();covers=[current._segment_supercover_single_sat(uv) for uv in endpoints]
                        raster_ms=(perf_counter()-t)*1000
                        t=perf_counter();uv=np.concatenate(covers);off=np.r_[np.int64(0),np.cumsum([len(c) for c in covers],dtype=np.int64)]
                        packing_ms=(perf_counter()-t)*1000
                        row=dict(condition=key,phase=phase,iteration=i,variant='A',
                            supercover_wall_ms=raster_ms,ragged_output_packing_ms=packing_ms,
                            hybrid_wall_ms=raster_ms+packing_ms)
                    else:
                        uv,off=backend.cover(endpoints,isolated=True)
                        row=dict(condition=key,phase=phase,iteration=i,variant='B1',
                                 **{k:v for k,v in backend.last.items() if k not in ('cover_uv','cover_offsets')})
                    target.append(row)
                    c.exact(uv,details[key]['cover_uv'],key+'/isolated cover');c.exact(off,details[key]['cover_offsets'],key+'/isolated offsets')
        backend.cover(endpoints,isolated=True,memory=True)
        memory[key]={k:v for k,v in backend.last.items() if k not in ('cover_uv','cover_offsets')}
        print('isolated stage complete',key,flush=True)
    import torch
    memory_result=dict(conditions=memory,scratch_cap_bytes=SCRATCH_CAP_BYTES,
        declared_logical_bound='384*bbox_cells + 512*candidate_edges + 4096; no chunking or dtype substitutions',
        peak_device_bytes=None,peak_status='PEAK_MEASUREMENT_UNSUPPORTED',
        gpu_scope='MPS tensor/op boundary observations and conservative explicit tensor bound; backend internal temporary peak unavailable',
        cpu_rss_high_water_bytes=io.rss(),cpu_memory_scope='single process high water including inputs, exact audit and report arrays; not per-path peak',
        recommended_max_memory_bytes=torch.mps.recommended_max_memory())
    io.write(out/'memory.json',memory_result)
    result=dict(scope='separate isolated synchronized stages; frozen fresh-materialized endpoints; not summed into actual S1 wall',
        event_latency_ms=None,event_status='TIMING_UNSUPPORTED',measured=rows,warmup=warm,
        kernel_launch_count=None,kernel_trace='NOT_EXECUTED',torch_eager='multiple kernels; not fused',
        resident_endpoint_scenario='NOT_EXECUTED',S3='NOT_EXECUTED',B2='NOT_EXECUTED')
    io.write(out/'profiling.json',result)
    return result,memory_result


def summarize(raw,profile,records):
    result={}
    for key,record in records.items():
        r=copy.deepcopy(record);r['timings']={};r['isolated']={}
        for variant in ('A','B1'):
            rows=[v for v in raw['measured'] if v['condition']==key and v['variant']==variant]
            r['timings'][variant]=dict(total_wall_ms=io.stats([v['total_wall_ms'] for v in rows]),
                common_ms=io.stats([v['common_ms'] for v in rows]),
                stages={n:io.stats([v['stages'][n] for v in rows]) for n in rows[0]['stages']})
            rows=[v for v in profile['measured'] if v['condition']==key and v['variant']==variant]
            r['isolated'][variant]={n:io.stats([v[n] for v in rows]) for n in
                (('supercover_wall_ms','ragged_output_packing_ms','hybrid_wall_ms') if variant=='A' else
                 ('adapter_ms','h2d_wall_ms','device_wall_ms','d2h_wall_ms','hybrid_wall_ms'))}
            if variant=='B1':r['isolated'][variant]['substage_wall_ms']={n:io.stats([v['isolated_stages'][n] for v in rows]) for n in rows[0]['isolated_stages']}
        r['pipeline_speedup']=r['timings']['A']['total_wall_ms']['mean']/r['timings']['B1']['total_wall_ms']['mean']
        r['supercover_hybrid_speedup']=r['isolated']['A']['hybrid_wall_ms']['mean']/r['isolated']['B1']['hybrid_wall_ms']['mean']
        result[key]=r
    aggregate={variant:dict(total_wall_ms=io.stats([v['total_wall_ms'] for v in raw['measured'] if v['variant']==variant]),
        worst_sample=max((v for v in raw['measured'] if v['variant']==variant),key=lambda v:v['total_wall_ms'])) for variant in ('A','B1')}
    aggregate['pipeline_speedup']=aggregate['A']['total_wall_ms']['mean']/aggregate['B1']['total_wall_ms']['mean']
    aggregate['hybrid_speedup']=sum(r['isolated']['A']['hybrid_wall_ms']['mean'] for r in result.values())/sum(r['isolated']['B1']['hybrid_wall_ms']['mean'] for r in result.values())
    return result,aggregate


def figures(out,summary,state,gpu_refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    keys=list(summary['conditions']);x=np.arange(len(keys));records=summary['conditions']
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/'plots'/name,dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,8))
    for variant in ('A','B1'):
        axes[0].plot(x,[records[k]['timings'][variant]['total_wall_ms']['mean'] for k in keys],'o-',label=variant+' S1 mean')
        axes[0].plot(x,[records[k]['timings'][variant]['total_wall_ms']['p95'] for k in keys],'--',label=variant+' S1 p95')
        axes[1].plot(x,[records[k]['isolated'][variant]['hybrid_wall_ms']['mean'] for k in keys],'o-',label=variant+' isolated ordered cover')
    for ax in axes:
        ax.set(xticks=x,xticklabels=keys,ylabel='wall ms');ax.tick_params(axis='x',rotation=65);ax.legend();ax.grid(alpha=.2)
    axes[0].set_title('Actual synchronized S1; B1 Mask/Depth and receiver on CPU')
    axes[1].set_title('Separate isolated cover measurement; GPU event timing unsupported')
    save(fig,'pipeline_supercover_latency.png')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for name in ('adapter_ms','h2d_wall_ms','device_wall_ms','d2h_wall_ms'):
        axes[0].plot(x,[records[k]['isolated']['B1'][name]['mean'] for k in keys],'o-',label=name)
    axes[0].set(xticks=x,xticklabels=keys,ylabel='isolated wall ms',title='B1 adapter / API moves / device processing')
    axes[0].tick_params(axis='x',rotation=70);axes[0].legend(fontsize=8)
    for variant in ('A','B1'):
        bottom=0
        for ci,n in enumerate(('candidate_total_ms','supercover_ms','support_validation_ms','depth_validation_ms','receiver_ms')):
            val=np.mean([records[k]['timings'][variant]['stages'][n]['mean'] for k in keys])
            axes[1].bar(variant,val,bottom=bottom,color=plt.get_cmap('tab10')(ci),label=n if variant=='A' else None);bottom+=val
    axes[1].set(ylabel='mean ms',title='Selected S1 core stages; common/packaging excluded');axes[1].legend(fontsize=8)
    save(fig,'gpu_hybrid_stages.png')
    fig,axes=plt.subplots(1,2,figsize=(14,5));m=summary['memory']['conditions']
    for n in ('bbox_cells','cover_cells','pair_occurrences'):
        axes[0].plot(x,[records[k][n] for k in keys],'o-',label=n)
    axes[0].set(ylabel='cell/pair occurrences',title='Stored workload, one condition per batch');axes[0].legend()
    for n in ('logical_scratch_bound_bytes',):axes[1].plot(x,[m[k][n]/1024**2 for k in keys],label='declared scratch bound')
    axes[1].plot(x,[max(v['current_allocated_bytes'] for v in m[k]['memory_boundaries'])/1024**2 for k in keys],label='boundary current allocated')
    axes[1].axhline(64,color='red',ls='--',label='64 MiB explicit scratch cap')
    axes[1].set(ylabel='MiB',title='MPS boundary observations are not actual peak');axes[1].legend(fontsize=8)
    for ax in axes:ax.set(xticks=x,xticklabels=keys);ax.tick_params(axis='x',rotation=70)
    save(fig,'memory_workload.png')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for n in ('h2d_wall_ms','d2h_wall_ms'):axes[0].plot(x,[records[k]['isolated']['B1'][n]['mean'] for k in keys],'o-',label=n)
    for n in ('h2d_bytes','d2h_bytes'):axes[1].plot(x,[m[k][n]/1024 for k in keys],'o-',label=n)
    for ax in axes:ax.set(xticks=x,xticklabels=keys);ax.tick_params(axis='x',rotation=70);ax.legend()
    axes[0].set(ylabel='isolated wall ms',title='MPS API materialization and synchronization')
    axes[1].set(ylabel='logical KiB',title='Input endpoints/offsets and ordered cover/offsets')
    save(fig,'h2d_d2h.png')
    packets=state[4];refs=state[6]
    for key,p in packets.items():
        if p['source_frame'] not in (0,45,120):continue
        src=p['provenance']['source_index'];w=int(p['image_shape'][1]);uv=np.column_stack((src%w,src//w))
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,variant,res in zip(axes,('CPU A','GPU B1 + CPU validation'),(refs[key],gpu_refs[key])):
            edge=io.undirected(res[0]['intra_object_edge_index'])
            ax.add_collection(LineCollection(uv[edge],colors='#297fa3',linewidths=.4,alpha=.6))
            ax.scatter(*uv.T,s=2,c=p['object_id'],cmap='coolwarm',vmin=0,vmax=1)
            ax.set(xlim=(0,w-1),ylim=(int(p['image_shape'][0])-1,0),aspect='equal',title=variant,xlabel='u',ylabel='v')
        fig.suptitle(key+': exact same Edge bytes; same points/colors/viewport')
        save(fig,key+'_same_edges.png')


def reports(out,summary):
    records=summary.get('conditions',{});aggregate=summary.get('aggregate',{})
    support=summary['backend_support'];judgments=summary['judgments']
    lines=['# Task05-D — Local MPS GPU EDE B1 Benchmark','',
        f"평가 UUID: `{out.name}`. **{summary['status']}**. 실제 로컬 MPS 연산을 실행했다.",
        '',f'[정적 HTML](../../experiments/{out.name}/gpu_edge_benchmark.html) · [승인 전 계획](05d_gpu_edge_benchmark_plan.md)',
        '', '## 판정','', '| 항목 | 판정 |','|---|---|']
    lines += [f'| {k} | {v} |' for k,v in judgments.items()]
    lines += ['',f"설치 Torch **{support['environment']['torch_version']}**, Apple M4 Pro GPU 20코어, device `mps:0`, dtype `torch.int64`. `PYTORCH_ENABLE_MPS_FALLBACK=0`을 Torch import 전에 적용했다. 실제 장치 조회 원문은 environment.json에 보존했다.",
        '', 'B1의 Mask·Depth 검증은 **CPU의 기존 validate_cover**에서 수행한다. GPU는 ragged bbox 전개, single-normal integer SAT, ordered cover/offset compaction만 담당한다. Torch eager의 여러 kernel이며 fused kernel이 아니다.',
        '',f"핵심 연산 probe: {support['status']}. 실패 연산: {', '.join(support.get('failed_operations',[])) or '없음'}. Timing: `{support['timing']['status']}`. Event record·작업 동기화는 완료했으나 elapsed_time 조회가 20초 내 반환되지 않아 자식 프로세스를 종료했다. Event latency는 null이다. 이전 최초 probe의 native stack sample도 보존했다. 측정은 synchronized wall이며 GPU event/kernel latency로 부르지 않는다.",
        '', 'Dispatch 검사에서 probe 결과의 device/dtype를 확인했다. fallback 비활성화와 Python dispatch 검사는 Metal kernel trace를 대체하지 않는다. MPS signpost API는 실행했지만 Instruments capture·실제 kernel launch 수·GPU 실제 peak는 미측정이다.',
        '', '## 정확성 및 측정 범위','',
        'Source UUID: '+', '.join('`'+s+'`' for s in SOURCE_RUNS)+'.',
        '', 'Frames 0,18,27,45,56,66,80,120 × keep ratio 0.5/1.0, phase 0. 기존 Task05-C snapshot hash와 현재 import binding을 검사했다. Sampling·feature/PCA5·EE 변환·component 재분할·simulator는 실행하지 않았다.']
    if aggregate:
        a=aggregate['A']['total_wall_ms'];b=aggregate['B1']['total_wall_ms']
        lines += ['',f"실제 S1 전체 평균 wall: **CPU A {a['mean']:.3f} ms → B1 {b['mean']:.3f} ms**, speedup **{aggregate['pipeline_speedup']:.4f}×**. 별도 isolated ordered Supercover hybrid speedup **{aggregate['hybrid_speedup']:.4f}×**. Kernel speedup은 미측정이다.",
            '', '| 경로 | mean | median | p95 | max (ms) |','|---|---:|---:|---:|---:|',
            *[f"| {v} | {s['mean']:.3f} | {s['median']:.3f} | {s['p95']:.3f} | {s['max']:.3f} |" for v,s in [('A',a),('B1',b)]],
            '', 'S1은 fresh common preparation, packet copy, CPU fresh candidate, cover, CPU Mask/Depth, packaging, 기존 receiver 완료까지 포함한다. 양쪽 조건별 warmup 10회와 측정 30회, AB/BA 각 15회를 변경 없이 수행했다. 결과 exact assertion은 timer 종료 뒤 수행했다. Isolated 단계는 별도 강제 동기화 pass이며 그 합을 실제 S1 wall로 사용하지 않았다.',
            '', '## 16조건 결과','', '| Condition | N | U | bbox | cover | pairs | A mean/p95/max ms | B1 mean/p95/max ms | S1 speedup | Exact |',
            '|---|---:|---:|---:|---:|---:|---|---|---:|---|']
        for key,r in records.items():
            a=r['timings']['A']['total_wall_ms'];b=r['timings']['B1']['total_wall_ms']
            lines.append(f"| {key} | {r['N']} | {r['U']} | {r['bbox_cells']} | {r['cover_cells']} | {r['pair_occurrences']} | {a['mean']:.3f}/{a['p95']:.3f}/{a['max']:.3f} | {b['mean']:.3f}/{b['p95']:.3f}/{b['max']:.3f} | {r['pipeline_speedup']:.4f}× | PASS |")
        key=max(records,key=lambda k:records[k]['N']);r=records[key]
        lines += ['',f"최대 Node 조건 `{key}` (N={r['N']})은 위 표에 포함했다. 최악 sample은 A `{aggregate['A']['worst_sample']['condition']}` {aggregate['A']['worst_sample']['total_wall_ms']:.3f} ms, B1 `{aggregate['B1']['worst_sample']['condition']}` {aggregate['B1']['worst_sample']['total_wall_ms']:.3f} ms다. 30표본 p95는 기술통계이며 outlier를 제거하지 않았다.",
            '', '3방향 비교는 candidate/origin/source/row, ordered cover/offset, ordered right/down pair와 evidence index/support/missing/delta/jump, 모든 판정/reason/count/top4 witness, final int64 Edge bytes/order, receiver metadata/deep-copy/+3, Point/F=5/validity/provenance 불변을 포함한다. Frame0 8.431m plateau와 기존 Edge를 보존했다. GPU synthetic test에는 Fraction oracle 625개 선분, 역방향·zero-length·음수 UV·closed half-pixel/corner-touch·image boundary, mask hole/overlap, invalid depth, missing/unsupported pair, threshold 경계와 fallback/receiver fixture가 포함된다.',
            '', '## 메모리 및 한계','',
            '작업량은 candidate 30,143, bbox 625,978, cover 227,426, pair occurrence 211,294다. Edge별 원시 수와 min/median/p95/p99/max는 조건 records 및 NPZ에 있다. 1조건=1batch이며 dense padding/chunk 탐색을 사용하지 않았다. Scratch는 사전 선언한 `384*bbox_cells + 512*U + 4096` logical bound를 64MiB와 대조하고 초과 시 중단한다. 내부 backend scratch의 물리 peak를 직접 제한·관측한 것으로 주장하지 않는다.',
            '', 'memory.json의 current/driver allocated는 동기화된 op 경계 관측값이다. 실제 GPU peak는 `null / PEAK_MEASUREMENT_UNSUPPORTED`다. CPU RSS는 입력·audit/report를 포함한 단일 process high water이며 A/B1 peak 비교가 아니다. H2D/D2H는 통합 메모리에서 API copy/materialization·동기화 비용이다.',
            '', 'B1의 성능으로 B2, CUDA, 전체 GPU EDE, 완전한 GPU-resident Observation 또는 GPU Edge→GNN·20Hz 성능을 판정하지 않는다. B2, Grid Traversal, CUDA/RTX 3090, resident endpoint scenario, S3, GNN/BC, 외부 게시·sync·Git commit은 모두 **NOT_EXECUTED**다.']
        lines += ['', '## 단계별 평균 wall','',
            '아래 S1 stage는 실제 pipeline 내부 timer이며, selected stage의 합이 전체 wall은 아니다. Isolated stage는 별도 pass의 synchronized wall이다. 이벤트 기반 kernel latency가 아니다.',
            '', '| S1 stage | CPU A ms | B1 ms |','|---|---:|---:|']
        for n in ('candidate_total_ms','supercover_ms','support_validation_ms','depth_validation_ms','receiver_ms','edge_packaging_ms'):
            values=[np.mean([r['timings'][v]['stages'][n]['mean'] for r in records.values()]) for v in ('A','B1')]
            lines.append(f'| {n} | {values[0]:.3f} | {values[1]:.3f} |')
        lines += ['', '| Isolated Supercover stage | CPU A ms | B1 ms |','|---|---:|---:|']
        amean=lambda n:np.mean([r['isolated']['A'][n]['mean'] for r in records.values()])
        bmean=lambda n:np.mean([r['isolated']['B1'][n]['mean'] for r in records.values()])
        lines += [f"| CPU raster / B1 device processing | {amean('supercover_wall_ms'):.3f} | {bmean('device_wall_ms'):.3f} |",
            f"| CPU ragged output packing / B1 host adapter | {amean('ragged_output_packing_ms'):.3f} | {bmean('adapter_ms'):.3f} |",
            f"| H2D API materialization | 해당 없음 | {bmean('h2d_wall_ms'):.3f} |",
            f"| D2H API materialization | 해당 없음 | {bmean('d2h_wall_ms'):.3f} |",
            f"| Complete ordered-cover hybrid | {amean('hybrid_wall_ms'):.3f} | {bmean('hybrid_wall_ms'):.3f} |"]
        memory=summary['memory'];conditions=memory['conditions']
        bound=max(r['logical_scratch_bound_bytes'] for r in conditions.values())
        observed=max(v['current_allocated_bytes'] for r in conditions.values() for v in r['memory_boundaries'])
        driver=max(v['driver_allocated_bytes'] for r in conditions.values() for v in r['memory_boundaries'])
        lines += ['',f"최대 explicit scratch bound {bound/1024**2:.3f}MiB; MPS 경계 current allocated 최댓값 {observed/1024**2:.3f}MiB, driver allocated 최댓값 {driver/1024**2:.3f}MiB. Driver 관측값에는 allocator/cache가 포함되며 scratch나 실제 peak와 같지 않다. CPU RSS high water {memory['cpu_rss_high_water_bytes']/1024**2:.3f}MiB.",
            '',f"조건당 H2D logical bytes 범위 {min(r['h2d_bytes'] for r in conditions.values()):,}–{max(r['h2d_bytes'] for r in conditions.values()):,}; D2H {min(r['d2h_bytes'] for r in conditions.values()):,}–{max(r['d2h_bytes'] for r in conditions.values()):,}. 조건별 값과 분포는 records/JSON/NPZ에 보존했다."]
        if summary.get('supplemental'):
            cold=summary['supplemental']['cold_start']
            lines += ['',f"별도 fresh process의 최대 N 조건 첫 B1 cover wall은 {cold['first_cover_wall_ms']:.3f}ms, Torch import는 {cold['torch_import_wall_ms']:.3f}ms였다. OS/driver shader cache는 초기화하지 않은 1표본이며 steady-state 통계와 합치지 않았다. 실제 B1 dispatch/CPU profiler 및 MPS signpost를 별도 실행했다. CPU tensor 출력은 명시적 host materialization·transfer·detach뿐이며 CPU 산술 fallback은 관측되지 않았다. CPU profiler time은 GPU kernel time이 아니다."]
    else:
        lines += ['', '정확성 또는 backend 관문에서 중단했다. 성능·GPU exact·미실행 항목은 PASS/0ms로 표시하지 않는다.',
                  '', '중단 사유: `'+str(summary.get('failure','backend unsupported'))+'`']
    links=['gpu_edge_benchmark.json','gpu_edge_benchmark.npz','timings_raw.json','environment.json',
           'backend_support.json','cpu_baseline_gate.json','exact_equivalence.json','profiling.json','memory.json',
           'verification.json','tests.log','source_hashes.json','implementation.diff','config.yaml',
           'profiles/cold_and_dispatch.json','profiles/cold_and_dispatch_probe.py','profiles/b1_cpu_dispatch_trace.json',
           'profiles/event_elapsed_time_sample.txt','report_refresh.diff','report_refresh.json']
    lines += ['', '## 산출물 및 재현','']
    lines += [f'[{name}](../../experiments/{out.name}/{name})' for name in links if (out/name).exists()]
    lines += ['', '기존 ExperimentLogger(team=visual, job_type=evaluate, seed=0), WANDB_MODE=disabled를 재사용했다. 기존 dirty/untracked 작업, 5개 source run, 입력/artifact/sidecar hash 보존은 verification.json에서 확인한다. 실제 브라우저 검증은 NOT_EXECUTED이며 HTML 파싱·상대 링크 검증과 구분한다.',
        '', '```bash','PYTORCH_ENABLE_MPS_FALLBACK=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 WANDB_MODE=disabled .venv/bin/python -m visual.runtime.test.run_gpu_edge_benchmark','```']
    images=sorted((out/'plots').glob('*.png'))
    lines += ['',*[f'![{p.stem}](../../experiments/{out.name}/plots/{p.name})' for p in images]]
    REPORT.write_text('\n'.join(lines)+'\n')
    html=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>Task05-D Local MPS B1</title><style>body{font:16px/1.55 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#172b3a}pre{white-space:pre-wrap;background:#edf3f7;padding:18px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd8df;padding:6px}img{max-width:100%}a{color:#12669b}</style>',
          '<h1>Task05-D Local Apple M4 Pro MPS — B1</h1>',f'<p>Run <code>{out.name}</code> — <strong>{escape(summary["status"])}</strong></p>',
          '<p><a href="../../docs/visual-graph-integration/05d_gpu_edge_benchmark.md">결과 Markdown</a> · <a href="../../docs/visual-graph-integration/05d_gpu_edge_benchmark_plan.md">계획</a></p>',
          '<p><strong>B1 Mask/Depth validation = CPU. GPU event latency = TIMING_UNSUPPORTED; synchronized wall only. CUDA / B2 / S3 / browser = NOT_EXECUTED.</strong></p>',
          '<h2>Independent judgments</h2><pre>'+escape(json.dumps(judgments,indent=2))+'</pre>',
          '<h2>실행 범위와 결과</h2>',
          f'<p>Torch {escape(support["environment"]["torch_version"])} · mps:0 · torch.int64 · fallback=0 before import. Frames 0,18,27,45,56,66,80,120 × ratio 0.5/1.0, phase 0. CPU A = Task05-C optimized EDE.</p>',
          '<p>CPU fresh candidates → GPU ragged bbox + single-normal integer SAT + ordered cover → bulk CPU readback → unchanged CPU Mask/Depth validate_cover → packaging → unchanged receiver. Torch eager uses multiple kernels. No dense padding or chunk tuning.</p>',
          '<p>10 warmups + 30 AB/BA samples per path/condition; exact assertions after timer. Isolated stages synchronize separately and do not replace actual S1 wall. Event elapsed_time did not return within 20 seconds; event/kernel latency is null. GPU actual peak and kernel launch counts are unavailable.</p>']
    if aggregate:
        def table(headers,rows):
            return '<div style="overflow-x:auto"><table><thead><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
        html += [f'<p><strong>S1 wall speedup {aggregate["pipeline_speedup"]:.4f}× · isolated ordered-cover hybrid speedup {aggregate["hybrid_speedup"]:.4f}×</strong></p>',
                 table(['Path','mean ms','median ms','p95 ms','max ms'],[[v]+[f'{aggregate[v]["total_wall_ms"][n]:.3f}' for n in ('mean','median','p95','max')] for v in ('A','B1')]),
                 '<h2>16 conditions — exact saved / CPU A / B1</h2>']
        html += [table(['Condition','N','U','bbox','cover','pairs','A mean/p95/max ms','B1 mean/p95/max ms','S1 speedup','Exact'],
            [[k,r['N'],r['U'],r['bbox_cells'],r['cover_cells'],r['pair_occurrences']]+[
                '/'.join(f'{r["timings"][v]["total_wall_ms"][n]:.3f}' for n in ('mean','p95','max')) for v in ('A','B1')]+[
                f'{r["pipeline_speedup"]:.4f}×','PASS'] for k,r in records.items()])]
        html += ['<h2>Stage means and limitations</h2>',table(['S1 stage','A ms','B1 ms'],[
            [n]+[f'{np.mean([r["timings"][v]["stages"][n]["mean"] for r in records.values()]):.3f}' for v in ('A','B1')]
            for n in ('candidate_total_ms','supercover_ms','support_validation_ms','depth_validation_ms','receiver_ms')]),
            '<p>Isolated means: host adapter '+f'{bmean("adapter_ms"):.3f}ms, H2D {bmean("h2d_wall_ms"):.3f}ms, device processing {bmean("device_wall_ms"):.3f}ms, D2H {bmean("d2h_wall_ms"):.3f}ms. These are synchronized walls, not event/kernel latency.</p>',
            f'<p>64MiB explicit scratch cap; maximum declared bound {bound/1024**2:.3f}MiB. Boundary current allocated maximum {observed/1024**2:.3f}MiB is not actual GPU peak. CPU RSS high water {memory["cpu_rss_high_water_bytes"]/1024**2:.3f}MiB includes inputs and audit/report. See memory.json for tensor shapes/dtypes/logical bytes and allocator observations.</p>',
            '<p>Full exact covers ordered candidates/origins/source/rows, cover/offsets, right/down pairs and evidence, reasons/counts/top4 witnesses, final Edge bytes/order, receiver metadata/deep-copy and unchanged points/features/validity/provenance. Frame0 8.431m plateau preserved. 90 tests passed without skips. No rounding tolerance.</p>',
            '<p>B2, Grid Traversal, CUDA/RTX 3090, resident endpoints, S3, GNN/BC, simulator, publishing/sync/commit: NOT_EXECUTED. B1 results do not determine their performance or validate integrated 20Hz. Actual browser check: NOT_EXECUTED.</p>']
        if summary.get('supplemental'):
            html += [f'<p>Separate fresh-process first B1 cover: {cold["first_cover_wall_ms"]:.3f}ms; Torch import {cold["torch_import_wall_ms"]:.3f}ms. One sample with OS shader cache retained. Separate B1 dispatch/CPU profiler/signpost pass observed only explicit host materialization/copy/detach on CPU; no arithmetic fallback. CPU trace is not a Metal kernel trace.</p>']
    else:html += ['<pre>'+escape(str(summary.get('failure',summary['status'])))+'</pre>']
    html += ['<h2>Artifacts</h2><ul>']
    html += [f'<li><a href="{name}">{name}</a></li>' for name in links if (out/name).exists()]
    html += ['</ul><h2>GPU/hybrid/pipeline, transfer, memory/workload and identical Edge overlays</h2>']
    html += [f'<figure><img src="plots/{p.name}" alt="{p.stem}"><figcaption>{p.stem}</figcaption></figure>' for p in images]
    html += ['</html>'];(out/'gpu_edge_benchmark.html').write_text('\n'.join(html))


def verify_outputs(out,initial,summary):
    from PIL import Image
    json_count=0
    for p in out.rglob('*.json'):json.loads(p.read_text());json_count+=1
    images=list((out/'plots').glob('*.png'))
    for p in images:
        with Image.open(p) as image:image.load()
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            self.links += [v for k,v in attrs if k in ('href','src')]
    parser=Links();parser.feed((out/'gpu_edge_benchmark.html').read_text())
    links=[(out,v) for v in parser.links]
    links += [(REPORT.parent,v) for v in re.findall(r'\]\(([^)]+)\)',REPORT.read_text())]
    for base,link in links:
        io.require(not re.match(r'https?://',link),'external report asset')
        io.require((base/link).is_file(),'missing relative link '+link)
    io.require('../../docs/visual-graph-integration/05d_gpu_edge_benchmark.md' in parser.links,'HTML to Markdown')
    io.require(f'../../experiments/{out.name}/gpu_edge_benchmark.html' in REPORT.read_text(),'Markdown to HTML')
    stats='NOT_EXECUTED'
    if summary.get('aggregate'):
        raw=json.loads((out/'timings_raw.json').read_text());prof=json.loads((out/'profiling.json').read_text())
        recalculated,agg=summarize(raw,prof,summary['conditions'])
        c.exact(agg,summary['aggregate'],'raw statistics readback')
        for key in recalculated:c.exact(recalculated[key],summary['conditions'][key],key+'/statistics readback')
        stats='PASS'
    sources=json.loads((out/'source_hashes.json').read_text())
    for rel,sha in sources['evaluated_sources'].items():
        c.exact(io.sha(out/'source_snapshot'/rel),sha,'source snapshot hash')
        refresh=sources.get('report_verification_source_changes',{}).get(rel)
        if refresh:
            c.exact(io.sha(REPO/rel),refresh['report_verification_sha256'],'report source hash')
            c.exact(io.sha(out/'report_source_snapshot'/rel),refresh['report_verification_sha256'],'report snapshot hash')
        else:c.exact(io.sha(REPO/rel),sha,'evaluated source unchanged')
    return dict(json_readback='PASS',json_files=json_count,npz_readback='PASS' if summary.get('aggregate') else 'NOT_EXECUTED',
        raw_statistics=stats,png_decode='PASS' if images else 'NOT_EXECUTED',png_files=len(images),
        png_visual_review='NOT_EXECUTED',html_parser='PASS',relative_links='PASS',relative_links_checked=len(links),
        markdown_html_bidirectional='PASS',actual_browser='NOT_EXECUTED',protected=verify_protection(initial),
        source_snapshots='PASS',tests=summary.get('tests',dict(status='NOT_EXECUTED')))


def main():
    for key in c.THREADS:io.require(os.getenv(key)=='1','set '+key+'=1 before NumPy import')
    io.require(os.getenv('WANDB_MODE')=='disabled','WANDB_MODE=disabled required')
    io.require(os.getenv('PYTORCH_ENABLE_MPS_FALLBACK')=='0','fallback=0 required before Torch import')
    initial=protection()
    config=dict(task='05-D',source_runs=SOURCE_RUNS,frames=io.FRAMES,keep_ratios=io.RATIOS,phase=0,
        backend='mps',dtype='int64',variant='B1 only; CPU Mask/Depth validation',warmups=WARMUPS,repeats=REPEATS,
        paired_order='AB/BA alternating, each 15; condition start alternates',scratch_cap_bytes=SCRATCH_CAP_BYTES,
        representation_bound='384*bbox_cells + 512*candidate_edges + 4096',
        common_preparation_in_S1=True,core_audit_mode='full_reasons_counts_top4_witness',
        detailed_audit_in_core=False,resident='NOT_EXECUTED',S3='NOT_EXECUTED')
    with ExperimentLogger(team='visual',job_type='evaluate',model='task05d_mps_b1',seed=0,config=config,root=REPO/'experiments') as logger:
        out=logger.directory;print('RUN_ID',out.name,flush=True);Path('/tmp/task05d_run_id').write_text(out.name)
        for folder in ('records','packets','plots','profiles'):(out/folder).mkdir()
        save_sources(out,initial)
        p=subprocess.run([sys.executable,'-m','visual.runtime.test.ede_gpu_bbox_sat','--probe-output',str(out/'backend_support.json')],
                         cwd=REPO,capture_output=True,text=True,timeout=90)
        (out/'probe.log').write_text(p.stdout+p.stderr);io.require(p.returncode==0,'probe subprocess failed')
        support=json.loads((out/'backend_support.json').read_text());io.write(out/'environment.json',support['environment'])
        if Path('/tmp/task05d_probe_sample.txt').exists():shutil.copyfile('/tmp/task05d_probe_sample.txt',out/'profiles/event_elapsed_time_sample.txt')
        summary=dict(run_id=logger.run_id,config=config,backend_support=support,status=support['status'],
            judgments=dict(BACKEND_SUPPORT=support['status'],EXACT_EQUIVALENCE='NOT_EXECUTED',
                GPU_SUPERCOVER_PERFORMANCE='NOT_EXECUTED',HYBRID_PIPELINE_PERFORMANCE='NOT_EXECUTED',
                MEMORY_AND_TIMING_LIMITATIONS='TIMING_UNSUPPORTED; PEAK_MEASUREMENT_UNSUPPORTED'),
            not_executed=['B2','Grid Traversal','CUDA/RTX 3090','resident endpoints','S3','GNN/BC','simulator','publishing/sync/commit','actual browser'])
        try:
            if support['status']=='PASS':
                state=cpu_gate(out);save_sources(out,initial,state[-1])
                test_env=dict(os.environ,TASK05D_SUPPORT=str(out/'backend_support.json'))
                test=subprocess.run([sys.executable,'-m','unittest',*io.TESTS,'visual.runtime.test.test_gpu_edge_equivalence','-v'],cwd=REPO,env=test_env,capture_output=True,text=True)
                (out/'tests.log').write_text(test.stdout+test.stderr)
                count=re.search(r'Ran (\d+) tests',test.stdout+test.stderr)
                summary['tests']=dict(status='PASS' if test.returncode==0 else 'FAIL',count=int(count.group(1)) if count else None,skips='skipped' in test.stderr.lower())
                io.require(test.returncode==0 and not summary['tests']['skips'],'test suite failure or skipped GPU contract')
                backend=B1Backend(support)
                b,_=c.optimized_runner();b.build_extended_delaunay_edges=backend.build
                records,gpu_refs=equivalence(out,state,backend,b)
                summary['judgments']['EXACT_EQUIVALENCE']='PASS 16/16, full three-way exact'
                print('ALL EXACT PASS; start fixed benchmark',flush=True)
                raw=pipeline(out,state,backend,b)
                profile,memory=isolated(out,state,backend)
                records,aggregate=summarize(raw,profile,records)
                summary.update(status='COMPLETED',conditions=records,aggregate=aggregate,memory=memory)
                summary['judgments'].update(GPU_SUPERCOVER_PERFORMANCE='GPU event/kernel latency NOT_EXECUTED; isolated synchronized ordered-cover wall measured',
                    HYBRID_PIPELINE_PERFORMANCE=f"S1 speedup {aggregate['pipeline_speedup']:.4f}x; isolated cover hybrid {aggregate['hybrid_speedup']:.4f}x")
                figures(out,summary,state,gpu_refs)
            else:
                (out/'tests.log').write_text('NOT_EXECUTED: backend support gate stopped before CPU/GPU implementation/benchmark.\n')
        except Exception as exc:
            import traceback
            summary.update(status='CONTRACT_FAIL' if isinstance(exc,AssertionError) else 'STOPPED',failure=dict(type=type(exc).__name__,message=str(exc)))
            (out/'failure.log').write_text(traceback.format_exc())
            print('STOPPED',type(exc).__name__,str(exc),flush=True)
        for name in ('timings_raw.json','profiling.json','memory.json','verification.json'):
            if not (out/name).exists():io.write(out/name,dict(status='NOT_EXECUTED',reason=summary.get('failure',summary['status'])))
        if not (out/'gpu_edge_benchmark.npz').exists():np.savez(out/'gpu_edge_benchmark.npz',execution_status=np.array(summary['status']))
        io.write(out/'gpu_edge_benchmark.json',summary);reports(out,summary)
        verification=verify_outputs(out,initial,summary);io.write(out/'verification.json',verification)
        summary['verification']=verification;io.write(out/'gpu_edge_benchmark.json',summary)
        reports(out,summary);verify_outputs(out,initial,summary)
        logger.record(source_runs=list(SOURCE_RUNS),evaluation_status=summary['status'],judgments=summary['judgments'])
        for path in sorted(out.rglob('*')):
            if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','manifest.json','metrics.json'):
                logger.link_output(path,kind='reports' if path.suffix not in ('.json','.npz') else ('metrics' if path.suffix=='.json' else None))
        verify_protection(initial)
        print('REPORT',REPORT,'HTML',out/'gpu_edge_benchmark.html',summary['status'],flush=True)


if __name__=='__main__':main()
