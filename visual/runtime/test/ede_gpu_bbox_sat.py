"""Task05-D evaluation-only MPS exact-integer capability gate.

Run in a fresh process with PYTORCH_ENABLE_MPS_FALLBACK=0 before Torch import.
No alternate dtype, CPU fallback, extension, or dependency installation is used.
B1 is evaluation-only and requires the successful separate-process gate.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter


def integer_guard(m):
    """Same conservative Python-integer domain as Task05-C, without fallback."""
    m = abs(int(m))
    return 8*m*m + 4*m <= 2**63-1


def timing_probe():
    """Event APIs may hang in native code: isolate and bound the capability test."""
    code = '''
import json,torch
from time import perf_counter
assert torch.backends.mps.is_available()
x=torch.arange(4096,dtype=torch.int64,device='mps');torch.mps.synchronize()
a=torch.mps.Event(enable_timing=True);b=torch.mps.Event(enable_timing=True)
tick=perf_counter();a.record();y=x*x+x;b.record();b.synchronize();torch.mps.synchronize()
wall_ms=(perf_counter()-tick)*1000
assert y.device.type=='mps' and torch.equal(y.cpu(),torch.arange(4096)**2+torch.arange(4096))
print('WORK_SYNCHRONIZED',flush=True)
event=a.elapsed_time(b)
print('WORK_EVENT_MS',event,flush=True)
a.record();b.record();b.synchronize();torch.mps.synchronize()
empty=a.elapsed_time(b)
print(json.dumps(dict(status='PASS',probe_event_ms=event,empty_event_ms=empty,synchronized_wall_ms=wall_ms,query=b.query())),flush=True)
'''
    try:
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=20)
        if p.returncode: raise RuntimeError(p.stderr)
        result=json.loads(p.stdout.splitlines()[-1])
        assert all(math.isfinite(result[k]) and result[k]>=0 for k in ('probe_event_ms','empty_event_ms'))
        return result
    except subprocess.TimeoutExpired as exc:
        return dict(status='TIMING_UNSUPPORTED',exception_type='TimeoutExpired',
            exception='MPS event capability subprocess did not finish within 20s; terminated; no event latency reported',
            stdout=(exc.stdout or b'').decode() if isinstance(exc.stdout,bytes) else exc.stdout,
            event_latency_ms=None)
    except Exception as exc:
        return dict(status='TIMING_UNSUPPORTED',exception_type=type(exc).__name__,exception=str(exc),event_latency_ms=None)


def probe():
    if 'torch' in sys.modules or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('fresh process and fallback=0 required before importing Torch')
    tick = perf_counter()
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    import_ms = (perf_counter()-tick)*1000

    def command(args):
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())

    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               torch_version=torch.__version__, torch_path=torch.__file__,
               mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available(),
               cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
               fallback_env=os.environ['PYTORCH_ENABLE_MPS_FALLBACK'],
               system_profiler=command(['/usr/sbin/system_profiler', 'SPHardwareDataType', 'SPDisplaysDataType', '-json']),
               thread_environment={k:os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    report = dict(environment=env, torch_import_wall_ms=import_ms, operations=[],
                  scope='B1 int64 bbox/SAT/ordered compaction only; Mask/Depth stay on CPU',
                  fallback_inspection='fallback=0 before Torch import; output device assertions; Python dispatcher trace. Not a Metal kernel trace.',
                  cuda='NOT_EXECUTED', b2='NOT_EXECUTED')
    if not env['mps_available']:
        report.update(status='GPU_UNAVAILABLE', timing=dict(status='NOT_EXECUTED'), memory=dict(status='NOT_EXECUTED'))
        return report
    torch.use_deterministic_algorithms(True)
    trace = []

    def tensors(x):
        if isinstance(x, torch.Tensor): return [x]
        if isinstance(x, (tuple,list)): return [t for v in x for t in tensors(v)]
        if isinstance(x, dict): return [t for v in x.values() for t in tensors(v)]
        return []

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            trace.append(dict(op=str(func), inputs=[str(t.device) for t in tensors(args)],
                              outputs=[str(t.device) for t in tensors(result)],
                              dtypes=[str(t.dtype) for t in tensors(result)]))
            return result

    def t(values, device, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device=device)

    def check(name, fn, description):
        row = dict(name=name, description=description, required=True)
        start = len(trace)
        try:
            expected = fn('cpu')
            with Trace(): actual = fn('mps')
            torch.mps.synchronize()
            at, et = tensors(actual), tensors(expected)
            assert at and len(at) == len(et)
            row['outputs'] = []
            for a,e in zip(at,et):
                row['outputs'].append(dict(device=str(a.device), dtype=str(a.dtype), shape=list(a.shape)))
                assert a.device.type == 'mps', 'non-MPS result'
                assert a.dtype == e.dtype and a.shape == e.shape, 'dtype/shape mismatch'
                cpu = a.cpu()
                assert torch.equal(cpu,e), f'exact mismatch: actual={cpu.tolist()} expected={e.tolist()}'
            row.update(status='PASS', expected=[e.tolist() for e in et], actual=[a.cpu().tolist() for a in at])
        except Exception as exc:
            row.update(status='FAIL', exception_type=type(exc).__name__, exception=str(exc))
        row['dispatch'] = trace[start:]
        report['operations'].append(row)
        print(name,row['status'],row.get('exception',''),flush=True)

    extrema = [-(2**63),-(2**53)-1,-(2**32)-1,-(2**31)-1,-1,0,1,2**31,2**32+1,2**53+1,2**63-1]
    check('int64_create_gather_roundtrip', lambda d: t(extrema,d)[t([10,0,9,1,8,2,7,3,6,4,5],d)],
          'int64 extrema and >2^53 values; int64 gather; exact host roundtrip')
    check('integer_add_sub_mul_neg_abs', lambda d: (
        t([2**53+1,-2**53-1,2**31+3],d)+t([3,-3,-7],d),
        t([2**53+1,-2**53-1,2**31+3],d)-t([3,-3,-7],d),
        t([2**30-1,-2**30+1,2**31+3],d)*t([2**30-1,2**30-1,-7],d),
        -t([2**62,-2**62,0],d), t([-2**62,0,2**62],d).abs()),
          'exact signed arithmetic including products beyond 32-bit/53-bit; no overflow inputs')
    m = math.isqrt((2**63-1)//8)
    while not integer_guard(m): m -= 1
    report['overflow_guard'] = dict(max_safe_m=m, safe_bound=8*m*m+4*m,
        next_m=m+1, next_bound=8*(m+1)**2+4*(m+1),
        safe=integer_guard(m), next_rejected=not integer_guard(m+1), int64_max=2**63-1)
    check('SAT_integer_multiply_sum', lambda d: (
        (t([[2*m,2*m],[-2*m,-2*m],[2*m,-2*m]],d)*t([2*m,2*m],d)).sum(1),
        (t([[2*m,2*m],[-2*m,-2*m]],d)*t([2*m,2*m],d)).sum(1)+t([4*m,-4*m],d)),
          'SAT dot and c+/-radius at Task05-C conservative int64 overflow-guard boundary')
    check('inclusive_comparison_boolean_and', lambda d: (
        (t([2**53,2**53+1,2**53+2],d)>=2**53+1)&(t([2**53,2**53+1,2**53+2],d)<=2**53+1),
        t([True,False,True],d,torch.bool)&t([True,True,False],d,torch.bool)),
          'inclusive equality and adjacent large integers; boolean AND')
    check('min_max_reduce', lambda d: (t(extrema,d).min(),t(extrema,d).max()), 'int64 min/max extrema')
    check('integer_division_remainder', lambda d: (
        torch.div(t([-2**53-1,0,2**53+1],d),3,rounding_mode='floor'),
        t([-2**53-1,0,2**53+1],d).remainder(3)), 'ragged row/column integer floor division/remainder')

    def ragged(d):
        offsets=t([0,2,2,8],d); flat=torch.arange(8,device=d,dtype=torch.int64)
        edge=torch.searchsorted(offsets[1:],flat,right=True)
        local=flat-offsets[edge]; width=t([2,1,3],d)[edge]
        uv=torch.stack((local%width+t([-2,0,7],d)[edge],
                        torch.div(local,width,rounding_mode='floor')+t([1,0,-1],d)[edge]),1)
        return edge,local,uv
    check('ragged_bbox_searchsorted_indexing',ragged,'empty middle bbox, negative UV, row-major flat indices')
    check('cumsum_int64',lambda d:t([2**53+1,0,2**31,1],d).cumsum(0), 'int64 prefix sum beyond float precision')

    def compact(d):
        keep=t([True,False,True,True,False,True],d,torch.bool)
        prefix=keep.to(torch.int64).cumsum(0)
        uv=torch.arange(12,device=d,dtype=torch.int64).reshape(6,2)
        ids=torch.arange(6,device=d,dtype=torch.int64)[keep]
        boundaries=t([0,2,2,6],d)
        padded=torch.cat((t([0],d),prefix))
        return uv[keep],ids,prefix,padded[boundaries]
    check('ordered_boolean_compaction_offsets',compact,'boolean indexing preserves row-major order; empty edge offsets')
    check('noncontiguous_gather',lambda d:t([[1,2,3],[4,5,6]],d).T[t([2,0],d)],'noncontiguous int64 gather')
    check('empty_sentinel',lambda d:(t([],d),t([],d).cumsum(0),
        torch.searchsorted(t([],d),t([-1,0,9],d)),
        t([-1,0,9],d)[t([True,False,True],d,torch.bool)]), 'empty scan/search/roundtrip and sentinel -1 retained')
    report['timing']=timing_probe()
    try:
        torch.mps.synchronize()
        report['memory']=dict(status='PASS',current_allocated_bytes=torch.mps.current_allocated_memory(),
            driver_allocated_bytes=torch.mps.driver_allocated_memory(),
            recommended_max_bytes=torch.mps.recommended_max_memory(),
            peak_device_bytes=None,peak_status='PEAK_MEASUREMENT_UNSUPPORTED',
            scope='probe boundary observation, not actual GPU peak',scratch_cap_bytes=64*1024**2)
    except Exception as exc:
        report['memory']=dict(status='MEMORY_QUERY_UNSUPPORTED',exception_type=type(exc).__name__,exception=str(exc))
    try:
        with torch.mps.profiler.profile(mode='interval,event',wait_until_completed=False):
            x=t([2**53+1,3],'mps'); y=x+1; torch.mps.synchronize()
        assert y.device.type=='mps'
        report['profiler']=dict(status='SIGNPOST_API_PASS',kernel_trace='NOT_EXECUTED',kernel_launch_count=None,
            activities=[str(a) for a in torch.profiler.supported_activities()],
            limitation='OS Signpost API executed; no Instruments capture; dispatcher trace is not kernel-level evidence')
    except Exception as exc:
        report['profiler']=dict(status='PROFILER_UNSUPPORTED',exception_type=type(exc).__name__,exception=str(exc))
    failed=[r['name'] for r in report['operations'] if r['status']!='PASS']
    report.update(status='EXACT_GPU_PATH_UNSUPPORTED' if failed else 'PASS',failed_operations=failed,
                  dispatch_trace_entries=len(trace),strict_deterministic_algorithms=True,
                  initialization_scope='Torch import and capability-probe initialization only; no B1 cold/steady measurement')
    return report


SCRATCH_CAP_BYTES = 64*1024**2


def host_adapter(endpoints):
    """Only endpoint/bbox size metadata. Never compute SAT or cover on host."""
    import numpy as np
    p=np.asarray(endpoints)
    if p.dtype != np.int64 or p.ndim != 3 or p.shape[1:] != (2,2):
        raise ValueError('int64 [U,2,2] endpoints required')
    if not all(integer_guard(v) for v in p.ravel()):
        raise ValueError('EXACT_GPU_PATH_UNSUPPORTED: endpoint exceeds Task05-C int64 guard')
    sizes=np.abs(p[:,1]-p[:,0])+1
    counts=sizes[:,0]*sizes[:,1]
    total=sum(map(int,counts))
    # Declared before measurement: conservative sum of explicit tensors including
    # all gathered endpoint/SAT operands, compare temporaries and worst-case cover.
    # Internal Metal allocator/sort scratch is not claimed to be measured here.
    logical_bound=384*total+512*len(p)+4096
    if logical_bound>SCRATCH_CAP_BYTES:
        raise MemoryError('SCRATCH_LIMIT_EXCEEDED: no automatic chunking or fallback')
    offsets=np.empty(len(p)+1,np.int64);offsets[0]=0
    np.cumsum(counts,out=offsets[1:])
    return dict(endpoints=p.copy(),offsets=offsets,total=total,
                logical_scratch_bound_bytes=logical_bound,
                h2d_bytes=p.nbytes+offsets.nbytes)


def device_cover(inputs,total,observe=None):
    """Ragged int64 bbox -> single-normal SAT -> ordered cover; Torch eager.

    No U*H*W/U*max_bbox padding, per-edge dispatch, host SAT or host compaction.
    Dynamic boolean indexing may synchronize internally; its full cost is kept.
    """
    import torch
    p,offsets=inputs
    assert p.device.type=='mps' and offsets.device.type=='mps'
    assert p.dtype==offsets.dtype==torch.int64
    def mark(stage,values):
        if observe is not None: observe(stage,values)
    low=torch.minimum(p[:,0],p[:,1]);sizes=(p[:,1]-p[:,0]).abs()+1
    flat=torch.arange(total,dtype=torch.int64,device=p.device)
    edge=torch.searchsorted(offsets[1:],flat,right=True)
    local=flat-offsets[edge]
    widths=sizes[:,0][edge]
    u=local%widths+low[:,0][edge]
    v=torch.div(local,widths,rounding_mode='floor')+low[:,1][edge]
    uv=torch.stack((u,v),1)
    mark('bbox',locals())
    delta=p[:,1]-p[:,0]
    axis=torch.stack((-delta[:,1],delta[:,0]),1)
    projection=(2*p*axis[:,None,:]).sum(2)
    lower=torch.minimum(projection[:,0],projection[:,1])
    upper=torch.maximum(projection[:,0],projection[:,1])
    radius=axis.abs().sum(1)
    centre=2*(uv*axis[edge]).sum(1)
    keep=(centre+radius[edge]>=lower[edge])&(centre-radius[edge]<=upper[edge])
    mark('sat',locals())
    prefix=keep.to(torch.int64).cumsum(0)
    prefix0=torch.cat((torch.zeros(1,dtype=torch.int64,device=p.device),prefix))
    cover_uv=uv[keep]
    cover_offsets=prefix0[offsets]
    mark('compaction',locals())
    assert cover_uv.device.type==cover_offsets.device.type=='mps'
    return cover_uv,cover_offsets


class B1Backend:
    """All Mask/Depth validation remains CPU; no final GPU Edge/S3 path."""
    def __init__(self,support):
        if support.get('status')!='PASS':
            raise RuntimeError('EXACT_GPU_PATH_UNSUPPORTED: separate probe must pass')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':
            raise RuntimeError('fallback=0 must be set before Torch import')
        self.last=None

    def cover(self,endpoints,*,isolated=False,memory=False):
        import numpy as np
        import torch
        start=perf_counter();host=host_adapter(endpoints);adapter=(perf_counter()-start)*1000
        stages={};boundaries=[]
        if isolated:torch.mps.synchronize()
        tick=perf_counter()
        inputs=(torch.tensor(host['endpoints'],device='mps',dtype=torch.int64),
                torch.tensor(host['offsets'],device='mps',dtype=torch.int64))
        if isolated:torch.mps.synchronize()
        h2d=(perf_counter()-tick)*1000
        checkpoint=perf_counter()
        def observe(stage,values):
            nonlocal checkpoint
            torch.mps.synchronize();now=perf_counter()
            stages[stage+'_wall_ms']=(now-checkpoint)*1000;checkpoint=now
            if memory:
                tensors={k:v for k,v in values.items() if isinstance(v,torch.Tensor)}
                unique={v.data_ptr():v for v in tensors.values()}
                boundaries.append(dict(stage=stage,current_allocated_bytes=torch.mps.current_allocated_memory(),
                    driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                    explicit_tensor_logical_bytes=sum(v.numel()*v.element_size() for v in unique.values()),
                    tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size()) for k,v in tensors.items()}))
        tick=perf_counter()
        gpu=device_cover(inputs,host['total'],observe if isolated else None)
        if isolated:torch.mps.synchronize()
        device_wall=(perf_counter()-tick)*1000
        tick=perf_counter()
        uv=gpu[0].cpu().numpy();offsets=gpu[1].cpu().numpy()
        torch.mps.synchronize()
        d2h=(perf_counter()-tick)*1000
        self.last=dict(cover_uv=uv,cover_offsets=offsets,adapter_ms=adapter,
            hybrid_wall_ms=(perf_counter()-start)*1000,h2d_wall_ms=h2d,d2h_wall_ms=d2h,
            device_wall_ms=device_wall,device_event_ms=None,
            h2d_bytes=host['h2d_bytes'],d2h_bytes=uv.nbytes+offsets.nbytes,
            logical_scratch_bound_bytes=host['logical_scratch_bound_bytes'],bbox_cells=host['total'],
            isolated=isolated,isolated_stages=stages,memory_boundaries=boundaries,
            stage_scope='isolated synchronized walls' if isolated else 'enqueue walls; D2H includes pending device completion; use hybrid/total wall')
        assert uv.dtype==offsets.dtype==np.int64
        return uv,offsets

    def build(self,camera_points,object_id,source_index,component_id,evidence,*,
              extended=True,contexts=None,stored_triangles=None):
        """CPU A structure with one batched cover call; unchanged CPU validator."""
        import numpy as np
        from visual.runtime import extended_delaunay_edges as cpu
        begin=perf_counter()
        xyz,oid,source=map(np.asarray,(camera_points,object_id,source_index))
        g=evidence.frame.geometry;n=len(source)
        if (xyz.shape!=(n,3) or xyz.dtype!=np.float32 or not np.isfinite(xyz).all()
                or not np.array_equal(xyz,g.xyz.reshape(-1,3)[source])):
            raise ValueError('source camera XYZ mismatch')
        audit=cpu.candidate_edges(oid,source,component_id,g.valid_depth.shape,
                                 extended=extended,stored_triangles=stored_triangles)
        for obj in (0,1):
            if not g.observed_mask(obj).ravel()[source[oid==obj]].all():
                raise ValueError('selected source not observed')
        timing=audit['timing_ms'];tick=perf_counter()
        contexts=cpu.prepare_validation(evidence) if contexts is None else contexts
        timing['validation_lookup_ms']=(perf_counter()-tick)*1000
        timing.update(supercover_ms=0.,support_validation_ms=0.,depth_validation_ms=0.)
        w=g.valid_depth.shape[1];uv=np.column_stack((source%w,source//w))
        tick=perf_counter();endpoints=uv[audit['candidates']]
        cover,offsets=self.cover(endpoints)
        timing['supercover_ms']=(perf_counter()-tick)*1000
        accepted=[];checks=[];object_ms={0:0.,1:0.};cover_bytes=bbox_bytes=0
        for i,pair in enumerate(audit['candidates']):
            tick=perf_counter();c=cover[offsets[i]:offsets[i+1]]
            result=cpu.validate_cover(c,contexts[int(oid[pair[0]])],timing)
            object_ms[int(oid[pair[0]])]+=(perf_counter()-tick)*1000
            accepted.append(not result['reasons']);checks.append(result)
            cover_bytes=max(cover_bytes,c.nbytes)
            bbox_bytes=max(bbox_bytes,int(np.prod(uv[pair].max(0)-uv[pair].min(0)+1))*8*8)
        accepted=np.asarray(accepted,bool)
        tick=perf_counter();edge=cpu.directed_edges(audit['candidates'][accepted])
        timing['edge_packaging_ms']=(perf_counter()-tick)*1000
        timing['helper_wall_ms']=(perf_counter()-begin)*1000
        audit.update(accepted=accepted,checks=checks,validation_object_ms=object_ms,
            raster_work=dict(cover_peak_array_bytes=cover_bytes,bbox_peak_scratch_estimate_bytes=bbox_bytes))
        return edge,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-output',type=Path,required=True)
    args=parser.parse_args()
    result=probe()
    args.probe_output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'])
    for row in result['operations']:
        print(row['name'],row['status'],row.get('exception',''))


if __name__=='__main__': main()
