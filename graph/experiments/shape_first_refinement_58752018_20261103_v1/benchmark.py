"""GPU-output baseline/paired sampling and isolated pre-change stage costs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import triton

from utils.experiment_logger import ExperimentLogger
from visual.runtime.graph_perception import process_cuda
from graph.experiments.shape_first_cuda_58752018_20261103_v1.benchmark import load_frame
from graph.experiments.shape_first_device_output_58752018_20261103_v1.benchmark import flatten


def load_reference(path):
    def load(name,p):
        spec=importlib.util.spec_from_file_location(name,p)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module);return module
    module=load('refinement_reference',path)
    module.kernels=load('refinement_reference_kernels',path.with_name('cuda_sampling_kernels.py'))
    return module


def samples(fn,repeats,warmups=5):
    for _ in range(warmups):result=fn()
    times=[]
    for _ in range(repeats):
        torch.cuda.synchronize();start=perf_counter();result=fn()
        torch.cuda.synchronize();times.append((perf_counter()-start)*1000)
    return times


def stages(module,projected,features,depth,masks,k,repeats):
    """Standalone operations, not an additive instrumented whole-frame budget."""
    result=module.select_cuda(projected,features,depth,masks,k)
    raw={};details={}
    for oid in (0,1):
        rows=(projected['object_id']==oid).nonzero().flatten()
        if not len(rows):continue
        x=projected['pixels'][rows];xyz=projected['points'][rows]
        support=masks[oid]&torch.isfinite(depth)&(depth>0)&~(masks[0]&masks[1])
        raw[f'{oid}:contour']=samples(lambda:module.external_contour_sources(masks[oid],support),repeats)
        def scale():
            z=torch.sort(xyz[:,2].to(torch.float64)).values
            return (z[(len(z)-1)//2]+z[len(z)//2])*.5*torch.hypot(2/k[0,0].double(),2/k[1,1].double())/(3**.5)
        raw[f'{oid}:median_scale']=samples(scale,repeats);a=scale()
        for name in ('pca','sil'):
            c=result['prepared'][oid]['C'][name];points=xyz[torch.searchsorted(x,c)]
            raw[f'{oid}:voxel_{name}']=samples(lambda:module.voxel_representatives(c,points,a),repeats)
        initial=torch.searchsorted(x,result['selected'][oid]['P'])
        n=len(x);groups=triton.cdiv(n,128);extra=len(result['selected'][oid]['S'])
        def seed():
            distance=torch.empty(n,dtype=torch.float64,device=depth.device)
            used=torch.zeros(n,dtype=torch.bool,device=depth.device);used[initial]=True
            bd=torch.empty(groups,dtype=torch.float64,device=depth.device)
            bi=torch.empty(groups,dtype=torch.int32,device=depth.device)
            out=torch.empty(extra,dtype=torch.int64,device=depth.device)
            module.kernels.fps_seed[(groups,)](xyz,initial,used,distance,bd,bi,n,len(initial),128,32,
                num_warps=8,enable_fp_fusion=False)
            return distance,bd,bi,out
        raw[f'{oid}:fps_seed']=samples(seed,repeats)
        def loop(state):
            distance,bd,bi,out=state
            for step in range(extra):
                module.kernels.fps_choose[(1,)](bd,bi,out,step,groups,triton.next_power_of_2(groups),num_warps=4)
                if step+1<extra:module.kernels.fps_update[(groups,)](xyz,distance,out,bd,bi,step,n,128,num_warps=4,enable_fp_fusion=False)
            return out
        loop_times=[]
        for _ in range(repeats+5):
            state=seed();torch.cuda.synchronize();start=perf_counter();loop(state)
            torch.cuda.synchronize();loop_times.append((perf_counter()-start)*1000)
        raw[f'{oid}:fps_iterations']=loop_times[5:]
        raw[f'{oid}:fps_total']=samples(lambda:module.surface_fps(xyz,initial,extra),repeats)
        details[str(oid)]=dict(points=n,structural=len(initial),extra=extra,
            seed_distance_pairs=n*len(initial),iteration_launches=2*extra-1)
    raw['output_provenance']=samples(lambda:module.selection_output(result),repeats)
    return raw,details


def optimized_stages(module,p,f,d,m,k,repeats):
    result=module.select_cuda(p,f,d,m,k);raw={}
    for oid in (0,1):
        rows=(p['object_id']==oid).nonzero().flatten()
        if not len(rows):continue
        x=p['pixels'][rows];xyz=p['points'][rows]
        support=m[oid]&torch.isfinite(d)&(d>0)&~(m[0]&m[1])
        raw[f'{oid}:contour']=samples(lambda:module.external_contour_sources(m[oid],support),repeats)
        raw[f'{oid}:median_scale']=samples(lambda:module.nominal_scale_cuda(xyz,k),repeats)
        a=module.nominal_scale_cuda(xyz,k)
        for name in ('pca','sil'):
            source=result['prepared'][oid]['C'][name];points=xyz[torch.searchsorted(x,source)]
            raw[f'{oid}:voxel_{name}']=samples(lambda:module.voxel_representatives(source,points,a,source_sorted=name=='pca'),repeats)
        initial=torch.searchsorted(x,result['selected'][oid]['P']);extra=len(result['selected'][oid]['S'])
        raw[f'{oid}:fps_total']=samples(lambda:module.surface_fps(xyz,initial,extra),repeats)
    raw['output_provenance']=samples(lambda:module.selection_output(result),repeats)
    return raw


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--paired',action='store_true')
    p.add_argument('--repeats',type=int,default=100);p.add_argument('--warmups',type=int,default=20)
    p.add_argument('--stage-repeats',type=int,default=30)
    args=p.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('another CUDA process is active; do not stop it')
    args.output.mkdir(parents=True,exist_ok=False)
    reference=load_reference(args.reference)
    frame=load_frame(args.inputs);depth=torch.as_tensor(frame.depth,device='cuda')
    masks=torch.as_tensor(np.stack(list(frame.masks.values())),device='cuda');k=torch.as_tensor(frame.intrinsic,device='cuda')
    projected,features=process_cuda(depth,masks,k)
    functions={'baseline':lambda:reference.sample_cuda(projected,features,depth,masks,k)}
    if args.paired:
        from visual.runtime.cuda_sampling import sample_cuda
        functions['optimized']=lambda:sample_cuda(projected,features,depth,masks,k)
    config=dict(task='10i_shape_first_refinement_58752018_20261103_v1',paired=args.paired,
        repeats=args.repeats,warmups=args.warmups,stage_repeats=args.stage_repeats,
        input_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(args.inputs.glob('*.npz'))},
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    logger=ExperimentLogger(team='graph',job_type='benchmark',config=config,seed=42,root=args.output/'experiments')
    try:
        names=list(functions);timings={name:[] for name in names};first_call={}
        for name,fn in functions.items():
            torch.cuda.synchronize();start=perf_counter();result=fn();torch.cuda.synchronize()
            first_call[name]=(perf_counter()-start)*1000
        for _ in range(args.warmups):
            for fn in functions.values():result=fn()
        for step in range(args.repeats):
            for name in (names if step%2==0 else names[::-1]):
                torch.cuda.synchronize();start=perf_counter();result=functions[name]()
                torch.cuda.synchronize();timings[name].append((perf_counter()-start)*1000)
        expected=flatten(functions['baseline']())
        for name,fn in functions.items():
            actual=flatten(fn());assert actual.keys()==expected.keys()
            for key,t in actual.items():
                assert t.device.type=='cuda'
                torch.testing.assert_close(t,expected[key],rtol=0,atol=0,equal_nan=True)
        if args.paired:
            old_diagnostic=flatten(reference.sample_cuda(projected,features,depth,masks,k,diagnostic=True))
            from visual.runtime.cuda_sampling import sample_cuda
            new_diagnostic=flatten(sample_cuda(projected,features,depth,masks,k,diagnostic=True))
            assert old_diagnostic.keys()==new_diagnostic.keys()
            for key,value in old_diagnostic.items():
                torch.testing.assert_close(value,new_diagnostic[key],rtol=0,atol=0,equal_nan=True)
        profile={};details={};post_profile={}
        if args.stage_repeats:profile,details=stages(reference,projected,features,depth,masks,k,args.stage_repeats)
        if args.paired and args.stage_repeats:
            from visual.runtime import cuda_sampling
            post_profile=optimized_stages(cuda_sampling,projected,features,depth,masks,k,args.stage_repeats)
        bbox=[]
        for mask in frame.masks.values():
            v,u=np.where(mask)
            bbox.append(None if not len(v) else dict(v0=int(v.min()),v1=int(v.max()+1),u0=int(u.min()),u1=int(u.max()+1),area=int((v.max()-v.min()+1)*(u.max()-u.min()+1))))
        stats=lambda t:dict(median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)))
        summary=dict(run_id=logger.run_id,resolution=list(frame.depth.shape),bbox=bbox,details=details,
            timings={n:stats(t) for n,t in timings.items()},first_call_ms=first_call,
            isolated_baseline_stages={n:stats(t) for n,t in profile.items()},
            isolated_optimized_stages={n:stats(t) for n,t in post_profile.items()},
            all_outputs_cuda_and_exact=True,candidate_representative_lists_exact=bool(args.paired))
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(args.output/'timings.npz',**{n:np.array(t) for n,t in timings.items()})
        np.savez(args.output/'stage_timings.npz',**{n:np.array(t) for n,t in profile.items()})
        np.savez(args.output/'optimized_stage_timings.npz',**{n:np.array(t) for n,t in post_profile.items()})
        logger.record(summary=summary)
        for f in args.output.glob('*.*'):logger.link_output(f)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:logger.finish(failed=True);raise


if __name__=='__main__':main()
