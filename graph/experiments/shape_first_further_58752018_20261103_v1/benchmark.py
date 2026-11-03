"""Paired GPU-output benchmark with direct canonical-walk kernel timing."""
import argparse,hashlib,json,os,subprocess,importlib.util,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from visual.runtime.graph_perception import process_cuda
from graph.experiments.shape_first_cuda_58752018_20261103_v1.benchmark import load_frame
from graph.experiments.shape_first_device_output_58752018_20261103_v1.benchmark import flatten
try:
    from graph.experiments.shape_first_refinement_58752018_20261103_v1.benchmark import load_reference,samples,optimized_stages
except ModuleNotFoundError as exc:
    if not exc.name.startswith('graph.experiments.shape_first_refinement_58752018_20261103_v1'):raise
    # Isolated remote snapshots can supply the unchanged measurement helpers
    # beside this runner instead of altering the preserved runtime packages.
    spec=importlib.util.spec_from_file_location('further_measurement_helpers',Path(__file__).with_name('refinement_helpers.py'))
    helpers=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helpers;spec.loader.exec_module(helpers)
    load_reference,samples,optimized_stages=helpers.load_reference,helpers.samples,helpers.optimized_stages


def canonical_profile(module,masks,depth,repeats):
    original=module.kernels.canonical;captured=[]
    class Capture:
        def __getitem__(self,grid):
            def invoke(*args,**kwargs):
                captured.append((grid,args,kwargs))
                return original[grid](*args,**kwargs)
            return invoke
    module.kernels.canonical=Capture()
    try:
        for oid in (0,1):
            if not bool(masks[oid].any()):continue
            support=masks[oid]&torch.isfinite(depth)&(depth>0)&~(masks[0]&masks[1])
            module.external_contour_sources(masks[oid],support)
    finally:module.kernels.canonical=original
    result={}
    for index,(grid,args,kwargs) in enumerate(captured):
        result[f'canonical_{index}']=samples(lambda:original[grid](*args,**kwargs),repeats)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--paired',action='store_true')
    parser.add_argument('--repeats',type=int,default=100);parser.add_argument('--stage-repeats',type=int,default=30)
    args=parser.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise RuntimeError('another CUDA process is active; do not stop it')
    args.output.mkdir(parents=True,exist_ok=False)
    reference=load_reference(args.reference)
    frame=load_frame(args.inputs);d=torch.as_tensor(frame.depth,device='cuda')
    m=torch.as_tensor(np.stack(list(frame.masks.values())),device='cuda');k=torch.as_tensor(frame.intrinsic,device='cuda')
    p,f=process_cuda(d,m,k);functions={'baseline':lambda:reference.sample_cuda(p,f,d,m,k)}
    if args.paired:
        from visual.runtime import cuda_sampling as current
        functions['optimized']=lambda:current.sample_cuda(p,f,d,m,k)
    config=dict(task='10i_shape_first_further_58752018_20261103_v1',repeats=args.repeats,stage_repeats=args.stage_repeats,
        paired=args.paired,warmups=20,output_device='cuda',scope='ready GPU geometry/features -> completed GPU sampling output',
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in args.inputs.glob('*.npz')})
    logger=ExperimentLogger(team='graph',job_type='benchmark',config=config,seed=42,root=args.output/'experiments')
    try:
        timing={name:[] for name in functions};first={}
        for name,fn in functions.items():
            torch.cuda.synchronize();start=perf_counter();result=fn();torch.cuda.synchronize();first[name]=(perf_counter()-start)*1000
        for _ in range(20):
            for fn in functions.values():result=fn()
        names=list(functions)
        for i in range(args.repeats):
            for name in (names if i%2==0 else names[::-1]):
                torch.cuda.synchronize();start=perf_counter();result=functions[name]();torch.cuda.synchronize()
                timing[name].append((perf_counter()-start)*1000)
        exact=False
        if args.paired:
            a=flatten(reference.sample_cuda(p,f,d,m,k,diagnostic=True));b=flatten(current.sample_cuda(p,f,d,m,k,diagnostic=True))
            assert a.keys()==b.keys()
            for name,value in a.items():
                assert b[name].device.type=='cuda'
                torch.testing.assert_close(value,b[name],rtol=0,atol=0,equal_nan=True)
            exact=True
        profile={}
        for name,module in [('baseline',reference)]+([('optimized',current)] if args.paired else []):
            profile.update({name+':'+key:value for key,value in optimized_stages(module,p,f,d,m,k,args.stage_repeats).items()})
            profile.update({name+':'+key:value for key,value in canonical_profile(module,m,d,args.stage_repeats).items()})
        stats=lambda v:dict(median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)))
        summary=dict(run_id=logger.run_id,resolution=list(frame.depth.shape),first_call_ms=first,
            timings={name:stats(v) for name,v in timing.items()},isolated_stages={name:stats(v) for name,v in profile.items()},all_diagnostic_fields_cuda_and_exact=exact)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(args.output/'timings.npz',**{n:np.array(v) for n,v in timing.items()})
        np.savez(args.output/'stage_timings.npz',**{n:np.array(v) for n,v in profile.items()})
        logger.record(summary=summary)
        for path in args.output.glob('*.*'):logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:logger.finish(failed=True);raise


if __name__=='__main__':main()
