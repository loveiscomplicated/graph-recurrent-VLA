"""FP64 seed tiling sweep; compare every distance to the original kernel."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
import triton
from utils.experiment_logger import ExperimentLogger
from visual.runtime import cuda_sampling as module
from visual.runtime.graph_perception import process_cuda
from graph.experiments.shape_first_cuda_58752018_20261103_v1.benchmark import load_frame


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pruned',action='store_true')
    args=parser.parse_args();assert os.environ.get('CONTAINER_ID')=='58752018'
    args.output.mkdir(parents=True,exist_ok=False)
    frame=load_frame(args.inputs);d=torch.as_tensor(frame.depth,device='cuda');m=torch.as_tensor(np.stack(list(frame.masks.values())),device='cuda');k=torch.as_tensor(frame.intrinsic,device='cuda')
    p,f=process_cuda(d,m,k);result=module.select_cuda(p,f,d,m,k)
    rows=(p['object_id']==0).nonzero().flatten();x=p['pixels'][rows];xyz=p['points'][rows]
    initial=torch.searchsorted(x,result['selected'][0]['P']);n=len(x)
    used=torch.zeros(n,dtype=torch.bool,device='cuda');used[initial]=True
    configs=[(128,32,8),(64,16,4),(64,32,4),(128,16,4),(128,32,4),(128,64,8),(256,16,8),(32,32,4)]
    if args.pruned:configs=[(128,32,8),(128,16,4),(64,16,4),(128,32,4),(256,16,8)]
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,config=dict(task='10i_refinement_seed_tiling',configs=configs,pruned=args.pruned),root=args.output/'experiments')
    try:
        reference=None;results=[]
        for trial,(b,t,w) in enumerate(configs):
            g=triton.cdiv(n,b);distance=torch.empty(n,dtype=torch.float64,device='cuda');bd=torch.empty(g,dtype=torch.float64,device='cuda');bi=torch.empty(g,dtype=torch.int32,device='cuda')
            counter=torch.zeros((),dtype=torch.int64,device='cuda')
            def run():
                if args.pruned and trial:
                    module.kernels.fps_seed_pruned[(g,)](xyz,initial,used,distance,bd,bi,counter,n,len(initial),b,t,num_warps=w,enable_fp_fusion=False)
                else:module.kernels.fps_seed[(g,)](xyz,initial,used,distance,bd,bi,n,len(initial),b,t,num_warps=w,enable_fp_fusion=False)
            for _ in range(5):run()
            torch.cuda.synchronize()
            if reference is None:reference=distance.clone()
            else:torch.testing.assert_close(distance,reference,rtol=0,atol=0)
            times=[]
            for _ in range(30):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();run();end.record();end.synchronize();times.append(start.elapsed_time(end))
            pairs=n*len(initial)
            if args.pruned and trial:
                module.kernels.fps_seed_pruned[(g,)](xyz,initial,used,distance,bd,bi,counter,n,len(initial),b,t,True,num_warps=w,enable_fp_fusion=False)
                pairs=int(counter)
            results.append(dict(block=b,tile=t,warps=w,pruned=bool(args.pruned and trial),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),distance_pairs=pairs,all_distances_exact=True))
        output=dict(run_id=logger.run_id,points=n,seeds=len(initial),configs=results)
        path=args.output/'tuning.json';path.write_text(json.dumps(output,indent=2)+'\n');logger.link_output(path);logger.record(results=output);logger.finish();print(json.dumps(output,indent=2))
    except BaseException:logger.finish(failed=True);raise


if __name__=='__main__':main()
