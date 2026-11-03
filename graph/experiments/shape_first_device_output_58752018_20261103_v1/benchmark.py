"""Same GPU sampler, three terminal interfaces; synchronized wall time."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from visual.runtime.cuda_sampling import sample_cuda
from visual.runtime.graph_perception import process_cuda
from graph.experiments.shape_first_cuda_58752018_20261103_v1.benchmark import load_frame


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(tree,prefix=''):
    if isinstance(tree,dict):
        return {name:t for key,value in tree.items() for name,t in flatten(value,prefix+'/'+str(key)).items()}
    return {prefix:tree}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmups',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=100)
    args=parser.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if processes:raise RuntimeError('another CUDA process is active; do not stop it')
    assert args.repeats>0 and args.warmups>=0
    args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('preserved_cpu_return_sampler',args.reference)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    frame=load_frame(args.inputs)
    depth=torch.as_tensor(frame.depth,device='cuda')
    masks=torch.as_tensor(np.stack(list(frame.masks.values())),device='cuda')
    k=torch.as_tensor(frame.intrinsic,device='cuda')
    projected,features=process_cuda(depth,masks,k);torch.cuda.synchronize()
    functions=dict(
        legacy_cpu_numpy=lambda:legacy.to_cpu_selection(legacy.select_cuda(projected,features,depth,masks,k)),
        cuda_tensor=lambda:sample_cuda(projected,features,depth,masks,k),
        explicit_cpu_tensor=lambda:sample_cuda(projected,features,depth,masks,k,device='cpu'))
    config=dict(task='10i_shape_first_device_output_58752018_20261103_v1',
        warmups=args.warmups,repeats=args.repeats,
        scope='GPU feature-ready -> complete sampling result on chosen device; projection/PCA/Graph excluded',
        reference_sha256=sha(args.reference),runner_sha256=sha(__file__),
        inputs={p.name:sha(p) for p in sorted(args.inputs.glob('*.npz'))})
    logger=ExperimentLogger(team='graph',job_type='benchmark',seed=42,config=config,root=args.output/'experiments')
    try:
        names=list(functions);timings={name:[] for name in names}
        for _ in range(args.warmups):
            for function in functions.values():result=function()
        for repeat in range(args.repeats):
            order=names[repeat%3:]+names[:repeat%3]
            if repeat%2:order=order[::-1]
            for name in order:
                torch.cuda.synchronize();start=perf_counter()
                result=functions[name]()
                torch.cuda.synchronize();timings[name].append((perf_counter()-start)*1000)
        # Downloads below are verification only, outside measured intervals.
        expected=flatten(functions['legacy_cpu_numpy']())
        gpu=functions['cuda_tensor']();cpu=functions['explicit_cpu_tensor']()
        for name,output,device in [('cuda_tensor',gpu,'cuda'),('explicit_cpu_tensor',cpu,'cpu')]:
            flat=flatten(output);assert flat.keys()==expected.keys()
            for key,t in flat.items():
                assert isinstance(t,torch.Tensor) and t.device.type==device,(name,key)
                np.testing.assert_array_equal(t.cpu().numpy(),expected[key],err_msg=name+':'+key)
        summary=dict(run_id=logger.run_id,resolution=list(frame.depth.shape),
            mask_pixels=[int(m.sum()) for m in frame.masks.values()],
            selected_points=sum(len(obj['F']) for obj in gpu['selected'].values()),
            default_output_device=str(gpu['selected'][0]['xyz'].device),
            output_tensor_fields=len(flatten(gpu)),all_fields_exact=True,
            timings={name:dict(median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),
                min_ms=min(t),max_ms=max(t)) for name,t in timings.items()})
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez(args.output/'timings.npz',**{name:np.array(t) for name,t in timings.items()})
        np.savez_compressed(args.output/'verified_output.npz',**expected)
        logger.record(summary=summary)
        for path in args.output.glob('*.*'):logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:
        logger.finish(failed=True);raise


if __name__=='__main__':main()
