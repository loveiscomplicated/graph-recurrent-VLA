"""Paired World wrist_gate_tensor before/after; Graph preparation excluded."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from types import ModuleType

import numpy as np
import torch
import triton

from graph.experiments.wrist_gate_cuda_10i_20261104_v1.benchmark import load_case, sha, array_sha, write
from graph.persistent_geometry import tensor_gate, tensor_fusion
from graph.persistent_geometry.tensor_camera import tensor_camera
from utils.experiment_logger import ExperimentLogger


def load_before(folder):
    # Isolate the frozen relative kernel import as well as the Python gate.
    # Fusion is unchanged and intentionally shared with the same World views.
    if sha(folder/'tensor_fusion.py') != sha(tensor_fusion.__file__):
        raise AssertionError('baseline fusion is not identical')
    name = 'graph.persistent_geometry._compact_fused_before'
    package = ModuleType(name); package.__path__ = [str(folder)]
    sys.modules[name] = package
    sys.modules[name+'.tensor_fusion'] = tensor_fusion
    spec = importlib.util.spec_from_file_location(name+'.tensor_gate',folder/'tensor_gate.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def profile_call(call, path):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        value = call(); torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    raw = json.loads(path.read_text())['traceEvents']
    events = prof.key_averages(group_by_input_shape=True)
    return value, dict(kernel_events=sum(e.get('cat')=='kernel' for e in raw),
        operations=[dict(name=e.key,count=e.count,shapes=e.input_shapes) for e in events
            if e.key in ('aten::nonzero','aten::_unique2','aten::unique_consecutive','aten::argsort',
                         'cudaLaunchKernel','cudaStreamSynchronize')])


@torch.inference_mode()
def run(args):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    if min(args.warmups,args.repeats)<1: raise ValueError('positive warmups/repeats required')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    baseline=Path(args.baseline); before=load_before(baseline)
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    cameras,raw,clean,cfg,prior,input_path=load_case(Path(args.input))
    cameras=tuple(tensor_camera(c) for c in cameras)
    raw={k:torch.as_tensor(v,device='cuda') for k,v in raw.items()}
    clean={k:torch.as_tensor(v,device='cuda') for k,v in clean.items()}
    views=tensor_fusion.prepare_tensor_views(cameras,clean,coordinate_frame='world')
    def call(module):return module.wrist_gate_tensor(cameras,raw,clean,cfg,views=views)
    def check(value):
        mask,trace=value;host=mask.cpu().numpy()
        if array_sha(host)!=prior['accepted_mask_sha256'] or trace!=prior['trace']:
            raise AssertionError('mask/full trace differs from frozen CPU reference')
        return host
    config=dict(seed=42,warmups=args.warmups,repeats=args.repeats,
        boundary='wrist_gate_tensor only; same ready CUDA masks/depth/intrinsic/World calibration -> accepted CUDA mask and full Python trace; synchronized wall time',
        excluded='input upload, World view preparation, prepare_tensor_gate, EE/Graph preparation, C3 updates, SAM2, robot filter, Graph, simulator, verification readback, file I/O and profiler',
        order='adjacent before/after GPU pair; alternate which runs first',
        input_sha256=sha(input_path),clean_points=29775,gate_config=cfg,
        gpu=torch.cuda.get_device_name(),python=platform.python_version(),torch=torch.__version__,
        numpy=np.__version__,triton=triton.__version__,
        cpu=next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),
        local_before_identity=json.loads((baseline.parent/'before_identity.json').read_text()),
        source_sha256={str(p):sha(p) for p in (Path(__file__),Path(tensor_gate.__file__),
            Path(tensor_fusion.__file__),Path('graph/persistent_geometry/tensor_gate_kernels.py'))},
        baseline_sha256={p.name:sha(p) for p in baseline.glob('*.py')})
    write(output/'config.json',config)
    with ExperimentLogger(team='integration',model='task10i_wrist_gate_compact_fused',seed=42,
            job_type='evaluate',root=output/'experiments',config=config) as logger:
        logger.record(input_sha256=config['input_sha256'],source_run_id='bedd7daa578342678fcfc7154b17bcba')
        for _ in range(args.warmups):
            for module in (before,tensor_gate):check(call(module));torch.cuda.synchronize()
        rows=[];masks={}
        for repeat in range(args.repeats):
            order=('before','after') if repeat%2==0 else ('after','before')
            row=dict(repeat=repeat,order=order,wall_ms={})
            for name in order:
                torch.cuda.synchronize();start=time.perf_counter_ns()
                value=call(before if name=='before' else tensor_gate)
                torch.cuda.synchronize();row['wall_ms'][name]=(time.perf_counter_ns()-start)/1e6
                masks[name]=check(value)
            rows.append(row);print('MEASURE',json.dumps(row),flush=True)
        stats={}
        for name in ('before','after'):
            values=[r['wall_ms'][name] for r in rows]
            stats[name]=dict(median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values))
        audits={}
        for name,module in (('before',before),('after',tensor_gate)):
            value,audits[name]=profile_call(lambda:call(module),output/(name+'_profile.json'))
            check(value)
        result=dict(status='PASS',run_id=logger.run_id,statistics=stats,rows=rows,audits=audits,
            mask_exact=True,full_trace_exact=True,accepted_points=prior['trace']['accepted_points'],
            accepted_mask_sha256=prior['accepted_mask_sha256'],trace=prior['trace'],
            speedup=stats['before']['median_ms']/stats['after']['median_ms'],
            reduction_percent=100*(1-stats['after']['median_ms']/stats['before']['median_ms']))
        write(output/'results.json',result);np.savez_compressed(output/'accepted_masks.npz',**masks)
        logger.log(dict(latency_ms={k:v['median_ms'] for k,v in stats.items()},speedup=result['speedup']))
        for p in ('config.json','results.json','accepted_masks.npz','before_profile.json','after_profile.json'):
            logger.link_output(output/p)
        print('COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ('rows','trace')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    p.add_argument('--warmups',type=int,default=5);p.add_argument('--repeats',type=int,default=30)
    run(p.parse_args())
