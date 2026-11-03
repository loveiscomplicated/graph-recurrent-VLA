"""Finite CPU-input -> CPU-final-Graph benchmark. No models/FK/probe in timer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import shutil

FROZEN=Path(os.environ.get('TASK10I_ROOT','/workspace/task10i_resolution_perception_graph_20261103_v1/repaired_graph_v2'))
REFERENCE=Path(os.environ.get('TASK10I_REFERENCE_CODE',Path(__file__).resolve().parents[1]/'resolution_perception_graph_20261103_v1'))
sys.path.insert(0,str(REFERENCE));sys.path.insert(0,str(FROZEN/'code'))
import common as C
import numpy as np
import torch
from graph_adapter import geometry,load_inputs
from graph_worker import masks_for
from fast_graph import FastGraph
from utils.experiment_logger import ExperimentLogger
sys.path.insert(0,str(Path(__file__).resolve().parent))
from candidates import Candidate


def deadline(*_):raise TimeoutError('180s Graph build limit')


def execute(geo,inputs,masks,stage,trace=False):
    ctx=FastGraph('fast_v5_tree') if stage==0 else Candidate(stage)
    with ctx:
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        if Path('/proc/self/clear_refs').exists():Path('/proc/self/clear_refs').write_text('5\n')
        signal.alarm(180)
        try:
            if trace:
                from graph_adapter import build
                result=build(geo,inputs,masks,trace=True)
                return dict(ms=result['graph_ms'],graph=result['graph'],sources=result['sources'],gate=result['gate'],spans=result['spans'])
            start=time.perf_counter()
            graph,records,gate=geo.build(*inputs[:3],masks)
            torch.cuda.synchronize()
            elapsed=(time.perf_counter()-start)*1000
        finally:
            signal.alarm(0);geo.reset()
            # The inherited diagnostic helper restores an instance-bound method.
            # Remove that shadow so the next candidate sees its class patch.
            if trace:geo.system.__dict__.pop('represent',None)
    # Copies, hashes, comparisons, output writes and memory inspection are outside timing.
    arrays={k:getattr(graph,k).numpy().copy() for k in C.FIELDS}
    peak=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
    return dict(ms=elapsed,graph=arrays,sources=[r['source_observation_ids'] for r in records],gate=gate,
        cpu_peak_rss_bytes=peak,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def compare(actual,reference):
    for name in ('node_type','edge_index','edge_type','edge_attr'):
        np.testing.assert_array_equal(actual['graph'][name],reference['graph'][name],err_msg=name)
    a,b=actual['graph']['node_input'],reference['graph']['node_input']
    np.testing.assert_array_equal(a[:,:3],b[:,:3]);np.testing.assert_array_equal(a[:,8:],b[:,8:])
    np.testing.assert_allclose(a[:,3:8],b[:,3:8],atol=2e-5,rtol=0)
    assert actual['sources']==reference['sources'],'selected/merged source IDs changed'
    assert actual['gate']==reference['gate'],'wrist gate decision or diagnostics changed'
    return dict(discrete_graph_exact=True,positions_exact=True,sources_exact=True,gate_exact=True,
        feature_max_abs=float(np.max(np.abs(a[:,3:8]-b[:,3:8]),initial=0)),feature_atol=2e-5)


def main(a):
    C.assert_source();torch.set_num_threads(a.threads)
    resource.setrlimit(resource.RLIMIT_DATA,(80*1024**3,80*1024**3))
    signal.signal(signal.SIGALRM,deadline)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    import native
    native.library()  # Compile before any timed call.
    specs={s['id']:s for s in C.read(FROZEN/'protocol.json')['scenes']}
    cases=(C.read(a.cases)['cases'] if a.cases else [dict(scene=a.scene,size=s,repeat=0) for s in a.sizes])
    identity=dict(host=platform.platform(),torch=torch.__version__,numpy=np.__version__,
        gpu=torch.cuda.get_device_name(0),cpu_model=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        affinity=sorted(os.sched_getaffinity(0)),threads=a.threads,
        environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in (Path(__file__).parent,REFERENCE) for p in folder.iterdir() if p.suffix in ('.py','.cpp')},
        original_source_sha256=C.read(FROZEN/'protocol.json')['source_sha256'])
    import scipy,triton
    identity.update(scipy=scipy.__version__,triton=triton.__version__,python=sys.version,
        compiler=__import__('subprocess').check_output(['c++','--version'],text=True).splitlines()[0],
        data_limit_bytes=80*1024**3,build_deadline_seconds=180)
    identity['cgroup_limits']={p:Path('/sys/fs/cgroup',p).read_text().strip()
        for p in ('cpu.max','memory.max') if Path('/sys/fs/cgroup',p).exists()}
    identity['gc_enabled']=__import__('gc').isenabled()
    repo=Path(os.environ['TASK10I_REPO']).resolve()
    identity['loaded_repo_sources']={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
        for module in list(sys.modules.values()) if (filename:=getattr(module,'__file__',None))
        and (p:=Path(filename)).suffix=='.py' and p.is_relative_to(repo) and p.is_file()}
    snapshot=output/'source';snapshot.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.cpp'):shutil.copy2(p,snapshot/p.name)
    C.write(output/'config.json',dict(args=vars(a),identity=identity,boundary='CPU sensor arrays + masks + robot surfaces -> CPU final Graph + records + gate; internal transfers included; no I/O/FK/probe/SAM/action',scope='Graph-only optimization; original experiment remains stopped',warmups=a.warmups,repeats=a.repeats))
    rows=[]
    with ExperimentLogger(team='integration',model='task10i_graph_latency_v2',job_type='evaluate',seed=42,
                          root=output/'experiments',config=dict(args=vars(a),identity=identity)) as log:
        for case in cases:
            geo=geometry();inputs=load_inputs(case['scene'],case['size'],geo)
            masks=masks_for(specs[case['scene']],case['size'],case.get('repeat',0),'actual')
            input_hashes=dict(cameras={c.camera:{k:C.array_sha(getattr(c,k)) for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','tcp_pose')} for c in inputs[0]},
                masks={k:C.array_sha(v) for k,v in masks.items()},robot={k:C.array_sha(v) for k,v in inputs[1].items() if isinstance(v,np.ndarray)},
                surfaces=[dict(link=s.link,vertices=C.array_sha(s.world_vertices)) for s in inputs[2]])
            C.write(output/f"input_{case['scene']}_{case['size']}_r{case.get('repeat',0)}.json",input_hashes)
            reference=execute(geo,inputs,masks,0)
            # Validate the new baseline against the preserved previous output if available.
            old=REFERENCE.parent/'validation/fast_v5_tree'/case['scene']/str(case['size'])
            if old.exists() and case.get('repeat',0)==0:
                with np.load(old/'graph.npz') as z:
                    for k in C.FIELDS:np.testing.assert_array_equal(reference['graph'][k],z[k])
            for _ in range(a.warmups):
                for stage in a.stages:compare(execute(geo,inputs,masks,stage),reference)
            for repeat in range(a.repeats):
                # Alternate direction to reduce fixed ordering bias. No concurrent builds.
                for stage in (a.stages if repeat%2==0 else list(reversed(a.stages))):
                    result=execute(geo,inputs,masks,stage)
                    comparison=compare(result,reference)
                    if repeat==0:
                        folder=output/'graphs'/f"{case['scene']}_{case['size']}_r{case.get('repeat',0)}"/f's{stage}'
                        folder.mkdir(parents=True)
                        np.savez_compressed(folder/'graph.npz',**result['graph'])
                        C.write(folder/'sources_gate.json',dict(sources=result['sources'],gate=result['gate']))
                    row=dict(case=case,stage=stage,repeat=repeat,**{k:v for k,v in result.items() if k not in ('graph','sources','gate')},comparison=comparison,
                        graph_hashes={k:C.array_sha(v) for k,v in result['graph'].items()})
                    rows.append(row);C.write(output/'results.json',dict(status='PARTIAL',rows=rows))
                    print(json.dumps(dict(scene=case['scene'],size=case['size'],stage=stage,repeat=repeat,ms=row['ms'],error=comparison['feature_max_abs'])),flush=True)
            if a.trace:
                for stage in a.stages:
                    result=execute(geo,inputs,masks,stage,trace=True);compare(result,reference)
                    C.write(output/f"trace_{case['scene']}_{case['size']}_s{stage}.json",{k:v for k,v in result.items() if k not in ('graph','sources')})
            log.log(dict(cases_complete=cases.index(case)+1));
        path=output/'results.json';C.write(path,dict(status='PASS',rows=rows));log.link_output(path);log.link_output(output/'config.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scene',default='10i_s1_0686_t000')
    p.add_argument('--sizes',nargs='+',type=int,default=[128,512,768,1024]);p.add_argument('--stages',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--warmups',type=int,default=2);p.add_argument('--repeats',type=int,default=7);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--cases');p.add_argument('--trace',action='store_true');main(p.parse_args())
