"""Freeze paired Graph outputs; deliberately never open actor ground truth."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import shutil
import subprocess
import sys
import time
import traceback

FROZEN=Path(os.environ['TASK10I_ROOT'])
CODE=Path(os.environ['TASK10I_LATENCY_CODE'])
REFERENCE=Path(os.environ['TASK10I_REFERENCE_CODE'])
sys.path.insert(0,str(REFERENCE));sys.path.insert(0,str(CODE));sys.path.insert(0,str(FROZEN/'code'))
import common as C
import numpy as np
import torch
from graph_adapter import geometry,load_inputs
from graph_worker import masks_for
from candidates import Candidate
from utils.experiment_logger import ExperimentLogger
sys.path.insert(0,str(Path(__file__).parent))
import policies


def deadline(*_):raise TimeoutError('per-build limit')


def execute(geo,inputs,masks,mode,capture=False):
    import graph.plan_a_geometry.core as core
    captured={}
    with Candidate(9) as ctx:
        policies.apply(ctx,mode)
        if capture:
            original=core.PlanA.prepare
            def observed(system,*a,**kw):
                result=original(system,*a,**kw)
                captured.update(clean=result['clean'],effective=result['conditions']['C3'])
                return result
            ctx.patch(core.PlanA,'prepare',observed)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        if Path('/proc/self/clear_refs').exists():Path('/proc/self/clear_refs').write_text('5\n')
        signal.alarm(120)
        try:
            start=time.perf_counter()
            graph,records,gate=geo.build(*inputs[:3],masks)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        finally:
            signal.alarm(0);geo.reset()
    arrays={name:getattr(graph,name).numpy().copy() for name in C.FIELDS}
    rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
    info=dict(graph_ms=elapsed,hashes={k:C.array_sha(v) for k,v in arrays.items()},
        peak_rss_bytes=rss,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    rows=[dict(object_id=r['object_id'],world_xyz=r['world_xyz'],sources=r['source_observation_ids']) for r in records]
    return info,arrays,rows,gate,captured


def worker(args):
    C.assert_source();torch.set_num_threads(4)
    resource.setrlimit(resource.RLIMIT_DATA,(80*1024**3,80*1024**3))
    signal.signal(signal.SIGALRM,deadline)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.cases).read_text());spec=next(s for s in C.read(FROZEN/'protocol.json')['scenes'] if s['id']==args.scene)
    geo=geometry();inputs=load_inputs(args.scene,args.size,geo);masks=masks_for(spec,args.size,0,'actual')
    C.write(output/'inputs.json',dict(scene=args.scene,size=args.size,parent=spec['parent'],cohort=spec['cohort'],
        timestep=spec['timestep'],masks={k:C.array_sha(v) for k,v in masks.items()},
        sensor_files={p.name:C.sha(p) for p in (FROZEN/'sensors'/args.scene/str(args.size)).glob('*.npz')}))
    modes=config['modes'];fail={};samples={m:[] for m in modes}
    for mode in modes:
        try:execute(geo,inputs,masks,mode)
        except Exception as exc:
            fail[mode]=dict(stage='warmup',type=type(exc).__name__,message=str(exc));torch.cuda.empty_cache()
    for r in range(config['timed_repeats_per_mode']):
        order=modes[r%len(modes):]+modes[:r%len(modes)]
        if r%2:order=order[::-1]
        for mode in order:
            if mode in fail:continue
            try:
                info,*_=execute(geo,inputs,masks,mode);samples[mode].append(info)
                print(json.dumps(dict(scene=args.scene,size=args.size,mode=mode,repeat=r,ms=info['graph_ms'])),flush=True)
            except Exception as exc:
                fail[mode]=dict(stage='timed',repeat=r,type=type(exc).__name__,message=str(exc));torch.cuda.empty_cache()
    np.savez_compressed(output/'raw_masks.npz',**masks)
    for mode in modes:
        target=output/mode;target.mkdir()
        if mode not in fail:
            try:
                info,arrays,rows,gate,capture=execute(geo,inputs,masks,mode,capture=True)
                if any(row['hashes']!=info['hashes'] for row in samples[mode]):
                    raise AssertionError('same-mode Graph tensor hashes varied')
                prior=CODE.parent/'final_paired_t4'/'graphs'/f'{args.scene}_{args.size}_r0'/'s9'
                if mode=='full' and prior.exists():
                    with np.load(prior/'graph.npz',allow_pickle=False) as previous:
                        for k in C.FIELDS:np.testing.assert_array_equal(arrays[k],previous[k],err_msg=k)
                    expected=C.read(prior/'sources_gate.json')
                    assert json.loads(json.dumps([r['sources'] for r in rows],default=C.serial))==expected['sources']
                    assert gate==expected['gate']
                np.savez_compressed(target/'graph.npz',**arrays)
                np.savez_compressed(target/'masks.npz',**{f'{stage}__{cam}':v for stage in ('clean','effective') for cam,v in capture[stage].items()})
                C.write(target/'prediction.json',dict(status='PASS',gate=gate,records=rows,hashes=info['hashes'],samples=samples[mode]))
            except Exception as exc:
                fail[mode]=dict(stage='freeze',type=type(exc).__name__,message=str(exc));torch.cuda.empty_cache()
        if mode in fail:C.write(target/'prediction.json',dict(status='FAILED',failure=fail[mode],samples=samples[mode]))
    C.write(output/'receipt.json',dict(status='PASS' if not fail else 'PARTIAL',failures=fail,
        files={str(p.relative_to(output)):C.sha(p) for p in output.rglob('*') if p.is_file()}))


def main(args):
    C.assert_source();config=json.loads(Path(args.cases).read_text())
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.json')}
    candidate_sources={p.name:C.sha(p) for p in CODE.iterdir() if p.suffix in ('.py','.cpp')}
    identity=dict(config=config,sources=sources,candidate_sources=candidate_sources,
        frozen_protocol_sha256=C.sha(FROZEN/'protocol.json'),capture_manifest_sha256=C.sha(FROZEN/'capture_manifest.json'),
        host=platform.platform(),cpu=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=np.__version__,
        boundary='CPU sensor inputs, masks and precomputed robot surfaces -> CPU Graph, selected records and gate; includes transfers; excludes input I/O, FK, SAM2, action and scoring',
        no_model_calls=True,no_training=True,no_rollout=True,no_publishing=True)
    order=sorted(config['scenes'],key=lambda scene:hashlib.sha256(('task10i-filter-run:'+scene).encode()).hexdigest())
    identity['execution_order']=order
    C.write(output/'protocol.json',identity)
    snapshot=output/'source';snapshot.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.json'):shutil.copy2(p,snapshot/p.name)
    total=len(config['scenes'])*len(config['sizes']);finished=0;failures=[]
    with ExperimentLogger(team='integration',model='task10i_robot_filter_ablation',job_type='evaluate',seed=42,
                          root=output/'experiments',config=identity) as logger:
        for i,scene in enumerate(order):
            sizes=config['sizes'] if i%2==0 else list(reversed(config['sizes']))
            for size in sizes:
                folder=output/'cases'/scene/str(size)
                cmd=[sys.executable,__file__,'--worker','--scene',scene,'--size',str(size),'--output',str(folder),'--cases',args.cases]
                try:
                    result=subprocess.run(cmd,timeout=360)
                    if result.returncode:failures.append(dict(scene=scene,size=size,error='worker_exit',code=result.returncode))
                except subprocess.TimeoutExpired:failures.append(dict(scene=scene,size=size,error='worker_timeout'))
                receipt=folder/'receipt.json'
                if not receipt.exists():failures.append(dict(scene=scene,size=size,error='missing_receipt'))
                elif C.read(receipt)['status']!='PASS':failures.append(dict(scene=scene,size=size,error='partial_modes'))
                finished+=1;logger.log(dict(cases_attempted=finished,cases_failed=len(failures)),step=finished)
                C.write(output/'progress.json',dict(attempted=finished,total=total,failures=failures))
        frozen=output/'prediction_freeze.json'
        C.write(frozen,dict(status='COMPLETE' if not failures else 'PARTIAL',attempted=finished,expected=total,failures=failures,
            receipts={str(p.relative_to(output)):C.sha(p) for p in (output/'cases').glob('*/*/receipt.json')},GT_used=False))
        logger.link_output(frozen);logger.link_output(output/'protocol.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cases',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--scene');p.add_argument('--size',type=int)
    a=p.parse_args();worker(a) if a.worker else main(a)
