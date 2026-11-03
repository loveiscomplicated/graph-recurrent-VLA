"""Finite same-host before/after benchmark of frozen actual inputs. No API calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

BASE = Path('/workspace/task10i_resolution_perception_graph_20261103_v1/repaired_graph_v2')
OUT = BASE.parent/'graph_optimization_v1'
sys.path.insert(0, str(BASE/'code'))
import common as C


def deadline(*_):
    raise TimeoutError('180 second per-build deadline')


def worker(args):
    import numpy as np
    import torch
    from graph_adapter import geometry, load_inputs, build
    from graph_worker import masks_for
    C.assert_source()
    resource.setrlimit(resource.RLIMIT_DATA, (80*1024**3, 80*1024**3))
    signal.signal(signal.SIGALRM, deadline)
    torch.set_num_threads(4)
    geo = geometry()
    spec = next(s for s in C.read(BASE/'protocol.json')['scenes'] if s['id']==args.scene)
    rows = []
    context = None
    if args.backend != 'original':
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        if args.backend.startswith('fast_'):
            from fast_graph import FastGraph
            context = FastGraph(args.backend)
        else:
            from gpu_graph import OptimizedGraph
            context = OptimizedGraph(args.backend)
        context.__enter__()
    try:
        for repeat in range(args.repeats+1):
            meta = dict(scene=args.scene, size=args.size, backend=args.backend,
                        repeat=repeat, warmup=repeat==0)
            try:
                signal.alarm(180)
                numeric_comparison=None
                inputs = load_inputs(args.scene, args.size, geo)
                masks = masks_for(spec, args.size, 0, 'actual')
                if torch.cuda.is_available():
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                Path('/proc/self/clear_refs').write_text('5\n')
                start = time.perf_counter()
                result = build(geo, inputs, masks)
                if torch.cuda.is_available():torch.cuda.synchronize()
                elapsed = (time.perf_counter()-start)*1000
                if args.backend in ('fast_v2_pca','fast_v4_combined','fast_v5_tree'):
                    reference=OUT/'validation/original'/args.scene/str(args.size)
                    with np.load(reference/'graph.npz',allow_pickle=False) as f:
                        old={k:f[k].copy() for k in f.files}
                    for k in ['node_type','edge_index','edge_type','edge_attr']:
                        np.testing.assert_array_equal(result['graph'][k],old[k],err_msg=k)
                    observed=result['graph']['node_input'];expected=old['node_input']
                    np.testing.assert_array_equal(observed[:,:3],expected[:,:3])
                    np.testing.assert_array_equal(observed[:,8:],expected[:,8:])
                    np.testing.assert_allclose(observed[:,3:8],expected[:,3:8],atol=2e-5,rtol=0)
                    assert json.loads(json.dumps(result['sources'],default=C.serial))==C.read(reference/'result.json')['sources'],'source points changed'
                    numeric_comparison=dict(discrete_graph_exact=True,source_identity_exact=True,
                        feature_max_abs=float(np.max(np.abs(observed[:,3:8]-expected[:,3:8]),initial=0)),
                        feature_atol=2e-5,reference='existing PCA5 optimization contract')
                elif args.backend.startswith('fast_'):
                    reference=OUT/f'original_{args.scene}_{args.size}.json'
                    if reference.exists():
                        expected=C.read(reference)['rows'][0]['hashes']
                        assert result['hashes']==expected,('optimized Graph differs from original',args.scene,args.size,result['hashes'],expected)
                peak = int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
                row = dict(meta, status='COMPLETE', build_through_probe_hash_ms=elapsed,
                    graph_ms=result['graph_ms'], fk_ms=result['fk_ms'],
                    file_read_decode_ms=result['file_read_decode_ms'],
                    input_join_ms=result['input_join_ms'], probe_ms=result['probe_ms'],
                    hashes=result['hashes'], host_peak_rss_bytes=peak,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    numeric_comparison=numeric_comparison)
                if repeat == 0:
                    target=OUT/'validation'/args.backend/args.scene/str(args.size)
                    target.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(target/'graph.npz',**result['graph'])
                    C.write(target/'result.json',{k:v for k,v in result.items() if k not in ('graph','stages','records')})
            except Exception as e:
                row = dict(meta,status='OOM' if isinstance(e,MemoryError) else 'TIMEOUT' if isinstance(e,TimeoutError) else 'ERROR',error_type=type(e).__name__,error=str(e))
            finally:
                signal.alarm(0);geo.reset()
            rows.append(row)
            C.write(OUT/f'{args.backend}_{args.scene}_{args.size}.json',dict(rows=rows,status='PARTIAL'))
            print(json.dumps(row),flush=True)
            if row['status']!='COMPLETE':break
    finally:
        if context is not None:context.__exit__(None,None,None)
    C.write(OUT/f'{args.backend}_{args.scene}_{args.size}.json',dict(rows=rows,
        status='COMPLETE' if len(rows)==args.repeats+1 and all(r['status']=='COMPLETE' for r in rows) else 'FAILED'))


def main(args):
    import torch
    from utils.experiment_logger import ExperimentLogger
    OUT.mkdir(exist_ok=True)
    contract=OUT/'scope.json'
    if not contract.exists():
        C.write(contract,dict(objective='Finish original Task10-I; optimize Graph only as prerequisite',
            frozen_root=str(BASE),API_requests_already_used=840,new_API_requests=0,
            shared_source_changes=False,training=False,rollout=False,publishing=False,
            before_after=dict(scene='10i_s1_0686_t000',sizes=[128,512,768,1024],
                repeats=args.repeats,warmups=1,threads=4,concurrency=1,data_limit_gib=80,deadline_seconds=180),
            metrics='native Graph builder; separate input I/O/FK/join/probe; RSS/GPU peak; same input/output boundary',
            completion='verified optimized actual Graph freeze -> fresh original GT controls -> original timing/scoring/verification/report',
            excluded='new API requests, model/prompt/threshold changes, extra diagnostic frameworks, package/driver install'))
    config=C.read(contract)
    identity=dict(host=platform.platform(),cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),None),
        affinity_cpu_count=len(os.sched_getaffinity(0)),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        original_source_sha256=C.read(BASE/'protocol.json')['source_sha256'],
        optimization_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parent.glob('*.py')})
    scene=args.scene or '10i_s1_0686_t000'
    sizes=[args.size] if args.size is not None else [128,512,768,1024]
    C.write(OUT/(args.backend+'_'+scene+'_runtime.json'),identity)
    with ExperimentLogger(team='integration',model='task10i_graph_optimization',job_type='evaluate',seed=42,
            root=OUT/'experiments',config=dict(backend=args.backend,contract=config,identity=identity)) as log:
        for size in sizes:
            output=OUT/f'{args.backend}_{scene}_{size}.json'
            if output.exists() and C.read(output).get('status')=='COMPLETE':
                saved=C.read(output)['rows']
                if any(r['status']!='COMPLETE' for r in saved):raise RuntimeError('preserved benchmark failure; choose a new revision')
                continue
            result=subprocess.run([sys.executable,__file__,'--worker','--backend',args.backend,'--scene',scene,'--size',str(size),'--repeats',str(args.repeats)])
            if result.returncode:raise RuntimeError(f'worker exited {result.returncode}')
            rows=C.read(output)['rows']
            if any(r['status']!='COMPLETE' for r in rows):raise RuntimeError('benchmark failure; stop before next condition')
            log.log(dict(successes=sum(r['status']=='COMPLETE' for r in rows)),step=size)
            log.link_output(output)
    print('BENCHMARK_COMPLETE',args.backend,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',default='original',choices=['original','cpu_cells','cuda','cuda_fused','cuda_fused_v2','cuda_fused_v3','cuda_fused_v4','cuda_fused_v5','fast_v1','fast_v2_pca','fast_v3_dedup','fast_v4_combined','fast_v5_tree','fast_v6_tree_exact'])
    p.add_argument('--worker',action='store_true');p.add_argument('--scene');p.add_argument('--size',type=int);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();worker(a) if a.worker else main(a)
