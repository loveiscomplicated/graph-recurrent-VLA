"""Paired full-cache throughput without competing Task10D vision workers."""
import argparse
import os
import signal
from pathlib import Path
import time
import psutil
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha
from graph.plan_a_bc.cache import make_vision,cache_episode as serial
from graph.geometry_reach10d.pilot_cache import cache_root
from graph.geometry_reach10d.parallel_cache import cache_episode as parallel,make_pool
from graph.geometry_reach10d.optimizations import DetectionMemo,components
from graph.geometry_reach10d.profile import Resources


def run(root):
    assert all((root/f'collection_worker{i}_result.json').exists() for i in [0,1])
    pid=read(root/'data_processes.json')['children']['cache_full']
    args=psutil.Process(pid).cmdline()
    assert 'graph.geometry_reach10d.cache' in args and str(root) in args
    started=time.perf_counter();os.kill(pid,signal.SIGSTOP)
    try:
        torch.set_num_threads(1);torch.manual_seed(42)
        paths=read(root/'runtime_paths.json');dino,tracker=make_vision('cuda',paths['dino'],paths['sam'])
        detector=DetectionMemo(dino,'isolated_full_cache_gate',read(root/'optimization_equivalence.json')['model_identity'])
        import graph.plan_a_geometry.core as core
        core.components=components
        selected=[r for r in read(root/'capture_manifest.json')['episodes'] if r['spec']['episode_id'] in [1000001,1000014]]
        entries=[dict(episode_id=r['spec']['episode_id'],seed=r['spec']['seed'],split='pilot',actions=r['actions'],
            path=str(Path(r['raw_path']).relative_to(root)),sha256=r['sha256']) for r in selected]
        results={};folders={}
        with ExperimentLogger(team='graph',model='Task10D_isolated_parallel_cache_gate',seed=42,
                config=dict(comparison_uuid=read(root/'run_identity.json')['run_id'],workers=2,production_worker_paused=pid),
                root=root/'experiments',job_type='prepare') as log:
            for condition in ['serial_optimized','parallel_2']:
                folder=root/'pilot_cache'/('isolated_'+condition);folders[condition]=folder
                cache_root(root,folder,entries);begin=time.perf_counter()
                with Resources(root,'isolated_cache_'+condition):
                    if condition=='parallel_2':
                        with make_pool(root,2) as pool:
                            for e in entries:
                                parallel(folder,root,e,detector,tracker,log,pool,2);detector.cache.clear();torch.cuda.empty_cache()
                    else:
                        for e in entries:
                            serial(folder,root,e,detector,tracker,log);detector.cache.clear();torch.cuda.empty_cache()
                results[condition]=time.perf_counter()-begin
            checks=[]
            for e in entries:
                with np.load(folders['serial_optimized']/'cache'/f'ep{e["episode_id"]}'/'graphs.npz') as a,np.load(folders['parallel_2']/'cache'/f'ep{e["episode_id"]}'/'graphs.npz') as b:
                    for k in a.files:
                        if k!='generation_seconds':np.testing.assert_array_equal(a[k],b[k])
                assert read(folders['serial_optimized']/'cache'/f'ep{e["episode_id"]}'/'gate_trace.json')==read(folders['parallel_2']/'cache'/f'ep{e["episode_id"]}'/'gate_trace.json')
                checks.append(dict(episode_id=e['episode_id'],all_non_timing_arrays_exact=True,gate_exact=True))
            speedup=results['serial_optimized']/results['parallel_2']
            write(root/'isolated_parallel_cache_gate.json',dict(status='PASS',adopted=speedup>1.05,workers=2,
                condition_seconds=results,speedup=speedup,checks=checks,writer_source_sha256=sha(Path(__file__).with_name('parallel_cache.py')),
                note='production Task10D cache process SIGSTOP/SIGCONT only; original Plan A services untouched; models loaded once',
                production_pause_seconds=time.perf_counter()-started),immutable=True)
            print('ISOLATED GATE',results,'speedup',speedup,'adopt',speedup>1.05,flush=True)
    finally:
        os.kill(pid,signal.SIGCONT)
        write(root/'isolated_benchmark_resume.json',dict(cache_pid=pid,resumed=True,pause_seconds=time.perf_counter()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
