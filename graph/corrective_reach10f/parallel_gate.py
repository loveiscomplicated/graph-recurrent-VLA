"""Isolated current-host end-to-end serial/overlap gate on one fresh raw branch."""
import argparse,time,os
from pathlib import Path
import numpy as np,torch
from graph.plan_a_bc.common import read,write,sha,digest
from graph.plan_a_bc.cache import make_vision,cache_episode
from graph.geometry_reach10d.parallel_cache import make_pool,cache_episode as parallel_episode
from graph.geometry_reach10d.optimizations import components,DetectionMemo
from utils.experiment_logger import ExperimentLogger

def run(root,base,pilot,dino,sam):
    assert os.getenv('CONTAINER_ID')=='57640710';torch.set_num_threads(1)
    assert read(base/'pilot_completion_gate.json')['status']=='PASS';root.mkdir(parents=True,exist_ok=False)
    raw=pilot/'raw/30004_t25_5_0/P/episode.h5';r=read(raw.with_name('receipt.json'));assert r['approved'] and sha(raw)==r['sha256']
    entry=dict(episode_id=4100010,seed=30004,split='pilot',path=str(raw),actions=r['actions'],sha256=r['sha256'])
    for n in ['inputs','normalizer.json','action_codec.json']:(root/n).symlink_to(pilot/n)
    write(root/'config.json',read(pilot/'config.json'));write(root/'dataset_audit.json',dict(dataset_sha256=digest(entry),split_sha256=digest('pilot'),episodes=[entry]))
    detector,tracker=make_vision('cuda',dino,sam)
    import graph.plan_a_geometry.core as core
    core.components=components;ident=dict(sam=sha(sam),dino=str(dino));results=[]
    with ExperimentLogger(team='graph',model='Task10F_current_host_overlap_gate',config=dict(pilot=read(pilot/'identity.json')['run_id'],fresh_branch=entry),root=root/'experiments',job_type='prepare') as log:
        for workers in [0,1,2]:
            folder=root/f'w{workers}';folder.mkdir()
            for n in ['inputs','normalizer.json','action_codec.json','config.json','dataset_audit.json']:(folder/n).symlink_to(root/n)
            memo=DetectionMemo(detector,'Task10F_overlap_gate',ident);begin=time.perf_counter()
            if workers:
                with make_pool(folder,workers) as pool:receipt=parallel_episode(folder,Path('/'),entry,memo,tracker,log,pool,workers)
            else:receipt=cache_episode(folder,Path('/'),entry,memo,tracker,log)
            elapsed=time.perf_counter()-begin;results.append(dict(workers=workers,seconds=elapsed,graphs=r['actions'],graphs_per_second=r['actions']/elapsed,receipt=receipt))
            print('OVERLAP_GATE',workers,elapsed,flush=True)
        first=root/'w0/cache/ep4100010/graphs.npz'
        for workers in [1,2]:
            with np.load(first) as a,np.load(root/f'w{workers}/cache/ep4100010/graphs.npz') as b:
                for k in a.files:
                    if k!='generation_seconds':np.testing.assert_array_equal(a[k],b[k],err_msg=k)
            assert read(root/'w0/cache/ep4100010/gate_trace.json')==read(root/f'w{workers}/cache/ep4100010/gate_trace.json')
        best=min(results,key=lambda r:r['seconds'])
        write(root/'parallel_gate.json',dict(status='PASS',adopted_workers=best['workers'],all_non_timing_arrays_exact=True,results=results,
            limitations='one fresh episode per mode, fixed order, models loaded once, worker startup/shutdown included; no general speedup guarantee',source_sha256=sha(Path(__file__).resolve().parents[1]/'geometry_reach10d/parallel_cache.py')),immutable=True)
        log.link_output(root/'parallel_gate.json',kind='metrics')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True);a=p.parse_args();run(a.root,a.base,a.pilot,a.dino,a.sam)
