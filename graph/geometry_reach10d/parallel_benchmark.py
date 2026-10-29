"""New-offset, full-episode equality and throughput gate for CPU overlap."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,append_csv
from graph.plan_a_bc.cache import make_vision
from graph.geometry_reach10d.pilot_cache import cache_root
from graph.geometry_reach10d.parallel_cache import cache_episode,make_pool
from graph.geometry_reach10d.optimizations import DetectionMemo
from graph.geometry_reach10d.profile import Resources


def run(root):
    torch.set_num_threads(1);torch.manual_seed(42)
    selected=[r for r in read(root/'capture_manifest.json')['episodes'] if r['spec']['episode_id'] in [1000001,1000014]]
    entries=[dict(episode_id=r['spec']['episode_id'],seed=r['spec']['seed'],split='pilot',actions=r['actions'],
        path=str(Path(r['raw_path']).relative_to(root)),sha256=r['sha256']) for r in selected]
    folder=root/'pilot_cache/parallel';cache_root(root,folder,entries)
    paths=read(root/'runtime_paths.json');dino,tracker=make_vision('cuda',paths['dino'],paths['sam'])
    detector=DetectionMemo(dino,'parallel_gate',read(root/'optimization_equivalence.json')['model_identity'])
    results=[];start=time.perf_counter()
    with make_pool(root,2) as pool, ExperimentLogger(team='graph',model='Task10D_parallel_cache_gate',seed=42,
            config=dict(comparison_uuid=read(root/'run_identity.json')['run_id'],workers=2),
            root=root/'experiments',job_type='prepare') as log,Resources(root,'parallel_cache_gate'):
        for e in entries:
            results.append(cache_episode(folder,root,e,detector,tracker,log,pool,2))
            detector.cache.clear();torch.cuda.empty_cache()
    wall=time.perf_counter()-start;checks=[]
    for e in entries:
        with np.load(root/'pilot_cache/original/cache'/f'ep{e["episode_id"]}'/'graphs.npz') as a, np.load(folder/'cache'/f'ep{e["episode_id"]}'/'graphs.npz') as b:
            for k in a.files:
                if k!='generation_seconds':np.testing.assert_array_equal(a[k],b[k])
        assert read(root/'pilot_cache/original/cache'/f'ep{e["episode_id"]}'/'gate_trace.json')==read(folder/'cache'/f'ep{e["episode_id"]}'/'gate_trace.json')
        checks.append(dict(episode_id=e['episode_id'],all_arrays_except_timing_exact=True,gate_exact=True))
    baseline=sum(r['wall_seconds'] for r in read(root/'optimization_equivalence.json')['results'] if r['condition']=='optimized')
    speedup=baseline/wall
    write(root/'parallel_cache_gate.json',dict(status='PASS',adopted=speedup>1.05,workers=2,
        serial_optimized_seconds=baseline,parallel_seconds=wall,speedup=speedup,checks=checks,
        source_sha256=sha(Path(__file__).with_name('parallel_cache.py')),results=results,
        note='87 complete new-offset frames; includes process startup/IPC; production serial cache was running concurrently'),immutable=True)
    append_csv(root/'gpu_cpu_benchmark.csv',dict(stage='full_causal_cache',condition='parallel_geometry_2',seconds=wall,frames=87))
    print('PARALLEL GATE PASS',speedup,'adopt',speedup>1.05,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
