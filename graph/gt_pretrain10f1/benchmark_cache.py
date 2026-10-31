"""Concurrent independent CUDA episode worker: exact output and wall cost."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from graph.plan_a_bc.common import read,write,sha
from graph.plan_a_bc.cache import make_vision
from graph.geometry_reach10d.optimizations import DetectionMemo
from graph.geometry_reach10d.parallel_cache import make_pool,cache_episode
from utils.experiment_logger import ExperimentLogger


def run(root):
    torch.set_num_threads(1)
    dest=root/'concurrent_worker_benchmark';dest.mkdir(exist_ok=False)
    for name in ('config.json','normalizer.json','dataset_audit.json'):shutil.copy2(root/name,dest/name)
    shutil.copytree(root/'inputs',dest/'inputs')
    spec=read(root/'dataset_manifest.json')['episodes'][0]
    raw=root/'raw'/str(spec['episode_id']);receipt=read(raw/'receipt.json')
    entry=dict(episode_id=spec['episode_id'],seed=spec['seed'],split=spec['split'],actions=receipt['actions'],
        path=str((raw/'episode.h5').relative_to(root)),sha256=receipt['hashes']['episode.h5'])
    paths=read(root/'runtime_paths.json');wall=time.perf_counter()
    dino,tracker=make_vision('cuda',paths['dino'],paths['sam'])
    memo=DetectionMemo(dino,'Task10F1_concurrent_benchmark',dict(dino=paths['dino'],sam_sha256=sha(paths['sam'])))
    with ExperimentLogger(team='graph',model='Task10F1_independent_worker_benchmark',seed=42,
            config=dict(parent_run_id=read(root/'run_identity.json')['run_id']),root=root/'experiments',job_type='prepare') as log,make_pool(dest,4) as pool:
        result=cache_episode(dest,root,entry,memo,tracker,log,pool,4)
    checked=[]
    with np.load(root/'cache'/f'ep{spec["episode_id"]}'/'graphs.npz') as a,np.load(dest/'cache'/f'ep{spec["episode_id"]}'/'graphs.npz') as b:
        for key in a.files:
            if key=='generation_seconds':continue
            np.testing.assert_array_equal(a[key],b[key],err_msg=key);checked.append(key)
    write(root/'concurrent_worker_benchmark.json',dict(status='PASS',keys_exact=checked,graphs=entry['actions'],
        cache_wall_seconds=result['wall_seconds'],including_startup_seconds=time.perf_counter()-wall,
        comparison='same first Full-task episode while primary CUDA worker and CPU collectors run; all payload arrays exact',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()),immutable=True)
    print('CONCURRENT BENCHMARK PASS',result['wall_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
