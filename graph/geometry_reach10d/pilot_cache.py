"""New-offset causal cache/optimization equivalence with frozen Plan A writer."""
import argparse
import copy
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,digest,append_csv
from graph.plan_a_bc.cache import cache_episode,make_vision,GRAPH_KEYS
from graph.geometry_reach10d.profile import Resources,hardware
from graph.geometry_reach10d.optimizations import DetectionMemo,components


def cache_root(root,folder,entries):
    folder.mkdir(parents=True,exist_ok=True)
    for name in ['inputs','normalizer.json']:
        p=folder/name
        if not p.exists():p.symlink_to(root/name,target_is_directory=name=='inputs')
    write(folder/'config.json',read(root/'config.json'))
    write(folder/'dataset_audit.json',dict(dataset_sha256=digest(entries),split_sha256=digest([e['split'] for e in entries]),episodes=entries))


def main(root):
    torch.set_num_threads(1);torch.manual_seed(42)
    write(root/'hardware.json',hardware())
    receipts=read(root/'capture_manifest.json')['episodes']
    assert len(receipts)==10 and all(r['status']=='ACCEPTED' for r in receipts)
    selected=[r for r in receipts if r['spec']['episode_id'] in (1000001,1000014)]
    entries=[dict(episode_id=r['spec']['episode_id'],seed=r['spec']['seed'],split='pilot',
        actions=r['actions'],path=str(Path(r['raw_path']).relative_to(root)),sha256=r['sha256']) for r in selected]
    paths=read(root/'runtime_paths.json');dino,tracker=make_vision('cuda',paths['dino'],paths['sam'])
    assert not dino.model.training
    assert not tracker.predictor.training
    import graph.plan_a_geometry.core as core
    original=core.components;results=[]
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from graph.plan_a_bc.geometry import CurrentGeometry
    ident=dict(dino_files={p.name:sha(p) for p in Path(paths['dino']).iterdir() if p.is_file()},sam_sha256=sha(paths['sam']))
    for condition in ['original','optimized']:
        folder=root/'pilot_cache'/condition;cache_root(root,folder,entries)
        core.components=components if condition=='optimized' else original
        detector=DetectionMemo(dino,'pilot_new_offsets',ident) if condition=='optimized' else dino
        with ExperimentLogger(team='graph',model='Task10D_cache_equivalence_'+condition,
                config=dict(comparison_uuid=read(root/'run_identity.json')['run_id'],condition=condition),
                root=root/'experiments',job_type='prepare') as log,Resources(root,'pilot_cache_'+condition):
            begin=time.perf_counter()
            for e in entries:
                results.append(dict(condition=condition,**cache_episode(folder,root,e,detector,tracker,log)))
                # Caches are confined to this deterministic pure detector. The
                # causal adapter still resets every episode/camera.
                if isinstance(detector,DetectionMemo):detector.cache.clear()
                torch.cuda.empty_cache()
            append_csv(root/'gpu_cpu_benchmark.csv',dict(stage='full_causal_cache',condition=condition,
                seconds=time.perf_counter()-begin,frames=sum(e['actions'] for e in entries)))
    core.components=original
    checks=[]
    for e in entries:
        with np.load(root/'pilot_cache/original/cache'/f'ep{e["episode_id"]}'/'graphs.npz') as a, np.load(root/'pilot_cache/optimized/cache'/f'ep{e["episode_id"]}'/'graphs.npz') as b:
            for k in a.files:
                if k=='generation_seconds':continue
                assert np.array_equal(a[k],b[k],equal_nan=True),k
            checks.append(dict(episode_id=e['episode_id'],frames=e['actions'],all_cache_arrays_exact=True))
        assert read(root/'pilot_cache/original/cache'/f'ep{e["episode_id"]}'/'gate_trace.json')==read(root/'pilot_cache/optimized/cache'/f'ep{e["episode_id"]}'/'gate_trace.json')
    write(root/'optimization_equivalence.json',dict(status='PASS',checks=checks,
        scope='two complete new Peg-offset episodes; masks, gate, graph, source/F5, robot, actions, timestamps exact',
        optimized_source_sha256=sha(Path(__file__).with_name('optimizations.py')),
        model_identity=ident,precision='FP32; AMP not adopted',results=results,
        memo_hits=sum(r['hit'] for r in detector.trace),memo_calls=len(detector.trace)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
