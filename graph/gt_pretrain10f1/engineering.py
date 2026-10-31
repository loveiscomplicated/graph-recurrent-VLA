"""Real cached-frame GPU packing/gradient/resume checks; smoke weights discarded."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import torch
from graph.plan_a_bc.common import read,write,sha
from graph.plan_a_bc.cache import graph_at
from graph.integration_baseline.graph import pack_graphs
from .data import GraphPool
from .model import StateEncoder
from .train import resume_check


def run(root):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    dest=root/'engineering_data';dest.mkdir(exist_ok=False)
    for name in ('normalizer.json','dataset_manifest.json'):shutil.copy2(root/name,dest/name)
    (dest/'raw').symlink_to(root/'raw',target_is_directory=True)
    entries=[]
    for p in sorted((root/'cache').glob('*/receipt.json'))[:2]:
        e=read(p);e['path']=str(p.parent/'graphs.npz')
        e['gt_sidecar_sha256']=sha(root/'raw'/str(e['episode_id'])/'gt_supervision.npz');entries.append(e)
    assert len(entries)==2
    write(dest/'graph_cache_manifest.json',dict(episodes=entries,paired_episodes=[]),immutable=True)
    pool=GraphPool(dest,'cuda');ids=[0,1,entries[0]['graphs'],0];actual,_,_=pool.select(ids)
    graphs=[]
    for idx in ids:
        meta=pool.meta[idx];path=root/'cache'/f'ep{meta["episode_id"]}'/'graphs.npz'
        with np.load(path) as z:graphs.append(graph_at(z,meta['t']))
    expected=pack_graphs(graphs,[(i,0) for i in range(4)]).to('cuda')
    for key,value in vars(actual).items():torch.testing.assert_close(value,getattr(expected,key),atol=0,rtol=0)
    stats=pool.normalization;torch.manual_seed(42)
    model=StateEncoder(stats['robot_mean'],stats['robot_std'],stats['target_mean']).to('cuda')
    result=resume_check(model,pool,dest,stats)
    result.update(GPU_packing_matches_original=True,duplicate_frame_sampling=True,cross_episode_packing=True,
        fixture_source='first two completed Full-task Train episodes; not primary fitting',params=sum(p.numel() for p in model.parameters()))
    write(root/'engineering_verification.json',result,immutable=True)
    print('ENGINEERING PASS',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
