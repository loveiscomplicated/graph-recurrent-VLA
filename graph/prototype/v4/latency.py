"""Idle-host cached and uncached latency, kept separate from rollout reports."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from ..common import seed_all, write_json
from ..data import load_dataset, split_indices
from ..v2.diagnostics import qvel_dataset
from .history_dataset import HistoryDataset
from .evaluate import load, Runtime


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/graph/v4'));p.add_argument('--repeats',type=int,default=100);args=p.parse_args()
    if args.repeats<1:p.error('positive repeat count')
    dataset=Path('outputs/graph/dataset.pt');data,g=load_dataset(dataset);q=qvel_dataset(data);test=split_indices(data,'test');result={}
    with torch.inference_mode():
        for name in ['flat','surface']:
            for mode in ['current','action','state','both']:
                ck,m,norm,sn,builder=load(args.root/'training'/f'{name}_{mode}'/'best.pt',data,dataset)
                seed_all(0,ck['config']['threads']);runtime=Runtime(m,norm,sn,builder);windows=HistoryDataset(data,m.length)
                uncached=[];commit=[];agreement=0.
                # One real contiguous held-out episode, reset on each sweep.
                rows=torch.where(data['episode_id']==data['manifest']['splits']['test'][0])[0]
                rows=rows[torch.argsort(data['timestep'][rows])]
                for repeat in range(args.repeats+10):
                    index=repeat%len(rows)
                    if index==0:runtime.buffer.reset()
                    ids=rows[index:index+1];geom=g.select(ids)
                    raw,h,*_=runtime.predict(geom,q[ids]);start=perf_counter();runtime.buffer.commit(h,data['actions'][ids]);elapsed=(perf_counter()-start)*1000
                    start=perf_counter();expected=windows.forward(m,ids,g,q,norm,sn,builder);uncached_ms=(perf_counter()-start)*1000
                    agreement=max(agreement,float((raw-expected).abs().max()))
                    if repeat>=10:uncached.append(uncached_ms);commit.append(elapsed)
                if agreement>2e-5:raise AssertionError(f'cache/offline differ {name}/{mode}: {agreement}')
                means=np.mean(runtime.timing[10:],axis=0)
                metrics=dict(zip(['feature_construction_ms','encoder_ms','history_processing_ms','action_head_ms'],means.tolist()))
                metrics.update(cache_commit_ms=float(np.mean(commit)),total_ms=float(means.sum()+np.mean(commit)),uncached_total_ms=float(np.mean(uncached)),cache_max_prediction_difference=agreement,repeats=args.repeats,threads=ck['config']['threads'],device='cpu',excludes='live GT extraction and simulator stepping',includes='normalization, construction, current encoder, stacking, head and cache commit')
                result[f'{name}_{mode}']=metrics
    write_json(args.root/'comparison'/'latency.json',result)


if __name__=='__main__':main()
