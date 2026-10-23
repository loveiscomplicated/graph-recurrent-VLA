"""Actual TRAIN demonstration small-batch optimization/checkpoint diagnostic."""
import argparse
from pathlib import Path
from time import perf_counter
import tempfile
import torch
from ..common import config,seed_all,write_json
from ..data import load_dataset,split_indices
from ..graph_builder import GraphBuilder
from ..normalization import FeatureNormalizer
from ..metrics import action_metrics
from .diagnostics import qvel_dataset
from .state import StateNormalizer
from .models import CommonPolicy,bc_loss


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=200);args=p.parse_args()
    cfg=config(Path(__file__).with_name('config.toml'));data,g=load_dataset('outputs/graph/dataset.pt');q=qvel_dataset(data);train=split_indices(data,'train')
    episodes=data['manifest']['splits']['train'][:2];ids=[]
    for eid in episodes:
        rows=torch.where(data['episode_id']==eid)[0];ids.append(rows[torch.linspace(0,len(rows)-1,8).long()])
    ids=torch.cat(ids);geometry=g.select(ids);target=data['actions'][ids]
    norm=FeatureNormalizer.fit(g.select(train));sn=StateNormalizer.fit(g.select(train),q[train],'goal');shared=sn.apply(geometry,q[ids],'goal');result={}
    for name in ['flat','center','surface']:
        seed_all(0,4);model=CommonPolicy(name,cfg);builder=GraphBuilder(center=name=='center');inputs=norm.apply(name,geometry,builder)
        optimizer=torch.optim.Adam(model.parameters(),lr=.003);initial=action_metrics(model(inputs,shared).detach(),target);start=perf_counter()
        for _ in range(args.steps):
            optimizer.zero_grad();raw=model.raw_action(inputs,shared);loss=bc_loss(raw,target)
            if not torch.isfinite(loss):raise FloatingPointError('overfit')
            loss.backward();optimizer.step()
        model.eval();prediction=model(inputs,shared).detach();final=action_metrics(prediction,target)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'smoke.pt';torch.save(model.state_dict(),path)
            loaded=CommonPolicy(name,cfg);loaded.load_state_dict(torch.load(path,weights_only=True));loaded.eval();same=torch.equal(prediction,loaded(inputs,shared).detach())
        passed=final['action_mse']<initial['action_mse']*.1 and same
        result[name]=dict(initial=initial,final=final,steps=args.steps,seconds=perf_counter()-start,checkpoint_identical=same,passed=passed,training_episode_ids=episodes,sample_indices=ids.tolist(),samples=len(ids),note='Optimization smoke on TRAIN expert data; not held-out performance or a deployment checkpoint')
        print(name,'initial',initial['action_mse'],'final',final['action_mse'],'passed',passed,flush=True)
    write_json(args.output,result)
    if not all(r['passed'] for r in result.values()):raise AssertionError('real demonstration small-batch overfit failed')

if __name__=='__main__':main()
