"""Discarded engineering-only E0 updates on real pilot caches, no A/B score."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch
from graph.plan_a_bc.common import read,write,tensor_identity
from graph.plan_a_bc.cache import CachedDataset,input_identity
from graph.plan_a_bc.model import CachedGraphPolicy,normalizer
from graph.plan_a_bc.train import optimizer,train_update,assert_nested_equal
from graph.integration_baseline.bc08_codec import ActionCodec
from graph.integration_baseline.direct_train import seed_all
from graph.geometry_reach10d.resident import ResidentDataset
from graph.geometry_reach10d.profile import Resources


def run(root):
    pilot=root/'pilot_cache/original'
    episodes=[read(p) for p in (pilot/'cache').glob('ep*/receipt.json')]
    write(pilot/'cache_manifest.json',dict(identity=input_identity(pilot),episodes=episodes,complete_train_validation=False))
    ds=CachedDataset(pilot);resident=ResidentDataset(ds);norm=normalizer(root);codec=ActionCodec(read(root/'action_codec.json'))
    ids=sorted(ds.arrays);cfg=read(root/'config.json');costs={}
    with Resources(root,'discarded_engineering_CUDA_training'):
        for name,data in [('original_cpu_pack',ds),('resident_cuda_pack',resident)]:
            seed_all(42,1);model=CachedGraphPolicy(norm.schema,'E0',**cfg['model']).cuda()
            initial=tensor_identity(model.state_dict());opt,sched=optimizer(model,cfg)
            progress=dict(step=0,epoch=1,chunk=0,hidden={})
            torch.cuda.synchronize();start=time.perf_counter();records=[]
            for i in range(4):
                offset=0 if i%2==0 else 32
                selected=[e for e in ids if offset<ds.episodes[e]['actions']]
                records.append(train_update(model,opt,data,norm,codec,progress,selected,offset,cfg,'cuda'))
            torch.cuda.synchronize();costs[name]=(time.perf_counter()-start)/4
            if name=='original_cpu_pack':expected=copy.deepcopy(model.state_dict());init=initial
            else:assert_nested_equal(expected,model.state_dict());assert initial==init
    write(root/'engineering_training_profile.json',dict(status='PASS',four_update_weights_exact=True,
        initial_weight_sha256=init,seconds_per_update=costs,adopted='resident CUDA packing',
        CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),weights_discarded=True,
        no_policy_performance_evaluated=True,training_budget_decision='retain maximum 11487 updates per A/B model'),immutable=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
