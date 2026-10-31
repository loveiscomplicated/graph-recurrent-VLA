"""Actual CUDA cached-graph/history/masked-loss/save-resume engineering tests.

Weights produced here are discarded; no A/B model or checkpoint selection.
"""
import argparse,copy,os
from pathlib import Path
import numpy as np,torch
from graph.plan_a_bc.common import read,write,sha,checkpoint_load,tensor_identity
from graph.plan_a_bc.cache import CachedDataset,input_identity
from graph.plan_a_bc.model import CachedGraphPolicy,normalizer
from graph.plan_a_bc.train import train_update,optimizer,save,assert_nested_equal
from graph.integration_baseline.bc08_codec import ActionCodec
from graph.geometry_reach10d.resident import ResidentDataset
from graph.corrective_reach10f.masked import SupervisedDataset
from graph.integration_baseline.direct_train import seed_all
from utils.experiment_logger import ExperimentLogger

def run(cache,initial,output):
    assert os.getenv('CONTAINER_ID')=='57640710';assert torch.cuda.is_available();seed_all(42,1)
    output.mkdir(parents=True,exist_ok=False)
    cfg=read(cache/'config.json');cfgpath=cache/'config.json';receipts=[read(p) for p in (cache/'cache').glob('ep*/receipt.json')]
    if not (cache/'cache_manifest.json').exists():write(cache/'cache_manifest.json',dict(identity=input_identity(cache),episodes=receipts),immutable=True)
    ds=CachedDataset(cache);resident=ResidentDataset(ds);masks={}
    for eid in ds.episodes:
        with np.load(cache/'cache'/f'ep{eid}'/'supervision.npz') as a:masks[eid]=a['supervision_mask'].copy()
    cpu=SupervisedDataset(ds,masks);cuda=SupervisedDataset(resident,masks);norm=normalizer(cache);codec=ActionCodec(read(cache/'action_codec.json'))
    cp=checkpoint_load(initial);ids=sorted(ds.episodes)
    model=CachedGraphPolicy(norm.schema,'E0',**cfg['model']).cuda();model.load_state_dict(cp['model'],strict=True);model.eval();checks={}
    for start in [0,32]:
        a=cpu.batch(ids,start,32,'cuda');b=cuda.batch(ids,start,32,'cuda')
        for k in vars(a[0]):torch.testing.assert_close(getattr(a[0],k),getattr(b[0],k),rtol=0,atol=0)
        for x,y in zip(a[1:],b[1:]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    checks['resident_masked_batch_equivalence']='PASS'
    # Every frame enters GRU; labels alone are masked. Chunk/step scans agree.
    outputs=[];hidden=None
    with torch.no_grad():
        for start in [0,32]:
            g,s,y,mask=cuda.batch(ids,start,32,'cuda');pred,hidden=model(g,s,norm,hidden);outputs.append(pred)
            modified=y.clone();modified[~mask]=12345
            torch.testing.assert_close(codec.loss(pred,y,mask),codec.loss(pred,modified,mask),rtol=0,atol=0)
        expected=torch.cat(outputs,1);slow=[];h=None
        for t in range(64):
            g,s,_,_=cpu.batch(ids,t,1,'cuda');pred,h=model(g,s,norm,h);slow.append(pred)
        torch.testing.assert_close(expected,torch.cat(slow,1),rtol=1e-5,atol=1e-6);torch.testing.assert_close(hidden,h,rtol=1e-5,atol=1e-6)
        hprefix=[]
        for eid in ids:
            g,s,_,_=cpu.batch([eid],0,10,'cuda');_,hh=model(g,s,norm,None);hprefix.append(hh)
        torch.testing.assert_close(*hprefix,rtol=0,atol=0)
    checks.update(masked_loss='PASS',causal_chunk_step_history='PASS',independent_branch_prefix_hidden='PASS')
    with ExperimentLogger(team='graph',model='Task10F_CUDA_history_smoke_DISCARD',config=dict(source_checkpoint_sha256=sha(initial),smoke_only=True),root=output/'experiments',job_type='test') as log:
        opt,sched=optimizer(model,cfg);progress=dict(step=0,epoch=1,chunk=0,hidden={})
        before=tensor_identity(model.state_dict())
        for start in [0,32]:
            metric=train_update(model,opt,cuda,norm,codec,progress,ids,start,cfg,'cuda');progress['step']+=1;log.log(metric,step=progress['step'])
        assert before!=tensor_identity(model.state_dict());receipt=save(output,cache,model,opt,sched,progress,'E0');back=checkpoint_load(receipt['path'])
        other=CachedGraphPolicy(norm.schema,'E0',**cfg['model']).cuda();other.load_state_dict(back['model']);opt2,sched2=optimizer(other,cfg);opt2.load_state_dict(back['optimizer']);sched2.load_state_dict(back['scheduler']);p2=copy.deepcopy(back['progress'])
        train_update(model,opt,cuda,norm,codec,progress,ids,0,cfg,'cuda');train_update(other,opt2,cuda,norm,codec,p2,ids,0,cfg,'cuda')
        assert tensor_identity(model.state_dict())==tensor_identity(other.state_dict());assert_nested_equal(opt.state_dict(),opt2.state_dict())
        checks.update(CUDA_optimizer_update='PASS',hidden_detach='PASS',checkpoint_save_resume='PASS',formal_AB_budget='NOT_TESTED')
        write(output/'verification.json',dict(checks=checks,source_checkpoint_unchanged=sha(initial)==read(initial.parent/'initial_checkpoint_identity.json')['sha256'],weights_discarded=True,formal_training_executed=False),immutable=True);log.link_output(output/'verification.json',kind='metrics')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.cache,a.initial,a.output)
