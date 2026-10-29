"""Queue completed raw episodes into the unchanged pre-GNN Plan A writer."""
import argparse
import copy
from contextlib import nullcontext
import fcntl
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,append_csv,now
from graph.plan_a_bc.cache import cache_episode,make_vision,input_identity
from graph.geometry_reach10d.optimizations import DetectionMemo,components
from graph.geometry_reach10d.profile import Resources


def adopt_pilot(root,entry,log):
    old=root/'pilot_cache/optimized/cache'/f'ep{entry["episode_id"]}'
    if not (old/'receipt.json').exists():return None
    receipt=read(old/'receipt.json');expected=input_identity(root)
    assert receipt['raw_episode_sha256']==entry['sha256']
    for k in ['graph_config','frozen_geometry_protocol_sha256','frozen_source_sha256','cache_version']:
        assert receipt['identity'][k]==expected[k]
    assert sha(old/'graphs.npz')==receipt['sha256']
    folder=root/'cache'/old.name;folder.mkdir(parents=True,exist_ok=True)
    for n in ['graphs.npz','gate_trace.json']:
        if not (folder/n).exists():shutil.copy2(old/n,folder/n)
    receipt.update(identity=expected,split=entry['split'],path=str((folder/'graphs.npz').relative_to(root)),
        adopted_from=str(old/'receipt.json'),adopted_receipt_sha256=sha(old/'receipt.json'))
    write(folder/'receipt.json',receipt,immutable=True);log.link_output(folder/'receipt.json',kind='metrics')
    return receipt


def instrument(root):
    """Timing wrappers delegate unchanged functions with unchanged arguments."""
    from graph.plan_a_bc.geometry import CurrentGeometry
    context={}
    old=CurrentGeometry.from_h5
    def source(self,f,t,episode):
        context.update(t=t,episode_id=int(episode));return old(self,f,t,episode)
    CurrentGeometry.from_h5=source
    for name in ['prepare','represent','graph']:
        from graph.plan_a_geometry.core import PlanA
        original=getattr(PlanA,name)
        def timed(self,*args,_original=original,_name=name,**kwargs):
            start=time.perf_counter();out=_original(self,*args,**kwargs)
            append_csv(root/'pipeline_profile.csv',dict(stage='geometry_'+_name,
                episode_id=context.get('episode_id',-1),t=context.get('t',-1),seconds=time.perf_counter()-start))
            return out
        setattr(PlanA,name,timed)
    return context


def run(root,stream=False):
    torch.set_num_threads(1);torch.manual_seed(42)
    assert read(root/'optimization_equivalence.json')['status']=='PASS'
    candidates=read(root/'collection_candidates.json')
    paths=read(root/'runtime_paths.json')
    dino,tracker=make_vision('cuda',paths['dino'],paths['sam'])
    context=instrument(root)
    model_identity=read(root/'optimization_equivalence.json')['model_identity']
    detector=DetectionMemo(dino,'Task10D_full_cache',model_identity)
    import graph.plan_a_geometry.core as core
    core.components=components
    gate=read(root/'isolated_parallel_cache_gate.json') if (root/'isolated_parallel_cache_gate.json').exists() else {}
    parallel=bool(gate.get('adopted'))
    if parallel:
        from graph.geometry_reach10d.parallel_cache import make_pool,cache_episode as parallel_episode
        assert sha(Path(__file__).with_name('parallel_cache.py'))==gate['writer_source_sha256']
        pool_context=make_pool(root,gate['workers'])
    else:pool_context=nullcontext(None)
    with pool_context as pool,ExperimentLogger(team='graph',model='Task10D_causal_graph_cache',seed=42,
            config=dict(comparison_uuid=read(root/'run_identity.json')['run_id'],optimized=True),
            root=root/'experiments',job_type='prepare') as log,Resources(root,'full_graph_cache'):
        pending=list(candidates);accepted=[];rejected=[]
        started=time.perf_counter()
        while pending:
            progress=False
            for candidate in list(pending):
                spec=candidate['spec'];raw=root/'raw'/str(spec['episode_id'])/'receipt.json'
                if not raw.exists():continue
                receipt=read(raw)
                assert receipt['spec']==spec
                pending.remove(candidate);progress=True
                if receipt['status']!='ACCEPTED':rejected.append(receipt);continue
                entry=dict(episode_id=spec['episode_id'],seed=spec['seed'],split=candidate['effective_split'],
                    actions=receipt['actions'],path=str(Path(receipt['raw_path']).relative_to(root)),sha256=receipt['sha256'])
                folder=root/'cache'/f'ep{entry["episode_id"]}';folder.parent.mkdir(exist_ok=True)
                lock=root/'cache'/f'ep{entry["episode_id"]}.lock'
                with lock.open('a') as f:
                    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    if not (folder/'receipt.json').exists() and folder.exists():
                        folder.rename(folder.with_name(folder.name+'.interrupted.'+str(time.time_ns())))
                    old=adopt_pilot(root,entry,log) if not (folder/'receipt.json').exists() else None
                    tick=time.perf_counter()
                    result=old or (parallel_episode(root,root,entry,detector,tracker,log,pool,gate['workers']) if parallel else cache_episode(root,root,entry,detector,tracker,log))
                result.update(parent_seed=spec['seed'],variant_id=spec['variant'],data_kind=candidate['data_kind'])
                # Offset/GT live solely in collection metadata, never NPZ inputs.
                accepted.append(result);detector.cache.clear();torch.cuda.empty_cache()
                write(root/'graph_cache_progress.json',dict(completed=len(accepted),pending=len(pending),rejected=len(rejected),
                    graphs=sum(e['graphs'] for e in accepted),last_episode=spec['episode_id'],updated_at=now()))
                append_csv(root/'cache_profile.csv',dict(episode_id=spec['episode_id'],frames=entry['actions'],
                    seconds=time.perf_counter()-tick,reused=old is not None))
            if pending and not progress:
                if not stream:raise ValueError('collection incomplete')
                failed=[p for p in root.glob('collection_worker*_failure.json')]
                if failed:raise RuntimeError('collection worker failure: '+str(failed))
                if (root/'STOP').exists():raise RuntimeError('cooperative stop')
                time.sleep(2)
        manifest=dict(identity=input_identity(root),episodes=accepted,rejected=rejected,source_before_gnn=True,
            complete_train_validation=True,generated_at=now(),wall_seconds=time.perf_counter()-started)
        write(root/'cache_manifest.json',manifest);write(root/'graph_cache_manifest.json',manifest)
        print('FULL CACHE COMPLETE',len(accepted),sum(e['graphs'] for e in accepted),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stream',action='store_true')
    a=p.parse_args();run(a.root,a.stream)
