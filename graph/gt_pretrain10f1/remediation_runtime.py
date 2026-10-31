"""Empirical cross-runtime causal graph equivalence on preregistered episodes.

This is an engineering contract check, not policy/state prediction evaluation.
"""
import argparse,gc,time
from pathlib import Path
import numpy as np,torch
from graph.plan_a_bc.common import read,write,sha
from graph.plan_a_bc.cache import make_vision,GRAPH_KEYS
from graph.geometry_reach10d.parallel_cache import make_pool,cache_episode
from graph.geometry_reach10d.optimizations import DetectionMemo
from utils.experiment_logger import ExperimentLogger
from recurrent.common import versions


def run(root,native,dino,sam):
    torch.set_num_threads(1);torch.manual_seed(42)
    plan=read(root/'comparison_plan.json');refs=read(root/'references.json')['episodes']
    assert len(refs)==len(plan['logical_capture_ids'])
    actual_versions=versions()
    assert actual_versions==read(native/'environment_worker0.json')['versions'],'native execution runtime drift'
    detector,tracker=make_vision('cuda',dino,sam)
    memo=DetectionMemo(detector,'Task10F1_cross_runtime',dict(dino=str(dino),sam_sha256=sha(sam)))
    rows=[]
    with ExperimentLogger(team='graph',model='Task10F1_cross_runtime_causal_graph_equivalence',root=root/'experiments',job_type='prepare',
        config=dict(plan=plan,architecture_or_loss_selection=False)) as log,make_pool(root,4) as pool:
        for ep in refs:
            raw=root/ep['raw_path'];assert sha(raw)==ep['raw_sha256']
            reference=root/ep['graph_path'];assert sha(reference)==ep['graph_sha256']
            entry=dict(episode_id=ep['sampling_episode_id'],seed=ep['parent'],split=ep['split'],actions=ep['actions'],path=str(raw),sha256=ep['raw_sha256'])
            result=cache_episode(root,Path('/'),entry,memo,tracker,log,pool,4)
            with np.load(reference,allow_pickle=False) as a,np.load(root/result['path'],allow_pickle=False) as b:
                keys=sorted(set(a.files)-{'generation_seconds','episode_id','action7'})
                differences=[]
                for k in keys:
                    if a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape or a[k].tobytes()!=b[k].tobytes():differences.append(k)
                source_gate=read(root/ep['gate_path']);native_gate=read((root/result['path']).with_name('gate_trace.json'))
                if source_gate!=native_gate:differences.append('gate_trace.json')
                rows.append(dict(logical_capture_id=ep['capture_id'],sampling_episode_id=entry['episode_id'],graphs=entry['actions'],
                    source_graph_sha256=ep['graph_sha256'],native_graph_sha256=result['sha256'],payload_arrays_checked=keys,
                    status='PASS' if not differences else 'FAIL',different_arrays=differences))
            write(root/'equivalence_progress.json',dict(episodes=rows))
            memo.cache.clear();gc.collect();torch.cuda.empty_cache()
    status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL'
    write(root/'runtime_equivalence.json',dict(status=status,plan_sha256=sha(root/'comparison_plan.json'),episodes=rows,
        causal_masks_geometry_provenance_exact=status=='PASS',native_environment_sha256=sha(native/'environment_worker0.json'),
        native_actual_versions=actual_versions,
        scope='entire causal sequences for preregistered episodes, including missing geometry and disturbed/recovery frames; no floating tolerance',
        not_a_proof_for_all_future_observations=True),immutable=True)
    print('RUNTIME_EQUIVALENCE',status,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True)
    a=p.parse_args();run(a.root,a.native,a.dino,a.sam)
