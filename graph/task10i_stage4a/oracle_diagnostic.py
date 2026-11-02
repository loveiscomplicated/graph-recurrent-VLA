"""O3 GT-mask and O4 gate-only diagnostic, never practical candidate inference."""
import argparse
import os
from pathlib import Path
import h5py
import numpy as np
from .run_pipeline import STAGE3,CAMERAS,read,write,sha
from .stagewise import trace_geometry


def run(root,previous):
    import fcntl
    root.mkdir(parents=True,exist_ok=False)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from utils.experiment_logger import ExperimentLogger
        from graph.plan_a_bc.geometry import CurrentGeometry
        old=Path('/workspace/task10f1_gt_pretrain_20261030');runtime=Path('/workspace/task10g_f1_abc_20261031/full')
        assert read(previous/'baseline_forward_equivalence.json')['status']=='PASS'
        source=Path(__file__).resolve().parents[2];old_source=read(previous/'source_freeze.json')['files']
        changed=[k for k,v in old_source.items() if not k.startswith('graph/task10i_stage4a/') and sha(source/k)!=v]
        assert not changed,changed
        entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
        with ExperimentLogger(team='integration',model='Task10I_Stage4A_ORACLE_DIAGNOSTIC',seed=42,root=root/'experiments',job_type='evaluation',config=dict(previous_root=str(previous),condition_type='SIM_ONLY_ORACLE',practical_candidate=False)) as log:
            write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),previous_root=str(previous),source_sha256=sha(__file__)))
            allrows=[];summary=[]
            try:
                for rrpath in sorted((previous/'replay_reference').glob('*/receipt.json')):
                    rr=read(rrpath);eid=str(rr['episode_id']);e=entries[eid];cache=old/'cache'/('ep'+eid)/'graphs.npz'
                    if not cache.exists():continue
                    cr=read(cache.parent/'receipt.json');assert sha(cache)==cr['sha256'] and cr['raw_episode_sha256']==e['raw_sha256']
                    for condition in ('O3_GT_VISIBLE_MASK','O4_WRIST_GATE_REMOVED_DIAGNOSTIC'):
                        write(root/'pipeline_state.json',dict(stage='ORACLE_CAUSAL_ABLATIONS_PRELIMINARY',status='RUNNING',condition=condition,episode_id=eid,run_id=log.run_id,pid=os.getpid()))
                        geom=CurrentGeometry(runtime/'inputs',read(old/'normalizer.json')['schema'])
                        if condition.startswith('O4'):
                            original=geom.system.prepare
                            def gate_removed(*args):
                                out=original(*args)
                                out['conditions']['C3']={c:m.copy() for c,m in out['conditions']['C2'].items()}
                                out['gate']['diagnostic_bypass']=True
                                return out
                            geom.system.prepare=gate_removed
                        with h5py.File(e['raw_path']) as f,np.load(cache,allow_pickle=False) as cached,np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as gt:
                            for t in range(e['actions']):
                                if not gt['exact_frame'][t]:
                                    allrows.append(dict(episode_id=eid,timestep=t,condition=condition,status='GT_UNSUPPORTED'));continue
                                cameras,data,surfaces=geom.from_h5(f,t,eid)
                                references={c:gt['actor_segmentation'][t,i] for i,c in enumerate(CAMERAS)}
                                if condition.startswith('O3'):
                                    masks={c:np.stack([np.isin(references[c],rr['object_ids'][n]) for n in ('peg','box')]) for c in CAMERAS}
                                else:masks=dict(zip(CAMERAS,np.unpackbits(cached['mask_bits'][t]).reshape(2,2,128,128).astype(bool)))
                                graph,points,rows=trace_geometry(geom,cameras,data,surfaces,masks,references,[rr['object_ids']['peg'],rr['object_ids']['box']],rr['robot_ids'],root/'traces'/condition/eid/f't{t:04d}')
                                for row in rows:row.update(parent_id=e['parent_id'],condition=condition,condition_type='ORACLE_DIAGNOSTIC',student_GT_input=condition.startswith('O3'))
                                allrows.extend(rows)
                                if t%25==0:print(condition,eid,t,flush=True)
                        summary.append(dict(episode_id=eid,condition=condition,frames=e['actions'],condition_type='ORACLE_DIAGNOSTIC',scope='preliminary one development episode; terminal graph remains pending'))
                        write(root/'oracle_rows.json',allrows)
                write(root/'oracle_summary.json',dict(status='PRELIMINARY_NOT_FULL_DATASET',episodes=summary,practical_adoption=False,source_dependency_equality=True))
                log.link_output(root/'oracle_summary.json',kind='metrics')
                write(root/'pipeline_state.json',dict(stage='ORACLE_PRELIMINARY_FINISHED',status='SUBSTAGE_FINISHED',completed_goal=False,run_id=log.run_id))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous',type=Path);a=p.parse_args();run(a.root,a.previous)
