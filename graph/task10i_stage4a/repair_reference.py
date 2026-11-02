"""Replay collection discontinuities into NEW diagnostic references only."""
import argparse
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .run_pipeline import read,write,sha,STAGE3,CAMERAS
from .replay_reference import seed_for


def run(root,previous):
    from utils.experiment_logger import ExperimentLogger
    from graph.geometry_reach10d.runtime import environment
    from graph.gt_pretrain10f1.remediation_collect import portable
    from graph.corrective_reach10f.pilot import restore
    from recurrent.expert import ScriptedExpert
    torch.set_num_threads(1);torch.manual_seed(42);root.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_collection_discontinuity_reference_replay',seed=42,root=root/'experiments',job_type='evaluation',config=dict(previous_root=str(previous),student_GT_input=False)) as log:
        write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),source_sha256=sha(__file__)))
        write(root/'pipeline_state.json',dict(stage='WAITING_REFERENCE_MANIFEST',status='RUNNING',run_id=log.run_id,pid=os.getpid()))
        while not (previous/'development_reference_summary.json').exists():
            state=read(previous/'pipeline_state.json')
            if state.get('status')=='FAILED':raise RuntimeError('previous reference stage failed')
            time.sleep(10)
        entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()};summary=[]
        env=environment(Path('/workspace/task10g_f1_abc_20261031/full'))
        try:
            for rrpath in sorted((previous/'references').glob('*/receipt.json')):
                rr=read(rrpath);e=entries[str(rr['episode_id'])];spec=e.get('collection_spec',{})
                if rr['exact_frames']==rr['total_frames']:continue
                snapshot=None
                if e['source']=='new_stage3' and e['branch'].startswith('t'):
                    anchor=int(e['branch'].split('_')[0][1:]);snapshot=Path(e['raw_path']).parent.parent/f'anchor_t{anchor}.pt'
                    assert snapshot.exists()
                elif not (e['source']=='existing_full_task' and spec.get('paired')):continue
                out=root/'references'/str(e['episode_id']);out.mkdir(parents=True,exist_ok=False)
                obs,_=env.reset(seed=seed_for(e));task=env.unwrapped
                live_ids={str(k):v.name for k,v in task.segmentation_id_map.items()}
                assert live_ids=={str(k):v for k,v in rr['actor_ids'].items()},'GT actor mapping changed'
                write(root/'pipeline_state.json',dict(stage='COLLECTION_DISCONTINUITY_REFERENCE_REPLAY',status='RUNNING',episode_id=e['episode_id'],run_id=log.run_id,pid=os.getpid()))
                expert=ScriptedExpert(env) if snapshot else None
                with h5py.File(e['raw_path']) as f,np.load(rrpath.parent/'segmentation.npz') as original:
                    segs=original['actor_segmentation'].copy();exact=original['exact_frame'].copy();rows=list(rr['rows'])
                    if snapshot:
                        state=portable(torch.load(snapshot,map_location='cpu',weights_only=False),True)
                        obs=restore(env,expert,state);start=anchor
                    else:start=0
                    for t in range(start,e['actions']+1):
                        comparisons={};ss=[]
                        for c in CAMERAS:
                            sd=obs['sensor_data'][c];sp=obs['sensor_param'][c]
                            values=dict(rgb=sd['rgb'][0],depth_raw_mm=sd['depth'][0,...,0],intrinsic=sp['intrinsic_cv'][0],extrinsic_cv=sp['extrinsic_cv'][0],cam2world_gl=sp['cam2world_gl'][0]);comparisons[c]={}
                            for k,v in values.items():
                                a=v.cpu().numpy();b=f[c+'/'+k][t];comparisons[c][k]=dict(exact=bool(np.array_equal(a,b)),different_values=int(np.count_nonzero(a!=b)),max_abs=float(np.max(abs(a.astype(float)-b.astype(float)))))
                            ss.append(task._sensors[c].get_obs(rgb=False,depth=False,position=False,segmentation=True)['segmentation'][0,...,0].cpu().numpy().copy())
                        matches=all(v['exact'] for cc in comparisons.values() for v in cc.values())
                        if matches or not exact[t]:
                            # Keep previously verified exact labels even if the
                            # new replay has a mismatch at that separate frame.
                            segs[t]=ss;exact[t]=matches;rows[t]=dict(timestep=t,reference_kind='EXACT_GT' if matches else 'GT_UNSUPPORTED',comparisons=comparisons)
                        if t<e['actions']:
                            if not snapshot and t in spec['paired_protocol']['anchors']:
                                # Same nonphysical teleport/readback/restore sequence
                                # as original F1 collect.interventions, no added step.
                                from mani_skill.utils.structs.pose import Pose
                                for name in ('peg','box'):
                                    actor=getattr(task,name);pose=actor.pose.raw_pose.clone()
                                    for offset in spec['paired_protocol']['offsets_mm']:
                                        moved=pose.clone();moved[0,:3]+=moved.new_tensor(offset)*.001
                                        actor.set_pose(Pose.create(moved));task.get_obs()
                                        actor.set_pose(Pose.create(pose))
                            obs,*_=env.step(torch.tensor(f['actions'][t])[None])
                np.savez_compressed(out/'segmentation.npz',actor_segmentation=segs,exact_frame=exact)
                result=dict(**{k:v for k,v in rr.items() if k not in ('rows','exact_frames','method','segmentation_sha256')},rows=rows,exact_frames=int(exact.sum()),
                    previous_reference=str(rrpath),previous_reference_sha256=sha(rrpath),segmentation_sha256=sha(out/'segmentation.npz'),
                    snapshot=str(snapshot) if snapshot else None,snapshot_sha256=sha(snapshot) if snapshot else None,
                    method='verified original collection snapshot discontinuity replay' if snapshot else 'verified original paired actor teleport/readback/restore replay',
                    previous_exact_frames=rr['exact_frames'],student_GT_used=False)
                write(out/'receipt.json',result);summary.append({k:v for k,v in result.items() if k!='rows'})
                write(root/'repair_progress.json',summary);print('reference repair',e['episode_id'],rr['exact_frames'],'->',result['exact_frames'],'/',rr['total_frames'],flush=True)
        finally:env.close()
        write(root/'reference_repair_summary.json',dict(status='REFERENCE_REPAIR_DIAGNOSTIC_FINISHED',rows=summary,old_exact=sum(r['previous_exact_frames'] for r in summary),new_exact=sum(r['exact_frames'] for r in summary),no_tolerance_upgrade=True,original_references_modified=False))
        log.link_output(root/'reference_repair_summary.json',kind='metrics')
        write(root/'pipeline_state.json',dict(stage='REFERENCE_REPAIR_FINISHED',status='SUBSTAGE_FINISHED',completed_goal=False,run_id=log.run_id))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous',type=Path);a=p.parse_args();run(a.root,a.previous)
