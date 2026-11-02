"""Evaluation-only new Development36 replay with original discontinuity handling.

Replay is not silently exact. Each stored RGB-D/calibration field must equal the
actual replay frame. New independent benchmark uses synchronous capture instead.
"""
import argparse
import fcntl
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS


def run(root,partition,partition_key='development',freeze=None):
    from utils.experiment_logger import ExperimentLogger
    from graph.geometry_reach10d.runtime import environment
    from graph.task10i_stage4a.development_reference import mesh_reference
    from graph.task10i_stage4a.replay_reference import seed_for
    from graph.gt_pretrain10f1.remediation_collect import portable
    from graph.corrective_reach10f.pilot import restore
    from recurrent.expert import ScriptedExpert
    if partition_key not in ('development','internal_holdout'):raise ValueError('unsupported partition')
    if partition_key=='internal_holdout':
        from .final_capture import freeze_barrier
        assert freeze is not None,'Internal GT requires the frozen candidate barrier'
        freeze_barrier(freeze,None,None,scope='internal')
    torch.set_num_threads(1);torch.manual_seed(42)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(source_sha256=sha(Path(__file__)),partition_sha256=sha(partition),partition_key=partition_key,
            candidate_freeze_sha256=sha(freeze/'candidate_config_frozen.json') if freeze else None)
        if (root/'source_config_frozen.json').exists():assert read(root/'source_config_frozen.json')==identity
        else:write(root/'source_config_frozen.json',identity)
        parents=set(read(partition)[partition_key])
        entries=sorted((e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()
                        if e['parent_id'] in parents),key=lambda e:str(e['episode_id']))
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_'+partition_key+'_reference',seed=42,
             root=root/'experiments',job_type='evaluation',config=identity) as logger:
            env=environment(Path('/workspace/task10g_f1_abc_20261031/full'));summary=[]
            try:
                for e in entries:
                    eid=str(e['episode_id']);folder=root/'references'/eid
                    if (folder/'receipt.json').exists():
                        r=read(folder/'receipt.json')
                        assert r['raw_sha256']==e['raw_sha256']
                        assert all(sha(folder/p)==h for p,h in r['files'].items())
                        summary.append(r);continue
                    if folder.exists():folder.rename(folder.with_name(eid+'.partial_'+str(time.time_ns())))
                    folder.mkdir(parents=True)
                    assert sha(e['raw_path'])==e['raw_sha256'] and sha(e['gt_path'])==e['gt_sha256']
                    obs,_=env.reset(seed=seed_for(e));task=env.unwrapped
                    mesh_reference(task,folder,e)
                    ids={int(k):v.name for k,v in task.segmentation_id_map.items()}
                    objects={n:[k for k,v in ids.items() if v.startswith(n)] for n in ('peg','box')}
                    robot_names={link.name for link in task.agent.robot.get_links()}
                    robot=[k for k,v in ids.items() if v in robot_names]
                    assert all(objects.values()) and robot
                    snapshot=None;expert=None;anchor=None
                    if e['source']=='new_stage3' and e['branch'].startswith('t'):
                        anchor=int(e['branch'].split('_')[0][1:])
                        snapshot=Path(e['raw_path']).parent.parent/f'anchor_t{anchor}.pt'
                        assert snapshot.exists();expert=ScriptedExpert(env)
                    segmentations=[];rows=[]
                    write(root/'pipeline_state.json',dict(status='RUNNING',episode=eid,pid=os.getpid(),
                        completed_episodes=len(summary),run_id=logger.run_id))
                    with h5py.File(e['raw_path']) as f:
                        for t in range(e['actions']+1):
                            if snapshot is not None and t==anchor:
                                state=portable(torch.load(snapshot,map_location='cpu',weights_only=False),True)
                                obs=restore(env,expert,state)
                            checks={};segs=[]
                            for c in CAMERAS:
                                sd=obs['sensor_data'][c];sp=obs['sensor_param'][c]
                                values=dict(rgb=sd['rgb'][0],depth_raw_mm=sd['depth'][0,...,0],
                                    intrinsic=sp['intrinsic_cv'][0],extrinsic_cv=sp['extrinsic_cv'][0],cam2world_gl=sp['cam2world_gl'][0])
                                checks[c]={}
                                for name,value in values.items():
                                    a=value.cpu().numpy();b=f[c+'/'+name][t]
                                    checks[c][name]=dict(exact=bool(np.array_equal(a,b)),
                                        different_values=int(np.count_nonzero(a!=b)),
                                        max_abs=float(np.max(abs(a.astype(float)-b.astype(float)))))
                                segs.append(task._sensors[c].get_obs(rgb=False,depth=False,position=False,
                                    segmentation=True)['segmentation'][0,...,0].cpu().numpy().copy())
                            exact=all(v['exact'] for c in checks.values() for v in c.values())
                            rows.append(dict(timestep=t,reference_kind='EXACT_GT' if exact else 'GT_UNSUPPORTED',comparisons=checks))
                            segmentations.append(segs)
                            if t<e['actions']:
                                spec=e.get('collection_spec',{})
                                if e['source']=='existing_full_task' and spec.get('paired') and t in spec['paired_protocol']['anchors']:
                                    from mani_skill.utils.structs.pose import Pose
                                    for name in ('peg','box'):
                                        actor=getattr(task,name);pose=actor.pose.raw_pose.clone()
                                        for offset in spec['paired_protocol']['offsets_mm']:
                                            moved=pose.clone();moved[0,:3]+=moved.new_tensor(offset)*.001
                                            actor.set_pose(Pose.create(moved));task.get_obs();actor.set_pose(Pose.create(pose))
                                obs,*_=env.step(torch.tensor(f['actions'][t])[None])
                    exact=np.asarray([r['reference_kind']=='EXACT_GT' for r in rows])
                    np.savez_compressed(folder/'segmentation.npz',actor_segmentation=np.asarray(segmentations),exact_frame=exact)
                    receipt=dict(episode_id=eid,parent_id=e['parent_id'],source=e['source'],branch=e['branch'],
                        actor_ids=ids,object_ids=objects,robot_ids=robot,raw_sha256=e['raw_sha256'],gt_sha256=e['gt_sha256'],
                        exact_frames=int(exact.sum()),total_frames=len(rows),rows=rows,
                        snapshot_sha256=sha(snapshot) if snapshot else None,GT_inference_input=False,
                        segmentation_sha256=sha(folder/'segmentation.npz'),
                        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
                        method='native action replay, original snapshot/paired discontinuities; complete RGB-D/calibration equality')
                    write(folder/'receipt.json',receipt);summary.append(receipt)
                    write(root/'progress.json',[{k:v for k,v in r.items() if k!='rows'} for r in summary])
                    print('new Development GT',len(summary),'/',len(entries),eid,int(exact.sum()),'/',len(rows),flush=True)
                assert len(summary)==len(entries)
                write(root/'pipeline_state.json',dict(status='NEW_DEVELOPMENT_REFERENCE_COMPLETE' if partition_key=='development' else 'INTERNAL_REFERENCE_COMPLETE',run_id=logger.run_id,
                    episodes=len(summary),frames=sum(r['total_frames'] for r in summary),
                    exact_frames=sum(r['exact_frames'] for r in summary),
                    unsupported_frames=sum(r['total_frames']-r['exact_frames'] for r in summary),whole_goal_complete=False))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),
                    completed_episodes=len(summary),partial_preserved=True));raise
            finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('partition',type=Path)
    p.add_argument('--partition-key',default='development');p.add_argument('--freeze',type=Path)
    a=p.parse_args();run(a.root,a.partition,a.partition_key,a.freeze)
