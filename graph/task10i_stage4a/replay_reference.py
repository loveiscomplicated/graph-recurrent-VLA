"""Validated action replay GT. Exact frame agreement is checked before labeling."""
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .run_pipeline import CAMERAS,STAGE3,read,write


def seed_for(e):
    if 'parent_seed' in e:return int(e['parent_seed'])
    return int(read(Path(e['raw_path']).parent.parent/'parent_result.json')['seed'])


def run(root):
    from graph.geometry_reach10d.runtime import environment
    from graph.integration_baseline.direct_train import seed_all
    seed_all(42,1);protocol=read(root/'protocol.json');dev=set(protocol['development'])
    entries=list(read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values());chosen=[]
    for family in sorted({(e['source'],e['branch']=='full_task') for e in entries if e['parent_id'] in dev}):
        pool=[e for e in entries if e['parent_id'] in dev and (e['source'],e['branch']=='full_task')==family]
        chosen.append(min(pool,key=lambda e:str(e['episode_id'])))
    env=environment(Path('/workspace/task10g_f1_abc_20261031/full'));summaries=[]
    try:
        for e in chosen:
            obs,_=env.reset(seed=seed_for(e));task=env.unwrapped;rows=[];segmentations=[]
            ids={int(k):v.name for k,v in task.segmentation_id_map.items()}
            object_ids={n:[k for k,name in ids.items() if name.startswith(n)] for n in ('peg','box')}
            robot_names={link.name for link in task.agent.robot.get_links()}
            robot_ids=[k for k,name in ids.items() if name in robot_names]
            assert all(object_ids.values()) and robot_ids
            with h5py.File(e['raw_path']) as f:
                for t in range(e['actions']+1):
                    differences={};segs=[]
                    for c in CAMERAS:
                        sd=obs['sensor_data'][c];sp=obs['sensor_param'][c]
                        values=dict(rgb=sd['rgb'][0],depth_raw_mm=sd['depth'][0,...,0],intrinsic=sp['intrinsic_cv'][0],extrinsic_cv=sp['extrinsic_cv'][0],cam2world_gl=sp['cam2world_gl'][0])
                        differences[c]={}
                        for k,v in values.items():
                            a=v.cpu().numpy();b=f[c+'/'+k][t]
                            differences[c][k]=dict(exact=bool(np.array_equal(a,b)),different_values=int(np.count_nonzero(a!=b)),max_abs=float(np.max(abs(a.astype(float)-b.astype(float)))))
                        seg=task._sensors[c].get_obs(rgb=False,depth=False,position=False,segmentation=True)['segmentation'][0,...,0].cpu().numpy().copy()
                        segs.append(seg)
                    exact=all(v['exact'] for d in differences.values() for v in d.values())
                    rows.append(dict(timestep=t,reference_kind='EXACT_GT' if exact else 'GT_UNSUPPORTED',comparisons=differences))
                    segmentations.append(segs)
                    if t<e['actions']:obs,*_=env.step(torch.tensor(f['actions'][t])[None])
            folder=root/'replay_reference'/str(e['episode_id']);folder.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(folder/'segmentation.npz',actor_segmentation=np.asarray(segmentations),
                exact_frame=np.asarray([r['reference_kind']=='EXACT_GT' for r in rows]))
            result=dict(episode_id=e['episode_id'],parent_id=e['parent_id'],source=e['source'],branch=e['branch'],
                seed=seed_for(e),actor_ids=ids,object_ids=object_ids,robot_ids=robot_ids,rows=rows,
                exact_frames=sum(r['reference_kind']=='EXACT_GT' for r in rows),total_frames=len(rows),
                raw_sha256=e['raw_sha256'],gt_sha256=e['gt_sha256'],method='verified native action replay, no expert replanning, no student GT',
                physics_steps=5*e['actions'])
            write(folder/'receipt.json',result);summaries.append({k:v for k,v in result.items() if k!='rows'})
            write(root/'replay_reference_probe.json',dict(status='RUNNING',episodes=summaries))
            print('verified action replay',e['episode_id'],result['exact_frames'],result['total_frames'],flush=True)
    finally:env.close()
    write(root/'replay_reference_probe.json',dict(status='PASS' if all(e['exact_frames']==e['total_frames'] for e in summaries) else 'PARTIAL_EXACT_REFERENCE',episodes=summaries,
        no_tolerance_upgrade=True,student_inference_GT=False,validation_pixels_accessed=False))
