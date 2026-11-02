"""All preregistered development references; no validation images are accessed."""
from pathlib import Path
import argparse
import shutil
import h5py
import numpy as np
import torch
from .run_pipeline import CAMERAS,STAGE3,read,write,sha
from .replay_reference import seed_for

def mesh_reference(task,folder,e):
    import sapien.physx as physx
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    from mani_skill.utils.structs.pose import Pose
    meshes={}
    for oid,name in enumerate(('peg','box')):
        count=0
        for entity in getattr(task,name)._objs:
            for component in entity.components:
                if isinstance(component,physx.PhysxRigidBaseComponent):
                    for mesh in get_component_meshes(component):
                        if mesh is not None:
                            meshes[f'object{oid}_vertices{count}']=np.asarray(mesh.vertices)
                            meshes[f'object{oid}_faces{count}']=np.asarray(mesh.faces);count+=1
        assert count
    np.savez_compressed(folder/'meshes.npz',**meshes)
    with np.load(e['gt_path'],allow_pickle=False) as z:
        peg=z['raw_peg_pose'].copy()
        if 'physics_box_pose' in z:box=z['physics_box_pose'].copy()
        else:box=(Pose.create(torch.tensor(z['raw_hole_pose']))*task.box_hole_offsets.inv()).raw_pose.cpu().numpy()
    np.savez_compressed(folder/'geometry_reference.npz',actor_pose=np.stack([peg,box],axis=1))

def run(root,previous,smoke=False,partition='development'):
    from graph.geometry_reach10d.runtime import environment
    torch.set_num_threads(1);torch.manual_seed(42)
    assert partition in ('development','internal_test','independent_validation')
    dev=set(read(root/'protocol.json')[partition])
    entries=sorted([e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values() if e['parent_id'] in dev],key=lambda e:str(e['episode_id']))
    if smoke:entries=[dict(e,actions=2,engineering_only=True,original_actions=e['actions']) for e in entries if str(e['episode_id'])=='2000003']
    env=environment(Path('/workspace/task10g_f1_abc_20261031/full'));summary=[]
    try:
        for e in entries:
            folder=root/'references'/str(e['episode_id']);folder.mkdir(parents=True,exist_ok=False)
            obs,_=env.reset(seed=seed_for(e));task=env.unwrapped
            mesh_reference(task,folder,e)
            prior=previous/'replay_reference'/str(e['episode_id'])
            if (prior/'receipt.json').exists() and not smoke:
                rr=read(prior/'receipt.json');assert rr['raw_sha256']==e['raw_sha256'] and rr['gt_sha256']==e['gt_sha256']
                shutil.copy2(prior/'segmentation.npz',folder/'segmentation.npz')
                rr.update(reused_reference=str(prior),segmentation_sha256=sha(folder/'segmentation.npz'))
                write(folder/'receipt.json',rr)
                summary.append({k:v for k,v in rr.items() if k!='rows'});write(root/'reference_progress.json',summary);continue
            assert sha(e['raw_path'])==e['raw_sha256'] and sha(e['gt_path'])==e['gt_sha256']
            ids={int(k):v.name for k,v in task.segmentation_id_map.items()}
            object_ids={n:[k for k,name in ids.items() if name.startswith(n)] for n in ('peg','box')}
            robot_names={link.name for link in task.agent.robot.get_links()};robot_ids=[k for k,n in ids.items() if n in robot_names]
            assert all(object_ids.values()) and robot_ids
            rows=[];segs=[]
            with h5py.File(e['raw_path'],'r') as f:
                for t in range(e['actions']+1):
                    comparisons={};ss=[]
                    for c in CAMERAS:
                        sd=obs['sensor_data'][c];sp=obs['sensor_param'][c]
                        values=dict(rgb=sd['rgb'][0],depth_raw_mm=sd['depth'][0,...,0],intrinsic=sp['intrinsic_cv'][0],extrinsic_cv=sp['extrinsic_cv'][0],cam2world_gl=sp['cam2world_gl'][0])
                        comparisons[c]={}
                        for k,v in values.items():
                            a=v.cpu().numpy();b=f[c+'/'+k][t]
                            comparisons[c][k]=dict(exact=bool(np.array_equal(a,b)),different_values=int(np.count_nonzero(a!=b)),max_abs=float(np.max(abs(a.astype(float)-b.astype(float)))))
                        ss.append(task._sensors[c].get_obs(rgb=False,depth=False,position=False,segmentation=True)['segmentation'][0,...,0].cpu().numpy().copy())
                    exact=all(v['exact'] for cc in comparisons.values() for v in cc.values())
                    rows.append(dict(timestep=t,reference_kind='EXACT_GT' if exact else 'GT_UNSUPPORTED',comparisons=comparisons));segs.append(ss)
                    if t<e['actions']:obs,*_=env.step(torch.tensor(f['actions'][t])[None])
            np.savez_compressed(folder/'segmentation.npz',actor_segmentation=np.asarray(segs),exact_frame=np.asarray([r['reference_kind']=='EXACT_GT' for r in rows]))
            rr=dict(episode_id=e['episode_id'],parent_id=e['parent_id'],source=e['source'],branch=e['branch'],seed=seed_for(e),
                actor_ids=ids,object_ids=object_ids,robot_ids=robot_ids,rows=rows,exact_frames=sum(r['reference_kind']=='EXACT_GT' for r in rows),
                total_frames=len(rows),raw_sha256=e['raw_sha256'],gt_sha256=e['gt_sha256'],segmentation_sha256=sha(folder/'segmentation.npz'),
                method='native saved-action replay; every frame paired RGB/depth/calibration equality; no tolerance upgrade',physics_steps=e['actions']*5)
            write(folder/'receipt.json',rr);summary.append({k:v for k,v in rr.items() if k!='rows'});write(root/'reference_progress.json',summary)
            print('development reference',len(summary),'/',len(entries),e['episode_id'],rr['exact_frames'],'/',rr['total_frames'],flush=True)
    finally:env.close()
    write(root/'development_reference_summary.json',dict(status='REFERENCE_AUDIT_FINISHED',episodes=len(summary),total_frames=sum(r['total_frames'] for r in summary),exact_frames=sum(r['exact_frames'] for r in summary),
        unsupported_frames=sum(r['total_frames']-r['exact_frames'] for r in summary),evaluation_partition=partition,validation_accessed=partition=='independent_validation',rows=summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--partition',choices=('development','internal_test','independent_validation'),default='development');a=p.parse_args();run(a.root,a.previous,a.smoke,a.partition)
