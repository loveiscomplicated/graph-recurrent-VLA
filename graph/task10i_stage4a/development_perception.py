"""Causal GPU mask producer and bounded independent CPU graph audit workers."""
import argparse
import concurrent.futures
import hashlib
import multiprocessing
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .run_pipeline import STAGE3,CAMERAS,read,write,sha

GEOMETRY=None
REFERENCE_CACHE={}
MESH_CACHE={}
CURRENT_EPISODE=None

def initialize_worker():
    global GEOMETRY
    torch.set_num_threads(1)
    from graph.plan_a_bc.geometry import CurrentGeometry
    old=Path('/workspace/task10f1_gt_pretrain_20261030');runtime=Path('/workspace/task10g_f1_abc_20261031/full')
    GEOMETRY=CurrentGeometry(runtime/'inputs',read(old/'normalizer.json')['schema'])

def geometry_frame(root,e,t,mask):
    global CURRENT_EPISODE
    from .stagewise import trace_geometry
    rr=read(root/'references'/str(e['episode_id'])/'receipt.json')
    with h5py.File(e['raw_path'],'r') as f:
        if CURRENT_EPISODE!=str(e['episode_id']):GEOMETRY.reset();CURRENT_EPISODE=str(e['episode_id'])
        cameras,data,surfaces=GEOMETRY.from_h5(f,t,str(e['episode_id']))
        # GT is evaluator-only: trace never passes references to geometry methods.
        eid=str(e['episode_id'])
        if eid not in REFERENCE_CACHE:
            REFERENCE_CACHE.clear();MESH_CACHE.clear()
            with np.load(root/'references'/eid/'segmentation.npz',allow_pickle=False) as z:
                REFERENCE_CACHE[eid]={k:z[k].copy() for k in z.files}
            from graph.persistent_geometry.metrics import sample_meshes
            MESH_CACHE[eid]=sample_meshes(root/'references'/eid/'meshes.npz',samples_per_object=10000)
            with np.load(root/'references'/eid/'geometry_reference.npz') as poses:REFERENCE_CACHE[eid]['actor_pose']=poses['actor_pose'].copy()
        z=REFERENCE_CACHE[eid]
        refs={c:z['actor_segmentation'][t,i] if z['exact_frame'][t] else None for i,c in enumerate(CAMERAS)}
        graph,points,rows=trace_geometry(GEOMETRY,cameras,data,surfaces,dict(zip(CAMERAS,mask)),refs,[rr['object_ids']['peg'],rr['object_ids']['box']],rr['robot_ids'],root/'traces'/str(e['episode_id'])/f't{t:04d}')
    for r in rows:r.update(parent_id=e['parent_id'],branch=e['branch'],source=e['source'],terminal_observation=t==e['actions'])
    # Representative3D source purity is distinct from attached-source support.
    from scipy.spatial import cKDTree
    from scipy.ndimage import binary_erosion
    from visual.runtime.vision_packet import pose_matrix
    from graph.persistent_geometry.contracts import transform
    representative=[]
    for oid in (0,1):
        selected=[r for r in points if r['object_id']==oid];labels=[]
        if all(refs[c] is not None for c in CAMERAS):
            for point in selected:
                _,_,c,p=point['source_observation_ids'][0];actor=int(refs[c].ravel()[p])
                labels.append(0 if actor in rr['object_ids'][('peg','box')[oid]] else 1 if actor in rr['robot_ids'] else 2 if actor in rr['object_ids'][('box','peg')[oid]] else 3)
        spatial={}
        if all(refs[c] is not None for c in CAMERAS):
            xyz=np.asarray([r['world_xyz'] for r in selected]).reshape(-1,3)
            local=transform(xyz,np.linalg.inv(pose_matrix(z['actor_pose'][t,oid])))
            distance=MESH_CACHE[eid][oid]['tree'].query(local)[0]
            visible=[];boundary=[]
            for camera in cameras:
                if oid==1 and camera.camera!='base_camera':continue
                gtmask=np.isin(refs[camera.camera],rr['object_ids'][('peg','box')[oid]])&np.isfinite(camera.depth)
                world=camera.world_points();visible.append(world[gtmask]);boundary.append(world[gtmask&~binary_erosion(gtmask)])
            vv=np.concatenate(visible).reshape(-1,3);bb=np.concatenate(boundary).reshape(-1,3)
            tree=cKDTree(xyz) if len(xyz) else None
            spatial=dict(surface_distance_mean_mm=float(distance.mean()*1000) if len(distance) else None,surface_distance_p95_mm=float(np.percentile(distance,95)*1000) if len(distance) else None,
                visible_reference_points=len(vv),visible_coverage_4mm=float((tree.query(vv)[0]<.004).mean()) if tree and len(vv) else 0. if len(vv) else None,
                projected_boundary_reference_points=len(bb),projected_boundary_coverage_4mm=float((tree.query(bb)[0]<.004).mean()) if tree and len(bb) else 0. if len(bb) else None,
                surface_reference='10000 deterministic area-weighted external collision-union samples; approximate mesh distance, not exact analytic distance',
                normal_valid_fraction=float(np.mean([r['last_feature_valid'][0] for r in selected])) if selected else None)
        representative.append(dict(**spatial,episode=str(e['episode_id']),parent_id=e['parent_id'],timestep=t,object_id=oid,points=len(selected),
            correct=sum(x==0 for x in labels) if all(refs[c] is not None for c in CAMERAS) else None,robot=sum(x==1 for x in labels) if all(refs[c] is not None for c in CAMERAS) else None,wrong_object=sum(x==2 for x in labels) if all(refs[c] is not None for c in CAMERAS) else None,background=sum(x==3 for x in labels) if all(refs[c] is not None for c in CAMERAS) else None,
            reference_kind='EXACT_GT' if all(refs[c] is not None for c in CAMERAS) else 'GT_UNSUPPORTED'))
    return rows,representative

class Memo:
    def __init__(self,detector):self.detector=detector;self.cache={};self.calls=[]
    def __call__(self,rgb,prompt):
        key=(hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),prompt)
        hit=key in self.cache
        if not hit:self.cache[key]=self.detector(rgb,prompt)
        self.calls.append(dict(**self.detector.context,prompt=prompt,rgb_sha256=key[0],memo_hit=hit))
        return self.cache[key]

def run(root,previous,smoke=False,partition='development'):
    from .instrumentation import AuditedDino
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from visual.runtime.tracking import SAM2Tracker
    torch.set_num_threads(1);torch.manual_seed(42)
    assert not torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.allow_tf32
    assert partition in ('development','internal_test','independent_validation')
    dev=set(read(root/'protocol.json')[partition]);entries=sorted([e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values() if e['parent_id'] in dev],key=lambda e:str(e['episode_id']))
    if smoke:entries=[dict(e,actions=2,engineering_only=True,original_actions=e['actions']) for e in entries if str(e['episode_id'])=='2000003']
    runtime=read(Path('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json'))
    detector=AuditedDino(device='cuda',model_id=runtime['dino'],output=root/'dino_raw');memo=Memo(detector)
    reused=root/'reuse_manifest.json'
    if reused.exists():
        prior=Path(read(reused)['previous_baseline_root'])
        detector.events=read(prior/'dino_events.json');memo.calls=read(prior/'dino_memo_calls.json')
    tracker=SAM2Tracker(device='cuda',checkpoint=runtime['sam']);summary=[]
    (root/'model_identity.json').write_text(__import__('json').dumps(dict(sam_sha256=sha(runtime['sam']),dino_files={str(p.relative_to(Path(runtime['dino']))):sha(p) for p in Path(runtime['dino']).rglob('*') if p.is_file()},torch_version=torch.__version__,settings=dict(deterministic=torch.are_deterministic_algorithms_enabled(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32)),indent=2)+'\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
        try:
            for e in entries:
                folder=root/'perception'/str(e['episode_id'])
                if (folder/'receipt.json').exists():
                    old_receipt=read(folder/'receipt.json')
                    assert old_receipt['raw_sha256']==e['raw_sha256'] and sha(folder/'masks.npz')==old_receipt['masks_sha256'] and sha(folder/'stage_rows.json')==old_receipt['rows_sha256']
                    summary.append(old_receipt);write(root/'perception_progress.json',summary)
                    print('reused verified complete episode',e['episode_id'],flush=True);continue
                folder.mkdir(parents=True,exist_ok=False)
                adapters={c:OnlineVisionAdapter(memo,tracker) for c in CAMERAS}
                for a in adapters.values():a.reset(str(e['episode_id']))
                pending=[];allrows=[];rep=[];maskbits=[];vision_records=[];times=[]
                oldcache=Path('/workspace/task10f1_gt_pretrain_20261030/cache')/('ep'+str(e['episode_id']))/'graphs.npz'
                reuse=str(e['episode_id'])==read(previous/'baseline_forward_equivalence.json')['episode_id']
                historical=np.load(oldcache,allow_pickle=False) if reuse else None
                try:
                    with h5py.File(e['raw_path'],'r') as f:
                        for t in range(e['actions']+1):
                            tick=time.perf_counter();mm=[]
                            # Reuse ONLY the actually replay-verified full prefix.
                            # Terminal t=T needs the same causal states, so process it
                            # only in a fresh full forward if terminal is requested.
                            # Consequently all frames use one coherent episode pass.
                            for c in CAMERAS:
                                detector.context=dict(episode_id=str(e['episode_id']),parent_id=e['parent_id'],camera=c,decision_timestep=t)
                                masks,record=adapters[c].process(f[c+'/rgb'][t],str(e['episode_id']),t);mm.append(masks);vision_records.append(dict(camera=c,**record))
                            mm=np.asarray(mm);maskbits.append(np.packbits(mm.ravel()));times.append((time.perf_counter()-tick)*1000)
                            if historical is not None and t<e['actions'] and not np.array_equal(maskbits[-1],historical['mask_bits'][t]):raise AssertionError('frozen baseline differs from original cache')
                            pending.append(pool.submit(geometry_frame,root,e,t,mm))
                            if len(pending)>=9:
                                rows,rr=pending.pop(0).result();allrows.extend(rows);rep.extend(rr)
                        for future in pending:
                            rows,rr=future.result();allrows.extend(rows);rep.extend(rr)
                    np.savez_compressed(folder/'masks.npz',mask_bits=np.asarray(maskbits),vision_ms=np.asarray(times))
                    write(folder/'vision_records.json',vision_records);write(folder/'stage_rows.json',allrows);write(folder/'representative_points.json',rep)
                    receipt=dict(episode_id=e['episode_id'],parent_id=e['parent_id'],frames=e['actions']+1,raw_sha256=e['raw_sha256'],masks_sha256=sha(folder/'masks.npz'),rows_sha256=sha(folder/'stage_rows.json'),
                        terminal_included=True,causal_episode_camera_reset=True,student_GT_used=False,baseline_original_comparison='PASS' if reuse else 'not a historical forward-equivalence fixture')
                    write(folder/'receipt.json',receipt);summary.append(receipt)
                    write(root/'perception_progress.json',summary);write(root/'dino_events.json',detector.events);write(root/'dino_memo_calls.json',memo.calls)
                    print('development perception',len(summary),'/',len(entries),e['episode_id'],receipt['frames'],flush=True)
                finally:
                    for a in adapters.values():a.close()
                    memo.cache.clear()
                    if historical is not None:historical.close()
        finally:detector.close()
    write(root/'development_perception_summary.json',dict(status='BASELINE_AUDIT_FINISHED',evaluation_partition=partition,episodes=len(summary),frames=sum(r['frames'] for r in summary),validation_accessed=partition=='independent_validation',rows=summary))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--partition',choices=('development','internal_test','independent_validation'),default='development');a=p.parse_args();run(a.root,a.previous,a.smoke,a.partition)
