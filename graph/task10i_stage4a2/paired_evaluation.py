"""Streaming verified B0/A outputs into identical original current-only Graphs.

No model forward, mask selection, or threshold fitting occurs here. GT is read
only by this evaluator after per-episode inference receipts exist.
"""
import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import h5py
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from .evaluation_metrics import pixel_metrics,correct_source_coverage,hierarchical_paired
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def initialize_worker():
    import torch
    torch.set_num_threads(1)


def reference_points(cameras,segmentation,rr,oid):
    visible=[];boundary=[]
    for ci,c in enumerate(cameras):
        if oid==1 and c.camera!='base_camera':continue
        target=np.isin(segmentation[ci],rr['object_ids'][('peg','box')[oid]])&np.isfinite(c.depth)
        world=c.world_points();visible.append(world[target]);boundary.append(world[target&~binary_erosion(target)])
    return np.concatenate(visible).reshape(-1,3),np.concatenate(boundary).reshape(-1,3)


def evaluate_episode(e,inference,references,output,candidate='A'):
    if candidate=='B0' or not candidate.isidentifier():raise ValueError('distinct named challenger required')
    from graph.plan_a_bc.geometry import CurrentGeometry,validate_graph
    from visual.runtime.vision_packet import pose_matrix
    from graph.persistent_geometry.contracts import transform
    from graph.persistent_geometry.metrics import sample_meshes
    from graph.persistent_geometry.geometry_audit import aperture_reference
    eid=str(e['episode_id']);folder=Path(output)/'episodes'/eid
    inp=Path(inference)/'perception'/eid;ref=Path(references)/'references'/eid
    if (folder/'receipt.json').exists():
        r=read(folder/'receipt.json')
        assert r.get('candidate','A')==candidate
        assert r['raw_sha256']==e['raw_sha256']
        assert r['inference_receipt_sha256']==sha(inp/'receipt.json')
        assert r['reference_receipt_sha256']==sha(ref/'receipt.json')
        assert all(sha(folder/p)==h for p,h in r['files'].items())
        return r
    if folder.exists():folder.rename(folder.with_name(eid+'.partial_'+str(time.time_ns())))
    folder.mkdir(parents=True)
    ir=read(inp/'receipt.json');rr=read(ref/'receipt.json')
    assert ir['frames']==e['actions']+1
    if e.get('engineering_only'):assert rr['total_frames']>=ir['frames']
    else:assert rr['total_frames']==ir['frames']
    assert ir['raw_sha256']==rr['raw_sha256']==e['raw_sha256']
    assert all(sha(inp/p)==h for p,h in ir['files'].items())
    assert all(sha(ref/p)==h for p,h in rr['files'].items())
    assert sha(e['raw_path'])==e['raw_sha256']
    with np.load(ref/'segmentation.npz') as z:
        seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
    # An engineering prefix uses the full reference source, but its receipt
    # denominator must count only the frames actually evaluated.
    seg=seg[:e['actions']+1];exact=exact[:e['actions']+1]
    with np.load(ref/'geometry_reference.npz') as z:poses=z['actor_pose'].copy()
    with np.load(e['gt_path']) as z:
        holes=z['raw_hole_pose'].copy();radii=z['raw_hole_radius'].copy()
    meshes=sample_meshes(ref/'meshes.npz',samples_per_object=10000)
    rt=Path('/workspace/task10g_f1_abc_20261031/full')
    schema=read('/workspace/task10f1_gt_pretrain_20261030/normalizer.json')['schema']
    geometries={c:CurrentGeometry(rt/'inputs',schema) for c in ('B0',candidate)}
    masks={}
    for c in geometries:
        with np.load(inp/(c+'.npz')) as z:
            assert z['mask_bits'].shape==(e['actions']+1,8192)
            masks[c]=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
    pixels=[];points=[];paired=[];timing=[];proof=[]
    with h5py.File(e['raw_path']) as f:
        for t in range(e['actions']+1):
            cameras,data,surfaces=geometries['B0'].from_h5(f,t,eid)
            metadata=dict(episode=eid,parent_id=e['parent_id'],timestep=t,
                reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                branch=e['branch'],source=e['source'],terminal=t==e['actions'],
                segment=int(f['segment'][min(t,len(f['segment'])-1)]) if 'segment' in f else None,
                phase='NOT_RECORDED_IN_OBSERVATION')
            predictions={}
            for condition,geometry in geometries.items():
                start=time.perf_counter()
                # Original public builder receives current RGB-D, robot FK and
                # predicted masks only. No reference/actor labels are arguments.
                graph,records,gate=geometry.build(cameras,data,surfaces,dict(zip(CAMERAS,masks[condition][t])))
                validate_graph(graph)
                timing.append(dict(**metadata,condition=condition,graph_ms=1000*(time.perf_counter()-start)))
                arrays={k:getattr(graph,k).numpy() for k in ('node_input','node_type','edge_index','edge_attr','edge_type')}
                provenance=[]
                for r in records:
                    for episode,step,camera,pixel in r['source_observation_ids']:
                        assert str(episode)==eid and step==t
                        assert not (r['object_id']==1 and camera!='base_camera')
                        assert 0<=pixel<128*128
                    provenance.append(dict(object_id=r['object_id'],sources=r['source_observation_ids']))
                frame=folder/condition/f't{t:04d}';frame.mkdir(parents=True)
                np.savez_compressed(frame/'graph.npz',**arrays,
                    world_points=np.asarray([r['world_xyz'] for r in records]).reshape(-1,3),
                    object_id=np.asarray([r['object_id'] for r in records],np.int8))
                write(frame/'provenance.json',dict(sources=provenance,gate=gate,
                    current_only=True,box_main_only=True,cap128=True,graph21_edge4=True))
                predictions[condition]=(graph,records,arrays)
                for ci,camera in enumerate(CAMERAS):
                    for oid in (0,1):
                        row=dict(**metadata,condition=condition,camera=camera,object_id=oid,
                                 graph_eligible=not (ci==1 and oid==1))
                        if exact[t]:
                            target=np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]])
                            row.update(pixel_metrics(masks[condition][t,ci,oid],target,
                                np.isin(seg[t,ci],rr['robot_ids']),
                                np.isin(seg[t,ci],rr['object_ids'][('box','peg')[oid]])))
                        pixels.append(row)
                for oid in (0,1):
                    selected=[r for r in records if r['object_id']==oid]
                    row=dict(**metadata,condition=condition,object_id=oid,geometry_points=len(selected))
                    if exact[t]:
                        xyz=np.asarray([r['world_xyz'] for r in selected]).reshape(-1,3)
                        labels=[]
                        for r in selected:
                            _,_,camera,pixel=r['source_observation_ids'][0]
                            actor=int(seg[t,CAMERAS.index(camera)].ravel()[pixel])
                            labels.append('correct' if actor in rr['object_ids'][('peg','box')[oid]] else
                                'robot' if actor in rr['robot_ids'] else
                                'wrong_object' if actor in rr['object_ids'][('box','peg')[oid]] else 'background')
                        correct=np.asarray([v=='correct' for v in labels],bool)
                        visible,boundary=reference_points(cameras,seg[t],rr,oid)
                        row.update(correct_points=int(correct.sum()),visible_samples=len(visible),
                            correct_representative_fraction=float(correct.mean()) if len(correct) else None,
                            correct_source_coverage=correct_source_coverage(xyz,correct,visible),
                            correct_boundary_coverage=correct_source_coverage(xyz,correct,boundary),
                            correct_point_graph_missing=float(not correct.any()) if len(visible)>=8 else None,
                            false_object_presence=float(len(xyz)>0) if len(visible)==0 else None)
                        for label in ('robot','background','wrong_object'):
                            row[label+'_point_fraction']=labels.count(label)/len(labels) if labels else None
                            row[label+'_points']=labels.count(label)
                        if oid==0:
                            # Ends are evaluation-only outer20% along longest
                            # collision-mesh axis, restricted to currently visible GT.
                            local=transform(visible,np.linalg.inv(pose_matrix(poses[t,0])))
                            extent=np.ptp(meshes[0]['points'],axis=0);axis=int(extent.argmax())
                            lo,hi=meshes[0]['points'][:,axis].min(),meshes[0]['points'][:,axis].max()
                            ends=(local[:,axis]<=lo+.1*(hi-lo))|(local[:,axis]>=hi-.1*(hi-lo))
                            row['peg_end_reference_samples']=int(ends.sum())
                            row['peg_ends_coverage']=correct_source_coverage(xyz,correct,visible[ends])
                        else:
                            _,_,_,_,rim=aperture_reference(meshes,dict(hole_pose=holes[t],
                                actor_pose=poses[t],hole_radius=radii[t]))
                            observed=cKDTree(visible).query(rim)[0]<.004 if len(visible) else np.zeros(len(rim),bool)
                            row['box_observed_rim_reference_samples']=int(observed.sum())
                            row['box_observed_rim_coverage']=correct_source_coverage(xyz,correct,rim[observed])
                    points.append(row)
            same=np.array_equal(masks['B0'][t],masks[candidate][t])
            if same:
                assert all(predictions['B0'][2][k].tobytes()==predictions[candidate][2][k].tobytes()
                           for k in predictions['B0'][2]),('SAME_MASK_GRAPH_MISMATCH',eid,t)
            proof.append(dict(timestep=t,same_masks=same,same_mask_graph_equal=True if same else None))
            if exact[t]:
                for condition in ('B0',candidate):
                    target_rows=[r for r in pixels[-8:] if r['condition']==condition and r['object_id']==0]
                    paired.append(dict(**metadata,condition=condition,
                        both_camera_peg_correct_pixel_missing=float(all(r['tp']==0 for r in target_rows))
                        if len(target_rows)==2 and all(r['visible_pixels']>=8 for r in target_rows) else None))
    csv_write(folder/'pixel_metrics.csv',pixels);csv_write(folder/'graph_metrics.csv',points)
    write(folder/'paired_frame_metrics.json',paired);write(folder/'graph_control.json',proof)
    csv_write(folder/'graph_timing.csv',timing)
    receipt=dict(episode=eid,parent_id=e['parent_id'],candidate=candidate,frames=e['actions']+1,exact_frames=int(exact.sum()),
        raw_sha256=e['raw_sha256'],inference_receipt_sha256=sha(inp/'receipt.json'),
        reference_receipt_sha256=sha(ref/'receipt.json'),
        files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()},
        no_model_forwards=True,current_only=True,same_original_graph_builder=True)
    write(folder/'receipt.json',receipt)
    return receipt


def _main(root,inference,references,expected,engineering=None,candidate='A'):
    from utils.experiment_logger import ExperimentLogger
    entries=read(expected)['entries'];pending={};done=[];submitted=set()
    if engineering is not None:
        initialize_worker()
        engineering_entries=read(engineering/'expected_counts.json')['entries']
        assert len(engineering_entries)==1 and engineering_entries[0]['engineering_only']
        proof=evaluate_episode(engineering_entries[0],engineering,references,root/'engineering',candidate)
        assert proof['frames']==3
        write(root/'engineering_barrier.json',dict(status='PASS',receipt=proof))
    source=Path(__file__).resolve().parents[2]
    identity=dict(files={str(p.relative_to(source)):sha(p)
        for directory in ('graph/task10i_stage4a2','graph/plan_a_bc','graph/plan_a_geometry')
        for p in (source/directory).glob('*.py')},inference_root=str(inference),reference_root=str(references),
        expected_sha256=sha(expected),candidate=candidate)
    identity_path=root/'source_identity.json'
    if identity_path.exists():assert read(identity_path)==identity,'changed evaluator source/config needs new root'
    else:write(identity_path,identity)
    deadline=time.monotonic()+24*3600
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_paired_pixel_graph',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(inference=str(inference),references=str(references))) as log:
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
            while len(done)<len(entries):
                for e in entries:
                    eid=str(e['episode_id'])
                    if eid in submitted or len(pending)>=3:continue
                    if (inference/'perception'/eid/'receipt.json').exists() and (references/'references'/eid/'receipt.json').exists():
                        pending[pool.submit(evaluate_episode,e,inference,references,root,candidate)]=eid;submitted.add(eid)
                for future,eid in list(pending.items()):
                    if future.done():
                        done.append(future.result());del pending[future]
                        write(root/'progress.json',done);print('paired graph',len(done),'/',len(entries),eid,flush=True)
                write(root/'pipeline_state.json',dict(status='RUNNING',pid=os.getpid(),completed_episodes=len(done),
                    active_episodes=list(pending.values()),expected_episodes=len(entries),run_id=log.run_id))
                for dependency in (inference,references):
                    state=dependency/'pipeline_state.json'
                    if state.exists() and read(state)['status']=='FAILED':
                        raise RuntimeError('Inference/reference dependency failed: '+str(dependency))
                if time.monotonic()>deadline:raise TimeoutError('paired evaluation deadline')
                if len(done)<len(entries):time.sleep(5)
        assert sum(r['frames'] for r in done)==read(expected)['frames']
        write(root/'pipeline_state.json',dict(status='PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE',run_id=log.run_id,
            episodes=len(done),frames=sum(r['frames'] for r in done),exact_frames=sum(r['exact_frames'] for r in done),
            aggregation_pending=True,whole_goal_complete=False))


def main(root,inference,references,expected,engineering=None,candidate='A',resume_existing=False):
    root.mkdir(parents=True,exist_ok=resume_existing)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _main(root,inference,references,expected,engineering,candidate)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('inference',type=Path)
    p.add_argument('references',type=Path);p.add_argument('expected',type=Path)
    p.add_argument('--engineering',type=Path);p.add_argument('--engineering-only',action='store_true')
    p.add_argument('--candidate',default='A');p.add_argument('--resume-existing',action='store_true');a=p.parse_args()
    if a.engineering_only:
        initialize_worker()
        entry=read(a.expected)['entries'][0]
        assert entry.get('engineering_only')
        proof=evaluate_episode(entry,a.inference,a.references,a.root,a.candidate)
        assert proof['exact_frames']<=proof['frames']==3
        write(a.root/'engineering_barrier.json',dict(status='PASS',receipt=proof))
    else:main(a.root,a.inference,a.references,a.expected,a.engineering,a.candidate,a.resume_existing)
