"""Evaluate matched frozen GNNs on Transformer tracks with and without rigid Box correction."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from collect_moving import schedule
from data_collection.episode_dataset import file_sha256
from data_collection.object21_bc import predict_graphs
from graph.gated_mpnn import VertexEdgeGatedMPNN
from data_collection.hole_motion import EpisodeHoleMotion
from data_collection.scripted_policy import make_policy
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.initial_corner_detection import verify_camera
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from utils.experiment_logger import ExperimentLogger, write
from visual.corner_detection import TorchCornerDetector

from oracle_reference import CAMERA, target
from rigid_box import RigidBoxTranslation

def observe(obs, detector, tracker, rigid):
    img={k:v[0].cpu().numpy() for k,v in obs['sensor_data']['base_camera'].items()}
    camera={k:v[0].cpu().numpy() for k,v in obs['sensor_param']['base_camera'].items()}
    verify_camera(img,camera,CAMERA)
    raw=detector(img['rgb'],img['depth'],camera['intrinsic_cv'],camera['extrinsic_cv'])
    xyz,kinds=raw.xyz.cpu().numpy(),raw.object_id.cpu().numpy()
    robot=np.r_[obs['extra']['tcp_pose'][0].cpu().numpy(),
                obs['agent']['qpos'][0,-2:].cpu().numpy()].astype(np.float32)
    tracks=tracker.step(xyz,kinds,robot)
    track_types=np.array([t.kind for t in tracker.tracks],np.int64)
    seen=tracks['observed']
    before=tracks['positions']
    np.testing.assert_array_equal(before[tracks['detection_tracks']],xyz)
    exterior_ids={int(tid) for tid,oid,meta in zip(tracks['detection_tracks'],kinds,raw.metadata,strict=True)
                  if oid==1 and not meta.get('contour_hole',False) and meta.get('method')!='box_opening'}
    after,fit=rigid.update(before,seen,track_types,exterior_ids)
    np.testing.assert_array_equal(after[seen | (track_types==0)],before[seen | (track_types==0)])
    graph=((before,track_types,seen),(after,track_types,seen))
    return graph,robot[:3],dict(fit=fit,total_tracks=len(seen),observed=int(seen.sum()),
                                corrected_box_points=int((np.linalg.norm(after-before,axis=1)>1e-7).sum()))


@torch.inference_mode()
def episode(env,seed,detector,model,gnns,reference,device):
    obs,_=env.reset(seed=seed)
    policy=make_policy()
    policy.reset(env,seed)
    task=env.unwrapped
    motion=None
    tracker=DetectedCornerTracker(model,.030,device)
    rigid=RigidBoxTranslation(use_velocity=True)
    rows,tcp_rows=[],[]
    frames={'transformer':[],'rigid':[]}
    elapsed_motion=False
    offset=np.zeros(3)
    terminal='step_limit'
    for step in range(200):
        phase=policy.phase
        if phase=='insert':
            terminal='insert_entry'
            break
        held=bool(task.agent.is_grasping(task.peg)[0])
        if motion is None and phase=='align' and held:
            motion=EpisodeHoleMotion(env,schedule(seed,step/task.control_freq),policy)
        previous=offset
        if motion is not None:
            if motion.advance(step,policy):
                obs=task.get_obs(task.get_info())
            offset=motion.frame_data()['offset_m']
        moving=np.linalg.norm(offset-previous)>1e-6
        elapsed_motion|=bool(moving)
        gt={key:value[0].cpu().numpy() for key,value in vars(get_gt_state(env)).items()}
        ref=reference[step]
        assert phase==ref['phase'] and held==ref['grasp'] and moving==ref['moving_now']
        np.testing.assert_allclose(offset,ref['offset_m'],atol=1e-7,rtol=0)
        tcp=obs['extra']['tcp_pose'][0,:3].cpu().numpy()
        np.testing.assert_allclose(target(gt,tcp),ref['target'],atol=1e-6,rtol=0)
        graph,tcp,metrics=observe(obs,detector,tracker,rigid)
        for key,frame in zip(frames,graph,strict=True):
            frames[key].append(frame)
        tcp_rows.append(tcp)
        rows.append(dict(seed=seed,step=step,phase=phase,grasp=held,moving=bool(moving),
                         window='during' if moving else 'after' if elapsed_motion else 'before',
                         offset_m=offset.tolist(),gt21=ref['target'].tolist(),**metrics))
        obs,_,term,trunc,_=env.step(policy.act(obs))
        if bool(term[0]) or bool(trunc[0]):
            terminal='terminated' if bool(term[0]) else 'truncated'
            break
    assert len(rows)==len(reference) and terminal=='insert_entry',(seed,len(rows),len(reference),terminal)
    tcp_array=np.asarray(tcp_rows)
    for model_name,gnn in gnns.items():
        for mode,inputs in frames.items():
            pred=predict_graphs(gnn,inputs,tcp_array,32,device)
            assert pred.shape==(len(rows),21) and np.isfinite(pred).all(),(seed,model_name,mode)
            for row,out in zip(rows,pred,strict=True):
                row[f'{model_name}_{mode}']=out.tolist()
    return rows


def main():
    p=argparse.ArgumentParser()
    for key in ('tracker','original','fixed','mixed','reference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('start','stop'):
        p.add_argument('--'+key,type=int,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    sha=file_sha256(a.tracker)
    assert sha==a.tracker.parent.name
    state=torch.load(a.tracker,map_location='cpu',weights_only=True)
    config=state['config']
    assert config['model']['architecture']=='transformer'
    for file in ('graph/temporal/fixed_id_corner.py','graph/temporal/causal_transformer.py',
                 'data_collection/temporal/fixed_id_corner_sequences.py'):
        assert file_sha256(file)==config['implementation']['files_sha256'][file]
    model=FixedIdCornerPredictor(**config['model']).to(device).eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True)
    gnns={}
    hashes={}
    for key in ('original','fixed','mixed'):
        path=getattr(a,key)
        hashes[key]=file_sha256(path)
        if key=='original':
            assert hashes[key]==path.parent.name
        weights=torch.load(path,map_location='cpu',weights_only=True)
        gnn=VertexEdgeGatedMPNN(**weights['config']['gnn']).to(device).eval().requires_grad_(False)
        gnn.load_state_dict(weights['model'],strict=True)
        gnns[key]=gnn
    seeds=list(range(a.start,a.stop))
    with np.load(a.reference,allow_pickle=False) as d:
        refs={seed:[{k:d[k][i] for k in ('phase','grasp','target','offset_m','moving_now')}
                    for i in np.flatnonzero(d['seed']==seed)] for seed in seeds}
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=dict(seeds=seeds,camera=CAMERA,tracker_sha256=sha,gnn_sha256=hashes,
             reference_sha256=file_sha256(a.reference),
             detector='RANSAC compiled CUDA, NumPy RGB-D',association_m=.030,
             evaluation_identity_gate_m=.005,inlier_m=.008,min_support=3,max_speed_m_per_frame=.012,
             max_prediction_frames=5,script_sha256=file_sha256(__file__),
             rigid_sha256=file_sha256(Path(__file__).with_name('rigid_box.py')))
    start=monotonic()
    with ExperimentLogger(team='graph',model='rigid_box_object21_eval',job_type='evaluate',
                          config=cfg,seed=0,root=a.output/'experiments') as logger:
        env=make_peg_insertion_env(device='cpu',render_backend='cuda',obs_mode='rgb+depth+segmentation',
                                  control_mode='pd_ee_delta_pose',max_episode_steps=200)
        detector=TorchCornerDetector.main(device=device,plane_fit_method='ransac')
        logger.record(runtime=dict(backend=str(env.unwrapped.backend),gpu=torch.cuda.get_device_name()))
        counts={}
        try:
            path=a.output/'frames.jsonl'
            with path.open('w') as stream:
                for i,seed in enumerate(seeds,1):
                    rows=episode(env,seed,detector,model,gnns,refs[seed],device)
                    counts[str(seed)]=len(rows)
                    for row in rows:
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                    stream.flush()
                    print(f'{i}/{len(seeds)} seed {seed}, {len(rows)} frames, {monotonic()-start:.1f}s',flush=True)
            out=a.output/'summary.json'
            write(out,dict(run_id=logger.run_id,config=cfg,counts=counts,seconds=monotonic()-start))
            logger.log({'frames':sum(counts.values())})
            logger.link_output(out)
            logger.link_output(path)
            print('Completed',logger.run_id,flush=True)
        finally:
            env.close()


if __name__=='__main__':
    main()
