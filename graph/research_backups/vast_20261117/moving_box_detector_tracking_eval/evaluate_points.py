"""Measure RGB-D 3D detections and Transformer tracks on moving Box GT."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from collect_moving import schedule
from data_collection.episode_dataset import file_sha256
from data_collection.hole_motion import EpisodeHoleMotion
from data_collection.scripted_policy import make_policy
from envs.maniskill import get_gt_state, make_peg_insertion_env
from evaluation.initial_corner_detection import verify_camera
from graph.temporal.detected_corner_tracking import DetectedCornerTracker, match_positions
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from utils.experiment_logger import ExperimentLogger, write
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import corners, visible_reference

from oracle_reference import CAMERA, target

FAMILIES=('peg','box_outer','hole_front','hole_rear')


def identity(r):
    return {'peg':0,'box_outer':8,'hole_rim':16}[r['kind']]+r['corner_id']


def family(i):
    return FAMILIES[0 if i<8 else 1 if i<16 else 2 if i<20 else 3]


def visible_scores(xyz,kinds,v_xyz,v_kinds,v_ids):
    result={}
    for gate in (3,5,10):
        pairs=match_positions(xyz,kinds,v_xyz,v_kinds,gate*.001)
        by_group={g:[] for g in FAMILIES}
        for i,j in pairs:
            by_group[family(v_ids[j])].append(float(np.linalg.norm(xyz[i]-v_xyz[j])*1000))
        result[str(gate)]=by_group
    return result


def score(obs,env,detector,tracker,certified,gt):
    img={k:v[0].cpu().numpy() for k,v in obs['sensor_data']['base_camera'].items()}
    camera={k:v[0].cpu().numpy() for k,v in obs['sensor_param']['base_camera'].items()}
    verify_camera(img,camera,CAMERA)
    raw=detector(img['rgb'],img['depth'],camera['intrinsic_cv'],camera['extrinsic_cv'])
    xyz,kinds=raw.xyz.cpu().numpy(),raw.object_id.cpu().numpy()
    robot=np.r_[obs['extra']['tcp_pose'][0].cpu().numpy(),obs['agent']['qpos'][0,-2:].cpu().numpy()].astype(np.float32)
    tracks=tracker.step(xyz,kinds,robot)
    track_types=np.array([t.kind for t in tracker.tracks],np.int64)
    assert np.array_equal(tracks['positions'][tracks['detection_tracks']],xyz)
    task=env.unwrapped
    actor_ids={0:int(task.peg.per_scene_id[0]),1:int(task.box.per_scene_id[0])}
    refs=visible_reference(gt,{**img,**camera},actor_ids)
    v_xyz=np.array([r['xyz'] for r in refs]).reshape(-1,3)
    v_kinds=np.array([r['object_id'] for r in refs],np.int64)
    v_ids=[identity(r) for r in refs]
    truth={identity(r):r['xyz'] for r in corners(gt)}
    conflicts={g:0 for g in FAMILIES}
    ids_seen=set(certified)
    for i,j in match_positions(xyz,kinds,v_xyz,v_kinds,.005):
        track_id=int(tracks['detection_tracks'][i])
        vid=v_ids[j]
        if track_id in certified and certified[track_id]!=vid:
            conflicts[family(certified[track_id])]+=1
        certified.setdefault(track_id,vid)
    predicted_only={g:[] for g in FAMILIES}
    stale_only={g:[] for g in FAMILIES}
    known_observed={g:[] for g in FAMILIES}
    visible_set=set(v_ids)
    hidden={g:[] for g in FAMILIES}
    unverified=int((~tracks['observed']).sum())
    for tid,vid in certified.items():
        label=family(vid)
        distance=float(np.linalg.norm(tracks['positions'][tid]-truth[vid])*1000)
        if tracks['observed'][tid]:
            known_observed[label].append(distance)
        elif tid in ids_seen:
            unverified-=1
            predicted_only[label].append(distance)
            stale_only[label].append(float(np.linalg.norm(tracker.tracks[tid].last_observed-truth[vid])*1000))
            if vid not in visible_set:
                hidden[label].append(distance)
    return dict(visible={g:sum(family(i)==g for i in v_ids) for g in FAMILIES},
                detector=visible_scores(xyz,kinds,v_xyz,v_kinds,v_ids),
                tracker=visible_scores(tracks['positions'],track_types,v_xyz,v_kinds,v_ids),
                predicted_only=predicted_only,previous_detection=stale_only,hidden_prediction=hidden,
                observed_known=known_observed,id_conflicts=conflicts,
                counts=dict(detections=len(xyz),tracks=len(track_types),observed=int(tracks['observed'].sum()),
                            unmatched_xyz=int((~tracks['observed']).sum()),unverified_predictions=unverified,
                            new_tracks=tracks['new_tracks']))


def episode(env,seed,detector,model,reference,device):
    obs,_=env.reset(seed=seed)
    policy=make_policy()
    policy.reset(env,seed)
    task=env.unwrapped
    motion=None
    tracker=DetectedCornerTracker(model,.030,device)
    certified={}
    rows=[]
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
        metrics=score(obs,env,detector,tracker,certified,gt)
        rows.append(dict(seed=seed,step=step,phase=phase,moving=bool(moving),
                         window='during' if moving else 'after' if elapsed_motion else 'before',
                         offset_m=offset.tolist(),**metrics))
        obs,_,term,trunc,_=env.step(policy.act(obs))
        if bool(term[0]) or bool(trunc[0]):
            terminal='terminated' if bool(term[0]) else 'truncated'
            break
    assert len(rows)==len(reference) and terminal=='insert_entry',(seed,len(rows),len(reference),terminal)
    return rows


def main():
    p=argparse.ArgumentParser()
    for key in ('tracker','reference','output'):
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
    seeds=list(range(a.start,a.stop))
    with np.load(a.reference,allow_pickle=False) as d:
        refs={seed:[{k:d[k][i] for k in ('phase','grasp','target','offset_m','moving_now')}
                    for i in np.flatnonzero(d['seed']==seed)] for seed in seeds}
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=dict(seeds=seeds,camera=CAMERA,tracker_sha256=sha,reference_sha256=file_sha256(a.reference),
             detector='RANSAC compiled CUDA, NumPy RGB-D',association_m=.030,
             evaluation_identity_gate_m=.005,script_sha256=file_sha256(__file__))
    start=monotonic()
    with ExperimentLogger(team='graph',model='moving_box_detector_tracking_eval',job_type='evaluate',
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
                    rows=episode(env,seed,detector,model,refs[seed],device)
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
