"""Collect paired oracle and real track correspondences on the held-out moving Box replay."""
import argparse
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
from evaluate_points import identity

from oracle_reference import CAMERA, target
from rigid_box import RigidBoxTranslation

def observe(env, obs, gt, detector, tracker, rigid, certified):
    images = {key: value[0].cpu().numpy() for key, value in obs['sensor_data']['base_camera'].items()}
    camera = {key: value[0].cpu().numpy() for key, value in obs['sensor_param']['base_camera'].items()}
    verify_camera(images, camera, CAMERA)
    raw = detector(images['rgb'], images['depth'], camera['intrinsic_cv'], camera['extrinsic_cv'])
    xyz, kinds = raw.xyz.cpu().numpy(), raw.object_id.cpu().numpy()
    tcp = obs['extra']['tcp_pose'][0, :3].cpu().numpy()
    robot = np.r_[obs['extra']['tcp_pose'][0].cpu().numpy(),
                  obs['agent']['qpos'][0, -2:].cpu().numpy()].astype(np.float32)

    gt_corners = corners(gt)
    ids = [identity(corner) for corner in gt_corners]
    assert ids == list(range(24)), ids
    gt_xyz = np.asarray([corner['xyz'] for corner in gt_corners[:20]], dtype=np.float32)
    gt_kinds = np.asarray([corner['object_id'] for corner in gt_corners[:20]], dtype=np.int64)
    task = env.unwrapped
    actor_ids = {0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])}
    visible = visible_reference(gt, {**images, **camera}, actor_ids)
    visible_xyz = np.asarray([corner['xyz'] for corner in visible]).reshape(-1, 3)
    visible_kinds = np.asarray([corner['object_id'] for corner in visible], dtype=np.int64)
    visible_ids = [identity(corner) for corner in visible]

    tracks = tracker.step(xyz, kinds, robot)
    labels = np.asarray([t.kind for t in tracker.tracks], dtype=np.int64)
    observed = tracks['observed']
    exterior = {int(tid) for tid, kind, meta in zip(tracks['detection_tracks'], kinds, raw.metadata, strict=True)
                if kind == 1 and not meta.get('contour_hole', False) and meta.get('method') != 'box_opening'}
    corrected, fit = rigid.update(tracks['positions'], observed, labels, exterior)
    np.testing.assert_array_equal(corrected[observed | (labels == 0)],
                                 tracks['positions'][observed | (labels == 0)])

    visible_mask = np.isin(ids[:20], visible_ids)
    detected_mask = np.zeros(20, dtype=bool)
    tracked_mask = np.zeros(20, dtype=bool)
    clean_association = gt_xyz.copy()
    raw_association = gt_xyz.copy()
    for i, j in match_positions(xyz, kinds, visible_xyz, visible_kinds, .005):
        track_id = int(tracks['detection_tracks'][i])
        gt_id = visible_ids[j]
        certified.pop(track_id, None)
        certified[track_id] = gt_id
        if gt_id < 20:
            detected_mask[gt_id] = tracked_mask[gt_id] = True
            clean_association[gt_id] = raw_association[gt_id] = xyz[i]

    # Use a historical identity only for an unobserved track. If multiple track
    # IDs claim the same vertex, keep the one with the most recent GT match.
    for track_id, gt_id in reversed(tuple(certified.items())):
        if not observed[track_id] and gt_id < 20 and not tracked_mask[gt_id]:
            assert labels[track_id] == gt_kinds[gt_id]
            tracked_mask[gt_id] = True
            clean_association[gt_id] = corrected[track_id]
            raw_association[gt_id] = tracks['positions'][track_id]

    return dict(xyz=gt_xyz, kinds=gt_kinds, tcp=tcp, visible=visible_mask, detected=detected_mask,
                tracked=tracked_mask, corrected_xyz=clean_association, tracked_xyz=raw_association,
                fit_m=fit['pose_m'], total_tracks=len(labels), observed_tracks=int(observed.sum()))


@torch.inference_mode()
def episode(env,seed,detector,model,reference,device):
    obs,_=env.reset(seed=seed)
    policy=make_policy()
    policy.reset(env,seed)
    task=env.unwrapped
    motion=None
    tracker=DetectedCornerTracker(model,.030,device)
    rigid=RigidBoxTranslation(use_velocity=True)
    rows, certified = [], {}
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
        inputs = observe(env, obs, gt, detector, tracker, rigid, certified)
        rows.append(dict(seed=seed, step=step, **inputs))
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
    cfg=dict(seeds=seeds,camera=CAMERA,tracker_sha256=sha,
             reference_sha256=file_sha256(a.reference),
             detector='RANSAC compiled CUDA, NumPy RGB-D',association_m=.030,
             evaluation_identity_gate_m=.005,inlier_m=.008,min_support=3,max_speed_m_per_frame=.012,
             max_prediction_frames=5,script_sha256=file_sha256(__file__),
             rigid_sha256=file_sha256(Path(__file__).with_name('rigid_box.py')))
    start=monotonic()
    with ExperimentLogger(team='graph',model='gt_corner_input_ablation_collection',job_type='evaluate',
                          config=cfg,seed=0,root=a.output/'experiments') as logger:
        env=make_peg_insertion_env(device='cpu',render_backend='cuda',obs_mode='rgb+depth+segmentation',
                                  control_mode='pd_ee_delta_pose',max_episode_steps=200)
        detector=TorchCornerDetector.main(device=device,plane_fit_method='ransac')
        logger.record(runtime=dict(backend=str(env.unwrapped.backend),gpu=torch.cuda.get_device_name()))
        counts={}
        try:
            path = a.output / 'matches.npz'
            all_rows = []
            for i, seed in enumerate(seeds, 1):
                rows = episode(env, seed, detector, model, refs[seed], device)
                counts[str(seed)] = len(rows)
                all_rows.extend(rows)
                print(f'{i}/{len(seeds)} seed {seed}, {len(rows)} frames, {monotonic()-start:.1f}s', flush=True)
            np.savez_compressed(path, **{key: np.asarray([row[key] for row in all_rows])
                                        for key in all_rows[0]})
            out=a.output/'summary.json'
            write(out,dict(run_id=logger.run_id,config=cfg,counts=counts,matches_sha256=file_sha256(path),seconds=monotonic()-start))
            logger.log({'frames':sum(counts.values())})
            logger.link_output(out)
            logger.link_output(path)
            print('Completed',logger.run_id,flush=True)
        finally:
            env.close()


if __name__=='__main__':
    main()
