"""Installed-controller pose feedback and trace-based Reach diagnostics."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from graph.subtasks.metrics import rotation


def controller_evidence(env):
    import inspect
    c=env.unwrapped.agent.controller.controllers['arm']
    return dict(config=repr(c.config),clip_and_scale_source=inspect.getsource(type(c)._clip_and_scale_action),
                target_pose_source=inspect.getsource(type(c).compute_target_pose),
                gripper_config=repr(env.unwrapped.agent.controller.controllers['gripper'].config),
                action_low=env.action_space.low.tolist(),action_high=env.action_space.high.tolist())


def reference_action(env):
    arm=env.unwrapped.agent.controller.controllers['arm']
    cfg=arm.config
    if cfg.frame!='root_translation:root_aligned_body_rotation' or not cfg.use_delta or cfg.use_target:
        raise ValueError('unverified controller frame/target convention')
    state=env.live_state
    target=np.array(env.metrics['reach_goal_pose'])
    root=env.unwrapped.agent.robot.pose.raw_pose[0].detach().cpu().numpy()
    rr=rotation(root)
    translation=rr.T@(target[:3]-state['ee_pose'][:3])
    relative=rr.T@rotation(target)@rotation(state['ee_pose']).T@rr
    # Intrinsic XYZ matches installed euler_angles_to_matrix(delta_rot, "XYZ").
    euler=Rotation.from_matrix(relative).as_euler('XYZ')
    pos=.5*translation/np.asarray(cfg.pos_upper)
    rot=.5*euler/float(cfg.rot_lower)
    rot=rot/max(1.,np.linalg.norm(rot))
    return torch.tensor(np.r_[np.clip(pos,-1,1),rot,1.],dtype=torch.float32)[None]


def snapshot(env):
    s=env.live_state;m=env.metrics
    return dict(metrics=dict(m),ee_pose=s['ee_pose'].tolist(),peg_pose=s['peg_pose'].tolist(),
                qpos=s['qpos'].tolist(),qvel=s['qvel'].tolist(),gripper_qpos=s['gripper_qpos'].tolist())


def trace_summary(initial, trace, success):
    states=[initial]+[r['after'] for r in trace]
    pos=[s['metrics']['position_error'] for s in states]
    rot=[s['metrics']['rotation_error'] for s in states]
    if success:
        case='success'
    elif min(pos)>.5*pos[0]:
        case='A_little_approach'
    elif min(pos)<=.04:
        case='B_near_goal' if pos[-1]<=.04 else 'C_approach_then_diverge'
    else:
        case='C_approach_then_diverge' if pos[-1]>min(pos)+.04 else 'approach_incomplete'
    actions=np.array([r['applied_action'] for r in trace])
    return dict(case=case,minimum_position_error_m=min(pos),final_position_error_m=pos[-1],
                minimum_rotation_error_rad=min(rot),final_rotation_error_rad=rot[-1],
                rotation_error_at_minimum_distance_rad=rot[int(np.argmin(pos))],
                max_peg_displacement_m=max(s['metrics']['peg_motion'] for s in states),
                gripper_closed_steps=sum(not s['metrics']['gripper_open'] for s in states),
                clipping_steps=sum(r['clipped'] for r in trace),
                rotation_norm_saturation_steps=sum(r['rotation_norm_saturated'] for r in trace),
                final_translation_action_norm=float(np.linalg.norm(actions[-1,:3])) if len(actions) else None,
                translation_action_sign_reversals=int(((actions[1:,:3]*actions[:-1,:3])<0).sum()) if len(actions)>1 else 0,
                workspace_exit_steps=sum(not s['metrics']['workspace_valid'] for s in states))


def distribution_audit(output, dataset):
    """Replay measured actions only; no learning, threshold or policy changes."""
    import json
    from pathlib import Path
    from graph.prototype.common import seed_all, write_json
    from graph.prototype.data import load_dataset, split_indices
    from graph.subtasks.env import make_subtask_env
    from .policy import LivePolicy, inputs
    from .evaluate import reset, fingerprint
    seed_all(0,4);output=Path(output)
    path=output/'reports/distribution_audit.json'
    if path.exists():raise FileExistsError(path)
    data,g=load_dataset(dataset)
    policy=LivePolicy(output/'checkpoints/best.pt',dataset)
    train_ids=split_indices(data,'train');test_ids=split_indices(data,'test')
    tx,ts=inputs(g.select(train_ids),data['qvel'][train_ids],policy.norm,policy.sn)
    vx,vs=inputs(g.select(test_ids),data['qvel'][test_ids],policy.norm,policy.sn)
    train=torch.cat((tx,ts),-1)
    expert_dist=torch.cdist(torch.cat((vx,vs),-1),train).min(-1).values/(train.shape[1]**.5)
    def stats(a):
        a=np.asarray(a)
        return dict(median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max()))
    main=json.loads((output/'evaluation/flat.json').read_text())
    chosen=[11033,11029,12001,12000,12017,12005]
    manifest=json.loads((output/'evaluation/paired_starts.json').read_text())
    env=make_subtask_env('reach',index=manifest['index'])
    records=[]
    try:
        with torch.inference_mode():
            for seed in chosen:
                original=next(r for r in main['records'] if r['start']['eval_seed']==seed)
                reset(env,original['start'])
                if fingerprint(env)!=original['initial_state_fingerprint']:raise ValueError('initial state differs')
                samples=[];max_error=0.
                for step,row in enumerate(original['trace']):
                    features=policy.features(env)
                    predicted=policy.model(*features)
                    max_error=max(max_error,float((predicted-torch.tensor(row['raw_action'])[None]).abs().max()))
                    if step%8==0 or step==len(original['trace'])-1:
                        all_features=torch.cat(features,-1)
                        distance,near=torch.cdist(all_features,train).min(-1)
                        samples.append(dict(t=step,position_error_m=env.metrics['position_error'],rotation_error_rad=env.metrics['rotation_error'],
                                            nearest_train_feature_rms=float(distance[0]/(train.shape[1]**.5)),
                                            nearest_train_episode=int(data['episode_id'][train_ids[near[0]]]),
                                            shared_qvel_z_rms=float(features[1][:,:9].square().mean().sqrt()),
                                            shared_goal_z_rms=float(features[1][:,9:].square().mean().sqrt()),
                                            geometry_z_max=float(features[0].abs().max())))
                    env.step(torch.tensor(row['applied_action']))
                    max_error=max(max_error,float(np.max(np.abs(env.live_state['qpos']-np.array(row['after']['qpos'])))))
                if max_error>2e-5:raise ValueError('stored-action replay or feature/prediction mismatch')
                records.append(dict(seed=seed,case=original['diagnostics']['case'],success=original['success'],samples=samples,
                                    replay_prediction_and_qpos_max_error=max_error,
                                    nearest_train_feature_rms=stats([s['nearest_train_feature_rms'] for s in samples])))
                print('distribution',seed,records[-1]['nearest_train_feature_rms'],flush=True)
    finally:env.close()
    report=dict(heldout_expert_nearest_train_feature_rms=stats(expert_dist.numpy()),records=records,
                selected_seeds=chosen,selection='post-hoc representatives of failures and success; no tuning',
                metric='Euclidean distance / sqrt(1946), train-normalized feature concatenation',
                caution='descriptive covariate shift evidence, not proof of a unique causal bottleneck')
    write_json(path,report)
    return report


if __name__=='__main__':
    import argparse
    from .policy import ROOT,DATASET
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=ROOT);p.add_argument('--dataset',default=DATASET)
    a=p.parse_args();distribution_audit(a.output,a.dataset)
