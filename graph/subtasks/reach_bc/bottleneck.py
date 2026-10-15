"""Frozen-checkpoint causal probes; never retrain or replace baseline results."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from graph.prototype.common import seed_all, write_json, versions
from graph.prototype.data import load_dataset, split_indices, sha256
from graph.prototype.graph_builder import context_features
from graph.prototype.geometry import rotation as torch_rotation
from graph.prototype.v2.state import shared_features
from graph.subtasks.env import make_subtask_env
from graph.subtasks.metrics import rotation
from .policy import ROOT, DATASET, LivePolicy, inputs, decode
from .evaluate import reset, fingerprint, summary
from .diagnostics import snapshot, trace_summary, reference_action

OUTPUT=ROOT/'bottleneck_v1'
CONTEXT_START=1834
FINGER_CONTEXT=(0,1,9,10)
EE_ROTATION_CONTEXT=tuple(range(14,23))
MODES=('finger_mean','qvel_zero','ee_floor_clip','low_variance_clip')


def context_labels():
    labels=[]
    for name,n in [('gripper',2),('qpos',9),('ee_position',3),('ee_rotation',9),
                   ('root_position',3),('root_rotation',9),('peg_relative_position',3),('peg_rotation',9),
                   ('peg_half_size',3),('box_relative_position',3),('box_rotation',9),('box_half_size',3),
                   ('hole_relative_position',3),('hole_rotation',9),('hole_radius',1),('hole_offset',3)]:
        labels.extend(f'{name}[{i}]' for i in range(n))
    assert len(labels)==81
    return labels


class Probes:
    def __init__(self,dataset=DATASET):
        self.policy=LivePolicy(ROOT/'checkpoints/best.pt',dataset)
        self.policy.model.requires_grad_(False)
        self.data,self.geometry=load_dataset(dataset)
        ids=split_indices(self.data,'train')
        self.train_x,self.train_s=inputs(self.geometry.select(ids),self.data['qvel'][ids],self.policy.norm,self.policy.sn)
        self.x_min,self.x_max=self.train_x.amin(0),self.train_x.amax(0)
        self.s_min,self.s_max=self.train_s.amin(0),self.train_s.amax(0)
        self.x_lowvar=torch.zeros(self.train_x.shape[1],dtype=torch.bool)
        self.x_lowvar[CONTEXT_START:]=self.policy.norm.context_std<=.00100001
        self.s_lowvar=self.policy.sn.std<=.00100001
        # The existing node normalizer uses pooled-node stats; none hits floor.
        if bool((self.policy.norm.node_std<=.00100001).any()):
            raise ValueError('unexpected floored node channels; audit their ordering')

    def modify(self,features,mode):
        x,s=(f.clone() for f in features)
        if mode=='finger_mean':
            x[:,[CONTEXT_START+i for i in FINGER_CONTEXT]]=0.
            s[:,7:9]=0.
        elif mode=='qvel_zero':
            # Physical qvel=0, not normalized zero/train mean.
            s[:,:9]=-self.policy.sn.mean[:9]/self.policy.sn.std[:9]
        elif mode=='ee_floor_clip':
            i=CONTEXT_START+22
            x[:,i]=x[:,i].clamp(self.x_min[i],self.x_max[i])
        elif mode=='low_variance_clip':
            x[:,self.x_lowvar]=torch.maximum(self.x_min[self.x_lowvar],torch.minimum(x[:,self.x_lowvar],self.x_max[self.x_lowvar]))
            s[:,self.s_lowvar]=torch.maximum(self.s_min[self.s_lowvar],torch.minimum(s[:,self.s_lowvar],self.s_max[self.s_lowvar]))
        else:raise ValueError(mode)
        return x,s

    def all_predictions(self,features):
        batch=[self.modify(features,m) for m in MODES]
        with torch.inference_mode():
            original=self.policy.model(*features)[0]
            pred=self.policy.model(torch.cat([f[0] for f in batch]),torch.cat([f[1] for f in batch]))
        return dict(original=original,**{m:pred[i] for i,m in enumerate(MODES)})

    def jacobian(self,features):
        x,s=(f.detach().clone().requires_grad_() for f in features)
        action=self.policy.model(x,s)
        dx=[];ds=[]
        for i in range(6):
            gx,gs=torch.autograd.grad(action[0,i],(x,s),retain_graph=i<5)
            dx.append(gx[0]);ds.append(gs[0])
        jx,js=torch.stack(dx),torch.stack(ds)
        return dict(finger_qvel_motion_derivative_per_mps=(js[:,7:9]/self.policy.sn.std[7:9]).tolist(),
                    arm_qvel_motion_derivative_per_radps=(js[:,:7]/self.policy.sn.std[:7]).tolist(),
                    ee_rzz_motion_derivative_per_matrix_value=(jx[:,CONTEXT_START+22]/self.policy.norm.context_std[22]).tolist(),
                    finger_context_motion_derivative_per_m=(jx[:,[CONTEXT_START+i for i in FINGER_CONTEXT]]/self.policy.norm.context_std[list(FINGER_CONTEXT)]).tolist())


def config_report(probes,output):
    output=Path(output);path=output/'config.json'
    if path.exists():return
    write_json(path,dict(checkpoint=str(ROOT/'checkpoints/best.pt'),checkpoint_sha256=sha256(ROOT/'checkpoints/best.pt'),
                        dataset_sha256=sha256(DATASET),baseline_manifest_sha256=sha256(ROOT/'evaluation/paired_starts.json'),
                        frozen_model=True,no_retraining=True,train_only_bounds=True,modes=list(MODES),
                        descriptions=dict(finger_mean='replace finger qpos/gripper/context duplicates and qvel inputs with train means; learned gripper command retained',
                                          qvel_zero='replace all nine qvel inputs with physical zero; actual qpos/controller unchanged',
                                          ee_floor_clip='bound only context EE rotation R[2,2] to its train feature min/max',
                                          low_variance_clip='bound context/state channels whose existing std hits 0.001 floor to train min/max'),
                        x_low_variance_context_indices=torch.where(probes.x_lowvar[CONTEXT_START:])[0].tolist(),
                        state_low_variance_indices=torch.where(probes.s_lowvar)[0].tolist(),
                        caveat='diagnostic inference interventions, potentially inconsistent feature combinations; not learned-policy main results',
                        no_hyperparameter_search=True,success_predicate_unchanged=True,versions=versions()))


def audit(output=OUTPUT):
    seed_all(0,4);output=Path(output)
    path=output/'audit.json'
    if path.exists():raise FileExistsError(path)
    p=Probes();config_report(p,output)
    main=json.loads((ROOT/'evaluation/flat.json').read_text())
    manifest=json.loads((ROOT/'evaluation/paired_starts.json').read_text())
    ctx=context_features(p.geometry.select(split_indices(p.data,'train')))
    shared=shared_features(p.geometry.select(split_indices(p.data,'train')),p.data['qvel'][split_indices(p.data,'train')],'goal')
    labels=context_labels()
    train_feature_report=dict(context=[dict(index=i,label=labels[i],mean=float(ctx[:,i].mean()),raw_std=float(ctx[:,i].std(correction=0)),normalization_std=float(p.policy.norm.context_std[i])) for i in range(81)],
                              shared_mean=p.policy.sn.mean.tolist(),shared_std=p.policy.sn.std.tolist(),
                              shared_raw_std=shared.std(0,correction=0).tolist(),
                              train_action_min=p.data['actions'][split_indices(p.data,'train')].amin(0).tolist(),
                              train_action_max=p.data['actions'][split_indices(p.data,'train')].amax(0).tolist())
    env=make_subtask_env('reach',index=manifest['index'])
    records=[]
    try:
        for original in main['records']:
            if not original['reset_valid']:continue
            start=original['start'];reset(env,start)
            if fingerprint(env)!=original['initial_state_fingerprint']:raise ValueError('paired initial state mismatch')
            rows=[];max_error=0.
            distances=[original['initial']['metrics']['position_error']]+[r['after']['metrics']['position_error'] for r in original['trace']]
            min_state=int(np.argmin(distances))
            for t,saved in enumerate(original['trace']):
                features=p.policy.features(env)
                predictions=p.all_predictions(features)
                max_error=max(max_error,float((predictions['original']-torch.tensor(saved['raw_action'])).abs().max()))
                zc=features[0][0,CONTEXT_START:];zs=features[1][0]
                top=torch.topk(zc.abs(),5).indices.tolist()
                delta=np.array(saved['applied_action'][:3])*.1
                error=np.array(env.metrics['reach_goal_pose'][:3])-env.live_state['ee_pose'][:3]
                root=env.unwrapped.agent.robot.pose.raw_pose[0].numpy()
                alignment=float(np.dot(rotation(root)@delta,error)/(np.linalg.norm(delta)*np.linalg.norm(error))) if np.linalg.norm(delta)>1e-10 else None
                row=dict(t=t,position_error_m=env.metrics['position_error'],rotation_error_rad=env.metrics['rotation_error'],
                         gripper_qpos=env.live_state['gripper_qpos'].tolist(),finger_qvel=env.live_state['qvel'][-2:].tolist(),
                         finger_qvel_z=zs[7:9].tolist(),ee_rzz_z=float(zc[22]),
                         finger_context_z=zc[list(FINGER_CONTEXT)].tolist(),
                         arm_qvel_z_rms=float(zs[:7].square().mean().sqrt()),
                         context_z_max=float(zc.abs().max()),state_z_max=float(zs.abs().max()),
                         top_context=[dict(index=i,label=labels[i],z=float(zc[i])) for i in top],
                         command_target_alignment=alignment,
                         actions={m:a.tolist() for m,a in predictions.items()},
                         counterfactual_motion_delta={m:float((a[:6]-predictions['original'][:6]).norm()) for m,a in predictions.items() if m!='original'})
                if t in set((0,1,2,min_state,max(0,min_state-1))) and start['eval_seed'] in [11015,11029,11031,11033,11041,12000,12001,12005,12017]:
                    row['jacobian']=p.jacobian(features)
                env.step(torch.tensor(saved['applied_action']))
                for k in ('qpos','qvel','ee_pose','peg_pose'):
                    max_error=max(max_error,float(np.abs(env.live_state[k]-np.array(saved['after'][k])).max()))
                rows.append(row)
            if max_error>2e-5:raise ValueError('saved replay or prediction drift')
            records.append(dict(seed=start['eval_seed'],group=start['group'],success=original['success'],steps=original['steps'],
                                failure_reason=original['failure_reason'],minimum_distance_state=min_state,
                                replay_and_prediction_max_error=max_error,rows=rows))
            write_json(path,dict(partial=len(records)<55,train_features=train_feature_report,records=records))
            print('audit',start['eval_seed'],original['success'],max_error,flush=True)
    finally:env.close()
    return records


def rollout(mode,output=OUTPUT):
    if mode not in MODES:raise ValueError(mode)
    seed_all(0,4);output=Path(output);path=output/f'{mode}.json'
    if path.exists():raise FileExistsError(path)
    p=Probes();config_report(p,output)
    manifest=json.loads((ROOT/'evaluation/paired_starts.json').read_text())
    env=make_subtask_env('reach',index=manifest['index']);records=[]
    try:
        with torch.inference_mode():
            for start in manifest['starts']:
                reset(env,start)
                if fingerprint(env)!=start['initial_state_fingerprint']:raise ValueError('paired initial state mismatch')
                initial=snapshot(env);trace=[]
                while True:
                    features=p.policy.features(env);modified=p.modify(features,mode)
                    raw=p.policy.model(*modified);applied=decode(raw)
                    before=snapshot(env)
                    _,_,terminated,truncated,info=env.step(applied)
                    trace.append(dict(t=len(trace),before=before,after=snapshot(env),raw_action=raw[0].tolist(),applied_action=applied[0].tolist(),
                                      clipped=bool((raw.abs()>1).any()),rotation_norm_saturated=bool(applied[:,3:6].norm()>1),
                                      success=info['subtask_success'],failure_reason=info['failure_reason'],terminated=terminated,truncated=truncated,
                                      input_changed_dimensions=int(torch.count_nonzero(features[0]!=modified[0])+torch.count_nonzero(features[1]!=modified[1]))))
                    if terminated or truncated:break
                r=dict(start=start,reset_valid=True,success=info['subtask_success'],steps=len(trace),failure_reason=info['failure_reason'],
                       initial_state_fingerprint=start['initial_state_fingerprint'],initial=initial,trace=trace,
                       diagnostics=trace_summary(initial,trace,info['subtask_success']))
                records.append(r)
                write_json(output/f'{mode}_progress.json',dict(partial=len(records)<55,completed=len(records),successes=sum(r['success'] for r in records)))
                print(mode,start['group'],start['eval_seed'],r['success'],len(trace),r['failure_reason'],flush=True)
    finally:env.close()
    result=dict(mode=mode,diagnostic_only=True,checkpoint_sha256=sha256(ROOT/'checkpoints/best.pt'),
                records=records,held_out=summary([r for r in records if r['start']['group']=='held_out']),fresh=summary([r for r in records if r['start']['group']=='fresh']),
                versions=versions(),backend='physx_cpu')
    write_json(path,result);print('SUMMARY',mode,result['held_out'],result['fresh'],flush=True)
    return result


def feedback(output=OUTPUT):
    """Finite differences through a real simulator step, with repeat resets.

    This is a local action-to-next-action amplification probe, not a proof of
    asymptotic stability of the full robot/controller state.
    """
    seed_all(0,4);output=Path(output);path=output/'feedback.json'
    if path.exists():raise FileExistsError(path)
    p=Probes();config_report(p,output)
    manifest=json.loads((ROOT/'evaluation/paired_starts.json').read_text())
    env=make_subtask_env('reach',index=manifest['index'])
    seeds=[11000,11001,11005,11029,11033,12000,12005]
    records=[];epsilon=.01
    try:
        for seed in seeds:
            start=next((s for s in manifest['starts'] if s['eval_seed']==seed),None)
            if start is None:
                ep=next(e for e in p.data['episodes'] if e['seed']==seed)
                start=dict(eval_seed=seed,reset_kwargs=dict(seed=seed,options={'source_episode_id':ep['episode_id']}))
            reset(env,start);fp=fingerprint(env)
            with torch.inference_mode():raw=p.policy(env);base=decode(raw)
            if bool((base[:,:6].abs()>.98).any()):raise ValueError('finite difference hits action clipping boundary')
            matrices={m:[] for m in ('original',)+MODES+('reference',)}
            responses=[]
            for axis in range(6):
                plusminus=[]
                for sign in (1,-1):
                    reset(env,start)
                    if fingerprint(env)!=fp:raise ValueError('perturbation reset differs')
                    action=base.clone();action[0,axis]+=sign*epsilon
                    env.step(action)
                    predictions=p.all_predictions(p.policy.features(env))
                    predictions['reference']=reference_action(env)[0]
                    plusminus.append(predictions)
                for mode in matrices:
                    matrices[mode].append(((plusminus[0][mode][:6]-plusminus[1][mode][:6])/(2*epsilon)).numpy())
                responses.append(dict(axis=axis,plus={m:a.tolist() for m,a in plusminus[0].items()},minus={m:a.tolist() for m,a in plusminus[1].items()}))
            values={}
            for mode,columns in matrices.items():
                matrix=np.stack(columns,axis=1)
                values[mode]=dict(matrix=matrix.tolist(),maximum_singular_gain=float(np.linalg.svd(matrix,compute_uv=False)[0]),
                                  action_feedback_spectral_radius=float(np.abs(np.linalg.eigvals(matrix)).max()))
            records.append(dict(seed=seed,initial_state_fingerprint=fp,base_action=base[0].tolist(),results=values,responses=responses))
            print('feedback',seed,{m:round(v['maximum_singular_gain'],3) for m,v in values.items()},flush=True)
    finally:env.close()
    write_json(path,dict(epsilon_normalized_action=epsilon,translation_perturbation_m=.001,rotation_component_perturbation_rad=.001,
                         records=records,backend='physx_cpu',versions=versions(),
                         caveat='one-step local action-to-next-action sensitivity; not full-state closed-loop eigenanalysis'))
    return records


def reference_switch(output=OUTPUT):
    """Switch measured learner prefixes to reference commands for diagnosis.

    No training data or recovery demonstration is exported. The original
    success/failure context and 130-step budget remain intact.
    """
    seed_all(0,4);output=Path(output);path=output/'reference_switch.json'
    if path.exists():raise FileExistsError(path)
    main=json.loads((ROOT/'evaluation/flat.json').read_text())
    manifest=json.loads((ROOT/'evaluation/paired_starts.json').read_text())
    env=make_subtask_env('reach',index=manifest['index']);records=[]
    try:
        for seed in [11029,11033,12000,12001,12017]:
            original=next(r for r in main['records'] if r['start']['eval_seed']==seed)
            distances=[original['initial']['metrics']['position_error']]+[s['after']['metrics']['position_error'] for s in original['trace']]
            branch=1 if seed==11033 else min(50,len(original['trace'])-1,max(1,int(np.argmin(distances))))
            reset(env,original['start'])
            if fingerprint(env)!=original['initial_state_fingerprint']:raise ValueError('prefix reset differs')
            error=0.
            for t in range(branch):
                _,_,term,trunc,info=env.step(torch.tensor(original['trace'][t]['applied_action']))
                error=max(error,float(np.abs(env.live_state['qpos']-np.array(original['trace'][t]['after']['qpos'])).max()))
                if term or trunc:raise ValueError('diagnostic branch must precede original termination')
            if error>2e-5:raise ValueError('prefix drift')
            before=dict(env.metrics);ever=env.context.ever_grasped;steps=0
            while True:
                _,_,term,trunc,info=env.step(reference_action(env));steps+=1
                if term or trunc:break
            records.append(dict(seed=seed,learner_prefix_steps=branch,branch_metrics=before,ever_grasped_at_branch=ever,
                                original_outcome=original['failure_reason'],prefix_qpos_max_error=error,
                                reference_success=info['subtask_success'],reference_steps=steps,total_episode_steps=info['subtask_step'],
                                final_metrics=dict(env.metrics),failure_reason=info['failure_reason']))
            print('reference switch',seed,branch,info['subtask_success'],steps,flush=True)
    finally:env.close()
    write_json(path,dict(records=records,diagnostic_only=True,no_dataset_export=True,original_budget_and_predicate_preserved=True))
    return records


def offline(output=OUTPUT):
    """Descriptive metrics and seed-0 comparisons, without optimization."""
    from graph.prototype.metrics import action_metrics
    from graph.prototype.v2.models import CommonPolicy
    seed_all(0,4);output=Path(output);p=Probes();data,g=p.data,p.geometry
    ids=split_indices(data,'train');v=data['qvel'][ids,-2:]
    correlations=torch.corrcoef(torch.cat((v,data['actions'][ids,:6]),-1).T)[:2,2:]
    metrics={}
    with torch.inference_mode():
        for split in ('train','validation','test'):
            rows=split_indices(data,split)
            pred=p.policy.model(*inputs(g.select(rows),data['qvel'][rows],p.policy.norm,p.policy.sn))
            first=torch.tensor([int(torch.where(data['episode_id']==eid)[0][0]) for eid in data['manifest']['splits'][split]])
            last=torch.tensor([int(torch.where(data['episode_id']==eid)[0][-1]) for eid in data['manifest']['splits'][split]])
            initial=p.policy.model(*inputs(g.select(first),data['qvel'][first],p.policy.norm,p.policy.sn))
            final=p.policy.model(*inputs(g.select(last),data['qvel'][last],p.policy.norm,p.policy.sn))
            metrics[split]=dict(metrics=action_metrics(pred,data['actions'][rows]),initial_metrics=action_metrics(initial,data['actions'][first]),
                                last_presuccess_metrics=action_metrics(final,data['actions'][last]),initial_gripper_below_one=int((initial[:,6]<1).sum()),
                                episodes=len(first),last_motion_command_norm_mean=float(final[:,:6].norm(dim=-1).mean()),
                                last_expert_motion_command_norm_mean=float(data['actions'][last,:6].norm(dim=-1).mean()))
    write_json(output/'offline.json',dict(splits=metrics,finger_qvel_train_correlation_with_actions=correlations.tolist(),
                                         train_finger_qvel_min=v.amin(0).tolist(),train_finger_qvel_max=v.amax(0).tolist(),
                                         train_gripper_command_unique=torch.unique(data['actions'][ids,6]).tolist(),
                                         note='No training or model selection; offline measurements at frozen selected checkpoint'))
    seed_all(0,4);initial=CommonPolicy('flat',p.policy.ck['config'],'goal','regression')
    changes=[]
    for layer in ('action_head.0.weight','encoder.net.0.weight'):
        before=initial.state_dict()[layer];after=p.policy.ck['model_state'][layer]
        if layer.startswith('action_head'):
            groups=dict(arm_qvel=list(range(128,135)),finger_qvel=[135,136],goal=list(range(137,159)),embedding=list(range(128)))
        else:
            groups=dict(finger_context=[CONTEXT_START+i for i in FINGER_CONTEXT],ee_rzz=[CONTEXT_START+22],
                        other_context=[CONTEXT_START+i for i in range(81) if i not in list(FINGER_CONTEXT)+[22]],nodes=list(range(CONTEXT_START)))
        for name,columns in groups.items():
            init=before[:,columns].square().mean().sqrt();delta=(after[:,columns]-before[:,columns]).square().mean().sqrt()
            changes.append(dict(layer=layer,group=name,initial_rms=float(init),trained_rms=float(after[:,columns].square().mean().sqrt()),
                                update_rms=float(delta),relative_update_rms=float(delta/init)))
    write_json(output/'weight_updates.json',dict(seed=0,selected_epoch=p.policy.ck['epoch'],groups=changes))
    print('Offline diagnostics reproduced without training.',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,default=OUTPUT)
    a.add_argument('--audit',action='store_true');a.add_argument('--mode',choices=MODES)
    a.add_argument('--feedback',action='store_true')
    a.add_argument('--reference-switch',action='store_true')
    a.add_argument('--offline',action='store_true')
    args=a.parse_args()
    if args.audit:audit(args.output)
    elif args.feedback:feedback(args.output)
    elif args.reference_switch:reference_switch(args.output)
    elif args.offline:offline(args.output)
    elif args.mode:rollout(args.mode,args.output)
    else:a.error('choose --audit or --mode')
