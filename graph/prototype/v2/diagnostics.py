"""Executable audit and richly traced, paired v1/v2 policy evaluation."""
from utils.experiment_logger import ExperimentLogger
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import h5py
import numpy as np
import torch
from ..common import seed_all, write_json, versions
from ..data import load_dataset, reset_episode, state_at, split_indices, sha256, PHASES
from ..geometry import rotation
from ..maniskill_gt import GTAdapter
from ..graph_builder import GraphBuilder
from ..models import make_policy
from ..normalization import FeatureNormalizer
from ..metrics import action_metrics
from .goals import goal_geometry, angle

THRESHOLDS = dict(approach_distance_m=.025, approach_angle_rad=float(np.deg2rad(20)),
                  retention_steps=5, lift_height_m=.03, alignment_transverse_m=.01,
                  alignment_axis_rad=float(np.deg2rad(10)), alignment_entry_distance_m=.04)


def snapshot(env, g):
    goal = goal_geometry(g)
    relative = goal['head_in_hole'][0]
    grasp_angle = torch.minimum(angle(goal['grasp_relative_rotation']),
        angle(goal['grasp_relative_rotation'] * g.qpos.new_tensor([-1.,-1.,1.])[None,None,:]))
    task = env.unwrapped
    forces = [task.scene.get_pairwise_contact_forces(link, task.peg)[0]
              for link in (task.agent.finger1_link, task.agent.finger2_link)]
    grasped = bool(task.agent.is_grasping(task.peg).item())
    axis_angle = torch.acos(goal['peg_relative_rotation'][0,0,0].clamp(-1,1))
    depth = float(g.box_half_size[0,0])
    transverse = float(relative[1:].norm())
    lifted = float(g.peg_pose[0,2]-g.peg_half_size[0,2]) > THRESHOLDS['lift_height_m']
    return dict(ee_pose=g.ee_pose[0].tolist(), peg_pose=g.peg_pose[0].tolist(),
        hole_pose=g.hole_pose[0].tolist(), qpos=g.qpos[0].tolist(),
        qvel=task.agent.robot.get_qvel()[0].tolist(), gripper_state=g.gripper[0].tolist(),
        grasp_target_position=goal['grasp_position'][0].tolist(),
        grasp_distance_m=float(goal['ee_to_grasp'].norm()), grasp_angle_rad=float(grasp_angle[0]),
        head_in_hole_m=relative.tolist(), transverse_error_m=transverse,
        axis_angle_rad=float(axis_angle), rotation_error_rad=float(angle(goal['peg_relative_rotation'])[0]),
        clearance_m=float(g.hole_radius[0,0]-g.peg_half_size[0,1]),
        contact_force_N=[f.tolist() for f in forces], contact_norm_N=[float(f.norm()) for f in forces],
        grasped=grasped, lifted=lifted,
        approach=(float(goal['ee_to_grasp'].norm()) <= THRESHOLDS['approach_distance_m'] and float(grasp_angle[0]) <= THRESHOLDS['approach_angle_rad']),
        alignment=(transverse <= THRESHOLDS['alignment_transverse_m'] and float(axis_angle) <= THRESHOLDS['alignment_axis_rad'] and abs(float(relative[0])+depth) <= THRESHOLDS['alignment_entry_distance_m']),
        insertion=(float(relative[0]) >= -depth and abs(float(relative[1])) <= float(g.hole_radius[0,0]) and abs(float(relative[2])) <= float(g.hole_radius[0,0])),
        insertion_progress=float(((relative[0]+depth)/(2*depth)).clamp(0,1)))


def stage_summary(trace, initial=None):
    rows = ([initial] if initial else []) + [r['after'] for r in trace]
    run, acquisition, transport, losses, previous = 0, False, False, 0, False
    for r in rows:
        run = run+1 if r['grasped'] else 0
        acquisition |= run >= THRESHOLDS['retention_steps']
        transport |= run >= THRESHOLDS['retention_steps'] and r['lifted']
        losses += int(previous and not r['grasped'])
        previous = r['grasped']
    success = any(r['success'] for r in trace)
    aligned = any(r['alignment'] for r in rows)
    failure = None if success else ('acquisition' if not acquisition else 'retention' if not rows[-1]['grasped'] else 'transport' if not transport else 'insertion' if aligned else 'alignment')
    return dict(approach=any(r['approach'] for r in rows), acquisition=acquisition, transport=transport,
                alignment=aligned, insertion=any(r['insertion'] for r in rows), success=success,
                grasp_losses=losses, failure_stage=failure)


class ReplaySession:
    def __init__(self, data, control_mode='pd_ee_delta_pose', horizon=250, render_mode=None):
        from envs.maniskill import make_peg_insertion_env
        self.env = make_peg_insertion_env(control_mode=control_mode,render_mode=render_mode)
        self.data, self.horizon = data, horizon
        self.converted = h5py.File(data['manifest']['source'], 'r')
        self.original = h5py.File(data['manifest']['initial_state_source'], 'r')
        self.checks = {r['episode_id']:r for r in data['manifest']['snapshot_checks']}
        cfg = data['manifest']['config']
        self.adapter = GTAdapter(cfg['peg_points'],cfg['hole_points'],cfg['sampling_seed'])
        self.original_mode = control_mode == 'pd_joint_pos'

    def group(self, eid):
        return self.original[f"traj_{self.checks[eid]['source_episode_id']}"] if self.original_mode else self.converted[f'traj_{eid}']

    def prepare(self, ep, prefix=0):
        from mani_skill.utils.registration import TimeLimitWrapper
        self.env = TimeLimitWrapper(self.env.unwrapped, self.horizon+prefix)
        reset_episode(self.env, ep)
        if not ep.get('new_configuration',False):
            original_id = self.checks[ep['episode_id']]['source_episode_id']
            self.env.unwrapped.set_state_dict(state_at(self.original[f'traj_{original_id}/env_states'],0))
        self.env.unwrapped.agent.controller.reset()
        error=0.
        prefix_actions=self.group(ep['episode_id'])['actions'][:prefix] if prefix else []
        for t, action in enumerate(prefix_actions):
            obs,*_ = self.env.step(torch.tensor(action)[None])
            if not self.original_mode:
                error=max(error,float(np.max(np.abs(obs.numpy()[0]-self.group(ep['episode_id'])['obs'][t+1]))))
        if error > .001:
            raise ValueError(f'prefix diverged {error}')
        g=self.adapter(self.env)
        fingerprint=hashlib.sha256(b''.join(v.numpy().tobytes() for v in g.as_dict().values())).hexdigest()
        return error,fingerprint

    def close(self):
        self.env.close(); self.original.close(); self.converted.close()


def run_episode(session, ep, predictor=None, prefix=0, expert=False, trace_limit=None, frame_callback=None):
    error, fingerprint=session.prepare(ep,prefix)
    env=session.env
    initial=snapshot(env,session.adapter(env))
    trace=[]
    actions=session.group(ep['episode_id'])['actions'] if expert else None
    limit=min(session.horizon,len(actions)-prefix) if expert else session.horizon
    if trace_limit is not None:
        limit=min(limit,trace_limit)
    obs_error=0.
    for t in range(limit):
        g=session.adapter(env)
        before=snapshot(env,g)
        raw=torch.tensor(actions[prefix+t])[None] if expert else predictor(g,torch.tensor(before['qvel'])[None])
        if raw.shape!=(1,len(env.action_space.low)) or not torch.isfinite(raw).all():raise ValueError('invalid policy action shape/values')
        applied=raw.clamp(torch.tensor(env.action_space.low),torch.tensor(env.action_space.high))
        obs,_,terminated,truncated,info=env.step(applied)
        after=snapshot(env,session.adapter(env))
        realized=(rotation(g.ee_pose[:,3:]).transpose(-1,-2) @ rotation(session.adapter(env).ee_pose[:,3:]))
        row=dict(t=t,absolute_t=prefix+t,before=before,after=after,
                 prediction=raw[0].tolist(),applied_action=applied[0].tolist(),
                 realized_ee_translation_m=(torch.tensor(after['ee_pose'][:3])-g.ee_pose[0,:3]).tolist(),
                 realized_ee_rotation_rad=float(angle(realized)[0]),
                 success=bool(info['success'].item()),terminated=bool(terminated.item()),truncated=bool(truncated.item()),
                 elapsed_steps=int(env.unwrapped.elapsed_steps.item()))
        if expert and not session.original_mode:
            row['expert_action']=raw[0].tolist()
            obs_error=max(obs_error,float(np.max(np.abs(obs.numpy()[0]-session.group(ep['episode_id'])['obs'][prefix+t+1]))))
        trace.append(row)
        if frame_callback is not None and t % 4 == 0:frame_callback(env)
        if not expert and (row['success'] or row['terminated'] or row['truncated']):
            break
    return dict(episode_id=ep['episode_id'],seed=ep['episode_seed'],prefix=prefix,
                initial_geometry_sha256=fingerprint,prefix_observation_max_error=error,
                observation_max_error=obs_error if expert and not session.original_mode else None,
                initial=initial,trace=trace,steps=len(trace),final_success=trace[-1]['success'],
                final=trace[-1]['after'],**stage_summary(trace,initial))


def load_predictor(path,data,dataset_path='outputs/graph/dataset.pt'):
    ck=torch.load(path,weights_only=True)
    if ck['splits'] != data['manifest']['splits'] or ck['dataset_sha256'] != sha256(dataset_path):
        raise ValueError('dataset/split mismatch')
    if ck.get('schema_version')==2:
        from .state import StateNormalizer
        from .models import CommonPolicy
        model=CommonPolicy(ck['model_name'],ck['config'],ck['state_mode'],ck['gripper_mode'])
        sn=StateNormalizer(**ck['state_normalizer'])
    else:
        model=make_policy(ck['model_name'],ck['config']); sn=None
    model.load_state_dict(ck['model_state']); model.eval()
    norm=FeatureNormalizer(**ck['normalizer'])
    builder=GraphBuilder(ck['config']['surface_k'],ck['config']['interaction_k'],center=ck['model_name']=='center')
    def predict(g,qvel):
        inputs=norm.apply(ck['model_name'],g,builder)
        return model(inputs,sn.apply(g,qvel,ck['state_mode'])) if sn else model(inputs)
    return predict,ck,model,norm,builder,sn


def qvel_dataset(data):
    """Validated state observations start with robot qpos9,qvel9."""
    result=torch.empty(len(data['actions']),9)
    with h5py.File(data['manifest']['source'],'r') as f:
        for ep in data['manifest']['episodes']:
            rows=torch.where(data['episode_id']==ep['episode_id'])[0]
            obs=np.array(f[f"traj_{ep['episode_id']}/obs"])
            t=data['timestep'][rows].numpy()
            if np.max(np.abs(obs[t,:9]-data['geometry']['qpos'][rows].numpy()))>2e-5:
                raise ValueError('qpos observation ordering mismatch')
            result[rows]=torch.from_numpy(obs[t,9:18].copy())
    if not torch.isfinite(result).all(): raise ValueError('qvel not finite')
    return result


def audit(output,data):
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.utils.registration import TimeLimitWrapper
    env=make_peg_insertion_env(control_mode='pd_ee_delta_pose')
    default_limit=env._max_episode_steps
    records=[]
    for horizon in (100,250):
        wrapped=TimeLimitWrapper(env.unwrapped,horizon)
        wrapped.reset(seed=50,options={'reconfigure':True})
        for t in range(horizon):
            _,_,terminated,truncated,info=wrapped.step(torch.zeros(1,7))
            if t in (98,99,100,horizon-1):
                records.append(dict(horizon=horizon,t=t+1,elapsed=int(env.unwrapped.elapsed_steps.item()),truncated=bool(truncated.item()),terminated=bool(terminated.item()),success=bool(info['success'].item())))
    controller=env.unwrapped.agent.controller.controllers['arm']
    cfg=controller.config
    report=dict(default_limit=default_limit,records=records,action_low=env.action_space.low.tolist(),action_high=env.action_space.high.tolist(),control_mode=env.unwrapped.agent.control_mode,controller_config=repr(cfg),versions=versions(),backend='physx_cpu',device='cpu')
    env.close()
    actions=data['actions']; grip=actions[:,6]
    report['actions']=dict(shape=list(actions.shape),minimum=actions.min(0).values.tolist(),maximum=actions.max(0).values.tolist(),mean=actions.mean(0).tolist(),std=actions.std(0).tolist(),gripper_unique=torch.unique(grip).tolist(),open_fraction=float((grip>.5).float().mean()),close_fraction=float((grip<-.5).float().mean()),intermediate_fraction=float((grip.abs()<=.5).float().mean()))
    write_json(output,report)
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('outputs/graph/dataset.pt'))
    p.add_argument('--output',type=Path,required=True)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--audit',action='store_true')
    choice.add_argument('--expert',action='store_true')
    choice.add_argument('--original-expert',action='store_true')
    choice.add_argument('--checkpoint',type=Path)
    p.add_argument('--stages',nargs='+',default=['full','alignment'],choices=['full','grasp','transport','alignment'])
    p.add_argument('--expert-report',type=Path,default=Path('outputs/graph/v2/diagnostics/expert.json'))
    p.add_argument('--horizon',type=int,default=250)
    p.add_argument('--episode-ids',nargs='+',type=int)
    p.add_argument('--new-seeds',nargs='+',type=int,help='actual fresh reset configurations; full stage only')
    args=p.parse_args()
    if args.horizon<1:p.error('horizon must be positive')
    data,geometry=load_dataset(args.dataset); seed_all(0,4)
    with ExperimentLogger(team='graph', config=vars(args), seed=0, job_type='evaluate') as experiment:
        if args.audit:
            audit(args.output,data);return
        predictor,ck=None,None
        if args.checkpoint:
            predictor,ck,*_=load_predictor(args.checkpoint,data,args.dataset)
            experiment.record(parameters=ck['config'], model=ck['model_name'], dataset_sha256=ck['dataset_sha256'], environment=data['manifest']['config'])
            experiment.link_output(args.checkpoint, checkpoint=True)
        elif not (args.expert or args.original_expert):
            p.error('choose audit, expert, original-expert or checkpoint')
        if args.episode_ids and not set(args.episode_ids).issubset(data['manifest']['splits']['test']):p.error('episode ids must be held-out')
        episodes=[e for e in data['manifest']['episodes'] if e['episode_id'] in data['manifest']['splits']['test'] and (not args.episode_ids or e['episode_id'] in args.episode_ids)]
        if args.new_seeds:
            if args.stages != ['full'] or not args.checkpoint:p.error('fresh seeds require checkpoint and --stages full')
            used={e['episode_seed'] for e in data['manifest']['episodes']}
            if any(s<0 for s in args.new_seeds) or len(set(args.new_seeds))!=len(args.new_seeds) or used & set(args.new_seeds):p.error('fresh seeds must be unique and absent from dataset')
            episodes=[dict(episode_id=-s,episode_seed=s,reset_kwargs={'seed':s},new_configuration=True) for s in args.new_seeds]
        starts={}
        if 'grasp' in args.stages or 'transport' in args.stages:
            previous=json.loads(args.expert_report.read_text())
            starts=previous['starts']
        report=dict(thresholds=THRESHOLDS,horizon=args.horizon,checkpoint=str(args.checkpoint) if args.checkpoint else None,control_mode='pd_joint_pos' if args.original_expert else 'pd_ee_delta_pose',versions=versions(),settings={},starts={},new_configuration_seeds=args.new_seeds,physics_backend='physx_cpu',torch_device='cpu')
        session=ReplaySession(data,report['control_mode'],args.horizon)
        try:
            with torch.inference_mode():
                for stage in args.stages:
                    records=[]
                    for ep in episodes:
                        eid=ep['episode_id'];prefix=0
                        if stage=='alignment':
                            rows=torch.where((data['episode_id']==eid)&(data['phase']==3))[0]
                            if not len(rows): rows=torch.where((data['episode_id']==eid)&(data['phase']==4))[0]
                            prefix=int(data['timestep'][rows[0]])
                        elif stage!='full':
                            prefix=starts[str(eid)].get(stage)
                            if prefix is None: continue
                        if args.original_expert and stage!='full':
                            raise ValueError('original control uses original full trajectories only')
                        r=run_episode(session,ep,predictor,prefix,args.expert or args.original_expert)
                        records.append(r)
                        if args.expert and stage=='full':
                            run=0; start={'grasp':None,'transport':None}
                            for row in r['trace']:
                                run=run+1 if row['after']['grasped'] else 0
                                if run>=5:
                                    if start['grasp'] is None: start['grasp']=row['absolute_t']+1
                                    if start['transport'] is None and row['after']['lifted']:start['transport']=row['absolute_t']+1
                            report['starts'][eid]=start
                        print(f"{stage} ep{eid} success={r['success']} steps={r['steps']} failure={r['failure_stage']}",flush=True)
                    report['settings'][stage]=dict(episodes=records,denominator=len(records),successes=sum(r['success'] for r in records),stage_counts={k:sum(r[k] for r in records) for k in ['approach','acquisition','transport','alignment','insertion']},failure_counts={k:sum(r['failure_stage']==k for r in records) for k in ['acquisition','retention','transport','alignment','insertion']})
                    write_json(args.output,report)
                    experiment.log(report, prefix='eval')
                    experiment.link_output(args.output)
        finally:session.close()


if __name__=='__main__':main()
