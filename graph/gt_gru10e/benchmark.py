"""Frozen original GRU evaluation; no training and no Graph policy input changes."""
import argparse, csv, hashlib, json, time, subprocess, sys
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
import torch
from recurrent.common import raw_gt, to_action_tensor, versions, make_env
from recurrent.evaluate import LearnedPolicy, run_episode
from recurrent.corruption import Corruptor, level_config
from recurrent.features import build_features, FEATURE_LAYOUT
from recurrent.train import normalize_obs
from recurrent.gpu_eval import TensorPolicy, raw_tensor, features
from graph.geometry_reach10d.runtime import environment
from graph.geometry_reach10d.collect import setup
from graph.geometry_reach10d.evaluate import measure
from graph.integration_baseline.geometry09a import collision_audit, physical_state
from graph.integration_baseline.rollout_stop_clock import controller_audit, validate_controller_command, controller_semantics
from utils.experiment_logger import ExperimentLogger

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False))
def serial(x):
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    if isinstance(x,(torch.Tensor,np.ndarray)):return x.tolist()
    return x

def snapshot(env):
    t=env.unwrapped
    return dict(physical={k:v.tolist() for k,v in physical_state(env).items()},
        simulator=serial(t.get_state_dict()),controller=serial(t.agent.controller.get_state()))

def compare(a,b):
    if isinstance(a,dict):return {k:compare(a[k],b[k]) for k in a}
    x,y=np.asarray(a),np.asarray(b)
    return dict(exact=bool(np.array_equal(x,y)),max_abs=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.)
def all_exact(x):
    return x['exact'] if 'exact' in x else all(all_exact(v) for v in x.values())

def contracts(env,cp):
    """Native CPU evaluator vs official tensor encoder/action/streamed hidden."""
    native=LearnedPolicy(cp);tensor=TensorPolicy(LearnedPolicy(cp),'cpu');cuda=TensorPolicy(LearnedPolicy(cp),'cuda')
    native.reset(env);corr=Corruptor(level_config('none'),np.random.default_rng(2000));rows=[]
    for t in range(12):
        raw,f,a=corr.step(raw_gt(env));encoded=build_features(raw,np.float32(f),np.float32(a))
        rt=raw_tensor(env);fr=torch.ones(1);age=torch.zeros(1);active=torch.ones(1,dtype=torch.bool)
        tf=features(rt,fr,age)[0].numpy();np.testing.assert_array_equal(encoded,tf)
        na=native.act(env,raw,f,a);ta=tensor.act(rt,fr,age,active)[0].numpy()
        np.testing.assert_allclose(na,ta,atol=1e-6,rtol=0)
        torch.testing.assert_close(native.state,tensor.state,atol=1e-6,rtol=0)
        ca=cuda.act({k:v.cuda() for k,v in rt.items()},fr.cuda(),age.cuda(),active.cuda())[0].cpu().numpy()
        rows.append(dict(t=t,feature_exact=True,action_cpu_max_abs=float(abs(na-ta).max()),
            hidden_cpu_max_abs=float((native.state-tensor.state).abs().max()),cuda_action_max_abs=float(abs(na-ca).max()),
            observation38=encoded.tolist(),native_action7=na.tolist(),native_hidden=serial(native.state)))
        env.step(to_action_tensor(na,env))
    return rows

def episode(root,out,cp,spec,primary_horizon=130,video=False):
    env=environment(root);obs,expert,initial=setup(env,spec);snap=snapshot(env)
    policy=LearnedPolicy(cp);policy.reset(env);corr=Corruptor(level_config('none'),np.random.default_rng(spec['seed']))
    threshold=json.loads((root/'collection_protocol.json').read_text())['threshold'];audit=controller_audit(env)
    folder=out/f"s{policy.ckpt['seed']}"/f"{spec['seed']}_{spec['variant']}";folder.mkdir(parents=True,exist_ok=False)
    approach=None;early=False;first_close=None;reach=False;reach250=False;insertion=False;grasp=False;lift=False;trace=[];inputs=[];hidden=[]
    peg0=np.array(initial['perturbed']['peg_pose'])[:3];writer=None
    if video:writer=cv2.VideoWriter(str(folder/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(512,256))
    try:
        for t in range(250):
            raw,f,a=corr.step(raw_gt(env));inputs.append(build_features(raw,np.float32(f),np.float32(a)));pre=measure(env,expert,threshold)
            action=policy.act(env,raw,f,a);hidden.append(policy.state.detach().numpy().copy())
            if action[6]<=0:
                if first_close is None:first_close=t
                if not reach250 and not (approach is not None and pre['final']):early=True
            audit.clear();obs,_,term,trunc,info=env.step(to_action_tensor(action,env));validate_controller_command(action,audit)
            post=measure(env,expert,threshold)
            if post['approach'] and approach is None:approach=t+1
            peg=env.unwrapped.peg.pose.raw_pose[0].cpu().numpy();drift=float(np.linalg.norm(peg[:3]-peg0))
            candidate=approach is not None and post['final'] and post['opening_m']>=threshold['opening_min_m'] and not early and drift<threshold['max_peg_motion_m']
            ok=bool(candidate and collision_audit(env,'peg')['valid'])
            if t<primary_horizon:reach |= ok
            reach250 |= ok
            ins=bool(info['success'].item());insertion |= ins
            g=bool(env.unwrapped.agent.is_grasping(env.unwrapped.peg).item());grasp |= g
            l=bool(peg[2]-raw['peg_half'][2]>.03);lift |= l
            trace.append(dict(timestep=t,post_step=t+1,action=action.tolist(),controller=serial(audit),early_close=early,
                peg_motion_m=drift,grasp=g,lift=l,insertion=ins,reach_candidate=ok,peg_pose=peg.tolist(),
                qpos=env.unwrapped.agent.robot.get_qpos()[0].cpu().tolist(),
                saturation_count=int((abs(action[:6])>=1).sum()),**post))
            if writer:
                frame=np.concatenate([cv2.resize(obs['sensor_data'][c]['rgb'][0].cpu().numpy(),(256,256)) for c in ['base_camera','hand_camera']],axis=1)
                writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            if ins or bool(term.item()) or bool(trunc.item()):break
        window=trace[:primary_horizon]
        reach_step=next((x['post_step'] for x in window if x['reach_candidate']),None)
        primary=window[:reach_step] if reach_step else window;last=primary[-1]
        row=dict(train_seed=policy.ckpt['seed'],eval_seed=spec['seed'],variant=spec['variant'],offset_mm=spec['offset_mm'],
            reach_success=reach,reach_success_250=reach250,actions=len(trace),primary_actions=len(primary),reach_step=reach_step,
            fixed130_position_error_m=window[-1]['target_position_error_m'],fixed130_orientation_error_rad=window[-1]['target_orientation_error_rad'],
            final_target_position_error_m=last['target_position_error_m'],final_target_orientation_error_rad=last['target_orientation_error_rad'],
            min_target_position_error_m=min(x['target_position_error_m'] for x in primary),
            min_target_orientation_error_rad=min(x['target_orientation_error_rad'] for x in primary),
            min_approach_position_error_m=min(x['approach_position_error_m'] for x in primary),
            target_step=next((x['post_step'] for x in primary if x['final']),None),
            approach_step=approach,first_close_step=first_close,early_close=early,grasp=grasp,lift=lift,insertion=insertion,
            threshold_before_close=any(x['final'] and (first_close is None or x['post_step']<=first_close) for x in primary),
            position_threshold_reached=any(x['target_position_error_m']<threshold['final_position_m'] for x in primary),
            orientation_threshold_reached=any(x['target_orientation_error_rad']<threshold['final_orientation_rad'] for x in primary))
        write(folder/'trace.json',trace);write(folder/'episode.json',dict(metrics=row,initial=initial,snapshot=snap))
        np.savez_compressed(folder/'inference_trace.npz',observation38=np.array(inputs),hidden=np.array(hidden),action7=np.array([x['action'] for x in trace]))
        print('EPISODE',row,flush=True);return row,snap
    finally:
        if writer:writer.release()
        env.close()

def run(root,out,inputs,smoke):
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True)
    cps=sorted((inputs/'training').rglob('best.pt'));assert len(cps)==3
    proto=json.loads((root/'collection_protocol.json').read_text());specs=proto['final_cases']
    inv=[]
    for cp in cps:
        c=torch.load(cp,map_location='cpu',weights_only=False);s=c['seed'];ident=json.loads((inputs/f'original/outputs/recurrent/reevaluation_main_20261023/eval/gru_none_s{s}.identity.json').read_text())
        assert sha(cp)==ident['checkpoint'] and c['kind']=='gru' and c['train_level']=='none' and c['obs_dim']==38
        inv.append(dict(path=str(cp),sha256=sha(cp),train_seed=s,architecture=c['cfg'],params=c['params'],training_identity=c['identity'],step=c['step'],stats=c['stats']))
    with ExperimentLogger(team='graph',model='Task10E_GT_GRU',seed=None,config=dict(task='10E',checkpoints=inv,final_cases=specs,primary_horizon=130,full_horizon=250),root=out/'experiments',job_type='evaluate') as log:
        write(out/'run_identity.json',dict(run_id=log.run_id));write(out/'recurrent_checkpoint_inventory.json',inv)
        subprocess.run([sys.executable, '-m', 'graph.gt_gru10e.benchmark', '--root', str(root), '--out', str(out), '--inputs', str(inputs), '--native-only'],check=True)
        env=environment(root);setup(env,specs[0]);t=env.unwrapped
        write(out/'evaluation_environment_contract.json',dict(versions=versions(),backend=str(t.backend),controller={k:asdict(v.config) for k,v in t.agent.controller.controllers.items()},control_freq=t.control_freq,sim_freq=t.sim_freq,primary_horizon=130,secondary_horizon=250))
        env.close();rows=[];audits=[]
        for cp in cps[:1] if smoke else cps:
            for spec in specs[:2] if smoke else specs:
                row,snap=episode(root,out/'rollouts',cp,spec,video=spec['variant']=='nominal');rows.append(row)
                references=[]
                for model in ['A','B']:
                    matches=list((root/f'evaluations/final/{model}').glob(f"step*/{spec['seed']}_{spec['variant']}/episode.json"));assert len(matches)==1
                    ref=json.loads(matches[0].read_text());d=compare(snap['physical'],ref['initial_state']['perturbed'])
                    references.append(dict(model=model,source=str(matches[0]),sha256=sha(matches[0]),fields=d,exact=all_exact(d)))
                env=environment(root);setup(env,spec);repeat=snapshot(env);env.close();d=compare(snap,repeat)
                audits.append(dict(train_seed=row['train_seed'],spec=spec,AB=references,repeat=d,repeat_exact=all_exact(d)))
                write(out/'reset_identity_audit.json',audits)
                write(out/'rollout_rows.json',rows)
        for cp,entry in zip(cps,inv):assert sha(cp)==entry['sha256']
        log.record(checkpoint_weight_hashes_unchanged=True,episodes=len(rows));log.log(dict(episodes=len(rows),reach_success=sum(r['reach_success'] for r in rows)))
        for name in ['rollout_rows.json','reset_identity_audit.json','native_clean_smoke.json']:log.link_output(out/name,kind='metrics')
def native_only(out,inputs):
    torch.set_num_threads(1)
    cps=sorted((inputs/'training').rglob('best.pt'))
    env=make_env();env.reset(seed=2000)
    try:
        results=[]
        for cp in cps:
            env.reset(seed=2000);results.append(dict(train_seed=LearnedPolicy(cp).ckpt['seed'],steps=contracts(env,cp)))
        write(out/'native_contract_test.json',results)
    finally:env.close()
    # Native fixed Clean seeds, unmodified original run_episode.
    native=[]
    env=make_env()
    try:
        for cp in cps:
            for seed in [2000,2001]:
                native.append(dict(train_seed=LearnedPolicy(cp).ckpt['seed'],**run_episode(env,LearnedPolicy(cp),seed,level_config('none'),{},250)))
                print('NATIVE',native[-1],flush=True)
    finally:env.close()
    write(out/'native_clean_smoke.json',native)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--native-only',action='store_true');a=p.parse_args();native_only(a.out,a.inputs) if a.native_only else run(a.root,a.out,a.inputs,a.smoke)
