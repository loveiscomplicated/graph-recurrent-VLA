"""Task08 adapter around Task07's actual stop-the-clock rollout loop."""
import argparse
from dataclasses import asdict
import inspect
from pathlib import Path

import numpy as np
import torch

from graph.integration_baseline.bc08_codec import DecodedPolicy, ActionCodec
from graph.integration_baseline.bc08_train import load_plan, load_checkpoint, load_normalizer
from graph.integration_baseline.direct_runtime import atomic, read, Events
from graph.integration_baseline.model import IntegrationPolicy
from graph.integration_baseline.rollout_stop_clock import (run_episode, reset_diagnostics,
    controller_semantics, time_contract, numpy, PhysicsAudit)
from graph.integration_baseline.rollout_checkpoint import load_bundle, vision_runtime_identity
from graph.integration_baseline.rollout_metrics import wilson, OFFICIAL_NAME
from utils.experiment_logger import ExperimentLogger


def validate_checkpoint_plan(cp,codec,receipt,plan,plan_path,phase,method):
    from graph.integration_baseline.bc08_train import make_identity,normalizer_hash
    from graph.integration_baseline.bc08_codec import digest
    from graph.integration_baseline.direct_runtime import sha
    if cp['identity']['phase']!=phase or cp['identity']['method']!=method:
        raise ValueError('evaluation checkpoint phase/method mismatch')
    if cp['identity']['plan_sha256']==sha(plan_path):
        if cp['identity']!=make_identity(plan,plan_path,phase,method):raise ValueError('checkpoint experiment identity mismatch')
        return
    adopted=plan.get('adopted_overfit',{}).get(method) if phase=='overfit' else None
    if (not adopted or receipt!=adopted['receipt'] or cp['identity']!=adopted['identity'] or
        codec.identity!=digest(plan['codecs']['overfit'][method]) or
        cp['identity']['initial_sha256']!=plan['initial_weight_identity'] or
        normalizer_hash(cp['normalizer'])!=plan['robot_normalizer_identity'] or
        cp['identity']['train_ids']!=plan['overfit_ids']):
        raise ValueError('checkpoint is not an explicitly audited adopted overfit result')


def environment(plan):
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.utils.registration import TimeLimitWrapper
    from recurrent.common import versions
    env=TimeLimitWrapper(make_peg_insertion_env(obs_mode='rgb+depth',control_mode='pd_ee_delta_pose',device='cpu').unwrapped,250)
    task=env.unwrapped;reference=plan['reference_provenance']['conditions']
    actual=dict(versions=versions(),control_freq=int(task.control_freq),physics_freq=int(task.sim_freq),
        horizon=env._max_episode_steps,robot=task.agent.uid,backend=str(task.backend),
        controller_config={k:asdict(v.config) for k,v in task.agent.controller.controllers.items()})
    try:
        for key in ['control_freq','physics_freq','horizon','robot']:
            if actual[key]!=reference[key]:raise ValueError('reference/live simulator mismatch: '+key)
        for key in ['mani_skill','sapien','gymnasium']:
            if actual['versions'][key]!=reference['versions'][key]:raise ValueError('simulator version mismatch: '+key)
        if controller_semantics(actual['controller_config'])!=controller_semantics(reference['controller_config']):
            raise ValueError('controller convention mismatch')
        if task._sim_steps_per_control!=5 or not np.isclose(task.scene.px.timestep,.01,atol=1e-9,rtol=0):
            raise ValueError('20Hz/100Hz physical clock contract')
    except BaseException:
        env.close();raise
    return env,actual


def freeze(plan_path):
    plan=load_plan(plan_path);root=Path(plan_path).parent
    if (root/'seeds_manifest.json').exists():return read(root/'seeds_manifest.json')
    original={e['seed']:e['reset_group'] for e in plan['reference_provenance']['dataset']['episodes']}
    selected={e['seed']:e['reset_group'] for e in plan['overfit_selection']}
    groups={'train_seed':list(selected),'development':plan['development_seeds'],'final':plan['final_seeds'],
        'historical7':plan['old_comparison_seeds']}
    devfinal=groups['development']+groups['final']
    if len(devfinal)!=len(set(devfinal)) or set(devfinal)&(set(original)|set(plan['diagnostic_seeds'])|set(groups['historical7'])):
        raise ValueError('development/final/test/diagnostic seed leakage')
    env,conditions=environment(plan);records={};new_groups=set()
    try:
        contract=time_contract(env,dict(time_contract_seed=44000))
        for name,seeds in groups.items():
            records[name]=[]
            for seed in seeds:
                env.reset(seed=seed);group=reset_diagnostics(env)
                if name in ('development','final'):
                    if group in set(original.values())|new_groups:raise ValueError('runtime reset group leakage')
                    new_groups.add(group)
                records[name].append(dict(seed=seed,group='heldout' if name!='train_seed' else 'train_seed',
                    reset_group=group,original_reset_group=original.get(seed),
                    identity_matches_collection=(group==original[seed]) if seed in original else None,
                    status='BLOCKED_RESET_IDENTITY' if name=='train_seed' and group!=selected[seed] else 'READY'))
        # Audit all frozen test reset identities; never call these new states the
        # original test episodes, and never choose only those that happen to match.
        test=[]
        for ep in plan['reference_provenance']['dataset']['episodes']:
            if ep['episode_id'] not in plan['splits']['test']:continue
            env.reset(seed=ep['seed']);actual=reset_diagnostics(env)
            test.append(dict(seed=ep['seed'],original=ep['reset_group'],actual=actual,match=actual==ep['reset_group']))
        result=dict(groups=records,conditions=conditions,frozen_test_reset_audit=test,time_contract=contract,
            frozen_before_policy_training=True,final_model_selection=plan['selection'])
        atomic(root/'seeds_manifest.json',result,immutable=True);return result
    finally:env.close()


def summarize(rows):
    complete=[r for r in rows if r['status']=='COMPLETE'];n=len(complete)
    if n!=len(rows):
        return dict(status='INCOMPLETE',planned=len(rows),complete=n,success_rate=None,
            blocked_reset_identity=sum(r['status']=='BLOCKED_RESET_IDENTITY' for r in rows))
    k=sum(r['success'] for r in complete);g=sum(r['grasp_success'] for r in complete)
    return dict(status='COMPLETE',episodes=n,successes=k,success_rate=k/n if n else None,
        success_ci95=wilson(k,n),grasp_successes=g,grasp_success_rate=g/n if n else None,
        grasp_ci95=wilson(g,n),mean_episode_length=float(np.mean([r['episode_length'] for r in complete])) if n else None,
        first_success_steps=[r['first_success_step'] for r in complete],timeouts=sum(r['timeout'] for r in complete),
        failures=sum(not r['success'] for r in complete),
        metric='Full autonomous Peg Insertion / info.success; stop-the-clock, no real-time deadline')


def evaluate(plan_path, phase, method, group, receipt=None, *, smoke=False):
    plan=load_plan(plan_path);frozen=read(Path(plan_path).parent/'seeds_manifest.json')
    device=plan['config']['device'];torch.set_num_threads(1);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if device=='cuda':
        torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if method=='reference':
        model,norm,_=load_bundle(plan['reference_checkpoint_manifest'],device=device)
        codec=ActionCodec(plan['codecs']['full']['A']);selected=plan['reference_checkpoint_manifest']
    else:
        cp,codec=load_checkpoint(receipt)
        validate_checkpoint_plan(cp,codec,read(receipt),plan,plan_path,phase,method)
        model=IntegrationPolicy(cp['normalizer']['schema'],**plan['config']['model']).to(device)
        model.load_state_dict(cp['model'],strict=True);norm=load_normalizer(cp['normalizer']);selected=read(receipt)
    model.eval();policy=DecodedPolicy(model,codec)
    runtime=vision_runtime_identity()
    if runtime!=plan['reference_provenance']['vision_runtime']:
        raise ValueError('original DINO/SAM2 runtime identity mismatch')
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    vision=OnlineVisionAdapter(GDino(device=device),SAM2Tracker(device=device))
    specs=frozen['groups'][group]
    if smoke:specs=[dict(seed=44001,group='smoke',reset_group=None,status='READY')]
    config=dict(plan['config'],smoke_steps=3)
    with ExperimentLogger(team='integration',model=f'task08_{phase}_{method}_{group}',seed=42,
        config=dict(phase=phase,method=method,group=group,selected=selected,smoke=smoke),
        root=plan['experiments_root'],job_type='evaluate') as logger:
        run=logger.directory;events=Events(run)
        from graph.integration_baseline.bc08_report import snapshot,verify
        snapshot(run,plan)
        result=dict(status='RUNNING',official_name=OFFICIAL_NAME,phase=phase,method=method,group=group,
            checkpoint=selected,action_codec=codec.state,episodes=[dict(s,status='NOT_EXECUTED') for s in specs],smoke=smoke)
        atomic(run/'seeds_manifest.json',frozen,immutable=True)
        try:
            for i,spec in enumerate(specs):
                if spec['status']=='BLOCKED_RESET_IDENTITY':
                    result['episodes'][i]=dict(spec);continue
                env,conditions=environment(plan);diag=[];original_step=env.step
                from envs.maniskill import get_gt_state
                # Outcome recorder is downstream of action application; none of
                # these values are passed to observation()/Vision/Graph/policy.
                baseline=[None]
                def step(action):
                    if baseline[0] is None:baseline[0]=float(get_gt_state(env).peg_pose[0,2])
                    out=original_step(action);task=env.unwrapped;gt=get_gt_state(env)
                    diag.append(dict(timestep=len(diag),grasp=bool(task.agent.is_grasping(task.peg).item()),
                        peg_lift_m=float(gt.peg_pose[0,2])-baseline[0],
                        finger_forces=np.stack([numpy(task.scene.get_pairwise_contact_forces(link,task.peg))[0]
                            for link in [task.agent.finger1_link,task.agent.finger2_link]]).tolist()))
                    return out
                env.step=step;policy.raw_outputs=[]
                folder=run/'episodes'/f'{group}_{spec["seed"]}'
                try:
                    ep=run_episode(env,policy,norm,vision,spec,config,folder,smoke=smoke,
                        forbidden_groups=[] if group in ('train_seed','historical7') else
                        [e['reset_group'] for e in plan['reference_provenance']['dataset']['episodes']])
                    atomic(folder/'grasp_contact_lift.json',diag)
                    np.savez_compressed(folder/'encoded_action_output.npz',raw_output=np.asarray(policy.raw_outputs),codec_identity=codec.identity)
                    ep.update(grasp_success=any(d['grasp'] for d in diag),
                        bilateral_contact_observed=any(all(np.linalg.norm(d['finger_forces'],axis=1)>=.5) for d in diag),
                        lifted_4cm=any(d['peg_lift_m']>=.04 for d in diag),
                        first_grasp_step=next((d['timestep']+1 for d in diag if d['grasp']),None),
                        conditions=conditions,artifact_directory=str(folder.resolve()))
                    if len(diag)!=ep['episode_length']:raise ValueError('outcome trace/action count mismatch')
                    atomic(folder/'episode.json',ep);result['episodes'][i]=ep
                    atomic(run/'evaluation.json',result)
                    events.emit('evaluation_episode',method=method,group=group,seed=spec['seed'],success=ep['success'],grasp=ep['grasp_success'],steps=ep['episode_length'])
                    if ep['status']=='ERROR':raise RuntimeError('closed-loop contract failed')
                finally:env.close()
            result.update(status='SMOKE_COMPLETE' if smoke else 'COMPLETE',summary=None if smoke else summarize(result['episodes']))
            atomic(run/'evaluation.json',result)
            atomic(run/'verification.json',verify(run),immutable=True)
            logger.link_output(run/'evaluation.json',kind='metrics');events.emit('evaluation_complete',method=method,group=group,summary=result['summary'],artifacts=str(run.resolve()))
            return dict(directory=str(run.resolve()),**result)
        except BaseException as exc:
            from graph.integration_baseline.bc08_train import error_bundle
            error_bundle(run,exc,dict(phase=phase,method=method,group=group));raise
        finally:events.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--freeze',action='store_true');p.add_argument('--phase',choices=['overfit','full'])
    p.add_argument('--method',choices=['A','B','C','reference']);p.add_argument('--receipt',type=Path)
    p.add_argument('--group',choices=['train_seed','development','final','historical7']);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.freeze:freeze(a.plan)
    else:
        if not a.phase or not a.method or not a.group or (a.method!='reference' and not a.receipt):p.error('phase/method/group/receipt required')
        evaluate(a.plan,a.phase,a.method,a.group,a.receipt,smoke=a.smoke)


if __name__=='__main__':main()
