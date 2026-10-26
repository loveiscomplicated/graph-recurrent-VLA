"""Task08-F: causal action branch on a repeated expert prefix, without training."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch

from envs.maniskill import get_gt_state
from graph.integration_baseline.bc08_codec import DecodedPolicy, controller_commands
from graph.integration_baseline.bc08_eval import environment, validate_checkpoint_plan
from graph.integration_baseline.bc08_train import load_checkpoint, load_normalizer, load_plan
from graph.integration_baseline.direct_dataset import DirectDataset, load_episode
from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import save_npz, save_packet
from graph.integration_baseline.model import EpisodeStream, IntegrationPolicy
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.robot import flatten_state, robot_observation
from graph.integration_baseline.rollout_checkpoint import vision_runtime_identity
from graph.integration_baseline.rollout_stop_clock import (
    PhysicsAudit, controller_audit, numpy, observation, reset_diagnostics,
    validate_controller_command,
)
from utils.experiment_logger import ExperimentLogger
from visual.runtime.rgbd import TrackedRGBDFrame

SEED = 30003
BRANCHES = (0, 10, 25, 49)
SHORT_HORIZON = 10
EXPERT_H5_SHA = 'ba578d3e17e804c48c5490dc9970406a2b560d9ea214519094965fa815a09122'
GATE = dict(physical_atol=1e-5, float_atol=1e-5, rgb_lsb=1, depth_mm=1,
            topology_exact=True, action_atol=1e-5)
FIELDS = ('qpos', 'qvel', 'ee_pose', 'left_finger_pose', 'right_finger_pose',
          'peg_pose', 'box_pose', 'left_fingertip_pos', 'right_fingertip_pos')


def put(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    atomic(path, json.loads(json.dumps(value, allow_nan=False)), immutable=True)


def gt(env):
    state = get_gt_state(env)
    task = env.unwrapped
    out = {name: numpy(getattr(state, name))[0].tolist() for name in FIELDS}
    out['grasp'] = bool(task.agent.is_grasping(task.peg).item())
    out['finger_forces'] = [numpy(task.scene.get_pairwise_contact_forces(link, task.peg))[0].tolist()
                            for link in (task.agent.finger1_link, task.agent.finger2_link)]
    return out


def choose_action(method, arm, t, branch, prediction, expert):
    if method not in 'ABC' or arm not in ('EM', 'ME', 'MM') or branch not in BRANCHES:
        raise ValueError('invalid frozen branch specification')
    if t < branch:
        return expert.copy(), 'EXPERT_PREFIX'
    if t == branch:
        return (expert.copy(), 'EXPERT_BRANCH') if arm[0] == 'E' else (prediction.copy(), 'MODEL_BRANCH')
    return (expert.copy(), 'EXPERT_SUFFIX') if arm[1] == 'E' else (prediction.copy(), 'MODEL_SUFFIX')


def load_policy(plan, plan_path, receipt, method):
    cp, codec = load_checkpoint(receipt)
    validate_checkpoint_plan(cp, codec, read(receipt), plan, plan_path, 'full', method)
    model = IntegrationPolicy(cp['normalizer']['schema'], **plan['config']['model']).to(plan['config']['device'])
    model.load_state_dict(cp['model'], strict=True)
    model.eval()
    return DecodedPolicy(model, codec), load_normalizer(cp['normalizer'])


def reference_replay(plan, actions, norm, reset_group, folder):
    """One physical E/E trajectory; the stored expert action sequence is never refitted."""
    folder.mkdir(parents=True, exist_ok=False)
    env, conditions = environment(plan)
    obs, _ = env.reset(seed=SEED)
    if reset_diagnostics(env) != reset_group:
        env.close()
        raise ValueError('BLOCKED_RESET_IDENTITY')
    physics = PhysicsAudit(env)
    audit = controller_audit(env)
    states, frames, trace = [gt(env)], [], []
    try:
        for t, action in enumerate(actions):
            token = physics.token()
            row = observation(env, obs, norm.schema, plan['config']['vision']['camera'])
            frames.append({key: row[key] for key in ('rgb', 'depth_raw_mm', 'robot_state', 'intrinsic', 'cam2world')})
            command = np.asarray(action, np.float32).copy()
            audit.clear()
            obs, _, terminated, truncated, info = env.step(torch.from_numpy(command[None]).to(env.unwrapped.device))
            physics.stepped(token)
            validate_controller_command(command, audit)
            states.append(gt(env))
            trace.append(dict(t=t, action7=command.tolist(), controller={k:v.tolist() for k,v in audit.items()},
                              success=bool(info['success'].item()), terminated=bool(terminated.item()),
                              truncated=bool(truncated.item())))
            if terminated.item() or truncated.item() or bool(info['success'].item()):
                break
        if len(trace) != len(actions) or [x['t'] for x in trace if x['success']] != [136]:
            raise ValueError('expert replay failed its existing-success contract')
        row = observation(env, obs, norm.schema, plan['config']['vision']['camera'])
        frames.append({key: row[key] for key in ('rgb', 'depth_raw_mm', 'robot_state', 'intrinsic', 'cam2world')})
        put(folder/'states.json', states)
        put(folder/'trace.json', trace)
        save_npz(folder/'observations.npz', **{key:np.stack([f[key] for f in frames]) for key in frames[0]})
        put(folder/'summary.json', dict(status='COMPLETE', success=True, success_action_index=136,
                                        actions=len(trace), reset_group=reset_group, conditions=conditions,
                                        scope='current-runtime recorded-action replay; later states may differ from original HDF5'))
        return states, frames
    finally:
        physics.close()
        env.close()


def branch_input(row, masks, packet, graph, hidden, predicted, state):
    data = {key:row[key] for key in ('rgb', 'depth_raw_mm', 'robot_state', 'intrinsic', 'cam2world')}
    data.update(masks=masks.astype(np.uint8),
                graph_node_input=numpy(graph.node_input), graph_node_type=numpy(graph.node_type),
                graph_edge_index=numpy(graph.edge_index), graph_edge_attr=numpy(graph.edge_attr),
                graph_edge_type=numpy(graph.edge_type),
                object_id=np.asarray(packet['object_id']),
                hidden=numpy(hidden), predicted_action=predicted,
                **{'gt_'+name:np.asarray(state[name],np.float32) for name in FIELDS})
    return data


def trial(plan, method, branch, arm, actions, policy, norm, vision, reset_group, folder):
    from graph.integration_baseline.vision import packet_from_frame
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    env, conditions = environment(plan)
    obs, _ = env.reset(seed=SEED)
    if reset_diagnostics(env) != reset_group:
        env.close()
        raise ValueError('BLOCKED_RESET_IDENTITY')
    identity = method  # matches the Task08-E episode identity for this method
    vision.reset(identity)
    policy.raw_outputs.clear()
    stream = EpisodeStream(policy, norm)
    stream.reset()
    physics = PhysicsAudit(env)
    audit = controller_audit(env)
    device = next(policy.parameters()).device
    initial = gt(env)
    states, rows, masks_saved, hidden_saved, predicted_saved, applied_saved, after_rgb, trace = [initial], [], [], [], [], [], [], []
    limit = 250 if arm == 'MM' else branch + SHORT_HORIZON
    result = dict(status='RUNNING', method=method, branch=branch, arm=arm, seed=SEED,
                  reset_group=reset_group, scope='FULL_250' if arm=='MM' else 'SHORT_10',
                  success=False, first_success_step=None, steps=0, conditions=conditions)
    try:
        for t in range(limit):
            token = physics.token()
            if token[1] != t:
                raise ValueError('observation/control clock discontinuity')
            row = observation(env, obs, norm.schema, plan['config']['vision']['camera'])
            masks, _ = vision.process(row['rgb'], identity, t)
            frame = TrackedRGBDFrame(t, plan['config']['vision']['camera'], row['rgb'], row['depth'],
                row['intrinsic'], row['cam2world'], row['tcp_pose'], {0:masks[0],1:masks[1]}, t, t/20)
            packet, _ = packet_from_frame(frame, identity, keep_ratio=.5)
            packet['observation_provenance'] = dict(seed=SEED, timestep=t, future_observations_used=False,
                gt_used_as_student_input=False)
            save_packet(folder/f'packet_{t:04d}.npz', packet)
            graph = build_graph(packet, row['robot_ee_geometry'], **plan['config']['graph'])
            save_npz(folder/f'graph_{t:04d}.npz', **{name:numpy(getattr(graph,name)) for name in
                ('node_input','node_type','edge_index','edge_attr','edge_type')})
            with torch.inference_mode():
                command = stream.step(pack_graphs([graph]).to(device),
                    torch.from_numpy(row['robot_state'][None]).to(device), identity, t)
            predicted = numpy(command)[0]
            if not np.isfinite(predicted).all() or predicted.shape != (7,):
                raise ValueError('finite predicted Action7 required')
            hidden = numpy(stream.hidden)
            if t == branch:
                np.savez_compressed(folder/'branch_input.npz', **branch_input(row,masks,packet,graph,stream.hidden,predicted,states[-1]))
            expert = actions[t] if t < len(actions) else None
            if expert is None and (t < branch or t == branch and arm[0]=='E' or t>branch and arm[1]=='E'):
                raise ValueError('expert action sequence exhausted')
            applied, source = choose_action(method, arm, t, branch, predicted, expert)
            np.testing.assert_array_equal(flatten_state(robot_observation(env)[0],norm.schema),row['robot_state'])
            physics.unchanged(token)
            audit.clear()
            obs, _, terminated, truncated, info = env.step(torch.from_numpy(applied[None]).to(env.unwrapped.device))
            physics.stepped(token)
            validate_controller_command(applied, audit)
            after = gt(env)
            states.append(after)
            next_row = observation(env, obs, norm.schema, plan['config']['vision']['camera'])
            after_rgb.append(next_row['rgb'])
            rows.append({key:row[key] for key in ('rgb','depth_raw_mm','robot_state','intrinsic','cam2world')})
            masks_saved.append(masks)
            hidden_saved.append(hidden)
            predicted_saved.append(predicted)
            applied_saved.append(applied)
            success = bool(info['success'].item())
            trace.append(dict(t=t, source=source, predicted_action7=predicted.tolist(),
                encoded_action7=numpy(policy.raw_outputs[-1])[0].tolist() if torch.is_tensor(policy.raw_outputs[-1]) else np.asarray(policy.raw_outputs[-1])[0].tolist(),
                expert_action7=None if expert is None else expert.tolist(), applied_action7=applied.tolist(),
                controller={k:v.tolist() for k,v in audit.items()}, before=states[-2], after=after,
                point_counts=[int(np.sum(packet['object_id']==i)) for i in (0,1)],
                success=success, terminated=bool(terminated.item()), truncated=bool(truncated.item()),
                physics_before=token[0], physics_after=physics.count))
            result.update(steps=t+1, success=success)
            if success:
                result['first_success_step']=t+1
            if t % 25 == 0 or t == branch:
                print(f'{method} k={branch} {arm} t={t} grasp={after["grasp"]} success={success}',flush=True)
            if success or terminated.item() or truncated.item():
                break
        result.update(status='COMPLETE', terminated=bool(terminated.item()), truncated=bool(truncated.item()),
                      grasp=any(s['grasp'] for s in states[1:]),
                      first_grasp=next((i-1 for i,s in enumerate(states[1:],1) if s['grasp']),None),
                      bilateral_contact=any(all(np.linalg.norm(f)>=.5 for f in s['finger_forces']) for s in states[1:]),
                      lift_4cm=any(s['peg_pose'][2]-states[0]['peg_pose'][2]>=.04 for s in states[1:]),
                      max_peg_lift_m=max(s['peg_pose'][2]-states[0]['peg_pose'][2] for s in states[1:]))
        return result
    finally:
        physics.close()
        env.close()
        if trace:
            put(folder/'summary.json',result)
            put(folder/'states.json',states)
            put(folder/'trace.json',trace)
            save_npz(folder/'actions.npz', predicted=np.stack(predicted_saved), applied=np.stack(applied_saved),
                     hidden=np.stack(hidden_saved), timestep=np.arange(len(trace)))
            save_npz(folder/'observations.npz', **{key:np.stack([x[key] for x in rows]) for key in rows[0]},
                     masks=np.stack(masks_saved), after_rgb=np.stack(after_rgb))
            writer=cv2.VideoWriter(str(folder/'after_action_rgb.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(512,512))
            if not writer.isOpened():
                raise RuntimeError('MP4 encoder unavailable')
            for t,image in enumerate(after_rgb):
                output=cv2.resize(image,(512,512),interpolation=cv2.INTER_NEAREST)
                cv2.putText(output,f'{method} k={branch} {arm} after t={t}',(8,24),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                writer.write(cv2.cvtColor(output,cv2.COLOR_RGB2BGR))
            writer.release()
        stream.reset()
        vision.reset(None)


def difference(a,b,name):
    if a.shape != b.shape:
        return dict(pass_gate=False,reason='shape',a=list(a.shape),b=list(b.shape))
    if name in ('rgb','depth_raw_mm'):
        tol=GATE['rgb_lsb'] if name=='rgb' else GATE['depth_mm']
    elif name in ('masks','graph_node_type','graph_edge_index','graph_edge_type','object_id'):
        tol=0
    else:
        tol=GATE['float_atol']
    gap=np.abs(a.astype(np.float64)-b.astype(np.float64))
    maximum=float(gap.max(initial=0))
    return dict(pass_gate=maximum<=tol,max_abs=maximum,tolerance=tol)


def gate_pair(reference,candidate,physical_reference=None):
    with np.load(reference/'branch_input.npz',allow_pickle=False) as x, np.load(candidate/'branch_input.npz',allow_pickle=False) as y:
        tests={key:difference(x[key],y[key],key) for key in x.files}
        if physical_reference is not None:
            for key in FIELDS:
                v=np.asarray(physical_reference[key],np.float32)
                tests['reference_gt_'+key]=difference(x['gt_'+key],v,'gt_'+key)
                tests['reference_gt_'+key]['tolerance']=GATE['physical_atol']
                tests['reference_gt_'+key]['pass_gate']=tests['reference_gt_'+key]['max_abs']<=GATE['physical_atol']
        return dict(pass_gate=all(v['pass_gate'] for v in tests.values()),fields=tests,
                    reference=str(reference),candidate=str(candidate))


def vector(state):
    return np.asarray(state['ee_pose'][:3],np.float64)-np.asarray(state['peg_pose'][:3],np.float64)


def branch_metrics(run,method,branch,reference_states):
    base=run/'branches'/method/f'k{branch:03d}'
    pairs={arm:read(base/arm/'states.json') for arm in ('EM','ME','MM')}
    # State index k is pre-branch; state index k+1 is after the branch action.
    def distance(a,b,index):
        return float(np.linalg.norm(vector(a[index])-vector(b[index])))
    end=branch+SHORT_HORIZON
    fixed=distance(pairs['ME'],reference_states,end)
    feedback=distance(pairs['MM'],pairs['EM'],end)
    first=distance(pairs['MM'],pairs['EM'],branch+1)
    result=dict(method=method,branch=branch,horizon=SHORT_HORIZON,
                first_action_tcp_peg_difference_m=first,
                fixed_expert_suffix_difference_m=fixed,
                policy_feedback_difference_m=feedback,
                feedback_minus_fixed_m=feedback-fixed,
                feedback_amplification_supported=bool(feedback>=fixed+.01 and feedback>=2*fixed),
                full_prefix_rescue=read(base/'MM'/'summary.json'))
    put(base/'comparison.json',result)
    return result


def run(plan_path):
    plan_path=Path(plan_path).resolve()
    plan=load_plan(plan_path)
    if plan['config']['device']!='cuda' or [len(plan['splits'][s]) for s in ('train','validation','test')]!=[96,12,12]:
        raise ValueError('frozen Full protocol/device mismatch')
    if SEED not in plan['splits']['train'] or SEED in plan['splits']['validation']+plan['splits']['test']:
        raise ValueError('seed 30003 not Train-only')
    if vision_runtime_identity()!=plan['reference_provenance']['vision_runtime']:
        raise ValueError('Vision runtime mismatch')
    original=read(plan_path.parent/'status.json')
    if original['status']=='RUNNING' or any(k.startswith('final_') for k in original['stages']):
        raise ValueError('original Full job active or Final state changed')
    ds=DirectDataset(plan['dataset_directory'])
    episode=ds.episodes[SEED]
    if episode['sha256']!=EXPERT_H5_SHA or episode['actions']!=137 or not episode['success']:
        raise ValueError('Expert episode identity mismatch')
    data=load_episode(ds.run/episode['path'])
    actions=data['actions']
    reset_group=episode['reset_group']
    receipts={m:Path(original['stages']['full_'+m]['directory'])/'best.json' for m in 'ABC'}
    for m in 'ABC':
        if original['stages']['full_'+m]['status']!='COMPLETE' or read(receipts[m])['epoch']!={'A':35,'B':48,'C':38}[m]:
            raise ValueError('Best validation checkpoint selection mismatch')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with ExperimentLogger(team='integration',model='task08f_error_accumulation_branch',seed=SEED,
        config=dict(plan=str(plan_path),branches=BRANCHES,short_horizon=SHORT_HORIZON,
                    arms=['EE_reference','EM','ME','MM'],fixed_best={m:read(receipts[m]) for m in 'ABC'},
                    no_training=True),root=plan['experiments_root'],job_type='evaluate') as logger:
        root=logger.directory
        print('RUN_ID',logger.run_id,flush=True)
        put(root/'protocol.json',dict(plan=str(plan_path),plan_sha256=sha(plan_path),
            original_full_status=original['status'],hdf5_sha256=EXPERT_H5_SHA,
            split='Train96',seed=SEED,branches=BRANCHES,short_horizon=SHORT_HORIZON,
            gate=GATE,checkpoints={m:read(receipts[m]) for m in 'ABC'},
            interpretation='only t0 is original saved initial state; later branch states condition on current-runtime expert replay'))
        shutil.copy2(__file__,root/'source_snapshot_branch.py')
        source_diff=subprocess.run(['git','diff','--no-index','--','/dev/null',str(Path(__file__).resolve())],capture_output=True,text=True)
        (root/'implementation.diff').write_text(source_diff.stdout)
        norm=load_policy(plan,plan_path,receipts['A'],'A')[1]
        states,_=reference_replay(plan,actions,norm,reset_group,root/'reference_expert')
        status=dict(status='RUNNING',run_id=logger.run_id,completed=[],invalid=[])
        atomic(root/'status.json',status)
        from visual.runtime.grounding import GDino
        from visual.runtime.tracking import SAM2Tracker
        results=[]
        for m in 'ABC':
            policy,norm=load_policy(plan,plan_path,receipts[m],m)
            vision=OnlineVisionAdapter(GDino(device='cuda'),SAM2Tracker(device='cuda'))
            try:
                for k in BRANCHES:
                    base=root/'branches'/m/f'k{k:03d}'
                    for arm in ('EM','ME'):
                        trial(plan,m,k,arm,actions,policy,norm,vision,reset_group,base/arm)
                        status['completed'].append(f'{m}:{k}:{arm}')
                        atomic(root/'status.json',status)
                    gate=gate_pair(base/'EM',base/'ME',states[k])
                    put(base/'prefix_gate_em_me.json',gate)
                    if not gate['pass_gate']:
                        status['invalid'].append(f'{m}:{k}:EM_ME')
                        atomic(root/'status.json',status)
                        continue
                    trial(plan,m,k,'MM',actions,policy,norm,vision,reset_group,base/'MM')
                    status['completed'].append(f'{m}:{k}:MM')
                    gate=gate_pair(base/'EM',base/'MM',states[k])
                    put(base/'prefix_gate_em_mm.json',gate)
                    if not gate['pass_gate']:
                        status['invalid'].append(f'{m}:{k}:EM_MM')
                    else:
                        results.append(branch_metrics(root,m,k,states))
                    atomic(root/'status.json',status)
            finally:
                vision.close()
                del vision,policy
                torch.cuda.empty_cache()
        put(root/'results.json',results)
        status.update(status='COMPLETE',valid_comparisons=len(results))
        atomic(root/'status.json',status)
        logger.log(dict(valid_comparisons=len(results),invalid_comparisons=len(status['invalid'])))
        logger.link_output(root/'results.json',kind='metrics')
        return root


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args()
    print(run(a.plan),flush=True)
