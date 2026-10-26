"""Task07-F: three fresh, autonomous, fixed-checkpoint closed-loop episodes."""
import argparse
from dataclasses import asdict
import inspect
import os
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
import torch
from transforms3d.quaternions import quat2mat

from graph.integration_baseline.autonomous_reach_diagnosis import geometry, relative_pose, file_inventory
from graph.integration_baseline.autonomous_reach_report import metric_at, load_arrays
from graph.integration_baseline.direct_runtime import read, sha
from graph.integration_baseline.grasp_intervention import atomic, digest
from graph.integration_baseline.graph import pack_graphs
from graph.integration_baseline.io import save_npz, save_packet
from graph.integration_baseline.model import EpisodeStream
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.rollout_checkpoint import load_bundle, required_sources
from graph.integration_baseline.rollout_stop_clock import (PhysicsAudit, observation,
    controller_audit, validate_controller_command, controller_semantics, infer, numpy, reset_diagnostics)
from graph.integration_baseline.vision_oracle_adapter import (
    POLICIES, SCOPE, CurrentSegmentation, rgbd_hash, build_input, OwnedHistory)


def validate_config(c):
    if (c['scope'] != SCOPE or c['seed'] != 41005 or c['conditions'] != ['A', 'B', 'C'] or
        c['horizon'] != 250 or c['keep_ratio'] != .5 or c['camera'] != 'base_camera' or
        c['graph'] != dict(k=4, M=16, block_rows=64) or
        c['checkpoint_sha256'] != '3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776'):
        raise ValueError('fixed oracle protocol changed')


def current_gt(env, row, episode, t, physics):
    """Read existing textures. No rendering, capture, reset, or physics call."""
    token = physics.token()
    texture = env.unwrapped._sensors['base_camera'].get_obs(
        rgb=True, depth=True, position=False, segmentation=True)
    np.testing.assert_array_equal(numpy(texture['rgb'])[0], row['rgb'])
    np.testing.assert_array_equal(numpy(texture['depth'])[0, ..., 0], row['depth_raw_mm'])
    seg = numpy(texture['segmentation'])[0, ..., 0].copy()
    ids = np.array([env.unwrapped.peg._objs[0].per_scene_id,
                    env.unwrapped.box._objs[0].per_scene_id], np.int64)
    if ids[0] == ids[1] or np.any(ids <= 0) or seg.shape != row['depth'].shape:
        raise ValueError('BLOCKED: ambiguous actor ID or pixel alignment')
    physics.unchanged(token)
    return CurrentSegmentation(episode, t, rgbd_hash(row), seg[None] == ids[:, None, None]), seg, ids


def collision_meshes(env):
    # Diagnostic-only meshes; these never enter build_input or model.step.
    import sapien.physx as physx
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    import trimesh
    result = {}
    for name in ('peg', 'box'):
        entity = getattr(env.unwrapped, name)._objs[0]
        meshes = [m for comp in entity.components if isinstance(comp, physx.PhysxRigidBaseComponent)
                  for m in get_component_meshes(comp)]
        result[name] = trimesh.util.concatenate(meshes)
    return result


def surface_distances(camera_points, row, state, name, meshes):
    import trimesh
    world = camera_points @ row['cam2world'][:3, :3].T + row['cam2world'][:3, 3]
    pose = np.asarray(state[name+'_pose'])
    local = (world-pose[:3]) @ quat2mat(pose[3:])
    return trimesh.proximity.closest_point_naive(meshes[name], local)[1]*1000


def mapping_check(row, gt, state, meshes):
    from visual.runtime.geometry import build_organized_geometry
    from visual.runtime.rgbd import TrackedRGBDFrame
    frame = TrackedRGBDFrame(0, 'base_camera', row['rgb'], row['depth'], row['intrinsic'],
        row['cam2world'], row['tcp_pose'], {0:gt.masks[0], 1:gt.masks[1]}, 0, 0.)
    geom = build_organized_geometry(frame)
    out = {}
    for i, name in enumerate(('peg', 'box')):
        points = geom.xyz[geom.observed_mask(i)]
        if not len(points):
            raise ValueError('BLOCKED: no visible object support at mapping preflight')
        d = surface_distances(points, row, state, name, meshes)
        # Integer-mm depth readback plus pixel-centre projection: <=3mm p95.
        if np.quantile(d, .95) > 3:
            raise ValueError('BLOCKED: GT actor mapping/calibration fails surface audit')
        out[name] = dict(visible_valid_pixels=len(points), surface_mean_mm=float(d.mean()),
                         surface_p95_mm=float(np.quantile(d,.95)), surface_max_mm=float(d.max()))
    return dict(status='PASS', rgb_depth_exact_current_texture=True, physics_delta=0, objects=out)


def initial_signature(env, row, state):
    keys = ('qpos','qvel','ee_pose','left_finger_pose','right_finger_pose',
            'left_fingertip_pos','right_fingertip_pos','peg_pose','box_pose')
    return dict(reset_identity=reset_diagnostics(env), rgbd_calibration_hash=rgbd_hash(row),
        physical_hash=digest(*[state[k] for k in keys]), robot76_hash=digest(row['robot_state']),
        fields={k:state[k] for k in keys})


def assert_initial(reference, candidate):
    for k in ('reset_identity','rgbd_calibration_hash','physical_hash','robot76_hash'):
        if reference[k] != candidate[k]:
            raise ValueError('BLOCKED: initial reset mismatch '+k)


def make_env(provenance):
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.utils.registration import TimeLimitWrapper
    from recurrent.common import versions
    env = TimeLimitWrapper(make_peg_insertion_env(obs_mode='rgb+depth',
        control_mode='pd_ee_delta_pose', device='cpu').unwrapped, 250)
    task = env.unwrapped
    actual = dict(control_freq=task.control_freq, physics_freq=task.sim_freq,
        physics_substeps=task._sim_steps_per_control, horizon=env._max_episode_steps,
        robot=task.agent.uid, versions=versions(), backend=str(task.backend),
        controller_config={k:asdict(v.config) for k,v in task.agent.controller.controllers.items()})
    trained = provenance['conditions']
    for k in ('control_freq','physics_freq','horizon','robot'):
        if actual[k] != trained[k]:
            env.close(); raise ValueError('environment mismatch '+k)
    for k in ('mani_skill','sapien','gymnasium'):
        if actual['versions'][k] != trained['versions'][k]:
            env.close(); raise ValueError('package mismatch '+k)
    if actual['physics_substeps'] != 5 or controller_semantics(actual['controller_config']) != controller_semantics(trained['controller_config']):
        env.close(); raise ValueError('physics/controller mismatch')
    return env, actual


def run_trial(env, condition, c, model, norm, vision, folder, initial):
    from recurrent.expert import ScriptedExpert
    folder.mkdir(parents=True, exist_ok=False)
    identity = 'task07f_'+condition+'_41005'
    obs, _ = env.reset(seed=41005)
    physics = PhysicsAudit(env); audit = controller_audit(env)
    stream = EpisodeStream(model, norm); history = OwnedHistory(stream, identity)
    vision.reset(identity)
    state = geometry(env, False)
    row = observation(env, obs, norm.schema, 'base_camera')
    signature = initial_signature(env, row, state); assert_initial(initial, signature)
    # Expert instantiated only to compute the fixed diagnostic reference. Never act().
    reference = ScriptedExpert(env).grasp
    trial = dict(condition=condition, input_policy=POLICIES[condition], scope=SCOPE,
        initial=signature, grasp_reference=np.r_[reference.p,reference.q],
        reference_relative=relative_pose(state['peg_pose'],np.r_[reference.p,reference.q]),
        hidden_reset_events=[dict(timestep=0, reason='fresh_independent_reset')])
    meshes = collision_meshes(env)
    for name, mesh in meshes.items():
        save_npz(folder/(name+'_diagnostic_mesh.npz'), vertices=mesh.vertices, faces=mesh.faces)
    rows=[]; trace=[]; masks=[]; segmentations=[]; hidden=[]
    writer = cv2.VideoWriter(str(folder/'rollout_after_action.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),20,(512,512))
    if not writer.isOpened(): raise RuntimeError('MP4 writer unavailable')
    start=perf_counter(); first_close=None
    try:
        for t in range(250):
            before=physics.token(); history.before(t)
            if before != (t*5,t): raise ValueError('clock discontinuity')
            row=observation(env,obs,norm.schema,'base_camera'); rows.append(row)
            state=geometry(env,t>0)
            predicted,vision_time=vision.process(row['rgb'],identity,t)
            predicted=predicted.copy()  # tracker state retains its original prediction
            gt,seg,ids=current_gt(env,row,identity,t,physics)
            if t==0: trial['mapping_check']=mapping_check(row,gt,state,meshes)
            applied,packet,graph,build_time=build_input(row,predicted,gt,condition=condition,
                episode=identity,timestep=t,graph_config=c['graph'])
            save_packet(folder/f'packet_{t:04d}.npz',packet)
            save_npz(folder/f'graph_{t:04d}.npz',**{k:numpy(getattr(graph,k)) for k in
                ('node_input','node_type','edge_index','edge_attr','edge_type')},
                source_index=packet['provenance']['source_index'])
            command,timing=infer(stream,pack_graphs([graph]).to('cuda'),
                torch.from_numpy(row['robot_state'][None]).to('cuda'),identity,t)
            history.after(t); action=numpy(command)[0].copy(); hidden.append(numpy(stream.hidden).copy())
            if first_close is None and action[6]<0: first_close=t
            masks.append(np.stack([predicted,predicted&gt.masks,gt.masks,applied]))
            segmentations.append(seg)
            physics.unchanged(before)
            audit.clear()
            obs,_,terminated,truncated,info=env.step(torch.from_numpy(action[None]).to(env.unwrapped.device))
            physics.stepped(before);validate_controller_command(action,audit)
            after=geometry(env,True)
            r=dict(timestep=t,condition=condition,input_policy=POLICIES[condition],
                before=state,after=after,metric=metric_at(state,trial),action=action,
                controller=dict(audit),physics_steps_before=before[0],physics_steps_after=physics.count,
                simulation_time_before=t/20,simulation_time_after=(t+1)/20,
                inference_physics_steps=0,segmentation_physics_steps=0,
                presence=[bool(np.any(packet['object_id']==i)) for i in (0,1)],
                point_counts=[int(np.sum(packet['object_id']==i)) for i in (0,1)],
                mask_hash=digest(applied),packet_mask_hash=packet['oracle_audit']['applied_mask_hash'],
                hidden_norm=float(np.linalg.norm(hidden[-1])),vision_timing=vision_time,
                geometry_timing=build_time,model_timing=timing,
                success=bool(info['success'].item()),terminated=bool(terminated.item()),truncated=bool(truncated.item()))
            trace.append(r)
            rgb=numpy(obs['sensor_data']['base_camera']['rgb'])[0]
            image=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_NEAREST)
            cv2.putText(image,f'{condition} {POLICIES[condition]} AFTER action t={t}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cv2.putText(image,f'first close={first_close} grasp={after["grasp"]}',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            if t%25==0 or t==first_close:
                print(f'FRAME {condition} t={t} close={first_close} error_mm={r["metric"]["reference_position_error_mm"]:.3f} grasp={after["grasp"]}',flush=True)
            if terminated.item() or truncated.item():break
        rows.append(observation(env,obs,norm.schema,'base_camera'))
        first_grasp=next((r['timestep'] for r in trace if r['after']['grasp']),None)
        baseline_z=float(state['peg_pose'][2]) if not trace else float(trace[0]['before']['peg_pose'][2])
        trial.update(status='COMPLETE',steps=len(trace),physics_steps=physics.count,hidden_steps=history.next_t,
            first_close=first_close,close_event_status='OBSERVED' if first_close is not None else 'NOT_OBSERVED',
            first_grasp_after_action=first_grasp,
            first_bilateral_force_after_action=next((r['timestep'] for r in trace if r['after']['bilateral_force']),None),
            first_grasp_loss_after_action=next((r['timestep'] for r in trace if first_grasp is not None and r['timestep']>first_grasp and not r['after']['grasp']),None),
            first_success_step=next((r['timestep']+1 for r in trace if r['success']),None),
            success=any(r['success'] for r in trace),grasp_observed=first_grasp is not None,
            peg_lift_observed=any(r['after']['peg_pose'][2]-baseline_z>=c['lift_observation_threshold_m'] for r in trace),
            max_peg_lift_mm=max((r['after']['peg_pose'][2]-baseline_z)*1000 for r in trace),
            close_geometry=trace[first_close]['metric'] if first_close is not None else None,
            minimum_target_error_before_close_mm=min(r['metric']['reference_position_error_mm'] for r in (trace[:first_close] if first_close is not None else trace)) if first_close!=0 else None,
            wall_seconds=perf_counter()-start,policy_success_rate=None,
            actor_ids=ids)
        atomic(folder/'trial.json',trial)
        return trial
    finally:
        writer.release();physics.close()
        atomic(folder/'trace.json',trace)
        if rows:save_npz(folder/'observations.npz',**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        if masks:save_npz(folder/'masks.npz',masks=np.stack(masks),axis_names=np.array(['predicted','cleaned','gt','applied']))
        if segmentations:save_npz(folder/'gt_segmentation_oracle_audit.npz',segmentation=np.stack(segmentations),actor_ids=ids)
        if hidden:save_npz(folder/'hidden.npz',hidden=np.stack(hidden),timestep=np.arange(len(hidden)),episode_identity=np.array(identity))
        if trace:save_npz(folder/'actions.npz',actions=np.stack([r['action'] for r in trace]),controller_input=np.stack([r['controller']['controller_input'] for r in trace]))
        atomic(folder/'vision_initialization.json',dict(found=vision.found,detector_calls=vision.detector_calls))


def probes(run,model,norm,c):
    """Same A observation and PRE-observation A hidden. Model.step only."""
    folder=run/'trials/A'; obs=load_arrays(folder/'observations.npz')
    masks=load_arrays(folder/'masks.npz')['masks'];hidden=load_arrays(folder/'hidden.npz')['hidden']
    results=[];dest=run/'probes';dest.mkdir()
    with torch.inference_mode():
        for t in c['probe_timesteps']:
            row={k:v[t] for k,v in obs.items()}; identity='read_only_A_probe'
            gt=CurrentSegmentation(identity,t,rgbd_hash(row),masks[t,2])
            h=None if t==0 else torch.from_numpy(hidden[t-1].copy()).to('cuda')
            actions=[]
            for condition in POLICIES:
                applied,packet,graph,_=build_input(row,masks[t,0],gt,condition=condition,
                    episode=identity,timestep=t,graph_config=c['graph'])
                action,hout=model.step(pack_graphs([graph]).to('cuda'),
                    torch.from_numpy(row['robot_state'][None]).to('cuda'),norm,
                    None if h is None else h.clone())
                actions.append(numpy(action)[0])
                save_packet(dest/f'{condition}_packet_{t:04d}.npz',packet)
                save_npz(dest/f'{condition}_graph_{t:04d}.npz',**{k:numpy(getattr(graph,k)) for k in
                    ('node_input','node_type','edge_index','edge_attr','edge_type')})
            np.testing.assert_allclose(actions[0],load_arrays(folder/'actions.npz')['actions'][t],atol=c['atol'],rtol=c['rtol'])
            results.append(dict(timestep=t,actions=np.stack(actions),delta_B_A=np.asarray(actions[1])-actions[0],delta_C_A=np.asarray(actions[2])-actions[0]))
            save_npz(dest/f'probe_{t:04d}.npz',actions=np.stack(actions),
                pre_observation_hidden=np.zeros_like(hidden[0]) if t==0 else hidden[t-1],
                hidden_is_initial_zero=np.array(t==0),robot76=row['robot_state'])
    atomic(run/'one_step_probes.json',dict(status='COMPLETE',scope='READ_ONLY_NOT_CLOSED_LOOP',
        no_actions_applied=True,no_live_hidden_modified=True,frames=results))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint-manifest','original-episode','prior-run','output-root'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();c=read(args.config);validate_config(c)
    os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    torch.set_num_threads(1);torch.manual_seed(41005);np.random.seed(41005)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    from utils.experiment_logger import ExperimentLogger
    with ExperimentLogger(team='integration',model='vision_oracle_reach_ablation',seed=41005,
        job_type='evaluate',config=c,root=args.output_root) as logger:
        run=logger.directory;print('RUN_DIR='+str(run),flush=True)
        report=dict(run_id=logger.run_id,scope=SCOPE,status='RUNNING',protocol=c,
            trials={k:dict(status='NOT_EXECUTED',input_policy=v) for k,v in POLICIES.items()},
            policy_success_rate=None)
        atomic(run/'results.json',report)
        vision=None;before={};model=norm=None
        try:
            if torch.cuda.mem_get_info()[0]<c['minimum_cuda_free_bytes']:
                raise RuntimeError('BLOCKED_RESOURCE: existing processes protected')
            torch.cuda.set_per_process_memory_fraction(c['cuda_memory_fraction'])
            model,norm,provenance=load_bundle(args.checkpoint_manifest,device='cuda')
            if provenance['manifest']['files']['checkpoint']['sha256']!=c['checkpoint_sha256']:
                raise ValueError('checkpoint differs')
            atomic(run/'checkpoint_provenance.json',provenance)
            protected=[Path(x) for x in required_sources()]
            for root in (args.checkpoint_manifest.parent,args.original_episode,args.prior_run,
                         Path('/workspace/graph-recurrent-VLA/experiments/2be5e62357a44b14a5651a1a346edc7a')):
                protected += [x for x in root.rglob('*') if x.is_file()]
            before=file_inventory(protected);atomic(run/'protected_before.json',before)
            weights=digest(*[numpy(x) for x in model.parameters()]);normalizer=digest(numpy(norm.stats.mean),numpy(norm.stats.std))
            report['model_hash_before']=weights;report['normalizer_hash_before']=normalizer
            # Capture new implementation and the actual imported dependencies.
            paths=set(protected[:len(required_sources())]) | set(Path(__file__).parent.glob('vision_oracle*.py')) | {args.config}
            paths |= {Path(inspect.getfile(geometry)),Path(inspect.getfile(metric_at)),Path(inspect.getfile(atomic))}
            sources={}
            for src in paths:
                if src.is_file():
                    rel=src.relative_to(Path.cwd()) if src.is_relative_to(Path.cwd()) else Path(src.name)
                    dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
                    sources[str(src)]=sha(src)
            atomic(run/'source_hashes.json',sources);shutil.copy2(args.config,run/'protocol.json')
            # Three independent reset preflights, before any policy action.
            initial=None;preflights={}
            for condition in POLICIES:
                env,actual=make_env(provenance)
                try:
                    obs,_=env.reset(seed=41005);physics=PhysicsAudit(env)
                    try:
                        row=observation(env,obs,norm.schema,'base_camera');state=geometry(env,False)
                        gt,seg,ids=current_gt(env,row,'preflight_'+condition,0,physics)
                        check=mapping_check(row,gt,state,collision_meshes(env))
                        sig=initial_signature(env,row,state)
                        if initial is None:initial=sig
                        else:assert_initial(initial,sig)
                        preflights[condition]=dict(initial=sig,mapping=check,actor_ids=ids)
                    finally:physics.close()
                finally:env.close()
            atomic(run/'reset_preflight.json',dict(status='PASS',conditions=preflights))
            atomic(run/'executed_conditions.json',actual)
            # Stored same-frame fixture: real sampling/features/topology; negative reuse test.
            from graph.integration_baseline.tests.test_vision_oracle_reach import fixture_contract
            atomic(run/'fixture_verification.json',fixture_contract(args.prior_run/'trials/C',run/'fixture',require_saved_exact=True))
            from visual.runtime.grounding import GDino
            from visual.runtime.tracking import SAM2Tracker
            vision=OnlineVisionAdapter(GDino(device='cuda'),SAM2Tracker(device='cuda'))
            for condition in POLICIES:
                env,_=make_env(provenance)
                try:
                    result=run_trial(env,condition,c,model,norm,vision,run/'trials'/condition,initial)
                finally:env.close()
                report['trials'][condition]=result;atomic(run/'results.json',report)
                logger.log(dict(success=int(result['success']),grasp=int(result['grasp_observed'])),prefix=condition)
            probes(run,model,norm,c)
            report['status']='COMPLETE'
        except Exception as exc:
            report.update(status='BLOCKED',error_type=type(exc).__name__,error=str(exc))
            atomic(run/'results.json',report)
            raise
        finally:
            if vision is not None:vision.close()
            if model is not None:
                report['model_hash_after']=digest(*[numpy(x) for x in model.parameters()])
                report['normalizer_hash_after']=digest(numpy(norm.stats.mean),numpy(norm.stats.std))
                report['weights_and_normalizer_unchanged']=(report.get('model_hash_before')==report['model_hash_after'] and report.get('normalizer_hash_before')==report['normalizer_hash_after'])
            changed=[path for path,h in before.items() if not Path(path).is_file() or sha(Path(path))!=h]
            atomic(run/'protection.json',dict(status='PASS' if not changed else 'FAIL',protected_file_count=len(before),changed=changed))
            atomic(run/'results.json',report)
            logger.record(diagnostic_status=report['status'],scope=SCOPE,source_run_id='2be5e62357a44b14a5651a1a346edc7a')
            for path in sorted(run.rglob('*')):
                if path.is_file() and not path.name.endswith('.tracking.json') and path.name not in ('metadata.json','metrics.json','manifest.json','config.yaml'):
                    logger.link_output(path,kind='rollouts' if path.suffix=='.mp4' else 'trajectory' if path.suffix=='.npz' else 'reports')
        print('RESULT='+report['status'],flush=True)


if __name__=='__main__':main()
