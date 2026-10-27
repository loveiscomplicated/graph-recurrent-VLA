"""Opt-in Task09-A physical geometry interventions; never applies student actions.

Only simulator condition setup and audit sidecars see object GT. Student input
uses the unchanged observation whitelist, causal RGB tracker and packet builder.
"""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from envs.maniskill import make_peg_insertion_env, get_gt_state
from graph.integration_baseline.bc08_codec import controller_commands
from graph.integration_baseline.bc08_eval import validate_checkpoint_plan
from graph.integration_baseline.bc08_train import load_checkpoint, load_plan, load_normalizer
from graph.integration_baseline.direct_dataset import DirectDataset, load_episode
from graph.integration_baseline.direct_runtime import read, sha
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import save_npz, save_packet, write_json
from graph.integration_baseline.model import IntegrationPolicy
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.point_resampling_adapter import permute_graph
from graph.integration_baseline.rollout_checkpoint import vision_runtime_identity
from graph.integration_baseline.rollout_stop_clock import (PhysicsAudit, observation, numpy,
    reset_diagnostics, controller_audit, validate_controller_command, controller_semantics)
from recurrent.expert import ScriptedExpert
from utils.experiment_logger import ExperimentLogger

SEED = 30003
FULL_PLAN = Path('/workspace/task08/experiments/3ed77d8b64884f879b9a628e34a9426a/plan.json')
SINGLE_PLAN = Path('/workspace/task08b/experiments/60468642a39545318ebf0256727d7a08/plan.json')
B_PLAN = Path('/workspace/task08d/plan.json')
EXPECTED_FULL = dict(A='409a9456740193c3a46337c4e172dac7d64dc22c45250c365d3879d4fb337b5e',
    B='f094f3cc85f8013c37e70d6043a7035071f5733b7db04371207c31fbce7b4e02',
    C='e615ec3ded96b0242004e08ee13273b782ba4af1032c8a84f205144f4ab8d177')
NORMALIZER = '4e607d9a0af71ca65134f71b187946c2c47640bbe9f5cf81fb4942903ec52446'
TOL = dict(float_atol=1e-6, float_rtol=1e-5, robot_exact=True, rgb_lsb=2, depth_mm=1,
    expert_translation_discriminative_mm=.1, model_translation_noise_mm=.001,
    rotation_noise_rad=1e-6, gripper_noise_mm=.001, contact_tolerance_m=.0002,
    geometry_min_centroid_response_mm=1., geometry_min_purity=.5)


def protocol():
    primary = [dict(name='baseline', object='peg', xyz_mm=[0,0,0], yaw_deg=0, group='baseline')]
    for axis in (0,1):
        for mm in (-10,-5,5,10):
            xyz=[0,0,0]; xyz[axis]=mm
            primary.append(dict(name=f'peg_{"xy"[axis]}{mm:+d}',object='peg',xyz_mm=xyz,yaw_deg=0,group='primary'))
    primary.append(dict(name='baseline_repeat',object='peg',xyz_mm=[0,0,0],yaw_deg=0,group='repeat'))
    for axis in (0,1):
        for mm in (-10,10):
            xyz=[0,0,0];xyz[axis]=mm
            primary.append(dict(name=f'box_{"xy"[axis]}{mm:+d}',object='box',xyz_mm=xyz,yaw_deg=0,group='box_control'))
    extra=[]
    for axis in (0,1):
        for mm in (-20,20):
            xyz=[0,0,0];xyz[axis]=mm
            extra.append(dict(name=f'peg_{"xy"[axis]}{mm:+d}',object='peg',xyz_mm=xyz,yaw_deg=0,group='additional'))
    for deg in (-5,5):
        extra.append(dict(name=f'peg_yaw{deg:+d}',object='peg',xyz_mm=[0,0,0],yaw_deg=deg,group='additional'))
    return dict(seed=SEED,timesteps=[0,10,25],primary=primary,additional=extra,tolerances=TOL,
        extra_trigger='at a given t: any valid primary Expert response <=0.1mm OR all primary translation magnitudes saturated; run all six additional candidates at that t',
        scope='SAME_HISTORY_ONE_STEP_GEOMETRY_INTERVENTION',
        no_student_step=True,no_training=True,student_input='current RGB-D / predicted masks / Robot76 only',
        expert_reference='fresh geometry target, baseline phase and counter; current-stage counterfactual, not a fresh demonstration',
        physical_validation='all actor and robot convex collision meshes SAT; no scene.step; cached PhysX contacts are recorded but not fresh after teleport',
        decision='No interpretation after failed repeat/permutation/physical/hidden contracts. Invalid references excluded from alignment.')


def digest(a):
    a=np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def environment(plan):
    """Explicit compatibility adapter, with Gymnasium drift recorded, not hidden."""
    from mani_skill.utils.registration import TimeLimitWrapper
    from recurrent.common import versions
    env=TimeLimitWrapper(make_peg_insertion_env(obs_mode='rgb+depth',
        control_mode='pd_ee_delta_pose',device='cpu').unwrapped,250)
    t=env.unwrapped;ref=plan['reference_provenance']['conditions']
    actual=dict(versions=versions(),control_freq=int(t.control_freq),physics_freq=int(t.sim_freq),
        horizon=env._max_episode_steps,robot=t.agent.uid,backend=str(t.backend),
        controller_config={k:asdict(v.config) for k,v in t.agent.controller.controllers.items()})
    for k in ('control_freq','physics_freq','horizon','robot'):
        assert actual[k]==ref[k],k
    for k in ('mani_skill','sapien'):
        assert actual['versions'][k]==ref['versions'][k],k
    assert controller_semantics(actual['controller_config'])==controller_semantics(ref['controller_config'])
    assert t._sim_steps_per_control==5 and np.isclose(t.scene.px.timestep,.01)
    actual['gymnasium_drift']=dict(reference=ref['versions']['gymnasium'],current=actual['versions']['gymnasium'],
        acceptance='reset identity + same runtime paired states + exact historical prefix physical comparison + installed controller audit')
    return env,actual


def load_models():
    plans={p:load_plan(p) for p in (FULL_PLAN,SINGLE_PLAN,B_PLAN)}
    full=plans[FULL_PLAN];status=read(FULL_PLAN.parent/'status.json')
    assert status['status']=='ERROR' and not any(k.startswith('final_') for k in status['stages'])
    receipts={f'Full_{m}':Path(status['stages']['full_'+m]['directory'])/'best.json' for m in 'ABC'}
    receipts.update(Single_A=Path('/workspace/task08b/experiments/cac57daccc2044a8a8e9de4a5e8f7adc/best.json'),
        Single_B=Path('/workspace/task08d/experiments/4fb8f19b0b3f4624a26bbca4df59a529/best.json'))
    taskb=read(SINGLE_PLAN.parent/'task08b_status.json')
    # A/C selected receipts are the training result's pre-rollout Best pointer.
    receipts['Single_C']=Path(taskb['selected']['C']['path']).parents[1]/'best.json'
    assert read(receipts['Single_A'])==taskb['selected']['A']
    assert read(receipts['Single_C'])==taskb['selected']['C']
    models={};identities={};protected={}
    for label,rpath in receipts.items():
        method=label[-1];phase='full' if label.startswith('Full') else 'overfit'
        pp=FULL_PLAN if phase=='full' else B_PLAN if method=='B' else SINGLE_PLAN
        plan=plans[pp];receipt=read(rpath);cp,codec=load_checkpoint(rpath)
        validate_checkpoint_plan(cp,codec,receipt,plan,pp,phase,method)
        assert cp['identity']['normalizer_sha256']==NORMALIZER
        assert cp['identity']['initial_sha256']==full['initial_weight_identity']
        if phase=='full':assert receipt['sha256']==EXPECTED_FULL[method] and receipt['epoch']==dict(A=35,B=48,C=38)[method]
        else:assert receipt['global_step']==20000 and cp['identity']['train_ids']==[SEED]
        model=IntegrationPolicy(cp['normalizer']['schema'],**plan['config']['model']).cuda().eval()
        model.load_state_dict(cp['model'],strict=True)
        norm=load_normalizer(cp['normalizer']);assert norm.fitted_episode_ids==sorted(full['splits']['train'])
        models[label]=(model,norm,codec)
        identities[label]=dict(receipt=receipt,receipt_path=str(rpath),plan=str(pp),plan_sha256=sha(pp),identity=cp['identity'],codec=codec.state)
        for p in (rpath,Path(receipt['path']),pp):protected[str(p)]=sha(p)
    for plan in plans.values():
        protected.update(plan['input_hashes']);protected.update(plan['source_hashes'])
    protected[str(FULL_PLAN.parent/'status.json')]=sha(FULL_PLAN.parent/'status.json')
    assert vision_runtime_identity()==full['reference_provenance']['vision_runtime']
    return full,models,identities,protected


def physical_state(env):
    s=get_gt_state(env);t=env.unwrapped
    fields=('qpos','qvel','ee_pose','left_finger_pose','right_finger_pose','left_fingertip_pos',
        'right_fingertip_pos','peg_pose','box_pose','hole_pose')
    out={k:numpy(getattr(s,k))[0] for k in fields}
    out.update(goal_pose=numpy(t.goal_pose.raw_pose)[0],peg_linear=numpy(t.peg.linear_velocity)[0],
        peg_angular=numpy(t.peg.angular_velocity)[0],box_linear=numpy(t.box.linear_velocity)[0],
        box_angular=numpy(t.box.angular_velocity)[0])
    return out


def collision_meshes(env):
    """Read installed component meshes including shape-local transforms."""
    import sapien.physx as physx
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    t=env.unwrapped;out={}
    entries=[(name,a._objs[0]) for name,a in t.scene.actors.items()]
    entries += [('robot:'+a.name,a._objs[0].entity) for a in t.agent.robot.get_links()]
    for name,entity in entries:
        meshes=[]
        for comp in entity.components:
            if isinstance(comp,physx.PhysxRigidBaseComponent):
                for mesh in get_component_meshes(comp):
                    if mesh is not None and len(mesh.vertices)>=4:
                        v=np.asarray(mesh.vertices);mat=entity.pose.to_transformation_matrix()
                        meshes.append(v@mat[:3,:3].T+mat[:3,3])
        out[name]=meshes
    return out


def convex_gap(a,b):
    """SAT signed separation: >0 separated; <0 overlap (metres)."""
    alo,ahi=a.min(0),a.max(0);blo,bhi=b.min(0),b.max(0)
    coarse=np.maximum(blo-ahi,alo-bhi).max()
    if coarse>.001:return float(coarse)
    ha,hb=ConvexHull(a),ConvexHull(b)
    def edges(v,h):
        ix=np.concatenate([h.simplices[:,[0,1]],h.simplices[:,[1,2]],h.simplices[:,[2,0]]])
        d=v[ix[:,1]]-v[ix[:,0]]
        return np.unique(np.round(d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-15),7),axis=0)
    ea,eb=edges(a,ha),edges(b,hb)
    cross=np.cross(ea[:,None],eb[None]).reshape(-1,3)
    axes=np.concatenate((ha.equations[:,:3],hb.equations[:,:3],cross))
    norm=np.linalg.norm(axes,axis=1);axes=axes[norm>1e-8]/norm[norm>1e-8,None]
    aa,bb=a@axes.T,b@axes.T
    return float(np.maximum(bb.min(0)-aa.max(0),aa.min(0)-bb.max(0)).max())


def collision_audit(env,moving):
    meshes=collision_meshes(env);name='peg_0' if moving=='peg' else 'box_with_hole_0'
    gaps={n:min((convex_gap(a,b) for a in meshes[name] for b in values),default=10.)
          for n,values in meshes.items() if n!=name}
    verts=np.concatenate(meshes[name]);t=env.unwrapped
    table=t.table_scene
    # SAT table top test is supplemented by actual table dimensions/vertices.
    table_vertices=np.concatenate(meshes['table-workspace'])
    within=bool(np.all(verts[:,:2].min(0)>=table_vertices[:,:2].min(0)) and
                np.all(verts[:,:2].max(0)<=table_vertices[:,:2].max(0)))
    bad=[n for n,g in gaps.items() if g < -TOL['contact_tolerance_m']]
    near=[n for n,g in gaps.items() if g <= TOL['contact_tolerance_m']]
    return dict(gaps_m=gaps,table_xy_contained=within,penetrations=bad,near_contacts=near,
        vertices_world=verts.tolist(),valid=within and not bad,
        cached_physx_contact_count=len(t.scene.px.get_contacts()),fresh_contact_forces_available=False)


@torch.inference_mode()
def infer(model,norm,codec,graph,robot,hidden):
    g=pack_graphs([graph]).to('cuda');r=torch.from_numpy(robot[None]).cuda()
    captured={}
    def enc_hook(module,args,out):captured.update(readout=numpy(out[0]),presence=numpy(out[1]))
    def rnn_hook(module,args):captured['gru_input']=numpy(args[0])
    def fused_hook(module,args):captured['fused_input']=numpy(args[0])
    h1=model.graph_encoder.register_forward_hook(enc_hook)
    h2=model.policy.rnn.register_forward_pre_hook(rnn_hook)
    h3=model.policy.register_forward_pre_hook(fused_hook)
    try:raw,h=model.step(g,r,norm,hidden)
    finally:h1.remove();h2.remove();h3.remove()
    action=codec.decode(codec.tag(raw))
    assert captured['readout'].shape==(1,320) and captured['gru_input'].shape[-1]==128
    return dict(**captured,encoded=numpy(raw),action=numpy(action),hidden=numpy(h),
        previous_hidden=np.zeros((1,1,144),np.float32) if hidden is None else numpy(hidden),
        controller=controller_commands(numpy(action))),h


def expert_reference(env,baseline_expert,object_name,offset):
    """Recompute geometry in a separate expert, reconstruct only verified stages."""
    import sapien
    from recurrent.common import to_sapien
    ref=ScriptedExpert(env)
    phase=baseline_expert.phase
    valid=phase in ('reach','descend')
    details=dict(phase_before=phase,counter_before=baseline_expert.counter,
        status='VALID_STAGE_CONDITIONED' if valid else 'INVALID_EXPERT_REFERENCE',
        fresh_grasp_p=ref.grasp.p.tolist(),fresh_grasp_q=ref.grasp.q.tolist(),
        baseline_grasp_p=baseline_expert.grasp.p.tolist(),baseline_grasp_q=baseline_expert.grasp.q.tolist(),
        fresh_target_recomputed=True)
    if not valid:return np.zeros(7,np.float32),details
    ref.phase=phase;ref.counter=baseline_expert.counter
    # Translation must translate the independently constructed target exactly;
    # yaw can change OBB orientation, so validate the target against the new OBB.
    if offset['yaw_deg']==0:
        expected=baseline_expert.grasp.p+np.array(offset['xyz_mm'])*.001 if object_name=='peg' else baseline_expert.grasp.p
        details['target_translation_error_m']=float(np.max(abs(ref.grasp.p-expected)))
        valid=bool(np.allclose(ref.grasp.p,expected,atol=2e-6,rtol=0) and
            abs(np.dot(ref.grasp.q,baseline_expert.grasp.q))>1-1e-6)
    target=ref.grasp*sapien.Pose([0,0,-.05]) if phase=='reach' else ref.grasp
    limit=ref.max_trans if phase=='reach' else .01
    tcp=to_sapien(env.unwrapped.agent.tcp.pose)
    dp=(target.p-tcp.p)*ref.gain;dp=dp/max(1.,np.linalg.norm(dp)/limit)
    action=ref.act()
    details.update(phase_after=ref.phase,target_p=target.p.tolist(),
        independent_translation_error=float(np.max(abs(action[:3]*.1-dp))),
        translation_saturated=bool(np.linalg.norm(dp)>=limit-1e-7))
    valid=valid and details['independent_translation_error']<1e-7
    details['status']='VALID_STAGE_CONDITIONED' if valid else 'INVALID_EXPERT_REFERENCE'
    return action,details


def geometry_sidecar(env,packet,masks):
    """Audit current render segmentation, never passed into packet construction."""
    sensor=env.unwrapped._sensors['base_camera']
    seg=numpy(sensor.get_obs(rgb=False,depth=False,segmentation=True)['segmentation'])[0,...,0]
    ids=[]
    for actor in (env.unwrapped.peg,env.unwrapped.box):
        ids.append([actor._objs[0].per_scene_id])
    gt_masks=np.stack([np.isin(seg,x) for x in ids])
    source=np.asarray(packet['provenance']['source_index']);oid=np.asarray(packet['object_id'])
    stats=[]
    for o in (0,1):
        rows=source[oid==o];union=np.logical_or(masks[o],gt_masks[o]).sum()
        stats.append(dict(object_id=o,visible_gt_pixels=int(gt_masks[o].sum()),predicted_pixels=int(masks[o].sum()),
            mask_iou=float(np.logical_and(masks[o],gt_masks[o]).sum()/max(1,union)),
            selected_gt_purity=float(gt_masks[o].ravel()[rows].mean()) if len(rows) else None))
    return gt_masks,stats


def trial(plan,models,vision,actions,reset_group,k,spec,root,baseline=None):
    from mani_skill.utils.structs.pose import Pose
    from transforms3d.quaternions import qmult
    from visual.runtime.rgbd import TrackedRGBDFrame
    from graph.integration_baseline.vision import packet_from_frame
    folder=root/f't{k:02d}'/spec['name'];folder.mkdir(parents=True,exist_ok=False)
    env,conditions=environment(plan);obs,_=env.reset(seed=SEED)
    assert reset_diagnostics(env)==reset_group
    vision.reset('seed30003');hidden={label:None for label in models}
    norm=next(iter(models.values()))[1];physics=PhysicsAudit(env);audit=controller_audit(env)
    expert=ScriptedExpert(env);prefix=[];prefix_phases=[];prefix_expert=[];prefix_hashes=[]
    try:
        for t in range(k+1):
            token=physics.token()
            if t==k:
                before=physical_state(env)
                oldrow=observation(env,obs,norm.schema,'base_camera')
                collisions_before=collision_audit(env,spec['object'])
                actor=getattr(env.unwrapped,spec['object']);raw=actor.pose.raw_pose.clone()
                raw[0,:3]+=torch.tensor(spec['xyz_mm'],dtype=raw.dtype)*.001
                angle=np.deg2rad(spec['yaw_deg']);yaw=np.array([np.cos(angle/2),0,0,np.sin(angle/2)])
                raw[0,3:]=torch.from_numpy(qmult(yaw,numpy(raw)[0,3:]).astype(np.float32))
                actor.set_pose(Pose.create(raw))
                obs=env.unwrapped.get_obs()
                after=physical_state(env)
                row=observation(env,obs,norm.schema,'base_camera')
                collisions_after=collision_audit(env,spec['object'])
                unchanged=[n for n in before if n!=spec['object']+'_pose' and n not in ('hole_pose','goal_pose')]
                if spec['object']=='peg':unchanged+=['hole_pose','goal_pose']
                exact={n:bool(np.array_equal(before[n],after[n])) for n in unchanged}
                exact['robot76']=bool(np.array_equal(oldrow['robot_state'],row['robot_state']))
                exact['intrinsic']=bool(np.array_equal(oldrow['intrinsic'],row['intrinsic']))
                exact['cam2world']=bool(np.array_equal(oldrow['cam2world'],row['cam2world']))
                new_contacts=set(collisions_after['near_contacts'])-set(collisions_before['near_contacts'])
                valid=all(exact.values()) and collisions_after['valid'] and not new_contacts
                np.testing.assert_allclose(after[spec['object']+'_pose'],numpy(raw)[0],atol=1e-7,rtol=0)
                ref,refdetails=expert_reference(env,expert,spec['object'],spec)
                physical=dict(status='VALID' if valid else 'INVALID_PHYSICAL_COUNTERFACTUAL',exact=exact,
                    before={n:a.tolist() for n,a in before.items()},after={n:a.tolist() for n,a in after.items()},
                    requested_pose=numpy(raw)[0].tolist(),new_near_contacts=sorted(new_contacts),
                    collision_before=collisions_before,collision_after=collisions_after,
                    render_rgb_changed_values=int((row['rgb']!=oldrow['rgb']).sum()),
                    render_depth_changed_pixels=int((row['depth_raw_mm']!=oldrow['depth_raw_mm']).sum()),
                    physics_token=list(token),conditions=conditions)
            else:row=observation(env,obs,norm.schema,'base_camera')
            masks,record=vision.process(row['rgb'],'seed30003',t)
            frame=TrackedRGBDFrame(t,'base_camera',row['rgb'],row['depth'],row['intrinsic'],row['cam2world'],
                row['tcp_pose'],{0:masks[0],1:masks[1]},t,t/20)
            packet,_=packet_from_frame(frame,'seed30003',keep_ratio=.5)
            packet['observation_provenance']=dict(seed=SEED,timestep=t,gt_used_as_student_input=False,
                future_observations_used=False,mask_source='causal RGB-only DINO SAM2')
            graph=build_graph(packet,row['robot_ee_geometry'],**plan['config']['graph'])
            if t<k:
                prefix_phases.append(expert.phase);ea=expert.act();prefix_expert.append(ea)
                # Fixed original Expert actions, independent of model outputs.
                for label,(model,n,c) in models.items():_,hidden[label]=infer(model,n,c,graph,row['robot_state'],hidden[label])
                gh=digest(numpy(graph.node_input))+digest(numpy(graph.edge_index))+digest(numpy(graph.edge_attr))
                prefix_hashes.append(dict(t=t,rgb=digest(row['rgb']),depth=digest(row['depth_raw_mm']),
                    robot=digest(row['robot_state']),masks=digest(masks),graph=gh,
                    hidden={l:digest(numpy(h)) for l,h in hidden.items()}))
                prefix.append({n:row[n] for n in ('rgb','depth_raw_mm','robot_state')})
                physics.unchanged(token);audit.clear()
                obs,*_=env.step(torch.from_numpy(actions[t][None]).to(env.unwrapped.device))
                physics.stepped(token);validate_controller_command(actions[t],audit)
                continue
            physics.unchanged(token)
            gt_masks,support=geometry_sidecar(env,packet,masks)
            physical['geometry_support']=support
            # Check all six checkpoints on exactly this one observed graph.
            controls={};outputs={}
            for label,(model,n,c) in models.items():
                out,_=infer(model,n,c,graph,row['robot_state'],hidden[label]);outputs[label]=out
                repeat,_=infer(model,n,c,graph,row['robot_state'],hidden[label])
                pg=permute_graph(graph,np.random.default_rng(42))
                perm,_=infer(model,n,c,pg,row['robot_state'],hidden[label])
                tests={}
                for key in ('readout','presence','encoded','action','hidden','previous_hidden','gru_input'):
                    tests[key]=dict(repeat_max=float(np.max(abs(out[key]-repeat[key]))),
                        permutation_max=float(np.max(abs(out[key]-perm[key]))),
                        pass_gate=bool(np.allclose(out[key],repeat[key],atol=1e-6,rtol=1e-5) and
                            np.allclose(out[key],perm[key],atol=1e-6,rtol=1e-5)))
                controls[label]=tests
                save_npz(folder/(label+'.npz'),**out)
            same_history=True
            if baseline is not None:
                bp=read(baseline/'prefix.json');same_history=prefix_hashes==bp['hashes']
                with np.load(baseline/'observation.npz') as b:
                    physical['cross_trial_robot76_exact']=bool(np.array_equal(row['robot_state'],b['robot_state']))
                    same_history=same_history and physical['cross_trial_robot76_exact']
                for label in models:
                    with np.load(baseline/(label+'.npz')) as b:
                        assert np.array_equal(outputs[label]['previous_hidden'],b['previous_hidden']), 'previous hidden exact equality'
            physical['same_history_exact']=same_history
            if not same_history:physical['status']='INVALID_PHYSICAL_COUNTERFACTUAL'
            physical['negative_controls_pass']=all(v['pass_gate'] for tests in controls.values() for v in tests.values())
            if not physical['negative_controls_pass']:physical['status']='INVALID_NEGATIVE_CONTROL'
            save_packet(folder/'packet.npz',packet)
            meshes=collision_meshes(env)
            save_npz(folder/'collision_geometry.npz',**{name.replace(':','_')+'_'+str(i):v
                for name,values in meshes.items() for i,v in enumerate(values)})
            save_npz(folder/'graph.npz',**{n:numpy(getattr(graph,n)) for n in ('node_input','node_type','edge_index','edge_attr','edge_type')})
            save_npz(folder/'observation.npz',**{n:row[n] for n in ('rgb','depth_raw_mm','robot_state','intrinsic','cam2world','tcp_pose','robot_ee_geometry')},masks=masks,gt_masks=gt_masks)
            save_npz(folder/'expert.npz',action=ref,controller=controller_commands(ref[None])[0])
            write_json(folder/'physical.json',physical);write_json(folder/'expert.json',refdetails)
            write_json(folder/'controls.json',controls);write_json(folder/'vision.json',vision.records)
            write_json(folder/'prefix.json',dict(hashes=prefix_hashes,phases=prefix_phases,
                newly_computed_expert_actions=np.asarray(prefix_expert).tolist(),applied_actions=actions[:k].tolist(),
                phase_at_branch=expert.phase,counter_at_branch=expert.counter))
            if prefix:save_npz(folder/'prefix_observations.npz',**{n:np.stack([v[n] for v in prefix]) for n in prefix[0]})
            physics.unchanged(token)
            return dict(t=k,**spec,status=physical['status'],expert_status=refdetails['status'],
                expert_phase=refdetails['phase_before'],directory=str(folder.relative_to(root)),
                point_counts=[int(np.sum(packet['object_id']==o)) for o in (0,1)])
    finally:
        physics.close();env.close();vision.reset(None)


def run(output):
    torch.cuda.init();torch.set_num_threads(1);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan,models,identities,protected=load_models()
    ds=DirectDataset(plan['dataset_directory']);ep=ds.episodes[SEED]
    assert ep['sha256']=='ba578d3e17e804c48c5490dc9970406a2b560d9ea214519094965fa815a09122'
    path=ds.run/ep['path'];protected[str(path)]=sha(path);actions=load_episode(path)['actions']
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    with ExperimentLogger(team='integration',model='task09a_object_geometry_sensitivity',seed=SEED,
        config=protocol(),root=output,job_type='evaluate') as logger:
        root=logger.directory;print('RUN_ID',logger.run_id,flush=True)
        write_json(root/'protocol.json',protocol());write_json(root/'identities.json',identities)
        write_json(root/'protected_before.json',protected)
        src=root/'source_snapshot';src.mkdir()
        base=Path(__file__).resolve().parents[2]
        inventory={}
        for part in ('envs','graph','recurrent','visual/runtime','utils','tools'):
            for p in (base/part).rglob('*.py'):
                if '.venv' in p.parts or 'cache' in p.parts:continue
                dst=src/p.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
                inventory[str(p.relative_to(base))]=sha(p)
        write_json(root/'source_hashes.json',inventory)
        (root/'implementation.diff').write_text(subprocess.run(['git','diff','--no-index','--','/dev/null',str(Path(__file__).resolve())],capture_output=True,text=True).stdout)
        import mani_skill, sapien
        from mani_skill.utils.structs.actor import Actor
        from mani_skill.envs.sapien_env import BaseEnv
        from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
        (root/'installed_api_source.txt').write_text('\n'.join(inspect.getsource(x) for x in
            (Actor.set_pose,Actor.pose.fset,BaseEnv.get_obs,BaseEnv._get_obs_with_sensor_data,
             PegInsertionSideEnv.box_hole_pose.fget,PegInsertionSideEnv.goal_pose.fget)))
        subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],stdout=(root/'gpu_before.txt').open('w'))
        subprocess.run(['ps','-eo','pid,comm,etime,pcpu'],stdout=(root/'processes_before.txt').open('w'))
        vision=OnlineVisionAdapter(GDino(device='cuda'),SAM2Tracker(device='cuda'))
        rows=[];extra_decisions=[]
        try:
            for k in protocol()['timesteps']:
                baseline=None
                for spec in protocol()['primary']:
                    row=trial(plan,models,vision,actions,ep['reset_group'],k,spec,root,baseline)
                    rows.append(row)
                    if spec['name']=='baseline':baseline=root/row['directory']
                    print(json.dumps(row),flush=True);write_json(root/'trials.json',rows)
                # Pre-registered saturation/small-response trigger, before extra model comparisons.
                be=np.load(baseline/'expert.npz')['controller']
                primary=[r for r in rows if r['t']==k and r['group']=='primary' and r['status']=='VALID']
                norms=[float(np.linalg.norm(np.load(root/r['directory']/'expert.npz')['controller'][:3]-be[:3])*1000) for r in primary]
                saturated=all(read(root/r['directory']/'expert.json')['translation_saturated'] for r in primary)
                trigger=bool(any(n<=.1 for n in norms) or saturated)
                extra_decisions.append(dict(t=k,primary_expert_translation_mm=norms,all_saturated=saturated,execute_additional=trigger))
                write_json(root/'additional_decisions.json',extra_decisions)
                if trigger:
                    for spec in protocol()['additional']:
                        row=trial(plan,models,vision,actions,ep['reset_group'],k,spec,root,baseline)
                        rows.append(row);print(json.dumps(row),flush=True);write_json(root/'trials.json',rows)
        finally:vision.close()
        after={p:sha(p) for p in protected}
        changed=[p for p in protected if after[p]!=protected[p]]
        write_json(root/'protection_audit.json',dict(pass_gate=not changed,files=len(protected),changed=changed,after=after))
        subprocess.run(['ps','-eo','pid,comm,etime,pcpu'],stdout=(root/'processes_after.txt').open('w'))
        logger.log(dict(trials=len(rows),valid_trials=sum(r['status']=='VALID' for r in rows)))
        logger.link_output(root/'trials.json',kind='metrics')
        write_json(root/'status.json',dict(status='INFERENCE_COMPLETE',run_id=logger.run_id,trials=len(rows)))
        return root


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(run(a.output),flush=True)
