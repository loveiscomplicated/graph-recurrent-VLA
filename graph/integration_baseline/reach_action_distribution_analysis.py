"""Task07-G: saved-trajectory analysis only. No environment or policy is created."""
import csv
import hashlib
import html
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

E = 'c76d3cf8f8354c7199e13f3068d2d1c3'
A = 'a878851dd4ee4b7a853502915115454c'
C = '268e3b8cc21943fe97d554d69249a495'
D = 'd046e3a679a3430aab5e69f279c3f0a7'
CHECKPOINT = '3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776'
ROOT = Path(__file__).resolve().parents[2]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def serial(x):
    if isinstance(x, dict): return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [serial(v) for v in x]
    if isinstance(x, np.ndarray): return serial(x.tolist())
    if isinstance(x, np.generic): return serial(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write(p, d):
    Path(p).write_text(json.dumps(serial(d), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def rot(q):
    return Rotation.from_quat(np.asarray(q), scalar_first=True)


def angle(q, target):
    # Rotation composition is invariant under either quaternion's sign.
    return (rot(target).inv()*rot(q)).magnitude()*180/np.pi


def controller(actions):
    """Execute installed preprocessing on a config-only stub; no simulator object."""
    import torch
    from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
    from mani_skill.utils.gym_utils import clip_and_scale_action
    action = torch.as_tensor(np.asarray(actions, np.float32))
    stub = SimpleNamespace(action_space_low=torch.full((6,), -.1),
        action_space_high=torch.full((6,), .1), config=SimpleNamespace(rot_lower=-.1))
    arm = PDEEPoseController._clip_and_scale_action(stub, action[:, :6]).numpy()
    grip = clip_and_scale_action(action[:, 6:], torch.tensor([-.01]), torch.tensor([.04])).numpy()
    return np.c_[arm, grip]


def cosine(a, b):
    den = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    return np.divide(np.sum(a*b, axis=-1), den, out=np.full_like(den, np.nan), where=den>0)


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(n=len(x), mean=x.mean() if len(x) else None,
        median=np.median(x) if len(x) else None, p05=np.quantile(x,.05) if len(x) else None,
        p95=np.quantile(x,.95) if len(x) else None, min=x.min() if len(x) else None,
        max=x.max() if len(x) else None)


def csv_arrays(path, arrays):
    columns = {}
    for key, x in arrays.items():
        x = np.asarray(x)
        if x.ndim == 1: columns[key] = x
        else:
            for i in range(x.shape[1]): columns[f'{key}_{i}'] = x[:, i]
    with Path(path).open('w') as f:
        w=csv.writer(f); w.writerow(columns); w.writerows(zip(*columns.values()))


def digest(*values):
    h=hashlib.sha256()
    for v in values:
        v=np.asarray(v); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
    return h.hexdigest()


def load_trial(run, expert, source_paths, checks):
    base=ROOT/'experiments'/run; p=base/'trials/A'
    manifest=read(base/'manifest.json')
    assert manifest['run_id']==run
    registered={str((base/x).resolve()) for key, values in manifest.items()
        if key.endswith('_paths') for x in values}
    def use(q):
        assert str(q.resolve()) in registered, f'unregistered input: {q}'
        source_paths.add(q)
        return q
    trace=read(use(p/'trace.json')); trial=read(use(p/'trial.json'))
    actions=dict(np.load(use(p/'actions.npz'), allow_pickle=False))
    hidden=dict(np.load(use(p/'hidden.npz'), allow_pickle=False))
    obs=dict(np.load(use(p/'observations.npz'), allow_pickle=False))
    n=len(trace); ts=np.arange(n)
    np.testing.assert_array_equal(hidden['timestep'], ts)
    assert len(hidden['hidden'])==n and len(obs['robot_state'])==n+1
    assert len(trial['hidden_reset_events'])==1 and trial['hidden_reset_events'][0]['timestep']==0
    predicted=actions['predicted' if expert else 'actions']
    applied=actions['applied' if expert else 'actions']
    scaled=controller(applied)
    np.testing.assert_array_equal(actions['controller_input'],applied)
    for t,r in enumerate(trace):
        assert r['timestep']==t and r['physics_steps_after']-r['physics_steps_before']==5
        assert abs(r['simulation_time_before']-t*.05)<1e-12
        np.testing.assert_array_equal(predicted[t], np.asarray(r['predicted_action' if expert else 'action'],np.float32))
        np.testing.assert_array_equal(applied[t],np.asarray(r['controller']['controller_input'],np.float32))
        np.testing.assert_allclose(scaled[t,:6], r['controller']['arm_scaled_input'],atol=2e-8,rtol=1e-6)
        np.testing.assert_allclose(scaled[t,6:], r['controller']['gripper_scaled_input'],atol=1e-8,rtol=0)
        for key, sl in [('ee_pose',slice(24,31)),('qpos',slice(0,9)),('qvel',slice(9,18))]:
            np.testing.assert_array_equal(obs['robot_state'][t,sl],np.asarray(r['before'][key],np.float32))
            np.testing.assert_array_equal(obs['robot_state'][t+1,sl],np.asarray(r['after'][key],np.float32))
        if t+1<n:
            for key in r['before']:
                if key == 'peg_relative_tcp_pose':
                    # Stored derived quaternion is renormalized by F's metric path.
                    np.testing.assert_allclose(r['after'][key],trace[t+1]['before'][key],atol=1e-6,rtol=0)
                elif key not in ('contact_valid',):
                    np.testing.assert_array_equal(r['after'][key],trace[t+1]['before'][key])
    graph_counts=[]; availability=[]; validity=[]
    masks=np.load(use(p/'masks.npz'),allow_pickle=False)['masks'] if expert else None
    for t,r in enumerate(trace):
        g=dict(np.load(use(p/f'graph_{t:04d}.npz'),allow_pickle=False))
        pack=dict(np.load(use(p/f'packet_{t:04d}.npz'),allow_pickle=False))
        schema=json.loads(str(pack['schema_json']))
        assert schema['source_frame']==t and schema['episode']==('A' if expert else 'task07f_A_41005') and schema['coordinate_frame']=='EE_metres'
        np.testing.assert_array_equal(pack['packet/observation_ee_pose'],obs['tcp_pose'][t])
        np.testing.assert_array_equal(g['source_index'],pack['packet/provenance/source_index'])
        oid=pack['packet/object_id']; counts=[int((oid==i).sum()) for i in (0,1)]
        assert counts==r['point_counts']; assert [x>0 for x in counts]==r['presence']
        np.testing.assert_array_equal(g['node_input'][3:,:3],pack['packet/points'])
        for key in ('node_input','edge_attr'): assert np.isfinite(g[key]).all()
        graph_counts.append([len(g['node_type']),g['edge_index'].shape[1]])
        availability.append(pack['packet/feature_field_available'].mean(0))
        validity.append(pack['packet/feature_field_valid'].mean(0))
        if expert and t<=40:
            prefix=trial['prefix'][t]
            values=[obs[k][t] for k in ('rgb','depth_raw_mm','intrinsic','cam2world','robot_state','robot_world_geometry','robot_ee_geometry')]
            actual=digest(*values,masks[t],pack['packet/provenance/source_index'],*[g[k] for k in ('node_input','node_type','edge_index','edge_attr','edge_type')])
            assert actual==prefix['rgbd_calibration_robot_graph_mask_sha256']
            np.testing.assert_array_equal(np.asarray(prefix['action'],np.float32),predicted[t])
            np.testing.assert_array_equal(np.asarray(prefix['hidden'],np.float32),hidden['hidden'][t])
            if t<40: assert digest(applied[t])==prefix['expert_action_sha256']
    h=hidden['hidden'].reshape(n,-1)
    assert np.isfinite(h).all() and np.isfinite(predicted).all()
    checks.append(f'{run}: {n} Observation_t→Action_t→Observation_t+1 joins; controller audits; sequential hidden metadata; graph/packet source joins PASS')
    ref=np.asarray(trial['grasp_reference']); rel=np.asarray(trial['reference_relative'])
    def metric(state):
        tcp=np.asarray(state['ee_pose']); peg=np.asarray(state['peg_pose'])
        current_target=peg[:3]+rot(peg[3:]).apply(rel[:3])
        target_q=(rot(peg[3:])*rot(rel[3:])).as_quat(scalar_first=True)
        return [np.linalg.norm(tcp[:3]-ref[:3])*1000,
            np.linalg.norm(tcp[:3]-current_target)*1000,angle(tcp[3:],ref[3:]),angle(tcp[3:],target_q)]
    before=np.asarray([metric(r['before']) for r in trace]); after=np.asarray([metric(r['after']) for r in trace])
    if not expert:
        np.testing.assert_allclose(before[:,0],[r['metric']['reference_position_error_mm'] for r in trace],atol=3e-5,rtol=1e-6)
        np.testing.assert_allclose(before[:,1],[r['metric']['peg_relative_position_error_mm'] for r in trace],atol=3e-5,rtol=1e-6)
    return dict(run=run,trace=trace,trial=trial,predicted=predicted,applied=applied,scaled=scaled,
        shadow_scaled=controller(predicted),hidden=h,hidden_norm=np.linalg.norm(h,axis=1),
        hidden_update_norm=np.linalg.norm(np.diff(h,axis=0,prepend=np.zeros((1,h.shape[1]))),axis=1),
        obs=obs,before=before,after=after,graph_counts=np.asarray(graph_counts),
        availability=np.asarray(availability),validity=np.asarray(validity))


def action_metrics(expert, student):
    error=student-expert; ce=controller(expert); cs=controller(student)
    c=cosine(expert[:,:3],student[:,:3])
    rd=(Rotation.from_euler('XYZ',ce[:,3:6]).inv()*Rotation.from_euler('XYZ',cs[:,3:6])).magnitude()*180/np.pi
    return dict(n=len(error), action7_mse=np.mean(error**2),action7_mae=np.mean(abs(error)),
        translation_mse=np.mean(error[:,:3]**2),translation_mae=np.mean(abs(error[:,:3])),
        translation_vector_error_native=stats(np.linalg.norm(error[:,:3],axis=1)),
        translation_direction_cosine=stats(c),positive_direction_dot_count=int((c>0).sum()),
        rotation_native_mse=np.mean(error[:,3:6]**2),rotation_native_mae=np.mean(abs(error[:,3:6])),
        gripper_mse=np.mean(error[:,6]**2),gripper_mae=np.mean(abs(error[:,6])),
        per_component_mse=np.mean(error**2,0),per_component_mae=np.mean(abs(error),0),
        per_component_abs_error_p95=np.quantile(abs(error),.95,axis=0),per_component_abs_error_max=abs(error).max(0),
        controller_translation_vector_error_mm=stats(np.linalg.norm(cs[:,:3]-ce[:,:3],axis=1)*1000),
        controller_rotation_component_error_rad=stats(np.linalg.norm(cs[:,3:6]-ce[:,3:6],axis=1)),
        controller_rotation_command_geodesic_deg=stats(rd),
        controller_gripper_joint_target_error_mm=stats(abs(cs[:,6]-ce[:,6])*1000),
        expert_translation_clip_count=int((abs(expert[:,:3])>1).any(1).sum()),
        student_translation_clip_count=int((abs(student[:,:3])>1).any(1).sum()),
        expert_rotation_norm_clip_count=int((np.linalg.norm(expert[:,3:6],axis=1)>1).sum()),
        student_rotation_norm_clip_count=int((np.linalg.norm(student[:,3:6],axis=1)>1).sum()),
        expert_gripper_clip_count=int((abs(expert[:,6])>1).sum()),student_gripper_clip_count=int((abs(student[:,6])>1).sum()))


def state_arrays(x):
    trace=x['trace']
    def field(k):return np.asarray([r['before'][k] for r in trace])
    relative=field('peg_relative_tcp_pose')
    return dict(timestep=np.arange(len(trace)),target_error_mm=x['before'][:,0],
        relative_target_error_mm=x['before'][:,1],target_rotation_error_deg=x['before'][:,2],
        relative_target_rotation_error_deg=x['before'][:,3],relative_position_mm=relative[:,:3]*1000,
        relative_orientation_rotvec_rad=rot(relative[:,3:]).as_rotvec(),
        tcp_position_m=field('ee_pose')[:,:3],peg_position_m=field('peg_pose')[:,:3],
        tcp_velocity_m_s=field('tcp_linear_velocity'),tcp_angular_velocity_rad_s=field('tcp_angular_velocity'),
        peg_velocity_m_s=field('peg_linear_velocity'),peg_angular_velocity_rad_s=field('peg_angular_velocity'),
        fingertip_surface_distance_mm=field('fingertip_centre_to_peg_surface_m')*1000,
        gripper_opening_mm=field('gripper_opening_m')*1000,
        point_counts=np.asarray([r['point_counts'] for r in trace]),
        presence=np.asarray([r['presence'] for r in trace],int),graph_counts=x['graph_counts'],
        f5_available_fraction=x['availability'],f5_valid_fraction=x['validity'],
        hidden_norm=x['hidden_norm'],hidden_update_norm=x['hidden_update_norm'])


def build_analysis(e,a,out):
    stages={'reach':(0,26),'descend':(26,40),'reach_end_22_25':(22,26),'transition_25_28':(25,29),'descend_end_36_39':(36,40)}
    metrics={k:action_metrics(e['applied'][lo:hi],e['predicted'][lo:hi]) for k,(lo,hi) in stages.items()}
    ce=e['scaled'][:40]; cs=e['shadow_scaled'][:40]; err=e['predicted'][:40]-e['applied'][:40]
    comp=dict(timestep=np.arange(40),expert=e['applied'][:40],student=e['predicted'][:40],
        native_error=err,expert_controller=ce,student_virtual_controller=cs,
        translation_error_mm=np.linalg.norm(cs[:,:3]-ce[:,:3],axis=1)*1000,
        direction_cosine=cosine(e['applied'][:40,:3],e['predicted'][:40,:3]))
    csv_arrays(out/'action_comparison.csv',comp);np.savez_compressed(out/'action_comparison.npz',**comp)
    es,as_=state_arrays(e),state_arrays(a)
    # Same-state action error only above. This section describes applied autonomous actions.
    target=np.asarray(a['trial']['grasp_reference'])[:3]
    base_q=a['obs']['robot_state'][:-1,66:70]
    world_command=rot(base_q).apply(a['scaled'][:,:3])
    toward=target-as_['tcp_position_m']
    direction=cosine(world_command,toward)
    physical_delta=np.asarray([r['after']['ee_pose'][:3] for r in a['trace']])-as_['tcp_position_m']
    progress=dict(**as_,delta_e_mm=a['after'][:,0]-a['before'][:,0],
        delta_relative_e_mm=a['after'][:,1]-a['before'][:,1],action=a['applied'],controller=a['scaled'],
        world_translation_command_m=world_command,target_direction_cosine=direction,
        radial_command_mm=np.sum(world_command*toward/np.linalg.norm(toward,axis=1)[:,None],axis=1)*1000,
        actual_tcp_displacement_mm=physical_delta*1000,
        translation_command_norm_mm=np.linalg.norm(world_command,axis=1)*1000,
        translation_norm_change_mm=np.r_[np.nan,np.diff(np.linalg.norm(world_command,axis=1)*1000)],
        translation_direction_step_change_deg=np.r_[np.nan,np.degrees(np.arccos(np.clip(cosine(world_command[:-1],world_command[1:]),-1,1)))],
        actual_orientation_step_deg=np.array([angle(r['after']['ee_pose'][3:],r['before']['ee_pose'][3:]) for r in a['trace']]))
    csv_arrays(out/'reach_progress.csv',progress);np.savez_compressed(out/'reach_progress.npz',**progress)
    np.savez_compressed(out/'state_hidden_comparison.npz',**{f'{label}_{k}':v for label,st in [('expert',es),('autonomous',as_)] for k,v in st.items()},expert_hidden=e['hidden'],autonomous_hidden=a['hidden'],expert_robot76=e['obs']['robot_state'],autonomous_robot76=a['obs']['robot_state'])
    # Four intervals are defined by the request, not by observed metric thresholds.
    intervals={}
    for name,(lo,hi) in {'0_10':(0,11),'11_18':(11,19),'19_26':(19,27),'27_34':(27,35),'35_249':(35,250)}.items():
        intervals[name]=dict(observation_t=[lo,hi-1],end_observation_t=hi,
            start_error_mm=a['before'][lo,0],end_error_mm=a['after'][hi-1,0],
            net_delta_e_mm=progress['delta_e_mm'][lo:hi].sum(),increasing_steps=int((progress['delta_e_mm'][lo:hi]>0).sum()),
            direction_cosine=stats(direction[lo:hi]),radial_command_mm=stats(progress['radial_command_mm'][lo:hi]),
            translation_command_norm_mm=stats(np.linalg.norm(a['scaled'][lo:hi,:3],axis=1)*1000),
            rotation_error_start_end_deg=[a['before'][lo,2],a['after'][hi-1,2]],
            actual_tcp_displacement_sum_mm=physical_delta[lo:hi].sum(0)*1000,
            gripper_command_range=[a['applied'][lo:hi,6].min(),a['applied'][lo:hi,6].max()],
            hidden_update_norm=stats(a['hidden_update_norm'][lo:hi]),
            peg_displacement_mm=np.linalg.norm(np.asarray(a['trace'][hi-1]['after']['peg_pose'][:3])-np.asarray(a['trace'][lo]['before']['peg_pose'][:3]))*1000,
            point_counts_mean=as_['point_counts'][lo:hi].mean(0),f5_valid_fraction_mean=a['validity'][lo:hi].mean(0))
    distributions={}
    # Expert t0..39 vs autonomous t0..26: unequal phase durations, explicitly descriptive.
    for k in es:
        if k=='timestep':continue
        distributions[k]={'expert_0_39':dict(mean=es[k][:40].mean(0),std=es[k][:40].std(0),min=es[k][:40].min(0),max=es[k][:40].max(0)),
            'autonomous_0_26':dict(mean=as_[k][:27].mean(0),std=as_[k][:27].std(0),min=as_[k][:27].min(0),max=as_[k][:27].max(0))}
    # Progress alignment pairs all autonomous approach observations to nearest scalar distance.
    # This does NOT assert equal physical state or assign counterfactual expert actions.
    pairs=[]
    for t in range(28):
        j=int(np.argmin(abs(es['target_error_mm'][:41]-as_['target_error_mm'][t])))
        er=np.asarray(e['trace'][j]['before']['peg_relative_tcp_pose']); ar=np.asarray(a['trace'][t]['before']['peg_relative_tcp_pose'])
        pairs.append(dict(autonomous_t=t,expert_t=j,distance_gap_mm=as_['target_error_mm'][t]-es['target_error_mm'][j],
            relative_position_gap_mm=np.linalg.norm(ar[:3]-er[:3])*1000,
            relative_orientation_gap_deg=angle(ar[3:],er[3:]),opening_gap_mm=as_['gripper_opening_mm'][t]-es['gripper_opening_mm'][j],
            hidden_vector_distance=np.linalg.norm(a['hidden'][t]-e['hidden'][j])))
    write(out/'state_hidden_comparison.json',dict(windows='Expert t0..39 vs autonomous t0..26; descriptive, phase/duration confounded',
        summaries=distributions,progress_alignment='nearest scalar world-target distance among expert t0..40; not equal state/history',progress_pairs=pairs,
        same_t_hidden_distance=np.linalg.norm(e['hidden'][:40]-a['hidden'][:40],axis=1),
        gnn_readout='pending restricted offline forward verification; stored hidden is authoritative'))
    top=np.argsort(comp['translation_error_mm'])[::-1][:8]
    key={}
    for label,x,times in [('expert',e,[0,12,13,14,15,18,19,22,23,24,25,26,27,28,36,39,40]),('autonomous',a,[0,10,13,18,19,20,26,27,28,34])]:
        key[label]={str(t):dict(trace=x['trace'][t],world_error_mm=x['before'][t,0],relative_error_mm=x['before'][t,1],
            rotation_error_deg=x['before'][t,2],hidden_norm=x['hidden_norm'][t],hidden_update_norm=x['hidden_update_norm'][t],
            virtual_student_controller=x['shadow_scaled'][t]) for t in times}
    write(out/'key_diagnostics.json',key)
    first_inc=int(np.flatnonzero(progress['delta_e_mm']>0)[0]); neg=np.flatnonzero(a['applied'][:,6]<0)
    result=dict(scope='SAVED_SEED41005_TRAJECTORY_ANALYSIS_NOT_CAUSAL_ATTRIBUTION',expert=metrics,
        expert_first_negative_shadow_gripper=int(np.flatnonzero(e['predicted'][:,6]<0)[0]),
        expert_approach_minimum_gripper=dict(timestep=int(e['predicted'][:40,6].argmin()),raw=e['predicted'][:40,6].min(),controller_joint_target_mm=e['shadow_scaled'][:40,6].min()*1000),
        expert_top_translation_error_timesteps=top,autonomous_intervals=intervals,
        autonomous_first_distance_increase_action_t=first_inc,
        autonomous_first_negative_radial_command_t=int(np.flatnonzero(progress['radial_command_mm']<0)[0]),
        autonomous_minimum_pre_action_error_t=int(a['before'][:,0].argmin()),autonomous_first_close_t=int(neg[0]),
        expert_transition_actions={str(t):dict(expert=e['applied'][t],student=e['predicted'][t],expert_controller=e['scaled'][t],student_controller=e['shadow_scaled'][t]) for t in [0,22,25,26,27,28,36,39]},
        autonomous_key_actions={str(t):dict(action=a['applied'][t],world_command_mm=world_command[t]*1000,direction_cosine=direction[t],
            delta_e_mm=progress['delta_e_mm'][t],error_mm=a['before'][t,0],rotation_deg=a['before'][t,2]) for t in [0,10,13,18,19,20,26,27,34]},
        world_relative_error_max_gap_before_close_mm=np.max(abs(a['before'][:28,0]-a['before'][:28,1])),
        findings={'H1':'SUPPORTED_ON_EXPERT_STATE_IMITATION_ERROR','H2':'SUPPORTED_DISTRIBUTION_SHIFT_HYPOTHESIS',
            'H3':'INPUT_REPRESENTATION_REQUIRES_FURTHER_TEST','causal_contribution':'INCONCLUSIVE'})
    write(out/'analysis.json',result)
    return result,comp,progress,es,as_


def figures(out,e,a,comp,p,es,as_):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.grid':True})
    def save(fig,name):fig.tight_layout();fig.savefig(out/name,dpi=155);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,12))
    for i in range(7):
        ax=axes.flat[i]
        ax.plot(comp['timestep'],comp['expert'][:,i],label='Expert'); ax.plot(comp['timestep'],comp['student'][:,i],label='Student shadow')
        ax.set_title(['Translation X','Translation Y','Translation Z','Rotation X (native)','Rotation Y (native)','Rotation Z (native)','Gripper'][i]);ax.legend()
    axes.flat[7].plot(comp['translation_error_mm'],label='Controller translation command error [mm]');axes.flat[7].legend()
    for ax in axes.flat:ax.axvspan(25.5,39.5,color='orange',alpha=.12);ax.axvline(25.5,color='gray');ax.set_xlabel('Expert observation t (Reach 0..25; Descend 26..39)')
    save(fig,'expert_native_actions.png')
    fig,axes=plt.subplots(3,1,figsize=(12,10))
    for i,sl in enumerate([slice(0,3),slice(3,6),slice(6,7)]):
        for j in range(sl.start,sl.stop):
            scale=1000 if i!=1 else 1
            axes[i].plot(comp['expert_controller'][:,j]*scale,label=f'Expert {j}')
            axes[i].plot(comp['student_virtual_controller'][:,j]*scale,'--',label=f'Shadow {j}')
        axes[i].axvspan(25.5,39.5,color='orange',alpha=.12);axes[i].legend(ncol=3)
        axes[i].set_ylabel(['Translation command mm','Euler XYZ command rad','Finger joint target mm'][i])
    save(fig,'expert_controller_commands.png')
    fig,axes=plt.subplots(4,2,figsize=(15,12)); t=np.arange(35)
    axes[0,0].plot(t,p['target_error_mm'][:35],label='World target');axes[0,0].plot(t,p['relative_target_error_mm'][:35],'--',label='Current peg target');axes[0,0].set_ylabel('Distance mm');axes[0,0].legend()
    axes[0,1].bar(t,p['delta_e_mm'][:35]);axes[0,1].axhline(0,color='gray');axes[0,1].set_ylabel('e(t+1)-e(t), mm')
    for i in range(3):
        axes[1,0].plot(t,p['controller'][:35,i]*1000,label='XYZ'[i]);axes[1,1].plot(t,p['controller'][:35,i+3],label='XYZ'[i]);axes[2,0].plot(t,p['tcp_position_m'][:35,i]*1000,label='XYZ'[i]);axes[2,1].plot(t,p['tcp_velocity_m_s'][:35,i],label='XYZ'[i])
    axes[1,0].set_ylabel('Root translation command mm');axes[1,1].set_ylabel('Euler rotation command rad');axes[2,0].set_ylabel('TCP world position mm');axes[2,1].set_ylabel('TCP world velocity m/s')
    axes[3,0].plot(t,p['target_rotation_error_deg'][:35],label='Target orientation error deg');axes[3,0].legend()
    axes[3,1].plot(t,p['action'][:35,6],label='Raw gripper');axes[3,1].plot(t,p['target_direction_cosine'][:35],label='World command / target cosine');axes[3,1].legend()
    for ax in axes.flat:ax.axvline(19,color='green',ls=':',label='t19');ax.axvline(27,color='red',ls=':',label='t27');ax.set_xlabel('Autonomous observation t; green=t19, red=first close t27')
    for ax in axes[1:3].flat:ax.legend()
    save(fig,'autonomous_progress.png')
    fig=plt.figure(figsize=(12,5))
    for i,(title,limits) in enumerate([('Approach paths, independent histories',(41,28)),('Full autonomous trajectory',(41,250))]):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        for st,label,stop in [(es,'Expert-controlled',limits[0]),(as_,'Autonomous',limits[1])]:
            xyz=st['tcp_position_m'][:stop]*1000;ax.plot(*xyz.T,label=label)
        ax.scatter(*(np.asarray(a['trial']['grasp_reference'])[:3]*1000),marker='*',s=80,label='Fixed GT target')
        ax.scatter(*(as_['tcp_position_m'][[19,27]]*1000).T,c=['green','red']);ax.set(xlabel='World X mm',ylabel='World Y mm',zlabel='World Z mm',title=title);ax.legend()
    save(fig,'tcp_paths_3d.png')
    fig,axes=plt.subplots(4,2,figsize=(15,12))
    for label,st,stop in [('Expert (own t)',es,41),('Autonomous (own t)',as_,35)]:
        tt=np.arange(stop)
        for i in range(3):
            axes[0,0].plot(tt,st['relative_position_mm'][:stop,i],label=f'{label} {"XYZ"[i]}')
        for ax,k in [(axes[0,1],'relative_target_rotation_error_deg'),(axes[1,0],'gripper_opening_mm'),(axes[2,0],'hidden_norm'),(axes[2,1],'hidden_update_norm')]:ax.plot(tt,st[k][:stop],label=label)
        for i in range(2):axes[1,1].plot(tt,st['point_counts'][:stop,i],label=f'{label} {i}')
        axes[3,0].plot(tt,st['graph_counts'][:stop,1],label=f'{label} edges')
        for i in range(3):axes[3,1].plot(tt,st['f5_valid_fraction'][:stop,i],label=f'{label} F{i}')
    for ax,title in zip(axes.flat,['Peg-frame TCP XYZ mm','Peg-relative target orientation deg','Gripper opening mm','Peg / Box point counts (presence=1 throughout approach)','Hidden norm','Hidden update norm (h0 minus zero)','Graph edge count','F5 valid fractions']):ax.set_title(title);ax.legend(fontsize=7);ax.set_xlabel('Own observation t; equal t does not mean equal state')
    save(fig,'state_history_comparison.png')
    # Existing images only: display native masks; no mask/geometry inference.
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,(x,label,times) in enumerate([(e,'Expert',[19,26,39]),(a,'Autonomous',[19,26,27])]):
        base=ROOT/'experiments'/x['run']/'trials/A'
        z=np.load(base/'masks.npz');mask=z['masks'];mask=mask[:,list(z['axis_names']).index('applied')] if mask.ndim==5 else mask
        for col,t in enumerate(times):
            ax=axes[row,col];ax.imshow(x['obs']['rgb'][t]);ax.contour(mask[t,0],levels=[.5],colors=['red'],linewidths=.7);ax.contour(mask[t,1],levels=[.5],colors=['cyan'],linewidths=.7);ax.set_title(f'{label} own t={t}; red Peg / cyan Box');ax.axis('off')
    save(fig,'stored_mask_montage.png')


def offline_forward(e,a,out,paths):
    import torch
    from graph.integration_baseline.rollout_checkpoint import load_bundle
    from graph.integration_baseline.graph import Graph,pack_graphs
    from graph.integration_baseline.model import EpisodeStream
    bundle=ROOT/'outputs/graph/integration_baseline/task07_vast57640710/bundle'
    for q in bundle.rglob('*'):
        if q.is_file():paths.add(q)
    model,norm,prov=load_bundle(bundle/'checkpoint_manifest.json',device='cpu',check_runtime=False)
    assert prov['manifest']['files']['checkpoint']['sha256']==CHECKPOINT
    def tensor_hash():
        return digest(*[v.detach().numpy() for v in model.state_dict().values()],norm.stats.mean.numpy(),norm.stats.std.numpy())
    before=tensor_hash();results={};arrays={}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for label,x,count in [('expert',e,41),('autonomous',a,35)]:
        stream=EpisodeStream(model,norm);actions=[];hidden=[];readouts=[]
        for t in range(count):
            g=dict(np.load(ROOT/'experiments'/x['run']/'trials/A'/f'graph_{t:04d}.npz'))
            graph=Graph(*[torch.from_numpy(g[k]) for k in ('node_input','node_type','edge_index','edge_attr','edge_type')],{}, {})
            packed=pack_graphs([graph]);robot=torch.from_numpy(x['obs']['robot_state'][t:t+1])
            with torch.inference_mode():
                z,_=model.graph_encoder(packed)
                action=stream.step(packed,robot,label,t)
            actions.append(action.numpy()[0]);hidden.append(stream.hidden.numpy().reshape(-1));readouts.append(z.numpy()[0])
        actions=np.asarray(actions);hidden=np.asarray(hidden);readouts=np.asarray(readouts)
        action_ok=np.isclose(actions,x['predicted'][:count],atol=1e-6,rtol=1e-5)
        hidden_ok=np.isclose(hidden,x['hidden'][:count],atol=1e-6,rtol=1e-5)
        # Repeat the identical local computation with an independent t0 stream.
        repeat=EpisodeStream(model,norm)
        for t in range(count):
            g=dict(np.load(ROOT/'experiments'/x['run']/'trials/A'/f'graph_{t:04d}.npz'))
            graph=Graph(*[torch.from_numpy(g[k]) for k in ('node_input','node_type','edge_index','edge_attr','edge_type')],{}, {})
            with torch.inference_mode():
                action=repeat.step(pack_graphs([graph]),torch.from_numpy(x['obs']['robot_state'][t:t+1]),label,t)
            np.testing.assert_array_equal(action.numpy()[0],actions[t]);np.testing.assert_array_equal(repeat.hidden.numpy().reshape(-1),hidden[t])
        results[label]=dict(timesteps=[0,count-1],original_runtime='Linux Torch2.8.0 CUDA FP32',
            replay_runtime=f'macOS Torch{torch.__version__} CPU FP32',atol=1e-6,rtol=1e-5,
            action_max_abs_difference=np.max(abs(actions-x['predicted'][:count])),
            hidden_max_abs_difference=np.max(abs(hidden-x['hidden'][:count])),same_runtime_repeat='EXACT',cross_device_tolerance_pass=bool(action_ok.all() and hidden_ok.all()),
            action_outside_tolerance=np.argwhere(~action_ok),hidden_outside_tolerance=np.argwhere(~hidden_ok),
            readout_usage='DIAGNOSTIC_ONLY_NOT_USED_FOR_CAUSAL_OR_DISTRIBUTION_CLAIMS' if not (action_ok.all() and hidden_ok.all()) else 'VERIFIED_SAVED_INPUT_FORWARD',
            readout_norm=stats(np.linalg.norm(readouts,axis=1)))
        arrays.update({label+'_actions':actions,label+'_hidden':hidden,label+'_gnn_readout':readouts})
    assert before==tensor_hash()
    c=np.load(ROOT/'experiments'/C/'normalizer.npz')
    for k in ('mean','std'):
        np.testing.assert_array_equal(c[k],getattr(norm.stats,k).numpy())
    np.savez_compressed(out/'offline_forward.npz',**arrays)
    write(out/'offline_forward.json',dict(status='PASS' if all(v['cross_device_tolerance_pass'] for v in results.values()) else 'CROSS_DEVICE_TOLERANCE_NOT_MET',results=results,weights_normalizer_before_after_hash=before,
        source_validation='existing load_bundle all source/config/checkpoint/normalizer/split checks; check_runtime=False skips unused Vision asset inventory only',
        gnn_readout_scope='Expert0..40 / autonomous0..34; diagnostic, not causal representation probe'))
    return arrays


def main():
    os.chdir(ROOT);os.environ['WANDB_MODE']='disabled'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from utils.experiment_logger import ExperimentLogger
    import torch
    from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
    protected=read('/tmp/task07g-protected-before.json')
    cfg=dict(task='07-G',seed=41005,expert_run=E,autonomous_run=A,checkpoint_sha256=CHECKPOINT,
        policy='stored actions/hidden only',scope='READ_ONLY_SAVED_TRAJECTORY',offline_forward='limited CPU stored graph forward; expert0..40/autonomous0..34',
        metrics='no posthoc pass/fail thresholds; cosine undefined at zero norm; all 40 expert frames',
        interval_contract='Observation_t -> Action_t -> Observation_(t+1)')
    with ExperimentLogger(team='integration',config=cfg,seed=41005,model='reach_action_distribution_analysis',job_type='evaluate') as log:
        out=log.directory;print('RUN_ID',log.run_id,flush=True)
        write(out/'protected_before.json',protected)
        paths=set(); checks=[]
        for rid in [C,D,E,A]:
            paths.add(ROOT/'experiments'/rid/'manifest.json');paths.add(ROOT/'experiments'/rid/'metadata.json')
        for rid in [E,A]:
            q=ROOT/'experiments'/rid/'checkpoint_provenance.json';paths.add(q);prov=read(q)
            assert prov['manifest']['files']['checkpoint']['sha256']==CHECKPOINT
            assert prov['progress']['completed_epoch']==39 and prov['progress']['global_step']==8959
        e=load_trial(E,True,paths,checks); a=load_trial(A,False,paths,checks)
        assert e['trial']['reset_group']==a['trial']['initial']['reset_identity']
        for k in ('rgb','depth_raw_mm','intrinsic','cam2world','robot_state'):
            np.testing.assert_array_equal(e['obs'][k][0],a['obs'][k][0])
        np.testing.assert_array_equal(e['predicted'][0],a['predicted'][0]);np.testing.assert_array_equal(e['hidden'][0],a['hidden'][0])
        assert [r['stage'] for r in e['trace'][:40]]==['reach']*26+['descend']*14
        # Independent recorded Expert prefix, not regenerated Expert labels.
        pc=ROOT/'experiments'/E/'trials/positive_control'
        for name in ['actions.npz','hidden.npz','observations.npz','trial.json']:paths.add(pc/name)
        ep=np.load(pc/'actions.npz');hp=np.load(pc/'hidden.npz');op=np.load(pc/'observations.npz')
        np.testing.assert_array_equal(ep['applied'][:40],e['applied'][:40])
        np.testing.assert_array_equal(ep['predicted'][:41],e['predicted'][:41]);np.testing.assert_array_equal(hp['hidden'][:41].reshape(41,-1),e['hidden'][:41])
        np.testing.assert_array_equal(op['robot_state'][:41],e['obs']['robot_state'][:41])
        checks.append('Reset RGB-D/calibration/Robot76, t0 action/hidden exact across E/F; Expert A/positive-control applied40 and shadow/hidden41 exact')
        # Full autonomous trace vs original saved held-out41005 actions.
        orig=ROOT/'experiments'/A/'dependencies/original_41005/actions.npz';paths.add(orig)
        original=dict(np.load(orig));origact=original.get('actions',original.get('predicted'))
        np.testing.assert_array_equal(origact,a['applied'])
        review=ROOT/'outputs/graph/integration_baseline/task07_vast57640710/review_41005/rollout/heldout_41005'
        for name in ['observations.npz','hidden.npz']:
            q=review/name;paths.add(q);data=np.load(q)
            if name=='observations.npz':
                for key in a['obs']:np.testing.assert_array_equal(a['obs'][key],data[key])
            else:np.testing.assert_array_equal(a['hidden'],data['hidden'].reshape(250,-1))
        checks.append('Original41005 vs 07-F A: all250 native actions exact')
        result,comp,p,es,as_=build_analysis(e,a,out)
        forward=offline_forward(e,a,out,paths)
        state=read(out/'state_hidden_comparison.json');state['gnn_readout']=read(out/'offline_forward.json');write(out/'state_hidden_comparison.json',state)
        checks.append('Offline CPU sequential41+35 forward: see offline_forward.json for fixed-tolerance failures; same-runtime repeat exact; core analysis uses stored output only; weights/normalizer unchanged')
        figures(out,e,a,comp,p,es,as_)
        for q in [ROOT/'experiments'/C/'normalizer.npz',ROOT/'experiments'/C/'provenance.json',ROOT/'experiments'/C/'summary.json',ROOT/'experiments'/A/'training_distribution.json']:
            paths.add(q)
        review=ROOT/'outputs/graph/integration_baseline/task07_vast57640710/review_41005'
        share=read(review/'share_manifest.json');paths.add(review/'share_manifest.json')
        for name in ['training/test_metrics.json','training/metrics.json','training/best.json']:
            q=review/name;assert sha(q)==share['files'][name]['sha256'];paths.add(q)
        write(out/'training_metrics_reference.json',dict(test=read(review/'training/test_metrics.json'),best_history=read(ROOT/'experiments'/E/'checkpoint_provenance.json')['progress']['history'][-1],scope='original full episode metrics; not stage-matched comparison'))
        sourcehash={str(q.relative_to(ROOT)):sha(q) for q in sorted(paths)}
        write(out/'input_manifest.json',dict(sources=sourcehash,checkpoint_bytes='VERIFIED_LOCAL_BUNDLE',
            checkpoint_identity='load_bundle validates checkpoint SHA and manifest identities',
            normalizer='checkpoint original train96; exact stored C mean/std; not refitted',
            missing=['raw training run directory absent; verified checkpoint bundle and training metrics copy available'],
            inference_input='saved Graph and Robot76 only, sequential t0 hidden; GT only metrics'))
        write(out/'source_hashes.json',sourcehash)
        (out/'installed_controller_source.txt').write_text(inspect.getsource(PDEEPoseController._clip_and_scale_action)+'\n'+inspect.getsource(PDEEPoseController.compute_target_pose))
        shutil.copy2('/tmp/task07g-run.log',out/'earlier_forward_attempt.log')
        source=out/'source_snapshot';source.mkdir()
        shutil.copy2(ROOT/'graph/integration_baseline/reach_action_distribution_report.py',source/'reach_action_distribution_report.py')
        shutil.copy2('/tmp/task07g-tests.txt',out/'test_log.txt')
        for name in ['reach_action_distribution_analysis.py','test_reach_action_distribution_analysis.py']:
            q=ROOT/'graph/integration_baseline'/('tests/'+name if name.startswith('test_') else name)
            if q.exists():shutil.copy2(q,source/name)
        (out/'implementation.diff').write_text(''.join(subprocess.run(['git','diff','--no-index','--','/dev/null',str(q)],capture_output=True,text=True).stdout for q in [Path(__file__).resolve(),ROOT/'graph/integration_baseline/reach_action_distribution_report.py',ROOT/'graph/integration_baseline/tests/test_reach_action_distribution_analysis.py']))
        failed=[]
        for filename in ['/tmp/task07g-first-attempt.log','/tmp/task07g-report-attempt.log']:
            q=Path(filename)
            if q.exists():
                shutil.copy2(q,out/q.name)
                ids=[line.split()[-1] for line in q.read_text().splitlines() if line.startswith('RUN_ID ')]
                failed.append(dict(run_ids=ids,log=q.name))
        write(out/'prior_attempts.json',dict(attempts=failed,final_core_analysis='stored outputs; no threshold relaxation'))
        write(out/'execution.json',dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
            executed='CPU saved array analysis and installed controller preprocessing only',cuda='NOT_EXECUTED',mps='NOT_EXECUTED',
            simulator_reset_steps=0,training_steps=0,offline_forward='CPU expert0..40/autonomous0..34, plus exact repeat',uploads=0))
        write(out/'implementation_source_hashes.json',{str(q.relative_to(out)):sha(q) for q in sorted((out/'source_snapshot').glob('*'))})
        log.record(source_run_id=E,source_runs=[E,A,C,D],checkpoint_sha256=CHECKPOINT,
            task='07-G',environment={'execution':'offline analysis; no simulator'},input_manifest='input_manifest.json')
        log.log({'expert':result['expert'],'autonomous':result['autonomous_intervals']},prefix='analysis')
        (out/'contract_checks.log').write_text('\n'.join(checks)+'\n')
        # Report content is a dedicated analysis-only module, no production changes.
        from graph.integration_baseline.reach_action_distribution_report import render_report
        render_report(out,result,e,a,p,es,as_)
        changed=[name for name,before in protected.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=before]
        assert not changed,changed
        write(out/'protection.json',dict(protected_files=len(protected),changed=changed,status='PASS',
            note='all preexisting source/docs and four local source-run trees; includes local checkpoint bundle and heldout/training review copy'))
        for q in sorted(out.rglob('*')):
            if q.is_file() and not q.name.endswith('.tracking.json') and q.name not in ('metadata.json','manifest.json','metrics.json','config.yaml'):
                log.link_output(q,kind='trajectory' if q.suffix=='.npz' else 'reports')
        write(out/'verification.json',{'status':'PENDING_READBACK'})
        verify(out)
        log.link_output(out/'verification.json',kind='reports')
        print('COMPLETED',out,flush=True)


def verify(out):
    import re
    from PIL import Image
    counts={}
    for ext in ['json','npz','csv','png']:
        files=[q for q in out.rglob('*.'+ext) if not q.name.endswith('.tracking.json')]
        for q in files:
            if ext=='json': read(q)
            elif ext=='npz':
                with np.load(q,allow_pickle=False) as z:
                    for k in z.files:_=z[k]
            elif ext=='csv':
                with q.open() as f:rows=list(csv.reader(f));assert len(rows)>1
            elif ext=='png':
                with Image.open(q) as im:im.verify()
        counts[ext]=len(files)
    links=[]
    for q in [out/'reach_action_distribution_analysis.html', ROOT/'docs/visual-graph-integration/07g_reach_action_distribution_analysis.md']:
        text=q.read_text()
        refs=re.findall(r'(?:href|src)="([^"]+)"',text) if q.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',text)
        for ref in refs:
            if ref.startswith(('https:','http:','#')):continue
            assert (q.parent/ref).exists(),(q,ref);links.append(ref)
    write(out/'verification.json',dict(status='PASS',decoded=counts,relative_links=len(links),
        source_protection=read(out/'protection.json'),
        contracts='contract_checks.log',tests='test_log.txt',
        numerical='stored outputs exact where compared; controller preprocessing tolerance atol2e-8/rtol1e-6',
        offline_forward='offline_forward.json: cross-device tolerance and exact same-runtime repeat'))


if __name__=='__main__':main()
