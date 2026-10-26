"""Fixed-checkpoint Task07-G offline sampling and recurrent-history diagnostics.

No environments, vision inference, training, publishing, or production mutations.
"""
import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from graph.integration_baseline import point_resampling_adapter as adapter
from graph.integration_baseline.direct_dataset import DirectDataset
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import read_packet, save_npz, write_json, sha256
from graph.integration_baseline.rollout_checkpoint import load_bundle
from utils.experiment_logger import ExperimentLogger
from visual.runtime.rgbd import TrackedRGBDFrame


def arr(x):
    return x.detach().cpu().numpy().copy()


def digest(*values):
    h = hashlib.sha256()
    for value in values:
        value = np.asarray(value)
        h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    # Identical zero slots are equal; one empty slot is undefined (NaN in NPZ).
    return np.divide((a*b).sum(-1), n, out=np.where((a == b).all(-1), 1., np.nan), where=n>0)


def stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None,
        p95=float(np.quantile(x,.95)) if len(x) else None, max=float(x.max()) if len(x) else None)


def controller(actions):
    """Execute the installed controller preprocessing without instantiating an env."""
    from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
    from mani_skill.utils.gym_utils import clip_and_scale_action
    a = torch.as_tensor(np.asarray(actions, np.float32))
    stub = SimpleNamespace(action_space_low=torch.full((6,), -.1), action_space_high=torch.full((6,), .1),
        config=SimpleNamespace(rot_lower=-.1))
    arm = PDEEPoseController._clip_and_scale_action(stub, a[:, :6])
    grip = clip_and_scale_action(a[:, 6:], torch.tensor([-.01]), torch.tensor([.04]))
    return np.c_[arm.numpy(), grip.numpy()]


def rotation_distance(command_a, command_b):
    # ManiSkill compute_target_pose calls euler_angles_to_matrix(delta_rot, 'XYZ').
    from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix
    ma = euler_angles_to_matrix(torch.tensor(command_a[:, 3:6]), 'XYZ').numpy()
    mb = euler_angles_to_matrix(torch.tensor(command_b[:, 3:6]), 'XYZ').numpy()
    return (Rotation.from_matrix(ma).inv()*Rotation.from_matrix(mb)).magnitude()


def action_metrics(a, b, expert):
    a, b, expert = [np.asarray(x, np.float32) for x in (a, b, expert)]
    ca, cb, ce = [controller(x) for x in (a, b, expert)]
    result = dict(native7_l2=np.linalg.norm(b-a, axis=1),
        native_translation_l2=np.linalg.norm(b[:,:3]-a[:,:3], axis=1),
        native_rotation_l2=np.linalg.norm(b[:,3:6]-a[:,3:6], axis=1),
        native_gripper_abs=np.abs(b[:,6]-a[:,6]),
        command_translation_mm=np.linalg.norm(cb[:,:3]-ca[:,:3], axis=1)*1000,
        command_rotation_rad=rotation_distance(ca,cb),
        command_rotation_xyz_l2_rad=np.linalg.norm(cb[:,3:6]-ca[:,3:6], axis=1),
        command_gripper_mm=np.abs(cb[:,6]-ca[:,6])*1000,
        translation_direction_cosine=cosine(ca[:,:3],cb[:,:3]),
        rotation_direction_cosine=cosine(ca[:,3:6],cb[:,3:6]),
        gripper_sign_changed=(a[:,6]<0)!=(b[:,6]<0),
        expert_mse_a=np.mean((a.astype(float)-expert)**2,axis=1),
        expert_mse_b=np.mean((b.astype(float)-expert)**2,axis=1))
    result['expert_mse_delta'] = result['expert_mse_b']-result['expert_mse_a']
    for name, sl in [('translation',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]:
        for label, pred in [('a',a),('b',b)]:
            result[f'expert_native_{name}_mse_{label}'] = np.mean((pred[:,sl].astype(float)-expert[:,sl])**2,axis=1)
        result[f'expert_native_{name}_mse_delta'] = result[f'expert_native_{name}_mse_b']-result[f'expert_native_{name}_mse_a']
    for label, pred in [('a',ca),('b',cb)]:
        result[f'expert_command_translation_mm_{label}'] = np.linalg.norm(pred[:,:3]-ce[:,:3],axis=1)*1000
        result[f'expert_command_rotation_rad_{label}'] = rotation_distance(pred,ce)
        result[f'expert_command_gripper_mm_{label}'] = np.abs(pred[:,6]-ce[:,6])*1000
    for name in ('translation_mm','rotation_rad','gripper_mm'):
        result['expert_command_'+name+'_delta'] = result['expert_command_'+name+'_b']-result['expert_command_'+name+'_a']
    return result


def recurrent(model, fused, pre_hidden=None):
    actions, hidden = [], []
    carry = None
    for t in range(len(fused)):
        before = carry if pre_hidden is None else torch.as_tensor(pre_hidden[t][None,None], device=fused.device)
        a, carry = model.policy(fused[t:t+1][None], before)
        actions.append(arr(a[0,0])); hidden.append(arr(carry[0,0]))
    return np.asarray(actions), np.asarray(hidden)


def encode(model, norm, graphs, states, device):
    zs, ps = [], []
    for start in range(0,len(graphs),8):
        packed = pack_graphs(graphs[start:start+8]).to(device)
        z, presence = model.graph_encoder(packed)
        zs.append(z); ps.append(presence)
    z, presence = torch.cat(zs), torch.cat(ps)
    state = torch.tensor(states, device=device)
    return z, presence, torch.cat((z, presence, norm.apply(state)), -1)


def tensor_controls(model, norm, graphs, states, device, config):
    rng = np.random.default_rng(config['seed'])
    maxima = dict(repeat_action=0.,repeat_hidden=0.,permutation_z=0.,permutation_action=0.,permutation_hidden=0.)
    hidden = None
    for t, graph in enumerate(graphs):
        p = pack_graphs([graph], [(0,0)]).to(device)
        state = torch.tensor(states[t:t+1], device=device)
        z, _ = model.graph_encoder(p)
        a, h = model.step(p,state,norm,hidden)
        repeat, rh = model.step(p,state,norm,hidden)
        perm = pack_graphs([adapter.permute_graph(graph,rng)],[(0,0)]).to(device)
        pz, _ = model.graph_encoder(perm)
        pa, ph = model.step(perm,state,norm,hidden)
        for key,x,y in [('repeat_action',a,repeat),('repeat_hidden',h,rh),('permutation_z',z,pz),
                        ('permutation_action',a,pa),('permutation_hidden',h,ph)]:
            torch.testing.assert_close(x,y,atol=config['atol'],rtol=config['rtol'])
            maxima[key] = max(maxima[key],float((x-y).abs().max()))
        hidden = h
    return dict(status='PASS',frames=len(graphs),maximum_absolute_errors=maxima,
        permutation='all object node rows, including cross-object interleaving; identity-addressed robot rows remain 0,1,2; edge endpoints remapped')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with Path(path).open() as f:
        assert len(list(csv.DictReader(f))) == len(rows)


def source_audit(training, base, run):
    expected = json.loads((training/'source_hashes.json').read_text())
    sources = {}
    for folder in ('graph/integration_baseline','visual/runtime','recurrent'):
        for path in (base/folder).rglob('*.py'):
            if '__pycache__' not in path.parts:
                relative = str(path.relative_to(base)); target = run/'source_snapshot'/relative
                target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target); sources[relative] = sha256(path)
    for relative in ('graph/vision_input.py','graph/prototype/models.py','graph/prototype/v2/models.py',
                     'graph/prototype/v2/state.py','utils/experiment_logger.py','utils/experiment_artifacts.py'):
        path=base/relative;target=run/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);sources[relative]=sha256(path)
    # Current inference/evaluation AST must still be the training implementation.
    def nodes(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    before=nodes(training/'source_snapshot/graph/integration_baseline/direct_dataset.py')
    after=nodes(base/'graph/integration_baseline/direct_dataset.py')
    for key in ('DirectDataset','load_episode'):
        assert before[key]==after[key], 'training loader changed: '+key
    for relative in ('graph/integration_baseline/model.py','graph/integration_baseline/graph.py',
                     'graph/integration_baseline/robot.py','recurrent/models.py'):
        assert sources[relative]==expected[relative], 'training source changed: '+relative
    stage=Path(__file__).parent
    for path in stage.glob('point_resampling*'):
        if path.is_file():
            relative='graph/integration_baseline/'+path.name; target=run/'source_snapshot'/relative
            shutil.copy2(path,target);sources[relative]=sha256(path)
    for name in ('bootstrap.py','run.sh','preflight.json','tests.log'):
        if (stage/name).exists(): shutil.copy2(stage/name,run/name)
    test=stage/'test_point_resampling_stability.py'
    if test.exists():
        target=run/'source_snapshot/graph/integration_baseline/tests'/test.name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(test,target)
        sources['graph/integration_baseline/tests/'+test.name]=sha256(test)
    diff=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=base,text=True)
    (run/'implementation.diff').write_text(diff)
    added=[]
    for path in sorted(stage.glob('point_resampling*')):
        if path.is_file():
            content=path.read_text().splitlines()
            relative='graph/integration_baseline/'+path.name
            added.extend([f'diff --git a/{relative} b/{relative}','new file mode 100644','--- /dev/null',
                '+++ b/'+relative,f'@@ -0,0 +1,{len(content)} @@']+['+'+line for line in content])
    (run/'diagnostic_added_files.diff').write_text('\n'.join(added)+'\n')
    from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
    from mani_skill.utils import gym_utils
    for cls,name in [(PDEEPoseController,'installed_pd_ee_pose.py'),(gym_utils,'installed_gym_utils.py')]:
        path=Path(inspect.getfile(cls)); shutil.copy2(path,run/name);sources[name]=sha256(path)
    write_json(run/'source_hashes.json', sources)
    return dict(training_loader_ast_exact=True,core_model_graph_robot_exact=True,source_count=len(sources))


def build_episode(dataset, spec, config, run, protected):
    eid=spec['episode_id']; ep=dataset.episodes[eid]
    assert eid in dataset.splits[spec['split']]
    vision=next(e for e in dataset.graph_manifest['episodes'] if e['episode_id']==eid)
    raw_path=dataset.run/ep['path'];mask_path=dataset.output/vision['mask_path']
    assert sha256(raw_path)==ep['sha256'] and sha256(mask_path)==vision['mask_sha256']
    protected[str(raw_path)]=ep['sha256'];protected[str(mask_path)]=vision['mask_sha256']
    with np.load(mask_path,allow_pickle=False) as z:
        masks=z['masks']; np.testing.assert_array_equal(z['timestep'],np.arange(ep['observations']))
        assert int(z['episode_id'])==eid
    with h5py.File(raw_path,'r') as f:
        camera=dataset.graph_manifest['camera']
        rgb=f[camera+'/rgb'][()];depth=f[camera+'/depth'][()]
        intrinsic=f[camera+'/intrinsic'][()];cam2world=f[camera+'/cam2world'][()]
        tcp=f['tcp_pose'][()];timestamps=f['timestamp'][()]
    n=ep['actions']; assert masks.shape==(ep['observations'],2,*depth.shape[1:])
    assert list(sorted(t for (i,t) in dataset.packet_rows if i==eid))==list(range(ep['observations']))
    raw=dataset.raw(eid)
    assert not raw['terminated'][:n-1].any() and not raw['truncated'][:n-1].any()
    folder=run/f"{spec['split']}_{eid}";folder.mkdir();(folder/'graphs').mkdir()
    graphs={c:[] for c in config['conditions']}; metrics=[];local=[];states=[];targets=[];demos={}
    for t in range(n):
        aligned=dataset.frame(eid,t); packet=aligned.packet
        path=dataset.output/dataset.packet_rows[(eid,t)]['path'];protected[str(path)]=sha256(path)
        assert packet['observation_provenance']['rgb_sha256']==hashlib.sha256(rgb[t].tobytes()).hexdigest()
        frame=TrackedRGBDFrame(t,camera,rgb[t],depth[t],intrinsic[t],cam2world[t],tcp[t],
            {0:masks[t,0],1:masks[t,1]},t,float(timestamps[t]))
        evidence,prep,cases,maps=adapter.prepare(frame)
        saved_graph=build_graph(packet,aligned.robot_positions_ee,**config['graph'])
        frame_graphs={}; frame_local={}
        for condition in config['conditions']:
            built,replaced=adapter.build(frame,packet['episode'],evidence,prep,cases,maps,condition)
            built['observation_provenance']=packet['observation_provenance']
            graph=build_graph(built,aligned.robot_positions_ee,**config['graph'])
            frame_graphs[condition]=graph; graphs[condition].append(graph)
            if condition=='phase0':
                adapter.assert_packet_equal(packet,built)
                for key in ('node_input','node_type','edge_index','edge_attr','edge_type'):
                    assert torch.equal(getattr(saved_graph,key),getattr(graph,key)), 'Phase0 graph mismatch '+key
            else:
                comparison=adapter.sampling_metrics(frame_graphs['phase0'],graph,prep)
                metrics.append(dict(episode_id=eid,timestep=t,condition=condition,stage=ep['phases'][t]['before'],**comparison))
            if replaced:
                local.append(dict(episode_id=eid,timestep=t,condition=condition,objects=replaced))
                frame_local[condition]=replaced
            arrays=dict(source_index=built['provenance']['source_index'],points=built['points'],camera_points=built['camera_points'],
                object_id=built['object_id'],component=built['observed_component_id'],F5=built['geometric_features'],
                available=built['feature_field_available'],valid=built['feature_field_valid'],
                surface=built['provenance']['surface_selected'],node_input=arr(graph.node_input),
                edge_index=arr(graph.edge_index),edge_attr=arr(graph.edge_attr),edge_type=arr(graph.edge_type))
            save_npz(folder/'graphs'/f'{t:04d}_{condition}.npz',**arrays)
        if t in config['impulse_timesteps']:
            demos[t]=dict(rgb=rgb[t],depth=depth[t],graphs=frame_graphs)
        states.append(aligned.robot_state); targets.append(aligned.action)
        if t%25==0:print(f'BUILD ep={eid} t={t}/{n} phase0 exact',flush=True)
    # Protect final unlabelled packet too, without evaluating a nonexistent target.
    final=dataset.output/dataset.packet_rows[(eid,n)]['path'];protected[str(final)]=sha256(final)
    write_json(folder/'sampling_comparison.json',metrics);write_json(folder/'local_replacement.json',local)
    return dict(spec=spec,folder=folder,graphs=graphs,states=np.asarray(states),expert=np.asarray(targets),
        metrics=metrics,local=local,demos=demos,stages=np.asarray([p['before'] for p in ep['phases']]),
        inputs=dict(raw=str(raw_path),raw_sha256=ep['sha256'],mask=str(mask_path),mask_sha256=vision['mask_sha256'],
            actions=n,observations=ep['observations'],mask_provenance=vision['mask_provenance']))


def recovery(values, impulse, fraction, floor):
    tail=np.asarray(values)[impulse:]
    threshold=max(float(tail[0])*fraction,floor)
    # Earliest frame after which every remaining frame is below threshold.
    exceed=np.flatnonzero(tail>threshold)
    offset=int(exceed[-1]+1) if len(exceed) else 0
    return dict(initial=float(tail[0]),peak=float(tail.max()),threshold=threshold,
        stable_below_after_steps=offset if offset<len(tail) else None,
        observed_tail_steps=len(tail),tail_value=float(tail[-1]),censored=bool(offset>=len(tail)))


def analyze_episode(model,norm,built,config,device):
    folder=built['folder'];n=len(built['expert']);data=dict(expert=built['expert'],robot76=built['states'],stage=built['stages'])
    z,p,fused=encode(model,norm,built['graphs']['phase0'],built['states'],device)
    baseline,ha=recurrent(model,fused)
    pre=np.r_[np.zeros((1,ha.shape[1]),np.float32),ha[:-1]]
    data.update(z_phase0=arr(z),presence_phase0=arr(p),action_phase0=baseline,hidden_phase0=ha,pre_hidden_phase0=pre,
        normalized_robot76=arr(norm.apply(torch.tensor(built['states'],device=device))))
    whole,_=model.policy(fused[None]);np.testing.assert_allclose(arr(whole[0]),baseline,atol=config['atol'],rtol=config['rtol'])
    summary={};rows=[];impulses=[]
    for condition in config['conditions'][1:]:
        zb,pb,fb=encode(model,norm,built['graphs'][condition],built['states'],device)
        assert torch.equal(p,pb),'object presence changed'
        instant,hi=recurrent(model,fb,pre)
        total,hb=recurrent(model,fb)
        delta=arr(zb-z); za=arr(z);zba=arr(zb)
        data.update({f'z_{condition}':zba,f'presence_{condition}':arr(pb),f'action_instant_{condition}':instant,
            f'action_history_{condition}':total,f'hidden_instant_{condition}':hi,f'hidden_history_{condition}':hb})
        measures={}
        for mode,a,b in [('instant',baseline,instant),('history',baseline,total),('history_extra',instant,total)]:
            values=action_metrics(a,b,built['expert'])
            measures[mode]=values
            for k,v in values.items():data[f'{condition}_{mode}_{k}']=v
        zl2=np.linalg.norm(delta,axis=1);rel=zl2/np.maximum(np.linalg.norm(za,axis=1),1e-12)
        slots=np.linalg.norm(delta.reshape(n,5,64),axis=2)
        slot_rel=slots/np.maximum(np.linalg.norm(za.reshape(n,5,64),axis=2),1e-12)
        slot_cos=cosine(za.reshape(n,5,64),zba.reshape(n,5,64))
        hl2=np.linalg.norm(hb-ha,axis=1);hc=cosine(ha,hb)
        data.update({f'{condition}_z_l2':zl2,f'{condition}_z_relative_l2':rel,f'{condition}_z_cosine':cosine(za,zba),
            f'{condition}_slot_l2':slots,f'{condition}_slot_relative_l2':slot_rel,f'{condition}_slot_cosine':slot_cos,
            f'{condition}_hidden_l2':hl2,f'{condition}_hidden_cosine':hc})
        first=np.flatnonzero(hl2>config['atol'])
        geometry=[r for r in built['metrics'] if r['condition']==condition]
        for t in range(n):
            gm=geometry[t]
            row=dict(episode_id=built['spec']['episode_id'],split=built['spec']['split'],timestep=t,stage=built['stages'][t],
                condition=condition,z_l2=zl2[t],z_relative_l2=rel[t],z_cosine=data[f'{condition}_z_cosine'][t],
                hidden_l2=hl2[t],hidden_cosine=hc[t],presence_peg=int(p[t,0]),presence_box=int(p[t,1]),
                edge_count_a=sum(gm['edge_counts_a']),edge_count_b=sum(gm['edge_counts_b']))
            for oid,name in [(0,'peg'),(1,'box')]:
                obj=gm['objects'][oid]
                for key in ('count_a','count_b','common','source_jaccard','structural_a','structural_b','surface_a','surface_b',
                            'chamfer_mean_m','nn_p95_m','nn_max_m','voxel_jaccard','voxel_coverage_a','voxel_coverage_b','raw_ede_jaccard'):
                    row[name+'_'+key]=obj[key]
            for j,name in enumerate(('tcp','left','right','peg_mean','box_mean')):
                row[name+'_z_l2']=slots[t,j];row[name+'_z_relative_l2']=slot_rel[t,j];row[name+'_z_cosine']=slot_cos[t,j]
            for mode,values in measures.items():
                for key,value in values.items():row[mode+'_'+key]=value[t]
            rows.append(row)
        error=measures['history']['expert_mse_delta']
        summary[condition]=dict(frames=n,z_l2=stats(zl2),z_relative_l2=stats(rel),z_cosine=stats(cosine(za,zba)),
            slot_relative_l2=[stats(slot_rel[:,j]) for j in range(5)],hidden_l2=stats(hl2),hidden_cosine=stats(hc),
            first_hidden_difference=int(first[0]) if len(first) else None,
            hidden_growth_steps=int((np.diff(hl2)>config['atol']).sum()),hidden_decline_steps=int((np.diff(hl2)<-config['atol']).sum()),
            modes={mode:{k:stats(v) for k,v in values.items()} for mode,values in measures.items()},
            expert_error_increased=int((error>config['atol']).sum()),expert_error_decreased=int((error < -config['atol']).sum()),
            expert_error_unchanged=int((np.abs(error)<=config['atol']).sum()),
            first_negative_gripper_a=next((int(t) for t in range(n) if baseline[t,6]<0),None),
            first_negative_gripper_b=next((int(t) for t in range(n) if total[t,6]<0),None),
            gripper_switches_a=np.flatnonzero(np.diff(baseline[:,6]<0)).tolist(),
            gripper_switches_b=np.flatnonzero(np.diff(total[:,6]<0)).tolist())
        # Every predeclared impulse, separately for every alternative.
        for t0 in config['impulse_timesteps']:
            if t0>=n:
                impulses.append(dict(condition=condition,timestep=t0,status='NOT_TESTED_SHORT_EPISODE'));continue
            fi=fused.clone();fi[t0]=fb[t0]
            ai,h=recurrent(model,fi)
            np.testing.assert_array_equal(ai[:t0],baseline[:t0]);np.testing.assert_array_equal(h[:t0],ha[:t0])
            am=action_metrics(baseline,ai,built['expert']);hd=np.linalg.norm(h-ha,axis=1)
            data[f'impulse_{condition}_{t0}_action']=ai;data[f'impulse_{condition}_{t0}_hidden']=h
            data[f'impulse_{condition}_{t0}_hidden_l2']=hd
            data[f'impulse_{condition}_{t0}_command_translation_mm']=am['command_translation_mm']
            data[f'impulse_{condition}_{t0}_command_rotation_rad']=am['command_rotation_rad']
            data[f'impulse_{condition}_{t0}_command_gripper_mm']=am['command_gripper_mm']
            impulses.append(dict(label='SINGLE_FRAME_IMPULSE',condition=condition,timestep=t0,status='PASS',
                hidden=recovery(hd,t0,config['recovery_relative_fraction'],config['atol']),
                translation_mm=recovery(am['command_translation_mm'],t0,config['recovery_relative_fraction'],1e-3),
                rotation_rad=recovery(am['command_rotation_rad'],t0,config['recovery_relative_fraction'],1e-6),
                gripper_mm=recovery(am['command_gripper_mm'],t0,config['recovery_relative_fraction'],1e-3)))
    natural=np.linalg.norm(np.diff(arr(z),axis=0),axis=1)
    data['natural_z_l2']=np.r_[np.nan,natural]
    data['natural_z_cosine']=np.r_[np.nan,cosine(arr(z)[:-1],arr(z)[1:])]
    data['natural_slot_l2']=np.r_[np.full((1,5),np.nan),np.linalg.norm(np.diff(arr(z).reshape(n,5,64),axis=0),axis=2)]
    presence=arr(p).astype(bool)
    transition=np.zeros(n,bool)
    for t in np.flatnonzero(np.any(presence[1:]!=presence[:-1],axis=1))+1:
        transition[max(0,t-2):min(n,t+3)]=True
    for row in rows:
        row['near_presence_transition']=bool(transition[row['timestep']])
    for key,value in data.items():
        if key.startswith(('action_','hidden_','z_','pre_hidden','robot76','normalized_robot','expert')):
            assert np.isfinite(value).all(),'nonfinite model tensor '+key
    # Different-timestep source IDs are image pixels, not tracked physical points.
    natural_graph=[]
    for t in range(1,n):
        ga,gb=built['graphs']['phase0'][t-1:t+1]
        pa,pb=ga.packet,gb.packet
        natural_graph.append(dict(timestep=t,point_counts_previous=[int((pa['object_id']==o).sum()) for o in (0,1)],
            point_counts_current=[int((pb['object_id']==o).sum()) for o in (0,1)],
            edge_counts_previous=torch.bincount(ga.edge_type,minlength=3).tolist(),edge_counts_current=torch.bincount(gb.edge_type,minlength=3).tolist(),
            source_pixel_jaccard=[adapter.jaccard(set(pa['provenance']['source_index'][pa['object_id']==o]),set(pb['provenance']['source_index'][pb['object_id']==o])) for o in (0,1)],
            edge_source_pixel_jaccard=[adapter.jaccard(adapter.edge_keys(ga,k),adapter.edge_keys(gb,k)) for k in range(3)]))
    write_json(folder/'natural_graph_changes.json',natural_graph)
    save_npz(folder/'tensors.npz',**data)
    csv_write(folder/'frames.csv',rows)
    write_json(folder/'summary.json',summary);write_json(folder/'impulses.json',impulses)
    from graph.integration_baseline.point_resampling_report import episode_plots
    episode_plots(built,data,rows,impulses,config)
    return dict(**built['spec'],frames=n,conditions=summary,impulses=impulses,natural_z_l2=stats(natural)),rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--checkpoint-manifest',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('point_resampling_config.json'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    config=json.loads(args.config.read_text());torch.set_num_threads(1);torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    train=json.loads((args.training_run/'config.yaml').read_text());config['graph']=train['graph']
    base=Path(inspect.getfile(build_graph)).resolve().parents[2]
    protected={str(args.training_run/name):sha256(args.training_run/name) for name in
        ('config.yaml','dataset_manifest.json','split_manifest.json','vision_manifest.json','robot_state_schema.json','status.json','training/best.json')}
    with ExperimentLogger(team='integration',model='point_resampling_stability',seed=config['seed'],job_type='evaluate',
        config=dict(config,device=args.device,training_run=str(args.training_run)),root=args.output_root) as logger:
        run=logger.directory;print('RUN_DIR='+str(run),flush=True)
        write_json(run/'selection.json',dict(episodes=config['episodes'],selection_before_predictions=True,full_valid_action_sequence=True))
        if args.device=='cuda':
            if torch.cuda.mem_get_info()[0]<config['minimum_cuda_free_bytes']:
                raise RuntimeError('BLOCKED_RESOURCE: no existing jobs will be stopped')
            torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
        model,norm,provenance=load_bundle(args.checkpoint_manifest,device=args.device)
        assert provenance['manifest']['files']['checkpoint']['sha256']==config['checkpoint_sha256']
        assert provenance['progress']['global_step']==8959
        assert not model.training
        for path,item in provenance['manifest']['files'].items():
            protected[str(args.checkpoint_manifest.parent/item['path'])]=item['sha256']
        dataset=DirectDataset(args.training_run);assert dataset.schema==norm.schema
        assert dataset.manifest['dataset_id']=='d1fd6173cb3648bfa4c03507f84b7e71'
        train_counts=np.asarray([p['object_counts'] for e in dataset.graph_manifest['episodes']
            if e['episode_id'] in dataset.splits['train'] for p in e['packets'] if p['has_action_target']])
        write_json(run/'training_point_count_reference.json',dict(scope='all original train action-frame packet manifest counts',
            episodes=len(dataset.splits['train']),frames=len(train_counts),
            peg=stats(train_counts[:,0]),box=stats(train_counts[:,1]),
            note='structural/Surface and F5 distributions use the two frozen train episodes separately'))
        audit=source_audit(args.training_run,base,run)
        write_json(run/'provenance.json',dict(checkpoint=provenance['manifest'],identity=provenance['identity'],progress=provenance['progress'],
            source_audit=audit,device=args.device,torch=torch.__version__,python=platform.python_version(),
            gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,simulator_executed=False,
            dino_sam_inference_executed=False,controller='installed preprocessing and XYZ matrix conversion; config-only stubs'))
        save_npz(run/'normalizer.npz',mean=arr(norm.stats.mean),std=arr(norm.stats.std),fitted_episode_ids=np.asarray(norm.fitted_episode_ids))
        write_json(run/'normalizer_hash.json',dict(sha256=sha256(run/'normalizer.npz'),tensor_digest=digest(arr(norm.stats.mean),arr(norm.stats.std)),train_only=True))
        weights=digest(*(arr(p) for p in model.parameters()));normalizer=digest(arr(norm.stats.mean),arr(norm.stats.std))
        built=[];blocked=[]
        for spec in config['episodes']:
            # Missing actual observation/mask support is a blocker, never inferred from selected source IDs.
            try:
                built.append(build_episode(dataset,spec,config,run,protected))
            except (FileNotFoundError,KeyError) as exc:
                blocked.append(dict(**spec,status='BLOCKED_INPUT',reason=str(exc)))
        write_json(run/'input_manifest.json',dict(episodes=[dict(**e['spec'],**e['inputs']) for e in built],blocked=blocked))
        controls=[]
        with torch.inference_mode():
            for item in built:
                print('CONTROLS',item['spec'],flush=True)
                controls.append(dict(**item['spec'],phase0_exact_all_frames=True,
                    **tensor_controls(model,norm,item['graphs']['phase0'],item['states'],args.device,config)))
            write_json(run/'negative_controls.json',dict(status='PASS' if built and not blocked else 'PARTIAL',episodes=controls,
                source_support_component_f5_raw_ede='PASS for every built frame/condition via existing validation and dense source gather',
                robot76_normalizer_timestep_episode='PASS; fixed source, exact Robot76; zero initial carry per episode'))
            # No sensitivity outputs are computed until every available episode passes all controls.
            summaries=[];rows=[]
            for item in built:
                print('ANALYZE',item['spec'],flush=True)
                summary,frames=analyze_episode(model,norm,item,config,args.device)
                summaries.append(summary);rows.extend(frames)
                logger.log({'frames':summary['frames']},step=item['spec']['episode_id'])
        assert weights==digest(*(arr(p) for p in model.parameters()))
        assert normalizer==digest(arr(norm.stats.mean),arr(norm.stats.std))
        for path,expected in protected.items():assert sha256(path)==expected,'protected input changed: '+path
        write_json(run/'input_hashes.json',protected)
        write_json(run/'protection.json',dict(status='PASS',files=len(protected),weights_and_normalizer_unchanged=True))
        csv_write(run/'frames.csv',rows)
        results=dict(status='COMPLETE' if len(built)==4 and not blocked else 'PARTIAL_BLOCKED_INPUT',run_id=logger.run_id,
            episodes=summaries,blocked=blocked,frames=sum(e['frames'] for e in summaries),
            scope=config['scope'],checkpoint_sha256=config['checkpoint_sha256'])
        write_json(run/'results.json',results)
        write_json(run/'graph_artifact_index.json',dict(files={str(p.relative_to(run)):sha256(p)
            for p in sorted(run.glob('*/graphs/*.npz'))},description='Rebuilt arrays for every frame and alternative; all NPZ read back'))
        from graph.integration_baseline.point_resampling_report import finalize_report,verify
        finalize_report(run,results,rows,config)
        verification=verify(run)
        write_json(run/'verification.json',verification)
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.name not in ('metadata.json','manifest.json','metrics.json','config.yaml') and not path.name.endswith('.tracking.json'):
                if 'graphs' not in path.parts and 'source_snapshot' not in path.parts:
                    logger.link_output(path,kind='reports' if path.suffix in ('.html','.md','.png') else 'metrics')
        logger.record(source_run_id=provenance['manifest']['source_run_id'],training_pipeline_run_id=args.training_run.name,
            sampling_diagnostic_status=results['status'],frames=results['frames'],verification=verification,
            normalizer_digest=normalizer,model_weight_digest=weights,publishing_requested=False)
        print('COMPLETE '+str(run),flush=True)


if __name__=='__main__':main()
