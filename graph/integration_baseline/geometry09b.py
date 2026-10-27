"""Task09-B: isolated frozen-readout geometry probes. No student actions/training.

Object GT is used only for target/audit. The existing RGB-D -> predicted masks
-> Shape-First/F5/Raw EDE -> graph path is the sole encoder input.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import time

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from graph.integration_baseline import geometry09a as a
from graph.integration_baseline.direct_dataset import DirectDataset
from graph.integration_baseline.direct_runtime import read, sha
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import read_packet, save_npz, save_packet, write_json
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.rollout_stop_clock import observation, numpy, PhysicsAudit, reset_diagnostics
from utils.experiment_logger import ExperimentLogger

TASK09A = Path('/workspace/task09a/experiments/073de7b8e59d4f23b650c87c377c1829')
MODELS = ['Full_B','Full_C','Single_B','Single_C','Full_A','Single_A']
MEMORY_KEYS = ('states','found','rgb','images','records','detector_calls','scan_events')


def config(plan):
    return dict(task='09-B',seed=42,train=[s for s in plan['splits']['train'] if s!=30003][:16],
        validation=plan['splits']['validation'][:6],final=[30003],timesteps=[0,10,25],
        intervention=a.protocol()['primary'][:10],
        trajectory='fresh live ScriptedExpert actions; stored dataset prefix read-only comparison, not assumed identical',
        selection='first 16/6 in immutable split order, exclude30003; before collection/results',
        validity=dict(min_visible_pixels=20,min_peg_points=10,min_selected_gt_purity=.5,
            primary='VISIBLE_VALID peg only; box contamination separately recorded',
            order='visible<20:OCCLUDED; points=0:VISIBLE_MISSING; points<10 or purity<.5:GEOMETRY_CONTAMINATED; else VISIBLE_VALID',
            pair_support='both VISIBLE_VALID and physical valid; response support separately uses09A projected centroid>=1mm'),
        target='R_TCP.T @ (t_PegActorOrigin - t_TCP); metres float32[3]; wxyz quaternion',
        linear=dict(alphas=[0.,.01,.1,1.,10.,100.],objective='mean normalized target squared error + alpha/n * weight squared',selection='minimum validation MSE, first tie'),
        mlp=dict(hidden=64,activation='ReLU',lr=.001,weight_decay=[.0001,.01],max_epochs=2000,
            patience=150,min_delta=1e-8,batch='full',optimizer='AdamW',selection='minimum validation MSE, first tie',
            target_standardization='train mean/std floor .001m; identical for linear and MLP'),
        feature_standardization='train VISIBLE_VALID only; population std floor1e-6; never policy normalizer/readout normalization',
        primary_train_filter='physical valid VISIBLE_VALID, baseline + XY primary; repeat excluded',
        shuffle='fixed seed43 permutation of train target rows; same validation selection and budget; all P0-P4 and models',
        fixed_capacity=True,policy_updates=False,gru_probe='NOT_TESTED',rotation_probe='NOT_TESTED',
        snapshot='deep-copy causal tracker memory before current frame; same prefix, no predictor weights copied; smoke compares independent reset/replayed prefix',
        no_publish=True,source09a=str(TASK09A),device='RTX3090 frozen encoder; CPU PhysX/render; CPU diagnostic decoders')


def target(peg, tcp):
    peg=np.asarray(peg);tcp=np.asarray(tcp)
    r=Rotation.from_quat(tcp[[4,5,6,3]]).as_matrix()
    return (r.T@(peg[:3]-tcp[:3])).astype(np.float32)


def classify(support, points, c):
    if support['visible_gt_pixels']<c['min_visible_pixels']:return 'OCCLUDED'
    if points==0:return 'VISIBLE_MISSING'
    if points<c['min_peg_points'] or support['selected_gt_purity']<c['min_selected_gt_purity']:
        return 'GEOMETRY_CONTAMINATED'
    return 'VISIBLE_VALID'


def geometry_features(xyz):
    """Student point XYZ only. Missing rows are zero; presence stored separately."""
    xyz=np.asarray(xyz)
    if len(xyz)==0:return np.zeros(15,np.float32)
    lo,hi=xyz.min(0),xyz.max(0)
    return np.concatenate([xyz.mean(0),lo,hi,(lo+hi)/2,hi-lo]).astype(np.float32)


def memory(vision):
    return deepcopy({k:getattr(vision,k) for k in MEMORY_KEYS})


def restore(vision,state):
    for k,v in deepcopy(state).items():setattr(vision,k,v)


def weight_hash(model, prefix=''):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        if k.startswith(prefix):h.update(k.encode()+a.digest(numpy(v)).encode())
    return h.hexdigest()


def freeze(models):
    out={}
    for name,(m,_,_) in models.items():
        out[name]=dict(all=weight_hash(m),gnn=weight_hash(m,'graph_encoder.'),
            downstream=weight_hash(m,'policy.'),no_requires_grad=all(not p.requires_grad for p in m.parameters()),
            no_grad=all(p.grad is None for p in m.parameters()),eval=not m.training)
    return out


@torch.inference_mode()
def encode(models,graph):
    batch=pack_graphs([graph]).to('cuda');out={}
    for label,(model,_,_) in models.items():
        z,p=model.graph_encoder(batch);zr,pr=model.graph_encoder(batch)
        assert torch.equal(z,zr) and torch.equal(p,pr),'repeat readout'
        assert not z.requires_grad and z.grad_fn is None
        out[label]=dict(readout=numpy(z)[0].copy(),presence=numpy(p)[0].copy())
    return out


def packet_graph(row,masks,t,seed,plan):
    from visual.runtime.rgbd import TrackedRGBDFrame
    from graph.integration_baseline.vision import packet_from_frame
    identity=f'seed{seed}'
    f=TrackedRGBDFrame(t,'base_camera',row['rgb'],row['depth'],row['intrinsic'],row['cam2world'],
        row['tcp_pose'],{0:masks[0],1:masks[1]},t,t/20)
    packet,_=packet_from_frame(f,identity,keep_ratio=.5)
    packet['observation_provenance']=dict(seed=seed,timestep=t,gt_used_as_student_input=False,
        future_observations_used=False,mask_source='causal RGB-only DINO SAM2')
    return packet,build_graph(packet,row['robot_ee_geometry'],**plan['config']['graph'])


def audit_source(ds, seeds):
    rows=[]
    for seed in seeds:
        ep=ds.episodes[seed];p=ds.run/ep['path'];assert sha(p)==ep['sha256']
        with h5py.File(p,'r') as f:
            keys=[];f.visititems(lambda k,v:keys.append(k) if isinstance(v,h5py.Dataset) else None)
            assert 'peg_pose' not in keys
            rows.append(dict(seed=seed,path=str(p),sha256=sha(p),keys=keys,
                reset_group=ep['reset_group'],gt_peg_pose_present=False))
    return dict(source=str(ds.run),episodes=rows,conclusion='No timestep Peg actor pose. No labels inferred from Robot76/actions/reset hash.')


def extract_sample(root,folder,row,packet,graph,physical,support,outputs,seed,t,spec,split,c):
    folder.mkdir(parents=True,exist_ok=False)
    save_packet(folder/'packet.npz',packet)
    save_npz(folder/'graph.npz',**{n:numpy(getattr(graph,n)) for n in
        ('node_input','node_type','edge_index','edge_attr','edge_type')})
    state=physical['after'];y=target(state['peg_pose'],state['ee_pose'])
    np.testing.assert_array_equal(np.asarray(row['tcp_pose']),np.asarray(state['ee_pose'],np.float32))
    xyz=np.asarray(packet['points'])[np.asarray(packet['object_id'])==0]
    np.testing.assert_array_equal(xyz,numpy(graph.node_input)[numpy(graph.node_type)==3,:3])
    raw=geometry_features(xyz)
    save_npz(folder/'features.npz',robot76=row['robot_state'],target_peg_position_tcp=y,raw_geometry15=raw,
        **{label+'_readout320':v['readout'] for label,v in outputs.items()},
        **{label+'_slots5x64':v['readout'].reshape(5,64) for label,v in outputs.items()},
        **{label+'_presence2':v['presence'] for label,v in outputs.items()})
    write_json(folder/'physical.json',physical)
    meta=dict(sample_id=str(folder.relative_to(root)),seed=seed,episode_group=seed,t=t,condition=spec['name'],
        group=spec['group'],split=split,physical_valid=physical['valid'],support=support,
        peg_points=len(xyz),validity=classify(support[0],len(xyz),c['validity']),
        box_validity=classify(support[1],int((np.asarray(packet['object_id'])==1).sum()),c['validity']),
        graph_sha256=sha(folder/'graph.npz'),packet_sha256=sha(folder/'packet.npz'),
        feature_sha256=sha(folder/'features.npz'),gt_source='live GT actor world pose, same observation, sidecar only',
        graph_has_gt=False,repeat_exact=True)
    write_json(folder/'sample.json',meta)
    return meta


@torch.inference_mode()
def collect_episode(root,plan,models,vision,ds,seed,split,c,smoke=False):
    from mani_skill.utils.structs.pose import Pose
    identity=f'seed{seed}';ep=ds.episodes[seed];path=ds.run/ep['path']
    with h5py.File(path,'r') as f:
        actions=f['actions'][:26];saved_robot=f['robot_state'][:26];saved_tcp=f['tcp_pose'][:26]
    env,conditions=a.environment(plan);obs,_=env.reset(seed=seed)
    actual_reset_group=reset_diagnostics(env)
    physics=PhysicsAudit(env);audit=a.controller_audit(env);vision.reset(identity)
    expert=a.ScriptedExpert(env);prefix=[];samples=[];replay=[];smoke_rows=[]
    schema=next(iter(models.values()))[1].schema
    try:
        for t in range(26):
            row=observation(env,obs,schema,'base_camera');before=a.physical_state(env)
            replay.append(dict(t=t,robot_max=float(abs(saved_robot[t]-row['robot_state']).max()),
                tcp_max=float(abs(saved_tcp[t]-row['tcp_pose']).max())))
            token=physics.token()
            if t not in c['timesteps']:
                masks,_=vision.process(row['rgb'],identity,t)
            else:
                pre=memory(vision);oldpose=env.unwrapped.peg.pose.raw_pose.clone()
                oldcollision=a.collision_audit(env,'peg');baseline_output=None;post=None
                for spec in c['intervention']:
                    restore(vision,pre)
                    raw=oldpose.clone();raw[0,:3]+=raw.new_tensor(spec['xyz_mm'])*.001
                    env.unwrapped.peg.set_pose(Pose.create(raw));o=env.unwrapped.get_obs()
                    altrow=observation(env,o,schema,'base_camera');after=a.physical_state(env)
                    exact={k:np.array_equal(before[k],after[k]) for k in before if k!='peg_pose'}
                    exact['robot76']=np.array_equal(row['robot_state'],altrow['robot_state'])
                    exact['intrinsic']=np.array_equal(row['intrinsic'],altrow['intrinsic'])
                    exact['cam2world']=np.array_equal(row['cam2world'],altrow['cam2world'])
                    coll=a.collision_audit(env,'peg')
                    new=set(coll['near_contacts'])-set(oldcollision['near_contacts'])
                    valid=all(exact.values()) and coll['valid'] and not new
                    np.testing.assert_allclose(after['peg_pose'][:3]-before['peg_pose'][:3],np.asarray(spec['xyz_mm'])*.001,atol=3e-8,rtol=0)
                    masks,record=vision.process(altrow['rgb'],identity,t)
                    packet,graph=packet_graph(altrow,masks,t,seed,plan)
                    gt_masks,support=a.geometry_sidecar(env,packet,masks)
                    outputs=encode(models,graph)
                    folder=root/'collection'/identity/f't{t:02d}'/spec['name']
                    physical=dict(valid=bool(valid),exact=exact,before=before,after=after,
                        new_near_contacts=sorted(new),collision_before=oldcollision,collision_after=coll,
                        requested_offset_mm=spec['xyz_mm'],conditions=conditions,geometry_support=support,
                        no_physics_step=True)
                    sample=extract_sample(root,folder,altrow,packet,graph,physical,support,outputs,seed,t,spec,split,c)
                    save_npz(folder/'observation.npz',**{k:altrow[k] for k in
                        ('rgb','depth_raw_mm','robot_state','intrinsic','cam2world','tcp_pose','robot_ee_geometry')},masks=masks,gt_masks=gt_masks)
                    write_json(folder/'vision.json',vision.records)
                    if spec['name']=='baseline':
                        baseline_output=outputs;baseline_masks=masks.copy();post=memory(vision)
                        baseline_jpg=(Path(vision.directory.name)/f'{t:06d}.jpg').read_bytes()
                    if spec['name']=='baseline_repeat':
                        assert np.array_equal(masks,baseline_masks)
                        for l in models:np.testing.assert_array_equal(outputs[l]['readout'],baseline_output[l]['readout'])
                    samples.append(sample);physics.unchanged(token)
                env.unwrapped.peg.set_pose(Pose.create(oldpose));obs=env.unwrapped.get_obs()
                np.testing.assert_array_equal(a.physical_state(env)['peg_pose'],before['peg_pose'])
                restore(vision,post);(Path(vision.directory.name)/f'{t:06d}.jpg').write_bytes(baseline_jpg)
                # Independent reset/prefix for contract only. No labels enter probes.
                if smoke and t==10:
                    saved=memory(vision);rgbs=deepcopy(vision.rgb)
                    vision.reset(identity)
                    for ti,rgb in enumerate(rgbs):mm,_=vision.process(rgb,identity,ti)
                    np.testing.assert_array_equal(mm,baseline_masks)
                    pp,gg=packet_graph(row,mm,t,seed,plan);oo=encode(models,gg)
                    for l in models:np.testing.assert_array_equal(oo[l]['readout'],baseline_output[l]['readout'])
                    # reset() removed JPEG prefix: recreate observed prefix only.
                    import cv2
                    restore(vision,saved)
                    for ti,rgb in enumerate(rgbs):cv2.imwrite(str(Path(vision.directory.name)/f'{ti:06d}.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    smoke_rows.append(dict(t=t,independent_tracker_prefix_masks_readout_exact=True))
            prefix.append(dict(rgb=row['rgb'],depth_raw_mm=row['depth_raw_mm'],robot76=row['robot_state'],
                peg_pose=before['peg_pose'],tcp_pose=before['ee_pose']))
            physics.unchanged(token)
            if t<25:
                live=expert.act();replay[-1]['expert_action_max']=float(abs(live-actions[t]).max())
                audit.clear();obs,*_=env.step(torch.from_numpy(live[None]).to(env.unwrapped.device))
                physics.stepped(token);a.validate_controller_command(live,audit)
        epfolder=root/'collection'/identity
        save_npz(epfolder/'prefix_observations.npz',**{k:np.stack([p[k] for p in prefix]) for k in prefix[0]})
        write_json(epfolder/'replay_audit.json',dict(seed=seed,source_hdf5_sha256=sha(path),
            reset_identity_exact=actual_reset_group==ep['reset_group'],actual_reset_group=actual_reset_group,
            source_reset_group=ep['reset_group'],frames=replay,claim='LIVE_DIAGNOSTIC; fresh live ScriptedExpert, not identical stored observations',
            stored_observation_robot_tcp_exact=all(r['robot_max']==0 and r['tcp_max']==0 for r in replay)))
        if smoke:
            assert len(smoke_rows)==1 and sum(s['physical_valid'] for s in samples)>=3
            write_json(root/'collection_contract.json',dict(pass_gate=True,checks=smoke_rows,
            valid_samples=sum(s['physical_valid'] for s in samples),samples=len(samples),
            same_robot_same_causal_prefix=True,no_student_step=True,no_physics_after_teleport=True,
            note='pose restoration then live expert action; independent tracker prefix at t10'))
        return samples
    finally:physics.close();env.close();vision.reset(None)


def import_final(root,c):
    """Read-only 09A final evaluation, never fit or select on these rows."""
    rows=[]
    for trial in read(TASK09A/'trials.json'):
        folder=TASK09A/trial['directory'];physical=read(folder/'physical.json')
        with np.load(folder/'observation.npz') as d:robot=d['robot_state'].copy();tcp=d['tcp_pose'].copy()
        packet=read_packet(folder/'packet.npz')
        with np.load(folder/'graph.npz') as g:xyz=g['node_input'][g['node_type']==3,:3]
        y=target(physical['after']['peg_pose'],physical['after']['ee_pose'])
        np.testing.assert_array_equal(tcp,np.asarray(physical['after']['ee_pose'],np.float32))
        dest=root/'final09a'/trial['directory'];dest.mkdir(parents=True,exist_ok=False)
        arrays=dict(robot76=robot,target_peg_position_tcp=y,raw_geometry15=geometry_features(xyz))
        for l in MODELS:
            with np.load(folder/(l+'.npz')) as f:
                arrays[l+'_readout320']=f['readout'].reshape(320)
                arrays[l+'_slots5x64']=f['readout'].reshape(5,64)
                arrays[l+'_presence2']=f['presence'].reshape(2)
        save_npz(dest/'features.npz',**arrays)
        s=physical['geometry_support']
        meta=dict(sample_id=str(dest.relative_to(root)),seed=30003,episode_group=30003,t=trial['t'],
            condition=trial['name'],group=trial['group'],split='final',physical_valid=trial['status']=='VALID',
            support=s,peg_points=len(xyz),validity=classify(s[0],len(xyz),c['validity']),
            box_validity=classify(s[1],trial['point_counts'][1],c['validity']),graph_sha256=sha(folder/'graph.npz'),
            packet_sha256=sha(folder/'packet.npz'),feature_sha256=sha(dest/'features.npz'),
            gt_source=str(folder/'physical.json'),source_directory=str(folder),graph_has_gt=False,repeat_exact=True)
        write_json(dest/'sample.json',meta);rows.append(meta)
    return rows


def validate_splits(c,ds):
    assert len(c['train'])==16 and len(c['validation'])==6
    groups=[]
    for part in ('train','validation','final'):
        ids=c[part];assert len(set(ids))==len(ids)
        groups.append({ds.episodes[s]['reset_group'] for s in ids})
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    assert set(c['train'])<=set(ds.splits['train']) and set(c['validation'])<=set(ds.splits['validation'])
    return dict(splits={part:[dict(seed=s,reset_group=ds.episodes[s]['reset_group']) for s in c[part]]
        for part in ('train','validation','final')},selection=c['selection'],pass_gate=True,
        note='Probe-heldout episodes; Full encoders previously trained on Train96. Single encoders trained on30003, so final is NOT unseen to original encoders.')


def snapshot(root):
    base=Path(__file__).resolve().parents[2];out=root/'source_snapshot';out.mkdir();inventory={}
    for part in ('envs','graph','recurrent','visual/runtime','utils','tools'):
        for p in (base/part).rglob('*.py'):
            if '.venv' in p.parts or 'cache' in p.parts:continue
            dest=out/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            inventory[str(p.relative_to(base))]=sha(p)
    write_json(root/'source_hashes.json',inventory)
    diffs=[]
    for name in ('geometry09b.py','geometry09b_probe.py','geometry09b_report.py','tests/test_geometry09b.py'):
        p=base/'graph/integration_baseline'/name
        if p.exists():diffs.append(subprocess.run(['git','diff','--no-index','/dev/null',str(p)],capture_output=True,text=True).stdout)
    (root/'implementation.diff').write_text('\n'.join(diffs))


def run(output):
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan,models,identities,protected=a.load_models()
    for m,_,_ in models.values():
        m.eval();m.requires_grad_(False)
        for p in m.parameters():p.grad=None
    before=freeze(models)
    assert identities==read(TASK09A/'identities.json'),'09A identities exact'
    ds=DirectDataset(plan['dataset_directory']);c=config(plan);split=validate_splits(c,ds)
    # Existing09A artifact tree protected in its entirety, no checkpoint sidecars written.
    protected.update({str(p):sha(p) for p in TASK09A.rglob('*') if p.is_file()})
    for s in c['train']+c['validation']+[30003]:
        p=ds.run/ds.episodes[s]['path'];protected[str(p)]=sha(p)
    with ExperimentLogger(team='integration',model='task09b_frozen_readout_geometry_probe',seed=42,
        config=c,root=output,job_type='evaluate') as logger:
        root=logger.directory;print('RUN_ID',logger.run_id,flush=True)
        write_json(root/'protocol.json',c);write_json(root/'split_manifest.json',split)
        write_json(root/'identities.json',identities);write_json(root/'protected_before.json',protected)
        write_json(root/'source_data_audit.json',audit_source(ds,c['train']+c['validation']+[30003]))
        snapshot(root)
        from visual.runtime.grounding import GDino
        from visual.runtime.tracking import SAM2Tracker
        vision=OnlineVisionAdapter(GDino(device='cuda'),SAM2Tracker(device='cuda'));rows=[]
        try:
            for part in ('train','validation'):
                for seed in c[part]:
                    print('COLLECT',part,seed,flush=True)
                    rows+=collect_episode(root,plan,models,vision,ds,seed,part,c,smoke=not rows)
                    write_json(root/'samples.json',rows)
                    print('COLLECTED',len(rows),'valid',sum(r['validity']=='VISIBLE_VALID' and r['physical_valid'] for r in rows),flush=True)
        finally:vision.close()
        after=freeze(models);assert before==after
        write_json(root/'frozen_encoder_audit.json',dict(before=before,after=after,pass_gate=before==after,
            extraction_mode='torch.inference_mode, eval, requires_gradFalse; graph encoder only',
            no_policy_optimizer=True,all_six_original_checkpoint_hashes={l:sha(v['receipt']['path']) for l,v in identities.items()}))
        # Contract must pass before broader collection and all probe fitting.
        assert read(root/'collection_contract.json')['pass_gate']
        actual_groups=[read(root/'collection'/f'seed{s}'/'replay_audit.json')['actual_reset_group'] for s in c['train']+c['validation']]
        assert len(actual_groups)==len(set(actual_groups)),'duplicate live reset groups'
        write_json(root/'live_reset_groups.json',dict(pass_gate=True,seeds=c['train']+c['validation'],groups=actual_groups))
        rows+=import_final(root,c);write_json(root/'samples.json',rows)
        from graph.integration_baseline.geometry09b_probe import train_all
        train_all(root,rows,c,logger)
        assert freeze(models)==before
        write_json(root/'post_probe_frozen_audit.json',dict(pass_gate=True,before=before,after=freeze(models)))
        changed=[p for p,h in protected.items() if sha(p)!=h]
        write_json(root/'protection_audit.json',dict(pass_gate=not changed,files=len(protected),changed=changed))
        assert not changed
        write_json(root/'status.json',dict(status='PROBES_COMPLETE',run_id=logger.run_id,samples=len(rows)))
        logger.log(dict(samples=len(rows),train_episodes=16,validation_episodes=6,final_episodes=1))
        for name in ('protocol.json','split_manifest.json','identities.json','frozen_encoder_audit.json','protection_audit.json','probe_metrics.json'):
            logger.link_output(root/name,kind='metrics')
        return root


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(run(args.output),flush=True)
