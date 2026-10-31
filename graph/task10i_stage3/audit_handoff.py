"""Independent final handoff audit. Does not import the collector or assembler."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def rotate(q,v):
    t=2*np.cross(q[...,1:],v)
    return v+q[...,:1]*t+np.cross(q[...,1:],t)

def canonical(q):return np.where(q[...,:1]<0,-q,q)

def audit(root):
    started=time.time();stage2=Path('/workspace/task10i_stage2_20261031_v2/run')
    assert read(root/'pipeline_state.json')['status']=='COMPLETED'
    assert read(root/'completion_audit.json')['status']=='PASS'
    assert read(root/'stage4_raw_dataset_receipt.json')['STAGE4_READY'] is True
    receipt=read(stage2/'stage3_input_receipt.json')
    for name,h in receipt['files'].items():assert sha(stage2/name)==h,name
    for name,h in read(stage2/'selection_identity.json')['files'].items():assert sha(stage2/name)==h,name
    frozen=read(stage2/'stage3_demonstration_plan.json')['per_parent_allocations']
    mapping=read(root/'stage3_final_parent_selection.json')['mapping']
    preserved=[e for e in read(stage2/'existing_episode_preservation_manifest.json')['episodes'] if e['disposition']=='PRESERVE']
    files={'R':'dataset_R_train_manifest.json','D':'dataset_D_train_manifest.json','validation':'shared_validation_manifest.json'}
    manifests={ds:read(root/name) for ds,name in files.items()};keys={};sources={};sets={}
    for ds,m in manifests.items():
        split='validation' if ds=='validation' else 'train'
        expected={(e['parent_id'],e['branch']) for e in preserved if e['split']==split}
        for a in frozen:
            if a['dataset']==ds:expected.update((mapping[ds][a['new_parent']],b['branch']) for b in a['branch_templates'])
        actual={(e['parent_id'],e['branch']) for e in m['episodes']}
        assert actual==expected and len(actual)==len(m['episodes'])
        sets[ds]={p for p,b in actual};keys[ds]={(e['parent_id'],e['branch']):e for e in m['episodes']}
        assert len(actual)==(48 if ds=='validation' else 272)
        assert len(sets[ds])==(16 if ds=='validation' else 96)
        assert sum(b=='full_task' for p,b in actual)==(12 if ds=='validation' else 96)
        for e in m['episodes']:
            assert e['split']==split
            if e['raw_path'] in sources:assert sources[e['raw_path']]['gt_path']==e['gt_path']
            sources[e['raw_path']]=e
    assert not sets['validation']&(sets['R']|sets['D'])
    split=read(stage2/'candidate_pool_split_manifest.json')
    assert not set(split['holdout'])&set.union(*sets.values())
    assert not set(split['validation'])&(sets['R']|sets['D'])
    common=read(stage2/'sampling_trace.json')['common'][:44]
    for cid in common:assert mapping['R'][cid]==mapping['D'][cid]
    shared=set(keys['R'])&set(keys['D'])
    for key in shared:
        for k in ('raw_path','raw_sha256','gt_path','gt_sha256'):assert keys['R'][key][k]==keys['D'][key][k]
    # The already approved original105 must retain both raw and GT identity.
    for e in preserved:
        actual=sources[e['raw_path']]
        assert actual['raw_sha256']==e['raw_sha256'] and actual['gt_sha256']==e['gt_sha256']
    schema=read(root/'runtime_identity.json')['schema']
    slices={f['name']:slice(f['offset'],f['offset']+f['dimension']) for f in schema['fields']}
    nominal={e['parent_id']:e for e in sources.values() if e['branch']=='nominal'}
    counts=Counter();episode_results=[];frame_counts={}
    stage4=read(root/'stage4_recache_input_manifest.json')
    for e in sources.values():
        raw=Path(e['raw_path']);assert sha(raw)==e['raw_sha256'] and sha(e['gt_path'])==e['gt_sha256']
        n=e['actions'];branch=e['branch'];new=e['source']=='new_stage3'
        with h5py.File(raw) as f,np.load(e['gt_path'],allow_pickle=False) as g:
            robot=f['robot_state'][:];y=f['actions'][:]
            assert robot.shape==(n+1,76) and y.shape==(n,7) and np.isfinite(robot).all() and np.isfinite(y).all()
            np.testing.assert_array_equal(f['timestep'][:],np.arange(n+1))
            np.testing.assert_array_equal(f['timestamp'][:],np.arange(n+1)/20)
            for k in ('timestep','timestamp'):np.testing.assert_array_equal(g[k],f[k][:])
            tcp=robot[:,slices['tcp_pose']];gripper=robot[:,slices['gripper_position']]
            tips=[]
            for side,off in [('left',[0,.00758,.04525]),('right',[0,-.00758,.04525])]:
                p=robot[:,slices[side+'_finger_pose']]
                tips.append(p[:,:3]+rotate(p[:,3:],np.broadcast_to(np.asarray(off,np.float32),p[:,:3].shape)))
            for key,value in [('tcp_pose',tcp),('gripper',gripper),('left_tip',tips[0]),('right_tip',tips[1])]:np.testing.assert_allclose(value,g['raw_'+key],atol=1e-7,rtol=0)
            peg=g['raw_peg_pose'];hole=g['raw_hole_pose'];half=g['raw_peg_half'];radius=g['raw_hole_radius']
            head=peg[:,:3]+rotate(peg[:,3:],np.concatenate([half[:,:1],np.zeros((n+1,2),np.float32)],axis=1))
            gt38=np.concatenate([tcp[:,:3],canonical(tcp[:,3:]),tips[0]-tcp[:,:3],tips[1]-tcp[:,:3],gripper,
                peg[:,:3]-tcp[:,:3],canonical(peg[:,3:]),head-hole[:,:3],hole[:,:3]-tcp[:,:3],canonical(hole[:,3:]),half,radius,
                np.ones((n+1,1),np.float32),np.zeros((n+1,1),np.float32)],axis=1)
            np.testing.assert_allclose(gt38,g['gt38'],atol=1e-7,rtol=0)
            for p in (tcp,peg,hole):np.testing.assert_allclose(np.linalg.norm(p[:,3:],axis=1),1,atol=2e-5,rtol=0)
            for cam in ('base_camera','hand_camera'):
                assert f[cam+'/rgb'].shape==(n+1,128,128,3) and f[cam+'/rgb'].dtype==np.uint8
                assert f[cam+'/depth_raw_mm'].shape==(n+1,128,128) and f[cam+'/depth_raw_mm'].dtype==np.int16
                assert f[cam+'/intrinsic'].shape==(n+1,3,3) and f[cam+'/extrinsic_cv'].shape==(n+1,3,4)
                assert f[cam+'/cam2world_gl'].shape==(n+1,4,4)
            seg=f['segment'][:] if 'segment' in f else np.zeros(n,np.uint8)
            mask=f['supervision_mask'][:] if 'supervision_mask' in f else np.ones(n,bool)
            assert not (mask&(seg==1)).any() and (np.diff(seg.astype(int))>=0).all()
            if new:
                np.testing.assert_array_equal(f['executed_action'][:],y);np.testing.assert_array_equal(f['controller_input'][:],y)
                assert len(f['terminated'])==len(f['truncated'])==n
                # No ground-truth object fields may live in the sensor bundle.
                names=[];f.visititems(lambda k,v:names.append(k) if isinstance(v,h5py.Dataset) else None)
                assert not any(any(x in k.lower() for x in ('peg_pose','hole_pose','gt38','target18')) for k in names)
                for k in names:
                    assert len(f[k]) in (n,n+1) and np.isfinite(f[k][:]).all(),k
                trace=read(raw.parent/'trace.json');assert len(trace)==n
                for i,tr in enumerate(trace):
                    assert tr['clock']==i+1 and len(tr['substep_contacts'])==5
                    np.testing.assert_array_equal(np.asarray(tr['action'],np.float32),y[i])
                    np.testing.assert_array_equal(np.asarray(tr['qpos_before'],np.float32),robot[i,slices['qpos']])
                    np.testing.assert_array_equal(np.asarray(tr['tcp_before'],np.float32),tcp[i])
                    assert tr['physical']['valid']
            if branch=='full_task':
                if new:
                    assert bool(f['environment_success'][-1]) and any(tr['physical']['is_grasping'] for tr in trace)
                    assert set(('reach','descend','close','lift','align','insert'))<={tr['phase'] for tr in trace}
                else:assert read(raw.parent/'receipt.json')['success']
            elif branch!='nominal':
                prefix=np.flatnonzero(seg==0);dist=np.flatnonzero(seg==1);rec=np.flatnonzero(seg==2)
                assert len(dist)>0 and len(rec)>0 and np.array_equal(prefix,np.arange(len(prefix)))
                ne=nominal[e['parent_id']]
                with h5py.File(ne['raw_path']) as nf:
                    for k in ('actions','robot_state','tcp_pose','timestep','timestamp','base_camera/rgb','base_camera/depth_raw_mm','hand_camera/rgb','hand_camera/depth_raw_mm'):
                        np.testing.assert_array_equal(f[k][:len(prefix)],nf[k][:len(prefix)],err_msg=k)
                if new:
                    anchor,x,y_mm=map(int,branch[1:].split('_'));assert len(prefix)==anchor
                    goal=tcp[anchor,:3].copy();goal+=np.asarray([x,y_mm,0])*.001
                    for last in dist[-3:]:
                        assert np.linalg.norm(tcp[last+1,:3]-goal)<.001
                        assert abs(robot[last+1,slices['qvel']]).max()<.03
                    meta=read(raw.parent/'receipt.json');actual=(tcp[dist[-1]+1,:3]-tcp[anchor,:3])*1000
                    np.testing.assert_array_equal(actual,np.asarray(meta['disturbance']['actual_offset_mm'],np.float32))
                    assert meta['position_error_m']<.004 and meta['orientation_error_rad']<.03
                    assert meta['target_refresh']['position_delta_m']<2e-6 and meta['target_refresh']['rotation_delta_rad']<1e-5
                    assert gripper[-1].sum()>=.065
            if branch!='full_task':
                physical=read(raw.parent/'physical_audit.json');assert physical['approved'] and len(physical['checks'])==n
                assert all(c['valid'] for c in physical['checks'])
            mask_record=stage4['mask_index'][str(e['episode_id'])]
            assert sha(mask_record['path'])==mask_record['sha256']
            with np.load(mask_record['path']) as mz:
                np.testing.assert_array_equal(mz['bc_supervision_mask'],seg!=1)
                np.testing.assert_array_equal(mz['geometry_eligible'],np.ones(n,bool))
            frame_counts[str(e['episode_id'])]=dict(actions=n,full_task_action_frames=n if branch=='full_task' else 0,
                corrective_prefix_frames=int((seg==0).sum()) if branch!='full_task' else 0,
                disturbance_frames=int((seg==1).sum()),recovery_frames=int((seg==2).sum()),
                BC_supervised_frames=int((seg!=1).sum()),geometry_eligible_frames=n)
            counts.update(episodes=1,actions=n,observations=n+1,new_episodes=int(new),new_actions=n if new else 0)
            episode_results.append(dict(episode_id=e['episode_id'],parent=e['parent_id'],branch=branch,actions=n,raw_sha256=e['raw_sha256'],gt_sha256=e['gt_sha256']))
    totals=read(root/'dataset_completion_summary.json')['datasets']
    for ds,m in manifests.items():
        for key in next(iter(frame_counts.values())):
            assert totals[ds][key]==sum(frame_counts[str(e['episode_id'])][key] for e in m['episodes']),(ds,key)
    final_new={p for p in set.union(*sets.values()) if p.startswith('10i_s1_')}
    pool=Path('/workspace/task10i_stage1_20261031/verified_pool_v2')
    for cid in final_new:
        folder=root/'parents'/cid;parent=read(folder/'parent_result.json');assert parent['status']=='PASS'
        initial=read(folder/'initial_reset.json');assert initial['status']=='PASS'
        assert all(c['exact'] for c in initial['checks'].values())
        with np.load(folder/'initial_readback.npz',allow_pickle=False) as actual,np.load(pool/'candidates'/cid/'initial.npz',allow_pickle=False) as expected:
            assert set(actual.files)==set(expected.files)
            for k in actual.files:np.testing.assert_array_equal(actual[k],expected[k],err_msg=cid+'/'+k)
        full=read(folder/'full_task/receipt.json');assert full['official_success'] is True
        for name,h in full['hashes'].items():assert sha(folder/'full_task'/name)==h
        with h5py.File(folder/'full_task/episode.h5') as f:assert bool(f['environment_success'][-1])
        required_anchors={int(b[1:].split('_')[0]) for b in parent['branches'] if b not in ('full_task','nominal')}
        for anchor in required_anchors:
            assert (folder/f'anchor_t{anchor}.pt').stat().st_size>0
            replay=read(folder/f'replay_t{anchor}.json')
            assert replay['status']=='PASS' and replay['disk_snapshot_restore'] and replay['pixel_depth_exact']
            assert replay['steps']==5 and replay['repeat']==2
    for name,entry in read(root/'artifact_index.json')['files'].items():assert sha(root/name)==entry['sha256'],name
    finalization=read(root/'finalization_identity.json') if (root/'finalization_identity.json').exists() else None
    source=Path(finalization['collection_source_root']) if finalization else root.parent/'source'
    for name,h in read(root/'collector_source_identity.json').items():assert sha(source/name)==h,name
    if finalization:
        for name,h in read(root/'finalizer_source_identity.json').items():assert sha(root.parent/'source'/name)==h,name
    for path,entry in read(root/'existing_checkpoint_identity.json').items():assert sha(path)==entry['sha256']
    old_files=read(root/'existing_source_identity_verified.json')['files']
    preserved_file_count=0
    for path,entry in old_files.items():
        if isinstance(entry,dict) and 'sha256' in entry:
            assert Path(path).stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],path
            preserved_file_count+=1
    pool_inventory=read(stage2/'input_pool_inventory.json')
    for name,entry in pool_inventory.items():
        assert (pool/name).stat().st_size==entry['bytes'] and sha(pool/name)==entry['sha256'],name
    # Seal parent-level initial readback, branch snapshots and replay evidence,
    # which sit outside the independently sealed episode directories.
    auxiliary={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in root.glob('parents/*/*') if p.is_file() and p.name!='owner.lock'}
    failures=[]
    failure_files={}
    for path in root.glob('parents/*/parent_result.json'):
        result=read(path)
        if result['status']=='PASS':continue
        diagnostic=dict(parent_id=result['candidate_id'],reason=result['reason'],
            evidence_path=str(path),source_raw_preserved=True,training_inclusion=False)
        assert result['reason'] not in ('RUNTIME_ERROR','CONTROLLER_ERROR','DATA_ALIGNMENT_FAILURE','INITIAL_RESET_MISMATCH','VISIBILITY_REGRESSION')
        for trace_path in path.parent.glob('*.partial/failure_trace.json'):
            tr=read(trace_path)
            if not tr:continue
            phase_counts=dict(Counter(x['phase'] for x in tr))
            diagnostic.update(actions_recorded=len(tr),phase_counts=phase_counts,last_phase=tr[-1]['phase'],
                final_transition=tr[-1].get('environment_transition'),trace_path=str(trace_path),trace_sha256=sha(trace_path))
            if trace_path.parent.name=='full_task.partial':
                diagnostic['ever_grasped']=any(x['physical'].get('is_grasping',False) for x in tr)
                diagnostic['all_recorded_physical_checks_valid']=all(x['physical']['valid'] for x in tr)
                if result['reason']=='EXPERT_INSERTION_FAILURE':
                    assert diagnostic['ever_grasped'] and not tr[-1]['success_after']
        failures.append(diagnostic)
        for f in path.parent.glob('*.partial/*'):
            if f.is_file():failure_files[str(f.relative_to(root))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    native_path=Path(__file__).with_name('native_initial_replay_audit.json')
    native=read(native_path)
    assert native['status']=='PASS' and native['final_new_parents']==len(final_new)
    assert native['all_final_new_main_wrist_RGBD_calibration_exact'] and native['actions_executed']==0
    assert native['source_run_id']==read(root/'run_identity.json')['run_id']
    return dict(status='PASS',independent_of_collector_assembler=True,run_id=read(root/'run_identity.json')['run_id'],
        counts=dict(counts),shared_parents=len(sets['R']&sets['D']),shared_logical=len(shared),
        exact_original_templates=True,original105_preserved=True,common44_shared=True,train_validation_leakage=False,
        finalized_new_parent_full_task_approvals=len(final_new),all_dataset_frame_statistics_independently_recomputed=True,
        all_historical_source_files_preserved=preserved_file_count,canonical_pool_files_preserved=len(pool_inventory),
        all_final_new_initial_readbacks_exact=True,all_required_disk_snapshot_replay_evidence=True,
        auxiliary_parent_files=auxiliary,
        failed_attempt_diagnostics=failures,
        failed_attempt_files=failure_files,
        independent_native_initial_replay_sha256=sha(native_path),
        independent_GT38_reconstruction=True,all_pre_action_robot_GT_controller_trace_readback=True,
        corrective_actual_offset_and_stabilization=True,all_approved_physical_audits_readback=True,
        terminal_action_labels=False,all_artifact_and_checkpoint_hashes=True,episodes=episode_results,
        historical_full_substep_contacts='NOT_TESTED',continuous_collision_sweeps='NOT_TESTED',policy_improvement='NOT_TESTED',
        seconds=time.time()-started,source_sha256=sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=audit(a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='episodes'}))
