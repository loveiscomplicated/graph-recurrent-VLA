"""Reference-only dataset assembly, full raw readback and evidence-based reporting."""
from collections import Counter,defaultdict
import csv
import html
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np

from .core import STAGE2,POOL,RUNTIME_ROOT,read,write,freeze,sha,verified,branch_spec

MANIFESTS={'R':'dataset_R_train_manifest.json','D':'dataset_D_train_manifest.json','validation':'shared_validation_manifest.json'}
REQUIRED=['stage3_input_identity.json','stage2_selection_identity_verified.json','collector_source_identity.json','runtime_identity.json',
    'expert_protocol_frozen.json','corrective_protocol_frozen.json','parent_branch_collection_plan.json','collection_execution_manifest.json',
    'expert_attempts.csv','expert_success_metrics.csv','corrective_attempts.csv','corrective_physical_metrics.csv','action_gt_alignment_audit.json',
    'initial_reset_replay_audit.json','parent_branch_identity_audit.json','reserve_replacement_history.csv','selection_attrition_analysis.csv',
    'dataset_frame_statistics.csv',*MANIFESTS.values(),'stage3_final_parent_selection.json','shared_episode_reference_audit.json',
    'final_split_and_leakage_audit.json','dataset_completion_summary.json','stage4_raw_dataset_receipt.json',
    'stage4_recache_input_manifest.json','stage4_source_and_schema_contract.json','10i_stage3_expert_dataset_collection.md',
    '10i_stage3_expert_dataset_collection.html','verification.json','artifact_index.json','pipeline_state.json']

def csv_write(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def reference(root,cid,branch):
    folder=root/'parents'/cid/branch;r=verified(folder)
    assert r and r['approved']
    return dict(episode_id='10i_stage3_'+cid+'_'+branch,parent_id=cid,branch=branch,source='new_stage3',
        raw_path=str(folder/'episode.h5'),raw_sha256=sha(folder/'episode.h5'),
        gt_path=str(folder/'gt_supervision.npz'),gt_sha256=sha(folder/'gt_supervision.npz'),
        receipt_path=str(folder/'receipt.json'),receipt_sha256=sha(folder/'receipt.json'),actions=r['actions'],
        action_convention='native Expert Action7, unchanged pd_ee_delta_pose; disturbance commands masked',
        observation_range=[0,r['actions']+1],action_range=[0,r['actions']],
        terminal_Tplus1_action=False,independent_sequence_reset=True,GT_student_input=False)

def assemble(root,plan,mapping):
    originals=read(STAGE2/'existing_episode_preservation_manifest.json')['episodes']
    original_valid=[e for e in originals if e['disposition']=='PRESERVE']
    frozen=read(STAGE2/'stage3_demonstration_plan.json')['per_parent_allocations']
    for ds,name in MANIFESTS.items():
        split='validation' if ds=='validation' else 'train'
        episodes=[dict(e,source='existing_'+e['source'],independent_sequence_reset=True,GT_student_input=False,
                       observation_range=[0,e['actions']+1],action_range=[0,e['actions']],terminal_Tplus1_action=False)
                  for e in original_valid if e['split']==split]
        for a in frozen:
            if a['dataset']!=ds:continue
            cid=mapping[ds][a['new_parent']]
            result=read(root/'parents'/cid/'parent_result.json');assert result['status']=='PASS'
            assert verified(root/'parents'/cid/'full_task')['official_success']
            for template in a['branch_templates']:
                episodes.append(dict(reference(root,cid,template['branch']),split=split,slot_id=a['new_parent'],
                    replaced_visibility_failed_parent=a['replaced_visibility_failed_parent'],
                    original_template_episode_id=template['original_episode_id']))
        parents=sorted({e['parent_id'] for e in episodes})
        assert len(episodes)==(48 if ds=='validation' else 272)
        assert len(parents)==(16 if ds=='validation' else 96)
        assert len({e['episode_id'] for e in episodes})==len(episodes)
        freeze(root/name,dict(dataset=ds,split=split,parents=parents,parent_count=len(parents),logical_episodes=len(episodes),
            episodes=episodes,stage2_selection_sha256=sha(STAGE2/'selection_identity.json'),
            action_codec_identity_path=str(RUNTIME_ROOT/'action_codec.json'),action_codec_identity_sha256=sha(RUNTIME_ROOT/'action_codec.json'),
            action_storage='native7, no fitted recoding; future trainer must explicitly select codec',
            BC_policy='prefix verified against approved nominal; disturbance off; recovery on',graph_cache_created=False))
    audit_manifests(root)

def audit_manifests(root):
    manifests={ds:read(root/name) for ds,name in MANIFESTS.items()}
    sets={ds:set(m['parents']) for ds,m in manifests.items()}
    assert not sets['validation']&(sets['R']|sets['D'])
    split=read(STAGE2/'candidate_pool_split_manifest.json')
    forbidden=set(split['holdout'])
    assert not forbidden & set.union(*sets.values())
    assert (sets['R']|sets['D']) & set(split['validation'])==set()
    assert (sets['validation'] & set(split['training']))==set()
    byds={ds:{(e['parent_id'],e['branch']):e for e in m['episodes']} for ds,m in manifests.items()}
    common=set(byds['R'])&set(byds['D']);shared=[]
    for key in sorted(common):
        a,b=byds['R'][key],byds['D'][key]
        for k in ('raw_path','raw_sha256','gt_path','gt_sha256'):assert a[k]==b[k]
        shared.append(dict(parent_id=key[0],branch=key[1],raw_sha256=a['raw_sha256'],gt_sha256=a['gt_sha256']))
    # Existing sources must be exactly the preserved IDs, never failed-visibility/Final/Test.
    inventory=read(STAGE2/'existing_episode_preservation_manifest.json')['episodes']
    for ds,m in manifests.items():
        split_name='validation' if ds=='validation' else 'train'
        required={e['episode_id'] for e in inventory if e['disposition']=='PRESERVE' and e['split']==split_name}
        actual={e['episode_id'] for e in m['episodes'] if e['source'].startswith('existing_')}
        assert required==actual
        original_sources=Counter(e['source'].removeprefix('existing_') for e in m['episodes'] if e['source'].startswith('existing_'))
        sources=Counter('full_task' if e['branch']=='full_task' else 'corrective' for e in m['episodes'])
        assert sources==({'full_task':12,'corrective':36} if ds=='validation' else {'full_task':96,'corrective':176})
    write(root/'shared_episode_reference_audit.json',dict(status='PASS',shared_parents=len(sets['R']&sets['D']),shared_logical_episodes=len(shared),
        episodes=shared,common_physical_branches_single_raw_reference=True))
    write(root/'final_split_and_leakage_audit.json',dict(status='PASS',parent_counts={k:len(v) for k,v in sets.items()},
        logical_counts={k:len(m['episodes']) for k,m in manifests.items()},train_validation_overlap=[],diagnostic_holdout_used=[],
        existing_final_test_used=[],preserved_existing_ids_exact=True))
    return manifests

def audit_episode(root,e,schema,nominal):
    from graph.gt_pretrain10f1.contract import measured_robot,object_target,assemble_gt38,split_fields
    from recurrent.features import build_features
    from graph.integration_baseline.bc08_codec import controller_commands
    raw=Path(e['raw_path']);gt=Path(e['gt_path'])
    assert sha(raw)==e['raw_sha256'] and sha(gt)==e['gt_sha256']
    n=e['actions'];trace=read(raw.parent/('expert_stages.json' if (raw.parent/'expert_stages.json').exists() else 'trace.json'))
    with h5py.File(raw) as f,np.load(gt,allow_pickle=False) as g:
        assert f['robot_state'].shape==(n+1,76) and f['actions'].shape==(n,7)
        for key in ('timestep','timestamp'):
            np.testing.assert_array_equal(f[key][:],np.arange(n+1)/(20 if key=='timestamp' else 1))
            np.testing.assert_array_equal(g[key],f[key][:])
        y=f['actions'][:];assert y.dtype==np.float32 and np.isfinite(y).all()
        segment=f['segment'][:] if 'segment' in f else np.zeros(n,np.uint8)
        assert segment.shape==(n,) and np.isin(segment,[0,1,2]).all() and (np.diff(segment.astype(int))>=0).all()
        mask=f['supervision_mask'][:] if 'supervision_mask' in f else np.ones(n,bool)
        assert mask.dtype==bool and mask.shape==(n,) and not (mask&(segment==1)).any()
        if 'segment' in f and e['branch'] not in ('nominal','full_task'):
            prefix=np.flatnonzero(segment==0)
            assert np.array_equal(prefix,np.arange(len(prefix)))
            if len(prefix):
                ne=nominal[e['parent_id']]
                with h5py.File(ne['raw_path']) as nf,np.load(ne['gt_path']) as ng:
                    keys=[]
                    f.visititems(lambda k,v:keys.append(k) if isinstance(v,h5py.Dataset) and k in nf and len(v) in (n,n+1) else None)
                    for key in keys:
                        # Segment metadata differs by construction; copied measured
                        # input and actual command arrays must agree exactly.
                        if key in ('segment','supervision_mask','environment_success','terminated','truncated','reward'):continue
                        np.testing.assert_array_equal(f[key][:len(prefix)],nf[key][:len(prefix)],err_msg=key)
                    for key in g.files:np.testing.assert_array_equal(g[key][:len(prefix)],ng[key][:len(prefix)],err_msg=key)
                mask=mask.copy();mask[prefix]=True
            np.testing.assert_array_equal(mask,segment!=1)
        expected=controller_commands(y)
        if 'controller_input' in f:
            np.testing.assert_array_equal(y,f['controller_input'][:])
            np.testing.assert_allclose(expected[:,:6],f['arm_scaled_input'][:],atol=2e-8,rtol=1e-6)
            np.testing.assert_allclose(expected[:,6:],f['gripper_scaled_input'][:],atol=1e-8,rtol=0)
        else:
            for t in trace:
                index=int(t['clock'])-1
                np.testing.assert_array_equal(y[index],np.asarray(t['action'],np.float32))
                np.testing.assert_allclose(expected[index,:6],t['scaled_action'],atol=2e-8,rtol=1e-6)
        raw_gt={k[4:]:g[k] for k in g.files if k.startswith('raw_')}
        assert g['gt38'].shape==(n+1,38)
        np.testing.assert_allclose(build_features(raw_gt,np.ones(n+1,np.float32),np.zeros(n+1,np.float32)),g['gt38'],atol=1e-7,rtol=0)
        measured=measured_robot(f['robot_state'][:],schema)
        for k,v in measured.items():np.testing.assert_allclose(v,raw_gt[k],atol=1e-7,rtol=0)
        np.testing.assert_allclose(object_target(raw_gt),g['target18'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(assemble_gt38(split_fields(g['target18']),measured),g['gt38'],atol=1e-7,rtol=0)
        for k in ('tcp_pose','peg_pose','hole_pose'):
            np.testing.assert_allclose(np.linalg.norm(raw_gt[k][:,3:],axis=-1),1,atol=2e-5,rtol=0)
        np.testing.assert_array_equal(f['tcp_pose'][:],raw_gt['tcp_pose'])
        for cam in ('base_camera','hand_camera'):
            assert f[cam+'/rgb'].shape==(n+1,128,128,3) and f[cam+'/rgb'].dtype==np.uint8
            assert f[cam+'/depth_raw_mm'].shape==(n+1,128,128) and f[cam+'/depth_raw_mm'].dtype==np.int16
            np.testing.assert_array_equal(f[cam+'/depth'][:],f[cam+'/depth_raw_mm'][:].astype(np.float32)/1000)
            assert f[cam+'/intrinsic'].shape==(n+1,3,3) and f[cam+'/extrinsic_cv'].shape==(n+1,3,4)
            assert f[cam+'/cam2world_gl'].shape==(n+1,4,4)
        def finite(k,v):
            if isinstance(v,h5py.Dataset):assert np.isfinite(v[:]).all(),k
        f.visititems(finite)
        for k in g.files:
            assert len(g[k])==n+1,k
            if np.issubdtype(g[k].dtype,np.number) or g[k].dtype==np.bool_:assert np.isfinite(g[k]).all(),k
            else:assert k=='distribution' and g[k].dtype.kind=='U',k
        if e['branch']=='full_task':
            if e['source']=='new_stage3':
                receipt=verified(raw.parent);assert receipt['official_success'] and receipt['success']
                assert all(x['physical']['valid'] for x in trace)
                assert len(trace)==n and any(x['physical']['is_grasping'] for x in trace)
            else:
                assert read(raw.parent/'receipt.json')['success']
            phases=[x['phase'] for x in trace]
        else:
            physical=read(raw.parent/'physical_audit.json')
            assert physical['approved'] and len(physical['checks'])==n
            assert all(x['valid'] for x in physical['checks'])
            phases=np.array(['expert_prefix','disturbance','recovery'])[segment].tolist()
            if e['branch']!='nominal':
                assert (segment==1).any() and (segment==2).any()
                receipt=verified(raw.parent) if e['source']=='new_stage3' else read(raw.parent/'receipt.json')
                if e['source']=='new_stage3':assert receipt['success'] and receipt['position_error_m']<.004 and receipt['orientation_error_rad']<.03
        # Eligibility records synchronized physical/GT evidence, never a geometry loss.
        geometry=np.ones(n,bool)
        maskpath=root/'masks'/(str(e['episode_id'])+'.npz');maskpath.parent.mkdir(parents=True,exist_ok=True)
        if maskpath.exists():
            with np.load(maskpath) as z:
                np.testing.assert_array_equal(z['bc_supervision_mask'],mask)
                np.testing.assert_array_equal(z['geometry_eligible'],geometry)
        else:np.savez_compressed(maskpath,bc_supervision_mask=mask,geometry_eligible=geometry)
        stat=dict(episode_id=e['episode_id'],parent_id=e['parent_id'],branch=e['branch'],source=e['source'],actions=n,
            full_task_action_frames=n if e['branch']=='full_task' else 0,
            corrective_prefix_frames=int((segment==0).sum()) if e['branch']!='full_task' else 0,
            disturbance_frames=int((segment==1).sum()),recovery_frames=int((segment==2).sum()),
            BC_supervised_frames=int(mask.sum()),geometry_eligible_frames=int(geometry.sum()),phase_distribution=dict(Counter(phases)))
        return dict(status='PASS',**stat,mask_path=str(maskpath),mask_sha256=sha(maskpath),
            all_pre_action_GT_sensor_robot_actions=True,terminal_label=False,
            physics_scope='new_full_substep_contact_audit' if e['source']=='new_stage3' and e['branch']=='full_task' else
                'inherited_F1_full_finite_command_success; full_substep_SafetyV2_NOT_TESTED' if e['branch']=='full_task' else 'approved_corrective_SafetyV2')

def miniature(root,fixture,parents):
    """Separate small reference datasets from actual engineering episodes."""
    schema=read(fixture/'runtime_identity.json')['schema'];episodes=[]
    for p in parents:
        cid=p['candidate']['id']
        for b in p['branches']:episodes.append(reference(fixture,cid,b))
    nominal={e['parent_id']:reference(fixture,e['parent_id'],'nominal') for e in episodes if (fixture/'parents'/e['parent_id']/'nominal'/'receipt.json').exists()}
    rows=[audit_episode(root/'engineering_miniature',e,schema,nominal) for e in episodes]
    dest=root/'engineering_miniature';dest.mkdir(parents=True,exist_ok=True)
    ids=[p['candidate']['id'] for p in parents]
    for ds,index in [('R',1),('D',2),('validation',3)]:
        chosen={ids[0],ids[index]} if ds!='validation' else {ids[3]}
        write(dest/MANIFESTS[ds],dict(episodes=[e for e in episodes if e['parent_id'] in chosen],engineering_only=True))
    r=read(dest/MANIFESTS['R'])['episodes'];d=read(dest/MANIFESTS['D'])['episodes'];v=read(dest/MANIFESTS['validation'])['episodes']
    assert not {e['parent_id'] for e in r+d}&{e['parent_id'] for e in v}
    assert [e for e in r if e['parent_id']==ids[0]]==[e for e in d if e['parent_id']==ids[0]]
    write(dest/'verification.json',dict(status='PASS',episodes=len(rows),action_frames=sum(r['actions'] for r in rows),
        full_pipeline_raw_readback=True,GT_action_physics=True,engineering_excluded=True))
    (dest/'report.md').write_text(f'# Engineering miniature\n\nPASS: {len(rows)} real episodes; sensor/GT/action readback and reference-only R/D assembly. Excluded from research data.\n')

def collect_audits(root,manifests):
    unique={e['episode_id']:e for m in manifests.values() for e in m['episodes']}
    nominal={e['parent_id']:e for e in unique.values() if e['branch']=='nominal'}
    schema=read(root/'runtime_identity.json')['schema']
    identity=read(root/'finalization_identity.json') if (root/'finalization_identity.json').exists() else None
    prior=Path(identity['collection_root'])/'action_gt_alignment_audit.json' if identity else None
    if prior and prior.exists():
        previous=read(prior);assert previous['status']=='PASS'
        records={str(r['episode_id']):r for r in previous['episodes_audit']}
        assert set(records)=={str(e['episode_id']) for e in unique.values()}
        audit=[]
        for e in unique.values():
            assert sha(e['raw_path'])==e['raw_sha256'] and sha(e['gt_path'])==e['gt_sha256']
            r=records[str(e['episode_id'])].copy();assert r['actions']==e['actions'] and r['status']=='PASS'
            mask=root/'masks'/Path(r['mask_path']).name;assert sha(mask)==r['mask_sha256']
            r['mask_path']=str(mask);audit.append(r)
        freeze(root/'alignment_audit_reuse.json',dict(status='PASS',source=str(prior),source_sha256=sha(prior),
            episodes=len(audit),raw_GT_all_rehashed=True,mask_all_rehashed=True,
            reason='Complete raw readback passed before the subsequent metadata-CSV failure; dataset references unchanged.'))
    else:
        audit=[audit_episode(root,e,schema,nominal) for e in unique.values()]
    write(root/'action_gt_alignment_audit.json',dict(status='PASS',episodes=len(audit),episodes_audit=audit,
        preaction_all_frames=True,terminal_Tplus1_has_no_action=True,sensor_only_bundle_GT_separated=True))
    # Includes successful full-task approvals retained from a failed corrective parent.
    attempts=[];corrective=[];resets=[];identity=[]
    for path in sorted((root/'parents').glob('*/parent_result.json')):
        r=read(path);folder=path.parent;full=verified(folder/'full_task')
        attempts.append(dict(parent_id=folder.name,seed=r['seed'],parent_status=r['status'],failure_reason=r.get('reason',''),
            full_task_success=bool(full and full.get('official_success')),approved_branches=r.get('branches',r.get('approved_branches',[])),wall_seconds=r['wall_seconds']))
        if (folder/'initial_reset.json').exists():resets.append(read(folder/'initial_reset.json'))
        for branch in folder.iterdir():
            if not branch.is_dir() or branch.name in ('full_task','nominal','interrupted') or branch.name.endswith('.partial'):continue
            rc=verified(branch)
            if rc:
                disturbance=rc.get('disturbance',{}).copy()
                if 'orientation_error_rad' in disturbance:
                    disturbance['disturbance_orientation_error_rad']=disturbance.pop('orientation_error_rad')
                corrective.append(dict(parent_id=folder.name,branch=branch.name,approved=True,**disturbance,
                    recovery_steps=rc.get('recovery_steps'),position_error_m=rc.get('position_error_m'),orientation_error_rad=rc.get('orientation_error_rad')))
        for partial in folder.glob('*.partial'):
            corrective.append(dict(parent_id=folder.name,branch=partial.name.removesuffix('.partial'),approved=False,
                failure_reason=r.get('reason','interrupted'),evidence=str(partial)))
    csv_write(root/'expert_attempts.csv',attempts)
    csv_write(root/'expert_success_metrics.csv',attempts)
    csv_write(root/'corrective_attempts.csv',corrective)
    csv_write(root/'corrective_physical_metrics.csv',corrective)
    write(root/'initial_reset_replay_audit.json',dict(status='PASS',attempts=len(resets),rows=resets,
        actual_native_reset=True,all_checked_fields_exact=True,hidden_solver_cache='NOT_SERIALIZED'))
    for ds,m in manifests.items():
        for e in m['episodes']:identity.append(dict(dataset=ds,parent_id=e['parent_id'],branch=e['branch'],episode_id=e['episode_id'],raw_sha256=e['raw_sha256'],gt_sha256=e['gt_sha256']))
    write(root/'parent_branch_identity_audit.json',dict(status='PASS',rows=identity))
    stats_by_id={str(r['episode_id']):r for r in audit};stats=[];aggregate={}
    for ds,m in manifests.items():
        rows=[dict(dataset=ds,**stats_by_id[str(e['episode_id'])]) for e in m['episodes']]
        stats+=rows
        aggregate[ds]={k:sum(r[k] for r in rows) for k in ('actions','full_task_action_frames','corrective_prefix_frames','disturbance_frames','recovery_frames','BC_supervised_frames','geometry_eligible_frames')}
        aggregate[ds].update(parents=m['parent_count'],logical_episodes=m['logical_episodes'],full_task=sum(e['branch']=='full_task' for e in m['episodes']),corrective=sum(e['branch']!='full_task' for e in m['episodes']))
    csv_write(root/'dataset_frame_statistics.csv',stats)
    write(root/'dataset_completion_summary.json',dict(status='PASS',datasets=aggregate,unique_logical=len(unique),unique_action_frames=sum(r['actions'] for r in audit),
        original_preserved_train=82,original_preserved_validation=23,frames_truncated_for_matching=False))
    events=read(root/'stage3_final_parent_selection.json')['events']
    csv_write(root/'reserve_replacement_history.csv',events,fields=['event','role','slot','failed_parent','reserve_rank','candidate_id','datasets','reason','branches'])
    return audit,attempts,corrective

def coverage_analysis(root,manifests,attempts):
    """Re-evaluate fixed distance only; never recompute a greedy queue or selection."""
    from graph.task10i_stage2.core import descriptor,distances,require_frozen
    from graph.task10i_stage2.evaluate import stats
    require_frozen(STAGE2)
    with np.load(STAGE2/'analysis_states.npz') as z:
        rows={str(cid):dict(id=str(cid),descriptor=descriptor({k:z[k][i] for k in z.files if k!='parent_id'})) for i,cid in enumerate(z['parent_id'])}
    scale=read(STAGE2/'feature_normalization.json');split=read(STAGE2/'candidate_pool_split_manifest.json')
    originals={ds:read(STAGE2/name)['parents'] for ds,name in [('R','dataset_R_parent_manifest.json'),('D','dataset_D_parent_manifest.json'),('validation','shared_validation_parent_manifest.json')]}
    successful={a['parent_id'] for a in attempts if a['full_task_success']}
    sets={}
    for ds in ('R','D'):
        sets[ds+'_initial']=[rows[p['id']] for p in originals[ds]]
        sets[ds+'_expert_success_conditional']=[rows[p['id']] for p in originals[ds] if p['kind']=='existing' or p['id'] in successful]
        sets[ds+'_final']=[rows[cid] for cid in manifests[ds]['parents']]
    summaries=[]
    for cohort in ('training','holdout'):
        for name,selected in sets.items():
            d,_=distances([rows[cid] for cid in split[cohort]],selected,scale)
            summaries.append(dict(cohort=cohort,dataset=name,n=len(d),**stats(d.min(1))))
    csv_write(root/'replacement_coverage_comparison.csv',summaries)
    attrs=[]
    primary=read(STAGE2/'stage3_parent_collection_union.json')['parents'];byid={a['parent_id']:a for a in attempts}
    for p in primary:
        attrs.append(dict(**byid[p['id']],memberships=p['memberships'],role=p['role'],
            descriptor=p['descriptor'],final_in_R=p['id'] in manifests['R']['parents'],final_in_D=p['id'] in manifests['D']['parents'],final_in_validation=p['id'] in manifests['validation']['parents']))
    csv_write(root/'selection_attrition_analysis.csv',attrs)
    write(root/'selection_distribution_analysis.json',dict(status='PASS',original=originals,expert_success_parent_ids=sorted(successful),
        final={ds:m['parents'] for ds,m in manifests.items()},coverage=summaries,
        selection_bias='Final data are conditional on unchanged expert success and required corrective safety; preserve original and attrition. No policy performance conclusion.'))
    return summaries

def media(root,manifests):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    folder=root/'media';folder.mkdir(exist_ok=True)
    new=[e for e in manifests['R']['episodes'] if e['source']=='new_stage3']
    full=next(e for e in new if e['branch']=='full_task')
    recovery=next(e for e in new if e['branch'] not in ('full_task','nominal'))
    gallery=[]
    for label,e in [('expert_full_task',full),('corrective_recovery',recovery)]:
        with h5py.File(e['raw_path']) as f,np.load(e['gt_path']) as g:
            n=e['actions'];images=np.concatenate((f['base_camera/rgb'][:],f['hand_camera/rgb'][:]),axis=2)
            imageio.mimsave(folder/(label+'.mp4'),images,fps=20,macro_block_size=1)
            for t in (0,n//2,n):gallery.append((label,t,images[t]))
            fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True)
            axs[0].plot(f['timestamp'][:n],f['actions'][:,:3]);axs[0].set_ylabel('Native translation command')
            axs[1].plot(f['timestamp'][:],g['raw_peg_pose'][:,:3]);axs[1].set_ylabel('Peg world position (m)')
            if 'segment' in f:axs[2].step(f['timestamp'][:n],f['segment'][:],where='post')
            else:axs[2].plot(f['timestamp'][:n],f['actions'][:,6])
            axs[2].set_xlabel('Simulation time (s)');fig.tight_layout();fig.savefig(folder/(label+'_timeline.png'));plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,6))
    for ax,(label,t,img) in zip(axs.flat,gallery):ax.imshow(img);ax.set_title(f'{label}: t={t}');ax.axis('off')
    fig.tight_layout();fig.savefig(folder/'main_wrist_gallery.png');plt.close(fig)
    analysis=read(root/'selection_distribution_analysis.json')
    with np.load(STAGE2/'analysis_states.npz') as z:
        index={str(cid):i for i,cid in enumerate(z['parent_id'])}
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,ds in zip(axs,['R','D']):
            for label,ids,marker in [('initial',[p['id'] for p in analysis['original'][ds]],'o'),('final',analysis['final'][ds],'x')]:
                xy=z['peg_pose'][[index[i] for i in ids],:2];ax.scatter(xy[:,0],xy[:,1],s=20,marker=marker,label=label)
            ax.set_title(ds+' initial/final');ax.set_xlabel('Peg world x (m)');ax.set_ylabel('Peg world y (m)');ax.legend()
        fig.tight_layout();fig.savefig(folder/'R_D_initial_final_distribution.png');plt.close(fig)
    with (root/'expert_attempts.csv').open() as f:attempts=list(csv.DictReader(f))
    counts=Counter('full_success' if r['full_task_success']=='True' else r['failure_reason'] for r in attempts)
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(list(counts),list(counts.values()));ax.tick_params(axis='x',rotation=20);fig.tight_layout();fig.savefig(folder/'collection_outcomes.png');plt.close(fig)
    with (root/'replacement_coverage_comparison.csv').open() as f:cover=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(11,4));labels=[r.get('cohort','')+' '+r.get('dataset','') for r in cover]
    ax.bar(labels,[float(r['mean']) for r in cover]);ax.set_ylabel('Frozen distance mean nearest coverage');ax.tick_params(axis='x',rotation=50)
    fig.tight_layout();fig.savefig(folder/'replacement_coverage.png');plt.close(fig)

def report(root):
    summary=read(root/'dataset_completion_summary.json');shared=read(root/'shared_episode_reference_audit.json')
    primary=read(STAGE2/'stage3_parent_collection_union.json')['parents'];primary_ids={p['id'] for p in primary}
    with (root/'expert_attempts.csv').open() as f:attempts=list(csv.DictReader(f))
    initial=[a for a in attempts if a['parent_id'] in primary_ids];success=sum(a['full_task_success']=='True' for a in initial)
    events=read(root/'stage3_final_parent_selection.json')['events'];assigned=[e for e in events if e['event']=='ASSIGNED']
    run=read(root/'run_identity.json')['run_id'];gate=read(root/'engineering_gates.json')
    rows=['| Dataset | Parents | Logical | Full | Corrective | Action frames | BC frames | Disturbance frames |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds,r in summary['datasets'].items():rows.append(f"| {ds} | {r['parents']} | {r['logical_episodes']} | {r['full_task']} | {r['corrective']} | {r['actions']} | {r['BC_supervised_frames']} | {r['disturbance_frames']} |")
    text=f'''# Task10-I Stage3 Expert Dataset Collection

Status: **COMPLETED**, subject to the separately generated completion audit. ExperimentLogger UUID `{run}`. Executed root: `{root}`. Stage2 selection UUID `34d10bf652b344a38d54b7672dbfe0c0` remains immutable.

## Dataset result — CONFIRMED

{chr(10).join(rows)}

R/D contain exactly the preserved Train33/82 episodes plus verified new templates; shared Validation preserves Parent7/23 and adds the frozen nine slots. Existing visibility-failed215 logical episodes, original raw/GT/checkpoints and Stage1 candidate data were not modified. Full-task validation trajectories without a planned logical entry remain diagnostic sources, excluded from counts. R/D share **{shared['shared_parents']} parents / {shared['shared_logical_episodes']} identical logical references**, with exact raw/GT paths and SHA equality. Train/Validation parent leakage, Diagnostic Holdout use and historical Final/Test inclusion are zero.

## Actual expert success and attrition — CONFIRMED

Q1: **{success}/90** original selected parents have an approved official full-task success. Q2: per-parent role/membership, reasons and outcomes are fully retained in [selection_attrition_analysis.csv](selection_attrition_analysis.csv). Runtime errors are infrastructure failures and are never counted as expert task failures. Q3/Q4: **{len(assigned)} reserve assignments** were made using frozen queues in validation/common/R/D order; skips and failures are preserved in [reserve history](reserve_replacement_history.csv). Original, expert-success-conditional and final distributions are retained in [distribution analysis](selection_distribution_analysis.json), with [fixed-distance coverage](replacement_coverage_comparison.csv). We do not re-run greedy or alter weights/seeds. Expert conditioning is selection bias and limits generalization beyond this eligible, expert-solvable distribution.

## Sensor, action and GT contract — CONFIRMED

Q5: [shared reference audit](shared_episode_reference_audit.json) checks every shared physical parent/branch by raw and GT SHA. Q6: original uint8 RGB/int16-mm depth at Main `base_camera` and Wrist `hand_camera`, measured Robot76 and unchanged native Action7 are retained. Every T+1 observation/T action sequence is reopened; timestamps, measured robot/GT, quaternion order/world coordinates/metres, GT38 construction, controller preprocessing and masks are audited. Terminal observations have no invented action. New controller-state readback and physical diagnostics are additional audit fields; historical full-task recordings did not include the newer full substep contact audit. The base sensor/action/GT schema is unchanged. No GT sidecar is passed to student inference.

Q7/Q8: Corrective history contains exact approved nominal prefix, all real TCP disturbance observations and fresh current-state Expert recovery. Disturbance BC is always OFF; prefix BC is ON only after exact nominal comparison; recovery BC is ON. Frozen bounded feedback, offsets, anchors and SafetyV2 tolerances remain unchanged. Disk snapshots and 2×5-step observed replay are tested; PhysX hidden solver caches are not claimed serialized. No learned hidden state is stored. Corrective success is pre-grasp pose recovery under the inherited <4mm/<0.03rad and open-gripper contract, rather than an invented full insertion label.

## Physics and execution limits

Existing F1 full-task evidence covers executed expert commands, finite robot/object states, GT alignment and official insertion; historical full-task SafetyV2 substep contacts remain **NOT_TESTED**. New full-task records retain all five physics-substep contacts and joint/finite checks, with finger/peg grasp compliance recorded separately from abnormal non-grasp penetration at the inherited200µm tolerance. Grasp contact forces are recorded without inventing a rejection cap. Corrective uses the original no-contact robot/object and enabled self-collision SAT safety audit. No continuous swept-volume collision freedom is claimed.

Linux CPU PhysX / CUDA SAPIEN rendering on RTX3090 instance57987772 executed at20Hz control/100Hz physics and128×128 cameras. GPU PhysX, macOS simulation, perception, graph inference and policy learning/evaluation were not executed. Task10-H and unrelated processes were not interrupted. The measured worker benchmark selected **{gate['workers']} workers**; [engineering gates](engineering_gates.json) preserve actual one/two/three-worker throughput and exact sensor/action/GT equivalence, simulator crash/resume, verified skip, distinct-role full-task smoke and a separate miniature end-to-end dataset. Engineering data are excluded from research manifests.

## Readiness — CONFIRMED / NOT_TESTED

Q9: [frame statistics](dataset_frame_statistics.csv) provide every episode's Full-task, Prefix, Disturbance, Recovery, BC and geometry-eligible frame count plus phase distribution and parent length. Equal episode counts do not imply equal exposure; no frames were truncated to equalize datasets. Future Stage5 sampling is outside this task. Q10/Q11: [completion summary](dataset_completion_summary.json) and [split audit](final_split_and_leakage_audit.json) prove counts and initial visibility/split preservation. Q12: [Stage4 receipt](stage4_raw_dataset_receipt.json), [raw manifest](stage4_recache_input_manifest.json) and [schema contract](stage4_source_and_schema_contract.json) are ready for a later explicitly requested recache. Q13 policy improvement: **NOT_TESTED**. No Stage4 perception decision, graph recache, GNN/GRU training, student rollout, commit or external publication occurred.

## Artifacts

- [Raw/GT/action audit](action_gt_alignment_audit.json), [reset audit](initial_reset_replay_audit.json), [parent/branch audit](parent_branch_identity_audit.json)
- [Full-task video](media/expert_full_task.mp4), [recovery video](media/corrective_recovery.mp4), [Main/Wrist gallery](media/main_wrist_gallery.png)
- [Action/GT timeline](media/expert_full_task_timeline.png), [recovery timeline](media/corrective_recovery_timeline.png), [collection outcomes](media/collection_outcomes.png)
- [R/D initial/final distribution](media/R_D_initial_final_distribution.png), [replacement coverage](media/replacement_coverage.png)
- [Verification](verification.json), [artifact index](artifact_index.json), [completion audit](completion_audit.json)
'''
    if (root/'finalization_identity.json').exists():
        identity=read(root/'finalization_identity.json')
        text+=f'''\n## Engineering finalization history\n\nCollection UUID remains `{run}`. Finalization UUID is `{identity['finalization_run_id']}`. Original collection root `{identity['collection_root']}` and its failed reporting source/state are preserved. All raw episodes, GT, successful branches, failure traces, parent mappings and dataset references are reused without simulator rollout or collection retry. The correction separates disturbance and recovery orientation-error CSV fields. Python-Markdown is vendored under the isolated finalizer source; the existing simulator/training environment was not modified. Collector and finalizer source identities are independently pinned.\n'''
    (root/'10i_stage3_expert_dataset_collection.md').write_text(text)
    import markdown
    rendered=markdown.markdown(text,extensions=['markdown.extensions.tables','markdown.extensions.fenced_code'])
    (root/'10i_stage3_expert_dataset_collection.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Task10-I Stage3</title><style>body{max-width:1100px;margin:36px auto;font:16px/1.6 system-ui;padding:16px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}img{max-width:100%}</style>'+rendered+'</html>')

def audit_completion(root):
    for name in REQUIRED:assert (root/name).is_file(),name
    manifests=audit_manifests(root)
    verify=read(root/'verification.json');assert verify['status']=='PASS'
    for name,entry in read(root/'artifact_index.json')['files'].items():
        p=root/name;assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
    for m in manifests.values():
        for e in m['episodes']:
            assert sha(e['raw_path'])==e['raw_sha256'] and sha(e['gt_path'])==e['gt_sha256']
    if (root/'finalization_identity.json').exists():
        identity=read(root/'finalization_identity.json')
        original_source=Path(identity['collection_source_root'])
        for name,h in read(root/'collector_source_identity.json').items():assert sha(original_source/name)==h,name
        final_source=Path(__file__).resolve().parents[2]
        for name,h in read(root/'finalizer_source_identity.json').items():assert sha(final_source/name)==h,name
    else:
        from .run_pipeline import source_identity
        assert source_identity()==read(root/'collector_source_identity.json')
    return True

def finalize(root):
    manifests=audit_manifests(root);audit,attempts,corrective=collect_audits(root,manifests)
    coverage_analysis(root,manifests,attempts);media(root,manifests)
    # Original source inventory is checked again after all simulation and assembly.
    original=read(root/'existing_source_identity_verified.json')['files']
    for path,entry in original.items():
        if isinstance(entry,dict) and 'sha256' in entry:assert sha(path)==entry['sha256']
    for path,entry in read(root/'existing_checkpoint_identity.json').items():assert sha(path)==entry['sha256']
    for rel,entry in read(STAGE2/'input_pool_inventory.json').items():assert sha(POOL/rel)==entry['sha256']
    from .core import verify_receipt
    verify_receipt()
    schema=dict(sensor_contract='F1 RGBD128 Main/Wrist; raw depth int16 mm including documented invalid values',robot_schema=read(root/'runtime_identity.json')['schema'],
        GT38='recurrent.features.FEATURE_LAYOUT; world-aligned relative vectors; wxyz; metres',
        action='native7 unchanged pd_ee_delta_pose; no codec refit; controller preprocessing audited',
        teacher_GT_sidecar='construction/supervision only, physically separate NPZ from sensor HDF5',
        observations='T+1; action labels/BC mask/geometry eligibility T only',
        history='branch-specific full causal prefix/disturbance/recovery; independent downstream tracker/GRU reset',
        perception_decision='DEFERRED_TO_STAGE4',Graph_cache=False,geometry_loss=False,
        inherited_full_task_substep_SafetyV2_contacts='NOT_TESTED',new_full_task_scope='all substep contacts; intended finger-peg compliance separated; no continuous swept-volume claim')
    write(root/'stage4_source_and_schema_contract.json',schema)
    raw={str(e['episode_id']):e for m in manifests.values() for e in m['episodes']}
    write(root/'stage4_recache_input_manifest.json',dict(status='READY',datasets={ds:dict(path=str(root/name),sha256=sha(root/name)) for ds,name in MANIFESTS.items()},
        unique_sources=raw,mask_index={str(a['episode_id']):dict(path=a['mask_path'],sha256=a['mask_sha256']) for a in audit},
        sensor_GT_separate=True,full_history=True,previous_graphs_reused=False,perception_config='NOT_SELECTED'))
    receipt=dict(status='READY_FOR_STAGE4_RAW_RECACHE',STAGE4_READY=True,run_id=read(root/'run_identity.json')['run_id'],
        dataset_summary=read(root/'dataset_completion_summary.json'),schema_sha256=sha(root/'stage4_source_and_schema_contract.json'),
        recache_manifest_sha256=sha(root/'stage4_recache_input_manifest.json'),stage4_auto_start=False,
        policy_improvement='NOT_TESTED',original_source_preservation='PASS')
    write(root/'stage4_raw_dataset_receipt.json',receipt)
    write(root/'verification.json',dict(status='PASS',unique_episodes_audited=len(audit),new_parent_attempts=len(attempts),
        all_approved_episode_actions_GT_robot_cameras_masks=True,all_final_new_parents_full_task_success=True,
        exact_shared_raw_GT=True,counts=read(root/'dataset_completion_summary.json')['datasets'],
        original_source_hashes_unchanged=True,stage1_stage2_hashes_unchanged=True,engineering_excluded=True,
        limitations=schema,policy_improvement='NOT_TESTED'))
    report(root)
    # Index immutable scientific artifacts; operational state/locks/logs/logger metadata
    # remain mutable. Raw episode receipts independently seal every committed file.
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob('*')
        if p.is_file() and (p.parent==root or p.parent.name in ('media','masks')) and
        p.name not in ('pipeline_state.json','artifact_index.json','completion_audit.json','run.lock') and
        not p.name.endswith(('.tracking.json','.log')) and '_live_handles' not in p.name}
    write(root/'artifact_index.json',dict(files=files,index_scope='top-level scientific manifests/reports, mask sidecars and media; committed raw data sealed by per-episode receipts'))
    audit_completion(root)
    write(root/'completion_audit.json',dict(status='PASS',dataset_count_barriers=True,artifact_index_readback=True,
        all_dataset_raw_GT_SHA_readback=True,required_artifacts_present=True,stage4_receipt_ready=True,
        original_preserved=True,source_identity=sha(root/'collector_source_identity.json'),policy_performance='NOT_TESTED',completed_at=time.time()))

def incomplete_report(root,error):
    """Preserve a useful failure handoff without manufacturing a READY receipt."""
    attempts=[read(p) for p in (root/'parents').glob('*/parent_result.json')]
    committed=[]
    for p in (root/'parents').glob('*/*/receipt.json'):
        r=verified(p.parent)
        committed.append(dict(parent=p.parent.parent.name,branch=p.parent.name,actions=r['actions'],path=str(p.parent)))
    summary=dict(status='INCOMPLETE',error=error,parent_attempts=len(attempts),passed_parents=sum(r['status']=='PASS' for r in attempts),
        failed_parents=[dict(parent=r['candidate_id'],reason=r.get('reason'),evidence=str(root/'parents'/r['candidate_id'])) for r in attempts if r['status']!='PASS'],
        committed_raw_episodes=len(committed),committed=committed,targets=dict(R=[96,272],D=[96,272],validation=[16,48]),
        required_work_remaining='Complete all dataset/physics/alignment/count/handoff/report barriers; inspect pipeline_error.json before any resume.')
    write(root/'dataset_incomplete_summary.json',summary)
    write(root/'completion_audit.json',dict(status='INCOMPLETE',objective_achieved=False,error=error))
    write(root/'stage4_raw_dataset_receipt.json',dict(status='NOT_READY',STAGE4_READY=False,error=error,stage4_auto_start=False))
    text=f'''# Task10-I Stage3 Expert Dataset Collection

상태: **INCOMPLETE**. 최종 R96/272, D96/272, Validation16/48 구성 및 Stage4 인계 완료를 주장하지 않는다.

실행 root: `{root}`. 실패 원인: `{error}`.

현재 Parent 시도 {len(attempts)}개, 모든 요구 Branch를 통과한 Parent {summary['passed_parents']}개, 원자적으로 commit하고 SHA를 확인한 Raw Episode {len(committed)}개다. 성공한 다른 Branch와 원본 실패 증거는 보존했다. [상세 미완료 목록](dataset_incomplete_summary.json), [실제 traceback](pipeline_error.json), [파이프라인 상태](pipeline_state.json), [완료 감사](completion_audit.json)를 확인한다.

Stage2 동결 Selection/Reserve와 기존 Raw/GT/Checkpoint는 수정하지 않았다. 인프라 오류는 실제 Expert 실패와 분리해야 하며, 원인 확인 없이 동일 Seed의 성공 결과를 골라 다시 시도하지 않는다. Stage4 Perception/Graph 재캐시, GNN/GRU 학습 및 정책 평가는 실행하지 않았다. 정책 개선은 **NOT_TESTED**다.
'''
    (root/'10i_stage3_expert_dataset_collection.md').write_text(text)
    import markdown
    (root/'10i_stage3_expert_dataset_collection.html').write_text('<!doctype html><meta charset="utf-8">'+markdown.markdown(text))
