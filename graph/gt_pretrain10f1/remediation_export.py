"""Seal a directly GT-labeled supplement; publish marker only after validation."""
import argparse,copy,os,shutil,subprocess,sys,time
from pathlib import Path
import h5py,numpy as np
from graph.plan_a_bc.common import read,write,sha,now,npz
from recurrent.features import build_features
from .ready_contract import validate

ENCODER='80de6fd742be4e7c9a3477cde0419e803b72253c09d1227fc0fec23b965b0761'

def run(root,source,dino,sam,native_graph_root=None):
    if (source/'gnn_dataset_ready.json').exists():raise FileExistsError('existing READY marker preserved')
    specs=read(root/'supplement_plan.json')['episodes'];assert len(specs)==212
    assert read(root/'graph_cache_manifest.json')['complete']
    suffix='_native_v2' if native_graph_root else ''
    export=root/('sealed_export'+suffix);export.mkdir(exist_ok=False)
    native_rows=None;native_runtime=None
    if native_graph_root:
        nm=read(native_graph_root/'native_graph_manifest.json');assert nm['status']=='PASS' and len(nm['episodes'])==212
        native_runtime=read(native_graph_root/'native_runtime.json')
        assert sha(native_graph_root/'native_runtime.json')==nm['native_runtime_sha256']
        assert sha(native_graph_root/'native_label_contract.json')==nm['native_label_contract_sha256']
        assert read(native_graph_root/'native_label_contract.json')['status']=='PASS'
        native_rows={r['capture_id']:r for r in nm['episodes']}
    copies=[]
    def copy(p,rel):
        dest=export/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert sha(p)==sha(dest);copies.append(dict(source=str(p),sealed=str(dest.relative_to(source)),sha256=sha(dest)))
        return str(dest.relative_to(source))
    # Include independent frozen inputs/config/source; never rely on source symlinks.
    preservation=read(root/'original_preservation_before.json')
    for p,h in preservation['files'].items():assert sha(Path(p))==h,'original data changed '+p
    for p,h in preservation['original_source_sha256'].items():
        assert sha(Path('/workspace/task10f_corrective_reach/source')/p)==h,'original source changed '+p
    preservation_name='original_preservation_after'+suffix+'.json';execution_name='executed_source_identity'+suffix+'.json'
    write(root/preservation_name,dict(status='PASS',data_files=len(preservation['files']),
        source_files=len(preservation['original_source_sha256']),at=now()),immutable=True)
    write(root/execution_name,dict(current_source_sha256={str(p.relative_to(root/'source')):sha(p) for p in (root/'source').rglob('*.py')},
        collector_versions={p.name:sha(p) for p in root.glob('remediation_collect_v*.py')},
        engineering_nominal_collector='preserved remediation_collect_v2.py; import fixed; nominal capture/GT/physics functions unchanged in v3',
        corrective_and_primary_collector='v3, frozen before independent workers launch; offset vector corrected before any disturbance capture'),immutable=True)
    for name in ['config.json','normalizer.json','action_codec.json','supplement_plan.json','config_lineage.json','original_cache_config_reconstructed.json','source_identity.json',execution_name,'engineering_adoption.json','original_reuse_scope.json','original_preservation_before.json',preservation_name]:
        copy(root/name,'provenance/'+name)
    for p in sorted((root/'source').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.json','.toml') and '__pycache__' not in p.parts:
            copy(p,'source/'+str(p.relative_to(root/'source')))
    for p in sorted((root/'inputs').rglob('*')):
        if p.is_file():copy(p,'inputs/'+str(p.relative_to(root/'inputs')))
    for p in sorted(root.glob('remediation_collect_v*.py')):copy(p,'provenance/collector_history/'+p.name)
    runtimes=[]
    for w in range(4):
        runtime=read(root/f'runtime_worker{w}.json');runtimes.append(runtime);copy(root/f'runtime_worker{w}.json',f'provenance/runtime_worker{w}.json')
    assert all(r==runtimes[0] for r in runtimes)
    live=runtimes[0];assert live['control_freq']==20 and live['physics_freq']==100 and live['sim_steps_per_control']==5
    assert "sim_backend='physx_cpu'" in live['backend'] and "render_backend='sapien_cuda'" in live['backend']
    deps=['graph/plan_a_geometry/core.py','graph/plan_a_geometry/protocol.json','graph/plan_a_bc/geometry.py',
        'graph/persistent_geometry/graph.py','graph/persistent_geometry/self_filter.py',
        'graph/integration_baseline/online_vision_adapter.py','visual/runtime/tracking.py','visual/runtime/grounding.py','recurrent/features.py']
    src=root/'source'
    oldsrc=Path('/workspace/task10f_corrective_reach/source')
    for k in deps:assert sha(src/k)==sha(oldsrc/k),'original graph/vision source changed or differs '+k
    assert sha(src/'recurrent/features.py')==ENCODER
    actual_vision=dict(dino_files={p.name:sha(p) for p in dino.iterdir() if p.is_file()},sam_sha256=sha(sam))
    original_vision=read(source/'cache_gate/corrective_cache_gate.json')['model_identity']
    assert actual_vision==original_vision,'actual vision weights differ from reused cache producer'
    camera=read(root/'inputs/sensor_contract.json');camera['backend']=live['backend'];camera['reference_contract_sha256']=sha(root/'inputs/sensor_contract.json')
    camera['live_capture_shapes']=live['camera_shapes'];camera['actual_runtime_source']='runtime_worker0..3.json; per-timestep calibration retained in every H5'
    runtime=dict(simulator=dict(mani_skill=live['versions']['mani_skill'],sapien=live['versions']['sapien'],gymnasium=live['versions']['gymnasium'],
        sim_backend='physx_cpu',render_backend='sapien_cuda',control_mode='pd_ee_delta_pose',control_freq=20,physics_freq=100,
        actual_controller=live['controller']),tensor_runtime={k:live['versions'][k] for k in ['python','torch','numpy']},
        camera_contract=camera,robot_schema=live['schema'],geometry=read(root/'config.json')['geometry'],
        geometry_protocol_sha256=sha(src/'graph/plan_a_geometry/protocol.json'),vision_weight_sha256=actual_vision,
        source_sha256={k:sha(src/k) for k in deps},gt38_encoder_sha256=ENCODER,quaternion_order='wxyz',
        mixing_policy='This source runtime is explicit. No equality to F1/other runtime is asserted; strict release compatibility/equivalence remains required before mixing.',
        config_sha256=sha(root/'config.json'),config_source='sealed supplement config, immutable during actual collection/cache')
    if native_runtime:
        for k in deps:assert native_runtime['source_sha256'][k]==sha(src/k),'native graph source differs '+k
        assert native_runtime['gt38_encoder_sha256']==ENCODER
        from envs.maniskill import get_gt_state
        import inspect
        assert native_runtime['gt_accessor_source_sha256']==sha(Path(inspect.getfile(get_gt_state)))
        runtime['collector_execution']=dict(instance=57640710,versions=live['versions'],backend=live['backend'],controller=live['controller'])
        runtime['tensor_runtime']={k:native_runtime['actual_versions'][k] for k in ['python','torch','numpy']}
        runtime['source_sha256']=native_runtime['source_sha256']
        runtime['vision_weight_sha256']=native_runtime['vision_weight_sha256']
        runtime['graph_generation_instance']=57987772
        runtime['config_source']='collector config and native graph config are independently SHA-pinned; inherited training/model fields were never executed'
        runtime['mixing_policy']='All332 potential GNN-input episodes use the same F1 native graph runtime. Collection runtime remains explicit; old576 graph tensors are excluded.'
        for name in ['native_runtime.json','native_label_contract.json','native_graph_manifest.json','native_input_manifest.json']:
            copy(native_graph_root/name,'provenance/native/'+name)
        if (root/'native_software_versions.json').exists():
            software=read(root/'native_software_versions.json')
            runtime['vision_library_versions']=software['packages']
            copy(root/'native_software_versions.json','provenance/native/native_software_versions.json')
    rows=[];max_oracle=0.;prefix_count=0
    parents={s['parent']:s for s in specs if s['kind']=='nominal'}
    for spec in specs:
        eid=spec['episode_id'];raw=root/'raw'/str(eid);graph=root/'cache'/f'ep{eid}';receipt=read(raw/'receipt.json')
        if native_rows:
            nr=native_rows[eid];graph=(native_graph_root/nr['graph_path']).parent
            assert sha(graph/'graphs.npz')==nr['graph_sha256'] and sha(graph/'receipt.json')==nr['receipt_sha256']
            assert nr['raw_sha256']==receipt['hashes']['episode.h5']
        assert receipt['approved'] and receipt['physical_safety'] and receipt['spec']==spec
        for n,h in receipt['hashes'].items():assert sha(raw/n)==h
        n=receipt['actions'];max_oracle=max(max_oracle,receipt['oracle_max_abs'])
        if spec['kind']=='recovery':
            anchor=spec['anchor'];baseline=root/'raw'/str(parents[spec['parent']]['episode_id'])
            with h5py.File(raw/'episode.h5') as a,h5py.File(baseline/'episode.h5') as b,np.load(raw/'gt_supervision.npz') as ga,np.load(baseline/'gt_supervision.npz') as gb:
                from .remediation_cache import SENSOR_KEYS
                for k in SENSOR_KEYS:np.testing.assert_array_equal(a[k][:anchor],b[k][:anchor],err_msg='prefix '+k)
                for k in ga.files:np.testing.assert_array_equal(ga[k][:anchor],gb[k][:anchor],err_msg='prefix GT '+k)
            prefix_count+=1
        paths={}
        for key,p in [('raw_path',raw/'episode.h5'),('gt_path',raw/'gt_supervision.npz'),('gnn_mask_path',raw/'gnn_mask.npz'),
                      ('physical_audit_path',raw/'physical_audit.json')]:
            paths[key]=copy(p,f'episodes/{eid}/'+p.name)
        # Sampling includes the episode identity. Preserve the original sampling
        # namespace for byte-identical source observations; logical capture ID
        # remains independent. V1 staging outputs are retained without mutation.
        gr=read(graph/'receipt.json');reuse=gr.get('reuse_proof');sampling_id=eid
        with np.load(graph/'graphs.npz',allow_pickle=False) as g:arrays={k:g[k] for k in g.files}
        if native_rows:sampling_id=int(arrays['episode_id']);assert sampling_id==native_rows[eid]['sampling_episode_id']
        if reuse:
            oldgraph=Path(reuse['old_graph']);assert sha(oldgraph)==reuse['old_graph_sha256']
            with np.load(oldgraph,allow_pickle=False) as old:
                sampling_id=int(old['episode_id'])
                for k in old.files:
                    if k not in ('episode_id','action7'):np.testing.assert_array_equal(arrays[k],old[k],err_msg='reused graph '+k)
            from .remediation_cache import matching
            assert matching(Path(reuse['old_raw']),raw/'episode.h5')[0]
        arrays['episode_id']=np.int64(sampling_id)
        dest=export/f'episodes/{eid}/graphs.npz'
        if native_rows:paths['graph_path']=copy(graph/'graphs.npz',f'episodes/{eid}/graphs.npz')
        else:
            npz(dest,arrays);paths['graph_path']=str(dest.relative_to(source))
        gr.update(sha256=sha(dest),path=paths['graph_path'],graph_sampling_episode_id=sampling_id,
            logical_capture_episode_id=eid,sampling_identity_policy='original episode RNG identity preserved for exact-input reuse; independent capture identity in manifest')
        if native_runtime:
            gr['graph_generation_instance']=57987772
            gr['graph_runtime_sha256']=sha(native_graph_root/'native_runtime.json')
        write(dest.with_name('receipt.json'),gr,immutable=True);paths['graph_receipt_path']=str(dest.with_name('receipt.json').relative_to(source))
        copy(raw/'receipt.json',f'episodes/{eid}/capture_receipt.json');copy(raw/'trace.json',f'episodes/{eid}/trace.json');copy(graph/'gate_trace.json',f'episodes/{eid}/gate_trace.json')
        rows.append(dict(episode_id=eid,parent_id=spec['parent'],branch_id=spec['branch_id'],split=spec['split'],
            distribution='nominal_reach' if spec['kind']=='nominal' else 'disturbed_robot_expert_recovery',actions=n,
            graph_sampling_episode_id=sampling_id,**paths,
            provenance=dict(kind='independent direct GT recollection; original212 excluded from GT fitting',spec=spec,
                exact_graph_reuse=read(graph/'receipt.json').get('reuse_proof'),prefix_source_episode_id=parents[spec['parent']]['episode_id'])))
    for p in (root/'snapshots').rglob('*'):
        if p.is_file():copy(p,'snapshots/'+str(p.relative_to(root/'snapshots')))
    # Tensor runtimes differ between instances. Publish no readiness event until
    # the independent native comparison has actually passed its frozen protocol.
    while not native_rows and not (root/'runtime_acceptance.json').exists():
        write(root/'export_waiting.json',dict(status='WAITING_NATIVE_RUNTIME_EQUIVALENCE',at=now()))
        time.sleep(10)
    if native_rows:
        acceptance=dict(status='PASS',mode='ALL_GRAPHS_REBUILT_IN_F1_NATIVE_RUNTIME',episodes=212,
            target_environment_sha256=native_runtime['native_environment_sha256'],prior_cross_runtime_comparison='FAIL4/6, preserved; never reclassified',
            collector_runtime_not_claimed_identical=True,native_label_contract='PASS',old576_graphs_included=False,
            graph_runtime_sha256=sha(native_graph_root/'native_runtime.json'),
            graph_runtime_path=str((export/'provenance/native/native_runtime.json').relative_to(source)))
        write(export/'native_runtime_acceptance.json',acceptance,immutable=True)
        runtime['native_runtime_acceptance']=dict(acceptance,path=str((export/'native_runtime_acceptance.json').relative_to(source)),
            sha256=sha(export/'native_runtime_acceptance.json'))
        copy(root/'f1_runtime_equivalence_FAILED.json','provenance/failed_cross_runtime_comparison.json')
    else:
        acceptance=read(root/'runtime_acceptance.json')
        if acceptance.get('status')!='PASS':raise ValueError('native runtime equivalence failed; DATA_READY forbidden')
        equivalence=read(root/'f1_runtime_equivalence.json')
        assert equivalence['status']=='PASS' and equivalence['causal_masks_geometry_provenance_exact'] is True
        assert len(equivalence['episodes'])==6 and all(r['status']=='PASS' for r in equivalence['episodes'])
        evidence_path=copy(root/'f1_runtime_equivalence.json','provenance/f1_runtime_equivalence.json')
        copy(root/'runtime_acceptance.json','provenance/runtime_acceptance.json')
        copy(root/'runtime_comparison_plan.json','provenance/runtime_comparison_plan.json')
        runtime['f1_runtime_equivalence']=dict(status='PASS',path=evidence_path,sha256=sha(root/'f1_runtime_equivalence.json'),
            target_environment_sha256=equivalence['native_environment_sha256'],scope=equivalence['scope'],not_a_proof_for_all_future_observations=True)
    write(export/'runtime_contract.json',runtime,immutable=True)
    write(export/'dataset_manifest.json',dict(dataset_id='Task10F1_GT_supplement_'+sha(root/'supplement_plan.json')[:20],episodes=rows,
        oracle_max_abs=max_oracle,prefix_identity_verified=prefix_count,old_episodes_GT_reused=0,source_runtime_explicit=True,
        supervised_actions_not_used=True,original_collection_plan_sha256=sha(source/'full_experiment_plan.json')),immutable=True)
    write(export/'copy_provenance.json',dict(files=copies,source_files_modified=False),immutable=True)
    files=[dict(path=str(p.relative_to(source)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(export.rglob('*')) if p.is_file()]
    marker=dict(schema_version=1,status='DATA_READY',task='10-F',source_instance=57640710,
        dataset_id=read(export/'dataset_manifest.json')['dataset_id'],collection_complete=True,graph_cache_readback='PASS',physical_validity='PASS',gt38_alignment='PASS',
        parent_branch_provenance='PASS',files_sealed=True,runtime_contract=str((export/'runtime_contract.json').relative_to(source)),
        runtime_contract_sha256=sha(export/'runtime_contract.json'),dataset_manifest=str((export/'dataset_manifest.json').relative_to(source)),
        dataset_manifest_sha256=sha(export/'dataset_manifest.json'),files=files)
    candidate=root/('gnn_dataset_ready'+suffix+'.candidate.json');write(candidate,marker,immutable=True)
    # Candidate is outside the watched marker path. Two complete semantic/hash reads precede publication.
    first=validate(source,ENCODER,marker_path=candidate);assert first['status']=='PASS'
    write(root/('prepublish_validation'+suffix+'_first.json'),first,immutable=True)
    code='from graph.gt_pretrain10f1.ready_contract import validate; import json; print(json.dumps(validate('+repr(str(source))+','+repr(ENCODER)+',marker_path='+repr(str(candidate))+')))'
    second=subprocess.check_output([sys.executable,'-B','-c',code],text=True,cwd=src)
    import json
    second=json.loads(second);assert second['status']=='PASS' and second['marker_sha256']==first['marker_sha256']
    write(root/('prepublish_validation'+suffix+'_independent.json'),second,immutable=True)
    # Atomic, no overwrite: the complete marker becomes visible only after all gates PASS.
    os.link(candidate,source/'gnn_dataset_ready.json')
    assert sha(source/'gnn_dataset_ready.json')==sha(candidate)
    write(root/('data_ready_publication'+suffix+'.json'),dict(status='PASS',marker_sha256=sha(candidate),dataset_id=marker['dataset_id'],
        episodes=len(rows),gnn_fit_frames=first['gnn_fit_frames'],masked_action_fit_frames=first['action_mask_off_eligible_frames'],
        at=now(),original_jobs_controlled=False),immutable=True)
    print('DATA_READY published',marker['dataset_id'],len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True)
    p.add_argument('--native-graph-root',type=Path)
    a=p.parse_args();run(a.root,a.source,a.dino,a.sam,a.native_graph_root)
