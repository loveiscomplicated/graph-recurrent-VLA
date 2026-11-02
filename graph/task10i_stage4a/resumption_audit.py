"""Read-only migration/resumption inspection. Never launches research stages."""
import hashlib
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np

ROOT=Path('/workspace/task10i_stage4a_20261031_v5/run')
REPAIR=Path('/workspace/task10i_stage4a_20261031_v6/run')
STAGE3=Path('/workspace/task10i_stage3_20261031_v5_finalization/run')
CAMERAS=('base_camera','hand_camera')

def payload(p):
    p=Path(p)
    if p.exists():return p.read_bytes()
    with gzip.open(str(p)+'.gz','rb') as f:return f.read()
def read(p):return json.loads(payload(p))
def sha(p):
    p=Path(p);h=hashlib.sha256()
    stream=p.open('rb') if p.exists() else gzip.open(str(p)+'.gz','rb')
    with stream as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def npz_arrays(p):
    with Path(p).open('rb') as stream:
        with np.load(stream,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def audit():
    at=time.time();failures=[];results=dict(inspection_only=True,research_stages_started=False,model_forward_executed=False,simulator_executed=False,inputs_modified=False)
    def check(condition,label,detail=None):
        if not condition:failures.append(dict(check=label,detail=detail))
    results['storage']=dict(total=shutil.disk_usage(ROOT).total,used=shutil.disk_usage(ROOT).used,free=shutil.disk_usage(ROOT).free)
    receipt=read(STAGE3/'stage4_raw_dataset_receipt.json');manifest=read(STAGE3/'stage4_recache_input_manifest.json')
    check(sha(STAGE3/'stage4_recache_input_manifest.json')==receipt['recache_manifest_sha256'],'stage3_manifest_identity')
    check(sha(STAGE3/'stage4_source_and_schema_contract.json')==receipt['schema_sha256'],'stage3_schema_identity')
    for name,ref in manifest['datasets'].items():check(sha(ref['path'])==ref['sha256'],'dataset_manifest_identity',name)
    raw_checks=[]
    for eid,e in manifest['unique_sources'].items():
        for field in ('raw','gt'):
            p=Path(e[field+'_path']);actual=sha(p);passed=actual==e[field+'_sha256'];check(passed,'raw_or_gt_sha',str(p))
            raw_checks.append(dict(episode_id=eid,kind=field,path=str(p),sha256=actual,passed=passed))
    results['raw_gt_integrity']=dict(files_checked=len(raw_checks),all_passed=all(r['passed'] for r in raw_checks),rows=raw_checks)
    protocol=read(ROOT/'protocol.json');previous=read('/workspace/task10i_stage4a_20261031_v3/run/protocol.json')
    check(protocol==previous,'preregistered_protocol_preserved')
    dev=set(protocol['development']);entries={str(e['episode_id']):e for e in manifest['unique_sources'].values() if e['parent_id'] in dev}
    completed=[];remaining=[];reference_rows=[];graph_count=0;source_count=0;trace_bytes=0
    for eid,e in sorted(entries.items()):
        folder=ROOT/'perception'/eid;p=folder/'receipt.json'
        if not p.exists():
            remaining.append(dict(episode_id=eid,parent_id=e['parent_id'],branch=e['branch'],source=e['source'],actions=e['actions'],observations=e['actions']+1,
                raw_path=e['raw_path'],gt_path=e['gt_path'],partial_perception_directory=folder.exists(),partial_trace_frames=len(list((ROOT/'traces'/eid).glob('t*')))))
        else:
            cr=read(p);check(cr['raw_sha256']==e['raw_sha256'],'completed_receipt_raw_identity',eid)
            check(sha(folder/'masks.npz')==cr['masks_sha256'],'completed_masks_sha',eid)
            check(sha(folder/'stage_rows.json')==cr['rows_sha256'],'completed_stage_rows_sha',eid)
            masks=npz_arrays(folder/'masks.npz');bits=masks['mask_bits'];n=e['actions']+1
            check(bits.shape==(n,8192) and bits.dtype==np.uint8,'mask_array_contract',eid)
            rows=read(folder/'stage_rows.json');rep=read(folder/'representative_points.json');vision=read(folder/'vision_records.json')
            check(len(rows)==n*40,'stage_row_count',eid);check(len(rep)==n*2,'representative_metric_count',eid);check(len(vision)==n*2,'vision_record_count',eid)
            mask_counts=np.unpackbits(bits,axis=1).reshape(n,2,2,128,128).sum((3,4))
            for row in rows:
                if row['stage']=='sam2':check(int(mask_counts[row['timestep'],CAMERAS.index(row['camera']),row['object_id']])==row['output_support'],'sam2_mask_funnel_support',eid)
            frame_hash=hashlib.sha256();episode_sources=0
            for t in range(n):
                tf=ROOT/'traces'/eid/f't{t:04d}';data=npz_arrays(tf/'support.npz');trace=read(tf/'trace.json');N=data['node_input'].shape[0];E=data['edge_index'].shape[1];P=data['point_world'].shape[0]
                check(data['node_input'].shape==(N,21) and data['edge_index'].shape==(2,E) and data['edge_attr'].shape==(E,4),'graph_tensor_shapes',[eid,t])
                check(data['node_type'].shape==(N,) and data['edge_type'].shape==(E,) and N==P+3,'graph_point_node_contract',[eid,t])
                check(np.isfinite(data['node_input']).all() and np.isfinite(data['edge_attr']).all(),'graph_finite',[eid,t])
                check(np.array_equal(data['node_type'][:3],np.arange(3)) and (data['node_input'][:,14:]==0).all(),'robot_nodes_and_disabled_temporal_fields',[eid,t])
                check(all(int((data['point_object_id']==oid).sum())<=128 for oid in (0,1)),'cap128',[eid,t])
                check(E==0 or (data['edge_index'].min()>=0 and data['edge_index'].max()<N),'graph_edge_endpoints',[eid,t])
                check(len(trace['representative_sources'])==P and len(trace['rows'])==40,'trace_source_and_stage_count',[eid,t])
                for point in trace['representative_sources']:
                    for ep,ts,c,pixel in point['sources']:
                        episode_sources+=1
                        check(str(ep)==eid and ts==t and c in CAMERAS and 0<=pixel<128*128,'current_pixel_source_identity',[eid,t])
                        if point['object_id']==1:check(c=='base_camera','box_main_only',[eid,t])
                        if c=='hand_camera':check(trace['gate']['accepted'],'wrist_gate_source_contract',[eid,t])
                frame_hash.update(sha(tf/'support.npz').encode());frame_hash.update(sha(tf/'trace.json').encode());trace_bytes+=(tf/'support.npz').stat().st_size
                graph_count+=1
            source_count+=episode_sources
            completed.append(dict(episode_id=eid,parent_id=e['parent_id'],frames=n,receipt_sha256=sha(p),masks_sha256=sha(folder/'masks.npz'),stage_rows_sha256=sha(folder/'stage_rows.json'),
                representative_metrics_sha256=sha(folder/'representative_points.json'),vision_records_sha256=sha(folder/'vision_records.json'),trace_tree_digest=frame_hash.hexdigest(),source_observations_checked=episode_sources))
        ref_folder=ROOT/'references'/eid;oldrr=read(ref_folder/'receipt.json')
        rp=REPAIR/'references'/eid/'receipt.json';chosen=rp.parent if rp.exists() else ref_folder;rr=read(chosen/'receipt.json')
        check(rr['raw_sha256']==e['raw_sha256'] and rr['gt_sha256']==e['gt_sha256'],'reference_raw_gt_identity',eid)
        check(sha(chosen/'segmentation.npz')==rr['segmentation_sha256'],'reference_segmentation_sha',eid)
        reference=npz_arrays(chosen/'segmentation.npz');shape=reference['actor_segmentation'].shape
        check(shape==(e['actions']+1,2,128,128),'reference_array_shape',eid)
        check(reference['exact_frame'].shape==(e['actions']+1,) and reference['exact_frame'].all(),'all_frames_exact_reference',eid)
        check(int(reference['exact_frame'].sum())==rr['exact_frames'],'reference_exact_count',eid)
        check(set(rr['object_ids'])=={'peg','box'} and not set(rr['object_ids']['peg'])&set(rr['object_ids']['box']),'actor_id_mapping',eid)
        npz_arrays(ref_folder/'meshes.npz');geom=npz_arrays(ref_folder/'geometry_reference.npz')
        check(geom['actor_pose'].shape==(e['actions']+1,2,7) and np.isfinite(geom['actor_pose']).all(),'geometry_reference_shape',eid)
        reference_rows.append(dict(episode_id=eid,path=str(chosen),frames=e['actions']+1,exact_frames=rr['exact_frames'],segmentation_sha256=sha(chosen/'segmentation.npz'),repaired=rp.exists()))
    results['development']=dict(parents=len(dev),episodes=len(entries),expected_observations=sum(e['actions']+1 for e in entries.values()),completed_episodes=len(completed),completed_frames=sum(r['frames'] for r in completed),
        graph_frame_archives_read=graph_count,source_observations_checked=source_count,npz_trace_bytes_read=trace_bytes,completed=completed,remaining=remaining)
    results['reference_integrity']=dict(episodes=len(reference_rows),exact_frames=sum(r['exact_frames'] for r in reference_rows),rows=reference_rows)
    events=read(ROOT/'dino_events.json');query_files=[]
    for event in events:
        z=npz_arrays(event['raw_path']);check(z['logits'].ndim==3 and z['pred_boxes'].shape[:2]==z['logits'].shape[:2] and z['pred_boxes'].shape[-1]==4,'raw_dino_query_shape',event['raw_path'])
        query_files.append(dict(path=event['raw_path'],sha256=sha(event['raw_path'])))
    results['raw_dino']=dict(events=len(events),all_archives_read=True,files=query_files)
    frozen=read(ROOT/'source_freeze.json');source=Path('/workspace/task10i_stage4a_20261031_v5/source');source_mismatch=[]
    for name,expected in frozen['files'].items():
        p=source/name
        if not p.exists() or sha(p)!=expected:source_mismatch.append(name)
    check(not source_mismatch,'frozen_source_identity',source_mismatch)
    results['source']=dict(frozen_files_checked=len(frozen['files']),mismatches=source_mismatch,
        resume_entrypoint_in_copied_v5=(source/'graph/task10i_stage4a/resume_baseline.py').exists())
    identities=read(ROOT/'model_identity.json');runtime=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json');models=[]
    check(sha(runtime['sam'])==identities['sam_sha256'],'sam_checkpoint_sha')
    models.append(dict(kind='sam',path=runtime['sam'],sha256=sha(runtime['sam'])))
    for name,expected in identities['dino_files'].items():
        p=Path(runtime['dino'])/name;actual=sha(p);check(actual==expected,'dino_model_file_sha',name);models.append(dict(kind='dino',path=str(p),sha256=actual))
    results['models']=dict(files=models,expected_torch=identities['torch_version'],expected_settings=identities['settings'])
    assets=Path('/workspace/task10g_f1_abc_20261031/full/inputs');asset_identity=read(assets/'geometry_identity.json')
    check(sha(assets/'robot.urdf')==asset_identity['robot_urdf_sha256'],'robot_urdf_sha')
    npz_arrays(assets/'robot_meshes.npz')
    results['required_runtime_files']={str(p):p.is_file() for p in [Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'),Path('/workspace/task10g_f1_abc_20261031/full/normalizer.json'),
        Path('/workspace/full_pipeline_sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml'),Path('/workspace/full_pipeline_sam2/sam2/build_sam.py')]}
    for p,exists in results['required_runtime_files'].items():check(exists,'required_runtime_file',p)
    results['status']='DATA_RESUMPTION_AUDIT_PASS' if not failures else 'DATA_RESUMPTION_AUDIT_HAS_FAILURES'
    results['failures']=failures;results['elapsed_seconds']=time.time()-at
    return results

if __name__=='__main__':
    try:result=audit()
    except BaseException as exc:result=dict(status='AUDIT_FAILED',exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),research_stages_started=False)
    print(json.dumps(result,indent=2,allow_nan=False))
