"""Registered image tiles supply real same-scene background hard negatives.

Tile coordinates are fixed before reading GT. All embedding forwards finish and
are saved before evaluation labels. Serialized after P model inference, never
opens Internal/Final. References are diagnostic t0 stored vectors, not B0 calls.
"""
import argparse
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .dinov2_runtime import CurrentFeatures
from .dinov2_matching_audit import csv_write
from .prompt_audit import inputs
from .reference_pair_audit import rectangle_identity,curves
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def wait_p(root,coordinator):
    deadline=time.monotonic()+24*3600
    while True:
        state=read(coordinator/'pipeline_state.json')
        if state['status']=='DEVELOPMENT_P_FORWARD_GRAPH_SUMMARY_COMPLETE' or (state['status']=='RUNNING' and state['stage'] in ('paired_graph','summary')):
            assert read(coordinator/'development/pipeline_state.json')['status']=='DEVELOPMENT_P_INFERENCE_COMPLETE'
            return
        if state['status']=='FAILED':raise RuntimeError('P prerequisite failed')
        pid=state.get('pid');p=Path(f'/proc/{pid}/cmdline')
        if not p.exists() or b'graph.task10i_stage4a2.lossless_pipeline' not in p.read_bytes():
            raise RuntimeError('P coordinator handle missing without completion')
        write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_P_MODEL_INFERENCE',dependency_pid=pid,pid=os.getpid()))
        if time.monotonic()>deadline:raise TimeoutError('GPU serialization deadline')
        time.sleep(10)


def run(root,coordinator,stored):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    protocol=dict(modes=['full_patch448','bbox_patch448'],grid=[4,4],tile_size=[32,32],
        sample='historical24 first source per parent: t0/midpoint/terminal, both cameras',
        GT_selection=False,GT_forward_input=False,threshold_selected=False,source_sha256=sha(Path(__file__)),
        feature_source_sha256=sha(Path(__file__).with_name('dinov2_runtime.py')),
        stored_embeddings_sha256=sha(stored/'embeddings.npz'),independent_test=False)
    write(root/'registered_protocol.json',protocol);wait_p(root,coordinator)
    torch.set_num_threads(1);torch.manual_seed(42)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_background_tile_matching',seed=42,
            root=root/'experiments',job_type='evaluation',config=protocol) as logger:
        extractor=CurrentFeatures('/workspace/task10i_stage4a2_models/dinov2-small')
        chosen=inputs();records=[];vectors=[]
        write(root/'dinov2_model_identity.json',dict(dinov2=extractor.identity,other_models_used=False))
        for parent,(e,receipt) in sorted(chosen.items()):
            eid=str(e['episode_id']);extractor.reset(eid);sample={0,e['actions']//2,e['actions']}
            assert sha(e['raw_path'])==e['raw_sha256']
            with h5py.File(e['raw_path']) as raw:
                for t in range(e['actions']+1):
                    for camera in CAMERAS:
                        rgb=raw[camera+'/rgb'][t];extractor.observe(eid,camera,t,rgb)
                        if t not in sample:continue
                        for row in range(4):
                            for col in range(4):
                                mask=np.zeros((128,128),bool);mask[row*32:(row+1)*32,col*32:(col+1)*32]=True
                                for mode in protocol['modes']:
                                    vector=extractor.embed(camera,t,mask,mode);assert vector is not None
                                    records.append(dict(index=len(records),parent_id=parent,episode=eid,timestep=t,
                                        camera=camera,row=row,col=col,mode=mode))
                                    vectors.append(vector)
            write(root/'pipeline_state.json',dict(status='FEATURE_EXTRACTION',parents_completed=len({r['parent_id'] for r in records}),pid=os.getpid()))
        vectors=np.stack(vectors);np.savez_compressed(root/'embeddings.npz',embeddings=vectors)
        write(root/'prediction_only_tile_coordinates.json',records);write(root/'dinov2_call_trace.json',extractor.calls)
        write(root/'inference_completed.json',dict(embeddings=len(records),embedding_sha256=sha(root/'embeddings.npz'),GT_opened=False))
        # Evaluation-only boundary: reference sidecars are first opened here.
        labels=[];gts={};by_episode={str(e['episode_id']):(e,receipt) for e,receipt in chosen.values()}
        for eid,(e,receipt) in by_episode.items():
            path=Path(receipt['corrected_reference']);assert sha(path)==receipt['corrected_reference_sha256']
            with np.load(path.parent/'segmentation.npz') as z:gts[eid]=(read(path),z['actor_segmentation'].copy(),z['exact_frame'].copy())
        for r in records:
            rr,seg,exact=gts[r['episode']];t=r['timestep']
            mask=np.zeros((128,128),bool);mask[r['row']*32:(r['row']+1)*32,r['col']*32:(r['col']+1)*32]=True
            label,detail=rectangle_identity(mask,seg[t,CAMERAS.index(r['camera'])],rr) if exact[t] else (None,dict(reason='GT_UNSUPPORTED'))
            labels.append(dict(**r,actor=label,label_detail=detail,reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED'))
        refs=read(stored/'dinov2_reference_provenance.json')
        with np.load(stored/'embeddings.npz') as z:reference_vectors=z['embeddings'].copy()
        pairs=[]
        for ref in refs:
            if ref['timestep']!=0 or ref['requested'] not in ('peg','box') or ref['mode'] not in protocol['modes']:continue
            if ref['camera']=='hand_camera' and ref['requested']=='box':continue
            if ref['branch'] not in ('ORACLE_DIAGNOSTIC','PREDICTION_CROP_DIAGNOSTIC'):continue
            for r in labels:
                if r['parent_id']!=ref['parent_id'] or r['mode']!=ref['mode'] or r['actor'] is None:continue
                # Strictly past references. Same-time agreement was audited elsewhere.
                if r['timestep']<=ref['timestep']:continue
                pairs.append(dict(parent_id=r['parent_id'],reference_origin='ORACLE_IDEAL_T0' if ref['branch']=='ORACLE_DIAGNOSTIC' else 'HYPOTHETICAL_B0_T0',
                    mode=r['mode'],cross_camera=r['camera']!=ref['camera'],candidate_actor=r['actor'],
                    intended_positive=r['actor']==ref['requested'],reference_target=ref['requested'],
                    contaminated_reference=ref['target_purity']<.8,candidate_index=r['index'],reference_index=ref['index'],
                    similarity=float(reference_vectors[ref['index']]@vectors[r['index']]),
                    time_relation='PAST_REFERENCE',baseline_DINOv2_active=False))
        csv_write(root/'dinov2_positive_negative_pairs.csv',pairs)
        csv_write(root/'dinov2_reidentification_metrics.csv',curves(pairs,('reference_origin','mode','cross_camera')))
        # Candidate actor strata retain genuine background/robot denominators.
        csv_write(root/'hard_negative_metrics.csv',curves(pairs,('reference_origin','mode','cross_camera','candidate_actor')))
        write(root/'tile_actor_labels.json',labels)
        write(root/'scope.json',dict(status='BACKGROUND_TILE_MATCHING_AUDIT_COMPLETE',run_id=logger.run_id,
            parents=len(chosen),embeddings=len(records),actual_model_forwards=sum(c['model_forward'] for c in extractor.calls),
            pairs=len(pairs),background_pairs=sum(r['candidate_actor']=='background' for r in pairs),
            GT_after_saved_forward=True,GT_tile_selection=False,threshold_selected=False,independent_test=False,
            source=protocol,whole_goal_complete=False))
        write(root/'pipeline_state.json',dict(status='BACKGROUND_TILE_MATCHING_AUDIT_COMPLETE',whole_goal_complete=False))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','coordinator','stored'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(a.root,a.coordinator,a.stored)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,
            message=str(exc),partial_preserved=True,whole_goal_complete=False))
        raise
