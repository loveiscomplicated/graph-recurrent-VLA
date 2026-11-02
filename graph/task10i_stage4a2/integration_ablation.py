"""Real RGB-D C/BC engineering and historical Development causal ablations.

The engineering stage runs live B0 models and a preregistered prediction fault
to exercise past-reference Re-ID. Scientific windows have no fault injection.
All GT metrics and graphs are computed only after saved mask inference outputs.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .dinov2_runtime import CurrentFeatures
from .tri_model_integration import CausalCorrection
from .reanchor_state_machine import Rules
from .prompt_audit import AUDITS
from .dinov2_matching_audit import csv_write
from .evaluation_metrics import pixel_metrics,correct_source_coverage
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS

WINDOWS=[dict(episode='10i_stage3_10i_s1_0924_full_task',last=20),
         dict(episode='10i_stage3_10i_s1_0121_full_task',last=106),
         dict(episode='2000003',last=70)]


def evaluation(root,windows,entries,engineering):
    from graph.plan_a_bc.geometry import CurrentGeometry
    from scipy.ndimage import binary_erosion
    rt=Path('/workspace/task10g_f1_abc_20261031/full')
    schema=read('/workspace/task10f1_gt_pretrain_20261030/normalizer.json')['schema']
    pixel_rows=[];graph_rows=[];reference_rows=[];candidate_rows=[]
    for w in windows:
        eid=w['episode'];e=entries[eid]
        refpath=Path(read(AUDITS/'corrected/perception'/eid/'receipt.json')['corrected_reference']);rr=read(refpath)
        with np.load(refpath.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
        geometry={condition:CurrentGeometry(rt/'inputs',schema) for condition in ('B0','C','BC')}
        masks={}
        for condition in geometry:
            with np.load(root/'episodes'/eid/(condition+'.npz')) as z:
                masks[condition]=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
        with h5py.File(e['raw_path']) as raw:
            for t in range(w['last']+1):
                cameras,data,surfaces=geometry['B0'].from_h5(raw,t,eid)
                for condition,g in geometry.items():
                    graph,records,gate=g.build(cameras,data,surfaces,dict(zip(CAMERAS,masks[condition][t])))
                    out=root/'graphs'/eid/condition;out.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(out/f't{t:04d}.npz',**{k:getattr(graph,k).numpy() for k in
                        ('node_input','node_type','edge_index','edge_attr','edge_type')},
                        world_points=np.asarray([r['world_xyz'] for r in records]).reshape(-1,3),
                        object_id=np.asarray([r['object_id'] for r in records],np.int8))
                    write(out/f't{t:04d}_provenance.json',[r['source_observation_ids'] for r in records])
                    for ci,camera in enumerate(CAMERAS):
                        for oid in (0,1):
                            row=dict(episode=eid,parent_id=e['parent_id'],condition=condition,timestep=t,
                                camera=camera,object_id=oid,reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                                engineering_fault_injection=engineering)
                            if exact[t]:
                                row.update(pixel_metrics(masks[condition][t,ci,oid],
                                    np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]]),
                                    np.isin(seg[t,ci],rr['robot_ids']),np.isin(seg[t,ci],rr['object_ids'][('box','peg')[oid]])))
                            pixel_rows.append(row)
                    for oid in (0,1):
                        selected=[r for r in records if r['object_id']==oid]
                        row=dict(episode=eid,parent_id=e['parent_id'],condition=condition,timestep=t,object_id=oid,
                            points=len(selected),reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED')
                        if exact[t]:
                            xyz=np.asarray([r['world_xyz'] for r in selected]).reshape(-1,3)
                            labels=[]
                            for r in selected:
                                _,_,c,p=r['source_observation_ids'][0];actor=int(seg[t,CAMERAS.index(c)].ravel()[p])
                                labels.append('correct' if actor in rr['object_ids'][('peg','box')[oid]] else
                                    'robot' if actor in rr['robot_ids'] else
                                    'wrong_object' if actor in rr['object_ids'][('box','peg')[oid]] else 'background')
                            correct=np.asarray([v=='correct' for v in labels]);visible=[];boundary=[]
                            for ci,c in enumerate(cameras):
                                if oid==1 and ci==1:continue
                                target=np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]])&np.isfinite(c.depth)
                                world=c.world_points();visible.append(world[target]);boundary.append(world[target&~binary_erosion(target)])
                            visible=np.concatenate(visible).reshape(-1,3);boundary=np.concatenate(boundary).reshape(-1,3)
                            row.update(correct_points=int(correct.sum()),
                                correct_source_coverage=correct_source_coverage(xyz,correct,visible),
                                correct_boundary_coverage=correct_source_coverage(xyz,correct,boundary),
                                **{label+'_points':labels.count(label) for label in ('robot','wrong_object','background')})
                        graph_rows.append(row)
        for condition in ('C','BC'):
            folder=root/'episodes'/eid/condition
            for r in read(folder/'reference_provenance.json'):
                t=r['timestep'];ci=CAMERAS.index(r['camera']);oid=r['object_id']
                mask=np.unpackbits(np.asarray(r['mask_bits'],np.uint8)).reshape(128,128).astype(bool)
                purity=float(np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]])[mask].mean()) if exact[t] else None
                reference_rows.append(dict(episode=eid,condition=condition,camera=r['camera'],object_id=oid,timestep=t,
                    reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',gt_semantic_purity=purity,
                    contaminated_at_80pct=purity<.8 if purity is not None else None))
            for r in read(folder/'candidate_masks.json'):
                t=r['timestep'];ci=CAMERAS.index(r['camera']);oid=r['object_id']
                mask=np.unpackbits(np.asarray(r['mask_bits'],np.uint8)).reshape(128,128).astype(bool)
                purity=float(np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]])[mask].mean()) if exact[t] and mask.any() else None
                candidate_rows.append(dict(**{k:v for k,v in r.items() if k!='mask_bits'},condition=condition,
                    reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',gt_semantic_purity=purity,
                    correct_at_80pct=purity>=.8 if purity is not None else None))
    csv_write(root/'tri_model_pixel_metrics.csv',pixel_rows);csv_write(root/'final_graph_control_metrics.csv',graph_rows)
    csv_write(root/'reference_contamination.csv',reference_rows);csv_write(root/'reid_candidate_outcomes.csv',candidate_rows)


def run(root,engineering=False,uncertain_output_policy='suppress'):
    from utils.experiment_logger import ExperimentLogger
    from graph.task10i_stage4a.instrumentation import AuditedDino
    from graph.task10i_stage4a.development_perception import Memo
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from visual.runtime.tracking import SAM2Tracker
    from graph.plan_a_bc.geometry import CurrentGeometry
    root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(42)
    assert not torch.are_deterministic_algorithms_enabled() and not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    windows=[dict(episode='2000003',last=20)] if engineering else WINDOWS
    source=Path(__file__).resolve().parents[2]
    source_files={str(p.relative_to(source)):sha(p) for folder in ('graph','visual/runtime','utils') for p in (source/folder).rglob('*.py')}
    rt=Path('/workspace/task10g_f1_abc_20261031/full');paths=read(rt/'runtime_paths.json')
    import sam2
    installed=Path(sam2.__file__).parent
    model_identity=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json')
    assert sha(paths['sam'])==model_identity['sam_sha256']
    assert str(torch.__version__)==model_identity['torch_version']
    assert all(sha(Path(paths['dino'])/p)==h for p,h in model_identity['dino_files'].items())
    spec=dict(windows=windows,conditions=['C','BC'],engineering=engineering,
        uncertain_output_policy=uncertain_output_policy,
        engineering_fault=dict(camera='base_camera',object_id=0,empty_prediction_steps=list(range(8,19))) if engineering else None,
        engineering_live_baseline=engineering,baseline='historical verified cache' if not engineering else 'live original model forwards checked against historical bytes',
        GT_practical_input=False,holdout_or_final_access=False,rules=asdict(Rules(minimum_detection_score=.2)),
        feature='same-camera full-image448 masked patch; first supported reference frozen',
        camera_object_slots=[['base_camera',0],['hand_camera',0],['base_camera',1]],
        preprocessing='original JPEG SAM2; raw RGB DINO/DINOv2',model_identity=model_identity,source_files=source_files,
        installed_sam_source_sha256=sha(installed/'sam2_video_predictor.py'),
        installed_sam_config_sha256=sha(installed/'configs/sam2.1/sam2.1_hiera_s.yaml'),
        packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','transformers','tokenizers','numpy')},
        graph_protocol_sha256=sha(source/'graph/plan_a_geometry/protocol.json'),
        schema_sha256=sha('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'),
        robot_assets={str(p.relative_to(rt/'inputs')):sha(p) for p in (rt/'inputs').rglob('*') if p.is_file()})
    write(root/'three_model_integration_protocol.json',spec)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_causal_C_BC',seed=42,
         root=root/'experiments',job_type='evaluation',config=spec) as logger:
        detector=AuditedDino(device='cuda',model_id=paths['dino'],output=root/'dino_raw');memo=Memo(detector)
        tracker=SAM2Tracker(device='cuda',checkpoint=paths['sam'])
        features=CurrentFeatures('/workspace/task10i_stage4a2_models/dinov2-small')
        write(root/'dinov2_model_identity.json',features.identity)
        schema=read('/workspace/task10f1_gt_pretrain_20261030/normalizer.json')['schema']
        sam_calls=[];module_calls=[];original=tracker.predictor._run_single_frame_inference
        original_get=tracker.predictor._get_image_feature
        def observe_get(*args,**kwargs):
            frame=kwargs.get('frame_idx',args[1] if len(args)>1 else None)
            old_context=getattr(tracker,'research_context',{}).copy()
            context=dict(old_context);context.update(image_frame_idx=frame,
                input_observation_timestep=old_context.get('anchor_t',0)+frame)
            assert context['input_observation_timestep']<=context['timestep'],'future SAM2 image read'
            tracker.research_context=context
            try:return original_get(*args,**kwargs)
            finally:tracker.research_context=old_context
        tracker.predictor._get_image_feature=observe_get
        handles=[]
        def module_observer(name):
            def observer(module,args,output):
                row=dict(**getattr(tracker,'research_context',{}),module=name)
                row['tensor_arg_shapes']=[list(v.shape) for v in args if torch.is_tensor(v)]
                if name=='sam_mask_decoder':
                    assert len(output)==4
                    row['mask_shape']=list(output[0].shape)
                    row['internal_iou_predictions']=output[1].detach().float().cpu().tolist()
                    row['object_score_logits']=output[3].detach().float().cpu().tolist()
                    row['quality_is_semantic_confidence']=False
                module_calls.append(row)
            return observer
        for name in ('image_encoder','memory_encoder','sam_mask_decoder'):
            handles.append(getattr(tracker.predictor,name).register_forward_hook(module_observer(name)))
        def observe(*args,**kwargs):
            result=original(*args,**kwargs)
            row=dict(**getattr(tracker,'research_context',{}),local_frame_idx=kwargs.get('frame_idx'))
            row['output_keys']=list(result[0])
            row['object_score_logits']=result[0]['object_score_logits'].detach().float().cpu().reshape(-1).tolist()
            row['is_init_cond_frame']=kwargs.get('is_init_cond_frame')
            row['run_mem_encoder']=kwargs.get('run_mem_encoder')
            state=kwargs.get('inference_state',{})
            objidx=next((i for i,value in state.get('output_dict_per_obj',{}).items() if value is kwargs.get('output_dict')),None)
            row['tracker_object_id']=state.get('obj_idx_to_id',{}).get(objidx)
            if row.get('kind')=='candidate_prompt':
                out=root/'sam_prompt_logits';out.mkdir(exist_ok=True)
                path=out/f'call{len(sam_calls):06d}.npz'
                np.savez_compressed(path,logits=result[0]['pred_masks'].detach().float().cpu().numpy())
                row['raw_logits_path']=str(path)
            sam_calls.append(row);return result
        tracker.predictor._run_single_frame_inference=observe
        try:
            for w in windows:
                eid=w['episode'];e=entries[eid]
                assert w['last']<=e['actions'] and sha(e['raw_path'])==e['raw_sha256']
                with np.load(AUDITS/'baseline/perception'/eid/'masks.npz') as z:
                    baseline_bits=z['mask_bits'][:w['last']+1].copy()
                record_list=read(AUDITS/'baseline/perception'/eid/'vision_records.json')
                folder=root/'episodes'/eid;folder.mkdir(parents=True)
                np.savez_compressed(folder/'B0.npz',mask_bits=baseline_bits)
                for condition in ('C','BC'):
                    geometry=CurrentGeometry(rt/'inputs',schema)
                    policy=CausalCorrection(memo,tracker,geometry,features=features if condition=='BC' else None,
                        uncertain_output_policy=uncertain_output_policy)
                    policy.reset(eid);values=[];found={};times=[];feature_start=len(features.calls);sam_start=len(sam_calls)
                    adapters={c:OnlineVisionAdapter(memo,tracker) for c in CAMERAS} if engineering else {}
                    for a in adapters.values():a.reset(eid)
                    try:
                        with h5py.File(e['raw_path']) as raw:
                            for t in range(w['last']+1):
                                detector.context=dict(episode_id=eid,condition=condition,decision_timestep=t)
                                cameras,data,surfaces=geometry.from_h5(raw,t,eid)
                                baseline=dict(zip(CAMERAS,np.unpackbits(baseline_bits[t]).reshape(2,2,128,128).astype(bool)))
                                if engineering:
                                    actual=[]
                                    for c in cameras:
                                        tracker.research_context=dict(episode=eid,camera=c.camera,timestep=t,anchor_t=0,kind='live_B0')
                                        mask,_=adapters[c.camera].process(c.rgb,eid,t);actual.append(mask)
                                    assert np.array_equal(np.packbits(np.stack(actual).ravel()),baseline_bits[t]),('live engineering B0 mismatch',t)
                                    baseline=dict(zip(CAMERAS,actual))
                                for r in record_list:
                                    if r['timestep']==t:
                                        for oid in r['new_object_ids']:found[r['camera'],oid-1]=True
                                if engineering and 8<=t<=18:baseline['base_camera'][0]=np.zeros((128,128),bool)
                                tick=time.perf_counter();current=policy.process(cameras,data,surfaces,baseline,found)
                                times.append(1000*(time.perf_counter()-tick));values.append(np.packbits(np.stack([current[c] for c in CAMERAS]).ravel()))
                                if t%20==0:print('causal integration',engineering,eid,condition,t,flush=True)
                        np.savez_compressed(folder/(condition+'.npz'),mask_bits=np.stack(values))
                        out=folder/condition;out.mkdir()
                        for name,obj in [('events.json',policy.events),('candidate_masks.json',policy.candidate_masks),
                            ('reference_provenance.json',policy.reference_provenance),('feature_calls.json',features.calls[feature_start:]),
                            ('sam_calls.json',sam_calls[sam_start:])]:write(out/name,obj)
                        write(out/'state_machine.json',{str(key):slot.events for key,slot in policy.slots.items()})
                        write(out/'receipt.json',dict(condition=condition,frames=len(values),raw_sha256=e['raw_sha256'],
                            GT_input=False,actual_dinov2_forwards=sum(r['model_forward'] for r in features.calls[feature_start:]),
                            actual_reid_comparisons=sum(p['reference_available'] and p['similarity'] is not None for ev in policy.events for p in ev['proposals']),
                            sam_forward_calls=len(sam_calls)-sam_start,accepted=sum(r['accepted'] for r in policy.events),
                            mask_sha256=sha(folder/(condition+'.npz')),mean_correction_ms=float(np.mean(times)),
                            diagnostic_not_fair_latency=True))
                    finally:
                        policy.close()
                        for a in adapters.values():a.close()
                        adapters.clear();torch.cuda.empty_cache()
                memo.cache.clear();write(root/'dino_events.json',detector.events)
            write(root/'inference_completed.json',dict(status='INFERENCE_COMPLETE',windows=windows))
            write(root/'sam2_actual_module_call_trace.json',module_calls)
            evaluation(root,windows,entries,engineering)
            if engineering:
                r=read(root/'episodes/2000003/BC/receipt.json')
                assert r['actual_dinov2_forwards']>0 and r['actual_reid_comparisons']>0 and r['sam_forward_calls']>0
                assert detector.events and list((root/'graphs').rglob('*.npz'))
            write(root/'scope.json',dict(status='TRI_MODEL_ENGINEERING_COMPLETE' if engineering else 'HISTORICAL_C_BC_ABLATION_COMPLETE',
                uncertain_output_policy=uncertain_output_policy,
                run_id=logger.run_id,windows=windows,GT_metrics_after_saved_inference=True,
                engineering_fault_injected=engineering,practical_trigger_GT_free=True,independent_final=False,
                full_population_estimate=False,dino_actual_forwards=len(detector.events),sam_actual_forwards=len(sam_calls),
                dinov2_actual_forwards=sum(r['model_forward'] for r in features.calls),
                sam2_module_calls={name:sum(r['module']==name for r in module_calls) for name in
                    ('image_encoder','memory_encoder','sam_mask_decoder')}))
            logger.link_output(root/'scope.json',kind='metrics')
        finally:
            for handle in handles:handle.remove()
            tracker.predictor._run_single_frame_inference=original
            tracker.predictor._get_image_feature=original_get;detector.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--engineering',action='store_true')
    p.add_argument('--uncertain-output-policy',choices=['suppress','preserve_current'],default='suppress')
    a=p.parse_args();run(a.root,a.engineering,a.uncertain_output_policy)
