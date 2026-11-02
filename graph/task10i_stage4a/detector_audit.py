"""Raw-query diagnostics and exact cached reconstruction of initialization rules."""
from pathlib import Path
from collections import defaultdict
import hashlib
import h5py
import numpy as np
from visual.runtime.initialization import detect_initial,box_iou,PROMPTS
from .run_pipeline import read,write,CAMERAS,STAGE3

def bbox(mask):
    v,u=np.nonzero(mask)
    return [int(u.min()),int(v.min()),int(u.max()+1),int(v.max()+1)] if len(u) else None

def pixel_boxes(cxcywh,height=128,width=128):
    a=np.asarray(cxcywh);return np.column_stack((a[:,0]-a[:,2]/2,a[:,1]-a[:,3]/2,a[:,0]+a[:,2]/2,a[:,1]+a[:,3]/2))*np.array([width,height,width,height])

def run(root,baseline,corrected):
    events=read(baseline/'dino_events.json');memo={};byepisode=defaultdict(list)
    for event in events:
        with np.load(event['raw_path']) as z:memo[event['input_rgb_sha256'],event['prompt']]=(z['selected_boxes'].tolist(),z['selected_scores'].tolist())
        byepisode[str(event['episode_id'])].append(event)
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()};queries=[];initializations=[];scan_checks=[];tokenization={}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/workspace/full_pipeline_models/dino',local_files_only=True)
    for receipt in sorted((baseline/'perception').glob('*/receipt.json')):
        eid=str(read(receipt)['episode_id']);e=entries[eid];rrpath=Path(read(corrected/'perception'/eid/'receipt.json')['corrected_reference']);rr=read(rrpath)
        with np.load(rrpath.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
        with h5py.File(e['raw_path']) as f:frames={c:f[c+'/rgb'][:] for c in CAMERAS}
        source_hash={}
        for c in CAMERAS:
            for t,rgb in enumerate(frames[c]):source_hash.setdefault((c,hashlib.sha256(rgb.tobytes()).hexdigest()),[]).append(t)
        for event in byepisode[eid]:
            times=source_hash[event['camera'],event['input_rgb_sha256']]
            assert any(t<=event['decision_timestep'] for t in times),'future detector source'
            with np.load(event['raw_path']) as z:
                boxes=pixel_boxes(z['pred_boxes'][0]);scores=(1/(1+np.exp(-np.clip(z['logits'][0],-80,80)))).max(-1)
                selected=z['selected_boxes'];selected_scores=z['selected_scores'];ids=z['input_ids'][0].tolist()
                tokenization[event['prompt']]=dict(input_ids=ids,tokens=tokenizer.convert_ids_to_tokens(ids),normalized_text=event['normalized_text'],preprocessing_shape=z['pixel_values_shape'].tolist())
            oid=0 if event['prompt']==PROMPTS['peg'] else 1
            # Identical RGB may occur more than once. Retain all matching source
            # frame evaluations instead of silently assigning a new timestamp.
            for t in [x for x in times if x<=event['decision_timestep']]:
                target=np.isin(seg[t,CAMERAS.index(event['camera'])],rr['object_ids'][('peg','box')[oid]]) if exact[t] else None
                gtbox=bbox(target) if target is not None else None
                ious=np.array([box_iou(b,gtbox) for b in boxes]) if gtbox else np.zeros(len(boxes))
                siou=np.array([box_iou(b,gtbox) for b in selected]) if gtbox else np.zeros(len(selected))
                localized=ious>=.5
                top=[]
                for rank,(box,score) in enumerate(zip(selected[:8],selected_scores[:8])):
                    x0,y0,x1,y1=box;u0,u1=np.clip([int(np.floor(x0)),int(np.ceil(x1))],0,128);v0,v1=np.clip([int(np.floor(y0)),int(np.ceil(y1))],0,128)
                    region=seg[t,CAMERAS.index(event['camera']),v0:v1,u0:u1]
                    top.append(dict(rank=rank,box=box.tolist(),score=float(score),gt_iou=float(siou[rank]),area_fraction=float((x1-x0)*(y1-y0)/(128*128)),
                        gt_pixels_in_region=int(np.isin(region,rr['object_ids'][('peg','box')[oid]]).sum()) if exact[t] else None,
                        robot_pixels_in_region=int(np.isin(region,rr['robot_ids']).sum()) if exact[t] else None))
                queries.append(dict(episode=eid,parent_id=e['parent_id'],camera=event['camera'],object_id=oid,call=event['call'],decision_timestep=event['decision_timestep'],source_timestep=t,
                    reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',gt_visible_pixels=int(target.sum()) if target is not None else None,gt_box=gtbox,
                    box_threshold=event['box_threshold'],text_threshold=event['text_threshold'],query_count=len(boxes),raw_localized_queries=int(localized.sum()),
                    localized_query_max_uncalibrated_score=float(scores[localized].max()) if localized.any() else None,
                    raw_max_iou=float(ious.max()) if len(ious) else None,post_threshold_localized_count=int((siou>=.5).sum()),selected_count=len(selected),selected_top=top,
                    localized_query_not_confident_detection=True,source_rgb_sha256=event['input_rgb_sha256']))
        records=read(receipt.parent/'vision_records.json')
        for c in CAMERAS:
            found=set();previous_scan=0
            for record in (r for r in records if r['camera']==c):
                t=record['timestep']
                if record['initialization_scan_events']==previous_scan:continue
                previous_scan=record['initialization_scan_events']
                def detector(image,prompt):return memo[hashlib.sha256(image.tobytes()).hexdigest(),prompt]
                selected=detect_initial(frames[c][:t+1],detector)
                new=[i+1 for i,n in enumerate(('peg','box')) if n not in found and selected[n]['frame']==t]
                assert new==record['new_object_ids'],'cached initialization differs from real causal scan'
                scan_checks.append(dict(episode=eid,camera=c,timestep=t,new_object_ids=new,exact_cached_reconstruction=True))
                for oid,name in enumerate(('peg','box')):
                    if oid+1 not in new:continue
                    found.add(name);box=selected[name]['box'];target=np.isin(seg[t,CAMERAS.index(c)],rr['object_ids'][name]);gtbox=bbox(target)
                    initializations.append(dict(episode=eid,parent_id=e['parent_id'],camera=c,object_id=oid,decision_timestep=t,
                        selected_box=box,score=selected[name]['score'],prompt=selected[name]['prompt'],scans=selected[name]['scans'],gt_box=gtbox,gt_visible_pixels=int(target.sum()),
                        gt_iou=box_iou(box,gtbox) if gtbox else None,reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED'))
    write(root/'detector_raw_query_metrics.json',queries);write(root/'actual_initializations.json',initializations);write(root/'initialization_reconstruction_checks.json',scan_checks)
    write(root/'tokenization_and_preprocessing.json',tokenization)
    write(root/'detector_audit_scope.json',dict(status='RAW_PROPOSAL_AND_INITIALIZATION_AUDITED',model_forwards=0,query_iou_criterion=.5,
        text_threshold_effect='installed processor uses text threshold only for decoded text labels; baseline discards them; box keep uses max token score > box threshold',
        raw_query_max_scores_are_uncalibrated=True,all_queries_preserved=True,NMS='no NMS in installed GDino forward/postprocessing path',
        duplicate_frame_matching='all identical observed source matches retained; do not count repeated detector calls as independent frames',initialization_scan_checks=len(scan_checks)))
