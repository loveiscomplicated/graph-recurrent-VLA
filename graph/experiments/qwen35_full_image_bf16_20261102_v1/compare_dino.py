from common import *
import numpy as np, collections
from visual.runtime.initialization import assign,box_iou
v=rd(R/'verification.json');assert v['stage']=='EVALUATION_COMPLETE'
assert sha(R/'full_image_grounding_raw_responses.json')==v['raw_response_SHA_before_GT']
v['DINO_outputs_opened']=True;wr('verification.json',v)
mf=rd(R/'input_manifest.json');events=[];hashes={}
p=Path('/workspace/task10i_stage4a2_newdev_A_20261101_v1/run/development/dino_events.json')
if p.exists():
 hashes[str(p)]=sha(p)
 events=[e for e in rd(p) if e.get('condition')=='B0']
qrows=list(csv.DictReader((R/'.execution/object_evaluation.csv').open()));paired=[];coverage=[]
for c in [c for c in mf['cases'] if c['phase']=='B']:
 per={}
 for k,prompt in [('peg','red and white block'),('box','cardboard box')]:
  found=[e for e in events if e.get('input_rgb_sha256')==c['RGB_sha256'] and e.get('camera')==c['camera'] and e.get('prompt')==prompt and str(e.get('episode_id'))==c['episode'] and e.get('decision_timestep',c['timestep'])<=c['timestep']]
  found=sorted(found,key=lambda e:e['call'])
  if found:
   e=found[0];path=Path(e['raw_path']);hashes[str(path)]=sha(path)
   with np.load(path) as z: boxes=z['selected_boxes'].tolist();scores=z['selected_scores'].tolist()
   eligible=[(b,float(s)) for b,s in zip(boxes,scores) if (b[2]-b[0])*(b[3]-b[1])<.25*128**2][:5]
   per[k]=dict(event=e,eligible=eligible,raw_postprocessed_boxes=len(boxes),raw_top_box=boxes[int(np.argmax(scores))] if scores else None)
 coverage.append(dict(id=c['id'],camera=c['camera'],frame_kind=c['frame_kind'],matched_labels=sorted(per)))
 if set(per)!={'peg','box'}:continue
 joint=assign(per['peg']['eligible'],per['box']['eligible'])
 for k,chosen in zip(['peg','box'],joint):
  qr=next(r for r in qrows if r['id']==c['id'] and r['object']==k)
  gb=json.loads(qr['GT_bbox']) if qr['GT_bbox'] and qr['GT_bbox']!='None' else None
  # CSV None fields are empty; list serialization above is literal JSON.
  db=chosen[0] if chosen else None;top=per[k]['raw_top_box'];npix=int(qr['visible_pixels']) if qr['visible_pixels'] else None
  paired.append(dict(id=c['id'],parent_id=c['parent_id'],camera=c['camera'],frame_kind=c['frame_kind'],object=k,GT_status=qr['GT_status'],visible_pixels=npix,GT_bbox=gb,DINO_consumer_box=db,DINO_consumer_IoU=box_iou(db,gb) if db and gb else 0. if gb else None,DINO_raw_top_box=top,DINO_raw_top_IoU=box_iou(top,gb) if top and gb else 0. if gb else None,Qwen_strict_IoU=float(qr['bbox_IoU']) if qr['bbox_IoU'] else None,Qwen_prediction_status=qr['prediction_status'],DINO_source_raw=per[k]['event']['raw_path'],DINO_source_RGB_sha256=per[k]['event']['input_rgb_sha256'],RGB_sha256=c['RGB_sha256']))
def ratio(n,d):return dict(numerator=n,denominator=d,rate=n/d if d else None)
summary={}
for k in ['peg','box']:
 rs=[r for r in paired if r['object']==k and r['GT_status']=='EXACT_GT' and r['visible_pixels']>=16]
 summary[k]=dict(evaluable_pairs=len(rs),parents=len({r['parent_id'] for r in rs}),DINO_joint_IoU05_recall=ratio(sum(r['DINO_consumer_IoU']>=.5 for r in rs),len(rs)),DINO_raw_top_IoU05_recall=ratio(sum(r['DINO_raw_top_IoU']>=.5 for r in rs),len(rs)),Qwen_strict_IoU05_recall=ratio(sum(r['Qwen_strict_IoU']>=.5 for r in rs),len(rs)),DINO_joint_mean_IoU=float(np.mean([r['DINO_consumer_IoU'] for r in rs])) if rs else None,Qwen_strict_mean_IoU=float(np.mean([r['Qwen_strict_IoU'] for r in rs])) if rs else None)
status='MATCHED_DINO_BASELINE_INSUFFICIENT'
wr('.execution/dino_matched_comparison.json',dict(status=status,source='preserved B0 events only; no new DINO inference',match_contract='exact episode + camera + RGB bytes SHA; no later event',consumer_reconstruction='unchanged .25 area/top5/.3 missing/.3 exclusion assign at matched frame; counterfactual for later frames, not actual late consumer decision',coverage=coverage,matched_object_rows=paired,summary=summary,SAM2_matched_comparison='INSUFFICIENT_SUPPORT: no common Qwen-initialized masks',latency_matched_comparison='NOT_AVAILABLE: no fresh detector timing; old global latency not substituted',conclusion='limited common frames only; no claim of relative superiority or replacement',source_hashes=hashes))
v.update(DINO_comparison_status=status,DINO_matched_source_hashes=hashes);wr('verification.json',v)
print('DINO_MATCHED',len(paired),json.dumps(summary),flush=True)
