from common import *
import collections
import numpy as np
from PIL import Image,ImageDraw
from utils.experiment_logger import ExperimentLogger

v=rd(R/'verification.json');assert v['stage']=='PREDICTIONS_FROZEN' and v['completed_requests']==298
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT']
assert sha(R/'verifier_predictions.csv')==v['predictions_sha256_before_GT']
assert sha(R/'.execution/candidate_manifest.json')==v['candidate_manifest_sha256']
raw=rd(R/'verifier_raw_responses.json')['responses'];pred={(r['candidate_id'],r['condition']):r for r in raw}
cm=rd(R/'.execution/candidate_manifest.json');candidates=cm['candidates'];images={c['id']:c for c in cm['images']};parents=cm['parents']
old={r['id']:r['strict_label'] for r in rd(V/'verifier_raw_responses.json')['responses']}
v.update(stage='POSTHOC_GT_EVALUATION',GT_opened=True);wr('verification.json',v)
gcases={c['id']:c for c in rd(D/'.execution/evaluation_manifest.json')['cases']}
gtinfo={};truth={}
def bbox(mask):
 y,x=np.nonzero(mask);return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
for c in images.values():
 g=gcases[c['id']];assert sha(g['GT_path'])==g['GT_sha256']
 with np.load(g['GT_path']) as z:seg=z['actor_segmentation'].copy()
 masks={k:np.isin(seg,g['object_ids'][k]) for k in ['peg','box','robot']};masks['background']=~(masks['peg']|masks['box']|masks['robot']);truth[c['id']]=masks
 gtinfo[c['id']]=dict(bboxes={k:bbox(m) for k,m in masks.items()},visible_pixels={k:int(m.sum()) for k,m in masks.items()},GT_sha256=g['GT_sha256'])
yy,xx=np.indices((512,512));conditions=['current_only','with_initial_RGB'];rows=[];byimage=collections.defaultdict(list)
for cand in candidates:
 b=cand['box'];g=gtinfo[cand['image_id']];rect=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]);counts={k:int((rect&m).sum()) for k,m in truth[cand['image_id']].items()};fg=sum(counts[k] for k in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda k:counts[k]);role=dom.upper() if fg>=16 and counts[dom]>=16 and counts[dom]/fg>=.7 else 'OTHER' if fg<16 else 'AMBIGUOUS'
 for condition in conditions:
  p=pred[(cand['id'],condition)];lab=p['strict_label'];target=cand['query_object'];ov=iou(b,g['bboxes'][target])
  row=dict(**cand,condition=condition,strict_label=lab,historical_label=old[cand['id']],GT_role=role,actor_pixels=counts,role_correct=lab==role if role!='AMBIGUOUS' else None,format_valid=lab in LABELS,peg_IoU=iou(b,g['bboxes']['peg']),box_IoU=iou(b,g['bboxes']['box']),target_bbox_IoU=ov,target_good=ov>=.5,accepted_for_query=lab==target.upper(),rejected_good=ov>=.5 and lab!=target.upper(),accepted_bad=ov<.5 and lab==target.upper())
  rows.append(row)
 byimage[cand['image_id']].append(cand)
csvwr('candidate_evaluation.csv',rows)
selected=[]
for image in images.values():
 for obj in ['peg','box']:
  pool=sorted([c for c in byimage[image['id']] if c['query_object']==obj],key=lambda c:(-c['DINO_score'],c['rank']))
  for method in ['raw_top1','historical_verifier']+conditions:
   eligible=pool if method=='raw_top1' else [c for c in pool if (old[c['id']] if method=='historical_verifier' else pred[(c['id'],method)]['strict_label'])==obj.upper()]
   pick=eligible[0] if eligible else None;b=pick['box'] if pick else None;g=gtinfo[image['id']];gb=g['bboxes'][obj];ov=iou(b,gb)
   if b:
    rect=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]);counts={k:int((rect&m).sum()) for k,m in truth[image['id']].items()}
   else:counts={k:0 for k in ['peg','box','robot','background']}
   fg=sum(counts[k] for k in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda k:counts[k]);wrong=bool(b and ov<.5 and dom!=obj and fg>=16 and counts[dom]/fg>=.7)
   selected.append(dict(method=method,image_id=image['id'],parent_id=image['parent_id'],camera=image['camera'],object=obj,candidate_id=pick['id'] if pick else None,box=b,GT_bbox=gb,visible_pixels=g['visible_pixels'][obj],bbox_IoU=ov,correct=bool(b and ov>=.5),null=pick is None,wrong_object=wrong,actor_pixels=counts))
csvwr('selected_boxes.csv',selected)
def rate(rs,fun):
 n=sum(bool(fun(r)) for r in rs);return dict(numerator=n,denominator=len(rs),rate=n/len(rs) if rs else None)
def dist(xs):
 return dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,median=float(np.median(xs)) if xs else None,p95=float(np.quantile(xs,.95)) if xs else None,min=min(xs) if xs else None,max=max(xs) if xs else None)
def selection_metrics(rs):
 result={}
 for obj in ['peg','box']:
  z=[r for r in rs if r['object']==obj];vis=[r for r in z if r['visible_pixels']>=16];emit=[r for r in z if not r['null']]
  result[obj]=dict(recall=rate(vis,lambda r:r['correct']),null_rate=rate(z,lambda r:r['null']),precision=rate(emit,lambda r:r['correct']),wrong_object=rate(emit,lambda r:r['wrong_object']),mean_IoU_including_null=dist([r['bbox_IoU'] for r in vis]))
 scenes=collections.defaultdict(list)
 for r in rs:scenes[r['image_id']].append(r)
 result['scene']=dict(both_correct=rate(list(scenes.values()),lambda rs:all(r['correct'] for r in rs)),any_wrong_object=rate(list(scenes.values()),lambda rs:any(r['wrong_object'] for r in rs)))
 return result
methodmetrics={}
for method in ['raw_top1','historical_verifier']+conditions:
 z=[r for r in selected if r['method']==method];perparent={p:selection_metrics([r for r in z if r['parent_id']==p]) for p in parents}
 methodmetrics[method]=dict(image_level=selection_metrics(z),per_parent=perparent,parent_macro={k:float(np.mean([m[k]['recall']['rate'] for m in perparent.values()])) for k in ['peg','box']},by_camera={cam:selection_metrics([r for r in z if r['camera']==cam]) for cam in ['base_camera','hand_camera']})
candmetrics={}
for condition in conditions:
 z=[r for r in rows if r['condition']==condition];clear=[r for r in z if r['GT_role']!='AMBIGUOUS']
 candmetrics[condition]=dict(labels=dict(collections.Counter(r['strict_label'] for r in z)),format_validity=rate(z,lambda r:r['format_valid']),clear_role_accuracy=rate(clear,lambda r:r['role_correct']),confusion={role:dict(collections.Counter(r['strict_label'] for r in z if r['GT_role']==role)) for role in ['PEG','BOX','ROBOT','AMBIGUOUS']},by_query={})
 for obj in ['peg','box']:
  q=[r for r in z if r['query_object']==obj];good=[r for r in q if r['target_good']];bad=[r for r in q if not r['target_good']]
  candmetrics[condition]['by_query'][obj]=dict(candidates=len(q),good_box_retention=rate(good,lambda r:r['accepted_for_query']),bad_box_approval=rate(bad,lambda r:r['accepted_for_query']),null_or_invalid=rate(q,lambda r:r['strict_label'] in ['UNCERTAIN','INVALID_OUTPUT']))
paired=[]
for c in candidates:
 a=pred[(c['id'],conditions[0])]['strict_label'];b=pred[(c['id'],conditions[1])]['strict_label']
 paired.append(dict(candidate_id=c['id'],image_id=c['image_id'],parent_id=c['parent_id'],camera=c['camera'],query_object=c['query_object'],current_only=a,with_initial_RGB=b,changed=a!=b,historical=old[c['id']]))
csvwr('paired_label_changes.csv',paired)
specific=[r for r in paired if r['candidate_id'] in ['10i_s1_0275_F1_hand_camera_native512_peg_rank000','10i_s1_0305_F1_hand_camera_native512_peg_rank000']];assert len(specific)==2
lookup={(r['method'],r['image_id'],r['object']):r for r in selected};effects={};rng=np.random.default_rng(42)
for obj in ['peg','box']:
 changes=[]
 for parent in parents:
  a=[r for r in selected if r['method']=='current_only' and r['parent_id']==parent and r['object']==obj];b=[lookup[('with_initial_RGB',r['image_id'],obj)] for r in a]
  changes.append(dict(parent_id=parent,images=len(a),recall_delta=float(np.mean([int(y['correct'])-int(x['correct']) for x,y in zip(a,b)])),IoU_delta=float(np.mean([y['bbox_IoU']-x['bbox_IoU'] for x,y in zip(a,b)]))))
 effects[obj]=dict(per_parent=changes)
 for key in ['recall_delta','IoU_delta']:
  values=np.array([r[key] for r in changes]);effects[obj][key]=dict(parent_macro_delta=float(values.mean()),exploratory_CI95=np.quantile(values[rng.integers(0,len(values),(10000,len(values)))].mean(1),[.025,.975]).tolist())
metric=dict(methods=methodmetrics,candidates=candmetrics,specific_regressions=specific,paired_label_transitions=dict(collections.Counter(r['current_only']+'->'+r['with_initial_RGB'] for r in paired)),changed_labels=sum(r['changed'] for r in paired),historical_control_agreement=rate(paired,lambda r:r['current_only']==r['historical']),effects=effects,GT_by_image=gtinfo,statistics=dict(unit='parent',parents=4,known_development_scenes=True,bootstrap=10000,seed=42,exploratory=True))
wr('temporal_reference_metrics.json',metric)
lat=rd(R/'verifier_latency_vram.json');lat['summary']={}
for condition in conditions:
 z=[r for r in lat['requests'] if r['condition']==condition and not r['cold_first_request']]
 lat['summary'][condition]=dict(warm_end_to_end_ms=dist([r['end_to_end_ms'] for r in z]),preprocessing_ms=dist([r['preprocessing_ms'] for r in z]),vision_ms=dist([r['vision_cuda_ms'] for r in z]),prefill_excluding_vision_approx_ms=dist([r['vision_prefill_cuda_ms']-r['vision_cuda_ms'] for r in z]),decode_forward_ms=dist([r['decode_forward_cuda_ms'] for r in z]),input_tokens=dist([r['input_tokens'] for r in z]),output_tokens=dist([r['output_tokens'] for r in z]),peak_allocated_bytes=max(r['memory']['peak_allocated'] for r in z),peak_reserved_bytes=max(r['memory']['peak_reserved'] for r in z))
lat['global_cold_first_request']=lat['requests'][0];wr('verifier_latency_vram.json',lat)
# Prespecified two regression parents; show their initial reference and current selections.
W,H=350,400;canvas=Image.new('RGB',(W*3,H*2),'white');draw=ImageDraw.Draw(canvas);anchors={x['current_image_id']:x for x in rd(R/'input_manifest.json')['camera_frames']}
for j,parent in enumerate(['10i_s1_0275','10i_s1_0305']):
 image=next(c for c in images.values() if c['parent_id']==parent and c['camera']=='hand_camera')
 for i,method in enumerate(['initial_reference']+conditions):
  path=anchors[image['id']]['initial_RGB_path'] if i==0 else image['image_path'];tile=Image.open(path).convert('RGB').resize((340,340));d=ImageDraw.Draw(tile)
  if i:
   for obj,color,gcolor in [('peg','orange','lime'),('box','cyan','magenta')]:
    row=lookup[(method,image['id'],obj)]
    for box,col,width in [(row['GT_bbox'],gcolor,1),(row['box'],color,3)]:
     if box:d.rectangle([x*340/512 for x in box],outline=col,width=width)
  x=i*W;y=j*H;canvas.paste(tile,(x,y+36));draw.text((x+3,y+3),parent+' Wrist '+method,fill='black')
  if i:draw.text((x+3,y+18),' '.join(obj+' IoU='+f'{lookup[(method,image["id"],obj)]["bbox_IoU"]:.3f}' for obj in ['peg','box']),fill='black')
canvas.save(R/'temporal_reference_comparison.png')
with ExperimentLogger(team='integration',model='Task10I_Qwen_temporal_reference_evaluation',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
 logger.log({method:{obj:m['image_level'][obj]['recall']['rate'] for obj in ['peg','box']} for method,m in methodmetrics.items()});logger.link_output(R/'temporal_reference_metrics.json')
v.update(stage='EVALUATION_COMPLETE',raw_sha256_after_GT=sha(R/'verifier_raw_responses.json'),predictions_sha256_after_GT=sha(R/'verifier_predictions.csv'));wr('verification.json',v)
print('TEMPORAL_EVALUATED',json.dumps(dict(methods={method:{obj:m['image_level'][obj]['recall'] for obj in ['peg','box']} for method,m in methodmetrics.items()},regressions=specific,changes=metric['changed_labels'])),flush=True)
