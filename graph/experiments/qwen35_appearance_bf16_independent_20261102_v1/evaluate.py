from common import *
import collections
import numpy as np
from PIL import Image,ImageDraw
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='PREDICTIONS_FROZEN' and v['completed_requests']==v['request_count']
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT'] and sha(R/'verifier_predictions.csv')==v['predictions_sha256_before_GT']
assert sha(R/'.execution/candidate_manifest.json')==v['candidate_manifest_sha256']
cm=rd(R/'.execution/candidate_manifest.json');images={c['id']:c for c in cm['images']};parents=cm['parents'];candidates=cm['candidates'];conditions=['original_three','appearance_three']
raw=rd(R/'verifier_raw_responses.json')['responses'];pred={(r['candidate_id'],r['condition']):r for r in raw};assert len(pred)==2*len(candidates)
truth={};gtinfo={};yy,xx=np.indices((512,512))
def bbox(mask):
 y,x=np.where(mask);return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
def rate(rs,fn):
 n=sum(bool(fn(r)) for r in rs);return dict(numerator=n,denominator=len(rs),rate=n/len(rs) if rs else None)
def dist(xs):
 xs=[float(x) for x in xs if x is not None];return dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,median=float(np.median(xs)) if xs else None,p95=float(np.quantile(xs,.95)) if xs else None)
def actor_counts(box,masks):
 if box is None:return {k:0 for k in masks}
 rect=(xx>=box[0])&(xx<box[2])&(yy>=box[1])&(yy<box[3]);return {k:int((rect&m).sum()) for k,m in masks.items()}
def role(counts):
 fg=sum(counts[k] for k in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda k:counts[k]);return dom.upper() if fg>=16 and counts[dom]>=16 and counts[dom]/fg>=.7 else 'OTHER' if fg<16 else 'AMBIGUOUS'
for c in images.values():
 assert sha(c['GT_path'])==c['GT_sha256']
 with np.load(c['GT_path']) as z:seg=z['actor_segmentation'].copy()
 masks={k:np.isin(seg,c['object_ids'][k]) for k in ['peg','box','robot']};masks['background']=~(masks['peg']|masks['box']|masks['robot']);truth[c['id']]=masks
 gtinfo[c['id']]=dict(visible_pixels={k:int(m.sum()) for k,m in masks.items()},bboxes={k:bbox(m) for k,m in masks.items()},GT_sha256=c['GT_sha256'])
rows=[];pool=collections.defaultdict(list)
for c in candidates:
 g=gtinfo[c['image_id']];counts=actor_counts(c['box'],truth[c['image_id']]);gtrole=role(counts);ovs={k:iou(c['box'],g['bboxes'][k]) for k in ['peg','box']}
 pool[c['image_id']].append(c)
 for cond in conditions:
  lab=pred[(c['id'],cond)]['strict_label'];good=ovs[c['query_object']]>=.5;accepted=lab==c['query_object'].upper()
  rows.append(dict(**c,condition=cond,strict_label=lab,format_valid=lab in LABELS,GT_role=gtrole,role_correct=lab==gtrole if gtrole!='AMBIGUOUS' else None,actor_pixels=counts,peg_IoU=ovs['peg'],box_IoU=ovs['box'],target_good=good,accepted_for_query=accepted,rejected_good=good and not accepted,accepted_bad=not good and accepted))
csvwr('candidate_evaluation.csv',rows)
frames={f['id']:f for f in rd(R/'dino_frame_selections.json')['frames']};methods=['legacy','raw_top1','original_query_gate','appearance_query_gate'];selected=[]
for image in images.values():
 for obj in ['peg','box']:
  query=sorted([c for c in pool[image['id']] if c['query_object']==obj],key=lambda c:(-c['DINO_score'],c['rank']))
  leg=frames[image['id']]['selected'][obj];legacy=next((c for c in query if leg is not None and c['box']==leg['box']),None);assert (legacy is None)==(leg is None)
  picks=dict(legacy=legacy,raw_top1=query[0] if query else None)
  for cond,method in zip(conditions,methods[2:]):picks[method]=next((c for c in query if pred[(c['id'],cond)]['strict_label']==obj.upper()),None)
  g=gtinfo[image['id']];n=g['visible_pixels'][obj];gb=g['bboxes'][obj];visibility='VISIBLE' if n>=16 else 'TINY_VISIBLE' if n else 'ABSENT'
  for method,pick in picks.items():
   b=pick['box'] if pick else None;ov=iou(b,gb);counts=actor_counts(b,truth[image['id']]);gtrole=role(counts);wrong=bool(b and ov<.5 and gtrole in ['PEG','BOX','ROBOT'] and gtrole!=obj.upper())
   center=float(np.linalg.norm((np.array(b[:2])+b[2:])/2-(np.array(gb[:2])+gb[2:])/2)) if b and gb else None
   selected.append(dict(method=method,image_id=image['id'],pair_id=image['pair_id'],parent_id=image['parent_id'],camera=image['camera'],frame_kind=image['frame_kind'],timestep=image['timestep'],object=obj,candidate_id=pick['id'] if pick else None,box=b,DINO_score=pick['DINO_score'] if pick else None,GT_bbox=gb,visible_pixels=n,visibility=visibility,bbox_IoU=ov if n else None,correct=bool(b and ov>=.5),null=pick is None,wrong_object=wrong,semantic_confusion=f'{obj}->{gtrole.lower()}' if wrong else None,GT_role=gtrole,center_error_px=center,actor_pixels=counts))
csvwr('selected_boxes.csv',selected);csvwr('failure_cases.csv',[r for r in selected if r['visibility']=='VISIBLE' and not r['correct']])
def metric(rs):
 out={}
 for obj in ['peg','box']:
  aa=[r for r in rs if r['object']==obj];vis=[r for r in aa if r['visibility']=='VISIBLE'];em=[r for r in aa if not r['null'] and r['visibility']!='TINY_VISIBLE'];absent=[r for r in aa if r['visibility']=='ABSENT']
  out[obj]=dict(recall=rate(vis,lambda r:r['correct']),IoU_including_null=dist([r['bbox_IoU'] for r in vis]),IoU_thresholds={str(t):rate(vis,lambda r:not r['null'] and r['bbox_IoU']>=t) for t in [.3,.5,.7]},selected_precision=rate(em,lambda r:r['correct']),null_rate=rate(aa,lambda r:r['null']),missing_visible=rate(vis,lambda r:r['null']),wrong_object=rate(em,lambda r:r['wrong_object']),hallucination_absent=rate(absent,lambda r:not r['null']),center_error_detected=dist([r['center_error_px'] for r in vis if not r['null']]),visibility_counts=dict(collections.Counter(r['visibility'] for r in aa)),semantic_confusion=dict(collections.Counter(r['semantic_confusion'] for r in aa if r['semantic_confusion'])))
 scenes=collections.defaultdict(list)
 for r in rs:scenes[r['image_id']].append(r)
 both=[z for z in scenes.values() if len(z)==2 and all(r['visibility']=='VISIBLE' for r in z)]
 out['scene']=dict(both_correct=rate(both,lambda z:all(r['correct'] for r in z)),any_wrong_object=rate(list(scenes.values()),lambda z:any(r['wrong_object'] for r in z)))
 return out
metrics={}
for method in methods:
 rs=[r for r in selected if r['method']==method];pp={p:metric([r for r in rs if r['parent_id']==p]) for p in parents}
 metrics[method]=dict(image_object_level=metric(rs),per_parent=pp,parent_macro={obj:float(np.mean([m[obj]['recall']['rate'] for m in pp.values() if m[obj]['recall']['rate'] is not None])) for obj in ['peg','box']},by_camera={c:metric([r for r in rs if r['camera']==c]) for c in ['base_camera','hand_camera']},by_time={t:metric([r for r in rs if r['frame_kind']==t]) for t in ['F0','F1']})
cmetrics={}
for cond in conditions:
 rs=[r for r in rows if r['condition']==cond];clear=[r for r in rs if r['GT_role']!='AMBIGUOUS'];byquery={}
 for obj in ['peg','box']:
  qs=[r for r in rs if r['query_object']==obj];good=[r for r in qs if r['target_good']];bad=[r for r in qs if not r['target_good']];accepted=[r for r in qs if r['accepted_for_query']]
  byquery[obj]=dict(proposals=len(qs),good_boxes=len(good),good_box_retention=rate(good,lambda r:r['accepted_for_query']),bad_box_acceptance=rate(bad,lambda r:r['accepted_for_query']),accepted_box_precision=rate(accepted,lambda r:r['target_good']))
 cmetrics[cond]=dict(candidates=len(rs),format_valid=rate(rs,lambda r:r['format_valid']),clear_role_accuracy=rate(clear,lambda r:r['role_correct']),clear_role_parent_macro=float(np.mean([np.mean([r['role_correct'] for r in clear if r['parent_id']==p]) for p in parents if any(r['parent_id']==p for r in clear)])),labels=dict(collections.Counter(r['strict_label'] for r in rs)),GT_roles=dict(collections.Counter(r['GT_role'] for r in rs)),confusion={role:dict(collections.Counter(r['strict_label'] for r in rs if r['GT_role']==role)) for role in ['PEG','BOX','ROBOT','OTHER','AMBIGUOUS']},by_query=byquery)
lookup={(r['method'],r['image_id'],r['object']):r for r in selected};rng=np.random.default_rng(42);effects={};trans=[]
for baseline in methods[:-1]:
 effects[baseline]={}
 for obj in ['peg','box']:
  changes=[]
  for pid in parents:
   aa=[r for r in selected if r['method']==baseline and r['parent_id']==pid and r['object']==obj and r['visibility']=='VISIBLE'];bb=[lookup[('appearance_query_gate',a['image_id'],obj)] for a in aa]
   if aa:changes.append(dict(parent_id=pid,n=len(aa),recall_delta=float(np.mean([b['correct']-a['correct'] for a,b in zip(aa,bb)])),IoU_delta=float(np.mean([b['bbox_IoU']-a['bbox_IoU'] for a,b in zip(aa,bb)]))))
   for a,b in zip(aa,bb):trans.append(dict(baseline=baseline,image_id=a['image_id'],parent_id=pid,object=obj,old_correct=a['correct'],new_correct=b['correct'],recovered=not a['correct'] and b['correct'],regressed=a['correct'] and not b['correct'],old_IoU=a['bbox_IoU'],new_IoU=b['bbox_IoU']))
  effects[baseline][obj]=dict(per_parent=changes)
  for name in ['recall_delta','IoU_delta']:
   a=np.array([x[name] for x in changes]);effects[baseline][obj][name]=dict(parent_macro_delta=float(a.mean()),CI95=np.quantile(a[rng.integers(0,len(a),(10000,len(a)))].mean(1),[.025,.975]).tolist())
csvwr('paired_transitions.csv',trans)
oracle={obj:rate([c for c in images.values() if gtinfo[c['id']]['visible_pixels'][obj]>=16],lambda c:any(iou(p['box'],gtinfo[c['id']]['bboxes'][obj])>=.5 for p in pool[c['id']] if p['query_object']==obj)) for obj in ['peg','box']}
wr('evaluation_metrics.json',dict(status='EVALUATION_COMPLETE',primary='appearance_query_gate vs original_query_gate and raw_top1 on identical proposals',parents=len(parents),images=len(images),candidates=len(candidates),requests=len(raw),methods=metrics,candidate_verifier=cmetrics,paired_effects=effects,oracle_pool_recall=oracle,GT_by_image=gtinfo,bootstrap=dict(unit='physical parent',clusters=len(parents),replicates=10000,seed=42,exploratory=True,multiplicity_adjusted=False)))
lat=rd(R/'verifier_latency_vram.json');lat['summary']={}
cmap={c['id']:c for c in candidates}
for cond in conditions:
 rr=[r for r in raw if r['condition']==cond];warm=[r['timing'] for r in rr if not r['timing']['cold_first_request']];sums=[sum(r['timing']['end_to_end_ms'] for r in rr if cmap[r['candidate_id']]['image_id']==im) for im in images]
 lat['summary'][cond]=dict(warm_end_to_end_ms=dist([t['end_to_end_ms'] for t in warm]),preprocessing_ms=dist([t['preprocessing_ms'] for t in warm]),vision_ms=dist([t['vision_cuda_ms'] for t in warm]),prefill_excluding_vision_approx_ms=dist([t['vision_prefill_cuda_ms']-t['vision_cuda_ms'] for t in warm]),decode_ms=dist([t['decode_forward_cuda_ms'] for t in warm]),output_tokens=dist([t['output_tokens'] for t in warm]),peak_allocated_GiB=max(t['memory']['peak_allocated'] for t in warm)/2**30,per_image_serial_sum_ms=dist(sums))
wr('verifier_latency_vram.json',lat)
# Prespecified first eligible parent, four views; no post-result gallery selection.
cs=sorted([c for c in images.values() if c['parent_id']==parents[0]],key=lambda c:(c['frame_kind'],c['camera']));canvas=Image.new('RGB',(1020,1500),'white');draw=ImageDraw.Draw(canvas)
for j,c in enumerate(cs):
 for i,method in enumerate(methods[1:]):
  tile=Image.open(c['image_path']).convert('RGB').resize((330,330));d=ImageDraw.Draw(tile)
  for obj,color,gcolor in [('peg','orange','lime'),('box','cyan','magenta')]:
   r=lookup[(method,c['id'],obj)]
   for b,col,width in [(r['GT_bbox'],gcolor,1),(r['box'],color,3)]:
    if b:d.rectangle([x*330/512 for x in b],outline=col,width=width)
  x=i*340;y=j*375;canvas.paste(tile,(x,y+40));draw.text((x+3,y+2),c['parent_id']+' '+c['frame_kind']+' '+c['camera'],fill='black');draw.text((x+3,y+19),method,fill='black')
canvas.save(R/'comparison.png')
with ExperimentLogger(team='integration',model='Task10I_appearance_heldout_analysis',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
 logger.log({method:{obj:z['image_object_level'][obj]['recall']['rate'] for obj in ['peg','box']} for method,z in metrics.items()});logger.link_output(R/'evaluation_metrics.json')
v.update(stage='EVALUATION_COMPLETE',raw_sha256_after_GT=sha(R/'verifier_raw_responses.json'),predictions_sha256_after_GT=sha(R/'verifier_predictions.csv'));wr('verification.json',v)
print('EVALUATED',json.dumps({method:{obj:z['image_object_level'][obj]['recall'] for obj in ['peg','box']} for method,z in metrics.items()}),flush=True)
