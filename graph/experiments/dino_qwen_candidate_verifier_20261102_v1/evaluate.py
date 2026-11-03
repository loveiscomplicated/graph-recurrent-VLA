from common import *
import collections
import numpy as np
from PIL import Image,ImageDraw
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='PREDICTIONS_FROZEN' and v['completed_requests']==251
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT'] and sha(R/'verifier_predictions.csv')==v['predictions_sha256_before_GT']
assert sha(R/'.execution/candidate_manifest.json')==v['candidate_manifest_sha256']
cm=rd(R/'.execution/candidate_manifest.json');candidates=cm['candidates'];images={c['id']:c for c in cm['images']};parents=cm['parents']
raw=rd(R/'verifier_raw_responses.json')['responses'];pred={r['id']:r for r in raw};assert len(pred)==len(candidates)==251
v.update(stage='POSTHOC_GT_EVALUATION',GT_opened=True);wr('verification.json',v)
gtm=rd(D/'.execution/evaluation_manifest.json');gtcases={c['id']:c for c in gtm['cases']};truth={};gtinfo={}
def bbox(mask):
 y,x=np.nonzero(mask);return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
def rate(rs,fn):
 n=sum(bool(fn(r)) for r in rs);return dict(numerator=n,denominator=len(rs),rate=n/len(rs) if rs else None)
def dist(xs):
 xs=[x for x in xs if x is not None];return dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,median=float(np.median(xs)) if xs else None,p95=float(np.quantile(xs,.95)) if xs else None,min=min(xs) if xs else None,max=max(xs) if xs else None)
for c in images.values():
 g=gtcases[c['id']];assert sha(g['GT_path'])==g['GT_sha256']
 with np.load(g['GT_path']) as z:seg=z['actor_segmentation'].copy()
 masks={k:np.isin(seg,g['object_ids'][k]) for k in ['peg','box','robot']};masks['background']=~(masks['peg']|masks['box']|masks['robot']);truth[c['id']]=masks;gtinfo[c['id']]=dict(visible_pixels={k:int(m.sum()) for k,m in masks.items()},bboxes={k:bbox(m) for k,m in masks.items()},GT_sha256=g['GT_sha256'])
rows=[];byimage=collections.defaultdict(list)
yy,xx=np.indices((512,512))
for cand in candidates:
 b=cand['box'];masks=truth[cand['image_id']];g=gtinfo[cand['image_id']];p=pred[cand['id']];lab=p['strict_label'];rect=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]);counts={k:int((rect&m).sum()) for k,m in masks.items()};fg=sum(counts[k] for k in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda k:counts[k]);role=dom.upper() if fg>=16 and counts[dom]>=16 and counts[dom]/fg>=.7 else 'OTHER' if fg<16 else 'AMBIGUOUS';target=cand['query_object'];ov={k:iou(b,g['bboxes'][k]) for k in ['peg','box']}
 row=dict(**cand,strict_label=lab,GT_role=role,role_correct=lab==role if role!='AMBIGUOUS' else None,format_valid=lab in LABELS,actor_pixels=counts,target_bbox_IoU=ov[target],peg_IoU=ov['peg'],box_IoU=ov['box'],target_good=ov[target]>=.5,accepted_for_query=lab==target.upper(),rejected_good=ov[target]>=.5 and lab!=target.upper(),accepted_bad=ov[target]<.5 and lab==target.upper())
 rows.append(row);byimage[cand['image_id']].append(row)
csvwr('candidate_evaluation.csv',rows)
frames={f['id']:f for f in rd(D/'dino_frame_selections.json')['frames'] if f['native_size']==512}
methods=['legacy','raw_top1','legacy_gate','verifier_query_gate','verifier_union_roles'];selected=[]
for c in images.values():
 pool=byimage[c['id']]
 for k in ['peg','box']:
  query=sorted([r for r in pool if r['query_object']==k],key=lambda r:(-r['DINO_score'],r['rank']))
  leg=frames[c['id']]['selected'][k];legacy=next((r for r in query if leg and r['box']==leg['box']),None);assert (legacy is None)==(leg is None)
  accepted=[r for r in query if r['strict_label']==k.upper()];union=sorted([r for r in pool if r['strict_label']==k.upper()],key=lambda r:(-r['DINO_score'],r['query_object'],r['rank']))
  picks=dict(legacy=legacy,raw_top1=query[0] if query else None,legacy_gate=legacy if legacy and legacy['strict_label']==k.upper() else None,verifier_query_gate=accepted[0] if accepted else None,verifier_union_roles=union[0] if union else None)
  for method,p in picks.items():
   b=p['box'] if p else None;g=gtinfo[c['id']];n=g['visible_pixels'][k];gb=g['bboxes'][k];vis='VISIBLE' if n>=16 else 'TINY_VISIBLE' if n else 'ABSENT';ov=iou(b,gb)
   if b:
    rect=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]);counts={a:int((rect&m).sum()) for a,m in truth[c['id']].items()}
   else:counts={a:0 for a in ['peg','box','robot','background']}
   fg=sum(counts[a] for a in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda a:counts[a]);wrong=bool(b and ov<.5 and dom!=k and counts[dom]>=16 and fg and counts[dom]/fg>=.7)
   selected.append(dict(method=method,image_id=c['id'],pair_id=c['pair_id'],parent_id=c['parent_id'],camera=c['camera'],frame_kind=c['frame_kind'],object=k,candidate_id=p['id'] if p else None,predicted_label=p['strict_label'] if p else None,box=b,DINO_score=p['DINO_score'] if p else None,GT_bbox=gb,visible_pixels=n,visibility=vis,bbox_IoU=ov if n else None,correct=bool(b and ov>=.5),null=p is None,wrong_object=wrong,semantic_confusion=f'{k}->{dom}' if wrong else None,actor_pixels=counts))
csvwr('hybrid_selected_boxes.csv',selected);csvwr('hybrid_failure_cases.csv',[r for r in selected if r['visibility']=='VISIBLE' and not r['correct']])
def metric(rs):
 out={}
 for k in ['peg','box']:
  allk=[r for r in rs if r['object']==k];vis=[r for r in allk if r['visibility']=='VISIBLE'];em=[r for r in allk if not r['null'] and r['visibility']!='TINY_VISIBLE'];absent=[r for r in allk if r['visibility']=='ABSENT']
  out[k]=dict(recall=rate(vis,lambda r:r['correct']),IoU_including_null=dist([r['bbox_IoU'] for r in vis]),IoU_emitted=dist([r['bbox_IoU'] for r in em]),IoU_thresholds={str(t):rate(vis,lambda r:not r['null'] and r['bbox_IoU']>=t) for t in [.3,.5,.7]},null_rate=rate(allk,lambda r:r['null']),selected_precision=rate(em,lambda r:r['correct']),wrong_object=rate(em,lambda r:r['wrong_object']),hallucination_absent=rate(absent,lambda r:not r['null']),semantic_confusion=dict(collections.Counter(r['semantic_confusion'] for r in allk if r['semantic_confusion'])))
 scenes=collections.defaultdict(list)
 for r in rs:scenes[r['image_id']].append(r)
 both=[z for z in scenes.values() if len(z)==2 and all(r['visibility']=='VISIBLE' for r in z)]
 out['scene']=dict(both_correct=rate(both,lambda z:all(r['correct'] for r in z)),any_wrong_object=rate(list(scenes.values()),lambda z:any(r['wrong_object'] for r in z)))
 return out
metrics={}
for method in methods:
 rs=[r for r in selected if r['method']==method];pp={p:metric([r for r in rs if r['parent_id']==p]) for p in parents}
 metrics[method]=dict(image_object_level=metric(rs),parent_macro={k:float(np.mean([m[k]['recall']['rate'] for m in pp.values() if m[k]['recall']['rate'] is not None])) for k in ['peg','box']},per_parent=pp,by_camera={cam:metric([r for r in rs if r['camera']==cam]) for cam in ['base_camera','hand_camera']},by_time={f:metric([r for r in rs if r['frame_kind']==f]) for f in ['F0','F1']})
clear=[r for r in rows if r['GT_role']!='AMBIGUOUS'];candmetrics=dict(candidates=len(rows),strict_format_valid=rate(rows,lambda r:r['format_valid']),labels=dict(collections.Counter(r['strict_label'] for r in rows)),GT_roles=dict(collections.Counter(r['GT_role'] for r in rows)),clear_role_accuracy=rate(clear,lambda r:r['role_correct']),confusion={k:dict(collections.Counter(r['strict_label'] for r in rows if r['GT_role']==k)) for k in ['PEG','BOX','ROBOT','OTHER','AMBIGUOUS']},by_query={})
for k in ['peg','box']:
 rs=[r for r in rows if r['query_object']==k];good=[r for r in rs if r['target_good']];bad=[r for r in rs if not r['target_good']];acc=[r for r in rs if r['accepted_for_query']]
 candmetrics['by_query'][k]=dict(proposals=len(rs),good_boxes=len(good),good_box_retention=rate(good,lambda r:r['accepted_for_query']),bad_box_acceptance=rate(bad,lambda r:r['accepted_for_query']),accepted_box_precision=rate(acc,lambda r:r['target_good']),UNCERTAIN=rate(rs,lambda r:r['strict_label']=='UNCERTAIN'))
candmetrics['parent_macro_clear_role_accuracy']=float(np.mean([np.mean([r['role_correct'] for r in clear if r['parent_id']==p]) for p in parents if any(r['parent_id']==p for r in clear)]))
lookup={(r['method'],r['image_id'],r['object']):r for r in selected};rng=np.random.default_rng(42);effects={};trans=[]
for baseline in ['legacy','raw_top1']:
 effects[baseline]={}
 for k in ['peg','box']:
  changes=[]
  for p in parents:
   aa=[r for r in selected if r['method']==baseline and r['parent_id']==p and r['object']==k and r['visibility']=='VISIBLE'];bb=[lookup[('verifier_query_gate',r['image_id'],k)] for r in aa];changes.append(dict(parent_id=p,n=len(aa),recall_delta=float(np.mean([b['correct']-a['correct'] for a,b in zip(aa,bb)])),IoU_delta=float(np.mean([b['bbox_IoU']-a['bbox_IoU'] for a,b in zip(aa,bb)]))))
   for a,b in zip(aa,bb):trans.append(dict(baseline=baseline,image_id=a['image_id'],parent_id=p,object=k,old_correct=a['correct'],new_correct=b['correct'],recovered=not a['correct'] and b['correct'],regressed=a['correct'] and not b['correct'],old_IoU=a['bbox_IoU'],new_IoU=b['bbox_IoU']))
  effects[baseline][k]=dict(per_parent=changes)
  for name in ['recall_delta','IoU_delta']:
   a=np.array([x[name] for x in changes]);effects[baseline][k][name]=dict(parent_macro_delta=float(a.mean()),exploratory_CI95=np.quantile(a[rng.integers(0,len(a),(10000,len(a)))].mean(1),[.025,.975]).tolist())
csvwr('hybrid_transitions.csv',trans)
wr('hybrid_evaluation_metrics.json',dict(primary='verifier_query_gate vs raw_top1 on identical raw candidate pools',methods=metrics,candidate_verifier=candmetrics,paired_effects=effects,GT_by_image=gtinfo,bootstrap=dict(unit='physical parent',clusters=4,replicates=10000,seed=42,exploratory=True,no_multiplicity_correction=True),interpretation='Candidate role accuracy and localization quality differ. A correct role on a loose box does not imply IoU>=.5. Oracle pool coverage is not predicted verifier recall.'))
lat=rd(R/'verifier_latency_vram.json');warm=lat['requests'][1:];lat['summary']=dict(cold=lat['requests'][0],warm_requests=len(warm),warm_end_to_end_ms=dist([r['end_to_end_ms'] for r in warm]),preprocessing_ms=dist([r['preprocessing_ms'] for r in warm]),vision_ms=dist([r['vision_cuda_ms'] for r in warm]),prefill_excluding_vision_approx_ms=dist([r['vision_prefill_cuda_ms']-r['vision_cuda_ms'] for r in warm]),decode_forward_ms=dist([r['decode_forward_cuda_ms'] for r in warm]),output_tokens=dist([r['output_tokens'] for r in lat['requests']]),peak_allocated_bytes=max(r['memory']['peak_allocated'] for r in lat['requests']),peak_reserved_bytes=max(r['memory']['peak_reserved'] for r in lat['requests']))
cmap={c['id']:c for c in candidates};sums=[]
for image in images.values():
 rr=[r for r in lat['requests'] if cmap[r['id']]['image_id']==image['id']];sums.append(dict(image_id=image['id'],candidates=len(rr),serial_request_sum_ms=sum(r['end_to_end_ms'] for r in rr),includes_cold=any(r['cold_first_request'] for r in rr)))
lat['per_image_serial_sums']=sums;lat['summary']['serial_sum_per_image_ms']=dist([x['serial_request_sum_ms'] for x in sums]);wr('verifier_latency_vram.json',lat)
# Fixed first parent, all four camera/time combinations, legacy vs primary hybrid.
pid=parents[0];cs=sorted([c for c in images.values() if c['parent_id']==pid],key=lambda c:(c['frame_kind'],c['camera']));W,H=350,405;canvas=Image.new('RGB',(W*2,H*4),'white');draw=ImageDraw.Draw(canvas)
for j,c in enumerate(cs):
 for i,method in enumerate(['legacy','verifier_query_gate']):
  tile=Image.open(c['image_path']).convert('RGB').resize((340,340));d=ImageDraw.Draw(tile)
  for k,col,gcol in [('peg','orange','lime'),('box','cyan','magenta')]:
   r=lookup[(method,c['id'],k)]
   for b,color,width in [(r['GT_bbox'],gcol,1),(r['box'],col,3)]:
    if b:d.rectangle([x*340/512 for x in b],outline=color,width=width)
  x=i*W;y=j*H;canvas.paste(tile,(x,y+34));draw.text((x+3,y+3),pid+' '+c['frame_kind']+' '+c['camera'],fill='black');draw.text((x+3,y+18),method,fill='black');draw.text((x+3,y+378),' '.join(k+'='+f'{lookup[(method,c["id"],k)]["bbox_IoU"]:.2f}' for k in ['peg','box']),fill='black')
canvas.save(R/'hybrid_verifier_comparison.png')
with ExperimentLogger(team='integration',model='Task10I_DINO_Qwen_verifier_evaluation',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
 logger.log({method:{k:m['image_object_level'][k]['recall']['rate'] for k in ['peg','box']} for method,m in metrics.items()});logger.link_output(R/'hybrid_evaluation_metrics.json')
v.update(stage='EVALUATION_COMPLETE',raw_sha256_after_GT=sha(R/'verifier_raw_responses.json'),predictions_sha256_after_GT=sha(R/'verifier_predictions.csv'));wr('verification.json',v)
print('HYBRID_EVALUATED',json.dumps(dict(format=candmetrics['strict_format_valid'],methods={method:{k:m['image_object_level'][k]['recall'] for k in ['peg','box']} for method,m in metrics.items()})),flush=True)
