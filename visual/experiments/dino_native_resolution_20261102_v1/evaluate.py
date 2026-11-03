from common import *
import collections
import numpy as np
from PIL import Image,ImageDraw
from visual.runtime import initialization as init
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='PREDICTIONS_FROZEN' and v['completed_forwards']==64
assert sha(R/'dino_raw_responses.json')==v['raw_sha256_before_GT']
assert sha(R/'dino_frame_selections.json')==v['selections_sha256_before_GT']
assert sha(R/'dino_predictions.csv')==v['predictions_sha256_before_GT']
assert sha(R/'.execution/evaluation_manifest.json')==v['evaluation_manifest_sha256']
mf=rd(R/'.execution/evaluation_manifest.json');cases={c['id']:c for c in mf['cases']};parents=[p['parent_id'] for p in mf['parents']]
raw=rd(R/'dino_raw_responses.json')['responses'];lookup={(r['id'],r['object']):r for r in raw};frames=rd(R/'dino_frame_selections.json')['frames']
v.update(stage='POSTHOC_GT_EVALUATION',GT_opened=True);wr('verification.json',v)
def bbox(mask):
 y,x=np.nonzero(mask);return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
def rate(rs,fn):
 n=sum(bool(fn(r)) for r in rs);return dict(numerator=n,denominator=len(rs),rate=n/len(rs) if rs else None)
def dist(xs):
 xs=[x for x in xs if x is not None];return dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,median=float(np.median(xs)) if xs else None,p95=float(np.quantile(xs,.95)) if xs else None,min=min(xs) if xs else None,max=max(xs) if xs else None)
truth={};gtinfo={};rows=[];proposalrows=[]
for frame in frames:
 c=cases[frame['id']];assert sha(c['GT_path'])==c['GT_sha256']
 with np.load(c['GT_path'],allow_pickle=False) as z:seg=z['actor_segmentation'].copy()
 masks={k:np.isin(seg,c['object_ids'][k]) for k in ['peg','box','robot']};masks['background']=~(masks['peg']|masks['box']|masks['robot']);truth[c['id']]=masks
 gtinfo[c['id']]=dict(visible_pixels={k:int(m.sum()) for k,m in masks.items()},bboxes={k:bbox(m) for k,m in masks.items()},GT_sha256=c['GT_sha256'])
 yy,xx=np.indices(seg.shape)
 for k in ['peg','box']:
  r=lookup[(c['id'],k)];boxes=r['boxes'];scores=r['scores'];n=int(masks[k].sum());gb=bbox(masks[k]);visibility='VISIBLE' if n>=16 else 'TINY_VISIBLE' if n else 'ABSENT';ious=[iou(b,gb) for b in boxes]
  s=frame['selected'][k];b=s['box'] if s else None;ov=iou(b,gb);cand=frame['candidates'][k]
  rect=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]) if b else np.zeros(seg.shape,bool);counts={a:int((rect&m).sum()) for a,m in masks.items()};fg=sum(counts[a] for a in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda a:counts[a]);wrong=bool(b and ov<.5 and dom!=k and counts[dom]>=16 and fg and counts[dom]/fg>=.7)
  good=[j for j,x in enumerate(ious) if x>=.5];area_good=[j for j in good if (boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1])<init.MAX_BOX_FRAC*c['width']*c['height']];filtered_good=[x for x in cand if iou(x[0],gb)>=.5]
  if visibility!='VISIBLE':drop=visibility
  elif b and ov>=.5:drop='SUCCESS'
  elif not good:drop='NO_GOOD_PROPOSAL_AT_THRESHOLD'
  elif not area_good:drop='AREA'
  elif not filtered_good:drop='TOP5'
  elif max(x[1] for x in filtered_good)<=init.MISSING_SCORE:drop='NULL_SCORE'
  else:drop='JOINT_OR_RANKING'
  center=float(math.hypot((b[0]+b[2]-gb[0]-gb[2])/2,(b[1]+b[3]-gb[1]-gb[3])/2)) if b and gb else None
  rows.append(dict(id=c['id'],pair_id=c['pair_id'],parent_id=c['parent_id'],native_size=c['native_size'],camera=c['camera'],frame_kind=c['frame_kind'],timestep=c['timestep'],object=k,visibility=visibility,visible_pixels=n,GT_bbox=gb,raw_proposals=len(boxes),any_proposal_correct=bool(good),raw_top1_box=boxes[0] if boxes else None,raw_top1_score=scores[0] if scores else None,raw_top1_IoU=ious[0] if ious else 0.,raw_top1_correct=bool(ious and ious[0]>=.5),best_proposal_IoU=max(ious,default=0.),filtered_candidates=len(cand),selected_box=b,selected_score=s['score'] if s else None,selected_IoU=ov if n else None,selected_correct=bool(b and ov>=.5),null=s is None,center_error_px=center,center_error_128px_units=center*128/c['native_size'] if center is not None else None,actor_pixels=counts,wrong_object=wrong,semantic_confusion=f'{k}->{dom}' if wrong else None,drop=drop))
  for j,(pb,ps) in enumerate(zip(boxes,scores)):
   proposalrows.append(dict(id=c['id'],pair_id=c['pair_id'],parent_id=c['parent_id'],native_size=c['native_size'],camera=c['camera'],frame_kind=c['frame_kind'],object=k,rank=j+1,box=pb,score=ps,GT_bbox=gb,GT_visible_pixels=n,IoU=ious[j] if n else None,in_image=all(0<=x<=c['native_size'] for x in pb),finite_order_valid=True,passes_area=(pb[2]-pb[0])*(pb[3]-pb[1])<init.MAX_BOX_FRAC*c['width']*c['height']))
csvwr('dino_object_evaluation.csv',rows);csvwr('dino_proposal_evaluation.csv',proposalrows)
csvwr('dino_failure_cases.csv',[r for r in rows if r['visibility']=='VISIBLE' and not r['selected_correct']])
def metrics(rs):
 out=dict(object_rows=len(rs),images=len({r['id'] for r in rs}),objects={})
 for k in ['peg','box']:
  allk=[r for r in rs if r['object']==k];vis=[r for r in allk if r['visibility']=='VISIBLE'];emitted=[r for r in allk if not r['null'] and r['visibility']!='TINY_VISIBLE'];absent=[r for r in allk if r['visibility']=='ABSENT']
  out['objects'][k]=dict(selected_recall=rate(vis,lambda r:r['selected_correct']),raw_top1_recall=rate(vis,lambda r:r['raw_top1_correct']),any_proposal_coverage=rate(vis,lambda r:r['any_proposal_correct']),selected_IoU_including_null=dist([r['selected_IoU'] for r in vis]),selected_IoU_emitted_only=dist([r['selected_IoU'] for r in vis if not r['null']]),raw_top1_IoU=dist([r['raw_top1_IoU'] for r in vis]),oracle_best_IoU=dist([r['best_proposal_IoU'] for r in vis]),IoU_thresholds={str(t):rate(vis,lambda r:not r['null'] and r['selected_IoU']>=t) for t in [.3,.5,.7]},null_rate=rate(allk,lambda r:r['null']),missing_detection=rate(vis,lambda r:not r['selected_correct']),hallucinated_selected_box_absent=rate(absent,lambda r:not r['null']),false_discovery=rate(emitted,lambda r:not r['selected_correct']),wrong_object_fraction=rate(emitted,lambda r:r['wrong_object']),center_error_px=dist([r['center_error_px'] for r in vis]),center_error_128px_units=dist([r['center_error_128px_units'] for r in vis]),visibility=dict(collections.Counter(r['visibility'] for r in allk)),drop_reasons=dict(collections.Counter(r['drop'] for r in vis)),semantic_confusion=dict(collections.Counter(r['semantic_confusion'] for r in allk if r['semantic_confusion'])))
 scenes=collections.defaultdict(list)
 for r in rs:scenes[r['id']].append(r)
 both=[rs for rs in scenes.values() if len(rs)==2 and all(r['visibility']=='VISIBLE' for r in rs)]
 out['scene']=dict(both_selected_correct=rate(both,lambda rs:all(r['selected_correct'] for r in rs)),any_wrong_object=rate(list(scenes.values()),lambda rs:any(r['wrong_object'] for r in rs)))
 return out
summaries={}
for size in [128,512]:
 rs=[r for r in rows if r['native_size']==size];pp={p:metrics([r for r in rs if r['parent_id']==p]) for p in parents};summaries[str(size)]=dict(image_object_level=metrics(rs),per_parent=pp,parent_macro={k:float(np.mean([z['objects'][k]['selected_recall']['rate'] for z in pp.values() if z['objects'][k]['selected_recall']['rate'] is not None])) for k in ['peg','box']},by_camera={cam:metrics([r for r in rs if r['camera']==cam]) for cam in ['base_camera','hand_camera']},by_time={f:metrics([r for r in rs if r['frame_kind']==f]) for f in ['F0','F1']})
lookuprow={(r['native_size'],r['pair_id'],r['object']):r for r in rows};pairs=[];rng=np.random.default_rng(42);effects={}
for k in ['peg','box']:
 pp=[]
 for p in parents:
  aa=[r for r in rows if r['native_size']==128 and r['parent_id']==p and r['object']==k and r['visibility']=='VISIBLE' and lookuprow[(512,r['pair_id'],k)]['visibility']=='VISIBLE'];bb=[lookuprow[(512,r['pair_id'],k)] for r in aa]
  pp.append(dict(parent_id=p,n=len(aa),selected_recall_delta=float(np.mean([b['selected_correct']-a['selected_correct'] for a,b in zip(aa,bb)])) if aa else None,raw_top1_recall_delta=float(np.mean([b['raw_top1_correct']-a['raw_top1_correct'] for a,b in zip(aa,bb)])) if aa else None,proposal_coverage_delta=float(np.mean([b['any_proposal_correct']-a['any_proposal_correct'] for a,b in zip(aa,bb)])) if aa else None,selected_mean_IoU_delta=float(np.mean([b['selected_IoU']-a['selected_IoU'] for a,b in zip(aa,bb)])) if aa else None))
  for a,b in zip(aa,bb):pairs.append(dict(pair_id=a['pair_id'],parent_id=p,camera=a['camera'],frame_kind=a['frame_kind'],object=k,old_selected=a['selected_correct'],new_selected=b['selected_correct'],recovered=not a['selected_correct'] and b['selected_correct'],regressed=a['selected_correct'] and not b['selected_correct'],old_IoU=a['selected_IoU'],new_IoU=b['selected_IoU'],old_drop=a['drop'],new_drop=b['drop']))
 effects[k]=dict(per_parent=pp)
 for metric in ['selected_recall_delta','raw_top1_recall_delta','proposal_coverage_delta','selected_mean_IoU_delta']:
  a=np.array([z[metric] for z in pp if z[metric] is not None]);effects[k][metric]=dict(parent_macro_delta=float(a.mean()) if len(a) else None,parents=len(a),exploratory_parent_cluster_CI95=np.quantile(a[rng.integers(0,len(a),(10000,len(a)))].mean(1),[.025,.975]).tolist() if len(a) else None)
csvwr('dino_resolution_transitions.csv',pairs)
wr('dino_evaluation_metrics.json',dict(native_resolution=summaries,paired_512_minus_128=effects,GT_by_image=gtinfo,bootstrap=dict(unit='physical parent',parents=4,replicates=10000,seed=42,interpretation='Exploratory pilot, no multiplicity correction; no rare-event/generalization claim'),proposal_validity=dict(boxes=len(proposalrows),finite_order_valid=sum(r['finite_order_valid'] for r in proposalrows),outside_image=sum(not r['in_image'] for r in proposalrows),no_clipping=True),scope='Existing default prompt pair and frozen consumer only; not best possible DINO or AP evaluation'))
lat=rd(R/'dino_latency_vram.json');lat['summary']={}
for size in [128,512]:
 calls=[r for r in lat['calls'] if r['native_size']==size];warm=[r for r in calls if not r['cold_first_call']];images=[r for r in lat['images'] if r['native_size']==size and not r['cold_first_image']]
 lat['summary'][str(size)]=dict(calls=len(calls),cold_calls=sum(r['cold_first_call'] for r in calls),warm_per_call_end_to_end_ms=dist([r['end_to_end_ms'] for r in warm]),warm_preprocessing_H2D_ms=dist([r['pre'] for r in warm]),warm_forward_ms=dist([r['forward'] for r in warm]),warm_postprocessing_ms=dist([r['post'] for r in warm]),warm_two_prompt_image_end_to_end_ms=dist([r['end_to_end_ms'] for r in images]),peak_allocated_bytes=max(r['memory']['peak_allocated'] for r in calls),peak_reserved_bytes=max(r['memory']['peak_reserved'] for r in calls))
wr('dino_latency_vram.json',lat)
# Frozen Qwen outputs now have an exact same-image DINO baseline. No Qwen reload.
q=rd(Q/'grounding_evaluation_metrics.json');qd=rd(Q/'posthoc_coordinate_diagnostic.json');matched={}
for size in [128,512]:
 d=summaries[str(size)]['image_object_level']['objects'];matched[str(size)]={k:dict(DINO_selected=d[k]['selected_recall'],DINO_raw_top1=d[k]['raw_top1_recall'],DINO_any_proposal_oracle=d[k]['any_proposal_coverage'],Qwen_P0_accepted=q['conditions'][f'native{size}_P0']['image_object_level']['objects'][k]['visible_recall_IoU05'],Qwen_P1_accepted=q['conditions'][f'native{size}_P1']['image_object_level']['objects'][k]['visible_recall_IoU05'],Qwen_P0_diagnostic_only=qd['conditions'][f'native{size}_P0']['objects'][k],Qwen_P1_diagnostic_only=qd['conditions'][f'native{size}_P1']['objects'][k]) for k in ['peg','box']}
wr('matched_qwen_comparison.json',dict(same_images=True,images=32,parents=4,source_input_manifest_sha256=sha(Q/'input_manifest.json'),native_resolution=matched,interpretation='Matched pipeline comparison under different task prompts. Qwen diagnostic-only values are not accepted grounding. Any-proposal DINO coverage is oracle, not a deployed selection. No claim of intrinsic model superiority.'))
# Eight panels, fixed first selected parent, t0/t80, Main/Wrist x two resolutions.
pid=parents[0];panelcases=sorted([c for c in cases.values() if c['parent_id']==pid],key=lambda c:(c['frame_kind'],c['camera'],c['native_size']))
W,H=355,410;canvas=Image.new('RGB',(2*W,4*H),'white');draw=ImageDraw.Draw(canvas)
for j,c in enumerate(panelcases):
 tile=Image.open(c['image_path']).convert('RGB').resize((340,340));d=ImageDraw.Draw(tile);scale=340/c['native_size']
 for k,color,gtcolor in [('peg','orange','lime'),('box','cyan','magenta')]:
  r=lookuprow[(c['native_size'],c['pair_id'],k)]
  for box,col,width in [(r['GT_bbox'],gtcolor,1),(r['selected_box'],color,3)]:
   if box:d.rectangle([x*scale for x in box],outline=col,width=width)
 x=(j%2)*W;y=(j//2)*H;canvas.paste(tile,(x,y+34));draw.text((x+4,y+3),pid+' '+c['frame_kind']+' '+c['camera'],fill='black');draw.text((x+4,y+18),'native '+str(c['native_size'])+' / GT + selected DINO',fill='black')
 rs=[lookuprow[(c['native_size'],c['pair_id'],k)] for k in ['peg','box']];draw.text((x+4,y+378),'IoU '+ ' '.join(f'{r["object"]}={r["selected_IoU"]:.2f}' if r['selected_IoU'] is not None else f'{r["object"]}=absent' for r in rs),fill='black')
canvas.save(R/'dino_native_resolution_comparison.png')
with ExperimentLogger(team='integration',model='Task10I_DINO_native_resolution_evaluation',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
 logger.log({n:{k:z['selected_recall']['rate'] for k,z in m['image_object_level']['objects'].items()} for n,m in summaries.items()});logger.link_output(R/'dino_evaluation_metrics.json')
v.update(stage='EVALUATION_COMPLETE',Qwen_predictions_opened=True,raw_sha256_after_GT=sha(R/'dino_raw_responses.json'),selections_sha256_after_GT=sha(R/'dino_frame_selections.json'));wr('verification.json',v)
print('DINO_EVALUATED',json.dumps({n:{k:dict(selected=z['selected_recall'],top1=z['raw_top1_recall'],any=z['any_proposal_coverage']) for k,z in m['image_object_level']['objects'].items()} for n,m in summaries.items()}),flush=True)
