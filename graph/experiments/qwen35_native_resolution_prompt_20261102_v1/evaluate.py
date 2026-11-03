from common import *
import collections
import numpy as np
from PIL import Image,ImageDraw
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='INFERENCE_COMPLETE' and (R/'all_predictions_frozen.json').exists()
assert v['grounding_requests']==64 and v['sam2_camera_frames']==64
assert sha(R/'full_image_grounding_raw_responses.json')==v['raw_SHA_before_GT']
assert sha(R/'full_image_grounding_predictions.csv')==v['predictions_SHA_before_GT']
assert sha(R/'.execution/sam2_raw_initializations.json')==v['sam2_raw_SHA_before_GT']
mf=rd(R/'input_manifest.json');cases={c['id']:c for c in mf['cases']};parents=[p['parent_id'] for p in mf['parents']]
raw=rd(R/'full_image_grounding_raw_responses.json')['responses'];pred={(r['variant'],r['pair_id']):r for r in raw}
truth={};gtinfo={}
def bbox(mask):
 y,x=np.where(mask);return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
def iou(a,b):
 if a is None or b is None:return 0.
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
 return inter/u if u else 0.
def dist(xs):
 xs=[x for x in xs if x is not None];return dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,median=float(np.median(xs)) if xs else None,p95=float(np.quantile(xs,.95)) if xs else None,min=min(xs) if xs else None,max=max(xs) if xs else None)
def rate(rs,fn):
 n=sum(bool(fn(r)) for r in rs);return dict(numerator=n,denominator=len(rs),rate=n/len(rs) if rs else None)
def serialized(rs):return [{k:json.dumps(x) if isinstance(x,(list,dict)) else x for k,x in r.items()} for r in rs]
v.update(stage='POSTHOC_GT_EVALUATION',GT_metrics_opened=True);wr('verification.json',v)
for c in cases.values():
 assert sha(c['GT_path'])==c['GT_sha256']
 with np.load(c['GT_path'],allow_pickle=False) as z:seg=z['actor_segmentation'].copy()
 assert seg.shape==(c['height'],c['width'])
 masks={k:np.isin(seg,c['object_ids'][k]) for k in ['peg','box','robot']};masks['background']=~(masks['peg']|masks['box']|masks['robot']);truth[c['id']]=masks
 gtinfo[c['id']]=dict(visible_pixels={k:int(m.sum()) for k,m in masks.items()},bboxes={k:bbox(m) for k,m in masks.items()},segmentation_sha256=c['GT_sha256'])
rows=[]
for r in raw:
 c=cases[r['id']];g=gtinfo[c['id']];masks=truth[c['id']];yy,xx=np.indices((c['height'],c['width']))
 for k,o in r['parsed']['objects'].items():
  b=o['pixel'];n=g['visible_pixels'][k];gb=g['bboxes'][k];vis='VISIBLE' if n>=16 else 'TINY_VISIBLE' if n else 'ABSENT';ov=iou(b,gb)
  rect=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]) if b else np.zeros_like(xx,dtype=bool)
  counts={a:int((rect&m).sum()) for a,m in masks.items()};fg=sum(counts[a] for a in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda a:counts[a]);wrong=bool(b and ov<.5 and dom!=k and counts[dom]>=16 and fg and counts[dom]/fg>=.7)
  result=('CORRECT' if b and ov>=.5 else 'NULL_MISS' if o['status']=='NULL' else 'INVALID_MISS' if b is None else 'WRONG_OBJECT' if wrong else 'MISLOCALIZED') if vis=='VISIBLE' else 'TINY_VISIBLE' if vis=='TINY_VISIBLE' else 'HALLUCINATED_BOX' if b else 'CORRECT_ABSTENTION' if o['status']=='NULL' else 'INVALID_ABSENT'
  center=float(math.hypot((b[0]+b[2]-gb[0]-gb[2])/2,(b[1]+b[3]-gb[1]-gb[3])/2)) if b and gb else None
  rows.append(dict(variant=r['variant'],id=c['id'],pair_id=c['pair_id'],parent_id=c['parent_id'],camera=c['camera'],frame_kind=c['frame_kind'],timestep=c['timestep'],native_size=c['native_size'],object=k,json_valid=r['parsed']['json_valid'],bare_json=r['parsed']['strict_json_valid'],status=o['status'],normalized=o['normalized'],pixel=b,visible_pixels=n,visibility=vis,GT_bbox=gb,bbox_IoU=ov if n else None,center_error_px=center,center_error_in_128px_units=center*128/c['native_size'] if center is not None else None,actor_pixels=counts,wrong_object=wrong,semantic_confusion=f'{k}->{dom}' if wrong else None,result=result))
csvwr('.execution/object_evaluation.csv',serialized(rows));csvwr('grounding_failure_cases.csv',serialized([r for r in rows if r['result'] not in ['CORRECT','CORRECT_ABSTENTION']]))
def met(rs):
 imgs={r['id']:r for r in rs};m=dict(images=len(imgs),object_rows=len(rs),image_JSON_validity=rate(list(imgs.values()),lambda r:r['json_valid']),image_bare_JSON_validity=rate(list(imgs.values()),lambda r:r['bare_json']),coordinate_or_null_validity=rate(rs,lambda r:r['status'] in ['VALID_BOX','NULL']),null_rate=rate(rs,lambda r:r['status']=='NULL'),objects={})
 for k in ['peg','box']:
  allk=[r for r in rs if r['object']==k];vis=[r for r in allk if r['visibility']=='VISIBLE'];absent=[r for r in allk if r['visibility']=='ABSENT'];em=[r for r in allk if r['status']=='VALID_BOX' and r['visibility']!='TINY_VISIBLE']
  m['objects'][k]=dict(visible_recall_IoU05=rate(vis,lambda r:r['result']=='CORRECT'),IoU_thresholds={str(t):rate(vis,lambda r:r['status']=='VALID_BOX' and r['bbox_IoU']>=t) for t in [.3,.5,.7]},valid_box_emission=rate(vis,lambda r:r['status']=='VALID_BOX'),missing_detection=rate(vis,lambda r:r['result']!='CORRECT'),hallucination_absent=rate(absent,lambda r:r['status']=='VALID_BOX'),false_discovery=rate(em,lambda r:r['result']!='CORRECT'),wrong_object_fraction=rate(em,lambda r:r['wrong_object']),bbox_IoU_including_misses=dist([r['bbox_IoU'] for r in vis]),bbox_IoU_emitted_only=dist([r['bbox_IoU'] for r in vis if r['status']=='VALID_BOX']),center_error_px=dist([r['center_error_px'] for r in vis]),center_error_in_128px_units=dist([r['center_error_in_128px_units'] for r in vis]),semantic_confusion=dict(collections.Counter(r['semantic_confusion'] for r in allk if r['semantic_confusion'])),visibility=dict(collections.Counter(r['visibility'] for r in allk)))
 scenes=collections.defaultdict(list)
 for r in rs:scenes[r['id']].append(r)
 both=[z for z in scenes.values() if len(z)==2 and all(r['visibility']=='VISIBLE' for r in z)]
 m['scene']=dict(both_correct=rate(both,lambda z:all(r['result']=='CORRECT' for r in z)),any_wrong_object=rate(list(scenes.values()),lambda z:any(r['wrong_object'] for r in z)),any_incorrect_emitted=rate(list(scenes.values()),lambda z:any(r['status']=='VALID_BOX' and r['result']!='CORRECT' for r in z)))
 return m
# Paired comparisons use only targets visible >=16px at BOTH native resolutions.
common_support={(r['pair_id'],r['object']) for r in rows if r['native_size']==128 and r['visibility']=='VISIBLE'} & {(r['pair_id'],r['object']) for r in rows if r['native_size']==512 and r['visibility']=='VISIBLE'}
metrics={}
for mode in VARIANTS:
 rs=[r for r in rows if r['variant']==mode];pp={p:met([r for r in rs if r['parent_id']==p]) for p in parents}
 metrics[mode]=dict(image_object_level=met(rs),common_support=met([r for r in rs if (r['pair_id'],r['object']) in common_support]),per_parent=pp,parent_macro={k:float(np.mean([z['objects'][k]['visible_recall_IoU05']['rate'] for z in pp.values() if z['objects'][k]['visible_recall_IoU05']['rate'] is not None])) for k in ['peg','box']},by_camera={cam:met([r for r in rs if r['camera']==cam]) for cam in ['base_camera','hand_camera']},by_time={f:met([r for r in rs if r['frame_kind']==f]) for f in ['F0','F1']})
lookup={(r['variant'],r['pair_id'],r['object']):r for r in rows};rng=np.random.default_rng(42);effects={}
contrasts={'resolution_P0':{'native512_P0':1,'native128_P0':-1},'resolution_P1':{'native512_P1':1,'native128_P1':-1},'prompt_native128':{'native128_P1':1,'native128_P0':-1},'prompt_native512':{'native512_P1':1,'native512_P0':-1},'interaction':{'native512_P1':1,'native128_P1':-1,'native512_P0':-1,'native128_P0':1}}
for name,weights in contrasts.items():
 effects[name]={}
 for k in ['peg','box']:
  per=[]
  for p in parents:
   keys=[pair for pair,obj in sorted(common_support) if obj==k and lookup[(VARIANTS[0],pair,k)]['parent_id']==p]
   if not keys:continue
   per.append(dict(parent=p,objects=len(keys),IoU_delta=float(np.mean([sum(w*lookup[(mode,pair,k)]['bbox_IoU'] for mode,w in weights.items()) for pair in keys])),recall_delta=float(np.mean([sum(w*(lookup[(mode,pair,k)]['result']=='CORRECT') for mode,w in weights.items()) for pair in keys]))))
  vals={metric:np.array([p[metric] for p in per]) for metric in ['IoU_delta','recall_delta']}
  effects[name][k]=dict(parents=len(per),per_parent=per,**{metric:dict(parent_macro_delta=float(a.mean()) if len(a) else None,exploratory_parent_cluster_CI95=np.quantile(a[rng.integers(0,len(a),(10000,len(a)))].mean(1),[.025,.975]).tolist() if len(a) else None) for metric,a in vals.items()})
samraw=rd(R/'.execution/sam2_raw_initializations.json');srows=[]
for s in samraw:
 c=cases[s['id']];masks=truth[c['id']];assert sha(s['masks_path'])==s['masks_sha256']
 with np.load(s['masks_path']) as z:saved={k:z[k].copy() for k in z.files}
 for k in ['peg','box']:
  mask=saved.get(k);g=masks[k];n=int(g.sum());npix=int(mask.sum()) if mask is not None else 0;tp=int((mask&g).sum()) if mask is not None else 0;union=npix+n-tp;miou=tp/union if union else None;counts={a:int((mask&m).sum()) if mask is not None else 0 for a,m in masks.items()};fg=sum(counts[a] for a in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda a:counts[a]);wrong=bool(dom!=k and counts[dom]>=16 and fg and counts[dom]/fg>=.7)
  srows.append(dict(variant=s['variant'],id=c['id'],pair_id=c['pair_id'],parent_id=c['parent_id'],camera=c['camera'],frame_kind=c['frame_kind'],native_size=c['native_size'],object=k,initialized=mask is not None,prediction_status=lookup[(s['variant'],s['pair_id'],k)]['status'],visible_pixels=n,visibility='VISIBLE' if n>=16 else 'TINY_VISIBLE' if n else 'ABSENT',mask_pixels=npix,mask_IoU=miou,precision=tp/npix if npix else None,recall=tp/n if n else None,actor_pixels=counts,robot_contamination=counts['robot']/npix if npix else None,background_contamination=counts['background']/npix if npix else None,wrong_object=wrong,initialization_success=bool(mask is not None and n>=16 and miou>=.8)))
csvwr('sam2_initialization_cases.csv',serialized(srows))
def sm(rs):
 return {k:dict(initialized=rate([r for r in rs if r['object']==k],lambda r:r['initialized']),IoU08_success=rate([r for r in rs if r['object']==k and r['visibility']=='VISIBLE'],lambda r:r['initialization_success']),mask_IoU_including_noninitialization=dist([r['mask_IoU'] for r in rs if r['object']==k and r['visibility']=='VISIBLE']),mask_IoU_initialized_only=dist([r['mask_IoU'] for r in rs if r['object']==k and r['visibility']=='VISIBLE' and r['initialized']]),precision=dist([r['precision'] for r in rs if r['object']==k]),recall=dist([r['recall'] for r in rs if r['object']==k]),wrong_object=rate([r for r in rs if r['object']==k and r['initialized']],lambda r:r['wrong_object']),robot_contamination=dist([r['robot_contamination'] for r in rs if r['object']==k]),background_contamination=dist([r['background_contamination'] for r in rs if r['object']==k])) for k in ['peg','box']}
sammetrics={mode:dict(objects=sm([r for r in srows if r['variant']==mode]),by_camera={cam:sm([r for r in srows if r['variant']==mode and r['camera']==cam]) for cam in ['base_camera','hand_camera']},by_time={f:sm([r for r in srows if r['variant']==mode and r['frame_kind']==f]) for f in ['F0','F1']}) for mode in VARIANTS}
wr('sam2_initialization_metrics.json',sammetrics)
lat=rd(R/'grounding_latency_vram.json')
lat['summary']={mode:dict(n=sum(r['variant']==mode for r in lat['requests']),cold_requests=sum(r['variant']==mode and r['cold_first_request'] for r in lat['requests']),warm_end_to_end_ms=dist([r['end_to_end_ms'] for r in lat['requests'] if r['variant']==mode and not r['cold_first_request']]),preprocessing_ms=dist([r['preprocessing_ms'] for r in lat['requests'] if r['variant']==mode]),vision_ms=dist([r['vision_cuda_ms'] for r in lat['requests'] if r['variant']==mode]),prefill_excluding_vision_approx_ms=dist([r['first_forward_vision_prefill_cuda_ms']-r['vision_cuda_ms'] for r in lat['requests'] if r['variant']==mode]),decode_forward_ms=dist([r['decode_forward_cuda_ms'] for r in lat['requests'] if r['variant']==mode]),output_tokens=dist([r['output_tokens'] for r in lat['requests'] if r['variant']==mode]),at_output_limit=sum(r['output_tokens']==128 for r in lat['requests'] if r['variant']==mode),peak_allocated_bytes=max(r['memory']['peak_allocated'] for r in lat['requests'] if r['variant']==mode),SAM2_frame_wall_ms=dist([r['wall_ms'] for r in lat['sam2'] if r['variant']==mode])) for mode in VARIANTS}
wr('grounding_latency_vram.json',lat)
wr('grounding_evaluation_metrics.json',dict(conditions=metrics,paired_effects=effects,GT_by_image=gtinfo,common_support_objects={k:sum(o==k for _,o in common_support) for k in ['peg','box']},bootstrap=dict(unit='physical parent',parents=4,replicates=10000,seed=42,interpretation='Exploratory pilot only; no multiplicity adjustment; no rare-error/generalization claim'),DINO_comparison='MATCHED_DINO_BASELINE_INSUFFICIENT: fresh static recaptures, no saved DINO on these exact images; no new DINO run',DINO_replacement='NOT_SUPPORTED_BY_PILOT'))
# Compact prespecified 8-panel visualization: first selected parent F0, both cameras.
pid=parents[0];pairids=[next(c['pair_id'] for c in cases.values() if c['parent_id']==pid and c['frame_kind']=='F0' and c['camera']==cam) for cam in ['base_camera','hand_camera']]
W,H=330,390;canvas=Image.new('RGB',(4*W,2*H),'white');draw=ImageDraw.Draw(canvas)
for j,pair in enumerate(pairids):
 for i,mode in enumerate(VARIANTS):
  r=pred[(mode,pair)];c=cases[r['id']];im=np.asarray(Image.open(c['image_path']).convert('RGB')).copy();s=next(s for s in samraw if s['variant']==mode and s['pair_id']==pair)
  with np.load(s['masks_path']) as z:
   for k in z.files:im[z[k]]=(im[z[k]]*.7+np.array([255,100,0] if k=='peg' else [0,150,255])*.3).astype(np.uint8)
  img=Image.fromarray(im).resize((320,320));d=ImageDraw.Draw(img);factor=320/c['native_size']
  for k,color,gtcolor in [('peg','orange','lime'),('box','cyan','magenta')]:
   b=r['parsed']['objects'][k]['pixel'];gb=gtinfo[c['id']]['bboxes'][k]
   if b:d.rectangle([x*factor for x in b],outline=color,width=2)
   if gb:d.rectangle([x*factor for x in gb],outline=gtcolor,width=1)
  x=i*W;y=j*H;canvas.paste(img,(x,y+35));draw.text((x+4,y+3),mode+' '+c['camera'],fill='black');draw.text((x+4,y+18),pid+' t0',fill='black')
  rr=[q for q in rows if q['variant']==mode and q['pair_id']==pair];draw.text((x+4,y+358),'IoU '+ ' '.join(f'{z["object"]}={z["bbox_IoU"]:.2f}' if z['bbox_IoU'] is not None else f'{z["object"]}=absent' for z in rr),fill='black')
canvas.save(R/'full_image_grounding_comparison.png')
with ExperimentLogger(team='integration',model='Task10I_native_resolution_prompt_evaluation',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
 logger.log({mode:{k:z['visible_recall_IoU05']['rate'] for k,z in m['image_object_level']['objects'].items()} for mode,m in metrics.items()});logger.link_output(R/'grounding_evaluation_metrics.json')
v.update(stage='EVALUATION_COMPLETE',GT_metrics_opened=True,raw_SHA_after_GT=sha(R/'full_image_grounding_raw_responses.json'),predictions_SHA_after_GT=sha(R/'full_image_grounding_predictions.csv'));wr('verification.json',v)
print('EVALUATION_COMPLETE',json.dumps({mode:{k:z['visible_recall_IoU05'] for k,z in m['image_object_level']['objects'].items()} for mode,m in metrics.items()}),flush=True)
