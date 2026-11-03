from common import *
import numpy as np, collections, subprocess
from PIL import Image,ImageDraw,ImageFont
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');mf=rd(R/'input_manifest.json')
assert (P/'all_predictions_frozen.json').is_file(), 'All precision variants must finish before GT access'
assert v['stage'] in ['INFERENCE_COMPLETE','RUNTIME_STOPPED'] and v.get('grounding_requests')==99,v.get('error')
assert sha(R/'full_image_grounding_raw_responses.json')==v['raw_response_SHA_before_GT']
assert sha(R/'full_image_grounding_predictions.csv')==v['predictions_SHA_before_GT']
raw=rd(R/'full_image_grounding_raw_responses.json')['responses'];pred={x['id']:x for x in raw}
assert len(raw)==v['grounding_requests']<=99
v.update(GT_semantic_opened=True,stage='POSTHOC_GT_EVALUATION');wr('verification.json',v)
refs=collections.defaultdict(list);refhash={}
prior_GT_info=rd(OLD/'grounding_evaluation_metrics.json')['GT_support_by_image']
# Only preserved sidecars, never simulator construction or replay.
for path in sorted({Path(x['path']) for x in prior_GT_info.values() if x['status']=='EXACT_GT'}):
 receipt=path.parent/'receipt.json'
 if receipt.is_file():
  r=rd(receipt)
  if r.get('raw_sha256'): refs[r['raw_sha256']].append((path,r,receipt))
def bbox(mask):
 y,x=np.where(mask)
 return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
def iou(a,b):
 if a is None or b is None:return 0.
 z=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
 union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z
 return z/union if union else 0.
cache={};truth={};gtinfo={}
def gettruth(c):
 if prior_GT_info[c['id']]['status']!='EXACT_GT':return None,prior_GT_info[c['id']]
 key=(c['raw_sha256'],c['timestep'],c['camera'])
 if key in cache:return cache[key]
 reason='NO_PRESERVED_ACTOR_SEGMENTATION'
 for path,r,receipt in refs.get(c['raw_sha256'],[]):
  expected=r.get('segmentation_sha256') or r.get('files',{}).get('segmentation.npz')
  if str(path) not in refhash:
   actual=sha(path);assert expected and expected==actual,('GT_HASH_MISMATCH',str(path));refhash[str(path)]=actual;refhash[str(receipt)]=sha(receipt)
  with np.load(path,allow_pickle=False) as z:
   t=c['timestep']
   if t>=len(z['exact_frame']) or not bool(z['exact_frame'][t]): reason='PRESERVED_REFERENCE_FRAME_NOT_EXACT';continue
   seg=z['actor_segmentation'][t,0 if c['camera']=='base_camera' else 1].copy()
  assert seg.shape==(128,128)
  masks={k:np.isin(seg,r['object_ids'][k]) for k in ['peg','box']};masks['robot']=np.isin(seg,r['robot_ids']);masks['background']=~(masks['peg']|masks['box']|masks['robot'])
  info=dict(status='EXACT_GT',path=str(path),receipt=str(receipt),visible={k:int(m.sum()) for k,m in masks.items()},bboxes={k:bbox(m) for k,m in masks.items()},source='preserved exact_frame + raw_sha256 match + sidecar SHA validation')
  cache[key]=(masks,info);return cache[key]
 cache[key]=(None,dict(status='GT_UNSUPPORTED',reason=reason));return cache[key]
def composition(b,masks):
 yy,xx=np.indices((128,128));rect=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]);return {k:int((rect&m).sum()) for k,m in masks.items()}
rows=[];diagnostics=[]
for c in mf['cases']:
 if c['id'] not in pred:continue
 masks,info=gettruth(c);gtinfo[c['id']]=info;truth[c['id']]=masks
 for mode,targetrows in [('strict',rows),('diagnostic_only',diagnostics)]:
  p=pred[c['id']]['parsed' if mode=='strict' else 'diagnostic_only']
  for k in ['peg','box']:
   o=p['objects'][k] if p else dict(status='INVALID_JSON',pixel=None,normalized=None);b=o['pixel']
   row=dict(id=c['id'],phase=c['phase'],parent_id=c['parent_id'],frame_kind=c['frame_kind'],timestep=c['timestep'],camera=c['camera'],object=k,mode=mode,json_valid=bool(p and p['json_valid']),prediction_status=o['status'],normalized=o['normalized'],pixel=b,GT_status=info['status'],visible_pixels=None,visibility='GT_UNSUPPORTED',GT_bbox=None,bbox_IoU=None,center_error_px=None,actor_pixels=None,wrong_object=False,semantic_confusion=None,result='GT_UNSUPPORTED')
   if masks is not None:
    n=info['visible'][k];gb=info['bboxes'][k];vis='EVALUABLE_VISIBLE' if n>=16 else 'TINY_VISIBLE' if n>0 else 'ABSENT'
    ov=iou(b,gb) if b else 0.;counts=composition(b,masks) if b else {x:0 for x in masks};other='box' if k=='peg' else 'peg';fg=sum(counts[x] for x in ['peg','box','robot']);dominant=max(['peg','box','robot'],key=lambda x:counts[x]);wrong=bool(b and ov<.5 and dominant!=k and counts[dominant]>=16 and fg and counts[dominant]/fg>=.7)
    result=('CORRECT' if b and ov>=.5 else 'NULL_MISS' if o['status']=='NULL' else 'INVALID_MISS' if b is None else 'WRONG_OBJECT' if wrong else 'MISLOCALIZED') if vis=='EVALUABLE_VISIBLE' else 'TINY_VISIBLE' if vis=='TINY_VISIBLE' else 'HALLUCINATED_BOX' if b else 'CORRECT_ABSTENTION' if o['status']=='NULL' else 'INVALID_ABSENT'
    row.update(visible_pixels=n,visibility=vis,GT_bbox=gb,bbox_IoU=ov if n else None,center_error_px=float(math.hypot((b[0]+b[2]-gb[0]-gb[2])/2,(b[1]+b[3]-gb[1]-gb[3])/2)) if b and gb else None,actor_pixels=counts,wrong_object=wrong,semantic_confusion=f'{k}->{dominant}' if wrong else None,result=result)
   targetrows.append(row)
def csvrows(rs):return [{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rs]
csvwr('.execution/object_evaluation.csv',csvrows(rows));csvwr('.execution/diagnostic_only_object_evaluation.csv',csvrows(diagnostics))
csvwr('grounding_failure_cases.csv',csvrows([r for r in rows if r['result'] not in ['CORRECT','CORRECT_ABSTENTION']]))
def rate(rs,fn):
 n=sum(bool(fn(r)) for r in rs);return dict(numerator=n,denominator=len(rs),rate=n/len(rs) if rs else None)
def dist(xs):
 xs=[x for x in xs if x is not None]
 return dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,median=float(np.median(xs)) if xs else None,p05=float(np.quantile(xs,.05)) if xs else None,p95=float(np.quantile(xs,.95)) if xs else None,min=min(xs) if xs else None,max=max(xs) if xs else None)
def metrics(rs):
 m=dict(object_rows=len(rs),images=len({r['id'] for r in rs}),JSON_validity=rate(rs,lambda r:r['json_valid']),coordinate_contract_validity=rate(rs,lambda r:r['prediction_status'] in ['VALID_BOX','NULL']),null_rate=rate(rs,lambda r:r['prediction_status']=='NULL'),GT_support=collections.Counter(r['visibility'] for r in rs),objects={})
 for k in ['peg','box']:
  allk=[r for r in rs if r['object']==k];visible=[r for r in allk if r['visibility']=='EVALUABLE_VISIBLE'];absent=[r for r in allk if r['visibility']=='ABSENT'];exact_pred=[r for r in allk if r['GT_status']=='EXACT_GT' and r['prediction_status']=='VALID_BOX' and r['visibility']!='TINY_VISIBLE']
  m['objects'][k]=dict(recall_IoU05=rate(visible,lambda r:r['result']=='CORRECT'),valid_box_emission_recall=rate(visible,lambda r:r['prediction_status']=='VALID_BOX'),missing_detection_rate=rate(visible,lambda r:r['result']!='CORRECT'),null_miss_rate=rate(visible,lambda r:r['prediction_status']=='NULL'),hallucinated_box_absent_FPR=rate(absent,lambda r:r['prediction_status']=='VALID_BOX'),false_discovery_fraction=rate(exact_pred,lambda r:r['result']!='CORRECT'),wrong_object_fraction_of_predictions=rate(exact_pred,lambda r:r['wrong_object']),wrong_grounding_fraction_all_evaluable=rate([r for r in allk if r['GT_status']=='EXACT_GT' and r['visibility']!='TINY_VISIBLE'],lambda r:r['prediction_status']=='VALID_BOX' and r['result']!='CORRECT'),semantic_confusion=collections.Counter(r['semantic_confusion'] for r in allk if r['semantic_confusion']),bbox_IoU_including_misses=dist([r['bbox_IoU'] for r in visible]),bbox_IoU_emitted_only=dist([r['bbox_IoU'] for r in visible if r['prediction_status']=='VALID_BOX']),center_error_px_emitted_only=dist([r['center_error_px'] for r in visible]),IoU_thresholds={str(t):rate(visible,lambda r:r['prediction_status']=='VALID_BOX' and r['bbox_IoU']>=t) for t in [.3,.5,.7]},tiny_visible_rows=[r['id'] for r in allk if r['visibility']=='TINY_VISIBLE'])
 scenes=collections.defaultdict(list)
 for r in rs:scenes[r['id']].append(r)
 both=[r for r in scenes.values() if len(r)==2 and all(x['visibility']=='EVALUABLE_VISIBLE' for x in r)]
 supported=[r for r in scenes.values() if all(x['GT_status']=='EXACT_GT' for x in r)]
 m['scene']=dict(both_correct=rate(both,lambda rr:all(x['result']=='CORRECT' for x in rr)),any_incorrect_emitted=rate(supported,lambda rr:any(x['prediction_status']=='VALID_BOX' and x['result'] not in ['CORRECT','TINY_VISIBLE'] for x in rr)),any_wrong_object=rate(supported,lambda rr:any(x['wrong_object'] for x in rr)))
 return m
br=[r for r in rows if r['phase']=='B'];bd=[r for r in diagnostics if r['phase']=='B']
parentids=[p['parent_id'] for p in mf['parents']]
def aggregate(rs):
 overall=metrics(rs);per={p:metrics([r for r in rs if r['parent_id']==p]) for p in parentids};macro={};rng=np.random.default_rng(42)
 paths=[('JSON_validity',),('coordinate_contract_validity',),('null_rate',),('scene','both_correct')]+[('objects',k,m) for k in ['peg','box'] for m in ['recall_IoU05','valid_box_emission_recall','missing_detection_rate','hallucinated_box_absent_FPR','wrong_grounding_fraction_all_evaluable']]
 for path in paths:
  get=lambda m: __import__('functools').reduce(lambda a,k:a[k],path,m)
  rates=[get(m)['rate'] for m in per.values() if get(m)['rate'] is not None]
  nums=np.array([get(m)['numerator'] for m in per.values()]);dens=np.array([get(m)['denominator'] for m in per.values()]);ix=rng.integers(0,len(parentids),(10000,len(parentids)));ns=nums[ix].sum(1);ds=dens[ix].sum(1);valid=ds>0;ci=np.quantile(ns[valid]/ds[valid],[.025,.975]).tolist() if valid.any() else None
  mci=np.quantile(np.array(rates)[rng.integers(0,len(rates),(10000,len(rates)))].mean(1),[.025,.975]).tolist() if rates else None
  macro['/'.join(path)]=dict(parent_macro_rate=float(np.mean(rates)) if rates else None,parents_with_support=len(rates),parent_macro_CI95=mci,parent_cluster_image_rate_CI95=ci)
 return dict(image_object_level=overall,parent_macro=macro,per_parent=per,by_camera={cam:metrics([r for r in rs if r['camera']==cam]) for cam in ['base_camera','hand_camera']},by_time={f:metrics([r for r in rs if r['frame_kind']==f]) for f in ['F0','F1']},by_camera_time={cam+'_'+f:metrics([r for r in rs if r['camera']==cam and r['frame_kind']==f]) for cam in ['base_camera','hand_camera'] for f in ['F0','F1']})
met=dict(primary=aggregate(br),diagnostic_only=aggregate(bd),phase_A=rows[:6],phase_A_diagnostic_only=diagnostics[:6],GT_support_by_image=gtinfo,metric_contract=mf['protocol'],bootstrap=dict(unit='physical parent',replicates=10000,seed=42,warning='degenerate zero-error/zero-success intervals do not prove rare-event safety'),strict_vs_diagnostic='Both views use the unchanged frozen native-array parser; bare JSON is reported separately in the precision summary')
# Assess only the prespecified SAM2 cases, retaining null/invalid cases in denominator.
sr=rd(R/'.execution/sam2_raw_initializations.json') if (R/'.execution/sam2_raw_initializations.json').exists() else []
sammap={x['id']:x for x in sr};srows=[]
for cid in mf['protocol']['sam2_ids']:
 info=gtinfo.get(cid,dict(status='GT_UNSUPPORTED'));masks=truth.get(cid);s=sammap.get(cid)
 saved={}
 if s:
  assert sha(s['masks_path'])==s['masks_sha256']
  with np.load(s['masks_path']) as z:saved={k:z[k].copy() for k in z.files}
 for k in ['peg','box']:
  mask=saved.get(k);rr=next((r for r in rows if r['id']==cid and r['object']==k),None)
  row=dict(id=cid,parent_id=rr['parent_id'] if rr else next(c['parent_id'] for c in mf['cases'] if c['id']==cid),camera=next(c['camera'] for c in mf['cases'] if c['id']==cid),frame_kind=next(c['frame_kind'] for c in mf['cases'] if c['id']==cid),object=k,status='INITIALIZED' if mask is not None else 'NOT_INITIALIZED',reason=rr['prediction_status'] if rr else 'NO_QWEN_RESPONSE',GT_status=info['status'],visible_pixels=None,visibility='GT_UNSUPPORTED',mask_pixels=int(mask.sum()) if mask is not None else 0,mask_IoU=None,precision=None,recall=None,actor_pixels=None,robot_contamination=None,background_contamination=None,wrong_object=False,initialization_success=False)
  if masks is not None:
   n=info['visible'][k];tp=int((mask&masks[k]).sum()) if mask is not None else 0;npix=row['mask_pixels'];counts={a:int((mask&m).sum()) if mask is not None else 0 for a,m in masks.items()};miou=tp/(npix+n-tp) if (npix+n-tp) else None
   fg=sum(counts[a] for a in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda a:counts[a]);wrong=dom!=k and fg>0 and counts[dom]>=16 and counts[dom]/fg>=.7
   row.update(visible_pixels=n,visibility='EVALUABLE_VISIBLE' if n>=16 else 'TINY_VISIBLE' if n else 'ABSENT',mask_IoU=miou,precision=tp/npix if npix else None,recall=tp/n if n else None,actor_pixels=counts,robot_contamination=counts['robot']/npix if npix else None,background_contamination=counts['background']/npix if npix else None,wrong_object=wrong,initialization_success=bool(mask is not None and n>=16 and miou>=.8))
  srows.append(row)
csvwr('sam2_initialization_cases.csv',csvrows(srows))
def smetrics(rs):
 return {k:dict(selected_objects=sum(r['object']==k for r in rs),initialized=rate([r for r in rs if r['object']==k],lambda r:r['status']=='INITIALIZED'),end_to_end_IoU08_success=rate([r for r in rs if r['object']==k and r['visibility']=='EVALUABLE_VISIBLE'],lambda r:r['initialization_success']),mask_IoU_initialized_only=dist([r['mask_IoU'] for r in rs if r['object']==k and r['visibility']=='EVALUABLE_VISIBLE' and r['status']=='INITIALIZED']),mask_IoU_including_noninitialization=dist([r['mask_IoU'] for r in rs if r['object']==k and r['visibility']=='EVALUABLE_VISIBLE']),precision=dist([r['precision'] for r in rs if r['object']==k]),recall=dist([r['recall'] for r in rs if r['object']==k]),wrong_object=rate([r for r in rs if r['object']==k and r['status']=='INITIALIZED' and r['GT_status']=='EXACT_GT'],lambda r:r['wrong_object'])) for k in ['peg','box']}
sm=dict(camera_frames_selected=16,camera_frames_processed=len(sr),actual_initialized_objects=sum(r['status']=='INITIALIZED' for r in srows),objects=smetrics(srows),by_camera={c:smetrics([r for r in srows if r['camera']==c]) for c in ['base_camera','hand_camera']},by_time={f:smetrics([r for r in srows if r['frame_kind']==f]) for f in ['F0','F1']},per_parent={p:smetrics([r for r in srows if r['parent_id']==p]) for p in sorted({r['parent_id'] for r in srows})},null_invalid_not_replaced_with_GT=True,tracking=False,production_memory_commit=False)
wr('sam2_initialization_metrics.json',sm)
lat=rd(R/'grounding_latency_vram.json');warm=lat['requests'][1:]
lat['summary']=dict(first_request=lat['requests'][0] if lat['requests'] else None,warm_requests=len(warm),warm_end_to_end_ms=dist([r['end_to_end_ms'] for r in warm]),warm_B_only_end_to_end_ms=dist([r['end_to_end_ms'] for r in lat['requests'] if r['phase']=='B']),warm_preprocessing_ms=dist([r['preprocessing_ms'] for r in warm]),warm_vision_ms=dist([r['vision_cuda_ms'] for r in warm]),warm_first_forward_vision_prefill_ms=dist([r['first_forward_vision_prefill_cuda_ms'] for r in warm]),warm_prefill_excluding_vision_approx_ms=dist([r['first_forward_vision_prefill_cuda_ms']-r['vision_cuda_ms'] for r in warm]),warm_decode_forward_ms=dist([r['decode_forward_cuda_ms'] for r in warm]),tokens=collections.Counter(r['output_tokens'] for r in lat['requests']),peak_allocated_bytes=max(r['memory']['peak_allocated'] for r in lat['requests']),peak_reserved_bytes=max(r['memory']['peak_reserved'] for r in lat['requests']),scope='model load excludes processor and download; file-cache warmed by checkpoint hashing; E2E PNG read+processor+H2D+generation+text decode; CUDA event first-forward includes vision; prefill difference approximate; no extra timing requests')
wr('grounding_latency_vram.json',lat)
overall=met['primary']['image_object_level'];valid=overall['coordinate_contract_validity']['rate'];pg=overall['objects']['peg'];bx=overall['objects']['box']
status='GROUNDING_RUNTIME_BLOCKED' if valid==0 else 'VLM_GROUNDING_INSUFFICIENT' if any(m['recall_IoU05']['rate'] is None or m['recall_IoU05']['rate']<.5 for m in [pg,bx]) else 'VLM_GROUNDING_PARTIAL'
met['final_status']=status
met['DINO_replacement_decision']=dict(status='NOT_SUPPORTED',checks=dict(peg_recall=pg['recall_IoU05'],box_recall=bx['recall_IoU05'],wrong_peg_grounding=pg['wrong_grounding_fraction_all_evaluable'],valid_coordinate_output=overall['coordinate_contract_validity'],SAM2_peg=sm['objects']['peg']['end_to_end_IoU08_success'],SAM2_box=sm['objects']['box']['end_to_end_IoU08_success']),support='INSUFFICIENT_SUPPORT for complete 24-parent GT, rare errors, absent/tiny strata, and SAM2 conditional accuracy when no initialized masks; failed criteria are not rescued by missing support')
wr('grounding_evaluation_metrics.json',met)
wr('.execution/GT_reference_hashes_at_open.json',refhash)
print('EVALUATED',status,'strict',json.dumps(overall),'diagnostic',json.dumps(met['diagnostic_only']['image_object_level']),flush=True)
# A compact fixed panel: Phase A plus first prespecified case in each camera/time condition.
selected=['P','N','C1']
for cam in ['base_camera','hand_camera']:
 for f in ['F0','F1']:
  selected.append(next(cid for cid in mf['protocol']['sam2_ids'] if next(c for c in mf['cases'] if c['id']==cid)['camera']==cam and next(c for c in mf['cases'] if c['id']==cid)['frame_kind']==f))
scale=3;W=430;H=475;canvas=Image.new('RGB',(W*4,H*2),(245,245,245));draw=ImageDraw.Draw(canvas)
for index,cid in enumerate(selected):
 c=next(c for c in mf['cases'] if c['id']==cid);im=Image.open(c['image_path']).convert('RGB');smasks={}
 if cid in sammap:
  with np.load(sammap[cid]['masks_path']) as z:smasks={k:z[k] for k in z.files}
 arr=np.asarray(im).copy()
 for k,mask in smasks.items():arr[mask]=(arr[mask]*.6+np.array([255,0,255] if k=='peg' else [0,200,220])*.4).astype(np.uint8)
 im=Image.fromarray(arr).resize((384,384),Image.Resampling.NEAREST);d=ImageDraw.Draw(im);info=gtinfo[cid]
 for k,color in [('peg',(220,40,60)),('box',(30,100,240))]:
  p=pred[cid]['parsed']['objects'][k];b=p['pixel'];diag=pred[cid]['diagnostic_only'];db=diag['objects'][k]['pixel'] if diag else None
  if db and not b:
   z=[x*scale for x in db];x1,y1,x2,y2=z
   for x in range(int(x1),int(x2),10):d.line((x,y1,min(x+5,x2),y1),fill=color,width=2);d.line((x,y2,min(x+5,x2),y2),fill=color,width=2)
   for y in range(int(y1),int(y2),10):d.line((x1,y,x1,min(y+5,y2)),fill=color,width=2);d.line((x2,y,x2,min(y+5,y2)),fill=color,width=2)
  if b:d.rectangle([x*scale for x in b],outline=color,width=3)
  if info['status']=='EXACT_GT' and info['bboxes'][k]:d.rectangle([x*scale for x in info['bboxes'][k]],outline=(0,210,80) if k=='peg' else (255,180,0),width=2)
 x=(index%4)*W;y=(index//4)*H;canvas.paste(im,(x+15,y+55));draw.text((x+15,y+10),f"{cid[:40]}\nt{c['timestep']} {c['camera']} / {info['status']}",fill='black');draw.text((x+15,y+443),'JSON '+str(pred[cid]['parsed']['json_valid'])+' / SAM2 '+str(bool(smasks)),fill='black')
lx=3*W+15;ly=H+30
draw.multiline_text((lx,ly),'Accepted Qwen: solid\nQwen PEG: red\nQwen BOX: blue\nGT PEG: green\nGT BOX: orange\nSAM2: overlay only if run\n\nFrozen bare/whole-fence parser\nRejected output has no box\n\nNo DINO inputs\nNo GT used for predictions',fill='black',spacing=7)
canvas.save(R/'full_image_grounding_comparison.png')
v.update(stage='EVALUATION_COMPLETE',raw_response_SHA_after_GT=sha(R/'full_image_grounding_raw_responses.json'),predictions_SHA_after_GT=sha(R/'full_image_grounding_predictions.csv'),GT_reference_files=refhash,final_status=status)
wr('verification.json',v)
