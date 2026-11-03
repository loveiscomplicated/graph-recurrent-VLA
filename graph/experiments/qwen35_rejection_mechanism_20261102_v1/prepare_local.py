"""Freeze known-error/control cases and one-factor probes before model inference."""
from pathlib import Path
import json,csv,hashlib,collections,math
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'outputs/task10i/stage4a2';PREV=BASE/'qwen35_appearance_bf16_independent_20261102_v1';R=BASE/'qwen35_rejection_mechanism_20261102_v1'
R.mkdir(parents=True,exist_ok=True);assert not (R/'protocol.json').exists()
REMOTE='/workspace/task10i_stage4a2_qwen35_rejection_mechanism_20261102_v1'
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wr(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
rows=[r for r in csv.DictReader((PREV/'candidate_evaluation.csv').open()) if r['condition']=='appearance_three'];cmap={r['id']:r for r in rows};byimage=collections.defaultdict(list)
for r in rows:byimage[r['image_id']].append(r)
images={c['id']:c for c in rd(PREV/'input_manifest.json')['cases']};requests={(r['candidate_id'],r['condition']):r for r in rd(PREV/'request_manifest.json')['requests']}
def loc(p):return PREV/Path(p).relative_to('/workspace/task10i_stage4a2_qwen35_appearance_bf16_independent_20261102_v1')
features=[];cache={};yy,xx=np.indices((512,512))
for r in rows:
 if r['target_good']!='True':continue
 c=images[r['image_id']]
 if c['id'] not in cache:
  with np.load(loc(c['GT_path'])) as z:seg=z['actor_segmentation'].copy()
  cache[c['id']]=(np.array(Image.open(loc(c['image_path'])).convert('RGB')),seg)
 rgb,seg=cache[c['id']];obj=r['query_object'];b=json.loads(r['box']);mask=np.isin(seg,c['object_ids'][obj])&(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]);pix=rgb[mask].astype(float);red=(pix[:,0]>100)&(pix[:,0]>1.25*pix[:,1])&(pix[:,0]>1.25*pix[:,2]);ys,xs=np.where(mask);w=int(xs.max()-xs.min()+1);h=int(ys.max()-ys.min()+1);counts=json.loads(r['actor_pixels']);fg=sum(counts[k] for k in ['peg','box','robot'])
 features.append(dict(candidate_id=r['id'],image_id=r['image_id'],parent_id=r['parent_id'],object=obj,camera=r['camera'],frame_kind=r['frame_kind'],rank=int(r['rank']),label=r['strict_label'],GT_proxy_role=r['GT_role'],accepted=r['accepted_for_query']=='True',target_IoU=float(r[obj+'_IoU']),target_foreground_fraction=counts[obj]/fg,target_visible_pixels=len(pix),red_like_pixels=int(red.sum()),red_like_fraction=float(red.mean()),GT_target_bbox_aspect=max(w,h)/min(w,h),crop_box=json.loads(r['crop_box'])))
wr('all_good_candidate_audit.json',dict(purpose='Posthoc color/actor/geometry audit, not inference input',red_proxy='R>100 and R>1.25G and R>1.25B inside GT target actor and proposal; RGB proxy, not a material-label oracle',candidates=features))
casegroups=collections.defaultdict(list);selected={};pairing=[]
def add(group,r):
 if r['id'] in selected:return
 selected[r['id']]=r;casegroups[group].append(r['id'])
def rank(rs):return sorted(rs,key=lambda r:(-float(r['DINO_score']),int(r['rank']),r['id']))
for obj,group in [('peg','peg_rejected'),('box','box_bbox_rejected')]:
 errors=[r for r in rows if r['query_object']==obj and r['target_good']=='True' and r['accepted_for_query']=='False'];error_images=sorted({r['image_id'] for r in errors})
 for image in error_images:
  error=rank([r for r in errors if r['image_id']==image])[0];add(group,error)
  controls=rank([r for r in rows if r['parent_id']==error['parent_id'] and r['camera']==error['camera'] and r['frame_kind']=='F0' and r['query_object']==obj and r['target_good']=='True' and r['accepted_for_query']=='True'])
  control=controls[0] if controls else None
  if control:add(obj+'_matched_initial_control',control)
  pairing.append(dict(error=error['id'],control=control['id'] if control else None,reason='same parent/camera, initial, IoU>=.5 accepted' if control else 'No matching initial accepted good proposal; no replacement'))
for camera in ['base_camera','hand_camera']:
 available=[r for r in rows if r['camera']==camera and r['frame_kind']=='F1' and r['GT_role']=='ROBOT' and r['strict_label']=='ROBOT' and float(r['peg_IoU'])<.3 and float(r['box_IoU'])<.3 and r['id'] not in selected];best=[rank([r for r in available if r['image_id']==i])[0] for i in sorted({r['image_id'] for r in available})];best.sort(key=lambda r:hashlib.sha256(('rejection_robot_control|'+r['id']).encode()).hexdigest());assert len(best)>=4
 for r in best[:4]:add('robot_control',r)
assert len(casegroups['peg_rejected'])==20 and len(casegroups['box_bbox_rejected'])==6
base=rd(PREV/'prompt_identity.json')['conditions']['appearance_three'];text=base['prompt'];first='Identify the object shown in the last image, which is a crop of the candidate.\nIf other images are provided, the first shows the full scene and the second shows the candidate outlined.';assert text.startswith(first)
prompts={
 'baseline':dict(**base),
 'crop_only':dict(**base),
 'remove_peg_color':dict(**base),
}
# Each non-context contrast changes only the declared part of the same frozen text.
prompts['remove_peg_color']['prompt']=text.replace('A red-and-white elongated rectangular piece.','An elongated rectangular piece.')
prompts['remove_peg_shape']=dict(**base);prompts['remove_peg_shape']['prompt']=text.replace('A red-and-white elongated rectangular piece.','A red-and-white piece.')
prompts['explicit_image_binding']=dict(**base);prompts['explicit_image_binding']['prompt']=text.replace(first,'Identify the candidate outlined in image 2.\nImage 1 shows the complete scene. Image 3 is a crop of the outlined region.')
caption='Images 1, 2, and 3 show the full scene, a marked candidate, and a crop of that candidate. Describe only the candidate in image 3: its visible colors, shape, and recognizable physical parts. If it cannot be identified clearly, say so. Do not infer unseen parts or a task role. Use at most two short sentences.'
prompts['caption']=dict(preamble='',prompt=caption,images='three',labels=None,max_new_tokens=96)
prompts['crop_only']['images']='crop'
requests_out=[];case_manifest=[]
for cid,r in selected.items():
 old=requests[(cid,'appearance_three')];assert all(sha(loc(p))==h for p,h in zip(old['images'],old['image_sha256']));error=cid in casegroups['peg_rejected']+casegroups['box_bbox_rejected']
 for condition,p in prompts.items():
  if condition=='caption' and not error:continue
  paths=old['images'][-1:] if condition=='crop_only' else old['images'];hashes=old['image_sha256'][-1:] if condition=='crop_only' else old['image_sha256'];requests_out.append(dict(id=cid+'__'+condition,candidate_id=cid,condition=condition,images=paths,image_sha256=hashes))
 f=next((f for f in features if f['candidate_id']==cid),None)
 case_manifest.append(dict(candidate_id=cid,group=next(g for g,ids in casegroups.items() if cid in ids),image_id=r['image_id'],parent_id=r['parent_id'],camera=r['camera'],frame_kind=r['frame_kind'],query_object=r['query_object'],baseline_label=r['strict_label'],GT_proxy_role=r['GT_role'],GT_target_box_IoU=float(r[r['query_object']+'_IoU']),feature_audit=f))
requests_out.sort(key=lambda r:hashlib.sha256(('rejection_mechanism_order|'+r['id']).encode()).hexdigest());assert len(requests_out)<=400
wr('request_manifest.json',dict(requests=requests_out));wr('case_manifest.json',dict(groups=dict(casegroups),cases=case_manifest,matched_initial_pairs=pairing,selection='One highest DINO-score rejected IoU>=.5 proposal per object/camera-frame; all20 Peg and6 Box error frames, same-parent/camera initial accepted controls where available,8 deterministic Robot negative controls. Known-error diagnostic, not accuracy/generalization cohort.'))
for condition,p in prompts.items():(R/(condition+'_prompt.txt')).write_text(p['prompt'])
wr('prompt_identity.json',dict(conditions=prompts,source_baseline_sha256=sha(PREV/'appearance_three_prompt.txt'),prompt_sha256={k:sha(R/(k+'_prompt.txt')) for k in prompts}))
wr('protocol.json',dict(task='10I Stage4A-2 why good boxes are rejected: controlled mechanism probes',model='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',precision='BF16',device='cuda:0',attention='sdpa',offloading=False,processor=dict(min_pixels=262144,max_pixels=262144),cases=len(selected),requests=len(requests_out),classification_conditions=[k for k in prompts if k!='caption'],caption_error_requests=26,selection='Known historical errors and matched controls; no new performance/generalization claim',contrasts=dict(crop_only='same text, only last clean crop image',remove_peg_color='same3images, delete red-and-white requirement only',remove_peg_shape='same3images, delete elongated rectangular requirement only',explicit_image_binding='same3images, replace first2sentences with explicit image numbers'),budget=dict(Qwen_loads=1,Qwen_requests=len(requests_out),DINO=0,SAM2=0,simulator=0,training=0,retries=0,warmups=0,adaptive_prompt_edits=0),input='Preserved full/marked/crop RGB + frozen text only; no DINO scores/query/GT labels or positions in VLM text',interpretation='Color proxy is observational. Captions are generated descriptions, not access to internal reasoning. Label changes under a one-factor probe are measured causal effects of that input edit on these selected cases only.',production=False,original_runner='PAUSED_SCOPE_REVIEW'))
wr('model_identity.json',rd(PREV/'model_identity.json'))
print(json.dumps(dict(cases=len(selected),requests=len(requests_out),groups={k:len(v) for k,v in casegroups.items()},missing_controls=sum(x['control'] is None for x in pairing)),indent=2))
