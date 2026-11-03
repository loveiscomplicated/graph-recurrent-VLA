from common import *
import numpy as np, collections
from PIL import Image,ImageDraw
v=rd(R/'verification.json');assert v['stage']=='COMPLETE_VERIFIED'
mf=rd(R/'input_manifest.json');raw=rd(R/'full_image_grounding_raw_responses.json')['responses'];pm={x['id']:x for x in raw};cm={x['id']:x for x in mf['cases']}
m=rd(R/'grounding_evaluation_metrics.json');b=[x for x in raw if cm[x['id']]['phase']=='B']
m['image_level_format']=dict(strict_JSON_valid=dict(numerator=sum(x['parsed']['json_valid'] for x in b),denominator=len(b)),both_objects_valid_coordinates=dict(numerator=sum(all(o['status']=='VALID_BOX' for o in x['parsed']['objects'].values()) for x in b),denominator=len(b)),null_images=sum(any(o['status']=='NULL' for o in x['parsed']['objects'].values()) for x in b),fenced_JSON_images=sum(x['raw_text'].lstrip().startswith('```') for x in b),diagnostic_coordinate_images=sum(x['diagnostic_only'] is not None and all(o['status']=='VALID_BOX' for o in x['diagnostic_only']['objects'].values()) for x in b))
m['GT_support_summary']=dict(selected_parents=24,exact_supported_parents=sum(z['objects']['peg']['recall_IoU05']['denominator']>0 for z in m['primary']['per_parent'].values()),images=dict(collections.Counter(z['status'] for cid,z in m['GT_support_by_image'].items() if cm[cid]['phase']=='B')),unsupported_reasons=dict(collections.Counter(z.get('reason') for cid,z in m['GT_support_by_image'].items() if cm[cid]['phase']=='B' and z['status']!='EXACT_GT')),reference_contract='Only Stage3 R/D TRAIN selected raw SHA matched sidecars. Historical cache directory names do not redefine selected split; no Stage4 new Final/Test collection or input.')
wr('grounding_evaluation_metrics.json',m)
sam=rd(R/'sam2_initialization_metrics.json');rs=list(csv.DictReader((R/'sam2_initialization_cases.csv').open()));ps=sorted({r['parent_id'] for r in rs});rng=np.random.default_rng(42);bs={}
for k in ['peg','box']:
 counts=[];rates=[]
 for p in ps:
  rr=[r for r in rs if r['parent_id']==p and r['object']==k and r['visibility']=='EVALUABLE_VISIBLE'];n=sum(r['initialization_success']=='True' for r in rr);d=len(rr);counts.append((n,d))
  if d:rates.append(n/d)
 a=np.array(counts);ix=rng.integers(0,len(a),(10000,len(a)));num=a[ix,0].sum(1);den=a[ix,1].sum(1);good=den>0
 bs[k]=dict(parent_macro_rate=float(np.mean(rates)) if rates else None,parents_with_visible_GT=len(rates),selected_unique_parents=len(ps),parent_macro_CI95=np.quantile(np.array(rates)[rng.integers(0,len(rates),(10000,len(rates)))].mean(1),[.025,.975]).tolist() if rates else None,parent_cluster_image_rate_CI95=np.quantile(num[good]/den[good],[.025,.975]).tolist() if good.any() else None)
sam['parent_macro_bootstrap']=bs;sam['actual_temporary_states_created']=sum(any(o['status']=='INITIALIZED' for o in r['objects'].values()) for r in rd(R/'.execution/sam2_raw_initializations.json'));sam['mask_failure_contract']='wrong_object flag is non-target task-actor dominance; background-dominated masks are separately identified as contamination/IoU failure, never success'
wr('sam2_initialization_metrics.json',sam)
lat=rd(R/'grounding_latency_vram.json');fw=[r for r in lat['sam2'] if r['valid_prompts']]
lat['SAM2_actual_forward_summary']=dict(camera_frames_with_forward=len(fw),box_prompt_forward_count=sum(r['valid_prompts'] for r in fw),first_actual_forward=fw[0] if fw else None,warm_forward_samples=max(0,len(fw)-1),timing_annotation='first actual nonempty initialization is cold; subsequent nonempty initializations are warm; no added warmup',co_resident_Qwen_additional_requests=0)
wr('grounding_latency_vram.json',lat)
# Keep small panel and include the actual initialized frame. Figure selection is posthoc display only.
selected=['P','N','C1']+[next(cid for cid in mf['protocol']['sam2_ids'] if cm[cid]['camera']==cam and cm[cid]['frame_kind']==f) for cam in ['base_camera','hand_camera'] for f in ['F0','F1']]
sr={s['id']:s for s in rd(R/'.execution/sam2_raw_initializations.json')}
for cid,s in sr.items():
 if any(o['status']=='INITIALIZED' for o in s['objects'].values()) and cid not in selected and len(selected)<8:selected.append(cid)
assert len(selected)<=8
W=430;H=465;scale=3;canvas=Image.new('RGB',(1720,1050),(245,245,245));d=ImageDraw.Draw(canvas)
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
from PIL import ImageFont
small=ImageFont.truetype(font,13);legend=ImageFont.truetype(font,16)
for i,cid in enumerate(selected):
 c=cm[cid];arr=np.asarray(Image.open(c['image_path']).convert('RGB')).copy();masks={}
 if cid in sr:
  with np.load(sr[cid]['masks_path']) as z:masks={k:z[k].copy() for k in z.files}
 for k,mask in masks.items():arr[mask]=(arr[mask]*.65+np.array([255,0,255] if k=='peg' else [0,200,220])*.35).astype(np.uint8)
 im=Image.fromarray(arr).resize((384,384),Image.Resampling.NEAREST);dr=ImageDraw.Draw(im);info=m['GT_support_by_image'][cid]
 for k,color in [('peg',(220,40,60)),('box',(30,100,240))]:
  b=pm[cid]['parsed']['objects'][k]['pixel'];diag=pm[cid]['diagnostic_only'];db=diag['objects'][k]['pixel'] if diag else None
  if db and not b:
   x1,y1,x2,y2=[z*scale for z in db]
   for x in range(int(x1),int(x2),10):dr.line((x,y1,min(x+5,x2),y1),fill=color,width=2);dr.line((x,y2,min(x+5,x2),y2),fill=color,width=2)
   for y in range(int(y1),int(y2),10):dr.line((x1,y,x1,min(y+5,y2)),fill=color,width=2);dr.line((x2,y,x2,min(y+5,y2)),fill=color,width=2)
  if b:dr.rectangle([x*scale for x in b],outline=color,width=3)
  if info['status']=='EXACT_GT' and info['bboxes'][k]:dr.rectangle([x*scale for x in info['bboxes'][k]],outline=(0,210,80) if k=='peg' else (255,180,0),width=2)
 x=(i%4)*W;y=(i//4)*H;canvas.paste(im,(x+15,y+50));d.text((x+15,y+5),f"{cid}\nt{c['timestep']} {c['camera']} / {info['status']}",font=small,fill='black');d.text((x+15,y+439),f"Strict JSON: {pm[cid]['parsed']['json_valid']} | SAM2 mask: {bool(masks)}",font=small,fill='black')
d.multiline_text((20,945),'Qwen PEG: red | Qwen BOX: blue | GT PEG: green | GT BOX: orange\nSolid: strict accepted box | Dashed: diagnostic ONLY, rejected | Magenta/cyan overlays: actual SAM2 initial masks\nP/N/C1 are technical checks. Other panels are a small posthoc display from the fixed evaluation manifest.',font=legend,fill='black',spacing=8)
canvas.save(R/'full_image_grounding_comparison.png')
assert sha(R/'full_image_grounding_raw_responses.json')==v['raw_response_SHA_before_GT'] and sha(R/'full_image_grounding_predictions.csv')==v['predictions_SHA_before_GT']
v['supplemental_readback']=dict(image_level_denominators=True,SAM2_parent_bootstrap=True,actual_forward_cold_annotation=True,figure_includes_actual_initialized_masks=True,no_new_inference=True)
v['execution_sources']={p.name:sha(p) for p in (R/'.execution').glob('*.py')};wr('verification.json',v)
index=rd(R/'artifact_index.json')
for n in index['required']:index['required'][n]=dict(path=n,sha256=sha(R/n),bytes=(R/n).stat().st_size)
index['supporting']={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in (R/'.execution').rglob('*') if p.is_file() and p.suffix in ['.py','.json','.csv','.png','.npz']}
wr('artifact_index.json',index)
print('SUPPLEMENT_VERIFIED',json.dumps(m['image_level_format']),json.dumps(bs),flush=True)
