from common import *
import collections,subprocess,time,ast
import numpy as np
from PIL import Image,ImageDraw
from utils.experiment_logger import ExperimentLogger

parent=rd(P/'verification.json');assert parent['stage']=='INFERENCE_AND_EVALUATION_FINISHED'
barrier=rd(P/'all_predictions_frozen.json');assert barrier['all_inference_processes_terminated'] and not barrier['GT_opened'] and not barrier['DINO_opened']
protocol=rd(P/'protocol.json');first=rd(P/'bf16/verification.json')
for n,h in first['execution_sources_before'].items(): assert sha(P/'.execution'/n)==h,n
for mode in VARIANTS:
 root=P/mode;v=rd(root/'verification.json');frozen=barrier['variants'][mode]
 assert v['model_loads']<=1
 if frozen['raw_sha256']:assert sha(root/'full_image_grounding_raw_responses.json')==frozen['raw_sha256']
 if frozen['predictions_sha256']:assert sha(root/'full_image_grounding_predictions.csv')==frozen['predictions_sha256']
 assert sha(root/'grounding_prompt.txt')==first['prompt_sha256']
 assert sha(root/'input_manifest.json')==v['manifest_sha256'] and sha(root/'model_identity.json')==v['identity_sha256']

# All input/weight/preservation checks are read-only. Saved DINO comparison is
# accessed only after the all-variant prediction barrier above.
protected=dict(first['protected_before']);protected.update(first['deferred_DINO_hashes_from_previous_verified_receipt'])
for mode in VARIANTS:protected.update(rd(P/mode/'verification.json').get('GT_reference_files',{}))
after={p:sha(p) for p in protected};assert after==protected
assert {n:sha(B/'model'/n) for n in first['model_hashes_before']}==first['model_hashes_before']
packages={p:subprocess.check_output([p,'-m','pip','list','--format=json'],text=True) for p in first['packages_before']};assert packages==first['packages_before']
mf=rd(P/'bf16/input_manifest.json');cases={c['id']:c for c in mf['cases']};parents=[p['parent_id'] for p in mf['parents']]
assert all(sha(c['image_path'])==c['image_sha256'] for c in mf['cases'])
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert not gpu.strip(),gpu
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
active=[s for s in processes if 'python' in s and any(t in s for t in [' -m graph.task10i_stage4a2.','/run_pipeline.py','/registered_runner.py'])];assert not active,active
base_met=rd(DETAIL/'grounding_evaluation_metrics.json');base_rows=list(csv.DictReader((DETAIL/'.execution/object_evaluation.csv').open()));base_by={(r['id'],r['object']):r for r in base_rows}
db=rd(OLD/'.execution/dino_matched_comparison.json')
reference=dict(model='saved detailed prompt, all-module NF4',output_root=str(DETAIL),prompt_sha256=sha(DETAIL/'grounding_prompt.txt'),metrics_sha256=sha(DETAIL/'grounding_evaluation_metrics.json'),raw_sha256=sha(DETAIL/'full_image_grounding_raw_responses.json'),inference_repeated=False)
assert reference['prompt_sha256']==first['prompt_sha256']
rng=np.random.default_rng(42)
def paired_summary(rows,baseline):
 out={}
 for obj in ['peg','box']:
  rr=[r for r in rows if r['phase']=='B' and r['object']==obj and r['visibility']=='EVALUABLE_VISIBLE'];pairs=[(r,baseline[r['id'],obj]) for r in rr]
  assert all(r['GT_bbox']==b['GT_bbox'] and r['visible_pixels']==b['visible_pixels'] for r,b in pairs)
  num=[];den=[];mac=[]
  for p in parents:
   pp=[(r,b) for r,b in pairs if r['parent_id']==p];n=sum(int(r['result']=='CORRECT')-int(b['result']=='CORRECT') for r,b in pp);d=len(pp);num.append(n);den.append(d)
   if d:mac.append(n/d)
  ix=rng.integers(0,len(parents),(10000,len(parents)));ns=np.array(num)[ix].sum(1);ds=np.array(den)[ix].sum(1);ok=ds>0
  mi=np.array(mac)[rng.integers(0,len(mac),(10000,len(mac)))].mean(1)
  out[obj]=dict(n=len(pairs),parents_with_support=len(mac),baseline_correct=sum(b['result']=='CORRECT' for r,b in pairs),new_correct=sum(r['result']=='CORRECT' for r,b in pairs),improved=sum(r['result']=='CORRECT' and b['result']!='CORRECT' for r,b in pairs),regressed=sum(r['result']!='CORRECT' and b['result']=='CORRECT' for r,b in pairs),mean_IoU_difference=float(np.mean([float(r['bbox_IoU'])-float(b['bbox_IoU']) for r,b in pairs])),parent_macro_recall_difference=float(np.mean(mac)),parent_macro_difference_CI95=np.quantile(mi,[.025,.975]).tolist(),parent_cluster_image_rate_difference_CI95=np.quantile(ns[ok]/ds[ok],[.025,.975]).tolist())
 return out

summaries={};loaded={};all_fail=[];rows_by={}
for mode in VARIANTS:
 root=P/mode;v=rd(root/'verification.json');execution=parent['results'][mode]
 if execution.get('evaluation_returncode')!=0:
  summaries[mode]=dict(status='GROUNDING_RUNTIME_BLOCKED',execution=execution,error=v.get('error'),model_loads=v['model_loads']);continue
 assert v['stage']=='EVALUATION_COMPLETE'
 raw=rd(root/'full_image_grounding_raw_responses.json')['responses'];rows=list(csv.DictReader((root/'full_image_grounding_predictions.csv').open()))
 assert len(raw)==99 and len(rows)==198 and v['model_loads']==1
 assert sha(root/'full_image_grounding_raw_responses.json')==v['raw_response_SHA_before_GT']==v['raw_response_SHA_after_GT']
 assert sha(root/'full_image_grounding_predictions.csv')==v['predictions_SHA_before_GT']==v['predictions_SHA_after_GT']
 for r in raw:
  assert parse(r['raw_text'])==r['parsed']
  assert r['timing']['input_tokens']==1292 and r['timing']['image_grid_thw']==[[1,16,16]]
  for obj,p in r['parsed']['objects'].items():
   cr=next(x for x in rows if x['id']==r['id'] and x['object']==obj);assert cr['status']==p['status'] and json.loads(cr['pixel'])==p['pixel']
 audit=rd(root/'.execution/loaded_model_audit.json');loaded[mode]=audit['summary'];assert audit['summary']['vision_quantized_layers']==0 and audit['summary']['vision_parameter_dtypes']==['torch.bfloat16']
 met=rd(root/'grounding_evaluation_metrics.json');sam=rd(root/'sam2_initialization_metrics.json');lat=rd(root/'grounding_latency_vram.json');object_rows=list(csv.DictReader((root/'.execution/object_evaluation.csv').open()));rows_by[mode]=object_rows
 assert met['GT_support_by_image']==base_met['GT_support_by_image']
 geometry={}
 for obj in ['peg','box']:
  vals=[]
  for r in object_rows:
   if r['phase']!='B' or r['object']!=obj or r['visibility']!='EVALUABLE_VISIBLE':continue
   a=json.loads(r['pixel']) if r['pixel'] else None;gt=json.loads(r['GT_bbox']);value=0.
   if a is not None:
    intersection=max(0.,min(a[2],gt[2])-max(a[0],gt[0]))*max(0.,min(a[3],gt[3])-max(a[1],gt[1]))
    value=intersection/((a[2]-a[0])*(a[3]-a[1])+(gt[2]-gt[0])*(gt[3]-gt[1])-intersection)
   assert math.isclose(value,float(r['bbox_IoU']),abs_tol=1e-12)
   assert (value>=.5)==(r['result']=='CORRECT')
   vals.append(value)
  z=met['primary']['image_object_level']['objects'][obj]
  assert len(vals)==z['recall_IoU05']['denominator'] and sum(x>=.5 for x in vals)==z['recall_IoU05']['numerator']
  assert math.isclose(float(np.mean(vals)),z['bbox_IoU_including_misses']['mean'],abs_tol=1e-12)
  geometry[obj]=dict(rows=len(vals),successes=sum(x>=.5 for x in vals),mean_IoU=float(np.mean(vals)))
 wr(root/'.execution/independent_csv_geometry_check.json',dict(status='PASS',recomputed_from='saved pixel and GT boxes; fresh arithmetic after prediction barrier',objects=geometry))
 b=[r for r in raw if cases[r['id']]['phase']=='B'];met['image_level_format']=dict(bare_JSON=dict(numerator=sum(r['parsed']['strict_json_valid'] for r in b),denominator=96),accepted_JSON=dict(numerator=sum(r['parsed']['json_valid'] for r in b),denominator=96),both_valid_boxes=sum(all(o['status']=='VALID_BOX' for o in r['parsed']['objects'].values()) for r in b),null_object_slots=sum(o['status']=='NULL' for r in b for o in r['parsed']['objects'].values()),wrapper_counts=dict(collections.Counter(r['parsed']['wrapper'] for r in b)))
 dup_ids=[]
 for r in b:
  duplicates=[]
  def detect_pairs(xs):
   seen=set()
   for k,_ in xs:
    if k in seen:duplicates.append(k)
    seen.add(k)
   return dict(xs)
  text=r['raw_text'].strip();f=re.fullmatch(r'```(?:json)?\s*\n(.*)\n```',text,re.S)
  try:json.loads(f.group(1) if f else text,object_pairs_hook=detect_pairs)
  except Exception:pass
  if duplicates:dup_ids.append(r['id'])
 met['output_diagnostics']=dict(duplicate_key_image_ids=dup_ids,invalid_JSON_image_ids=[r['id'] for r in b if not r['parsed']['json_valid']],token_limit_image_ids=[r['id'] for r in b if r['timing']['output_tokens']>=128],invalid_coordinate_object_slots=sum(o['status']=='INVALID_BOX' for r in b for o in r['parsed']['objects'].values()))
 sr=rd(root/'.execution/sam2_raw_initializations.json') if (root/'.execution/sam2_raw_initializations.json').exists() else []
 for r in sr:
  assert sha(r['masks_path'])==r['masks_sha256']
  with np.load(r['masks_path']) as z:assert set(z.files)=={k for k,o in r['objects'].items() if o['status']=='INITIALIZED'} and all(z[k].shape==(128,128) for k in z.files)
 if v.get('sam2_raw_SHA_before_GT'): assert sha(root/'.execution/sam2_raw_initializations.json')==v['sam2_raw_SHA_before_GT']
 sam['runtime_status']='COMPLETE' if len(sr)==16 and execution['returncode']==0 else 'RESOURCE_OR_RUNTIME_BLOCKED'
 sam['actual_temporary_states_created']=sum(any(o['status']=='INITIALIZED' for o in r['objects'].values()) for r in sr)
 samrows=list(csv.DictReader((root/'sam2_initialization_cases.csv').open()));sam['parent_macro_bootstrap']={}
 for obj in ['peg','box']:
  rr=[r for r in samrows if r['object']==obj and r['visibility']=='EVALUABLE_VISIBLE'];ps=sorted({r['parent_id'] for r in rr});rates=[np.mean([r['initialization_success']=='True' for r in rr if r['parent_id']==p]) for p in ps]
  sam['parent_macro_bootstrap'][obj]=dict(parents_with_GT=len(ps),parent_macro_rate=float(np.mean(rates)),parent_macro_CI95=np.quantile(np.array(rates)[rng.integers(0,len(ps),(10000,len(ps)))].mean(1),[.025,.975]).tolist())
 fw=[r for r in lat['sam2'] if r['valid_prompts']];lat['SAM2_forward_summary']=dict(frames=len(fw),box_prompts=sum(r['valid_prompts'] for r in fw),cold_first_actual_frame=fw[0] if fw else None,warm_frame_wall_ms=[r['wall_ms'] for r in fw[1:]],warm_peak_allocated_bytes=max((r['memory']['peak_allocated'] for r in fw[1:]),default=None))
 lat['output_token_mean']=float(np.mean([r['timing']['output_tokens'] for r in raw]));lat['inference_log_bytes']=(root/'.execution/infer.log').stat().st_size;wr(root/'grounding_latency_vram.json',lat)
 wr(root/'grounding_evaluation_metrics.json',met);wr(root/'sam2_initialization_metrics.json',sam)
 paired=paired_summary(object_rows,base_by)
 dcomp={}
 for obj in ['peg','box']:
  rr=[r for r in db['matched_object_rows'] if r['object']==obj and r['GT_status']=='EXACT_GT' and r['visible_pixels']>=16];nby={(r['id'],r['object']):r for r in object_rows}
  dcomp[obj]=dict(n=len(rr),DINO_correct=sum(r['DINO_consumer_IoU']>=.5 for r in rr),Qwen_correct=sum(nby[r['id'],obj]['result']=='CORRECT' for r in rr),DINO_mean_IoU=float(np.mean([r['DINO_consumer_IoU'] for r in rr])),Qwen_mean_IoU=float(np.mean([float(nby[r['id'],obj]['bbox_IoU']) for r in rr])))
 summaries[mode]=dict(status=met['final_status'],format=met['image_level_format'],output_diagnostics=met['output_diagnostics'],primary=met['primary'],phase_A=met['phase_A'],DINO_replacement_decision=met['DINO_replacement_decision'],paired_vs_saved_all_NF4=paired,SAM2=sam,latency_vram=lat,actual_precision=audit['summary'],matched_DINO=dict(status='MATCHED_DINO_BASELINE_INSUFFICIENT',images=12,parents=6,only_t0=True,objects=dcomp),execution=execution)
 for row in csv.DictReader((root/'grounding_failure_cases.csv').open()):all_fail.append(dict(variant=mode,**row))
 v.update(stage='COMPLETE_VERIFIED',completion='VERIFIED',all_protected_SHA_unchanged=True,environments_unchanged=True,model_processor_unchanged=True,raw_CSV_readback_pass=True,all_three_prediction_barrier_pass=True,final_GPU_processes=gpu,original_runners_active=active,DINO_outputs_opened=True,new_DINO_forwards=0);wr(root/'verification.json',v)

between={}
if 'bf16' in rows_by:
 bfby={(r['id'],r['object']):r for r in rows_by['bf16']}
 for mode in VARIANTS[1:]:
  if mode in rows_by:between[mode+'_minus_bf16']=paired_summary(rows_by[mode],bfby)
comparison=dict(protocol=protocol,saved_reference=reference,variants=summaries,paired_between_new_variants=between,GT_support=dict(images=56,attempted_images_per_variant=96,physical_parents_with_GT=15,selected_parents=24,visible_PEG=56,visible_BOX=54,absent_BOX=2,tiny_visible=0,GT_unsupported_images=40,SAM2_GT_images=12,SAM2_GT_parents=9),bootstrap=dict(unit='physical parent',replicates=10000,seed=42),interpretation='Only precision varies among the three new variants. Matched retest, not fresh independent generalization; all-module NF4 is a saved reference, no baseline rerun; matched DINO baseline insufficient.')
wr(P/'precision_comparison_metrics.json',comparison)
if all_fail:csvwr(P/'precision_failure_cases.csv',all_fail)

def rate(x):
 n=x.get('numerator',0);d=x.get('denominator',0);return f'{n}/{d} ({n/d:.2%})' if d else 'INSUFFICIENT_SUPPORT (0)'
def num(x,d=4):return f'{x:.{d}f}' if x is not None else '미지원'
def gib(x):return num(x/2**30,3) if x is not None else '미지원'
names=dict(bf16='전체 BF16',vision_bf16_language_nf4='Vision·Merger BF16 / 언어 NF4',vision_bf16_language_int8='Vision·Merger BF16 / 언어 INT8')
flat=[]
for mode,s in summaries.items():
 if 'primary' not in s:flat.append(dict(variant=mode,status=s['status']));continue
 o=s['primary']['image_object_level'];l=s['latency_vram'];sm=s['SAM2']
 flat.append(dict(variant=mode,status=s['status'],PEG_recall=o['objects']['peg']['recall_IoU05']['rate'],BOX_recall=o['objects']['box']['recall_IoU05']['rate'],PEG_mean_IoU=o['objects']['peg']['bbox_IoU_including_misses']['mean'],BOX_mean_IoU=o['objects']['box']['bbox_IoU_including_misses']['mean'],accepted_JSON=s['format']['accepted_JSON']['numerator']/96,SAM2_PEG_IoU08=sm['objects']['peg']['end_to_end_IoU08_success']['rate'],SAM2_BOX_IoU08=sm['objects']['box']['end_to_end_IoU08_success']['rate'],warm_E2E_mean_ms=l['summary']['warm_end_to_end_ms']['mean'],warm_E2E_p95_ms=l['summary']['warm_end_to_end_ms']['p95'],Qwen_peak_GiB=l['summary']['peak_allocated_bytes']/2**30))
csvwr(P/'precision_comparison.csv',flat,fields=sorted({k for r in flat for k in r}))

# A compact, fixed comparison: one of the same preselected SAM2 cases in each
# camera/time condition, never selected by prediction accuracy.
selected=[next(cid for cid in mf['protocol']['sam2_ids'] if cases[cid]['camera']==cam and cases[cid]['frame_kind']==t) for cam in ['base_camera','hand_camera'] for t in ['F0','F1']]
W=315;H=325;fig=Image.new('RGB',(W*4,H*3+100),(245,245,245));d=ImageDraw.Draw(fig)
for j,mode in enumerate(VARIANTS):
 root=P/mode
 if 'primary' not in summaries[mode]:d.text((10,j*H+10),mode+' RUNTIME BLOCKED',fill='black');continue
 preds={r['id']:r for r in rd(root/'full_image_grounding_raw_responses.json')['responses']};sammap={r['id']:r for r in rd(root/'.execution/sam2_raw_initializations.json')} if (root/'.execution/sam2_raw_initializations.json').exists() else {}
 for i,cid in enumerate(selected):
  c=cases[cid];arr=np.asarray(Image.open(c['image_path']).convert('RGB')).copy()
  if cid in sammap:
   with np.load(sammap[cid]['masks_path']) as z:
    for obj in z.files:
     mask=z[obj];arr[mask]=(arr[mask]*.6+np.array([255,0,255] if obj=='peg' else [0,200,220])*.4).astype(np.uint8)
  im=Image.fromarray(arr).resize((256,256),Image.Resampling.NEAREST);dr=ImageDraw.Draw(im);info=base_met['GT_support_by_image'][cid]
  for obj,color,gtcolor in [('peg',(220,40,60),(0,210,80)),('box',(30,100,240),(255,180,0))]:
   b=preds[cid]['parsed']['objects'][obj]['pixel']
   if b:dr.rectangle([x*2 for x in b],outline=color,width=2)
   if info['status']=='EXACT_GT' and info['bboxes'][obj]:dr.rectangle([x*2 for x in info['bboxes'][obj]],outline=gtcolor,width=1)
  x=i*W;y=j*H;fig.paste(im,(x+10,y+52));d.text((x+10,y+5),f'{mode}\n{cid}\n{info["status"]}',fill='black')
d.multiline_text((10,H*3+5),'Predicted PEG: red; BOX: blue. GT PEG: green; BOX: orange. SAM2: magenta/cyan overlay only when initialized.\nColumns: Main F0 / Main F1 / Wrist F0 / Wrist F1. Fixed preselected cases; same original RGB for each row.\nAll Qwen predictions frozen before GT/DINO evaluation. No detector proposals, crops, markers, or GT inputs.',fill='black',spacing=7);fig.save(P/'precision_grounding_comparison.png')

# Report construction follows below; all scientific values come from receipts.
report_name='10i_stage4a2_qwen35_precision_comparison.md'
lines=['# Task 10-I Stage 4A-2 — Qwen3.5-9B 세 가지 정밀도 비교','',
'2026-11-02. 사용자가 승인한 전체 BF16 / Vision·Merger BF16+언어 NF4 / Vision·Merger BF16+언어 INT8 구성을 기존 방식으로 실제 측정했다. Production과 기존 `PAUSED_SCOPE_REVIEW`를 유지했다.','',
'## 1. 비교가 답하는 질문','',
'이번 비교는 같은 Qwen3.5-9B의 기능적 역할 Grounding 실패가 가중치 양자화 정밀도를 바꾸면 얼마나 달라지는지 측정한다. 세 구성 사이에서 프롬프트·출력 계약·Parser·이미지·해상도·Generation·SAM2 선정은 고정했다. 이전에 보존한 전체 NF4 실행도 같은 상세 Prompt의 참고 기준으로 사용했으며 다시 실행하지 않았다.','',
'**이미 실패 분석에 사용한 같은 24개 Train Parent의 RETEST다.** 독립적인 새 일반화 성능으로 해석하지 않는다. 원래 짧은 기능적 역할 Prompt의 좌표 진단과 이번 세 구성의 점수를 직접 비교해 양자화 효과라고 부르지 않는다.','',
'## 2. 모델·Prompt·실행 계약','',
'| 항목 | 고정값 |','|---|---|',
'| 모델 /Revision |Qwen/Qwen3.5-9B /c202236235762e1c871ad0ccb60c8ee5ba337b9a|',
'| 입력 |원본128×128 전체 RGB 한 장+직전 상세 Task Prompt|',
f"| Prompt |889단어, 이미지 포함1292 Input Token; SHA256 `{first['prompt_sha256']}`|",
'| Processor |같은 AutoProcessor 소스, min/max pixels65536; 내부256×256, grid[1,16,16], 이미지64 Token|',
'| 실행 |cuda:0, SDPA, Batch1, Greedy, enable_thinking=False, max_new_tokens=128, use_cache=True|',
'| 좌표 |원본 전체 이미지에 대한0–1000 xyxy; Pixel=normalized×128/1000; Clipping 없음|',
'| 출력 Parser |직전과 동일한 PEG/BOX bbox_2d 배열; Bare JSON 또는 완전한 단일 JSON Fence; 중복 Key/설명문/부분 추출 거부|',
'| 평가 |IoU≥.5 Grounding 성공, 가시 Pixel≥16; Tiny1–15 Pixel 별도; 미검출=IoU0|','',
f"환경: `{json.dumps(rd(P/'bf16/model_identity.json')['runtime'])}`. 기존 원본 Checkpoint에서 각 구성별로 한 번 로드했다. 양자화된 값을 BF16으로 재캐스팅하지 않았다. 새 Weight/Processor/모델을 다운로드하지 않았다.",'',
'| 구성 | Vision·Merger | 언어 Linear | 실제 4-bit /8-bit Linear |','|---|---|---:|---:|']
for mode in VARIANTS:
 if mode not in loaded:lines.append(f'|{names[mode]}|로드 실패|미지원|미지원|');continue
 a=loaded[mode];lines.append(f"|{names[mode]}|BF16, 양자화0개|{'BF16' if mode=='bf16' else 'NF4, Double Quant ON' if mode.endswith('nf4') else 'LLM.int8, threshold6'}|{a['linear4bit_layers']} /{a['linear8bit_layers']}|")
lines += ['', '설치된 Transformers의4-bit/8-bit 변환 코드를 직접 확인하고, 사용자 제외 목록을 설정할 때 사라질 수 있는 기본 `lm_head` 제외도 보존했다. 혼합 구성의 제외 목록은 `lm_head`, `model.visual`이다. Meta 모델 사전 검사와 실제 로드 후 모듈 검사에서 Vision Linear110개와 내부 Merger가 BF16이며 언어 Linear248개만 해당 양자화 타입임을 검증했다. `loaded_model_audit.json`에 모듈별 실제 Type/Dtype/Device를 저장했다.','',
'Checkpoint Header 전체의 BF16 환산 용량은17.98GiB였고 실행 아키텍처의 가중치는 약17.53GiB다. CPU Meta/Checkpoint Key 대조에서 추가15개 Key가 모두 `mtp.*`이며 필수 Key 누락과 Shape 불일치는0개임을 확인했다. 기존과 같은 ConditionalGeneration 클래스·일반 Generation 경로를 유지했고 MTP를 새로 활성화하지 않았다. 이는 추가 Weight Load나 GPU Forward가 없는 Header/Meta 검사다. [대조 Receipt](.execution/checkpoint_accounting.json).','',
'언어 INT8 구성은 로드 Dtype과 Vision 가중치를 BF16으로 유지하지만, 설치된 bitsandbytes의 `MatMul8bitLt`가 양자화 과정에서 BF16 입력을 FP16으로 Cast한다는 기본 경고를 출력했다. 실제8-bit 커널의 동작을 그대로 유지하고 로그에 보존했다. 전체 연산이 BF16이라는 의미로 해석하지 않는다.','',
'## 3. 대상·동결·보존','',
'각 구성에서 기술 확인 P(0355/t0 Main), N(0355/t30 Main), C1(0645/t0 Main)3장 후 같은24개 Parent×F0/F1×Main/Wrist96장을 실행한다. F0=t0, F1은23 Parent에서t80이고0134에서 사전 규칙의t78이다. 신규 Frame/Parent 선택, Final/Test 사용, Simulator 실행은 없다.','',
'기존 입력 Manifest와 SAM2 선정16개를 재사용했다. SAM2는 Main/t0, Main/F1, Wrist/t0, Wrist/F1 각각4개이며 전체12개 물리 Parent다. 결과를 보고 성공 Frame만 고르지 않았다.','',
'GT 지원도 고정했다: EXACT_GT56/96 이미지,15개 Parent; Peg56개와 Box54개가16 Pixel 이상, Box2개는 완전 비가시, Tiny0개,40개 이미지는 GT 미지원이다. SAM2 선정 중 GT 지원은12개 Camera-Frame /9개 Parent이며 각 Object 분모12다. GT 미지원 사례를 실패나 성공으로 임의 대체하지 않았다.','',
'**세 구성의 새 예측에 대한 GT 채점과 보존 DINO 비교는 모든 추론 프로세스가 종료하고 원시 응답·예측 CSV의 SHA를 저장한 뒤에 실행했다.** 추론 프로세스에는 GT/DINO 의미 정보를 읽는 경로가 없다. Qwen에는 RGB1장+Prompt만 전달했다. DINO Proposal/Confidence/Label, SAM2 Mask, GT Actor/Position/Pose, Crop/Point/Rectangle Marker는 입력하지 않았다. [전체 동결 Barrier](all_predictions_frozen.json).','',
'이 Barrier는 이번 추론·채점 Pipeline의 경계이며 연구자가 과거 GT를 전혀 보지 않았다는 뜻은 아니다. 이전 결과와 GT는 이미 알려져 있고, 준비 중 보존된 전체 NF4의 예전 CSV로 좌표 산술도 재확인했다. 이 확인에 새 구성의 예측을 사용하지 않았으며 프롬프트·구성·선정은 바꾸지 않았다. 독립 Blind 평가와 구분한다.','',
'## 4. 형식·발견·위치 성능','',
'이전 참고 기준도 같은 상세 Prompt와 같은 Parser이므로 아래 NF4 비교는 Parser 정책 변경을 포함하지 않는다. Recall은 평가 가능한 가시 물체에 대한 IoU≥.5 성공률이다.','',
'| 구성 | Bare JSON | 채택 JSON | Peg Recall | Box Recall | Peg/Box Mean IoU |','|---|---:|---:|---:|---:|---:|']
bo=base_met['primary']['image_object_level'];bf=base_met['image_level_format']
lines.append(f"|보존 전체 NF4|{rate(bf['bare_native_JSON'])}|{rate(bf['accepted_JSON_with_frozen_wrapper_policy'])}|{rate(bo['objects']['peg']['recall_IoU05'])}|{rate(bo['objects']['box']['recall_IoU05'])}|{num(bo['objects']['peg']['bbox_IoU_including_misses']['mean'])} /{num(bo['objects']['box']['bbox_IoU_including_misses']['mean'])}|")
for mode,s in summaries.items():
 if 'primary' not in s:lines.append(f"|{names[mode]}|{s['status']}|—|—|—|—|");continue
 o=s['primary']['image_object_level'];pg=o['objects']['peg'];bx=o['objects']['box']
 lines.append(f"|{names[mode]}|{rate(s['format']['bare_JSON'])}|{rate(s['format']['accepted_JSON'])}|{rate(pg['recall_IoU05'])}|{rate(bx['recall_IoU05'])}|{num(pg['bbox_IoU_including_misses']['mean'])} /{num(bx['bbox_IoU_including_misses']['mean'])}|")
for mode,s in summaries.items():
 if 'primary' not in s:continue
 o=s['primary']['image_object_level'];pg=o['objects']['peg'];bx=o['objects']['box'];lines += ['', f'### {names[mode]}', '',f"판정: `{s['status']}`. 두 물체 동시 정확 {rate(o['scene']['both_correct'])}; 하나 이상의 잘못된 채택 Box {rate(o['scene']['any_incorrect_emitted'])}.",'',
 '| 지표 | Peg | Box |','|---|---:|---:|',
 f"|유효 Box 출력, 가시 물체 기준|{rate(pg['valid_box_emission_recall'])}|{rate(bx['valid_box_emission_recall'])}|",
 f"|Missing Detection, IoU≥.5 미달 포함|{rate(pg['missing_detection_rate'])}|{rate(bx['missing_detection_rate'])}|",
 f"|False Discovery, 유효 출력 중 실패|{rate(pg['false_discovery_fraction'])}|{rate(bx['false_discovery_fraction'])}|",
 f"|완전 비가시 시 Hallucinated Box|{rate(pg['hallucinated_box_absent_FPR'])}|{rate(bx['hallucinated_box_absent_FPR'])}|",
 f"|의미 혼동|{json.dumps(pg['semantic_confusion'])}|{json.dumps(bx['semantic_confusion'])}|",
 f"|IoU≥.3|{rate(pg['IoU_thresholds']['0.3'])}|{rate(bx['IoU_thresholds']['0.3'])}|",
 f"|IoU≥.7|{rate(pg['IoU_thresholds']['0.7'])}|{rate(bx['IoU_thresholds']['0.7'])}|",
 f"|IoU Median, 미검출0 포함|{num(pg['bbox_IoU_including_misses']['median'])}|{num(bx['bbox_IoU_including_misses']['median'])}|",
 f"|IoU p05~p95|{num(pg['bbox_IoU_including_misses']['p05'])}~{num(pg['bbox_IoU_including_misses']['p95'])}|{num(bx['bbox_IoU_including_misses']['p05'])}~{num(bx['bbox_IoU_including_misses']['p95'])}|",
 f"|Center Error Mean, 유효 출력만|{num(pg['center_error_px_emitted_only']['mean'])}px|{num(bx['center_error_px_emitted_only']['mean'])}px|",'',
 f"좌표 계약 {rate(o['coordinate_contract_validity'])}; Null Object Slot {s['format']['null_object_slots']}/192. JSON 분모96은 이미지, 좌표/Null 분모192는 Object Slot이며 서로 독립 Parent 수가 아니다.",'',
 '| Parent-macro | 평균 | Macro95% CI | 지원 Parent |','|---|---:|---:|---:|']
 for label,key in [('Peg Recall','objects/peg/recall_IoU05'),('Box Recall','objects/box/recall_IoU05'),('좌표 계약','coordinate_contract_validity')]:
  z=s['primary']['parent_macro'][key];ci=z['parent_macro_CI95'];lines.append(f"|{label}|{z['parent_macro_rate']:.2%}|{ci[0]:.2%}~{ci[1]:.2%}|{z['parents_with_support']}|")
 lines += ['', '| 보존 전체 NF4 대비 | 실패→성공 | 성공→실패 | Parent-macro Recall 차이 | Image-rate 차이의 Parent-cluster95% CI |','|---|---:|---:|---:|---:|']
 for obj,z in s['paired_vs_saved_all_NF4'].items():
  ci=z['parent_cluster_image_rate_difference_CI95'];lines.append(f"|{obj}|{z['improved']}|{z['regressed']}|{z['parent_macro_recall_difference']:.2%}|{ci[0]:.2%}~{ci[1]:.2%}|")
lines += ['', 'Parent를 독립 단위로10,000회 Bootstrap했다(seed42). 같은 Parent의 두 시점·두 Camera를 네 개 Parent로 세지 않았다. BF16을 기준으로 한 두 새 혼합 구성의 Paired 차이와 CI도 [전체 비교 JSON](precision_comparison_metrics.json)에 저장했다. 소표본의 퇴화 구간은 희귀 오류 안전성 보증이 아니다.','',
'## 5. Camera·시점별 및 Phase A','',
'| 구성 /구분 | Peg Recall | Box Recall |','|---|---:|---:|']
for mode,s in summaries.items():
 if 'primary' not in s:continue
 for group in ['by_camera','by_time']:
  for key,z in s['primary'][group].items():lines.append(f"|{names[mode]} /{key}|{rate(z['objects']['peg']['recall_IoU05'])}|{rate(z['objects']['box']['recall_IoU05'])}|")
lines += ['', '| 구성 /기술 사례 | Peg IoU | Box IoU |','|---|---:|---:|']
for mode,s in summaries.items():
 if 'primary' not in s:continue
 for cid in ['P','N','C1']:
  z={r['object']:r for r in s['phase_A'] if r['id']==cid};lines.append(f"|{names[mode]} /{cid}|{num(z['peg']['bbox_IoU'])}|{num(z['box']['bbox_IoU'])}|")
lines += ['', '기술 확인3장은96장 재평가 성능에 합산하지 않았다. 각 Camera×시점의 실제 분모와 Parent별 값은 구성별 JSON/CSV에 보존했다.','',
'## 6. Qwen Box로 실제 SAM2 초기화','',
'기존 SAM2.1 Hiera Small Checkpoint와 단일 JPEG 프레임 초기화 경로를 유지했다. Qwen이 채택된 유효 Box를 출력한 Object만 Temporary State에 제공했다. GT/DINO Box 대체, 장기 Tracking, Production Memory Commit은 없다. 잘못된 Box와 Null/형식 실패도 선정16개에서 제외하지 않았다.','',
'| 구성 | 실제 State /선정 Frame | Peg/Box 초기화 수 | Peg IoU≥.8 | Box IoU≥.8 |','|---|---:|---:|---:|---:|']
for mode,s in summaries.items():
 if 'primary' not in s:continue
 sm=s['SAM2'];o=sm['objects'];lines.append(f"|{names[mode]}|{sm['actual_temporary_states_created']}/16|{o['peg']['initialized']['numerator']} /{o['box']['initialized']['numerator']}|{rate(o['peg']['end_to_end_IoU08_success'])}|{rate(o['box']['end_to_end_IoU08_success'])}|")
lines += ['', '| 구성 /Object | 실제 초기화 Mask Mean IoU | 미초기화0 포함 Mean IoU | Precision /Recall | Wrong-object |','|---|---:|---:|---:|---:|']
for mode,s in summaries.items():
 if 'primary' not in s:continue
 for obj,z in s['SAM2']['objects'].items():lines.append(f"|{names[mode]} /{obj}|{num(z['mask_IoU_initialized_only']['mean'])}|{num(z['mask_IoU_including_noninitialization']['mean'])}|{num(z['precision']['mean'])} /{num(z['recall']['mean'])}|{rate(z['wrong_object'])}|")
lines += ['', 'IoU≥.8 성공의 분모는 Object별 GT 지원12개다. 조건부 초기화 Mask 평균은 유효 Box가 있던 표본만 포함하므로 End-to-end 성공률과 구분한다. 원본 Actor Pixel 구성과 Robot/Background 오염은 각 `sam2_initialization_cases.csv`에 있다. GT 지원9 Parent의 Macro Bootstrap도 저장했다.','',
'## 7. 실제 Latency·VRAM','',
'| 구성 | Load ms | 로드 후 Allocated GiB | Qwen Peak GiB | Warm E2E Mean /Median /p95 ms | Warm n | 출력 Token 평균 |','|---|---:|---:|---:|---:|---:|---:|']
for mode,s in summaries.items():
 if 'primary' not in s:continue
 l=s['latency_vram'];z=l['summary']['warm_end_to_end_ms'];lines.append(f"|{names[mode]}|{num(l['load_ms'],2)}|{gib(l['after_load']['allocated'])}|{gib(l['summary']['peak_allocated_bytes'])}|{num(z['mean'],2)} /{num(z['median'],2)} /{num(z['p95'],2)}|{l['summary']['warm_requests']}|{num(l['output_token_mean'],2)}|")
lines += ['', '| 구성 | Cold 첫 요청 ms | CPU Peak RSS GiB | Qwen+SAM2 로드 GiB | SAM2 Warm Peak GiB /Forward Frame |','|---|---:|---:|---:|---:|']
for mode,s in summaries.items():
 if 'primary' not in s:continue
 l=s['latency_vram'];co=l.get('co_residency',{}).get('loaded_memory',{});fw=l['SAM2_forward_summary'];lines.append(f"|{names[mode]}|{num(l['summary']['first_request']['end_to_end_ms'],2)}|{gib(l['CPU_process_peak_sampled_bytes'])}|{gib(co.get('allocated'))}|{gib(fw['warm_peak_allocated_bytes'])} /{fw['frames']}|")
lines += ['', '| 구성 | Warm Preprocess ms | Vision CUDA ms | Prefill-Vision 차이 근사 ms | Decode Forward 합 ms |','|---|---:|---:|---:|---:|']
for mode,s in summaries.items():
 if 'primary' not in s:continue
 z=s['latency_vram']['summary'];lines.append(f"|{names[mode]}|{num(z['warm_preprocessing_ms']['mean'],2)}|{num(z['warm_vision_ms']['mean'],2)}|{num(z['warm_prefill_excluding_vision_approx_ms']['mean'],2)}|{num(z['warm_decode_forward_ms']['mean'],2)}|")
lines += ['', '매 이미지 새로운 Vision Encoding과 Prefill을 실행했다. Cold는 모델을 로드한 뒤 첫 요청이고 Warm은 후속98개다. 별도 Warmup/Timing 요청은 없다. E2E는 PNG 읽기·Processor·H2D·Generation·Text Decode를 포함한다. Vision CUDA Event, Vision을 포함한 첫 Forward, Prefill 근사, Decode Forward 합, Peak Reserved 및 출력 Token 분포도 저장했다. 출력 길이 차이도 E2E 비용에 반영되므로 평균 Latency만으로 순수 Kernel 속도를 단정하지 않는다. 초기 Object Discovery나 필요한 Re-grounding의 비용이며20Hz 제어 루프를 요구하는 평가가 아니다.','',
'모델 로드는 사전 SHA 검증으로 File Cache가 Warm한 조건이며 다운로드/Processor 준비를 포함하지 않는다. CPU RSS는50ms 표본 Peak다. 같은 모델 객체를 구성 내99개 요청에 재사용했고 새 이미지마다 Cache는 새로 생성한다. 최적화 커널 추가 설치 없이 기존 Reference PyTorch DeltaNet/Conv 경로를 유지했다.','',
'INT8에서는 BF16→FP16 Cast 경고가 Forward마다 반복 출력됐다. 중간에 Logging 수준을 바꾸지 않았으므로 위 E2E는 이 기본 경고의 Host Logging/I/O 비용도 포함한다. 이 수치만으로 경고를 억제한 최적화8-bit 서빙의 속도를 단정하지 않는다. 구성별 실제 `inference_log_bytes`를 비용 JSON에 저장했다.','',
'## 8. 보존된 동일 Frame DINO 비교','',
'| 구성 /Object | 공통 n | DINO IoU≥.5 | Qwen IoU≥.5 |','|---|---:|---:|---:|']
for mode,s in summaries.items():
 if 'primary' not in s:continue
 for obj,z in s['matched_DINO']['objects'].items():lines.append(f"|{names[mode]} /{obj}|{z['n']}|{z['DINO_correct']}|{z['Qwen_correct']}|")
lines += ['', '`MATCHED_DINO_BASELINE_INSUFFICIENT`: 같은12개t0 이미지/6개 Parent의 보존 결과만 사용했다. 새로운 DINO Forward는0이다. 이 소수 공통 표본으로 전체 분포의 상대 우열이나 DINO 제거를 확정하지 않는다.','',
'## 9. 운영 기준·해석·실패 사례','',
'| 구성 | 판정 | Peg/Box Recall≥90% | 잘못된 Peg≤5% | 좌표 계약≥99% | SAM2 Peg/Box≥80% |','|---|---|---|---|---|']
for mode,s in summaries.items():
 if 'primary' not in s:lines.append(f"|{names[mode]}|{s['status']}|미측정|미측정|미측정|미측정|");continue
 o=s['primary']['image_object_level'];pg=o['objects']['peg'];bx=o['objects']['box'];sm=s['SAM2']['objects']
 checks=[pg['recall_IoU05']['rate']>=.9 and bx['recall_IoU05']['rate']>=.9,pg['wrong_grounding_fraction_all_evaluable']['rate']<=.05,o['coordinate_contract_validity']['rate']>=.99,sm['peg']['end_to_end_IoU08_success']['rate']>=.8 and sm['box']['end_to_end_IoU08_success']['rate']>=.8]
 lines.append(f"|{names[mode]}|{s['status']}|{'충족' if checks[0] else '미달'}|{'충족' if checks[1] else '미달'}|{'충족' if checks[2] else '미달'}|{'충족' if checks[3] else '미달'}|")
lines += ['', '운영 기준의 실제 분모는 위 표와 원시 Metric에 명시했다. 전체24 Parent GT, 희귀 오인식 확률, 비가시/Tiny 계층 및 실제 환경 일반화에는 `INSUFFICIENT_SUPPORT`가 남는다. 통계적 안전성이나 Closed-loop Success 보증이 아니다. Production DINO 삭제는 수행하지 않았다.','']
if 'primary' in summaries.get('vision_bf16_language_nf4',{}):
 z=summaries['vision_bf16_language_nf4']['paired_vs_saved_all_NF4'];lines += [f"언어 NF4를 유지하고 Vision·Merger만 BF16으로 바꾼 직접 비교에서 Peg 성공은{z['peg']['baseline_correct']}/{z['peg']['n']}→{z['peg']['new_correct']}/{z['peg']['n']}, Box는{z['box']['baseline_correct']}/{z['box']['n']}→{z['box']['new_correct']}/{z['box']['n']}였다. 이 변화와 Parent-cluster CI를 함께 해석해야 하며, 단일 숫자로 전체 VLM의 잠재력을 결론 내리지 않는다.",'']
if 'primary' in summaries.get('bf16',{}):
 z=summaries['bf16']['primary']['image_object_level']['objects']
 if z['peg']['recall_IoU05']['rate']<.9 or z['box']['recall_IoU05']['rate']<.9:lines += ['전체 BF16에서도 두 물체 Recall≥90% 운영 기준을 함께 충족하지 못했다. 따라서 이 고정 조건에서는 가중치 양자화를 제거하는 것만으로 Grounding 문제를 해결하지 못했다. 이는 다른 Prompt·입력 Token 예산·Thinking 조건까지 검증했다는 뜻은 아니다.','']
lines += [
'실패 원인을 분리할 때는 형식 실패·역할 혼동·위치 오차·SAM2 Mask 실패를 구분한다. [구성별 실패 CSV](precision_failure_cases.csv)와 [같은 장면 비교 그림](precision_grounding_comparison.png)에 관측을 남겼다. 아래 표는 전체 오류 원인이 아니라 Actor 지배 기준으로 확인된 의미 혼동과 출력 형식 상태다.','',
'| 구성 | Peg 의미 혼동 | Box 의미 혼동 | Null Slot | 중복 Key 응답 /Token 한도 도달 |','|---|---|---|---:|---:|']
for mode,s in summaries.items():
 if 'primary' not in s:continue
 o=s['primary']['image_object_level'];diag=s['output_diagnostics'];lines.append(f"|{names[mode]}|{json.dumps(o['objects']['peg']['semantic_confusion'])}|{json.dumps(o['objects']['box']['semantic_confusion'])}|{s['format']['null_object_slots']}/192|{len(diag['duplicate_key_image_ids'])} /{len(diag['token_limit_image_ids'])}|")
lines += ['', '정밀도 변경 뒤 성능이 달라진 것은 이 고정된 입력 조건의 관측이다. NF4 Vision 제외 비교는 Vision 정밀도의 영향을 가장 직접적으로 확인하며, 전체 BF16 및 INT8 구성은 언어 정밀도도 함께 다르다. 양자화 외에도 제한된 이미지 정보·역할 판단·Prompt·시각 Token 예산·Thinking 설정의 영향은 별도 검증하지 않았으므로 모든 실패를 특정 원인으로 확정하지 않는다. Prompt Sweep이나 새 모델 비교를 추가하지 않았다.','',
'## 10. 예산·보존·종료','',
f"전체 Model Load 시도 {sum(rd(P/m/'verification.json')['model_loads'] for m in VARIANTS)}회, 완료 Qwen 요청 {sum(z.get('grounding_requests',0) for z in parent['results'].values())}개 /상한297개다. 구성별 Runtime 상태·오류는 [실행 Receipt](execution_progress.json)에 보존했다. 자동 Retry·재샘플링·추가 Warmup은0이다.",'',
f"보호 파일 {len(protected)}개의 SHA, 모델·Processor {len(first['model_hashes_before'])}개 파일 SHA, 입력99장 SHA, 기존 환경의 Package 목록을 확인했다. 기존 모델·Dataset·GT·Processor·Stop Stub·이전 실험이 그대로다. 종료 시 GPU Compute Process0, 원래 Stage4A-2 Runner0을 확인했다. 모든 실제 CUDA 실행은58576054 RTX3090이며 macOS MPS/Simulator/Renderer는 실행하지 않았다.",'',
'공통 ExperimentLogger와 W&B disabled 로컬 기록을 재사용했다. 공통 Logger/Simulator API는 변경하지 않았으며 CUDA 가중치 타입·요청별 입력 Contract·Parser 계약·원시 응답/CSV Readback·Model/Source/Data Hash 및 예산을 검증했다. CSV의 Pixel/GT Box에서 가시 물체110개×3구성의 IoU·성공 수·평균을 별도 산술로 재계산해 일치함을 확인했다. 변경은 독립 실험 Source/Output와 이 보고서에 한정했다. Training·새 Simulator Reset/Render/Rollout·Graph 재캐시·새 모델 다운로드·기존 프로세스 종료·외부 게시·Git Commit·Production 변경은0이다.','',
'[실험 Protocol](protocol.json), [정밀도별 모듈 계획](expected_precision.json), [비교 CSV](precision_comparison.csv), [전체 지표 JSON](precision_comparison_metrics.json), [검증](verification.json), [Artifact Index](artifact_index.json).','',
'| 구성 | Identity /Raw /Metric /SAM2 /비용 |','|---|---|']
for mode in VARIANTS:
 links=' /'.join(f'[{label}]({mode}/{file})' for label,file in [('Identity','model_identity.json'),('Raw','full_image_grounding_raw_responses.json'),('Metric','grounding_evaluation_metrics.json'),('SAM2','sam2_initialization_metrics.json'),('비용','grounding_latency_vram.json')] if (P/mode/file).exists());lines.append(f'|{names[mode]}|{links}|')
lines += ['', '## 11. 동결 Prompt 원문','', '```text',(P/'bf16/grounding_prompt.txt').read_text(),'```','',
'![고정된 네 조건의 세 구성 비교](precision_grounding_comparison.png)','']
(P/report_name).write_text('\n'.join(lines));(P/'grounding_prompt.txt').write_bytes((P/'bf16/grounding_prompt.txt').read_bytes())

with ExperimentLogger(team='integration',model='Task10I_Stage4A2_QwenThreePrecisionComparison',seed=42,job_type='evaluate',root=P/'.execution/experiments',config=dict(protocol=protocol,prompt_sha256=first['prompt_sha256'],source_sha256=first['execution_sources_before'])) as logger:
 for mode,s in summaries.items():
  if 'primary' not in s:continue
  o=s['primary']['image_object_level'];logger.log({mode:dict(PEG_recall=o['objects']['peg']['recall_IoU05']['rate'],BOX_recall=o['objects']['box']['recall_IoU05']['rate'],JSON_accepted=s['format']['accepted_JSON']['numerator']/96)})
 logger.record(status_by_variant={m:s['status'] for m,s in summaries.items()},source_inference_run_ids={m:rd(P/m/'verification.json').get('logger_run_id') for m in VARIANTS})
 logger.link_output(P/'precision_comparison_metrics.json',kind='metrics');logger.link_output(P/report_name,kind='reports');logger.link_output(P/'precision_grounding_comparison.png',kind='reports')
 evaluation_run_id=logger.run_id
parent.update(stage='COMPLETE_VERIFIED',completion='VERIFIED',evaluation_logger_run_id=evaluation_run_id,protected_file_count=len(protected),protected_after=after,all_protected_SHA_unchanged=True,environments_unchanged=True,model_processor_unchanged=True,input_image_SHA_unchanged=True,all_three_prediction_barrier_pass=True,prompt_sha256=first['prompt_sha256'],report_sha256=sha(P/report_name),actual_Qwen_loads=sum(rd(P/m/'verification.json')['model_loads'] for m in VARIANTS),actual_Qwen_requests=sum(z.get('grounding_requests',0) for z in parent['results'].values()),actual_SAM2_frames=sum(rd(P/m/'verification.json').get('sam2_camera_frames',0) for m in VARIANTS),active_original_stage4a2_runners=active,final_GPU_processes=gpu,production_changes=0,publishing=0,git_commits=0,new_models=0,new_DINO_forwards=0,new_simulator_rollouts=0,training=0,source_final={p.name:sha(p) for p in (P/'.execution').glob('*.py')});wr(P/'verification.json',parent)
for n,h in first['execution_sources_before'].items():assert sha(P/'.execution'/n)==h,n
files={str(p.relative_to(P)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in P.rglob('*') if p.is_file() and p.name!='artifact_index.json' and p.name!='finalize.log' and '__pycache__' not in p.parts}
wr(P/'artifact_index.json',dict(files=files,report='docs/visual-graph-integration/'+report_name,publishing=False))
print('COMPARISON_COMPLETE',json.dumps({m:dict(status=s['status'],PEG=s.get('primary',{}).get('image_object_level',{}).get('objects',{}).get('peg',{}).get('recall_IoU05'),BOX=s.get('primary',{}).get('image_object_level',{}).get('objects',{}).get('box',{}).get('recall_IoU05')) for m,s in summaries.items()}),flush=True)
