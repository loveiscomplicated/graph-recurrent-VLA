from common import *
import subprocess,collections
import numpy as np
from PIL import Image
from visual.runtime import initialization as init
v=rd(R/'verification.json');assert v['stage'] in ['EVALUATION_COMPLETE','COMPLETE_VERIFIED']
assert v['model_loads']==1 and v['completed_forwards']==64
prot=rd(R/'protocol.json');mf=rd(R/'.execution/evaluation_manifest.json');cases={c['id']:c for c in mf['cases']}
m=rd(R/'dino_evaluation_metrics.json');lat=rd(R/'dino_latency_vram.json');matched=rd(R/'matched_qwen_comparison.json')
protected={p:sha(p) for p in v['protected_before']};assert protected==v['protected_before']
packages=subprocess.check_output([str(PYTHON),'-m','pip','list','--format=json'],text=True);assert json.loads(packages)==json.loads(v['packages_before'])
assert sha(R/'dino_raw_responses.json')==v['raw_sha256_before_GT']==v['raw_sha256_after_GT']
assert sha(R/'dino_frame_selections.json')==v['selections_sha256_before_GT']==v['selections_sha256_after_GT']
assert sha(R/'dino_predictions.csv')==v['predictions_sha256_before_GT']
raw=rd(R/'dino_raw_responses.json')['responses'];by={(r['id'],r['object']):r for r in raw};frames=rd(R/'dino_frame_selections.json')['frames'];assert len(raw)==64 and len(frames)==32
# Replay exact existing consumer from frozen detector proposals, with no new forward.
for fr in frames:
 c=cases[fr['id']];assert sha(c['image_path'])==c['image_sha256'];rgb=np.asarray(Image.open(c['image_path']).convert('RGB'))
 def cached(image,prompt):
  obj=next(k for k,p in init.PROMPTS.items() if p==prompt);r=by[(c['id'],obj)];return r['boxes'],r['scores']
 ca={k:init.candidates(cached,rgb,p) for k,p in init.PROMPTS.items()};ss=init.assign(ca['peg'],ca['box'])
 for k,s in zip(['peg','box'],ss):
  got=dict(box=s[0],score=s[1]) if s else None;assert got==fr['selected'][k]
# Independent GT bbox and IoU readback for selected and every raw proposal.
def score(a,b):
 if a is None or b is None:return 0.
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);extent=np.maximum(0.,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));inter=float(extent.prod());union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter);return inter/union if union else 0.
gtcache={};checks=0
for filename,boxkey,iouskey in [('dino_object_evaluation.csv','selected_box','selected_IoU'),('dino_proposal_evaluation.csv','box','IoU')]:
 with (R/filename).open() as f:
  for row in csv.DictReader(f):
   c=cases[row['id']];key=(c['id'],row['object'])
   if key not in gtcache:
    with np.load(c['GT_path']) as z:yx=np.argwhere(np.isin(z['actor_segmentation'],c['object_ids'][row['object']]))
    gtcache[key]=[int(yx[:,1].min()),int(yx[:,0].min()),int(yx[:,1].max()+1),int(yx[:,0].max()+1)] if len(yx) else None
   gb=gtcache[key];assert (json.loads(row['GT_bbox']) if row['GT_bbox'] else None)==gb
   if row[iouskey]=='':continue
   box=json.loads(row[boxkey]) if row[boxkey] else None;assert abs(score(box,gb)-float(row[iouskey]))<1e-12;checks+=1
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not processes
v.update(stage='COMPLETE_VERIFIED',protected_after=protected,all_protected_SHA_unchanged=True,environments_unchanged=True,model_processor_unchanged=True,consumer_replay_from_saved_predictions='PASS 32 images / 0 forwards',independent_IoUs_checked=checks,final_GPU_compute_processes=processes,execution_sources_after={p.name:sha(p) for p in (R/'.execution').glob('*.py')},simulator_created=False,SAM2_loaded=False,Qwen_loaded=False,production_changed=False,original_runner='PAUSED_SCOPE_REVIEW');wr('verification.json',v)
def rr(x):return f'{x["numerator"]}/{x["denominator"]} ({100*x["rate"]:.1f}%)' if x['rate'] is not None else f'0/{x["denominator"]} (N/A)'
def fmt(x):return '-' if x is None else f'{x:.3f}'
a=m['native_resolution']['128']['image_object_level'];b=m['native_resolution']['512']['image_object_level']
delta={k:b['objects'][k]['selected_recall']['rate']-a['objects'][k]['selected_recall']['rate'] for k in ['peg','box']}
status='NATIVE_RESOLUTION_BENEFIT_OBSERVED' if all(x>0 for x in delta.values()) else 'NATIVE_RESOLUTION_PARTIAL_BENEFIT' if any(x>0 for x in delta.values()) and all(x>=0 for x in delta.values()) else 'NATIVE_RESOLUTION_MIXED' if any(x>0 for x in delta.values()) else 'NO_SELECTED_ACCURACY_GAIN_OBSERVED'
wr('experiment_conclusion.json',dict(status=status,selected_recall_delta=delta,scope='Four previously evaluated train parents, fixed current production prompt pair and selection rules; no broad model/generalization claim'))
lines=['# Task 10-I Stage4A-2 — Grounding DINO Native Resolution 128/512', '',
f'**판정: `{status}`.** 동일 상태의 시뮬레이터 원본 128→512에서 기존 DINO 선택 경로의 PEG Recall은 {rr(a["objects"]["peg"]["selected_recall"])} → {rr(b["objects"]["peg"]["selected_recall"])}, BOX Recall은 {rr(a["objects"]["box"]["selected_recall"])} → {rr(b["objects"]["box"]["selected_recall"])}였다. 기준은 visible pixel≥16, Box IoU≥0.5다. 이 결과는 기존 Train Parent 4개의 대응 비교 파일럿이며 일반화·희귀 오류 보증이 아니다.', '',
'## 1. 질문과 통제 조건', '',
'Qwen 실험에서 확보한 동일 상태의 원본 128/512 RGB·EXACT actor segmentation을 그대로 사용했다. 새 이미지 확대본을 고해상도 입력으로 대체하거나 새 장면을 선택하지 않았다. 각 해상도 16개 Camera-Frame, 총 32개 이미지다. Parent는 `0531`, `0275`, `0197`, `0305`; 각 t0/t80, Main `base_camera` / Wrist `hand_camera`다. 전체 32개를 모두 사용했다.', '',
'Vast.ai **58576054 / RTX 3090 24GB**, 기존 `IDEA-Research/grounding-dino-tiny` snapshot revision `'+str(prot['revision'])+'`. 기존 `GDino`와 `initialization.candidates/assign`을 직접 재사용했다. FP32, `cuda:0`, quantization/offload 없음. 모델 로드 1회, 64 forwards(32장 × 2프롬프트), 별도 warmup/retry 0회.', '',
'- PEG prompt: `red and white block`',
'- BOX prompt: `cardboard box`',
'- Box/Text threshold 각각 0.1. 면적 < 이미지의 25%, top-5, Null score 0.3, 두 선택 Box exclusion IoU 0.3을 유지했다.',
'- Task10-J의 별도 최적화 프롬프트로 바꾸거나 Prompt/threshold를 탐색하지 않았다. 이 결과는 현재 운영 프롬프트 고정 조건이며 DINO가 낼 수 있는 최고 성능이라는 뜻이 아니다.',
'- Processor default: shortest edge800 / longest edge1333. 실제 두 해상도 모두 `[1,3,800,800]`을 확인했다. 내부 처리 크기를 늘린 실험이 아니므로 native 세부 정보 효과를 비교한다.',
'- Postprocessing target_sizes는 원본 H/W다. Box를 Processor의 800 좌표로 GT와 비교하지 않는다. 모델 원시 Box를 clipping/GT 보정하지 않았다. 이미지 밖으로 일부 나가는 Box도 기존 DINO consumer 계약대로 유지하고 별도 기록했다.', '',
'## 2. 입력 경로와 동결', '',
'DINO inference용 Manifest에는 RGB 경로·SHA·카메라/Frame ID·원본 크기만 두고 GT path/Actor ID/Privileged state/Qwen 결과를 제거했다. RGB와 고정 문자열만 Detector에 제공했다. 원시 후보·Score·기존 규칙의 선택 Box를 모두 저장하고 SHA 동결한 뒤 별도 평가 프로세스에서 GT를 열었다. 과거 GT/성적은 이미 알려져 있으므로 연구자 맹검이라고 표현하지 않는다.', '',
'이번에는 Simulator 생성/Reset/Render/Step 0, Qwen load/forward 0, SAM2 load/forward 0, training 0이다. 기존 캡처의 같은 상태·카메라 위치/FOV/마운트·Native GT 검증을 재사용하며 이번 작업에서 그 검증을 새 동적 replay로 주장하지 않는다.', '',
'## 3. Detector 후보와 실제 선택 정확도', '',
'| 원본 | 물체 | 기존 Joint 선택 Recall | Score Top-1 Recall | 후보 중 정답 존재(Oracle) | 선택 평균 IoU(Null 포함) |',
'|---|---|---:|---:|---:|---:|']
for size in ['128','512']:
 for k,o in m['native_resolution'][size]['image_object_level']['objects'].items():lines.append(f'| {size} | {k} | {rr(o["selected_recall"])} | {rr(o["raw_top1_recall"])} | {rr(o["any_proposal_coverage"])} | {fmt(o["selected_IoU_including_null"]["mean"])} |')
lines += ['', 'Oracle 열은 GT로 후보 중 가장 좋은 Box를 찾아 진단한 coverage다. 실제 Consumer는 GT를 보지 않았고 이 Oracle Box를 출력하거나 SAM2에 제공하지 않았다. Detector가 후보를 발견하는 능력과 기존 선택 규칙의 동작을 구분해야 한다. 이 수치는 COCO AP/mAP가 아니다.', '',
'| 원본 | 물체 | IoU≥0.3 | IoU≥0.7 | Null | 잘못된 선택 / 출력 | Hallucination / ABSENT |',
'|---|---|---:|---:|---:|---:|---:|']
for size in ['128','512']:
 for k,o in m['native_resolution'][size]['image_object_level']['objects'].items():lines.append(f'| {size} | {k} | {rr(o["IoU_thresholds"]["0.3"])} | {rr(o["IoU_thresholds"]["0.7"])} | {rr(o["null_rate"])} | {rr(o["false_discovery"])} | {rr(o["hallucinated_selected_box_absent"])} |')
lines += ['', 'Null은 가시 목표에 대한 Miss로 계산한다. ABSENT 분모가 0이면 해당 False Positive는 N/A이며 0%라고 해석하지 않는다. False discovery는 선택 Box 중 IoU<0.5의 비율이다. 원시 후보·선택 Box의 누락을 분리해서 보존했다.', '',
'## 4. Parent / Camera / 시점', '',
'| 원본 | PEG Parent-macro Recall | BOX Parent-macro Recall | 두 물체 동시 선택 성공 | 하나 이상 Wrong-object |',
'|---|---:|---:|---:|---:|']
for size,mm in m['native_resolution'].items():
 sc=mm['image_object_level']['scene'];lines.append(f'| {size} | {fmt(mm["parent_macro"]["peg"])} | {fmt(mm["parent_macro"]["box"])} | {rr(sc["both_selected_correct"])} | {rr(sc["any_wrong_object"])} |')
lines += ['', '| 원본 | 구분 | PEG Recall | BOX Recall |', '|---|---|---:|---:|']
for size,mm in m['native_resolution'].items():
 for group in ['by_camera','by_time']:
  for name,z in mm[group].items():lines.append(f'| {size} | {name} | {rr(z["objects"]["peg"]["selected_recall"])} | {rr(z["objects"]["box"]["selected_recall"])} |')
lines += ['', 't80에도 같은 per-frame candidates/assign을 독립 적용한 결과다. Production이 실제 t80마다 다시 초기화한다거나 temporal scan을 replay했다는 뜻은 아니다. Saved t0/t80을 t5 단위 연속 시퀀스로 취급하지 않았다.', '',
'## 5. 대응 변화와 통계', '',
'같은 목표가 양쪽 해상도에서 visible≥16인 경우만 대응시켰다. 독립 집계 단위는 물리 Parent 4개이며 각 Parent 안의 Camera/Frame을 별개 Parent로 세지 않았다. Parent별 변화 평균, 10,000회 Parent-cluster Bootstrap(seed42)을 보고한다. 작은 파일럿의 탐색적 CI이며 다중 비교 보정은 없다. 퇴화 CI는 희귀 오류 안전성을 뜻하지 않는다.', '',
'| 물체 | 지표 (512−128) | Parent-macro 차이 | 탐색적 95% CI |', '|---|---|---:|---|']
for k,e in m['paired_512_minus_128'].items():
 for name in ['selected_recall_delta','raw_top1_recall_delta','proposal_coverage_delta','selected_mean_IoU_delta']:
  z=e[name];lines.append(f'| {k} | {name} | {fmt(z["parent_macro_delta"])} | {z["exploratory_parent_cluster_CI95"]} |')
with (R/'dino_resolution_transitions.csv').open() as f:trans=list(csv.DictReader(f))
lines += ['', '선택 Recall의 복구/회귀 수: '+', '.join(f'{k} 복구 {sum(r["object"]==k and r["recovered"]=="True" for r in trans)}, 회귀 {sum(r["object"]==k and r["regressed"]=="True" for r in trans)}' for k in ['peg','box'])+'. [전체 대응 사례](dino_resolution_transitions.csv)에 실패 원인 변화를 보존했다.', '',
'## 6. 실패 원인과 공간 오차', '',
'| 원본 | 물체 | Drop reason | Semantic confusion | 평균 Center error / native px | 128px 환산 |', '|---|---|---|---|---:|---:|']
for size,mm in m['native_resolution'].items():
 for k,o in mm['image_object_level']['objects'].items():lines.append(f'| {size} | {k} | `{json.dumps(o["drop_reasons"])}` | `{json.dumps(o["semantic_confusion"])}` | {fmt(o["center_error_px"]["mean"])} | {fmt(o["center_error_128px_units"]["mean"])} |')
lines += ['', 'NO_GOOD_PROPOSAL은 0.1 threshold를 넘은 후보에 IoU≥0.5 정답이 없는 경우다. AREA/TOP5/NULL_SCORE/JOINT_OR_RANKING은 기존 후보 선택 경로에서의 손실을 나타낸다. Threshold 미만의 잠재 Query를 GT로 되살리지 않았다. NULL_SCORE는 정답과 일치하는 후보가 Null score 이하였다는 사후 손실 원인이다. 최종 출력 자체가 Null이라는 뜻은 아니며, 그 경우 다른 잘못된 Box가 선택될 수도 있다. Wrong-object는 다른 Actor가 선택 Box foreground의 ≥70%, pixel≥16이며 목표 IoU<0.5인 경우다. 오검출과 위치 오차를 동일시하지 않는다. Center error는 출력이 있는 경우만의 조건부 지표이며 128/512 원본 픽셀 척도를 직접 비교하지 않고 환산값을 병기했다.', '',
'고해상도가 경계/텍스처 정보를 바꾸어 score·rank·IoU·null 선택을 바꿀 수 있다는 것은 원인 가설이다. 이번 실험으로 어느 texture나 단일 feature가 원인이었는지 확정하지 않는다. 관측된 변화는 위 Frame별 원시 score·Box·GT와 consumer replay로 확인한 범위다.', '',
'## 7. Latency / GPU', '',
f'기존 DINO 모델 로드 {lat["load_ms"]/1000:.3f}s, 로드 후 allocated {lat["loaded_memory"]["allocated"]/2**30:.3f} GiB. 파일 SHA 사전 검사로 OS file cache는 warm일 수 있다. CPU process sampled peak {lat["CPU_peak_sampled_bytes"]/2**30:.3f} GiB(50ms RSS sampling).', '',
'| 원본 | Warm call n | 1프롬프트 평균 / median / p95 ms | 두 프롬프트 이미지 평균 / median / p95 ms | Peak allocated GiB |', '|---|---:|---|---|---:|']
for size,z in lat['summary'].items():
 call=z['warm_per_call_end_to_end_ms'];im=z['warm_two_prompt_image_end_to_end_ms'];lines.append(f'| {size} | {call["n"]} | {fmt(call["mean"])} / {fmt(call["median"])} / {fmt(call["p95"])} | {fmt(im["mean"])} / {fmt(im["median"])} / {fmt(im["p95"])} (n={im["n"]}) | {z["peak_allocated_bytes"]/2**30:.3f} |')
first=lat['calls'][0];lines += ['', f'Cold 첫 call {first["native_size"]}/{first["object"]} {first["end_to_end_ms"]:.2f}ms는 Warm call에서 제외했다. 첫 이미지도 Warm 이미지 집계에서 제외했다. 16개 대응 pair 안에서 128→512와 512→128 순서를 8개씩 교차했다. 입력 해상도별 내부 Tensor 크기는 동일했다. 두 프롬프트 이미지 E2E는 RGB PNG read+2회 Detector+실제 선택+호출별 측정/원시 결과 저장 overhead를 포함한다. 1프롬프트 E2E는 GDino 전체 호출이며 pre(H2D 포함)/forward/post 세부 원자료를 제공한다. 전처리·후처리·forward 시간은 [Latency JSON](dino_latency_vram.json)에 있다. 모델 외 다른 GPU resident 모델은 없었다.', '',
'## 8. 동일 이미지 Qwen과의 제한된 비교', '',
'이제 동일 Native RGB에 대응하는 DINO 결과가 생겼다. 기존 Qwen을 재실행하지 않고 동결 결과만 join했다. GT/카메라/시점/IoU 기준은 동일하지만 DINO와 Qwen의 Prompt·선택 경로·정밀도는 다르므로 파이프라인 조건의 제한된 비교다. DINO 1이미지=2회 text-conditioned forward, Qwen 1이미지=두 물체를 한 번에 생성한다. 모델의 본질적 우열·외부 분포 성능을 주장하지 않는다.', '',
'| 원본 | 물체 | DINO 실제 선택 | Qwen P0 Accepted | Qwen P1 Accepted | Qwen P1 사후 좌표 진단(미승인) |', '|---|---|---:|---:|---:|---:|']
for size,mm in matched['native_resolution'].items():
 for k,z in mm.items():
  d=z['Qwen_P1_diagnostic_only'];lines.append(f'| {size} | {k} | {rr(z["DINO_selected"])} | {rr(z["Qwen_P0_accepted"])} | {rr(z["Qwen_P1_accepted"])} | {d["numerator"]}/{d["denominator"]} |')
lines += ['', 'Qwen 진단 값은 중복 키가 동일하거나 추가 필드가 있는 완전한 JSON에서 읽은 좌표이며 운영 Accepted/SAM2 성공으로 계산하지 않았다. DINO Oracle 후보 coverage와도 혼동하지 않는다. [동일 이미지 비교 JSON](matched_qwen_comparison.json)에 각각 분리했다.', '',
'## 9. 결론과 제한', '',
'관측상 Native512에서 PEG/BOX 선택 Recall이 각각 12.5 percentage points 증가했다. PEG 평균 선택 IoU는 0.393→0.571, BOX는 0.372→0.506이었다. 다만 Native512의 Main BOX는 7/8, Wrist BOX는 1/8로 집중 실패가 남았다. BOX→Robot 오인식은 128의 6건에서 512의 8건으로 증가했다. 높은 해상도가 모든 오류 유형을 줄였다는 주장은 성립하지 않는다. 탐색적 Recall 변화 CI는 두 물체 모두 [0,0.25]여서 이 표본으로 통계적 우월성을 단정하지 않는다.', '',
'이번 결과는 현행 DINO와 선택 규칙을 고정했을 때 native capture resolution 변화가 만든 성능 차이다. 원본 고해상도 효과를 Detector 후보 단계와 실제 선택 단계로 구분해 해석한다. 4개 Parent 표본만으로 현장 안정성이나 높은 confidence를 의미 정확성으로 보증하지 않는다. GT가 없는 경우/미가시 객체/아주 작은 객체의 충분한 검증은 별도다. 추가 Prompt나 Parent를 결과 뒤에 늘리지 않았다.', '',
'SAM2 Mask, Object Graph, 장기 Tracking, 정책 Closed-loop success는 이번에 평가하지 않았다. DINO Box가 좋아졌다고 Visual-to-Graph 전체 성능 향상을 확정할 수 없다. Production source·threshold와 기존 Stage4A-2 `PAUSED_SCOPE_REVIEW` 및 Stop Stub은 그대로다.', '',
'## 10. 검증·보존·산출물', '',
f'공통 ExperimentLogger를 W&B disabled 로컬 기록으로 재사용했다. 보호 파일 {len(protected)}개 SHA 및 Python package 목록을 전후 대조해 보존을 확인했다. 기존 Qwen 실험 전체 artifact와 DINO snapshot도 포함했다. 저장된 원시 후보만으로 실제 consumer 32개 이미지의 선택을 재생했고 추가 forward는 0회다. Native GT에서 Bounding Box를 별도 재구성하고 선택/전체 후보 IoU {checks}개를 독립 재계산했다. Raw/선택 CSV SHA는 GT 평가 전후 불변이다. 기존 DINO 평가 테스트 5개 및 새 Source compile 통과. 실제 GPU 실행은 RTX3090에서 했고 로컬 macOS GPU·Simulator·renderer는 실행하지 않았다.', '',
'- [Protocol / 모델·Source SHA](protocol.json) · [RGB-only input manifest](input_manifest.json) · [실제 모델 Identity](model_identity.json)',
'- [원시 DINO 응답](dino_raw_responses.json) · [선택 Box](dino_predictions.csv) · [객체별 전체 평가](dino_object_evaluation.csv) · [후보별 전체 평가](dino_proposal_evaluation.csv)',
'- [전체 지표](dino_evaluation_metrics.json) · [실패 사례](dino_failure_cases.csv) · [대응 변화](dino_resolution_transitions.csv) · [Latency / VRAM](dino_latency_vram.json) · [보존 검증](verification.json)', '',
'![동일 Parent의 t0/t80 Main/Wrist — Native128/512](dino_native_resolution_comparison.png)', '',
'주황=DINO 선택 PEG, 하늘색=DINO 선택 BOX, 초록=GT PEG, 자홍=GT BOX. 원본 해상도와 표시 크기는 별개이며, 표시용 Resize가 실험 입력이 아니다. SAM2 Mask는 그리지 않았다.', '']
(R/'10i_stage4a2_dino_native_resolution.md').write_text('\n'.join(lines))
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in R.rglob('*') if p.is_file() and p.name not in ['artifact_index.json','finalize.log','infer.log','evaluate.log','prepare.log'] and p.suffix!='.tmp'}
wr('artifact_index.json',dict(status='COMPLETE_VERIFIED',files=files));print('COMPLETE_VERIFIED',status,'protected',len(protected),'IoU checks',checks,'artifacts',len(files),flush=True)
