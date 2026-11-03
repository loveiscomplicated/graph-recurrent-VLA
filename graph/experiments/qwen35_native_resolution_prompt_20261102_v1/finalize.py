from common import *
import subprocess,collections
import numpy as np
v=rd(R/'verification.json');assert v['stage'] in ['EVALUATION_COMPLETE','COMPLETE_VERIFIED']
diag=rd(R/'posthoc_coordinate_diagnostic.json');m=rd(R/'grounding_evaluation_metrics.json');sm=rd(R/'sam2_initialization_metrics.json');lat=rd(R/'grounding_latency_vram.json');cap=rd(R/'capture_verification.json');proto=rd(R/'protocol.json');pre=rd(R/'preflight_verification.json');mf=rd(R/'input_manifest.json')
protected={p:sha(p) for p in v['protected_before']};assert protected==v['protected_before']
ident=rd(R/'model_identity.json');assert all(sha(B/'model'/n)==h for n,h in ident['checkpoint_hashes'].items())
packages=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True);assert json.loads(packages)==json.loads(v['packages_before'])
assert sha(R/'full_image_grounding_raw_responses.json')==v['raw_SHA_before_GT']==v['raw_SHA_after_GT']
assert sha(R/'full_image_grounding_predictions.csv')==v['predictions_SHA_before_GT']==v['predictions_SHA_after_GT']
with (R/'.execution/object_evaluation.csv').open() as f:rows=list(csv.DictReader(f))
# Independent scalar IoU formula from persisted row coordinates, not evaluator helper.
checked=0
for r in rows:
 if r['bbox_IoU'] in ['','None']:continue
 a=json.loads(r['pixel']) if r['pixel'] else None;b=json.loads(r['GT_bbox']) if r['GT_bbox'] else None;expected=0.
 if a is not None and b is not None:
  area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1]);overlap=[min(a[2],b[2])-max(a[0],b[0]),min(a[3],b[3])-max(a[1],b[1])];intersection=max(0.,overlap[0])*max(0.,overlap[1]);expected=intersection/(area_a+area_b-intersection)
 assert abs(expected-float(r['bbox_IoU']))<1e-12;checked+=1
raw=rd(R/'full_image_grounding_raw_responses.json')['responses'];assert len(raw)==64
cases={c['id']:c for c in mf['cases']}
for r in raw:
 c=cases[r['id']];assert parse(r['raw_text'],c['width'],c['height'])==r['parsed']
 assert sha(c['image_path'])==r['input_image_sha256']==c['image_sha256']
assert all(len([r for r in raw if r['variant']==mode])==16 for mode in VARIANTS)
# Check all SAM2 mask IoUs independently from saved masks and actor segmentation.
with (R/'sam2_initialization_cases.csv').open() as f:srows=list(csv.DictReader(f))
sams={(s['variant'],s['pair_id']):s for s in rd(R/'.execution/sam2_raw_initializations.json')};maskchecks=0
for row in srows:
 if row['visible_pixels']=='0':continue
 c=cases[row['id']];s=sams[(row['variant'],row['pair_id'])]
 with np.load(c['GT_path']) as z:gt=np.isin(z['actor_segmentation'],c['object_ids'][row['object']])
 with np.load(s['masks_path']) as z:pred=z[row['object']] if row['object'] in z.files else np.zeros_like(gt)
 denom=np.logical_or(gt,pred).sum();score=np.logical_and(gt,pred).sum()/denom if denom else None
 assert score is not None and abs(score-float(row['mask_IoU']))<1e-12;maskchecks+=1
assert os.environ.get('CONTAINER_ID')=='58576054'
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not processes
v.update(stage='COMPLETE_VERIFIED',protected_after=protected,all_protected_SHA_unchanged=True,model_processor_unchanged=True,environments_unchanged=True,parser_raw_readback_pass=True,independent_bbox_IoUs_checked=checked,independent_mask_IoUs_checked=maskchecks,final_GPU_compute_processes=processes,new_DINO_forwards=0,actual_instance_id=os.environ.get('CONTAINER_ID'),original_runner='PAUSED_SCOPE_REVIEW',source_after={p.name:sha(p) for p in (R/'.execution').glob('*.py')});wr('verification.json',v)
def fmt(x):return '-' if x is None else f'{x:.3f}'
def rr(x):return f'{x["numerator"]}/{x["denominator"]} ({100*x["rate"]:.1f}%)' if x['rate'] is not None else f'0/{x["denominator"]} (N/A)'
lines=['# Task 10-I Stage4A-2 — Native Resolution × Prompt 2×2 Pilot','',
'**Native 512와 P1에서 위치를 읽는 능력은 개선 신호가 있었지만, 출력 계약 실패가 실제 사용을 막았다.** 사후 진단에서 512/P1 좌표는 PEG 13/16, BOX 11/16이 IoU≥0.5였으나 정식 JSON 계약 통과는 0/16이다. 이는 Accepted 성능이나 SAM2 성공으로 계산하지 않는다.\n\nBF16 Qwen3.5-9B에서 시뮬레이터 원본 128/512 해상도와 두 프롬프트를 비교했다. 동일 물리 상태의 16개 Camera-Frame에 4조건을 적용했고, 64개 Qwen 요청과 64개 Temporary SAM2 대상을 완료했다. 이 결과는 이미 연구에 사용한 Train Parent 4개의 파일럿이며 독립 일반화 결과가 아니다.','',
'## 1. 실험 계약과 Identity','',
'- Vast.ai instance `58576054`, RTX 3090 24GB. 기존 모델·Processor·환경·SAM2 가중치 재사용. 새 모델/패키지 설치 없음.',
'- `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 전체 BF16, SDPA, `cuda:0`, offload 없음. 모델 로드 1회.',
'- 모든 조건에서 Processor `min_pixels=max_pixels=262144`, 실제 grid `[1,32,32]`, vision token 256개. Native 128은 이 동일 Processor 안에서 리사이즈되고 Native 512는 처음부터 512로 렌더링된다.',
'- Batch 1, greedy, `enable_thinking=False`, `max_new_tokens=128`, KV cache ON. 별도 warmup, 자동 retry, sampling 없음.',
'- 좌표 0–1000을 각 원본 너비/높이로 변환한다. 128/512를 하드코딩한 동일 배율로 처리하지 않는다. 범위 밖 좌표를 clipping하거나 GT로 보정하지 않는다.',
'- P0/P1 모두 같은 native array schema와 동일 파서. bare JSON을 별도로 집계하고, 응답 전체를 감싼 단일 JSON fence도 기존 계약대로 허용한다. 중복 키, 부분 응답 복구, 설명문, nonfinite/bool/역전·범위 밖 좌표는 불허한다.',
'- 원래 128 RGB/256 Processor 점수를 이번 A조건 점수로 재사용하지 않았다. 네 조건 모두 새로 추론했다.','',
'| Prompt | 단어 수 | SHA256 |','|---|---:|---|']
for name,p in v['prompt_identities'].items():lines.append(f'| {name} | {p["words"]} | `{p["sha256"]}` |')
lines += ['', 'P0는 기존 상세 프롬프트를 그대로 유지했다. P1은 작업의 두 역할과 물체/로봇 구분, 가시 부분, 좌표·출력 규칙을 간결하게 정리한 단일 후보다. 색상·위치·Episode 정답·후보 Box를 넣지 않았고 결과 후 수정하지 않았다. 두 Prompt의 입력 토큰 차이는 Prompt 요인의 일부다.','',
'## 2. 대상 선정과 Native Capture 검증','',
'24개 기존 Train Parent에서 `SHA256("Task10I_native_resolution_prompt_pilot_v1|" + parent_id)` 오름차순 앞 4개를 고정했다. GT 가시성·DINO/Qwen 성적 필터는 사용하지 않았다. 선택 Parent: '+', '.join('`'+p['parent_id']+'`' for p in mf['parents'])+'. 각 t0/t80, base_camera/hand_camera를 모두 사용했다. 누락·대체 Frame은 없다. 나머지 20개 Parent는 이 파일럿에서 제외했다.','',
'저장된 seed로 동일 geometry를 만들고 기록된 qpos/qvel, Peg/Box pose를 복원했다. TCP/hole pose·geometry·카메라 calibration을 검증한 뒤, 기존 카메라 설정에서 uid와 width/height만 바꾼 512 카메라를 같은 Scene에 추가했다. mount/pose/FOV는 동일하며, 두 해상도 RGB·Depth·Actor Segmentation은 한 번의 동일 상태 관측에서 얻었다. 물리 Step은 0회다. 이는 정적 렌더 복원이며 PhysX solver cache까지 동적 재현했다는 뜻은 아니다.','',
'실행 backend는 `physx_cpu / sapien_cpu`이며 CUDA 물리 시뮬레이션은 실행하지 않았다. Native 512 렌더는 RTX 3090 호스트에서 실제 실행했다. 원래 보존 RGB와 새 128 RGB는 '+str(sum(p['original128_rgb_exact'] for p in cap['pairs']))+'/16에서 픽셀 일치했고, 나머지도 최대 평균 절대 RGB 차이 '+fmt(max(p['original128_mean_abs_rgb_difference'] for p in cap['pairs']))+' (0–255 척도)였다. 과거 이미지와 완전히 같다고 주장하지 않으며, 모든 비교는 이번 동시 캡처끼리 수행한다.','',
'## 3. DINO-free 입력과 평가 장벽','',
'Capture 프로세스는 장면 복원에 필요한 privileged state를 읽고 GT sidecar를 저장한다. Qwen inference 프로세스는 원본 전체 RGB 1장과 고정 Prompt만 사용한다. GT/로봇 상태/캡처 seed/Depth/Segmentation/후보 Box는 Qwen message에 들어가지 않는다. 64개 원시 응답과 좌표 CSV를 SHA 동결한 뒤 Qwen Box만으로 SAM2를 실행했고, 모든 Mask를 동결한 후 별도 CPU 평가 프로세스가 GT sidecar를 열었다. 과거 실험 결과가 이미 알려진 파일럿으로, 연구자 전체가 GT를 처음 보는 맹검 실험이라고 표현하지 않는다. DINO 원시 출력은 열지 않았고 새 DINO forward는 없다. 이전 보고서의 baseline 부족 요약을 확인했지만 이번 지표 계산이나 VLM 입력에 DINO 결과를 사용하지 않았다.','',
'## 4. Grounding 결과','',
'주 Recall은 가시 픽셀 ≥16인 목표에 대한 Bounding Box IoU≥0.5 비율이다. Null/invalid는 분모에서 제거하지 않고 Miss로 처리한다. 작은 가시 영역(1–15)은 TINY_VISIBLE, 0은 ABSENT로 구분한다.','',
'| 조건 | PEG Recall | BOX Recall | PEG 평균 IoU | BOX 평균 IoU | JSON 허용 | Bare JSON |','|---|---:|---:|---:|---:|---:|---:|']
for mode in VARIANTS:
 a=m['conditions'][mode]['image_object_level'];o=a['objects'];lines.append(f'| {mode} | {rr(o["peg"]["visible_recall_IoU05"])} | {rr(o["box"]["visible_recall_IoU05"])} | {fmt(o["peg"]["bbox_IoU_including_misses"]["mean"])} | {fmt(o["box"]["bbox_IoU_including_misses"]["mean"])} | {rr(a["image_JSON_validity"])} | {rr(a["image_bare_JSON_validity"])} |')
lines += ['', '| 조건 | PEG Parent-macro Recall | BOX Parent-macro Recall | 두 물체 동시 성공 | 하나 이상 Wrong-object |','|---|---:|---:|---:|---:|']
for mode in VARIANTS:
 a=m['conditions'][mode];s=a['image_object_level']['scene'];lines.append(f'| {mode} | {fmt(a["parent_macro"]["peg"])} | {fmt(a["parent_macro"]["box"])} | {rr(s["both_correct"])} | {rr(s["any_wrong_object"])} |')
lines += ['', '| 조건 | 물체 | IoU≥0.3 | IoU≥0.7 | Hallucination / ABSENT | False discovery / emitted | Null |','|---|---|---:|---:|---:|---:|---:|']
for mode in VARIANTS:
 a=m['conditions'][mode]['image_object_level']
 for k,o in a['objects'].items():lines.append(f'| {mode} | {k} | {rr(o["IoU_thresholds"]["0.3"])} | {rr(o["IoU_thresholds"]["0.7"])} | {rr(o["hallucination_absent"])} | {rr(o["false_discovery"])} | {rr(a["null_rate"])} |')
lines += ['', 'Null 열은 조건 전체의 Object-level 비율이다. ABSENT 분모가 없으면 Hallucination은 측정 불가이며 0%로 해석하지 않는다. False discovery는 평가 가능 상태에서 Box를 출력했으나 IoU≥0.5를 만족하지 못한 비율이다. 잘못된 물체는 IoU<0.5이면서 다른 Actor가 Box 내부 foreground의 ≥70%이고 그 Actor pixel≥16인 보수적 진단이다. Mislocalization과 별도로 저장했다.','',
'## 5. Camera / 시점 / 공간 오차','',
'| 조건 | 구분 | PEG Recall | BOX Recall |','|---|---|---:|---:|']
for mode in VARIANTS:
 for group in ['by_camera','by_time']:
  for name,a in m['conditions'][mode][group].items():lines.append(f'| {mode} | {name} | {rr(a["objects"]["peg"]["visible_recall_IoU05"])} | {rr(a["objects"]["box"]["visible_recall_IoU05"])} |')
lines += ['', '| 조건 | PEG center error / 원본 px | BOX center error / 원본 px | PEG / 128px 환산 | BOX / 128px 환산 |','|---|---:|---:|---:|---:|']
for mode in VARIANTS:
 a=m['conditions'][mode]['image_object_level']['objects'];lines.append(f'| {mode} | {fmt(a["peg"]["center_error_px"]["mean"])} | {fmt(a["box"]["center_error_px"]["mean"])} | {fmt(a["peg"]["center_error_in_128px_units"]["mean"])} | {fmt(a["box"]["center_error_in_128px_units"]["mean"])} |')
lines += ['', '원본 pixel 오차는 128/512의 척도가 다르므로 환산값도 함께 제시했다. Center error는 Box를 출력한 경우만의 조건부 수치다. Null/invalid를 반영한 평균 IoU·Recall과 함께 해석한다.','',
'## 6. 2×2 Paired 효과','',
'두 해상도에서 모두 가시 pixel≥16인 공통 목표만 비교한다. 공통 분모 PEG '+str(m['common_support_objects']['peg'])+', BOX '+str(m['common_support_objects']['box'])+'. Parent별 차이를 평균했고 Parent 4개 단위 Bootstrap 10,000회, seed 42를 사용했다. CI는 작은 파일럿의 탐색적 수치다. 희귀 오류·독립 일반화·통계적 우월성의 증거로 사용하지 않는다. 다중 비교 보정은 없다.','',
'| 대비 | 물체 | Parent-macro Recall 차이 | 탐색적 95% CI | 평균 IoU 차이 |','|---|---|---|---:|']
for name,e in m['paired_effects'].items():
 for k,z in e.items():lines.append(f'| {name} | {k} | {fmt(z["recall_delta"]["parent_macro_delta"])} | {z["recall_delta"]["exploratory_parent_cluster_CI95"]} | {fmt(z["IoU_delta"]["parent_macro_delta"])} |')
lines += ['', 'resolution_P0/P1은 512−128, prompt_native128/512는 P1−P0다. interaction은 `(D−C)−(B−A)`이며 해상도 효과가 Prompt에 따라 달라지는지를 나타낸다.','',
'## 7. Temporary SAM2 실제 초기화','',
'동일한 사전 선정 16개 Camera-Frame 전부를 조건마다 평가했다. 총 64개 평가 대상 중 유효 계약을 통과한 18개에서 실제 Temporary State를 생성해 36개 Object Box를 초기화했고, 나머지는 invalid로 미초기화 실패 처리했다. SAM2.1 Hiera Small 기존 가중치, 임시 State, Qwen 유효 Box만 제공했다. RGB를 기존 경로대로 JPEG에 기록하고 SAM2 내부 1024 처리 후 원본 128/512 Mask를 얻었다. GT Box 대체·장기 추적·기존 Memory Commit은 없다.','',
'| 조건 | PEG Mask IoU≥0.8 | BOX Mask IoU≥0.8 | PEG 평균 Mask IoU(실패 포함) | BOX 평균 Mask IoU(실패 포함) |','|---|---:|---:|---:|---:|']
for mode in VARIANTS:
 a=sm[mode]['objects'];lines.append(f'| {mode} | {rr(a["peg"]["IoU08_success"])} | {rr(a["box"]["IoU08_success"])} | {fmt(a["peg"]["mask_IoU_including_noninitialization"]["mean"])} | {fmt(a["box"]["mask_IoU_including_noninitialization"]["mean"])} |')
lines += ['', 'Native 512에서는 SAM2에 전달하는 RGB도 더 선명하므로 Mask 성능 차이는 Qwen Box와 SAM2 입력 정보가 함께 달라진 End-to-end 효과다. Qwen 자체의 위치 성능은 Box IoU로 구분하며, SAM2 개선 전부를 Qwen 개선으로 귀속하지 않는다. Pixel precision/recall, Actor composition, robot/background contamination, Wrong-object와 초기화 여부는 [SAM2 사례 CSV](sam2_initialization_cases.csv) 및 [전체 SAM2 지표](sam2_initialization_metrics.json)에 보존했다. Null/invalid 미초기화는 가시 목표의 End-to-end 성공률에서 실패로 유지했다.','',
'## 8. Latency / 자원','',
f'Model load {lat["load_ms"]/1000:.2f}s; load 후 allocated {lat["after_load"]["allocated"]/2**30:.3f} GiB. CPU process sampled peak {lat["CPU_process_peak_sampled_bytes"]/2**30:.3f} GiB (50ms RSS sampling), ru_maxrss {lat["CPU_ru_maxrss_KiB"]} KiB. 모델 파일은 사전 hash 검사로 file cache가 warm일 수 있다.', '',
'| 조건 | Warm n | 평균 E2E ms | Median ms | p95 ms | 평균 output tokens | Peak allocated GiB |','|---|---:|---:|---:|---:|---:|---:|']
for mode in VARIANTS:
 a=lat['summary'][mode];w=a['warm_end_to_end_ms'];lines.append(f'| {mode} | {w["n"]} | {fmt(w["mean"])} | {fmt(w["median"])} | {fmt(w["p95"])} | {fmt(a["output_tokens"]["mean"])} | {a["peak_allocated_bytes"]/2**30:.3f} |')
co=lat['co_residency'];first=lat['requests'][0]
lines += ['', f'Cold 첫 요청은 {first["variant"]}, {first["end_to_end_ms"]:.1f}ms이며 Warm 집계에서 제외했다. 조건 순서는 각 16개 pair 안에서 Latin rotation으로 교차시켜 각 순서 위치에 조건당 4회씩 배치했다. 한 번 로드한 모델을 재사용했다.', '',
f'Qwen+SAM2 동시 GPU 상주: allocated {co["loaded_memory"]["allocated"]/2**30:.3f} GiB, offload 없음. SAM2 warm forward를 포함한 peak와 preprocessing/vision/prefill/decode/tokens는 [Latency 원자료](grounding_latency_vram.json)에 저장했다. E2E는 PNG read + Processor + H2D + generation + decode이며 CUDA event first-forward에는 vision이 포함된다. Prefill−vision 차이는 근사치다. 설치 환경의 flash-linear-attention/causal_conv1d 부재로 일부 Gated DeltaNet·causal-conv 연산은 reference PyTorch 경로를 사용했다. 최적화 kernel 환경의 최고 속도 측정이 아니다. Camera rendering 시간은 별도 capture 총 {cap["wall_seconds"]:.2f}s로, 환경 생성·seed reset·복원·RGB/GT 저장을 포함하여 순수 sensor latency로 쓰지 않는다.', '',
'## 9. 실패 관측과 원인 해석','',
'관측은 저장된 응답·Actor segmentation·Box/Mask 지표에 근거한다. 포맷 실패, 의미 혼동, 공간 오차를 분리했다. Input/output token 제한에 도달했다는 사실만으로 truncation이라 판정하지 않는다.','']
for mode in VARIANTS:
 a=m['conditions'][mode]['image_object_level'];conf={k:o['semantic_confusion'] for k,o in a['objects'].items()};lines.append(f'- {mode}: JSON 허용 {rr(a["image_JSON_validity"])}; semantic confusion {json.dumps(conf)}; 128 output token 도달 {lat["summary"][mode]["at_output_limit"]}/16.')
lines += ['', '가설: Native 고해상도는 실제 경계·opening·가림 정보의 손실을 줄일 수 있지만, 의미 역할의 혼동이나 좌표 생성 능력 부족을 자동으로 해결하지는 않는다. P1 효과는 길이 축소·표현·중복 키 지시를 함께 바꾼 Prompt 전체 효과이며, 특정 문장 하나의 인과 효과로 해석할 수 없다. Processor 512를 공통으로 고정했기 때문에 이번 resolution 대비는 시각 토큰 수 증가의 효과와 분리된다.','',
'## 10. 판단과 범위','',
'이 파일럿으로 DINO 제거 여부를 확정하지 않는다. 이번 Native512 및 전체 paired matrix에 대응하는 충분한 보존 DINO baseline을 구성하지 않았고 새 DINO를 실행하지 않았으므로 `MATCHED_DINO_BASELINE_INSUFFICIENT`다. DINO 대비 우월성 주장은 보류한다. 4개 Parent·16개 Camera-Frame은 운영 기준의 충분한 근거가 아니며 `DINO_REPLACEMENT_CANDIDATE`를 확정할 수 없다. Native 512와 Prompt의 효과는 위 paired 결과로 판단한다.','',
'나머지 20개 Parent는 이번 Pilot에서 실행하지 않았다. 후속 확인을 한다면 Prompt·해상도·선택 기준과 새 실행 예산을 먼저 고정해야 한다. 이번 결과를 보고 같은 파일럿에서 Prompt를 추가 수정하거나 Parent를 늘리지 않았다. 장기 SAM2 tracking, Graph 생성, Controller, closed-loop success, 외부 분포는 검증하지 않았다. Production과 Stage4A-2 `PAUSED_SCOPE_REVIEW` 및 Stop Stub은 유지했다.','',
'## 11. 보존·검증과 산출물','',
f'공통 ExperimentLogger를 재사용하고 W&B disabled 로컬 기록만 남겼다. 보호 파일 {len(protected)}개 SHA, 모델/Processor 15개 파일, Python package 목록이 변하지 않았다. 원시 응답→파서 재해석 및 이미지 SHA 재검증 통과; Box IoU {checked}개와 Mask IoU {maskchecks}개 독립 재계산 통과. 추론 종료 후 GPU compute process는 없다. 공통 simulator/logger API 변경 없음. 로컬 공통 logger/artifact 계약 테스트 42개 통과. macOS renderer·CUDA physics는 이 작업에서 실행하지 않았다.', '',
'- [Protocol](protocol.json) · [Input manifest](input_manifest.json) · [Capture audit](capture_verification.json) · [Preflight](preflight_verification.json)',
'- [P0](prompt_P0.txt) · [P1](prompt_P1.txt) · [Prompt identity](grounding_prompt_identity.json) · [Model identity](model_identity.json)',
'- [Raw responses](full_image_grounding_raw_responses.json) · [Predictions CSV](full_image_grounding_predictions.csv) · [Grounding metrics](grounding_evaluation_metrics.json) · [Failure cases](grounding_failure_cases.csv)',
'- [SAM2 metrics](sam2_initialization_metrics.json) · [SAM2 cases](sam2_initialization_cases.csv) · [Latency/VRAM](grounding_latency_vram.json) · [Verification](verification.json)',
'', '![동일 Parent t0 Main/Wrist의 4조건 비교](full_image_grounding_comparison.png)', '',
'그림: 주황 Qwen PEG, 하늘색 Qwen BOX, 초록 GT PEG, 자홍 GT BOX. 반투명 영역은 실제 SAM2 초기 Mask이며 invalid/null에는 Mask가 없다. Qwen Box는 정식 계약을 통과한 경우에만 표시한다. 사후 진단 좌표는 위 별도 CSV에 보존한다. 두 해상도를 같은 크기로 표시한 것은 보고서 비교용이며, 512 입력은 시뮬레이터의 native RGB다.','',
'## Appendix — Frozen P1','', '```text',(R/'prompt_P1.txt').read_text(),'```','']
diagnostic_section=['### 출력 계약과 분리한 사후 좌표 진단', '',
'Primary accepted 결과의 0점은 좌표가 전혀 없었다는 뜻이 아니다. 모든 예측·SAM2 평가를 끝낸 뒤 사후 진단을 추가했다. 완전한 JSON에서 반복 키의 값이 정확히 일치하는 경우에만 좌표를 읽고, extra field는 진단에서만 무시했다. 서로 다른 반복 좌표나 불완전 JSON은 복구하지 않았다. GT로 좌표를 고르거나 보정하지 않았다. 이 파서는 Accepted 파서로 승격하지 않았고 이 좌표로 SAM2를 추가 실행하지 않았다.', '',
'| 조건 | 진단 가능 응답 | PEG 진단 IoU≥0.5 | BOX 진단 IoU≥0.5 | PEG 평균 IoU | BOX 평균 IoU |',
'|---|---:|---:|---:|---:|---:|']
for mode in VARIANTS:
 d=diag['conditions'][mode];a=d['objects'];diagnostic_section.append(f'| {mode} | {d["diagnostic_readable"]}/16 | {a["peg"]["numerator"]}/{a["peg"]["denominator"]} | {a["box"]["numerator"]}/{a["box"]["denominator"]} | {fmt(a["peg"]["mean_bbox_IoU_including_unreadable"])} | {fmt(a["box"]["mean_bbox_IoU_including_unreadable"])} |')
diagnostic_section += ['', '모든 조건에서 가시 분모는 동일하며 Parent당 물체별 4개이므로 이 표의 Image recall과 Parent-macro recall은 같다. 상세 Parent 결과와 진단 가능 공통 subset 비교는 [사후 진단 JSON](posthoc_coordinate_diagnostic.json), 각 좌표는 [진단 CSV](posthoc_coordinate_diagnostic.csv)에 보존했다. 이 진단은 사전 Primary 지표가 아니고 네 Parent의 사후 탐색 결과다.', '',
'P0에서 Native 128→512의 진단 좌표 Recall은 PEG 6→8/16, BOX 3→9/16이었다. P1에서는 PEG 10→13/16, BOX 5→11/16이었다. 이 파일럿을 근거로 고해상도나 P1이 공간 Grounding을 악화시켰다고 해석할 수 없다. 형식 실패를 분리하면 개선 가능성이 보이지만, 512/P1조차 Peg/Box의 운영 Recall 90%에 미달하고 이 좌표는 Mask로 검증되지 않았다.', '',
'정식 Accepted와 진단 값을 섞으면 결론이 왜곡된다. 현재 출력 계약 준수는 큰 병목이다. 동일 값의 반복 키를 포함한 응답 수는 Native128/P0 2개, Native512/P0 9개, Native128/P1 15개, Native512/P1 12개였다. 한 응답에서 동일 반복과 충돌 반복이 함께 나타나면 issue count는 중복될 수 있다. P1/512에는 추가 키 응답도 3개 있었다. 원인 후보는 Prompt/schema/runtime의 상호작용이며, 이번 데이터만으로 학습된 패턴이나 특정 구현 원인을 확정하지 않는다.', '',
'운영 경로의 판단은 `VLM_GROUNDING_INSUFFICIENT`다. 공간 능력에는 개선 신호가 있어 Native 512를 배제할 근거가 되지 않는다. 후속 작업을 한다면 고해상도를 유지하면서 출력 계약을 사전에 고정해 검증하는 것이 우선이며, 이번 결과 뒤에 추가 추론·프롬프트 변경·SAM2 보완 실행은 하지 않았다.', '']
idx=lines.index('## 10. 판단과 범위');lines[idx:idx]=diagnostic_section
(R/'10i_stage4a2_qwen35_native_resolution_prompt.md').write_text('\n'.join(lines))
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in R.rglob('*') if p.is_file() and p.name not in ['artifact_index.json'] and p.suffix!='.tmp' and p.name not in ['infer.log','capture.log','finalize.log']}
wr('artifact_index.json',dict(status='COMPLETE_VERIFIED',files=files))
print('COMPLETE_VERIFIED',len(files),'protected',len(protected),'bbox',checked,'mask',maskchecks,flush=True)
