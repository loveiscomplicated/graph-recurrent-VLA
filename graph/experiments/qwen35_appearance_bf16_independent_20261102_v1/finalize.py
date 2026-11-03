from common import *
import subprocess,collections
import numpy as np
v=rd(R/'verification.json');assert v['stage']=='EVALUATION_COMPLETE'
assert v['model_loads']==v['dino_loads']==1 and v['completed_requests']==v['request_count']
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT']==v['raw_sha256_after_GT']
assert sha(R/'verifier_predictions.csv')==v['predictions_sha256_before_GT']==v['predictions_sha256_after_GT']
assert sha(R/'dino_raw_responses.json')==v['dino_raw_sha256_before_GT']
assert sha(R/'dino_frame_selections.json')==v['dino_selections_sha256_before_GT']
raw=rd(R/'verifier_raw_responses.json')['responses'];pred={(r['candidate_id'],r['condition']):r['strict_label'] for r in raw};requests=rd(R/'request_manifest.json')['requests'];requestmap={(r['candidate_id'],r['condition']):r for r in requests}
for r in raw:
 assert (r['raw_text'].strip() if r['raw_text'].strip() in LABELS else 'INVALID_OUTPUT')==r['strict_label']
 assert r['timing']['initial_forward_cache']['cache_length']==0
for req in requests:assert all(sha(p)==h for p,h in zip(req['images'],req['image_sha256']))
cm=rd(R/'.execution/candidate_manifest.json')
for c in cm['candidates']:
 a=requestmap[(c['id'],'original_three')];b=requestmap[(c['id'],'appearance_three')];assert a['images']==b['images'] and a['image_sha256']==b['image_sha256']
protected={p:sha(p) for p in v['protected_before']};assert protected==v['protected_before']
packages=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True);assert json.loads(packages)==json.loads(v['packages_before'])
for root,name in [(B/'model','model_identity.json'),(MODEL,'dino_model_identity.json')]:
 for name,h in rd(R/name)['checkpoint_hashes'].items():assert sha(root/name)==h
with (R/'selected_boxes.csv').open() as f:selected=list(csv.DictReader(f))
replayed=0;checked=0
for r in selected:
 if r['method'] in ['original_query_gate','appearance_query_gate']:
  cond='original_three' if r['method']=='original_query_gate' else 'appearance_three';pool=sorted([c for c in cm['candidates'] if c['image_id']==r['image_id'] and c['query_object']==r['object'] and pred[(c['id'],cond)]==r['object'].upper()],key=lambda c:(-c['DINO_score'],c['rank']))
  assert r['candidate_id']==(pool[0]['id'] if pool else '');replayed+=1
 a=json.loads(r['box']) if r['box'] else None;b=json.loads(r['GT_bbox']) if r['GT_bbox'] else None
 if not r['bbox_IoU']:continue
 z=float(np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()) if a is not None and b is not None else 0.
 u=float(np.prod(np.subtract(a[2:],a[:2]))+np.prod(np.subtract(b[2:],b[:2]))-z) if a is not None and b is not None else 0.
 assert abs((z/u if u else 0.)-float(r['bbox_IoU']))<1e-12;checked+=1
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert {p.name:sha(p) for p in (R/'.execution').glob('*.py')}==v['source_before']
v.update(stage='COMPLETE_VERIFIED',protected_after=protected,all_protected_SHA_unchanged=True,environments_unchanged=True,model_processor_unchanged=True,parser_requests_verified=len(raw),empty_initial_cache_requests=len(raw),identical_image_pairs=len(cm['candidates']),selection_replays=replayed,independent_selected_IoUs_checked=checked,final_GPU_compute_processes='',production_changed=False);wr('verification.json',v)
m=rd(R/'evaluation_metrics.json');lat=rd(R/'verifier_latency_vram.json');sel=rd(R/'selection_manifest.json');prot=rd(R/'protocol.json')
def rr(x):return f"{x['numerator']}/{x['denominator']} ({100*x['rate']:.1f}%)" if x['rate'] is not None else f"0/{x['denominator']} (N/A)"
def fmt(x):return '-' if x is None else f'{x:.3f}'
new=m['methods']['appearance_query_gate']['image_object_level'];old=m['methods']['original_query_gate']['image_object_level'];base=m['methods']['raw_top1']['image_object_level']
ds=[new[k]['recall']['rate']-old[k]['recall']['rate'] for k in ['peg','box']];status='APPEARANCE_PROMPT_HELDOUT_GAIN' if all(d>=0 for d in ds) and any(d>0 for d in ds) else 'APPEARANCE_PROMPT_HELDOUT_MIXED' if any(d>0 for d in ds) else 'APPEARANCE_PROMPT_HELDOUT_NO_RECALL_GAIN'
wr('experiment_conclusion.json',dict(status=status,prompt_heldout_parents=15,globally_unseen=False,production_ready=False,scope='Candidate verifier, not DINO-free grounding or closed-loop control'))
lines=['# Task 10-I Stage4A-2 — Frozen Appearance Prompt, BF16 Parent-held-out Evaluation','',f'**판정: `{status}`.** 외형 프롬프트 개발에 쓰지 않은 Parent15개, Native512 Camera-Frame60개, 후보{len(cm["candidates"])}개에서 두 프롬프트를 대응 비교했다. PEG Recall은 기존 역할 Prompt {rr(old["peg"]["recall"])} → 외형 Prompt {rr(new["peg"]["recall"])}, BOX는 {rr(old["box"]["recall"])} → {rr(new["box"]["recall"])}였다. 기준은 EXACT Actor visible pixel≥16, 선택 Box IoU≥0.5다.','',
'## 1. 사전 고정 범위와 자료의 독립성','',
'직전 Native512/외형 Prompt 진단에 쓰인 Parent0531/0275/0197/0305를 제외한 기존20개 Parent를 전부 계획 Manifest에 고정했다. 그중5개는 기록된 physics_qpos/qvel 및 Actor pose가 없어 Native512 복원이 불가능했다. 추론 전에 필드 존재만 확인해 제외하고 대체 Parent/Frame을 뽑지 않았다. 나머지15개는 모두 평가했다. 과거 다른 Prompt의 Qwen 실험에는 사용됐으므로 전체 연구에서 미관측 자료나 외부 분포라는 의미의 독립성은 아니다.','',
'| 항목 | 실제 범위 |','|---|---|',
f'| Parent | 계획20, 평가15, 복원 자료 부재5 |',
'| Frame / Camera | 기존 F0=t0, F1=t80 또는 기존 fallback; Main/Wrist 각각 별도 입력 |',
f'| 이미지 / 후보 / Qwen 요청 | 60 / {len(cm["candidates"])} / {len(raw)} |',
'| Qwen | Qwen/Qwen3.5-9B, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a |',
'| 실행 | Vast.ai58576054, RTX3090, 전체BF16, SDPA, cuda:0, offload 없음 |',
'| Generation | Batch1, Greedy, thinking=False, max_new_tokens16, fresh user/cache |',
'| Processor | 동일 Source, min=max262144px, Native512; Crop 종횡비 유지/Patch 배수 반올림 |',
'| DINO | 기존 Tiny Weight/Revision/두 Prompt/threshold0.1, FP32, 120 forwards, Load1회 |',
'| 예산 | Qwen Load1회, SAM2/Training/Rollout/Physics Step/자동Retry/추가Warmup/Prompt 수정0회 |','',
'새 Rollout 대신 기존 상태를 정적으로 복원했다. 환경 Constructor15회와 명시적 Seeded Reset15회가 있었고, 이후 저장 qpos/qvel·Peg/Box pose를 설정해 Camera만 Render했다. 원래128과 Native512 Camera의 Pose/FOV/Mount/Intrinsic scaling 및 TCP/Hole pose를 확인하고 get_obs 전후 물리 상태가 바뀌지 않음을 검증했다. Solver 상태의 동적 Replay를 주장하지 않는다. 두 해상도를 관측했지만 Qwen/DINO 정확도 평가는512만 사용했다.','',
'평가 Parent: '+', '.join(cm['parents'])+'.',
'자료 부재 Parent: '+', '.join(p['parent_id'] for p in sel['unavailable'])+'.','',
'## 2. 입력·Prompt·선택 정책','',
'전체 RGB, 청록색4px 후보 Rectangle 복사본, 각 변10% Margin Crop을 같은 순서로 입력했다. 원본과 Crop은 Marker가 없다. DINO Query/Score/Rank/좌표 숫자/기존 Consumer 결과/GT/Mask를 Qwen 텍스트에 넣지 않았다. 기존 역할 Prompt와 외형 Prompt는 직전 진단의 원문·Preamble을 그대로 복사했다. A/B에서 각 후보의 이미지 경로와 SHA가 동일하다. 결과에 따른 Prompt 수정은 없다.','',
'두 Prompt가 제공하는 사전정보와 지칭·길이 등이 함께 다르므로 외형 Prompt 패키지의 효과를 평가한다. 색상 한 문장만의 인과 효과를 분리한 것은 아니다. 외형 Prompt는 알려진 Asset의 색·형태를 사용하므로 다른 색/모양에 대한 불변 일반화로 표현하지 않는다.','',
'DINO의 두 고정 Query(red and white block/cardboard box)가 threshold0.1 이후 출력한 후보 전체를 검사했다. GT/면적/Top5/Null0.3으로 후보를 먼저 추리지 않았다. Qwen은 좌표를 수정하지 않는다. Primary는 같은 원래 Query 안에서 해당 역할 Label을 받은 후보 중 DINO Score1위를 선택하며 없으면 Null이다. 동점은 원래 Rank로 결정한다. 원시 Top1, 기존 Production 선택 로직, 기존 역할 Prompt의 동일 Query Gate를 함께 비교했다. 별도 Threshold 최적화나 Cross-query 재분류는 없다.','',
'모든 Qwen 응답과 예측 CSV의 SHA를 동결한 뒤 Actor Segmentation을 열어 평가했다. Capture의 Privileged 복원 정보는 RGB-only 추론 Manifest와 분리했다. 과거 연구 결과가 알려진 자료이므로 완전 맹검으로 표현하지 않는다.','',
'## 3. 최종 선택 성능','',
'| 정책 | PEG Recall | BOX Recall | PEG 평균 IoU(Null 포함) | BOX 평균 IoU | 두 물체 동시 성공 |','|---|---:|---:|---:|---:|---:|']
for method,z in m['methods'].items():
 a=z['image_object_level'];lines.append(f'| {method} | {rr(a["peg"]["recall"])} | {rr(a["box"]["recall"])} | {fmt(a["peg"]["IoU_including_null"]["mean"])} | {fmt(a["box"]["IoU_including_null"]["mean"])} | {rr(a["scene"]["both_correct"])} |')
lines+=['','| 정책 | 물체 | 선택 Box Precision | Null | Wrong-object / 출력 |','|---|---|---:|---:|---:|']
for method,z in m['methods'].items():
 for k in ['peg','box']:
  a=z['image_object_level'][k];lines.append(f'| {method} | {k} | {rr(a["selected_precision"])} | {rr(a["null_rate"])} | {rr(a["wrong_object"])} |')
lines+=['','Box Precision은 IoU≥0.5 비율이다. Wrong-object는 IoU<0.5이며 다른 명확한 Actor가 Box 내부 foreground의70% 이상일 때다. 느슨한 Box, 혼합 영역과 Wrong-object를 구분한다. VISIBLE은16px 이상, TINY_VISIBLE은1–15px, ABSENT는0px이며 모든 표본의 실제 분모는 JSON에 있다. 미가시/Tiny 사례가 없으면 해당 오류율의 검증 근거가 없는 것이다.','',
'## 4. 후보 분류·정답 거부·오승인','',
'| Prompt | 형식 준수 | 명확한 Actor 역할 일치 | Parent-macro 역할 일치 |','|---|---:|---:|---:|']
for cond,z in m['candidate_verifier'].items():lines.append(f'| {cond} | {rr(z["format_valid"])} | {rr(z["clear_role_accuracy"])} | {fmt(z["clear_role_parent_macro"])} |')
lines+=['','AMBIGUOUS는 역할 정확도에서 양 조건 모두 제외하며 최종 선택 평가에는 그대로 포함했다. 역할 Proxy는 Box 내부 Actor 구성에 대한 진단이며 혼합 후보의 intended object를 완벽히 정의하지 않는다. 동일 물체 중복 Proposal을 독립 Parent로 세지 않는다.','',
'| Prompt | Query | IoU≥0.5 정답 후보 유지 | IoU<0.5 오답 Box 승인 | 승인 Box 중 IoU≥0.5 |','|---|---|---:|---:|---:|']
for cond,z in m['candidate_verifier'].items():
 for k,a in z['by_query'].items():lines.append(f'| {cond} | {k} | {rr(a["good_box_retention"])} | {rr(a["bad_box_acceptance"])} | {rr(a["accepted_box_precision"])} |')
lines+=['','| Prompt | GT Actor 역할 | 출력 Label 분포 |','|---|---|---|']
for cond,z in m['candidate_verifier'].items():
 for role,c in z['confusion'].items():lines.append(f'| {cond} | {role} | `{json.dumps(c)}` |')
lines+=['','오답 Box 승인에는 목표가 보이지만 Box 범위가 부정확한 경우도 포함된다. 의미 Label 개선만으로 Localization이 개선됐다고 간주하지 않는다. Oracle 후보 풀 Recall은 '+', '.join(k+'='+rr(z) for k,z in m['oracle_pool_recall'].items())+'이며 실제 선택 성능이 아니다.','',
'## 5. Camera·시점·Parent 통계','',
'| 정책 | 구분 | PEG Recall | BOX Recall |','|---|---|---:|---:|']
for method,z in m['methods'].items():
 for group in ['by_camera','by_time']:
  for name,a in z[group].items():lines.append(f'| {method} | {name} | {rr(a["peg"]["recall"])} | {rr(a["box"]["recall"])} |')
lines+=['','Parent별 자료는 evaluation_metrics.json의 per_parent에 있다. 같은 Parent의 두 Frame·두 Camera를 네 독립 표본으로 세지 않았다. Parent-macro Recall:','',
'| 정책 | PEG | BOX |','|---|---:|---:|']
for method,z in m['methods'].items():lines.append(f'| {method} | {fmt(z["parent_macro"]["peg"])} | {fmt(z["parent_macro"]["box"])} |')
lines+=['','| 대조군 | 물체 | 외형 Prompt Recall 차이(Parent-macro) | Parent-cluster Bootstrap 95% CI | 회복 / 회귀 |','|---|---|---:|---|---:|']
with (R/'paired_transitions.csv').open() as f:trs=list(csv.DictReader(f))
for baseline,z in m['paired_effects'].items():
 for k,a in z.items():
  ts=[r for r in trs if r['baseline']==baseline and r['object']==k];rec=sum(r['recovered']=='True' for r in ts);reg=sum(r['regressed']=='True' for r in ts);lines.append(f'| {baseline} | {k} | {fmt(a["recall_delta"]["parent_macro_delta"])} | {a["recall_delta"]["CI95"]} | {rec} / {reg} |')
lines+=['','물리 Parent15개 Cluster Bootstrap10,000회(seed42)이며 탐색적 CI다. 다중 비교 보정이 없고, 작은 수의 Train Parent만으로 희귀 오승인이나 실환경 안정성을 입증하지 않는다.','',
'## 6. 비용과 검증','',f'Qwen Load {lat["load_ms"]/1000:.3f}s, loaded GPU {lat["loaded_memory"]["allocated"]/2**30:.3f}GiB, CPU sampled peak {lat["CPU_process_peak_sampled_bytes"]/2**30:.3f}GiB. 첫 요청 하나만 Cold로 구분했으며 별도 Warmup은 없다. 조건별 Warm 비용:','',
'| Prompt | n | Mean / Median / p95 ms | Peak allocated GiB | 이미지별 전체 후보 요청 합 Mean ms |','|---|---:|---|---:|---:|']
for cond,a in lat['summary'].items():
 t=a['warm_end_to_end_ms'];lines.append(f'| {cond} | {t["n"]} | {t["mean"]:.2f} / {t["median"]:.2f} / {t["p95"]:.2f} | {a["peak_allocated_GiB"]:.3f} | {a["per_image_serial_sum_ms"]["mean"]:.2f} |')
lines+=['','전체 후보를 각자 새로 인코딩하는 exhaustive 진단 비용이다. 후보 간 Vision/KV Cache 공유나 Score순 조기 종료는 구현하지 않았다. 20Hz Control 경로의 비용으로 해석하지 않는다. DINO 비용은 dino_latency_vram.json에 별도 기록됐고 Qwen 수치에 포함하지 않았다. DINO와 Qwen은 순차 실행했으며 동시 상주·SAM2 연계·Tracking 비용을 측정하지 않았다.','',
f'공통 ExperimentLogger(W&B disabled)를 재사용했다. 기존 보호파일{len(protected)}개 SHA/모델·Processor/환경이 전후 동일했다. 모든{len(raw)}개 요청 strict parser·초기 Cache0을 확인했고, 동일 이미지{len(cm["candidates"])}쌍, GT-free 선택 replay{replayed}개, 선택 IoU{checked}개를 별도 재계산했다. 원시 응답의 GT 채점 전후 SHA는 동일하다. CUDA/Renderer는 원격58576054에서 실행했고 macOS에서 실행하지 않았다. Production/Checkpoint/Dataset/Threshold/Stop Stub을 변경하지 않았으며 PAUSED_SCOPE_REVIEW를 유지했다. Commit·외부 게시·후속 Graph/Controller 실행은 없다.','',
'## 7. 해석의 범위','',
'이번 비교는 외형 Prompt의 효과가 개발 사례 밖의 같은 Task Parent로 이어지는지, 그리고 해당 후보 풀에서 DINO 점수1위보다 유용한 선택을 하는지 평가한다. 색상·형태 불변성, 완전히 새로운 배경/Asset, DINO-free Box 생성, SAM2 Mask, 장기 시간적 정체성, Closed-loop Success는 평가하지 않았다. 후보 역할 정확도와 실제 선택 Recall을 함께 보고, 나쁜 Camera/시점의 결과를 제거하지 않는다. 어떤 결과이든 Prompt 추가 수정이나 Parent 추가 실행 없이 이 고정 평가로 종료한다.','',
'## 8. 산출물','',
'- [Protocol](protocol.json), [선정·누락 Manifest](selection_manifest.json), [Capture 검증](capture_verification.json), [입력 Manifest](input_manifest.json), [Prompt Identity](prompt_identity.json), [Model Identity](model_identity.json).',
'- [Qwen 원시 응답](verifier_raw_responses.json), [DINO 원시 후보](dino_raw_responses.json), [후보 평가](candidate_evaluation.csv), [최종 선택](selected_boxes.csv), [실패 사례](failure_cases.csv), [대응 회복·회귀](paired_transitions.csv).',
'- [전체 지표](evaluation_metrics.json), [Qwen 비용](verifier_latency_vram.json), [DINO 비용](dino_latency_vram.json), [보존 검증](verification.json), [Artifact Index](artifact_index.json).','',
'![첫 고정 Parent 네 Camera/시점의 세 정책 비교](comparison.png)','',
'주황=선택Peg, 청록=선택Box, 초록=GT Peg, 자홍=GT Box. 사전 고정 첫 Parent를 표시했으며 성공 사례를 골라 만든 그림이 아니다. SAM2 Mask는 없다.','',
'## 부록 — Frozen appearance prompt','', '```text',rd(R/'prompt_identity.json')['conditions']['appearance_three']['prompt'],'```','']
(R/'10i_stage4a2_qwen35_appearance_bf16_independent_evaluation.md').write_text('\n'.join(lines))
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in R.rglob('*') if p.is_file() and p.name!='artifact_index.json' and p.suffix not in ['.log','.tmp']}
wr('artifact_index.json',dict(status='COMPLETE_VERIFIED',files=files));print('COMPLETE_VERIFIED',status,'artifacts',len(files),'protected',len(protected),flush=True)
