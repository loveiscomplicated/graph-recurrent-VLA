from common import *
import subprocess
import numpy as np

v=rd(R/'verification.json');assert v['stage']=='EVALUATION_COMPLETE' and v['model_loads']==1 and v['completed_requests']==298
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT']==v['raw_sha256_after_GT']
assert sha(R/'verifier_predictions.csv')==v['predictions_sha256_before_GT']==v['predictions_sha256_after_GT']
protected={p:sha(p) for p in v['protected_before']};assert protected==v['protected_before']
assert json.loads(subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True))==json.loads(v['packages_before'])
assert all(sha(B/'model'/p)==h for p,h in rd(R/'model_identity.json')['checkpoint_hashes'].items())
raw=rd(R/'verifier_raw_responses.json')['responses'];pred={(r['candidate_id'],r['condition']):r['strict_label'] for r in raw}
assert len(raw)==len(pred)==298
for r in raw:
 assert (r['raw_text'].strip() if r['raw_text'].strip() in LABELS else 'INVALID_OUTPUT')==r['strict_label']
 assert r['timing']['initial_forward_cache']['cache_length']==0 and r['timing']['vision_forward_count']==1
for req in rd(R/'request_manifest.json')['requests']:assert all(sha(p)==h for p,h in zip(req['images'],req['image_sha256']))
cm=rd(R/'.execution/candidate_manifest.json');replayed=0;ious=0
with (R/'selected_boxes.csv').open() as f:selected=list(csv.DictReader(f))
for r in selected:
 if r['method'] in ['current_only','with_initial_RGB']:
  pool=sorted([c for c in cm['candidates'] if c['image_id']==r['image_id'] and c['query_object']==r['object'] and pred[(c['id'],r['method'])]==r['object'].upper()],key=lambda c:(-c['DINO_score'],c['rank']))
  assert r['candidate_id']==(pool[0]['id'] if pool else '');replayed+=1
 a=json.loads(r['box']) if r['box'] else None;b=json.loads(r['GT_bbox']) if r['GT_bbox'] else None
 if a is None or b is None:ov=0.
 else:
  inter=float(np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod());union=float(np.prod(np.subtract(a[2:],a[:2]))+np.prod(np.subtract(b[2:],b[:2]))-inter);ov=inter/union if union else 0.
 assert abs(ov-float(r['bbox_IoU']))<1e-12;ious+=1
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert v['source_before']=={p.name:sha(p) for p in (R/'.execution').glob('*.py')}
v.update(stage='COMPLETE_VERIFIED',protected_after=protected,all_protected_SHA_unchanged=True,environments_unchanged=True,model_processor_unchanged=True,strict_labels_reparsed=298,initial_cache_empty_verified=298,selection_replay=replayed,independent_selected_IoUs_checked=ious,final_GPU_compute_processes='',production_changed=False);wr('verification.json',v)
m=rd(R/'temporal_reference_metrics.json');lat=rd(R/'verifier_latency_vram.json');pid=rd(R/'prompt_identity.json')
def rr(x):return f'{x["numerator"]}/{x["denominator"]} ({100*x["rate"]:.1f}%)' if x['rate'] is not None else f'0/{x["denominator"]} (N/A)'
lines=['# Task 10-I Stage4A-2 — Initial RGB Reference Paired Comparison','','## 1. 고정 실험과 비교 조건','',
'Qwen3.5-9B가 후기 후보의 역할을 혼동하는 원인에 초기 시각 맥락 부재가 관련되는지 검사했다. Vast.ai58576054 RTX3090에서 revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, 전체 BF16·SDPA·cuda:0·offload 없음·greedy·thinking=False·max_new_tokens16·Batch1을 유지했다. 모델1회 Load, 후보149개×2조건=298요청. 기존4개 Train Parent(0531/0275/0197/0305)의 t80 Main/Wrist8장에 있는 모든 보존 DINO 후보를 사용했다. GT·이전 성적에 따른 후보 필터링이나 신규 DINO/SAM2 실행은 없다.','',
'- current_only: 현재 전체 RGB + 현재 표시 RGB + 현재 Crop.','- with_initial_RGB: 같은 세 이미지에 같은 Parent·같은 Camera의 t0 전체 RGB를 네 번째 이미지로 추가.','- 초기 RGB는 native512 원본이며 Mark/Crop/GT/Label/이전 답변/물체 대응 정답을 제공하지 않았다. 전체 장면이 추가되므로 두 시점만으로 물체 연속성을 모델이 스스로 추론해야 한다.','',
'두 조건의 모든 텍스트는 동일하다. 기존 기능적 역할 Prompt 원문은 그대로이고, 선택적 네 번째 이미지 설명을 두 조건에 똑같이 추가했다. 따라서 인과 비교는 새 current_only와 with_initial_RGB 사이에서 수행한다. 과거 검사기와는 안내 문구가 달라졌으므로 그 결과는 별도 재현 대조다. Prompt sweep/결과 후 수정은 없다.','',
f'역할 Prompt SHA `{pid["semantic_prompt_sha256"]}`. 공통 Preamble SHA `{pid["shared_preamble_sha256"]}`. 공통 추가 안내:','', '```text',pid['temporal_note'],'```','',
'각 요청은 새 user 메시지 한 개이며 이전 응답이나 cache를 전달하지 않았다. 실행 Hook으로 최초 Forward의 cache length가0이고 현재 입력 전체를 Prefill했음을298/298 확인했다. 요청은 고정 Hash 순서로 두 조건을 교차했다. 모든 원시 응답과 예측CSV를 SHA 동결한 후 GT Actor Segmentation을 열었다. 이미 알려진 개발 장면의 기전 진단이며 연구자 맹검·독립 일반화 평가가 아니다.','',
'## 2. 동일 후보 풀에서 실제 선택 Recall','',
'Strict Label이 Query 역할과 같은 후보 중 최고 기존 DINO score를 선택하고 없으면 null이다. 기존 순위·좌표를 유지하며 면적/Top5/Null-score/joint 제한은 추가하지 않았다. 성공은 visible≥16px, Box IoU≥0.5다.','',
'| 방법 | PEG Recall | BOX Recall | PEG 평균 IoU, null 포함 | BOX 평균 IoU, null 포함 | 두 물체 동시 성공 |','|---|---:|---:|---:|---:|---:|']
for method,z in m['methods'].items():
 a=z['image_level'];lines.append(f'| {method} | {rr(a["peg"]["recall"])} | {rr(a["box"]["recall"])} | {a["peg"]["mean_IoU_including_null"]["mean"]:.4f} | {a["box"]["mean_IoU_including_null"]["mean"]:.4f} | {rr(a["scene"]["both_correct"])} |')
lines+=['','raw_top1과 historical_verifier는 이전 동결 출력을 동일8장으로 제한한 값이다. 초기 t0의8/8 결과를 이번 후기 Recall 분모에 섞지 않았다.','',
'## 3. 정답 후보 거부·오답 승인 및 Camera별 결과','','| 조건 | Query | 정답 후보 유지 | 오답 후보 승인 |','|---|---|---:|---:|']
for condition in ['current_only','with_initial_RGB']:
 z=m['candidates'][condition]
 for obj,q in z['by_query'].items():lines.append(f'| {condition} | {obj} | {rr(q["good_box_retention"])} | {rr(q["bad_box_approval"])} |')
for condition in ['current_only','with_initial_RGB']:
 z=m['candidates'][condition];lines+=['',f'{condition}: 형식 {rr(z["format_validity"])}, 명확 Actor-role 일치 {rr(z["clear_role_accuracy"])}, Label 분포 `{json.dumps(z["labels"])}`.','']
lines+=['후보 수는 중복 영역을 포함하며 독립 Parent 수가 아니다. 역할 Label이 맞아도 느슨한 Box는 IoU 기준에서 오답이다. 명확 Actor-role은 foreground≥16px에서 한 Actor가70% 이상인 경우이며 intended-object의 완벽한 정답을 뜻하지 않는다.','','| 조건 | Camera | PEG Recall | BOX Recall |','|---|---|---:|---:|']
for condition in ['current_only','with_initial_RGB']:
 for camera,z in m['methods'][condition]['by_camera'].items():lines.append(f'| {condition} | {camera} | {rr(z["peg"]["recall"])} | {rr(z["box"]["recall"])} |')
lines+=['','## 4. 사전 지정 회귀 사례와 재현성','','| 후보 | 과거 Label | current_only | with_initial_RGB |','|---|---|---|---|']
for r in m['specific_regressions']:lines.append(f'| {r["candidate_id"]} | {r["historical"]} | {r["current_only"]} | {r["with_initial_RGB"]} |')
lines+=['',f'current_only의 과거149개 Label 재현 일치 {rr(m["historical_control_agreement"])}. 공통 안내 문구가 있으므로 과거 대비 변화를 초기 이미지 효과로 계산하지 않는다. 두 새 조건 사이 Label 변화 {m["changed_labels"]}/149. 전체 전이는 지표JSON과 CSV에 보존했다.','',
'![초기 참고 영상과 두 조건의 후기 선택](temporal_reference_comparison.png)','','왼쪽은 초기 참고 RGB, 가운데·오른쪽은 현재 RGB에 사후 평가 Box를 표시한 그림이다. 주황=선택PEG, 하늘색=선택BOX, 초록=GT PEG, 자홍=GT BOX. SAM2 Mask는 없다. 두 회귀 사례는 사전에 지정한 진단 대상이며 대표 표본으로 해석하지 않는다.','',
'## 5. Parent 단위 통계','','독립 집계 단위는 물리 Parent4개다. 각 Parent의 두 Camera Recall을 평균한 paired 차이에 cluster bootstrap10,000회, seed42를 적용했다. 탐색적95% CI이고 다중 비교 보정이 없으며 작은 표본이다.','','| Object | Parent-macro Recall 차이, 추가−현재 | 탐색적95% CI |','|---|---:|---|']
for obj,z in m['effects'].items():lines.append(f'| {obj} | {z["recall_delta"]["parent_macro_delta"]:.4f} | {z["recall_delta"]["exploratory_CI95"]} |')
lines+=['','## 6. 실제 비용','','| 조건 | Warm n | Mean ms | Median ms | p95 ms | Peak allocated GiB |','|---|---:|---:|---:|---:|---:|']
for condition,z in lat['summary'].items():
 d=z['warm_end_to_end_ms'];lines.append(f'| {condition} | {d["n"]} | {d["mean"]:.2f} | {d["median"]:.2f} | {d["p95"]:.2f} | {z["peak_allocated_bytes"]/1024**3:.3f} |')
lines+=['',f'Model Load {lat["load_ms"]:.2f}ms, global cold 첫 요청 {lat["global_cold_first_request"]["condition"]} {lat["global_cold_first_request"]["end_to_end_ms"]:.2f}ms는 위 Warm에서 제외했다. 각 이미지에512² pixel budget을 적용해 초기 RGB는256개 visual token을 추가한다. 세 이미지의 첫 두 grid는[1,32,32]이고 Crop은 종횡비를 유지한다. 입력 Token·Vision·Prefill·Decode 상세는 Latency JSON에 있다. 기존 reference kernel fallback 환경을 유지했다. 모든 요청은 새 Vision/Prefill이며 과거 cache 재사용 최적화는 없다. DINO는 저장 결과를 썼으므로 비용에 포함하지 않았다.','',
'## 7. 해석 범위와 보존','',
'초기 RGB 추가로 성능이 변하면 이 고정 입력 구성에서 참고 영상 제공의 효과를 지지한다. 다만 추가 이미지 유무는 시각 Token 수와 문맥량도 함께 바꾸므로, 개선만으로 시간적 정체 추적이 내부적으로 성공했다고 증명할 수 없다. 반대로 개선이 없더라도 모든 시간 정보 방식이 무효라는 뜻은 아니다. 이번에는 초기 RGB 한 장만 제공했고, 중간 궤적·초기 확정 Label·Object correspondence·명시적 Tracking은 제공하지 않았다. Parent 추가나 새로운 Prompt 탐색은 수행하지 않았다.','',
f'기존 보호 파일{len(protected)}개 SHA와 Qwen 모델/Processor 파일·package 목록이 전후 동일했다. Strict Label298개 재해석, 빈 최초 cache298회, GT-free 선택 재현{replayed}개, 선택 IoU 독립 재계산{ious}개를 통과했다. 공통 ExperimentLogger를 W&B disabled로 재사용했다. Production·PAUSED_SCOPE_REVIEW·Stop Stub을 유지했다. 새 Simulator/Training/SAM2/Graph/Controller/Closed-loop 실행이나 Git Commit·외부 게시는 없다. 종료 후 GPU compute process가 없었다. macOS에서는 GPU/Simulator를 실행하지 않았다.','',
'## 8. 산출물','','- [Protocol](protocol.json), [Prompt identity](prompt_identity.json), [Model identity](model_identity.json), [Input manifest](input_manifest.json), [Request manifest](request_manifest.json)','- [원시 응답](verifier_raw_responses.json), [Label 예측](verifier_predictions.csv), [후보 평가](candidate_evaluation.csv), [Label 변화](paired_label_changes.csv)','- [선택 Box](selected_boxes.csv), [전체 지표](temporal_reference_metrics.json), [Latency/VRAM](verifier_latency_vram.json), [Verification](verification.json)','']
(R/'10i_stage4a2_qwen35_temporal_reference.md').write_text('\n'.join(lines))
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in R.rglob('*') if p.is_file() and p.name!='artifact_index.json' and not p.name.endswith('.log') and p.suffix!='.tmp'}
wr('artifact_index.json',dict(status='COMPLETE_VERIFIED',files=files));print('TEMPORAL_COMPLETE_VERIFIED',len(files),'artifacts',len(protected),'protected',flush=True)
