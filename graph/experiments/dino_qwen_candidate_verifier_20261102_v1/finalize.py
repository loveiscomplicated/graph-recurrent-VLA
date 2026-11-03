from common import *
import subprocess,collections
import numpy as np
from PIL import Image
v=rd(R/'verification.json');assert v['stage'] in ['EVALUATION_COMPLETE','COMPLETE_VERIFIED']
m=rd(R/'hybrid_evaluation_metrics.json');lat=rd(R/'verifier_latency_vram.json');prot=rd(R/'protocol.json');cm=rd(R/'.execution/candidate_manifest.json');cmap={c['id']:c for c in cm['candidates']}
assert v['model_loads']==1 and v['completed_requests']==251
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT']==v['raw_sha256_after_GT']
assert sha(R/'verifier_predictions.csv')==v['predictions_sha256_before_GT']==v['predictions_sha256_after_GT']
raw=rd(R/'verifier_raw_responses.json')['responses'];pred={r['id']:r['strict_label'] for r in raw}
for r in raw:assert (r['raw_text'].strip() if r['raw_text'].strip() in LABELS else 'INVALID_OUTPUT')==r['strict_label']
reqs=rd(R/'request_manifest.json')['requests'];assert len(reqs)==251
for req in reqs:
 assert all(sha(p)==h for p,h in zip(req['images'],req['image_sha256']))
protected={p:sha(p) for p in v['protected_before']};assert protected==v['protected_before']
packages=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True);assert json.loads(packages)==json.loads(v['packages_before'])
ident=rd(R/'model_identity.json');assert all(sha(B/'model'/p)==h for p,h in ident['checkpoint_hashes'].items())
# Independent selection replay for primary, from labels + proposal scores only.
with (R/'hybrid_selected_boxes.csv').open() as f:selected=list(csv.DictReader(f))
replayed=0;checks=0
for r in selected:
 if r['method']=='verifier_query_gate':
  pool=sorted([c for c in cm['candidates'] if c['image_id']==r['image_id'] and c['query_object']==r['object'] and pred[c['id']]==r['object'].upper()],key=lambda c:(-c['DINO_score'],c['rank']))
  assert r['candidate_id']==(pool[0]['id'] if pool else '');replayed+=1
 a=json.loads(r['box']) if r['box'] else None;b=json.loads(r['GT_bbox']) if r['GT_bbox'] else None
 if r['bbox_IoU']=='':continue
 inter=0.
 if a is not None and b is not None:inter=float(np.maximum(np.minimum(np.array(a[2:]),b[2:])-np.maximum(a[:2],b[:2]),0).prod());union=float(np.prod(np.subtract(a[2:],a[:2]))+np.prod(np.subtract(b[2:],b[:2]))-inter);ov=inter/union if union else 0.
 else:ov=0.
 assert abs(ov-float(r['bbox_IoU']))<1e-12;checks+=1
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
v.update(stage='COMPLETE_VERIFIED',protected_after=protected,all_protected_SHA_unchanged=True,environments_unchanged=True,model_processor_unchanged=True,label_parser_readback='PASS 251',primary_selection_replay=replayed,independent_selected_IoUs_checked=checks,final_GPU_compute_processes='',new_DINO_forwards=0,new_SAM2_forwards=0,production_changed=False,source_after={p.name:sha(p) for p in (R/'.execution').glob('*.py')});wr('verification.json',v)
def rr(x):return f'{x["numerator"]}/{x["denominator"]} ({100*x["rate"]:.1f}%)' if x['rate'] is not None else f'0/{x["denominator"]} (N/A)'
def fmt(x):return '-' if x is None else f'{x:.3f}'
primary=m['methods']['verifier_query_gate']['image_object_level'];base=m['methods']['raw_top1']['image_object_level'];legacy=m['methods']['legacy']['image_object_level'];cc=m['candidate_verifier']
deltas=[primary[k]['recall']['rate']-base[k]['recall']['rate'] for k in ['peg','box']]
status='CANDIDATE_VERIFIER_LOCAL_GAIN' if all(x>=0 for x in deltas) and any(x>0 for x in deltas) else 'CANDIDATE_VERIFIER_MIXED' if any(x>0 for x in deltas) else 'CANDIDATE_VERIFIER_NO_RECALL_GAIN'
wr('experiment_conclusion.json',dict(status=status,comparison='Primary query-gate vs raw Top1, same proposal pools',recall_delta=dict(zip(['peg','box'],deltas)),independent_generalization=False,production_ready=False))
lines=['# Task 10-I Stage4A-2 — DINO → Qwen Candidate Verifier', '',
f'**판정: `{status}`.** 512 원본의 저장된 DINO 후보 251개를 Qwen3.5-9B로 검사했다. 단일 Label 형식 통과는 {rr(cc["strict_format_valid"])}. 동일 원시 후보 풀에서 Score Top-1 → Qwen 검사 후 선택의 PEG Recall은 {rr(base["peg"]["recall"])} → {rr(primary["peg"]["recall"])}, BOX Recall은 {rr(base["box"]["recall"])} → {rr(primary["box"]["recall"])}였다. 기준은 visible pixel≥16, Box IoU≥0.5다. 기존 Train Parent 4개/Camera-Frame 16개를 재사용한 파일럿이며 독립 일반화나 Production 승인을 뜻하지 않는다.', '',
'## 1. 연구 질문과 고정 구성', '',
'Grounding DINO가 후보 좌표를 생성하고 Qwen이 후보의 기능적 역할을 검사하면, 자유 좌표·JSON 생성 부담을 줄이고 DINO의 잘못된 후보 선택을 보완할 수 있는지 평가했다. Qwen은 새 좌표를 만들거나 Box를 수정하지 않는다.', '',
'- Vast.ai **58576054 / RTX3090 24GB**. `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 전체 BF16·SDPA·cuda:0·offload 없음.',
'- 512 native RGB 16개: Parent 0531/0275/0197/0305, t0/t80, Main/Wrist. 새 Simulator Reset/Render/Step 없음.',
'- 직전 동결 DINO 원시 후보 251개 전체를 사용했다. Box/text threshold0.1 이후 후보이며 면적25%/Top5/Null0.3이나 GT로 Qwen 검사 대상을 줄이지 않았다. 새 DINO forward 없음.',
'- 기존 성공 Functional-role Prompt SHA `691ea9e22ec710dad77084db6aa812455d525190f7d6f589821bfaf082889a06` 그대로. PEG/BOX/ROBOT/OTHER/UNCERTAIN 중 정확히 단어 하나만 출력한다. 양끝 공백 제거 외 복구·JSON/fence 추출은 하지 않는다.',
'- Batch1, greedy, enable_thinking=False, max_new_tokens=16, use_cache=True. 모델 Load1회, 요청251회, warmup/retry/sampling/Prompt 수정0회.',
'- 과거 3후보의 성공 실험은 NF4·128 원본/256 Processor였다. 이번은 BF16·512이므로 그 결과와의 차이를 Prompt 하나의 인과 효과로 주장하지 않는다.', '',
'## 2. 입력과 평가 장벽', '',
'각 요청에는 ①원본 전체 RGB ②중립 청록색(0,180,220) 4px Rectangle을 그린 복사본 ③Box 각 변10% margin Crop 순서로 이미지3장과 기존 Prompt를 제공했다. Marker 4px는 128의1px와 같은 상대 두께다. Crop은 화면과의 교집합을 floor/ceil로 잘랐지만 원래 후보 좌표는 바꾸지 않았다. 전체 장면과 clean Crop은 Marker로 덮지 않았다.', '',
'Qwen에 DINO query/label/score/rank, 후보 좌표 숫자, Consumer decision, GT Actor/Segmentation/Pose, SAM2 mask를 전달하지 않았다. Request ID/파일명은 orchestration용이며 모델 text에 넣지 않았다. Processor는 각 이미지 min=max262144 pixels, 원본/표시 grid는 [1,32,32], Crop은 종횡비 유지 Patch 반올림이다. 한 요청에 한 후보만 검사했다.', '',
'모든 원시 응답251개와 예측CSV를 저장·SHA 동결한 뒤 GT sidecar와 후보 메타데이터로 평가했다. 과거 이미지·GT/성적은 이미 알려져 있으므로 연구자 맹검으로 표현하지 않는다. 모델 inference 프로그램은 RGB-only Request Manifest만 읽는다. DINO/SAM2/Qwen의 과거 동결 산출물은 보존했다.', '',
'## 3. 선택 정책과 대조군', '',
'기존 선택 Box만 검사하는 legacy_gate에서는 기존 오답 Box 선택13개 중10개를 제거했고, 기존 정답 선택16개는 모두 유지했다. 이는16개 이미지의29개 Object 출력에 대한 관측이며 독립 정답16개 Parent가 아니다. PEG Precision은8/13→8/8, BOX는8/16→8/11이었고, Miss를 복구하지 않는 이 rejection-only 경로의 Recall은 그대로였다.\n\n모든 정책은 예측을 보기 전에 protocol.json에 고정했다. 순위는 기존 DINO score 내림차순, 동점은 원래 rank로 결정하며 GT는 선택에 사용하지 않는다.', '',
'| 정책 | 정의 |','|---|---|',
'| legacy | 저장된 기존 area25%/Top5/Null0.3/joint0.3 결과 |',
'| raw_top1 | 각 원래 Query의 전체 원시 후보에서 최고 score |',
'| legacy_gate | 기존 선택 Box를 Qwen이 해당 역할로 판단할 때만 유지, 교체 없음 |',
'| verifier_query_gate (Primary) | 같은 Query의 원시 후보 중 Qwen이 요청 역할로 판단한 후보에서 최고 DINO score, 없으면 null |',
'| verifier_union_roles (Secondary) | 두 Query의 전체 후보를 합쳐 Qwen의 역할 Label별 최고 DINO score 선택 |', '',
'Primary와 raw_top1은 후보 풀이 같고 의미 검사 유무가 차이다. legacy 대비 변화에는 면적/Top5/Null/joint 제한을 사용하지 않는 새 선택 경로의 효과도 포함된다. 따라서 legacy 대비 개선 전부를 Qwen의 효과로 돌리지 않는다. 독립 Object별 선택이며 새 score/overlap threshold를 만들지 않았다. Secondary는 사전 정의한 대안으로, 결과에 따라 선택한 Production 정책이 아니다. 모든 정책은 실험 전용이며 기존 Production을 수정하지 않았다.', '',
'## 4. 실제 Box 선택 결과', '',
'| 정책 | PEG Recall | BOX Recall | PEG 평균 IoU(Null 포함) | BOX 평균 IoU(Null 포함) | 두 물체 동시 성공 |', '|---|---:|---:|---:|---:|---:|']
for method,z in m['methods'].items():
 a=z['image_object_level'];lines.append(f'| {method} | {rr(a["peg"]["recall"])} | {rr(a["box"]["recall"])} | {fmt(a["peg"]["IoU_including_null"]["mean"])} | {fmt(a["box"]["IoU_including_null"]["mean"])} | {rr(a["scene"]["both_correct"])} |')
lines += ['', '| 정책 | 물체 | 출력 Box Precision | Null | Wrong-object / 출력 |', '|---|---|---:|---:|---:|']
for method,z in m['methods'].items():
 for k in ['peg','box']:
  a=z['image_object_level'][k];lines.append(f'| {method} | {k} | {rr(a["selected_precision"])} | {rr(a["null_rate"])} | {rr(a["wrong_object"])} |')
lines += ['', 'Box Precision은 선택 Box의 IoU≥0.5 비율이며 의미 Label 정확도와 다르다. Null은 가시 목표의 Miss로 계산한다. 미가시 물체의 Hallucination 분모는 JSON에 별도로 보존했고 분모가 없으면 N/A다.', '',
'## 5. 검사기 형식과 의미 분류', '',
f'전체 strict 형식 통과 {rr(cc["strict_format_valid"])}; 출력 분포 `{json.dumps(cc["labels"])}`. GT 역할이 명확한 후보의 분류 일치율 {rr(cc["clear_role_accuracy"])}; Parent-macro 분류 일치율 {fmt(cc["parent_macro_clear_role_accuracy"])}. GT role 분포 `{json.dumps(cc["GT_roles"])}`.', '',
'GT 역할 진단은 Box 내부 peg/box/robot foreground가16px 이상이고 한 Actor가 foreground≥70%를 차지하면 그 역할, foreground<16이면 OTHER, 나머지는 AMBIGUOUS로 정의했다. 이는 intended object의 완벽한 정답을 증명하는 기준이 아니라 Actor 구성 기반의 진단이다. Mixed 후보는 분류 정확도 분모에서 제외하되 전체 선택 Recall에는 계속 포함한다. 동일 물체의 중복 후보 때문에 251개를 독립 Scene251개로 세지 않는다.', '',
'| Query | 후보 수 | IoU≥0.5 정답 후보 | 정답 후보 유지율 | 오답 Box 승인율 | 승인 Box 중 정답 | UNCERTAIN |', '|---|---:|---:|---:|---:|---:|---:|']
for k,z in cc['by_query'].items():lines.append(f'| {k} | {z["proposals"]} | {z["good_boxes"]} | {rr(z["good_box_retention"])} | {rr(z["bad_box_acceptance"])} | {rr(z["accepted_box_precision"])} | {rr(z["UNCERTAIN"])} |')
lines += ['', '여기서 오답 Box 승인은 IoU<0.5인 Box에 해당 역할을 출력한 경우다. 그 Box 안에 실제 목표 일부가 있어 의미 분류는 맞고 Box 범위만 부정확할 수도 있다. Qwen 역할 분류가 정확하다고 Box geometry가 자동으로 수정되지 않는다.', '',
'| GT role | Qwen 출력 분포 |','|---|---|']
for k,z in cc['confusion'].items():lines.append(f'| {k} | `{json.dumps(z)}` |')
lines += ['', '## 6. Camera/시점과 Parent 통계', '',
'| Primary 구분 | PEG Recall | BOX Recall |','|---|---:|---:|']
for group in ['by_camera','by_time']:
 for name,a in m['methods']['verifier_query_gate'][group].items():lines.append(f'| {name} | {rr(a["peg"]["recall"])} | {rr(a["box"]["recall"])} |')
lines += ['', 'Primary Parent-macro Recall: '+', '.join(k+'='+fmt(m['methods']['verifier_query_gate']['parent_macro'][k]) for k in ['peg','box'])+'. 물리 Parent 4개 단위 Bootstrap10,000회(seed42), 탐색적95% CI다. 다중 비교 보정이 없고 작은 파일럿에서의 CI/0오류가 일반화·희귀 오류 안전성을 보증하지 않는다.', '',
'| 대조군 | 물체 | Recall 차이(Primary−대조군) | 탐색적 95% CI | 평균 IoU 차이 |','|---|---|---:|---|---:|']
for baseline,z in m['paired_effects'].items():
 for k,e in z.items():lines.append(f'| {baseline} | {k} | {fmt(e["recall_delta"]["parent_macro_delta"])} | {e["recall_delta"]["exploratory_CI95"]} | {fmt(e["IoU_delta"]["parent_macro_delta"])} |')
lines += ['', '개별 Parent·회복/회귀는 [대응 변화 CSV](hybrid_transitions.csv)와 [전체 지표](hybrid_evaluation_metrics.json)에 보존했다.', '',
'## 7. 실제 비용', '',
f'Model load {lat["load_ms"]/1000:.3f}s, loaded allocated {lat["loaded_memory"]["allocated"]/2**30:.3f} GiB, peak inference allocated {lat["summary"]["peak_allocated_bytes"]/2**30:.3f} GiB. CPU sampled peak {lat["CPU_process_peak_sampled_bytes"]/2**30:.3f} GiB(50ms RSS). 모델 Hash 검사로 OS file cache는 warm일 수 있다.', '',
'| 비용 | n | Mean ms | Median ms | p95 ms |','|---|---:|---:|---:|---:|']
for name,key in [('Warm 후보1개 E2E','warm_end_to_end_ms'),('Preprocessing','preprocessing_ms'),('Vision CUDA','vision_ms'),('Prefill−Vision 근사','prefill_excluding_vision_approx_ms'),('Decode forward CUDA','decode_forward_ms'),('이미지별 전체 후보 직렬 요청 합','serial_sum_per_image_ms')]:
 z=lat['summary'][key];lines.append(f'| {name} | {z["n"]} | {fmt(z["mean"])} | {fmt(z["median"])} | {fmt(z["p95"])} |')
lines += ['', f'Cold 첫 요청 {lat["summary"]["cold"]["end_to_end_ms"]:.2f}ms는 Warm에서 제외했다. 이미지별 합은 후보마다 scene/marked/crop을 새로 처리한 비용의 합이며 이미지 작업을 연속 실행한 벽시계 값이 아니다. 한 이미지의 Vision feature/KV cache를 후보들 사이에 재사용하는 최적화는 구현하지 않았다. 요청마다 입력·세 이미지 Vision·Prefill이 새로 필요하며 20Hz 경로의 지연으로 볼 수 없다. 초기 Discovery/필요 시 검사 용도에서도 전체 후보 검사 비용을 고려해야 한다. 이 값은 exhaustive 평가 비용이다. 점수순으로 검사해 첫 합격 후 중단하는 온라인 구현이나 최적화된 최소 배포 비용은 측정하지 않았다.', '',
'이 실험에서는 저장 DINO를 재사용했으므로 Detector 비용은 위 합에 포함하지 않았다. 이전 동일 이미지 DINO 2프롬프트 비용은 별도 보고서에 있다. reference PyTorch Gated DeltaNet/causal-conv fallback 환경을 유지했고 새 kernel을 설치하지 않았다. DINO/SAM2/Qwen 동시 GPU 상주나 실제 온라인 Scheduling은 검증하지 않았다.', '',
'## 8. 해석과 한계', '',
'단일 Label은 자유 좌표·JSON 생성과 다른 출력 계약이며 실제 형식 통과율로 판단한다. 높은 분류 일치율이나 일부 후보 회복만으로 전체 시스템 안정성을 주장하지 않는다. Primary의 raw Top1 대비 이득과 잘못된 승인·정답 거부·geometry 오류를 함께 평가해야 한다. 고득점 오답이 같은 역할 Label을 받으면 최고 DINO score tie-break는 계속 실패할 수 있다. 후보 좌표가 부정확하거나 올바른 후보를 모두 거부하면 null/오류가 남는다.', '',
'이 결과는 이미 관측했던4개 Train Parent다. 추가 Parent/Prompt/threshold를 결과 뒤에 최적화하지 않았다. SAM2 초기 Mask·장기 Tracking·Object Graph·Recurrent controller·Closed-loop 성공은 미실행이다. 기존 Stage4A-2 PAUSED_SCOPE_REVIEW 및 Stop Stub을 유지하며 Production 통합·Git Commit·외부 게시를 하지 않았다.', '',
'## 9. 보존과 산출물', '',
f'공통 ExperimentLogger를 W&B disabled 로컬 기록으로 재사용했다. 보호 파일 {len(protected)}개 SHA, Qwen 모델/Processor15개 파일, package 목록이 전후 동일했다. 응답251개 strict parser 재해석, Primary 선택32개 GT-free replay, 선택 Box IoU {checks}개 독립 재계산을 통과했다. 요청 이미지3장 모두 SHA를 검증했다. 기존 역할 Prompt·모델·DINO/Qwen 과거 결과를 변경하지 않았다. GPU compute process는 종료 후 없었다. macOS GPU·Simulator·renderer는 실행하지 않았다.', '',
'- [Protocol](protocol.json) · [Prompt identity](prompt_identity.json) · [Model identity](model_identity.json) · [RGB-only Request manifest](request_manifest.json)',
'- [원시 응답](verifier_raw_responses.json) · [단일 Label 예측](verifier_predictions.csv) · [후보 전체 평가](candidate_evaluation.csv)',
'- [선택 Box](hybrid_selected_boxes.csv) · [전체 지표](hybrid_evaluation_metrics.json) · [실패 사례](hybrid_failure_cases.csv) · [Latency/VRAM](verifier_latency_vram.json) · [보존 검증](verification.json)', '',
'![사전 선정 Parent의 기존 선택과 Primary 검사 후 선택](hybrid_verifier_comparison.png)', '',
'주황=선택 PEG, 하늘색=선택 BOX, 초록=GT PEG, 자홍=GT BOX. 후보별 Marker/Crop은 별도 원자료이며 이 그림은 첫 고정 Parent의 네 Camera/Frame을 결과와 무관하게 표시한다. SAM2 Mask는 없다.', '',
'## Appendix — Frozen verifier prompt', '', '```text',(R/'verifier_prompt.txt').read_text(),'```','']
time_section=['### 초기 Discovery와 후기 Re-grounding을 분리한 관측', '',
'| 시점 | 정책 | PEG Recall | BOX Recall |', '|---|---|---:|---:|']
for f in ['F0','F1']:
 for method in ['legacy','raw_top1','verifier_query_gate']:
  z=m['methods'][method]['by_time'][f];time_section.append(f'| {f} | {method} | {rr(z["peg"]["recall"])} | {rr(z["box"]["recall"])} |')
time_section += ['', 't0에서는 Primary가 PEG/BOX 모두 8/8이지만 t80에서는 PEG 0/8·BOX 3/8이었다. 이는 4개 Parent의 두 Camera씩을 센 값이며 독립 Parent8개가 아니다. 초기 Object Discovery용 검사기로 후속 검증 가치가 있지만 Later Re-grounding을 같은 성능으로 가정할 수 없다. 이 시점 구분은 사전에 정한 F0/F1 분석이며 t0 성적만으로 전체 결과를 대체하지 않는다.', '',
'Actor 구성 기반 진단에서 PEG 우세 후보81개 중53개와 ROBOT 우세79개 중49개를 BOX로 출력했다. BOX 출력187/251, UNCERTAIN0개라는 편향이 관측됐다. 형식 통과100%와 안정적인 의미 검증은 별개다. 특히 가림·삽입·파지 상태에서의 역할 혼동, marked region과 crop 문맥 해석은 원인 후보이며 이번 결과만으로 특정 문장이나 구현을 원인으로 확정하지 않는다.', '']
idx=lines.index('## 7. 실제 비용');lines[idx:idx]=time_section
lines.insert(3,'**초기/후기 차이:** t0 Primary PEG/BOX 모두 8/8, t80 PEG0/8·BOX3/8. 초기 발견의 국소 개선과 후기 의미 검증 실패가 함께 관측됐다.\n')
(R/'10i_stage4a2_dino_qwen_candidate_verifier.md').write_text('\n'.join(lines))
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in R.rglob('*') if p.is_file() and p.name not in ['artifact_index.json','infer.log','prepare.log','evaluate.log','finalize.log'] and p.suffix!='.tmp'}
wr('artifact_index.json',dict(status='COMPLETE_VERIFIED',files=files));print('COMPLETE_VERIFIED',status,'artifacts',len(files),'protected',len(protected),'IoUs',checks,flush=True)
