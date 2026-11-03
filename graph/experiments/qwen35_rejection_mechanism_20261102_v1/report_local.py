"""Write the full Korean report from the frozen diagnostic and independent audit."""
from pathlib import Path
import json,hashlib,re,collections,numpy as np
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'outputs/task10i/stage4a2/qwen35_rejection_mechanism_20261102_v1';REPORT=ROOT/'docs/visual-graph-integration/10i_stage4a2_qwen35_rejection_mechanism.md'
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=rd(R/'mechanism_metrics.json');v=rd(R/'verification.json');lv=rd(R/'local_verification.json');lat=rd(R/'verifier_latency_vram.json');prot=rd(R/'protocol.json');cases=m['cases'];prompts=rd(R/'prompt_identity.json')['conditions'];assert lv['status']=='PASS'
prefix='../../outputs/task10i/stage4a2/qwen35_rejection_mechanism_20261102_v1/'
def rr(z):return f"{z['numerator']}/{z['denominator']}"
def link(title,path):return '['+title+']('+prefix+path+')'
groups=['peg_rejected','peg_matched_initial_control','box_bbox_rejected','box_matched_initial_control','robot_control']
lines=['# Task 10-I Stage4A-2 — Qwen이 정답 후보를 거부하는 이유: 통제 진단','',
'작성일: 2026-11-02. `COMPLETE_VERIFIED`. 알려진 오류를 골라 원인을 좁힌 진단이며 새 독립 성능 평가가 아니다.','',
'## 1. 결론과 증거의 강도','',
'거부는 두 종류로 나뉜다. **Peg에서는 후기 부분 외형을 BOX/ROBOT로 분류하는 실제 오류가 남아 있다. Box에서는 좌표상 정답인 Proposal에 다른 물체가 많이 포함돼, 요구한 Query 역할과 후보의 주 물체 분류가 충돌한 사례가 섞여 있다.** 두 문제를 모두 “물체를 못 봐서 거부”라고 묶은 설명은 부정확했다.','',
'Main 후기의 흰색·회색 Peg 끝부분을 모델은 자유 묘사에서도 회색 Block/Cube와 옆의 검은 부분으로 기술했다. 색·형태를 전혀 읽지 못한다는 근거는 없다. 그러나 그 외형을 같은 작업물의 PEG 범주로 연결하는 데 실패했다. 현재 실험은 내부 특징 표현의 결함과 출력 Label의 사전 편향까지 분리하지는 못했다.','',
'색상·형태 문구 또는 이미지 지칭 한 가지를 바꾸는 것으로는 Peg 거부 대부분을 해결하지 못했다. 실패 대표20개 중 Crop-only는1개, 색상 조건 삭제·형태 조건 삭제·명시적 이미지 지칭은 각각2개만 PEG로 바꿨다. 네 편집을 합쳐도16/20은 어느 편집에서도 PEG가 되지 않았다. 따라서 **“빨간 면이 안 보여서 발생한 단일 문구 오류”로 원인을 확정할 수 없다.**','',
'## 2. 전체 보존 후보의 거부 재분류','',
'직전15 Parent/60 Camera-Frame 평가의 모든 Proposal을 다시 검사했다. IoU≥0.5인 후보를 정답 후보라고 불렀던 기존 정의는 유지한다. Actor 역할 Proxy는 Proposal 내부 Peg/Box/Robot 전경 픽셀 중70% 이상을 차지하는 Actor이며, 혼합 영역은 AMBIGUOUS다. 이 Proxy는 의도한 Target의 완전한 의미 GT를 대신하지 않는다.','',
'| Query | IoU≥0.5 후보 | 해당 역할로 승인되지 않은 후보 | 거부 후보의 Actor 구성 | Qwen 출력 |','|---|---:|---:|---|---|',
'| Peg | 131 | 35 | PEG 우세25, 혼합10 | BOX23, ROBOT12 |',
'| Box | 76 | 7 | ROBOT 우세5, PEG 우세1, 혼합1 | ROBOT6, PEG1 |','',
'Peg 우세25개에서도 PEG→BOX16개·PEG→ROBOT9개가 발생했다. 이는 혼합 영역만으로 설명할 수 없는 오류다. 반면 Box 거부7개 중6개는 Actor 우세 물체와 Qwen Label이 일치했다. 가려진 Fixture의 Visible Bounding Box가 Robot 또는 Peg와 크게 겹치면 IoU는 높아도 Crop의 주 물체는 다른 Actor일 수 있다. “그 안의 물체를 분류”라는 계약과 “이 Proposal이 Box 위치에 맞는가”라는 판정은 다르다. **후자의 Detection Miss는 그대로 보고하되, 전자를 전부 시각 오인식으로 단정하지 않는다.**','',
'## 3. 색상·가시 외형의 관측','',
'원본 RGB와 EXACT Actor Pixel을 사용해 Proposal 안 Target 픽셀의 red-like 비율을 계산했다. 규칙은 R>100, R>1.25G, R>1.25B다. GT는 사후 통계에만 사용했으며 VLM 입력에는 제공하지 않았다. 조명·반사·Antialiasing의 영향을 받는 RGB Proxy이므로 실제 빨간 Material 면의 가시성 GT라고 주장하지 않는다.','',
'| Peg 정답 후보 | 승인 | 거부 |','|---|---:|---:|',
'| red-like pixel=0 | 0/7 | 7/7 |',
'| red-like pixel>0 | 96/124 | 28/124 |','',
'전체 거부35개의 red-like 비율 Median은2.58%, 승인96개는29.00%였다. Main의 모든 정답 Peg 후보를 시점별로 보면 초기38/38 승인, 후기0/24 승인이다. 그 색상 비율 Median은19.74%→1.88%였다. 모든 거부는 F1에서 발생했다. 이 연관은 시점·가림·시점에 따른 형상 변화와 얽혀 있으므로 색상만의 인과 효과는 아래 입력 편집 실험으로 별도 점검했다. AABB Aspect도 기록했지만 3D 물체 길이나 회전을 직접 측정한 값은 아니다.','',
'이전 보고서의 “빨간 면이 보이지 않는다”라는 표현은 직접 본 Crop의 인상에 근거했다. 엄격한 RGB Proxy로는 소량의 red-like 픽셀이 존재하는 사례도 있으므로, **빨간 부분의 단서가 약해졌다**고 표현하는 것이 정확하다.','',
'## 4. 사전 고정한 통제 진단','',
'- Peg 거부가 있었던20개 Camera-Frame과 Box 거부6개 Camera-Frame을 모두 포함했다. 각 Frame에서는 IoU≥0.5 거부 후보 중 원래 DINO Score가 가장 높은 하나를 사전 규칙으로 선택했다.',
'- 26개 오류 후보마다 같은 Parent·Camera의 F0에서 승인됐던 IoU≥0.5 후보를 대조군으로 고정했다. 여기에 Hash로 선정한 명확한 Robot 대조군8개를 더해60개 후보를 사용했다.',
'- 각 후보에 다섯 조건을 적용해 Label300개를 생성했다. 오류 후보26개에 대해서만 중립적 외형 묘사를26회 추가해 총326회다.',
'- 후보 선정은 기존 GT·오류를 알고 수행했다. 이 집합의 Label 변화율은 일반화 정확도나 개선된 최종 Recall이 아니다.','',
'| 조건 | Baseline에서 바뀐 것 |','|---|---|',
'| baseline | 직전 외형 Prompt·원본/표시/Crop3이미지 그대로 |',
'| crop_only | 텍스트 그대로, 마지막 clean Crop 한 장만 입력 |',
'| remove_peg_color | 같은3이미지, PEG 정의의 red-and-white만 삭제 |',
'| remove_peg_shape | 같은3이미지, PEG 정의의 elongated rectangular만 삭제 |',
'| explicit_image_binding | 같은3이미지, 첫2문장만 Image1/2/3 번호로 명시 |',
'| caption | 오류 후보만 별도 중립 묘사. 내부 이유 설명으로 취급하지 않음 |','',
'프롬프트와 요청 Manifest는 실행 전에 SHA로 고정했고 응답에 따라 수정하지 않았다. 새 DINO Proposal·GT Box·Mask·Pose는 추론에 제공하지 않았다. Query/Score/GT 역할도 텍스트에 넣지 않았다.','',
'## 5. 실제 개입 결과','',
'Baseline은 과거 답변60/60을 정확히 재현했다.','',
'| 조건 | Peg 오류 후보→PEG | 초기 Peg 대조군→PEG | Box 오류 후보→BOX | 초기 Box 대조군→BOX | Robot 대조군→ROBOT |','|---|---:|---:|---:|---:|---:|']
for cond,z in m['conditions'].items():lines.append('| '+cond+' | '+' | '.join(rr(z['by_group'][g]['query_role_agreement']) for g in groups)+' |')
lines+=['',
'Box 오류 후보→BOX는 Query 역할 일치 수이며 의미 분류 정확도라고 부르지 않는다. 전체7개 거부에서 우세 Actor가 BOX인 후보는0개였다. 여섯 대표 후보도 ROBOT5개·PEG1개다. 명시적 지칭에서5/6이 BOX로 바뀐 것은 입력 지칭에 민감하다는 증거다. 하지만 명확한 Robot 대조군0023/F1/Main/peg_rank003도 ROBOT→BOX로 바뀌었다. Fixture와 로봇을 섞어 보는 문제를 해결했다고 판단할 수 없다.','',
'Peg 오류20개 중 Actor Proxy가 명확한 PEG는14개, 혼합은6개였다. 각 편집으로 PEG가 된 후보에는 혼합 사례도 포함돼 있다. Crop-only는 실제PEG1개, 색상 삭제는실제PEG1개+혼합1개, 형태 삭제와 명시적 지칭도 각각실제PEG1개+혼합1개를 바꿨다. 명확한 Peg 오류14개 중12개는 네 조건 어디에서도 PEG로 바뀌지 않았다.','',
'텍스트를 그대로 둔 Crop-only는 Main 후보를 BOX에서 ROBOT로 바꾸기도 했지만, Main의 오류를 PEG로 회복하지는 못했다. Full Scene을 제거하면 해결되는 단순한 Context 간섭만으로 대다수 Main 거부를 설명하기 어렵다.','',
'## 6. 실제 묘사와 해석의 경계','',
'다음은 별도 중립 묘사에서 관측한 응답이다. Caption은 눈에 보이는 특징을 설명하도록 했으며 Task 역할 추론은 요청하지 않았다. 그러므로 Caption에 PEG라는 단어가 없다는 사실 자체는 오류가 아니다. 생성된 문장으로 모델의 내부 판단 이유를 읽었다고 주장하지 않는다.','',
'| 대표 후보 | 실제 관측된 묘사의 요지 | 다섯 조건의 Label |','|---|---|---|']
examples=['10i_s1_0767_F1_base_camera_native512_peg_rank000','10i_s1_0023_F1_base_camera_native512_peg_rank002','10i_s1_0686_F1_hand_camera_native512_peg_rank001','10i_s1_0157_F1_base_camera_native512_box_rank000']
paraphrases=['작은 밝은 회색 Cube와 옆의 검은 직사각형 부분','회색 직사각형 Block/Panel, 평평한 윗면과 수직 옆면','큰 Light-blue 직사각형과 위쪽의 작은 적갈색 직사각형','회색 Block형 물체와 위쪽 돌출부, Mechanical assembly의 일부']
for cid,desc in zip(examples,paraphrases):
 c=next(c for c in cases if c['candidate_id']==cid);lines.append('| '+cid.replace('10i_s1_','').replace('_native512','')+' | '+desc+' | '+' / '.join(c['outputs'].values())+' |')
lines+=['',
'관측: 모델은 실제 이미지의 회색/흰색 Block형 부분과 주변 검은 부품을 묘사한다. Peg로 우세한 영역에도 BOX/ROBOT Label을 붙이는 오류가 남고, 실제로 여러 Actor가 겹친 Crop에서는 어떤 물체를 후보의 주체로 삼는지에 따라 Label이 흔들린다.','',
'가능한 설명: 후기의 부분 관측에서 길이·두 색·Gripper와의 경계 같은 정체 단서가 약해지고, 가까운 Cube/로봇 부품 범주로 연결될 수 있다. 그러나 색상·형태·지칭을 하나씩 바꾼 결과가 대부분을 회복하지 못했으므로, 특정 한 문구 또는 한 단서가 근본 원인이라고 확정하지 못했다. 시각 특징 표현, 주변 부품의 경계 분리, Label 이름의 사전 편향, 이 요인들의 상호작용은 추가로 분리되지 않았다.','',
'배제할 수 있는 설명의 범위: 같은 입력의 Baseline60/60 재현과 모든 요청의 빈 초기 Cache를 확인했다. 본 실험은 양자화 없는 BF16이고300개 Label은 모두 엄격한 형식을 지켰다. 따라서 이번 거부를 대화 누적, JSON Parsing 실패, NF4만의 문제로 설명할 근거는 없다. 이것이 모든 구현 경로의 무오류나9B 크기 자체의 한계를 증명하지는 않는다.','',
'## 7. 다음 설계 판단에 필요한 의미','',
'정답 Box의 공간 일치, Crop 안 물체의 주 역할, SAM2가 실제 선택할 Mask를 분리해야 한다. 이번에 확인한 Box의 기준 충돌과 Peg의 실제 분류 실패를 같은 거부율 하나로 고치면 잘못된 승인도 늘 수 있다. 단어를 더 추가해 전체 문제를 해결했다고 선언할 근거는 없다. 이번 편집 중 하나를 새 Production Prompt로 채택하거나 전체 평가 결과를 덮어쓰지 않았다.','',
'초기 Discovery와 후기 Re-grounding의 차이는 별도다. 직전 전체60장에서는 외형 Prompt의 두 물체 동시 Box 성공이F0 25/28, F1 5/30이었다. 이 값은 기존 예측을 시점별로 나눈 관측이다. SAM2 Tracking이 후기 실패를 해결한다고 검증한 결과는 아니며, 현재 학습 입력이 준비됐다는 판정도 유지하지 않는다.','',
'## 8. 실행·보존·검증','',
'Vast.ai58576054 / RTX3090, Qwen/Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. 전체BF16·SDPA·cuda:0·Offload없음·Batch1·Greedy·enable_thinking=False·use_cache=True. Processor는 기존Source/min=max262144px. Label16Token, Caption96Token. Load1회, 요청326회, 자동Retry/추가Warmup/추가DINO/SAM2/Simulator/Training0회.','',
f'Model Load{lat["load_ms"]:.2f}ms, loaded GPU{lat["loaded_memory"]["allocated"]/2**30:.3f}GiB. 초기 빈 Cache326/326, 보호파일{len(v["protected_before"])}개 SHA 및 환경 전후 동일. 공통 ExperimentLogger(W&B disabled)를 재사용했다. 원격CUDA만 실행했고 macOS GPU/Simulator는 실행하지 않았다. Production과기존Checkpoint/Dataset은 보존했고 Stage4A-2 PAUSED_SCOPE_REVIEW/Stop Stub을 유지했다. Commit·외부 게시는 없다.','',
'| 조건 | Warm n | Mean / Median / p95 ms | Peak Allocated GiB |','|---|---:|---|---:|']
for cond,z in lat['summary'].items():lines.append(f'| {cond} | {z["warm_requests"]} | {z["mean_ms"]:.2f} / {z["median_ms"]:.2f} / {z["p95_ms"]:.2f} | {z["peak_allocated_GiB"]:.3f} |')
lines+=['',
f'로컬 수신 후 Artifact{lv["indexed_remote_artifacts"]}개와 이전 동결 Artifact{lv["old_frozen_artifacts_unchanged"]}개 SHA, Source{lv["source_files"]}개, 요청 이미지SHA{lv["image_hashes"]}개를 확인했다. Case역할60개와 전체정답후보207개의 색상 통계를 독립 재계산했다. 원시 응답은 채점 전후 동일하다.','',
'## 9. 산출물','',
'- '+', '.join(link(t,p) for t,p in [('Protocol','protocol.json'),('선정 Manifest','case_manifest.json'),('RGB-only 요청 Manifest','request_manifest.json'),('Prompt Identity','prompt_identity.json'),('Model Identity','model_identity.json')])+'.',
'- '+', '.join(link(t,p) for t,p in [('전체 정답 후보 감사','all_good_candidate_audit.json'),('326개 원시 응답','verifier_raw_responses.json'),('전체 진단 지표·Caption','mechanism_metrics.json'),('후보별 출력','mechanism_cases.csv'),('Label 변경','mechanism_label_changes.csv')])+'.',
'- '+', '.join(link(t,p) for t,p in [('Latency/VRAM','verifier_latency_vram.json'),('원격 검증','verification.json'),('로컬 독립 검증','local_verification.json'),('동결 Artifact Index','artifact_index.json')])+'.','',
'## 부록 — 입력 편집 원문','']
for cond in ['remove_peg_color','remove_peg_shape','explicit_image_binding']:
 lines+=['### '+cond,'','```text',prompts[cond]['prompt'],'```','']
REPORT.write_text('\n'.join(lines));text=REPORT.read_text();links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text);assert all((REPORT.parent/l).resolve().exists() for l in links if not l.startswith('http'))
assert all(sha(R/n)==e['sha256'] for n,e in rd(R/'artifact_index.json')['files'].items())
index=dict(status='COMPLETE_VERIFIED',report_path=str(REPORT.relative_to(ROOT)),report_sha256=sha(REPORT),local_verification_sha256=sha(R/'local_verification.json'),remote_artifact_index_sha256=sha(R/'artifact_index.json'),local_source_hashes={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()},interpretation='Partial-view classification failures + box/semantic-contract mismatch; tested single input edits do not explain most Peg errors; internal cause unresolved',production_changed=False)
(R/'local_completion_index.json').write_text(json.dumps(index,indent=2)+'\n');print('FULL_REPORT_VERIFIED',str(REPORT))
