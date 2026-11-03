# Task 10-I Stage 4A-2 — Frozen Qwen Semantic Verifier Independent Evaluation

2026-11-02. **사전 Semantic Gate: PASS. GPU 단기 동시 상주: PASS. 초기화 연결 Smoke: 완료.**
**실제 Peg 누락·오승인 감소 효과는 지원 부족으로 미입증이며 Production 통합은 진행하지 않는다.**
`PAUSED_SCOPE_REVIEW`와 기존 Stop Stub을 유지했다.

24개 Parent·48프레임·192후보에서 PEG Recall30/31, BOX Recall30/30,
비PEG→PEG0/107, 형식 통과192/192로 동결한 운영 Gate를 통과했다.
그러나 **ROBOT73/77을 BOX로 분류**했고, 정상 승인 Peg1개를 OTHER로 거부했다.
따라서 Peg/Box/Robot 전체를 안정적으로 구별한다는 결론은 성립하지 않는다.
이 보고서의 PASS는 아래에 명시한 좁은 Gate의 충족이며 종합 안전성 승인이 아니다.

## 평가 집합·동결 계약

기존 Stage3 `dataset_R_train_manifest.json`/`dataset_D_train_manifest.json`의 물리 Parent를
합집합으로 중복 제거했다. `full_task` Raw와 기존 EXACT_GT Reference가 존재하는 Train
Parent만 Eligible로 삼고, Shared Validation과 Prompt 개발 Parent0355/0645를 제외했다.
Eligible34개를 `SHA256("Task10I-QwenFrozenIndependent-v1" + "|" + parent_id)` 순으로 정렬해
첫24개를 선택했다. Manifest의 `selection_seed`에 같은 문자열을 보존했다. ID/시점/후보/평가 규칙/Gate는 VLM 실행 전에 파일과 SHA로 동결했다.

Parent별 F0=t0, F1=`max(1,floor(actions/2))`, Main(`base_camera`)만 사용했다.
선정 후 Parent·시점·후보를 교체하지 않았다. 새 Reset/Expert Demonstration/Random Benchmark는 없다.
Prompt 개발로부터 독립인 기존 Train 평가이며 새로운 Simulator 분포의 Blind Test는 아니다.

```text
10i_s1_0519, 10i_s1_0531, existing_30008, 10i_s1_0685, existing_30046, 10i_s1_0449
10i_s1_0780, 10i_s1_0120, 10i_s1_0514, 10i_s1_0909, 10i_s1_0786, 10i_s1_0171
10i_s1_0387, 10i_s1_0023, 10i_s1_0613, 10i_s1_0305, 10i_s1_0276, 10i_s1_0589
existing_30107, existing_30077, 10i_s1_0919, 10i_s1_0756, 10i_s1_0029, existing_30044
```

[Parent Manifest·실제 Episode/Raw/시점/Hash](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/evaluation_parent_manifest.json)에 전체 Identity가 있다.

모델은 `Qwen/Qwen3.5-9B`, Revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`다. NF4/BF16 compute/Double Quantization/
SDPA/Batch1/Greedy/`enable_thinking=False`/`max_new_tokens=16`/Offload OFF를 그대로 유지했다.
공식 Multimodal `Qwen3_5ForConditionalGeneration`과 기존 Processor·독립 환경을 재사용했다.
실제 Linear4bit358개, 전체 Parameter/Buffer GPU를 확인했다. 모든 Parameter가4-bit인 것은 아니다.
Torch2.14.1+cu130, Transformers5.19.0, BitsAndBytes0.50.2이며 설치·Kernel Patch는 없다.

Prompt를 직접 다시 쓰지 않고 이전 성공 실험의 `prompt_identity.json`에서 읽었다.
**Prompt SHA256: `691ea9e22ec710dad77084db6aa812455d525190f7d6f589821bfaf082889a06`.**
원문·Generation·Quantization·모델15개 파일 SHA와 Source는
[frozen_verifier_identity.json](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/frozen_verifier_identity.json)에 보존했다.

원본128² RGB→청록색(0,180,220)1px Rectangle→각 변10% Margin Crop 순서를 유지했다.
Marker/Crop 반올림, 이미지 순서 안내문, Chat Template와 min/max pixels65536도 같다.
기존 Raw를 읽어 새 후보용 입력 View만 만들었고 색·형상은 바꾸지 않았다.
DINO Label/Score·GT/Actor/Pose·SAM 평가 결과는 VLM 메시지에 넣지 않았다.

## DINO 후보와 GT 계약

저장된 B0 Main DINO 결과를 RGB SHA로 대조해 우선 재사용했다. 부족한 선정 프레임에만
기존 DINO를48회 호출했다. PEG/BOX Prompt 각24회이며 전체 Sequence 재실행은 없다.
Checkpoint·Processor·box/text threshold .1과 기존 Consumer의 area<.25/Top5/
Joint Null .3/exclusion IoU .3을 유지했다.

각 Frame 후보 우선순위는 ①기존 Joint 승인 PEG/BOX ②각 Label의 최상위 미선정 후보
③나머지 Score 내림차순이다. IoU≥.99의 거의 동일 Box는 Semantic Source를 합쳐 중복 제거하고
최대4개를 골랐다. Consumer Area 기준을 넘은 Raw 후보도 진단 후보로 남겼다.
결과는48×4=192개였다. GT로 후보를 찾거나 VLM 응답을 보고 추가하지 않았다.
900개 Decoder Query 전체와 threshold .1 이후의 사용 가능 Proposal은 구분했다.

원래 t0 Consumer의 승인 목록은 저장 B0 `new_object_ids`와 변경 없는 Joint Assignment를
대조했다. 이후 시점의 원래 Consumer가 재검출하지 않은 경우는 `NO_NEW_APPROVAL`로
남기고 Semantic 거부로 세지 않았다. 해당 Frame의 새 Joint 판정은 **Counterfactual 진단**으로만
저장했다. [DINO 전체/선정 후보·Consumer 근거](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/dino_candidate_manifest.json).

**192개 응답과 입력 SHA를 전부 저장·동결한 뒤 처음 GT 의미 내용을 파싱했다.**
보존용 GT Byte Hash 읽기는 의미 평가와 분리했다. GT Barrier 전후 응답 SHA는 같다.
44/48프레임이 EXACT_GT였고4프레임(16후보)은 `GT_UNSUPPORTED`로 남겼다.
지원 Frame만의 성능을 전체48프레임 성능으로 주장하지 않는다.

사전 GT 규칙은 Table/Ground/Background를 전경에서 제외하고, 후보 내 최소16 Pixel·
전경 중70% 이상 지배·전체 가시 Actor Category의25% 이상 포함일 때만 지배 Category를 부여한다.
그 외는 `AMBIGUOUS`다. 전체 Pixel 구성과 지배 비율을 저장했다. 이는 **영역 Pixel 기준의
운영 Annotation**이며, 큰 Rectangle 안에서 사람이 의도한 물체를 직접 Annotate한 Label은 아니다.
여러 물체가 포함된 큰 Box에서의 해석상 한계를 유지한다. 공간적으로 올바른 Raw 후보는
추가로 가시 Actor50% 이상 포함 및 GT 가시 Bounding Box IoU≥.5를 요구했다.

## 독립 Semantic 결과와 실패

| GT 지원 Category | 후보 수 | 정확한 Label | PEG로 오승인 |
|---|---:|---:|---:|
| PEG |31|30 (96.77%)|해당 없음|
| BOX |30|30 (100%)|0/30|
| ROBOT |77|4 (5.19%)|0/77|
| OTHER |0|INSUFFICIENT_SUPPORT|INSUFFICIENT_SUPPORT|
| AMBIGUOUS |38|정답 강제 배정 없음|출력 PEG2개, 정확성 미판정|
| GT_UNSUPPORTED |16|미판정|출력 PEG1개, 정확성 미판정|

명확한 지원 후보의 전체4-class 정답은 **64/138=46.38%**다.
ROBOT73개가 BOX여서, 명확한 지원 후보에서 BOX 출력 Precision은 **30/103=29.13%**다.
**BOX Recall100%와 BOX 출력의 신뢰성은 다르다.** 비PEG→PEG0/107은 관측됐지만
모든 비PEG 의미를 올바르게 구별했다는 뜻은 아니다.
UNCERTAIN0/192, INVALID_OUTPUT0/192, 형식 성공192/192였다.
AMBIGUOUS38개에서도 UNCERTAIN은0이었으므로, 시각적 모호성에서 안전하게 보류한다는 증거는 없다.

핵심 Peg 실패는 **0613/t0 DINO rank0**, score .361889다. 후보 안 Peg452 Pixel,
다른 Task Actor0, 가시 Peg Coverage100%, GT bbox IoU .864538인데 Qwen은 **OTHER**였다.
기존 Consumer가 올바르게 승인한 Peg를 새 Verifier가 거부하는 실제 회귀다.
Robot→BOX 오류는21개 Parent에 분포한 전체 Semantic 실패와 함께
[evaluation_metrics.json](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/evaluation_metrics.json)의 `failure_cases`/`parent_failures`에 공개했다.
DINO와 Qwen이 같은 BOX라고 해도 실제 지배 Actor가 Robot인 사례가 있었다.

| 집계 범위 | PEG Recall | BOX Recall | 비PEG→PEG |
|---|---:|---:|---:|
| Candidate-level |30/31|30/30|0/107|
| Parent-macro |95.65% (지원23 Parent)|100% (23 Parent)|0% (24 Parent)|
| F0=t0 |30/31|29/29|0/41|
| F1=조작 시점 |INSUFFICIENT_SUPPORT (0후보)|1/1|0/66|
| 원래 승인 후보 |22/23|20/20|0/22|
| 원래 폐기 후보 |8/8|9/9|0/19|

Parent Bootstrap(Seed42,10,000회)의 PEG Macro95% CI는 **86.96–100%**다.
BOX는100–100%, 비PEG→PEG는0–0%지만, 이는 표본에 오류가 없어서 생긴 퇴화 Bootstrap
구간이며 희귀 오류가 없다는 보증으로 쓰지 않는다. 후보 수를 독립 Parent 수처럼 취급하지 않았다.
DINO 승인/폐기/미실행, Agreement/Conflict별 전체 분모와 Parent별 결과도 JSON에 있다.
[원시 Text/판정 CSV](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/vlm_candidate_predictions.csv),
[Confusion Matrix](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/semantic_confusion_matrix.csv).

사전 Gate는 **PEG/BOX Recall≥90%, 비PEG→PEG≤5%, 형식≥99%,
PEG/BOX/비PEG 지원 각≥15**였고 모두 충족했다. ROBOT Recall이나 BOX Precision은 이 Gate에
포함하지 않았다. 결과를 보고 기준을 바꾸지 않았으며, 이 좁은 Gate의 맹점도 함께 보고한다.
[Gate 결정](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/semantic_gate_decision.json).

## 실제 누락 복구·오승인 차단 가능성

| 진단 | 실제 결과 |
|---|---|
| EXACT_GT-visible Peg Frame |44개|
| Peg Prompt Proposal 자체가0개 |0/44|
| 공간적으로 올바른 Raw Peg 후보 없음 |10/44|
| t0에서 올바른 Raw Peg가 있으나 Consumer가 최종 폐기 |0/23|
| 실제 잘못 승인된 t0 Peg 후보의 차단 효과 |0건, INSUFFICIENT_SUPPORT|
| 정상 승인 Peg 보존 |22/23;1개 OTHER 거부|
| 폐기된 의미상 Peg 후보의 PEG 분류 |8/8,6 Parent; 주 후보가 이미 승인된 대안 포함|

폐기된8개를 PEG로 분류한 것은 Semantic 보완 신호다. 다만 그 Parent들은 이미 올바른
주 Peg가 승인돼 있었고,6개는 DINO의 BOX Phrase가 실제 Peg를 가리킨 대안이었다.
따라서8건을 누락 물체8개 복구로 세지 않았다. 공간 기준을 만족하는 폐기된 **Peg Phrase**
후보의 실제 Recovery, 잘못 승인된 Peg 차단, 둘 다 틀린 승인 비율은 분모0으로 미판정이다.

F0의 정확한 가시 Peg23 Frame은 Raw23→선정23→Qwen PEG22였고,
F1의 정확한 가시 Peg21 Frame은 Raw11→선정0→Qwen의 올바른 공간 후보0이었다.
조작 중 정상 Peg를 선택하지 못한 것은 후보 경로의 한계이며 VLM 분류 실패와 분리한다.
이번 최대4개 우선순위가 낮은 순위의 정상 Peg를 제외할 수 있음을 확인했지만 규칙을 바꾸지 않았다.
**F1 PEG 분류 일반화나 최종 Detection/SAM2 Recall 개선은 입증되지 않았다.**
[candidate_recall_and_false_acceptance.json](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/candidate_recall_and_false_acceptance.json).

## RTX3090 단기 동시 상주와 비용

Semantic Gate PASS 후에만 Grounding DINO Tiny, SAM2.1 Hiera Small, DINOv2 Small을
Qwen NF4와 함께 GPU에 로드했다. 모든 모델 GPU Weight·Forward를 확인했고 Offload/OOM은 없었다.
사전에 동결한2GiB Headroom 기준을 통과했다. 서로 다른 Stream 병렬화는 하지 않았다.

| 누적 모델 로드 상태 | CUDA Allocation | CUDA Reservation |
|---|---:|---:|
| Qwen NF4 |7.349GiB|7.406GiB|
| +GroundingDINO |7.993GiB|8.004GiB|
| +SAM2 |8.303GiB|8.324GiB|
| +DINOv2 |8.386GiB|8.414GiB|

전체 Forward 중 최대 Allocation은 **9.189GiB**, 최대 Reservation은 **9.482GiB**,
관측한 최소 Device Free는 **13.757GiB**였다.
짧은3Cycle에서만 확인했으며 긴 SAM2 Tracking Memory·장시간 안정성·동시 Stream 경쟁은 미검증이다.

| 네 모델 상주 후 실행 | 첫 Cycle | 후속 두 Cycle |
|---|---:|---:|
| DINO 단일 Prompt |473.72ms|106.90–109.72ms|
| SAM2 단일 Frame/Box |152.76ms|73.73–73.95ms|
| DINOv2 Forward |13.07ms|8.66–8.85ms|
| Qwen 동일 상주 모델 요청 |713.56ms|437.38–439.25ms|

VLM은 초기화/불확실 재검증 시 호출하는 조건이다. 이 수치는 개별 Forward의 단기 상주
Microbenchmark이며 전체 Pipeline 초기화 Wall Time이나20Hz Closed-loop 성능이 아니다.
SAM2의 기존 `_C` 확장 부재로 Fill-hole 후처리가 생략되는 경고를 그대로 보존했고 패치하지 않았다.

별도 단독 Semantic192요청은 Load **3026.17ms**, 후속191요청 E2E
p50 **446.32ms** / p95 **461.10ms** / max **504.13ms**였다.
첫 요청은1160.11ms로 초기 비용을 포함했다. 출력은4 Token185개/5 Token7개,
단독 Peak Allocation **7.592GiB**였다. 새 이미지마다 Vision Encoding/Prefill을 실행했다.
[gpu_residency_audit.json](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/gpu_residency_audit.json),
[warm_inference_latency.json](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/warm_inference_latency.json).

첫 상주 시도는 Forward Hook Stack 변수 `st`를 SAM Timing Dictionary로 덮어쓴 계측 오류로
Qwen 첫 Forward 직전 Hook에서 중단됐다. OOM이나 모델 미지원이 아니다. 오류 Receipt를 보존하고 변수만
수정해 **상주 단계만1회 재실행**했다. 독립 Semantic192개 재시도0, 총 Qwen Load2회,
성공 상주 Qwen Timing3회다. 첫 시도의 DINO/SAM2/DINOv2 Forward도 로그에 남았다.

## 독립 Wrapper 초기화 Smoke

새 후보 ID2개는 VLM 평가 전에 첫 두 Hash-selected Parent의 첫 후보로 동결했다.
P/N/C1은 이전 성공 응답, 새 후보는 이번 동결 응답을 재사용했다. **추가 Semantic 요청0**이다.
이 검사는 저장된 DINO/Qwen Record→Wrapper 처분→실제 Temporary SAM2 전달을 실행한
연결 Smoke이며, 모든 Detector/VLM을 새로 호출한 Live End-to-end 성능 검사는 아니다.

| 후보 | Wrapper 처분 | 실제 Temporary SAM2 |
|---|---|---|
| P |ACCEPT|542 Pixel Mask|
| N |DEFER: DINO PEG / Qwen BOX 충돌|미실행|
| C1 |ACCEPT|840 Pixel Mask|
| 0519/F0 첫 후보 |DEFER: 겹치는 PEG 경쟁 후보|미실행|
| 0531/F0 첫 후보 |ACCEPT|437 Pixel Mask|

Wrapper는 일치한 후보도 같은 Label의 겹치는 경쟁 후보가 있으면 DEFER하도록 제한했다.
이 처분 규칙은 Production Controller가 아니다. 기존 SAM2 State와 분리된 임시 State에
한 Frame/Box만 전달해 Mask를 받고 폐기했다. Production Memory Commit·Sequence Propagation은 없다.
Regression3개를 독립 정확도에 다시 합산하지 않았고, 새 Mask 생성만으로 Tracking/Recovery/
PegInsertion 성공을 주장하지 않는다.
[initialization_smoke_results.json](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/initialization_smoke_results.json).

## 보존·종료 판단

선정 Raw/GT·기존 Source/Checkpoint/Stop Stub 등을 합친 **1414개 파일**과
Qwen 모델/Processor15개 파일, 새 입력 View의 시작/종료 SHA가 일치했다.
기존 연구 Python Package 목록도 동일했다. `PAUSED_SCOPE_REVIEW`와 Stop Stub은 유지했다.
공통 `ExperimentLogger`(W&B disabled)를 재사용했다. 독립 평가 UUID는 `3c4517a1ccf244f6bdb471349f2002ce`,
상주 계측 재실행 UUID는 `8108d093ffd14ecaa6eecb3129394961`다. 실행 Source·원시 응답·입력 View·로그는 `.execution/`에 있다.
[verification.json](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/verification.json), [artifact_index.json](../../outputs/task10i/stage4a2/qwen35_independent_20261102_v1/artifact_index.json).

Production Source/Merge·기존 Dataset/Weight 수정·새 Episode·전체 Dataset 추론·Graph 재캐시·
GNN/GRU 학습·Internal/Final Runner 재개·외부 게시·Git Commit은0이다.
Simulator/MPS는 실행하지 않았고 RTX3090 CUDA 모델만 실행했다. 공통 Pipeline 변경이 없어
Simulator/학습 테스트 대신 실제 요청·Mask Readback·예산·SHA·GT Barrier·CSV/JSON 계약을 검사했다.

**Q1:** 초기 Peg/Box와 PEG-vs-nonPEG 구분에는 긍정적 신호가 있으나 Robot 구별과 F1 Peg
지원 부족 때문에 전체 조작 과정의 안정적 의미 인식은 미입증이다.
**Q2:** 폐기 대안 후보를 재분류할 수는 있었지만 실제 누락/잘못된 Peg 승인 독립 사례가 없었고
정상 Peg1건 거부도 발생했다. 누락·오승인 감소라는 핵심 효과는 아직 입증하지 못했다.
**Q3:** 네 모델의 GPU 단기 상주와 제한된 초기화 연결은 가능했다. 이것은 Production 채택 근거와 다르다.

다음 Stage나 Production 통합의 GO는 내리지 않는다. 사전 허용된 Gate 이후 검사까지만 완료했고
추가 Prompt/후보 탐색·Benchmark 확장은 하지 않았다. 결과가 보여 준 한계와 정상 거부 사례를
남기고 종료한다.

원격 Root: `/workspace/task10i_stage4a2_qwen35_independent_20261102_v1/`.
