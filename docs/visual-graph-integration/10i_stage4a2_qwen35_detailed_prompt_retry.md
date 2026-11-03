# Task 10-I Stage 4A-2 — Qwen3.5-9B 상세 Prompt 재평가

2026-11-02. **최종 판정: `VLM_GROUNDING_INSUFFICIENT`.** 기존 `PAUSED_SCOPE_REVIEW`와 Production Source를 유지했다.

사용자의 상세한 Prompt 개선 및 한 번의 재실행 요청에 따라 기능적 역할·오인식 방지·가림·좌표 지시를 구체화한 **새 Prompt1종**을 동결하고 실행했다. 먼저 준비한 짧은 초안은 모델 로드/추론0회 상태에서 대체했으며 보존했다. 결과를 보고 Prompt를 다시 수정하거나 요청을 재생성하지 않았다.

이번은 **이미 평가하고 실패를 분석한 같은24 Parent의 재평가(RETEST)**다. 새로운 독립 일반화 성능으로 해석하지 않는다.

## 1. Prompt와 통제 조건

Prompt는 **889단어**, 이미지 포함 입력 **1292 Token**이다. SHA256은 `40b8c68e18ab9d588f86760ea5023cf0366e85e216fe7d94e18f749348f6f3cb`다. 전체 원문은 아래와 [grounding_prompt.txt](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/grounding_prompt.txt)에 보존했다.

PEG의 파지·부분 삽입 상태, 그리퍼와 물체의 경계, BOX 본체와 빈 구멍의 차이, 가림/화면 경계, 겹치는 두 물체, Tight Box의 기준, Null과 좌표 순서 및 최종 점검을 명시했다. 색·재질·고정 위치 정답 규칙, GT 좌표, DINO Proposal, Crop/Marker는 추가하지 않았다.

공식2D Grounding 예시의 `bbox_2d`/`label` 배열을 사용했다. 전체 응답이 순수 JSON이거나 단일 완전 JSON Fence일 때만 파싱한다. 설명문·부분 추출·중복 Key·추가 Label은 거부하며 범위/양수 크기를 검증하고 Clipping하지 않는다. 이 Parser는 추론 전에 동결했다. Bare JSON 준수율은 별도 보고한다.

공식 참고: [Grounding Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb), [Qwen3.5 좌표 Q&A](https://github.com/QwenLM/Qwen3.8/discussions/56).

| 통제 항목 | 실제 |
|---|---|
| 모델 /Revision |Qwen/Qwen3.5-9B /c202236235762e1c871ad0ccb60c8ee5ba337b9a|
| Weight/Compute |기존 NF4 /BF16, Double Quantization ON; Vision110개 포함 Linear4bit358개 유지|
| 실행 |SDPA, cuda:0, Batch1, Greedy, enable_thinking=False, max_new_tokens=128|
| Processor |기존 소스·min/max pixels65536; 원본128²→내부256²|
| 입력 |원본 전체 RGB1장+상세 Task Prompt만|
| 예산 |모델 로드1회, Phase A3요청, 재평가96요청|

모델/환경/해상도/Generation은 이전과 같고, Prompt 및 요청한 JSON 구조를 변경했다. Vision 정밀도 개선은 동시에 적용하지 않았다. 새 Identity에는 이번 실제 Rendered Template/Shape를 기록해 이전 Identity의 과거3-image 메타데이터 혼입 문제를 피했다.

## 2. 대상·동결·평가

이전 Manifest의99개 이미지(P/N/C1 3개+같은24 Parent의96개)와 기존 SAM2 대상16개 ID를 그대로 사용했다. Main/Wrist×F0/F1은 각각24장이다. F1은23 Parent에서t80,0134에서 사전 규칙의t78이다. Parent·Frame·Camera를 결과에 따라 바꾸지 않았다.

기존 GT 지원도 그대로 고정했다:56/96개 이미지,15개 Parent에서 EXACT_GT를 지원하고40개는 미지원이다. Peg56개, Box54개가16Pixel 이상이며, Box2개는 완전 비가시다. Tiny1–15Pixel 사례는0개다. 다른 GT Cache로 지원 범위를 늘리지 않았다.

원시 응답 SHA는 `cb83def974d2d396b26af0aec2967ee2fa8d98545ed94989a387828a21dbf4e5`이며 새 예측 전체와 CSV를 저장·Hash 동결한 뒤에만 사후 GT 평가를 실행했다. 이 실행의 GT Barrier와 이전에 이미 GT를 본 재평가라는 사실을 구분한다.

[입력 Manifest](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/input_manifest.json), [모델 Identity](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/model_identity.json), [원시 응답](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/full_image_grounding_raw_responses.json), [예측 CSV](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/full_image_grounding_predictions.csv).

## 3. Phase A 기술 확인

| 사례 | 엄격한 Bare JSON | Parser 채택 | PEG IoU | BOX IoU |
|---|---|---|---:|---:|
|P|True|True|0.4055|0.1816|
|N|False|False|0.0000|0.0000|
|C1|True|True|0.8201|0.3236|

N에서 관측된 중복 `bbox_2d` Key도 원문에 보존했으며 동일 값이어도 자동 정리하지 않았다. 이3장은 기술 확인이며96개 재평가 집계에 합산하지 않는다.

## 4. 형식과 위치 성능을 분리한 비교

이전 Bare JSON은2/96 (2.08%), 새 Bare Native JSON은76/96 (79.17%), 새 사전 Wrapper 허용 정책의 채택 JSON은76/96 (79.17%)다.

**위치 비교의 이전 기준은 Strict 실패를0으로 만든 값이 아니라, 이전 Fence 내부 좌표의 사후 진단값**이다. 따라서 기존2.08% 형식 성공률과 새 Parser 성공률의 차이를 그대로 Prompt 위치 개선으로 부르지 않는다.

| 동일 GT/이미지 위치 성능 | 이전 Peg | 새 Peg | 이전 Box | 새 Box |
|---|---:|---:|---:|---:|
|IoU≥.5 성공|25/56|3/56|9/54|4/54|
|Mean IoU, Miss=0|0.4014|0.1526|0.2752|0.1889|
|IoU≥0.3|36/56 (64.29%)|15/56 (26.79%)|23/54 (42.59%)|17/54 (31.48%)|
|IoU≥0.7|6/56 (10.71%)|0/56 (0.00%)|2/54 (3.70%)|1/54 (1.85%)|

| 같은 Object의 변화 | Peg | Box |
|---|---:|---:|
|실패→성공|1|4|
|성공→실패|23|9|

Parent를 독립 집계 단위로10,000회 Bootstrap했다. 다음 구간은 새 Recall−이전 좌표 진단 Recall의 Parent-cluster95% CI다. 표본의0-error 퇴화 구간은 희귀 오류 안전성 보증이 아니다.

- peg: 지원15 Parent, Macro 차이-41.67%, Image-rate 차이 CI-50.00%~-29.03%.
- box: 지원15 Parent, Macro 차이-9.44%, Image-rate 차이 CI-23.68%~5.17%.

| 새 Parent-macro 지표 | 평균 | Macro 95% CI | 지원 Parent |
|---|---:|---:|---:|
|PEG Recall|5.00%|0.00%~10.00%|15|
|BOX Recall|7.22%|1.67%~13.89%|15|
|좌표 계약|79.17%|71.88%~85.42%|24|

두 물체 동시 정확은0/54 (0.00%), 하나 이상의 잘못된 채택 Box를 낸 Scene은43/56 (76.79%)다. Null/형식/좌표 실패는 Detection 성공으로 계산하지 않았다.

| 상세 실패 지표 | Peg | Box |
|---|---:|---:|
|Missing Detection|53/56 (94.64%)|50/54 (92.59%)|
|False Discovery, 유효 출력 중 실패|40/43 (93.02%)|39/43 (90.70%)|
|완전 비가시 시 Box 출력|INSUFFICIENT_SUPPORT (0)|2/2 (100.00%)|
|Center Error Mean, 유효 출력만|29.3568px|31.8977px|

명백한 Task Actor 의미 혼동: Peg {"peg->robot": 8, "peg->box": 6}, Box {"box->peg": 9, "box->robot": 4}. Rectangle의 Actor Pixel 지배 기준과 전체 Pixel 구성은 Metric/CSV에 있다. 좌표가 이미지 안이라는 이유만으로 올바른 대상이라고 처리하지 않는다.

## 5. Camera·시점별 결과

| 구분 | 이전 Peg | 새 Peg | 이전 Box | 새 Box |
|---|---:|---:|---:|---:|
|base_camera|13/28 (46.43%)|2/28 (7.14%)|7/28 (25.00%)|3/28 (10.71%)|
|hand_camera|12/28 (42.86%)|1/28 (3.57%)|2/26 (7.69%)|1/26 (3.85%)|
|F0|16/30 (53.33%)|2/30 (6.67%)|7/29 (24.14%)|2/29 (6.90%)|
|F1|9/26 (34.62%)|1/26 (3.85%)|2/25 (8.00%)|2/25 (8.00%)|

[전체 지표·Parent별/Camera×시점별 결과](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/grounding_evaluation_metrics.json), [새/이전 Paired 비교](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/prompt_comparison_metrics.json), [실패 사례](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/grounding_failure_cases.csv).

## 6. 같은16개 Camera-Frame의 실제 SAM2 초기화

모델/Weight/Config와 JPEG 입력 경로를 유지했다. 사전 선정16개에서 실제 Temporary State12개, Box Prompt24개를 실행했다. 유효한 새 Qwen Box만 사용했고 GT/DINO Box 대체, Memory Commit, Propagation은 없다. GT 지원은 Object별12개다.

| SAM2 지표 | Peg | Box |
|---|---:|---:|
|End-to-end IoU≥.8 성공|0/12 (0.00%)|0/12 (0.00%)|
|선정16개 중 실제 초기화|12/16 (75.00%)|12/16 (75.00%)|
|실제 초기화한 Mask Mean IoU|0.2411|0.2050|
|미초기화=0 포함 Mean IoU|0.1607|0.1367|
|Pixel Precision Mean|0.5596|0.4334|
|Pixel Recall Mean|0.2064|0.2024|
peg SAM2 Parent-macro 성공률0.00%, 지원9 Parent, Macro95% CI0.00%~0.00%.
box SAM2 Parent-macro 성공률0.00%, 지원9 Parent, Macro95% CI0.00%~0.00%.

**이전 SAM2와의 성공률 차이는 Prompt만의 인과 효과가 아니다.** 이전은 Fence를 거부해1개 Frame/2개 Box만 실행했다. 이번은 사전 허용 Wrapper 정책으로 더 많은 Box를 전달할 수 있으므로 입력 가용성 변경과 좌표 변경이 함께 작용한다. 실제 Mask별 IoU·Actor 구성·Robot/Background 오염·Wrong-object 결과를 함께 남겼다.

[SAM2 지표](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/sam2_initialization_metrics.json), [32개 Object 처분과 Mask 평가](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/sam2_initialization_cases.csv). 기존 `_C` 확장 부재 경고는 보존하고 설치/패치하지 않았다.

## 7. 기존 DINO 공통 Frame 비교

새 예측 동결 이후에만 이전 보존 비교를 읽었다. 같은12개t0 이미지/6 Parent에 한정되며 새 DINO 추론은0이다.

| 공통12장 IoU≥.5 | DINO Joint | 이전 Qwen 좌표 진단 | 새 Qwen |
|---|---:|---:|---:|
|peg|7/12|6/12|1/12|
|box|7/12|4/12|0/12|

`MATCHED_DINO_BASELINE_INSUFFICIENT`를 유지한다. t0 소수 공통 표본으로 전체 분포의 상대 우열이나 DINO 제거를 확정하지 않는다.

## 8. 실제 비용

Weight Load3536.37ms, 로드 후 Allocated7.349GiB, Qwen 요청 Peak Allocated7.770GiB /Reserved7.828GiB다. CPU50ms 표본 Peak RSS18.492GiB다. Model File Hash로 File Cache가 Warm한 Load이며 다운로드/Processor 준비는 포함하지 않는다.

첫 요청 E2E 4649.49ms, 후속 98개 Warm E2E 평균 4429.37ms / Median 4036.58ms / p95 6704.73ms다. 매 이미지 새 Vision/Prefill을 실행했고 Warmup·별도 Timing 요청은 없다.

출력 Token 분포는{"69": 47, "121": 10, "68": 4, "66": 6, "67": 19, "65": 3, "95": 4, "119": 1, "70": 1, "61": 1, "117": 1, "113": 1, "93": 1}다. Preprocess/Vision/Prefill 근사/Decode Forward 합과 개별 GPU Peak는 [Latency·VRAM JSON](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/grounding_latency_vram.json)에 있다. First-forward는 Vision을 포함하고 Decode Event 합은 전체 Host Loop Wall Time과 다르다. VLM을20Hz Controller로 평가하지 않는다.

Qwen과 SAM2 로드 후 Allocated7.665GiB다. SAM2 실제 Forward Frame12개, 후속 Warm Frame11개, Warm Forward Peak Allocated7.959GiB를 관측했다. Offload·OOM·다른 연구 프로세스 종료는 없다.

## 9. 해석·보존·종료

**상세화가 이번 후보의 성능을 높이지 못했다.** 좌표 계약 채택은76/96이고 Peg/Box Recall은 각각3/56,4/54로 운영 기준90%에 못 미쳤다. SAM2 IoU≥.8은 각각0/12로80% 기준에 못 미쳤다. 두 Camera 중 일부 조건에만 국한된 충분한 성공 근거도 없다. `DINO_REPLACEMENT_CANDIDATE`를 지지하지 않으며 `INSUFFICIENT_SUPPORT` 범위는 여전히 남는다.

관측: 재평가96개 중20개 응답에서 중복 JSON Key가 발생했다. 일부는 같은 Key의 좌표도 서로 달라 임의로 첫 값/마지막 값을 선택하지 않았다. 출력 최대 Token은121로128 한도에 도달하지 않았으므로 이번 실패를 단순 Token 절단으로 설명할 수 없다. Null은0개이고, 완전 비가시 Box2개에서도 Box를 출력했다.

형식 실패만 제거한 사후 진단에서도 새 유효 Peg Box43개의 같은 이미지에서 이전 성공16/43→새 성공3/43이다. Box41개의 같은 이미지에서는4/41→4/41이다. 이 결과는 위치 저하가 Parser 거부만의 결과가 아님을 보인다. 다만 새 출력 유효성으로 고른 사후 부분집합이므로 Primary/독립 성능으로 쓰지 않는다. [진단 Receipt](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/prompt_retry_diagnostics.json).

관측된 위치 실패에는 Wrist에서 BOX를 전체 화면 `[0,0,1000,1000]`으로 넓게 반환한 사례, Peg/Robot 및 Box/Peg Actor 혼동, 두 역할에 동일한 좌표를 준 사례가 있다. 가설로는 긴 지시의 부담, 역할 판단의 불확실성, Native 구조 변경과 양자화된 Vision의 영향이 남지만, 이번에는 이 요인을 분리하는 Ablation을 실행하지 않았으므로 원인을 확정하지 않는다. GT 위치·DINO Proposal·평가 정답을 Prompt에 넣지 않은 역할 설명은 정답 누출에 해당하지 않는다. 다만 이미 본 장면의 실패를 분석한 뒤 수행한 재평가라는 적응 편향은 남는다.

이 비교가 검증한 것은 이 상세 Prompt와 출력 구조/사전 Parser 계약을 적용했을 때의 동작이다. 상세 지시가 항상9B 성능을 높인다는 일반 법칙이나 최적 Prompt를 찾았다는 증거가 아니다. 역할 이해·양자화·Vision Token 예산의 원인은 분리되지 않았고, 추가 Prompt Sweep으로 유리한 결과를 고르지 않았다.

1회 실행을 완료한 뒤 종료했다. 기존 Source/Raw/GT/Stop Stub, 이전 실험, 모델/Processor 등 **1598개 보호 파일**의 SHA와 기존/독립 환경 Package 목록이 같았다. 원래 Stage4A-2 Runner0, 종료 GPU Process0을 확인했다. 공통 ExperimentLogger를 W&B disabled로 사용했고 추론 UUID는 `84669582e5784489b68099364815f9cd`, 지표 UUID는 `43a8f9e988064db8ba5a4d6d0f29c8b6`다.

Production 변경·새 모델 다운로드·새 Simulator Reset/Render/Rollout·Graph 재캐시·Training·Internal/Final 재개·외부 게시·Git Commit은0이다. CUDA는58576054 RTX3090에서 실행했고 macOS MPS/Simulator/Renderer는 실행하지 않았다. 공통 Pipeline 변경이 없어 학습/Simulator 회귀 테스트 대신 실제 요청/Mask·Parser 계약·99요청 예산·입력/모델/원본 Hash·Raw/CSV Readback을 검사했다.

[검증 Receipt](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/verification.json), [Artifact Index](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/artifact_index.json). 원격 Root: `/workspace/task10i_stage4a2_qwen35_detailed_prompt_retry_20261102_v1/`.

## 10. 동결한 상세 Prompt 원문

```text
You are locating physical task objects in a single RGB image of a robot performing peg insertion.

TASK CONTEXT
The manipulation task has two different physical objects: an insertable workpiece and a receiving fixture. The robot is intended to grasp and move the workpiece, align it with the receiving fixture's opening, and insert it. Your task is to locate these two objects directly in the complete image. No candidate regions are provided.

TARGET 1: PEG
- PEG means the physical workpiece that is grasped, moved, and inserted into the receiving opening.
- "Peg" describes its role in the task. It is not a requirement that the object look like a cylindrical pin. A block, bar, or other non-cylindrical component can be the PEG.
- The PEG may be resting on a surface, held between the gripper fingers, approaching the fixture, or partly inserted. Its role does not change when the robot grasps it.
- Locate the actual workpiece, not the robot finger, gripper body, shadow, or nearby fixture.
- If the gripper holds the workpiece, separate the visible workpiece from the fingers touching it. Do not include the fingers just because they touch or surround the workpiece.

TARGET 2: BOX
- BOX means the physical receiving fixture containing the opening into which the PEG is inserted.
- "Box" describes the receiving role. It is not a requirement that it look like an ordinary cardboard container.
- Locate the visible fixture body that contains the receiving opening, not just the empty opening or the hole's interior.
- The opening can be small, viewed at an angle, or partly occluded. If the visible fixture can still be identified reliably, locate its visible body.
- Do not label the insertable workpiece as BOX merely because the workpiece has a block-like shape.

HOW TO DISTINGUISH THE TWO TARGETS
Use the visible scene and the insertion relationship: which physical component is inserted, and which physical fixture receives it? Use shape, boundaries, the opening, contact with the gripper, and the overall scene as visual evidence. No fixed color, material, size, orientation, or image position identifies either role.

The robot arm, gripper, and fingers belong to the manipulator. They are not either target. The support surface, background, shadows, highlights, and empty spaces are not either target. Do not substitute one of these regions when the requested target cannot be located.

The PEG and BOX are distinct physical objects. Identify each separately. Do not assign the same object to both roles merely to fill both output entries. Their image boxes may overlap when the physical objects overlap in the view, for example during insertion; overlap alone is not an error and does not make them the same object.

VISIBILITY AND OCCLUSION
Use only what can be localized from this image. Do not infer an object's position from an assumed usual arrangement or from what ought to be present in a peg-insertion task.

A partially visible target can still be located. If a gripper or fixture occludes part of the PEG, include the visible PEG pixels without expanding the box to an imagined full length. If the image border cuts off an object, enclose only the part inside the image. If several visible pieces clearly belong to the same partly occluded object, one box should enclose those visible pieces.

Return null for a target when it is fully hidden, outside the image, not distinguishable from another object, or cannot be localized reliably. Failure to locate one target does not require returning null for the other target. Do not invent a box just because two target entries are requested.

BOUNDING BOX DEFINITION
For each target, find its leftmost, topmost, rightmost, and bottommost visible extent in the complete original image. Return a tight, axis-aligned rectangle around that visible target. Include the target's visible body, not only its most distinctive small patch. Exclude surrounding background and robot parts as far as a rectangular box permits. Do not add an arbitrary margin.

COORDINATE SYSTEM
- The complete image uses normalized coordinates from 0 to 1000 in each axis.
- x increases from the left edge, 0, to the right edge, 1000.
- y increases from the top edge, 0, to the bottom edge, 1000.
- Every box is [x1, y1, x2, y2]: left, top, right, bottom.
- Require 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.
- Give numeric coordinates for the complete image, not pixel coordinates for an internal resized image and not coordinates relative to a crop.

OUTPUT CONTRACT
Return one JSON array containing exactly two objects, in PEG then BOX order. Each object must have only the keys "label" and "bbox_2d". The label must be "PEG" or "BOX". The bbox_2d value must be four numeric coordinates or null.

Use this structure, replacing the coordinate placeholders with your own predictions or replacing the entire coordinate list with null:
[{"label": "PEG", "bbox_2d": [x1, y1, x2, y2]}, {"label": "BOX", "bbox_2d": [x1, y1, x2, y2]}]

Before answering, check that each non-null box encloses the object with the requested role, that it refers to the full image, that the coordinates are in the required order and range, and that neither box is merely the robot or background.

Output only the JSON array. Do not output coordinate placeholder names, explanations, confidence scores, additional objects, or any other text.
```

![새 Qwen Box·사후 GT·실제 SAM2 Initial Mask의 소규모 비교](../../outputs/task10i/stage4a2/qwen35_detailed_prompt_retry_20261102_v1/full_image_grounding_comparison.png)
