# Task 10-I Stage 4A-2 — Qwen3.5-9B Full-Image Grounding, DINO-Free Evaluation

2026-11-02. **최종 판정: `VLM_GROUNDING_INSUFFICIENT`. DINO 대체: `NOT_SUPPORTED`.**
`DINO_REPLACEMENT_CANDIDATE`로 판정하지 않는다. `PAUSED_SCOPE_REVIEW`와 Stop Stub을 유지했다.

Vast.ai **58576054 / RTX 3090 24GB**에서 원본 전체 RGB 한 장과 고정 Task Prompt만으로
실제 추론했다. 모델 로드 1회, Phase A 3요청, 독립 Parent 24개의 Phase B 96요청을 완료했다.
독립 이미지의 엄격한 JSON 성공은 **2/96=2.08%**였다. 보존 EXACT_GT가 지원하는 범위에서
IoU≥0.5 Peg Recall은 **1/56**, Box Recall은 **0/54**다.

Fence 내부 좌표를 별도로 진단해도 Peg **25/56=44.64%**, Box **9/54=16.67%**였고,
두 물체가 함께 정확한 경우는 **5/54=9.26%**였다. 이 진단은 Accepted Grounding이 아니다.
형식 문제와 함께 위치·의미 혼동이 남았다. 사전 선정 SAM2 대상 16개 중 유효한 입력을 가진
1개 Camera-Frame에서 두 Mask를 실제 생성했다. Peg IoU는 **0.989029**, Box IoU는
**0.000216**이었다. 이 결과는 현재 조건에서 DINO의 Object Discovery/Box 생성 역할을
대체할 근거를 제공하지 않는다.

## 1. 모델·환경·Prompt Identity

먼저 Functional-role, Semantic Verifier Feasibility, Dual Grounding, Initial Candidate
Verification, Paused Scope Review 및 후속 Frozen Verifier Independent Evaluation 보고서를
확인했다. 이전의 후보 영역 분류 성공과 전체 이미지에서 위치를 직접 생성하는 능력을 구분했다.

| 항목 | 실제 실행 |
|---|---|
| 모델 | `Qwen/Qwen3.5-9B` |
| Revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| 모델 클래스 | `Qwen3_5ForConditionalGeneration` |
| 양자화 | BitsAndBytes NF4 4-bit, BF16 compute, Double Quantization ON, uint8 storage |
| Attention / Device | SDPA / `cuda:0`, `device_map={"":0}` |
| Generation | Batch1, Greedy, `enable_thinking=False`, `max_new_tokens=128`, `use_cache=True` |
| Processor | 기존 `AutoProcessor`/`Qwen3VLProcessor`, min/max pixels=65536 |
| 실행 환경 | 기존 독립 `qwen35_verifier_20261102_v1/env` |
| 패키지 | Python3.12.15, Torch2.14.1+cu130, Transformers5.19.0, BitsAndBytes0.50.2 |
| 실제 배치 | Linear4bit 358개, 모든 Parameter/Buffer CUDA, CPU/Disk Weight Offload 없음 |

Embedding/LM Head 등 비양자화 Parameter가 있으므로 모든 Weight가 INT4라는 뜻은 아니다.
기존 모델/Processor 15개 파일과 설치 Source SHA를 대조했다. 설치·다운로드·Kernel Patch는 없다.
기존 Hybrid Attention의 Reference PyTorch Fallback 경고를 보존했다.

사용자가 지정한 Prompt 원문 하나를 UTF-8, 끝 줄바꿈 없이 저장했다. 추가 System Instruction,
이미지 순서 설명, Few-shot, Episode별 규칙은 넣지 않았다. Prompt Sweep·수정은 0회다.

**Prompt SHA256:** `66a8621c89784e51b20be107ad57af726cdf275c34acc04947057cb1ea26c62d`.

[Prompt 원문](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/grounding_prompt.txt),
[Prompt Identity](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/grounding_prompt_identity.json),
[모델·양자화·Source Identity](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/model_identity.json).

## 2. DINO-free 입력과 예측 동결

각 요청의 User Content는 **①원본128×128 RGB 한 장 ②지정 Prompt 텍스트**뿐이다.
원본 HDF5 RGB와 추출 PNG의 Pixel SHA를 연결했다. Rectangle/Crop/Point/멀티이미지,
DINO Proposal·Confidence·Label·Consumer Decision, GT Actor/Segmentation/Pose,
Privileged Position, SAM2 Mask는 Qwen 입력에 없다.

Processor는 실제 Grid `[1,16,16]`으로 내부256×256 이미지를 처리했다. 출력 좌표는
지정 계약대로 원본 전체 이미지의 정규화0–1000으로 해석하고 각 축에128/1000을 곱했다.
Processor 해상도256을 Pixel 변환에 쓰지 않았고, GT를 보고 위치를 이동·확대·Clipping하지 않았다.
사후 좌표 오차가 내부 Resize 오해 때문인지 시각적 추정 오류 때문인지 이 실험으로 분리하지 않는다.

JSON 전체를 엄격히 파싱하고 Key 집합, 중복 Key, 유한한 숫자/Bool 배제,4개 좌표,
0–1000 범위, 양수 Width/Height를 검사했다. Fence나 추가 텍스트는 `INVALID_JSON`이다.
범위 위반은 `INVALID_BOX`이며 보정하지 않는다. 사후 JSON 부분 추출은 진단 열에만 기록했다.

Phase A 3개 응답도 먼저 별도 저장·Hash 동결했다. 기계적으로 읽을 좌표가 존재해 Phase B로
진행했다. **전체99개 원시 응답과198개 Object Prediction CSV를 저장·Hash 동결한 뒤에만
GT를 의미적으로 열었다.** Temporary SAM2도 GT를 열기 전에 완료했다.
기존 DINO 출력은 Qwen 추론과 SAM2 초기화가 끝난 뒤 비교 단계에서 처음 의미적으로 읽었다.
보존 검사의 불투명 GT Byte Hash와 GT 내용 해석은 구분했다. DINO Raw/Event 보존 Hash는
이전 검증 Receipt를 기준으로 동결 후 대조했다.

**전체 원시 응답 SHA256:** `4d1d7414064de64b526a5daa348d02f7d9276f2de999b5993ad1e8f8cfdaf654`.
GT 전후 원시 응답·예측 CSV SHA가 동일하다.

[수정하지 않은 원시 응답·Token ID](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/full_image_grounding_raw_responses.json),
[엄격한 Prediction CSV](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/full_image_grounding_predictions.csv).

## 3. Phase A: P/N/C1 기술 확인

세 응답 모두 JSON 좌표 앞뒤에 Markdown Fence를 출력했다. 엄격한 채택은 전부 실패했다.
다음 IoU는 **Fence 내부 좌표의 진단값**이며, 이3장은 독립 성능에 포함하지 않는다.

| 사례 | Qwen PEG 정규화 좌표 | Qwen BOX 정규화 좌표 | Peg/Box 가시 Pixel | 진단 Peg/Box IoU |
|---|---|---|---|---|
| P 0355/t0 Main | `[200,500,450,800]` | `[350,200,650,500]` |530 /716|.506329 /.332847|
| N 0355/t30 Main | `[380,280,620,520]` | `[420,420,660,660]` |434 /899|0 /.215689|
| C1 0645/t0 Main | `[366,566,633,833]` | `[433,266,566,433]` |849 /517|.667267 /.311614|

N에서 실제 Peg는 **434 Pixel 보였다**. 정답을 null로 가정하지 않았다. Qwen의 PEG Rectangle은
Peg0 / Box664 / Robot0 / Background297 Pixel을 포함해 실제 Box를 Peg로 Grounding했다.
기능적 역할 설명만으로 이전의 잘못된 물체 지정을 해결하지 못한 국소 사례다.
P/C1에서는 Peg 위치에 접근했지만 Box 위치가 부정확했다. Strict JSON 실패와 위치 진단을 합산하지 않았다.

## 4. Phase B: Parent 선정·Frame·Camera·GT 지원

Stage3 최종 `dataset_R_train_manifest.json`/`dataset_D_train_manifest.json`의 물리 Parent를
합집합으로 중복 제거했다. 공통 Validation과 Prompt 개발 Parent0355/0645를 제외한
**적격112개**를 `SHA256("Task10I-QwenFullImageGrounding-20261102-v1|" + parent_id)`로
정렬해 첫24개를 선택했다. GT/Reference 보유 여부, 가시성, Box 위치, DINO 성공,
Qwen 결과는 선정에 사용하지 않았다. 과거 Semantic 독립 평가 Parent를 Prompt 개발 Parent로
간주해 임의 제외하지 않았으며, 그 자료와 완전히 불연속인 외부 Blind Test라고 주장하지 않는다.

```text
10i_s1_0362, 10i_s1_0157, existing_30082, 10i_s1_0921, 10i_s1_0844, existing_30032
10i_s1_0767, existing_30124, 10i_s1_0305, 10i_s1_0449, 10i_s1_0327, 10i_s1_0044
10i_s1_0686, 10i_s1_0720, 10i_s1_0621, existing_30017, 10i_s1_0193, 10i_s1_0924
10i_s1_0197, 10i_s1_0531, 10i_s1_0134, 10i_s1_0023, 10i_s1_0275, existing_30040
```

각 Parent의 `full_task`에서 F0=t0, F1=t80을 사용했다.23개는 t80이 있었고,
**0134는 저장 마지막 시점 t78**을 사전 대체 규칙으로 사용했다. 모든 Frame에 Main
`base_camera`와 Wrist `hand_camera`를 독립 요청으로 넣어 **24×2×2=96개 이미지**를 완료했다.
입력 누락·추론 Runtime 실패는0이다. Manifest는 추론 전에 동결했고 이후 Parent/Frame을 바꾸지 않았다.

**Input Manifest SHA256:** `5838c4095bd0b637056dad851073bb42b1056b43ed7cfdb5f7bec220a016b14a`.
[전체 Parent/Frame/Camera·RGB Hash·SAM2 선정 ID](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/input_manifest.json).

GT는 저장된 Actor Segmentation과 Receipt의 Raw SHA/Segmentation SHA/`exact_frame`을
검증해 재사용했다. 선택한 입력은 모두 **현재 Stage3 Train**이다. 일부 GT Cache가 이전 Stage의
`internal_test` 경로에 보관돼 있어도 이번 선정 Train Raw의 SHA가 일치할 때만 사용했다.
Stage4 신규 Final/Test 입력·수집·평가는 사용하지 않았다.

| GT 지원 | 독립 이미지 |
|---|---:|
| EXACT_GT |56 /96,15개 Parent에서 지원|
| 보존 Actor Segmentation 없음 |36 /96|
| 보존 Reference의 해당 Frame이 EXACT 아님 |4 /96|

40개 미지원 이미지는 정확도 분모에서 제외하고 형식/Latency 및 누락 원인은 유지했다.
가시 Pixel≥16을 평가 가능으로 고정했다. 지원 이미지에서 Peg는56/56개가 평가 가능했고,
Box는54개 평가 가능·2개 완전 비가시였다. `TINY_VISIBLE`(1–15 Pixel)은0개다.
따라서 작은 물체나 Peg 비가시 조건의 성능은 `INSUFFICIENT_SUPPORT`다.
신규 Simulator Reset/Render/Replay로 GT를 보충하지 않았다.

## 5. Primary 결과: 엄격한 채택 계약

Detection 성공은 평가 가능 GT 가시 Bounding Box와 **IoU≥0.5**인 유효 Box다.
Box를 출력했다는 사실만의 Emission Recall과 위치까지 맞춘 Detection Recall을 분리했다.
Null·Invalid·위치 실패는 성공으로 세지 않는다. GT Box는 Actor Pixel의 half-open
`[xmin,ymin,xmax+1,ymax+1]`이며 연속 좌표 IoU를 계산했다.

| 항목 | Peg | Box |
|---|---:|---:|
| 평가 가능한 Visible Object |56|54|
| 유효 Box Emission Recall |1/56|1/54|
| **Detection Recall (IoU≥.5)** |**1/56 (1.79%)**|**0/54 (0%)**|
| Missing Detection |55/56 (98.21%)|54/54 (100%)|
| IoU≥.3 /≥.5 /≥.7 |1/56 /1/56 /1/56|0/54 /0/54 /0/54|
| Mean IoU, Invalid/Miss=0 |.016076|.002503|
| Mean IoU, 유효 출력·GT 지원만 |.900281 (n=1)|.135135 (n=1)|
| Center Error, 유효 출력·GT 지원만 |3.041px (n=1)|60.632px (n=1)|
| 비가시 Object에 유효 Box 출력 |분모0: 미판정|0/2|
| False Discovery: 유효 출력 중 실패 |0/1|1/1|

JSON 성공은 **2/96 이미지**, 좌표 계약 성공은 **4/192 Object Slot**이다.
성공한 이미지 둘은0193/t0/Wrist와0134/t78/Wrist다.0193에는 보존 GT가 없어 정확도를 판정하지 않았다.
94/96개 이미지의 응답은 Fence를 포함했고 나머지2개는 순수 JSON이었다. 사후 추출 좌표는
96/96개 모두 수치·범위·순서 계약을 만족했으나 이를 Primary에 승격하지 않았다.
Raw나 진단에서 null은0/192 Slot이다. 불확실하면 Abstain한다는 능력은 입증되지 않았다.

Scene-level 두 물체 동시 성공은 **0/54**, 잘못된 유효 Box를 하나 이상 출력한 경우는 **1/56**이다.
후자의 낮은 수치는 대부분의 출력이 형식에서 거부됐기 때문이며 높은 인식 안전성을 뜻하지 않는다.
Task Actor 지배 기반의 명백한 Peg↔Box/Robot 혼동은 Primary 유효 출력 각1개에서0건이다.
이는 **공간적으로 틀린 Box가 없다는 의미가 아니다**. 실제 Box 출력1개는 IoU .135였다.

[전체 Metric·분모·Parent별 결과](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/grounding_evaluation_metrics.json),
[실패·미지원 사례 CSV](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/grounding_failure_cases.csv).

## 6. 사후 좌표 진단: 형식 실패 외의 위치·의미 문제

아래 수치는 Fence를 제거해 읽은 좌표의 진단이다. Accepted Grounding이나 SAM2 입력으로
사용하지 않았다. 잘못된 좌표의 GT 맞춤 보정은 없다.

| 지표 | Peg | Box |
|---|---:|---:|
| IoU≥.3 |36/56 (64.29%)|23/54 (42.59%)|
| **IoU≥.5** |**25/56 (44.64%)**|**9/54 (16.67%)**|
| IoU≥.7 |6/56 (10.71%)|2/54 (3.70%)|
| IoU Mean /Median |.401423 /.385384|.275177 /.270619|
| IoU P05–P95 |.018705–.754630|.004245–.606066|
| Center Error Mean /Median |15.575 /8.352px|31.226 /19.598px|
| IoU<.5 또는 비가시 False Discovery |31/56|47/56|
| 명백한 비대상 Task Actor 지배 |6/56|10/56|
| 비가시 Object에 Box 생성 |분모0|2/2|

Semantic Confusion은 Peg→Box3, Peg→Robot3, Box→Peg7, Box→Robot3이다.
이 지표는 대상 IoU<.5이고 예측 Rectangle 내 다른 Task Actor가16Pixel 이상이면서
Task Foreground의70% 이상을 차지하는 경우다. 애매한 영역이나 배경 실패를 억지로
특정 Actor 오분류로 부르지 않는다. 전체 Pixel 구성은 Object Evaluation CSV에 남겼다.

두 물체 동시 정확은 **5/54**이고, 하나 이상의 위치 실패/비가시 Box 출력은 **51/56**,
하나 이상의 명백한 다른 Task Actor 지정은 **13/56**이다. 단순한 이미지 내부 좌표 유효성과
정확한 Object Grounding은 크게 달랐다.

| 구분 | Primary Peg /Box Recall | 진단 Peg /Box Recall | 진단 두 물체 동시 성공 |
|---|---|---|---|
| Main |0/28 /0/28|13/28 /7/28|4/28|
| Wrist |1/28 /0/26|12/28 /2/26|1/26|
| F0=t0 |0/30 /0/29|16/30 /7/29|3/29|
| F1=t80,0134=t78 |1/26 /0/25|9/26 /2/25|2/25|

특히 진단 Box는 Wrist2/26, F1 2/25로 낮다. F1의 명백한 Task Actor 오지정 Scene은12/26이며,
F0는1/30이었다. 서로 다른 가림·물체 크기·장면 구성을 통제한 인과 비교는 아니다.

## 7. Parent 단위 통계

동일 Parent의 Frame/Camera를 독립 Parent로 계산하지 않았다. 아래 Parent-macro는
해당 지표의 지원이 있는15개 Parent를 동일 가중했다. Bootstrap Seed42,10,000회다.

| 지표 | Parent-macro | Macro95% CI | Parent-cluster Image-rate95% CI |
|---|---:|---|---|
| Primary Peg Recall |1.67%|0–5.00%|0–5.77%|
| Primary Box Recall |0%|0–0%|0–0%|
| 진단 Peg Recall |46.67%|36.67–58.33%|35.00–55.17%|
| 진단 Box Recall |16.67%|6.67–26.67%|7.14–26.98%|
| 진단 두 물체 동시 성공 |10.00%|3.33–18.33%|2.86–16.28%|

JSON/좌표 형식은24개 Parent 전부 지원하며 Macro2.08%, CI0–5.21%다.
모든 Parent별 실제 분모와 Camera×시점별 결과도 JSON에 있다.0–0%처럼 퇴화한 Bootstrap은
관측 표본의 결과이며, 희귀 오류 확률의 상한이나 안전성 보증이 아니다.
비가시 Box 분모2개·Peg 비가시 분모0·Tiny 분모0에서 일반화하지 않는다.

## 8. Phase C: 실제 Temporary SAM2 초기화

Qwen 결과를 보기 전에 별도 ID Hash 순서로 Main/F0, Main/F1, Wrist/F0, Wrist/F1 각4개,
총16개 Camera-Frame을 고정했다. 결과가 틀렸거나 형식에 실패한 Frame을 교체하지 않았다.
GT가 지원하는 대상은12개 Frame/각 Object12개이고, 선정의 독립 물리 Parent는12개,
평가 가능 GT를 가진 Parent는9개다.

기존 SAM2.1 Hiera Small Weight SHA
`6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`와
기존 Config/Processor를 사용했다. 원본 RGB를 기존 경로대로 OpenCV RGB→BGR JPEG로
저장하고 현재 Frame 하나만 새 State의 frame0으로 넣었다. 형식·좌표가 유효한 Qwen Box만
전달했다. GT/DINO Box나 사후 복구 좌표로 대체하지 않았다.

선정16개 중 **0134/t78/Wrist 한 Frame만** 유효했다. 실제 Temporary State1개,
Object Box Prompt2개다. 나머지15개는 `INVALID_JSON`으로 미초기화했다.
State 내 즉시 Initial Mask만 읽고 Reset/폐기했다. Propagation·장기 Tracking·원래 Memory Commit은 없다.

| 실제 초기화 | Peg | Box |
|---|---:|---:|
| Qwen Pixel Box |`[64,32,128,128]`|`[0,32,64,128]`|
| GT 가시 Pixel |6014|5161|
| Box IoU |.900281|.135135|
| SAM2 Mask Pixel |5952|4096|
| **Mask IoU** |**.989029**|**.000216**|
| Pixel Precision /Recall |.999664 /.989358|.000488 /.000388|
| Mask 내 Peg /Box /Robot /Background |5950 /0 /2 /0|0 /2 /0 /4094|
| Robot /Background Contamination |.0336% /0%|0% /99.9512%|

Box Mask는 사실상 배경을 초기화했다. JSON이 정상이고 Box가 이미지 내부에 있다는 이유로
성공 처리하지 않았다. `wrong_object` Flag의 비대상 Task Actor 지배 조건에는 걸리지 않지만,
Background Contamination과 Mask IoU로 분명한 잘못된 초기화다.

| End-to-end 성공: IoU≥.8 | Peg | Box |
|---|---:|---:|
| GT 평가 가능 대상, Invalid도 실패 포함 |**1/12 (8.33%)**|**0/12**|
| Main |0/6|0/6|
| Wrist |1/6|0/6|
| F0 |0/8|0/8|
| F1 |1/4|0/4|
| Parent-macro, 지원9 Parent |5.56%|0%|
| Parent-macro Bootstrap95% CI |0–16.67%|0–0%|
| Parent-cluster Image-rate95% CI |0–25%|0–0%|

실제로 초기화한1개 Peg의 높은 IoU를 전체 초기화 능력으로 확대하지 않는다.
Mask 품질 조건부 표본은 Object별1개뿐이다. GT 미지원4개를 성공/실패 정확도로 강제 판정하지 않았다.
기존 SAM2 `_C` 확장 부재로 Fill-hole 후처리가 생략되는 경고를 보존했고 환경을 패치하지 않았다.

[SAM2 전체 지표](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/sam2_initialization_metrics.json),
[선정16개·Object32개 처분/Mask 평가](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/sam2_initialization_cases.csv).

## 9. 실제 Load·Latency·VRAM

VLM은 초기 Discovery/Re-grounding 용도로 해석한다.20Hz 실행 요구를 적용하지 않았다.
요청마다 Vision Encoding과 Prefill을 새로 수행했고 Warmup/Timing용 추가 요청은0이다.

| Load/상주 | 측정 |
|---|---:|
| Qwen Weight Load |3414.69ms,1회|
| Qwen 로드 후 Allocated /Reserved |7.349 /7.406GiB|
| Qwen 요청 중 Peak Allocated /Reserved |7.581 /7.629GiB|
| CPU Peak RSS:50ms 표본 /OS ru_maxrss |18.282 /18.492GiB|
| Qwen+SAM2 로드 후 Allocated /Reserved |7.667 /7.676GiB|
| 두 모델 상주 중 SAM2 Forward Peak Allocated /Reserved |7.960 /8.182GiB|
| 해당 Forward 완료 시 Device Free |15.058GiB|

Checkpoint Hash 읽기로 File Cache가 Warm한 Load다. 다운로드·Processor 준비 시간은 제외했다.
CPU Peak는 Process RSS이며 GPU Memory와 별개다. OOM·Weight Offload·다른 연구 프로세스 종료는 없다.

| 요청 구간 | 표본 | Mean | Median | p95 |
|---|---:|---:|---:|---:|
| 첫 요청 P E2E |1|4199.93ms|해당 없음|해당 없음|
| 후속 Warm E2E |98|3303.63ms|3309.18ms|3451.29ms|
| 독립 Phase B E2E |96|3302.66ms|3308.61ms|3456.19ms|
| Warm Preprocess |98|4.16ms|4.00ms|5.07ms|
| Warm Vision CUDA |98|28.43ms|28.09ms|30.39ms|
| Warm 첫 Forward: Vision+Prefill CUDA |98|267.51ms|266.09ms|270.97ms|
| Prefill 근사: 첫 Forward−Vision |98|239.07ms|237.69ms|242.14ms|
| Warm 후속 Decode Forward CUDA 합 |98|3011.85ms|3021.03ms|3133.83ms|

출력은 종료 Token 포함 **55–62 Token**, 최대128에 도달한 응답은0개다.
첫 요청은 Cold Kernel/Processor 효과를 포함한다. E2E는 PNG 읽기·Processor·H2D·Generate·
Text Decode까지이며 산출물 저장은 제외했다. CUDA Synchronization과 실제 Forward Hook/Event를
사용했다. Vision은 첫 Forward 안에 포함되므로 합산하지 않는다. Prefill 차감값은 근사치이며
별도로 순수 Language Prefill만 분리 계측한 값은 아니다. Decode Event 합도 Host Generation
Loop 전체 Wall Time과 다르다. 기존 단일 Label의 약430ms와 동일한 비용으로 간주하지 않는다.

SAM2 로드는793.48ms였다. 실제 실행한 Camera-Frame의 두 Object Prompt 합은112.64ms로,
첫 실제 Forward의 초기 비용이 포함된다. **Warm SAM2 Forward 표본은0개**다. Raw Timing의
`cold_first_SAM2_frame`은16개 선정 Loop의 첫 인덱스 표시여서 실제 첫 Forward를 표시하지
못했으며, 이 의미를 비용 JSON에 명시했다. Warm Forward 수치라고 보고하지 않는다.
두 모델의 단기 GPU 동시 상주는 확인했지만, 같은 시점의 추가 Qwen 요청이나 장시간 동시
추론·Controller/DINOv2 상주는 실행하지 않았다.

[요청별 비용·상주·Peak](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/grounding_latency_vram.json).

## 10. 보존 DINO와 동일 Frame 비교

Qwen 결과 동결 뒤 보존 B0 DINO Event를 Episode·Camera·RGB SHA로 매칭했다.
**6개 Parent×t0×2 Camera=12개 공통 이미지**에서 Peg/Box 양쪽 Raw가 있었다.
F1 공통 자료는 없었다. 새 DINO Forward는0회다.

기존 Area<.25·Top5·Null .3·Exclusion IoU .3의 변경 없는 `assign`을 저장 Proposal에 적용했다.
이는 공통 Frame의 Joint 초기화 재구성이며 Detector 자체의 모든 Proposal Recall과 구분한다.
실제 사후 Late Consumer 결정을 재현했다는 주장도 하지 않는다.

| 공통12개 이미지,6 Parent | DINO Joint 재구성 | Qwen Strict |
|---|---:|---:|
| Peg IoU≥.5 Recall |7/12|0/12|
| Box IoU≥.5 Recall |7/12|0/12|
| Peg Mean Box IoU |.522089|0|
| Box Mean Box IoU |.535617|0|

참고로 저장 Postprocess Proposal의 Score Top1만 사용하면 Peg5/12, Box7/12다.
동일 Frame SAM2 Mask 비교와 Detector Latency 비교는 지원되지 않는다. 다른 실험의
전체 정확도나 Detector Timing을 끌어와 이번 수치와 비교하지 않았다.

최종 비교 지원 상태는 **`MATCHED_DINO_BASELINE_INSUFFICIENT`**다. 위 제한된 t0 공통
부분집합의 관측값은 공개하지만 전체24 Parent/후기 상태에 대한 상대 우열을 일반화하지 않는다.
Qwen이 DINO보다 우수하다는 근거는 없다.

[공통 Frame·Raw Hash·동일 분모 비교](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/.execution/dino_matched_comparison.json).

## 11. 실패 관측과 해석의 범위

**관측:**94/96개 독립 응답이 Fence를 포함했다. 진단 좌표가 모두 수치 계약을 만족해도
Box Recall9/54, Wrist Box2/26, F1 Box2/25로 낮았다. N에서는 Box를 Peg로 지목했고,
독립 진단에서도 Peg/Box/Robot의 명백한 교차 지정이 있었다.0134의 BOX는 대부분 배경을
분할했다. 따라서 JSON만 허용하면 해결된다고 결론 내릴 수 없다.

**가설:**128² 원본의 작은 물체와 부분 가림, 기능적 역할의 정지영상 추론, 좌표 지시 준수의
불안정성 등이 함께 영향을 줬을 수 있다. 양자화·Prompt 길이·Processor Resize 중 무엇이
원인인지 통제 비교하지 않았다. 고해상도·비양자화·다른 Prompt/모델에서의 가능성을 반증하지 않는다.
역할 분류가 가능한 특정 Crop 사례와 Zero-shot Full-image Localization은 별도 능력이다.

## 12. DINO 대체 운영 기준

| 사전 기준 | 관측 | 판단 |
|---|---|---|
| Peg Recall≥90% |1/56=1.79%|FAIL|
| Box Recall≥90% |0/54|FAIL|
| 잘못된 Peg Grounding≤5% |Primary 유효·GT 지원1개에서0건; 진단 공간 실패31/56|희귀 오인식 검증은 INSUFFICIENT_SUPPORT|
| 유효 좌표 출력≥99% |엄격한 전체 응답2/96=2.08%|FAIL|
| SAM2 Peg/Box IoU≥.8 각각≥80% |1/12 및0/12|FAIL; 조건부 Mask 표본은 각각1개|
| Main/Wrist 심각한 집중 실패 없음 |Main Primary0/28, Wrist Box 진단2/26|미충족|

선정24개 중 GT 지원15 Parent, 실제 Mask2개, 비가시/Tiny 지원 부족 때문에 표본 범위
전체의 완전한 성능·희귀 오인식 안전성을 주장할 수 없다. 그러나 현재 관측된 여러 필수 기준
실패 자체는 명확하다. **최종 `VLM_GROUNDING_INSUFFICIENT`, `DINO_REPLACEMENT_CANDIDATE` 아님.**
Runtime은99요청을 정상 완료했고 일부 Strict Box와 실제 Mask가 있어 전체를
`GROUNDING_RUNTIME_BLOCKED`로 분류하지 않았다.

장기 Tracking, Re-grounding 반복, Graph 재생성, 실제 Closed-loop Success, 실환경·외부 분포,
안전한 Abstention, 장기 상주 경쟁은 검증하지 않았다. Production DINO 제거·통합·Threshold 변경은 없다.

## 13. 보존·검증·완료 범위

원격 Output Root는
`/workspace/task10i_stage4a2_qwen35_full_image_grounding_20261102_v1/`다.
새 독립 실행 Source는 그 안의 `.execution/`에 보존했다. 기존 Runner를 호출하지 않았다.

모델/Processor15개, 기존 Source/Raw/GT/Stop Stub과 이번 참조 자료 등 **1,449개 보호 파일**의
시작 기준/종료 SHA가 일치했다. DINO Raw/Event는 이전 검증 Hash와 동결 후 대조했다.
독립·기존 연구 Python Package 목록도 동일했다. 종료 GPU Compute Process0,
원래 Stage4A-2 Runner0을 확인했다. 보호 파일 변경·Dataset/Checkpoint 수정은0이다.

공통 `utils.experiment_logger.ExperimentLogger`를 W&B disabled로 재사용했다.
추론 UUID는 `60ef29dbce0c4e8eac8b4e92c68e925b`, 사후 지표 UUID는
`c59176927336427a8be2f2b315a7d7ad`다. Local Artifact Manifest/Sidecar를 유지했고 Publisher는
실행하지 않았다. 팀별 Logger·외부 게시·Git Commit은 없다.

실제 모델 요청/Mask Readback 외에 JSON 중복 Key·Bool·범위·제로 크기·Fence 거부·128/1000
좌표 변환 계약,99요청/24 Parent/16 Camera-Frame 예산, Raw↔CSV 동일성, GT Barrier,
입력/모델/환경 SHA, SAM2 Mask Shape/Hash, 산출물 JSON/CSV·비교 PNG를 검증했다.
Production/Common 코드 변경이 없어 Simulator/Training 회귀 테스트는 실행하지 않았다.
CUDA 모델은 지정 RTX3090에서 실행했고, macOS MPS·Simulator·PhysX/Renderer는 실행하지 않았다.

실행 중 파일 전송 연결 종료와 원격 rsync 권한 제한은 SCP/Tar 전송으로 해결했다.
모델 Process는99요청·SAM2 단계까지 정상 종료했다. GPU OOM·Generation Retry·재로드는0회다.

| 실행 예산 항목 | 실제 |
|---|---:|
| Qwen Weight Load |1|
| 기존3장 /독립 추론 |3 /96|
| SAM2 선정 Camera-Frame /실제 State /Box Prompt |16 /1 /2|
| 추가 Prompt /새 모델 /새 DINO Forward |0 /0 /0|
| Simulator Rollout /Training /Graph 재캐시 |0 /0 /0|
| Internal/Final·기존 Stage4A-2 Runner 재개 |0|

[보존·계약·종료 검증](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/verification.json),
[산출물 Index·SHA](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/artifact_index.json).

다음 단계나 Production 변경은 시작하지 않고 이 독립 실험을 종료한다.

![Qwen Strict/진단 Bounding Box, 사후 GT와 실제 SAM2 Initial Mask의 소규모 비교](../../outputs/task10i/stage4a2/qwen35_full_image_grounding_20261102_v1/full_image_grounding_comparison.png)
