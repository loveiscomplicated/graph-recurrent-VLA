# Task 10-I Stage 4A-2 — Qwen3.5-9B 세 가지 정밀도 비교

**세 구성 모두 `VLM_GROUNDING_INSUFFICIENT`이며 DINO 대체 기준에 미달했다.** 다만 정밀도 변경은 Peg 성능에 영향을 보였다. 보존된 전체 NF4의 Peg Recall 3/56에서 전체 BF16은12/56, Vision·Merger BF16+언어 NF4는7/56, Vision·Merger BF16+언어 INT8은15/56으로 증가했다. Box는 각각7/54,8/54,6/54였다. SAM2 IoU≥.8 성공은 BF16과 혼합 NF4에서 Peg1/12·Box2/12, 혼합 INT8에서 각각3/12였다.

보존 전체 NF4 대비 Peg Recall 차이의 Parent-cluster95% CI는 BF16에서+4.76~+28.26%p, 혼합 INT8에서+8.33~+35.42%p였다. Vision·Merger만 복원한 혼합 NF4는0~+16.13%p로 효과가 더 불확실하고, Box는 세 구성 모두 차이 구간에0을 포함했다. 알려진15개 GT 지원 Parent의 조건부·기술적 비교이며 다중 비교 보정은 적용하지 않았다. INT8이 BF16보다 본질적으로 우수하다는 결론도 내리지 않는다. 정밀도 복원만으로 이 고정 조건의 Grounding 문제를 해결하지 못했다.

2026-11-02. 사용자가 승인한 전체 BF16 / Vision·Merger BF16+언어 NF4 / Vision·Merger BF16+언어 INT8 구성을 기존 방식으로 실제 측정했다. Production과 기존 `PAUSED_SCOPE_REVIEW`를 유지했다.

## 1. 비교가 답하는 질문

이번 비교는 같은 Qwen3.5-9B의 기능적 역할 Grounding 실패가 가중치 양자화 정밀도를 바꾸면 얼마나 달라지는지 측정한다. 세 구성 사이에서 프롬프트·출력 계약·Parser·이미지·해상도·Generation·SAM2 선정은 고정했다. 이전에 보존한 전체 NF4 실행도 같은 상세 Prompt의 참고 기준으로 사용했으며 다시 실행하지 않았다.

**이미 실패 분석에 사용한 같은 24개 Train Parent의 RETEST다.** 독립적인 새 일반화 성능으로 해석하지 않는다. 원래 짧은 기능적 역할 Prompt의 좌표 진단과 이번 세 구성의 점수를 직접 비교해 양자화 효과라고 부르지 않는다.

## 2. 모델·Prompt·실행 계약

| 항목 | 고정값 |
|---|---|
| 모델 /Revision |Qwen/Qwen3.5-9B /c202236235762e1c871ad0ccb60c8ee5ba337b9a|
| 입력 |원본128×128 전체 RGB 한 장+직전 상세 Task Prompt|
| Prompt |889단어, 이미지 포함1292 Input Token; SHA256 `40b8c68e18ab9d588f86760ea5023cf0366e85e216fe7d94e18f749348f6f3cb`|
| Processor |같은 AutoProcessor 소스, min/max pixels65536; 내부256×256, grid[1,16,16], 이미지64 Token|
| 실행 |cuda:0, SDPA, Batch1, Greedy, enable_thinking=False, max_new_tokens=128, use_cache=True|
| 좌표 |원본 전체 이미지에 대한0–1000 xyxy; Pixel=normalized×128/1000; Clipping 없음|
| 출력 Parser |직전과 동일한 PEG/BOX bbox_2d 배열; Bare JSON 또는 완전한 단일 JSON Fence; 중복 Key/설명문/부분 추출 거부|
| 평가 |IoU≥.5 Grounding 성공, 가시 Pixel≥16; Tiny1–15 Pixel 별도; 미검출=IoU0|

환경: `{"torch": "2.14.1+cu130", "transformers": "5.19.0", "bitsandbytes": "0.50.2", "accelerate": "1.15.0", "cuda": "13.0", "python": "3.12.15 (main, Oct 13 2026, 16:52:04) [GCC 15.3.0]", "GPU": "NVIDIA GeForce RTX 3090", "bf16_supported": true, "CPU_threads": 4}`. 기존 원본 Checkpoint에서 각 구성별로 한 번 로드했다. 양자화된 값을 BF16으로 재캐스팅하지 않았다. 새 Weight/Processor/모델을 다운로드하지 않았다.

| 구성 | Vision·Merger | 언어 Linear | 실제 4-bit /8-bit Linear |
|---|---|---:|---:|
|전체 BF16|BF16, 양자화0개|BF16|0 /0|
|Vision·Merger BF16 / 언어 NF4|BF16, 양자화0개|NF4, Double Quant ON|248 /0|
|Vision·Merger BF16 / 언어 INT8|BF16, 양자화0개|LLM.int8, threshold6|0 /248|

설치된 Transformers의4-bit/8-bit 변환 코드를 직접 확인하고, 사용자 제외 목록을 설정할 때 사라질 수 있는 기본 `lm_head` 제외도 보존했다. 혼합 구성의 제외 목록은 `lm_head`, `model.visual`이다. Meta 모델 사전 검사와 실제 로드 후 모듈 검사에서 Vision Linear110개와 내부 Merger가 BF16이며 언어 Linear248개만 해당 양자화 타입임을 검증했다. `loaded_model_audit.json`에 모듈별 실제 Type/Dtype/Device를 저장했다.

Checkpoint Header 전체의 BF16 환산 용량은17.98GiB였고 실행 아키텍처의 가중치는 약17.53GiB다. CPU Meta/Checkpoint Key 대조에서 추가15개 Key가 모두 `mtp.*`이며 필수 Key 누락과 Shape 불일치는0개임을 확인했다. 기존과 같은 ConditionalGeneration 클래스·일반 Generation 경로를 유지했고 MTP를 새로 활성화하지 않았다. 이는 추가 Weight Load나 GPU Forward가 없는 Header/Meta 검사다. [대조 Receipt](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/.execution/checkpoint_accounting.json).

언어 INT8 구성은 로드 Dtype과 Vision 가중치를 BF16으로 유지하지만, 설치된 bitsandbytes의 `MatMul8bitLt`가 양자화 과정에서 BF16 입력을 FP16으로 Cast한다는 기본 경고를 출력했다. 실제8-bit 커널의 동작을 그대로 유지하고 로그에 보존했다. 전체 연산이 BF16이라는 의미로 해석하지 않는다.

## 3. 대상·동결·보존

각 구성에서 기술 확인 P(0355/t0 Main), N(0355/t30 Main), C1(0645/t0 Main)3장 후 같은24개 Parent×F0/F1×Main/Wrist96장을 실행한다. F0=t0, F1은23 Parent에서t80이고0134에서 사전 규칙의t78이다. 신규 Frame/Parent 선택, Final/Test 사용, Simulator 실행은 없다.

기존 입력 Manifest와 SAM2 선정16개를 재사용했다. SAM2는 Main/t0, Main/F1, Wrist/t0, Wrist/F1 각각4개이며 전체12개 물리 Parent다. 결과를 보고 성공 Frame만 고르지 않았다.

GT 지원도 고정했다: EXACT_GT56/96 이미지,15개 Parent; Peg56개와 Box54개가16 Pixel 이상, Box2개는 완전 비가시, Tiny0개,40개 이미지는 GT 미지원이다. SAM2 선정 중 GT 지원은12개 Camera-Frame /9개 Parent이며 각 Object 분모12다. GT 미지원 사례를 실패나 성공으로 임의 대체하지 않았다.

**세 구성의 새 예측에 대한 GT 채점과 보존 DINO 비교는 모든 추론 프로세스가 종료하고 원시 응답·예측 CSV의 SHA를 저장한 뒤에 실행했다.** 추론 프로세스에는 GT/DINO 의미 정보를 읽는 경로가 없다. Qwen에는 RGB1장+Prompt만 전달했다. DINO Proposal/Confidence/Label, SAM2 Mask, GT Actor/Position/Pose, Crop/Point/Rectangle Marker는 입력하지 않았다. [전체 동결 Barrier](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/all_predictions_frozen.json).

이 Barrier는 이번 추론·채점 Pipeline의 경계이며 연구자가 과거 GT를 전혀 보지 않았다는 뜻은 아니다. 이전 결과와 GT는 이미 알려져 있고, 준비 중 보존된 전체 NF4의 예전 CSV로 좌표 산술도 재확인했다. 이 확인에 새 구성의 예측을 사용하지 않았으며 프롬프트·구성·선정은 바꾸지 않았다. 독립 Blind 평가와 구분한다.

## 4. 형식·발견·위치 성능

이전 참고 기준도 같은 상세 Prompt와 같은 Parser이므로 아래 NF4 비교는 Parser 정책 변경을 포함하지 않는다. Recall은 평가 가능한 가시 물체에 대한 IoU≥.5 성공률이다.

| 구성 | Bare JSON | 채택 JSON | Peg Recall | Box Recall | Peg/Box Mean IoU |
|---|---:|---:|---:|---:|---:|
|보존 전체 NF4|76/96 (79.17%)|76/96 (79.17%)|3/56 (5.36%)|4/54 (7.41%)|0.1526 /0.1889|
|전체 BF16|2/96 (2.08%)|82/96 (85.42%)|12/56 (21.43%)|7/54 (12.96%)|0.2633 /0.1953|
|Vision·Merger BF16 / 언어 NF4|83/96 (86.46%)|83/96 (86.46%)|7/56 (12.50%)|8/54 (14.81%)|0.1883 /0.2038|
|Vision·Merger BF16 / 언어 INT8|24/96 (25.00%)|81/96 (84.38%)|15/56 (26.79%)|6/54 (11.11%)|0.2725 /0.1917|

### 전체 BF16

판정: `VLM_GROUNDING_INSUFFICIENT`. 두 물체 동시 정확 2/54 (3.70%); 하나 이상의 잘못된 채택 Box 47/56 (83.93%).

| 지표 | Peg | Box |
|---|---:|---:|
|유효 Box 출력, 가시 물체 기준|49/56 (87.50%)|47/54 (87.04%)|
|Missing Detection, IoU≥.5 미달 포함|44/56 (78.57%)|47/54 (87.04%)|
|False Discovery, 유효 출력 중 실패|37/49 (75.51%)|42/49 (85.71%)|
|완전 비가시 시 Hallucinated Box|INSUFFICIENT_SUPPORT (0)|2/2 (100.00%)|
|의미 혼동|{"peg->box": 11, "peg->robot": 1}|{"box->peg": 11, "box->robot": 8}|
|IoU≥.3|24/56 (42.86%)|15/54 (27.78%)|
|IoU≥.7|4/56 (7.14%)|4/54 (7.41%)|
|IoU Median, 미검출0 포함|0.1981|0.0823|
|IoU p05~p95|0.0000~0.7868|0.0000~0.7312|
|Center Error Mean, 유효 출력만|20.2243px|33.7501px|

좌표 계약 164/192 (85.42%); Null Object Slot 0/192. JSON 분모96은 이미지, 좌표/Null 분모192는 Object Slot이며 서로 독립 Parent 수가 아니다.

| Parent-macro | 평균 | Macro95% CI | 지원 Parent |
|---|---:|---:|---:|
|Peg Recall|20.00%|10.00%~31.67%|15|
|Box Recall|13.89%|5.00%~23.89%|15|
|좌표 계약|85.42%|78.12%~91.67%|24|

| 보존 전체 NF4 대비 | 실패→성공 | 성공→실패 | Parent-macro Recall 차이 | Image-rate 차이의 Parent-cluster95% CI |
|---|---:|---:|---:|---:|
|peg|11|2|15.00%|4.76%~28.26%|
|box|5|2|6.67%|-2.08%~13.79%|

### Vision·Merger BF16 / 언어 NF4

판정: `VLM_GROUNDING_INSUFFICIENT`. 두 물체 동시 정확 1/54 (1.85%); 하나 이상의 잘못된 채택 Box 48/56 (85.71%).

| 지표 | Peg | Box |
|---|---:|---:|
|유효 Box 출력, 가시 물체 기준|49/56 (87.50%)|47/54 (87.04%)|
|Missing Detection, IoU≥.5 미달 포함|49/56 (87.50%)|46/54 (85.19%)|
|False Discovery, 유효 출력 중 실패|42/49 (85.71%)|41/49 (83.67%)|
|완전 비가시 시 Hallucinated Box|INSUFFICIENT_SUPPORT (0)|2/2 (100.00%)|
|의미 혼동|{"peg->box": 11, "peg->robot": 5}|{"box->peg": 11, "box->robot": 2}|
|IoU≥.3|14/56 (25.00%)|14/54 (25.93%)|
|IoU≥.7|0/56 (0.00%)|2/54 (3.70%)|
|IoU Median, 미검출0 포함|0.1390|0.1619|
|IoU p05~p95|0.0000~0.5883|0.0000~0.5821|
|Center Error Mean, 유효 출력만|27.6286px|29.7372px|

좌표 계약 166/192 (86.46%); Null Object Slot 0/192. JSON 분모96은 이미지, 좌표/Null 분모192는 Object Slot이며 서로 독립 Parent 수가 아니다.

| Parent-macro | 평균 | Macro95% CI | 지원 Parent |
|---|---:|---:|---:|
|Peg Recall|15.00%|6.67%~25.00%|15|
|Box Recall|13.89%|6.67%~22.22%|15|
|좌표 계약|86.46%|79.17%~92.71%|24|

| 보존 전체 NF4 대비 | 실패→성공 | 성공→실패 | Parent-macro Recall 차이 | Image-rate 차이의 Parent-cluster95% CI |
|---|---:|---:|---:|---:|
|peg|5|1|10.00%|0.00%~16.13%|
|box|6|2|6.67%|-3.33%~18.52%|

### Vision·Merger BF16 / 언어 INT8

판정: `VLM_GROUNDING_INSUFFICIENT`. 두 물체 동시 정확 2/54 (3.70%); 하나 이상의 잘못된 채택 Box 42/56 (75.00%).

| 지표 | Peg | Box |
|---|---:|---:|
|유효 Box 출력, 가시 물체 기준|44/56 (78.57%)|43/54 (79.63%)|
|Missing Detection, IoU≥.5 미달 포함|41/56 (73.21%)|48/54 (88.89%)|
|False Discovery, 유효 출력 중 실패|29/44 (65.91%)|38/44 (86.36%)|
|완전 비가시 시 Hallucinated Box|INSUFFICIENT_SUPPORT (0)|1/2 (50.00%)|
|의미 혼동|{"peg->box": 8}|{"box->peg": 7, "box->robot": 8}|
|IoU≥.3|22/56 (39.29%)|15/54 (27.78%)|
|IoU≥.7|3/56 (5.36%)|2/54 (3.70%)|
|IoU Median, 미검출0 포함|0.2072|0.0782|
|IoU p05~p95|0.0000~0.6961|0.0000~0.6117|
|Center Error Mean, 유효 출력만|16.6400px|31.2215px|

좌표 계약 162/192 (84.38%); Null Object Slot 0/192. JSON 분모96은 이미지, 좌표/Null 분모192는 Object Slot이며 서로 독립 Parent 수가 아니다.

| Parent-macro | 평균 | Macro95% CI | 지원 Parent |
|---|---:|---:|---:|
|Peg Recall|26.67%|16.67%~38.33%|15|
|Box Recall|11.67%|3.33%~21.67%|15|
|좌표 계약|84.38%|77.08%~91.67%|24|

| 보존 전체 NF4 대비 | 실패→성공 | 성공→실패 | Parent-macro Recall 차이 | Image-rate 차이의 Parent-cluster95% CI |
|---|---:|---:|---:|---:|
|peg|14|2|21.67%|8.33%~35.42%|
|box|5|3|4.44%|-6.67%~13.73%|

Parent를 독립 단위로10,000회 Bootstrap했다(seed42). 같은 Parent의 두 시점·두 Camera를 네 개 Parent로 세지 않았다. BF16을 기준으로 한 두 새 혼합 구성의 Paired 차이와 CI도 [전체 비교 JSON](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/precision_comparison_metrics.json)에 저장했다. 소표본의 퇴화 구간은 희귀 오류 안전성 보증이 아니다.

## 5. Camera·시점별 및 Phase A

| 구성 /구분 | Peg Recall | Box Recall |
|---|---:|---:|
|전체 BF16 /base_camera|3/28 (10.71%)|3/28 (10.71%)|
|전체 BF16 /hand_camera|9/28 (32.14%)|4/26 (15.38%)|
|전체 BF16 /F0|9/30 (30.00%)|5/29 (17.24%)|
|전체 BF16 /F1|3/26 (11.54%)|2/25 (8.00%)|
|Vision·Merger BF16 / 언어 NF4 /base_camera|5/28 (17.86%)|6/28 (21.43%)|
|Vision·Merger BF16 / 언어 NF4 /hand_camera|2/28 (7.14%)|2/26 (7.69%)|
|Vision·Merger BF16 / 언어 NF4 /F0|5/30 (16.67%)|4/29 (13.79%)|
|Vision·Merger BF16 / 언어 NF4 /F1|2/26 (7.69%)|4/25 (16.00%)|
|Vision·Merger BF16 / 언어 INT8 /base_camera|7/28 (25.00%)|2/28 (7.14%)|
|Vision·Merger BF16 / 언어 INT8 /hand_camera|8/28 (28.57%)|4/26 (15.38%)|
|Vision·Merger BF16 / 언어 INT8 /F0|10/30 (33.33%)|5/29 (17.24%)|
|Vision·Merger BF16 / 언어 INT8 /F1|5/26 (19.23%)|1/25 (4.00%)|

| 구성 /기술 사례 | Peg IoU | Box IoU |
|---|---:|---:|
|전체 BF16 /P|0.5063|0.3835|
|전체 BF16 /N|0.0082|0.5175|
|전체 BF16 /C1|0.5753|0.3100|
|Vision·Merger BF16 / 언어 NF4 /P|0.2405|0.2323|
|Vision·Merger BF16 / 언어 NF4 /N|0.0000|0.4838|
|Vision·Merger BF16 / 언어 NF4 /C1|0.7890|0.2914|
|Vision·Merger BF16 / 언어 INT8 /P|0.5854|0.3835|
|Vision·Merger BF16 / 언어 INT8 /N|0.0000|0.0000|
|Vision·Merger BF16 / 언어 INT8 /C1|0.5958|0.4465|

기술 확인3장은96장 재평가 성능에 합산하지 않았다. 각 Camera×시점의 실제 분모와 Parent별 값은 구성별 JSON/CSV에 보존했다.

## 6. Qwen Box로 실제 SAM2 초기화

기존 SAM2.1 Hiera Small Checkpoint와 단일 JPEG 프레임 초기화 경로를 유지했다. Qwen이 채택된 유효 Box를 출력한 Object만 Temporary State에 제공했다. GT/DINO Box 대체, 장기 Tracking, Production Memory Commit은 없다. 잘못된 Box와 Null/형식 실패도 선정16개에서 제외하지 않았다.

| 구성 | 실제 State /선정 Frame | Peg/Box 초기화 수 | Peg IoU≥.8 | Box IoU≥.8 |
|---|---:|---:|---:|---:|
|전체 BF16|12/16|12 /12|1/12 (8.33%)|2/12 (16.67%)|
|Vision·Merger BF16 / 언어 NF4|13/16|13 /13|1/12 (8.33%)|2/12 (16.67%)|
|Vision·Merger BF16 / 언어 INT8|14/16|14 /14|3/12 (25.00%)|3/12 (25.00%)|

| 구성 /Object | 실제 초기화 Mask Mean IoU | 미초기화0 포함 Mean IoU | Precision /Recall | Wrong-object |
|---|---:|---:|---:|---:|
|전체 BF16 /peg|0.2979|0.2483|0.6319 /0.2489|3/10 (30.00%)|
|전체 BF16 /box|0.2944|0.2453|0.3724 /0.2963|3/10 (30.00%)|
|Vision·Merger BF16 / 언어 NF4 /peg|0.2584|0.2369|0.5701 /0.2482|3/11 (27.27%)|
|Vision·Merger BF16 / 언어 NF4 /box|0.2578|0.2363|0.3371 /0.2883|3/11 (27.27%)|
|Vision·Merger BF16 / 언어 INT8 /peg|0.4327|0.3966|0.6606 /0.4082|2/11 (18.18%)|
|Vision·Merger BF16 / 언어 INT8 /box|0.3470|0.3181|0.3868 /0.3697|2/11 (18.18%)|

IoU≥.8 성공의 분모는 Object별 GT 지원12개다. 조건부 초기화 Mask 평균은 유효 Box가 있던 표본만 포함하므로 End-to-end 성공률과 구분한다. 원본 Actor Pixel 구성과 Robot/Background 오염은 각 `sam2_initialization_cases.csv`에 있다. GT 지원9 Parent의 Macro Bootstrap도 저장했다.

## 7. 실제 Latency·VRAM

| 구성 | Load ms | 로드 후 Allocated GiB | Qwen Peak GiB | Warm E2E Mean /Median /p95 ms | Warm n | 출력 Token 평균 |
|---|---:|---:|---:|---:|---:|---:|
|전체 BF16|3681.21|17.527|17.952|3842.80 /3649.42 /5685.37|98|82.73|
|Vision·Merger BF16 / 언어 NF4|3483.63|7.973|8.396|4150.60 /4020.72 /5584.94|98|71.03|
|Vision·Merger BF16 / 언어 INT8|20493.74|11.090|11.515|13667.53 /12140.31 /19408.25|98|87.80|

| 구성 | Cold 첫 요청 ms | CPU Peak RSS GiB | Qwen+SAM2 로드 GiB | SAM2 Warm Peak GiB /Forward Frame |
|---|---:|---:|---:|---:|
|전체 BF16|4399.85|18.420|17.844|18.137 /12|
|Vision·Merger BF16 / 언어 NF4|4694.60|18.310|8.290|8.583 /13|
|Vision·Merger BF16 / 언어 INT8|11947.02|18.362|11.406|11.699 /14|

| 구성 | Warm Preprocess ms | Vision CUDA ms | Prefill-Vision 차이 근사 ms | Decode Forward 합 ms |
|---|---:|---:|---:|---:|
|전체 BF16|5.77|16.34|479.39|3316.11|
|Vision·Merger BF16 / 언어 NF4|5.57|16.39|512.59|3593.18|
|Vision·Merger BF16 / 언어 INT8|6.22|16.60|432.72|13184.39|

매 이미지 새로운 Vision Encoding과 Prefill을 실행했다. Cold는 모델을 로드한 뒤 첫 요청이고 Warm은 후속98개다. 별도 Warmup/Timing 요청은 없다. E2E는 PNG 읽기·Processor·H2D·Generation·Text Decode를 포함한다. Vision CUDA Event, Vision을 포함한 첫 Forward, Prefill 근사, Decode Forward 합, Peak Reserved 및 출력 Token 분포도 저장했다. 출력 길이 차이도 E2E 비용에 반영되므로 평균 Latency만으로 순수 Kernel 속도를 단정하지 않는다. 초기 Object Discovery나 필요한 Re-grounding의 비용이며20Hz 제어 루프를 요구하는 평가가 아니다.

모델 로드는 사전 SHA 검증으로 File Cache가 Warm한 조건이며 다운로드/Processor 준비를 포함하지 않는다. CPU RSS는50ms 표본 Peak다. 같은 모델 객체를 구성 내99개 요청에 재사용했고 새 이미지마다 Cache는 새로 생성한다. 최적화 커널 추가 설치 없이 기존 Reference PyTorch DeltaNet/Conv 경로를 유지했다.

INT8에서는 BF16→FP16 Cast 경고가 Forward마다 반복 출력됐다. 중간에 Logging 수준을 바꾸지 않았으므로 위 E2E는 이 기본 경고의 Host Logging/I/O 비용도 포함한다. 이 수치만으로 경고를 억제한 최적화8-bit 서빙의 속도를 단정하지 않는다. 구성별 실제 `inference_log_bytes`를 비용 JSON에 저장했다.

## 8. 보존된 동일 Frame DINO 비교

| 구성 /Object | 공통 n | DINO IoU≥.5 | Qwen IoU≥.5 |
|---|---:|---:|---:|
|전체 BF16 /peg|12|7|3|
|전체 BF16 /box|12|7|3|
|Vision·Merger BF16 / 언어 NF4 /peg|12|7|2|
|Vision·Merger BF16 / 언어 NF4 /box|12|7|0|
|Vision·Merger BF16 / 언어 INT8 /peg|12|7|5|
|Vision·Merger BF16 / 언어 INT8 /box|12|7|3|

`MATCHED_DINO_BASELINE_INSUFFICIENT`: 같은12개t0 이미지/6개 Parent의 보존 결과만 사용했다. 새로운 DINO Forward는0이다. 이 소수 공통 표본으로 전체 분포의 상대 우열이나 DINO 제거를 확정하지 않는다.

## 9. 운영 기준·해석·실패 사례

| 구성 | 판정 | Peg/Box Recall≥90% | 잘못된 Peg≤5% | 좌표 계약≥99% | SAM2 Peg/Box≥80% |
|---|---|---|---|---|
|전체 BF16|VLM_GROUNDING_INSUFFICIENT|미달|미달|미달|미달|
|Vision·Merger BF16 / 언어 NF4|VLM_GROUNDING_INSUFFICIENT|미달|미달|미달|미달|
|Vision·Merger BF16 / 언어 INT8|VLM_GROUNDING_INSUFFICIENT|미달|미달|미달|미달|

운영 기준의 실제 분모는 위 표와 원시 Metric에 명시했다. 전체24 Parent GT, 희귀 오인식 확률, 비가시/Tiny 계층 및 실제 환경 일반화에는 `INSUFFICIENT_SUPPORT`가 남는다. 통계적 안전성이나 Closed-loop Success 보증이 아니다. Production DINO 삭제는 수행하지 않았다.

언어 NF4를 유지하고 Vision·Merger만 BF16으로 바꾼 직접 비교에서 Peg 성공은3/56→7/56, Box는4/54→8/54였다. 이 변화와 Parent-cluster CI를 함께 해석해야 하며, 단일 숫자로 전체 VLM의 잠재력을 결론 내리지 않는다.

전체 BF16에서도 두 물체 Recall≥90% 운영 기준을 함께 충족하지 못했다. 따라서 이 고정 조건에서는 가중치 양자화를 제거하는 것만으로 Grounding 문제를 해결하지 못했다. 이는 다른 Prompt·입력 Token 예산·Thinking 조건까지 검증했다는 뜻은 아니다.

실패 원인을 분리할 때는 형식 실패·역할 혼동·위치 오차·SAM2 Mask 실패를 구분한다. [구성별 실패 CSV](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/precision_failure_cases.csv)와 [같은 장면 비교 그림](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/precision_grounding_comparison.png)에 관측을 남겼다. 아래 표는 전체 오류 원인이 아니라 Actor 지배 기준으로 확인된 의미 혼동과 출력 형식 상태다.

| 구성 | Peg 의미 혼동 | Box 의미 혼동 | Null Slot | 중복 Key 응답 /Token 한도 도달 |
|---|---|---|---:|---:|
|전체 BF16|{"peg->box": 11, "peg->robot": 1}|{"box->peg": 11, "box->robot": 8}|0/192|11 /10|
|Vision·Merger BF16 / 언어 NF4|{"peg->box": 11, "peg->robot": 5}|{"box->peg": 11, "box->robot": 2}|0/192|13 /0|
|Vision·Merger BF16 / 언어 INT8|{"peg->box": 8}|{"box->peg": 7, "box->robot": 8}|0/192|9 /3|

정밀도 변경 뒤 성능이 달라진 것은 이 고정된 입력 조건의 관측이다. NF4 Vision 제외 비교는 Vision 정밀도의 영향을 가장 직접적으로 확인하며, 전체 BF16 및 INT8 구성은 언어 정밀도도 함께 다르다. 양자화 외에도 제한된 이미지 정보·역할 판단·Prompt·시각 Token 예산·Thinking 설정의 영향은 별도 검증하지 않았으므로 모든 실패를 특정 원인으로 확정하지 않는다. Prompt Sweep이나 새 모델 비교를 추가하지 않았다.

## 10. 예산·보존·종료

전체 Model Load 시도 3회, 완료 Qwen 요청 297개 /상한297개다. 구성별 Runtime 상태·오류는 [실행 Receipt](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/execution_progress.json)에 보존했다. 자동 Retry·재샘플링·추가 Warmup은0이다.

보호 파일 1648개의 SHA, 모델·Processor 15개 파일 SHA, 입력99장 SHA, 기존 환경의 Package 목록을 확인했다. 기존 모델·Dataset·GT·Processor·Stop Stub·이전 실험이 그대로다. 종료 시 GPU Compute Process0, 원래 Stage4A-2 Runner0을 확인했다. 모든 실제 CUDA 실행은58576054 RTX3090이며 macOS MPS/Simulator/Renderer는 실행하지 않았다.

공통 ExperimentLogger와 W&B disabled 로컬 기록을 재사용했다. 공통 Logger/Simulator API는 변경하지 않았으며 CUDA 가중치 타입·요청별 입력 Contract·Parser 계약·원시 응답/CSV Readback·Model/Source/Data Hash 및 예산을 검증했다. CSV의 Pixel/GT Box에서 가시 물체110개×3구성의 IoU·성공 수·평균을 별도 산술로 재계산해 일치함을 확인했다. 변경은 독립 실험 Source/Output와 이 보고서에 한정했다. Training·새 Simulator Reset/Render/Rollout·Graph 재캐시·새 모델 다운로드·기존 프로세스 종료·외부 게시·Git Commit·Production 변경은0이다.

[실험 Protocol](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/protocol.json), [정밀도별 모듈 계획](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/expected_precision.json), [비교 CSV](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/precision_comparison.csv), [전체 지표 JSON](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/precision_comparison_metrics.json), [검증](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/verification.json), [Artifact Index](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/artifact_index.json).

| 구성 | Identity /Raw /Metric /SAM2 /비용 |
|---|---|
|전체 BF16|[Identity](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/bf16/model_identity.json) /[Raw](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/bf16/full_image_grounding_raw_responses.json) /[Metric](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/bf16/grounding_evaluation_metrics.json) /[SAM2](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/bf16/sam2_initialization_metrics.json) /[비용](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/bf16/grounding_latency_vram.json)|
|Vision·Merger BF16 / 언어 NF4|[Identity](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_nf4/model_identity.json) /[Raw](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_nf4/full_image_grounding_raw_responses.json) /[Metric](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_nf4/grounding_evaluation_metrics.json) /[SAM2](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_nf4/sam2_initialization_metrics.json) /[비용](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_nf4/grounding_latency_vram.json)|
|Vision·Merger BF16 / 언어 INT8|[Identity](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_int8/model_identity.json) /[Raw](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_int8/full_image_grounding_raw_responses.json) /[Metric](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_int8/grounding_evaluation_metrics.json) /[SAM2](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_int8/sam2_initialization_metrics.json) /[비용](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/vision_bf16_language_int8/grounding_latency_vram.json)|

## 11. 동결 Prompt 원문

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

![고정된 네 조건의 세 구성 비교](../../outputs/task10i/stage4a2/qwen35_precision_comparison_20261102_v1/precision_grounding_comparison.png)
