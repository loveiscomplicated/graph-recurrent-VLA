# Task 10-I Stage 4A-2 — Qwen3.5 Grounding 구현·평가 감사

2026-11-02. 기존 Full-image Grounding 실험의 추가 분석이다.
**새 Weight Load·VLM Generation·DINO/SAM2 Forward·Simulator 실행은 모두0회**다.
기존 Frozen 결과와 Production Source는 수정하지 않았다.

현재 증거로 **“Qwen3.5-9B는 원래 DINO보다 Grounding 성능이 낮다”라고 결론 내릴 수 없다.**
확인된 것은 특정 NF4·Non-thinking·Greedy·기능적 역할 Prompt·출력 계약 조합의 성능이다.
좌표를 뒤집거나 이미지를 잘못 전달한 큰 구현 오류는 발견하지 못했지만,
**형식 실패가 위치 능력 비교를 크게 가렸고, Vision Encoder까지 양자화한 조건의 품질은 검증되지 않았다.**

## 1. 공식 자료가 말하는 Qwen3.5-9B의 Grounding 능력

Qwen 공식9B 카드의 Vision Language 표는 **RefCOCO(avg)=89.7**을 보고한다.
이는 물체를 지칭하는 표현에 대한 Grounding 능력이 있는 모델이라는 근거다.
동일 표의 Qwen3-VL-30B-A3B는89.3이다. 다만 이 숫자를 우리의128² Simulator 이미지,
기능적 역할 추론, NF4, Non-thinking 조건의 예상 Recall로 환산할 수는 없다.
공식 카드에서 이번 실행과 같은 세부 설정의 결과를 확인한 것은 아니다.
[공식 Qwen3.5-9B 카드](https://huggingface.co/Qwen/Qwen3.5-9B#vision-language).

Grounding DINO의 공식 Repository는 별도의 Open-set Object Detection 모델과 COCO AP를
보고한다. RefCOCO 정확도와 COCO AP는 Dataset·지표·Task가 달라 숫자만 직접 비교할 수 없다.
공식 자료에서 **지금의9B NF4 조건과 현재 DINO Tiny를 동일 로봇 이미지·동일 표현으로
비교한 결과는 확인하지 못했다.** 공개 점수로 어느 쪽이 본래 우월하다고 확정하지 않는다.
[Grounding DINO 공식 구현·평가](https://github.com/IDEA-Research/GroundingDINO#-checkpoints).

## 2. 가장 크게 바로잡아야 할 해석: JSON 실패와 위치 능력

독립96개 응답 중94개가 Markdown JSON Fence를 포함했다. 기존 실험은 사용자가 고정한
엄격한 계약에 따라 이를 거부했으므로 **Primary 결과 자체는 당시 프로토콜에 맞다.**
그러나 이 결과를 모델의 시각적 위치 능력으로 곧바로 읽으면 능력을 과소평가한다.

Qwen의 공식2D Grounding Cookbook은 출력의 Markdown Fence를 제거해 JSON을 읽는
Helper를 포함한다. 모델이 Fence를 붙이는 것은 그 사용 예에서도 예상한 형식이다.
Native 예시는 `bbox_2d`·`label` 구조도 사용한다. 따라서 “순수 JSON 외 모든 출력 거부”와
“모델이 생성한 좌표의 정확도”는 분리해야 한다. 이는 기존 결과를 사후 통과시키라는 뜻이 아니다.
[공식2D Grounding Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb).

전체 독립 GT 지원 범위에서 형식만 분리한 좌표 진단은 Peg25/56, Box9/54다.
위치 오류도 여전히 크므로 Parser 변경만으로 DINO 대체가 완성된다고 볼 수는 없다.

## 3. 같은12장에서 다시 비교한 실제 위치 결과

이전 실험에 보존된 공통12개 이미지,6 Parent, t0, Main/Wrist 각6개만 사용했다.
새 DINO 실행이나 새로운 Frame 선택은 없다. Qwen의 기존 응답에서 Fence만 제거한 좌표와
기존 DINO Joint 선택 Box를 동일한 GT Bounding Box에 대조했다. 성공 기준은 IoU≥.5다.

| 공통 Frame 범위 | DINO Peg | Qwen Peg 좌표 진단 | DINO Box | Qwen Box 좌표 진단 |
|---|---:|---:|---:|---:|
| 전체12장 |7/12|**6/12**|7/12|**4/12**|
| Main6장 |5/6|4/6|6/6|3/6|
| Wrist6장 |2/6|2/6|1/6|1/6|
| 전체 Mean IoU |.5221|.4636|.5356|.3722|
| Wrist Mean IoU |.3045|.4185|.1377|.2642|

앞선 Strict 비교의 **DINO7/12 대 Qwen0/12**는 같은 Frame의 **좌표 능력 비교와 다르다.**
형식 영향을 분리하면 특히 Peg의 차이가 훨씬 작다. Wrist에서는 성공 건수가 같고,
평균 IoU는 이 작은 표본에서 Qwen 쪽이 높다. 반면 Main Box는 DINO가 더 정확했다.
이 관측을 전체 장면의 우열로 확장하지 않는다.

Peg는 DINO만 성공4건, Qwen만 성공3건이었다. Box는 각각4건과1건이다.
물리 Parent6개를 Cluster로10,000회 Bootstrap한 DINO−Qwen Recall 차이의95% 구간은
Peg **−33.3~58.3%p**, Box **0~58.3%p**다. 지원이 적고 후기 Frame도 없다.
`MATCHED_DINO_BASELINE_INSUFFICIENT` 판단을 유지한다.

추가로 두 경로의 텍스트 조건도 다르다. 기존 DINO는 `red and white block`과
`cardboard box`라는 시각적 표현을 받고, Qwen은 색/모양 고정 규칙 없이 Peg/Box의
기능적 역할을 해석한다. 이는 실제 응용 요구에는 맞지만 **모델 구조만의 차이를 분리한 비교는 아니다.**

[새 분석 JSON: 공통 Frame·승패·CI](../../outputs/task10i/stage4a2/qwen35_grounding_audit_20261102_v1/saved_prediction_audit.json).

## 4. 이미지·Processor·좌표 구현 감사

58576054의 기존 환경에서 GPU를 숨기고 Processor만 CPU로 재실행했다.
Weight는 로드하지 않았다. 아래는 실행으로 확인한 내용이다.

| 점검 | 결과 |
|---|---|
| 원본 HDF5 RGB와 입력 PNG |99/99 Pixel 배열 일치 |
| PNG SHA와 Frozen Manifest |99/99 일치 |
| RGB→BGR 혼동 |Qwen 입력 경로에서 발견하지 못함; PIL RGB 사용 |
| 이미지 수·순서 |전체 이미지1장과 지정 Prompt; Marker/Crop 없음 |
| P/N/C1 및 실제 SAM2 대상0134/F1의 재생성 입력 |기존 Token 수584와 Grid 기록 일치 |
| Vision 입력 |Grid `[1,16,16]`, Patch256개, Merge 후 Image Placeholder64개 |
| Image 정규화 |설정 mean/std=.5/.5; 실제 Pixel Tensor 범위 정상 |
| Chat Template |`enable_thinking=False`의 Assistant Prefix 확인 |
| 좌표 변환 |공식0–1000 xyxy를 원본128×128로 변환하는 구현과 일치 |
| 평가 IoU 독립 재계산 |기존 GT 지원116개 Object Row 전부 일치; Phase A 포함 |
| GT Actor Mapping |참조 Receipt17개의 Peg/Box ID·이름·SHA 재확인 |
| 출력128 Token 제한에 의한 잘림 |기존 모든 출력55–62 Token; 제한 도달 없음 |

Qwen 담당자의 공식 Q&A도 Qwen3.5가 Qwen3-VL과 같은0–1000 정규화 좌표를 사용한다고
설명한다. Processor 내부256² 좌표를 출력 기준으로 삼아야 한다는 근거는 없다.
[공식 Repository의 좌표 Q&A](https://github.com/QwenLM/Qwen3.8/discussions/56?sort=top),
[Preprocessing Q&A](https://github.com/QwenLM/Qwen3.8/discussions/60).

단순 좌표 계약 착오를 점검하기 위해 기존 출력만 재해석한 결과도 저장했다.
분모1000에서는 Peg25/56·Box9/54,1024에서는22/56·7/54,
256 또는128 Pixel 좌표 가정에서는 모두0, xy를 교환하면3/56·1/54였다.
이것은 **사후 오류 진단이며 좌표 보정이나 새로운 채택 결과가 아니다.**
공식 계약과 독립 재계산을 함께 보면 단순한 Scale/축 교환 실수가 주원인이라는 증거는 없다.

### 256²가 잘못된 설정이었는가

원본 모델의 Pinned `preprocessor_config.json`은 shortest area65536,
longest area16777216이다. 이전 실험은 둘 다65536으로 고정했다.
그런데 **128² 이미지는 기본 Processor도256²로 올린다.** P/N/C1 및0134/F1의4개 CPU 재현에서
고정 설정과 기본 설정의 `pixel_values`·Token·Grid 등 반환 Tensor가 모두 **완전히 같았다.**
따라서 현재 이미지에서 “max_pixels 고정 때문에 기본보다 축소됐다”는 설명은 성립하지 않는다.
[동일 Revision의 Processor Config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/preprocessor_config.json).

다만 DINO Processor는 같은128² 원본을 **800×800**으로 바꾼다. Qwen은256²/64 Vision Token이다.
같은 Raw RGB를 사용해도 내부 표현 예산은 다르다. 향후512² 등으로 올리는 검사는 의미가 있지만,
그 자체가 정보를 새로 복원하거나 정확도를 개선한다는 것은 아직 검증되지 않았다.

[CPU 재현·원본 보존 JSON](../../outputs/task10i/stage4a2/qwen35_grounding_audit_20261102_v1/runtime_cpu_audit.json),
[실제 Full-image Chat Template](../../outputs/task10i/stage4a2/qwen35_grounding_audit_20261102_v1/actual_full_image_rendered_template.txt).

## 5. 가장 우선해서 통제할 설정: Vision Encoder의 NF4

기존 실제 양자화 목록에서 `model.visual.*`의 Linear4bit가 **110개**다.
Attention/MLP뿐 아니라 Vision-to-Language Merger의 두 Linear도 NF4다.
**BF16 Compute는 Vision Weight가 BF16으로 보존됐다는 뜻이 아니다.**
이는 사용자 지정 NF4와 기존 모델 로드 정책을 유지한 실행이며, 규약 위반이나 Runtime 오류로
판정하지 않는다. 그러나 공식 모델 자체의 Grounding 능력과 양자화된 Vision 경로의 품질을
동일시하면 안 된다. 이 설정이 성능 저하의 원인인지는 아직 통제 실험으로 확인하지 않았다.

비교할 만한 구현 예로 RedHatAI의 Qwen3.5-9B W4A16은 Vision Encoder와 Linear Attention을
BF16으로 남긴다. vLLM LLM Compressor의 Qwen3.5 NVFP4 예도 Vision을 제외한다.
양자화 방식은 NF4와 다르므로 이 문서를 근거로 “NF4 Vision은 항상 잘못됐다”라고 주장하지 않는다.
다만 **Vision 정밀도를 보존한 대조군이 필요하다**는 구체적인 점검 근거다.
[RedHatAI9B 양자화 구현](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w4a16#model-optimizations),
[LLM Compressor 구현 예](https://docs.vllm.ai/projects/llm-compressor/en/0.12.0.1/key-models/qwen3.5/nvfp4-vl-example/).

현재 Vision Linear의 Packed Weight는225,589,248 bytes이며 같은110개를 BF16으로 표현하면
902,356,992 bytes다. 차이는 약**0.630GiB**다. Quantization State 제거/Allocator/Activation은
별도이므로 이것은 **Weight 산술 추정이지 실제 추가 VRAM 측정값이나 상주 보장이 아니다.**
새 Checkpoint 다운로드 없이 기존 원본 Weight의 양자화 제외 범위를 바꾸는 대조가 가능하다.
이번 조사에서는 로드 정책을 바꾸거나 모델을 다시 실행하지 않았다.

## 6. 나머지 미통제 조건과 기록상의 결함

공식 모델은 Thinking/Non-thinking 양쪽을 지원하고 모드별 Sampling 권장값을 제시한다.
이번은 Non-thinking/Greedy로 고정했으며, 공식89.7을 동일 조건으로 재현한 적은 없다.
Thinking이나 Sampling이 이 Task를 개선한다고 주장하지 않는다. 모델 상한과 현재 고정 설정의
성능을 구분해야 한다는 의미다.
[Qwen 공식 Generation 안내](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices).

기존 기능적 역할 Prompt는 후보 영역 분류의 국소 성공을 기반으로 확장됐다. Global Discovery에서는
물체 식별과 위치 생성, 두 Target의 역할 배정, 엄격한 JSON까지 동시에 요구한다.
공식의 짧은 Referring Expression Grounding과 같은 Task가 아니며, 정지영상에서 보이지 않는
로봇의 의도를 실제로 관측한 것도 아니다. 어느 하위 과제가 실패했는지 아직 완전히 분리하지 못했다.

**기록 결함도1개 발견했다.** 새 `model_identity.json`은 이전 Identity를 복사하면서
`processor_contract` 안에 과거3-image Verifier의 Rendered Prompt/Input Shape를 남겼다.
실제 이번 `infer.py`는 Full-image1장을 전달하고 Raw Timing은584 Token·Image Grid1개를
기록한다. 이번 CPU 재현도 이를 확인했다. 따라서 실제로 옛 Crop이 들어간 오류는 아니지만,
Identity의 해당 필드를 이번 실험의 실제 Input Snapshot으로 읽으면 안 된다.
기존 Frozen Identity는 수정하지 않고 이 감사에서 명시한다. 후속 Runner는 과거 Contract를
출처 필드로 분리하고 실제 이번 Template/Shape를 새로 저장해야 한다.

## 7. 다음 진단의 우선순위

현재 단계의 판단은 **VLM Grounding 방향 폐기 보류, 현재 설정의 Production 대체도 보류**다.
기존 실험의 `VLM_GROUNDING_INSUFFICIENT`는 해당 고정 조건에 대해 유지한다.
이를 Qwen 전체 또는 VLM 방식 일반의 한계로 확대하지 않는다.

추가 실험을 진행한다면 우선 순서는 다음과 같다. 이번 조사에서는 실행하지 않았다.

1. **같은 모델·같은 Prompt·같은 이미지에서 Vision/Merger만 BF16으로 보존한 대조.**
   Decoder NF4와 나머지 조건은 유지해 가장 구체적인 정밀도 가설부터 확인한다.
2. **짧은 지칭 표현과 Native `bbox_2d` 형식의 진단 대조.** 위치 생성 자체와 기능적 역할
   이해의 결합 난도를 분리한다. 원 실험의 Prompt를 바꿔 성공 처리하지 않고 별도 진단으로 둔다.
3. **원본 RGB 그대로 Vision Token 예산만 올리는 대조.**64개와 더 높은 예산을 비교하며
   Resize가 원본 세부 정보를 새로 만든다고 주장하지 않는다.
4. 작은 고정 진단 집합에서 구현·설정이 정상임을 확인한 뒤에만 별도 독립 평가를 설계한다.
   기존 결과를 보고 선택한 진단 사례의 성공을 새 일반화 성능으로 세지 않는다.

형식은 Raw Strict Compliance와 내용 기반 Grounding을 항상 함께 보고하고, SAM2 입력에
허용할 Parser 계약은 후속 실험 전에 명시적으로 고정하는 것이 적절하다.

## 8. 검증·산출물·범위

기존14개 필수 산출물과135개 Supporting Artifact의 Hash를 다시 대조했다.
기존 IoU116개를 별도 구현으로 재계산했고 원래 수치와 일치했다.
CPU Processor4개 재현,99개 RGB 동일성, GT Receipt17개 및 기존 결과의 불변성을 확인했다.
기존 Primary/Diagnostic 분모를 변경하거나 Accepted Prediction을 추가하지 않았다.

공통 `ExperimentLogger`를 W&B disabled로 재사용한 분석 UUID는
`495187c0b65649518ccd99c174339a7b`다. 새 코드·로그·지표는
`outputs/task10i/stage4a2/qwen35_grounding_audit_20261102_v1/`에 있다.
원격 CPU 감사 Root는 `/workspace/task10i_stage4a2_qwen35_grounding_audit_20261102_v1/`다.

외부에서는 공식 문서와 작은 Config/README·Cookbook만 조회했다. 새 모델 Weight 다운로드,
GPU 모델 Load/Forward, Simulator, Training, Production 변경, 기존 Runner 재개,
Git Commit·외부 게시를 하지 않았다. 기존 `PAUSED_SCOPE_REVIEW`를 유지했다.
공통 Pipeline 변경이 없어 Simulator/Training 회귀 테스트는 실행하지 않았다.

**현재 답:** Qwen3.5-9B는 공개 자료상 Grounding 능력이 있는 모델이다. 우리 설정에서의 실패는
실제지만, 형식 계약·Vision NF4·입력 표현 예산·역할 추론 난도가 분리되지 않았다.
“원래 DINO보다 못해서 실패했다”보다 **“현재 구현 조건에서 성능을 제한한 원인을 아직
분리하지 못했다”**가 증거에 맞는 해석이다.
