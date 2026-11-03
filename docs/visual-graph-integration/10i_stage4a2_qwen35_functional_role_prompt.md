# Task 10-I Stage 4A-2 — Qwen3.5-9B Functional-role Prompt Verification

2026-11-02. **최종 판정: `LOCAL_PROMPT_IMPROVEMENT`. `PAUSED_SCOPE_REVIEW` 유지.**

지정한 기능적 역할 Prompt 하나로 **P=`PEG`, N=`BOX`, C1=`PEG`**를 출력했다.
원시 응답은 각각 `"PEG\n"`, `"BOX\n"`, `"PEG\n"`이다. 허용된 양끝 공백 처리 후 세 응답 모두
엄격한 단일 Label 계약을 통과하고 EXACT_GT Actor Label과 일치했다. 이전의 실제 Peg 두 건에
대한 OTHER 판단을 복구했고, 기존 BOX 판단도 유지했다. 이 세 후보에서의 국소 개선이며
Production 적용 승인·일반화 성능·안전한 Consumer 통합을 입증하지 않는다.

## 실행과 동결 Identity

기존 Vast.ai **58576054 / RTX3090 24GB**에서 실행했다. 시작 GPU 사용1MiB,
다른 GPU 프로세스0, 디스크 약144GiB 여유를 확인했다. 다운로드·설치·환경 변경 없이 기존
독립 환경 `qwen35_verifier_20261102_v1/env`와 공식 모델 Cache를 재사용했다.
모델 로드 **1회**, 요청 **3회**, Warmup·재생성·추가 Sampling **0회**다.

모델은 Multimodal **Qwen/Qwen3.5-9B**, Revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`다. 기존 `Qwen3_5ForConditionalGeneration`과
`AutoProcessor`(`Qwen3VLProcessor`), SDPA, `device_map={"":0}`를 유지했다.
Torch2.14.1+cu130 / Transformers5.19.0 / BitsAndBytes0.50.2,
BF16 compute·NF4·Double Quantization·uint8 storage·CPU/Disk offload 비활성이다.
기존 전체 모델/Processor 파일 **15개**의 SHA256을 시작/종료에 모두 대조했다.
실제 `Linear4bit`358개와 Parameter/Buffer 전부 GPU임을 다시 확인했다.
기존 Packed Weight3,684,841,472 bytes, Quantization State116,974,552 bytes,
비양자화 BF16 Parameter4,080,261,600 bytes이며 모든 Parameter가 INT4인 것은 아니다.

모델 Identity와 파일별 SHA, 기존 Quantization Config 동등성, 이미지 SHA,
새 Generation 설정은 [prompt_identity.json](../../outputs/task10i/stage4a2/qwen35_functional_role_20261102_v1/prompt_identity.json)에 있다.
Generation은 `do_sample=False`, `enable_thinking=False`, `max_new_tokens=16`,
`use_cache=True`, Batch1이다. 모델 객체 하나로 세 요청을 순차 실행했다.

## 동일 입력 및 사전 동결 Prompt

기존 `vlm_input_manifest.json`을 기준으로 P call133/rank1/query3,
N call153/rank0/query0, C1 call333/rank0/query0을 그대로 사용했다.
기존 PNG **9개 SHA256이 전부 일치**했다. 새 이미지·후보·Crop은 만들지 않았다.
Original128×128, 청록색 RGB(0,180,220)1px Marker, 각 변10% Crop과 이미지 순서,
Chat Template, `min_pixels=max_pixels=65536`을 유지했다.
실제 세 Image Grid도 이전 결과와 동일했다: Scene/Marked `[1,16,16]`,
Crop P`[1,12,22]` / N`[1,16,18]` / C1`[1,14,20]`.

기존의 중립적인 이미지 순서 안내문과 이미지3장 뒤에 아래 새 Prompt를 넣었다.
유일한 변경 변수는 텍스트 Prompt와 `max_new_tokens`64→16이다.
Prompt는 로드 전 `prompt_identity.json`에 원문을 저장하고 UTF-8 SHA256을 동결했다.
Hash 범위는 아래 원문의 끝 줄바꿈을 제외한 문자열이다.

**Prompt SHA256: `691ea9e22ec710dad77084db6aa812455d525190f7d6f589821bfaf082889a06`**

```text
You are a visual object verifier for robotic manipulation.

The scene shows a robot performing a peg-insertion task. Your job is to identify the physical object inside a marked candidate region.

Important: Classify objects by their functional role in the manipulation task, not merely by their everyday appearance or shape.

Object categories:

PEG: The separate physical component that the robot is intended to grasp, move, and insert into a matching opening. A peg does not have to be cylindrical. Its shape, color, and material can vary. An object that resembles a block or another everyday item can still be a PEG if it serves this role.

BOX: The receiving object or fixture containing the opening into which the PEG is inserted.

ROBOT: Any part of the robotic manipulator, including its arm, gripper, and fingers.

OTHER: A clearly identifiable object that does not serve any of these roles.

UNCERTAIN: The visual evidence is insufficient to determine the object's role reliably.

You are given three images:
1. The complete scene.
2. The same scene with a colored rectangle marking the candidate region.
3. A crop of that candidate region.

Examine the marked region carefully. Use the complete scene to understand the object's role.

The candidate region is only a proposal and may contain an incorrectly identified object. Do not assume that the proposal is correct.

Distinguish the insertable component from the receiving fixture and the robot. Do not classify an insertable component as OTHER simply because its appearance resembles a familiar object.

If the marked region contains multiple objects and its intended object cannot be identified reliably, choose UNCERTAIN.

Output exactly one of these labels:

PEG
BOX
ROBOT
OTHER
UNCERTAIN

Do not output JSON, Markdown, explanations, punctuation, or additional text.
```

색상·길이·사례별 정답 규칙·Few-shot은 추가하지 않았고 결과 후 수정하지 않았다.
DINO Semantic Label·Score, GT Mask/Actor ID/Pose, SAM 평가 결과는 메시지에 넣지 않았다.
보존용 GT 파일의 불투명 Byte Hash 검사와 의미 평가를 구분했다. 추론 전에는 GT 내용을
해석하지 않았으며, 세 원시 응답을 모두 저장하고 SHA를 고정한 뒤 처음으로 EXACT_GT를
파싱했다. Barrier 전후 응답 SHA가 동일하다.

## 실제 결과와 이전 Prompt 비교

| 사례 | 사후 GT (Peg/Box Pixel) | 기존 내용 / 형식 | 새 원시 응답 | 새 판정 |
|---|---|---|---|---|
| P 0355/t0 | PEG (530/9) | OTHER / INVALID_OUTPUT | `PEG\n` | CORRECT, 형식 통과 |
| N 0355/t30 | BOX (33/899) | BOX / INVALID_OUTPUT | `BOX\n` | CORRECT, 형식 통과 |
| C1 0645/t0 | PEG (849/0) | OTHER / INVALID_OUTPUT | `PEG\n` | CORRECT, 형식 통과 |

P와 C1의 Peg 인식은 복구됐고 N에 새로운 Peg 오판은 생기지 않았다.
UNCERTAIN은 나오지 않았다. UNCERTAIN이었다면 정체성 확인을 보류한 결과이며 Peg 인식
복구로 계산하지 않았을 것이다. 이전 형식 통과0개에서 이번3개로 개선됐다.
이전 OTHER/BOX/OTHER는 Fence 내부 내용의 사후 진단이고 이전 Accepted 출력은 모두
INVALID_OUTPUT이었다는 차이를 유지한다. 이번에는 JSON/Fence/설명이 전혀 없다.

[비교 CSV](../../outputs/task10i/stage4a2/qwen35_functional_role_20261102_v1/prompt_comparison.csv),
[수정하지 않은 원시 응답·Token ID](../../outputs/task10i/stage4a2/qwen35_functional_role_20261102_v1/vlm_raw_responses.json)에 결과를 보존했다.

## Latency와 VRAM

File→CPU→GPU Load는 **3516.86ms**다. 직전 모델 SHA 읽기로 OS File Cache가 Warm한
로드이며 다운로드·Processor 준비 시간을 포함하지 않는다.
로드 후 CUDA Allocation **7.349GiB**, Reservation **7.406GiB**,
Device Free **15.854GiB**다. 요청 중 Peak Allocation **7.592GiB**,
Peak Reservation **7.648GiB**였다. CPU/Disk Offload와 CUDA OOM은 없었다.

| 요청 | 입력 Token | 출력 Token | Preprocess | Vision CUDA | 첫 Forward: Vision+Prefill | Decode Forward CUDA 합 | E2E |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 609 | 4 | 55.82ms | 337.40ms | 718.53ms | 159.88ms | **1096.85ms** |
| N | 615 | 4 | 8.81ms | 34.07ms | 263.70ms | 151.79ms | **433.31ms** |
| C1 | 613 | 4 | 6.36ms | 34.60ms | 264.05ms | 152.12ms | **431.71ms** |

출력 Token 수는 종료 Token과 줄바꿈을 포함한 실제 생성 ID 개수다.
이전45/44/47 Token에서 각각4 Token으로 감소했다. 이전 Warm E2E
2493.71/2561.27/2683.83ms에 비해 이번 E2E는1096.85/433.31/431.71ms였다.
이전 Decode Forward CUDA 합2242.85/2317.50/2436.81ms는
이번159.88/151.79/152.12ms로 감소했다.

**P는 모델 로드 직후 첫 요청**이어서 Cold Kernel·초기 Processor 비용을 포함한다.
N/C1은 동일 상주 모델의 후속 요청이다. 별도 Warmup/반복을 수행하지 않았으므로 P를
Warm 수치로 부르지 않으며 세 요청으로 p50/p95 등 장기 지연 통계는 계산하지 않는다.
기존 대비 입력 Token도328/334/332→609/615/613으로 증가했다.
Prompt 내용·길이와 출력 형식·최대 길이가 함께 달라졌으므로 개선을 특정 문장 하나나
Token 제한만의 인과 효과로 분리할 수 없다. Decode 감소는 관측됐으나 Prompt 자체의
비용도 바뀌었다. 길어진 Prefill과 줄어든 Decode를 함께 보고한다.

E2E는 PNG 열기·Processor·H2D·Generate·Text Decode를 포함하고 산출물 저장/GC는 제외했다.
요청 경계 CUDA Synchronization과 실제 Forward Hook의 CUDA Event를 사용했다.
Vision은 첫 Forward에 포함되므로 두 값을 더하지 않는다. Decode 수치는 후속 Forward의
CUDA Event 경과 합이며 Generation Loop 전체 Decode Wall Time과 다르다.
새 이미지마다 실제 Vision 호출1회와 새 Prefill이 수행됐다.
기존과 동일한 Hybrid Layer Reference PyTorch Fallback을 사용했으며 Kernel을 설치/패치하지 않았다.

[vlm_latency_vram.json](../../outputs/task10i/stage4a2/qwen35_functional_role_20261102_v1/vlm_latency_vram.json)에 요청별 측정값을 기록했다.
기존 DINO/SAM2/DINOv2 로드·Forward와 동시 상주 검사는 이번에 수행하지 않았다.

## 보존·해석·종료

기존 Source/Raw/GT/Checkpoint/Stop Stub 등 **1342개 보호 파일**의 시작/종료 SHA가
일치했다. 모델/Processor 및 입력9개도 종료 시 그대로였고 기존 연구 Python Package 목록은
동일했다. Stage4A-2 Runner·Internal/Final 평가·Graph 재캐시·학습·Simulator는 실행하지 않았다.
기존 Consumer/Pipeline Source·Checkpoint·Dataset 변경, 외부 게시·Git Commit은 없다.
실험 프로세스 종료 후 GPU Compute Process가 없는 것을 확인했다.

공통 `ExperimentLogger`를 W&B disabled로 재사용했다. 실행 UUID는 `75877091e17042158872cd302e0862a4`다.
실행 Source/로그와 로컬 Logger 기록만 `.execution/`에 보존했고 추가 연구 Gallery 등은 없다.
새 Production 코드 변경이 없어 Pipeline/Simulator 테스트 대신 실제3요청, 엄격한 출력 판정,
GT Barrier·CSV/JSON Readback·모델/이미지/원본 Hash를 검증했다.
[verification.json](../../outputs/task10i/stage4a2/qwen35_functional_role_20261102_v1/verification.json)에 보존 및 완료 검사를 기록했다.

기능적 역할 설명과 단일 Label 계약은 이 세 후보에서 개선을 보였다. 하지만 장면의 실제
조작 이력을 관찰한 것이 아니라 정지 이미지와 Task 설명으로 역할을 추정한 결과다.
추가 후보의 Recall/오승인·시각적 Ambiguity·분포 변화·장기 상주 안정성·기존 모델 공존·
Closed-loop 비용은 미검증이다. **Production 통합이나 다른 Stage 재개는 수행하지 않았다.**
추가 통합 실험의 타당성을 검토할 국소 근거로만 남기고 여기서 종료한다.

원격 Output Root: `/workspace/task10i_stage4a2_qwen35_functional_role_20261102_v1/`.
